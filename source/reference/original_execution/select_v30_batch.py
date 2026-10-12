from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


FEATURES = ["P_Cap", "J_eCO2R", "T_eCO2R", "N_eCO2R", "P_WE", "N_WE", "P_MeOH", "T_utility"]
INTEGER_FEATURES = {"N_eCO2R", "N_WE"}
TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]
PREDICTED = [f"{target}_pred" for target in TARGETS]
PREQUERY_STD = [f"{target}_std_prequery" for target in TARGETS]
CANDIDATE_SCORE_NUMERIC = [
    *PREDICTED,
    *PREQUERY_STD,
    "local_error_knn_score_fixed_range",
    "input_ad_distance_to_development2400",
]
GPROMS_INPUT_COLUMNS = [
    "Current_density_eCO2R_Acm2",
    "WE_Energy_input_rate_KW",
    "Number_of_cells_stacked_in_series_WE",
    "Number_of_stacks_eCO2R",
    "Outlet_pressure",
    "Outlet_pressure_cap",
    "T_eCO2R_gproms_input",
    "Utility_constant_temperature",
]
RAW_OUTPUT_COLUMNS = [
    "converged",
    "failed_reason",
    "MeOH_conversion",
    "MeOH_yield_kt_y",
    "allpower_kw",
    "Total_ele_power_kw",
    "S",
    "CAP_conversion",
    "eCO2R_conversion",
    "Coolduty_kw",
    "Heatduty_kw",
]
PRIMITIVE_RAW_COLUMNS = [
    "Component_molar_flowrate_methanol",
    "Component_molar_flowrate_carbon_dioxide_CAP",
    "Component_molar_flowrate_carbon_monoxide_CE",
    "Component_molar_flowrate_hydrogen_WE",
    "Component_molar_flowrate_hydrogen_CE",
    "Component_molar_flowrate_carbon_dioxide_CE",
]
ECHO_COLUMNS = [
    "echo_Current_density_specification_1",
    "echo_Energy_input_rate",
    "echo_Number_of_cells_stacked_in_series",
    "echo_Number_of_stacks",
    "echo_Outlet_pressure",
    "echo_Outlet_pressure_cap",
    "echo_Temperature",
    "echo_Utility_constant_temperature",
]
PROVENANCE_COLUMNS = [
    "gproms_model_version",
    "solver_settings_id",
    "execution_batch_id",
    "gproms_scenario_id",
    "gproms_attempt_id",
    "solver_status_code",
    "runtime_seconds",
    "execution_timestamp",
    "raw_output_file",
    "raw_output_file_sha256",
    "operator_notes",
]
DERIVED_RETURN_COLUMNS = [
    "MeOH_production_kg_h_rebuilt",
    "carbon_utilization_rebuilt",
    "specific_energy_kwh_per_kg_meoh_rebuilt",
    "A_eCO2R_m2_rebuilt",
    "A_WE_m2_rebuilt",
    "capex_proxy_rebuilt",
    "formula_qc_status",
]


def canonical_feature_string(row: pd.Series) -> str:
    parts: list[str] = []
    for name in FEATURES:
        if name in INTEGER_FEATURES:
            parts.append(str(int(round(float(row[name])))))
        else:
            parts.append(f"{float(row[name]):.10f}")
    return "|".join(parts)


def feature_hash(row: pd.Series) -> str:
    return hashlib.sha256(canonical_feature_string(row).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Artifact file does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"Artifact path is not a file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def normalise_declared_sha256(value: str, label: str) -> str:
    declared = str(value).strip().lower()
    if len(declared) != 64:
        raise ValueError(f"{label} must be a 64-character SHA256 digest")
    try:
        int(declared, 16)
    except ValueError as error:
        raise ValueError(f"{label} must contain only hexadecimal SHA256 characters") from error
    return declared


def verify_declared_artifact(path: Path, declared_sha256: str, label: str) -> dict[str, object]:
    declared = normalise_declared_sha256(declared_sha256, label)
    fingerprint = file_fingerprint(path)
    if fingerprint["sha256"] != declared:
        raise ValueError(
            f"{label} does not match the actual file bytes: "
            f"declared={declared}, actual={fingerprint['sha256']}, path={fingerprint['resolved_path']}"
        )
    fingerprint["declared_sha256"] = declared
    return fingerprint


def assert_artifact_unchanged(fingerprint: dict[str, object], label: str) -> None:
    current = file_fingerprint(Path(str(fingerprint["resolved_path"])))
    if current["bytes"] != fingerprint["bytes"] or current["sha256"] != fingerprint["sha256"]:
        raise RuntimeError(
            f"{label} changed after its prequery fingerprint was captured: "
            f"before_bytes={fingerprint['bytes']}, after_bytes={current['bytes']}, "
            f"before_sha256={fingerprint['sha256']}, after_sha256={current['sha256']}"
        )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_protocol_bool(value: object) -> bool:
    if pd.isna(value):
        raise ValueError("missing value")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
        raise ValueError(f"unsupported numeric Boolean {value!r}")
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "y"}:
        return True
    if token in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"unsupported Boolean token {value!r}")


def parse_current_front_flags(values: pd.Series) -> pd.Series:
    parsed: list[bool] = []
    invalid: list[dict[str, object]] = []
    for index, value in values.items():
        try:
            parsed.append(parse_protocol_bool(value))
        except (TypeError, ValueError):
            invalid.append({"row": index, "value": value})
            parsed.append(False)
    if invalid:
        raise ValueError(
            "current_front_flag must contain only explicit Boolean values "
            f"(true/false, 1/0, yes/no, y/n); invalid entries: {invalid[:5]}"
        )
    return pd.Series(parsed, index=values.index, dtype=bool)


def require_finite_numeric(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        bad_columns = [
            name for index, name in enumerate(columns) if not finite[:, index].all()
        ]
        raise ValueError(f"{context} contains non-finite numeric values in: {bad_columns}")


def validate_feature_inputs(
    frame: pd.DataFrame, protocol: dict[str, object], context: str
) -> None:
    require_columns(frame, FEATURES)
    require_finite_numeric(frame, FEATURES, context)
    bounds = protocol["fixed_input_bounds"]
    for name in FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        lower = float(bounds[name]["min"])
        upper = float(bounds[name]["max"])
        outside = (values < lower) | (values > upper)
        if outside.any():
            bad_rows = frame.index[outside].tolist()[:5]
            raise ValueError(
                f"{context} feature {name} is outside frozen bounds [{lower}, {upper}] "
                f"at rows {bad_rows}"
            )
        if name in INTEGER_FEATURES and not np.equal(values, np.rint(values)).all():
            bad_rows = frame.index[~np.equal(values, np.rint(values))].tolist()[:5]
            raise ValueError(
                f"{context} integer feature {name} must contain exact integers; "
                f"invalid rows {bad_rows}"
            )


def computed_feature_hashes(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(feature_hash, axis=1).astype(str)


def validate_candidate_scores(frame: pd.DataFrame, protocol: dict[str, object]) -> pd.Series:
    require_columns(
        frame,
        [
            "candidate_id",
            *FEATURES,
            *CANDIDATE_SCORE_NUMERIC,
            "current_front_flag",
        ],
    )
    if frame["candidate_id"].isna().any() or frame["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate scored table contains a missing or empty candidate_id")
    if frame["candidate_id"].duplicated().any():
        duplicates = frame.loc[frame["candidate_id"].duplicated(keep=False), "candidate_id"].astype(str).unique()[:5]
        raise ValueError(f"Candidate scored table candidate_id values must be unique; duplicates: {duplicates.tolist()}")
    validate_feature_inputs(frame, protocol, "Candidate scored table")
    require_finite_numeric(frame, CANDIDATE_SCORE_NUMERIC, "Candidate scored table")
    parse_current_front_flags(frame["current_front_flag"])
    hashes = computed_feature_hashes(frame)
    if hashes.duplicated().any():
        duplicate_hashes = hashes.loc[hashes.duplicated(keep=False)].unique()[:5]
        raise ValueError(
            "Candidate scored table canonical input hashes must be unique; "
            f"duplicates: {duplicate_hashes.tolist()}"
        )
    if "input_row_sha256" in frame.columns:
        supplied = frame["input_row_sha256"].astype(str)
        mismatch = supplied.ne(hashes)
        if mismatch.any():
            raise ValueError(
                "Candidate scored table input_row_sha256 does not match the canonical feature hash "
                f"at rows {frame.index[mismatch].tolist()[:5]}"
            )
    return hashes


def validate_labelled_rows(frame: pd.DataFrame, protocol: dict[str, object]) -> set[str]:
    require_columns(frame, [*FEATURES, *PREDICTED])
    validate_feature_inputs(frame, protocol, "Labelled table")
    require_finite_numeric(frame, PREDICTED, "Labelled table")
    return set(computed_feature_hashes(frame))


def pct_rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, ascending=ascending).to_numpy(dtype=float)


def input_norm(frame: pd.DataFrame, protocol: dict[str, object]) -> np.ndarray:
    bounds = protocol["fixed_input_bounds"]
    lower = np.array([float(bounds[name]["min"]) for name in FEATURES])
    upper = np.array([float(bounds[name]["max"]) for name in FEATURES])
    return (frame[FEATURES].to_numpy(dtype=float) - lower) / np.maximum(upper - lower, 1e-12)


def objective_cost(frame: pd.DataFrame, front: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.column_stack(
        [
            -frame["carbon_utilization_pred"].to_numpy(dtype=float),
            frame["specific_energy_kwh_per_kg_meoh_pred"].to_numpy(dtype=float),
            frame["capex_proxy_pred"].to_numpy(dtype=float),
        ]
    )
    front_raw = np.column_stack(
        [
            -front["carbon_utilization_pred"].to_numpy(dtype=float),
            front["specific_energy_kwh_per_kg_meoh_pred"].to_numpy(dtype=float),
            front["capex_proxy_pred"].to_numpy(dtype=float),
        ]
    )
    ideal = np.min(front_raw, axis=0)
    nadir = np.max(front_raw, axis=0)
    normalised = (raw - ideal) / np.maximum(nadir - ideal, 1e-12)
    return normalised, ideal, nadir


def add_mcdm_ranks(front: pd.DataFrame) -> pd.DataFrame:
    out = front.copy()
    values = np.column_stack(
        [
            out["carbon_utilization_pred"].to_numpy(dtype=float),
            -out["specific_energy_kwh_per_kg_meoh_pred"].to_numpy(dtype=float),
            -out["capex_proxy_pred"].to_numpy(dtype=float),
        ]
    )
    utility = (values - values.min(axis=0)) / np.maximum(values.max(axis=0) - values.min(axis=0), 1e-12)
    weighted = utility / np.maximum(np.sqrt((utility**2).sum(axis=0)), 1e-12) / 3.0
    ideal = weighted.max(axis=0)
    anti = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    out["topsis_closeness_prequery"] = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    out["topsis_rank_prequery"] = pd.Series(-out["topsis_closeness_prequery"].to_numpy()).rank(method="first").astype(int).to_numpy()

    regret = 1.0 - utility
    weighted_regret = regret / 3.0
    s_val = weighted_regret.sum(axis=1)
    r_val = weighted_regret.max(axis=1)
    q_val = 0.5 * (s_val - s_val.min()) / max(float(s_val.max() - s_val.min()), 1e-12) + 0.5 * (
        r_val - r_val.min()
    ) / max(float(r_val.max() - r_val.min()), 1e-12)
    out["vikor_Q_prequery"] = q_val
    out["vikor_rank_prequery"] = pd.Series(q_val).rank(method="first").astype(int).to_numpy()
    return out


def prepare_candidate_scores(frame: pd.DataFrame, protocol: dict[str, object]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_candidate_scores(frame, protocol)
    out = frame.copy()
    out["current_front_flag"] = parse_current_front_flags(out["current_front_flag"])
    out = out.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    out["canonical_input_string_10dp"] = out.apply(canonical_feature_string, axis=1)
    out["input_row_sha256"] = computed_feature_hashes(out)
    front = out.loc[out["current_front_flag"]].copy()
    if len(front) < 5:
        raise ValueError("Current scored table has fewer than five front points")
    objective_norm, ideal, nadir = objective_cost(out, front)
    front_objective_norm, _, _ = objective_cost(front, front)
    distance_to_front = (
        NearestNeighbors(n_neighbors=1)
        .fit(front_objective_norm)
        .kneighbors(objective_norm, return_distance=True)[0][:, 0]
    )
    out["objective_distance_to_current_front"] = distance_to_front
    fixed_ranges = protocol["fixed_target_ranges"]
    uncertainty = np.column_stack(
        [
            out[f"{target}_std_prequery"].to_numpy(dtype=float) / max(float(fixed_ranges[target]), 1e-12)
            for target in TARGETS
        ]
    )
    out["committee_disagreement_fixed_range"] = np.sqrt(np.mean(uncertainty**2, axis=1))
    out["dominance_ambiguity_score"] = out["committee_disagreement_fixed_range"] / (
        out["committee_disagreement_fixed_range"] + out["objective_distance_to_current_front"] + 1e-12
    )
    x_norm = input_norm(out, protocol)
    out["boundary_proximity_fixed_bounds"] = np.min(np.minimum(x_norm, 1.0 - x_norm), axis=1)
    out["boundary_stress_score"] = 1.0 - pct_rank(
        out["boundary_proximity_fixed_bounds"].to_numpy(dtype=float)
    )
    out["rank_local_error"] = pct_rank(out["local_error_knn_score_fixed_range"].to_numpy(dtype=float))
    out["rank_disagreement"] = pct_rank(out["committee_disagreement_fixed_range"].to_numpy(dtype=float))
    out["rank_input_ad"] = pct_rank(out["input_ad_distance_to_development2400"].to_numpy(dtype=float))
    out["rank_front_proximity"] = pct_rank(out["objective_distance_to_current_front"].to_numpy(dtype=float), ascending=False)
    out["pareto_local_risk_score"] = (
        0.40 * out["rank_local_error"] + 0.35 * out["rank_disagreement"] + 0.25 * out["rank_input_ad"]
    )
    out["near_front_challenge_score"] = (
        0.45 * out["rank_front_proximity"]
        + 0.30 * out["pareto_local_risk_score"]
        + 0.25 * out["dominance_ambiguity_score"]
    )
    out["global_sentinel_score"] = (
        0.55 * out["rank_input_ad"] + 0.25 * out["rank_disagreement"] + 0.20 * out["boundary_stress_score"]
    )
    joint = np.column_stack([x_norm / math.sqrt(8.0), np.clip(objective_norm, -1.0, 2.0) / math.sqrt(3.0)])
    return out, x_norm, objective_norm, joint, np.array([*ideal, *nadir], dtype=float)


def excluded_hashes(paths: list[Path], protocol: dict[str, object]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        frame = read_csv(path)
        validate_feature_inputs(frame, protocol, f"Exclude table {path}")
        hashes.update(computed_feature_hashes(frame))
    return hashes


def collect_forbidden_hashes(
    paths: list[Path], labelled: pd.DataFrame, protocol: dict[str, object]
) -> set[str]:
    hashes = excluded_hashes(paths, protocol)
    hashes.update(validate_labelled_rows(labelled, protocol))
    return hashes


def greedy_select(
    frame: pd.DataFrame,
    joint: np.ndarray,
    pool_mask: np.ndarray,
    count: int,
    priority_column: str,
    selected: set[int],
    labelled_vectors: np.ndarray,
    priority_weight: float = 0.68,
) -> list[int]:
    pool = frame.loc[pool_mask & ~frame.index.isin(selected), ["candidate_id", priority_column]].copy()
    pool = pool.dropna(subset=[priority_column]).sort_values(
        [priority_column, "candidate_id"], ascending=[False, True], kind="mergesort"
    ).head(5000)
    if len(pool) < count:
        raise ValueError(f"Insufficient eligible rows for {priority_column}: {len(pool)} < {count}")
    indices = pool.index.to_numpy(dtype=int)
    priority = pool[priority_column].to_numpy(dtype=float)
    priority = (priority - priority.min()) / max(float(priority.max() - priority.min()), 1e-12)
    vectors = joint[indices]
    if len(labelled_vectors):
        minimum_distance = (
            NearestNeighbors(n_neighbors=1)
            .fit(labelled_vectors)
            .kneighbors(vectors, return_distance=True)[0][:, 0]
        )
    else:
        minimum_distance = np.ones(len(indices), dtype=float)
    available = np.ones(len(indices), dtype=bool)
    chosen: list[int] = []
    for _ in range(count):
        distance_rank = pct_rank(minimum_distance)
        utility = priority_weight * priority + (1.0 - priority_weight) * distance_rank
        utility[~available] = -np.inf
        best = float(np.max(utility))
        ties = np.flatnonzero(np.isclose(utility, best, rtol=0.0, atol=1e-15))
        chosen_local = min(ties, key=lambda loc: str(frame.loc[indices[loc], "candidate_id"]))
        chosen_idx = int(indices[chosen_local])
        chosen.append(chosen_idx)
        selected.add(chosen_idx)
        available[chosen_local] = False
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(vectors - vectors[chosen_local], axis=1)
        )
    return chosen


def add_queue_contract(
    selected: pd.DataFrame,
    mode: str,
    round_no: int | None,
    model_hash_set: str,
    front_sha256: str,
    target_definition_sha256: str,
) -> pd.DataFrame:
    out = selected.drop(columns=["selected_final_result"], errors="ignore").copy().reset_index(drop=True)
    if mode == "al32":
        prefix = f"V30ALR{round_no}_"
        role = "development_local_acquisition"
        train = True
        independent = False
    else:
        prefix = "V30CONF_"
        role = "test_v30_front_prospective_postfreeze"
        train = False
        independent = True
    out.insert(0, "record_id", [f"{prefix}{idx:03d}" for idx in range(1, len(out) + 1)])
    out.insert(1, "run_order", np.arange(1, len(out) + 1))
    out["dataset_role"] = role
    out["eligible_for_training_after_qc"] = train
    out["eligible_for_independent_validation"] = independent
    out["eligible_for_iterative_stop"] = False
    out["label_available_at_selection"] = False
    out["execute_gate"] = (
        "EXECUTE_AFTER_CURRENT_ROUND_QUEUE_FREEZE"
        if mode == "al32"
        else "EXECUTE_ONLY_AFTER_FINAL_V30_MODEL_FRONT_MCDM_ANALYSIS_AND_CONFIRMATION_QUEUE_FREEZE"
    )
    out["surrogate_model_hash_set_prequery"] = model_hash_set
    out["front_sha256_prequery"] = front_sha256
    out["target_definition_sha256"] = target_definition_sha256
    out["canonical_input_string_10dp"] = out.apply(canonical_feature_string, axis=1)
    out["input_row_sha256"] = out.apply(feature_hash, axis=1)
    out["Current_density_eCO2R_Acm2"] = out["J_eCO2R"]
    out["WE_Energy_input_rate_KW"] = out["P_WE"]
    out["Number_of_cells_stacked_in_series_WE"] = np.rint(out["N_WE"]).astype(int)
    out["Number_of_stacks_eCO2R"] = np.rint(out["N_eCO2R"]).astype(int)
    out["Outlet_pressure"] = out["P_MeOH"]
    out["Outlet_pressure_cap"] = out["P_Cap"]
    out["T_eCO2R_gproms_input"] = out["T_eCO2R"]
    out["Utility_constant_temperature"] = out["T_utility"] - 273.15
    for column in [
        *RAW_OUTPUT_COLUMNS,
        *PRIMITIVE_RAW_COLUMNS,
        *ECHO_COLUMNS,
        *PROVENANCE_COLUMNS,
        *DERIVED_RETURN_COLUMNS,
    ]:
        out[column] = pd.NA
    return out


def select_al32(frame: pd.DataFrame, joint: np.ndarray, eligible: np.ndarray, labelled_vectors: np.ndarray) -> pd.DataFrame:
    front = parse_current_front_flags(frame["current_front_flag"]).to_numpy(dtype=bool)
    nonfront = ~front
    positive = frame.loc[nonfront, "objective_distance_to_current_front"]
    near = nonfront & (
        frame["objective_distance_to_current_front"].to_numpy(dtype=float) <= float(positive.quantile(0.08))
    )
    frame["round_dominance_score"] = (
        0.50 * frame["dominance_ambiguity_score"]
        + 0.35 * frame["pareto_local_risk_score"]
        + 0.15 * pct_rank(frame["input_ad_distance_to_development2400"].to_numpy(dtype=float))
    )
    frame["front_diversity_priority"] = pct_rank(
        frame["input_ad_distance_to_development2400"].to_numpy(dtype=float)
    )
    selected: set[int] = set()
    rows: list[tuple[int, str]] = []
    for mask, count, priority, stratum, weight in [
        (front | near, 16, "round_dominance_score", "dominance_uncertain", 0.68),
        (front, 8, "front_diversity_priority", "front_maximin", 0.48),
        (near, 4, "near_front_challenge_score", "near_front_shell", 0.68),
        (np.ones(len(frame), dtype=bool), 4, "global_sentinel_score", "global_sentinel", 0.68),
    ]:
        chosen = greedy_select(frame, joint, eligible & mask, count, priority, selected, labelled_vectors, weight)
        rows.extend((idx, stratum) for idx in chosen)
        labelled_vectors = np.vstack([labelled_vectors, joint[chosen]]) if len(labelled_vectors) else joint[chosen]
    out = frame.loc[[idx for idx, _ in rows]].copy()
    out.insert(0, "selection_stratum", [stratum for _, stratum in rows])
    return out


def boundary_or_low_ad_mask(frame: pd.DataFrame) -> np.ndarray:
    boundary_proximity = frame["boundary_proximity_fixed_bounds"].to_numpy(dtype=float)
    input_ad_distance = frame["input_ad_distance_to_development2400"].to_numpy(dtype=float)
    boundary = boundary_proximity <= float(np.quantile(boundary_proximity, 0.08))
    low_applicability_domain = input_ad_distance >= float(np.quantile(input_ad_distance, 0.92))
    return boundary | low_applicability_domain


def select_confirmation96(
    frame: pd.DataFrame,
    objective_norm: np.ndarray,
    joint: np.ndarray,
    eligible: np.ndarray,
    labelled_vectors: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    front_mask = parse_current_front_flags(frame["current_front_flag"]).to_numpy(dtype=bool)
    front_ranked = add_mcdm_ranks(frame.loc[front_mask])
    frame = frame.drop(
        columns=["topsis_rank_prequery", "vikor_rank_prequery"], errors="ignore"
    )
    frame = frame.merge(
        front_ranked[["candidate_id", "topsis_rank_prequery", "vikor_rank_prequery"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    frame.index = np.arange(len(frame))
    front_indices = np.flatnonzero(front_mask)
    anchor_candidates = {
        "TOPSIS": int(frame.loc[front_indices, "topsis_rank_prequery"].astype(float).idxmin()),
        "VIKOR": int(frame.loc[front_indices, "vikor_rank_prequery"].astype(float).idxmin()),
        "Max CU": int(frame.loc[front_indices, "carbon_utilization_pred"].astype(float).idxmax()),
        "Min SEC": int(frame.loc[front_indices, "specific_energy_kwh_per_kg_meoh_pred"].astype(float).idxmin()),
        "Min CAPEX": int(frame.loc[front_indices, "capex_proxy_pred"].astype(float).idxmin()),
    }
    selected: set[int] = set()
    rows: list[tuple[int, str, str | None]] = []
    anchor_audit: list[dict[str, object]] = []
    eligible_front = eligible & front_mask
    for role, anchor_idx in anchor_candidates.items():
        if eligible[anchor_idx] and anchor_idx not in selected:
            chosen_idx = anchor_idx
            selection_type = "exact_unlabelled_anchor"
        else:
            candidates = np.flatnonzero(eligible_front & ~frame.index.isin(selected))
            distance = np.linalg.norm(joint[candidates] - joint[anchor_idx], axis=1)
            order = sorted(
                range(len(candidates)),
                key=lambda loc: (float(distance[loc]), str(frame.loc[candidates[loc], "candidate_id"])),
            )
            chosen_idx = int(candidates[order[0]])
            selection_type = "unlabelled_nearest_front_replacement"
        selected.add(chosen_idx)
        rows.append((chosen_idx, "updated_v30_front_coverage", role))
        anchor_audit.append(
            {
                "anchor_role": role,
                "exact_anchor_candidate_id": frame.loc[anchor_idx, "candidate_id"],
                "exact_anchor_was_eligible_unlabelled": bool(eligible[anchor_idx]),
                "confirmation_metric_candidate_id": frame.loc[chosen_idx, "candidate_id"],
                "selection_type": selection_type,
            }
        )
    remaining_front = 40 - len(rows)
    chosen = greedy_select(
        frame,
        joint,
        eligible_front,
        remaining_front,
        "input_ad_distance_to_development2400",
        selected,
        np.vstack([labelled_vectors, joint[list(selected)]]) if len(labelled_vectors) else joint[list(selected)],
        0.48,
    )
    rows.extend((idx, "updated_v30_front_coverage", None) for idx in chosen)
    front_selected = [idx for idx, stratum, _ in rows if stratum == "updated_v30_front_coverage"]

    nonfront = ~front_mask
    positive = frame.loc[nonfront, "objective_distance_to_current_front"]
    selected_front_cost = objective_norm[front_selected]
    candidate_cost = objective_norm
    dominated_by_selected_front = np.zeros(len(frame), dtype=bool)
    for start in range(0, len(frame), 5000):
        stop = min(start + 5000, len(frame))
        block = candidate_cost[start:stop]
        weak = selected_front_cost[:, None, :] <= block[None, :, :] + 1e-12
        strict = selected_front_cost[:, None, :] < block[None, :, :] - 1e-12
        dominated_by_selected_front[start:stop] = np.any(
            np.all(weak, axis=2) & np.any(strict, axis=2), axis=0
        )
    near_distance_threshold = float(positive.quantile(0.08))
    eligible_dominated_distances = frame.loc[
        eligible & nonfront & dominated_by_selected_front,
        "objective_distance_to_current_front",
    ].sort_values()
    if len(eligible_dominated_distances) < 24:
        raise RuntimeError("Fewer than 24 eligible challengers are dominated by the 40 issued front points")
    near_distance_threshold = max(
        near_distance_threshold, float(eligible_dominated_distances.iloc[23])
    )
    near = nonfront & dominated_by_selected_front & (
        frame["objective_distance_to_current_front"].to_numpy(dtype=float) <= near_distance_threshold
    )
    chosen = greedy_select(
        frame,
        joint,
        eligible & near,
        24,
        "near_front_challenge_score",
        selected,
        np.vstack([labelled_vectors, joint[list(selected)]]) if len(labelled_vectors) else joint[list(selected)],
    )
    rows.extend((idx, "updated_v30_near_front_challenger", None) for idx in chosen)

    topsis_idx = anchor_candidates["TOPSIS"]
    vikor_idx = anchor_candidates["VIKOR"]
    distance_top = np.linalg.norm(joint - joint[topsis_idx], axis=1)
    distance_vik = np.linalg.norm(joint - joint[vikor_idx], axis=1)
    frame["preference_neighbourhood_priority"] = 1.0 - pct_rank(
        np.minimum(distance_top, distance_vik)
    )
    for role, distance in [("TOPSIS", distance_top), ("VIKOR", distance_vik)]:
        group = distance <= float(np.quantile(distance, 0.08))
        chosen = greedy_select(
            frame,
            joint,
            eligible & group,
            10,
            "preference_neighbourhood_priority",
            selected,
            np.vstack([labelled_vectors, joint[list(selected)]]) if len(labelled_vectors) else joint[list(selected)],
        )
        rows.extend((idx, f"updated_v30_{role.lower()}_neighbourhood", None) for idx in chosen)

    boundary_or_low_ad = boundary_or_low_ad_mask(frame)
    chosen = greedy_select(
        frame,
        joint,
        eligible & boundary_or_low_ad,
        12,
        "global_sentinel_score",
        selected,
        np.vstack([labelled_vectors, joint[list(selected)]]) if len(labelled_vectors) else joint[list(selected)],
    )
    rows.extend((idx, "updated_v30_boundary_or_low_AD", None) for idx in chosen)

    if len(rows) != 96 or len({idx for idx, _, _ in rows}) != 96:
        raise RuntimeError("Confirmation selection did not produce 96 unique rows")
    out = frame.loc[[idx for idx, _, _ in rows]].copy()
    out.insert(0, "forced_anchor_role", [role for _, _, role in rows])
    out.insert(0, "selection_stratum", [stratum for _, stratum, _ in rows])
    anchor_type_map = {
        str(item["confirmation_metric_candidate_id"]): str(item["selection_type"])
        for item in anchor_audit
    }
    out["anchor_metric_selection_type"] = out["candidate_id"].astype(str).map(anchor_type_map)
    out["paired_front_candidate_id"] = pd.NA
    out["predicted_pair_relation"] = pd.NA
    front_obj = objective_norm[front_selected]
    challenger_mask = out["selection_stratum"].eq("updated_v30_near_front_challenger")
    challenger_indices = out.index[challenger_mask]
    pair_ids: list[str] = []
    for challenger_idx in challenger_indices:
        challenger_cost = objective_norm[int(challenger_idx)]
        weak = front_obj <= challenger_cost[None, :] + 1e-12
        strict = front_obj < challenger_cost[None, :] - 1e-12
        dominator_locations = np.flatnonzero(np.all(weak, axis=1) & np.any(strict, axis=1))
        if len(dominator_locations) == 0:
            raise RuntimeError("A selected challenger has no verified predicted dominator among the 40 issued front rows")
        distances = np.linalg.norm(front_obj[dominator_locations] - challenger_cost, axis=1)
        best_location = min(
            range(len(dominator_locations)),
            key=lambda loc: (
                float(distances[loc]),
                str(frame.loc[front_selected[int(dominator_locations[loc])], "candidate_id"]),
            ),
        )
        front_location = int(dominator_locations[best_location])
        pair_ids.append(str(frame.loc[front_selected[front_location], "candidate_id"]))
    out.loc[challenger_indices, "paired_front_candidate_id"] = pair_ids
    out.loc[challenger_indices, "predicted_pair_relation"] = "selected_front_predicted_to_dominate_challenger"
    return out, anchor_audit


def expected_selection_strata(
    mode: str, protocol: dict[str, object]
) -> dict[str, int]:
    if mode == "al32":
        active_learning = protocol["active_learning"]
        quotas = active_learning["quotas"]
        expected = {
            "dominance_uncertain": int(quotas["dominance_uncertain"]),
            "front_maximin": int(quotas["front_maximin"]),
            "near_front_shell": int(quotas["near_front_shell"]),
            "global_sentinel": int(quotas["global_sentinel"]),
        }
        if sum(expected.values()) != int(active_learning["batch_size"]):
            raise ValueError("Active-learning protocol quotas do not sum to batch_size")
        return expected

    quotas = protocol["postfreeze_confirmation96"]["quotas"]
    preference_count = int(quotas["preference_neighbourhood"])
    if preference_count % 2:
        raise ValueError("Confirmation preference_neighbourhood quota must split evenly between TOPSIS and VIKOR")
    expected = {
        "updated_v30_front_coverage": int(quotas["updated_front_coverage"]),
        "updated_v30_near_front_challenger": int(quotas["near_front_challengers"]),
        "updated_v30_topsis_neighbourhood": preference_count // 2,
        "updated_v30_vikor_neighbourhood": preference_count // 2,
        "updated_v30_boundary_or_low_AD": int(quotas["boundary_or_low_AD"]),
    }
    if sum(expected.values()) != 96:
        raise ValueError("Post-freeze confirmation protocol quotas do not sum to 96")
    return expected


def validate_selected_queue(
    queue: pd.DataFrame,
    mode: str,
    protocol: dict[str, object],
    forbidden: set[str],
) -> None:
    require_columns(
        queue,
        ["candidate_id", "selection_stratum", "input_row_sha256", *FEATURES, *PREDICTED],
    )
    expected = expected_selection_strata(mode, protocol)
    expected_count = sum(expected.values())
    if len(queue) != expected_count:
        raise RuntimeError(f"Selected queue row count is {len(queue)}, expected {expected_count}")
    if queue["candidate_id"].duplicated().any():
        raise RuntimeError("Selected queue contains duplicate candidate_id values")
    if queue["input_row_sha256"].isna().any() or queue["input_row_sha256"].duplicated().any():
        raise RuntimeError("Selected queue input hashes must be present and unique")
    recomputed_hashes = computed_feature_hashes(queue)
    if not queue["input_row_sha256"].astype(str).equals(recomputed_hashes):
        raise RuntimeError("Selected queue input hashes do not match its canonical feature values")
    validate_feature_inputs(queue, protocol, "Selected queue")
    require_finite_numeric(queue, PREDICTED, "Selected queue")
    overlap = set(queue["input_row_sha256"].astype(str)).intersection(forbidden)
    if overlap:
        raise RuntimeError(
            "Selected queue overlaps excluded or prior-labelled input hashes: "
            f"{sorted(overlap)[:5]}"
        )
    observed = {
        str(name): int(count)
        for name, count in queue["selection_stratum"].value_counts(dropna=False).items()
    }
    if observed != expected:
        raise RuntimeError(
            f"Selected queue strata do not match protocol quotas; observed={observed}, expected={expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic V30 AL or post-freeze confirmation selector")
    parser.add_argument("--mode", choices=["al32", "confirm96"], required=True)
    parser.add_argument("--candidate-scored", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--labelled", type=Path, required=True, help="All currently labelled rows used only for diversity coordinates")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--model-hash-set", required=True)
    parser.add_argument(
        "--model-hash-set-file",
        type=Path,
        required=True,
        help="Immutable model bundle or model-hash-set sidecar whose raw SHA256 equals --model-hash-set",
    )
    parser.add_argument("--front-sha256", required=True)
    parser.add_argument(
        "--front-file",
        type=Path,
        required=True,
        help="Frozen predicted-front artifact whose raw SHA256 equals --front-sha256",
    )
    parser.add_argument("--target-definition-sha256", required=True)
    parser.add_argument(
        "--target-definition-file",
        type=Path,
        required=True,
        help="Frozen target-definition artifact whose raw SHA256 equals --target-definition-sha256",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "al32" and (args.round is None or args.round not in {2, 3, 4}):
        raise ValueError("--round 2, 3 or 4 is required for al32")

    candidate_artifact = file_fingerprint(args.candidate_scored)
    model_artifact = verify_declared_artifact(
        args.model_hash_set_file, args.model_hash_set, "--model-hash-set"
    )
    front_artifact = verify_declared_artifact(
        args.front_file, args.front_sha256, "--front-sha256"
    )
    target_artifact = verify_declared_artifact(
        args.target_definition_file,
        args.target_definition_sha256,
        "--target-definition-sha256",
    )

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    candidates_raw = read_csv(args.candidate_scored)
    candidates, _, objective_norm, joint, ideal_nadir = prepare_candidate_scores(candidates_raw, protocol)
    labelled = read_csv(args.labelled)
    forbidden = collect_forbidden_hashes(args.exclude, labelled, protocol)
    eligible = ~candidates["input_row_sha256"].isin(forbidden).to_numpy(dtype=bool)
    labelled_input = input_norm(labelled, protocol) / math.sqrt(8.0)
    ideal = ideal_nadir[:3]
    nadir = ideal_nadir[3:]
    labelled_cost = np.column_stack(
        [
            -labelled["carbon_utilization_pred"].to_numpy(dtype=float),
            labelled["specific_energy_kwh_per_kg_meoh_pred"].to_numpy(dtype=float),
            labelled["capex_proxy_pred"].to_numpy(dtype=float),
        ]
    )
    labelled_objective = (labelled_cost - ideal) / np.maximum(nadir - ideal, 1e-12)
    labelled_vectors = np.column_stack(
        [labelled_input, np.clip(labelled_objective, -1.0, 2.0) / math.sqrt(3.0)]
    )

    anchor_audit: list[dict[str, object]] = []
    if args.mode == "al32":
        selected = select_al32(candidates, joint, eligible, labelled_vectors)
    else:
        selected, anchor_audit = select_confirmation96(
            candidates, objective_norm, joint, eligible, labelled_vectors
        )
    queue = add_queue_contract(
        selected,
        args.mode,
        args.round,
        str(model_artifact["sha256"]),
        str(front_artifact["sha256"]),
        str(target_artifact["sha256"]),
    )
    validate_selected_queue(queue, args.mode, protocol, forbidden)

    prequery_artifacts = {
        "candidate_scored": candidate_artifact,
        "model_hash_set": model_artifact,
        "predicted_front": front_artifact,
        "target_definition": target_artifact,
    }
    for label, fingerprint in prequery_artifacts.items():
        assert_artifact_unchanged(fingerprint, label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / (
        f"V30_AL_round{args.round}_issued.csv" if args.mode == "al32" else "V30_postfreeze_confirmation96_issued.csv"
    )
    queue.to_csv(queue_path, index=False)
    freeze = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "round": args.round,
        "candidate_scored_path": str(args.candidate_scored),
        "candidate_scored_resolved_path": candidate_artifact["resolved_path"],
        "candidate_scored_bytes": candidate_artifact["bytes"],
        "candidate_scored_sha256": candidate_artifact["sha256"],
        "protocol_sha256": file_sha256(args.protocol),
        "surrogate_model_hash_set_prequery": model_artifact["sha256"],
        "front_sha256_prequery": front_artifact["sha256"],
        "target_definition_sha256": target_artifact["sha256"],
        "prequery_artifacts": prequery_artifacts,
        "exclude_files": {str(path): file_sha256(path) for path in args.exclude},
        "labelled_file": str(args.labelled),
        "labelled_file_sha256": file_sha256(args.labelled),
        "predicted_front_cost_ideal": ideal_nadir[:3].tolist(),
        "predicted_front_cost_nadir": ideal_nadir[3:].tolist(),
        "hypervolume_primary_reference_normalised": [2.0, 2.0, 2.0],
        "hypervolume_outside_reference_rule": "Never clip; if any predicted or actual coordinate is >=2.0, use max(2.0,1.05*maximum_union_coordinate+0.01) for each affected common coordinate and report fallback",
        "selected_rows": len(queue),
        "selected_unique_hashes": int(queue["input_row_sha256"].nunique()),
        "queue_sha256": file_sha256(queue_path),
        "anchor_audit": anchor_audit,
    }
    freeze_path = args.output_dir / "selection_freeze_record.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(freeze, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
