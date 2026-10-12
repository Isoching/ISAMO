from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from isamo_green_meoh.moea import nsga3_optimize_auto
from isamo_green_meoh.surrogate import predict_surrogates


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "handoff" / "V30_publication_pareto_local_validation"
DATA = PACKAGE / "data"
EXECUTION = ROOT / "outputs" / "V30_publication_validation_execution"
FREEZE = EXECUTION / "phase3_final_model" / "final_freeze"
OUT = EXECUTION / "phase3_final_analysis_freeze"
RUNS = OUT / "nsga3_runs"

FEATURES = [
    "P_Cap",
    "J_eCO2R",
    "T_eCO2R",
    "N_eCO2R",
    "P_WE",
    "N_WE",
    "P_MeOH",
    "T_utility",
]
INTEGER_FEATURES = {"N_eCO2R", "N_WE"}
TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]
PREDICTED = [f"{target}_pred" for target in TARGETS]
STD = [f"{target}_std_prequery" for target in TARGETS]
SEEDS = [42, 137, 271, 419, 613, 809, 1009, 1217, 1429, 1693]
POPULATION = 256
GENERATIONS = 300
REFERENCE_PARTITIONS = 20
HV_REFERENCE = np.array([1.10, 1.10, 1.10], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def canonical_string(row: pd.Series) -> str:
    values = []
    for feature in FEATURES:
        value = float(row[feature])
        values.append(str(int(round(value))) if feature in INTEGER_FEATURES else f"{value:.10f}")
    return "|".join(values)


def canonical_hashes(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: hashlib.sha256(canonical_string(row).encode("utf-8")).hexdigest(), axis=1
    )


def rounded_inputs(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(values, dtype=float), lower, upper).copy()
    for feature in INTEGER_FEATURES:
        index = FEATURES.index(feature)
        out[:, index] = np.rint(out[:, index])
    return out


def cost_matrix(mu: np.ndarray) -> np.ndarray:
    return np.column_stack([-mu[:, 0], mu[:, 1], mu[:, 2]])


def nondominated_indices(cost: np.ndarray) -> np.ndarray:
    return np.asarray(
        NonDominatedSorting(method="efficient_non_dominated_sort").do(
            np.asarray(cost, dtype=float), only_non_dominated_front=True
        ),
        dtype=int,
    )


def objective_reference() -> dict[str, float]:
    legacy = pd.concat(
        [read(DATA / "01_legacy_development_2400.csv"), read(DATA / "02_legacy_holdout_600.csv")],
        ignore_index=True,
    )
    return {
        "cu_min": float(legacy[TARGETS[0]].min()),
        "cu_max": float(legacy[TARGETS[0]].max()),
        "sec_min": float(legacy[TARGETS[1]].min()),
        "sec_max": float(legacy[TARGETS[1]].max()),
        "capex_min": float(legacy[TARGETS[2]].min()),
        "capex_max": float(legacy[TARGETS[2]].max()),
    }


def normalize_cost(cost: np.ndarray, reference: dict[str, float]) -> np.ndarray:
    values = np.asarray(cost, dtype=float)
    return np.column_stack(
        [
            (values[:, 0] + reference["cu_max"])
            / max(reference["cu_max"] - reference["cu_min"], 1e-12),
            (values[:, 1] - reference["sec_min"])
            / max(reference["sec_max"] - reference["sec_min"], 1e-12),
            (values[:, 2] - reference["capex_min"])
            / max(reference["capex_max"] - reference["capex_min"], 1e-12),
        ]
    )


def add_mcdm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    utility = np.column_stack(
        [
            out[PREDICTED[0]].to_numpy(dtype=float),
            -out[PREDICTED[1]].to_numpy(dtype=float),
            -out[PREDICTED[2]].to_numpy(dtype=float),
        ]
    )
    utility = (utility - utility.min(axis=0)) / np.maximum(
        utility.max(axis=0) - utility.min(axis=0), 1e-12
    )
    weighted = utility / np.maximum(np.sqrt((utility**2).sum(axis=0)), 1e-12) / 3.0
    ideal = weighted.max(axis=0)
    anti = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    out["topsis_closeness_prequery"] = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    out["topsis_rank_prequery"] = (
        pd.Series(-out["topsis_closeness_prequery"].to_numpy())
        .rank(method="first")
        .astype(int)
        .to_numpy()
    )
    regret = (1.0 - utility) / 3.0
    s_value = regret.sum(axis=1)
    r_value = regret.max(axis=1)
    out["vikor_Q_prequery"] = 0.5 * (s_value - s_value.min()) / max(
        float(s_value.max() - s_value.min()), 1e-12
    ) + 0.5 * (r_value - r_value.min()) / max(
        float(r_value.max() - r_value.min()), 1e-12
    )
    out["vikor_rank_prequery"] = (
        out["vikor_Q_prequery"].rank(method="first").astype(int).to_numpy()
    )
    return out


def pct_rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, ascending=ascending).to_numpy()


def load_models() -> tuple[dict[str, object], dict[str, object]]:
    freeze_record = json.loads((FREEZE / "final_model_freeze.json").read_text(encoding="utf-8"))
    if freeze_record["status"] != "FROZEN_BEFORE_LOCKBOX_OPENING":
        raise RuntimeError("Final model was not frozen before lockbox opening")
    models = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for target in TARGETS:
            path = FREEZE / "models" / f"{target}.joblib"
            expected = freeze_record["selected_model_assets"][target]
            if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Frozen final model mismatch: {target}")
            models[target] = joblib.load(path)
    messages = sorted({str(item.message) for item in caught})
    expected = {
        "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n"
        "As an alternative, you can create a new view using np.reshape (with copy=False if needed)."
    }
    unexpected = sorted(set(messages) - expected)
    if unexpected:
        raise RuntimeError("Frozen final model load emitted unexpected warnings: " + " | ".join(unexpected))
    return models, {
        "warning_count": len(caught),
        "warning_messages": messages,
        "classification": "KNOWN_JOBLIB_NUMPY_2_5_SHAPE_DEPRECATION_ONLY",
        "prediction_warnings_permitted": False,
    }


def evaluate_factory(
    models: dict[str, object],
    lower: np.ndarray,
    upper: np.ndarray,
    audit: dict[str, list[np.ndarray]],
):
    def evaluate(values: np.ndarray) -> np.ndarray:
        x = rounded_inputs(values, lower, upper)
        frame = pd.DataFrame(x, columns=FEATURES)
        prediction = predict_surrogates(models, frame, FEATURES, return_uncertainty=False)
        mu = prediction[TARGETS].to_numpy(dtype=float)
        if not np.isfinite(mu).all():
            raise RuntimeError("Non-finite final surrogate prediction during NSGA-III")
        cost = cost_matrix(mu)
        audit["X"].append(x)
        audit["F"].append(cost)
        return cost

    return evaluate


def archive_from_audit(audit: dict[str, list[np.ndarray]], seed: int) -> pd.DataFrame:
    x = np.vstack(audit["X"])
    cost = np.vstack(audit["F"])
    keys = np.column_stack(
        [np.rint(x[:, i]) if feature in INTEGER_FEATURES else np.round(x[:, i], 10) for i, feature in enumerate(FEATURES)]
    )
    _, unique = np.unique(keys, axis=0, return_index=True)
    unique = np.sort(unique)
    x, cost = x[unique], cost[unique]
    front_index = nondominated_indices(cost)
    out = pd.DataFrame(x[front_index], columns=FEATURES)
    out[PREDICTED[0]] = -cost[front_index, 0]
    out[PREDICTED[1]] = cost[front_index, 1]
    out[PREDICTED[2]] = cost[front_index, 2]
    out.insert(0, "pareto_id_within_seed", [f"S{seed}_P{i:05d}" for i in range(len(out))])
    out.insert(0, "source_seed", seed)
    return out


def load_actual_local() -> pd.DataFrame:
    frames = [read(DATA / "03_existing60_discovery_round0.csv")]
    for round_index in range(1, 5):
        frames.append(
            read(
                EXECUTION
                / f"phase1_AL_R{round_index}_32_finalized"
                / "gproms_return_rebuilt_qc.csv"
            )
        )
    local = pd.concat(frames, ignore_index=True, sort=False)
    for target in TARGETS:
        direct = (
            pd.to_numeric(local[target], errors="coerce")
            if target in local.columns
            else pd.Series(np.nan, index=local.index, dtype=float)
        )
        rebuilt_name = f"{target}_rebuilt"
        rebuilt = (
            pd.to_numeric(local[rebuilt_name], errors="coerce")
            if rebuilt_name in local.columns
            else pd.Series(np.nan, index=local.index, dtype=float)
        )
        local[target] = direct.fillna(rebuilt)
    local = local.loc[np.isfinite(local[FEATURES + TARGETS].to_numpy(dtype=float)).all(axis=1)].copy()
    hashes = canonical_hashes(local)
    local = local.loc[~hashes.duplicated()].reset_index(drop=True)
    if len(local) != 186:
        raise RuntimeError(f"Expected 186 permitted local labels, found {len(local)}")
    return local


def annotate_combined(
    original: pd.DataFrame,
    nsga_union: pd.DataFrame,
    models: dict[str, object],
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lower = np.array([float(protocol["fixed_input_bounds"][name]["min"]) for name in FEATURES])
    upper = np.array([float(protocol["fixed_input_bounds"][name]["max"]) for name in FEATURES])
    existing_hashes = set(canonical_hashes(original))
    additions = nsga_union.copy()
    additions["input_row_sha256"] = canonical_hashes(additions)
    additions = additions.loc[~additions["input_row_sha256"].isin(existing_hashes)].copy()
    additions = additions.drop_duplicates("input_row_sha256").reset_index(drop=True)
    additions["candidate_id"] = additions["input_row_sha256"].map(lambda value: f"V30NSGA_{value[:16]}")

    prediction = predict_surrogates(models, additions, FEATURES, return_uncertainty=True)
    for target in TARGETS:
        additions[f"{target}_pred"] = prediction[target].to_numpy(dtype=float)
        additions[f"{target}_std_prequery"] = prediction[f"{target}_std"].to_numpy(dtype=float)

    template_columns = list(original.columns)
    for column in template_columns:
        if column not in additions:
            additions[column] = pd.NA
    for column in additions.columns:
        if column not in original:
            original[column] = pd.NA
    combined = pd.concat([original, additions[original.columns]], ignore_index=True, sort=False)
    combined["canonical_input_string_10dp"] = combined.apply(canonical_string, axis=1)
    combined["feature_key_10dp"] = combined["canonical_input_string_10dp"]
    combined["input_row_sha256"] = canonical_hashes(combined)
    if combined["input_row_sha256"].duplicated().any():
        raise RuntimeError("Combined NSGA/candidate universe contains duplicate canonical hashes")

    mu = combined[PREDICTED].to_numpy(dtype=float)
    front_index = nondominated_indices(cost_matrix(mu))
    combined["current_front_flag"] = False
    combined.loc[front_index, "current_front_flag"] = True
    front = add_mcdm(combined.iloc[front_index].copy().reset_index(drop=True))
    rank_map = front.set_index("candidate_id")
    combined["topsis_rank_prequery"] = combined["candidate_id"].map(rank_map["topsis_rank_prequery"])
    combined["vikor_rank_prequery"] = combined["candidate_id"].map(rank_map["vikor_rank_prequery"])
    combined["topsis_closeness_prequery"] = combined["candidate_id"].map(
        rank_map["topsis_closeness_prequery"]
    )
    combined["vikor_Q_prequery"] = combined["candidate_id"].map(
        rank_map["vikor_Q_prequery"]
    )

    x_norm = (combined[FEATURES].to_numpy(dtype=float) - lower) / np.maximum(upper - lower, 1e-12)
    development = read(DATA / "01_legacy_development_2400.csv")
    dev_norm = (development[FEATURES].to_numpy(dtype=float) - lower) / np.maximum(upper - lower, 1e-12)
    combined["input_ad_distance_to_development2400"] = (
        NearestNeighbors(n_neighbors=1).fit(dev_norm).kneighbors(x_norm, return_distance=True)[0][:, 0]
    )

    front_cost = cost_matrix(front[PREDICTED].to_numpy(dtype=float))
    ideal, nadir = front_cost.min(axis=0), front_cost.max(axis=0)
    objective_norm = (cost_matrix(mu) - ideal) / np.maximum(nadir - ideal, 1e-12)
    front_norm = (front_cost - ideal) / np.maximum(nadir - ideal, 1e-12)
    combined["objective_distance_to_current_front"] = (
        NearestNeighbors(n_neighbors=1).fit(front_norm).kneighbors(objective_norm, return_distance=True)[0][:, 0]
    )

    local = load_actual_local()
    local_prediction = predict_surrogates(models, local, FEATURES, return_uncertainty=False)
    local_input_norm = (local[FEATURES].to_numpy(dtype=float) - lower) / np.maximum(upper - lower, 1e-12)
    local_cost_norm = (cost_matrix(local_prediction[TARGETS].to_numpy(dtype=float)) - ideal) / np.maximum(nadir - ideal, 1e-12)
    joint = np.column_stack([x_norm / math.sqrt(8.0), np.clip(objective_norm, -1.0, 2.0) / math.sqrt(3.0)])
    local_joint = np.column_stack([local_input_norm / math.sqrt(8.0), np.clip(local_cost_norm, -1.0, 2.0) / math.sqrt(3.0)])
    distances, neighbours = NearestNeighbors(n_neighbors=5).fit(local_joint).kneighbors(joint, return_distance=True)
    ranges = np.array([float(protocol["fixed_target_ranges"][target]) for target in TARGETS])
    residual = np.abs(local_prediction[TARGETS].to_numpy(dtype=float) - local[TARGETS].to_numpy(dtype=float)) / ranges
    composite = np.sqrt(np.mean(residual**2, axis=1))
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    combined["local_error_knn_score_fixed_range"] = np.sum(weights * composite[neighbours], axis=1)
    combined["joint_distance_to_existing60"] = distances[:, 0]

    uncertainty = combined[STD].to_numpy(dtype=float) / ranges
    combined["committee_disagreement_fixed_range"] = np.sqrt(np.mean(uncertainty**2, axis=1))
    combined["dominance_ambiguity_score"] = combined["committee_disagreement_fixed_range"] / (
        combined["committee_disagreement_fixed_range"] + combined["objective_distance_to_current_front"] + 1e-12
    )
    combined["boundary_proximity_fixed_bounds"] = np.min(np.minimum(x_norm, 1.0 - x_norm), axis=1)
    combined["boundary_stress_score"] = 1.0 - pct_rank(combined["boundary_proximity_fixed_bounds"].to_numpy(dtype=float))
    combined["rank_local_error"] = pct_rank(combined["local_error_knn_score_fixed_range"].to_numpy(dtype=float))
    combined["rank_disagreement"] = pct_rank(combined["committee_disagreement_fixed_range"].to_numpy(dtype=float))
    combined["rank_input_ad"] = pct_rank(combined["input_ad_distance_to_development2400"].to_numpy(dtype=float))
    combined["rank_front_proximity"] = pct_rank(combined["objective_distance_to_current_front"].to_numpy(dtype=float), ascending=False)
    combined["pareto_local_risk_score"] = 0.40 * combined["rank_local_error"] + 0.35 * combined["rank_disagreement"] + 0.25 * combined["rank_input_ad"]
    combined["near_front_challenge_score"] = 0.45 * combined["rank_front_proximity"] + 0.30 * combined["pareto_local_risk_score"] + 0.25 * combined["dominance_ambiguity_score"]
    combined["global_sentinel_score"] = 0.55 * combined["rank_input_ad"] + 0.25 * combined["rank_disagreement"] + 0.20 * combined["boundary_stress_score"]

    anchors = pd.DataFrame(
        [
            {"case": "TOPSIS", **front.loc[front["topsis_rank_prequery"].idxmin()].to_dict()},
            {"case": "VIKOR", **front.loc[front["vikor_rank_prequery"].idxmin()].to_dict()},
            {"case": "Max CU", **front.loc[front[PREDICTED[0]].idxmax()].to_dict()},
            {"case": "Min SEC", **front.loc[front[PREDICTED[1]].idxmin()].to_dict()},
            {"case": "Min CAPEX", **front.loc[front[PREDICTED[2]].idxmin()].to_dict()},
        ]
    )
    export_columns = [
        "candidate_id",
        *FEATURES,
        *PREDICTED,
        *STD,
        "current_front_flag",
        "objective_distance_to_current_front",
        "input_ad_distance_to_development2400",
        "local_error_knn_score_fixed_range",
        "committee_disagreement_fixed_range",
        "dominance_ambiguity_score",
        "boundary_proximity_fixed_bounds",
        "boundary_stress_score",
        "rank_local_error",
        "rank_disagreement",
        "rank_input_ad",
        "rank_front_proximity",
        "pareto_local_risk_score",
        "near_front_challenge_score",
        "global_sentinel_score",
        "topsis_closeness_prequery",
        "topsis_rank_prequery",
        "vikor_Q_prequery",
        "vikor_rank_prequery",
        "canonical_input_string_10dp",
        "input_row_sha256",
        "source_seed",
        "source_seed_support_count",
        "source_seeds",
    ]
    export_columns = [column for column in export_columns if column in combined.columns]
    combined = combined[export_columns].copy()
    clean_front = combined.loc[combined["current_front_flag"]].copy().reset_index(drop=True)
    return combined, clean_front, anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Final fixed V30 ten-seed NSGA-III and pre-lockbox analysis freeze")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    requested_seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if any(seed not in SEEDS for seed in requested_seeds):
        raise ValueError(f"Seeds must be members of the frozen ten-seed set: {SEEDS}")
    if args.run_only and args.aggregate_only:
        raise ValueError("--run-only and --aggregate-only are mutually exclusive")
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    protocol_path = DATA / "protocol_config.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bounds = protocol["fixed_input_bounds"]
    lower = np.array([float(bounds[name]["min"]) for name in FEATURES])
    upper = np.array([float(bounds[name]["max"]) for name in FEATURES])
    models, model_load_audit = load_models()
    reference = objective_reference()
    hv = HV(ref_point=HV_REFERENCE)
    run_summaries: list[dict[str, object]] = []
    fronts: list[pd.DataFrame] = []
    run_seeds = [] if args.aggregate_only else requested_seeds
    for position, seed in enumerate(run_seeds, 1):
        run_dir = RUNS / f"seed_{seed}"
        archive_path = run_dir / "pareto_archive.csv"
        history_path = run_dir / "history.csv"
        summary_path = run_dir / "summary.json"
        if args.resume and archive_path.exists() and history_path.exists() and summary_path.exists():
            archive = read(archive_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[{position}/{len(run_seeds)}] seed={seed}: resumed", flush=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            audit: dict[str, list[np.ndarray]] = {"X": [], "F": []}
            started = time.perf_counter()

            def history_callback(generation: int, _x: np.ndarray, cost: np.ndarray) -> dict[str, float]:
                front_cost = cost[nondominated_indices(cost)]
                return {"hypervolume": float(hv(normalize_cost(front_cost, reference)))}

            print(f"[{position}/{len(run_seeds)}] seed={seed}: running", flush=True)
            result = nsga3_optimize_auto(
                evaluate_factory(models, lower, upper, audit),
                list(zip(lower, upper)),
                pop_size=POPULATION,
                n_generations=GENERATIONS,
                n_reference_partitions=REFERENCE_PARTITIONS,
                crossover_prob=0.9,
                mutation_prob=0.2,
                mutation_scale=0.08,
                random_state=seed,
                prefer_pymoo=True,
                history_callback=history_callback,
            )
            archive = archive_from_audit(audit, seed)
            history = pd.DataFrame(result.history)
            history.insert(0, "seed", seed)
            evaluated = np.vstack(audit["X"])
            evaluated_frame = pd.DataFrame(evaluated, columns=FEATURES)
            unique_evaluated = int(canonical_hashes(evaluated_frame).nunique())
            final_hv = float(hv(normalize_cost(cost_matrix(archive[PREDICTED].to_numpy(dtype=float)), reference)))
            summary = {
                "seed": seed,
                "status": "complete",
                "engine": result.engine,
                "evaluated_unique": unique_evaluated,
                "pareto_size": len(archive),
                "final_hypervolume": final_hv,
                "elapsed_seconds": time.perf_counter() - started,
            }
            archive.to_csv(archive_path, index=False, lineterminator="\n")
            history.to_csv(history_path, index=False, lineterminator="\n")
            write_json(summary_path, summary)
            print(
                f"[{position}/{len(run_seeds)}] seed={seed}: complete; Pareto={len(archive)}, HV={final_hv:.8f}, elapsed={summary['elapsed_seconds'] / 60:.2f} min",
                flush=True,
            )
        fronts.append(archive)
        run_summaries.append(summary)

    if args.run_only:
        print(f"Completed run-only seeds: {run_seeds}", flush=True)
        return

    if args.aggregate_only:
        for seed in SEEDS:
            run_dir = RUNS / f"seed_{seed}"
            archive_path = run_dir / "pareto_archive.csv"
            summary_path = run_dir / "summary.json"
            if not archive_path.exists() or not summary_path.exists():
                raise RuntimeError(f"Cannot aggregate: seed {seed} is incomplete")
            fronts.append(read(archive_path))
            run_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    pooled = pd.concat(fronts, ignore_index=True, sort=False)
    pooled["input_row_sha256"] = canonical_hashes(pooled)
    seed_support = (
        pooled.groupby("input_row_sha256")["source_seed"]
        .agg(
            source_seed_support_count="nunique",
            source_seeds=lambda values: ";".join(map(str, sorted(set(map(int, values))))),
        )
        .reset_index()
    )
    pooled = pooled.drop_duplicates("input_row_sha256", keep="first").reset_index(drop=True)
    union = pooled.iloc[nondominated_indices(cost_matrix(pooled[PREDICTED].to_numpy(dtype=float)))].copy().reset_index(drop=True)
    union = union.merge(seed_support, on="input_row_sha256", how="left", validate="one_to_one")
    union = add_mcdm(union)
    union.to_csv(OUT / "cross_seed_nondominated_union.csv", index=False, lineterminator="\n")

    distance_rows = []
    normalized = {
        int(frame["source_seed"].iloc[0]): normalize_cost(cost_matrix(frame[PREDICTED].to_numpy(dtype=float)), reference)
        for frame in fronts
    }
    completed_seeds = sorted(normalized)
    if completed_seeds != sorted(SEEDS):
        raise RuntimeError(f"Formal aggregation requires all frozen seeds; found {completed_seeds}")
    for left_index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[left_index + 1 :]:
            distances = cdist(normalized[left_seed], normalized[right_seed])
            left_to_right, right_to_left = distances.min(axis=1), distances.min(axis=0)
            distance_rows.append(
                {
                    "seed_a": left_seed,
                    "seed_b": right_seed,
                    "mean_symmetric_nearest_distance": float(0.5 * (left_to_right.mean() + right_to_left.mean())),
                    "hausdorff_distance": float(max(left_to_right.max(), right_to_left.max())),
                }
            )
    pairwise = pd.DataFrame(distance_rows)
    pairwise.to_csv(OUT / "pairwise_front_distances.csv", index=False, lineterminator="\n")
    pd.DataFrame(run_summaries).to_csv(OUT / "run_summary.csv", index=False, lineterminator="\n")

    original = read(FREEZE / "V30_final_candidate_scored.csv")
    combined, combined_front, anchors = annotate_combined(original, union, models, protocol)
    combined_path = OUT / "V30_final_candidate_plus_nsga_scored.csv"
    front_path = OUT / "V30_final_predicted_front_plus_nsga.csv"
    anchor_path = OUT / "V30_final_five_decision_anchors.csv"
    labelled_path = OUT / "V30_final_labelled_for_diversity.csv"
    combined.to_csv(combined_path, index=False, lineterminator="\n")
    combined_front.to_csv(front_path, index=False, lineterminator="\n")
    anchors.to_csv(anchor_path, index=False, lineterminator="\n")
    labelled = load_actual_local()
    labelled_prediction = predict_surrogates(
        models, labelled, FEATURES, return_uncertainty=False
    )
    for target in TARGETS:
        labelled[f"{target}_pred"] = labelled_prediction[target].to_numpy(dtype=float)
    labelled["input_row_sha256"] = canonical_hashes(labelled)
    labelled[["record_id", "input_row_sha256", *FEATURES, *PREDICTED]].to_csv(
        labelled_path, index=False, lineterminator="\n"
    )

    freeze_record_path = FREEZE / "final_model_freeze.json"
    conformal_path = FREEZE / "conformal" / "cross_conformal_freeze.json"
    code_path = Path(__file__).resolve()
    artifacts = {
        "final_model_freeze": sha256(freeze_record_path),
        "cross_conformal_freeze": sha256(conformal_path),
        "run_summary": sha256(OUT / "run_summary.csv"),
        "pairwise_front_distances": sha256(OUT / "pairwise_front_distances.csv"),
        "cross_seed_nondominated_union": sha256(OUT / "cross_seed_nondominated_union.csv"),
        "combined_candidate_scored": sha256(combined_path),
        "combined_predicted_front": sha256(front_path),
        "five_decision_anchors": sha256(anchor_path),
        "final_labelled_for_diversity": sha256(labelled_path),
        "protocol_config": sha256(protocol_path),
        "analysis_code": sha256(code_path),
    }
    analysis_freeze = {
        "schema_version": "V30-final-analysis-freeze-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANALYSIS_FROZEN_BEFORE_CONFIRMATION96_LABEL_OPENING",
        "development_stopping_decision": "BUDGET_EXHAUSTED_NO_PLATEAU",
        "optimizer": {
            "algorithm": "NSGA-III",
            "seeds": SEEDS,
            "population_size": POPULATION,
            "generations": GENERATIONS,
            "reference_partitions": REFERENCE_PARTITIONS,
            "crossover_probability": 0.9,
            "mutation_probability": 0.2,
            "mutation_scale": 0.08,
            "integer_variables_rounded_before_every_prediction": sorted(INTEGER_FEATURES),
            "objectives": {"maximize": [TARGETS[0]], "minimize": TARGETS[1:]},
        },
        "run_count": len(run_summaries),
        "all_runs_complete": all(item["status"] == "complete" for item in run_summaries),
        "cross_seed_nondominated_union_rows": len(union),
        "combined_candidate_rows": len(combined),
        "combined_front_rows": len(combined_front),
        "labelled_diversity_rows": len(labelled),
        "mcdm": {"weights": [1 / 3, 1 / 3, 1 / 3], "vikor_v": 0.5},
        "model_load_audit": model_load_audit,
        "five_anchor_cases": anchors["case"].tolist(),
        "final_model_freeze_binding_sha256": json.loads(freeze_record_path.read_text(encoding="utf-8"))["freeze_binding_sha256"],
        "cross_conformal_binding_sha256": json.loads(conformal_path.read_text(encoding="utf-8"))["calibration_binding_sha256"],
        "artifacts": artifacts,
    }
    analysis_freeze["analysis_freeze_binding_sha256"] = json_sha256(analysis_freeze)
    write_json(OUT / "final_analysis_freeze.json", analysis_freeze)
    print(json.dumps(analysis_freeze, indent=2), flush=True)


if __name__ == "__main__":
    main()
