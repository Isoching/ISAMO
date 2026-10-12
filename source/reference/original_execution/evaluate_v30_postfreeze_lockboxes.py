from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_compare_freeze_v30_models import (  # noqa: E402
    gated_prediction,
    load_frozen,
    load_pooled_models,
    predict_models,
)


PACKAGE = ROOT / "handoff" / "V30_publication_pareto_local_validation"
DATA = PACKAGE / "data"
EXECUTION = ROOT / "outputs" / "V30_publication_validation_execution"
MODEL_ROOT = EXECUTION / "phase3_final_model"
FINAL_FREEZE = MODEL_ROOT / "final_freeze"
CONFIRMATION_ISSUED = EXECUTION / "phase3_confirmation96_issued"
DEFAULT_OUTPUT = EXECUTION / "phase5_postfreeze_scientific_evaluation"

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
TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]
ACTUAL = {target: f"{target}_rebuilt" for target in TARGETS}
VARIANTS = [
    "v29_frozen_baseline",
    "pooled_existing60",
    "gated_local_residual",
    "pooled_all_al",
]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260826


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finite_frame(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite values")


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def bootstrap_error_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    fixed_range: float,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    n = len(actual)
    if n < 2:
        return {
            "bootstrap_replicates": replicates,
            "rmse_ci95_low": math.nan,
            "rmse_ci95_high": math.nan,
            "mae_ci95_low": math.nan,
            "mae_ci95_high": math.nan,
            "fixed_range_nrmse_pct_ci95_low": math.nan,
            "fixed_range_nrmse_pct_ci95_high": math.nan,
        }
    error = predicted - actual
    indices = rng.integers(0, n, size=(replicates, n))
    sampled = error[indices]
    rmse = np.sqrt(np.mean(sampled**2, axis=1))
    mae = np.mean(np.abs(sampled), axis=1)
    q = [0.025, 0.975]
    rmse_ci = np.quantile(rmse, q)
    mae_ci = np.quantile(mae, q)
    return {
        "bootstrap_replicates": replicates,
        "rmse_ci95_low": float(rmse_ci[0]),
        "rmse_ci95_high": float(rmse_ci[1]),
        "mae_ci95_low": float(mae_ci[0]),
        "mae_ci95_high": float(mae_ci[1]),
        "fixed_range_nrmse_pct_ci95_low": float(100.0 * rmse_ci[0] / fixed_range),
        "fixed_range_nrmse_pct_ci95_high": float(100.0 * rmse_ci[1] / fixed_range),
    }


def safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    left = pd.Series(actual)
    right = pd.Series(predicted)
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return math.nan
    return float(left.corr(right, method="spearman"))


def metric_row(
    dataset: str,
    stratum: str,
    variant: str,
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    fixed_range: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    local_range = float(np.ptp(actual))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    row: dict[str, Any] = {
        "dataset": dataset,
        "stratum": stratum,
        "analysis_scope": (
            "DESCRIPTIVE_CONVERGED_SUBSET_ONLY"
            if dataset == "global40"
            else "FORMAL_INDEPENDENT_LOCKBOX"
        ),
        "variant": variant,
        "target": target,
        "n": int(len(actual)),
        "rmse": rmse,
        "mae": mae,
        "local_observed_range": local_range,
        "nrmse_local_range_pct": 100.0 * rmse / local_range if local_range > 0.0 else math.nan,
        "frozen_legacy_holdout_range": fixed_range,
        "nrmse_fixed_range_pct": 100.0 * rmse / fixed_range,
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator > 0.0 else math.nan,
        "spearman_rho": safe_spearman(actual, predicted),
        "bias_pred_minus_gproms": float(np.mean(error)),
        "absolute_bias_pct_fixed_range": 100.0 * abs(float(np.mean(error))) / fixed_range,
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
        "p90_absolute_error_pct_fixed_range": 100.0 * float(np.quantile(np.abs(error), 0.90)) / fixed_range,
    }
    row.update(bootstrap_error_intervals(actual, predicted, fixed_range, rng))
    return row


def cost_matrix(values: np.ndarray) -> np.ndarray:
    return np.column_stack([-values[:, 0], values[:, 1], values[:, 2]])


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        NonDominatedSorting(method="efficient_non_dominated_sort").do(
            cost_matrix(values), only_non_dominated_front=True
        ),
        dtype=int,
    )
    mask = np.zeros(len(values), dtype=bool)
    mask[indices] = True
    return mask


def relation(left_cost: np.ndarray, right_cost: np.ndarray, tolerance: float = 1e-12) -> str:
    left_dominates = bool(
        np.all(left_cost <= right_cost + tolerance)
        and np.any(left_cost < right_cost - tolerance)
    )
    right_dominates = bool(
        np.all(right_cost <= left_cost + tolerance)
        and np.any(right_cost < left_cost - tolerance)
    )
    if left_dominates:
        return "left_dominates_right"
    if right_dominates:
        return "right_dominates_left"
    if np.all(np.abs(left_cost - right_cost) <= tolerance):
        return "equal"
    return "incomparable"


def mcdm(frame: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    out = frame[["record_id", "candidate_id"]].copy()
    values = frame[columns].to_numpy(dtype=float)
    utility = np.column_stack([values[:, 0], -values[:, 1], -values[:, 2]])
    utility = (utility - utility.min(axis=0)) / np.maximum(
        utility.max(axis=0) - utility.min(axis=0), 1e-12
    )
    weighted = utility / np.maximum(np.sqrt((utility**2).sum(axis=0)), 1e-12) / 3.0
    ideal = weighted.max(axis=0)
    anti = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    closeness = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    regret = (1.0 - utility) / 3.0
    s_value = regret.sum(axis=1)
    r_value = regret.max(axis=1)
    q_value = 0.5 * (s_value - s_value.min()) / max(float(np.ptp(s_value)), 1e-12) + 0.5 * (
        r_value - r_value.min()
    ) / max(float(np.ptp(r_value)), 1e-12)
    out[f"topsis_closeness_{suffix}"] = closeness
    out[f"topsis_rank_{suffix}"] = pd.Series(-closeness).rank(method="first").astype(int)
    out[f"vikor_q_{suffix}"] = q_value
    out[f"vikor_rank_{suffix}"] = pd.Series(q_value).rank(method="first").astype(int)
    return out


def load_and_validate_lockboxes() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    specifications = {
        "fixed60": (
            EXECUTION / "phase4_fixed60_finalized" / "fixed_region60_execution_result.json",
            "PASS_QC_ASSEMBLED",
            60,
        ),
        "global40": (
            EXECUTION / "phase4_global40_finalized" / "global40_execution_result.json",
            "FAIL_LOW_CONVERGENCE",
            12,
        ),
        "confirmation96": (
            EXECUTION / "phase4_confirmation96_finalized" / "confirmation96_execution_result.json",
            "PASS_QC_ASSEMBLED",
            94,
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for dataset, (result_path, expected_status, expected_accepted) in specifications.items():
        result = read_json(result_path)
        if result["status"] != expected_status or int(result["accepted_labels"]) != expected_accepted:
            raise RuntimeError(f"Unexpected {dataset} execution result")
        accepted_path = result_path.parent / "gproms_return_rebuilt_qc.csv"
        if sha256(accepted_path) != result["accepted_output_csv_sha256"]:
            raise RuntimeError(f"{dataset} accepted CSV hash mismatch")
        frame = read_csv(accepted_path)
        if len(frame) != expected_accepted:
            raise RuntimeError(f"{dataset} accepted row count mismatch")
        finite_frame(frame, FEATURES + list(ACTUAL.values()), dataset)
        if frame["record_id"].duplicated().any() or frame["input_row_sha256"].duplicated().any():
            raise RuntimeError(f"{dataset} contains duplicate accepted records/hashes")
        frames[dataset] = frame
        evidence[dataset] = {
            "result_path": result_path.relative_to(ROOT).as_posix(),
            "result_sha256": sha256(result_path),
            "status": result["status"],
            "issued_records": int(result["issued_records"]),
            "accepted_labels": int(result["accepted_labels"]),
            "failed_records": int(result["failed_records"]),
            "convergence_pct": float(result["convergence_pct"]),
            "accepted_csv": accepted_path.relative_to(ROOT).as_posix(),
            "accepted_csv_sha256": sha256(accepted_path),
        }
    return frames, evidence


def predict_all_variants(combined: pd.DataFrame, protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}

    print("Loading frozen V29 baseline through the approved inference-only recovery...", flush=True)
    base_models, recovery = load_frozen()
    predictions["v29_frozen_baseline"], _ = predict_models(base_models, combined)
    predictions["gated_local_residual"], _ = gated_prediction(base_models, combined, protocol)
    audit["v29_portable_recovery"] = recovery
    del base_models
    gc.collect()

    print("Loading pooled-existing60 variant...", flush=True)
    pooled_existing = load_pooled_models(MODEL_ROOT / "pooled_existing60")
    predictions["pooled_existing60"], _ = predict_models(pooled_existing, combined)
    del pooled_existing
    gc.collect()

    print("Loading selected pooled-all-AL final variant...", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_models = {
            target: joblib.load(FINAL_FREEZE / "models" / f"{target}.joblib")
            for target in TARGETS
        }
    final_mu, final_sd = predict_models(final_models, combined)
    predictions["pooled_all_al"] = final_mu
    audit["selected_model_load_warning_count"] = len(caught)
    audit["selected_model_load_warning_messages"] = sorted({str(item.message) for item in caught})
    del final_models
    gc.collect()
    return predictions, final_sd, audit


def conformal_tables(
    combined: pd.DataFrame,
    final_mu: np.ndarray,
    final_sd: np.ndarray,
    dataset_slices: dict[str, slice],
    conformal: dict[str, Any],
    fixed_ranges: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pointwise_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for dataset, dataset_slice in dataset_slices.items():
        frame = combined.iloc[dataset_slice].reset_index(drop=True)
        for target_index, target in enumerate(TARGETS):
            actual = frame[ACTUAL[target]].to_numpy(dtype=float)
            predicted = final_mu[dataset_slice, target_index]
            sd = np.maximum(final_sd[dataset_slice, target_index], 0.0)
            contract = conformal["quantiles"][target]
            half_width = float(contract["finite_sample_quantile"]) * np.maximum(
                sd, float(contract["scale_floor"])
            )
            lower = predicted - half_width
            upper = predicted + half_width
            if target == "carbon_utilization":
                lower = np.clip(lower, 0.0, 0.999)
                upper = np.clip(upper, 0.0, 0.999)
            else:
                lower = np.clip(lower, 1e-6, None)
                upper = np.clip(upper, 1e-6, None)
            covered = (actual >= lower) & (actual <= upper)
            for index in range(len(frame)):
                pointwise_rows.append(
                    {
                        "dataset": dataset,
                        "record_id": frame.loc[index, "record_id"],
                        "selection_stratum": frame.loc[index, "selection_stratum"],
                        "target": target,
                        "actual_gproms": actual[index],
                        "predicted_final": predicted[index],
                        "ensemble_sd": sd[index],
                        "interval_lower_95": lower[index],
                        "interval_upper_95": upper[index],
                        "interval_width_95": upper[index] - lower[index],
                        "covered_95": bool(covered[index]),
                    }
                )
            lower_ci, upper_ci = clopper_pearson(int(covered.sum()), len(covered))
            metric_rows.append(
                {
                    "dataset": dataset,
                    "analysis_scope": (
                        "DESCRIPTIVE_CONVERGED_SUBSET_ONLY"
                        if dataset == "global40"
                        else "FORMAL_INDEPENDENT_LOCKBOX"
                    ),
                    "target": target,
                    "n": int(len(covered)),
                    "covered_n": int(covered.sum()),
                    "nominal_coverage_pct": 95.0,
                    "empirical_coverage_pct": 100.0 * float(covered.mean()),
                    "exact_binomial_ci95_low_pct": 100.0 * lower_ci,
                    "exact_binomial_ci95_high_pct": 100.0 * upper_ci,
                    "mean_interval_width": float(np.mean(upper - lower)),
                    "mean_interval_width_pct_fixed_range": 100.0
                    * float(np.mean(upper - lower))
                    / fixed_ranges[target],
                    "coverage_ge_90_pct": bool(float(covered.mean()) >= 0.90),
                }
            )
    return pd.DataFrame(pointwise_rows), pd.DataFrame(metric_rows)


def confirmation_front_and_decision(
    confirmation: pd.DataFrame,
    final_mu: np.ndarray,
    selection_freeze: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    frame = confirmation.reset_index(drop=True).copy()
    actual = frame[list(ACTUAL.values())].to_numpy(dtype=float)
    predicted = final_mu
    predicted_mask = nondominated_mask(predicted)
    actual_mask = nondominated_mask(actual)
    membership = frame[["record_id", "candidate_id", "selection_stratum", "forced_anchor_role"]].copy()
    membership["predicted_nondominated"] = predicted_mask
    membership["actual_nondominated"] = actual_mask
    membership["membership_changed"] = predicted_mask != actual_mask
    intersection = int(np.logical_and(predicted_mask, actual_mask).sum())
    union = int(np.logical_or(predicted_mask, actual_mask).sum())

    ideal = np.asarray(selection_freeze["predicted_front_cost_ideal"], dtype=float)
    nadir = np.asarray(selection_freeze["predicted_front_cost_nadir"], dtype=float)
    predicted_normalized = (cost_matrix(predicted) - ideal) / (nadir - ideal)
    actual_normalized = (cost_matrix(actual) - ideal) / (nadir - ideal)
    reference = np.asarray(selection_freeze["hypervolume_primary_reference_normalised"], dtype=float)
    maximum = np.maximum(predicted_normalized.max(axis=0), actual_normalized.max(axis=0))
    fallback_axes = maximum >= reference
    common_reference = reference.copy()
    common_reference[fallback_axes] = 1.05 * maximum[fallback_axes] + 0.01
    hv = HV(ref_point=common_reference)
    predicted_hv = float(hv(predicted_normalized[predicted_mask]))
    actual_hv = float(hv(actual_normalized[actual_mask]))
    hv_relative_error = 100.0 * abs(actual_hv - predicted_hv) / max(predicted_hv, 1e-12)

    pair_rows: list[dict[str, Any]] = []
    by_candidate = {str(row.candidate_id): index for index, row in frame.iterrows()}
    challengers = frame.loc[frame["paired_front_candidate_id"].notna()]
    for challenger_index, challenger in challengers.iterrows():
        paired_id = str(challenger["paired_front_candidate_id"])
        paired_index = by_candidate.get(paired_id)
        if paired_index is None:
            pair_rows.append(
                {
                    "challenger_record_id": challenger["record_id"],
                    "challenger_candidate_id": challenger["candidate_id"],
                    "front_candidate_id": paired_id,
                    "evaluable": False,
                    "reason": "paired front candidate did not converge",
                    "predicted_relation": "left_dominates_right",
                    "actual_relation": None,
                    "relation_correct": False,
                }
            )
            continue
        predicted_relation = relation(cost_matrix(predicted)[paired_index], cost_matrix(predicted)[challenger_index])
        actual_relation = relation(cost_matrix(actual)[paired_index], cost_matrix(actual)[challenger_index])
        pair_rows.append(
            {
                "challenger_record_id": challenger["record_id"],
                "challenger_candidate_id": challenger["candidate_id"],
                "front_candidate_id": paired_id,
                "evaluable": True,
                "reason": "",
                "predicted_relation": predicted_relation,
                "actual_relation": actual_relation,
                "relation_correct": predicted_relation == actual_relation,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    evaluable_pairs = pairs.loc[pairs["evaluable"]]

    predicted_rank = mcdm(frame, [f"_pred_{target}" for target in TARGETS], "predicted")
    actual_columns = list(ACTUAL.values())
    actual_rank = mcdm(frame, actual_columns, "actual")
    ranking = predicted_rank.merge(actual_rank, on=["record_id", "candidate_id"], validate="one_to_one")
    for method in ["topsis", "vikor"]:
        pred_top = set(ranking.nsmallest(10, f"{method}_rank_predicted")["candidate_id"])
        actual_top = set(ranking.nsmallest(10, f"{method}_rank_actual")["candidate_id"])
        ranking[f"{method}_predicted_top10"] = ranking["candidate_id"].isin(pred_top)
        ranking[f"{method}_actual_top10"] = ranking["candidate_id"].isin(actual_top)

    anchor_columns = ["record_id", "candidate_id", "forced_anchor_role", "selection_stratum"]
    anchors = frame.loc[frame["forced_anchor_role"].notna(), anchor_columns].merge(
        ranking, on=["record_id", "candidate_id"], how="left", validate="one_to_one"
    )
    anchors = anchors.merge(
        membership[["record_id", "actual_nondominated", "predicted_nondominated"]],
        on="record_id",
        validate="one_to_one",
    )
    for target_index, target in enumerate(TARGETS):
        mapping_pred = dict(zip(frame["record_id"], predicted[:, target_index]))
        mapping_actual = dict(zip(frame["record_id"], actual[:, target_index]))
        anchors[f"{target}_predicted"] = anchors["record_id"].map(mapping_pred)
        anchors[f"{target}_actual"] = anchors["record_id"].map(mapping_actual)
        anchors[f"{target}_error_pred_minus_actual"] = (
            anchors[f"{target}_predicted"] - anchors[f"{target}_actual"]
        )

    front_summary = {
        "confirmation_converged_n": len(frame),
        "predicted_nondominated_n": int(predicted_mask.sum()),
        "actual_nondominated_n": int(actual_mask.sum()),
        "intersection_n": intersection,
        "union_n": union,
        "nondominated_membership_jaccard": intersection / max(union, 1),
        "jaccard_ge_0_75": intersection / max(union, 1) >= 0.75,
        "jaccard_ge_0_80_preferred": intersection / max(union, 1) >= 0.80,
        "issued_dominance_pairs": 24,
        "evaluable_dominance_pairs": int(len(evaluable_pairs)),
        "unevaluable_dominance_pairs": int(24 - len(evaluable_pairs)),
        "dominance_pair_correct_n": int(evaluable_pairs["relation_correct"].sum()),
        "dominance_pair_accuracy_pct_evaluable": 100.0 * float(evaluable_pairs["relation_correct"].mean()),
        "dominance_pair_accuracy_pct_issued_conservative": 100.0 * float(evaluable_pairs["relation_correct"].sum()) / 24.0,
        "dominance_pair_accuracy_ge_90_pct": 100.0 * float(evaluable_pairs["relation_correct"].sum()) / 24.0 >= 90.0,
        "normalisation_ideal": ideal.tolist(),
        "normalisation_nadir": nadir.tolist(),
        "primary_reference": reference.tolist(),
        "common_reference_used": common_reference.tolist(),
        "outside_reference_fallback_triggered": bool(fallback_axes.any()),
        "outside_reference_fallback_axes": [int(i) for i in np.flatnonzero(fallback_axes)],
        "predicted_hypervolume": predicted_hv,
        "actual_hypervolume": actual_hv,
        "hypervolume_retention_pct": 100.0 * actual_hv / max(predicted_hv, 1e-12),
        "hypervolume_relative_error_pct": hv_relative_error,
        "hypervolume_relative_error_le_5_pct": hv_relative_error <= 5.0,
    }
    ranking_summary = {
        "rank_universe_converged_n": len(frame),
        "failed_issued_rows_excluded_but_reported_n": 2,
        "topsis_rank_spearman": float(ranking["topsis_rank_predicted"].corr(ranking["topsis_rank_actual"], method="spearman")),
        "vikor_rank_spearman": float(ranking["vikor_rank_predicted"].corr(ranking["vikor_rank_actual"], method="spearman")),
        "topsis_top10_overlap_n": int((ranking["topsis_predicted_top10"] & ranking["topsis_actual_top10"]).sum()),
        "topsis_top10_overlap_pct": 10.0 * int((ranking["topsis_predicted_top10"] & ranking["topsis_actual_top10"]).sum()),
        "vikor_top10_overlap_n": int((ranking["vikor_predicted_top10"] & ranking["vikor_actual_top10"]).sum()),
        "vikor_top10_overlap_pct": 10.0 * int((ranking["vikor_predicted_top10"] & ranking["vikor_actual_top10"]).sum()),
    }
    ranking_summary["rank_spearman_ge_0_90_both"] = bool(
        ranking_summary["topsis_rank_spearman"] >= 0.90
        and ranking_summary["vikor_rank_spearman"] >= 0.90
    )
    ranking_summary["top10_overlap_ge_70_pct_both"] = bool(
        ranking_summary["topsis_top10_overlap_pct"] >= 70.0
        and ranking_summary["vikor_top10_overlap_pct"] >= 70.0
    )
    return membership, front_summary, ranking, ranking_summary, anchors, pairs


def build_report_markdown(summary: dict[str, Any], metrics: pd.DataFrame, conformal: pd.DataFrame) -> str:
    confirmation = metrics.loc[
        (metrics["dataset"] == "confirmation96")
        & (metrics["stratum"] == "ALL")
        & (metrics["variant"] == "pooled_all_al")
    ]
    fixed = metrics.loc[
        (metrics["dataset"] == "fixed60")
        & (metrics["stratum"] == "ALL")
        & (metrics["variant"] == "pooled_all_al")
    ]
    lines = [
        "# V30 post-freeze scientific evaluation",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Outcome",
        "",
        f"- Overall status: **{summary['overall_status']}**.",
        f"- confirmation96 convergence: {summary['lockboxes']['confirmation96']['accepted_labels']}/96 ({summary['lockboxes']['confirmation96']['convergence_pct']:.3f}%).",
        f"- fixed60 convergence: {summary['lockboxes']['fixed60']['accepted_labels']}/60 ({summary['lockboxes']['fixed60']['convergence_pct']:.3f}%).",
        f"- global40 convergence: {summary['lockboxes']['global40']['accepted_labels']}/40 ({summary['lockboxes']['global40']['convergence_pct']:.3f}%); this lockbox failed its convergence gate, so its 12-row metrics are descriptive only.",
        "",
        "## Selected final model: confirmation96",
        "",
        "| Target | RMSE | local NRMSE | fixed-range NRMSE | Spearman | bias (% fixed range) | p90 abs error (% fixed range) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in confirmation.itertuples():
        lines.append(
            f"| {row.target} | {row.rmse:.6g} | {row.nrmse_local_range_pct:.3f}% | {row.nrmse_fixed_range_pct:.3f}% | {row.spearman_rho:.4f} | {row.absolute_bias_pct_fixed_range:.3f}% | {row.p90_absolute_error_pct_fixed_range:.3f}% |"
        )
    lines += [
        "",
        "## Selected final model: fixed60",
        "",
        "| Target | RMSE | local NRMSE | fixed-range NRMSE | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fixed.itertuples():
        lines.append(
            f"| {row.target} | {row.rmse:.6g} | {row.nrmse_local_range_pct:.3f}% | {row.nrmse_fixed_range_pct:.3f}% | {row.spearman_rho:.4f} |"
        )
    lines += [
        "",
        "## Front and decision fidelity",
        "",
        f"- Nondominated-membership Jaccard: {summary['front_fidelity']['nondominated_membership_jaccard']:.4f}.",
        f"- Dominance-pair accuracy (conservative denominator 24): {summary['front_fidelity']['dominance_pair_accuracy_pct_issued_conservative']:.2f}%.",
        f"- Common-reference hypervolume relative error: {summary['front_fidelity']['hypervolume_relative_error_pct']:.3f}%.",
        f"- TOPSIS rank Spearman/top-10 overlap: {summary['decision_fidelity']['topsis_rank_spearman']:.4f} / {summary['decision_fidelity']['topsis_top10_overlap_pct']:.1f}%.",
        f"- VIKOR rank Spearman/top-10 overlap: {summary['decision_fidelity']['vikor_rank_spearman']:.4f} / {summary['decision_fidelity']['vikor_top10_overlap_pct']:.1f}%.",
        "",
        "## Conformal coverage",
        "",
        "| Dataset | Target | n | coverage | exact 95% CI | mean width (% fixed range) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in conformal.itertuples():
        lines.append(
            f"| {row.dataset} | {row.target} | {row.n} | {row.empirical_coverage_pct:.2f}% | [{row.exact_binomial_ci95_low_pct:.2f}, {row.exact_binomial_ci95_high_pct:.2f}]% | {row.mean_interval_width_pct_fixed_range:.2f}% |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "The evidence may support only the exact claims marked supported in `claim_evidence.csv`. The global40 convergence failure prevents a global gPROMS-validation claim. The defensible scope remains a prospectively gPROMS-confirmed surrogate-assisted Pareto-local design region and compromise recommendation within the prescribed domain, subject to the failed gates reported here.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the three post-freeze V30 lockboxes without refitting")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protocol_path = DATA / "protocol_config.json"
    protocol = read_json(protocol_path)
    fixed_ranges = {target: float(protocol["fixed_target_ranges"][target]) for target in TARGETS}
    final_freeze_path = FINAL_FREEZE / "final_model_freeze.json"
    final_freeze = read_json(final_freeze_path)
    if final_freeze["status"] != "FROZEN_BEFORE_LOCKBOX_OPENING" or final_freeze["selected_variant"] != "pooled_all_al":
        raise RuntimeError("Unexpected final model freeze")
    conformal_path = FINAL_FREEZE / "conformal" / "cross_conformal_freeze.json"
    conformal = read_json(conformal_path)
    selection_freeze_path = CONFIRMATION_ISSUED / "selection_freeze_record.json"
    selection_freeze = read_json(selection_freeze_path)

    frames, lockbox_evidence = load_and_validate_lockboxes()
    slices: dict[str, slice] = {}
    start = 0
    pieces = []
    for dataset in ["fixed60", "global40", "confirmation96"]:
        frame = frames[dataset].copy()
        frame["_dataset"] = dataset
        pieces.append(frame)
        slices[dataset] = slice(start, start + len(frame))
        start += len(frame)
    combined = pd.concat(pieces, ignore_index=True, sort=False)

    predictions, final_sd, model_audit = predict_all_variants(combined, protocol)
    prediction_columns: dict[str, np.ndarray] = {}
    for variant, values in predictions.items():
        for target_index, target in enumerate(TARGETS):
            prediction_columns[f"_pred_{variant}_{target}"] = values[:, target_index]
    for target_index, target in enumerate(TARGETS):
        prediction_columns[f"_pred_{target}"] = predictions["pooled_all_al"][:, target_index]
        prediction_columns[f"_std_{target}"] = final_sd[:, target_index]
    combined = pd.concat(
        [combined.reset_index(drop=True), pd.DataFrame(prediction_columns)], axis=1
    )

    confirmation_slice = slices["confirmation96"]
    issued_delta = {}
    for target_index, target in enumerate(TARGETS):
        observed = combined.iloc[confirmation_slice][f"{target}_pred"].to_numpy(dtype=float)
        recomputed = predictions["pooled_all_al"][confirmation_slice, target_index]
        issued_delta[target] = float(np.max(np.abs(observed - recomputed)))
    if any(value > 1e-8 for value in issued_delta.values()):
        raise RuntimeError(f"Confirmation issued prediction binding failed: {issued_delta}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_rows: list[dict[str, Any]] = []
    for dataset, dataset_slice in slices.items():
        frame = combined.iloc[dataset_slice].reset_index(drop=True)
        strata = [("ALL", np.ones(len(frame), dtype=bool))]
        for stratum in sorted(frame["selection_stratum"].astype(str).unique()):
            strata.append((stratum, frame["selection_stratum"].astype(str).eq(stratum).to_numpy()))
        for stratum, mask in strata:
            for variant in VARIANTS:
                for target_index, target in enumerate(TARGETS):
                    actual = frame[ACTUAL[target]].to_numpy(dtype=float)[mask]
                    predicted = predictions[variant][dataset_slice, target_index][mask]
                    metric_rows.append(
                        metric_row(
                            dataset,
                            stratum,
                            variant,
                            target,
                            actual,
                            predicted,
                            fixed_ranges[target],
                            rng,
                        )
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "model_variant_lockbox_metrics.csv", index=False, lineterminator="\n")

    conformal_pointwise, conformal_metrics = conformal_tables(
        combined,
        predictions["pooled_all_al"],
        final_sd,
        slices,
        conformal,
        fixed_ranges,
    )
    conformal_pointwise.to_csv(output / "cross_conformal_pointwise.csv", index=False, lineterminator="\n")
    conformal_metrics.to_csv(output / "cross_conformal_metrics.csv", index=False, lineterminator="\n")

    confirmation = combined.iloc[confirmation_slice].reset_index(drop=True).copy()
    confirmation_final_mu = predictions["pooled_all_al"][confirmation_slice]
    membership, front_summary, ranking, ranking_summary, anchors, pairs = confirmation_front_and_decision(
        confirmation, confirmation_final_mu, selection_freeze
    )
    membership.to_csv(output / "confirmation96_nondominated_membership.csv", index=False, lineterminator="\n")
    pairs.to_csv(output / "confirmation96_dominance_pairs.csv", index=False, lineterminator="\n")
    ranking.to_csv(output / "confirmation96_mcdm_ranking.csv", index=False, lineterminator="\n")
    anchors.to_csv(output / "confirmation96_five_anchors.csv", index=False, lineterminator="\n")

    primary = metrics.loc[
        (metrics["dataset"] == "confirmation96")
        & (metrics["stratum"] == "ALL")
        & (metrics["variant"] == "pooled_all_al")
    ].set_index("target")
    fixed_primary = metrics.loc[
        (metrics["dataset"] == "fixed60")
        & (metrics["stratum"] == "ALL")
        & (metrics["variant"] == "pooled_all_al")
    ].set_index("target")
    fixed_baseline = metrics.loc[
        (metrics["dataset"] == "fixed60")
        & (metrics["stratum"] == "ALL")
        & (metrics["variant"] == "v29_frozen_baseline")
    ].set_index("target")
    improvement_rows = []
    for target in TARGETS:
        old = float(fixed_baseline.loc[target, "rmse"])
        new = float(fixed_primary.loc[target, "rmse"])
        improvement_rows.append(
            {
                "target": target,
                "v29_baseline_rmse": old,
                "selected_v30_rmse": new,
                "rmse_reduction_pct": 100.0 * (old - new) / max(old, 1e-12),
            }
        )
    improvement = pd.DataFrame(improvement_rows)
    improvement.to_csv(output / "fixed60_v29_to_v30_improvement.csv", index=False, lineterminator="\n")

    confirmation_all = metrics.loc[
        (metrics["dataset"] == "confirmation96") & (metrics["stratum"] == "ALL")
    ]
    contribution_rows = []
    for target in TARGETS:
        target_rows = confirmation_all.loc[confirmation_all["target"] == target].set_index("variant")
        selected_rmse = float(target_rows.loc["pooled_all_al", "rmse"])
        competitors = target_rows.drop(index="pooled_all_al")
        strongest_variant = str(competitors["rmse"].idxmin())
        strongest_rmse = float(competitors.loc[strongest_variant, "rmse"])
        pooled60_rmse = float(target_rows.loc["pooled_existing60", "rmse"])
        contribution_rows.append(
            {
                "target": target,
                "selected_variant": "pooled_all_al",
                "selected_rmse": selected_rmse,
                "strongest_competing_variant": strongest_variant,
                "strongest_competing_rmse": strongest_rmse,
                "rmse_reduction_vs_strongest_competitor_pct": 100.0
                * (strongest_rmse - selected_rmse)
                / max(strongest_rmse, 1e-12),
                "pooled_existing60_rmse": pooled60_rmse,
                "rmse_reduction_vs_pooled_existing60_pct": 100.0
                * (pooled60_rmse - selected_rmse)
                / max(pooled60_rmse, 1e-12),
            }
        )
    contribution = pd.DataFrame(contribution_rows)
    contribution.to_csv(output / "confirmation96_method_contribution.csv", index=False, lineterminator="\n")

    global_all = metrics.loc[
        (metrics["dataset"] == "global40") & (metrics["stratum"] == "ALL")
    ]
    global_rows = []
    for target in TARGETS:
        target_rows = global_all.loc[global_all["target"] == target].set_index("variant")
        baseline_rmse = float(target_rows.loc["v29_frozen_baseline", "rmse"])
        selected_rmse = float(target_rows.loc["pooled_all_al", "rmse"])
        global_rows.append(
            {
                "target": target,
                "converged_subset_n": int(target_rows.loc["pooled_all_al", "n"]),
                "v29_baseline_rmse": baseline_rmse,
                "selected_v30_rmse": selected_rmse,
                "selected_v30_relative_change_vs_v29_pct": 100.0
                * (selected_rmse - baseline_rmse)
                / max(baseline_rmse, 1e-12),
                "formal_gate_assessable": False,
                "interpretation": "DESCRIPTIVE_CONVERGED_SUBSET_ONLY; 12/40 converged",
            }
        )
    global_comparison = pd.DataFrame(global_rows)
    global_comparison.to_csv(
        output / "global40_converged_subset_v29_vs_v30.csv", index=False, lineterminator="\n"
    )

    anchors_all_converged = len(anchors) == 5
    topsis_anchor = anchors.loc[anchors["forced_anchor_role"] == "TOPSIS"]
    vikor_anchor = anchors.loc[anchors["forced_anchor_role"] == "VIKOR"]
    decision_anchor_gate = bool(
        anchors_all_converged
        and not topsis_anchor.empty
        and not vikor_anchor.empty
        and topsis_anchor.iloc[0]["actual_nondominated"]
        and vikor_anchor.iloc[0]["actual_nondominated"]
        and int(topsis_anchor.iloc[0]["topsis_rank_actual"]) <= 5
        and int(vikor_anchor.iloc[0]["vikor_rank_actual"]) <= 5
    )
    confirmation_strata = metrics.loc[
        (metrics["dataset"] == "confirmation96")
        & (metrics["stratum"] != "ALL")
        & (metrics["variant"] == "pooled_all_al")
    ]
    contribution_reductions = contribution[
        "rmse_reduction_vs_strongest_competitor_pct"
    ].to_numpy(dtype=float)
    method_contribution_gate = bool(
        int(np.sum(contribution_reductions >= 15.0)) >= 2
        and int(np.sum(contribution_reductions < -5.0)) == 0
    )
    primary_gates = {
        "confirmation_convergence_ge_95_pct": lockbox_evidence["confirmation96"]["convergence_pct"] >= 95.0,
        "all_five_forced_anchors_converged": anchors_all_converged,
        "confirmation_local_range_nrmse_le_5_pct_all_targets": bool((primary["nrmse_local_range_pct"] <= 5.0).all()),
        "confirmation_fixed_range_nrmse_le_5_pct_all_targets": bool((primary["nrmse_fixed_range_pct"] <= 5.0).all()),
        "confirmation_spearman_ge_0_90_all_targets": bool((primary["spearman_rho"] >= 0.90).all()),
        "confirmation_absolute_bias_le_2_pct_fixed_range_all_targets": bool((primary["absolute_bias_pct_fixed_range"] <= 2.0).all()),
        "confirmation_p90_abs_error_le_10_pct_fixed_range_all_targets": bool((primary["p90_absolute_error_pct_fixed_range"] <= 10.0).all()),
        "confirmation_p90_abs_error_le_10_pct_fixed_range_all_strata": bool(
            (confirmation_strata["p90_absolute_error_pct_fixed_range"] <= 10.0).all()
        ),
        "confirmation_conformal_coverage_ge_90_pct_all_targets": bool(
            conformal_metrics.loc[conformal_metrics["dataset"] == "confirmation96", "coverage_ge_90_pct"].all()
        ),
        "fixed60_conformal_coverage_ge_90_pct_all_targets": bool(
            conformal_metrics.loc[conformal_metrics["dataset"] == "fixed60", "coverage_ge_90_pct"].all()
        ),
        "front_membership_jaccard_ge_0_75": bool(front_summary["jaccard_ge_0_75"]),
        "dominance_pair_accuracy_ge_90_pct_all_issued": bool(front_summary["dominance_pair_accuracy_ge_90_pct"]),
        "hypervolume_relative_error_le_5_pct": bool(front_summary["hypervolume_relative_error_le_5_pct"]),
        "updated_topsis_vikor_anchor_gate": decision_anchor_gate,
        "tested_candidate_rank_spearman_ge_0_90_both": bool(ranking_summary["rank_spearman_ge_0_90_both"]),
        "tested_candidate_top10_overlap_ge_70_pct_both": bool(ranking_summary["top10_overlap_ge_70_pct_both"]),
        "global40_convergence_ge_90_pct": lockbox_evidence["global40"]["convergence_pct"] >= 90.0,
        "global40_full_lockbox_degradation_gate_assessable_and_pass": False,
        "method_contribution_vs_strongest_competitor_gate": method_contribution_gate,
    }
    failed_gates = [name for name, passed in primary_gates.items() if not passed]
    overall_status = "PASS_ALL_PREREGISTERED_GATES" if not failed_gates else "FAIL_ONE_OR_MORE_PREREGISTERED_GATES"

    claims = pd.DataFrame(
        [
            {
                "claim": "The final V30 surrogate meets the pre-registered confirmation96 point-accuracy floor",
                "supported": bool(
                    primary_gates["confirmation_local_range_nrmse_le_5_pct_all_targets"]
                    and primary_gates["confirmation_fixed_range_nrmse_le_5_pct_all_targets"]
                ),
                "evidence": "confirmation96; all converged rows; no outlier deletion",
            },
            {
                "claim": "The post-freeze confirmation supports the updated TOPSIS and VIKOR compromise recommendations",
                "supported": decision_anchor_gate,
                "evidence": "forced anchors, actual nondominance and actual rank gate",
            },
            {
                "claim": "The V30 Pareto front is globally validated by gPROMS",
                "supported": False,
                "evidence": "Forbidden claim; sparse sampled confirmation and global40 convergence failure",
            },
            {
                "claim": "The global40 lockbox passed",
                "supported": False,
                "evidence": "12/40 converged (30%); required >=90%",
            },
            {
                "claim": "A prospectively gPROMS-confirmed surrogate-assisted Pareto-local design region is supported",
                "supported": not any(
                    name in failed_gates
                    for name in [
                        "confirmation_convergence_ge_95_pct",
                        "confirmation_local_range_nrmse_le_5_pct_all_targets",
                        "confirmation_fixed_range_nrmse_le_5_pct_all_targets",
                        "front_membership_jaccard_ge_0_75",
                    ]
                ),
                "evidence": "confirmation96 point accuracy and sampled-front fidelity; see failed gates for qualifications",
            },
        ]
    )
    claims.to_csv(output / "claim_evidence.csv", index=False, lineterminator="\n")

    summary: dict[str, Any] = {
        "schema_version": "V30-postfreeze-scientific-evaluation-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "failed_preregistered_gates": failed_gates,
        "development_stopping_decision": final_freeze["development_stopping_decision"],
        "selected_final_variant": final_freeze["selected_variant"],
        "lockboxes": lockbox_evidence,
        "confirmation96_primary_metrics": primary.reset_index().to_dict(orient="records"),
        "fixed60_selected_model_metrics": fixed_primary.reset_index().to_dict(orient="records"),
        "fixed60_v29_to_v30_improvement": improvement.to_dict(orient="records"),
        "confirmation96_method_contribution": contribution.to_dict(orient="records"),
        "method_contribution_gate_pass": method_contribution_gate,
        "global40_interpretation": "FAIL_LOW_CONVERGENCE; metrics on 12 converged rows are descriptive only and cannot satisfy the global lockbox gate",
        "global40_converged_subset_comparison": global_comparison.to_dict(orient="records"),
        "front_fidelity": front_summary,
        "decision_fidelity": ranking_summary,
        "five_anchor_gate_pass": decision_anchor_gate,
        "primary_gates": primary_gates,
        "model_prediction_audit": {
            **model_audit,
            "confirmation_issued_vs_recomputed_max_abs_delta": issued_delta,
        },
        "bootstrap": {
            "method": "paired row bootstrap percentile interval",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "provenance": {
            "protocol_config": {"path": protocol_path.relative_to(ROOT).as_posix(), "sha256": sha256(protocol_path)},
            "final_model_freeze": {"path": final_freeze_path.relative_to(ROOT).as_posix(), "sha256": sha256(final_freeze_path)},
            "cross_conformal_freeze": {"path": conformal_path.relative_to(ROOT).as_posix(), "sha256": sha256(conformal_path)},
            "confirmation_selection_freeze": {"path": selection_freeze_path.relative_to(ROOT).as_posix(), "sha256": sha256(selection_freeze_path)},
            "evaluation_code": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__).resolve())},
        },
    }
    summary["evaluation_binding_sha256"] = json_sha256(summary)
    summary_path = output / "run_summary.json"
    write_json(summary_path, summary)
    (output / "README_CN.md").write_text(
        build_report_markdown(summary, metrics, conformal_metrics),
        encoding="utf-8",
        newline="\n",
    )

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "V30-postfreeze-evaluation-artifact-manifest-1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
        },
    )
    print(json.dumps({"overall_status": overall_status, "failed_gates": failed_gates}, indent=2))


if __name__ == "__main__":
    main()
