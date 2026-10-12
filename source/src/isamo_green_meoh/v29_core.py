
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .moea_nsga3 import fast_non_dominated_sort
from .surrogate import predict_surrogates


INTEGER_OPTIMIZATION_FEATURES = ("N_eCO2R", "N_WE")


def _project_integer_features_array(X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    original = np.asarray(X, dtype=float)
    arr = original.reshape(1, -1).copy() if original.ndim == 1 else original.copy()
    integer_names = set(INTEGER_OPTIMIZATION_FEATURES)
    for index, name in enumerate(feature_names):
        if name in integer_names:
            arr[:, index] = np.rint(arr[:, index])
    return arr.reshape(original.shape) if original.ndim == 1 else arr


def _objective_decision_matrix(pareto_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, str]]]:
    specs = [
        ("carbon_utilization_pred", "Carbon utilisation (-)", "larger"),
        ("specific_energy_kwh_per_kg_meoh_pred", "SEC (kWh/kg MeOH)", "smaller"),
        ("capex_proxy_pred", "CAPEX proxy (-)", "smaller"),
    ]
    return pareto_df[[item[0] for item in specs]].astype(float).copy(), specs


def _normalised_utility_matrix(
    matrix: pd.DataFrame,
    specs: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=matrix.index)
    for column, _label, direction in specs:
        values = matrix[column].astype(float)
        low, high = float(values.min()), float(values.max())
        span = max(high - low, 1e-12)
        out[column] = (values - low) / span if direction == "larger" else (high - values) / span
    return out


def _compute_topsis_vikor(
    pareto_df: pd.DataFrame,
    *,
    weights: Optional[np.ndarray] = None,
    vikor_v: float = 0.5,
) -> pd.DataFrame:
    """Rank an already fixed Pareto archive; this function does not optimize it."""
    matrix, specs = _objective_decision_matrix(pareto_df)
    weights_arr = np.asarray(weights if weights is not None else np.ones(3) / 3.0, dtype=float)
    weights_arr /= max(float(weights_arr.sum()), 1e-12)

    utility = _normalised_utility_matrix(matrix, specs).to_numpy(dtype=float)
    denominator = np.sqrt((utility ** 2).sum(axis=0))
    weighted = utility / np.where(denominator <= 1e-12, 1.0, denominator) * weights_arr
    ideal, anti_ideal = weighted.max(axis=0), weighted.min(axis=0)
    distance_best = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    closeness = distance_worst / np.maximum(distance_best + distance_worst, 1e-12)

    regret_columns = []
    for column, _label, direction in specs:
        values = matrix[column].to_numpy(dtype=float)
        best = float(values.max() if direction == "larger" else values.min())
        worst = float(values.min() if direction == "larger" else values.max())
        span = max(abs(best - worst), 1e-12)
        regret = (best - values) / span if direction == "larger" else (values - best) / span
        regret_columns.append(np.clip(regret, 0.0, 1.0))
    weighted_regret = np.column_stack(regret_columns) * weights_arr
    s_value = weighted_regret.sum(axis=1)
    r_value = weighted_regret.max(axis=1)
    s_best, s_worst = float(s_value.min()), float(s_value.max())
    r_best, r_worst = float(r_value.min()), float(r_value.max())
    q_value = (
        float(vikor_v) * (s_value - s_best) / max(s_worst - s_best, 1e-12)
        + (1.0 - float(vikor_v)) * (r_value - r_best) / max(r_worst - r_best, 1e-12)
    )

    out = pareto_df.copy()
    out["topsis_closeness"] = closeness
    out["topsis_rank"] = pd.Series(-closeness, index=out.index).rank(method="first").astype(int)
    out["vikor_S"], out["vikor_R"], out["vikor_Q"] = s_value, r_value, q_value
    out["vikor_rank"] = pd.Series(q_value, index=out.index).rank(method="first").astype(int)
    out["mcdm_mean_rank"] = (out["topsis_rank"] + out["vikor_rank"]) / 2.0
    out["mcdm_rank_gap"] = (out["topsis_rank"] - out["vikor_rank"]).abs()
    return out.sort_values(["mcdm_mean_rank", "mcdm_rank_gap"]).copy()


def _build_optimizer_evaluate_fn(
    models,
    feature_names: Sequence[str],
    audit_log: Optional[Dict[str, List[np.ndarray]]] = None,
    uncertainty_penalty_lambda: float = 0.0,
):
    """Build the unconstrained three-objective V29 point-prediction evaluator."""
    if abs(float(uncertainty_penalty_lambda)) > 1e-12:
        raise ValueError("Frozen V29 uses point predictions; uncertainty penalties are not part of NSGA-III.")

    def evaluate_fn(X: np.ndarray) -> np.ndarray:
        projected = _project_integer_features_array(X, feature_names)
        frame = pd.DataFrame(projected, columns=feature_names)
        predictions = predict_surrogates(models, frame, list(feature_names), return_uncertainty=False)
        carbon = np.clip(predictions["carbon_utilization"].to_numpy(dtype=float), 0.0, 0.999)
        energy = np.clip(predictions["specific_energy_kwh_per_kg_meoh"].to_numpy(dtype=float), 0.0, None)
        capex = np.clip(predictions["capex_proxy"].to_numpy(dtype=float), 0.0, None)
        objectives = np.column_stack([-carbon, energy, capex])
        if audit_log is not None:
            audit_log.setdefault("X", []).append(projected.copy())
            audit_log.setdefault("F", []).append(objectives.copy())
            audit_log.setdefault("carbon_utilization_raw_pred", []).append(carbon.copy())
            audit_log.setdefault("specific_energy_kwh_per_kg_meoh_raw_pred", []).append(energy.copy())
            audit_log.setdefault("capex_proxy_raw_pred", []).append(capex.copy())
        return objectives

    return evaluate_fn


def _build_objective_df_from_F(X: np.ndarray, F: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    projected = _project_integer_features_array(X, feature_names)
    frame = pd.DataFrame(projected, columns=feature_names)
    for integer_name in INTEGER_OPTIMIZATION_FEATURES:
        if integer_name in frame:
            frame[integer_name] = frame[integer_name].round().astype(int)
    frame["carbon_utilization_pred"] = np.clip(-F[:, 0], 0.0, 0.999)
    frame["specific_energy_kwh_per_kg_meoh_pred"] = np.clip(F[:, 1], 0.0, None)
    frame["capex_proxy_pred"] = np.clip(F[:, 2], 0.0, None)
    return frame


def _deduplicate_candidate_df(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    rounded = frame[list(feature_names)].round(8)
    keep = ~rounded.duplicated(keep="first")
    return frame.loc[keep].reset_index(drop=True)


def _cumulative_pareto_archive_from_logs(
    eval_log: Dict[str, List[np.ndarray]],
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not eval_log.get("X") or not eval_log.get("F"):
        raise ValueError("The evaluation log contains no X/F batches.")
    all_evaluations = _build_objective_df_from_F(
        np.vstack(eval_log["X"]), np.vstack(eval_log["F"]), feature_names
    )
    for key, chunks in eval_log.items():
        if key in {"X", "F"} or not chunks:
            continue
        values = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks])
        if len(values) == len(all_evaluations):
            all_evaluations[key] = values
    raw_to_public = {
        "carbon_utilization_raw_pred": "carbon_utilization_pred",
        "specific_energy_kwh_per_kg_meoh_raw_pred": "specific_energy_kwh_per_kg_meoh_pred",
        "capex_proxy_raw_pred": "capex_proxy_pred",
    }
    for raw_name, public_name in raw_to_public.items():
        if raw_name in all_evaluations:
            all_evaluations[public_name] = all_evaluations[raw_name].astype(float)
    all_evaluations = _deduplicate_candidate_df(all_evaluations, feature_names)
    objectives = np.column_stack([
        -all_evaluations["carbon_utilization_pred"].to_numpy(dtype=float),
        all_evaluations["specific_energy_kwh_per_kg_meoh_pred"].to_numpy(dtype=float),
        all_evaluations["capex_proxy_pred"].to_numpy(dtype=float),
    ])
    pareto_indices = np.asarray(fast_non_dominated_sort(objectives)[0], dtype=int)
    pareto = all_evaluations.iloc[pareto_indices].reset_index(drop=True)
    return all_evaluations, pareto
