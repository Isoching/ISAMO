from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from evaluate_v30_phase0_gate import (
    load_frozen_models_with_portable_xgboost_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "handoff" / "V30_publication_pareto_local_validation"
DATA = PACKAGE / "data"
EXECUTION = ROOT / "outputs" / "V30_publication_validation_execution"
FROZEN_MODEL_DIR = (
    ROOT.parents[2] / "GitHub" / "ISAMO" / "outputs" / "surrogatemodellingresults" / "models"
)
FEATURES = [
    "P_Cap", "J_eCO2R", "T_eCO2R", "N_eCO2R",
    "P_WE", "N_WE", "P_MeOH", "T_utility",
]
TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def normalise_inputs(frame: pd.DataFrame, protocol: dict[str, object]) -> np.ndarray:
    bounds = protocol["fixed_input_bounds"]
    lo = np.array([float(bounds[name]["min"]) for name in FEATURES])
    hi = np.array([float(bounds[name]["max"]) for name in FEATURES])
    values = frame[FEATURES].to_numpy(dtype=float)
    return (values - lo) / (hi - lo)


def base_prediction_frame(models: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    from isamo_green_meoh.surrogate import predict_surrogates

    prediction = predict_surrogates(models, frame, FEATURES, return_uncertainty=True)
    return prediction.rename(
        columns={
            target: f"{target}_pred" for target in TARGETS
        } | {
            f"{target}_std": f"{target}_std_prequery" for target in TARGETS
        }
    )


def actual_targets(frame: pd.DataFrame) -> np.ndarray:
    values_by_target = []
    for target in TARGETS:
        rebuilt = f"{target}_rebuilt"
        if rebuilt in frame.columns and target in frame.columns:
            values_by_target.append(
                pd.to_numeric(frame[rebuilt], errors="coerce").fillna(
                    pd.to_numeric(frame[target], errors="coerce")
                ).to_numpy(dtype=float)
            )
        elif rebuilt in frame.columns:
            values_by_target.append(pd.to_numeric(frame[rebuilt], errors="coerce").to_numpy(dtype=float))
        else:
            values_by_target.append(pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float))
    values = np.column_stack(values_by_target)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite actual target values")
    return values


def stored_base(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = frame[[f"{target}_pred" for target in TARGETS]].to_numpy(dtype=float)
    std_columns = [f"{target}_std_prequery" for target in TARGETS]
    if set(std_columns).issubset(frame.columns):
        sd = frame[std_columns].to_numpy(dtype=float)
        sd = np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        sd = np.zeros_like(mu)
    return mu, sd


def clip_predictions(values: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, 0.999)
    clipped[:, 1:] = np.clip(clipped[:, 1:], 1e-6, None)
    return clipped


def residual_features(x_norm: np.ndarray, base_mu: np.ndarray) -> np.ndarray:
    return np.column_stack([x_norm, base_mu])


def fit_residual_models(
    features: np.ndarray, residuals: np.ndarray, seed: int
) -> list[ExtraTreesRegressor]:
    models = []
    for index in range(len(TARGETS)):
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=1.0,
            bootstrap=True,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(features, residuals[:, index])
        models.append(model)
    return models


def residual_prediction(
    models: list[ExtraTreesRegressor], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for model in models:
        tree = np.column_stack([estimator.predict(features) for estimator in model.estimators_])
        means.append(tree.mean(axis=1))
        stds.append(tree.std(axis=1, ddof=0))
    return np.column_stack(means), np.column_stack(stds)


def gate_values(reference_x: np.ndarray, query_x: np.ndarray, tau: float) -> np.ndarray:
    neighbours = NearestNeighbors(n_neighbors=min(5, len(reference_x))).fit(reference_x)
    distances = neighbours.kneighbors(query_x, return_distance=True)[0].mean(axis=1)
    return np.exp(-((distances / tau) ** 2))


def corrected_prediction(
    models: list[ExtraTreesRegressor],
    reference_x: np.ndarray,
    x_norm: np.ndarray,
    base_mu: np.ndarray,
    base_sd: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_mu, residual_sd = residual_prediction(models, residual_features(x_norm, base_mu))
    gate = gate_values(reference_x, x_norm, tau)
    mu = clip_predictions(base_mu + gate[:, None] * residual_mu)
    sd = np.sqrt(np.maximum(base_sd, 0.0) ** 2 + (gate[:, None] * residual_sd) ** 2)
    return mu, sd, gate


def metrics(actual: np.ndarray, predicted: np.ndarray, ranges: np.ndarray) -> dict[str, object]:
    rows = []
    for index, target in enumerate(TARGETS):
        rmse = math.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))
        rows.append(
            {
                "target": target,
                "rmse": rmse,
                "mae": mean_absolute_error(actual[:, index], predicted[:, index]),
                "nrmse_pct": 100.0 * rmse / ranges[index],
            }
        )
    return {"by_target": rows, "mean_fixed_range_nrmse_pct": float(np.mean([r["nrmse_pct"] for r in rows]))}


def objective_cost(mu: np.ndarray) -> np.ndarray:
    return np.column_stack([-mu[:, 0], mu[:, 1], mu[:, 2]])


def nondominated_indices(mu: np.ndarray) -> np.ndarray:
    return np.asarray(NonDominatedSorting().do(objective_cost(mu), only_non_dominated_front=True), dtype=int)


def mcdm_rank(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = cost.min(axis=0)
    hi = cost.max(axis=0)
    norm = (cost - lo) / np.maximum(hi - lo, 1e-12)
    weights = np.full(3, 1.0 / 3.0)
    weighted = norm * weights
    d_ideal = np.linalg.norm(weighted, axis=1)
    d_nadir = np.linalg.norm(weighted - weights, axis=1)
    topsis = d_nadir / np.maximum(d_ideal + d_nadir, 1e-12)
    best = norm.min(axis=0)
    worst = norm.max(axis=0)
    regret_terms = weights * (norm - best) / np.maximum(worst - best, 1e-12)
    s = regret_terms.sum(axis=1)
    r = regret_terms.max(axis=1)
    q = 0.5 * (s - s.min()) / max(s.max() - s.min(), 1e-12) + 0.5 * (
        r - r.min()
    ) / max(r.max() - r.min(), 1e-12)
    return np.argsort(-topsis), np.argsort(q)


def front_transition_metrics(
    old_ids: np.ndarray,
    old_mu: np.ndarray,
    new_ids: np.ndarray,
    new_mu: np.ndarray,
    protocol: dict[str, object],
) -> dict[str, float]:
    contract = protocol["iterative_stop"]["round_hypervolume_contract"]
    ideal = np.asarray(contract["fixed_v29_ideal"], dtype=float)
    nadir = np.asarray(contract["fixed_v29_nadir"], dtype=float)
    old_norm = (objective_cost(old_mu) - ideal) / (nadir - ideal)
    new_norm = (objective_cost(new_mu) - ideal) / (nadir - ideal)
    old_nn = NearestNeighbors(n_neighbors=1).fit(old_norm)
    new_nn = NearestNeighbors(n_neighbors=1).fit(new_norm)
    symmetric = 0.5 * (
        old_nn.kneighbors(new_norm, return_distance=True)[0].mean()
        + new_nn.kneighbors(old_norm, return_distance=True)[0].mean()
    )
    old_t, old_v = mcdm_rank(objective_cost(old_mu))
    new_t, new_v = mcdm_rank(objective_cost(new_mu))
    topsis_overlap = 100.0 * len(set(old_ids[old_t[:10]]) & set(new_ids[new_t[:10]])) / 10.0
    vikor_overlap = 100.0 * len(set(old_ids[old_v[:10]]) & set(new_ids[new_v[:10]])) / 10.0
    topsis_move = float(np.linalg.norm(new_norm[new_t[0]] - old_norm[old_t[0]]))
    vikor_move = float(np.linalg.norm(new_norm[new_v[0]] - old_norm[old_v[0]]))
    reference = np.asarray(contract["primary_reference_point"], dtype=float)
    maximum = new_norm.max(axis=0)
    reference = np.maximum(reference, np.where(maximum >= reference, 1.05 * maximum + 0.01, reference))
    hypervolume = float(HV(ref_point=reference)(new_norm))
    return {
        "fixed_reference_hypervolume": hypervolume,
        "mean_symmetric_nearest_front_distance": float(symmetric),
        "topsis_top10_overlap_pct": topsis_overlap,
        "vikor_top10_overlap_pct": vikor_overlap,
        "topsis_selected_objective_movement": topsis_move,
        "vikor_selected_objective_movement": vikor_move,
    }


def canonical_hashes(frame: pd.DataFrame) -> pd.Series:
    if "input_row_sha256" in frame.columns:
        return frame["input_row_sha256"].astype(str)
    raise ValueError("Labelled table lacks input_row_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the frozen V29 predictor with the pre-registered gated local residual model")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--al-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.round not in {1, 2, 3, 4} or len(args.al_labels) != args.round:
        raise ValueError("--round must be 1..4 and equal the number of cumulative --al-labels files")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    protocol_path = DATA / "protocol_config.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ranges = np.array([float(protocol["fixed_target_ranges"][target]) for target in TARGETS])
    candidate_path = DATA / "04_candidate_pool_76468_annotated.csv"
    holdout_path = DATA / "02_legacy_holdout_600.csv"
    existing_path = DATA / "03_existing60_discovery_round0.csv"
    stop_path = EXECUTION / "phase1_STOP_30_finalized" / "gproms_return_rebuilt_qc.csv"

    candidates = read(candidate_path)
    holdout = read(holdout_path)
    existing = read(existing_path)
    stop = read(stop_path)
    al_frames = [read(path) for path in args.al_labels]
    local = pd.concat([existing, *al_frames], ignore_index=True, sort=False)
    if canonical_hashes(local).duplicated().any():
        raise ValueError("Cumulative local development labels contain duplicate hashes")

    provenance = read(DATA / "19_existing60_retrospective_provenance.csv")
    groups_existing = existing[["record_id"]].merge(
        provenance[["record_id", "selected_source_run"]], on="record_id", how="left", validate="one_to_one"
    )["selected_source_run"].fillna("existing_unknown").astype(str)
    groups_al = []
    for round_index, frame in enumerate(al_frames, start=1):
        # The pre-issued R1 queue calls this stable candidate grouping key
        # ``source_cluster_id``; dynamically selected R2-R4 queues preserve the
        # same one-candidate-per-group identity as ``candidate_id``.
        group_column = "source_cluster_id" if "source_cluster_id" in frame else "candidate_id"
        if group_column not in frame:
            raise ValueError(
                f"AL round {round_index} labels contain neither source_cluster_id nor candidate_id"
            )
        groups_al.extend((f"R{round_index}:" + frame[group_column].astype(str)).tolist())
    groups = np.concatenate([groups_existing.to_numpy(), np.asarray(groups_al, dtype=str)])

    base_local, base_local_sd = stored_base(local)
    base_stop, base_stop_sd = stored_base(stop)
    base_candidate, base_candidate_sd = stored_base(candidates)
    actual_local = actual_targets(local)
    actual_stop = actual_targets(stop)
    x_local = normalise_inputs(local, protocol)
    x_stop = normalise_inputs(stop, protocol)
    x_candidate = normalise_inputs(candidates, protocol)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        frozen_models, recovery = load_frozen_models_with_portable_xgboost_recovery(
            FROZEN_MODEL_DIR, joblib, xgboost
        )
    holdout_base_frame = base_prediction_frame(frozen_models, holdout)
    base_holdout, base_holdout_sd = stored_base(holdout_base_frame)
    actual_holdout = actual_targets(holdout)
    x_holdout = normalise_inputs(holdout, protocol)

    residuals = actual_local - base_local
    tau_grid = [float(value) for value in protocol["model_variants"]["gated_local_residual"]["tau_grid"]]
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    oof_residual = np.zeros_like(residuals)
    oof_gate_distance_reference: list[np.ndarray] = []
    oof_indices: list[np.ndarray] = []
    for fold, (train_index, test_index) in enumerate(cv.split(x_local, groups=groups)):
        fold_models = fit_residual_models(
            residual_features(x_local[train_index], base_local[train_index]),
            residuals[train_index],
            seed=4200 + args.round * 10 + fold,
        )
        fold_mu, _ = residual_prediction(
            fold_models, residual_features(x_local[test_index], base_local[test_index])
        )
        oof_residual[test_index] = fold_mu
        neighbours = NearestNeighbors(n_neighbors=min(5, len(train_index))).fit(x_local[train_index])
        oof_gate_distance_reference.append(
            neighbours.kneighbors(x_local[test_index], return_distance=True)[0].mean(axis=1)
        )
        oof_indices.append(test_index)
    oof_distance = np.zeros(len(local))
    for indices, distances in zip(oof_indices, oof_gate_distance_reference):
        oof_distance[indices] = distances

    residual_models = fit_residual_models(
        residual_features(x_local, base_local), residuals, seed=42 + args.round
    )
    tau_rows = []
    baseline_holdout_metrics = metrics(actual_holdout, base_holdout, ranges)
    for tau in tau_grid:
        oof_gate = np.exp(-((oof_distance / tau) ** 2))
        oof_pred = clip_predictions(base_local + oof_gate[:, None] * oof_residual)
        cv_metrics = metrics(actual_local, oof_pred, ranges)
        stop_pred, _, _ = corrected_prediction(
            residual_models, x_local, x_stop, base_stop, base_stop_sd, tau
        )
        stop_metrics = metrics(actual_stop, stop_pred, ranges)
        holdout_pred, _, _ = corrected_prediction(
            residual_models, x_local, x_holdout, base_holdout, base_holdout_sd, tau
        )
        holdout_metrics = metrics(actual_holdout, holdout_pred, ranges)
        guardrail = True
        for updated, baseline in zip(holdout_metrics["by_target"], baseline_holdout_metrics["by_target"]):
            relative = 100.0 * (updated["nrmse_pct"] - baseline["nrmse_pct"]) / max(baseline["nrmse_pct"], 1e-12)
            absolute = updated["nrmse_pct"] - baseline["nrmse_pct"]
            guardrail &= relative <= 10.0 and absolute <= 0.5
        tau_rows.append(
            {
                "tau": tau,
                "grouped_cv_mean_nrmse_pct": cv_metrics["mean_fixed_range_nrmse_pct"],
                "stop_mean_nrmse_pct": stop_metrics["mean_fixed_range_nrmse_pct"],
                "legacy_guardrail_pass": bool(guardrail),
                "cv_metrics": cv_metrics,
                "stop_metrics": stop_metrics,
                "legacy_metrics": holdout_metrics,
            }
        )
    eligible = [row for row in tau_rows if row["legacy_guardrail_pass"]]
    if not eligible:
        raise RuntimeError("No tau satisfies the frozen legacy holdout guardrail")
    best_cv = min(row["grouped_cv_mean_nrmse_pct"] for row in eligible)
    finalists = [row for row in eligible if row["grouped_cv_mean_nrmse_pct"] <= 1.01 * best_cv]
    selected = min(finalists, key=lambda row: (row["stop_mean_nrmse_pct"], row["tau"]))
    tau = float(selected["tau"])

    candidate_mu, candidate_sd, candidate_gate = corrected_prediction(
        residual_models, x_local, x_candidate, base_candidate, base_candidate_sd, tau
    )
    stop_mu, _, _ = corrected_prediction(residual_models, x_local, x_stop, base_stop, base_stop_sd, tau)
    holdout_mu, _, _ = corrected_prediction(
        residual_models, x_local, x_holdout, base_holdout, base_holdout_sd, tau
    )

    old_front_idx = nondominated_indices(base_candidate)
    new_front_idx = nondominated_indices(candidate_mu)
    candidates["current_front_flag"] = False
    candidates.loc[new_front_idx, "current_front_flag"] = True
    for index, target in enumerate(TARGETS):
        candidates[f"{target}_pred"] = candidate_mu[:, index]
        candidates[f"{target}_std_prequery"] = candidate_sd[:, index]

    old_cost = objective_cost(base_candidate[old_front_idx])
    ideal = old_cost.min(axis=0)
    nadir = old_cost.max(axis=0)
    candidate_obj = (objective_cost(candidate_mu) - ideal) / np.maximum(nadir - ideal, 1e-12)
    local_obj = (objective_cost(base_local) - ideal) / np.maximum(nadir - ideal, 1e-12)
    candidate_joint = np.column_stack([x_candidate / math.sqrt(8.0), np.clip(candidate_obj, -1, 2) / math.sqrt(3.0)])
    local_joint = np.column_stack([x_local / math.sqrt(8.0), np.clip(local_obj, -1, 2) / math.sqrt(3.0)])
    nn = NearestNeighbors(n_neighbors=min(5, len(local))).fit(local_joint)
    distance, neighbour_index = nn.kneighbors(candidate_joint, return_distance=True)
    prequery_error = np.sqrt(np.mean(((base_local - actual_local) / ranges) ** 2, axis=1))
    weights = 1.0 / np.maximum(distance, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    candidates["local_error_knn_score_fixed_range"] = np.sum(
        weights * prequery_error[neighbour_index], axis=1
    )
    candidates["residual_gate_value"] = candidate_gate

    scored_path = out / f"V30_AL_round{args.round}_candidate_scored.csv"
    candidates.to_csv(scored_path, index=False, lineterminator="\n")
    front_path = out / f"V30_AL_round{args.round}_predicted_front.csv"
    candidates.loc[new_front_idx].to_csv(front_path, index=False, lineterminator="\n")

    labelled = local[["record_id", "input_row_sha256", *FEATURES]].copy()
    local_current_mu, _, _ = corrected_prediction(
        residual_models, x_local, x_local, base_local, base_local_sd, tau
    )
    for index, target in enumerate(TARGETS):
        labelled[f"{target}_pred"] = local_current_mu[:, index]
    labelled_path = out / f"V30_AL_round{args.round}_labelled_for_diversity.csv"
    labelled.to_csv(labelled_path, index=False, lineterminator="\n")

    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    model_assets = {}
    for target, model in zip(TARGETS, residual_models):
        path = model_dir / f"{target}_gated_residual.joblib"
        joblib.dump(model, path, compress=("zlib", 3))
        model_assets[target] = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
    model_set = {
        "schema_version": "V30-gated-residual-model-set-1.0.0",
        "al_round": args.round,
        "base_model_hash_set": "c7cb6e1d6d1b8d351504787d00a583c4e8e4f4f7a86a6e7909a0a74abdf8c8e8",
        "portable_recovery_binding_sha256": recovery["recovery_binding_sha256"],
        "scope": "INFERENCE_ONLY_TRAINING_RNG_STATE_NOT_RESTORED",
        "residual_contract": "ExtraTreesRegressor(500,min_samples_leaf=3,max_features=1.0,bootstrap=True,random_state=42+round)",
        "selected_tau": tau,
        "local_label_count": len(local),
        "local_label_hash_set_sha256": hashlib.sha256("\n".join(sorted(canonical_hashes(local))).encode()).hexdigest(),
        "models": model_assets,
    }
    model_set_path = out / "model_hash_set.json"
    write_json(model_set_path, model_set)

    transition = front_transition_metrics(
        candidates.iloc[old_front_idx]["candidate_id"].astype(str).to_numpy(),
        base_candidate[old_front_idx],
        candidates.iloc[new_front_idx]["candidate_id"].astype(str).to_numpy(),
        candidate_mu[new_front_idx],
        protocol,
    )
    stop_metrics = metrics(actual_stop, stop_mu, ranges)
    legacy_metrics = metrics(actual_holdout, holdout_mu, ranges)
    latest_accepted = len(al_frames[-1])
    if latest_accepted > 32:
        raise ValueError(f"latest AL label file has {latest_accepted} rows; batch size is 32")
    round_metrics = {
        "al_round": args.round,
        **transition,
        "latest_gproms_batch_convergence_pct": 100.0 * latest_accepted / 32.0,
    }
    for row in stop_metrics["by_target"]:
        round_metrics[f"stop_{row['target']}_nrmse_pct"] = row["nrmse_pct"]
        round_metrics[f"stop_{row['target']}_mae"] = row["mae"]
    for row, baseline in zip(legacy_metrics["by_target"], baseline_holdout_metrics["by_target"]):
        round_metrics[f"legacy_{row['target']}_nrmse_pct"] = row["nrmse_pct"]
        round_metrics[f"baseline_legacy_{row['target']}_nrmse_pct"] = baseline["nrmse_pct"]
    metrics_path = out / f"V30_round{args.round}_stopping_metrics.csv"
    pd.DataFrame([round_metrics]).to_csv(metrics_path, index=False, lineterminator="\n")

    audit = {
        "schema_version": "V30-gated-residual-round-update-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "al_round": args.round,
        "scientific_roles": {"development_local_labels": len(local), "stop_rows": len(stop), "legacy_holdout_rows": len(holdout)},
        "selected_tau": tau,
        "tau_selection_rule": "legacy guardrail; grouped-CV candidates within 1% of best; lowest STOP mean fixed-range NRMSE; smaller tau tie-break",
        "tau_audit": tau_rows,
        "baseline_stop_metrics": metrics(actual_stop, base_stop, ranges),
        "selected_stop_metrics": stop_metrics,
        "baseline_legacy_metrics": baseline_holdout_metrics,
        "selected_legacy_metrics": legacy_metrics,
        "candidate_scored": {"path": scored_path.relative_to(ROOT).as_posix(), "sha256": sha256(scored_path), "rows": len(candidates)},
        "predicted_front": {"path": front_path.relative_to(ROOT).as_posix(), "sha256": sha256(front_path), "rows": len(new_front_idx)},
        "labelled_for_diversity": {"path": labelled_path.relative_to(ROOT).as_posix(), "sha256": sha256(labelled_path), "rows": len(labelled)},
        "model_hash_set": {"path": model_set_path.relative_to(ROOT).as_posix(), "sha256": sha256(model_set_path)},
        "round_metrics": {"path": metrics_path.relative_to(ROOT).as_posix(), "sha256": sha256(metrics_path)},
        "fixed_lockboxes_opened": False,
    }
    audit_path = out / "round_update_audit.json"
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
