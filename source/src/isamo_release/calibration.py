from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .analysis import finite_sample_quantile
from .core import FEATURES, PROJECT, TARGETS, config, load_model, read, sha256, validate_splits, write_json


def calibrate(model_dir: Path, output: Path) -> None:
    """Port of the frozen five-fold grouped calibration; keep the original fold/seed/SD rules."""
    from isamo_green_meoh.surrogate import _fit_model, _inverse_target_prediction, _target_internal_space

    validate_splits()
    development = read(PROJECT / "data/development_2586.csv")
    x = development[FEATURES].to_numpy(dtype=float)
    groups = development["conformal_group_id"].to_numpy(dtype=str)
    folds = list(GroupKFold(n_splits=5).split(x, groups=groups))
    scores = development[["input_row_sha256", "conformal_group_id", "development_source"]].copy()
    fold_id = np.full(len(x), -1, dtype=int)
    for index, (_, validation) in enumerate(folds):
        fold_id[validation] = index
    scores["conformal_fold"] = fold_id
    quantiles = {}
    for target_index, target in enumerate(TARGETS):
        entry = load_model(model_dir, target)
        weights = np.asarray(entry["weights"], dtype=float)
        weights /= weights.sum()
        y, _, space = _target_internal_space(target, development, development)
        mean_internal, sd_internal = np.zeros(len(x)), np.zeros(len(x))
        for fold_index, (train, validation) in enumerate(folds):
            trained = []
            for member_index, prototype in enumerate(entry["members"]):
                trained.append(_fit_model(clone(prototype), x[train], y[train],
                    seed=20260826 + 1000 * target_index + 100 * fold_index + member_index, use_early_stop=True))
            train_matrix = np.column_stack([m.predict(x[train]) for m in trained])
            validation_matrix = np.column_stack([m.predict(x[validation]) for m in trained])
            train_mean, validation_mean = train_matrix @ weights, validation_matrix @ weights
            residual = entry.get("residual_model")
            residual_scale = float(entry.get("residual_scale", 0.0))
            if residual is not None and residual_scale > 0:
                fitted = clone(residual).fit(x[train], y[train] - train_mean)
                validation_mean += residual_scale * np.asarray(fitted.predict(x[validation]), dtype=float)
            # Match the historical cross-conformal definition, including CU residual correction.
            validation_sd = np.sqrt(np.maximum(((validation_matrix - validation_mean[:, None])**2) @ weights, 0.0))
            mean_internal[validation], sd_internal[validation] = validation_mean, validation_sd
            del trained
            gc.collect()
            print(f"Calibrated {target}, fold {fold_index + 1}/5", flush=True)
        mean, sd = _inverse_target_prediction(target, mean_internal, sd_internal, space)
        mean = np.clip(mean, 0, 0.999) if target == "carbon_utilization" else np.clip(mean, 1e-6, None)
        sd = np.maximum(sd, 0)
        floor = float(config()["fixed_target_ranges"][target]) * 1e-6
        actual = development[target].to_numpy()
        score = np.abs(actual - mean) / np.maximum(sd, floor)
        quantiles[target] = {"nominal_coverage": 0.95, "finite_sample_quantile": finite_sample_quantile(score),
                             "scale_floor": floor, "calibration_rows": len(x)}
        for suffix, value in [("actual", actual), ("oof_pred", mean), ("oof_sd", sd), ("scaled_abs_residual", score)]:
            scores[f"{target}_{suffix}"] = value
        del entry
        gc.collect()
    scores.to_csv(output / "cross_conformal_oof_scores.csv", index=False)
    write_json(output / "cross_conformal_calibration.json", {"status": "NEW_CALIBRATION_NOT_HISTORICAL_FREEZE",
               "method": "five-fold source/cluster-grouped cross-conformal scaled by ensemble SD", "calibration_rows": len(x),
               "unique_groups": len(set(groups)), "held_out_rows_used": False, "quantiles": quantiles,
               "model_sha256": {t: sha256(model_dir / f"{t}.joblib") for t in TARGETS}})
