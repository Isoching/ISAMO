from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fit_compare_freeze_v30_models import (  # noqa: E402
    AL_FILES,
    DATA,
    EXECUTION,
    FEATURES,
    TARGETS,
    actualise,
    canonical_hashes,
    read,
    sha256,
    write_json,
)


def json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def build_development_and_groups() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    legacy = actualise(read(DATA / "01_legacy_development_2400.csv"))
    existing = actualise(read(DATA / "03_existing60_discovery_round0.csv"))
    provenance = read(DATA / "19_existing60_retrospective_provenance.csv")
    existing_groups = existing[["record_id"]].merge(
        provenance[["record_id", "selected_source_run"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )["selected_source_run"].fillna("existing_unknown")
    al_frames = [actualise(read(path)) for path in AL_FILES]
    development = pd.concat([legacy, existing, *al_frames], ignore_index=True, sort=False)

    groups: list[str] = [f"legacy:{value}" for value in legacy["sample_id"].astype(str)]
    groups.extend(f"existing:{value}" for value in existing_groups.astype(str))
    for round_index, frame in enumerate(al_frames, start=1):
        group_column = "source_cluster_id" if "source_cluster_id" in frame else "candidate_id"
        groups.extend(f"R{round_index}:{value}" for value in frame[group_column].astype(str))
    group_array = np.asarray(groups, dtype=str)
    if len(group_array) != len(development):
        raise RuntimeError("Grouped calibration identity count differs from development rows")

    hashes = canonical_hashes(development)
    if hashes.duplicated().any():
        raise ValueError("Conformal development contains duplicate canonical hashes")
    stop = actualise(
        read(EXECUTION / "phase1_STOP_30_finalized" / "gproms_return_rebuilt_qc.csv")
    )
    if set(hashes) & set(canonical_hashes(stop)):
        raise ValueError("STOP_30 leaked into cross-conformal development")

    identity = pd.DataFrame(
        {
            "row_index": np.arange(len(development), dtype=int),
            "input_row_sha256": hashes,
            "conformal_group_id": group_array,
            "development_source": ["legacy2400"] * len(legacy)
            + ["existing60"] * len(existing)
            + sum(
                ([f"AL_R{round_index}"] * len(frame) for round_index, frame in enumerate(al_frames, start=1)),
                [],
            ),
        }
    )
    return development, group_array, identity


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> tuple[float, float]:
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * coverage))
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    return float(sorted_scores[rank - 1]), float(rank / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze five-fold source/cluster-grouped cross-conformal calibration for the selected V30 model"
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=EXECUTION / "phase3_final_model" / "final_freeze",
    )
    args = parser.parse_args()
    freeze_dir = args.freeze_dir.resolve()
    freeze_path = freeze_dir / "final_model_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_LOCKBOX_OPENING":
        raise ValueError("Final model is not in the pre-lockbox frozen state")
    if freeze["selected_variant"] != "pooled_all_al":
        raise ValueError(
            "This calibrator is bound to the selected pooled_all_al learner; regenerate explicitly for another selected variant"
        )
    if any(
        bool(freeze[key])
        for key in ["fixed_region60_opened", "global40_opened", "confirmation96_opened"]
    ):
        raise ValueError("A final lockbox was already opened before conformal freeze")

    models = {
        target: joblib.load(freeze_dir / "models" / f"{target}.joblib")
        for target in TARGETS
    }
    development, groups, identity = build_development_and_groups()
    X = development[FEATURES].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(X, groups=groups))
    fold_id = np.full(len(development), -1, dtype=int)
    for index, (_train, validation) in enumerate(folds):
        fold_id[validation] = index
    if (fold_id < 0).any():
        raise RuntimeError("Cross-conformal folds do not cover all development rows")
    identity["conformal_fold"] = fold_id

    from isamo_green_meoh.surrogate import (
        _fit_model,
        _inverse_target_prediction,
        _target_internal_space,
    )

    result = identity.copy()
    score_columns = []
    quantiles: dict[str, object] = {}
    for target_index, target in enumerate(TARGETS):
        entry = models[target]
        prototypes = entry["members"]
        weights = np.asarray(entry["weights"], dtype=float)
        weights /= weights.sum()
        y_internal, _unused, space = _target_internal_space(target, development, development)
        actual = development[target].to_numpy(dtype=float)
        oof_internal = np.zeros(len(development), dtype=float)
        oof_sd_internal = np.zeros(len(development), dtype=float)
        for fold_index, (train_index, validation_index) in enumerate(folds):
            trained = []
            for member_index, prototype in enumerate(prototypes):
                member = clone(prototype)
                member = _fit_model(
                    member,
                    X[train_index],
                    y_internal[train_index],
                    seed=20260826 + 1000 * target_index + 100 * fold_index + member_index,
                    use_early_stop=True,
                )
                trained.append(member)
            train_matrix = np.column_stack(
                [member.predict(X[train_index]) for member in trained]
            )
            validation_matrix = np.column_stack(
                [member.predict(X[validation_index]) for member in trained]
            )
            train_mean = train_matrix @ weights
            validation_mean = validation_matrix @ weights
            residual_prototype = entry.get("residual_model")
            residual_scale = float(entry.get("residual_scale", 0.0))
            if residual_prototype is not None and residual_scale > 0:
                residual_model = clone(residual_prototype)
                residual_model.fit(
                    X[train_index], y_internal[train_index] - train_mean
                )
                validation_mean = validation_mean + residual_scale * np.asarray(
                    residual_model.predict(X[validation_index]), dtype=float
                )
            validation_sd = np.sqrt(
                np.maximum(
                    ((validation_matrix - validation_mean[:, None]) ** 2) @ weights,
                    0.0,
                )
            )
            oof_internal[validation_index] = validation_mean
            oof_sd_internal[validation_index] = validation_sd
        oof, oof_sd = _inverse_target_prediction(
            target, oof_internal, oof_sd_internal, space
        )
        if target == "carbon_utilization":
            oof = np.clip(oof, 0.0, 0.999)
        else:
            oof = np.clip(oof, 1e-6, None)
        oof_sd = np.maximum(oof_sd, 0.0)
        target_range = float(
            json.loads((DATA / "protocol_config.json").read_text(encoding="utf-8"))[
                "fixed_target_ranges"
            ][target]
        )
        scale_floor = target_range * 1e-6
        scale = np.maximum(oof_sd, scale_floor)
        score = np.abs(actual - oof) / scale
        quantile, effective_probability = finite_sample_quantile(score, 0.95)
        result[f"{target}_actual"] = actual
        result[f"{target}_oof_pred"] = oof
        result[f"{target}_oof_sd"] = oof_sd
        result[f"{target}_scale_floor"] = scale_floor
        result[f"{target}_scaled_abs_residual"] = score
        score_columns.append(f"{target}_scaled_abs_residual")
        quantiles[target] = {
            "nominal_coverage": 0.95,
            "finite_sample_quantile": quantile,
            "effective_order_statistic_probability": effective_probability,
            "scale_floor": scale_floor,
            "calibration_rows": len(score),
        }

    output_dir = freeze_dir / "conformal"
    output_dir.mkdir(exist_ok=True)
    score_path = output_dir / "cross_conformal_oof_scores.csv"
    result.to_csv(score_path, index=False, lineterminator="\n")
    protocol_path = DATA / "protocol_config.json"
    calibration = {
        "schema_version": "V30-cross-conformal-freeze-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_LOCKBOX_OPENING",
        "final_model_freeze_sha256": sha256(freeze_path),
        "final_model_freeze_binding_sha256": freeze["freeze_binding_sha256"],
        "selected_variant": freeze["selected_variant"],
        "method": "five-fold source/cluster-grouped cross-conformal, absolute residual scaled by selected ensemble SD",
        "grouping": "legacy Sobol rows are independent sample_id groups; existing60 share selected_source_run groups; each AL candidate retains its round-qualified cluster/candidate group",
        "outer_fold_training": "clone the final selected five ensemble members and weights in each outer group fold; fit the selected CU residual-model architecture on outer-training ensemble residuals only",
        "scale_floor_rule": "fixed target range multiplied by 1e-6; chosen and frozen before any final lockbox label",
        "finite_sample_quantile_rule": "order statistic ceil((n+1)*0.95), capped at n",
        "calibration_rows": len(result),
        "unique_groups": int(pd.Series(groups).nunique()),
        "fold_rows": {
            str(index): int((fold_id == index).sum()) for index in range(5)
        },
        "stop30_used": False,
        "fixed_region60_used": False,
        "global40_used": False,
        "confirmation96_used": False,
        "quantiles": quantiles,
        "score_file": {
            "path": score_path.relative_to(ROOT).as_posix(),
            "bytes": score_path.stat().st_size,
            "sha256": sha256(score_path),
        },
        "score_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(result[score_columns].to_numpy(dtype="<f8")).tobytes()
        ).hexdigest(),
        "protocol_config_sha256": sha256(protocol_path),
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
    }
    calibration["calibration_binding_sha256"] = json_sha256(calibration)
    output_path = output_dir / "cross_conformal_freeze.json"
    write_json(output_path, calibration)
    print(json.dumps(calibration, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
