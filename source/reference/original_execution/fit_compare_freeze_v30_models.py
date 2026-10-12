from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluate_v30_phase0_gate import load_frozen_models_with_portable_xgboost_recovery
from update_v30_gated_residual_round import (
    FROZEN_MODEL_DIR,
    corrected_prediction,
    normalise_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "handoff" / "V30_publication_pareto_local_validation"
DATA = PACKAGE / "data"
EXECUTION = ROOT / "outputs" / "V30_publication_validation_execution"
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
AL_FILES = [
    EXECUTION / f"phase1_AL_R{round_index}_32_finalized" / "gproms_return_rebuilt_qc.csv"
    for round_index in range(1, 5)
]
VARIANTS = {
    "pooled_existing60": "V29 learner protocol pooled with provenance-approved existing60",
    "pooled_all_al": "pooled update with all converged AL development labels",
}


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


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def canonical_string(row: pd.Series) -> str:
    integer = {"N_eCO2R", "N_WE"}
    values = []
    for feature in FEATURES:
        value = float(row[feature])
        values.append(str(int(round(value))) if feature in integer else f"{value:.10f}")
    return "|".join(values)


def canonical_hashes(frame: pd.DataFrame) -> pd.Series:
    if "input_row_sha256" in frame:
        supplied = frame["input_row_sha256"].astype(str).str.lower()
        if supplied.str.fullmatch(r"[0-9a-f]{64}").all():
            return supplied
    strings = frame.apply(canonical_string, axis=1)
    return strings.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def actualise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for target in TARGETS:
        rebuilt = f"{target}_rebuilt"
        gproms = f"{target}_gproms"
        if rebuilt in result:
            values = pd.to_numeric(result[rebuilt], errors="coerce")
            if target in result:
                values = values.fillna(pd.to_numeric(result[target], errors="coerce"))
            result[target] = values
        elif gproms in result:
            result[target] = pd.to_numeric(result[gproms], errors="coerce")
        else:
            result[target] = pd.to_numeric(result[target], errors="coerce")
    if not np.isfinite(result[FEATURES + TARGETS].to_numpy(dtype=float)).all():
        raise ValueError("Development frame contains non-finite feature or target values")
    return result


def load_sources() -> dict[str, pd.DataFrame]:
    legacy = actualise(read(DATA / "01_legacy_development_2400.csv"))
    holdout = actualise(read(DATA / "02_legacy_holdout_600.csv"))
    existing = actualise(read(DATA / "03_existing60_discovery_round0.csv"))
    stop = actualise(
        read(EXECUTION / "phase1_STOP_30_finalized" / "gproms_return_rebuilt_qc.csv")
    )
    al = [actualise(read(path)) for path in AL_FILES]
    all_local = pd.concat([existing, *al], ignore_index=True, sort=False)
    for name, frame in {
        "legacy": legacy,
        "holdout": holdout,
        "existing": existing,
        "stop": stop,
        "all_local": all_local,
    }.items():
        hashes = canonical_hashes(frame)
        if hashes.duplicated().any():
            raise ValueError(f"{name} contains duplicate canonical hashes")
    if set(canonical_hashes(legacy)) & set(canonical_hashes(all_local)):
        raise ValueError("Legacy development and local development overlap")
    return {
        "legacy": legacy,
        "holdout": holdout,
        "existing": existing,
        "stop": stop,
        "all_local": all_local,
    }


def target_arrays(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGETS].to_numpy(dtype=float)


def metric_bundle(
    actual: np.ndarray, predicted: np.ndarray, ranges: np.ndarray
) -> dict[str, object]:
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
    return {
        "by_target": rows,
        "mean_fixed_range_nrmse_pct": float(np.mean([row["nrmse_pct"] for row in rows])),
    }


def predict_models(models: dict[str, object], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from isamo_green_meoh.surrogate import predict_surrogates

    predicted = predict_surrogates(models, frame, FEATURES, return_uncertainty=True)
    mu = predicted[TARGETS].to_numpy(dtype=float)
    sd = predicted[[f"{target}_std" for target in TARGETS]].to_numpy(dtype=float)
    return mu, np.maximum(sd, 0.0)


def load_frozen() -> tuple[dict[str, object], dict[str, object]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return load_frozen_models_with_portable_xgboost_recovery(
            FROZEN_MODEL_DIR, joblib, xgboost
        )


def fit_variant(name: str, output_root: Path) -> None:
    if name not in VARIANTS:
        raise ValueError(f"Unsupported variant: {name}")
    sources = load_sources()
    if name == "pooled_existing60":
        training = pd.concat(
            [sources["legacy"], sources["existing"]], ignore_index=True, sort=False
        )
    else:
        training = pd.concat(
            [sources["legacy"], sources["all_local"]], ignore_index=True, sort=False
        )
    hashes = canonical_hashes(training)
    if hashes.duplicated().any():
        raise ValueError("Pooled training data contain duplicate hashes")

    from isamo_green_meoh import MODEL_VERSION, __version__
    from isamo_green_meoh.config import config_manifest, default_config
    from isamo_green_meoh.surrogate import train_surrogates

    cfg = default_config()
    variant_dir = output_root / name
    model_dir = variant_dir / "models"
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models, training_metrics, oof, candidate_audit = train_surrogates(
            training, sources["holdout"], cfg
        )
    unexpected = [f"{type(item.message).__name__}: {item.message}" for item in caught]
    if unexpected:
        raise RuntimeError("Pooled V29 learner emitted warnings: " + " | ".join(unexpected[:10]))

    assets: dict[str, object] = {}
    for target in TARGETS:
        path = model_dir / f"{target}.joblib"
        joblib.dump(models[target], path, compress=("zlib", 3))
        assets[target] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    training_metrics.to_csv(
        variant_dir / "learner_training_metrics.csv", index=False, lineterminator="\n"
    )
    candidate_audit.to_csv(
        variant_dir / "learner_candidate_audit.csv", index=False, lineterminator="\n"
    )
    oof_frame = training[[*FEATURES, *TARGETS]].reset_index(drop=True).copy()
    oof_frame["input_row_sha256"] = hashes.to_numpy()
    for column in oof.columns:
        oof_frame[f"oof_{column}"] = oof[column].to_numpy()
    oof_frame.to_csv(variant_dir / "development_oof_predictions.csv", index=False, lineterminator="\n")
    contract = {
        "schema_version": "V30-pooled-V29-learner-variant-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "variant_id": name,
        "variant_description": VARIANTS[name],
        "installed_isamo_version": __version__,
        "installed_model_version": MODEL_VERSION,
        "learner_protocol": "installed V29 train_surrogates(default_config), including candidate calibration, five-member weighted heterogeneous ensemble and CU residual correction",
        "config": config_manifest(cfg),
        "training_rows": len(training),
        "training_hash_set_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode("utf-8")
        ).hexdigest(),
        "legacy_rows": len(sources["legacy"]),
        "existing60_rows": len(sources["existing"]),
        "al_rows": 0 if name == "pooled_existing60" else len(sources["all_local"]) - len(sources["existing"]),
        "models": assets,
    }
    contract["model_set_binding_sha256"] = json_sha256(contract)
    write_json(variant_dir / "variant_contract.json", contract)
    print(json.dumps(contract, indent=2, ensure_ascii=False))


def load_pooled_models(variant_dir: Path) -> dict[str, object]:
    contract_path = variant_dir / "variant_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing fitted variant contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    models = {}
    for target in TARGETS:
        path = ROOT / contract["models"][target]["path"]
        if path.stat().st_size != int(contract["models"][target]["bytes"]):
            raise ValueError(f"{target} pooled model size mismatch")
        if sha256(path) != contract["models"][target]["sha256"]:
            raise ValueError(f"{target} pooled model hash mismatch")
        models[target] = joblib.load(path)
    return models


def guardrail_status(
    updated: dict[str, object], baseline: dict[str, object]
) -> tuple[bool, list[dict[str, object]]]:
    rows = []
    passed = True
    for new, old in zip(updated["by_target"], baseline["by_target"]):
        absolute = float(new["nrmse_pct"] - old["nrmse_pct"])
        relative = float(100.0 * absolute / max(old["nrmse_pct"], 1e-12))
        target_pass = relative <= 10.0 and absolute <= 0.5
        rows.append(
            {
                "target": new["target"],
                "absolute_worsening_pp": absolute,
                "relative_worsening_pct": relative,
                "pass": bool(target_pass),
            }
        )
        passed &= target_pass
    return bool(passed), rows


def gated_prediction(
    base_models: dict[str, object], frame: pd.DataFrame, protocol: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    local = load_sources()["all_local"]
    base_local, base_local_sd = predict_models(base_models, local)
    base_query, base_query_sd = predict_models(base_models, frame)
    residual_models = [
        joblib.load(
            EXECUTION
            / "round4_model_update"
            / "models"
            / f"{target}_gated_residual.joblib"
        )
        for target in TARGETS
    ]
    tau = float(
        json.loads(
            (EXECUTION / "round4_model_update" / "model_hash_set.json").read_text(
                encoding="utf-8"
            )
        )["selected_tau"]
    )
    mu, sd, _gate = corrected_prediction(
        residual_models,
        normalise_inputs(local, protocol),
        normalise_inputs(frame, protocol),
        base_query,
        base_query_sd,
        tau,
    )
    return mu, sd


def nondominated(mu: np.ndarray) -> np.ndarray:
    cost = np.column_stack([-mu[:, 0], mu[:, 1], mu[:, 2]])
    return np.asarray(
        NonDominatedSorting().do(cost, only_non_dominated_front=True), dtype=int
    )


def finalize(output_root: Path) -> None:
    sources = load_sources()
    protocol = json.loads((DATA / "protocol_config.json").read_text(encoding="utf-8"))
    ranges = np.array([protocol["fixed_target_ranges"][target] for target in TARGETS])
    base_models, recovery = load_frozen()
    pooled_existing = load_pooled_models(output_root / "pooled_existing60")
    pooled_all = load_pooled_models(output_root / "pooled_all_al")

    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, models in {
        "v29_frozen_baseline": base_models,
        "pooled_existing60": pooled_existing,
        "pooled_all_al": pooled_all,
    }.items():
        stop_mu, stop_sd = predict_models(models, sources["stop"])
        hold_mu, hold_sd = predict_models(models, sources["holdout"])
        variants[name] = (stop_mu, stop_sd, hold_mu, hold_sd)
    gated_stop = gated_prediction(base_models, sources["stop"], protocol)
    gated_hold = gated_prediction(base_models, sources["holdout"], protocol)
    variants["gated_local_residual"] = (*gated_stop, *gated_hold)

    actual_stop = target_arrays(sources["stop"])
    actual_holdout = target_arrays(sources["holdout"])
    baseline_legacy = metric_bundle(actual_holdout, variants["v29_frozen_baseline"][2], ranges)
    rows = []
    detail: dict[str, object] = {}
    for name, (stop_mu, _stop_sd, hold_mu, _hold_sd) in variants.items():
        stop_metrics = metric_bundle(actual_stop, stop_mu, ranges)
        legacy_metrics = metric_bundle(actual_holdout, hold_mu, ranges)
        guardrail, guardrail_rows = guardrail_status(legacy_metrics, baseline_legacy)
        rows.append(
            {
                "variant_id": name,
                "stop_mean_fixed_range_nrmse_pct": stop_metrics["mean_fixed_range_nrmse_pct"],
                "legacy_mean_fixed_range_nrmse_pct": legacy_metrics["mean_fixed_range_nrmse_pct"],
                "legacy_guardrail_pass": guardrail,
            }
        )
        detail[name] = {
            "stop_metrics": stop_metrics,
            "legacy_metrics": legacy_metrics,
            "legacy_guardrail": guardrail_rows,
        }
    table = pd.DataFrame(rows).sort_values("stop_mean_fixed_range_nrmse_pct")
    eligible = table.loc[table["legacy_guardrail_pass"]].copy()
    if eligible.empty:
        raise RuntimeError("No final model variant passes the legacy global guardrail")
    best = float(eligible["stop_mean_fixed_range_nrmse_pct"].min())
    finalists = eligible.loc[
        eligible["stop_mean_fixed_range_nrmse_pct"] <= 1.01 * best
    ].copy()
    simplicity = {
        "v29_frozen_baseline": 0,
        "gated_local_residual": 1,
        "pooled_existing60": 2,
        "pooled_all_al": 3,
    }
    selected = min(finalists["variant_id"], key=lambda name: simplicity[name])
    table["within_one_pct_of_best"] = (
        table["stop_mean_fixed_range_nrmse_pct"] <= 1.01 * best
    ) & table["legacy_guardrail_pass"]
    table["selected_final_variant"] = table["variant_id"].eq(selected)

    freeze_dir = output_root / "final_freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(freeze_dir / "model_variant_comparison.csv", index=False, lineterminator="\n")

    candidates = read(DATA / "04_candidate_pool_76468_annotated.csv")
    if selected == "gated_local_residual":
        candidate_mu, candidate_sd = gated_prediction(base_models, candidates, protocol)
    else:
        selected_models = {
            "v29_frozen_baseline": base_models,
            "pooled_existing60": pooled_existing,
            "pooled_all_al": pooled_all,
        }[selected]
        candidate_mu, candidate_sd = predict_models(selected_models, candidates)
    for index, target in enumerate(TARGETS):
        candidates[f"{target}_pred"] = candidate_mu[:, index]
        candidates[f"{target}_std_prequery"] = candidate_sd[:, index]
    front_index = nondominated(candidate_mu)
    candidates["current_front_flag"] = False
    candidates.loc[front_index, "current_front_flag"] = True
    candidate_path = freeze_dir / "V30_final_candidate_scored.csv"
    front_path = freeze_dir / "V30_final_predicted_front.csv"
    candidates.to_csv(candidate_path, index=False, lineterminator="\n")
    candidates.iloc[front_index].to_csv(front_path, index=False, lineterminator="\n")

    model_assets: dict[str, object] = {}
    if selected == "gated_local_residual":
        for target in TARGETS:
            source = (
                EXECUTION
                / "round4_model_update"
                / "models"
                / f"{target}_gated_residual.joblib"
            )
            destination = freeze_dir / "models" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            model_assets[target] = {
                "path": destination.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
    elif selected.startswith("pooled_"):
        for target in TARGETS:
            source = output_root / selected / "models" / f"{target}.joblib"
            destination = freeze_dir / "models" / f"{target}.joblib"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            model_assets[target] = {
                "path": destination.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }

    local_hashes = canonical_hashes(sources["all_local"])
    selected_stop_rows = detail[selected]["stop_metrics"]["by_target"]
    stop_over_five = [
        row["target"] for row in selected_stop_rows if float(row["nrmse_pct"]) > 5.0
    ]
    if stop_over_five:
        scientific_limitation = (
            "The active-learning budget ended without the required two-transition plateau, "
            "and the selected final variant still has STOP fixed-range NRMSE above 5% for: "
            + ", ".join(stop_over_five)
            + ". This freeze permits independent evaluation but is not a success claim."
        )
    else:
        scientific_limitation = (
            "The active-learning budget ended without the required two-transition plateau. "
            "All selected-variant STOP fixed-range NRMSE development estimates are below 5%, "
            "but STOP is not independent evidence and the 5% success floor can only be decided "
            "on the post-freeze confirmation96."
        )
    freeze = {
        "schema_version": "V30-final-development-model-freeze-1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_LOCKBOX_OPENING",
        "development_stopping_decision": "BUDGET_EXHAUSTED_NO_PLATEAU",
        "scientific_limitation": scientific_limitation,
        "selection_rule": "lowest mean fixed-range STOP NRMSE subject to every legacy guardrail; variants within 1% of the best use the simpler/frozen-global tie-break",
        "selected_variant": selected,
        "selected_stop_metrics": detail[selected]["stop_metrics"],
        "selected_legacy_metrics": detail[selected]["legacy_metrics"],
        "variant_detail": detail,
        "variant_comparison_sha256": sha256(freeze_dir / "model_variant_comparison.csv"),
        "permitted_local_development_rows": len(sources["all_local"]),
        "permitted_local_hash_set_sha256": hashlib.sha256(
            "\n".join(sorted(local_hashes)).encode("utf-8")
        ).hexdigest(),
        "stop_rows_used_for_selection_never_train": len(sources["stop"]),
        "legacy_holdout_rows_used_for_guardrail_never_refit": len(sources["holdout"]),
        "fixed_region60_opened": False,
        "global40_opened": False,
        "confirmation96_opened": False,
        "frozen_v29_model_set_sha256": "c7cb6e1d6d1b8d351504787d00a583c4e8e4f4f7a86a6e7909a0a74abdf8c8e8",
        "portable_recovery_binding_sha256": recovery["recovery_binding_sha256"],
        "selected_model_assets": model_assets,
        "candidate_scored": {
            "path": candidate_path.relative_to(ROOT).as_posix(),
            "rows": len(candidates),
            "sha256": sha256(candidate_path),
        },
        "predicted_front": {
            "path": front_path.relative_to(ROOT).as_posix(),
            "rows": len(front_index),
            "sha256": sha256(front_path),
        },
        "protocol_config_sha256": sha256(DATA / "protocol_config.json"),
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
    }
    freeze["freeze_binding_sha256"] = json_sha256(freeze)
    write_json(freeze_dir / "final_model_freeze.json", freeze)
    print(json.dumps(freeze, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the two missing V30 pooled variants, compare all four variants and freeze the final development model"
    )
    parser.add_argument(
        "--action",
        choices=["fit-pooled-existing60", "fit-pooled-all-al", "finalize"],
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXECUTION / "phase3_final_model",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.action == "fit-pooled-existing60":
        fit_variant("pooled_existing60", output_root)
    elif args.action == "fit-pooled-all-al":
        fit_variant("pooled_all_al", output_root)
    else:
        finalize(output_root)


if __name__ == "__main__":
    main()
