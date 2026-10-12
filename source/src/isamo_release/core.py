from __future__ import annotations

import gc
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
FEATURES = ["P_Cap", "J_eCO2R", "T_eCO2R", "N_eCO2R", "P_WE", "N_WE", "P_MeOH", "T_utility"]
TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]
PREDICTED = [f"{target}_pred" for target in TARGETS]
INTEGER_FEATURES = {"N_eCO2R", "N_WE"}
KNOWN_LOAD_WARNING = "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\nAs an alternative, you can create a new view using np.reshape (with copy=False if needed)."


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def config(name: str = "protocol.json") -> dict:
    return json.loads((PROJECT / "config" / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    protected = [PROJECT / name for name in ["data", "freeze", "config", "reference", "src"]]
    protected.append(PROJECT.parent / "results")
    if any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("An output directory cannot replace packaged scientific assets")
    path.mkdir(parents=True, exist_ok=True)
    return path


def identities(frame: pd.DataFrame) -> pd.Series:
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input features")
    values = []
    for row in frame[FEATURES].itertuples(index=False, name=None):
        text = "|".join(str(int(round(value))) if feature in INTEGER_FEATURES else f"{value:.10f}"
                        for feature, value in zip(FEATURES, row))
        values.append(hashlib.sha256(text.encode()).hexdigest())
    return pd.Series(values, index=frame.index)


def validate_splits() -> dict:
    roles = json.loads((PROJECT / "data/dataset_roles.json").read_text())
    identities_by_role = {}
    frames = {}
    for name, metadata in roles.items():
        frame = read(PROJECT / "data" / name)
        hashes = identities(frame)
        if len(frame) != metadata["rows"] or hashes.duplicated().any():
            raise ValueError(f"Wrong row count or repeated inputs: {name}")
        if hashes.tolist() != frame["input_row_sha256"].tolist():
            raise ValueError(f"Canonical input identity differs: {name}")
        if not np.isfinite(frame[TARGETS].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite target values: {name}")
        frames[name], identities_by_role[name] = frame, set(hashes)
    training = identities_by_role["development_2586.csv"]
    for name, values in identities_by_role.items():
        if name != "development_2586.csv" and training & values:
            raise ValueError(f"Held-out rows overlap development: {name}")
    contract = json.loads((PROJECT / "freeze/variant_contract.json").read_text())
    digest = hashlib.sha256("\n".join(sorted(training)).encode()).hexdigest()
    if digest != contract["training_hash_set_sha256"]:
        raise ValueError("Development identities differ from the original V30 freeze")
    grouped = read(PROJECT / "data/cross_conformal_oof_scores.csv")
    dev = frames["development_2586.csv"]
    if dev["input_row_sha256"].tolist() != grouped["input_row_sha256"].tolist():
        raise ValueError("Development order differs from frozen cross-conformal scores")
    if dev["conformal_group_id"].tolist() != grouped["conformal_group_id"].tolist():
        raise ValueError("Cross-conformal source/cluster groups differ")
    return {"rows": {k: len(v) for k, v in frames.items()}, "training_hash_set_sha256": digest,
            "training_vs_all_validation_overlap": 0, "conformal_unique_groups": dev["conformal_group_id"].nunique()}


def model_directory(path: Path | None) -> Path:
    candidates = [path] if path else [PROJECT / "models", PROJECT.parent / "results/v30_model_freeze/models"]
    for candidate in candidates:
        if candidate and all((candidate / f"{t}.joblib").is_file() for t in TARGETS):
            return candidate.resolve()
    raise FileNotFoundError("Supply --models PATH to the frozen models, or run the train command first")


def verify_models(path: Path) -> dict:
    frozen = json.loads((PROJECT / "freeze/final_model_freeze.json").read_text())
    verified = {}
    for target, entry in frozen["selected_model_assets"].items():
        f = path / f"{target}.joblib"
        if f.stat().st_size != entry["bytes"] or sha256(f) != entry["sha256"]:
            raise ValueError(f"Frozen model bytes differ: {target}")
        verified[target] = entry["sha256"]
    return verified


def load_model(path: Path, target: str):
    import joblib
    # Preserve the original pickle module names and report any unfamiliar warning.
    from isamo_green_meoh import surrogate  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path / f"{target}.joblib")
    unexpected = [str(w.message) for w in caught if not
                  (issubclass(w.category, DeprecationWarning) and str(w.message) == KNOWN_LOAD_WARNING)]
    if unexpected:
        raise RuntimeError(f"Unexpected model-load warnings: {unexpected[:3]}")
    return model


def predict(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    from isamo_green_meoh.surrogate import predict_surrogates

    outputs = []
    for target in TARGETS:
        model = load_model(path, target)
        outputs.append(predict_surrogates({target: model}, frame, FEATURES, return_uncertainty=True))
        del model
        gc.collect()
    return pd.concat(outputs, axis=1)


def verify_inference(path: Path) -> dict:
    verify_models(path)
    sample = read(PROJECT / "data/frozen_prediction_sample.csv")
    predictions = predict(path, sample)
    checks = {}
    for target in TARGETS:
        for column, reference in [(target, f"{target}_pred"), (f"{target}_std", f"{target}_std_prequery")]:
            actual, expected = predictions[column].to_numpy(), sample[reference].to_numpy()
            if not np.allclose(actual, expected, rtol=0, atol=1e-10):
                raise ValueError(f"Frozen inference differs: {column}")
            checks[column] = float(np.abs(actual - expected).max())
    return {"rows": len(sample), "max_absolute_errors": checks}
