from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


EVALUATOR_VERSION = "V30-phase0-gate-evaluator-1.0.1"
MANIFEST_PATTERN = re.compile(r"^([0-9a-fA-F]{64})  ([^\r\n]+)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

FEATURES = (
    "P_Cap",
    "J_eCO2R",
    "T_eCO2R",
    "N_eCO2R",
    "P_WE",
    "N_WE",
    "P_MeOH",
    "T_utility",
)
INTEGER_FEATURES = {"N_eCO2R", "N_WE"}
TARGETS = (
    "carbon_utilization",
    "specific_energy_kwh_per_kg_meoh",
    "capex_proxy",
)
EXPECTED_MODEL_ASSETS: dict[str, dict[str, Any]] = {
    "carbon_utilization.joblib": {
        "size_bytes": 319_228_148,
        "sha256": "d524b51e38c58a6f8137da5cb3a7fa5e227d6a8898d393e516c17a80f3a4945b",
    },
    "specific_energy_kwh_per_kg_meoh.joblib": {
        "size_bytes": 391_757_204,
        "sha256": "98bfaad8314a04252865255642439746468eec9e7828999fa9622046c5b85a1d",
    },
    "capex_proxy.joblib": {
        "size_bytes": 109_158_110,
        "sha256": "0b768a5b74745fd84ce8ed5113de7c9ccb61f2b9b1621834aecbb5309f0abe98",
    },
}
EXPECTED_MODEL_SET_SHA256 = "c7cb6e1d6d1b8d351504787d00a583c4e8e4f4f7a86a6e7909a0a74abdf8c8e8"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_CORE_PACKAGES = {
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
    "xgboost": "3.3.0",
    "joblib": "1.5.3",
}
EXPECTED_WORKFLOW_PACKAGES = {
    "scipy": "1.18.1",
    "matplotlib": "3.11.1",
    "openpyxl": "3.1.5",
    "xlrd": "2.0.2",
    "pymoo": "0.6.2",
    "lightgbm": "4.7.0",
    "Pillow": "12.3.0",
    "shap": "0.52.0",
    "pytest": "9.1.1",
    "isamo-green-meoh": "29.0.0",
}
EXPECTED_GPROMS_PROJECT_SHA256 = "b7e0ed3d3e35129b7b3a3d729fd5e80427dd67f6717f415b465eac89c11e4a48"
EXPECTED_CANDIDATE_ROWS = 76_468
EXPECTED_PREDICTION_CODE_LF_SHA256 = {
    "__init__.py": "e4fa6da49557e35835fdf6b864ebdfe4b19934daaa41c5017979f5302923765c",
    "config.py": "821d2ff1ce758d79317c299a96d3f53a3ac583e6443738d94df7f0b6007d5484",
    "surrogate.py": "7d89402015c1abb25e3e6c05b50c0ab2dfe61dee460a2e46cb62684f2ca495c0",
}
EXPECTED_XGBOOST_BOOSTER_COUNTS = {
    "carbon_utilization": 3,
    "specific_energy_kwh_per_kg_meoh": 2,
    "capex_proxy": 4,
}
KNOWN_JOBLIB_NUMPY_SHAPE_DEPRECATION = (
    "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n"
    "As an alternative, you can create a new view using np.reshape "
    "(with copy=False if needed)."
)

ROLE_SPECS: tuple[tuple[str, str, int], ...] = (
    ("legacy_development", "01_legacy_development_2400.csv", 2400),
    ("legacy_holdout", "02_legacy_holdout_600.csv", 600),
    ("existing60", "03_existing60_discovery_round0.csv", 60),
    ("al_round1", "06_execute_now_AL_R1_32.csv", 32),
    ("stop", "07_execute_now_STOP_30.csv", 30),
    ("fixed_v29_region_test", "08_LOCKED_fixed_V29_region_60.csv", 60),
    ("global_final", "09_LOCKED_global_final_40.csv", 40),
)
QUEUE_PATHS = (
    "data/06_execute_now_AL_R1_32.csv",
    "data/07_execute_now_STOP_30.csv",
    "data/08_LOCKED_fixed_V29_region_60.csv",
    "data/09_LOCKED_global_final_40.csv",
)

GPROMS_INPUT_COLUMNS = (
    "Current_density_eCO2R_Acm2",
    "WE_Energy_input_rate_KW",
    "Number_of_cells_stacked_in_series_WE",
    "Number_of_stacks_eCO2R",
    "Outlet_pressure",
    "Outlet_pressure_cap",
    "T_eCO2R_gproms_input",
    "Utility_constant_temperature",
)
RAW_OUTPUT_COLUMNS = (
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
    "Component_molar_flowrate_carbon_dioxide_CAP",
    "Component_molar_flowrate_carbon_monoxide_CE",
    "Component_molar_flowrate_hydrogen_WE",
    "Component_molar_flowrate_hydrogen_CE",
    "Component_molar_flowrate_carbon_dioxide_CE",
    "Component_molar_flowrate_methanol",
)
ECHO_COLUMNS = (
    "echo_Current_density_specification_1",
    "echo_Energy_input_rate",
    "echo_Number_of_cells_stacked_in_series",
    "echo_Number_of_stacks",
    "echo_Outlet_pressure",
    "echo_Outlet_pressure_cap",
    "echo_Temperature",
    "echo_Utility_constant_temperature",
)
PROVENANCE_COLUMNS = (
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
)
REQUIRED_RUNNER_INPUT_COLUMNS = (
    "record_id",
    "dataset_role",
    "input_row_sha256",
    *FEATURES,
    *GPROMS_INPUT_COLUMNS,
)
REQUIRED_RUNNER_OUTPUT_COLUMNS = (
    *REQUIRED_RUNNER_INPUT_COLUMNS,
    "selected_final_result",
    *RAW_OUTPUT_COLUMNS,
    *ECHO_COLUMNS,
    *PROVENANCE_COLUMNS,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lf_normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


_UBJSON_INTEGER_MARKERS = {
    ord("i"): (1, True),
    ord("U"): (1, False),
    ord("I"): (2, True),
    ord("l"): (4, True),
    ord("L"): (8, True),
}
_UBJSON_FLOAT_MARKERS = {ord("d"): 4, ord("D"): 8}
_UBJSON_SIMPLE_MARKERS = {ord("Z"), ord("N"), ord("T"), ord("F")}


def _read_ubjson_integer(data: memoryview, position: int) -> tuple[int, int]:
    if position >= len(data):
        raise ValueError("truncated UBJSON integer")
    marker = data[position]
    position += 1
    if marker not in _UBJSON_INTEGER_MARKERS:
        raise ValueError(f"invalid UBJSON integer marker: {marker!r}")
    width, signed = _UBJSON_INTEGER_MARKERS[marker]
    end = position + width
    if end > len(data):
        raise ValueError("truncated UBJSON integer payload")
    value = int.from_bytes(data[position:end], "big", signed=signed)
    return value, end


def _skip_ubjson_value(
    data: memoryview, position: int, forced_marker: int | None = None
) -> int:
    if forced_marker is None:
        if position >= len(data):
            raise ValueError("truncated UBJSON value")
        marker = data[position]
        position += 1
    else:
        marker = forced_marker
    if marker in _UBJSON_SIMPLE_MARKERS:
        return position
    if marker in _UBJSON_INTEGER_MARKERS:
        end = position + _UBJSON_INTEGER_MARKERS[marker][0]
        if end > len(data):
            raise ValueError("truncated UBJSON numeric payload")
        return end
    if marker in _UBJSON_FLOAT_MARKERS:
        end = position + _UBJSON_FLOAT_MARKERS[marker]
        if end > len(data):
            raise ValueError("truncated UBJSON floating payload")
        return end
    if marker == ord("C"):
        if position >= len(data):
            raise ValueError("truncated UBJSON character")
        return position + 1
    if marker in {ord("S"), ord("H")}:
        length, position = _read_ubjson_integer(data, position)
        if length < 0 or position + length > len(data):
            raise ValueError("invalid UBJSON string length")
        return position + length
    if marker not in {ord("["), ord("{")}:
        raise ValueError(f"unsupported UBJSON marker: {marker!r}")

    close_marker = ord("]") if marker == ord("[") else ord("}")
    child_marker: int | None = None
    count: int | None = None
    if position < len(data) and data[position] == ord("$"):
        position += 1
        if position >= len(data):
            raise ValueError("truncated UBJSON optimized type")
        child_marker = data[position]
        position += 1
    if position < len(data) and data[position] == ord("#"):
        position += 1
        count, position = _read_ubjson_integer(data, position)
        if count < 0:
            raise ValueError("negative UBJSON container count")

    completed = 0
    while count is None or completed < count:
        if position >= len(data):
            raise ValueError("unterminated UBJSON container")
        if count is None and data[position] == close_marker:
            return position + 1
        if marker == ord("{"):
            key_length, position = _read_ubjson_integer(data, position)
            if key_length < 0 or position + key_length > len(data):
                raise ValueError("invalid UBJSON object key length")
            position += key_length
        position = _skip_ubjson_value(data, position, child_marker)
        completed += 1
    if position < len(data) and data[position] == close_marker:
        position += 1
    return position


def _ubjson_object_entries(raw: bytes | bytearray) -> dict[str, bytes]:
    data = memoryview(raw)
    if not data or data[0] != ord("{"):
        raise ValueError("UBJSON value is not an object")
    position = 1
    entries: dict[str, bytes] = {}
    while True:
        if position >= len(data):
            raise ValueError("unterminated UBJSON object")
        if data[position] == ord("}"):
            position += 1
            break
        key_length, position = _read_ubjson_integer(data, position)
        if key_length < 0 or position + key_length > len(data):
            raise ValueError("invalid UBJSON object key")
        key = bytes(data[position : position + key_length]).decode("utf-8")
        position += key_length
        if key in entries:
            raise ValueError(f"duplicate UBJSON object key: {key}")
        start = position
        position = _skip_ubjson_value(data, position)
        entries[key] = bytes(data[start:position])
    if position != len(data):
        raise ValueError("UBJSON object has trailing bytes")
    return entries


def _parse_ubjson_integer_array(raw: bytes) -> tuple[int, ...]:
    data = memoryview(raw)
    if not data or data[0] != ord("["):
        raise ValueError("UBJSON value is not an array")
    position = 1
    forced_marker: int | None = None
    if position < len(data) and data[position] == ord("$"):
        position += 1
        forced_marker = data[position]
        position += 1
    if position >= len(data) or data[position] != ord("#"):
        raise ValueError("UBJSON version array must be count-delimited")
    position += 1
    count, position = _read_ubjson_integer(data, position)
    values: list[int] = []
    for _ in range(count):
        if forced_marker is None:
            value, position = _read_ubjson_integer(data, position)
        else:
            width, signed = _UBJSON_INTEGER_MARKERS[forced_marker]
            end = position + width
            if end > len(data):
                raise ValueError("truncated optimized UBJSON integer array")
            value = int.from_bytes(data[position:end], "big", signed=signed)
            position = end
        values.append(value)
    if position < len(data) and data[position] == ord("]"):
        position += 1
    if position != len(data):
        raise ValueError("UBJSON version array has trailing bytes")
    return tuple(values)


def extract_xgboost_stable_model_slice(snapshot: bytes | bytearray) -> tuple[bytes, dict[str, Any]]:
    snapshot_bytes = bytes(snapshot)
    envelope = _ubjson_object_entries(snapshot_bytes)
    if set(envelope) != {"Config", "Model"}:
        raise ValueError("XGBoost snapshot envelope must contain exactly Config and Model")
    config_entries = _ubjson_object_entries(envelope["Config"])
    if set(config_entries) != {"learner", "version"}:
        raise ValueError("XGBoost snapshot Config must contain exactly learner and version")
    config_version = _parse_ubjson_integer_array(config_entries["version"])
    model = envelope["Model"]
    model_entries = _ubjson_object_entries(model)
    if set(model_entries) != {"learner", "version"}:
        raise ValueError("XGBoost stable Model must contain exactly learner and version")
    version = _parse_ubjson_integer_array(model_entries["version"])
    if config_version != (3, 3, 0) or version != (3, 3, 0):
        raise ValueError(
            "unexpected XGBoost snapshot versions: "
            f"Config={config_version!r}, Model={version!r}"
        )
    evidence = {
        "snapshot_bytes": len(snapshot_bytes),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "config_bytes": len(envelope["Config"]),
        "config_sha256": hashlib.sha256(envelope["Config"]).hexdigest(),
        "config_version": list(config_version),
        "model_bytes": len(model),
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "model_version": list(version),
        "extraction": "EXACT_RAW_UBJSON_MODEL_VALUE_SLICE_NO_REENCODING",
        "scope": "INFERENCE_ONLY_TRAINING_RNG_STATE_NOT_RESTORED",
    }
    return model, evidence


def load_frozen_models_with_portable_xgboost_recovery(
    model_dir: Path, joblib_module: Any, xgboost_module: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_setstate = xgboost_module.core.Booster.__setstate__
    records: list[dict[str, Any]] = []
    current_target: str | None = None

    def portable_setstate(booster: Any, state: dict[str, Any]) -> None:
        copied = dict(state)
        raw = copied.get("handle")
        if raw is not None:
            model, record = extract_xgboost_stable_model_slice(raw)
            proxy = xgboost_module.Booster()
            proxy.load_model(bytearray(model))
            copied["handle"] = proxy.handle
            proxy.handle = None
            record["target"] = current_target
            record["target_booster_index"] = 1 + sum(
                item["target"] == current_target for item in records
            )
            records.append(record)
        booster.__dict__.update(copied)

    models: dict[str, Any] = {}
    xgboost_module.core.Booster.__setstate__ = portable_setstate
    try:
        for target in TARGETS:
            current_target = target
            models[target] = joblib_module.load(model_dir / f"{target}.joblib")
    finally:
        xgboost_module.core.Booster.__setstate__ = original_setstate

    counts = {target: sum(item["target"] == target for item in records) for target in TARGETS}
    if counts != EXPECTED_XGBOOST_BOOSTER_COUNTS:
        raise ValueError(
            f"recovered XGBoost Booster counts differ from frozen contract: {counts!r}"
        )
    recovery_binding = {
        "mode": "XGBOOST_3_3_0_EXACT_STABLE_MODEL_SLICE_RECOVERY",
        "booster_counts": counts,
        "boosters": records,
    }
    return models, {
        **recovery_binding,
        "recovery_binding_sha256": json_sha256(recovery_binding),
    }


def summarize_model_warnings(caught: Sequence[warnings.WarningMessage]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], int] = {}
    for item in caught:
        key = (
            item.category.__name__,
            str(item.message),
            Path(item.filename).name,
            int(item.lineno),
        )
        grouped[key] = grouped.get(key, 0) + 1
    details: list[dict[str, Any]] = []
    actionable = 0
    acknowledged = 0
    for (category, message, filename, line), count in sorted(grouped.items()):
        is_acknowledged = (
            category == "DeprecationWarning"
            and message == KNOWN_JOBLIB_NUMPY_SHAPE_DEPRECATION
            and filename == "numpy_pickle.py"
            and line == 207
        )
        if is_acknowledged:
            acknowledged += count
        else:
            actionable += count
        details.append(
            {
                "category": category,
                "message": message,
                "filename": filename,
                "line": line,
                "count": count,
                "classification": (
                    "ACKNOWLEDGED_JOBLIB_NUMPY_2_5_DEPRECATION"
                    if is_acknowledged
                    else "ACTIONABLE"
                ),
            }
        )
    return {
        "total": len(caught),
        "unique": len(details),
        "acknowledged_nonfatal": acknowledged,
        "actionable": actionable,
        "details": details,
    }


def model_set_sha256(raw_hashes: Mapping[str, str]) -> str:
    # This deliberately reproduces prepare_v30_publication_validation_handoff.load_models().
    payload = json.dumps(dict(raw_hashes), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gate_result(status: str, problems: Sequence[str], **evidence: Any) -> dict[str, Any]:
    return {"status": status, "problems": list(problems), **evidence}


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def has_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    for item in (candidate, *candidate.parents):
        if item.exists() and item.is_symlink():
            return True
    return False


def default_package_dir() -> Path:
    script = Path(__file__).resolve()
    packaged = script.parent.parent
    if (packaged / "SHA256SUMS.txt").is_file():
        return packaged
    return script.parents[1] / "handoff" / "V30_publication_pareto_local_validation"


def discover_repo_root(package_dir: Path) -> Path:
    for candidate in (package_dir.resolve(), *package_dir.resolve().parents):
        if (candidate / "src").is_dir() and (candidate / ".venv-v30").is_dir():
            return candidate
    if package_dir.parent.name == "handoff":
        return package_dir.parent.parent.resolve()
    return Path.cwd().resolve()


def verify_manifest(package_dir: Path) -> dict[str, Any]:
    package_input = package_dir.absolute()
    if package_input.is_symlink():
        return gate_result(
            "FAIL",
            ["package directory must not be a symbolic link"],
            package_dir=str(package_input),
            checked_files=0,
        )
    package = package_input.resolve()
    manifest = package / "SHA256SUMS.txt"
    if not manifest.is_file():
        return gate_result(
            "UNKNOWN", ["missing SHA256SUMS.txt"], package_dir=str(package), checked_files=0
        )

    problems: list[str] = []
    entries: dict[str, str] = {}
    folded_entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return gate_result("FAIL", [f"cannot read manifest: {error}"], checked_files=0)
    for number, line in enumerate(lines, start=1):
        match = MANIFEST_PATTERN.fullmatch(line)
        if match is None:
            problems.append(f"invalid manifest syntax at line {number}")
            continue
        expected, raw_relative = match.groups()
        relative = raw_relative.replace("\\", "/")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            problems.append(f"unsafe manifest path at line {number}: {raw_relative}")
            continue
        canonical = pure.as_posix()
        if canonical != raw_relative:
            problems.append(f"non-canonical manifest path at line {number}: {raw_relative}")
        if canonical == "SHA256SUMS.txt":
            problems.append("manifest must not list itself")
            continue
        if canonical in entries:
            problems.append(f"duplicate manifest entry: {canonical}")
            continue
        folded = canonical.casefold()
        if folded in folded_entries:
            problems.append(
                "case-insensitive duplicate manifest entry: "
                f"{folded_entries[folded]} and {canonical}"
            )
            continue
        folded_entries[folded] = canonical
        entries[canonical] = expected.lower()

    for relative, expected in entries.items():
        candidate = package.joinpath(*PurePosixPath(relative).parts)
        cursor = package
        linked = None
        for part in PurePosixPath(relative).parts:
            cursor /= part
            if cursor.is_symlink():
                linked = cursor
                break
        if linked is not None:
            problems.append(f"symbolic links are not allowed: {relative}")
            continue
        resolved = candidate.resolve()
        if not path_is_within(resolved, package):
            problems.append(f"manifest path escapes package: {relative}")
            continue
        if not resolved.is_file():
            problems.append(f"missing file: {relative}")
            continue
        actual = file_sha256(resolved)
        if actual != expected:
            problems.append(f"hash mismatch: {relative} expected={expected} actual={actual}")

    actual_list = [
        item.relative_to(package).as_posix()
        for item in package.rglob("*")
        if item.is_file() and item != manifest
    ]
    actual = set(actual_list)
    actual_folded: dict[str, str] = {}
    for relative in sorted(actual_list):
        folded = relative.casefold()
        if folded in actual_folded and actual_folded[folded] != relative:
            problems.append(
                "case-insensitive duplicate package path: "
                f"{actual_folded[folded]} and {relative}"
            )
        actual_folded[folded] = relative
    for relative in sorted(actual - set(entries)):
        problems.append(f"unlisted extra file: {relative}")

    return gate_result(
        "PASS" if not problems else "FAIL",
        problems,
        package_dir=str(package),
        manifest_path=str(manifest),
        manifest_sha256=file_sha256(manifest),
        checked_files=len(entries),
    )


def windows_cloud_attributes_unavailable(attributes: int) -> bool:
    offline = getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    recall_on_open = getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x00040000)
    recall_on_data = getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x00400000)
    return bool(attributes & (offline | recall_on_open | recall_on_data))


def is_unavailable_cloud_file(path: Path) -> bool:
    return windows_cloud_attributes_unavailable(getattr(path.stat(), "st_file_attributes", 0))


def reproduce_frozen_predictions(
    model_dir: Path, candidate_pool: Path, repo_root: Path
) -> dict[str, Any]:
    if not candidate_pool.is_file():
        return gate_result("UNKNOWN", [f"missing frozen candidate pool: {candidate_pool}"])
    # repo_root is intentionally not added to sys.path: root/src is MODEL_VERSION=V1.
    # The frozen prediction contract is the installed 29.0.0 distribution in .venv-v30.
    _ = repo_root
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import joblib
            import numpy as np
            import pandas as pd
            import xgboost

            import isamo_green_meoh
            import isamo_green_meoh.config as config_module
            from isamo_green_meoh.surrogate import predict_surrogates
            import isamo_green_meoh.surrogate as surrogate_module

            distribution_version = importlib.metadata.version("isamo-green-meoh")
            package_path = Path(isamo_green_meoh.__file__).resolve()
            config_path = Path(config_module.__file__).resolve()
            surrogate_path = Path(surrogate_module.__file__).resolve()
            environment_root = Path(sys.prefix).resolve()
            identity_problems: list[str] = []
            if distribution_version != "29.0.0":
                identity_problems.append(
                    f"isamo-green-meoh distribution must be 29.0.0, got {distribution_version}"
                )
            if getattr(isamo_green_meoh, "MODEL_VERSION", None) != "V29_FINAL":
                identity_problems.append(
                    "imported isamo_green_meoh MODEL_VERSION is not V29_FINAL"
                )
            for label, module_path in (
                ("package", package_path),
                ("config", config_path),
                ("surrogate", surrogate_path),
            ):
                if not path_is_within(module_path, environment_root) or "site-packages" not in {
                    part.casefold() for part in module_path.parts
                }:
                    identity_problems.append(
                        f"{label} module is not loaded from the active .venv-v30 site-packages: {module_path}"
                    )
            code_paths = {
                "__init__.py": package_path,
                "config.py": config_path,
                "surrogate.py": surrogate_path,
            }
            actual_code_hashes = {name: file_sha256(path) for name, path in code_paths.items()}
            normalized_code_hashes = {
                name: lf_normalized_source_sha256(path) for name, path in code_paths.items()
            }
            if normalized_code_hashes != EXPECTED_PREDICTION_CODE_LF_SHA256:
                identity_problems.append(
                    "installed V29 prediction LF-normalized source hashes differ from the frozen manifest contract"
                )
            if identity_problems:
                return gate_result("FAIL", identity_problems)
            prediction_code = {
                "distribution": "isamo-green-meoh",
                "distribution_version": distribution_version,
                "model_version": "V29_FINAL",
                "module_paths": {name: str(path) for name, path in code_paths.items()},
                "module_raw_sha256": actual_code_hashes,
                "module_lf_normalized_sha256": normalized_code_hashes,
            }

            model_load: dict[str, Any] = {
                "mode": "DIRECT_JOBLIB",
                "direct_load_error": None,
                "portable_recovery": None,
            }
            try:
                models = {
                    target: joblib.load(model_dir / f"{target}.joblib")
                    for target in TARGETS
                }
            except xgboost.core.XGBoostError as error:
                if str(error).strip() != "input stream corrupted":
                    raise
                model_load["mode"] = "EXACT_UBJSON_STABLE_MODEL_SLICE_RECOVERY"
                model_load["direct_load_error"] = {
                    "type": type(error).__name__,
                    "message": str(error).strip(),
                }
                gc.collect()
                models, recovery = load_frozen_models_with_portable_xgboost_recovery(
                    model_dir, joblib, xgboost
                )
                model_load["portable_recovery"] = recovery
            frame = pd.read_csv(candidate_pool, float_precision="round_trip")
            required = [*FEATURES, *[f"{target}_pred" for target in TARGETS]]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                return gate_result(
                    "FAIL", [f"candidate pool lacks prediction contract columns: {missing}"]
                )
            if len(frame) != EXPECTED_CANDIDATE_ROWS:
                return gate_result(
                    "FAIL",
                    [
                        "candidate pool row count mismatch: "
                        f"expected={EXPECTED_CANDIDATE_ROWS} actual={len(frame)}"
                    ],
                )
            prediction = predict_surrogates(models, frame, list(FEATURES), return_uncertainty=True)
            max_delta: dict[str, float] = {}
            reference_digest = hashlib.sha256()
            recomputed_digest = hashlib.sha256()
            for target in TARGETS:
                observed = frame[f"{target}_pred"].to_numpy(dtype=float)
                recomputed = prediction[target].to_numpy(dtype=float)
                if not np.isfinite(observed).all() or not np.isfinite(recomputed).all():
                    return gate_result("FAIL", [f"non-finite frozen prediction for {target}"])
                delta = float(np.max(np.abs(observed - recomputed)))
                max_delta[target] = delta
                reference_digest.update(target.encode("utf-8"))
                reference_digest.update(np.asarray(observed, dtype="<f8").tobytes(order="C"))
                recomputed_digest.update(target.encode("utf-8"))
                recomputed_digest.update(np.asarray(recomputed, dtype="<f8").tobytes(order="C"))
            warning_audit = summarize_model_warnings(caught)
        problems = [
            f"frozen prediction mismatch for {target}: max_abs_delta={delta}"
            for target, delta in max_delta.items()
            if delta > 1e-8
        ]
        if warning_audit["actionable"]:
            problems.append(
                "model load/prediction emitted actionable warnings; reproduction is rejected"
            )
        recovery = model_load.get("portable_recovery")
        stable_binding = {
            "schema_version": "V30-frozen-prediction-binding-1.0.1",
            "candidate_pool_sha256": file_sha256(candidate_pool),
            "candidate_rows": EXPECTED_CANDIDATE_ROWS,
            "prediction_code_sha256": json_sha256(prediction_code),
            "model_load_mode": model_load["mode"],
            "portable_recovery_binding_sha256": (
                recovery.get("recovery_binding_sha256")
                if isinstance(recovery, dict)
                else None
            ),
            "reference_prediction_sha256": reference_digest.hexdigest(),
            "absolute_tolerance": 1e-8,
            "all_targets_within_tolerance": all(
                delta <= 1e-8 for delta in max_delta.values()
            ),
            "actionable_warning_count": warning_audit["actionable"],
            "warning_policy": (
                "Reject every warning except the fully recorded joblib 1.5.3 / NumPy 2.5 "
                "numpy_pickle.py:207 shape DeprecationWarning."
            ),
        }
        evidence = {
            "candidate_pool_sha256": stable_binding["candidate_pool_sha256"],
            "candidate_rows": EXPECTED_CANDIDATE_ROWS,
            "prediction_code": prediction_code,
            "prediction_code_sha256": stable_binding["prediction_code_sha256"],
            "model_load": model_load,
            "max_abs_delta": max_delta,
            "reference_prediction_sha256": reference_digest.hexdigest(),
            "recomputed_prediction_sha256_diagnostic_only": recomputed_digest.hexdigest(),
            "warning_audit": warning_audit,
            "stable_binding": stable_binding,
        }
        return gate_result(
            "PASS" if not problems else "FAIL",
            problems,
            prediction_evidence=evidence,
            prediction_evidence_sha256=json_sha256(stable_binding),
            warnings=warning_audit["details"],
        )
    except Exception as error:
        return gate_result(
            "FAIL",
            [f"model deserialisation/prediction reproduction failed: {type(error).__name__}: {error}"],
        )


def evaluate_frozen_models(
    model_dir: Path,
    environment_audit: Path,
    candidate_pool: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not environment_audit.is_file():
        return gate_result("UNKNOWN", [f"missing environment audit: {environment_audit}"])
    try:
        audit = read_json_object(environment_audit)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return gate_result("FAIL", [f"invalid environment audit: {error}"])

    problems: list[str] = []
    baseline = audit.get("scientific_baseline")
    if not isinstance(baseline, dict):
        problems.append("environment audit lacks scientific_baseline object")
        baseline = {}
    if baseline.get("model_version") != "V29_FINAL":
        problems.append("scientific model_version must be V29_FINAL")
    if baseline.get("model_hash_set_sha256") != EXPECTED_MODEL_SET_SHA256:
        problems.append("environment audit model set hash differs from frozen contract")
    if baseline.get("expected_model_assets") != EXPECTED_MODEL_ASSETS:
        problems.append("environment audit model asset size/hash contract differs")

    if not model_dir.is_dir():
        return gate_result(
            "UNKNOWN",
            [*problems, f"model directory is unavailable: {model_dir}"],
            model_dir=str(model_dir),
        )
    raw_hashes: dict[str, str] = {}
    assets: dict[str, Any] = {}
    missing = False
    for filename, expected in EXPECTED_MODEL_ASSETS.items():
        path = model_dir / filename
        item: dict[str, Any] = {"path": str(path)}
        if not path.is_file():
            item["status"] = "UNKNOWN"
            problems.append(f"missing frozen model asset: {filename}")
            missing = True
            assets[filename] = item
            continue
        if has_symlink_component(path):
            item["status"] = "FAIL"
            problems.append(f"model asset must not use a symbolic link: {filename}")
            assets[filename] = item
            continue
        size = path.stat().st_size
        item["size_bytes"] = size
        if size != expected["size_bytes"]:
            item["status"] = "FAIL"
            problems.append(
                f"model size mismatch: {filename} expected={expected['size_bytes']} actual={size}"
            )
            assets[filename] = item
            continue
        if is_unavailable_cloud_file(path):
            item["status"] = "UNKNOWN"
            problems.append(f"model is offline/recall-on-access; raw bytes not audited: {filename}")
            missing = True
            assets[filename] = item
            continue
        actual = file_sha256(path)
        item["sha256"] = actual
        if actual != expected["sha256"]:
            item["status"] = "FAIL"
            problems.append(
                f"model hash mismatch: {filename} expected={expected['sha256']} actual={actual}"
            )
        else:
            item["status"] = "PASS"
            raw_hashes[filename.removesuffix(".joblib")] = actual
        assets[filename] = item

    aggregate = model_set_sha256(raw_hashes) if len(raw_hashes) == len(TARGETS) else None
    if aggregate is not None and aggregate != EXPECTED_MODEL_SET_SHA256:
        problems.append(
            f"model set hash mismatch: expected={EXPECTED_MODEL_SET_SHA256} actual={aggregate}"
        )
    unknown_prefixes = (
        "missing frozen model asset:",
        "model is offline/recall-on-access;",
    )
    hard_fail = any(item.get("status") == "FAIL" for item in assets.values()) or any(
        not text.startswith(unknown_prefixes) for text in problems
    )
    reproduction = gate_result("UNKNOWN", ["prediction reproduction not run"])
    if not hard_fail and not missing and not problems:
        if candidate_pool is None or repo_root is None:
            reproduction = gate_result(
                "UNKNOWN", ["candidate pool/repository path not supplied for prediction reproduction"]
            )
        else:
            reproduction = reproduce_frozen_predictions(model_dir, candidate_pool, repo_root)
        problems.extend(reproduction.get("problems", []))
    status = "FAIL" if hard_fail else ("UNKNOWN" if missing else reproduction.get("status", "FAIL"))
    return gate_result(
        status,
        problems,
        model_dir=str(model_dir.resolve()),
        assets=assets,
        model_set_sha256=aggregate,
        expected_model_set_sha256=EXPECTED_MODEL_SET_SHA256,
        prediction_reproduction_status=reproduction.get("status"),
        prediction_evidence_sha256=reproduction.get("prediction_evidence_sha256"),
        prediction_reproduction=reproduction,
    )


def run_pip_check(repo_root: Path) -> tuple[bool, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "pip", "check"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = canonical_distribution_name(raw_name)
        if name in versions and versions[name] != distribution.version:
            raise ValueError(f"conflicting installed distributions for {name}")
        versions[name] = distribution.version
    return dict(sorted(versions.items()))


def parse_distribution_lock(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"invalid distribution lock syntax at line {number}")
        raw_name, version = line.split("==", 1)
        name = canonical_distribution_name(raw_name.strip())
        if not name or not version.strip():
            raise ValueError(f"invalid distribution lock entry at line {number}")
        if name in versions:
            raise ValueError(f"duplicate distribution lock entry: {name}")
        versions[name] = version.strip()
    if not versions:
        raise ValueError("distribution lock is empty")
    return dict(sorted(versions.items()))


def evaluate_python_environment(
    environment_audit: Path, repo_root: Path, distribution_lock: Path | None = None
) -> dict[str, Any]:
    if not environment_audit.is_file():
        return gate_result("UNKNOWN", [f"missing environment audit: {environment_audit}"])
    try:
        audit = read_json_object(environment_audit)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return gate_result("FAIL", [f"invalid environment audit: {error}"])
    distribution_lock = distribution_lock or environment_audit.parent / "phase0_python_packages.txt"
    if not distribution_lock.is_file():
        return gate_result("UNKNOWN", [f"missing exact distribution lock: {distribution_lock}"])
    try:
        locked_distributions = parse_distribution_lock(distribution_lock)
        actual_distributions = installed_distribution_versions()
    except (OSError, UnicodeError, ValueError) as error:
        return gate_result("FAIL", [f"invalid exact distribution lock/runtime metadata: {error}"])

    problems: list[str] = []
    if actual_distributions != locked_distributions:
        missing = sorted(set(locked_distributions) - set(actual_distributions))
        extra = sorted(set(actual_distributions) - set(locked_distributions))
        changed = {
            name: {"expected": locked_distributions[name], "actual": actual_distributions[name]}
            for name in sorted(set(actual_distributions) & set(locked_distributions))
            if actual_distributions[name] != locked_distributions[name]
        }
        problems.append(
            f"exact installed distribution set mismatch: missing={missing} extra={extra} changed={changed}"
        )
    actual_packages: dict[str, str | None] = {}
    expected_packages = {**EXPECTED_CORE_PACKAGES, **EXPECTED_WORKFLOW_PACKAGES}
    for name, expected in expected_packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        actual_packages[name] = actual
        if actual != expected:
            problems.append(f"package version mismatch: {name} expected={expected} actual={actual}")
    actual_python = platform.python_version()
    if actual_python != EXPECTED_PYTHON_VERSION:
        problems.append(
            f"Python version mismatch: expected={EXPECTED_PYTHON_VERSION} actual={actual_python}"
        )
    implementation = platform.python_implementation()
    if implementation != "CPython":
        problems.append(f"Python implementation must be CPython, got {implementation}")

    reference = audit.get("reference_python_environment")
    if not isinstance(reference, dict):
        problems.append("environment audit lacks reference_python_environment object")
        reference = {}
    expected_audit = {
        "relative_path": ".venv-v30",
        "python": EXPECTED_PYTHON_VERSION,
        "implementation": "CPython",
        "platform": platform.platform(),
        "core_packages": EXPECTED_CORE_PACKAGES,
        "workflow_packages": EXPECTED_WORKFLOW_PACKAGES,
        "imports": "PASS",
        "pip_check": "PASS_NO_BROKEN_REQUIREMENTS",
        "core_contract": "PASS",
    }
    for key, expected in expected_audit.items():
        if reference.get(key) != expected:
            problems.append(f"environment audit mismatch for reference_python_environment.{key}")

    expected_executable = (repo_root / ".venv-v30" / "python.exe").resolve()
    actual_executable = Path(sys.executable).resolve()
    if actual_executable != expected_executable:
        problems.append(
            f"wrong interpreter: expected={expected_executable} actual={actual_executable}"
        )
    try:
        pip_ok, pip_output = run_pip_check(repo_root)
    except (OSError, subprocess.SubprocessError) as error:
        pip_ok, pip_output = False, f"pip check could not run: {error}"
    if not pip_ok:
        problems.append(f"pip check failed: {pip_output}")

    fingerprint_payload = {
        "implementation": implementation,
        "python": actual_python,
        "executable": str(actual_executable),
        "packages": actual_packages,
        "platform": platform.platform(),
        "distribution_lock_sha256": file_sha256(distribution_lock),
        "installed_distributions": actual_distributions,
        "pip_check_pass": pip_ok,
    }
    return gate_result(
        "PASS" if not problems else "FAIL",
        problems,
        environment_audit=str(environment_audit.resolve()),
        distribution_lock=str(distribution_lock.resolve()),
        actual=fingerprint_payload,
        environment_fingerprint_sha256=json_sha256(fingerprint_payload),
        pip_check_output=pip_output,
    )


def canonical_feature_string(row: Mapping[str, str]) -> str:
    values: list[str] = []
    for name in FEATURES:
        raw = row.get(name)
        if raw is None or raw.strip() == "":
            raise ValueError(f"missing feature {name}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite feature {name}")
        if name in INTEGER_FEATURES:
            if not value.is_integer():
                raise ValueError(f"non-integral value for {name}: {raw}")
            values.append(str(int(value)))
        else:
            values.append(f"{value:.10f}")
    return "|".join(values)


def evaluate_role_hash_audit(package_dir: Path) -> dict[str, Any]:
    data_dir = package_dir / "data"
    problems: list[str] = []
    unknown = False
    role_hashes: dict[str, list[str]] = {}
    role_details: dict[str, Any] = {}
    for role, filename, expected_rows in ROLE_SPECS:
        path = data_dir / filename
        if not path.is_file():
            problems.append(f"missing authoritative role file: data/{filename}")
            unknown = True
            continue
        hashes: list[str] = []
        row_problems: list[str] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError("missing CSV header")
                has_canonical = "canonical_input_string_10dp" in reader.fieldnames
                has_hash = "input_row_sha256" in reader.fieldnames
                if has_canonical != has_hash:
                    row_problems.append("canonical/hash columns must either both exist or both be absent")
                for number, row in enumerate(reader, start=2):
                    try:
                        canonical = canonical_feature_string(row)
                        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                        if has_canonical and row.get("canonical_input_string_10dp") != canonical:
                            raise ValueError("canonical_input_string_10dp mismatch")
                        if has_hash and row.get("input_row_sha256", "") != digest:
                            raise ValueError("input_row_sha256 mismatch")
                        hashes.append(digest)
                    except (TypeError, ValueError, OverflowError) as error:
                        if len(row_problems) < 20:
                            row_problems.append(f"row {number}: {error}")
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            row_problems.append(f"cannot audit CSV: {error}")
        if len(hashes) != expected_rows:
            row_problems.append(f"row count mismatch: expected={expected_rows} valid={len(hashes)}")
        if len(set(hashes)) != len(hashes):
            row_problems.append("duplicate canonical input hash within role")
        if row_problems:
            problems.extend(f"{role}: {item}" for item in row_problems)
        role_hashes[role] = sorted(hashes)
        role_details[role] = {
            "path": str(path),
            "expected_rows": expected_rows,
            "audited_rows": len(hashes),
            "unique_hashes": len(set(hashes)),
        }

    overlaps: list[dict[str, Any]] = []
    role_names = [item[0] for item in ROLE_SPECS]
    for index, role_a in enumerate(role_names):
        for role_b in role_names[index + 1 :]:
            overlap = sorted(set(role_hashes.get(role_a, [])) & set(role_hashes.get(role_b, [])))
            overlaps.append({"role_a": role_a, "role_b": role_b, "count": len(overlap)})
            if overlap:
                problems.append(f"cross-role overlap: {role_a}/{role_b} count={len(overlap)}")

    claimed_path = data_dir / "14_split_overlap_audit.csv"
    if not claimed_path.is_file():
        problems.append("missing data/14_split_overlap_audit.csv")
        unknown = True
    else:
        claimed: dict[tuple[str, str], tuple[int, str]] = {}
        try:
            with claimed_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("role_a", ""), row.get("role_b", ""))
                    claimed[key] = (int(row.get("overlap_10dp_count", "")), row.get("status", ""))
            for item in overlaps:
                key = (item["role_a"], item["role_b"])
                if claimed.get(key) != (item["count"], "PASS" if item["count"] == 0 else "FAIL"):
                    problems.append(f"claimed split audit disagrees for {key[0]}/{key[1]}")
            if len(claimed) != len(overlaps):
                problems.append("claimed split audit does not contain exactly all role pairs")
        except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
            problems.append(f"invalid claimed split audit: {error}")

    digest = json_sha256(role_hashes) if len(role_hashes) == len(ROLE_SPECS) else None
    status = "UNKNOWN" if unknown and not any("mismatch" in p or "overlap" in p for p in problems) else (
        "PASS" if not problems else "FAIL"
    )
    return gate_result(
        status,
        problems,
        roles=role_details,
        pairwise_overlaps=overlaps,
        canonical_role_hash_set_sha256=digest,
    )


def resolve_artifact_path(evidence_path: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact path must be a non-empty string")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (evidence_path.parent / path).resolve()


def audit_bound_artifact(
    evidence_path: Path, name: str, declaration: Any
) -> tuple[Path | None, list[str], dict[str, Any]]:
    problems: list[str] = []
    detail: dict[str, Any] = {}
    if not isinstance(declaration, dict):
        return None, [f"artifact {name} declaration must be an object"], detail
    try:
        path = resolve_artifact_path(evidence_path, declaration.get("path"))
    except ValueError as error:
        return None, [f"artifact {name}: {error}"], detail
    detail["path"] = str(path)
    if has_symlink_component(path):
        return path, [f"artifact {name} must not use a symbolic link"], detail
    if not path.is_file():
        return path, [f"artifact {name} is missing: {path}"], detail
    size = path.stat().st_size
    digest = file_sha256(path)
    detail.update({"bytes": size, "sha256": digest})
    if declaration.get("bytes") != size:
        problems.append(f"artifact {name} byte count mismatch")
    expected_hash = declaration.get("sha256")
    if not isinstance(expected_hash, str) or expected_hash.lower() != digest:
        problems.append(f"artifact {name} SHA256 mismatch")
    return path, problems, detail


def evaluate_runner_evidence(
    runner_evidence: Path | None, package_dir: Path
) -> dict[str, Any]:
    if runner_evidence is None or not runner_evidence.is_file():
        return gate_result("UNKNOWN", ["final external runner/schema dry-run evidence is missing"])
    if "template" in runner_evidence.name.casefold():
        return gate_result("FAIL", ["runner evidence template cannot satisfy the gate"])
    if path_is_within(runner_evidence, package_dir):
        return gate_result("FAIL", ["runner evidence must be outside the manifest-listed handoff"])
    if has_symlink_component(runner_evidence):
        return gate_result("FAIL", ["runner evidence must not use a symbolic link"])
    try:
        evidence = read_json_object(runner_evidence)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return gate_result("FAIL", [f"invalid runner evidence: {error}"])

    problems: list[str] = []
    expected_scalars = {
        "schema_version": "V30-gproms-runner-schema-dry-run-evidence-1.0.0",
        "record_status": "FINAL_EVIDENCE",
        "template": False,
        "dry_run": True,
        "dry_run_status": "PASS",
        "gproms_process_launched": False,
        "labels_requested": 0,
    }
    for key, expected in expected_scalars.items():
        if evidence.get(key) != expected:
            problems.append(f"runner evidence field {key} must equal {expected!r}")
    if evidence.get("required_input_columns") != list(REQUIRED_RUNNER_INPUT_COLUMNS):
        problems.append("runner evidence required_input_columns differs from frozen schema")
    if evidence.get("required_return_columns") != list(REQUIRED_RUNNER_OUTPUT_COLUMNS):
        problems.append("runner evidence required_return_columns differs from frozen schema")

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        return gate_result("FAIL", [*problems, "runner evidence artifacts must be an object"])
    required_artifacts = {"runner", "frozen_project", "input_queue", "output_schema", "dry_run_report", "dry_run_log"}
    if set(artifacts) != required_artifacts:
        problems.append("runner evidence must declare exactly the six required artifacts")
    paths: dict[str, Path] = {}
    audited: dict[str, Any] = {}
    for name in sorted(required_artifacts):
        path, artifact_problems, detail = audit_bound_artifact(
            runner_evidence, name, artifacts.get(name)
        )
        problems.extend(artifact_problems)
        audited[name] = detail
        if path is not None and path.is_file():
            paths[name] = path

    project_detail = audited.get("frozen_project", {})
    if project_detail.get("sha256") != EXPECTED_GPROMS_PROJECT_SHA256:
        problems.append("runner evidence frozen project is not the approved B7E0ED3D project")
    runner_path = paths.get("runner")
    if runner_path is not None and runner_path.suffix.casefold() != ".py":
        problems.append("runner artifact must be a Python source file")
    input_path = paths.get("input_queue")
    if input_path is not None:
        try:
            with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames or []
                rows = list(reader)
            if not set(REQUIRED_RUNNER_INPUT_COLUMNS).issubset(header):
                problems.append("dry-run input queue lacks required input columns")
            if not rows:
                problems.append("dry-run input queue must contain at least one design")
            identifiers: list[str] = []
            hashes: list[str] = []
            for number, row in enumerate(rows, start=2):
                try:
                    record_id = row.get("record_id", "").strip()
                    if not record_id:
                        raise ValueError("record_id is blank")
                    if not row.get("dataset_role", "").strip():
                        raise ValueError("dataset_role is blank")
                    canonical = canonical_feature_string(row)
                    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if row.get("input_row_sha256", "") != digest:
                        raise ValueError("input_row_sha256 mismatch")
                    mapping = {
                        "Current_density_eCO2R_Acm2": float(row["J_eCO2R"]),
                        "WE_Energy_input_rate_KW": float(row["P_WE"]),
                        "Number_of_cells_stacked_in_series_WE": float(row["N_WE"]),
                        "Number_of_stacks_eCO2R": float(row["N_eCO2R"]),
                        "Outlet_pressure": float(row["P_MeOH"]),
                        "Outlet_pressure_cap": float(row["P_Cap"]),
                        "T_eCO2R_gproms_input": float(row["T_eCO2R"]),
                        "Utility_constant_temperature": float(row["T_utility"]) - 273.15,
                    }
                    for column, expected in mapping.items():
                        if not math.isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=1e-9):
                            raise ValueError(f"gPROMS mapping mismatch for {column}")
                    identifiers.append(record_id)
                    hashes.append(digest)
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    problems.append(f"dry-run input queue row {number}: {error}")
            if len(set(identifiers)) != len(identifiers):
                problems.append("dry-run input queue record_id values are duplicated")
            if len(set(hashes)) != len(hashes):
                problems.append("dry-run input queue input hashes are duplicated")
        except (OSError, UnicodeError, csv.Error, StopIteration) as error:
            problems.append(f"cannot validate dry-run input queue: {error}")

    schema_path = paths.get("output_schema")
    if schema_path is not None:
        try:
            schema = read_json_object(schema_path)
            if schema.get("schema_version") != "V30-gproms-return-schema-1.0.0":
                problems.append("output schema version mismatch")
            if schema.get("columns") != list(REQUIRED_RUNNER_OUTPUT_COLUMNS):
                problems.append("output schema columns differ from frozen return schema")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            problems.append(f"invalid output schema artifact: {error}")

    report_path = paths.get("dry_run_report")
    if report_path is not None:
        try:
            report = read_json_object(report_path)
            expected_report = {
                "schema_version": "V30-gproms-runner-schema-dry-run-report-1.0.0",
                "status": "PASS",
                "dry_run": True,
                "writes_performed": False,
                "gproms_process_launched": False,
                "labels_requested": 0,
                "runner_sha256": audited.get("runner", {}).get("sha256"),
                "frozen_project_sha256": audited.get("frozen_project", {}).get("sha256"),
                "input_queue_sha256": audited.get("input_queue", {}).get("sha256"),
                "output_schema_sha256": audited.get("output_schema", {}).get("sha256"),
            }
            for key, expected in expected_report.items():
                if report.get(key) != expected:
                    problems.append(f"dry-run report binding mismatch for {key}")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            problems.append(f"invalid dry-run report artifact: {error}")
    log_path = paths.get("dry_run_log")
    if log_path is not None and log_path.stat().st_size == 0:
        problems.append("dry-run log must not be empty")

    return gate_result(
        "PASS" if not problems else "FAIL",
        problems,
        runner_evidence_path=str(runner_evidence.resolve()),
        runner_evidence_sha256=file_sha256(runner_evidence),
        artifacts=audited,
    )


def parse_aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def evaluate_existing60_scientific_approval(
    package_dir: Path,
    gate_path: Path,
    approval_record: Path | None,
    trusted_approval_sha256: str | None,
    manifest_gate: Mapping[str, Any],
    models_gate: Mapping[str, Any],
    environment_gate: Mapping[str, Any],
    roles_gate: Mapping[str, Any],
    runner_gate: Mapping[str, Any],
) -> dict[str, Any]:
    if not gate_path.is_file():
        return gate_result("UNKNOWN", [f"existing60 provenance gate is missing: {gate_path}"])
    try:
        candidate = read_json_object(gate_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return gate_result("FAIL", [f"invalid existing60 provenance gate: {error}"])

    problems: list[str] = []
    if candidate.get("schema_version") != "V30-existing60-retrospective-provenance-1.0.0":
        problems.append("existing60 provenance schema version mismatch")
    if candidate.get("rows") != 60:
        problems.append("existing60 provenance gate must cover 60 rows")
    checks = candidate.get("checks")
    required_true_checks = (
        "row_count_60",
        "validation_ids_unique_and_equal",
        "all_input_hashes_match",
        "all_v30_selected_numeric_values_identical_within_1e-9",
        "all_selected_rows_match_hashed_source_csv_scenarios_within_1e-9",
        "all_60_converged",
        "all_source_csv_netcdf_message_files_present_and_hashed",
        "all_execution_project_hashes_match_manifests",
        "actual_gproms_source_model_attested",
    )
    if not isinstance(checks, dict):
        problems.append("existing60 provenance gate lacks checks object")
        checks = {}
    for key in required_true_checks:
        if checks.get(key) is not True:
            problems.append(f"existing60 objective check is not true: {key}")
    if checks.get("placeholder_simulator_used_any") is not False:
        problems.append("existing60 evidence does not objectively exclude placeholder simulation")

    ledger_path = package_dir / "data" / "19_existing60_retrospective_provenance.csv"
    if not ledger_path.is_file():
        return gate_result("UNKNOWN", [*problems, "existing60 provenance ledger is missing"])
    ledger_hash = file_sha256(ledger_path)
    ledger = candidate.get("ledger")
    if not isinstance(ledger, dict):
        problems.append("existing60 gate lacks ledger binding")
        ledger = {}
    if ledger.get("sha256") != ledger_hash or ledger.get("bytes") != ledger_path.stat().st_size:
        problems.append("existing60 gate ledger hash/size binding mismatch")
    try:
        with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 60:
            problems.append(f"existing60 ledger row count must be 60, got {len(rows)}")
        ids = [row.get("validation_id", "") for row in rows]
        if not all(ids) or len(set(ids)) != len(ids):
            problems.append("existing60 ledger validation IDs are missing or duplicated")
        for number, row in enumerate(rows, start=2):
            try:
                canonical = canonical_feature_string(row)
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if row.get("canonical_input_string_10dp") != canonical:
                    raise ValueError("canonical string mismatch")
                if row.get("input_row_sha256", "") != digest:
                    raise ValueError("input hash mismatch")
                if row.get("input_hash_match_status") != "PASS":
                    raise ValueError("source input hash status is not PASS")
                if row.get("numeric_identity_status") != "PASS":
                    raise ValueError("numeric identity status is not PASS")
                if row.get("source_csv_row_match_status") != "PASS":
                    raise ValueError("source CSV row status is not PASS")
                if row.get("actual_gproms_source_model_used", "").casefold() != "true":
                    raise ValueError("actual gPROMS source is not attested")
                if row.get("placeholder_simulator_used", "").casefold() != "false":
                    raise ValueError("placeholder simulator is not excluded")
            except (TypeError, ValueError, OverflowError) as error:
                if len(problems) < 30:
                    problems.append(f"existing60 ledger row {number}: {error}")
    except (OSError, UnicodeError, csv.Error) as error:
        problems.append(f"cannot audit existing60 ledger: {error}")

    if approval_record is None or not approval_record.is_file():
        return gate_result(
            "FAIL" if problems else "UNKNOWN",
            [*problems, "final external scientific approval record is missing"],
            source_gate_status=candidate.get("status"),
            existing60_gate_sha256=file_sha256(gate_path),
            existing60_ledger_sha256=ledger_hash,
        )
    if "template" in approval_record.name.casefold():
        return gate_result("FAIL", [*problems, "approval template cannot satisfy the gate"])
    if path_is_within(approval_record, package_dir):
        return gate_result(
            "FAIL", [*problems, "final approval record must be outside the manifest-listed handoff"]
        )
    if has_symlink_component(approval_record):
        return gate_result("FAIL", [*problems, "approval record must not use a symbolic link"])
    actual_approval_hash = file_sha256(approval_record)
    if trusted_approval_sha256 is None:
        return gate_result(
            "FAIL" if problems else "UNKNOWN",
            [*problems, "independent trusted approval SHA256 was not supplied"],
            approval_record_sha256=actual_approval_hash,
        )
    trusted = trusted_approval_sha256.casefold()
    if not SHA256_PATTERN.fullmatch(trusted):
        return gate_result("FAIL", [*problems, "trusted approval SHA256 is malformed"])
    if trusted != actual_approval_hash:
        return gate_result(
            "FAIL",
            [*problems, "approval record does not match the independent trusted SHA256"],
            approval_record_sha256=actual_approval_hash,
        )
    try:
        approval = read_json_object(approval_record)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return gate_result("FAIL", [*problems, f"invalid approval record: {error}"])

    expected_fields = {
        "schema_version": "V30-phase0-gate-opening-1.0.0",
        "record_status": "FINAL_SCIENTIFIC_APPROVAL",
        "authorizes_phase1_execution": True,
        "handoff_manifest_sha256": manifest_gate.get("manifest_sha256"),
        "manifest_verifier_status": "PASS",
        "manifest_checked_files": manifest_gate.get("checked_files"),
        "frozen_model_assets_status": "PASS",
        "frozen_model_set_sha256": models_gate.get("model_set_sha256"),
        "frozen_prediction_reproduction_status": "PASS",
        "frozen_prediction_evidence_sha256": models_gate.get("prediction_evidence_sha256"),
        "python_environment_status": "PASS",
        "python_environment_fingerprint_sha256": environment_gate.get(
            "environment_fingerprint_sha256"
        ),
        "existing60_provenance_decision": "APPROVED_RETROSPECTIVE_PROVENANCE",
        "existing60_gate_record_sha256": file_sha256(gate_path),
        "existing60_ledger_sha256": ledger_hash,
        "canonical_role_overlap_status": "PASS",
        "canonical_role_hash_set_sha256": roles_gate.get(
            "canonical_role_hash_set_sha256"
        ),
        "gproms_project_sha256": EXPECTED_GPROMS_PROJECT_SHA256,
        "runner_protocol_validation_status": "PASS",
        "runner_evidence_sha256": runner_gate.get("runner_evidence_sha256"),
        "scientific_owner_decision": "APPROVED",
        "approval_scope": "V30_PHASE1_EXECUTION",
    }
    for key, expected in expected_fields.items():
        if expected is None or approval.get(key) != expected:
            problems.append(f"final approval binding mismatch for {key}")
    if not parse_aware_timestamp(approval.get("created_at_utc")):
        problems.append("final approval created_at_utc must be timezone-aware ISO-8601")
    if not isinstance(approval.get("scientific_owner"), str) or not approval["scientific_owner"].strip():
        problems.append("final approval must identify the scientific owner")
    if not isinstance(approval.get("approval_notes"), str) or not approval["approval_notes"].strip():
        problems.append("final approval must record approval notes")

    expected_queue_hashes = {
        relative: file_sha256(package_dir / relative)
        for relative in QUEUE_PATHS
        if (package_dir / relative).is_file()
    }
    if approval.get("specified_queue_paths") != list(QUEUE_PATHS):
        problems.append("final approval queue path list differs from the frozen four queues")
    if approval.get("specified_queue_sha256") != expected_queue_hashes:
        problems.append("final approval queue hashes do not match current queue bytes")
    candidate_ids = candidate.get("candidate_identifiers")
    if not isinstance(candidate_ids, dict):
        candidate_ids = {}
    if approval.get("gproms_model_version") != candidate_ids.get("gproms_model_version_candidate"):
        problems.append("final approval gPROMS model version is not bound to existing60 evidence")
    if approval.get("solver_settings_id") != candidate_ids.get("solver_settings_id_candidate"):
        problems.append("final approval solver settings ID is not bound to existing60 evidence")

    upstream = {
        "strict_handoff_manifest": manifest_gate.get("status"),
        "frozen_models": models_gate.get("status"),
        "python_environment": environment_gate.get("status"),
        "role_hash_audit": roles_gate.get("status"),
        "gproms_runner_schema_dry_run": runner_gate.get("status"),
    }
    for name, status in upstream.items():
        if status != "PASS":
            problems.append(f"final approval cannot override upstream gate {name}={status}")
    return gate_result(
        "PASS" if not problems else "FAIL",
        problems,
        source_gate_status=candidate.get("status"),
        approval_record_path=str(approval_record.resolve()),
        approval_record_sha256=actual_approval_hash,
        trusted_approval_sha256=trusted,
        existing60_gate_sha256=file_sha256(gate_path),
        existing60_ledger_sha256=ledger_hash,
    )


def safe_gate(name: str, operation: Any) -> dict[str, Any]:
    try:
        return operation()
    except Exception as error:  # A total gate must fail closed even on an unexpected audit error.
        return gate_result("FAIL", [f"unexpected {name} audit error: {type(error).__name__}: {error}"])


def evaluate_phase0_gate(
    package_dir: Path,
    *,
    repo_root: Path | None = None,
    model_dir: Path | None = None,
    environment_audit: Path | None = None,
    existing60_gate: Path | None = None,
    approval_record: Path | None = None,
    trusted_approval_sha256: str | None = None,
    runner_evidence: Path | None = None,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    repo_root = (repo_root or discover_repo_root(package_dir)).resolve()
    model_dir = (model_dir or repo_root / "outputs" / "surrogatemodellingresults" / "models").resolve()
    environment_audit = (
        environment_audit or package_dir / "data" / "phase0_environment_audit.json"
    ).resolve()
    existing60_gate = (
        existing60_gate or package_dir / "data" / "existing60_provenance_gate.json"
    ).resolve()

    manifest = safe_gate("manifest", lambda: verify_manifest(package_dir))
    models = safe_gate(
        "frozen models",
        lambda: evaluate_frozen_models(
            model_dir,
            environment_audit,
            package_dir / "data" / "04_candidate_pool_76468_annotated.csv",
            repo_root,
        ),
    )
    environment = safe_gate(
        "Python environment", lambda: evaluate_python_environment(environment_audit, repo_root)
    )
    roles = safe_gate("role/hash", lambda: evaluate_role_hash_audit(package_dir))
    runner = safe_gate(
        "runner/schema dry-run", lambda: evaluate_runner_evidence(runner_evidence, package_dir)
    )
    existing60 = safe_gate(
        "existing60 scientific approval",
        lambda: evaluate_existing60_scientific_approval(
            package_dir,
            existing60_gate,
            approval_record,
            trusted_approval_sha256,
            manifest,
            models,
            environment,
            roles,
            runner,
        ),
    )
    manifest_after = safe_gate("final manifest", lambda: verify_manifest(package_dir))
    manifest_identity = (
        manifest.get("status"),
        manifest.get("manifest_sha256"),
        manifest.get("checked_files"),
        manifest.get("problems"),
    )
    manifest_after_identity = (
        manifest_after.get("status"),
        manifest_after.get("manifest_sha256"),
        manifest_after.get("checked_files"),
        manifest_after.get("problems"),
    )
    if manifest_identity != manifest_after_identity:
        manifest = gate_result(
            "FAIL",
            ["handoff or manifest changed during Phase-0 evaluation"],
            initial_verification=manifest,
            final_verification=manifest_after,
            verification_passes=2,
        )
    else:
        manifest = {**manifest, "verification_passes": 2}
    gates = {
        "strict_handoff_manifest": manifest,
        "frozen_model_assets": models,
        "exact_python_environment": environment,
        "canonical_role_hash_audit": roles,
        "gproms_runner_schema_dry_run": runner,
        "existing60_final_scientific_approval": existing60,
    }
    authorizes = all(item.get("status") == "PASS" for item in gates.values())
    failures = [
        {"gate": name, "status": item.get("status"), "problems": item.get("problems", [])}
        for name, item in gates.items()
        if item.get("status") != "PASS"
    ]
    return {
        "schema_version": "V30-phase0-total-gate-result-1.0.0",
        "evaluator_version": EVALUATOR_VERSION,
        "package_dir": str(package_dir),
        "repo_root": str(repo_root),
        "overall_status": "OPEN" if authorizes else "CLOSED",
        "authorizes_phase1_execution": authorizes,
        "decision_rule": "OPEN only when every independently recomputed gate is PASS",
        "evidence_semantics": {
            "default_mode": "READ_ONLY",
            "missing_evidence": "UNKNOWN_AND_GATE_CLOSED",
            "invalid_or_mismatched_evidence": "FAIL_AND_GATE_CLOSED",
            "templates": "NEVER_APPROVAL",
            "candidate_existing60_record": "EVIDENCE_ONLY_NOT_APPROVAL",
            "approval_trust": "EXTERNAL_RECORD_PLUS_INDEPENDENT_TRUSTED_SHA256",
        },
        "gates": gates,
        "nonpassing_gates": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed, machine-decidable V30 Phase-0 total execution gate"
    )
    parser.add_argument("--package-dir", type=Path, default=default_package_dir())
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--environment-audit", type=Path)
    parser.add_argument("--existing60-gate", type=Path)
    parser.add_argument("--approval-record", type=Path)
    parser.add_argument("--trusted-approval-sha256")
    parser.add_argument("--runner-evidence", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit JSON output path. Without this option the evaluator is read-only.",
    )
    args = parser.parse_args()
    result = evaluate_phase0_gate(
        args.package_dir,
        repo_root=args.repo_root,
        model_dir=args.model_dir,
        environment_audit=args.environment_audit,
        existing60_gate=args.existing60_gate,
        approval_record=args.approval_record,
        trusted_approval_sha256=args.trusted_approval_sha256,
        runner_evidence=args.runner_evidence,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if path_is_within(output, args.package_dir.resolve()):
            parser.error("--output must be outside the manifest-listed handoff package")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    sys.stdout.write(rendered)
    raise SystemExit(0 if result["authorizes_phase1_execution"] else 2)


if __name__ == "__main__":
    main()
