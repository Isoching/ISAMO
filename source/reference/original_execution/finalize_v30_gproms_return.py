from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
TARGET_RECONSTRUCTION_CONTRACT_VERSION = "V30-target-reconstruction-1.0.1"
LEGACY_MEOH_YIELD_HOURS_PER_YEAR = 8400.0
V30_SEC_HOURS_PER_YEAR = 8760.0
METHANOL_MOLAR_MASS_KG_PER_KMOL = 32.04
CAPTURE_CO2_FEED_MOLAR_FLOW = 11.8
REQUIRED_RAW = [
    "MeOH_conversion",
    "MeOH_yield_kt_y",
    "allpower_kw",
    "Total_ele_power_kw",
]
REQUIRED_DIAGNOSTIC_RAW = [
    "S",
    "CAP_conversion",
    "eCO2R_conversion",
    "Coolduty_kw",
    "Heatduty_kw",
]
RECONSTRUCTABLE_DIAGNOSTICS = ["CAP_conversion", "eCO2R_conversion", "S"]
DIRECT_REQUIRED_DIAGNOSTICS = ["Coolduty_kw", "Heatduty_kw"]
DIAGNOSTIC_PRIMITIVE_RAW_COLUMNS = {
    "CO2_CAP": "Component_molar_flowrate_carbon_dioxide_CAP",
    "CO_CE": "Component_molar_flowrate_carbon_monoxide_CE",
    "H2_WE": "Component_molar_flowrate_hydrogen_WE",
    "H2_CE": "Component_molar_flowrate_hydrogen_CE",
    "CO2_CE": "Component_molar_flowrate_carbon_dioxide_CE",
}
MEOH_YIELD_PRIMITIVE_RAW_COLUMN = "Component_molar_flowrate_methanol"
DIAGNOSTIC_FORMULAS = {
    "CAP_conversion": "CO2_CAP / 11.8",
    "eCO2R_conversion": "CO_CE / (CAP_conversion * 11.8)",
    "S": "(H2_WE + H2_CE - CO2_CE) / (CO_CE + CO2_CE)",
}
DIAGNOSTIC_EXPANDED_FORMULAS = {
    "CAP_conversion": "Component_molar_flowrate_carbon_dioxide_CAP / 11.8",
    "eCO2R_conversion": (
        "Component_molar_flowrate_carbon_monoxide_CE / "
        "((Component_molar_flowrate_carbon_dioxide_CAP / 11.8) * 11.8)"
    ),
    "S": (
        "(Component_molar_flowrate_hydrogen_WE + "
        "Component_molar_flowrate_hydrogen_CE - "
        "Component_molar_flowrate_carbon_dioxide_CE) / "
        "(Component_molar_flowrate_carbon_monoxide_CE + "
        "Component_molar_flowrate_carbon_dioxide_CE)"
    ),
}
INPUT_ECHO_PAIRS = [
    ("J_eCO2R", "echo_Current_density_specification_1", 1e-9),
    ("P_WE", "echo_Energy_input_rate", 1e-9),
    ("N_WE", "echo_Number_of_cells_stacked_in_series", None),
    ("N_eCO2R", "echo_Number_of_stacks", None),
    ("P_MeOH", "echo_Outlet_pressure", 1e-9),
    ("P_Cap", "echo_Outlet_pressure_cap", 1e-9),
    ("T_eCO2R", "echo_Temperature", 1e-9),
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
FAILED_RECORD_QC_COLUMNS = [
    "record_id",
    "input_row_sha256",
    "dataset_role",
    "attempt_count",
    "converged_attempt_count",
    "selected_final_result_count",
    "record_level_qc_status",
    "attempt_ids_semicolon",
    "last_gproms_attempt_id",
    "last_converged",
    "last_solver_status_code",
    "last_failed_reason",
    "last_execution_timestamp",
    "last_execution_batch_id",
    "last_gproms_scenario_id",
    "last_raw_output_file",
    "last_raw_output_file_sha256",
    "gproms_model_version",
    "solver_settings_id",
    "all_attempt_raw_hash_qc",
    "all_attempt_input_echo_qc",
]
SELECTION_COLUMN = "selected_final_result"
MUTABLE_RETURN_COLUMNS = {
    SELECTION_COLUMN,
    "converged",
    "failed_reason",
    *REQUIRED_RAW,
    *REQUIRED_DIAGNOSTIC_RAW,
    *DIAGNOSTIC_PRIMITIVE_RAW_COLUMNS.values(),
    MEOH_YIELD_PRIMITIVE_RAW_COLUMN,
    *[echo for _, echo, _ in INPUT_ECHO_PAIRS],
    "echo_Utility_constant_temperature",
    *PROVENANCE_COLUMNS,
    *DERIVED_RETURN_COLUMNS,
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_feature_string(row: pd.Series) -> str:
    parts: list[str] = []
    for name in FEATURES:
        if name in INTEGER_FEATURES:
            parts.append(str(int(float(row[name]))))
        else:
            parts.append(f"{float(row[name]):.10f}")
    return "|".join(parts)


def feature_hash(row: pd.Series) -> str:
    return hashlib.sha256(canonical_feature_string(row).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "converged", "success"}:
        return True
    if text in {"false", "0", "no", "n", "failed", "failure"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def load_table(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        if sheet is None:
            raise ValueError("--sheet is required for an Excel return file")
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, float_precision="round_trip")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def require_text(series: pd.Series, column: str) -> pd.Series:
    missing = blank_mask(series)
    if missing.any():
        raise ValueError(f"Every attempt requires non-empty {column}")
    return series.astype(str).str.strip()


def require_sha256(series: pd.Series, column: str) -> pd.Series:
    values = require_text(series, column)
    if not values.str.fullmatch(SHA256_PATTERN).all():
        raise ValueError(f"{column} must contain lowercase 64-character SHA256 values")
    return values


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    required_mask: pd.Series | None = None,
) -> pd.Series:
    original = frame[column]
    missing = blank_mask(original)
    numeric = pd.to_numeric(original, errors="coerce")
    finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=frame.index)
    invalid_supplied = ~missing & (~numeric.notna() | ~finite)
    if invalid_supplied.any():
        raise ValueError(f"{column} contains non-numeric or non-finite values")
    if required_mask is not None and (required_mask & (missing | ~finite)).any():
        raise ValueError(f"Required {column} values are missing or non-finite")
    return numeric.astype(float)


def validate_feature_contract(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, ["record_id", "input_row_sha256", *FEATURES])
    frame["record_id"] = require_text(frame["record_id"], "record_id")
    frame["input_row_sha256"] = require_sha256(frame["input_row_sha256"], "input_row_sha256")
    required = pd.Series(True, index=frame.index)
    for column in FEATURES:
        frame[column] = numeric_column(frame, column, required_mask=required)
    for column in INTEGER_FEATURES:
        values = frame[column]
        if not values.eq(np.rint(values)).all():
            raise ValueError(f"{label} {column} must contain exact integer values")
    rebuilt_strings = frame.apply(canonical_feature_string, axis=1)
    rebuilt_hashes = rebuilt_strings.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not rebuilt_hashes.eq(frame["input_row_sha256"]).all():
        bad = frame.loc[~rebuilt_hashes.eq(frame["input_row_sha256"]), "record_id"].tolist()
        raise ValueError(f"{label} input hash mismatch: {bad[:10]}{'...' if len(bad) > 10 else ''}")
    if "canonical_input_string_10dp" in frame.columns:
        supplied = require_text(frame["canonical_input_string_10dp"], "canonical_input_string_10dp")
        if not supplied.eq(rebuilt_strings).all():
            raise ValueError(f"{label} canonical_input_string_10dp does not match the frozen hash contract")


def finite_metadata_number(
    mapping: dict[str, Any],
    key: str,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if key not in mapping:
        raise ValueError(f"Metadata is missing {path}.{key}")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metadata {path}.{key} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"Metadata {path}.{key} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"Metadata {path}.{key} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"Metadata {path}.{key} must be non-negative")
    return value


def metadata_mapping(mapping: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata {path}.{key} must be an object")
    return value


def validate_metadata(issued: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    contract_version = str(metadata.get("contract_version", "")).strip()
    if contract_version != TARGET_RECONSTRUCTION_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported target reconstruction contract_version; expected "
            f"{TARGET_RECONSTRUCTION_CONTRACT_VERSION}, observed {contract_version or '<missing>'}"
        )
    require_columns(issued, ["target_definition_sha256"])
    issued_hashes = require_sha256(issued["target_definition_sha256"], "target_definition_sha256")
    unique_hashes = issued_hashes.unique().tolist()
    if len(unique_hashes) != 1:
        raise ValueError("Issued queue must contain exactly one target_definition_sha256")
    source_hash = str(metadata.get("source_target_definition_sha256", "")).strip()
    if not SHA256_PATTERN.fullmatch(source_hash):
        raise ValueError("Metadata source_target_definition_sha256 must be a lowercase SHA256")
    if source_hash != unique_hashes[0]:
        raise ValueError(
            "Metadata source_target_definition_sha256 does not match the issued target_definition_sha256"
        )

    annualisation = metadata_mapping(metadata, "annualisation_hours_per_year", "metadata")
    legacy_hours = finite_metadata_number(
        annualisation,
        "legacy_MeOH_yield_kt_y",
        "annualisation_hours_per_year",
        positive=True,
    )
    v30_sec_hours = finite_metadata_number(
        annualisation,
        "V30_SEC_reconstruction",
        "annualisation_hours_per_year",
        positive=True,
    )
    if legacy_hours != LEGACY_MEOH_YIELD_HOURS_PER_YEAR:
        raise ValueError("Legacy MeOH_yield_kt_y annualisation must remain frozen at 8400 h/y")
    if v30_sec_hours != V30_SEC_HOURS_PER_YEAR:
        raise ValueError("V30 SEC annualisation must remain frozen at 8760 h/y")
    if legacy_hours == v30_sec_hours:
        raise ValueError("Legacy MeOH yield and V30 SEC hour bases must not be silently conflated")
    operating_hours = finite_metadata_number(
        metadata, "operating_hours_per_year", "metadata", positive=True
    )
    if operating_hours != v30_sec_hours:
        raise ValueError("operating_hours_per_year must equal the frozen V30 SEC 8760 h/y basis")

    legacy_yield = metadata_mapping(metadata, "legacy_meoh_yield_reconstruction", "metadata")
    expected_legacy_text = {
        "output_column": "MeOH_yield_kt_y",
        "primitive_raw_column": MEOH_YIELD_PRIMITIVE_RAW_COLUMN,
        "formula": (
            "Component_molar_flowrate_methanol * 8400 * 32.04 / 1e6"
        ),
    }
    for key, expected in expected_legacy_text.items():
        if legacy_yield.get(key) != expected:
            raise ValueError(f"Invalid legacy_meoh_yield_reconstruction.{key}")
    legacy_block_hours = finite_metadata_number(
        legacy_yield,
        "annualisation_hours_per_year",
        "legacy_meoh_yield_reconstruction",
        positive=True,
    )
    molar_mass = finite_metadata_number(
        legacy_yield,
        "methanol_molar_mass_kg_per_kmol",
        "legacy_meoh_yield_reconstruction",
        positive=True,
    )
    if legacy_block_hours != legacy_hours or molar_mass != METHANOL_MOLAR_MASS_KG_PER_KMOL:
        raise ValueError("Legacy MeOH yield reconstruction constants do not match the frozen contract")

    sec_contract = metadata_mapping(metadata, "v30_sec_reconstruction", "metadata")
    expected_sec_text = {
        "output_column": "specific_energy_kwh_per_kg_meoh_rebuilt",
        "power_column": "allpower_kw",
        "yield_column": "MeOH_yield_kt_y",
        "formula": "allpower_kw / (MeOH_yield_kt_y * 1e6 / 8760)",
    }
    for key, expected in expected_sec_text.items():
        if sec_contract.get(key) != expected:
            raise ValueError(f"Invalid v30_sec_reconstruction.{key}")
    sec_block_hours = finite_metadata_number(
        sec_contract,
        "annualisation_hours_per_year",
        "v30_sec_reconstruction",
        positive=True,
    )
    if sec_block_hours != v30_sec_hours:
        raise ValueError("V30 SEC reconstruction hours do not match the frozen 8760 h/y basis")
    derived_target_mapping = metadata_mapping(metadata, "derived_target_mapping", "metadata")
    if derived_target_mapping.get("specific_energy_kwh_per_kg_meoh") != (
        "allpower_kw / (MeOH_yield_kt_y * 1e6 / 8760)"
    ):
        raise ValueError(
            "derived_target_mapping must retain the frozen 8760 h/y V30 SEC formula"
        )

    diagnostic_contract = metadata_mapping(
        metadata, "diagnostic_reconstruction_contract", "metadata"
    )
    feed_molar_flow = finite_metadata_number(
        diagnostic_contract,
        "capture_co2_feed_molar_flow",
        "diagnostic_reconstruction_contract",
        positive=True,
    )
    if feed_molar_flow != CAPTURE_CO2_FEED_MOLAR_FLOW:
        raise ValueError("Diagnostic capture CO2 feed must remain frozen at 11.8")
    primitives = metadata_mapping(
        diagnostic_contract, "primitive_raw_columns", "diagnostic_reconstruction_contract"
    )
    formulas = metadata_mapping(
        diagnostic_contract, "formulas", "diagnostic_reconstruction_contract"
    )
    expanded_formulas = metadata_mapping(
        diagnostic_contract, "expanded_formulas", "diagnostic_reconstruction_contract"
    )
    if primitives != DIAGNOSTIC_PRIMITIVE_RAW_COLUMNS:
        raise ValueError("Diagnostic primitive_raw_columns do not match the frozen full raw names")
    if formulas != DIAGNOSTIC_FORMULAS:
        raise ValueError("Diagnostic formulas do not match the frozen reconstruction contract")
    if expanded_formulas != DIAGNOSTIC_EXPANDED_FORMULAS:
        raise ValueError("Diagnostic expanded_formulas do not match the frozen full-name formulas")
    tolerances = metadata_mapping(
        diagnostic_contract,
        "cross_check_tolerances",
        "diagnostic_reconstruction_contract",
    )
    relative_tolerance = finite_metadata_number(
        tolerances,
        "relative",
        "diagnostic_reconstruction_contract.cross_check_tolerances",
        nonnegative=True,
    )
    absolute_tolerance = finite_metadata_number(
        tolerances,
        "absolute",
        "diagnostic_reconstruction_contract.cross_check_tolerances",
        nonnegative=True,
    )
    if relative_tolerance != 1e-9 or absolute_tolerance != 1e-12:
        raise ValueError("Diagnostic cross-check tolerances do not match the frozen contract")

    weights = metadata_mapping(metadata, "capex_proxy_weights", "metadata")
    references = metadata_mapping(metadata, "capex_proxy_references", "metadata")
    geometry = metadata_mapping(metadata, "geometry_constants_m", "metadata")
    capex_keys = [
        "capex_eco2r_area_m2",
        "capex_we_area_m2",
        "total_electric_power_kw",
        "P_Cap",
        "P_MeOH",
    ]
    if set(weights) != set(capex_keys):
        raise ValueError("CAPEX proxy weights must contain exactly the five frozen components")
    if set(references) != set(capex_keys):
        raise ValueError("CAPEX proxy references must contain exactly the five frozen components")
    parsed_weights = {
        key: finite_metadata_number(weights, key, "capex_proxy_weights", nonnegative=True)
        for key in capex_keys
    }
    parsed_references = {
        key: finite_metadata_number(references, key, "capex_proxy_references", positive=True)
        for key in capex_keys
    }
    if not math.isclose(
        math.fsum(parsed_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("CAPEX proxy weights must sum to exactly 1 within 1e-12")
    return {
        "contract_version": contract_version,
        "target_definition_sha256": source_hash,
        "operating_hours_per_year": v30_sec_hours,
        "legacy_meoh_yield_hours_per_year": legacy_hours,
        "methanol_molar_mass_kg_per_kmol": molar_mass,
        "meoh_yield_primitive_raw_column": MEOH_YIELD_PRIMITIVE_RAW_COLUMN,
        "diagnostic_feed_molar_flow": feed_molar_flow,
        "diagnostic_primitive_raw_columns": primitives,
        "diagnostic_relative_tolerance": relative_tolerance,
        "diagnostic_absolute_tolerance": absolute_tolerance,
        "weights": parsed_weights,
        "references": parsed_references,
        "eco2r_side_length_m": finite_metadata_number(
            geometry, "eCO2R_side_length", "geometry_constants_m", positive=True
        ),
        "we_side_length_m": finite_metadata_number(
            geometry, "WE_side_length", "geometry_constants_m", positive=True
        ),
    }


def validate_return_contract(frame: pd.DataFrame, issued: pd.DataFrame) -> None:
    issued_ids = set(issued["record_id"])
    returned_ids = set(frame["record_id"])
    if returned_ids != issued_ids:
        missing = sorted(issued_ids - returned_ids)
        unexpected = sorted(returned_ids - issued_ids)
        raise ValueError(f"Returned record-ID set differs from issued queue; missing={missing}, unexpected={unexpected}")
    issued_hash_by_id = issued.set_index("record_id")["input_row_sha256"]
    expected_hash = frame["record_id"].map(issued_hash_by_id)
    if not frame["input_row_sha256"].eq(expected_hash).all():
        raise ValueError("At least one returned record_id/input_row_sha256 pair differs from the issued queue")


def validate_attempt_selection(frame: pd.DataFrame, issued: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    require_columns(frame, [SELECTION_COLUMN, "converged"])
    frame["selected_final_result_parsed"] = [parse_bool(value) for value in frame[SELECTION_COLUMN]]
    frame["converged_parsed"] = [parse_bool(value) for value in frame["converged"]]
    if frame["selected_final_result_parsed"].isna().any():
        raise ValueError(f"Every attempt must explicitly state {SELECTION_COLUMN} TRUE or FALSE")
    if frame["converged_parsed"].isna().any():
        raise ValueError("Every attempt must explicitly state converged TRUE or FALSE")
    selected = frame["selected_final_result_parsed"].astype(bool)
    converged = frame["converged_parsed"].astype(bool)
    if (selected & ~converged).any():
        bad = frame.loc[selected & ~converged, "record_id"].tolist()
        raise ValueError(f"Selected final results must be successful/converged: {bad}")
    selected_counts = selected.groupby(frame["record_id"]).sum().reindex(issued["record_id"], fill_value=0)
    converged_counts = converged.groupby(frame["record_id"]).sum().reindex(
        issued["record_id"], fill_value=0
    )
    has_success = converged_counts.gt(0)
    invalid_success_selection = has_success & ~selected_counts.eq(1)
    if invalid_success_selection.any():
        bad = selected_counts.index[invalid_success_selection].tolist()
        raise ValueError(
            "Every issued record with any converged attempt requires exactly one selected "
            f"final result: {bad}"
        )
    invalid_failed_selection = ~has_success & ~selected_counts.eq(0)
    if invalid_failed_selection.any():
        bad = selected_counts.index[invalid_failed_selection].tolist()
        raise ValueError(
            "Permanently failed records must have zero selected_final_result rows: "
            f"{bad}"
        )
    return selected, converged


def validate_raw_outputs(
    frame: pd.DataFrame,
    converged: pd.Series,
    constants: dict[str, Any],
) -> None:
    primitive_columns = [
        *constants["diagnostic_primitive_raw_columns"].values(),
        constants["meoh_yield_primitive_raw_column"],
    ]
    always_required_columns = [
        *REQUIRED_RAW,
        *DIRECT_REQUIRED_DIAGNOSTICS,
        *primitive_columns,
    ]
    require_columns(frame, always_required_columns)
    supplied_masks: dict[str, pd.Series] = {}
    for column in always_required_columns:
        supplied_masks[column] = ~blank_mask(frame[column])
        frame[column] = numeric_column(frame, column, required_mask=converged)

    for column in primitive_columns:
        supplied = supplied_masks[column]
        if (supplied & frame[column].lt(0.0)).any():
            raise ValueError(f"Primitive raw output {column} must be non-negative whenever supplied")
    meoh_primitive = constants["meoh_yield_primitive_raw_column"]
    if (converged & frame[meoh_primitive].le(0.0)).any():
        raise ValueError(f"Successful attempts require positive {meoh_primitive}")

    relative_tolerance = constants["diagnostic_relative_tolerance"]
    absolute_tolerance = constants["diagnostic_absolute_tolerance"]
    legacy_yield_rebuilt = (
        frame[meoh_primitive]
        * constants["legacy_meoh_yield_hours_per_year"]
        * constants["methanol_molar_mass_kg_per_kmol"]
        / 1_000_000.0
    )
    yield_close = np.isclose(
        frame["MeOH_yield_kt_y"].to_numpy(dtype=float),
        legacy_yield_rebuilt.to_numpy(dtype=float),
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=False,
    )
    yield_mismatch = converged & ~pd.Series(yield_close, index=frame.index)
    if yield_mismatch.any():
        bad = frame.loc[yield_mismatch, "record_id"].tolist()
        raise ValueError(
            "MeOH_yield_kt_y does not match the frozen legacy 8400 h/y primitive-raw "
            f"reconstruction: {bad[:10]}"
        )
    frame["MeOH_yield_kt_y_legacy_basis_rebuilt"] = np.where(
        converged, legacy_yield_rebuilt, np.nan
    )
    frame["MeOH_yield_legacy_basis_qc"] = np.where(
        converged, "PASS_8400_H_PER_YEAR_CROSSCHECK", "NOT_APPLICABLE_FAILED_ATTEMPT"
    )

    primitive_names = constants["diagnostic_primitive_raw_columns"]
    co2_cap = frame[primitive_names["CO2_CAP"]]
    co_ce = frame[primitive_names["CO_CE"]]
    h2_we = frame[primitive_names["H2_WE"]]
    h2_ce = frame[primitive_names["H2_CE"]]
    co2_ce = frame[primitive_names["CO2_CE"]]
    feed = constants["diagnostic_feed_molar_flow"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cap_rebuilt = co2_cap / feed
        eco2r_rebuilt = co_ce / (cap_rebuilt * feed)
        s_rebuilt = (h2_we + h2_ce - co2_ce) / (co_ce + co2_ce)
    reconstructed = {
        "CAP_conversion": cap_rebuilt,
        "eCO2R_conversion": eco2r_rebuilt,
        "S": s_rebuilt,
    }
    reconstruction_frame = pd.DataFrame(reconstructed, index=frame.index)
    if not np.isfinite(reconstruction_frame.loc[converged].to_numpy(dtype=float)).all():
        raise ValueError(
            "Successful attempts produced a non-finite diagnostic reconstruction; "
            "check primitive denominators"
        )
    for column in ["CAP_conversion", "eCO2R_conversion"]:
        if not reconstruction_frame.loc[converged, column].between(
            0.0, 1.0, inclusive="both"
        ).all():
            raise ValueError(f"Rebuilt {column} must lie in [0,1] for successful attempts")

    for column in RECONSTRUCTABLE_DIAGNOSTICS:
        rebuilt = reconstructed[column]
        if column in frame.columns:
            direct_supplied = ~blank_mask(frame[column])
            direct = numeric_column(frame, column)
        else:
            direct_supplied = pd.Series(False, index=frame.index)
            direct = pd.Series(np.nan, index=frame.index, dtype=float)
        close = np.isclose(
            direct.to_numpy(dtype=float),
            rebuilt.to_numpy(dtype=float),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=False,
        )
        mismatch = converged & direct_supplied & ~pd.Series(close, index=frame.index)
        if mismatch.any():
            bad = frame.loc[mismatch, "record_id"].tolist()
            raise ValueError(
                f"Direct {column} does not match its primitive-raw reconstruction: {bad[:10]}"
            )
        frame[column] = direct.where(direct_supplied, rebuilt)
        frame[f"{column}_primitive_rebuilt"] = np.where(converged, rebuilt, np.nan)
        frame[f"{column}_diagnostic_qc"] = np.select(
            [~converged, direct_supplied],
            ["NOT_APPLICABLE_FAILED_ATTEMPT", "PASS_DIRECT_CROSSCHECKED"],
            default="PASS_REBUILT_FROM_PRIMITIVES",
        )

    for column in ["MeOH_conversion", "CAP_conversion", "eCO2R_conversion"]:
        supplied_or_rebuilt = frame[column].notna()
        outside = supplied_or_rebuilt & ~frame[column].between(0.0, 1.0, inclusive="both")
        if outside.any():
            raise ValueError(f"{column} must lie in [0,1] whenever supplied or rebuilt")
    for column in ["MeOH_yield_kt_y", "allpower_kw", "Total_ele_power_kw"]:
        supplied = supplied_masks[column]
        if (supplied & frame[column].le(0.0)).any():
            raise ValueError(f"{column} must be positive whenever supplied")

    if "failed_reason" not in frame.columns:
        frame["failed_reason"] = pd.NA
    failed = ~converged
    if (failed & blank_mask(frame["failed_reason"])).any():
        raise ValueError("Every failed attempt requires failed_reason")


def validate_input_echoes(frame: pd.DataFrame, converged: pd.Series) -> None:
    echo_columns = [echo for _, echo, _ in INPUT_ECHO_PAIRS]
    utility_echo = "echo_Utility_constant_temperature"
    require_columns(frame, [*echo_columns, utility_echo])
    echo_mismatch = np.zeros(len(frame), dtype=bool)
    all_echoes_supplied = np.ones(len(frame), dtype=bool)
    for canonical, echo, tolerance in INPUT_ECHO_PAIRS:
        supplied = ~blank_mask(frame[echo])
        observed = numeric_column(frame, echo, required_mask=converged)
        expected = frame[canonical].astype(float)
        frame[echo] = observed
        all_echoes_supplied &= supplied.to_numpy(dtype=bool)
        if tolerance is None:
            observed_is_integer = observed.eq(np.rint(observed))
            expected_is_integer = expected.eq(np.rint(expected))
            ok = observed_is_integer & expected_is_integer & observed.eq(expected)
        else:
            ok = (observed - expected).abs().le(tolerance)
        echo_mismatch |= (supplied & ~ok).to_numpy(dtype=bool)

    utility_supplied = ~blank_mask(frame[utility_echo])
    observed_utility = numeric_column(frame, utility_echo, required_mask=converged)
    frame[utility_echo] = observed_utility
    expected_utility = frame["T_utility"].astype(float) - 273.15
    utility_ok = (observed_utility - expected_utility).abs().le(1e-9)
    all_echoes_supplied &= utility_supplied.to_numpy(dtype=bool)
    echo_mismatch |= (utility_supplied & ~utility_ok).to_numpy(dtype=bool)
    if echo_mismatch.any():
        bad = frame.loc[echo_mismatch, ["record_id", "gproms_attempt_id"]].to_dict("records")
        raise ValueError(f"At least one gPROMS input echo does not exactly match its issued input: {bad[:10]}")
    frame["input_echo_qc"] = np.where(
        all_echoes_supplied,
        "PASS",
        "NOT_COMPLETE_FAILED_ATTEMPT",
    )


def validate_provenance(
    frame: pd.DataFrame,
    expected_model_version: str,
    expected_solver_settings_id: str,
    raw_root: Path,
) -> None:
    require_columns(frame, PROVENANCE_COLUMNS)
    required_text = [
        "gproms_model_version",
        "solver_settings_id",
        "execution_batch_id",
        "gproms_scenario_id",
        "gproms_attempt_id",
        "solver_status_code",
        "execution_timestamp",
        "raw_output_file",
    ]
    for column in required_text:
        frame[column] = require_text(frame[column], column)
    frame["raw_output_file_sha256"] = require_sha256(
        frame["raw_output_file_sha256"], "raw_output_file_sha256"
    )
    if frame.duplicated(["record_id", "gproms_attempt_id"]).any():
        raise ValueError("Each (record_id, gproms_attempt_id) pair must be unique")
    if not frame["gproms_model_version"].eq(expected_model_version).all():
        raise ValueError("Returned gproms_model_version is not the approved expected version")
    if not frame["solver_settings_id"].eq(expected_solver_settings_id).all():
        raise ValueError("Returned solver_settings_id is not the approved expected settings")

    required = pd.Series(True, index=frame.index)
    frame["runtime_seconds"] = numeric_column(frame, "runtime_seconds", required_mask=required)
    if frame["runtime_seconds"].lt(0.0).any():
        raise ValueError("runtime_seconds must be non-negative and finite")
    parsed_timestamp = pd.to_datetime(frame["execution_timestamp"], errors="coerce", utc=True)
    if parsed_timestamp.isna().any():
        raise ValueError("execution_timestamp must be a parseable timestamp for every attempt")

    raw_hash_status: list[str] = []
    for row in frame.itertuples(index=False):
        raw_path = Path(str(getattr(row, "raw_output_file")))
        if not raw_path.is_absolute():
            raw_path = raw_root / raw_path
        if not raw_path.is_file():
            raise ValueError(f"Raw output file not found: {raw_path}")
        expected_hash = str(getattr(row, "raw_output_file_sha256"))
        observed_hash = sha256(raw_path)
        if observed_hash != expected_hash:
            raise ValueError(f"Raw output hash mismatch: {raw_path}")
        raw_hash_status.append("PASS")
    frame["raw_output_file_hash_qc"] = raw_hash_status


def attach_issued_metadata(frame: pd.DataFrame, issued: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    issued_by_id = issued.set_index("record_id")
    for column in issued.columns:
        if column == "record_id":
            continue
        if column not in out.columns or column not in MUTABLE_RETURN_COLUMNS:
            out[column] = out["record_id"].map(issued_by_id[column])
    first_columns = list(issued.columns)
    remaining_columns = [column for column in out.columns if column not in first_columns]
    out = out[first_columns + remaining_columns]
    out["issued_metadata_qc"] = "PASS_SOURCED_FROM_ISSUED_QUEUE"
    return out


def rebuild_targets(
    frame: pd.DataFrame,
    converged: pd.Series,
    constants: dict[str, Any],
) -> None:
    hours = constants["operating_hours_per_year"]
    weights = constants["weights"]
    references = constants["references"]
    eco2r_side = constants["eco2r_side_length_m"]
    we_side = constants["we_side_length_m"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        meoh_kg_h = frame["MeOH_yield_kt_y"] * 1_000_000.0 / hours
        a_eco2r = frame["N_eCO2R"].astype(float) * eco2r_side**2
        a_we = frame["N_WE"].astype(float) * we_side**2
        sec = frame["allpower_kw"] / meoh_kg_h
        capex = (
            weights["capex_eco2r_area_m2"] * a_eco2r / references["capex_eco2r_area_m2"]
            + weights["capex_we_area_m2"] * a_we / references["capex_we_area_m2"]
            + weights["total_electric_power_kw"]
            * frame["Total_ele_power_kw"]
            / references["total_electric_power_kw"]
            + weights["P_Cap"] * frame["P_Cap"].astype(float) / references["P_Cap"]
            + weights["P_MeOH"] * frame["P_MeOH"].astype(float) / references["P_MeOH"]
        )
    rebuilt_numeric = pd.DataFrame(
        {
            "MeOH_production_kg_h_rebuilt": meoh_kg_h,
            "specific_energy_kwh_per_kg_meoh_rebuilt": sec,
            "A_eCO2R_m2_rebuilt": a_eco2r,
            "A_WE_m2_rebuilt": a_we,
            "capex_proxy_rebuilt": capex,
        }
    )
    if not np.isfinite(rebuilt_numeric.loc[converged].to_numpy(dtype=float)).all():
        raise ValueError("Frozen target reconstruction produced a non-finite value")
    frame["MeOH_production_kg_h_rebuilt"] = np.where(converged, meoh_kg_h, np.nan)
    frame["carbon_utilization_rebuilt"] = np.where(converged, frame["MeOH_conversion"], np.nan)
    frame["specific_energy_kwh_per_kg_meoh_rebuilt"] = np.where(
        converged, sec, np.nan
    )
    frame["A_eCO2R_m2_rebuilt"] = np.where(converged, a_eco2r, np.nan)
    frame["A_WE_m2_rebuilt"] = np.where(converged, a_we, np.nan)
    frame["capex_proxy_rebuilt"] = np.where(converged, capex, np.nan)
    frame["formula_qc_status"] = np.where(
        converged, "PASS_REBUILT_FROM_FROZEN_CONSTANTS", "NOT_APPLICABLE_FAILED_ATTEMPT"
    )


def build_failed_record_qc(attempt_log: pd.DataFrame, issued: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    issued_by_id = issued.set_index("record_id")
    for record_id in issued["record_id"]:
        attempts = attempt_log.loc[attempt_log["record_id"].eq(record_id)].copy()
        if attempts["converged_parsed"].astype(bool).any():
            continue
        attempts["_timestamp_utc"] = pd.to_datetime(
            attempts["execution_timestamp"], errors="raise", utc=True
        )
        attempts["_attempt_order"] = np.arange(len(attempts))
        last = attempts.sort_values(
            ["_timestamp_utc", "_attempt_order"], kind="stable"
        ).iloc[-1]
        issued_row = issued_by_id.loc[record_id]
        rows.append(
            {
                "record_id": record_id,
                "input_row_sha256": str(issued_row["input_row_sha256"]),
                "dataset_role": issued_row.get("dataset_role", pd.NA),
                "attempt_count": len(attempts),
                "converged_attempt_count": 0,
                "selected_final_result_count": int(
                    attempts["selected_final_result_parsed"].astype(bool).sum()
                ),
                "record_level_qc_status": "PERMANENT_FAILURE_ALL_ATTEMPTS_FAILED",
                "attempt_ids_semicolon": ";".join(attempts["gproms_attempt_id"].astype(str)),
                "last_gproms_attempt_id": str(last["gproms_attempt_id"]),
                "last_converged": False,
                "last_solver_status_code": str(last["solver_status_code"]),
                "last_failed_reason": str(last["failed_reason"]),
                "last_execution_timestamp": str(last["execution_timestamp"]),
                "last_execution_batch_id": str(last["execution_batch_id"]),
                "last_gproms_scenario_id": str(last["gproms_scenario_id"]),
                "last_raw_output_file": str(last["raw_output_file"]),
                "last_raw_output_file_sha256": str(last["raw_output_file_sha256"]),
                "gproms_model_version": str(last["gproms_model_version"]),
                "solver_settings_id": str(last["solver_settings_id"]),
                "all_attempt_raw_hash_qc": (
                    "PASS"
                    if attempts["raw_output_file_hash_qc"].eq("PASS").all()
                    else "FAIL"
                ),
                "all_attempt_input_echo_qc": (
                    "PASS"
                    if attempts["input_echo_qc"].eq("PASS").all()
                    else "NOT_COMPLETE_FAILED_ATTEMPT"
                ),
            }
        )
    return pd.DataFrame(rows, columns=FAILED_RECORD_QC_COLUMNS)


def finalize_return(
    *,
    input_path: Path,
    issued_queue_path: Path,
    output_dir: Path,
    metadata_path: Path,
    expected_model_version: str,
    expected_solver_settings_id: str,
    raw_root: Path,
    sheet: str | None = None,
    issued_sheet: str | None = None,
) -> dict[str, Any]:
    frame = load_table(input_path, sheet).copy()
    issued = load_table(issued_queue_path, issued_sheet).copy()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Metadata root must be a JSON object")

    validate_feature_contract(issued, "Issued queue")
    if issued["record_id"].duplicated().any() or issued["input_row_sha256"].duplicated().any():
        raise ValueError("Issued queue contains duplicate IDs or hashes")
    constants = validate_metadata(issued, metadata)
    validate_feature_contract(frame, "Returned attempt")
    validate_return_contract(frame, issued)
    _, converged = validate_attempt_selection(frame, issued)
    validate_raw_outputs(frame, converged, constants)
    validate_provenance(frame, expected_model_version, expected_solver_settings_id, raw_root)
    validate_input_echoes(frame, converged)

    attempt_log = attach_issued_metadata(frame, issued)
    attempt_log["canonical_input_string_10dp_rebuilt"] = attempt_log.apply(
        canonical_feature_string, axis=1
    )
    attempt_log["input_row_sha256_rebuilt"] = attempt_log.apply(feature_hash, axis=1)
    attempt_log["input_hash_qc"] = "PASS"
    attempt_log["source_target_definition_sha256"] = constants["target_definition_sha256"]
    attempt_log["target_definition_hash_qc"] = "PASS_METADATA_EQUALS_ISSUED"
    attempt_log["target_reconstruction_contract_version"] = constants["contract_version"]
    attempt_log["annualisation_basis_qc"] = (
        "PASS_LEGACY_YIELD_8400_AND_V30_SEC_8760_DISTINCT"
    )
    attempt_converged = attempt_log["converged_parsed"].astype(bool)
    rebuild_targets(attempt_log, attempt_converged, constants)
    attempt_log["attempt_selection_qc"] = np.select(
        [
            attempt_log["selected_final_result_parsed"].astype(bool),
            attempt_log["converged_parsed"].astype(bool),
        ],
        ["SELECTED_FINAL_SUCCESS", "PRESERVED_NONFINAL_SUCCESS"],
        default="PRESERVED_FAILED_ATTEMPT",
    )

    issued_order = pd.Series(range(len(issued)), index=issued["record_id"])
    attempt_log["_issued_order"] = attempt_log["record_id"].map(issued_order)
    attempt_log["_attempt_source_order"] = np.arange(len(attempt_log))
    attempt_log = attempt_log.sort_values(["_issued_order", "_attempt_source_order"], kind="stable").drop(
        columns=["_issued_order", "_attempt_source_order"]
    )
    accepted = attempt_log.loc[attempt_log["selected_final_result_parsed"].astype(bool)].copy()
    successful_record_count = int(
        attempt_log.groupby("record_id")["converged_parsed"].any().sum()
    )
    if (
        len(accepted) != successful_record_count
        or accepted["record_id"].duplicated().any()
        or not accepted["converged_parsed"].astype(bool).all()
    ):
        raise RuntimeError(
            "Internal error: accepted output is not one selected converged row per "
            "successful issued record"
        )
    failed_record_qc = build_failed_record_qc(attempt_log, issued)
    failed_record_count = len(failed_record_qc)
    if successful_record_count + failed_record_count != len(issued):
        raise RuntimeError("Internal error: record-level success/failure accounting is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "gproms_return_rebuilt_qc.csv"
    attempt_log_path = output_dir / "gproms_attempt_log_qc.csv"
    failed_record_qc_path = output_dir / "gproms_failed_records_qc.csv"
    attempt_log.to_csv(attempt_log_path, index=False)
    accepted.to_csv(completed_path, index=False)
    failed_record_qc.to_csv(failed_record_qc_path, index=False)
    failed_attempts = ~attempt_log["converged_parsed"].astype(bool)
    successful_attempts = ~failed_attempts
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "source_sha256": sha256(input_path),
        "issued_queue": str(issued_queue_path),
        "issued_queue_sha256": sha256(issued_queue_path),
        "rows": len(accepted),
        "issued_records": len(issued),
        "attempt_rows": len(attempt_log),
        "selected_final_results": len(accepted),
        "converged": successful_record_count,
        "failed": failed_record_count,
        "convergence_pct": (
            100.0 * successful_record_count / len(issued) if len(issued) else 0.0
        ),
        "successful_attempt_rows": int(successful_attempts.sum()),
        "failed_attempt_rows": int(failed_attempts.sum()),
        "input_hash_qc_all_pass": True,
        "input_echo_qc": {
            str(key): int(value) for key, value in attempt_log["input_echo_qc"].value_counts().items()
        },
        "expected_model_version": expected_model_version,
        "expected_solver_settings_id": expected_solver_settings_id,
        "raw_output_hash_qc_all_pass": bool(attempt_log["raw_output_file_hash_qc"].eq("PASS").all()),
        "metadata_sha256": sha256(metadata_path),
        "target_reconstruction_contract_version": constants["contract_version"],
        "legacy_meoh_yield_hours_per_year": constants[
            "legacy_meoh_yield_hours_per_year"
        ],
        "v30_sec_hours_per_year": constants["operating_hours_per_year"],
        "source_target_definition_sha256": constants["target_definition_sha256"],
        "issued_target_definition_sha256": constants["target_definition_sha256"],
        "target_definition_hash_qc": "PASS",
        "geometry_constants_m": {
            "eCO2R_side_length": constants["eco2r_side_length_m"],
            "WE_side_length": constants["we_side_length_m"],
        },
        "diagnostic_qc": {
            column: {
                str(key): int(value)
                for key, value in attempt_log[f"{column}_diagnostic_qc"]
                .value_counts(dropna=False)
                .items()
            }
            for column in RECONSTRUCTABLE_DIAGNOSTICS
        },
        "attempt_log_csv": str(attempt_log_path),
        "attempt_log_csv_sha256": sha256(attempt_log_path),
        "failed_record_qc_csv": str(failed_record_qc_path),
        "failed_record_qc_csv_sha256": sha256(failed_record_qc_path),
        "output_csv": str(completed_path),
        "output_csv_sha256": sha256(completed_path),
    }
    summary_path = output_dir / "gproms_return_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and rebuild V30 gPROMS queue returns")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--issued-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--issued-sheet", default=None)
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument("--expected-solver-settings-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize_return(
        input_path=args.input,
        issued_queue_path=args.issued_queue,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        expected_model_version=args.expected_model_version,
        expected_solver_settings_id=args.expected_solver_settings_id,
        raw_root=args.raw_root,
        sheet=args.sheet,
        issued_sheet=args.issued_sheet,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
