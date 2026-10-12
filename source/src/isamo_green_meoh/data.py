from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PipelineConfig, VariableSpec


CO2_GRID_EMISSION_FACTOR_KG_PER_KWH = 0.177
CO2_PER_MEOH_KG_STOICH = 44.0095 / 32.0419
OPERATING_HOURS_PER_YEAR = 8760.0
ECO2R_UNIT_ACTIVE_AREA_M2 = 0.175 * 0.175
WE_UNIT_ACTIVE_AREA_M2 = 1.710 * 1.710


REAL_DATA_TARGET_RENAMES = {
    "MeOH_conversion": "carbon_utilization",
    "E_sys": "legacy_energy_efficiency_ratio",
    "I_MeOH_m2_kg_h": "legacy_source_response_3",
}

REAL_DATA_FEATURE_RENAMES = {
    "Outlet_pressure_cap": "P_Cap",
    "Current_density_eCO2R_Acm2": "J_eCO2R",
    "T_eCO2R": "T_eCO2R",
    "Number_of_stacks_eCO2R": "N_eCO2R",
    "WE_Energy_input_rate_KW": "P_WE",
    "Number_of_cells_stacked_in_series_WE": "N_WE",
    "Outlet_pressure": "P_MeOH",
    "Utility_constant_temperature": "T_utility",
}

REAL_DATA_FEATURE_ORDER = [
    "P_Cap",
    "J_eCO2R",
    "T_eCO2R",
    "N_eCO2R",
    "P_WE",
    "N_WE",
    "P_MeOH",
    "T_utility",
]

REAL_DATA_SUPPLEMENT_RENAMES = {
    "CAP_conversion": "capture_efficiency",
    "eCO2R_conversion": "eco2r_conversion",
    "MeOH_yield_kt_y": "methanol_annual_production_kt_y",
    "Total_ele_power_kw": "total_electric_power_kw",
    "allpower_kw": "total_power_kw",
}

REAL_DATA_SUPPLEMENT_UNITS = {
    "capture_efficiency": "dimensionless fraction",
    "eco2r_conversion": "dimensionless fraction",
    "methanol_annual_production_kt_y": "kt/y",
    "total_electric_power_kw": "kW",
    "total_power_kw": "kW",
}


def _attach_bound_attrs(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    for spec in cfg.variables:
        df[spec.name].attrs["lower"] = spec.lower
        df[spec.name].attrs["upper"] = spec.upper


def _feature_round_dedup(df: pd.DataFrame, feature_names, ndigits: int = 8) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    round_cols = []
    for col in feature_names:
        rc = f"__r_{col}"
        tmp[rc] = tmp[col].round(ndigits)
        round_cols.append(rc)
    tmp = tmp.drop_duplicates(subset=round_cols, keep="first").drop(columns=round_cols)
    tmp.reset_index(drop=True, inplace=True)
    return tmp


def _boundary_margin_from_bounds(df: pd.DataFrame, feature_names) -> pd.Series:
    margins = []
    for col in feature_names:
        lo = float(df[col].min())
        hi = float(df[col].max())
        span = max(hi - lo, 1e-12)
        scaled = np.clip((df[col].to_numpy(dtype=float) - lo) / span, 0.0, 1.0)
        margins.append(np.minimum(scaled, 1.0 - scaled))
    return pd.Series(np.min(np.column_stack(margins), axis=1), index=df.index)


def load_real_excel_dataset(
    file_path: Union[str, Path],
    sheet_name: str = "new",
    base_cfg: Optional[PipelineConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, PipelineConfig, Dict[str, object]]:
    """
    Load the measured/new gPROMS table:
    column A = sample id, B:I = features, J:L = three objectives.

    The project keeps the downstream objective schema fixed, so the three
    measured system objectives are mapped to:
    MeOH_conversion -> carbon_utilization (maximize)
    E_sys -> legacy_energy_efficiency_ratio (retained for traceability)
    The optimization energy target is derived after merging supplementary
    methanol production and total-power columns:
    specific_energy_kwh_per_kg_meoh = total_power_kw / MeOH_kg_h.
    The absolute area target is estimated from stack counts:
    absolute_electrolyzer_area_m2 = N_eCO2R * A_eCO2R_unit + N_WE * A_WE_unit.
    """
    path = Path(file_path).expanduser()
    df_raw = pd.read_excel(path, sheet_name=sheet_name)
    if df_raw.shape[1] < 12:
        raise ValueError(f"Expected at least 12 columns in sheet {sheet_name!r}; found {df_raw.shape[1]}.")

    sample_col = df_raw.columns[0]
    original_feature_cols = list(df_raw.columns[1:9])
    target_cols = list(df_raw.columns[9:12])
    target_renames = {
        target_cols[0]: "carbon_utilization",
        target_cols[1]: "legacy_energy_efficiency_ratio",
        target_cols[2]: "legacy_source_response_3",
    }

    feature_renames = {col: REAL_DATA_FEATURE_RENAMES.get(str(col), str(col)) for col in original_feature_cols}
    feature_cols = [feature_renames[col] for col in original_feature_cols]
    keep_cols = [sample_col] + original_feature_cols + target_cols
    df = df_raw.loc[:, keep_cols].copy()
    df = df.rename(columns={sample_col: "sample_id", **feature_renames, **target_renames})
    feature_cols = [col for col in REAL_DATA_FEATURE_ORDER if col in df.columns]
    numeric_cols = feature_cols + list(target_renames.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "T_utility" in df.columns:
        df["T_utility"] = df["T_utility"] + 273.15
    for integer_col in ["N_WE", "N_eCO2R"]:
        if integer_col in df.columns:
            df[integer_col] = df[integer_col].round(0).astype(int)

    supplementary_meta: Dict[str, object] = {
        "sheet_name": "sup",
        "available": False,
        "columns_merged": [],
    }
    try:
        sup_raw = pd.read_excel(path, sheet_name="sup")
        if sup_raw.shape[1] >= 2:
            sup_sample_col = sup_raw.columns[0]
            sup_cols = [col for col in REAL_DATA_SUPPLEMENT_RENAMES if col in sup_raw.columns]
            if sup_cols:
                sup_df = sup_raw.loc[:, [sup_sample_col] + sup_cols].copy()
                sup_df = sup_df.rename(
                    columns={
                        sup_sample_col: "sample_id",
                        **{col: REAL_DATA_SUPPLEMENT_RENAMES[col] for col in sup_cols},
                    }
                )
                for col in [REAL_DATA_SUPPLEMENT_RENAMES[c] for c in sup_cols]:
                    sup_df[col] = pd.to_numeric(sup_df[col], errors="coerce")
                df = df.merge(sup_df, on="sample_id", how="left", validate="one_to_one")
                numeric_cols.extend([REAL_DATA_SUPPLEMENT_RENAMES[c] for c in sup_cols])
                supplementary_meta = {
                    "sheet_name": "sup",
                    "available": True,
                    "source_shape": [int(sup_raw.shape[0]), int(sup_raw.shape[1])],
                    "sample_id_column": str(sup_sample_col),
                    "source_column_mapping": {str(c): REAL_DATA_SUPPLEMENT_RENAMES[c] for c in sup_cols},
                    "columns_merged": [REAL_DATA_SUPPLEMENT_RENAMES[c] for c in sup_cols],
                    "units": {
                        REAL_DATA_SUPPLEMENT_RENAMES[c]: REAL_DATA_SUPPLEMENT_UNITS.get(
                            REAL_DATA_SUPPLEMENT_RENAMES[c], "-"
                        )
                        for c in sup_cols
                    },
                }
    except ValueError:
        supplementary_meta["note"] = "supplementary sheet 'sup' not found"

    required_emissions_cols = [
        "carbon_utilization",
        "methanol_annual_production_kt_y",
        "total_electric_power_kw",
        "total_power_kw",
        "N_eCO2R",
        "N_WE",
    ]
    if all(col in df.columns for col in required_emissions_cols):
        carbon_conversion = df["carbon_utilization"].astype(float).clip(lower=1e-12, upper=0.999999)
        methanol_kg_h = (
            df["methanol_annual_production_kt_y"].astype(float)
            * 1_000_000.0
            / OPERATING_HOURS_PER_YEAR
        ).clip(lower=1e-12)
        inferred_flue_co2_feed_kg_h = float(np.median(methanol_kg_h * CO2_PER_MEOH_KG_STOICH / carbon_conversion))
        df["carbon_utilization_from_meoh_yield"] = (
            methanol_kg_h * CO2_PER_MEOH_KG_STOICH / max(inferred_flue_co2_feed_kg_h, 1e-12)
        ).clip(lower=0.0, upper=0.999999)
        unconverted_co2_kg_per_kg_meoh = CO2_PER_MEOH_KG_STOICH * (1.0 / carbon_conversion - 1.0)
        electricity_co2_kg_per_kg_meoh = (
            df["total_electric_power_kw"].astype(float) * CO2_GRID_EMISSION_FACTOR_KG_PER_KWH / methanol_kg_h
        )
        df["inferred_flue_co2_feed_kg_h"] = inferred_flue_co2_feed_kg_h
        df["unconverted_co2_kg_per_kg_meoh"] = unconverted_co2_kg_per_kg_meoh
        df["electricity_co2e_kg_per_kg_meoh"] = electricity_co2_kg_per_kg_meoh
        df["co2_emissions_kg_per_kg_meoh"] = unconverted_co2_kg_per_kg_meoh + electricity_co2_kg_per_kg_meoh
        df["specific_energy_kwh_per_kg_meoh"] = df["total_power_kw"].astype(float) / methanol_kg_h
        df["capex_eco2r_area_m2"] = df["N_eCO2R"].astype(float) * ECO2R_UNIT_ACTIVE_AREA_M2
        df["capex_we_area_m2"] = df["N_WE"].astype(float) * WE_UNIT_ACTIVE_AREA_M2
        df["absolute_electrolyzer_area_m2"] = df["capex_eco2r_area_m2"] + df["capex_we_area_m2"]
        capex_refs = {
            "capex_eco2r_area_m2": float(np.median(df["capex_eco2r_area_m2"].astype(float))),
            "capex_we_area_m2": float(np.median(df["capex_we_area_m2"].astype(float))),
            "total_electric_power_kw": float(np.median(df["total_electric_power_kw"].astype(float))),
            "P_Cap": float(np.median(df["P_Cap"].astype(float))),
            "P_MeOH": float(np.median(df["P_MeOH"].astype(float))),
        }
        capex_weights = {
            "capex_eco2r_area_m2": 0.35,
            "capex_we_area_m2": 0.25,
            "total_electric_power_kw": 0.20,
            "P_Cap": 0.10,
            "P_MeOH": 0.10,
        }
        df["capex_proxy"] = (
            capex_weights["capex_eco2r_area_m2"]
            * df["capex_eco2r_area_m2"].astype(float)
            / max(capex_refs["capex_eco2r_area_m2"], 1e-12)
            + capex_weights["capex_we_area_m2"]
            * df["capex_we_area_m2"].astype(float)
            / max(capex_refs["capex_we_area_m2"], 1e-12)
            + capex_weights["total_electric_power_kw"]
            * df["total_electric_power_kw"].astype(float)
            / max(capex_refs["total_electric_power_kw"], 1e-12)
            + capex_weights["P_Cap"] * df["P_Cap"].astype(float) / max(capex_refs["P_Cap"], 1e-12)
            + capex_weights["P_MeOH"] * df["P_MeOH"].astype(float) / max(capex_refs["P_MeOH"], 1e-12)
        )
        numeric_cols.extend(
            [
                "carbon_utilization_from_meoh_yield",
                "inferred_flue_co2_feed_kg_h",
                "unconverted_co2_kg_per_kg_meoh",
                "electricity_co2e_kg_per_kg_meoh",
                "co2_emissions_kg_per_kg_meoh",
                "specific_energy_kwh_per_kg_meoh",
                "capex_eco2r_area_m2",
                "capex_we_area_m2",
                "absolute_electrolyzer_area_m2",
                "capex_proxy",
            ]
        )
    else:
        missing = [col for col in required_emissions_cols if col not in df.columns]
        raise ValueError(
            "Cannot compute co2_emissions_kg_per_kg_meoh; missing columns after preprocessing: "
            + ", ".join(missing)
        )

    before_drop = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    dropped_rows = int(before_drop - len(df))
    if df.empty:
        raise ValueError("No complete numeric rows remained after preprocessing.")

    cfg = base_cfg or PipelineConfig()
    feature_units = {
        "P_Cap": "bar",
        "J_eCO2R": "A/cm2",
        "T_eCO2R": "K",
        "N_eCO2R": "count",
        "P_WE": "MW",
        "N_WE": "count",
        "P_MeOH": "bar",
        "T_utility": "K",
    }
    cfg.variables = [
        VariableSpec(
            name=str(col),
            unit=feature_units.get(str(col), "-"),
            lower=float(df[col].min()),
            upper=float(df[col].max()),
        )
        for col in feature_cols
    ]
    cfg.data.target_converged_samples = int(len(df))
    cfg.data.initial_sobol_target = int(len(df))
    cfg.data.raw_sobol_samples = int(len(df))
    cfg.data.pareto_neighbour_rounds = 0
    cfg.data.pareto_validation_sample_size = 0
    cfg.data.max_train_pool_size = int(len(df))
    cfg.surrogate.prefer_xgboost = True
    cfg.surrogate.k_folds = min(int(cfg.surrogate.k_folds), 3)
    cfg.surrogate.max_iter = max(int(cfg.surrogate.max_iter), 1200)
    cfg.surrogate.xgb_n_estimators = max(int(cfg.surrogate.xgb_n_estimators), 1800)
    setattr(cfg, "data_source", "real_excel")
    setattr(cfg, "input_data_path", str(path))
    setattr(cfg, "input_sheet_name", str(sheet_name))

    df["converged"] = True
    df["sample_source"] = "real_excel_new"
    df["al_round"] = -1
    df["boundary_margin"] = _boundary_margin_from_bounds(df, feature_cols)
    _attach_bound_attrs(df, cfg)

    meta: Dict[str, object] = {
        "source_file": str(path),
        "sheet_name": str(sheet_name),
        "source_shape": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "rows_used": int(len(df)),
        "rows_dropped_missing_or_non_numeric": dropped_rows,
        "sample_id_column": str(sample_col),
        "original_feature_columns": [str(c) for c in original_feature_cols],
        "feature_column_mapping": {str(k): str(v) for k, v in feature_renames.items()},
        "feature_columns": [str(c) for c in feature_cols],
        "target_column_mapping": {str(k): str(v) for k, v in target_renames.items()},
        "derived_target_mapping": {
            "specific_energy_kwh_per_kg_meoh": "allpower_kw / (MeOH_yield_kt_y * 1e6 / 8760)",
            "absolute_electrolyzer_area_m2": "N_eCO2R * (0.175*0.175) + N_WE * (1.710*1.710)",
            "capex_proxy": (
                "0.35*(A_eCO2R/median(A_eCO2R)) + 0.25*(A_WE/median(A_WE)) "
                "+ 0.20*(Total_ele_power_kw/median(Total_ele_power_kw)) "
                "+ 0.10*(P_Cap/median(P_Cap)) + 0.10*(P_MeOH/median(P_MeOH))"
            ),
            "co2_emissions_kg_per_kg_meoh": (
                "CO2_PER_MEOH_STOICH * (1 / MeOH_conversion - 1) "
                "+ Total_ele_power_kw * 0.177 / (MeOH_yield_kt_y * 1e6 / 8760)"
            ),
            "carbon_utilization": "Equivalent to methanol-yield carbon flow divided by inferred constant flue-gas CO2 feed.",
        },
        "capex_proxy_weights": capex_weights if "capex_weights" in locals() else None,
        "capex_proxy_references": capex_refs if "capex_refs" in locals() else None,
        "emissions_factor_kg_co2e_per_kwh": CO2_GRID_EMISSION_FACTOR_KG_PER_KWH,
        "co2_per_meoh_kg_stoich": CO2_PER_MEOH_KG_STOICH,
        "operating_hours_per_year": OPERATING_HOURS_PER_YEAR,
        "inferred_flue_co2_feed_kg_h": float(df["inferred_flue_co2_feed_kg_h"].iloc[0])
        if "inferred_flue_co2_feed_kg_h" in df
        else None,
        "feature_units": feature_units,
        "feature_descriptions": {
            "P_Cap": "membrane-separation carbon-capture pressure",
            "J_eCO2R": "eCO2R current density",
            "T_eCO2R": "eCO2R temperature",
            "N_eCO2R": "number of eCO2R stacks",
            "P_WE": "water electrolyser power input",
            "N_WE": "number of WE cells stacked in series",
            "P_MeOH": "methanol reactor feed pressure",
            "T_utility": "methanol reactor utility temperature converted from degC to K",
        },
        "target_units": {
            "MeOH_conversion": "dimensionless fraction",
            "E_sys": "legacy dimensionless ratio of methanol LHV energy output to total system energy input",
            "specific_energy_kwh_per_kg_meoh": "kWh/kg MeOH",
            "absolute_electrolyzer_area_m2": "m2",
            "capex_proxy": "dimensionless normalized equipment/process-scale proxy",
            "co2_emissions_kg_per_kg_meoh": "kg CO2e/kg MeOH",
            "I_MeOH_m2_kg_h": "legacy m2/(kg/h), retained for traceability but not optimized",
        },
        "target_directions": {
            "MeOH_conversion": "maximize",
            "specific_energy_kwh_per_kg_meoh": "minimize",
            "capex_proxy": "minimize",
        },
        "supplementary_data": supplementary_meta,
        "preprocessing": "numeric coercion, complete-case filtering, T_utility converted from degC to K, N_WE and N_eCO2R rounded to nearest integer, feature StandardScaler in model pipelines, optimization bounds from observed feature min/max",
    }
    return df.copy(), df.copy(), cfg, meta


def split_dataset(df: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        train_size=cfg.data.train_fraction,
        random_state=cfg.data.seed,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return {"train": train_df, "test": test_df}


def save_datasets(raw_df: pd.DataFrame, conv_df: pd.DataFrame, splits: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "raw_sobol_simulation.csv", index=False)
    conv_df.to_csv(out_dir / "converged_dataset.csv", index=False)
    splits["train"].to_csv(out_dir / "train.csv", index=False)
    splits["test"].to_csv(out_dir / "test.csv", index=False)
