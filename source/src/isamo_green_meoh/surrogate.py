from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import PipelineConfig


PROCESS_PHYSICS_TARGETS = {
    "carbon_utilization",
    "specific_energy_kwh_per_kg_meoh",
    "capex_proxy",
    "methanol_annual_production_kt_y",
    "total_electric_power_kw",
}


def _is_xgboost_model(model: Any) -> bool:
    return model.__class__.__module__.startswith("xgboost")


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str = "rich"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = [X, X**2]
        pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 6),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 3),
            (2, 6),
            (2, 7),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        feats.append(np.column_stack([X[:, i] * X[:, j] for i, j in pairs]))
        if self.mode == "rich":
            eps = 1e-9
            ratio_pairs = [(1, 3), (1, 4), (3, 5), (4, 5), (6, 0), (7, 2)]
            feats.append(np.column_stack([X[:, i] / (np.abs(X[:, j]) + eps) for i, j in ratio_pairs]))
            feats.append(np.log(np.abs(X) + 1.0))
        return np.column_stack(feats)


class ProcessPhysicsFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Adds physically motivated interactions shared by the six V21 response models.

    Feature order is:
    P_Cap, J_eCO2R, T_eCO2R, N_eCO2R, P_WE, N_WE, P_MeOH, T_utility.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        eps = 1e-9
        p_cap = X[:, 0]
        j = X[:, 1]
        t_eco2r = X[:, 2]
        n_eco2r = X[:, 3]
        p_we = X[:, 4]
        n_we = X[:, 5]
        p_meoh = X[:, 6]
        t_utility = X[:, 7]
        physics = np.column_stack(
            [
                j * n_eco2r,
                p_we / (n_we + eps),
                p_cap / (p_meoh + eps),
                t_utility - t_eco2r,
                j * p_we,
                n_eco2r / (n_we + eps),
                p_meoh - p_cap,
                (t_utility - t_eco2r) * j,
                p_we / (j * n_eco2r + eps),
                (p_meoh / (p_cap + eps)) * j,
            ]
        )
        return np.column_stack([X, physics])


# Retain this module attribute so previously saved V21 joblib models remain loadable.
CarbonPhysicsFeatureAugmenter = ProcessPhysicsFeatureAugmenter


class V23PhysicsFeatureAugmenter(BaseEstimator, TransformerMixin):
    """Append the nine leakage-safe V23 physics-informed features.

    The eight raw inputs are retained in their original order. All reference
    values are medians estimated in ``fit`` so a cloned sklearn pipeline learns
    them from the current training/calibration/CV fold only. ``transform``
    therefore returns exactly 17 columns: eight raw plus nine PI features.
    """

    n_features_in_: int

    def fit(self, X, y=None):
        X = self._validated_raw_inputs(X)
        self.n_features_in_ = int(X.shape[1])
        p_cap, j, t_eco2r, n_eco2r, p_we, n_we, p_meoh, t_utility = X.T
        load = j * n_eco2r
        hydrogen_load = p_we / n_we
        relative_scale = n_eco2r / n_we
        compression = np.log(p_meoh / p_cap)
        pressure_lift = p_meoh - p_cap
        thermal_lift = t_utility - t_eco2r
        references = {
            "j_ref_": np.median(j),
            "load_ref_": np.median(load),
            "hydrogen_load_ref_": np.median(hydrogen_load),
            "relative_scale_ref_": np.median(relative_scale),
            "compression_ref_": np.median(compression),
            "pressure_lift_ref_": np.median(pressure_lift),
            "thermal_lift_ref_": np.median(thermal_lift),
        }
        for name, value in references.items():
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid V23 PI reference {name}={value!r}")
            setattr(self, name, value)
        return self

    @staticmethod
    def _validated_raw_inputs(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError(
                f"V23 PI requires an (n, 8) raw-input matrix; received {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError("V23 PI inputs must all be finite")
        p_cap = X[:, 0]
        j = X[:, 1]
        n_eco2r = X[:, 3]
        p_we = X[:, 4]
        n_we = X[:, 5]
        p_meoh = X[:, 6]
        positive = np.column_stack([p_cap, j, n_eco2r, p_we, n_we, p_meoh])
        if np.any(positive <= 0.0):
            raise ValueError("V23 PI ratio and logarithm inputs must be strictly positive")
        if np.any(p_meoh <= p_cap):
            raise ValueError("V23 PI requires P_MeOH > P_Cap")
        if np.any(X[:, 7] <= X[:, 2]):
            raise ValueError("V23 PI requires T_utility > T_eCO2R")
        return X

    def transform(self, X):
        required = (
            "j_ref_",
            "load_ref_",
            "hydrogen_load_ref_",
            "relative_scale_ref_",
            "compression_ref_",
            "pressure_lift_ref_",
            "thermal_lift_ref_",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "V23PhysicsFeatureAugmenter must be fitted before transform; missing "
                + ", ".join(missing)
            )
        X = self._validated_raw_inputs(X)
        p_cap, j, t_eco2r, n_eco2r, p_we, n_we, p_meoh, t_utility = X.T
        load_star = (j * n_eco2r) / self.load_ref_
        hydrogen_star = (p_we / n_we) / self.hydrogen_load_ref_
        balance_log = np.log(hydrogen_star / load_star)
        relative_scale_log = np.log(
            (n_eco2r / n_we) / self.relative_scale_ref_
        )
        compression_star = np.log(p_meoh / p_cap) / self.compression_ref_
        pressure_lift_star = (p_meoh - p_cap) / self.pressure_lift_ref_
        thermal_lift_star = (t_utility - t_eco2r) / self.thermal_lift_ref_
        centred_j = j / self.j_ref_ - 1.0
        electrothermal = centred_j * (thermal_lift_star - 1.0)
        electropressure = centred_j * (compression_star - 1.0)
        physics = np.column_stack(
            [
                load_star,
                hydrogen_star,
                balance_log,
                relative_scale_log,
                compression_star,
                pressure_lift_star,
                thermal_lift_star,
                electrothermal,
                electropressure,
            ]
        )
        transformed = np.column_stack([X, physics])
        if transformed.shape[1] != 17 or not np.isfinite(transformed).all():
            raise ValueError("V23 PI transformation did not produce 17 finite columns")
        return transformed


class V23ReducedPhysicsFeatureAugmenter(V23PhysicsFeatureAugmenter):
    """Return the full-rank V23 representation used by new model training.

    The legacy :class:`V23PhysicsFeatureAugmenter` is intentionally retained
    unchanged so V26 joblib files trained with 17 columns remain loadable. New
    models remove ``dP_star`` and ``dT_star`` because, with raw-input
    passthrough, they are exact linear combinations of ``P_MeOH``/``P_Cap``
    and ``T_utility``/``T_eCO2R``. The remaining representation contains eight
    raw variables plus seven physics-informed features (15 columns total).
    """

    def transform(self, X):
        transformed = super().transform(X)
        # Full V23 order: 8 raw, L*, H*, ln B*, ln N*, Pi*, dP*, dT*, E_T*, E_P*.
        reduced = np.delete(transformed, [13, 14], axis=1)
        if reduced.shape[1] != 15 or not np.isfinite(reduced).all():
            raise ValueError("Reduced V23 PI transformation did not produce 15 finite columns")
        return reduced


def _make_histgb(cfg: PipelineConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=cfg.surrogate.learning_rate,
        max_depth=cfg.surrogate.max_depth,
        max_iter=cfg.surrogate.max_iter,
        min_samples_leaf=cfg.surrogate.min_samples_leaf,
        l2_regularization=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=cfg.surrogate.random_state,
    )


def _make_histgb_high_accuracy(cfg: PipelineConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=min(cfg.surrogate.learning_rate, 0.02),
        max_depth=max(cfg.surrogate.max_depth, 9),
        max_iter=max(cfg.surrogate.max_iter, 2500),
        min_samples_leaf=3,
        l2_regularization=1e-6,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=35,
        random_state=cfg.surrogate.random_state,
    )


def _make_extra_trees(cfg: PipelineConfig, target: str) -> ExtraTreesRegressor:
    min_leaf = 1 if target == "carbon_utilization" else 2
    return ExtraTreesRegressor(
        n_estimators=1800,
        max_features=1.0,
        min_samples_leaf=min_leaf,
        bootstrap=False,
        random_state=cfg.surrogate.random_state,
        n_jobs=cfg.surrogate.xgb_n_jobs,
    )


def _make_random_forest(cfg: PipelineConfig, target: str) -> RandomForestRegressor:
    min_leaf = 1 if target == "carbon_utilization" else 2
    return RandomForestRegressor(
        n_estimators=650,
        max_features=1.0,
        min_samples_leaf=min_leaf,
        bootstrap=True,
        random_state=cfg.surrogate.random_state + 17,
        n_jobs=cfg.surrogate.xgb_n_jobs,
    )


def _with_feature_scaler(model: Any) -> SklearnPipeline:
    return SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _make_regressor(cfg: PipelineConfig) -> Tuple[Any, str, str]:
    """
    Returns (model, backend_name, backend_note).
    XGBoost is preferred but may fail on macOS if libomp is unavailable.
    """
    if cfg.surrogate.prefer_xgboost:
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=cfg.surrogate.xgb_n_estimators,
                learning_rate=cfg.surrogate.learning_rate,
                max_depth=cfg.surrogate.max_depth,
                min_child_weight=max(1, cfg.surrogate.min_samples_leaf // 2),
                subsample=cfg.surrogate.xgb_subsample,
                colsample_bytree=cfg.surrogate.xgb_colsample_bytree,
                reg_lambda=cfg.surrogate.xgb_reg_lambda,
                random_state=cfg.surrogate.random_state,
                n_jobs=cfg.surrogate.xgb_n_jobs,
                tree_method="hist",
                eval_metric="rmse",
                early_stopping_rounds=cfg.surrogate.xgb_early_stopping_rounds,
            )
            return model, "xgboost", "xgboost.XGBRegressor"
        except Exception as e:
            note = f"xgboost unavailable ({type(e).__name__}): {str(e).splitlines()[0]}"
            return _with_feature_scaler(_make_histgb(cfg)), "sklearn_histgb_scaled_fallback", note

    return _with_feature_scaler(_make_histgb(cfg)), "sklearn_histgb_scaled", "StandardScaler + sklearn.HistGradientBoostingRegressor"


def _candidate_catalog(target: str) -> List[Dict[str, str]]:
    """Return every candidate configuration eligible for a target.

    Optional candidates remain in the catalog even when their dependency is
    unavailable. This lets the exported audit table distinguish an unavailable
    learner from a learner that fitted successfully but ranked outside the
    retained ensemble.
    """

    rows: List[Dict[str, str]] = [
        {
            "candidate": "extra_trees",
            "model_family": "ExtraTreesRegressor",
            "feature_representation": "raw",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "random_forest",
            "model_family": "RandomForestRegressor",
            "feature_representation": "raw",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "histgb_scaled",
            "model_family": "HistGradientBoostingRegressor",
            "feature_representation": "standardized_raw",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "rich_extra_trees",
            "model_family": "ExtraTreesRegressor",
            "feature_representation": "rich_nonlinear",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "poly2_extra_trees",
            "model_family": "ExtraTreesRegressor",
            "feature_representation": "polynomial_degree_2",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "rich_histgb",
            "model_family": "HistGradientBoostingRegressor",
            "feature_representation": "rich_nonlinear",
            "required_dependency": "scikit-learn",
            "scope": "all_targets",
        },
        {
            "candidate": "xgboost_depth4",
            "model_family": "XGBRegressor",
            "feature_representation": "raw",
            "required_dependency": "xgboost",
            "scope": "all_targets",
        },
        {
            "candidate": "xgboost_depth6",
            "model_family": "XGBRegressor",
            "feature_representation": "raw",
            "required_dependency": "xgboost",
            "scope": "all_targets",
        },
        {
            "candidate": "rich_xgboost_depth4",
            "model_family": "XGBRegressor",
            "feature_representation": "rich_nonlinear",
            "required_dependency": "xgboost",
            "scope": "all_targets",
        },
    ]
    if target in PROCESS_PHYSICS_TARGETS:
        rows.extend(
            [
                {
                    "candidate": "process_physics_extra_trees",
                    "model_family": "ExtraTreesRegressor",
                    "feature_representation": "v23_reduced_pi_8_raw_plus_7_physics",
                    "required_dependency": "scikit-learn",
                    "scope": "all_process_targets",
                },
                {
                    "candidate": "process_physics_histgb",
                    "model_family": "HistGradientBoostingRegressor",
                    "feature_representation": "v23_reduced_pi_8_raw_plus_7_physics",
                    "required_dependency": "scikit-learn",
                    "scope": "all_process_targets",
                },
                {
                    "candidate": "process_physics_xgboost",
                    "model_family": "XGBRegressor",
                    "feature_representation": "v23_reduced_pi_8_raw_plus_7_physics",
                    "required_dependency": "xgboost",
                    "scope": "all_process_targets",
                },
                {
                    "candidate": "process_physics_stacking",
                    "model_family": "StackingRegressor",
                    "feature_representation": "v23_reduced_pi_stacking_with_raw_passthrough",
                    "required_dependency": "xgboost,scikit-learn",
                    "scope": "all_process_targets",
                },
            ]
        )
    if target == "co2_emissions_kg_per_kg_meoh":
        rows.append(
            {
                "candidate": "nystroem_ridge",
                "model_family": "Nystroem+RidgeCV",
                "feature_representation": "rbf_kernel_approximation",
                "required_dependency": "scikit-learn",
                "scope": "co2_emissions_only",
            }
        )
    rows.append(
        {
            "candidate": "lightgbm",
            "model_family": "LGBMRegressor",
            "feature_representation": "raw",
            "required_dependency": "lightgbm",
            "scope": "all_targets",
        }
    )
    for order, row in enumerate(rows, start=1):
        row["candidate_order"] = str(order)
    return rows


def _make_candidate_regressors(cfg: PipelineConfig, target: str) -> List[Tuple[str, Any, str]]:
    candidates: List[Tuple[str, Any, str]] = [
        ("extra_trees", _make_extra_trees(cfg, target), "sklearn.ExtraTreesRegressor"),
        ("random_forest", _make_random_forest(cfg, target), "sklearn.RandomForestRegressor"),
        (
            "histgb_scaled",
            _with_feature_scaler(_make_histgb_high_accuracy(cfg)),
            "StandardScaler + sklearn.HistGradientBoostingRegressor",
        ),
        (
            "rich_extra_trees",
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("features", FeatureAugmenter("rich")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=2200 if target == "carbon_utilization" else 1400,
                            max_features=1.0,
                            min_samples_leaf=1 if target == "carbon_utilization" else 2,
                            bootstrap=False,
                            random_state=cfg.surrogate.random_state + 101,
                            n_jobs=cfg.surrogate.xgb_n_jobs,
                        ),
                    ),
                ]
            ),
            "rich features + sklearn.ExtraTreesRegressor",
        ),
        (
            "poly2_extra_trees",
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=1800 if target == "carbon_utilization" else 1200,
                            max_features=1.0,
                            min_samples_leaf=1 if target == "carbon_utilization" else 2,
                            bootstrap=False,
                            random_state=cfg.surrogate.random_state + 102,
                            n_jobs=cfg.surrogate.xgb_n_jobs,
                        ),
                    ),
                ]
            ),
            "degree-2 polynomial features + sklearn.ExtraTreesRegressor",
        ),
        (
            "rich_histgb",
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("features", FeatureAugmenter("rich")),
                    ("model", _make_histgb_high_accuracy(cfg)),
                ]
            ),
            "rich features + sklearn.HistGradientBoostingRegressor",
        ),
    ]

    try:
        from xgboost import XGBRegressor

        xgb_common = {
            "objective": "reg:squarederror",
            "learning_rate": min(cfg.surrogate.learning_rate, 0.035),
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_lambda": 0.8,
            "random_state": cfg.surrogate.random_state,
            "n_jobs": cfg.surrogate.xgb_n_jobs,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "early_stopping_rounds": max(80, cfg.surrogate.xgb_early_stopping_rounds),
        }
        candidates.append(
            (
                "xgboost_depth4",
                XGBRegressor(
                    n_estimators=max(cfg.surrogate.xgb_n_estimators, 1800),
                    max_depth=4,
                    min_child_weight=1,
                    **xgb_common,
                ),
                "xgboost.XGBRegressor depth=4",
            )
        )
        candidates.append(
            (
                "xgboost_depth6",
                XGBRegressor(
                    n_estimators=max(cfg.surrogate.xgb_n_estimators, 1600),
                    max_depth=6,
                    min_child_weight=2,
                    reg_alpha=1e-4,
                    **xgb_common,
                ),
                "xgboost.XGBRegressor depth=6",
            )
        )
        candidates.append(
            (
                "rich_xgboost_depth4",
                SklearnPipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("features", FeatureAugmenter("rich")),
                        (
                            "model",
                            XGBRegressor(
                                objective="reg:squarederror",
                                n_estimators=max(cfg.surrogate.xgb_n_estimators, 2200),
                                learning_rate=min(cfg.surrogate.learning_rate, 0.015),
                                max_depth=4,
                                min_child_weight=1,
                                subsample=0.98,
                                colsample_bytree=0.98,
                                reg_lambda=0.25,
                                random_state=cfg.surrogate.random_state + 103,
                                n_jobs=cfg.surrogate.xgb_n_jobs,
                                tree_method="hist",
                                eval_metric="rmse",
                            ),
                        ),
                    ]
                ),
                "rich features + xgboost.XGBRegressor depth=4",
            )
        )
    except Exception:
        pass

    if target in PROCESS_PHYSICS_TARGETS:
        candidates.extend(
            [
                (
                    "process_physics_extra_trees",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                ExtraTreesRegressor(
                                    n_estimators=2200,
                                    max_features=1.0,
                                    min_samples_leaf=1,
                                    bootstrap=False,
                                    random_state=cfg.surrogate.random_state + 301,
                                    n_jobs=cfg.surrogate.xgb_n_jobs,
                                ),
                            ),
                        ]
                    ),
                    "process physics features + sklearn.ExtraTreesRegressor",
                ),
                (
                    "process_physics_histgb",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                HistGradientBoostingRegressor(
                                    max_iter=2600,
                                    learning_rate=0.014,
                                    max_depth=9,
                                    min_samples_leaf=3,
                                    l2_regularization=1e-7,
                                    early_stopping=True,
                                    validation_fraction=0.1,
                                    n_iter_no_change=60,
                                    random_state=cfg.surrogate.random_state + 302,
                                ),
                            ),
                        ]
                    ),
                    "process physics features + sklearn.HistGradientBoostingRegressor",
                ),
            ]
        )
        try:
            from xgboost import XGBRegressor

            candidates.append(
                (
                    "process_physics_xgboost",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                XGBRegressor(
                                    objective="reg:squarederror",
                                    n_estimators=2200,
                                    learning_rate=0.014,
                                    max_depth=4,
                                    min_child_weight=1,
                                    subsample=0.98,
                                    colsample_bytree=0.98,
                                    reg_lambda=0.25,
                                    random_state=cfg.surrogate.random_state + 303,
                                    n_jobs=cfg.surrogate.xgb_n_jobs,
                                    tree_method="hist",
                                    eval_metric="rmse",
                                ),
                            ),
                        ]
                    ),
                    "process physics features + xgboost.XGBRegressor depth=4",
                )
            )
        except Exception:
            pass

        try:
            from xgboost import XGBRegressor

            stack_estimators = [
                (
                    "et",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                ExtraTreesRegressor(
                                    n_estimators=1000,
                                    max_features=1.0,
                                    min_samples_leaf=1,
                                    bootstrap=False,
                                    random_state=cfg.surrogate.random_state + 201,
                                    n_jobs=cfg.surrogate.xgb_n_jobs,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "histgb",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                HistGradientBoostingRegressor(
                                    max_iter=2200,
                                    learning_rate=0.015,
                                    max_depth=10,
                                    min_samples_leaf=2,
                                    l2_regularization=1e-8,
                                    early_stopping=True,
                                    n_iter_no_change=60,
                                    random_state=cfg.surrogate.random_state + 202,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "xgb",
                    SklearnPipeline(
                        [
                            ("physics", V23ReducedPhysicsFeatureAugmenter()),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                XGBRegressor(
                                    objective="reg:squarederror",
                                    n_estimators=1800,
                                    learning_rate=0.015,
                                    max_depth=4,
                                    min_child_weight=1,
                                    subsample=0.98,
                                    colsample_bytree=0.98,
                                    reg_lambda=0.25,
                                    random_state=cfg.surrogate.random_state + 203,
                                    n_jobs=cfg.surrogate.xgb_n_jobs,
                                    tree_method="hist",
                                    eval_metric="rmse",
                                ),
                            ),
                        ]
                    ),
                ),
            ]
            candidates.append(
                (
                    "process_physics_stacking",
                    StackingRegressor(
                        estimators=stack_estimators,
                        final_estimator=RidgeCV(alphas=np.logspace(-6, 2, 10)),
                        cv=3,
                        passthrough=True,
                    ),
                    "stacking process-physics ExtraTrees/HistGB/XGBoost with raw passthrough features",
                )
            )
        except Exception:
            pass

    if target == "co2_emissions_kg_per_kg_meoh":
        candidates.append(
            (
                "nystroem_ridge",
                SklearnPipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("kernel", Nystroem(kernel="rbf", gamma=0.08, n_components=1200, random_state=cfg.surrogate.random_state)),
                        ("model", RidgeCV(alphas=np.logspace(-8, 2, 12))),
                    ]
                ),
                "RBF Nystroem features + RidgeCV",
            )
        )

    try:
        from lightgbm import LGBMRegressor

        candidates.append(
            (
                "lightgbm",
                LGBMRegressor(
                    n_estimators=1600,
                    learning_rate=min(cfg.surrogate.learning_rate, 0.035),
                    num_leaves=63,
                    min_child_samples=8,
                    subsample=0.95,
                    colsample_bytree=0.95,
                    reg_lambda=0.5,
                    random_state=cfg.surrogate.random_state + 31,
                    n_jobs=cfg.surrogate.xgb_n_jobs,
                    verbose=-1,
                ),
                "lightgbm.LGBMRegressor",
            )
        )
    except Exception:
        pass

    return candidates


def candidate_library_status(cfg: PipelineConfig, target_names: List[str]) -> pd.DataFrame:
    """Report whether every catalogued surrogate can be constructed.

    This is a dependency/constructor preflight, deliberately performed before
    model calibration. It prevents optional libraries such as XGBoost or
    LightGBM from being silently omitted from a new run.
    """

    package_versions: Dict[str, str] = {}
    for package_name in ("scikit-learn", "xgboost", "lightgbm"):
        try:
            package_versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            package_versions[package_name] = "not_installed"

    rows: List[Dict[str, Any]] = []
    for target in dict.fromkeys(target_names):
        constructed = {
            name: (model, note)
            for name, model, note in _make_candidate_regressors(cfg, target)
        }
        for catalog_row in _candidate_catalog(target):
            name = str(catalog_row["candidate"])
            available = name in constructed
            model, note = constructed.get(name, (None, ""))
            dependency_versions = ";".join(
                f"{dependency}={package_versions.get(dependency, 'unknown')}"
                for dependency in str(catalog_row["required_dependency"]).split(",")
            )
            rows.append(
                {
                    "target": target,
                    **catalog_row,
                    "available_in_environment": bool(available),
                    "preflight_status": "available" if available else "unavailable",
                    "constructed_model_class": type(model).__name__ if model is not None else "",
                    "model_note": note,
                    "dependency_versions": dependency_versions,
                }
            )
    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def _predict_model_entry(model_entry: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(model_entry, dict):
        members = model_entry.get("members", [])
        if len(members) == 0:
            raise ValueError("Model entry has no members.")
        preds = np.column_stack([m.predict(X) for m in members])
        weights = np.asarray(model_entry.get("weights", []), dtype=float)
        if len(weights) == preds.shape[1] and np.all(np.isfinite(weights)) and weights.sum() > 0:
            weights = weights / weights.sum()
            mean = preds @ weights
            std = np.sqrt(np.maximum(((preds - mean[:, None]) ** 2) @ weights, 0.0))
            residual_model = model_entry.get("residual_model")
            if residual_model is not None:
                residual_scale = float(model_entry.get("residual_scale", 0.0))
                residual = np.asarray(residual_model.predict(X), dtype=float)
                mean = mean + residual_scale * residual
            return mean, std
        mean = preds.mean(axis=1)
        residual_model = model_entry.get("residual_model")
        if residual_model is not None:
            residual_scale = float(model_entry.get("residual_scale", 0.0))
            residual = np.asarray(residual_model.predict(X), dtype=float)
            mean = mean + residual_scale * residual
        return mean, preds.std(axis=1)
    if isinstance(model_entry, list):
        preds = np.column_stack([m.predict(X) for m in model_entry])
        return preds.mean(axis=1), preds.std(axis=1)
    y = np.asarray(model_entry.predict(X), dtype=float)
    return y, np.zeros_like(y)


def _target_internal_space(target: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    if target in {"co2_emissions_kg_per_kg_meoh", "specific_energy_kwh_per_kg_meoh", "absolute_electrolyzer_area_m2", "capex_proxy"}:
        y_train = np.log(np.clip(train_df[target].to_numpy(dtype=float), 1e-12, None))
        y_test = np.log(np.clip(test_df[target].to_numpy(dtype=float), 1e-12, None))
        return y_train, y_test, "log_positive"
    y_train = train_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)
    return y_train, y_test, "raw"


def _inverse_target_prediction(target: str, y_internal: np.ndarray, y_std_internal: np.ndarray, space: str) -> Tuple[np.ndarray, np.ndarray]:
    if space == "log_positive":
        y = np.exp(y_internal)
        y_std = np.abs(y * y_std_internal)
        return y, y_std
    return y_internal, y_std_internal


def _fit_model(model: Any, X: np.ndarray, y: np.ndarray, *, seed: int, use_early_stop: bool = True) -> Any:
    if _is_xgboost_model(model):
        if use_early_stop and len(X) >= 40:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.1, random_state=seed, shuffle=True
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X, y, verbose=False)
        return model
    model.fit(X, y)
    return model


def _cross_validate(X: np.ndarray, y: np.ndarray, base_model: Any, k_folds: int, seed: int) -> Dict[str, float]:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_scores: List[Dict[str, float]] = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = clone(base_model)
        if _is_xgboost_model(model):
            model.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
        else:
            model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        fold_scores.append(_metrics(y[val_idx], pred))
    return {
        "cv_r2_mean": float(np.mean([m["r2"] for m in fold_scores])),
        "cv_r2_std": float(np.std([m["r2"] for m in fold_scores])),
        "cv_mae_mean": float(np.mean([m["mae"] for m in fold_scores])),
        "cv_rmse_mean": float(np.mean([m["rmse"] for m in fold_scores])),
    }


def _cross_validate_candidate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    candidates: List[Tuple[str, Any, str]],
    k_folds: int,
    seed: int,
) -> Dict[str, float]:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_scores: List[Dict[str, float]] = []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        preds = []
        for name, base_model, _note in candidates:
            if name.startswith("stack_"):
                continue
            try:
                model = clone(base_model)
                model = _fit_model(
                    model,
                    X[train_idx],
                    y[train_idx],
                    seed=seed + 100 * fold_i + len(name),
                    use_early_stop=True,
                )
                preds.append(np.asarray(model.predict(X[val_idx]), dtype=float))
            except Exception:
                continue
        if not preds:
            model = clone(_with_feature_scaler(_make_histgb_high_accuracy(PipelineConfig())))
            model.fit(X[train_idx], y[train_idx])
            preds.append(np.asarray(model.predict(X[val_idx]), dtype=float))
        pred = np.column_stack(preds).mean(axis=1)
        fold_scores.append(_metrics(y[val_idx], pred))
    return {
        "cv_r2_mean": float(np.mean([m["r2"] for m in fold_scores])),
        "cv_r2_std": float(np.std([m["r2"] for m in fold_scores])),
        "cv_mae_mean": float(np.mean([m["mae"] for m in fold_scores])),
        "cv_rmse_mean": float(np.mean([m["rmse"] for m in fold_scores])),
    }


def _calibrate_and_select_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidates: List[Tuple[str, Any, str]],
    seed: int,
    target: str,
    keep_n: int = 5,
) -> Tuple[List[Tuple[str, Any, str, float]], List[float], List[Dict[str, Any]]]:
    """Calibrate every eligible candidate and retain the lowest-RMSE members."""

    if len(X) >= 120:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed + 909,
            shuffle=True,
        )
    else:
        X_fit, X_cal, y_fit, y_cal = X, X, y, y

    constructed_names = {name for name, _model, _note in candidates}
    audit_by_name: Dict[str, Dict[str, Any]] = {}
    for catalog_row in _candidate_catalog(target):
        name = str(catalog_row["candidate"])
        available = name in constructed_names
        dependency = str(catalog_row["required_dependency"])
        if "xgboost" in dependency:
            missing_dependency = "xgboost"
        elif "lightgbm" in dependency:
            missing_dependency = "lightgbm"
        else:
            missing_dependency = dependency
        row: Dict[str, Any] = {
            **catalog_row,
            "candidate_order": int(catalog_row["candidate_order"]),
            "target": target,
            "available_in_environment": bool(available),
            "fitting_status": "not_evaluated" if available else "dependency_unavailable",
            "calibration_rmse_internal": float("nan"),
            "calibration_rank": float("nan"),
            "selected": False,
            "ensemble_weight": 0.0,
            "exclusion_reason": "" if available else f"required_dependency_not_available:{missing_dependency}",
            "error_type": "",
            "error_message": "",
            "fit_sample_size": int(len(X_fit)),
            "calibration_sample_size": int(len(X_cal)),
            "calibration_seed": int(seed + 909),
            "selection_cutoff": int(keep_n),
            "model_note": "",
        }
        audit_by_name[name] = row

    usable: List[Tuple[str, Any, str, float]] = []
    for name, base_model, note in candidates:
        if name not in audit_by_name:
            audit_by_name[name] = {
                "candidate_order": int(len(audit_by_name) + 1),
                "candidate": name,
                "model_family": type(base_model).__name__,
                "feature_representation": "unspecified",
                "required_dependency": "unknown",
                "scope": "runtime_candidate",
                "target": target,
                "available_in_environment": True,
                "fitting_status": "not_evaluated",
                "calibration_rmse_internal": float("nan"),
                "calibration_rank": float("nan"),
                "selected": False,
                "ensemble_weight": 0.0,
                "exclusion_reason": "",
                "error_type": "",
                "error_message": "",
                "fit_sample_size": int(len(X_fit)),
                "calibration_sample_size": int(len(X_cal)),
                "calibration_seed": int(seed + 909),
                "selection_cutoff": int(keep_n),
                "model_note": note,
            }
        row = audit_by_name[name]
        row["model_note"] = note
        try:
            cal_model = clone(base_model)
            cal_model = _fit_model(cal_model, X_fit, y_fit, seed=seed + len(name), use_early_stop=True)
            cal_pred = np.asarray(cal_model.predict(X_cal), dtype=float)
            rmse = float(np.sqrt(mean_squared_error(y_cal, cal_pred)))
            if np.isfinite(rmse):
                usable.append((name, base_model, note, rmse))
                row["fitting_status"] = "success"
                row["calibration_rmse_internal"] = rmse
                row["exclusion_reason"] = ""
            else:
                row["fitting_status"] = "invalid_calibration_metric"
                row["exclusion_reason"] = "non_finite_calibration_rmse"
        except Exception as exc:
            row["fitting_status"] = "fit_failed"
            row["exclusion_reason"] = "calibration_fit_failed"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc).splitlines()[0][:500]

    if not usable:
        fallback = (
            "histgb_scaled",
            _with_feature_scaler(_make_histgb_high_accuracy(PipelineConfig())),
            "fallback HistGB",
            1.0,
        )
        usable = [fallback]
        fallback_row = audit_by_name["histgb_scaled"]
        fallback_row["available_in_environment"] = True
        fallback_row["fitting_status"] = "fallback_selected"
        fallback_row["calibration_rmse_internal"] = 1.0
        fallback_row["exclusion_reason"] = ""
        fallback_row["model_note"] = "fallback HistGB after all candidate calibration fits failed"

    ranked = sorted(usable, key=lambda item: item[3])
    for rank, (name, _model, _note, _rmse) in enumerate(ranked, start=1):
        audit_by_name[name]["calibration_rank"] = int(rank)

    selected = ranked[: min(int(keep_n), len(ranked))]
    raw_weights = np.array([1.0 / max(rmse, 1e-12) ** 2 for *_rest, rmse in selected], dtype=float)
    raw_weights = np.clip(raw_weights, 0.02 * raw_weights.max(), raw_weights.max())
    weights = (raw_weights / raw_weights.sum()).tolist()
    selected_names = {name for name, _model, _note, _rmse in selected}
    weight_by_name = {
        name: float(weight)
        for (name, _model, _note, _rmse), weight in zip(selected, weights)
    }

    for name, row in audit_by_name.items():
        if name in selected_names:
            row["selected"] = True
            row["ensemble_weight"] = weight_by_name[name]
            row["exclusion_reason"] = ""
        elif row["fitting_status"] == "success":
            row["exclusion_reason"] = "outside_top_five_by_calibration_rmse"

    audit_rows = sorted(audit_by_name.values(), key=lambda row: int(row["candidate_order"]))
    return selected, weights, audit_rows


def _fit_weighted_candidate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    candidates: List[Tuple[str, Any, str]],
    seed: int,
    target: str,
) -> Tuple[List[Any], List[str], List[str], List[float], List[float], Any, float, Dict[str, float], List[Dict[str, Any]]]:
    usable, weights, candidate_audit_rows = _calibrate_and_select_candidates(
        X=X,
        y=y,
        candidates=candidates,
        seed=seed,
        target=target,
        keep_n=5,
    )
    calibration_rmse = [item[3] for item in usable]

    residual_model = None
    residual_scale = 0.0
    residual_meta: Dict[str, float] = {}
    if target == "carbon_utilization" and len(usable) >= 2 and len(X) >= 120:
        try:
            kf = KFold(n_splits=min(5, max(3, len(X) // 300)), shuffle=True, random_state=seed + 707)
            oof_pred = np.zeros(len(X), dtype=float)
            for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X)):
                fold_preds = []
                for name, base_model, _note, _rmse in usable:
                    model = clone(base_model)
                    model = _fit_model(
                        model,
                        X[tr_idx],
                        y[tr_idx],
                        seed=seed + 3000 + 100 * fold_i + len(name),
                        use_early_stop=True,
                    )
                    fold_preds.append(np.asarray(model.predict(X[va_idx]), dtype=float))
                fold_pred_mat = np.column_stack(fold_preds)
                oof_pred[va_idx] = fold_pred_mat @ np.asarray(weights, dtype=float)

            residual = y - oof_pred
            residual_model = SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("features", FeatureAugmenter("rich")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=700,
                            learning_rate=0.025,
                            max_depth=4,
                            min_samples_leaf=10,
                            l2_regularization=1e-3,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            random_state=seed + 404,
                        ),
                    ),
                ]
            )
            residual_model.fit(X, residual)
            residual_scale = 0.50
            residual_meta = {
                "residual_oof_rmse_before": float(np.sqrt(mean_squared_error(y, oof_pred))),
                "residual_oof_mae_before": float(mean_absolute_error(y, oof_pred)),
                "residual_train_std": float(np.std(residual)),
                "residual_scale": float(residual_scale),
            }
        except Exception:
            residual_model = None
            residual_scale = 0.0
            residual_meta = {}

    trained_members: List[Any] = []
    member_names: List[str] = []
    member_notes: List[str] = []
    for i, (name, base_model, note, _rmse) in enumerate(usable):
        model = clone(base_model)
        model = _fit_model(model, X, y, seed=seed + 1000 + i, use_early_stop=True)
        trained_members.append(model)
        member_names.append(name)
        member_notes.append(note)

    return (
        trained_members,
        member_names,
        member_notes,
        weights,
        calibration_rmse,
        residual_model,
        residual_scale,
        residual_meta,
        candidate_audit_rows,
    )


def _cross_validated_selected_ensemble_predictions(
    X: np.ndarray,
    y: np.ndarray,
    candidates: List[Tuple[str, Any, str]],
    selected_names: List[str],
    selected_weights: List[float],
    target: str,
    k_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = [(name, model, note) for name, model, note in candidates if name in set(selected_names)]
    if target != "carbon_utilization":
        # Stacking is expensive inside OOF and already represented by its base learners elsewhere.
        selected = [(name, model, note) for name, model, note in selected if not name.startswith("stack_")]
    if not selected:
        selected = [(name, model, note) for name, model, note in candidates if not name.startswith("stack_")][:3]
    if not selected:
        selected = candidates[:1]

    weight_map = {name: float(w) for name, w in zip(selected_names, selected_weights)}
    weights = np.asarray([weight_map.get(name, 1.0) for name, _model, _note in selected], dtype=float)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones(len(selected), dtype=float)
    weights = weights / weights.sum()

    n_splits = min(max(2, int(k_folds)), len(X))
    if target == "carbon_utilization" and n_splits >= 3:
        try:
            bins = min(10, max(3, len(y) // 120))
            y_bins = pd.qcut(pd.Series(y), q=bins, labels=False, duplicates="drop").to_numpy()
            if len(np.unique(y_bins)) >= n_splits and pd.Series(y_bins).value_counts().min() >= n_splits:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 515)
                split_iter = splitter.split(X, y_bins)
            else:
                split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 515).split(X)
        except Exception:
            split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 515).split(X)
    else:
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 515).split(X)
    oof_pred = np.zeros(len(X), dtype=float)
    oof_std = np.zeros(len(X), dtype=float)
    for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
        fold_preds = []
        fold_train_preds = []
        for name, base_model, _note in selected:
            model = clone(base_model)
            model = _fit_model(
                model,
                X[tr_idx],
                y[tr_idx],
                seed=seed + 6000 + 100 * fold_i + len(name),
                use_early_stop=True,
            )
            fold_train_preds.append(np.asarray(model.predict(X[tr_idx]), dtype=float))
            fold_preds.append(np.asarray(model.predict(X[va_idx]), dtype=float))
        train_pred_mat = np.column_stack(fold_train_preds)
        pred_mat = np.column_stack(fold_preds)
        train_mean = train_pred_mat @ weights
        mean = pred_mat @ weights
        if target == "carbon_utilization":
            try:
                residual = y[tr_idx] - train_mean
                residual_model = SklearnPipeline(
                    [
                        ("physics", V23ReducedPhysicsFeatureAugmenter()),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                max_iter=550,
                                learning_rate=0.025,
                                max_depth=4,
                                min_samples_leaf=12,
                                l2_regularization=1e-3,
                                early_stopping=True,
                                validation_fraction=0.15,
                                n_iter_no_change=30,
                                random_state=seed + 7200 + fold_i,
                            ),
                        ),
                    ]
                )
                residual_model.fit(X[tr_idx], residual)
                mean = mean + 0.35 * np.asarray(residual_model.predict(X[va_idx]), dtype=float)
            except Exception:
                pass
        oof_pred[va_idx] = mean
        oof_std[va_idx] = np.sqrt(np.maximum(((pred_mat - mean[:, None]) ** 2) @ weights, 0.0))
    return oof_pred, oof_std


def audit_candidate_surrogates(
    train_df: pd.DataFrame,
    cfg: PipelineConfig,
    target_names: List[str],
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Evaluate candidate calibration performance without fitting final ensembles."""

    feature_names = cfg.feature_names
    X_train = train_df[feature_names].to_numpy(dtype=float)
    audit_rows: List[Dict[str, Any]] = []
    for target in target_names:
        if target not in train_df.columns:
            continue
        y_train_internal, _unused, prediction_space = _target_internal_space(target, train_df, train_df)
        candidates = _make_candidate_regressors(cfg, target)
        _selected, _weights, target_rows = _calibrate_and_select_candidates(
            X=X_train,
            y=y_train_internal,
            candidates=candidates,
            seed=cfg.surrogate.random_state + int(seed_offset),
            target=target,
            keep_n=5,
        )
        for row in target_rows:
            row["prediction_space"] = prediction_space
            row["audit_seed_offset"] = int(seed_offset)
        audit_rows.extend(target_rows)
    return pd.DataFrame(audit_rows)


def fit_surrogate_point_models(
    train_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Fit final point-prediction ensembles without consuming an external calibration set.

    This is the proper-training stage used by split conformal prediction. Candidate
    selection and all reference-feature fitting remain inside ``train_df``; a caller
    can therefore reserve a disjoint dataframe for conformal calibration.
    """

    feature_names = cfg.feature_names
    target_names = cfg.objective_names
    X_train = train_df[feature_names].to_numpy(dtype=float)
    models: Dict[str, Any] = {}
    candidate_audit_rows: List[Dict[str, Any]] = []

    for target in target_names:
        y_train_internal, _unused, pred_space = _target_internal_space(
            target,
            train_df,
            train_df.iloc[:1],
        )
        candidates = _make_candidate_regressors(cfg, target)
        (
            trained_members,
            member_names,
            member_notes,
            member_weights,
            calibration_rmse,
            residual_model,
            residual_scale,
            residual_meta,
            target_candidate_audit_rows,
        ) = _fit_weighted_candidate_ensemble(
            X_train,
            y_train_internal,
            candidates=candidates,
            seed=cfg.surrogate.random_state,
            target=target,
        )
        for audit_row in target_candidate_audit_rows:
            audit_row["prediction_space"] = pred_space
            audit_row["audit_seed_offset"] = 0
        candidate_audit_rows.extend(target_candidate_audit_rows)

        model_entry: Dict[str, Any] = {
            "members": trained_members,
            "member_names": member_names,
            "member_notes": member_notes,
            "weights": member_weights,
            "calibration_rmse_internal": calibration_rmse,
            "prediction_space": pred_space,
            "target_name": target,
            "backend": "weighted_heterogeneous_ensemble",
            "backend_note": "Weighted by internal validation RMSE; candidates: "
            + ", ".join(member_names),
            "proper_training_size": int(len(train_df)),
        }
        if residual_model is not None:
            model_entry["residual_model"] = residual_model
            model_entry["residual_scale"] = float(residual_scale)
            model_entry["backend_note"] += "; conservative cross-fitted residual correction"
            model_entry.update(residual_meta)
        models[target] = model_entry

    return models, pd.DataFrame(candidate_audit_rows)


def _finite_sample_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Conformal calibration requires at least one finite score.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    rank = min(int(np.ceil((len(values) + 1) * (1.0 - float(alpha)))), len(values))
    return float(np.sort(values)[rank - 1])


def calibrate_split_conformal_intervals(
    models: Dict[str, Any],
    calibration_df: pd.DataFrame,
    feature_names: List[str],
    alpha: float = 0.05,
    scale_floor_quantile: float = 0.10,
) -> pd.DataFrame:
    """Attach absolute and ensemble-SD-scaled split-conformal calibrators.

    Calibration is performed in each target's reported physical scale. The
    finite-sample ``higher`` order statistic gives marginal coverage under the
    usual exchangeability assumption. The function mutates model metadata only;
    fitted point-prediction members are unchanged.
    """

    predictions = predict_surrogates(
        models,
        calibration_df,
        feature_names,
        return_uncertainty=True,
    )
    rows: List[Dict[str, Any]] = []
    for target, model_entry in models.items():
        if not isinstance(model_entry, dict):
            raise TypeError("Split-conformal metadata requires dictionary model entries.")
        actual = calibration_df[target].to_numpy(dtype=float)
        mean = predictions[target].to_numpy(dtype=float)
        ensemble_sd = np.clip(
            predictions[f"{target}_std"].to_numpy(dtype=float),
            0.0,
            None,
        )
        absolute_error = np.abs(actual - mean)
        scale_floor = float(np.quantile(ensemble_sd, float(scale_floor_quantile)))
        scale_floor = max(scale_floor, np.finfo(float).eps)
        adaptive_scale = np.maximum(ensemble_sd, scale_floor)
        absolute_qhat = _finite_sample_conformal_quantile(absolute_error, alpha)
        scaled_scores = absolute_error / adaptive_scale
        scaled_qhat = _finite_sample_conformal_quantile(scaled_scores, alpha)
        metadata = {
            "alpha": float(alpha),
            "nominal_coverage": float(1.0 - alpha),
            "calibration_size": int(len(calibration_df)),
            "finite_sample_rank": int(
                min(np.ceil((len(calibration_df) + 1) * (1.0 - alpha)), len(calibration_df))
            ),
            "absolute_qhat": float(absolute_qhat),
            "scaled_qhat": float(scaled_qhat),
            "scale_floor": float(scale_floor),
            "scale_floor_quantile": float(scale_floor_quantile),
            "score_space": "reported_target_scale",
            "methods": ["absolute_residual", "scaled_ensemble_sd"],
        }
        model_entry["split_conformal"] = metadata
        rows.append({"target": target, **metadata})
    return pd.DataFrame(rows)


def train_surrogates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = cfg.feature_names
    target_names = cfg.objective_names
    X_train = train_df[feature_names].to_numpy(dtype=float)
    X_test = test_df[feature_names].to_numpy(dtype=float)
    boundary_mask = test_df["boundary_margin"].to_numpy(dtype=float) <= test_df["boundary_margin"].quantile(0.25)
    if boundary_mask.sum() < 5:
        boundary_mask = test_df["boundary_margin"].to_numpy(dtype=float) <= test_df["boundary_margin"].quantile(0.40)

    models: Dict[str, Any] = {}
    report_rows: List[Dict[str, Any]] = []
    candidate_audit_rows: List[Dict[str, Any]] = []
    validation_pred_df = pd.DataFrame(index=train_df.index)
    validation_pred_df["validation_source"] = "cross_validated_oof"

    for target in target_names:
        y_train_internal, y_test_internal, pred_space = _target_internal_space(target, train_df, test_df)
        y_test_true = test_df[target].to_numpy(dtype=float)
        candidates = _make_candidate_regressors(cfg, target)
        cv = _cross_validate_candidate_ensemble(
            X_train,
            y_train_internal,
            candidates=candidates,
            k_folds=cfg.surrogate.k_folds,
            seed=cfg.surrogate.random_state,
        )

        (
            trained_members,
            member_names,
            member_notes,
            member_weights,
            calibration_rmse,
            residual_model,
            residual_scale,
            residual_meta,
            target_candidate_audit_rows,
        ) = _fit_weighted_candidate_ensemble(
            X_train,
            y_train_internal,
            candidates=candidates,
            seed=cfg.surrogate.random_state,
            target=target,
        )
        for audit_row in target_candidate_audit_rows:
            audit_row["prediction_space"] = pred_space
            audit_row["audit_seed_offset"] = 0
        candidate_audit_rows.extend(target_candidate_audit_rows)

        model_entry: Any = {
            "members": trained_members,
            "member_names": member_names,
            "member_notes": member_notes,
            "weights": member_weights,
            "calibration_rmse_internal": calibration_rmse,
            "prediction_space": pred_space,
            "target_name": target,
            "backend": "weighted_heterogeneous_ensemble",
            "backend_note": "Weighted by internal validation RMSE; candidates: " + ", ".join(member_names),
        }
        if residual_model is not None:
            model_entry["residual_model"] = residual_model
            model_entry["residual_scale"] = float(residual_scale)
            model_entry["backend_note"] += "; conservative cross-fitted residual correction"
            model_entry.update(residual_meta)
        pred_test_internal, pred_test_std_internal = _predict_model_entry(model_entry, X_test)
        pred_boundary_internal, pred_boundary_std_internal = _predict_model_entry(model_entry, X_test[boundary_mask])
        pred_test, pred_test_std = _inverse_target_prediction(target, pred_test_internal, pred_test_std_internal, pred_space)
        pred_boundary, pred_boundary_std = _inverse_target_prediction(
            target, pred_boundary_internal, pred_boundary_std_internal, pred_space
        )
        metrics_test = _metrics(y_test_true, pred_test)
        metrics_boundary = _metrics(y_test_true[boundary_mask], pred_boundary)
        abs_err = np.abs(y_test_true - pred_test)
        std_err_corr = float(np.corrcoef(pred_test_std, abs_err)[0, 1]) if np.std(pred_test_std) > 1e-12 else 0.0
        q = float(np.quantile(pred_test_std, 0.8)) if len(pred_test_std) > 0 else 0.0
        hi_mask = pred_test_std >= q
        hi_mae = float(np.mean(abs_err[hi_mask])) if np.any(hi_mask) else float(np.mean(abs_err))
        lo_mae = float(np.mean(abs_err[~hi_mask])) if np.any(~hi_mask) else float(np.mean(abs_err))

        # store metadata on base members for downstream SHAP reporting
        targets_for_meta = trained_members
        for mm in targets_for_meta:
            try:
                setattr(mm, "_isamo_backend", "weighted_heterogeneous_ensemble")
                setattr(mm, "_isamo_backend_note", model_entry["backend_note"])
            except Exception:
                pass
        models[target] = model_entry

        row: Dict[str, Any] = {"target": target}
        row.update(cv)
        row.update({f"test_{k}": v for k, v in metrics_test.items()})
        row.update({f"boundary_{k}": v for k, v in metrics_boundary.items()})
        row["boundary_n"] = int(boundary_mask.sum())
        row["surrogate_backend"] = "weighted_heterogeneous_ensemble"
        row["backend_note"] = model_entry["backend_note"]
        row["prediction_space"] = pred_space
        row["ensemble_size"] = int(len(trained_members))
        row["ensemble_members"] = ",".join(member_names)
        row["ensemble_weights"] = ",".join(f"{w:.6g}" for w in member_weights)
        row["calibration_rmse_internal"] = ",".join(f"{e:.6g}" for e in calibration_rmse)
        row["residual_correction"] = bool(residual_model is not None)
        row["residual_scale"] = float(residual_scale)
        for k, v in residual_meta.items():
            row[k] = v
        row["test_pred_std_mean"] = float(np.mean(pred_test_std))
        row["boundary_pred_std_mean"] = float(np.mean(pred_boundary_std))
        row["uncertainty_abs_error_corr"] = std_err_corr
        row["test_mae_top20pct_uncertainty"] = hi_mae
        row["test_mae_bottom80pct_uncertainty"] = lo_mae
        report_rows.append(row)

        oof_internal, oof_std_internal = _cross_validated_selected_ensemble_predictions(
            X_train,
            y_train_internal,
            candidates=candidates,
            selected_names=member_names,
            selected_weights=member_weights,
            target=target,
            k_folds=cfg.surrogate.k_folds,
            seed=cfg.surrogate.random_state,
        )
        oof_pred, oof_std = _inverse_target_prediction(target, oof_internal, oof_std_internal, pred_space)
        if target == "carbon_utilization":
            oof_pred = np.clip(oof_pred, 0.0, 0.999)
        elif target == "energy_efficiency_ratio":
            oof_pred = np.clip(oof_pred, 0.0, 1.0)
            oof_std = np.clip(oof_std, 0.0, None)
        elif target in {"co2_emissions_kg_per_kg_meoh", "specific_energy_kwh_per_kg_meoh", "absolute_electrolyzer_area_m2", "capex_proxy"}:
            oof_pred = np.clip(oof_pred, 1e-6, None)
            oof_std = np.clip(oof_std, 0.0, None)
        validation_pred_df[target] = oof_pred
        validation_pred_df[f"{target}_std"] = oof_std

    metrics_df = pd.DataFrame(report_rows)
    candidate_audit_df = pd.DataFrame(candidate_audit_rows)
    return models, metrics_df, validation_pred_df, candidate_audit_df


def predict_surrogates(
    models: Dict[str, Any],
    X: pd.DataFrame,
    feature_names: List[str],
    return_uncertainty: bool = False,
    return_intervals: bool = False,
    interval_method: str = "scaled_ensemble_sd",
) -> pd.DataFrame:
    arr = X[feature_names].to_numpy(dtype=float)
    out: Dict[str, np.ndarray] = {}
    for name, model_entry in models.items():
        mu_i, sigma_i = _predict_model_entry(model_entry, arr)
        if isinstance(model_entry, dict):
            pred_space = str(model_entry.get("prediction_space", "raw"))
        else:
            pred_space = "raw"
        mu, sigma = _inverse_target_prediction(name, mu_i, sigma_i, pred_space)
        if name == "carbon_utilization":
            mu = np.clip(mu, 0.0, 0.999)
        elif name == "energy_efficiency_ratio":
            mu = np.clip(mu, 0.0, 1.0)
            sigma = np.clip(sigma, 0.0, None)
        elif name in {"co2_emissions_kg_per_kg_meoh", "specific_energy_kwh_per_kg_meoh", "absolute_electrolyzer_area_m2", "capex_proxy"}:
            mu = np.clip(mu, 1e-6, None)
            sigma = np.clip(sigma, 0.0, None)
        out[name] = mu
        if return_uncertainty:
            out[f"{name}_std"] = sigma
        if return_intervals:
            if not isinstance(model_entry, dict) or "split_conformal" not in model_entry:
                raise ValueError(f"Model '{name}' has no split-conformal calibration metadata.")
            conformal = model_entry["split_conformal"]
            coverage = float(conformal["nominal_coverage"])
            if interval_method == "absolute_residual":
                half_width = np.full(len(mu), float(conformal["absolute_qhat"]), dtype=float)
            elif interval_method == "scaled_ensemble_sd":
                scale = np.maximum(np.asarray(sigma, dtype=float), float(conformal["scale_floor"]))
                half_width = float(conformal["scaled_qhat"]) * scale
            else:
                raise ValueError(
                    "interval_method must be 'absolute_residual' or 'scaled_ensemble_sd'."
                )
            lower = mu - half_width
            upper = mu + half_width
            if name == "carbon_utilization":
                lower = np.clip(lower, 0.0, 0.999)
                upper = np.clip(upper, 0.0, 0.999)
            elif name in {
                "co2_emissions_kg_per_kg_meoh",
                "specific_energy_kwh_per_kg_meoh",
                "absolute_electrolyzer_area_m2",
                "capex_proxy",
            }:
                lower = np.clip(lower, 1e-6, None)
                upper = np.clip(upper, 1e-6, None)
            suffix = f"{100.0 * coverage:g}".replace(".", "p")
            out[f"{name}_lower_{suffix}"] = lower
            out[f"{name}_upper_{suffix}"] = upper
    return pd.DataFrame(out)


def save_models(models: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model_entry in models.items():
        if isinstance(model_entry, dict):
            members = model_entry.get("members", [])
            meta = {k: v for k, v in model_entry.items() if k != "members"}
            for i, member in enumerate(members):
                joblib.dump(member, out_dir / f"{name}__ens{i}.joblib")
            joblib.dump(meta, out_dir / f"{name}__meta.joblib")
        elif isinstance(model_entry, list):
            for i, member in enumerate(model_entry):
                joblib.dump(member, out_dir / f"{name}__ens{i}.joblib")
        else:
            joblib.dump(model_entry, out_dir / f"{name}.joblib")
