from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import PALETTE, configure_matplotlib_nature, savefig, style_axis


@dataclass
class ExplainResult:
    shap_values: np.ndarray
    backend: str
    shap_explanation: object = None


def _pretty_target_label(target_name: str) -> str:
    mapping = {
        "carbon_utilization": "Carbon utilisation (-)",
        "specific_energy_kwh_per_kg_meoh": "SEC (kWh/kg MeOH)",
        "absolute_electrolyzer_area_m2": "Electrolyser area",
        "capex_proxy": "CAPEX proxy (-)",
        "energy_efficiency_ratio": "Legacy energy efficiency",
        "co2_emissions_kg_per_kg_meoh": "CO2 emissions intensity",
    }
    return mapping.get(target_name, target_name)


def _pretty_feature_label(feature: str) -> str:
    mapping = {
        "P_Cap": r"$P_{\mathrm{Cap}}$",
        "J_eCO2R": r"$J_{\mathrm{eCO_2R}}$",
        "T_eCO2R": r"$T_{\mathrm{eCO_2R}}$",
        "N_eCO2R": r"$N_{\mathrm{eCO_2R}}$",
        "P_WE": r"$P_{\mathrm{WE}}$",
        "N_WE": r"$N_{\mathrm{WE}}$",
        "P_MeOH": r"$P_{\mathrm{MeOH}}$",
        "T_utility": r"$T_{\mathrm{utility}}$",
    }
    return mapping.get(feature, feature)


def _all_subset_masks(n_features: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for r in range(n_features + 1):
        for idxs in combinations(range(n_features), r):
            mask = np.zeros(n_features, dtype=bool)
            if idxs:
                mask[list(idxs)] = True
            masks.append(mask)
    return masks


def approximate_shapley_regression(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    X_background: np.ndarray,
    n_permutations: int = 64,
    random_state: int = 42,
) -> np.ndarray:
    """
    Lightweight Shapley estimator for regression.

    For small feature counts (<=8), this computes an exact Shapley decomposition
    over a sampled background set via subset enumeration.
    Falls back to Monte Carlo permutation sampling for larger dimensions.
    """
    n_eval, n_features = X_eval.shape
    shap_values = np.zeros((n_eval, n_features), dtype=float)

    if n_features <= 8:
        subset_masks = _all_subset_masks(n_features)
        fact = math.factorial
        weight_cache = {}
        for i in range(n_eval):
            x = X_eval[i]
            v = {}
            for mask in subset_masks:
                key = tuple(mask.tolist())
                X_mix = X_background.copy()
                X_mix[:, mask] = x[mask]
                v[key] = float(np.mean(predict_fn(X_mix)))
            for j in range(n_features):
                phi = 0.0
                for mask in subset_masks:
                    if mask[j]:
                        continue
                    s = int(mask.sum())
                    key_s = tuple(mask.tolist())
                    mask_with_j = mask.copy()
                    mask_with_j[j] = True
                    key_sj = tuple(mask_with_j.tolist())
                    if s not in weight_cache:
                        weight_cache[s] = fact(s) * fact(n_features - s - 1) / fact(n_features)
                    phi += weight_cache[s] * (v[key_sj] - v[key_s])
                shap_values[i, j] = phi
        return shap_values

    rng = np.random.default_rng(random_state)
    for i in range(n_eval):
        x = X_eval[i]
        contrib = np.zeros(n_features, dtype=float)
        for _ in range(n_permutations):
            z = X_background[rng.integers(0, len(X_background))].copy()
            order = rng.permutation(n_features)
            current = z.copy()
            prev_pred = float(predict_fn(current.reshape(1, -1))[0])
            for j in order:
                current[j] = x[j]
                new_pred = float(predict_fn(current.reshape(1, -1))[0])
                contrib[j] += new_pred - prev_pred
                prev_pred = new_pred
        shap_values[i] = contrib / float(n_permutations)
    return shap_values


def compute_shap_values_auto(
    model,
    X_eval_df: pd.DataFrame,
    X_background_df: pd.DataFrame,
    n_permutations: int = 64,
    random_state: int = 42,
    prefer_shap_package: bool = True,
) -> ExplainResult:
    if prefer_shap_package:
        try:
            import shap

            configure_matplotlib_nature()
            try:
                # Try model-aware explainer first (often TreeExplainer for tree ensembles).
                explainer = shap.Explainer(model, X_background_df, seed=random_state)
            except Exception:
                # Guaranteed fallback path if model type is not supported directly.
                explainer = shap.Explainer(model.predict, X_background_df, seed=random_state)

            explanation = explainer(X_eval_df)
            values = np.asarray(getattr(explanation, "values", explanation), dtype=float)
            if values.ndim == 3:
                values = values[..., 0]
            if values.ndim != 2:
                raise RuntimeError(f"Unexpected SHAP values shape: {values.shape}")
            return ExplainResult(shap_values=values, backend=f"shap.{explainer.__class__.__name__}", shap_explanation=explanation)
        except Exception as e:
            print(f"    SHAP package path failed ({type(e).__name__}); using internal Shapley fallback.")

    values = approximate_shapley_regression(
        predict_fn=model.predict,
        X_eval=X_eval_df.to_numpy(dtype=float),
        X_background=X_background_df.to_numpy(dtype=float),
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return ExplainResult(shap_values=values, backend="internal.shapley_exact_or_mc", shap_explanation=None)


def _save_shap_package_figures(
    target_name: str,
    X_eval_df: pd.DataFrame,
    shap_values: np.ndarray,
    out_dir: Path,
) -> None:
    import shap

    configure_matplotlib_nature()
    X_eval_display_df = X_eval_df.rename(columns={col: _pretty_feature_label(col) for col in X_eval_df.columns})

    # Dot/beeswarm-like summary
    fig = plt.figure(figsize=(3.6, 2.8))
    shap.summary_plot(
        shap_values,
        X_eval_display_df,
        show=False,
        plot_type="dot",
        max_display=min(8, X_eval_df.shape[1]),
        color_bar=True,
    )
    plt.gcf().suptitle(f"SHAP Summary ({_pretty_target_label(target_name)})", y=0.995, fontsize=8)
    savefig(plt.gcf(), out_dir / f"{target_name}_shap_summary_dot.png")

    # Bar summary
    fig = plt.figure(figsize=(3.0, 2.6))
    shap.summary_plot(
        shap_values,
        X_eval_display_df,
        show=False,
        plot_type="bar",
        max_display=min(8, X_eval_df.shape[1]),
    )
    plt.gcf().suptitle(f"SHAP Global Importance ({_pretty_target_label(target_name)})", y=0.995, fontsize=8)
    savefig(plt.gcf(), out_dir / f"{target_name}_shap_summary_bar.png")


def save_shap_artifacts(
    target_name: str,
    feature_names: List[str],
    X_eval_df: pd.DataFrame,
    shap_values: np.ndarray,
    out_dir: Path,
    shap_explanation: object = None,
    backend: Optional[str] = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_nature()

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_eval_df.index)
    shap_df.to_csv(out_dir / f"{target_name}_shap_values.csv", index=True)

    global_importance = (
        shap_df.abs().mean(axis=0).rename("mean_abs_shap").sort_values(ascending=False).reset_index()
    )
    global_importance.columns = ["feature", "mean_abs_shap"]
    global_importance["backend"] = backend or "unknown"
    global_importance.to_csv(out_dir / f"{target_name}_global_importance.csv", index=False)

    # Nature-style bar chart (custom, always available)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.barh(
        [_pretty_feature_label(f) for f in global_importance["feature"][::-1]],
        global_importance["mean_abs_shap"][::-1],
        color=PALETTE["blue"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(_pretty_target_label(target_name))
    style_axis(ax, grid=False)
    savefig(fig, out_dir / f"{target_name}_shap_importance_nature.png")

    # Dependence plots for top-2 features (custom, Nature-style)
    top_features = global_importance["feature"].head(2).tolist()
    for feat in top_features:
        fig, ax = plt.subplots(figsize=(3.3, 2.7))
        ax.scatter(
            X_eval_df[feat].to_numpy(dtype=float),
            shap_df[feat].to_numpy(dtype=float),
            s=16,
            alpha=0.85,
            c=PALETTE["red"],
            edgecolors="none",
        )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.7)
        ax.set_xlabel(_pretty_feature_label(feat))
        ax.set_ylabel("SHAP value")
        ax.set_title(f"{_pretty_target_label(target_name)} vs {_pretty_feature_label(feat)}")
        style_axis(ax, grid=False)
        savefig(fig, out_dir / f"{target_name}_dependence_{feat}_nature.png")

    # Official shap plots if explanation/package path is available
    if shap_explanation is not None:
        try:
            _save_shap_package_figures(target_name, X_eval_df, shap_values, out_dir)
        except Exception as e:
            print(f"    SHAP plotting fallback used for {target_name} ({type(e).__name__}).")

    return global_importance
