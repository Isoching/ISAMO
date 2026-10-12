from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .core import PROJECT, PREDICTED, TARGETS, config, read


def cost_matrix(mu: np.ndarray) -> np.ndarray:
    return np.column_stack([-mu[:, 0], mu[:, 1], mu[:, 2]])


def nondominated(cost: np.ndarray) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    return np.asarray(NonDominatedSorting(method="efficient_non_dominated_sort").do(
        np.asarray(cost, dtype=float), only_non_dominated_front=True), dtype=int)


def rank(frame: pd.DataFrame, weights=(1.0, 1.0, 1.0)) -> pd.DataFrame:
    """V30 utility normalization and first-occurrence tie handling; no optimisation."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Supply three non-negative finite weights with a positive sum")
    if frame.empty or not np.isfinite(frame[PREDICTED].to_numpy()).all():
        raise ValueError("A finite non-empty objective table is required")
    w = w / w.sum()
    out = frame.copy()
    utility = np.column_stack([out[PREDICTED[0]], -out[PREDICTED[1]], -out[PREDICTED[2]]])
    utility = (utility - utility.min(axis=0)) / np.maximum(utility.max(axis=0) - utility.min(axis=0), 1e-12)
    weighted = utility / np.maximum(np.sqrt((utility**2).sum(axis=0)), 1e-12) * w
    ideal, anti = weighted.max(axis=0), weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti)**2).sum(axis=1))
    out["topsis_closeness_prequery"] = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    out["topsis_rank_prequery"] = pd.Series(-out["topsis_closeness_prequery"].to_numpy()).rank(method="first").astype(int).to_numpy()
    regret = (1.0 - utility) * w
    s, r = regret.sum(axis=1), regret.max(axis=1)
    out["vikor_Q_prequery"] = 0.5 * (s - s.min()) / max(float(s.max() - s.min()), 1e-12) + 0.5 * (r - r.min()) / max(float(r.max() - r.min()), 1e-12)
    out["vikor_rank_prequery"] = out["vikor_Q_prequery"].rank(method="first").astype(int).to_numpy()
    return out


def finite_sample_quantile(scores: np.ndarray, coverage=0.95) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all() or not 0 < coverage < 1:
        raise ValueError("Finite non-empty scores and coverage in (0, 1) are required")
    index = min(len(scores), math.ceil((len(scores) + 1) * coverage)) - 1
    return float(np.sort(scores)[index])


def evaluate() -> pd.DataFrame:
    """Recompute primary errors/coverage from recorded simulator labels and predictions."""
    evidence = PROJECT / "data/validation_evidence"
    points = read(evidence / "cross_conformal_pointwise.csv")
    references = read(evidence / "model_variant_lockbox_metrics.csv")
    coverage_reference = read(evidence / "cross_conformal_metrics.csv")
    ranges = config()["fixed_target_ranges"]
    rows = []
    for (dataset, target), block in points.groupby(["dataset", "target"], sort=False):
        actual = block["actual_gproms"].to_numpy()
        prediction = block["predicted_final"].to_numpy()
        rmse = float(np.sqrt(np.mean((actual - prediction)**2)))
        fixed = 100 * rmse / ranges[target]
        covered = (actual >= block["interval_lower_95"]) & (actual <= block["interval_upper_95"])
        coverage = float(100 * covered.mean())
        original = references.loc[(references["dataset"] == dataset) & (references["stratum"] == "ALL") &
                                  (references["target"] == target) & (references["variant"] == "pooled_all_al")]
        cov = coverage_reference.loc[(coverage_reference["dataset"] == dataset) & (coverage_reference["target"] == target)]
        if len(original) != 1 or len(cov) != 1:
            raise ValueError("Missing/ambiguous recorded validation metric")
        if not np.isclose(rmse, original.iloc[0]["rmse"], rtol=0, atol=1e-10) or not np.isclose(coverage, cov.iloc[0]["empirical_coverage_pct"], rtol=0, atol=1e-10):
            raise ValueError(f"Published metric differs: {dataset}/{target}")
        rows.append({"dataset": dataset, "analysis_scope": cov.iloc[0]["analysis_scope"], "target": target,
                     "n": len(block), "rmse": rmse, "fixed_range_nrmse_pct": fixed,
                     "empirical_coverage_pct": coverage})
    return pd.DataFrame(rows)


def plot_validation(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = read(PROJECT / "data/validation_evidence/cross_conformal_pointwise.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, target, label in zip(axes, TARGETS, ["CU", "SEC", "Capital-scale proxy"]):
        block = points.loc[(points["dataset"] == "confirmation96") & (points["target"] == target)]
        actual, predicted = block["actual_gproms"], block["predicted_final"]
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.scatter(actual, predicted, s=14, color="#4E79A7", alpha=0.7)
        ax.plot([low, high], [low, high], color="black", lw=0.8)
        ax.set(xlabel="gPROMS", ylabel="Surrogate prediction", title=label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"confirmation_parity.{suffix}", dpi=300)
    plt.close(fig)
