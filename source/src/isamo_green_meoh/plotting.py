from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap


# Shared low-saturation Nature-style reference palette.  The eight chromatic
# colours follow the manuscript reference supplied for V29 and are the default
# for all newly generated figures.
PALETTE: Dict[str, str] = {
    "navy": "#2F4B66",
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "red": "#E15759",
    "teal": "#76B7B2",
    "green": "#59A14F",
    "yellow": "#EDC948",
    "purple": "#B07AA1",
    "brown": "#9C755F",
    "gray": "#6F6F6F",
}

HIGHLIGHT_COLORS: Dict[str, str] = {
    "joint_best": PALETTE["red"],
    "topsis_best": PALETTE["red"],
    "vikor_best": PALETTE["teal"],
    "max_carbon": PALETTE["blue"],
    "max_energy_efficiency": PALETTE["green"],
    "min_energy": PALETTE["green"],
    "min_specific_energy": PALETTE["green"],
    "min_area": PALETTE["purple"],
    "min_capex": PALETTE["purple"],
    "min_emissions": PALETTE["purple"],
}

NATURE_SEQ_CMAP = LinearSegmentedColormap.from_list(
    "nature_seq",
    [PALETTE["blue"], PALETTE["teal"], PALETTE["green"], PALETTE["yellow"]],
)

NATURE_SEQ_CMAP_LIGHT = LinearSegmentedColormap.from_list(
    "nature_seq_light",
    [PALETTE["teal"], PALETTE["green"], PALETTE["yellow"]],
)

FIGURE_OUTPUT_SUFFIX = ".svg"

FIGURE_ZORDER: Dict[str, int] = {
    "background": 1,
    "pareto": 5,
    "selected": 10,
    "base_case": 11,
}

FIGURE_MARKER_STYLE = {
    "candidate_cloud_color": "#8A8A8A",
    "candidate_cloud_alpha": 0.13,
}

FIGURE_LEGEND_DEFAULTS = {
    "frameon": False,
    "fontsize": 6,
}


def configure_matplotlib_nature() -> None:
    cycle_colors: List[str] = [
        PALETTE["blue"],
        PALETTE["green"],
        PALETTE["orange"],
        PALETTE["purple"],
        PALETTE["red"],
        PALETTE["teal"],
        PALETTE["navy"],
        PALETTE["yellow"],
    ]
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.0,
            "axes.prop_cycle": cycler(color=cycle_colors),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def horizontal_ticklabels(ax) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")


def apply_publication_figure_standard(fig) -> None:
    """Apply the shared V11 manuscript figure standard before saving."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        try:
            ax.set_facecolor("white")
        except Exception:
            pass
        try:
            horizontal_ticklabels(ax)
        except Exception:
            pass
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)


def unique_legend_entries(handles: Sequence, labels: Sequence[str]) -> Tuple[List, List[str]]:
    unique_handles: List = []
    unique_labels: List[str] = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def figure_legend(fig, handles: Sequence, labels: Sequence[str], **kwargs):
    unique_handles, unique_labels = unique_legend_entries(handles, labels)
    if not unique_handles:
        return None
    legend_kwargs = dict(FIGURE_LEGEND_DEFAULTS)
    legend_kwargs.update(kwargs)
    return fig.legend(unique_handles, unique_labels, **legend_kwargs)


def style_axis(ax, grid: bool = False) -> None:
    if grid:
        ax.grid(True, which="major")
    for spine in ("left", "bottom"):
        if spine in ax.spines:
            ax.spines[spine].set_linewidth(0.8)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def savefig(fig, path: Path, *, tight: bool = True, pad_inches: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_publication_figure_standard(fig)
    out_path = path.with_suffix(FIGURE_OUTPUT_SUFFIX)
    if tight:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    else:
        fig.savefig(out_path, facecolor="white")
    plt.close(fig)
