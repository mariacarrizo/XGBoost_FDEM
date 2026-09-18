"""Composite figure summarizing steps 1-3 of the architecture-selection methodology
run on the fixed-depth (10 m) synthetic database, including the 600k -> 4M
data-size extension that revised steps 2 and 3's conclusions.

Panel A (step 1, data size): test R2 vs. dataset size (100k-4M), one line per
layer count.
Panel B (step 2, model complexity ceiling): best max_depth (the depth-sweep
ceiling, fixed light regularization) by layer count, at 600k vs. 4M -- shows the
ceiling shift right (deeper trees become viable) once enough data supports them.
No 4M depth-sweep exists for layers 3-5 (not data-starved at 600k already, so
never re-swept); the 4M line starts at layer 6 to reflect that honestly.
Panel C (step 3, final architecture): test R2 by layer count for the final
per-layer-tuned architecture at 600k vs. 4M, showing the gain from correcting
the 600k-era conclusions.

Reads the CSVs already produced by learning_curve_synthetic.py,
model_complexity_synthetic.py, and architecture_search_synthetic.py, plus the
gap-fill model_complexity run and the blended final_architecture_per_layer_4M
table assembled after the 4M re-analysis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"

LAYER_COUNTS = list(range(3, 13))

# Ordinal single-hue (blue) ramp, palette.md steps 250->700 -- one step per layer
# count, light=easy (3 layers) to dark=hard (12 layers). Values copied verbatim
# from the documented palette's certified light-surface ordinal range.
ORDINAL_RAMP = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
LAYER_COLOR = dict(zip(LAYER_COUNTS, ORDINAL_RAMP))

# Categorical slots (palette.md slots 1/2) for the 600k-vs-4M comparison shared
# by panels B and C: blue = 600k (the original, data-starved conclusion),
# orange = 4M (the corrected, validated conclusion).
CAT_600K, CAT_4M = "#2a78d6", "#eb6834"

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS_LINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": AXIS_LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "font.family": "sans-serif",
})


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.grid(True, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def size_formatter(value, _pos):
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def load_panel_a():
    """100k-2M from one consistent pool (d10m2000k), plus the 4M point spliced
    in from the separate d10m4000k pool (only 2M/4M were evaluated there)."""
    up_to_2m = pd.read_csv(RESULTS_DIR / "learning_curve_d10m2000k.csv")
    four_m = pd.read_csv(RESULTS_DIR / "learning_curve_d10m4000k.csv")
    four_m = four_m[four_m.dataset_size == 4_000_000]
    return pd.concat([up_to_2m, four_m], ignore_index=True)


def load_panel_b():
    """Best-depth-per-layer at 600k (all layers) and at 4M (layers 6-12 only --
    the depth sweep was never re-run for 3-5 at 4M, since 600k already wasn't
    data-starved for those easier problems)."""
    b_600k = pd.read_csv(RESULTS_DIR / "model_complexity_d10m1000k_600000_best_depth.csv")
    b_4m_a = pd.read_csv(RESULTS_DIR / "model_complexity_d10m4000k_4000000_best_depth.csv")
    b_4m_b = pd.read_csv(RESULTS_DIR / "model_complexity_d10m4000k_gapfill" / "model_complexity_d10m4000k_4000000_best_depth.csv")
    b_4m = pd.concat([b_4m_a, b_4m_b], ignore_index=True)
    return b_600k, b_4m


def load_panel_c():
    """Final per-layer test R2 at 600k (the original step-3 joint search) vs.
    4M (the blended search + depth-sweep answer, validated end to end)."""
    c_600k = pd.read_csv(RESULTS_DIR / "architecture_search_d10m_600k" / "architecture_search_summary.csv")
    c_4m = pd.read_csv(RESULTS_DIR / "final_architecture_per_layer_4M.csv")
    return c_600k.set_index("n_layers")["test_r2"], c_4m.set_index("n_layers")["test_r2_4M"]


def main():
    panel_a = load_panel_a()
    b_600k, b_4m = load_panel_b()
    c_600k, c_4m = load_panel_c()

    fig = plt.figure(figsize=(15.5, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1.05], wspace=0.5)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[0, 3])

    # --- Panel A: step 1, data size (100k -> 4M) -------------------------------------
    for n in LAYER_COUNTS:
        sub = panel_a[panel_a.n_layers == n].sort_values("dataset_size")
        axA.plot(sub.dataset_size, sub.test_r2, color=LAYER_COLOR[n], linewidth=1.8)
    axA.set_xscale("log")
    axA.xaxis.set_major_formatter(mticker.FuncFormatter(size_formatter))
    axA.set_xticks([100_000, 300_000, 1_000_000, 4_000_000])
    axA.set_xlabel("Dataset size (samples)")
    axA.set_ylabel("Test R²")
    axA.text(0.0, 1.08, "a", transform=axA.transAxes, fontsize=12, fontweight="bold")
    axA.set_title("Step 1 — Data size", loc="left", fontsize=10.5, color=SECONDARY)
    style_axis(axA)

    # --- Panel B: step 2, complexity ceiling shift (600k vs 4M) ----------------------
    b600_sorted = b_600k.sort_values("n_layers")
    axB.plot(b600_sorted.n_layers, b600_sorted.max_depth, color=CAT_600K,
              marker="o", markersize=5, linewidth=1.8, label="600k")
    b4m_sorted = b_4m.sort_values("n_layers")
    axB.plot(b4m_sorted.n_layers, b4m_sorted.max_depth, color=CAT_4M,
              marker="o", markersize=5, linewidth=1.8, label="4M")
    axB.set_xlabel("Number of layers")
    axB.set_ylabel("Complexity ceiling (best max_depth)")
    axB.set_xticks(LAYER_COUNTS)
    axB.text(0.0, 1.08, "b", transform=axB.transAxes, fontsize=12, fontweight="bold")
    axB.set_title("Step 2 — Complexity ceiling shift", loc="left", fontsize=10.5, color=SECONDARY)
    axB.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5, title="Dataset size")
    style_axis(axB)

    # Shared discrete colorbar for the ordinal "number of layers" ramp (panel A only,
    # kept in its own column so all three panels stay aligned)
    cmap = ListedColormap(ORDINAL_RAMP)
    bounds = np.arange(2.5, 13.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, ticks=LAYER_COUNTS)
    cb.set_label("Number of layers", fontsize=9, color=SECONDARY)
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=8, color=SECONDARY, length=0)

    # --- Panel C: step 3, final architecture outcome (600k vs 4M) --------------------
    layers_sorted = sorted(c_600k.index)
    axC.plot(layers_sorted, [c_600k[n] for n in layers_sorted], color=CAT_600K,
              marker="o", markersize=5, linewidth=1.8, label="600k-tuned (original)")
    axC.plot(layers_sorted, [c_4m[n] for n in layers_sorted], color=CAT_4M,
              marker="o", markersize=5, linewidth=1.8, label="4M-tuned (validated)")
    axC.set_xlabel("Number of layers")
    axC.set_ylabel("Test R²")
    axC.set_xticks(layers_sorted)
    axC.text(0.0, 1.08, "c", transform=axC.transAxes, fontsize=12, fontweight="bold")
    axC.set_title("Step 3 — Final architecture", loc="left", fontsize=10.5, color=SECONDARY)
    axC.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5)
    style_axis(axC)

    fig.suptitle(
        "Selecting dataset size, model complexity, and architecture\n"
        "for the fixed-depth (10 m) synthetic database, across layer counts 3–12",
        fontsize=12, y=1.08,
    )

    out_png = RESULTS_DIR / "figure_steps123_summary.png"
    out_pdf = RESULTS_DIR / "figure_steps123_summary.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved -> {out_png}")
    print(f"[figure] saved -> {out_pdf}")


if __name__ == "__main__":
    main()
