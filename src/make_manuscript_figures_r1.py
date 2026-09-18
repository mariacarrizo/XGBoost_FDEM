"""Build the manuscript-ready figures/tables for the R1-1/R3-1 (GN baseline) and
R1-3 (training-set-size sufficiency) reviewer responses, from results already
produced by gn_baseline_synthetic.py, the learning-curve run (100k-800k), and the
noise-augmentation sufficiency sweep (400k-4M).

Writes into MachineLearningFDEM/figures/ (figures) and
Results/manuscript_tables/ (LaTeX table snippets to paste into Manuscript.tex).

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/make_manuscript_figures_r1.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from gn_baseline_synthetic import (  # noqa: E402
    SCENARIOS, build_transect, regrid, THK,
)

RESULTS_DIR = ROOT_DIR / "Results"
FIGURES_DIR = ROOT_DIR / "MachineLearningFDEM" / "figures"
TABLES_DIR = ROOT_DIR / "MachineLearningFDEM" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_0 = "#2a78d6"
COLOR_5 = "#eb6834"
COLOR_10 = "#1baf7a"
NOISE_COLORS = {0.0: COLOR_0, 0.05: COLOR_5, 0.10: COLOR_10}
NOISE_MARKERS = {0.0: "o", 0.05: "s", 0.10: "^"}


def save_fig(fig, out_path_no_ext, **kwargs):
    fig.savefig(out_path_no_ext.with_suffix(".eps"), **kwargs)
    fig.savefig(out_path_no_ext.with_suffix(".png"), dpi=200, **kwargs)


def to_latex_table(df, float_format="%.3g", col_align=None):
    def fmt(v):
        if isinstance(v, float):
            return float_format % v
        return str(v)
    align = col_align or ("l" * len(df.columns))
    lines = [
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(str(c) for c in df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# R1-1 / R3-1: GN-homogeneous baseline vs. XGBoost, combined MC+MR figure
# ---------------------------------------------------------------------------

# Which trained XGBoost model backs Figure 14 / Table 3 -- the noise-augmented,
# per-size-tuned 400k model (see R1-2/R1-3), not the original clean-data-only one.
GN_MODEL_TAG = "new400_noiseaug_tuned"


def make_gn_baseline_figure():
    bl_dir = RESULTS_DIR / "gn_baseline_synthetic"
    summary = pd.read_csv(bl_dir / f"gn_baseline_summary_{GN_MODEL_TAG}.csv")
    fig, axes = plt.subplots(4, 2, figsize=(8, 10), constrained_layout=True)
    vmin, vmax = 1, 1000
    depthmax = 10.0
    row_names = ["True model", "XGBoost", "GN (homogeneous start)", "XGBoost + GN"]
    row_methods = [None, "xgboost", "gn_homog", "xgboost_gn"]

    for col, key in enumerate(["MC", "MR"]):
        cfg = SCENARIOS[key]
        transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
        positions = np.arange(len(transect))
        true_grid = regrid(transect[:, :2], transect[:, 2:])
        top = transect[:, 0]
        bottom = transect[:, 0] + transect[:, 1]
        xgb_pred = np.load(bl_dir / f"{key}_noise00_xgb_pred.npy")
        gn_pred = np.load(bl_dir / f"{key}_noise00_gn_pred.npy")
        xgb_gn_path = bl_dir / f"{key}_noise00_xgb_gn_pred.npy"
        xgb_grid = regrid(np.tile(THK, (len(positions), 1)), xgb_pred)
        gn_grid = regrid(np.tile(THK, (len(positions), 1)), gn_pred)
        grids = [true_grid, xgb_grid, gn_grid]
        if xgb_gn_path.exists():
            xgb_gn_pred = np.load(xgb_gn_path)
            grids.append(regrid(np.tile(THK, (len(positions), 1)), xgb_gn_pred))
        else:
            grids.append(gn_grid)  # placeholder, unused if not present

        row = summary[(summary["scenario"] == key) & (np.isclose(summary["noise_level"], 0.0))].iloc[0]

        extent = [positions[0], positions[-1], depthmax, 0]
        for row_idx, (grid, name, method) in enumerate(zip(grids, row_names, row_methods)):
            ax = axes[row_idx, col]
            im = ax.imshow(grid.T * 1000, extent=extent, aspect="auto",
                            norm="log", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.plot(positions, top, ":", color="red", linewidth=1.2)
            ax.plot(positions, bottom, ":", color="red", linewidth=1.2)
            if method is not None:
                rmse_pct = row[f"model_error_{method}_pct"]
                ax.text(0.03, 0.06, f"RMSE $\\sigma$: {rmse_pct:.1f}%", color="white",
                        fontsize=8, transform=ax.transAxes, va="bottom", ha="left")
            if row_idx == 0:
                ax.set_title(cfg["label"], fontsize=11)
            if col == 0:
                ax.set_ylabel("Depth [m]", fontsize=9)
                ax.annotate(name, xy=(-0.34, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=10,
                            fontweight="bold")
            else:
                ax.set_ylabel("")
            if row_idx == 3:
                ax.set_xlabel("Position [m]")

    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.7)
    cbar.set_label("$\\sigma$ [mS/m]")
    save_fig(fig, FIGURES_DIR / "GN_Baseline_Comparison")
    plt.close(fig)
    print(f"[figure] -> {FIGURES_DIR / 'GN_Baseline_Comparison'}.eps (+ .png)")


def make_gn_baseline_table():
    df = pd.read_csv(RESULTS_DIR / "gn_baseline_synthetic" / f"gn_baseline_summary_{GN_MODEL_TAG}.csv")
    df["scenario"] = df["scenario"].map({"MC": "Middle conductive", "MR": "Middle resistive"})
    df["noise_level"] = (df["noise_level"] * 100).astype(int).astype(str) + "\\%"
    df = df.rename(columns={
        "scenario": "Scenario", "noise_level": "Noise",
        "model_error_xgboost_pct": "XGBoost (\\%)",
        "model_error_gn_homog_pct": "GN homog. (\\%)",
        "model_error_xgboost_gn_pct": "XGBoost+GN (\\%)",
    })
    model_err = df[["Scenario", "Noise", "XGBoost (\\%)", "GN homog. (\\%)", "XGBoost+GN (\\%)"]]
    with open(TABLES_DIR / "gn_baseline_model_error.tex", "w") as f:
        f.write(to_latex_table(model_err, float_format="%.1f", col_align="llrrr"))
    print(f"[table] -> {TABLES_DIR / 'gn_baseline_model_error.tex'}")

    df2 = pd.read_csv(RESULTS_DIR / "gn_baseline_synthetic" / f"gn_baseline_summary_{GN_MODEL_TAG}.csv")
    df2["scenario"] = df2["scenario"].map({"MC": "Middle conductive", "MR": "Middle resistive"})
    df2["noise_level"] = (df2["noise_level"] * 100).astype(int).astype(str) + "\\%"
    df2 = df2.rename(columns={
        "scenario": "Scenario", "noise_level": "Noise",
        "data_error_xgboost_pct": "XGBoost (\\%)",
        "data_error_gn_homog_pct": "GN homog. (\\%)",
        "data_error_xgboost_gn_pct": "XGBoost+GN (\\%)",
    })
    data_err = df2[["Scenario", "Noise", "XGBoost (\\%)", "GN homog. (\\%)", "XGBoost+GN (\\%)"]]
    with open(TABLES_DIR / "gn_baseline_data_error.tex", "w") as f:
        f.write(to_latex_table(data_err, float_format="%.1f", col_align="llrrr"))
    print(f"[table] -> {TABLES_DIR / 'gn_baseline_data_error.tex'}")


# ---------------------------------------------------------------------------
# R1-3: training-set-size sufficiency (learning curve 100k-800k + 400k-4M check)
# ---------------------------------------------------------------------------

def make_learning_curve_figure():
    lc = pd.read_csv(RESULTS_DIR / "learning_curve_d10m800k.csv")
    lc9 = lc[lc["n_layers"] == 9].sort_values("dataset_size")

    suff = pd.read_csv(RESULTS_DIR / "architecture_search_noiseaug" / "architecture_search_noiseaug_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(lc9["dataset_size"], lc9["test_rmse"], "o-", color=COLOR_0, linewidth=2, markersize=6)
    ax.axvline(400_000, color="#898781", linestyle=":", linewidth=1.2, zorder=0)
    ax.text(400_000 * 1.05, ax.get_ylim()[1], "chosen: 400k", color="#52514e", fontsize=8.5,
            va="top", ha="left")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Test RMSE (log$_{10}$ $\\sigma$, normalized)")
    ax.set_title("(a) 9-layer model, 100k–800k", fontsize=11)
    ax.set_xticks(lc9["dataset_size"])
    ax.set_xticklabels([f"{int(s / 1000)}k" for s in lc9["dataset_size"]],
                        rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    for noise_level, sub in suff.groupby("noise_level"):
        sub = sub.sort_values("n_train_samples")
        ax.plot(sub["n_train_samples"], sub["rmse_log10_sigma"],
                color=NOISE_COLORS[noise_level], marker=NOISE_MARKERS[noise_level],
                markersize=6, linewidth=2, label=f"{noise_level:.0%} noise")
    ax.axvline(400_000, color="#898781", linestyle=":", linewidth=1.2, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Held-out RMSE (log$_{10}$ $\\sigma$)")
    ax.set_title("(b) Extended check, 400k–4M", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "LearningCurve")
    plt.close(fig)
    print(f"[figure] -> {FIGURES_DIR / 'LearningCurve'}.eps (+ .png)")


def make_architecture_table():
    # Straight from architecture_search_noiseaug.py's own output -- no
    # intermediate reformatting script needed.
    df = pd.read_csv(RESULTS_DIR / "architecture_search_noiseaug" / "architectures_found_by_size.csv")
    size_labels = {"new400": "400k", "d10m800k": "800k", "d10m1600k": "1.6M",
                   "d10m3200k": "3.2M", "d10m4000k": "4M"}
    df["Training samples"] = df["tag"].map(size_labels)
    cols = ["Training samples", "max_depth", "n_estimators", "learning_rate",
            "reg_lambda", "reg_alpha"]
    df = df[cols].rename(columns={
        "max_depth": "max\\_depth", "n_estimators": "n\\_estimators",
        "learning_rate": "learning\\_rate", "reg_lambda": "reg\\_$\\lambda$",
        "reg_alpha": "reg\\_$\\alpha$",
    })
    with open(TABLES_DIR / "architecture_convergence.tex", "w") as f:
        f.write(to_latex_table(df, float_format="%.3g", col_align="lrrrrr"))
    print(f"[table] -> {TABLES_DIR / 'architecture_convergence.tex'}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_gn_baseline_figure()
    make_gn_baseline_table()
    make_learning_curve_figure()
    make_architecture_table()


if __name__ == "__main__":
    main()
