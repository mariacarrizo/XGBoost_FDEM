"""Regenerate the "Numerical examples" figures (manuscript Figures MC/MR and their
data-fit companions, fig:DataMC/fig:DataMR) using the noise-augmented, per-size-tuned
XGBoost model, replacing the original static Picture4.jpg/Picture1.jpg/Data_9L_MC.eps/
Data_9L_MR.eps.

Reuses the predictions already computed by gn_baseline_synthetic.py (run against
Models/xgb_9layers_new400_noiseaug_tuned.joblib, output in
Results/gn_baseline_synthetic/) -- no new inference needed for the section
figure. The data-fit figure additionally forward-models the predicted sigma
profiles (XGBoost, conventional GN started from a homogeneous model, and
XGBoost + GN) back through FDEM() to compare simulated vs. true response, for
the noise-free case.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/make_numerical_example_figures.py
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

from FDEM import FDEM  # noqa: E402
from gn_baseline_synthetic import (  # noqa: E402
    SCENARIOS, build_transect, forward_transect, regrid, THK,
)

RESULTS_DIR = ROOT_DIR / "Results" / "gn_baseline_synthetic"
FIGURES_DIR = ROOT_DIR / "MachineLearningFDEM" / "figures"
SUMMARY_CSV = RESULTS_DIR / "gn_baseline_summary_new400_noiseaug_tuned.csv"

NOISE_LEVELS = [0.0, 0.05, 0.10]
INTERFACE_COLOR = "red"


def true_interfaces(transect):
    """Top/bottom depth of the embedded conductive/resistive layer, per position."""
    top = transect[:, 0]
    bottom = transect[:, 0] + transect[:, 1]
    return top, bottom


def load_metrics(key, noise_level, method):
    """Model/data error (%), matching the manuscript's Table 3/5 exactly."""
    df = pd.read_csv(SUMMARY_CSV)
    row = df[(df["scenario"] == key) & (np.isclose(df["noise_level"], noise_level))]
    model_pct = float(row[f"model_error_{method}_pct"].iloc[0])
    data_pct = float(row[f"data_error_{method}_pct"].iloc[0])
    return model_pct, data_pct


def save_fig(fig, out_path_no_ext, **kwargs):
    fig.savefig(out_path_no_ext.with_suffix(".eps"), **kwargs)
    fig.savefig(out_path_no_ext.with_suffix(".png"), dpi=200, **kwargs)


def make_section_figure(key):
    cfg = SCENARIOS[key]
    transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
    positions = np.arange(len(transect))
    true_grid = regrid(transect[:, :2], transect[:, 2:])
    top, bottom = true_interfaces(transect)

    fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
    vmin, vmax = 1, 1000
    depthmax = 10.0
    extent = [positions[0], positions[-1], depthmax, 0]
    row_names = ["True model", "0% noise", "5% noise", "10% noise"]
    method_keys = {"xgb": "xgboost", "gn": "gn_homog", "xgb_gn": "xgboost_gn"}
    col_titles = {"xgb": "XGBoost", "gn": "GN (homogeneous start)", "xgb_gn": "XGBoost + GN"}
    methods = ["xgb", "gn", "xgb_gn"]

    for row in range(4):
        if row > 0:
            noise_level = NOISE_LEVELS[row - 1]
        for col, method in enumerate(methods):
            ax = axes[row, col]
            if row == 0:
                grid = true_grid
            else:
                tag = f"{key}_noise{int(round(noise_level * 100)):02d}_{method}_pred"
                pred = np.load(RESULTS_DIR / f"{tag}.npy")
                grid = regrid(np.tile(THK, (len(positions), 1)), pred)
                model_pct, data_pct = load_metrics(key, noise_level, method_keys[method])
                ax.text(0.03, 0.06, f"Model Error: {model_pct:.1f} %\nData Error: {data_pct:.1f} %",
                        color="white", fontsize=7.5, transform=ax.transAxes, va="bottom", ha="left")
            im = ax.imshow(grid.T * 1000, extent=extent, aspect="auto",
                            norm="log", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.plot(positions, top, ":", color=INTERFACE_COLOR, linewidth=1.2)
            ax.plot(positions, bottom, ":", color=INTERFACE_COLOR, linewidth=1.2)
            if row == 0:
                ax.set_title(col_titles[method], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{row_names[row]}\nDepth [m]", fontsize=9)
            if row == 3:
                ax.set_xlabel("Position [m]")

    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.7)
    cbar.set_label("$\\sigma$ [mS/m]")
    fig.suptitle(cfg["label"])
    name = "NumericalExample_MC" if key == "MC" else "NumericalExample_MR"
    save_fig(fig, FIGURES_DIR / name)
    plt.close(fig)
    print(f"[figure] -> {FIGURES_DIR / name}.eps (+ .png)")


# Matches Maria's original notebook template (notebooks/TrainAndTestXGBoostSynthetic.ipynb):
# quadrature and in-phase shown as separate raw channels (in ppt, not combined into
# amplitude/phase), with a paired "Error %" subplot under each data subplot.
# FDEM() returns [HOP, HIP, VOP, VIP, POP, PIP] (3 offsets each) -- np.split gives
# that same grouping directly, so channel/offset indexing follows without remapping.
CHANNELS = [("op", "Quadrature"), ("ip", "In-Phase")]
COIL_ROWS = [("h", "HCP", "r", [2, 4, 8]), ("v", "VCP", "b", [2, 4, 8]), ("p", "PRP", "k", [2.1, 4.1, 8.1])]
METHOD_STYLES = [("True", "-", None), ("XGBoost", "--", "xgb"), ("GN (homogeneous start)", "-.", "gn"),
                  ("XGBoost + GN", ":", "xgb_gn")]


def split_channels(data):
    """FDEM()-shaped (npos, 18) array -> {"h"/"v"/"p": {"op"/"ip": (npos, 3)}}."""
    hop, hip, vop, vip, pop, pip = np.split(data, 6, axis=1)
    return {"h": {"op": hop, "ip": hip}, "v": {"op": vop, "ip": vip}, "p": {"op": pop, "ip": pip}}


def make_datafit_figure(key):
    cfg = SCENARIOS[key]
    transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
    positions = np.arange(len(transect))
    data_true = forward_transect(transect)

    xgb_pred = np.load(RESULTS_DIR / f"{key}_noise00_xgb_pred.npy")
    gn_pred = np.load(RESULTS_DIR / f"{key}_noise00_gn_pred.npy")
    xgb_gn_pred = np.load(RESULTS_DIR / f"{key}_noise00_xgb_gn_pred.npy")
    blocks = {
        None: split_channels(data_true),
        "xgb": split_channels(np.array([FDEM(s, THK, height=0.1) for s in xgb_pred])),
        "gn": split_channels(np.array([FDEM(s, THK, height=0.1) for s in gn_pred])),
        "xgb_gn": split_channels(np.array([FDEM(s, THK, height=0.1) for s in xgb_gn_pred])),
    }

    fig, axes = plt.subplots(6, 6, figsize=(14, 9), sharex=True, constrained_layout=True)

    for row_group, (prefix, coil_label, color, offsets) in enumerate(COIL_ROWS):
        data_row = row_group * 2
        err_row = data_row + 1
        for chan_idx, (chan, chan_label) in enumerate(CHANNELS):
            true_vals_raw = blocks[None][prefix][chan]
            for off_idx, offset in enumerate(offsets):
                col = chan_idx * 3 + off_idx
                ax_data, ax_err = axes[data_row, col], axes[err_row, col]
                first_col = (chan_idx == 0 and off_idx == 0)

                for method_label, ls, method_key in METHOD_STYLES:
                    vals_raw = blocks[method_key][prefix][chan][:, off_idx]
                    ax_data.plot(positions, vals_raw * 1000, ls, color=color,
                                 label=f"{coil_label} {method_label}" if first_col else None)
                    if method_key is not None:
                        err_pct = 100 * np.abs((true_vals_raw[:, off_idx] - vals_raw) / true_vals_raw[:, off_idx])
                        ax_err.plot(positions, err_pct, ls, color=color)

                ax_data.set_title(f"{coil_label} {offset}m ${chan_label}$", fontsize=9)
                ax_err.set_title("Error %", fontsize=9)
                if col == 0:
                    ax_data.set_ylabel("[ppt]", fontsize=8)
                if err_row == 5:
                    ax_err.set_xlabel("Position [m]", fontsize=9)

    for a in axes.ravel():
        a.tick_params(labelsize=8)

    handles, labels = [], []
    for a in axes.ravel():
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(f"Data fit -- {cfg['label']} (noise-free case)", y=1.12)
    name = "Data_9L_MC" if key == "MC" else "Data_9L_MR"
    save_fig(fig, FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] -> {FIGURES_DIR / name}.eps (+ .png)")


def main():
    for key in ["MC", "MR"]:
        make_section_figure(key)
        make_datafit_figure(key)


if __name__ == "__main__":
    main()
