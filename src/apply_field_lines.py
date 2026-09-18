"""Apply the field-configuration XGBoost model to the real DUALEM field survey
(Data/Field_data.npy), refine each estimate with a per-position Gauss-Newton
inversion (matching the manuscript's "XGBoost + GN" figures), and reproduce the two
field-application manuscript figures:

  - Field_Line<N>.png: per-channel data fit (measured vs. XGBoost vs. XGBoost + GN)
    along one survey line, with an error subplot per channel.
  - Lines_Field.png: XGBoost-only vs. XGBoost + GN conductivity-section comparison
    for a handful of lines, annotated with the data RMSE each achieves.

Generalizes notebooks/TrainAndTestXGBoostField.ipynb's cells 70-79 ("Test on field
data"), which only ever predicted a single profile with XGBoost and never applied
GN refinement. Data/Field_data.npy actually holds 37 survey lines of 50 positions
each (1850 rows total, "Position" resets 1..50 every 50 rows); this script predicts
every line but only runs GN refinement (one pygimli inversion per position) for the
lines actually being plotted, since that's the expensive step.

The GN step reuses the exact conductivity-only 1D forward model and inversion setup
already established for synthetic data in src/gn_baseline_synthetic.py (fixed 9-layer
thickness grid, TransLogLU(1e-3, 1) S/m bounds, 1% assumed relative data error),
just built on FDEM_field's channel set and pointed at real measurements instead of a
synthetic transect, and started from each position's XGBoost prediction.

Requires Models/xgb_9layers_new400_field.joblib (src/train_test_xgboost_field.py).

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/apply_field_lines.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygimli as pg
from matplotlib.colors import LogNorm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from FDEM import FDEM_field  # noqa: E402
from gn_baseline_synthetic import SIGMA_LOWER, SIGMA_UPPER, regrid  # noqa: E402
from manuscript_figures_common import op_ip_to_amp_pha  # noqa: E402
from train_test_xgboost_field import CONFIGS, N_LAYERS  # noqa: E402

DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "Models"
DEFAULT_MODEL_PATH = MODELS_DIR / f"xgb_{N_LAYERS}layers_new400_field.joblib"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "MachineLearningFDEM" / "figures"

THK = np.logspace(-0.3, 0.35, 8)
HEIGHT = 0.47  # FDEM_field's default instrument height
N_POS_PER_LINE = 50
GN_RELATIVE_ERROR = 0.01  # matches src/gn_baseline_synthetic.py's convention

RAW_COLUMNS = ["X", "Y", "Position", "Z", "H2Q", "H4Q", "H8Q", "V2Q", "V4Q", "V8Q",
               "P2Q", "P4Q", "P8Q", "H4IP", "H8IP", "V4IP", "V8IP"]
OP_COLS = {"h4": "H4Q", "h8": "H8Q", "v4": "V4Q", "v8": "V8Q"}
IP_COLS = {"h4": "H4IP", "h8": "H8IP", "v4": "V4IP", "v8": "V8IP"}
# Real measurement order: quadrature block then in-phase block, H then V (matches
# Field_data.npy's own column order after dropping the unusable 2m/PRP channels).
MEASURED_COLS = ["H4Q", "H8Q", "V4Q", "V8Q", "H4IP", "H8IP", "V4IP", "V8IP"]
# Index of each MEASURED_COLS entry inside FDEM_field()'s raw 10-value output
# [h2op, h4op, h8op, h4ip, h8ip, v2op, v4op, v8op, v4ip, v8ip].
RAW10_INDEX = [1, 2, 6, 7, 3, 4, 8, 9]
# (row_label, channel_label, index into the 8-channel MEASURED_COLS/RAW10_INDEX order)
CHANNEL_LAYOUT = [
    [("HCP Quadrature", "s=4m", 0), ("HCP Quadrature", "s=8m", 1),
     ("HCP In-Phase", "s=4m", 4), ("HCP In-Phase", "s=8m", 5)],
    [("VCP Quadrature", "s=4m", 2), ("VCP Quadrature", "s=8m", 3),
     ("VCP In-Phase", "s=4m", 6), ("VCP In-Phase", "s=8m", 7)],
]


def load_field_lines():
    raw = np.load(DATA_DIR / "Field_data.npy")
    df = pd.DataFrame(raw, columns=RAW_COLUMNS)
    n_lines = len(df) // N_POS_PER_LINE
    if len(df) % N_POS_PER_LINE != 0:
        raise ValueError(f"Field_data.npy has {len(df)} rows, not a multiple of {N_POS_PER_LINE}")
    return df, n_lines


def build_features_for_lines(df):
    feats = {}
    for cfg in CONFIGS:
        amp, pha = op_ip_to_amp_pha(df[OP_COLS[cfg]].to_numpy(), df[IP_COLS[cfg]].to_numpy())
        feats[f"{cfg}amp"] = amp
        feats[f"{cfg}pha"] = pha
    amp_cols = [f"{c}amp" for c in CONFIGS]
    pha_cols = [f"{c}pha" for c in CONFIGS]
    return pd.DataFrame(feats, columns=amp_cols + pha_cols)


def predict_all_lines(df, bundle):
    """XGBoost-only prediction for every line (fast; used for all 37 lines)."""
    X = build_features_for_lines(df)
    X_norm = bundle["scale_X"].transform(X)
    y_norm = np.column_stack([m.predict(X_norm) for m in bundle["models"]])
    return 10 ** bundle["scale_y"].inverse_transform(y_norm)  # (n_rows, N_LAYERS) sigmas, S/m


def forward_usable(sigma):
    """Forward model a conductivity profile and keep only the 8 channels the real
    instrument records, in MEASURED_COLS order."""
    raw10 = FDEM_field(sigma, THK, height=HEIGHT)
    return raw10[RAW10_INDEX]


class SigmaOnly1DModellingField(pg.Modelling):
    """pygimli forward operator over conductivity only, on the field instrument's
    8 usable channels; thickness is fixed at THK (same grid the XGBoost model was
    trained on), matching src/gn_baseline_synthetic.py's synthetic-case setup."""

    def __init__(self, thk=THK, height=HEIGHT):
        self.thk = np.asarray(thk)
        self.height = height
        super().__init__()
        self.setMesh(pg.meshtools.createMesh1D(len(self.thk) + 1))

    def response(self, model):
        return forward_usable(np.asarray(model))


def run_gn_field(data_measured, start_sigma, relative_error=GN_RELATIVE_ERROR):
    """Per-position GN refinement starting from the XGBoost prediction.

    data_measured: (npos, 8) real measurements, MEASURED_COLS order.
    start_sigma: (npos, N_LAYERS) XGBoost-predicted starting conductivities, S/m.
    """
    fop = SigmaOnly1DModellingField()
    trans = pg.trans.TransLogLU(SIGMA_LOWER, SIGMA_UPPER)

    npos = data_measured.shape[0]
    estimates = np.zeros((npos, N_LAYERS))
    responses = np.zeros_like(data_measured)
    for i in range(npos):
        inv = pg.Inversion()
        inv.setForwardOperator(fop)
        inv.modelTrans = trans
        start = pg.Vector(np.asarray(start_sigma[i], dtype=float))
        rel_err = np.full(data_measured.shape[1], relative_error)
        est = inv.run(data_measured[i], rel_err, startModel=start, verbose=False)
        estimates[i] = np.asarray(est)
        responses[i] = np.asarray(inv.response)
    return estimates, responses


def line_slice(line_number):
    return slice((line_number - 1) * N_POS_PER_LINE, line_number * N_POS_PER_LINE)


def process_line(df, models_pred, line_number):
    """Return everything plot_field_line_datafit/plot_lines_overview need for one line."""
    sl = line_slice(line_number)
    sub = df.iloc[sl]
    measured = sub[MEASURED_COLS].to_numpy()
    xgb_sigma = models_pred[sl]
    xgb_response = np.array([forward_usable(s) for s in xgb_sigma])
    gn_sigma, gn_response = run_gn_field(measured, xgb_sigma)

    position = sub["Position"].to_numpy()
    elevation = sub["Z"].to_numpy(dtype=float)
    if np.any(np.isnan(elevation)):
        valid = ~np.isnan(elevation)
        elevation = np.interp(position, position[valid], elevation[valid])

    return {
        "position": position,
        "elevation": elevation,
        "measured": measured,
        "xgb_sigma": xgb_sigma, "xgb_response": xgb_response,
        "gn_sigma": gn_sigma, "gn_response": gn_response,
    }


def plot_field_line_datafit(line_number, result, out_path):
    fig, ax = plt.subplots(4, 4, figsize=(12, 8), sharex=True)
    position = result["position"]

    for block_row, block in enumerate(CHANNEL_LAYOUT):
        data_row, err_row = block_row * 2, block_row * 2 + 1
        for col, (row_label, spacing_label, ch) in enumerate(block):
            true_ppt = result["measured"][:, ch] * 1000
            xgb_ppt = result["xgb_response"][:, ch] * 1000
            gn_ppt = result["gn_response"][:, ch] * 1000

            a = ax[data_row, col]
            a.plot(position, true_ppt, "-r", label="True")
            a.plot(position, xgb_ppt, "-.g", label="XGBoost")
            a.plot(position, gn_ppt, "--b", label="XGBoost + GN")
            a.set_title(f"{row_label} {spacing_label}", fontsize=10)

            e = ax[err_row, col]
            e.plot(position, np.abs(true_ppt - xgb_ppt), "-.g")
            e.plot(position, np.abs(true_ppt - gn_ppt), "--b")
            e.set_title("Error", fontsize=9)

        ax[data_row, 0].set_ylabel("[ppt]", fontsize=9)
        ax[err_row, 0].set_ylabel("[ppt]", fontsize=9)

    for col in range(4):
        ax[-1, col].set_xlabel("Distance [m]", fontsize=9)
    ax[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))

    fig.suptitle(f"Field Line {line_number}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".eps"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def regrid_topo(thk_per_pos, sigma_per_pos, elevation, y_abs):
    """Like gn_baseline_synthetic.regrid, but sampled on a shared *absolute*
    elevation grid (y_abs) instead of a shared relative-depth grid: each column's
    depth-below-surface is elevation[i] - y_abs, which varies per column. Depths
    beyond the last real interface are clipped to the deepest (half-space) layer,
    same as regrid -- so the last layer's color naturally fills down to y_abs's
    common bottom limit for every column, instead of stopping at a fixed relative
    depth that leaves a jagged gap wherever the terrain sits higher than average.
    """
    npos, nlay = sigma_per_pos.shape
    out = np.full((npos, len(y_abs)), np.nan)
    for i in range(npos):
        bounds = np.hstack(([0.0], np.cumsum(thk_per_pos[i]), [np.inf]))
        depth_i = elevation[i] - y_abs
        above_ground = depth_i < 0
        layer_idx = np.clip(np.searchsorted(bounds, depth_i[~above_ground], side="right") - 1, 0, nlay - 1)
        out[i, ~above_ground] = sigma_per_pos[i, layer_idx]
    return out


def plot_lines_overview(lines, results, out_path, depthmax=10.0, vmin=1, vmax=1000, ny=201):
    """Topographic section: each position's conductivity column is vertically placed
    at its real ground elevation minus depth (using Field_data.npy's Z column),
    rather than a flat depth-below-instrument reference -- so the section follows
    actual terrain relief along the survey line. The bottom of the section is a
    common absolute elevation (the lowest point's full depthmax) for every column,
    so the deepest (half-space) layer's color fills all the way down instead of
    stopping at a fixed relative depth."""
    fig, ax = plt.subplots(len(lines), 2, figsize=(10, 3 * len(lines)), sharex=True, squeeze=False,
                            constrained_layout=True)

    for row, line_number in enumerate(lines):
        r = results[line_number]
        position = r["position"]
        elevation = r["elevation"]
        y_abs = np.linspace(elevation.max(), elevation.min() - depthmax, ny)
        X = np.tile(position, (len(y_abs), 1))
        Y = np.tile(y_abs[:, None], (1, len(position)))

        xgb_grid = regrid_topo(np.tile(THK, (len(position), 1)), r["xgb_sigma"], elevation, y_abs)
        gn_grid = regrid_topo(np.tile(THK, (len(position), 1)), r["gn_sigma"], elevation, y_abs)
        xgb_rmse_ppt = float(np.sqrt(np.mean((r["measured"] - r["xgb_response"]) ** 2)) * 1000)
        gn_rmse_ppt = float(np.sqrt(np.mean((r["measured"] - r["gn_response"]) ** 2)) * 1000)

        for col, (grid, name, rmse) in enumerate([
            (xgb_grid, "XGBoost", xgb_rmse_ppt), (gn_grid, "XGBoost + GN", gn_rmse_ppt)
        ]):
            a = ax[row, col]
            im = a.pcolormesh(X, Y, grid.T * 1000, shading="auto",
                               norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
            a.plot(position, elevation, "-", color="white", linewidth=1)
            a.set_title(f"Line {line_number} - {name}", fontsize=10)
            a.text(0.03, 0.05, f"Data RMSE: {rmse:.2f} ppt", transform=a.transAxes,
                   color="white", fontsize=8, va="bottom")
            # Survey lines run SW->NE (increasing position), confirmed against
            # Field_data.npy's X/Y coordinates -- label section orientation.
            a.text(0.02, 0.95, "SW", transform=a.transAxes, color="black",
                   fontsize=9, fontweight="bold", va="top", ha="left")
            a.text(0.98, 0.95, "NE", transform=a.transAxes, color="black",
                   fontsize=9, fontweight="bold", va="top", ha="right")
            if col == 0:
                a.set_ylabel("Elevation [m]")
        fig.colorbar(im, ax=ax[row, :], label="mS/m", location="right", shrink=0.8)

    for col in range(2):
        ax[-1, col].set_xlabel("Distance [m]")
    fig.savefig(out_path.with_suffix(".eps"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--lines", type=int, nargs="+", default=[1, 18, 36],
                         help="1-indexed line numbers for per-line data-fit figures")
    parser.add_argument("--overview-lines", type=int, nargs="+", default=[1, 2, 3],
                         help="1-indexed line numbers for the Lines_Field.png section comparison")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-predictions", type=str, default=str(MODELS_DIR / "Models_9Layers_Field.npy"),
                         help="Where to save all lines' XGBoost-only predicted models (set to '' to skip)")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"{args.model_path} not found -- train it first with:\n"
            f"  conda run -n xgb-fdem python src/train_test_xgboost_field.py"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, n_lines = load_field_lines()
    print(f"[data] {len(df)} positions across {n_lines} lines")

    bundle = joblib.load(args.model_path)
    models_pred = predict_all_lines(df, bundle)

    if args.save_predictions:
        np.save(args.save_predictions, models_pred)
        print(f"[predictions] all lines (XGBoost-only) -> {args.save_predictions}")

    needed_lines = sorted(set(args.lines) | set(args.overview_lines))
    results = {}
    for line in needed_lines:
        if not (1 <= line <= n_lines):
            print(f"[skip] line {line} out of range (1..{n_lines})")
            continue
        print(f"[gn] refining line {line} ({N_POS_PER_LINE} positions)...")
        results[line] = process_line(df, models_pred, line)

    for line in args.lines:
        if line not in results:
            continue
        out_path = output_dir / f"Field_Line{line}.png"
        plot_field_line_datafit(line, results[line], out_path)
        print(f"[figure] {out_path}")

    overview_lines = [l for l in args.overview_lines if l in results]
    if overview_lines:
        overview_path = output_dir / "Lines_Field.png"
        plot_lines_overview(overview_lines, results, overview_path)
        print(f"[figure] {overview_path}")


if __name__ == "__main__":
    main()
