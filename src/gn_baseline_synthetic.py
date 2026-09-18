"""Conventional Gauss-Newton baseline (homogeneous starting model) vs. XGBoost vs.
XGBoost + GN, for the two synthetic transects used in the manuscript's "Numerical
examples" section (Figures MC/"middle conductive body" and MR/"middle resistive body").

Addresses reviewer comments R1-1 / R3-1: reviewers asked for a comparison against a
conventional inversion that starts from a homogeneous model, rather than only showing
XGBoost and "XGBoost + GN" (which starts GN from the XGBoost prediction).

For each of the two scenarios, and for noise-free / 5% / 10% relative noise (matching
the noise levels already used in the manuscript figures), this script:

  1. Builds the true 3-layer transect (fixed thicknesses h1, varying middle-layer
     thickness h2 across position, per Maria's original notebook cell), and forward
     models the noise-free FDEM response with the same physics used everywhere else
     in this repo (src/FDEM.py).
  2. Predicts electrical conductivity with a trained XGBoost model bundle (e.g.
     Models/xgb_9layers_new400.joblib, the 400k-sample model used for the manuscript,
     or Models/xgb_9layers_d10m4000k.joblib, trained on the 4M-sample database).
  3. Runs a plain Gauss-Newton inversion with pygimli, starting every sounding from a
     homogeneous conductivity model (same value at all 9 layers, on the same fixed
     9-layer thickness grid the XGBoost model was trained on) -- no XGBoost prediction
     involved. This is the new "conventional GN baseline".
  4. Runs the same GN inversion again, this time starting each sounding from its own
     XGBoost prediction ("XGBoost + GN"), matching the manuscript's existing Figures 7/8.
  5. Saves stitched-section comparison plots (True / XGBoost / GN-homogeneous /
     XGBoost + GN) and a CSV of model and data misfit for each scenario/noise-level
     combination.

Usage (run inside the `xgb-fdem` conda env, which has pygimli + empymod + xgboost):
    conda run -n xgb-fdem python src/gn_baseline_synthetic.py \\
        --model-path Models/xgb_9layers_d10m4000k.joblib
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from FDEM import FDEM  # noqa: E402
from train_test_xgboost_synthetic import add_noise, build_features  # noqa: E402

MODELS_DIR = ROOT_DIR / "Models"
RESULTS_DIR = ROOT_DIR / "Results" / "gn_baseline_synthetic"

# Same fixed 9-layer thickness grid used to generate the training database and to
# train Models/xgb_9layers_new400.joblib (src/create_database_synthetic.py,
# notebooks/CreateDatabaseSynthetic.ipynb). GN only ever inverts the 9 conductivities;
# thickness is held fixed here, exactly like the XGBoost prediction and like the
# original "XGBoost + GN" step in the manuscript.
THK = np.logspace(-0.3, 0.35, 8)
N_LAYERS = 9
HEIGHT = 0.1  # instrument height used for the synthetic database (m)

# Physically motivated conductivity bounds for the log-transform: the training
# database spans 1-1000 mS/m (see Manuscript.tex, "Generating synthetic data").
SIGMA_LOWER, SIGMA_UPPER = 1e-3, 1.0  # S/m

# Two synthetic transects, from Maria's original notebook cell (kept only as a
# comment there -- reconstructed here for R1-1/R3-1):
#   sig1 = 10/1000; sig2 = 500/1000; sig3 = 10/1000; h1 = 2; h2 = np.linspace(1.5, 3.5)
# MR (middle resistive body) mirrors MC with the conductive/background roles swapped.
SCENARIOS = {
    "MC": {
        "label": "Middle conductive body",
        "sig1": 10 / 1000,
        "sig2": 500 / 1000,
        "sig3": 10 / 1000,
    },
    "MR": {
        "label": "Middle resistive body",
        "sig1": 500 / 1000,
        "sig2": 10 / 1000,
        "sig3": 500 / 1000,
    },
}
H1 = 2.0
H2 = np.linspace(1.5, 3.5)  # 50 positions, matches the original notebook default
NOISE_LEVELS = [0.0, 0.05, 0.10]
SEED = 42


def build_transect(sig1, sig2, sig3, h1=H1, h2=H2):
    """3-layer true model per position: columns [thk1, thk2, sig1, sig2, sig3]."""
    npos = len(h2)
    transect = np.zeros((npos, 5))
    transect[:, 0] = h1
    transect[:, 1] = h2
    transect[:, 2] = sig1
    transect[:, 3] = sig2
    transect[:, 4] = sig3
    return transect


def forward_transect(transect, height=HEIGHT):
    """Noise-free FDEM response at every position of a 3-layer transect."""
    return np.array([FDEM(m[2:], m[:2], height=height) for m in transect])


class SigmaOnly1DModelling(pg.Modelling):
    """pygimli forward operator over conductivity only; thickness is fixed at `thk`.

    Same 9-parameter representation the XGBoost model was trained on, so the GN
    baseline and the XGBoost prediction are directly comparable model-for-model.
    """

    def __init__(self, thk, height=HEIGHT):
        self.thk = np.asarray(thk)
        self.height = height
        super().__init__()
        self.setMesh(pg.meshtools.createMesh1D(len(self.thk) + 1))

    def response(self, model):
        return FDEM(np.asarray(model), self.thk, height=self.height)


def predict_xgboost(data_noisy, bundle):
    """Predict electrical conductivity (S/m) on the fixed 9-layer grid."""
    X = build_features(data_noisy)
    X_scaled = bundle["scale_X"].transform(X)
    y_scaled = np.column_stack([m.predict(X_scaled) for m in bundle["models"]])
    y_log10 = bundle["scale_y"].inverse_transform(y_scaled)
    return 10 ** y_log10


def run_gn(data_noisy, relative_error, start_models):
    """Gauss-Newton inversion per position, starting each sounding from the
    corresponding row of `start_models` (shape (npos, N_LAYERS)).

    Pass a homogeneous array (same value repeated at every layer/position) for the
    conventional GN baseline, or the XGBoost prediction for "XGBoost + GN".
    """
    fop = SigmaOnly1DModelling(THK, HEIGHT)
    trans = pg.trans.TransLogLU(SIGMA_LOWER, SIGMA_UPPER)

    npos = data_noisy.shape[0]
    estimates = np.zeros((npos, N_LAYERS))
    responses = np.zeros_like(data_noisy)
    for i in range(npos):
        inv = pg.Inversion()
        inv.setForwardOperator(fop)
        inv.modelTrans = trans
        start = pg.Vector(np.asarray(start_models[i], dtype=float))
        rel_err = np.full(data_noisy.shape[1], relative_error)
        est = inv.run(data_noisy[i], rel_err, startModel=start, verbose=False)
        estimates[i] = np.asarray(est)
        responses[i] = np.asarray(inv.response)
    return estimates, responses


def homogeneous_start(npos, start_value):
    return np.full((npos, N_LAYERS), start_value)


def rel_pct_rmse(true, pred):
    """Relative-percent RMSE-like error metric, matching the convention already
    used in notebooks/TrainAndTestXGBoostSynthetic.ipynb for model/data error."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    return float(np.sum(np.sqrt(np.abs((true - pred) / true) * 100)) / true.size)


def regrid(thk_per_pos, sigma_per_pos, depthmax=10.0, ny=201):
    """Regrid a stitched section (arbitrary thickness/number of layers, possibly
    varying per position) onto a shared depth axis, for plotting/comparison."""
    npos, nlay = sigma_per_pos.shape
    y = np.linspace(0, depthmax, ny)
    out = np.zeros((npos, ny))
    for i in range(npos):
        bounds = np.hstack(([0.0], np.cumsum(thk_per_pos[i]), [np.inf]))
        layer_idx = np.clip(np.searchsorted(bounds, y, side="right") - 1, 0, nlay - 1)
        out[i] = sigma_per_pos[i, layer_idx]
    return out


def plot_comparison(true_grid, xgb_grid, gn_grid, xgb_gn_grid, positions, title,
                     out_path, depthmax=10.0):
    fig, ax = plt.subplots(4, figsize=(9, 9), sharex=True, constrained_layout=True)
    vmin, vmax = 1, 1000
    extent = [positions[0], positions[-1], depthmax, 0]
    imgs = []
    grids = [true_grid, xgb_grid, gn_grid, xgb_gn_grid]
    names = ["True model", "XGBoost", "GN (homogeneous start)", "XGBoost + GN"]
    for a, grid, name in zip(ax, grids, names):
        imgs.append(a.imshow(grid.T * 1000, extent=extent, aspect="auto",
                              norm="log", vmin=vmin, vmax=vmax, cmap="viridis"))
        a.set_title(name)
        a.set_ylabel("Depth [m]")
    ax[-1].set_xlabel("Position [m]")
    fig.colorbar(imgs[0], ax=ax, label="$\\sigma$ [mS/m]", location="bottom", shrink=0.6)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                         choices=list(SCENARIOS.keys()))
    parser.add_argument("--noise-levels", nargs="+", type=float, default=NOISE_LEVELS)
    parser.add_argument("--start-value", type=float, default=0.1,
                         help="Homogeneous GN starting conductivity, S/m (default 0.1 = 100 mS/m)")
    parser.add_argument("--model-path", type=str,
                         default=str(MODELS_DIR / "xgb_9layers_new400.joblib"))
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Default: Results/gn_baseline_synthetic_<model tag>, "
                              "derived from --model-path's filename")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_tag = model_path.stem.replace("xgb_9layers_", "")
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT_DIR / "Results" / f"gn_baseline_synthetic_{model_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    rows = []

    for key in args.scenarios:
        cfg = SCENARIOS[key]
        transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
        positions = np.arange(len(transect))
        data_true = forward_transect(transect)
        true_grid = regrid(transect[:, :2], transect[:, 2:])

        for noise_level in args.noise_levels:
            print(f"[{key}] noise={noise_level:.0%} ...")
            data_noisy = add_noise(data_true, noise_level, seed=SEED)

            xgb_pred = predict_xgboost(data_noisy, bundle)
            xgb_grid = regrid(np.tile(THK, (len(positions), 1)), xgb_pred)
            xgb_data = np.array([FDEM(s, THK, height=HEIGHT) for s in xgb_pred])

            rel_err = max(noise_level, 0.02)

            gn_pred, gn_data = run_gn(
                data_noisy, rel_err, homogeneous_start(len(positions), args.start_value)
            )
            gn_grid = regrid(np.tile(THK, (len(positions), 1)), gn_pred)

            xgb_gn_pred, xgb_gn_data = run_gn(data_noisy, rel_err, xgb_pred)
            xgb_gn_grid = regrid(np.tile(THK, (len(positions), 1)), xgb_gn_pred)

            tag = f"{key}_noise{int(round(noise_level * 100)):02d}"
            plot_comparison(
                true_grid, xgb_grid, gn_grid, xgb_gn_grid, positions,
                title=f"{cfg['label']} -- {noise_level:.0%} noise ({model_tag})",
                out_path=output_dir / f"{tag}.png",
            )

            rows.append({
                "scenario": key,
                "noise_level": noise_level,
                "model_error_xgboost_pct": rel_pct_rmse(true_grid, xgb_grid),
                "model_error_gn_homog_pct": rel_pct_rmse(true_grid, gn_grid),
                "model_error_xgboost_gn_pct": rel_pct_rmse(true_grid, xgb_gn_grid),
                "data_error_xgboost_pct": rel_pct_rmse(data_true, xgb_data),
                "data_error_gn_homog_pct": rel_pct_rmse(data_true, gn_data),
                "data_error_xgboost_gn_pct": rel_pct_rmse(data_true, xgb_gn_data),
            })

            np.save(output_dir / f"{tag}_xgb_pred.npy", xgb_pred)
            np.save(output_dir / f"{tag}_gn_pred.npy", gn_pred)
            np.save(output_dir / f"{tag}_xgb_gn_pred.npy", xgb_gn_pred)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"gn_baseline_summary_{model_tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[summary] -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
