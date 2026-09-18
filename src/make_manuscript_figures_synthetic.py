"""Build the core synthetic-case manuscript figures from a trained XGBoost model bundle:
feature importance, predicted-vs-true density plot, error histogram, and train/test
profile fits (with per-example model/data RMSE).

Generalizes the figure-producing cells of notebooks/TrainAndTestXGBoostSynthetic.ipynb
(correlation heatmap, scatter matrix, FeatureImportance, DensityPlot, Histogram,
"Train Models"/"Test Models") using the same 400k-sample, 9-layer database and model
bundle already produced by src/train_test_xgboost_synthetic.py
(Models/xgb_9layers_new400.joblib). Out of scope here: the "different grid"
buried-layer test scenario and its GN-refined comparison, which are already covered
(and superseded, for the GN part) by src/gn_baseline_synthetic.py and
src/make_manuscript_figures_r1.py's GN_Baseline_Comparison figure -- see
src/compare_synthetic_field.py for the synthetic-vs-field version of that scenario.

Run inside the `xgb-fdem` conda env (needs xgboost + pygimli/empymod via FDEM.py):
    conda run -n xgb-fdem python src/make_manuscript_figures_synthetic.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from FDEM import FDEM  # noqa: E402
from manuscript_figures_common import (  # noqa: E402
    pick_random_examples, per_layer_rmse_normalized, plot_correlation_heatmap,
    plot_density_grid, plot_error_histogram, plot_feature_importance_bars,
    plot_profile_fit_grid, plot_scatter_matrix, plot_scatter_nrmse, reconstruct_split,
)
from train_test_xgboost_synthetic import build_dataset, build_features  # noqa: E402

MODELS_DIR = ROOT_DIR / "Models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_9layers_new400.joblib"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Results" / "manuscript_figures_synthetic"
FIGURES_DIR = ROOT_DIR / "MachineLearningFDEM" / "figures"

THK = np.logspace(-0.3, 0.35, 8)
HEIGHT = 0.1
N_LAYERS = 9
TAG = "new400"

# Feature order matches build_features(): amp for every CONFIG, then phase for every
# CONFIG (CONFIGS = h2,h4,h8,v2,v4,v8,p2,p4,p8). Colors/hatch match the manuscript's
# original FeatureImportance.eps: shade darkens with spacing, hatched = phase.
PIE_KEYS = [
    r"$A_{\text{HCP-2}}$", r"$A_{\text{HCP-4}}$", r"$A_{\text{HCP-8}}$",
    r"$A_{\text{VCP-2}}$", r"$A_{\text{VCP-4}}$", r"$A_{\text{VCP-8}}$",
    r"$A_{\text{PRP-2.1}}$", r"$A_{\text{PRP-4.1}}$", r"$A_{\text{PRP-8.1}}$",
    r"$\varphi_{\text{HCP-2}}$", r"$\varphi_{\text{HCP-4}}$", r"$\varphi_{\text{HCP-8}}$",
    r"$\varphi_{\text{VCP-2}}$", r"$\varphi_{\text{VCP-4}}$", r"$\varphi_{\text{VCP-8}}$",
    r"$\varphi_{\text{PRP-2.1}}$", r"$\varphi_{\text{PRP-4.1}}$", r"$\varphi_{\text{PRP-8.1}}$",
]
PIE_COLORS = (["lightblue", "blue", "darkblue", "lightgreen", "green", "darkgreen",
               "salmon", "red", "darkred"] * 2)
PIE_HATCH = [""] * 9 + ["/"] * 9


def data_fit_norm(models_pred_sigma, scale_X):
    """Forward model each predicted conductivity profile and re-encode it into
    normalized amp/phase features, so predicted models can be scored on how well
    they reproduce the FDEM response (not just the conductivities themselves)."""
    data_pred_raw = np.array([FDEM(m, THK, height=HEIGHT) for m in models_pred_sigma])
    return scale_X.transform(build_features(data_pred_raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used to train the bundle")
    parser.add_argument("--example-seed", type=int, default=10, help="Seed for picking the 10 example profiles")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_path)
    models, scale_X, scale_y = bundle["models"], bundle["scale_X"], bundle["scale_y"]

    X, y = build_dataset(N_LAYERS, TAG, args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = reconstruct_split(bundle, X, y, args.seed)

    plot_correlation_heatmap(X, output_dir / "CorrelationMatrix.png")
    plot_scatter_matrix(X, output_dir / "ScatterMatrix.png")
    print("[figure] correlation heatmap + scatter matrix saved")

    plot_feature_importance_bars(models, PIE_KEYS, PIE_COLORS, PIE_HATCH,
                                  output_dir / "FeatureImportance.png")
    plot_feature_importance_bars(models, PIE_KEYS, PIE_COLORS, PIE_HATCH,
                                  FIGURES_DIR / "FeatureImportance.eps")
    print("[figure] FeatureImportance.png (+ figures/FeatureImportance.eps)")

    y_pred_test = np.column_stack([m.predict(X_test) for m in models])
    per_layer_rmse = per_layer_rmse_normalized(y_test, y_pred_test)
    y_test_sigmas = 10 ** scale_y.inverse_transform(y_test)
    y_pred_sigmas = 10 ** scale_y.inverse_transform(y_pred_test)

    plot_density_grid(y_test_sigmas, y_pred_sigmas, per_layer_rmse, output_dir / "DensityPlot.png")
    plot_density_grid(y_test_sigmas, y_pred_sigmas, per_layer_rmse, FIGURES_DIR / "DensityPlot.eps")
    print("[figure] DensityPlot.png (+ figures/DensityPlot.eps)")
    plot_error_histogram(y_test_sigmas, y_pred_sigmas, output_dir / "Histogram.png")
    plot_error_histogram(y_test_sigmas, y_pred_sigmas, FIGURES_DIR / "Histogram.eps")
    print("[figure] Histogram.png (+ figures/Histogram.eps)")
    plot_scatter_nrmse(y_test_sigmas, y_pred_sigmas, output_dir / "ScatterNRMSE.png", n_samples=1000)
    plot_scatter_nrmse(y_test_sigmas, y_pred_sigmas, FIGURES_DIR / "ScatterNRMSE.eps", n_samples=1000)
    print("[figure] ScatterNRMSE.png (+ figures/ScatterNRMSE.eps)")

    for split_name, X_split, y_split, title in [
        ("train", X_train, y_train, "Train Models"),
        ("test", X_test, y_test, "Test Models"),
    ]:
        data_norm, models_norm = pick_random_examples(X_split, y_split, n=10, seed=args.example_seed)
        models_pred_norm = np.column_stack([m.predict(data_norm) for m in models])
        models_true_sigma = 10 ** scale_y.inverse_transform(models_norm)
        models_pred_sigma = 10 ** scale_y.inverse_transform(models_pred_norm)
        data_pred_norm = data_fit_norm(models_pred_sigma, scale_X)
        plot_profile_fit_grid(models_true_sigma, models_pred_sigma, models_norm, models_pred_norm,
                               data_norm, data_pred_norm, THK, title,
                               output_dir / f"{title.replace(' ', '')}.png")
        plot_profile_fit_grid(models_true_sigma, models_pred_sigma, models_norm, models_pred_norm,
                               data_norm, data_pred_norm, THK, title,
                               FIGURES_DIR / f"{title.replace(' ', '')}.eps")
        print(f"[figure] {title.replace(' ', '')}.png (+ figures/{title.replace(' ', '')}.eps)")

    print(f"\nOverall test RMSE={np.sqrt(np.mean((y_test - y_pred_test) ** 2)):.5f}  "
          f"(matches src/train_test_xgboost_synthetic.py's reported metric for this bundle)")


if __name__ == "__main__":
    main()
