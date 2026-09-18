"""Field-configuration equivalent of make_manuscript_figures_synthetic.py: builds
feature importance, density plot, error histogram, and train/test profile fits from
the field-configuration XGBoost model bundle (src/train_test_xgboost_field.py).

Generalizes the figure-producing cells of notebooks/TrainAndTestXGBoostField.ipynb
that are still consistent with the current field database's 10-column raw layout
(see the module docstring of src/train_test_xgboost_field.py for why several of that
notebook's own dataframe cells are not). Application to the real field survey lines
(Data/Field_data.npy) is handled separately by src/apply_field_lines.py, since that
doesn't need a train/test split.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/train_test_xgboost_field.py   # first, if not already trained
    conda run -n xgb-fdem python src/make_manuscript_figures_field.py
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

from FDEM import FDEM_field  # noqa: E402
from manuscript_figures_common import (  # noqa: E402
    pick_random_examples, per_layer_rmse_normalized, plot_correlation_heatmap,
    plot_density_grid, plot_error_histogram, plot_feature_importance_pie,
    plot_profile_fit_grid, plot_scatter_matrix, reconstruct_split,
)
from train_test_xgboost_field import N_LAYERS, TAG, build_dataset, build_features  # noqa: E402

MODELS_DIR = ROOT_DIR / "Models"
DEFAULT_MODEL_PATH = MODELS_DIR / f"xgb_{N_LAYERS}layers_{TAG}.joblib"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Results" / "manuscript_figures_field"

THK = np.logspace(-0.3, 0.35, 8)
HEIGHT = 0.47  # FDEM_field's default instrument height

# Feature order matches train_test_xgboost_field.build_features(): amp for h4,h8,v4,v8,
# then phase for the same. No 2 m/PRP channels -- the field instrument doesn't record
# an in-phase channel for them, so they can't form an amplitude/phase feature pair.
PIE_KEYS = ["HCP_4_AMP", "HCP_8_AMP", "VCP_4_AMP", "VCP_8_AMP",
            "HCP_4_PHA", "HCP_8_PHA", "VCP_4_PHA", "VCP_8_PHA"]
PIE_COLORS = ["blue", "darkblue", "green", "darkgreen", "blue", "darkblue", "green", "darkgreen"]
PIE_HATCH = ["", "", "", "", "/", "/", "/", "/"]


def data_fit_norm(models_pred_sigma, scale_X):
    data_pred_raw = np.array([FDEM_field(m, THK, height=HEIGHT) for m in models_pred_sigma])
    return scale_X.transform(build_features(data_pred_raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used to train the bundle")
    parser.add_argument("--example-seed", type=int, default=10, help="Seed for picking the 10 example profiles")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"{args.model_path} not found -- train it first with:\n"
            f"  conda run -n xgb-fdem python src/train_test_xgboost_field.py"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_path)
    models, scale_X, scale_y = bundle["models"], bundle["scale_X"], bundle["scale_y"]

    X, y = build_dataset(TAG, args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = reconstruct_split(bundle, X, y, args.seed)

    plot_correlation_heatmap(X, output_dir / "CorrelationMatrix.png")
    plot_scatter_matrix(X, output_dir / "ScatterMatrix.png")
    print("[figure] correlation heatmap + scatter matrix saved")

    plot_feature_importance_pie(models, PIE_KEYS, PIE_COLORS, PIE_HATCH,
                                 output_dir / "FeatureImportance.png")
    print("[figure] FeatureImportance.png")

    y_pred_test = np.column_stack([m.predict(X_test) for m in models])
    per_layer_rmse = per_layer_rmse_normalized(y_test, y_pred_test)
    y_test_sigmas = 10 ** scale_y.inverse_transform(y_test)
    y_pred_sigmas = 10 ** scale_y.inverse_transform(y_pred_test)

    plot_density_grid(y_test_sigmas, y_pred_sigmas, per_layer_rmse, output_dir / "DensityPlot.png")
    print("[figure] DensityPlot.png")
    plot_error_histogram(y_test_sigmas, y_pred_sigmas, output_dir / "Histogram.png")
    print("[figure] Histogram.png")

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
        print(f"[figure] {title.replace(' ', '')}.png")

    print(f"\nOverall test RMSE={np.sqrt(np.mean((y_test - y_pred_test) ** 2)):.5f}  "
          f"(matches src/train_test_xgboost_field.py's reported metric for this bundle)")


if __name__ == "__main__":
    main()
