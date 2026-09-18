"""Compare the synthetic (18-feature, 9-config) and field (8-feature, 4-config)
XGBoost models: overall/per-layer test metrics side by side, plus how each recovers
the same buried-layer transects (the "MC"/"MR" scenarios already used for the
manuscript's GN-baseline comparison, see src/gn_baseline_synthetic.py) when each
model only gets the measurements its own instrument configuration would actually
record.

This answers the "does the reduced field channel set cost us resolution" question
directly: same true subsurface, same depth grid, forward modeled once with the full
synthetic instrument response and once with the field instrument's response, each
fed to the model trained for that configuration.

Requires both Models/xgb_9layers_new400.joblib and Models/xgb_9layers_new400_field.joblib.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/compare_synthetic_field.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from FDEM import FDEM, FDEM_field  # noqa: E402
from gn_baseline_synthetic import (  # noqa: E402
    SCENARIOS, THK, build_transect, regrid, rel_pct_rmse,
)
from train_test_xgboost_field import build_features as build_features_field  # noqa: E402
from train_test_xgboost_synthetic import build_features as build_features_synthetic  # noqa: E402

MODELS_DIR = ROOT_DIR / "Models"
RESULTS_DIR = ROOT_DIR / "Results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "compare_synthetic_field"

SYNTHETIC_HEIGHT = 0.1
FIELD_HEIGHT = 0.47


def predict(bundle, data_raw, build_features_fn):
    X = build_features_fn(data_raw)
    X_norm = bundle["scale_X"].transform(X)
    y_norm = np.column_stack([m.predict(X_norm) for m in bundle["models"]])
    return 10 ** bundle["scale_y"].inverse_transform(y_norm)


def compare_metrics(output_dir):
    synth_summary = pd.read_csv(RESULTS_DIR / "xgb_synthetic_comparison_new400_summary.csv")
    synth_row = synth_summary[synth_summary["n_layers"] == 9].iloc[0]
    with open(RESULTS_DIR / "xgb_field_metrics_new400_field.json") as f:
        field_metrics = json.load(f)

    rows = [
        {"config": "Synthetic (9 configs, 18 features)", "mae": synth_row["mae"],
         "mse": synth_row["mse"], "rmse": synth_row["rmse"], "r2": synth_row["r2"]},
        {"config": "Field (4 configs, 8 features)", "mae": field_metrics["mae"],
         "mse": field_metrics["mse"], "rmse": field_metrics["rmse"], "r2": field_metrics["r2"]},
    ]
    df = pd.DataFrame(rows)
    out_path = output_dir / "metrics_comparison.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"[table] -> {out_path}")

    per_layer_path = output_dir / "per_layer_rmse_comparison.csv"
    synth_per_layer = pd.read_csv(RESULTS_DIR / "xgb_synthetic_comparison_new400_per_layer.csv")
    synth_per_layer = synth_per_layer[synth_per_layer["n_layers"] == 9]
    per_layer_df = pd.DataFrame({
        "layer_index": synth_per_layer["layer_index"].to_numpy(),
        "synthetic_rmse": synth_per_layer["rmse"].to_numpy(),
        "field_rmse": field_metrics["per_layer_rmse"],
    })
    per_layer_df.to_csv(per_layer_path, index=False)
    print(f"[table] -> {per_layer_path}")


def plot_scenario_comparison(scenario_key, synth_bundle, field_bundle, output_dir, depthmax=10.0):
    cfg = SCENARIOS[scenario_key]
    transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
    positions = np.arange(len(transect))
    true_grid = regrid(transect[:, :2], transect[:, 2:], depthmax=depthmax)

    data_synth = np.array([FDEM(m[2:], m[:2], height=SYNTHETIC_HEIGHT) for m in transect])
    data_field = np.array([FDEM_field(m[2:], m[:2], height=FIELD_HEIGHT) for m in transect])

    pred_synth = predict(synth_bundle, data_synth, build_features_synthetic)
    pred_field = predict(field_bundle, data_field, build_features_field)

    grid_synth = regrid(np.tile(THK, (len(positions), 1)), pred_synth, depthmax=depthmax)
    grid_field = regrid(np.tile(THK, (len(positions), 1)), pred_field, depthmax=depthmax)

    err_synth = rel_pct_rmse(true_grid, grid_synth)
    err_field = rel_pct_rmse(true_grid, grid_field)

    fig, ax = plt.subplots(3, figsize=(9, 7), sharex=True, constrained_layout=True)
    vmin, vmax = 1, 1000
    extent = [positions[0], positions[-1], depthmax, 0]
    grids = [true_grid, grid_synth, grid_field]
    names = ["True model",
             f"XGBoost (synthetic config) -- {err_synth:.2f}% error",
             f"XGBoost (field config) -- {err_field:.2f}% error"]
    imgs = []
    for a, grid, name in zip(ax, grids, names):
        imgs.append(a.imshow(grid.T * 1000, extent=extent, aspect="auto",
                              norm="log", vmin=vmin, vmax=vmax, cmap="viridis"))
        a.set_title(name, fontsize=10)
        a.set_ylabel("Depth [m]")
    ax[-1].set_xlabel("Position [m]")
    fig.colorbar(imgs[0], ax=ax, label="$\\sigma$ [mS/m]", location="bottom", shrink=0.6)
    fig.suptitle(cfg["label"])

    out_path = output_dir / f"scenario_{scenario_key}_synthetic_vs_field.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure] {scenario_key}: synthetic {err_synth:.2f}% vs. field {err_field:.2f}% model error -> {out_path}")
    return err_synth, err_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-model-path", type=str, default=str(MODELS_DIR / "xgb_9layers_new400.joblib"))
    parser.add_argument("--field-model-path", type=str, default=str(MODELS_DIR / "xgb_9layers_new400_field.joblib"))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()), choices=list(SCENARIOS.keys()))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_metrics(output_dir)

    synth_bundle = joblib.load(args.synthetic_model_path)
    field_bundle = joblib.load(args.field_model_path)

    scenario_errors = {}
    for key in args.scenarios:
        scenario_errors[key] = plot_scenario_comparison(key, synth_bundle, field_bundle, output_dir)

    summary_path = output_dir / "scenario_model_error_pct.csv"
    pd.DataFrame([
        {"scenario": k, "synthetic_error_pct": v[0], "field_error_pct": v[1]}
        for k, v in scenario_errors.items()
    ]).to_csv(summary_path, index=False)
    print(f"[table] -> {summary_path}")


if __name__ == "__main__":
    main()
