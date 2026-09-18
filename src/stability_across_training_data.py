"""R1-2: how stable are XGBoost predictions under different training data?

Compares two 9-layer XGBoost models that share the identical architecture and
training procedure, but were trained on two independently-drawn 400,000-sample
databases (different random seeds -- new400 vs new400_seedB). If predictions
agree closely on data neither model has seen, that is direct evidence that the
non-uniqueness introduced by the choice of training data is small in practice.

Compares on:
  1. An independent 20k-sample held-out set (not used to train either model).
  2. The MC/MR synthetic transects already used elsewhere in the manuscript.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/stability_across_training_data.py
"""

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

from FDEM import FDEM  # noqa: E402
from gn_baseline_synthetic import SCENARIOS, build_transect, forward_transect, THK  # noqa: E402
from train_test_xgboost_synthetic import build_features  # noqa: E402

MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Data"
RESULTS_DIR = ROOT_DIR / "Results" / "stability_across_training_data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT_DIR / "MachineLearningFDEM" / "figures"

EVAL_TAG = "noiseaug_eval_holdout"  # independent 20k-sample set from an earlier experiment
N_LAYERS = 9

COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"


def predict_log10(bundle, data_raw):
    X = build_features(data_raw)
    X_scaled = bundle["scale_X"].transform(X)
    y_scaled = np.column_stack([m.predict(X_scaled) for m in bundle["models"]])
    return bundle["scale_y"].inverse_transform(y_scaled)


def predict_sigma(bundle, data_raw):
    return 10 ** predict_log10(bundle, data_raw)


def compare_holdout(bundle_a, bundle_b):
    models_log = np.load(MODELS_DIR / f"models_log_{N_LAYERS}layers_{EVAL_TAG}.npy")
    data_raw = np.load(DATA_DIR / f"data_{N_LAYERS}layers_{EVAL_TAG}.npy")

    pred_a = predict_log10(bundle_a, data_raw)
    pred_b = predict_log10(bundle_b, data_raw)

    rmse_a_true = float(np.sqrt(np.mean((pred_a - models_log) ** 2)))
    rmse_b_true = float(np.sqrt(np.mean((pred_b - models_log) ** 2)))
    rmse_a_b = float(np.sqrt(np.mean((pred_a - pred_b) ** 2)))
    abs_pct_diff = np.abs(10 ** pred_a - 10 ** pred_b) / (10 ** models_log) * 100
    mean_abs_diff_pct = float(np.mean(abs_pct_diff))
    median_abs_diff_pct = float(np.median(abs_pct_diff))
    disagreement_fraction = rmse_a_b / ((rmse_a_true + rmse_b_true) / 2)

    print(f"[holdout] RMSE(model A, truth)  = {rmse_a_true:.4f} (log10 sigma)")
    print(f"[holdout] RMSE(model B, truth)  = {rmse_b_true:.4f} (log10 sigma)")
    print(f"[holdout] RMSE(model A, model B) = {rmse_a_b:.4f} (log10 sigma) -- disagreement between the two models")
    print(f"[holdout] mean |A-B|/true sigma  = {mean_abs_diff_pct:.2f} % (median {median_abs_diff_pct:.2f} %)")
    print(f"[holdout] disagreement fraction  = {disagreement_fraction:.2f} "
          f"(model-to-model RMSE relative to each model's own RMSE against truth)")

    return {
        "rmse_modelA_vs_truth": rmse_a_true,
        "rmse_modelB_vs_truth": rmse_b_true,
        "rmse_modelA_vs_modelB": rmse_a_b,
        "mean_abs_pct_diff_A_vs_B": mean_abs_diff_pct,
        "median_abs_pct_diff_A_vs_B": median_abs_diff_pct,
        "disagreement_fraction": disagreement_fraction,
    }


def true_sigma_on_grid(h1, h2, sig1, sig2, sig3, thk):
    """True 3-layer transect model (thicknesses h1/h2, conductivities sig1/sig2/sig3,
    sig3 extending to infinity), resampled onto the 9-layer prediction grid `thk` as a
    depth-weighted average over each grid layer's true extent.

    A grid layer that straddles a true interface (the 9-layer grid boundaries don't
    generally line up with h1/h1+h2) is given the overlap-weighted blend of the true
    conductivities it spans, not just whichever zone contains its top edge -- the
    latter over- or under-represents whichever zone happens to sit at the layer's
    very top, and (since roughly half of the anomaly's depth range can fall in a
    single grid layer here) can make the true profile look shifted relative to
    where the model's predictions actually land.
    """
    true_bounds = np.array([0.0, h1, h1 + h2, np.inf])
    true_sigmas = np.array([sig1, sig2, sig3])
    edges = np.hstack(([0.0], np.cumsum(thk), [np.inf]))

    out = np.zeros(len(thk) + 1)
    for i in range(len(out)):
        top, bottom = edges[i], edges[i + 1]
        if np.isinf(bottom):
            zone = np.clip(np.searchsorted(true_bounds, top, side="right") - 1, 0, 2)
            out[i] = true_sigmas[zone]
            continue
        overlap_weighted = 0.0
        for z0, z1, sig in zip(true_bounds[:-1], true_bounds[1:], true_sigmas):
            overlap = max(0.0, min(bottom, z1) - max(top, z0))
            overlap_weighted += overlap * sig
        out[i] = overlap_weighted / (bottom - top)
    return out


def compare_transects_combined_figure(bundle_a, bundle_b):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, cfg) in zip(axes, SCENARIOS.items()):
        transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
        data_true = forward_transect(transect)
        pred_a = predict_sigma(bundle_a, data_true)
        pred_b = predict_sigma(bundle_b, data_true)
        pos_mid = len(transect) // 2
        h1, h2, sig1, sig2, sig3 = transect[pos_mid]
        true_sigma = true_sigma_on_grid(h1, h2, sig1, sig2, sig3, THK)
        layer_idx = np.arange(1, N_LAYERS + 1)
        ax.plot(layer_idx, true_sigma * 1000, "^-", color="black", label="True")
        ax.plot(layer_idx, pred_a[pos_mid] * 1000, "o-", color=COLOR_A, label="Model A")
        ax.plot(layer_idx, pred_b[pos_mid] * 1000, "s--", color=COLOR_B, label="Model B")
        ax.set_yscale("log")
        ax.set_xlabel("Layer")
        ax.set_title(cfg["label"], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("$\\sigma$ [mS/m]")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Predictions at the transect midpoint, two independently-trained models")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stability_combined.eps")
    fig.savefig(RESULTS_DIR / "stability_combined.png", dpi=200)
    fig.savefig(FIGURES_DIR / "StabilityAcrossTrainingData.eps")
    fig.savefig(FIGURES_DIR / "StabilityAcrossTrainingData.png", dpi=200)
    plt.close(fig)


def compare_transects(bundle_a, bundle_b):
    rows = []
    for key, cfg in SCENARIOS.items():
        transect = build_transect(cfg["sig1"], cfg["sig2"], cfg["sig3"])
        data_true = forward_transect(transect)

        pred_a = predict_sigma(bundle_a, data_true)
        pred_b = predict_sigma(bundle_b, data_true)

        abs_pct_diff = np.abs(pred_a - pred_b) / pred_a * 100
        mean_abs_diff_pct = float(np.mean(abs_pct_diff))
        median_abs_diff_pct = float(np.median(abs_pct_diff))
        rows.append({
            "scenario": key,
            "mean_abs_pct_diff_A_vs_B": mean_abs_diff_pct,
            "median_abs_pct_diff_A_vs_B": median_abs_diff_pct,
        })
        print(f"[{key}] mean |A-B|/A = {mean_abs_diff_pct:.2f} % (median {median_abs_diff_pct:.2f} %)")

        fig, ax = plt.subplots(figsize=(7, 4))
        positions = np.arange(len(transect))
        pos_mid = len(positions) // 2
        h1, h2, sig1, sig2, sig3 = transect[pos_mid]
        true_sigma = true_sigma_on_grid(h1, h2, sig1, sig2, sig3, THK)
        layer_idx = np.arange(1, N_LAYERS + 1)
        ax.plot(layer_idx, true_sigma * 1000, "^-", color="black", label="True")
        ax.plot(layer_idx, pred_a[pos_mid] * 1000, "o-", color=COLOR_A, label="Model A")
        ax.plot(layer_idx, pred_b[pos_mid] * 1000, "s--", color=COLOR_B, label="Model B")
        ax.set_yscale("log")
        ax.set_xlabel("Layer")
        ax.set_ylabel("$\\sigma$ [mS/m]")
        ax.set_title(f"{cfg['label']} -- predictions at position {pos_mid}, two independently-trained models")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"stability_{key}.png", dpi=150)
        plt.close(fig)

    return pd.DataFrame(rows)


def main():
    bundle_a = joblib.load(MODELS_DIR / "xgb_9layers_new400.joblib")
    bundle_b = joblib.load(MODELS_DIR / "xgb_9layers_new400_seedB.joblib")

    print("=== Held-out set comparison ===")
    holdout_metrics = compare_holdout(bundle_a, bundle_b)

    print("\n=== MC/MR transect comparison ===")
    transect_df = compare_transects(bundle_a, bundle_b)
    compare_transects_combined_figure(bundle_a, bundle_b)

    summary = {**holdout_metrics}
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "holdout_stability_summary.csv", index=False)
    transect_df.to_csv(RESULTS_DIR / "transect_stability_summary.csv", index=False)
    print(f"\n[summary] -> {RESULTS_DIR / 'holdout_stability_summary.csv'}")
    print(f"[summary] -> {RESULTS_DIR / 'transect_stability_summary.csv'}")


if __name__ == "__main__":
    main()
