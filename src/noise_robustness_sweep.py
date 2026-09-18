"""Noise-robustness vs. dataset-size sweep for the 9-layer XGBoost model.

Follows up on the finding that the 4M-sample model (Models/xgb_9layers_d10m4000k.joblib)
becomes erratic under measurement noise even though it was the best model noise-free.
Trains noise-augmented 9-layer models (every training sample gets its own random
noise level in [0, 10%], see train_test_xgboost_synthetic.add_noise_range) at four
dataset sizes -- 400k, 800k, 1.6M, 3.2M -- all sharing the same architecture (the one
found for the 4M model), so dataset size is the only thing varying. Then evaluates
all four on a single independent held-out set at 0%/5%/10% noise, to see whether more
data alone fixes noise robustness.

Run inside the `xgb-fdem` conda env (has pygimli + empymod + xgboost):
    conda run -n xgb-fdem python src/noise_robustness_sweep.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from create_database_synthetic import generate_database  # noqa: E402
from train_test_xgboost_synthetic import add_noise, build_features  # noqa: E402

MODELS_DIR = ROOT_DIR.parent / "Models"
DATA_DIR = ROOT_DIR.parent / "Data"
RESULTS_DIR = ROOT_DIR.parent / "Results" / "noise_robustness_sweep"
ARCH_PATH = ROOT_DIR.parent / "Results" / "xgb_architecture_9layers_d10m4000k.json"
N_LAYERS = 9

# (tag, n_extract) -- 400k and 800k already exist under these tags; 1.6M/3.2M are new.
SIZES = [
    ("new400", 400_000),
    ("d10m800k", 800_000),
    ("d10m1600k", 1_600_000),
    ("d10m3200k", 3_200_000),
]
NOISE_AUGMENT_MAX = 0.10
EVAL_TAG = "noiseaug_eval_holdout"
EVAL_N = 20_000
EVAL_SEED = 12345
EVAL_NOISE_LEVELS = [0.0, 0.05, 0.10]


def ensure_database(tag, n_extract):
    models_path = MODELS_DIR / f"models_log_{N_LAYERS}layers_{tag}.npy"
    data_path = DATA_DIR / f"data_{N_LAYERS}layers_{tag}.npy"
    if models_path.exists() and data_path.exists():
        print(f"[data] {tag}: already exists, skipping generation")
        return
    print(f"[data] {tag}: generating {n_extract:,} samples...")
    t0 = time.time()
    generate_database(n_layers=N_LAYERS, n_models=max(n_extract, 100_000), n_extract=n_extract,
                       sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=42, tag=tag,
                       n_jobs=-1)
    print(f"[data] {tag}: done in {time.time() - t0:.0f}s")


def ensure_eval_set():
    models_path = MODELS_DIR / f"models_log_{N_LAYERS}layers_{EVAL_TAG}.npy"
    data_path = DATA_DIR / f"data_{N_LAYERS}layers_{EVAL_TAG}.npy"
    if models_path.exists() and data_path.exists():
        print(f"[eval-data] already exists, skipping generation")
        return
    print(f"[eval-data] generating {EVAL_N:,} independent held-out samples...")
    generate_database(n_layers=N_LAYERS, n_models=100_000, n_extract=EVAL_N,
                       sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=EVAL_SEED,
                       tag=EVAL_TAG, n_jobs=-1)


def train_noiseaug(tag):
    model_path = MODELS_DIR / f"xgb_{N_LAYERS}layers_{tag}_noiseaug.joblib"
    if model_path.exists():
        print(f"[train] {tag}: model already exists at {model_path}, skipping")
        return model_path
    print(f"[train] {tag}: training noise-augmented model...")
    t0 = time.time()
    cmd = [
        sys.executable, str(ROOT_DIR / "train_test_xgboost_synthetic.py"),
        "--layers", str(N_LAYERS),
        "--tag", tag,
        "--architecture-json", str(ARCH_PATH),
        "--noise-augment-max", str(NOISE_AUGMENT_MAX),
        "--model-suffix", "_noiseaug",
        "--output-dir", str(RESULTS_DIR),
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"train_{tag}_noiseaug.log"
    with open(log_path, "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"[train] {tag}: done in {time.time() - t0:.0f}s -> {model_path}")
    return model_path


def predict(bundle, data_raw):
    X = build_features(data_raw)
    X_scaled = bundle["scale_X"].transform(X)
    y_scaled = np.column_stack([m.predict(X_scaled) for m in bundle["models"]])
    y_log10 = bundle["scale_y"].inverse_transform(y_scaled)
    return y_log10  # log10(sigma), same units as models_log arrays


def evaluate_on_holdout():
    models_log = np.load(MODELS_DIR / f"models_log_{N_LAYERS}layers_{EVAL_TAG}.npy")
    data_raw = np.load(DATA_DIR / f"data_{N_LAYERS}layers_{EVAL_TAG}.npy")

    rows = []
    for tag, n_extract in SIZES:
        model_path = MODELS_DIR / f"xgb_{N_LAYERS}layers_{tag}_noiseaug.joblib"
        bundle = joblib.load(model_path)
        for noise_level in EVAL_NOISE_LEVELS:
            data_noisy = add_noise(data_raw, noise_level, seed=EVAL_SEED)
            pred_log10 = predict(bundle, data_noisy)
            rmse = float(np.sqrt(np.mean((pred_log10 - models_log) ** 2)))
            rows.append({
                "tag": tag,
                "n_train_samples": n_extract,
                "noise_level": noise_level,
                "rmse_log10_sigma": rmse,
            })
            print(f"[eval] {tag} @ {noise_level:.0%} noise: RMSE(log10 sigma) = {rmse:.4f}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "noise_robustness_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[eval] summary -> {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for noise_level in EVAL_NOISE_LEVELS:
        sub = df[df["noise_level"] == noise_level].sort_values("n_train_samples")
        ax.plot(sub["n_train_samples"], sub["rmse_log10_sigma"], "o-",
                label=f"{noise_level:.0%} noise")
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Held-out RMSE (log10 sigma)")
    ax.set_title("Noise-augmented 9-layer model: robustness vs. dataset size")
    ax.legend()
    fig.tight_layout()
    plot_path = RESULTS_DIR / "noise_robustness_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[eval] plot -> {plot_path}")
    return df


def main():
    for tag, n_extract in SIZES:
        ensure_database(tag, n_extract)
    ensure_eval_set()

    with open(ARCH_PATH) as f:
        print(f"[train] shared architecture (from 4M search): {json.load(f)}")

    for tag, _ in SIZES:
        train_noiseaug(tag)

    evaluate_on_holdout()


if __name__ == "__main__":
    main()
