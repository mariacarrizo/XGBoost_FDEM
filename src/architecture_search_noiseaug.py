"""R1-3: reproduce "how we arrived at the right training-set size and the right
XGBoost architecture" (manuscript Figure 1b and Table 1).

Self-contained: generates any of the five required synthetic databases that are
missing (400k/800k/1.6M/3.2M/4M samples, 9 layers -- ~35 min total if none exist
yet, seconds if they do), then for each dataset size runs
train_test_xgboost_synthetic.py with no --architecture-json (so it performs its
own RandomizedSearchCV from scratch) and --noise-augment-max 0.10 (so both the
search and the final training see noise-augmented data, since a search that
only ever sees clean data can land on an architecture that is fragile under
real measurement noise -- see response_to_reviewers_tracking.md, R3-1).

Answers two questions at once:
  - "Is 400k samples enough?" -- by checking whether independently re-running
    this search at much larger sizes (up to 4M) changes the answer.
  - "Which architecture is right?" -- by checking whether the search converges
    to the same hyperparameters regardless of dataset size.

Saves each found architecture (Table 1) and evaluates all five resulting models
on one independent held-out set at 0/5/10% noise (Figure 1b).

For the companion "how we arrived at the right data size" fine-grained curve
(100k-800k in 100k steps, Figure 1a), see learning_curve_synthetic.py.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/architecture_search_noiseaug.py
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
RESULTS_DIR = ROOT_DIR.parent / "Results" / "architecture_search_noiseaug"
N_LAYERS = 9
NOISE_AUGMENT_MAX = 0.10
SUFFIX = "_noiseaug_tuned"

# 400k is the production dataset size (tag "new400"); the other four extend the
# check up to 4M to confirm 400k was already enough.
SIZES = [
    ("new400", 400_000),
    ("d10m800k", 800_000),
    ("d10m1600k", 1_600_000),
    ("d10m3200k", 3_200_000),
    ("d10m4000k", 4_000_000),
]

# An independent evaluation set, held out of every model's training, used only
# to score the five resulting models against each other on equal footing.
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
        print("[eval-data] already exists, skipping generation")
        return
    print(f"[eval-data] generating {EVAL_N:,} independent held-out samples...")
    generate_database(n_layers=N_LAYERS, n_models=100_000, n_extract=EVAL_N,
                       sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=EVAL_SEED,
                       tag=EVAL_TAG, n_jobs=-1)


def search_and_train(tag):
    model_path = MODELS_DIR / f"xgb_{N_LAYERS}layers_{tag}{SUFFIX}.joblib"
    arch_path = RESULTS_DIR / f"xgb_architecture_{tag}{SUFFIX}.json"
    if model_path.exists() and arch_path.exists():
        print(f"[search] {tag}: already done, skipping")
        return model_path, arch_path

    print(f"[search] {tag}: tuning + training noise-augmented architecture from scratch...")
    t0 = time.time()
    cmd = [
        sys.executable, str(ROOT_DIR / "train_test_xgboost_synthetic.py"),
        "--layers", str(N_LAYERS),
        "--tag", tag,
        "--noise-augment-max", str(NOISE_AUGMENT_MAX),
        "--model-suffix", SUFFIX,
        "--output-dir", str(RESULTS_DIR),
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"search_{tag}{SUFFIX}.log"
    with open(log_path, "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"[search] {tag}: done in {time.time() - t0:.0f}s -> {model_path}")
    return model_path, arch_path


def predict_log10(bundle, data_raw):
    X = build_features(data_raw)
    X_scaled = bundle["scale_X"].transform(X)
    y_scaled = np.column_stack([m.predict(X_scaled) for m in bundle["models"]])
    return bundle["scale_y"].inverse_transform(y_scaled)


def evaluate_on_holdout():
    models_log = np.load(MODELS_DIR / f"models_log_{N_LAYERS}layers_{EVAL_TAG}.npy")
    data_raw = np.load(DATA_DIR / f"data_{N_LAYERS}layers_{EVAL_TAG}.npy")

    rows = []
    for tag, n_extract in SIZES:
        model_path = MODELS_DIR / f"xgb_{N_LAYERS}layers_{tag}{SUFFIX}.joblib"
        bundle = joblib.load(model_path)
        for noise_level in EVAL_NOISE_LEVELS:
            data_noisy = add_noise(data_raw, noise_level, seed=12345)
            pred_log10 = predict_log10(bundle, data_noisy)
            rmse = float(np.sqrt(np.mean((pred_log10 - models_log) ** 2)))
            rows.append({"tag": tag, "n_train_samples": n_extract,
                          "noise_level": noise_level, "rmse_log10_sigma": rmse})
            print(f"[eval] {tag} @ {noise_level:.0%} noise: RMSE(log10 sigma) = {rmse:.4f}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "architecture_search_noiseaug_summary.csv"
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
    ax.set_title("Per-size tuned noise-augmented architecture: robustness vs. dataset size")
    ax.legend()
    fig.tight_layout()
    plot_path = RESULTS_DIR / "architecture_search_noiseaug_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[eval] plot -> {plot_path}")


def compare_architectures():
    rows = []
    for tag, n_extract in SIZES:
        arch_path = RESULTS_DIR / f"xgb_architecture_{tag}{SUFFIX}.json"
        with open(arch_path) as f:
            arch = json.load(f)
        rows.append({"tag": tag, "n_train_samples": n_extract, **arch})
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "architectures_found_by_size.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[architectures] -> {csv_path}")
    print(df.to_string(index=False))
    return df


def main():
    for tag, n_extract in SIZES:
        ensure_database(tag, n_extract)
    ensure_eval_set()

    for tag, _ in SIZES:
        search_and_train(tag)
    compare_architectures()
    evaluate_on_holdout()


if __name__ == "__main__":
    main()
