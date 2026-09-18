"""Generate the field-configuration synthetic FDEM database (9 layers).

Field-instrument equivalent of create_database_synthetic.py: draws Latin Hypercube
conductivity models and forward models them with FDEM_field() (src/FDEM.py) instead
of the full synthetic FDEM(), matching the DUALEM instrument's actual channel set
(HCP/VCP at 4 m/8 m, no PRP, no 2 m in-phase -- see train_test_xgboost_field.py's
module docstring for the resulting 10-column raw layout).

This script did not previously exist in the repo -- Data/data_9layers_new400_field.npy
and Models/models_log_9layers_new400_field.npy were already present but nothing
regenerated them from scratch. Its defaults (400k extracted samples, 1-1000 mS/m,
seed 42) match the "new400" convention used everywhere else in this repo, so a fresh
run reproduces a statistically equivalent database (not bit-identical, since the
original generation run/seed weren't recorded).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from FDEM import FDEM_field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Data"

N_LAYERS = 9


def _forward_one(args):
    log10_sigma, thicknesses, height = args
    return FDEM_field(10 ** log10_sigma, thicknesses, height=height)


def lhs(n_samples, n_dims, rng):
    """Basic Latin Hypercube Sampling in [0, 1]. Returns array shape (n_samples, n_dims)."""
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, n_dims))
    pts = cut[:-1, None] + u * (1.0 / n_samples)

    lhs_samples = np.zeros_like(pts)
    for j in range(n_dims):
        lhs_samples[:, j] = pts[rng.permutation(n_samples), j]
    return lhs_samples


def generate_database(n_models, n_extract, sigma_min, sigma_max, height, random_seed, tag, n_jobs=1):
    log10_min = np.log10(sigma_min)
    log10_max = np.log10(sigma_max)
    log10_range = log10_max - log10_min

    rng = np.random.default_rng(random_seed)

    lhs_unit = lhs(n_models, N_LAYERS, rng)
    log10_sigma = log10_min + lhs_unit * log10_range
    models = log10_sigma[rng.choice(log10_sigma.shape[0], n_extract, replace=False)]

    thicknesses = np.logspace(-0.3, 0.35, N_LAYERS - 1)

    workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    if workers <= 1:
        data = []
        report_every = max(n_extract // 10, 1)
        for i, m in enumerate(models):
            data.append(FDEM_field(10 ** m, thicknesses, height=height))
            if (i + 1) % report_every == 0:
                print(f"  [field, {N_LAYERS} layers] {i + 1}/{n_extract} models forward modeled")
        data = np.array(data)
    else:
        tasks = [(m, thicknesses, height) for m in models]
        print(f"  [field, {N_LAYERS} layers] forward modeling {n_extract} models across {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            data = np.array(list(ex.map(_forward_one, tasks, chunksize=200)))
        print(f"  [field, {N_LAYERS} layers] {n_extract}/{n_extract} models forward modeled")

    models_path = MODELS_DIR / f"models_log_{N_LAYERS}layers_{tag}"
    data_path = DATA_DIR / f"data_{N_LAYERS}layers_{tag}"

    np.save(models_path, models)
    np.save(data_path, data)

    print(f"[field] saved {models.shape[0]} models -> {models_path}.npy")
    print(f"[field] saved {data.shape[0]} responses ({data.shape[1]} raw channels) -> {data_path}.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-models", type=int, default=1_000_000,
                         help="Number of LHS conductivity models to sample")
    parser.add_argument("--n-extract", type=int, default=400_000,
                         help="Number of models to randomly extract and forward model")
    parser.add_argument("--sigma-min", type=float, default=1e-3, help="Lower conductivity bound (S/m)")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="Upper conductivity bound (S/m)")
    parser.add_argument("--height", type=float, default=0.47,
                         help="Instrument height above ground (m) -- FDEM_field's own default")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default="new400_field", help="Suffix tag for output file names")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Worker processes for forward modeling (1=sequential, -1=all cores)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    generate_database(
        n_models=args.n_models, n_extract=args.n_extract,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        height=args.height, random_seed=args.seed, tag=args.tag, n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
