"""Generate synthetic FDEM conductivity/response databases for several layer counts.

Generalizes notebooks/CreateDatabaseSynthetic.ipynb: for each requested number of
layers, draws Latin Hypercube conductivity models, forward models their FDEM
response, and saves both arrays under Models/ and Data/.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from FDEM import FDEM

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Data"


def _forward_one(args):
    log10_sigma, thicknesses, height = args
    return FDEM(10 ** log10_sigma, thicknesses, height=height)


def lhs(n_samples, n_dims, rng):
    """Basic Latin Hypercube Sampling in [0, 1]. Returns array shape (n_samples, n_dims)."""
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, n_dims))
    pts = cut[:-1, None] + u * (1.0 / n_samples)

    lhs_samples = np.zeros_like(pts)
    for j in range(n_dims):
        lhs_samples[:, j] = pts[rng.permutation(n_samples), j]
    return lhs_samples


def generate_database(n_layers, n_models, n_extract, sigma_min, sigma_max,
                       height, random_seed, tag, total_depth=None, n_jobs=1):
    """Build and save a synthetic database for a given number of layers."""
    log10_min = np.log10(sigma_min)
    log10_max = np.log10(sigma_max)
    log10_range = log10_max - log10_min

    rng = np.random.default_rng(random_seed)

    # Latin Hypercube sample of log10(sigma) for every layer
    lhs_unit = lhs(n_models, n_layers, rng)
    log10_sigma = log10_min + lhs_unit * log10_range

    # Extract a random subset of models to forward model
    models = log10_sigma[rng.choice(log10_sigma.shape[0], n_extract, replace=False)]

    # Layer thicknesses (n_layers - 1 values), log-increasing with depth.
    thicknesses = np.logspace(-0.3, 0.35, n_layers - 1)
    if total_depth is not None:
        # Rescale to a fixed total profiled depth, so n_layers changes only how
        # finely that depth is sliced, not how deep the profile reaches.
        thicknesses = thicknesses / thicknesses.sum() * total_depth

    workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    if workers <= 1:
        data = []
        report_every = max(n_extract // 10, 1)
        for i, m in enumerate(models):
            data.append(FDEM(10 ** m, thicknesses, height=height))
            if (i + 1) % report_every == 0:
                print(f"  [{n_layers} layers] {i + 1}/{n_extract} models forward modeled")
        data = np.array(data)
    else:
        tasks = [(m, thicknesses, height) for m in models]
        print(f"  [{n_layers} layers] forward modeling {n_extract} models across {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            data = np.array(list(ex.map(_forward_one, tasks, chunksize=200)))
        print(f"  [{n_layers} layers] {n_extract}/{n_extract} models forward modeled")

    models_path = MODELS_DIR / f"models_log_{n_layers}layers_{tag}"
    data_path = DATA_DIR / f"data_{n_layers}layers_{tag}"

    np.save(models_path, models)
    np.save(data_path, data)

    print(f"[{n_layers} layers] saved {models.shape[0]} models -> {models_path}.npy")
    print(f"[{n_layers} layers] saved {data.shape[0]} responses -> {data_path}.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 11, 12],
                         help="Numbers of layers to generate databases for")
    parser.add_argument("--n-models", type=int, default=1_000_000,
                         help="Number of LHS conductivity models to sample")
    parser.add_argument("--n-extract", type=int, default=400_000,
                         help="Number of models to randomly extract and forward model")
    parser.add_argument("--sigma-min", type=float, default=1e-3, help="Lower conductivity bound (S/m)")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="Upper conductivity bound (S/m)")
    parser.add_argument("--height", type=float, default=0.1, help="Instrument height above ground (m)")
    parser.add_argument("--total-depth", type=float, default=None,
                         help="If set, rescale layer thicknesses so every --layers value profiles this same "
                              "total depth (m), isolating layer-count/resolution from depth-of-exploration. "
                              "If omitted, keeps the original behavior where total depth grows with n_layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default="new400", help="Suffix tag for output file names")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Worker processes for forward modeling (1=sequential, -1=all cores)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    for n_layers in args.layers:
        print(f"Generating database for {n_layers} layers...")
        generate_database(
            n_layers=n_layers,
            n_models=args.n_models,
            n_extract=args.n_extract,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            height=args.height,
            random_seed=args.seed,
            tag=args.tag,
            total_depth=args.total_depth,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
