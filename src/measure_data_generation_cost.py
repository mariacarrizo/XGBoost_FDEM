"""Measure the computational cost (wall time, peak memory) of generating the
synthetic FDEM training database, for reviewer comment R2-8.

Measures:
  1. Parallel generation (all 16 logical cores) of the actual production size,
     400,000 9-layer samples -- the practical, as-used cost.
  2. Single-core generation of a small batch, to get a clean per-sample time
     and extrapolate the single-core cost of 400,000 and 800,000 samples
     (single-core run to completion would take too long to run directly).

Peak memory is the sum of the resident set size (RSS) of the main process and
all its worker child processes, sampled every 0.5 s while generation runs.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/measure_data_generation_cost.py
"""

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from create_database_synthetic import generate_database  # noqa: E402

MODELS_DIR = ROOT_DIR.parent / "Models"
DATA_DIR = ROOT_DIR.parent / "Data"
RESULTS_DIR = ROOT_DIR.parent / "Results"
N_LAYERS = 9


class MemoryMonitor:
    """Polls RSS of this process + all children, tracks the peak total."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        total = self._proc.memory_info().rss
        try:
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._sample())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_bytes = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def cleanup(tag):
    for d in (MODELS_DIR, DATA_DIR):
        for f in d.glob(f"*{tag}*"):
            f.unlink()


def measure_parallel_400k():
    tag = "_r2_8_timing_400k"
    t0 = time.time()
    with MemoryMonitor() as mon:
        generate_database(n_layers=N_LAYERS, n_models=400_000, n_extract=400_000,
                           sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=1,
                           tag=tag, n_jobs=-1)
    elapsed = time.time() - t0
    cleanup(tag)
    return {
        "config": "Parallel (16 logical cores)",
        "n_samples": 400_000,
        "wall_time_s": elapsed,
        "samples_per_s": 400_000 / elapsed,
        "peak_memory_gb": mon.peak_bytes / 1e9,
        "measured": True,
    }


def measure_single_core_small(n_extract=8_000):
    tag = "_r2_8_timing_1core"
    t0 = time.time()
    with MemoryMonitor() as mon:
        generate_database(n_layers=N_LAYERS, n_models=max(n_extract, 100_000), n_extract=n_extract,
                           sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=2,
                           tag=tag, n_jobs=1)
    elapsed = time.time() - t0
    cleanup(tag)
    return {
        "config": "Single core (measured)",
        "n_samples": n_extract,
        "wall_time_s": elapsed,
        "samples_per_s": n_extract / elapsed,
        "peak_memory_gb": mon.peak_bytes / 1e9,
        "measured": True,
    }


def main():
    rows = []
    print("[measure] parallel, 400,000 samples, 16 cores...")
    rows.append(measure_parallel_400k())
    print(f"  -> {rows[-1]}")

    print("[measure] single core, small batch (for extrapolation)...")
    rows.append(measure_single_core_small())
    print(f"  -> {rows[-1]}")

    single_rate = rows[-1]["samples_per_s"]
    for n in (400_000, 800_000):
        rows.append({
            "config": "Single core (extrapolated)",
            "n_samples": n,
            "wall_time_s": n / single_rate,
            "samples_per_s": single_rate,
            # Memory scales with n_extract (final array size), not core count, so the
            # small single-core batch's peak isn't representative at full size -- left
            # blank here; the measured 400k parallel run's peak is the reference figure.
            "peak_memory_gb": float("nan"),
            "measured": False,
        })

    df = pd.DataFrame(rows)
    df["wall_time_min"] = df["wall_time_s"] / 60
    out_dir = RESULTS_DIR / "r2_8_data_generation_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "data_generation_cost.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[table] -> {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
