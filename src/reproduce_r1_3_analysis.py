"""Single entry point to reproduce the R1-3 analysis end to end: "how we arrived
at 400,000 training samples and this XGBoost architecture" (manuscript Figure 1
and Table 1; see response_to_reviewers_tracking.md, R1-3 and R3-1).

Runs, in order:
  1. learning_curve_synthetic.py    -- fine-grained curve, 100k-800k samples in
     100k steps, for the 9-layer model (Figure 1a). Generates the underlying
     800k-sample database itself if it doesn't already exist.
  2. architecture_search_noiseaug.py -- an independent hyperparameter search
     repeated at five dataset sizes (400k, 800k, 1.6M, 3.2M, 4M), training on
     noise-augmented data, to check whether more data changes which
     architecture is optimal (it doesn't) (Figure 1b, Table 1). Generates any
     of the five databases that don't already exist.
  3. make_manuscript_figures_r1.py  -- assembles Figure 1 and Table 1 (this
     script also rebuilds the R1-1/R1-2 figures/tables from their own existing
     results; rerun gn_baseline_synthetic.py / stability_across_training_data.py
     first if you want those refreshed too).

From a clean checkout (no databases generated yet) this takes roughly 2-3
hours, dominated by generating the 3.2M/4M-sample databases and the five
hyperparameter searches. If Models/ and Data/ already contain the databases
(e.g. from a previous run), it takes only as long as retraining -- a few
minutes to ~30 minutes depending on what's missing.

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/reproduce_r1_3_analysis.py
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(ROOT_DIR / script), *args]
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    subprocess.run(cmd, check=True)


def main():
    run("learning_curve_synthetic.py", "--layers", "9", "--tag", "d10m800k")
    run("architecture_search_noiseaug.py")
    run("make_manuscript_figures_r1.py")
    print("\nDone. Figure 1 -> MachineLearningFDEM/figures/LearningCurve.png; "
          "Table 1 -> MachineLearningFDEM/tables/architecture_convergence.tex")


if __name__ == "__main__":
    main()
