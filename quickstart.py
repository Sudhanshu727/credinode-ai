"""
CrediNode AI — Quick Start (One Command Setup)
================================================
Runs all setup steps in sequence:
  1. Download datasets
  2. Generate synthetic merchant data
  3. Train Gate 1 (Isolation Forest)
  4. Train Gate 2A (BSI calibrator)
  5. Train Gate 2B (GNN / fallback)
  6. Train Gate 3 (XGBoost + LightGBM ensemble)
  7. Run demo inference

Usage:
  python quickstart.py             # Full pipeline
  python quickstart.py --data-only # Only data steps
  python quickstart.py --api       # Start API after training

Expected time: ~5-10 minutes on a modern laptop (CPU only).
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

BASE = Path(__file__).parent

STEPS = [
    ("01_download_data.py",       "Download & generate datasets"),
    ("02_generate_synthetic.py",  "Generate India merchant profiles"),
    ("03_train_gate1_anomaly.py", "Train Gate 1: Isolation Forest"),
    ("04_train_gate2a_bsi.py",    "Train Gate 2A: BSI Calibrator"),
    ("05_train_gate2b_gnn.py",    "Train Gate 2B: GNN"),
    ("06_train_gate3_ensemble.py","Train Gate 3: XGBoost + LightGBM"),
    ("07_run_pipeline.py",        "Demo: Full Inference Pipeline"),
]


def run_step(script: str, desc: str) -> bool:
    print(f"\n{'═'*60}")
    print(f"  ▶  {desc}")
    print(f"{'═'*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(BASE / "scripts" / script)],
        cwd=str(BASE)
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✅  Done in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌  Failed (exit code {result.returncode})")
        return False


def check_requirements():
    print("Checking requirements...")
    missing = []
    pkgs = ["sklearn", "xgboost", "lightgbm", "shap", "fastapi", "uvicorn",
            "pandas", "numpy", "networkx", "imblearn", "plotly"]
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  ⚠  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    print("  ✓  All required packages available")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrediNode AI Quick Start")
    parser.add_argument("--data-only", action="store_true", help="Only run data steps")
    parser.add_argument("--train-only", action="store_true", help="Only training steps (skip data)")
    parser.add_argument("--api", action="store_true", help="Start API server after training")
    parser.add_argument("--skip-demo", action="store_true", help="Skip the demo inference step")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  🔮  CrediNode AI — Full Setup")
    print("      FIN-O-HACK | Paytm × ASSETS DTU")
    print("═"*60)

    if not check_requirements():
        sys.exit(1)

    if args.data_only:
        steps_to_run = STEPS[:2]
    elif args.train_only:
        steps_to_run = STEPS[2:]
    else:
        steps_to_run = STEPS

    if args.skip_demo:
        steps_to_run = [s for s in steps_to_run if "07_" not in s[0]]

    failed = []
    total_t = time.time()

    for script, desc in steps_to_run:
        ok = run_step(script, desc)
        if not ok:
            failed.append(script)

    total_elapsed = time.time() - total_t

    print(f"\n\n{'═'*60}")
    print(f"  SUMMARY — {len(steps_to_run) - len(failed)}/{len(steps_to_run)} steps passed")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("═"*60)

    if failed:
        print(f"\n  ⚠  Failed steps: {', '.join(failed)}")
        print("  Check error messages above for details.")
    else:
        print("""
  ✅  All done! Your CrediNode AI system is ready.

  NEXT STEPS:
  ─────────────────────────────────────────────────
  1. Open the dashboard:
     → Open dashboard/index.html in your browser

  2. Start the API:
     → uvicorn api.main:app --reload --port 8000
     → API docs at: http://localhost:8000/docs
     → Demo scores: http://localhost:8000/demo

  3. Explore the notebook:
     → jupyter notebook notebooks/eda_and_evaluation.ipynb

  4. Test the pipeline:
     → python scripts/07_run_pipeline.py
  ─────────────────────────────────────────────────
        """)

    if args.api and not failed:
        print("\n  Starting API server...")
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--reload", "--port", "8000"],
            cwd=str(BASE)
        )
