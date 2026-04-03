"""Run GCR + combine_data with 5 different seeds, saving each result to outputs/seeds/seed_{N}/."""
import argparse
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SEEDS = [42, 43, 44, 45, 46]

_parser = argparse.ArgumentParser(description="Run seed sensitivity analysis for GCR models.")
_parser.add_argument(
    "--n-samples", type=int, default=1000000,
    help="Monte Carlo samples per fund per seed (default: 1,000,000).",
)
_parser.add_argument(
    "--n-batches", type=int, default=10,
    help="Batches to run for GCR simulations (default=10).",
)
_parser.add_argument(
    "--gcr-dmr-scenario", default="median",
    choices=["optimistic", "pessimistic", "median", "fund_estimated"],
    help="GCR diminishing returns scenario passed to combine_data.py (default: median)",
)
_parser.add_argument(
    "--seeds", type=int, nargs="+", default=SEEDS,
    help=f"Seeds to run (default: {SEEDS})",
)
_parser.add_argument(
    "--quiet", action="store_true",
    help="Suppress per-fund progress output.",
)
_args = _parser.parse_args()

seeds_root = ROOT / "outputs" / "seeds"
seeds_root.mkdir(parents=True, exist_ok=True)

gcr_output   = ROOT / "gcr-models" / "gcr_output.csv"
gcr_stats    = ROOT / "gcr-models" / "gcr_output_summary_stats.csv"
gcr_abs_ev   = ROOT / "gcr-models" / "gcr_output_absolute_ev_percentiles.csv"
gcr_hists    = ROOT / "gcr-models" / "histograms"

for seed in _args.seeds:
    seed_dir = seeds_root / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"SEED {seed}")
    print(f"{'=' * 70}")

    # --- GCR model ---
    gcr_cmd = [
        sys.executable, str(ROOT / "gcr-models" / "export_rp_csv.py"),
        "--seed", str(seed),
        "--n-samples", str(_args.n_samples),
        "--n-batches", str(_args.n_batches)
    ]
    if _args.quiet:
        gcr_cmd.append("--quiet")
    subprocess.run(gcr_cmd, cwd=ROOT, check=True)

    # --- combine_data ---
    subprocess.run(
        [sys.executable, str(ROOT / "combine_data.py"),
         "--gcr-dmr-scenario", _args.gcr_dmr_scenario],
        cwd=ROOT, check=True,
    )

    # --- Copy all seed-relevant outputs to outputs/seeds/seed_{N}/ ---
    # GCR artefacts
    for src in [gcr_output, gcr_stats, gcr_abs_ev]:
        if src.exists():
            shutil.copy2(src, seed_dir / src.name)

    if gcr_hists.exists():
        dest_hists = seed_dir / "gcr_histograms"
        if dest_hists.exists():
            def _force_remove(func, path, exc):
                import os
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(dest_hists, onexc=_force_remove)
        shutil.copytree(gcr_hists, dest_hists)

    # combine_data outputs
    for f in (ROOT / "outputs").glob("*.csv"):
        shutil.copy2(f, seed_dir / f.name)
    for f in (ROOT / "outputs").glob("*.json"):
        shutil.copy2(f, seed_dir / f.name)

    print(f"  Saved to {seed_dir.relative_to(ROOT)}")

print(f"\n{'=' * 70}")
print(f"Seed sweep complete. {len(_args.seeds)} seeds saved under outputs/seeds/")
print(f"{'=' * 70}")
