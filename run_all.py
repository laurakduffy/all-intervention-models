"""Run all intervention model scripts in sequence to refresh all data."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

scripts = [
    ROOT / "aw-models/data/inputs/aw_intervention_models.py",
    ROOT / "gw-models/gw_cea_modeling.py",
    ROOT / "leaf-models/leaf_cea_model.py",
    ROOT / "gcr-models/export_rp_csv.py",
    ROOT / "combine_data.py",
]

for script in scripts:
    print(f"\n{'='*60}\nRunning: {script.name}\n{'='*60}")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

print("\nAll models complete.")
