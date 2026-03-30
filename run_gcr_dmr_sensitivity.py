"""Run combine_data.py for all four GCR diminishing returns scenarios.

Each run writes its own output files to outputs/:
  outputs/output_data_{scenario}.json
  outputs/all_risk_adjusted_{scenario}.csv
  outputs/all_diminishing_returns_{scenario}.csv
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

SCENARIOS = ['optimistic', 'pessimistic', 'median', 'fund_estimated']

for scenario in SCENARIOS:
    print(f"\n{'='*60}\nScenario: {scenario}\n{'='*60}")
    subprocess.run(
        [sys.executable, str(ROOT / 'combine_data.py'), '--gcr-dmr-scenario', scenario],
        cwd=ROOT, check=True
    )

print("\nAll GCR DMR scenarios complete.")
