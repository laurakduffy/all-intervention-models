"""Diminishing returns sensitivity analysis across scenario anchors.

Specify named scenario triplets in SCENARIOS at the top of this file, then run:

    python diminishing_returns_sensitivity.py

One CSV is written to diminishing_returns/ for each entry in SCENARIOS.

Anchor format: list of (budget_multiple, marginal_ce_multiplier) tuples,
sorted by budget_multiple.  A budget_multiple of 1.0 means 1× the fund's
annual budget.

Fund 3-year budgets:
  Sentinel Bio              $21.6M   (7.2M/yr × 3 yrs)
  Longview Nuclear          $17.1M   (5.7M/yr × 3 yrs)
  Longview AI              $210.0M   (70M/yr  × 3 yrs)
"""

import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from export_rp_csv import DR_SPEND_POINTS, compute_diminishing_row

# ===========================================================================
# SCENARIOS — edit these to run your sensitivity analysis
# ===========================================================================
#
# Each entry is a named scenario triplet with anchors for all three funds.
# Add, remove, or rename entries freely.
#
# Format:
#   "scenario_name": {
#       "sentinel":        [(budget_multiple, ce_multiplier), ...],
#       "longview_nuclear": [(budget_multiple, ce_multiplier), ...],
#       "longview_ai":     [(budget_multiple, ce_multiplier), ...],
#   }

SCENARIOS = {
    "baseline": {
        "sentinel": [
            (1, 1.0), (2, 1.0), (5, 1.0), (10, 0.3), (20, 0.05),
        ],
        "longview_nuclear": [
            (1, 1.0), (2, 1.0), (5, 0.8), (8, 0.25), (20, 0.05),
        ],
        "longview_ai": [
            (50/70, 1.0), (190/70, 0.50), (260/70, 0.25),
        ],
    },
    "all_faster_diminishing": {
        "sentinel": [
            (1, 1.0), (2, 0.8), (5, 0.5), (10, 0.15), (20, 0.02),
        ],
        "longview_nuclear": [
            (1, 1.0), (2, 0.8), (5, 0.4), (8, 0.1), (20, 0.02),
        ],
        "longview_ai": [
            (50/70, 1.0), (120/70, 0.50), (190/70, 0.20), (260/70, 0.05),
        ],
    },
    "all_slower_diminishing": {
        "sentinel": [
            (1, 1.0), (2, 1.0), (5, 1.0), (10, 0.6), (20, 0.2),
        ],
        "longview_nuclear": [
            (1, 1.0), (2, 1.0), (5, 1.0), (8, 0.5), (20, 0.15),
        ],
        "longview_ai": [
            (50/70, 1.0), (190/70, 0.70), (350/70, 0.40), (500/70, 0.20),
        ],
    },
    "nuclear_faster_diminishing": {
        "sentinel": [
            (1, 1.0), (2, 1.0), (5, 1.0), (10, 0.3), (20, 0.05),
        ],
        "longview_nuclear": [
            (1, 1.0), (2, 0.8), (5, 0.5), (8, 0.1), (20, 0.01),
        ],
        "longview_ai": [
            (50/70, 1.0), (190/70, 0.50), (260/70, 0.25),
        ],
    },
}

# ===========================================================================
# Fund budgets in $M (kept here for reference; used by compute_diminishing_row)
# ===========================================================================

FUND_BUDGETS_M = {
    "sentinel":        7.2 * 3,   # $21.6M
    "longview_nuclear": 5.7 * 3,  # $17.1M
    "longview_ai":     70.0 * 3,  # $210.0M
}

FUND_PROJECT_IDS = {
    "sentinel":        "sentinel_bio",
    "longview_nuclear": "longview_nuclear",
    "longview_ai":     "longview_ai",
}

# ===========================================================================
# Core logic — no need to edit below this line
# ===========================================================================

OUTPUT_DIR = Path(__file__).parent / "diminishing_returns"


def _build_header_rows():
    """Return the two header rows (project_id + spend labels)."""
    spend_labels = [f"${s}M" for s in DR_SPEND_POINTS]
    return (
        ["project_id"] + spend_labels,
        [""] + ["cumulative $M invested"] * len(DR_SPEND_POINTS),
    )


def write_scenario_csv(scenario_combo, output_path, verbose=True):
    """Write a single diminishing-returns CSV for one scenario combination.

    Args:
        scenario_combo: dict mapping fund_key → (scenario_name, anchors)
        output_path: Path to write the CSV.
        verbose: print progress.
    """
    header1, header2 = _build_header_rows()
    rows = [header1, header2]

    for fund_key, (scenario_name, anchors) in scenario_combo.items():
        budget_m = FUND_BUDGETS_M[fund_key]
        project_id = FUND_PROJECT_IDS[fund_key]
        dr_vals = compute_diminishing_row(budget_m, anchors)
        rows.append([project_id] + [f"{v:.3f}" for v in dr_vals])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    if verbose:
        label = "  +  ".join(
            f"{k}={sn}" for k, (sn, _) in scenario_combo.items()
        )
        print(f"  Written: {output_path.name}  [{label}]")


def run_sensitivity():
    """Generate one CSV per entry in SCENARIOS."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SCENARIOS)} scenario(s) → {OUTPUT_DIR}/")

    for scenario_name, fund_anchors in SCENARIOS.items():
        scenario_combo = {
            fund_key: (scenario_name, anchors)
            for fund_key, anchors in fund_anchors.items()
        }
        filename = f"dr_{scenario_name}.csv"
        write_scenario_csv(scenario_combo, OUTPUT_DIR / filename)

    print(f"\nDone. {len(SCENARIOS)} CSV(s) written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_sensitivity()
