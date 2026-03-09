"""CLI entry point for AW fund marginal cost-effectiveness pipeline.

Usage:
    source ../test_env/bin/activate
    python run.py                      # full pipeline, default outputs
    python run.py --fund aw_combined   # specific fund profile
    python run.py --verbose            # detailed progress output
    python run.py -o custom_dir/       # custom output directory
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline.build_dataset import build_all_effects
from src.pipeline.export import (
    export_dataset, export_assumptions, export_sensitivity,
    # export_diminishing,  # Removed - generating separately
)


def main():
    parser = argparse.ArgumentParser(
        description="AW Fund Marginal Cost-Effectiveness Pipeline"
    )
    parser.add_argument(
        "--fund", default="aw_combined",
        help="Fund profile key (default: aw_combined).",
    )
    parser.add_argument(
        "-o", "--output-dir", default="outputs",
        help="Output directory (default: outputs/).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("AW FUND MARGINAL COST-EFFECTIVENESS PIPELINE")
    print(f"Fund: {args.fund}")
    print("=" * 70)

    dataset = build_all_effects(
        fund_key=args.fund,
        verbose=args.verbose,
    )

    project_id = dataset["fund_config"]["project_id"]

    csv_path = os.path.join(args.output_dir, f"{project_id}_dataset.csv")
    export_dataset(dataset, csv_path, verbose=args.verbose)

    assumptions_path = os.path.join(args.output_dir, f"{project_id}_assumptions.md")
    export_assumptions(dataset, assumptions_path, verbose=args.verbose)

    sensitivity_path = os.path.join(args.output_dir, f"{project_id}_sensitivity.csv")
    export_sensitivity(dataset, sensitivity_path, verbose=args.verbose)

    # Diminishing returns removed - generating separately
    # diminishing_path = os.path.join(args.output_dir, f"{project_id}_diminishing.csv")
    # export_diminishing(dataset, diminishing_path, verbose=args.verbose)

    print(f"\nAll outputs written to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
