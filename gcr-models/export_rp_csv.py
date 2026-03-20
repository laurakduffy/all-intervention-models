"""Export RP-style CSV for all fund profiles.

Produces an RP-format CSV with two sections:
  1. Diminishing returns: marginal CE multiplier at $10M..$900M steps
  2. Effects at time horizon: human life years/$1M by period and risk profile

All risk profiles:

  Informal adjustments:
    neutral  = risk-neutral expected value (mean)
    upside   = upside skepticism — truncate upper tail at p99, renormalise
    downside = downside protection — loss-averse utility (lambda=2.5, ref=median)
    combined = percentile-based weighting + loss aversion 

  Formal models (Duffy 2023):
    dmreu    = Difference-Making Risk-Weighted EU (p=0.05, moderate aversion)
    wlu      = Weighted Linear Utility (c=0.01, 0.05, 0.1 concavity)
    ambiguity — Ambiguity Aversion with new percentile-based weighting (97.5-99.9% decay to 1% weight, above 99.9% zero weight)

Usage:
    python export_rp_csv.py                  # default output: gcr_output.csv
    python export_rp_csv.py -o my_output.csv
    python export_rp_csv.py --batch-size 2000 --quiet
"""

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

import numpy as np

from fund_profiles import get_fund_profile, YEARS_LOOK_AHEAD
from gcr_model import run_monte_carlo
from risk_profiles import compute_risk_profiles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUND_KEYS = ["sentinel", "longview_nuclear", "longview_ai"]

# Model period keys → t0..t4; t5 is computed as residual from total.
SHORT_PERIOD_KEYS = [
    "0 to 5",
    "5 to 10",
    "10 to 20",
    "20 to 100",
    "100 to 500",
]

RISK_PROFILES = [
    "neutral", "upside", "downside", "combined",
    "dmreu", "wlu - low", "wlu - moderate", "wlu - high", "ambiguity",
]


DR_SPEND_POINTS = list(range(10, 901, 10))  # $10M .. $900M in $10M steps

# ---------------------------------------------------------------------------
# Diminishing returns curve
# ---------------------------------------------------------------------------


def _eval_diminishing_raw(budget_m, anchors, spend_m):
    """Evaluate piecewise diminishing-returns curve (un-normalised).

    Args:
        budget_m: fund budget in $M.
        anchors: list of (budget_multiple, ce_multiplier) tuples, sorted by
            multiple.  Survey-derived at 1x/2x/5x with optional post-threshold
            points for sharp saturation.
        spend_m: cumulative spend in $M at which to evaluate.

    Returns:
        Raw marginal CE multiplier.
    """
    multiple = spend_m / budget_m

    if multiple <= anchors[0][0]:
        return anchors[0][1]

    if multiple >= anchors[-1][0]:
        last_mult, last_ce = anchors[-1]
        return last_ce * (last_mult / multiple)

    for i in range(len(anchors) - 1):
        m0, ce0 = anchors[i]
        m1, ce1 = anchors[i + 1]
        if m0 <= multiple <= m1:
            t = (multiple - m0) / (m1 - m0)
            return ce0 + t * (ce1 - ce0)

    return anchors[-1][1]


def compute_diminishing_row(budget_m, anchors):
    """Return normalised diminishing-returns values for each $10M step.

    Normalised so the $10M column = 1.000.
    """
    raw = [_eval_diminishing_raw(budget_m, anchors, s) for s in DR_SPEND_POINTS]
    base = raw[0]
    if base <= 0:
        return [0.0] * len(raw)
    return [v / base for v in raw]

# ---------------------------------------------------------------------------
# Sub-extinction tiers (simple EV model)
# ---------------------------------------------------------------------------


# Period boundaries in years, matching SHORT_PERIOD_KEYS + after_500_plus.
_PERIOD_BOUNDS = [(0, 5), (5, 10), (10, 20), (20, 100), (100, 500), (500, None)]


def _years_in_period(persistence, start, end):
    """How many years of [0, persistence] overlap with [start, end)."""
    if end is None:
        return max(0.0, persistence - start)
    return max(0.0, min(persistence, end) - max(0.0, start))

def _compute_sub_extinction_rows(profile, n_samples=100000, verbose=True):
    """Compute sub-extinction effect rows using Monte Carlo sampling with stratification."""
    p_harm = profile.get("p_harm", 0.0)
    harm_multiplier = profile.get("harm_multiplier", 1.0)
    tiers = profile.get("sub_extinction_tiers", [])
    if not tiers:
        return []

    budget = profile["budget"]
    adj = profile["adjustment_factor"]
    project_id = profile["export"]["project_id"]
    all_pk = SHORT_PERIOD_KEYS + ["after_500_plus"]
    rows = []

    # Get all combinations to stratify by
    first_tier = tiers[0]
    combos = list(itertools.product(
        first_tier["sweep_rel_rr"], 
        first_tier["sweep_persistence"]
    ))
    n_combos = len(combos)
    
    # Samples per combo
    samples_per_combo = n_samples // n_combos
    remainder = n_samples % n_combos
    
    # Build stratified samples
    rel_rr_samples = []
    persistence_samples = []
    shared_causes_harm = []
    
    for i, (rel_rr, pers) in enumerate(combos):
        n_in_combo = samples_per_combo + (1 if i < remainder else 0)
        
        rel_rr_samples.extend([rel_rr] * n_in_combo)
        persistence_samples.extend([pers] * n_in_combo)
        
        # Probabilistic rounding for exact p_harm
        n_harmful_expected = n_in_combo * p_harm
        n_harmful = int(n_harmful_expected)
        if np.random.random() < (n_harmful_expected - n_harmful):
            n_harmful += 1
        
        harm_mask = np.array([True] * n_harmful + [False] * (n_in_combo - n_harmful))
        np.random.shuffle(harm_mask)
        shared_causes_harm.extend(harm_mask)
    
    rel_rr_samples = np.array(rel_rr_samples)
    persistence_samples = np.array(persistence_samples)
    shared_causes_harm = np.array(shared_causes_harm)

    for tier in tiers:
        p_annual = 1 - (1 - tier["p_10yr"]) ** (1 / 10)
        discount = tier.get("discount", 1.0)

        annual_evs = (
            p_annual * tier["expected_deaths"] * rel_rr_samples
            * adj * discount
        )
        
        annual_evs = np.where(shared_causes_harm, -annual_evs * harm_multiplier, annual_evs)
        
        horizon_data = {}
        for pk, (t_start, t_end) in zip(all_pk, _PERIOD_BOUNDS):
            yrs = np.array([_years_in_period(p, t_start, t_end)
                            for p in persistence_samples])
            period_evs = annual_evs * yrs
            per_1m = period_evs / budget * 1e6
            horizon_data[pk] = compute_risk_profiles(per_1m)

        total_per_1m = annual_evs * persistence_samples / budget * 1e6
        total_profiles = compute_risk_profiles(total_per_1m)

        if verbose:
            n_positive = np.sum(total_per_1m > 0)
            n_negative = np.sum(total_per_1m < 0)
            actual_samples = len(total_per_1m)
            print(f"  Sub-ext tier '{tier['tier_name']}': "
                  f"{actual_samples:,} MC samples (stratified), "
                  f"neutral={total_profiles['neutral']:.4g} lives-eq/$1M "
                  f"({n_positive:,} pos, {n_negative:,} neg = {100*n_negative/actual_samples:.1f}% harm)")

        rows.append({
            "export_meta": {
                "project_id": tier.get("project_id", project_id),
                "near_term_xrisk": tier.get("near_term_xrisk", False),
                "effect_id": tier["effect_id"],
                "recipient_type": tier["recipient_type"],
                "tier_name": tier["tier_name"],
            },
            "horizon_data": horizon_data,
            "total_per_1m": total_per_1m,
        })

    return rows

# ---------------------------------------------------------------------------
# Sweep runner + risk profile extraction
# ---------------------------------------------------------------------------
def run_fund_and_extract(fund_key, n_samples=100000, verbose=True):
    """Run Monte Carlo sampling for one fund, return horizon data + summary."""
    profile = get_fund_profile(fund_key)
    budget = profile["budget"]
    adj = profile["adjustment_factor"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running MC: {profile['display_name']}")
        print(f"  Budget: ${budget / 1e6:.1f}M  |  Adjustment: {adj:.3f}")
        print(f"  Samples: {n_samples:,}")
        print(f"{'=' * 60}")

    t0 = time.time()
    results = run_monte_carlo(
        sweep_params=profile["sweep_params"],
        fixed_params=profile["fixed_params"],
        n_samples=n_samples,
        verbose=verbose,
        p_harm=profile.get("p_harm", 0.0),
        p_zero=profile.get("p_zero", 0.0),
        harm_multiplier=profile.get("harm_multiplier", 1.0),
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"  Done: {n_samples:,} samples in {elapsed:.1f}s")

    evp = results["ev_per_period"]
    zeros = np.zeros(n_samples)

    horizon_raw = {}
    for pk in SHORT_PERIOD_KEYS:
        horizon_raw[pk] = evp.get(pk, zeros.copy())

    total_raw = evp["Total Value"]
    sum_short = sum(horizon_raw[pk] for pk in SHORT_PERIOD_KEYS)
    horizon_raw["after_500_plus"] = total_raw - sum_short

    all_period_keys = SHORT_PERIOD_KEYS + ["after_500_plus"]

    horizon_data = {}
    for pk in all_period_keys:
        per_1m = horizon_raw[pk] * adj / budget * 1e6
        horizon_data[pk] = compute_risk_profiles(per_1m)

    total_per_1m = total_raw * adj / budget * 1e6
    total_profiles = compute_risk_profiles(total_per_1m)
    summary = {
        "n_samples": n_samples,
        **{f"total_{k}": v for k, v in total_profiles.items()},
    }

    if verbose:
        print(f"  Total human life years/$1M (informal):  "
              f"neutral={total_profiles['neutral']:.4g}  "
              f"upside={total_profiles['upside']:.4g}  "
              f"downside={total_profiles['downside']:.4g}  "
              f"combined={total_profiles['combined']:.4g}")
        print(f"  Total human life years/$1M (formal):    "
              f"dmreu={total_profiles['dmreu']:.4g}  "
              f"wlu low={total_profiles['wlu - low']:.4g}  "
              f"wlu mod={total_profiles['wlu - moderate']:.4g}  "
              f"wlu high={total_profiles['wlu - high']:.4g}  "
              f"ambiguity={total_profiles['ambiguity']:.4g}")

    sub_ext_rows = _compute_sub_extinction_rows(profile, n_samples=n_samples, verbose=verbose)

    return {
        "profile": profile,
        "horizon_data": horizon_data,
        "summary": summary,
        "sub_ext_rows": sub_ext_rows,
        "total_per_1m": total_per_1m,
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

N_DR_COLS = len(DR_SPEND_POINTS)
TOTAL_COLS = 1 + N_DR_COLS  # widest section determines CSV width


def _pad(row):
    """Pad or trim row to TOTAL_COLS."""
    if len(row) < TOTAL_COLS:
        row.extend([""] * (TOTAL_COLS - len(row)))
    return row[:TOTAL_COLS]


def _fmt(v):
    """Format a human life years value for CSV (4 significant figures)."""
    return f"{v:.4g}"

def write_diminishing_returns_csv(fund_results, output_path, verbose=True):
    """Write separate CSV with just diminishing returns."""
    rows = []
    
    # Header
    rows.append(["project_id"] + [f"${s}M" for s in DR_SPEND_POINTS])
    rows.append([""] + ["cumulative $M invested"] * len(DR_SPEND_POINTS))
    
    # Data rows - one per fund
    for fr in fund_results:
        export = fr["profile"]["export"]
        budget_m = fr["profile"]["budget"] / 1e6
        dr_vals = compute_diminishing_row(budget_m, export["diminishing_anchors"])
        rows.append([export["project_id"]] + [f"{v:.3f}" for v in dr_vals])
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    
    if verbose:
        print(f"\nDiminishing returns CSV written to: {output_path}")
        print(f"  {len(fund_results)} funds, {len(DR_SPEND_POINTS)} spend levels")


def write_rp_csv(fund_results, output_path, verbose=True):
    """Write RP-format CSV with both sections."""
    rows = []

    # ── Effects at Time Horizon ──
    n_t = 6  # t0..t5
    rp_labels = {
        "neutral": "Risk profile: NEUTRAL",
        "upside": "Risk profile: UPSIDE sceptical",
        "downside": "Risk profile: DOWNSIDE CRITICAL",
        "combined": "Risk profile: COMBINED",
        "dmreu": "Risk profile: DMREU",
        "wlu - low": "Risk profile: WLU (low, 0.01)",
        "wlu - moderate": "Risk profile: WLU (moderate, 0.05)",
        "wlu - high": "Risk profile: WLU (high, 0.1)",
        "ambiguity": "Risk profile: AMBIGUITY AVERSION",
    }
    header2 = ["effects at time horizon", "", "", ""]
    for rp in RISK_PROFILES:
        header2.append(rp_labels[rp])
        header2.extend([""] * (n_t - 1))
    rows.append(_pad(header2))

    col_names = ["project_id", "near_term_xrisk", "effect_id", "recipient_type"]
    for rp in RISK_PROFILES:
        for ti in range(n_t):
            col_names.append(f"{rp}_t{ti}")
    rows.append(_pad(col_names))

    all_pk = SHORT_PERIOD_KEYS + ["after_500_plus"]

    def _effect_row(export_meta, hd):
        data_row = [
            export_meta["project_id"],
            str(export_meta["near_term_xrisk"]).upper(),
            export_meta["effect_id"],
            export_meta["recipient_type"],
        ]
        for rp in RISK_PROFILES:
            for pk in all_pk:
                data_row.append(_fmt(hd[pk][rp]))
        return _pad(data_row)

    for fr in fund_results:
        rows.append(_effect_row(fr["profile"]["export"], fr["horizon_data"]))
        for sub in fr.get("sub_ext_rows", []):
            rows.append(_effect_row(sub["export_meta"], sub["horizon_data"]))

    rows.append(_pad([""]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    if verbose:
        print(f"\nCSV written to: {output_path}")
        print(f"  {len(fund_results)} funds, {TOTAL_COLS} columns")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_output(output_path, n_funds, n_effect_rows, verbose=True):
    """Structural checks on the output CSV."""
    with open(output_path, "r") as f:
        all_rows = list(csv.reader(f))

    errors = []

    # Main CSV should start with effects
    if all_rows[0][0] != "effects at time horizon":
        errors.append("CSV should start with 'effects at time horizon'")
        return len(errors) == 0
    effects_idx = 0
    if all_rows[effects_idx + 1][0] != "project_id":
        errors.append("Missing effects column headers")
    for i in range(n_effect_rows):
        row = all_rows[effects_idx + 2 + i]
        if not row[0]:
            errors.append(f"Missing project_id in effects row {i}")

    if verbose:
        if errors:
            print(f"\nValidation FAILED ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"\nValidation PASSED: {n_funds} funds, "
                  f"{n_effect_rows} effect rows, both sections OK.")

    return len(errors) == 0


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def create_and_save_histograms(fund_results, output_dir, verbose=True):
    """Save linear-scale and log-scale histograms for each fund's total QALYs/$1M."""
    os.makedirs(output_dir, exist_ok=True)

    def _save_histogram(project_id, display_name, samples):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{display_name}\nTotal QALYs per $1M", fontsize=13)

        # Linear scale
        axes[0].hist(samples, bins=60, alpha=0.75, color="steelblue", edgecolor="none")
        axes[0].set_xlabel("QALYs / $1M")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Linear scale")
        axes[0].axvline(float(np.mean(samples)), color="red", linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(samples):.3g}")
        axes[0].axvline(float(np.median(samples)), color="orange", linestyle="--", linewidth=1.2, label=f"Median = {np.median(samples):.3g}")
        axes[0].legend(fontsize=9)

        # Log scale (x-axis); drop non-positive values
        pos = samples[samples > 0]
        if len(pos) > 0:
            log_bins = np.logspace(np.log10(np.percentile(pos, 1)), np.log10(np.percentile(pos, 99)), 60)
            axes[1].hist(pos, bins=log_bins, alpha=0.75, color="steelblue", edgecolor="none")
            axes[1].set_xscale("log")
            axes[1].axvline(float(np.mean(pos)), color="red", linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(pos):.3g}")
            axes[1].axvline(float(np.median(pos)), color="orange", linestyle="--", linewidth=1.2, label=f"Median = {np.median(pos):.3g}")
            axes[1].legend(fontsize=9)
            pct_negative = 100 * np.sum(samples <= 0) / len(samples)
            axes[1].set_title(f"Log scale (positive values only; {pct_negative:.1f}% ≤ 0 excluded)")
        else:
            axes[1].text(0.5, 0.5, "No positive values", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Log scale")
        axes[1].set_xlabel("QALYs / $1M")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{project_id}_histogram.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        if verbose:
            print(f"  Saved histogram: {out_path}")

    for fr in fund_results:
        _save_histogram(
            fr["profile"]["export"]["project_id"],
            fr["profile"]["display_name"],
            fr["total_per_1m"],
        )
        for sub in fr.get("sub_ext_rows", []):
            _save_histogram(
                sub["export_meta"]["effect_id"],
                sub["export_meta"]["tier_name"],
                sub["total_per_1m"],
            )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def write_summary_statistics(fund_results, output_path, verbose=True):
    """Write a CSV with per-fund summary statistics for total QALYs/$1M."""
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    fieldnames = ["fund", "display_name", "tier_name", "n_samples", "mean"] + [f"p{p}" for p in percentiles]

    def _make_row(fund_id, display_name, tier_name, samples):
        row = {
            "fund": fund_id,
            "display_name": display_name,
            "tier_name": tier_name,
            "n_samples": len(samples),
            "mean": float(np.mean(samples)),
        }
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(samples, p))
        return row

    rows = []
    for fr in fund_results:
        rows.append(_make_row(
            fr["profile"]["export"]["project_id"],
            fr["profile"]["display_name"],
            "",
            fr["total_per_1m"],
        ))
        for sub in fr.get("sub_ext_rows", []):
            rows.append(_make_row(
                sub["export_meta"]["project_id"],
                sub["export_meta"]["tier_name"],
                sub["export_meta"]["tier_name"],
                sub["total_per_1m"],
            ))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"\nSummary statistics written to: {output_path}")
        for row in rows:
            label = f"{row['display_name']}" + (f" [{row['tier_name']}]" if row["tier_name"] else "")
            print(f"  {label}: mean={row['mean']:.4g}, "
                  f"p5={row['p5']:.4g}, p50={row['p50']:.4g}, p95={row['p95']:.4g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export RP-style CSV for all GCR fund profiles (Monte Carlo)."
    )
    parser.add_argument(
        "-o", "--output", default="gcr_output.csv",
        help="Output CSV path for effects (default: gcr_output.csv). Diminishing returns will be saved to diminishing_returns/gcr_diminishing_returns_{N}yr.csv",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100000,
        help="Number of Monte Carlo samples per fund (default: 100000).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-fund progress output.",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 70)
    print("RP CSV EXPORT — ALL FUNDS (MONTE CARLO)")
    print("=" * 70)

    fund_results = []
    for fk in FUND_KEYS:
        fr = run_fund_and_extract(fk, n_samples=args.n_samples, verbose=verbose)
        fund_results.append(fr)

    # Write main effects CSV
    write_rp_csv(fund_results, args.output, verbose=verbose)
    
    # Write separate diminishing returns CSV
    dr_dir = Path(args.output).parent / "diminishing_returns"
    dr_dir.mkdir(exist_ok=True)
    dr_output = str(dr_dir / f"gcr_diminishing_returns_{YEARS_LOOK_AHEAD}yr.csv")
    write_diminishing_returns_csv(fund_results, dr_output, verbose=verbose)

    n_effect_rows = sum(
        1 + len(fr.get("sub_ext_rows", []))
        for fr in fund_results
    )
    ok = validate_output(args.output, len(FUND_KEYS), n_effect_rows, verbose=verbose)

    # Histograms
    hist_dir = str(Path(args.output).parent / "histograms")
    print(f"\nCreating histograms in: {hist_dir}")
    create_and_save_histograms(fund_results, hist_dir, verbose=verbose)

    # Summary statistics
    stats_output = str(Path(args.output).with_suffix("")) + "_summary_stats.csv"
    write_summary_statistics(fund_results, stats_output, verbose=verbose)

    if not ok:
        print("\nExport completed with validation errors!")
        sys.exit(1)
    else:
        print("\nExport completed successfully.")


if __name__ == "__main__":
    main()