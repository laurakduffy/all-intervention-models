"""Fund-specific parameter profiles for GCR sweep runs.

This module keeps survey mappings in one place so each fund can be run
independently without editing model code.
"""

from copy import deepcopy

import numpy as np

from gcr_model import M, _solve_r_max


def _r_max_from_cumulative_risk(
    cumulative_risk_100_yrs,
    year_max_risk=15,
    year_risk_1pct_max=100,
    r_inf=1e-7,
):
    """Exact r_max for a given cumulative risk and Gaussian shape parameters.

    Delegates to _solve_r_max (vectorized bisection). Defaults use the central
    values from _RP_WORLD_PRIORS (year_max_risk=15, year_risk_1pct_max=100,
    r_inf=1e-7) so the CSV reflects a representative scenario.

    Pass scalar or array inputs; returns an array of the same shape.
    """
    scalar = np.ndim(cumulative_risk_100_yrs) == 0
    cum = np.atleast_1d(np.asarray(cumulative_risk_100_yrs, dtype=float))
    n = len(cum)
    result = _solve_r_max(
        cum,
        np.full(n, year_max_risk, dtype=float),
        np.full(n, year_risk_1pct_max, dtype=float),
        np.full(n, r_inf, dtype=float),
    )
    return float(result[0]) if scalar else result

# ---------------------------------------------------------------------------
# Cause-specific risk fractions (share of total x-risk per cause).
# Derived from RP house-view: (AI direct + AI indirect) / total, etc.
# Source: RP Cross Cause Model
# ---------------------------------------------------------------------------
_AI_CAUSE_FRACTION      = (0.5541 + 0.06157) / 0.67   # ~0.919
_NUCLEAR_CAUSE_FRACTION = 0.02354 / 0.67               # ~0.035
_BIO_CAUSE_FRACTION     = 0.004183 / 0.67              # ~0.006

# Q4.4 declined. Derived from field-level reasoning (bio field ~$30-80M/yr,
# Sentinel is one of two funders >$10M/yr).
# rel_risk_reduction = rel_per_10m * (budget / $10M), independent of total risk level.
# Assume same as nuclear risk reduction per $10M, with uncertainty envelope.
_SENTINEL_REL_REDUCTION_PER_10M = [0.002/50, 0.002/10, 0.002] 
_SENTINEL_REL_RISK_REDUCTION = [
    rel * (7.2 * M / (10 * M)) for rel in _SENTINEL_REL_REDUCTION_PER_10M
]

# Total x-risk (all causes) cumulative probability over 100 years.
# The Gaussian "Time of Perils" peak is calibrated to total x-risk so that
# all hazards (AI, bio, nuclear, etc.) are present in every simulation.
# Fund-specific rel_risk_reduction * cause_fraction gives the fraction of
# total r_max reduced, computed inside the model — no dependency on risk level.
_TOTAL_XRISK_100YR = [0.05, 0.10, 0.65]

_RP_WORLD_PRIORS = {
    # Total x-risk framing (all causes), from RP house-view inputs.
    "r_inf": [1e-10, 1e-7, 1e-3],
    "year_risk_1pct_max": [20, 100, 200],
    "year_max_risk": [5, 15, 50],
    # Future trajectory priors.
    "carrying_capacity_multiplier": {'values': [1.5, 5.0, 100.0], 'p': [0.6, 0.3, 0.1]},  # unlikely high growth
    "rate_growth": [0.005, 0.01, 0.04],
    "cubic_growth": {"values": [False, True], "p": [0.90, 0.10]},
    "T_c": {'values': [500, 300, 80], 'p': [0.6, 0.3, 0.1]}, # seems unlikely in the next 80 years
    "s": [0.01, 0.1],
}

# Longview Nuclear 4.4 (extinction): 0.2% per $10M, with uncertainty envelope.
# Might be too optimistic, downweight central by 10x, pessimistic by 50x
# Thus, we get 0.02% per $10M as the central estimate, with a range from 0.01% to 0.03% per $10M.
# This translates to 1-3bp rel risk reduction per $10M (i.e. 1e-4 to 3e-4 as a fraction).
_NUCLEAR_REL_REDUCTION_PER_10M = [0.002/50, 0.002/10, 0.002] 
_NUCLEAR_REL_RISK_REDUCTION = [
    rel * (5.7 * M / (10 * M)) for rel in _NUCLEAR_REL_REDUCTION_PER_10M
]

# Longview AI declined 4.3 and 4.4: use RP world priors + explicit assumption.
# Cost-effectiveness assumed as cost-effective as nuclear risk reduction (rel reduction per $10M).

_AI_REL_REDUCTION_PER_10M = [0.002/50, 0.002/10, 0.002] 
_AI_REL_RISK_REDUCTION = [
    rel * (70 * M / (10 * M)) for rel in _AI_REL_REDUCTION_PER_10M
]

FUND_PROFILES = {
    "sentinel": {
        "display_name": "Sentinel Bio",
        "budget": 7.2 * M,
        "counterfactual_factor": 0.80 * 1.0 + 0.15 * 0.5 + 0.05 * 0.0,  # 0.875
        "p_harm": 0.05, 
        "p_zero": 0.50, 
        "harm_multiplier": 1.0,
        "sweep_params": {
            **_RP_WORLD_PRIORS,
            "cumulative_risk_100_yrs": _TOTAL_XRISK_100YR,
            "rel_risk_reduction": _SENTINEL_REL_RISK_REDUCTION,
        },
        "fixed_params": {
            "budget": 7.2 * M,
            "periods_value": [0, 5, 10, 20, 100, 500],
            "T_h": 1e14,
            "year_effect_starts": 0,
            "persistence_effect": 15,
            "initial_value": 8e9,
            "cause_fraction": _BIO_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "sentinel_bio",
            "near_term_xrisk": True,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
            # (budget_multiple, marginal_ce_multiplier) from survey Q3.2/Q3.3.
            # Q3.2 (2x): "stay roughly the same"; Q3.3 (5x): "would improve"
            # (weakest-link threshold dynamics at ~70% DNA screening prevalence).
            # Beyond 5x (~$36M): saturates core bio-prevention areas; stated
            # max deployable at current CE is $15-20M (Q3.1).
            "diminishing_anchors": [
                (1, 1.0), (2, 1.0), (5, 1), (10, 0.3), (20, 0.05),
            ],
        },
        # Sub-extinction tiers (simple EV: P(event) × deaths × rel_rr × persistence).
        # From survey Q4.3 risk estimates + derived rel_rr (Q4.4 declined).
        #
        # sweep_rel_rr = rel_per_10m × (budget / $10M)
        # Uses the same conservative/central/optimistic scenarios as the extinction pathway.
        "sub_extinction_tiers": [
            {
                "tier_name": "100M-1B deaths",
                "project_id": "sentinel_bio_100m_1b",
                "effect_id": "effect_human_lives_sub_ext_100m_1b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.02, # from survey, question 4.3.1
                "expected_deaths": 316e6,  # geomean(100M, 1B)
                "natural_pandemic_discount": 1.0,  # no discount
                "sweep_rel_rr": _SENTINEL_REL_RISK_REDUCTION,
                "sweep_persistence": [10, 15, 25],
            },
            {
                "tier_name": "10M-100M deaths",
                "project_id": "sentinel_bio_10m_100m",
                "effect_id": "effect_human_lives_sub_ext_10m_100m",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.30,
                "expected_deaths": 31.6e6,  # geomean(10M, 100M)
                "natural_pandemic_discount": 0.3,  # Sentinel focuses on engineered bio
                "sweep_rel_rr": _SENTINEL_REL_RISK_REDUCTION,
                "sweep_persistence": [10, 15, 25],
            },
        ],
    },
    "longview_nuclear": {
        "display_name": "Longview Philanthropy Nuclear Weapons Policy Fund",
        "budget": 5.7 * M,
        "counterfactual_factor": 0.80 * 1.0 + 0.10 * 0.5 + 0.10 * 0.0,  # 0.85
        "p_harm": 0.05, 
        "p_zero": 0.50,  
        "harm_multiplier": 1.0, 

        "sweep_params": {
            **_RP_WORLD_PRIORS,
            "cumulative_risk_100_yrs": _TOTAL_XRISK_100YR,
            "rel_risk_reduction": _NUCLEAR_REL_RISK_REDUCTION,
        },
        "fixed_params": {
            "budget": 5.7 * M,
            "periods_value": [0, 5, 10, 20, 100, 500],
            "T_h": 1e14,
            # Derived from Section 6.1 weighted timing (~4 years).
            "year_effect_starts": 4,
            # Derived from Section 6.2 weighted persistence (~21 years).
            "persistence_effect": 21,
            "initial_value": 8e9,
            "cause_fraction": _NUCLEAR_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "longview_nuclear",
            "near_term_xrisk": True,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
            # Q3.2 (2x ~$11M): "cost-effectiveness would improve" — increasing
            # returns up to ~$25M due to underfunded field restoring capacity.
            # Q3.3 (5x ~$29M): "begin to see diminishing marginal effectiveness"
            # but field still small enough to compare favorably.
            # Beyond 5x: total nuclear philanthropy field is ~$45M/yr; at ~$25M
            # Longview would be half the field. Sharp saturation beyond 8x.
            "diminishing_anchors": [
                (1, 1.0), (2, 1), (5, 0.8), (8, 0.25), (20, 0.05),
            ],
        },
    },
    "longview_ai": {
        "display_name": "Longview Philanthropy AI Program",
        "budget": 70 * M,
        "counterfactual_factor": 0.60 * 1.0 + 0.25 * 0.5 + 0.15 * 0.0,  # 0.725
        "p_harm": 0.15,  
        "p_zero": 0.50,
        "harm_multiplier": 1.0,
        "sweep_params": {
            **_RP_WORLD_PRIORS,
            "cumulative_risk_100_yrs": _TOTAL_XRISK_100YR,
            # AI survey declined 4.4; use explicit low intervention-effect priors.
            "rel_risk_reduction": _AI_REL_RISK_REDUCTION,
        },
        "fixed_params": {
            "budget": 70 * M,
            "periods_value": [0, 5, 10, 20, 100, 500],
            "T_h": 1e14,
            # Section 6.1 weighted timing (~2.8 years).
            "year_effect_starts": 3,
            # Section 6.2 skipped in survey; conservative prior assumption.
            "persistence_effect": 12,
            "initial_value": 8e9,
            "cause_fraction": _AI_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "longview_ai",
            "near_term_xrisk": True,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
            # Q3.2 (2x ~$140M): "stay approximately the same".
            # Q3.3 (5x ~$350M): "decline by ~75%".
            # Modified to assume that returns start to diminish around 50M, 50% by $190M
            "diminishing_anchors": [(50/70, 1.0), (190/70, 0.50), (260/70, 0.25)],
        },
    },
}


def list_fund_profiles():
    return sorted(FUND_PROFILES.keys())


def get_fund_profile(fund_key):
    key = fund_key.strip().lower()
    if key not in FUND_PROFILES:
        valid = ", ".join(list_fund_profiles())
        raise KeyError(f"Unknown fund '{fund_key}'. Valid options: {valid}")

    profile = deepcopy(FUND_PROFILES[key])
    profile["fund_key"] = key
    # NEW (only counterfactual, harm is handled in simulation):
    profile["adjustment_factor"] = profile["counterfactual_factor"]
    return profile


def make_earth_only_profile(profile):
    """Return a profile variant with stellar expansion disabled."""
    out = deepcopy(profile)
    out["sweep_params"].pop("cubic_growth", None)
    out["sweep_params"].pop("T_c", None)
    out["sweep_params"].pop("s", None)
    out["fixed_params"]["cubic_growth"] = False
    out["fixed_params"]["T_c"] = 500
    out["fixed_params"]["s"] = 0.01
    return out


if __name__ == "__main__":
    # ── Option A calibration display ─────────────────────────────────────────
    # rel_risk_reduction = rel_per_10m * (budget / $10M)  [independent of risk level]
    # cause_fraction     = cause-specific share of total x-risk
    # rel_rr_from_int    = rel_risk_reduction * cause_fraction  [used by model]
    # Shown at three total x-risk scenarios for reference.

    SEP = "=" * 70

    def _show_fund(name, budget_label, rel_per_10m_list, rel_rr_list, cause_frac, scenarios):
        print(SEP)
        print(f"  {name}")
        print(SEP)
        print(f"  Budget: {budget_label}  |  cause_fraction: {cause_frac:.5f}")
        print()
        print(f"  {'Scenario':<14}  {'rel/$10M':>10}  {'rel_rr':>10}  {'rel_rr_from_int':>16}  "
              f"{'% total r_max @5%':>18}  {'% total r_max @10%':>19}  {'% total r_max @65%':>19}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*18}  {'-'*19}  {'-'*19}")
        for label, rel_per_10m, rel_rr in zip(scenarios, rel_per_10m_list, rel_rr_list):
            rr_int = rel_rr * cause_frac
            pcts = [rr_int / _r_max_from_cumulative_risk(c) * 100 for c in [0.05, 0.10, 0.65]]
            print(f"  {label:<14}  {rel_per_10m:>9.3%}  {rel_rr:>9.4%}  {rr_int:>15.5%}  "
                  f"  {pcts[0]:>16.4f}%  {pcts[1]:>17.4f}%  {pcts[2]:>17.4f}%")
        print()

    _show_fund(
        "Sentinel Bio", "$7.2M",
        _SENTINEL_REL_REDUCTION_PER_10M, _SENTINEL_REL_RISK_REDUCTION,
        _BIO_CAUSE_FRACTION,
        ["conservative", "central", "optimistic"],
    )
    _show_fund(
        "Longview Nuclear", "$5.7M",
        _NUCLEAR_REL_REDUCTION_PER_10M, _NUCLEAR_REL_RISK_REDUCTION,
        _NUCLEAR_CAUSE_FRACTION,
        ["conservative", "central", "optimistic"],
    )
    _show_fund(
        "Longview AI", "$70M",
        _AI_REL_REDUCTION_PER_10M, _AI_REL_RISK_REDUCTION,
        _AI_CAUSE_FRACTION,
        ["central"],
    )

    # ── Abs risk reduction per $10M — CSV output ──────────────────────────────
    # For each fund, enumerate all (rel_scenario x cum_risk_scenario) combinations.
    # abs_risk_reduction_per_10m = rel_per_10m * cause_fraction * r_max(cum_risk)
    import csv
    import math
    import os
    import statistics

    _HERE = os.path.dirname(os.path.abspath(__file__))

    _CSV_FUND_CONFIGS = [
        {"fund": "sentinel_bio",    "rel_scenarios": ["conservative", "central", "optimistic"], "rel_per_10m": _SENTINEL_REL_REDUCTION_PER_10M, "cause_fraction": _BIO_CAUSE_FRACTION},
        {"fund": "longview_nuclear","rel_scenarios": ["conservative", "central", "optimistic"], "rel_per_10m": _NUCLEAR_REL_REDUCTION_PER_10M,  "cause_fraction": _NUCLEAR_CAUSE_FRACTION},
        {"fund": "longview_ai",     "rel_scenarios": ["conservative", "central", "optimistic"], "rel_per_10m": _AI_REL_REDUCTION_PER_10M,       "cause_fraction": _AI_CAUSE_FRACTION},
    ]
    _CUM_LABELS = ["low_5pct", "central_10pct", "high_65pct"]

    detail_rows = []
    for cfg in _CSV_FUND_CONFIGS:
        for rel_label, rel in zip(cfg["rel_scenarios"], cfg["rel_per_10m"]):
            for cum_label, cum in zip(_CUM_LABELS, _TOTAL_XRISK_100YR):
                r_max_val = _r_max_from_cumulative_risk(cum)
                detail_rows.append({
                    "fund": cfg["fund"], "rel_scenario": rel_label, "cum_risk_scenario": cum_label,
                    "rel_per_10m": rel, "cum_risk_100yr": cum,
                    "cause_fraction": cfg["cause_fraction"], "r_max": r_max_val,
                    # abs risk reduction at the Gaussian peak (peak annual, not cumulative).
                    # Note: persistence of effect << 100yr so peak annual is the relevant unit.
                    # r_max solved exactly for central Gaussian params (T_c=15, sigma=100/3, r_inf=1e-7).
                    "peak_annual_abs_risk_reduction_per_10m": rel * cfg["cause_fraction"] * r_max_val,
                    "peak_annual_abs_risk_reduction_bp_per_1b": rel * cfg["cause_fraction"] * r_max_val * 1_000_000,
                })

    detail_path = os.path.join(_HERE, "calibration_abs_risk_reduction_detail.csv")
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fund","rel_scenario","cum_risk_scenario","rel_per_10m","cum_risk_100yr","cause_fraction","r_max","peak_annual_abs_risk_reduction_per_10m","peak_annual_abs_risk_reduction_bp_per_1b"])
        w.writeheader(); w.writerows(detail_rows)

    summary_rows = []
    for cfg in _CSV_FUND_CONFIGS:
        bp_vals = [r["peak_annual_abs_risk_reduction_bp_per_1b"] for r in detail_rows if r["fund"] == cfg["fund"]]
        geo_mean_bp = math.exp(sum(math.log(v) for v in bp_vals) / len(bp_vals))
        summary_rows.append({"fund": cfg["fund"], "n_scenarios": len(bp_vals), "min_bp": min(bp_vals), "max_bp": max(bp_vals), "mean_bp": statistics.mean(bp_vals), "median_bp": statistics.median(bp_vals), "geometric_mean_bp": geo_mean_bp})

    summary_path = os.path.join(_HERE, "calibration_abs_risk_reduction_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fund","n_scenarios","min_bp","max_bp","mean_bp","median_bp","geometric_mean_bp"])
        w.writeheader(); w.writerows(summary_rows)

    print(f"Detail CSV:  {detail_path}")
    print(f"Summary CSV: {summary_path}")
