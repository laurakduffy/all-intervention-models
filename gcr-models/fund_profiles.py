"""Fund-specific parameter profiles for GCR sweep runs.

This module keeps survey mappings in one place so each fund can be run
independently without editing model code.
"""

from copy import deepcopy

from gcr_model import M


def _r_max_from_cumulative_risk(cumulative_risk_100_yrs):
    """Mirror model's r_max derivation for calibration."""
    return 1.38 * (1 - (1 - cumulative_risk_100_yrs) ** (1 / 100))


def _convert_cumulative_probability(probability, years_from, years_to):
    """Convert cumulative probability between horizons via constant annual rate."""
    annual = 1 - (1 - probability) ** (1 / years_from)
    return 1 - (1 - annual) ** years_to


def _abs_reduction_from_per_10m(
    rel_reduction_per_10m,
    budget,
    cumulative_risk_100_ref,
):
    """Convert per-$10M relative reduction priors into absolute peak reductions."""
    r_max_ref = _r_max_from_cumulative_risk(cumulative_risk_100_ref)
    budget_factor = budget / (10 * M)
    return [r_max_ref * rel * budget_factor for rel in rel_reduction_per_10m]


# Sentinel Bio: Q4.3 gives bio extinction at 0.01% over 10 years.
# Convert to 100-year cumulative for calibration reference.
_SENTINEL_BIO_EXTINCTION_10YR = 0.0001
_SENTINEL_BIO_EXTINCTION_100YR = _convert_cumulative_probability(
    _SENTINEL_BIO_EXTINCTION_10YR, years_from=10, years_to=100,
)
# Q4.4 declined. Derived from field-level reasoning (bio field ~$30-80M/yr,
# Sentinel is one of two funders >$10M/yr).
# rel_reduction_per_10m: conservative=0.2%, central=1%, optimistic=5%
_SENTINEL_REL_REDUCTION_PER_10M = [0.001, 0.004, 0.01]
_SENTINEL_ABS_RISK_REDUCTION = _abs_reduction_from_per_10m(
    rel_reduction_per_10m=_SENTINEL_REL_REDUCTION_PER_10M,
    budget=7.2 * M,
    cumulative_risk_100_ref=_SENTINEL_BIO_EXTINCTION_100YR,
)


# Total x-risk (all causes) cumulative probability over 100 years.
# The Gaussian "Time of Perils" peak is calibrated to total x-risk so that
# all hazards (AI, bio, nuclear, etc.) are present in every simulation.
# Fund-specific intervention effects (abs_risk_reduction) only reduce the
# cause-specific slice of this total risk.
_TOTAL_XRISK_100YR = [0.05, 0.10, 0.65]

_RP_WORLD_PRIORS = {
    # Total x-risk framing (all causes), from RP house-view inputs.
    "r_inf": [1e-10, 1e-7, 1e-3],
    "year_risk_1pct_max": [20, 100, 200],
    "year_max_risk": [5, 15, 50],
    # Future trajectory priors.
    "carrying_capacity_multiplier": [1.5, 5.0, 100.0],
    "rate_growth": [0.005, 0.01, 0.04],
    "cubic_growth": [False, True],
    "T_c": [500, 300, 80],
    "s": [0.01, 0.1],
}


# Longview Nuclear: extinction risk estimates were given over 30 years.
_NUCLEAR_EXTINCTION_30YR = [0.001, 0.003, 0.005]  # 0.1%, 0.3%, 0.5%
_NUCLEAR_EXTINCTION_100YR = [
    _convert_cumulative_probability(p, years_from=30, years_to=100)
    for p in _NUCLEAR_EXTINCTION_30YR
]

# Longview Nuclear 4.4 (extinction): 0.2% per $10M, with uncertainty envelope.
_NUCLEAR_REL_REDUCTION_PER_10M = [0.001, 0.002, 0.003]  # 0.1%..0.3%
_NUCLEAR_ABS_RISK_REDUCTION = _abs_reduction_from_per_10m(
    rel_reduction_per_10m=_NUCLEAR_REL_REDUCTION_PER_10M,
    budget=5.7 * M,
    cumulative_risk_100_ref=_NUCLEAR_EXTINCTION_100YR[1],
)


# Longview AI declined 4.3 and 4.4: use RP world priors + explicit assumption.
# Assumption: $1B buys 1% relative risk reduction over effect window.
# Equivalent per-$10M relative reduction: 0.01% (fraction = 0.0001).
_AI_REL_REDUCTION_PER_10M = [0.00015]
_AI_ABS_RISK_REDUCTION = _abs_reduction_from_per_10m(
    rel_reduction_per_10m=_AI_REL_REDUCTION_PER_10M,
    budget=70 * M,
    cumulative_risk_100_ref=0.10,
)


FUND_PROFILES = {
    "sentinel": {
        "display_name": "Sentinel Bio",
        "budget": 7.2 * M,
        "counterfactual_factor": 0.80 * 1.0 + 0.15 * 0.5 + 0.05 * 0.0,  # 0.875
        "p_harm": 0.05, 
        "p_zero": 0.50, 
        "harm_multiplier": 1.0,
        "sweep_params": {
            "r_inf": [1e-8, 1e-7, 1e-3],
            "cumulative_risk_100_yrs": _TOTAL_XRISK_100YR,
            "year_risk_1pct_max": [100, 150, 300],
            "year_max_risk": [5, 15, 50],
            "carrying_capacity_multiplier": [1.5, 2.0, 10.0],
            "rate_growth": [0.005, 0.01, 0.04],
            "cubic_growth": [False, True],
            "T_c": [500, 300, 80],
            "s": [0.01, 0.1],
            "abs_risk_reduction": _SENTINEL_ABS_RISK_REDUCTION,
        },
        "fixed_params": {
            "budget": 7.2 * M,
            "periods_value": [0, 5, 10, 20, 100, 500],
            "T_h": 1e14,
            "year_effect_starts": 0,
            "persistence_effect": 15,
            "initial_value": 8e9,
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
        "sub_extinction_tiers": [
            {
                "tier_name": "100M-1B deaths",
                "project_id": "sentinel_bio_100m_1b",
                "effect_id": "effect_human_lives_sub_ext_100m_1b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.02,
                "expected_deaths": 316e6,  # geomean(100M, 1B)
                "natural_pandemic_discount": 1.0,  # no discount
                # Sweep: (rel_risk_reduction, persistence_years) scenarios
                # derived from bp_reduction_per_bn [2000, 10000, 50000]
                "sweep_rel_rr": [0.00144, 0.0072, 0.036],
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
                "sweep_rel_rr": [0.00144, 0.0072, 0.036],
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
            "abs_risk_reduction": _NUCLEAR_ABS_RISK_REDUCTION,
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
            "abs_risk_reduction": _AI_ABS_RISK_REDUCTION,
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
        },
        "export": {
            "project_id": "longview_ai",
            "near_term_xrisk": True,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
            # Q3.2 (2x ~$140M): "stay approximately the same".
            # Q3.3 (5x ~$350M): "decline by ~75%".
            "diminishing_anchors": [(50/70, 1.0), (190/70, 1.0), (260/70, 0.25)],
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
