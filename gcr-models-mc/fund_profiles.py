"""Fund-specific parameter profiles for GCR Monte Carlo runs.

Each fund profile now carries `param_specs` (a dict of distribution specs)
rather than `sweep_params` (a dict of discrete scenario lists).
run_monte_carlo in gcr_model.py reads param_specs and uses the hybrid
LHS + discrete-strata sampler.

Distribution specs are imported from param_distributions.py — edit that
file to change any prior without touching model code.
"""

from copy import deepcopy

import numpy as np

from gcr_model import M, _solve_r_max
from param_distributions import (
    AI_REL_REDUCTION_PER_10M_DIST,
    NUCLEAR_REL_REDUCTION_PER_10M_DIST,
    PERSISTENCE_EFFECT_DIST,
    SENTINEL_REL_REDUCTION_PER_10M_DIST,
    TOTAL_XRISK_100YR_DIST,
    WORLD_PRIOR_DISTRIBUTIONS,
)


def _r_max_from_cumulative_risk(
    cumulative_risk_100_yrs,
    year_max_risk=15,
    year_risk_1pct_max=100,
    r_inf=1e-7,
):
    """Exact r_max for a given cumulative risk and Gaussian shape parameters."""
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
# ---------------------------------------------------------------------------
_AI_CAUSE_FRACTION      = 0.9
_NUCLEAR_CAUSE_FRACTION = 0.03
_BIO_CAUSE_FRACTION     = 0.03

_SENTINEL_BUDGET = 7.2 * M
_NUCLEAR_BUDGET  = 5.7 * M
_AI_BUDGET       = 70  * M

_INITIAL_WORLD_VALUE = 8e9
_TIME_HORIZON = 1e14
_PERIODS_VALUE = [0, 5, 10, 20, 100, 500]


def _scale_rel_risk_dist(per_10m_dist, budget):
    """Scale a per-$10M lognormal CI spec to the actual fund budget.

    Multiplies both CI bounds by (budget / $10M), which is exact for
    lognormal because the scale parameter shifts linearly in log-space.
    """
    lo, hi = per_10m_dist["ci_90"]
    scale = budget / (10 * M)
    return {"dist": "lognormal", "ci_90": [lo * scale, hi * scale]}


_SENTINEL_REL_RISK_DIST = _scale_rel_risk_dist(SENTINEL_REL_REDUCTION_PER_10M_DIST, _SENTINEL_BUDGET)
_NUCLEAR_REL_RISK_DIST  = _scale_rel_risk_dist(NUCLEAR_REL_REDUCTION_PER_10M_DIST,  _NUCLEAR_BUDGET)
_AI_REL_RISK_DIST       = _scale_rel_risk_dist(AI_REL_REDUCTION_PER_10M_DIST,       _AI_BUDGET)

# Sub-extinction tiers still use the old discrete format (unchanged from gcr-models).
# They are processed by _compute_sub_extinction_rows in export_rp_csv.py,
# which has its own stratified sampling logic independent of run_monte_carlo.
_SENTINEL_REL_RISK_REDUCTION_DISCRETE = {
    "values": [rel * (_SENTINEL_BUDGET / (10 * M)) for rel in [0.002/10, 0.002, 0.002*10]],
    "p": [0.25, 0.60, 0.15],
}
_NUCLEAR_REL_RISK_REDUCTION_DISCRETE = {
    "values": [rel * (_NUCLEAR_BUDGET / (10 * M)) for rel in [0.002/10, 0.002, 0.002*10]],
    "p": [0.25, 0.60, 0.15],
}
_AI_REL_RISK_REDUCTION_DISCRETE = {
    "values": [rel * (_AI_BUDGET / (10 * M)) for rel in [v / 4 for v in [0.002/10, 0.002, 0.002*10]]],
    "p": [0.25, 0.60, 0.15],
}
_PERSISTENCE_DISCRETE = {"values": [2.5, 10, 22.5, 30], "p": [0.25, 0.3, 0.15, 0.30]}


FUND_PROFILES = {
    "sentinel": {
        "display_name": "Sentinel Bio",
        "budget": _SENTINEL_BUDGET,
        "counterfactual_factor": 0.80 * 1.0 + 0.15 * 0.5 + 0.05 * 0.0,  # 0.875
        "p_harm": 0.05,
        "p_zero": 0.50,
        "harm_multiplier": 1.0,
        "param_specs": {
            **WORLD_PRIOR_DISTRIBUTIONS,
            "cumulative_risk_100_yrs": TOTAL_XRISK_100YR_DIST,
            "rel_risk_reduction": _SENTINEL_REL_RISK_DIST,
            "persistence_effect": PERSISTENCE_EFFECT_DIST,
        },
        "fixed_params": {
            "budget": _SENTINEL_BUDGET,
            "periods_value": _PERIODS_VALUE,
            "T_h": _TIME_HORIZON,
            "year_effect_starts": (3 + 4) / 2,
            "initial_value": _INITIAL_WORLD_VALUE,
            "cause_fraction": _BIO_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "sentinel_bio",
            "near_term_xrisk": False,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
        },
        "sub_extinction_tiers": [
            {
                "tier_name": "100M-1B deaths",
                "project_id": "sentinel_bio_100m_1b",
                "effect_id": "effect_human_lives_sub_ext_100m_1b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.02,
                "expected_deaths": 316e6,
                "discount": 1.0,
                "sweep_rel_rr": _SENTINEL_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "10M-100M deaths",
                "project_id": "sentinel_bio_10m_100m",
                "effect_id": "effect_human_lives_sub_ext_10m_100m",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.30,
                "expected_deaths": 31.6e6,
                "discount": 0.3,
                "sweep_rel_rr": _SENTINEL_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "1B-8B deaths",
                "project_id": "sentinel_bio_1b_8b",
                "effect_id": "effect_human_lives_sub_ext_1b_8b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 0.005,
                "expected_deaths": 2.83e9,
                "discount": 1.0,
                "sweep_rel_rr": _SENTINEL_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
        ],
    },

    "longview_nuclear": {
        "display_name": "Longview Philanthropy Nuclear Weapons Policy Fund",
        "budget": _NUCLEAR_BUDGET,
        "counterfactual_factor": 0.80 * 1.0 + 0.10 * 0.5 + 0.10 * 0.0,  # 0.85
        "p_harm": 0.05,
        "p_zero": 0.50,
        "harm_multiplier": 1.0,
        "param_specs": {
            **WORLD_PRIOR_DISTRIBUTIONS,
            "cumulative_risk_100_yrs": TOTAL_XRISK_100YR_DIST,
            "rel_risk_reduction": _NUCLEAR_REL_RISK_DIST,
            "persistence_effect": PERSISTENCE_EFFECT_DIST,
        },
        "fixed_params": {
            "budget": _NUCLEAR_BUDGET,
            "periods_value": _PERIODS_VALUE,
            "T_h": _TIME_HORIZON,
            "year_effect_starts": 4,
            "initial_value": _INITIAL_WORLD_VALUE,
            "cause_fraction": _NUCLEAR_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "longview_nuclear",
            "near_term_xrisk": False,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
        },
        "sub_extinction_tiers": [
            {
                "tier_name": "100M-1B deaths",
                "project_id": "longview_nuclear_100m_1b",
                "effect_id": "effect_human_lives_sub_ext_100m_1b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 1 - 0.98 ** (10 / 30),
                "expected_deaths": 316e6,
                "discount": 1.0,
                "sweep_rel_rr": _NUCLEAR_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "10M-100M deaths",
                "project_id": "longview_nuclear_10m_100m",
                "effect_id": "effect_human_lives_sub_ext_10m_100m",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 1 - 0.90 ** (10 / 30),
                "expected_deaths": 31.6e6,
                "discount": 1.0,
                "sweep_rel_rr": _NUCLEAR_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "1B-8B deaths",
                "project_id": "longview_nuclear_1b_8b",
                "effect_id": "effect_human_lives_sub_ext_1b_8b",
                "near_term_xrisk": False,
                "recipient_type": "human_life_years",
                "p_10yr": 1 - (1 - 0.01) ** (10 / 30),
                "expected_deaths": 2.83e9,
                "discount": 1.0,
                "sweep_rel_rr": _NUCLEAR_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
        ],
    },

    "longview_ai": {
        "display_name": "Longview Philanthropy AI Program",
        "budget": _AI_BUDGET,
        "counterfactual_factor": 0.60 * 1.0 + 0.25 * 0.5 + 0.15 * 0.0,  # 0.725
        "p_harm": 0.15,
        "p_zero": 0.50,
        "harm_multiplier": 1.0,
        "param_specs": {
            **WORLD_PRIOR_DISTRIBUTIONS,
            "cumulative_risk_100_yrs": TOTAL_XRISK_100YR_DIST,
            "rel_risk_reduction": _AI_REL_RISK_DIST,
            "persistence_effect": PERSISTENCE_EFFECT_DIST,
        },
        "fixed_params": {
            "budget": _AI_BUDGET,
            "periods_value": _PERIODS_VALUE,
            "T_h": _TIME_HORIZON,
            "year_effect_starts": 3,
            "initial_value": _INITIAL_WORLD_VALUE,
            "cause_fraction": _AI_CAUSE_FRACTION,
        },
        "export": {
            "project_id": "longview_ai",
            "near_term_xrisk": True,
            "effect_id": "effect_human_lives_extinction",
            "recipient_type": "human_life_years",
        },
        "sub_extinction_tiers": [
            {
                "tier_name": "100M-1B deaths",
                "project_id": "longview_ai_100m_1b",
                "effect_id": "effect_human_lives_sub_ext_100m_1b",
                "near_term_xrisk": True,
                "recipient_type": "human_life_years",
                "p_10yr": (0.02 * (1 - 0.98 ** (10 / 30))) ** 0.5,
                "expected_deaths": 316e6,
                "discount": 1.0,
                "sweep_rel_rr": _AI_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "10M-100M deaths",
                "project_id": "longview_ai_10m_100m",
                "effect_id": "effect_human_lives_sub_ext_10m_100m",
                "near_term_xrisk": True,
                "recipient_type": "human_life_years",
                "p_10yr": (0.3 * (1 - 0.90 ** (10 / 30))) ** 0.5,
                "expected_deaths": 31.6e6,
                "discount": 1.0,
                "sweep_rel_rr": _AI_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
            {
                "tier_name": "1B-8B deaths",
                "project_id": "longview_ai_1b_8b",
                "effect_id": "effect_human_lives_sub_ext_1b_8b",
                "near_term_xrisk": True,
                "recipient_type": "human_life_years",
                "p_10yr": (0.005 * (1 - (1 - 0.01) ** (10 / 30))) ** 0.5,
                "expected_deaths": 2.83e9,
                "discount": 1.0,
                "sweep_rel_rr": _AI_REL_RISK_REDUCTION_DISCRETE,
                "sweep_persistence": _PERSISTENCE_DISCRETE,
            },
        ],
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
    profile["adjustment_factor"] = profile["counterfactual_factor"]
    return profile


def make_earth_only_profile(profile):
    """Return a profile variant with stellar expansion disabled."""
    out = deepcopy(profile)
    out["param_specs"].pop("cubic_growth", None)
    out["param_specs"].pop("T_c", None)
    out["param_specs"].pop("s", None)
    out["fixed_params"]["cubic_growth"] = False
    out["fixed_params"]["T_c"] = 500
    out["fixed_params"]["s"] = 0.01
    return out
