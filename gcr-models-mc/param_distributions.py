"""Distribution specifications for all GCR model parameters.

Edit this file to adjust priors without touching any model code.
fund_profiles.py imports these specs and passes them to run_monte_carlo.

--- Distribution types ---

  {"dist": "lognormal", "ci_90": [lo, hi]}
      Lognormal whose 5th/95th percentiles are lo and hi.
      mu_log  = (log(lo) + log(hi)) / 2
      sig_log = (log(hi) - log(lo)) / (2 * norm.ppf(0.95))

  {"dist": "beta", "mean": m, "ci_90": [lo, hi]}
      Beta distribution. alpha, beta are solved numerically so that
      ppf(0.05) = lo and ppf(0.95) = hi.  'mean' is informational
      and used only to seed the numerical solver.

  {"dist": "bernoulli", "p": float}
      Fixed-probability Bernoulli.  Used for parameters whose probability
      is itself certain (e.g. cubic_growth).

  {"dist": "normal", "ci_90": [lo, hi]}
      Normal whose 5th/95th percentiles are lo and hi.
      mean = (lo + hi) / 2,  std = (hi - lo) / (2 * norm.ppf(0.95))
      Alternatively: {"dist": "normal", "mean": m, "std": s}

  {"dist": "uniform", "range": [lo, hi]}
      Uniform on [lo, hi].  Hard bounds — lo and hi are the min/max.
      Alternatively: {"dist": "uniform", "ci_90": [lo, hi]} treats lo/hi as
      the 5th/95th percentiles, so the actual range extends slightly beyond.

  {"dist": "dirichlet", "alpha": [a1, ..., ak], "keys": ["p1", ..., "pk"]}
      Dirichlet(alpha) — generates k non-negative values that sum to 1.
      Each component is assigned to the parameter named in "keys".
      Sampled via independent LHS on Gamma marginals, then row-normalised.
      Alternatively: {"dist": "dirichlet", "means": [m1,...], "concentration": c,
      "keys": [...]} sets alpha_i = c * m_i.
      The spec appears under any descriptive key in param_specs; that key itself
      is not a model parameter — only the "keys" entries are written to samples.

  {"dist": "bernoulli_from", "depends_on": key}
      Bernoulli whose p is drawn from another sampled parameter (must lie
      in [0, 1]).  Used for digital_minds, which depends on p_digital_minds.

  {"dist": "conditional", "depends_on": key, "cases": {val: spec, ...}}
      Different distribution depending on the value of another parameter.
      Used for carrying_capacity_multiplier, which depends on digital_minds.

--- Stratification ---

run_monte_carlo uses a hybrid approach:
  - Discrete parameters (bernoulli, bernoulli_from) form a Cartesian-product
    stratum grid.  p_digital_minds is additionally stratified into N_p_strata
    equal-probability quantile bins (default N=3).
  - All continuous parameters (lognormal, beta) are sampled via Latin
    Hypercube Sampling (LHS) independently within each discrete stratum.
  - Conditional parameters are LHS-sampled using the case selected by the
    stratum's discrete value.
"""

# ---------------------------------------------------------------------------
# World-level priors (shared across all funds)
# ---------------------------------------------------------------------------

WORLD_PRIOR_DISTRIBUTIONS = {

    # ── Risk shape ──────────────────────────────────────────────────────────

    "r_inf": {
        "dist": "lognormal",
        "ci_90": [1e-10, 1e-3],
        # Background (floor) annual extinction risk.
        # Previous discrete values: [1e-10, 1e-7, 1e-3]
        # Lognormal median ≈ 3e-7.
    },

    "year_risk_1pct_max": {
        "dist": "lognormal",
        "ci_90": [20, 200],
        # Gaussian width: sigma = year_risk_1pct_max / 3.
        # Previous discrete values: [20, 100, 200]
        # Lognormal median ≈ 63 yr.
    },

    "year_max_risk": {
        "dist": "lognormal",
        "ci_90": [5, 50],
        # Year of peak catastrophic risk.
        # Previous discrete values: [5, 15, 50]
        # Lognormal median ≈ 16 yr  ✓ (close to previous central of 15).
    },

    # ── Digital minds / carrying capacity hierarchy ──────────────────────────
    # Replaces the old discrete carrying_capacity_multiplier: {1.5, 100} at p=[0.9, 0.1].
    # Now modelled as a two-level hierarchy:
    #   1. p_digital_minds (Beta) — uncertain probability that digital minds emerge.
    #   2. digital_minds (Bernoulli) — draw from that probability.
    #   3. carrying_capacity_multiplier (lognormal, conditional on digital_minds).

    "p_digital_minds": {
        "dist": "beta",
        "mean": 0.05,
        "ci_90": [0.01, 0.15],
        # Prior probability that digital minds emerge this century,
        # driving a high carrying capacity.
        # Mean 5%, 90% CI [1%, 15%].
    },

    "digital_minds": {
        "dist": "bernoulli_from",
        "depends_on": "p_digital_minds",
        # Boolean: True if digital minds emerge, False otherwise.
        # Sampled as Bernoulli(p_digital_minds) within each p-stratum.
        # Forms part of the discrete strata grid (see stratification note above).
    },

    "carrying_capacity_multiplier": {
        "dist": "conditional",
        "depends_on": "digital_minds",
        "cases": {
            True: {
                "dist": "lognormal",
                "ci_90": [20, 100],
                # Digital minds: large population/value multiplier.
                # Lognormal median ≈ 45× current world value.
            },
            False: {
                "dist": "lognormal",
                "ci_90": [0.5, 2.5],
                # No digital minds: growth near current levels, or possible decline.
                # Lognormal median ≈ 1.1× (near-current), 90% CI includes shrinkage.
            },
        },
        # Previous: {'values': [1.5, 100.0], 'p': [0.9, 0.1]}
    },

    # ── Future growth ────────────────────────────────────────────────────────

    "rate_growth": {
        "dist": "lognormal",
        "ci_90": [0.01, 0.04],
        # Logistic growth rate for Earth value.
        # Previous discrete values: [0.01, 0.04]
        # Lognormal median ≈ 0.02.
    },

    "cubic_growth": {
        "dist": "bernoulli",
        "p": 0.10,
        # Whether stellar expansion occurs (cubic value growth).
        # Fixed Bernoulli — the probability itself is considered certain here.
        # Previous: {"values": [False, True], "p": [0.90, 0.10]}
        # Forms part of the discrete strata grid.
    },

    "T_c": {
        "dist": "lognormal",
        "ci_90": [80, 500],
        # Year when cubic (stellar) growth begins if cubic_growth=True.
        # Previous: {'values': [500, 300, 80], 'p': [0.6, 0.3, 0.1]}
        # Lognormal median ≈ 200 yr.
    },

    "s": {
        "dist": "lognormal",
        "ci_90": [0.001, 0.1],
        # Speed of stellar settlement (fraction of speed of light, ly/yr).
        # Previous discrete values: [0.001, 0.01, 0.1]
        # Lognormal median ≈ 0.01  ✓ (matches previous central).
    },
}


# ---------------------------------------------------------------------------
# Shared cross-fund parameters
# ---------------------------------------------------------------------------

TOTAL_XRISK_100YR_DIST = {
    "dist": "lognormal",
    "ci_90": [0.05, 0.40],
    # Total extinction risk (all causes) over 100 years.
    # Previous discrete values: [0.05, 0.15, 0.40]
    # Lognormal median ≈ 0.14  ✓ (close to previous central of 0.15).
    # Interpretation: 10% chance below 5%, 10% chance above 40%.
}

PERSISTENCE_EFFECT_DIST = {
    "dist": "lognormal",
    "ci_90": [2.5, 30],
    # Years that the intervention's risk-reduction effect persists.
    # Previous: {values: [2.5, 10, 22.5, 30], p: [0.25, 0.3, 0.15, 0.30]}
    # Lognormal median ≈ 8.7 yr.
    # To adjust: change ci_90; median = sqrt(lo * hi).
}


# ---------------------------------------------------------------------------
# Fund-specific: relative risk reduction per $10M
# ---------------------------------------------------------------------------
# These are the per-$10M specs.  fund_profiles.py scales them by
# (budget / $10M) to produce the per-fund rel_risk_reduction spec.

SENTINEL_REL_REDUCTION_PER_10M_DIST = {
    "dist": "lognormal",
    "ci_90": [0.0002, 0.02],
    # Relative cause-specific risk reduction per $10M for Sentinel Bio.
    # Previous: {values: [0.0002, 0.002, 0.02], p: [0.25, 0.60, 0.15]}
    # Lognormal median ≈ 0.002  ✓ (matches previous central).
}

NUCLEAR_REL_REDUCTION_PER_10M_DIST = {
    "dist": "lognormal",
    "ci_90": [0.0002, 0.02],
    # Same as Sentinel per $10M.
    # Previous: same as Sentinel.
}

AI_REL_REDUCTION_PER_10M_DIST = {
    "dist": "lognormal",
    "ci_90": [0.00005, 0.005],
    # 1/4 of nuclear per $10M (Longview AI is ~10× more funded).
    # Previous: 1/4 × nuclear values: [0.00005, 0.0005, 0.005]
    # Lognormal median ≈ 0.0005  ✓.
}
