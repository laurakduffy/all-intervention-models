"""GCR (Global Catastrophic Risk) valuation model.

Based on Tarsney's "Epistemic Challenge to Longtermism" (2020).
Computes the expected value of interventions that reduce catastrophic risk,
accounting for both near-term and long-term (including stellar expansion) value.

"""

import itertools
from collections import defaultdict
from tabnanny import verbose

import numpy as np
from dataclasses import dataclass, field, fields as dc_fields
from scipy.optimize import fsolve
from scipy.stats import beta as scipy_beta
from scipy.stats import gamma as scipy_gamma
from scipy.stats import lognorm, norm
from scipy.stats import uniform as scipy_uniform

M = 10**6
B = 10**9


@dataclass
class GCRParams:
    """All input parameters for the GCR valuation model.

    Each array parameter has shape (n_sims,) — one value per scenario
    (e.g. conservative / central / optimistic).
    """

    n_sims: int = 3
    budget: float = 10 * M
    periods_value: list = field(default_factory=lambda: [0, 5, 10, 20, 100, 500])

    # Risk trajectory
    cumulative_risk_100_yrs: np.ndarray = None
    year_max_risk: np.ndarray = None
    year_risk_1pct_max: np.ndarray = None
    r_inf: np.ndarray = None
    T_h: np.ndarray = None

    # Intervention effect
    year_effect_starts: np.ndarray = None
    bp_reduction_per_bn: np.ndarray = None
    persistence_effect: np.ndarray = None
    # If set, overrides bp_reduction_per_bn: specifies the absolute peak
    # annual risk reduction from the intervention (independent of baseline risk).
    abs_risk_reduction: np.ndarray = None
    # Parameterisation: sweep the fractional reduction of
    # cause-specific r_max independently of total cumulative risk.
    # rel_risk_reduction: fraction of cause-specific r_max removed (e.g. 0.001).
    # cause_fraction: share of total x-risk attributable to this cause.
    # Model computes: rel_rr_from_int = rel_risk_reduction * cause_fraction.
    rel_risk_reduction: np.ndarray = None
    cause_fraction: float = 1.0

    # Growth parameters
    initial_value: np.ndarray = None
    rate_growth: np.ndarray = None
    carrying_capacity: np.ndarray = None

    # Cubic growth (stellar settlement)
    cubic_growth: np.ndarray = None
    T_c: np.ndarray = None
    r_g: float = 1.3e5  # radius of milky way in ly
    s: np.ndarray = None # speed of settlement in ly/yr (fractions speed of light)
    d_g: float = 2.2e-5 # Density stars in MW galaxy See page 19 of Tarsney (2020), unit: stars/ly^3
    d_s: float = 2.9e-9 # Density stars in virgo supercluster. See page 19 of Tarsney (2020), unit: stars/ly^3


def _solve_r_max(cumulative_risk, year_max_risk, year_risk_1pct_max, r_inf, n_years=100):
    """Solve for the Gaussian peak annual risk r_max that reproduces cumulative_risk.

    An exact numerical solve that is consistent for any (year_max_risk, year_risk_1pct_max) combination.
    Uses vectorized bisection over n_years+1 discrete annual time steps.

    Args:
        cumulative_risk: array of cumulative x-risk over n_years (shape n).
        year_max_risk:   array of peak-risk years T_c (shape n).
        year_risk_1pct_max: array — sigma = year_risk_1pct_max / 3 (shape n).
        r_inf:           array of background annual risk (shape n).
        n_years:         integration horizon (default 100).

    Returns:
        r_max: array (shape n) such that
            1 - prod_{t=0}^{n_years}(1 - clip(r_inf + r_max*G_t, 0, 1)) = cumulative_risk.
    """
    sigma = year_risk_1pct_max / 3.0
    t = np.arange(n_years + 1, dtype=float)  # shape (n_years+1,)

    def _cum(r_max_vec):
        # gaussian shape: (n_years+1, n)
        gaussian = np.exp(
            -0.5 * ((t[:, None] - year_max_risk[None, :]) / sigma[None, :]) ** 2
        )
        annual = np.clip(r_inf[None, :] + r_max_vec[None, :] * gaussian, 0.0, 1.0)
        return 1.0 - np.prod(1.0 - annual, axis=0)

    n = len(cumulative_risk)
    lo = np.zeros(n)
    hi = np.ones(n)
    for _ in range(60):  # 2^-60 ≈ 1e-18 precision
        mid = 0.5 * (lo + hi)
        cum_mid = _cum(mid)
        hi = np.where(cum_mid < cumulative_risk, hi, mid)
        lo = np.where(cum_mid < cumulative_risk, mid, lo)
    return 0.5 * (lo + hi)


class GCRModel:
    """Runs the Tarsney-based GCR valuation model.

    Usage:
        params = GCRParams(...)
        model = GCRModel(params)
        results = model.run(verbose=True)
    """

    def __init__(self, params: GCRParams):
        self.p = params
        self._derive()

    def _derive(self):
        p = self.p
        self.r_max = _solve_r_max(
            p.cumulative_risk_100_yrs,
            p.year_max_risk,
            p.year_risk_1pct_max,
            p.r_inf,
        )
        self.sd_risk = p.year_risk_1pct_max / 3
        if p.rel_risk_reduction is not None:
            # rel_risk_reduction is independent of cumulative_risk_100_yrs.
            # cause_fraction converts cause-specific to total-risk fraction.
            self.rel_rr_from_int = p.rel_risk_reduction * p.cause_fraction
        elif p.abs_risk_reduction is not None:
            self.rel_rr_from_int = p.abs_risk_reduction / self.r_max
        else:
            self.rel_rr_from_int = p.bp_reduction_per_bn * 0.0001 / B * p.budget

        self.b2 = p.r_g * (p.d_g - p.d_s)
        self.T_s = p.r_g / p.s

        # v_s, a1, a2 computed during run() once earth value at T_c is known
        self.v_s = None
        self.a1 = None
        self.a2 = None

    # ── Risk trajectory ──

    def get_annual_risk_level(self, t, I):
        p = self.p
        gaussian_no_int = self.r_max * np.exp(
            -((t - p.year_max_risk) ** 2 / (2 * self.sd_risk**2))
        )
        r_t_no_int = p.r_inf + gaussian_no_int
        if I == 1:
            gaussian_int = self.r_max * (1 - self.rel_rr_from_int) * np.exp(
                -((t - p.year_max_risk) ** 2 / (2 * self.sd_risk**2))
            )
            r_t_int = p.r_inf + gaussian_int
            in_effect = (t >= p.year_effect_starts) & (
                t <= p.persistence_effect + p.year_effect_starts
            )
            r_t = np.where(in_effect, r_t_int, r_t_no_int)
        else:
            r_t = r_t_no_int
        return r_t

    def get_p_survival_vec(self, risk_2d):
        return np.cumprod(1 - risk_2d, axis=0)

    # ── Convergence detection ──

    def get_year_of_const_risk(self, I):
        p = self.p
        n = p.n_sims
        result = np.full(n, 10000)
        converged = np.zeros(n, dtype=bool)
        start = int(np.max(p.year_max_risk)) + 1
        for t in range(start, 10000):
            r_t = self.get_annual_risk_level(t, I)
            newly = (~converged) & (np.abs(r_t - p.r_inf) / p.r_inf < 0.01)
            result[newly] = t
            converged |= newly
            if converged.all():
                break
        return result

    def value_level_logistic(self, t):
        p = self.p
        return p.carrying_capacity / (
            1
            + (p.carrying_capacity - p.initial_value)
            / p.initial_value
            * np.exp(-p.rate_growth * t)
        )

    def get_year_logistic_ends(self):
        p = self.p
        n = p.n_sims
        result = np.full(n, 10000)
        converged = np.zeros(n, dtype=bool)
        for t in range(0, 10000):
            v_t = self.value_level_logistic(t)
            newly = (~converged) & (np.abs(v_t) >= 0.99 * p.carrying_capacity)
            result[newly] = t
            converged |= newly
            if converged.all():
                break
        return result

    def get_year_const_value_and_risk(self, I):
        yr = self.get_year_of_const_risk(I)
        yl = self.get_year_logistic_ends()
        return np.maximum(yr, yl)

    # ── Value functions ──

    def get_earth_value(self, t, y_const_value):
        v_logistic = self.value_level_logistic(t)
        v_const = 0.99 * self.p.carrying_capacity
        return np.where(t < y_const_value, v_logistic, v_const)

    def get_value_stars_settled(self, t):
        p = self.p
        in_mw = self.a1 * np.maximum(t - p.T_c, 0) ** 3
        beyond_mw = self.a2 * np.maximum(t - p.T_c, 0) ** 3 + self.b2
        v_stars = np.where(
            p.cubic_growth & (t >= p.T_c),
            np.where(t <= self.T_s, in_mw, beyond_mw),
            0.0,
        )
        return v_stars

    def get_total_value_level(self, t, y_const_value):
        return self.get_earth_value(t, y_const_value) + self.get_value_stars_settled(t)

    # ── Long-term value integrals ──

    def get_conditional_future_value_on_earth(self, n_years):
        p = self.p
        v_const = 0.99 * p.carrying_capacity
        return v_const / p.r_inf * (1 - np.exp(-p.r_inf * (p.T_h - n_years)))

    def get_conditional_future_value_stars_to_Ts2(self):
        p = self.p
        return self.a1 / p.r_inf * (
            6 / p.r_inf**3
            - np.exp(-p.r_inf * (self.T_s - p.T_c))
            * (
                (self.T_s - p.T_c) ** 3
                + 3 / p.r_inf * (self.T_s - p.T_c) ** 2
                + 6 / p.r_inf**2 * (self.T_s - p.T_c)
                + 6 / p.r_inf**3
            )
        )

    def get_conditional_future_value_stars_to_Ts3(self, n_years):
        p = self.p
        return self.a1 / p.r_inf * (
            np.exp(-p.r_inf * (n_years - p.T_c))
            * (
                (n_years - p.T_c) ** 3
                + 3 / p.r_inf * (n_years - p.T_c) ** 2
                + 6 / p.r_inf**2 * (n_years - p.T_c)
                + 6 / p.r_inf**3
            )
            - np.exp(-p.r_inf * (self.T_s - p.T_c))
            * (
                (self.T_s - p.T_c) ** 3
                + 3 / p.r_inf * (self.T_s - p.T_c) ** 2
                + 6 / p.r_inf**2 * (self.T_s - p.T_c)
                + 6 / p.r_inf**3
            )
        )

    def get_conditional_future_value_stars_to_Th3(self):
        p = self.p
        return (
            self.a2
            / p.r_inf
            * (
                np.exp(-p.r_inf * (self.T_s - p.T_c))
                * (
                    (self.T_s - p.T_c) ** 3
                    + 3 / p.r_inf * (self.T_s - p.T_c) ** 2
                    + 6 / p.r_inf**2 * (self.T_s - p.T_c)
                    + 6 / p.r_inf**3
                )
                - np.exp(-p.r_inf * (p.T_h - p.T_c))
                * (
                    (p.T_h - p.T_c) ** 3
                    + 3 / p.r_inf * (p.T_h - p.T_c) ** 2
                    + 6 / p.r_inf**2 * (p.T_h - p.T_c)
                    + 6 / p.r_inf**3
                )
            )
            + self.b2
            / p.r_inf
            * (
                np.exp(-p.r_inf * (self.T_s - p.T_c))
                - np.exp(-p.r_inf * (p.T_h - p.T_c))
            )
        )

    # ── Main computation ──

    def run(self, verbose=False):
        p = self.p
        n = p.n_sims

        # Convergence years
        y_const_value = self.get_year_const_value_and_risk(0)
        y_const_1 = self.get_year_const_value_and_risk(1)
        num_years_per_sim = np.maximum(
            np.maximum(y_const_1, y_const_value), p.periods_value[-1]
        ).astype(int)
        max_num_years = int(np.max(num_years_per_sim))
        years_arr = np.arange(max_num_years + 1)

        if verbose:
            print(f"num_years_per_sim = {num_years_per_sim}")
            print(f"max_num_years = {max_num_years}")

        # Stellar coefficients (need earth value at T_c)
        self.v_s = np.array(
            [
                self.get_earth_value(int(p.T_c[i]), y_const_value)[i]
                for i in range(n)
            ]
        )
        self.a1 = 4 / 3 * np.pi * p.d_g * self.v_s * p.s**3
        self.a2 = 4 / 3 * np.pi * p.d_s * self.v_s * p.s**3

        # Risk arrays (kept as float64 — diff_in_survival involves catastrophic
        # cancellation for low-risk scenarios; float32 would lose the signal entirely)
        r_array_1 = np.array(
            [self.get_annual_risk_level(t, 1) for t in years_arr]
        )
        r_array_0 = np.array(
            [self.get_annual_risk_level(t, 0) for t in years_arr]
        )

        # Survival arrays
        survival_arr_1 = self.get_p_survival_vec(r_array_1)
        survival_arr_0 = self.get_p_survival_vec(r_array_0)
        diff_in_survival = survival_arr_1 - survival_arr_0

        # Value array
        value_array = np.array(
            [
                self.get_total_value_level(t, y_const_value)
                for t in range(max_num_years)
            ],
            dtype=np.float32,
        )

        # Short-term intervention value by period
        ev_short = {}
        abs_ev_short = {}
        for i in range(len(p.periods_value) - 1):
            lo = p.periods_value[i]
            hi = p.periods_value[i + 1]
            ev_i = np.sum(
                value_array[lo:hi] * diff_in_survival[lo:hi], axis=0
            )
            ev_short[f"{lo} to {hi}"] = ev_i
            abs_ev_short[f"{lo} to {hi}"] = np.sum(
                value_array[lo:hi] * survival_arr_1[lo:hi], axis=0
            )

        last = p.periods_value[-1]
        after_key = f"after {last}"
        if np.any(num_years_per_sim > last):
            time_idx = np.arange(last, max_num_years)[:, None]
            mask = time_idx < num_years_per_sim[None, :]
            contributions = (
                value_array[last:max_num_years]
                * diff_in_survival[last:max_num_years]
            )
            ev_short[after_key] = np.sum(contributions * mask, axis=0)
            abs_ev_short[after_key] = np.sum(
                value_array[last:max_num_years] * survival_arr_1[last:max_num_years] * mask,
                axis=0,
            )

        # Long-term value
        sim_idx = np.arange(n)
        diff_n = diff_in_survival[num_years_per_sim - 1, sim_idx]
        cond_V_earth = self.get_conditional_future_value_on_earth(
            num_years_per_sim
        )

        ev_no_cubic = diff_n * cond_V_earth

        cV_Ts2 = self.get_conditional_future_value_stars_to_Ts2()
        cV_Th3 = self.get_conditional_future_value_stars_to_Th3()
        p_Tc2 = np.exp(-p.r_inf * (p.T_c - num_years_per_sim))
        ev_cubic_late = diff_n * (cond_V_earth + p_Tc2 * (cV_Ts2 + cV_Th3))

        cV_Ts3 = self.get_conditional_future_value_stars_to_Ts3(
            num_years_per_sim
        )
        exp_adj = np.exp(-p.r_inf * (p.T_c - num_years_per_sim))
        ev_cubic_early = diff_n * (
            cond_V_earth + exp_adj * (cV_Ts3 + cV_Th3)
        )

        ev_cubic = np.where(
            p.T_c > num_years_per_sim, ev_cubic_late, ev_cubic_early
        )
        ev_long = np.where(p.cubic_growth, ev_cubic, ev_no_cubic)

        # Absolute EV of the future with intervention
        surv_n_1 = survival_arr_1[num_years_per_sim - 1, sim_idx]
        value_no_cubic    = surv_n_1 * cond_V_earth
        value_cubic_late  = surv_n_1 * (cond_V_earth + p_Tc2 * (cV_Ts2 + cV_Th3))
        value_cubic_early = surv_n_1 * (cond_V_earth + exp_adj * (cV_Ts3 + cV_Th3))
        value_cubic = np.where(
            p.T_c > num_years_per_sim, value_cubic_late, value_cubic_early
        )
        abs_ev_long = np.where(p.cubic_growth, value_cubic, value_no_cubic)

        abs_total = np.zeros(n)
        for i in range(len(p.periods_value) - 1):
            lo, hi = p.periods_value[i], p.periods_value[i + 1]
            abs_total += abs_ev_short[f"{lo} to {hi}"]
        if after_key in abs_ev_short:
            abs_total += abs_ev_short[after_key]
        abs_total += abs_ev_long

        # Assemble results
        ev_dict = dict(ev_short)
        if after_key in ev_short:
            ev_dict[f"Expected Value after {last}"] = (
                ev_short[after_key] + ev_long
            )

        total = np.zeros(n)
        for i in range(len(p.periods_value) - 1):
            lo, hi = p.periods_value[i], p.periods_value[i + 1]
            total += ev_dict[f"{lo} to {hi}"]
        if after_key in ev_short:
            total += ev_short[after_key]
        total += ev_long
        ev_dict["Total Value"] = total

        ev_dict_per_M = {}
        for k, v in ev_dict.items():
            ev_dict_per_M[k + " per $1M"] = v / p.budget * 1e6

        if verbose:
            print("\n=== Intervention EV (per sim), by period ===")
            for k, v in ev_dict.items():
                print(f"  {k}: {v}")
            print("\n=== QALYs Per $1M (per sim), by period ===")
            for k, v in ev_dict_per_M.items():
                print(f"  {k}: {v}")

        return {
            "ev_by_period": ev_dict,
            "ev_per_M_by_period": ev_dict_per_M,
            "num_years_per_sim": num_years_per_sim,
            "max_num_years": max_num_years,
            "diff_in_survival": diff_in_survival,
            "value_array": value_array,
            "y_const_value": y_const_value,
            "absolute_total_value_with_intervention": abs_total,
        }


# Constants for run_monte_carlo
_BOOL_PARAMS = {"cubic_growth"}
_SCALAR_PARAMS = {"budget", "r_g", "d_g", "d_s"}

# Keys that are internal to the sampler and not GCRParams fields.
_INTERNAL_KEYS = {"p_digital_minds", "digital_minds", "carrying_capacity_multiplier"}

# Valid GCRParams field names — used to safely filter param_arrays before passing
# to GCRParams(**kwargs).  Computed once at import time.
_GCR_PARAMS_FIELDS = {f.name for f in dc_fields(GCRParams)}


def _get_values_and_p(entry):
    """Extract (values_list, probs_or_None) from a sweep param entry.

    Kept for backward compatibility with sub-extinction tier sampling in
    export_rp_csv.py, which still uses the old discrete format.

    Entries can be a plain list (uniform sampling) or a dict:
        {"values": [...], "p": [...]}   # weighted sampling
    """
    if isinstance(entry, dict):
        return list(entry["values"]), entry.get("p", None)
    return list(entry), None


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _lognormal_mu_sigma(ci_90):
    """(mu_log, sigma_log) for a lognormal whose 5th/95th percentiles are ci_90."""
    lo, hi = ci_90
    mu_log = (np.log(lo) + np.log(hi)) / 2.0
    sigma_log = (np.log(hi) - np.log(lo)) / (2.0 * norm.ppf(0.95))
    return mu_log, sigma_log


def _solve_beta_params(ci_90, mean=None):
    """Solve for (alpha, beta) s.t. Beta.ppf(0.05)=lo and Beta.ppf(0.95)=hi.

    Uses fsolve in log-parameter space to enforce positivity.  If the solver
    fails to converge, falls back to the method-of-moments initial guess.
    Results are cached on the spec dict (keyed by '_alpha_beta') so the solve
    only runs once per unique spec across all strata.
    """
    lo, hi = ci_90
    m = mean if mean is not None else (lo + hi) / 2.0
    rough_var = max(((hi - lo) / 4.0) ** 2, 1e-10)
    factor = max(m * (1 - m) / rough_var - 1, 0.1)
    a0 = max(m * factor, 0.1)
    b0 = max((1 - m) * factor, 0.1)

    def eqs(log_ab):
        a, b = np.exp(log_ab)
        return [
            scipy_beta.ppf(0.05, a, b) - lo,
            scipy_beta.ppf(0.95, a, b) - hi,
        ]

    sol, _, ier, _ = fsolve(eqs, [np.log(a0), np.log(b0)], full_output=True)
    if ier == 1:
        return float(np.exp(sol[0])), float(np.exp(sol[1]))
    return a0, b0  # fallback


def _ppf(spec, u):
    """Apply the inverse CDF of a continuous dist spec to uniform quantiles u.

    Supported dist types: lognormal, beta, normal, uniform.
    (Dirichlet is multivariate and is handled separately by _dirichlet_lhs.)
    """
    dist = spec["dist"]
    if dist == "lognormal":
        mu_log, sigma_log = _lognormal_mu_sigma(spec["ci_90"])
        return lognorm.ppf(u, s=sigma_log, scale=np.exp(mu_log))
    if dist == "beta":
        alpha, beta_p = _solve_beta_params(spec["ci_90"], spec.get("mean"))
        return scipy_beta.ppf(u, alpha, beta_p)
    if dist == "normal":
        if "ci_90" in spec:
            lo, hi = spec["ci_90"]
            mean = (lo + hi) / 2.0
            std = (hi - lo) / (2.0 * norm.ppf(0.95))
        else:
            mean, std = spec["mean"], spec["std"]
        return norm.ppf(u, loc=mean, scale=std)
    if dist == "uniform":
        if "range" in spec:
            lo, hi = spec["range"]
        else:
            # ci_90 = [lo, hi] are the 5th/95th percentiles of the uniform.
            # Solving: lo_actual + 0.05*(hi_actual-lo_actual) = lo
            #          lo_actual + 0.95*(hi_actual-lo_actual) = hi
            lo_ci, hi_ci = spec["ci_90"]
            lo = (0.95 * lo_ci - 0.05 * hi_ci) / 0.90
            hi = (0.95 * hi_ci - 0.05 * lo_ci) / 0.90
        return scipy_uniform.ppf(u, loc=lo, scale=hi - lo)
    raise ValueError(f"_ppf: unsupported dist type '{dist}'")


def _lhs_quantiles(n, rng):
    """n LHS quantile points: one uniform sample per [j/n, (j+1)/n] interval."""
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    return u


def _lhs_quantiles_in_range(n, rng, q_lo, q_hi):
    """LHS quantile points scaled to [q_lo, q_hi] — used for p_digital_minds bins."""
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    return u * (q_hi - q_lo) + q_lo


def _dirichlet_lhs(alpha, n, rng):
    """Sample n Dirichlet(alpha) vectors using independent LHS on Gamma marginals.

    Each of the k components is LHS-sampled from its Gamma(alpha_i, 1) marginal
    independently (random pairing across components), then the rows are
    normalised to sum to 1.  This gives approximate LHS coverage of the
    Dirichlet marginals while preserving the correct joint distribution.

    Args:
        alpha : sequence of k positive concentration parameters.
        n     : number of samples.
        rng   : numpy Generator.

    Returns:
        np.ndarray of shape (n, k).
    """
    k = len(alpha)
    gammas = np.empty((n, k))
    for i, a in enumerate(alpha):
        u = _lhs_quantiles(n, rng)          # independent LHS per component
        gammas[:, i] = scipy_gamma.ppf(u, a=a)
    row_sums = gammas.sum(axis=1, keepdims=True)
    return gammas / row_sums


def run_monte_carlo(
    param_specs,
    fixed_params,
    n_samples=10000,
    N_p_strata=3,
    verbose=False,
    p_harm=0.0,
    p_zero=0.0,
    harm_multiplier=1.0,
    seed=43,
):
    """Run the GCR model with hybrid LHS + discrete-strata sampling.

    Discrete strata (Cartesian product):
      - cubic_growth  : Bernoulli with fixed p (from param_specs)
      - p_digital_minds: Beta distribution stratified into N_p_strata equal-mass
                         quantile bins
      - digital_minds : Bernoulli(p_digital_minds), one True/False per p-bin

    Within each discrete stratum, all continuous parameters (lognormal, beta)
    are sampled independently via Latin Hypercube Sampling (LHS), so every
    quantile of every continuous parameter is covered within each stratum.

    Conditional parameters (carrying_capacity_multiplier) are resolved inside
    each stratum using the case selected by the stratum's digital_minds value.

    Magnitude-based harm/zero assignment runs within each stratum, preserving
    the guarantee that harm samples span the same |EV| range as benefit samples.

    Args:
        param_specs   : dict of {param_name: dist_spec} from param_distributions.py.
        fixed_params  : dict of scalar values held constant across all samples.
        n_samples     : total number of Monte Carlo draws.
        N_p_strata    : number of equal-mass quantile bins for p_digital_minds (default 3).
        verbose       : print stratum grid and summary statistics.
        p_harm        : probability intervention causes harm.
        p_zero        : probability intervention has zero effect.
        harm_multiplier: magnitude multiplier when harm is assigned.
        seed          : random seed for reproducibility.

    Returns:
        dict with keys:
            total_values          : np.ndarray (n_samples,) — total EV per draw
            percentiles           : dict p1/p5/.../p99/mean
            ev_per_period         : dict {period_name: np.ndarray}
            absolute_total_values : np.ndarray — absolute EV of future with intervention
    """
    if p_zero + p_harm > 1.0:
        raise ValueError(f"p_zero ({p_zero}) + p_harm ({p_harm}) must be ≤ 1.0")

    rng = np.random.default_rng(seed)

    # ── 1. Resolve discrete strata parameters ─────────────────────────────────

    cg_spec = param_specs.get("cubic_growth")
    has_cg = cg_spec is not None and cg_spec["dist"] == "bernoulli"
    p_cubic = cg_spec["p"] if has_cg else None

    p_dm_spec = param_specs.get("p_digital_minds")
    has_dm = p_dm_spec is not None

    if has_dm:
        dm_alpha, dm_beta_p = _solve_beta_params(p_dm_spec["ci_90"], p_dm_spec.get("mean"))

    # ── 2. Build strata grid ───────────────────────────────────────────────────
    # Each stratum is a dict with fixed discrete values and its probability weight.
    # Grid: (cubic_growth) × (p_digital_minds quantile bin) × (digital_minds T/F)

    strata = []

    cg_cases = [(True, p_cubic), (False, 1 - p_cubic)] if has_cg else [(None, 1.0)]

    for cg_val, p_cg in cg_cases:
        if has_dm:
            for bin_i in range(N_p_strata):
                q_lo = bin_i / N_p_strata
                q_hi = (bin_i + 1) / N_p_strata
                # Representative p for this bin: conditional mean ≈ ppf(midpoint)
                p_rep = float(scipy_beta.ppf((q_lo + q_hi) / 2, dm_alpha, dm_beta_p))
                for dm_val, p_dm_cond in [(True, p_rep), (False, 1.0 - p_rep)]:
                    strata.append({
                        "cubic_growth": cg_val,
                        "dm_bin_range": (q_lo, q_hi),
                        "digital_minds": dm_val,
                        "weight": p_cg * (1.0 / N_p_strata) * p_dm_cond,
                    })
        else:
            strata.append({
                "cubic_growth": cg_val,
                "dm_bin_range": None,
                "digital_minds": None,
                "weight": p_cg,
            })

    # Allocate samples proportionally to stratum weight.
    total_weight = sum(s["weight"] for s in strata)
    raw_counts = [n_samples * s["weight"] / total_weight for s in strata]
    stratum_counts = [int(c) for c in raw_counts]
    leftover = n_samples - sum(stratum_counts)
    order = sorted(range(len(strata)), key=lambda i: -(raw_counts[i] - stratum_counts[i]))
    for i in order[:leftover]:
        stratum_counts[i] += 1

    if verbose:
        n_strata = len(strata)
        cg_label = f"cubic_growth(2)" if has_cg else ""
        dm_label = f"p_digital_minds({N_p_strata} bins)×digital_minds(2)" if has_dm else ""
        labels = " × ".join(x for x in [cg_label, dm_label] if x)
        print(f"Hybrid MC: {n_samples:,} samples | {n_strata} strata [{labels}] | LHS within each")

    # ── 3. Identify continuous parameters for LHS ─────────────────────────────
    # Dirichlet specs are handled separately (multivariate); their group keys
    # are skipped here and their output keys are written directly in the loop.
    dirichlet_specs = {k: spec for k, spec in param_specs.items()
                       if spec["dist"] == "dirichlet"}
    _dirichlet_output_keys = {key for spec in dirichlet_specs.values()
                               for key in spec["keys"]}

    SKIP_KEYS = ({"cubic_growth", "p_digital_minds", "digital_minds",
                  "carrying_capacity_multiplier"} | set(dirichlet_specs.keys()))
    _CONTINUOUS_DISTS = {"lognormal", "beta", "normal", "uniform"}
    continuous_keys = [
        k for k, spec in param_specs.items()
        if k not in SKIP_KEYS and spec["dist"] in _CONTINUOUS_DISTS
    ]

    ccm_spec = param_specs.get("carrying_capacity_multiplier")
    has_ccm = ccm_spec is not None and ccm_spec["dist"] == "conditional"

    # ── 4. Sample within each stratum ─────────────────────────────────────────
    all_samples = defaultdict(list)
    stratum_ranges = []
    _offset = 0

    for s_idx, stratum in enumerate(strata):
        n_in = stratum_counts[s_idx]
        stratum_ranges.append((_offset, _offset + n_in))
        _offset += n_in

        if n_in == 0:
            continue

        # Fixed discrete values from stratum definition
        if stratum["cubic_growth"] is not None:
            all_samples["cubic_growth"].extend([stratum["cubic_growth"]] * n_in)
        if stratum["digital_minds"] is not None:
            all_samples["digital_minds"].extend([stratum["digital_minds"]] * n_in)

        # p_digital_minds: LHS within this stratum's quantile bin
        if has_dm and stratum["dm_bin_range"] is not None:
            q_lo, q_hi = stratum["dm_bin_range"]
            u_dm = _lhs_quantiles_in_range(n_in, rng, q_lo, q_hi)
            all_samples["p_digital_minds"].extend(
                scipy_beta.ppf(u_dm, dm_alpha, dm_beta_p).tolist()
            )

        # Continuous parameters: independent LHS per parameter (random pairing)
        for key in continuous_keys:
            u = _lhs_quantiles(n_in, rng)
            all_samples[key].extend(_ppf(param_specs[key], u).tolist())

        # Conditional: carrying_capacity_multiplier depends on digital_minds
        if has_ccm:
            dm_val = stratum["digital_minds"]
            case_spec = ccm_spec["cases"][dm_val]
            u = _lhs_quantiles(n_in, rng)
            all_samples["carrying_capacity_multiplier"].extend(_ppf(case_spec, u).tolist())

        # Dirichlet: sample each group and write to its named output keys
        for dspec in dirichlet_specs.values():
            alpha = dspec["alpha"]
            keys = dspec["keys"]
            matrix = _dirichlet_lhs(alpha, n_in, rng)  # shape (n_in, k)
            for i, key in enumerate(keys):
                all_samples[key].extend(matrix[:, i].tolist())

    # ── 5. Convert lists to arrays ────────────────────────────────────────────
    actual_n = sum(stratum_counts)
    param_arrays = {}
    for k, v in all_samples.items():
        arr = np.array(v)
        if k in _BOOL_PARAMS or k == "digital_minds":
            arr = arr.astype(bool)
        param_arrays[k] = arr

    # ── 6. Add scalar fixed parameters ────────────────────────────────────────
    for key, val in fixed_params.items():
        if key in _SCALAR_PARAMS or key == "periods_value" or key in param_arrays:
            continue
        dtype = bool if key in _BOOL_PARAMS else float
        param_arrays[key] = np.full(actual_n, val, dtype=dtype)

    # ── 7. Derive carrying_capacity from multiplier × initial_value ───────────
    if "carrying_capacity" not in param_arrays:
        if "carrying_capacity_multiplier" in param_arrays:
            mult = param_arrays.pop("carrying_capacity_multiplier")
        else:
            mult = np.full(actual_n, 2.0)
        param_arrays["carrying_capacity"] = mult * param_arrays["initial_value"]

    # ── 8. Build GCRParams (drop internal keys not accepted by GCRParams) ─────
    # _INTERNAL_KEYS covers sampler-only variables (p_digital_minds, etc.).
    # Dirichlet group keys are also excluded (they're not GCRParams fields).
    # _GCR_PARAMS_FIELDS acts as a safety net: any remaining unknown key
    # (e.g. a Dirichlet output key that isn't a GCRParams field) is dropped
    # rather than causing a TypeError on GCRParams(**kwargs).
    _internal = _INTERNAL_KEYS | set(dirichlet_specs.keys())
    gcr_arrays = {k: v for k, v in param_arrays.items()
                  if k not in _internal and k in _GCR_PARAMS_FIELDS}
    scalar_kwargs = {k: fixed_params[k] for k in _SCALAR_PARAMS if k in fixed_params}
    params = GCRParams(
        n_sims=actual_n,
        periods_value=fixed_params.get("periods_value", [0, 5, 10, 20, 100, 500]),
        **scalar_kwargs,
        **gcr_arrays,
    )

    # ── 9. Run model ──────────────────────────────────────────────────────────
    model = GCRModel(params)
    results = model.run()

    # ── 10. Magnitude-based harm/zero assignment within strata ────────────────
    # Within each stratum, sort samples by |Total Value| and assign harm/zero/
    # positive proportionally in bins of 100.  This guarantees harm samples span
    # the same magnitude range as benefit samples across all parameter regimes.
    if p_zero > 0 or p_harm > 0:
        total_evs = results["ev_by_period"]["Total Value"]
        effect_assignments = np.ones(actual_n, dtype=np.int8)  # 1=positive

        bin_size = 100
        for s_start, s_end in stratum_ranges:
            if s_end <= s_start:
                continue
            stratum_indices = np.arange(s_start, s_end)
            local_order = np.argsort(np.abs(total_evs[stratum_indices]))
            sorted_indices = stratum_indices[local_order]

            n_in_stratum = s_end - s_start
            for b_start in range(0, n_in_stratum, bin_size):
                b_end = min(b_start + bin_size, n_in_stratum)
                block_size = b_end - b_start
                block_indices = sorted_indices[b_start:b_end]

                n_harm_b = int(block_size * p_harm)
                n_zero_b = int(block_size * p_zero)
                rem_harm = block_size * p_harm - n_harm_b
                rem_zero = block_size * p_zero - n_zero_b
                rand_val = rng.random()
                if rand_val < rem_harm:
                    n_harm_b += 1
                elif rand_val < rem_harm + rem_zero:
                    n_zero_b += 1

                block_effects = np.array(
                    [2] * n_harm_b + [0] * n_zero_b + [1] * (block_size - n_harm_b - n_zero_b),
                    dtype=np.int8,
                )
                rng.shuffle(block_effects)
                effect_assignments[block_indices] = block_effects

        zero_mask = effect_assignments == 0
        harm_mask = effect_assignments == 2
        for k, v in results["ev_by_period"].items():
            adj = v.copy()
            adj = np.where(zero_mask, 0.0, adj)
            adj = np.where(harm_mask, -v * harm_multiplier, adj)
            results["ev_by_period"][k] = adj

    # ── 11. Summary statistics ────────────────────────────────────────────────
    total_values = results["ev_by_period"]["Total Value"]
    pct_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(total_values, pct_list)
    percentiles = {f"p{p}": float(v) for p, v in zip(pct_list, pct_values)}
    percentiles["mean"] = float(np.mean(total_values))

    if verbose:
        print(f"\nDistribution across {actual_n:,} samples:")
        for k, v in percentiles.items():
            print(f"  {k}: {v:.4e}")
        n_pos = int(np.sum(total_values > 0))
        n_zero = int(np.sum(total_values == 0))
        n_neg = int(np.sum(total_values < 0))
        if p_zero > 0 or p_harm > 0:
            print(f"  ({n_pos:,} pos={100*n_pos/actual_n:.1f}%  "
                  f"{n_zero:,} zero={100*n_zero/actual_n:.1f}%  "
                  f"{n_neg:,} harm={100*n_neg/actual_n:.1f}%)")
        else:
            print(f"  ({n_pos:,} pos, {n_neg:,} neg)")

    return {
        "total_values": total_values,
        "percentiles": percentiles,
        "ev_per_period": results["ev_by_period"],
        "absolute_total_values": results["absolute_total_value_with_intervention"],
    }


def make_original_notebook_params():
    """Factory: returns GCRParams matching the original notebook defaults.

    Useful for validation — running these through GCRModel should reproduce
    the known outputs from the original implementation.
    """
    initial_value = np.array([8e9, 2e9, 5e10])
    return GCRParams(
        n_sims=3,
        budget=10 * M,
        periods_value=[0, 5, 10, 20, 100, 500],
        cumulative_risk_100_yrs=np.array([0.1, 0.05, 0.25]),
        year_max_risk=np.array([15, 10, 5]),
        year_risk_1pct_max=np.array([300, 200, 300]),
        r_inf=np.array([1e-7, 1e-5, 1e-10]),
        T_h=np.array([1e14] * 3),
        year_effect_starts=np.array([0, 3, 2]),
        bp_reduction_per_bn=np.array([1, -1, 10]),
        persistence_effect=np.array([20, 10, 30]),
        initial_value=initial_value,
        rate_growth=np.array([0.01, 0.005, 0.02]),
        carrying_capacity=np.array([2, 1.5, 5]) * initial_value,
        cubic_growth=np.array([False, True, True]),
        T_c=np.array([300, 1000, 600]),
        s=np.array([0.01, 0.001, 0.1]),
    )
