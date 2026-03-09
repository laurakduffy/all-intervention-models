"""Uncertainty fitting: wraps rp-distribution-fitting for sparse percentile inputs.

Fits candidate distribution families to sparse percentile data (p10/p50/p90,
optionally p1/p99), selects the lowest fit-error model, and generates
deterministic empirical draws for downstream risk-profile calculations.

Requires: pip install rp-distribution-fitting
  (or: pip install git+https://github.com/CharlesD353/rp-distribution-fitting.git)
"""

import numpy as np
from distributions import fit_all, FitResult, PercentileSpec


def percentiles_to_quantile_spec(pct_dict):
    """Convert {p10: val, p50: val, p90: val, ...} to {0.10: val, 0.50: val, ...}.

    Accepts keys like 'p10', 'p50', 'p90', 'p1', 'p99' or numeric 10, 50, 90.
    """
    spec = {}
    for key, val in pct_dict.items():
        if isinstance(key, str) and key.startswith("p"):
            q = int(key[1:]) / 100.0
        elif isinstance(key, (int, float)):
            q = key / 100.0 if key > 1 else float(key)
        else:
            raise ValueError(f"Unrecognized percentile key: {key}")
        spec[q] = float(val)
    return spec


def fit_best(pct_dict, verbose=False):
    """Fit candidate distributions and return the one with lowest fit error.

    When multiple distributions have near-identical fit errors (within 1e-6),
    prefer the one with the smallest mean/median ratio to avoid heavy-tailed
    distributions producing extreme means from tiny tail differences.

    Args:
        pct_dict: dict like {p10: 500000, p50: 2000000, p90: 8000000}
        verbose: print comparison table

    Returns:
        FitResult from rp-distribution-fitting with lowest error.
    """
    spec = percentiles_to_quantile_spec(pct_dict)
    fits = fit_all(spec)

    if not fits:
        raise ValueError(f"No distributions could be fit to {pct_dict}")

    # Among fits with near-identical error, pick least heavy-tailed
    best_error = fits[0].error
    tied = [f for f in fits if abs(f.error - best_error) < 1e-6]
    if len(tied) > 1:
        def _tail_ratio(f):
            med = f.median()
            if med <= 0:
                return float("inf")
            return abs(f.mean() / med)
        tied.sort(key=_tail_ratio)

    best = tied[0]

    if verbose:
        print(f"  Fitted {len(fits)} distributions. Best: {best.name} "
              f"(error={best.error:.6f})")
        for f in fits:
            print(f"    {f.name}: error={f.error:.6f}, "
                  f"mean={f.mean():.2f}, median={f.median():.2f}")

    return best


def generate_draws(fit, n_samples=10000, bounds_q=(0.0001, 0.9999)):
    """Generate deterministic quantile-spaced draws from a fitted distribution.

    Same approach as rp-distribution-fitting/_generate_samples.
    """
    quantile_points = np.linspace(bounds_q[0], bounds_q[1], n_samples)
    return np.asarray(fit.ppf(quantile_points), dtype=float)


def fit_and_draw(pct_dict, n_samples=10000, verbose=False):
    """Convenience: fit best distribution, then generate empirical draws.

    Returns:
        (FitResult, np.ndarray of draws)
    """
    fit = fit_best(pct_dict, verbose=verbose)
    draws = generate_draws(fit, n_samples=n_samples)
    return fit, draws

