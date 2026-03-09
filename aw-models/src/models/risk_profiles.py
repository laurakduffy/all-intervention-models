"""Risk-adjusted expected value summaries.

Computes risk-neutral EV plus risk-averse variants from empirical draws.
Mirrors the informal adjustments from gcr-fund-evaluations/export_rp_csv.py
and the formal models from rp-distribution-fitting/risk_analysis.py.

Risk profiles:
  Informal adjustments:
    neutral  — risk-neutral EV (mean)
    upside   — truncate upper tail at p99, renormalize
    downside — loss-averse utility (lambda=2.5, reference=median)
    combined — percentile-based weighting + loss aversion (NEW)

  Formal models (Duffy 2023):
    dmreu    — Difference-Making Risk-Weighted EU (p=0.05, moderate aversion)
    wlu      — Weighted Linear Utility (c=0.05, low-moderate concavity)
    ambiguity — Ambiguity Aversion with percentile-based weighting (NEW)
"""

import numpy as np

from risk_analysis import (
    compute_dmreu,
    compute_wlu,
    compute_ambiguity_aversion,
)

TRUNCATION_PERCENTILE = 0.99
LOSS_AVERSION_LAMBDA = 2.5

# Formal model defaults (Duffy 2023, matching GCR fund evaluations).
DMREU_P = 0.05       # thought-experiment probability → exponent a = -2/log10(p)
WLU_L = 0.01 
WLU_M = 0.05         # concavity; 0=neutral, 0.05=low-moderate
WLU_H = 0.1
AMBIGUITY_K = 4.0    # cubic coefficient; 0=neutral, 4=mild (1.5x weight-to-worst)

RISK_PROFILES = [
    "neutral", "upside", "downside", "combined",
    "dmreu", "wlu - low", "wlu - moderate", "wlu - high", "ambiguity",
]

def compute_ambiguity_aversion_new(samples: np.ndarray) -> float:
    """Percentile-Based Weighting for upside skepticism.

    Applies exponential decay to extreme positive outcomes:
    - X in [0, 97.5 percentile]: weight = 1.0
    - X in (97.5, 99.9 percentile]: weight = exp(-ln(100)/1.5 * (percentile - 97.5))
    - X >= 99.9 percentile: weight = 0.0
    
    Final weights normalized to sum to N for weighted mean calculation.
    
    Args:
        samples: np.ndarray of outcome values
        
    Returns:
        Weighted mean (float)
    """
    d = np.sort(samples)  # worst to best
    N = len(d)
    
    # Calculate percentile for each sample (0 to 100 scale)
    percentiles = np.arange(N) / (N - 1) * 100
    
    # Initialize preliminary weights
    prelim_weights = np.ones(N)
    
    # Apply exponential decay for (97.5, 99.9] percentile range
    mask_decay = (percentiles > 97.5) & (percentiles <= 99.9)
    if np.any(mask_decay):
        x = percentiles[mask_decay]
        decay_coef = -np.log(100) / 1.5  # ≈ -3.07
        prelim_weights[mask_decay] = np.exp(decay_coef * (x - 97.5))
    
    # Zero weight for samples above 99.9th percentile
    mask_zero = percentiles > 99.9
    prelim_weights[mask_zero] = 0.0
    
    # Normalize weights so they sum to N (for weighted mean)
    w_sum = np.sum(prelim_weights)
    if w_sum <= 0:
        return float(np.mean(samples))
    
    final_weights = prelim_weights * (N / w_sum)
    
    # Weighted mean: sum(weights * values) / N
    return float(np.sum(final_weights * d) / N)


def compute_combined_new(samples: np.ndarray, reference_point: float, loss_lambda: float) -> float:
    """Loss aversion + percentile-based weighting applied together.

    Combines:
    1. Loss aversion utility (Kahneman-Tversky) around reference point
    2. Percentile-based weighting (downweights extreme outcomes)

    Args:
        samples: np.ndarray of outcome values
        reference_point: Reference point for loss aversion (typically median)
        loss_lambda: Loss aversion coefficient (typically 2.5)
        
    Returns:
        Combined risk-adjusted value (float)
    """
    # Sort samples worst to best
    outcomes = np.sort(samples)
    N = len(outcomes)
    
    # Calculate percentile for each outcome (0 to 100 scale)
    percentiles = np.arange(N) / (N - 1) * 100
    
    # Apply percentile-based weights
    weights = np.ones(N)
    
    # Decay region: (97.5, 99.9]
    mask_decay = (percentiles > 97.5) & (percentiles <= 99.9)
    if np.any(mask_decay):
        x = percentiles[mask_decay]
        decay_coef = -np.log(100) / 1.5
        weights[mask_decay] = np.exp(decay_coef * (x - 97.5))
    
    # Zero weight region: >99.9
    mask_zero = percentiles > 99.9
    weights[mask_zero] = 0.0
    
    # Apply loss aversion utility to each outcome
    def loss_aversion_utility(x, ref, lam):
        gain = x - ref
        return gain if gain >= 0 else lam * gain
    
    utilities = np.array([
        loss_aversion_utility(outcome, reference_point, loss_lambda)
        for outcome in outcomes
    ])
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return float(np.mean(utilities)) + reference_point
    
    final_weights = weights * (N / w_sum)
    
    # Weighted mean of utilities, then add back reference point
    weighted_utility = np.sum(final_weights * utilities) / N
    return float(weighted_utility + reference_point)


def compute_risk_profiles(draws):
    """Compute all risk-adjusted values from an empirical draw array.

    Args:
        draws: 1-D numpy array of outcome values (e.g. animal-DALYs/$1M).

    Returns:
        dict with keys for all 9 risk profiles.
    """
    draws = np.asarray(draws, dtype=float)

    # ── Informal adjustments ──

    neutral = float(np.mean(draws))

    # Upside skepticism: truncate at p99
    trunc_val = np.percentile(draws, TRUNCATION_PERCENTILE * 100)
    upside = float(np.mean(np.minimum(draws, trunc_val)))

    # Downside protection: loss-averse utility around median
    ref = float(np.median(draws))
    gains = draws - ref
    utilities = np.where(gains >= 0, gains, LOSS_AVERSION_LAMBDA * gains)
    downside = float(np.mean(utilities) + ref)

    # Combined: percentile-based weighting + loss aversion (NEW)
    combined = compute_combined_new(draws, ref, LOSS_AVERSION_LAMBDA)

    # ── Formal models (Duffy 2023) ──

    dmreu = compute_dmreu(fit=None, p=DMREU_P, samples=draws)
    wlu_low = compute_wlu(fit=None, c=WLU_L, samples=draws)
    wlu_moderate = compute_wlu(fit=None, c=WLU_M, samples=draws)
    wlu_high = compute_wlu(fit=None, c=WLU_H, samples=draws)
    ambiguity = compute_ambiguity_aversion_new(samples=draws)

    return {
        "neutral": neutral,
        "upside": upside,
        "downside": downside,
        "combined": combined,
        "dmreu": dmreu,
        "wlu - low": wlu_low,
        "wlu - moderate": wlu_moderate,
        "wlu - high": wlu_high,
        "ambiguity": ambiguity,
    }
