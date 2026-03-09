"""Marginal CE diminishing-returns curves.

Piecewise linear interpolation from anchor points, with 1/x decay beyond
the last anchor. Tracks where marginal CE drops to 20% of initial.

Reuses the same approach as gcr-fund-evaluations/export_rp_csv.py.
"""


def eval_diminishing_raw(budget_m, anchors, spend_m):
    """Evaluate piecewise diminishing-returns curve (un-normalised).

    Args:
        budget_m: fund budget in $M.
        anchors: list of [budget_multiple, ce_multiplier] pairs, sorted by multiple.
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


def compute_diminishing_row(budget_m, anchors, spend_points=None):
    """Return normalised diminishing-returns values for each spend point.

    Normalised so the first spend point = 1.000.
    """
    if spend_points is None:
        spend_points = list(range(10, 901, 10))

    raw = [eval_diminishing_raw(budget_m, anchors, s) for s in spend_points]
    base = raw[0]
    if base <= 0:
        return [0.0] * len(raw), spend_points
    return [v / base for v in raw], spend_points


def find_20pct_threshold(budget_m, anchors, max_spend_m=2000):
    """Find cumulative $M at which marginal CE drops to 20% of initial.

    Scans in $1M increments.  Returns None if threshold not reached
    within max_spend_m.
    """
    initial_ce = eval_diminishing_raw(budget_m, anchors, budget_m)
    target = 0.20 * initial_ce

    for spend_m in range(int(budget_m) + 1, max_spend_m + 1):
        ce = eval_diminishing_raw(budget_m, anchors, spend_m)
        if ce <= target:
            return spend_m
    return None


def years_in_period(persistence, start, end):
    """How many years of [0, persistence] overlap with [start, end).

    Same as gcr-fund-evaluations/export_rp_csv.py._years_in_period.
    """
    if end is None:
        return max(0.0, persistence - start)
    return max(0.0, min(persistence, end) - max(0.0, start))


PERIOD_BOUNDS = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 100),
]

PERIOD_KEYS = ["0_to_5", "5_to_10", "10_to_20", "20_to_100"]


def allocate_to_periods(effect_start_year, persistence_years):
    """Allocate effect across time periods based on start and persistence.

    Returns dict mapping period_key -> fraction of total effect in that period.
    """
    total_active = persistence_years
    if total_active <= 0:
        return {pk: 0.0 for pk in PERIOD_KEYS}

    fractions = {}
    for pk, (t_start, t_end) in zip(PERIOD_KEYS, PERIOD_BOUNDS):
        active_start = max(0, t_start - effect_start_year)
        active_end = t_end - effect_start_year if t_end is not None else persistence_years
        yrs = years_in_period(persistence_years, max(0, active_start), active_end)
        fractions[pk] = yrs / total_active

    return fractions
