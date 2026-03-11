# Animal Welfare Intervention Model: Documentation

This document explains the animal welfare cost-effectiveness model implemented in
[`ccm_extract.py`](ccm_extract.py). It covers how the model works conceptually, the
reasoning behind each intervention's parameters, and what happens to the outputs
downstream.

---

## 1. How the model works

The goal of the model is to produce a **probability distribution** over cost-effectiveness
(animal suffering-years averted per $1,000 spent) for each intervention. This distribution
is then used in a downstream risk-adjustment pipeline rather than summarised to a single
point estimate, which allows the model to reflect genuine uncertainty and to apply
different attitudes toward risk.

### Sampling approach

The model draws **100,000 Monte Carlo samples** (seed 42 for reproducibility) from
probability distributions representing each uncertain parameter. Each sample represents one
possible world: a draw of cost-effectiveness, efficacy, persistence, and so on. Multiplying
these samples together propagates uncertainty through the full causal chain, producing a
distribution over the final cost-effectiveness estimate that reflects joint uncertainty across
all inputs.

Parameters are represented as:

- **Lognormal distributions** for quantities that are strictly positive and right-skewed
  (cost per animal affected, persistence in years, hours of suffering). These are specified
  by a 90% confidence interval `(lo, hi)` and parameterised analytically from the log-space
  midpoint and implied standard deviation.
- **Normal distributions** (with clipping) for quantities that are positive but roughly
  symmetric, such as hours of suffering when the range is not too wide.
- **Beta distributions** for proportions — fractions bounded between 0 and 1 (e.g. percent
  pain reduced, probability of success, funding share).

The unit of the final output is **animal suffering-years averted per $1,000 spent** — a
"pre-moral-weight" quantity. Moral weight adjustments are applied downstream, outside this
script.

### Outputs

For each intervention, `ccm_extract.py` writes:

1. **Percentile summaries** (p1/p5/p10/p50/p90/p95/p99 + mean) — human-readable
   summaries stored in `ccm_intervention_estimates.yaml`.
2. **10,000 downsampled empirical draws** — stored in the same YAML, produced by
   quantile-spacing the 100k samples. These preserve the shape of the distribution for
   downstream use.
3. **Full 100,000 samples** — saved as `data/inputs/samples/ccm_intervention_samples_100k.npz`
   for the highest-accuracy downstream use.
4. **QC outputs** — histograms (PNG) and an extended statistics CSV saved under
   `data/outputs/`.

---

## 2. Interventions

### Chicken corporate campaigns

**Source:** [Laura Duffy's estimates](https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?usp=sharing)

This is a direct estimate of hen-DALYs averted per $1,000, constructed by combining two
external cost-effectiveness estimates using an 80/20 weighting:

- **THL (self-reported, 2025):** ~2 hens/$ from their 2015–2024 average. Modelled as a
  lognormal with 90% CI from 0.2 to 8 hens/$ (mean ~2.4 hens/$).
- **ACE (external evaluation, 2025):** 2 to 44 hens/$ (mean ~14 hens/$). Modelled as a
  lognormal with 90% CI from 2 to 44 hens/$.
- **Weighted average:** 5 hens/$ with a 90% CI of roughly 1 to 15 hens/$.

These were then fed into a Causal model using Welfare Footprint Project data to convert
hens affected to hen-DALYs averted, producing a final estimate of **~1,200 hen-DALYs per
$1,000** (90% CI: 177–3,600). This is close to — but with a slightly lower mean than — the
CCM's built-in estimate of 160–3,630 (mean ~1,500). The prior RP estimate of 9–120
chickens affected per dollar is considered approximately 8× too high.

```python
chicken_dalys_per_1000 = sample_lognorm_ci(177, 3600, lclip=50, rclip=10000, credibility=90)
```

The result is used directly as `chicken_sy_per_1000` with no further transformation.

---

### Shrimp welfare

**Source:** [Analyst estimates](https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?tab=t.0) combining SWP and McKay (2023) data.

This intervention is modelled as a weighted average of two sub-interventions: **humane
slaughter** (estimated to represent ~90% of future shrimp spending) and **sludge removal /
stocking density reduction** (~10%).

#### Humane slaughter

The causal chain: shrimp affected per dollar × years of impact × hours of pain per shrimp
in conventional slaughter × proportion of pain averted.

- **Shrimp/$/year:** SWP estimates 1,400 shrimp/$ affected per year. Modelled as lognormal
  90% CI of 800–2,200 (mean ~1,400).
- **Persistence:** Modelled as lognormal 90% CI of 6–15 years (mean ~10 years, consistent
  with ACE's assumption).
- **Hours of pain per shrimp (conventional slaughter):** McKay (2023) estimates ~2 hours of
  shrimp-DALY-equivalent pain (90% CI: 0.38–7.8). Approximated as a lognormal with 90% CI
  of 0.28–6.4 (preserves a mean of 2.1 hours).
- **Pain reduction:** HSI is assumed to avert ~70% of slaughter pain. Modelled as
  `beta(7, 3)`, giving a 90% CI of ~46%–90%.
- **Spending share:** HSI is assumed to represent 90% of shrimp-related spending going
  forward, given other interventions are in pilot stages. Modelled as `beta(18, 2)`.

```python
shrimp_per_dollar_slaughter   = shrimp_per_dollar_per_yr_slaughter * shrimp_slaughter_persistence
shrimp_dalys_reduced_per_dollar_slaughter = (
    shrimp_per_dollar_slaughter
    * shrimp_dalys_suffering_per_shrimp_conventional_slaughter
    * shrimp_hsi_percent_suffering_reduced
)
```

#### Sludge removal

- **Shrimp affected:** ACE estimates that at a cost of ~$71,342, sludge removal affects
  36M–89M shrimp (mean ~71M). Modelled as lognormal 90% CI of 50M–90M at that fixed cost.
- **Hours of pain (poor water quality):** McKay (2023) estimates ~88 shrimp-DALY-equivalent
  hours from poor water quality. Modelled as lognormal 90% CI of 32–180 hours.
- **Pain reduction:** Sludge removal is assumed to reduce suffering from bad water quality
  by ~50% on average. High uncertainty represented as `beta(4, 4)`.

#### Stocking density reduction

The same number of shrimp are assumed to be affected as in sludge removal (same cost basis).

- **Hours of pain (high stocking density):** McKay (2023) estimates ~90 shrimp-DALY-equivalent
  hours from high stocking density. Modelled as lognormal 90% CI of 50–160 hours.
- **Pain reduction:** Assumed ~20% reduction on average. Modelled as `beta(3, 12)`.

#### Overall shrimp estimate

```python
shrimp_avg_dalys_reduced_per_dollar = (
    shrimp_slaughter_pct_funding * shrimp_dalys_reduced_per_dollar_slaughter
    + (1 - shrimp_slaughter_pct_funding) * shrimp_total_stocking_and_sludge_dalys_reduced_per_dollar
)
shrimp_sy_per_1000 = shrimp_avg_dalys_reduced_per_dollar * 1000
```

---

### Fish welfare (carp proxy)

**Source:** [Analyst estimates](https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?tab=t.0), using FWI cost data and CCM carp parameters.

The causal chain: fish affected per dollar × hours of suffering per fish per culture cycle
÷ hours per year × proportion of suffering averted.

- **Fish/dollar:** FWI's farm program reaches ~7 fish/$, but overall cost-effectiveness is
  ~1 fish/$ when other programs are included. Marginal effectiveness is modelled as a
  lognormal 90% CI of 0.5–15 fish/$ (mean ~4.6 fish/$).
- **Suffering per fish:** A typical farmed fish suffers for approximately ⅙ to ⅓ of a DALY
  per year. Using a 383-day culture cycle, this translates to a normal distribution from
  `24×383/6` to `24×383/3` hours, clipped to a plausible range.
- **Proportion of suffering averted:** Interventions are assumed to avert ~15% of suffering
  on average. Modelled as `beta(3, 17)` (mean = 15%, 90% CI roughly 5%–30%).

```python
carp_dalys_reduced_per_dollar = (
    carp_affected_per_dollar
    * (carp_hours_suffering / HOURS_PER_YEAR)
    * carp_prop_reduced
)
carp_sy_per_1000 = carp_dalys_reduced_per_dollar * 1000
```

---

### Invertebrate welfare (BSF proxy)

**Source:** CCM defaults for a generic black soldier fly (BSF) intervention, with one
modification.

The causal chain: BSF born per year × proportion whose welfare is affected × hours of larval
suffering ÷ hours per year × proportion of suffering averted × probability of policy success
× persistence ÷ cost.

Parameters are taken from the CCM's `DEFAULT_BSF_PARAMS`, with the exception of:

- **Probability of success:** Lowered to 20% (modelled as `beta(4, 16)`) to reflect the
  speculative nature of invertebrate welfare advocacy. The probability of harm is not
  modelled.

The `bsf_success` variable is a binary draw (0 or 1 per sample) indicating whether the
advocacy campaign succeeds. Multiplying the annual impact by this draw means approximately
80% of samples are zero, producing a strongly zero-heavy distribution.

```python
bsf_success = (bsf_prob_success >= np.random.uniform(0, 1, N)).astype(float)
bsf_sy_per_1000 = bsf_annual_averted * bsf_persistence / bsf_cost * 1000
```

---

### Policy advocacy (multi-species)

This is a derived intervention, constructed analytically from the chicken and shrimp
estimates. It is not independently parameterised.

```python
policy_blend = 0.5 * (0.6 * chicken_sy_per_1000 + 0.4 * shrimp_sy_per_1000)
```

Rationale: policy campaigns are modelled as affecting chickens (60%) and shrimp (40%) in
proportion to their relative representation in farmed animal advocacy portfolios, at a 50%
effectiveness discount relative to direct corporate campaigns to reflect the additional
causal distance of policy work. Effects are assumed to begin in year 4 and persist for 15
years.

---

### Movement building

Also derived analytically, as a further-discounted version of the same chicken/shrimp blend:

```python
movement = 0.25 * (0.6 * chicken_sy_per_1000 + 0.4 * shrimp_sy_per_1000)
```

Rationale: movement building is modelled as an indirect multiplier on direct advocacy — at
25% the effectiveness of direct campaigns — reflecting the additional causal steps between
capacity building and animal outcomes. Effects start in year 4 and persist for 10 years.

---

### Wild animal welfare

**Source:** [Analyst estimates](https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?usp=sharing), no CCM model available.

Wild animal welfare spending is modelled as a mixture of two sub-interventions — wild
mammals and wild invertebrates — in unknown proportions:

```python
wild_sy_per_1000 = wild_share_mammals * wild_mammal_sy_per_1000 + (1 - wild_share_mammals) * wild_invert_sy_per_1000
```

where `wild_share_mammals ~ beta(1, 1)` (uniform, reflecting genuine ignorance about how
future funding will be split between the two areas).

#### Wild invertebrates

Uses the same scale, duration, and suffering parameters as the BSF intervention, but with a
lower probability of success — **10%** (`beta(1, 9)`) — because wild invertebrate
interventions are more speculative than farmed invertebrate ones.

#### Wild mammals (rat contraception BOTEC)

A bottom-up model using rat birth control advocacy in a mid-sized US city as the example
intervention.

- **Target population:** 4,100–56,000 rats per year die from poisoning in the target area
  (lognormal 90% CI), derived from US urban rat density estimates (0.05–0.25 rats/person),
  a 0.1% urban area share, and an estimated 30% of rat deaths from poisoning per year.
- **Suffering per poisoning death:** Welfare Footprint Institute GPT Pain Track estimates
  for rat death from poison sum to ~170 hours of rat-DALY-equivalent pain (90% CI: 60–330
  hours). Modelled as a lognormal.
- **Probability of success:** 20% (`beta(4, 16)`) — probability that lobbying results in a
  rat birth control ordinance.
- **Proportion of poisoning deaths averted:** 50% (`beta(2, 2)`) — birth control reduces
  but does not eliminate poisoning (some private use continues).
- **Impact duration:** Lognormal 90% CI of 5–20 years.
- **Cost:** Lognormal 90% CI of $100k–$10M to lobby for and implement the intervention.

```python
wild_mammal_sy_per_dollar = (
    wild_mammal_target_pop
    * (wild_mammal_suffering_hrs_per_rat / HOURS_PER_YEAR)
    * wild_mammal_percent_deaths_averted_if_success
    * wild_mammal_years_impact
    * wild_mammal_success
) / wild_mammal_cost
```

---

## 3. What happens to the samples downstream

After `ccm_extract.py` runs, the samples are consumed by the main pipeline (`run.py` →
`build_dataset.py` → `effects.py`).

### Step 1: Scale to fund level (`effects.py`)

Each intervention's per-$1,000 samples are scaled in two ways:

1. **Unit conversion:** multiplied by 1,000 to express impact per $1M.
2. **Fund split weighting:** multiplied by the intervention's share of the fund's budget
   (from a fund YAML such as `aw_combined.yaml`). For example, if chicken campaigns receive
   51.2% of the combined fund's marginal budget, each sample is multiplied by 0.512. This
   gives **animal-DALYs per $1M of fund spending** — i.e. the marginal impact of a
   dollar into the fund, attributed to this intervention.

The pipeline prefers the full 100k `.npz` samples for maximum accuracy, falling back to the
10k YAML samples, and finally to percentile-only summaries for any legacy data.

### Step 2: Risk adjustment (`risk_profiles.py`)

Each intervention's scaled sample array is passed to `compute_risk_profiles()`, which
computes **9 risk-adjusted expected values** from the empirical distribution:

| Profile | Description |
|---|---|
| `neutral` | Risk-neutral mean |
| `upside` | Mean after clipping at p99 (discounts extreme optimism) |
| `downside` | Loss-averse utility with λ=2.5 around the median |
| `combined` | Tail weight decay (p97.5–p99.9) + loss aversion |
| `dmreu` | Difference-Making Risk-Weighted EU (Duffy 2023, p=0.05) |
| `wlu - low` | Weighted Linear Utility, low concavity (c=0.01) |
| `wlu - moderate` | Weighted Linear Utility, moderate concavity (c=0.05) |
| `wlu - high` | Weighted Linear Utility, high concavity (c=0.10) |
| `ambiguity` | Ambiguity-averse: exponential weight decay above p97.5, zero above p99.9 |

These profiles span a range from risk-neutral to strongly risk-averse, allowing the
final cost-effectiveness outputs to be presented under multiple ethical frameworks.

### Step 3: Time allocation (`diminishing_returns.py`)

Each risk-adjusted value is further split across time periods using the intervention's
`effect_start_year` and `persistence_years` parameters. This allocates impact to near-term
vs. longer-term budget periods and supports diminishing returns modelling.

### Step 4: Export

The assembled dataset is written to CSV (`{fund_id}_dataset.csv`) alongside an assumptions
summary (`.md`) and a sensitivity report (`.csv`), all under `aw-models/outputs/`.
