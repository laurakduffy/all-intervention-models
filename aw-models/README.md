# AW Fund Marginal Cost-Effectiveness Evaluations

Estimates the marginal cost-effectiveness of EA Animal Welfare funds in terms of **animal suffering-years averted per dollar**, using intervention estimates from the Rethink Priorities Cross-Cause Model (CCM) and other modeling. 

## Quick Start

```bash
source ../test_env/bin/activate
cd aw-models
pip install -r requirements.txt
python run.py --fund ea_awf --verbose
```

Run for a specific fund:
```bash
python run.py --fund ea_awf --verbose       # EA Animal Welfare Fund
python run.py --fund aw_combined --verbose   # Combined AW estimate
python run.py --fund navigation_fund         # (template — fill in splits first)
```

## Architecture

```
aw-models/
├── data/inputs/
│   ├── aw_model_intervention_estimates.yaml  # CCM- or otherwise-derived CE percentiles per intervention
│   ├── aw_intervention_models.py       # Script that generated the above from CCM params
│   ├── funds/
│   │   ├── ea_awf.yaml                 # EA AWF splits (estimated from 2024 payouts)
│   │   ├── navigation_fund.yaml        # Navigation Fund (general)
│   │   ├── navigation_fund_cagefree.yaml   # Navigation Fund cage-free sub-portfolio
│   │   ├── navigation_fund_general.yaml    # Navigation Fund general sub-portfolio
│   │   ├── coefficient_giving.yaml     # Template — awaiting Lewis's data
│   │   ├── aw_combined.yaml            # Weighted aggregate
│   │   └── TEMPLATE.yaml               # Instructions for adding a new fund
│   └── README.md                        # Data provenance and status
├── src/
│   ├── models/
│   │   ├── effects.py              # CCM estimates + fund splits → effect rows
│   │   ├── uncertainty.py          # Distribution fitting (wraps rp-distribution-fitting)
│   │   ├── risk_profiles.py        # Risk-adjusted EV summaries
│   │   └── diminishing_returns.py  # Marginal CE scaling curves
│   └── pipeline/
│       ├── build_dataset.py        # Assembles full effect table
│       └── export.py               # CSV/MD output writers
├── tests/                          # Unit and integration tests
├── outputs/                        # Generated outputs (gitignored)
├── run.py                          # CLI entry point
└── requirements.txt
```

## Methodology

1. **CCM intervention estimates**: Pre-computed cost-effectiveness distributions from the Rethink Priorities CCM for invertebrates (farmed and wild), extracted as p1/p5/p10/p50/p90/p95/p99 percentiles of suffering-years averted per $1000. Source parameters are in `aw_intervention_models.py`. Other estimate methods are described in this document here: https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?usp=sharing

2. **Fund budget splits**: Each fund has a YAML file specifying what percentage of its budget goes to each intervention type (chicken campaigns, fish welfare, shrimp, etc.).

3. **Sample-based estimation**: Empirical samples (100k per intervention) are loaded directly from a pre-computed `.npz` file and used without distribution fitting. This preserves the full shape of the CCM output. Parametric fitting from percentiles is retained as a legacy fallback only.

4. **Risk adjustments**: Compute risk-neutral EV plus risk-averse variants:
   - **Upside skepticism**: Clip upper tail at p99 (values above p99 are set to p99)
   - **Downside protection**: Loss-averse utility (lambda=2.5, reference=median)
   - **Combined**: Percentile-based weight decay (97.5–99.9%) plus loss aversion

5. **Diminishing returns**: Piecewise linear scaling curve from fund allocation data, tracking where marginal CE drops to 20% of initial.

6. **Time allocation**: Effects distributed across time periods (0-5, 5-10, 10-20, 20-100 years).

## Adding a New Fund

1. Copy `data/inputs/funds/TEMPLATE.yaml` to `data/inputs/funds/<fund_name>.yaml`
2. Fill in `annual_budget_M` and the intervention `splits` (decimal fractions summing to ~1.0)
3. Run `python run.py --fund <fund_name> --verbose`

Available intervention keys match those in `aw_model_intervention_estimates.yaml`:
- `chicken_corporate_campaigns`, `shrimp_welfare`, `fish_welfare`
- `invertebrate_welfare`, `policy_advocacy_multi_species`
- `movement_building`, `wild_animal_welfare`

## Outputs

- `outputs/aw_combined_dataset.csv` — One row per intervention with CE summaries and risk profiles
- `outputs/aw_combined_assumptions.md` — Assumption register with sources
- `outputs/aw_combined_sensitivity.csv` — One-way sensitivity analysis
- `outputs/aw_combined_diminishing_returns.csv` — Marginal CE scaling curve

## Data Provenance

| Data | Source | Status |
|------|--------|--------|
| Chicken/Shrimp/Fish CE | CCM direct override (Laura Duffy) | Data and Models |
| Invertebrates CE | CCM bottom-up models | Real CCM parameters |
| Policy/Movement/Wild CE | Analyst priors from CCM baselines | Derived estimates |
| EA AWF splits | 2024 payout reports (EA Forum) | Estimated from public data |
| Navigation Fund splits | Jesse | Provided from org |
| Coefficient Giving splits | Awaiting Lewis | Template only |
| Diminishing returns | Placeholder anchors | Needs fund manager input |

## What Still Needs Human Input
- **Uncertainty of intervention success**: Some interventions may fail. We currently treat this by discounting the expected cost-effectiveness to ensure we can fit the distributions, but in the future more sophisticated risk models could be used. 
- **CG splits**: Fill `coefficient_giving.yaml` if Lewis responds
- **Diminishing returns anchors**: Need fund manager input per fund
