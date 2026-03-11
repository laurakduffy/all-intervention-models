# All-Intervention Models

Cost-effectiveness models for three intervention areas used by Rethink Priorities to evaluate charitable giving opportunities. Each model produces risk-adjusted, time-decomposed estimates of impact per $1M, which are combined into a single JSON file for downstream use.

## Repository Structure

```
All-intervention-models/
├── gw-models/          # GiveWell global health portfolio
├── gcr-models/         # Global catastrophic risk funds (Sentinel Bio, Longview Nuclear/AI)
├── aw-models/          # Animal welfare funds (EA AWF, combined estimate)
└── combine_data.py     # Merges all model outputs into output_data.json
```

## Models

### GiveWell (`gw-models/`)

Estimates the cost-effectiveness of GiveWell's grant portfolio in terms of **life-years saved**, **YLDs averted**, and **income doublings** per $1M.

- Models the portfolio as a weighted mixture of cost-effectiveness distributions across eight cause areas (malaria, vaccines, malnutrition, water quality, VAS, iron fortification, livelihoods, family planning)
- Decomposes effects by time horizon using fixed temporal breakdowns per cause type
- Applies risk adjustments directly to 10,000 Monte Carlo simulation draws

**Entry point:** `gw-models/gw_cea_modeling.py`
**Output:** `gw-models/gw_risk_adjusted.csv`

---

### GCR (`gcr-models/`)

Estimates the expected value of reducing existential and catastrophic risk, based on [Tarsney (2020)](https://doi.org/10.1017/S0953820820000060). Models three funds:

| Fund | Cause | Budget |
|---|---|---|
| Sentinel Bio | Biorisk (engineered pandemics) | $7.2M |
| Longview Nuclear | Nuclear weapons policy | $5.7M |
| Longview AI | AI safety | $70M |

- Models a Gaussian "Time of Perils" risk trajectory plus long-run residual risk
- Integrates a logistic civilizational value trajectory, optionally extended with cubic stellar expansion
- Runs 100,000-sample stratified Monte Carlo over world priors (total x-risk, risk timing, growth trajectory) and fund-specific intervention effect sizes
- Applies counterfactual, zero-effect, and harm adjustments per fund
- Sentinel Bio additionally models two sub-extinction tiers (recoverable catastrophes) via a simple EV formula

**Entry point:** `gcr-models/export_rp_csv.py`
**Output:** `gcr-models/rp_output.csv`, `gcr-models/rp_output_diminishing_returns.csv`

---

### Animal Welfare (`aw-models/`)

Estimates the marginal cost-effectiveness of EA animal welfare funds in terms of **animal suffering-years averted** per $1M, prior to moral weight adjustments.

- Loads 100,000 empirical samples per intervention from Rethink Priorities' Cross-Cause Cost-Effectiveness Model (CCM)
- Converts from per-$1000 spent on the intervention to per-$1M spent on the fund, weighted by each intervention's allocation share
- Applies risk adjustments and distributes effects across time periods based on each intervention's persistence
- Covers: chicken corporate campaigns, movement building, policy advocacy, fish welfare, shrimp welfare, wild animal welfare, invertebrate welfare

**Entry point:** `aw-models/run.py --fund aw_combined`
**Output:** `aw-models/outputs/aw_combined_dataset.csv`, `aw-models/outputs/aw_combined_diminishing_returns.csv`

---

## Shared Methodology

All three models produce outputs in a common format: a **6 time period × 9 risk profile** matrix of cost-effectiveness values per $1M, plus a normalised **diminishing returns curve** evaluated at $10M increments.

### Time periods
| Index | Window |
|---|---|
| t0 | 0–5 years |
| t1 | 5–10 years |
| t2 | 10–20 years |
| t3 | 20–100 years |
| t4 | 100–500 years |
| t5 | 500+ years |

### Risk profiles
| Profile | Description |
|---|---|
| `neutral` | Risk-neutral expected value (mean) |
| `upside` | Clip at p99 — values above the 99th percentile are set to p99 |
| `downside` | Loss aversion (λ=2.5) relative to the median |
| `combined` | Percentile-based weight decay (97.5–99.9%) plus loss aversion |
| `wlu - low` | Weighted Linear Utility, concavity c=0.01 |
| `wlu - moderate` | Weighted Linear Utility, concavity c=0.05 |
| `wlu - high` | Weighted Linear Utility, concavity c=0.10 |
| `dmreu` | Difference-Making Risk-Weighted EU, p=0.05 |
| `ambiguity` | Percentile-based ambiguity aversion (exponential decay 97.5–99.9%, zero above 99.9%) |

### Diminishing returns
Piecewise linear interpolation between analyst-specified anchor points (expressed as budget multiples and CE multipliers), with 1/x decay beyond the final anchor. Normalised so the first $10M increment = 1.0.

---

## Combining Outputs

`combine_data.py` reads the CSVs from all three models and merges them into `output_data.json` for use by downstream tools. It maps each model's column naming conventions to a unified structure and assembles the 6×9 values matrices and diminishing returns arrays per fund.

```bash
python combine_data.py
```

**Output:** `output_data.json`

---

## Running the Models

Each model can be run independently. Regenerating all outputs before combining:

```bash
# GiveWell
cd gw-models
python gw_cea_modeling.py

# GCR
cd gcr-models
python export_rp_csv.py --n-samples 100000

# Animal Welfare
cd aw-models
python run.py --fund aw_combined --verbose

# Combine
cd ..
python combine_data.py
```

## Dependencies

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `squigglepy` (GiveWell model)
- `pyyaml` (Animal Welfare model)

Install per-model requirements where provided (`gw-models/requirements_integrated.txt`).
