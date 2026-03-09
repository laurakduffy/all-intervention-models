# GiveWell CEA Modeling with Integrated Risk Adjustments

## Overview

This is a modified version of `gw_cea_modeling.py` that **directly integrates risk adjustments** into the simulation workflow. No CSV intermediaries - the simulations flow directly into risk adjustment calculations.

## What's New

### Original `gw_cea_modeling.py`:
1. Generate simulations using squigglepy
2. Create summary statistics (mean, 5th, 95th percentiles)
3. Create histograms

### Modified `gw_cea_modeling_with_risk.py`:
1. Generate simulations using squigglepy ✓ (same)
2. Create summary statistics ✓ (same)
3. Create histograms ✓ (same)
4. **Apply risk adjustments directly to simulations** ✨ (NEW)
5. **Output RP-format CSV with 9 risk profiles** ✨ (NEW)

## Key Features

✅ **No distribution fitting** - uses raw simulations directly
✅ **No CSV intermediaries** - all in memory
✅ **Keeps all original functionality** - summary stats and histograms
✅ **Adds risk-adjusted output** - 9 profiles × 6 time horizons

## Usage

```bash
python gw_cea_modeling_with_risk.py
```

That's it! The script:
1. Runs all simulations
2. Creates summary statistics → `summary_statistics.csv`
3. Creates histograms → `histograms/` directory
4. Applies risk adjustments
5. Outputs risk-adjusted results → `gw_risk_adjusted.csv`

## Output Files

### 1. summary_statistics.csv (original)
Contains mean, 5th percentile, 95th percentile for each effect×horizon combination.

### 2. histograms/ directory (original)
Contains PNG histograms for each effect×horizon combination.

### 3. gw_risk_adjusted.csv (NEW)
Standard RP format with 58 columns:
- 4 metadata columns
- 54 risk-adjusted values (9 profiles × 6 time horizons)

## Risk Profiles Computed

All 9 risk profiles from Rethink Priorities framework:

### Informal
1. **neutral** - Risk-neutral EV (mean)
2. **upside** - Clip at 99th percentile (values above p99 set to p99)
3. **downside** - Loss aversion (λ=2.5, reference=median)
4. **combined** - Percentile-based weight decay (97.5–99.9%) + loss aversion

### Formal (Duffy 2023)
5. **dmreu** - DMREU (p=0.05)
6. **wlu - low** - WLU (c=0.01)
7. **wlu - moderate** - WLU (c=0.05)
8. **wlu - high** - WLU (c=0.1)
9. **ambiguity** - Percentile-based ambiguity aversion

## Data Flow

```
Squigglepy simulations
  ↓
effect_per_M_by_time = {
    'life_years_saved': {
        '0-5 years': np.array([13500, 14200, ...]),  # 10000 samples
        '5-10 years': np.array([1050, 1100, ...]),
        ...
    },
    'YLDs_averted': {...},
    'income_doublings': {...}
}
  ↓
apply_risk_adjustments_to_simulations(effect_per_M_by_time)
  ↓
pandas DataFrame (RP format)
  ↓
gw_risk_adjusted.csv
```

## Example Output

```
======================================================================
GIVEWELL COST-EFFECTIVENESS MODELING WITH RISK ADJUSTMENTS
======================================================================

1. Generating cost-effectiveness simulations...

2. Creating summary statistics...
✓ Saved to: summary_statistics.csv

3. Creating histograms...
✓ Saved to: histograms/ directory

======================================================================
APPLYING RISK ADJUSTMENTS
======================================================================

Processing: life_years_saved
  0-5 years: 10000 samples, mean=13727.57
  5-10 years: 10000 samples, mean=1067.70
  ...

Processing: YLDs_averted
  0-5 years: 10000 samples, mean=3122.34
  ...

Processing: income_doublings
  0-5 years: 10000 samples, mean=2658.01
  ...

✓ Processed 3 effect types
✓ Output: 3 rows × 58 columns

✓ Risk-adjusted results saved to: gw_risk_adjusted.csv

======================================================================
RISK ADJUSTMENT SUMMARY
======================================================================

Life Years Saved (0-5 years):
  neutral        :  13,727.57  ( +0.00%)
  dmreu          :  11,048.64  (-19.51%)
  downside       :  11,818.24  (-13.91%)
  combined       :  11,760.74  (-14.33%)

======================================================================
✓ COMPLETE!
======================================================================
```

## Customization

Edit the risk parameters in the script:

```python
RISK_PARAMS = {
    'dmreu_p': 0.05,              # 0.01=neutral, 0.05=moderate, 0.10=high
    'wlu_low': 0.01,
    'wlu_moderate': 0.05,
    'wlu_high': 0.1,
    'truncation_percentile': 0.99,
    'loss_aversion_lambda': 2.5,   # 1.0=none, 2.5=standard, 5.0=high
}
```

## Dependencies

```bash
pip install numpy pandas scipy matplotlib squigglepy
```

Or with requirements.txt:
```bash
pip install -r requirements.txt
```

Where requirements.txt contains:
```
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0
matplotlib>=3.0.0
squigglepy>=0.26
```

## Comparison: Before vs After

| Feature | Original | Modified |
|---------|----------|----------|
| Simulations | ✓ | ✓ |
| Summary stats | ✓ | ✓ |
| Histograms | ✓ | ✓ |
| Risk adjustments | ✗ | ✓ |
| RP format output | ✗ | ✓ |
| Distribution fitting | N/A | Not needed |
| CSV intermediaries | N/A | Not needed |

## Key Differences from Percentile-Based Approach

### Old Workflow (CSV-based):
```
gw_cea_modeling.py → summary_statistics.csv
                        ↓
             Extract percentiles (p5, mean, p95)
                        ↓
             gw_risk_analysis.py → interpolate samples
                        ↓
             Apply risk adjustments
                        ↓
             gw_risk_adjusted.csv
```

### New Workflow (Integrated):
```
gw_cea_modeling_with_risk.py → simulations in memory
                                  ↓
                        Apply risk adjustments directly
                                  ↓
                        gw_risk_adjusted.csv
```

## Advantages

1. **No information loss** - Uses all 10,000 simulation draws
2. **No distribution fitting** - No interpolation or approximation
3. **Simpler workflow** - One script does everything
4. **Faster** - No CSV I/O overhead
5. **More accurate** - No loss from percentile→samples conversion

## Technical Details

### Sample Structure

The simulations are stored as:
```python
effect_per_M_by_time = {
    'life_years_saved': {
        '0-5 years': np.array([...]),      # N_SAMPLES values
        '5-10 years': np.array([...]),
        '10-20 years': np.array([...]),
        '20-100 years': np.array([...]),
        '100-500 years': np.array([...]),
        '500+ years': np.array([...]),
    },
    'YLDs_averted': {...},
    'income_doublings': {...}
}
```

### Risk Adjustment Function

```python
def compute_all_risk_profiles(samples):
    """
    Takes numpy array of simulation draws.
    Returns dict with 9 risk-adjusted values.
    """
    # 1. Neutral: mean
    # 2. Upside: truncate at p99
    # 3. Downside: loss aversion
    # 4. Combined: percentile-based weight decay (97.5-99.9%) + loss aversion
    # 5. DMREU: probability weighting
    # 6-8. WLU: outcome weighting (3 levels)
    # 9. Ambiguity: percentile-based weighting
    return {
        'neutral': ...,
        'dmreu': ...,
        # etc.
    }
```

### Output Format

RP standard format CSV:
```csv
project_id,near_term_xrisk,effect_id,recipient_type,neutral_t0,neutral_t1,...
givewell,FALSE,life_years_saved,life_years,13727.57,1067.70,...
givewell,FALSE,YLDs_averted,ylds,3122.34,242.85,...
givewell,FALSE,income_doublings,income_doublings,2658.01,206.73,...
```

## Interpreting Results

### Risk Adjustment Impact

Look at the RISK ADJUSTMENT SUMMARY at the end:

```
Life Years Saved (0-5 years):
  neutral        :  13,727.57  ( +0.00%)
  dmreu          :  11,048.64  (-19.51%)
  downside       :  11,818.24  (-13.91%)
  combined       :  11,760.74  (-14.33%)
```

**Interpretation**:
- **19% reduction under DMREU** → Moderate risk aversion significantly reduces value
- **14% reduction under downside** → Loss aversion has strong effect
- **Large gaps** → Distribution has significant tail risk

### Distribution Characteristics

Large gaps between risk profiles indicate:
- ✓ Significant uncertainty in the distribution
- ✓ Heavy right tail (positive skew)
- ✓ Potential for extreme outcomes
- ⚠ Risk-sensitive interventions may rank differently

Small gaps (<5%) would indicate:
- ✓ Low uncertainty
- ✓ Symmetric distribution
- ✓ Robust to risk preferences

## Common Questions

### Q: Can I still use the original script?
**A:** Yes! Both versions coexist. Use original for just simulations and histograms, use modified for risk adjustments.

### Q: Are the simulations identical?
**A:** The simulation logic is identical, so with the same random seed, yes.

### Q: Do I need to change downstream analysis?
**A:** No, the output format is standard RP format, same as before.

### Q: Can I modify the risk parameters?
**A:** Yes, edit `RISK_PARAMS` dictionary in the script.

### Q: How long does it take to run?
**A:** ~2-3 seconds for all simulations + risk adjustments.

## Troubleshooting

### "ModuleNotFoundError: No module named 'squigglepy'"
**Solution**: `pip install squigglepy`

### "No such file or directory: 'histograms/'"
**Solution**: Script creates this automatically, but if permission issues, create manually:
```bash
mkdir histograms
```

### "Values differ from summary_statistics.csv"
**Normal**: Risk-adjusted values will differ from the mean in summary_statistics.csv. That's the point!

## Next Steps

1. **Run the script**: `python gw_cea_modeling_with_risk.py`
2. **Check outputs**: Look at `gw_risk_adjusted.csv`
3. **Compare profiles**: See which risk adjustments matter most
4. **Use downstream**: Feed into your analysis pipeline

## Summary

This integrated version:
- ✅ Keeps all original functionality (simulations, stats, histograms)
- ✅ Adds direct risk adjustment calculations
- ✅ Outputs RP-format CSV
- ✅ No distribution fitting needed
- ✅ No information loss
- ✅ Single script workflow

**Bottom line**: One script, complete workflow, no intermediaries.
