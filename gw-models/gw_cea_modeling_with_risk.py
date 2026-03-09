## Generate estimates for the cost-effectiveness of GW's spending in terms of 
## life-years saved, YLDs averted, and income doublings per $1M spent. 
## Modified to include risk adjustment calculations directly.

import numpy as np
import pandas as pd
import squigglepy as sq
import matplotlib.pyplot as plt
import os
import math

UNITS_VALUE_PER_M_PER_X_CASH = 3280
N_SAMPLES = 10000
LIFE_YEARS_PER_LIFE = 60

## overall cost-effectiveness distribution for GW's portfolio, in terms of units value per $1M spent, using GW moral weights. 
below_8x_dist = sq.lognorm(2, 8, lclip=0.5, rclip=16, credibility=90)
between_8x_and_16x_dist = sq.norm(8, 16, lclip=2, rclip=32, credibility=90)
above_16x_dist = sq.lognorm(16, 44, lclip=8, rclip=80, credibility=90)

percent_portfolio_by_costeffectiveness = {
    'below_8x': 0.05, 
    'between_8x_and_16x': 0.68, 
    'above_16x': 0.27}

gw_moral_weights = {
    'YLDs_averted': 2.3,
    'lives_saved': 115.6,
    'income_doublings': 1
}

def summarize_array(arr):
    return {
        'mean': np.mean(arr),
        '5th_percentile': np.percentile(arr, 5),
        '95th_percentile': np.percentile(arr, 95),
    }

def sample_units_value_per_m():
    below_8x_samples = sq.sample(below_8x_dist, N_SAMPLES)
    between_8x_and_16x_samples = sq.sample(between_8x_and_16x_dist, N_SAMPLES)
    above_16x_samples = sq.sample(above_16x_dist, N_SAMPLES)

    sample_multiples_of_cash = percent_portfolio_by_costeffectiveness['below_8x'] * below_8x_samples + \
        percent_portfolio_by_costeffectiveness['between_8x_and_16x'] * between_8x_and_16x_samples + \
        percent_portfolio_by_costeffectiveness['above_16x'] * above_16x_samples
    
    sample_units_value_per_M = sample_multiples_of_cash * UNITS_VALUE_PER_M_PER_X_CASH
    
    return sample_units_value_per_M

sample_units_value_per_M = sample_units_value_per_m()
summarize_array(sample_units_value_per_M)


## Estimate the percent of GW's effect that is in the form of life-years saved, YLDs averted, and income doublings.
percent_effect_by_type_dict = {
    'Malaria prevention and treatment': 
        {'YLDs_averted': 0.142, 'lives_saved': 0.583, 'income_doublings': 0.274},
    'Vaccinations': 
        {'YLDs_averted': 0.070, 'lives_saved': 0.707, 'income_doublings': 0.223},
    'Malnutrition treatment': 
        {'YLDs_averted': 0.060, 'lives_saved': 0.782, 'income_doublings': 0.158},
    'Water quality': 
        {'YLDs_averted': 0.028, 'lives_saved': 0.665, 'income_doublings': 0.307},
    'VAS': 
        {'YLDs_averted': 0.164, 'lives_saved': 0.668, 'income_doublings': 0.167},
    'Iron fortification': 
        {'YLDs_averted': 0.580, 'lives_saved': 0.000, 'income_doublings': 0.420},
    'Livelihood programs': 
        {'YLDs_averted': 0.091, 'lives_saved': 0.093, 'income_doublings': 0.816},
    'Family planning': 
        {'YLDs_averted': 0.400, 'lives_saved': 0.200, 'income_doublings': 0.400},
}

percent_funding_by_dist_dict = {
    'Malaria prevention and treatment': 0.38,
    'Vaccinations': 0.12,
    'Malnutrition treatment': 0.09,
    'Water quality': 0.09,
    'VAS': 0.07,
    'Iron fortification': 0.07,
    'Livelihood programs': 0.03,
    'Family planning': 0.02,
}

def get_weighted_average_percent_effect_by_type(percent_effect_by_type_dict, percent_funding_by_dist_dict):
    # make dataframe with percent effect by type and percent funding by type, then calculate weighted average percent effect by type
    percent_effect_by_type_df = pd.DataFrame(percent_effect_by_type_dict).T
    percent_effect_by_type_df['percent_funding'] = percent_effect_by_type_df.index.map(percent_funding_by_dist_dict)

    sum_percent_funding = percent_effect_by_type_df['percent_funding'].sum()
    
    for effect_type in ['YLDs_averted', 'lives_saved', 'income_doublings']:
        percent_effect_by_type_df[effect_type] = percent_effect_by_type_df[effect_type] * percent_effect_by_type_df['percent_funding']/sum_percent_funding
    weighted_average_percent_effect_by_type = percent_effect_by_type_df[['YLDs_averted', 'lives_saved', 'income_doublings']].sum()
    return weighted_average_percent_effect_by_type

def get_sample_units_value_by_type(sample_units_value_per_M, weighted_average_percent_effect_by_type, to_print=False):
    sample_effect_by_type = pd.DataFrame({
        'YLDs_averted': sample_units_value_per_M * weighted_average_percent_effect_by_type['YLDs_averted'],
        'lives_saved': sample_units_value_per_M * weighted_average_percent_effect_by_type['lives_saved'],
        'income_doublings': sample_units_value_per_M * weighted_average_percent_effect_by_type['income_doublings'],
    })
    if to_print: 
        print('YLDs_averted avg: {}'.format(np.mean(sample_effect_by_type['YLDs_averted'])))
        print('lives_saved avg: {}'.format(np.mean(sample_effect_by_type['lives_saved'])))
        print('income_doublings avg: {}'.format(np.mean(sample_effect_by_type['income_doublings'])))

    return sample_effect_by_type

def get_distribution_effect_per_M(sample_effect_by_type, to_print=False):
    distribution_effect_per_M = {}
    for effect_type in ['YLDs_averted', 'lives_saved', 'income_doublings']:
        distribution_effect_per_M[effect_type] = sample_effect_by_type[effect_type]/gw_moral_weights[effect_type]

    if to_print:
        for effect_type in ['YLDs_averted', 'lives_saved', 'income_doublings']:
            print('{} avg: {}'.format(effect_type, np.mean(distribution_effect_per_M[effect_type])))

    return distribution_effect_per_M

temporal_breakdown_by_type_dict = {
    'YLDs_averted': 
        {'0-5 years': 0.900, '5-10 years': 0.070, '10-20 years': 0.025, '20-100 years': 0.005, '100-500 years': 0, '500+ years': 0},
    'lives_saved': 
        {'0-5 years': 0.900, '5-10 years': 0.070, '10-20 years': 0.025, '20-100 years': 0.005, '100-500 years': 0, '500+ years': 0},
    'income_doublings': 
        {'0-5 years': 0.180, '5-10 years': 0.014, '10-20 years': 0.125, '20-100 years': 0.681, '100-500 years': 0, '500+ years': 0},
}

def get_effect_per_M_by_time(distribution_effect_by_type, temporal_breakdown_by_type_dict):
    effect_per_M_by_time = {}
    for effect_type in ['YLDs_averted', 'lives_saved', 'income_doublings']:
        effect_per_M_by_time[effect_type] = {}
        for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
            effect_per_M_by_time[effect_type][time_horizon] = distribution_effect_by_type[effect_type] * temporal_breakdown_by_type_dict[effect_type][time_horizon]
    return effect_per_M_by_time

def convert_lives_saved_to_life_years_saved(effect_per_M_by_time):
    effect_per_M_by_time['life_years_saved'] = {}
    for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
        effect_per_M_by_time['life_years_saved'][time_horizon] = effect_per_M_by_time['lives_saved'][time_horizon] * LIFE_YEARS_PER_LIFE
    del effect_per_M_by_time['lives_saved']
    return effect_per_M_by_time

def create_summary_statistics(effect_per_M_by_time):
    summary_statistics = {}
    for effect_type in ['YLDs_averted', 'life_years_saved', 'income_doublings']:
        summary_statistics[effect_type] = {}
        for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
            summary_statistics[effect_type][time_horizon] = {
                'mean': np.mean(effect_per_M_by_time[effect_type][time_horizon]),
                '5th_percentile': np.percentile(effect_per_M_by_time[effect_type][time_horizon], 5),
                '95th_percentile': np.percentile(effect_per_M_by_time[effect_type][time_horizon], 95),
            }
    csv_file = 'summary_statistics.csv'
    with open(csv_file, 'w') as f:
        f.write('effect_type,time_horizon,mean,5th_percentile,95th_percentile\n')
        for effect_type in ['YLDs_averted', 'life_years_saved', 'income_doublings']:
            for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
                stats = summary_statistics[effect_type][time_horizon]
                f.write(f'{effect_type},{time_horizon},{stats["mean"]},{stats["5th_percentile"]},{stats["95th_percentile"]}\n')

    return summary_statistics

def create_and_save_histograms(effect_per_M_by_time):
    # Create histograms directory if it doesn't exist
    os.makedirs('histograms', exist_ok=True)
    
    for effect_type in ['YLDs_averted', 'life_years_saved', 'income_doublings']:
        for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
            plt.hist(effect_per_M_by_time[effect_type][time_horizon], bins=30, alpha=0.7, label=f'{effect_type} - {time_horizon}')
            plt.xlabel(f'{effect_type} per $1M')
            plt.ylabel('Frequency')
            plt.title(f'{effect_type} - {time_horizon}')
            plt.legend()
            plt.savefig(f'histograms/{effect_type}_{time_horizon}_histogram.png')
            plt.close()


# ============================================================================
# RISK ADJUSTMENT FUNCTIONS
# ============================================================================

# Risk parameters
RISK_PARAMS = {
    'dmreu_p': 0.05,
    'wlu_low': 0.01,
    'wlu_moderate': 0.05,
    'wlu_high': 0.1,
    'truncation_percentile': 0.99,
    'loss_aversion_lambda': 2.5,
}

def compute_dmreu(samples, p=0.05):
    """Difference-Making Risk-Weighted Expected Utility."""
    if len(samples) == 0 or np.all(samples == 0):
        return 0.0
    
    a = -2.0 / math.log10(p)
    d = np.sort(samples)
    N = len(d)
    P = 1.0 - np.arange(N + 1) / N
    m_P = np.power(P, a)
    weights = m_P[:-1] - m_P[1:]
    return float(np.dot(d, weights))


def compute_wlu(samples, c=0.05):
    """Weighted Linear Utility."""
    if len(samples) == 0 or np.all(samples == 0):
        return 0.0
    
    if c <= 0:
        return float(np.mean(samples))
    
    abs_samples = np.abs(samples)
    powered = np.power(np.clip(abs_samples, 0, 1e15), c)
    w_positive = 1.0 / (1.0 + powered)
    w_negative = 2.0 - 1.0 / (1.0 + powered)
    weights = np.where(samples >= 0, w_positive, w_negative)
    w_mean = np.mean(weights)
    
    if w_mean <= 0:
        return float(np.mean(samples))
    
    w_hat = weights / w_mean
    return float(np.mean(w_hat * samples))


def compute_ambiguity_percentile(samples):
    """Percentile-based ambiguity aversion with exponential decay."""
    if len(samples) == 0 or np.all(samples == 0):
        return 0.0
    
    d = np.sort(samples)
    N = len(d)
    percentiles = np.arange(N) / (N - 1) * 100
    prelim_weights = np.ones(N)
    
    # Decay region: (97.5, 99.9]
    mask_decay = (percentiles > 97.5) & (percentiles <= 99.9)
    if np.any(mask_decay):
        x = percentiles[mask_decay]
        decay_coef = -np.log(100) / 1.5
        prelim_weights[mask_decay] = np.exp(decay_coef * (x - 97.5))
    
    # Zero weight: >99.9
    prelim_weights[percentiles > 99.9] = 0.0
    
    w_sum = np.sum(prelim_weights)
    if w_sum <= 0:
        return float(np.mean(samples))
    
    final_weights = prelim_weights * (N / w_sum)
    return float(np.sum(final_weights * d) / N)


def compute_all_risk_profiles(samples):
    """
    Compute all 9 risk profiles from simulation samples.
    
    Args:
        samples: numpy array of simulation draws
        
    Returns:
        Dictionary with 9 risk-adjusted values
    """
    if len(samples) == 0 or np.all(samples == 0):
        return {k: 0.0 for k in ['neutral', 'upside', 'downside', 'combined',
                                  'dmreu', 'wlu_low', 'wlu_moderate', 'wlu_high', 'ambiguity']}
    
    # Convert to numpy array if needed
    samples = np.array(samples)
    
    # 1. Neutral: Standard expected value
    neutral = float(np.mean(samples))
    
    # 2. Upside: Truncate at 99th percentile
    p99 = np.percentile(samples, RISK_PARAMS['truncation_percentile'] * 100)
    truncated = np.minimum(samples, p99)
    upside = float(np.mean(truncated))
    
    # 3. Downside: Loss aversion
    median = np.median(samples)
    lam = RISK_PARAMS['loss_aversion_lambda']
    utility = np.where(samples >= median, samples - median, lam * (samples - median))
    downside = float(median + np.mean(utility))
    
    # 4. Combined: Percentile-based weight decay (97.5-99.9%) + loss aversion
    outcomes_c = np.sort(samples)
    N_c = len(outcomes_c)
    pcts_c = np.arange(N_c) / max(N_c - 1, 1) * 100
    w_c = np.ones(N_c)
    mask_decay_c = (pcts_c > 97.5) & (pcts_c <= 99.9)
    if np.any(mask_decay_c):
        w_c[mask_decay_c] = np.exp(-np.log(100) / 1.5 * (pcts_c[mask_decay_c] - 97.5))
    w_c[pcts_c > 99.9] = 0.0
    util_c = np.where(outcomes_c >= median, outcomes_c - median, lam * (outcomes_c - median))
    w_sum_c = np.sum(w_c)
    if w_sum_c > 0:
        combined = float(median + np.sum(w_c * (N_c / w_sum_c) * util_c) / N_c)
    else:
        combined = float(median + np.mean(util_c))
    
    # 5. DMREU
    dmreu = compute_dmreu(samples, p=RISK_PARAMS['dmreu_p'])
    
    # 6-8. WLU
    wlu_low = compute_wlu(samples, c=RISK_PARAMS['wlu_low'])
    wlu_moderate = compute_wlu(samples, c=RISK_PARAMS['wlu_moderate'])
    wlu_high = compute_wlu(samples, c=RISK_PARAMS['wlu_high'])
    
    # 9. Ambiguity
    ambiguity = compute_ambiguity_percentile(samples)
    
    return {
        'neutral': neutral,
        'upside': upside,
        'downside': downside,
        'combined': combined,
        'dmreu': dmreu,
        'wlu_low': wlu_low,
        'wlu_moderate': wlu_moderate,
        'wlu_high': wlu_high,
        'ambiguity': ambiguity,
    }


def apply_risk_adjustments_to_simulations(effect_per_M_by_time):
    """
    Apply risk adjustments to the simulation data.
    
    Args:
        effect_per_M_by_time: Dictionary with structure:
            {effect_type: {time_horizon: numpy_array}}
    
    Returns:
        pandas DataFrame in RP standard format
    """
    print("\n" + "=" * 70)
    print("APPLYING RISK ADJUSTMENTS")
    print("=" * 70)
    
    # Time horizon mapping
    time_horizon_map = {
        '0-5 years': 0,
        '5-10 years': 1,
        '10-20 years': 2,
        '20-100 years': 3,
        '100-500 years': 4,
        '500+ years': 5,
    }
    
    # Effect type mapping
    effect_mapping = {
        'life_years_saved': 'life_years',
        'YLDs_averted': 'ylds',
        'income_doublings': 'income_doublings',
    }
    
    # Risk profile names
    risk_profiles = ['neutral', 'upside', 'downside', 'combined', 'dmreu',
                     'wlu_low', 'wlu_moderate', 'wlu_high', 'ambiguity']
    
    results = []
    
    for effect_type in ['life_years_saved', 'YLDs_averted', 'income_doublings']:
        print(f"\nProcessing: {effect_type}")
        
        # Build row for this effect
        row = {
            'project_id': 'givewell',
            'near_term_xrisk': 'FALSE',
            'effect_id': effect_type,
            'recipient_type': effect_mapping[effect_type],
        }
        
        # Process each time horizon
        for time_horizon in ['0-5 years', '5-10 years', '10-20 years', '20-100 years', '100-500 years', '500+ years']:
            t_idx = time_horizon_map[time_horizon]
            samples = effect_per_M_by_time[effect_type][time_horizon]
            
            print(f"  {time_horizon}: {len(samples)} samples, mean={np.mean(samples):.2f}")
            
            # Compute risk profiles
            risk_values = compute_all_risk_profiles(samples)
            
            # Add to row
            for rp in risk_profiles:
                rp_display = rp.replace('_', ' - ') if 'wlu' in rp else rp
                row[f"{rp_display}_t{t_idx}"] = risk_values[rp]
        
        results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure column order
    metadata_cols = ['project_id', 'near_term_xrisk', 'effect_id', 'recipient_type']
    risk_cols = []
    for rp in ['neutral', 'upside', 'downside', 'combined', 'dmreu',
               'wlu - low', 'wlu - moderate', 'wlu - high', 'ambiguity']:
        for t in range(6):
            risk_cols.append(f"{rp}_t{t}")
    
    df = df[metadata_cols + risk_cols]
    
    print(f"\n✓ Processed {len(results)} effect types")
    print(f"✓ Output: {len(df)} rows × {len(df.columns)} columns")
    
    return df


def main():
    print("=" * 70)
    print("GIVEWELL COST-EFFECTIVENESS MODELING WITH RISK ADJUSTMENTS")
    print("=" * 70)
    
    # Generate simulations (original code)
    print("\n1. Generating cost-effectiveness simulations...")
    weighted_average_percent_effect_by_type = get_weighted_average_percent_effect_by_type(
        percent_effect_by_type_dict, percent_funding_by_dist_dict)
    
    sample_units_value_per_M = sample_units_value_per_m()
    sample_effect_by_type = get_sample_units_value_by_type(
        sample_units_value_per_M, weighted_average_percent_effect_by_type, to_print=False)
    
    distribution_effect_by_type = get_distribution_effect_per_M(
        sample_effect_by_type, to_print=False)
    
    effect_per_M_by_time = get_effect_per_M_by_time(
        distribution_effect_by_type, temporal_breakdown_by_type_dict)
    
    effect_per_M_by_time = convert_lives_saved_to_life_years_saved(effect_per_M_by_time)
    
    # Create summary statistics (original code)
    print("\n2. Creating summary statistics...")
    summary_statistics = create_summary_statistics(effect_per_M_by_time)
    print("✓ Saved to: summary_statistics.csv")
    
    # Create histograms (original code)
    print("\n3. Creating histograms...")
    create_and_save_histograms(effect_per_M_by_time)
    print("✓ Saved to: histograms/ directory")
    
    # Apply risk adjustments (NEW)
    risk_adjusted_df = apply_risk_adjustments_to_simulations(effect_per_M_by_time)
    
    # Save risk-adjusted output
    output_file = 'gw_risk_adjusted.csv'
    risk_adjusted_df.to_csv(output_file, index=False)
    print(f"\n✓ Risk-adjusted results saved to: {output_file}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("RISK ADJUSTMENT SUMMARY")
    print("=" * 70)
    print("\nLife Years Saved (0-5 years):")
    for rp in ['neutral', 'dmreu', 'downside', 'combined']:
        rp_col = rp.replace('_', ' - ') if 'wlu' in rp else rp
        value = risk_adjusted_df[risk_adjusted_df['effect_id'] == 'life_years_saved'][f'{rp_col}_t0'].values[0]
        neutral_value = risk_adjusted_df[risk_adjusted_df['effect_id'] == 'life_years_saved']['neutral_t0'].values[0]
        pct_change = ((value - neutral_value) / neutral_value) * 100
        print(f"  {rp:15s}: {value:10,.2f}  ({pct_change:+6.2f}%)")
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print("=" * 70)
    
    return summary_statistics, risk_adjusted_df


if __name__ == "__main__":
    summary_statistics, risk_adjusted_df = main()
