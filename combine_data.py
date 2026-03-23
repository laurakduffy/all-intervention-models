## Combine all the data on risk scores and diminishing returns from the fund into a single JSON file
import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

BUDGET_M = 897 # in Millions
INCREMENT_SIZE = 10 # in Millions

_gcr_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcr-models")
sys.path.insert(0, _gcr_models_dir)
from fund_profiles import YEARS_LOOK_AHEAD 
sys.path.pop(0)

DIMINISHING_RETURNS_YRS = 1

if DIMINISHING_RETURNS_YRS != YEARS_LOOK_AHEAD:
    raise ValueError(f"Mismatch between DIMINISHING_RETURNS_YRS ({DIMINISHING_RETURNS_YRS}) and YEARS_LOOK_AHEAD ({YEARS_LOOK_AHEAD}). Please ensure these are aligned.")

# combine the diminishing returns data from the three models into a single dataframe
gw_diminishing_returns = pd.read_csv('gw-models/givewell_diminishing_returns_{}yr.csv'.format(DIMINISHING_RETURNS_YRS))
aw_diminishing_returns = pd.read_csv('aw-models/outputs/aw_combined_diminishing_returns_{}yr.csv'.format(DIMINISHING_RETURNS_YRS))
gcr_diminishing_returns = pd.read_csv('gcr-models/diminishing_returns/gcr_diminishing_returns_{}yr.csv'.format(DIMINISHING_RETURNS_YRS))
leaf_diminishing_returns = pd.read_csv('leaf-models/leaf_diminishing_returns_{}yr.csv'.format(DIMINISHING_RETURNS_YRS))

FUND_NAME_MAP = {
    'sentinel': 'sentinel_bio',
    'longview_nuclear': 'longview_nuclear',
    'longview_ai': 'longview_ai',
    'aw_combined': 'animal_welfare_funds',
    'givewell': 'givewell', 
    'leaf': 'leaf'
}

PROJECT_METADATA = {
    'givewell': {'name': 'Givewell', 'color': "#85E4FF"},
    "sentinel_bio": {"name": "Biorisk fund (Sentinel bio)", "color": "#85E4FF"},
    "longview_nuclear": {"name": "Nuclear fund (Longview)", "color": "#e74c3c"},
    "longview_ai": {"name": "AI fund (Longview)", "color": "#85E4FF"},
    "animal_welfare_funds": {"name": "Animal Welfare fund (combined)", "color": "#85E4FF"},
    "leaf": {'name': "LEAF", "color": "#85E4FF"},
}

RISK_PROFILES = [
    'neutral',         # Maps to 0: Neutral
    'wlu - low',       # Maps to 1: WLU Low (0.01)
    'wlu - moderate',  # Maps to 2: WLU Moderate (0.05)
    'wlu - high',      # Maps to 3: WLU High (0.1)
    'upside',          # Maps to 4: Upside Sceptical
    'downside',        # Maps to 5: Downside Critical
    'combined',        # Maps to 6: Combined
    'ambiguity'        # Maps to 7: Continuous Upside Sceptical 
]

EFFECT_KEY_MAP = {
    'life_years_saved': 'effect_lives_saved',
    'YLDs_averted': 'effect_disability_reduction',
    'income_doublings': 'effect_income',
}

RECIPIENT_TYPE_MAP = {
    'life_years': 'human_life_years',
    'ylds': 'human_ylds',
    'income_doublings': 'human_income_doublings',
    'birds': 'chickens_birds',
}

RECIPIENT_TYPE_OVERRIDES_BY_EFFECT = {
    'movement_building': 'chickens_birds',
    'policy_advocacy_multi_species': 'chickens_birds',
    'wild_animal_welfare': 'non_shrimp_invertebrates',
}

NEAR_TERM_XRISK_OVERRIDES = {
    'sentinel_bio': False,
    'longview_nuclear': False,
}

# Mapping standard time suffixes (t0-t5) to the AW time suffixes (0_to_5, etc.)
TIME_MAPPINGS = [
    ('t0', '0_to_5'),
    ('t1', '5_to_10'),
    ('t2', '10_to_20'),
    ('t3', '20_to_100'),
    ('t4', '100_to_500'), # Default to 0.0 if not found
    ('t5', '500_plus')    # Default to 0.0 if not found
]



diminishing_returns_df = pd.concat([gw_diminishing_returns, \
                                    aw_diminishing_returns, \
                                    gcr_diminishing_returns, \
                                    leaf_diminishing_returns], ignore_index=True)

# upload the risk scores from the three models into a single dataframe
gw_risk_scores = pd.read_csv('gw-models/gw_risk_adjusted.csv')
aw_risk_scores = pd.read_csv('aw-models/outputs/aw_combined_dataset.csv')
gcr_risk_scores = pd.read_csv('gcr-models/gcr_output.csv', skiprows=1)
leaf_risk_scores = pd.read_csv('leaf-models/leaf_risk_adjusted.csv')


risk_scores_df = pd.concat([gw_risk_scores, \
                            aw_risk_scores, \
                            gcr_risk_scores, 
                            leaf_risk_scores], ignore_index=True)

# --- 2. Helper Functions ---

def parse_diminishing_returns(df):
    """Converts the diminishing returns dataframe into a dictionary keyed by mapped project_id."""
    returns_dict = {}
    for _, row in df.iterrows():
        pid = row['project_id']
        if pd.isna(pid): continue
            
        vals = []
        for col in df.columns:
            if col == 'project_id': continue
            
            val = row[col]
            # Handle the GiveWell percentage strings
            if isinstance(val, str) and '%' in val:
                vals.append(float(val.strip('%')) / 100.0)
            elif pd.notna(val):
                vals.append(float(val))
                
        out_pid = FUND_NAME_MAP.get(pid, pid)
        returns_dict[out_pid] = vals
        
    return returns_dict

def parse_effects(df):
    """Converts the risk scores dataframe into the nested JSON structure."""
    effects_dict = {}
    
    for _, row in df.iterrows():
        raw_pid = row.get('project_id')
        if pd.isna(raw_pid): continue
            
        pid = FUND_NAME_MAP.get(raw_pid, raw_pid)
        effect_id = row.get('effect_id')
        recipient_type = row.get('recipient_type')
        recipient_type = RECIPIENT_TYPE_OVERRIDES_BY_EFFECT.get(effect_id, RECIPIENT_TYPE_MAP.get(recipient_type, recipient_type))
        effect_id = EFFECT_KEY_MAP.get(effect_id, effect_id)
        
        near_term = row.get('near_term_xrisk', False)
        if pd.isna(near_term): 
            near_term = False
        elif isinstance(near_term, str): 
            near_term = near_term.strip().upper() == 'TRUE'
        
        if pid not in effects_dict:
            meta = PROJECT_METADATA.get(pid, {"name": pid, "color": "#85E4FF"})
            effects_dict[pid] = {
                "name": meta["name"],
                "color": meta["color"],
                "tags": {"near_term_xrisk": NEAR_TERM_XRISK_OVERRIDES.get(pid, bool(near_term))},
                "diminishing_returns": [], # Will populate in main execution
                "effects": {}
            }
            
        # Build the 6x8 values matrix
        values_matrix = []
        for t_std, t_aw in TIME_MAPPINGS:
            time_row = []
            for rp in RISK_PROFILES:
                col_std = f"{rp}_{t_std}"
                col_aw = f"{rp}_{t_aw}"
                
                val = 0.0
                if col_std in df.columns and pd.notna(row.get(col_std)):
                    val = float(row[col_std]) 
                elif col_aw in df.columns and pd.notna(row.get(col_aw)):
                    val = float(row[col_aw]) 
                    
                time_row.append(val)
            values_matrix.append(time_row)
            
        effects_dict[pid]["effects"][effect_id] = {
            "recipient_type": recipient_type,
            "values": values_matrix
        }
        
    return effects_dict

# --- 3. Main Execution ---

dim_returns_data = parse_diminishing_returns(diminishing_returns_df)
projects_data = parse_effects(risk_scores_df)

for pid in projects_data:
    projects_data[pid]["diminishing_returns"] = dim_returns_data.get(pid, [])

# Merge sub-extinction tier effects into their parent funds, then remove sub-fund projects
_SUB_FUND_MERGES = [
    ('sentinel_bio_100m_1b',     'sentinel_bio'),
    ('sentinel_bio_10m_100m',    'sentinel_bio'),
    ('sentinel_bio_1b_8b',       'sentinel_bio'),
    ('longview_nuclear_100m_1b', 'longview_nuclear'),
    ('longview_nuclear_10m_100m','longview_nuclear'),
    ('longview_nuclear_1b_8b',   'longview_nuclear'),
    ('longview_ai_100m_1b',      'longview_ai'),
    ('longview_ai_10m_100m',     'longview_ai'),
    ('longview_ai_1b_8b',        'longview_ai'),
]
for sub_pid, parent_pid in _SUB_FUND_MERGES:
    if sub_pid in projects_data and parent_pid in projects_data:
        projects_data[parent_pid]['effects'].update(projects_data[sub_pid]['effects'])
        del projects_data[sub_pid]

now = datetime.now()
final_json_structure = {
  "name": now.strftime("%B %#d, %Y at %#I:%M %p"),
  "description": "Updated: " + now.strftime("%B %#d, %Y"),
  "budget": BUDGET_M,
  "incrementSize": INCREMENT_SIZE,
  "moralWeightKeys": [
    {"key": "human_life_years", "label": "Human Life-Years"},
    {"key": "human_ylds", "label": "Human YLDs"},
    {"key": "human_income_doublings", "label": "Human Income Doublings"},
    {"key": "chickens_birds", "label": "Chickens/Birds"},
    {"key": "fish", "label": "Fish"},
    {"key": "shrimp", "label": "Shrimp"},
    {"key": "non_shrimp_invertebrates", "label": "Non-Shrimp Invertebrates"},
    {"key": "mammals", "label": "Mammals"}
  ],
  "discountFactorLabels": [
    "0-5 years", "5-10 years", "10-20 years", "20-100 years", "100-500 years", "500+ years"
  ],
  "riskProfileOptions": [
    {"value": 0, "label": "Neutral"},
    {"value": 1, "label": "WLU Low (0.01)"},
    {"value": 2, "label": "WLU Moderate (0.05)"},
    {"value": 3, "label": "WLU High (0.1)"},
    {"value": 4, "label": "Upside Sceptical"},
    {"value": 5, "label": "Downside Critical"},
    {"value": 6, "label": "Combined"},
    {"value": 7, "label": "Continuous Upside Sceptical"}
  ],
  "projects": projects_data
}

# Export the generated dictionary to JSON
with open('output_data_{}yr.json'.format(DIMINISHING_RETURNS_YRS), 'w') as f:
    json.dump(final_json_structure, f, indent=2)

print("Data successfully mapped and exported to output_data_{}yr.json".format(DIMINISHING_RETURNS_YRS))

# Export normalized risk-adjusted data to CSV
time_labels = ['t0', 't1', 't2', 't3', 't4', 't5']
rows = []
for pid, proj in projects_data.items():
    for eid, effect in proj['effects'].items():
        row = {
            'project_id': pid,
            'effect_id': eid,
            'recipient_type': effect['recipient_type'],
            'near_term_xrisk': proj['tags']['near_term_xrisk'],
        }
        for rp_idx, rp in enumerate(RISK_PROFILES):
            for t_idx, t in enumerate(time_labels):
                row[f"{rp}_{t}"] = effect['values'][t_idx][rp_idx]
        rows.append(row)

normalized_df = pd.DataFrame(rows)
normalized_df.to_csv('all_risk_adjusted.csv', index=False)
print("Risk-adjusted data exported to all_risk_adjusted.csv")

