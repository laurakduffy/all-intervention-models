# AW Fund Marginal CE: Assumptions Register

Generated: 2026-03-09

## Fund Configuration

- **Project ID**: aw_combined
- **Display name**: Combined AW Funds (Marginal)
- **Annual budget**: $26.5M/year
- **Room for more funding**: $79.3M

## CE Source

- **Model**: Rethink Priorities Cross-Cause Cost-Effectiveness Model (CCM)
- **Source repo**: rethinkpriorities/cross-cause-cost-effectiveness-model-public
- **Unit**: suffering-years averted per $1000 (pre-moral-weight)
- **Samples**: 100000
- **Note**: These are animal suffering-years, not human-equivalent DALYs. The CCM applies moral weight adjustments downstream. For this pipeline we use these values directly as 'animal-DALYs' pending confirmation on which moral weights to apply. Each intervention includes both percentile summaries (for human readability) and downsampled empirical distributions (for direct risk analysis).

## Effect-Level Summary

| Intervention | Species | Recipient | Split | Persistence | Fit | Neutral aDALYs/$1M |
|---|---|---|---|---|---|---|
| chicken_corporate_campaigns | chicken | birds | 51% | 15yr | empirical | 1,205,171 |
| movement_building | multiple | multiple | 23% | 10yr | empirical | 454,078 |
| policy_advocacy_multi_species | multiple | multiple | 12% | 15yr | empirical | 908,156 |
| fish_welfare | carp | fish | 4% | 10yr | empirical | 182,790 |
| shrimp_welfare | shrimp | shrimp | 2% | 10yr | empirical | 2,733,026 |
| wild_animal_welfare | wild | multiple | 2% | 10yr | empirical | 795,928 |
| invertebrate_welfare | bsf | non_shrimp_invertebrates | 1% | 10yr | empirical | 3,143,296 |

## Key Sources

- CE estimates: Rethink Priorities CCM (github.com/rethinkpriorities/cross-cause-cost-effectiveness-model-public),  https://docs.google.com/document/d/1Kuu08LFYpjG-wGzt7_QmBLkFTzsv4FaQHYRQKn9p3A8/edit?usp=sharing
- Chicken, Shrimp, Carp estimates: Laura Duffy direct override
- BSF: CCM bottom-up models
- Wild: Mixture of BSF model and constructed wild mammal model- Policy/Movement: Analyst priors derived from CCM chicken/shrimp baselines
- Fund splits: EA AWF 2024 payout reports (forum.effectivealtruism.org)
- Distribution fitting: rp-distribution-fitting (lowest fit-error selection)

## Caveats

- CCM estimates are pre-moral-weight or sentience-adjustments (animal suffering-years, not human DALYs).
- Interventions do not consider possibility of zero effect or unintended consequences.
- Fund splits are estimated from public payout reports and may not reflect the fund's marginal allocation.
- No time discounting is applied.
