# V30 post-freeze scientific evaluation

Generated: 2026-08-26T18:29:11.852711+00:00

## Outcome

- Overall status: **FAIL_ONE_OR_MORE_PREREGISTERED_GATES**.
- confirmation96 convergence: 94/96 (97.917%).
- fixed60 convergence: 60/60 (100.000%).
- global40 convergence: 12/40 (30.000%); this lockbox failed its convergence gate, so its 12-row metrics are descriptive only.

## Selected final model: confirmation96

| Target | RMSE | local NRMSE | fixed-range NRMSE | Spearman | bias (% fixed range) | p90 abs error (% fixed range) |
|---|---:|---:|---:|---:|---:|---:|
| carbon_utilization | 0.0165596 | 2.662% | 3.989% | 0.9929 | 1.299% | 6.211% |
| specific_energy_kwh_per_kg_meoh | 0.738541 | 4.642% | 1.814% | 0.9796 | 0.902% | 3.124% |
| capex_proxy | 0.0138336 | 7.598% | 1.523% | 0.9710 | 1.425% | 2.147% |

## Selected final model: fixed60

| Target | RMSE | local NRMSE | fixed-range NRMSE | Spearman |
|---|---:|---:|---:|---:|
| carbon_utilization | 0.0103162 | 2.411% | 2.485% | 0.9852 |
| specific_energy_kwh_per_kg_meoh | 0.210855 | 2.611% | 0.518% | 0.9528 |
| capex_proxy | 0.0115715 | 5.660% | 1.274% | 0.9913 |

## Front and decision fidelity

- Nondominated-membership Jaccard: 0.4918.
- Dominance-pair accuracy (conservative denominator 24): 41.67%.
- Common-reference hypervolume relative error: 2.184%.
- TOPSIS rank Spearman/top-10 overlap: 0.9872 / 60.0%.
- VIKOR rank Spearman/top-10 overlap: 0.9843 / 60.0%.

## Conformal coverage

| Dataset | Target | n | coverage | exact 95% CI | mean width (% fixed range) |
|---|---|---:|---:|---:|---:|
| fixed60 | carbon_utilization | 60 | 100.00% | [94.04, 100.00]% | 20.56% |
| fixed60 | specific_energy_kwh_per_kg_meoh | 60 | 91.67% | [81.61, 97.24]% | 2.86% |
| fixed60 | capex_proxy | 60 | 100.00% | [94.04, 100.00]% | 7.96% |
| global40 | carbon_utilization | 12 | 75.00% | [42.81, 94.51]% | 6.02% |
| global40 | specific_energy_kwh_per_kg_meoh | 12 | 41.67% | [15.17, 72.33]% | 3.16% |
| global40 | capex_proxy | 12 | 100.00% | [73.54, 100.00]% | 5.31% |
| confirmation96 | carbon_utilization | 94 | 100.00% | [96.15, 100.00]% | 37.34% |
| confirmation96 | specific_energy_kwh_per_kg_meoh | 94 | 96.81% | [90.96, 99.34]% | 9.26% |
| confirmation96 | capex_proxy | 94 | 100.00% | [96.15, 100.00]% | 7.74% |

## Claim boundary

The evidence may support only the exact claims marked supported in `claim_evidence.csv`. The global40 convergence failure prevents a global gPROMS-validation claim. The defensible scope remains a prospectively gPROMS-confirmed surrogate-assisted Pareto-local design region and compromise recommendation within the prescribed domain, subject to the failed gates reported here.
