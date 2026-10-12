# Data and reproducibility

## Roles

| File | Rows | Permitted use |
| --- | ---: | --- |
| `development_2586.csv` | 2,586 | Learner development and grouped cross-conformal calibration |
| `legacy_holdout_600.csv` | 600 | Historical learner evaluation/guardrail; never refit |
| `stop_30.csv` | 30 | Iterative stopping and variant selection; never train |
| `confirmation_94.csv` | 94 | Independent post-freeze confirmation; 96 issued |
| `fixed_region_60.csv` | 60 | Independent fixed-region evaluation |
| `global_converged_12.csv` | 12 | Descriptive converged subset of 40 issued global cases |

Development combines 2,400 original cases, 60 provenance-approved local cases, and AL rounds with 32, 31, 32 and 31 accepted cases. Canonical input hashes match the original frozen 2,586-row training contract. Grouped cross-conformal uses the original 2,531 source/cluster groups, excludes every held-out dataset, and retains the original row order.

Canonical dataset exports preserve all eight feature values and three scientific target values at floating-point round-trip precision. Rebuilt target columns take precedence exactly as in the original learner script. Source-run paths, operator notes and redundant raw diagnostic columns are excluded from the lightweight canonical tables. Unabridged simulation returns and QC/attempt records are retained in the local results package. Gzip snapshots decompress to their original bytes.

## What the fast checks establish

- Source and dataset integrity against `SOURCE_MANIFEST.json`.
- Original canonical training identities and exclusion of holdout, STOP and validation rows from development.
- Recomputed final RMSE/fixed-range NRMSE and interval coverage from recorded pointwise predictions and simulator labels.
- Original finite-sample calibration quantiles from grouped OOF scores.
- TOPSIS/VIKOR rankings against original execution code across the entire supplied final archive.
- Optional frozen-model verification checks all three original model hashes and 140 sampled prediction means and ensemble SDs.

Recorded comparator RMSEs and their reported improvements are supplied in `data/validation_evidence/confirmation96_method_contribution.csv`; the fast metric check does not retrain comparator models. A new full training/search/calibration run is an explicit operation, not a claimed reproduction performed during packaging.

The confirmed final fixed-range NRMSE values are 3.99%, 1.81% and 1.52%. Confirmation interval coverage is 100%, 96.81% and 100%. The 12-case global converged subset is descriptive and has substantially lower CU/SEC coverage; confirmation coverage does not establish global calibration reliability.

## Scientific versions

V30 is the model/validation release. The original `V29_FINAL` learner namespace is retained for serialization and exact learner compatibility; it does not turn the release into the root V1 synthetic demonstration.

Two V30 archive scopes are preserved separately. `manuscript_fixed_pool_front.csv` has 589 points and reproduces the current manuscript recommendations `C012542` (TOPSIS) and `C012422` (VIKOR). `final_pareto_archive.csv.gz` has 6,221 points after the final ten-seed NSGA-III union and yields different recommendations. Default ranking/explanation uses the manuscript fixed-pool scope. `final_decision_anchors.csv` preserves the separate five formal post-freeze expanded-archive anchors.

The current manuscript's representative-solution table is copied unchanged to `data/manuscript_reported_six_solutions.csv`. Its reported LGWP reduction remains approximately 40% for TOPSIS, as requested by the author. Previously proposed R1 equation-consistent LGWP values and R2 physical area corrections are stored separately under the results package's review records and are not silently substituted into manuscript values. The documented equation/table inconsistency remains a review issue.

SEC, annual production and area reconstruction retain their original frozen contracts. The 8,400-hour legacy yield basis and 8,760-hour V30 SEC reporting basis must not be globally substituted. Geometry-based review corrections do not modify the frozen learner targets, weights, or model hashes.

## Model and simulator assets

The three frozen models total approximately 1.04 GiB and are kept in the separate results folder. A standalone source checkout can train using the complete real development data or load those models with `--models PATH`. Model pickle loading requires the declared matching software environment and preserved original class names.

The original gPROMS runner/schema and target-reconstruction code are preserved as reference code. The native process project is included when available under `gproms/`. Running the native simulation requires gPROMS and the relevant original execution environment; no simulator is replaced with synthetic ground truth.
