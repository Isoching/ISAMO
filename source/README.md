# ISAMO: portable V30 research workflow

Interpretable surrogate-assisted multi-objective optimisation for electrified methanol production from dilute flue gas. This repository contains real gPROMS-derived data and the original target-specific heterogeneous learner, plus small portable entry points for verification, training, NSGA-III, TOPSIS/VIKOR, grouped cross-conformal calibration and permutation-SHAP.

The eight inputs are `P_Cap`, `J_eCO2R`, `T_eCO2R`, `N_eCO2R`, `P_WE`, `N_WE`, `P_MeOH`, and `T_utility`. Objectives are maximising carbon utilisation and minimising equivalent SEC and the dimensionless capital-scale proxy. Methanol output, LGWP and electrode-area intensity are post-hoc indicators rather than additional optimisation objectives.

## Install and verify

Use Python 3.12 and run commands from the repository root. Dependencies are pinned to the locally verified recovered-model environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m isamo_release verify
python -m unittest discover -s tests -v
python -m isamo_release evaluate --output outputs/recorded_validation
python -m isamo_release rank --output outputs/equal_weight_ranking
```

On Windows activate with `.venv\Scripts\activate`. Verification, recorded-metric evaluation and archive ranking use the bundled datasets and do not need trained weights or gPROMS.

Default ranking uses the **589-point frozen fixed-candidate-pool front**, reproducing the current manuscript's TOPSIS candidate `C012542` and VIKOR candidate `C012422`. The **6,221-point combined ten-seed archive** is a separate analysis scope and has different compromise points. To rank that expanded archive explicitly:

```bash
python -m isamo_release rank --input data/final_pareto_archive.csv.gz --output outputs/expanded_archive_ranking
```

## Predict with original frozen models

Large model files are deliberately outside this GitHub source folder. In the two-folder local release they are at `../results/v30_model_freeze/models/`. Their expected sizes and SHA-256 hashes are in `freeze/final_model_freeze.json`.

```bash
python -m pip install -e '.[learn]'
python -m isamo_release verify --models ../results/v30_model_freeze/models
python -m isamo_release predict --models ../results/v30_model_freeze/models --output outputs/frozen_confirmation_prediction
```

For a standalone GitHub checkout, supply `--models /path/to/the/three/models`, or train from the included data. Historical model paths inside the unchanged freeze JSON are provenance, not workstation paths required by the portable commands.

## Train and search

```bash
python -m isamo_release train --variant pooled_all_al --output outputs/retrained_v30
python -m isamo_release train --variant pooled_existing60 --output outputs/retrained_comparator
python -m isamo_release calibrate --models outputs/retrained_v30/models --output outputs/retrained_calibration
python -m isamo_release optimise --models outputs/retrained_v30/models --output outputs/new_nsga3
python -m isamo_release rank --weights 2 1 1 --output outputs/carbon_priority
python -m isamo_release explain --models outputs/retrained_v30/models --output outputs/new_explanations
```

Training retains the original candidate library, five-member ensemble selection, target transformations and CU residual correction. It uses 2,586 permitted development rows; the alternative comparator uses 2,460 rows before the 126 new AL cases. The 600 historical holdout cases remain a learner guardrail, while STOP and post-freeze datasets are excluded from training and interval calibration. New runs are labelled as new runs and cannot overwrite bundled scientific snapshots. Retraining is computationally intensive and is not part of the fast checks.

The publication NSGA-III configuration is population 256, 300 generations and 20 reference partitions. Ten seeds are `42 137 271 419 613 809 1009 1217 1429 1693`; pass them with `--seeds`. The portable command uses the actual pymoo backend and preserves integer cell-count projection. It saves the cumulative evaluated front and ranks it after optimisation.

```bash
python -m isamo_release optimise --models ../results/v30_model_freeze/models --seeds 42 137 271 419 613 809 1009 1217 1429 1693 --output outputs/new_ten_seed_search
```

The original path-configurable AL/confirmation selector is available under `reference/original_execution/select_v30_batch.py`. Invoke that file with `--help`; its immutable pre-query model/front/target hashes and excluded-row inputs are required. The complete scored/annotated frozen candidate pool is supplied in `data/candidate_pool.csv.gz`. Original simulation return reconstruction and gPROMS schema code are also preserved in that directory. Live simulation requires the original licensed gPROMS environment; real accepted labels are already bundled.

## Plot recorded validation

```bash
python -m pip install -e '.[plot]'
python -m isamo_release plot --output outputs/confirmation_plots
```

This renders fresh parity plots from the recorded independent confirmation evidence. Original manuscript-ready figures and their source tables are in the separate results folder; the portable plots do not overwrite them. Explanation commands generate new permutation-SHAP estimates and do not claim bitwise reproduction of every historical interpretation plot.

## Layout and evidence

```text
config/                  Frozen bounds, target ranges and reconstruction constants
data/                    Real development, STOP, holdout, independent-validation and candidate datasets
freeze/                  Original model, analysis and cross-conformal freeze records
reference/original_execution/  Original algorithm scripts for provenance
src/isamo_green_meoh/     Original serialized-model-compatible learner
src/isamo_release/        Portable I/O, validation and workflow commands
tests/                   Data-exclusion, metric, quantile and ranking checks
```

See `docs/DATA_AND_REPRODUCIBILITY.md` for data roles, evidence scope and the original/proposed indicator distinction. `SOURCE_MANIFEST.json` seals every included source/data file; `DATA_PROVENANCE.csv` maps datasets back to their original hashes. Text-byte normalization is disabled to preserve scientific checksums across Git checkouts.

Copyright remains with the project authors; this preparation does not select a new open-source license. See `LICENSE_NOTICE.md`.
