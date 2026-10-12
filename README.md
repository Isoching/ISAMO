# ISAMO: interpretable multi-objective design for electrified methanol production

**ISAMO** stands for **Interpretable Surrogate-Assisted Multi-objective Optimisation**. This project supports engineering research on producing green methanol from dilute flue gas through an electrified process that couples carbon capture, electrochemical CO2 reduction, water electrolysis, methanol synthesis and utility operation.

The complete lightweight V30 code and required datasets are in **[source/](source/)**. The step-by-step Chinese guide is **[NOTES.md](NOTES.md)**.

## What is this project for?

ISAMO helps researchers compare operating and equipment configurations when carbon use, energy demand and equipment scale compete. It combines process-simulation data with fitted surrogate models, searches for Pareto trade-offs, ranks compromise designs under explicit preferences and explains the model predictions.

Eight bounded design variables describe the coupled process. The three optimisation objectives are:

| Objective | Meaning | Direction |
| --- | --- | --- |
| Carbon utilisation, CU | Fraction of feed carbon converted to methanol under the frozen process convention | Maximise |
| Equivalent specific energy consumption, SEC | Energy demand per kilogram of methanol, using the frozen reporting convention | Minimise |
| Capital-scale proxy | Dimensionless indicator of equipment/process scale | Minimise |

Methanol output, the LGWP proxy and electrolyser-area intensity are reported post-hoc indicators. The capital-scale proxy supports relative comparisons; it does not provide a monetary investment estimate.

## Why it matters, and where it could lead

A change to one subsystem can alter the performance of the coupled process. A design that favours carbon conversion may impose different energy or equipment requirements, so design selection needs an explicit comparison of competing objectives. Detailed process simulation supplies the reference responses, while surrogate models support repeated candidate evaluation.

Optimisation also directs the search towards a narrower, decision-relevant region. Good predictive accuracy over a broad sample does not by itself establish reliability in that region. ISAMO addresses this limitation by returning targeted Pareto-local gPROMS cases to model development and assessing the frozen models on independent confirmation cases.

The resulting workflow is intended for early-stage process screening, simulation-budget allocation and interpretable comparison of design compromises. It provides a pattern that could be adapted to other electrified CO2-utilisation or hydrogen-based processes after replacing the process model, datasets, units, bounds and objective definitions. Time-varying renewable electricity, electricity-carbon intensity and explicit economic or operability constraints are possible future extensions; they require additional modelling and validation.

## Main contributions

| Engineering limitation | ISAMO contribution | Practical use |
| --- | --- | --- |
| Repeated evaluation through detailed process simulation | Target-specific heterogeneous surrogate ensembles fitted to real gPROMS-derived cases | Evaluate candidate configurations with the learned models |
| Optimisation-induced shift towards a locally demanding region | Pareto-local active-learning enrichment with recorded simulator returns | Improve accuracy where the design decisions are made |
| Uncertainty and validation roles can become mixed | Grouped cross-conformal calibration, frozen model records and distinct development/STOP/confirmation datasets | Audit calibration and assess decision-region reliability |
| Competing objectives require preference-sensitive choices | NSGA-III Pareto search with TOPSIS/VIKOR and user-specified objective weights | Compare compromise designs under stated priorities |
| Surrogate decisions can be difficult to interpret | SHAP-based interpretation linked to candidate inputs and model predictions | Explain fitted relationships and inspect design drivers |

These contributions concern the integrated research workflow and its methanol case study. SHAP attributions describe the fitted models, and recommendations remain conditional on the process assumptions, model validity and chosen weights.

## Evidence included in this release

The recorded independent post-freeze confirmation set contains **94 accepted cases from 96 issued cases**:

| Target | RMSE reduction versus the corresponding strongest model trained without the 126 new AL cases | Fixed-range NRMSE | Empirical coverage of nominal 95% intervals |
| --- | ---: | ---: | ---: |
| CU | 42.5% | 3.99% | 100.00% |
| SEC | 8.5% | 1.81% | 96.81% |
| Capital-scale proxy | 11.2% | 1.52% | 100.00% |

These are recorded case-study results, not results from a new run performed by cloning the repository. The comparator evidence is in [confirmation96_method_contribution.csv](source/data/validation_evidence/confirmation96_method_contribution.csv); the fast checks recompute the final metrics from recorded predictions and simulator labels. Other validation subsets have different coverage, as documented in [the reproducibility guide](source/docs/DATA_AND_REPRODUCIBILITY.md).

## Start with the workflow that matches your goal

| Your goal | What you need | Guide |
| --- | --- | --- |
| Check data integrity, recompute recorded metrics and reproduce manuscript rankings | Python 3.12 and the bundled data | Quick start below; NOTES sections 3–4 |
| Predict new designs, run optimisation or interpret predictions | The three original frozen models, or models retrained from the bundled data | NOTES sections 5–8 |
| Adapt the method to a different process | A new process model, data contract, simulator interface and independent validation plan | NOTES section 10 |

## Quick start: use the bundled evidence

Clone the repository and enter its Python project directory:

```bash
git clone https://github.com/Isoching/ISAMO.git
cd ISAMO/source
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
python -m pip install -e .
python -m isamo_release verify
python -m isamo_release evaluate --output outputs/recorded_validation
python -m isamo_release rank --output outputs/manuscript_equal_weight
```

Expected outputs:

- `verify` reports `PASS` after checking source hashes, dataset identities and recorded metrics.
- `outputs/recorded_validation/validation_metrics_recomputed.csv` contains the recorded error and coverage metrics.
- `outputs/manuscript_equal_weight/pareto_ranked.csv` adds TOPSIS/VIKOR scores and ranks to the **589-point manuscript fixed-pool front**. The rank-1 candidates are TOPSIS `C012542` and VIKOR `C012422`.

The CSV retains its input row order: use the rank columns to find the preferred rows. Every output path must be new or empty. These commands use bundled records and do not require model weights or live gPROMS simulation.

## Training, new optimisation and interpretation

Install the additional learner and plotting dependencies before a new model run:

```bash
python -m pip install -e ".[learn,plot]"
python -m isamo_release train --variant pooled_all_al --output outputs/retrained_v30
python -m isamo_release optimise --models outputs/retrained_v30/models --output outputs/new_nsga3
python -m isamo_release rank --input outputs/new_nsga3/pareto_ranked.csv --weights 2 1 1 --output outputs/new_nsga3_carbon_priority
```

The weights are ordered **CU, SEC, capital-scale proxy**. Specifying the new search file is essential: omitting `--input` ranks the bundled manuscript front. Full training and optimisation are computationally intensive. New runs produce new models and search results; reproduction of historical model hashes is a separate frozen-model check.

The three historical frozen weights total approximately 1.04 GiB and are held in the separate local results package. Users who have those weights can supply `--models PATH` instead of retraining. [NOTES.md](NOTES.md) explains both paths, prediction inputs, calibration, output interpretation and common errors.

## Correct interpretation and reuse

The **6,221-point combined ten-seed archive** is a separate analysis scope with different recommendations. Select it explicitly with `--input data/final_pareto_archive.csv.gz`. Use the frozen bounds and units in [protocol.json](source/config/protocol.json); predictions for a different feed, process configuration or input domain require new validation.

Original reported LGWP/area values are preserved alongside their provenance. Their documented equation/geometry consistency issues are described in the reproducibility guide and the separate local review records. This source publication does not resolve those manuscript issues or turn proxy values into a full life-cycle or financial assessment.

Further resources: [source guide](source/README.md), [data and reproducibility](source/docs/DATA_AND_REPRODUCIBILITY.md), and [license notice](source/LICENSE_NOTICE.md). Copyright remains with the project authors; no new open-source license is assigned by this release.
