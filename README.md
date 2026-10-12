# ISAMO: lightweight V30 research source

Interpretable surrogate-assisted multi-objective optimisation for electrified green methanol production from dilute flue gas.

The current V30 source and required datasets are in **[source/](source/)**. The package connects the original target-specific surrogate learners with active-learning selection, NSGA-III, TOPSIS/VIKOR, grouped cross-conformal calibration and model interpretation.

## Start here

| Item | Location |
| --- | --- |
| Current source and installation guide | [source/README.md](source/README.md) |
| Release notes and data roles | [NOTES.md](NOTES.md) |
| Data provenance and reproducibility | [source/docs/DATA_AND_REPRODUCIBILITY.md](source/docs/DATA_AND_REPRODUCIBILITY.md) |
| Copyright and license notice | [source/LICENSE_NOTICE.md](source/LICENSE_NOTICE.md) |

From the repository root, with Python 3.12:

```bash
cd source
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m isamo_release verify
python -m unittest discover -s tests -v
python -m isamo_release evaluate --output outputs/recorded_validation
python -m isamo_release rank --output outputs/equal_weight_ranking
```

On Windows, activate with `.venv\Scripts\activate`. The fast verification, metric evaluation and ranking commands use the bundled data. Prediction and optimisation require separately supplied model weights or a new training run; live process simulation requires the original licensed gPROMS environment.

Default ranking reproduces the current manuscript's 589-point fixed-pool recommendations. The 6,221-point expanded archive is supplied as a separate analysis scope.

This is the standalone lightweight V30 source repository. Follow the source guide for training, prediction, optimisation and interpretation. Large frozen model weights and manuscript result bundles are held in the separate local results package. See [NOTES.md](NOTES.md) for the data roles and reproduction scope.
