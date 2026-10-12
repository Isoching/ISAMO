from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import cost_matrix, nondominated, rank
from .core import FEATURES, INTEGER_FEATURES, PREDICTED, PROJECT, TARGETS, config, identities, load_model, read, sha256, validate_splits, write_json


def train(output: Path, variant: str) -> None:
    import joblib
    from isamo_green_meoh.config import config_manifest, default_config
    from isamo_green_meoh.surrogate import train_surrogates

    validate_splits()
    development = read(PROJECT / "data/development_2586.csv")
    if variant == "pooled_existing60":
        development = development.loc[development["development_source"].isin(["legacy2400", "existing60"])].reset_index(drop=True)
    cfg = default_config()
    models, metrics, oof, audit = train_surrogates(development, read(PROJECT / "data/legacy_holdout_600.csv"), cfg)
    model_dir = output / "models"
    model_dir.mkdir()
    assets = {}
    for target in TARGETS:
        path = model_dir / f"{target}.joblib"
        joblib.dump(models[target], path, compress=("zlib", 3))
        assets[target] = {"path": f"models/{path.name}", "sha256": sha256(path), "bytes": path.stat().st_size}
    metrics.to_csv(output / "training_metrics.csv", index=False)
    audit.to_csv(output / "candidate_audit.csv", index=False)
    oof.to_csv(output / "development_oof_predictions.csv", index=False)
    write_json(output / "training_record.json", {"status": "NEW_RETRAINED_RUN_NOT_HISTORICAL_FREEZE", "variant": variant,
               "config": config_manifest(cfg), "training_rows": len(development),
               "training_hash_set_sha256": __import__("hashlib").sha256("\n".join(sorted(identities(development))).encode()).hexdigest(), "models": assets})


def optimise(model_dir: Path, output: Path, seeds: list[int], population: int, generations: int, partitions: int) -> None:
    from isamo_green_meoh.moea import _nsga3_optimize_pymoo
    from isamo_green_meoh.surrogate import predict_surrogates

    bounds = config()["fixed_input_bounds"]
    lower = np.array([bounds[f]["min"] for f in FEATURES])
    upper = np.array([bounds[f]["max"] for f in FEATURES])
    models = {target: load_model(model_dir, target) for target in TARGETS}
    fronts = []
    for seed in dict.fromkeys(seeds):
        evaluated_x, evaluated_f = [], []

        def evaluate(values):
            x = np.clip(np.asarray(values, dtype=float), lower, upper).copy()
            for feature in INTEGER_FEATURES:
                column = FEATURES.index(feature)
                x[:, column] = np.rint(x[:, column])
            prediction = predict_surrogates(models, pd.DataFrame(x, columns=FEATURES), FEATURES)
            cost = cost_matrix(prediction[TARGETS].to_numpy())
            if not np.isfinite(cost).all():
                raise ValueError("Non-finite objective predictions")
            evaluated_x.append(x); evaluated_f.append(cost)
            return cost

        # Use the actual publication backend; an optimizer failure cannot silently select a different algorithm.
        result = _nsga3_optimize_pymoo(evaluate, list(zip(lower, upper)), population, generations, partitions, 0.9, 0.2, 0.08, seed)
        archive = pd.DataFrame(np.vstack(evaluated_x), columns=FEATURES)
        costs = np.vstack(evaluated_f)
        archive[PREDICTED] = np.column_stack([-costs[:, 0], costs[:, 1], costs[:, 2]])
        archive["input_row_sha256"] = identities(archive)
        archive = archive.drop_duplicates("input_row_sha256", keep="first").reset_index(drop=True)
        front = archive.iloc[nondominated(cost_matrix(archive[PREDICTED].to_numpy()))].copy()
        front["source_seed"] = seed
        fronts.append(front)
        front.to_csv(output / f"pareto_seed_{seed}.csv", index=False)
        pd.DataFrame(result.history).to_csv(output / f"history_seed_{seed}.csv", index=False)
        print(f"Completed seed {seed}: {len(front)} Pareto rows", flush=True)
    union = pd.concat(fronts, ignore_index=True).drop_duplicates("input_row_sha256", keep="first").reset_index(drop=True)
    front = union.iloc[nondominated(cost_matrix(union[PREDICTED].to_numpy()))].reset_index(drop=True)
    rank(front).to_csv(output / "pareto_ranked.csv", index=False)
    write_json(output / "run_record.json", {"engine": "pymoo.nsga3", "seeds": seeds, "population": population, "generations": generations,
               "reference_partitions": partitions, "front_rows": len(front), "scope": "new search; historical archive remains unchanged"})


def explain(model_dir: Path, inputs: Path, output: Path, n_eval: int, n_background: int, n_permutations: int) -> None:
    from isamo_green_meoh.surrogate import predict_surrogates

    frame = read(inputs)[FEATURES].iloc[:n_eval].reset_index(drop=True)
    background = read(PROJECT / "data/candidate_pool.csv.gz")[FEATURES].sample(n=n_background, random_state=42).to_numpy()
    rng = np.random.default_rng(42)
    rows = []
    for target in TARGETS:
        model = load_model(model_dir, target)

        def prediction(values):
            return predict_surrogates({target: model}, pd.DataFrame(values, columns=FEATURES), FEATURES)[target].to_numpy()

        baseline = float(prediction(background).mean())
        for row_index, x in enumerate(frame.to_numpy()):
            contributions = np.zeros(len(FEATURES))
            for _ in range(n_permutations):
                current = background.copy()
                previous = prediction(current).mean()
                for j in rng.permutation(len(FEATURES)):
                    current[:, j] = x[j]
                    new = prediction(current).mean()
                    contributions[j] += new - previous
                    previous = new
            for feature, value in zip(FEATURES, contributions / n_permutations):
                rows.append({"target": target, "row_index": row_index, "feature": feature, "permutation_shap": value, "background_prediction": baseline})
        del model
        gc.collect()
    pd.DataFrame(rows).to_csv(output / "permutation_shap.csv", index=False)
    write_json(output / "explanation_record.json", {"evaluation_rows": len(frame), "background_rows": n_background,
               "permutations": n_permutations, "seed": 42, "scope": "surrogate associations; no causal process inference"})
