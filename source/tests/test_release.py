from __future__ import annotations

import ast
import json
import types
import unittest

import numpy as np
import pandas as pd

from isamo_release.analysis import evaluate, finite_sample_quantile, rank
from isamo_release.core import PROJECT, PREDICTED, TARGETS, read, validate_splits


class ReleaseIntegrityTests(unittest.TestCase):
    def test_development_identity_and_validation_exclusion(self):
        result = validate_splits()
        self.assertEqual(result["rows"]["development_2586.csv"], 2586)
        self.assertEqual(result["training_vs_all_validation_overlap"], 0)
        self.assertEqual(result["conformal_unique_groups"], 2531)

    def test_recorded_pointwise_predictions_reproduce_confirmation_metrics(self):
        metrics = evaluate()
        confirmation = metrics.loc[metrics.dataset == "confirmation96"].set_index("target")
        self.assertTrue((confirmation.n == 94).all())
        np.testing.assert_allclose(confirmation.loc[TARGETS, "fixed_range_nrmse_pct"],
                                   [3.9885804822214643, 1.8143719771400935, 1.5230199121780588], rtol=0, atol=1e-10)
        np.testing.assert_allclose(confirmation.loc[TARGETS, "empirical_coverage_pct"], [100, 100*91/94, 100], rtol=0, atol=1e-10)

    def test_grouped_oof_scores_reproduce_all_frozen_quantiles(self):
        scores = read(PROJECT / "data/cross_conformal_oof_scores.csv")
        freeze = json.loads((PROJECT / "freeze/cross_conformal_freeze.json").read_text())
        for target in TARGETS:
            self.assertAlmostEqual(finite_sample_quantile(scores[f"{target}_scaled_abs_residual"].to_numpy()),
                                   freeze["quantiles"][target]["finite_sample_quantile"], places=11)

    def test_portable_ranking_matches_original_code_on_entire_frozen_archive(self):
        original = PROJECT / "reference/original_execution/run_v30_final_nsga3_analysis.py"
        tree = ast.parse(original.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add_mcdm")
        module = types.ModuleType("original_mcdm")
        module.__dict__.update(np=np, pd=pd, PREDICTED=PREDICTED)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(original), "exec"), module.__dict__)
        archive = read(PROJECT / "data/final_pareto_archive.csv.gz")
        expected, actual = module.add_mcdm(archive), rank(archive)
        for col in ["topsis_rank_prequery", "vikor_rank_prequery"]:
            np.testing.assert_array_equal(actual[col], expected[col])
        for col in ["topsis_closeness_prequery", "vikor_Q_prequery"]:
            np.testing.assert_allclose(actual[col], expected[col], rtol=0, atol=2e-15)

    def test_ranking_rejects_invalid_stakeholder_weights(self):
        frame = pd.DataFrame({c: [0.5, 0.8] for c in PREDICTED})
        for weights in [(0, 0, 0), (-1, 1, 1), (float("nan"), 1, 1)]:
            with self.assertRaises(ValueError):
                rank(frame, weights)

    def test_fixed_pool_ranking_reproduces_current_manuscript_recommendations(self):
        front = rank(read(PROJECT / "data/manuscript_fixed_pool_front.csv"))
        table = read(PROJECT / "data/manuscript_reported_six_solutions.csv").set_index("solution")
        self.assertEqual(len(front), 589)
        for name, column in [("TOPSIS", "topsis_rank_prequery"), ("VIKOR", "vikor_rank_prequery")]:
            selected = front.loc[front[column].idxmin()]
            self.assertEqual(selected["candidate_id"], table.loc[name, "candidate_id"])
            np.testing.assert_allclose(selected[PREDICTED].to_numpy(dtype=float),
                table.loc[name, ["carbon_utilisation", "sec_kwh_per_kg_meoh", "capex_proxy"]].to_numpy(dtype=float),
                rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
