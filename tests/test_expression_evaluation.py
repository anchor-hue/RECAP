"""Scientific evaluation checks; no training or GPU is required."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from recap.metrics import (
    compute_standardized_expression_metrics, compute_weighted_expression_metrics,
    constant_expression_proteins, standardize_expression_all_entries,
)
from scripts.summarize_benchmarks import (
    METRICS, main as summarize_main, protein_coverage_union, seed_macro_means,
    summarize_seed, summarize_union_coverage,
)


class ExpressionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[[1., 3.], [2., 8.]], [[4., 2.], [6., 7.]]])
        self.pred = np.array([[[2., 5.], [7., 4.]], [[3., 1.], [9., 6.]]])
        self.fractions = np.array([[1., 0.], [0.2, 0.8]])

    def test_affine_invariance_and_independent_protein_parameters(self):
        pred = self.truth * np.array([3., 0.1]) + np.array([100., -20.])
        metrics, _ = compute_standardized_expression_metrics(pred, self.truth, self.fractions)
        self.assertAlmostEqual(metrics["CCC"], 1.)
        self.assertAlmostEqual(metrics["Corr"], 1.)
        direct = compute_weighted_expression_metrics(pred, self.truth, self.fractions)
        self.assertLess(direct["CCC"], 0.5)

    def test_zero_fraction_entries_affect_standardization_only(self):
        changed = self.pred.copy()
        changed[0, 1] = [1000., -500.]
        original = compute_weighted_expression_metrics(self.pred, self.truth, self.fractions)
        direct = compute_weighted_expression_metrics(changed, self.truth, self.fractions)
        self.assertEqual(original["CCC"], direct["CCC"])
        a, _ = compute_standardized_expression_metrics(self.pred, self.truth, self.fractions)
        b, _ = compute_standardized_expression_metrics(changed, self.truth, self.fractions)
        self.assertGreater(abs(a["CCC"] - b["CCC"]), 0.01)

    def test_sample_sd_and_unweighted_standardization(self):
        result = standardize_expression_all_entries(self.pred)
        flat = self.pred.reshape(-1, 2)
        expected = (flat - flat.mean(axis=0)) / flat.std(axis=0, ddof=1)
        np.testing.assert_allclose(result.reshape(-1, 2), expected)

    def test_constant_truth_mask_is_independent_of_predictions(self):
        truth = self.truth.copy()
        truth[:, :, 0] = 0.1
        _, audit = compute_standardized_expression_metrics(self.pred, truth, self.fractions)
        _, other = compute_standardized_expression_metrics(self.pred + 5, truth, self.fractions)
        np.testing.assert_array_equal(audit["retained_proteins"], [False, True])
        np.testing.assert_array_equal(audit["retained_proteins"], other["retained_proteins"])

    def test_constant_prediction_retained_and_undefined_pearson(self):
        pred = np.full_like(self.truth, 0.1)
        metrics, audit = compute_standardized_expression_metrics(pred, self.truth, self.fractions)
        self.assertEqual(audit["retained_proteins"].sum(), 2)
        self.assertEqual(audit["constant_prediction_proteins"].sum(), 2)
        self.assertEqual(metrics["CCC"], 0.)
        self.assertTrue(np.isnan(metrics["Corr"]))
        self.assertTrue((standardize_expression_all_entries(pred) == 0).all())
        direct = compute_weighted_expression_metrics(pred, self.truth, self.fractions)
        self.assertEqual(direct["CCC"], 0.)
        self.assertTrue(np.isnan(direct["Corr"]))

    def test_near_constant_proteins_are_not_discarded(self):
        values = np.full((2, 2, 1), 0.1)
        values[1, 1, 0] = np.nextafter(0.1, 1.)
        self.assertFalse(constant_expression_proteins(values)[0])
        self.assertTrue(np.isfinite(standardize_expression_all_entries(values)).all())

    def test_weighted_moments_match_independent_probability_calculation(self):
        x, y = self.pred.ravel(), self.truth.ravel()
        weights = np.repeat(self.fractions.ravel(), 2)
        weights /= weights.sum()
        mx, my = np.dot(weights, x), np.dot(weights, y)
        vx, vy = np.dot(weights, (x - mx)**2), np.dot(weights, (y - my)**2)
        cov = np.dot(weights, (x - mx) * (y - my))
        metrics = compute_weighted_expression_metrics(self.pred, self.truth, self.fractions)
        self.assertAlmostEqual(metrics["CCC"], 2 * cov / (vx + vy + (mx - my)**2))
        self.assertAlmostEqual(metrics["Corr"], cov / np.sqrt(vx * vy))

    def test_invalid_inputs_do_not_silently_drop_entries(self):
        pred = self.pred.copy()
        pred[0, 1, 0] = np.nan  # zero fraction still participates in normalization
        with self.assertRaises(ValueError):
            compute_standardized_expression_metrics(pred, self.truth, self.fractions)
        with self.assertRaises(ValueError):
            compute_standardized_expression_metrics(self.pred, self.truth, -self.fractions)
        with self.assertRaises(ValueError):
            compute_standardized_expression_metrics(self.pred, np.ones_like(self.truth), self.fractions)

    def test_macro_requires_all_datasets_for_each_metric(self):
        table = pd.DataFrame([
            {"seed": seed, **{m: value for m in METRICS}}
            for seed, value in [(2021, 1.), (2021, 3.), (2022, 4.), (2022, 6.)]
        ])
        table.loc[0, METRICS[0]] = np.nan
        macro = seed_macro_means(table, 2)
        self.assertTrue(np.isnan(macro.loc[2021, METRICS[0]]))
        self.assertEqual(macro.loc[2021, METRICS[1]], 2.)
        self.assertEqual(macro.loc[2022, METRICS[0]], 5.)
        self.assertEqual(macro[METRICS[0]].count(), 1)
        self.assertAlmostEqual(macro[METRICS[1]].std(ddof=1), np.std([2., 5.], ddof=1))

    def test_summary_aligns_sample_cell_type_and_protein_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "results/example/seed_2021"
            run.mkdir(parents=True)
            spots, types, proteins = ["s1", "s2"], ["a", "b"], ["p1", "p2"]
            np.savez(root / "truth.npz", E=self.truth, spot_names=spots,
                     cell_types=types, proteins=proteins)
            pd.DataFrame(self.fractions, index=spots, columns=types).to_csv(root / "fractions.csv")
            pd.DataFrame([{m: 0.9 for m in METRICS if m.startswith("target_fraction_")}]).to_csv(
                run / "metrics.csv", index=False,
            )
            # Reverse each axis independently relative to the GT files.
            np.savez(run / "target_celltype_expression_input_scale.npz",
                     E=self.pred[::-1, ::-1, ::-1], spot_names=spots[::-1],
                     cell_types=types[::-1], proteins=proteins[::-1])
            manifest = [{"name": "example", "target_expression": "truth.npz",
                         "target_fractions": "fractions.csv"}]
            with patch("scripts.summarize_benchmarks.ROOT", root):
                row = summarize_seed(2021, root / "results", manifest, {"example": proteins}).iloc[0]
            expected_a, _ = compute_standardized_expression_metrics(self.pred, self.truth, self.fractions)
            expected_b = compute_weighted_expression_metrics(self.pred, self.truth, self.fractions)
            self.assertAlmostEqual(row["table2A_standardized_weighted_CCC"], expected_a["CCC"])
            self.assertAlmostEqual(row["table2B_input_scale_weighted_CCC"], expected_b["CCC"])
            self.assertEqual(row["zero_fraction_entries"], 1)


class CoverageUnionTests(unittest.TestCase):
    def test_union_counts_distinct_gt_proteins_and_excludes_extra_outputs(self):
        panels = [["a", "b", "extra", "a"], ["b", "c"]]
        self.assertEqual(protein_coverage_union(["a", "b", "c", "d"], panels), 0.75)
        self.assertEqual(protein_coverage_union(["a", "b", "c", "d"], panels[::-1]), 0.75)
        self.assertEqual(protein_coverage_union(["a", "b", "c", "d"], panels[:1]), 0.5)
        with self.assertRaises(ValueError):
            protein_coverage_union([], panels)
        with self.assertRaises(ValueError):
            protein_coverage_union(["a"], [])

    def test_missing_requested_seed_is_not_silently_ignored(self):
        data = {"example": {"truth": {"a", "b"}, "predictions": {2021: {"a"}}}}
        with self.assertRaises(ValueError):
            summarize_union_coverage(data, [2021, 2022])

    def test_cli_reports_union_then_equal_dataset_mean_without_seed_sd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            manifest, panels = [], []
            for dataset, n_proteins in [("small", 3), ("large", 4)]:
                proteins = [f"p{i}" for i in range(n_proteins)]
                truth = np.arange(4 * n_proteins, dtype=float).reshape(2, 2, n_proteins)
                spots, types = ["s1", "s2"], ["a", "b"]
                np.savez(root / f"{dataset}.npz", E=truth, spot_names=spots,
                         cell_types=types, proteins=proteins)
                pd.DataFrame([[0.6, 0.4], [1., 0.]], index=spots, columns=types).to_csv(
                    root / f"{dataset}.csv")
                manifest.append({"name": dataset, "target_expression": f"{dataset}.npz",
                                 "target_fractions": f"{dataset}.csv"})
                panels.append({"dataset": dataset, "proteins": ["p1"]})
                for seed, selected in [(2021, ["p0", "p1"]), (2022, ["p1", "p2"])]:
                    run = root / "results" / dataset / f"seed_{seed}"
                    run.mkdir(parents=True)
                    pd.DataFrame([{m: 0.9 for m in METRICS if m.startswith("target_fraction_")}]).to_csv(
                        run / "metrics.csv", index=False)
                    np.savez(run / "target_celltype_expression_input_scale.npz",
                             E=truth[:, :, [proteins.index(p) for p in selected]] * 1.1 + 0.2,
                             spot_names=spots, cell_types=types, proteins=selected)
            (root / "data/benchmark_manifest.json").write_text(json.dumps(manifest))
            (root / "data/benchmark_common_protein_panels.json").write_text(json.dumps(panels))
            out = root / "summary"
            argv = ["summarize_benchmarks.py", "--seeds", "2021,2022",
                    "--results-root", str(root / "results"), "--outdir", str(out)]
            with patch("scripts.summarize_benchmarks.ROOT", root), patch("sys.argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize_main()
            coverage = pd.read_csv(out / "coverage_by_dataset.csv").set_index("dataset")
            self.assertEqual(coverage.loc["small", "protein_coverage"], 1.)
            self.assertEqual(coverage.loc["large", "protein_coverage"], 0.75)
            self.assertTrue((coverage["n_seeds"] == 2).all())
            self.assertAlmostEqual(pd.read_csv(out / "coverage_summary.csv")["mean"].iloc[0], 87.5)
            summary = pd.read_csv(out / "cross_dataset_summary.csv", index_col="metric")
            self.assertAlmostEqual(summary.loc["protein_coverage", "mean"], 0.875)
            self.assertTrue(np.isnan(summary.loc["protein_coverage", "sample_std"]))
            self.assertTrue(np.isnan(summary.loc["protein_coverage", "valid_seed_count"]))
            self.assertNotIn("protein_coverage", pd.read_csv(out / "seed_macro.csv").columns)
            self.assertIn("run_protein_coverage", pd.read_csv(out / "per_run.csv").columns)
            self.assertEqual(summary.loc["table2A_standardized_weighted_CCC", "valid_seed_count"], 2)
            self.assertAlmostEqual(summary.loc["table2A_standardized_weighted_CCC", "mean"], 1.)
            protocol = json.loads((out / "evaluation_protocol.json").read_text())
            self.assertEqual(protocol["coverage"]["seeds"], [2021, 2022])


if __name__ == "__main__":
    unittest.main()
