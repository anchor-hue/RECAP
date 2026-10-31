"""Check the packaged downstream run settings without training a model."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class DownstreamProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only the command routing; model training needs optional PyTorch.
        package = types.ModuleType("recap")
        package.DAProDeconvConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
        package.DAProDeconvPipeline = lambda config: None
        script = Path(__file__).resolve().parents[1] / "run_recap.py"
        spec = importlib.util.spec_from_file_location("downstream_run_recap", script)
        cls.runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"recap": package}):
            spec.loader.exec_module(cls.runner)

    def run_with_mock_pipeline(self, *arguments):
        with patch.object(sys, "argv", ["run_recap.py", *arguments]), patch.object(
            self.runner, "DAProDeconvPipeline"
        ) as pipeline:
            self.runner.main()
        return pipeline.call_args.args[0], pipeline.return_value.fit_from_files.call_args.kwargs

    def test_proteomixture_uses_all_pure_reference_without_second_z_score(self):
        config, inputs = self.run_with_mock_pipeline(
            "physical", "--design", "all_pure_reference", "--seed", "2021"
        )
        self.assertEqual(config.seed, 2021)
        self.assertEqual(config.normalize, "none")
        self.assertEqual(config.dirichlet_alphas, [0.3])
        self.assertEqual(config.dirichlet_alpha_weights, [1.0])
        self.assertEqual(Path(inputs["ref_path"]).name, "reference_expression.csv")
        self.assertEqual(Path(inputs["ref_path"]).parent.name, "all_pure_reference")
        self.assertIsNone(inputs["target_expression_path"])

    def test_scpro_keeps_original_model_normalization(self):
        config, inputs = self.run_with_mock_pipeline(
            "scpro", "--mode", "immune9", "--seed", "2021"
        )
        self.assertEqual(config.normalize, "z_score")
        self.assertEqual(config.dirichlet_alphas, [0.1, 0.2, 0.3])
        self.assertEqual(Path(inputs["ref_path"]).parent.name, "immune9")

    def test_proteomixture_rejects_other_seeds(self):
        with patch.object(sys, "argv", [
            "run_recap.py", "physical", "--design", "all_pure_reference",
            "--seed", "2022",
        ]):
            with self.assertRaisesRegex(ValueError, "fixed random seed 2021"):
                self.runner.main()


if __name__ == "__main__":
    unittest.main()
