from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class KagglePackagingTests(unittest.TestCase):
    def test_dataset_metadata_is_private_upload_ready(self) -> None:
        metadata = json.loads(
            (ROOT / "kaggle" / "runtime_dataset" / "dataset-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["id"], "danielsolo1770/alphamath-runtime-bundle")
        self.assertEqual(metadata["licenses"], [{"name": "MIT"}])
        self.assertGreaterEqual(len(metadata["title"]), 6)

    def test_kernel_metadata_and_entrypoint_are_consistent(self) -> None:
        folder = ROOT / "kaggle" / "kernel"
        metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertTrue((folder / metadata["code_file"]).is_file())
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertEqual(metadata["language"], "python")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertIn("danielsolo1770/alphamath-runtime-bundle", metadata["dataset_sources"])
        external_sources = metadata["dataset_sources"][1:] + metadata["model_sources"]
        self.assertTrue(external_sources, "an offline weight source must be attached")

    def test_kernel_accepts_kaggle_expanded_dataset(self) -> None:
        script = ROOT / "kaggle" / "kernel" / "alpha_math_kaggle.py"
        spec = importlib.util.spec_from_file_location("alphamath_kaggle_entrypoint", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            expanded = temp / "input" / "runtime" / "AlphaMath"
            (expanded / "src").mkdir(parents=True)
            (expanded / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            (expanded / "src" / "agent.py").write_text("# marker\n", encoding="utf-8")
            module.INPUT_ROOT = temp / "input"
            module.SOURCE_ROOT = temp / "working" / "source"
            repo, source = module._prepare_repository()
            self.assertEqual(source, str(expanded))
            self.assertTrue((repo / "src" / "agent.py").is_file())


if __name__ == "__main__":
    unittest.main()
