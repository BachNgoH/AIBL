"""Regressions for public layouts, positional references, and resume provenance."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
from aibl.layout_attention import build_layout_attention_bias, compute_identity_and_spatial_heatmaps
from aibl.layout_validation import normalize_overlap_pairs, validate_layout
from aibl.inference_config import read_generation_metadata, write_generation_metadata


class LayoutValidationTests(unittest.TestCase):
    def test_bad_boxes_rejected_by_builder_and_heatmap(self):
        bad_boxes = [
            [0, 0, 0, 1], [0, 0, -0.1, 1], [-0.1, 0, 0.5, 1],
            [0.8, 0, 0.5, 1], [0, 0.8, 0.5, 0.5],
            [float("nan"), 0, 0.5, 1], [0, 0, float("inf"), 1],
            [0, 0, 1], ["0", 0, 1, 1], [True, 0, 1, 1], None,
        ]
        for bbox in bad_boxes:
            with self.subTest(bbox=bbox):
                instances = [{"bbox": bbox}]
                with self.assertRaises(ValueError):
                    build_layout_attention_bias(instances, 2, 2, 1, 1, "cpu", torch.float32)
                with self.assertRaises(ValueError):
                    compute_identity_and_spatial_heatmaps(instances, 2, 2)

    def test_malformed_layouts_rejected(self):
        for layout in (None, [], {"prompt": 3, "instances": []},
                       {"prompt": " ", "instances": []},
                       {"prompt": "one cup", "instances": {}},
                       {"prompt": "one cup", "instances": [None]}):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                validate_layout(layout)

    def test_pair_positions_do_not_change_with_metadata_ids(self):
        instances = [
            {"id": 1, "bbox": [0, 0, 0.6, 1], "depth": 0.1},
            {"id": 0, "bbox": [0.2, 0, 0.6, 1], "depth": 0.9},
        ]

        def bias(rows, pairs=None):
            return build_layout_attention_bias(rows, 1, 5, 1, 1, "cpu", torch.float32,
                                               overlap_pairs=pairs)

        expected = [dict(instances[0], depth=0.9), dict(instances[1], depth=0.1)]
        torch.testing.assert_close(bias(instances, [{"front": 0, "back": 1}]), bias(expected))
        torch.testing.assert_close(bias(instances, [{"front_id": 0, "back_id": 1}]), bias(instances))
        relabeled = [dict(instances[0], id=99), dict(instances[1], id=42)]
        torch.testing.assert_close(bias(instances, [{"front": 0, "back": 1}]),
                                   bias(relabeled, [{"front": 0, "back": 1}]))

    def test_ambiguous_or_invalid_pairs_rejected(self):
        rows = [{"id": 4}, {"id": 4}]
        for pair in ({"front_id": 4, "back_id": 4}, {"front": 0, "back": 2},
                     {"front": 0, "back": 0}, {"front": 0, "back_id": 4}):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                normalize_overlap_pairs(rows, [pair])


class GenerationMetadataTests(unittest.TestCase):
    def test_checkpoint_survives_exit_before_run_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text('[{"sample_id": 1, "seed": 42}]')
            script = (
                "import os, sys; from aibl.inference_config import read_generation_metadata, write_generation_metadata; "
                "write_generation_metadata(sys.argv[1], [{'sample_id': 2, 'seed': 99}], "
                "read_generation_metadata(sys.argv[1])); os._exit(0)"
            )
            result = subprocess.run([sys.executable, "-c", script, str(path)],
                                    env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), timeout=30)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(read_generation_metadata(path),
                             [{"sample_id": 1, "seed": 42}, {"sample_id": 2, "seed": 99}])

    def test_merge_preserves_old_samples_and_updates_only_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            old = [{"sample_id": 1, "seed": 42}, {"sample_id": 2, "seed": 43}]
            path.write_text(json.dumps(old))
            previous = read_generation_metadata(path)
            write_generation_metadata(path, [{"sample_id": 2, "seed": 99}, {"sample_id": 3, "seed": 44}], previous)
            self.assertEqual(read_generation_metadata(path),
                             [old[0], {"sample_id": 2, "seed": 99}, {"sample_id": 3, "seed": 44}])
            self.assertEqual(list(Path(directory).glob(".generation-*")), [])

    def test_corrupt_provenance_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            for content in ("{", "{}", '[{"sample_id":1},{"sample_id":1}]', '[{"seed":42}]',
                            '[{"sample_id":1,"seed":NaN}]'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_generation_metadata(path)
                self.assertEqual(path.read_text(), content)

    def test_cli_validates_external_layout_before_model_and_preserves_noop_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "data/samples/000000"
            sample.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(sample / "image.png")
            (sample / "meta.json").write_text(json.dumps({
                "sample_id": 0, "n_objects": 1, "prompt": "one cup",
                "instances": [{"bbox": [0, 0, 1, 1]}],
            }))
            layouts, output = root / "layouts", root / "output"
            layouts.mkdir()
            output.mkdir()
            (layouts / "000000.json").write_text(json.dumps({"prompt": [], "instances": []}))
            command = [sys.executable, str(ROOT / "scripts/inference.py"),
                       "--data_root", str(root / "data"), "--lora_path", str(root / "absent_weights"),
                       "--output_dir", str(output), "--layout_dir", str(layouts), "--save_metadata",
                       "--start_idx", "0", "--end_idx", "1"]
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("layout prompt must be a nonempty string", result.stderr)
            self.assertNotIn("Loading FLUX pipeline", result.stdout)
            Image.new("RGB", (4, 4)).save(output / "000000.png")
            metadata_path = output / "_generation_metadata.json"
            content = '[{"sample_id": 0, "seed": 99, "layout_source": "shared"}]\n'
            metadata_path.write_text(content)
            result = subprocess.run(command + ["--resume"], capture_output=True, text=True,
                                    env=environment, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Loading FLUX pipeline", result.stdout)
            self.assertEqual(metadata_path.read_text(), content)


if __name__ == "__main__":
    unittest.main()
