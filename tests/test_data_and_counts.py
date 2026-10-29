"""Dataset contract and strict category-count evaluation regressions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aibl.dataset import OverlapDepth45KDataset
from aibl.counting import evaluate_counting


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sample = self.root / "samples/000000"
        self.sample.mkdir(parents=True)
        Image.new("RGB", (4, 4), "white").save(self.sample / "image.png")
        self.meta = {"sample_id": 0, "n_objects": 1, "prompt": "one cup",
                     "instances": [{"bbox": [0., 0., 1., 1.], "category": "cup"}]}
        self.write_meta()
        np.savez(self.sample / "masks.npz", amodal_0=np.ones((4, 4)),
                 modal_0=np.ones((4, 4)), depth_map=np.ones((4, 4)),
                 ownership_map=np.zeros((4, 4), dtype=np.int32))

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self):
        (self.sample / "meta.json").write_text(json.dumps(self.meta))

    def test_load_and_zero_end_index(self):
        dataset = OverlapDepth45KDataset(self.root)
        self.assertEqual(dataset[0]["n_objects"], 1)
        self.assertEqual(len(OverlapDepth45KDataset(self.root, start_idx=0, end_idx=0)), 0)

    def test_invalid_bbox_rejected(self):
        self.meta["instances"][0]["bbox"][0] = float("nan")
        self.write_meta()
        with self.assertRaises(ValueError):
            OverlapDepth45KDataset(self.root)[0]

    def test_pair_ids_are_distinct_from_positions(self):
        self.meta["n_objects"] = 2
        self.meta["instances"] = [
            {"id": 1, "bbox": [0., 0., 1., 1.]},
            {"id": 0, "bbox": [0., 0., 1., 1.]},
        ]
        self.meta["overlap_pairs"] = [{"front": 0, "back": 1}]
        self.write_meta()
        sample = OverlapDepth45KDataset(self.root, load_masks=False)[0]
        self.assertEqual(sample["overlap_pairs"], [{"front": 0, "back": 1}])
        self.meta["overlap_pairs"] = [{"front_id": 0, "back_id": 1}]
        self.write_meta()
        sample = OverlapDepth45KDataset(self.root, load_masks=False)[0]
        self.assertEqual(sample["overlap_pairs"], [{"front": 1, "back": 0}])

    def test_no_silent_missing_masks(self):
        (self.sample / "masks.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            OverlapDepth45KDataset(self.root)[0]
        self.assertEqual(OverlapDepth45KDataset(self.root, load_masks=False)[0]["n_objects"], 1)


class CountingTests(unittest.TestCase):
    def test_strict_relative_metrics_and_rmse(self):
        # One-count miss on target 3 fails both tolerances; on target 10 passes.
        metrics = evaluate_counting([2, 9, 0], [3, 10, 0])
        self.assertAlmostEqual(metrics.exact_accuracy, 1 / 3)
        self.assertAlmostEqual(metrics.accuracy_within_10_percent, 2 / 3)
        self.assertAlmostEqual(metrics.accuracy_within_20_percent, 2 / 3)
        self.assertAlmostEqual(metrics.root_mean_squared_error, np.sqrt(2 / 3))

    def test_length_and_integer_validation(self):
        for pred, gt in (([1], []), ([-1], [1]), ([1.5], [1]), ([True], [1])):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                evaluate_counting(pred, gt)


if __name__ == "__main__":
    unittest.main()
