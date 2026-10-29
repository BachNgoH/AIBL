"""Paper names and recorded checkpoint settings stay consistent."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aibl.settings import migrate_checkpoint_config, validate_attention_settings, ATTENTION_DEFAULTS
from aibl.train_config import parse_args


class PaperInterfaceTests(unittest.TestCase):
    def test_public_training_arguments_follow_paper_terms(self):
        args = parse_args(["--data_root", "data", "--omega_intra", "3", "--beta", "2.5"])
        self.assertEqual(args.omega_intra, 3)
        self.assertEqual(args.beta, 2.5)
        self.assertEqual(args.alpha_max, 5)
        self.assertEqual(args.train_end_idx, 44000)
        self.assertTrue(args.use_layout_attention)
        self.assertTrue(args.use_overlap_ownership)

    def test_recorded_legacy_settings_migrate_without_changing_values(self):
        config = migrate_checkpoint_config({"intra_bbox_bias": 3, "inter_bbox_bias": -2,
                                             "gligen_boost": 2.5, "use_overlap_zones": False,
                                             "omega_loser": 0.3, "aibl_max_boost": 4})
        self.assertEqual(config["omega_intra"], 3)
        self.assertEqual(config["omega_inter"], -2)
        self.assertEqual(config["beta"], 2.5)
        self.assertEqual(config["alpha_max"], 4)
        self.assertFalse(config["use_overlap_ownership"])
        with self.assertRaises(ValueError):
            migrate_checkpoint_config({"use_overlap_zones": True, "omega_loser": 0.3})
        with self.assertRaises(ValueError):
            migrate_checkpoint_config({"intra_bbox_bias": 3, "omega_intra": 2})

    def test_malformed_saved_attention_settings_fail_before_model_load(self):
        for override in ({"use_layout_attention": "false"}, {"beta": float("nan")},
                         {"sigma": 0}, {"omega_inter": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_attention_settings(dict(ATTENTION_DEFAULTS, **override))


if __name__ == "__main__":
    unittest.main()
