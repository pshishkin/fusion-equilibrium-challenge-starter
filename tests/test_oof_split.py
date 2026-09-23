"""Fivefold filename proof and sealed gate tests; no competition labels."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from my_experiments import baseline_model
from my_experiments.oof_split import build_manifest, select_fit_files
from tools.oof_aux_cache import select_blocks
from tools.oof_normal_pilot import verify_pilot_artifact


class OofSplitTests(unittest.TestCase):
    def test_auxiliary_block_copy_keeps_order_and_nan_bits(self) -> None:
        source = np.arange(35, dtype=np.uint32).reshape(5, 7).copy()
        source[1, 2] = np.uint32(0x7FC12345)
        targets = source.view(np.float32)
        selected, lengths = select_blocks(
            targets, np.array([2, 1, 2]), ["a", "b", "c"], ["c", "a"])
        self.assertEqual(lengths.tolist(), [2, 2])
        np.testing.assert_array_equal(selected.view(np.uint32),
                                      source[[3, 4, 0, 1]])
        with self.assertRaisesRegex(ValueError, "absent from allowed cache"):
            select_blocks(targets, np.array([2, 1, 2]), ["a", "b", "c"], ["d"])

    def test_default_training_still_uses_original_split(self) -> None:
        corpus = [Path(f"shot_{index:04d}.parquet") for index in range(7041)]
        with (
            patch.object(baseline_model, "require_artifact_path",
                         return_value=Path("/tmp/oof-default-path.joblib")),
            patch("my_experiments.data_check.check_dataset", return_value={"files": 7041}),
            patch.object(baseline_model, "sorted_shots", return_value=corpus),
            patch.object(baseline_model, "split_train_val",
                         side_effect=RuntimeError("original split selected")) as split,
            self.assertRaisesRegex(RuntimeError, "original split selected"),
        ):
            baseline_model.train(
                "0.80/1.0", "0.19/1.0", Path("/tmp"), "diii_d_train",
                Path(__file__).resolve().parent.parent / "baselines/iteration.params.yaml",
                only=["mlp"],
            )
        self.assertEqual(split.call_count, 1)
        self.assertEqual(split.call_args.args[:3], (corpus, 0.80, 0.19))

    def test_fivefold_coverage_and_gate_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = [folder / f"shot_{index:04d}.parquet" for index in range(7041)]
            reference = folder / "reference.joblib"
            joblib.dump({
                "train_files": [path.name for path in paths[:5633]],
                "val_files": [path.name for path in paths[5633:6971]],
            }, reference)
            spec = folder / "spec.json"
            spec.write_text("{}\n")
            params = folder / "params.yaml"
            params.write_text("models: {}\n")
            manifest = build_manifest(paths, reference, spec, params)
            self.assertEqual([len(fold["oof_files"]) for fold in manifest["folds"]],
                             [1355, 1354, 1354, 1354, 1354])
            self.assertEqual(len(manifest["allowed_files"]), 6771)
            self.assertEqual(len(manifest["gate_files"]), 200)
            self.assertEqual(len(manifest["reserved_tail_files"]), 70)
            self.assertEqual(len({name for fold in manifest["folds"]
                                  for name in fold["oof_files"]}),
                             6771)
            self.assertTrue(set(manifest["pilot_train_oof_files"]).issubset(
                set(path.name for path in paths[:5633])))
            fold0_oof = set(manifest["folds"][0]["oof_files"])
            base = {
                "train_files": [name for name in manifest["allowed_files"]
                                if name not in fold0_oof],
                "val_files": [],
                "oof_files": manifest["folds"][0]["oof_files"],
                "oof_fold": 0,
                "oof_manifest_sha256": "manifest-hash",
                "source_sha256": "source-hash",
            }
            campaign = {"files_manifest_sha256": "manifest-hash",
                        "source_bundle_sha256": "source-hash"}
            verify_pilot_artifact(base, campaign, manifest, reference)
            base["train_files"][0] = manifest["gate_files"][0]
            with self.assertRaisesRegex(ValueError, "artifact overlap/provenance"):
                verify_pilot_artifact(base, campaign, manifest, reference)
            recorded = folder / "manifest.json"
            recorded.write_text(json.dumps(manifest))
            gate = set(manifest["gate_files"])
            tail = set(manifest["reserved_tail_files"])
            for fold_index in range(5):
                fitting, oof, reserved = select_fit_files(recorded, paths, reference, spec,
                                                          params, fold_index)
                self.assertEqual(len(fitting) + len(oof), 6771)
                self.assertFalse({path.name for path in fitting} &
                                 ({path.name for path in oof} | gate | tail))
                self.assertEqual([path.name for path in reserved],
                                 manifest["reserved_tail_files"])
            manifest["folds"][0]["oof_files"][0] = manifest["gate_files"][0]
            recorded.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest content/identity differs"):
                select_fit_files(recorded, paths, reference, spec, params, 0)

    def test_reference_order_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = [folder / f"shot_{index:04d}.parquet" for index in range(7041)]
            reference = folder / "reference.joblib"
            train = [path.name for path in paths[:5633]]
            train[0], train[1] = train[1], train[0]
            joblib.dump({"train_files": train,
                         "val_files": [path.name for path in paths[5633:6971]]}, reference)
            spec = folder / "spec.json"
            spec.write_text("{}\n")
            params = folder / "params.yaml"
            params.write_text("models: {}\n")
            with self.assertRaisesRegex(ValueError, r"ordered train\+validation"):
                build_manifest(paths, reference, spec, params)


if __name__ == "__main__":
    unittest.main()
