"""The four validation fits cannot see the shared salt-0 score files."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib

from my_experiments.baseline_model import split_train_val
from my_experiments.cross_salt_split import select_fit_files
from my_experiments.oof_split import names_hash
from toolkit.records import file_hash


class CrossSaltSplitTest(unittest.TestCase):
    def test_exact_roles_and_changed_manifest_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero = [f"shot_{index:05d}.parquet" for index in range(7041)]
            score = zero[-70:]
            reference = root / "reference.joblib"
            joblib.dump({"train_files": zero[:5633], "val_files": zero[5633:-70]}, reference)
            dataset = root / "dataset.json"
            dataset.write_text("pinned input identity\n")
            record: dict[str, Any] = {
                "corpus_salt0_ordered_sha256": names_hash(zero),
                "pinned_dataset_manifest_sha256": file_hash(dataset),
                "reference_artifact_sha256": file_hash(reference),
                "score_files": score,
                "score_ordered_sha256": names_hash(score),
                "salts": {},
            }
            orders: dict[int, list[str]] = {}
            for salt in (1, 2, 3, 4):
                order = random.Random(salt).sample(zero, len(zero))
                remaining = [name for name in order if name not in set(score)]
                train_names, validation_names = remaining[:5633], remaining[5633:]
                holdout = [name for name in order if name in set(score)]
                record["salts"][str(salt)] = {
                    "train_files": train_names,
                    "train_ordered_sha256": names_hash(train_names),
                    "validation_files": validation_names,
                    "validation_ordered_sha256": names_hash(validation_names),
                    "holdout_ordered_sha256": names_hash(holdout),
                }
                orders[salt] = order
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(record))
            with patch.dict("os.environ", {"FUSION_CROSS_SALT_MANIFEST_SHA256":
                                        file_hash(manifest)}):
                for salt in (1, 2, 3, 4):
                    fit_paths, val_paths, held_paths = select_fit_files(
                        manifest, [Path(name) for name in orders[salt]],
                        [Path(name) for name in zero], reference, dataset, salt)
                    self.assertEqual((len(fit_paths), len(val_paths), len(held_paths)),
                                     (5633, 1338, 70))
                    self.assertFalse({path.name for path in fit_paths + val_paths} & set(score))
                record["salts"]["1"]["validation_files"][0] = score[0]
                manifest.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    select_fit_files(manifest, [Path(name) for name in orders[1]],
                                     [Path(name) for name in zero], reference, dataset, 1)

    def test_ordinary_split_still_requires_a_tail(self) -> None:
        files = [Path(f"shot_{index:05d}.parquet") for index in range(7041)]
        train, validation = split_train_val(files, 0.80, 0.19)
        self.assertEqual((len(train), len(validation)), (5633, 1338))
        with self.assertRaisesRegex(SystemExit, "leaves no tail"):
            split_train_val(files, 5633 / 6971, 1338 / 6971)


if __name__ == "__main__":
    unittest.main()
