"""Pinned public manifests and streaming blend arithmetic require no target columns."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import validate_submission
from toolkit.records import file_hash, write_json
from tools.historical_compare import _save
from tools.public_blend import assemble, prepare_manifest


def _spec(dataset: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "sources": [{"name": f"m{i}", "artifact_sha256": f"artifact-{i}"}
                    for i in range(6)],
        "psi_member_weights": [5, 1, 1, 1, 2, 1],
        "q95_betaN_source_indices": [0, 1, 2, 3],
        "q95_betaN_member_weights": [5, 1, 1, 1],
    }


class PublicBlendChecks(unittest.TestCase):
    def test_public_manifest_uses_sorted_input_only_parquet_footers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/diii_d_public_test"
            folder.mkdir(parents=True)
            for name, times in (("shot_b.parquet", [0.0, 0.02, 0.04]),
                                ("shot_a.parquet", [0.0, 0.02])):
                pd.DataFrame([{"source": "DIII-D", "efit_times": times}]).to_parquet(folder / name)
            names = ["shot_a.parquet", "shot_b.parquet"]
            revision = "a" * 40
            manifest = root / "pinned.json"
            download = root / "download.json"
            write_json(manifest, {"revision": revision, "files": [
                {"path": f"data/diii_d_public_test/{name}",
                 "size": (folder / name).stat().st_size, "oid": f"sha256:{name}"}
                for name in names]})
            write_json(download, {"revision": revision, "sha256_verified": True})
            dataset: dict[str, object] = {
                "config": "diii_d_public_test", "revision": revision, "repo_id": "test",
                "shots": 2, "frames": 5,
                "ordered_filenames_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                "local_dataset_manifest": str(manifest),
                "local_dataset_manifest_sha256": file_hash(manifest),
                "local_download_record": str(download),
                "local_download_record_sha256": file_hash(download),
            }
            spec = root / "spec.json"
            write_json(spec, _spec(dataset))
            result = prepare_manifest(spec, root, root / "public_manifest.json")
            self.assertEqual(result["files"], names)
            self.assertEqual(result["frames"], [2, 3])
            self.assertTrue(result["target_columns_absent_in_all_files"])

    def test_streaming_final_npz_uses_eleven_maps_and_eight_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/diii_d_public_test"
            folder.mkdir(parents=True)
            pd.DataFrame([{"source": "DIII-D", "efit_times": [0.0, 0.02]}]).to_parquet(
                folder / "shot.parquet")
            manifest = {"files": ["shot.parquet"], "frames": [2]}
            spec = _spec({"config": "diii_d_public_test"})
            inputs = []
            for i in range(6):
                path = root / f"model_{i}.npz"
                value = float(i + 1)
                _save(path, {
                    "shot_0000_psirz": np.full((2, 65, 65), value, np.float32),
                    "shot_0000_q95": np.full(2, value, np.float32),
                    "shot_0000_betaN": np.full(2, value, np.float32),
                })
                write_json(path.with_suffix(".json"), {
                    "files": ["shot.parquet"],
                    "prediction_sha256": file_hash(path),
                    "artifact_sha256": f"artifact-{i}",
                })
                inputs.append(path)
            output = root / "final.npz"
            record = assemble(spec, manifest, inputs, output)
            self.assertEqual(record["shots"], 1)
            with np.load(output, allow_pickle=False) as packed:
                self.assertEqual(packed.files, ["shot_0000_psirz", "shot_0000_q95",
                                                "shot_0000_betaN"])
                self.assertEqual(packed["shot_0000_psirz"].dtype, np.float32)
                self.assertEqual(packed["shot_0000_psirz"][0, 0, 0], np.float32(30 / 11))
                self.assertEqual(packed["shot_0000_q95"][0], np.float32(14 / 8))
                self.assertEqual(packed["shot_0000_betaN"][0], np.float32(14 / 8))
            self.assertEqual(validate_submission.validate(output, "diii_d_public_test", 0,
                                                          "local", root), 0)


if __name__ == "__main__":
    unittest.main()
