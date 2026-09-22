"""The local and streamed submission paths retain float32 flux and reject float16 caches."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import submission_skeleton as submit
import validate_submission
from tools import blend_key, blend_salts
from tools.submission_format import EXPORT_FORMAT_VERSION, check_file


def _prediction(value: float = 0.123456) -> dict[str, np.ndarray]:
    return {"psirz": np.full((2, 65, 65), value, dtype=np.float32),
            "q95": np.array([2.123456, 3.456789], dtype=np.float32),
            "betaN": np.array([0.234567, 0.654321], dtype=np.float32)}


class SubmissionPrecisionChecks(unittest.TestCase):
    def test_end_to_end_local_and_streamed_npz_validate_as_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/diii_d_public_test"
            folder.mkdir(parents=True)
            row = {"efit_times": np.array([0.0, 0.02]), "source": "DIII-D"}
            pd.DataFrame([row]).to_parquet(folder / "shot.parquet")
            predicted = _prediction()
            local_out, stream_out = root / "local", root / "stream"
            local_out.mkdir()
            stream_out.mkdir()
            with patch.object(submit, "your_model_predict", return_value=predicted):
                local = submit.build_submission("diii_d_public_test", "public_test", local_out,
                                                0, "local", root, None, 1)
            with (patch.object(submit, "iter_dataset_rows", return_value=iter([row])),
                  patch.object(submit, "your_model_predict", return_value=predicted)):
                streamed = submit.build_submission("diii_d_public_test", "public_test",
                                                   stream_out, 0, "hf", root, None, 0)
            for path in (local, streamed):
                self.assertEqual(validate_submission.validate(path, "diii_d_public_test", 0,
                                                              "local", root), 0)
                check_file(path)
                with np.load(path, allow_pickle=False) as packed:
                    self.assertEqual(packed["shot_0000_psirz"].dtype, np.float32)
                    self.assertEqual(packed["shot_0000_psirz"].tobytes(),
                                     predicted["psirz"].tobytes())

    def test_parallel_task_and_streaming_path_keep_float32(self) -> None:
        row = {"efit_times": np.array([0.0, 0.02]), "source": "DIII-D"}
        frame = pd.DataFrame([row])
        predicted = _prediction()
        with (patch("pandas.read_parquet", return_value=frame),
              patch.object(submit, "your_model_predict", return_value=predicted)):
            local = submit.predict_one((Path("shot.parquet"), None))
        self.assertEqual(local["psirz"].dtype, np.float32)
        self.assertEqual(local["psirz"].tobytes(), predicted["psirz"].tobytes())
        self.assertEqual(local["q95"].dtype, np.float32)
        with (patch.object(submit, "iter_dataset_rows", return_value=iter([row])),
              patch.object(submit, "your_model_predict", return_value=predicted)):
            streamed = list(submit._predictions("diii_d_public_test", "public_test", "hf",
                                                Path("/tmp"), 1, None, 0))
        self.assertEqual(streamed[0]["psirz"].dtype, np.float32)
        self.assertEqual(streamed[0]["psirz"].tobytes(), predicted["psirz"].tobytes())

    def test_mast_cache_key_versions_export_and_rejects_old_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(submit, "EXPORT_FORMAT_VERSION", "old-format"):
                old_key = submit._mast_cache_key("mast_public_test", root, 1)
            new_key = submit._mast_cache_key("mast_public_test", root, 1)
            self.assertNotEqual(old_key, new_key)
            cache = root / "cache"
            cached = cache / new_key / "mast_public_test.npz"
            cached.parent.mkdir(parents=True)
            np.savez_compressed(cached,
                                shot_0000_psirz=np.zeros((2, 65, 65), dtype=np.float16),
                                shot_0000_q95=np.zeros(2, dtype=np.float32),
                                shot_0000_betaN=np.zeros(2, dtype=np.float32))
            with (patch.object(submit, "MAST_CACHE", cache),
                  patch.object(submit, "_predictions",
                               side_effect=AssertionError("old cache must not be rebuilt")),
                  self.assertRaisesRegex(ValueError, "mast_public_test.npz: "
                                         "shot_0000_psirz dtype float16, expected float32")):
                submit.build_submission("mast_public_test", "public_test", root / "out", 1,
                                        "local", root, None, 1)

    def test_per_salt_blend_checks_dtype_and_casts_once_to_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = [root / "s1", root / "s2"]
            for folder, value in zip(inputs, (0.123456, 0.123458), strict=True):
                folder.mkdir()
                pred = _prediction(value)
                np.savez_compressed(folder / "diii_d_public_test.npz",
                                    shot_0000_psirz=pred["psirz"],
                                    shot_0000_q95=pred["q95"],
                                    shot_0000_betaN=pred["betaN"])
            output = root / "out"
            with patch.object(sys, "argv", ["blend_salts.py", "--inputs",
                                            *[str(p) for p in inputs], "--out", str(output),
                                            "--configs", "diii_d_public_test"]):
                self.assertEqual(blend_salts.main(), 0)
            dst = output / "diii_d_public_test.npz"
            check_file(dst)
            with np.load(dst, allow_pickle=False) as packed:
                expected = np.float32((0.12345600128173828 + 0.12345799803733826) / 2)
                self.assertEqual(packed["shot_0000_psirz"].dtype, np.float32)
                self.assertEqual(packed["shot_0000_psirz"][0, 0, 0], expected)
            pred = _prediction()
            bad = inputs[1] / "diii_d_public_test.npz"
            np.savez_compressed(bad, shot_0000_psirz=pred["psirz"].astype(np.float16),
                                shot_0000_q95=pred["q95"], shot_0000_betaN=pred["betaN"])
            with (patch.object(sys, "argv", ["blend_salts.py", "--inputs",
                                             *[str(p) for p in inputs], "--out", str(root / "bad"),
                                             "--configs", "diii_d_public_test"]),
                  self.assertRaisesRegex(ValueError, "diii_d_public_test.npz: "
                                         "shot_0000_psirz dtype float16, expected float32")):
                blend_salts.main()

    def test_blend_key_includes_export_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "baseline_single_s0.joblib").write_bytes(b"fixed weights")
            command = ["blend_key.py", "--salt", "0", "--pct", "75"]
            def run_key() -> str:
                captured = io.StringIO()
                with patch.object(sys, "argv", command), contextlib.redirect_stdout(captured):
                    self.assertEqual(blend_key.main(), 0)
                return captured.getvalue().strip()
            with (patch.object(blend_key, "LANES", root),
                  patch.object(blend_key, "COMPOSITIONS", {75: [("single", 1)]})):
                current = run_key()
                with patch.object(blend_key, "EXPORT_FORMAT_VERSION", "old-format"):
                    old = run_key()
            self.assertNotEqual(current, old)
            self.assertIn("s0_top75_", current)
            self.assertEqual(EXPORT_FORMAT_VERSION, "psirz-float32-scalars-float32-v1")


if __name__ == "__main__":
    unittest.main()
