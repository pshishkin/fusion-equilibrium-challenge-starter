"""Fast raw-current descriptors, strict support, slim parity, and sidecar identity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import local_score
import machines
from my_experiments import baseline_model
from my_experiments import fast_magnetics as fast
from my_experiments.baseline_model import thin_frames
from toolkit.records import file_hash

A = (("ECOILA", 5.0),)
B = (("ECOILA", 5.0), ("plasma_current", 20.0))


class _IdentityScaler:
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return matrix


class _ConstantMapPCA:
    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        return np.broadcast_to(coefficients[:, :1, None],
                               (len(coefficients), 65, 65)).copy()


def _prediction_from_fast_columns(artifact: dict[str, Any], model: str,
                                  scaled: np.ndarray) -> np.ndarray:
    """Exercise the appended columns through the ordinary prediction code path."""
    if scaled.shape[1] != 23:
        raise ValueError(f"expected 21 levels plus 2 fast columns, got {scaled.shape[1]}")
    return np.column_stack([scaled[:, -2], scaled[:, -2], scaled[:, -1]])


def _row() -> dict[str, Any]:
    """A short shot whose shipped plasma-current axis is deliberately mis-originated."""
    raw_time = np.arange(0.0, 240.5, 0.5)
    return {
        "efit_times": np.asarray([80.0, 100.0, 120.0, 140.0, 160.0]),
        "magnetics_time": raw_time,
        "magnetics_ECOILA": (2.0 * raw_time + 3.0).astype(np.float32),
        "magnetics_plasma_current_times": -858.1871 + raw_time,
        "magnetics_plasma_current": np.maximum(0.0, 100.0 - abs(raw_time - 120.0))
        .astype(np.float32),
    }


class FastMagneticsChecks(unittest.TestCase):
    def test_scorer_filename_context_and_mast_dispatch(self) -> None:
        with patch.object(local_score, "_shot_from_row", side_effect=ValueError(
            "row: ECOILA window outside raw support"
        )):
            with self.assertRaisesRegex(ValueError, "synthetic.parquet: row: ECOILA"):
                local_score._shot_with_context({}, "synthetic.parquet", ("mlp",))
            with self.assertRaisesRegex(ValueError, "streamed shot index 7: row: ECOILA"):
                local_score._shot_with_context({}, "streamed shot index 7", ("mlp",))
        row = {"source": "MAST"}
        with patch("mast.predict.slim_row", return_value={"source": "MAST"}) as slim_mast:
            self.assertEqual(machines.slim_row(row, model_specs=("salts:candidate",)),
                             {"source": "MAST"})
            slim_mast.assert_called_once_with(row)

    def test_model_full_and_slim_prediction_parity(self) -> None:
        row = _row()
        row.update({
            "source": "DIII-D",
            "efit_grid_R": np.linspace(0.84, 2.54, 65),
            "efit_grid_Z": np.linspace(-1.6, 1.6, 65),
            "coil_input_column": np.asarray(["magnetics_ECOILA"]),
            "coil_R": np.asarray([1.0]), "coil_Z": np.asarray([0.0]),
            "coil_width": np.asarray([0.1]), "coil_height": np.asarray([0.1]),
            baseline_model.FEATURES_KEY: np.zeros((5, baseline_model.FEATURE_WIDTH), np.float32),
        })
        plan = {
            "subtract": False, "inputs": "currents", "derivatives": "none",
            "frame_gaps": False, "vessel": "none", "thomson": "",
            "fast_magnetics": fast.spec_to_yaml_value(B),
        }
        artifact: dict[str, Any] = {
            "coil": plan, "n_pca": 1, "scaler": _IdentityScaler(),
            "pca": _ConstantMapPCA(),
        }
        with (patch.object(fast, "required_for_models", return_value=B),
              patch.object(baseline_model, "_load", return_value=artifact),
              patch.object(baseline_model, "_predict_targets",
                           side_effect=_prediction_from_fast_columns)):
            slim = baseline_model.slim_row(row, model_specs=("mlp",))
            for column in ("magnetics_time", "magnetics_ECOILA",
                           "magnetics_plasma_current"):
                self.assertNotIn(column, slim)
            full_prediction = baseline_model.predict_row(row, model="mlp")
            slim_prediction = baseline_model.predict_row(slim, model="mlp")
        for key in full_prediction:
            np.testing.assert_array_equal(full_prediction[key], slim_prediction[key])

    def test_spec_and_corrected_raw_derivative(self) -> None:
        spec = fast.normalize_spec(
            [{"signal": "ECOILA", "halfwidth_ms": 5.0},
             {"signal": "plasma_current", "halfwidth_ms": 20.0}], "test spec"
        )
        self.assertEqual(spec, B)
        row = _row()
        features = fast.compute_from_row(row, spec, "test.parquet")
        self.assertEqual(features.shape, (5, 2))
        np.testing.assert_allclose(features[:, 0], 2.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(features[:, 1], [1.0, 1.0, 0.0, -1.0, -1.0],
                                   rtol=0, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fast.normalize_spec([{"signal": "ECOILA", "halfwidth_ms": 5.0}] * 2,
                                "duplicate spec")
        with self.assertRaisesRegex(ValueError, "halfwidth_ms"):
            fast.normalize_spec([{"signal": "ECOILA", "halfwidth_ms": -1.0}], "bad spec")

    def test_strict_support_and_slim_union(self) -> None:
        row = _row()
        union = fast.slim_features(row, B, "test.parquet")
        slim = {"efit_times": row["efit_times"], fast.FAST_KEY: union}
        np.testing.assert_array_equal(fast.features_for_row(slim, A),
                                      fast.compute_from_row(row, A, "test.parquet"))
        np.testing.assert_array_equal(fast.features_for_row(slim, B),
                                      fast.compute_from_row(row, B, "test.parquet"))
        with self.assertRaisesRegex(ValueError, "slim row lacks fast feature"):
            fast.features_for_row({"efit_times": row["efit_times"], fast.FAST_KEY: {}}, A)
        boundary = dict(row, efit_times=np.asarray([2.0, 40.0]))
        with self.assertRaisesRegex(ValueError, "test.parquet.*1/2 windows outside raw support"):
            fast.compute_from_row(boundary, A, "test.parquet")
        duplicate = dict(row, magnetics_time=np.asarray(row["magnetics_time"]).copy())
        duplicate["magnetics_time"][1] = duplicate["magnetics_time"][0]
        with self.assertRaisesRegex(ValueError, "duplicate or decreasing"):
            fast.compute_from_row(duplicate, A, "test.parquet")

    def test_sidecar_full_count_thinning_and_config_invalidation(self) -> None:
        """The cache stores all frames, then applies base training's per-shot indices."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "diii_d_train"
            data.mkdir(parents=True)
            path = data / "synthetic.parquet"
            pd.DataFrame({key: [value] for key, value in _row().items()}).to_parquet(path)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"configs": {"diii_d_train": {
                path.name: [path.stat().st_size, file_hash(path)]
            }}}))
            lock = root / "uv.lock"
            lock.write_text("first dependency identity\n")
            with (patch.object(fast, "CACHE", root / "sidecars"),
                  patch.object(fast, "MANIFEST", manifest),
                  patch.object(fast, "LOCK", lock)):
                fast._manifest_doc.cache_clear()
                fast._producer_identity.cache_clear()
                first, counts = fast.load_many([path], 0.6, A, jobs=1, label="sidecar test")
                self.assertEqual(counts.tolist(), [len(thin_frames(5, 0.6))])
                with np.load(fast._cache_path(path, A)) as saved:
                    self.assertEqual(int(saved["full_frames"]), 5)
                    np.testing.assert_array_equal(first, saved["features"][thin_frames(5, 0.6)])
                    old_key = str(saved["key"])
                second, counts_full = fast.load_many([path], 1.0, A, jobs=1,
                                                     label="sidecar hit")
                self.assertEqual(counts_full.tolist(), [5])
                np.testing.assert_array_equal(first, second[thin_frames(5, 0.6)])
                fast.load_many([path], 1.0, B, jobs=1, label="different spec")
                self.assertNotEqual(fast._cache_path(path, A), fast._cache_path(path, B))
                lock.write_text("changed dependency identity\n")
                fast._producer_identity.cache_clear()
                fast.load_many([path], 1.0, A, jobs=1, label="dependency invalidation")
                with np.load(fast._cache_path(path, A)) as saved:
                    self.assertNotEqual(str(saved["key"]), old_key)
            fast._manifest_doc.cache_clear()
            fast._producer_identity.cache_clear()

    def test_mixed_artifact_requirement_union_and_missing_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            main = folder / "baseline.joblib"
            main.touch()
            artifacts = {
                main: {"coil": {"fast_magnetics": []}},
                folder / "baseline_a.joblib": {"coil": {"fast_magnetics": [
                    {"signal": "ECOILA", "halfwidth_ms": 5.0}
                ]}},
                folder / "baseline_b.joblib": {"coil": {"fast_magnetics": [
                    {"signal": "ECOILA", "halfwidth_ms": 5.0},
                    {"signal": "plasma_current", "halfwidth_ms": 20.0}
                ]}},
            }
            with (patch.object(baseline_model, "ARTIFACT", main),
                  patch.object(baseline_model, "artifact_path",
                               side_effect=lambda name: folder / f"baseline_{name}.joblib"),
                  patch.object(baseline_model, "_load", side_effect=lambda path: artifacts[path])):
                self.assertEqual(fast.required_for_models(
                    ("ensemble", "salts:a+b", "salts:b")), B)
            main.unlink()
            with patch.object(baseline_model, "ARTIFACT", main):
                self.assertEqual(fast.required_for_models((None,)), ())
                with self.assertRaises(FileNotFoundError):
                    fast.required_for_models(("mlp",))


if __name__ == "__main__":
    unittest.main()
