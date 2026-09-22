"""Protect split identity, frozen campaigns, resume integrity and scorer JSON output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_experiments import data_check
from my_experiments import run as runner
from toolkit.records import file_hash, write_json


class DatasetChecks(unittest.TestCase):
    def test_missing_changed_and_unexpected_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/diii_d_train"
            folder.mkdir(parents=True)
            shot = folder / "shot.parquet"
            shot.write_bytes(b"PAR1contentPAR1")
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "revision": "test",
                    "configs": {
                        "diii_d_train": {shot.name: [shot.stat().st_size, file_hash(shot)]}
                    },
                },
            )
            with patch.object(data_check, "MANIFEST", manifest):
                self.assertEqual(data_check.check_dataset(root, verify_hashes=True)["files"], 1)
                shot.write_bytes(b"PAR1changedPAR1")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    data_check.check_dataset(root, verify_hashes=True)
                shot.write_bytes(b"partial")
                with self.assertRaisesRegex(ValueError, "bytes, expected"):
                    data_check.check_dataset(root)
                shot.unlink()
                with self.assertRaisesRegex(ValueError, "0 parquet files, expected 1"):
                    data_check.check_dataset(root)
                (folder / "unexpected.parquet").write_bytes(b"PAR1PAR1")
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    data_check.check_dataset(root)


class CampaignChecks(unittest.TestCase):
    def test_submission_helper_imports_from_isolated_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = runner.prepare(
                runner.ROOT / "experiments/smoke.json", Path(tmp) / "runs", jobs=1
            )
            source = campaign / "source"
            script = (
                "from pathlib import Path\n"
                "import sys\n"
                "sys.path.insert(0, str(Path.cwd()))\n"
                "import submission_skeleton\n"
                "import tools.submission_format\n"
                "root = Path.cwd().resolve()\n"
                "assert Path(submission_skeleton.__file__).resolve().is_relative_to(root)\n"
                "assert Path(tools.submission_format.__file__).resolve().is_relative_to(root)\n"
            )
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-I", "-c", script], cwd=source, env=env,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_other_pid_namespace_is_not_mistaken_for_a_dead_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope = runner.process_scope()
            write_json(root / "status.json", {
                "state": "running", "pgid": 999999999,
                "process_scope": {**scope, "pid_namespace": "another namespace"},
            })
            with self.assertRaisesRegex(RuntimeError, "worker PID namespace"):
                runner.active_group(root)

    def test_execute_uses_snapshot_and_resumes_only_verified_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "my_experiments/train_eval.py"
            source.parent.mkdir()
            source.write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(os.environ['FUSION_ARTIFACT']).write_bytes(b'frozen fit')\n"
                "path = pathlib.Path(sys.argv[sys.argv.index('--metrics-json') + 1])\n"
                "path.write_text(json.dumps({'results': {'ensemble': {'S': 0.5}}}))\n"
            )
            spec = root / "spec.json"
            write_json(spec, {
                "name": "integration", "params": str(runner.ROOT / "params.yaml"),
                "profile": "smoke", "salts": [0], "jobs": 1,
                "arms": [{"name": "base", "only": ["mlp"], "set": [],
                          "score_models": ["ensemble"]}],
            })
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner, "source_files", return_value=[source]),
                patch.object(runner.subprocess, "check_output", return_value="revision"),
                patch.object(runner, "check_dataset", return_value={"files": 7041}),
                patch.object(runner, "check_device", return_value={"device": "stub"}),
            ):
                campaign = runner.prepare(spec, root / "runs")
                source.write_text("raise RuntimeError('working tree must not be executed')\n")
                runner.execute(campaign, root)
                folder = campaign / "base_s0"
                self.assertTrue(runner.completed(folder))
                artifact = folder / "baseline.joblib"
                before = artifact.stat().st_mtime_ns
                runner.execute(campaign, root)
                self.assertEqual(before, artifact.stat().st_mtime_ns)
                artifact.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "completed record expects"):
                    runner.execute(campaign, root)

    def test_prepare_freezes_source_and_config_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "example.py"
            source.write_text("original\n")
            spec = root / "experiment.json"
            write_json(
                spec,
                {
                    "name": "test",
                    "params": str(runner.ROOT / "params.yaml"),
                    "profile": "smoke",
                    "salts": [0],
                    "jobs": 1,
                    "arms": [
                        {
                            "name": "base",
                            "only": ["mlp"],
                            "set": ["models.mlp.device=cpu"],
                            "score_models": ["ensemble"],
                        }
                    ],
                },
            )
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner, "source_files", return_value=[source]),
                patch.object(runner.subprocess, "check_output", return_value="revision"),
            ):
                campaign = runner.prepare(spec, root / "runs")
                self.assertEqual(campaign, runner.prepare(spec, root / "runs"))
                source.write_text("workspace changed\n")
                self.assertEqual((campaign / "source/example.py").read_text(), "original\n")
                runner.verify_source(campaign)
                second = runner.prepare(spec, root / "runs")
                self.assertNotEqual(second, campaign)
                (campaign / "base_s0/params.yaml").write_text("tampered\n")
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    runner.verify_source(campaign)

    def test_success_requires_artifact_and_metrics_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(runner.completed(folder))
            artifact = folder / "baseline.joblib"
            artifact.write_bytes(b"fit")
            (folder / "params.yaml").write_text("frozen")
            write_json(folder / "metrics.json", {"results": {"ensemble": {"S": 0.9}}})
            write_json(
                folder / "complete.json",
                {
                    "sha256": {
                        p.name: file_hash(p)
                        for p in (artifact, folder / "metrics.json", folder / "params.yaml")
                    }
                },
            )
            self.assertTrue(runner.completed(folder))
            artifact.write_bytes(b"another fit")
            with self.assertRaisesRegex(ValueError, "completed record expects"):
                runner.completed(folder)

    def test_failed_process_cannot_publish_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(runner, "ROOT", root),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                runner.stream_process(
                    [sys.executable, "-c", 'print("visible"); raise SystemExit(7)'],
                    root,
                    dict(os.environ),
                    root / "run.log",
                    root / "status.json",
                )
            self.assertEqual(runner.read_json(root / "status.json")["returncode"], 7)
            self.assertIn("visible", (root / "run.log").read_text())
            self.assertFalse(runner.completed(root))

    def test_paired_summary_refuses_different_scored_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                {"name": "base_s0", "arm": "base", "salt": 0},
                {"name": "treat_s0", "arm": "treat", "salt": 0},
            ]
            for run, files in zip(runs, [["shot-a"], ["shot-b"]], strict=True):
                write_json(
                    root / run["name"] / "metrics.json",
                    {"files": files, "results": {"ensemble": {"S": 0.9}}},
                )
            with (
                patch.object(
                    runner,
                    "verify_source",
                    return_value={"identity": {"runs": runs}, "source_hash": "code"},
                ),
                patch.object(runner, "completed", return_value=True),
                self.assertRaisesRegex(ValueError, "different files"),
            ):
                runner.summary(root, "base")


class ScorerJson(unittest.TestCase):
    def test_native_results_are_written_without_another_metric(self) -> None:
        import numpy as np

        import local_score

        shot = {"psi": np.ones((1, 2, 2)), "q95": np.ones(1), "betaN": np.ones(1)}
        ref = (None, np.ones((1, local_score.N_CONS)), np.ones((1, local_score.N_CONS), dtype=bool))
        result = {"S": 0.123456, "dlcfs": 0.012345, "r2_cons_each": {"li": None}}
        coords = np.arange(2)
        mask = np.ones((2, 2), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.json"
            with (
                patch.object(local_score, "load_shots", return_value=[shot]),
                patch.object(
                    local_score, "_machine_of", return_value=("DIII-D", coords, coords, mask, mask)
                ),
                patch.object(local_score, "_map", side_effect=[[ref], []]),
                patch.object(local_score, "predict", return_value=[{}]),
                patch.object(local_score, "choose_psi_sign", return_value=1),
                patch.object(local_score, "finalize_machine", return_value=result),
                patch.object(local_score, "report"),
            ):
                local_score.main(
                    [
                        "--files",
                        "shot.parquet",
                        "--jobs",
                        "1",
                        "--models",
                        "ensemble",
                        "--metrics-json",
                        str(output),
                    ]
                )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["results"]["ensemble"], result)
            self.assertEqual(payload["files"], ["shot.parquet"])


class OverlapChecks(unittest.TestCase):
    def test_augmented_blend_checks_candidate_training_files(self) -> None:
        from my_experiments import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "baseline.joblib"
            artifact.touch()
            score_file = root / "unseen.parquet"
            base = {
                "split_salt": 0,
                "train_files": ["train.parquet"],
                "val_files": ["val.parquet"],
                "n_train_shots": 1,
                "n_val_shots": 1,
                "train_share": 0.8,
                "val_share": 0.15,
            }
            candidate = {**base, "train_files": [score_file.name]}
            with (
                patch.object(evaluate, "ARTIFACT", artifact),
                patch.object(evaluate.joblib, "load", side_effect=[base, candidate]),
                patch.object(evaluate, "sorted_shots", return_value=[score_file]),
                patch.object(evaluate, "take_share", return_value=[score_file]),
                patch.object(evaluate, "model_names", return_value=["ensemble"]),
                patch.object(
                    sys, "argv", ["evaluate", "--share", "0.05", "--models", "salts:candidate"]
                ),
                patch.object(evaluate.local_score, "main") as score,
            ):
                with self.assertRaisesRegex(SystemExit, "Splits overlap: 1"):
                    evaluate.main()
                score.assert_not_called()


class AuxiliaryAlignment(unittest.TestCase):
    def test_thinning_restarts_at_each_shot_boundary(self) -> None:
        import numpy as np

        from my_experiments import aux_targets
        from my_experiments.baseline_model import thin_frames

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / "a.parquet", root / "b.parquet"]
            for f in files:
                f.write_bytes(b"source")
            with patch.object(aux_targets, "CACHE", root):
                cache = aux_targets.cache_path(files)
                truth = np.arange(17 * 7).reshape(17, 7)
                np.savez(cache, cons=truth, lengths=[10, 7])
                actual = aux_targets.load(files, 0.2)
                expected = np.concatenate(
                    [truth[:10][thin_frames(10, 0.2)], truth[10:][thin_frames(7, 0.2)]]
                )
                np.testing.assert_array_equal(actual, expected)
                np.savez(cache, cons=truth, lengths=[9, 7])
                with self.assertRaisesRegex(ValueError, "expected 2 shots/17 frames"):
                    aux_targets.load(files)


class ArtifactSelection(unittest.TestCase):
    def test_single_external_artifact_is_a_valid_candidate(self) -> None:
        import numpy as np

        from my_experiments import baseline_model as baseline

        predict = baseline.predict_row
        expected = {"psirz": np.ones((1, 65, 65)), "q95": np.ones(1), "betaN": np.ones(1)}
        with (
            patch.object(baseline, "artifact_path", return_value=Path("candidate.joblib")),
            patch.object(baseline, "predict_row", return_value=expected) as nested,
        ):
            self.assertIs(predict({}, model="salts:candidate*5"), expected)
            nested.assert_called_once_with({}, "DIII-D", "ensemble", Path("candidate.joblib"))
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                predict({}, model="salts:candidate*nan")


if __name__ == "__main__":
    unittest.main()
