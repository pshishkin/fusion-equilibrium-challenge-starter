"""Offline contract tests for private single-file publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tools import publish_private_candidate as publisher


class FakeApi:
    """A small in-memory private dataset with independent read/write credentials."""

    def __init__(self, token: str, state: dict[str, object]) -> None:
        self.token = token
        self.state = state

    def whoami(self, token: str) -> dict[str, object]:
        assert token == self.token
        permission = "repo.write" if token == "secret-write" else "repo.read"
        return {
            "name": "pshishkin",
            "auth": {
                "accessToken": {
                    "role": "fineGrained",
                    "fineGrained": {
                        "scoped": [
                            {
                                "entity": {"name": publisher.REPO},
                                "permissions": [permission],
                            }
                        ]
                    },
                }
            },
        }

    def repo_info(self, repo: str, **kwargs: object) -> SimpleNamespace:
        assert repo == publisher.REPO
        assert kwargs["token"] == self.token
        files = self.state["files"]
        assert isinstance(files, dict)
        siblings = [
            SimpleNamespace(
                rfilename=name,
                size=data[0],
                blob_id=data[1],
                lfs=SimpleNamespace(sha256=data[2]),
            )
            for name, data in files.items()
        ]
        return SimpleNamespace(
            private=True,
            sha=self.state["sha"],
            siblings=siblings,
        )

    def upload_file(self, **kwargs: object) -> SimpleNamespace:
        assert self.token == "secret-write"
        assert kwargs["path_in_repo"] == publisher.FILENAME
        assert kwargs["parent_commit"] == self.state["sha"]
        assert kwargs["token"] == "secret-write"
        assert "float16-probe" in str(kwargs["commit_message"])
        self.state["calls"] = cast(int, self.state["calls"]) + 1
        source = Path(str(kwargs["path_or_fileobj"]))
        files = self.state["files"]
        assert isinstance(files, dict)
        files[publisher.FILENAME] = (
            source.stat().st_size,
            "new-blob",
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        self.state["sha"] = "b" * 40
        return SimpleNamespace(oid="b" * 40)


class PublishPrivateCandidateTests(unittest.TestCase):
    def test_single_file_upload_preserves_mast_and_creates_private_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / publisher.FILENAME
            candidate.write_bytes(b"float32-candidate")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            evidence = root / "evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "variant": "float16-flux-only-probe",
                        "final_npz": {
                            "path": str(candidate),
                            "bytes": candidate.stat().st_size,
                            "sha256": digest,
                        },
                    }
                )
            )
            dotenv = root / ".env"
            dotenv.write_text("HF_WRITE_TOKEN=secret-write\nHF_READ_TOKEN=secret-read\n")
            dotenv.chmod(0o600)
            state: dict[str, object] = {
                "sha": "a" * 40,
                "files": {"mast_public_test.npz": (4, "mast-blob", "mast-sha")},
                "calls": 0,
            }

            def download(repo: str, filename: str, **kwargs: object) -> str:
                self.assertEqual(repo, publisher.REPO)
                self.assertEqual(filename, publisher.FILENAME)
                self.assertEqual(kwargs["token"], "secret-read")
                self.assertEqual(kwargs["revision"], "b" * 40)
                output = Path(str(kwargs["local_dir"])) / filename
                output.write_bytes(candidate.read_bytes())
                return str(output)

            with (
                patch.object(publisher, "EVIDENCE", evidence),
                patch.object(publisher, "POINTER_DIR", root / "submissions"),
                patch("tools.publish_private_candidate.tempfile.TemporaryDirectory") as temporary,
            ):
                temporary.return_value.__enter__.return_value = str(root)
                receipt = publisher.publish(
                    api_type=lambda token: FakeApi(token, state),
                    download=download,
                    dotenv=dotenv,
                    receipt_path=root / "receipt.json",
                    expected_parent="a" * 40,
                    variant="float16-probe",
                )
            self.assertEqual(state["calls"], 1)
            files = cast(dict[str, tuple[int, str, str]], state["files"])
            self.assertEqual(files["mast_public_test.npz"], (4, "mast-blob", "mast-sha"))
            self.assertEqual(receipt["revision"], "b" * 40)
            self.assertEqual(receipt["parent_revision"], "a" * 40)
            self.assertEqual(receipt["variant"], "float16-probe")
            self.assertNotIn("secret-read", (root / "receipt.json").read_text())
            self.assertNotIn("secret-write", (root / "receipt.json").read_text())
            pointer = Path(str(receipt["pointer_path"]))
            self.assertTrue(pointer.name.startswith("submission_pointer_f16_probe_"))
            with zipfile.ZipFile(pointer) as archive:
                self.assertEqual(archive.namelist(), ["manifest.json"])
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest,
                {
                    "repo_id": publisher.REPO,
                    "revision": "b" * 40,
                    "token": "secret-read",
                },
            )
            self.assertEqual(pointer.stat().st_mode & 0o777, 0o600)

    def test_read_only_write_token_fails_before_upload(self) -> None:
        state: dict[str, object] = {"sha": "a" * 40, "files": {}, "calls": 0}
        with self.assertRaisesRegex(publisher.PublishError, "lacks exact.*write"):
            publisher._preflight(
                FakeApi("secret-read", state),
                FakeApi("secret-read", state),
                "secret-read",
                "secret-read",
            )
        self.assertEqual(state["calls"], 0)

    def test_insecure_dotenv_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            dotenv = Path(folder) / ".env"
            dotenv.write_text("HF_WRITE_TOKEN=secret-write\nHF_READ_TOKEN=secret-read\n")
            os.chmod(dotenv, 0o644)
            with self.assertRaisesRegex(publisher.PublishError, "readable by group"):
                publisher._credentials(dotenv)

    def test_parent_guard_blocks_upload_before_remote_mutation(self) -> None:
        state: dict[str, object] = {
            "sha": "a" * 40,
            "files": {"mast_public_test.npz": (4, "mast-blob", "mast-sha")},
            "calls": 0,
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(publisher, "_candidate", return_value=(Path("unused"), 1, "0" * 64)),
            patch.object(publisher, "_credentials", return_value=("secret-write", "secret-read")),
            self.assertRaisesRegex(publisher.PublishError, "parent changed"),
        ):
            publisher.publish(
                api_type=lambda token: FakeApi(token, state),
                download=lambda *args, **kwargs: "unused",
                receipt_path=Path(folder) / "receipt.json",
                expected_parent="c" * 40,
                variant="float16-probe",
            )
        self.assertEqual(state["calls"], 0)


if __name__ == "__main__":
    unittest.main()
