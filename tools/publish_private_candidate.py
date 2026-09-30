"""Publish the verified DIII-D candidate without touching the private repo's MAST file.

Credentials are read from the gitignored .env in memory. Neither credential is accepted
as a CLI argument, included in a log/receipt, or written outside the pointer ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "pshishkin/fusion"
FILENAME = "diii_d_public_test.npz"
EVIDENCE = Path("results/public_diii_d_candidate_20260923.json")
RECEIPT = Path("results/private_hf_candidate_20260923.json")
POINTER_DIR = Path("submissions")
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class PublishError(Exception):
    """A sanitized publication failure with no credential-bearing response body."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _candidate(evidence: Path | None = None, variant: str = "float32") -> tuple[Path, int, str]:
    if evidence is None:
        evidence = EVIDENCE
    record = json.loads(evidence.read_text())
    if variant == "float16-probe" and record.get("variant") != "float16-flux-only-probe":
        raise PublishError(f"{evidence}: missing explicit float16 probe identity")
    if variant not in ("float32", "float16-probe"):
        raise PublishError(f"unsupported publication variant {variant}")
    item = record["final_npz"]
    path = Path(item["path"])
    if path.name != FILENAME or not path.is_file():
        raise PublishError("candidate is missing or has the wrong filename")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != item["bytes"] or digest != item["sha256"]:
        raise PublishError("candidate size or SHA-256 differs from validated evidence")
    return path, size, digest


def _credentials(dotenv: Path = Path(".env")) -> tuple[str, str]:
    if dotenv.is_symlink() or not dotenv.is_file():
        raise PublishError("secure .env file is missing or is a symlink")
    mode = stat.S_IMODE(dotenv.stat().st_mode)
    if mode & 0o077:
        raise PublishError(".env must not be readable by group or others")
    values: dict[str, str] = {}
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise PublishError(".env has a malformed assignment")
        if key in values:
            raise PublishError(f".env has duplicate {key}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    pair = (values.get("HF_WRITE_TOKEN", ""), values.get("HF_READ_TOKEN", ""))
    if not all(pair) or pair[0] == pair[1]:
        raise PublishError("distinct HF_WRITE_TOKEN and HF_READ_TOKEN are required")
    return pair


def _scope(who: dict[str, Any], permission: str) -> bool:
    auth = (who.get("auth") or {}).get("accessToken") or {}
    if auth.get("role") != "fineGrained":
        return False
    scopes = (auth.get("fineGrained") or {}).get("scoped") or []
    # Writing may also come from a scope over the owner's whole namespace (the repository was
    # recreated, and a namespace scope survives that); reading must stay scoped to this repo.
    owner = REPO.split("/")[0]
    return any(
        ((scope.get("entity") or {}).get("name") == REPO
         or (permission == "write" and (scope.get("entity") or {}).get("type") == "user"
             and (scope.get("entity") or {}).get("name") == owner))
        and any(permission in p for p in (scope.get("permissions") or []))
        for scope in scopes
    )


def _read_only(who: dict[str, Any]) -> bool:
    auth = (who.get("auth") or {}).get("accessToken") or {}
    if auth.get("role") != "fineGrained":
        return False
    scopes = (auth.get("fineGrained") or {}).get("scoped") or []
    permissions = [p for scope in scopes for p in (scope.get("permissions") or [])]
    return _scope(who, "read") and not any("write" in p or "admin" in p for p in permissions)


FileIdentity = tuple[int | None, str | None, str | None]


def _files(info: Any) -> dict[str, FileIdentity]:
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        raise PublishError("remote file metadata is unavailable")
    result = {}
    for sibling in siblings:
        lfs = getattr(sibling, "lfs", None)
        result[sibling.rfilename] = (
            getattr(sibling, "size", None),
            getattr(sibling, "blob_id", None),
            getattr(lfs, "sha256", None),
        )
    return result


def _preflight(
    write_api: Any, read_api: Any, write: str, read: str
) -> tuple[str, dict[str, FileIdentity]]:
    writer = write_api.whoami(token=write)
    reader = read_api.whoami(token=read)
    if writer.get("name") != REPO.split("/")[0] or not _scope(writer, "write"):
        raise PublishError("write token is valid but lacks exact private repo write scope")
    if not _read_only(reader):
        raise PublishError("read token lacks exact repo read scope or has write permission")
    before = write_api.repo_info(REPO, repo_type="dataset", token=write, files_metadata=True)
    read_info = read_api.repo_info(REPO, repo_type="dataset", token=read, files_metadata=True)
    if not before.private or not read_info.private or before.sha != read_info.sha:
        raise PublishError("private repo identity differs between write and read tokens")
    files = _files(before)
    if "mast_public_test.npz" not in files:
        raise PublishError("existing MAST file is missing; aborting single-file upload")
    if not SHA_RE.fullmatch(before.sha):
        raise PublishError("remote parent commit SHA is malformed")
    return before.sha, files


def _pointer(path: Path, sha: str, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps({"repo_id": REPO, "revision": sha, "token": token}, indent=2).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with (
        os.fdopen(descriptor, "wb") as output,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        archive.writestr("manifest.json", manifest)


def publish(
    *,
    api_type: Callable[..., Any],
    download: Callable[..., str],
    dotenv: Path = Path(".env"),
    pointer: Path | None = None,
    evidence: Path | None = None,
    receipt_path: Path | None = None,
    expected_parent: str | None = None,
    variant: str = "float32",
) -> dict[str, Any]:
    """Upload one file, verify pinned remote bytes with read access, then create pointer."""
    if evidence is None:
        evidence = EVIDENCE
    if receipt_path is None:
        receipt_path = RECEIPT
    if variant == "float16-probe" and receipt_path == RECEIPT:
        raise PublishError("float16 probe requires a separate receipt path")
    if expected_parent is not None and not SHA_RE.fullmatch(expected_parent):
        raise PublishError("expected parent commit SHA is malformed")
    if receipt_path.exists():
        raise PublishError(f"{receipt_path}: refusing to replace an existing receipt")
    candidate, size, digest = _candidate(evidence, variant)
    write, read = _credentials(dotenv)
    write_api = api_type(token=write)
    read_api = api_type(token=read)
    parent, old_files = _preflight(write_api, read_api, write, read)
    if expected_parent is not None and parent != expected_parent:
        raise PublishError(f"private repo parent changed: {parent}, expected {expected_parent}")
    commit = write_api.upload_file(
        path_or_fileobj=str(candidate),
        path_in_repo=FILENAME,
        repo_id=REPO,
        repo_type="dataset",
        token=write,
        parent_commit=parent,
        commit_message=f"Publish validated {variant} DIII-D public predictions",
    )
    sha = commit.oid
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise PublishError("upload returned malformed commit SHA")
    after = read_api.repo_info(
        REPO, revision=sha, repo_type="dataset", token=read, files_metadata=True
    )
    if not after.private or after.sha != sha:
        raise PublishError("read token cannot inspect pinned private upload")
    new_files = _files(after)
    old_other = {k: v for k, v in old_files.items() if k != FILENAME}
    new_other = {k: v for k, v in new_files.items() if k != FILENAME}
    if old_other != new_other:
        raise PublishError("a non-target remote file changed during upload")
    remote = new_files.get(FILENAME)
    if remote is None or remote[0] != size or remote[2] != digest:
        raise PublishError("pinned remote DIII-D size or LFS SHA-256 does not match")
    with tempfile.TemporaryDirectory(prefix="hf-read-verify-", dir="runs") as temp:
        downloaded = Path(
            download(
                REPO,
                FILENAME,
                repo_type="dataset",
                revision=sha,
                token=read,
                local_dir=temp,
                force_download=True,
            )
        )
        if downloaded.stat().st_size != size or _sha256(downloaded) != digest:
            raise PublishError("read-token download of pinned DIII-D bytes differs")
    if pointer is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "f16_probe_" if variant == "float16-probe" else ""
        pointer = POINTER_DIR / f"submission_pointer_{label}{timestamp}.zip"
    _pointer(pointer, sha, read)
    receipt: dict[str, Any] = {
        "repo_id": REPO,
        "private": True,
        "variant": variant,
        "parent_revision": parent,
        "revision": sha,
        "uploaded_file": FILENAME,
        "bytes": size,
        "sha256": digest,
        "mast_preserved": True,
        "non_target_files_preserved": sorted(old_other),
        "pinned_read_token_download_verified": True,
        "pointer_path": str(pointer),
        "pointer_bytes": pointer.stat().st_size,
        "pointer_sha256": _sha256(pointer),
        "evidence_path": str(evidence),
        "evidence_sha256": _sha256(evidence),
        "token_values_recorded": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "preflight", "publish"])
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--expected-parent")
    parser.add_argument("--variant", choices=("float32", "float16-probe"), default="float32")
    args = parser.parse_args()
    try:
        if args.mode == "publish":
            from huggingface_hub import HfApi, hf_hub_download

            receipt = publish(
                api_type=HfApi,
                download=hf_hub_download,
                evidence=args.evidence,
                receipt_path=args.receipt,
                expected_parent=args.expected_parent,
                variant=args.variant,
            )
            print(json.dumps(receipt))
            return 0
        candidate, size, digest = _candidate(args.evidence, args.variant)
        if args.mode == "plan":
            print(
                json.dumps(
                    {
                        "repo_id": REPO,
                        "file": str(candidate),
                        "bytes": size,
                        "sha256": digest,
                        "upload": False,
                    }
                )
            )
            return 0
        from huggingface_hub import HfApi

        write, read = _credentials()
        parent, files = _preflight(HfApi(token=write), HfApi(token=read), write, read)
        if args.expected_parent is not None and parent != args.expected_parent:
            raise PublishError(
                f"private repo parent changed: {parent}, expected {args.expected_parent}"
            )
        if args.mode == "preflight":
            print(
                json.dumps(
                    {
                        "repo_id": REPO,
                        "private": True,
                        "write_scope": True,
                        "read_scope": True,
                        "parent": parent,
                        "non_target_files": sorted(k for k in files if k != FILENAME),
                        "upload": False,
                    }
                )
            )
            return 0
        raise AssertionError("unknown mode")
    except PublishError as exc:
        print(f"Publication stopped: {exc}")
    except Exception as exc:
        print(
            f"Publication stopped at remote operation ({type(exc).__name__}); "
            "credentials and response details suppressed"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
