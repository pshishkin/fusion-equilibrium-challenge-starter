"""Give each Final candidate a revision that holds ONLY diii_d_private_test.npz, and repoint it.

    uv run python tools/hf_final_cleanup.py

Organizers' Final guide, section 3: "Keep only the Final files in your predictions repo ... The
scorer downloads every submission file it recognises there, and one worker serves every team."
The three published revisions (results/final_*_hf_receipt.json) still carry the Development
diii_d_public_test.npz and mast_public_test.npz. User approved the cleanup 2026-10-23.

Three commits on top of main, which must still be the compactA revision: (1) delete both public
files -> compactA's clean revision; (2), (3) server-side copy of diii_d_private_test.npz from the
full39e / full45e revisions (CommitOperationCopy: same LFS object, nothing is uploaded). Every new
revision is checked with the READ token: exactly {.gitattributes, diii_d_private_test.npz}, the
LFS sha256 of the receipt, and the pinned file's metadata readable. Only then the pointer ZIP is
written (old one kept as *_v1.zip) and swapped in with os.replace, so the waiting auto-submitter
always reads a complete pointer. Tokens come from .env in memory and land only in the pointer ZIP.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_url
from huggingface_hub import get_hf_file_metadata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_private_candidate import REPO, _credentials, _files, _pointer, _sha256  # noqa: E402

TARGET = "diii_d_private_test.npz"
STALE = ["diii_d_public_test.npz", "mast_public_test.npz"]
ORDER = ["compactA", "full39e", "full45e"]   # compactA first: it is what main holds now


def receipt(c: str) -> dict:
    return json.loads(Path(f"results/final_{c}_hf_receipt.json").read_text())


def check(read_api: HfApi, read: str, rev: str, sha: str) -> None:
    info = read_api.repo_info(REPO, repo_type="dataset", revision=rev, files_metadata=True)
    files = _files(info)
    if set(files) != {".gitattributes", TARGET}:
        raise SystemExit(f"{rev}: unexpected file set {sorted(files)}")
    if files[TARGET][2] != sha:
        raise SystemExit(f"{rev}: {TARGET} sha256 {files[TARGET][2]} != {sha}")
    meta = get_hf_file_metadata(hf_hub_url(REPO, TARGET, repo_type="dataset", revision=rev),
                                token=read)
    if meta.commit_hash != rev or meta.etag.strip('"') != sha:
        raise SystemExit(f"{rev}: pinned metadata mismatch ({meta.commit_hash}, {meta.etag})")


def main() -> int:
    write, read = _credentials()
    w, r = HfApi(token=write), HfApi(token=read)
    rec = {c: receipt(c) for c in ORDER}
    head = w.repo_info(REPO, repo_type="dataset").sha
    if head != rec["compactA"]["revision"]:
        raise SystemExit(f"main is {head}, expected compactA's {rec['compactA']['revision']}")
    new = {}
    for c in ORDER:
        if c == "compactA":
            ops = [CommitOperationDelete(path_in_repo=f) for f in STALE]
        else:
            ops = [CommitOperationCopy(src_path_in_repo=TARGET, path_in_repo=TARGET,
                                       src_revision=rec[c]["revision"])]
        info = w.create_commit(REPO, ops, repo_type="dataset", parent_commit=head,
                               commit_message=f"Final candidate {c}: {TARGET} only")
        head = info.oid
        check(r, read, head, rec[c]["sha256"])
        new[c] = head
        print(f"{c}: clean revision {head} verified", flush=True)
    for c in ORDER:
        final = Path(f"submissions/final_{c}_pointer.zip")
        tmp = final.with_suffix(".zip.tmp")
        tmp.unlink(missing_ok=True)
        _pointer(tmp, new[c], read)
        old = final.with_name(f"final_{c}_pointer_v1.zip")
        if not old.exists():
            os.link(final, old)
        os.replace(tmp, final)
        out = {"repo_id": REPO, "candidate": c, "revision": new[c], "file": TARGET,
               "sha256": rec[c]["sha256"], "replaces_revision": rec[c]["revision"],
               "removed": STALE, "pointer_path": str(final), "pointer_sha256": _sha256(final),
               "previous_pointer": str(old), "read_token_verified": True,
               "token_values_recorded": False}
        Path(f"results/final_{c}_hf_clean_receipt.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"{c}: pointer {final} -> {new[c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
