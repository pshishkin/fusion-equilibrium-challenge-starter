"""Reject a partial or changed DIII-D corpus before constructing a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toolkit.records import file_hash

MANIFEST = Path(__file__).with_name("diii_d_dataset.json")


def check_dataset(
    root: Path, config: str = "diii_d_train", *, verify_hashes: bool = False
) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    if config not in manifest["configs"]:
        raise ValueError(f"{MANIFEST}: unsupported DIII-D config {config!r}")
    entries = manifest["configs"][config]
    folder = root / "data" / config
    actual = {p.name for p in folder.glob("*.parquet")}
    expected = set(entries)
    if actual != expected:
        raise ValueError(
            f"{folder}: {len(actual)} parquet files, expected {len(expected)}; "
            f"missing={sorted(expected - actual)[:3]}, unexpected={sorted(actual - expected)[:3]}"
        )
    total = 0
    for name, (size, digest) in entries.items():
        path = folder / name
        got = path.stat().st_size
        if got != size:
            raise ValueError(f"{path}: {got} bytes, expected {size}")
        with path.open("rb") as stream:
            start = stream.read(4)
            stream.seek(-4, 2)
            end = stream.read(4)
        if start != b"PAR1" or end != b"PAR1":
            raise ValueError(f"{path}: parquet markers {start!r}/{end!r}, expected PAR1/PAR1")
        if verify_hashes:
            got_hash = file_hash(path)
            if got_hash != digest:
                raise ValueError(f"{path}: SHA-256 {got_hash}, expected {digest}")
        total += size
    return {
        "revision": manifest["revision"],
        "manifest_sha256": file_hash(MANIFEST),
        "config": config,
        "files": len(entries),
        "bytes": total,
        "verification": "sha256" if verify_hashes else "names_sizes_parquet_markers",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("../downloaded_huggingface/hf_dataset"))
    ap.add_argument("--config", default="diii_d_train")
    ap.add_argument("--sha256", action="store_true")
    args = ap.parse_args()
    print(
        json.dumps(check_dataset(args.data_dir, args.config, verify_hashes=args.sha256), indent=2)
    )


if __name__ == "__main__":
    main()
