"""Submit Final pointers through the organizers' website (Codabench outage, codalab/codabench#2679).

    uv run python tools/final_web_submit.py full39e full45e compactA

The organizers' 2026-10-23 email moved Final entries to
https://fusion-equilibrium-challenge.sophelio.io/final-submission/ ; the page POSTs JSON to
/final-submit.php, which checks the pointer with Hugging Face and refuses (no slot used) an entry it
cannot score. User approved submitting the three candidates this way, 2026-10-23. In order; stops on
the first refusal. Token and username are read in memory (pointer ZIP, .env) and never written out;
receipts go to results/final_web_receipts.json.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import httpx

URL = "https://fusion-equilibrium-challenge.sophelio.io/final-submit.php"
EMAIL = "pshishkin@gmail.com"
OUT = Path("results/final_web_receipts.json")


def env(key: str) -> str:
    for line in Path(".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    raise SystemExit(f"{key} missing in .env")


def main() -> int:
    user = env("CODABENCH_USERNAME")
    log = json.loads(OUT.read_text()) if OUT.exists() else []
    for c in sys.argv[1:]:
        m = json.loads(zipfile.ZipFile(f"submissions/final_{c}_pointer.zip").read("manifest.json"))
        body = {"codabenchUser": user, "email": EMAIL, "repoId": m["repo_id"],
                "revision": m["revision"], "token": m["token"], "teamName": "", "website": "",
                "notes": f"Challenge 1 only (DIII-D). Candidate {c}.", "ackRules": True}
        r = httpx.post(URL, content=json.dumps(body), timeout=300,
                       headers={"Content-Type": "text/plain;charset=utf-8"})
        try:
            p = r.json()
        except ValueError:
            p = {"ok": False, "error": f"HTTP {r.status_code}, non-JSON body"}
        p.pop("token", None)
        rec = {"candidate": c, "revision": m["revision"], "http": r.status_code, "response": p}
        log.append(rec)
        OUT.write_text(json.dumps(log, indent=2) + "\n")
        print(json.dumps(rec), flush=True)
        if r.status_code != 200 or p.get("ok") is not True:
            print(f"STOPPED at {c}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
