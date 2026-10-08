#!/usr/bin/env python3
"""Submit a pointer zip to the Fusion Equilibrium Challenge on Codabench — ONLY when the user approved it.

    uv run python tools/codabench_submit.py submissions/<name>_pointer.zip [--phase 29580]

Credentials come from `.env` (CODABENCH_USERNAME, CODABENCH_PASSWORD; written by `bb secret request`,
never printed). Flow, as the Codabench web client does it: session login (Django allauth form with
CSRF), create a submission dataset (returns a presigned upload URL), PUT the zip there, mark the
dataset complete, create the submission in the phase. Prints the submission id; the score is read
later from the public leaderboard / detail endpoints.

Phase 29580 = Development (public test, 5 per day, ends 2026-10-18); 29581 = Final (private test,
3 per person in total, 2026-10-19..26) — never submit there without an explicit decision.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

BASE = "https://www.codabench.org"
COMPETITION = 17739


def creds(path: Path) -> tuple[str, str]:
    vals = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(CODABENCH_USERNAME|CODABENCH_PASSWORD)\s*=\s*(.*)\s*$", line)
        if m:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            vals[m.group(1)] = v
    if set(vals) != {"CODABENCH_USERNAME", "CODABENCH_PASSWORD"}:
        raise SystemExit("CODABENCH_USERNAME / CODABENCH_PASSWORD missing from .env")
    return vals["CODABENCH_USERNAME"], vals["CODABENCH_PASSWORD"]


def login(s: requests.Session, user: str, password: str) -> None:
    r = s.get(f"{BASE}/accounts/login/", timeout=60)
    r.raise_for_status()
    token = s.cookies.get("csrftoken")
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)
    form = {"csrfmiddlewaretoken": m.group(1) if m else token, "login": user, "username": user,
            "password": password}
    r = s.post(f"{BASE}/accounts/login/", data=form, headers={"Referer": f"{BASE}/accounts/login/"},
               timeout=60, allow_redirects=True)
    r.raise_for_status()
    me = s.get(f"{BASE}/api/my_profile/", timeout=60)
    if me.status_code != 200 or "username" not in me.text:
        raise SystemExit(f"login failed (profile endpoint returned {me.status_code})")


def short(r: requests.Response) -> str:
    """The error body if it is JSON/text, a marker if Codabench answered with an HTML page."""
    t = r.text
    return "<html error page>" if t.lstrip().startswith("<") else t[:300]


def api_headers(s: requests.Session) -> dict[str, str]:
    return {"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": f"{BASE}/competitions/{COMPETITION}/"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--phase", type=int, default=29580)
    ap.add_argument("--env", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    args = ap.parse_args()
    if args.phase != 29580:
        raise SystemExit("only the Development phase (29580) is allowed from this tool")
    if not args.zip.is_file():
        raise SystemExit(f"{args.zip}: no such file")
    user, password = creds(args.env)
    s = requests.Session()
    login(s, user, password)
    import time
    name = args.zip.name
    # Dataset names must be unique per account; a retried upload would otherwise be refused.
    dsname = f"{args.zip.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    r = s.post(f"{BASE}/api/datasets/", headers=api_headers(s), timeout=60, json={
        "type": "submission", "name": dsname, "description": "", "is_public": False,
        "request_sassy_file_name": name, "file_name": name, "file_size": args.zip.stat().st_size,
        "competition": COMPETITION})
    if r.status_code >= 300:
        raise SystemExit(f"dataset create failed: {r.status_code} {short(r)}")
    ds = r.json()
    up = requests.put(ds["sassy_url"], data=args.zip.read_bytes(),
                      headers={"Content-Type": "application/zip"}, timeout=300)
    if up.status_code >= 300:
        raise SystemExit(f"upload failed: {up.status_code} {short(up)}")
    r = s.put(f"{BASE}/api/datasets/completed/{ds['key']}/", headers=api_headers(s), timeout=60)
    if r.status_code >= 300:
        raise SystemExit(f"dataset completion failed: {r.status_code} {short(r)}")
    r = s.post(f"{BASE}/api/submissions/", headers=api_headers(s), timeout=60,
               json={"phase": args.phase, "data": ds["key"], "fact_sheet_answers": None})
    if r.status_code >= 300:
        raise SystemExit(f"submission create failed: {r.status_code} {short(r)}")
    sub = r.json()
    print(f"SUBMITTED id={sub.get('id')} phase={args.phase} file={name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
