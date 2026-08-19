"""Read the public MAST archive at STFC Echo — used for MACHINE GEOMETRY and nothing else.

    https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr   (zarr v2, CC BY 4.0)
    https://mastapp.site/json/shots                            (15,969 shots, campaigns M5-M9)

**Read this before using it for anything else.** The archive's `efm/` group is the competition's
withheld MAST ground truth, bit for bit. Verified on demo shot 28348, whose truth the competition
does ship: frames match to 0.0000 ms and then `efm/psirz` against `efit_psirz` has a maximum
absolute difference of **0.000e+00**, as do `li`, `betan`, `elongation`, `magnetic_axis_r` and
`psi_axis`. The 1206-shot public test fold is drawn from these same campaigns — the three demo
shots are M8 — so its answers are in here.

The competition terms allow external public datasets **with disclosure** and separately forbid any
attempt to "de-anonymize, memorize, or otherwise leak the hidden ground truth", on pain of
disqualification. Training on this archive would be the second thing even without intending it,
and excluding the test fold first requires identifying it, which is also the second thing. So this
module is used for ONE purpose, which is neither: `efm/limiterr` / `efm/limiterz`, MAST's first
wall. That is a machine constant, byte-identical on shots sampled from all five campaigns, of the
same kind as the coil rectangles the competition ships on every row — not an equilibrium, and not
an answer to any shot. **The organisers have been asked about the wider question; until they
answer, nothing here reads a per-shot label.**

Two infrastructure facts, both measured on this box and both worth keeping:

Two infrastructure facts, both measured on this box and both worth keeping:

* the STFC hosts publish AAAA records this container cannot route, and every request spends ~90 s
  on the v6 attempt before falling back — so the resolver is pinned to A records only;
* `s3.echo.stfc.ac.uk` round-robins over four A records and **130.246.179.222 black-holes**, so the
  address is pinned past it. DNS is patched rather than the socket, because connecting by IP breaks
  SNI and the certificate check with it.
"""
import http.client
import itertools
import json
import socket
import ssl

import numpy as np

HOST = "s3.echo.stfc.ac.uk"
GOOD = "130.246.179.110"          # of 4 A records; .222 black-holes
_real = socket.getaddrinfo


def _pinned(host: str, port: int, family: int = 0, type: int = 0,
            proto: int = 0, flags: int = 0) -> list:
    if host == HOST:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (GOOD, port))]
    return _real(host, port, family, type, proto, flags)


socket.getaddrinfo = _pinned  # type: ignore[assignment]
_conn = None


def get(path: str) -> bytes:
    global _conn
    for attempt in (0, 1):
        try:
            if _conn is None:
                _conn = http.client.HTTPSConnection(HOST, 443, timeout=60,
                                                    context=ssl.create_default_context())
            _conn.request("GET", path, headers={"Host": HOST})
            r = _conn.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status} for {path}")
            return body
        except (http.client.HTTPException, OSError):
            if _conn is not None:
                _conn.close()
            _conn = None
            if attempt:
                raise
    raise RuntimeError("unreachable")


_META: dict = {}


def meta(shot: int) -> dict:
    """`.zmetadata` for one shot, cached — it is 1.5 MB and every array lookup wants it."""
    if shot not in _META:
        _META[shot] = json.loads(get(f"/mast/level1/shots/{shot}.zarr/.zmetadata"))["metadata"]
    return _META[shot]


def array(shot: int, name: str, md: dict | None = None) -> np.ndarray:
    """One array out of a shot's zarr group. Needs `numcodecs`, which is NOT a project dependency:
    the only thing this module contributes to a submission is MAST's first wall, and that is baked
    into `solver/machine.py` as a literal. Re-fetch it with
    `uv run --with numcodecs python -c ...` if the machine geometry is ever in doubt.
    """

    import numcodecs

    md = md or meta(shot)
    za = md[f"{name}/.zarray"]
    shape, chunks = tuple(za["shape"]), tuple(za["chunks"])
    dt = np.dtype(za["dtype"])
    comp = numcodecs.get_codec(za["compressor"])
    out = np.full(shape, np.nan, dtype=dt)
    grid = [-(-s // c) for s, c in zip(shape, chunks, strict=True)]
    for idx in itertools.product(*[range(g) for g in grid]):
        raw = comp.decode(get(f"/mast/level1/shots/{shot}.zarr/{name}/" + ".".join(map(str, idx))))
        blk = np.frombuffer(raw, dtype=dt).reshape(chunks)
        sl = tuple(slice(i * c, min((i + 1) * c, s))
                   for i, c, s in zip(idx, chunks, shape, strict=True))
        out[sl] = blk[tuple(slice(0, x.stop - x.start) for x in sl)]
    return out
