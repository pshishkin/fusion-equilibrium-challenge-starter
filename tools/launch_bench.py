#!/usr/bin/env python3
"""Where an MLP step's time goes, and whether any of the standard fixes touch it.

Run this on an IDLE GPU. Measured on 2026-09-02 while the card was 72% busy with another fit, every
variant came back within 4% of the baseline — including CUDA graphs, which remove launch overhead
almost entirely. Either the step is genuinely compute-bound at this shape, or the contention was
what was being measured all along. Those are different answers and only an idle card separates them.

The three things being distinguished, because they are usually conflated:

  * `fused` / `foreach` Adam    — fewer, larger optimizer kernels. Fixes optimizer launch cost.
  * `torch.compile`             — different kernels: fusion, so less memory traffic and fewer of
                                  them. Changes what the GPU does.
  * CUDA graphs                 — the same kernels, submitted once instead of ~70 times. Changes
                                  nothing about the GPU's work, only the CPU's.
  * `mode="reduce-overhead"`    — compile AND graphs together, which is the composition worth
                                  knowing about: fuse first, then make the survivors cheap to send.
"""
import argparse
import time

import torch

N_IN, H, N_OUT = 84, 512, 52


def build(dev, **adam):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(N_IN, H), torch.nn.ReLU(),
                              torch.nn.Linear(H, H), torch.nn.ReLU(),
                              torch.nn.Linear(H, N_OUT)).to(dev)
    return net, torch.optim.Adam(net.parameters(), lr=1e-3, **adam)


def timed(fn, n=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=4096, help="production batch_size")
    a = ap.parse_args()
    dev = torch.device("cuda")
    B = a.batch
    fl = 3 * 2 * B * (N_IN * H + H * H + H * N_OUT)
    peak = 15.7e12
    util = torch.cuda.utilization()
    print(f"  {torch.cuda.get_device_name(0)}, utilisation at start {util}% "
          f"{'(NOT IDLE — the numbers below measure contention)' if util > 5 else '(idle)'}")
    x = torch.randn(B, N_IN, device=dev)
    y = torch.randn(B, N_OUT, device=dev)
    res = {}

    def plain(compiled=False, mode=None, **adam):
        net, opt = build(dev, **adam)
        fwd = torch.compile(net, mode=mode) if compiled else net

        def step():
            opt.zero_grad(set_to_none=True)
            ((fwd(x) - y) ** 2).mean().backward()
            opt.step()
        return timed(step, warm=30 if compiled else 15)

    res["Adam default"] = plain()
    res["Adam foreach"] = plain(foreach=True)
    res["Adam fused"] = plain(fused=True)
    for label, mode in (("compile default", None), ("compile reduce-overhead", "reduce-overhead")):
        try:
            res[label] = plain(compiled=True, mode=mode, fused=True)
        except Exception as exc:
            res[label] = f"FAILED: {type(exc).__name__}"
    try:
        net, opt = build(dev, fused=True, capturable=True)
        sx, sy = x.clone(), y.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                opt.zero_grad(set_to_none=False)
                ((net(sx) - sy) ** 2).mean().backward()
                opt.step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            opt.zero_grad(set_to_none=False)
            ((net(sx) - sy) ** 2).mean().backward()
            opt.step()
        res["CUDA graph"] = timed(g.replay)
    except Exception as exc:
        res["CUDA graph"] = f"FAILED: {type(exc).__name__}"

    base = res["Adam default"]
    print(f"\n  batch {B}, {fl / 1e9:.2f} GFLOP per step\n")
    print(f"  {'variant':<26s} {'ms/step':>9s} {'speedup':>9s} {'TFLOPS':>8s} {'% of peak':>10s}")
    for k, v in res.items():
        if isinstance(v, str):
            print(f"  {k:<26s} {v}")
        else:
            print(f"  {k:<26s} {v * 1e3:9.3f} {base / v:8.2f}x {fl / v / 1e12:8.2f} "
                  f"{100 * fl / v / peak:9.1f}%")
    # Does the step time care how much work it is given? If it does not, the work is not the cost.
    print("\n  does time scale with work?")
    def scaling(b: int) -> float:
        net, opt = build(dev)
        xb = torch.randn(b, N_IN, device=dev)
        yb = torch.randn(b, N_OUT, device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            ((net(xb) - yb) ** 2).mean().backward()
            opt.step()
        return timed(step)

    for b in (256, 1024, 4096, 16384):
        t = scaling(b)
        print(f"    batch {b:>6d}: {t * 1e3:7.3f} ms   {t / b * 1e9:7.1f} ns per row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
