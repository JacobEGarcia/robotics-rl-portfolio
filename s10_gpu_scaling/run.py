"""
S10 driver. Three subcommands, each writing one JSON file into results/.

    python -m s10_gpu_scaling.run parity     [--models ...] [--steps 300]
    python -m s10_gpu_scaling.run throughput [--models ...] [--max-envs 8192]
    python -m s10_gpu_scaling.run solver     [--models ...]
    python -m s10_gpu_scaling.run all

Everything is written incrementally, one model at a time. A free Colab session
can be reclaimed at any moment, and a driver that accumulates results in memory
and writes at the end turns a session that completed six of eight models into a
session that produced nothing.

The float64 MJX case runs as a subprocess with `JAX_ENABLE_X64=1`, because
`jax_enable_x64` is process-global and the mjx model's array dtypes are fixed
when it is built. `parity.rollout_mjx` verifies the dtype it actually got, so a
subprocess that failed to pick the flag up is an error rather than a duplicate
float32 run reported as agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _write(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)                       # never leave a half-written result
    print(f"  wrote {p.relative_to(RESULTS.parent.parent)}", flush=True)
    return p


def _merge(name: str, key: str, value: dict) -> None:
    """Update one model's entry in a results file without losing the others."""
    p = RESULTS / name
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    doc[key] = value
    _write(name, doc)


def _specs(names: list[str] | None):
    from s10_gpu_scaling import models
    avail = models.available()
    if not names:
        return avail
    by_key = {s.key: s for s in avail}
    missing = [n for n in names if n not in by_key]
    if missing:
        raise SystemExit(
            f"unknown or unavailable model(s): {missing}. "
            f"available: {sorted(by_key)}"
        )
    return [by_key[n] for n in names]


def _device() -> dict:
    import jax
    d = jax.devices()[0]
    return {
        "platform": d.platform,
        # `device_kind` is the string the driver reports, and it has been
        # wrong before: a Kaggle session advertised as T4 x2 reported a P100.
        # Record it, quote it as "what the driver said", never as the hardware.
        "device_kind_reported": getattr(d, "device_kind", "?"),
        "n_devices": len(jax.devices()),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
    }


# --------------------------------------------------------------------------

def cmd_parity(args: argparse.Namespace) -> None:
    from s10_gpu_scaling import models, parity

    for spec in _specs(args.models):
        print(f"[parity] {spec.key}", flush=True)
        m = models.load(spec)
        try:
            res = parity.run_parity(m, n_steps=args.steps, seed=args.seed,
                                    include_mjx=not args.no_mjx)
        except Exception as exc:                       # noqa: BLE001
            # An MJX conversion refusal is a real finding about the model, not
            # a driver failure. Record which feature it rejected and continue.
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            _merge(args.out, spec.key, {"error": f"{type(exc).__name__}: {exc}",
                                        "model": models.describe(m)})
            continue
        res["model"] = models.describe(m)
        res["regime"] = spec.regime
        res["device"] = _device() if not args.no_mjx else None
        _merge(args.out, spec.key, res)

        s = res["summary"]
        for k in ("envelope", "roundoff", "mjx_f32", "mjx_f64", "mjc_dt"):
            if k in s:
                print(f"    {k:<10} vs {s[k]['baseline']:<14} final "
                      f"{s[k]['final_m']:.3e} m  1mm at {s[k]['seconds_to_1mm']}",
                      flush=True)
        print(f"    reference floor (REFINE self-gap) "
              f"{res['reference_check']['self_gap_max_m']:.3e} m", flush=True)


def cmd_parity_x64(args: argparse.Namespace) -> None:
    """Re-run parity in a float64 subprocess and merge the extra curve in."""
    env = dict(os.environ, JAX_ENABLE_X64="1")
    cmd = [sys.executable, "-u", "-m", "s10_gpu_scaling.run", "parity",
           "--steps", str(args.steps), "--seed", str(args.seed),
           "--out", "parity_f64.json"]
    if args.models:
        cmd += ["--models", *args.models]
    print("$ JAX_ENABLE_X64=1 " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True,
                   cwd=str(Path(__file__).resolve().parent.parent))

    # Fold the f64 curve into the main file so the figures read one document.
    src, dst = RESULTS / "parity_f64.json", RESULTS / args.out
    if not (src.exists() and dst.exists()):
        return
    a = json.loads(src.read_text(encoding="utf-8"))
    b = json.loads(dst.read_text(encoding="utf-8"))
    for key, val in a.items():
        if key not in b or "curves_vs_mjc" not in val:
            continue
        if "mjx_f64" not in val.get("curves_vs_mjc", {}):
            continue
        b[key]["curves_vs_mjc"]["mjx_f64"] = val["curves_vs_mjc"]["mjx_f64"]
        b[key]["summary"]["mjx_f64"] = val["summary"]["mjx_f64"]
        b[key].setdefault("mjx", {})["mjx_f64"] = val.get("mjx", {}).get("mjx_f64")
    _write(args.out, b)


def cmd_integrator(args: argparse.Namespace) -> None:
    from s10_gpu_scaling import models, parity

    for spec in _specs(args.models):
        print(f"[integrator] {spec.key}", flush=True)
        m = models.load(spec)
        res = parity.integrator_sweep(m, n_steps=args.steps, seed=args.seed)
        res["model"] = models.describe(m)
        res["device"] = _device()
        for r in res["rows"]:
            tag = " (shipped)" if r["shipped"] else ""
            if "unsupported" in r:
                print(f"    {r['integrator']:<20} NOT SUPPORTED BY MJX{tag}", flush=True)
            elif "error" in r:
                print(f"    {r['integrator']:<20} {r['error']}{tag}", flush=True)
            else:
                print(f"    {r['integrator']:<20} mjx vs mjc {r['mjx_vs_mjc_m']:.3e} m  "
                      f"float32 floor {r['float32_floor_m']:.3e}  "
                      f"ratio {r['ratio_to_floor']:>8.2f}{tag}", flush=True)
        _merge(args.out, spec.key, res)


def cmd_throughput(args: argparse.Namespace) -> None:
    from s10_gpu_scaling import models, throughput

    sizes = [1 << i for i in range(0, 20) if (1 << i) <= args.max_envs]

    for spec in _specs(args.models):
        print(f"[throughput] {spec.key}", flush=True)
        m = models.load(spec)
        cpu = throughput.mjc_throughput(m, n_steps=args.cpu_steps)
        print(f"    cpu baseline  {cpu.steps_per_s:,.0f} steps/s "
              f"({cpu.n_envs} thread(s)){' - ' + cpu.error if cpu.error else ''}",
              flush=True)

        pts = throughput.mjx_sweep(m, sizes, n_steps=args.steps, seed=args.seed)
        k = throughput.knee(pts)
        best = max((p for p in pts if p.steps_per_s),
                   key=lambda p: p.steps_per_s, default=None)

        _merge(args.out, spec.key, {
            "model": models.describe(m),
            "regime": spec.regime,
            "device": _device(),
            "cpu_baseline": cpu.to_dict(),
            "points": [p.to_dict() for p in pts],
            "knee": k.to_dict() if k else None,
            "best": best.to_dict() if best else None,
            # The headline ratio, computed once here rather than in three
            # different figure scripts that could each round it differently.
            "gpu_over_cpu": (best.steps_per_s / cpu.steps_per_s
                             if best and cpu.steps_per_s else None),
        })
        if best and cpu.steps_per_s:
            print(f"    best {best.steps_per_s:,.0f} steps/s at n={best.n_envs:,} "
                  f"= {best.steps_per_s / cpu.steps_per_s:.1f}x the CPU baseline",
                  flush=True)


def cmd_solver(args: argparse.Namespace) -> None:
    from s10_gpu_scaling import models, solver_cost

    for spec in _specs(args.models):
        print(f"[solver] {spec.key}", flush=True)
        m = models.load(spec)
        res = solver_cost.sweep_iterations(
            m, n_steps=args.steps, n_envs=args.n_envs, seed=args.seed,
            measure_throughput=not args.no_mjx)
        res["model"] = models.describe(m)
        res["regime"] = spec.regime
        res["device"] = _device() if not args.no_mjx else None
        limited = solver_cost.is_solver_limited(res["rows"])
        rec = solver_cost.cheapest_within(res["rows"])
        res["solver_limited"] = limited
        res["recommended"] = rec
        if rec:
            print(f"    within 2x of best error at {rec['iterations']} iterations "
                  f"(model ships {res['shipped']['iterations']})", flush=True)
        else:
            errs = [r["final_m"] for r in res["rows"]]
            print(f"    NOT solver-limited: error spans only "
                  f"{max(errs)/max(min(errs), 1e-12):.2f}x across the whole grid. "
                  f"No recommendation, the ordering is scatter.", flush=True)
        _merge(args.out, spec.key, res)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="S10: GPU simulation scaling and fidelity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--models", nargs="*", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-mjx", action="store_true",
                       help="CPU-only; skips every measurement that needs JAX")

    p = sub.add_parser("parity"); common(p)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="parity.json")
    p.set_defaults(fn=cmd_parity)

    p = sub.add_parser("parity-x64"); common(p)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="parity.json")
    p.set_defaults(fn=cmd_parity_x64)

    p = sub.add_parser("integrator"); common(p)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="integrator.json")
    p.set_defaults(fn=cmd_integrator)

    p = sub.add_parser("throughput"); common(p)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cpu-steps", type=int, default=2000)
    p.add_argument("--max-envs", type=int, default=8192)
    p.add_argument("--out", default="throughput.json")
    p.set_defaults(fn=cmd_throughput)

    p = sub.add_parser("solver"); common(p)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--out", default="solver.json")
    p.set_defaults(fn=cmd_solver)

    p = sub.add_parser("all"); common(p)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--cpu-steps", type=int, default=2000)
    p.add_argument("--max-envs", type=int, default=8192)
    p.set_defaults(fn=None)

    args = ap.parse_args()

    if args.cmd == "all":
        for name, fn, kw in (
            ("parity", cmd_parity, {"out": "parity.json", "steps": args.steps}),
            ("parity-x64", cmd_parity_x64, {"out": "parity.json", "steps": args.steps}),
            ("integrator", cmd_integrator, {"out": "integrator.json", "steps": 40}),
            ("throughput", cmd_throughput,
             {"out": "throughput.json", "steps": 100,
              "cpu_steps": args.cpu_steps, "max_envs": args.max_envs}),
            ("solver", cmd_solver,
             {"out": "solver.json", "steps": 200, "n_envs": 256}),
        ):
            print(f"\n=== {name} ===", flush=True)
            sub_args = argparse.Namespace(**{**vars(args), **kw})
            fn(sub_args)
        return

    args.fn(args)


if __name__ == "__main__":
    main()
