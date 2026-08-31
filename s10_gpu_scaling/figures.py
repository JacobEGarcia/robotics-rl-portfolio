"""
Figures for S10. Reads results/*.json, writes media/s10/*.png.

    python -m s10_gpu_scaling.figures --dark

Three plots, one per question:

  parity_<model>.png   divergence vs time, log y, with the chaos envelope
                       shaded. The shaded band is the point of the figure:
                       anything inside it is not a discrepancy.
  throughput.png       steps/s vs batch size, per model, with the CPU
                       baseline as a horizontal line and the knee marked.
  solver.png           error against throughput, one point per iteration
                       budget, with the value the model ships circled.

Every axis that spans orders of magnitude is log. A linear axis on a
divergence curve that runs from 1e-9 to 1e-2 shows a flat line and then a
cliff, which is a picture of the axis rather than of the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MEDIA = Path(__file__).resolve().parent.parent / "media" / "s10"

# Colours chosen to survive both the dark site and a greyscale print, and to
# stay distinguishable under the two most common colour-vision deficiencies:
# no red/green pair carries meaning on its own, and the four series also
# differ in lightness.
COLORS = {
    "envelope": "#5a6478",
    "roundoff": "#8892a6",
    "mjx_f32": "#4cc9f0",
    "mjx_f64": "#f4a261",
    "mjc_dt": "#e76f51",
}
LABELS = {
    "envelope": "floor A: one float32-sized error, then float64",
    "roundoff": "floor B: float32 state truncation every step",
    "mjx_f32": "MJX float32 vs MuJoCo C, same dt",
    "mjx_f64": "MJX float64 vs MuJoCo C, same dt",
    "mjc_dt": "MuJoCo C at dt vs the refined reference",
}


def _style(dark: bool) -> None:
    if dark:
        try:
            from common.plotting import use_dark_style
            use_dark_style()
            return
        except Exception:                       # noqa: BLE001
            pass
        plt.rcParams.update({
            "figure.facecolor": "#12141a", "axes.facecolor": "#12141a",
            "savefig.facecolor": "#12141a", "text.color": "#e8eaf0",
            "axes.labelcolor": "#e8eaf0", "xtick.color": "#b9bfcc",
            "ytick.color": "#b9bfcc", "axes.edgecolor": "#39404f",
            "grid.color": "#242835", "legend.edgecolor": "#39404f",
        })


def _load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------

def fig_parity(dark: bool) -> list[Path]:
    doc = _load("parity.json")
    if not doc:
        return []
    out = []
    for key, v in doc.items():
        if "curves_vs_mjc" not in v:
            continue
        dt = v["dt"]
        env = np.asarray(v["curves_vs_mjc"]["envelope"])
        t = np.arange(env.size) * dt

        fig, ax = plt.subplots(figsize=(8, 4.6))
        # The floors are drawn as filled regions from zero, not as lines. A
        # curve inside a shaded band reads as "indistinguishable from
        # nothing"; two lines that happen to overlap read as a coincidence the
        # viewer is invited to doubt. Floor B is the one that matters, so it
        # is the larger, lighter band; floor A is drawn inside it because
        # seeing how far apart they are is half the point of the figure.
        ro = v["curves_vs_mjc"].get("roundoff")
        if ro is not None:
            ro = np.maximum(np.asarray(ro), 1e-18)
            ax.fill_between(t, 1e-18, ro, color=COLORS["roundoff"],
                            alpha=0.22, lw=0, label=LABELS["roundoff"])
            ax.plot(t, ro, color=COLORS["roundoff"], lw=1.0, alpha=0.9)

        ax.fill_between(t, 1e-18, np.maximum(env, 1e-18),
                        color=COLORS["envelope"], alpha=0.40, lw=0,
                        label=LABELS["envelope"])
        ax.plot(t, np.maximum(env, 1e-18), color=COLORS["envelope"], lw=1.0, alpha=0.9)

        for name in ("mjx_f32", "mjx_f64"):
            if name in v["curves_vs_mjc"]:
                y = np.maximum(np.asarray(v["curves_vs_mjc"][name]), 1e-18)
                ax.plot(t, y, color=COLORS[name], lw=1.7, label=LABELS[name])

        if "mjc_dt" in v.get("curves_vs_ref", {}):
            y = np.maximum(np.asarray(v["curves_vs_ref"]["mjc_dt"]), 1e-18)
            ax.plot(t, y, color=COLORS["mjc_dt"], lw=1.7, ls="--",
                    label=LABELS["mjc_dt"])

        ax.axhline(1e-3, color="#8892a6", lw=0.8, ls=":", alpha=0.7)
        ax.text(t[-1], 1.3e-3, "1 mm", ha="right", va="bottom",
                fontsize=8, color="#8892a6")

        ax.set_yscale("log")
        ax.set_xlabel("simulated time (s)")
        ax.set_ylabel("max body position difference (m)")
        m = v.get("model", {})
        ax.set_title(f"{key}  ·  nv={m.get('nv')}  ngeom={m.get('ngeom')}  "
                     f"dt={dt*1e3:.1f} ms  ·  {v.get('regime','')}", fontsize=10)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
        fig.tight_layout()

        MEDIA.mkdir(parents=True, exist_ok=True)
        p = MEDIA / f"parity_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
    return out


def fig_throughput(dark: bool) -> Path | None:
    doc = _load("throughput.json")
    if not doc:
        return None

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = plt.get_cmap("viridis")
    keys = [k for k, v in doc.items() if v.get("points")]

    for i, key in enumerate(keys):
        v = doc[key]
        c = cmap(0.12 + 0.76 * i / max(1, len(keys) - 1))
        ok = [(p["n_envs"], p["steps_per_s"]) for p in v["points"] if p["steps_per_s"]]
        if not ok:
            continue
        x, y = zip(*ok)
        ax.plot(x, y, "o-", color=c, ms=3.5, lw=1.5, label=key)

        cpu = v.get("cpu_baseline", {}).get("steps_per_s")
        if cpu:
            ax.axhline(cpu, color=c, lw=0.9, ls=":", alpha=0.75)

        k = v.get("knee")
        if k and k.get("steps_per_s"):
            ax.plot([k["n_envs"]], [k["steps_per_s"]], "*", color=c, ms=13,
                    markeredgecolor="none")

        comp = [(p["n_envs"], p["compile_s"]) for p in v["points"] if p["compile_s"]]
        if comp:
            cx, cy = zip(*comp)
            ax2.plot(cx, cy, "o-", color=c, ms=3.5, lw=1.5, label=key)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("environments stepped in parallel")
    ax.set_ylabel("environment steps / second")
    ax.set_title("Throughput vs batch size\n"
                 "dotted = threaded MuJoCo C on the same model; star = knee",
                 fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7.5)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("environments stepped in parallel")
    ax2.set_ylabel("XLA compile time (s)")
    ax2.set_title("What you pay before the first step", fontsize=10)
    ax2.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = MEDIA / "throughput.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_solver(dark: bool) -> Path | None:
    doc = _load("solver.json")
    if not doc:
        return None

    keys = [k for k, v in doc.items() if v.get("rows")]
    if not keys:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    cmap = plt.get_cmap("viridis")

    for i, key in enumerate(keys):
        v = doc[key]
        c = cmap(0.12 + 0.76 * i / max(1, len(keys) - 1))
        rows = [r for r in v["rows"] if r.get("mjx_steps_per_s")]
        if not rows:
            continue
        x = [r["mjx_steps_per_s"] for r in rows]
        y = [max(r["final_m"], 1e-18) for r in rows]
        ax.plot(x, y, "o-", color=c, ms=4, lw=1.4, label=key)
        for r, xi, yi in zip(rows, x, y):
            ax.annotate(str(r["iterations"]), (xi, yi), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points", color=c)
        shipped = v.get("shipped", {}).get("iterations")
        for r, xi, yi in zip(rows, x, y):
            if r["iterations"] == shipped:
                ax.plot([xi], [yi], "o", ms=12, mfc="none", mec=c, mew=1.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MJX throughput (steps / s)  →  faster")
    ax.set_ylabel("final divergence from the tight reference (m)  →  worse")
    ax.set_title("Solver iteration budget: the actual trade\n"
                 "labels are iteration counts; ring marks the shipped default",
                 fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()

    MEDIA.mkdir(parents=True, exist_ok=True)
    p = MEDIA / "solver.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    _style(args.dark)

    made = [*fig_parity(args.dark)]
    for f in (fig_throughput(args.dark), fig_solver(args.dark)):
        if f:
            made.append(f)

    if not made:
        raise SystemExit("no results/*.json found; run s10_gpu_scaling.run first")
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
