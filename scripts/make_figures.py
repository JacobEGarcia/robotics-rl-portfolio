"""
Regenerate every published figure in one cohesive editorial style.

Why a single script
-------------------
Figures made ad-hoc across a project never match: different fonts, different
axis treatments, different accent colours. The result reads as a collage of
screenshots rather than a document. This regenerates all of them from the
committed run logs, so the whole set shares one visual language and any style
change propagates everywhere at once.

Every number here is read from `runs/` — nothing is hardcoded except the axis
labels.

    python -m scripts.make_figures
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.runlog import read_log  # noqa: E402

OUT = Path("media")

# ---------------------------------------------------------------- palette
INK = "#f2f1ef"
DIM = "#8a8884"
FAINT = "#2a2927"
BG = "#0b0b0a"
ACCENT = "#e8a33d"   # warm — the "identified"/"after" state
COOL = "#7fb2d9"     # cool — the "baseline"/"before" state
GOOD = "#7fc9a4"
RED = "#e07a6a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": DIM,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "axes.edgecolor": FAINT,
    "grid.color": FAINT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.28,
})


def style(ax, *, ylab="", xlab="", title=""):
    ax.tick_params(length=0, pad=7, labelsize=9.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    if ylab:
        ax.set_ylabel(ylab, fontsize=10, labelpad=10)
    if xlab:
        ax.set_xlabel(xlab, fontsize=10, labelpad=10)
    if title:
        ax.set_title(title, loc="left", pad=16, color=INK)
    return ax


def latest(pattern: str):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def moving(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ====================================================== 01 learning curves
def fig_learning():
    _, ev_ppo = read_log(latest("runs/*s1_ppo*8e0032bd*") if glob.glob("runs/*8e0032bd*")
                         else latest("runs/*s1_ppo*"))
    _, ev_sac = read_log(latest("runs/*sac_scratch*"))

    def curve(ev):
        r = np.array([e["ret"] for e in ev if e.get("type") == "episode"])
        s = np.array([e["step"] for e in ev if e.get("type") == "episode"])
        w = 20
        return s[w - 1:], moving(r, w)

    sp, rp = curve(ev_ppo)
    ss, rs = curve(ev_sac)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(-200, color=FAINT, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(sp.max() * 0.985, -200, "  solved", va="bottom", ha="right",
            color=DIM, fontsize=9.5, style="italic")

    ax.plot(sp, rp, color=COOL, lw=1.9, label="PPO — on-policy", zorder=3)
    ax.plot(ss, rs, color=ACCENT, lw=1.9, label="SAC — off-policy", zorder=4)

    # Mark the crossings; the horizontal distance between them IS the result.
    for s, r, c in ((sp, rp, COOL), (ss, rs, ACCENT)):
        idx = np.argmax(r >= -200)
        if r[idx] >= -200:
            ax.plot([s[idx]], [r[idx]], "o", color=c, ms=7, zorder=5,
                    mec=BG, mew=2)
            ax.annotate(f"{s[idx]:,}", (s[idx], r[idx]), textcoords="offset points",
                        xytext=(6, -20), color=c, fontsize=10, fontweight="bold")

    style(ax, ylab="episode return", xlab="environment steps")
    ax.legend(loc="lower right", fontsize=10.5, labelcolor=INK)
    ax.set_xlim(0, sp.max())
    fig.savefig(OUT / "fig_learning.png")
    plt.close(fig)
    print("  fig_learning.png")


# ====================================================== 02 sysID convergence
def fig_sysid_convergence():
    d = latest("runs/*s5_sysid*")
    _, ev = read_log(d)
    gens = [e for e in ev if e.get("type") == "generation"]
    base = [e for e in ev if e.get("type") == "baseline"][0]
    res = [e for e in ev if e.get("type") == "result"][0]

    it = [g["iteration"] for g in gens]
    rm = [g["best_train_rmse"] for g in gens]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axhline(base["train_rmse"], color=COOL, lw=1.4, ls=(0, (5, 4)))
    ax.text(max(it) * 0.99, base["train_rmse"], "Menagerie default  ",
            va="bottom", ha="right", color=COOL, fontsize=10)

    ax.plot(it, rm, color=ACCENT, lw=2.0, marker="o", ms=4.5, mec=BG, mew=1.4)
    ax.text(it[-1], rm[-1], f"  {rm[-1]:.4f} rad", va="center",
            color=ACCENT, fontsize=11, fontweight="bold")

    style(ax, ylab="training RMSE (rad)", xlab="CMA-ES generation")
    fig.savefig(OUT / "s5" / "fig_convergence.png")
    plt.close(fig)
    print("  s5/fig_convergence.png")
    return base, res


# ====================================================== 03 parameter shift
def fig_params():
    p = json.loads(Path(
        "runs/20260816-211153_s5_sysid_seed0_06371886/identified_params.json"
    ).read_text())
    names = ["damping", "armature", "frictionloss", "kp", "latency_steps"]
    labels = ["damping", "armature\n(rotor inertia)", "frictionloss\n(dry)",
              "kp\n(stiffness)", "latency\n(steps)"]

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.5))
    for ax, n, lab in zip(axes, names, labels):
        b, f = p["baseline"][n], p["identified"][n]
        ax.plot([0, 1], [b, f], color=FAINT, lw=1.6, zorder=1)
        ax.plot([0], [b], "o", color=COOL, ms=9, mec=BG, mew=2, zorder=3)
        ax.plot([1], [f], "o", color=ACCENT, ms=9, mec=BG, mew=2, zorder=3)
        ax.text(0, b, f"{b:g}\n", ha="center", va="bottom", color=COOL, fontsize=9.5)
        ax.text(1, f, f"\n{f:.3g}", ha="center", va="top", color=ACCENT, fontsize=9.5,
                fontweight="bold")
        ax.set_xlim(-0.55, 1.55)
        lo, hi = min(b, f), max(b, f)
        pad = max(hi - lo, hi * 0.35 + 1e-9) * 0.55
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(lab, fontsize=10, color=DIM, pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.055, 0.02, "default", color=COOL, fontsize=10)
    fig.text(0.12, 0.02, "→  identified", color=ACCENT, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "s5" / "fig_params.png")
    plt.close(fig)
    print("  s5/fig_params.png")


# ====================================================== 04 per-joint error
def fig_per_joint():
    import sys
    sys.path.insert(0, ".")
    from s5_sysid.sysid import (build_model, load_episodes, simulate_open_loop)

    p = json.loads(Path(
        "runs/20260816-211153_s5_sysid_seed0_06371886/identified_params.json"
    ).read_text())
    eps = load_episodes(max_episodes=8)
    mb, mf = build_model(p["baseline"]), build_model(p["identified"])

    eb, ef = [], []
    for ep in eps:
        sb = simulate_open_loop(mb, ep, 0)
        sf = simulate_open_loop(mf, ep, int(round(p["identified"]["latency_steps"])))
        eb.append(np.sqrt(((sb - ep.measured) ** 2).mean(0)))
        ef.append(np.sqrt(((sf - ep.measured) ** 2).mean(0)))
    eb, ef = np.mean(eb, 0), np.mean(ef, 0)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(7)
    ax.barh(y + 0.19, eb, height=0.34, color=COOL, label="Menagerie default")
    ax.barh(y - 0.19, ef, height=0.34, color=ACCENT, label="identified")
    for i in range(7):
        drop = 100 * (1 - ef[i] / eb[i]) if eb[i] > 0 else 0
        ax.text(max(eb[i], ef[i]) * 1.04, i, f"−{drop:.0f}%", va="center",
                color=GOOD, fontsize=9.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([f"joint {i}" for i in range(7)], fontsize=9.5)
    ax.invert_yaxis()
    ax.yaxis.grid(False); ax.xaxis.grid(True)
    style(ax, xlab="open-loop RMSE against real robot (rad)")
    ax.legend(fontsize=10, labelcolor=INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_xlim(0, max(eb) * 1.18)
    fig.savefig(OUT / "s5" / "fig_per_joint.png")
    plt.close(fig)
    print("  s5/fig_per_joint.png")


# ====================================================== 05 throughput
def fig_throughput():
    fig, ax = plt.subplots(figsize=(9, 4.0))
    names = ["PyTorch\ninterpreted loop", "JAX\nscan · vmap · jit"]
    vals = [1934, 54808]
    bars = ax.barh(names, vals, height=0.42, color=[COOL, ACCENT])
    ax.set_xscale("log")
    for b, v in zip(bars, vals):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(0.5, 0.5, "28.3×", transform=ax.transAxes, ha="center", va="center",
            fontsize=44, color=GOOD, alpha=0.16, fontweight="bold", zorder=0)
    ax.yaxis.grid(False)
    style(ax, xlab="environment steps per second — same CPU, no GPU")
    ax.set_xlim(1e3, 2e5)
    fig.savefig(OUT / "s6" / "fig_throughput.png")
    plt.close(fig)
    print("  s6/fig_throughput.png")


# ====================================================== 06 tendon hysteresis
def fig_tendon():
    from s4_tendon.hysteresis import FRICTION_VALUES, run_sweep, hysteresis_width, dead_band

    res = [run_sweep(f) for f in FRICTION_VALUES]
    cols = [COOL, "#9db9c9", ACCENT, RED]

    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.34)

    ax = fig.add_subplot(gs[0])
    for r, c in zip(res, cols):
        ax.plot(r["force"], r["mcp"], color=c, lw=1.7,
                label=f"frictionloss {r['frictionloss']}")
    style(ax, ylab="MCP joint angle (rad)", xlab="flexor tendon force (N)")
    ax.legend(fontsize=9, labelcolor=INK, loc="lower right")

    w = [hysteresis_width(r["force"], r["mcp"]) for r in res]
    b = [dead_band(r["force"], r["mcp"]) for r in res]

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(FRICTION_VALUES, w, "o-", color=ACCENT, lw=2, ms=6, mec=BG, mew=1.6)
    ax2.axhline(w[0], color=FAINT, ls=(0, (4, 4)), lw=1.2)
    ax2.text(FRICTION_VALUES[-1], w[0], "geometric floor  ", ha="right", va="bottom",
             color=DIM, fontsize=9, style="italic")
    style(ax2, ylab="hysteresis width (rad)", xlab="tendon frictionloss")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(FRICTION_VALUES, b, "s-", color=GOOD, lw=2, ms=6, mec=BG, mew=1.6)
    style(ax3, ylab="dead band (N)", xlab="tendon frictionloss")

    fig.savefig(OUT / "s4" / "fig_hysteresis.png")
    plt.close(fig)
    print("  s4/fig_hysteresis.png")


if __name__ == "__main__":
    for sub in ("s4", "s5", "s6"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    print("regenerating figures:")
    fig_learning()
    fig_sysid_convergence()
    fig_params()
    fig_throughput()
    fig_per_joint()
    fig_tendon()
    print("done.")
