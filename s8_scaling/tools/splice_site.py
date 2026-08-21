"""
Splice the S8 chapter into docs/index.html and copy its media.

Idempotent: refuses to run twice (checks for id="scaling"). Renumbers the
existing chapters and plates, adds the nav link, ledger figures, the five
S8 entries in the bug ledger, and the status row.

Run:  python -m s8_scaling.tools.splice_site <chapter_fragment.html>
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

N_S8_PLATES = 9
WORDS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven"]


def replace_chapter(fragment_path: str, ledger: dict[str, str] | None = None) -> None:
    """Swap the existing S8 article for a new fragment; optionally update ledger numbers."""
    index = Path("docs/index.html")
    html = index.read_text(encoding="utf-8")
    start = html.index("<!-- ============================================ I — SCALING")
    end = html.index("<!-- ============================================ I — REALITY GAP")
    chapter = Path(fragment_path).read_text(encoding="utf-8")
    html = html[:start] + chapter + "\n" + html[end:]
    for old, new in (ledger or {}).items():
        assert old in html, old
        html = html.replace(old, new)
    index.write_text(html, encoding="utf-8")
    dst = Path("docs/media/s8")
    for fig in Path("s8_scaling/figures/dark").glob("*.png"):
        shutil.copy(fig, dst / fig.name)
    for vid in ("data_engine.mp4", "policy.mp4"):
        src = Path("s8_scaling/media") / vid
        if src.exists():
            shutil.copy(src, dst / vid)
    print("replaced S8 chapter and refreshed media")


def main(fragment_path: str) -> None:
    index = Path("docs/index.html")
    html = index.read_text(encoding="utf-8")
    if 'id="scaling"' in html:
        raise SystemExit("S8 chapter already spliced; use replace_chapter()")
    chapter = Path(fragment_path).read_text(encoding="utf-8")

    # --- renumber existing chapters (highest first so replacements don't chain)
    for i in range(len(WORDS) - 2, -1, -1):
        html = html.replace(f"<p class=\"chapno\">Chapter {WORDS[i]}</p>",
                            f"<p class=\"chapno\">Chapter {WORDS[i + 1]}</p>")

    # --- renumber existing plates
    html = re.sub(r"Plate (\d+) ·", lambda m: f"Plate {int(m.group(1)) + N_S8_PLATES} ·", html)
    html = re.sub(r"Plates (\d+) &amp; (\d+)",
                  lambda m: f"Plates {int(m.group(1)) + N_S8_PLATES} &amp; {int(m.group(2)) + N_S8_PLATES}",
                  html)

    # --- nav
    html = html.replace('    <a href="#gap">Reality Gap</a>\n',
                        '    <a href="#scaling">Scaling</a>\n    <a href="#gap">Reality Gap</a>\n')

    # --- title page
    html = html.replace(
        '<p class="kicker">Robotics · Reinforcement Learning · Sim-to-Real</p>',
        '<p class="kicker">Robotics · Scaling Laws · Reinforcement Learning · Sim-to-Real</p>')
    html = html.replace(
        '<p class="standfirst">Six projects on the distance between a simulator and a physical robot —\n'
        '     and what it takes to measure that distance honestly.</p>',
        '<p class="standfirst">Seven projects on what it takes to measure a robot-learning claim honestly —\n'
        '     from the scaling laws of imitation to the distance between a simulator and a physical arm.</p>')

    # --- ledger
    old_ledger_start = html.index('<div class="ledgergrid">')
    old_ledger_end = html.index('</div>', old_ledger_start) + len('</div>')
    ledger = '''<div class="ledgergrid">
    <figure><span class="n amber">0.98</span><span class="l">R² of the power-law fit<br>over a 64× data range</span></figure>
    <figure><span class="n sky">+32<span style="font-size:26px"> pts</span></span><span class="l">Paired transfer gain<br>from pretraining</span></figure>
    <figure><span class="n">56.7%</span><span class="l">Reality gap closed<br>on held-out data</span></figure>
    <figure><span class="n sage">28.3×</span><span class="l">JAX throughput<br>same CPU</span></figure>
    <figure><span class="n">70</span><span class="l">Training runs<br>in the scaling grid</span></figure>
    <figure><span class="n">12</span><span class="l">Silent bugs found<br>none raised an exception</span></figure>
  </div>'''
    html = html[:old_ledger_start] + ledger + html[old_ledger_end:]

    # --- insert chapter before the old Chapter One marker
    marker = "<!-- ============================================ I — REALITY GAP"
    i = html.index(marker)
    html = html[:i] + chapter + "\n" + html[i:]

    # --- method chapter: title + five new bugs
    html = html.replace("<h2>Seven bugs, none of which raised an <i>exception</i></h2>",
                        "<h2>Twelve bugs, none of which raised an <i>exception</i></h2>")
    new_bugs = '''    <div class="bug"><div>
      <h5>A resized geom with a stale bounding radius</h5>
      <p>Writing <code>geom_size</code> at reset does not update the compiled collision bounds.
         Pillars sampled taller than their XML size had their table contact culled by the
         broadphase and settled a centimetre inside the table, contact flickering one tick in
         five. Found by review before any training run; fixed by writing
         <code>geom_rbound</code> and <code>geom_aabb</code> alongside the size.</p></div></div>
    <div class="bug"><div>
      <h5>Friction randomisation that was never applied</h5>
      <p>MuJoCo combines the friction of two geoms by taking the maximum. Against a 0.8 table and
         1.0 fingertip pads, most of the sampled [0.6, 1.2] range was inert. The data card would
         have described a randomisation the physics never saw. Objects now carry contact
         priority, so the sampled coefficient is the one that acts.</p></div></div>
    <div class="bug"><div>
      <h5>Parked objects remembering the previous episode</h5>
      <p>Objects a task family does not use are parked out of the workspace — with whatever size
         the previous family gave them. A 10 cm pillar parked at a 5 cm object's height sat
         inside the table and bounced for the whole of the next episode. Episode physics depended
         on what ran before it, and the documented seed-to-episode determinism was false.</p></div></div>
    <div class="bug"><div>
      <h5>A validation set that moved with the data scale</h5>
      <p>Each grid cell validated on the held-out episodes of its own data prefix, so every point
         on the hours axis was scored on a different set. Error appeared to rise from sixteen to
         thirty-two hours. Re-scoring every checkpoint on one common held-out set made every
         curve monotone.</p></div></div>
    <div class="bug"><div>
      <h5>Confidence intervals of zero width</h5>
      <p>Bootstrap percentile intervals on success flags degenerate at 0% and 100%: a policy that
         fails every episode resamples to itself and reports certainty. Replaced with Wilson
         intervals, which have correct coverage at the extremes — where a reader most needs to
         see uncertainty.</p></div></div>
  </div>
'''
    bugs_end = html.index("</div>\n\n  <div class=\"col\">\n    <h3>What reproducibility means here</h3>")
    html = html[:bugs_end] + new_bugs + html[bugs_end + len("</div>\n"):]

    # --- run commands block
    html = html.replace(
        'python -m s5_sysid.fit       --iterations 45\n',
        'python -m s8_scaling.data_engine --hours 33   # then .sweep .reeval .transfer .eval_sweep\n'
        'python -m s5_sysid.fit       --iterations 45\n')

    # --- status board
    html = html.replace(
        '  <div class="status">\n',
        '  <div class="status">\n'
        '    <div><span class="s done">Complete</span><span class="t">Scaling laws for behaviour cloning</span>'
        '<span class="d">70-run grid, α ≈ 0.15, +32 pt paired transfer</span></div>\n')

    # --- footer link
    html = html.replace(
        '    <a href="https://droid-dataset.github.io/">DROID</a> ·',
        '    <a href="https://generalistai.com/blog/nov-04-2025-GEN-0">GEN-0</a> ·\n'
        '    <a href="https://droid-dataset.github.io/">DROID</a> ·')

    index.write_text(html, encoding="utf-8")

    # --- media
    dst = Path("docs/media/s8")
    dst.mkdir(parents=True, exist_ok=True)
    for fig in Path("s8_scaling/figures/dark").glob("*.png"):
        shutil.copy(fig, dst / fig.name)
    for vid in ("data_engine.mp4", "policy.mp4"):
        src = Path("s8_scaling/media") / vid
        if src.exists():
            shutil.copy(src, dst / vid)
    print("spliced chapter, renumbered, copied media to docs/media/s8")


if __name__ == "__main__":
    main(sys.argv[1])
