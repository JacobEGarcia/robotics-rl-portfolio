"""
Generate the Colab notebooks from this file.

    python colab/build.py

`.ipynb` is JSON with source split into lines and no way to diff sensibly, so
the notebooks are generated rather than edited. Same reason the manga guide's
`index.html` is generated from `parts/`: edit the source, never the artifact.

Three notebooks, in the order they should be run:

    S10_gpu_scaling.ipynb   one session, no training, produces numbers
    S2_inhand.ipynb         the flagship: PPO on the LEAP hand, many sessions
    S3_domain_rand.ipynb    two training arms plus the held-out evaluation

S10 is first on purpose. It finishes inside a single free session, it measures
the throughput that decides how S2 should be configured, and it produces a
result even if the GPU quota runs out that afternoon. Starting with the
multi-day training run and discovering the batch size was wrong is the
expensive ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/JacobEGarcia/robotics-rl-portfolio.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ==========================================================================
# Shared cells
# ==========================================================================

HEADER_WARNING = """
### Before you run anything

1. **Runtime → Change runtime type → T4 GPU.** Every cell below assumes it.
2. **Keep this tab visible.** Free Colab disconnects an idle notebook after
   about 90 minutes and reclaims the runtime; `/content` does not survive it.
3. **The free tier has a quota you cannot see.** It is not published, it
   varies, and it is consumed by wall-clock GPU time whether or not you are
   computing. Expect a few hours a day, and expect to be cut off mid-run
   without warning. Every long-running cell here is written to survive that.

Checkpoints go to Google Drive, not to `/content`. That is the whole reason
the Drive cell exists, a run that checkpoints only to local disk loses
everything the moment the runtime is reclaimed, which on free Colab is the
normal way a session ends rather than an exceptional one.
"""

CELL_GPU = '''
# --- what hardware did we actually get? ---
import subprocess, sys
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                      '--format=csv'], capture_output=True, text=True).stdout)

# A label is not hardware. A Kaggle session advertised as "T4 x2" reported a
# P100 to the driver, which has compute capability 6.0 and cannot run several
# things a T4 can. Record what the driver says and quote it as such; never
# write "measured on a T4" because the runtime menu said T4.
'''

CELL_INSTALL = '''
# --- packages ---
# jax with CUDA is preinstalled on Colab GPU runtimes. mujoco-mjx is not, and
# installing it can drag in a CPU-only jax wheel that silently replaces the
# working one. So: install, then re-check the device, and only reinstall jax
# if the check fails.
import subprocess, sys

def sh(cmd):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True)
    if r.returncode:
        raise SystemExit(f'command failed: {cmd}')

sh(f'{sys.executable} -m pip install -q mujoco mujoco-mjx optax')

import importlib, jax
importlib.reload(jax)
if not any(d.platform == 'gpu' for d in jax.devices()):
    print('jax lost the GPU during install; reinstalling the CUDA wheel')
    sh(f'{sys.executable} -m pip install -q -U "jax[cuda12]"')
    raise SystemExit(
        'Reinstalled jax. Runtime -> Restart session, then run this cell '
        'again. (A restart is required: the CPU-only jax is already imported '
        'into this process and reimporting will not replace it.)')
'''

CELL_VERIFY = '''
import jax, mujoco
print('jax     ', jax.__version__)
print('mujoco  ', mujoco.__version__)
print('devices ', jax.devices())
print('kind    ', getattr(jax.devices()[0], 'device_kind', '?'), '(as reported by the driver)')

# Hard stop, not a warning. On CPU a single mjx.step of the LEAP scene costs
# about 4 seconds and the compile runs past half an hour: a CPU session is not
# a slow run, it is no run.
assert any(d.platform == 'gpu' for d in jax.devices()), \\
    'No GPU. Runtime -> Change runtime type -> T4 GPU, then restart and re-run.'
print()
print('GPU OK')
'''

CELL_DRIVE = '''
# --- Drive, for anything that must outlive this runtime ---
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

DRIVE = Path('/content/drive/MyDrive/robotics-rl-portfolio')
DRIVE.mkdir(parents=True, exist_ok=True)
WORK = Path('/content/work'); WORK.mkdir(exist_ok=True)
print('drive :', DRIVE)
print('local :', WORK)
'''


def cell_repo(needs: str) -> str:
    return f'''
# --- code ---
import os, shutil, subprocess, sys
from pathlib import Path

SRC = WORK / 'robotics-rl-portfolio'
REPO = '{REPO_URL}'

def sh(cmd, **kw):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)

if SRC.exists():
    sh(f'cd {{SRC}} && git pull --ff-only')
else:
    sh(f'git clone --depth 1 {{REPO}} {{SRC}}')

need = SRC / '{needs}'
if not need.exists():
    # Fallback for code that is committed locally but not pushed. Tar the
    # repo on your machine, drop it in Drive, and this picks it up:
    #   tar czf portfolio.tgz --exclude=assets --exclude=runs .
    tgz = DRIVE / 'portfolio.tgz'
    if tgz.exists():
        print(f'{{need}} missing from the clone; unpacking {{tgz}} over it')
        sh(f'tar xzf {{tgz}} -C {{SRC}}')
    if not need.exists():
        raise SystemExit(
            f'{{need}} is not in the cloned repo and no {{tgz}} was found.\\n'
            f'Push it from your machine:\\n'
            f'    cd ~/Downloads/hermestes/robotics-rl-portfolio && git push origin main\\n'
            f'or upload a tarball to {{tgz}}.')

os.environ['PYTHONPATH'] = str(SRC)
sys.path.insert(0, str(SRC))
print()
print('code OK at', SRC)
'''


CELL_MENAGERIE = '''
# --- the robot models ---
# assets/menagerie is gitignored in the portfolio repo on purpose: Menagerie
# is 2.3 GB of third-party assets and is itself a git repo, so it is fetched
# rather than vendored. A sparse checkout gets what is needed in a few
# seconds instead of pulling all of it.
MENAGERIE = SRC / 'assets' / 'menagerie'
WANT = ['leap_hand', 'franka_emika_panda', 'unitree_z1',
        'unitree_go1', 'unitree_h1', 'shadow_hand', 'dynamixel_2r']

if not (MENAGERIE / 'leap_hand' / 'right_hand.xml').exists():
    shutil.rmtree(MENAGERIE, ignore_errors=True)
    MENAGERIE.parent.mkdir(parents=True, exist_ok=True)
    sh('git clone --depth 1 --filter=blob:none --sparse '
       'https://github.com/google-deepmind/mujoco_menagerie.git ' + str(MENAGERIE))
    sh(f'cd {MENAGERIE} && git sparse-checkout set ' + ' '.join(WANT))

missing = [w for w in WANT if not (MENAGERIE / w).exists()]
assert not missing, f'sparse checkout did not produce: {missing}'
print('models OK:', sorted(p.name for p in MENAGERIE.iterdir() if p.is_dir())[:12])
'''


def bootstrap(needs: str, title: str, blurb: str) -> list[dict]:
    return [
        md(f"# {title}\n\n{blurb}\n\n---\n{HEADER_WARNING}"),
        md("## 1. Hardware, packages, Drive, code"),
        code(CELL_GPU),
        code(CELL_INSTALL),
        code(CELL_VERIFY),
        code(CELL_DRIVE),
        code(cell_repo(needs)),
        code(CELL_MENAGERIE),
    ]


# ==========================================================================
# S10
# ==========================================================================

S10_BLURB = """
Measures what a GPU actually buys a physics simulator, and what it costs in
fidelity. **No training**, this finishes inside one free session and produces
numbers whether or not the quota holds out.

Three questions:

1. **Does MJX agree with MuJoCo C?** Not "do the trajectories overlap" (they
   never do, the systems are chaotic) but "does MJX diverge by more than
   float32 alone would".
2. **Steps per second against batch size**, with a threaded MuJoCo C baseline
   on the same model, so the speedup has a denominator.
3. **What the solver iteration budget costs and buys**, on the GPU, where the
   trade is different from CPU.

Run this first. It also tells you the batch size to configure S2 with.
"""

S10_CELLS = [
    md("## 2. Fidelity: MJX against MuJoCo C\n\n"
       "About 10-20 minutes. The float64 pass runs as a subprocess because "
       "`jax_enable_x64` is process-global and MJX fixes its array dtypes when "
       "the model is built."),
    code('''
import subprocess, sys, os
os.chdir(SRC)

MODELS = ['dynamixel_2r', 'unitree_z1', 'franka_emika_panda',
          'leap_hand', 's2_inhand', 'unitree_go1', 'unitree_h1']

for cmd in (['parity', '--steps', '300'],
            ['parity-x64', '--steps', '300'],
            ['integrator', '--steps', '40']):
    r = subprocess.run([sys.executable, '-u', '-m', 's10_gpu_scaling.run',
                        *cmd, '--models', *MODELS],
                       env=dict(os.environ, PYTHONPATH=str(SRC)))
    print('exit', r.returncode)
'''),
    md("### What the numbers mean\n\n"
       "For each model you get five curves. The two that decide the verdict:\n\n"
       "* **floor B (`roundoff`)**, MuJoCo C, float64 arithmetic, state\n"
       "  truncated to float32 after every step. A lower bound on what\n"
       "  float32 costs, measured in the same binary with no implementation\n"
       "  difference in it.\n"
       "* **`mjx_f32`**, MJX against MuJoCo C at the same timestep.\n\n"
       "`mjx_f32` near or below floor B means precision explains the gap.\n"
       "`mjx_f64` still large is the only evidence of an algorithmic\n"
       "difference, and then the place to look is the constraint solver.\n\n"
       "Do **not** read `mjx_f32` against floor A (`envelope`, a single\n"
       "float32-sized perturbation then pure float64). Floor A runs tens of\n"
       "thousands of times below floor B and makes every port look broken."),
    code('''
import json
from pathlib import Path
d = json.loads((SRC / 's10_gpu_scaling/results/parity.json').read_text())
print(f"{'model':<20}{'floorA':>11}{'floorB':>11}{'mjx_f32':>11}{'mjx_f64':>11}{'dt cost':>11}")
for k, v in d.items():
    if 'summary' not in v:
        print(f'{k:<20} {v.get("error","?")[:60]}'); continue
    g = lambda n: v['summary'].get(n, {}).get('final_m')
    fmt = lambda x: f'{x:>11.2e}' if x is not None else f'{"-":>11}'
    print(f'{k:<20}' + ''.join(fmt(g(n)) for n in
          ('envelope', 'roundoff', 'mjx_f32', 'mjx_f64', 'mjc_dt')))
'''),

    md("""
### Localising a disagreement: the integrator

Measured on CPU, this is where the Panda's whole MJX/MuJoCo gap lives, and it
is neither precision nor contacts. Under `mjINT_EULER` the two agree to
1.4e-07 m, five orders of magnitude below the float32 floor. Under
`mjINT_IMPLICITFAST`, which is what `panda.xml`, `leap_hand` and **the S2
scene itself** ship, they are 16 mm apart within 40 ms. MJX refuses
`mjINT_IMPLICIT` outright.

The Euler row is also the positive control this study needs: it proves the
harness can show agreement, so the large numbers elsewhere are results rather
than bugs in the comparison. Check whether the CUDA backend reproduces it.
"""),
    code('''
d = json.loads((SRC / 's10_gpu_scaling/results/integrator.json').read_text())
for k, v in d.items():
    print(k, '(ships ' + v['shipped'] + ')')
    for r in v['rows']:
        if 'unsupported' in r:
            print(f"  {r['integrator']:<20} NOT SUPPORTED BY MJX")
        elif 'error' in r:
            print(f"  {r['integrator']:<20} {r['error'][:70]}")
        else:
            print(f"  {r['integrator']:<20} {r['mjx_vs_mjc_m']:.3e} m  "
                  f"floor {r['float32_floor_m']:.3e}  ratio {r['ratio_to_floor']:>9.5f}")
'''),

    md("## 3. Throughput: steps/s against batch size\n\n"
       "20-40 minutes. The sweep climbs until two consecutive allocations "
       "fail, so it finds the memory ceiling rather than assuming one. "
       "Compile time is reported separately: it is a real cost when you are "
       "sizing a job against a session that might last two hours."),
    code('''
r = subprocess.run([sys.executable, '-u', '-m', 's10_gpu_scaling.run',
                    'throughput', '--max-envs', '16384',
                    '--models', 'dynamixel_2r', 'franka_emika_panda',
                    'leap_hand', 's2_inhand', 'unitree_go1'],
                   env=dict(os.environ, PYTHONPATH=str(SRC)))
print('exit', r.returncode)
'''),
    code('''
d = json.loads((SRC / 's10_gpu_scaling/results/throughput.json').read_text())
print(f"{'model':<20}{'cpu':>12}{'gpu best':>12}{'at n':>8}{'x cpu':>8}{'knee n':>8}{'compile':>9}")
for k, v in d.items():
    b, kn, c = v.get('best'), v.get('knee'), v.get('cpu_baseline', {})
    if not b:
        continue
    print(f"{k:<20}{c.get('steps_per_s',0):>12,.0f}{b['steps_per_s']:>12,.0f}"
          f"{b['n_envs']:>8,}{(v.get('gpu_over_cpu') or 0):>8.1f}"
          f"{(kn or {}).get('n_envs',0):>8,}{b['compile_s']:>8.0f}s")

s2 = d.get('s2_inhand', {})
if s2.get('knee'):
    print()
    print(f"=> configure S2 with --num-envs {s2['knee']['n_envs']}")
    print(f"   ({s2['knee']['steps_per_s']:,.0f} steps/s; "
          f"1e8 steps ~ {1e8/s2['knee']['steps_per_s']/3600:.1f} GPU-hours)")
'''),

    md("## 4. The solver iteration budget\n\n"
       "The knob that actually sets the speed/fidelity trade on a GPU. On CPU "
       "an environment whose constraints are already satisfied exits the "
       "solver early; under `vmap` the whole batch runs the full budget every "
       "step, so a value tuned on CPU is usually wrong here.\n\n"
       "Worth noticing in the output: DeepMind's own MJX demo scene "
       "(`panda_mjx_cube`) ships `iterations=5` where `panda.xml` ships 100."),
    code('''
r = subprocess.run([sys.executable, '-u', '-m', 's10_gpu_scaling.run',
                    'solver', '--n-envs', '512',
                    '--models', 'franka_emika_panda', 'leap_hand', 'unitree_go1'],
                   env=dict(os.environ, PYTHONPATH=str(SRC)))
print('exit', r.returncode)
'''),

    md("## 5. Figures, and get the results off this machine\n\n"
       "Copy to Drive **before** doing anything else. A runtime reclaimed "
       "with results only in `/content` produced nothing."),
    code('''
r = subprocess.run([sys.executable, '-u', '-m', 's10_gpu_scaling.figures', '--dark'],
                   env=dict(os.environ, PYTHONPATH=str(SRC)))

import shutil
out = DRIVE / 's10_results'
out.mkdir(parents=True, exist_ok=True)
for sub in ('s10_gpu_scaling/results', 'media/s10'):
    src = SRC / sub
    if src.exists():
        shutil.copytree(src, out / Path(sub).name, dirs_exist_ok=True)
print('copied to', out)
print(sorted(p.name for p in out.rglob('*') if p.is_file()))
'''),
]

# ==========================================================================
# S2
# ==========================================================================

S2_BLURB = """
PPO on in-hand cube reorientation with the LEAP hand, in MJX.

A 5 cm cube starts held in a partially closed hand. A target orientation is
sampled each episode; the policy commands 16 finger joints and must rotate the
cube to within 0.1 rad without dropping it. Roughly 10^8 environment steps.

**This is the only project in the portfolio that has never trained.** It was
written and unit-tested on a machine with no GPU, where one `mjx.step` costs
about 4 seconds. The smoke gate in section 2 is not a formality: until it
prints an iteration line, the training loop has never closed anywhere.

Designed to be run many times. Each session resumes from the newest checkpoint
in Drive.
"""

S2_CELLS = [
    md("## 2. Does the scene still hold the cube?\n\n"
       "Loads `scene.xml` and re-checks the reset grasp. A broken or partial "
       "asset fetch shows up here in seconds instead of twenty minutes into a "
       "compile.\n\n"
       "Initial penetration is a cliff, not a gradient: a configuration "
       "starting the cube 10 mm or more inside a finger geom does not settle "
       "badly, it ejects the cube at tens of metres per second on the first "
       "step. Both numbers are checked."),
    code('''
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(str(SRC / 's2_inhand' / 'scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cube')
cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
pen = min((d.contact[c].dist for c in range(d.ncon)
           if cg in (d.contact[c].geom1, d.contact[c].geom2)), default=0.0)
start = d.xpos[cb].copy()
for _ in range(2500):
    mujoco.mj_step(m, d)
drift = float(np.linalg.norm(d.xpos[cb] - start))

print(f'nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}')
print(f'reset penetration {pen*1e3:.3f} mm   drift over 5 s {drift*1e3:.2f} mm')
assert abs(pen) < 0.010, 'penetration at reset is in the ejection regime'
assert drift < 0.03, 'grasp is not holding; do not train against this'
print('scene OK')
'''),

    md("### Smoke gate, do not skip\n\n"
       "Four environments, two iterations, one checkpoint written and reloaded. "
       "It proves the rollout, the GAE, the PPO update and the checkpoint path "
       "close end to end **on this machine**.\n\n"
       "It must print at least one `step ...` line and one `checkpoint @ ...` "
       "line. Two minutes here against an afternoon of a run that was never "
       "going to work."),
    code('''
import os, subprocess, sys, time
os.chdir(SRC)
SMOKE = WORK / 'smoke_ckpt'

t0 = time.time()
r = subprocess.run([sys.executable, '-u', '-m', 's2_inhand.train', '--smoke',
                    '--ckpt-dir', str(SMOKE)],
                   env=dict(os.environ, PYTHONPATH=str(SRC)),
                   capture_output=True, text=True)
print(r.stdout[-4000:])
if r.stderr.strip():
    print('--- stderr ---'); print(r.stderr[-3000:])
print(f'\\nexit={r.returncode} in {time.time()-t0:.0f}s')

wrote = list(SMOKE.glob('ckpt_*.pkl'))
assert r.returncode == 0, 'smoke run failed, see stderr'
assert wrote, 'smoke run wrote no checkpoint'
assert 'step ' in r.stdout, 'smoke run never closed an iteration'
print('SMOKE GATE PASSED ->', [p.name for p in wrote])
'''),

    md("## 3. Size the batch by measurement, not by guess\n\n"
       "Reuses S10's throughput harness on this exact scene. Picks the "
       "**knee**, the smallest batch within 10% of the best throughput, not "
       "the largest batch that fits. Past the knee each extra environment buys "
       "under 10% while costing proportional memory and a longer compile, and "
       "on a preemptible session a longer compile is a direct loss.\n\n"
       "Takes 5-15 minutes and saves considerably more than that."),
    code('''
from s10_gpu_scaling import throughput

pts = throughput.mjx_sweep(m, [256, 512, 1024, 2048, 4096, 8192], n_steps=50)
k = throughput.knee(pts)
best = max((p for p in pts if p.steps_per_s), key=lambda p: p.steps_per_s, default=None)

NUM_ENVS = k.n_envs if k else 1024
print()
print(f'chosen --num-envs {NUM_ENVS}')
if best:
    print(f'best {best.steps_per_s:,.0f} steps/s at n={best.n_envs:,}')
    print(f'1e8 steps at the knee ~ {1e8/k.steps_per_s/3600:.1f} GPU-hours')
    print(f'  = {1e8/k.steps_per_s/3600/2.5:.1f} sessions at 2.5 h each')
'''),

    md("## 4. Train\n\n"
       "`--resume` searches Drive as well as local disk, so this is the same "
       "cell every session. `--max-hours 2.5` makes the trainer write a final "
       "checkpoint and exit cleanly before the free tier is likely to cut it "
       "off; being killed between checkpoints throws away everything since the "
       "last one.\n\n"
       "**Watch the `curr` column, not the reward.** If the curriculum is not "
       "widening after a few million steps, the reward is not shaping "
       "behaviour and the fix is the reward, not more steps. That is the "
       "expected failure mode for this project and it is worth catching in "
       "hour one rather than hour eleven."),
    code('''
CKPT_LOCAL  = WORK / 'checkpoints'
CKPT_DRIVE  = DRIVE / 's2_checkpoints'
TOTAL_STEPS = 100_000_000
MAX_HOURS   = 2.5

cmd = [sys.executable, '-u', '-m', 's2_inhand.train',
       '--total-steps', str(TOTAL_STEPS),
       '--num-envs', str(NUM_ENVS),
       '--ckpt-dir', str(CKPT_LOCAL),
       '--ckpt-mirror', str(CKPT_DRIVE),
       '--ckpt-every', '2000000',
       '--max-hours', str(MAX_HOURS),
       '--resume']
print('$', ' '.join(cmd), flush=True)

# Stream the output. A multi-hour run that prints nothing until it exits
# cannot be babysat, and the point is to notice a flat curriculum early.
p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=str(SRC)),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
try:
    for line in p.stdout:
        print(line, end='', flush=True)
except KeyboardInterrupt:
    p.terminate()
    print('\\ninterrupted; the newest checkpoint is already in Drive')
p.wait()
'''),

    md("## 5. Where the run is"),
    code('''
import json
for name, d in (('local', CKPT_LOCAL), ('drive', CKPT_DRIVE)):
    cks = sorted(d.glob('ckpt_*.pkl')) if d.exists() else []
    print(f'{name:6} {len(cks)} checkpoint(s)',
          [f'{c.name} ({c.stat().st_size/1e6:.1f} MB)' for c in cks[-3:]])
meta = CKPT_DRIVE / 'latest.json'
if meta.exists():
    print(); print(meta.read_text())
print()
print('To continue: new session, run sections 1 and 4. Skip 2 and 3.')
'''),
]

# ==========================================================================
# S3
# ==========================================================================

S3_BLURB = """
Does randomizing training physics improve performance on physics the policy
never saw?

Two arms, identical in every respect except randomization: a **DR** arm that
samples cube mass, friction, actuator gain, joint damping, observation noise
and integer-frame action latency per episode, and a **baseline** arm with all
of it pinned to nominal. Both are then evaluated on a held-out band that is
disjoint from the training band **by construction**, defined in
`randomize.py` before either arm trained.

This is sim-to-sim transfer, not sim-to-real. There is no physical LEAP hand
here and calling it sim-to-real would be a lie. The methodology is the point
either way, and most published sim-to-real results answer a weaker version of
this question.

**Run S2 first.** Not because S3 needs its weights, the arms train from
scratch, but because S2 tells you whether the task is learnable at all and
what throughput to expect. Two arms is two training budgets.
"""

S3_CELLS = [
    md("## 2. Check the split before spending anything on it\n\n"
       "The whole claim rests on the held-out band being outside the training "
       "band. That is enforced by construction in `randomize.py`, and "
       "verified here against actual draws, because \"by construction\" is a "
       "claim about code and this is the measurement."),
    code('''
import jax, numpy as np
from s2_inhand.env import InHandConfig
from s3_domain_rand.env import DomainRandEnv
from s3_domain_rand.randomize import DomainRandConfig

env = DomainRandEnv(InHandConfig(), DomainRandConfig(), enabled=True)
keys = jax.random.split(jax.random.PRNGKey(0), 4096)
tr = env.sample_params(keys, held_out=False)
te = env.sample_params(keys, held_out=True)

cfg = env.dr
print(f"{'parameter':<16}{'train band':>22}{'held-out draws':>26}{'disjoint':>10}")
ok = True
for name, rng in (('cube_mass', cfg.cube_mass), ('cube_friction', cfg.cube_friction),
                  ('actuator_gain', cfg.actuator_gain), ('joint_damping', cfg.joint_damping)):
    a, b = np.asarray(tr[name]), np.asarray(te[name])
    clean = bool(((b < rng.train_lo) | (b > rng.train_hi)).all())
    ok &= clean
    print(f'{name:<16}[{rng.train_lo:.2f}, {rng.train_hi:.2f}]'.ljust(38)
          + f'[{b.min():.3f}, {b.max():.3f}]'.rjust(26) + f'{str(clean):>10}')

lat_tr, lat_te = np.asarray(tr['latency']), np.asarray(te['latency'])
clean = bool(lat_te.min() > lat_tr.max()); ok &= clean
print(f"{'latency':<16}[{lat_tr.min()}, {lat_tr.max()}] frames".ljust(38)
      + f'[{lat_te.min()}, {lat_te.max()}]'.rjust(26) + f'{str(clean):>10}')

assert ok, 'held-out draws landed inside the training band'
print('\\nheld-out set is disjoint from training on every parameter')
'''),

    md("### The batched model must not replicate the whole model\n\n"
       "`jax.vmap` adds a leading axis to every leaf of whatever it returns, "
       "so vmapping a function that returns an MJX model replicates mesh "
       "vertices, convex hulls and body trees once per environment. At the "
       "batch sizes MJX is worth using at, that is an out-of-memory error "
       "whose message says nothing about domain randomization. Only four "
       "arrays should carry a batch axis."),
    code('''
import jax.tree_util as tu
mb, axes = env.batched_model(tr)
n_batched = sum(1 for l in tu.tree_leaves(axes) if l == 0)
shared = sum(np.prod(l.shape) * l.dtype.itemsize
             for l in tu.tree_leaves(env.base.model) if hasattr(l, 'shape'))
total = sum(np.prod(l.shape) * l.dtype.itemsize
            for l in tu.tree_leaves(mb) if hasattr(l, 'shape'))
print(f'leaves carrying a batch axis : {n_batched} (expected 4)')
print(f'model bytes, shared          : {shared/1e6:.2f} MB')
print(f'model bytes, batched x4096   : {total/1e6:.2f} MB  ({total/shared:.1f}x)')
assert n_batched == 4, 'more than the four randomized fields got batched'
print('\\nbatching OK')
'''),

    md("## 3. Train both arms\n\n"
       "Same file, same optimiser, same seed handling, same curriculum. The "
       "curriculum is a fixed function of step count rather than of success "
       "rate: under S2's adaptive rule the DR arm, solving a harder problem, "
       "advances more slowly, so at any given step the two arms would be "
       "training on *different task distributions* and the final comparison "
       "would silently mix \"DR generalises better\" with \"one arm spent "
       "longer on easy targets\".\n\n"
       "Set `ARM` and run. Each arm needs its own budget; expect to come back "
       "to this cell across several sessions."),
    code('''
ARM         = 'dr'          # 'dr' or 'baseline' -- run both
TOTAL_STEPS = 30_000_000
NUM_ENVS    = 2048          # use the knee S2's section 3 measured
MAX_HOURS   = 2.5

import os, subprocess, sys
os.chdir(SRC)
CKPT_LOCAL = WORK / 's3' / ARM
CKPT_DRIVE = DRIVE / 's3_checkpoints' / ARM

cmd = [sys.executable, '-u', '-m', 's3_domain_rand.train',
       '--arm', ARM,
       '--total-steps', str(TOTAL_STEPS),
       '--num-envs', str(NUM_ENVS),
       '--ckpt-dir', str(CKPT_LOCAL),
       '--ckpt-mirror', str(CKPT_DRIVE),
       '--max-hours', str(MAX_HOURS),
       '--resume']
print('$', ' '.join(cmd), flush=True)

p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=str(SRC)),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
try:
    for line in p.stdout:
        print(line, end='', flush=True)
except KeyboardInterrupt:
    p.terminate(); print('\\ninterrupted; newest checkpoint is in Drive')
p.wait()
'''),

    md("## 4. Evaluate, once both arms exist\n\n"
       "Three conditions, not two: nominal physics, in-distribution physics, "
       "held-out physics. Without the first two, a DR arm that is simply worse "
       "everywhere and a DR arm that trades in-distribution performance for "
       "robustness look identical, and that trade is the interesting result.\n\n"
       "Both arms see the identical held-out draws, from a seed unrelated to "
       "either training seed. The headline is the **paired** difference: the "
       "arms are scored on the same episodes, and pairing removes the "
       "episode-difficulty variance that dominates the unpaired numbers."),
    code('''
DR_DIR   = DRIVE / 's3_checkpoints' / 'dr'
BASE_DIR = DRIVE / 's3_checkpoints' / 'baseline'
for d in (DR_DIR, BASE_DIR):
    assert list(d.glob('ckpt_*.pkl')), f'no checkpoints in {d}; train that arm first'

r = subprocess.run([sys.executable, '-u', '-m', 's3_domain_rand.eval',
                    '--dr', str(DR_DIR), '--baseline', str(BASE_DIR),
                    '--episodes', '512',
                    '--out', str(DRIVE / 's3_eval.json')],
                   env=dict(os.environ, PYTHONPATH=str(SRC)))
print('exit', r.returncode)
'''),
    code('''
import json
res = json.loads((DRIVE / 's3_eval.json').read_text())
print(f"{'condition':<12}{'DR':>18}{'baseline':>18}{'paired DR - base':>26}")
for kind, row in res['conditions'].items():
    p = row['paired_dr_minus_baseline']
    print(f"{kind:<12}{row['dr']['success_rate']:>17.1%}"
          f"{row['baseline']['success_rate']:>18.1%}"
          f"{p['point']:>+16.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]")
print()
print('An interval that straddles zero is a null result. Report it as one.')
'''),
]


# ==========================================================================

def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    specs = [
        ("S10_gpu_scaling.ipynb",
         bootstrap("s10_gpu_scaling/run.py",
                   "S10, What a GPU buys a physics simulator", S10_BLURB) + S10_CELLS),
        ("S2_inhand.ipynb",
         bootstrap("s2_inhand/train.py",
                   "S2, In-hand cube reorientation (LEAP hand, MJX)", S2_BLURB) + S2_CELLS),
        ("S3_domain_rand.ipynb",
         bootstrap("s3_domain_rand/train.py",
                   "S3, Domain randomization, two arms, held-out physics", S3_BLURB) + S3_CELLS),
    ]
    for name, cells in specs:
        p = HERE / name
        p.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {p}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
