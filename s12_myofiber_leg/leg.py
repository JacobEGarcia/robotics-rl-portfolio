"""
Sizing a braided hydraulic actuator for every muscle in MS-Human-700's leg.

The question
------------
S11 built a Myofiber-class actuator and checked it against Clone's published
figures. The obvious next step is to put it in a body: take the 50 muscles that
span MS-Human-700's right leg and replace each one with a braid.

It does not work one-for-one, and the reason is geometric rather than
engineering. A braided sleeve's maximum contraction is

    eps_max(theta0) = 1 - 1 / (sqrt(3) cos(theta0))

which increases as the rest braid angle falls, and has a supremum as
theta0 -> 0 of

    1 - 1/sqrt(3) = 0.42265...

**No McKibben-class braid, at any braid angle, contracts more than 42.3%.**
Seventeen of the leg's fifty muscles need more excursion than that. They are
not a random seventeen: they are the glutei, the adductors, sartorius, gracilis
and psoas, which is to say the large hip muscles, the ones with long fibres and
long throws.

The fix, and what it costs
--------------------------
A stroke-amplifying transmission. A lever or pulley of ratio `n` lets the
actuator travel `1/n` of the tendon's excursion, at the cost of needing `n`
times the force.

Choosing `n = required_excursion / eps_max` makes the actuator exactly as long
as the muscle it replaces, which is the natural design rule when the actuator
has to fit in the same limb. Then:

    L0     = the muscle's longest length
    force  = n x the muscle's peak isometric force
    r0     = sqrt(n F / (pi P (a - b)))

and because radius scales as sqrt(n), **fluid volume scales linearly with the
transmission ratio**. The transmission is not free; it is paid for in water,
and therefore in pump flow.

What this is not
----------------
Not a claim about Clone's design. Clone does not publish its routing, and a
real machine would not replace human muscles one-for-one anyway; it would
choose its own actuator layout. This measures what the *human* arrangement
demands of a braided actuator, which is the relevant question if you are using
a musculoskeletal model as a design reference, and it is a question with a
clean answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s11_myofiber.myofiber import PSI, WATER_DENSITY, Myofiber

# Supremum of eps_max over all braid angles, reached in the limit theta0 -> 0.
BRAID_CEILING = 1.0 - 1.0 / np.sqrt(3.0)


def eps_max(theta0_deg: float) -> float:
    t = np.radians(theta0_deg)
    return float(1.0 - 1.0 / (np.sqrt(3.0) * np.cos(t)))


@dataclass
class SizedActuator:
    """One braid sized to stand in for one muscle."""

    name: str
    peak_force_N: float
    length_min: float
    length_max: float
    required_excursion: float
    feasible_without_transmission: bool
    feasible_at_any_braid_angle: bool
    transmission_ratio: float
    L0: float
    r0: float
    actuator_force_N: float
    rest_volume_mL: float
    stroke_volume_mL: float
    water_mass_g: float

    def fibre(self, theta0_deg: float, supply_psi: float) -> Myofiber:
        return Myofiber(r0=self.r0, L0=self.L0, theta0_deg=theta0_deg,
                        supply_pressure=supply_psi * PSI)


def size_actuator(name, peak_force, length_min, length_max, *,
                  theta0_deg=30.0, supply_psi=100.0) -> SizedActuator:
    """
    Size one braid to replace one muscle.

    The transmission ratio is chosen so the actuator's rest length equals the
    muscle's longest length: the actuator has to fit where the muscle was.
    Muscles that already fit inside the braid's stroke get ratio 1 and are left
    alone rather than given a transmission they do not need.
    """
    e_max = eps_max(theta0_deg)
    required = (length_max - length_min) / length_max
    ratio = max(1.0, required / e_max)

    L0 = length_max
    force = ratio * peak_force
    t = np.radians(theta0_deg)
    a, b = 3.0 / np.tan(t) ** 2, 1.0 / np.sin(t) ** 2
    supply = supply_psi * PSI
    r0 = float(np.sqrt(force / (np.pi * supply * (a - b))))

    probe = Myofiber(r0=r0, L0=L0, theta0_deg=theta0_deg, supply_pressure=supply)
    rest_v = probe.rest_volume()
    stroke_v = abs(float(probe.volume(probe.eps_max) - probe.volume(0.0)))

    return SizedActuator(
        name=name,
        peak_force_N=float(peak_force),
        length_min=float(length_min),
        length_max=float(length_max),
        required_excursion=float(required),
        feasible_without_transmission=bool(required <= e_max),
        feasible_at_any_braid_angle=bool(required <= BRAID_CEILING),
        transmission_ratio=float(ratio),
        L0=float(L0),
        r0=r0,
        actuator_force_N=float(force),
        rest_volume_mL=1e6 * rest_v,
        stroke_volume_mL=1e6 * stroke_v,
        water_mass_g=1e3 * WATER_DENSITY * rest_v,
    )


def size_leg(model, muscles, names, *, theta0_deg=30.0, supply_psi=100.0):
    """Size a braid for every muscle in a set, from the compiled MuJoCo model."""
    lengthrange = model.actuator_lengthrange[muscles]
    peak = model.actuator_gainprm[muscles, 2]
    return [
        size_actuator(names[int(m)], peak[i], lengthrange[i, 0], lengthrange[i, 1],
                      theta0_deg=theta0_deg, supply_psi=supply_psi)
        for i, m in enumerate(muscles)
    ]


def verify_sizing(sized, *, theta0_deg=30.0, supply_psi=100.0, tol=1e-6) -> dict:
    """
    Round-trip check: does each sized actuator actually make the force asked of it?

    Sizing inverts a closed form, so an error here means the inversion and the
    forward model disagree, which no downstream number would reveal.
    """
    worst = 0.0
    for s in sized:
        f = s.fibre(theta0_deg, supply_psi)
        delivered = float(f.force(supply_psi * PSI, 0.0))
        worst = max(worst, abs(delivered - s.actuator_force_N) / max(s.actuator_force_N, 1.0))
    return {"max_relative_force_error": worst,
            "sizing_round_trips": bool(worst < tol)}


def leg_totals(sized) -> dict:
    """Fluid, mass and force totals for a whole limb's worth of actuators."""
    rest = sum(s.rest_volume_mL for s in sized)
    stroke = sum(s.stroke_volume_mL for s in sized)
    return {
        "n_actuators": len(sized),
        "total_rest_volume_mL": round(rest, 2),
        "total_stroke_volume_mL": round(stroke, 2),
        "total_water_mass_g": round(sum(s.water_mass_g for s in sized), 1),
        "largest_r0_mm": round(1e3 * max(s.r0 for s in sized), 2),
        "smallest_r0_mm": round(1e3 * min(s.r0 for s in sized), 3),
        "max_transmission_ratio": round(max(s.transmission_ratio for s in sized), 3),
        "median_transmission_ratio": round(
            float(np.median([s.transmission_ratio for s in sized])), 3),
        "n_needing_transmission": int(sum(1 for s in sized if s.transmission_ratio > 1.0)),
        "n_impossible_at_any_angle": int(
            sum(1 for s in sized if not s.feasible_at_any_braid_angle)),
    }
