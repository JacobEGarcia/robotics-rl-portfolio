"""
A hydraulic contractile actuator, sized from Clone Robotics' published numbers.

Why this exists
---------------
S9 characterised MS-Human-700 and found that MuJoCo's Hill muscle cannot stand
in for a Myofiber-class actuator on the properties that matter. Two findings in
particular:

  * `tendon_stiffness` is zero for all 700 tendons, so the model has no series
    elasticity. Co-contraction stiffness then comes only from the force-length
    slope and the moment-arm derivative, and its *sign flips* over roughly 40%
    of every joint's range.
  * Activation is a first-order filter with fixed time constants. A valve
    driving a fluid volume is not that: its fill and vent rates depend on the
    pressure difference across the orifice, so the dynamics slow down as the
    actuator approaches the supply pressure.

Neither is a defect in MuJoCo. A Hill muscle is a good model of a muscle. It is
the wrong model of a braided hydraulic actuator, and the difference shows up
exactly where a controller lives.

The model
---------
A McKibben-class braided contractile actuator: a bladder inside an inextensible
braided sleeve. Pressurising the bladder inflates it radially, and because the
braid threads have fixed length the sleeve shortens axially.

Static force, the standard virtual-work result (Chou and Hannaford, 1996):

    F(P, eps) = pi r0^2 P [ a (1 - eps)^2 - b ]
    a = 3 / tan^2(theta0),   b = 1 / sin^2(theta0)

with `eps` the contraction ratio (L0 - L)/L0 and `theta0` the braid angle at
rest. Force reaches zero at the braid's blocking angle 54.7 degrees, which is
the classic result this implementation is checked against.

Braid kinematics, from the thread length and turn count being constant:

    cos(theta) = (1 - eps) cos(theta0)
    r(eps)     = r0 sin(theta) / sin(theta0)
    V(eps)     = pi r0^2 L0 (1 - eps) [1 - (1-eps)^2 cos^2(theta0)] / sin^2(theta0)

The volume matters as much as the force here: it is what a pump has to supply,
and Protoclone's pump is a published, finite number.

Sizing, from two published specifications
-----------------------------------------
Clone publishes that a Myofiber gives over 30% unloaded contraction, and that a
3 gram fibre produces at least 1 kg of force. Those two numbers pin the two
geometry parameters, which is why this is an inference rather than a fit:

    eps_max = 1 - 1 / (sqrt(3) cos(theta0))          set F = 0

    eps_max > 0.30   =>   cos(theta0) > 1/(0.30 ... )   =>   theta0 < 34.4 deg

`theta0 = 30 deg` is taken as the design point. It gives 33.3% contraction, and
a 2 mm x 100 mm fibre holding 1.26 mL of water (about 1.3 g, plus sleeve and
fittings, consistent with the published 3 g) produces 43 N at the published
100 psi system pressure, comfortably above the published 10 N minimum. The
check is an inequality in the direction the specification states, not an
equality fitted to it.

Published figures used, all from Clone's public material and press coverage:

    >30% unloaded contraction, <50 ms response
    3 g fibre, at least 1 kg force
    500 W pump, 40 L/min at 100 psi
    roughly 1000 muscles on Protoclone V1

None of this is Clone's actual design, which is not public. It is the closest
publicly documented actuator class, sized so its published behaviour matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PSI = 6894.757                      # Pa
WATER_DENSITY = 997.0               # kg/m^3
BLOCKING_ANGLE_DEG = 54.7356        # arctan(sqrt(2)), where F = 0 regardless of P

# Clone's published figures, used as external checks rather than as parameters.
PUBLISHED = {
    "min_contraction_fraction": 0.30,
    "min_force_N": 9.81,            # "at least 1 kg"
    "fibre_mass_g": 3.0,
    "response_ms": 50.0,
    "supply_psi": 100.0,
    "pump_lpm": 40.0,
    "pump_watts": 500.0,
    "protoclone_muscles": 1000,
}


@dataclass
class Myofiber:
    """One braided hydraulic contractile actuator."""

    r0: float = 0.002               # rest radius, m
    L0: float = 0.100               # rest length, m
    theta0_deg: float = 30.0        # braid angle at rest
    # Series compliance. A real braid-and-bladder assembly stretches under
    # load; without this the actuator is a rigid rope and the joint stiffness
    # it produces is wrong in the same way MS-Human-700's is.
    k_series: float = 8.0e4         # N/m
    slack: float = 0.0              # series element slack length, m
    # Hydraulic capacitance: how much extra volume the bladder accepts per
    # pascal. Water is near-incompressible, so essentially all of the pressure
    # dynamics come from this compliance, and it sets the response time.
    # Derived, not guessed: a braided bladder taking up about 1% of its rest
    # volume in extra fluid at the 100 psi supply. 1.257 mL * 0.01 / 689 kPa.
    hydraulic_capacitance: float = 1.82e-14   # m^3/Pa
    # Valve. Orifice flow, which is where the fill/vent asymmetry comes from:
    # filling is driven by (supply - P) and venting by (P - ambient), and those
    # are not the same number at any operating point.
    # Derived from the two published figures rather than chosen: one full
    # stroke is 0.977 mL and the published response is under 50 ms, so the
    # valve has to pass 1.17 L/min. At a mean driving head of half the supply
    # that needs 1.2e-6 m^2, a 1.24 mm orifice. The response time this produces
    # is therefore a prediction to check, not a parameter fitted to the spec.
    valve_area: float = 1.20e-6     # m^2, fully open
    discharge_coeff: float = 0.62
    supply_pressure: float = 100.0 * PSI
    ambient_pressure: float = 0.0
    # Back-pressure on the return line. Zero by default, which makes fill and
    # vent *exactly* symmetric: both orifices see the same head at their
    # respective extremes. That symmetry is worth knowing, because it is the
    # opposite of a Hill muscle, where a 4x deactivation penalty is baked into
    # the model and cannot be plumbed away.
    return_pressure: float = 0.0

    @property
    def theta0(self) -> float:
        return np.radians(self.theta0_deg)

    @property
    def a(self) -> float:
        return 3.0 / np.tan(self.theta0) ** 2

    @property
    def b(self) -> float:
        return 1.0 / np.sin(self.theta0) ** 2

    @property
    def eps_max(self) -> float:
        """
        Contraction at which force reaches zero, in closed form.

        Setting a(1-eps)^2 = b and simplifying gives 1 - 1/(sqrt(3) cos(theta0)),
        which is where the braid angle hits 54.7 degrees.
        """
        return 1.0 - 1.0 / (np.sqrt(3.0) * np.cos(self.theta0))

    # ------------------------------------------------------------ statics

    def force(self, pressure, eps):
        """
        Contractile force at a given gauge pressure and contraction ratio.

        Clipped at zero: a braid pulls and cannot push. Past `eps_max` the
        formula turns negative, which physically means the sleeve has reached
        its blocking angle and gone slack, not that it pushes.
        """
        eps = np.asarray(eps, dtype=float)
        f = np.pi * self.r0 ** 2 * np.asarray(pressure, dtype=float) * (
            self.a * (1.0 - eps) ** 2 - self.b
        )
        return np.maximum(f, 0.0)

    def braid_angle(self, eps):
        """Instantaneous braid angle, radians."""
        return np.arccos(np.clip((1.0 - np.asarray(eps, dtype=float)) * np.cos(self.theta0),
                                 -1.0, 1.0))

    def radius(self, eps):
        return self.r0 * np.sin(self.braid_angle(eps)) / np.sin(self.theta0)

    def volume(self, eps):
        """
        Enclosed fluid volume at a given contraction.

        This is the quantity a pump has to deliver, so it is what turns a
        published pump rating into a constraint on how many actuators can move
        at once.
        """
        eps = np.asarray(eps, dtype=float)
        return (np.pi * self.r0 ** 2 * self.L0 * (1.0 - eps)
                * (1.0 - (1.0 - eps) ** 2 * np.cos(self.theta0) ** 2)
                / np.sin(self.theta0) ** 2)

    def rest_volume(self) -> float:
        return float(self.volume(0.0))

    def water_mass_g(self) -> float:
        return 1e3 * WATER_DENSITY * self.rest_volume()

    # ------------------------------------------------------------ dynamics

    def valve_flow(self, pressure: float, command: float) -> float:
        """
        Volumetric flow into the actuator, m^3/s.

        `command` in [-1, 1]: positive opens the fill port to supply, negative
        opens the vent to ambient. Orifice flow, Q = Cd A sqrt(2 dP / rho).

        The asymmetry is not a modelling choice, it falls out: near the supply
        pressure the fill driving head goes to zero while the vent head is at
        its largest, so an actuator fills slowly at the top of its range and
        empties fastest exactly there.
        """
        opening = abs(float(np.clip(command, -1.0, 1.0)))
        if opening <= 0.0:
            return 0.0
        area = self.valve_area * opening
        if command > 0.0:
            head = self.supply_pressure - pressure
            sign = 1.0
        else:
            head = pressure - max(self.ambient_pressure, self.return_pressure)
            sign = -1.0
        if head <= 0.0:
            return 0.0
        return sign * self.discharge_coeff * area * np.sqrt(2.0 * head / WATER_DENSITY)

    def pressure_rate(self, pressure: float, eps: float, eps_rate: float,
                      command: float) -> float:
        """
        dP/dt from the volume balance.

        Total fluid volume is the braid's geometric volume plus what the
        bladder's compliance absorbs, so

            Q = dV_braid/dt + C dP/dt

        and the actuator shortening under load pushes fluid back out, which is
        why `eps_rate` appears. That coupling is the reason a loaded actuator
        and an unloaded one do not have the same response time.
        """
        dV_deps = self._dvolume_deps(eps)
        flow = self.valve_flow(pressure, command)
        return (flow - dV_deps * eps_rate) / self.hydraulic_capacitance

    def _dvolume_deps(self, eps: float, h: float = 1e-7) -> float:
        return float((self.volume(eps + h) - self.volume(eps - h)) / (2.0 * h))

    # -------------------------------------------------- muscle-tendon unit

    def solve_length(self, pressure: float, mtu_length: float,
                     *, tol: float = 1e-12, iters: int = 80) -> dict:
        """
        Split a total length between the contractile element and its series spring.

        The actuator is a contractile element in series with a spring, so a
        commanded pressure does not set a length by itself. Equilibrium is where
        the braid's pull equals the spring's stretch:

            F_braid(P, eps(L_c)) = k_series * (L_mtu - L_c - slack)

        Solved by bisection on L_c. Bisection rather than Newton because the
        braid force has a kink where it clips at zero, and a derivative-based
        solver walks off it.
        """
        # The bracket has to allow *extension*, not just contraction. A braided
        # sleeve pulled longer than its rest length has a smaller braid angle
        # and therefore pulls harder, which is exactly the regime an antagonist
        # pair puts one of its members into. Capping the bracket at L0 pins the
        # stretched actuator at the bound instead of finding equilibrium, and
        # the resulting joint stiffness curve is flat and wrong over most of
        # the range. `max_extension` is where the braid angle would reach zero.
        max_extension = 1.0 / np.cos(self.theta0) - 1.0
        lo = self.L0 * (1.0 - self.eps_max)
        hi = self.L0 * (1.0 + 0.999 * max_extension)

        def residual(length_c: float) -> float:
            eps = (self.L0 - length_c) / self.L0
            stretch = mtu_length - length_c - self.slack
            return float(self.force(pressure, eps)) - self.k_series * stretch

        r_lo, r_hi = residual(lo), residual(hi)
        if r_lo * r_hi > 0.0:
            # No sign change: the unit is either slack (no tension anywhere) or
            # pinned at a limit. Report the endpoint rather than a bracket that
            # does not exist.
            length_c = lo if abs(r_lo) < abs(r_hi) else hi
        else:
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if residual(lo) * residual(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < tol:
                    break
            length_c = 0.5 * (lo + hi)

        eps = (self.L0 - length_c) / self.L0
        tension = max(0.0, self.k_series * (mtu_length - length_c - self.slack))
        res = residual(length_c)
        return {
            "length_contractile": length_c,
            "eps": eps,
            "tension_N": tension,
            "braid_force_N": float(self.force(pressure, eps)),
            "residual_N": res,
            # A pinned bracket returns a number that looks like a solution and
            # is not one. At low pressure the braid can be too weak to hold the
            # stretch an antagonist puts on it, and the solve lands on the
            # bound instead of an equilibrium. Callers must check this rather
            # than let a 24 N tension that should be 13 N into a statistic.
            "converged": bool(abs(res) < 1e-6 * max(1.0, abs(tension))),
        }

    # ------------------------------------------------------------- checks

    def specification_check(self) -> dict:
        """
        Compare the sized actuator against Clone's published figures.

        Every comparison is an inequality in the direction the specification
        states. A specification that says "at least 1 kg" is not satisfied by
        fitting to exactly 1 kg, and treating it as an equality would be
        fitting the model to the number rather than testing it.
        """
        supply = PUBLISHED["supply_psi"] * PSI
        blocked = np.degrees(self.braid_angle(self.eps_max))
        return {
            "theta0_deg": self.theta0_deg,
            "eps_max": round(float(self.eps_max), 4),
            "meets_contraction_spec": bool(self.eps_max > PUBLISHED["min_contraction_fraction"]),
            "max_theta0_for_spec_deg": round(float(np.degrees(np.arccos(
                1.0 / (np.sqrt(3.0) * (1.0 - PUBLISHED["min_contraction_fraction"]))))), 2),
            "blocking_angle_deg": round(float(blocked), 3),
            "blocking_angle_matches_54.7": bool(abs(blocked - BLOCKING_ANGLE_DEG) < 1e-3),
            "force_at_supply_N": round(float(self.force(supply, 0.0)), 2),
            "meets_force_spec": bool(self.force(supply, 0.0) > PUBLISHED["min_force_N"]),
            "water_mass_g": round(self.water_mass_g(), 3),
            "mass_under_published_fibre_g": bool(self.water_mass_g() < PUBLISHED["fibre_mass_g"]),
            "rest_volume_mL": round(1e6 * self.rest_volume(), 4),
            "stroke_volume_mL": round(
                1e6 * abs(self.volume(self.eps_max) - self.volume(0.0)), 4),
            "force_per_gram_N": round(float(self.force(supply, 0.0)) / self.water_mass_g(), 1),
        }
