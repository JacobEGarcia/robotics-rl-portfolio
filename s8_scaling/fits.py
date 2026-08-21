"""
Power-law fitting for the scaling curves.

Two forms are fitted and both are reported:

    pure:    L(D) = (Dc / D)^alpha              (GEN-0's stated form)
    offset:  L(D) = (Dc / D)^alpha + L_inf      (irreducible-loss variant)

The pure form is a straight line in log-log space and is fitted by least
squares there - robust, no initialisation. The offset form acknowledges
that behaviour-cloning loss cannot reach zero (expert action noise sets a
floor) and is fitted with `scipy.optimize.curve_fit` seeded from the pure
fit. When L_inf's confidence interval includes zero, the pure form is the
honest summary; when it excludes zero, the floor is real and reported.

R^2 is computed in log space for the pure form (that is the space the
straight-line claim lives in) and in linear space for the offset form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    d_c: float
    l_inf: float
    r2: float
    form: str  # "pure" | "offset"

    def predict(self, d: np.ndarray) -> np.ndarray:
        return (self.d_c / np.asarray(d, dtype=float)) ** self.alpha + self.l_inf

    def label(self) -> str:
        if self.form == "pure":
            return f"L = (Dc/D)^{self.alpha:.2f},  R^2 = {self.r2:.3f}"
        return (
            f"L = (Dc/D)^{self.alpha:.2f} + {self.l_inf:.3f},  R^2 = {self.r2:.3f}"
        )


def fit_pure(d: np.ndarray, loss: np.ndarray) -> PowerLawFit:
    """Straight-line fit in log-log space."""
    d = np.asarray(d, dtype=float)
    loss = np.asarray(loss, dtype=float)
    logd, logl = np.log(d), np.log(loss)
    slope, intercept = np.polyfit(logd, logl, 1)
    alpha = -slope
    d_c = np.exp(intercept / alpha) if alpha != 0 else float("nan")
    pred = slope * logd + intercept
    ss_res = np.sum((logl - pred) ** 2)
    ss_tot = np.sum((logl - logl.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return PowerLawFit(alpha=float(alpha), d_c=float(d_c), l_inf=0.0, r2=float(r2), form="pure")


def fit_offset(d: np.ndarray, loss: np.ndarray) -> PowerLawFit | None:
    """Offset power law, seeded from the pure fit. None if it fails."""
    d = np.asarray(d, dtype=float)
    loss = np.asarray(loss, dtype=float)
    seed = fit_pure(d, loss)

    def f(x, alpha, d_c, l_inf):
        return (d_c / x) ** alpha + l_inf

    try:
        popt, _ = curve_fit(
            f,
            d,
            loss,
            p0=[seed.alpha, seed.d_c, 0.5 * loss.min()],
            bounds=([1e-3, 1e-6, 0.0], [10.0, 1e6, loss.min()]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return None
    pred = f(d, *popt)
    ss_res = np.sum((loss - pred) ** 2)
    ss_tot = np.sum((loss - loss.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return PowerLawFit(
        alpha=float(popt[0]), d_c=float(popt[1]), l_inf=float(popt[2]), r2=float(r2), form="offset"
    )
