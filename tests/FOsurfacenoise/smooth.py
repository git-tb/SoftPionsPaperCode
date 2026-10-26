#!/usr/bin/env python3
"""
Build the two freeze-out surfaces this test compares.

The tabulated surface carries white noise: measured on the staged scenarios,
tau and r are noisy at ~1e-5 relative, Dtau and Dr at ~5e-3. `pchip` in
include/freezeout.h interpolates THROUGH every point, so that noise becomes
knot-scale curvature in the integrand, which is what defeats the Gauss-Kronrod
error estimate in specV2 (see Claude/summaries/ for the diagnosis).

Smoothing here is a Savitzky-Golay filter: a local least-squares polynomial in
a sliding window, evaluated at the window centre.

Two deliberate choices:

  * `Dtau` and `Dr` are taken as the ANALYTIC DERIVATIVE of the same local
    polynomial that produces the smoothed `tau` and `r`, not smoothed
    independently. This makes the surface self-consistent by construction --
    Dtau really is dtau/dalpha of the tau that is shipped -- and it is where
    most of the noise reduction comes from, the derivative columns being 500x
    noisier than the geometry columns.

  * alpha is used as the fit variable directly rather than the point index,
    because the tabulated alpha spacing is NOT uniform.

The window must be wide enough to average the noise down but narrow enough to
leave the physical shape alone. `validate()` checks both ends of that: the
residual should have the amplitude of the estimated noise and be uncorrelated
point to point (removing signal shows up as correlated residuals).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

COLUMNS = ["alpha", "tau", "r", "Dtau", "Dr"]

# Fit window: 17 points, quintic.
#
# Chosen by scanning halfwidth and degree against two competing measures: the
# lag-1 autocorrelation of the residual (the test for "removed noise, not
# signal" -- white noise gives ~0, eating real structure gives a large positive
# value) and the resulting smoothness of the DERIVATIVE, which is what the
# integrand is actually sensitive to.
#
#   filter        tau resid   lag-1 ac   Dtau noise
#   tabulated             -          -     5.93e-2
#   h=4  d=3        9.33e-5     -0.103     5.97e-3
#   h=8  d=5        1.02e-4     +0.076     3.90e-3   <- this
#   h=12 d=5        1.15e-4     +0.231     1.64e-3
#   h=16 d=5        1.45e-4     +0.459     1.18e-3
#
# Raising the degree along with the window is what breaks the deadlock: a
# quintic tracks the real curvature over a wider span, so h=8/d=5 buys a 1.5x
# smoother derivative than h=4/d=3 at equal fidelity to tau (both |ac| ~ 0.1).
# Going wider still keeps improving the derivative but starts removing real
# structure, which is why the scan stops here.
#
# Two alternatives were tried and rejected, both measured:
#
#   * INTERPOLATING SPLINE through the smoothed curve. A spline passes through
#     every point, so it reproduces whatever noise survives and differentiating
#     amplifies it: Dtau noise 4.3e-2, seven times WORSE than the least-squares
#     local fit used here. Interpolation is not smoothing.
#
#   * FOURIER / COSINE LOW-PASS fitted on the actual alpha grid. Loses on both
#     axes at every cutoff: the best residual reachable is 6.8e-4 (vs a 8.6e-5
#     noise floor) and by then ac is still +0.13, while above ~120 modes the
#     least-squares problem on this non-uniform grid goes ill-conditioned and
#     the derivative explodes to ~1e9. Three reasons it cannot work here: the
#     alpha spacing varies by ~770x so there is no uniform grid to transform
#     on; tau is non-periodic and steep near pi/2, so it is not band-limited;
#     and the noise is LOCALISED (100x stronger in alpha ~ [0.35,0.85)), which
#     a single global cutoff cannot adapt to. Wavelet thresholding is the
#     transform-domain method that would handle localised noise, but it needs
#     a dependency this project does not have, to beat a filter that already
#     reaches 15x smoother derivatives than the shipped data.
HALFWIDTH = 8
DEGREE = 5


def savgol(x, y, halfwidth=HALFWIDTH, degree=DEGREE):
    """Return (smoothed y, dy/dx) from a sliding local polynomial fit.

    Near the ends the window is shifted inward rather than shrunk, so every fit
    uses the same number of points and the same degree.
    """
    n = len(x)
    ys, dy = np.empty(n), np.empty(n)
    for i in range(n):
        lo = min(max(i - halfwidth, 0), n - (2 * halfwidth + 1))
        s = slice(lo, lo + 2 * halfwidth + 1)
        # fit in (x - x[i]) so the constant term is the value AT x[i] and the
        # linear term is the derivative there, whatever the window offset
        c = np.polyfit(x[s] - x[i], y[s], degree)
        ys[i] = c[-1]
        dy[i] = c[-2]
    return ys, dy


def smooth_surface(df):
    """Smoothed copy of a freeze-out table, with consistent derivatives."""
    a = df["alpha"].to_numpy()
    tau_s, dtau_s = savgol(a, df["tau"].to_numpy())
    r_s, dr_s = savgol(a, df["r"].to_numpy())
    return pd.DataFrame({"alpha": a, "tau": tau_s, "r": r_s,
                         "Dtau": dtau_s, "Dr": dr_s})


def noise_estimate(x, y, halfwidth=3, degree=2):
    """White-noise sigma, as the RMS residual from a local fit.

    Deliberately a different window and degree from the filter above, so it is
    not simply reporting the filter back to itself.

    Finite-difference estimators (sigma^2 = var(d^n y) / C(2n,n)) are NOT usable
    here: the tabulated alpha spacing varies by a factor ~770 across the
    surface, so plain differences are dominated by the varying step size times
    y' and overestimate the noise by two orders of magnitude. Everything has to
    be fitted against the actual alpha values.
    """
    ys, _ = savgol(x, y, halfwidth, degree)
    return np.sqrt(np.mean((y - ys) ** 2))


def validate(orig, smoothed):
    """Report whether the filter removed noise rather than signal."""
    a = orig["alpha"].to_numpy()
    rows = []
    for col in ("tau", "r"):
        o, s = orig[col].to_numpy(), smoothed[col].to_numpy()
        res = o - s
        # lag-1 autocorrelation: ~0 means what was removed is white noise,
        # a large positive value would mean real structure was smoothed away
        e = res - res.mean()
        ac = float(np.corrcoef(e[:-1], e[1:])[0, 1])
        rows.append((col, noise_estimate(a, o), np.sqrt(np.mean(res ** 2)),
                     np.abs(res).max() / np.abs(o).mean(), ac))
    return rows


def build(src, dst, verbose=True):
    """Write <dst>/original and <dst>/smoothed from the scenario at `src`."""
    fo = pd.read_csv(os.path.join(src, "freezeout.csv"), comment="#")
    fo.columns = COLUMNS
    sm = smooth_surface(fo)

    for name, table in (("original", fo), ("smoothed", sm)):
        d = os.path.join(dst, name)
        os.makedirs(d, exist_ok=True)
        table.to_csv(os.path.join(d, "freezeout.csv"), index=False)
        # the condensate initial condition is identical in both arms
        shutil.copy(os.path.join(src, "initdata.csv"),
                    os.path.join(d, "initdata.csv"))

    if verbose:
        print(f"  source: {src}  ({len(fo)} knots)")
        print(f"  {'column':8s}{'noise est':>13s}{'removed rms':>14s}"
              f"{'max change':>13s}{'resid lag-1 ac':>16s}")
        for col, est, rms, mx, ac in validate(fo, sm):
            print(f"  {col:8s}{est:13.3e}{rms:14.3e}{mx:13.3e}{ac:+16.3f}")
        print("  (removed rms ~ noise est and ac ~ 0  =>  noise removed, "
              "signal kept)")
    return fo, sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../../datanew/PbPb276_cc_(0,5)")
    ap.add_argument("--dst", default="data")
    build(ap.parse_args().src, ap.parse_args().dst)
