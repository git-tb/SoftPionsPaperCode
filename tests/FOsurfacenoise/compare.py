#!/usr/bin/env python3
"""
Figures and report for the freeze-out-surface noise test.

Reads whatever run.py left in out/ and produces:

  fig_surface.pdf   tau, r, Dtau, Dr against alpha for both surfaces, with the
                    difference underneath -- this is where the noise is visible
  fig_rtau.pdf      the surface itself in the (r, tau) plane, with a zoom
  fig_spectra.pdf   primary and decay spectra for both arms, and their ratios
  fig_sweep.pdf     GSL roundoff rate against requested tolerance

and prints the numbers the figures are meant to support.

Usage:  make compare
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

matplotlib.rcParams["mathtext.default"] = "rm"

# Fully vector PDFs. The curves were already vector paths, but matplotlib's
# default embeds text as Type 3 fonts, which several viewers render badly and
# most journals reject outright. Type 42 embeds real TrueType, so the text
# stays selectable, searchable and scalable.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 6
# never let a dense line collection silently become a bitmap
matplotlib.rcParams["agg.path.chunksize"] = 0

ARMS = ("original", "smoothed")
COL = {"original": "#c1121f", "smoothed": "#003049"}
LW = 1.6
FS_LABEL = 12
FS_TICK = 10


def style(ax, xlabel, ylabel, legend=False):
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        ax.legend(fontsize=FS_LABEL - 2)


def load_surface(arm):
    return pd.read_csv(os.path.join("data", arm, "freezeout.csv"))


def load_spec(arm, kind):
    """kind: 'spec' -> primary spectrum, 'decay' -> decay spectrum."""
    name = "spectr.txt" if kind == "spec" else "decayspec.txt"
    hits = glob.glob(os.path.join("out", arm, kind, name))
    if not hits:
        sys.exit(f"missing {kind} output for {arm} -- run `make run` first")
    return pd.read_csv(hits[0], comment="#").to_numpy().T


# ---------------------------------------------------------------- figure 1
def fig_surface(surf):
    cols = [("tau", r"$\tau\ [fm]$"), ("r", r"$r\ [fm]$"),
            ("Dtau", r"$d\tau/d\alpha\ [fm]$"), ("Dr", r"$dr/d\alpha\ [fm]$")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for j, (c, lab) in enumerate(cols):
        ax = axes[0, j]
        for arm in ARMS:
            ax.plot(surf[arm]["alpha"], surf[arm][c], lw=LW, c=COL[arm],
                    label=arm, alpha=0.85)
        style(ax, r"$\alpha$", lab, legend=(j == 0))

        ax = axes[1, j]
        d = surf["original"][c] - surf["smoothed"][c]
        ax.plot(surf["original"]["alpha"], d, lw=0.8, c="k")
        ax.axhline(0, lw=0.8, c=COL["smoothed"], ls="--")
        style(ax, r"$\alpha$", "original - smoothed")
        ax.set_title(f"rms {np.sqrt((d**2).mean()):.2e}", fontsize=FS_LABEL - 2)
    fig.suptitle("Freeze-out surface: tabulated vs denoised  "
                 "(bottom row = what the filter removed)", fontsize=FS_LABEL + 2)
    fig.tight_layout()
    fig.savefig("fig_surface.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_rtau(surf):
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 6))
    for arm in ARMS:
        ax.plot(surf[arm]["r"], surf[arm]["tau"], lw=LW, c=COL[arm],
                label=arm, alpha=0.85)
    style(ax, r"$r\ [fm]$", r"$\tau\ [fm]$", legend=True)
    ax.set_title("freeze-out surface", fontsize=FS_LABEL)

    # zoom on the densely sampled, steeply falling end
    a = surf["original"]["alpha"].to_numpy()
    sel = a > a.max() - 0.05
    for arm in ARMS:
        axz.plot(surf[arm]["r"][sel], surf[arm]["tau"][sel], lw=LW,
                 marker="o", ms=2.5, c=COL[arm], label=arm, alpha=0.85)
    style(axz, r"$r\ [fm]$", r"$\tau\ [fm]$", legend=True)
    axz.set_title(r"zoom: last $0.05$ rad of $\alpha$ (markers = knots)",
                  fontsize=FS_LABEL)
    fig.tight_layout()
    fig.savefig("fig_rtau.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_spectra(prim, dec):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    for arm in ARMS:
        ax.plot(prim[arm][0], np.abs(prim[arm][1]), lw=LW, c=COL[arm],
                label=arm, alpha=0.85)
    ax.set_yscale("log")
    style(ax, r"$q_T\ [GeV]$", "primary spectrum", legend=True)

    ax = axes[0, 1]
    q, a, b = prim["original"][0], prim["original"][1], prim["smoothed"][1]
    g = a != 0
    rel = b[g] / a[g] - 1
    ax.plot(q[g], rel, lw=0.9, c="k")
    ax.axhline(0, lw=0.8, c=COL["smoothed"], ls="--")
    # The largest excursions sit at the deep minima of the oscillating
    # spectrum, where the value nearly vanishes and the RELATIVE change
    # explodes while the absolute difference stays negligible. Clip to a
    # robust range so the bulk behaviour is readable, and say so.
    lim = 1.3 * np.percentile(np.abs(rel), 98)
    ax.set_ylim(-lim, lim)
    style(ax, r"$q_T\ [GeV]$", "smoothed / original $-$ 1")
    ax.set_title("primary spectrum, relative change", fontsize=FS_LABEL)
    ax.text(0.02, 0.03,
            f"clipped to $\\pm${lim:.2f}; peaks reach {np.abs(rel).max():.0f} "
            f"at the spectrum's zeros",
            transform=ax.transAxes, fontsize=FS_LABEL - 3, color="0.35")

    ax = axes[1, 0]
    for arm in ARMS:
        ax.plot(dec[arm][0], dec[arm][1], lw=LW, c=COL[arm], label=arm,
                alpha=0.85)
    ax.set_yscale("log")
    style(ax, r"$p_T\ [GeV]$", "decay spectrum", legend=True)

    ax = axes[1, 1]
    p, a, b = dec["original"][0], dec["original"][1], dec["smoothed"][1]
    g = a != 0
    ax.plot(p[g], b[g] / a[g] - 1, lw=0.9, c="k")
    ax.axhline(0, lw=0.8, c=COL["smoothed"], ls="--")
    style(ax, r"$p_T\ [GeV]$", "smoothed / original $-$ 1")
    ax.set_title("decay spectrum, relative change", fontsize=FS_LABEL)

    fig.tight_layout()
    fig.savefig("fig_spectra.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_sweep():
    path = os.path.join("out", "sweep.csv")
    if not os.path.isfile(path):
        return
    s = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arm in ARMS:
        ax.plot(s["epsrel"], 100 * s[arm] / s["total"], marker="o", lw=LW,
                c=COL[arm], label=arm)
    ax.set_xscale("log")
    ax.invert_xaxis()
    style(ax, "requested epsrel", "GSL roundoff reports [%]", legend=True)
    ax.set_title("how far the source integral can be pushed", fontsize=FS_LABEL)
    fig.tight_layout()
    fig.savefig("fig_sweep.pdf")
    plt.close(fig)


def yield_of(pT, spec):
    return np.sum(np.diff(pT).mean() * 2 * np.pi * pT * spec)


def main():
    surf = {a: load_surface(a) for a in ARMS}
    prim = {a: load_spec(a, "spec") for a in ARMS}
    dec = {a: load_spec(a, "decay") for a in ARMS}

    fig_surface(surf)
    fig_rtau(surf)
    fig_spectra(prim, dec)
    fig_sweep()

    print("=" * 70)
    print("SURFACE")
    print("=" * 70)
    for c in ("tau", "r", "Dtau", "Dr"):
        o, s = surf["original"][c].to_numpy(), surf["smoothed"][c].to_numpy()
        d = o - s
        print(f"  {c:5s}  rms change {np.sqrt((d**2).mean()):.3e}"
              f"   relative {np.sqrt((d**2).mean())/np.abs(o).mean():.3e}")

    # The removed noise is not spread evenly over the surface; it sits in
    # bursts. Worth reporting, because it says the input data quality varies
    # along alpha rather than being uniformly imprecise.
    a = surf["original"]["alpha"].to_numpy()
    res = (surf["original"]["tau"] - surf["smoothed"]["tau"]).to_numpy()
    print("\n  where the noise lives (tau residual by alpha band):")
    print(f"    {'band':>16s}{'knots':>8s}{'med spacing':>14s}{'resid rms':>12s}")
    for lo, hi in ((0, 0.35), (0.35, 0.85), (0.85, 1.30), (1.30, a.max() + 1e-9)):
        m = (a >= lo) & (a < hi)
        if not m.any():
            continue
        print(f"    [{lo:.2f},{hi:.2f}){m.sum():8d}"
              f"{np.median(np.gradient(a)[m]):14.3e}"
              f"{np.sqrt((res[m] ** 2).mean()):12.3e}")

    print()
    print("=" * 70)
    print("SPECTRA  (smoothed vs original)")
    print("=" * 70)
    for lab, d in (("primary", prim), ("decay", dec)):
        x, a, b = d["original"][0], d["original"][1], d["smoothed"][1]
        g = a != 0
        rel = np.abs(b[g] / a[g] - 1)
        print(f"  {lab:8s} median {np.median(rel):9.3e}   "
              f"90th pct {np.percentile(rel, 90):9.3e}   max {rel.max():9.3e}")
    yo = yield_of(dec["original"][0], dec["original"][1])
    ys = yield_of(dec["smoothed"][0], dec["smoothed"][1])
    print(f"\n  decay yield on the computed range: "
          f"original {yo:.6g}   smoothed {ys:.6g}   "
          f"change {100*(ys/yo-1):+.3f}%")
    print("\nwrote fig_surface.pdf, fig_rtau.pdf, fig_spectra.pdf, fig_sweep.pdf")


if __name__ == "__main__":
    main()
