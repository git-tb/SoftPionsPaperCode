#!/usr/bin/env python3
"""
Batch driver for the full soft-pion evaluation, replacing a manual pass through
evaluate.ipynb for every dataset.

Three stages, matching cells 2, 4 and 3+5 of the notebook:

    spec    ./bin/specV2   x n_mass_samples   coherent sigma source spectrum
    decay   ./bin/decayV4  x n_mass_samples   sigma -> pi+ pi- decay spectrum
    plot    weighted sum over the sigma spectral function + figures

Physics constants, sampling grids and integrator settings are NOT in this file.
Each dataset carries its own `config.toml` (see config.template.toml, whose
values are the notebook's), and this script has no defaults to fall back on: a
missing file or a missing key is an error. That keeps a dataset's numbers
attached to the dataset instead of to whatever the script happened to say the
day it was run, and lets scenarios differ in e.g. `scale` without editing code.

Plot styling and figure layout are still carried over verbatim from
evaluate.ipynb so output is directly comparable.

The one deliberate departure is the decay integrator: decayV4 rather than the
notebook's decayV2 (--decay-bin switches back). Same formula and same domain,
but the inner integration variable is sine-substituted so both endpoint
singularities cancel analytically, which is what lets it actually reach the
requested epsrel. decayV4 also reports Cuhre's `fail` flag, which V2/V3 ignore,
so an under-converged run is now visible instead of silent.

Examples
    ./run_evaluation.py --list                      # what would be processed
    ./run_evaluation.py                             # all datasets, all stages
    ./run_evaluation.py 'XeXe*'                     # glob select
    ./run_evaluation.py 'XeXe544_cc_(0,5)plus15' 'AuAu200_cc_(0,5)'
    ./run_evaluation.py --stages plot               # replot only, no recompute
    ./run_evaluation.py '*plus15' --jobs 8 --force  # recompute, decay 8 at a time

--jobs parallelises the DECAY stage only. The spec stage defaults to one run at
a time (--spec-jobs) because specV2 is already OpenMP-parallel across every
core, so running several at once just oversubscribes: measured on 12 cores, 8
spec runs took 13.8 s sequentially and 19.3 s at 8-way concurrency.

By default a stage is SKIPPED when its output already looks complete; --force
recomputes. Recomputation always clears the old output directory first: the
notebook appends timestamped folders instead, so re-running a cell there leaves
two batches behind that the glob in the plotting step would silently merge. A
stage is also NOT skipped when a config key it depends on has changed since the
output was produced -- the changed keys are named.
"""

import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams["mathtext.default"] = "rm"

# ============================================================================
# PER-DATASET CONFIGURATION
# ============================================================================
#
# Physics constants, sampling grids and integrator settings live in
# <dataset>/config.toml, not here. Nothing below is a default: every schema
# entry is required, so a dataset with an incomplete config fails loudly instead
# of being computed at values nobody chose.

CONFIG_NAME = "config.toml"
CONFIG_TEMPLATE = "config.template.toml"

# Degree of the log-space smoothing fit applied to every decay spectrum and to
# the FluiduM baseline. Not in the config; it is defined here because the config
# validation below has to check the sampling grid against it.
POLYFIT_DEG = 10

# (toml section, toml key, attribute, type). The attribute names are what the
# rest of the script reads; see config.template.toml for what each one means.
CONFIG_SCHEMA = (
    ("physics", "m_pi",               "m_pi",               float),
    ("physics", "m_pole",             "m_pole",             float),
    ("physics", "g_pole",             "g_pole",             float),
    ("physics", "amplitude",          "amplitude",          float),
    ("physics", "branching_compute",  "branching_compute",  float),
    ("physics", "branching",          "branching",          float),
    # "scale" is handled separately -- it takes a number OR the string "fit".
    ("physics", "q_scale",            "q_scale",            float),
    ("physics", "charge_multiplicity", "charge_multiplicity", int),

    ("sampling", "n_mass_samples",    "n_mass_samples",     int),
    ("sampling", "msigma_min",        "msigma_min",         float),
    ("sampling", "msigma_max",        "msigma_max",         float),
    ("sampling", "qt_max",            "qt_max",             float),
    ("sampling", "n_qt",              "n_qt",               int),
    ("sampling", "pt_max",            "pt_max",             float),
    ("sampling", "n_pt",              "n_pt",               int),

    ("integration.spec", "epsabs",    "spec_epsabs",        float),
    ("integration.spec", "epsrel",    "spec_epsrel",        float),
    ("integration.spec", "max_iter",  "spec_max_iter",      int),

    ("integration.decay", "epsabs",   "decay_epsabs",       float),
    ("integration.decay", "epsrel",   "decay_epsrel",       float),
    ("integration.decay", "max_iter", "decay_max_iter",     int),
)

# Which config keys each cached stage actually depends on. Editing a key outside
# a stage's set leaves its output valid, so it is not needlessly recomputed.
SPEC_CACHE_KEYS = ("msigma_min", "msigma_max", "n_mass_samples",
                   "qt_max", "n_qt",
                   "spec_epsabs", "spec_epsrel", "spec_max_iter")
DECAY_CACHE_KEYS = ("m_pi", "pt_max", "n_pt", "branching_compute", "q_scale",
                    "decay_epsabs", "decay_epsrel", "decay_max_iter")

CONFIG_SNAPSHOT = ".config.json"    # written into each stage's output root

# physics.scale accepts this instead of a number: solve for the scale that
# makes baseline + coherent reproduce the measured yield where there is data.
SCALE_FIT = "fit"


class Config:
    """The contents of one dataset's config.toml, validated and flattened."""

    def __init__(self, path):
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        seen = set()
        for section, key, attr, typ in CONFIG_SCHEMA:
            node = raw
            for part in section.split("."):
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    raise ValueError(f"{path}: missing section [{section}]")
            if key not in node:
                raise ValueError(f"{path}: missing key '{key}' in [{section}]")
            val = node[key]
            # bool is an int subclass; TOML true would silently become 1.
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{path}: [{section}].{key} must be a number, "
                                 f"got {val!r}")
            if typ is int and not isinstance(val, int):
                raise ValueError(f"{path}: [{section}].{key} must be an integer, "
                                 f"got {val!r}")
            setattr(self, attr, typ(val))
            seen.add((section, key))

        # physics.scale: a number, or "fit" to solve for it at plot time so
        # that baseline + coherent reproduces the measured yield in the
        # measured bins. `scale` stays None until fit_scale() fills it in.
        phys = raw.get("physics")
        if not isinstance(phys, dict) or "scale" not in phys:
            raise ValueError(f"{path}: missing key 'scale' in [physics]")
        sc = phys["scale"]
        seen.add(("physics", "scale"))
        if isinstance(sc, str):
            if sc != SCALE_FIT:
                raise ValueError(f"{path}: [physics].scale must be a number or "
                                 f"the string {SCALE_FIT!r}, got {sc!r}")
            self.scale_is_fit = True
            self.scale = None
        elif isinstance(sc, bool) or not isinstance(sc, (int, float)):
            raise ValueError(f"{path}: [physics].scale must be a number or "
                             f"the string {SCALE_FIT!r}, got {sc!r}")
        else:
            self.scale_is_fit = False
            self.scale = float(sc)

        # Unknown keys are almost always typos, and a typo that silently does
        # nothing is worse here than a hard stop -- the run costs ~6 minutes.
        extra = []
        for section, node in _walk_sections(raw):
            for key in node:
                if (section, key) not in seen:
                    extra.append(f"[{section}].{key}")
        if extra:
            raise ValueError(f"{path}: unknown key(s): {', '.join(sorted(extra))}")

        self._validate()

    def _validate(self):
        if self.msigma_min >= self.msigma_max:
            raise ValueError("sampling.msigma_min must be below msigma_max")
        if self.msigma_min < 2 * self.m_pi:
            raise ValueError(f"sampling.msigma_min ({self.msigma_min}) is below "
                             f"the two-pion threshold 2*m_pi ({2 * self.m_pi}); "
                             f"the decay map is undefined there")
        for attr in ("n_mass_samples", "n_qt", "n_pt", "spec_max_iter",
                     "decay_max_iter", "charge_multiplicity"):
            if getattr(self, attr) < 1:
                raise ValueError(f"{attr} must be >= 1")
        for attr in ("qt_max", "pt_max", "branching", "amplitude"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be > 0")
        if not self.scale_is_fit and self.scale <= 0:
            raise ValueError("physics.scale must be > 0")
        # The yield integration needs a grid spacing, so at least two points.
        # (Repairing a spectrum that came back with non-finite points needs
        # more than POLYFIT_DEG + 1, but that path is only taken when the
        # integrator actually failed, so it is not required up front.)
        if self.n_pt < 2 or self.n_qt < 2:
            raise ValueError("sampling.n_pt and n_qt must be at least 2")
        if self.qt_max < self.pt_max:
            raise ValueError(f"sampling.qt_max ({self.qt_max}) is below pt_max "
                             f"({self.pt_max}); the decay map integrates the "
                             f"primary spectrum above the pion p_T it produces")

    @property
    def amp(self):
        """Physical condensate amplitude in GeV, recovered from `scale`."""
        if self.scale is None:
            raise RuntimeError("physics.scale = 'fit' has not been solved yet; "
                               "fit_scale() must run before amp is read")
        return self.amplitude * np.sqrt(self.scale / self.branching)

    def describe_scale(self):
        return SCALE_FIT if self.scale is None else "%g" % self.scale

    def as_dict(self, keys):
        return {k: getattr(self, k) for k in keys}


def _walk_sections(raw, prefix=""):
    """Yield (dotted section name, leaf-key dict) for every table in `raw`."""
    leaves = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    if prefix:
        yield prefix, leaves
    elif leaves:
        yield "<toplevel>", leaves
    for k, v in raw.items():
        if isinstance(v, dict):
            yield from _walk_sections(v, f"{prefix}.{k}" if prefix else k)


def load_config(ds):
    """Read <ds>/config.toml, or exit with instructions for creating it."""
    path = os.path.join(ds, CONFIG_NAME)
    if not os.path.isfile(path):
        sys.exit(f"no {CONFIG_NAME} in {ds}\n"
                 f"  every dataset carries its own physics, sampling and\n"
                 f"  integrator settings; this script has no defaults.\n"
                 f"  create it with:  cp {CONFIG_TEMPLATE} '{path}'")
    try:
        return Config(path)
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"bad config: {path}: {e}")
    except ValueError as e:
        sys.exit(f"bad config: {e}")


def cache_status(outroot, cfg, keys):
    """(stale_keys, snapshot) for a stage's cached output.

    Returns the config keys whose value differs from the snapshot written when
    that output was produced. Output from before snapshots existed reports no
    staleness -- it is indistinguishable from output that matches.
    """
    want = cfg.as_dict(keys)
    path = os.path.join(outroot, CONFIG_SNAPSHOT)
    if not os.path.isfile(path):
        return [], want
    try:
        with open(path) as f:
            had = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], want
    return sorted(k for k in keys if had.get(k) != want[k]), want


def write_snapshot(outroot, snapshot):
    with open(os.path.join(outroot, CONFIG_SNAPSHOT), "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)


# ============================================================================
# CONSTANTS  (verbatim from evaluate.ipynb cells 1-5)
# ============================================================================

PRIMARYSPECFOLDER = "primaryspec"
DECAYSPECFOLDER = "decayspec"

# --- styling ---
TICKLABELSIZE = 20
FIGSIZE = (7, 7)
AXISLABELSIZE = 20                  # cell 3
AXISLABELSIZE_FULL = 23.5           # cell 5
LINEWIDTH = 2
MARKERSIZE = 5

CMAP = LinearSegmentedColormap.from_list("custom", ["blue", "red"])
CMAP_LBWH = [0.025, 0.025, 0.05, 0.45]
CMAP_LABELSIZE = 15
CMAP_TICKSIZE = 15
LC_LABEL = r"$m\ [GeV]$"

PRIM_XLABEL = r"$q_T\ [GeV]$"
PRIM_YLABEL = r"$(2\pi q_T)^{-1}dN_{coherent}/(dq_Td\eta_q)\ [GeV^{-2}]$"
SPEC_XLABEL = r"$p_T\ [GeV]$"
SPEC_YLABEL = r"$(2\pi p_T)^{-1}dN/(dp_Tdy)\ [GeV^{-2}]$"
LEGEND = r"$\sigma_{PRCC}\to\pi^+\,\pi^-$"

FO_XLABEL = r"$r\ [fm]$"
FO_YLABEL = r"$\tau\ [fm]$"
FO_LC_LABEL = r"$\alpha$"

SPECBIN = "./bin/specV2"

# decayV4 applies the sine substitution that removes both endpoint
# singularities of the v-integral analytically. decayV2 (what the notebook
# used) never reaches the epsrel requested below -- its true error plateaus
# around 0.5% however tight the tolerance -- and decayV3 returns NaN outright
# for epsrel <~ 1e-4. See Claude/summaries/*_formulacheck.md and
# tests/decay_v2_v3/ for the measurements. Pass --decay-bin ./bin/decayV2 to
# reproduce the notebook's original numbers.
DECAYBIN_DEFAULT = "./bin/decayV4"


# ============================================================================
# HELPERS
# ============================================================================

def header_value(path, prefix):
    """Read a '# <prefix>\\t<value>' comment from a result file.

    The notebook indexes lines[3] directly; searching by prefix is equivalent
    for the current layout but survives a header change.
    """
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith(prefix):
                return float(line.replace(prefix, ""))
    raise ValueError(f"{prefix!r} not found in {path}")


def title_for(name):
    """TITLE string, cell 5's regex applied to the dataset name."""
    m = re.match(r"([a-zA-Z]*)(\d*)", name)
    if m.group(1) in ("PbPb", "XeXe"):
        return "%s@%.2f" % (m.group(1), float(m.group(2)) * 1e-2) + r"$\ TeV$"
    if m.group(1) == "AuAu":
        return "%s@%.0f" % (m.group(1), float(m.group(2))) + r"$\ GeV$"
    return "???"


def short_for(name):
    """expShort, cell 5's regex; drives the figure filenames."""
    return re.sub(r"([a-zA-Z]{4}\d{3})_cc_\(0,5\)", r"\1", name)


def spectral_S(k, cfg):
    """Sill spectral function, cell 5, from the config's pole parameters."""
    mpi, mpole, gpole = cfg.m_pi, cfg.m_pole, cfg.g_pole
    msigma = np.sqrt(1 / 4 * (16 * mpi**2 +
                              np.sqrt(16 * gpole**2 * mpole**2 +
                                      (-16 * mpi**2 - gpole**2 + 4 * mpole**2)**2)))
    gam = np.sqrt(1 / 2 * (16 * mpi**2 + gpole**2 - 4 * mpole**2 +
                           np.sqrt(16 * gpole**2 * mpole**2 +
                                   (-16 * mpi**2 - gpole**2 + 4 * mpole**2)**2)))
    delta = 1 / (k**2 - msigma**2 + 1j * gam * np.sqrt(k**2 - (2 * mpi)**2))
    return -1 / np.pi * np.imag(delta)


class PlotLog:
    """Collects everything the plot stage has to say, for <dataset>/plot.log.

    The plot stage is where silent degradation is easiest: a repaired mass, a
    dropped mass, a scale anchored on extrapolated baseline. Those messages
    scroll past in a 12-dataset run, so they are also written next to the
    figures they describe.
    """

    NAME = "plot.log"

    def __init__(self, ds, name):
        self.ds = ds
        self.lines = [
            f"plot stage for {name}",
            "written %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
        ]
        self.problems = 0

    def info(self, msg, echo=None):
        self.lines.append(msg)
        if echo:
            print(echo)

    def warn(self, msg):
        self.problems += 1
        self.lines.append("WARNING: " + msg)
        print(f"      WARNING: {msg}", file=sys.stderr)

    def section(self, title):
        self.lines += ["", "-" * 68, title, "-" * 68]

    def write(self):
        tail = ("no problems reported" if not self.problems
                else f"{self.problems} problem(s) reported above")
        self.lines += ["", "-" * 68, tail, ""]
        path = os.path.join(self.ds, self.NAME)
        with open(path, "w") as f:
            f.write("\n".join(self.lines))
        return path


def bin_edges(centers):
    """Bin edges for experimental data that records only bin centres.

    experimentaldata.csv carries no edges, and the binning is not uniform
    (PbPb: 0.02 GeV wide up to 0.2, then 0.05, then 0.1; XeXe starts with
    [0.05,0.08], [0.08,0.10], [0.10,0.16]). Midpoints between neighbouring
    centres are therefore wrong exactly where the width changes.

    Assuming contiguous bins with each centre the midpoint of its bin, the
    edges follow from the first one alone, e[k+1] = 2 c[k] - e[k], so
    e[k] = (-1)^k e[0] + a[k]. Each interior midpoint between centres is one
    estimate of e[0]; most bins sit inside a stretch of constant width where
    that estimate is exact, so the median picks it out and the width
    transitions are outvoted. Reproduces 0.20 / 0.10 / 0.05 GeV as the lowest
    edge for AuAu / PbPb / XeXe.
    """
    c = np.asarray(centers, dtype=float)
    if len(c) < 2:
        raise ValueError("need at least two experimental points to infer "
                         "bin edges")
    a = np.zeros(len(c) + 1)
    for k in range(len(c)):
        a[k + 1] = 2 * c[k] - a[k]
    sign = (-1.0) ** np.arange(len(c) + 1)
    mid = (c[:-1] + c[1:]) / 2
    e0 = float(np.median(sign[1:-1] * (mid - a[1:-1])))
    edges = sign * e0 + a
    if np.any(np.diff(edges) <= 0):
        raise ValueError("experimental pT points are not consistent with "
                         "contiguous bins centred on them; cannot infer edges")
    return edges


def fit_scale(pT_full, coherent_unscaled, fluidum_interp, pTs_exp, spec_exp,
              spec_exp_err, pTs_fluid):
    """Solve for the `scale` that makes baseline + coherent reproduce the
    measured pion YIELD integrated over the measured bins.

    The window is every experimental bin that lies inside the computed range,
    i.e. with its upper edge at or below pt_max. Each bin contributes

        dN/dy = 2 pi pT dpT * E(pT),   E = (1/2 pi pT) d2N/dpT dy

    at its centre, and the model is summed with the SAME weights at the SAME
    centres, so the data and both model terms are integrated identically and
    the midpoint-rule error of a coarse bin largely cancels in the difference.
    The coherent spectrum is linear in `scale`, so

        scale = (N_exp - N_fluidum) / N_coherent_unscaled

    Both model curves are sampled at the bin centres by interpolating in log
    space, which is how every other resampling in this script is done -- these
    spectra fall by orders of magnitude across the range.

    The statistical error on N_exp is propagated to the scale assuming the
    bins are uncorrelated; it is reported, not used in the fit.

    Returns (scale, diagnostics dict).
    """
    order = np.argsort(pTs_exp)
    pT_e, y_e, err_e = pTs_exp[order], spec_exp[order], spec_exp_err[order]
    edges = bin_edges(pT_e)

    # small tolerance so a bin ending exactly at pt_max is not lost to roundoff
    inside = (edges[:-1] >= pT_full.min()) & (edges[1:] <= pT_full.max() + 1e-9)
    if not inside.any():
        raise ValueError(
            f"no experimental bin lies inside the computed range "
            f"[{pT_full.min():g}, {pT_full.max():g}] GeV; raise "
            f"sampling.pt_max so the fit window is covered")
    lo, hi = edges[:-1][inside], edges[1:][inside]
    pT_e, y_e, err_e = pT_e[inside], y_e[inside], err_e[inside]
    w = 2 * np.pi * pT_e * (hi - lo)

    def at(y):
        good = np.isfinite(y) & (y > 0)
        return np.exp(np.interp(pT_e, pT_full[good], np.log(y[good])))

    n_exp = float(np.sum(w * y_e))
    n_fluid = float(np.sum(w * at(fluidum_interp)))
    n_coh = float(np.sum(w * at(coherent_unscaled)))
    n_exp_err = float(np.sqrt(np.sum((w * err_e) ** 2)))

    excess = n_exp - n_fluid
    if excess <= 0:
        raise ValueError(
            f"cannot fit the scale: on [{lo[0]:g}, {hi[-1]:g}] GeV the "
            f"measured yield ({n_exp:g}) is already at or below the incoherent "
            f"baseline ({n_fluid:g}), so no positive coherent contribution "
            f"reproduces it. Set physics.scale to a number instead.")
    if n_coh <= 0:
        raise ValueError(f"cannot fit the scale: the unscaled coherent yield "
                         f"is {n_coh:g} on [{lo[0]:g}, {hi[-1]:g}] GeV")

    # Bins whose centre lies below the first FluiduM point use an EXTRAPOLATED
    # baseline. fluidumdata.csv starts at 0.11 GeV in the staged scenarios
    # while XeXe experimental data reaches down to 0.065, so this fires.
    extrap = pT_e < np.min(pTs_fluid)
    w_fluid = w * at(fluidum_interp)
    scale = excess / n_coh
    return scale, {
        "window": (float(lo[0]), float(hi[-1])), "n_bins": int(len(pT_e)),
        "exp": n_exp, "exp_err": n_exp_err, "fluidum": n_fluid,
        "coherent_unscaled": n_coh, "excess": excess,
        "scale_err": n_exp_err / n_coh,
        "n_extrapolated": int(extrap.sum()),
        "fluidum_extrapolated_part": float(np.sum(w_fluid[extrap])),
        "fluidum_min": float(np.min(pTs_fluid)),
    }


def logpolyfit(datax, datay, deg=POLYFIT_DEG):
    """np.poly1d of a degree-`deg` fit to log(datay), cell 5."""
    return np.poly1d(np.polyfit(datax, np.log(datay), deg))


def style(ax, xlabel, ylabel, labelsize=AXISLABELSIZE):
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(False)


def inset_colorbar(fig, ax, linecol, label):
    cax = ax.inset_axes(CMAP_LBWH)
    cbar = fig.colorbar(linecol, cax=cax)
    cbar.set_label(label, fontsize=CMAP_LABELSIZE)
    cbar.ax.tick_params(labelsize=CMAP_TICKSIZE)
    return cbar


# Integrator self-reports on stdout (via DEBUGMSG) that mean a run is not
# trustworthy. The first two are decayV4's Cuhre `fail` / non-finite checks;
# decayV2/V3 never inspect that flag, so they can never emit them. The rest are
# GSL's gsl_strerror() text, which specV2 prints per failed qag call.
#
# These were decayV4-only until it was noticed that specV2's GSL messages match
# none of them -- so the spec stage was reporting "all N runs converged" without
# ever having checked anything. Measured at production settings, specV2 emits
# GSL messages on roughly half its integrations (see ROUNDOFF_MARKERS).
QUALITY_MARKERS = ("did not reach the requested accuracy",   # decayV4 / Cuhre
                   "came back non-finite",                   # decayV4
                   "exceeded max number of iterations",      # GSL_EMAXITER
                   "bad integrand behavior",                 # GSL_ESING
                   "integral is divergent")                  # GSL_EDIVERGE

# Treated separately because it is not fixable by spending more effort and, at
# this pipeline's settings, not a correctness problem either. GSL_EROUND means
# qag could not *certify* the requested tolerance on some subinterval; the
# achieved accuracy still tracks the request. Verified on specV2 at m=0.8,
# comparing whole spectra against an epsrel=1e-8 reference:
#
#     epsrel   max rel deviation      GSL roundoff reports
#     1e-4     2.0e-4                  38 / 1600
#     1e-5     2.2e-5                 212 / 1600
#     1e-6     2.3e-6                 360 / 1600
#
# i.e. the error delivered is ~2x the error requested throughout, which is what
# a correctly converged adaptive rule looks like. Counted and reported, not
# flagged as a failure -- otherwise every single spec run trips the alarm.
ROUNDOFF_MARKERS = ("roundoff error",)


def run_batch(cmds, jobs, label):
    """Run subprocess arg-lists, optionally in parallel.

    Returns (failures, quality) where `quality` lists runs that exited 0 but
    reported an under-converged or non-finite result. Those would otherwise be
    invisible: a nonzero-error integral is still written to file as a number.
    """
    failures, quality = [], []
    roundoff = 0
    total = len(cmds)
    done = 0

    def one(idx_cmd):
        idx, cmd = idx_cmd
        p = subprocess.run(cmd, capture_output=True, text=True)
        return idx, cmd, p.returncode, p.stdout, p.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        for idx, cmd, rc, out, err in ex.map(one, enumerate(cmds)):
            done += 1
            if rc != 0:
                failures.append((idx, rc, err.strip()[-400:]))
            else:
                lines = out.splitlines()
                hits = [ln.strip() for ln in lines
                        if any(mk in ln for mk in QUALITY_MARKERS)]
                if hits:
                    quality.append((idx, hits[-1]))
                if any(mk in ln for ln in lines for mk in ROUNDOFF_MARKERS):
                    roundoff += 1
            if done % 25 == 0 or done == total:
                print(f"      {label}: {done}/{total}", flush=True)
    return failures, quality, roundoff


def report_quality(label, quality, total, masses=None, roundoff=0):
    """Surface integrator self-reports from runs that still exited 0."""
    if roundoff:
        print(f"    {label:5}: {roundoff}/{total} runs reported GSL roundoff "
              f"(tolerance not certifiable; achieved accuracy still tracks "
              f"epsrel -- not a failure)")
    if not quality:
        print(f"    {label:5}: all {total} runs converged to the requested accuracy")
        return
    print(f"    {label:5}: {len(quality)}/{total} runs reported a quality problem",
          file=sys.stderr)
    for idx, msg in quality[:5]:
        tag = f"mass {masses[idx]:.4f}" if masses is not None else f"#{idx}"
        print(f"      {tag}: {msg}", file=sys.stderr)
    if len(quality) > 5:
        print(f"      ... and {len(quality) - 5} more", file=sys.stderr)


# ============================================================================
# STAGE 1 - COHERENT SIGMA SPECTRA  (cell 2)
# ============================================================================

def sigma_masses(cfg):
    """The sampled sigma masses: interior points, both endpoints excluded."""
    return np.linspace(cfg.msigma_min, cfg.msigma_max,
                       cfg.n_mass_samples + 2)[1:-1]


def stage_spec(ds, cfg, jobs, force):
    outroot = os.path.join(ds, PRIMARYSPECFOLDER)
    existing = sorted(glob.glob(os.path.join(outroot, "*", "spectr.txt")))
    stale, snapshot = cache_status(outroot, cfg, SPEC_CACHE_KEYS)
    if len(existing) == cfg.n_mass_samples and not force and not stale:
        print(f"    spec : skip ({len(existing)} runs present; --force to redo)")
        return True
    if stale and len(existing) == cfg.n_mass_samples:
        print(f"    spec : config changed since these runs ({', '.join(stale)}), "
              f"recomputing")
    if os.path.isdir(outroot):
        print(f"    spec : clearing {len(existing)} stale run(s)")
        shutil.rmtree(outroot)

    initdata = os.path.join(ds, "initdata.csv")
    freezeout = os.path.join(ds, "freezeout.csv")
    masses = sigma_masses(cfg)
    width = int(np.ceil(np.log10(len(masses))))
    stamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    cmds = []
    for i, m in enumerate(masses):
        folder = f"{PRIMARYSPECFOLDER}/spec_{stamp}_{str(i).zfill(width)}"
        cmds.append([SPECBIN,
                     "--m=%.17g" % m,
                     "--pTmax=%.17g" % cfg.qt_max,
                     "--NpT=%d" % cfg.n_qt,
                     "--epsabs=%.17g" % cfg.spec_epsabs,
                     "--epsrel=%.17g" % cfg.spec_epsrel,
                     "--iter=%d" % cfg.spec_max_iter,
                     "--parentdir=%s" % ds,
                     "--initpath=%s" % initdata,
                     "--foldername=%s" % folder,
                     "--freezeoutpath=%s" % freezeout])

    print(f"    spec : {len(cmds)} runs, jobs={jobs}, "
          f"epsrel={cfg.spec_epsrel:g}, NpT={cfg.n_qt} on [0,{cfg.qt_max:g}]")
    fails, quality, roundoff = run_batch(cmds, jobs, "spec")
    for idx, rc, err in fails[:5]:
        print(f"      FAILED mass #{idx} (rc={rc}): {err}", file=sys.stderr)
    if fails:
        print(f"    spec : {len(fails)} FAILURES", file=sys.stderr)
    report_quality("spec", quality, len(cmds), roundoff=roundoff)
    if not fails:
        write_snapshot(outroot, snapshot)
    return not fails


# ============================================================================
# STAGE 2 - DECAY SPECTRA  (cell 4)
# ============================================================================

def stage_decay(ds, cfg, jobs, force, decaybin):
    outroot = os.path.join(ds, DECAYSPECFOLDER)
    primefiles = sorted(glob.glob(os.path.join(ds, PRIMARYSPECFOLDER, "*", "spectr.txt")))
    if not primefiles:
        print(f"    decay: SKIP - no primary spectra, run the spec stage first",
              file=sys.stderr)
        return False
    existing = sorted(glob.glob(os.path.join(outroot, "*", "decayspec.txt")))
    stale, snapshot = cache_status(outroot, cfg, DECAY_CACHE_KEYS)
    if len(existing) == len(primefiles) and not force and not stale:
        print(f"    decay: skip ({len(existing)} runs present; --force to redo)")
        return True
    if stale and len(existing) == len(primefiles):
        print(f"    decay: config changed since these runs ({', '.join(stale)}), "
              f"recomputing")
    if os.path.isdir(outroot):
        print(f"    decay: clearing {len(existing)} stale run(s)")
        shutil.rmtree(outroot)

    masses = np.array([header_value(f, "# particle mass:") for f in primefiles])
    width = int(np.ceil(np.log10(len(masses))))
    stamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    cmds = []
    for i, msigma in enumerate(masses):
        folder = f"{DECAYSPECFOLDER}/decay_{stamp}_{str(i).zfill(width)}"
        cmds.append([decaybin,
                     "--ma=%.17g" % msigma,
                     "--mb=%.17g" % cfg.m_pi,
                     "--mc=%.17g" % cfg.m_pi,
                     "--pTmax=%.17g" % cfg.pt_max,
                     "--NpT=%d" % cfg.n_pt,
                     "--epsabs=%.17g" % cfg.decay_epsabs,
                     "--epsrel=%.17g" % cfg.decay_epsrel,
                     "--iter=%d" % cfg.decay_max_iter,
                     "--primespecpath=%s" % primefiles[i],
                     "--parentdir=%s" % ds,
                     "--foldername=%s" % folder,
                     "--B=%.17g" % cfg.branching_compute,
                     "--Q=%.17g" % cfg.q_scale])

    print(f"    decay: {len(cmds)} runs, jobs={jobs}, bin={decaybin}, "
          f"epsrel={cfg.decay_epsrel:g}, NpT={cfg.n_pt} on [0,{cfg.pt_max:g}]")
    fails, quality, roundoff = run_batch(cmds, jobs, "decay")
    for idx, rc, err in fails[:5]:
        print(f"      FAILED mass #{idx} (rc={rc}): {err}", file=sys.stderr)
    if fails:
        print(f"    decay: {len(fails)} FAILURES", file=sys.stderr)
    report_quality("decay", quality, len(cmds), masses, roundoff)
    if not fails:
        write_snapshot(outroot, snapshot)
    return not fails


# ============================================================================
# STAGE 3 - PLOTS  (cells 3 and 5, plus the freeze-out surface)
# ============================================================================

def plot_freezeout(ds, short, saveplots, log):
    """Freeze-out surface in the (r, tau) plane, coloured by alpha.

    Not in the notebook; drawn in the same visual grammar as the spectra
    (LineCollection + inset colourbar) for consistency.
    """
    df = pd.read_csv(os.path.join(ds, "freezeout.csv"), comment="#")
    alpha, tau, r = df.to_numpy().T[:3]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    pts = np.column_stack((r, tau)).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    linecol = LineCollection(segs, array=alpha, cmap=CMAP, lw=LINEWIDTH)
    ax.add_collection(linecol)
    ax.set_xlim(0, 1.05 * np.max(r))
    ax.set_ylim(0, 1.05 * np.max(tau))
    style(ax, FO_XLABEL, FO_YLABEL)
    inset_colorbar(fig, ax, linecol, FO_LC_LABEL)
    ax.set_title(os.path.basename(ds))
    fig.tight_layout()
    if saveplots:
        fig.savefig(os.path.join(ds, "fig_%s_freezeout.pdf" % short))
    plt.close(fig)


def plot_primaries(ds, cfg, short, saveplots, log):
    """Cell 3: the sampled coherent sigma spectra, coloured by mass."""
    specfiles = sorted(glob.glob(os.path.join(ds, PRIMARYSPECFOLDER, "*", "spectr.txt")))
    if not specfiles:
        return False
    masses = np.array([header_value(f, "# particle mass:") for f in specfiles])

    lines = []
    for f in specfiles:
        qTs, spec = pd.read_csv(f, comment="#").to_numpy().T
        lines.append(np.column_stack((qTs, spec)))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    linecol = LineCollection(lines, array=masses, cmap=CMAP, lw=LINEWIDTH)
    ax.add_collection(linecol)
    ax.set_yscale("log")
    ax.set_xlim(0, cfg.qt_max)
    style(ax, PRIM_XLABEL, PRIM_YLABEL)
    inset_colorbar(fig, ax, linecol, LC_LABEL)
    fig.tight_layout()
    if saveplots:
        fig.savefig(os.path.join(ds, "fig_%s_primespecs.pdf" % short))
    plt.close(fig)
    return True


def plot_decays(ds, cfg, short, saveplots, log):
    """Cell 5: spectral density, per-mass decay spectra, and the full spectrum.

    Returns (fluidum_yield, prcc_yield) on [0, cfg.pt_max].
    """
    name = os.path.basename(ds.rstrip("/"))
    decayfiles = sorted(glob.glob(os.path.join(ds, DECAYSPECFOLDER, "*", "decayspec.txt")))
    if not decayfiles:
        log.warn("no decay spectra found; run the decay stage first")
        if saveplots:
            log.write()
        return None
    masses = np.array([header_value(f, "# ma:") for f in decayfiles])

    # spectral-function weights: discretised int dmu^2 rho = int 2 mu dmu rho
    weights = 2 * masses * spectral_S(masses, cfg) * np.ptp(masses) / len(masses)
    weights /= np.sum(weights)

    # The notebook ran every decay spectrum through the degree-POLYFIT_DEG
    # log-fit unconditionally, commented "avoid gaps due to failed integral
    # evaluations that lead to inf". That justification is void for decayV4:
    # across 900 converged spectra (AuAu200, PbPb276, XeXe544 at epsrel=1e-5)
    # there were ZERO non-finite or non-positive points.
    #
    # What the fit actually removes is not noise but the two-body decay
    # KINEMATIC EDGE. The pion spectrum from a -> b c falls sharply above
    # p* = sqrt(ma^2/4 - m_pi^2), and a degree-10 polynomial cannot turn that
    # corner. Verified on PbPb276: the steepest log-drop sits at 1.03-1.10 x p*
    # for every mass from 0.35 to 1.4 GeV (the excess is the sigma's motion
    # boosting the edge up), and the fit misses by up to 43% right at it.
    #
    # So the fit was smearing a real physical feature, not repairing a broken
    # one. Effect of dropping it on the PRCC yield (smoothed -> unsmoothed):
    #
    #     AuAu200   70.83 -> 70.86   +0.04%
    #     PbPb276  225.70 -> 232.93  +3.20%
    #     XeXe544  339.39 -> 361.28  +6.45%
    #
    # CORRECTED 2026-09-02: these two columns were previously written the other
    # way round here, reporting the change as -0.04/-3.10/-6.06%. Removing the
    # smoothing RAISES the yields. Re-measured directly from the stored
    # decayspec/ data; the unsmoothed column reproduces yields.csv
    # (70.858 / 232.932 / 361.276) to 6 significant figures.
    #
    # These are changes to the published numbers, deliberately: the smoothed
    # column is what the notebook reported. The fit is kept as a per-mass repair
    # path for spectra that really do come back with non-finite points, so a
    # decayV2 run (--decay-bin) still works -- decayV2's output is genuinely
    # noisy (15% median RMS on XeXe) and does lean on it.
    #
    # NOTE: with n_pt = 50 the edge falls between grid points, so the yield
    # integral resolves it only coarsely. Raising sampling.n_pt sharpens it.
    # Everything below the scale factor is accumulated UNSCALED, because
    # physics.scale may be "fit", in which case it is only solvable once the
    # coherent sum and the FluiduM baseline both exist. The spectrum is linear
    # in scale, so applying it at the end is equivalent for a fixed scale.
    lines = []
    pT_full = None
    spec_full = np.zeros(1)
    repaired, skipped = [], []
    for i, f in enumerate(decayfiles):
        pTs, spec = pd.read_csv(f, comment="#").to_numpy().T
        spec = cfg.charge_multiplicity * spec               # pi+ and pi-

        # np.isinf, which the notebook used, does not catch NaN -- and
        # decayV3/V4 fail with NaN rather than inf. isfinite covers both.
        good = np.isfinite(spec) & (spec > 0)
        if not good.all():
            if good.sum() <= POLYFIT_DEG + 1:
                skipped.append(masses[i])
                lines.append(np.column_stack((pTs, spec)))
                continue
            spec = np.exp(logpolyfit(pTs[good], spec[good])(pTs))
            repaired.append(masses[i])

        lines.append(np.column_stack((pTs, spec)))
        pT_full = pTs
        spec_full = spec_full + weights[i] * spec

    log.section("decay spectra")
    log.info(f"masses read             : {len(decayfiles)}")
    if repaired:
        msg = (f"{len(repaired)}/{len(decayfiles)} masses had non-finite or "
               f"non-positive points and were repaired by the log-poly fit; "
               f"the rest are used as integrated")
        log.info("repaired masses         : "
                 + ", ".join("%.4f" % m for m in repaired), echo=f"      note: {msg}")
    else:
        log.info("repaired masses         : none")
    if skipped:
        log.warn(f"{len(skipped)} mass(es) had too few usable points to repair "
                 f"and were DROPPED from the sum: "
                 + ", ".join("%.4f" % m for m in skipped))

    # Every mass was dropped. Without this the sum stays empty and pT_full stays
    # None, and the failure only surfaces as a TypeError inside np.polyval.
    if pT_full is None:
        log.warn(f"none of the {len(decayfiles)} masses had enough usable "
                 f"points; no spectrum could be assembled")
        log.write()
        print(f"    plot : SKIP - none of the {len(decayfiles)} masses had "
              f"enough usable points", file=sys.stderr)
        return None

    # --- experiment and baseline, needed before the scale can be fitted ------
    pTs_exp, spec_exp, spec_exp_err = pd.read_csv(
        os.path.join(ds, "experimentaldata.csv"), comment="#").to_numpy().T
    pTs_fluid, spec_fluid = pd.read_csv(
        os.path.join(ds, "fluidumdata.csv"), comment="#").to_numpy().T

    # Here the log-fit is an interpolant, not smoothing: fluidumdata.csv is on
    # its own coarse pT grid (36 points from 0.11 GeV) and has to be resampled
    # onto pT_full, which starts at 0 -- so it also extrapolates over the lowest
    # 6 grid points. Measured on AuAu200 it reproduces its own data to 0.07% rms
    # and extrapolates smoothly and monotonically to pT=0, so it stays.
    spec_fluidum_interp = np.exp(logpolyfit(pTs_fluid, spec_fluid)(pT_full))

    # --- the scale ----------------------------------------------------------
    log.section("scale")
    if cfg.scale_is_fit:
        try:
            scale, d = fit_scale(pT_full, spec_full, spec_fluidum_interp,
                                 pTs_exp, spec_exp, spec_exp_err, pTs_fluid)
        except ValueError as e:
            log.warn(str(e))
            log.write()
            print(f"    plot : SKIP - {e}", file=sys.stderr)
            return None
        cfg.scale = scale
        lo, hi = d["window"]
        print(f"    plot : scale fitted to the measured yield on "
              f"[{lo:g}, {hi:g}] GeV ({d['n_bins']} bins)")
        print(f"           measured {d['exp']:.4g} = incoherent "
              f"{d['fluidum']:.4g} + coherent {d['excess']:.4g}"
              f"  ->  scale = {scale:.6g} +- {d['scale_err']:.2g}, "
              f"dsigma_coherent = {1e3 * cfg.amp:.1f} MeV")
        log.info(f"mode                    : fitted (physics.scale = \"fit\")")
        log.info(f"fit window              : [{lo:g}, {hi:g}] GeV, "
                 f"{d['n_bins']} experimental bins")
        log.info(f"measured yield          : {d['exp']:.6g} "
                 f"+- {d['exp_err']:.3g} (bins uncorrelated)")
        log.info(f"incoherent yield        : {d['fluidum']:.6g}")
        log.info(f"required coherent yield : {d['excess']:.6g}")
        log.info(f"unscaled coherent yield : {d['coherent_unscaled']:.6g}")
        log.info(f"FITTED SCALE            : {scale:.8g} "
                 f"+- {d['scale_err']:.3g}")
        log.info(f"dsigma_coherent         : {1e3 * cfg.amp:.4f} MeV")
        if d["n_extrapolated"]:
            log.warn(
                f"{d['n_extrapolated']} bin(s) of the fit window lie BELOW the "
                f"lowest FluiduM data point ({d['fluidum_min']:g} GeV), so the "
                f"incoherent baseline there is EXTRAPOLATED by the degree-"
                f"{POLYFIT_DEG} log-polynomial, not measured. They carry "
                f"{d['fluidum_extrapolated_part']:.4g} of the "
                f"{d['fluidum']:.4g} incoherent yield in the window.")
    else:
        log.info(f"mode                    : fixed (physics.scale)")
        log.info(f"scale                   : {cfg.scale:.8g}")
        log.info(f"dsigma_coherent         : {1e3 * cfg.amp:.4f} MeV")

    spec_full = cfg.scale * spec_full
    lines = [np.column_stack((L[:, 0], cfg.scale * L[:, 1])) for L in lines]
    extra_spec = np.copy(spec_full)          # coherent contribution alone

    # --- spectral density figure ---
    # The notebook wrote the two-pion threshold as a literal 0.28; it is 2*m_pi
    # and now follows the config, so it cannot drift away from the mass used
    # everywhere else.
    threshold = 2 * cfg.m_pi
    fig_dens, ax_dens = plt.subplots(figsize=FIGSIZE)
    ax_dens.plot([threshold, *masses], [0, *spectral_S(masses, cfg)],
                 lw=LINEWIDTH, c="b", label=r"Sill parametrization")
    style(ax_dens, r"$\mu$", r"$\rho(\mu^2)$", AXISLABELSIZE_FULL)
    ax_dens.legend(fontsize=AXISLABELSIZE_FULL, loc=1)
    ax_dens.vlines(threshold, ymin=0, ymax=1.3, lw=2, ls="-.", colors="k")
    ax_dens.text(0.12, 0.9, r"$\mu=2m_\pi$", fontsize=AXISLABELSIZE_FULL, rotation=90)
    ax_dens.set_ylim(0.0001, 1.3)
    ax_dens.set_xlim(0, cfg.msigma_max)
    fig_dens.tight_layout()

    # --- per-mass decay spectra ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    linecol = LineCollection(lines, array=masses, cmap=CMAP, lw=LINEWIDTH)
    ax.add_collection(linecol)
    ax.set_yscale("log")
    ax.set_xlim(0, cfg.pt_max)
    style(ax, SPEC_XLABEL, SPEC_YLABEL, AXISLABELSIZE_FULL)
    inset_colorbar(fig, ax, linecol, LC_LABEL)
    ax.set_title(name)
    fig.tight_layout()

    # --- full spectrum vs experiment and FluiduM ---
    # (experiment, baseline and the interpolant were read above, since the
    #  fitted scale needs them)
    spec_full = spec_full + spec_fluidum_interp

    COL = (1, 0, 0)

    def full_figure(minimal):
        """The full spectrum figure. With `minimal` the plotted content is
        identical; only the legend changes, to the process and the coherent
        amplitude, the latter shown against the PRCC band's red swatch."""
        fig_, ax_ = plt.subplots(figsize=FIGSIZE)
        h_fluid, = ax_.plot(pT_full, spec_fluidum_interp, lw=LINEWIDTH, c="b")
        h_prcc = ax_.fill_between(pT_full, spec_fluidum_interp, spec_full,
                                  facecolor=(*COL, 0.2), edgecolor=COL,
                                  ls="--", lw=LINEWIDTH)
        h_amp, = ax_.plot([], [], lw=0)
        h_exp = ax_.errorbar(pTs_exp, spec_exp, spec_exp_err, c="b", fmt="o",
                             markersize=MARKERSIZE, lw=LINEWIDTH)

        ax_.set_xlim(0, cfg.pt_max)
        ax_.set_ylim(ax_.get_ylim()[0], 1.2 * ax_.get_ylim()[1])
        style(ax_, SPEC_XLABEL, SPEC_YLABEL, AXISLABELSIZE_FULL)
        entries = [(h_exp, title_for(name)),
                   (h_fluid, r"incoherent source" + "\n" + r"(Lu et al., 2024)"),
                   (h_prcc, LEGEND),
                   (h_amp, r"$\Delta\sigma_{\text{coherent}}=$"
                    + "%.0f" % (1e3 * cfg.amp) + r"$\,MeV$")]
        if minimal:
            entries = [entries[0], (h_prcc, entries[3][1])]
        ax_.legend(*zip(*entries), fontsize=AXISLABELSIZE_FULL)
        ax_.text(0.05, 0, r"$\pi^++\pi^-$", fontsize=1.5 * AXISLABELSIZE_FULL)
        fig_.tight_layout()
        return fig_

    fig_full = full_figure(minimal=False)
    fig_min = full_figure(minimal=True)

    if saveplots:
        fig.savefig(os.path.join(ds, "fig_%s_massspecs.pdf" % short))
        fig_full.savefig(os.path.join(ds, "fig_%s_fullspec.pdf" % short))
        fig_min.savefig(os.path.join(ds, "fig_%s_fullspec_min.pdf" % short))
        fig_dens.savefig(os.path.join(ds, "fig_%s_specdense.pdf" % short))
        log.section("figures")
        for f in ("freezeout", "primespecs", "massspecs", "specdense",
                  "fullspec", "fullspec_min"):
            log.info(f"  fig_{short}_{f}.pdf")
    for f in (fig, fig_full, fig_min, fig_dens):
        plt.close(f)

    # --- how many extra pions ---
    dp = pT_full[1] - pT_full[0]
    y_fluid = np.sum(dp * 2 * np.pi * pT_full * spec_fluidum_interp)
    y_prcc = np.sum(dp * 2 * np.pi * pT_full * extra_spec)
    return y_fluid, y_prcc


def stage_plot(ds, cfg, saveplots):
    name = os.path.basename(ds.rstrip("/"))
    short = short_for(name)
    log = PlotLog(ds, name)
    log.info(f"dataset short name      : {short}")

    plot_freezeout(ds, short, saveplots, log)
    if not plot_primaries(ds, cfg, short, saveplots, log):
        log.warn("no primary spectra found; run the spec stage first")
        if saveplots:
            log.write()
        print("    plot : SKIP - no primary spectra", file=sys.stderr)
        return None
    yields = plot_decays(ds, cfg, short, saveplots, log)
    if yields is None:
        print("    plot : SKIP - no decay spectra", file=sys.stderr)
        return None
    y_fluid, y_prcc = yields

    log.section("yields")
    log.info(f"integration range       : [0, {cfg.pt_max:g}] GeV")
    log.info(f"incoherent (FluiduM)    : {y_fluid:.6g}")
    log.info(f"coherent (PRCC)         : {y_prcc:.6g}")
    log.info(f"ratio                   : {100 * y_prcc / y_fluid:.4f}%")

    print(f"    plot : figures written (short={short})")
    print(f"           pions/dy on [0,{cfg.pt_max:g}] GeV -- "
          f"Fluidum: {y_fluid:.2f}   PRCC: {y_prcc:.2f} "
          f"({100 * y_prcc / y_fluid:.1f}%)")
    if saveplots:
        path = log.write()
        note = (f", {log.problems} problem(s)" if log.problems else "")
        print(f"           log: {os.path.basename(path)}{note}")
    return y_fluid, y_prcc


# ============================================================================
# DRIVER
# ============================================================================

def discover(root, selectors):
    all_ds = sorted(d.rstrip("/") for d in glob.glob(os.path.join(root, "*/"))
                    if os.path.isfile(os.path.join(d, "freezeout.csv")))
    if not selectors:
        return all_ds
    picked = []
    for sel in selectors:
        sel = sel.rstrip("/")
        hits = [d for d in all_ds
                if d == sel or os.path.basename(d) == sel
                or glob.fnmatch.fnmatch(os.path.basename(d), sel)]
        if not hits:
            sys.exit(f"no dataset matches {sel!r} under {root}/")
        picked += hits
    return sorted(set(picked))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*",
                    help="dataset names, globs or paths (default: all under --root)")
    ap.add_argument("--root", default="datanew", help="dataset tree (default: datanew)")
    ap.add_argument("--stages", nargs="+", default=["spec", "decay", "plot"],
                    choices=["spec", "decay", "plot"], help="stages to run")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel subprocesses for the DECAY stage (default 1). "
                         "The spec stage ignores this and uses --spec-jobs, "
                         "because specV2 is already OpenMP-parallel internally")
    ap.add_argument("--spec-jobs", type=int, default=1,
                    help="parallel subprocesses for the spec stage (default 1). "
                         "specV2 already saturates every core via OpenMP, so "
                         "running several at once only oversubscribes the "
                         "machine: measured on 12 cores, 8 runs took 13.8 s "
                         "sequentially but 19.3 s at 8-way concurrency. Raise "
                         "this only if you have pinned specV2 to fewer threads "
                         "with OMP_NUM_THREADS")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when output is present (clears it first)")
    ap.add_argument("--decay-bin", default=DECAYBIN_DEFAULT,
                    help=f"decay binary (default {DECAYBIN_DEFAULT}, the "
                         "numerically sound one). Pass ./bin/decayV2 to "
                         "reproduce the notebook's original numbers; see "
                         "Claude/summaries/*_formulacheck.md")
    ap.add_argument("--no-save", dest="saveplots", action="store_false",
                    help="build figures but do not write them")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--summary-csv", default=None,
                    help="write the per-dataset pion yields to this CSV")
    args = ap.parse_args()

    datasets = discover(args.root, args.datasets)
    if args.list:
        print(f"{len(datasets)} dataset(s) under {args.root}/:")
        for d in datasets:
            has_cfg = os.path.isfile(os.path.join(d, CONFIG_NAME))
            note = "" if has_cfg else f"   [no {CONFIG_NAME}]"
            print("   ", os.path.basename(d) + note)
        if not all(os.path.isfile(os.path.join(d, CONFIG_NAME)) for d in datasets):
            print(f"\ncreate a missing config with:  "
                  f"cp {CONFIG_TEMPLATE} '<dataset>/{CONFIG_NAME}'")
        return

    need = []
    if "spec" in args.stages:
        need.append(SPECBIN)
    if "decay" in args.stages:
        need.append(args.decay_bin)
    for b in need:
        if not os.path.isfile(b):
            sys.exit(f"missing binary {b}\n  build it with: "
                     f"make EXE={os.path.basename(b)} build")

    print(f"root    : {args.root}")
    print(f"datasets: {len(datasets)}")
    print(f"stages  : {', '.join(args.stages)}")
    print(f"jobs    : {args.jobs} (decay), {args.spec_jobs} (spec)")
    if args.spec_jobs > 1:
        print(f"          NOTE: specV2 is OpenMP-parallel internally; "
              f"--spec-jobs {args.spec_jobs} will oversubscribe unless "
              f"OMP_NUM_THREADS is set")

    # Load every config before computing anything: a typo in the last dataset's
    # config should not surface an hour into the run.
    configs = {ds: load_config(ds) for ds in datasets}

    summary, failed = [], []
    for n, ds in enumerate(datasets, 1):
        name = os.path.basename(ds)
        cfg = configs[ds]
        print(f"\n[{n}/{len(datasets)}] {name}")
        tail = ("scale=fit (solved at plot time)" if cfg.scale_is_fit else
                f"scale={cfg.scale:g} -> dsigma={1e3 * cfg.amp:.1f} MeV")
        print(f"    config: {cfg.n_mass_samples} masses on "
              f"({cfg.msigma_min:g}, {cfg.msigma_max:g}) GeV, {tail}")
        ok = True
        if "spec" in args.stages:
            ok &= stage_spec(ds, cfg, args.spec_jobs, args.force)
        if ok and "decay" in args.stages:
            ok &= stage_decay(ds, cfg, args.jobs, args.force, args.decay_bin)
        if ok and "plot" in args.stages:
            res = stage_plot(ds, cfg, args.saveplots)
            if res:
                summary.append((name, *res))
        if not ok:
            failed.append(name)

    if summary:
        print("\n" + "=" * 64)
        print(f"{'dataset':<28}{'Fluidum':>11}{'PRCC':>11}{'PRCC/Fl':>11}")
        print("-" * 64)
        for name, yf, yp in summary:
            print(f"{name:<28}{yf:>11.2f}{yp:>11.2f}{100 * yp / yf:>10.1f}%")
        print("=" * 64)
        if args.summary_csv:
            pd.DataFrame(summary, columns=["dataset", "fluidum", "prcc"]).to_csv(
                args.summary_csv, index=False)
            print(f"summary written to {args.summary_csv}")
    if failed:
        print(f"\n{len(failed)} dataset(s) had failures: {', '.join(failed)}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
