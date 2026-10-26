# Soft pions from a coherent chiral condensate

Computes the soft-pion spectrum produced by a coherent $\sigma$ condensate decaying
$\sigma\to\pi^+\pi^-$ on a heavy-ion freeze-out surface, and compares the result — added on
top of the incoherent (FluiduM) hydrodynamic baseline — against measured spectra.

The $\sigma$ is broad, so the whole chain runs over `n_mass_samples` sampled masses (set per
dataset in `config.toml`) which are then integrated against a Sill spectral function.

```
freezeout.csv + initdata.csv
        │   specV2   × n_mass_samples      coherent σ source spectrum
        ▼
primaryspec/spec_*/spectr.txt
        │   decayV4  × n_mass_samples      σ → π⁺π⁻ decay map
        ▼
decayspec/decay_*/decayspec.txt
        │   weighted sum over ρ(μ²), + FluiduM baseline
        ▼
fig_*_fullspec.pdf   +   pion yields
```

**Further documentation**

| | |
|---|---|
| [Claude/VALIDATION.md](Claude/VALIDATION.md) | what has been tested, what it settled, and the numbers behind every claim in §5 |
| [Claude/OPEN_ITEMS.md](Claude/OPEN_ITEMS.md) | known defects and untested assumptions |
| [Claude/HISTORY.md](Claude/HISTORY.md) | chronological record of changes, and how the results moved |
| [Claude/reference/](Claude/reference/) | the two integrals as `.tex` |

---

## 1. Install

Dependencies:

- gcc 13.3.0 (needs `-std=c++2b`) with OpenMP
- boost 1.83 — `sudo apt install libboost-program-options1.83-dev`
  (the header-only `math/interpolators/pchip.hpp` is used too, so the **dev** package is
  required, not just the runtime)
- gsl 2.7.1 — `sudo apt install libgsl-dev`
- Cuba 4.2.2 — bundled in `Cuba-4.2.2/`
- Python 3.12 with `numpy`, `pandas`, `matplotlib` (a `venv/` is expected at the repo root)

Build Cuba first, if `Cuba-4.2.2/libcuba.a` is not already present:

```sh
cd Cuba-4.2.2 && ./configure && make lib && cd ..
```

Then the binaries. The Makefile builds one target at a time via `EXE=`; all land in `bin/`:

```sh
make EXE=specV2  build      # coherent source spectrum
make EXE=decayV4 build      # decay map  (the default — see §5)
make EXE=decayV2 build      # decay map, original — only to reproduce old numbers
```

Every binary takes `--help`. `make EXE=<name> checklibs` runs `ldd` on one of them.

## 2. Set up data

`datanew/` is the dataset tree the pipeline consumes: one directory per dataset, holding four
CSV inputs and a config. The 12 scenarios are already staged there.

> `data/` is an **outdated** tree, kept only for reference and to reproduce the old results.
> `prepare_datanew.py` was the one-time migration that produced `datanew/` from it and does
> not need to be run again.

### 2.1 Adding a dataset

Create `datanew/<Name>/` containing:

| File | Cols | Format |
|---|---|---|
| `freezeout.csv` | 5 | `alpha,tau,r,Dtau,Dr` — surface geometry, **τ and r in fm** |
| `initdata.csv` | 5 | `alpha,RePhi,ImPhi,ReDPhi,ImDPhi` — condensate $\phi$, $D\phi$ |
| `fluidumdata.csv` | 2 | `pT, spectrum` — incoherent baseline |
| `experimentaldata.csv` | 3 | `pT, spectrum, error` |

- `alpha` must span $[0,\pi/2]$ in `freezeout.csv` and `initdata.csv` — that is the
  integration range. Row count is free (both are pchip-interpolated).
- **A header line is mandatory** in `freezeout.csv` and `initdata.csv`; the C++ reader
  discards line 1 unconditionally.
- Column *names* are ignored — everything is positional. Column count and order are not.
- `#` comment lines are allowed at the top of all four.

**The directory name is load-bearing.** Two regexes in the plotting stage parse it, so use
`<4 letters><3 digits>_cc_(0,5)[plusN]`:

```
XeXe544_cc_(0,5)plus15  ->  title "XeXe@5.44 TeV",  figures fig_XeXe544plus15_*.pdf   ✓
MyNewSystem             ->  title "???",            figure path breaks                ✗
```

Only `PbPb`/`XeXe`/`AuAu` are recognised, and the digits mean **×10 GeV** for PbPb/XeXe
(276 → 2.76 TeV) but **GeV** for AuAu (200 → 200 GeV).

**If the new dataset is a freeze-out temperature variation** (a `plusN`), note that the
FluiduM fit is only valid at the best-fit freeze-out temperature. A `plusN` scenario varies
only the freeze-out *geometry*, so it gets its own shifted `freezeout.csv` but must take
`fluidumdata.csv` from the **unshifted system baseline** — not from a shifted FluiduM file.
The staged scenarios follow this rule.

### 2.2 The per-dataset `config.toml`

Physics constants, sampling grids and integrator error goals are **not** in
`run_evaluation.py`; every dataset carries its own `config.toml` and the script has no
defaults to fall back on. A missing file, a missing key, an unknown key (i.e. a typo) or an
out-of-range value is a hard error before any computing starts.

```sh
cp config.template.toml 'datanew/<Name>/config.toml'    # create one
./run_evaluation.py --list                              # flags datasets missing it
```

`config.template.toml` documents every key and carries the notebook's original values, so a
fresh copy reproduces the old settings. The groups are `[physics]`, `[sampling]` and
`[integration.spec]` / `[integration.decay]`. `sampling.n_mass_samples` sets how many $\sigma$
masses the chain runs over; both expensive stages scale linearly in it.

Editing a key invalidates any cached output that depended on it: the script snapshots the
config next to `primaryspec/` and `decayspec/` and recomputes rather than reusing stale
output, naming the changed keys when it does. Editing a plot-only key such as `scale` leaves
both expensive stages cached.

**`scale` is the one free parameter.** The spectrum goes as $B|\mathcal{A}_\sigma|^2$, so the
expensive stages run once at $\tilde B=1$, $\tilde{\mathcal{A}}_\sigma=0.1$ GeV and the
physical result is recovered at plot time by a single multiplier, with
$\mathcal{A}_\sigma=\tilde{\mathcal{A}}_\sigma\sqrt{\text{scale}/B}$. It accepts a number, or
`"fit"` to solve for it against the measured pion yield, summed over the experimental bins
inside $[0, p_{T,\max}]$ — read the caveats in §5 before
using `"fit"`. Changing it needs only a `--stages plot` re-run.

## 3. Run

```sh
./run_evaluation.py --list                    # what would be processed
./run_evaluation.py                           # all datasets, all stages
./run_evaluation.py 'XeXe*'                   # glob select
./run_evaluation.py 'XeXe544_cc_(0,5)plus15'  # one dataset
./run_evaluation.py '*plus15' --jobs 8        # subset, decay stage in parallel
./run_evaluation.py --stages plot             # replot only, no recompute
./run_evaluation.py --summary-csv yields.csv  # cross-dataset yield table
```

Selectors accept exact names, globs or paths. `--root data --stages plot` runs against the
old tree.

| stage | produces |
|---|---|
| `spec` | `primaryspec/spec_<ts>_<idx>/` × `n_mass_samples` |
| `decay` | `decayspec/decay_<ts>_<idx>/` × `n_mass_samples` |
| `plot` | 5 PDFs + `plot.log` + printed pion yields |

A stage is **skipped when its output is already complete and its config keys are unchanged**;
`--force` recomputes and clears the stale output first.

**Runtime**, at the staged datasets' `n_mass_samples = 300`: ~17 min per dataset serially, and
linear in `n_mass_samples`. `--jobs` parallelises the **decay stage only**,
dropping it to ~1 min at `--jobs 8` on 12 cores; the spec stage is core-bound at ~5–6 min
either way. The spec stage deliberately ignores `--jobs` and uses `--spec-jobs` (default 1),
because `specV2` is already OpenMP-parallel across every core — running several at once
measured ~40% *slower*. Only raise `--spec-jobs` if you have pinned `specV2` to fewer threads
with `OMP_NUM_THREADS`.

## 4. What comes out

Per dataset, written into the dataset directory:

| output | content |
|---|---|
| `fig_<short>_fullspec.pdf` | **the headline plot** — baseline + coherent excess vs experiment |
| `fig_<short>_fullspec_min.pdf` | the same plot with a reduced legend: only the process and $\Delta\sigma_\text{coherent}$ |
| `fig_<short>_freezeout.pdf` | freeze-out surface in the $(r,\tau)$ plane, coloured by $\alpha$ |
| `fig_<short>_primespecs.pdf` | the coherent $\sigma$ spectra, coloured by mass |
| `fig_<short>_massspecs.pdf` | the decay spectra |
| `fig_<short>_specdense.pdf` | the Sill spectral function |
| `plot.log` | the plot stage's full reasoning (below) |
| `primaryspec/`, `decayspec/` | `n_mass_samples` runs each, plus a `.config.json` snapshot |

Across datasets, `--summary-csv <file>` writes `dataset,fluidum,prcc` — the
$\mathrm{d}N/\mathrm{d}y$ pion yields integrated over $[0, \texttt{pt\_max}]$. `yields.csv`
and `yields_fit.csv` in the repo root are the fixed-scale and fitted-scale production runs.

### Diagnostics

**`<dataset>/plot.log`** is written whenever figures are, and records the scale mode with its
full arithmetic, any repaired or dropped masses, the figures written, the yields, and a
closing problem count. The plot stage is where silent degradation is easiest, so its messages
sit next to the figures they describe rather than only scrolling past in a 12-dataset run.

**Integrator self-reports** are surfaced during the run, in three tiers:

- *quality problem* — `decayV4` reporting Cuhre's `fail` flag or a non-finite result, or GSL
  reporting `EMAXITER`/`ESING`/`EDIVERGE`. Treat these as real; they name the affected masses.
- *GSL roundoff* — counted and reported as a note, not a failure. The spec stage reports this
  on ~200 of its 1600 integrations per run at `epsrel=1e-5`. Expected and benign: the
  achieved accuracy still tracks `epsrel` — see [VALIDATION.md](Claude/VALIDATION.md).
- *hard failure* — non-zero exit.

An under-converged integral still exits 0 and still writes a plausible-looking number to
file, which is why this reporting exists. Note it only became correct on 2026-08-31: logs
from before then that claim "all runs converged" for the **spec** stage checked nothing.

## 5. Limitations to be aware of

**These results supersede the notebook's.** Two corrections account for nearly all of the
difference — a better-behaved decay integrator, and dropping the decay-spectrum smoothing —
and both are described below. Together they raise the coherent yields by
**+0.87% / +3.66% / +6.82%** (AuAu / PbPb / XeXe) relative to `evaluate.ipynb`, whose numbers
should be treated as superseded rather than as a reference. [HISTORY.md](Claude/HISTORY.md)
has the ledger with before/after.

**Use `decayV4`, not `decayV2`.** Three implementations of the same decay integral exist.
They agree analytically; they do not agree numerically. `decayV4` handles the integrand's
endpoint divergences analytically instead of sampling through them:

| | behaviour |
|---|---|
| `decayV2` | notebook's original. True error **plateaus at ~0.5%** regardless of `epsrel` |
| `decayV3` | exact integration domain, but returns **NaN** for `epsrel <~ 1e-4` |
| `decayV4` | endpoint singularities cancelled analytically; converges, and reports non-convergence |

`run_evaluation.py` defaults to `decayV4`. `--decay-bin ./bin/decayV2` reproduces the
notebook's old numbers. Measurements in [tests/decay_v2_v3/](tests/decay_v2_v3/README.md).

**Decay spectra are not smoothed.** The notebook pushed every decay spectrum through a
degree-10 log-polynomial fit, to "avoid gaps due to failed integral evaluations". With
`decayV4` there are no such gaps, and what the fit actually smeared is the two-body decay
kinematic edge — by up to 43% right at the edge. It survives only as a per-mass repair path
for spectra that genuinely come back non-finite, which keeps the `decayV2` path working. The
FluiduM baseline still uses the same fit, but there it is an *interpolant* and extrapolator,
not smoothing, and it stays.

**The kinematic edge is under-resolved at the staged `n_pt = 50`.** The edge falls between grid points,
so the yield integral resolves it only coarsely. Raising `sampling.n_pt` sharpens it, and the
yields should be expected to move when it does. This is a quadrature limitation, not an error
in the computation — each point is independently converged.

**`spec.epsrel` cannot usefully go below ~1e-5.** The source integral is roundoff-limited
there; tightening buys progressively less and eventually nothing. Further accuracy would have
to come from reformulating the integrand, not from `epsrel` or `max_iter`.

**Do not quote the primary spectrum point-by-point better than ~5%.** The tabulated
freeze-out surface carries noise at ~1e-5 relative, and phase amplification through the
Bessel factors turns that into ~5% median on the primary spectrum. It does **not** threaten
the yields — propagated through the decay stage it is −0.01% — but the intermediate spectrum
is not that precise. See [tests/FOsurfacenoise/](tests/FOsurfacenoise/README.md).

**`scale = "fit"` is not comparable across systems.** It matches the yield on the measured
window, which starts at 0.05 / 0.10 / 0.20 GeV for XeXe / PbPb / AuAu, so each system leaves
a different stretch of low $p_T$ unconstrained. For all four XeXe scenarios the lowest two
bins lie below the lowest FluiduM point, so the baseline there is extrapolated (32 of the 720
incoherent pions in the window). The plot stage warns. Fitting also absorbs the freeze-out
temperature dependence into the scale, so the `plusN` comparison means something different
than it does at fixed scale.

**The normalisation constant has never been checked numerically**, only by algebra, and the
claimed independence of the `--Q` scale has never been tested. Both are in
[OPEN_ITEMS.md](Claude/OPEN_ITEMS.md) and worth closing before the numbers go in the paper.

## 6. The original notebook — superseded

`evaluate.ipynb` is the original analysis and still runs, but **its results are outdated**:
it uses `decayV2` and smooths the decay spectra, both of which §5 explains are wrong. It is
kept for reference and for reproducing the old numbers, not as a source of truth.

`run_evaluation.py` began as a port of cells 2–5, verified by reproducing the notebook's
hard-coded yields to the digit *before* the integrator and the smoothing were changed, and
then diverged from it deliberately. It adds dataset selection, parallelism, resumability,
per-dataset configs, diagnostics and the freeze-out figure.

Two caveats if you run the notebook directly: cells 2 and 4 *append* timestamped output
folders rather than replacing them, so delete `primaryspec/` and `decayspec/` before
recomputing; and the gap filter in cell 5 tests `np.isinf`, which does not catch NaN — fine
with `decayV2`, but switching it to `decayV3`/`decayV4` needs `~np.isfinite` instead.

## 7. Layout

```
src/            specV2.cpp, specV2eta.cpp, decayV{2,3,4}.cpp
include/        headers: freezeout, initdata, filereader, savedata, integrate_nested, ...
bin/            built binaries
Cuba-4.2.2/     bundled Cuba integration library
datanew/        the dataset tree the pipeline uses
data/           outdated tree, kept for reference and old-result reproduction
tests/          decay_v2_v3/     — integrator comparison harness
                FOsurfacenoise/  — freeze-out surface noise propagation
Claude/         VALIDATION.md, OPEN_ITEMS.md, HISTORY.md, INSTRUCTIONS.md
                reference/  — the formulae as .tex
                summaries/  — dated analysis logs
evaluate.ipynb  original notebook (superseded, see §6)
run_evaluation.py, config.template.toml
prepare_datanew.py  one-time data/ -> datanew/ migration; not part of the workflow
```
