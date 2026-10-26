# Freeze-out surface noise

Does the noise in the tabulated freeze-out surface matter?

`freezeout.csv` is not a smooth curve: it carries white noise at ~1e-5 relative in `tau`
and `r`, and ~5e-3 in `Dtau` and `Dr`. `pchip` in `include/freezeout.h` interpolates
**through** every point, so that noise becomes knot-scale curvature in `specV2`'s integrand
— which is what defeats GSL's Gauss-Kronrod error estimate and produces the `roundoff error`
reports seen throughout the spec stage.

This test takes `datanew/PbPb276_cc_(0,5)`, builds a denoised copy of its surface, and runs
the full chain on both at one sigma mass.

```sh
make          # build ./bin/specV2 and ./bin/decayV4 from ../../src
make run      # build both surfaces, compute primary + decay spectra
make compare  # figures + report
```

Nothing is written outside this directory; `../../src` and `../../datanew` are only read.

## The two arms

Identical in every respect — same `initdata.csv`, mass, grids, tolerances, binaries — except
the surface. `smooth.py` builds them:

| arm | surface |
|---|---|
| `original` | `datanew/PbPb276_cc_(0,5)/freezeout.csv` verbatim (397 knots) |
| `smoothed` | Savitzky-Golay filtered: local **quintic in a 17-point window** |

**`Dtau` and `Dr` are the analytic derivative of the same local polynomial that smooths
`tau` and `r`** — the tabulated derivative columns are discarded, not filtered. The shipped
surface is then self-consistent by construction, and this is where the noise reduction
matters most: the derivative columns are ~500x noisier than the geometry columns, and they
are what the integrand is sensitive to.

### Choosing the filter

Two competing measures: the lag-1 autocorrelation of the residual (the test for "removed
noise, not signal" — white noise gives ~0, eating real structure gives a large positive
value) and the smoothness of the resulting derivative.

| filter | `tau` resid | lag-1 ac | `Dtau` noise |
|---|---|---|---|
| tabulated | — | — | 5.93e-2 |
| h=4, d=3 | 9.33e-5 | −0.103 | 5.97e-3 |
| **h=8, d=5** | **1.02e-4** | **+0.076** | **3.90e-3** |
| h=12, d=5 | 1.15e-4 | +0.231 | 1.64e-3 |
| h=16, d=5 | 1.45e-4 | +0.459 | 1.18e-3 |

Raising the degree along with the window is what breaks the deadlock: a quintic tracks real
curvature over a wider span, so h=8/d=5 gives a 1.5x smoother derivative than h=4/d=3 at
equal fidelity to `tau`. Wider windows keep improving the derivative but start removing real
structure. (`r` is slightly less forgiving than `tau` at this setting: ac +0.20 against
+0.08.)

### Alternatives tried and rejected

- **Interpolating spline through the smoothed curve.** A spline passes through every point,
  so it reproduces whatever noise survives and differentiating amplifies it: `Dtau` noise
  4.3e-2, **seven times worse** than the least-squares local fit. Interpolation is not
  smoothing.
- **Global polynomial fit.** A degree-50 Chebyshev still leaves residuals of 2.1e-4 with
  autocorrelation +0.73 — the surface has real features no low-order global basis represents.
- **Fourier / cosine low-pass**, fitted on the actual `alpha` grid to avoid resampling. Loses
  on both axes at every cutoff: the best reachable residual is 6.8e-4 against a 8.6e-5 noise
  floor, and by then ac is still +0.13, while above ~120 modes the least-squares problem on
  this non-uniform grid goes ill-conditioned and the derivative explodes to ~1e9. Three
  reasons it cannot work here: the `alpha` spacing varies by **~770x** so there is no uniform
  grid to transform on; `tau` is non-periodic and steep near π/2, so it is not band-limited;
  and the noise is **localised** (100x stronger in `alpha` ~ [0.35, 0.85)), which a single
  global cutoff cannot adapt to. Wavelet thresholding is the transform-domain method that
  would handle localised noise, but it needs a dependency this project does not have, to beat
  a filter that already reaches 15x smoother derivatives.

The `alpha` spacing also means finite-difference noise estimators are useless here —
everything must be fitted against the actual `alpha` values.

## Settings

`m_sigma = 0.6` GeV. Tolerances are far tighter than the pipeline's 1e-5, deliberately: the
question is not what the production numbers are but whether the integrator can converge at
all on each surface.

| | grid | epsrel | iter |
|---|---|---|---|
| primary | 800 points on [0,4] GeV | 1e-8 | 1e5 |
| decay | 200 points on [0,1] GeV | 1e-8 | 1e6 |

## Results

### The surface changes by very little

| column | rms change | relative |
|---|---|---|
| `tau` | 1.0e-4 | 1.2e-5 |
| `r` | 7.8e-5 | 8.8e-6 |
| `Dtau` | 6.3e-2 | 4.8e-3 |
| `Dr` | 4.6e-2 | 8.8e-3 |

**The noise is not spread evenly.** It sits in bursts, mostly in `alpha` ∈ [0.35, 0.85):

| band | knots | median spacing | residual rms |
|---|---|---|---|
| [0.00, 0.35) | 24 | 1.5e-2 | 7.7e-6 |
| **[0.35, 0.85)** | 140 | 2.5e-3 | **1.7e-4** |
| [0.85, 1.30) | 165 | 2.6e-3 | 3.6e-5 |
| [1.30, 1.57) | 68 | 3.4e-3 | 2.1e-6 |

A factor of 100 between bands, only weakly correlated with sampling density (+0.31 on
log-log). So the input data's quality varies along the surface rather than being uniformly
imprecise — worth knowing before quoting an error bar on it.

### Smoothing helps the integrator, but does not rescue it

GSL roundoff reports, out of 800 integrations (`fig_sweep.pdf`):

| epsrel | original | smoothed |
|---|---|---|
| 1e-3 | 6 | 4 |
| 1e-4 | 68 | 48 |
| **1e-5** | **422** | **318** |
| 1e-6 | 730 | 672 |
| 1e-7 | 798 | 790 |
| 1e-8 | 800 | 800 |

At the production tolerance the failure count drops by ~30%, but it does not vanish. Below
1e-6 both surfaces fail everywhere — that is under the floor regardless. Denoising at the
measured noise amplitude is not enough to make `epsrel=1e-5` cleanly attainable; a stronger
filter would, but only by removing structure the autocorrelation test says is real.

### The primary spectrum is very sensitive; the decay spectrum is not

| | median | 90th pct | max |
|---|---|---|---|
| primary | 4.9e-2 | 7.7e-1 | 81 |
| decay | 1.8e-3 | 1.0e-2 | 1.5e-2 |

**decay yield over [0,1] GeV: −0.010%**

This is the main result, and it cuts both ways:

- A 1e-5 relative change in the surface moves the **primary** spectrum by **5% median**.
  The mechanism is phase amplification — `r*pT` reaches ~140, so 1e-5 in `r` is ~1.4e-3 rad
  of Bessel phase, and the spectrum oscillates. The extreme values (up to 81x) are at the
  spectrum's deep minima, where the relative change explodes while the absolute difference
  stays negligible. Every excursion above 100% sits at qT = 2.6-4.0 GeV, where the spectrum
  is 1e6-1e9 times below its peak; below qT = 1.5 GeV the largest change is 12.5%. So this
  is not a threat to anything. `fig_spectra.pdf` clips these and says so.
- The **decay** stage integrates over that oscillating spectrum, and the noise largely
  averages out: 0.18% median, and only **-0.01%** on the yield.

So the noise does *not* threaten the published yields through this channel. Two things that
do survive, though:

- The primary spectrum should not be quoted point-by-point at better than ~5% even in the
  region that matters.
- The smooth drift in the decay spectrum toward `pT = 1` (`fig_spectra.pdf`, bottom right) is
  a **systematic, not noise** — it does not average away, and would grow if the `pT` range
  were extended. It reaches 1.5% at `pT = 1` with this filter, against 4% with the lighter
  h=4/d=3 one, so smoother derivatives shrink it.

### Effect of the filter strength

Both filters were run through the full chain. Smoother derivatives give a *smaller*
perturbation to the spectra, which is the consistency check one wants:

| | h=4, d=3 | h=8, d=5 (shipped) |
|---|---|---|
| `Dtau` noise | 5.97e-3 | 3.90e-3 |
| primary, median change | 8.1e-2 | 4.9e-2 |
| decay, median change | 6.1e-3 | 1.8e-3 |
| decay drift at `pT`=1 | +4.1% | +1.5% |
| **decay yield change** | **+0.069%** | **−0.010%** |

The yield is stable at the 0.1% level either way, so the conclusion does not depend on the
filter choice. Note the roundoff count at `epsrel=1e-5` is marginally *worse* for the
smoother filter (318 vs 294 of 800) — the GSL failure count tracks the interpolated
integrand's local curvature, not the derivative columns' smoothness, so the two do not move
together.

## Figures

| file | content |
|---|---|
| `fig_surface.pdf` | `tau`, `r`, `Dtau`, `Dr` vs `alpha` for both, with the removed residual underneath |
| `fig_rtau.pdf` | the surface in the (r, tau) plane, plus a zoom on the dense end |
| `fig_spectra.pdf` | primary and decay spectra for both arms, and their ratios |
| `fig_sweep.pdf` | roundoff rate against requested tolerance |

## Caveats

- One mass (0.6 GeV) and one scenario. The decay-stage averaging that suppresses the
  sensitivity should hold generally, but has not been checked across masses.
- The "noise" is whatever a local cubic cannot follow. Genuine structure on a scale below
  ~9 knots would be misattributed; the autocorrelation test argues against it but does not
  exclude it.
- `Dtau`/`Dr` in the smoothed arm are derived from `tau`/`r` rather than filtered in place,
  so they also absorb any inconsistency between the tabulated derivatives and the tabulated
  curves. Median agreement between the two is ~2e-5, so this is a small effect.
