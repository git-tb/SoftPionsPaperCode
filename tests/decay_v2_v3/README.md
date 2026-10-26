# `decayV2` vs `decayV3` vs `decayV4`

Isolated harness comparing the three implementations of `decayformula.tex`. Self-contained:
it builds the **unmodified** sources from `../../src` into `./bin` and writes every run
into `./out`, so it never touches the top-level `bin/` or any measurement data. Both are
gitignored.

```sh
make            # build ../../src/decayV{2,3,4}.cpp into ./bin
./compare.py    # run the sweep and print the report
./compare.py --quick --NpT 20      # fast smoke test
./compare.py --ma 0.6              # further above threshold
make distclean  # remove ./bin and ./out
```

## Why the comparison is meaningful

All three binaries evaluate the *same* integral over the *same* analytic domain by three
routes:

- **`decayV2`** integrates over a hyperrectangle *containing* the domain (eqs.
  `Cubature_vmin` / `Cubature_tminmax`) and masks the outside with `if(u<=1) return 0`.
  The mask places a $(v-v_1)^{-1/2}$ singular discontinuity along a **curved line in the
  interior** of the region.
- **`decayV3`** maps the exact domain $[v_1(t),1]$ onto the unit square, so the same
  singularity lands on a **face** of the region.
- **`decayV4`** substitutes $v\to\theta$ so the singularity is **cancelled analytically**
  and no singularity is sampled at all.

They agree analytically (verified in `Claude/summaries/*_formulacheck.md`), so any
difference between them is pure discretisation error and the `epsrel` sweep shows which
one owns it.

The report has three parts: non-finite output, agreement at matched `epsrel`, and true
error against a tight-tolerance reference alongside each integrator's *self-reported*
error — a converged integrator's true error should track its requested tolerance.

## What it found: `decayV3` returns NaN at tight tolerance

The headline result is not an accuracy gap but an outright failure. Measured with the
near-threshold parameters of `data/examplescenario` (`ma=0.285714`, `mb=mc=0.14`, so
`p_abc=0.0284`), `NpT=30`, `iter=1e6`:

| `epsrel` | V2 NaN | V3 NaN | V2-vs-V3 median rel. diff | finite pts compared |
|---|---|---|---|---|
| `1e-2` | 0.0% | 0.0% | 1.48e-02 | 30 |
| `1e-3` | 0.0% | 0.0% | 4.96e-03 | 30 |
| `1e-4` | 0.0% | **90.0%** | 3.69e-03 | 3 |
| `1e-5` | 0.0% | **96.7%** | — | 1 |
| `1e-6` | **3.3%** | **100.0%** | — | 0 |

Tightening the tolerance makes the answer *worse*, monotonically, until nothing is left.
Loose tolerances hide it completely: at `1e-2` and `1e-3` the two routes agree to ~1.5%
and ~0.5% with no NaN at all, which is why this has gone unnoticed.

Note the trap for anyone benchmarking these against each other: at `epsrel=1e-5` the two
appear to agree *exactly* (`0.000e+00`), but that statistic rests on the **single** point
that was still finite. The "npts" column exists to stop that from being read as success.

**Root cause**, confirmed by instrumenting the real code path (real `integrate_nested.h`,
real pchip spline, real bounds) rather than by inspection:

```
BAD f: t=0.14668541079976283 v=1  u=1.0000286495604942  u*u-1=5.73e-05  1-v*v=0
       vmin(t)=0.99989561987446041  primespec(tQ)=891.277  -> f=inf
```

Cuhre evaluates the corner `v == vmax == 1` **exactly**. There the integrand's
$1/\sqrt{1-v^2}$ factor is `+inf` — this point is inside the domain, so the `if(u<=1)`
guard does not fire ($u=1.0000286>1$). Genz–Malik cubature rules carry **negative
weights**, so once `+inf` enters the weighted sum the rule evaluates `inf - inf = NaN`,
and Cuhre propagates that through every enclosing region to the final result. It surfaces
only as the tolerance tightens because adaptive subdivision puts progressively more
subregions flush against that face.

Note the singularity is *integrable* — the mathematics is fine, and `decayformula.tex` is
not at fault. This is purely an artefact of sampling a removable-in-the-integral
singularity at the exact point where it diverges.

### `decayV2` is lucky, not immune — and the sweep proves it

`decayV2` maps `v = vmin + (vmax-vmin)*xx[1]`, and at `xx[1]==1` that lands on exactly
`1.0` for **50/50** sampled $p_T$ values — the same `+inf`. It survives longer only
because its rectangle is wide in $v$ (`vmin` is a constant $\approx 0.96$, not $\to 1$),
so Cuhre needs far more subdivision before it crowds subregions against the top face.

This is not speculation: **at `epsrel=1e-6` V2 starts failing too** (3.3% NaN). Pushed
hard enough it breaks by exactly the same mechanism. **Neither version is robust here**;
V3 is simply the one whose geometry triggers it at tolerances you would actually use.

## The fix: `decayV4`

`../../src/decayV4.cpp` applies the sine substitution
$v=\tfrac{a+1}{2}+\tfrac{1-a}{2}\sin\theta$ (with $a=\max(v_1,-1)$), whose $\cos\theta$
Jacobian cancels the $(v-a)^{-1/2}$ and $(1-v)^{-1/2}$ endpoint behaviour **analytically**
— the cancellation is done in closed form per case, not left to the floating-point
arithmetic:

| | transformed integrand |
|---|---|
| $a=v_1$ (i.e. $v_1>-1$) | $C/\sqrt{\beta(u+1)(1+v)}$ |
| $a=-1$ (i.e. $v_1\le-1$) | $C/\sqrt{u^2-1}$ |

with $u=u_0+\beta v$, $u_0=\frac{m_aE_{abc}}{\omega_{T,q}\omega_{T,p}}$,
$\beta=\frac{tQp_T}{\omega_{T,q}\omega_{T,p}}$ and

> **Notation.** $\beta$ here is *not* the branching ratio $B$ of `decayformula.tex`.
> $B$ is a constant (the `--B` flag), applied once at the very end in `decayspec()`;
> $\beta=tQp_T/(\omega_{T,q}\omega_{T,p})$ depends on $t$ and $p_T$ and exists only
> inside the integrand. They are unrelated.

$C=\frac{t\,\text{primespec}(tQ)}{\omega_{T,q}\omega_{T,p}}$. Both are bounded (uniformly
so for $p_T\ge p^a_{b|c}$ — see the residual below), and the domain becomes a plain
rectangle $[t_{\min},t_{\max}]\times[-\pi/2,\pi/2]$ with no $x_1$-dependent inner bound and
no vanishing Jacobian. `decayV4` also checks Cuhre's `fail` flag and the finiteness of each
result, which V2/V3 ignore.

### Residual: the colliding branch points at $v_1=-1$

The substitution removes a *square-root* endpoint. It cannot remove the case where the two
lower branch points **collide**: at $v_1=-1$ both $(v-v_1)$ and $(1+v)$ vanish together and
$g(v)\sim C/(2\sqrt{\beta}\,(v+1))$ — a $1/(v+1)$ pole, so the inner integral is
logarithmically divergent there. Verified numerically: $I(t)$ grows by ~7.7 per decade as
$t\to t^\*$, i.e. $I\sim-k\ln|t-t^\*|$.

- **Occurs iff $p_T<p^a_{b|c}$.** For $p_T\ge p^a_{b|c}$, $\min_t v_1(t)\ge0$ and the
  collision cannot happen; below it $v_1\to-\infty$ as $t\to0$ and must cross $-1$.
- **It belongs to `decayformula.tex`, not to any implementation** — V2, V3 and V4 inherit
  it equally.
- **The physical answer is still finite**: only the inner integral at one isolated $t$
  diverges, and $\int\ln(1/|t-t^\*|)\,\mathrm{d}t$ converges.

So "bounded by construction" above holds *pointwise in $t$* always, but **uniformly only
for $p_T\ge p^a_{b|c}$**. Measured in the untested regime (21 points on $[0,0.05]$, 12 of
them below $p^a_{b|c}$): V4 gives no NaN and agrees with V3 to ~1e-4, but its
self-convergence degrades to ~7e-4 there versus ~1e-5 in the clean region. The earlier
"0% NaN" sweep never exercised this: at `NpT=30, pTmax=1` the only grid point below
$p^a_{b|c}$ is $p_T=0$, which is the special $\beta=0$ case with no divergence.

### Measured result

| `epsrel` | V2 NaN | V3 NaN | **V4 NaN** | V2 true err | V3 true err | **V4 true err** |
|---|---|---|---|---|---|---|
| `1e-2` | 0.0% | 0.0% | **0.0%** | 1.82e-02 | 1.29e-03 | **4.58e-04** |
| `1e-3` | 0.0% | 0.0% | **0.0%** | 5.63e-03 | 3.00e-04 | **9.80e-05** |
| `1e-4` | 0.0% | 90.0% | **0.0%** | 4.86e-03 | 4.54e-05 | **1.24e-05** |
| `1e-5` | 0.0% | 96.7% | **0.0%** | 4.86e-03 | — | **1.83e-06** |
| `1e-6` | 3.3% | 100.0% | **0.0%** | 4.86e-03 | — | **0.00e+00** |

(true error = max relative deviation from a `decayV4 @ epsrel=1e-7` reference)

**V4 is NaN-free everywhere and converges monotonically** to the reference. Two
cross-checks confirm the substitution is the *same* integral, not a different one: where
V3 still works it agrees with V4 to 3.0e-4 at `1e-3` and 4.5e-5 at `1e-4`, tightening as
both converge.

### Bonus finding: V2 does not converge at all

V2's true error **plateaus at 4.86e-3 and stops improving** from `epsrel=1e-4` onward —
tightening the tolerance buys nothing. This vindicates the original prediction about V2:
the curved interior singularity really does defeat Cuhre's adaptive subdivision, so V2
silently stops meeting its requested tolerance while reporting success (it never checks
Cuhre's `fail` flag). V2's numbers are trustworthy only to ~0.5%.

## Caveat

The `--ma` default sits close to threshold, where `p_abc` is small and the true domain is
a thin sliver — the regime the physics runs actually use, and the one that stresses the
geometry hardest. Run with a larger `--ma` to see the better-behaved case.
