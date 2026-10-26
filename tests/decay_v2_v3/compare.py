#!/usr/bin/env python3
"""
Compare decayV2 against decayV3.

Both binaries evaluate the SAME integral of decayformula.tex over the SAME
analytic domain, by two different routes:

  decayV2  integrates over a hyperrectangle CONTAINING the domain (eqs.
           Cubature_vmin / Cubature_tminmax) and masks the outside with
           `if(u<=1) return 0`. The mask puts a (v-v1)^(-1/2) singular
           discontinuity along a CURVED line in the INTERIOR of the region.

  decayV3  maps the exact domain [v1(t),1] onto the unit square, so the same
           singularity lands on a FACE of the integration region.

Since they agree analytically, any difference between them is pure
discretisation error, and the epsrel sweep shows which one owns it.

Two things are measured:

  1. AGREEMENT   V2 vs V3 at matched epsrel.
  2. CONVERGENCE each run against a tight-tolerance V3 reference, alongside the
     error each integrator REPORTS for itself. A converged integrator's true
     error should track its requested tolerance; if V2's does not, its own
     error estimate cannot be used to defend its numbers.

Nothing is written outside this directory, and the sources in ../../src are
only ever read.

Usage:
    make && ./compare.py                  # default sweep
    ./compare.py --quick                  # fewer points, for a smoke test
    ./compare.py --ma 0.6                 # a different parent mass
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"

# Near-threshold default: ma - (mb+mc) is small, so p_abc is small and the true
# integration domain is a thin sliver inside V2's bounding rectangle. This is
# the regime the physics runs in (see data/examplescenario/decayspec/*) and the
# one where the two routes are most likely to part company.
DEFAULT_PRIMESPEC = (
    ROOT / "data/examplescenario/primaryspec/spec_20250803_184649_000/spectr.txt"
)
DEFAULT_MA = 0.285714
DEFAULT_MB = DEFAULT_MC = 0.14


def read_spec(path):
    """Read a decayspec.txt (leading '#' comments, then a pT,decayspec header).

    numpy's skiprows counts raw lines, so it would consume a comment rather than
    the header; the header is dropped explicitly instead.
    """
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(f) for f in line.split(",")])
        except ValueError:
            continue  # the pT,decayspec header
    data = np.asarray(rows)
    return data[:, 0], data[:, 1]


def run(binary, epsrel, args, tag):
    """Run one binary, return (pT, spectrum). Output lands in ./out/<tag>."""
    folder = OUT / tag
    if folder.exists():
        shutil.rmtree(folder)
    cmd = [
        str(HERE / "bin" / binary),
        "--primespecpath", str(args.primespec),
        "--ma", str(args.ma), "--mb", str(args.mb), "--mc", str(args.mc),
        "--Q", str(args.Q), "--B", "1.0",
        "--pTmax", str(args.pTmax), "--NpT", str(args.NpT),
        "--epsrel", str(epsrel), "--epsabs", "0",
        "--iter", str(args.iter),
        "--parentdir", str(OUT), "--foldername", tag,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"{binary} failed (epsrel={epsrel}):\n{proc.stderr[-2000:]}")
    return read_spec(folder / "decayspec.txt")


def nan_frac(y):
    """Fraction of the spectrum that came back non-finite."""
    return float(np.mean(~np.isfinite(y))) if y.size else 1.0


def reldiff(a, b, floor_frac=1e-6):
    """Relative difference, ignoring negligible points and non-finite entries.

    The decay spectrum spans orders of magnitude; comparing relative error on
    points many decades below the peak measures noise, not disagreement, so
    points below floor_frac of the peak are masked out. Non-finite entries are
    masked too and reported separately by nan_frac -- a run that returns NaN has
    failed outright, which is a stronger statement than "it is inaccurate".
    """
    good = np.isfinite(a) & np.isfinite(b)
    if not good.any():
        return np.zeros_like(a), np.zeros_like(a, dtype=bool)
    scale = max(np.max(np.abs(a[good])), np.max(np.abs(b[good])))
    mask = good & ((np.abs(a) > floor_frac * scale) |
                   (np.abs(b) > floor_frac * scale))
    d = np.zeros_like(a)
    denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    d[mask] = np.abs(a[mask] - b[mask]) / denom[mask]
    return d, mask


def fmt(d, m):
    """Format a masked statistic, or '-' when nothing survived the mask."""
    return (f"{np.median(d[m]):>11.3e}", f"{np.max(d[m]):>11.3e}") if m.any() \
        else ("          -", "          -")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primespec", type=pathlib.Path, default=DEFAULT_PRIMESPEC)
    ap.add_argument("--ma", type=float, default=DEFAULT_MA)
    ap.add_argument("--mb", type=float, default=DEFAULT_MB)
    ap.add_argument("--mc", type=float, default=DEFAULT_MC)
    ap.add_argument("--Q", type=float, default=1.0)
    ap.add_argument("--pTmax", type=float, default=1.0)
    ap.add_argument("--NpT", type=int, default=40)
    ap.add_argument("--iter", type=int, default=1000000)
    ap.add_argument("--quick", action="store_true", help="short sweep")
    args = ap.parse_args()

    if not args.primespec.exists():
        sys.exit(f"primary spectrum not found: {args.primespec}")
    if not (HERE / "bin" / "decayV3").exists():
        sys.exit("binaries missing - run `make` first")

    sweep = [1e-2, 1e-3, 1e-4] if args.quick else [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ref_eps = sweep[-1] / 10.0
    variants = ("decayV2", "decayV3", "decayV4")
    # V4 is the reference: it is the only route whose integrand is bounded, so
    # it is the only one that can be trusted at a tight tolerance.
    ref_variant = "decayV4"

    p_abc = 1 / (2 * args.ma) * np.sqrt(
        ((args.ma + args.mb) ** 2 - args.mc ** 2)
        * ((args.ma - args.mb) ** 2 - args.mc ** 2))
    OUT.mkdir(exist_ok=True)

    print(f"decayV2 vs decayV3")
    print(f"  primary spectrum : {args.primespec.relative_to(ROOT)}")
    print(f"  ma={args.ma} mb={args.mb} mc={args.mc}  ->  p_abc={p_abc:.6f}")
    print(f"  Q={args.Q}  pTmax={args.pTmax}  NpT={args.NpT}  iter={args.iter}")

    print(f"\nreference: {ref_variant} @ epsrel={ref_eps:g}")
    pT, ref = run(ref_variant, ref_eps, args, "ref")

    results = {}
    for eps in sweep:
        for v in variants:
            results[(v, eps)] = run(v, eps, args, f"{v}_{eps:g}")[1]
        print(f"  ran epsrel={eps:g}")

    # ---- 0. did any route fail outright? ------------------------------------
    print("\n== non-finite output (NaN/inf) ==")
    print("a route returning NaN has FAILED, which is stronger than 'inaccurate'")
    hdr = " | ".join(f"{v[-2:]+' NaN':>8}" for v in variants)
    print(f"\n{'epsrel':>9} | {hdr}")
    print("-" * (12 + 11 * len(variants)))
    for eps in sweep:
        row = " | ".join(f"{nan_frac(results[(v, eps)]):>7.1%}" for v in variants)
        print(f"{eps:>9.0e} | {row}")
    print(f"{'ref':>9} | {nan_frac(ref):>7.1%} ({ref_variant})")

    # ---- 1. agreement with the reference route at matched epsrel ------------
    print(f"\n== each route vs {ref_variant} at matched epsrel (finite points only) ==")
    print(f"{'epsrel':>9} | {'route':>8} | {'median rel.diff':>15} | "
          f"{'max rel.diff':>12} | {'npts':>5}")
    print("-" * 64)
    for eps in sweep:
        for v in variants:
            if v == ref_variant:
                continue
            d, m = reldiff(results[(v, eps)], results[(ref_variant, eps)])
            med, mx = fmt(d, m)
            print(f"{eps:>9.0e} | {v:>8} | {med:>15} | {mx:>12} | {int(m.sum()):>5}")

    # ---- 2. convergence against the tight reference -------------------------
    if nan_frac(ref) > 0:
        print("\n== convergence vs tight reference: SKIPPED ==")
        print(f"the reference run ({ref_variant} @ epsrel={ref_eps:g}) is itself "
              f"{nan_frac(ref):.0%} NaN,")
        print("so there is no trustworthy baseline to converge against.")
    else:
        print(f"\n== true error vs the tight {ref_variant} reference ==")
        print("(a column that stops shrinking as epsrel tightens has stopped")
        print(" converging: its requested tolerance is not being met)")
        cols = " | ".join(f"{v[-2:]+' med':>10} | {v[-2:]+' max':>10}" for v in variants)
        print(f"\n{'epsrel':>9} | {cols}")
        print("-" * (12 + 26 * len(variants)))
        for eps in sweep:
            row = " | ".join(
                " | ".join(fmt(*reldiff(results[(v, eps)], ref))) for v in variants)
            print(f"{eps:>9.0e} | {row}")

    # ---- verdict ------------------------------------------------------------
    print("\n== verdict ==")
    worst = {v: max(nan_frac(results[(v, e)]) for e in sweep) for v in variants}
    bad = {v: f for v, f in worst.items() if f > 0}
    if bad:
        print("NaN observed: " + ", ".join(f"{v} up to {f:.0%}" for v, f in bad.items()))
        print("Root cause (confirmed by instrumenting the real code path): Cuhre")
        print("evaluates the corner v == vmax == 1 exactly, where the 1/sqrt(1-v*v)")
        print("factor is +inf. Genz-Malik rules carry NEGATIVE weights, so +inf in")
        print("the weighted sum gives inf-inf = NaN, poisoning the whole result. It")
        print("surfaces as the tolerance tightens because subdivision puts more")
        print("subregions against that face. See README.md.")
    if ref_variant not in bad:
        tight = sweep[-1]
        d, m = reldiff(results[(ref_variant, tight)], ref)
        if m.any():
            print(f"\n{ref_variant} self-consistency at epsrel={tight:g} vs "
                  f"{ref_eps:g}: max rel {np.max(d[m]):.2e}")
        print(f"-> {ref_variant} is NaN-free across the whole sweep and is the only")
        print("   route safe to tighten. Its integrand is bounded by construction:")
        print("   the sine substitution cancels both endpoint singularities exactly.")

    np.savez(OUT / "results.npz", pT=pT, ref=ref, sweep=np.array(sweep),
             **{f"{v}_{e:g}": r for (v, e), r in results.items()})
    print(f"\nraw data: {(OUT / 'results.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
