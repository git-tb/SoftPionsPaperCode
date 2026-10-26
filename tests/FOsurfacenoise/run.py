#!/usr/bin/env python3
"""
Compute the primary and decay spectra for one sigma mass on two freeze-out
surfaces: the tabulated one and a denoised copy of it.

Both arms are identical in every other respect -- same initdata.csv, same mass,
same grids, same tolerances, same binaries -- so any difference in the output
is attributable to the surface alone.

The tolerances are deliberately far tighter than the pipeline's 1e-5. The point
is not to produce production numbers but to ask whether the integrator can
CONVERGE at all on each surface: specV2 reports GSL's per-call status and
decayV4 reports Cuhre's fail flag, and those counts are collected here.

Nothing is written outside this directory, and ../../src and ../../datanew are
only ever read.

Usage:  make run          (builds ./bin first)
        ./run.py --help
"""
import argparse
import os
import re
import subprocess
import sys
import time

import smooth

ARMS = ("original", "smoothed")

# Grids and tolerances. NpT for the primary spectrum is generous because the
# decay stage interpolates it, and the whole thing costs seconds at one mass.
MSIGMA = 0.6
MPI = 0.14
QTMAX, NQT = 4.0, 800
PTMAX, NPT = 1.0, 200
SPEC_EPSREL, SPEC_ITER = 1e-8, 100000
DECAY_EPSREL, DECAY_ITER = 1e-8, 1000000

# A single tolerance is not enough to see what smoothing buys: at 1e-8 both
# surfaces fail everywhere (that is below the floor either way) and at 1e-3
# both succeed. The sweep finds where the two part company. Run on a coarser
# qT grid since only the diagnostic counts are wanted.
SWEEP_EPSREL = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
SWEEP_NQT = 200

GSL_PATTERNS = {
    "roundoff": "roundoff error",
    "maxiter": "exceeded max number of iterations",
    "singular": "bad integrand behavior",
    "divergent": "integral is divergent",
}


def count_diagnostics(text):
    """Tally the integrator self-reports on a run's stdout."""
    out = {k: len(re.findall(re.escape(v), text)) for k, v in GSL_PATTERNS.items()}
    out["cuhre_fail"] = len(re.findall(
        r"Cuhre did not reach the requested accuracy", text))
    out["nonfinite"] = len(re.findall(r"non-finite result", text))
    return out


def run(cmd, logpath):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(logpath, "w") as f:
        f.write(p.stdout)
        if p.stderr:
            f.write("\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        sys.exit(f"FAILED (rc={p.returncode}): {' '.join(cmd)}\n"
                 f"  see {logpath}")
    return p.stdout, time.time() - t0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../../datanew/PbPb276_cc_(0,5)",
                    help="scenario to take the surface and initdata from")
    ap.add_argument("--mass", type=float, default=MSIGMA,
                    help=f"sigma mass in GeV (default {MSIGMA})")
    args = ap.parse_args()

    for b in ("bin/specV2", "bin/decayV4"):
        if not os.path.isfile(b):
            sys.exit(f"missing {b} -- run `make` first")

    print("=" * 74)
    print("BUILDING THE TWO SURFACES")
    print("=" * 74)
    smooth.build(args.src, "data")

    print()
    print("=" * 74)
    print("TOLERANCE SWEEP - can the source integral converge on each surface?")
    print(f"  GSL roundoff reports out of {4 * SWEEP_NQT} integrations, "
          f"NqT={SWEEP_NQT}")
    print("=" * 74)
    print(f"  {'epsrel':>10s}{'original':>14s}{'smoothed':>14s}")
    sweep = {}
    for eps in SWEEP_EPSREL:
        row = []
        for arm in ARMS:
            data = os.path.join("data", arm)
            out, _ = run(["./bin/specV2",
                          "--m=%.17g" % args.mass,
                          "--pTmax=%.17g" % QTMAX,
                          "--NpT=%d" % SWEEP_NQT,
                          "--epsabs=0",
                          "--epsrel=%.17g" % eps,
                          "--iter=%d" % SPEC_ITER,
                          "--parentdir=out",
                          "--initpath=%s" % os.path.join(data, "initdata.csv"),
                          "--freezeoutpath=%s" % os.path.join(data, "freezeout.csv"),
                          "--foldername=sweep/%s_%g" % (arm, eps)],
                         os.path.join("out", "sweep_%s_%g.log" % (arm, eps)))
            row.append(count_diagnostics(out)["roundoff"])
        sweep[eps] = row
        n = 4 * SWEEP_NQT
        print(f"  {eps:10.0e}{row[0]:>9d}/{n}{row[1]:>9d}/{n}")
    with open(os.path.join("out", "sweep.csv"), "w") as f:
        f.write("epsrel,original,smoothed,total\n")
        for eps, row in sweep.items():
            f.write(f"{eps:g},{row[0]},{row[1]},{4 * SWEEP_NQT}\n")

    print()
    print("=" * 74)
    print(f"COMPUTING  m_sigma = {args.mass} GeV")
    print(f"  primary: NqT={NQT} on [0,{QTMAX}]  epsrel={SPEC_EPSREL:g} "
          f"iter={SPEC_ITER}")
    print(f"  decay  : NpT={NPT} on [0,{PTMAX}]  epsrel={DECAY_EPSREL:g} "
          f"iter={DECAY_ITER}")
    print("=" * 74)

    for arm in ARMS:
        d = os.path.join("out", arm)
        os.makedirs(d, exist_ok=True)
        data = os.path.join("data", arm)

        out, dt = run(["./bin/specV2",
                       "--m=%.17g" % args.mass,
                       "--pTmax=%.17g" % QTMAX,
                       "--NpT=%d" % NQT,
                       "--epsabs=0",
                       "--epsrel=%.17g" % SPEC_EPSREL,
                       "--iter=%d" % SPEC_ITER,
                       "--parentdir=out",
                       "--initpath=%s" % os.path.join(data, "initdata.csv"),
                       "--freezeoutpath=%s" % os.path.join(data, "freezeout.csv"),
                       "--foldername=%s/spec" % arm],
                      os.path.join(d, "spec.log"))
        dg = count_diagnostics(out)
        # 2 spectra (particle and anti) x 2 parts (Re, Im) per qT point
        ntot = 4 * NQT
        print(f"\n  [{arm}] spec  {dt:5.1f}s   "
              f"roundoff {dg['roundoff']:5d}/{ntot}   "
              f"maxiter {dg['maxiter']}   singular {dg['singular']}   "
              f"divergent {dg['divergent']}")

        out, dt = run(["./bin/decayV4",
                       "--ma=%.17g" % args.mass,
                       "--mb=%.17g" % MPI,
                       "--mc=%.17g" % MPI,
                       "--pTmax=%.17g" % PTMAX,
                       "--NpT=%d" % NPT,
                       "--epsabs=0",
                       "--epsrel=%.17g" % DECAY_EPSREL,
                       "--iter=%d" % DECAY_ITER,
                       "--primespecpath=%s" % os.path.join(d, "spec", "spectr.txt"),
                       "--parentdir=out",
                       "--foldername=%s/decay" % arm,
                       "--B=1", "--Q=1"],
                      os.path.join(d, "decay.log"))
        dg = count_diagnostics(out)
        print(f"  [{arm}] decay {dt:5.1f}s   "
              f"cuhre-fail {dg['cuhre_fail']:5d}/{NPT}   "
              f"non-finite {dg['nonfinite']}")

    print("\nDone. Now:  make compare")


if __name__ == "__main__":
    main()
