#!/usr/bin/env python3
"""
Build a clean dataset tree (default: ./datanew) from ./data, one directory per
experimental scenario, ready for the evaluate.ipynb pipeline.

Supersedes the copyfiles.sh + addheaders.sh pair: it does the same renaming and
header normalisation, but per-scenario, idempotently, and with the FluiduM
sourcing rule below.

THE FLUIDUM RULE
    The FluiduM fit is only valid at the best-fit freeze-out temperature, i.e.
    for the unshifted _cc_(0,5) scenario. The temperature-shifted scenarios
    (plus5/plus10/plus15) vary only the freeze-out GEOMETRY; they must still be
    compared against the BASELINE FluiduM spectrum.

    So for every scenario of a system:
        freezeout.csv        <- that scenario's own Freeze_out_*.csv   (shifted)
        fluidumdata.csv      <- the SYSTEM BASELINE FluiduMData_*_cc_(0,5).csv
        initdata.csv         <- that scenario's own
        experimentaldata.csv <- that scenario's own

    The scenario's own (unused) FluiduMData_*_plusNMeV.csv is still copied
    across for provenance, but it is NOT what fluidumdata.csv is made from.

HEADER NORMALISATION (--fix-headers, ON by default)
    ProcessFreezeoutData() in include/freezeout.h reads freezeout.csv with
    has_header=true, so it always discards the first line. 9 of the 12
    Freeze_out_*.csv originals (every plusN one) have NO header line, so their
    alpha=0 row is silently eaten and the spline starts at the second point
    instead. The 3 unshifted ones do have a header and are unaffected -- which
    means baseline and shifted scenarios are currently treated inconsistently.

    This script adds the missing header so no data row is lost. Pass
    --no-fix-headers to reproduce the old byte-for-byte behaviour (and the bug).

Usage:
    ./prepare_datanew.py --dry-run     # show what would happen
    ./prepare_datanew.py               # build ./datanew
    ./prepare_datanew.py --dst /tmp/dn --src data
"""

import argparse
import glob
import math
import os
import re
import shutil
import sys

# Canonical file names consumed by evaluate.ipynb / specV2 / decayV2.
CANON_FREEZEOUT = "freezeout.csv"
CANON_FLUIDUM = "fluidumdata.csv"
CANON_INITDATA = "initdata.csv"
CANON_EXPERIMENT = "experimentaldata.csv"

# Headers that must be present (the C++ reader consumes line 1 unconditionally).
HDR_FREEZEOUT = "alpha,tau,r,Dtau,Dr"
HDR_FLUIDUM = "PT [GEV],(1/Nev)*(1/(2*PI*PT))*D2(N)/DPT/DYRAP [GEV**-2]"

# data/<System><Energy>_cc_(0,5)[plusNNN]  -- the shape evaluate.ipynb's own
# regexes require; anything else would break TITLE and the figure filenames.
SCENARIO_RE = re.compile(r"^(?P<system>[A-Za-z]{4})(?P<energy>\d{3})_cc_\(0,5\)(?P<offset>plus\d+)?$")

ALPHA_MAX_NEEDED = math.pi / 2


def looks_like_header(line):
    """True if the first line is a header rather than a data row."""
    s = line.strip()
    return not (s[:1].isdigit() or s[:1] in "-.")


def read_text(path):
    with open(path, "r") as f:
        return f.read()


def find_one(directory, pattern, exclude=()):
    hits = [p for p in sorted(glob.glob(os.path.join(directory, pattern)))
            if os.path.basename(p) not in exclude]
    return hits


class Scenario:
    def __init__(self, srcdir):
        self.src = srcdir
        self.name = os.path.basename(srcdir.rstrip("/"))
        m = SCENARIO_RE.match(self.name)
        self.valid = m is not None
        if self.valid:
            self.system = m.group("system")
            self.energy = m.group("energy")
            self.offset = m.group("offset") or ""
            self.is_baseline = not self.offset
            self.key = self.system + self.energy
        else:
            self.system = self.energy = self.offset = self.key = None
            self.is_baseline = False


def discover(src):
    scen = []
    for d in sorted(glob.glob(os.path.join(src, "*/"))):
        s = Scenario(d)
        if s.valid:
            scen.append(s)
    return scen


def emit(path, content, dry, actions, label):
    actions.append(label)
    if not dry:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def normalise(src_path, header, fix, dry, actions, canon):
    """Return the text for a canonical file, adding `header` when missing."""
    text = read_text(src_path)
    first = text.split("\n", 1)[0]
    if looks_like_header(first):
        actions.append(f"{canon} <- {os.path.basename(src_path)}")
        return text
    if fix:
        actions.append(f"{canon} <- {os.path.basename(src_path)}  [+HEADER ADDED]")
        return header + "\n" + text
    actions.append(f"{canon} <- {os.path.basename(src_path)}  [!! NO HEADER: "
                   f"alpha=0 row will be eaten by the C++ reader]")
    return text


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data", help="source tree (default: data)")
    ap.add_argument("--dst", default="datanew", help="destination tree (default: datanew)")
    ap.add_argument("--dry-run", action="store_true", help="print actions, write nothing")
    ap.add_argument("--no-fix-headers", dest="fix", action="store_false",
                    help="copy byte-for-byte, keeping the missing-header bug")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing destination tree")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"source tree not found: {args.src}")

    scenarios = discover(args.src)
    if not scenarios:
        sys.exit(f"no scenarios matching {SCENARIO_RE.pattern} under {args.src}/")

    # Baseline FluiduM original per system -- the ONLY valid FluiduM spectrum.
    baselines = {}
    for s in scenarios:
        if s.is_baseline:
            hits = find_one(s.src, "FluiduMData*.csv")
            if len(hits) != 1:
                sys.exit(f"expected exactly one FluiduMData*.csv in {s.src}, found {len(hits)}")
            baselines[s.key] = hits[0]

    missing = sorted({s.key for s in scenarios} - set(baselines))
    if missing:
        sys.exit(f"no baseline _cc_(0,5) scenario found for: {', '.join(missing)}")

    print(f"source      : {args.src}")
    print(f"destination : {args.dst}{'  (DRY RUN, nothing written)' if args.dry_run else ''}")
    print(f"scenarios   : {len(scenarios)}")
    print(f"header fix  : {'ON' if args.fix else 'OFF (byte-for-byte)'}")
    print("\nbaseline FluiduM spectrum per system (used by ALL its scenarios):")
    for k in sorted(baselines):
        print(f"  {k}: {os.path.basename(baselines[k])}")

    if os.path.exists(args.dst) and not args.dry_run:
        if not args.force:
            sys.exit(f"\ndestination {args.dst} already exists; pass --force to overwrite")
        shutil.rmtree(args.dst)

    header_fixed, warnings = [], []
    print()
    for s in scenarios:
        actions = []
        dstdir = os.path.join(args.dst, s.name)

        # --- the four canonical inputs -------------------------------------
        fo = find_one(s.src, "Freeze_out*.csv")
        if len(fo) != 1:
            sys.exit(f"expected one Freeze_out*.csv in {s.src}, found {len(fo)}")
        txt = normalise(fo[0], HDR_FREEZEOUT, args.fix, args.dry_run, actions, CANON_FREEZEOUT)
        if "[+HEADER ADDED]" in actions[-1]:
            header_fixed.append(s.name)
        if "[!!" in actions[-1]:
            warnings.append(s.name)
        emit(os.path.join(dstdir, CANON_FREEZEOUT), txt, args.dry_run, [], "")

        # THE FLUIDUM RULE: always the system baseline, never the shifted file.
        base_fl = baselines[s.key]
        txt = normalise(base_fl, HDR_FLUIDUM, args.fix, args.dry_run, actions, CANON_FLUIDUM)
        if not s.is_baseline:
            actions[-1] += "   <-- BASELINE (not this scenario's own)"
        emit(os.path.join(dstdir, CANON_FLUIDUM), txt, args.dry_run, [], "")

        for canon in (CANON_INITDATA, CANON_EXPERIMENT):
            p = os.path.join(s.src, canon)
            if not os.path.isfile(p):
                sys.exit(f"missing required {canon} in {s.src}")
            emit(os.path.join(dstdir, canon), read_text(p), args.dry_run, actions,
                 f"{canon} <- {canon}")

        # --- originals, for provenance --------------------------------------
        originals = []
        for pat in ("Exp_*.csv", "FluiduMData*.csv", "Freeze_out*.csv"):
            originals += find_one(s.src, pat)
        # also carry the baseline FluiduM original when it lives elsewhere
        if os.path.dirname(base_fl) != os.path.dirname(originals[0] if originals else ""):
            if base_fl not in originals:
                originals.append(base_fl)
        for p in originals:
            emit(os.path.join(dstdir, os.path.basename(p)), read_text(p),
                 args.dry_run, actions, f"orig: {os.path.basename(p)}")

        if not find_one(s.src, "Exp_*.csv"):
            actions.append("(no Exp_*.csv original in source -- only the derived "
                           "experimentaldata.csv)")

        print(f"[{s.name}]")
        for a in actions:
            print(f"    {a}")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 72)
    if header_fixed:
        print(f"HEADER ADDED to freezeout.csv for {len(header_fixed)} scenario(s):")
        for n in header_fixed:
            print(f"    {n}")
        print("  Their Freeze_out originals had no header line, so the C++ reader")
        print("  (has_header=true) was silently discarding the alpha=0 row. With the")
        print("  header present that row is now kept, so results from this tree will")
        print("  differ slightly from data/ for these scenarios. This is a FIX --")
        print("  pass --no-fix-headers to reproduce the old behaviour instead.")
    if warnings:
        print(f"WARNING: {len(warnings)} scenario(s) left without a freezeout header;")
        print("  their alpha=0 row will be discarded by the C++ reader.")
    if not header_fixed and not warnings:
        print("All freezeout files already carried a header; nothing to fix.")
    print("=" * 72)
    if args.dry_run:
        print("\nDRY RUN -- nothing was written. Re-run without --dry-run to build.")
    else:
        print(f"\nBuilt {len(scenarios)} scenarios under {args.dst}/")
        print(f"Validate with:  ./prepare_datanew.py --validate-only --dst {args.dst}"
              if False else
              f"Point evaluate.ipynb at e.g. parentdir = \"{args.dst}/{scenarios[0].name}\"")


if __name__ == "__main__":
    main()
