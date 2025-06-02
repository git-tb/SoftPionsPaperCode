#%%

import os
import subprocess
import glob
import datetime
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
get_ipython().run_line_magic("matplotlib","qt")
matplotlib.rcParams['mathtext.default'] = 'rm'

MPI = 0.14                          # pion mass = 0.14 GeV
MSIGMA_MIN, MSIGMA_MAX = 2*MPI, 2   # sigma mass, spectral function considered up to 2 GeV
NMASSSAMPLES = 200                      # 200 masses sampled

#%%
####
####    PARAMETERS FOR PRIMARY (SIGMA) SPECTRUM COMPUTATION
####

parentdir = "data/examplescenario"
dccinitdata = glob.glob(parentdir+"/initdata.csv")[0]
freezeoutdata = glob.glob(parentdir+"/freezeout.csv")[0]
masses = np.linspace(MSIGMA_MIN, MSIGMA_MAX,NMASSSAMPLES+2)[1:-1]
qTmax = 3
NqT = 300
epsrel = 1e-5
iterations=1000
primaryspecfolder = "primaryspec"

#%%
####
####    COMPUTE PRIMARY SPECTRA
####

for (i,m) in enumerate(masses):
    foldername = primaryspecfolder+"/spec_{:%Y%m%d_%H%M%S}_".format(datetime.datetime.now())+str(i).zfill(int(np.ceil(np.log10(len(masses)))))
    print("save to", foldername)
    result = subprocess.run(args=[
        "./bin/specV2",
        "--m=%f"%(masses[i]),
        "--pTmax=%f"%(qTmax),
        "--NpT=%d"%(NqT),
        "--epsabs=0",
        "--epsrel=%f"%(epsrel),
        "--iter=%d"%(iterations),
        "--parentdir=%s"%(parentdir),
        "--initpath=%s"%(dccinitdata),
        "--foldername=%s"%(foldername),
        "--freezeoutpath=%s"%(freezeoutdata)
    ])
    print(result)

#%%
####
####    VISUALIZE PRIMARY SPECTRA
####

TICKLABELSIZE=20
FIGSIZE = (7,7)
AXISLABELSIZE = 20
LINEWIDTH = 2

CMAP = LinearSegmentedColormap.from_list("custom", ["blue","red"])
CMAP_LBWH = [0.025, 0.025, 0.05, 0.45]
CMAP_LABELSIZE = 15
CMAP_TICKSIZE = 15
LC_LABEL = r"$m\ [GeV]$"

SPEC_XLABEL = r"$p_T\ [GeV]$"
SPEC_YLABEL = r"$(2\pi p_T)^{-1}dN_{coherent}/(dp_Td\eta_p)\ [GeV^{-2}]$"

fig, ax = plt.subplots(figsize=(7,7))
lines = []

specfiles = sorted(glob.glob(parentdir+"/"+primaryspecfolder+"/*/*spectr.txt"))
for (i, file) in enumerate(specfiles):
    df = pd.read_csv(file,comment="#")

    pTs, spec = df.to_numpy().T
    lines.append(np.column_stack((pTs, spec)))
    
linecol = LineCollection(lines,array=masses,cmap=CMAP,lw=LINEWIDTH)
ax.add_collection(linecol)
ax.set_yscale("log")
ax.set_xlim(0,pTmax)
ax.set_xlabel(SPEC_XLABEL, fontsize=AXISLABELSIZE)
ax.set_ylabel(SPEC_YLABEL, fontsize=AXISLABELSIZE)

ax.tick_params(axis="both",labelsize=TICKLABELSIZE)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.grid(False)

cax = ax.inset_axes(CMAP_LBWH)
cbar = fig.colorbar(linecol, cax=cax)
cbar.set_label(LC_LABEL, fontsize=CMAP_LABELSIZE)
cbar.ax.tick_params(labelsize=CMAP_TICKSIZE)

fig.tight_layout()
fig.show()

#%%
####
####    COMPUTE DECAY SPECTRA
####

decayspecfolder = "decayspec"
pTmax = 1
NpT = 100
epsrel=1e-5
iterations = 10000
primespecfiles = sorted(glob.glob(parentdir+"/"+primaryspecfolder+"/*/*spectr.txt"))
B=1
Q=1

for (i,MSIGMA) in enumerate(masses):
    foldername = decayspecfolder+"/decay_{:%Y%m%d_%H%M%S}_".format(datetime.datetime.now())+str(i).zfill(int(np.ceil(np.log10(len(masses)))))
    print("save to", foldername)
    result = subprocess.run(args=[
        "./bin/decayV2",
        "--ma=%f"%(MSIGMA),
        "--mb=%f"%(MPI),
        "--mc=%f"%(MPI),
        "--pTmax=%f"%(pTmax),
        "--NpT=%d"%(NpT),
        "--epsabs=0",
        "--epsrel=%f"%(epsrel),
        "--iter=%d"%(iterations),
        "--primespecpath=%s"%(primespecfiles[i]),
        "--parentdir=%s"%(parentdir),
        "--foldername=%s"%(foldername),
        "--B=%f"%(B),
        "--Q=%f"%(Q)
    ])
    print(result)