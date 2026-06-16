#%%

import os
import subprocess
import glob
import datetime  
import numpy as np
import pandas as pd
import re

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker
# get_ipython().run_line_magic("matplotlib","qt")
%matplotlib inline
matplotlib.rcParams['mathtext.default'] = 'rm'

TICKLABELSIZE=20
FIGSIZE = (7,7)
AXISLABELSIZE = 20
LINEWIDTH = 2
MARKERSIZE = 5

SPEC_XLABEL = r"$p_T\ [GeV]$"
SPEC_YLABEL = r"$(2\pi p_T)^{-1}dN_{\pi^++\pi^-}/(dp_Td\eta_p)\ [GeV^{-2}]$"

MPI = 0.14                          # pion mass = 0.14 GeV
MSIGMA_MIN, MSIGMA_MAX = 2*MPI, 2   # sigma mass, spectral function considered up to 2 GeV
NMASSSAMPLES = 300                  # 300 masses sampled
# parentdir = "data/examplescenario"
parentdir = "data/AuAu200_cc_(0,5)"
# parentdir = "data/PbPb276_cc_(0,5)"
# parentdir = "data/XeXe544_cc_(0,5)"


fig, ax = plt.subplots(figsize=(7,7))
df_exp = pd.read_csv(parentdir+"/experimentaldata.csv",comment="#")
pTs_exp, spec_exp, spec_exp_err = df_exp.to_numpy().T

df_fluid = pd.read_csv(parentdir+"/fluidumdata.csv",comment="#")
pTs_fluid, spec_fluid = df_fluid.to_numpy().T

ax.scatter(pTs_exp, spec_exp,label="experiment")
ax.scatter(pTs_fluid, spec_fluid,label=r"Fluid$\mathdefault{u}$m")

ax.set_yscale("log")

ax.set_title(parentdir.replace("data/",""))
ax.set_xlabel(SPEC_XLABEL, fontsize=AXISLABELSIZE)
ax.set_ylabel(SPEC_YLABEL, fontsize=AXISLABELSIZE)

ax.tick_params(axis="both",labelsize=TICKLABELSIZE)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.grid()
ax.legend()

fig.tight_layout()
fig.show()


#%%
####
####    PARAMETERS FOR PRIMARY (SIGMA) SPECTRUM COMPUTATION
####

dccinitdata = glob.glob(parentdir+"/initdata.csv")[0]
freezeoutdata = glob.glob(parentdir+"/freezeout.csv")[0]
masses = np.linspace(MSIGMA_MIN, MSIGMA_MAX,NMASSSAMPLES+2)[1:-1]
qTmax = 4
NqT = 400
epsrel = 1e-5
iterations=10000
primaryspecfolder = "primaryspec"

#%%
####
####    COMPUTE PRIMARY SPECTRA
####

# prevent accidental execution of notebook cell
confirm = input("start computation? (y/n):").strip().lower()
if confirm != 'y':
    raise RuntimeError("aborted")

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

SPEC_XLABEL = r"$q_T\ [GeV]$"
SPEC_YLABEL = r"$(2\pi q_T)^{-1}dN_{coherent}/(dq_Td\eta_q)\ [GeV^{-2}]$"

fig, ax = plt.subplots(figsize=(7,7))
lines = []

specfiles = sorted(glob.glob(parentdir+"/"+primaryspecfolder+"/*/*spectr.txt"))

masses = np.zeros(len(specfiles))
for (i,file) in enumerate(specfiles):
    with open(file) as f:
        mylines = f.readlines()
        masses[i] = float(mylines[3].replace("# particle mass:\t",""))

for (i, file) in enumerate(specfiles):
    df = pd.read_csv(file,comment="#")

    pTs, spec = df.to_numpy().T
    lines.append(np.column_stack((pTs, spec)))
    
linecol = LineCollection(lines,array=masses,cmap=CMAP,lw=LINEWIDTH)
ax.add_collection(linecol)
ax.set_yscale("log")
ax.set_xlim(0,qTmax)
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

fig.savefig(parentdir+"/out_primespecs.png")
fig.tight_layout()
fig.show()

#%%
####
####    PARAMETERS FOR DECAY (PION) SPECTRUM COMPUTATION
####

decayspecfolder = "decayspec"
pTmax = 1
NpT = 50
epsrel=1e-5
iterations = 100000
primespecfiles = sorted(glob.glob(parentdir+"/"+primaryspecfolder+"/*/*spectr.txt"))
B=1
Q=1

#%%
####
####    COMPUTE DECAY SPECTRA
####

# prevent accidental execution of notebook cell
confirm = input("start computation? (y/n):").strip().lower()
if confirm != 'y':
    raise RuntimeError("aborted")

masses = np.zeros(len(primespecfiles))
for (i,file) in enumerate(primespecfiles):
    with open(file) as f:
        mylines = f.readlines()
        masses[i] = float(mylines[3].replace("# particle mass:\t",""))

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

#%%
####
####    VISUALIZE DECAY SPECTRA
####

SCALE	= 0.06
B		= 1./2.
AMP		= 0.1 * np.sqrt(SCALE/B)
Mpole = 0.449
Gpole = 2* 0.275
YMIN = 2e1
YMAX = 5e3
print("amplitude:",AMP)

### \begin{equation}
###     \frac{1}{2\pi p_{\text{T}}}\frac{\dt N}{\dt p_{\text{T}}\dt\eta_p}\Big\vert_{\pi^+}\sim B\,\vert\underbrace{\mathcal{A}_\sigma}_{\text{condensate amplitude}}\vert^2=\underbrace{\tilde{B}}_{\to 1}\ \vert\underbrace{\tilde{\mathcal{A}}_\sigma}_{\to\SI{0.1}{\GeV}}\vert^2\cdot\text{scale}\quad\implies\quad\mathcal{A}_\sigma=\tilde{\mathcal{A}}_\sigma\sqrt{\text{scale}/B}
### \end{equation}

### B... branching ratio, A... sigma condensate amplitude
### in computation we instead use for simplicity
###     B~ = 1
###     A~ = 0.1 GeV
### and adjust the scale in the plotting script.
###
### ->  spec \propto B * |A|^2 = B~ * |A~|^2 * scale        =>      A = A~ * sqrt(scale/B)
###

SAVEPLOTS = True
# SAVEPLOTS = False
PUTINSET = False
# PUTINSET = True
ONLYLABELEXPERIMENT = False
# ONLYLABELEXPERIMENT = True
TICKLABELSIZE=20
FIGSIZE = (7,7)
AXISLABELSIZE = 23.5
LINEWIDTH = 2
MARKERSIZE = 5

m = re.match(r"data/([a-zA-Z]*)(\d*)",parentdir)
TITLE = "???"
if(m.group(1) == "PbPb" or m.group(1) == "XeXe"):
	TITLE = "%s@%.2f"%(m.group(1),float(m.group(2))*1e-2)+"$\ TeV$"
if(m.group(1) == "AuAu"):
	TITLE = "%s@%.0f"%(m.group(1),float(m.group(2)))+"$\ GeV$"



CMAP = LinearSegmentedColormap.from_list("custom", ["blue","red"])
CMAP_LBWH = [0.025, 0.025, 0.05, 0.45]
CMAP_LABELSIZE = 15
CMAP_TICKSIZE = 15
LC_LABEL = r"$m\ [GeV]$"

SPEC_XLABEL = r"$p_T\ [GeV]$"
SPEC_YLABEL = r"$(2\pi p_T)^{-1}dN/(dp_Tdy)\ [GeV^{-2}]$"
LEGEND= r"$\sigma_{PRCC}\to\pi^+\,\pi^-$"

fig_dens, ax_dens = plt.subplots(figsize=(7*np.sqrt(1.618),7/np.sqrt(1.618)))
fig, ax = plt.subplots(figsize=(7,7))
fig_full, ax_full = plt.subplots(figsize=(7,7))

lines = []
pT_full, spec_full = np.zeros(shape=(2,1))

decayspecfolder="decayspec"
decayfiles = sorted(glob.glob(parentdir+"/"+decayspecfolder+"/*/*decayspec.txt"))

### COMPUTE SPECTRAL FUNCTION AND WEIGHTS
msigma = np.sqrt(1/4 * (16 * MPI**2 + 
                        np.sqrt(16 * Gpole**2 * Mpole**2 + 
                                (-16 * MPI**2 - Gpole**2 + 4*Mpole**2)**2)))
Gam = np.sqrt(1/2 * (16 * MPI**2 + Gpole**2 - 
     4*  Mpole**2 + np.sqrt(16*Gpole**2 * Mpole**2 + (-16 * MPI**2 - Gpole**2 + 4*Mpole**2)**2)))

def Delta(s):
    return 1/(s-msigma**2+1j*Gam*np.sqrt(s-(2*MPI)**2))

def S(k):
    return -1/np.pi*np.imag(Delta(k**2))

masses = np.zeros(len(decayfiles))
for (i,file) in enumerate(decayfiles):
    with open(file) as f:
        mylines = f.readlines()
        masses[i] = float(mylines[3].replace("# ma:\t",""))

weights = 2*masses*S(masses)*np.ptp(masses)/len(masses)
weights /= np.sum(weights)
###


### SUM OVER INDIVIDUAL MASSES FOR THE FINAL DECAY SPECTRUM
for (i, file) in enumerate(decayfiles):
    df = pd.read_csv(file,comment="#")

    pTs, spec = df.to_numpy().T
    spec =  2 * SCALE * spec ### factor 2 for pi+ and pi^-
    lines.append(np.column_stack((pTs, spec)))

    ### avoid gaps due to failed integral evaluations that lead to inf
    exclude = np.isinf(spec)
    idcs = np.where(1-exclude)
    spec = np.exp(np.poly1d(np.polyfit(pTs[idcs],np.log(spec[idcs]),10))(pTs))


    pT_full = pTs
    spec_full = spec_full + weights[i] * spec
    
extra_spec = np.copy(spec_full)


# Breit Wigner test
fullmasses = np.linspace(0,np.max(masses),100)
MBW = np.sqrt(4*Mpole**2-Gpole**2)/2
GBW = 2*np.sqrt(Mpole**2/(4*Mpole**2-Gpole**2))
BW = 1/np.pi * np.imag(1/(-fullmasses**2+MBW**2-1j*MBW*GBW))

### SPECTRAL DENSITY
# ax_dens.plot([0.28,*masses],[0,*S(masses)],lw=LINEWIDTH,c="b",label=r"$M_{\text{pole}}=$"+str(Mpole)+r"$GeV$"+"\n"+r"$\Gamma_{\text{pole}}=$"+str(Gpole)+r"$GeV$")
ax_dens.plot([0.28,*masses],[0,*S(masses)],lw=LINEWIDTH,c="b",label=r"Sill parametrization")
# ax_dens.plot(fullmasses,BW,lw=LINEWIDTH,c="r",ls=":",label=r"Breit-Wigner")
ax_dens.set_xlabel(r"$\mu\ [GeV]$", fontsize=AXISLABELSIZE)
ax_dens.set_ylabel(r"$\rho(\mu^2)\ [GeV^{-2}]$", fontsize=AXISLABELSIZE)
ax_dens.set_xlim(0,ax_dens.get_xlim()[1])
ax_dens.set_xticks([0,0.5,1,1.5,2])

ax_dens.tick_params(axis="both",labelsize=TICKLABELSIZE)
ax_dens.xaxis.set_ticks_position("bottom")
ax_dens.yaxis.set_ticks_position("left")
ax_dens.grid(False)
ax_dens.legend(fontsize=AXISLABELSIZE,loc=1)
lims = ax_dens.get_ylim()
ax_dens.vlines(0.28,ymin=0,ymax=1.2,lw=2,ls="-.",colors="k")
ax_dens.text(0.12,0.7,r"$\mu=2m_\pi$",fontsize=AXISLABELSIZE,rotation=90)
ax_dens.set_ylim(0.0001,1.2)
ax_dens.set_xlim(0,2)

fig_dens.tight_layout()
fig_dens.show()

### INDIVIDUAL DECAYSPEC PLOT
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

ax.set_title(parentdir)
fig.tight_layout()
fig.show()
###

### FULLSPEC PLOT
# PROCESS EXPERIMENTAL AND FLUIDUM DATA
df_exp = pd.read_csv(parentdir+"/experimentaldata.csv",comment="#")
pTs_exp, spec_exp, spec_exp_err = df_exp.to_numpy().T

df_fluid = pd.read_csv(parentdir+"/fluidumdata.csv",comment="#")
pTs_fluid, spec_fluid = df_fluid.to_numpy().T

def mylogpolyfit(datax, datay):
    popt = np.polyfit(datax,np.log(datay),10)
    return np.poly1d(popt)

spec_fluidum_loginterp = mylogpolyfit(pTs_fluid, spec_fluid)
spec_fluidum_interp = np.exp(spec_fluidum_loginterp(pT_full))
spec_full += spec_fluidum_interp

# PLOT
COL = (1,0,0)
ax_full.plot(pT_full, spec_fluidum_interp,lw=LINEWIDTH,c="b",label=r"incoherent source"+"\n"+r"(Lu et al., 2025)")
ax_full.fill_between(pT_full,spec_fluidum_interp,spec_full,facecolor=(*COL,0.2),edgecolor=COL,ls="--",lw=LINEWIDTH,label=LEGEND)
# ax_full.plot([],[],lw=0,label=r"$\Delta\sigma_{\text{coherent}}=$"+"%.0f"%(1e3*AMP)+r"$\,MeV$"+"\n"+r"$M_{\sigma}=$"+"%.0f"%(1e3*Mpole)+r"$\,MeV$"+"\n"+r"$\Gamma_{\sigma}=$"+"%.0f"%(1e3*Gpole)+r"$\,MeV$")
ax_full.plot([],[],lw=0,label=r"$\Delta\sigma_{\text{coherent}}=$"+"%.0f"%(1e3*AMP)+r"$\,MeV$")
ax_full.errorbar(pTs_exp, spec_exp, spec_exp_err,label=TITLE,c="b",fmt="o",markersize=MARKERSIZE,lw=LINEWIDTH)

ax_full.set_xlim(0,pTmax)
ax_full.set_ylim(ax_full.get_ylim()[0],1.2*ax_full.get_ylim()[1])
ax_full.set_xlabel(SPEC_XLABEL, fontsize=AXISLABELSIZE)

ax_full.tick_params(axis="both",labelsize=TICKLABELSIZE)
ax_full.xaxis.set_ticks_position("bottom")
ax_full.yaxis.set_ticks_position("left")
ax_full.grid(False)

### UNCOMMENT HERE FOR LOG SCALE
# ax_full.set_yscale("log")
# ax_full.set_ylim(YMIN,YMAX)
# ax_full.set_ylim(3e0,2e3)
# locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8))
# ax_full.yaxis.set_minor_locator(locmin)
# ax_full.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(base=10,labelOnlyBase=False,minor_thresholds=(5,2.5))) # means: the data spans ~5 decades and we want to see all minor ticks if we zoom in on a region of 2.5 decades


ax_full.legend(fontsize=AXISLABELSIZE)
if(ONLYLABELEXPERIMENT):
	hand, labl = ax_full.get_legend_handles_labels()
	ax_full.legend([hand[-1]],[labl[-1]],fontsize=AXISLABELSIZE)
else:
    hand, labl = ax_full.get_legend_handles_labels()
    order = [3,0,1,2]
    ax_full.legend([hand[i] for i in order],[labl[i] for i in order],fontsize=AXISLABELSIZE)
    ax_full.set_ylabel(SPEC_YLABEL, fontsize=AXISLABELSIZE)
    ax_full.text(0.05,0,r"$\pi^++\pi^-$",fontsize=1.5*AXISLABELSIZE)
    
if(PUTINSET):
	hand, labl = ax_full.get_legend_handles_labels()
	ax_full.legend([hand[-1]],[labl[-1]],fontsize=AXISLABELSIZE,loc=3)

if(PUTINSET):
	s, p = 0.55, 0.05 # size and padding
	ax_in = ax_full.inset_axes([1-s-p,1-s-p,s,s])
	ax_in.plot([0.28,*masses],[0,*weights],lw=LINEWIDTH,c="b",label=r"$M_{\text{pole}}=$"+str(Mpole)+r"$GeV$"+"\n"+r"$\Gamma_{\text{pole}}=$"+str(Gpole)+r"$GeV$")
	### set labels
	scalefonts = 0.65
	ax_in.set_xlabel(r"$\mu\ [GeV]$", fontsize=scalefonts*AXISLABELSIZE)
	ax_in.set_ylabel(r"$2\mu \rho(\mu)$", fontsize=scalefonts*AXISLABELSIZE)
	ax_in.tick_params(axis="both",labelsize=scalefonts*TICKLABELSIZE)
	### adjust axis
	ax_in.set_ylim(0,0.01)
	ax_in.set_yticklabels([f"{t*100:.1f}" for t in ax_in.get_yticks()])
	ax_in.text(0, 1.02, "×1e−2", transform=ax_in.transAxes,fontsize=scalefonts*AXISLABELSIZE)
	###
	ax_in.xaxis.set_ticks_position("bottom")
	ax_in.yaxis.set_ticks_position("left")
	ax_in.grid(False)

fig_full.tight_layout()
fig_full.show()

if(SAVEPLOTS):
	expShort = re.sub(r"data/([a-zA-Z]{4}\d{3})_cc_\(0,5\)",r"\1",parentdir)
	fig.savefig(parentdir+"/fig_%s_massspecs.pdf"%(expShort))
	fig_full.savefig(parentdir+"/fig_%s_fullspec.pdf"%(expShort))
	fig_dens.savefig(parentdir+"/fig_%s_specdense.pdf"%(expShort))

#%%
### How many extra pions
dp = pT_full[1]-pT_full[0]
print("# Pions/dy in the interval [0,1] GeV")
print("Fluidum:", np.sum(dp*2*np.pi*pT_full*spec_fluidum_interp))
print("PRCC:", np.sum(dp*2*np.pi*pT_full*extra_spec))

"""
AuAu:
	Fluidum:	217.24753313176097
	PRCC:		70.24826934279795
PbPb: 
	Fluidum:	523.2242575712232
	PRCC:		224.69706127763953
XeXe:
	Fluidum:	733.7669341188322
    PRCC:		338.21269795049756
"""

###
# %%
