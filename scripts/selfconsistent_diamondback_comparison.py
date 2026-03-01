# %%
import os
import warnings
warnings.filterwarnings('ignore')

# os.environ["PYSYN_CDBS"] = os.path.join(os.environ['picaso_refdata'],'stellar_grids')

import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

import sys
sys.path.append(".")
from load_diamondback import find_rcb_diamondback, read_diamondback_optical_properties, diamondback_pt
# %%
# matching Mang et al. 2026 figure 8 two ways
# first, as he did, fully self-consistent within PICASO
# second, fixing the clouds at the Diamondback cloud solution
fsed = 8
grav = 316
cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
out_selfconsistent, out_fixed = {}, {}

mh = '+000'
CtoO = '100'

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
virga_dir = os.path.join("/Users", "adityasengupta",'virga', "refrind")
# %%
teffs = [900, 1300, 2400]
# %%
for teff in teffs:
    # a self-consistent run
    sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')
    cl_run = jdi.inputs(calculation="browndwarf", climate = True)
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
    cl_run.effective_temp(teff)

    opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

    nlevel = 91

    # start from Bobcat
    pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                                sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                                usecols=[1,2],unpack=True, skiprows = 1)

    # use the actual RCB from Diamondback so that we don't spend forever searching
    rcb_guess = find_rcb_diamondback(teff, grav, fsed)
    rfacv = 0.0
    cl_run.inputs_climate(temp_guess=temp_bobcat, pressure=pressure_bobcat, rcb_guess=rcb_guess, rfacv=rfacv)
    cl_run_fixed = deepcopy(cl_run)
    
    # set up self-consistent run
    cl_run.inputs['climate']['cloudy'] = 'selfconsistent'
    cl_run.virga(condensates=cloud_species, directory = virga_dir, mh=1,fsed=fsed, latent_heat=True)
    out_selfconsistent[teff] = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=False)

    # set up fixed run
    diamondback_optical_properties = read_diamondback_optical_properties(teff, grav, fsed)
    v_out = {}
    for (k, kv) in zip(['tau', 'g0', 'w0'], ["opd_per_layer", 'asymmetry', 'single_scattering']):
        v_out[kv] = np.array(diamondback_optical_properties[k]).reshape((90,196))
    cl_run_fixed.fix_virga_clouds(v_out)
    out_fixed[teff] = cl_run_fixed.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %%
for (c, teff) in zip(["xkcd:bright blue", "xkcd:turquoise", "xkcd:olive green"], teffs):
    pressure_grid, temp = diamondback_pt(teff, grav, fsed)
    plt.semilogy(temp, pressure_grid, label=f"Diamondback {teff}K", c=c, ls="--")
    plt.semilogy(out_selfconsistent[teff]["temperature"], out_selfconsistent[teff]["pressure"], label=f"PICASO cloudy {teff}K", ls="-", c=c)
    plt.semilogy(out_fixed[teff]["temperature"], out_fixed[teff]["pressure"], label=f"PICASO fixed {teff}K", ls=":", c=c)
plt.gca().invert_yaxis()
plt.xlim((300, 2600))
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.show()

# %%
