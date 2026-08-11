# stuff that's getting unified here:
# cloudless and cloudy
# saving sparse and full outputs
# parallel and serial (tbh I think I should just fully switch to parallel)
# unirradiated and irradiated
# better handling of when runs have already been done

import os
import psutil
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import traceback
from time import sleep

import picaso
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga
import virga.justdoit as vj
import astropy.units as u
import astropy.constants as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
import h5py
import gc
from mpi4py import MPI
from collections import namedtuple

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5
from calculate_t10 import t10, regrid_initial_guess

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
virga_path = os.getenv('virga')

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.getenv('picaso_refdata')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
pressure_grid = np.logspace(-4, 3 + np.log10(3), nlevel)
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(__refdata__,'sonora_grids', 'bobcat', "structures_m+0.0")
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])

gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','C2H2','C2H4','C2H6','Na','K','PH3','FeH','SO2','H2S']
# effective^4 = equilibrium^4 + intrinsic^4


def fsed_str(fsed):
    if fsed == -1:
        return "nc"
    else:
        return "f" + str(fsed)

def semi_major_str(semi_major):
    if semi_major == -1:
        return "ns"
    else:
        return f"semimajor{semi_major:.2f}"

def fname_from_params(grav, tint, semi_major, fsed):
    fname_stem = f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_{fsed_str(fsed)}"
    return os.path.join(picaso_path, "data", "unified", f"{fname_stem}.h5")

def initial_guess(fname):
    with h5py.File(fname) as f:
        temp_guess = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]
        nstr_upper += 5
        nstr_upper = min(nstr_upper, 89)
        # In case clouds make the RCB sink a bit, we want to allow this much
        # This is unmotivated and we may find it should go even deeper
        # However, having observed that the RCB tends to track the cloud base, I think it's fine

    return temp_guess, nstr_upper

def run_byhand(grav, tint, semi_major, fsed, opacity_ck, guess_from, rank=-1):
    try:
        # if this is a cloudy run and the equivalent cloudless run doesn't exist, you gotta skip
        if fsed > 0:
            cloudmode = "fixed"
            fname_cloudless = fname_from_params(grav, tint, semi_major, -1)
            if not os.path.exists(fname_cloudless):
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] Skipping - cloudless model unavailable")
                return False
        else:
            cloudmode = "cloudless"
        
        # now, if we're here, we are either at a cloudless run, or a cloudy run with a known cloudless start
        fname = fname_from_params(grav, tint, semi_major, fsed)
        if os.path.exists(fname):
            os.remove(fname)

        temp_guess, nstr_upper = initial_guess(guess_from)
        nstr_upper_init = nstr_upper
        temp_guess_init = np.copy(temp_guess)
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] Starting at nstr_upper = {nstr_upper} on rank {rank}")

        calc_type = "planet" if semi_major > 0 else "browndwarf"
        cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(tint) # input effective temperature

        if semi_major > 0:
            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
            rfacv = 0.5
        else:
            rfacv = 0.0

        cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
        if fsed > 0:
            kz = np.ones_like(pressure_grid) * 1e10 # idk
            virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
            virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
            virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
            virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
            cl_run.fix_virga_clouds(virga_out)

        cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
        out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=False)
        
        with h5py.File(fname, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            t10_this = t10(pressure_grid, out["temperature"])
            f.attrs["t10"] = t10_this
            print(f"t10 at {grav, tint, semi_major, fsed} = {t10_this:.3f}")
            out_to_hdf5(out, f)
        
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✓ Saved to disk")
    
        return True
        
    except Exception as e:
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✗ Error: {e}")
        print(traceback.format_exc())
        return False

opacity_ck = jdi.opannection(ck_db=os.path.join(__refdata__, "climate_INPUTS", "661"),method='resortrebin',preload_gases=gases_fly)
grav, semimajor = 177, 0.5
for tint in [1200]:
    run_byhand(grav, tint, semimajor, -1, opacity_ck, f"/Users/adityasengupta/picaso/data/unified/unified_tint{tint-50}_grav{grav}_{semi_major_str(semimajor)}_nc.h5")