# unifying all the scripts was stupid
# this starts from the no-star no-cloud runs in restart_bobcat_match.py
# and adds in irradiation

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

cloudmode = "cloudless"
calc_type = "planet"

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.getenv('picaso_refdata')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(__refdata__,'sonora_grids', 'bobcat', "structures_m+0.0")

gravs = [10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160]
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

def semi_major_str(semi_major):
    if semi_major == -1:
        return "ns"
    else:
        return f"semimajor{semi_major:.2f}"

def fname_from_params(grav, tint, semi_major):
    fname_stem = f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_nc"
    return os.path.join(picaso_path, "data", "unified_restart", f"{fname_stem}.h5")

def initial_guess(fname):
    with h5py.File(fname) as f:
        pressure_guess = np.array(f["pressure"])
        temp_guess = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]

    nstr_upper = 89
    return pressure_guess, temp_guess, nstr_upper

def run(grav, tint, semi_major, pressure_start, temperature_start, nstr_upper):
    fname = fname_from_params(grav, tint, semi_major)
    try:
        fname_start = fname_from_params(grav, tint, -1)
        max_pressure = np.max(pressure_start)
        acceptable = False
        while not acceptable:
            pressure_grid = np.logspace(np.log10(np.min(pressure_start)), np.log10(max_pressure), nlevel)
            temp_guess = regrid_initial_guess(pressure_start, temperature_start, pressure_grid)
            # rcb_pressure = pressure_start[nstr_upper]
            # nstr_upper = min(89, np.argmin(np.abs(pressure_grid - rcb_pressure)) + 1) # account for the regrid in picking nstr_upper
            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
            print(f"[{grav}, {tint}, {semi_major}] Starting at nstr_upper = {nstr_upper}")

            cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(tint) # input effective temperature
            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
            rfacv = 0.5
            cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
            cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
            out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=True)
            acceptable = np.max(out["temperature"]) < 5100.0
            if not acceptable:
                max_pressure = max_pressure / 2
                print(f"[{grav}, {tint}, {semi_major}] too hot, reducing max pressure")
        
            with h5py.File(fname, "w") as f:
                f["temp_guess"] = temp_guess_init
                f.attrs["nstr_upper_init"] = nstr_upper_init
                f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
                t10_this = t10(pressure_grid, out["temperature"])
                f.attrs["t10"] = t10_this
                print(f"t10 at {grav, tint, semi_major} = {t10_this:.3f}")
                out_to_hdf5(out, f)
        
        print(f"[{grav}, {tint}, {semi_major}] ✓ Saved to disk")
        
        del cl_run, out, temp_guess
        if 'temp_guess_init' in locals():
            del temp_guess_init
        if 'temperature_start' in locals():
            del temperature_start
        if 'pressure_start' in locals():
            del pressure_start
                
        return True
        
    except Exception as e:
        print(f"[{grav}, {tint}, {semi_major}] ✗ Error: {e}")
        print(traceback.format_exc())
        return False

pressure_start, temperature_start, nstr_upper_init = initial_guess(fname_from_params(31, 300, 0.04))
run(31, 300, 0.5, pressure_start, temperature_start, nstr_upper_init)