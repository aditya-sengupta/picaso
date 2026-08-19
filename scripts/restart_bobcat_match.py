# unifying all the scripts was stupid
# the purpose of this script is _only_ to generate a T10 grid for Bobcat matching: no-star, no-cloud runs
# and to implement the whole "moving Pmax up" algorithm

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
calc_type = "browndwarf"


picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.getenv('picaso_refdata')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(__refdata__,'sonora_grids', 'bobcat', "structures_m+0.0")
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])
# effective^4 = equilibrium^4 + intrinsic^4
tints = np.arange(100, 2401, 50)

gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]

def fname_from_params(grav, tint):
    fname_stem = f"unified_tint{tint}_grav{grav}_ns_nc"
    return os.path.join(picaso_path, "data", "unified_restart", f"{fname_stem}.h5")

def initial_guess(fname):
    with h5py.File(fname) as f:
        temp_guess = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]

    return temp_guess, nstr_upper

def generate_tasks():
    for grav in gravs:
        for tint in tints:
            yield (grav, tint)

def run(grav, tint, opacity_ck, rank=-1):
    try:
        fname = fname_from_params(grav, tint)
        teff_bobcat, grav_bobcat = bobcat_temps[np.argmin(np.abs(bobcat_temps - tint))], bobcat_gravs[np.argmin(np.abs(bobcat_gravs - grav))]
        pressure_bobcat, temp_bobcat = np.loadtxt(os.path.join(sonora_profile_db,f"t{teff_bobcat}g{grav_bobcat}nc_m0.0.dat"), usecols=[1,2],unpack=True, skiprows = 1)
        max_pressure = np.max(pressure_bobcat)
        acceptable = False
        while not acceptable:
            pressure_grid = np.logspace(np.log10(np.min(pressure_bobcat)), np.log10(max_pressure), nlevel)
            temp_guess = regrid_initial_guess(pressure_bobcat, temp_bobcat, pressure_grid)
            indices_below_bobcat = np.where(pressure_grid > np.max(pressure_bobcat))[0]
            if len(indices_below_bobcat) > 1:
                nstr_upper = min(89, np.min(indices_below_bobcat))
            else:
                nstr_upper = 89
            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
            print(f"[{grav}, {tint}] Starting at nstr_upper = {nstr_upper} on rank {rank}")

            cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(tint) # input effective temperature
            rfacv = 0.0
            cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
            cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
            out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=True)
            acceptable = np.max(out["temperature"]) < 5100.0
            if not acceptable:
                max_pressure = max_pressure / 5
                print(f"[{grav}, {tint}] too hot, reducing max pressure")
        
        with h5py.File(fname, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            t10_this = t10(pressure_grid, out["temperature"])
            f.attrs["t10"] = t10_this
            print(f"t10 at {grav, tint} = {t10_this:.3f}")
            out_to_hdf5(out, f)
        
        print(f"[{grav}, {tint}] ✓ Saved to disk")
        
        del cl_run, out, temp_guess
        if 'temp_guess_init' in locals():
            del temp_guess_init
        if 'temp_bobcat' in locals():
            del temp_bobcat
        if 'pressure_bobcat' in locals():
            del pressure_bobcat
                
        return True
        
    except Exception as e:
        print(f"[{grav}, {tint}] ✗ Error: {e}")
        print(traceback.format_exc())
        return False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

opacity_ck = None
if rank == 0:
    print(f"Starting run")

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
comm.Barrier()

local_completed = 0
local_failed = 0

while True:
    for i, (grav, tint) in enumerate(generate_tasks()):
        if i % size == rank and opacity_ck is not None:
            result = run(grav, tint, opacity_ck, rank)
            if result:
                local_completed += 1
            else:
                local_failed += 1

    completed = comm.allreduce(local_completed, op=MPI.SUM)
    failed = comm.allreduce(local_failed, op=MPI.SUM)

    if rank == 0:
        print(f"\n=== Summary ===")
        print(f"Total Completed: {completed}")
        print(f"Total Failed: {failed}")
