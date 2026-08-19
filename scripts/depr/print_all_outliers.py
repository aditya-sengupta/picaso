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

step = 50
tints = np.arange(100 + step, 2391, step)
gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
fseds = [-1]
semi_majors = [-1, 0.5, 0.13, 0.04, 0.02]

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
        # In case clouds make the RCB sink a bit, we want to allow this much
        # This is unmotivated and we may find it should go even deeper
        # However, having observed that the RCB tends to track the cloud base, I think it's fine

    return temp_guess, nstr_upper

def outlier_mag(grav, tint, semi_major, fsed):
    fname = fname_from_params(grav, tint, semi_major, fsed)
    fname_lower = fname_from_params(grav, tint - step, semi_major, fsed)
    if os.path.exists(fname) and os.path.exists(fname_lower):
        try:
            t10_lower, t10_current, outlier_magnitude = None, None, 0
            above_lower = True
            with h5py.File(fname) as f:
                t10_current = f.attrs["t10"]

            with h5py.File(fname_lower) as f:
                t10_lower = f.attrs["t10"]
                return max(t10_lower - t10_current, 0)
        except Exception as e:
            print(e)
            os.remove(fname)
    return 10_000

for grav in gravs:
    for tint in tints:
        for semi_major in semi_majors:
            for fsed in fseds:
                if os.path.exists(fname_from_params(grav, tint, semi_major, fsed)): 
                    outlier_magnitude = outlier_mag(grav, tint, semi_major, fsed)
                    if outlier_magnitude > 0:
                        print(grav, tint, semi_major, fsed, f": magnitude {outlier_magnitude:.3f}")
