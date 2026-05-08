# Sun-like host star
# 0.02, 0.025, 0.04, 0.06, 0.13, 0.5 AU
# Tint = 300, 500, 800, 1100, 1400, 1700, 2000, and 2300
# log g = 5.0

import os
import sys
import warnings
warnings.filterwarnings('ignore')
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

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5

grav, teff, semi_major = sys.argv[1:5]
grav, teff, semi_major = int(grav), int(teff), float(semi_major)

picaso_path = os.path.dirname(picaso.__path__[0])
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids', 'bobcat')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# effective^4 = equilibrium^4 + intrinsic^4

coarse_temperatures = np.arange(200, 2401, 200)
coarse_gravs = np.array([10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160])

# cloudmode, grav, teff, semi_major = "cloudless", 1000, 600, 0.02
cloudmode = "cloudless"
print(f"effective temperature = {teff} K, grav = {grav} m/s/s, semimajor axis = {semi_major:.2f} au")
fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major:.2f}"

closest_temp_ongrid = coarse_temperatures[np.argmin(np.abs(teff - coarse_temperatures))]
closest_grav_ongrid = coarse_gravs[np.argmin(np.abs(grav - coarse_gravs))]

fname_stem_closest = f"irr_teff{closest_temp_ongrid}_grav{closest_grav_ongrid}_semimajor{semi_major:.2f}"
fname = os.path.join(picaso_path, "data", "both_irradiated_withinversion", f"{fname_stem}nc.h5")
fname_closest = os.path.join(picaso_path, "data", "both_irradiated_withinversion", f"{fname_stem_closest}nc.h5")
if os.path.exists(fname):
    print("Rerunning")

nstr_upper = None
with h5py.File(fname_closest) as f:
    pressure_grid = np.array(f["pressure"])
    temp_guess = np.array(f["temperature"])
    try:
        nstr_upper = f.attrs["nstr_upper_init"]
    except Exception:
        pass

# Fix the temperature structure from cloudless run
# Also fix the RCB from cloudless run
try:
    with h5py.File(fname_closest) as f:
        temp_for_virga = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]
except Exception:
    pass

# In case clouds make the RCB sink a bit, we want to allow this much
# This is unmotivated and we may find it should go even deeper
# However, having observed that the RCB tends to track the cloud base, I think it's fine
nstr_upper += 5

nstr_upper_init = nstr_upper
temp_guess_init = np.copy(temp_guess)
print(f"Starting at {nstr_upper = }")

cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
rfacv = 0.5

cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
with h5py.File(fname, "w") as f:
    f["pressure"] = pressure_grid
    f["temperature"] = out["temperature"]
    f.attrs["nstr_upper_init"] = nstr_upper_init
