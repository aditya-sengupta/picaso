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

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
virga_path = "/home/adityars/virga/"

grav, teff, semi_major, fsed = sys.argv[1:5]
grav, teff, semi_major, fsed = int(grav), int(teff), float(semi_major), int(fsed)

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
# cloudmode, grav, teff, semi_major = "cloudless", 1000, 600, 0.02

cloudmode = "fixed"
print(f"effective temperature = {teff} K, grav = {grav} m/s/s, semimajor axis = {semi_major:.2f} au")
fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major:.2f}"
fname_stem_cloudy = fname_stem + f"fsed{fsed}"
# unlike the cloudless case, each of these runs has to start from the exact same cloudless point
fname_cloudless = os.path.join(picaso_path, "data", "both_irradiated_withinversion", f"{fname_stem}nc.h5")
fname = os.path.join(picaso_path, "data", "both_irradiated_withinversion", f"{fname_stem_cloudy}.h5")
if os.path.exists(fname):
    print("Exiting because run already complete")
    sys.exit()

nstr_upper = None
with h5py.File(fname_cloudless) as f:
    pressure_grid = np.array(f["pressure"])
    temp_guess = np.array(f["temperature"])
    temp_for_virga = temp_guess
    nstr_upper = f.attrs["nstr_upper_init"]

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
kz = np.ones_like(pressure_grid) * 1e10 # idk
virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_for_virga, 'kz': kz}), kz_min=1e5, latent_heat=True)
virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
cl_run.fix_virga_clouds(virga_out)

out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
with h5py.File(fname, "w") as f:
    f["pressure"] = pressure_grid
    f["temperature"] = out["temperature"]
    for k in virga_out:
        f[k] = virga_out[k]
    f.attrs["nstr_upper_init"] = nstr_upper_init
