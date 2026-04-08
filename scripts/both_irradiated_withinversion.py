# Sun-like host star
# 0.02, 0.025, 0.04, 0.06, 0.13, 0.5 AU
# Tint = 300, 500, 800, 1100, 1400, 1700, 2000, and 2300
# log g = 5.0

import pickle as pkl
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
cloud_colors = ['#CC5555', '#3BA39C', '#CCB84D', '#FF8C00']

virga_path = "/home/adityars/virga/"
#virga_path = os.path.dirname(virga.__path__[0])

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5

grav, teff, semi_major, fsed = sys.argv[1:5]
grav, teff, semi_major, fsed = int(grav), int(teff), float(semi_major), int(fsed)
# I'm going to pretend fsed=0 is cloudless with TiO/VO
# and fsed=-1 is cloudless with no TiO/VO

picaso_path = os.path.dirname(picaso.__path__[0])
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids', 'bobcat')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
if fsed == -1:
    ck_stem += "_NoTiOVO.hdf5"
else:
    ck_stem += ".hdf5"
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# effective^4 = equilibrium^4 + intrinsic^4

# cloudmode, grav, teff, semi_major = "cloudless", 1000, 600, 0.02
cloudmode = "fixed" if fsed > 0 else "cloudless"
#for grav in np.array([17, 31, 100, 316, 1000, 3160]):
#    for teff in np.arange(1000, 2401, 200):
#        for semi_major in np.array([0.02, 0.04, 0.13, 0.5]):
print(f"effective temperature = {teff} K, grav = {grav} m/s/s, cloud mode = {cloudmode}, semimajor axis = {semi_major:.2f} au, fsed = {fsed}")
fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major:.2f}"
fname_cloudless_stem = fname_stem + "nc"
if fsed > 0:
    fname_stem += f"fsed{fsed}"
elif fsed == 0:
    fname_stem += "nc"
elif fsed == -1:
    fname_stem += "nc_noTiOVO"
fname = os.path.join(picaso_path, "data", "both_irradiated_withinversion", f"{fname_stem}.h5")
fname_withoutinversion = os.path.join(picaso_path, "data", "both_irradiated", f"{fname_stem}.h5")
fname_cloudless = os.path.join(picaso_path, "data", "both_irradiated", f"{fname_cloudless_stem}.h5")

if os.path.exists(fname):
    sys.exit()

# This initial guess matters a lot
# For at least one case, (g = 100m/s^2, teff = 1000K, semimajor = 0.02 au, fsed = 2)
# Kazumasa's grid (no TiO/VO) gives us all four clouds
# My grid (with TiO/VO) only gives us one
# I don't want to work off of non PICASO runs
# so I think I should do a set of cloudless non-TiO/VO 
# actually no, it is physically real that we only have one cloud come up

# Guess the temperature structure from the last time we did this
# If the last run was based on Kazumasa's grid, with no TiO/VO, this is going to move around a bunch
# If it was based on the previous cloudless run, it'll stay where it is
nstr_upper = None
with h5py.File(fname_withoutinversion) as f:
    pressure_grid = np.array(f["pressure"])
    temp_guess = np.array(f["temperature"])

# Fix the temperature structure from cloudless run
# Also fix the RCB from cloudless run
with h5py.File(fname_cloudless) as f:
    temp_for_virga = np.array(f["temperature"])
    cvz_locs = np.array(f["cvz_locs"])
    if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
        nstr_upper = cvz_locs[-2]
    else:
        nstr_upper = cvz_locs[1]

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
virga_out = None
if fsed > 0:
    kz = np.ones_like(pressure_grid) * 1e10 # idk
    virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
    virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
    virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_for_virga, 'kz': kz}), kz_min=1e5, latent_heat=True)
    virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
    cl_run.fix_virga_clouds(virga_out)

out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
_, grad, _ = jpi.pt_adiabat(out, cl_run, opacity_ck, plot=False)

fig, axes = plt.subplots(3, 2, figsize=(8, 12))

layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
N = len(out["pressure"])

cvz_locs = out["cvz_locs"]
if cvz_locs[-2] > 0 and out["temperature"][cvz_locs[-2]] < 5199.9:
    convective_boundary = cvz_locs[-2]
else:
    convective_boundary = cvz_locs[1]
axes[0, 0].semilogy(out["temperature"], out["pressure"], label="solution")
max_temp = np.max(out["temperature"])
axes[0, 0].semilogy(temp_guess, out["pressure"], ls="--", c='b', label="guess")
max_temp = max(max_temp, np.max(temp_guess))
for (gas_name, gas_color) in zip(cloud_species, cloud_colors):
    p,t = vj.condensation_t(gas_name, 1, 2.2, pressure=out["pressure"])
    axes[0, 0].semilogy(t, p, c=gas_color, ls="--")
axes[0, 0].set_xlim((0, max_temp * 1.1))
axes[0, 0].set_ylim((np.min(out["pressure"]) * 0.9, np.max(out["pressure"]) * 1.1))
axes[0, 0].scatter([out["temperature"][convective_boundary]], [out["pressure"][convective_boundary]], c="k")
axes[0, 0].set_xlabel("Temperature (K)")
axes[0, 0].set_ylabel("Pressure (bar)")
axes[0, 0].invert_yaxis()
axes[0, 0].legend()

axes[1, 0].loglog(np.abs(out["fnet/fnetir"]), out["pressure"])
axes[1, 0].set_xlabel("Fnet/Fnet-IR")
axes[1, 0].set_ylabel("Pressure (bar)")
axes[1, 0].axvline(1e-3, ls="--", color="k")
axes[1, 0].invert_yaxis()

T_B = jpi.brightness_temperature(out["spectrum_output"], plot=False)
axes[1, 1].semilogx(1e4/out["spectrum_output"]["wavenumber"], T_B)
axes[1, 1].invert_yaxis()
axes[1, 1].axhline(np.max(out["temperature"]), ls="--", c='k')
axes[1, 1].set_xlabel("Wavelength (micron)")
axes[1, 1].set_ylabel("Brightness temperature (K)")

axes[0, 1].semilogy(out["dtdp"], layer_p)
axes[0, 1].semilogy(grad, layer_p)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel("dtdp (K/bar)")
axes[0, 1].set_ylabel("Pressure (bar)")
ax2 = axes[0, 1].twinx()
ax2.set_ylabel("Layer number")
ax2.set_yticks(np.arange(0, N-1, 10))
ax2.set_yticklabels(np.arange(0, N-1, 10)[::-1])

if virga_out is not None and all([x in virga_out.keys() for x in ["condensibles", "condensate_mmr"]]):
    for (i, condensible) in enumerate(virga_out["condensibles"]):
        axes[2, 0].loglog(virga_out["condensate_mmr"][:,i], virga_out["pressure"], label=condensible, color=cloud_colors[i])
    axes[2, 0].set_xlim((1e-10, 2 * np.max(virga_out["condensate_mmr"])))
    axes[2, 0].set_ylim((np.min(out["pressure"]), np.max(out["pressure"])))
    axes[2, 0].set_xlabel("Condensate mass mixing ratio")
    axes[2, 0].set_ylabel("Pressure (bar)")
    axes[2, 0].invert_yaxis()
    axes[2, 0].legend()

if virga_out is not None and "opd_per_layer" in virga_out.keys():
    axes[2, 1].loglog(virga_out["opd_per_layer"][:,55], layer_p)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim((1e-10, 2 * np.max(out["all_opd"][-(N-1):])))
    axes[2, 1].set_xlabel("Optical depth")
    axes[2, 1].set_ylabel("Pressure (bar)")

plt.tight_layout()
plt.savefig(os.path.join(picaso_path, "figures", "both_irradiated_withinversion", f"{fname_stem}.png"))
plt.close(fig)
with h5py.File(fname, "w") as f:
    f["temp_guess_init"] = temp_guess_init
    f.attrs["nstr_upper_init"] = nstr_upper_init
    out_to_hdf5(out, f)
