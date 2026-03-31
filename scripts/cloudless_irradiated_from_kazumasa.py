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
import virga.justdoit as vj
import astropy.units as u
import astropy.constants as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

SIGMA_SB = float(c.sigma_sb / (u.W / u.m**2 / u.K**4))

sys.path.append(".")
from diagnostic_plot import pre_fixed_plot, diagnostic_plot
from load_kazumasa_irradiated_models import kazumasa_hj_grid_interpolation
picaso_path = os.path.dirname(picaso.__path__[0])
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

mh = '0.0'
CtoO = '0.46'
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

nstr_upper = 88

# effective^4 = equilibrium^4 + intrinsic^4

# cloudmode, grav, teff, semi_major = "cloudless", 1000, 600, 0.02
cloudmode = "cloudless"
for grav in np.array([17, 31, 100, 316, 1000, 3160]):
    for teff in np.arange(200, 2401, 200):
        for semi_major in np.array([0.02, 0.04, 0.13, 0.5]):
            already_run = False
            print(f"effective temperature = {teff} K, grav = {grav} m/s/s, cloud mode = {cloudmode}, semimajor axis = {semi_major} au")
            fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major}"
            prev_run = os.path.join(picaso_path, f"data/match_sagnick/{fname_stem}.pkl")
            if os.path.exists(prev_run):
                try:
                    out = pkl.load(open(prev_run, "rb"))
                    if out["converged"] == 1 and np.max(out["temperature"]) < 5199:
                        already_run = True
                except Exception:
                    pass
            fname = os.path.join(picaso_path, f"data/cloudless_irradiated_from_kazumasa/{fname_stem}.pkl")

            cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(teff) # input effective temperature
            opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun,semi_major=semi_major, semi_major_unit=u.AU)

            nlevel = 91 # number of plane-parallel levels in your code
            rfacv = 0.5 # irradiated fixed clouds converge horribly with this = 0 (kinda obviously if you think about it)
            
            if not already_run:
                pressure_grid = np.logspace(-4, 3, nlevel)
                pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"), usecols=[1,2],unpack=True, skiprows = 1)
                temp_guess = np.minimum(kazumasa_hj_grid_interpolation(1.0, semi_major, teff, grav, pressure_grid=pressure_grid), 5199.0)

                cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)

                out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
                _, grad, _ = jpi.pt_adiabat(out, cl_run, opacity_ck, plot=False)
                diagnostic_plot_path = os.path.join(picaso_path, f"figures/cloudless_irradiated_from_kazumasa/{fname_stem}.png")
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))

                layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
                N = len(out["pressure"])

                cvz_locs = out["cvz_locs"]
                if cvz_locs[-2] > 0 and cvz_locs[2] != 89:
                    convective_boundary = cvz_locs[-2]
                else:
                    convective_boundary = cvz_locs[1]
                axes[0, 0].semilogy(out["temperature"], out["pressure"], label="solution")
                max_temp = np.max(out["temperature"])
                if not already_run and temp_guess is not None:
                    axes[0, 0].semilogy(temp_guess, out["pressure"], ls="--", c='b', label="guess")
                    max_temp = max(max_temp, np.max(temp_guess))
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

                plt.tight_layout()
                plt.savefig(os.path.join(picaso_path, "figures", "cloudless_irradiated", f"{fname_stem}.png"))
                plt.close(fig)
                pkl.dump(out, open(fname, 'wb'))

                # Teff = (-np.sum((lambda x: 0.5 * (x[1:] + x[:-1]))(out["spectrum_output"]['thermal']) * np.diff(1e4 / out["spectrum_output"]["wavenumber"])) / SIGMA_SB) ** (1/4)
