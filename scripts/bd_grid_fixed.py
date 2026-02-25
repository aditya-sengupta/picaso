import pickle as pkl
import os
import warnings
warnings.filterwarnings('ignore')
import picaso
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
picaso_path = os.path.dirname(picaso.__path__[0])

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')

nstr_upper = 88
fsed = 2
semi_major = np.inf

for cloudmode in ["fixed", "selfconsistent"]:
    for grav in [316, 1000, 3160]:
            for teff in [900, 1200, 1500, 1800, 2100]:
                print(f"effective temperature = {teff} K, grav = {grav} m/s/s, cloud mode = {cloudmode}")
                fname_stem = f"bd_fsed{fsed}_teff{teff}_grav{grav}_cloudmode{cloudmode}"
                fname = os.path.join(picaso_path, f"data/bd_fixed_2602/{fname_stem}.pkl")
                
                cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
                cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
                cl_run.effective_temp(teff) # input effective temperature
                opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

                if semi_major < np.inf:
                    cl_run.star(opacity_ck, filename="data/solspec_picaso.dat", w_unit="um", f_unit="flam", semi_major=semi_major, semi_major_unit = u.AU, radius=1.0, radius_unit=u.R_sun)

                nlevel = 91 # number of plane-parallel levels in your code
                rfacv = 0.5
                
                # sonora_df = pd.read_csv(f"reference/sonora_grids/diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
                if cloudmode == "fixed_bobcatstart":
                    pressure_grid,temp_guess = np.loadtxt(jdi.os.path.join(
                                sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                                usecols=[1,2],unpack=True, skiprows = 1)
                    cloudmode_ = "fixed"
                else:
                    sonora_df = pd.read_csv(f"reference/sonora_grids/diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
                    pressure_grid = np.array(sonora_df["P"])
                    temp_guess = np.array(sonora_df["T"])
                    cloudmode_ = cloudmode
                
                cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
                cl_run.virga(condensates=cloud_species, directory="/Users/adityasengupta/virga/refrind", runmode=cloudmode_, mh=1, fsed=fsed, latent_heat=True)
                out_fixed = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
                jpi.diagnostic_plot(out_fixed, cl_run, opacity_ck, os.path.join(picaso_path, f"figures/bd_fixed_figures/{fname_stem}.png"))
                pkl.dump(out_fixed, open(fname, 'wb'))
                
