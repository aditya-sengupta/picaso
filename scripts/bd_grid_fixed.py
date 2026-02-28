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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

sys.path.append(".")
from load_kazumasa_irradiated_models import kazumasa_hj_grid_interpolation
from diagnostic_plot import diagnostic_plot

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
picaso_path = os.path.dirname(picaso.__path__[0])

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')

nstr_upper = 88
fsed = 2

# effective^4 = equilibrium^4 + intrinsic^4

cloudmode = sys.argv[1]
grav = int(sys.argv[2])
teff = int(sys.argv[3])
try:
    semi_major = float(sys.argv[4])
except IndexError:
    semi_major = np.inf

print(f"effective temperature = {teff} K, grav = {grav} m/s/s, cloud mode = {cloudmode}, semimajor axis = {semi_major} au")
fname_stem = f"bd_cloudmode{cloudmode}_fsed{fsed}_teff{teff}_grav{grav}_semimajor{semi_major}_refactor260227"
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
sonora_df = pd.read_csv(f"reference/sonora_grids/diamondback/t900g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
pressure_grid = np.logspace(-5, 3, nlevel)
temp_guess = kazumasa_hj_grid_interpolation(0, semi_major, teff, grav, pressure_grid=pressure_grid)

cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
kz = np.ones_like(pressure_grid) * 1e10
virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
v_out = vj.compute(virga_planet, as_dict=True, directory="/Users/adityasengupta/virga/refrind")
cl_run.fix_virga_clouds(v_out)
out_fixed = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
diagnostic_plot(out_fixed, cl_run, opacity_ck, virga_out=v_out, fname=os.path.join(picaso_path, f"figures/bd_fixed_figures/{fname_stem}.png"), temp_guess=temp_guess)
pkl.dump(out_fixed, open(fname, 'wb'))
                