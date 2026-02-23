# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import pandas as pd
from copy import deepcopy

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')

nstr_upper = 74
fsed = 2
teff = 1200
cloudmode = "fixed"
grav = 316.0

cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

nlevel = 91 # number of plane-parallel levels in your code
rfacv = 0.0

sonora_df = pd.read_csv(f"/Users/adityasengupta/picaso/reference/sonora_grids/diamondback/t900g316f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
pressure_grid = np.array(sonora_df["P"])
temp_guess = np.array(sonora_df["T"])

cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
cl_run.virga(condensates=cloud_species, directory="/Users/adityasengupta/virga/refrind", runmode=cloudmode, mh=1, fsed=fsed, latent_heat=True)

# %%
out = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

# %%
_, grad, _ = jpi.pt_adiabat(out, cl_run, opacity_ck, plot=False)
jpi.diagnostic_plot(out, grad)
# %%
