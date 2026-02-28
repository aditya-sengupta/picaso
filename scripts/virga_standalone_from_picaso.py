# %%
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

cloudmode = "fixed"
grav = 316
teff = 900
semi_major = np.inf

cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

nlevel = 91 # number of plane-parallel levels in your code
rfacv = 0.5

# sonora_df = pd.read_csv(f"reference/sonora_grids/diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
pressure_grid = np.logspace(-5, 3, nlevel)
temp_guess = kazumasa_hj_grid_interpolation(0, semi_major, teff, grav, pressure_grid=pressure_grid)

cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
cl_run.inputs['climate']['cloudy'] = 'fixed'
cl_run.initialize_climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %%
# you *can* run virga to get your cloud opacities
# but you're not required to;
# you can also provide inputs for these arrays from some other source
v_out = cl_run.virga(condensates=cloud_species, directory="/Users/adityasengupta/virga/refrind", mh=1, fsed=fsed, latent_heat=True)
cl_run.CloudParameters.OPD[:,:,0] = v_out["opd_per_layer"]
cl_run.CloudParameters.W0[:,:,0] = v_out["single_scattering"]
cl_run.CloudParameters.G0[:,:,0] = v_out["asymmetry"]

cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
# %%
