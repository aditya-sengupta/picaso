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

nstr_upper = 63
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
import virga.justdoit as vj
kz = np.ones_like(pressure_grid) * 1e10 #

virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
v_out = vj.compute(virga_planet, as_dict=True, directory="/Users/adityasengupta/virga/refrind")
cl_run.fix_virga_clouds(v_out)
# cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
# %%
# I could've made this way simpler if I had just called virga directly!
import virga.justdoit as vj
kz = np.ones_like(pressure_grid) * 1e10 #

virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
virga_out = vj.compute(virga_planet, as_dict=True, directory="/Users/adityasengupta/virga/refrind")
# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot virga_out
for (i, species) in enumerate(cloud_species):
    axes[0].loglog(virga_out["opd_by_gas"][:,i], virga_out["pressure"], label=species)
axes[0].set_xlabel('Optical Depth')
axes[0].set_ylabel('Pressure (bar)')
axes[0].set_title('Virga Direct')
axes[0].set_xlim((1e-10, 1e5))
axes[0].legend()
axes[0].invert_yaxis()

# Plot v_out
for (i, species) in enumerate(cloud_species):
    axes[1].loglog(v_out["opd_by_gas"][:,i], v_out["pressure"], label=species)
axes[1].set_xlabel('Optical Depth')
axes[1].set_ylabel('Pressure (bar)')
axes[1].set_title('Virga via PICASO call')
axes[1].set_xlim((1e-10, 1e5))
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
# %%
