# - Diamondback / cloudless irradiated / cloudy irradiated plot
# %%
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

import sys
sys.path.append("..")
from load_diamondback import diamondback_pt

teff = 1000
grav = 1000
fsed = 2
metallicity = 1.0
mean_molecular_weight = 2.35
gases = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

# first, load Diamondback
diamondback_p, diamondback_t = diamondback_pt(teff, grav, fsed)

# then, what if you fixed the Diamondback cloud, but the Bobcat P-T profile, and then ran PICASO with a fixed cloud?
diamondback_ptk = read_diamondback_clouds(teff, grav_ms2, fsed)
recommended_gases = vdi.recommend_gas(diamondback_p, diamondback_t, metallicity, mean_molecular_weight)
recommended_gases = np.intersect1d(recommended_gases, gases)
sum_planet = vdi.Atmosphere(recommended_gases,fsed=fsed,mh=metallicity, mmw = mean_molecular_weight)
sum_planet.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)'))
sum_planet.ptk(df = diamondback_ptk)
virga_out = vdi.compute(sum_planet, "~/atmospheres/virga/refrind", og_solver=True)

mh = '0.0'
CtoO = '0.46'
nlevel = 91
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
cl_run = jdi.inputs(calculation="browndwarf", climate = True)
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
cl_run.effective_temp(teff)
cl_run.inputs_climate(temp_guess=diamondback_t, pressure=diamondback_t, rcb_guess=89, rfacv=0.0) # we'll change this later.
cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
cl_run.fix_virga_clouds(virga_out)
out = cl_run.climate(opacity_ck, save_all_profiles=False, with_spec=True, verbose=True)
# %%
plt.semilogy(diamondback_t, diamondback_p, label="Diamondback")
plt.semilogy(out["temperature"], out["pressure"], label="PICASO with clouds fixed at Diamondback's")
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.gca().invert_yaxis()
plt.legend()

# %%
