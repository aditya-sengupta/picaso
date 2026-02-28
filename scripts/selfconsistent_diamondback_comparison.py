# %%
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["PYSYN_CDBS"] = os.path.join(os.environ['picaso_refdata'],'stellar_grids')

import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# matching Mang et al. 2026 figure 8
# the Diamondback curve
fsed = 8
grav = 316
cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
out_d = {}

mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
virga_dir = os.path.join("/Users", "adityasengupta",'virga', "refrind")

teffs = [2400]
# %%
for teff in teffs:
    # a self-consistent run
    # #sonora bobcat cloud free structures file
    sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')
    cl_run = jdi.inputs(calculation="browndwarf", climate = True)
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(teff)

    opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

    nlevel = 91 # number of plane-parallel levels in your code

    pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                                sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                                usecols=[1,2],unpack=True, skiprows = 1)

    rcb_guess = 60
    rfacv = 0.0
    cl_run.inputs_climate(temp_guess=temp_bobcat, pressure=pressure_bobcat,
                        rcb_guess=rcb_guess, rfacv=rfacv)
    cl_run.inputs['climate']['cloudy'] = 'selfconsistent'
    cl_run.virga(condensates=cloud_species, directory = virga_dir, mh=1,fsed=fsed, latent_heat=True)

    out_d[teff] = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=False)

# %%
for (c, teff) in zip(["xkcd:bright blue", "xkcd:turquoise", "xkcd:olive green"], teffs):
    sonora_df = pd.read_csv(f"../reference/sonora_grids/diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    pressure_grid = np.array(sonora_df["P"])
    temp = np.array(sonora_df["T"])

    plt.semilogy(temp, pressure_grid, label=f"{teff} d'back", c=c, ls="--")
    plt.semilogy(out_d[teff]["temperature"], out_d[teff]["pressure"], label=f"{teff} PICASO", ls="-", c=c)
plt.gca().invert_yaxis()
plt.xlim((300, 2600))
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.show()

# %%
