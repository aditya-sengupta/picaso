# I did this before, but only on MMRs; this time I'll be comparing OPD/SSA/asymmetry as well.

import sys
sys.path.append(".")
from matplotlib import pyplot as plt
from virga import justdoit as vj
from astropy import units as u
import numpy as np
import pandas as pd
from load_diamondback import read_diamondback_optical_properties, read_diamondback_structure

teff, grav, fsed = 900, 316, 2
cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

diamondback_structure = read_diamondback_structure(teff, grav, fsed)
diamondback_optical_properties = read_diamondback_optical_properties(teff, grav, fsed)
diamondback_optical_properties = diamondback_optical_properties[diamondback_optical_properties.spectral_window == 55]

virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({x: np.array(diamondback_structure[x]) for x in ['pressure', 'temperature', 'kz']}), kz_min=1e5, latent_heat=True)
v_out = vj.compute(virga_planet, as_dict=True, directory="/Users/adityasengupta/virga/refrind")

for (c, k, kvirga) in zip(['r', 'b', 'k'], ['tau', 'g0', 'w0'], ['opd_per_layer', 'asymmetry', 'single_scattering']):
    plt.loglog(diamondback_optical_properties[k], diamondback_structure["pressure"], label=f"{k} db", color=c)
    plt.loglog(v_out[kvirga][:,55], v_out["pressure"], label=f"{k} virga", ls='--', color=c)
plt.gca().invert_yaxis()
plt.xlabel("Optical property")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.show()