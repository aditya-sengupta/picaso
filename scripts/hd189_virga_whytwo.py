import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')
from astropy import units as u
import h5py
from virga import justdoit as vj
import pandas as pd

virga_path = "~/atmospheres/virga"

tint = 200
grav = 10 ** 1.4
nlevel = 91
fsed = 1

for (ms, m) in zip(["-1.0", "0.0"], [0.1, 1]):
    with h5py.File(f"data/hd189/feh{ms}_co0.55_rfacv0.80.h5") as f:
        pressure, temp_cloudless = np.array(f["pressure"]), np.array(f["temperature"])

    kz = np.ones_like(pressure) * 1e12 # idk
    virga_planet = vj.Atmosphere(["SiO2"], fsed=fsed, mh=m, mmw=2.2)
    virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
    virga_planet.ptk(df = pd.DataFrame({'pressure':pressure, 'temperature':temp_cloudless, 'kz': kz}), kz_min=1e5, latent_heat=True)
    virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].semilogy(virga_out["temperature"], virga_out["pressure"], label="temperature")
    axs[0].semilogy(vj.condensation_t("SiO2", mh=m, mmw=2.2, pressure=virga_out["pressure"])[1], virga_out["pressure"], ls="--", color='k', label="SiO2 condensation curve")
    axs[0].invert_yaxis()
    axs[0].set_xlim((0, 2500))
    axs[0].set_xlabel("Temperature")
    axs[0].set_ylabel("Pressure (bar)")
    axs[0].legend(loc="upper left")

    axs[1].loglog(virga_out["condensate_mmr"].ravel(), virga_out["pressure"])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Condensate MMR")
    # axs[1].set_ylabel("Pressure (bar)")

    plt.savefig(f"figures/hd189_virga_whytwo_m{m}.png")
    print(os.path.abspath(f"figures/hd189_virga_whytwo_m{m}.png"))
    plt.close()