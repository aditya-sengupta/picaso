import os
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import ode
from picaso.grad import did_grad_cp
from collections import namedtuple
import json
import pandas as pd

__refdata__ = os.environ['picaso_refdata']
picaso_path = os.path.dirname(os.path.dirname(__refdata__))

cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)

bobcat_path = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

def _dTdp(p, t):
    grad_x, _ = did_grad_cp(np.asarray(t).item(), np.asarray(p).item(), AdiabatBundle)
    return float(grad_x) * t / p

def t10(p_col, t_col):
    solver = ode(_dTdp).set_integrator('dopri5', rtol=1e-8, atol=1e-8, nsteps=5000)
    idx = np.where(t_col < 5199)[0][-1]
    solver.set_initial_value(t_col[idx], p_col[idx])
    solver.integrate(10.0)
    return float(solver.y[0])

fig, axs = plt.subplots(6, 2, figsize=(10, 30))
gravs = [17, 31, 100, 316, 1000, 3160]
semimajors = [0.02, 0.04, 0.13, 0.5, np.inf]
teffs = np.arange(200, 2401, 200)
cloudmodes = ["cloudless", "fixed"]
cmap = cm.magma(np.linspace(0, 1, len(semimajors)+1))[:-1]
for (i, cloudmode) in enumerate(cloudmodes):
    for (j, grav) in enumerate(gravs):
        for (semimajor, c) in zip(semimajors, cmap):
            t10s = []
            teffs_this = []
            for teff in teffs:
                if semimajor < np.inf:
                    if cloudmode == "cloudless":
                        fname = f"data/cloudless_irradiated_from_kazumasa/irr_teff{teff}_grav{grav}_semimajor{semimajor}.npz"
                    else:
                        fname = f"data/cloudy_irradiated/irr_teff{teff}_grav{grav}_semimajor{semimajor}_fsed2.npz"
                    if os.path.exists(os.path.join(picaso_path, fname)):
                        f = np.load(os.path.join(picaso_path, fname))
                        pressure, temperature = f['p'], f['t']
                        t10s.append(t10(pressure, temperature))
                        teffs_this.append(teff)
                elif grav >= 31:
                    pressure_sonora, temp_sonora = None, None
                    if teff >= 900:
                        if cloudmode == "fixed":
                            cl = "f2"
                        else:
                            cl = "nc"
                        sonora_df = pd.read_csv(os.path.join(__refdata__, f"sonora_grids/diamondback/t{teff}g{grav}{cl}_m0.0_co1.0.pt"), sep=r"\s+", skiprows=[1])
                        pressure_sonora = np.array(sonora_df["P"])
                        temp_sonora = np.array(sonora_df["T"])
                    else:
                        pressure_sonora, temp_sonora = np.loadtxt(os.path.join(bobcat_path,f"t{teff}g{grav}nc_m0.0.cmp.gz"), usecols=[1,2],unpack=True, skiprows=1)
                    t10s.append(t10(pressure_sonora, temp_sonora))
                    teffs_this.append(teff)

            label = f"a = {semimajor} au" if semimajor < np.inf else "sonora"
            axs[j,i].plot(teffs_this, t10s, c=c, label=label, lw=1 if semimajor < np.inf else 3)
            axs[j,i].set_xlabel("Tint (K)")
            axs[j,i].set_ylabel("T10 (K)")
            axs[j,i].set_xlim((np.min(teffs), np.max(teffs)))
            axs[j,i].set_ylim((0, 5199))
            axs[j,i].set_title(f"g = {grav} m/s/s, {cloudmode}")
axs[0,0].legend()
plt.savefig(os.path.join(picaso_path, "figures/t10/first_t10s.pdf"))
plt.close(fig)