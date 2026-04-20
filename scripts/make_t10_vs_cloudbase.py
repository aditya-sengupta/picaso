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
import h5py

import sys
sys.path.append(".")
from scripts.diamondback.diamondback_cz import find_rcb_diamondback
from scripts.load_diamondback import diamondback_pt, find_cloudbase_diamondback

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

tag = "_withinversion" # or ""

def _dTdp(p, t):
    grad_x, _ = did_grad_cp(np.asarray(t).item(), np.asarray(p).item(), AdiabatBundle)
    return float(grad_x) * t / p

def t10(p_col, t_col):
    solver = ode(_dTdp).set_integrator('dopri5', rtol=1e-8, atol=1e-8, nsteps=5000)
    idx = np.where(t_col < 5199)[0][-1]
    solver.set_initial_value(t_col[idx], p_col[idx])
    solver.integrate(10.0)
    return float(solver.y[0])

for fsed in [1, 2, 3, 4, 8]:
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    gravs = [31, 100, 316, 1000, 3160]
    teffs = np.arange(1000, 2401, 200)
    semimajors = [0.02, 0.04, 0.13, 0.5, np.inf]
    cmap = cm.magma(np.linspace(0, 1, len(gravs)+1))[:-1]

    for (j, grav) in enumerate(gravs):
        cloud_bases, t10s, cloudy_cases = {}, {}, {}
        for semimajor in semimajors:
            cloud_bases[semimajor] = []
            t10s[semimajor] = []
            cloudy_cases[semimajor] = []
            for (k, teff) in enumerate(teffs):
                if semimajor < np.inf:
                    fsed_str = f"fsed{fsed}" if fsed != "nc" else "nc"
                    fname = f"data/both_irradiated{tag}/irr_teff{teff}_grav{grav}_semimajor{semimajor:.2f}{fsed_str}.h5"
                    with h5py.File(os.path.join(picaso_path, fname)) as f:
                        opd = np.array(f["fixed_opd"])
                        cloud_indices = np.where(np.sum(opd, axis=1) > 0)[0]
                        if len(cloud_indices) > 0:
                            cloud_base = f["layer_pressure"][cloud_indices[-1]]
                            cloud_bases[semimajor].append(cloud_base)
                            cloudy_cases[semimajor].append(k)
                        pressure, temperature = np.array(f["pressure"]), np.array(f["temperature"])
                        t10s[semimajor].append(t10(pressure, temperature))
                elif grav != 17:
                    cloud_base_diamondback = find_cloudbase_diamondback(teff, grav, fsed)
                    if cloud_base_diamondback > -1:
                        cloud_bases[semimajor].append(cloud_base_diamondback)
                        cloudy_cases[semimajor].append(k)
                    p, t = diamondback_pt(teff, grav, fsed)
                    t10s[semimajor].append(t10(p, t))
        
        for (semimajor, c) in zip(semimajors[:-1], cmap[:-1]):
            label = f"a = {semimajor} au"
            curr_ax = axs[j//3,j%3]
            indices_to_plot = np.array(cloudy_cases[semimajor])
            if len(indices_to_plot) > 0:
                t10s_this = np.array(t10s[semimajor])[indices_to_plot]
                t10s_sonora = np.array(t10s[np.inf])[indices_to_plot]
                curr_ax.scatter(t10s_this - t10s_sonora, cloud_bases[semimajor], color=c, label=label, s=5)
            curr_ax.axvline([0.0], ls="--", c='k')
            curr_ax.axhline([10.0], ls="--", c='xkcd:grey green')
            axs[-1,j%3].set_xlabel(r"$T_{10} - T_{10}(D'back)$ (K)")
            curr_ax.set_ylabel("Cloud base pressure (bar)")
            curr_ax.set_xlim((-400, 400))
            curr_ax.set_ylim((1e-4, 1e3))
            curr_ax.set_yscale('log')
            curr_ax.invert_yaxis()
            curr_ax.set_title(f"g = {grav}" + r"m/s${}^2$" + ", " + fsed_str)
            if j%3 > 0:
                curr_ax.yaxis.set_visible(False)
            if j//3 == 0:
                curr_ax.xaxis.set_visible(False)

    axs[1,2].axis('off')
    axs[1,1].legend(bbox_to_anchor=(2, 1))
    figpath = os.path.join(picaso_path, f"figures/t10/t10_vs_cloudbase{tag}_{fsed_str}.png")
    plt.savefig(figpath, dpi=300)
    print(figpath)
    plt.close(fig)