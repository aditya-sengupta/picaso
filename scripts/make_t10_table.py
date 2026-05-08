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
from scripts.load_diamondback import diamondback_pt

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

count_available_overall, count_all_overall = 0, 0
for fsed in ["nc"]:
    fsed_str = f"fsed{fsed}" if fsed != "nc" else "nc"
    fig, axs = plt.subplots(4, 3, figsize=(8, 10))
    #log_gravs = [3.25, 3.5, 4, 4.5, 5, 5.5]
    #gravs = [17, 31, 100, 316, 1000, 3160]
    log_gravs = [3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5]
    gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
    semimajors = [0.02, 0.04, 0.13, 0.5]
    teffs = np.arange(100, 2401, 10)
    cmap = cm.magma(np.linspace(0, 1, len(semimajors)+1))[:-1]

    for (semimajor, c) in zip(semimajors, cmap):
        all_teffs = []
        all_t10s = []
        count_available, count_all = 0, 0
        for (j, grav) in enumerate(gravs):
            teffs_this = []
            t10s = []
            for teff in teffs:
                if semimajor < np.inf:
                    fname = f"data/both_irradiated{tag}/irr_teff{teff}_grav{grav}_semimajor{semimajor:.2f}{fsed_str}.h5"
                    count_all += 1
                    count_all_overall += 1
                    if os.path.exists(os.path.join(picaso_path, fname)):
                        try:
                            with h5py.File(os.path.join(picaso_path, fname)) as f:
                                if "pressure" in f:
                                    count_available += 1
                                    count_available_overall += 1
                                    pressure, temperature = np.array(f["pressure"]), np.array(f["temperature"])
                                    teffs_this.append(teff)
                                    t10s.append(t10(pressure, temperature))
                        except Exception:
                            continue
                else:
                    try:
                        p, t = diamondback_pt(teff, grav, fsed)
                        teffs_this.append(teff)
                        t10s.append(t10(p, t))
                    except FileNotFoundError:
                        continue

            all_teffs.append(teffs_this)
            all_t10s.append(t10s)

            label = f"a = {semimajor} au" if semimajor < np.inf else "sonora"
            curr_ax = axs[j//3,j%3]
            curr_ax.plot(teffs_this, t10s, color=c, label=label)
            # lw=1 if semimajor < np.inf else 2
            curr_ax.invert_yaxis()
            axs[-1,j%3].set_xlabel("Tint (K)")
            curr_ax.set_ylabel("T10 (K)")
            curr_ax.set_xlim((np.min(teffs), np.max(teffs)))
            curr_ax.set_ylim((0, 5199))
            curr_ax.set_title(f"g = {grav} m/s/s, {fsed_str}")
            if j%3 > 0:
                curr_ax.yaxis.set_visible(False)
            if j//3 < 3:
                curr_ax.xaxis.set_visible(False)

        print(f"{fsed = }, {semimajor = }: {count_available} / {count_all}")
        T10_table = 0.1 * np.ones((len(gravs), len(teffs)))
        for (j, grav) in enumerate(gravs):
            if len(all_teffs[j]) > 1:
                T10_table[j,:] = np.interp(teffs, all_teffs[j], all_t10s[j])

        np.savez(f"data/t10_tables/atm_semimajor{semimajor:.2f}_{fsed_str}{tag}.npz", logGravity=log_gravs, logTeff=np.log10(teffs), logT10=np.log10(T10_table))

    axs[0,0].legend(fontsize='small')
    figpath = os.path.join(picaso_path, f"figures/t10/t10_table{tag}_{fsed_str}.png")
    plt.savefig(figpath, dpi=600)
    print(figpath)
    plt.close(fig)

print(f"{count_available_overall} / {count_all_overall}")