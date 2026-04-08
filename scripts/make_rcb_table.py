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

tag = "_startlow" # or ""

bobcat_path = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

for fsed in [1, 2, 3, 4, 8, "nc"]:
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    gravs = [17, 31, 100, 316, 1000, 3160]
    semimajors = [0.02, 0.04, 0.13, 0.5, np.inf]
    teffs = np.arange(1000, 2401, 200)
    cmap = cm.magma(np.linspace(0, 1, len(semimajors)+1))[:-1]

    for (j, grav) in enumerate(gravs):
        for (semimajor, c) in zip(semimajors, cmap):
            teffs_this = []
            rcbs = []
            for teff in teffs:
                if semimajor < np.inf:
                    fsed_str = f"fsed{fsed}" if fsed != "nc" else "nc"
                    fname = f"data/both_irradiated{tag}/irr_teff{teff}_grav{grav}_semimajor{semimajor:.2f}{fsed_str}.h5"
                    if os.path.exists(os.path.join(picaso_path, fname)):
                        with h5py.File(os.path.join(picaso_path, fname)) as f:
                            cvz_locs, pressure, temperature = np.array(f["cvz_locs"]), np.array(f["pressure"]), np.array(f["temperature"])
                            if cvz_locs[4] > 0 and temperature[cvz_locs[4]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
                                rcb_index = cvz_locs[4]
                            else:
                                rcb_index = cvz_locs[1]
                            teffs_this.append(teff)
                            rcbs.append(pressure[rcb_index])
                elif grav != 17:
                    teffs_this.append(teff)
                    p, _ = diamondback_pt(teff, grav, fsed)
                    rcbs.append(p[find_rcb_diamondback(teff, grav, fsed)])
                # add in Diamondback 

            label = f"a = {semimajor} au" if semimajor < np.inf else "sonora"
            curr_ax = axs[j//3,j%3]
            curr_ax.semilogy(teffs_this, rcbs, c=c, label=label, lw=1 if semimajor < np.inf else 2)
            curr_ax.invert_yaxis()
            axs[-1,j%3].set_xlabel("Tint (K)")
            curr_ax.set_ylabel("RCB pressure (bar)")
            curr_ax.set_xlim((np.min(teffs), np.max(teffs)))
            curr_ax.set_ylim((1e2, 1e-2))
            curr_ax.set_title(f"g = {grav} m/s/s")
            if j%3 > 0:
                curr_ax.yaxis.set_visible(False)
            if j//3 == 0:
                curr_ax.xaxis.set_visible(False)

    axs[0,0].legend(fontsize='small')
    figpath = os.path.join(picaso_path, f"figures/rcb/rcb_table{tag}_{fsed_str}.pdf")
    plt.savefig(figpath)
    print(figpath)
    plt.close(fig)