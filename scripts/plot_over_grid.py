import h5py
from matplotlib import pyplot as plt
from matplotlib import cm
import picaso
import os
from os import path
import numpy as np
from virga import justdoit as vj
from picaso import justplotit as jpi
import json
from collections import namedtuple
from picaso.grad import did_grad_cp
import pickle as pkl
from tqdm import tqdm
import imageio

tag = "_withinversion"

picaso_path = path.dirname(picaso.__path__[0])

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
cloud_colors = ['#CC5555', '#3BA39C', '#CCB84D', '#FF8C00']
fseds = [1, 2, 3, 4, 8, 0]

def plot_gridpoint(teff, grav, semi_major):
    # This will pull data from all the fseds and overlay them
    all_outs = {}
    fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major:.2f}"
    for fsed in fseds:
        fsed_str = f"fsed{fsed}" if fsed > 0 else "nc"
        h5_path = path.join(picaso_path, "data", "both_irradiated" + tag, f"{fname_stem}{fsed_str}.h5")
        out = {}
        spectrum_output = {}
        with h5py.File(h5_path) as f:
            for k in ["temperature", "pressure", "cvz_locs", "fnet/fnetir", "dtdp", "layer_temperature", "layer_pressure", "cloud_opd"]:
                out[k] = np.array(f[k])
            for k in filter(lambda x: x.startswith("spectrum_output"), f.keys()):
                spectrum_output[k[16:]] = np.array(f[k])
            out["spectrum_output"] = spectrum_output
        all_outs[fsed] = out

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for (c, fsed) in zip(cm.summer(np.linspace(0, 1, len(fseds)+1)), fseds):
        out = all_outs[fsed]
        cvz_locs = out["cvz_locs"]
        if cvz_locs[-2] > 0 and out["temperature"][cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            convective_boundary = cvz_locs[-2]
        else:
            convective_boundary = cvz_locs[1]
        axes[0].semilogy(out["temperature"], out["pressure"], label=f"{fsed = }" if fsed > 0 else "nc", c=c)
        axes[0].set_ylim((np.min(out["pressure"]) * 0.9, np.max(out["pressure"]) * 1.1))
        axes[0].scatter([out["temperature"][convective_boundary]], [out["pressure"][convective_boundary]], color=c)

        if np.max(out["cloud_opd"]) > 0:
            axes[1].loglog(out["cloud_opd"][:,55], out["layer_pressure"], c=c)

    for (gas_name, gas_color) in zip(cloud_species, cloud_colors):
        p,t = vj.condensation_t(gas_name, 1, 2.2, pressure=out["pressure"])
        axes[0].semilogy(t, p, c=gas_color, ls="--", label=gas_name)
    axes[0].set_xlim((0, 5199.9))
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize='small')

    axes[1].invert_yaxis()
    axes[1].set_xlim((1e-10, 1e2))
    axes[1].set_xlabel("Cloud optical depth")
    # axes[1].set_ylabel("Pressure (bar)")

    plt.suptitle(r"$T_{int} = $" + str(teff) + "K, surface gravity = " + str(grav) + r"m/s${}^2$" + f", semimajor axis = {semi_major} au")
    plt.savefig(path.join(picaso_path, "figures", "grid_plots" + tag, f"{fname_stem}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # teff, grav, semi_major, fsed = 1200, 1000, 0.04, 2
    # For the "both_irradiated" runs, this is how the initial temperature guesses got generated
    # fsed_str = f"fsed{fsed}" if fsed != 0 else "nc"
    # fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major}{fsed_str}"

    fpaths = []
    for semi_major in [0.02, 0.04, 0.13, 0.50]:
        for grav in [17, 31, 100, 316, 1000, 3160]:
            for teff in np.arange(1000, 2401, 200):
                plot_gridpoint(teff, grav, semi_major)
                fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major:.2f}"
                fpath = path.join(picaso_path, "figures", "grid_plots" + tag, f"{fname_stem}.png")
                fpaths.append(fpath)

    ims = [imageio.v2.imread(f) for f in fpaths]
    imageio.mimwrite(path.join(picaso_path, "figures", "grid_plots" + tag, "_irradiated_cloudy_grid.mp4"), ims, fps=4)