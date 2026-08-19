import h5py
from matplotlib import pyplot as plt
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

picaso_path = path.dirname(picaso.__path__[0])

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
cloud_colors = ['#CC5555', '#3BA39C', '#CCB84D', '#FF8C00']

__refdata__ = os.environ['picaso_refdata']
cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)


def plot_from_h5(h5_path, fig_path, temp_guess=None):
    # Currently, my output doesn't store the cloud MMRs, just the opacity
    # That's fine, I think
    out = {}
    spectrum_output = {}
    with h5py.File(h5_path) as f:
        for k in ["temperature", "pressure", "cvz_locs", "fnet/fnetir", "dtdp", "layer_temperature", "layer_pressure", "cloud_opd"]:
            out[k] = np.array(f[k])
        for k in filter(lambda x: x.startswith("spectrum_output"), f.keys()):
            spectrum_output[k[16:]] = np.array(f[k])
        out["spectrum_output"] = spectrum_output
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
    N = len(out["pressure"])

    cvz_locs = out["cvz_locs"]
    if cvz_locs[-2] > 0 and out["temperature"][cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
        convective_boundary = cvz_locs[-2]
    else:
        convective_boundary = cvz_locs[1]
    axes[0, 0].semilogy(out["temperature"], out["pressure"], label="solution")
    max_temp = np.max(out["temperature"])
    if temp_guess is not None:
        axes[0, 0].semilogy(temp_guess, out["pressure"], ls="--", c='b', label="guess")
        max_temp = max(max_temp, np.max(temp_guess))
    for (gas_name, gas_color) in zip(cloud_species, cloud_colors):
        p,t = vj.condensation_t(gas_name, 1, 2.2, pressure=out["pressure"])
        axes[0, 0].semilogy(t, p, c=gas_color, ls="--")
    axes[0, 0].set_xlim((0, max_temp * 1.1))
    axes[0, 0].set_ylim((np.min(out["pressure"]) * 0.9, np.max(out["pressure"]) * 1.1))
    axes[0, 0].scatter([out["temperature"][convective_boundary]], [out["pressure"][convective_boundary]], c="k")
    axes[0, 0].set_xlabel("Temperature (K)")
    axes[0, 0].set_ylabel("Pressure (bar)")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()

    axes[1, 0].loglog(np.abs(out["fnet/fnetir"]), out["pressure"])
    axes[1, 0].set_xlabel("Fnet/Fnet-IR")
    axes[1, 0].set_ylabel("Pressure (bar)")
    axes[1, 0].axvline(1e-3, ls="--", color="k")
    axes[1, 0].invert_yaxis()

    if np.max(out["cloud_opd"]) > 0:
        axes[1, 1].loglog(out["cloud_opd"][:,55], out["layer_pressure"])
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlim((1e-10, 2 * np.max(out["cloud_opd"][:,55])))
        axes[1, 1].set_xlabel("Cloud optical depth")
        axes[1, 1].set_ylabel("Pressure (bar)")

    """T_B = jpi.brightness_temperature(out["spectrum_output"], plot=False)
    axes[1, 1].semilogx(1e4/out["spectrum_output"]["wavenumber"], T_B)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axhline(np.max(out["temperature"]), ls="--", c='k')
    axes[1, 1].set_xlabel("Wavelength (micron)")
    axes[1, 1].set_ylabel("Brightness temperature (K)")
    """
    grad = np.array([did_grad_cp(np.asarray(t_i).item(), np.asarray(p_i).item(), AdiabatBundle)[0] for (t_i, p_i) in zip(out["layer_temperature"], out["layer_pressure"])])
    axes[0, 1].semilogy(out["dtdp"], layer_p)
    axes[0, 1].semilogy(grad, layer_p)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("dtdp (K/bar)")
    axes[0, 1].set_ylabel("Pressure (bar)")
    ax2 = axes[0, 1].twinx()
    ax2.set_ylabel("Layer number")
    ax2.set_yticks(np.arange(0, N-1, 10))
    ax2.set_yticklabels(np.arange(0, N-1, 10)[::-1])

    plt.tight_layout()
    plt.savefig(path.join(picaso_path, "figures", fig_path))
    plt.close(fig)

if __name__ == "__main__":
    # teff, grav, semi_major, fsed = 1200, 1000, 0.04, 2
    # For the "both_irradiated" runs, this is how the initial temperature guesses got generated
    # fsed_str = f"fsed{fsed}" if fsed != 0 else "nc"
    # fname_stem = f"irr_teff{teff}_grav{grav}_semimajor{semi_major}{fsed_str}"
    """
    # I could get this to work, but I'm pretty sure the cloudless runs and the initial guesses off Kazumasa's grid are the same anyway, so we can get the same effect by just comparing the accepted cloudless irradiated runs...
    legacy_fname_stem_cloudless = f"irr_teff{teff}_grav{grav}_semimajor{semi_major}"
        legacy_fname_cloudless = os.path.join(picaso_path, "data", "cloudless_irradiated_from_kazumasa", legacy_fname_stem_cloudless + ".pkl")
        temp_guess = None
        if os.path.exists(legacy_fname_cloudless):
            try:
                out_cloudless = pkl.load(open(legacy_fname_cloudless, "rb"))
                pressure_grid = out_cloudless["pressure"]
                temp_guess = out_cloudless["temperature"]
            except Exception:
                pass
        else:
            pressure_grid = np.logspace(-4, 3, nlevel)
            temp_guess = np.minimum(kazumasa_hj_grid_interpolation(1.0, semi_major, teff, grav, pressure_grid=pressure_grid), 5199.0)
    """
    for fname in tqdm(os.listdir(path.join(picaso_path, "data", "both_irradiated_startlow"))):
        data_path = path.join(picaso_path, "data", "both_irradiated_startlow", fname)
        plot_from_h5(data_path, f"both_irradiated_postrun/{fname[:-3]}_startlow.pdf")