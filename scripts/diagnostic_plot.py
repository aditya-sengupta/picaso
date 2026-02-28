import numpy as np
from matplotlib import pyplot as plt
from picaso.justplotit import pt_adiabat
import virga.justdoit as vj
from picaso.justplotit import brightness_temperature

def diagnostic_plot(out, cl_run, opacity_ck, virga_out=None, fname=None, temp_guess=None):
    if virga_out is None and "virga_output" in out.keys():
        virga_out = out["virga_output"]
    _, grad, _ = pt_adiabat(out, cl_run, opacity_ck, plot=False)
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    t = f"Teff = {cl_run.inputs['planet']['T_eff']}, g = {cl_run.inputs['planet']['gravity'] / 100}, cloud = {cl_run.inputs['climate']['cloudy']}"
    if 'virga_kwargs' in cl_run.inputs['climate'].keys():
        t += f", fsed = {cl_run.inputs['climate']['virga_kwargs']['fsed']}"
    plt.suptitle(t)

    layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
    N = len(out["pressure"])

    cloud_colors = ['#CC5555', '#3BA39C', '#CCB84D', '#FF8C00']
        
    for (i, condensible) in enumerate(virga_out["condensibles"]):
        axes[0, 0].loglog(virga_out["condensate_mmr"][:,i], virga_out["pressure"], label=condensible, color=cloud_colors[i])
    axes[0, 0].set_xlim((1e-10, 2 * np.max(virga_out["condensate_mmr"])))
    axes[0, 0].set_ylim((np.min(out["pressure"]), np.max(out["pressure"])))
    axes[0, 0].set_xlabel("Condensate mass mixing ratio")
    axes[0, 0].set_ylabel("Pressure (bar)")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()

    cvz_locs = out["cvz_locs"]
    if cvz_locs[-2] > 0 and cvz_locs[2] != 89:
        convective_boundary = cvz_locs[-2]
    else:
        convective_boundary = cvz_locs[1]
    axes[0, 1].semilogy(out["temperature"], out["pressure"], label="solution")
    if temp_guess is not None:
        axes[0, 1].semilogy(temp_guess, out["pressure"], ls="--", c='b', label="guess")
    axes[0, 1].set_xlim((0, np.max(out["temperature"])))
    axes[0, 1].set_ylim((np.min(out["pressure"]), np.max(out["pressure"])))
    axes[0, 1].scatter([out["temperature"][convective_boundary]], [out["pressure"][convective_boundary]], c="k")
    for (gas, c) in zip(virga_out["condensibles"], cloud_colors):
        _, condt = vj.condensation_t(gas, 1, 2.2, out["pressure"])
        axes[0, 1].semilogy(condt, out["pressure"], ls="--", label=gas, color=c)
    axes[0, 1].set_xlabel("Temperature (K)")
    axes[0, 1].set_ylabel("Pressure (bar)")
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend()

    axes[1, 0].loglog(np.abs(out["fnet/fnetir"]), out["pressure"])
    axes[1, 0].set_xlabel("Fnet/Fnet-IR")
    axes[1, 0].set_ylabel("Pressure (bar)")
    axes[1, 0].axvline(1e-3, ls="--", color="k")
    axes[1, 0].invert_yaxis()

    T_B = brightness_temperature(out["spectrum_output"], plot=False)
    axes[1, 1].semilogx(1e4/out["spectrum_output"]["wavenumber"], T_B)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axhline(np.max(out["temperature"]), ls="--", c='k')
    axes[1, 1].set_xlabel("Wavelength (micron)")
    axes[1, 1].set_ylabel("Brightness temperature (K)")
    
    axes[2, 0].loglog(virga_out["opd_per_layer"][:,55], layer_p)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim((1e-10, 2 * np.max(out["all_opd"][-(N-1):])))
    axes[2, 0].set_xlabel("Optical depth")
    axes[2, 0].set_ylabel("Pressure (bar)")

    axes[2, 1].semilogy(out["dtdp"], layer_p)
    axes[2, 1].semilogy(grad, layer_p)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlabel("dtdp (K/bar)")
    axes[2, 1].set_ylabel("Pressure (bar)")
    ax2 = axes[2, 1].twinx()
    ax2.set_ylabel("Layer number")
    ax2.set_yticks(np.arange(0, N-1, 10))
    ax2.set_yticklabels(np.arange(0, N-1, 10)[::-1])
    
    plt.tight_layout()    
    if fname is not None:
        plt.savefig(fname)
