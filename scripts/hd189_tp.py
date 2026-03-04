import pickle as pkl
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')
import picaso
from picaso import justdoit as jdi
from picaso.justplotit import pt_adiabat, brightness_temperature
from picaso.grad import did_grad_cp
from astropy import units as u
__refdata__ = os.environ['picaso_refdata']

picaso_path = os.path.dirname(picaso.__path__[0])

tint = 200
grav = 10 ** 1.4
semi_major = 0.03126
r_star = float(((semi_major * u.au) / 9.1) / u.R_sun)
all_feh = ["-100", "+000", "+030", "+050", "+070", "+100"]
all_co = ["025", "050", "100", "150", "200"]
all_rfacv = ["0.5", "0.65", "0.75", "0.8", "0.85"]

cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)
nlevel = 91

for mh in all_feh:
    for CtoO in all_co:
        for rfacv in all_rfacv:
            case_label = f"feh{mh}_co{CtoO}_rfacv{float(rfacv):.2f}"
            ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
            fname = os.path.join(picaso_path, f"data/hd189/{case_label}.pkl")
            if not os.path.isdir(ck_db):
                print(f"Skipping feh={mh} co={CtoO} rfacv={rfacv}: opacity not found")
                continue
            if os.path.isfile(fname):
                print(f"Skipping feh={mh} co={CtoO} rfacv={rfacv}: run already done")
                continue

            tp_file = os.path.join(picaso_path, f"hd189_tp_gagnebin/tp_feh{mh}_tint200_co{CtoO}_rfacv{rfacv}.txt")

            print(f"Running {case_label}...")

            tp = np.genfromtxt(tp_file)
            pressure, temperature = tp[:,0], tp[:,1]
            grad_tp = np.diff(np.log(temperature)) / np.diff(np.log(pressure))
            layer_pressure = np.sqrt(pressure[1:] * pressure[:-1])
            layer_temperature = np.sqrt(temperature[1:] * temperature[:-1])
            adiabatic_grad = np.array([did_grad_cp(t, p, AdiabatBundle)[0] for (t, p) in zip(layer_temperature, layer_pressure)])
            rcb_guess = np.max(np.where(grad_tp < 0.98 * adiabatic_grad)[0])

            cl_run = jdi.inputs(calculation="planet", climate=True)
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
            cl_run.effective_temp(tint)
            opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

            cl_run.star(opacity_ck, temp=5012.5,metal=0.03, logg=4.57, radius = r_star,database='phoenix', radius_unit=u.R_sun,semi_major=semi_major , semi_major_unit = u.AU)

            temp_guess = np.ones((nlevel,)) * 1500
            cl_run.inputs_climate(temp_guess=np.copy(temp_guess), pressure=pressure, rcb_guess=rcb_guess, rfacv=float(rfacv))
            out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
            _, grad, _ = pt_adiabat(out, cl_run, opacity_ck, plot=False)
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))

            layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
            N = len(out["pressure"])

            cvz_locs = out["cvz_locs"]
            if cvz_locs[-2] > 0 and cvz_locs[2] != 89:
                convective_boundary = cvz_locs[-2]
            else:
                convective_boundary = cvz_locs[1]
            axes[0, 0].semilogy(out["temperature"], out["pressure"], label="solution")
            max_temp = np.max(out["temperature"])
            axes[0, 0].semilogy(temperature, pressure, label="Anna's grid")
            max_temp = max(max_temp, np.max(temperature))
            if temp_guess is not None:
                axes[0, 0].semilogy(temp_guess, out["pressure"], ls="--", c='b', label="guess")
                max_temp = max(max_temp, np.max(temp_guess))
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

            T_B = brightness_temperature(out["spectrum_output"], plot=False)
            axes[1, 1].semilogx(1e4/out["spectrum_output"]["wavenumber"], T_B)
            axes[1, 1].invert_yaxis()
            axes[1, 1].axhline(np.max(out["temperature"]), ls="--", c='k')
            axes[1, 1].set_xlabel("Wavelength (micron)")
            axes[1, 1].set_ylabel("Brightness temperature (K)")

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
            plt.savefig(os.path.join(picaso_path, "figures", "hd189", f"{case_label}.png"))
            plt.close(fig)

            pkl.dump(out, open(fname, 'wb'))

# %%
