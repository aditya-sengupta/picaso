import sys
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
import h5py
from virga import justdoit as vj
import pandas as pd

virga_path = "~/atmospheres/virga"
__refdata__ = os.environ['picaso_refdata']

cloud_species = ["SiO2"]
cloud_colors = ['#CC5555']

picaso_path = os.path.dirname(picaso.__path__[0])

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5

tint = 200
grav = 10 ** 1.4
semi_major = 0.03126
r_star = float(((semi_major * u.au) / 9.1) / u.R_sun)
all_feh_3 = ["-100", "+000", "+030", "+050", "+070", "+100"]
all_co_3 = ["025", "050", "100", "150", "200"]
all_feh_4 = ["-1.0", "0.0", "0.3", "0.5", "0.7", "1.0"]
all_co_4 = ["0.14", "0.27", "0.55", "0.82", "1.10"]
all_rfacv = ["0.8"] # ["0.5", "0.65", "0.75", "0.8", "0.85"]

all_fsed = [0, 1, 2, 3, 4, 8]

cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)
nlevel = 91

for (mh3, mh) in zip(all_feh_3, all_feh_4):
    for (CtoO3, CtoO) in zip(all_co_3, all_co_4):
        for rfacv in all_rfacv:
            for fsed in all_fsed:
                cloudmode = "fixed" if fsed > 0 else "cloudless"
                fsed_str = f"_f{fsed}" if fsed != 0 else ""
                case_label = f"feh{mh}_co{CtoO}_rfacv{float(rfacv):.2f}{fsed_str}"
                print(f"Running {case_label}...")
                ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
                ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem + "_noTiOVO.hdf5")
                fname = os.path.join(picaso_path, f"data/hd189/{case_label}.h5")
                if os.path.isfile(fname):
                    print(f"Skipping feh={mh} co={CtoO} rfacv={rfacv} fsed={fsed}: run already done")
                    continue

                # Bring this back if needed
                # tp_file = os.path.join(picaso_path, f"data/hd189_tp_gagnebin/tp_feh{mh3}_tint200_co{CtoO3}_rfacv{rfacv}.txt")
                # tp = np.genfromtxt(tp_file)
                # pressure, temperature = tp[:,0], tp[:,1]
                with h5py.File(f"data/hd189/feh{mh}_co{CtoO}_rfacv{float(rfacv):.2f}.h5") as f:
                    pressure, temp_cloudless = np.array(f["pressure"]), np.array(f["temperature"])
                grad_tp = np.diff(np.log(temp_cloudless)) / np.diff(np.log(pressure))
                layer_pressure = np.sqrt(pressure[1:] * pressure[:-1])
                layer_temperature = np.sqrt(temp_cloudless[1:] * temp_cloudless[:-1])
                adiabatic_grad = np.array([did_grad_cp(t, p, AdiabatBundle)[0] for (t, p) in zip(layer_temperature, layer_pressure)])
                rcb_guess = np.max(np.where(grad_tp < 0.98 * adiabatic_grad)[0])

                cl_run = jdi.inputs(calculation="planet", climate=True)
                cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
                cl_run.effective_temp(tint)
                opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

                cl_run.star(opacity_ck, temp=5012.5,metal=0.03, logg=4.57, radius = r_star,database='phoenix', radius_unit=u.R_sun,semi_major=semi_major , semi_major_unit = u.AU)

                cl_run.inputs_climate(temp_guess=np.copy(temp_cloudless), pressure=pressure, rcb_guess=rcb_guess, rfacv=float(rfacv))
                if fsed > 0:
                    kz = np.ones_like(pressure) * 1e10 # idk
                    virga_planet = vj.Atmosphere(["SiO2"], fsed=fsed, mh=(10**float(mh)), mmw=2.2)
                    virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
                    virga_planet.ptk(df = pd.DataFrame({'pressure':pressure, 'temperature':temp_cloudless, 'kz': kz}), kz_min=1e5, latent_heat=True)
                    virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
                    cl_run.fix_virga_clouds(virga_out)

                out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
                _, grad, _ = pt_adiabat(out, cl_run, opacity_ck, plot=False)
                fig, axes = plt.subplots(3, 2, figsize=(8, 12))

                layer_p = np.sqrt(out["pressure"][:-1] * out["pressure"][1:])
                N = len(out["pressure"])

                cvz_locs = out["cvz_locs"]
                if cvz_locs[-2] > 0 and cvz_locs[2] != 89:
                    convective_boundary = cvz_locs[-2]
                else:
                    convective_boundary = cvz_locs[1]
                axes[0, 0].semilogy(out["temperature"], out["pressure"], label="solution")
                max_temp = np.max(out["temperature"])
                axes[0, 0].semilogy(temp_cloudless, pressure, label="cloudless")
                max_temp = max(max_temp, np.max(temp_cloudless))
                for (gas_name, gas_color) in zip(cloud_species, cloud_colors):
                    p,t = vj.condensation_t(gas_name, (10**float(mh)), 2.2, pressure=out["pressure"])
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

                if virga_out is not None and all([x in virga_out.keys() for x in ["condensibles", "condensate_mmr"]]):
                    for (i, condensible) in enumerate(virga_out["condensibles"]):
                        axes[2, 0].loglog(virga_out["condensate_mmr"][:,i], virga_out["pressure"], label=condensible, color=cloud_colors[i])
                    axes[2, 0].set_xlim((1e-10, 2 * np.max(virga_out["condensate_mmr"])))
                    axes[2, 0].set_ylim((np.min(out["pressure"]), np.max(out["pressure"])))
                    axes[2, 0].set_xlabel("Condensate mass mixing ratio")
                    axes[2, 0].set_ylabel("Pressure (bar)")
                    axes[2, 0].invert_yaxis()
                    axes[2, 0].legend()

                if virga_out is not None and "opd_per_layer" in virga_out.keys():
                    axes[2, 1].loglog(virga_out["opd_per_layer"][:,55], layer_p)
                    axes[2, 1].invert_yaxis()
                    axes[2, 1].set_xlim((1e-10, 2 * np.max(out["all_opd"][-(N-1):])))
                    axes[2, 1].set_xlabel("Optical depth")
                    axes[2, 1].set_ylabel("Pressure (bar)")


                plt.tight_layout()
                plt.savefig(os.path.join(picaso_path, "figures", "hd189", f"{case_label}.png"))
                plt.close(fig)

                with h5py.File(fname, "w") as f:
                    out_to_hdf5(out, f)

