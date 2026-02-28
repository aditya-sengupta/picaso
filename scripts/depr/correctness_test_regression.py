import pickle as pkl
import os
import logging
import warnings
warnings.filterwarnings('ignore')
import picaso
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# Configure logger to capture update_clouds OPD info
log_dir = os.path.join(os.path.dirname(picaso.__path__[0]), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "correctness_regression_opd.log")
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(message)s'
)

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
picaso_path = os.path.dirname(picaso.__path__[0])

mh = '+000'
CtoO = '100'
ck_db = os.path.join(os.getenv('picaso_refdata'), 'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

nstr_upper = 88
fsed = 2
semi_major = np.inf

out_dir = os.path.join(picaso_path, "figures/userdefinedclouds_comparison")

test_cases = [
    {"teff": 900, "grav": 316},
]

for case in test_cases:
    teff = case["teff"]
    grav = case["grav"]
    cloudmode = "fixed"
    print(f"\n=== Regression test: teff={teff}, grav={grav} ===")

    # 1. Load old output for comparison
    old_fname = os.path.join(picaso_path, f"data/bd_fixed_2602/bd_fsed{fsed}_teff{teff}_grav{grav}_cloudmode{cloudmode}.pkl")
    old_out = pkl.load(open(old_fname, 'rb'))

    # 2. Re-run using the old interface (virga via picaso, selfconsistent-then-fixed)
    cl_run = jdi.inputs(calculation="browndwarf", climate=True)
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
    cl_run.effective_temp(teff)
    opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

    if semi_major < np.inf:
        cl_run.star(opacity_ck, filename="data/solspec_picaso.dat", w_unit="um", f_unit="flam",
                    semi_major=semi_major, semi_major_unit=u.AU, radius=1.0, radius_unit=u.R_sun)

    # Use Diamondback initial guess (matching old runs)
    sonora_df = pd.read_csv(f"reference/sonora_grids/diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    pressure_grid = np.array(sonora_df["P"])
    temp_guess = np.array(sonora_df["T"])

    cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=0.5)
    cl_run.virga(condensates=cloud_species, directory="/Users/adityasengupta/virga/refrind",
                 runmode=cloudmode, mh=1, fsed=fsed, latent_heat=True)
    new_out = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True))

    # 3. Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.semilogy(old_out["temperature"], old_out["pressure"], label="Saved output", lw=2)
    ax.semilogy(new_out["temperature"], new_out["pressure"], label="Re-run (regression)", lw=2, ls="--")
    ax.semilogy(temp_guess, pressure_grid, label="Initial guess (Diamondback)", lw=1, ls=":", c='gray')

    for (gas, c) in zip(cloud_species, ['#CC5555', '#3BA39C', '#CCB84D', '#FF8C00']):
        _, condt = vj.condensation_t(gas, 1, 2.2, old_out["pressure"])
        ax.semilogy(condt, old_out["pressure"], ls="--", label=gas, color=c, alpha=0.5)

    ax.set_xlim((0, max(np.max(old_out["temperature"]), np.max(new_out["temperature"]))))
    ax.set_ylim((np.min(old_out["pressure"]), np.max(old_out["pressure"])))
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bar)")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"Regression: Teff={teff} K, g={grav} m/s², fsed={fsed}, fixed clouds")

    plt.tight_layout()
    fig_fname = os.path.join(out_dir, f"regression_teff{teff}_grav{grav}.png")
    plt.savefig(fig_fname, dpi=150)
    plt.close()
    print(f"  Saved {fig_fname}")
    print(f"  Max |ΔT| = {np.max(np.abs(old_out['temperature'] - new_out['temperature'])):.4f} K")

print(f"\nOPD log written to: {log_file}")
