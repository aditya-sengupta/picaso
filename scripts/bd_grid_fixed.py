import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import virga.justdoit as vj
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from copy import deepcopy
from datetime import datetime

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

cloudmode = "fixed"
nstr_upper = 75
fsed = 2

for semi_major in [0.02, 0.05, np.inf]:
    for teff in [900, 1400, 1900, 2400]:
        print(f"effective temperature = {teff} K")
        
        cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
        grav = 3160 # Gravity of your brown dwarf in m/s/s
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(teff) # input effective temperature
        opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

        if semi_major < np.inf:
            cl_run.star(opacity_ck, filename="data/solspec_picaso.dat", w_unit="um", f_unit="flam", semi_major=semi_major, semi_major_unit = u.AU, radius=1.0, radius_unit=u.R_sun)

        nlevel = 91 # number of plane-parallel levels in your code
        rfacv = 0.5
        
        sonora_df_cloudy = pd.read_csv(f"./data/sonora_diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
        pressure_grid = np.logspace(-6,2,91)
        temp_guess = np.array(sonora_df_cloudy["T"])
        cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
        cl_run.virga(condensates=cloud_species, directory="~/virga/refrind", runmode="fixed", mh=1, fsed=fsed, latent_heat=True)

        try:
            out_fixed = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=False))
            cld_out = out_fixed["virga_output"]
            cloud_outputs = {x: [] for x in ["temperature", "condensate_mmr", "cond_plus_gas_mmr", "cloud_deck"]}
            
            tstamp = datetime.now().isoformat().replace(":", ".")
            with h5py.File(f"data/bd_fixed/convergence_fsed{fsed}_teff{teff}_semimajor_{semi_major}_cloudmode{cloudmode}_dt{tstamp}.h5", "w") as f:
                p_picaso = f.create_dataset("pressure_picaso", data=out_fixed["pressure"])
                p_virga = f.create_dataset("pressure_virga", data=cld_out["pressure"])
                f.create_dataset("nstrs", data=np.array(out_fixed["nstr"]))
                for k in cloud_outputs:
                    f.create_dataset(k, data=cloud_outputs[k])
                f.create_dataset("altitude_virga", data=cld_out["altitude"].shape)
                p_virga.attrs["fsed"] = fsed
                p_virga.attrs["teff"] = teff
                p_virga.attrs["cloud_species"] = cloud_species
                p_virga.attrs["nstr_start"] = nstr_upper
                t = f.create_dataset("temperature_picaso", data=out_fixed["all_profiles"])
        except ValueError:
            with open(f"./data/convh5_fixed/fsed{fsed}_teff{teff}_nstrupper{nstr_upper}.txt", "w") as f:
                f.write(f"inf or NaN error at fsed = {fsed}, teff = {teff}, nstr_upper start = {nstr_upper}")

