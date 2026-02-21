# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.interpolate import interp1d

# Get all h5 files in the directory
data_dir = Path('/Users/adityasengupta/picaso/data/bd_fixed_2602')
h5_files = sorted(data_dir.glob('*.h5'))

# Read all h5 files
data = {}
for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        data[h5_file.name] = {key: f[key][()] for key in f.keys()}
        
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')
# %%
for grav in [316, 1000, 3160]:
    plt.figure(figsize=(8, 6))
    for (c, teff) in zip(['b', 'k', 'r'], [900, 1400, 1900]):
        pressure_grid,temp_guess = np.loadtxt(os.path.join(
                    sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                    usecols=[1,2],unpack=True, skiprows = 1)
        interp_func = interp1d(pressure_grid, temp_guess, bounds_error=False, fill_value="extrapolate")
        temp_at_pressure_10 = interp_func(10)
        plt.plot(temp_at_pressure_10, 10, 'o', color=c, label="Bobcat T10")
        for (ls, cloudmode) in zip(["--", "-.", "-"], ["fixed", "fixed_bobcatstart", "selfconsistent"]):
            try:
                d = data[f"bd_fsed2_teff{teff}_grav{grav}_cloudmode{cloudmode}.h5"]
                plt.semilogy(d["temperature_picaso"][-91:], d["pressure_picaso"], ls=ls, color=c, label=f"Tint = {teff} K, {cloudmode}")
            except KeyError:
                pass
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (bar)")
    plt.title(f"Surface gravity = {grav} m/s/s")
# %%
for grav in [316, 1000, 3160]:
    plt.figure(figsize=(8, 6))
    for (c, teff) in zip(['b', 'k', 'r'], [900, 1400, 1900]):
        pressure_grid,temp_guess = np.loadtxt(os.path.join(
                    sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                    usecols=[1,2],unpack=True, skiprows = 1)
        for (a, semi_major) in zip([0.8, 0.5, 0.2], [0.1, 0.05, 0.02]):
            try:
                d = data[f"bd_fsed2_teff{teff}_grav{grav}_semimajor{semi_major}_cloudmodefixed.h5"]
                plt.semilogy(d["temperature_picaso"][-91:], d["pressure_picaso"], alpha=a, color=c, label=f"Teff = {teff} K, semimajor = {semi_major} au")
            except KeyError:
                pass
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (bar)")
    plt.title(f"Surface gravity = {grav} m/s/s")
# %%
grav = 316
plt.figure(figsize=(8, 6))
for (c, teff) in zip(['b', 'k', 'r'], [900, 1400, 1900]):
    pressure_grid,temp_guess = np.loadtxt(os.path.join(
                sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                usecols=[1,2],unpack=True, skiprows = 1)
    for (a, semi_major) in zip([0.8, 0.5, 0.2], [0.1, 0.05, 0.02]):
        try:
            d = data[f"bd_fsed2_teff{teff}_grav{grav}_semimajor{semi_major}_cloudmodefixed.h5"]
            plt.semilogy(d["cond_plus_gas_mmr"], d["pressure_virga"], alpha=a, color=c, label=f"Teff = {teff} K, semimajor = {semi_major} au")
        except KeyError:
            pass
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.title(f"Surface gravity = {grav} m/s/s")
# %%
