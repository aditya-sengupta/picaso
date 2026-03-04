# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Get Integrated Level Fluxes For Solar Reflected and Thermal Radiation
#
# Here in this notebook we work on getting the pressure-dependent level fluxes and also comparing these level fluxes with our three methods of getting opacities (resort-rebin, resampled, and preweighted)

# %%
import picaso.justdoit as jdi
import numpy as np
import picaso.justplotit as jpi
from astropy import units as u

# %%
import os
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'# # CtoO absolute ratio
ck_db_path = ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')
sonora_diamondback_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','diamondback')

opacity_ck = jdi.opannection(ck_db=ck_db_path, method='preweighted')

cases = {}
df = {}

# Define some parameters for the simulation
stellar_temp   = 6000
stellar_met    = 0.0
stellar_logg   = 4.0
orbital_sep    = 500 # for some reason it won't let me just remove the star
stellar_radius = 1
rsol = 6.957e8
au   = 1.496e11
blackbody_temp = 1000

#run the get level-fluxes framework for these three different methods
key = 'ck' # or 'mono'
cases[key] = jdi.inputs()

cases[key].approx(get_lvl_flux=True)

# Set common parameters: phase angle and gravity.
cases[key].phase_angle(0)
grav = 3160
cases[key].gravity(gravity=grav, gravity_unit=u.cm/u.s**2)

# Set stellar parameters (same for both)
cases[key].star(calc[key],
                stellar_temp,
                stellar_met,
                stellar_logg,
                radius=stellar_radius,
                radius_unit=jdi.u.Unit('R_sun'),
                semi_major=orbital_sep,
                semi_major_unit=jdi.u.Unit('au'))
Teff = 1000
cases[key].sonora(sonora_profile_db, Teff, chem='grid')
atm = cases[key].inputs['atmosphere']['profile']

# %%
cases[key].atmosphere(df=atm)
df[key] = cases[key].spectrum(calc[key], full_output=True, calculation='thermal')

# %%
# Get the data out of the picaso run
level_out = df[key]['full_output']['level']
thermal_plus = level_out['thermal_fluxes']['flux_plus']
thermal_minus = level_out['thermal_fluxes']['flux_minus']
pressures = level_out['pressure']

# Sum the binned fluxes to get the bolometric values
integrated_thermal_plus = np.sum(thermal_plus, axis=1)
integrated_thermal_minus = np.sum(thermal_minus, axis=1)

# %% [markdown]
# ## Plot the thermal and reflected layer fluxes

# %%
plt=jpi.plt
plt.figure(figsize=(8, 6))

# Plot thermal and reflected fluxes, both up and down
plt.plot(integrated_thermal_plus, pressures, label='Thermal Plus', color='red')
plt.plot(integrated_thermal_minus, pressures, label='Thermal Minus', color='blue')
plt.plot(integrated_thermal_plus - integrated_thermal_minus, pressures, label='Thermal Net', color='purple')

# Add reference flux lines for thermal and starlight
plt.axvline(
         [5.67e-5 * blackbody_temp ** 4],
         label='Thermal Test (1000K)',
         color='orange',
         linestyle='dashed')

plt.gca().invert_yaxis()

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Layer Flux (erg/cm$^2$/s)')
plt.ylabel('Pressure (bar)')
plt.legend(loc='best')

# %%
plt.semilogy(cases[key].inputs['atmosphere']['profile']['temperature'], pressures)
plt.gca().invert_yaxis()
# %%
