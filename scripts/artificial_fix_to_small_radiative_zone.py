# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from picaso import justdoit as jdi
from astropy import units as u

rcb_key = 73

output_path = '../data/solution_family/outs.pkl'
with open(output_path, 'rb') as f:
    outputs = pickle.load(f)

stellar_temp   = 6000
stellar_met    = 0.0
stellar_logg   = 4.0
orbital_sep    = 500
stellar_radius = 1
blackbody_temp = 1000
blackbody_energy = 5.67e-5 * blackbody_temp ** 4

mh = '+000'
CtoO = '100'
grav = 1000
Teff = 1000
ck_db_path = ck_db = os.path.join(os.getenv('picaso_refdata'), 'opacities', 'preweighted',
                                   f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'), 'sonora_grids', 'bobcat')

opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

case = jdi.inputs()
case.approx(get_lvl_flux=True)
case.phase_angle(0)
case.gravity(gravity=grav, gravity_unit=u.m / u.s**2)
case.star(opacity_ck, stellar_temp, stellar_met, stellar_logg,
                  radius=stellar_radius, radius_unit=jdi.u.Unit('R_sun'),
                  semi_major=orbital_sep, semi_major_unit=jdi.u.Unit('au'))
case.sonora(sonora_profile_db, Teff, chem='grid')
atm = outputs[rcb_key]['ptchem_df']

case.atmosphere(df=atm)
result = case.spectrum(opacity_ck, full_output=True, calculation='thermal')
plus = np.sum(result['full_output']['level']['thermal_fluxes']['flux_plus'],  axis=1)
minus = np.sum(result['full_output']['level']['thermal_fluxes']['flux_minus'], axis=1)
net = plus - minus

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax1 = axes[0]
ax1.semilogy(outputs[rcb_key]['temperature'], outputs[rcb_key]['pressure'])
ax1.invert_yaxis()
ax1.set_xlabel("Temperature (K)")
ax1.set_ylabel("Pressure (bar)")

ax2 = axes[1]
ax2.scatter([net[rcb_key]], [outputs[rcb_key]['pressure'][rcb_key]], color='g')
ax2.loglog(net, outputs[rcb_key]['pressure'])
ax2.axvline(blackbody_energy, ls="--", color='k')
ax2.invert_yaxis()
ax2.set_xlabel("Net Flux")
ax2.set_ylabel("Pressure (bar)")

plt.tight_layout()
plt.show()
# %%
# Load cached level fluxes from inspect_solution_family.py
with open('../data/solution_family/level_results.pkl', 'rb') as f:
    level_results = pickle.load(f)

sorted_keys = sorted(level_results.keys())
max_key = int(max(sorted_keys))

# Build matrix: rows = rcb_key, cols = level index (0..max_key-1), NaN outside RZ
pct_matrix = np.full((len(sorted_keys), max_key), np.nan)
for row, key in enumerate(sorted_keys):
    lv = level_results[key]['full_output']['level']
    net_k = (np.sum(lv['thermal_fluxes']['flux_plus'],  axis=1) -
             np.sum(lv['thermal_fluxes']['flux_minus'], axis=1))
    pct_matrix[row, :int(key)] = (net_k[:int(key)] - blackbody_energy) / blackbody_energy * 100

fig, (ax, ax_mean) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={'width_ratios': [3, 1]})
clim = np.nanpercentile(np.abs(pct_matrix), 95)
im = ax.pcolormesh(np.arange(max_key), [int(k) for k in sorted_keys],
                   pct_matrix, cmap='RdBu_r', vmin=-clim, vmax=clim)
plt.colorbar(im, ax=ax, label='% deviation from blackbody flux')
ax.set_xlabel('Level index (0 = top of atmosphere)', fontsize=12)
ax.set_ylabel('RCB index', fontsize=12)
ax.set_title('Net flux % deviation from $\\sigma T_{\\rm eff}^4$ in radiative zone',
             fontsize=12)
ax.plot([int(k) - 0.5 for k in sorted_keys], [int(k) for k in sorted_keys],
        'k--', lw=1.5, label='Radiative zone boundary')
        
# Mark the best solution
best_solution_idx = np.min(sorted_keys) + np.argmin(np.abs(mean_pct))
ax.axhline([best_solution_idx], ls='--', color='g', label="Best flux balance")
ax.legend(fontsize=10)
ax.invert_yaxis()

# Mean deviation over the whole radiative zone per RCB key
mean_pct = np.nanmean(pct_matrix, axis=1)
ax_mean.plot(mean_pct, [int(k) for k in sorted_keys], 'o-', color='steelblue', lw=2, ms=3)
ax_mean.axvline(0, color='k', lw=0.8, ls=':')
ax_mean.set_xlabel('Mean % deviation', fontsize=12)
ax_mean.set_title('Mean flux deviation', fontsize=12)
ax_mean.set_yticklabels([])
ax_mean.yaxis.set_tick_params(length=0)
ax_mean.set_ylim(ax.get_ylim())

plt.tight_layout()
plt.savefig("../figures/solution_family/flux_deviation.png")
plt.show()
# %%
