# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from picaso import justdoit as jdi
from astropy import units as u

rcb_key = 46

output_path = '../data/solution_family/outs.pkl'
with open(output_path, 'rb') as f:
    outputs = pickle.load(f)

outputs_new = {}
for k in outputs.keys():
    if outputs[k]['converged']:
        outputs_new[k] = outputs[k]
outputs = outputs_new

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
for (rcb_key, c) in zip([46, 73], ['r', 'g']):
    ax1.semilogy(outputs[rcb_key]['temperature'], outputs[rcb_key]['pressure'], label=rcb_key, color=c)
ax1.invert_yaxis()
ax1.set_xlabel("Temperature (K)")
ax1.set_ylabel("Pressure (bar)")
ax1.legend()

ax2 = axes[1]
for (rk, c) in zip([46, 73], ['r', 'g']):
    a = outputs[rk]['ptchem_df'].copy()
    case.atmosphere(df=a)
    res = case.spectrum(opacity_ck, full_output=True, calculation='thermal')
    lv = res['full_output']['level']
    net_rk = (np.sum(lv['thermal_fluxes']['flux_plus'],  axis=1) -
              np.sum(lv['thermal_fluxes']['flux_minus'], axis=1))
    ax2.scatter([net_rk[rk]], [outputs[rk]['pressure'][rk]], color=c, zorder=5, label=f'rcb={rk}')
    ax2.loglog(np.abs(net_rk), outputs[rk]['pressure'], color=c)

ax2.axvline(blackbody_energy, ls="--", color='k')
ax2.invert_yaxis()
ax2.set_xlabel("Net Flux")
ax2.set_ylabel("Pressure (bar)")
ax2.legend()

plt.tight_layout()
plt.show()
# %%
# Load cached level fluxes from inspect_solution_family.py
with open('../data/solution_family/level_results.pkl', 'rb') as f:
    level_results = pickle.load(f)

sorted_keys = sorted(outputs.keys())
max_key = int(max(sorted_keys))

# Build matrix: rows = rcb_key, cols = level index (0..max_key-1), NaN outside RZ
pct_matrix = np.full((len(sorted_keys), max_key), np.nan)
for row, key in enumerate(sorted_keys):
    lv = level_results[key]['full_output']['level']
    net_k = (np.sum(lv['thermal_fluxes']['flux_plus'],  axis=1) -
             np.sum(lv['thermal_fluxes']['flux_minus'], axis=1))
    pct_matrix[row, :int(key)] = (net_k[:int(key)] - blackbody_energy) / blackbody_energy * 100

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, (ax, ax_mean) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

# x = RCB index, y = level index
pressure_grid = outputs[sorted_keys[0]]['pressure']
rcb_indices = np.array([int(k) for k in sorted_keys])
rcb_pressures = np.array([float(pressure_grid[int(k)]) for k in sorted_keys])
level_pressures = pressure_grid[:max_key]

# Transpose pct_matrix: rows=level index, cols=RCB key
clim = np.nanpercentile(np.abs(pct_matrix), 95)
im = ax.pcolormesh(rcb_indices, level_pressures,
                   pct_matrix.T, cmap='RdBu_r', vmin=-clim, vmax=clim)

# Attach colorbar to a right-side axes that won't affect layout alignment
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.08)
fig.colorbar(im, cax=cax, label='% deviation from blackbody flux')

# Matching spacer on the mean panel so x extents line up exactly
divider_mean = make_axes_locatable(ax_mean)
cax_mean = divider_mean.append_axes('right', size='4%', pad=0.08)
cax_mean.set_visible(False)

ax.set_yscale('log')
ax.invert_yaxis()
ax.set_ylabel('Pressure [bar]', fontsize=12)
ax.set_title('Net flux % deviation from $\\sigma T_{\\rm eff}^4$ in radiative zone',
             fontsize=12)
ax.plot(rcb_indices, rcb_pressures, 'k--', lw=1.5, label='Radiative zone boundary')

mean_pct = np.nanmean(pct_matrix, axis=1)
best_idx = int(sorted_keys[np.argmin(np.abs(mean_pct))])
ax.axvline(best_idx, ls='--', color='g', label='Best flux balance')
ax.axvline(np.max(sorted_keys), ls='--', color='r', label='PICASO-declared solution')
ax.legend(fontsize=10, loc='lower left')

# Mean panel
threshold = abs(mean_pct[np.argmax(rcb_indices)])
is_better = np.abs(mean_pct) <= threshold
ax_mean.plot(rcb_indices, mean_pct, '-', color='steelblue', lw=2, zorder=1)
ax_mean.scatter(rcb_indices[~is_better], mean_pct[~is_better], color='steelblue', s=20, zorder=2)
ax_mean.scatter(rcb_indices[is_better],  mean_pct[is_better],  color='orange',    s=20, zorder=3)
ax_mean.axhline(0, color='k', lw=0.8, ls=':')
ax_mean.axvline(best_idx, ls='--', color='g')
ax_mean.axvline(np.max(sorted_keys), ls='--', color='r')
ax_mean.set_xlabel('RCB index', fontsize=12)
ax_mean.set_ylabel('Mean % dev.', fontsize=12)

plt.tight_layout()
plt.savefig("../figures/solution_family/flux_deviation.png")
plt.show()
# %%
