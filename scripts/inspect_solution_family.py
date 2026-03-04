# %%
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
from collections import namedtuple
from picaso.grad import did_grad_cp
import json

# ── Adiabatic gradient tables (loaded once) ───────────────────────────────────
__refdata__ = os.environ['picaso_refdata']
cp_grad = json.load(open(os.path.join(__refdata__, 'climate_INPUTS',
                                      'specific_heat_p_adiabat_grad.json')))
AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad', 'cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat']),
)

# ── Load solution_family ──────────────────────────────────────────────────────
with open('data/solution_family/outs.pkl', 'rb') as f:
    solution_family = pickle.load(f)

sorted_keys = sorted(solution_family.keys())

# ── Part 1: Static T-P plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.cm.viridis_r
colors = cmap(np.linspace(0, 1, len(sorted_keys)))

for color, key in zip(colors, sorted_keys):
    T, P = solution_family[key]['temperature'], solution_family[key]['pressure']
    ax.semilogy(T, P, color=color, lw=1.5, alpha=0.8)

ax.set_ylim(500, 1e-4)
ax.set_xlim(0, 3000)
ax.set_xlabel('Temperature [K]', fontsize=14)
ax.set_ylabel('Pressure [bar]', fontsize=14)
ax.set_title('Solution family: T-P profiles\n(dot = RCB level)', fontsize=13)
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(int(sorted_keys[0]), int(sorted_keys[-1])))
plt.colorbar(sm, ax=ax, label='RCB level index')

# Secondary y-axis: level number
pressure_grid = solution_family[sorted_keys[0]]['pressure']  # same for all
nlevel = len(pressure_grid)
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(ax.get_ylim())
# Place a tick at every pressure level; label every 5th
tick_pressures = pressure_grid          # shape (91,)
tick_labels = [str(i) if i % 5 == 0 else '' for i in range(nlevel)]
ax2.set_yticks(tick_pressures)
ax2.set_yticklabels(tick_labels, fontsize=8)
ax2.yaxis.set_minor_locator(plt.NullLocator())

plt.tight_layout()
plt.savefig('figures/solution_family/solution_family_tp.png', dpi=150)
plt.close(fig)
print('Saved T-P plot.')

# ── Part 2: Mirror Level_Fluxes.py setup exactly ─────────────────────────────
mh = '+000'
CtoO = '100'
ck_db_path = ck_db = os.path.join(os.getenv('picaso_refdata'), 'opacities', 'preweighted',
                                   f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'), 'sonora_grids', 'bobcat')

opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

stellar_temp   = 6000
stellar_met    = 0.0
stellar_logg   = 4.0
orbital_sep    = 500
stellar_radius = 1
blackbody_temp = 1000
Teff = 1000
grav = 100000  # cm/s^2  (= 1000 m/s^2 from solution_family, log g = 5.0)

level_cache_path = 'data/solution_family/level_results.pkl'

if os.path.isfile(level_cache_path):
    print(f'Loading cached level fluxes from {level_cache_path}')
    with open(level_cache_path, 'rb') as f:
        level_results = pickle.load(f)
else:
    level_results = {}
    for key in sorted_keys:
        print(f'  Computing level fluxes for rcb_key={key} ...')
        case = jdi.inputs()
        case.approx(get_lvl_flux=True)
        case.phase_angle(0)
        case.gravity(gravity=grav, gravity_unit=u.cm / u.s**2)
        case.star(opacity_ck, stellar_temp, stellar_met, stellar_logg,
                  radius=stellar_radius, radius_unit=jdi.u.Unit('R_sun'),
                  semi_major=orbital_sep, semi_major_unit=jdi.u.Unit('au'))
        # Redundant reload (mirroring Level_Fluxes.py): load sonora first, then
        # reload with the converged profile from solution_family
        case.sonora(sonora_profile_db, Teff, chem='grid')
        atm = solution_family[key]['ptchem_df']
        case.atmosphere(df=atm)
        result = case.spectrum(opacity_ck, full_output=True, calculation='thermal')
        level_results[key] = result
    with open(level_cache_path, 'wb') as f:
        pickle.dump(level_results, f)
    print(f'Cached level fluxes to {level_cache_path}')

print('All level flux computations done.')

# ── Part 3: Precompute integrated fluxes ─────────────────────────────────────
all_plus, all_minus, all_pres_lv = {}, {}, {}
for key in sorted_keys:
    lv = level_results[key]['full_output']['level']
    all_plus[key]    = np.sum(lv['thermal_fluxes']['flux_plus'],  axis=1)
    all_minus[key]   = np.sum(lv['thermal_fluxes']['flux_minus'], axis=1)
    all_pres_lv[key] = lv['pressure']

all_flux_vals = np.concatenate([
    np.concatenate([all_plus[k], all_minus[k]]) for k in sorted_keys
])
all_flux_pos = all_flux_vals[all_flux_vals > 0]
flux_xmin = float(all_flux_pos.min()) * 0.5
flux_xmax = float(all_flux_pos.max()) * 2.0

all_dtdp_vals = np.concatenate([solution_family[k]['dtdp'] for k in sorted_keys])
dtdp_xmin = float(np.nanmin(all_dtdp_vals)) * 1.2
dtdp_xmax = float(np.nanmax(all_dtdp_vals)) * 1.2
# p_layer is the same for every key (shared pressure grid)
_P0 = solution_family[sorted_keys[0]]['pressure']
p_layer_anim = np.sqrt(_P0[:-1] * _P0[1:])

# Precompute adiabatic gradient at layer T/P for each key
adiabat_grad = {}
for key in sorted_keys:
    T = solution_family[key]['temperature']
    T_layer = np.sqrt(T[:-1] * T[1:])
    adiabat_grad[key] = np.array([
        did_grad_cp(float(t), float(p), AdiabatBundle)[0]
        for t, p in zip(T_layer, p_layer_anim)
    ])
all_adiabat_vals = np.concatenate(list(adiabat_grad.values()))
dtdp_xmax = max(dtdp_xmax, float(np.nanmax(all_adiabat_vals)) * 1.2)

# ── Part 4: Animation ─────────────────────────────────────────────────────────
fig, (ax_tp, ax_dtdp, ax_flux) = plt.subplots(1, 3, figsize=(15, 5))

# ── T-P panel ────
for key in sorted_keys:
    T, P = solution_family[key]['temperature'], solution_family[key]['pressure']
    ax_tp.semilogy(T, P, color='lightgray', lw=0.8, alpha=0.5, zorder=0)

anim_keys = list(reversed(sorted_keys))
key0 = anim_keys[0]
T0, P0 = solution_family[key0]['temperature'], solution_family[key0]['pressure']
(line_T,)   = ax_tp.semilogy(T0, P0, color='steelblue', lw=2, label='current', zorder=2)
(rcb_dot,)  = ax_tp.semilogy([T0[key0]], [P0[key0]], 'ro', ms=10, label='RCB level', zorder=5)
ax_tp.set_ylim(500, 1e-4)
ax_tp.set_xlim(0, 3000)
ax_tp.set_xlabel('Temperature [K]', fontsize=14)
ax_tp.set_ylabel('Pressure [bar]', fontsize=14)
ax_tp.set_title(f'T-P profile  (rcb_key={key0})', fontsize=12)
ax_tp.legend(fontsize=11)

# ── dT/dP panel ──
for key in sorted_keys:
    ax_dtdp.plot(solution_family[key]['dtdp'], p_layer_anim,
                 color='lightgray', lw=0.8, alpha=0.5, zorder=0)
dtdp0 = solution_family[key0]['dtdp']
(line_dtdp,) = ax_dtdp.plot(dtdp0, p_layer_anim, color='darkorange', lw=2, zorder=2, label=r'd$\ln T$/d$\ln P$')
(line_adiabat,) = ax_dtdp.plot(adiabat_grad[key0], p_layer_anim, color='green', lw=2, ls='--', zorder=2, label='Adiabatic gradient')
ax_dtdp.axvline(0, color='k', lw=0.8, ls=':')
ax_dtdp.set_yscale('log')
ax_dtdp.invert_yaxis()
ax_dtdp.set_xlim(dtdp_xmin, dtdp_xmax)
ax_dtdp.set_ylim(500, 1e-4)
ax_dtdp.set_xlabel(r'd$\ln T$/d$\ln P$', fontsize=14)
ax_dtdp.set_ylabel('Pressure [bar]', fontsize=14)
ax_dtdp.set_title('Temperature gradient', fontsize=12)
ax_dtdp.legend(fontsize=10)

# ── Flux panel ───
pres0   = all_pres_lv[key0]
tplus0  = all_plus[key0]
tminus0 = all_minus[key0]
fb0     = solution_family[key0]['flux_balance']['flux_net_ir']

(line_plus,)  = ax_flux.plot(tplus0,  pres0, color='red',    lw=2, label=r'Thermal $F^+$')
(line_minus,) = ax_flux.plot(tminus0, pres0, color='blue',   lw=2, label=r'Thermal $F^-$')
(line_net_r,) = ax_flux.plot(tplus0 - tminus0, pres0, color='purple', lw=2, label='Net flux')
ax_flux.axvline(5.67e-5 * blackbody_temp**4, color='orange', ls='--',
                label=f'Thermal test ({blackbody_temp} K)')
ax_flux.set_yscale('log')
ax_flux.invert_yaxis()
ax_flux.set_xscale('log')
ax_flux.set_xlim(flux_xmin, flux_xmax)
ax_flux.set_ylim(500, 1e-4)
ax_flux.set_xlabel(r'Layer Flux (erg/cm$^2$/s)', fontsize=14)
ax_flux.set_ylabel('Pressure (bar)', fontsize=14)
ax_flux.set_title('Level flux balance', fontsize=12)
ax_flux.legend(fontsize=9, loc='best')

title_text = fig.suptitle(f'rcb_key = {key0}', fontsize=13)
fig.tight_layout()


def animate(frame_idx):
    key = anim_keys[frame_idx]
    T, P = solution_family[key]['temperature'], solution_family[key]['pressure']
    line_T.set_xdata(T)
    line_T.set_ydata(P)
    rcb_dot.set_xdata([T[key]])
    rcb_dot.set_ydata([P[key]])
    ax_tp.set_title(f'T-P profile  (rcb_key={key})', fontsize=12)

    pres = all_pres_lv[key]
    tp   = all_plus[key]
    tm   = all_minus[key]
    line_plus.set_xdata(tp);   line_plus.set_ydata(pres)
    line_minus.set_xdata(tm);  line_minus.set_ydata(pres)
    net = np.where(tp - tm > 0, tp - tm, np.nan)
    line_net_r.set_xdata(net); line_net_r.set_ydata(pres)
    fb = solution_family[key]['flux_balance']['flux_net_ir']
    line_dtdp.set_xdata(solution_family[key]['dtdp'])
    line_adiabat.set_xdata(adiabat_grad[key])
    title_text.set_text(f'rcb_key = {key}')
    return line_T, rcb_dot, line_plus, line_minus, line_net_r, line_dtdp, line_adiabat, title_text


ani = animation.FuncAnimation(fig, animate, frames=len(anim_keys),
                               interval=500, blit=False)
plt.close(fig)
ani.save('figures/solution_family/solution_family_convergence.mp4',
         writer='ffmpeg', fps=4)
print('Animation saved.')