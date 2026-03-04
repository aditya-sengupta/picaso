# %%
import json
import re
import pickle as pkl
import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import picaso
from picaso.grad import did_grad_cp

import sys
sys.path.append(os.path.join(os.path.dirname(picaso.__path__[0]), "scripts"))
from load_diamondback import read_diamondback_optical_properties

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.environ['picaso_refdata']

# ── Adiabatic gradient tables (loaded once) ───────────────────────────────────
cp_grad = json.load(open(os.path.join(__refdata__, 'climate_INPUTS',
                                      'specific_heat_p_adiabat_grad.json')))
AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad', 'cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat']),
)


def parse_pkl_name(stem):
    """Parse teff/grav/fsed/cloudmode/semi_major from filename stem."""
    p = {}
    m = re.search(r'cloudmode(\w+?)(?:_|$)', stem)
    p['cloudmode'] = m.group(1) if m else 'unknown'
    m = re.search(r'fsed(\d+)', stem)
    p['fsed'] = int(m.group(1)) if m else 2
    m = re.search(r'teff(\d+)', stem)
    p['teff'] = int(m.group(1)) if m else None
    m = re.search(r'grav(\d+)', stem)
    p['grav'] = int(m.group(1)) if m else None
    m = re.search(r'semimajor([\d.]+)', stem)
    p['semi_major'] = float(m.group(1)) if m else np.inf
    return p


def convective_pressure_bounds(T_profile, pressure, p_layer):
    """Return list of (p_top, p_bot) tuples where dtdp >= 0.98 * adiabatic_grad."""
    T_layer = np.sqrt(T_profile[:-1] * T_profile[1:])
    grad_tp = np.diff(np.log(np.maximum(T_profile, 1e-10))) / np.diff(np.log(pressure))
    adiabatic_grad = np.array([did_grad_cp(float(t), float(p), AdiabatBundle)[0]
                                for t, p in zip(T_layer, p_layer)])
    is_conv = grad_tp >= 0.98 * adiabatic_grad
    zones, in_zone = [], False
    for k in range(len(is_conv)):
        if is_conv[k] and not in_zone:
            in_zone, i_top = True, k
        elif not is_conv[k] and in_zone:
            in_zone = False
            zones.append((p_layer[i_top], p_layer[k - 1]))
    if in_zone:
        zones.append((p_layer[i_top], p_layer[-1]))
    return zones


def get_static_opd(out, params):
    """
    OPD per layer at ~4 um (wno index 55), or None.
    Priority: fixed_opd in pkl -> Diamondback reconstruction -> None.
    """
    if 'fixed_opd' in out:
        return np.array(out['fixed_opd'])[:, 55]
    if np.isinf(params['semi_major']):
        teff, grav, fsed = params['teff'], params['grav'], params['fsed']
        cld = os.path.join(picaso_path, 'reference', 'sonora_grids',
                           'diamondback_alloutputs',
                           f"t{teff}g{grav}f{fsed}_m0.0_co1.0.cld")
        if os.path.isfile(cld):
            db = read_diamondback_optical_properties(teff, grav, fsed)
            return np.array(db['tau']).reshape((90, 196))[:, 55]
    return None


def animate_file(pkl_path, out_dir, overwrite=False):
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    out_mp4 = os.path.join(out_dir, f"animate_{stem}.mp4")
    if os.path.isfile(out_mp4) and not overwrite:
        print(f"  SKIP (exists): {os.path.basename(out_mp4)}")
        return

    print(f"  Processing: {stem}")
    params = parse_pkl_name(stem)
    out = pkl.load(open(pkl_path, 'rb'))

    if 'all_profiles' not in out:
        print("    SKIP: no all_profiles in pkl")
        return

    pressure = out['pressure']
    nlevel = len(pressure)
    nlayer = nlevel - 1
    all_profiles = out['all_profiles']
    nstep_T = len(all_profiles) // nlevel
    T_steps = all_profiles.reshape(nstep_T, nlevel)
    p_layer = np.sqrt(pressure[:-1] * pressure[1:])

    # OPD: dynamic (selfconsistent clouds) or static (fixed / reconstructed)
    all_opd_raw = out.get('all_opd', np.array([]))
    nstep_OPD = len(all_opd_raw) // nlayer if len(all_opd_raw) >= nlayer else 0
    opd_dynamic = nstep_OPD > 1 and np.any(all_opd_raw != 0)

    if opd_dynamic:
        opd_steps = all_opd_raw.reshape(nstep_OPD, nlayer)
        def get_opd(i):
            oi = min(int(i * nstep_OPD / nstep_T), nstep_OPD - 1)
            return opd_steps[oi], oi, nstep_OPD - 1
        all_opd_vals = opd_steps.ravel()
    else:
        static_opd = get_static_opd(out, params)
        if static_opd is None:
            print("    WARN: no OPD data; OPD panel will be blank")
        def get_opd(i):
            return static_opd, 0, 0
        all_opd_vals = static_opd if static_opd is not None else np.array([0.])

    # Pre-compute convective zones for every T step
    cvz_all = [convective_pressure_bounds(T_steps[i], pressure, p_layer)
               for i in range(nstep_T)]

    fig, (ax_T, ax_opd) = plt.subplots(1, 2, figsize=(12, 6))
    T_xmax = float(np.nanmax(T_steps)) * 1.15

    ax_T.set_xlabel("Temperature [K]", fontsize=14)
    ax_T.set_ylabel("Pressure [bar]", fontsize=14)
    ax_T.set_yscale("log")
    ax_T.invert_yaxis()
    ax_T.set_xlim(0, T_xmax)
    ax_T.set_ylim(float(max(pressure)), float(min(pressure)))
    ax_T.plot(T_steps[0], pressure, color="gray", lw=1, ls="--", label="initial", zorder=0)
    line_T, = ax_T.plot(T_steps[0], pressure, color="steelblue", lw=2, zorder=2)
    ax_T.set_title(f"{stem}\nStep 0", fontsize=8)
    ax_T.legend(fontsize=10)

    ax_opd.set_xlabel("OPD at ~4 um (per layer)", fontsize=14)
    ax_opd.set_ylabel("Pressure [bar]", fontsize=14)
    ax_opd.set_xscale("log")
    ax_opd.set_yscale("log")
    ax_opd.invert_yaxis()
    opd_pos = all_opd_vals[all_opd_vals > 0]
    ax_opd.set_xlim(float(opd_pos.min()) * 0.1 if len(opd_pos) else 1e-7,
                    float(opd_pos.max()) * 5   if len(opd_pos) else 1.0)
    ax_opd.set_ylim(float(max(p_layer)), float(min(p_layer)))
    first_opd, _, _ = get_opd(0)
    init_opd = np.maximum(first_opd, 1e-30) if first_opd is not None else np.full(nlayer, 1e-30)
    line_opd, = ax_opd.plot(init_opd, p_layer, color="darkorange", lw=2)
    title_opd = ax_opd.set_title(
        "OPD (dynamic)" if opd_dynamic else "OPD (static/reconstructed)", fontsize=11)

    fig.tight_layout()
    cvz_patches = []

    def animate(i):
        nonlocal cvz_patches
        ax_T.set_title(f"{stem}\nStep {i}/{nstep_T - 1}", fontsize=8)
        line_T.set_xdata(T_steps[i])
        for patch in cvz_patches:
            patch.remove()
        cvz_patches = []
        for (p_top, p_bot) in cvz_all[i]:
            cvz_patches.append(
                ax_T.fill_betweenx([p_top, p_bot], 0, T_xmax,
                                   color="green", alpha=0.20, zorder=1))
        opd_vals, oi, oi_max = get_opd(i)
        if opd_vals is not None:
            line_opd.set_xdata(np.maximum(opd_vals, 1e-30))
        if opd_dynamic:
            title_opd.set_text(f"OPD step {oi}/{oi_max}")
        return line_T, line_opd, title_opd

    ani = animation.FuncAnimation(fig, animate, frames=nstep_T, interval=80, blit=False)
    plt.close(fig)
    os.makedirs(out_dir, exist_ok=True)
    ani.save(out_mp4, writer='ffmpeg', fps=10)
    print(f"    Saved: {os.path.basename(out_mp4)}")


# ── Main loop ─────────────────────────────────────────────────────────────────
pkl_dir = os.path.join(picaso_path, "data/bd_fixed_2602")
out_dir = os.path.join(picaso_path,
                       "figures/bd_fixed_figures_260227_refactor/convergence_animations")

pkl_files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
print(f"Found {len(pkl_files)} pkl files in {pkl_dir}\n")
for fname in pkl_files:
    animate_file(os.path.join(pkl_dir, fname), out_dir)
print("\nDone.")
