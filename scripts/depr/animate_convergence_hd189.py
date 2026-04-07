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



def animate_file(pkl_path, out_dir, overwrite=False):
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    out_mp4 = os.path.join(out_dir, f"animate_{stem}.mp4")
    if os.path.isfile(out_mp4) and not overwrite:
        print(f"  SKIP (exists): {os.path.basename(out_mp4)}")
        return

    print(f"  Processing: {stem}")
    out = pkl.load(open(pkl_path, 'rb'))

    if 'all_profiles' not in out:
        print("    SKIP: no all_profiles in pkl")
        return

    pressure = out['pressure']
    nlevel = len(pressure)
    all_profiles = out['all_profiles']
    nstep_T = len(all_profiles) // nlevel
    T_steps = all_profiles.reshape(nstep_T, nlevel)
    p_layer = np.sqrt(pressure[:-1] * pressure[1:])

    # Pre-compute convective zones for every T step
    cvz_all = [convective_pressure_bounds(T_steps[i], pressure, p_layer)
               for i in range(nstep_T)]

    fig, ax_T = plt.subplots(1, 1, figsize=(6, 6))
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
        return line_T,

    ani = animation.FuncAnimation(fig, animate, frames=nstep_T, interval=80, blit=False)
    plt.close(fig)
    os.makedirs(out_dir, exist_ok=True)
    ani.save(out_mp4, writer='ffmpeg', fps=10)
    print(f"    Saved: {os.path.basename(out_mp4)}")


# ── Main loop ─────────────────────────────────────────────────────────────────
pkl_dir = os.path.join(picaso_path, "data/hd189")
out_dir = os.path.join(picaso_path, "figures/hd189/convergence_animations")

pkl_files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
print(f"Found {len(pkl_files)} pkl files in {pkl_dir}\n")
for fname in pkl_files:
    animate_file(os.path.join(pkl_dir, fname), out_dir)
print("\nDone.")
