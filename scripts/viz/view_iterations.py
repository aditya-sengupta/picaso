# %%
import numpy as np
import h5py
import warnings
warnings.filterwarnings('ignore')
import picaso
import json
from collections import namedtuple
from picaso.grad import did_grad_cp
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.environ['picaso_refdata']

def fsed_str(fsed):
    if fsed == -1:
        return "nc"
    else:
        return "f" + str(fsed)

def semi_major_str(semi_major):
    if semi_major == -1:
        return "ns"
    else:
        return f"semimajor{semi_major:.2f}"

def fname_from_params(grav, tint, semi_major, fsed):
    fname_stem = f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_{fsed_str(fsed)}"
    return os.path.join(picaso_path, "data", "unified", f"{fname_stem}.h5")

cp_grad = json.load(open(os.path.join(__refdata__, 'climate_INPUTS',
                                      'specific_heat_p_adiabat_grad.json')))
AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad', 'cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat']),
)

# %%
grav, tint = 10, 200
semi_major, fsed = 0.5, -1
semi_major_str_plotout = f"semimajor = {semi_major} au" if semi_major > 0 else "no star"
fsed_str_plotout = f"fsed = {fsed}" if fsed > 0 else "no cloud"

all_profiles = []
fname = fname_from_params(grav, tint, semi_major, fsed)
with h5py.File(fname) as f:
    final_temperature = np.array(f["temperature"])
    pressure_grid = np.array(f["pressure"])
    all_profiles_flat = np.array(f["all_profiles"])
    for i in range(all_profiles_flat.size // 91):
        all_profiles.append(all_profiles_flat[91*i:91*(i+1)])

fig, axs = plt.subplots(1, 2, figsize=(8,4))

def update(k):
    axs[0].clear()
    axs[1].clear()
    fig.suptitle(f"{grav = }, {tint = }, {semi_major_str_plotout}, {fsed_str_plotout}, iteration {k}")

    axs[0].semilogy(all_profiles[k], pressure_grid)
    axs[0].invert_yaxis()
    axs[0].set_xlim((0, 5199))
    axs[0].set_ylabel("Pressure (bar)")
    axs[0].set_xlabel("Temperature (K)")

    layer_temperature = np.sqrt(all_profiles[k][1:] * all_profiles[k][:-1])
    layer_pressure = np.sqrt(pressure_grid[1:] * pressure_grid[:-1])
    grad_tp = np.diff(np.log(np.maximum(all_profiles[k], 1e-10))) / np.diff(np.log(pressure_grid))
    adiabat_gradient = [did_grad_cp(t, p, AdiabatBundle)[0] for (t, p) in zip(layer_temperature, layer_pressure)]

    axs[1].semilogy(grad_tp, layer_pressure, label=r"$\nabla$")
    axs[1].semilogy(adiabat_gradient, layer_pressure, label=r"$\nabla_{ad}$")
    axs[1].set_xlim((0, 0.5))
    axs[1].invert_yaxis()
    axs[1].legend()
    axs[1].set_xlabel("Temperature gradient")

ani = FuncAnimation(fig, update, frames=range(len(all_profiles)), interval=50, repeat=True)

out_dir = os.path.join(picaso_path, 'figures', 'unified')
out_mp4 = os.path.join(out_dir, f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_{fsed_str(fsed)}.mp4")
ani.save(out_mp4, writer='ffmpeg', dpi=150)
# %%
