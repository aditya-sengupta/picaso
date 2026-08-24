# %%
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from calculate_t10 import t10
from matplotlib import cm

temps = np.arange(100, 2401, 50)
bobcat_gravs = np.array([10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])
# bobcat_gravs = np.array([100, 178, 316, 1000, 1780, 3160])

semimajor = 0.5
if semimajor > 0:
    semimajor_str = f"semimajor{semimajor:.2f}"
else:
    semimajor_str = "ns"

cmap = cm.magma(np.linspace(0, 1, len(bobcat_gravs) + 1))[:-1]

__refdata__ = os.environ["picaso_refdata"]
picaso_path = os.path.dirname(__refdata__)
if not picaso_path.endswith("picaso"):
    picaso_path = os.path.dirname(picaso_path)

bobcat_path = os.path.join(__refdata__, "sonora_grids", "bobcat", "structures_m+0.0")

for (c, grav) in zip(cmap, bobcat_gravs):
    track = []
    track_unified = []
    found_temps = []
    found_temps_bobcat = []
    for teff in temps:
        bobcat_model_path = os.path.join(bobcat_path, f"t{teff}g{grav}nc_m0.0.dat")
        if os.path.exists(bobcat_model_path):
            pressure_bobcat,temp_bobcat = np.loadtxt(bobcat_model_path,
                                usecols=[1,2],unpack=True, skiprows = 1)
            found_temps_bobcat.append(teff)
            track.append(t10(pressure_bobcat, temp_bobcat))
        grav_str = grav
        fname_unified = os.path.join(picaso_path, "data", "unified_restart", f"unified_tint{teff}_grav{grav_str}_{semimajor_str}_nc.h5")
        if os.path.exists(fname_unified):
            found_temps.append(teff)
            with h5py.File(fname_unified) as f:
                track_unified.append(f.attrs["t10"])
                # track_unified.append(t10(np.array(f["pressure"]), np.array(f["temperature"])))

    plt.plot(found_temps_bobcat, track, c=c)
    plt.plot(found_temps, track_unified, c=c, label=grav, ls="--")
plt.legend(loc="lower right")
plt.xlabel("Tint (K)")
plt.ylabel("T10 (K)")
plt.ylim((0, 5000))
plt.title(f"my grid dashed ({semimajor = }), Bobcat solid")

# %%
