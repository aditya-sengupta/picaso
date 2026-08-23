# %%
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from calculate_t10 import t10
from matplotlib import cm

bobcat_temps = np.arange(100, 401, 50)
bobcat_gravs = np.array([10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])
# bobcat_gravs = np.array([100, 178, 316, 1000, 1780, 3160])

semimajor = 0.02
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
    for teff in bobcat_temps:
        #pressure_bobcat,temp_bobcat = np.loadtxt(os.path.join(
        #                    bobcat_path, f"t{teff}g{grav}nc_m0.0.dat"),
        #                    usecols=[1,2],unpack=True, skiprows = 1)
        #track.append(t10(pressure_bobcat, temp_bobcat))
        if grav == 178:
            grav_str = "177"
        elif grav == 1780:
            grav_str = "1778"
        else:
            grav_str = grav
        fname_unified = os.path.join(picaso_path, "data", "unified_restart", f"unified_tint{teff}_grav{grav_str}_{semimajor_str}_nc.h5")
        if os.path.exists(fname_unified):
            found_temps.append(teff)
            with h5py.File(fname_unified) as f:
                track_unified.append(f.attrs["t10"])
                # track_unified.append(t10(np.array(f["pressure"]), np.array(f["temperature"])))

    # plt.plot(bobcat_temps, track, c=c, alpha=0.5)
    plt.plot(found_temps, track_unified, c=c, label=grav)
plt.legend(loc="lower right")
plt.xlabel("Tint (K)")
plt.ylabel("T10 (K)")
plt.title(f"my grid dashed ({semimajor = }), Bobcat solid")

# %%
