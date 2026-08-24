# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
from calculate_t10 import t10
# %%
# tint_s = 150
grav = 3160
semimajor = 0.02
for tint in np.arange(100, 401, 50):
    with h5py.File(f"../data/unified_restart/unified_tint{tint}_grav{grav}_semimajor{semimajor}_nc.h5") as f:
        nstr_upper_init = int(f.attrs["nstr_upper_init"])
        plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label=tint)
        plt.scatter([f.attrs["t10"]], [10])
    plt.title(f"{grav = }, {tint_s = }, {semimajor = }")
    plt.legend()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.gca().invert_yaxis()
# %%
