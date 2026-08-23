# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
from calculate_t10 import t10
# %%
tint_s = 500
grav = 10
semimajor = 0.5
for tint in [tint_s - 50, tint_s, tint_s + 50]:
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
