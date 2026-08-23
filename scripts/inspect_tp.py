# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
from calculate_t10 import t10
# %%

grav = 100
tint = 200
semimajor = 0.02
with h5py.File(f"../data/unified_restart/unified_tint{tint}_grav{grav}_ns_nc.h5") as f:
    plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label="unirr")
with h5py.File(f"../data/unified/unified_tint{tint}_grav{grav}_semimajor{semimajor}_nc.h5") as f:
    plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label="irr, before")
with h5py.File(f"../data/unified_restart/unified_tint{tint}_grav{grav}_semimajor{semimajor}_nc.h5") as f:
    nstr_upper_init = int(f.attrs["nstr_upper_init"])
    plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label=f"irr, after, {nstr_upper_init = }")
    plt.scatter(f.attrs["t10"], [10], color="green")
plt.title(f"{grav = }, {tint = }, {semimajor = }")
plt.legend()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.gca().invert_yaxis()
plt.show()
# %%
