# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
# %%

for tint in np.arange(1000, 1601, 50):
    grav = 177
    with h5py.File(f"../data/unified/unified_tint{tint}_grav{grav}_semimajor0.02_nc.h5") as f:
        plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label="a = 0.02au")
    with h5py.File(f"../data/unified/unified_tint{tint}_grav{grav}_ns_nc.h5") as f:
        plt.semilogy(np.array(f["temperature"]), np.array(f["pressure"]), label="no star")
    plt.title(f"{tint = }")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
# %%
