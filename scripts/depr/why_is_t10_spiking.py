import os
import numpy as np
import h5py
from calculate_t10 import t10

for tint in [500, 600, 700]:
    with h5py.File(f"data/unified/unified_tint{tint}_grav56_ns_nc.h5") as f:
        print(np.array(f["cvz_locs"]))
        t10_val = t10(np.array(f["pressure"]), np.array(f["temperature"]))
        print(f"{tint = }", f", t10 = {t10_val:.3f}")