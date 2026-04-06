import numpy as np
import os
import picaso
from os import path
import pickle as pkl
import h5py
import pandas as pd
from tqdm import tqdm

picaso_path = path.dirname(picaso.__path__[0])

def out_to_hdf5(out: dict, f: h5py.File, prefix=""):
    """
    out: the thing we're trying to save
    f: an open h5py file
    prefix: to handle recursive calls
    """
    for k in out.keys():
        if isinstance(out[k], dict):
            out_to_hdf5(out[k], f, k + "_")
        elif isinstance(out[k], str):
            f.attrs[k] = out[k]
        else:
            try:
                f.create_dataset(prefix + k, data=np.array(out[k]))
            except TypeError as e:
                pass

# I can unpickle something from lux on local, but not in reverse 
if __name__ == "__main__":
    run_name = "cloudy_irradiated"
    for f in tqdm(os.listdir(path.join(picaso_path, "data", run_name))):
        if f.endswith("pkl"):
            out = pkl.load(open(path.join(picaso_path, "data", run_name, f), "rb"))
            fname_h5 = f[:-4] + ".h5"
            path_h5 = path.join(picaso_path, "data", run_name, fname_h5)
            with h5py.File(path_h5, "w") as f_h5:
                out_to_hdf5(out, f_h5)
                