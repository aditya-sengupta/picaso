import pickle as pkl
import os
import sys
import numpy as np

foldername = sys.argv[1]

for f in os.listdir(f"data/{foldername}"):
	if f.endswith("pkl"):
		out = pkl.load(open(f"data/{foldername}/{f}", "rb"))
		t, p = out["temperature"], out["pressure"]
		np.savez(f"data/{foldername}/{f[:-4]}.npz", t=t, p=p)
