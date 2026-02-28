import re
from os import path
import numpy as np
import pandas as pd
from functools import reduce

diamondback_datapath = "/Users/adityasengupta/picaso/reference/sonora_grids"
lmap = lambda f, x: list(map(f, x))

def readInFile(filename):
	f = open(filename)
	line = f.readline()
	filelines = []
	while line != "":
		filelines.append(line)
		try: 
			line = f.readline()
		except UnicodeDecodeError: 
			line='xxx xxx'
	return filelines

def read_diamondback_cloud_structure(teff, grav_ms2, fsed):
    all_lines = readInFile(path.join(diamondback_datapath, f"diamondback_allmodels/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.out"))
    dfs = []
    for (i, line) in enumerate(all_lines):
        if "kz(cm^2/s)" in line:
            columns = line.split()
            condensate = re.match(r" condensing gas = (\S+)", all_lines[i-2])[1]
            columns[6] = condensate + " qc(g/g)"
            columns[7] = condensate + " col_opd"
            data = lmap(lambda l: lmap(float, l.split()), all_lines[i+1:i+91])
            dfs.append(pd.DataFrame(data=data, columns=columns))
        
    df = reduce(lambda x, y: pd.merge(x, y[y.columns.difference(x.columns)], left_index=True, right_index=True), dfs)
    return df.rename(columns={"P(bar)": "pressure", "T(K)": "temperature", "kz(cm^2/s)": "kz"})

def read_diamondback_optical_properties(teff, grav_ms2, fsed):
    cld_file = path.join(diamondback_datapath, f"diamondback_alloutputs/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.cld")
    df = pd.read_csv(cld_file, sep=r"\s+", header=None,
                      names=["level", "spectral_window", "tau", "g0", "w0", "sigma"])
    return df