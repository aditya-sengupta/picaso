import pickle
import numpy as np
with open('data/solution_family/outs.pkl', 'rb') as f:
    sf = pickle.load(f)

keys = sorted(sf.keys())
first = sf[keys[0]]

# ptchem_df columns
df = first['ptchem_df']
print('ptchem_df type:', type(df))
print('ptchem_df columns:', list(df.columns) if hasattr(df, 'columns') else 'no columns')
print('ptchem_df shape:', df.shape)

# flux_balance keys
fb = first['flux_balance']
print('flux_balance keys:', list(fb.keys()))
print('flux_net_ir top5:', fb['flux_net_ir'][:5])
print('flux_net_ir bot5:', fb['flux_net_ir'][-5:])

# spectrum_output full_output
so = first['spectrum_output']
fo = so.get('full_output', None)
print('full_output type:', type(fo))
if isinstance(fo, dict):
    print('full_output keys:', list(fo.keys()))
    if 'level' in fo:
        lv = fo['level']
        print('level keys:', list(lv.keys()))
        if 'thermal_fluxes' in lv:
            tf = lv['thermal_fluxes']
            print('thermal_fluxes keys:', list(tf.keys()))
            print('flux_plus shape:', np.shape(tf['flux_plus']))
