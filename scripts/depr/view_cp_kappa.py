# %%
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import picaso
from picaso import justdoit as jdi
from picaso.justplotit import pt_adiabat, brightness_temperature
from picaso.grad import did_grad_cp
from astropy import units as u
__refdata__ = os.environ['picaso_refdata']
# %%
picaso_path = os.path.dirname(picaso.__path__[0])

cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)

temp = np.array(cp_grad['temperature'])
pres = np.array(cp_grad['pressure'])
cp = np.array(cp_grad['specific_heat'])

temp, pres = np.meshgrid(temp, pres, indexing='ij')
plt.figure(figsize=(10, 6))
plt.pcolormesh(temp, pres, cp, shading='nearest')
plt.colorbar(label='Specific Heat')
plt.xlabel('log(temperature/K)')
plt.ylabel('log(pressure/bar)')
plt.gca().invert_yaxis()
plt.title('Specific Heat (Cp)')
plt.show()
# %%
