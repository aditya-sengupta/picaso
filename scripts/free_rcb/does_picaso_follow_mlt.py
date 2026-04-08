# 2026-04-06
# Does PICASO obey the mixing length theory prescription I would expect,
# i.e. if I pull a PICASO adiabat table, does that balance convective flux?

# first, I'll need to pull an adiabat table
# I just did this somewhere

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from picaso.grad import did_grad_cp
from collections import namedtuple
import json

__refdata__ = os.environ['picaso_refdata']
picaso_path = os.path.dirname(os.path.dirname(__refdata__))

cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)

def adiabatic_grad_cp(p, t):
    grad_x, cp_x = did_grad_cp(np.asarray(t).item(), np.asarray(p).item(), AdiabatBundle)
    return float(grad_x), float(cp_x)

def dT_dP(p, t):
    return adiabatic_grad(p, t)[0] * t / p

# okay great, this gives me the adiabatic dT/dP for any temperature and pressure
# next up, I need some prescription for the convective flux
# Marley and Robinson 2015 says this is

Rs = 8.314e7 / 2.2 # erg/K/mol normalized to 2.2u? correct?

def convective_flux(f, P, T, cp):
    # f - superadiabatic fraction
    cp = adiabatic_grad_cp(P, T)[1]
    return np.max(0.0, f) ** (3/2) * Rs * P * (T / cp) ** (1/2)

# this isn't actually a useful test...convective flux is 0 by definition if 