import numpy as np
import os
import picaso
from picaso.grad import did_grad_cp
import json
from collections import namedtuple
from scipy.integrate import ode

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

bobcat_path = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

def _dTdp(p, t):
    grad_x, _ = did_grad_cp(np.asarray(t).item(), np.asarray(p).item(), AdiabatBundle)
    return float(grad_x) * t / p

def t10(p_col, t_col):
    solver = ode(_dTdp).set_integrator('dopri5', rtol=1e-8, atol=1e-8, nsteps=5000)
    idx = np.where(t_col < 5199)[0][-1]
    solver.set_initial_value(t_col[idx], p_col[idx])
    solver.integrate(10.0)
    return float(solver.y[0])

def regrid_initial_guess(p_initial, t_initial, p_final):
    t_final = np.interp(p_final, p_initial, t_initial)
    # some number of these will have been set to the final value, so constant extrapolation, which causes numerical weirdness
    # instead, we're guessing along an adiabat
    indices_to_replace = np.where(p_final > np.max(p_initial))[0]
    solver = ode(_dTdp).set_integrator('dopri5', rtol=1e-8, atol=1e-8, nsteps=5000)
    solver.set_initial_value(t_initial[-1], p_initial[-1])
    for idx in indices_to_replace:
        solver.integrate(p_final[idx])
        t_final[idx] = float(solver.y[0])
    
    return t_final