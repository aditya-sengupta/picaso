# %%
import numpy as np
import h5py
import os
import sys
import json
sys.path.append("..")
from scripts.diamondback.diamondback_cz import find_rcb_diamondback
from scripts.load_diamondback import diamondback_pt
from collections import namedtuple
from scipy.integrate import ode
from picaso.grad import did_grad_cp


__refdata__ = os.environ['picaso_refdata']

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

p_db, t_db = diamondback_pt(1000, 316, 2)
plt.semilogy(t_db, p_db, label="diamondback")
t10_db = t10(p_db, t_db)
plt.scatter([t10_db], [10])
with h5py.File("../data/both_irradiated/irr_teff1000_grav316_semimajor0.02fsed2.h5") as f:
    t_irr, p_irr = np.array(f['temperature']), np.array(f['pressure'])
    plt.semilogy(t_irr, p_irr, label="irradiated")
    plt.scatter([t10(p_irr, t_irr)], [10])
plt.gca().invert_yaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
# %%
