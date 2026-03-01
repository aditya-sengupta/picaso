# %%
import numpy as np
import json
from collections import namedtuple
import picaso
from picaso import justdoit as jdi
from picaso.grad import did_grad_cp
from astropy import units as u
__refdata__ = os.environ['picaso_refdata']

import sys
sys.path.append(".")
from diagnostic_plot import diagnostic_plot

picaso_path = os.path.dirname(picaso.__path__[0])

tint = 200
grav = 10 ** 1.4
semi_major = 0.031 # au, exoplanet archive
r_star = float(((semi_major * u.au) / 9.1) / u.R_sun)
all_feh = ["-100", "+000", "+030", "+050", "+070", "+100"]
all_co = ["025", "050", "100", "150", "200"]
all_rfacv = ["0.5", "0.8", "0.65", "0.75", "0.85"]

mh, CtoO, rfacv = all_feh[1], all_co[2], all_rfacv[0]
cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
AdiabatBundle = AdiabatBundle(
    np.array(cp_grad['temperature']),
    np.array(cp_grad['pressure']),
    np.array(cp_grad['adiabat_grad']),
    np.array(cp_grad['specific_heat'])
)
tp = np.genfromtxt(f"../hd189_tp_gagnebin/tp_feh{feh}_tint200_co{co}_rfacv{rfacv}.txt")

pressure, temperature = tp[:,0], tp[:,1]
grad = np.diff(np.log(temperature)) / np.diff(np.log(pressure))
layer_pressure = np.sqrt(pressure[1:] * pressure[:-1])
layer_temperature = np.sqrt(temperature[1:] * temperature[:-1])
adiabatic_grad = np.array([did_grad_cp(t, p, AdiabatBundle)[0] for (t, p) in zip(layer_temperature, layer_pressure)])
rcb_guess = np.max(np.where(grad < 0.98 * adiabatic_grad)[0])
# %%
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

cl_run = jdi.inputs(calculation="planet", climate=True)
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
cl_run.effective_temp(tint)
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
cl_run.star(opacity_ck, filename=os.path.join(picaso_path, "data/solspec_picaso.dat"), w_unit="um", f_unit="flam", semi_major=semi_major, semi_major_unit = u.AU, radius=r_star, radius_unit=u.R_sun)
nlevel = 91

temp_guess = np.ones((nlevel,)) * 1500
cl_run.inputs_climate(temp_guess=np.copy(temp_guess), pressure=pressure, rcb_guess=rcb_guess, rfacv=float(rfacv))
diagnostic_plot_path = os.path.join(picaso_path, f"figures/hd189/initial_test.png")
out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %%
diagnostic_plot(out, cl_run, opacity_ck, fname=diagnostic_plot_path, temp_guess=temp_guess)

# %%
