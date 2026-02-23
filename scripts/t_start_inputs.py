# %%
import os
import warnings
import astropy.units as u
import numpy as np
import pandas as pd
from collections import namedtuple
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
from picaso.climate import profile
from picaso.fluxes import tidal_flux
from matplotlib.pyplot import cm

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')

fsed = 2
teff = 1200
cloudmode = "fixed"
grav = 316.0
nstr_uppers = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
cmap = cm.winter(np.linspace(0, 1, len(nstr_uppers)))
for (i, nstr_upper) in enumerate(nstr_uppers):
    print(f"{nstr_upper = }")
    color = cmap[i]
    cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(teff) # input effective temperature
    opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

    nlevel = 91 # number of plane-parallel levels in your code
    rfacv = 0.0

    sonora_df = pd.read_csv(f"/Users/adityasengupta/picaso/reference/sonora_grids/diamondback/t900g316f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    pressure_grid = np.array(sonora_df["P"])
    temp_guess = np.array(sonora_df["T"])

    cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
    cl_run.virga(condensates=cloud_species, directory="/Users/adityasengupta/virga/refrind", runmode=cloudmode, mh=1, fsed=fsed, latent_heat=True)

    cli = cl_run.inputs['climate']

    t_table = cli['t_table']
    p_table = cli['p_table']
    grad = cli['grad']
    cp = cli['cp']
    moist = cli['moistgrad']
    AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
    AdiabatBundle = AdiabatBundle(t_table,p_table,grad,cp)

    wno = opacity_ck.wno
    delta_wno = opacity_ck.delta_wno
    nwno = opacity_ck.nwno
    opacity_ck.relative_flux = np.ones(nwno)
    min_temp = min(opacity_ck.temps)
    max_temp = max(opacity_ck.temps)

    #we will extend the black body grid 30% beyond the min and max temp of the 
    #opacity grid just to be safe with the spline
    Teff = cl_run.inputs['planet']['T_eff']
    extension = 0.3 
    #add threshold for tmin for convergence *JM
    if Teff > 300:
        tmin = min_temp*(1-extension)
    else:
        tmin = 10

    if Teff > 1600:
        tmax = 10000
    else:
        tmax = max_temp*(1+extension)

    Opagrid = namedtuple('Opagrid',['nwno','delta_wno','wno','ngauss','gauss_wts','tmin','tmax'])
    Opagrid = Opagrid(nwno, delta_wno, wno, opacity_ck.ngauss,opacity_ck.gauss_wts,tmin,tmax)

    InjectionBundle = namedtuple('InjectionBundle', ['inject_energy','inject_beam','wave_in', 'pm', 'hratio', 'beam_profile'])
    InjectionBundle = InjectionBundle(False, False, 0, 1, 1, 0)
    grav_si = 0.01*grav # cgs to si
    col_den = 1e6*(pressure_grid[1:] -pressure_grid[:-1] ) / (grav_si/0.01) # cgs g/cm^2
    nlevel = len(pressure_grid)
    tidal = tidal_flux(Teff, nlevel, pressure_grid, col_den, InjectionBundle)

    virga_kwargs = cl_run.inputs['climate'].get('virga_kwargs',{})
    opd_cld_climate = np.zeros(shape=(cl_run.nlevel-1,nwno,4))
    g0_cld_climate = np.zeros(shape=(cl_run.nlevel-1,nwno,4))
    w0_cld_climate = np.zeros(shape=(cl_run.nlevel-1,nwno,4))
    #BUNDLING
    virga_specific =[['virga_'+i,val] for i ,val in virga_kwargs.items() if 'patchy' not in i]
    hole_specific =  [[i,val] for i ,val in virga_kwargs.items() if 'patchy' in i]
    CloudParametersT = namedtuple('CloudParameters',['cloudy', 'OPD','G0','W0','cld_out']
                                    +[i[0] for i in virga_specific]
                                    +[i[0] for i in hole_specific])
    #this adds the cloud params that are always needed plus the virga kwargs, if they are used 
    CloudParameters=CloudParametersT(*([cloudmode, opd_cld_climate,g0_cld_climate,w0_cld_climate,None]
                                    +[i[1] for i in virga_specific]
                                    +[i[1] for i in hole_specific]))

    convergence_criteriaT = namedtuple('Conv',['it_max','itmx','conv','convt','x_max_mult'])
    convergence_criteria = convergence_criteriaT(it_max=10, itmx=1, conv=5.0, convt=4.0, x_max_mult=7.0) 
    cl_run.inputs['atmosphere']['kzz']={}

    temp_out = profile(cl_run, 1, cl_run.inputs['climate']['nstr'], temp_guess, pressure_grid, AdiabatBundle, opacity_ck, grav, 0.0, 0.0, tidal, Opagrid, CloudParameters, 0, np.array([]), np.array([]), convergence_criteria, 0, verbose=False)[2]
    plt.semilogy(temp_out, pressure_grid, color=color, label=nstr_upper)
    plt.scatter([temp_out[nstr_upper]], [pressure_grid[nstr_upper]], color=color)

plt.gca().invert_yaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.savefig("profile_outputs.png")
# %%
