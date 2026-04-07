# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

cl_run = jdi.inputs(calculation="browndwarf", climate = True)
teff= 1000
grav = 1000 

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) 
cl_run.effective_temp(teff) 

mh = '+000'#'+0.0' #log metallicity
CtoO = '100'# # CtoO absolute ratio
ck_db_path = ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# and not the line by line opacities
opacity_ck = jdi.opannection(ck_db=ck_db,method='preweighted')

nlevel = 91

#Lets set the max and min at 1e-4 bars and 500 bars

Pmin = 1e-4 #bars
Pmax = 500 #bars
pressure=np.logspace(np.log10(Pmin),np.log10(Pmax),nlevel) # set your pressure grid

temp_guess = np.zeros(shape=(nlevel)) + 500 # K , isothermal atmosphere guess

pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

outs = {}
# %%
rcb_guesses = np.arange(33, 23, -1)
rfacv = 0.0
for rcb_guess in rcb_guesses:
    cl_run.inputs_climate(temp_guess=np.copy(temp_guess), pressure=pressure,rcb_guess=rcb_guess, rfacv=rfacv)
    out = outs[rcb_guess] = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
    plt.figure(figsize=(10,10))
    plt.ylabel("Pressure [Bars]", fontsize=25)
    plt.xlabel('Temperature [K]', fontsize=25)
    plt.ylim(500,1e-4)
    plt.xlim(200,3000)

    plt.semilogy(out['temperature'],out['pressure'],color="r",linewidth=3,label="Our Run")

    plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    plt.legend(fontsize=15)

    plt.title(r"T$_{\rm eff}$= 1000 K, log(g)=5.0",fontsize=25)
    plt.show()

# %%
pkl.dump(outs, open("data/solution_family/outs.pkl", 'wb'))