# stuff that's getting unified here:
# cloudless and cloudy
# saving sparse and full outputs
# parallel and serial (tbh I think I should just fully switch to parallel)
# unirradiated and irradiated
# better handling of when runs have already been done

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import traceback

import picaso
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga
import virga.justdoit as vj
import astropy.units as u
import astropy.constants as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
import h5py
import gc
from mpi4py import MPI
from collections import namedtuple

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5
from calculate_t10 import t10, regrid_initial_guess

parser = argparse.ArgumentParser()
parser.add_argument('sweep')
parser.add_argument('cloudy')
parser.add_argument('irradiated')
parser.add_argument('save')
parser.add_argument('rerun')
args = parser.parse_args()
sweep, cloudy, irradiated, save, rerun = str(args.sweep), str(args.cloudy), str(args.irradiated), str(args.save), str(args.rerun)

print(f"Starting run with {sweep = }, {cloudy = }, {irradiated = }, {save = }, {rerun = } ")

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
virga_path = os.getenv('virga')

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.getenv('picaso_refdata')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
pressure_grid = np.logspace(-4, 3 + np.log10(3), nlevel)
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(__refdata__,'sonora_grids','bobcat')
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])

# effective^4 = equilibrium^4 + intrinsic^4

step = 100 if sweep == "coarse" else 10
tints = np.arange(100, 2401, step)
# tints = np.delete(tints, 2) # We're taking 300K as our baseline, so not rerunning it
np.random.shuffle(tints)

gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
np.random.shuffle(gravs)

if cloudy == "cloudy":
    fseds = [-1, 1, 2, 3, 4, 8]
else:
    fseds = [-1]

if irradiated == "irradiated":
    semi_majors = [-1, 0.5, 0.13, 0.04, 0.02]
else:
    semi_majors = [-1]

def fsed_str(fsed):
    if fsed == -1:
        return "nc"
    else:
        return "f" + str(fsed)

def semi_major_str(semi_major):
    if semi_major == -1:
        return "ns"
    else:
        return f"semimajor{semi_major:.2f}"

def fname_from_params(grav, tint, semi_major, fsed):
    fname_stem = f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_{fsed_str(fsed)}"
    return os.path.join(picaso_path, "data", "unified", f"{fname_stem}.h5")

def initial_guess(fname):
    with h5py.File(fname) as f:
        temp_guess = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]
        nstr_upper += 5
        # In case clouds make the RCB sink a bit, we want to allow this much
        # This is unmotivated and we may find it should go even deeper
        # However, having observed that the RCB tends to track the cloud base, I think it's fine

    return temp_guess, nstr_upper

def generate_tasks():
    """Generate all task tuples without storing them all in memory."""
    for grav in gravs:
        for tint in tints:
            for semi_major in semi_majors:
                for fsed in fseds:
                    yield (grav, tint, semi_major, fsed)

def run(grav, tint, semi_major, fsed):
    try:
        # if this is a cloudy run and the equivalent cloudless run doesn't exist, you gotta skip
        if fsed > 0:
            cloudmode = "fixed"
            fname_cloudless = fname_from_params(grav, tint, semi_major, -1)
            if not os.path.exists(fname_cloudless):
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] Skipping - cloudless model unavailable")
                return False
        else:
            cloudmode = "cloudless"
        
        # now, if we're here, we are either at a cloudless run, or a cloudy run with a known cloudless start
        fname = fname_from_params(grav, tint, semi_major, fsed)
        if os.path.exists(fname):
            if rerun == "rerun":
                os.remove(fname)
            elif rerun == "no_rerun":
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] Skipping - already complete")
                return True

        temp_guess = None
        nstr_upper = None
        guess_bobcat = False
        if fsed > 0:
            temp_guess, nstr_upper = initial_guess(fname_cloudless)
        elif semi_major > 0:
            # irradiated cloudless
            fname_unirradiated = fname_from_params(grav, tint, -1, -1)
            if os.path.exists(fname_unirradiated):
                temp_guess, _ = initial_guess(fname_unirradiated)
                nstr_upper = 89 # if we're on the fine grid, this should be overwritten by the nearest neighbors at this irradiation level later
            else:
                guess_bobcat = True
        else:
            guess_bobcat = True
        
        if guess_bobcat:
            teff_bobcat, grav_bobcat = bobcat_temps[np.argmin(np.abs(bobcat_temps - tint))], bobcat_gravs[np.argmin(np.abs(bobcat_gravs - grav))]
            pressure_bobcat, temp_bobcat = np.loadtxt(os.path.join(sonora_profile_db,f"t{teff_bobcat}g{grav_bobcat}nc_m0.0.cmp.gz"), usecols=[1,2],unpack=True, skiprows = 1)
            temp_guess = regrid_initial_guess(pressure_bobcat, temp_bobcat, pressure_grid) # closest Bobcat, resampled on the pressure grid we're using in this work
            # 2026-05-11: temporarily, we're just starting at our equivalent 800 run
            # 2026-05-13: this worked to generate the 100K/200K grid locally, so now we're starting at the hottest run available that's colder than the current one
            # 2026-05-14: there's still strange jumps, but 300K seems to have run well for every logg/semimajor, so I'm starting them all from there
            # 2026-05-14 evening: ok wow I'm not even matching the tutorial docs any more so we're going back to Bobcat
            # I think the move is: no-cloud no-star, then irradiated guessing off of those.
            indices_below_bobcat = np.where(pressure_grid > np.max(pressure_bobcat))[0]
            if len(indices_below_bobcat) > 1:
                nstr_upper = min(89, np.min(indices_below_bobcat))
            else:
                nstr_upper = 89
                
        # we're going to look for neighbors on the coarse grid
        # if this point itself is on the coarse grid, we shouldn't be able to hit this
        # because it would've thrown an error when the file exists
        # if we happen to end up here, we simply won't do any of this

        lower_temperature, upper_temperature = 100 * (tint // 100), 100 * (tint // 100 + 1)
        if lower_temperature != tint and upper_temperature != tint:
            lower_neighbor = fname_from_params(grav, 100 * (tint // 100), semi_major, fsed)
            upper_neighbor = fname_from_params(grav, 100 * (tint // 100 + 1), semi_major, fsed)
            
            if os.path.exists(lower_neighbor) and os.path.exists(upper_neighbor):
                with h5py.File(lower_neighbor) as f:
                    nstr_upper = min(nstr_upper, f.attrs["nstr_upper_init"] + 5)
                with h5py.File(upper_neighbor) as f:
                    nstr_upper = min(nstr_upper, f.attrs["nstr_upper_init"] + 5)

        # next bit of logic, 'outlier' reruns.
        if rerun == "outlier" and os.path.exists(fname):
            step = 100 if sweep == "coarse" else 10
            temp_lower, temp_upper, t10_lower, t10_upper, t10_current = None, None, None, None, None
            above_lower, below_upper = True, True
            with h5py.File(fname) as f:
                t10_current = f.attrs["t10"]

            lower_temperature, upper_temperature = tint - step, tint + step
            while lower_temperature is not None and not os.path.exists(fname_from_params(grav, lower_temperature, semi_major, fsed)):
                lower_temperature -= step
                if lower_temperature < 100:
                    # we're off-grid
                    lower_temperature = None
                    
            while upper_temperature is not None and not os.path.exists(fname_from_params(grav, upper_temperature, semi_major, fsed)):
                upper_temperature += step
                if upper_temperature > 2400:
                    upper_temperature = None

            if lower_temperature is not None:
                with h5py.File(fname_from_params(grav, lower_temperature, semi_major, fsed)) as f:
                    t10_lower = f.attrs["t10"]
                    temp_lower = np.array(f["temperature"])
                    above_lower = t10_current > t10_lower

            if upper_temperature is not None:
                with h5py.File(fname_from_params(grav, upper_temperature, semi_major, fsed)) as f:
                    t10_upper = f.attrs["t10"]
                    temp_upper = np.array(f["temperature"])
                    below_upper = t10_current < t10_upper
            
            if above_lower and below_upper:
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] not an outlier, skipping.")
                return True
            elif (not above_lower) and (not below_upper):
                # weighted average
                w_down, w_up = tint - lower_temperature, upper_temperature - tint
                w_down, w_up = w_down / (w_down + w_up), w_up / (w_down + w_up)
                temp_guess = temp_lower * w_up + temp_upper * w_down
            elif not above_lower:
                temp_guess = temp_lower
            elif not below_upper:
                temp_guess = temp_upper
            nstr_upper = 89

        nstr_upper_init = nstr_upper
        temp_guess_init = np.copy(temp_guess)
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] Starting at nstr_upper = {nstr_upper}")

        calc_type = "planet" if semi_major > 0 else "browndwarf"
        cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(tint) # input effective temperature
        gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','C2H2','C2H4','C2H6','Na','K','PH3','FeH','SO2','H2S']
        opacity_ck = jdi.opannection(ck_db=os.path.join(__refdata__, "climate_INPUTS", "661"),method='resortrebin',preload_gases=gases_fly)

        if semi_major > 0:
            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
            rfacv = 0.5
        else:
            rfacv = 0.0

        cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
        if fsed > 0:
            kz = np.ones_like(pressure_grid) * 1e10 # idk
            virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
            virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
            virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
            virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
            cl_run.fix_virga_clouds(virga_out)

        cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
        out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=False)
        
        with h5py.File(fname, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            f.attrs["t10"] = t10(pressure_grid, out["temperature"])
            if save == "full":
                out_to_hdf5(out, f)
            else:
                f["pressure"] = pressure_grid
                f["temperature"] = out["temperature"]
                f["cvz_locs"] = out["cvz_locs"]
                f["spectrum_output_thermal"] = out["spectrum_output"]["thermal"]
                f["spectrum_output_wavenumber"] = out["spectrum_output"]["wavenumber"]
                
                if fsed > 0:
                    for k in ["opd_per_layer", "asymmetry", "single_scattering", "condensate_mmr"]:
                        f[k] = virga_out[k]
        
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✓ Saved to disk")
        
        del cl_run, opacity_ck, out, temp_guess
        if 'virga_out' in locals():
            del virga_out
        if 'virga_planet' in locals():
            del virga_planet
        if 'temp_guess_init' in locals():
            del temp_guess_init
        if 'temp_bobcat' in locals():
            del temp_bobcat
        if 'pressure_bobcat' in locals():
            del pressure_bobcat
        
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✗ Error: {e}")
        print(traceback.format_exc())
        gc.collect()
        return False

parallel = True # just for debugging

if parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        tasks = list(generate_tasks())
        total_tasks = len(tasks)
        print(f"Total tasks to process: {total_tasks}")
    else:
        tasks = None
        total_tasks = None

    tasks = comm.bcast(tasks, root=0)
    total_tasks = comm.bcast(total_tasks, root=0)

    local_completed = 0
    local_failed = 0

    for i, (grav, tint, semi_major, fsed) in enumerate(tasks):
        if i % size == rank:
            result = run(grav, tint, semi_major, fsed)
            if result:
                local_completed += 1
            else:
                local_failed += 1
            
            gc.collect()

    completed = comm.allreduce(local_completed, op=MPI.SUM)
    failed = comm.allreduce(local_failed, op=MPI.SUM)

    if rank == 0:
        print(f"\n=== Summary ===")
        print(f"Total Completed: {completed}")
        print(f"Total Failed: {failed}")
else:
    tasks = generate_tasks()
    completed, failed = 0, 0
    for task in tasks:
        result = run(*task)
        if result:
            completed += 1
        else:
            failed += 1

    print(f"\n=== Summary ===")
    print(f"Total Completed: {completed}")
    print(f"Total Failed: {failed}")
