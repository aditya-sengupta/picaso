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

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5
from load_diamondback import diamondback_pt

parser = argparse.ArgumentParser()
parser.add_argument('sweep')
parser.add_argument('do_cloudy')
parser.add_argument('full_save')
args = parser.parse_args()
sweep, do_cloudy. full_save = str(args.sweep), bool(args.do_cloudy), bool(args.full_save)

print(f"Starting run with {sweep = }, {do_cloudy = }, {full_save = }")

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
virga_path = os.getenv('virga')

picaso_path = os.path.dirname(picaso.__path__[0])
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids', 'bobcat')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
pressure_grid = np.logspace(-4, 3, nlevel)
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# effective^4 = equilibrium^4 + intrinsic^4

step = 100 if sweep == "coarse" else 10
tints = np.arange(100, 2401, step)
np.random.shuffle(tints)

gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
np.random.shuffle(gravs)

if do_cloudy:
    fseds = [-1, 1, 2, 3, 4, 8]
else:
    fseds = [-1]

semi_majors = [-1, 0.5, 0.13, 0.04, 0.02]

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

def generate_tasks():
    """Generate all task tuples without storing them all in memory."""
    for grav in gravs:
        for tint in tints:
            for semi_major in semi_majors:
                for fsed in fseds:
                    yield (grav, tint, semi_major, fsed)

def run(grav, tint, semi_major, fsed):
    try:
        print(f"grav = {grav} m/s/s, interior temperature = {tint} K, semimajor axis = {semi_major:.2f} au, fsed = {fsed}")
        # if this is a cloudy run and the equivalent cloudless run doesn't exist, you gotta skip
        if fsed > 0:
            cloudmode = "fixed"
            fname_cloudless = fname_from_params(grav, tint, semi_major, -1)
            if not os.path.exists(fname_cloudless):
                print(f"[{grav}, {teff}, {semi_major}, {fsed}] Skipping - cloudless model unavailable")
                return False
        else:
            cloudmode = "cloudless"
        
        # now, if we're here, we are either at a cloudless run, or a cloudy run with a known cloudless start
        fname = fname_from_params(grav, tint, semi_major, fsed)
        # I'll work out how I want to handle reruns later, for now we can skip existing files
        if os.path.exists(fname):
            print(f"[{grav}, {teff}, {semi_major}, {fsed}] Skipping - already complete")
            return True

        temp_guess = None
        nstr_upper = None
        if fsed > 0:
            with h5py.File(fname_cloudless) as f:
                temp_guess = np.array(f["temperature"])
                nstr_upper = f.attrs["nstr_upper_init"] + 5
                # In case clouds make the RCB sink a bit, we want to allow this much
                # This is unmotivated and we may find it should go even deeper
                # However, having observed that the RCB tends to track the cloud base, I think it's fine
        else:
            pressure_db, temp_db = read_diamondback_structure(tint, grav, fsed)
            temp_guess = np.interp(pressure_grid, pressure_db, temp_db) # closest Diamondback, resampled on the pressure grid we're using in this work
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

            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
        print(f"[{grav}, {teff}, {semi_major}, {fsed}] Starting at nstr_upper = {nstr_upper}")

        cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(teff) # input effective temperature
        opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

        if semi_major > 0:
            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
            rfacv = 0.5

        cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
        if fsed > 0:
            kz = np.ones_like(pressure_grid) * 1e10 # idk
            virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
            virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
            virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
            virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
            cl_run.fix_virga_clouds(virga_out)

        out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)
        
        if full_save:
            with h5py.File(fname, "w") as f:
                out_to_hdf5(out, f)
        else:
            with h5py.File(fname, "w") as f:
                f["pressure"] = pressure_grid
                f["temperature"] = out["temperature"]
                f.attrs["nstr_upper_init"] = nstr_upper_init
                f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
                f["cvz_locs"] = out["cvz_locs"]
                f["spectrum_output_thermal"] = out["spectrum_output"]["thermal"]
                f["spectrum_output_wavenumber"] = out["spectrum_output"]["wavenumber"]
                
                if fsed > 0:
                    for k in ["opd_per_layer", "asymmetry", "single_scattering", "condensate_mmr"]:
                        f[k] = virga_out[k]
        
        print(f"[{grav}, {teff}, {semi_major}, {fsed}] ✓ Saved to disk")
        
        # Clean up local variables
        del cl_run, opacity_ck, virga_planet, virga_out, out
        del pressure_grid, temp_guess, temp_for_virga, kz
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"[{grav}, {teff}, {semi_major}, {fsed}] ✗ Error: {e}")
        gc.collect()
        return False

def run_through_loop(num_threads=10):
    """Run all tasks using MPI, saving each result to disk before freeing memory."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Master process distributes tasks
    if rank == 0:
        tasks = list(generate_tasks())
        total_tasks = len(tasks)
        print(f"Total tasks to process: {total_tasks}")
    else:
        tasks = None
        total_tasks = None
    
    # Broadcast task list to all processes
    tasks = comm.bcast(tasks, root=0)
    total_tasks = comm.bcast(total_tasks, root=0)
    
    # Distribute tasks evenly across processes
    local_completed = 0
    local_failed = 0
    
    for i, (grav, teff, semi_major, fsed) in enumerate(tasks):
        # Only process tasks assigned to this rank
        if i % size == rank:
            result = run(grav, teff, semi_major, fsed)
            if result:
                local_completed += 1
            else:
                local_failed += 1
            
            # Force garbage collection after each completed task
            gc.collect()
    
    # Gather results from all processes
    completed = comm.allreduce(local_completed, op=MPI.SUM)
    failed = comm.allreduce(local_failed, op=MPI.SUM)
    
    if rank == 0:
        print(f"\n=== Summary ===")
        print(f"Total Completed: {completed}")
        print(f"Total Failed: {failed}")
    
if __name__ == "__main__":
    run_through_loop()  # Run with: mpirun -np N python script.py
