# unifying all the scripts was stupid
# this starts from the irradiated no-cloud runs in restart_cloudless_irradiated.py
# and adds in fixed clouds
# it remains after this to do cloudy unirradiated

import os

safe_threads = os.getenv("PICASO_SAFE_THREADS_PER_RANK", "1")
for thread_env in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_env] = safe_threads

import psutil
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import traceback
import faulthandler
from time import sleep

faulthandler.enable(all_threads=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

cloudmode = "fixed"
calc_type = "planet"

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
virga_path = os.getenv('virga')

picaso_path = os.path.dirname(picaso.__path__[0])
__refdata__ = os.getenv('picaso_refdata')

mh = '0.0'
CtoO = '0.46'
nlevel = 91
max_attempts = 8
ck_stem = f'sonora_2121grid_feh{mh}_co{CtoO}'
ck_stem += ".hdf5"
ck_db = os.path.join(__refdata__, 'opacities', 'preweighted', ck_stem)

sonora_profile_db = os.path.join(__refdata__,'sonora_grids', 'bobcat', "structures_m+0.0")

# effective^4 = equilibrium^4 + intrinsic^4
# tints = np.arange(100, 2401, 50)
semi_majors = [0.02, 0.04, 0.13, 0.5]
fseds = [1, 2, 3, 4, 8]

gravs = [10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160]
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])

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
    return os.path.join(picaso_path, "data", "unified_restart", f"{fname_stem}.h5")

def initial_guess(fname):
    with h5py.File(fname) as f:
        pressure_guess = np.array(f["pressure"])
        temp_guess = np.array(f["temperature"])
        cvz_locs = np.array(f["cvz_locs"])
        if cvz_locs[-2] > 0 and temp_guess[cvz_locs[-2]] < 5199.9 and cvz_locs[5] > cvz_locs[2]:
            nstr_upper = cvz_locs[-2]
        else:
            nstr_upper = cvz_locs[1]

    nstr_upper = 89
    return pressure_guess, temp_guess, nstr_upper

def generate_tasks():
    for grav in gravs:
        for tint in tints:
            for semi_major in semi_majors:
                for fsed in fseds:
                    yield (grav, tint, semi_major, fsed)

def crash_details(rank, stage):
    rss = psutil.Process().memory_info().rss / 1024**2
    return f"stage={stage}, host={MPI.Get_processor_name()}, pid={os.getpid()}, rank={rank}, RSS={rss:.1f} MiB"

def run(grav, tint, semi_major, fsed, opacity_ck, rank=-1):
    fname = fname_from_params(grav, tint, semi_major, fsed)
    stage = "loading initial profile"
    try:
        fname_start = fname_from_params(grav, tint, semi_major, -1)
        pressure_start, temperature_start, nstr_upper = initial_guess(fname_start)
        nstr_upper = 89
        max_pressure = np.max(pressure_start)
        acceptable = False
        attempt = 0
        while not acceptable and attempt < max_attempts:
            attempt += 1
            pressure_grid = np.logspace(np.log10(np.min(pressure_start)), np.log10(max_pressure * 16), nlevel)
            temp_guess = regrid_initial_guess(pressure_start, temperature_start, pressure_grid)
            rcb_pressure = pressure_start[nstr_upper]
            # nstr_upper = min(89, np.argmin(np.abs(pressure_grid - rcb_pressure)) + 1) # account for the regrid in picking nstr_upper
            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
            print(f"[{grav}, {tint}, {semi_major}] Starting at nstr_upper = {nstr_upper} on rank {rank}")

            stage = f"setting up model, attempt {attempt}/{max_attempts}"
            cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(tint) # input effective temperature
            cl_run.star(opacity_ck, temp=5778.0, metal=0.0, logg=4.4, radius=1.0, database='phoenix', radius_unit=u.R_sun, semi_major=semi_major, semi_major_unit=u.AU)
            rfacv = 0.5
            cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid, rcb_guess=nstr_upper, rfacv=rfacv)
            kz = np.ones_like(pressure_grid) * 1e10 # idk
            virga_planet = vj.Atmosphere(cloud_species, fsed=fsed, mh=1, mmw=2.2)
            virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
            virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_grid, 'temperature': temp_guess, 'kz': kz}), kz_min=1e5, latent_heat=True)
            stage = f"computing Virga clouds, attempt {attempt}/{max_attempts}"
            virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
            cl_run.fix_virga_clouds(virga_out)
            cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
            stage = f"running PICASO climate, attempt {attempt}/{max_attempts}"
            out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=False)
            max_temperature = np.max(out["temperature"])
            acceptable = np.isfinite(max_temperature) and max_temperature < 5100.0
            if not acceptable:
                max_pressure = max_pressure / 2
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] too hot, reducing max pressure")
                cl_run = out = temp_guess = temp_guess_init = virga_out = virga_planet = None
                gc.collect()

        if not acceptable:
            raise RuntimeError(f"temperature remained too hot ({max_temperature:.3f} K) after {max_attempts} attempts")

        stage = "writing HDF5 output"
        fname_tmp = f"{fname}.tmp.{MPI.Get_processor_name()}.{rank}.{os.getpid()}"
        with h5py.File(fname_tmp, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            t10_this = t10(pressure_grid, out["temperature"])
            f.attrs["t10"] = t10_this
            print(f"t10 at {grav, tint, semi_major} = {t10_this:.3f}")
            out_to_hdf5(out, f)
        stage = "replacing final HDF5 output"
        os.replace(fname_tmp, fname)

        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✓ Saved to disk")

        del cl_run, out, temp_guess
        if 'temp_guess_init' in locals():
            del temp_guess_init
        if 'temperature_start' in locals():
            del temperature_start
        if 'pressure_start' in locals():
            del pressure_start
        if 'virga_out' in locals():
            del virga_out
        if 'virga_planet' in locals():
            del virga_planet

        return True

    except Exception as e:
        if 'fname_tmp' in locals():
            try:
                os.remove(fname_tmp)
            except OSError:
                pass
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✗ Error ({crash_details(rank, stage)}): {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return False
    finally:
        gc.collect()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def abort_on_unhandled(exc_type, exc, tb):
    print(f"Unhandled crash ({crash_details(rank, 'MPI setup or task coordination')}): {exc_type.__name__}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exception(exc_type, exc, tb)
    if size > 1:
        comm.Abort(1)

sys.excepthook = abort_on_unhandled

opacity_ck = None
if rank == 0:
    print(f"Starting run")

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
comm.Barrier()

local_completed = 0
local_failed = 0

for i, (grav, tint, semi_major, fsed) in enumerate(generate_tasks()):
    if i % size == rank and opacity_ck is not None:
        result = run(grav, tint, semi_major, fsed, opacity_ck, rank)
        if result:
            local_completed += 1
        else:
            local_failed += 1

completed = comm.allreduce(local_completed, op=MPI.SUM)
failed = comm.allreduce(local_failed, op=MPI.SUM)

if rank == 0:
    print(f"\n=== Summary ===")
    print(f"Total Completed: {completed}")
    print(f"Total Failed: {failed}")

if failed:
    sys.exit(1)
