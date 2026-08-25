# unifying all the scripts was stupid
# this starts from the irradiated no-cloud runs in restart_cloudless_irradiated.py
# and adds in fixed clouds
# it remains after this to do cloudy unirradiated

# mpiexec -n 64 python -u -m mpi4py scripts/restart_cloudy_irradiated_safe.py

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
import math
from pathlib import Path
from time import sleep

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import picaso
import picaso.climate as picaso_climate
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
from numba_cache_guard import enforce_numba_cache_limit

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
tints = np.arange(100, 2401, 50)
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
                    fname = fname_from_params(grav, tint, semi_major, fsed)
                    if not os.path.exists(fname):
                        yield (grav, tint, semi_major, fsed)

def run(grav, tint, semi_major, fsed, opacity_ck, rank=-1):
    fname = fname_from_params(grav, tint, semi_major, fsed)
    try:
        fname_start = fname_from_params(grav, tint, semi_major, -1)
        pressure_start, temperature_start, nstr_upper = initial_guess(fname_start)
        max_pressure = np.max(pressure_start)
        acceptable = False
        attempt = 0
        while not acceptable and attempt < max_attempts:
            attempt += 1
            pressure_grid = np.logspace(np.log10(np.min(pressure_start)), np.log10(max_pressure * 16), nlevel)
            temp_guess = regrid_initial_guess(pressure_start, temperature_start, pressure_grid)
            rcb_pressure = pressure_start[nstr_upper]
            nstr_upper = min(89, np.argmin(np.abs(pressure_grid - rcb_pressure)) + 1) # account for the regrid in picking nstr_upper
            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
            print(f"[{grav}, {tint}, {semi_major}] Starting at nstr_upper = {nstr_upper} on rank {rank}")

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
            virga_out = vj.compute(virga_planet, as_dict=True, directory=os.path.join(virga_path, "refrind"))
            cl_run.fix_virga_clouds(virga_out)
            cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher') # on the fly mixing
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

        fname_tmp = f"{fname}.tmp.{MPI.Get_processor_name()}.{rank}.{os.getpid()}"
        with h5py.File(fname_tmp, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            t10_this = t10(pressure_grid, out["temperature"])
            f.attrs["t10"] = t10_this
            print(f"t10 at {grav, tint, semi_major} = {t10_this:.3f}")
            out_to_hdf5(out, f)
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
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] ✗ Error: {e}")
        print(traceback.format_exc())
        return False
    finally:
        gc.collect()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def abort_on_unhandled(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)
    if size > 1:
        comm.Abort(1)

sys.excepthook = abort_on_unhandled

cache_config_result = None
if rank == 0:
    try:
        configured_max_gib = float(os.getenv("PICASO_NUMBA_CACHE_MAX_GIB", "5"))
        configured_check_rounds = int(
            os.getenv("PICASO_NUMBA_CACHE_CHECK_ROUNDS", "8")
        )
        if not math.isfinite(configured_max_gib) or configured_max_gib < 0:
            raise ValueError("PICASO_NUMBA_CACHE_MAX_GIB must be finite and non-negative")
        if configured_check_rounds < 0:
            raise ValueError("PICASO_NUMBA_CACHE_CHECK_ROUNDS must be non-negative")
        cache_config_result = (
            None,
            configured_max_gib,
            configured_check_rounds,
        )
    except Exception as exc:
        cache_config_result = (f"{type(exc).__name__}: {exc}", 0, 0)

cache_config_error, numba_cache_max_gib, numba_cache_check_rounds = comm.bcast(
    cache_config_result,
    root=0,
)
if cache_config_error is not None:
    raise RuntimeError(f"Invalid Numba cache guard configuration: {cache_config_error}")

numba_cache_guard_enabled = numba_cache_max_gib > 0 and numba_cache_check_rounds > 0
numba_cache_max_bytes = int(numba_cache_max_gib * 1024**3)


def resolve_numba_cache_dir():
    cache_dir = Path(picaso_climate.t_start._cache.cache_path).resolve()
    unsafe_paths = {
        Path(cache_dir.anchor),
        Path.home().resolve(),
        Path(picaso.__path__[0]).resolve(),
    }
    if cache_dir in unsafe_paths:
        raise RuntimeError(f"Refusing unsafe Numba cache directory: {cache_dir}")
    return cache_dir


numba_cache_dir = None
if numba_cache_guard_enabled:
    cache_dir_result = None
    if rank == 0:
        try:
            cache_dir_result = (None, str(resolve_numba_cache_dir()))
        except Exception as exc:
            cache_dir_result = (f"{type(exc).__name__}: {exc}", None)
    cache_dir_error, numba_cache_dir = comm.bcast(cache_dir_result, root=0)
    if cache_dir_error is not None:
        raise RuntimeError(f"Could not resolve Numba cache directory: {cache_dir_error}")


def guard_numba_cache(reason):
    """Collectively clear oversized Numba cache artifacts between task blocks."""
    if not numba_cache_guard_enabled:
        return

    # Every rank in this MPI job reaches this only between models. The barrier
    # prevents an in-job cache race; separate concurrent jobs should use
    # different NUMBA_CACHE_DIR values.
    comm.Barrier()
    payload = None
    if rank == 0:
        try:
            result = enforce_numba_cache_limit(
                numba_cache_dir,
                numba_cache_max_bytes,
            )
            if result.after_bytes > numba_cache_max_bytes:
                raise RuntimeError(
                    f"cache remains at {result.after_bytes / 1024**3:.2f} GiB "
                    f"after removing {result.removed_files} Numba artifacts"
                )
            payload = (
                None,
                result.before_bytes,
                result.after_bytes,
                result.removed_files,
                result.removed_bytes,
            )
        except Exception as exc:
            payload = (f"{type(exc).__name__}: {exc}", 0, 0, 0, 0)

    error, before_bytes, after_bytes, removed_files, removed_bytes = comm.bcast(
        payload,
        root=0,
    )
    if error is not None:
        raise RuntimeError(f"Numba cache guard failed at {reason}: {error}")

    if rank == 0 and removed_files:
        print(
            f"Numba cache guard ({reason}): cache was "
            f"{before_bytes / 1024**3:.2f} GiB; removed {removed_files} files "
            f"({removed_bytes / 1024**3:.2f} GiB); "
            f"now {after_bytes / 1024**3:.2f} GiB",
            flush=True,
        )

opacity_ck = None
if rank == 0:
    print(f"Starting run")

guard_numba_cache("startup")

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
comm.Barrier()

local_completed = 0
local_failed = 0

tasks = list(generate_tasks()) if rank == 0 else None
if rank == 0 and len(tasks) != len(set(tasks)):
    raise RuntimeError("generate_tasks() produced duplicate tasks")
tasks = comm.bcast(tasks, root=0)

if numba_cache_guard_enabled:
    task_block_size = size * numba_cache_check_rounds
else:
    task_block_size = max(1, len(tasks))

for block_start in range(0, len(tasks), task_block_size):
    block_end = min(block_start + task_block_size, len(tasks))
    for i in range(block_start, block_end):
        grav, tint, semi_major, fsed = tasks[i]
        if i % size == rank and opacity_ck is not None:
            result = run(grav, tint, semi_major, fsed, opacity_ck, rank)
            if result:
                local_completed += 1
            else:
                local_failed += 1
    guard_numba_cache(f"after task block {block_start}:{block_end}")

completed = comm.allreduce(local_completed, op=MPI.SUM)
failed = comm.allreduce(local_failed, op=MPI.SUM)

if rank == 0:
    print(f"\n=== Summary ===")
    print(f"Total Completed: {completed}")
    print(f"Total Failed: {failed}")

if failed:
    sys.exit(1)
