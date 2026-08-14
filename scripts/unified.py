# stuff that's getting unified here:
# cloudless and cloudy
# saving sparse and full outputs
# parallel and serial (tbh I think I should just fully switch to parallel)
# unirradiated and irradiated
# better handling of when runs have already been done

import os
import psutil
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import traceback
from pathlib import Path
from time import perf_counter, sleep


parser = argparse.ArgumentParser()
parser.add_argument('sweep')
parser.add_argument('cloudy')
parser.add_argument('irradiated')
parser.add_argument('save')
parser.add_argument('rerun')
parser.add_argument(
    '--clear-numba-cache',
    action='store_true',
    help='Clear PICASO Numba cache files once, on MPI rank 0, before imports.',
)
parser.add_argument(
    '--verify-numba-cache',
    action='store_true',
    help='Run a small cache-reuse probe and exit before loading opacities.',
)

args = parser.parse_args()
sweep, cloudy, irradiated, save, rerun = str(args.sweep), str(args.cloudy), str(args.irradiated), str(args.save), str(args.rerun)

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def clear_numba_cache():
    """Remove PICASO's on-disk Numba specializations before importing PICASO."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        cache_dir = Path(os.environ["NUMBA_CACHE_DIR"]).expanduser().resolve()
    else:
        cache_dir = Path(__file__).resolve().parents[1] / "picaso" / "__pycache__"
    removed_files = 0
    removed_bytes = 0

    if cache_dir.exists():
        for pattern in ("*.nbc", "*.nbi"):
            for cache_file in cache_dir.rglob(pattern):
                try:
                    removed_bytes += cache_file.stat().st_size
                    cache_file.unlink()
                    removed_files += 1
                except FileNotFoundError:
                    # Another cleanup may already have removed the file.
                    pass

    return cache_dir, removed_files, removed_bytes


def other_unified_processes(current_job_pids):
    """Return PIDs of other jobs that may still be using the shared cache."""
    script_suffix = os.path.join("scripts", "unified.py")
    excluded_pids = set(current_job_pids)
    for pid in current_job_pids:
        try:
            excluded_pids.update(parent.pid for parent in psutil.Process(pid).parents())
        except psutil.Error:
            pass

    other_pids = []
    for process in psutil.process_iter(("pid", "cmdline")):
        if process.info["pid"] in excluded_pids:
            continue
        command = " ".join(process.info["cmdline"] or ())
        if script_suffix in command:
            other_pids.append(process.info["pid"])
    return sorted(other_pids)


# Cache removal must happen before any rank imports PICASO/Numba. Only rank 0
# mutates the shared cache; the barrier keeps ranks in this job from racing it.
# Do not use the flag while a different unified.py job is running against the
# same cache directory.
if args.clear_numba_cache:
    cleanup_result = None
    current_job_pids = set(comm.gather(os.getpid(), root=0) or ())
    if rank == 0:
        try:
            active_pids = other_unified_processes(current_job_pids)
            if active_pids:
                raise RuntimeError(
                    "another unified.py job is still running with PIDs "
                    + ", ".join(map(str, active_pids))
                )
            cache_dir, removed_files, removed_bytes = clear_numba_cache()
            cleanup_result = (None, str(cache_dir), removed_files, removed_bytes)
        except Exception as exc:
            cleanup_result = (f"{type(exc).__name__}: {exc}", "", 0, 0)

    error, cache_dir, removed_files, removed_bytes = comm.bcast(cleanup_result, root=0)
    if error is not None:
        raise RuntimeError(f"Could not clear the Numba cache: {error}")

    if rank == 0:
        print(
            f"Cleared {removed_files} Numba cache files "
            f"({removed_bytes / 1024**3:.2f} GiB) from {cache_dir}",
            flush=True,
        )
    comm.Barrier()

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
from collections import namedtuple as stdlib_namedtuple
from numba import njit

sys.path.append(".")
from out_to_hdf5 import out_to_hdf5


# PICASO creates these namedtuple classes inside inputs.climate() and
# find_strat(). Numba includes the class identity in a compiled signature, so
# a new class per model/process creates a new machine-code specialization even
# when every array and scalar type is unchanged. Bind one importable class for
# each Numba-visible record to picaso.climate, then intercept only those three
# factories. InjectionBundle and CloudParameters retain PICASO's normal dynamic
# behavior because they do not enter a cached Numba kernel.
ADIABAT_FIELDS = ("t_table", "p_table", "grad", "cp")
OPAGRID_FIELDS = (
    "nwno",
    "delta_wno",
    "wno",
    "ngauss",
    "gauss_wts",
    "tmin",
    "tmax",
)
CONVERGENCE_FIELDS = ("it_max", "itmx", "conv", "convt", "x_max_mult")

StableAdiabatBundle = stdlib_namedtuple(
    "AdiabatBundle",
    ADIABAT_FIELDS,
    module=picaso_climate.__name__,
)
StableOpagrid = stdlib_namedtuple(
    "Opagrid",
    OPAGRID_FIELDS,
    module=picaso_climate.__name__,
)

# This class is already created at module import, but PICASO did not bind it
# under its typename (picaso.climate.Conv), so it could not be restored by
# name from a Numba cache in a fresh interpreter.
StableConv = picaso_climate.convergence_criteriaT

picaso_climate.AdiabatBundle = StableAdiabatBundle
picaso_climate.Opagrid = StableOpagrid
picaso_climate.Conv = StableConv


def stable_opagrid(
    nwno,
    delta_wno,
    wno,
    ngauss,
    gauss_wts,
    tmin,
    tmax,
):
    """Normalize scalar field types as well as the Opagrid class identity."""
    return StableOpagrid(
        int(nwno),
        delta_wno,
        wno,
        int(ngauss),
        gauss_wts,
        float(tmin),
        float(tmax),
    )


def stable_convergence(it_max, itmx, conv, convt, x_max_mult):
    """Keep all convergence records on one predictable Numba signature."""
    return StableConv(
        int(it_max),
        int(itmx),
        float(conv),
        float(convt),
        float(x_max_mult),
    )


picaso_climate.convergence_criteriaT = stable_convergence

STABLE_NUMBA_RECORDS = {
    "AdiabatBundle": (ADIABAT_FIELDS, StableAdiabatBundle),
    "Opagrid": (OPAGRID_FIELDS, stable_opagrid),
    "Conv": (CONVERGENCE_FIELDS, stable_convergence),
}


def stable_namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    """Return stable classes for PICASO records that appear in Numba keys."""
    if isinstance(field_names, str):
        fields = tuple(field_names.replace(",", " ").split())
    else:
        fields = tuple(field_names)

    if typename in STABLE_NUMBA_RECORDS:
        expected_fields, stable_class = STABLE_NUMBA_RECORDS[typename]
        if fields != expected_fields or rename or defaults is not None:
            raise RuntimeError(
                f"PICASO's {typename} schema changed; refusing to create an "
                "unstable Numba cache key."
            )
        return stable_class

    if module is None:
        # collections.namedtuple normally records its direct caller's module.
        # Preserve that behavior even though calls now pass through this shim.
        module = sys._getframe(1).f_globals.get("__name__", "__main__")

    return stdlib_namedtuple(
        typename,
        field_names,
        rename=rename,
        defaults=defaults,
        module=module,
    )


# Both modules imported namedtuple into their own globals, so patch each
# binding. justplotit uses the same AdiabatBundle pattern in pt_adiabat().
picaso_climate.namedtuple = stable_namedtuple
jdi.namedtuple = stable_namedtuple
jpi.namedtuple = stable_namedtuple

import calculate_t10 as calculate_t10_module

# calculate_t10.py creates another module-local AdiabatBundle. Replace its
# instance before _dTdp can pass it into the cached grad.did_grad_cp kernel.
calculate_t10_adiabat = calculate_t10_module.AdiabatBundle
calculate_t10_module.AdiabatBundle = StableAdiabatBundle(
    calculate_t10_adiabat.t_table,
    calculate_t10_adiabat.p_table,
    calculate_t10_adiabat.grad,
    calculate_t10_adiabat.cp,
)
t10 = calculate_t10_module.t10
regrid_initial_guess = calculate_t10_module.regrid_initial_guess


@njit(cache=True)
def stable_record_cache_probe(adiabat, opagrid, convergence):
    """Compile a cheap signature containing all three stabilized records."""
    return adiabat.grad[0, 0] + opagrid.tmin + convergence.conv


def dispatcher_cache_counts(dispatcher):
    hits = sum(getattr(dispatcher, "_cache_hits", {}).values())
    misses = sum(getattr(dispatcher, "_cache_misses", {}).values())
    return hits, misses


if args.verify_numba_cache:
    from picaso.grad import did_grad_cp as cached_did_grad_cp

    assert jdi.inputs.climate.__globals__["namedtuple"] is stable_namedtuple
    assert picaso_climate.find_strat.__globals__["namedtuple"] is stable_namedtuple

    adiabat_factory = jdi.namedtuple("AdiabatBundle", list(ADIABAT_FIELDS))
    opagrid_factory = jdi.namedtuple("Opagrid", list(OPAGRID_FIELDS))
    convergence_factory = picaso_climate.namedtuple(
        "Conv",
        list(CONVERGENCE_FIELDS),
    )
    assert adiabat_factory is StableAdiabatBundle
    assert opagrid_factory is stable_opagrid
    assert convergence_factory is stable_convergence
    assert type(calculate_t10_module.AdiabatBundle) is StableAdiabatBundle

    probe_adiabat = adiabat_factory(
        np.linspace(1.0, 4.0, 53),
        np.linspace(-6.0, 3.0, 26),
        np.full((53, 26), 0.3),
        np.full((53, 26), 7.0),
    )
    probe_opagrid = opagrid_factory(
        1,
        np.ones(1),
        np.ones(1),
        1,
        np.ones(1),
        10,
        10000,
    )
    probe_convergence = convergence_factory(1, 1, 1, 1, 1)
    assert type(probe_opagrid) is StableOpagrid
    assert type(probe_convergence) is StableConv
    assert isinstance(probe_opagrid.tmin, float)
    assert isinstance(probe_convergence.conv, float)

    probe_value = stable_record_cache_probe(
        probe_adiabat,
        probe_opagrid,
        probe_convergence,
    )
    grad_value, cp_value = cached_did_grad_cp(1000.0, 1.0, probe_adiabat)
    probe_hits, probe_misses = dispatcher_cache_counts(stable_record_cache_probe)
    grad_hits, grad_misses = dispatcher_cache_counts(cached_did_grad_cp)

    if rank == 0:
        print(
            "NUMBA_CACHE_PROBE "
            f"record_hits={probe_hits} record_misses={probe_misses} "
            f"grad_hits={grad_hits} grad_misses={grad_misses} "
            "factory_identities=stable "
            f"values={probe_value:.6f},{grad_value:.6f},{cp_value:.6f}",
            flush=True,
        )
    sys.exit(0)

if rank == 0:
    print(
        "Using stable Numba record identities for "
        "AdiabatBundle, Opagrid, and Conv",
        flush=True,
    )

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

sonora_profile_db = os.path.join(__refdata__,'sonora_grids', 'bobcat', "structures_m+0.0")
bobcat_temps = np.arange(200, 2401, 100) # it's not quite this, but this'll do fine
bobcat_gravs = np.array([17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])

gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','C2H2','C2H4','C2H6','Na','K','PH3','FeH','SO2','H2S']
# effective^4 = equilibrium^4 + intrinsic^4

step = 100 if sweep == "coarse" else 50
tints = np.arange(100, 2401, step)

gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
np.random.shuffle(gravs)

if cloudy == "cloudy":
    fseds = [1, 2, 3, 4, 8]
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
        nstr_upper = min(nstr_upper + 5, 89)
        # In case clouds make the RCB sink a bit, we want to allow this much
        # This is unmotivated and we may find it should go even deeper
        # However, having observed that the RCB tends to track the cloud base, I think it's fine

    return temp_guess, nstr_upper

def is_outlier_guess(grav, tint, semi_major, fsed):
    """
    Check if the provided point is an outlier on the T10 grid.
    If it is, provide an initial temperature guess with which to do a rerun.
    If it's not, return None.
    """
    fname = fname_from_params(grav, tint, semi_major, fsed)
    try:
        step = 100 if sweep == "coarse" else 50
        temp_lower, temp_upper, t10_lower, t10_upper, t10_current, outlier_magnitude = None, None, None, None, None, 0
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
                above_lower = t10_current >= t10_lower
                if above_lower:
                    outlier_magnitude = max(outlier_magnitude, abs(t10_lower - t10_current))

        if upper_temperature is not None:
            with h5py.File(fname_from_params(grav, upper_temperature, semi_major, fsed)) as f:
                t10_upper = f.attrs["t10"]
                temp_upper = np.array(f["temperature"])
                below_upper = t10_current <= t10_upper
                if below_upper:
                    outlier_magnitude = max(outlier_magnitude, abs(t10_upper - t10_current))
        
        if above_lower and below_upper:
            return None, 0
        elif (not above_lower) and (not below_upper):
            # weighted average
            w_down, w_up = tint - lower_temperature, upper_temperature - tint
            w_down, w_up = w_down / (w_down + w_up), w_up / (w_down + w_up)
            temp_guess = temp_lower * w_up + temp_upper * w_down
        elif not above_lower:
            temp_guess = temp_lower
        elif not below_upper:
            temp_guess = temp_upper
        
        status_str = f"Identified {grav, tint, semi_major, fsed} as an outlier: "
        if temp_lower is not None:
            status_str += f"lower = {t10_lower:.3f}, "
        status_str += f"current = {t10_current:.3f}, "
        if temp_upper is not None:
            status_str += f"upper = {t10_upper:.3f}"

        print(status_str)
        return temp_guess, outlier_magnitude
    except Exception as e:
        # there's a problem with some file read
        # so search upwards until we get a file we can read, and use that as the guess
        tint_trial = tint + 10
        while tint_trial <= 2400:
            try:
                fname_trial = fname_from_params(grav, tint_trial, semi_major, fsed)
                with h5py.File(fname) as f:
                    temp_guess = np.array(f["temperature"])
                    t10_guess = f.attrs["t10"]
                return temp_guess, 0 
                # deprioritizes this point, but does mark it as needing a rerun
            except Exception as e2:
                tint_trial += 10
        
        return None, 0

def generate_tasks():
    for grav in gravs:
        for tint in tints:
            for semi_major in semi_majors:
                for fsed in fseds:
                    if rerun != "outlier":
                        yield (grav, tint, semi_major, fsed)
                    elif os.path.exists(fname_from_params(grav, tint, semi_major, fsed)): 
                        temp_guess, t10_diff = is_outlier_guess(grav, tint, semi_major, fsed)
                        if temp_guess is not None:
                            yield (grav, tint, semi_major, fsed)

def run(grav, tint, semi_major, fsed, opacity_ck, rank=-1):
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
            pressure_bobcat, temp_bobcat = np.loadtxt(os.path.join(sonora_profile_db,f"t{teff_bobcat}g{grav_bobcat}nc_m0.0.dat"), usecols=[1,2],unpack=True, skiprows = 1)
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
            temp_guess, _ = is_outlier_guess(grav, tint, semi_major, fsed)
            if temp_guess is None:
                print(f"[{grav}, {tint}, {semi_major}, {fsed}] Skipping - not an outlier")
                return True
                # in principle this path shouldn't be reachable, because generate_tasks should filter out all non outliers
                # in practice, I think there'll be weird execution order things
            nstr_upper = 89

        nstr_upper_init = nstr_upper
        temp_guess_init = np.copy(temp_guess)
        print(f"[{grav}, {tint}, {semi_major}, {fsed}] Starting at nstr_upper = {nstr_upper} on rank {rank}")

        calc_type = "planet" if semi_major > 0 else "browndwarf"
        cl_run = jdi.inputs(calculation=calc_type, climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(tint) # input effective temperature

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
        t_start_signatures_before = len(picaso_climate.t_start.signatures)
        climate_started = perf_counter()
        out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True, verbose=True)
        climate_seconds = perf_counter() - climate_started
        t_start_signatures_after = len(picaso_climate.t_start.signatures)
        t_start_hits, t_start_misses = dispatcher_cache_counts(picaso_climate.t_start)
        print(
            f"NUMBA_DISPATCH task={grav, tint, semi_major, fsed} "
            f"climate_seconds={climate_seconds:.3f} "
            f"new_signatures={t_start_signatures_after - t_start_signatures_before} "
            f"total_signatures={t_start_signatures_after} "
            f"cache_hits={t_start_hits} cache_misses={t_start_misses}",
            flush=True,
        )

        with h5py.File(fname, "w") as f:
            f["temp_guess"] = temp_guess_init
            f.attrs["nstr_upper_init"] = nstr_upper_init
            f.attrs["effective_temperature"] = out["spectrum_output"]["effective_temperature"]
            t10_this = t10(pressure_grid, out["temperature"])
            f.attrs["t10"] = t10_this
            print(f"t10 at {grav, tint, semi_major, fsed} = {t10_this:.3f}")
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
        
        del cl_run, out, temp_guess
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
        return False

opacity_ck = None
if rank == 0:
    print(f"Starting run with {sweep = }, {cloudy = }, {irradiated = }, {save = }, {rerun = } ")
    print(len(list(generate_tasks())))

# opacity_ck = jdi.opannection(ck_db=os.path.join(__refdata__, "climate_INPUTS", "661"),method='resortrebin',preload_gases=gases_fly)
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
