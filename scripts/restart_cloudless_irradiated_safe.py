#!/usr/bin/env python3
"""Memory-instrumented, restart-safe MPI cloudless irradiation driver.

The default scientific settings and task grid match
``restart_cloudless_irradiated.py``.  This variant adds:

* native thread caps before importing the scientific stack;
* MPI vendor, rank placement, CPU affinity, cgroup, and RSS diagnostics;
* optional ranks-per-node and tasks-per-rank limits for scaling tests;
* bounded retries with explicit cleanup between climate attempts;
* validated, same-directory temporary HDF5 writes followed by atomic replace;
* collective initialization failure handling and a non-zero final exit status.

Useful first run on a new machine::

    mpiexec -n 4 python -u scripts/restart_cloudless_irradiated_safe.py \
        --probe-only --ranks-per-node 1

Useful one-model-per-active-rank memory test::

    mpiexec -n 4 python -u scripts/restart_cloudless_irradiated_safe.py \
        --ranks-per-node 1 --max-tasks-per-rank 1 --no-write

SIGKILL cannot be caught by Python.  The last flushed ``SAFE`` phase line and
the scheduler's accounting record should identify the stage and resource peak.
"""

# These must be set before importing NumPy, SciPy, PICASO, or mpi4py.  Respect
# an explicit value supplied by the batch environment.
import os


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": None,
    "OMP_THREAD_LIMIT": None,
    "OPENBLAS_NUM_THREADS": None,
    "MKL_NUM_THREADS": None,
    "BLIS_NUM_THREADS": None,
    "NUMEXPR_NUM_THREADS": None,
    "VECLIB_MAXIMUM_THREADS": None,
}
_safe_threads = os.environ.get("PICASO_SAFE_THREADS_PER_RANK", "1")
if not _safe_threads.isdigit() or int(_safe_threads) < 1:
    raise RuntimeError("PICASO_SAFE_THREADS_PER_RANK must be a positive integer")
for _name in THREAD_ENV_DEFAULTS:
    os.environ[_name] = _safe_threads

import argparse
import faulthandler
import fcntl
import gc
import resource
import shutil
import signal
import socket
import sys
import traceback
import uuid
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(all_threads=True)
if hasattr(signal, "SIGUSR1"):
    # `kill -USR1 <pid>` prints every Python thread's stack for hang debugging.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "unified_restart"

# Ensure this checkout and its script helpers win over an unrelated installed
# PICASO checkout on the other machine.
for _path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


CLOUDMODE = "cloudless"
CALC_TYPE = "planet"
MH = "0.0"
CTOO = "0.46"
NLEVEL = 91
TINTS = tuple(range(100, 501, 50))
SEMI_MAJORS = (0.02, 0.13, 0.5)
GRAVS = (10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160)


mpi4py = None
psutil = None
MPI = None
PROCESS = None


def initialize_mpi_runtime():
    global mpi4py, psutil, MPI, PROCESS
    try:
        import mpi4py as mpi4py_module
        import psutil as psutil_module
        from mpi4py import MPI as mpi_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The safe driver requires mpi4py and psutil. Activate the Python "
            "environment built for this machine's MPI before launching it."
        ) from exc

    mpi4py = mpi4py_module
    psutil = psutil_module
    MPI = mpi_module
    PROCESS = psutil.Process()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe, instrumented MPI restart of cloudless irradiated models."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 0.04 au starting models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination directory (default: the original unified_restart directory).",
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=0,
        help="Use at most this many ranks on each host; 0 uses every launched rank.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=0,
        help="Fail unless MPI COMM_WORLD has this size; 0 accepts the discovered size.",
    )
    parser.add_argument(
        "--max-tasks-per-rank",
        type=int,
        default=0,
        help="Scaling-test limit after task distribution; 0 runs every assigned task.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Maximum pressure-reduction attempts for one model (default: 8).",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Print MPI/rank/memory diagnostics, then exit before scientific imports.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run calculations without writing or replacing any model files.",
    )
    parser.add_argument(
        "--no-save-all-profiles",
        action="store_true",
        help="Diagnostic: do not retain all climate iterations in memory/output.",
    )
    parser.add_argument(
        "--no-with-spec",
        action="store_true",
        help="Diagnostic: skip the final spectrum calculation.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that pass the safe HDF5 validation checks.",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help=(
            "Hide all warnings. By default, rank 0 prints each unique "
            "non-ResourceWarning once and the other ranks stay quiet."
        ),
    )
    args = parser.parse_args()

    if args.ranks_per_node < 0:
        parser.error("--ranks-per-node must be non-negative")
    if args.expected_world_size < 0:
        parser.error("--expected-world-size must be non-negative")
    if args.max_tasks_per_rank < 0:
        parser.error("--max-tasks-per-rank must be non-negative")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    reduced_output = args.no_save_all_profiles or args.no_with_spec
    if reduced_output and not args.no_write and args.output_dir == args.input_dir:
        parser.error(
            "Reduced-output diagnostics may not replace canonical inputs. "
            "Use --no-write or a separate --output-dir."
        )

    return args


def format_bytes(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _parse_cgroup_value(value, unlimited_threshold=None):
    if value is None:
        return None
    if value == "max":
        return value
    try:
        parsed = int(value)
    except ValueError:
        return value
    if unlimited_threshold is not None and parsed >= unlimited_threshold:
        return "max"
    return parsed


def cgroup_memory_snapshot():
    """Return useful cgroup-v2 or legacy cgroup-v1 memory fields."""
    cgroup_text = _read_text("/proc/self/cgroup")
    if not cgroup_text:
        return {}

    unified_relative = None
    memory_relative = None
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            unified_relative = fields[2]
        elif len(fields) == 3 and "memory" in fields[1].split(","):
            memory_relative = fields[2]

    if unified_relative is not None:
        base = Path("/sys/fs/cgroup") / unified_relative.lstrip("/")
        result = {"cgroup.version": 2}
        for name in ("memory.current", "memory.max", "memory.peak"):
            value = _parse_cgroup_value(_read_text(base / name))
            if value is not None:
                result[name] = value

        events = _read_text(base / "memory.events")
        if events:
            for line in events.splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0] in {"oom", "oom_kill", "max"}:
                    try:
                        result[f"event.{fields[0]}"] = int(fields[1])
                    except ValueError:
                        pass
        # A namespaced cgroup can expose /proc metadata but hide its files.
        if len(result) > 1:
            return result

    if memory_relative is None:
        return {}

    # Most HPC cgroup-v1 installations mount the memory controller here.
    # Try both layouts because some distributions mount each controller in a
    # subdirectory while others expose the controller at the cgroup root.
    relative = memory_relative.lstrip("/")
    candidates = (
        Path("/sys/fs/cgroup/memory") / relative,
        Path("/sys/fs/cgroup") / relative,
    )
    base = next(
        (candidate for candidate in candidates if (candidate / "memory.usage_in_bytes").exists()),
        None,
    )
    if base is None:
        return {}

    result = {"cgroup.version": 1}
    v1_fields = {
        "memory.usage_in_bytes": "memory.current",
        "memory.limit_in_bytes": "memory.max",
        "memory.max_usage_in_bytes": "memory.peak",
        "memory.failcnt": "event.failcnt",
    }
    for source_name, result_name in v1_fields.items():
        threshold = 1 << 60 if source_name == "memory.limit_in_bytes" else None
        value = _parse_cgroup_value(
            _read_text(base / source_name), unlimited_threshold=threshold
        )
        if value is not None:
            result[result_name] = value

    oom_control = _read_text(base / "memory.oom_control")
    if oom_control:
        for line in oom_control.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] in {"under_oom", "oom_kill"}:
                try:
                    result[f"event.{fields[0]}"] = int(fields[1])
                except ValueError:
                    pass
    return result


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def memory_values():
    info = PROCESS.memory_info()
    return {
        "rss": info.rss,
        "vms": info.vms,
        "peak_rss": peak_rss_bytes(),
        "available": psutil.virtual_memory().available,
        "cgroup": cgroup_memory_snapshot(),
    }


def mark(stage, rank, detail=""):
    """Emit a best-effort breadcrumb; diagnostics must never kill a rank."""
    try:
        memory = memory_values()
        cgroup = memory["cgroup"]
        cgroup_bits = []
        if "cgroup.version" in cgroup:
            cgroup_bits.append(f"cg_version={cgroup['cgroup.version']}")
        for key in ("memory.current", "memory.peak", "memory.max"):
            if key in cgroup:
                cgroup_bits.append(
                    f"cg_{key.split('.')[-1]}={format_bytes(cgroup[key])}"
                )
        for key in ("event.oom", "event.oom_kill", "event.max", "event.failcnt"):
            if key in cgroup:
                cgroup_bits.append(f"cg_{key.split('.')[-1]}={cgroup[key]}")
        suffix = f" {detail}" if detail else ""
        if cgroup_bits:
            suffix += " " + " ".join(cgroup_bits)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(
            f"SAFE time={timestamp} stage={stage} rank={rank} "
            f"host={socket.gethostname()} pid={os.getpid()} "
            f"rss={format_bytes(memory['rss'])} "
            f"peak_rss={format_bytes(memory['peak_rss'])} "
            f"vms={format_bytes(memory['vms'])} "
            f"node_available={format_bytes(memory['available'])}{suffix}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"SAFE DIAGNOSTIC_ERROR stage={stage} rank={rank} error={exc!r}",
            file=sys.stderr,
            flush=True,
        )


def cpu_affinity():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            return sorted(PROCESS.cpu_affinity())
        except (AttributeError, psutil.Error):
            return []


def numeric_env(name):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def mpi_preflight(comm, expected_world_size=0):
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()
    record = {
        "rank": rank,
        "host": hostname,
        "pid": os.getpid(),
        "affinity": cpu_affinity(),
        "python": sys.executable,
        "mpi4py": mpi4py.__file__,
        "vendor": repr(MPI.get_vendor()),
        "library": " ".join(MPI.Get_library_version().split()),
        "threads": tuple(
            (name, os.environ.get(name)) for name in THREAD_ENV_DEFAULTS
        ),
    }
    records = comm.allgather(record)

    signature_fields = ("python", "mpi4py", "vendor", "library", "threads")
    heterogeneous = {
        field: sorted({str(item[field]) for item in records})
        for field in signature_fields
        if len({str(item[field]) for item in records}) > 1
    }

    reliable_mismatches = []
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        hint = numeric_env(name)
        if hint is not None and hint != size:
            reliable_mismatches.append(f"{name}={hint} but MPI COMM_WORLD size={size}")
    if expected_world_size and expected_world_size != size:
        reliable_mismatches.append(
            f"--expected-world-size={expected_world_size} but MPI COMM_WORLD size={size}"
        )
    mismatch_records = comm.gather((rank, reliable_mismatches), root=0)

    if rank == 0:
        host_counts = Counter(item["host"] for item in records)
        print("SAFE MPI PREFLIGHT", flush=True)
        print(f"  python={records[0]['python']}", flush=True)
        print(f"  mpi4py={records[0]['mpi4py']}", flush=True)
        print(f"  mpiexec_on_path={shutil.which('mpiexec')}", flush=True)
        print(f"  vendor={records[0]['vendor']}", flush=True)
        print(f"  library={records[0]['library']}", flush=True)
        print(f"  world_size={size} host_counts={dict(sorted(host_counts.items()))}", flush=True)
        for host in sorted(host_counts):
            host_records = [item for item in records if item["host"] == host]
            affinity_sizes = sorted({len(item["affinity"]) for item in host_records})
            world_ranks = [item["rank"] for item in host_records]
            rank_affinity = {
                item["rank"]: item["affinity"] for item in host_records
            }
            print(
                f"  host={host} ranks={world_ranks} affinity_sizes={affinity_sizes} "
                f"rank_affinity={rank_affinity}",
                flush=True,
            )
        thread_values = {name: os.environ.get(name) for name in THREAD_ENV_DEFAULTS}
        print(f"  native_thread_limits={thread_values}", flush=True)
        scheduler_values = {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_NTASKS",
                "SLURM_NNODES",
                "SLURM_TASKS_PER_NODE",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "PBS_JOBID",
                "PBS_NODEFILE",
                "LSB_JOBID",
                "OMPI_COMM_WORLD_SIZE",
                "PMI_SIZE",
            )
            if os.environ.get(name) is not None
        }
        print(f"  scheduler_environment={scheduler_values}", flush=True)

        slurm_tasks = numeric_env("SLURM_NTASKS")
        if slurm_tasks is not None and slurm_tasks != size:
            print(
                f"SAFE WARNING: SLURM_NTASKS={slurm_tasks}, but MPI world size={size}. "
                "This can be intentional for a scaling test; otherwise verify the launcher.",
                flush=True,
            )
        all_mismatches = [
            f"rank {item_rank}: {message}"
            for item_rank, messages in mismatch_records
            for message in messages
        ]
        if all_mismatches:
            print("SAFE FATAL: launcher/MPI size mismatch detected:", flush=True)
            for message in all_mismatches:
                print(f"  {message}", flush=True)
        if heterogeneous:
            print(
                f"SAFE FATAL: rank environments/MPI libraries differ: {heterogeneous}",
                flush=True,
            )

    local_bad = bool(reliable_mismatches) or bool(heterogeneous)
    any_bad = comm.allreduce(local_bad, op=MPI.LOR)
    return not any_bad, record


def summarize_node_rss(comm, stage, loaded_opacity=False):
    rank = comm.Get_rank()
    try:
        sample = (
            socket.gethostname(),
            PROCESS.memory_info().rss,
            peak_rss_bytes(),
            bool(loaded_opacity),
            None,
        )
    except Exception as exc:
        # Preserve collective ordering even if diagnostics fail on one rank.
        sample = (socket.gethostname(), None, None, bool(loaded_opacity), repr(exc))
    samples = comm.gather(sample, root=0)
    if rank != 0:
        return

    grouped = defaultdict(list)
    for sample_item in samples:
        grouped[sample_item[0]].append(sample_item)
    print(f"SAFE NODE RSS SUMMARY stage={stage}", flush=True)
    for host, host_samples in sorted(grouped.items()):
        rss_values = [item[1] for item in host_samples if item[1] is not None]
        peak_values = [item[2] for item in host_samples if item[2] is not None]
        rss_sum = sum(rss_values) if rss_values else None
        max_peak = max(peak_values) if peak_values else None
        loaded = sum(item[3] for item in host_samples)
        errors = [item[4] for item in host_samples if item[4]]
        print(
            f"  host={host} ranks={len(host_samples)} opacity_ranks={loaded} "
            f"rank_rss_sum={format_bytes(rss_sum)} "
            f"max_rank_peak={format_bytes(max_peak)} diagnostic_errors={errors}",
            flush=True,
        )


def collective_errors(comm, stage, local_error):
    errors = comm.gather((comm.Get_rank(), local_error), root=0)
    any_error = comm.allreduce(local_error is not None, op=MPI.LOR)
    if comm.Get_rank() == 0 and any_error:
        print(f"SAFE FATAL: {stage} failed on one or more ranks", flush=True)
        for failed_rank, message in errors:
            if message:
                print(f"  rank={failed_rank}: {message}", flush=True)
    return any_error


def load_io_stack():
    global h5py, np
    import h5py
    import numpy as np


def load_science_stack():
    global picaso, jdi, u, out_to_hdf5, t10, regrid_initial_guess

    load_io_stack()
    import picaso
    import picaso.justdoit as jdi
    import astropy.units as u

    from out_to_hdf5 import out_to_hdf5
    from calculate_t10 import regrid_initial_guess, t10


def semi_major_str(semi_major):
    if semi_major == -1:
        return "ns"
    return f"semimajor{semi_major:.2f}"


def fname_from_params(directory, grav, tint, semi_major):
    stem = f"unified_tint{tint}_grav{grav}_{semi_major_str(semi_major)}_nc"
    return Path(directory) / f"{stem}.h5"


def generate_tasks():
    for grav in GRAVS:
        for tint in TINTS:
            for semi_major in SEMI_MAJORS:
                yield grav, tint, semi_major


def validate_profile_arrays(pressure, temperature, label):
    if pressure.ndim != 1 or temperature.ndim != 1:
        raise ValueError(f"{label}: pressure and temperature must be one-dimensional")
    if len(pressure) != len(temperature):
        raise ValueError(f"{label}: pressure and temperature lengths differ")
    if len(pressure) != NLEVEL:
        raise ValueError(f"{label}: expected {NLEVEL} profile levels, got {len(pressure)}")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError(f"{label}: pressure contains non-finite/non-positive values")
    if not np.all(np.diff(pressure) > 0):
        raise ValueError(f"{label}: pressure is not strictly increasing")
    if not np.all(np.isfinite(temperature)):
        raise ValueError(f"{label}: temperature contains non-finite values")


def initial_guess(path):
    required = ("pressure", "temperature", "cvz_locs")
    try:
        with h5py.File(path, "r") as handle:
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"missing datasets {missing}")
            pressure_guess = np.asarray(handle["pressure"])
            temp_guess = np.asarray(handle["temperature"])
            cvz_locs = np.asarray(handle["cvz_locs"])
    except OSError as exc:
        raise OSError(f"cannot open starting HDF5 file {path}: {exc}") from exc

    validate_profile_arrays(pressure_guess, temp_guess, str(path))
    if cvz_locs.ndim != 1 or len(cvz_locs) < 6:
        raise ValueError(f"{path}: cvz_locs is malformed")
    if len(pressure_guess) <= 89:
        raise ValueError(f"{path}: profile does not contain requested RCB index 89")

    # Preserve the original driver's deliberate override.
    return pressure_guess, temp_guess, 89


def validate_output_file(path, expect_spectrum, expect_all_profiles):
    required_datasets = [
        "pressure",
        "temperature",
        "cvz_locs",
        "converged",
        "temp_guess",
    ]
    if expect_spectrum:
        required_datasets.extend(
            ("spectrum_output_wavenumber", "spectrum_output_thermal")
        )
    if expect_all_profiles:
        required_datasets.extend(("all_profiles", "all_opd", "all_kzz"))
    required_attrs = ("nstr_upper_init", "t10")
    try:
        with h5py.File(path, "r") as handle:
            missing_datasets = [name for name in required_datasets if name not in handle]
            missing_attrs = [name for name in required_attrs if name not in handle.attrs]
            if expect_spectrum and "effective_temperature" not in handle.attrs:
                missing_attrs.append("effective_temperature")
            if missing_datasets or missing_attrs:
                return False, (
                    f"missing datasets={missing_datasets}, attrs={missing_attrs}"
                )

            pressure = np.asarray(handle["pressure"])
            temperature = np.asarray(handle["temperature"])
            temp_guess = np.asarray(handle["temp_guess"])
            cvz_locs = np.asarray(handle["cvz_locs"])
            converged = np.asarray(handle["converged"])
            validate_profile_arrays(pressure, temperature, str(path))
            if temp_guess.shape != temperature.shape or not np.all(np.isfinite(temp_guess)):
                return False, "temp_guess is malformed or non-finite"
            if cvz_locs.ndim != 1 or len(cvz_locs) < 6 or not np.all(
                np.isfinite(cvz_locs)
            ):
                return False, "cvz_locs is malformed or non-finite"
            if converged.size != 1:
                return False, "converged is not scalar"
            if not np.all(np.isfinite(converged)) or converged.item() not in (0, 1):
                return False, "converged is not a finite boolean value"
            if not np.all(np.isfinite(handle.attrs["nstr_upper_init"])):
                return False, "nstr_upper_init attribute is non-finite"
            if not np.all(np.isfinite(handle.attrs["t10"])):
                return False, "t10 attribute is non-finite"
            if expect_spectrum and not np.all(
                np.isfinite(handle.attrs["effective_temperature"])
            ):
                return False, "effective_temperature attribute is non-finite"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)
    return True, "ok"


def job_identifier():
    raw = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("PBS_JOBID")
        or os.environ.get("LSB_JOBID")
        or "nojob"
    )
    return "".join(character if character.isalnum() else "_" for character in raw)


def acquire_output_lock(destination, rank):
    """Take a non-blocking advisory lock to reject concurrent writers."""
    lock_path = Path(destination).with_name(f".{Path(destination).name}.lock")
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"another process/job owns output lock {lock_path}; refusing a write race"
        ) from exc
    except OSError as exc:
        lock_handle.close()
        raise RuntimeError(f"cannot acquire output lock {lock_path}: {exc}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(
        f"job={job_identifier()} host={socket.gethostname()} "
        f"rank={rank} pid={os.getpid()} time={datetime.now(timezone.utc).isoformat()}\n"
    )
    lock_handle.flush()
    return lock_handle


def probe_output_locking(directory):
    """Fail before model work if this filesystem cannot provide `flock`."""
    probe_path = Path(directory) / f".restart_safe_lock_probe.{uuid.uuid4().hex}"
    probe_handle = None
    try:
        probe_handle = probe_path.open("w")
        fcntl.flock(probe_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe_handle.fileno(), fcntl.LOCK_UN)
    finally:
        if probe_handle is not None:
            probe_handle.close()
        try:
            probe_path.unlink()
        except OSError:
            pass


def atomic_write_output(
    destination,
    out,
    temp_guess_init,
    nstr_upper_init,
    pressure_grid,
    expect_spectrum,
    expect_all_profiles,
    rank,
):
    destination = Path(destination)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{job_identifier()}.{socket.gethostname()}."
        f"rank{rank}.pid{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with h5py.File(temporary, "w") as handle:
            handle["temp_guess"] = temp_guess_init
            handle.attrs["nstr_upper_init"] = nstr_upper_init
            if expect_spectrum:
                handle.attrs["effective_temperature"] = out["spectrum_output"][
                    "effective_temperature"
                ]
            t10_value = t10(pressure_grid, out["temperature"])
            handle.attrs["t10"] = t10_value
            out_to_hdf5(out, handle)
            handle.flush()

        # HDF5 close may write metadata after flush; fsync the closed file.
        sync_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(sync_descriptor)
        finally:
            os.close(sync_descriptor)

        valid, reason = validate_output_file(
            temporary, expect_spectrum, expect_all_profiles
        )
        if not valid:
            raise RuntimeError(f"temporary output validation failed: {reason}")
        os.replace(temporary, destination)
        return t10_value
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def run_task(grav, tint, semi_major, opacity_ck, args, rank):
    label = f"grav={grav},tint={tint},semi_major={semi_major}"
    destination = fname_from_params(args.output_dir, grav, tint, semi_major)

    pressure_start = None
    temperature_start = None
    pressure_grid = None
    temp_guess = None
    temp_guess_init = None
    cl_run = None
    out = None
    output_lock = None

    try:
        mark("task-start", rank, label)
        if not args.no_write:
            output_lock = acquire_output_lock(destination, rank)
            mark("output-lock-acquired", rank, f"{label} destination={destination}")
        source = fname_from_params(args.input_dir, grav, tint, 0.04)
        pressure_start, temperature_start, nstr_upper = initial_guess(source)
        mark("initial-guess-loaded", rank, f"{label} source={source}")

        max_pressure = float(np.max(pressure_start))
        for attempt in range(1, args.max_attempts + 1):
            pressure_grid = np.logspace(
                np.log10(np.min(pressure_start)),
                np.log10(max_pressure * 16.0),
                NLEVEL,
            )
            temp_guess = regrid_initial_guess(
                pressure_start, temperature_start, pressure_grid
            )
            nstr_upper_init = nstr_upper
            temp_guess_init = np.copy(temp_guess)
            mark(
                "model-setup-start",
                rank,
                f"{label} attempt={attempt}/{args.max_attempts} "
                f"max_pressure={max_pressure:.6g}",
            )

            cl_run = jdi.inputs(calculation=CALC_TYPE, climate=True)
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit("m/(s**2)"))
            cl_run.effective_temp(tint)
            cl_run.star(
                opacity_ck,
                temp=5778.0,
                metal=0.0,
                logg=4.4,
                radius=1.0,
                database="phoenix",
                radius_unit=u.R_sun,
                semi_major=semi_major,
                semi_major_unit=u.AU,
            )
            mark("star-configured", rank, f"{label} attempt={attempt}")

            cl_run.inputs_climate(
                temp_guess=temp_guess,
                pressure=pressure_grid,
                rcb_guess=nstr_upper,
                rfacv=0.5,
            )
            cl_run.atmosphere(mh=1, cto_relative=1, chem_method="visscher")
            mark("before-climate", rank, f"{label} attempt={attempt}")
            out = cl_run.climate(
                opacity_ck,
                save_all_profiles=not args.no_save_all_profiles,
                with_spec=not args.no_with_spec,
                verbose=False,
            )
            mark("after-climate", rank, f"{label} attempt={attempt}")

            temperatures = np.asarray(out["temperature"])
            if not np.all(np.isfinite(temperatures)):
                raise RuntimeError("climate returned non-finite temperatures")
            max_temperature = float(np.max(temperatures))
            converged = bool(np.asarray(out.get("converged", False)).item())
            print(
                f"SAFE RESULT rank={rank} {label} attempt={attempt} "
                f"max_temperature={max_temperature:.6g} converged={converged}",
                flush=True,
            )
            if not converged:
                print(
                    f"SAFE WARNING rank={rank} {label}: PICASO reported converged=False",
                    flush=True,
                )
            if max_temperature < 5100.0:
                break

            if attempt == args.max_attempts:
                raise RuntimeError(
                    f"temperature remained too hot ({max_temperature:.3f} K) "
                    f"after {args.max_attempts} attempts"
                )

            print(
                f"SAFE RETRY rank={rank} {label}: too hot; reducing max pressure",
                flush=True,
            )
            max_pressure /= 2.0

            # Release the previous large result before the next climate call.  In
            # the original assignment, the old RHS stayed referenced until the
            # next RHS completed, allowing two attempts to overlap in memory.
            cl_run = None
            out = None
            temp_guess = None
            temp_guess_init = None
            gc.collect()
            mark("retry-cleanup", rank, f"{label} next_attempt={attempt + 1}")
        else:
            raise RuntimeError("internal error: retry loop ended without a result")

        if args.no_write:
            print(f"SAFE NO-WRITE rank={rank} {label}", flush=True)
        else:
            mark("before-hdf5-write", rank, f"{label} destination={destination}")
            t10_value = atomic_write_output(
                destination,
                out,
                temp_guess_init,
                nstr_upper_init,
                pressure_grid,
                expect_spectrum=not args.no_with_spec,
                expect_all_profiles=not args.no_save_all_profiles,
                rank=rank,
            )
            mark("after-hdf5-write", rank, f"{label} destination={destination}")
            print(
                f"SAFE SAVED rank={rank} {label} t10={t10_value:.3f} path={destination}",
                flush=True,
            )
        return True
    except Exception as exc:
        print(f"SAFE ERROR rank={rank} {label}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False
    finally:
        cl_run = None
        out = None
        if output_lock is not None:
            output_lock.close()
            output_lock = None
        pressure_start = None
        temperature_start = None
        pressure_grid = None
        temp_guess = None
        temp_guess_init = None
        gc.collect()
        mark("task-finalized", rank, label)


def inspect_opacity_file(path, rank):
    if rank != 0:
        return
    with h5py.File(path, "r") as handle:
        if "kcoeffs" not in handle:
            print(f"SAFE WARNING: opacity file has no kcoeffs dataset: {path}", flush=True)
            return
        dataset = handle["kcoeffs"]
        uncompressed = int(dataset.size * dataset.dtype.itemsize)
        print(
            f"SAFE OPACITY path={path} file_size={format_bytes(path.stat().st_size)} "
            f"kcoeffs_shape={dataset.shape} dtype={dataset.dtype} "
            f"uncompressed_kcoeffs_per_rank={format_bytes(uncompressed)}",
            flush=True,
        )


def main():
    args = parse_args()
    initialize_mpi_runtime()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI ranks otherwise repeat the same warning many times.  Keep one useful
    # copy from rank 0, hide noisy third-party file-cleanup ResourceWarnings,
    # and leave exceptions/tracebacks completely unaffected.
    warnings.resetwarnings()
    if args.suppress_warnings or rank != 0:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("once")
        warnings.filterwarnings("ignore", category=ResourceWarning)
    mark("process-start", rank)

    preflight_ok, preflight_record = mpi_preflight(
        comm, expected_world_size=args.expected_world_size
    )
    if not preflight_ok:
        return 2

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    local_rank = node_comm.Get_rank()
    local_size = node_comm.Get_size()
    node_comm.Free()
    active = args.ranks_per_node == 0 or local_rank < args.ranks_per_node
    active_world_ranks = [
        world_rank
        for world_rank, is_active in enumerate(comm.allgather(active))
        if is_active
    ]

    if rank == 0:
        print(
            f"SAFE CONFIG tasks={len(tuple(generate_tasks()))} world_ranks={size} "
            f"active_ranks={len(active_world_ranks)} ranks_per_node_limit={args.ranks_per_node or 'all'} "
            f"expected_world_size={args.expected_world_size or 'discovered'} "
            f"max_tasks_per_rank={args.max_tasks_per_rank or 'all'} "
            f"save_all_profiles={not args.no_save_all_profiles} "
            f"with_spec={not args.no_with_spec} no_write={args.no_write} "
            f"input_dir={args.input_dir} output_dir={args.output_dir}",
            flush=True,
        )
        if size > len(tuple(generate_tasks())):
            print(
                "SAFE WARNING: more MPI ranks were launched than tasks; idle ranks will not load opacity.",
                flush=True,
            )

    if args.probe_only:
        summarize_node_rss(comm, "probe-only")
        return 0

    tasks = list(generate_tasks())
    if active:
        active_rank = active_world_ranks.index(rank)
        assigned_tasks = tasks[active_rank::len(active_world_ranks)]
    else:
        assigned_tasks = []

    if not args.no_write and rank == 0:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            directory_error = None
        except OSError as exc:
            directory_error = f"cannot create output directory {args.output_dir}: {exc}"
    else:
        directory_error = None
    directory_error = comm.bcast(directory_error, root=0)
    if directory_error:
        if rank == 0:
            print(f"SAFE FATAL: {directory_error}", file=sys.stderr, flush=True)
        return 2
    if not args.no_write and rank == 0:
        try:
            probe_output_locking(args.output_dir)
            lock_probe_error = None
        except Exception as exc:
            lock_probe_error = (
                f"output filesystem does not support the required advisory lock: {exc}"
            )
    else:
        lock_probe_error = None
    lock_probe_error = comm.bcast(lock_probe_error, root=0)
    if lock_probe_error:
        if rank == 0:
            print(f"SAFE FATAL: {lock_probe_error}", file=sys.stderr, flush=True)
        return 2
    comm.Barrier()

    # Inactive/idle ranks remain lightweight so --ranks-per-node genuinely
    # lowers aggregate node memory.  Rank 0 imports only NumPy/HDF5 for schema
    # validation and opacity metadata.
    io_needed = rank == 0 or bool(assigned_tasks)
    local_error = None
    if io_needed:
        try:
            load_io_stack()
        except Exception:
            local_error = traceback.format_exc()
    if collective_errors(comm, "NumPy/HDF5 imports", local_error):
        return 2
    if io_needed:
        mark("numpy-hdf5-imports-complete", rank)
    else:
        mark("numpy-hdf5-imports-skipped", rank, "inactive/idle rank")

    local_skipped = 0
    work_tasks = []
    for task in assigned_tasks:
        if args.skip_existing and not args.no_write:
            destination = fname_from_params(args.output_dir, *task)
            valid, reason = validate_output_file(
                destination,
                expect_spectrum=not args.no_with_spec,
                expect_all_profiles=not args.no_save_all_profiles,
            )
            if valid:
                local_skipped += 1
                print(
                    f"SAFE SKIP rank={rank} task={task} valid_output={destination}",
                    flush=True,
                )
                continue
            if destination.exists():
                print(
                    f"SAFE RECOMPUTE rank={rank} task={task} invalid_output={destination} "
                    f"reason={reason}",
                    flush=True,
                )
        work_tasks.append(task)

    if args.max_tasks_per_rank:
        work_tasks = work_tasks[: args.max_tasks_per_rank]

    print(
        f"SAFE ASSIGNMENT rank={rank} host={socket.gethostname()} local_rank={local_rank} "
        f"local_size={local_size} active={active} assigned={len(assigned_tasks)} "
        f"work={len(work_tasks)} "
        f"skipped={local_skipped}",
        flush=True,
    )

    global_work = comm.allreduce(bool(work_tasks), op=MPI.LOR)
    source_errors = []
    source_paths = sorted(
        {
            fname_from_params(args.input_dir, grav, tint, 0.04)
            for grav, tint, _semi_major in work_tasks
        }
    )
    for source in source_paths:
        try:
            initial_guess(source)
        except Exception as exc:
            source_errors.append(f"{source}: {exc}")
    local_error = "; ".join(source_errors) if source_errors else None
    if collective_errors(comm, "starting-profile validation", local_error):
        return 2
    if work_tasks:
        mark(
            "starting-profiles-validated",
            rank,
            f"unique_sources={len(source_paths)}",
        )

    refdata = os.environ.get("picaso_refdata")
    ck_path = None
    if global_work:
        pysyn_cdbs = os.environ.get("PYSYN_CDBS")
        validation_errors = []
        if not refdata:
            validation_errors.append("picaso_refdata is unset")
        elif not Path(refdata).is_dir():
            validation_errors.append(
                f"picaso_refdata directory does not exist: {refdata}"
            )
        if not pysyn_cdbs:
            validation_errors.append(
                "PYSYN_CDBS is unset (required by the Phoenix stellar setup)"
            )
        elif not Path(pysyn_cdbs).is_dir():
            validation_errors.append(
                f"PYSYN_CDBS directory does not exist: {pysyn_cdbs}"
            )
        local_error = "; ".join(validation_errors) if validation_errors else None
        if collective_errors(comm, "reference-data validation", local_error):
            return 2

        ck_path = (
            Path(refdata)
            / "opacities"
            / "preweighted"
            / f"sonora_2121grid_feh{MH}_co{CTOO}.hdf5"
        )
        local_error = (
            None if ck_path.is_file() else f"opacity file does not exist: {ck_path}"
        )
        if collective_errors(comm, "opacity-path validation", local_error):
            return 2
        try:
            inspect_opacity_file(ck_path, rank)
            local_error = None
        except Exception:
            local_error = traceback.format_exc()
        if collective_errors(comm, "opacity metadata inspection", local_error):
            return 2

    local_error = None
    if work_tasks:
        try:
            load_science_stack()
        except Exception:
            local_error = traceback.format_exc()
    if collective_errors(comm, "scientific imports", local_error):
        return 2
    if work_tasks:
        mark("scientific-imports-complete", rank)
    else:
        mark("scientific-imports-skipped", rank, "no assigned work")

    local_picaso_path = str(picaso.__file__) if work_tasks else None
    picaso_paths = comm.gather(local_picaso_path, root=0)
    if rank == 0:
        unique_picaso_paths = sorted({path for path in picaso_paths if path})
        print(f"SAFE PICASO modules={unique_picaso_paths}", flush=True)

    opacity_ck = None
    local_error = None
    if work_tasks:
        try:
            mark("before-opacity-load", rank)
            opacity_ck = jdi.opannection(ck_db=str(ck_path), method="preweighted")
            mark("after-opacity-load", rank)
        except Exception:
            local_error = traceback.format_exc()
    else:
        mark("opacity-load-skipped", rank, "no assigned work")
    if collective_errors(comm, "opacity initialization", local_error):
        return 2
    comm.Barrier()
    summarize_node_rss(comm, "after-opacity-load", loaded_opacity=opacity_ck is not None)

    local_completed = 0
    local_failed = 0
    for grav, tint, semi_major in work_tasks:
        result = run_task(grav, tint, semi_major, opacity_ck, args, rank)
        if result:
            local_completed += 1
        else:
            local_failed += 1
        gc.collect()
        mark("between-tasks", rank)

    opacity_ck = None
    gc.collect()
    mark("opacity-released", rank)
    summarize_node_rss(comm, "final", loaded_opacity=False)

    completed = comm.allreduce(local_completed, op=MPI.SUM)
    failed = comm.allreduce(local_failed, op=MPI.SUM)
    skipped = comm.allreduce(local_skipped, op=MPI.SUM)
    scheduled = comm.allreduce(len(work_tasks), op=MPI.SUM)

    if rank == 0:
        print("\n=== Safe restart summary ===", flush=True)
        print(f"Scheduled this invocation: {scheduled}", flush=True)
        print(f"Completed: {completed}", flush=True)
        print(f"Failed: {failed}", flush=True)
        print(f"Skipped valid outputs: {skipped}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception:
        print("SAFE UNHANDLED ERROR", file=sys.stderr, flush=True)
        traceback.print_exc()
        if MPI is not None and MPI.Is_initialized() and not MPI.Is_finalized():
            _abort_comm = MPI.COMM_WORLD
            if _abort_comm.Get_size() > 1:
                _abort_comm.Abort(2)
        _exit_code = 2
    sys.exit(_exit_code)
