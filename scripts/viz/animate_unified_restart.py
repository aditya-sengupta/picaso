#!/usr/bin/env python3
"""Animate the climate convergence histories in ``data/unified_restart``.

The left block shows the current pressure--temperature profile for every
available intrinsic temperature and irradiation level, with one panel per
gravity.  The matching panels in the right block show T10 against Tint, as in
``make_t10_table.py``, using the T10 value at the same climate-state index.
Tracks that finish early remain at their terminal state while the longer-running
models continue.  The archive's repeated restart/stage states are retained, and
the returned final profile is appended because it is normally absent from
``all_profiles``.

Examples
--------
Render the complete archive using the repository defaults::

    python scripts/viz/animate_unified_restart.py

Render a small diagnostic subset::

    python scripts/viz/animate_unified_restart.py \
        --gravities 10 17 --tints 100 150 --irradiations ns 0.02 \
        --max-steps 20 --output /tmp/unified_restart_subset.mp4

T10 is the temperature at 10 bar obtained by extending the deepest profile
point below 5199 K along the PICASO dry adiabat, matching ``calculate_t10.py``.
Computed histories are cached next to the movie because evaluating every
stored profile in the full archive takes appreciably longer than rendering a
small subset.  Color represents the intrinsic/input ``tint`` coordinate parsed
from each filename; the stored emergent ``effective_temperature`` is ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Callable, Iterable, Sequence
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")

from matplotlib import animation, colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "unified_restart"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "figures" / "unified_restart" / "unified_restart_convergence.mp4"
)

MODEL_RE = re.compile(
    r"^unified_tint(?P<tint>\d+)_grav(?P<gravity>\d+)_"
    r"(?P<irradiation>ns|semimajor(?P<semi_major>\d+(?:\.\d+)?))_"
    r"(?P<cloud>nc|f\d+)\.h5$"
)
T10_CACHE_VERSION = 2
T10_TEMPERATURE_LIMIT = 5199.0
EXCLUDED_GRAVITIES = {178, 1780}


@dataclass(frozen=True)
class ModelMetadata:
    """Parameters encoded in one unified-model filename."""

    path: Path
    tint: int
    gravity: int
    irradiation: str
    semi_major: float | None
    cloud: str

    @property
    def irradiation_label(self) -> str:
        if self.semi_major is None:
            return "No star"
        return f"a = {self.semi_major:g} AU"

    @property
    def irradiation_sort_key(self) -> float:
        return math.inf if self.semi_major is None else self.semi_major


@dataclass(frozen=True)
class ModelTrack:
    """In-memory profile and T10 histories for one model."""

    metadata: ModelMetadata
    pressure: np.ndarray
    profiles: np.ndarray
    t10: np.ndarray | None
    source_size: int
    source_mtime_ns: int
    terminal_t10: float | None
    terminal_is_last: bool
    converged: bool | None

    @property
    def nsteps(self) -> int:
        return self.profiles.shape[0]


@dataclass
class PanelArtists:
    """Mutable artists associated with one gravity."""

    tracks: list[ModelTrack]
    t10_groups: list[list[ModelTrack]]
    profile_lines: LineCollection
    t10_lines: LineCollection
    t10_points: object
    unconverged_points: object


def parse_model_path(path: Path) -> ModelMetadata | None:
    """Parse a unified filename, returning ``None`` for unrelated HDF5 files."""

    match = MODEL_RE.fullmatch(path.name)
    if match is None:
        return None
    groups = match.groupdict()
    semi_major = None if groups["semi_major"] is None else float(groups["semi_major"])
    return ModelMetadata(
        path=path,
        tint=int(groups["tint"]),
        gravity=int(groups["gravity"]),
        irradiation=groups["irradiation"],
        semi_major=semi_major,
        cloud=groups["cloud"],
    )


def normalize_irradiation(value: str) -> str:
    """Normalize CLI spellings to the filename token used by the archive."""

    value = value.strip().lower()
    if value in {"ns", "none", "no-star", "nostar"}:
        return "ns"
    if value.startswith("semimajor"):
        value = value[len("semimajor") :]
    try:
        return f"semimajor{float(value):.2f}"
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid irradiation {value!r}; use 'ns' or a semi-major axis in AU"
        ) from exc


def discover_models(
    input_dir: Path,
    gravities: set[int] | None,
    tints: set[int] | None,
    irradiations: set[str] | None,
    clouds: set[str] | None,
) -> tuple[list[ModelMetadata], list[Path]]:
    """Recursively discover and filter model files."""

    metadata: list[ModelMetadata] = []
    ignored: list[Path] = []
    for path in sorted(input_dir.rglob("*.h5")):
        model = parse_model_path(path)
        if model is None:
            ignored.append(path)
            continue
        if model.gravity in EXCLUDED_GRAVITIES:
            continue
        if gravities is not None and model.gravity not in gravities:
            continue
        if tints is not None and model.tint not in tints:
            continue
        if irradiations is not None and model.irradiation not in irradiations:
            continue
        if clouds is not None and model.cloud not in clouds:
            continue
        metadata.append(model)

    metadata.sort(
        key=lambda model: (
            model.gravity,
            model.irradiation_sort_key,
            model.tint,
            model.cloud,
            str(model.path),
        )
    )

    keys: dict[tuple[int, int, str, str], Path] = {}
    for model in metadata:
        key = (model.tint, model.gravity, model.irradiation, model.cloud)
        previous = keys.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate model parameters found in " f"{previous} and {model.path}"
            )
        keys[key] = model.path
    return metadata, ignored


def archive_signatures(
    metadata: Sequence[ModelMetadata],
) -> dict[Path, tuple[int, int]]:
    """Capture file size and nanosecond mtime for a stable-input check."""

    signatures: dict[Path, tuple[int, int]] = {}
    for model in metadata:
        try:
            stat = model.path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"input archive changed during discovery; missing {model.path}"
            ) from exc
        signatures[model.path] = (stat.st_size, stat.st_mtime_ns)
    return signatures


def _profiles_from_dataset(dataset: h5py.Dataset, nlevel: int) -> np.ndarray:
    """Return saved profiles as ``(step, level)`` without assuming storage rank."""

    profiles = np.asarray(dataset, dtype=float)
    if profiles.ndim == 2:
        if profiles.shape[1] != nlevel:
            raise ValueError(
                f"2-D all_profiles has shape {profiles.shape}; expected "
                f"(nstep, {nlevel})"
            )
        return profiles
    if profiles.ndim != 1:
        raise ValueError(f"all_profiles must be 1-D or 2-D, got {profiles.ndim}-D")
    if profiles.size % nlevel != 0:
        raise ValueError(
            f"all_profiles has {profiles.size} values, not a multiple of {nlevel}"
        )
    return profiles.reshape(-1, nlevel)


def load_track(metadata: ModelMetadata, max_steps: int | None) -> ModelTrack:
    """Load one profile history and append the returned terminal solution."""

    stat = metadata.path.stat()
    with h5py.File(metadata.path, "r") as handle:
        missing = {
            name
            for name in ("pressure", "all_profiles", "temperature")
            if name not in handle
        }
        if missing:
            raise KeyError(f"missing required datasets: {', '.join(sorted(missing))}")

        pressure = np.asarray(handle["pressure"], dtype=float).reshape(-1)
        profiles = _profiles_from_dataset(handle["all_profiles"], pressure.size)
        terminal = np.asarray(handle["temperature"], dtype=float).reshape(-1)
        terminal_t10_attr = handle.attrs.get("t10")
        converged = None
        if "converged" in handle:
            converged = int(np.asarray(handle["converged"])[()]) == 1

    if pressure.size < 2:
        raise ValueError("pressure must contain at least two levels")
    if terminal.shape != pressure.shape:
        raise ValueError(
            f"temperature shape {terminal.shape} does not match pressure {pressure.shape}"
        )
    if profiles.shape[0] == 0:
        raise ValueError("all_profiles is empty")
    if not np.all(np.isfinite(profiles)) or np.any(profiles <= 0):
        raise ValueError("all_profiles temperatures must be finite and positive")
    if not np.all(np.isfinite(terminal)) or np.any(terminal <= 0):
        raise ValueError("terminal temperature must be finite and positive")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("pressure must be finite and strictly positive")
    if not np.all(np.diff(pressure) > 0):
        raise ValueError("pressure must increase monotonically from top to bottom")

    # PICASO stores the profile entering each Newton step.  Its returned
    # temperature is normally the state just after the last stored profile, so
    # append it to make the movie visibly arrive at the saved solution.
    appended_terminal = not np.array_equal(profiles[-1], terminal, equal_nan=True)
    if appended_terminal:
        profiles = np.concatenate((profiles, terminal[None, :]), axis=0)

    terminal_is_last = True
    if max_steps is not None and profiles.shape[0] > max_steps:
        profiles = profiles[:max_steps]
        terminal_is_last = False

    terminal_t10: float | None = None
    if terminal_is_last and terminal_t10_attr is not None:
        candidate = float(terminal_t10_attr)
        if np.isfinite(candidate):
            terminal_t10 = candidate

    return ModelTrack(
        metadata=metadata,
        pressure=pressure,
        profiles=profiles,
        t10=None,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        terminal_t10=terminal_t10,
        terminal_is_last=terminal_is_last,
        converged=converged,
    )


def resolve_reference_data(explicit: Path | None) -> Path:
    """Find the PICASO reference-data directory needed for dry adiabats."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_path = os.environ.get("picaso_refdata")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(REPOSITORY_ROOT / "reference")

    relative = Path("climate_INPUTS") / "specific_heat_p_adiabat_grad.json"
    for candidate in candidates:
        if (candidate / relative).is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "could not find climate_INPUTS/specific_heat_p_adiabat_grad.json; "
        f"searched {searched}. Pass --refdata or set picaso_refdata."
    )


def build_t10_calculator(
    reference_data: Path,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build the same dry-adiabat T10 calculation used by ``calculate_t10``."""

    # Executing a file inside scripts/viz places that directory, rather than the
    # repository root, on sys.path.  Prefer this checkout's PICASO implementation
    # so the calculation works without requiring an editable installation.
    repository_string = str(REPOSITORY_ROOT)
    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)
    try:
        from picaso.grad import did_grad_cp
    except ImportError as exc:
        raise RuntimeError(
            "PICASO must be importable to compute the T10 histories"
        ) from exc

    table_path = reference_data / "climate_INPUTS" / "specific_heat_p_adiabat_grad.json"
    with table_path.open(encoding="utf-8") as stream:
        cp_grad = json.load(stream)

    adiabat_type = namedtuple("AdiabatBundle", ["t_table", "p_table", "grad", "cp"])
    adiabat = adiabat_type(
        np.asarray(cp_grad["temperature"], dtype=float),
        np.asarray(cp_grad["pressure"], dtype=float),
        np.asarray(cp_grad["adiabat_grad"], dtype=float),
        np.asarray(cp_grad["specific_heat"], dtype=float),
    )

    def dtdp(pressure: float, temperature: np.ndarray) -> float:
        value = float(np.asarray(temperature).item())
        gradient, _ = did_grad_cp(value, float(pressure), adiabat)
        return float(gradient) * value / float(pressure)

    def calculate_t10(pressure: np.ndarray, temperature: np.ndarray) -> float:
        eligible = np.flatnonzero(
            np.isfinite(temperature)
            & (temperature > 0)
            & (temperature < T10_TEMPERATURE_LIMIT)
        )
        if eligible.size == 0:
            return math.nan
        index = int(eligible[-1])
        solver = ode(dtdp).set_integrator("dopri5", rtol=1e-8, atol=1e-8, nsteps=5000)
        solver.set_initial_value(float(temperature[index]), float(pressure[index]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = solver.integrate(10.0)
        if not solver.successful():
            return math.nan
        return float(result[0])

    return calculate_t10


def t10_calculator_signature(reference_data: Path) -> str:
    """Identify the adiabat data and numerical recipe used by the T10 cache."""

    table_path = reference_data / "climate_INPUTS" / "specific_heat_p_adiabat_grad.json"
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    return f"dry-adiabat-dopri5-v1:{digest}"


class T10Cache:
    """Small persistent cache keyed by source path and file signature."""

    def __init__(self, path: Path | None, calculator_signature: str):
        self.path = path
        self.calculator_signature = calculator_signature
        self.handle: h5py.File | None = None

    def __enter__(self) -> "T10Cache":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = h5py.File(self.path, "a")
            self.handle.attrs["cache_version"] = T10_CACHE_VERSION
            self.handle.attrs["calculator_signature"] = self.calculator_signature
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is not None:
            self.handle.close()

    @staticmethod
    def _key(track: ModelTrack) -> str:
        source = str(track.metadata.path.resolve()).encode("utf-8")
        return hashlib.sha256(source).hexdigest()

    def get(self, track: ModelTrack) -> np.ndarray | None:
        if self.handle is None:
            return None
        key = self._key(track)
        if key not in self.handle:
            return None
        dataset = self.handle[key]
        valid = (
            int(dataset.attrs.get("cache_version", -1)) == T10_CACHE_VERSION
            and dataset.attrs.get("calculator_signature", "")
            == self.calculator_signature
            and int(dataset.attrs.get("source_size", -1)) == track.source_size
            and int(dataset.attrs.get("source_mtime_ns", -1)) == track.source_mtime_ns
            and int(dataset.attrs.get("nsteps", -1)) == track.nsteps
        )
        if not valid:
            return None
        values = np.asarray(dataset, dtype=float)
        return values if values.shape == (track.nsteps,) else None

    def put(self, track: ModelTrack, values: np.ndarray) -> None:
        if self.handle is None:
            return
        key = self._key(track)
        if key in self.handle:
            del self.handle[key]
        dataset = self.handle.create_dataset(key, data=values)
        dataset.attrs["cache_version"] = T10_CACHE_VERSION
        dataset.attrs["calculator_signature"] = self.calculator_signature
        dataset.attrs["source_size"] = track.source_size
        dataset.attrs["source_mtime_ns"] = track.source_mtime_ns
        dataset.attrs["nsteps"] = track.nsteps
        dataset.attrs["source"] = str(track.metadata.path.resolve())
        self.handle.flush()


def add_t10_history(
    track: ModelTrack,
    calculator: Callable[[np.ndarray, np.ndarray], float],
    cache: T10Cache,
) -> tuple[ModelTrack, bool]:
    """Attach a cached or newly computed T10 value to every profile step."""

    def validate_terminal(values: np.ndarray) -> None:
        if (
            track.terminal_is_last
            and track.terminal_t10 is not None
            and not np.isclose(values[-1], track.terminal_t10, rtol=1e-4, atol=0.05)
        ):
            warnings.warn(
                f"{track.metadata.path}: calculated terminal T10 "
                f"({values[-1]:.6g} K) differs from archived t10 "
                f"({track.terminal_t10:.6g} K)",
                RuntimeWarning,
            )

    cached = cache.get(track)
    if cached is not None:
        validate_terminal(cached)
        return replace(track, t10=cached), True

    values = np.asarray(
        [calculator(track.pressure, profile) for profile in track.profiles],
        dtype=float,
    )
    validate_terminal(values)
    cache.put(track, values)
    return replace(track, t10=values), False


def _finite_extent(arrays: Iterable[np.ndarray], name: str) -> tuple[float, float]:
    low = math.inf
    high = -math.inf
    for array in arrays:
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            low = min(low, float(np.min(finite)))
            high = max(high, float(np.max(finite)))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"no finite {name} values found")
    return low, high


def _irradiation_styles(tracks: Sequence[ModelTrack]) -> dict[str, object]:
    """Assign a stable dash pattern to every irradiation value."""

    irradiated = sorted(
        {
            (track.metadata.irradiation, track.metadata.irradiation_sort_key)
            for track in tracks
            if track.metadata.semi_major is not None
        },
        key=lambda item: item[1],
    )
    dash_cycle: list[object] = [
        (0, (1, 1)),
        (0, (4, 1.5)),
        (0, (4, 1, 1, 1)),
        (0, (7, 2)),
        (0, (2, 1, 6, 1)),
        (0, (8, 1, 1, 1, 1, 1)),
    ]
    styles = {
        token: dash_cycle[index % len(dash_cycle)]
        for index, (token, _) in enumerate(irradiated)
    }
    if any(track.metadata.semi_major is None for track in tracks):
        styles["ns"] = "solid"
    return styles


def _irradiation_widths(tracks: Sequence[ModelTrack]) -> dict[str, float]:
    """Reinforce irradiation strength with line width as well as dashing."""

    weak_to_strong = sorted(
        {
            (track.metadata.irradiation, track.metadata.irradiation_sort_key)
            for track in tracks
            if track.metadata.semi_major is not None
        },
        key=lambda item: item[1],
        reverse=True,
    )
    widths = {
        token: 0.85 + 0.15 * index for index, (token, _) in enumerate(weak_to_strong)
    }
    if any(track.metadata.semi_major is None for track in tracks):
        widths["ns"] = 0.70
    return widths


def _irradiation_legend_key(track: ModelTrack) -> tuple[int, float]:
    """Order the legend from no star through increasing irradiation."""

    if track.metadata.semi_major is None:
        return (0, 0.0)
    return (1, -track.metadata.semi_major)


def _t10_curve_groups(tracks: Sequence[ModelTrack]) -> list[list[ModelTrack]]:
    """Group tracks into the irradiation curves drawn against Tint."""

    grouped: dict[tuple[str, str], list[ModelTrack]] = defaultdict(list)
    for track in tracks:
        grouped[(track.metadata.irradiation, track.metadata.cloud)].append(track)
    return [
        sorted(group, key=lambda track: track.metadata.tint)
        for group in sorted(
            grouped.values(),
            key=lambda group: (
                _irradiation_legend_key(group[0]),
                group[0].metadata.cloud,
            ),
        )
    ]


def _frame_steps(max_steps: int, stride: int) -> list[int]:
    steps = list(range(0, max_steps, stride))
    if steps[-1] != max_steps - 1:
        steps.append(max_steps - 1)
    return steps


def create_animation(
    tracks: Sequence[ModelTrack],
    output: Path,
    *,
    columns: int,
    fps: int,
    dpi: int,
    frame_stride: int,
    bitrate: int,
    quiet: bool,
) -> None:
    """Build and encode the left/right P--T and T10 panel blocks."""

    gravities = sorted({track.metadata.gravity for track in tracks})
    ncols = min(columns, len(gravities))
    gravity_rows = math.ceil(len(gravities) / ncols)
    max_steps = max(track.nsteps for track in tracks)
    frames = _frame_steps(max_steps, frame_stride)

    pressure_low, pressure_high = _finite_extent(
        (track.pressure for track in tracks), "pressure"
    )
    temperature_limits = (0.0, 5000.0)
    t10_limits = (0.0, 5000.0)

    tint_low = min(track.metadata.tint for track in tracks)
    tint_high = max(track.metadata.tint for track in tracks)
    tint_limits = (tint_low - 10.0, tint_high + 10.0)
    if tint_low == tint_high:
        tint_norm = Normalize(tint_low - 1, tint_high + 1)
    else:
        tint_norm = Normalize(tint_low, tint_high)
    tint_cmap = colormaps["plasma"]
    irradiation_styles = _irradiation_styles(tracks)
    irradiation_widths = _irradiation_widths(tracks)

    grouped: dict[int, list[ModelTrack]] = defaultdict(list)
    for track in tracks:
        grouped[track.metadata.gravity].append(track)

    figure_width = max(12.0, 6.0 * ncols)
    figure_height = 3.0 * gravity_rows + 1.6
    # Reserve fixed physical space for the title/legend and colorbar.  Computing
    # the fractions from inches keeps those elements separated for both a
    # one-panel diagnostic render and the much taller full-grid figure.
    grid_top = 1.0 - 1.12 / figure_height
    grid_bottom = 1.08 / figure_height
    style = {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(figure_width, figure_height))
        grid = figure.add_gridspec(
            gravity_rows,
            2 * ncols + 1,
            left=0.045,
            right=0.985,
            bottom=grid_bottom,
            top=grid_top,
            hspace=0.34,
            wspace=0.28,
            width_ratios=[1.0] * ncols + [0.18] + [1.0] * ncols,
        )

        panel_artists: list[PanelArtists] = []
        first_profile_axis = None
        first_t10_axis = None

        for index, gravity in enumerate(gravities):
            block_row, column = divmod(index, ncols)
            profile_axis = figure.add_subplot(
                grid[block_row, column],
                sharex=first_profile_axis,
                sharey=first_profile_axis,
            )
            t10_axis = figure.add_subplot(
                grid[block_row, ncols + 1 + column],
                sharex=first_t10_axis,
                sharey=first_t10_axis,
            )
            if first_profile_axis is None:
                first_profile_axis = profile_axis
                first_t10_axis = t10_axis

            gravity_tracks = grouped[gravity]
            colors = [
                tint_cmap(tint_norm(track.metadata.tint)) for track in gravity_tracks
            ]
            linestyles = [
                irradiation_styles[track.metadata.irradiation]
                for track in gravity_tracks
            ]
            linewidths = [
                irradiation_widths[track.metadata.irradiation]
                for track in gravity_tracks
            ]
            t10_groups = _t10_curve_groups(gravity_tracks)
            t10_edge_colors: list[object] = []
            t10_edge_styles: list[object] = []
            t10_edge_widths: list[float] = []
            for curve_tracks in t10_groups:
                irradiation = curve_tracks[0].metadata.irradiation
                for left, right in zip(curve_tracks[:-1], curve_tracks[1:]):
                    midpoint_tint = 0.5 * (left.metadata.tint + right.metadata.tint)
                    t10_edge_colors.append(tint_cmap(tint_norm(midpoint_tint)))
                    t10_edge_styles.append(irradiation_styles[irradiation])
                    t10_edge_widths.append(irradiation_widths[irradiation])

            profile_lines = LineCollection(
                [],
                colors=colors,
                linestyles=linestyles,
                linewidths=linewidths,
                alpha=0.80,
                rasterized=True,
            )
            t10_line_kwargs: dict[str, object] = {}
            if t10_edge_colors:
                t10_line_kwargs = {
                    "colors": t10_edge_colors,
                    "linestyles": t10_edge_styles,
                    "linewidths": t10_edge_widths,
                }
            t10_lines = LineCollection(
                [], alpha=0.80, rasterized=True, **t10_line_kwargs
            )
            profile_axis.add_collection(profile_lines)
            t10_axis.add_collection(t10_lines)
            initial_t10 = np.asarray(
                [
                    track.t10[0] if track.t10 is not None else math.nan
                    for track in gravity_tracks
                ],
                dtype=float,
            )
            t10_points = t10_axis.scatter(
                [track.metadata.tint for track in gravity_tracks],
                initial_t10,
                s=5,
                c=colors,
                edgecolors="none",
                alpha=0.85,
                zorder=3,
            )
            unconverged_tracks = [
                track for track in gravity_tracks if track.converged is False
            ]
            unconverged_points = t10_axis.scatter(
                [track.metadata.tint for track in unconverged_tracks],
                [
                    track.t10[0] if track.t10 is not None else math.nan
                    for track in unconverged_tracks
                ],
                s=14,
                marker="x",
                c="0.15",
                linewidths=0.65,
                zorder=4,
            )

            gravity_title = rf"$g = {gravity:g}\ \mathrm{{m\,s^{{-2}}}}$"
            profile_axis.set(
                xlim=temperature_limits,
                ylim=(pressure_high, pressure_low),
                yscale="log",
                xlabel="Temperature (K)",
                title=gravity_title,
            )
            t10_axis.set(
                xlim=tint_limits,
                ylim=t10_limits,
                xlabel="Tint (K)",
                title=gravity_title,
            )
            profile_axis.grid(alpha=0.16, linewidth=0.4)
            t10_axis.grid(alpha=0.16, linewidth=0.4)
            if column == 0:
                profile_axis.set_ylabel("Pressure (bar)")
                t10_axis.set_ylabel(r"$T_{10}$ (K)")
            else:
                profile_axis.tick_params(labelleft=False)
                t10_axis.tick_params(labelleft=False)

            panel_artists.append(
                PanelArtists(
                    tracks=gravity_tracks,
                    t10_groups=t10_groups,
                    profile_lines=profile_lines,
                    t10_lines=t10_lines,
                    t10_points=t10_points,
                    unconverged_points=unconverged_points,
                )
            )

        title = figure.suptitle("", y=1.0 - 0.08 / figure_height, fontsize=11)
        legend_handles: list[Line2D] = []
        irradiation_by_token = {
            track.metadata.irradiation: track.metadata.irradiation_label
            for track in tracks
        }
        for token in sorted(
            irradiation_by_token,
            key=lambda item: next(
                _irradiation_legend_key(track)
                for track in tracks
                if track.metadata.irradiation == item
            ),
        ):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.25",
                    linewidth=irradiation_widths[token],
                    linestyle=irradiation_styles[token],
                    label=irradiation_by_token[token],
                )
            )
        if any(track.converged is False for track in tracks):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.15",
                    marker="x",
                    linestyle="none",
                    markersize=4,
                    label="solver converged = 0",
                )
            )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 - 0.34 / figure_height),
            ncol=min(len(legend_handles), 6),
            frameon=False,
            title="Line style/width: irradiation",
        )

        colorbar_width = min(0.44, 8.0 / figure_width)
        colorbar_axis = figure.add_axes(
            (
                0.5 - colorbar_width / 2,
                0.45 / figure_height,
                colorbar_width,
                0.08 / figure_height,
            )
        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=tint_norm, cmap=tint_cmap),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        if tint_low == tint_high:
            colorbar.set_ticks([tint_low])
        colorbar.set_label(r"Filename $T_{\mathrm{int}}$ coordinate (K)")

        def update(step: int) -> list[object]:
            changing = 0
            artists: list[object] = [title]
            for panel in panel_artists:
                profile_segments: list[np.ndarray] = []
                t10_segments: list[np.ndarray] = []
                endpoints: list[tuple[float, float]] = []
                unconverged_endpoints: list[tuple[float, float]] = []
                current_t10: dict[Path, float] = {}
                for track in panel.tracks:
                    track_step = min(step, track.nsteps - 1)
                    changing += int(
                        not track.terminal_is_last or step < track.nsteps - 1
                    )
                    profile_segments.append(
                        np.column_stack((track.profiles[track_step], track.pressure))
                    )
                    assert track.t10 is not None
                    t10_value = float(track.t10[track_step])
                    current_t10[track.metadata.path] = t10_value
                    endpoints.append((track.metadata.tint, t10_value))
                    if track.converged is False:
                        unconverged_endpoints.append((track.metadata.tint, t10_value))

                for curve_tracks in panel.t10_groups:
                    curve_points = [
                        (track.metadata.tint, current_t10[track.metadata.path])
                        for track in curve_tracks
                    ]
                    t10_segments.extend(
                        np.asarray((left, right), dtype=float)
                        for left, right in zip(curve_points[:-1], curve_points[1:])
                    )

                panel.profile_lines.set_segments(profile_segments)
                panel.t10_lines.set_segments(t10_segments)
                panel.t10_points.set_offsets(np.asarray(endpoints, dtype=float))
                if unconverged_endpoints:
                    unconverged_offsets = np.asarray(unconverged_endpoints, dtype=float)
                else:
                    unconverged_offsets = np.empty((0, 2), dtype=float)
                panel.unconverged_points.set_offsets(unconverged_offsets)
                artists.extend(
                    (
                        panel.profile_lines,
                        panel.t10_lines,
                        panel.t10_points,
                        panel.unconverged_points,
                    )
                )

            title.set_text(
                "Unified restart convergence"
                f"  |  climate state {step}/{max_steps - 1}"
                f"  |  {changing}/{len(tracks)} not yet at terminal output"
            )
            return artists

        update(frames[0])
        movie = animation.FuncAnimation(
            figure,
            update,
            frames=frames,
            interval=1000 / fps,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )

        if not animation.FFMpegWriter.isAvailable():
            plt.close(figure)
            raise RuntimeError("Matplotlib cannot find ffmpeg on PATH")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate,
            metadata={
                "title": "Unified restart convergence tracks",
                "artist": "PICASO",
            },
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )

        progress_bar = None
        progress_callback = None
        if not quiet:
            progress_bar = tqdm(
                total=len(frames),
                desc="Encoding",
                unit="frame",
                dynamic_ncols=True,
            )

            def update_progress(frame: int, total: int) -> None:
                assert progress_bar is not None
                progress_bar.total = total
                progress_bar.update(max(0, frame + 1 - progress_bar.n))

            progress_callback = update_progress

        try:
            movie.save(
                str(output),
                writer=writer,
                dpi=dpi,
                progress_callback=progress_callback,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
            plt.close(figure)


def print_inventory(metadata: Sequence[ModelMetadata], ignored: Sequence[Path]) -> None:
    """Print a concise summary before expensive HDF5/T10 work."""

    gravities = sorted({model.gravity for model in metadata})
    tints = sorted({model.tint for model in metadata})
    irradiations = Counter(model.irradiation for model in metadata)
    clouds = Counter(model.cloud for model in metadata)
    gravity_counts = Counter(model.gravity for model in metadata)
    print(f"Found {len(metadata)} matching model files")
    print(f"Gravity panels ({len(gravities)}): {', '.join(map(str, gravities))}")
    if tints:
        print(
            f"Tint coordinates ({len(tints)}): "
            + ", ".join(str(tint) for tint in tints)
            + " K"
        )
    print(
        "Models per gravity: "
        + ", ".join(f"{gravity}={gravity_counts[gravity]}" for gravity in gravities)
    )
    print(
        "Irradiation: "
        + ", ".join(f"{key}={value}" for key, value in sorted(irradiations.items()))
    )
    print(
        "Cloud modes: "
        + ", ".join(f"{key}={value}" for key, value in sorted(clouds.items()))
    )
    if ignored:
        print(f"Ignored {len(ignored)} HDF5 file(s) with unrecognized names")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Animate P--T convergence profiles and iteration-aligned T10-vs-Tint "
            "curves, with one paired panel per surface gravity."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"directory searched recursively for model HDF5 files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output MP4 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--refdata",
        type=Path,
        help="PICASO reference-data directory; defaults to picaso_refdata or ./reference",
    )
    parser.add_argument(
        "--gravities", nargs="+", type=int, help="gravity values to include"
    )
    parser.add_argument(
        "--tints", nargs="+", type=int, help="input Tint values to include"
    )
    parser.add_argument(
        "--irradiations",
        nargs="+",
        type=normalize_irradiation,
        help="irradiation values to include: ns and/or semi-major axes in AU",
    )
    parser.add_argument(
        "--clouds",
        nargs="+",
        help="cloud filename suffixes to include, such as nc or f2",
    )
    parser.add_argument(
        "--columns",
        type=positive_int,
        default=4,
        help="gravity panels per row in each left/right block (default: 4)",
    )
    parser.add_argument("--fps", type=positive_int, default=10)
    parser.add_argument("--dpi", type=positive_int, default=100)
    parser.add_argument(
        "--bitrate",
        type=positive_int,
        default=5000,
        help="H.264 bitrate in kbit/s (default: 5000)",
    )
    parser.add_argument(
        "--frame-stride",
        type=positive_int,
        default=1,
        help="render every Nth solver step while retaining true step numbers",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="keep only the first N saved steps (useful for diagnostics)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="T10 cache path (default: OUTPUT with .t10_cache.h5 suffix)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not read or write a T10 cache"
    )
    parser.add_argument(
        "--skip-bad",
        action="store_true",
        help="skip unreadable/malformed files instead of requiring every selected model",
    )
    parser.add_argument(
        "--converged-only",
        action="store_true",
        help="exclude files whose scalar converged dataset is not 1",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output MP4"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected archive inventory without loading or rendering",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress messages"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output.suffix.lower() != ".mp4":
        raise SystemExit(f"Output must have an .mp4 extension: {output}")
    if args.no_cache and args.cache is not None:
        raise SystemExit("--cache and --no-cache cannot be used together")

    metadata, ignored = discover_models(
        input_dir,
        set(args.gravities) if args.gravities else None,
        set(args.tints) if args.tints else None,
        set(args.irradiations) if args.irradiations else None,
        set(args.clouds) if args.clouds else None,
    )
    if not metadata:
        raise SystemExit(f"No matching unified model files found under {input_dir}")
    if not args.quiet:
        print_inventory(metadata, ignored)
        sys.stdout.flush()
    if args.dry_run:
        return 0
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite): {output}")

    try:
        initial_signatures = archive_signatures(metadata)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    reference_data = resolve_reference_data(args.refdata)
    calculator = build_t10_calculator(reference_data)
    calculator_signature = t10_calculator_signature(reference_data)
    if args.no_cache:
        cache_path = None
    elif args.cache is not None:
        cache_path = args.cache.expanduser().resolve()
    else:
        cache_path = output.with_suffix(".t10_cache.h5")
    if cache_path == output:
        raise SystemExit("The T10 cache path must differ from the MP4 output path")

    tracks: list[ModelTrack] = []
    skipped: list[tuple[Path, Exception]] = []
    cache_hits = 0
    convergence_filtered = 0
    with T10Cache(cache_path, calculator_signature) as cache:
        for index, model in enumerate(metadata, start=1):
            try:
                track = load_track(model, args.max_steps)
                if args.converged_only and track.converged is not True:
                    convergence_filtered += 1
                else:
                    track, cache_hit = add_t10_history(track, calculator, cache)
                    tracks.append(track)
                    cache_hits += int(cache_hit)
            except Exception as exc:
                if not args.skip_bad:
                    raise
                skipped.append((model.path, exc))
                print(f"Skipping {model.path}: {exc}", file=sys.stderr)
            if not args.quiet and (index % 100 == 0 or index == len(metadata)):
                print(
                    f"Prepared {index}/{len(metadata)} files "
                    f"({cache_hits} T10 cache hits, {len(skipped)} skipped, "
                    f"{convergence_filtered} convergence-filtered)",
                    file=sys.stderr,
                    flush=True,
                )

    if not tracks:
        raise SystemExit("No readable model tracks remain after validation")

    # The supplied archive may be synchronized by another process.  Rendering
    # a mixture of generations would no longer mean "all models", so refuse to
    # encode if any selected file appeared, disappeared, or changed while the
    # profiles/T10 histories were being prepared.  A rerun can reuse the cache.
    metadata_after, _ = discover_models(
        input_dir,
        set(args.gravities) if args.gravities else None,
        set(args.tints) if args.tints else None,
        set(args.irradiations) if args.irradiations else None,
        set(args.clouds) if args.clouds else None,
    )
    try:
        final_signatures = archive_signatures(metadata_after)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if final_signatures != initial_signatures:
        before_paths = set(initial_signatures)
        after_paths = set(final_signatures)
        changed = sum(
            initial_signatures[path] != final_signatures[path]
            for path in before_paths & after_paths
        )
        raise SystemExit(
            "Input archive changed while preparing the movie "
            f"({len(after_paths - before_paths)} added, "
            f"{len(before_paths - after_paths)} removed, {changed} modified). "
            "Wait for the archive update to finish and rerun; cached T10 values "
            "will be reused where valid."
        )
    nan_t10 = sum(int(np.count_nonzero(~np.isfinite(track.t10))) for track in tracks)
    if nan_t10:
        print(
            f"Warning: {nan_t10} saved profile step(s) have undefined T10 and "
            "will appear as gaps",
            file=sys.stderr,
        )

    if not args.quiet:
        step_counts = np.asarray([track.nsteps for track in tracks])
        print(
            f"Rendering {len(tracks)} tracks across "
            f"{len(set(track.metadata.gravity for track in tracks))} gravity panels; "
            f"steps min/median/max = {step_counts.min()}/"
            f"{int(np.median(step_counts))}/{step_counts.max()}; "
            f"{sum(track.converged is False for track in tracks)} marked unconverged",
            file=sys.stderr,
        )
    create_animation(
        tracks,
        output,
        columns=args.columns,
        fps=args.fps,
        dpi=args.dpi,
        frame_stride=args.frame_stride,
        bitrate=args.bitrate,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(f"Saved {output}")
        if cache_path is not None:
            print(f"T10 cache: {cache_path}")
        if skipped:
            print(f"Skipped {len(skipped)} malformed/unreadable file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
