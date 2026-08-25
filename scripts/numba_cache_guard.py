"""Bound growth of rebuildable Numba cache artifacts."""

from dataclasses import dataclass
import os
import re
import stat


_NUMBA_CACHE_SUFFIXES = (".nbc", ".nbi")
_NUMBA_TEMP_RE = re.compile(r"\.nb[ci]\.tmp\.[0-9a-f]{16}$")


@dataclass(frozen=True)
class CacheGuardResult:
    before_bytes: int
    after_bytes: int
    removed_files: int
    removed_bytes: int

    @property
    def cleared(self):
        return self.removed_files > 0


def _regular_files(cache_dir):
    """Yield regular files below cache_dir without following nested symlinks."""
    def raise_walk_error(error):
        if isinstance(error, FileNotFoundError):
            return
        raise error

    walker = os.walk(cache_dir, onerror=raise_walk_error, followlinks=False)
    for root, _, filenames in walker:
        for filename in filenames:
            path = os.path.join(root, filename)
            if os.path.islink(path):
                continue
            try:
                mode = os.stat(path, follow_symlinks=False).st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(mode):
                continue
            yield path


def directory_size(cache_dir):
    """Return the apparent size of regular files below cache_dir."""
    total = 0
    for path in _regular_files(os.fspath(cache_dir)):
        try:
            total += os.stat(path, follow_symlinks=False).st_size
        except FileNotFoundError:
            # Another process may have removed a stale temporary file.
            pass
    return total


def is_numba_cache_artifact(path):
    """Identify Numba data, index, and interrupted atomic-write files."""
    filename = os.path.basename(os.fspath(path))
    return filename.endswith(_NUMBA_CACHE_SUFFIXES) or bool(
        _NUMBA_TEMP_RE.search(filename)
    )


def clear_numba_cache_artifacts(cache_dir):
    """Remove only rebuildable Numba artifacts, preserving Python bytecode."""
    removed_files = 0
    removed_bytes = 0
    artifacts = [
        path
        for path in _regular_files(os.fspath(cache_dir))
        if is_numba_cache_artifact(path)
    ]
    # Remove indexes first so an interrupted cleanup cannot leave an index
    # pointing to data that has already been removed.
    artifacts.sort(key=lambda path: ".nbi" not in os.path.basename(path))
    for path in artifacts:
        try:
            size = os.stat(path, follow_symlinks=False).st_size
            os.unlink(path)
        except FileNotFoundError:
            continue
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


def enforce_numba_cache_limit(cache_dir, max_bytes):
    """Clear Numba artifacts when the containing cache exceeds max_bytes."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    before_bytes = directory_size(cache_dir)
    if before_bytes <= max_bytes:
        return CacheGuardResult(
            before_bytes=before_bytes,
            after_bytes=before_bytes,
            removed_files=0,
            removed_bytes=0,
        )

    removed_files, removed_bytes = clear_numba_cache_artifacts(cache_dir)
    return CacheGuardResult(
        before_bytes=before_bytes,
        after_bytes=directory_size(cache_dir),
        removed_files=removed_files,
        removed_bytes=removed_bytes,
    )
