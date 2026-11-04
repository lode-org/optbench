"""The Wales ``svn.tar.bz2`` that GMIN, GMIN basin-hopping, and OPTIM build from.

The ``gmin`` row of ``codes/registry.toml`` names the URL. It serves a
snapshot that rolls daily, and the build scripts take the latest one, or
the local copy that ``WALES_TARBALL`` names. No sha256 is pinned. Each
build keeps the sha256 of the tarball it read and the ``SVNREV`` of the
tree it compiled. Every run writes both into the header of its raw log and
into each of its records, as ``wales_sha256`` and ``wales_svnrev``.
"""

from __future__ import annotations

import re
from pathlib import Path

SHA256_STAMP = "svn.tar.bz2.sha256"
REVISION_STAMP = "SVNREV"
LATEST = "latest"


def snapshot(sha256: str, svnrev: str) -> dict:
    """The ``wales_sha256`` and ``wales_svnrev`` of one build."""
    sha256 = str(sha256).strip()
    svnrev = str(svnrev).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise RuntimeError(f"tarball sha256 {sha256 or '(empty)'} is not 64 hexadecimal digits")
    if not svnrev:
        raise RuntimeError("the build names no SVNREV")
    return {"wales_sha256": sha256, "wales_svnrev": svnrev}


def stamped_sha256(folder: Path) -> str:
    """The sha256 that a build script wrote to ``folder/svn.tar.bz2.sha256``."""
    path = Path(folder) / SHA256_STAMP
    if not path.is_file():
        raise RuntimeError(f"{path} is missing. The build scripts under codes/gmin and codes/optim write it.")
    words = path.read_text(encoding="utf-8").split()
    return words[0] if words else ""


def same_build(record: dict, found: dict) -> bool:
    """True when ``record`` names the build ``found``."""
    return all(record.get(key) == value for key, value in found.items())


def header(found: dict) -> list[str]:
    """The raw-log header lines that name the bytes of a run."""
    return [f"# wales_sha256 {found['wales_sha256']}", f"# wales_svnrev {found['wales_svnrev']}"]
