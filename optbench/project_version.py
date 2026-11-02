"""The handwritten OptBench version.

``[project].version`` in ``pyproject.toml`` is the only copy written by
hand. Cog regenerates the other checked-in copies
(https://nedbatchelder.com/code/cog/,
https://cog.readthedocs.io/en/latest/running.html).
Run-time callers read the installed metadata
(https://docs.python.org/3/library/importlib.metadata.html#importlib.metadata.version),
because a normal install has no pyproject.toml. This module is for cog and
the tests only.
The field is the project version in
https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#version.
Hatchling copies it into the installed distribution
(https://hatch.pypa.io/latest/config/metadata/#project-metadata).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Checked-in files whose version text is cog output. ``pyproject.toml`` is
# not in this tuple.
GENERATED_RELATIVE = (
    "docs/source/conf.py",
    "optbench/__init__.py",
    "pixi.toml",
)


def project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        manifest = candidate / "pyproject.toml"
        package = candidate / "optbench" / "__init__.py"
        if manifest.is_file() and package.is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found")


def project_version(start: Path | None = None) -> str:
    root = project_root(start)
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    if not isinstance(version, str) or version.strip() == "":
        raise ValueError("project.version must be a non-empty string")
    return version


def generated_paths(start: Path | None = None) -> list[Path]:
    root = project_root(start)
    return [root / relative for relative in GENERATED_RELATIVE]


def cog_out(cog_module, template: str) -> None:
    """Write one cog line containing the handwritten version."""
    version = project_version(Path(cog_module.inFile))
    cog_module.outl(template.format(version=version))
