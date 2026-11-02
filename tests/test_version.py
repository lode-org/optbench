"""The OptBench version has one handwritten copy."""

import ast
import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

from optbench import __version__
from optbench.project_version import (
    GENERATED_RELATIVE,
    generated_paths,
    project_root,
    project_version,
)
from optbench.quadratic import build_smoke
from optbench.statistics import build_statistics

ROOT = project_root()
SKIP_PARTS = {".git", ".venv", ".pixi", ".asv", "__pycache__", "results", "softsaddle"}


def _has_cog_block(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        if stripped.startswith("[[[cog"):
            return True
    return False


def _discover_cog_files(root: Path) -> tuple[str, ...]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if "docs/build" in path.parts or "docs/doctrees" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if _has_cog_block(text):
            found.append(path.relative_to(root).as_posix())
    return tuple(sorted(found))


def _cog_output_lines(text: str) -> set[int]:
    output = set()
    emitting = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.endswith("]]]") and "[[[end]]]" not in stripped and "[[[cog" not in stripped:
            emitting = True
            continue
        if "[[[end]]]" in stripped:
            emitting = False
            continue
        if emitting:
            output.add(number)
    return output


def _quoted(version: str) -> tuple[str, str]:
    return (f'"{version}"', f"'{version}'")


def test_generated_files_match_the_handwritten_version():
    canonical = project_version(ROOT)
    discovered = _discover_cog_files(ROOT)
    assert discovered == tuple(sorted(GENERATED_RELATIVE))
    completed = subprocess.run(
        [sys.executable, "-m", "cogapp", "--check", "--diff", *map(str, generated_paths(ROOT))],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert __version__ == canonical
    assert importlib.metadata.version("optbench") == canonical

    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi["workspace"]["version"] == canonical

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = [item["version"] for item in lock["package"] if item["name"] == "optbench"]
    assert locked == [canonical]

    module = ast.parse((ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8"))
    assigned = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"version", "release"} and isinstance(node.value, ast.Constant):
                assigned[name] = node.value.value
    assert assigned == {"version": canonical, "release": canonical}

    smoke = build_smoke()
    for run in smoke["runs"]:
        assert run["code_version"] == canonical
        assert run["provenance"]["code_version"] == canonical
    statistics = build_statistics(commit="", machine="validation")
    assert statistics["provenance"]["code_version"] == canonical


def test_the_version_literal_is_only_in_allowed_copies():
    canonical = project_version(ROOT)
    quotes = _quoted(canonical)
    allowed_generated = set(GENERATED_RELATIVE)
    offenders = []
    pyproject_hits = []
    lock_hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if "docs/build" in path.parts or "docs/doctrees" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        output_lines = _cog_output_lines(text) if relative in allowed_generated else set()
        for number, line in enumerate(text.splitlines(), start=1):
            if not any(quote in line for quote in quotes):
                continue
            if relative in allowed_generated and number in output_lines:
                continue
            if relative == "pyproject.toml":
                pyproject_hits.append((number, line.strip()))
                continue
            if relative == "uv.lock":
                lock_hits.append((number, line.strip()))
                continue
            offenders.append(f"{relative}:{number}:{line.strip()}")
    assert offenders == []
    assert [line for _number, line in pyproject_hits] == [f'version = "{canonical}"']
    assert [line for _number, line in lock_hits] == [f'version = "{canonical}"']
