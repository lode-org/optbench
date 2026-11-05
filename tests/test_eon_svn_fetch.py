"""The eOn SVN r2549 manifest and the per-file fetch that checks against it."""

import hashlib
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "codes" / "eon-svn" / "fetch.sh"
MANIFEST = ROOT / "codes" / "eon-svn" / "r2549.manifest"
ROW = re.compile(r"(?P<sha>[0-9a-f]{64})  +(?P<size>\d+)  (?P<mode>644|755)  (?P<path>\S+)")


def _rows(text: str) -> list[dict]:
    return [ROW.fullmatch(line).groupdict() for line in text.splitlines() if not line.startswith("#")]


def test_the_manifest_is_the_registry_revision():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "eon-svn")
    text = MANIFEST.read_text(encoding="utf-8")
    assert f"{row['repository']} at revision {row['commit'].removeprefix('r')}:" in text.splitlines()[0]
    rows = _rows(text)
    paths = [item["path"] for item in rows]
    assert paths == sorted(set(paths))
    assert len(rows) == 1037
    modes = {item["path"]: item["mode"] for item in rows}
    assert modes["client/version.sh"] == "755"
    assert modes["client/Rules.mk"] == "644"
    assert "client/potentials/Morse/Morse.cpp" in modes
    assert not any(path.startswith("client/potentials/QSC/") for path in paths)
    assert sum(path.rsplit("/", 1)[-1] == "POTCAR" for path in paths) == 7


@pytest.mark.skipif(shutil.which("curl") is None, reason="fetch.sh uses curl")
def test_fetch_checks_every_file_and_resumes(tmp_path: Path):
    source = tmp_path / "source"
    (source / "client").mkdir(parents=True)
    (source / "client" / "version.sh").write_bytes(b"#!/bin/sh\necho 2549\n")
    (source / "LICENSE").write_bytes(b"license\n")
    (source / "examples" / "vasp").mkdir(parents=True)
    (source / "examples" / "vasp" / "POTCAR").write_bytes(b"pseudopotential\n")
    manifest = tmp_path / "r2549.manifest"
    lines = ["# test manifest"]
    for path, mode in (("LICENSE", "644"), ("client/version.sh", "755"), ("examples/vasp/POTCAR", "644")):
        body = (source / path).read_bytes()
        lines.append(f"{hashlib.sha256(body).hexdigest()}  {len(body):>8d}  {mode}  {path}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dest = tmp_path / "dest"
    env = {"PATH": "/usr/bin:/bin", "EON_SVN_URL": source.as_uri()}

    def fetch(*flags):
        return subprocess.run(
            ["bash", str(FETCH), *flags, str(dest), str(manifest)], capture_output=True, text=True, env=env
        )

    first = fetch()
    assert first.returncode == 0, first.stderr
    assert "fetched 2, already present 0, POTCAR left out 1" in first.stdout
    assert (dest / "client" / "version.sh").stat().st_mode & 0o777 == 0o755
    assert (dest / "LICENSE").stat().st_mode & 0o777 == 0o644
    assert not (dest / "examples").exists()

    second = fetch()
    assert "fetched 0, already present 2, POTCAR left out 1" in second.stdout

    with_potcar = fetch("--with-potcar")
    assert with_potcar.returncode == 0, with_potcar.stderr
    assert "fetched 1, already present 2, POTCAR left out 0" in with_potcar.stdout
    assert (dest / "examples" / "vasp" / "POTCAR").read_bytes() == b"pseudopotential\n"

    (source / "LICENSE").write_bytes(b"changed\n")
    (dest / "LICENSE").unlink()
    third = fetch()
    assert third.returncode == 1
    assert "LICENSE: 8 bytes, sha256" in third.stderr
    assert not (dest / "LICENSE").exists()
