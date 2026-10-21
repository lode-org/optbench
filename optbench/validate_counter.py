"""Compare the harness counter with eonclient on a Lennard-Jones subset.

The subset is not a published mean. A published Lennard-Jones 38 mean
uses all 1000 starts. This run checks that the counting server and
``eonclient`` report the same force-call count on each selected geometry,
and that the server's shifted Lennard-Jones matches the client's
``ljcluster`` path.

``ljcluster`` is rgpot ``LJClusterPot`` with ``u0 = 1``, ``psi = 1``,
and cutoff 15. The template does not set a cutoff.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

from .campaign import format_ini, geometry_archive, load_template, parse_client_dat
from .lj import lj_cluster, read_component_coordinates
from .potserver import PotServer, remote_count, remote_reset
from .v2 import write_starts

LJ_U0 = 1.0
LJ_PSI = 1.0
LJ_CUTOFF = 15.0
BOX = (100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)


def _evaluate(positions, atomic_numbers, box):
    return lj_cluster(positions, atomic_numbers, box, u0=LJ_U0, psi=LJ_PSI, cutoff=LJ_CUTOFF)


def client_version(client: Path) -> tuple[str, str, str]:
    """``(version, commit, first line)`` from ``eonclient --version``.

    The binary prints a line such as ``3.5.0 (7ec1840)`` and a compile
    line. The commit is the token in parentheses, which is what that
    binary embeds. It is not looked up as a git object.
    """
    out = subprocess.run([str(client), "--version"], check=False, capture_output=True, text=True)
    text = (out.stdout or "") + (out.stderr or "")
    version = ""
    commit = ""
    first = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not first:
            first = line
        if line.startswith("compiled"):
            continue
        if "(" in line and line.endswith(")"):
            version, rest = line.split("(", 1)
            version = version.strip()
            commit = rest[:-1].strip()
            break
        if not version:
            version = line.split()[0]
    if not version:
        raise RuntimeError(f"{client} --version produced no version line")
    return version, commit, first


def _wrapper(path: Path, root: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import os
            import sys
            sys.path.insert(0, {str(root)!r})
            from optbench.extpot import main
            main()
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _ini(template: dict, extpot: Path | None) -> str:
    sections = {name: dict(values) for name, values in template["ini"].items()}
    if extpot is not None:
        sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(extpot)}
    return format_ini(sections)


def _run_eon(client: Path, ini: str, con: Path, work: Path) -> dict:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "config.ini").write_text(ini, encoding="utf-8")
    shutil.copy(con, work / "pos.con")
    env = os.environ.copy()
    env["OPTBENCH_POTSERVER"] = env.get("OPTBENCH_POTSERVER", "")
    env["OPTBENCH_ROOT"] = env.get("OPTBENCH_ROOT", "")
    completed = subprocess.run(
        [str(client)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    dat = work / "results.dat"
    calls = None
    reason = None
    raw = ""
    if dat.is_file():
        raw = dat.read_text(encoding="utf-8", errors="replace")
        calls, reason = parse_client_dat(raw)
    potcalls = None
    pot_path = work / "_potcalls.json"
    if pot_path.is_file():
        import json

        try:
            rows = json.loads(pot_path.read_text(encoding="utf-8"))
            potcalls = sum(int(row.get("force_calls", 0)) for row in rows)
        except (ValueError, TypeError):
            potcalls = None
    return {
        "returncode": completed.returncode,
        "force_calls": calls,
        "termination_reason": reason,
        "potcalls": potcalls,
        "results": raw,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def select_geometries(root: Path, limit: int, scratch: Path) -> list[Path]:
    """First ``limit`` ``.con`` files in sorted order from the LJ38 archive."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    archive = geometry_archive(root, {"id": "minimization/lj38", "input": {"archive_path": "website/minimization/lj38/lj38-clusters.tgz"}})
    dest = scratch / "geometries"
    dest.mkdir(parents=True, exist_ok=True)
    import tarfile

    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    found = sorted(dest.rglob("*.con"))
    if len(found) < limit:
        raise FileNotFoundError(f"archive has {len(found)} con files, needed {limit}")
    return found[:limit]


def validate_subset(
    root: Path,
    client: Path,
    output: Path,
    *,
    limit: int = 3,
    work: Path | None = None,
) -> dict:
    """Run ``limit`` LJ38 starts two ways and write the comparison.

    Built-in ``ljcluster`` is the client's own potential. ``ext_pot``
    sends every force evaluation to the counting server. The server
    count is the harness count.
    """
    template = load_template(root, "eon-client/lbfgs-m20")
    version, commit, version_line = client_version(client)
    memory = template["ini"]["Optimizer"].get("lbfgs_memory")
    scratch = work or output.parent / "work"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    geometries = select_geometries(root, limit, scratch)
    wrapper = scratch / "extpot-bridge"
    _wrapper(wrapper, root)
    server = PotServer(_evaluate).start()
    rows = []
    try:
        env_endpoint = server.endpoint()
        os.environ["OPTBENCH_POTSERVER"] = env_endpoint
        os.environ["OPTBENCH_ROOT"] = str(root)
        builtin_ini = _ini(template, None)
        external_ini = _ini(template, wrapper)
        for con in geometries:
            builtin = _run_eon(client, builtin_ini, con, scratch / "runs" / "builtin" / con.stem)
            remote_reset(server.host, server.port)
            external = _run_eon(client, external_ini, con, scratch / "runs" / "extpot" / con.stem)
            harness, verify = remote_count(server.host, server.port)
            client_calls = external["force_calls"]
            builtin_calls = builtin["force_calls"]
            matched_client = client_calls is not None and client_calls == harness
            matched_builtin = builtin_calls is not None and builtin_calls == harness
            client_dir = output.parent / f"{output.stem}-client"
            client_dir.mkdir(parents=True, exist_ok=True)
            for label in ("builtin", "extpot"):
                source = scratch / "runs" / label / con.stem / "results.dat"
                if source.is_file():
                    shutil.copy(source, client_dir / f"{con.stem}-{label}-results.dat")
                potcalls_path = scratch / "runs" / label / con.stem / "_potcalls.json"
                if potcalls_path.is_file():
                    shutil.copy(potcalls_path, client_dir / f"{con.stem}-{label}-potcalls.json")
            rows.append(
                {
                    "schema": "optbench.start/v2",
                    "start": con.stem,
                    "force_calls": harness,
                    "client_force_calls": client_calls,
                    "builtin_force_calls": builtin_calls,
                    "potcalls": external["potcalls"],
                    "builtin_potcalls": builtin["potcalls"],
                    "verify_calls": verify,
                    "counter": "harness",
                    "comparable": bool(matched_client),
                    "success": bool(matched_client and external["termination_reason"] in (None, "0")),
                    "failure": None if matched_client else "crash",
                    "matched_client": matched_client,
                    "matched_builtin": matched_builtin,
                    "termination_reason": external["termination_reason"],
                    "builtin_termination_reason": builtin["termination_reason"],
                    "builtin_returncode": builtin["returncode"],
                    "external_returncode": external["returncode"],
                }
            )
    finally:
        server.stop()
    matched = sum(1 for row in rows if row["matched_client"])
    builtin_matched = sum(1 for row in rows if row["matched_builtin"])
    point = point_disagreement(geometries[0].read_text(encoding="utf-8"), client, scratch / "point")
    lines = [
        "kind validation",
        "problem minimization/lj38",
        f"template {template['id']}",
        f"subset {limit}",
        "published_mean false",
        "counter harness",
        "client_counter eonclient",
        f"code {template.get('code', 'eOn')}",
        f"code_version {version}",
        f"commit {commit}",
        f"version_line {version_line}",
        f"lbfgs_memory {memory}",
        f"lj_u0 {LJ_U0}",
        f"lj_psi {LJ_PSI}",
        f"lj_cutoff {LJ_CUTOFF}",
        f"n {len(rows)}",
        f"matched_client {matched}",
        f"matched_builtin {builtin_matched}",
        "potential ljcluster",
        f"point_start {geometries[0].stem}",
        f"point_server_energy {point['server_energy']!r}",
        f"point_client_energy {point['client_energy']!r}",
        "note subset comparison of the harness counter and eonclient; not a published mean",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), rows)
    return {"rows": rows, "version": version, "commit": commit, "matched_client": matched, "matched_builtin": builtin_matched}


def point_disagreement(con_text: str, client: Path, work: Path) -> dict:
    """One point job. Returns the client's energy and the server energy."""
    positions = read_component_coordinates(con_text)
    numbers = [1] * (len(positions) // 3)
    energy, forces = lj_cluster(positions, numbers, BOX, u0=LJ_U0, psi=LJ_PSI, cutoff=LJ_CUTOFF)
    work.mkdir(parents=True, exist_ok=True)
    (work / "pos.con").write_text(con_text, encoding="utf-8")
    (work / "config.ini").write_text(
        "[Main]\njob = point\n\n[Potential]\npotential = ljcluster\n",
        encoding="utf-8",
    )
    completed = subprocess.run([str(client)], cwd=work, capture_output=True, text=True, check=False)
    client_energy = None
    for line in completed.stdout.splitlines():
        if line.startswith("Energy:"):
            client_energy = float(line.split()[1])
    return {
        "server_energy": energy,
        "server_force_norm": sum(value * value for value in forces) ** 0.5,
        "client_energy": client_energy,
        "stdout": completed.stdout,
    }
