"""Global optimization of Lennard-Jones clusters with OGOLEM.

One force call is one harness evaluation of ``lennard_jones``: epsilon 1,
sigma 1, cutoff 40, no shift. OGOLEM stores cartesians in bohr and its
collision radii assume the argon Lennard-Jones sigma. Seeds are written
so one reduced sigma is that many bohr. The backend divides by the same
sigma. A repeated geometry is one call. The refused call at the force
cap is not counted.

The search is the Ar38 Sweden example: pool 100, 19900 iterations, LBFGS
at the OGOLEM defaults. One archived geometry is centered on the origin,
because ``setRandomCOM`` draws the rest of the pool about the origin, and
written as one xyz seed. The RNG seed is the start index plus one.

A start succeeds when the problem states ``energy_above_minimum`` and a
reference energy, the lowest harness energy is at or below that sum, and
the call count is under the cap. The reference is the problem's Cambridge
energy when that block is present. The cap is ``max_force_calls`` when the
problem states one, and the template ``failure_cap`` otherwise. Without
that rule the start is not a success. A finite finish under the cap is
``wrong_stationary``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import socketserver
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

COMMIT = "cce35a376d929852ab1e80e98105c0d13d784b8d"
ANGTOBOHR = 1.889725989
# AtomicProperties.ljSigmaTable Ar. Radii in that table are bohr; Ar is 1.85.
# A packed reduced cluster at Constants.ANGTOBOHR sits inside the collision
# threshold 1.2*(1.85+1.85) and every Sweden child is discarded.
AR_SIGMA_BOHR = 6.434516991
XYZ_SCALE = AR_SIGMA_BOHR / ANGTOBOHR
# The jar's version line is PRINTED_VERSION. The row uses the year.
CODE_VERSION = "2016"
PRINTED_VERSION = "2016.CURRENT"
# Archive eon-basinhopping stop. reference_energy + 0.001 lands here.
LJ38_STOP_ENERGY = -173.927427
GLOB_ALGO = (
    "cluster{xover(multiple:75%sweden:cutstyle=2|25%sweden:cutstyle=1)"
    "mutation(multiple:75%montecarlo:mode=some|25%montecarlo:mode=one)}"
)
_NUMBER = re.compile(r"(\d+)")


def ogolem_home() -> Path:
    override = os.environ.get("OGOLEM_HOME")
    if override:
        return Path(override)
    return Path.home() / "opt" / "ogolem" / COMMIT


def java_bin() -> Path:
    override = os.environ.get("OGOLEM_JAVA")
    if override:
        return Path(override)
    bundled = Path.home() / "opt" / "jdk-25" / "bin" / "java"
    if bundled.is_file():
        return bundled
    home = os.environ.get("JAVA_HOME")
    if home:
        return Path(home) / "bin" / "java"
    return Path("java")


def jars_ready(home: Path | None = None) -> bool:
    root = home or ogolem_home()
    backend = root / "optbench-backend.jar"
    snapshot = root / "ogolem-snapshot.jar"
    stamp = root / "COMMIT"
    if not backend.is_file() or not snapshot.is_file() or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == COMMIT


def _ensure_exploded(home: Path) -> Path:
    classes = home / "exploded" / "BOOT-INF" / "classes"
    libs = home / "exploded" / "BOOT-INF" / "lib"
    if classes.is_dir() and libs.is_dir():
        return home / "exploded"
    if not jars_ready(home):
        raise RuntimeError(f"{home} has no OGOLEM jars for commit {COMMIT}")
    exploded = home / "exploded"
    exploded.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(Path(java_bin()).with_name("jar")), "xf", str(home / "ogolem-snapshot.jar"), "BOOT-INF"],
        cwd=exploded,
        check=True,
    )
    if not classes.is_dir():
        raise RuntimeError(f"{home} snapshot has no BOOT-INF/classes")
    return exploded


def _classpath(home: Path) -> str:
    exploded = _ensure_exploded(home)
    return os.pathsep.join(
        [
            str(exploded / "BOOT-INF" / "classes"),
            str(exploded / "BOOT-INF" / "lib" / "*"),
            str(home / "optbench-backend.jar"),
        ]
    )


def classify_start(
    *,
    exit_code: int,
    calls: int,
    energy: float | None,
    cap: int,
    threshold: float | None,
    hit_cap: bool,
) -> tuple[bool, str | None, list[str]]:
    """Success bit, primary failure, and the reason list."""
    finite = energy is not None and math.isfinite(energy)
    reasons: list[str] = []
    if exit_code not in (0, 3):
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif exit_code == 3 or hit_cap or calls >= cap:
        reasons.append("cap")
    elif threshold is not None and energy is not None and energy <= threshold and calls < cap:
        pass
    else:
        # classify_failure maps an energy rejection to wrong_stationary.
        reasons.append("energy")
    failure = classify_failure(reasons)
    return failure is None, failure, reasons


def center_cluster(positions) -> "object":
    """Translate a cluster so its centroid is the origin."""
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
    pos -= pos.mean(axis=0)
    return pos


def _sort_key(name: str) -> tuple[int, str]:
    match = _NUMBER.search(name)
    return (int(match.group(1)) if match else -1, name)


def _num(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def ogo_text(
    *,
    atoms: int,
    pool: int,
    iterations: int,
    cell: float,
    seed: int,
    threshold: float | None,
) -> str:
    """One cluster input. Pool size must be at least 2."""
    if pool < 2:
        raise ValueError("OGOLEM rejects a pool smaller than 2")
    cell_text = _num(cell)
    lines = [
        "###OGOLEM###",
        "<GEOMETRY>",
        f"NumberOfParticles={atoms}",
        "<MOLECULE>",
        f"MoleculeRepetitions={atoms}",
        "Ar;0.0;0.0;0.0",
        "</MOLECULE>",
        "</GEOMETRY>",
        "<CLUSTERBACKEND>",
        "BackendTag=lj",
        "Backend=xyz:optbench",
        "</CLUSTERBACKEND>",
        f"CellSize={cell_text};{cell_text};{cell_text}",
        f"NumberOfGlobIterations={iterations}",
        f"PoolSize={pool}",
        "GeneticRecordsToSerial=1000000",
        "GeometriesToSerial=100000",
        "SerializeAfterNewBest=false",
        "LocOptAlgo=lbfgs:backend=lj",
        f"GlobOptAlgo={GLOB_ALGO}",
        "SeedingPath=seeds",
        f"RandomNumberGenerator=javarng:seed={seed}",
    ]
    if threshold is not None:
        lines.append(f"AcceptableFitness={threshold:.16g}")
    return "\n".join(lines) + "\n"


def write_xyz(path: Path, positions) -> None:
    """Write a seed. OGOLEM multiplies the file by ``ANGTOBOHR`` on read."""
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3) * XYZ_SCALE
    lines = [str(pos.shape[0]), "optbench seed"]
    for row in pos:
        lines.append(f"Ar {row[0]:.16e} {row[1]:.16e} {row[2]:.16e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class HarnessLJServer(socketserver.ThreadingTCPServer):
    """One Lennard-Jones counter, shared by every socket from one JVM."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *, cap: int, threshold: float | None, trace: bool = False):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.counter = HarnessCounter(cache=True, cap=cap)
        self.threshold = None if threshold is None else float(threshold)
        self.trace = trace
        self.traces: list[dict] = []
        self.best_energy: float | None = None
        self.found = False
        self.hit_cap = False
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, name="ogolem-lj", daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self.server_address
        return f"{host}:{port}"

    def close(self) -> None:
        self.shutdown()
        self.server_close()

    def dispatch(self, line: str) -> str | None:
        text = line.strip()
        if not text:
            return None
        if text == "hello":
            return "ok\n"
        if text == "bye":
            return None
        if not text.startswith("calculate"):
            return f"error unexpected {text.split()[0]}\n"
        parts = text.split()
        try:
            atoms = int(parts[1])
            coords = [float(value) for value in parts[2:]]
        except (IndexError, ValueError) as exc:
            return f"error {exc}\n"
        if len(coords) != 3 * atoms:
            return "error coordinate count\n"
        with self.lock:
            try:
                energy, gradient, calls = self._evaluate(coords)
            except ForceCallCap:
                self.hit_cap = True
                best = "nan" if self.best_energy is None else repr(self.best_energy)
                return f"cap {self.counter.calls} {best}\n"
            if self.trace:
                self.traces.append({"energy": energy, "coords": list(coords), "calls": calls})
            if self.best_energy is None or energy < self.best_energy:
                self.best_energy = energy
            cap = self.counter.cap
            if (
                self.threshold is not None
                and energy <= self.threshold
                and cap is not None
                and calls < cap
            ):
                self.found = True
                return f"found {calls} {energy!r}\n"
            body = " ".join(repr(float(value)) for value in gradient)
            return f"ok {calls} {energy!r} {body}\n"

    def _evaluate(self, coords: list[float]) -> tuple[float, tuple[float, ...], int]:
        key = tuple(coords)
        _fresh, cached = self.counter.charge(key)
        if cached is not None:
            energy, gradient = cached
            return float(energy), tuple(gradient), int(self.counter.calls)
        energy, gradient = lennard_jones(coords)
        stored = (float(energy), tuple(float(value) for value in gradient))
        self.counter.remember(key, stored)
        return stored[0], stored[1], int(self.counter.calls)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(60)
        server: HarnessLJServer = self.server
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    return
                reply = server.dispatch(raw.decode("ascii"))
                if reply is None:
                    return
                self.wfile.write(reply.encode("ascii"))
                self.wfile.flush()
        except (ConnectionError, TimeoutError, socket.timeout):
            return


def _geometry_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")]
    cons = [path for path in files if path.suffix == ".con"]
    if cons:
        return sorted(cons, key=lambda path: _sort_key(path.name))
    # A bare ``coords`` file in the lj98 and lj150 archives repeats coords.100.
    # The archived loops read coords.<index> only.
    coords = [
        path
        for path in files
        if _NUMBER.search(path.name) and (path.name.startswith("coords.") or path.suffix == ".xyz")
    ]
    if not coords:
        raise FileNotFoundError(f"{root} has no .con or coords files")
    return sorted(coords, key=lambda path: _sort_key(path.name))


def _read_rows(path: Path):
    import numpy as np

    if path.suffix == ".con":
        pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        return pos
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path} has no coordinates")
    return np.asarray(rows, dtype=np.float64)


def _archive_of(root: Path, problem: dict, template: dict) -> Path:
    from optbench.campaign import geometry_archive

    try:
        return geometry_archive(root, problem)
    except ValueError:
        archive_path = (template.get("input") or {}).get("archive_path")
        if not archive_path:
            raise
        return geometry_archive(root, {"id": problem["id"], "input": {"archive_path": archive_path}})


def load_starts(root: Path, problem: dict, template: dict, work: Path) -> list[tuple[str, object]]:
    """Archived geometries in numeric order. ``.con`` wins when both exist."""
    import tarfile

    archive = _archive_of(root, problem, template)
    dest = work / "geometries" / problem["id"].replace("/", "-")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    recorded = f"{archive.name}:{archive.stat().st_size}"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == recorded:
        pass
    else:
        if any(dest.iterdir()):
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text(recorded, encoding="utf-8")
    loaded = []
    for path in _geometry_files(dest):
        loaded.append((path.name, _read_rows(path)))
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(loaded) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(loaded)}")
    return loaded


def _threshold(problem: dict, template: dict) -> float | None:
    convergence = problem.get("convergence") or {}
    tolerance = convergence.get("energy_above_minimum")
    if tolerance in (None, ""):
        return None
    problem_ref = (problem.get("reference") or {}).get("energy")
    template_ref = template.get("reference_energy")
    if problem_ref not in (None, ""):
        reference = float(problem_ref)
        if template_ref not in (None, "") and abs(float(template_ref) - reference) > 1e-9:
            raise ValueError(
                f"{template.get('id', 'template')} reference_energy {template_ref} "
                f"is not the problem reference {reference}"
            )
    elif template_ref not in (None, ""):
        reference = float(template_ref)
    else:
        reference = convergence.get("minimum", convergence.get("reference_energy"))
        if reference in (None, ""):
            return None
    return float(reference) + float(tolerance)


def _force_cap(problem: dict, template: dict) -> int:
    """Problem ``max_force_calls`` when stated, otherwise the template cap."""
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    raw = template.get("failure_cap")
    if stated not in (None, ""):
        cap = int(stated)
        if raw not in (None, "") and int(raw) != cap:
            raise ValueError(f"{template.get('id', 'template')} failure_cap {raw} is not the problem cap {cap}")
        return cap
    if raw in (None, ""):
        raise ValueError(f"{template.get('id', 'template')} needs failure_cap")
    return int(raw)


def _ogolem_settings(template: dict) -> dict:
    section = template.get("ogolem") or {}
    return {
        "pool": int(section.get("pool_size", 100)),
        "iterations": int(section.get("glob_iterations", 19900)),
        "cell": float(section.get("cell_bohr", 60)),
        "threads": int(section.get("threads", 1)),
    }


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _partial_reason(problem: dict, records: list[dict], scored: bool, full: int | None) -> str:
    energies = [
        float(record["energy"])
        for record in records
        if record.get("energy") is not None and math.isfinite(float(record["energy"]))
    ]
    calls = [float(record["force_calls"]) for record in records]
    parts = []
    if not scored:
        parts.append(f"no success rule on {problem['id']}")
    if full is not None and len(records) < int(full):
        parts.append(f"scored {len(records)} of {int(full)} starts")
    elif full is None:
        parts.append(f"{len(records)} archive starts")
    if energies:
        lowest = min(energies)
        printed = f"{lowest:.8f}"
        parts.append(f"lowest energy {printed}")
        labels = [f"{energy:.8f}" for energy in energies]
        same = sum(1 for label in labels if label == printed)
        if same >= 2:
            parts.append(f"{same} of {len(records)} starts at {printed}")
        counts = Counter(labels)
        modal, modal_n = counts.most_common(1)[0]
        if modal != printed and modal_n >= 2:
            parts.append(f"{modal_n} of {len(records)} starts at {modal}")
    if calls:
        parts.append(f"mean calls {statistics.fmean(calls):.1f}")
        parts.append(f"median calls {statistics.median(calls):.1f}")
    if not scored:
        failures = sorted({str(record["failure"]) for record in records if record.get("failure")})
        if failures:
            parts.append("failures " + ", ".join(failures))
        else:
            parts.append("a finite finish under the force cap is wrong_stationary")
    return "; ".join(parts)


def _repeated_energy(records: list[dict]) -> str:
    """A repeated energy that is not the lowest, so a full row can still show it."""
    energies = [
        float(record["energy"])
        for record in records
        if record.get("energy") is not None and math.isfinite(float(record["energy"]))
    ]
    if len(energies) < 2:
        return ""
    labels = [f"{energy:.8f}" for energy in energies]
    printed = f"{min(energies):.8f}"
    modal, modal_n = Counter(labels).most_common(1)[0]
    if modal == printed or modal_n < 2:
        return ""
    return f"{modal_n} of {len(records)} starts at {modal}; lowest energy {printed}"


def _dat(
    summary: dict,
    template: dict,
    metrics: dict,
    commit: str,
    version: str,
    *,
    partial: bool,
    reason: str,
    finding: str = "",
) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "cap": template.get("failure_cap"),
        "par2_mean": metrics.get("par2_mean"),
        "rate_low": metrics.get("rate_low"),
        "rate_high": metrics.get("rate_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one lennard_jones evaluation through the OGOLEM socket.",
        "# A repeated geometry is one call. The call that would pass the cap is not counted.",
        "# The potential is epsilon 1, sigma 1, cutoff 40, no shift.",
        "# One reduced sigma is the Ar sigma 6.434516991 bohr. The socket divides by it.",
    ]
    if partial:
        rows["partial"] = "true"
        rows["partial_reason"] = reason
        lines.append("# partial. " + reason)
    if finding:
        rows["finding"] = finding
        lines.append("# finding. " + finding)
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _write_outputs(output: Path, records: list[dict], template: dict, problem: dict, cap: int, scored: bool) -> dict:
    ordered = sorted(records, key=lambda row: (row.get("index", 0), _sort_key(str(row.get("start")))))
    for record in ordered:
        record.pop("traces", None)
    legacy_counts = [int(record["force_calls"]) for record in ordered]
    summary = summarize_capped(legacy_counts, cap)
    under = [count for count in legacy_counts if count < cap]
    summary["median"] = statistics.median(under) if under else None
    energies = [
        float(record["energy"])
        for record in ordered
        if record.get("energy") is not None and math.isfinite(float(record["energy"]))
    ]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in ordered
        ],
        cap,
    )
    full = (problem.get("sample") or {}).get("starts")
    full_n = None if full in (None, "") else int(full)
    partial = (not scored) or (full_n is not None and len(ordered) < full_n)
    reason = _partial_reason(problem, ordered, scored, full_n) if partial else ""
    finding = "" if partial else _repeated_energy(ordered)
    public = []
    for record in ordered:
        row = dict(record)
        row.pop("index", None)
        public.append(row)
    version = str(template.get("code_version") or CODE_VERSION)
    commit = str(template.get("commit") or COMMIT)
    output.parent.mkdir(parents=True, exist_ok=True)
    dat_text = _dat(summary, template, metrics, commit, version, partial=partial, reason=reason, finding=finding)
    temporary = output.with_suffix(".dat.tmp")
    temporary.write_text(dat_text, encoding="utf-8")
    temporary.replace(output)
    starts_path = output.with_suffix(".starts.jsonl")
    starts_tmp = starts_path.with_suffix(".jsonl.tmp")
    write_starts(starts_tmp, public)
    starts_tmp.replace(starts_path)
    settings = _ogolem_settings(template)
    provenance = collect_provenance(
        code="ogolem",
        code_version=version,
        commit=commit,
        parameters={
            "pool_size": settings["pool"],
            "glob_iterations": settings["iterations"],
            "cell_bohr": settings["cell"],
            "threads": settings["threads"],
            "failure_cap": cap,
            "threshold": _threshold(problem, template),
            "ar_sigma_bohr": AR_SIGMA_BOHR,
            "ang_to_bohr": ANGTOBOHR,
        },
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": reason or str(template.get("note") or ""),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "partial": partial,
                "provenance": provenance,
                "metrics": metrics,
                "starts": public,
            }
        ],
    }
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(public)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["v2_successes"] = metrics["successes"]
    summary["partial"] = partial
    summary["failures"] = {name: metrics[f"n_{name}"] for name in ("crash", "divergence", "cap", "wrong_stationary")}
    return summary


def _completed(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("start") is not None and row.get("force_calls") is not None:
            found[str(row["start"])] = row
    return found


def printed_version(home: Path | None = None) -> str:
    """First line of ``MainOgolem --version``. The snapshot prints 2016.CURRENT."""
    root = home or ogolem_home()
    command = [
        str(java_bin()),
        "--enable-preview",
        "--add-modules=jdk.incubator.vector",
        "-cp",
        _classpath(root),
        "org.ogolem.general.MainOgolem",
        "--version",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(proc.stderr[-400:] or "OGOLEM printed no version")
    return lines[0]


def _java_command(home: Path, threads: int) -> list[str]:
    return [
        str(java_bin()),
        "--enable-preview",
        "--add-modules=jdk.incubator.vector",
        "-Xmx1024m",
        "-Dlogging.level.root=WARN",
        "-cp",
        _classpath(home),
        "org.ogolem.general.MainOgolem",
        "--shmem",
        "input.ogo",
        str(threads),
    ]


def execute_start(job: dict) -> dict:
    """Run one JVM. The record is the lowest harness energy and its call count."""
    import numpy as np

    started = time.monotonic()
    print(f"ogolem begin {job.get('problem')} {job.get('start')}", flush=True)
    cap = int(job["cap"])
    threshold = job.get("threshold")
    positions = center_cluster(job["coords"])
    atoms = int(np.asarray(positions).shape[0])
    work = Path(job["work"])
    if work.exists():
        shutil.rmtree(work)
    seeds = work / "seeds"
    seeds.mkdir(parents=True)
    copies = int(job.get("seed_copies") or 1)
    for copy in range(copies):
        write_xyz(seeds / f"seed-{copy}.xyz", positions)
    text = ogo_text(
        atoms=atoms,
        pool=int(job["pool"]),
        iterations=int(job["iterations"]),
        cell=float(job["cell"]),
        seed=int(job["seed"]),
        threshold=None if threshold is None else float(threshold),
    )
    (work / "input.ogo").write_text(text, encoding="utf-8")
    server = HarnessLJServer(cap=cap, threshold=None if threshold is None else float(threshold), trace=bool(job.get("trace")))
    log_path = work / "ogolem.log"
    exit_code = 1
    error = ""
    try:
        env = os.environ.copy()
        env["OPTBENCH_POTSERVER"] = server.endpoint
        command = _java_command(Path(job["home"]), int(job["threads"]))
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command,
                cwd=work,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                exit_code = int(proc.wait(timeout=float(job.get("wall") or 86400)))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
                exit_code = -9
                error = "wall timeout"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        server.close()
    if exit_code not in (0, 3) and not error and log_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-400:]
        error = tail.strip()
    calls = int(server.counter.calls)
    energy = server.best_energy
    success, failure, reasons = classify_start(
        exit_code=exit_code,
        calls=calls,
        energy=energy,
        cap=cap,
        threshold=None if threshold is None else float(threshold),
        hit_cap=server.hit_cap,
    )
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "found": bool(server.found),
        "exit_code": exit_code,
        "seconds": time.monotonic() - started,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
        "template": job["template"],
        "problem": job["problem"],
    }
    if error:
        record["error"] = error
    if job.get("trace"):
        record["traces"] = server.traces
    return record


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    commit = str(template.get("commit") or "")
    if commit != COMMIT:
        raise RuntimeError(f"{template['id']} commit {commit} is not the built pin {COMMIT}")
    version = str(template.get("code_version") or "")
    if version != CODE_VERSION:
        raise RuntimeError(f"{template['id']} version {version} is not {CODE_VERSION}")
    home = ogolem_home()
    if not jars_ready(home):
        raise RuntimeError(f"OGOLEM jars for {COMMIT} are not installed under {home}")
    if not java_bin().is_file():
        raise RuntimeError(f"java is missing at {java_bin()}")
    _ensure_exploded(home)
    printed = printed_version(home)
    if printed != PRINTED_VERSION:
        raise RuntimeError(f"OGOLEM printed {printed}, expected {PRINTED_VERSION}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    cap = _force_cap(problem, template)
    settings = _ogolem_settings(template)
    if settings["pool"] < 2:
        raise ValueError(f"{template['id']} pool_size must be at least 2")
    threshold = _threshold(problem, template)
    if problem["id"] == "globalopt/lj38":
        if threshold is None or abs(threshold - LJ38_STOP_ENERGY) > 1e-9:
            raise RuntimeError(
                f"lj38 threshold {threshold} is not the archived stop energy {LJ38_STOP_ENERGY}"
            )
    scratch = work or Path(os.environ.get("OPTBENCH_OGOLEM_WORK", "/tmp/optbench-ogolem")) / template["id"].replace("/", "-")
    if geometries is not None:
        raise ValueError("ogolem reads the problem archive; pass no geometries directory")
    loaded = load_starts(root, problem, template, scratch)
    if limit:
        loaded = loaded[: int(limit)]
    done = _completed(output.with_suffix(".starts.jsonl")) if resume else {}
    jobs = []
    kept = []
    for index, (name, coords) in enumerate(loaded):
        if name in done:
            row = dict(done[name])
            row["index"] = index
            kept.append(row)
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.reshape(-1, 3).tolist(),
                "cap": cap,
                "threshold": threshold,
                "pool": settings["pool"],
                "iterations": settings["iterations"],
                "cell": settings["cell"],
                "threads": settings["threads"],
                "seed": index + 1,
                "home": str(home),
                "work": str(scratch / "starts" / f"{index:04d}-{name}"),
                "wall": 86400,
                "trace": False,
                "code": template.get("code"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "template": template["id"],
                "problem": problem["id"],
            }
        )
    scored = threshold is not None
    records = list(kept)

    def _store(record: dict) -> None:
        records.append(record)
        _write_outputs(output, records, template, problem, cap, scored)
        print(
            f"{template['id']} {len(records)}/{len(loaded)} start={record['start']} "
            f"calls={record['force_calls']} energy={record.get('energy')} "
            f"failure={record.get('failure')} seconds={record.get('seconds', 0):.1f}",
            flush=True,
        )

    if jobs and workers <= 1:
        for job in jobs:
            _store(execute_start(job))
    elif jobs:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(execute_start, job) for job in jobs]
            for future in as_completed(futures):
                _store(future.result())
    elif records:
        _write_outputs(output, records, template, problem, cap, scored)
    else:
        raise RuntimeError(f"{template['id']} has no starts")
    summary = _write_outputs(output, records, template, problem, cap, scored)
    return summary


def smoke_dimer() -> dict:
    """Two atoms at the Lennard-Jones minimum, evaluated by a real JVM.

    The first socket call is the seeded geometry in reduced units. Its
    distance and energy check the argon-sigma conversion. Local
    optimization of a wrong scale can still fall into the well, so the
    first call is the check.
    """
    import numpy as np

    home = ogolem_home()
    if not jars_ready(home):
        raise RuntimeError(f"OGOLEM jars for {COMMIT} are not installed under {home}")
    distance = 2.0 ** (1.0 / 6.0)
    coords = [[-0.5 * distance, 0.0, 0.0], [0.5 * distance, 0.0, 0.0]]
    work = Path("/tmp/optbench-ogolem-dimer")
    job = {
        "index": 0,
        "start": "dimer",
        "coords": coords,
        "cap": 1000,
        "threshold": None,
        "pool": 2,
        "iterations": 0,
        "cell": 60.0,
        "threads": 1,
        "seed": 1,
        "seed_copies": 2,
        "home": str(home),
        "work": str(work),
        "wall": 180,
        "trace": True,
        "code": "ogolem",
        "code_version": CODE_VERSION,
        "commit": COMMIT,
        "tuning": "default",
        "template": "ogolem/dimer",
        "problem": "smoke",
    }
    record = execute_start(job)
    traces = record.get("traces") or []
    if not traces:
        raise RuntimeError(f"dimer made no harness call: {record.get('error')}")
    first = traces[0]["coords"]
    pos = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    span = float(np.linalg.norm(pos[0] - pos[1]))
    record["first_distance"] = span
    record["first_energy"] = float(traces[0]["energy"])
    if abs(span - distance) > 1e-6:
        raise RuntimeError(f"dimer distance {span} is not {distance}")
    if abs(record["first_energy"] + 1.0) > 1e-6:
        raise RuntimeError(f"dimer energy {record['first_energy']} is not -1")
    return record
