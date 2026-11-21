"""Pt-island catalog on rgsaddle develop, plus the eOn classic comparator.

Seed 20261016. Each row stops at 27 connected saddles or 20000 searches.
One process uses one core and one thread. Checkpoints are catalog.json
and searches.jsonl under build/rgsaddle/<method>/ so a row can resume.

The csv command writes results/campaign/rgsaddle-develop-aa6dd937-pt-island.csv
from the search ledgers, the campaign files, and
results/campaign/rgsaddle-develop-aa6dd937-pt-island-hardware.json.
Nothing in that file is typed by hand. Method rows keep the force-call
budget in ``budget`` and the cap failure count in ``cap_failures``.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHA = "aa6dd9371d6a63103010a6829cb62c74a39c7ff1"
SEED = 20261016
GOAL = 27
MAX_SEARCHES = 20000
BARRIER_EV = 1.5
CSV_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island.csv"
HARDWARE_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island-hardware.json"
TIMING_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island-timing.json"
COMMANDS_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island-commands.json"
PROVENANCE_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-eon-provenance.json"

RGSADDLE_ROWS = {
    "dimer-fire": "rgsaddle/develop-saddle-dimer-fire",
    "dimer-lbfgs": "rgsaddle/develop-saddle-dimer-lbfgs",
    "dimer-rotskip": "rgsaddle/develop-saddle-dimer-rotskip",
    "lanczos": "rgsaddle/develop-saddle-lanczos",
}
THREAD_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_UNEXPANDED_VAR = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}")
_GCC_LINE = re.compile(
    r"^(?P<bin>gcc|g\+\+|cc|c\+\+) \((?:GCC|(?:[A-Za-z][\w.+-]* )*\d+\.\d+[^)]*)\) (?P<ver>[1-9]\d*\.\d+(?:\.\d+)*)$"
)
_CLANG_LINE = re.compile(
    r"^(?:(?P<distro>Ubuntu|Debian|Amazon|Homebrew) )?(?P<bin>Apple clang|clang\+\+|clang) version "
    r"(?P<ver>[1-9]\d*\.\d+(?:\.\d+)*)(?: \([^)]+\))?$"
)
_FC_LINE = re.compile(
    r"^(?P<bin>GNU Fortran|gfortran) \((?:GCC|(?:[A-Za-z][\w.+-]* )*\d+\.\d+[^)]*)\) (?P<ver>[1-9]\d*\.\d+(?:\.\d+)*)$"
)
_BUILDTYPES = frozenset({"plain", "debug", "debugoptimized", "release", "minsize"})
_ALLOWED_KEYS = {
    "": frozenset({"argv_origin", "binary", "build", "run", "source"}),
    "binary": frozenset({"name", "sha256", "version_stdout"}),
    "build": frozenset(
        {
            "buildtype",
            "commands",
            "compilers",
            "meson",
            "ninja",
            "optimization",
            "options",
            "prefix",
            "toolchain",
        }
    ),
    "build.compilers": frozenset({"cc", "cxx", "fc"}),
    "source": frozenset({"commit", "tag"}),
    "run": frozenset({"communicator_jobs", "core", "threads"}),
}


class ProvenanceError(ValueError):
    """The eOn provenance document cannot be recorded."""


def _schema_where(where: str) -> str:
    """The schema path of a value, dropping list indexes."""
    return re.sub(r"\[\d+\]", "", where)


def _allowed_keys(where: str) -> frozenset[str] | None:
    """Keys a provenance object may hold, or None when the object is a thread map."""
    place = _schema_where(where)
    if place == "run.threads":
        return None
    return _ALLOWED_KEYS.get(place)


def _reject_unexpanded(value: str, where: str) -> None:
    if _UNEXPANDED_VAR.search(value):
        raise ProvenanceError(f"{where} contains an unexpanded variable")


def _walk_provenance(value, where: str) -> None:
    if isinstance(value, dict):
        allowed = _allowed_keys(where)
        for key, item in value.items():
            name = str(key)
            place = f"{where}.{name}" if where else name
            if allowed is None:
                if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None:
                    raise ProvenanceError(f"{place} is not an allowed key")
            elif name not in allowed:
                raise ProvenanceError(f"{place} is not an allowed key")
            if "sha256" in name.lower() and item is None:
                raise ProvenanceError(f"{place} is null")
            _walk_provenance(item, place)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_provenance(item, f"{where}[{index}]")
        return
    if isinstance(value, str):
        _reject_unexpanded(value, where)
        return
    if value is None and "sha256" in where.lower():
        raise ProvenanceError(f"{where} is null")


def _compiler_identity(line: str) -> dict[str, str]:
    """Parse one compiler banner into family, argv name, and version."""
    match = _GCC_LINE.fullmatch(line)
    if match is not None:
        binary = match.group("bin")
        family = "gnu-cxx" if binary in {"g++", "c++"} else "gnu-c"
        identity = {"family": family, "binary": binary, "version": match.group("ver")}
    else:
        match = _CLANG_LINE.fullmatch(line)
        if match is not None:
            token = match.group("bin")
            distro = bool(match.group("distro") or token == "Apple clang")
            binary = "clang++" if token == "clang++" else "clang"
            identity = {
                "family": "clang",
                "binary": binary,
                "version": match.group("ver"),
                "distro": "1" if distro else "",
            }
        else:
            match = _FC_LINE.fullmatch(line)
            if match is None:
                raise ProvenanceError(f"{line!r} is not a full compiler version string")
            identity = {"family": "fortran", "binary": "gfortran", "version": match.group("ver")}
    major = int(identity["version"].split(".", 1)[0])
    floor = 3 if identity["family"] == "clang" else 4
    if major < floor:
        raise ProvenanceError(f"{line!r} is not a full compiler version string")
    return identity


def compiler_version(stdout: str) -> str:
    """The first line of a compiler ``--version`` report.

    A bare compiler name, or a string that only contains a number, is not a version.
    """
    line = next((part.strip() for part in stdout.splitlines() if part.strip()), "")
    _compiler_identity(line)
    return line


def _compiler_argv(role: str, identity: dict[str, str]) -> str:
    if role == "fc":
        return "gfortran"
    if role == "cxx" and identity["family"] == "clang":
        return "clang++"
    return identity["binary"]


def _require_compiler_role(role: str, identity: dict[str, str]) -> None:
    family = identity["family"]
    if role == "cc" and family in {"gnu-c", "clang"}:
        return
    if role == "cxx" and family == "gnu-cxx":
        return
    if role == "cxx" and family == "clang" and (identity["binary"] == "clang++" or identity.get("distro")):
        return
    if role == "fc" and family == "fortran":
        return
    raise ProvenanceError(f"build.compilers.{role} is not a full version string")


def _meson_head(command: str) -> str:
    """The setup words before a comment or a shell operator."""
    cut = len(command)
    for marker in ("&&", "||", ";", "#"):
        index = command.find(marker)
        if index != -1:
            cut = min(cut, index)
    cursor = 0
    while True:
        index = command.find("|", cursor)
        if index == -1 or index >= cut:
            break
        if command[index : index + 2] == "||":
            cursor = index + 2
            continue
        cut = index
        break
    return command[:cut].strip()


def _parse_meson_setup(command: str) -> dict:
    """Read CC, CXX, FC, the build directory, and the meson options."""
    if not isinstance(command, str) or not command.strip():
        raise ProvenanceError("build.commands[0] is not the meson setup command")
    head = _meson_head(command)
    try:
        tokens = shlex.split(head)
    except ValueError as exc:
        raise ProvenanceError("build.commands[0] is not the meson setup command") from exc
    env: dict[str, str] = {}
    index = 0
    while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
        name, _, value = tokens[index].partition("=")
        if name in {"CC", "CXX", "FC"}:
            env[name] = value
        index += 1
    if tokens[index : index + 2] != ["meson", "setup"]:
        raise ProvenanceError("build.commands[0] is not the meson setup command")
    index += 2
    if index >= len(tokens) or tokens[index].startswith("-"):
        raise ProvenanceError("meson setup command omits the build directory")
    index += 1
    flags: dict[str, str] = {}
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-D") and "=" in token[2:]:
            body = token[2:]
        elif token.startswith("--") and "=" in token:
            body = token[2:]
        else:
            raise ProvenanceError("build.commands[0] is not the meson setup command")
        name, _, value = body.partition("=")
        flags[name] = value
        index += 1
    for name in ("CC", "CXX", "FC"):
        if not env.get(name, "").strip():
            raise ProvenanceError(f"meson setup command has an empty {name}")
    if flags.get("optimization") != "3":
        raise ProvenanceError("meson setup command omits -Doptimization=3")
    for name in ("prefix", "libdir", "buildtype"):
        value = flags.get(name, "")
        if (
            not value
            or _UNEXPANDED_VAR.search(value)
            or "<" in value
            or ">" in value
            or "~" in value
        ):
            raise ProvenanceError(f"meson setup command omits {name}")
    if flags["buildtype"] not in _BUILDTYPES:
        raise ProvenanceError("meson setup command has an invalid buildtype")
    return {"head": head, "env": env, "flags": flags}


def meson_setup_command(*, cc: str, cxx: str, fc: str, builddir: str, options: list[str]) -> str:
    """The meson setup line, with every required option spelled on the command."""
    for label, value in (("cc", cc), ("cxx", cxx), ("fc", fc), ("builddir", builddir)):
        _reject_unexpanded(value, label)
        if not value.strip():
            raise ProvenanceError(f"meson setup command has an empty {label.upper() if label != 'builddir' else 'build directory'}")
    rendered: list[str] = []
    for option in options:
        _reject_unexpanded(option, "option")
        if not option.startswith("-D") or "=" not in option[2:]:
            raise ProvenanceError(f"meson option {option!r} is not -Dname=value")
        rendered.append(option)
    flags = " ".join(rendered)
    command = f"CC={cc} CXX={cxx} FC={fc} meson setup {builddir} {flags}".rstrip()
    _parse_meson_setup(command)
    return command


def validate_provenance(document: dict) -> None:
    """Refuse a null sha256, a key outside the schema, or an unexpanded variable.

    ``cc``, ``cxx``, and ``fc`` are full version strings for that role. The
    first build command is a meson setup line whose compilers, build directory,
    optimization, prefix, libdir, and buildtype are real values.
    """
    if not isinstance(document, dict):
        raise ProvenanceError("provenance is not an object")
    _walk_provenance(document, "")
    binary = document.get("binary")
    if not isinstance(binary, dict):
        raise ProvenanceError("binary is not an object")
    sha = binary.get("sha256")
    if sha is None:
        raise ProvenanceError("binary.sha256 is null")
    if not isinstance(sha, str) or _SHA256_TEXT.fullmatch(sha) is None:
        raise ProvenanceError("binary.sha256 is invalid")
    build = document.get("build")
    if not isinstance(build, dict):
        raise ProvenanceError("build is not an object")
    compilers = build.get("compilers")
    if not isinstance(compilers, dict):
        raise ProvenanceError("build.compilers is not an object")
    identities: dict[str, dict[str, str]] = {}
    for name in ("cc", "cxx", "fc"):
        text = compilers.get(name)
        if not isinstance(text, str):
            raise ProvenanceError(f"build.compilers.{name} is not a full version string")
        try:
            identity = _compiler_identity(text)
        except ProvenanceError as exc:
            raise ProvenanceError(f"build.compilers.{name} is not a full version string") from exc
        _require_compiler_role(name, identity)
        identities[name] = identity
    options = build.get("options")
    commands = build.get("commands")
    if not isinstance(options, list) or not isinstance(commands, list) or not commands:
        raise ProvenanceError("build.commands has no meson setup command")
    parsed = _parse_meson_setup(commands[0])
    for name in ("cc", "cxx", "fc"):
        expected = _compiler_argv(name, identities[name])
        if parsed["env"][name.upper()] != expected:
            raise ProvenanceError(f"meson setup command {name.upper()} does not match build.compilers.{name}")
    for option in options:
        if not isinstance(option, str) or option not in parsed["head"]:
            raise ProvenanceError(f"meson setup command omits {option}")


def provenance_document(
    *,
    binary_sha256: str,
    version_stdout: str,
    cc_stdout: str,
    cxx_stdout: str,
    fc_stdout: str,
    cc: str,
    cxx: str,
    fc: str,
    builddir: str,
    options: list[str],
    meson: str,
    ninja: str,
    source: dict,
    run: dict,
    argv_origin: str,
) -> dict:
    """Build a provenance document and refuse one that cannot be recorded."""
    command = meson_setup_command(cc=cc, cxx=cxx, fc=fc, builddir=builddir, options=options)
    prefix = ""
    optimization = ""
    buildtype = ""
    for option in options:
        name, _, value = option[2:].partition("=")
        if name == "prefix":
            prefix = value
        elif name == "optimization":
            optimization = value
        elif name == "buildtype":
            buildtype = value
    document = {
        "argv_origin": argv_origin,
        "binary": {
            "name": "eonclient",
            "sha256": binary_sha256,
            "version_stdout": version_stdout,
        },
        "build": {
            "buildtype": buildtype,
            "commands": [command, f"meson install -C {builddir}"],
            "compilers": {
                "cc": compiler_version(cc_stdout),
                "cxx": compiler_version(cxx_stdout),
                "fc": compiler_version(fc_stdout),
            },
            "meson": meson,
            "ninja": ninja,
            "optimization": optimization,
            "options": list(options),
            "prefix": prefix,
            "toolchain": f"meson {meson} + ninja {ninja}",
        },
        "run": run,
        "source": source,
    }
    validate_provenance(document)
    return document


def write_provenance(path: Path, document: dict) -> None:
    """Write ``document`` after it passes :func:`validate_provenance`."""
    validate_provenance(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provenance_publishes(path: Path) -> bool:
    """True when ``path`` is a provenance document this writer accepts."""
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        validate_provenance(document)
    except ProvenanceError:
        return False
    return True


def _published_eon_dat() -> Path | None:
    """The eon-classic ledger, when its provenance passes :func:`validate_provenance`."""
    candidates = (
        (_output_path("eon-classic"), PROVENANCE_PATH),
    )
    for dat, provenance in candidates:
        if dat.is_file() and provenance_publishes(provenance):
            return dat
    return None


def _eon_record_ready() -> bool:
    return _published_eon_dat() is not None


def _method_dat(method: str) -> Path:
    if method == "eon-classic":
        published = _published_eon_dat()
        if published is not None:
            return published
    return _output_path(method)


def _provenance_kwargs(client: Path, facts: dict) -> dict:
    return {
        "binary_sha256": _sha256(client),
        "version_stdout": str(facts["version_stdout"]),
        "cc_stdout": str(facts["cc_stdout"]),
        "cxx_stdout": str(facts["cxx_stdout"]),
        "fc_stdout": str(facts["fc_stdout"]),
        "cc": str(facts["cc"]),
        "cxx": str(facts["cxx"]),
        "fc": str(facts["fc"]),
        "builddir": str(facts["builddir"]),
        "options": list(facts["options"]),
        "meson": str(facts["meson"]),
        "ninja": str(facts["ninja"]),
        "source": dict(facts["source"]),
        "run": dict(facts["run"]),
        "argv_origin": str(facts["argv_origin"]),
    }


def _load_provenance() -> dict:
    if not PROVENANCE_PATH.is_file():
        return {}
    document = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    validate_provenance(document)
    return document


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _merge_wall_comparable(stored: dict, extra: dict | None) -> None:
    if not extra:
        return
    merged = dict(extra)
    if stored.get("wall_comparable") is False:
        merged["wall_comparable"] = False
    stored.update(merged)


def _sanitize_token(token: str) -> str:
    candidate = Path(token)
    if candidate.is_absolute():
        if candidate.exists() and candidate.is_file():
            return candidate.name
        try:
            return str(candidate.relative_to(ROOT))
        except ValueError:
            return candidate.name
    return token


def _sanitize_argv(argv: list[str]) -> list[str]:
    return [_sanitize_token(item) for item in argv]


def _sanitize_command_payload(payload: dict) -> dict:
    cleaned = dict(payload)
    argv = cleaned.get("argv")
    if isinstance(argv, list):
        cleaned["argv"] = _sanitize_argv([str(token) for token in argv])
    return cleaned


def _timing_paths(method: str) -> tuple[Path, Path]:
    base = _checkpoint_dir(method)
    return base / "timing.json", base / "timing.segments.jsonl"


def _timing_start(method: str, *, extra: dict | None = None) -> float:
    now = time.time()
    timing_path, segments_path = _timing_paths(method)
    stored = _read_json(timing_path)
    if "started_unix" not in stored:
        stored["started_unix"] = now
    if "host" not in stored:
        stored["host"] = socket.gethostname()
    _merge_wall_comparable(stored, extra)
    timing_path.write_text(json.dumps(stored, sort_keys=True) + "\n", encoding="utf-8")
    segment = {
        "segment_started_unix": now,
        "argv": _sanitize_argv(sys.argv),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    with segments_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(segment, sort_keys=True) + "\n")
    return now


def _timing_finish(method: str, started: float, *, extra: dict | None = None) -> None:
    finished = time.time()
    duration = max(0.0, finished - started)
    timing_path, segments_path = _timing_paths(method)
    stored = _read_json(timing_path)
    stored["finished_unix"] = finished
    previous = float(stored.get("wall_seconds_total") or 0.0)
    stored["wall_seconds_total"] = previous + duration
    _merge_wall_comparable(stored, extra)
    timing_path.write_text(json.dumps(stored, sort_keys=True) + "\n", encoding="utf-8")
    with segments_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "segment_finished_unix": finished,
                    "segment_wall_seconds": duration,
                    "argv": _sanitize_argv(sys.argv),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _single_thread(cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    for name in THREAD_VARS:
        os.environ[name] = "1"


def _checkpoint_dir(method: str) -> Path:
    path = ROOT / "build" / "rgsaddle" / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_path(method: str) -> Path:
    if method == "eon-classic":
        return ROOT / "results" / "campaign" / "eon-3.5.0-d9cc6c5be-saddle-pt-classic-aa6dd937.dat"
    return ROOT / "results" / "campaign" / f"rgsaddle-develop-aa6dd937-saddle-{method}.dat"


def classify_search(record: dict, step_ceiling: int | None) -> str:
    """One class per search.

    Divergence is only a non-finite point or a step past the ceiling.
    When more than one reason is present the class is the first of
    duplicate, over_barrier, disconnected, cap, and hessian_index.
    eOn ``Saddle Search Terminated High Energy`` is ``high_energy``.
    Anything left is wrong_stationary.
    """
    if record.get("success"):
        return "connected"
    reasons = {str(item) for item in (record.get("reasons") or [])}
    step_max = int(record.get("step_calls_max") or 0)
    barrier = record.get("barrier")
    non_finite = "divergence" in reasons or "timeout" in reasons
    if barrier is not None:
        try:
            non_finite = non_finite or not (float(barrier) == float(barrier))
        except (TypeError, ValueError):
            non_finite = True
    if non_finite or (step_ceiling is not None and step_max > step_ceiling):
        return "divergence"
    if "duplicate" in reasons:
        return "duplicate"
    if "barrier" in reasons or "over_barrier" in reasons:
        return "over_barrier"
    if barrier is not None:
        try:
            if float(barrier) >= BARRIER_EV and "not_connected" not in reasons:
                return "over_barrier"
        except (TypeError, ValueError):
            return "divergence"
    if "not_connected" in reasons:
        return "disconnected"
    if "cap" in reasons:
        return "cap"
    if "hessian_index" in reasons:
        return "hessian_index"
    if str(record.get("result") or "") == "Saddle Search Terminated High Energy":
        return "high_energy"
    return "wrong_stationary"


def _overlap(record: dict) -> dict[str, int]:
    reasons = {str(item) for item in (record.get("reasons") or [])}
    barrier = record.get("barrier")
    high = False
    if barrier is not None:
        try:
            high = float(barrier) >= BARRIER_EV
        except (TypeError, ValueError):
            high = False
    return {
        "disconnected": int("not_connected" in reasons),
        "over_barrier": int("barrier" in reasons or "over_barrier" in reasons or high),
        "duplicate": int("duplicate" in reasons),
    }


def tally(records: list[dict], step_ceiling: int | None) -> dict[str, int]:
    counts = {
        "searches": len(records),
        "connected": 0,
        "disconnected": 0,
        "over_barrier": 0,
        "duplicate": 0,
        "wrong_stationary": 0,
        "divergence": 0,
        "cap": 0,
        "hessian_index": 0,
        "high_energy": 0,
        "overlap_disconnected": 0,
        "overlap_over_barrier": 0,
        "overlap_duplicate": 0,
    }
    for record in records:
        counts[classify_search(record, step_ceiling)] += 1
        if record.get("success"):
            continue
        for key, value in _overlap(record).items():
            counts[f"overlap_{key}"] += value
    return counts


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def step_ceiling(method: str) -> int | None:
    if method not in RGSADDLE_ROWS:
        return None
    from optbench.campaign import load_template
    from optbench.rgsaddle_minmode import _step_ceiling

    template = load_template(ROOT, RGSADDLE_ROWS[method])
    return int(_step_ceiling(template["ini"]["SaddleSearch"]))


def _ledger(method: str) -> list[dict]:
    finished = _method_dat(method).with_suffix(".starts.jsonl")
    if finished.is_file() and finished.stat().st_size:
        return _load_jsonl(finished)
    return _load_jsonl(_checkpoint_dir(method) / "searches.jsonl")


def progress_text() -> str:
    lines = [
        f"rgsaddle develop {SHA}",
        "includes fix/lanczos-pt-on-develop and the PAFI series",
        f"seed {SEED}",
        f"stop {GOAL} connected or {MAX_SEARCHES} searches",
        "exclusive class order: connected, divergence, duplicate, over_barrier, disconnected, cap, hessian_index, high_energy, wrong_stationary",
        "divergence: non-finite point or a step past the ceiling",
        f"over_barrier: barrier at least {BARRIER_EV} eV",
    ]
    methods = list(RGSADDLE_ROWS)
    if _eon_record_ready():
        methods.append("eon-classic")
    for method in methods:
        counts = tally(_ledger(method), step_ceiling(method))
        lines.append(
            "method {method} searches {searches} connected {connected} "
            "disconnected {disconnected} over_barrier {over_barrier} "
            "duplicate {duplicate} cap {cap} hessian_index {hessian_index} "
            "high_energy {high_energy} wrong_stationary {wrong_stationary} "
            "divergence {divergence} "
            "overlap_disconnected {overlap_disconnected} "
            "overlap_over_barrier {overlap_over_barrier} "
            "overlap_duplicate {overlap_duplicate}".format(method=method, **counts)
        )
    return "\n".join(lines) + "\n"


def _write_command(method: str, cpu: int) -> None:
    head = ""
    src = os.environ.get("RGSADDLE_SRC")
    if src:
        import subprocess

        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    payload = {
        "method": method,
        "argv": _sanitize_argv(sys.argv),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "requested_cpu": cpu,
        "threads": {name: os.environ.get(name, "") for name in THREAD_VARS},
        "rgsaddle_head": head,
    }
    lib = Path(os.environ.get("RGSADDLE_LIB", ""))
    if not lib.is_file():
        raise ProvenanceError("RGSADDLE_LIB is not a library file")
    payload["rgsaddle_lib_sha256"] = _sha256(lib)
    path = _checkpoint_dir(method) / "command.json"
    existing = _read_json(path)
    merged = dict(existing)
    merged.update(payload)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_rgsaddle(method: str, cpu: int) -> None:
    _single_thread(cpu)
    os.environ["OPTBENCH_RGSADDLE_CHECKPOINT_DIR"] = str(_checkpoint_dir(method))
    _write_command(method, cpu)
    from optbench.campaign import load_template, resolve_problem
    from optbench.rgsaddle_minmode import run_template

    template = load_template(ROOT, RGSADDLE_ROWS[method])
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    output = _output_path(method)
    started = _timing_start(method, extra={"method": method, "wall_comparable": False})
    run_template(ROOT, problem, template, output=output, workers=1, repetition=1)
    _timing_finish(method, started, extra={"method": method, "wall_comparable": False})


def _eon_records(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        if len(rows) >= MAX_SEARCHES:
            break
        try:
            barrier = float(fields[2])
            sad = int(fields[4])
            mins = int(fields[5])
        except ValueError:
            continue
        token = " ".join(fields[7:])
        reasons: list[str] = []
        success = False
        finite = barrier == barrier
        if not finite:
            reasons = ["divergence"]
        elif token.startswith("repeat"):
            reasons = ["duplicate"]
            if barrier >= BARRIER_EV:
                reasons.append("barrier")
        elif token.startswith("good") and barrier < BARRIER_EV:
            success = True
        elif token.startswith("good"):
            reasons = ["barrier"]
        elif "Not Connected" in token:
            reasons = ["not_connected"]
            if barrier >= BARRIER_EV:
                reasons.append("barrier")
        else:
            reasons = ["force_norm"]
        rows.append(
            {
                "success": success,
                "reasons": reasons,
                "barrier": barrier,
                "result": token,
                "force_calls": sad + mins,
            }
        )
        if sum(1 for row in rows if row["success"]) >= GOAL:
            break
    return rows


def _client_version(client: Path) -> str:
    import subprocess

    out = subprocess.check_output([str(client), "--version"], text=True, stderr=subprocess.STDOUT)
    return " ".join(out.split())


def _write_eon_ledger(folder: Path, records: list[dict]) -> None:
    ledger = folder / "searches.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def run_eon(cpu: int, client: Path, provenance_facts: dict | None = None) -> None:
    """Classic dimer, one core, seed 20261016, stop at 27 or 20000 searches.

    ``client`` is the source build of tag ``v3.5.0`` (``d9cc6c5be``).
    ``provenance_document`` and ``write_provenance`` record the Meson command,
    the compiler version strings, and the eonclient sha256 before the search.
    """
    import inspect
    import logging
    import traceback

    _single_thread(cpu)
    if not provenance_facts:
        raise ProvenanceError("eon-classic provenance is required")
    document = provenance_document(**_provenance_kwargs(client, provenance_facts))
    write_provenance(PROVENANCE_PATH, document)
    folder = _checkpoint_dir("eon-classic")
    _write_command("eon-classic", cpu)
    version = _client_version(client)
    from optbench.campaign import load_template
    from optbench.eon_akmc import _write_config, prepare_client_env
    from optbench.eon_parse import saddle_progress

    template = copy.deepcopy(load_template(ROOT, "eon/develop-saddle-pt-dimer"))
    template["ini"]["Communicator"]["num_jobs"] = 1
    template["ini"]["Communicator"]["number_of_CPUs"] = 1
    template["ini"]["Main"]["random_seed"] = SEED
    work = folder / "work"
    fresh = not (work / "states").is_dir()
    if fresh:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        pos = ROOT / "website" / "saddle-search" / "pt-island" / "pt-island.con"
        if not pos.is_file():
            from optbench.campaign import geometry_archive

            pos = geometry_archive(
                ROOT,
                {"id": "saddle", "input": {"archive_path": "website/saddle-search/pt-island/pt-island.con"}},
            )
        shutil.copy(pos, work / "pos.con")
        _write_config(template, work / "config.ini", client, SEED, 1)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    prepare_client_env(client)
    for name in ("akmc", "eon", "state", "communicator"):
        logging.getLogger(name).setLevel(logging.WARNING)
    started = _timing_start(
        "eon-classic",
        extra={"method": "eon-classic", "client_version": version, "wall_comparable": False},
    )
    previous = Path.cwd()
    last_error = ""
    records: list[dict] = []
    try:
        os.chdir(work)
        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

        parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
        if "config" in parameters:
            config = ConfigClass()
        else:
            config = shared_config
            config.init_done = False
        config.init()
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        for index in range(MAX_SEARCHES):
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break
            results = work / "states" / "0" / "search_results.txt"
            if not results.is_file():
                continue
            text = results.read_text(encoding="utf-8", errors="replace")
            records = _eon_records(text)
            _write_eon_ledger(folder, records)
            progress = saddle_progress(text)
            connected = sum(1 for record in records if record["success"])
            if (index + 1) % 25 == 0 or connected >= GOAL or len(records) >= MAX_SEARCHES:
                print(
                    f"eon-classic searches {len(records)} connected {connected} "
                    f"qualifying {progress['qualifying']}",
                    flush=True,
                )
            if connected >= GOAL or len(records) >= MAX_SEARCHES or progress["reached"]:
                break
    finally:
        os.chdir(previous)
    results = work / "states" / "0" / "search_results.txt"
    if results.is_file():
        records = _eon_records(results.read_text(encoding="utf-8", errors="replace"))
        _write_eon_ledger(folder, records)
    _timing_finish(
        "eon-classic",
        started,
        extra={
            "method": "eon-classic",
            "client_version": version,
            "error": last_error,
            "ledger_searches": len(records),
            "ledger_connected": sum(1 for record in records if record["success"]),
            "wall_comparable": False,
        },
    )
    timing_totals = _read_json(_checkpoint_dir("eon-classic") / "timing.json")
    wall_total = float(timing_totals.get("wall_seconds_total") or 0.0)
    wall_comparable = bool(timing_totals.get("wall_comparable"))
    _write_eon_dat(records, wall_total, version, last_error, wall_comparable=wall_comparable)


def _write_eon_dat(records: list[dict], wall: float, version: str, error: str, *, wall_comparable: bool) -> None:
    output = _output_path("eon-classic")
    connected = sum(1 for record in records if record["success"])
    calls = [int(record["force_calls"]) for record in records]
    stopped = connected >= GOAL or len(records) >= MAX_SEARCHES
    lines = [
        f"# rep 1 force_calls {sum(calls)} total_searches {len(records)} connected {connected} seed {SEED}",
        "code eOn",
        f"code_version {version}",
        "algorithm Dimer",
        "dimer_improved false",
        "template eon/develop-saddle-pt-dimer",
        "problem saddle-search/pt-island",
        "counter client-search-results",
        "count_keys sad-fcs+mins-fcs",
        f"seed {SEED}",
        "communicator_jobs 1",
        "repetitions 1",
        f"total_searches {len(records)}",
        f"connected {connected}",
        f"goal {GOAL}",
        f"nfailed {0 if stopped else 1}",
    ]
    if wall_comparable:
        lines.append(f"wall_seconds {wall:.6f}")
    if error:
        lines.append(f"error {error.splitlines()[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    starts = output.with_suffix(".starts.jsonl")
    with starts.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _hardware_rows() -> list[dict[str, str]]:
    recorded = json.loads(HARDWARE_PATH.read_text(encoding="utf-8"))
    keys = (
        "model_name",
        "cpu_cores",
        "cpus_online",
        "mem_total_kb",
        "affinity_policy",
        "thread_vars",
    )
    return [{"row_kind": "hardware", "key": key, "value": str(recorded[key])} for key in keys]


def _command_rows() -> list[dict[str, str]]:
    stored = json.loads(COMMANDS_PATH.read_text(encoding="utf-8")) if COMMANDS_PATH.is_file() else {}
    provenance = _load_provenance()
    rows = []
    methods = list(RGSADDLE_ROWS)
    if _eon_record_ready():
        methods.append("eon-classic")
    for method in methods:
        payload = stored.get(method)
        if payload is None:
            path = _checkpoint_dir(method) / "command.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload = _sanitize_command_payload(payload)
        row = {
            "row_kind": "command",
            "method": method,
            "argv": " ".join(payload.get("argv") or []) if isinstance(payload.get("argv"), list) else str(payload.get("argv") or ""),
            "cpu_affinity": ",".join(str(cpu) for cpu in payload.get("cpu_affinity") or []),
            "threads": " ".join(f"{key}={value}" for key, value in sorted((payload.get("threads") or {}).items())),
            "rgsaddle_head": payload.get("rgsaddle_head", ""),
            "rgsaddle_lib_sha256": payload.get("rgsaddle_lib_sha256", ""),
        }
        if method == "eon-classic":
            if not provenance:
                raise ProvenanceError("eon-classic provenance is required")
            row["argv_origin"] = str(provenance["argv_origin"])
        rows.append(row)
    return rows


def _protocol_budget(method: str, fields: dict[str, str]) -> str:
    """Force-call budget on the dat.

    The eOn classic dat has no cap line. Its template uses the same
    saddle and optimizer iteration limits as the dimer rows, so the
    bound is climb steps times the step ceiling plus both endpoint
    minimizations.
    """
    recorded = fields.get("cap", "")
    if recorded or method != "eon-classic":
        return recorded
    from optbench.campaign import load_template
    from optbench.rgsaddle_minmode import _step_ceiling

    template = load_template(ROOT, fields.get("template") or "eon/develop-saddle-pt-dimer")
    ini = template["ini"]
    saddle = dict(ini["Saddle Search"])
    dimer = ini.get("Dimer") or {}
    if "max_rotations" not in saddle and dimer.get("rotations_max") not in (None, ""):
        saddle["max_rotations"] = dimer["rotations_max"]
    ceiling = _step_ceiling(saddle)
    climb = int(saddle["max_iterations"]) * ceiling
    return str(climb + 2 * int(ini["Optimizer"]["max_iterations"]))


def _method_rows() -> list[dict[str, str]]:
    from optbench.score import score_v2

    rows = []
    methods = list(RGSADDLE_ROWS)
    if _eon_record_ready():
        methods.append("eon-classic")
    for method in methods:
        records = _ledger(method)
        counts = tally(records, step_ceiling(method))
        timing = {}
        if TIMING_PATH.is_file():
            timing = json.loads(TIMING_PATH.read_text(encoding="utf-8")).get(method) or {}
        wall_comparable = bool(timing.get("wall_comparable"))
        dat = _method_dat(method)
        fields = {}
        if dat.is_file():
            for line in dat.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                key, _, value = line.partition(" ")
                fields[key] = value
        scored = ""
        if records and method != "eon-classic":
            cap = int(fields.get("cap") or 0) or None
            metrics = score_v2(
                [
                    {
                        "force_calls": int(record["force_calls"]),
                        "success": bool(record["success"]),
                        "failure": record.get("failure"),
                    }
                    for record in records
                ],
                cap,
            )
            scored = f"{metrics['par2_mean']}"
            counts["par2_nfailed"] = metrics["nfailed"]
        row = {
            "row_kind": "method",
            "method": method,
            "template": fields.get("template", RGSADDLE_ROWS.get(method, "")),
            "commit": fields.get("commit", SHA if method != "eon-classic" else ""),
            "code_version": fields.get("code_version", ""),
            "client_version": timing.get("client_version", ""),
            "seed": fields.get("seed", str(SEED)),
            "wall_seconds": f"{float(timing['wall_seconds']):.6f}" if wall_comparable and timing.get("wall_seconds") is not None else "",
            "wall_comparable": str(wall_comparable).lower(),
            "par2_mean": scored or fields.get("par2_mean", ""),
            "force_calls": fields.get("force_calls", "")
            or (str(sum(int(record["force_calls"]) for record in records)) if records else ""),
            "budget": _protocol_budget(method, fields),
            "goal": fields.get("goal", str(GOAL)),
            "cap_failures": str(counts["cap"]),
        }
        for key, value in counts.items():
            if key == "cap":
                continue
            row[key] = str(value)
        rows.append(row)
    return rows


def csv_text() -> str:
    """The catalog CSV, without opening a record for writing."""
    hardware = _hardware_rows()
    commands = _command_rows()
    methods = _method_rows()
    columns: list[str] = []
    for row in hardware + commands + methods:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in hardware + commands + methods:
        writer.writerow(row)
    return buffer.getvalue()


def write_csv(path: Path | None = None) -> Path:
    """Write the catalog CSV. An existing destination stays unchanged."""
    destination = Path(path) if path is not None else CSV_PATH
    if destination.exists():
        raise ProvenanceError(f"{destination} is already a record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(csv_text(), encoding="utf-8")
    return destination


def _load_provenance_facts(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ProvenanceError("provenance facts are not an object")
    return document


def main(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: rgsaddle_develop_catalog.py run <method> <cpu> | "
            "eon <cpu> <client> <facts.json> | provenance <client> <facts.json> | "
            "progress <path> | csv",
            file=sys.stderr,
        )
        raise SystemExit(2)
    command = argv[0]
    if command == "run":
        method, cpu = argv[1], int(argv[2])
        if method not in RGSADDLE_ROWS:
            raise SystemExit(f"unknown method {method}")
        run_rgsaddle(method, cpu)
        return
    if command == "eon":
        run_eon(int(argv[1]), Path(argv[2]), _load_provenance_facts(Path(argv[3])))
        return
    if command == "provenance":
        client = Path(argv[1])
        facts = _load_provenance_facts(Path(argv[2]))
        document = provenance_document(**_provenance_kwargs(client, facts))
        write_provenance(PROVENANCE_PATH, document)
        print(PROVENANCE_PATH)
        return
    if command == "progress":
        Path(argv[1]).write_text(progress_text(), encoding="utf-8")
        return
    if command == "csv":
        print(write_csv())
        return
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
