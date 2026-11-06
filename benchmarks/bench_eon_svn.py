"""eOn SVN r2549 benchmarks.

The client is ``EON_SVN_CLIENT`` or ``~/opt/eon-svn/bin/eonclient``. It
must print ``r2549``. Every force evaluation goes through ``ExtPot`` and
``HarnessCounter``. Lennard-Jones kernels match the develop harness.
``morse_pt`` is r2549 ``Morse::force``: cutoff 9.5, the pair energy
shifted to zero at the cutoff, and the minimum image on the box
diagonal. The develop harness Morse has no cutoff and ignores the box.
r2549 reads L-BFGS keys from ``[Optimizer]``. It does not read
``energy_weighted``, ``ci_mmf``, or the modern L-BFGS catalog.

The default asv grid is one start. The campaign lists are
``campaign_seeds`` and ``campaign_starts``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from benchmarks.bench_eon_develop import (
    _client_files,
    _extract,
    classify_akmc,
    classify_client,
    kept_lowest,
    lammps_lj_params,
    lj_cut,
    selected,
)
from benchmarks.bench_eon_develop import MORSE_A, MORSE_D, MORSE_RE
from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
EON_REVISION = "r2549"
_CLIENT_OK = False
_BRIDGE: Path | None = None
_MORSE_LIB = None


def _client() -> str:
    global _CLIENT_OK
    env = os.environ.get("EON_SVN_CLIENT")
    if env:
        path = env
    else:
        default = Path.home() / "opt" / "eon-svn" / "bin" / "eonclient"
        path = str(default) if default.is_file() else (shutil.which("eonclient") or "")
    if not path or not Path(path).is_file():
        raise RuntimeError("put the r2549 eonclient on EON_SVN_CLIENT")
    if not _CLIENT_OK:
        out = subprocess.run([path, "-v"], check=False, capture_output=True, text=True)
        text = (out.stdout or "") + (out.stderr or "")
        if EON_REVISION not in text:
            raise RuntimeError(f"{path} -v printed {text!r}, expected {EON_REVISION}")
        _CLIENT_OK = True
    return path


def _bridge() -> Path:
    """Compile the repo bridge when it is newer than the cached binary.

    The client execs this path on every force call. Replacing the file
    is enough for a client that is already running.
    """
    global _BRIDGE
    source = ROOT / "optbench" / "extpot_bridge.c"
    dest = Path(tempfile.gettempdir()) / "optbench-eon-svn-extpot-bridge"
    stale = _BRIDGE is None or not Path(_BRIDGE).is_file()
    if not stale and source.is_file():
        stale = Path(_BRIDGE).stat().st_mtime < source.stat().st_mtime
    if not stale:
        return _BRIDGE
    if not dest.is_file() or (source.is_file() and dest.stat().st_mtime < source.stat().st_mtime):
        from optbench.fastpot import _link

        _link(["gcc", "-O2", str(source)], dest, check=True)
    _BRIDGE = dest
    return dest


MORSE_CUTOFF = 9.5
_MORSE_FLAGS = ("-O2", "-shared", "-fPIC")


def _morse_path() -> Path:
    """The compiled ``morse_cluster.c``, built once per source and flags.

    The name carries a hash of both, so another checkout's kernel is never
    loaded. Each build links in its own directory and is renamed into
    place, so a concurrent run never opens a partial file.
    """
    from optbench.fastpot import _build_dir, _link

    source = ROOT / "optbench" / "morse_cluster.c"
    digest = hashlib.sha256(source.read_bytes() + " ".join(_MORSE_FLAGS).encode()).hexdigest()[:16]
    dest = _build_dir() / f"eon-svn-morse-{digest}.so"
    if not dest.is_file():
        _link(["gcc", *_MORSE_FLAGS, str(source), "-lm"], dest, check=True)
    return dest


def _morse_lib():
    """C r2549 ``Morse::force`` for the 343-atom island.

    A Python pair loop spends tens of milliseconds on each call. A band
    or an AKMC repetition makes thousands of them.
    """
    import ctypes

    global _MORSE_LIB
    if _MORSE_LIB is not None:
        return _MORSE_LIB
    lib = ctypes.CDLL(str(_morse_path()))
    lib.morse_eon_flat.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    _MORSE_LIB = lib
    return lib


def _morse_fast(positions, atomic_numbers=None, box=None):
    import ctypes

    del atomic_numbers
    count = len(positions)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    atoms = count // 3
    pos = (ctypes.c_double * count)(*positions)
    forces = (ctypes.c_double * count)()
    energy = ctypes.c_double()
    box_buf = None
    if box is not None:
        values = [float(value) for value in box]
        if len(values) != 9:
            raise ValueError("box must be nine floats")
        box_buf = (ctypes.c_double * 9)(*values)
    _morse_lib().morse_eon_flat(
        atoms,
        pos,
        box_buf,
        ctypes.byref(energy),
        forces,
        MORSE_D,
        MORSE_A,
        MORSE_RE,
        MORSE_CUTOFF,
    )
    return float(energy.value), tuple(forces)


def _evaluate(potential: str):
    from optbench.lj import lj_cluster

    if potential == "ljcluster":
        return lambda positions, numbers, box: lj_cluster(positions, numbers, box)
    if potential == "morse_pt":
        return _morse_fast
    if potential == "lammps":
        lammps = ROOT / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
        epsilon, sigma, cutoff = lammps_lj_params(lammps)

        def _lj(positions, numbers, box, epsilon=epsilon, sigma=sigma, cutoff=cutoff):
            return lj_cut(positions, numbers, box, epsilon=epsilon, sigma=sigma, cutoff=cutoff)

        return _lj
    if potential == "lj_cut":
        return lambda positions, numbers, box: lj_cut(
            positions, numbers, box, epsilon=1.0, sigma=1.0, cutoff=40.0
        )
    raise ValueError(f"no harness kernel for potential {potential}")


class _Bridge:
    def __init__(self, potential: str, cap: int, evaluate=None):
        self.potential = potential
        self.cap = cap
        self.evaluate = evaluate
        self.server = None

    def __enter__(self):
        from optbench.potserver import PotServer

        function = self.evaluate or _evaluate(self.potential)
        self.server = PotServer(function, cap=self.cap).start()
        os.environ["OPTBENCH_POTSERVER"] = self.server.endpoint()
        return self.server, _bridge()

    def __exit__(self, exc_type, exc, tb):
        if self.server is not None:
            self.server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
        return False


def _ini(template: dict, seed: int, wrapper: Path, *, akmc: bool) -> dict:
    sections = json.loads(json.dumps(template["ini"]))
    sections.setdefault("Main", {})["random_seed"] = int(seed)
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    if akmc:
        # The Python schema rejects Main.parallel and a [LBFGS] section.
        sections.get("Main", {}).pop("parallel", None)
        sections.pop("LBFGS", None)
    return sections


def _guard(record: dict) -> dict:
    client_calls = record.get("client_calls")
    if record["force_calls"] == 0 and client_calls not in (None, 0):
        record["success"] = False
        record["failure"] = "crash"
    return record


def _stamp(template: dict, record: dict, seed, start, seconds: float) -> dict:
    record["seed"] = str(seed)
    record["start"] = str(start)
    record["seconds"] = float(seconds)
    record["counter"] = "harness"
    record["template"] = template["id"]
    record["problem"] = str(template.get("problem") or "")
    record["code"] = str(template.get("code") or "")
    record["code_version"] = str(template.get("code_version") or "")
    record["commit"] = str(template.get("commit") or "")
    record["tuning"] = str(template.get("tuning") or "")
    record["potential"] = template["ini"]["Potential"]["potential"]
    return record


def _neb_files(start: str, work: Path):
    from optbench.eon_jobs import _stored

    archive = _stored(ROOT, "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz")
    reactant = _extract(archive, f"minima/{start}/a.con", work / "a.con")
    product = _extract(archive, f"minima/{start}/b.con", work / "b.con")
    return [(str(reactant), "reactant.con"), (str(product), "product.con")], []


def _noise_files(start: str, work: Path):
    from optbench.eon_jobs import _stored

    archive = _stored(ROOT, "website/minimization/lj38/lj38-clusters.tgz")
    con = _extract(archive, f"lj38-clusters/{start}.con", work / "src.con")
    return [(str(con), "pos.con")], []


def _noise_evaluate(template: dict, start: str):
    """Clean lj/cut energy, Gaussian force noise, and the problem window.

    The client stops on a force norm. A sample under the threshold is
    scaled back above it until the clean-energy window is met, so one
    lucky draw does not end the search. Sigma 0 is the clean force.
    """
    from optbench.noise import WINDOW, energy_span, stop_threshold
    from optbench.quadratic import ENERGY_TOL

    sigma = float(template["noise_sigma"])
    kappa = float(template.get("noise_kappa", 3.0))
    window = int(template.get("noise_window", WINDOW))
    base_seed = int(template.get("noise_seed", 20261016))
    index = int(str(start).rsplit("_", 1)[-1])
    rng = random.Random(base_seed + index)
    components = 38 * 3
    clean_tol = float(template["ini"]["Optimizer"]["converged_force"])
    threshold = stop_threshold(clean_tol, sigma, components, kappa)
    energy_tol = max(ENERGY_TOL, kappa * sigma)
    energies: list[float] = []
    kernel = _evaluate("lj_cut")

    def evaluate(positions, numbers, box):
        energy, forces = kernel(positions, numbers, box)
        noisy = [float(value) for value in forces]
        if sigma > 0.0:
            for i in range(len(noisy)):
                noisy[i] += rng.gauss(0.0, sigma)
        energies.append(float(energy))
        norm = math.sqrt(sum(value * value for value in noisy))
        if sigma > 0.0 and norm <= threshold:
            recent = energies[-window:]
            span_ok = len(recent) >= window and energy_span(recent) <= energy_tol
            if not span_ok:
                target = threshold * (1.0 + 1e-8)
                if norm > 0.0:
                    scale = target / norm
                    noisy = [value * scale for value in noisy]
                elif noisy:
                    noisy[0] = target
        return float(energy), tuple(noisy)

    return evaluate, threshold, energy_tol


def _hessian_index(point: list[float], force_stop: float) -> int:
    import numpy as np

    from optbench.potentials import lennard_jones_hessian
    from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues

    matrix = lennard_jones_hessian(point)
    values = vibrational_eigenvalues(matrix, point, rotations=True)
    tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    return negative_index(values, tolerance)


def _read_con(path: Path) -> list[float]:
    from optbench.eon_jobs import _coords
    from optbench.lj import read_component_coordinates

    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return read_component_coordinates(text)
    except (StopIteration, ValueError):
        rows = _coords(text)
        flat: list[float] = []
        for x, y, z in rows:
            flat.extend((x, y, z))
        return flat


def _run_lowest(template: dict, seed: int, start: str, work: Path, server, wrapper: Path, cap: int) -> dict:
    from optbench.campaign import format_ini
    from optbench.eon_jobs import _lowest_once, _numpy_direction, _stored
    from optbench.eon_parse import abs_overlap, flat_floats
    from optbench.potserver import remote_count, remote_reset

    archive = _stored(ROOT, "website/lowest-eigenvector/lj38/coords-con.tar.gz")
    con = _extract(archive, f"coords-con/{start}.con", work / "src.con")
    mode = _extract(archive, f"coords-con/{start}_mode", work / "ref.mode")
    reference = flat_floats(mode.read_text(encoding="utf-8", errors="replace"))
    direction = _numpy_direction()
    sections = _ini(template, seed, wrapper, akmc=False)
    method = sections["Saddle Search"]["min_mode_method"]
    limit = int(template.get("iteration_limit", 1000))
    threshold = float(template.get("overlap", 0.99))
    client = _client()
    records = []
    for attempt in range(1, limit + 1):
        sections_k = json.loads(json.dumps(sections))
        if method == "dimer":
            sections_k.setdefault("Dimer", {})["rotations_max"] = attempt
        elif method == "lanczos":
            sections_k.setdefault("Lanczos", {})["max_iterations"] = attempt
        else:
            raise ValueError(f"unknown lowest mode {method}")
        remote_reset(server.host, server.port)
        found = _lowest_once(client, format_ini(sections_k), str(work / "run"), str(con), direction)
        calls, _verify = remote_count(server.host, server.port)
        overlap = None
        if found["mode_text"]:
            try:
                overlap = abs_overlap(flat_floats(found["mode_text"]), reference)
            except ValueError:
                overlap = None
        records.append({"calls": calls, "overlap": overlap, "found": found, "attempt": attempt})
        if overlap is not None and overlap >= threshold:
            break
    index = kept_lowest([(row["calls"], row["overlap"]) for row in records], threshold)
    if index is None:
        return classify_client(calls=None, reason=None, cap=cap, crashed=True)
    chosen = records[index]
    found = chosen["found"]
    overlap = chosen["overlap"]
    reached = overlap is not None and overlap >= threshold
    record = classify_client(
        calls=chosen["calls"],
        reason=found.get("reason"),
        cap=cap,
        overlap_ok=reached,
    )
    record["client_calls"] = found.get("raw")
    record["client_reason"] = found.get("reason")
    record["k"] = chosen["attempt"] if reached else None
    record["overlap"] = overlap
    return _guard(record)


def _run_akmc(template: dict, seed: int, work: Path, server, wrapper: Path, cap: int, passes: int, workers: int) -> dict:
    from optbench.eon_akmc import _one_repetition
    from optbench.potserver import remote_count

    patched = dict(template)
    patched["ini"] = _ini(template, seed, wrapper, akmc=True)
    row = _one_repetition(
        ROOT,
        patched,
        Path(_client()),
        work / "akmc",
        int(seed),
        workers,
        passes=passes,
    )
    calls, _verify = remote_count(server.host, server.port)
    crashed = bool(row.get("error")) and not row.get("total_searches")
    record = classify_akmc(calls=calls, cap=cap, reached=bool(row.get("reached")), crashed=crashed)
    record["client_calls"] = row.get("force_calls")
    record["fraction_connected"] = row.get("fraction_connected")
    record["passes"] = row.get("passes")
    record["qualifying"] = row.get("qualifying")
    record["total_searches"] = row.get("total_searches")
    record["unique_saddles"] = row.get("unique_saddles")
    record["error"] = row.get("error") or ""
    return _guard(record)


def _apply_noise_stop(template: dict, sections: dict) -> float | None:
    if "noise_sigma" not in template:
        return None
    from optbench.noise import stop_threshold

    sigma = float(template["noise_sigma"])
    kappa = float(template.get("noise_kappa", 3.0))
    clean = float(template["ini"]["Optimizer"]["converged_force"])
    threshold = stop_threshold(clean, sigma, 38 * 3, kappa)
    sections.setdefault("Optimizer", {})["converged_force"] = threshold
    return threshold


def run_case(template_name: str, kind: str, seed: int, start: str, *, passes: int = 0, workers: int = 1) -> dict:
    """One seed and one start. The dict is what the tracks report."""
    from optbench.campaign import format_ini, load_template
    from optbench.eon_jobs import _execute
    from optbench.potserver import remote_count

    template = load_template(ROOT, template_name)
    potential = template["ini"]["Potential"]["potential"]
    cap = int(template["failure_cap"])
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="optbench-eon-svn-"))
    noise_eval = None
    if kind == "noise":
        noise_eval, _threshold, _energy_tol = _noise_evaluate(template, start)
    try:
        with _Bridge(potential, cap, noise_eval) as (server, wrapper):
            if kind == "lowest":
                record = _run_lowest(template, seed, start, work, server, wrapper, cap)
            elif kind == "akmc":
                record = _run_akmc(template, seed, work, server, wrapper, cap, passes, workers)
            else:
                if kind == "neb-pt":
                    copies, texts = _neb_files(start, work)
                elif kind == "noise":
                    copies, texts = _noise_files(start, work)
                else:
                    copies, texts = _client_files(kind, start, work)
                sections = _ini(template, seed, wrapper, akmc=False)
                _apply_noise_stop(template, sections)
                found = _execute(
                    (
                        _client(),
                        format_ini(sections),
                        str(work / "run"),
                        copies,
                        texts,
                        "total_force_calls",
                    )
                )
                calls, _verify = remote_count(server.host, server.port)
                record = classify_client(calls=calls, reason=found.get("reason"), cap=cap)
                record["client_calls"] = found.get("raw")
                record["client_reason"] = found.get("reason")
                record["extrema"] = found.get("extrema")
                if kind == "noise" and record["success"]:
                    # MinimizationJob writes the relaxed geometry to min.con.
                    # pos.con is the start and is not a stationary point.
                    final = work / "run" / "min.con"
                    try:
                        point = _read_con(final)
                        index = _hessian_index(point, float(sections["Optimizer"]["converged_force"]))
                        record["hessian_index"] = index
                        if index != 0:
                            record["success"] = False
                            record["failure"] = "wrong_stationary"
                    except (OSError, ValueError, RuntimeError) as exc:
                        record["success"] = False
                        record["failure"] = "crash"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                record = _guard(record)
                err_path = work / "run" / "stderr.txt"
                if err_path.is_file() and not record.get("error"):
                    record["error"] = err_path.read_text(encoding="utf-8", errors="replace")[:400]
        return _stamp(template, record, seed, start, time.monotonic() - started)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _optional(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _bind(spec: dict):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [spec["seeds"], spec["starts"]]
        timeout = spec["timeout"]
        template_name = spec["template"]
        template_id = spec["template"]
        problem_id = spec["problem"]
        kind = spec["kind"]
        cap = spec["cap"]
        published_mean = False
        campaign_seeds = spec["campaign_seeds"]
        campaign_starts = spec["campaign_starts"]
        campaign_passes = spec.get("campaign_passes", 0)
        passes = spec.get("passes", 0)
        workers = spec.get("workers", 1)
        potential = spec["potential"]
        version = EON_REVISION

        def setup(self, seed, start):
            return None

        def measure(self, seed, start):
            self.record = run_case(
                self.template_name,
                self.kind,
                int(seed),
                str(start),
                passes=int(self.passes or 0),
                workers=int(self.workers or 1),
            )
            return self.record

        def time_search(self, seed, start):
            self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(self.measure(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "crash" else 0.0

    _Benchmark.__name__ = spec["cls"]
    _Benchmark.__qualname__ = spec["cls"]
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    if spec["kind"] == "akmc":

        def track_unique_saddles(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "unique_saddles")

        def track_fraction_connected(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "fraction_connected")

        def track_total_searches(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "total_searches")

        track_unique_saddles.unit = "saddles"
        track_fraction_connected.unit = "fraction"
        track_total_searches.unit = "searches"
        _Benchmark.track_unique_saddles = track_unique_saddles
        _Benchmark.track_fraction_connected = track_fraction_connected
        _Benchmark.track_total_searches = track_total_searches
    if spec["kind"] == "de-2ts":

        def track_extrema(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "extrema")

        track_extrema.unit = "count"
        _Benchmark.track_extrema = track_extrema
    return _Benchmark


def _akmc_passes(limit: int) -> int:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return int(limit)
    raw = os.environ.get("OPTBENCH_ASV_AKMC_PASSES")
    if raw:
        return int(raw)
    return 1


def neb_starts() -> list[str]:
    import tarfile

    from optbench.eon_jobs import _stored

    archive = _stored(ROOT, "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz")
    found = set()
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            parts = member.name.split("/")
            if len(parts) >= 2 and parts[0] == "minima" and parts[1].startswith("min"):
                found.add(parts[1])
    names = sorted(found)
    if len(names) != 13:
        raise RuntimeError(f"Pt-heptamer archive has {len(names)} minima, expected 13")
    return names


def _specs() -> list[dict]:
    from optbench.campaign import load_template

    lj = [f"{i:04d}" for i in range(200)]
    noise = [f"cluster_{i:04d}" for i in range(1000)]
    workers = min(8, os.cpu_count() or 1)
    rows = [
        ("EonSvnTsLj38Dimer", "eon-svn/ts-lj38-dimer", "ts-lj38", ["1"], lj, 900),
        ("EonSvnTsLj38Lanczos", "eon-svn/ts-lj38-lanczos", "ts-lj38", ["1"], lj, 900),
        ("EonSvnTsPtDimer", "eon-svn/ts-pt-dimer", "ts-pt", ["1"], [f"initial_{i}" for i in range(49)], 1800),
        ("EonSvnTsPtLanczos", "eon-svn/ts-pt-lanczos", "ts-pt", ["1"], [f"initial_{i}" for i in range(49)], 1800),
        ("EonSvnDeLj38Ci", "eon-svn/de-lj38-ci", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 7200),
        ("EonSvnDeLj38Neb", "eon-svn/de-lj38-neb", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 7200),
        ("EonSvnDePtCi", "eon-svn/de-pt-ci", "de-pt", ["1"], [f"product_{i}" for i in range(59)], 3600),
        ("EonSvnDePtNeb", "eon-svn/de-pt-neb", "de-pt", ["1"], [f"product_{i}" for i in range(59)], 3600),
        ("EonSvnDe2tsCi", "eon-svn/de-2ts-ci", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonSvnDe2tsNeb", "eon-svn/de-2ts-neb", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonSvnNebPtCi", "eon-svn/neb-pt-ci", "neb-pt", ["1"], neb_starts(), 3600),
        ("EonSvnNebPtNeb", "eon-svn/neb-pt-neb", "neb-pt", ["1"], neb_starts(), 3600),
        ("EonSvnLowLj38Dimer", "eon-svn/low-lj38-dimer", "lowest", ["1"], lj, 1800),
        ("EonSvnLowLj38Lanczos", "eon-svn/low-lj38-lanczos", "lowest", ["1"], lj, 1800),
        ("EonSvnLj38Noise0", "eon-svn/lj38-noise-0", "noise", ["20261016"], noise, 900),
        ("EonSvnLj38Noise1em4", "eon-svn/lj38-noise-1e-4", "noise", ["20261016"], noise, 900),
        ("EonSvnLj38Noise1em3", "eon-svn/lj38-noise-1e-3", "noise", ["20261016"], noise, 900),
        ("EonSvnLj38Noise1em2", "eon-svn/lj38-noise-1e-2", "noise", ["20261016"], noise, 900),
        (
            "EonSvnSaddlePtDimer",
            "eon-svn/saddle-pt-dimer",
            "akmc",
            [str(seed) for seed in range(1000, 1010)],
            ["pt-island"],
            7200,
        ),
        (
            "EonSvnSaddlePtDimerImproved",
            "eon-svn/saddle-pt-dimer-improved",
            "akmc",
            [str(seed) for seed in range(1000, 1010)],
            ["pt-island"],
            7200,
        ),
        (
            "EonSvnSaddlePtLanczos",
            "eon-svn/saddle-pt-lanczos",
            "akmc",
            ["1000", "1001"],
            ["pt-island"],
            7200,
        ),
    ]
    specs = []
    for cls, template_name, kind, seeds, starts, timeout in rows:
        template = load_template(ROOT, template_name)
        limit = int(template.get("pass_limit", 0) or 0)
        spec = {
            "cls": cls,
            "template": template_name,
            "problem": str(template.get("problem") or ""),
            "kind": kind,
            "campaign_seeds": list(seeds),
            "campaign_starts": list(starts),
            "seeds": selected(seeds),
            "starts": selected(starts),
            "timeout": timeout,
            "cap": int(template["failure_cap"]),
            "potential": template["ini"]["Potential"]["potential"],
            "workers": workers if kind == "akmc" else 1,
            "passes": _akmc_passes(limit) if kind == "akmc" else 0,
            "campaign_passes": limit if kind == "akmc" else 0,
        }
        specs.append(spec)
    return specs


SPECS = _specs()
for _spec in SPECS:
    globals()[_spec["cls"]] = _bind(_spec)
