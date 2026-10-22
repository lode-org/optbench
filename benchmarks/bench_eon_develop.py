"""eOn develop benchmarks for asv.

One class is one template. Parameters are the seed and the start.
``measure`` returns the record ``track_*`` reads: harness ``force_calls``,
``success``, and one ``failure`` class. ``time_search`` is the wall
clock. ``setup`` does not run the client, because asv does not time
``setup``. The first call writes a disk cache and later tracks read it,
so a suite that runs ``time_search`` first times the search once.

Every force evaluation goes through ``HarnessCounter``. eonclient is
started with ``potential = ext_pot``. Client inis set ``Main.parallel``
false, so one ``ExtPot`` serves every image. A per-image clone would put
dimer rotations and the forward image on a second instance.
Saddle-search ``results.dat`` stores only the primary
``forceCallCounter``, and the harness would then disagree with that
field. One instance records the same calls in both. The AKMC server
ini does not set ``parallel``: the Python schema rejects the option and
exits before the first search. Those jobs keep the client default. The
harness still counts every ``force()``. ``sad-fcs`` plus ``mins-fcs``
starts after the reactant evaluation, one real force call per search.
A native bridge forwards each call to a ``PotServer`` in this process,
and ``python -m optbench.extpot`` does that when the bridge is not built.
The server refuses a search call at ``failure_cap`` before the potential
runs, and that run is ``cap``. The client's ``total_force_calls`` is
stored beside the harness count and is not the track. The binary is the
develop source build named by ``EONCLIENT``
(commit 9d77066c375c11602514c7de2931c1ee936e93f6). The conda-forge 3.5.0
pin is a different environment.

The default grid is one seed and the first start. ``OPTBENCH_ASV_FULL=1``
at import selects the campaign grid. That subset is not a published
mean. AKMC defaults to one pass (``OPTBENCH_ASV_AKMC_PASSES`` overrides
it; the full flag uses the template ``pass_limit``). A pass that does
not reach 27 saddles is a cap failure. GPR-dimer is not a class: this
build reports GPRD disabled.

``cap`` is the template ``failure_cap``. The result row does not store it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import tarfile
import tempfile
import time
from pathlib import Path

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
EON_COMMIT = "9d77066c375c11602514c7de2931c1ee936e93f6"
EON_VERSION_TOKEN = "9d77066"
MORSE_D = 0.7102
MORSE_A = 1.6047
MORSE_RE = 2.897
# rgpot MorsePot defaults. eOn ``morse_pt`` is that kernel: cutoff 9.5,
# energy shifted so a pair at the cutoff is zero, no quintic switch.
MORSE_CUTOFF = 9.5
_MORSE_GAP_CUT = 1.0 - math.exp(-MORSE_A * (MORSE_CUTOFF - MORSE_RE))
MORSE_SHIFT = MORSE_D * _MORSE_GAP_CUT * _MORSE_GAP_CUT - MORSE_D
_CACHE = Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv"))
_CLIENT_OK = False


def full_grid() -> bool:
    return os.environ.get("OPTBENCH_ASV_FULL") == "1"


def selected(values: list) -> list:
    """Import-time parameter slice. One entry unless the full flag is set.

    ``OPTBENCH_ASV_SLICE=worker/count`` strides an axis that already has
    more than one entry. A length-1 axis is left whole, so the only seed
    or the only start stays on every worker.
    """
    if full_grid():
        chosen = list(values)
    else:
        chosen = list(values[:1])
    spec = os.environ.get("OPTBENCH_ASV_SLICE", "")
    if not spec or len(chosen) <= 1:
        return chosen
    worker_text, count_text = spec.split("/")
    worker = int(worker_text)
    count = int(count_text)
    if count <= 0 or worker < 0 or worker >= count:
        raise ValueError(f"OPTBENCH_ASV_SLICE must be worker/count, got {spec}")
    return chosen[worker::count]


def _morse_lengths(box) -> tuple[float, float, float] | None:
    """Orthogonal periodic widths, or ``None`` for a free cluster.

    eOn writes the cell column-major, which is how ``read_extpot`` stores
    the nine floats. A zero cell is the non-periodic box ``Matter`` hands
    a potential. Widths sit on the diagonal (indices 0, 4, 8), the same
    places rgpot reads for an orthorhombic cell.
    """
    if box is None:
        return None
    vals = [float(value) for value in box]
    if len(vals) != 9:
        raise ValueError("box is nine floats")
    if all(value == 0.0 for value in vals):
        return None
    off = (vals[1], vals[2], vals[3], vals[5], vals[6], vals[7])
    if any(value != 0.0 for value in off):
        raise ValueError("morse_pt minimum image is implemented for an orthogonal cell")
    lengths = (vals[0], vals[4], vals[8])
    if any(length <= 0.0 for length in lengths):
        raise ValueError("morse_pt box lengths must be positive")
    return lengths


def _morse_fold(delta: float, length: float) -> float:
    """Nearest image. Half-integers round away from zero, as rgpot ``visit_round``."""
    scaled = delta / length
    image = float(int(scaled + math.copysign(0.5, scaled)))
    return delta - length * image


def _morse_python(positions, atomic_numbers=None, box=None):
    """Pair Morse used by eOn ``morse_pt``, the rgpot ``MorsePot`` defaults.

    ``(D, a, r_e, cutoff)`` = (0.7102, 1.6047, 2.897, 9.5). The pair energy
    is ``D (1 - exp(-a (r - r_e)))^2 - D``, shifted so the cutoff is zero.
    Pairs at or beyond the cutoff are left out. A non-zero orthogonal box
    is periodic in all three directions and uses the minimum image. A
    missing or zero box is a free cluster, which is what eOn passes when
    periodic boundaries are off.
    """
    del atomic_numbers
    count = len(positions)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    atoms = count // 3
    lengths = _morse_lengths(box)
    cutoff2 = MORSE_CUTOFF * MORSE_CUTOFF
    forces = [0.0] * count
    energy = 0.0
    two_de_a = 2.0 * MORSE_D * MORSE_A
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = float(positions[ix]) - float(positions[jx])
            dy = float(positions[ix + 1]) - float(positions[jx + 1])
            dz = float(positions[ix + 2]) - float(positions[jx + 2])
            if lengths is not None:
                dx = _morse_fold(dx, lengths[0])
                dy = _morse_fold(dy, lengths[1])
                dz = _morse_fold(dz, lengths[2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0 or r2 > cutoff2:
                continue
            r = math.sqrt(r2)
            gap = 1.0 - math.exp(-MORSE_A * (r - MORSE_RE))
            energy += MORSE_D * gap * gap - MORSE_D - MORSE_SHIFT
            scale = two_de_a * gap * (gap - 1.0) / r
            fx = scale * dx
            fy = scale * dy
            fz = scale * dz
            forces[ix] += fx
            forces[ix + 1] += fy
            forces[ix + 2] += fz
            forces[jx] -= fx
            forces[jx + 1] -= fy
            forces[jx + 2] -= fz
    return energy, tuple(forces)


def morse_cluster(positions, atomic_numbers=None, box=None):
    """``morse_pt`` energy and forces. The native kernel matches ``_morse_python``."""
    if os.environ.get("OPTBENCH_FASTPOT") != "0":
        from optbench.fastpot import morse_forces

        lengths = _morse_lengths(box)
        got = morse_forces(positions, lengths, MORSE_D, MORSE_A, MORSE_RE, MORSE_CUTOFF, MORSE_SHIFT)
        if got is not None:
            return got
    return _morse_python(positions, atomic_numbers, box)


def _lj_cut_python(positions, atomic_numbers=None, box=None, *, epsilon=1.0, sigma=1.0, cutoff=40.0):
    """Unshifted Lennard-Jones, the archived ``pair_style lj/cut`` kernel.

    ``in.lammps`` sets ``lj/cut/opt`` with cutoff 40 and ``pair_coeff * * 1 1``.
    It has no ``pair_modify shift``. Pairs at or beyond the cutoff are
    left out. The box is ignored: these geometries are free clusters.
    """
    del atomic_numbers, box
    count = len(positions)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    atoms = count // 3
    forces = [0.0] * count
    if atoms < 2:
        return 0.0, tuple(forces)
    cutoff2 = cutoff * cutoff
    sigma2 = sigma * sigma
    energy = 0.0
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = float(positions[ix]) - float(positions[jx])
            dy = float(positions[ix + 1]) - float(positions[jx + 1])
            dz = float(positions[ix + 2]) - float(positions[jx + 2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0 or r2 >= cutoff2:
                continue
            inv_r2 = 1.0 / r2
            sr2 = sigma2 * inv_r2
            attraction = sr2 * sr2 * sr2
            well = 4.0 * epsilon * attraction
            energy += well * (attraction - 1.0)
            scale = 6.0 * well * inv_r2 * (2.0 * attraction - 1.0)
            fx = scale * dx
            fy = scale * dy
            fz = scale * dz
            forces[ix] += fx
            forces[ix + 1] += fy
            forces[ix + 2] += fz
            forces[jx] -= fx
            forces[jx + 1] -= fy
            forces[jx + 2] -= fz
    return energy, tuple(forces)


def lj_cut(positions, atomic_numbers=None, box=None, *, epsilon=1.0, sigma=1.0, cutoff=40.0):
    """``lj/cut`` energy and forces. The native kernel matches ``_lj_cut_python``."""
    if os.environ.get("OPTBENCH_FASTPOT") != "0":
        from optbench.fastpot import lj_cut_forces

        got = lj_cut_forces(positions, epsilon, sigma, cutoff)
        if got is not None:
            return got
    return _lj_cut_python(positions, atomic_numbers, box, epsilon=epsilon, sigma=sigma, cutoff=cutoff)


def lammps_lj_params(path: Path) -> tuple[float, float, float]:
    """``(epsilon, sigma, cutoff)`` from an ``lj/cut`` input."""
    epsilon = None
    sigma = None
    cutoff = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "pair_style" and parts[1].startswith("lj/cut"):
            cutoff = float(parts[2])
        if len(parts) >= 5 and parts[0] == "pair_coeff":
            epsilon = float(parts[3])
            sigma = float(parts[4])
    if epsilon is None or sigma is None or cutoff is None:
        raise ValueError(f"{path} is not an lj/cut pair_coeff")
    return epsilon, sigma, cutoff


def classify_client(*, calls, reason, cap, overlap_ok=None, crashed=False) -> dict:
    """Map one client result onto the v2 failure order.

    ``cap`` wins when the harness count reached the budget, including a
    client that died when the counter refused the next call. A non-zero
    reason under the cap is ``divergence``. A lowest-mode overlap miss
    under the cap is ``wrong_stationary``. Overlap success ignores client
    reason 5, which is the climb cap after the mode was computed.
    """
    if calls is None:
        return {"force_calls": 0.0, "success": False, "failure": "crash"}
    count = int(calls)
    if overlap_ok is True and count < cap:
        return {"force_calls": float(count), "success": True, "failure": None}
    if overlap_ok is False and count < cap:
        return {"force_calls": float(count), "success": False, "failure": "wrong_stationary"}
    if count >= cap:
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if crashed:
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if reason in (None, ""):
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if str(reason) != "0":
        return {"force_calls": float(count), "success": False, "failure": "divergence"}
    return {"force_calls": float(count), "success": True, "failure": None}


def classify_akmc(*, calls, cap, reached, crashed) -> dict:
    """A repetition that misses 27 saddles spent the pass budget.

    A harness count at the force cap is ``cap`` even when the client
    process died on the refused call. A crash under the cap stays a crash.
    """
    if calls is None:
        return {"force_calls": 0.0, "success": False, "failure": "crash"}
    count = int(calls)
    if count >= cap:
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if crashed:
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if reached:
        return {"force_calls": float(count), "success": True, "failure": None}
    return {"force_calls": float(count), "success": False, "failure": "cap"}


def _client() -> str:
    global _CLIENT_OK
    env = os.environ.get("EONCLIENT")
    if env:
        path = env
    else:
        default = Path.home() / "opt" / "eon" / "bin" / "eonclient"
        path = str(default) if default.is_file() else (shutil.which("eonclient") or "")
    if not path or not Path(path).is_file():
        raise RuntimeError("put the develop eonclient on EONCLIENT")
    if not _CLIENT_OK:
        from optbench.validate_counter import client_version

        _version, commit, _first = client_version(Path(path))
        if not commit.startswith(EON_VERSION_TOKEN):
            raise RuntimeError(f"{path} prints {commit}, expected {EON_COMMIT}")
        _CLIENT_OK = True
    return path


def _evaluate(potential: str):
    from optbench.lj import lj_cluster

    if potential == "ljcluster":
        return lambda positions, numbers, box: lj_cluster(positions, numbers, box)
    if potential == "morse_pt":
        return morse_cluster
    if potential == "lammps":
        lammps = ROOT / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
        epsilon, sigma, cutoff = lammps_lj_params(lammps)

        def _lj(positions, numbers, box, epsilon=epsilon, sigma=sigma, cutoff=cutoff):
            return lj_cut(positions, numbers, box, epsilon=epsilon, sigma=sigma, cutoff=cutoff)

        return _lj
    raise ValueError(f"no harness kernel for potential {potential}")


def _write_wrapper(path: Path) -> None:
    from optbench.fastpot import bridge_executable

    binary = bridge_executable()
    if binary is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        path.symlink_to(binary)
        return
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "root = os.environ.get('OPTBENCH_ROOT')\n"
        "if root:\n"
        "    sys.path.insert(0, root)\n"
        "from optbench.extpot import main\n"
        "main()\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class _Bridge:
    def __init__(self, potential: str, cap: int):
        self.potential = potential
        self.cap = cap
        self.server = None
        self.directory = None

    def __enter__(self):
        from optbench.potserver import PotServer

        self.directory = Path(tempfile.mkdtemp(prefix="optbench-extpot-"))
        wrapper = self.directory / "extpot-bridge"
        _write_wrapper(wrapper)
        self.server = PotServer(_evaluate(self.potential), cap=self.cap).start()
        os.environ["OPTBENCH_POTSERVER"] = self.server.endpoint()
        os.environ["OPTBENCH_ROOT"] = str(ROOT)
        return self.server, wrapper

    def __exit__(self, exc_type, exc, tb):
        if self.server is not None:
            self.server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
        return False


def _one_potential(sections: dict) -> dict:
    """Record every ``force()`` on the instance ``results.dat`` reads.

    ``ExtPot`` needs one instance per image when ``Main.parallel`` is
    true, and that flag defaults to true. The forward image and the
    dimer rotation trials then run on a second potential. Each of those
    calls is one force evaluation. A saddle-search ``total_force_calls``
    is only the primary counter, so it omits them. The harness counts
    every subprocess. Turning parallel off keeps one ``ExtPot`` and the
    two numbers are the same count.
    """
    sections.setdefault("Main", {})["parallel"] = False
    return sections


def _ini(template: dict, seed: int, wrapper: Path) -> str:
    from optbench.campaign import format_ini

    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = int(seed)
    _one_potential(sections)
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    return format_ini(sections)


def kept_lowest(attempts: list[tuple], threshold: float) -> int | None:
    """Index of the first attempt whose overlap meets ``threshold``.

    Each attempt is ``(calls, overlap)`` after the counter was reset, so
    the kept count is that attempt alone. Earlier attempts are not added.
    The index is the last attempt when none of them meet the threshold,
    and ``None`` when ``attempts`` is empty.
    """
    if not attempts:
        return None
    for index, (_calls, overlap) in enumerate(attempts):
        if overlap is not None and overlap >= threshold:
            return index
    return len(attempts) - 1


def _index(start: str) -> int:
    if "_" in start:
        return int(start.rsplit("_", 1)[1])
    return int(start)


def _extract(archive: Path, member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        extracted = packed.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is not in {archive}")
        dest.write_bytes(extracted.read())
    return dest


def _cache_path(template_name: str, seed: int, start: str, passes: int) -> Path:
    safe = template_name.replace("/", "-")
    return _CACHE / "harness-1" / EON_COMMIT / safe / f"passes-{passes}" / str(seed) / f"{start}.json"


def _read_cache(template_name: str, seed: int, start: str, passes: int) -> dict | None:
    path = _cache_path(template_name, seed, start, passes)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(template_name: str, seed: int, start: str, passes: int, row: dict) -> None:
    path = _cache_path(template_name, seed, start, passes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _client_files(kind: str, start: str, work: Path):
    from optbench.eon_jobs import _coords, _difference_mode, _seeded_mode, _stored

    if kind == "ts-lj38":
        archive = _stored(ROOT, "website/tsopt/LJ38/coords-con.tar.gz")
        con = _extract(archive, f"coords-con/{start}.con", work / "src.con")
        text = con.read_text(encoding="utf-8", errors="replace")
        mode = _seeded_mode(len(_coords(text)), _index(start))
        return [(str(con), "pos.con"), (str(con), "displacement.con")], [("direction.dat", mode)]
    if kind == "ts-pt":
        archive = _stored(ROOT, "website/tsopt/pt-island/pt-island-con.tgz")
        reactant = _extract(archive, "pt-island-con/reactant.con", work / "reactant.con")
        initial = _extract(
            archive,
            f"pt-island-con/initial_{_index(start)}.con",
            work / "initial.con",
        )
        mode = _difference_mode(reactant, initial)
        return (
            [(str(reactant), "pos.con"), (str(initial), "displacement.con")],
            [("direction.dat", mode)],
        )
    if kind == "de-lj38":
        archive = _stored(ROOT, "website/double-ended-saddle-search/lj38/coords-con.tgz")
        index = _index(start)
        reactant = _extract(archive, f"coords-con/start_{index:02d}.con", work / "start.con")
        product = _extract(archive, f"coords-con/end_{index:02d}.con", work / "end.con")
        lammps = ROOT / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
        return (
            [
                (str(reactant), "reactant.con"),
                (str(product), "product.con"),
                (str(lammps), "in.lammps"),
            ],
            [],
        )
    if kind == "de-pt":
        archive = _stored(ROOT, "website/double-ended-saddle-search/pt-island/pt-island-con.tgz")
        reactant = _extract(archive, "pt-island-con/reactant.con", work / "reactant.con")
        product = _extract(
            archive,
            f"pt-island-con/product_{_index(start)}.con",
            work / "product.con",
        )
        return [(str(reactant), "reactant.con"), (str(product), "product.con")], []
    if kind == "neb-pt":
        from optbench.eon_jobs import _extract as _unpack

        unpacked = _unpack(
            _stored(ROOT, "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz"),
            work / "unpacked",
        )
        src = unpacked / "minima" / start
        return (
            [(str(src / "a.con"), "reactant.con"), (str(src / "b.con"), "product.con")],
            [],
        )
    if kind == "de-2ts":
        from optbench.eon_parse import xyz_to_con

        archive = _stored(ROOT, "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz")
        index = _index(start)
        start_xyz = _extract(archive, f"coords/start_{index}.xyz", work / "start.xyz")
        end_xyz = _extract(archive, f"coords/end_{index}.xyz", work / "end.xyz")
        return (
            [],
            [
                ("reactant.con", xyz_to_con(start_xyz.read_text(encoding="utf-8"))),
                ("product.con", xyz_to_con(end_xyz.read_text(encoding="utf-8"))),
            ],
        )
    raise ValueError(f"unknown eon benchmark kind {kind}")


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
    sections = json.loads(json.dumps(template["ini"]))
    sections.setdefault("Main", {})["random_seed"] = int(seed)
    _one_potential(sections)
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
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
    return _guard_uncounted(record)


def _guard_uncounted(record: dict) -> dict:
    """A finished client that never touched the server did not use the harness."""
    client_calls = record.get("client_calls")
    if record["force_calls"] == 0 and client_calls not in (None, 0):
        record["success"] = False
        record["failure"] = "crash"
    return record


def _akmc_server_ini(template: dict, wrapper: Path) -> dict:
    """Server config for one AKMC repetition.

    ``Main.parallel`` stays unset. The Python schema has no such option
    and ``ConfigClass.init`` exits when it is present. The C++ client
    then keeps its default. ``_ini`` still sets the flag for the runs
    the client parses itself.
    """
    ini = json.loads(json.dumps(template["ini"]))
    ini["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    return ini


def _run_akmc(template: dict, seed: int, work: Path, server, wrapper: Path, cap: int, passes: int, workers: int) -> dict:
    from optbench.eon_akmc import _one_repetition
    from optbench.potserver import remote_count

    # ``load_template`` stores a Path on ``_path``. Copy the ini only.
    patched = dict(template)
    patched["ini"] = _akmc_server_ini(template, wrapper)
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
    return _guard_uncounted(record)


def run_case(template_name: str, kind: str, seed: int, start: str, *, passes: int = 0, workers: int = 1) -> dict:
    """One seed and one start. The dict is what the tracks report."""
    from optbench.campaign import load_template
    from optbench.eon_jobs import _execute
    from optbench.potserver import remote_count

    template = load_template(ROOT, template_name)
    potential = template["ini"]["Potential"]["potential"]
    cap = int(template["failure_cap"])
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="optbench-eon-"))
    try:
        with _Bridge(potential, cap) as (server, wrapper):
            if kind == "lowest":
                record = _run_lowest(template, seed, start, work, server, wrapper, cap)
            elif kind == "akmc":
                record = _run_akmc(template, seed, work, server, wrapper, cap, passes, workers)
            else:
                copies, texts = _client_files(kind, start, work)
                count_key = "force_calls_neb" if kind == "neb-pt" else "total_force_calls"
                found = _execute(
                    (
                        _client(),
                        _ini(template, seed, wrapper),
                        str(work / "run"),
                        copies,
                        texts,
                        count_key,
                    )
                )
                calls, _verify = remote_count(server.host, server.port)
                record = classify_client(calls=calls, reason=found.get("reason"), cap=cap)
                record["client_calls"] = found.get("raw")
                record["client_reason"] = found.get("reason")
                record["extrema"] = found.get("extrema")
                record = _guard_uncounted(record)
        record["seconds"] = time.monotonic() - started
        record["potential"] = potential
        record["counter"] = "harness"
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _ensure(bench, seed, start) -> dict:
    seed = int(seed)
    start = str(start)
    passes = int(getattr(bench, "passes", 0) or 0)
    cached = _read_cache(bench.template_name, seed, start, passes)
    if cached is not None:
        return cached
    row = run_case(
        bench.template_name,
        bench.kind,
        seed,
        start,
        passes=passes,
        workers=int(getattr(bench, "workers", 1) or 1),
    )
    _write_cache(bench.template_name, seed, start, passes, row)
    return row


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
        kind = spec["kind"]
        cap = spec["cap"]
        published_mean = False
        campaign_seeds = spec["campaign_seeds"]
        campaign_starts = spec["campaign_starts"]
        passes = spec.get("passes", 0)
        workers = spec.get("workers", 1)
        potential = spec["potential"]
        version = EON_COMMIT

        def setup(self, seed, start):
            return None

        def measure(self, seed, start):
            self.record = _ensure(self, seed, start)
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
    if full_grid():
        return int(limit)
    raw = os.environ.get("OPTBENCH_ASV_AKMC_PASSES")
    if raw:
        return int(raw)
    return 1


def _specs() -> list[dict]:
    from optbench.campaign import load_template

    lj = [f"{i:04d}" for i in range(200)]
    workers = min(8, os.cpu_count() or 1)
    rows = [
        ("EonDevelopTsLj38Lanczos", "eon/develop-ts-lj38-lanczos", "ts-lj38", ["1"], lj, 900),
        ("EonDevelopTsLj38Dimer", "eon/develop-ts-lj38-dimer-accept-energy", "ts-lj38", ["1"], lj, 900),
        ("EonDevelopTsLj38DimerMatched", "eon/develop-ts-lj38-dimer", "ts-lj38", ["1"], lj, 900),
        ("EonDevelopTsPtLanczos", "eon/develop-ts-pt-lanczos", "ts-pt", ["1"], [f"initial_{i}" for i in range(49)], 1800),
        ("EonDevelopTsPtDimer", "eon/develop-ts-pt-dimer-accept-energy", "ts-pt", ["1"], [f"initial_{i}" for i in range(49)], 1800),
        ("EonDevelopTsPtDimerMatched", "eon/develop-ts-pt-dimer", "ts-pt", ["1"], [f"initial_{i}" for i in range(49)], 1800),
        (
            "EonDevelopNebPtLbfgs",
            "eon/neb-pt-lbfgs",
            "neb-pt",
            ["1"],
            [f"min{i:02d}" for i in range(1, 14)],
            3600,
        ),
        (
            "EonDevelopNebPtLbfgsM100",
            "eon/neb-pt-lbfgs-m100",
            "neb-pt",
            ["1"],
            [f"min{i:02d}" for i in range(1, 14)],
            3600,
        ),
        ("EonDevelopDeLj38Ew", "eon/develop-de-lj38-ew", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 3600),
        ("EonDevelopDeLj38Oci", "eon/develop-de-lj38-oci", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 3600),
        ("EonDevelopDeLj38Cap", "eon/develop-de-lj38-cap-accept-energy", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 7200),
        ("EonDevelopDeLj38CapMatched", "eon/develop-de-lj38-cap", "de-lj38", ["1"], [f"pair_{i:02d}" for i in range(50)], 7200),
        ("EonDevelopDePtEw", "eon/develop-de-pt-ew", "de-pt", ["1"], [f"product_{i}" for i in range(59)], 3600),
        ("EonDevelopDePtOci", "eon/develop-de-pt-oci", "de-pt", ["1"], [f"product_{i}" for i in range(59)], 3600),
        ("EonDevelopDe2ts", "eon/develop-de-2ts-accept-energy", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonDevelopDe2tsMatched", "eon/develop-de-2ts", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonDevelopDe2tsOci", "eon/develop-de-2ts-oci", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonDevelopDe2tsEw", "eon/develop-de-2ts-ew", "de-2ts", ["1"], [f"pair_{i:02d}" for i in range(51)], 3600),
        ("EonDevelopLowLj38Dimer", "eon/develop-low-lj38-dimer", "lowest", ["1"], lj, 1800),
        ("EonDevelopLowLj38Lanczos", "eon/develop-low-lj38-lanczos", "lowest", ["1"], lj, 1800),
        # Asv bound for one repetition, not the problem cap. pass_limit is
        # what stops a search that has not reached 27 saddles.
        (
            "EonDevelopSaddlePtDimer",
            "eon/develop-saddle-pt-dimer",
            "akmc",
            [str(seed) for seed in range(1000, 1010)],
            ["pt-island"],
            21600,
        ),
        (
            "EonDevelopSaddlePtLanczos",
            "eon/develop-saddle-pt-lanczos",
            "akmc",
            ["1000", "1001"],
            ["pt-island"],
            21600,
        ),
    ]
    specs = []
    for cls, template_name, kind, seeds, starts, timeout in rows:
        template = load_template(ROOT, template_name)
        spec = {
            "cls": cls,
            "template": template_name,
            "kind": kind,
            "campaign_seeds": list(seeds),
            "campaign_starts": list(starts),
            "seeds": selected(seeds),
            "starts": selected(starts),
            "timeout": timeout,
            "cap": int(template["failure_cap"]),
            "potential": template["ini"]["Potential"]["potential"],
            "workers": workers if kind == "akmc" else 1,
        }
        if kind == "akmc":
            spec["passes"] = _akmc_passes(int(template.get("pass_limit", 1)))
        if not spec["seeds"] or not spec["starts"]:
            continue
        specs.append(spec)
    return specs


for _spec in _specs():
    _cls = _bind(_spec)
    globals()[_spec["cls"]] = _cls
