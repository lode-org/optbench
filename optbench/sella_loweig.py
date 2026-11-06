"""Lowest-eigenvector campaign for Sella 0.1.1.

Archive ``lowest-eigenvector/lj38/sella/lowest-eigv.py`` calls
``PESWrapper.diag(eta, 1e-16, vref=..., vreftol=0.99, shift=False)``
with ``eta`` 1e-6, a gradient initial mode, and projection of the rigid
modes. The score is ``PESWrapper.calls``. The potential is ASE
Lennard-Jones, epsilon 1, sigma 1, cutoff 40.

Sella 0.1.1 is not the harness Sella 2.6.0. Point ``SELLA_011`` at a
built tree of that release. NumPy 2 renamed ``np.infty`` and SciPy's
``LinearOperator`` now requires an array namespace; those two shims
keep the 0.1.1 objects callable. They do not change the diagonalization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETA = 1e-6
GAMMA = 1e-16
VREFTOL = 0.99


def _sella_root() -> Path:
    env = os.environ.get("SELLA_011", "")
    path = Path(env) if env else Path("/tmp/sella011/Sella-0.1.1")
    if not (path / "sella" / "peswrapper.py").is_file():
        raise FileNotFoundError(f"Sella 0.1.1 tree is not at {path}")
    return path


def _shim() -> None:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator

    np.infty = np.inf
    if not hasattr(np, "int"):
        np.int = int
    LinearOperator._check_matching_namespace = lambda self, x: None
    original = LinearOperator.dot

    def dot(self, x):
        if not hasattr(self, "_xp"):
            self._xp = np
        return original(self, x)

    LinearOperator.dot = dot


def _prepare_import() -> None:
    root = str(_sella_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    # A process that already imported Sella 2.6.0 must not reuse it.
    loaded = sys.modules.get("sella")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(root):
        raise RuntimeError("Sella 2.6.0 is already imported; run this in a fresh process")
    _shim()


def run_index(coords: Path, index: int) -> int:
    """Force calls for one archived structure, using the 2019 script."""
    import numpy as np
    from ase.calculators.lj import LennardJones
    from ase.io import read

    _prepare_import()
    from sella import PESWrapper

    prefix = f"{index:04d}"
    atoms = read(coords / f"{prefix}.con")
    atoms.pbc = False
    reference = []
    with (coords / f"{prefix}_mode").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            reference.append(float(line.strip()))
    vector = np.asarray(reference, dtype=float)
    vector /= np.linalg.norm(vector)
    saved = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        pes = PESWrapper(atoms, LennardJones(epsilon=1.0, sigma=1.0, rc=40.0), v0=None)
        pes.diag(ETA, GAMMA, vref=vector, vreftol=VREFTOL, shift=False)
    finally:
        sys.stdout.close()
        sys.stdout = saved
    return int(pes.calls)


def _job(spec: tuple[str, int]) -> tuple[int, int]:
    coords, index = spec
    return index, run_index(Path(coords), index)


def run_all(coords: Path, workers: int = 1) -> list[int]:
    specs = [(str(coords), index) for index in range(200)]
    if workers <= 1:
        return [run_index(coords, index) for index in range(200)]
    import multiprocessing as mp

    context = mp.get_context("spawn")
    with context.Pool(workers) as pool:
        rows = pool.map(_job, specs)
    rows.sort()
    return [calls for _index, calls in rows]


def write_dat(calls: list[int], output: Path) -> dict:
    from optbench.score import half_even

    count = len(calls)
    mean = half_even(sum(calls) / count) if count else None
    # The published file keeps one decimal. half_even of a .x5 may be an int.
    raw = sum(calls) / count
    text = [
        f"force_calls {raw:.6g}",
        f"force_calls_min {min(calls)}",
        f"force_calls_max {max(calls)}",
        "nfailed 0",
        f"starts {count}",
        "code Sella",
        "code_version 0.1.1",
        "commit 0.1.1",
        "algorithm lowest eigenvector",
        "template sella-0.1.1/lj38-loweig",
        "problem lowest-eigenvector/lj38",
        "counter harness",
        "reducer all",
        "parameter_source archive",
        "note Archive lowest-eigv.py: PESWrapper.diag eta 1e-6, gamma 1e-16, vreftol 0.99, gradient v0, projection. ASE Lennard-Jones cutoff 40. One harness force call is one PESWrapper.calls evaluation.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    starts = output.with_name(output.stem + ".starts.jsonl")
    import json

    lines = []
    for index, value in enumerate(calls):
        lines.append(
            json.dumps(
                {
                    "schema": "optbench.start/v2",
                    "start": f"{index:04d}",
                    "seed": 0,
                    "force_calls": value,
                    "success": True,
                    "counter": "harness",
                    "comparable": True,
                },
                sort_keys=True,
            )
        )
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"force_calls": raw, "rounded": mean, "min": min(calls), "max": max(calls), "n": count}


def write_asv(commit: str) -> Path:
    """asv tracks for the finished exact-setup campaigns on this branch."""
    import json
    import time

    from asv.results import Results

    from optbench.asv_machine import machine_dir, machine_record
    from optbench.failures import failure_of

    params = {**machine_record(), "python": "3.12"}
    result = Results(params, {}, commit, int(time.time() * 1000), "3.12", "uv-pip-py3.12", {})
    tracks = (
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    )
    jobs = (
        {
            "name": "bench_sella_loweig.Lj38Loweig",
            "path": ROOT / "results" / "campaign" / "sella-0.1.1-lj38-loweig.starts.jsonl",
            "version": "0.1.1",
            "cap": 10000,
        },
        {
            "name": "bench_ase_sella.AseLj38LbfgslinesearchTuned",
            "path": ROOT / "results" / "campaign" / "ase-3.29.0-lj38-lbfgslinesearch-tuned.starts.jsonl",
            "version": "3.29.0",
            "cap": 10000,
        },
        {
            "name": "bench_ase_sella.AseLj38PreconlbfgsTuned",
            "path": ROOT / "results" / "campaign" / "ase-3.29.0-lj38-preconlbfgs-tuned.starts.jsonl",
            "version": "3.29.0",
            "cap": 10000,
        },
    )
    wrote = 0
    for job in jobs:
        path = job["path"]
        if not path.is_file():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            continue
        starts = [str(row["start"]) for row in rows]
        param_lists = [[repr("0")], [repr(start) for start in starts]]
        columns = {method: [] for method in tracks}
        for row in rows:
            success = bool(row.get("success"))
            failure = None if success else failure_of(row, int(job["cap"]))
            columns["track_force_calls"].append(float(row["force_calls"]))
            columns["track_success"].append(1.0 if success else 0.0)
            columns["track_failure_cap"].append(1.0 if failure == "cap" else 0.0)
            columns["track_failure_wrong_stationary"].append(1.0 if failure == "wrong_stationary" else 0.0)
            columns["track_failure_divergence"].append(1.0 if failure == "divergence" else 0.0)
            columns["track_failure_crash"].append(1.0 if failure == "crash" else 0.0)
        for method in tracks:
            name = f"{job['name']}.{method}"
            result._results[name] = columns[method]
            result._benchmark_params[name] = param_lists
            result._benchmark_version[name] = job["version"]
            result._started_at[name] = int(time.time() * 1000)
            result._duration[name] = 0.0
            result._samples[name] = None
            result._stats[name] = None
        wrote += 1
    if wrote == 0:
        raise RuntimeError("no finished campaign to store")
    folder = machine_dir(ROOT / "results" / "asv")
    result.save(str(ROOT / "results" / "asv"))
    dest = folder / result._filename
    print(f"wrote {dest} classes {wrote}", flush=True)
    return dest


def main() -> None:
    coords = ROOT / "inputs" / "sha256" / "d24b26d791fd02fb1b67025195c2694330567359cb09c36ec1cedb9d9edf49ae"
    # The stored object is the tarball. Extract beside the campaign work dir.
    archive = None
    manifest = ROOT / "inputs" / "manifest.jsonl"
    import json

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == "website/lowest-eigenvector/lj38/sella/coords-con.tar.gz":
            archive = ROOT / row["stored"]
            break
    if archive is None or not archive.is_file():
        raise FileNotFoundError("coords-con.tar.gz is not in inputs/manifest.jsonl")
    import tarfile

    dest = Path("/tmp/optbench-sella-loweig")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest)
    coords_dir = dest / "coords-con"
    workers = int(os.environ.get("WORKERS", "8"))
    calls = run_all(coords_dir, workers=workers)
    output = ROOT / "results" / "campaign" / "sella-0.1.1-lj38-loweig.dat"
    summary = write_dat(calls, output)
    print(summary)


if __name__ == "__main__":
    main()
