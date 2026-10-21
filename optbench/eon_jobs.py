"""eOn jobs whose starts are not one minimization geometry.

Minimization uses :func:`optbench.campaign.run`. The jobs here follow the
archived ``report.sh`` rule. A ``total_force_calls`` or ``force_calls_neb``
value under the cap stays in the mean. ``nfailed`` counts a
``termination_reason`` other than 0, including a missing ``results.dat``.
The Pt-heptamer ``force_calls`` value is the band counter across every
image. ``force_calls_per_image`` divides that total by the image count.
The archived ``report.sh`` still prints the per-image quotient.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .campaign import (
    benchmark_text,
    client_result,
    format_ini,
    geometry_archive,
    load_template,
    write_starts,
)


def report_summary(rows: list[dict], cap: int, images: int = 1) -> dict:
    """Reduce one archived report.

    ``rows`` carry ``raw`` (the counter named by the report) and ``reason``.
    Values at or above ``cap`` stay out of the mean. ``cap`` is that same
    counter, a band total when the row is a NEB. ``force_calls`` is the
    mean of the kept totals. When ``images`` is greater than one,
    ``force_calls_per_image`` is the mean of those totals divided by the
    image count.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    if images <= 0:
        raise ValueError("images must be positive")
    failed = 0
    kept: list[float] = []
    for row in rows:
        raw = row.get("raw")
        reason = row.get("reason")
        if raw is None or (reason is not None and reason != "0"):
            failed += 1
        if raw is not None and float(raw) < cap:
            kept.append(float(raw))
    if not kept:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "call_unit": "total",
            "nfailed": failed,
        }
    summary = {
        "force_calls": round(sum(kept) / len(kept)),
        "force_calls_min": int(min(kept)),
        "force_calls_max": int(max(kept)),
        "force_calls_median": statistics.median(kept),
        "call_unit": "total",
        "nfailed": failed,
    }
    if images > 1:
        scaled = [value / images for value in kept]
        summary["force_calls_per_image"] = round(sum(scaled) / len(scaled))
        summary["force_calls_per_image_min"] = round(min(scaled))
        summary["force_calls_per_image_max"] = round(max(scaled))
        summary["force_calls_per_image_median"] = statistics.median(scaled)
    return summary


def _read_dat(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in found:
            found[parts[1]] = parts[0]
    return found


def _client_env(client: str) -> dict[str, str]:
    env = os.environ.copy()
    binary = Path(client).resolve()
    lib = binary.parent.parent / "lib"
    env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _execute(spec: tuple) -> dict:
    client, ini, work, copies, texts, count_key = spec
    dest = Path(work)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    for src, name in copies:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)
    for name, text in texts:
        (dest / name).write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [client],
        cwd=dest,
        env=_client_env(client),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0 or not (dest / "results.dat").is_file():
        (dest / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    fields = _read_dat(dest / "results.dat")
    base = client_result(dest)
    raw_text = fields.get(count_key)
    raw = int(float(raw_text)) if raw_text is not None else None
    extrema_text = fields.get("number_of_extrema")
    extrema = int(float(extrema_text)) if extrema_text is not None else None
    return {
        "geometry": dest.name,
        "force_calls": base["force_calls"],
        "raw": raw,
        "reason": base["reason"] if fields else None,
        "count_key": count_key,
        "extrema": extrema,
    }


def _run_specs(
    client: Path,
    ini: str,
    specs: list[tuple],
    workers: int,
) -> list[dict]:
    jobs = [
        (str(client), ini, work, copies, texts, count_key)
        for work, copies, texts, count_key in specs
    ]
    if workers <= 1:
        return [_execute(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_execute, jobs))


def _stored(root: Path, archive_path: str) -> Path:
    return geometry_archive(root, {"id": archive_path, "input": {"archive_path": archive_path}})


def _extract(archive: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    return dest


def _coords(text: str) -> list[tuple[float, float, float]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("coordinates"))
    points: list[tuple[float, float, float]] = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            if points:
                break
            continue
        try:
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            if points:
                break
    if not points:
        raise ValueError("no coordinates")
    return points


def _fmt_mode(rows: list[tuple[float, float, float]]) -> str:
    return "".join(f"{x:.16e} {y:.16e} {z:.16e}\n" for x, y, z in rows)


def _seeded_mode(natoms: int, seed: int) -> str:
    """Python ``random.Random(seed)``, three draws in [0, 1) per atom."""
    rng = random.Random(seed)
    rows = [(rng.random(), rng.random(), rng.random()) for _ in range(natoms)]
    return _fmt_mode(rows)


def _difference_mode(reactant: Path, displacement: Path) -> str:
    left = _coords(reactant.read_text(encoding="utf-8", errors="replace"))
    right = _coords(displacement.read_text(encoding="utf-8", errors="replace"))
    if len(left) != len(right):
        raise ValueError(f"{displacement} has {len(right)} atoms, reactant has {len(left)}")
    rows = [(b[0] - a[0], b[1] - a[1], b[2] - a[2]) for a, b in zip(left, right)]
    norm = sum(x * x + y * y + z * z for x, y, z in rows) ** 0.5
    if norm == 0:
        raise ValueError(f"{displacement} matches the reactant")
    return _fmt_mode([(x / norm, y / norm, z / norm) for x, y, z in rows])


def _finish(
    template: dict,
    rows: list[dict],
    output: Path,
    images: int,
    extras: dict | None = None,
) -> dict:
    cap = int(template["failure_cap"])
    summary = report_summary(rows, cap, images)
    if extras:
        summary.update(extras)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = benchmark_text(summary, template)
    extra_lines = []
    for key in ("transition_states", "extrema_total"):
        if summary.get(key) is not None:
            extra_lines.append(f"{key} {summary[key]}")
    if extra_lines:
        body = body.rstrip("\n") + "\n" + "\n".join(extra_lines) + "\n"
    output.write_text(body, encoding="utf-8")
    start_rows = []
    for row in rows:
        raw = row["raw"]
        reason = row["reason"]
        bad = raw is None or (reason is not None and reason != "0") or (raw is not None and raw >= cap)
        item = {
            "failed": bad,
            "force_calls": None if raw is None else raw,
            "call_unit": "total",
            "geometry": row["geometry"],
            "raw": raw,
            "reason": reason,
            "count_key": row.get("count_key"),
        }
        if images > 1 and raw is not None:
            item["force_calls_per_image"] = raw / images
        for key in ("extrema", "overlap", "k", "client_reason", "last_force_calls", "seconds", "seed"):
            if key in row and row[key] is not None:
                item[key] = row[key]
        start_rows.append(item)
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["starts"] = len(rows)
    summary["template"] = template["id"]
    summary["problem"] = template.get("problem")
    return summary


def _template(root: Path, name: str) -> tuple[dict, str]:
    template = load_template(root, name)
    return template, format_ini(template["ini"])


def run_neb_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    images = int(template["ini"]["Nudged Elastic Band"]["images"])
    unpacked = _extract(_stored(root, "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz"), work / "unpacked")
    minima = unpacked / "minima"
    specs = []
    for src in sorted(path for path in minima.iterdir() if path.is_dir() and path.name.startswith("min")):
        specs.append(
            (
                str(work / "runs" / src.name),
                [(str(src / "a.con"), "reactant.con"), (str(src / "b.con"), "product.con")],
                [],
                "force_calls_neb",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, images)


def run_de_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(
        _stored(root, "website/double-ended-saddle-search/pt-island/pt-island-con.tgz"),
        work / "unpacked",
    )
    folder = unpacked / "pt-island-con"
    specs = []
    for index in range(59):
        product = folder / f"product_{index}.con"
        specs.append(
            (
                str(work / "runs" / f"product_{index}"),
                [(str(folder / "reactant.con"), "reactant.con"), (str(product), "product.con")],
                [],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_de_lj38(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(
        _stored(root, "website/double-ended-saddle-search/lj38/coords-con.tgz"),
        work / "unpacked",
    )
    folder = unpacked / "coords-con"
    lammps = root / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
    specs = []
    for index in range(50):
        specs.append(
            (
                str(work / "runs" / f"pair_{index:02d}"),
                [
                    (str(folder / f"start_{index:02d}.con"), "reactant.con"),
                    (str(folder / f"end_{index:02d}.con"), "product.con"),
                    (str(lammps), "in.lammps"),
                ],
                [],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_ts_lj38(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(_stored(root, "website/tsopt/LJ38/coords-con.tar.gz"), work / "unpacked")
    folder = unpacked / "coords-con"
    specs = []
    for index in range(200):
        con = folder / f"{index:04d}.con"
        text = con.read_text(encoding="utf-8", errors="replace")
        mode = _seeded_mode(len(_coords(text)), index)
        specs.append(
            (
                str(work / "runs" / f"{index:04d}"),
                [(str(con), "pos.con"), (str(con), "displacement.con")],
                [("direction.dat", mode)],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_ts_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(_stored(root, "website/tsopt/pt-island/pt-island-con.tgz"), work / "unpacked")
    folder = unpacked / "pt-island-con"
    reactant = folder / "reactant.con"
    specs = []
    for index in range(49):
        initial = folder / f"initial_{index}.con"
        mode = _difference_mode(reactant, initial)
        specs.append(
            (
                str(work / "runs" / f"initial_{index}"),
                [(str(reactant), "pos.con"), (str(initial), "displacement.con")],
                [("direction.dat", mode)],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def _pair_count(rows: list[dict]) -> dict:
    """Pairs with at least two band extrema, and the sum of that counter.

    ``number_of_extrema`` is the climbing-image count written by the
    client. The 2014 optim and pele ``transition_states`` values are a
    different code's statistic. This row does not reuse those numbers.
    """
    found = 0
    total = 0
    for row in rows:
        extrema = row.get("extrema")
        if extrema is None:
            continue
        total += int(extrema)
        if int(extrema) >= 2:
            found += 1
    return {"transition_states": found, "extrema_total": total}


def run_de_2ts(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    """All 51 xyz pairs, 0 through 50, converted to free-cluster con files."""
    from .eon_parse import xyz_to_con

    template, ini = _template(root, template_name)
    unpacked = _extract(
        _stored(root, "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz"),
        work / "unpacked",
    )
    folder = unpacked / "coords"
    specs = []
    for index in range(51):
        start = folder / f"start_{index}.xyz"
        end = folder / f"end_{index}.xyz"
        if not start.is_file() or not end.is_file():
            raise FileNotFoundError(f"missing pair {index} in {folder}")
        specs.append(
            (
                str(work / "runs" / f"pair_{index:02d}"),
                [],
                [
                    ("reactant.con", xyz_to_con(start.read_text(encoding="utf-8"))),
                    ("product.con", xyz_to_con(end.read_text(encoding="utf-8"))),
                ],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1, _pair_count(rows))


def _numpy_direction() -> str:
    """The archived ``random_mode.py`` vector. The seed is not the geometry index."""
    import io
    import sys

    try:
        import numpy as np
    except ImportError:
        bundled = Path.home() / "opt" / "eon" / "py"
        sys.path.insert(0, str(bundled))
        import numpy as np

    np.random.seed(314159)
    mode = np.random.random((38, 3))
    mode /= np.linalg.norm(mode)
    buffer = io.StringIO()
    np.savetxt(buffer, mode)
    return buffer.getvalue()


def _lowest_once(client: str, ini: str, work: str, con: str, direction: str) -> dict:
    dest = Path(work)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    shutil.copy(con, dest / "pos.con")
    shutil.copy(con, dest / "displacement.con")
    (dest / "direction.dat").write_text(direction, encoding="utf-8")
    proc = subprocess.run(
        [client],
        cwd=dest,
        env=_client_env(client),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        (dest / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    fields = _read_dat(dest / "results.dat")
    raw_text = fields.get("total_force_calls")
    raw = int(float(raw_text)) if raw_text is not None else None
    mode_path = dest / "mode.dat"
    mode_text = mode_path.read_text(encoding="utf-8", errors="replace") if mode_path.is_file() else ""
    return {
        "raw": raw,
        "reason": fields.get("termination_reason"),
        "mode_text": mode_text,
    }


def _lowest_job(spec: tuple) -> dict:
    from .eon_parse import abs_overlap, flat_floats

    client, sections, work, con, reference_text, direction, method, limit, threshold = spec
    reference = flat_floats(reference_text)
    base = json.loads(sections)

    def evaluate(k: int):
        sections_k = json.loads(sections)
        if method == "dimer":
            sections_k.setdefault("Dimer", {})["rotations_max"] = k
        elif method == "lanczos":
            sections_k.setdefault("Lanczos", {})["max_iterations"] = k
        else:
            raise ValueError(f"unknown lowest mode {method}")
        found = _lowest_once(client, format_ini(sections_k), work, con, direction)
        overlap = None
        if found["mode_text"]:
            try:
                overlap = abs_overlap(flat_floats(found["mode_text"]), reference)
            except ValueError:
                overlap = None
        ok = overlap is not None and overlap >= threshold
        return ok, {"raw": found["raw"], "reason": found["reason"], "overlap": overlap, "k": k}

    # Walk k = 1, 2, ... and stop at the first overlap. That is the
    # archived protocol. A doubling search can skip the smallest k when
    # the overlap is not monotonic.
    del base
    k = None
    payload = None
    for attempt in range(1, limit + 1):
        ok_k, payload = evaluate(attempt)
        if ok_k:
            k = attempt
            break
    ok = k is not None
    last_raw = None if payload is None else payload.get("raw")
    return {
        "geometry": Path(work).name,
        "force_calls": last_raw if ok else None,
        "raw": last_raw if ok else None,
        "last_force_calls": last_raw,
        "reason": "0" if ok else "no-overlap",
        "client_reason": None if payload is None else payload.get("reason"),
        "count_key": "total_force_calls",
        "overlap": None if payload is None else payload.get("overlap"),
        "k": k,
    }


def run_lowest(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    """Smallest iteration count whose mode overlaps the reference by 0.99.

    ``saddle_search`` ``max_iterations`` is 0, so the client stops after
    the mode. A termination reason of 5 is the climb limit, not a failed
    mode. ``nfailed`` counts geometries that never reach the overlap.
    The recorded calls are that successful run's ``total_force_calls``.
    """
    template = load_template(root, template_name)
    method = template["ini"]["Saddle Search"]["min_mode_method"]
    limit = int(template.get("iteration_limit", 1000))
    threshold = float(template.get("overlap", 0.99))
    unpacked = _extract(
        _stored(root, "website/lowest-eigenvector/lj38/coords-con.tar.gz"),
        work / "unpacked",
    )
    folder = unpacked / "coords-con"
    direction = _numpy_direction()
    sections = json.dumps(template["ini"])
    specs = []
    for index in range(200):
        con = folder / f"{index:04d}.con"
        mode = folder / f"{index:04d}_mode"
        if not con.is_file() or not mode.is_file():
            raise FileNotFoundError(f"missing lowest-mode geometry {index}")
        specs.append(
            (
                str(client),
                sections,
                str(work / "runs" / f"{index:04d}"),
                str(con),
                mode.read_text(encoding="utf-8", errors="replace"),
                direction,
                method,
                limit,
                threshold,
            )
        )
    if workers <= 1:
        rows = [_lowest_job(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_lowest_job, specs))
    return _finish(template, rows, output, 1)


JOBS = {
    "neb-pt-lbfgs": ("eon/neb-pt-lbfgs", run_neb_pt),
    "neb-pt-lbfgs-m100": ("eon/neb-pt-lbfgs-m100", run_neb_pt),
    "de-pt-lbfgs": ("eon/de-pt-lbfgs", run_de_pt),
    "de-pt-lbfgs-m100": ("eon/de-pt-lbfgs-m100", run_de_pt),
    "de-lj38-lbfgs": ("eon/de-lj38-lbfgs", run_de_lj38),
    "de-lj38-lbfgs-m100": ("eon/de-lj38-lbfgs-m100", run_de_lj38),
    "ts-lj38-dimer": ("eon/ts-lj38-dimer", run_ts_lj38),
    "ts-lj38-dimer-m100": ("eon/ts-lj38-dimer-m100", run_ts_lj38),
    "ts-pt-dimer": ("eon/ts-pt-dimer", run_ts_pt),
    "ts-pt-dimer-m100": ("eon/ts-pt-dimer-m100", run_ts_pt),
    "de-pt-ci-single": ("eon/develop-de-pt-ci-single", run_de_pt),
    "de-pt-ci": ("eon/develop-de-pt-ci", run_de_pt),
    "de-pt-lanczos-improved": ("eon/develop-de-pt-lanczos-improved", run_de_pt),
    "de-lj38-lanczos-improved": ("eon/develop-de-lj38-lanczos-improved", run_de_lj38),
    "de-2ts-lanczos-improved": ("eon/develop-de-2ts-lanczos-improved", run_de_2ts),
    "ts-lj38-lanczos-improved": ("eon/develop-ts-lj38-lanczos-improved", run_ts_lj38),
    "ts-pt-lanczos-improved": ("eon/develop-ts-pt-lanczos-improved", run_ts_pt),
    "neb-pt-ci-lbfgs": ("eon/develop-neb-pt-lbfgs", run_neb_pt),
    "neb-pt-ci-cg": ("eon/develop-neb-pt-cg", run_neb_pt),
    "low-lj38-dimer": ("eon/develop-low-lj38-dimer", run_lowest),
    "low-lj38-lanczos": ("eon/develop-low-lj38-lanczos", run_lowest),
    "saddle-pt-dimer": ("eon/develop-saddle-pt-dimer", None),
    "saddle-pt-lanczos": ("eon/develop-saddle-pt-lanczos", None),
    "aladatom-akmc": ("eon/aladatom-akmc", "amd"),
    "standard-aladatom-akmc": ("eon/standard-aladatom-akmc", "amd"),
}


def run_named(
    root: Path,
    name: str,
    client: Path,
    output: Path,
    *,
    workers: int = 1,
    work: Path | None = None,
    repetitions: int | None = None,
) -> dict:
    if name not in JOBS:
        known = ", ".join(sorted(JOBS))
        raise SystemExit(f"unknown eon job {name}. known: {known}")
    template_name, fn = JOBS[name]
    scratch = work or output.parent / "work" / name
    if fn is None:
        from .eon_akmc import run_saddle

        fn = run_saddle
    elif fn == "amd":
        from .eon_amd import run_amd

        fn = run_amd
    if repetitions is None:
        summary = fn(root, template_name, client, output, scratch, workers)
    else:
        summary = fn(root, template_name, client, output, scratch, workers, repetitions)
    print(
        f"{summary['problem']} {summary['template']} "
        f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
        f"starts={summary['starts']}"
    )
    return summary


def result_document(template: dict, summary: dict, starts_path: Path, host: str, date: str) -> dict:
    """One optbench.result/v1 document for a finished job."""
    rows = [json.loads(line) for line in starts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = {
        key: summary[key]
        for key in ("force_calls", "force_calls_min", "force_calls_max", "force_calls_median", "nfailed")
        if summary.get(key) is not None
    }
    return {
        "schema": "optbench.result/v1",
        "status": "sota",
        "problem": template.get("problem"),
        "submitter": "Rohit Goswami",
        "date": date,
        "host": host,
        "runs": [
            {
                "entry": starts_path.with_suffix("").name,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "present_source": True,
                "starts": len(rows),
                "nfailed": int(summary.get("nfailed") or 0),
                "method": {
                    "client": "eonclient",
                    "reducer": template.get("reducer"),
                    "ini": template.get("ini"),
                    "note": template.get("note"),
                },
                "metrics": metrics,
                "per_start": rows,
                "notes": template.get("note", ""),
            }
        ],
    }
