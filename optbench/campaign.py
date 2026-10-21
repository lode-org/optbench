"""One command from a problem file and a template to benchmark.dat.

The template is the client configuration. The problem file is the stop
and the geometries. The starts stay in this loop. The record is one
benchmark.dat.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tarfile
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .problems import load_problem
from .score import summarize_capped
from .v2 import client_start_records

_INI_ORDER = ("Main", "Potential", "Optimizer")


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_ini(sections: dict) -> str:
    names = [name for name in _INI_ORDER if name in sections]
    names.extend(sorted(name for name in sections if name not in _INI_ORDER))
    blocks = []
    for name in names:
        lines = [f"[{name}]"]
        for key, value in sections[name].items():
            lines.append(f"{key} = {format_value(value)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def load_template(root: Path, name: str) -> dict:
    path = Path(name)
    if not path.is_file():
        path = root / "templates" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no template {name}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "id" not in data or "engine" not in data:
        raise ValueError(f"{path} needs id and engine")
    if data["engine"] == "eon-client" and "ini" not in data:
        raise ValueError(f"{path} needs ini")
    data["_path"] = path
    return data


def list_templates(root: Path) -> list[dict]:
    found = []
    for path in sorted((root / "templates").rglob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        found.append(data)
    return found


def resolve_problem(root: Path, name: str) -> dict:
    path = Path(name)
    if path.is_file():
        return load_problem(path)
    candidate = root / "problems" / f"{name}.toml"
    if not candidate.is_file():
        raise FileNotFoundError(f"no problem {name}")
    return load_problem(candidate)


def geometry_archive(root: Path, problem: dict) -> Path:
    archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        raise ValueError(f"{problem['id']} has no input.archive_path")
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(f"{archive_path} is not in the geometry manifest")


def iter_geometries(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(path.glob("*.con"))
    elif tarfile.is_tarfile(path):
        raise ValueError("pass a directory of .con files, or use the problem archive through run()")
    else:
        found = [path]
    if not found:
        raise FileNotFoundError(f"no .con files in {path}")
    return found


_EON_ENGINES = {"eon-client", "eon"}


def _check_stop(problem: dict, template: dict) -> int | None:
    """Return the force-call cap, or None when the problem does not state one.

    A template ``failure_cap`` fills that gap for an eOn run. It is the cap
    the archived report used when the problem file left the number out.
    ASE and Sella templates carry the same number on ``stop``.
    ``max_iterations`` is checked only against a cap the problem itself states.
    """
    convergence = problem.get("convergence") or {}
    optimizer = (template.get("ini") or {}).get("Optimizer") or {}
    stop = template.get("stop") or {}
    if "force_norm" in convergence and "force_norm" in stop:
        if abs(float(stop["force_norm"]) - float(convergence["force_norm"])) > 1e-12:
            raise ValueError("template force_norm is not the problem force norm")
    if "gradient_norm" in convergence and "gradient_norm" in stop:
        if abs(float(stop["gradient_norm"]) - float(convergence["gradient_norm"])) > 1e-12:
            raise ValueError("template gradient_norm is not the problem gradient norm")
    stated = convergence.get("max_force_calls")
    cap = int(stated) if stated is not None else None
    # A minimizer step is one force call, so its iteration limit cannot sit
    # under the force budget. A band step, a dimer rotation, and a saddle
    # translation are not one force call, and their iteration limits can.
    job = str(((template.get("ini") or {}).get("Main") or {}).get("job") or "")
    step_is_force_call = job == "minimization" or (
        job == "" and problem.get("suite") == "minimization" and "band" not in problem
    )
    if (
        cap is not None
        and step_is_force_call
        and "max_iterations" in optimizer
        and int(optimizer["max_iterations"]) < cap
    ):
        raise ValueError("template max_iterations is below the problem force-call cap")
    if "force_norm" in convergence and "converged_force" in optimizer:
        if abs(float(optimizer["converged_force"]) - float(convergence["force_norm"])) > 1e-12:
            raise ValueError("template converged_force is not the problem force norm")
    if cap is not None and "max_force_calls" in stop and int(stop["max_force_calls"]) < cap:
        raise ValueError("template max_force_calls is below the problem force-call cap")
    if cap is None and template.get("failure_cap") is not None:
        cap = int(template["failure_cap"])
    if cap is None:
        raw = stop.get("max_force_calls", stop.get("max_steps"))
        if raw is not None:
            cap = int(raw)
    if cap is None and "max_iterations" in optimizer:
        cap = int(optimizer["max_iterations"])
    return cap


def parse_client_dat(text: str) -> tuple[int | None, str | None]:
    """Read the eon client results.dat. The value is the first field."""
    calls = None
    reason = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == "total_force_calls":
            calls = int(float(parts[0]))
        elif parts[1] == "termination_reason":
            reason = parts[0]
    return calls, reason


def _launch_client(client: str, ini: str, con: str, work: str) -> Path:
    dest = Path(work)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini)
    shutil.copy(con, dest / "pos.con")
    subprocess.run(
        [client],
        cwd=dest,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return dest


def client_result(work: Path) -> dict:
    """Read one results.dat. Missing output has no call count and no reason."""
    dat = Path(work) / "results.dat"
    if not dat.is_file():
        return {"force_calls": None, "reason": None}
    calls, reason = parse_client_dat(dat.read_text(encoding="utf-8", errors="replace"))
    return {"force_calls": calls, "reason": reason}


def _bad_termination(calls: int | None, reason: str | None) -> bool:
    return calls is None or (reason is not None and reason != "0")


def run_client(client: str, ini: str, con: str, work: str) -> int | None:
    """Run one eonclient subprocess on one geometry. A bad termination is None."""
    found = client_result(_launch_client(client, ini, con, work))
    if _bad_termination(found["force_calls"], found["reason"]):
        return None
    return found["force_calls"]


def _one(job: tuple[str, str, str, str]) -> dict:
    client, ini, con, work = job
    found = client_result(_launch_client(client, ini, con, work))
    found["geometry"] = Path(con).name
    return found


def benchmark_text(summary: dict, template: dict) -> str:
    lines = []
    for key in (
        "force_calls",
        "force_calls_min",
        "force_calls_max",
        "force_calls_median",
        "force_calls_per_image",
        "force_calls_per_image_min",
        "force_calls_per_image_max",
        "force_calls_per_image_median",
        "call_unit",
        "nfailed",
    ):
        value = summary.get(key)
        if value is None:
            continue
        lines.append(f"{key} {value}")
    if template.get("code"):
        lines.append(f"code {template['code']}")
    if template.get("code_version"):
        lines.append(f"code_version {template['code_version']}")
    if template.get("commit"):
        lines.append(f"commit {template['commit']}")
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    if template.get("problem"):
        lines.append(f"problem {template['problem']}")
    if template.get("reducer"):
        lines.append(f"reducer {template['reducer']}")
    return "\n".join(lines) + "\n"


def write_starts(path: Path, rows: list[dict]) -> None:
    """One JSON object per start, failures included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    client: Path | None = None,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    template = load_template(root, template_name)
    if template["engine"] == "noise":
        from .runners.noise_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] == "gmin":
        from .runners.gmin_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] == "pele":
        method = str((template.get("pele") or {}).get("method") or "")
        if method == "basinhopping":
            from .runners.pele_bh import run_template
        else:
            from .runners.pele_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] == "vcsqnm":
        from .runners.vcsqnm_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] == "airss":
        from .runners.airss_go import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
        )
    if template["engine"] == "rgmin":
        from .runners.rgmin_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
        )
    if template["engine"] == "softsaddle":
        from .softsaddle_run import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            output=output,
            workers=workers,
            work=work,
        )
    if template["engine"] == "rgsaddle":
        from .rgsaddle_minmode import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            output=output,
            workers=workers,
            work=work,
        )
    if template["engine"] == "optking":
        from .runners.optking_ts import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
            resume=resume,
        )
    if template["engine"] in {"ase", "sella"} or (
        template["engine"] == "geodesic" and template.get("backend") == "ase"
    ):
        from .ase_sella import run_template

        if template.get("problem") and template["problem"] != problem_name:
            raise ValueError(f"{template_name} belongs to {template['problem']}")
        return run_template(
            root,
            template_name,
            output=output,
            workers=workers,
            limit=limit,
            resume=resume,
        )
    from .engines import find as find_engine

    plugged = find_engine(str(template["engine"]))
    if plugged is not None:
        stated = template.get("problem")
        if stated and stated != problem_name:
            raise ValueError(f"{template_name} belongs to {stated}")
        return plugged.run(
            root,
            problem_name,
            template_name,
            output=output,
            workers=workers,
        )
    task = str(template.get("task") or "")
    reducer = str(template.get("reducer") or "")
    if template["engine"] in _EON_ENGINES and task == "minimize" and reducer.startswith("harness"):
        from .runners.eon_min import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] in _EON_ENGINES and task == "htst" and reducer.startswith("harness"):
        from .runners.eon_htst import run_template

        problem = resolve_problem(root, problem_name)
        stated = template.get("problem")
        if stated and stated != problem["id"]:
            raise ValueError(f"{template_name} belongs to {stated}")
        return run_template(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    if template["engine"] in _EON_ENGINES and task == "htst-written":
        paired = template.get("paired") or "the barrier template"
        raise RuntimeError(f"{template['id']} is written by {paired}")
    if template["engine"] not in _EON_ENGINES:
        raise ValueError(f"unknown engine {template['engine']}")
    if client is None:
        raise ValueError("eon-client run needs a client")
    problem = resolve_problem(root, problem_name)
    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    ini = format_ini(template["ini"])
    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = work or output.parent / "geometries"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        cons = sorted(dest.rglob("*.con"))
        if not cons:
            raise FileNotFoundError(f"{archive} has no .con files")
    else:
        cons = iter_geometries(geometries)
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    jobs = [
        (str(client), ini, str(con), str(scratch / con.stem))
        for con in cons
    ]
    if workers <= 1:
        details = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            details = list(pool.map(_one, jobs))
    # A missing result or a bad termination is a failure at the cap, so it
    # stays out of the mean. The starts file keeps the raw call count.
    counts = []
    start_rows = []
    for detail in details:
        calls = detail["force_calls"]
        reason = detail["reason"]
        bad = _bad_termination(calls, reason)
        counted = cap if bad else calls
        counts.append(counted)
        start_rows.append(
            {
                "failed": bad or calls >= cap,
                "force_calls": calls,
                "geometry": detail["geometry"],
                "reason": reason,
            }
        )
    summary = summarize_capped(counts, cap)
    ok = [count for count in counts if count < cap]
    required = (problem.get("metrics") or {}).get("required") or []
    if "force_calls_median" in required and ok:
        summary["force_calls_median"] = statistics.median(ok)
    summary["counts"] = counts
    summary["mean_unrounded"] = (sum(ok) / len(ok)) if ok else None
    summary["start_rows"] = start_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(benchmark_text(summary, template), encoding="utf-8")
    write_starts(
        output.with_suffix(".starts.jsonl"),
        client_start_records([con.stem for con in cons], counts, cap),
    )
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["starts"] = len(counts)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


