"""asv classes for the eOn templates whose campaign count is the client's.

Three campaign paths have no harness class. ``optbench run`` gives an
``eon`` or ``eon-client`` template without ``kind`` one client run per
archived geometry, through ``campaign.client_jobs``. ``optbench eon-job``
gives an ``eon_jobs.JOBS`` template one client run per minimum, pair,
product, or initial geometry, through ``eon_jobs.plan``. It gives an
``amd`` job one AKMC catalogue per repetition, through ``eon_amd.plan``.
A class calls the same function for its start and scores the run as the
campaign file did, with ``campaign.client_record``,
``eon_jobs.run_start``, or ``eon_amd.start_record``. The count is the
client's: ``results.dat``, or sad-fcs plus mins-fcs for a catalogue. It
is not a harness count.

One class is one template. The parameters are the seed and the start.
Seed ``0`` is the only value: no path draws from it, and a catalogue
takes the seed ``run_amd`` gives its repetition. The start is the
geometry name of the campaign file, without ``.con``, or ``rep-<n>`` for
a catalogue. The default grid is the first start. ``OPTBENCH_ASV_FULL=1``
at import selects every start.

A catalogue runs with as many workers as the template's Communicator
``num_jobs``. ``eon_amd`` writes that count of jobs per pass for any
smaller worker count, so the config is the one the campaign wrote.

The client is the build that ``clients`` gives the template prefix. A
template under ``eon/`` is that build only when its ``commit`` is the
``eon-main`` registry commit. The other ``eon/`` templates are develop
builds with their own classes. ``setup`` raises NotImplementedError when
the client is not installed, so asv skips the start. The search raises
RuntimeError when the client's version flag does not print the build's
token.

asv runs each track in its own process. ``time_search`` runs the search
and stores its record under ``OPTBENCH_ASV_CACHE``. A track reads that
record, or runs the search when there is none.

A client-path template without ``problem`` is a minimization/lj38 cell,
the problem its campaign ran on. A problem with a ``noise`` table is not
a client-path cell. The client adds no noise, and ``bench_eon_svn``
measures those templates through its harness.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path

from optbench.asv_engines import EngineSearch, cache_dir, camel, full_grid

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"
DEFAULT_PROBLEM = "minimization/lj38"
_NAMES: dict[tuple, list[str]] = {}


def clients() -> dict[str, dict]:
    """The client build of each template prefix."""
    from optbench.engines.common import registry_pin
    from optbench.eon_harness import FAMILIES

    commit = str(registry_pin(ROOT, "eon-main")["commit"])
    return {
        "eon-svn/": {**FAMILIES["svn"], "label": "svn"},
        "eon-client/": {**FAMILIES["client"], "label": "client"},
        "eon/": {
            "label": "eon-main",
            "env": "EON_MAIN_CLIENT",
            "default": Path.home() / "opt" / "eon-main" / "bin" / "eonclient",
            "flag": "--version",
            "token": commit[:8],
            "commit": commit,
        },
    }


def client_of(template: dict) -> dict | None:
    ident = str(template["id"])
    for prefix, spec in clients().items():
        if ident.startswith(prefix):
            if spec.get("commit") and template.get("commit") != spec["commit"]:
                return None
            return spec
    return None


def cells(prefix: str, skip: frozenset[str] = frozenset()) -> list[dict]:
    """Every client-counted cell of the templates under ``prefix``."""
    from optbench.campaign import _EON_ENGINES, list_templates, resolve_problem
    from optbench.eon_jobs import JOBS, SPECS

    jobs = {template: (name, fn) for name, (template, fn) in JOBS.items()}
    found = []
    for template in list_templates(ROOT):
        ident = str(template["id"])
        if not ident.startswith(prefix) or ident in skip or client_of(template) is None:
            continue
        name, fn = jobs.get(ident, (None, None))
        if fn in SPECS:
            cell = {"path": "job", "job": name, "problem_id": str(template["problem"])}
        elif fn == "amd":
            cell = {"path": "amd", "job": name, "problem_id": str(template["problem"])}
        elif template.get("engine") in _EON_ENGINES and not template.get("kind"):
            problem_id = str(template.get("problem") or DEFAULT_PROBLEM)
            if not problem_id.startswith("minimization/"):
                continue
            if resolve_problem(ROOT, problem_id).get("noise"):
                continue
            cell = {"path": "client", "problem_id": problem_id}
        else:
            continue
        cell["template_id"] = ident
        cell["class_name"] = camel(ident)
        found.append(cell)
    return found


def start_names(cell: dict) -> list[str]:
    """The campaign's geometry names for ``cell``, in campaign order."""
    if cell["path"] == "job":
        from optbench.eon_jobs import JOBS, plan, start_name

        key = ("job", JOBS[cell["job"]][1].__name__)
        if key not in _NAMES:
            work = Path(tempfile.mkdtemp(prefix="optbench-asv-eon-names-"))
            try:
                _NAMES[key] = [start_name(spec) for spec in plan(ROOT, cell["job"], work)["specs"]]
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return _NAMES[key]
    from optbench.campaign import geometry_archive, resolve_problem

    if cell["path"] == "amd":
        repetitions = int(resolve_problem(ROOT, cell["problem_id"])["sample"]["repetitions"])
        return [f"rep-{index + 1}" for index in range(repetitions)]
    key = ("client", cell["problem_id"])
    if key not in _NAMES:
        archive = geometry_archive(ROOT, resolve_problem(ROOT, cell["problem_id"]))
        with tarfile.open(archive) as packed:
            members = sorted(
                Path(member.name) for member in packed.getmembers() if member.isfile() and member.name.endswith(".con")
            )
        _NAMES[key] = [member.stem for member in members]
    return _NAMES[key]


def cap_of(cell: dict) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, cell["template_id"])
    if cell["path"] in ("job", "amd"):
        return int(template["failure_cap"])
    cap = _check_stop(resolve_problem(ROOT, cell["problem_id"]), template)
    if cap is None:
        raise ValueError(f"{cell['problem_id']} has no force-call cap")
    return int(cap)


def _key(cell: dict) -> str:
    from optbench.campaign import load_template, resolve_problem

    template = load_template(ROOT, cell["template_id"])
    parts = {
        "template": template,
        "problem": resolve_problem(ROOT, cell["problem_id"]),
        "token": client_of(template)["token"],
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def record_path(cell: dict, seed: str, start: str) -> Path:
    return (
        cache_dir()
        / "eon-client"
        / cell["template_id"].replace("/", "-")
        / _key(cell)
        / f"{seed}-{start}.json"
    )


def search(cell: dict, client: str, start: str, work: Path) -> dict:
    """The campaign's client run for ``start``, scored as the campaign scored it."""
    from optbench.campaign import _one, client_jobs, client_record, load_template, resolve_problem

    if cell["path"] == "job":
        from optbench.eon_jobs import plan, run_start

        return run_start(Path(client), plan(ROOT, cell["job"], work), start)
    if cell["path"] == "amd":
        from optbench.eon_amd import plan, run_repetition, start_record

        planned = plan(ROOT, cell["template_id"])
        if start not in start_names(cell):
            raise KeyError(f"{cell['template_id']} has no start {start}")
        workers = int(planned["template"]["ini"]["Communicator"]["num_jobs"])
        repetition = int(start.removeprefix("rep-"))
        row = run_repetition(ROOT, planned, Path(client), work / start, workers, repetition)
        return start_record(planned["template"], planned["problem"], row)
    template = load_template(ROOT, cell["template_id"])
    problem = resolve_problem(ROOT, cell["problem_id"])
    jobs, cap, cons = client_jobs(
        ROOT, problem, template, Path(client), geometries=None, output=work / "campaign.dat", work=work
    )
    picked = [job for job, con in zip(jobs, cons, strict=True) if con.stem == start]
    if len(picked) != 1:
        raise KeyError(f"{cell['template_id']} has no start {start}")
    return client_record(_one(picked[0]), cap)


def compute(cell: dict, seed: str, start: str) -> dict:
    """The search for ``start``, run once and stored."""
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the campaign seed {SEED}")
    from optbench.campaign import load_template
    from optbench.eon_harness import check_client

    spec = client_of(load_template(ROOT, cell["template_id"]))
    client = check_client(spec, spec["label"])
    path = record_path(cell, SEED, str(start))
    path.unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix="optbench-asv-eon-"))
    try:
        record = search(cell, client, str(start), work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    record.update(template=cell["template_id"], problem=cell["problem_id"], start=str(start))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)
    return record


def load(cell: dict, seed: str, start: str) -> dict:
    path = record_path(cell, SEED, str(start))
    if str(seed) == SEED and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return compute(cell, seed, start)


class ClientSearch(EngineSearch):
    """Tracks for one client run. ``cell`` is set by ``bind``."""

    def setup(self, seed, start):
        from optbench.campaign import load_template
        from optbench.eon_harness import client_file

        spec = client_of(load_template(ROOT, self.cell["template_id"]))
        if not client_file(spec):
            raise NotImplementedError(f"set {spec['env']} to the {spec['label']} eonclient")
        self.record = None

    def time_search(self, seed, start):
        self.record = compute(self.cell, seed, start)

    def _stored(self, seed, start) -> dict:
        if getattr(self, "record", None) is None:
            self.record = load(self.cell, seed, start)
        return self.record


def bind(cell: dict):
    names = start_names(cell)
    bound = dict(cell)

    class _Benchmark(ClientSearch):
        param_names = ["seed", "start"]
        params = [[SEED], names if full_grid() else names[:1]]
        timeout = math.inf
        cell = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        cap = cap_of(bound)

    _Benchmark.__name__ = cell["class_name"]
    _Benchmark.__qualname__ = cell["class_name"]
    return _Benchmark


def install_benchmarks(prefix: str, namespace: dict, skip: frozenset[str] = frozenset()) -> None:
    """Define one asv class per client-counted template under ``prefix`` in ``namespace``."""
    for cell in cells(prefix, skip):
        namespace[cell["class_name"]] = bind(cell)
