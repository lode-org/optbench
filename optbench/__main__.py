"""Command line for the OptBench replay tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cyclopts

app = cyclopts.App(
    name="optbench",
    help="Run one OptBench problem and write benchmark.dat.",
)


def _eon_client(client: Path | None) -> Path:
    if client is not None:
        return client
    found = shutil.which("eonclient")
    if found:
        return Path(found)
    env = os.environ.get("EONCLIENT")
    if env:
        return Path(env)
    raise SystemExit("pass --client, set EONCLIENT, or put eonclient on PATH")


@app.command
def check(
    problems: Path = Path("problems"),
    oracle: Path = Path("oracle/entries.jsonl"),
) -> None:
    """Compare the oracle with the problem files."""
    from optbench.check import check as run_check

    raise SystemExit(run_check(problems, oracle))


@app.command(name="import-archive")
def import_archive(
    bare: Path,
    dest: Path = Path("."),
) -> None:
    """Copy scores and geometries out of a bare clone of the public archive."""
    from optbench.import_archive import import_archive as copy_archive

    copy_archive(bare, dest)


@app.command
def templates(root: Path = Path(".")) -> None:
    """List run templates."""
    from optbench.campaign import list_templates

    for template in list_templates(root):
        print(template["id"])


@app.command
def run(
    problem: str,
    template: str,
    *,
    root: Path = Path("."),
    client: Path | None = None,
    geometries: Path | None = None,
    output: Path = Path("benchmark.dat"),
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> None:
    """Run one problem template and write benchmark.dat."""
    from optbench.campaign import load_template
    from optbench.campaign import run as run_campaign

    loaded = load_template(root, template)
    if loaded["engine"] == "rgsaddle-band" or (
        loaded["engine"] == "geodesic" and loaded.get("backend") == "rgsaddle"
    ):
        from optbench.rgsaddle_band import run as run_band

        summary = run_band(
            root,
            problem,
            template,
            output=output,
            workers=workers,
        )
    elif loaded["engine"] == "lammps-neb":
        from optbench.lammps_neb import run as run_lammps

        summary = run_lammps(
            root,
            problem,
            template,
            output=output,
            workers=workers,
            limit=limit,
            resume=resume,
        )
    else:
        client_path = _eon_client(client) if loaded["engine"] in {"eon-client", "eon"} else client
        summary = run_campaign(
            root,
            problem,
            template,
            client=client_path,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
            resume=resume,
        )
    calls = summary.get("force_calls")
    print(
        f"{summary['problem']} {summary['template']} "
        f"force_calls={calls} nfailed={summary['nfailed']} "
        f"starts={summary['starts']}"
    )


@app.command(name="eon-job")
def eon_job(
    name: str,
    *,
    root: Path = Path("."),
    client: Path | None = None,
    output: Path = Path("benchmark.dat"),
    workers: int = 1,
    work: Path | None = None,
    repetitions: int | None = None,
) -> None:
    """Run one named eOn band or saddle job and write benchmark.dat."""
    from optbench.eon_jobs import run_named

    run_named(
        root,
        name,
        _eon_client(client),
        output,
        workers=workers,
        work=work,
        repetitions=repetitions,
    )


@app.command
def neb(
    entry: Path,
    minima: Path,
    *,
    client: Path | None = None,
    work: Path,
    output: Path,
    workers: int = 1,
    code: str = "Eon",
    version: str = "3.5.0",
) -> None:
    """Run an archived minima-directory NEB entry."""
    from optbench.archive_eon import run_minima

    text = run_minima(
        entry,
        minima,
        _eon_client(client),
        work,
        output,
        workers=workers,
        code=code,
        version=version,
    )
    print(text, end="")


@app.command
def pairs(
    entry: Path,
    coords: Path,
    *,
    client: Path | None = None,
    work: Path,
    output: Path,
    workers: int = 1,
    count: int = 50,
    code: str = "Eon",
    version: str = "3.5.0",
) -> None:
    """Run an archived reactant/product saddle or band entry."""
    from optbench.archive_eon import run_pairs

    text = run_pairs(
        entry,
        coords,
        _eon_client(client),
        work,
        output,
        workers=workers,
        count=count,
        code=code,
        version=version,
    )
    print(text, end="")


def _head(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


@app.command(name="campaign-ase")
def campaign_ase(
    *,
    root: Path = Path("."),
    output_dir: Path = Path("results/campaign"),
    workers: int = 1,
    limit: int = 0,
    match: str = "",
    ranked_only: bool = True,
    resume: bool = False,
) -> None:
    """Run the ranked ASE and Sella templates."""
    from optbench.ase_sella import run_many

    summaries = run_many(
        root,
        output_dir,
        workers=workers,
        limit=limit,
        match=match,
        ranked_only=ranked_only,
        resume=resume,
    )
    for summary in summaries:
        print(
            f"{summary['problem']} {summary['template']} "
            f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
            f"starts={summary['starts']}"
        )


@app.command
def quadratic(
    output: Path = Path("results/quadratic/quadratic.json"),
    root: Path = Path("."),
    commit: str = "",
) -> None:
    """Run the quadratic v2 sample. This is not a campaign."""
    from optbench.quadratic import write_smoke

    document = write_smoke(output, commit=commit or _head(root))
    print(f"wrote {output} runs={len(document['runs'])}")


@app.command
def score(
    starts: Path,
    cap: int | None = None,
    output: Path = Path("score.json"),
) -> None:
    """Reduce a per-start JSONL file to a v2 score."""
    import json

    from optbench.score import score_v2

    records = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = score_v2(records, cap)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"successes={summary['successes']}/{summary['n']} "
        f"par2_mean={summary['par2_mean']} median={summary['median']}"
    )


@app.command
def statistics(
    output: Path = Path("results/validation/statistics.json"),
    root: Path = Path("."),
    results: Path = Path("results/asv"),
    commit: str = "",
    machine: str = "",
) -> None:
    """Score the quadratic sample from saved asv tracks.

    ``--machine`` picks the asv machine directory when more than one
    holds quadratic tracks.
    """
    from optbench.asv_score import write_statistics_from_asv

    document = write_statistics_from_asv(output, root=root, results_dir=root / results, machine=machine or None)
    recorded = document["provenance"]["commit"]
    if commit and not recorded.startswith(commit):
        raise SystemExit(f"asv results are commit {recorded}, not {commit}")
    displacement = document["displacement"]["metrics"]
    dimer = document["dimer"]["metrics"]
    print(
        f"wrote {output} commit={recorded} displacement_median={displacement['median']} "
        f"dimer_median={dimer['median']} "
        f"displacement_failed={displacement['nfailed']} dimer_failed={dimer['nfailed']}"
    )


@app.command
def asv_machine() -> None:
    """Store the asv machine entry for ``BENCH_HOST_LABEL`` in ``~/.asv-machine.json``.

    The entry is named by the label, or ``unspecified`` when the variable
    is unset, and its other fields are ``unspecified``. Pass the same
    label to ``asv run --machine``.
    """
    from optbench.asv_machine import write_machine_file

    record = write_machine_file()
    print(" ".join(f"{key}={value}" for key, value in record.items()))


@app.command
def sitesize(
    root: Path = Path("."),
    build: Path = Path("docs/build"),
) -> None:
    """Measure built pages and fail when one exceeds the weight budget."""
    from optbench.site_weight import check_site_weight

    destination = build if build.is_absolute() else root / build
    raise SystemExit(check_site_weight(destination))


@app.command
def board(
    root: Path = Path("."),
    output: Path = Path("results/board"),
) -> None:
    """Score committed results and write the v2 leaderboard."""
    from optbench.board import write_board

    destination = output if output.is_absolute() else root / output
    document = write_board(root, destination)
    print(
        f"wrote {destination} problems={len(document['problems'])} "
        f"ranked={document['ranked_count']} short={len(document['short'])} "
        f"unparsed={len(document['unparsed'])} unmapped={len(document['unmapped'])}"
    )
    for problem in document["problems"].values():
        ranked = problem["ranked"]
        if not ranked:
            continue
        print(problem["id"])
        for index, row in enumerate(ranked[:5], start=1):
            score = row["par2_mean"] if row["par2_mean"] is not None else row["median"]
            print(
                f"  {index} {row['template'] or row['entry']} "
                f"source={row['source']} "
                f"par2_mean={row['par2_mean']} median={row['median']} "
                f"mean_calls={row['mean_calls']} median_calls={row['median_calls']} "
                f"mean_per_image={row.get('mean_per_image')} "
                f"min_calls={row['min_calls']} max_calls={row['max_calls']} "
                f"success_rate={row['success_rate']} "
                f"n={row['n']} nfailed={row['nfailed']} score={score}"
            )


@app.command(name="validate-counter")
def validate_counter(
    output: Path = Path("results/validation/lj38-counter.dat"),
    *,
    root: Path = Path("."),
    client: Path | None = None,
    limit: int = 3,
    work: Path | None = None,
) -> None:
    """Compare the harness counter with eonclient on an LJ38 subset."""
    from optbench.validate_counter import validate_subset

    summary = validate_subset(root, _eon_client(client), output, limit=limit, work=work)
    print(
        f"wrote {output} version={summary['version']} commit={summary['commit']} "
        f"matched_client={summary['matched_client']}/{limit} "
        f"matched_builtin={summary['matched_builtin']}/{limit}"
    )


def main(argv: list[str] | None = None) -> None:
    from optbench.v2 import host_label

    try:
        host_label()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    app(argv)


if __name__ == "__main__":
    main()
