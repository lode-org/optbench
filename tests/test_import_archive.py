"""Importing scores and geometries from a bare clone of the public archive.

The archive here is a real git repository built in a temporary directory, with
the layout of the public one: ``website/<suite>/<problem>/<entry>/benchmark.dat``
beside the files that entry ran with, geometries scattered through the tree,
and a second branch that holds structures the first does not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from optbench.__main__ import main
from optbench.import_archive import ARCHIVE_COMMIT, import_archive

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Archive Keeper",
    "GIT_AUTHOR_EMAIL": "keeper@example.invalid",
    "GIT_COMMITTER_NAME": "Archive Keeper",
    "GIT_COMMITTER_EMAIL": "keeper@example.invalid",
}

CON = (
    "Pt island on a fixed slab\n"
    "preBox header two\n"
    "11.08 9.6 30.0\n"
    "90.0 90.0 90.0\n"
    "postBox header one\n"
    "postBox header two\n"
    "1\n"
    "4\n"
    "195.084\n"
    "Pt\n"
    "Coordinates of Component 1\n"
    "0.0000000000 0.0000000000 0.0000000000 1 0\n"
    "2.7700000000 0.0000000000 0.0000000000 1 1\n"
    "1.3850000000 2.3990000000 0.0000000000 1 2\n"
    "1.3850000000 0.8000000000 2.3000000000 0 3\n"
)
XYZ = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n" + "# padding to clear the stub size\n" * 4
ALPHA = (
    "force_calls 100\nnfailed 0\ncode ToyCode\ncode_version 1.0\nalgorithm LBFGS\n"
    "contributer Someone\ndate 2024-01-01\nuncertanity 0.5\n"
)
BETA = "force_calls 90\nnfailed 1\nforce_calls_max 400\ncode ToyBeta\nuncertainty 0.4\n"
GAMMA = "force_calls 10\nforce_calls 12\nnfailed 0\ncode Gamma\n"
CLUSTERS = bytes(range(256))
POSCAR = b"Pt slab\n" + b"1.0\n" * 60


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, env=GIT_ENV)


def _put(root: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)


def _build_archive(root: Path) -> Path:
    work = root / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _put(
        work,
        {
            "website/minimization/toy/alpha/benchmark.dat": ALPHA,
            "website/minimization/toy/alpha/run.sh": "#!/bin/sh\necho alpha\n",
            "website/minimization/toy/alpha/notes.bin": b"\x00\x01\x02",
            "website/minimization/toy/alpha/search_results.txt": "left out by name\n",
            "website/minimization/toy/alpha/big.txt": "x" * 200_001,
            "website/minimization/toy/alpha/sub/inner.py": "print(1)\n",
            "website/minimization/toy/beta/benchmark.dat": BETA,
            "website/minimization/toy/beta/input.ini": "[Main]\njob = minimization\n",
            "website/saddle/pairs/gamma/benchmark.dat": GAMMA,
            "website/minimization/toy/clusters.tgz": CLUSTERS,
            "website/minimization/toy/start.con": CON,
            "website/saddle/pairs/start_copy.con": CON,
            "website/saddle/pairs/eon_r.con": CON + "tail\n",
            "website/saddle/pairs/ref.tgz": "../minimization/toy/clusters.tgz",
            "website/saddle/pairs/POSCAR_slab": POSCAR,
            "website/codes/toycode.tgz": bytes(300),
            "website/benchmark.dat": "force_calls 1\n",
            "elsewhere/minimization/toy/delta/benchmark.dat": "force_calls 2\nnfailed 0\n",
        },
    )
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "archive")
    _git(work, "checkout", "-q", "-b", "small_molecules")
    _put(work, {"molecules/water.xyz": XYZ, "website/saddle/pairs/start_copy.con": CON + "# changed on this branch\n"})
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "molecules")
    bare = root / "archive.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True, capture_output=True, env=GIT_ENV)
    return bare


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    root = tmp_path_factory.mktemp("archive")
    bare = _build_archive(root)
    dest = root / "copy"
    import_archive(bare, dest)
    return bare, dest


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_benchmark_dat_becomes_one_oracle_row_in_suite_problem_entry_order(imported):
    _bare, dest = imported
    rows = _lines(dest / "oracle" / "entries.jsonl")
    assert [(row["suite"], row["problem"], row["entry"]) for row in rows] == [
        ("minimization", "toy", "alpha"),
        ("minimization", "toy", "beta"),
        ("saddle", "pairs", "gamma"),
    ]
    alpha = rows[0]
    assert alpha["path"] == "website/minimization/toy/alpha/benchmark.dat"
    assert alpha["commit"] == ARCHIVE_COMMIT
    assert alpha["text"] == ALPHA
    assert alpha["sha256"] == hashlib.sha256(ALPHA.encode()).hexdigest()
    assert alpha["fields"]["force_calls"] == "100"
    assert alpha["fields"]["contributer"] == "Someone"
    assert alpha["fields"]["uncertanity"] == "0.5"
    assert alpha["duplicates"] == []


def test_a_repeated_key_keeps_its_first_value_and_the_later_one_is_listed(imported):
    _bare, dest = imported
    gamma = _lines(dest / "oracle" / "entries.jsonl")[2]
    assert gamma["fields"]["force_calls"] == "10"
    assert gamma["duplicates"] == [{"key": "force_calls", "line": 2, "kept": "10", "also": "12"}]


def test_a_problem_without_a_file_gets_one_that_requires_what_every_entry_reports(imported):
    _bare, dest = imported
    toy = tomllib.loads((dest / "problems" / "minimization" / "toy.toml").read_text(encoding="utf-8"))
    assert toy["id"] == "minimization/toy"
    assert toy["suite"] == "minimization"
    assert toy["problem"] == "toy"
    assert toy["contract"] == "shared"
    assert toy["entry_count"] == 2
    assert toy["metrics"]["required"] == ["force_calls", "nfailed", "uncertainty"]

    pairs = tomllib.loads((dest / "problems" / "saddle" / "pairs.toml").read_text(encoding="utf-8"))
    assert pairs["entry_count"] == 1
    assert pairs["metrics"]["required"] == ["nfailed"]


def test_an_existing_problem_file_is_left_as_it_was(tmp_path):
    bare = _build_archive(tmp_path)
    dest = tmp_path / "copy"
    kept = dest / "problems" / "saddle" / "pairs.toml"
    kept.parent.mkdir(parents=True)
    kept.write_text('id = "saddle/pairs"\ntitle = "Edited by hand"\n', encoding="utf-8")
    import_archive(bare, dest)
    assert kept.read_text(encoding="utf-8") == 'id = "saddle/pairs"\ntitle = "Edited by hand"\n'
    assert (dest / "problems" / "minimization" / "toy.toml").is_file()


def test_the_files_an_entry_ran_with_are_copied_beside_its_row_except_what_is_left_out(imported):
    _bare, dest = imported
    alpha = dest / "archive" / "entries" / "minimization" / "toy" / "alpha"
    assert (alpha / "run.sh").read_text(encoding="utf-8") == "#!/bin/sh\necho alpha\n"
    assert (alpha / "benchmark.dat").read_text(encoding="utf-8") == ALPHA
    assert not (alpha / "notes.bin").exists()
    assert not (alpha / "search_results.txt").exists()
    assert not (alpha / "big.txt").exists()
    assert not (alpha / "sub").exists()
    assert not (alpha / "inner.py").exists()
    beta = dest / "archive" / "entries" / "minimization" / "toy" / "beta"
    assert (beta / "input.ini").read_text(encoding="utf-8") == "[Main]\njob = minimization\n"


def test_geometries_are_kept_once_by_content_and_listed_by_branch_then_path(imported):
    _bare, dest = imported
    manifest = _lines(dest / "inputs" / "manifest.jsonl")
    assert [(row["branch"], row["path"]) for row in manifest] == [
        ("main", "website/minimization/toy/clusters.tgz"),
        ("main", "website/minimization/toy/start.con"),
        ("main", "website/saddle/pairs/POSCAR_slab"),
        ("main", "website/saddle/pairs/start_copy.con"),
        ("small_molecules", "molecules/water.xyz"),
    ]
    by_path = {row["path"]: row for row in manifest}
    digest = hashlib.sha256(CON.encode()).hexdigest()
    assert by_path["website/minimization/toy/start.con"]["sha256"] == digest
    assert by_path["website/saddle/pairs/start_copy.con"]["stored"] == by_path["website/minimization/toy/start.con"]["stored"]
    stored = dest / by_path["website/minimization/toy/start.con"]["stored"]
    assert stored == dest / "inputs" / "sha256" / digest / "start.con"
    assert stored.read_text(encoding="utf-8") == CON
    assert by_path["website/minimization/toy/clusters.tgz"]["bytes"] == len(CLUSTERS)
    assert (dest / by_path["website/minimization/toy/clusters.tgz"]["stored"]).read_bytes() == CLUSTERS
    assert (dest / by_path["molecules/water.xyz"]["stored"]).read_text(encoding="utf-8") == XYZ


def test_code_snapshots_path_stubs_and_a_branch_copy_of_a_listed_path_are_not_geometries(imported):
    _bare, dest = imported
    listed = {row["path"] for row in _lines(dest / "inputs" / "manifest.jsonl")}
    assert "website/codes/toycode.tgz" not in listed
    assert "website/saddle/pairs/eon_r.con" not in listed
    assert "website/saddle/pairs/ref.tgz" not in listed
    stored = {path.name for path in (dest / "inputs" / "sha256").rglob("*") if path.is_file()}
    assert stored == {"clusters.tgz", "start.con", "POSCAR_slab", "water.xyz"}
    start_copy = dest / next(
        row["stored"] for row in _lines(dest / "inputs" / "manifest.jsonl") if row["path"] == "website/saddle/pairs/start_copy.con"
    )
    assert "changed on this branch" not in start_copy.read_text(encoding="utf-8")


def test_the_counts_are_printed_and_a_second_import_changes_nothing(tmp_path, capsys):
    bare = _build_archive(tmp_path)
    dest = tmp_path / "copy"
    import_archive(bare, dest)
    first = capsys.readouterr().out
    assert first.splitlines() == ["oracle 3", "geometries 4", "manifest 5"]
    before = {path: path.read_bytes() for path in sorted(dest.rglob("*")) if path.is_file()}
    import_archive(bare, dest)
    assert capsys.readouterr().out == first
    assert {path: path.read_bytes() for path in sorted(dest.rglob("*")) if path.is_file()} == before


def test_a_directory_that_is_not_a_bare_repository_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="is not a bare git directory"):
        import_archive(tmp_path / "nothing-here", tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_the_command_imports_into_the_destination_it_is_given(tmp_path, capsys):
    bare = _build_archive(tmp_path)
    dest = tmp_path / "copy"
    with pytest.raises(SystemExit) as stopped:
        main(["import-archive", str(bare), "--dest", str(dest)])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.splitlines() == ["oracle 3", "geometries 4", "manifest 5"]
    assert (dest / "oracle" / "entries.jsonl").is_file()
