"""Record files are added once. A later commit cannot edit, delete, or rename one."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES_BASE = "9cd4ea7b3e7c11591bede7aabb1e3c1a5406f0e8"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT)


def _is_record(path: str) -> bool:
    if path.startswith("archive/entries/"):
        return True
    if not path.startswith("results/") or path.startswith("results/board/"):
        return False
    return not (path.endswith(".py") or path == "results/README.md" or path.endswith("/README.md"))


def _record_paths(repo: Path) -> set[str]:
    """Every tracked archive entry and results/ path, except the generated board, scripts, and README."""
    names: set[str] = set()
    for args in (
        ("log", "-z", "--name-only", "--pretty=format:", "--", "results", "archive/entries"),
        ("ls-files", "-z", "--", "results", "archive/entries"),
    ):
        try:
            out = _git(repo, *args)
        except subprocess.CalledProcessError:
            continue
        names.update(part.lstrip("\n") for part in out.split("\0") if part.lstrip("\n"))
    return {path for path in names if _is_record(path)}


def record_history_violations(repo: Path, base: str | None = None) -> list[str]:
    """Any non-add of a record after ``base``, including symlinks and each merge parent."""
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise RuntimeError(f"{repo} history is shallow; record history cannot be checked")
    if base is not None:
        try:
            _git(repo, "cat-file", "-e", f"{base}^{{commit}}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{repo} does not contain {base}; history is incomplete") from exc
        range_spec = f"{base}..HEAD"
    else:
        range_spec = "HEAD"
    try:
        out = _git(repo, "log", "-m", "-z", "--name-status", "--pretty=format:commit %H", range_spec)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.output) from exc
    records = _record_paths(repo)
    violations: list[str] = []
    # Each commit is "commit <sha>\n<status>\0<path>\0[<path>\0]..." with -z.
    fields: list[str] = []
    for field in out.split("\0"):
        field = field.lstrip("\n")
        if field.startswith("commit "):
            header, _, status = field.partition("\n")
            fields.append(header)
            if status:
                fields.append(status)
        else:
            fields.append(field)
    commit = "HEAD"
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if field.startswith("commit "):
            commit = field.split()[1]
            continue
        status = field
        count = 2 if status[0] in {"R", "C"} else 1
        paths = fields[index : index + count]
        index += count
        if status[0] == "A":
            continue
        for path in paths:
            if path in records:
                violations.append(f"{commit} {status} {path}")
    return violations


def test_this_series_does_not_rewrite_records():
    try:
        _git(ROOT, "merge-base", "--is-ancestor", SERIES_BASE, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{SERIES_BASE} is not in this history") from exc
    assert record_history_violations(ROOT, SERIES_BASE) == []


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "rgoswami@ieee.org")
    _git(repo, "config", "user.name", "Rohit Goswami")
    _git(repo, "config", "core.hooksPath", "/dev/null")
    _git(repo, "config", "commit.gpgsign", "false")
    record = repo / "results" / "campaign" / "row.dat"
    record.parent.mkdir(parents=True)
    record.write_text("force_calls 1\n", encoding="utf-8")
    _git(repo, "add", "results/campaign/row.dat")
    _git(repo, "commit", "-q", "-m", "add the record")
    return repo


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def test_an_edit_of_a_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = repo / "results" / "campaign" / "eon-3.5.0-saddle-pt-classic-develop-protocol.dat"
    path.write_text("force_calls 1\n", encoding="utf-8")
    _commit(repo, "add the eon ledger")
    path.write_text("force_calls 2\n", encoding="utf-8")
    _commit(repo, "edit the eon ledger")
    assert record_history_violations(repo)


def test_deleting_a_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "results" / "campaign" / "row.dat").unlink()
    _commit(repo, "delete the record")
    assert record_history_violations(repo)


def test_deleting_and_restoring_a_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = repo / "results" / "campaign" / "row.dat"
    original = path.read_text(encoding="utf-8")
    path.unlink()
    _commit(repo, "delete the record")
    path.write_text(original, encoding="utf-8")
    _commit(repo, "restore the record")
    assert record_history_violations(repo)


def test_an_edit_of_a_record_added_in_the_series_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    added = repo / "results" / "campaign" / "rgsaddle-develop-aa6dd937-saddle-lanczos.dat"
    added.write_text("n 1\n", encoding="utf-8")
    _commit(repo, "add a later record")
    added.write_text("n 2\n", encoding="utf-8")
    _commit(repo, "edit the later record")
    assert record_history_violations(repo)


def test_editing_provenance_json_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = repo / "results" / "campaign" / "rgsaddle-develop-aa6dd937-eon-provenance.json"
    path.write_text("{}\n", encoding="utf-8")
    _commit(repo, "add provenance")
    path.write_text('{"sha256": "0"}\n', encoding="utf-8")
    _commit(repo, "edit provenance")
    assert record_history_violations(repo)


def test_an_edit_under_results_asv_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = repo / "results" / "asv" / "bench.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    _commit(repo, "add an asv record")
    path.write_text('{"n": 1}\n', encoding="utf-8")
    _commit(repo, "edit the asv record")
    assert record_history_violations(repo)


def test_a_rename_of_a_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    _git(repo, "mv", "results/campaign/row.dat", "results/campaign/row-renamed.dat")
    _commit(repo, "rename the record")
    assert record_history_violations(repo)


def _scratch_clone(repo: Path, dest: Path) -> Path:
    subprocess.check_call(["git", "clone", "-q", repo.as_uri(), str(dest)])
    _git(dest, "config", "user.email", "rgoswami@ieee.org")
    _git(dest, "config", "user.name", "Rohit Goswami")
    _git(dest, "config", "core.hooksPath", "/dev/null")
    _git(dest, "config", "commit.gpgsign", "false")
    return dest


def test_a_nested_record_edit_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    nested = repo / "results" / "eon-develop-full" / "native" / "ts-pt-dimer" / "results.dat"
    nested.parent.mkdir(parents=True)
    nested.write_text("force_calls 1\n", encoding="utf-8")
    _commit(repo, "add the nested record")
    clone = _scratch_clone(repo, tmp_path / "clone")
    path = clone / "results" / "eon-develop-full" / "native" / "ts-pt-dimer" / "results.dat"
    path.write_text("force_calls 2\n", encoding="utf-8")
    _commit(clone, "edit the nested record")
    assert record_history_violations(clone)


def test_a_rename_of_the_only_nested_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    nested = repo / "results" / "dynamic" / "native" / "run" / "results.dat"
    nested.parent.mkdir(parents=True)
    nested.write_text("force_calls 1\n", encoding="utf-8")
    _commit(repo, "add the nested record")
    clone = _scratch_clone(repo, tmp_path / "clone")
    (clone / "notes").mkdir()
    _git(clone, "mv", "results/dynamic/native/run/results.dat", "notes/results.dat")
    _commit(clone, "rename the nested record out")
    violations = record_history_violations(clone)
    assert any("results/dynamic/native/run/results.dat" in item for item in violations)


def test_a_rename_out_of_a_record_directory_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    clone = _scratch_clone(repo, tmp_path / "clone")
    (clone / "notes").mkdir()
    _git(clone, "mv", "results/campaign/row.dat", "notes/row.dat")
    _commit(clone, "rename the record out")
    violations = record_history_violations(clone)
    assert any("results/campaign/row.dat" in item for item in violations)


def _add_record(repo: Path, relative: str, text: str = "{}\n") -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _commit(repo, f"add {relative}")
    return path


def test_a_top_level_result_edit_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = _add_record(repo, "results/sota-lj38.json")
    path.write_text('{"n": 1}\n', encoding="utf-8")
    _commit(repo, "edit the top-level record")
    assert record_history_violations(repo)


def test_a_quadratic_record_edit_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = _add_record(repo, "results/quadratic/quadratic.json")
    path.write_text('{"n": 1}\n', encoding="utf-8")
    _commit(repo, "edit the quadratic record")
    assert record_history_violations(repo)


def test_an_ase_sella_record_delete_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = _add_record(repo, "results/ase-sella-full/summary.json")
    path.unlink()
    _commit(repo, "delete the ase-sella record")
    assert record_history_violations(repo)


def test_replacing_a_record_with_a_symlink_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = repo / "results" / "campaign" / "row.dat"
    path.unlink()
    path.symlink_to("elsewhere")
    _commit(repo, "replace the record with a symlink")
    violations = record_history_violations(repo)
    assert any(item.split()[1].startswith("T") for item in violations)


def test_a_merge_that_rewrites_a_record_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _commit(repo, "side")
    _git(repo, "checkout", "-q", branch)
    (repo / "main.txt").write_text("main\n", encoding="utf-8")
    _commit(repo, "main")
    _git(repo, "merge", "--no-ff", "--no-commit", "side")
    (repo / "results" / "campaign" / "row.dat").write_text("evil\n", encoding="utf-8")
    _commit(repo, "merge")
    assert record_history_violations(repo)


def test_shallow_history_fails_loudly(tmp_path: Path):
    repo = _repo(tmp_path)
    clone = tmp_path / "shallow"
    subprocess.check_call(["git", "clone", "--depth", "1", repo.as_uri(), str(clone)])
    with pytest.raises(RuntimeError, match="shallow"):
        record_history_violations(clone)

def test_an_edit_of_a_quoted_path_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    quoted = repo / "results" / "campaign" / 'café "x".dat'
    quoted.write_text("force_calls 1\n", encoding="utf-8")
    _commit(repo, "add the quoted record")
    quoted.write_text("force_calls 2\n", encoding="utf-8")
    _commit(repo, "edit the quoted record")
    violations = record_history_violations(repo)
    assert any('café "x".dat' in item for item in violations)


def test_an_archive_entry_edit_is_a_violation(tmp_path: Path):
    repo = _repo(tmp_path)
    path = _add_record(repo, "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/benchmark.dat", "force_calls 827\n")
    path.write_text("force_calls 700\n", encoding="utf-8")
    _commit(repo, "edit the archive entry")
    violations = record_history_violations(repo)
    assert any("archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/benchmark.dat" in item for item in violations)
