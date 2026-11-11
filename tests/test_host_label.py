import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from optbench.__main__ import main
from optbench.asv_machine import FIELDS, machine_dir, machine_record, write_machine_file
from optbench.asv_score import load_quadratic_result, statistics_from_asv
from optbench.ase_sella import run_template
from optbench.v2 import HOST_LABEL_ENV, collect_provenance, host_label, relabel_machine

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "optbench"


def test_host_label_falls_back_to_unspecified(monkeypatch):
    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    assert host_label() == "unspecified"
    provenance = collect_provenance(code="quadratic", code_version="1", commit="abc", parameters={})
    assert provenance["machine"]["node"] == "unspecified"


def test_a_blank_host_label_falls_back_to_unspecified(monkeypatch):
    for blank in ("  ", "\n"):
        monkeypatch.setenv(HOST_LABEL_ENV, blank)
        assert host_label() == "unspecified"


def test_host_label_reads_bench_host_label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, " workstation-a\n")
    assert HOST_LABEL_ENV == "BENCH_HOST_LABEL"
    assert host_label() == "workstation-a"
    provenance = collect_provenance(code="quadratic", code_version="1", commit="abc", parameters={})
    assert provenance["machine"]["node"] == "workstation-a"


@pytest.mark.parametrize(
    "value",
    ["workstation-a\nforce_calls 1", "workstation-a\rranked false", "a\tb", "a\x1bb", "two words", "a/b", "../a", "-a"],
)
def test_a_label_that_would_add_a_field_or_a_path_is_rejected(monkeypatch, value):
    monkeypatch.setenv(HOST_LABEL_ENV, value)
    with pytest.raises(ValueError, match="BENCH_HOST_LABEL is .* A machine label is letters, digits"):
        host_label()


def test_the_command_line_stops_on_a_bad_label(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a\nranked false")
    with pytest.raises(SystemExit, match="BENCH_HOST_LABEL is 'workstation-a\\\\nranked false'"):
        main(["asv-machine"])
    assert not (tmp_path / ".asv-machine.json").exists()


def test_a_bad_label_stops_a_run_before_it_starts(monkeypatch, tmp_path):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a\npartial true")
    with pytest.raises(ValueError, match="BENCH_HOST_LABEL"):
        run_template(ROOT, "no-such-template", output=tmp_path / "benchmark.dat")


def test_an_explicit_machine_string_wins(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    provenance = collect_provenance(code="quadratic", code_version="1", commit="abc", parameters={}, machine="testbox")
    assert provenance["machine"]["node"] == "testbox"
    with pytest.raises(ValueError, match="machine is"):
        collect_provenance(code="quadratic", code_version="1", commit="abc", parameters={}, machine="a\nb")


def test_a_machine_mapping_keeps_its_fields_and_takes_the_label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    stored = {"machine": "node-name", "node": "node-name", "arch": "x86_64", "python": "3.12"}
    provenance = collect_provenance(code="quadratic", code_version="1", commit="abc", parameters={}, machine=stored)
    assert provenance["machine"] == {"machine": "workstation-a", "arch": "x86_64", "python": "3.12"}
    assert relabel_machine(stored) == provenance["machine"]


def test_the_ase_sella_summary_takes_the_label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    spec = importlib.util.spec_from_file_location("summarize", ROOT / "results" / "ase-sella-full" / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = module.build_document(ROOT / "results" / "asv", python=None, run_env=None)
    assert document["machine"]["machine"] == "workstation-a"


def test_the_asv_machine_record_is_the_label_and_nothing_else(monkeypatch, tmp_path):
    from asv.machine import Machine

    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    record = machine_record()
    assert record == {"machine": "workstation-a", **{field: "unspecified" for field in FIELDS}}
    assert set(record) == {name for name, _ in Machine.fields}
    path = tmp_path / "asv-machine.json"
    write_machine_file(path=path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"workstation-a": record, "version": 1}
    loaded = Machine.load(machine_name="workstation-a", _path=str(path))
    assert {name: getattr(loaded, name) for name, _ in Machine.fields} == record


def test_optbench_asv_machine_writes_the_home_machine_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    with pytest.raises(SystemExit) as exit:
        main(["asv-machine"])
    assert exit.value.code == 0
    stored = json.loads((tmp_path / ".asv-machine.json").read_text(encoding="utf-8"))
    assert stored == {"unspecified": machine_record("unspecified"), "version": 1}
    assert capsys.readouterr().out.split() == [
        "machine=unspecified",
        "os=unspecified",
        "arch=unspecified",
        "cpu=unspecified",
        "num_cpu=unspecified",
        "ram=unspecified",
    ]


def test_machine_dir_does_not_overwrite_another_record(tmp_path):
    folder = machine_dir(tmp_path, "workstation-a")
    assert folder == tmp_path / "workstation-a"
    assert json.loads((folder / "machine.json").read_text(encoding="utf-8"))["machine"] == "workstation-a"
    assert machine_dir(tmp_path, "workstation-a") == folder
    (tmp_path / "older").mkdir()
    older = {"machine": "older", "os": "Linux", "version": 1}
    (tmp_path / "older" / "machine.json").write_text(json.dumps(older), encoding="utf-8")
    with pytest.raises(ValueError, match="Use another label"):
        machine_dir(tmp_path, "older")


def test_a_second_machine_directory_is_picked_by_label(monkeypatch, tmp_path):
    archived = load_quadratic_result(ROOT / "results" / "asv")
    source = ROOT / "results" / "asv" / archived._filename
    old = tmp_path / archived.params["machine"]
    old.mkdir()
    shutil.copy2(source.parent / "machine.json", old / "machine.json")
    shutil.copy2(source, old / source.name)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["params"] = {**document["params"], **machine_record("workstation-a")}
    (machine_dir(tmp_path, "workstation-a") / source.name).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SystemExit, match="Pass --machine"):
        load_quadratic_result(tmp_path)
    with pytest.raises(SystemExit, match="no QuadraticDisplacement tracks for machine workstation-b"):
        load_quadratic_result(tmp_path, "workstation-b")
    assert load_quadratic_result(tmp_path, "workstation-a").params["machine"] == "workstation-a"

    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    fresh = statistics_from_asv(ROOT, tmp_path, "workstation-a")
    assert fresh["provenance"]["machine"] == {**machine_record("workstation-a"), "python": document["params"]["python"]}
    again = statistics_from_asv(ROOT, tmp_path, archived.params["machine"])
    assert again["provenance"]["machine"]["machine"] == "workstation-a"


def test_no_runner_reads_the_machine_name():
    readers = ("gethostname", "getfqdn", "platform.node", "platform.uname", "os.uname")
    hits = [
        f"{path.relative_to(PACKAGE)}: {reader}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for reader in readers
        if reader in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def _commands(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".yml":
        return text.splitlines()
    return [line for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S) for line in block.splitlines()]


def test_documented_asv_commands_name_the_label():
    paths = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
    ]
    lines = [(path.relative_to(ROOT), line.strip()) for path in paths for line in _commands(path)]
    assert [hit for hit in lines if "asv machine" in hit[1]] == []
    writers = [hit for hit in lines if re.search(r"asv (run|compare)\b", hit[1])]
    assert writers
    assert [hit for hit in writers if "--machine" not in hit[1]] == []
    workflow = (ROOT / ".github" / "workflows" / "asv.yml").read_text(encoding="utf-8")
    assert "uv run optbench asv-machine" in workflow
    assert "--machine unspecified" in workflow
