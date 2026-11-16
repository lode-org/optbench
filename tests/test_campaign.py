import json
import stat
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]


def test_a_changed_template_is_read_again(tmp_path: Path) -> None:
    directory = tmp_path / "templates"
    directory.mkdir()
    path = directory / "one.toml"
    path.write_text('id = "one"\nengine = "none"\n', encoding="utf-8")
    assert list_templates(tmp_path)[0]["id"] == "one"
    path.write_text('id = "two"\nengine = "none"\nextra = 1\n', encoding="utf-8")
    assert list_templates(tmp_path)[0]["id"] == "two"
    assert list_templates(tmp_path)[0]["id"] == "two"


def test_shipped_templates_use_the_lj38_stop():
    ids = []
    for template in list_templates(ROOT):
        ids.append(template["id"])
        problem_name = template.get("problem", "minimization/lj38")
        _check_stop(resolve_problem(ROOT, problem_name), template)
    assert "eon-client/lbfgs-m20" in ids
    assert "eon-svn/lbfgs-m100" in ids
    assert "eon/lj38-lbfgs" in ids
    assert "eon/lj38-lbfgs-m100" in ids
    assert "eon/neb-pt-lbfgs" in ids
    assert "eon/de-lj38-lbfgs-m100" in ids
    assert "eon/ts-pt-dimer" in ids
    assert "eon/develop-ts-lj38-lanczos" in ids
    assert "eon/develop-de-lj38-cap" in ids
    assert "eon/develop-low-lj38-dimer" in ids
    assert "eon/develop-saddle-pt-lanczos" in ids
    assert "eon/develop-de-2ts" in ids


def test_develop_de_2ts_cap_is_the_problem_cap():
    from optbench.campaign import load_template

    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38_2ts_separation")
    cap = int(problem["convergence"]["max_force_calls"])
    assert cap == 250000
    for name in (
        "eon/develop-de-2ts",
        "eon/develop-de-2ts-accept-energy",
        "eon/develop-de-2ts-oci",
        "eon/develop-de-2ts-ew",
    ):
        assert load_template(ROOT, name)["failure_cap"] == cap


def test_matched_dimer_and_neb_templates_omit_energy_accept():
    from optbench.campaign import load_template

    matched = (
        "eon/develop-ts-lj38-dimer",
        "eon/develop-ts-pt-dimer",
        "eon/develop-de-lj38-cap",
        "eon/develop-de-2ts",
    )
    energy = (
        "eon/develop-ts-lj38-dimer-accept-energy",
        "eon/develop-ts-pt-dimer-accept-energy",
        "eon/develop-de-lj38-cap-accept-energy",
        "eon/develop-de-2ts-accept-energy",
    )
    for name in matched:
        template = load_template(ROOT, name)
        ini = template["ini"]
        assert "LBFGS" not in ini
        assert ini["Optimizer"]["lbfgs_memory"] == 20
        assert "lbfgs_accept" not in ini["Optimizer"]
        assert template["tuning"] == "r2549-ini"
    for name in energy:
        template = load_template(ROOT, name)
        assert template["ini"]["LBFGS"]["lbfgs_accept"] == "energy"
        assert template["tuning"] == "accept-energy"
        assert "lbfgs_memory" not in template["ini"]["Optimizer"]


def test_run_writes_one_benchmark_dat(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    for name in ("cluster_0000.con", "cluster_0001.con"):
        (geometries / name).write_text("pos\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('config.ini').read_text()\n"
        "calls = 181 if 'lbfgs_memory = 100' in text else 449\n"
        "Path('results.dat').write_text("
        "'0 termination_reason\\n%f total_force_calls\\n' % calls)\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
    )
    text = output.read_text(encoding="utf-8")
    assert text == (
        "force_calls 181\n"
        "force_calls_min 181\n"
        "force_calls_max 181\n"
        "nfailed 0\n"
        "code Eon svn\n"
        "code_version r2549\n"
        "algorithm LBFGS\n"
        "template eon-svn/lbfgs-m100\n"
        "kernel client\n"
    )
    fields = parse(text)
    assert fields["force_calls"] == "181"
    assert fields["nfailed"] == "0"
    assert fields["template"] == "eon-svn/lbfgs-m100"
    assert summary["starts"] == 2
    import json

    starts = [
        json.loads(line)
        for line in (tmp_path / "benchmark.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in starts] == ["cluster_0000", "cluster_0001"]
    assert {row["force_calls"] for row in starts} == {181}
    assert {row["counter"] for row in starts} == {"client"}
    assert {row["comparable"] for row in starts} == {False}
    assert "potential = lj_cluster" in (tmp_path / "work" / "cluster_0000" / "config.ini").read_text()


def test_modern_eon_template_records_commit_and_each_start(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    (geometries / "cluster_0000.con").write_text("pos\n", encoding="utf-8")
    (geometries / "cluster_0001.con").write_text("pos\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('config.ini').read_text()\n"
        "assert '[LBFGS]' in text\n"
        "assert 'lbfgs_accept = energy' in text\n"
        "calls = 181 if 'lbfgs_memory = 100' in text else 201\n"
        "Path('results.dat').write_text("
        "'0 termination_reason\\n%f total_force_calls\\n' % calls)\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon/lj38-lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
    )
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "181"
    assert fields["commit"] == "1d71c944501832bfafb955ffe13b9aae50c8efd4"
    assert fields["problem"] == "minimization/lj38"
    assert fields["template"] == "eon/lj38-lbfgs-m100"
    assert summary["starts"] == 2
    starts = (tmp_path / "benchmark.starts").read_text(encoding="utf-8").splitlines()
    assert len(starts) == 2
    assert '"failed": false' in starts[0]
    assert "cluster_0000.con" in starts[0]


def test_run_stages_template_declared_inputs(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    (geometries / "cluster_0000.con").write_text("pos\n", encoding="utf-8")
    aux = tmp_path / "aux"
    aux.mkdir()
    in_lammps = aux / "in.lammps"
    in_lammps.write_text("pair_style lj/cut 40.0\n", encoding="utf-8")
    pair_table = aux / "pair.table"
    pair_table.write_text("PAIR TABLE\n", encoding="utf-8")
    template = tmp_path / "template.toml"
    template.write_text(
        f"""
id = "test/eon-aux"
engine = "eon"
problem = "minimization/lj38"

[ini.Main]
job = "minimization"

[ini.Potential]
potential = "ljcluster"

[ini.Optimizer]
opt_method = "lbfgs"
convergence_metric = "norm"
converged_force = 0.01
max_iterations = 10000

[[inputs]]
source = "{in_lammps.as_posix()}"
target = "in.lammps"

[[inputs]]
source = "{pair_table.as_posix()}"
target = "potentials/pair.table"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "assert Path('in.lammps').read_text().strip() == 'pair_style lj/cut 40.0'\n"
        "assert Path('potentials/pair.table').read_text().strip() == 'PAIR TABLE'\n"
        "Path('results.dat').write_text('0 termination_reason\\n123 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        str(template),
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
    )
    assert summary["starts"] == 1
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "123"
    assert fields["nfailed"] == "0"
    staged = tmp_path / "work" / "cluster_0000"
    assert (staged / "in.lammps").read_text(encoding="utf-8").strip() == "pair_style lj/cut 40.0"
    assert (staged / "potentials" / "pair.table").read_text(encoding="utf-8").strip() == "PAIR TABLE"


def test_a_limit_runs_only_the_first_geometries(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    for name in ("cluster_0000.con", "cluster_0001.con", "cluster_0002.con"):
        (geometries / name).write_text("pos\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('results.dat').write_text('0 termination_reason\\n10 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
        limit=1,
    )
    assert summary["starts"] == 1
    starts = [
        json.loads(line)
        for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in starts] == ["cluster_0000"]
    assert (tmp_path / "work" / "cluster_0000").is_dir()
    assert not (tmp_path / "work" / "cluster_0001").exists()


def test_resume_keeps_a_finished_client_geometry(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    for name in ("cluster_0000.con", "cluster_0001.con"):
        (geometries / name).write_text("pos\n", encoding="utf-8")
    kept = tmp_path / "work" / "cluster_0000"
    kept.mkdir(parents=True)
    kept.joinpath("results.dat").write_text(
        "0 termination_reason\n7 total_force_calls\n",
        encoding="utf-8",
    )
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('launched').write_text('yes', encoding='utf-8')\n"
        "Path('results.dat').write_text('0 termination_reason\\n181 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "kept.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
        workers=1,
        resume=True,
    )
    assert summary["counts"] == [7, 181]
    assert not (kept / "launched").exists()
    assert (tmp_path / "work" / "cluster_0001" / "launched").is_file()
    fresh = tmp_path / "fresh.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=fresh,
        work=tmp_path / "work",
        workers=1,
        resume=False,
    )
    assert summary["counts"] == [181, 181]
    assert (kept / "launched").is_file()


def test_a_nan_client_count_does_not_stop_the_other_starts(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    (geometries / "cluster_0000.con").write_text("nan-start\n", encoding="utf-8")
    (geometries / "cluster_0001.con").write_text("kept\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('pos.con').read_text()\n"
        "if 'nan-start' in text:\n"
        "    Path('results.dat').write_text('nan total_force_calls\\n')\n"
        "else:\n"
        "    Path('results.dat').write_text('0 termination_reason\\n10 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
        workers=2,
    )
    assert summary["starts"] == 2
    assert summary["nfailed"] == 1
    assert summary["force_calls"] == 10
    assert summary["counts"] == [summary["counts"][0], 10]
    assert summary["counts"][0] > 10


def test_a_missing_client_result_is_recorded_as_a_crash(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    (geometries / "cluster_0000.con").write_text("missing\n", encoding="utf-8")
    (geometries / "cluster_0001.con").write_text("kept\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('pos.con').read_text()\n"
        "if 'missing' in text:\n"
        "    raise SystemExit(1)\n"
        "Path('results.dat').write_text('0 termination_reason\\n10 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
        workers=1,
    )
    rows = [json.loads(line) for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["success"] is False
    assert rows[0]["failure"] == "crash"
    assert rows[1]["success"] is True
    assert rows[1]["failure"] is None
    assert summary["nfailed"] == 1
    assert summary["force_calls"] == 10


def test_a_negative_start_limit_is_rejected(tmp_path: Path):
    from optbench.__main__ import main

    with pytest.raises(ValueError, match="limit must be zero or positive"):
        run(
            ROOT,
            "minimization/lj38",
            "geometric/lj38-cart",
            output=tmp_path / "bench.dat",
            limit=-1,
        )
    with pytest.raises(SystemExit, match="limit must be zero or positive"):
        main(
            [
                "run",
                "minimization/lj38",
                "geometric/lj38-cart",
                "--root",
                str(ROOT),
                "--limit",
                "-1",
                "--output",
                str(tmp_path / "cli.dat"),
            ]
        )


def test_an_interrupted_starts_file_keeps_the_previous_rows(tmp_path: Path, monkeypatch):
    from optbench.campaign import write_starts

    path = tmp_path / "bench.starts.jsonl"
    path.write_text('{"start": "kept"}\n', encoding="utf-8")
    original = Path.write_text

    def fail_temp(self, text, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise RuntimeError("interrupted")
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_temp)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_starts(path, [{"start": "new"}])
    assert path.read_text(encoding="utf-8") == '{"start": "kept"}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_a_short_iteration_cap_is_refused():
    problem = resolve_problem(ROOT, "minimization/lj38")
    template = {"ini": {"Optimizer": {"max_iterations": 10, "converged_force": 0.01}}}
    with pytest.raises(ValueError):
        _check_stop(problem, template)
