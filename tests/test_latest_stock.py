"""Paper-row templates carry the archived method settings."""

from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem

ROOT = Path(__file__).resolve().parents[1]


def test_basinhopping_matches_the_archived_driver():
    template = load_template(ROOT, "pele/lj38-basinhopping")
    assert template["pele"] == {
        "method": "basinhopping",
        "temperature": 1.0,
        "stepsize": 0.41,
        "acc_ratio": 0.5,
        "interval": 100,
        "quench_tol": 0.001,
        "nquenches": 100000,
    }
    assert "factor" not in template["pele"]
    assert template["tuning"] == "default"
    assert template["code"] == "pele"
    assert template["code_version"] == "0.1.2"
    assert int(template["failure_cap"]) > 2533900
    _check_stop(resolve_problem(ROOT, template["problem"]), template)
    from benchmarks.bench_pele import _specs

    assert all(spec["template_id"] != "pele/lj38-basinhopping" for spec in _specs())


def test_morse_sd_matches_the_archived_config():
    svn = load_template(ROOT, "eon-svn/morse-sd-harness")
    optimizer = svn["ini"]["Optimizer"]
    assert optimizer["opt_method"] == "sd"
    assert float(optimizer["sd_alpha"]) == 0.04
    assert float(optimizer["max_move"]) == 0.2
    assert float(optimizer["converged_force"]) == 0.001
    assert "max_iterations" not in optimizer
    assert "sd_twopoint" not in optimizer
    assert "SD" not in svn["ini"]
    assert svn["code"] == "EON"
    assert svn["code_version"] == "(SVN r2549)"
    assert svn["commit"] == "r2549"
    assert svn["client_family"] == "svn"
    _check_stop(resolve_problem(ROOT, svn["problem"]), svn)

    develop = load_template(ROOT, "eon-develop/morse-sd-harness")
    optimizer = develop["ini"]["Optimizer"]
    assert optimizer["opt_method"] == "sd"
    assert "sd_alpha" not in optimizer
    assert float(optimizer["max_move"]) == 0.2
    assert float(optimizer["converged_force"]) == 0.001
    assert "max_iterations" not in optimizer
    assert float(develop["ini"]["SD"]["sd_alpha"]) == 0.04
    assert "sd_twopoint" not in develop["ini"]["SD"]
    assert develop["code"] == "eOn"
    assert develop["code_version"] == "(develop 500350d9)"
    assert develop["commit"] == "500350d9fae881f06c7ef194e5868da079c4be5a"
    assert develop["client_family"] == "develop"
    _check_stop(resolve_problem(ROOT, develop["problem"]), develop)


def test_barrier_window_waits_for_every_reference_barrier():
    from optbench.runners.eon_htst import _reference_ids, _windows

    table = "\n".join(
        [
            "proc # saddle energy prefactor product product energy product prefactor barrier rate repeats",
            "0 -1 1 -1 -1 1 0.74619 1 0",
            "1 -1 1 -1 -1 1 0.23285 1 0",
        ]
    )
    found, missed = _reference_ids(table)
    assert found == [1]
    assert missed == [0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
    rows = [
        {"sad": 100, "barrier": 0.74619, "status": "good", "process_id": 0},
        {"sad": 100, "barrier": 0.23281, "status": "good", "process_id": 1},
        {"sad": 100, "barrier": 0.0, "status": "Not Connected", "process_id": None},
    ]
    barriers, rates = _windows(rows, [10.0, 20.0, 30.0], found, missed)
    assert barriers == []
    assert rates == []
    complete = [
        {"sad": 1, "barrier": barrier, "status": "good", "process_id": index}
        for index, barrier in enumerate([0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372])
    ]
    closed, _rates = _windows(complete, [1.0] * 8, list(range(8)), [])
    assert len(closed) == 1
    assert closed[0]["jobs"] == 8
    assert closed[0]["force_calls"] == 8


def test_htst_matches_the_archived_protocol():
    for name in ("eon-svn/htst-al-dimer-cg-stock", "eon-develop/htst-al-dimer-cg-stock"):
        template = load_template(ROOT, name)
        ini = template["ini"]
        assert float(ini["AKMC"]["confidence"]) == 1.95
        assert ini["AKMC"]["server_side_process_search"] is False
        assert float(ini["Prefactor"]["default_value"]) == 1.2e12
        assert ini["Process Search"]["minimize_first"] is True
        assert int(ini["Optimizer"]["max_iterations"]) == 5000
        assert float(ini["Optimizer"]["converged_force"]) == 0.001
        assert int(ini["Communicator"]["num_jobs"]) == 10000
        assert int(template["search_limit"]) == 10000
        assert "pass_limit" not in template
        saddle = ini["Saddle Search"]
        assert float(saddle["displace_radius"]) == 3.3
        assert float(saddle["displace_magnitude"]) == 0.5
        assert float(saddle["max_energy"]) == 100.0
        assert float(saddle["displace_listed_atom_weight"]) == 1.0
        # r2549 reads -1 as the last atom. develop 500350d9 reads -1 as every free atom.
        expected_list = "-1" if name.startswith("eon-svn/") else "384"
        assert str(saddle["displace_atom_list"]) == expected_list
        assert saddle["min_mode_method"] == "dimer"
        dimer = ini["Dimer"]
        assert dimer == {"rotations_max": 50, "converged_angle": 1, "opt_method": "cg"}
        assert template["algorithm"] == "Dimer-CG rotation: max=50 tol=1"
        assert template["tuning"] == "default"
        assert "partial" not in template
        _check_stop(resolve_problem(ROOT, template["problem"]), template)


def test_worker_count_queues_in_front_of_the_slots():
    from optbench.runners.eon_htst import _live_communicator, _prepare_server

    assert _live_communicator(8, 10000) == (8, 64)
    assert _live_communicator(8, 2) == (8, 2)
    template = load_template(ROOT, "eon-svn/htst-al-dimer-cg-stock")
    text = _prepare_server(template, "/bin/true", 42, 8, Path("/tmp/bridge"), 10000)
    assert "number_of_CPUs = 8" in text
    assert "num_jobs = 64" in text
    assert "confidence = 1.95" in text
    assert "displace_radius = 3.3" in text
    assert "max_iterations = 5000" in text
    assert "opt_method = cg" in text
    assert "sd_alpha" not in text


def test_resume_keeps_registered_searches(tmp_path):
    from optbench.runners.eon_htst import _can_resume, _clear_inflight, _snapshots_from_text

    dest = tmp_path / "work"
    state = dest / "states" / "0"
    state.mkdir(parents=True)
    (state / "search_results.txt").write_text("kept\n", encoding="utf-8")
    (dest / "info.txt").write_text("first_run = False\n", encoding="utf-8")
    (dest / "prng.pkl").write_bytes(b"keep")
    scratch = dest / "jobs" / "scratch" / "0_9"
    scratch.mkdir(parents=True)
    (scratch / "stdout.dat").write_text("partial", encoding="utf-8")
    assert _can_resume(dest)
    _clear_inflight(dest)
    assert (state / "search_results.txt").read_text(encoding="utf-8") == "kept\n"
    assert (dest / "prng.pkl").read_bytes() == b"keep"
    assert list((dest / "jobs" / "scratch").iterdir()) == []
    text = "\n".join(
        [
            "noise",
            "id pass 1 searches=6 barrier_windows=0 rate_windows=0 calls=10",
            "id pass 2 searches=12 barrier_windows=0 rate_windows=0 calls=40",
            "id pass 2 searches=12 barrier_windows=0 rate_windows=0 calls=40",
        ]
    )
    assert _snapshots_from_text(text) == [(0, 0), (6, 10), (12, 40)]
