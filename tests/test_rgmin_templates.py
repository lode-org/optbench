import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem, run

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f15726f1c7983b3656c5dbc2f812e2394e144500"
METHODS = {
    "lbfgs",
    "fire",
    "fire2",
    "cg",
    "bfgs",
    "sr1",
    "sr2",
    "dogleg",
    "newton",
    "bb",
    "sd",
    "quickmin",
    "diis",
    "adam",
}


def test_rgmin_templates_cover_both_problems_and_both_tunings():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "rgmin")
    assert pinned["commit"] == COMMIT
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("rgmin/")]
    assert len(found) == 36
    seen = set()
    noise_sigmas = set()
    for template in found:
        assert template["engine"] == "rgmin"
        assert template["commit"] == COMMIT
        assert template["code_version"] == "0.4.1"
        assert template["tuning"] == "default"
        assert template["rgmin"]["method"] in METHODS
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        lj_like = template["problem"] in {"minimization/lj38", "minimization/lj38-noise"}
        if template["problem"] in {"minimization/lj38", "minimization/morse-bulk"}:
            assert template["tuning"] == "default"
        if lj_like or template["tuning"] == "tuned":
            assert template["rgmin"]["maxstep"] == 0.2
        else:
            assert "maxstep" not in template["rgmin"]
        if template["problem"] == "minimization/lj38-noise":
            assert "sigma" not in template["rgmin"]
            assert template["noise"]["sigma"] in (0.0, 1e-4, 1e-3, 1e-2)
            noise_sigmas.add((template["tuning"], template["noise"]["sigma"]))
        seen.add((template["problem"], template["rgmin"]["method"], template["tuning"]))
    for problem in ("minimization/lj38", "minimization/morse-bulk"):
        for method in METHODS:
            assert (problem, method, "default") in seen
    assert all(
        tuning == "default"
        for problem, _method, tuning in seen
        if problem in {"minimization/lj38", "minimization/morse-bulk"}
    )
    assert noise_sigmas == {
        (tuning, sigma)
        for tuning in ("default",)
        for sigma in (0.0, 1e-4, 1e-3, 1e-2)
    }
    memories = {
        row["id"]: row["rgmin"].get("memory", 10)
        for row in found
        if row["problem"] == "minimization/lj38" and row["rgmin"]["method"] == "lbfgs" and row["tuning"] == "default"
    }
    assert memories["rgmin/lj38-lbfgs"] == 10
    assert memories["rgmin/lj38-lbfgs-m20"] == 20
    assert memories["rgmin/lj38-lbfgs-m100"] == 100


def test_shared_work_directory_keeps_problem_archives_apart(tmp_path: Path):
    from optbench.runners.rgmin_min import _load_geometries

    work = tmp_path / "work"
    lj = resolve_problem(ROOT, "minimization/lj38")
    morse = resolve_problem(ROOT, "minimization/morse-bulk")
    lj_rows = _load_geometries(ROOT, lj, None, work, tmp_path / "lj.dat")
    morse_rows = _load_geometries(ROOT, morse, None, work, tmp_path / "morse.dat")
    assert len(lj_rows) == 1000
    assert len(morse_rows) == 100
    assert lj_rows[0][0].startswith("cluster_")
    assert morse_rows[0][0].startswith("pos_")
    again = _load_geometries(ROOT, morse, None, work, tmp_path / "morse.dat")
    assert [row[0] for row in again] == [row[0] for row in morse_rows]


def test_rgmin_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.rgmin_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "rgmin/lj38-lbfgs",
        output=tmp_path / "benchmark.dat",
        workers=2,
    )
    assert captured == {
        "template": "rgmin/lj38-lbfgs",
        "problem": "minimization/lj38",
        "workers": 2,
    }
    assert summary["template"] == "rgmin/lj38-lbfgs"
