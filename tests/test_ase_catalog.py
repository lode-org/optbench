import tomllib
from pathlib import Path

from optbench.ase_catalog import OPTIMIZERS, all_specs, write_all
from optbench.campaign import load_template
from optbench.site import campaign_problem

ROOT = Path(__file__).parents[1]
PROBLEMS = {
    "minimization/lj38",
    "minimization/morse-bulk",
    "tsopt/LJ38",
    "tsopt/pt-island",
    "neb/Pt-heptamer",
    "double-ended-saddle-search/lj38",
    "double-ended-saddle-search/lj38_2ts_separation",
    "double-ended-saddle-search/pt-island",
    "saddle-search/pt-island",
}


def test_catalog_covers_every_problem_and_both_profiles():
    specs = all_specs()
    ids = [spec["id"] for spec in specs]
    assert len(ids) == len(set(ids))
    by_problem: dict[str, set[str]] = {}
    for spec in specs:
        by_problem.setdefault(spec["problem"], set()).add(spec["profile"])
    assert set(by_problem) == PROBLEMS
    for problem, profiles in by_problem.items():
        if problem in {
            "minimization/lj38",
            "minimization/morse-bulk",
            "tsopt/LJ38",
            "tsopt/pt-island",
            "saddle-search/pt-island",
        }:
            assert profiles == {"default"}
        else:
            assert profiles == {"default", "tuned"}
    lj = [spec for spec in specs if spec["problem"] == "minimization/lj38"]
    assert {spec["algorithm"] for spec in lj} == set(OPTIMIZERS) | {"Sella"}
    dimer = [spec for spec in specs if spec["problem"] == "saddle-search/pt-island"]
    assert dimer and all(spec["ranked"] is False for spec in dimer)


def test_templates_match_the_catalog_and_registry():
    specs = all_specs()
    written = write_all(ROOT)
    assert len(written) == len(specs)
    for spec in specs:
        loaded = load_template(ROOT, spec["id"])
        assert loaded["code"] == spec["code"]
        assert loaded["code_version"] == spec["code_version"]
        assert loaded["profile"] == spec["profile"]
        assert loaded["problem"] == spec["problem"]
        assert loaded["optimizer"]["name"]
        assert loaded["stop"]["max_force_calls"] > 0
    default_lbfgs = load_template(ROOT, "ase/lj38-lbfgs-default")
    assert default_lbfgs["optimizer"]["parameters"]["maxstep"] == 0.1
    sella = load_template(ROOT, "sella/lj38-ts-default")
    assert sella["optimizer"]["parameters"]["order"] == 1
    assert "delta0" not in sella["optimizer"]["parameters"]
    assert sella["reducer"] == "all"
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    names = {row["id"] for row in registry["code"]}
    assert {"eon", "ase", "sella"} <= names
    ase = next(row for row in registry["code"] if row["id"] == "ase")
    sella_code = next(row for row in registry["code"] if row["id"] == "sella")
    assert ase["commit"].startswith("f27c0005ae6a")
    assert sella_code["commit"].startswith("6cfb60cdb100")
    assert "0.1.1" in sella_code["note"]
    assert "legacy" in sella_code["note"]
    templates = {row["id"] for row in registry["template"]}
    assert {spec["id"] for spec in specs} <= templates
    assert campaign_problem("sella-2.6.0-ts-lj38") == "tsopt/LJ38"
