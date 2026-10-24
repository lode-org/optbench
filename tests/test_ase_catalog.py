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
    for profiles in by_problem.values():
        assert profiles == {"default", "adjusted"}
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
    tuned = load_template(ROOT, "ase/lj38-lbfgs-tuned")
    assert tuned["optimizer"]["parameters"]["memory"] == 100
    assert abs(float(tuned["optimizer"]["parameters"]["alpha"]) - 1000.0) < 1e-9
    sella = load_template(ROOT, "sella/lj38-ts-default")
    assert sella["optimizer"]["parameters"]["order"] == 1
    assert "delta0" not in sella["optimizer"]["parameters"]
    assert sella["reducer"] == "all"
    tuned_sella = load_template(ROOT, "sella/lj38-ts-tuned")
    assert tuned_sella["optimizer"]["parameters"]["delta0"] == 1.3e-3
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    names = {row["id"] for row in registry["code"]}
    assert {"eon", "ase", "sella"} <= names
    ase = next(row for row in registry["code"] if row["id"] == "ase")
    sella_code = next(row for row in registry["code"] if row["id"] == "sella")
    assert ase["commit"].startswith("f27c0005ae6a")
    assert sella_code["commit"].startswith("6cfb60cdb100")
    assert "0.1.1" in sella_code["note"]
    assert "different pin" in sella_code["note"]
    templates = {row["id"] for row in registry["template"]}
    assert {spec["id"] for spec in specs} <= templates
    assert campaign_problem("sella-2.6.0-ts-lj38") == "tsopt/LJ38"


def test_write_all_keeps_a_code_pin_after_the_generated_block(tmp_path):
    registry = tmp_path / "codes" / "registry.toml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "\n".join(
            [
                "[[code]]",
                'id = "ase"',
                "",
                "# generated ASE and Sella templates",
                "",
                "[[template]]",
                'id = "stale"',
                "",
                "[[code]]",
                'id = "gpridge"',
                'name = "GPRidge"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_all(tmp_path)
    text = registry.read_text(encoding="utf-8")
    loaded = tomllib.loads(text)
    assert [row["id"] for row in loaded["code"]] == ["ase", "gpridge"]
    assert "stale" not in text
    assert any(row["id"] == "ase/lj38-bfgs-default" for row in loaded["template"])
    assert text.rstrip().endswith('name = "GPRidge"')
