import shutil
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


def test_templates_match_the_catalog_and_registry(tmp_path):
    specs = all_specs()
    (tmp_path / "codes").mkdir()
    shutil.copy(ROOT / "codes" / "registry.toml", tmp_path / "codes" / "registry.toml")
    for folder in ("ase", "sella"):
        shutil.copytree(ROOT / "templates" / folder, tmp_path / "templates" / folder)
    written = write_all(tmp_path)
    assert len(written) == len(specs)
    committed = {
        path.relative_to(ROOT) for folder in ("ase", "sella") for path in (ROOT / "templates" / folder).glob("*.toml")
    }
    assert {path.relative_to(tmp_path) for path in written} == committed
    for path in [tmp_path / "codes" / "registry.toml", *written]:
        relative = path.relative_to(tmp_path)
        assert path.read_bytes() == (ROOT / relative).read_bytes(), relative
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


def test_write_all_keeps_the_block_after_the_generated_templates(tmp_path):
    codes = tmp_path / "codes"
    codes.mkdir()
    handwritten = (
        "[[code]]\n"
        'id = "gpridge"\n'
        'name = "GPRidge"\n'
        'role = "Gaussian-process band and saddle search"\n'
        "\n"
        "# a later pin stays with the block above\n"
        "[[code]]\n"
        'id = "after"\n'
        'name = "After"\n'
    )
    registry = codes / "registry.toml"
    registry.write_text(
        '[[code]]\nid = "ase"\nname = "ASE"\n\n'
        "# generated ASE and Sella templates\n"
        "\n"
        "[[template]]\n"
        'id = "stale"\n'
        'code = "ase"\n'
        "\n"
        + handwritten,
        encoding="utf-8",
    )
    write_all(tmp_path)
    text = registry.read_text(encoding="utf-8")
    assert 'id = "stale"' not in text
    assert handwritten.strip() in text
    assert text.index("# generated ASE and Sella templates") < text.index('id = "gpridge"')
    specs = all_specs()
    assert f'id = "{specs[0]["id"]}"' in text
    again = text
    write_all(tmp_path)
    assert registry.read_text(encoding="utf-8") == again
