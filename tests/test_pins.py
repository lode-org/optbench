from pathlib import Path

from optbench.campaign import list_templates
from optbench.pins import load_registry, pin_errors, row_pin_gaps, template_pin_errors

ROOT = Path(__file__).resolve().parents[1]
PIN = "f15726f1c7983b3656c5dbc2f812e2394e144500"
EARLIER = "9a2e7ff11fa65826f9e847856bee10d3d70b7156"
BUILT = "6d9d629a086080f85687a6ca0a0220cdc3b084e4"


def _registry(root: Path) -> None:
    path = root / "codes" / "registry.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[[code]]",
                'id = "rgmin"',
                'name = "rgmin"',
                f'commit = "{PIN}"',
                "",
                "[[code]]",
                'id = "eon-develop-500350d9"',
                'name = "eOn"',
                'aliases = ["eOn (develop 500350d9)"]',
                'commit = "500350d9fae881f06c7ef194e5868da079c4be5a"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_every_template_names_a_registry_pin():
    assert template_pin_errors(list_templates(ROOT), load_registry(ROOT)) == []


def test_a_template_off_the_pin_is_caught_and_an_alias_selects_by_commit(tmp_path: Path):
    _registry(tmp_path)
    registry = load_registry(tmp_path)
    templates = [
        {"id": "noise/rgmin-cg-s0", "code": "rgmin", "commit": EARLIER},
        {"id": "rgmin/lj38-cg", "code": "rgmin", "commit": PIN[:8]},
        {"id": "eon-develop/htst-al-dimer", "code": "eOn (develop 500350d9)", "commit": BUILT},
        {"id": "eon-client/morse", "code": "eOn"},
    ]
    assert template_pin_errors(templates, registry) == [
        f"template eon-develop/htst-al-dimer names eOn (develop 500350d9) {BUILT}, "
        "registry pins 500350d9fae881f06c7ef194e5868da079c4be5a",
        f"template noise/rgmin-cg-s0 names rgmin {EARLIER}, registry pins {PIN}",
    ]
    registry.append({"id": "eon-develop-6d9d629a", "name": "eOn", "aliases": ["eOn (develop 500350d9)"], "commit": BUILT})
    assert template_pin_errors(templates, registry) == [
        f"template noise/rgmin-cg-s0 names rgmin {EARLIER}, registry pins {PIN}",
    ]


def test_a_row_off_the_pin_fails_until_it_is_listed(tmp_path: Path):
    _registry(tmp_path)
    rows = [
        {"source": "campaign", "path": "results/campaign/noise-rgmin-cg-s0.dat", "code": "rgmin", "commit": EARLIER},
        {"source": "asv", "path": "results/asv/" + "c" * 40, "entry": "bench_noise.NoiseRgminCgS0", "code": "rgmin", "commit": EARLIER},
        {"source": "campaign", "path": "results/campaign/rgmin-lj38-cg.dat", "code": "rgmin", "commit": PIN},
        {"source": "archive", "path": "website/x/benchmark.dat", "code": "rgmin", "commit": EARLIER},
    ]
    gaps = row_pin_gaps(rows, load_registry(tmp_path))
    assert gaps == [
        f"results/asv/{'c' * 40} bench_noise.NoiseRgminCgS0 names rgmin {EARLIER}, registry pins {PIN}",
        f"results/campaign/noise-rgmin-cg-s0.dat names rgmin {EARLIER}, registry pins {PIN}",
    ]
    errors = pin_errors(tmp_path, [], rows)
    assert len(errors) == 2
    assert all(line.startswith("pin gap not in oracle/pin_gaps.txt: ") for line in errors)

    listed = tmp_path / "oracle" / "pin_gaps.txt"
    listed.parent.mkdir()
    listed.write_text("# committed rows keep their recorded label\n" + "\n".join(gaps) + "\n", encoding="utf-8")
    assert pin_errors(tmp_path, [], rows) == []

    rows[0]["commit"] = PIN
    assert pin_errors(tmp_path, [], rows) == [f"oracle/pin_gaps.txt lists a row that is now pinned: {gaps[1]}"]
