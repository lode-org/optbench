"""Global optimisation problems share one force-call cap."""

import tarfile
from pathlib import Path

from optbench.campaign import geometry_archive
from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]


def test_globalopt_problems_declare_a_force_call_cap():
    """A missing cap lets each cell pick its own.

    Saddle search, accelerated MD, hTST, and mlopt omit the key on
    purpose. This gate covers globalopt only.
    """
    problems = load_all(ROOT / "problems")
    missing = [
        ident
        for ident, problem in sorted(problems.items())
        if problem["suite"] == "globalopt"
        and (problem.get("convergence") or {}).get("max_force_calls") in (None, "")
    ]
    assert missing == []
    caps = {
        ident: (problem.get("convergence") or {}).get("max_force_calls")
        for ident, problem in problems.items()
        if problem["suite"] == "globalopt"
    }
    assert caps == {
        "globalopt/lj100-binary": 2000000,
        "globalopt/lj150": 2000000,
        "globalopt/lj38": 2000000,
        "globalopt/lj75": 2000000,
        "globalopt/lj98": 2000000,
        "globalopt/tip4p10": 2000000,
        "globalopt/tip4p15": 2000000,
        "globalopt/tip4p20": 2000000,
    }


def _con_components(text: str) -> list[tuple[str, list[tuple[float, float, float]]]]:
    lines = text.splitlines()
    components: list[tuple[str, list[tuple[float, float, float]]]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("Coordinates of Component"):
            symbol = lines[index - 1].strip()
            index += 1
            rows: list[tuple[float, float, float]] = []
            while index < len(lines) and lines[index][:1].isspace():
                parts = lines[index].split()
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                index += 1
            components.append((symbol, rows))
            continue
        index += 1
    return components


def test_lj100_binary_starts_have_42_sigma_1_and_58_sigma_1_3_atoms():
    """Each start file is 42 atoms of sigma 1.0 and 58 atoms of sigma 1.3.

    Component 1 is species B and component 2 is species A, the order
    in the archived LAMMPS pair coefficients.
    """
    problem = load_all(ROOT / "problems")["globalopt/lj100-binary"]
    potential = problem["potential"]
    assert potential["n_A"] == 42
    assert potential["sigma_AA"] == 1.0
    assert potential["symbol_A"] == "C"
    assert potential["n_B"] == 58
    assert potential["sigma_BB"] == 1.3
    assert potential["symbol_B"] == "H"
    assert potential["energy_minimum"] == -604.796307
    assert problem["convergence"]["energy_above_minimum"] == 0.001
    assert problem["convergence"]["max_force_calls"] == 2000000
    archive = geometry_archive(ROOT, problem)
    with tarfile.open(archive) as packed:
        members = {
            member.name: member
            for member in packed.getmembers()
            if member.isfile()
        }
        cons = sorted(name for name in members if name.endswith(".con"))
        assert len(cons) == 100
        for name in cons:
            text = packed.extractfile(members[name]).read().decode("utf-8")
            components = _con_components(text)
            assert [symbol for symbol, _rows in components] == [
                potential["symbol_B"],
                potential["symbol_A"],
            ]
            assert len(components[0][1]) == potential["n_B"]
            assert len(components[1][1]) == potential["n_A"]
            stem = Path(name).stem
            coords_name = f"{Path(name).parent.as_posix()}/coords.{stem}"
            coords = [
                tuple(float(part) for part in line.split())
                for line in packed.extractfile(members[coords_name])
                .read()
                .decode("utf-8")
                .splitlines()
                if line.strip()
            ]
            assert coords == components[0][1] + components[1][1]
