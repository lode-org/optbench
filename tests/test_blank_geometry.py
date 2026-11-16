"""A blank XYZ comment or con header line stays in place."""

import pytest

from optbench.amd import parse_con as amd_parse
from optbench.ase_sella import _read_xyz
from optbench.pes import read_con as pes_read
from optbench.pes import read_xyz
from optbench.potentials import read_con as pot_read
from optbench.runners.gmin_paper import _coords_text
from optbench.runners.optking_ts import parse_con as optking_parse
from optbench.surfaces import read_con as surf_read

_XYZ = "2\nenergy\nA 0 0 0\nA 1 2 3\n"
_XYZ_BLANK = "2\n\nA 0 0 0\nA 1 2 3\n"
_CON = """\
seed

10 11 12
90 90 90


1
2
1.0
Pt
Coordinates
1 2 3 0 0
4 5 6 0 1
"""


def test_xyz_readers_keep_a_blank_comment():
    assert read_xyz(_XYZ_BLANK).tolist() == read_xyz(_XYZ).tolist()
    assert _coords_text(_XYZ_BLANK.encode(), "xyz") == _coords_text(_XYZ.encode(), "xyz")
    assert _coords_text(_XYZ_BLANK.encode(), "xyz") == "0 0 0\n1 2 3\n"


def test_a_blank_coordinate_line_is_not_skipped():
    raw = b"2\nenergy\nA 0 0 0\n\nA 1 2 3\n"
    with pytest.raises(ValueError):
        _coords_text(raw, "xyz")


def test_sella_keeps_a_blank_xyz_comment(tmp_path):
    path = tmp_path / "blank.xyz"
    path.write_text(_XYZ_BLANK, encoding="utf-8")
    atoms = _read_xyz(path)
    assert atoms.get_positions().tolist() == [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]


def test_sella_does_not_skip_a_blank_coordinate_line(tmp_path):
    path = tmp_path / "gap.xyz"
    path.write_text("2\nnote\nQ 0 0 0\n\nQ 1 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_xyz(path)


def test_con_readers_keep_a_blank_header_line():
    pes_pos, _free, pes_box = pes_read(_CON)
    surf_pos, _fixed, surf_box = surf_read(_CON)
    pot_pos, pot_box = pot_read(_CON)
    amd_pos, _mask = amd_parse(_CON)
    optking = optking_parse(_CON)
    assert pes_box.tolist() == [10.0, 11.0, 12.0]
    assert surf_box.tolist() == [10.0, 11.0, 12.0]
    assert pot_box.tolist() == [10.0, 11.0, 12.0]
    assert optking["box"] == [10.0, 11.0, 12.0]
    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert pes_pos.tolist() == expected
    assert surf_pos.tolist() == expected
    assert pot_pos.tolist() == expected
    assert amd_pos.tolist() == expected
    assert optking["positions"] == expected
