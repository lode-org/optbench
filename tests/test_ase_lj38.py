"""The LJ38 ASE reader keeps a blank XYZ comment."""

from optbench.runners.ase_lj38 import read_con


def test_a_blank_comment_keeps_every_atom():
    commented = "2\nenergy\nA 0 0 0\nA 1 0 0\n"
    blank = "2\n\nA 0 0 0\nA 1 0 0\n"
    assert read_con(blank).tolist() == read_con(commented).tolist()
    assert read_con(blank).shape == (2, 3)
