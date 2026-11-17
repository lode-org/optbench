"""Geometry readers and pair potentials behind the band campaigns.

A bad archive has to stop the run with a message that names the file or the
count at fault. The potentials are checked against the pair formulas they are
documented to implement, and a periodic cell against the same geometry laid
out by hand.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest

from optbench.pes import (
    LJ_CUTOFF,
    MORSE_CUTOFF,
    MORSE_D,
    MORSE_R0,
    energy_weighted_springs,
    evaluate,
    interpolate_free,
    lennard_jones,
    load_pairs,
    morse_pt,
    read_con,
    read_xyz,
)

SUBSTRATE = [
    (0.0, 0.0, 0.0, 1),
    (2.77, 0.0, 0.0, 1),
    (1.385, 2.399, 0.0, 1),
    (1.385, 0.8, 2.3, 0),
]


def _con(blocks, *, box=(11.08, 9.6, 30.0), angles=(90.0, 90.0, 90.0)) -> str:
    lines = [
        "Generated for a test",
        "0.0000 TIME",
        " ".join(str(length) for length in box),
        " ".join(str(angle) for angle in angles),
        "0 0",
        "0 0 0",
        str(len(blocks)),
        " ".join(str(len(rows)) for _symbol, _mass, rows in blocks),
        " ".join(str(mass) for _symbol, mass, _rows in blocks),
    ]
    index = 0
    for number, (symbol, _mass, rows) in enumerate(blocks, start=1):
        lines.extend([symbol, f"Coordinates of Component {number}"])
        for x, y, z, flag in rows:
            lines.append(f"{x} {y} {z} {flag} {index}")
            index += 1
    return "\n".join(lines) + "\n"


def _island(shift=(0.0, 0.0, 0.0), **kwargs) -> str:
    rows = [(x, y, z, flag) for x, y, z, flag in SUBSTRATE]
    x, y, z, flag = rows[3]
    rows[3] = (round(x + shift[0], 6), round(y + shift[1], 6), round(z + shift[2], 6), flag)
    return _con([("Pt", 195.084, rows)], **kwargs)


def _xyz(rows, comment="") -> str:
    body = [f"{len(rows)}", comment]
    body.extend(f"H {x} {y} {z}" for x, y, z in rows)
    return "\n".join(body) + "\n"


def _archive(tmp_path, files, name="geometries.tgz"):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as packed:
        folder = tarfile.TarInfo("sets")
        folder.type = tarfile.DIRTYPE
        packed.addfile(folder)
        for member, text in files.items():
            raw = text.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(raw)
            packed.addfile(info, io.BytesIO(raw))
    return path


def test_a_con_frame_gives_positions_the_free_mask_and_the_box():
    coords, free, box = read_con(_island())
    assert coords.shape == (4, 3)
    assert coords[3].tolist() == [1.385, 0.8, 2.3]
    assert free.tolist() == [False, False, False, True]
    assert box.tolist() == [11.08, 9.6, 30.0]


def test_a_con_frame_with_no_box_has_none():
    _coords, _free, box = read_con(_island(box=(0.0, 0.0, 0.0)))
    assert box is None


def test_a_con_frame_with_two_species_keeps_their_atoms_in_file_order():
    text = _con(
        [
            ("Pt", 195.084, [(0.0, 0.0, 0.0, 1), (2.77, 0.0, 0.0, 0)]),
            ("Cu", 63.546, [(1.385, 2.399, 2.0, 0)]),
        ]
    )
    coords, free, _box = read_con(text)
    assert coords.tolist() == [[0.0, 0.0, 0.0], [2.77, 0.0, 0.0], [1.385, 2.399, 2.0]]
    assert free.tolist() == [False, True, True]


def test_a_con_frame_that_is_cut_off_is_refused_with_the_part_that_is_missing():
    text = _island()
    lines = text.splitlines()
    with pytest.raises(ValueError, match="con file is too short"):
        read_con("\n".join(lines[:9]))
    with pytest.raises(ValueError, match="con file ended inside coordinates"):
        read_con("\n".join(lines[:-1]))
    two_species = _con(
        [
            ("Pt", 195.084, [(0.0, 0.0, 0.0, 1), (2.77, 0.0, 0.0, 0)]),
            ("Cu", 63.546, [(1.385, 2.399, 2.0, 0)]),
        ]
    ).splitlines()
    with pytest.raises(ValueError, match="con file ended before a species name"):
        read_con("\n".join(two_species[:13]))


def test_a_con_frame_with_a_malformed_header_is_refused():
    with pytest.raises(ValueError, match="three lengths and three angles"):
        read_con(_island(box=(11.08, 9.6)))
    with pytest.raises(ValueError, match="con cell is not orthorhombic"):
        read_con(_island(angles=(90.0, 90.0, 120.0)))
    lines = _island().splitlines()
    lines[6] = "2"
    with pytest.raises(ValueError, match="type counts do not match the type line"):
        read_con("\n".join(lines))


def test_a_con_atom_without_its_fixed_flag_is_named_by_line_number():
    lines = _island().splitlines()
    lines[12] = "1.385 0.8 2.3"
    with pytest.raises(ValueError, match="con coordinate 13 has no fixed flag"):
        read_con("\n".join(lines))


def test_an_xyz_block_reads_with_or_without_a_comment_line():
    rows = [(0.0, 0.0, 0.0), (0.74, 0.0, 0.0)]
    assert read_xyz(_xyz(rows)).tolist() == [list(row) for row in rows]
    assert read_xyz(_xyz(rows, comment="energy -1.0")).tolist() == [list(row) for row in rows]


def test_an_xyz_block_that_is_short_of_atoms_or_fields_is_refused():
    with pytest.raises(ValueError, match="xyz announced 3 atoms and held 2"):
        read_xyz("3\n\nH 0 0 0\nH 0.74 0 0\n")
    with pytest.raises(ValueError, match="xyz coordinate is missing x y z"):
        read_xyz("2\n\nH 0 0 0\nH 0.74 0\n")


def test_xyz_pairs_are_named_by_index_and_leave_every_atom_free(tmp_path):
    files = {
        "sets/start_00.xyz": _xyz([(0.0, 0.0, 0.0), (0.74, 0.0, 0.0)]),
        "sets/end_00.xyz": _xyz([(0.0, 0.0, 0.0), (0.90, 0.0, 0.0)]),
        "sets/start_01.xyz": _xyz([(0.0, 0.0, 0.0), (0.74, 0.1, 0.0)]),
        "sets/end_01.xyz": _xyz([(0.0, 0.0, 0.1), (1.00, 0.0, 0.0)]),
    }
    pairs = load_pairs(_archive(tmp_path, files), "xyz-pairs", 2)
    assert [pair.name for pair in pairs] == ["pair_00", "pair_01"]
    assert pairs[0].product[1].tolist() == [0.9, 0.0, 0.0]
    assert pairs[1].reactant[1].tolist() == [0.74, 0.1, 0.0]
    assert pairs[0].free.tolist() == [True, True]
    assert pairs[0].box is None


def test_xyz_pairs_with_a_lone_endpoint_or_unequal_lengths_are_refused(tmp_path):
    lone = {
        "start_00.xyz": _xyz([(0.0, 0.0, 0.0)]),
        "end_00.xyz": _xyz([(0.0, 0.0, 1.0)]),
        "start_01.xyz": _xyz([(0.0, 0.0, 0.0)]),
    }
    with pytest.raises(ValueError, match=r"xyz pairs are missing an endpoint at \[1\]"):
        load_pairs(_archive(tmp_path, lone, "lone.tgz"), "xyz-pairs", None)
    unequal = {
        "start_00.xyz": _xyz([(0.0, 0.0, 0.0)]),
        "end_00.xyz": _xyz([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
    }
    with pytest.raises(ValueError, match="pair 0 endpoints differ in length"):
        load_pairs(_archive(tmp_path, unequal, "unequal.tgz"), "xyz-pairs", None)


def test_the_pair_count_has_to_match_what_the_problem_says(tmp_path):
    files = {
        "start_00.xyz": _xyz([(0.0, 0.0, 0.0)]),
        "end_00.xyz": _xyz([(0.0, 0.0, 1.0)]),
    }
    archive = _archive(tmp_path, files)
    assert len(load_pairs(archive, "xyz-pairs", None)) == 1
    with pytest.raises(ValueError, match="xyz-pairs has 1 pairs, the problem says 3"):
        load_pairs(archive, "xyz-pairs", 3)
    empty = _archive(tmp_path, {"notes.txt": "nothing here"}, "empty.tgz")
    with pytest.raises(ValueError, match="xyz-pairs has no pairs"):
        load_pairs(empty, "xyz-pairs", None)
    with pytest.raises(ValueError, match="xyz-pairs has 0 pairs, the problem says 4"):
        load_pairs(empty, "xyz-pairs", 4)


def test_an_unknown_geometry_kind_is_named(tmp_path):
    archive = _archive(tmp_path, {"notes.txt": "nothing here"})
    with pytest.raises(ValueError, match="unknown geometry kind minima-by-name"):
        load_pairs(archive, "minima-by-name", None)


def test_minima_pairs_come_from_a_and_b_in_each_folder_in_sorted_order(tmp_path):
    files = {
        "sets/m02/a.con": _island(),
        "sets/m02/b.con": _island(shift=(1.3, 0.2, -0.1)),
        "sets/m01/a.con": _island(shift=(0.1, 0.0, 0.0)),
        "sets/m01/b.con": _island(shift=(-1.3, 0.2, 0.0)),
        "sets/orphan/a.con": _island(),
        "sets/m01/readme.txt": "not a geometry",
        "sets/m01/c.con": _island(),
    }
    pairs = load_pairs(_archive(tmp_path, files), "con-minima-pairs", 2)
    assert [pair.name for pair in pairs] == ["m01", "m02"]
    assert pairs[0].reactant[3].tolist() == [1.485, 0.8, 2.3]
    assert pairs[0].product[3].tolist() == [0.085, 1.0, 2.3]
    assert pairs[1].free.tolist() == [False, False, False, True]
    assert pairs[1].box.tolist() == [11.08, 9.6, 30.0]


def test_minima_whose_endpoints_differ_in_atoms_or_freedom_are_refused(tmp_path):
    shorter = _con([("Pt", 195.084, SUBSTRATE[:3])])
    with pytest.raises(ValueError, match="sets/m03 endpoints do not match"):
        load_pairs(
            _archive(tmp_path, {"sets/m03/a.con": _island(), "sets/m03/b.con": shorter}),
            "con-minima-pairs",
            None,
        )
    pinned = [(*row[:3], 1) for row in SUBSTRATE]
    with pytest.raises(ValueError, match="sets/m04 endpoints do not match"):
        load_pairs(
            _archive(
                tmp_path,
                {"sets/m04/a.con": _island(), "sets/m04/b.con": _con([("Pt", 195.084, pinned)])},
                "pinned.tgz",
            ),
            "con-minima-pairs",
            None,
        )


def test_reactant_products_are_ordered_by_product_number_not_by_name(tmp_path):
    files = {
        "run/reactant.con": _island(),
        "run/product_10.con": _island(shift=(1.3, 0.2, -0.1)),
        "run/product_2.con": _island(shift=(-1.3, 0.2, 0.0)),
        "run/product_x.txt": "ignored",
    }
    pairs = load_pairs(_archive(tmp_path, files), "con-reactant-products", 2)
    assert [pair.name for pair in pairs] == ["product_2", "product_10"]
    assert pairs[0].product[3].tolist() == [0.085, 1.0, 2.3]
    assert pairs[1].reactant[3].tolist() == [1.385, 0.8, 2.3]


def test_a_product_that_moves_a_pinned_atom_is_refused_beyond_the_tolerance(tmp_path):
    moved = [(0.0, 0.0, 0.0, 1), (2.77, 0.0, 0.01, 1), (1.385, 2.399, 0.0, 1), (1.385, 0.8, 2.3, 0)]
    files = {"reactant.con": _island(), "product_2.con": _con([("Pt", 195.084, moved)])}
    with pytest.raises(ValueError, match=r"product_2 moves a fixed atom by 0\.01"):
        load_pairs(_archive(tmp_path, files), "con-reactant-products", None)
    within = [(0.0, 0.0, 0.0, 1), (2.77, 0.0, 0.00005, 1), (1.385, 2.399, 0.0, 1), (1.385, 0.8, 2.3, 0)]
    files = {"reactant.con": _island(), "product_2.con": _con([("Pt", 195.084, within)])}
    pairs = load_pairs(_archive(tmp_path, files, "within.tgz"), "con-reactant-products", 1)
    assert pairs[0].name == "product_2"


def test_a_product_that_has_another_atom_set_than_the_reactant_is_refused(tmp_path):
    files = {
        "reactant.con": _island(),
        "product_3.con": _con([("Pt", 195.084, SUBSTRATE[:3])]),
    }
    with pytest.raises(ValueError, match="product_3 does not match the reactant"):
        load_pairs(_archive(tmp_path, files), "con-reactant-products", None)


def test_the_two_potentials_are_selected_by_name_and_an_unknown_one_is_named():
    coords = np.array([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0], [1.4, 2.5, 0.0]])
    box = np.array([8.0, 8.0, 8.0])
    assert evaluate("morse_pt", coords, box)[0] == morse_pt(coords, box)[0]
    assert np.array_equal(evaluate("morse_pt", coords, box)[1], morse_pt(coords, box)[1])
    assert evaluate("lj", coords, box)[0] == lennard_jones(coords, None)[0]
    assert np.array_equal(evaluate("lj", coords, box)[1], lennard_jones(coords, None)[1])
    with pytest.raises(ValueError, match="unknown potential eam"):
        evaluate("eam", coords, box)


def test_morse_is_at_its_well_depth_with_no_force_at_the_equilibrium_distance():
    pair = np.array([[0.0, 0.0, 0.0], [MORSE_R0, 0.0, 0.0]])
    energy, gradient = morse_pt(pair)
    assert energy == pytest.approx(-MORSE_D, abs=1e-4)
    assert np.abs(gradient).max() < 1e-12


def test_a_pair_past_the_cutoff_has_neither_energy_nor_force():
    morse = np.array([[0.0, 0.0, 0.0], [MORSE_CUTOFF + 0.1, 0.0, 0.0]])
    energy, gradient = morse_pt(morse)
    assert energy == 0.0
    assert not gradient.any()
    lj = np.array([[0.0, 0.0, 0.0], [LJ_CUTOFF + 1.0, 0.0, 0.0]])
    energy, gradient = lennard_jones(lj)
    assert energy == 0.0
    assert not gradient.any()


def test_the_shifted_morse_energy_reaches_zero_at_the_cutoff():
    inside = np.array([[0.0, 0.0, 0.0], [MORSE_CUTOFF - 1e-9, 0.0, 0.0]])
    assert morse_pt(inside)[0] == pytest.approx(0.0, abs=1e-9)


def test_lennard_jones_is_zero_at_sigma_and_has_its_minimum_depth_at_the_pair_minimum():
    at_sigma = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert lennard_jones(at_sigma)[0] == pytest.approx(0.0, abs=1e-12)
    minimum = np.array([[0.0, 0.0, 0.0], [2.0 ** (1.0 / 6.0), 0.0, 0.0]])
    energy, gradient = lennard_jones(minimum)
    assert energy == pytest.approx(-1.0, abs=1e-12)
    assert np.abs(gradient).max() < 1e-10


def test_a_periodic_cell_sees_the_nearest_image_of_each_atom():
    box = np.array([8.691, 8.691, 30.0])
    across = np.array([[0.0, 0.0, 0.0], [5.794, 0.0, 0.0]])
    unwrapped = np.array([[0.0, 0.0, 0.0], [5.794 - 8.691, 0.0, 0.0]])
    energy, gradient = morse_pt(across, box)
    expected_energy, expected_gradient = morse_pt(unwrapped)
    assert energy == pytest.approx(expected_energy, rel=1e-12)
    assert gradient == pytest.approx(expected_gradient, abs=1e-12)
    assert energy < morse_pt(across)[0] - 0.1


def test_spring_constants_rise_from_the_floor_to_the_ceiling_with_the_image_energy():
    energies = np.array([0.0, 0.2, 0.9, 0.4, 0.1])
    constants = energy_weighted_springs(energies)
    reference, highest = 0.1, 0.9

    def weighted(higher):
        alpha = (highest - higher) / (highest - reference)
        return (1.0 - alpha) * 9.7 + alpha * 0.97

    assert constants.tolist() == pytest.approx([weighted(0.2), 9.7, 9.7, weighted(0.4)], rel=1e-12)


def test_a_segment_below_the_higher_endpoint_keeps_the_floor_constant():
    constants = energy_weighted_springs(np.array([0.0, -0.4, 0.5, 1.0, 0.3]), k_min=1.0, k_max=8.0)
    alpha = (1.0 - 0.5) / (1.0 - 0.3)
    assert constants.tolist() == pytest.approx([1.0, (1.0 - alpha) * 8.0 + alpha * 1.0, 8.0, 8.0], rel=1e-12)


def test_a_flat_path_has_the_floor_constant_everywhere():
    assert energy_weighted_springs(np.array([0.3, 0.3, 0.3, 0.3])).tolist() == [0.97, 0.97, 0.97]


def test_springs_need_one_energy_per_image_and_at_least_two_images():
    with pytest.raises(ValueError, match="springs need one energy per image"):
        energy_weighted_springs(np.array([0.5]))
    with pytest.raises(ValueError, match="springs need one energy per image"):
        energy_weighted_springs(np.zeros((2, 3)))


def test_a_linear_path_starts_at_the_reactant_and_ends_at_the_product_free_atoms():
    reactant = np.array(SUBSTRATE)[:, :3]
    product = reactant.copy()
    product[3] += [1.2, 0.0, -0.3]
    free = np.array([False, False, False, True])
    frames = interpolate_free(reactant, product, free, 3, None)
    assert frames.shape == (5, 3)
    assert frames[0].tolist() == reactant[3].tolist()
    assert frames[-1] == pytest.approx(product[3].tolist(), abs=1e-12)
    assert frames[2] == pytest.approx(((reactant[3] + product[3]) / 2.0).tolist(), abs=1e-12)


def test_a_path_through_a_periodic_cell_takes_the_short_way_round():
    box = np.array([11.0, 11.0, 30.0])
    reactant = np.array([[0.2, 5.0, 2.0]])
    product = np.array([[10.9, 5.0, 2.0]])
    frames = interpolate_free(reactant, product, np.array([True]), 1, box)
    assert frames[:, 0] == pytest.approx([0.2, 0.05, -0.1], abs=1e-12)
    far = interpolate_free(reactant, product, np.array([True]), 1, None)
    assert far[1, 0] == pytest.approx(5.55, abs=1e-12)


def test_a_band_with_no_interior_image_is_refused():
    with pytest.raises(ValueError, match="a band needs at least one interior image"):
        interpolate_free(np.zeros((1, 3)), np.ones((1, 3)), np.array([True]), 0, None)
