from optbench.eon_parse import abs_overlap, first_success, flat_floats, saddle_progress, xyz_to_con


def test_xyz_to_con_centers_a_free_cluster():
    text = "2\nenergy -1\nA 0 0 0\nA 2 0 0\n"
    con = xyz_to_con(text)
    assert "\n2\n" in con
    assert con.strip().endswith("1")
    lines = [line for line in con.splitlines() if line.strip()[:1].isdigit() or line.startswith("  ")]
    coords = [line for line in con.splitlines() if "  0     " in line]
    assert len(coords) == 2
    xs = [float(line.split()[0]) for line in coords]
    assert abs((xs[0] + xs[1]) / 2 - 50) < 1e-6


def test_xyz_to_con_keeps_a_blank_comment_line():
    commented = "2\nenergy -1\nA 0 0 0\nA 2 0 0\n"
    blank = "2\n\nA 0 0 0\nA 2 0 0\n"
    assert xyz_to_con(blank) == xyz_to_con(commented)


def test_abs_overlap_does_not_renormalize():
    assert abs_overlap([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs_overlap([0.0, 2.0], [0.0, -0.5]) == 1.0


def test_flat_floats_skips_comments():
    values = flat_floats("#eigenvalue -1\n1 2\n\n3\n")
    assert values == [1.0, 2.0, 3.0]


def test_first_success_returns_the_smallest_k():
    calls = []

    def evaluate(k):
        calls.append(k)
        return k >= 5, {"k": k}

    found, payload = first_success(evaluate, limit=16)
    assert found == 5
    assert payload == {"k": 5}
    assert 5 in calls
    assert 4 in calls or 1 in calls


def test_saddle_progress_matches_done_py_including_not_connected():
    header = "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result\n"
    dashes = "-" * 20 + "\n"
    body = [
        "1 min 0.10000 0.1 10 1 0 good-1",
        "2 min 2.00000 0.1 10 1 0 good-2",
        "3 min 0.20000 0.1 4 1 0 Not Connected",
        "4 min 0.30000 0.1 5 2 9 good-4",
    ]
    # Target 2 so the third good saddle below 1.5 stops the sum.
    # good-2 is above the barrier, Not Connected is not good.
    text = header + dashes + "\n".join(body) + "\n"
    progress = saddle_progress(text, target=2, barrier=1.5)
    assert progress["reached"] is True
    assert progress["qualifying"] == 2
    assert progress["unique_saddles"] == 3
    assert progress["total_searches"] == 4
    assert progress["force_calls"] == (10 + 1) + (10 + 1) + (4 + 1) + (5 + 2)
    assert progress["fraction_connected"] == (4 - 1) / 4


def test_saddle_progress_stops_before_a_short_file():
    text = "h\n-\n1 min 0.1 0 3 4 0 good-1\n"
    progress = saddle_progress(text, target=2, barrier=1.5)
    assert progress["reached"] is False
    assert progress["force_calls"] == 7
    assert progress["qualifying"] == 1
