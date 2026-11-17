from pathlib import Path

TESTS = Path(__file__).resolve().parent
FORBIDDEN = tuple("/" + name for name in ("tmp", "home"))


def test_test_files_do_not_pin_tmp_or_home_paths():
    offenders = []
    for path in sorted(TESTS.rglob("test_*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(token in line for token in FORBIDDEN):
                offenders.append(f"{path.relative_to(TESTS.parent)}:{number}: {line.strip()}")
    assert not offenders, "found hard-coded absolute paths in tests:\n" + "\n".join(offenders)
