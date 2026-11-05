"""A force cap stated in template prose matches the TOML that sets it."""

import re
import tomllib
from pathlib import Path

from optbench.campaign import resolve_problem

ROOT = Path(__file__).resolve().parents[1]

NUMBER = r"(\d[\d_,]*)"
PROBLEM_CLAIMS = (
    rf"problem (?:force )?cap(?:,| of| is)? {NUMBER}",
    rf"problem run limit is {NUMBER}",
    rf"{NUMBER} harness calls, which is \S+ max_force_calls",
    rf"convergence\.max_force_calls, {NUMBER}",
    rf"failure_cap {NUMBER} is \S+ convergence",
)
OWN_CLAIMS = (
    rf"harness stop is {NUMBER}",
    rf"harness cap(?: is)? {NUMBER}",
    rf"force-call cap is {NUMBER}",
    rf"failure cap is {NUMBER}",
    rf"harness total (?:at|stops at) {NUMBER}",
    rf"file's failure_cap is {NUMBER}",
    rf"and failure_cap are {NUMBER}",
    rf"force budget is {NUMBER}",
    rf"harness calls under {NUMBER}",
    rf"campaign cap is the .{{0,40}}? budget {NUMBER}",
)
NO_CAP_CLAIMS = (
    r"problem (?:file )?(?:does not state|states no) (?:a )?(?:force )?(?:cap|max_force_calls)",
)

# eOn notes that still call 10000 the lj38_2ts problem cap. The problem
# file says 250000. Drop an entry when its note is corrected.
EON_NOTES = {
    "templates/eon-client/de-2ts-ci-neb.toml",
    "templates/eon-client/de-2ts-ci-swdneb.toml",
    "templates/eon-client/de-2ts-dimer.toml",
    "templates/eon-client/de-2ts-lanczos-improved.toml",
    "templates/eon-client/de-2ts-lanczos.toml",
    "templates/eon-svn/de-2ts-ci-neb.toml",
    "templates/eon-svn/de-2ts-ci-swdneb.toml",
    "templates/eon-svn/de-2ts-dimer.toml",
    "templates/eon-svn/de-2ts-lanczos-improved.toml",
    "templates/eon-svn/de-2ts-lanczos.toml",
    "templates/eon-svn/de-2ts-neb.toml",
}


def _number(text: str) -> int:
    return int(text.replace("_", "").replace(",", ""))


def _strings(node, key: str = ""):
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _strings(value, f"{key}.{name}" if key else name)
    elif isinstance(node, str):
        yield key, node


def _own_caps(template: dict) -> set[int]:
    stop = template.get("stop") or {}
    return {int(value) for value in (stop.get("max_force_calls"), template.get("failure_cap")) if value is not None}


def _problem_cap(template: dict) -> int | None:
    if not template.get("problem"):
        return None
    stated = (resolve_problem(ROOT, template["problem"]).get("convergence") or {}).get("max_force_calls")
    return None if stated is None else int(stated)


def cap_claims(path: Path) -> list[tuple[str, str, int | None, object]]:
    """``(field, phrase, claimed, toml)`` for every cap phrase in one template."""
    template = tomllib.loads(path.read_text(encoding="utf-8"))
    problem_cap = _problem_cap(template)
    own = _own_caps(template)
    found = []
    for field, text in _strings(template):
        for pattern in PROBLEM_CLAIMS:
            for match in re.finditer(pattern, text):
                found.append((field, match.group(0), _number(match.group(1)), problem_cap))
        for pattern in OWN_CLAIMS:
            for match in re.finditer(pattern, text):
                found.append((field, match.group(0), _number(match.group(1)), own))
        for pattern in NO_CAP_CLAIMS:
            for match in re.finditer(pattern, text):
                found.append((field, match.group(0), None, problem_cap))
    return found


def _agrees(claimed: int | None, toml: object) -> bool:
    if isinstance(toml, set):
        return claimed in toml
    return claimed == toml


def test_cap_prose_matches_the_stop_table_and_the_problem_file():
    paths = sorted((ROOT / "templates").glob("**/*.toml"))
    claims = {path: cap_claims(path) for path in paths}
    assert sum(len(found) for found in claims.values()) > 50
    wrong = {
        str(path.relative_to(ROOT)): [claim for claim in found if not _agrees(claim[2], claim[3])]
        for path, found in claims.items()
    }
    wrong = {name: found for name, found in wrong.items() if found}
    assert set(wrong) == EON_NOTES, {name: wrong[name] for name in set(wrong) ^ EON_NOTES if name in wrong}
    for name in EON_NOTES:
        assert all(claim[2] == 10000 and claim[3] == 250000 for claim in wrong[name]), wrong[name]


def test_the_checker_reads_a_wrong_problem_cap(tmp_path: Path):
    path = tmp_path / "template.toml"
    path.write_text(
        'problem = "double-ended-saddle-search/lj38_2ts_separation"\n'
        'note = "The force cap is the problem cap 10000. The harness stop is 250000."\n'
        "[stop]\nmax_force_calls = 250000\n",
        encoding="utf-8",
    )
    found = cap_claims(path)
    assert [(claim[1], _agrees(claim[2], claim[3])) for claim in found] == [
        ("problem cap 10000", False),
        ("harness stop is 250000", True),
    ]
