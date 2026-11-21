"""The eon-classic r2 arm record validates, and the comparison stays the two rgsaddle arms."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

from optbench.board import _skip_unpublished_eon
from optbench.catalog_report import eon_dat, markdown

ROOT = Path(__file__).resolve().parents[1]
STEM = "eon-d9cc6c5be-saddle-pt-classic-r2"


def _catalog():
    script = ROOT / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog_r2", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_r2_provenance_publishes():
    catalog = _catalog()
    path = ROOT / "results" / "campaign" / f"{STEM}-eon-provenance.json"
    document = __import__("json").loads(path.read_text(encoding="utf-8"))
    catalog.validate_provenance(document)
    assert catalog.provenance_publishes(path) is True


def test_a_prior_arm_key_is_refused(tmp_path: Path):
    catalog = _catalog()
    source = ROOT / "results" / "campaign" / f"{STEM}-eon-provenance.json"
    document = __import__("json").loads(source.read_text(encoding="utf-8"))
    document["prior_arm"] = {"connected": 1}
    path = tmp_path / f"{STEM}-eon-provenance.json"
    path.write_text(__import__("json").dumps(document), encoding="utf-8")
    assert catalog.provenance_publishes(path) is False


def test_the_comparison_report_stays_the_two_rgsaddle_arms():
    assert eon_dat(ROOT) is None
    text = markdown(ROOT)
    assert "eon-classic" not in text
    report = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-vs-179e90f2-report.md"
    assert text == report.read_text(encoding="utf-8")


def test_the_board_keeps_a_validated_eon_ledger():
    directory = ROOT / "results" / "campaign"
    assert _skip_unpublished_eon(directory, STEM) is False
    assert _skip_unpublished_eon(directory, "rgsaddle-develop-aa6dd937-saddle-lanczos") is False


# sha256 of lower-cased names that must never appear in a provenance capture.
# The names are not stored here. A word, two words joined by "_" or "-",
# and two words joined by a space are each hashed and compared.
_DENIED_LOWER = frozenset(
    {
        "eaae8cc90cf6749bc1662c2064fad89b5b12ef6dafaebb0f07a110b371f82ce3",
        "3924d5b9a1495d764687218f6f96743982049f60acccd236a7158c890049cb63",
        "2f0dc74858ff2dc6c56f7ebfde4d5e2ef44edc2fe943bbbd1cc727ce9dd93f8f",
        "15cf4d48b3c1ff8b2f4a30af53067e2972e08ec1e85b801b4268e0161abe4ee0",
        "6fdbb0e6fd2afffc947f55b151c7e9f88bc94d795f28e51cccedaeb1d0a9e234",
        "a19d81c80b07b74a850bd1351e87547276b6ffc9fd5a28ca178b0adea76b4058",
    }
)
# Matched with its case, because the lower-case word is ordinary English.
_DENIED_EXACT = frozenset({"0a8b0800d768d14b08cb708665f7fa5bfcd81b8c6e58bbe86ab31b26839dbcdd"})
_CANARY = hashlib.sha256(b"denylist-canary").hexdigest()
_WORD = re.compile(r"[A-Za-z0-9]+")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def denied_names(text: str, extra: frozenset[str] = frozenset()) -> list[str]:
    """Digests of denied names found in ``text``."""
    lower = _DENIED_LOWER | extra
    words = _WORD.findall(text)
    found: set[str] = set()
    for index, word in enumerate(words):
        candidates = [word.lower()]
        if index + 1 < len(words):
            pair = (word.lower(), words[index + 1].lower())
            candidates += [f"{pair[0]}_{pair[1]}", f"{pair[0]}-{pair[1]}", f"{pair[0]} {pair[1]}"]
        for candidate in candidates:
            digest = _digest(candidate)
            if digest in lower:
                found.add(digest)
        exact = _digest(word)
        if exact in _DENIED_EXACT:
            found.add(exact)
    return sorted(found)


def test_the_denylist_matcher_finds_a_canary(tmp_path: Path):
    capture = tmp_path / "x-eon-provenance.json"
    capture.write_text('{"run": {"env": {"DENYLIST_CANARY": "1"}}}\n', encoding="utf-8")
    assert denied_names(capture.read_text(encoding="utf-8"), frozenset({_CANARY})) == [_CANARY]


def test_committed_provenance_rejects_internal_names():
    captures = sorted((ROOT / "results").rglob("*provenance*.json"))
    assert captures
    for path in captures:
        assert denied_names(path.read_text(encoding="utf-8")) == [], path
