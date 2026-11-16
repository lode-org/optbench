"""Code page lookup keeps the longest exact alias."""

from optbench.site_pages import _alias_page, _id_page, code_page_name

REGISTRY = [
    {
        "name": "eOn",
        "page": "eOn",
        "aliases": ["eOn", "  ", "eOn (develop 500350d9)"],
    },
    {"name": "Eon", "page": "Eon", "aliases": "Eon"},
    {"id": "rgmin", "name": "rgmin"},
]


def test_the_longest_exact_alias_selects_the_page():
    assert _alias_page("eOn (develop 500350d9)", REGISTRY) == "eOn"
    assert _alias_page("eOn", REGISTRY) == "eOn"
    assert _alias_page("Eon", REGISTRY) == "Eon"
    assert _alias_page("", REGISTRY) is None
    assert code_page_name({"code": "eOn (develop 500350d9)"}, REGISTRY) == "eOn"


def test_an_id_match_ignores_case():
    assert _id_page("RGMIN", REGISTRY) == "rgmin"
    assert _id_page("eon", REGISTRY) == "eOn"
    assert code_page_name({"code": ""}, REGISTRY) == "unknown"
