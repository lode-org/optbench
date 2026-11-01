"""Fork pull-request selection and inert comparison comments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "poisoned-compare.txt"
FORK_HEAD = "6e6c92175f48551381b6e8d5c136c9e0ded02afb"


def _load():
    spec = importlib.util.spec_from_file_location(
        "asv_comment", ROOT / "ci" / "asv_comment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comment = _load()


def _pull(number: int, head: str, base: str = "a" * 40) -> dict:
    return {"number": number, "head": {"sha": head}, "base": {"sha": base}}


def _assert_inert(body: str, report: str) -> None:
    outside, inside = comment.fenced_parts(body)
    fence = comment.fence_for(report)
    assert fence in body
    longest = max((len(run) for run in __import__("re").findall(r"`+", report)), default=0)
    assert len(fence) == max(3, longest) + 1
    assert report.strip("\n") in inside or report in inside or report.replace("\r\n", "\n") in inside
    for needle in ("@", "![", "<img", "<a", "http://", "https://", "~~~~", "```"):
        if needle in report:
            assert needle not in outside
            assert needle in inside


def test_a_fork_payload_with_one_open_match_is_chosen():
    seen = {"listed": False}

    def list_open():
        seen["listed"] = True
        return [_pull(7, FORK_HEAD, "b" * 40)]

    def get_pull(number):
        assert number == 7
        return _pull(7, FORK_HEAD, "b" * 40)

    number, body = comment.plan_comment(
        {"head_sha": FORK_HEAD, "pull_requests": [], "event": "pull_request"},
        "same\n",
        list_open=list_open,
        get_pull=get_pull,
    )
    assert seen["listed"] is True
    assert number == 7
    assert f"Base `{'b' * 40}`." in body
    assert f"Head `{FORK_HEAD}`." in body


def test_a_named_pull_request_is_used_without_listing():
    def list_open():
        raise AssertionError("the open list is not consulted")

    def get_pull(number):
        assert number == 4
        return _pull(4, "c" * 40, "d" * 40)

    number, _body = comment.plan_comment(
        {
            "head_sha": "c" * 40,
            "pull_requests": [_pull(4, "c" * 40, "d" * 40)],
        },
        "same\n",
        list_open=list_open,
        get_pull=get_pull,
    )
    assert number == 4


def test_zero_or_several_matches_post_nothing():
    calls = []

    def get_pull(number):
        calls.append(number)
        return _pull(number, FORK_HEAD)

    with pytest.raises(comment.CommentError):
        comment.plan_comment(
            {"head_sha": FORK_HEAD, "pull_requests": []},
            "same\n",
            list_open=lambda: [],
            get_pull=get_pull,
        )
    with pytest.raises(comment.CommentError):
        comment.plan_comment(
            {"head_sha": FORK_HEAD, "pull_requests": []},
            "same\n",
            list_open=lambda: [_pull(1, FORK_HEAD), _pull(2, FORK_HEAD)],
            get_pull=get_pull,
        )
    with pytest.raises(comment.CommentError):
        comment.plan_comment(
            {
                "head_sha": FORK_HEAD,
                "pull_requests": [_pull(1, FORK_HEAD), _pull(2, FORK_HEAD)],
            },
            "same\n",
            list_open=lambda: [],
            get_pull=get_pull,
        )
    assert calls == []


def test_a_changed_head_posts_nothing():
    def get_pull(number):
        return _pull(number, "e" * 40)

    with pytest.raises(comment.CommentError):
        comment.plan_comment(
            {"head_sha": FORK_HEAD, "pull_requests": [_pull(9, FORK_HEAD)]},
            "same\n",
            list_open=lambda: [],
            get_pull=get_pull,
        )


def test_the_reviewer_poison_fixture_stays_inside_the_fence():
    report = FIXTURE.read_text(encoding="utf-8")
    body = comment.render_body(report, "1" * 40, "2" * 40)
    outside, inside = comment.fenced_parts(body)
    assert "@lode-org/owners" in inside
    assert "https://example.invalid/poison.png" in inside
    assert "@lode-org/owners" not in outside
    assert "https://example.invalid/poison.png" not in outside
    assert "```\n" in inside
    _assert_inert(body, report)


def test_tilde_html_and_nested_fences_stay_inside_the_fence():
    report = (
        "~~~~\n"
        "@team\n"
        "![tilde](https://example.invalid/tilde.png)\n"
        "<img src=\"https://example.invalid/html.png\">\n"
        "<a href=\"https://example.invalid/html\">link</a>\n"
        "`````\n"
        "nested\n"
        "`````\n"
    )
    body = comment.render_body(report, "1" * 40, "2" * 40)
    _assert_inert(body, report)
    outside, inside = comment.fenced_parts(body)
    assert "<img" in inside and "<img" not in outside
    assert "<a href" in inside and "<a href" not in outside
    assert "~~~~" in inside and "~~~~" not in outside
    assert "`````" in inside


def test_a_carriage_return_or_a_huge_report_is_rejected():
    with pytest.raises(comment.CommentError):
        comment.render_body("line\r\n@user\n", "1" * 40, "2" * 40)
    with pytest.raises(comment.CommentError):
        comment.render_body("a" * (comment.MAX_REPORT_BYTES + 1), "1" * 40, "2" * 40)
    body = comment.render_body("a" * comment.MAX_REPORT_BYTES, "1" * 40, "2" * 40)
    assert "a" * 40 in body


def test_publish_does_not_post_when_selection_fails():
    class Fake:
        def __init__(self):
            self.posts = 0

        def list_open(self):
            return []

        def get_pull(self, number):
            raise AssertionError(number)

        def upsert(self, number, body):
            self.posts += 1

    fake = Fake()
    with pytest.raises(comment.CommentError):
        comment.publish(
            {"head_sha": FORK_HEAD, "pull_requests": []},
            "same\n",
            fake,
        )
    assert fake.posts == 0
