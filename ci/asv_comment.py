"""Post one comparison comment from the trusted workflow.

The pull request comes from the finished run when that run names one.
Otherwise exactly one open pull request must match the measured head.
The chosen pull request is fetched again, and its head is checked,
before a comment is posted. The report is quoted inside a fence.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MAX_REPORT_BYTES = 40000
MARKER = "<!-- optbench-asv-compare -->"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0d-\x1f\x7f\x80-\x9f]")
_FENCE_RUN = re.compile(r"`+")


class CommentError(Exception):
    """A visible failure that posts nothing."""


def validate_report(report: str) -> None:
    if len(report.encode("utf-8")) > MAX_REPORT_BYTES:
        raise CommentError("the comparison report is too large")
    if _CONTROL.search(report):
        raise CommentError("the comparison report contains a control character")


def fence_for(report: str) -> str:
    runs = [len(item) for item in _FENCE_RUN.findall(report)]
    return "`" * (max([3, *runs]) + 1)


def render_body(report: str, base_sha: str, head_sha: str) -> str:
    validate_report(report)
    fence = fence_for(report)
    quoted = report if report.endswith("\n") else report + "\n"
    return (
        f"{MARKER}\n"
        "## Comparison\n"
        "\n"
        f"Base `{base_sha}`.\n"
        f"Head `{head_sha}`.\n"
        "\n"
        f"{fence}text\n"
        f"{quoted}"
        f"{fence}\n"
    )


def fenced_parts(body: str) -> tuple[str, str]:
    """Return the text outside the report fence and the text inside it."""
    lines = body.split("\n")
    open_at = None
    fence = ""
    for index, line in enumerate(lines):
        if line.endswith("text") and line[: -len("text")] and set(line[: -len("text")]) == {"`"}:
            open_at = index
            fence = line[: -len("text")]
            break
    if open_at is None:
        raise CommentError("the comment has no report fence")
    close_at = None
    for index in range(open_at + 1, len(lines)):
        if lines[index] == fence:
            close_at = index
            break
    if close_at is None:
        raise CommentError("the report fence is not closed")
    outside = "\n".join(lines[:open_at] + lines[close_at + 1 :])
    inside = "\n".join(lines[open_at + 1 : close_at])
    return outside, inside


def _numbers(pulls: list[dict]) -> list[int]:
    found: list[int] = []
    for pull in pulls:
        number = int(pull["number"])
        if number not in found:
            found.append(number)
    return found


def choose_number(workflow_run: dict, open_pulls: list[dict]) -> int:
    """Pick one pull request number, or fail without posting."""
    named = workflow_run.get("pull_requests") or []
    head = workflow_run["head_sha"]
    if named:
        numbers = _numbers(named)
    else:
        numbers = _numbers(
            [pull for pull in open_pulls if pull.get("head", {}).get("sha") == head]
        )
    if len(numbers) != 1:
        raise CommentError(f"expected one pull request for {head}, found {len(numbers)}")
    return numbers[0]


def plan_comment(
    workflow_run: dict,
    report: str,
    *,
    list_open,
    get_pull,
) -> tuple[int, str]:
    """Resolve the pull request and render the body. Do not post."""
    named = workflow_run.get("pull_requests") or []
    open_pulls = [] if named else list_open()
    number = choose_number(workflow_run, open_pulls)
    pull = get_pull(number)
    head = workflow_run["head_sha"]
    if pull.get("head", {}).get("sha") != head:
        raise CommentError("the pull request head no longer matches the measured commit")
    body = render_body(report, pull["base"]["sha"], pull["head"]["sha"])
    return number, body


def _next_url(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section, _, rel = part.partition(";")
        if 'rel="next"' not in rel:
            continue
        url = section.strip()
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        if url.startswith("https://api.github.com/"):
            return url
    return None


class GitHub:
    def __init__(self, token: str, repository: str) -> None:
        owner, repo = repository.split("/", 1)
        self._token = token
        self._owner = owner
        self._repo = repo

    def _request(self, method: str, url: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "optbench-asv-comment",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                raw = response.read()
                link = response.headers.get("Link")
                body = json.loads(raw.decode("utf-8")) if raw else None
                return body, link
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CommentError(f"GitHub API {exc.code} for {method} {url}: {detail}") from exc

    def _get(self, url: str):
        body, link = self._request("GET", url)
        return body, link

    def list_open(self) -> list[dict]:
        url = (
            "https://api.github.com/repos/"
            f"{self._owner}/{self._repo}/pulls?state=open&per_page=100"
        )
        found: list[dict] = []
        while url:
            page, link = self._get(url)
            found.extend(page or [])
            url = _next_url(link)
        return found

    def get_pull(self, number: int) -> dict:
        body, _link = self._get(
            f"https://api.github.com/repos/{self._owner}/{self._repo}/pulls/{number}"
        )
        return body

    def list_comments(self, number: int) -> list[dict]:
        url = (
            "https://api.github.com/repos/"
            f"{self._owner}/{self._repo}/issues/{number}/comments?per_page=100"
        )
        found: list[dict] = []
        while url:
            page, link = self._get(url)
            found.extend(page or [])
            url = _next_url(link)
        return found

    def create_comment(self, number: int, body: str) -> None:
        self._request(
            "POST",
            f"https://api.github.com/repos/{self._owner}/{self._repo}/issues/{number}/comments",
            {"body": body},
        )

    def update_comment(self, comment_id: int, body: str) -> None:
        self._request(
            "PATCH",
            f"https://api.github.com/repos/{self._owner}/{self._repo}/issues/comments/{comment_id}",
            {"body": body},
        )

    def upsert(self, number: int, body: str) -> None:
        for comment in self.list_comments(number):
            user = comment.get("user") or {}
            if user.get("login") == "github-actions[bot]" and MARKER in (comment.get("body") or ""):
                self.update_comment(int(comment["id"]), body)
                return
        self.create_comment(number, body)


def publish(workflow_run: dict, report: str, github: GitHub) -> tuple[int, str]:
    number, body = plan_comment(
        workflow_run,
        report,
        list_open=github.list_open,
        get_pull=github.get_pull,
    )
    github.upsert(number, body)
    return number, body


def main() -> int:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    workflow_run = event["workflow_run"]
    if workflow_run.get("event") != "pull_request":
        raise CommentError("the finished run is not a pull request")
    report_path = Path(os.environ.get("OPTBENCH_ASV_REPORT", "report/compare.txt"))
    report = report_path.read_text(encoding="utf-8")
    github = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    number, _body = publish(workflow_run, report, github)
    print(f"commented on pull request {number}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CommentError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
