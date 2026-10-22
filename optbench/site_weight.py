"""Measure built HTML pages and fail when one is too heavy to render.

Budgets, per page:

* HTML file under 300 KiB
* eager images under 300 KiB
* DOM under 6000 elements

An image is eager when the page fetches it during load. ``loading="lazy"``
thumbnails are not eager. Inside a ``<picture>``, a WebP ``<source>`` is
the file a current browser fetches.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

HTML_BUDGET = 300 * 1024
EAGER_IMAGE_BUDGET = 300 * 1024
DOM_BUDGET = 6000

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif", ".ico"}
_URL = re.compile(r"""url\(\s*(['"]?)([^)'"]+)\1\s*\)""", re.IGNORECASE)


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements = 0
        self.eager: list[str] = []
        self.stylesheets: list[str] = []
        self._pictures: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements += 1
        attr = {key.lower(): value or "" for key, value in attrs}
        if tag == "picture":
            self._pictures.append({"sources": []})
        elif tag == "source" and self._pictures:
            self._pictures[-1]["sources"].append(attr)
        elif tag == "img":
            self._image(attr)
        elif tag == "link":
            self._link(attr)

    def handle_endtag(self, tag: str) -> None:
        if tag == "picture" and self._pictures:
            self._pictures.pop()

    def _image(self, attr: dict[str, str]) -> None:
        if attr.get("loading", "").strip().lower() == "lazy":
            return
        chosen = attr.get("src", "")
        if self._pictures:
            for source in self._pictures[-1]["sources"]:
                kind = source.get("type", "").split(";", 1)[0].strip().lower()
                srcset = source.get("srcset", "").strip()
                if kind == "image/webp" and srcset:
                    chosen = srcset.split(",")[0].strip().split()[0]
                    break
        if chosen:
            self.eager.append(chosen)

    def _link(self, attr: dict[str, str]) -> None:
        rel = attr.get("rel", "").lower().split()
        href = attr.get("href", "")
        if not href:
            return
        if "stylesheet" in rel:
            self.stylesheets.append(href)
        if "icon" in rel or "apple-touch-icon" in rel:
            self.eager.append(href)
        if "preload" in rel and attr.get("as", "").lower() == "image":
            self.eager.append(href)


def _pages(build: Path) -> list[Path]:
    found = []
    for path in sorted(build.rglob("*.html")):
        if "_static" in path.parts or "_sources" in path.parts:
            continue
        found.append(path)
    return found


def _local_file(base: Path, ref: str) -> Path | None:
    text = ref.strip()
    if not text or text.startswith(("data:", "http://", "https://", "//", "mailto:")):
        return None
    text = unquote(text.split("#", 1)[0].split("?", 1)[0])
    if not text:
        return None
    return (base.parent / text).resolve()


def _stylesheet_images(css_path: Path) -> list[Path]:
    if not css_path.is_file():
        return []
    found: list[Path] = []
    text = css_path.read_text(encoding="utf-8", errors="replace")
    for match in _URL.finditer(text):
        ref = match.group(2).strip()
        path = _local_file(css_path, ref)
        if path is None or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        found.append(path)
    return found


def measure_pages(build: Path) -> list[dict]:
    """One record per HTML page under ``build``."""
    rows = []
    for page in _pages(build):
        parser = _Page()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        seen: set[str] = set()
        eager = 0
        missing: list[str] = []
        refs = list(parser.eager)
        for href in parser.stylesheets:
            css_path = _local_file(page, href)
            if css_path is None:
                continue
            if not css_path.is_file():
                missing.append(href)
                continue
            refs.extend(str(path) for path in _stylesheet_images(css_path))
        for ref in refs:
            path = Path(ref) if ref.startswith("/") else _local_file(page, ref)
            if path is None:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_file():
                eager += path.stat().st_size
            else:
                missing.append(ref)
        rows.append(
            {
                "page": page.relative_to(build).as_posix(),
                "html": page.stat().st_size,
                "dom": parser.elements,
                "eager": eager,
                "missing": missing,
            }
        )
    return rows


def _failures(row: dict) -> list[str]:
    reasons = []
    if row["html"] >= HTML_BUDGET:
        reasons.append(f"HTML {row['html']} bytes >= {HTML_BUDGET}")
    if row["eager"] >= EAGER_IMAGE_BUDGET:
        reasons.append(f"eager images {row['eager']} bytes >= {EAGER_IMAGE_BUDGET}")
    if row["dom"] >= DOM_BUDGET:
        reasons.append(f"DOM {row['dom']} elements >= {DOM_BUDGET}")
    for ref in row["missing"]:
        reasons.append(f"missing eager image {ref}")
    return reasons


def check_site_weight(build: Path) -> int:
    """Print each page and return 0 when every page is inside the budget."""
    build = Path(build)
    if not build.is_dir():
        print(f"site build is missing: {build}")
        return 2
    rows = measure_pages(build)
    if not rows:
        print(f"no html pages in {build}")
        return 2
    print(f"{'page':<56} {'html':>10} {'dom':>8} {'eager':>10}")
    failed = False
    for row in rows:
        reasons = _failures(row)
        mark = " FAIL" if reasons else ""
        print(f"{row['page']:<56} {row['html']:10d} {row['dom']:8d} {row['eager']:10d}{mark}")
        for reason in reasons:
            print(f"  {row['page']}: {reason}")
            failed = True
    limit = (
        f"html < {HTML_BUDGET} bytes, eager images < {EAGER_IMAGE_BUDGET} bytes, "
        f"dom < {DOM_BUDGET} elements"
    )
    if failed:
        print(f"site weight exceeds the budget ({limit})")
        return 1
    print(f"site weight within budget ({limit})")
    return 0
