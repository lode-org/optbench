"""Sphinx configuration for the OptBench site."""

from pathlib import Path

project = "OptBench"
author = "OptBench"
copyright = "2026, Rohit Goswami"
html_title = "OptBench"
# Sphinx version and release are the project version.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-version
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-release
# Generated from pyproject.toml by cog (https://nedbatchelder.com/code/cog/).
# [[[cog
# from optbench.project_version import cog_out
# cog_out(cog, 'version = "{version}"')
# cog_out(cog, 'release = "{version}"')
# ]]]
version = "0.0.0"
release = "0.0.0"
# [[[end]]]

extensions = []
exclude_patterns = ["_build"]

html_theme = "shibuya"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_css_files = ["optbench.css"]
html_js_files = ["bench.js"]
html_copy_source = False
html_permalinks = False
html_theme_options = {
    "accent_color": "bronze",
    "github_url": "https://github.com/lode-org/optbench",
    "dark_code": True,
    "globaltoc_expand_depth": 0,
    "page_layout": "default",
    # The copy-page menu adds ChatGPT and Claude links, and the theme
    # default mounts a Read the Docs ad slot. Neither is part of the rank.
    "show_ai_links": False,
    "open_in_chatgpt": False,
    "open_in_claude": False,
    "ethical_ads_publisher": "",
    "nav_links": [
        {"title": "Tutorial", "url": "tutorial"},
        {
            "title": "Suites",
            "url": "index",
            "children": [
                {"title": "Minimization", "url": "generated/suites/minimization"},
                {"title": "Saddle search", "url": "generated/suites/saddle-search"},
                {"title": "Double-ended saddle search", "url": "generated/suites/double-ended-saddle-search"},
                {"title": "Global optimization", "url": "generated/suites/globalopt"},
                {"title": "Lowest eigenvector", "url": "generated/suites/lowest-eigenvector"},
                {"title": "Transition state optimization", "url": "generated/suites/tsopt"},
                {"title": "Machine learning optimization", "url": "generated/suites/mlopt"},
                {"title": "Nudged elastic band", "url": "generated/suites/neb"},
                {"title": "Harmonic transition state theory", "url": "generated/suites/hTST"},
                {"title": "Accelerated molecular dynamics", "url": "generated/suites/accelerated-md"},
            ],
        },
        {"title": "Board", "url": "board"},
        {"title": "Methodology", "url": "methodology"},
        {"title": "Codes", "url": "codes"},
    ],
}
html_baseurl = "https://lode-org.github.io/optbench/"


_FLAG_FONT = "@font-face{font-family:Twemoji Country Flags;"
# The theme JS is minified, so each edit matches its exact text in the
# shibuya release that pixi.lock pins.
_JS_EDITS = (
    ("function Tn(t,e){", "function Tn(t,e){return;"),
    ("//cdn.carbonads.com/carbon.js?serve=${t}&placement=${e}", ""),
    ("https://api.iconify.design", ""),
    ("https://api.simplesvg.com", ""),
    ("https://api.unisvg.com", ""),
    ("https://api.github.com/repos/", ""),
    ("https://gitlab.com/api/v4/projects/", ""),
    ('import("https://esm.sh/turndown")', "Promise.reject()"),
)


class ThemeAssetError(RuntimeError):
    """A theme file no longer holds the text that _drop_remote_assets removes."""


def _missing(path, target):
    return ThemeAssetError(
        f"{path.name} has no {target!r}, so the page would load a remote asset again. "
        "The theme changed. Find the new text and update docs/source/conf.py."
    )


def _drop_remote_assets(app, exception):
    """The theme ships a flag font and an ad loader that point at CDNs."""
    if exception is not None or app.builder.format != "html":
        return
    out = Path(app.outdir)
    css = out / "_static" / "shibuya.css"
    text = css.read_text(encoding="utf-8")
    start = text.find(_FLAG_FONT)
    end = text.find("}", start)
    if start == -1 or end == -1 or "jsdelivr" not in text[start:end]:
        raise _missing(css, _FLAG_FONT)
    css.write_text(text[:start] + text[end + 1 :], encoding="utf-8")
    js = out / "_static" / "shibuya.js"
    text = js.read_text(encoding="utf-8")
    for old, new in _JS_EDITS:
        if old not in text:
            raise _missing(js, old)
        text = text.replace(old, new)
    js.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("build-finished", _drop_remote_assets)
