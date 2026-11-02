"""Sphinx configuration for the OptBench site."""

project = "OptBench"
author = "OptBench"
copyright = "2014--present, OptBench contributors"
html_title = "OptBench"

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
    "accent_color": "teal",
    "github_url": "https://github.com/lode-org/optbench",
    "dark_code": True,
    "globaltoc_expand_depth": 0,
    # The copy-page menu adds ChatGPT and Claude links, and the theme
    # default mounts a Read the Docs ad slot. Neither is part of the rank.
    "show_ai_links": False,
    "open_in_chatgpt": False,
    "open_in_claude": False,
    "ethical_ads_publisher": "",
}
html_baseurl = "https://lode-org.github.io/optbench/"
