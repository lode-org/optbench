# Contributing to OptBench

Questions, bug reports, and requests for a new code or problem go to the GitHub issue tracker at <https://github.com/lode-org/optbench/issues>.
Open an issue before a large change.
Name the problem id and the template id in that issue first.

## Pull requests

Open a pull request against `main` for each change.
Keep one pull request to one purpose.
The repository carries the Massachusetts Institute of Technology (MIT) License in [LICENSE](LICENSE).
Offer a pull request under that same license.

## Checks

The check workflow in `.github/workflows/check.yml` runs on every pull request.
Run the same commands from a clone before you push.

```bash
uv run pytest
uv run optbench check
uv run cog --check --diff docs/source/conf.py optbench/__init__.py pixi.toml
```

`optbench check` scans the docs, the site text, and the board files.
The scan rejects words that the site leaves unused.
`pyproject.toml` records the version.
Refresh the generated copies after a version change.

```bash
uv run cog -r docs/source/conf.py optbench/__init__.py pixi.toml
```

The same workflow builds the site and measures page weight.

```bash
pixi run -e docs html
pixi run -e docs sitesize
```

## Docs

Site prose lives in Org under `docs/orgmode`.
`pixi run -e docs rst` exports that Org to reStructuredText under `docs/source`.
`pixi run -e docs html` runs the export before Sphinx.
`uv run python -m optbench.site` writes the rank tables.
Those tables go into `docs/include/` and into `docs/orgmode/generated/`.
The same command also writes `docs/orgmode/codes.org`.
Edit `optbench/site.py` when a table should change.
Leave the generated files untouched.

## Newsfragments

Each pull request adds one short Markdown file under `newsfragments/`.
Name the file by date and topic, for example `newsfragments/20261105-citation.md`.
Say what changed, in a few sentences, for a reader of the site or a user of the `optbench` command.

## Scores and codes

A new score comes from one `benchmark.dat`.
Produce that file with `uv run optbench run`, the problem id, and the template id, on every start in the problem.
The problem file owns the stop and the geometries.
The template owns only the client.
Read `docs/asv.md` for the steps another code uses to add an asv track.
