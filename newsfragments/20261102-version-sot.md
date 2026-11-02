The OptBench version lives only in `pyproject.toml` under
`[project].version`.
Cog writes `__version__`, the pixi workspace version and the Sphinx
`version` and `release` from it.
The check workflow fails when a copy drifts.
The quadratic sample and the statistics provenance read the installed
package version.
