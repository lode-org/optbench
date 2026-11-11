New results record the machine from `BENCH_HOST_LABEL`, or `unspecified` when it is unset, instead of the host name.
Campaign runs set `BENCH_HOST_LABEL=workstation-a`.
A label is letters, digits, `.`, `_`, and `-`, so a value with a newline, a space, or a slash inside it stops the command instead of adding a `benchmark.dat` field or a path.
`uv run optbench asv-machine` stores the label for asv with the other machine fields `unspecified`, and the documented and CI `asv run` commands pass `--machine`, so new asv results go under `results/asv/<label>/`.
`optbench statistics --machine` picks one machine directory, and provenance takes the label instead of the asv machine name.
Archived results keep the machine name they were written with, and `results/README.md` says so.
The site render deletes a retired include through any `*-pass.org` name.
