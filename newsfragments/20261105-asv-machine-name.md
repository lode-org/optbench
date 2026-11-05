The asv commands in `docs/asv.md`, `docs/v2.md` and the README pass the committed machine name to `asv run`.
They run `asv machine --yes` once without `--machine` and once with it, so the record keeps the host's OS, CPU and memory under that name.
asv stores only the name when it gets `--machine` alone and has no record for the host.
The next run then writes that bare record over the committed `machine.json`.
A run that follows them writes into the committed folder under `results/asv`, not into a new folder named after the host.
CI still names its machine after the runner, and it commits no result files.
