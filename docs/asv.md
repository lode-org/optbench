# asv

asv is the measurement store. A result file is keyed to the commit asv checked out, the environment that ran it, and the machine record next to it. The v2 score reads those tracks. It does not keep a second copy of the numbers.

`asv==0.6.6` and `asv-runner==0.3.1` are pinned in `pyproject.toml` and `uv.lock`. The benchmark classes use asv-runner's `track_*` and `time_*` discovery. CI uses `HaoZeke/asv-perch@v1` in `.github/workflows/asv.yml`. Gaps that would otherwise be local forks are `docs/asv-gaps.md`.

## What is committed

| Path | Role |
| --- | --- |
| `asv.conf.json` | `environment_type` `uv-pip`, `results_dir` `results/asv` |
| `uv.lock` | The Python environment asv installs, including the asv pin |
| `environments/eon-3.5.0.yml` | conda-forge pin for the eOn 3.5.0 binary |
| `environments/eon-develop.yml` | Python pin for the eOn develop source build. Not a conda eOn package |
| `environments/eon-svn.yml` | Python pin for the eOn SVN r2549 source build. Not a conda eOn package |
| `environments/vcsqnm.yml` | Python pin for the vc-SQNM v1.3 source build. Not a conda package |
| `benchmarks/` | The suite asv-runner imports |
| `results/asv/<machine>/machine.json` | The machine record asv wrote |
| `results/asv/<machine>/<commit[:8]>-<env>.json` | Track values for that commit |

`html_dir` stays `.asv/html` and is not committed. `asv publish` rebuilds it from `results/asv`.

`optbench.asv_uv.Uv` is the `uv-pip` environment. Upstream `asv.plugins.uv.Uv.matches` does not survive discovery, and that plugin installs with pip. Both are in the gaps file. This module is the bridge until those land. It is not a second statistics implementation. asv resolves `install_command` inside the environment's `bin`, so the environment also installs `uv==0.13.1` and `asv.conf.json` can call `uv pip`.

## Adding a code

1. Pin the code commit in `codes/registry.toml`. A published eOn score keeps commit `4d8b34baf1ba8d2d980df29838bac1f969d1662c`. A conda binary's printed token is not that commit.
2. If the code is a conda package, add `environments/<name>.yml` with the version. Do not invent a build string. A source build that is not a conda package still gets a yml so its pin is not the 3.5.0 file; leave the package out. asv has one `environment_type` for the run, so the yml is the pin and `asv run -E` is how a later unit selects it. There is no per-benchmark environment attribute. See the gaps file. `environments/eon-develop.yml` is that source pin. The develop classes call `EONCLIENT`.
3. Copy `benchmarks/quadratic.py`. `param_names` is `["seed", "start"]`. `measure` returns the `record_start` dict. Set `cap` on the class. The result row does not store the cap.
4. Inherit `optbench.asv_tracks.TrackedSearch`, or copy its methods. `track_*` returns a float. Set `.unit` on the function (`force calls`, `fraction`, `count`). A dict with `samples` and `number` is a timing result, and asv replaces the value. `time_search` is the wall clock.
5. Set `repeat = 1`, `rounds = 1`, `number = 1`, `sample_time = 0`, `warmup_time = 0`. `repeat == 0` means up to ten repeats.
6. Commit the benchmark. `asv run` installs that commit. Uncommitted edits are not the run.
7. `uv run asv machine --yes` once on the machine. That writes `~/.asv-machine.json`. The copy that is committed is `results/asv/<machine>/machine.json`.
8. `uv run asv run <commit>^! -b YourClass`. Do not pass `--quick`. On asv 0.6.6 a quick run still writes `results/asv`. The help text says it does not. That is `docs/asv-gaps.md`. The committed file is the run without `--quick`.
9. Commit the new files under `results/asv`.
10. Score with `asv.results.iter_results` and `Results.get_result_value`, then `score_v2`. `optbench.asv_score` does that for the quadratic sample. Parameter strings in the result JSON are wrapped in `repr` quotes; the reader unwraps them with `ast.literal_eval`. `uv run optbench statistics` refuses to write a document when those tracks are absent.

`benchmarks/pending.py` has one skipped class per registry code, named `Pending` and the code.
Its problems are the ones that no class measures yet.
Each pending class has `track_force_calls` and `time_search` only.
`setup` raises `NotImplementedError`, which asv-runner stores as NaN.
A pair leaves its pending class when a real class measures it, and `pending.covered` lists those pairs.
The Lennard-Jones classes in `benchmarks/bench_lj38.py` are separate: they call eonclient when `OPTBENCH_ASV_EON=1`.
The default parameter list is one seed and the first geometry.
`OPTBENCH_ASV_FULL=1` selects all 1000 starts.
It is read at import, so it has to be set in the process that launches asv.

`benchmarks/bench_eon_develop.py` is the eOn develop entry.
Counting is `HarnessCounter` through the `ext_pot` bridge, not the client's own tally.
The default parameter list is one seed and the first start.
`OPTBENCH_ASV_FULL=1` selects the campaign grid at import, for every template.
The Pt-island AKMC classes default to one pass.
With `OPTBENCH_ASV_FULL=1` they run every pass up to the template's `pass_limit`.
A one-start file is not a published mean.

`optbench.asv_engines` builds the classes for the campaign engines in `optbench/engines`.
Each engine has one `benchmarks/bench_<engine>.py`.
A class is one template on one problem.
Its parameters are the seed and the campaign job name, which is the `start` field of the committed `.starts.jsonl`.
The class runs the job that the engine's `plan` gives `optbench run`, alone in a spawned process.
The default parameter list is the first start, and `OPTBENCH_ASV_FULL=1` selects every start.
Each track runs in a fresh process.
It reads the record that `time_search` stored under `OPTBENCH_ASV_CACHE`.
When that record is missing, the track runs the search.
The force-call cap ends every search, so the class timeout is unbounded.
Run these classes with `--launch-method forkserver`, because the spawn launcher cannot wait without a bound.

The quadratic sample is the run that `results/validation/statistics.json` scores. A subset of Lennard-Jones 38 is `results/validation/lj38-counter.dat` and it is not a published mean.

## Commands

```bash
uv run asv check
uv run asv run --quick
uv run asv machine --yes
uv run asv run COMMIT^! -b quadratic
uv run optbench statistics --output results/validation/statistics.json
uv run asv publish
uv run asv compare COMMIT COMMIT
```

`asv check` imports the suite. `asv run --quick` is the gate before the saved run. Run it before the saved `asv run`, because asv 0.6.6 rewrites the result file. `asv compare` on one commit compares that commit with itself.

## CI

`.github/workflows/asv.yml` fetches the full history, installs Python 3.12 and syncs the lockfile.
It then runs `HaoZeke/asv-perch@v1`.
The benchmark command is `uv run asv run --skip-existing --bench quadratic {sha}^!`.
That is the in-process sample, not the 1000-start eonclient grid.
`post-comment` is true when the head branch of a pull request lives in this repository.
It stays false on a push, on a manual run and on a pull request from a fork.
A fork's run has a read-only token, so GitHub refuses its comment request.
Those runs still write the table to the step summary.
A failure in the benchmarks or the comparison ends the step before the summary, so it still fails the job.
A pull request whose base commit lacks `benchmarks/quadratic.py` fails this job, because asv runs that commit.
`check.yml` is still the oracle and pytest job.

asv-spyglass `compare` exits 1 when any benchmark is worse by `--factor`.
The default factor is 1.1.
`HaoZeke/asv-perch@v1` does not ignore that status, so the step stops before `regression-threshold` is applied.
A pull request compares two commits.
A push compares a commit with itself, which is why those runs stay green.
Quadratic timings move by about ten percent between two runs, and that is enough to exit 1.
The workflow sets `regression-threshold` to 10 and passes `--factor 10`.
A slowdown past that factor still fails the job.
