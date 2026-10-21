# asv gaps

optbench is pinned to asv 0.6.6 and asv-runner 0.3.1 until these land. Each gap is a behavior this tree needs, the smallest change in the upstream project, and a reproducer that was run here. The reproducers live in `docs/asv-gaps/` and print the observation. They exit non-zero if the gap has closed.

The local pieces that stay until then:

- `optbench/asv_uv.py` creates the `uv-pip` environment. It is the discovery bridge, not a scoring fork.
- `optbench/asv_score.py` reads track values with `asv.results` and calls `score_v2`. It does not patch `Results.save` or `compute_stats`.
- Failure classes are four `track_failure_*` floats, because a result file cannot store a per-parameter cause.
- `cap` is a class attribute. The score records the cap it used in `results/validation/statistics.json`.

## 1. Track units other than seconds and bytes

`asv.util.human_value` understands `seconds` and `bytes`. Any other unit is omitted, so a force-call track renders as a bare number. The value in the result file is unchanged.

Upstream: `airspeed-velocity/asv`, `asv/util.py`, `human_value`. When `unit` is a non-empty string and it is not `seconds` or `bytes`, append it to the display. `asv compare` and an asv-perch comment both go through this display. asv-perch has no separate unit input. Fixing the display fixes the comment text those tools already print.

```bash
uv run python docs/asv-gaps/track_unit.py
```

```
force calls: '139'
seconds: '1.50s'
```

`asv compare d4513f42 d4513f42` prints the same omission. `quadratic.QuadraticDisplacement.track_force_calls('20261016', '0')` is `139.0` on both sides. `quadratic.QuadraticKept.time_search('0', '0')` is `322±0μs`.

## 2. Per-parameter failure cause is not in the result file

`Results.add_result` keeps `errcode` and `stderr` in memory. `Results.save` does not write them. The comment in `asv/results.py` says so. A failed parameter becomes `None`. A `NotImplementedError` in `setup` becomes NaN. Neither carries the exception class.

`asv/runner.py` already has `res.errcode` and `res.stderr` from each parameter process (`_run_benchmark_single_param`). The combine loop collapses those to one int and one string, and then save drops both.

Upstream: `airspeed-velocity/asv`. Keep a list of exit codes and a list of stderr strings, one entry per parameter, on `BenchmarkResult`, and write those lists from `Results.save`. Restore them in `Results.load`. asv-runner does not need a new process protocol for the exit code it already returns. A benchmark-supplied failure class, if wanted as a column, is a further field beside `result`. optbench does not add that field locally. It stores the class as `track_failure_cap`, `track_failure_wrong_stationary`, `track_failure_divergence`, and `track_failure_crash`, which asv can already save as floats.

```bash
uv run python docs/asv-gaps/errcode.py
```

```
saved errcode: {'bench.track_force_calls': 3}
loaded errcode: {}
loaded stderr: {}
file mentions errcode: False
file mentions boom: False
```

## 3. The upstream uv environment does not match, and it installs with pip

`asv.plugins.uv.Uv.matches` is a classmethod and reads `self._uv_path`. `__init__` sets `_uv_path` on the instance. Discovery calls `matches` on the class, so `environment_type` `uv` is never selected.

`Uv._setup` creates the environment with `uv venv` and then installs with `pip`.

Upstream: `airspeed-velocity/asv`, `asv/plugins/uv.py`. `matches` should resolve the `uv` executable itself (`util.which("uv")`), and requirement installs should use `uv pip install`. `optbench.asv_uv` stays loaded from `asv.conf.json` (`plugins` and `environment_type` `uv-pip`) until both of those are true.

```bash
uv run python docs/asv-gaps/uv_matches.py
```

```
matches is classmethod: True
setup doc:
Setup the environment on disk using 'uv venv'.
Then, all of the requirements are installed into
it using `pip install`.
run_pip:
def _run_pip(self, args, **kwargs):
        # Run pip via python -m pip, so that it works on Windows when
        # upgrading pip itself, and avoids shebang length limit on Linux
        return self.run_executable('python', ['-m', 'pip'] + list(args), **kwargs)
AttributeError: type object 'Uv' has no attribute '_uv_path'
```

The `AttributeError` is the call `Uv.matches("3.12")`. `_run_pip` is the installer that call uses.

## 4. Timing statistics are not the v2 score

`asv_runner.statistics.compute_stats` summarizes repeated timings of one parameter. The keys are a 99% interval around the median time, the quartiles, the repeat count, and `number`. asv calls it when a benchmark returns `samples` and `number`. Track benchmarks that return a float skip it, which is why force calls survive.

It does not compute Wilson bounds, a PAR-2 mean with a cap penalty, a bootstrap median across parameters, or a Dolan-Moré profile. asv-perch compares commits with asv-spyglass (Mann-Whitney on timing samples, by that action's own description). It has no input for a censored-cost reducer.

Upstream: `asv-runner` (`compute_stats` or a new reducer entry point that receives the per-parameter track vector, not the timing repeats) and `asv-perch` / asv-spyglass (print that summary). The smallest useful addition is a hook that accepts the vector of track values for one benchmark and returns a summary object the result file can store. Wilson, PAR-2 given a cap, and a bootstrap median belong there. The Dolan-Moré profile needs several benchmarks on the same problems, so it is a cross-benchmark reducer, not a per-benchmark `compute_stats` extension.

optbench calls `score_v2` and `performance_profile` on the saved floats. That is the product score, not a copy of `compute_stats`.

```bash
uv run python docs/asv-gaps/compute_stats.py
```

```
value: 25.0
keys: ci_99_a ci_99_b number q_25 q_75 repeat
score keys present: none
```

## 5. One environment type, and no per-benchmark environment

`asv.conf.json` has a single `environment_type` string. `asv_runner.benchmarks._base.Benchmark` has no environment attribute. `asv run -E` can select `existing` or `conda` for the whole invocation. It cannot bind one class to the eOn conda environment and another to `uv-pip`.

Upstream: `airspeed-velocity/asv`. An optional `environment` string on the benchmark class, honored when `asv run` builds the benchmark set for an environment. Benchmarks without it keep today's behavior.

Until that exists, the default run is `uv-pip` from the lockfile. `environments/eon-3.5.0.yml` is the conda-forge pin (`python=3.12`, `eon=3.5.0`, no build string). The Lennard-Jones classes skip unless `OPTBENCH_ASV_EON=1`. They are not given a fake per-class environment.

```bash
uv run python docs/asv-gaps/one_environment.py
```

```
environment_type: 'uv-pip'
type of environment_type: str
Benchmark.__init__ mentions environment: False
Benchmark attributes containing env: none
```

## 6. The result row does not repeat benchmark attributes

Discovery writes public attributes such as `unit` and `cap` into `benchmarks.json`. The result JSON stores the values and the parameter lists, not those attributes. Rescoring an old file with a later class can apply a different cap. `Results.save` would have to copy the attribute dict it already gives discovery into the result entry.

Upstream: `airspeed-velocity/asv`, `Results.save`. One metadata object per benchmark name, the same public attributes `list_benchmarks` already emits.

optbench reads `cap` from the class when it writes `results/validation/statistics.json`, and stores that cap in the document next to the commit and the benchmark version hash. It does not write the attribute back into the asv file.

## 7. install_command is resolved inside the environment

`Environment.find_executable` searches the environment `bin` and then `util.which`. It does not search `PATH`. `asv.conf.json` runs `uv pip install` for the project. A fresh uv virtualenv does not contain `uv`, so project install fails before any benchmark runs.

Upstream: `airspeed-velocity/asv`, `Environment.find_executable` (or the install-command runner). Search `PATH` after the environment `bin`, or document that every executable named in `install_command` must already be inside the environment. optbench installs `uv==0.13.1` into the environment during `_setup` so the existing install command works. That pin is the installer, not a change to asv's lookup.

```bash
uv run python docs/asv-gaps/install_executable.py
```

This run of the script printed:

```
Could not find 'uv' in /tmp/tmpa0820ny2/bin
```

The same lookup, from `asv run a20986fa^! -b quadratic` before `uv` was installed into the environment:

```
OSError: Could not find 'uv' in /workspace/env/88db567784e3bd521507a26797493af6/bin
```

## 8. `--quick` saves, and the help says it does not

`asv run --help` says a quick run's results "are not saved." `Run.run` still calls `result.save` unless the run is a dry run or an `ExistingEnvironment` without `--set-commit-hash`. `quick` is not part of that condition. `--quick` only forces `number`, `repeat`, `warmup_time`, and `rounds` to one via `extra_params` in `asv/runner.py`.

Upstream: `airspeed-velocity/asv`, `asv/commands/run.py`. Set `skip_save` when `quick` is true, before `load_data` and `save`, so a smoke run cannot rewrite a result file. The help text can stay.

On this tree, `uv run asv run --quick` exited 0 after 149 benchmarks and rewrote `results/asv/cursor/d4513f42-uv-pip-py3.12.json`. sha256 before that command was `b94dd44605cdf3cccd91aa87e6de147bcbbae3c15c5d4a10fc8879b59c4a2df6`. sha256 after was `74836b8328cc1fc68b440fbf390c70b3e2186ce05a27005bee744b91d4ac9daa`. The force-call tracks were the same. `started_at` and `duration` were not. The file committed after that is a run without `--quick`.

```bash
uv run python docs/asv-gaps/quick_saves.py
```

```
help:
Do a "quick" run, where each benchmark function is
            run only once.  This is useful to find basic errors in the
            benchmark functions faster.  The results are unlikely to
            be useful, and thus are not saved.
skip_save:
skip_save = dry_run or (
                            isinstance(env, environment.ExistingEnvironment)
                            and set_commit_hash is None
                        )
```
