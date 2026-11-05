# asv gaps

optbench is pinned to asv 0.6.6 and asv-runner 0.3.1 until these land. Each gap is a behavior this tree needs, the smallest change in the upstream project, and a check that was run here. The checks live in `docs/asv-gaps/` and print the observation. They exit non-zero if the gap has closed.

The local pieces that stay until then:

- `optbench/asv_uv.py` creates the `uv-pip` environment. It is the discovery bridge, not a scoring fork.
- `optbench/asv_score.py` reads track values with `asv.results` and calls `score_v2`. It does not patch `Results.save` or `compute_stats`.
- Failure classes are four `track_failure_*` floats, because a result file cannot store a per-parameter cause.
- `cap` is a class attribute. The score records the cap it used in `results/validation/statistics.json`.
- `optbench/asv_launch.py` writes the whole machine entry and gives each run its own machine file.
- `optbench/asv_launch.py` also sets the BLAS thread variables before asv starts its server.

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

It does not compute a success-rate interval, a PAR-2 mean with a cap penalty, a bootstrap median across parameters, or a Dolan-Moré profile. asv-perch compares commits with asv-spyglass (Mann-Whitney on timing samples, by that action's own description). It has no input for a censored-cost reducer.

Upstream: `asv-runner` (`compute_stats` or a new reducer entry point that receives the per-parameter track vector, not the timing repeats) and `asv-perch` / asv-spyglass (print that summary). The smallest useful addition is a hook that accepts the vector of track values for one benchmark and returns a summary object the result file can store. A success-rate interval, PAR-2 given a cap, and a bootstrap median belong there. The Dolan-Moré profile needs several benchmarks on the same problems, so it is a cross-benchmark reducer, not a per-benchmark `compute_stats` extension.

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

Upstream: `airspeed-velocity/asv`, `asv/commands/run.py`. Set `skip_save` when `quick` is true, before `load_data` and `save`, so a quick run cannot rewrite a result file. The help text can stay.

On this tree, `uv run asv run --quick` exited 0 after 149 benchmarks and rewrote `results/asv/xeon-8c/d4513f42-uv-pip-py3.12.json`. sha256 before that command was `b94dd44605cdf3cccd91aa87e6de147bcbbae3c15c5d4a10fc8879b59c4a2df6`. sha256 after was `74836b8328cc1fc68b440fbf390c70b3e2186ce05a27005bee744b91d4ac9daa`. The force-call tracks were the same. `started_at` and `duration` were not. The file committed after that is a run without `--quick`.

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

## 9. `asv machine --machine NAME` stores only the name

`asv/commands/machine.py` passes `Machine.load` only the options that differ from `Machine.get_defaults()`.
With any option given, the prompt is skipped and the defaults are never filled in.
`Machine.load` starts from the entry stored under the hostname, or from the only entry in the file, and ignores the entry named by `--machine`.
On a new machine file, `asv machine --yes --machine NAME` therefore stores `{"machine": NAME}`.
`asv run --machine NAME` copies that entry into `results/<machine>/machine.json`, without `arch`, `cpu`, `num_cpu`, `os`, or `ram`.
`asv machine --yes` with no option takes the prompt path and stores every field under the hostname.

Upstream: `airspeed-velocity/asv`, `asv/machine.py`, `Machine.load`.
When `use_defaults` is set, fill each field the entry lacks from `get_defaults()` after the given options are applied.
Look the entry up under the given `machine` option, so an existing entry keeps its stored fields.

Until then, `python -m optbench.asv_launch machine --yes --machine NAME` writes the whole entry.
Each field is the given value, else the stored one, else the detected one.

`python -m optbench.asv_launch run` also fills each field the stored entry lacks from detection.
A run from an entry that `asv machine` saved as just the name then records every field, both in `results/<machine>/machine.json` and in each result's `params`.
The asv workflow writes its entry with the launcher.

```bash
uv run python docs/asv-gaps/machine_fields.py
```

```
stored fields: ['machine']
missing fields: ['arch', 'cpu', 'num_cpu', 'os', 'ram']
```

## 10. Concurrent runs race on `~/.asv-machine.json`

`asv run`, `asv continuous`, and `asv find` call `Machine.load`, and every load ends in `MachineCollection.save`.
`save` reads the whole file, sets one entry, and writes the file back with `util.write_json`.
`write_json` opens the file for writing, so it truncates the file in place.
Another asv process that reads in that window sees an empty or half-written file.
Its `Machine.load` logs the parse error and goes on with an empty entry, or its `save` fails with `UserError`.
`asv run` loads with `interactive=True`, so off a console the empty entry fails with `UserError: Run asv at the console the first time to generate one, or run asv machine --yes.`
On a console it starts the machine prompt instead.
Two saves that overlap keep only the later file, so an entry that one of them added is lost.
Two writes that overlap on the same open file can also leave it unparseable.
Both the read and the write run unlocked.
The machine file is always `~/.asv-machine.json`, and `Run.run` takes `_machine_file` only from Python.

`docs/asv-gaps/machine_race_trials.py` starts eight processes together, each loading an existing entry 200 times, and repeats that in nine trials.
On an eight-CPU machine, the loads alone failed a median of 1181 of 1600 times, with a range of 1162 to 1362.
A ninth process added 50 entries with `asv machine --yes --machine NAME` during the loads.
A median of 1285 loads then failed (986 to 1395), and a median of 1 of the 50 entries was missing at the end (0 to 3).
When that process added the entries with `python -m optbench.asv_launch machine` instead, 1285 loads failed (1118 to 1414).
A median of 48 of the 50 entries were missing (41 to 50), most of them because the merge read a truncated file.
One of those nine trials left the machine file unparseable.
With every load launched, no load or merge failed in any trial, and all 50 entries were kept.

Upstream: `airspeed-velocity/asv`, `asv/util.py`, `write_json`.
Write a temporary file in the same directory and `os.replace` it over the target, so a reader sees the old file or the new one.
`asv/machine.py`: hold a lockfile around the read and the write in `MachineCollection.save`, and skip the save in `Machine.load` when the entry is unchanged.
Add an `ASV_MACHINE_FILE` variable or a `--machine-file` option.
Two runs that share a results directory also write `results/<machine>/machine.json` and `results/benchmarks.json` with `write_json`.
Two runs of the same commit and environment load and save the same result file, and the later save drops the other run's results.

Until then, `optbench/asv_launch.py` reads the shared file under a shared `flock` on `~/.asv-machine.json.lock`.
Each launched `run`, `continuous`, or `find` gets a private file with only the entry it loads, and asv writes that one.
A launched `machine` or `update` runs under the exclusive lock.
A bare `asv` takes no lock.

```bash
uv run python docs/asv-gaps/machine_race.py
```

```
a load writes the file: True
same inode, so truncated in place: True
asv run --machine -m: Use the given name to retrieve machine information.
```

## 11. The benchmark server loads NumPy before the suite

`asv_runner.benchmarks` imports every benchmark type at import.
`asv_runner/benchmarks/mem.py` imports `pympler.asizeof` at module level, and Pympler imports NumPy.
The server imports that package before it imports any benchmark module.
OpenBLAS sizes its thread pool once, when it loads.
A benchmark module that sets `OPENBLAS_NUM_THREADS` at import is in time for SciPy's bundled OpenBLAS and too late for NumPy's.
With the variables unset, NumPy's pool is one thread per core.

The one `asv.conf.json` route to the server's environment is a matrix `env_nobuild` value.
That value becomes part of the environment name, and so of every result file name.
`-E existing` ignores the matrix.

Upstream: `airspeed-velocity/asv_runner`, `asv_runner/benchmarks/mem.py`.
Import `pympler.asizeof` inside `MemBenchmark` when a memory benchmark runs, so discovery leaves NumPy unloaded.
`airspeed-velocity/asv`: an `asv.conf.json` key for variables that reach the server, that is not part of the environment name, and that an `existing` environment honors.

Until then, `optbench/asv_launch.py` sets `optbench.blas_threads.PINNED` in the process that starts asv.
`.github/workflows/asv.yml` sets the same variables on the job.

```bash
uv run python docs/asv-gaps/numpy_preimport.py
```

```
numpy loaded before asv_runner.benchmarks: False
numpy loaded by asv_runner.benchmarks: True
pympler.asizeof loaded: True
environment name with env_nobuild OPENBLAS_NUM_THREADS=1: uv-pip-py3.12-OPENBLAS_NUM_THREADS1
existing environments ignore the matrix: True
```
