# Harness

The harness is this repository. `uv run pytest` and `uv run optbench check` are the hosted checks. A published Lennard-Jones or Pt-island campaign is run on rg.terra, not in CI.

## Where the files are

Problem statements are TOML under `problems/`. JSON schemas are under `contract/`. `contract/README.md` is the list. `optbench.result/v1` is the 2014 board. `optbench.result/v2` is the comparable score.

## Force calls

`CountedPotential` wraps `evaluate(coords) -> (energy, gradient)`. One entry is one force call. The same coordinates a second time return the cached energy and forces and do not add a call. `attach_ase_counter` counts ASE `calculate` entries, which is the same rule, because ASE does not enter `calculate` again when the geometry is unchanged. `PeleCounter` counts `getEnergyGradient`, and `getEnergy` goes through that method so it is one call.

Verification of the Hessian uses `CountedPotential.verification`. Those calls increment `verify_calls` and are not the search cost.

`optbench/runners/ase_lj38.py`, `optbench/pele_lj38.py`, and `optbench/sella_lj38.py` take the number from that wrapper. The pele files already in `results/campaign/` that predate the harness runner were charged with `result.nfev`. `optbench/runners/pele_min.py` charges `getEnergyGradient` on the problem potential for LBFGS, MYLBFGS, FIRE, and steepest descent. Those stored nfev files are not rewritten.

`uv run optbench run` still shells out to `eonclient` and leaves the client's potential in the binary. That command writes `benchmark.dat` unchanged and a sibling `benchmark.starts.jsonl`. Each line has `counter` `client` and `comparable` false. It is not a v2 score.

eonclient can be scored with the harness counter without rebuilding it. `optbench.potserver` serves one `HarnessCounter`. `python -m optbench.extpot` is the `ext_pot` program eOn already runs: one force evaluation is one socket `calculate`. `RgpotCounter.calculate` is the same counter for an rgpot client. The wire protocol, the seed sample, the failure classes, and the noise rule are `docs/v2.md`.

```bash
uv run optbench validate-counter --limit 3 --output results/validation/lj38-counter.dat
uv run optbench statistics --output results/validation/statistics.json
```

`optbench statistics` reads asv tracks in `results/asv`. It does not rerun the sample. `docs/asv.md` is the run that produces those files.

`validate-counter` needs `eonclient` on `PATH`. The output is a subset. A published Lennard-Jones 38 mean still uses all 1000 starts.

## One start

`record_start` runs the optimizer on the wrapper, then a central-difference Hessian in verification mode. The start is a success only when all of these hold:

- harness calls are strictly below the cap
- the force norm is at or below the stop
- a minimum has Hessian index 0, a saddle has index 1, with eigenvalues below `-1e-6` counted as negative
- the energy is within `energy_tol` of the reference
- the Kabsch RMSD is within `rmsd_tol`

One point uses the Cartesian distance, because centering a single point would report zero for every placement. Two or more points use a proper rotation. A cheap stop on the wrong stationary point is a failure.

## The score

`score_v2` reduces the starts.

- Successes `k` out of `n` carry Wilson bounds with `z = 1.959963984540054`.
- A failure costs `2 * cap`. The ranking number is the mean of the penalised costs.
- The median and the interquartile range are taken on those same penalised costs. A percentile bootstrap of 2000 resamples, seed `20261015`, gives the 95% interval.

If the problem states no cap and every record is a success, `par_applied` is false and the mean is the mean of the recorded costs. That is how the ten Pt-island repetitions are summarised: each cost is `force_calls / total_searches` from the `# rep` line. The `benchmark.dat` `force_calls` value remains the mean of the campaign totals.

`tuning` is `default` or `tuned`. Provenance is the code, version, commit, parameters, and machine. `collect_provenance` fills the machine from the process when the caller does not pass one.

## Smoke

```bash
uv run optbench smoke --output results/smoke/quadratic.json
```

The bowl is `(x - 1)^2 + 10 (y + 2)^2`. Gradient descent with step `0.05` from `(0, 0)` and from `(2, -1)` takes 139 harness calls and matches the minimum. Step `0.01` takes 720 calls. Step `0.001` with cap 200 stops at 200 calls and the PAR-2 cost is 400. A Newton step on `-x^2 + y^2` takes 2 calls and is index 1. The bowl minimum reported as a saddle takes 139 search calls and is charged 10000, because the cap is 5000 and the index is 0.

`uv run optbench score STARTS.jsonl --cap 5000 --output score.json` reduces a per-start file the same way.

## What is not rerun here

The 1000 Lennard-Jones starts, the Pt-island repetitions, and the NEB bands stay as recorded. The site keeps those rows on the 2014 archive table and adds the v2 tables from counts that are already in the files or from the quadratic smoke.
