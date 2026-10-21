# Alternative A. PAR-2 penalised cost

Status: selected. The reasons are in `decision.md`.

## What is wrong with the half-even mean

`summarize_capped` averages force calls that sit strictly below the cap and rounds with half-even. A start at the cap is counted in `nfailed` and then left out of the mean. On the published Lennard-Jones 38 board that changes the number. The rounded 2014 means below are taken from `results/sota-lj38.json` (1000 starts, cap 10000).

| Entry | 2014 mean | Failed | Illustration PAR-2 mean |
| --- | ---: | ---: | ---: |
| eon-3.5.0-lbfgs-m100 | 181 | 0 | 181 |
| rgmin-quickmin-0.3.0 | 1849 | 1 | 1867.151 |
| eon-3.5.0-qm | 3514 | 25 | 3926.15 |
| eon-3.5.0-sd | 4901 | 96 | 6350.504 |

The illustration rebuilds a PAR-2 mean as `(successes * 2014 mean + failures * 2 * 10000) / 1000`. It uses the rounded 2014 mean as if every success had that cost. It is not a v2 score. It only shows that dropping failures moves the published number.

## Score

A start is a success only when all of the following hold:

- the harness force-call count is strictly below the cap
- the force norm is at or below the problem stop
- a minimum has Hessian index 0, a saddle has Hessian index 1
- the energy is within the problem tolerance of the reference stationary point
- the Kabsch RMSD to that reference is within the problem tolerance

A start that stops cheaply on the wrong stationary point is a failure. Its search cost is not the number that enters the mean.

The penalised cost of a success is its harness force-call count. The penalised cost of a failure is `2 * cap`. The ranking number is the mean of those penalised costs. Factor 2 is the PAR-2 penalty used for unsolved runs in solver competitions. Factor 10 would make the mean mostly a failure count whenever any starts fail, and would hide the cost of the starts that finished.

Beside that mean the record carries:

- successes `k` out of `n`, and the Wilson bounds for `k/n`
- the median and the interquartile range of the same penalised costs
- a percentile bootstrap 95% interval for the median and for the interquartile range, with 2000 resamples and seed `20261015`

The Wilson bounds use `z = 1.959963984540054`. For the published 1000/1000 successes the interval is 0.996173 to 1. For the published 975/1000 successes it is 0.963355 to 0.983010. For 904/1000 it is 0.884165 to 0.920743. Those three intervals use only `k` and `n` from `results/sota-lj38.json`. They do not need the per-start vector.

## Pt islands

The number in `results/campaign/eon-3.5.0-saddle-pt-dimer.dat` is the mean of ten campaign totals, 4982564.3 force calls. That total grows with the number of searches. The v2 cost of a repetition is `force_calls / total_searches` from that repetition's comment line. The dimer file's ten ratios have mean 263.590450 and median 263.284133, both to 6 decimals. The classic file's ten ratios have mean 470.691074 and median 469.759056, both to 6 decimals. The problem file states no per-search cap and both files record `nfailed 0`, so the PAR-2 penalty is not applied to these ten ratios. The median and its bootstrap interval are the summary.

The rgsaddle displacement rows are a different sample. Each start is one search. The result file stores the cap, and PAR-2 uses it. Those rows do not replace the ratios above.

## Tuning and provenance

`tuning` is `default` or `tuned`. `default` means every optimizer parameter is the value the code ships. `tuned` means a parameter was changed, including a change that copies the 2014 archive. The eOn 3.5.0 client default L-BFGS memory is 20, so `eon-client/lbfgs-m20` is `default` and `eon-client/lbfgs-m100` is `tuned`. Rows without that fact stay unlabelled.

Provenance is the code name, the version, the commit, the parameter object, and the machine. A score without those fields is not a v2 score.

## What the 2014 table keeps

The 2014 `benchmark.dat` mean stays the half-even mean of starts under the cap. Archive rows remain on the site. They are not recomputed.
