# rgsaddle 0.2.2 min-mode

## main 38c3137f

rgsaddle `38c3137f62ce1436d662bb37188a8d7cd2d789b8` (0.2.2), the merge of pull request 21. The min-mode translator keeps its step memory inside the mode cone. `benchmark.dat` copies the template commit, so these files name that tree.

The board ranks the latest stock template of a code. These files are the tsopt rows: Pt-island Lanczos, dimer L-BFGS, dimer FIRE, and dimer rotation-skip, plus LJ38 P-RFO, dimer L-BFGS, dimer FIRE, dimer rotation-skip, and Lanczos.

Cost is harness force calls. A finite-difference Hessian check is `verify_calls` and stays out of `force_calls`. The grid is every start: 200 on LJ38, 49 on the Pt island. `par2_mean` charges a failure `2 * 10000`. The success mean is the half-even mean of the successes, the `force_calls` line in the dat file.

| problem | template | n | success mean | median | min | max | par2 mean | successes | nfailed | cap | wrong stationary | divergence | crash | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tsopt/pt-island | rgsaddle/pt-lanczos | 49 | 210 | 192 | 123 | 426 | 209.9795918367 | 49 | 0 | 0 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-pt-lanczos-ts-pt.dat |
| tsopt/pt-island | rgsaddle/pt-dimer-lbfgs | 49 | 148 | 145 | 99 | 194 | 147.8775510204 | 49 | 0 | 0 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-pt-dimer-lbfgs-ts-pt.dat |
| tsopt/pt-island | rgsaddle/pt-dimer-fire | 49 | 483 | 375 | 299 | 1254 | 483.1428571429 | 49 | 0 | 0 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-pt-dimer-fire-ts-pt.dat |
| tsopt/pt-island | rgsaddle/pt-dimer-rotskip | 49 | 153 | 145 | 99 | 336 | 153.1632653061 | 49 | 0 | 0 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-pt-dimer-rotskip-ts-pt.dat |
| tsopt/LJ38 | rgsaddle/prfo | 200 | 68 | 59.0 | 25 | 283 | 68.4700000000 | 200 | 0 | 0 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-prfo-ts-lj38.dat |
| tsopt/LJ38 | rgsaddle/dimer-lbfgs | 200 | 455 | 256 | 75 | 6782 | 552.7800000000 | 199 | 1 | 1 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-dimer-lbfgs-ts-lj38.dat |
| tsopt/LJ38 | rgsaddle/dimer-fire | 200 | 1414 | 1166.0 | 448 | 5332 | 1785.4550000000 | 196 | 4 | 4 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-dimer-fire-ts-lj38.dat |
| tsopt/LJ38 | rgsaddle/dimer-rotskip | 200 | 312 | 208.0 | 71 | 8680 | 3265.1300000000 | 170 | 30 | 12 | 18 | 0 | 0 | results/campaign/rgsaddle-38c3137-dimer-rotskip-ts-lj38.dat |
| tsopt/LJ38 | rgsaddle/lanczos | 200 | 597 | 355 | 75 | 6544 | 694.1000000000 | 199 | 1 | 1 | 0 | 0 | 0 | results/campaign/rgsaddle-38c3137-lanczos-ts-lj38.dat |

PAR-2 median in each dat file is the bootstrap median of the penalised costs (2000 resamples, seed 20261015): 192, 145, 375, and 145 on the Pt island, in the table order, and 59, 256, 1169.5, 220.5, and 359.5 on LJ38.

## lowest-eigenvector at c0db2b8d

The lowest-eigenvector files stay. They are a different suite. asv result `results/asv/cursor/a914203c-uv-pip-py3.12.json`, optbench commit `a914203cbb891442232aea2c92f960df9a48dc1d`. rgsaddle `c0db2b8d333c557251635cb8bb7b3d81f0088a71` (0.2.2). Cost is harness force calls. Hessian checks are `verify_calls` and are not in `force_calls`. Seed is `0`. The grid is every start, 200 on LJ38. The 2014 mean is half-even over successes. `par2_mean` charges a failure `2 * 10000`.

| problem | template | tuning | n | 2014 mean | 2014 median | min | max | par2 mean | nfailed | cap | wrong stationary | divergence | crash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lowest-eigenvector/lj38 | rgsaddle/mode-dimer | default | 200 | 31 | 23 | 23 | 65 | 31.4000 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-dimer-tuned | tuned | 200 | 46 | 33 | 33 | 126 | 45.5550 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos | default | 200 | 26 | 26 | 14 | 98 | 26.1200 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos-tuned | tuned | 200 | 37 | 34 | 18 | 130 | 36.5600 | 0 | 0 | 0 | 0 | 0 |

PAR-2 median is the bootstrap median of the penalised costs (2000 resamples, seed 20261015). It is the median of the penalised costs, and the 2014 median above is the median of the successes.

| template | par2 median | median low | median high | wilson low | wilson high | rotation steps | library evaluations | harness calls | mean wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle/mode-dimer | 23.0000 | 23.0000 | 23.0000 | 0.9812 | 1.0000 | 5600 | 6080 | 6280 | 0.975 |
| rgsaddle/mode-dimer-tuned | 33.0000 | 33.0000 | 33.0000 | 0.9812 | 1.0000 | 8430 | 8911 | 9111 | 0.977 |
| rgsaddle/mode-lanczos | 26.0000 | 26.0000 | 26.0000 | 0.9812 | 1.0000 | 4824 | 5024 | 5224 | 0.981 |
| rgsaddle/mode-lanczos-tuned | 34.0000 | 34.0000 | 34.0000 | 0.9812 | 1.0000 | 6912 | 7112 | 7312 | 0.975 |

## Recorded Sella and eOn rows

These files were already in the tree.

| row | 2014 mean | median | min | max | nfailed | file |
| --- | --- | --- | --- | --- | --- | --- |
| Sella 2.6.0 tsopt/LJ38 | 93 | 68.0 | 34 | 1846 | 2 | results/campaign/sella-2.6.0-ts-lj38.dat |
| Sella 0.1.1 tsopt/LJ38 | 69.55 | 64.0 | 24 | 158 | 0 | archive/entries/tsopt/LJ38/sella/benchmark.dat |
| Sella 0.1.1 tsopt/pt-island | 53 | 51.0 | 31 | 108 | 0 | archive/entries/tsopt/pt-island/sella/benchmark.dat |
| Sella 0.1.1 lowest-eigenvector/lj38 | 17.8 |  | 4.0 | 37.0 | 0 | archive/entries/lowest-eigenvector/lj38/sella/benchmark.dat |
| eOn r2156 dimer tsopt/LJ38 | 528 | 388.5 | 92 | 3581 | 0 | archive/entries/tsopt/LJ38/eon-dimer/benchmark.dat |
| eOn r2156 dimer tsopt/pt-island | 116 | 114 | 83 | 160 | 0 | archive/entries/tsopt/pt-island/eon-dimer/benchmark.dat |

Sella 2.6.0 on LJ38 is mean 93 with 2 failures (`results/campaign/sella-2.6.0-ts-lj38.dat`). The ranked Sella row is `sella/lj38-ts-default`, PAR-2 68.74, 200 starts, 0 failures (`results/campaign/sella-2.6.0-lj38-ts-default.dat`, success mean 69, median 59.5). Archive Sella 0.1.1 is 69.55 with 0 failures.

LJ38 P-RFO at 38c3137f is PAR-2 68.4700000000, 200 successes, success mean 68, median 59. That row is the table above.

On the Pt island the stock dimer L-BFGS row is PAR-2 147.8775510204, 49 successes. Archive Sella 0.1.1 is 53 and archive eOn r2156 dimer is 116.

On lowest-eigenvector/lj38 every c0db2b8 row succeeds. Lanczos default is 26. Archive Sella 0.1.1 is 17.8.

## Outside the tsopt rows

Pt P-RFO (`rgsaddle/pt-prfo`) has no score. A finished score would be a 525-degree Hessian on every force-converged start, and no asv track was saved.

saddle-search/pt-island has no min-mode score in this file. The templates exist. The archived eOn dimer campaign is millions of Morse calls per repetition (`archive/entries/saddle-search/pt-island/eon-dimer/benchmark.dat`, `force_calls 3.77888e+06`).

## What the library did

Nonlinear CG is not a row. `check_force_driven` in rgsaddle `src/band.rs` accepts FIRE, L-BFGS, Barzilai-Borwein, and quick-min, and refuses a method that tests the oracle value. The C ABI maps the accepted code other than L-BFGS to FIRE. The FIRE rows are that translator.

`RgsaddleSellaSaddleConfig` sets `delta`, `force_tol`, `order`, and `force_gate`. It does not set `gamma`. The library default is 0.1 (`SellaSaddleConfig`). The Sella 2.6.0 LJ38 driver in `optbench/sella_lj38.py` uses `gamma=0.4`.
