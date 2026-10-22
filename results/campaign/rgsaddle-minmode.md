# rgsaddle 0.2.2 min-mode

## Measured at 2db34b42

rgsaddle `2db34b42e6e029ebcf5b9868b94fcdf969103c9f` is the merge of pull request 23. That change folds the Lanczos residual into the push. The rows below were scored with the optbench harness as it stands: same problems, criteria, caps, and counters. PAR-2 charges a failure twice the cap. The `force_calls` line is the half-even mean of the successes.

The library was built from that commit with `cargo +stable build --release --features capi`. `capi` is the C ABI the harness calls. Every other feature is a crate default. No optimizer setting was changed. `RGSADDLE_LIB` and `RGSADDLE_SRC` pointed at that tree.

Templates with `min_mode_method = "lanczos"`:

- `rgsaddle/mode-lanczos` and `rgsaddle/mode-lanczos-tuned` on lowest-eigenvector/lj38
- `rgsaddle/lanczos` and `rgsaddle/lanczos-tuned` on tsopt/LJ38
- `rgsaddle/pt-lanczos` and `rgsaddle/pt-lanczos-tuned` on tsopt/pt-island
- `rgsaddle/saddle-lanczos` and `rgsaddle/saddle-lanczos-tuned` on saddle-search/pt-island

`rgsaddle/oci-neb-lbfgs` sets `refine = "lanczos"` and uses the same push. Its four problems are scored in the next section. `rgsaddle/oci-neb` refines with the dimer and was not rerun. `rgsaddle/prfo-tuned` and `rgsaddle/pt-prfo-tuned` set `expand = "lanczos"` for the eigenvector expansion. That is not this push, and those rows were not rerun.

The two saddle-search grids were started and not finished. A full cell is 20000 searches. No partial file is committed. The c0db2b8 saddle cells remain the recorded scores.

| problem | template | legacy mean | min | median | max | successes | nfailed | par2 mean | cap | wrong stationary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos | 18 | 5 | 17.0 | 37 | 200 | 0 | 17.5850000000 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos-tuned | 24 | 12 | 22.5 | 51 | 200 | 0 | 23.7850000000 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/lanczos | 597 | 75 | 355 | 6544 | 199 | 1 | 694.1000000000 | 1 | 0 |
| tsopt/LJ38 | rgsaddle/lanczos-tuned | 436 | 59 | 309 | 3002 | 199 | 1 | 533.5650000000 | 0 | 1 |
| tsopt/pt-island | rgsaddle/pt-lanczos | 210 | 123 | 192 | 426 | 49 | 0 | 209.9795918367 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-lanczos-tuned | 148 | 101 | 133 | 237 | 49 | 0 | 147.5918367347 | 0 | 0 |

Pull request 23 reported mode-lanczos 17.585 with 200/200, tsopt/LJ38 lanczos 694.10 with 199/200, and pt-lanczos 209.98 with 49/49. Those match `par2_mean` here. 209.98 is 209.9795918367 rounded. The legacy lines are 18, 597, and 210. The unrounded means of the raw calls are 3517/200 = 17.585, 128829/200 = 644.145, and 10289/49 = 209.9795918367. On LJ38 the one failure hit the cap, so PAR-2 charges 20000 for that start.

The pull request's mode-lanczos baseline of 18.81 is not the c0db2b8 cell. That cell is par2 26.1200000000. Only the merge was scored, so 18.81 was not measured on this tree.

tsopt/LJ38 lanczos and tsopt/pt-island pt-lanczos match the 38c3137 cells on `origin/cells/rgsaddle/minmode-main` start for start on `force_calls`, success, failure, rotations, and library evaluations. Relative to that tree the residual fold did not move those two grids.

Commands, from `/workspace`, with the library env above:

```
uv run optbench run lowest-eigenvector/lj38 rgsaddle/mode-lanczos --output results/campaign/rgsaddle-2db34b42-mode-lanczos-loweig.dat
uv run optbench run lowest-eigenvector/lj38 rgsaddle/mode-lanczos-tuned --workers 6 --output results/campaign/rgsaddle-2db34b42-mode-lanczos-tuned-loweig.dat
uv run optbench run tsopt/LJ38 rgsaddle/lanczos --workers 6 --output results/campaign/rgsaddle-2db34b42-lanczos-ts-lj38.dat
uv run optbench run tsopt/LJ38 rgsaddle/lanczos-tuned --workers 6 --output results/campaign/rgsaddle-2db34b42-lanczos-tuned-ts-lj38.dat
uv run optbench run tsopt/pt-island rgsaddle/pt-lanczos --workers 6 --output results/campaign/rgsaddle-2db34b42-pt-lanczos-ts-pt.dat
uv run optbench run tsopt/pt-island rgsaddle/pt-lanczos-tuned --workers 6 --output results/campaign/rgsaddle-2db34b42-pt-lanczos-tuned-ts-pt.dat
```

The grid script pinned `--workers 6` on the last five commands. The default mode-lanczos file was produced by the same `optbench run` form.

## OCI-NEB L-BFGS at 2db34b42

`rgsaddle/oci-neb-lbfgs` refines the climbing image with Lanczos. The band harness reads `RGSADDLE_LIBRARY`, set to the same `librgsaddle.so`. All four problems used `--workers 6`. Pull request 23 did not publish these rows. The c0db2b8 cells are the previous board numbers. The commit range from c0db2b8 to 2db34b42 contains more than the residual fold, so the drop is not that one change.

| problem | legacy mean | min | median | max | successes | nfailed | par2 mean | failure class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neb/Pt-heptamer | 121 | 90 | 126 | 139 | 13 | 0 | 121.3076923076923 |  |
| double-ended-saddle-search/lj38 | 182 | 97 | 135 | 1693 | 50 | 0 | 182.16 |  |
| double-ended-saddle-search/pt-island | 120 | 73 | 115 | 271 | 59 | 0 | 119.72881355932203 |  |
| double-ended-saddle-search/lj38_2ts_separation | 586 | 118 | 327 | 5162 | 49 | 2 | 1347.3529411764705 | wrong_stationary 2 |

On lj38_2ts_separation the two start records are `wrong_stationary` (Hessian index 2 and 3, 167 and 124 force calls). PAR-2 still charges each one `2 * 10000`. The `.v2.json` tally says `n_divergence` 2 because the band writer scores `force_calls` and `success` only, and that tally has no Hessian class. The board follows the start records.

```
uv run optbench run neb/Pt-heptamer rgsaddle/oci-neb-lbfgs --workers 6 --output results/campaign/rgsaddle-2db34b42-neb-Pt-heptamer--rgsaddle-oci-neb-lbfgs.dat
uv run optbench run double-ended-saddle-search/lj38 rgsaddle/oci-neb-lbfgs --workers 6 --output results/campaign/rgsaddle-2db34b42-double-ended-saddle-search-lj38--rgsaddle-oci-neb-lbfgs.dat
uv run optbench run double-ended-saddle-search/pt-island rgsaddle/oci-neb-lbfgs --workers 6 --output results/campaign/rgsaddle-2db34b42-double-ended-saddle-search-pt-island--rgsaddle-oci-neb-lbfgs.dat
uv run optbench run double-ended-saddle-search/lj38_2ts_separation rgsaddle/oci-neb-lbfgs --workers 6 --output results/campaign/rgsaddle-2db34b42-double-ended-saddle-search-lj38_2ts_separation--rgsaddle-oci-neb-lbfgs.dat
```

## c0db2b8 board

asv result `results/asv/cursor/a914203c-uv-pip-py3.12.json`, optbench commit `a914203cbb891442232aea2c92f960df9a48dc1d`.
rgsaddle `c0db2b8d333c557251635cb8bb7b3d81f0088a71` (0.2.2).
Cost is `RgpotCounter.calculate` on `HarnessCounter(cache=True)`. Hessian checks are `verify_calls` and are not in `force_calls`.
Seed is `0`. The grid is every start: 200 on LJ38, 49 on the Pt island. The legacy mean is half-even over successes. `par2_mean` charges a failure `2 * 10000`.

| problem | template | tuning | n | legacy mean | legacy median | min | max | par2 mean | nfailed | cap | wrong stationary | divergence | crash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tsopt/LJ38 | rgsaddle/dimer-lbfgs | default | 200 |  |  |  |  | 20000.0000 | 200 | 199 | 1 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/dimer-lbfgs-tuned | tuned | 200 | 4953 | 4953 | 4953 | 4953 | 19924.7650 | 199 | 199 | 0 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/dimer-fire | default | 200 | 6374 | 6327.5 | 4824 | 9108 | 19454.9800 | 192 | 28 | 0 | 164 | 0 |
| tsopt/LJ38 | rgsaddle/dimer-fire-tuned | tuned | 200 | 4421 | 4030 | 1588 | 9816 | 16338.9650 | 153 | 7 | 0 | 146 | 0 |
| tsopt/LJ38 | rgsaddle/lanczos | default | 200 | 3395 | 3991 | 860 | 6482 | 19418.8200 | 193 | 193 | 0 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/lanczos-tuned | tuned | 200 | 2922 | 1667.5 | 447 | 8862 | 19146.1200 | 190 | 190 | 0 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/dimer-rotskip | default | 200 | 640 | 340.5 | 83 | 7349 | 7803.1900 | 74 | 27 | 47 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/dimer-rotskip-tuned | tuned | 200 | 812 | 189 | 80 | 9574 | 16738.1050 | 166 | 96 | 70 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/prfo | default | 200 | 588 | 486 | 260 | 1592 | 15146.9950 | 150 | 3 | 147 | 0 | 0 |
| tsopt/LJ38 | rgsaddle/prfo-tuned | tuned | 200 | 193 | 173 | 135 | 1691 | 192.6000 | 0 | 0 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-lbfgs | default | 49 | 994 | 1077 | 502 | 1846 | 11854.5102 | 28 | 28 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-lbfgs-tuned | tuned | 49 | 561 | 360 | 225 | 5462 | 7701.7143 | 18 | 18 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-fire | default | 49 | 2926 | 2296.5 | 1206 | 9879 | 3274.1429 | 1 | 1 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-fire-tuned | tuned | 49 | 1097 | 918 | 536 | 3039 | 1096.8980 | 0 | 0 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-lanczos | default | 49 | 658 | 673.5 | 298 | 1273 | 9736.7959 | 23 | 23 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-lanczos-tuned | tuned | 49 | 381 | 381 | 198 | 635 | 9990.3265 | 24 | 24 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-rotskip | default | 49 | 224 | 145 | 105 | 3962 | 223.8980 | 0 | 0 | 0 | 0 | 0 |
| tsopt/pt-island | rgsaddle/pt-dimer-rotskip-tuned | tuned | 49 | 300 | 167 | 103 | 1589 | 299.6327 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-dimer | default | 200 | 31 | 23 | 23 | 65 | 31.4000 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-dimer-tuned | tuned | 200 | 46 | 33 | 33 | 126 | 45.5550 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos | default | 200 | 26 | 26 | 14 | 98 | 26.1200 | 0 | 0 | 0 | 0 | 0 |
| lowest-eigenvector/lj38 | rgsaddle/mode-lanczos-tuned | tuned | 200 | 37 | 34 | 18 | 130 | 36.5600 | 0 | 0 | 0 | 0 | 0 |

PAR-2 median is the bootstrap median of the penalised costs (2000 resamples, seed 20261015), not the legacy median of successes.

| template | par2 median | median low | median high | wilson low | wilson high | rotation steps | library evaluations | harness calls | mean wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle/dimer-lbfgs | 20000.0000 | 20000.0000 | 20000.0000 | 0.0000 | 0.0188 | 1813890 | 1998287 | 1998288 | 2.857 |
| rgsaddle/dimer-lbfgs-tuned | 20000.0000 | 20000.0000 | 20000.0000 | 0.0009 | 0.0278 | 1652381 | 1996706 | 1996707 | 2.850 |
| rgsaddle/dimer-fire | 20000.0000 | 20000.0000 | 20000.0000 | 0.0204 | 0.0769 | 804501 | 958454 | 958468 | 1.949 |
| rgsaddle/dimer-fire-tuned | 20000.0000 | 20000.0000 | 20000.0000 | 0.1816 | 0.2984 | 485686 | 664216 | 664263 | 1.638 |
| rgsaddle/lanczos | 20000.0000 | 20000.0000 | 20000.0000 | 0.0171 | 0.0705 | 1805199 | 1955870 | 1955877 | 2.710 |
| rgsaddle/lanczos-tuned | 20000.0000 | 20000.0000 | 20000.0000 | 0.0274 | 0.0896 | 1714632 | 1929214 | 1929224 | 2.647 |
| rgsaddle/dimer-rotskip | 856.5000 | 600.0000 | 1483.0000 | 0.5612 | 0.6939 | 244749 | 585190 | 585363 | 1.483 |
| rgsaddle/dimer-rotskip-tuned | 20000.0000 | 20000.0000 | 20000.0000 | 0.1243 | 0.2282 | 427798 | 1301202 | 1301306 | 2.078 |
| rgsaddle/prfo | 20000.0000 | 20000.0000 | 20000.0000 | 0.1951 | 0.3143 | 0 | 288906 | 144653 | 239.720 |
| rgsaddle/prfo-tuned | 173.0000 | 168.5000 | 177.5000 | 0.9812 | 1.0000 | 0 | 76640 | 38520 | 76.243 |
| rgsaddle/pt-dimer-lbfgs | 20000.0000 | 1519.0000 | 20000.0000 | 0.3002 | 0.5673 | 273303 | 301155 | 301176 | 185.193 |
| rgsaddle/pt-dimer-lbfgs-tuned | 542.0000 | 423.0000 | 20000.0000 | 0.4927 | 0.7533 | 163765 | 197515 | 197546 | 169.504 |
| rgsaddle/pt-dimer-fire | 2337.0000 | 2109.0000 | 3091.0000 | 0.8931 | 0.9964 | 121174 | 150389 | 150437 | 163.579 |
| rgsaddle/pt-dimer-fire-tuned | 918.0000 | 692.0000 | 981.0000 | 0.9273 | 1.0000 | 35819 | 53699 | 53748 | 149.589 |
| rgsaddle/pt-lanczos | 974.0000 | 697.0000 | 20000.0000 | 0.3938 | 0.6630 | 228150 | 247330 | 247356 | 177.576 |
| rgsaddle/pt-lanczos-tuned | 635.0000 | 458.0000 | 20000.0000 | 0.3747 | 0.6442 | 221497 | 249501 | 249526 | 178.370 |
| rgsaddle/pt-dimer-rotskip | 145.0000 | 141.0000 | 151.0000 | 0.9273 | 1.0000 | 3856 | 10922 | 10971 | 142.517 |
| rgsaddle/pt-dimer-rotskip-tuned | 167.0000 | 152.0000 | 188.0000 | 0.9273 | 1.0000 | 4905 | 14633 | 14682 | 143.524 |
| rgsaddle/mode-dimer | 23.0000 | 23.0000 | 23.0000 | 0.9812 | 1.0000 | 5600 | 6080 | 6280 | 0.975 |
| rgsaddle/mode-dimer-tuned | 33.0000 | 33.0000 | 33.0000 | 0.9812 | 1.0000 | 8430 | 8911 | 9111 | 0.977 |
| rgsaddle/mode-lanczos | 26.0000 | 26.0000 | 26.0000 | 0.9812 | 1.0000 | 4824 | 5024 | 5224 | 0.981 |
| rgsaddle/mode-lanczos-tuned | 34.0000 | 34.0000 | 34.0000 | 0.9812 | 1.0000 | 6912 | 7112 | 7312 | 0.975 |

## Recorded Sella and eOn rows

These are files already in the tree. They are not reruns from this session.

| row | legacy mean | median | min | max | nfailed | file |
| --- | --- | --- | --- | --- | --- | --- |
| Sella 2.6.0 tsopt/LJ38 | 93 | 68.0 | 34 | 1846 | 2 | results/campaign/sella-2.6.0-ts-lj38.dat |
| Sella 0.1.1 tsopt/LJ38 | 69.55 | 64.0 | 24 | 158 | 0 | archive/entries/tsopt/LJ38/sella/benchmark.dat |
| Sella 0.1.1 tsopt/pt-island | 53 | 51.0 | 31 | 108 | 0 | archive/entries/tsopt/pt-island/sella/benchmark.dat |
| Sella 0.1.1 lowest-eigenvector/lj38 | 17.8 |  | 4.0 | 37.0 | 0 | archive/entries/lowest-eigenvector/lj38/sella/benchmark.dat |
| eOn r2156 dimer tsopt/LJ38 | 528 | 388.5 | 92 | 3581 | 0 | archive/entries/tsopt/LJ38/eon-dimer/benchmark.dat |
| eOn r2156 dimer tsopt/pt-island | 116 | 114 | 83 | 160 | 0 | archive/entries/tsopt/pt-island/eon-dimer/benchmark.dat |

Sella 2.6.0 on LJ38 is mean 93 with 2 failures (`results/campaign/sella-2.6.0-ts-lj38.dat`). Archive Sella 0.1.1 is 69.55 with 0 failures.

The best rgsaddle LJ38 legacy mean is tuned P-RFO, 193 with 0 failures and PAR-2 192.6000. That is above both Sella rows. Default P-RFO's legacy mean on its 50 successes is 588, with 147 wrong-stationary failures. The best dimer row is rotation-skip, legacy 640 with 74 failures. Default dimer L-BFGS has no success (199 cap, 1 wrong stationary): 1813890 rotation steps against 1998288 harness calls.

On the Pt island the best row is dimer rotation-skip, legacy 224 with 0 failures. Archive Sella 0.1.1 is 53 and archive eOn r2156 dimer is 116.

On lowest-eigenvector/lj38 every row succeeds. Lanczos default is 26. Archive Sella 0.1.1 is 17.8.

## Not in this asv suite

Pt P-RFO (`rgsaddle/pt-prfo`, `rgsaddle/pt-prfo-tuned`) was not run. A finished score would be a 525-degree Hessian on every force-converged start, and no asv track was saved.

saddle-search/pt-island was not run. The templates exist. The archived eOn dimer campaign is millions of Morse calls per repetition (`archive/entries/saddle-search/pt-island/eon-dimer/benchmark.dat`, `force_calls 3.77888e+06`), and this session did not execute that loop.

## What the library did on these starts

Nonlinear CG is not a row. `check_force_driven` in rgsaddle `src/band.rs` accepts FIRE, L-BFGS, Barzilai-Borwein, and quick-min, and refuses a method that tests the oracle value. The C ABI maps the accepted code other than L-BFGS to FIRE. The FIRE rows are that translator.

The C callback surface (`CbSurface`) does not implement `excluded_modes`, so dimer and Lanczos rotate in the full Cartesian space. The vibrational index used for success removes a rigid direction only when its overlap with the translation-rotation subspace is at least 0.9 and its absolute eigenvalue is below 1e-3, and only for a free cluster. That rule is not loosened.

`RgsaddleSellaSaddleConfig` sets `delta`, `force_tol`, `order`, and `force_gate`. It does not set `gamma`. The library default is 0.1 (`SellaSaddleConfig`). The Sella 2.6.0 LJ38 driver in `optbench/sella_lj38.py` uses `gamma=0.4`.

Default dimer and Lanczos use `rotation_angle_tol = 0`. Rotation then stops on `rotation_tol` or `max_rotations`. The rotation-step totals above are the count of those rotations. Where harness calls sit on the 10000 cap and rotation steps are orders of magnitude larger, the climb is spending the rotation budget instead of translating. The tuned angle stop (0.2 rad, 4 rotations) raised LJ38 failures from 74 to 166.

FIRE is not CG. A FIRE step that repeats one geometry is a harness cache hit, so the host stops and the failure class is divergence: 164 of 200 default LJ38 dimer-fire starts. `Accept::Step` is set for L-BFGS only.

P-RFO's host Jacobi eigen (`jacobi_eigh`, up to `8 n^2` sweeps) and the Rayleigh-Ritz seed (the identity on the first iterative diagonalisation) make one step cost minutes when the free Hessian is ill-conditioned. Mean wall is 239.720 s for default P-RFO and 76.243 s for the tuned row. A dense symmetric eigensolver, and a `gamma` the host can set, are the concrete library changes. They are not in the scored commit `c0db2b8d`.
