# Integration

Branch `integration/v2-board`, started from `design/v2-hardening` `b64538628ce060894f7ccb4c41e6f353be3fa5ca`.
Merges are `git merge` with no rebase and no force-push.
`entries/rgmin-fixes` was not on origin when `entries/rgmin` and `entries/rgsaddle-minmode` were merged.
It was merged when it appeared, after those two, without rewriting them.

| Order | Branch | Tip | Result |
| --- | --- | --- | --- |
| 1 | `entries/ase-sella-v2` | `ade7c5132e47bb4008b9d4c8ddb26492e9e7287d` | already in `design/v2-hardening` |
| 2 | `cursor/eon-develop-asv-dee4` | `c4dde48c2d8171991a931ebab95fa59a68308017` | already in `design/v2-hardening` |
| 3 | `entries/eon-full` | `f00867a5eb6a050c46e282143f8ab88f5646af9b` | already in `design/v2-hardening` |
| 4 | `entries/rgmin` | `bddbe2ddbac6606357b9043b41612ec337539a00` | merged |
| 5 | `entries/rgmin-fixes` | `5960e693f7d543f9498c81e62d467961b858ca0f` | merged after `entries/rgsaddle-minmode` |
| 6 | `entries/rgsaddle-minmode` | `cd6151590c09545fc158b9f46e2317047ec4c600` | merged |
| 7 | `entries/rgsaddle-neb` | `20a9361de70db2d444d6747431e66ae1aa7f217f` | already in `design/v2-hardening` |
| 8 | `entries/ase-sella-full` | absent | not on origin |
| 9 | `entries/eon-develop-full` | absent | not on origin |

No campaign was run. `uv run optbench board` scored the committed `results/campaign` files and `results/asv` tracks.
The ranked cell is the v2 PAR-2 mean of harness force calls. For a band method that cost is the harness total across every moving image.
Per-image mean divides that same total by the problem's `band.interior_images`. Every code on the problem uses that divisor.
Only the newest full version of a template is ranked. Older full versions are history.
A campaign file and an asv track with the same score are one row. Source stays on the row.
An asv result with one start is a smoke run and is not in a ranked mean.
The Pt-island saddle problem states no full set, so the 16-start ASE dimer files stay off the ranked table because those result files set `ranked false`.
`entries/ase-sella-full` and `entries/eon-develop-full` were not on origin when this board was scored.
Pages were not deployed.

Ranked 178. History 12. Partial 79. Unparsed 44. Unmapped 14. Not a harness score 12.

## Corrected leaderboard

PAR-2, mean, median, min, and max are harness totals. Per-image is blank when the problem has no moving images.

### `double-ended-saddle-search/lj38`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgsaddle/ew-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 853.56 | 673.94 | 134.788 | 529 | 304 | 2500 | 3 | 50 |
| 2 | `rgsaddle/ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 1208.52 | 767.34 | 153.468 | 522 | 322 | 2500 | 7 | 50 |
| 3 | `rgsaddle/oci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 1230.8 | 880.8 | 176.16 | 553 | 348 | 2500 | 7 | 50 |
| 4 | `rgsaddle/idpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 1253.92 | 811.24 | 162.248 | 522 | 307 | 2500 | 7 | 50 |
| 5 | `rgsaddle/sidpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 1282.98 | 793.8 | 158.76 | 534.5 | 312 | 2500 | 8 | 50 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ase | `ase/de-lj38-default` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 4912.84 | 2462.84 | 492.568 | 2500 | 642 | 2500 | 49 | 50 |

No ranked row: `rgmin`, `eon`, `sella`.

### `double-ended-saddle-search/lj38_2ts_separation`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgsaddle/ew-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 8024.43 | 4639.8 | 927.961 | 2414 | 584 | 10000 | 17 | 51 |
| 2 | `rgsaddle/ew-ci-neb` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 12098.1 | 6803.94 | 1360.79 | 10000 | 1044 | 10000 | 27 | 51 |
| 3 | `rgsaddle/sidpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 15922.3 | 8154.75 | 1630.95 | 10000 | 932 | 10000 | 39 | 51 |
| 4 | `rgsaddle/idpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 16111.7 | 8109.35 | 1621.87 | 10000 | 907 | 10000 | 39 | 51 |
| 5 | `rgsaddle/ci-neb` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 16662.7 | 8309.86 | 1661.97 | 10000 | 1467 | 10000 | 41 | 51 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ase | `ase/de-lj38-2ts-default` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 20000 | 2500 | 500 | 2500 | 2500 | 2500 | 51 | 51 |

No ranked row: `rgmin`, `eon`, `sella`.

### `double-ended-saddle-search/pt-island`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgsaddle/ew-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 422.136 | 224.017 | 74.6723 | 226 | 100 | 460 | 2 | 59 |
| 2 | `rgsaddle/oci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 677.831 | 474.441 | 158.147 | 238 | 127 | 3000 | 4 | 59 |
| 3 | `rgsaddle/idpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 707.847 | 213.373 | 71.1243 | 206 | 95 | 863 | 5 | 59 |
| 4 | `rgsaddle/ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 733.576 | 238.085 | 79.3616 | 197 | 95 | 2336 | 5 | 59 |
| 5 | `rgsaddle/sidpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 741.458 | 247.237 | 82.4124 | 209 | 95 | 2903 | 5 | 59 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ase | `ase/de-pt-island-tuned` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 847.678 | 593.864 | 197.955 | 374 | 173 | 3005 | 5 | 59 |

No ranked row: `rgmin`, `eon`, `sella`.

### `lowest-eigenvector/lj38`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgsaddle/mode-lanczos` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 26.12 | 26.12 |  | 26 | 14 | 98 | 0 | 200 |
| 2 | `rgsaddle/mode-dimer` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 31.4 | 31.4 |  | 23 | 23 | 65 | 0 | 200 |
| 3 | `rgsaddle/mode-lanczos-tuned` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 36.56 | 36.56 |  | 34 | 18 | 130 | 0 | 200 |
| 4 | `rgsaddle/mode-dimer-tuned` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 45.555 | 45.555 |  | 33 | 33 | 126 | 0 | 200 |

No ranked row: `rgmin`, `eon`, `ase`, `sella`.

### `minimization/lj38`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgmin/lj38-dogleg` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 81.881 | 81.881 |  | 79 | 45 | 162 | 0 | 1000 |
| 2 | `rgmin/lj38-dogleg-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 81.881 | 81.881 |  | 79 | 45 | 162 | 0 | 1000 |
| 3 | `rgmin/lj38-newton` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 166.337 | 66.649 |  | 64 | 35 | 128 | 5 | 1000 |
| 4 | `rgmin/lj38-newton-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 166.337 | 66.649 |  | 64 | 35 | 128 | 5 | 1000 |
| 5 | `eon/lj38-lbfgs-m100` | eOn 3.6.0 | campaign | `1d71c9445018` | 180.137 | 180.137 |  | 171 | 90 | 435 | 0 | 1000 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ase | `ase/lj38-lbfgs-tuned` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 395.645 | 375.645 |  | 319 | 166 | 10000 | 2 | 1000 |
| sella | `sella/lj38-min-tuned` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 253.921 | 253.921 |  | 248 | 150 | 533 | 0 | 1000 |

No ranked row: `rgsaddle`.

### `minimization/morse-bulk`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgmin/morse-newton-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 8.35 | 8.35 |  | 9 | 4 | 13 | 0 | 100 |
| 2 | `rgmin/morse-dogleg` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 10.11 | 10.11 |  | 10 | 6 | 17 | 0 | 100 |
| 3 | `rgmin/morse-lbfgs-m20-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 26.87 | 26.87 |  | 27 | 22 | 31 | 0 | 100 |
| 4 | `rgmin/morse-lbfgs-m100-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 26.91 | 26.91 |  | 27 | 22 | 31 | 0 | 100 |
| 5 | `rgmin/morse-lbfgs-tuned` | rgmin 0.4.0 | campaign, asv | `24236559f381` | 26.93 | 26.93 |  | 27 | 22 | 31 | 0 | 100 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eon | `eon/morse-lbfgs-m100` | eOn 3.6.0 | campaign | `1d71c9445018` | 54.18 | 54.18 |  | 53 | 36 | 101 | 0 | 100 |
| ase | `ase/morse-bfgslinesearch-default` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 28.91 | 28.91 |  | 28 | 21 | 37 | 0 | 100 |
| sella | `sella/morse-min-default` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 28.89 | 28.89 |  | 28 | 20 | 41 | 0 | 100 |

No ranked row: `rgsaddle`.

### `neb/Pt-heptamer`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `rgsaddle/idpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 318.308 | 318.308 | 39.7885 | 314 | 210 | 458 | 0 | 13 |
| 2 | `rgsaddle/sidpp-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 318.923 | 318.923 | 39.8654 | 314 | 210 | 466 | 0 | 13 |
| 3 | `rgsaddle/ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 319.538 | 319.538 | 39.9423 | 330 | 218 | 410 | 0 | 13 |
| 4 | `rgsaddle/oci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 345.538 | 345.538 | 43.1923 | 356 | 244 | 436 | 0 | 13 |
| 5 | `rgsaddle/ew-ci-neb-lbfgs` | rgsaddle 0.2.2 | asv | `c0db2b8d333c` | 357.231 | 357.231 | 44.6538 | 332 | 220 | 556 | 0 | 13 |

Best ranked row of a code that is outside the top 5:

| Code | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ase | `ase/neb-pt-heptamer-tuned` | ASE 3.29.0 | campaign | `f27c0005ae6a` | 442.615 | 442.615 | 55.3269 | 450 | 250 | 594 | 0 | 13 |

No ranked row: `rgmin`, `eon`, `sella`.

### `saddle-search/pt-island`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No ranked row: `rgmin`, `rgsaddle`, `eon`, `ase`, `sella`.

No full harness row.

### `tsopt/LJ38`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `sella/lj38-ts-default` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 68.74 | 68.74 |  | 59.5 | 25 | 257 | 0 | 200 |
| 2 | `rgsaddle/prfo-tuned` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 192.6 | 192.6 |  | 173 | 135 | 1691 | 0 | 200 |
| 3 | `sella/lj38-ts-tuned` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 274.46 | 92.775 |  | 68 | 34 | 1840 | 2 | 200 |
| 4 | `rgsaddle/dimer-rotskip` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 7803.19 | 2926.82 |  | 856.5 | 83 | 10006 | 74 | 200 |
| 5 | `rgsaddle/prfo` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 15147 | 723.265 |  | 460.5 | 230 | 10000 | 150 | 200 |

No ranked row: `rgmin`, `eon`, `ase`.

The rgsaddle lines above are the c0db2b8 files scored for this merge. The stock rows are the 38c3137f files in `results/campaign/rgsaddle-minmode.md`.

### `tsopt/pt-island`

| Rank | Template | Version | Source | Commit | PAR-2 | Mean | Per-image | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `sella/pt-island-ts-default` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 74.2857 | 74.2857 |  | 71 | 41 | 128 | 0 | 49 |
| 2 | `sella/pt-island-ts-tuned` | Sella 2.6.0 | campaign | `6cfb60cdb100` | 87.551 | 87.551 |  | 88 | 67 | 141 | 0 | 49 |
| 3 | `rgsaddle/pt-dimer-rotskip` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 223.898 | 223.898 |  | 145 | 105 | 3962 | 0 | 49 |
| 4 | `rgsaddle/pt-dimer-rotskip-tuned` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 299.633 | 299.633 |  | 167 | 103 | 1589 | 0 | 49 |
| 5 | `rgsaddle/pt-dimer-fire-tuned` | rgsaddle 0.2.2 | campaign, asv | `c0db2b8d333c` | 1096.9 | 1096.9 |  | 918 | 536 | 3039 | 0 | 49 |

No ranked row: `rgmin`, `eon`, `ase`.

The rgsaddle lines above are the c0db2b8 files scored for this merge. The stock rows are the 38c3137f files in `results/campaign/rgsaddle-minmode.md`.

