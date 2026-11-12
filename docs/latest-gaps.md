# Latest stock rows beside the 2014 archive

The archived `benchmark.dat` values are the paper rows. These runs do not
rebuild a paper-era client, tarball, pin, or compiler, and they do not
change an optimizer parameter to move a number toward the archive. Each
code is its latest release, stock. The problems, starts, and convergence
tests are the harness.

pele is 0.1.2 at `b7081b932a060be11342fbe8287ac35da8ada879`. ASE is 3.29.0
at `f27c0005ae6a67ea419f996e728668865bfc1f86`. EON is SVN r2549, display
`EON (SVN r2549)`, built with the Fortran potentials. eOn is develop
`500350d9fae881f06c7ef194e5868da079c4be5a` on
`fix/develop-regressions-vs-350`, display `eOn (develop 500350d9)`.
`eonclient --version` prints `3.6.0 (500350d9)`. eOn is a separate code
from EON. One force call is one harness evaluation.

The ranked cost is the v2 PAR-2 mean. A NEB row also has the per-image
mean, the harness total divided by 8. An hTST barrier row ranks each
closed reference window. `force_calls` on that row is the harness mean of
those windows, and `barrier_sad_fcs` is the mean of the client sad-fcs.
A rate row uses the same 2014 rule.

| Row | Archive | Latest stock | What the gap is |
| --- | --- | --- | --- |
| globalopt/lj38 pele basin hopping | 521595 calls, 0 failures, max 2533900 | PAR-2 528657.52, 0 failures, 100 starts. Min 6208, max 2765495, median 385463.5 | Archived driver: temperature 1.0, `RandomDisplacement` stepsize 0.41, `AdaptiveStepsize` acceptance ratio 0.5 and interval 100, quench `lbfgs_cpp` tolerance 0.001, 100000 quenches. Other quench arguments stay at the constructor. The call ceiling is 4000000 |
| minimization/lj38 pele FIRE | 709 calls, 0 failures | PAR-2 2141.38 over 1000 starts. The mean of starts under the cap is 1264. v2 failures 48 (7 cap, 41 wrong stationary) | FIRE `dt` 0.1, `maxstep` 0.5, `dtmax` 1, the constructor |
| minimization/morse-bulk ASE FIRE | 147.47 | PAR-2 113.36, 0 failures, 100 starts | ASE 3.29.0 FIRE constructor. `maxstep` 0.2, `dt` 0.1, `dtmax` 1 |
| minimization/morse-bulk ASE LBFGS | 53.8 | PAR-2 53.36, 0 failures, 100 starts | Constructor memory 100, `alpha` 70, `maxstep` 0.2 |
| minimization/morse-bulk ASE PreconLBFGS | 34, 0 failures | PAR-2 59.83, 0 failures, 100 starts | Constructor memory 100, `maxstep` 0.04, Exp preconditioner `A=3` |
| minimization/morse-bulk EON SD | 196 calls, 0 failures, min 95, max 360 | PAR-2 122, 0 failures, 100 starts. Min 88, max 136, median 123 | Archived r2030 steepest descent: `sd_alpha` 0.04, `max_move` 0.2, `converged_force` 0.001. r2549 reads `sd_alpha` from Optimizer. One develop log on `pos_0000` reaches that force stop: the step falls to about 4e-6, the energy falls to -1396.17324, and the client writes `min.con` |
| minimization/morse-bulk eOn SD | 196 calls, 0 failures | PAR-2 122, 0 failures, 100 starts. Min 88, max 136, median 123 | Same archived step. This client reads `sd_alpha` from the SD section. The same `pos_0000` log is a force stop, 88 calls |
| minimization/morse-bulk pele LBFGS | 51, 0 failures | PAR-2 92.06, 0 failures, 100 starts | Constructor memory 4, `maxstep` 0.1, `maxErise` 1e-4, `H0` 0.1 |
| neb/Pt-heptamer EON CG | 120 per image, min 53, max 211 | Total PAR-2 929.385, per-image mean 116.173, 0 failures, 13 pairs | Climbing-image CG, `spring` 5, 8 images, force norm 0.01, `max_move` 0.2. The harness kernel is r2549 `Morse::force` |
| neb/Pt-heptamer eOn CG | 120 per image | Total PAR-2 1141.08, per-image mean 142.635, 0 failures, 13 pairs | Same climbing-image CG stop on develop `500350d9`. The kernel is the same Fortran Morse force |
| hTST/Al EON dimer CG | barrier window 28870 calls and 48 jobs over 639 windows; rate window 897900 calls and 1511 jobs over 20 windows | 99 barrier windows in 4660 searches: PAR-2 83937, mean jobs 47, mean sad-fcs 76415, harness total 8316818, 0 failures. 2 rate windows: PAR-2 4156285.5, mean jobs 2327, mean sad-fcs 3783544 | Archived dimer CG: confidence 1.95, displace radius 3.3, magnitude 0.5, listed atom -1, max energy 100, prefactor 1.2e12, optimizer force 0.001 and 5000 iterations. Barrier jobs match the archive. sad-fcs per window are higher. The stop is 639 barrier windows and 20 rate windows, or 10000 searches. The reactant has 128 fixed atoms and 257 mobile atoms, periodic cell 22.845357 x 22.845357 x 64 |
| hTST/Al eOn dimer CG | same archive windows | 71 barrier windows in 3342 searches: PAR-2 117877, mean jobs 47, mean sad-fcs 110242, harness total 8371637, 0 failures. 1 rate window: harness 6955057, jobs 2811, sad-fcs 6498368 | Same archived dimer CG. develop 500350d9 treats listed atom -1 as every free atom, so the list is atom 384, the free adatom. Same geometry and the same window rule |

ASE 3.29.0 and pele 0.1.2 minimization rows above are the constructor
campaigns stored for those pins. Basin hopping uses the archived driver.
Both steepest-descent campaigns use the archived r2030 step. Both
heptamer NEB rows use climbing-image CG. Both aluminum reactant searches
use the archived dimer-CG protocol and score windows with the 2014 rule.
The hTST search stops at 10000 registered searches, or when 639 barrier
windows and 20 rate windows have closed.
