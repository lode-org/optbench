# Saddle and kMC codes in this lane

Lane list of 2026-10-18. A code is a module in `optbench/engines`, a pin in
`codes/registry.toml`, and a template under `templates/`. `campaign.run`
calls `optbench.engines.find`. `available` returns the reason a code cannot
start. `run` writes harness campaign files or raises that reason. The score
is the harness force count. A client's own tally is not a score.

Pt-island geometries use the Morse potential in `optbench.pes` (`morse_pt`).
A code runs those problems when it can take that potential or an ASE or
LAMMPS calculator that calls it. The harness has no EAM evaluator. `hTST/Al`
and the accelerated-md problems state rates and barriers, not a force-call
cap, so they are not given a force-call row. `hTST/Pt` states a cap of
20000000 and a sample of 13 barrier windows.

## Rows on disk

Failure classes are the v2 counts. They sum to `nfailed`. The median is the
success-only median. A row with no success has no median.

| problem | template | successes | failures | median calls |
| --- | --- | --- | --- | --- |
| lowest-eigenvector/lj38 | partn/mode-lj38 | 174/200 | 26 divergence | 22 |
| tsopt/pt-island | partn/ts-pt | 45/49 | 4 crash | 247 |
| tsopt/LJ38 | partn/ts-lj38 | 0/200 | 200 crash | — |
| lowest-eigenvector/lj38 | tsase/mode-lj38 | 178/200 | 22 divergence | 221.5 |
| tsopt/pt-island | tsase/dimer-pt | 49/49 | 0 | 892 |
| tsopt/LJ38 | tsase/dimer-lj38 | 0/200 | 200 cap | — |
| minimization/lj38 | tsase/lbfgs-lj38 | 996/1000 | 4 wrong stationary | 169 |
| minimization/lj38 | tsase/mdmin-lj38 | 0/1000 | 1000 cap | — |
| minimization/morse-bulk | tsase/lbfgs-morse | 12/100 | 88 wrong stationary | 53.5 |
| neb/Pt-heptamer | tsase/ci-neb | 13/13 | 0 | 1427 |
| neb/Pt-heptamer | tsase/neb | 0/13 | 13 divergence | — |
| double-ended-saddle-search/lj38 | tsase/ci-neb | 26/50 | 17 cap, 7 divergence | 1233 |
| double-ended-saddle-search/lj38 | tsase/neb | 0/50 | 36 divergence, 14 cap | — |
| double-ended-saddle-search/lj38_2ts_separation | tsase/ci-neb | 5/51 | 43 cap, 2 wrong stationary, 1 divergence | 7693 |
| double-ended-saddle-search/lj38_2ts_separation | tsase/neb | 0/51 | 44 cap, 4 wrong stationary, 3 divergence | — |
| double-ended-saddle-search/pt-island | tsase/ci-neb | 38/59 | 20 divergence, 1 cap | 426 |
| double-ended-saddle-search/pt-island | tsase/neb | 0/59 | 56 divergence, 3 cap | — |
| minimization/lj38 | lammps/cg-lj38 | 848/1000 | 148 cap, 3 wrong stationary, 1 divergence | 1410 |
| minimization/lj38 | lammps/sd-lj38 | 383/1000 | 617 cap | 5252 |
| minimization/lj38 | lammps/hftn-lj38 | 997/1000 | 3 wrong stationary | 974 |
| minimization/lj38 | lammps/quickmin-lj38 | 991/1000 | 5 divergence, 4 wrong stationary | 2543 |
| minimization/lj38 | lammps/fire-lj38 | 727/1000 | 224 divergence, 49 wrong stationary | 472 |
| minimization/morse-bulk | lammps/cg-morse | 5/100 | 95 wrong stationary | 95 |
| minimization/morse-bulk | lammps/sd-morse | 4/100 | 96 wrong stationary | 255.5 |
| minimization/morse-bulk | lammps/hftn-morse | 3/100 | 97 wrong stationary | 511 |
| minimization/morse-bulk | lammps/quickmin-morse | 5/100 | 95 wrong stationary | 2820 |
| minimization/morse-bulk | lammps/fire-morse | 12/100 | 88 wrong stationary | 447 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s0 | 848/1000 | 148 cap, 3 wrong stationary, 1 divergence | 1410 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em4 | 896/1000 | 100 cap, 3 divergence, 1 wrong stationary | 2480 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em3 | 997/1000 | 3 divergence | 1776 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em2 | 874/1000 | 88 divergence, 38 wrong stationary | 1310 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s0 | 383/1000 | 617 cap | 5252 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em4 | 12/1000 | 988 cap | 6037 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em3 | 684/1000 | 313 cap, 3 wrong stationary | 5905.5 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em2 | 698/1000 | 227 cap, 75 wrong stationary | 5627.5 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s0 | 997/1000 | 3 wrong stationary | 974 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em4 | 224/1000 | 776 divergence | 6945 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em3 | 283/1000 | 713 divergence, 4 wrong stationary | 6882 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em2 | 587/1000 | 243 divergence, 170 wrong stationary | 5438 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s0 | 991/1000 | 5 divergence, 4 wrong stationary | 2543 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em4 | 983/1000 | 13 divergence, 4 wrong stationary | 2450 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em3 | 741/1000 | 9 cap, 231 divergence, 19 wrong stationary | 5454 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em2 | 576/1000 | 10 cap, 248 divergence, 166 wrong stationary | 4455 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s0 | 727/1000 | 224 divergence, 49 wrong stationary | 472 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em4 | 729/1000 | 224 divergence, 47 wrong stationary | 462 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em3 | 706/1000 | 224 divergence, 70 wrong stationary | 435 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em2 | 564/1000 | 224 divergence, 212 wrong stationary | 335 |
| minimization/lj38 | dlfind/lbfgs-lj38 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38 | dlfind/sd-lj38 | 990/1000 | 10 cap | 2563.5 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s0 | 990/1000 | 10 cap | 2563.5 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em4 | 991/1000 | 8 cap, 1 wrong stationary | 2564 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em3 | 996/1000 | 3 cap, 1 wrong stationary | 2371 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em2 | 955/1000 | 45 wrong stationary | 1778 |
| minimization/lj38 | dlfind/cg-lj38 | 991/1000 | 9 cap | 2541 |
| minimization/morse-bulk | dlfind/cg-morse | 18/100 | 82 wrong stationary | 250 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s0 | 991/1000 | 9 cap | 2541 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em4 | 993/1000 | 6 cap, 1 wrong stationary | 2511 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em3 | 995/1000 | 3 cap, 2 wrong stationary | 2329 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em2 | 956/1000 | 44 wrong stationary | 1718 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s0 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em4 | 998/1000 | 2 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em3 | 999/1000 | 1 wrong stationary | 209 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em2 | 974/1000 | 26 wrong stationary | 199 |
| minimization/morse-bulk | dlfind/lbfgs-morse | 32/100 | 68 wrong stationary | 77 |
| minimization/morse-bulk | dlfind/sd-morse | 13/100 | 87 wrong stationary | 253 |
| minimization/morse-bulk | optim/lbfgs-morse | 13/100 | 87 wrong stationary | 72 |
| minimization/lj38 | optim/lbfgs-lj38 | 998/1000 | 2 wrong stationary | 217 |
| double-ended-saddle-search/lj38 | compass/sqnm | 47/50 | 3 cap | 1221 |
| double-ended-saddle-search/lj38_2ts_separation | compass/sqnm | 41/51 | 9 cap, 1 wrong stationary | 2265 |
| neb/Pt-heptamer | compass/sqnm | 13/13 | 0 | 548 |
| double-ended-saddle-search/pt-island | compass/sqnm | 58/59 | 1 cap | 1000.5 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s0 | 996/1000 | 4 wrong stationary | 206 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s0-tuned | 996/1000 | 4 wrong stationary | 206 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em4 | 994/1000 | 6 wrong stationary | 205 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em4-tuned | 994/1000 | 6 wrong stationary | 205 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em3 | 989/1000 | 11 wrong stationary | 195 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em3-tuned | 989/1000 | 11 wrong stationary | 195 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em2 | 884/1000 | 116 wrong stationary | 161 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em2-tuned | 884/1000 | 116 wrong stationary | 161 |
| lowest-eigenvector/lj38 | pysisyphus/lanczos-lj38 | 160/200 | 40 divergence | 22 |
| minimization/lj38 | pysisyphus/lbfgs-lj38 | 0/1000 | 1000 cap | — |
| minimization/lj38 | pysisyphus/cg-lj38 | 254/1000 | 745 cap, 1 wrong stationary | 693 |
| minimization/lj38 | pysisyphus/sd-lj38 | 205/1000 | 795 cap | 808 |
| hTST/Pt | eon-client/htst-pt-dimer-lbfgs | 13/13 | 0 | 586508 |
| hTST/Pt | eon-client/htst-pt-lanczos-lbfgs | 13/13 | 0 | 580349 |

The rgsaddle band campaigns on `neb/Pt-heptamer` and the three double-ended
problems are on disk (12 templates each). The saddle-search catalog on
`saddle-search/pt-island` is still running and has not written a file. A
sample of that process was on search 387 of the first repetition, with 1
connected saddle. The eight rgmin `lj38-noise` L-BFGS rows are on disk.
Both tunings set `maxstep` 0.2, and each pair matches. Every start met the
force stop. The v2 failures are Hessian index not 0. Zero noise is 996 of
1000. `1e-4` is 994 of 1000. `1e-3` is 989 of 1000. `1e-2` is 884 of 1000,
116 wrong stationary, success median 161.

pARTn on `tsopt/LJ38` stops with `pARTn error -9: BOX EXPLOSION` on every
start. That string is the library check in `block_perprelax.f90`: the
perpendicular step aborts when any atom's force norm exceeds `1e4`. It is
not a harness box, unit, or cluster-cell error. Start `0000` lies in
`[50.000, 53.394]`, spans `3.24712`, `3.34183`, and `3.39366`. The missing-box cell is the diagonal
`223.57464`. A cell of diagonal `1e6` aborted on the same step, call 49,
energy `3.04939e6`. `engine_units` `lammps/lj` leaves force, length, and
time unscaled, and the initial energy is `-167.657` with force norm
`2.98891`. With `fire_dt_init` 0.05 the library's own FIRE step grew to
length 10.2, which is larger than the cluster, and the next atomic force
crossed `1e4`. The library's `LAMMPS.LJ.SaddleRefine` input for these same
geometries sets free boundaries, timestep 0.015, and `nperp_limitation`
`10, 20, 30, 40, 50`. Those settings on this `next_displ` loop still
aborted, at call 147, energy `4.0435e6`, after a perpendicular step of
1.656. The LAMMPS example caps that step with `min_style fire`. The
standalone FIRE cap in `m_artn_fire.f90` is commented out. The mode search
and the Morse tsopt row are the other two pARTn point scores. `nevalf` is
not the count.

TSASE MDMin on LJ38 hits the cap of 10000 on every start. The step limit
keeps the force above the stop. The cap record has no final force norm,
because the cap is raised before that geometry is stored. The class is cap.

## OPTIM and PATHSAMPLE

Pin `optim`. The snapshot is `http://www-wales.ch.cam.ac.uk/svn.tar.bz2`,
122743078 bytes, sha256
`02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f`.
There is no git commit. `OPTIM/source/py.f90` is the Paramonov–Yaliraki
ellipsoid potential, not a Python hook. The external hook is the ASE socket
in `INTERFACES/pysocket/ase_interface.f90`. The tarball does not contain
`fmlip_relay.py`. The harness server speaks that protocol: each `COMPUTE`
is one `pes.evaluate`. `EV` and `ANG` scales are 1. The OPTIM target was
built with gfortran, `WITH_ALIGN` off and `WITH_OPENMP` off. Minimization
L-BFGS sends `BFGSMIN` with an RMS stop equal to the Euclidean force norm
divided by `sqrt(3N)`, `UPDATES` 4, `MAXSTEP` 0.2, and `STEPS` equal to
the cap. The relay refuses the call that would pass the cap and does not
count it. A one-start check on LJ38 `cluster_0000` made 228 harness calls.
OPTIM reported 228 energy-and-gradient calls. The stored point re-evaluates
to energy `-163.2168156980899` and Euclidean force `0.00817050460275337`.
That check is not itself a campaign row. The 1000-start LJ38 L-BFGS row is
998 of 1000, 2 wrong stationary, success median 217. Morse bulk L-BFGS is 13 of 100. The
other 87 meet the force stop with a Hessian index other than 0, so they
are wrong stationary. The success median is 72. Start `pos_0000` took 55
harness calls and stopped at force `0.000842553062284986`, energy
`-1421.4513232624358`, vibrational index 6. `optim/unhooked` writes nothing.
Noise is not scored: the relay has no noisy-force window. Saddle search,
NEB, tsopt, and hTST are not scored: those OPTIM keywords are not connected
to this socket. PATHSAMPLE is in the same snapshot. The PATHSAMPLE target linked
(gfortran, `WITH_OPENMP` off). It does not call `pes.evaluate`. It writes
`odata.*` and submits the `EXEC` program, which is OPTIM. No PATHSAMPLE
number is written.

## pARTn

Pin `artn`, 1.1.0, DEVEL `9b753517`. `libartn.so` is the engine plugin and
the standalone `next_displ` loop. Morse uses `lammps/metal`. Lennard-Jones
uses `lammps/lj`. The saddle-search catalog on `saddle-search/pt-island`
is running. It has not written a file.

## TSASE

Pin `tsase`, commit `98aac60` (2025-09-26). `setup.py` says 1.0. The
checkout has no LICENSE file. Fortran calculators use `numpy.distutils` and
are not built. SSNEB and the solid-state dimer are loaded without
`tsase/__init__.py`. `ss` is false. A periodic all-free Hessian drops the
three translations before the index is counted.

The LJ38 dimer hits the cap on every start. The Pt-island dimer and the
climbing-image NEB on Pt-heptamer are the saddle rows. The plain NEB on
that heptamer diverges on all 13 pairs. Morse bulk L-BFGS meets the force
stop and the Hessian index is not 0 for 88 of 100 starts. The success
median is 53.5.

Still open, and not given an invented budget: basin hopping and minima
hopping on `globalopt/*`. Those problems state `energy_above_minimum` and
do not state `max_force_calls`. AMD-SVM on `accelerated-md/aladatom` is not
scored. The problem has no force-call protocol and the harness has no EAM
potential. hTST prefactors are not scored for the same reason.

## LAMMPS

Pin `lammps`, `stable_30Sep2026`, commit `8de817dd`, version string
`30 Sep 2026`. Minimizers `cg`, `sd`, `hftn`, `quickmin`, and `fire` call
the harness through `fix external pf/callback` and `pair_style zero`. A
native pair style is not the score. FIRE and quickmin use timestep 0.002.
The velocity kick is not limited by `dmax`, and a larger timestep loses
atoms on the first LJ38 step.

On Morse bulk the force stop is met and the Hessian index is not 0 for 95
of 100 conjugate-gradient starts, 96 of 100 steepest-descent starts, 97 of
100 Hessian-free Newton starts, 95 of 100 quickmin starts, and 88 of 100
FIRE starts. FIRE is 12 of 100, success median 447. A fresh Hessian at the
stored CG point `pos_0000` (67 calls, force `0.0006534685520009086`, energy
`-1421.4513146864156`) has vibrational eigenvalues `-7.023206723482451` and
`-6.994643641886293` after the three translations are removed. `pos_0002`
(75 calls, force `0.0008961440060698902`) has seven eigenvalues from
`-8.207944407072567` to `-5.991636654081658`. `pos_0013` meets the stop
with index 0. The curvature tolerance at the `1e-3` force stop is `0.001`
(the Hessian Frobenius norms are `419.652` and `418.714`). Both that
tolerance and the fixed `1e-6` cutoff count the same negative modes. These
are index-2 and index-7 stationary points, not a cutoff artifact. Zero-noise
LJ38 conjugate gradient is 848 of 1000, the same failure split as the clean
LJ38 row, success median 1410. At sigma `1e-4` it is 896 of 1000, 100 cap,
3 divergence, 1 wrong stationary, success median 2480. At sigma `1e-3` it
is 997 of 1000, 3 divergence, success median 1776. At sigma `1e-2` it is
874 of 1000, 88 divergence, 38 wrong stationary, success median 1310.
Zero-noise steepest descent is 383 of 1000, 617 cap, success median 5252,
the same split as the clean LJ38 row. At sigma `1e-4` steepest descent is
12 of 1000, 988 cap, success median 6037. At sigma `1e-3` it is 684 of
1000, 313 cap, 3 wrong stationary, success median 5905.5. At sigma `1e-2`
it is 698 of 1000, 227 cap, 75 wrong stationary, success median 5627.5.
Zero-noise Hessian-free Newton is 997 of 1000, 3 wrong stationary, success
median 974, the same split as the clean LJ38 row. At sigma `1e-4` it is
224 of 1000, 776 divergence, success median 6945. At sigma `1e-3` it is
283 of 1000, 713 divergence, 4 wrong stationary, success median 6882. At
sigma `1e-2` it is 587 of 1000, 243 divergence, 170 wrong stationary,
success median 5438. Zero-noise quickmin is 991 of 1000, 5 divergence, 4
wrong stationary, success median 2543, the same split as the clean LJ38
row. At sigma `1e-4` it is 983 of 1000, 13 divergence, 4 wrong stationary,
success median 2450. At sigma `1e-3` it is 741 of 1000, 9 cap, 231
divergence, 19 wrong stationary, success median 5454. At sigma `1e-2` it
is 576 of 1000, 10 cap, 248 divergence, 166 wrong stationary, success
median 4455. Zero-noise FIRE is 727 of 1000, 224 divergence, 49 wrong
stationary, success median 472, the same split as the clean LJ38 row. At
sigma `1e-4` it is 729 of 1000, 224 divergence, 47 wrong stationary,
success median 462. At sigma `1e-3` it is 706 of 1000, 224 divergence, 70
wrong stationary, success median 435. At sigma `1e-2` it is 564 of 1000,
224 divergence, 212 wrong stationary, success median 335. `fix neb` needs a multi-partition MPI
run and is not scored. `prd`, `tad`,
`hyper`, and `dynamical_matrix` are not scored. The Al, Pt, and aladatom
problems are EAM, and the harness has no EAM evaluator.

## DL-FIND

Pin `dlfind`, libdlfind 0.0.3, commit `4167998d`. `iopt` 3 is L-BFGS, 0 is
steepest descent, and 1 is conjugate gradient. The gradient callback is the
score. Coordinates stay in the harness unit. The component tolerance is the
Euclidean stop divided by `sqrt` of the free components. On Morse bulk L-BFGS the force stop is met and the Hessian index is not 0
for 68 of 100 starts. Steepest descent on LJ38 is 990 of 1000, 10 cap,
success median 2563.5. Steepest descent on Morse bulk is 13 of 100, 87
wrong stationary, success median 253. Steepest descent on zero-noise LJ38
is 990 of 1000, 10 cap, success median 2563.5. At sigma `1e-4` it is 991
of 1000, 8 cap, 1 wrong stationary, success median 2564. At sigma `1e-3`
it is 996 of 1000, 3 cap, 1 wrong stationary, success median 2371. At sigma
`1e-2` it is 955 of 1000, 45 wrong stationary, success median 1778.
Conjugate gradient on LJ38 is 991 of 1000, 9 cap, success median 2541.
On Morse bulk it is 18 of 100, 82 wrong stationary, success median 250.
Zero-noise conjugate gradient matches the clean LJ38 row. At sigma `1e-4`
it is 993 of 1000, 6 cap, 1 wrong stationary, success median 2511. At
sigma `1e-3` it is 995 of 1000, 3 cap, 2 wrong stationary, success median
2329. At sigma `1e-2` it is 956 of 1000, 44 wrong stationary, success
median 1718. NEB (`icoord` 190)
and the dimer (`icoord` near 210) are not in these rows.

## COMPASS

Pin `compass`, 1.0.0, commit `22da7bd` (2024-08-08),
`https://gitlab.com/goedeckergroup/compass`. The ASE calculator is the
score. `forcetol`, `minimize_fmax`, and both target distances are the
problem force stop. The images stop that far apart. On LJ38 a target of
0.2 left a harness force near 1. A target equal to the force stop left a
harness force under the stop and Hessian index 1. ASE 3.29 moved
`ExpCellFilter` to `ase.filters`. The runner aliases it back. The library
treats any periodic boundary as a variable cell. These problems keep the
given cell, so `is_varshape` is false. Fixed atoms are restored on every
`set_positions`. Pt-island is 58 of 59, one cap, success median 1000.5.

## pysisyphus

Pin `pysisyphus`, commit `a4ce10dd` (2025-10-29). The clone has no tag, so
the installed package reports `0.1.dev1`. GPL-3.0. Lanczos on
`lowest-eigenvector/lj38` overlaps the reference on 160 of 200 starts.
The other 40 are divergence. The success median is 22. `dx` is 5e-3 and
`dl` is 1e-2. Six rigid modes are removed. Cartesian L-BFGS, conjugate
gradient, and steepest descent keep the library step limits in Bohr and
call the potential in the harness unit. On start `cluster_0000` each one
used the cap of 10000 with the force still above 0.01. The 1000-start
L-BFGS row does the same on every start. Conjugate gradient is 254 of 1000,
745 cap, 1 wrong stationary, success median 693. Steepest descent is 205 of
1000, 795 cap, success median 808. The dimer drops translation and not rotation. The NEB
springs stay at `k_min` 0.1 and `k_max` 0.3. Those two drivers are not
connected, so there is no pysisyphus row for `tsopt/*`,
`saddle-search/pt-island`, `double-ended-saddle-search/*`, or
`neb/Pt-heptamer`.

## k-ART

`https://gitlab.com/groupe_mousseau/kart` still answers `could not read
Username`. The public docs at `kart-doc.readthedocs.io` point at that same
repository and say access is granted on request. No public copy was found.
No commit is pinned.

## pyKMC

The lane name is SourceForge lkmc release 1.3, not `hockyg/PyKMC`. The
archive is Perl from February 2010. `TransSearchSSH.pl` shells out to
`LBOMD.exe`, and the zip also names `LBOMD.Linux` and `LBOMD.Darwin`.
None of those files is in release 1.3. The SourceForge file
`LKMC0.1.3.zip` (65791 bytes) is the same Perl `Scripts/` tree and has no
Fortran or C source. A 2018 materials-science thread asks whether LBOMD
is public and describes it as Loughborough's in-house molecular dynamics
code; no source repository is given. There is no licence file and no
potential callback. `pykmc/lattice` writes nothing.

## eOn develop on hTST/Pt

`eon-client/htst-pt-dimer-lbfgs` and `eon-client/htst-pt-lanczos-lbfgs` are
the ranked rows. Each file is 13 barrier windows on shifted Morse
(`kernel morse_shift`, r2549 `Morse::force`). The counter is `harness`.
The cap is `problems/hTST/Pt.toml` `convergence.max_force_calls`,
20000000. The client commit is `4c1b814f62f1dcf5fcd0cf18ba0b6df3bcff7a28`.
Dimer median 586508, PAR-2 671526.4615384615. Lanczos median 580349, PAR-2
607392.5384615385. A window closes when every reference barrier, rounded to
3 decimals, has been seen once. The one-pass fill path does not write
these two templates.

## Codes that were not scored

**softsaddle.** The tree under `softsaddle/` is MATLAB. `softsaddle/entry.toml`
is the April 2019 archive (mean force calls 423491.1 over 10 repetitions).
The GitHub workflow builds the Morse, Davidson, and Lanczos libraries and
runs a two-search MATLAB smoke. MATLAB is not installed here. That smoke is
not a harness campaign.

**CatLearn.** Commit `6a307f32` (2023-02-07). `setup.py` says 0.6.2 and
`catlearn/__init__.py` says 0.7.0. ASE 3.29 has no `ase.neb` (it is
`ase.mep.neb`). NumPy 2 rejects `np.argwhere` as a scalar index and rejects
a length-1 GP prediction stored in one gradient slot. After those two
edits, MLMin on LJ38 start 0 took 7 harness calls and 82 seconds and was
still at force 144678. Each new training point retrains the GP. A
1000-start minimization was not launched.

**SEAKMC_py.** Commit `3d94a64`. `LammpsRunner` writes `pair_style` and
shells out to a LAMMPS binary. The requirements include pymatgen. There is
no potential callback the harness can wrap.

**OpenFLY.** Commit `991baf0` (2023-06-01). The potential in the tree is
EAM. There is no Morse class. The accelerated-md problems need EAM, which
the harness does not evaluate.

**SaddleSearch.jl.** Commit `47b4f35`. `Project.toml` requires Julia 1.9 or
1.10. Julia is not installed. The dimer entry point is `run!(method, E, dE,
x0, v0)`.

**CP2K.** v2026.2 was not built. A FIST classical potential would still need
a harness callback, and the build is a separate stack from the running
campaigns.

**GULP.** The academic licence is registration-only. No tarball was
downloaded, and a registered binary is not redistributed here.

**i-PI.** Commit `a6b3503` (v3.3.1, 2026-10-17). Geometry optimisation is
`GeopMotion`, bound to beads, a cell, and a forcefield inside the socket
server. NEB and the string are in that same motion stack. The socket was
not wired to `pes.evaluate`.

**GPR_calculator.** Commit `c4d6327`. `setup.py` compiles C++ kernels and
requires cffi, mpi4py, and pyxtal. That extension was not built.

**geodesic-interpolate.** Commit `476aa04` (v1.0.0). It builds an initial
path from a molecular bond list (`min_neighbors` default 4). It is not a
saddle search. It was not paired with a NEB.

**pyGSM.** Commit `1b4d5a5`. `examples/ase_api_de_gsm.py` grows a string in
TRIC delocalized internals from a molecular topology. The dependency list
includes xtb. That path was not run on LJ38.

**phonopy.** hTST asks for barrier and rate force calls. phonopy is the
finite-displacement half of a prefactor. There is no EAM evaluator and no
force-call protocol on `hTST/Al` or `hTST/Pt`.

**ParSplice.** `https://gitlab.com/exaalt/parsplice`, commit `4a48ef8` on
master (2019-05-16). It schedules external molecular dynamics. The build is
C++ and MPI. The accelerated-md problems have no force-call protocol and no
EAM evaluator.

**TAMMBER.** Commit `05afc05`. Same class of build and the same missing EAM
evaluator. The published metrics are a rate and an uncertainty, not a
force-call sample.
