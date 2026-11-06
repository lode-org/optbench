The `saddle-search/pt-island` link and repeat tests now follow the eOn structure comparison.
Two structures match when every atom sits closer than `distance_difference` to its own position in the other, as in `Matter::compare`.
A connected saddle repeats a stored one when the two saddles match as structures and their barriers lie within `energy_difference`, as in `AKMCState.find_repeat`.
These two eOn defaults now sit in a `[connection]` table of the problem.
Three catalogs read them from that table: rgsaddle, SoftSaddle, and ARTn.
Those catalogs used to test a Root Mean Square Deviation (RMSD) below 0.5 Å.
Over the 175 free atoms, a one-atom hop of 2.897 Å has an RMSD of 0.22.
That RMSD let a hop endpoint pass as the reactant.
Each catalog adds only connected saddles to its process table.
Its `fraction_connected` is 1 minus its "Not Connected" share, as `done.py` reports it.

The rgsaddle `saddle-search/pt-island` searches start from the eOn listed-atom displacement.
A search draws one listed atom and moves it and every free atom closer than `displace_radius` 3.3 Å by a normal deviate of 0.1 Å per coordinate.
The min-mode seed is a second draw on the same atoms, as the 2014 `random_mode` default.
The searches used to kick one listed atom by 0.1 Å along the seed.
Each search record stores its epicenter and the number of displaced atoms.

An rgsaddle `saddle-search/pt-island` climb stops after `SaddleSearch.max_iterations` min-mode steps, 1000 as the archived eOn `[Saddle Search] max_iterations`.
It used to stop at 1000 force calls, which comes to about 45 dimer steps.
The endpoint minimizations start `Optimizer.max_move` along the mode on either side, as eOn `processSearchMinimizationOffset`, and each gets `Optimizer.max_iterations` calls.
A search fails with `bad_minima` when an endpoint minimization fails to converge.
After the minus side fails, the search skips the plus side.
The per-search cap is the step bound times the most force calls one rgsaddle 0.2.2 step can spend, plus both endpoint budgets.
That cap comes to 48000 for the default dimer templates and 30000 for Lanczos.
Each search records its steps and its largest per-step call count, and a step over that ceiling stops the run.

The Morse surface gives the same bits on every x86-64-v2 host.
Its pair loop now builds with `-march=x86-64-v2 -O2 -ffp-contract=off -fno-fast-math` instead of `-O3 -march=native`.
With the old flags, gcc took an Advanced Vector Extensions (AVX) path on AVX2 hosts.
That path used Fused Multiply Add (FMA) and four-lane sums.
Without that path, the loop runs the operations of the NumPy reference in its order.
That order is `(x² + z²) + y²`, NumPy's pairwise sum for the energy, and the two `np.add.at` passes for the gradient.
The loop and the reference both call `optbench/svml_exp.c`, a scalar transcription of NumPy's AVX-512 exp.
Neither glibc nor NumPy can then pick a different exp for the loop by CPU.
On the 112 committed Morse starts, the loop differed from the reference in every gradient, by up to 1.7e-14.
It now equals the reference bit for bit.
The reference equals the NumPy code behind the committed rows wherever `np.exp` runs its AVX-512 loop.
Without `gcc`, the surfaces now raise an error, because the reference needs the same exp.

`bench_rgsaddle` has one airspeed velocity (asv) class per rgsaddle `saddle-search/pt-island` template, such as `RgsaddleSaddlePtDimerLbfgs`.
Each class tracks force calls, success, unique saddles, the connected fraction, total searches, and connected saddles.
These match the fields of the eOn Adaptive Kinetic Monte Carlo (AKMC) rows.
One parameter of a class is one catalog repetition, named by the seed of its displacements.
A cap failure is a repetition with fewer than 27 connected saddles.
The `run_template` function takes `repetition` to run one repetition of the catalog.

The Lennard-Jones kernels give the same bits on every x86-64-v2 host.
The harness `lj40` kernel, `surfaces.lennard_jones`, took `(σ/r)^6` from `np.power`, which NumPy dispatches by CPU.
With the AVX-512 loops of NumPy off, that kernel changed on 1549 of the 1602 committed LJ38 starts.
It now calls `optbench/svml_pow.c`, a scalar transcription of the Short Vector Math Library (SVML) power routine in NumPy.
Its output matches `np.power` wherever NumPy runs that routine.
Its squared distance follows the einsum order, `(x² + z²) + y²`.
The fastpot and GMIN `lj40.c` kernels now build with `-march=x86-64-v2 -O2 -ffp-contract=off -fno-fast-math`.
Their build cache key now includes the flags.
The fastpot cutoff shift, `lj_shift`, now rounds `(psi/cutoff)^6` once from its exact value instead of calling libm `pow`.
That gives the same bits for every cutoff the harness uses.

The Atomic Simulation Environment (ASE) has a `LennardJones` calculator that takes each pair's `(σ²/r²)^3` from `np.power`.
With the AVX-512 loops of NumPy off, its energies and forces changed on 1534 of the 1602 committed LJ38 starts.
The new `optbench.ase_lj.LennardJones` gives the ASE and Sella Lennard-Jones runs the same bits on every x86-64-v2 host.
It does the operations of ASE in the same order.
It takes the power from `optbench/svml_pow.c` and rounds the cutoff shift once from its exact value.
It equals ASE bit for bit wherever NumPy runs the SVML power routine.
Its bits stay the same with glibc and NumPy CPU dispatch off.
The `sella_lj38` and `sella_loweig` runners use it, as do the `minimization/lj38` ASE runner and the ASE Sella templates.
It raises for the smooth cutoff and for periodic cells, which no template uses.

Every eOn template now names the force kernel of the harness in a `kernel` key.
The ini still names the 2014 potential `lammps`, and the harness ran that name as two kernels.
The harness families run 42 of their 43 `lammps` templates as `lj40`.
The `bench_eon_develop` and `bench_eon_svn` modules ran `lammps` as `lj_cut`.
The two kernels differ in the last bits.
The 81 templates without a key now name the kernel their runner already used, so no run changes.
Each template whose client computes its own forces names `client`.
The campaign files now write a `kernel` line.
The develop and svn modules also store the kernel in their per-start records.

The C ext_pot bridge now records two values when it stops a start at the harness cap.
The killed eonclient can no longer write either of them.
The first, `client_calls_at_cap`, is the number of force calls that client finished.
That number equals its own force-call counter at the kill.
The second, `client_stop_at_cap`, is the refusal the bridge received in place of the next call.
The bridge keeps the tally in `optbench-extpot-calls` and writes both values to `optbench-extpot-cap` just before the kill.
The harness force count stays the same, as do the failure class, the success bit, and every field that the board and `score_v2` read.

The ASE and Sella starts now run on one OpenBLAS thread under asv as well.
The module `optbench.ase_sella` sets `OPENBLAS_NUM_THREADS=1` at import.
OpenBLAS reads that variable only when it loads, and the asv runner imports NumPy before any benchmark.
NumPy's OpenBLAS then ran one thread per CPU.
With asv launched without the variable, `sella_lj38` changed on 6 of its 200 committed starts.
Each start now sets every OpenBLAS in the process to one thread through the library's own `set_num_threads`.
