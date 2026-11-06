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
