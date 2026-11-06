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
