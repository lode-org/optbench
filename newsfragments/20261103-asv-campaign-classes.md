asv now measures the campaign cells of COMPASS, the transition state library (TSASE), pARTn, LAMMPS, DL-FIND, pysisyphus and OPTIM one start at a time.
Each class calls the engine adapter that `optbench run` calls, with the budget and start set of its problem file in Tom's Obvious Minimal Language (TOML).
`bench_airss` gives each template of ab initio random structure searching (AIRSS) its own class, LJ75 included.
The eOn cells that the client counted, on Subversion r2549, 3.5.0 and main, have classes, and so do the 500350d9 campaign templates and the aladatom adaptive kinetic Monte Carlo (AKMC) repetitions.
With `OPTBENCH_ASV_FULL=1` every class reaches the full start set of its problem, including the global minimisation (GMIN) basin-hopping and eOn classes that stopped at start 0.
The fourteen GMIN global-optimisation rows rerun from the archived data files have classes, which run the serial GMIN that `GMIN_MPI=OFF codes/gmin/build-bh.sh` builds.
Every track reads the record that its own search wrote, so GMIN basin-hopping tracks no longer report NaN or crash in a fresh asv process.
optbench fetches the latest Wales snapshot for every GMIN and OPTIM build.
The URL serves a new snapshot every day.
Each run writes the sha256 and Subversion revision of its tarball into its records.
The engine build scripts use `CC=gcc`, `CXX=g++` and `FC=gfortran`.
