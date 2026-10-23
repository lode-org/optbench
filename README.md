# OptBench

OptBench compares geometry optimizers, saddle searches, and global optimization on a fixed set of problems. Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales, and Henkelman set the suite out in J. Chem. Theory Comput. 2014, 10, 5476-5482 (doi:10.1021/ct5008718). The 2014 tables are at <https://theory.cm.utexas.edu/benchmarks/>. This repository is a replay of that suite. The public board is <https://lode-org.github.io/optbench/>.

A problem file under `problems/` is TOML. It states the potential, the stopping rule, and the keys a result must carry. The JSON schemas are under `contract/`, and `contract/README.md` lists them. A result is one `benchmark.dat`: the first whitespace on a line splits the key from the value. The geometries are already in this clone, under `inputs/`. A campaign reads them from `inputs/manifest.jsonl`. Nothing else has to be downloaded to run a problem that ships here.

## Install the harness

The harness needs [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer. From a clone:

```bash
git clone https://github.com/lode-org/optbench
cd optbench
uv run pytest
uv run optbench check
uv run optbench templates
```

`uv run` reads `pyproject.toml`, installs the command-line library, and runs that command. `pytest` is the test runner. `check` exits 0 when every shared contract covers its archived rows and `oracle/archive_gaps.txt` matches the departures. `templates` prints every template id.

`uv run optbench board` reads `results/campaign` and `results/asv` and writes `results/board` plus the v2 table on the Board page. It does not start a search. A band method is ranked on the harness force-call total across every image, and the per-image mean is a second column from that same total. A ranked row is one code, one version, and one stock method. Older full versions stay in the same tables. A run that is both a campaign file and an asv track is one row, with the source kept as a column. A row that used fewer starts than the problem's full set stays off the ranked table. An asv result that covers one start stays out of the ranked mean.

The harness does not contain eOn, pele, or Sella. Those are installed separately, and the commands below say which release produced the files in `results/campaign/`.

## Lennard-Jones 38 L-BFGS mean

`results/campaign/eon-client-lbfgs-m100.dat` is:

```
force_calls 181
force_calls_min 90
force_calls_max 403
nfailed 0
code eOn
code_version 3.5.0
algorithm LBFGS
template eon-client/lbfgs-m100
```

That file is eOn 3.5.0, commit `4d8b34baf1ba8d2d980df29838bac1f969d1662c`, on all 1000 starts in `problems/minimization/lj38.toml`. The stop is the Euclidean norm of the force at 0.01, or 10000 force calls. L-BFGS memory is 100. The client default is 20, and that default is a different score (about 201).

Build `eonclient` from that commit of <https://github.com/TheochemUI/eon>. In an activated environment from that tree:

```bash
meson setup builddir --prefix="$CONDA_PREFIX" --libdir=lib --buildtype release
meson install -C builddir
command -v eonclient
```

A run prints `Eon version 3.5.0`. `pip install pyeonclient` is a different interface. It does not set L-BFGS memory, and the campaign does not call it.

From this repository, with `eonclient` on `PATH`:

```bash
uv run optbench run minimization/lj38 eon-client/lbfgs-m100 \
  --workers 4 \
  --output benchmark.dat
```

The command uses `eonclient` from `PATH`. `--client` or `EONCLIENT` overrides that. One line does both the binary and the Python command, which is the pixi-and-uv form:

```bash
pixi exec --spec eon --spec uv -- uv run optbench run \
  minimization/lj38 eon-client/lbfgs-m100 \
  --workers 4 \
  --output benchmark.dat
```

The recorded 181 is commit `4d8b34baf`. Read the version line the client prints before treating another build as that score.

`templates/eon/` is main, commit `1d71c944501832bfafb955ffe13b9aae50c8efd4`, the merge of pull request 776. The version string is 3.6.0. `eon/lj38-lbfgs` is L-BFGS at the struct defaults, including memory 20 and `lbfgs_accept = energy`. `eon/lj38-lbfgs-m100` changes only the memory to 100. Both set the problem stop. The same pair for Morse bulk is `eon/morse-lbfgs` and `eon/morse-lbfgs-m100`. L-BFGS keys sit in `[LBFGS]`. A run writes `benchmark.dat` and `benchmark.starts`, one JSON object per geometry, failures included. The same commit has a default and a memory-100 template for the Pt-heptamer NEB, the LJ38 and Pt-island double-ended bands, and the LJ38 and Pt-island dimer saddle searches. `uv run optbench eon-job neb-pt-lbfgs` runs one of those. The NEB and saddle means follow the archived report: a counter under the cap stays in the mean, and `nfailed` counts a termination reason other than 0. The heptamer mean divides `force_calls_neb` by 8. The 3.5.0 and SVN templates are unchanged. `--workers` is the number of client processes. The default is 1. The problem file names `website/minimization/lj38/lj38-clusters.tgz`, and the manifest stores that archive at `inputs/sha256/86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1/lj38-clusters.tgz`. The command unpacks it. `--work` chooses the scratch directory. The default is next to `--output`.

`benchmark.dat` is the result. `force_calls` is the mean over starts that finished under the cap. Compare it to the file quoted above.

A directory passed as `--geometries` runs only the `.con` files in that directory. That checks the client. It is not the published mean, because the published mean uses all 1000 starts.

## Other scores you can rerun

Each command writes one `benchmark.dat`. The published copy is under `results/campaign/`.

Morse bulk is 100 starts, force norm 0.001 eV/Ang, cap 10000. The same `eonclient` and `eon-client/morse-lbfgs` produce mean 52:

```bash
uv run optbench run minimization/morse-bulk eon-client/morse-lbfgs \
  --workers 4 \
  --output morse-lbfgs.dat
```

The Morse geometries are `inputs/sha256/db1ed9c0a17de8c1cf7dbf55f8b1d2baed263ad8686adacf5e892758d3497d8b/morse-bulk.tgz`. The other Morse templates are `eon-client/morse-cg`, `eon-client/morse-fire`, and `eon-client/morse-sd`.

pele 0.1.2 is <https://github.com/pele-python/pele>. The driver imports `pele.optimize._lbfgs_py.LBFGS` and `pele.potentials.LJ`. Install that release, unpack the same Lennard-Jones archive, and run every archived memory and step cap:

The script declares its own dependencies. `uv run` installs them for that file:

```bash
tar -xzf inputs/sha256/86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1/lj38-clusters.tgz -C "$WORKDIR"
uv run optbench/pele_lj38.py \
  --geometries "$WORKDIR/lj38-clusters" \
  --output-dir "$WORKDIR/pele" \
  --workers 4
```

Memory 100 at step 0.2 averages 178 calls. The 2014 file is 179. The other six files are `results/campaign/pele-0.1.2-lbfgs-*.dat`. Those files counted `result.nfev` on pele's Lennard-Jones potential.

`templates/pele/` is pele 0.1.2 at commit `b7081b932a060be11342fbe8287ac35da8ada879`, built from source. LBFGS, MYLBFGS, FIRE, and steepest descent call the harness potential. The stop is the Euclidean norm from the problem file. One row is:

```bash
uv run optbench run minimization/lj38 pele/lj38-lbfgs \
  --workers 4 \
  --output results/campaign/pele-lj38-lbfgs.dat
```

Sella 2.6.0 needs `sella==2.6.0` and ASE. The stop is the Euclidean norm of the force at 0.001, or 1000 steps, on 200 geometries, with cutoff 40 and the archived trust-region numbers:

```bash
uv run optbench/sella_lj38.py \
  --coords inputs/sha256/bda818f7fd3c2ee47ad4d4fd74524881814544c1a805054656f612724d9d5ce6/coords-con.tar.gz \
  --output sella.dat \
  --workers 4 \
  --version 2.6.0
```

The recorded file is `results/campaign/sella-2.6.0-ts-lj38.dat`: mean 93, two starts at the step cap. Sella 0.1.1 was mean 69.55 with none failed. Those 0.1.1 rows stay in the same tables as the later rows.

ASE 3.29.0 and Sella 2.6.0 templates live under `templates/ase/` and `templates/sella/`. A default row is the library constructor. An adjusted row is the archived script when one exists, and it stays off the ranked table. Force calls are calculator evaluations counted by the harness, for both codes. The ranked set is:

```bash
uv run --group campaign optbench campaign-ase --workers 4 --output-dir results/campaign
```

One template is the same command as the eOn runs, with the template id in place of `eon-client/lbfgs-m100`:

```bash
uv run --group campaign optbench run minimization/lj38 ase/lj38-lbfgs-default \
  --workers 4 \
  --output results/campaign/ase-3.29.0-lj38-lbfgs-default.dat
```

The per-start file is the `.starts.jsonl` next to that `benchmark.dat`. A count at the cap, an exception, or a final force above the problem stop is a failure and is kept in that file. The dimer templates are not in `campaign-ase`. They are single-ended searches, not the Pt-island saddle catalog.

Pt-heptamer NEB is 13 bands, Morse, eight images. The band method has to be the one named in the template. `eon-lbfgs` averages 40:

```bash
uv run optbench neb \
  --entry archive/entries/neb/Pt-heptamer/eon-lbfgs \
  --minima inputs/sha256/135c83a17e5a1769182e9a6cdb3ad98657a62a6cd20ef8ae0440e876e24c5e1b/minima.tgz \
  --work "$WORKDIR/heptamer-lbfgs" \
  --output heptamer-lbfgs.dat \
  --workers 4 \
  --version 3.5.0
```

The same command with `eon-cg`, `eon-fire`, or `eon-qm` is the other three rows. CG is 142, FIRE is 85, QuickMin is 165.

The LJ38 climbing-image band is 50 reactant-product pairs and `potential = lammps` with `in.lammps` cutoff 40. eOn 3.5.0 loads `liblammps.so` at run time. Put a LAMMPS build on `LD_LIBRARY_PATH` before this command. Without that library the client cannot start these entries. The Morse heptamer runs do not need it.

```bash
uv run optbench pairs \
  --entry archive/entries/double-ended-saddle-search/lj38/eon-ci-neb \
  --coords inputs/sha256/d1c74db6a00956a4e30487fb80e6331b9515e5f93bba1c6f823e38c3b2986dfc/coords-con.tgz \
  --work "$WORKDIR/ci-neb" \
  --output ci-neb.dat \
  --workers 4 \
  --count 50 \
  --version 3.5.0
```

`eon-ci-swdneb` is the doubly nudged row. A start at or above 2500 force calls is left out of the mean. The climbing-image mean is 809 with 2 failures. The doubly nudged mean is 708 with 5 failures.

The Henkelman client is revision 2549 of <https://theory.cm.utexas.edu/svn/eon>. A checkout of older revisions fails where historical paths are gone. Export revision 2549. That tree still names a QSC potential whose sources are not in the revision, so the QSC references have to come out of the build or the link fails. The binary should report `r2549`. The minimization templates under `templates/eon-svn/` use `lj_cluster`. The harness rows for the other problems name `ljcluster`, `lj_cut`, `lammps`, or `morse_pt` and run through `ext_pot`. The 3.5.0 templates use `ljcluster`.

A problem has one potential. Rows on another potential stay off the ranked table. `problems/tsopt/LJ38.toml` lines 8–13 and `problems/lowest-eigenvector/lj38.toml` lines 8–13 are Lennard-Jones, epsilon 1, sigma 1, cutoff 40, with periodic boundaries off. `problems/double-ended-saddle-search/lj38.toml` line 20 and `problems/double-ended-saddle-search/lj38_2ts_separation.toml` line 18 set `potential = "lj"`, which is that same unshifted cutoff. The lane rows on those problems use that kernel and stay ranked. The published `ljcluster` rows (shifted, cutoff 15) on `tsopt/LJ38`, `lowest-eigenvector/lj38`, and `lj38_2ts_separation` are unranked. The published `de-lj38` rows use the archived `lj/cut` cutoff 40 and stay ranked; their max iteration count is 8000, and the lane climbing-image rows use 1000. `problems/tsopt/pt-island.toml` lines 8–15 and `problems/saddle-search/pt-island.toml` lines 8–15 are `morse/smooth/linear`, cutoff 9.5, periodic in x and y. The lane rows match that and stay ranked. The published Pt tsopt rows use r2549 `Morse::force` and are unranked. `problems/double-ended-saddle-search/pt-island.toml` line 17 and `problems/neb/Pt-heptamer.toml` line 17 set `potential = "morse_pt"`. That name is the shifted Morse in `optbench/pes.py` lines 3–6 and 45–67: cutoff 9.5, pair energy zero at the cutoff. The lane double-ended and NEB templates were rerun on r2549 `Morse::force` with the Fortran binary. Against the smooth-linear starts, 35, 43, 48, 49, and 52 of the 59 Pt double-ended starts changed for climbing-image NEB, doubly nudged climbing image, dimer, Lanczos, and improved Lanczos. One, one, three, and one of the 13 NEB pairs changed for CG, FIRE, L-BFGS, and QuickMin. Both hTST Pt passes changed. They were not pooled with the published ids. The published climbing-image rows set spring 5 and `climbing_image_converged_only`, and the lane rows do not, so the starts are different methods. The 3.5.0 rows that still use `morse/smooth/linear` on those two problems are unranked variants. `hTST/Pt` has no potential block; the archive ini sets `morse_pt`, and those lane rows were rerun on `Morse::force`. They stay off the ranked table because success is one qualifying saddle. `hTST/Al` and the accelerated-md rows use the r2549 Aluminum EAM and stay off the ranked table. `globalopt/lj38.toml` has no potential block. The 2014 `in.lammps` is `lj/cut/opt` cutoff 40, which is the lane kernel. Basin hopping, seed 314159, has all 100 starts. The per-start wall budget is 21600 s. The force-call cap is still 2000000. Ninety-eight r2549 starts reached -173.928427. Starts 69 and 72 stopped at that cap. The success-only mean is 529113 and the PAR-2 mean is 598530. The 3.5.0 basin-hopping row has 100 starts, all below the stop. Its success-only mean and PAR-2 mean are 503146. This r2549 binary links libgfortran and the Fortran potentials, including Aluminum, Lenosky, SW, and EDIP. It is not a NO_FORTRAN build. LAMMPS is not linked. The 3.5.0 lane rows use commit 4d8b34baf1ba8d2d980df29838bac1f969d1662c. The conda binary that prints 3.5.0 (7ec1840) was not used. Hyperdynamics bond boost uses bb_dvmax 0. The bounded AKMC rows stop after 4 passes and do not reach 27 good saddles, so the PAR-2 of that incomplete catalogue is the cap penalty and the harness calls stay in the start file. The minimization rows stay as recorded. mlopt is neural-network training and was left out. tip4p has no harness kernel. lj100-binary has no binary-LJ kernel. lj75, lj98, and lj150 have no stated minimum or force-call success rule. standard_aladatom has no geometry, cap, or success rule beyond rate metrics. Minima hopping start 0 used 2000000 harness calls in 3600 s and did not reach -173.927427. The lowest monitoring energy was -173.25238. That run is not scored.

## What a template may set

The problem file owns the stop and the geometries. The template owns only the client. eOn 3.5.0 and SVN r2549 do not read the same keys. A key in the wrong section is ignored, not applied.

eOn 3.5.0, `templates/eon-client/`:

- QuickMin reads `time_step` from `[QuickMin]`. FIRE reads `time_step` and `time_step_max` from `[FIRE]`. The same keys under `[Optimizer]` do nothing.
- Steepest descent reads `sd_alpha` and `sd_twopoint` from `[SD]`.
- L-BFGS reads `lbfgs_memory` from `[LBFGS]`, or from `[Optimizer]` when `[LBFGS]` is absent.
- A nudged elastic band reads `opt_method` from `[Nudged Elastic Band]`. With that key absent the band is L-BFGS, including when `[Optimizer]` names CG, FIRE, or QuickMin. `converged_force` on `[Optimizer]` is the band tolerance when the band section does not set one.

SVN r2549, `templates/eon-svn/`:

- `sd_alpha`, `sd_twopoint`, and the QuickMin `time_step` are read from `[Optimizer]`.

`uv run optbench neb` and `uv run optbench pairs` copy an archived `[Optimizer]` timestep into `[QuickMin]` or `[FIRE]`, and copy `opt_method` into `[Nudged Elastic Band]`, then run that entry's `report.sh`. The values stay the archived values.

Templates can also declare auxiliary run files with `[[inputs]]`.
Each entry names a `source` file and an optional `target` path in the run
directory. If `target` is omitted, the file keeps its basename. `source`
is resolved relative to the repository root unless it is absolute. This is
how eOn global optimization templates stage `in.lammps` and any future
potential-side data files without hardcoding paths in the launcher.

The LJ38 minimization templates name `ljcluster` or `lj_cluster`. They do not pass the problem cutoff of 40. Every pair in that cluster is inside the cutoff, and the all-pairs potential matches the archived means. A geometry whose pairs reach the cutoff has to use `lammps` and the archived `in.lammps`, as the climbing-image command does.

## The two boards

`oracle/entries.jsonl` holds every `benchmark.dat` from branch `main` of <https://gitlab.com/optbench/optbench> at commit `0f26b796`. The archived LJ38 minimization mean counts every run under 10000 force calls as a success. A new run is a success only when the force norm is at or below the problem value and the call count is under the cap. Both rules are in `problems/minimization/lj38.toml`. The reference entry is `archive/entries/minimization/lj38/eon-cg`.

The front page is a result whose status is `sota` and whose commit is the present source in `codes/registry.toml`. The archive page is the 2014 scores plus later passes that do not meet that rule. A local `benchmark.dat` is a score whether or not it has been added to the front page. Adding it there means a `results/*.json` record with `optbench.result/v1`, the problem geometries, and the pinned commit.

`uv run optbench import-archive ARCHIVE` rebuilds the oracle and the geometry store from a bare clone of the public GitLab project. It does not replace a problem file that already exists.

## v2 score

The 2014 mean leaves out failures. The harness score keeps them, at twice the cap. It is specified in `docs/design/decision.md` and `docs/v2.md`, and the in-process wrapper is `docs/harness.md`.

A force call is one entry to `HarnessCounter`. ASE, pele, and Sella are charged through that counter. A compiled client reaches the same counter: eonclient through the `ext_pot` bridge in `optbench.extpot`, and an rgpot client through `RgpotCounter.calculate`. `uv run optbench run` still writes `benchmark.dat` from the client's own counter, and the per-start file next to it is marked `counter` `client`. That file is not a v2 score. `uv run optbench validate-counter` is the subset check that the server count matches eonclient. A subset is not a published mean.

A v2 success is under the cap, at the force stop, and, when a reference is supplied, at Hessian index 0 for a minimum or index 1 for a saddle, with the energy and Kabsch RMSD inside the tolerances. A failure costs `2` times the cap. The table shows that PAR-2 mean, and the median and interquartile range with a bootstrap of 2000 resamples seeded by `20261015`. The success-rate interval is computed and is not a column.

Pt-island cost is force calls in a repetition divided by the searches in that repetition. `eon-client/lbfgs-m20` is `tuning` `default`. `eon-client/lbfgs-m100` is `tuning` `adjusted`.

`uv run optbench quadratic` writes `results/quadratic/quadratic.json`. That file is a quadratic check of the counter and the score. It is not a Lennard-Jones campaign. Those campaigns stay on rg.terra.

## Statistics

`benchmark.dat` keeps the rounded mean, because that is the 2014 record. A v2 measurement is an asv track. Force calls, the success bit, and the four failure classes are `track_*` values. Wall time is `time_*`. The parameters are the seed and the start. `results/asv` is asv's result tree, one file per commit and environment, with `machine.json` beside it. `uv.lock` pins `asv==0.6.6` and `asv-runner==0.3.1`. `environments/eon-3.5.0.yml` pins the conda-forge eOn binary. `docs/asv.md` is how another unit adds a code. `docs/asv-gaps.md` is the upstream list.

The environment type is `uv-pip`, from `optbench.asv_uv`. It creates the environment with `uv venv` and installs with `uv pip`. The committed quadratic sample is that environment. eonclient is not part of it. The seven Lennard-Jones 38 classes in `benchmarks/bench_lj38.py` skip unless `OPTBENCH_ASV_EON=1`, and the default grid is one start unless `OPTBENCH_ASV_FULL=1`.

```bash
uv run asv machine --yes
uv run asv run COMMIT^! -b quadratic
uv run optbench statistics --output results/validation/statistics.json
uv run asv publish
```

`optbench statistics` reads the saved tracks. The saved file comes from `asv run` without `--quick`. CI runs the quadratic suite with `HaoZeke/asv-perch@v1` (`.github/workflows/asv.yml`).

## Build the site

Org-mode under `docs/orgmode` is the prose. `uv run python -m optbench.site` writes the rank tables. Emacs exports the org to reStructuredText, and Sphinx builds the HTML. Emacs is not a Python package, so the site build stays on pixi:

```bash
pixi run -e docs html
```

Open `docs/build/index.html`.

The hosted test run is `uv run pytest` and `uv run optbench check`. A published score uses every start in the problem. SoftSaddle is a MATLAB batch in `.github/workflows/softsaddle-check.yml`: it builds `libmorse`, `libdavidson`, and `liblanczos`, then runs two searches with a 600 second cap. That batch leaves the April 2019 score of 423491.1 force calls in `softsaddle/entry.toml` as the archived number.
