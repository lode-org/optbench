## Template-declared eOn auxiliary inputs

- `optbench.campaign` now stages template-declared auxiliary files before each
  `eonclient` launch.
- Templates declare these files with `[[inputs]]` entries (`source`, optional
  `target`) instead of relying on hardcoded launcher behavior.
- `eon-client/*-lj38` and `eon-svn/*-lj38` global-optimization templates now
  declare the archived `in.lammps` input explicitly.
