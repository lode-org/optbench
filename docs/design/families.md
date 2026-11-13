# Method families

A ranked row is one code, one version, and one stock method or preset.
The version is the recorded version string. A develop pin keeps the short
commit in that string, so the label is `eOn (develop 500350d9) · Lanczos improved`.
A release is `TSASE (1.0) · CI-NEB`. Two methods of one code are two rows.
Two versions of one method are two rows. An archived score uses the same
key and sits in the same tables.

The third key is the method when the row names one, and the preset when
the row names a preset. A `method` value that is a parameter record is
not that key. The algorithm name is used instead. TSASE records `CI-NEB` and `NEB` as two methods.
A preset-keyed code uses the same rule: anneal 0.11.1 would rank
`communicating` and `recommended` separately, and GMIN would rank `bh`
and `bhpt` separately, when a row records those tokens. This tree's GMIN
archive rows are split by the algorithm string they do record
(basin-hopping, linear temperature basin-paving, and the symmetrisation
variants). Those strings are the method key.

## Families

The family comes from the method string. The problem suite is only the
fallback when the string does not name a family.

| Id | Title | What sorts here |
| --- | --- | --- |
| `band` | Band / NEB | NEB, CI-NEB, energy-weighted CI-NEB, doubly nudged CI-NEB, OCI-NEB, IDPP and S-IDPP CI-NEB |
| `min-mode` | Minimum mode | dimer, Lanczos, Lanczos improved, rotation-skip dimer, lowest-eigenvector searches |
| `single-ended` | Single-ended saddle | P-RFO, RS-I-RFO, Sella saddle search, pARTn, SoftSaddle |
| `connect` | Connect / double-ended | DNEB followed by hybrid eigenvector-following, and other connect drivers that are not themselves a band |
| `global` | Basin hopping / global | basin-hopping, basin-paving, minima hopping, AIRSS |
| `minimize` | Minimizers | L-BFGS, FIRE, CG, steepest descent, Newton, dogleg, and the other local minimizers |

Sella used as a minimizer stays in `minimize`. A band that climbs an
image stays in `band`, including doubly nudged CI-NEB and OCI-NEB.
Dimer and Lanczos stay in `min-mode` when they are started from a pair.

## LJ38 two-step eOn rows

eOn develop `500350d9` has eight stock methods on `lj38_2ts_separation`:

| Method | Family | Template |
| --- | --- | --- |
| Lanczos improved | Minimum mode | `eon-client/de-2ts-500350d9-lanczos-improved` |
| Lanczos | Minimum mode | `eon-client/de-2ts-500350d9-lanczos` |
| dimer | Minimum mode | `eon-client/de-2ts-500350d9-dimer` |
| energy-weighted CI-NEB | Band / NEB | `eon/develop-500350d9-de-2ts-ew` |
| CI-NEB lj/cut | Band / NEB | `eon-client/de-2ts-500350d9-ci-neb` |
| CI-NEB ljcluster | Band / NEB | `eon/develop-500350d9-de-2ts` |
| doubly nudged CI-NEB | Band / NEB | `eon-client/de-2ts-500350d9-ci-swdneb` |
| OCI-NEB | Band / NEB | `eon/develop-500350d9-de-2ts-oci` |

`CI-NEB lj/cut` is the unshifted cutoff-40 kernel. `CI-NEB ljcluster` is
the shifted cutoff-15 kernel. They are different methods.

eOn 3.6.0 uses those same method names. A template that only recorded the
optimizer as `LBFGS` left OCI-NEB, energy-weighted CI-NEB, the accept-energy
line search, and plain CI-NEB as one row. The `method` field is the name
the board ranks. An asv file from before that field still takes the name
from the working template.

## Hessian columns

`Hessians` is the mean analytic Hessian count on the recorded starts.
No problem file defines a finite-difference multiplier. `FD calls` is
that count times the coordinate count: one force call per coordinate,
the forward-difference equivalent, so one full Hessian is 3N calls.
A method that calls no analytic Hessian shows 0 and 0. The coordinate
count is the problem's Cartesian size (`optbench.families.COORDINATES`,
or `atoms` / `mobile_atoms` on the problem). When the size is unknown
and the count is positive, `FD calls` is left blank.
