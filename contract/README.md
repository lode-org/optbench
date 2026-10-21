# Schemas

The JSON schemas live in this directory. Problem statements are TOML files under `problems/`. A sentence in the repository README that names only `problems/` is not pointing at these schemas.

| File | Document |
| --- | --- |
| `problem.schema.json` | `optbench.problem/v1` |
| `result.schema.json` | `optbench.result/v1`, the 2014 board |
| `result.v2.schema.json` | `optbench.result/v2` |
| `start.v2.schema.json` | `optbench.start/v2`, one start |
| `score.v2.schema.json` | `optbench.score/v2`, the PAR-2 summary |
| `problems/minimization/lj38.json` | the v1 Lennard-Jones 38 contract |

`optbench.result/v1` stays the schema of the files in `results/*.json`. A v2 result is a different document. The 2014 archive rows are not rewritten to match it.

The scoring rule is `docs/design/decision.md`. The counter, the seed sample, the failure classes, and the noise rule are `docs/v2.md`. The in-process wrapper and the smoke command are `docs/harness.md`. `failure` on a start is optional in the schema so a record written before that field still validates.
