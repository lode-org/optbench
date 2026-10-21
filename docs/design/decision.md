# Decision

Three designs are written next to this note.

- `par2.md` charges a failure `2 * cap` and ranks by the mean of the penalised costs.
- `performance-profile.md` leaves failures censored and compares distribution functions.
- `conditional-median.md` ranks by the Wilson lower bound and breaks ties with the median of the successes.

The implemented score is alternative A.

The site needs one cost that can sit in a table the way the 2014 force-call mean does, and that cost has to move when a start fails. The performance profile does not give that cell, and its area changes when another code is added to the comparison. The conditional median leaves failures out of the cost, which is the rule this revision replaces. PAR-2 depends only on the starts, the cap, and the success rule. Factor 2 is large enough that the published illustrations move (the rounded 2014 mean 3514 with 25 failures becomes an illustrative 3926.15) and small enough that the finished starts still affect the mean.

These pieces are shared with the designs that were not selected, and they are part of the implementation:

- Wilson bounds for the success rate, `z = 1.959963984540054`
- median and interquartile range of the penalised costs, with a percentile bootstrap of 2000 resamples, seed `20261015`
- Hessian index 1 for a saddle and index 0 for a minimum, plus an energy tolerance and a Kabsch RMSD tolerance
- Pt-island cost in force calls per search
- `tuning` of `default` or `tuned`
- provenance of code, version, commit, parameters, and machine

The harness counts a force call by wrapping the potential. A compiled client uses the same `HarnessCounter`, through the socket or through `RgpotCounter.calculate`. A code's own counter is not the v2 cost. The 2014 half-even mean stays in `benchmark.dat` and on the archive tables. `docs/v2.md` is the spec for that counter, the seed sample, the failure classes, and the noise rule. Those per-start counts are asv tracks under `results/asv`, keyed to the commit asv ran. `docs/asv.md` is how to add one. `docs/asv-gaps.md` is the upstream work that file does not paper over.

Illustrations in `par2.md` that rebuild a PAR-2 mean from a rounded 2014 mean are not v2 scores. A v2 score is computed from the per-start harness record.
