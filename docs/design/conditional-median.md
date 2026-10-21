# Alternative C. Wilson lower bound, then the median of successes

Status: not selected.

## Score

Rank codes by the lower end of the Wilson 95% interval for the success rate. Break ties with the median force-call count among successes, and publish a percentile bootstrap interval for that median and for the interquartile range.

Failures stay out of the cost. They appear only in the success rate. There is no PAR penalty and no imputed value at the cap. Success still requires the Hessian index, the energy match, and the Kabsch RMSD. Pt-island costs are per search. Tuning and provenance are required.

On the published Lennard-Jones 38 rows this puts eon-3.5.0-lbfgs-m100 first on both pieces: 1000/1000 successes, Wilson lower end 0.996173, 2014 mean 181. It separates eon-3.5.0-qm (975/1000, Wilson 0.963355 to 0.983010, 2014 mean 3514) from eon-3.5.0-sd (904/1000, Wilson 0.884165 to 0.920743, 2014 mean 4901) by the success bound before either median is consulted.

## Why it was not selected

The median of the successes is the half-even mean's robust cousin. Both leave failures out of the cost. A sort that looks at the median alone still prefers a fast method that abandons hard starts. Putting the Wilson lower bound first repairs the ranking only when success rates differ by enough to move that bound. Two methods that both finish 1000 of 1000 starts are then compared by a median that never saw a failure, which is fine, but a method that fails 96 of 1000 starts can still show a flattering median in the cost column. The board has already been read that way.

PAR-2 puts the failure into the same number as the cost. The median of the penalised sample is still published, so a heavy tail stays visible. When more than half the starts fail, that median equals `2 * cap` and the method cannot look typical.
