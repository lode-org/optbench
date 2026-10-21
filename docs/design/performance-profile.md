# Alternative B. Censored ECDF and a performance profile

Status: not selected. Per-start costs are kept so this view can be drawn later. It is not the ranking cell.

## Score

Do not impute a cost for a failure. Treat the force-call count as right-censored at the cap. On one problem the summary is the empirical distribution function of the harness counts: the fraction of starts that finished at or below each cost. A failure contributes to the denominator and never crosses the finish line.

Across problems, the Dolan-Moré performance profile of a code is the fraction of problems on which its cost is at most `τ` times the best code on that problem. A scalar used with that curve is the area under the profile, or the value of the profile at a fixed `τ` such as 2.

Success is the same predicate as in alternative A: force norm, cap, Hessian index, energy, and Kabsch RMSD. The Pt-island x-axis is force calls per search, not the campaign total. Tuning and provenance are required.

The Wilson bounds on the success rate are still reported. The median and interquartile range, with a bootstrap interval, are computed on the finished starts only, and the plot shows the censored mass at the cap.

## Why it was not selected

The public board is a table with one cost per code on one problem, in the shape of the 2014 tables. A distribution function does not fill that cell. The area under a performance profile also depends on the other codes in the table: adding a faster code changes the best reference and moves every other area. A PAR-2 mean depends only on that code, the cap, and the success rule.

The profile remains the right picture when someone compares many codes across the whole suite. The per-start file from the harness is the input it needs. The ranking number published on the problem page is the PAR-2 mean from alternative A.
