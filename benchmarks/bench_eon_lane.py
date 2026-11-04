"""eOn SVN r2549 lane harness benchmarks.

One class is one template under templates/eon-svn that sets ``kind``.
These rows use the lane kernels: unshifted lj/cut cutoff 40, and
morse/smooth/linear with periodic boundaries in x and y. They are not
the ljcluster and r2549 Morse::force rows in benchmarks.bench_eon_svn.
Parameters are the seed and the start. ``measure`` returns harness
``force_calls``, ``success``, and one ``failure``. The default grid is
one start unless ``OPTBENCH_ASV_FULL=1`` at import, which selects every
seed and start of the template. A warm cache makes
``time_search`` a cache read. The search wall time is the ``seconds``
field stored with the campaign start.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("svn", globals())
