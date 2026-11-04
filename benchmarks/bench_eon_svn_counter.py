"""Eon svn r2549 templates that the campaign scored with the client's count.

One class is one ``eon-svn/`` minimization template without ``kind``:
the Lennard-Jones 38 sweep and the Morse bulk set. The client is
``EON_SVN_CLIENT`` or ``~/opt/eon-svn/bin/eonclient`` and must print
``r2549``. ``optbench.asv_eon`` runs the ``optbench run`` job of the start.
The default grid is one start unless ``OPTBENCH_ASV_FULL=1`` at import.
"""

from __future__ import annotations

from optbench.asv_eon import install_benchmarks

install_benchmarks("eon-svn/", globals())
