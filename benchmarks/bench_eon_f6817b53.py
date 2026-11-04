"""eOn develop f6817b53 harness benchmarks.

One class is one template under templates/eon whose id starts with
``eon/develop-f6817b53-`` and sets ``kind``. The binary is the develop
build of commit f6817b537ec1d1563420023fb45fbd0e5b75e7fd named by
``EON_DEVELOP_F6817B53``. ``optbench.eon_harness`` runs the same
``run_case`` that wrote the campaign rows. The default grid is one start
unless ``OPTBENCH_ASV_FULL=1`` at import.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("f6817b53", globals())
