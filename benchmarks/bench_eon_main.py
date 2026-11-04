"""eOn main templates that the campaign scored with the client's count.

One class is one ``eon/`` template on the ``eon-main`` commit
1d71c944501832bfafb955ffe13b9aae50c8efd4: each minimization template
without ``kind``, each ``optbench eon-job`` job that is one client run
per start, and each AKMC catalogue job, whose start is one repetition.
The client is ``EON_MAIN_CLIENT`` or ``~/opt/eon-main/bin/eonclient`` and
must print ``1d71c944``. ``optbench.asv_eon`` runs the campaign job of the
start. The default grid is one start unless ``OPTBENCH_ASV_FULL=1`` at
import.
"""

from __future__ import annotations

from optbench.asv_eon import install_benchmarks

install_benchmarks("eon/", globals())
