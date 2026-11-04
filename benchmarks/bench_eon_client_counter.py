"""eOn 3.5.0 templates that the campaign scored with the client's count.

One class is one ``eon-client/`` minimization template without ``kind``
that ``benchmarks.bench_lj38`` does not measure: the Morse bulk set. The
client is ``EONCLIENT_350`` or ``~/opt/eon-350/bin/eonclient`` and must
print ``4d8b34ba``. ``optbench.asv_eon`` runs the ``optbench run`` job of
the start. The default grid is one start unless ``OPTBENCH_ASV_FULL=1``
at import.
"""

from __future__ import annotations

from benchmarks.bench_lj38 import TEMPLATES
from optbench.asv_eon import install_benchmarks

install_benchmarks("eon-client/", globals(), frozenset(template for _name, template, _limit in TEMPLATES))
