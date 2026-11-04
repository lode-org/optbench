"""pARTn searches, one class per cell.

``partn/artn`` is one class per band problem. ``partn/saddle-pt`` is the
Pt-island catalog walk, which has no per-start cell.
``optbench.asv_engines`` builds the classes from the partn templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("partn", globals())
