"""COMPASS between two endpoints, one class per band problem.

``optbench.asv_engines`` builds the classes from the compass templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("compass", globals())
