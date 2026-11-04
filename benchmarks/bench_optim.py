"""OPTIM L-BFGS minimization, one class per template.

``optim/unhooked`` names no problem and OPTIM has no band path, so it has
no cell. ``optbench.asv_engines`` builds the classes from the optim
templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("optim", globals())
