"""TSASE NEB, dimer, mode, and minimizers, one class per cell.

A band template is one class per band problem. ``optbench.asv_engines``
builds the classes from the tsase templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("tsase", globals())
