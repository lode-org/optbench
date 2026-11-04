"""DL-FIND minimizers, one class per template.

``optbench.asv_engines`` builds the classes from the dlfind templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("dlfind", globals())
