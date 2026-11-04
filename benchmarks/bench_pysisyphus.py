"""pysisyphus minimizers and Lanczos, one class per template.

``optbench.asv_engines`` builds the classes from the pysisyphus templates.
"""

from optbench.asv_engines import install_benchmarks as _install

_install("pysisyphus", globals())
