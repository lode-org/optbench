"""eOn develop 500350d9 harness benchmarks.

One class is one template that sets ``kind`` and whose id starts with
``eon/500350d9-``, ``eon/develop-500350d9-``, or
``eon-client/de-2ts-500350d9-``. The binary is the develop build of
commit 500350d9fae881f06c7ef194e5868da079c4be5a named by
``EON_DEVELOP_500350D9``. It must print ``500350d9``.
``optbench.eon_harness`` runs the same ``run_case`` as the other harness
families. The default grid is one start unless ``OPTBENCH_ASV_FULL=1``
at import.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("500350d9", globals())
