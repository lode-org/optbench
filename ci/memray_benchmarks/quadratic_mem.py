"""One quadratic success, profiled for peak memory by asv_bench_memray."""

from optbench.statistics import kept_start


class QuadraticKeptMemory:
    def ray_kept(self):
        kept_start()
