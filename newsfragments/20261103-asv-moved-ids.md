The eOn develop `f6817b53` templates have asv classes in `bench_eon_f6817b53`.
The five `Eon500350d9ClientDe2ts*` classes moved from `bench_eon_client` to `bench_eon_500350d9` with the same grids.
Their asv ids changed with the module.
asv no longer discovers `bench_gmin_bh.TrackedSearch`, the imported base class that it listed as a benchmark.
`pending.Pending` is now one pending class per code, named with a Python identifier.
`benchmarks/pending.py` names the engine and template modules that already measure cells.
Each pending class omits the cells that those modules measure.
