"""Show that the asv benchmark server has numpy loaded before any benchmark module.

Run: uv run python docs/asv-gaps/numpy_preimport.py
"""

import inspect
import sys

import asv.environment as environment

before = "numpy" in sys.modules
import asv_runner.benchmarks  # noqa: E402,F401

loaded = "numpy" in sys.modules
print("numpy loaded before asv_runner.benchmarks:", before)
print("numpy loaded by asv_runner.benchmarks:", loaded)
print("pympler.asizeof loaded:", "pympler.asizeof" in sys.modules)
name = environment.get_env_name("uv-pip", "3.12", {}, {("nobuild", "OPENBLAS_NUM_THREADS"): "1"})
print("environment name with env_nobuild OPENBLAS_NUM_THREADS=1:", name)
ignored = "# Ignore requirement matrix" in inspect.getsource(environment.get_environments)
print("existing environments ignore the matrix:", ignored)
if before:
    raise SystemExit("numpy was loaded before the check; this check is stale")
if not loaded:
    raise SystemExit("asv_runner.benchmarks no longer imports numpy")
