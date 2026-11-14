"""The recorded asv class fields match what importing the modules produces."""

import importlib

from optbench.board import _benchmark_index

MODULES = (
    "benchmarks.bench_lj38",
    "benchmarks.bench_eon_develop",
    "benchmarks.bench_eon_svn",
    "benchmarks.bench_eon_lane",
    "benchmarks.bench_eon_client",
    "benchmarks.bench_rgmin",
    "benchmarks.bench_ogolem",
    "benchmarks.bench_ase_sella",
    "benchmarks.bench_rgsaddle",
    "benchmarks.bench_geodesic",
    "benchmarks.bench_airss",
    "benchmarks.bench_softsaddle",
    "benchmarks.bench_lammpsneb",
    "benchmarks.bench_noise",
    "benchmarks.bench_geometric",
    "benchmarks.bench_globalopt",
    "benchmarks.bench_n2p2",
    "benchmarks.bench_gmin",
    "benchmarks.bench_vcsqnm",
    "benchmarks.bench_pele",
    "benchmarks.bench_optking",
    "benchmarks.bench_pyamff",
    "benchmarks.quadratic",
)


def _from_imports() -> dict:
    index = {}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        short = module_name.split(".")[-1]
        for name, value in vars(module).items():
            if not isinstance(value, type):
                continue
            spec = getattr(value, "spec", None)
            template = getattr(value, "template_id", None) or getattr(value, "template_name", None)
            problem = getattr(value, "problem_id", None) or getattr(value, "problem_name", None)
            cap = getattr(value, "cap", None)
            if isinstance(spec, dict):
                template = template or spec.get("template_id") or spec.get("template")
                problem = problem or spec.get("problem_id") or spec.get("problem")
                if cap is None:
                    cap = spec.get("cap")
            index[f"{short}.{name}"] = {
                "template": str(template) if template else "",
                "problem": str(problem) if problem else "",
                "cap": int(cap) if isinstance(cap, int) or (isinstance(cap, float) and float(cap).is_integer()) else None,
            }
    return index


def test_recorded_fields_match_the_imported_classes() -> None:
    assert _benchmark_index() == _from_imports()
