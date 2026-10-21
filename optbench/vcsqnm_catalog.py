"""vc-SQNM templates for the minimization family.

The pin is tag v1.3 of https://github.com/moritzgubler/vc-sqnm. The
distribution built from that commit reports version 1.2. The campaign
imports that Python package. The C++ and Fortran mains call the Bazant
potential and are not these rows.

``initial_step_size`` -1e-7 is the repaired first coefficient. The ASE
wrapper constructor uses -0.01. On LJ38 the stiffest archived force
component is about 6.1e5, and 0.01 times that component leaves the
cutoff, where the force is zero. A flat Hessian would then count as a
minimum. -1e-7 keeps the first displacement under 0.07 sigma on that
start. ``alpha_min`` stays 1e-3. At 1e-2 a later step dissociates the
cluster. Morse bulk uses the same coefficients. The estimator inside
SQNM replaces the coefficient after the first step.

The wrapper constructor uses ``eps_subsp`` 1e-3. On LJ38 that value
lets a later step leave the cutoff, and the flat region counts as a
minimum. Both rows use ``eps_subsp`` 1e-5, which keeps every archived
LJ38 start bound. The default history is 10. The tuned history is 20.

Lennard-Jones 38 has no cell coordinate in the spec potential, so the
variable-cell methods are Morse bulk only.
"""

from __future__ import annotations

from pathlib import Path

COMMIT = "4241692173cccb42a9ae05974bfd62285ec2cd7b"
TAG = "v1.3"
DIST_VERSION = "1.2"
CODE_VERSION = "1.3"
CODE = "vc-SQNM"
REPOSITORY = "https://github.com/moritzgubler/vc-sqnm"

SOURCE_SHA256 = {
    "sqnm.py": "8191e10b6ab005056885e87b2bfdff5443802961f2cbb2147d10e3ca54e93809",
    "free_or_fixed_cell_sqnm.py": "0f282637391459826afec22c9f5689428f92f18ad2df744a18926e11e1b7d31f",
    "periodic_sqnm.py": "e1e613fc9fdbbdadc385a06b26f5eaf14322b9b93c5703df87495dd6c67319e9",
    "vcsqnm_for_ase.py": "a1b8c2fb9e16341c8e756ee97d08e15ee9026ae31eeb99c0f56df670260d5a2a",
    "historylist.py": "7f04461f553356e41c35bee3eda38e3b9b64c3161dc0af2930be1507cde6b71f",
}

DEFAULT = {
    "initial_step_size": -1e-7,
    "nhist_max": 10,
    "alpha_min": 1e-3,
    "eps_subsp": 1e-5,
    "lattice_weight": 2.0,
}
TUNED = {
    "initial_step_size": -1e-7,
    "nhist_max": 20,
    "alpha_min": 1e-3,
    "eps_subsp": 1e-5,
    "lattice_weight": 2.0,
}

_PROBLEMS = {
    "minimization/lj38": ("lj38", 0.01),
    "minimization/morse-bulk": ("morse", 1.0e-3),
}
_METHODS = (
    ("sqnm", "SQNM", False),
    ("ase", "ASE-SQNM", False),
    ("vc", "vc-SQNM", True),
    ("ase-vc", "ASE-vc-SQNM", True),
)

_FIXED = (
    "Fixed-cell SQNM. initial_step_size is -1e-7 and alpha_min is 1e-3. "
    "The stop is the problem force norm. The ASE wrapper constructor uses "
    "initial_step_size -0.01 and eps_subsp 1e-3. On the stiffest LJ38 start "
    "a force component is about 6.1e5, and -0.01 carries that atom past the "
    "cutoff, where the force is zero. eps_subsp 1e-3 also lets a later step "
    "leave the cutoff. eps_subsp 1e-5 keeps every archived LJ38 start bound. "
    "Morse bulk uses the same coefficients."
)
_VC = (
    "Variable-cell SQNM on the Morse pair potential. The cell is a coordinate. "
    "The atomic force uses the problem L2 stop, and the max-norm of dE/dA "
    "uses that same number. Python periodic_sqnm multiplies the supplied "
    "lattice derivative by -1, so the harness passes -dE/dA and the step "
    "descends. lattice_weight is 2."
)


def _note(method: str, tuned: bool, variable_cell: bool) -> str:
    if variable_cell:
        body = _VC
        if method == "ase-vc":
            body += (
                " The call goes through aseOptimizer with vc_relax true. "
                "The calculator stress is (dE/dA) A^T / det(A), so the wrapper's "
                "deralat equals -dE/dA."
            )
        else:
            body += " The call goes through periodic_sqnm."
    else:
        body = _FIXED
        if method == "ase":
            body += " The call goes through aseOptimizer with vc_relax false. Force calls are harness calculator evaluations."
        else:
            body += " The call goes through free_sqnm."
    if tuned:
        body += " Tuned row: history length 20. eps_subsp stays 1e-5."
    else:
        body += " Default row: history length 10 and eps_subsp 1e-5."
    return body


def _toml_float(value: float) -> str:
    text = f"{float(value):.12g}"
    if "e" not in text and "." not in text:
        return text + ".0"
    return text


def specs() -> list[dict]:
    """One dict per minimization template, in stable order."""
    found = []
    for problem, (slug, force) in _PROBLEMS.items():
        for method, algorithm, variable_cell in _METHODS:
            if variable_cell and problem != "minimization/morse-bulk":
                continue
            for tuned, profile, parameters in (
                (False, "default", DEFAULT),
            ):
                suffix = "-tuned" if tuned else ""
                found.append(
                    {
                        "id": f"vcsqnm/{slug}-{method}{suffix}",
                        "engine": "vcsqnm",
                        "problem": problem,
                        "code": CODE,
                        "code_version": CODE_VERSION,
                        "commit": COMMIT,
                        "algorithm": algorithm,
                        "tuning": profile,
                        "note": _note(method, tuned, variable_cell),
                        "sqnm": {"method": method, **parameters},
                        "force_norm": float(force),
                        "cap": 10000,
                        "variable_cell": variable_cell,
                    }
                )
    return found


def render(spec: dict) -> str:
    lines = []
    for key in ("id", "engine", "problem", "code", "code_version", "commit", "algorithm", "tuning", "note"):
        value = spec[key]
        if key == "note":
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}")
    lines.append("")
    lines.append("[sqnm]")
    section = spec["sqnm"]
    lines.append(f'method = "{section["method"]}"')
    for key in ("initial_step_size", "nhist_max", "alpha_min", "eps_subsp", "lattice_weight"):
        value = section[key]
        if isinstance(value, float):
            lines.append(f"{key} = {_toml_float(value)}")
        else:
            lines.append(f"{key} = {int(value)}")
    lines.append("")
    lines.append("[ini.Optimizer]")
    lines.append(f"converged_force = {_toml_float(spec['force_norm'])}")
    lines.append(f"max_iterations = {int(spec['cap'])}")
    lines.append("")
    return "\n".join(lines)


def write_templates(root: Path) -> list[Path]:
    """Write templates/vcsqnm and return the paths."""
    directory = root / "templates" / "vcsqnm"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{spec['id'].split('/', 1)[1]}.toml" for spec in specs()}
    for path in directory.glob("*.toml"):
        if path.name not in expected:
            path.unlink()
    written = []
    for spec in specs():
        path = directory / f"{spec['id'].split('/', 1)[1]}.toml"
        path.write_text(render(spec), encoding="utf-8")
        written.append(path)
    return written
