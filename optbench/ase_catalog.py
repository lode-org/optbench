"""ASE 3.29.0 and Sella 2.6.0 template catalog.

The files under ``templates/ase`` and ``templates/sella`` are rendered
from this module. A default row is the library constructor. A tuned row
is the archived OptBench script when one exists.
"""

from __future__ import annotations

import json
from pathlib import Path

ASE_VERSION = "3.29.0"
ASE_COMMIT = "f27c0005ae6a67ea419f996e728668865bfc1f86"
ASE_REPOSITORY = "https://gitlab.com/ase/ase"
SELLA_VERSION = "2.6.0"
SELLA_COMMIT = "6cfb60cdb100a146c3c6f0587d5308b247aa8010"
SELLA_REPOSITORY = "https://github.com/zadorlab/sella"

OPTIMIZERS = (
    "BFGS",
    "LBFGS",
    "FIRE",
    "BFGSLineSearch",
    "GPMin",
    "PreconLBFGS",
    "PreconFIRE",
    "PreconODE12r",
)

_OPT_SLUG = {
    "BFGS": "bfgs",
    "LBFGS": "lbfgs",
    "FIRE": "fire",
    "BFGSLineSearch": "bfgslinesearch",
    "GPMin": "gpmin",
    "PreconLBFGS": "preconlbfgs",
    "PreconFIRE": "preconfire",
    "PreconODE12r": "preconode12r",
}

LJ = {
    "kind": "lennard-jones",
    "epsilon": 1.0,
    "sigma": 1.0,
    "rc": 40.0,
    "pbc": [False, False, False],
}
MORSE_PT = {
    "kind": "morse",
    "D": 0.7102,
    "alpha": 1.6047,
    "r0": 2.897,
    "rc": 7.0,
    "form": "shift",
    "pbc": [True, True, True],
}
MORSE_SLAB = {
    "kind": "morse",
    "D": 0.7102,
    "alpha": 1.6047,
    "r0": 2.897,
    "rc": 7.0,
    "form": "shift",
    "pbc": [True, True, False],
}
MORSE_SMOOTH = {
    "kind": "morse",
    "D": 0.7102,
    "alpha": 1.6047,
    "r0": 2.897,
    "rc": 9.5,
    "form": "smooth",
    "pbc": [True, True, False],
}

# Archived Sella trust region from the 0.1.1 LJ38 and Pt-island scripts.
SELLA_TUNED = {
    "eta": 1e-6,
    "gamma": 0.4,
    "rho_inc": 1.035,
    "rho_dec": 5.0,
    "sigma_inc": 1.15,
    "sigma_dec": 0.65,
    "delta0": 1.3e-3,
}
SELLA_TUNED_NOTE = (
    "Archived Sella trust region: eta 1e-6, gamma 0.4, delta0 1.3e-3, "
    "rho_inc 1.035, rho_dec 5, sigma_inc 1.15, sigma_dec 0.65."
)

_LJ_STOP = {
    "norm": "l2",
    "force_norm": 0.01,
    "max_force_calls": 10000,
    "max_steps": 10000,
}
_MORSE_STOP = {
    "norm": "l2",
    "force_norm": 0.001,
    "max_force_calls": 10000,
    "max_steps": 10000,
}
_TS_STOP = {
    "norm": "l2",
    "force_norm": 0.001,
    "max_steps": 1000,
    "max_force_calls": 100000,
}


def _opt_slug(name: str) -> str:
    return _OPT_SLUG[name]


def _minimization_specs() -> list[dict]:
    problems = (
        (
            "lj38",
            "minimization/lj38",
            "website/minimization/lj38/lj38-clusters.tgz",
            LJ,
            _LJ_STOP,
            "cluster",
        ),
        (
            "morse",
            "minimization/morse-bulk",
            "website/minimization/morse-bulk/morse-bulk.tgz",
            MORSE_PT,
            _MORSE_STOP,
            "con",
        ),
    )
    specs = []
    for slug, problem, archive, potential, stop, select in problems:
        for name in OPTIMIZERS:
            profile = "default"
            params = {}
            source = "default"
            note = f"ASE {ASE_VERSION} {name} constructor defaults."
            if problem == "minimization/lj38" and name in {"BFGS", "LBFGS"}:
                params = {"maxstep": 0.1}
                source = "repaired"
                note = (
                    f"ASE {ASE_VERSION} {name} constructor maxstep is 0.2. "
                    "On LJ38 starts that begin inside the repulsive core that step increases the force. "
                    "This default row uses maxstep 0.1. Morse defaults stay at the constructor value."
                )
            specs.append(
                {
                    "id": f"ase/{slug}-{_opt_slug(name)}-{profile}",
                    "engine": "ase",
                    "code": "ASE",
                    "code_version": ASE_VERSION,
                    "commit": ASE_COMMIT,
                    "algorithm": name,
                    "problem": problem,
                    "task": "minimize",
                    "profile": profile,
                    "reducer": "capped",
                    "ranked": True,
                    "parameter_source": source,
                    "note": note,
                    "optimizer": {"name": name, "parameters": params},
                    "potential": potential,
                    "stop": stop,
                    "input": {"archive_path": archive, "select": select},
                }
            )
        # Sella can minimize with order 0.
        profile = "default"
        params = {"order": 0}
        source = "default"
        note = f"Sella {SELLA_VERSION} minimum defaults, order 0."
        specs.append(
            {
                "id": f"sella/{slug}-min-{profile}",
                "engine": "sella",
                "code": "Sella",
                "code_version": SELLA_VERSION,
                "commit": SELLA_COMMIT,
                "algorithm": "Sella",
                "problem": problem,
                "task": "sella",
                "profile": profile,
                "reducer": "capped",
                "ranked": True,
                "parameter_source": source,
                "note": note,
                "optimizer": {"name": "Sella", "parameters": params},
                "potential": potential,
                "stop": stop,
                "input": {"archive_path": archive, "select": select},
                "sella": {"order": 0, "v0": "none"},
            }
        )
    return specs


def _ts_specs() -> list[dict]:
    rows = (
        {
            "slug": "lj38-ts",
            "problem": "tsopt/LJ38",
            "archive": "website/tsopt/LJ38/coords-con.tar.gz",
            "potential": LJ,
            "select": "con",
            "v0": "none",
            "pbc_note": "",
        },
        {
            "slug": "pt-island-ts",
            "problem": "tsopt/pt-island",
            "archive": "website/tsopt/pt-island/pt-island-con.tgz",
            "potential": MORSE_SMOOTH,
            "select": "initial",
            "v0": "reactant",
            "pbc_note": " The initial mode is the displacement from reactant.con.",
        },
    )
    specs = []
    for row in rows:
        profile = "default"
        params = {"order": 1}
        source = "default"
        note = f"Sella {SELLA_VERSION} saddle defaults, order 1." + row["pbc_note"]
        specs.append(
            {
                "id": f"sella/{row['slug']}-{profile}",
                "engine": "sella",
                "code": "Sella",
                "code_version": SELLA_VERSION,
                "commit": SELLA_COMMIT,
                "algorithm": "Sella",
                "problem": row["problem"],
                "task": "sella",
                "profile": profile,
                "reducer": "all",
                "ranked": True,
                "parameter_source": source,
                "note": note,
                "optimizer": {"name": "Sella", "parameters": params},
                "potential": row["potential"],
                "stop": _TS_STOP,
                "input": {"archive_path": row["archive"], "select": row["select"]},
                "sella": {"order": 1, "v0": row["v0"]},
            }
        )
    return specs


def _neb_specs() -> list[dict]:
    """NEB where the suite is a band between endpoints."""
    rows = (
        {
            "id": "neb-pt-heptamer",
            "problem": "neb/Pt-heptamer",
            "archive": "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz",
            "potential": MORSE_SLAB,
            "layout": "minima",
            "moving_images": 8,
            "reducer": "per_image",
            "stop": {
                "norm": "fmax",
                "force_norm": 0.01,
                "max_force_calls": 10000,
                "max_steps": 10000,
            },
            "tuned_optimizer": "LBFGS",
            "tuned_parameters": {"maxstep": 0.2},
            "tuned_spring": 5.0,
            "tuned_climb": True,
            "tuned_note": (
                "Archive neb/Pt-heptamer/eon-lbfgs uses 8 intermediate images, "
                "spring 5, climbing image, LBFGS, max move 0.2, force 0.01. "
                "force_calls is the harness total across the 8 images. "
                "force_calls_per_image divides that total by 8."
            ),
            "default_note": (
                "ASE NEB defaults on the archived 8 intermediate images: "
                "spring 0.1, climb off, FIRE. "
                "force_calls is the harness total across the 8 images. "
                "force_calls_per_image divides that total by 8."
            ),
        },
        {
            "id": "de-lj38",
            "problem": "double-ended-saddle-search/lj38",
            "archive": "website/double-ended-saddle-search/lj38/coords-con.tgz",
            "potential": LJ,
            "layout": "indexed",
            "count": 50,
            "moving_images": 5,
            "reducer": "capped",
            "stop": {
                "norm": "fmax",
                "gradient_norm": 0.001,
                "force_norm": 0.001,
                "max_force_calls": 2500,
                "max_steps": 1000,
            },
            "tuned_optimizer": "LBFGS",
            "tuned_parameters": {"maxstep": 0.01, "damping": 0.2},
            "tuned_spring": 5.0,
            "tuned_climb": False,
            "tuned_interpolate": "idpp",
            "tuned_climb_after": 0.4,
            "tuned_source": "repaired",
            "tuned_note": (
                "The previous ASE campaign, results/campaign/ase-3.29.0-de-lj38-tuned.dat, "
                "failed all 50 pairs at the 2500 cap with LBFGS maxstep 0.05 and climb from the first step. "
                "This row keeps 5 images, spring 5, and the 2500 eOn exclusion. "
                "Interpolation is IDPP. LBFGS maxstep is 0.01 and damping is 0.2. "
                "Climb stays off until the maximum residual is at or below 0.4."
            ),
            "default_note": (
                "ASE NEB defaults on those 50 pairs and 5 intermediate images: "
                "spring 0.1, climb off, FIRE. The same 2500 cap applies."
            ),
        },
        {
            "id": "de-pt-island",
            "problem": "double-ended-saddle-search/pt-island",
            "archive": "website/double-ended-saddle-search/pt-island/pt-island-con.tgz",
            "potential": MORSE_SLAB,
            "layout": "reactant-products",
            "moving_images": 3,
            "reducer": "capped",
            "stop": {
                "norm": "fmax",
                "force_norm": 0.001,
                "max_force_calls": 10000,
                "max_steps": 1000,
            },
            "tuned_optimizer": "LBFGS",
            "tuned_parameters": {"maxstep": 0.2},
            "tuned_spring": 0.1,
            "tuned_climb": True,
            "tuned_note": (
                "Archive eon-ci-neb uses 3 intermediate images, climbing image, "
                "and LBFGS. The config does not set a spring, so the spring stays 0.1. "
                "The problem file does not state a force cap; this row caps the harness total at 10000."
            ),
            "default_note": (
                "ASE NEB defaults on the same endpoints: 3 intermediate images, "
                "spring 0.1, climb off, FIRE, harness cap 10000."
            ),
        },
        {
            "id": "de-lj38-2ts",
            "problem": "double-ended-saddle-search/lj38_2ts_separation",
            "archive": "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz",
            "potential": LJ,
            "layout": "indexed",
            "moving_images": 5,
            "reducer": "capped",
            "stop": {
                "norm": "fmax",
                "force_norm": 0.001,
                "max_force_calls": 40000,
                "max_steps": 40000,
            },
            "tuned_optimizer": "LBFGS",
            "tuned_parameters": {"maxstep": 0.01, "damping": 0.2},
            "tuned_spring": 5.0,
            "tuned_climb": False,
            "tuned_interpolate": "idpp",
            "tuned_climb_after": 0.4,
            "tuned_source": "repaired",
            "tuned_note": (
                "The problem run limit is 10000. "
                "archive/entries/double-ended-saddle-search/lj38_2ts_separation/pele/benchmark.dat "
                "records median 20638 and maximum 114490, so this row's harness stop stays 40000 "
                "instead of that problem cap and instead of the LJ38 eOn exclusion of 2500. "
                "Campaign rows are scored at the problem cap. This asv class uses the harness stop. "
                "The previous ASE tuned campaign failed all 51 pairs at 2500. "
                "Interpolation is IDPP. LBFGS maxstep is 0.01 and damping is 0.2. "
                "Climb stays off until the maximum residual is at or below 0.4. "
                "Five images and spring 5 are unchanged."
            ),
            "default_note": (
                "ASE NEB defaults on the same pairs: spring 0.1, climb off, FIRE. "
                "The harness stop is 40000, the same as the tuned row. "
                "The problem run limit is 10000. Campaign rows use that cap. "
                "This asv class uses the harness stop."
            ),
        },
    )
    specs = []
    stock = {
        "double-ended-saddle-search/lj38",
        "double-ended-saddle-search/lj38_2ts_separation",
        "double-ended-saddle-search/pt-island",
    }
    for row in rows:
        for profile in ("default", "tuned"):
            if profile == "tuned" and row["problem"] in stock:
                continue
            if profile == "default":
                optimizer = "FIRE"
                params: dict = {}
                spring = 0.1
                climb = False
                source = "default"
                note = row["default_note"]
            else:
                optimizer = row["tuned_optimizer"]
                params = dict(row["tuned_parameters"])
                spring = row["tuned_spring"]
                climb = row["tuned_climb"]
                source = row.get("tuned_source", "archive")
                note = row["tuned_note"]
            neb = {
                "layout": row["layout"],
                "moving_images": row["moving_images"],
                "spring": spring,
                "climb": climb,
                "method": "improvedtangent",
            }
            if profile == "tuned" and row.get("tuned_interpolate"):
                neb["interpolate"] = row["tuned_interpolate"]
            if profile == "tuned" and row.get("tuned_climb_after") is not None:
                neb["climb_after"] = float(row["tuned_climb_after"])
            if "count" in row:
                neb["count"] = row["count"]
            spec = {
                "id": f"ase/{row['id']}-{profile}",
                "engine": "ase",
                "code": "ASE",
                "code_version": ASE_VERSION,
                "commit": ASE_COMMIT,
                "algorithm": f"NEB-{optimizer}",
                "problem": row["problem"],
                "task": "neb",
                "profile": profile,
                "reducer": row["reducer"],
                "ranked": True,
                "parameter_source": source,
                "note": note,
                "optimizer": {"name": optimizer, "parameters": params},
                "potential": row["potential"],
                "stop": row["stop"],
                "input": {"archive_path": row["archive"]},
                "neb": neb,
            }
            specs.append(spec)
    return specs


def _sella_neb_specs() -> list[dict]:
    """Sella order-1 refine after the ASE band on each double-ended problem."""
    specs = []
    for spec in _neb_specs():
        if not str(spec["problem"]).startswith("double-ended-saddle-search/"):
            continue
        profile = spec["profile"]
        if profile == "default":
            params = {"order": 1}
            source = "default"
            head = (
                f"Sella {SELLA_VERSION} order 1 defaults refine the highest-energy image "
                "after the ASE default band (FIRE, spring 0.1, climb off, linear interpolation). "
            )
        else:
            params = {"order": 1, **SELLA_TUNED}
            source = spec["parameter_source"]
            head = (
                "Sella refines the highest-energy image of this problem's tuned ASE band. "
                + SELLA_TUNED_NOTE
                + " "
            )
        neb = dict(spec["neb"])
        neb["band_optimizer"] = spec["optimizer"]["name"]
        neb["band_parameters"] = dict(spec["optimizer"].get("parameters") or {})
        slug = spec["id"].split("/", 1)[1]
        specs.append(
            {
                "id": f"sella/{slug}",
                "engine": "sella",
                "code": "Sella",
                "code_version": SELLA_VERSION,
                "commit": SELLA_COMMIT,
                "algorithm": "Sella+NEB",
                "problem": spec["problem"],
                "task": "sella-neb",
                "profile": profile,
                "reducer": spec["reducer"],
                "ranked": True,
                "parameter_source": source,
                "note": (
                    head
                    + "The band stops at 80 percent of the call cap and both stages share one harness counter. "
                    + "v0 is unset: a free cluster drops rigid modes, so a full Cartesian mode has the wrong length. "
                    + spec["note"]
                ),
                "optimizer": {"name": "Sella", "parameters": params},
                "potential": spec["potential"],
                "stop": spec["stop"],
                "input": spec["input"],
                "neb": neb,
                "sella": {"order": 1, "v0": "none", "reserve": 0.2},
            }
        )
    return specs


def _sella_search_specs() -> list[dict]:
    """Single-ended Sella samples on the Pt island. Not a ranked catalog score."""
    specs = []
    for profile, params, source, note in (
        (
            "default",
            {"order": 1},
            "default",
            f"Sella {SELLA_VERSION} saddle defaults, order 1. ",
        ),
    ):
        specs.append(
            {
                "id": f"sella/pt-island-search-{profile}",
                "engine": "sella",
                "code": "Sella",
                "code_version": SELLA_VERSION,
                "commit": SELLA_COMMIT,
                "algorithm": "Sella",
                "problem": "saddle-search/pt-island",
                "task": "sella-search",
                "profile": profile,
                "reducer": "capped",
                "ranked": False,
                "parameter_source": source,
                "note": (
                    note
                    + "Sixteen searches, each a Gaussian displacement of standard deviation 0.1 "
                    + "on the free atoms, then Sella order 1 with v0 unset. "
                    + "One search is not the saddle catalog, so this row is not ranked and "
                    + "fraction_connected is not reported. The cost is the per-search harness count."
                ),
                "optimizer": {"name": "Sella", "parameters": params},
                "potential": MORSE_SLAB,
                "stop": {
                    "norm": "l2",
                    "force_norm": 0.001,
                    "max_force_calls": 10000,
                    "max_steps": 5000,
                },
                "input": {
                    "archive_path": "website/saddle-search/pt-island/pt-island.con",
                    "select": "file",
                },
                "sella": {"order": 1, "v0": "none"},
                "sella_search": {"searches": 16, "gauss_std": 0.1},
            }
        )
    return specs


def _dimer_specs() -> list[dict]:
    """Single-ended dimer on the Pt island. Not a ranked catalog score."""
    specs = []
    for profile, params, source, note in (
        (
            "default",
            {},
            "default",
            "ASE 3.29.0 DimerControl defaults. One search is not the saddle catalog.",
        ),
    ):
        specs.append(
            {
                "id": f"ase/dimer-pt-island-{profile}",
                "engine": "ase",
                "code": "ASE",
                "code_version": ASE_VERSION,
                "commit": ASE_COMMIT,
                "algorithm": "Dimer",
                "problem": "saddle-search/pt-island",
                "task": "dimer",
                "profile": profile,
                "reducer": "capped",
                "ranked": False,
                "parameter_source": source,
                "note": note,
                "optimizer": {"name": "Dimer", "parameters": params},
                "potential": MORSE_SLAB,
                "stop": {
                    "norm": "l2",
                    "force_norm": 0.001,
                    "max_force_calls": 10000,
                    "max_steps": 2000,
                },
                "input": {
                    "archive_path": "website/saddle-search/pt-island/pt-island.con",
                    "select": "file",
                },
                "dimer": {"searches": 16},
            }
        )
    return specs


def all_specs() -> list[dict]:
    return (
        _minimization_specs()
        + _ts_specs()
        + _neb_specs()
        + _sella_neb_specs()
        + _dimer_specs()
        + _sella_search_specs()
    )


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(type(value))


def _write_table(lines: list[str], name: str, table: dict) -> None:
    nested = {key: value for key, value in table.items() if isinstance(value, dict)}
    flat = {key: value for key, value in table.items() if not isinstance(value, dict)}
    if flat or not nested:
        lines.append(f"[{name}]")
        for key, value in flat.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    for key, value in nested.items():
        _write_table(lines, f"{name}.{key}", value)


def render_toml(spec: dict) -> str:
    lines = []
    scalars = (
        "id",
        "engine",
        "code",
        "code_version",
        "commit",
        "algorithm",
        "problem",
        "task",
        "profile",
        "reducer",
        "ranked",
        "parameter_source",
        "note",
    )
    for key in scalars:
        lines.append(f"{key} = {_toml_scalar(spec[key])}")
    lines.append("")
    _write_table(lines, "optimizer", spec["optimizer"])
    _write_table(lines, "potential", spec["potential"])
    _write_table(lines, "stop", spec["stop"])
    if spec.get("input"):
        _write_table(lines, "input", spec["input"])
    for key in ("neb", "sella", "dimer", "sella_search"):
        if key in spec:
            _write_table(lines, key, spec[key])
    return "\n".join(lines).rstrip() + "\n"


_TEMPLATE_MARK = "# generated ASE and Sella templates"


def _registry_templates(specs: list[dict]) -> str:
    lines = [_TEMPLATE_MARK, ""]
    for spec in specs:
        lines.extend(
            [
                "[[template]]",
                f"id = {_toml_scalar(spec['id'])}",
                f"code = {_toml_scalar(spec['engine'])}",
                f"problem = {_toml_scalar(spec['problem'])}",
                f"profile = {_toml_scalar(spec['profile'])}",
                f"task = {_toml_scalar(spec['task'])}",
                f"ranked = {_toml_scalar(spec['ranked'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_all(root: Path) -> list[Path]:
    """Write ``templates/ase`` and ``templates/sella`` from the catalog."""
    specs = all_specs()
    written: list[Path] = []
    for spec in specs:
        path = root / "templates" / f"{spec['id']}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_toml(spec), encoding="utf-8")
        written.append(path)
    expected = {path.resolve() for path in written}
    for folder in ("ase", "sella"):
        directory = root / "templates" / folder
        if not directory.is_dir():
            continue
        for path in directory.glob("*.toml"):
            if path.resolve() not in expected:
                path.unlink()
    registry = root / "codes" / "registry.toml"
    text = registry.read_text(encoding="utf-8")
    if _TEMPLATE_MARK in text:
        text = text[: text.index(_TEMPLATE_MARK)]
    registry.write_text(text.rstrip() + "\n\n" + _registry_templates(specs), encoding="utf-8")
    return written
