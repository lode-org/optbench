"""Every eOn template names the force kernel its forces come from.

The ini still names the 2014 potential, and that name does not fix the
surface: ``lammps`` runs as ``lj40`` in one family and ``lj_cut`` in
another, and the two differ in the last bits. The kernel key does.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from benchmarks import bench_eon_develop, bench_eon_svn
from optbench import eon_harness
from optbench.campaign import benchmark_text, list_templates, load_template

ROOT = eon_harness.ROOT
SURFACES = {
    "lammps": {"lj40", "lj_cut"},
    "lj_cut": {"lj_cut"},
    "ljcluster": {"ljcluster"},
    "morse_pt": {"morse", "morse_shift", "morse_cluster"},
    "eam_al": {"eam"},
}


def _eon_templates() -> list[dict]:
    found = []
    for entry in list_templates(ROOT):
        if str(entry.get("id", "")).startswith("eon"):
            found.append(load_template(ROOT, entry["id"]))
    return found


def _harness_templates() -> set[str]:
    used = {spec["template"] for spec in bench_eon_develop._specs()}
    used |= {spec["template"] for spec in bench_eon_svn.SPECS}
    for family in eon_harness.FAMILIES:
        used |= {template["id"] for template in eon_harness._jobs(family)}
    return used


def test_every_eon_template_declares_a_known_kernel():
    templates = _eon_templates()
    assert len(templates) == 206
    unknown = {
        template["id"]: template.get("kernel")
        for template in templates
        if template.get("kernel") not in eon_harness.KERNELS | {eon_harness.CLIENT_KERNEL}
    }
    assert unknown == {}


def test_harness_templates_name_a_harness_kernel_and_the_rest_the_client():
    harness = _harness_templates()
    wrong = {}
    for template in _eon_templates():
        expected = template["id"] in harness or bool(template.get("kind"))
        if (template["kernel"] != eon_harness.CLIENT_KERNEL) != expected:
            wrong[template["id"]] = template["kernel"]
    assert wrong == {}


def test_the_kernel_computes_the_surface_the_ini_names():
    wrong = {}
    for template in _eon_templates():
        if template["kernel"] == eon_harness.CLIENT_KERNEL:
            continue
        potential = template["ini"]["Potential"]["potential"]
        if template["kernel"] not in SURFACES[potential]:
            wrong[template["id"]] = (potential, template["kernel"])
    assert wrong == {}


def test_the_client_kernel_has_no_harness_function():
    with pytest.raises(ValueError, match="client"):
        eon_harness.kernel_for(eon_harness.CLIENT_KERNEL)


def test_lj_cut_takes_the_archived_lammps_coefficients():
    archived = ROOT / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
    assert bench_eon_develop.lammps_lj_params(archived) == (1.0, 1.0, 40.0)
    rng = np.random.default_rng(20261103)
    positions = rng.uniform(0.0, 4.0, 3 * 38).tolist()
    assert eon_harness.kernel_for("lj_cut")(positions) == bench_eon_develop.lj_cut(
        positions, epsilon=1.0, sigma=1.0, cutoff=40.0
    )


def test_the_campaign_file_records_the_kernel():
    template = load_template(ROOT, "eon/500350d9-de-lj38-ci-swdneb")
    text = benchmark_text({"force_calls": 1, "nfailed": 0}, template)
    assert "kernel lj_cut" in text.splitlines()


def test_the_500350d9_family_runs_its_templates_with_their_kernels():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks import bench_eon_500350d9

    found = {
        template["id"]: (template["asv_class"], template["kernel"])
        for template in eon_harness._jobs("500350d9")
    }
    assert found == {
        "eon/500350d9-de-lj38-ci-swdneb": ("Eon500350d9DeLj38CiSwdneb", "lj_cut"),
        "eon/500350d9-neb-pt-cg": ("Eon500350d9NebPtCg", "morse_cluster"),
        "eon/500350d9-neb-pt-fire": ("Eon500350d9NebPtFire", "morse_cluster"),
        "eon/500350d9-neb-pt-lbfgs": ("Eon500350d9NebPtLbfgs", "morse_cluster"),
        "eon/500350d9-neb-pt-qm": ("Eon500350d9NebPtQm", "morse_cluster"),
        "eon/develop-500350d9-de-2ts": ("EonDevelop500350d9De2ts", "ljcluster"),
        "eon/develop-500350d9-de-2ts-ew": ("EonDevelop500350d9De2tsEw", "ljcluster"),
        "eon/develop-500350d9-de-2ts-oci": ("EonDevelop500350d9De2tsOci", "ljcluster"),
        "eon-client/de-2ts-500350d9-ci-neb": ("Eon500350d9ClientDe2tsCiNeb", "lj40"),
        "eon-client/de-2ts-500350d9-ci-swdneb": ("Eon500350d9ClientDe2tsCiSwdneb", "lj40"),
        "eon-client/de-2ts-500350d9-dimer": ("Eon500350d9ClientDe2tsDimer", "lj40"),
        "eon-client/de-2ts-500350d9-lanczos": ("Eon500350d9ClientDe2tsLanczos", "lj40"),
        "eon-client/de-2ts-500350d9-lanczos-improved": (
            "Eon500350d9ClientDe2tsLanczosImproved",
            "lj40",
        ),
    }
    for ident, (cls, _kernel) in found.items():
        bench = getattr(bench_eon_500350d9, cls)
        assert bench.template_name == ident
        assert len(bench.params[1]) == 1
