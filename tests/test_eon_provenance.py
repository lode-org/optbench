"""The eOn provenance writer records a binary and refuses an excuse."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
CC = "gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
CXX = "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
FC = "GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
OPTIONS = [
    "-Dbuildtype=release",
    "-Doptimization=3",
    "-Dprefix=/opt/eon",
    "-Dlibdir=lib",
]
COMMAND = (
    "CC=gcc CXX=g++ FC=gfortran meson setup builddir "
    "-Dbuildtype=release -Doptimization=3 -Dprefix=/opt/eon -Dlibdir=lib"
)


def _catalog():
    path = ROOT / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog_provenance", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document(catalog, **overrides):
    payload = dict(
        binary_sha256=SHA,
        version_stdout="3.5.0 (d9cc6c5be)",
        cc_stdout=CC,
        cxx_stdout=CXX,
        fc_stdout=FC,
        cc="gcc",
        cxx="g++",
        fc="gfortran",
        builddir="builddir",
        options=list(OPTIONS),
        meson="1.12.1",
        ninja="1.13.2",
        source={"commit": "d9cc6c5bee432d625eb6c56d7ea74b723a37ee3b", "tag": "v3.5.0"},
        run={"communicator_jobs": 1, "core": 4},
        argv_origin="Published command lines use the token eonclient.",
    )
    payload.update(overrides)
    return catalog.provenance_document(**payload)


def test_meson_setup_command_includes_optimization():
    catalog = _catalog()
    command = catalog.meson_setup_command(
        cc="gcc",
        cxx="g++",
        fc="gfortran",
        builddir="builddir",
        options=list(OPTIONS),
    )
    assert command == COMMAND
    assert "-Doptimization=3" in command


def test_provenance_records_full_compiler_versions():
    catalog = _catalog()
    document = _document(catalog)
    compilers = document["build"]["compilers"]
    assert compilers["cc"] == CC.strip()
    assert compilers["cxx"] == CXX.strip()
    assert compilers["fc"] == FC.strip()
    assert document["build"]["commands"][0] == COMMAND
    assert document["binary"]["sha256"] == SHA


def test_a_bare_fortran_version_is_refused():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="full compiler version"):
        _document(catalog, fc_stdout="gfortran\n")


def test_a_null_sha256_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["binary"]["sha256"] = None
    with pytest.raises(catalog.ProvenanceError, match="null"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_note_field_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["binary"]["sha256_note"] = "removed"
    with pytest.raises(catalog.ProvenanceError, match="note"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_an_excuse_field_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["excuse"] = "the prefix was removed"
    with pytest.raises(catalog.ProvenanceError, match="excuse"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_an_unexpanded_variable_is_refused():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="unexpanded"):
        catalog.meson_setup_command(
            cc="gcc",
            cxx="g++",
            fc="gfortran",
            builddir="builddir",
            options=["-Dprefix=$PREFIX", "-Doptimization=3"],
        )


def test_a_command_that_omits_optimization_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = COMMAND.replace(" -Doptimization=3", "")
    with pytest.raises(catalog.ProvenanceError, match="optimization"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_junk_compiler_output_is_refused():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="full compiler version"):
        _document(catalog, cc_stdout="junk 1.2\n", cxx_stdout="junk 1.2\n", fc_stdout="junk 1.2\n")


def test_an_empty_option_list_is_refused():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="optimization"):
        catalog.meson_setup_command(cc="gcc", cxx="g++", fc="gfortran", builddir="builddir", options=[])


def test_a_short_meson_command_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = "CC=gcc CXX=g++ FC=gfortran meson setup builddir"
    document["build"]["options"] = []
    with pytest.raises(catalog.ProvenanceError, match="optimization"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_junk_paren_compiler_strings_are_refused():
    catalog = _catalog()
    for text in ("gcc (junk) 1.2\n", "cc (x) 0.0\n"):
        with pytest.raises(catalog.ProvenanceError, match="full compiler version"):
            _document(catalog, cc_stdout=text, cxx_stdout=text, fc_stdout=FC)


def test_a_fortran_banner_is_refused_as_cc():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="not a full version string"):
        _document(catalog, cc_stdout=FC)


def test_a_c_banner_is_refused_as_fc():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="not a full version string"):
        _document(catalog, fc_stdout=CC)


def test_a_plain_clang_banner_is_refused_as_cxx():
    catalog = _catalog()
    with pytest.raises(catalog.ProvenanceError, match="full compiler version"):
        _document(catalog, cxx_stdout="clang version 1.0\n")


def test_ubuntu_clang_banners_match_their_argv():
    catalog = _catalog()
    line = "Ubuntu clang version 18.1.3 (1ubuntu1)\n"
    document = _document(catalog, cc_stdout=line, cxx_stdout=line, cc="clang", cxx="clang++")
    assert document["build"]["compilers"]["cc"] == line.strip()
    assert document["build"]["commands"][0].startswith("CC=clang CXX=clang++ FC=gfortran meson setup ")


def test_an_empty_meson_cc_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = COMMAND.replace("CC=gcc", "CC=", 1)
    with pytest.raises(catalog.ProvenanceError, match="empty CC"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_meson_cc_that_disagrees_with_the_compiler_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = COMMAND.replace("CC=gcc", "CC=clang", 1)
    with pytest.raises(catalog.ProvenanceError, match="CC does not match"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_trailing_hash_does_not_supply_meson_options(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = "CC=gcc CXX=g++ FC=gfortran meson setup builddir # " + " ".join(OPTIONS)
    document["build"]["options"] = []
    with pytest.raises(catalog.ProvenanceError, match="optimization"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_shell_operator_does_not_supply_meson_options(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = (
        "CC=gcc CXX=g++ FC=gfortran meson setup builddir && echo " + " ".join(OPTIONS)
    )
    document["build"]["options"] = []
    with pytest.raises(catalog.ProvenanceError, match="optimization"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_meson_command_without_a_build_directory_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = "CC=gcc CXX=g++ FC=gfortran meson setup " + " ".join(OPTIONS)
    with pytest.raises(catalog.ProvenanceError, match="build directory"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_an_invalid_buildtype_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = COMMAND.replace("-Dbuildtype=release", "-Dbuildtype=junk", 1)
    document["build"]["options"] = [option.replace("release", "junk") for option in OPTIONS]
    with pytest.raises(catalog.ProvenanceError, match="buildtype"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_tilde_prefix_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["build"]["commands"][0] = COMMAND.replace("-Dprefix=/opt/eon", "-Dprefix=~/eon", 1)
    document["build"]["options"] = [option.replace("/opt/eon", "~/eon") for option in OPTIONS]
    with pytest.raises(catalog.ProvenanceError, match="prefix"):
        catalog.write_provenance(tmp_path / "provenance.json", document)


def test_a_comment_key_is_refused(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    for key in ("comment", "caveat", "remark", "NoteText"):
        broken = dict(document)
        broken[key] = "aside"
        with pytest.raises(catalog.ProvenanceError, match="allowed key"):
            catalog.write_provenance(tmp_path / "provenance.json", broken)


def test_write_command_requires_the_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = _catalog()
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    monkeypatch.delenv("RGSADDLE_LIB", raising=False)
    with pytest.raises(catalog.ProvenanceError, match="RGSADDLE_LIB"):
        catalog._write_command("dimer-fire", 0)


def test_an_invalid_sha256_is_not_called_null(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    document["binary"]["sha256"] = "abcd"
    with pytest.raises(catalog.ProvenanceError, match="invalid") as caught:
        catalog.write_provenance(tmp_path / "provenance.json", document)
    assert "null" not in str(caught.value)


def _facts() -> dict:
    return {
        "version_stdout": "3.5.0 (d9cc6c5be)",
        "cc_stdout": CC,
        "cxx_stdout": CXX,
        "fc_stdout": FC,
        "cc": "gcc",
        "cxx": "g++",
        "fc": "gfortran",
        "builddir": "builddir",
        "options": list(OPTIONS),
        "meson": "1.12.1",
        "ninja": "1.13.2",
        "source": {"commit": "d9cc6c5bee432d625eb6c56d7ea74b723a37ee3b", "tag": "v3.5.0"},
        "run": {"communicator_jobs": 1, "core": 0, "threads": {"OMP_NUM_THREADS": "1"}},
        "argv_origin": "Published command lines use the token eonclient.",
    }


def test_run_eon_writes_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = _catalog()
    destination = tmp_path / "rgsaddle-develop-aa6dd937-eon-provenance.json"
    monkeypatch.setattr(catalog, "PROVENANCE_PATH", destination)
    monkeypatch.setattr(catalog, "_single_thread", lambda _cpu: None)

    def stop(_method: str) -> Path:
        raise RuntimeError("stop-after-provenance")

    monkeypatch.setattr(catalog, "_checkpoint_dir", stop)
    client = tmp_path / "eonclient"
    client.write_bytes(b"eonclient")
    with pytest.raises(RuntimeError, match="stop-after-provenance"):
        catalog.run_eon(0, client, _facts())
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["binary"]["sha256"] == hashlib.sha256(b"eonclient").hexdigest()
    assert written["build"]["compilers"]["fc"] == FC.strip()
    assert "-Doptimization=3" in written["build"]["commands"][0]
    assert "-Dprefix=/opt/eon" in written["build"]["commands"][0]
    assert "-Dlibdir=lib" in written["build"]["commands"][0]
    assert "-Dbuildtype=release" in written["build"]["commands"][0]


def test_cli_provenance_writes_the_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = _catalog()
    destination = tmp_path / "provenance.json"
    monkeypatch.setattr(catalog, "PROVENANCE_PATH", destination)
    client = tmp_path / "eonclient"
    client.write_bytes(b"eonclient")
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(_facts()), encoding="utf-8")
    catalog.main(["provenance", str(client), str(facts)])
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["binary"]["sha256"] == hashlib.sha256(b"eonclient").hexdigest()
    assert written["build"]["commands"][0].startswith("CC=gcc CXX=g++ FC=gfortran meson setup ")


def test_write_command_records_the_library_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = _catalog()
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    lib = tmp_path / "librgsaddle.so"
    lib.write_bytes(b"library")
    monkeypatch.setenv("RGSADDLE_LIB", str(lib))
    catalog._write_command("dimer-fire", 0)
    payload = json.loads((tmp_path / "build" / "rgsaddle" / "dimer-fire" / "command.json").read_text(encoding="utf-8"))
    assert payload["rgsaddle_lib_sha256"] == hashlib.sha256(b"library").hexdigest()


def test_write_provenance_records_the_document(tmp_path: Path):
    catalog = _catalog()
    document = _document(catalog)
    path = tmp_path / "rgsaddle-develop-aa6dd937-eon-provenance.json"
    catalog.write_provenance(path, document)
    written = path.read_text(encoding="utf-8")
    assert COMMAND in written
    assert FC.strip() in written
    assert "sha256_note" not in written
    assert "$" not in written
