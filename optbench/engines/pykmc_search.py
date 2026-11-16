"""pyKMC registry hook.

The lane name pyKMC is the SourceForge lkmc project, release 1.3
(M. Robinson, Loughborough), not the GitHub repository hockyg/PyKMC.
The 1.3 archive is Perl from February 2010. The kMC driver shells out
to ``LBOMD.exe``, which is not in the archive. There is no callback
the harness can wrap, and the tree has no licence file.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = "pykmc"
CODE = "pykmc"
REPOSITORY = "https://sourceforge.net/projects/lkmc/"
RELEASE = "1.3"


def available() -> str | None:
    return (
        f"pyKMC is {REPOSITORY} release {RELEASE}, a Perl on-the-fly kMC driver from 2010. "
        "TransSearchSSH.pl shells out to LBOMD.exe, and that binary is not in the archive. "
        "LKMC0.1.3.zip (65791 bytes) is the same Scripts tree and has no Fortran or C source. "
        "LBOMD is described as Loughborough's in-house molecular dynamics code, and no public source was found. "
        "There is no potential callback and no licence file, so the accelerated-md problems "
        "are not scored. hockyg/PyKMC is a different program and is not this pin."
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1, limit: int = 0) -> dict:
    del root, problem_name, template_name, output, workers, limit
    raise RuntimeError(available())
