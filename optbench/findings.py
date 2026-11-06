"""Archive rows beside later campaign files.

The archived numbers are the archived rows. A campaign file recorded next
to a row is a finding: the same metric, both numbers, and the gap.
There is no pass/fail window. ``oracle/findings.jsonl`` is that record.
The 2014 pele transition-state stop is the first RMS crossing plus one
eigenvector search, with the transverse-screen calls left out.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Numbered tables in Chill et al., JCTC 2014, 10, 5476. SoftSaddle is the
# April 2019 website, already in the oracle, and is not one of those tables.
PAPER = {
    "minimization/lj38/optim",
    "minimization/lj38/pele-lbfgs-M100-maxstep0.2",
    "minimization/lj38/eon-lbfgs",
    "minimization/lj38/ase-lbfgs",
    "minimization/lj38/eon-cg",
    "minimization/lj38/eon-fire",
    "minimization/lj38/eon-qm",
    "minimization/lj38/eon-sd",
    "minimization/morse-bulk/optim",
    "minimization/morse-bulk/pele-lbfgs",
    "minimization/morse-bulk/eon-lbfgs",
    "minimization/morse-bulk/ase-lbfgs",
    "minimization/morse-bulk/eon-cg",
    "minimization/morse-bulk/eon-fire",
    "minimization/morse-bulk/eon-sd",
    "globalopt/lj38/gmin-csm",
    "globalopt/lj38/gmin-symmetrise",
    "globalopt/lj38/gmin-nosym",
    "globalopt/lj38/eon-basinhopping",
    "globalopt/lj38/pele-basinhopping",
    "globalopt/lj100-binary/gmin",
    "globalopt/lj100-binary/eon-basinhopping",
    "globalopt/tip4p10/gmin",
    "globalopt/tip4p10/gmin-tbp",
    "globalopt/tip4p15/gmin",
    "globalopt/tip4p15/gmin-tbp",
    "globalopt/tip4p20/gmin",
    "globalopt/tip4p20/gmin-tbp",
    "lowest-eigenvector/lj38/optim-first-order",
    "lowest-eigenvector/lj38/pele-first-order",
    "lowest-eigenvector/lj38/tsase-lanczos",
    "lowest-eigenvector/lj38/tsase-dimer-bfgs",
    "lowest-eigenvector/lj38/eon-dimer-lbfgs",
    "lowest-eigenvector/lj38/eon-lanczos",
    "lowest-eigenvector/lj38/tsase-dimer-cg",
    "lowest-eigenvector/lj38/eon-dimer-cg",
    "tsopt/LJ38/optim_ffd",
    "tsopt/LJ38/pele_ffd",
    "tsopt/LJ38/eon-lanczos-improved",
    "tsopt/LJ38/eon-dimer",
    "tsopt/pt-island/optim_ffd",
    "tsopt/pt-island/pele",
    "tsopt/pt-island/eon-lanczos-improved",
    "tsopt/pt-island/eon-dimer",
    "double-ended-saddle-search/lj38/optim",
    "double-ended-saddle-search/lj38/eon-lanczos-improved",
    "double-ended-saddle-search/lj38/pele",
    "double-ended-saddle-search/lj38/eon-dimer",
    "double-ended-saddle-search/lj38/eon-ci-neb",
    "double-ended-saddle-search/lj38/eon-ci-swdneb",
    "double-ended-saddle-search/pt-island/optim",
    "double-ended-saddle-search/pt-island/eon-ci-neb-single",
    "double-ended-saddle-search/pt-island/eon-lanczos",
    "double-ended-saddle-search/pt-island/eon-dimer",
    "double-ended-saddle-search/pt-island/eon-ci-neb",
    "double-ended-saddle-search/pt-island/pele",
    "double-ended-saddle-search/pt-island/eon-ci-neb-5",
}

PAPER_METRIC = {
    "minimization/lj38/eon-fire": ("force_calls", 645.0),
    "double-ended-saddle-search/pt-island/eon-lanczos": ("force_calls", 165.0),
    "double-ended-saddle-search/pt-island/eon-ci-neb": ("nfailed", 3.0),
}

SAMPLE_N = {
    "minimization/lj38": 1000,
    "minimization/morse-bulk": 100,
    "tsopt/LJ38": 200,
    "tsopt/pt-island": 49,
    "double-ended-saddle-search/lj38": 50,
    "double-ended-saddle-search/lj38_2ts_separation": 51,
    "double-ended-saddle-search/pt-island": 59,
    "lowest-eigenvector/lj38": 200,
    "globalopt/lj38": 100,
    "globalopt/lj75": 200,
    "globalopt/lj98": 100,
    "globalopt/lj150": 100,
    "globalopt/tip4p10": 10000,
    "globalopt/tip4p15": 500,
    "globalopt/tip4p20": 100,
    "neb/Pt-heptamer": 13,
}

SHA = {
    "main": "883b4153baa013b52de36c4512f0859bea15dc35",
    "cells/optim/min": "f3cf41d74f73fb2ffb3b5199291c90a4743a1000",
    "cells/optim/ts": "3c03653c03cd714cd42db7e68d165c5158f40457",
    "cells/optim/saddle-pt-repro": "b60e4596c60931ce37347442b3bcc59c8877c4e7",
    "cells/pele/min": "5bb7c3801f52b697c48ada5811c5f766dd81a23d",
    "cells/pele/go": "0528a37b8ff6e54805b9fa75345ee3c40b756a5d",
    "cells/pele/ts-repro": "54ac7e7a1aa6581c261c17fc2c52a20e77121911",
    "cells/softsaddle/ts-reps": "bd1a5c9fc2685f5876266d53715accb31a049e73",
    "entries/eon-svn-htst": "715bc521c827f1f18102c1c027d99d0d8624b0fb",
    "entries/eon-svn-lane-a": "c140d970f659ba27d6048130f0e67fa02944bd7a",
    "cells/gmin/go-paper": "71f366d9e50b7eedcdc22c75ab0cbba0188d578c",
    "cells/gmin/go": "7e49bdf6da152047d721dc206bd2d526cfcdd189",
    "infra/aladatom-repro": "d3a28c98d4f544aa3f79876028f95d7e352fd17b",
}

# Exact-setup campaigns. kind is dat, screen, or htst.
# value_from renames a file field onto the archived metric name.
MATCHES: list[dict] = [
    # minimization / lj38
    {"key": "minimization/lj38/eon-lbfgs", "template": "eon-client/lbfgs-m100", "branch": "main", "path": "results/campaign/eon-client-lbfgs-m100.dat"},
    {"key": "minimization/lj38/eon-cg", "template": "eon-client/cg", "branch": "main", "path": "results/campaign/eon-client-cg.dat"},
    {"key": "minimization/lj38/eon-fire", "template": "eon-client/fire", "branch": "main", "path": "results/campaign/eon-client-fire.dat"},
    {"key": "minimization/lj38/eon-qm", "template": "eon-client/qm", "branch": "main", "path": "results/campaign/eon-client-qm.dat"},
    {"key": "minimization/lj38/eon-sd", "template": "eon-client/sd", "branch": "main", "path": "results/campaign/eon-client-sd.dat"},
    {"key": "minimization/lj38/eon-sd-twopoint", "template": "eon-client/sd-twopoint", "branch": "main", "path": "results/campaign/eon-client-sd-twopoint.dat"},
    {"key": "minimization/lj38/ase-lbfgs", "template": "ase/lj38-lbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-lbfgs-tuned.dat"},
    {"key": "minimization/lj38/ase-BFGS", "template": "ase/lj38-bfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-bfgs-tuned.dat"},
    {"key": "minimization/lj38/ase-bfgslinesearch", "template": "ase/lj38-bfgslinesearch-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-bfgslinesearch-tuned.dat"},
    {"key": "minimization/lj38/ase-fire", "template": "ase/lj38-fire-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-fire-tuned.dat"},
    {"key": "minimization/lj38/optim", "template": "optim/lj38-lbfgs-tuned", "branch": "cells/optim/min", "path": "results/campaign/optim-02aaf45245af-lj38-lbfgs-tuned.dat", "extra": {"unrounded_mean": 175.537}},
    {"key": "minimization/lj38/pele-lbfgs-M1-maxstep0.1", "template": "pele-lbfgs-M1-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M1-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M10-maxstep0.1", "template": "pele-lbfgs-M10-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M10-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M100-maxstep0.2", "template": "pele-lbfgs-M100-maxstep0.2", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M100-maxstep0.2.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M20-maxstep0.1", "template": "pele-lbfgs-M20-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M20-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.05", "template": "pele-lbfgs-M4-maxstep0.05", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.05.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.1", "template": "pele-lbfgs-M4-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.2", "template": "pele-lbfgs-M4-maxstep0.2", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.2.dat"},
    {"key": "minimization/lj38/pele-fire", "template": "pele/lj38-fire", "branch": "cells/pele/min", "path": "results/campaign/pele-b7081b93-lj38-fire.dat", "cause": "pele_fire"},
    # minimization / morse
    {"key": "minimization/morse-bulk/eon-lbfgs", "template": "eon-client/morse-lbfgs", "branch": "main", "path": "results/campaign/eon-client-morse-lbfgs.dat"},
    {"key": "minimization/morse-bulk/eon-cg", "template": "eon-client/morse-cg", "branch": "main", "path": "results/campaign/eon-client-morse-cg.dat"},
    {"key": "minimization/morse-bulk/eon-fire", "template": "eon-client/morse-fire", "branch": "main", "path": "results/campaign/eon-client-morse-fire.dat"},
    {"key": "minimization/morse-bulk/eon-sd", "template": "eon-client/morse-sd", "branch": "main", "path": "results/campaign/eon-client-morse-sd.dat", "cause": "morse_sd"},
    {"key": "minimization/morse-bulk/optim", "template": "optim/morse-lbfgs-tuned", "branch": "cells/optim/min", "path": "results/campaign/optim-02aaf45245af-morse-lbfgs-tuned.dat", "cause": "optim_morse_memory"},
    {"key": "minimization/morse-bulk/pele-lbfgs", "template": "pele/morse-lbfgs", "branch": "cells/pele/min", "path": "results/campaign/pele-b7081b93-morse-lbfgs.dat", "cause": "pele_morse"},
    {"key": "minimization/morse-bulk/ase-lbfgs", "template": "ase/morse-lbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-lbfgs-tuned.dat", "cause": "ase_morse_script"},
    {"key": "minimization/morse-bulk/ase-fire", "template": "ase/morse-fire-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-fire-tuned.dat", "cause": "ase_morse_script"},
    {"key": "minimization/morse-bulk/ase-preconLBFGS", "template": "ase/morse-preconlbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-preconlbfgs-tuned.dat", "cause": "ase_morse_script"},
    # lowest eigenvector
    {"key": "lowest-eigenvector/lj38/eon-dimer-cg", "template": "eon-client/low-lj38-dimer-cg", "branch": "main", "path": "results/campaign/eon-client-low-lj38-dimer-cg.dat"},
    {"key": "lowest-eigenvector/lj38/eon-dimer-lbfgs", "template": "eon-client/low-lj38-dimer-lbfgs", "branch": "main", "path": "results/campaign/eon-client-low-lj38-dimer-lbfgs.dat"},
    {"key": "lowest-eigenvector/lj38/eon-lanczos", "template": "eon-svn/low-lj38-lanczos-ljcut", "branch": "main", "path": "results/campaign/eon-svn-low-lj38-lanczos-ljcut.dat"},
    {"key": "lowest-eigenvector/lj38/pele-first-order", "template": "pele/lj38-loweig-forward", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-forward.dat"},
    {"key": "lowest-eigenvector/lj38/pele-second-order", "template": "pele/lj38-loweig-central", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-central.dat"},
    {"key": "lowest-eigenvector/lj38/optim-first-order", "template": "optim/rr-loweig", "branch": "cells/optim/ts", "path": "results/campaign/optim-df8da599-rr-loweig.dat", "cause": "optim_rr_defaults"},
    # tsopt
    {"key": "tsopt/LJ38/eon-dimer", "template": "eon-svn/ts-lj38-dimer-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-dimer-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos", "template": "eon-svn/ts-lj38-lanczos-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos-improved", "template": "eon-svn/ts-lj38-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-improved.dat", "cause": "improved_unset"},
    {"key": "tsopt/LJ38/sella", "template": "sella/lj38-ts-tuned", "branch": "main", "path": "results/campaign/sella-2.6.0-lj38-ts-tuned.dat", "cause": "sella_trust"},
    {"key": "tsopt/LJ38/pele_ffd", "template": "pele/lj38-ts-ffd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-ffd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts"},
    {"key": "tsopt/LJ38/pele_cd", "template": "pele/lj38-ts-cd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-cd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts_cd"},
    {"key": "tsopt/LJ38/optim_ffd", "template": "optim/hef-ffd-ts-lj38", "branch": "cells/optim/ts", "path": "results/campaign/optim-df8da599-hef-ffd-ts-lj38.dat", "cause": "optim_lj38_ts"},
    {"key": "tsopt/pt-island/eon-dimer", "template": "eon-svn/ts-pt-dimer-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-dimer-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos", "template": "eon-svn/ts-pt-lanczos-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos-improved", "template": "eon-svn/ts-pt-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-improved.dat"},
    {"key": "tsopt/pt-island/sella", "template": "sella/pt-island-ts-tuned", "branch": "main", "path": "results/campaign/sella-2.6.0-pt-island-ts-tuned.dat", "cause": "sella_trust"},
    {"key": "tsopt/pt-island/pele", "template": "pele/pt-island-ts-ffd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-pt-island-ts-ffd.screen.jsonl", "kind": "screen"},
    {"key": "tsopt/pt-island/optim_ffd", "template": "optim/hef-ffd-ts-pt", "branch": "cells/optim/ts", "path": "results/campaign/optim-df8da599-hef-ffd-ts-pt.dat", "cause": "optim_pt_units"},
    # double-ended
    {"key": "double-ended-saddle-search/lj38/eon-ci-neb", "template": "eon-svn/de-lj38-ci-neb", "branch": "main", "path": "results/campaign/eon-svn-de-lj38-ci-neb.dat"},
    {"key": "double-ended-saddle-search/lj38/eon-ci-swdneb", "template": "eon-3.5.0/de-lj38-ci-swdneb", "branch": "main", "path": "results/campaign/eon-3.5.0-de-lj38-ci-swdneb.dat", "cause": "de_exclusion"},
    {"key": "double-ended-saddle-search/lj38/eon-dimer", "template": "eon-svn/de-lj38-dimer", "branch": "main", "path": "results/campaign/eon-svn-de-lj38-dimer.dat", "cause": "improved_unset"},
    {"key": "double-ended-saddle-search/lj38/eon-lanczos", "template": "eon-client/de-lj38-lanczos", "branch": "main", "path": "results/campaign/eon-client-de-lj38-lanczos.dat"},
    {"key": "double-ended-saddle-search/lj38/eon-lanczos-improved", "template": "eon-client/de-lj38-lanczos-improved", "branch": "main", "path": "results/campaign/eon-client-de-lj38-lanczos-improved.dat"},
    {"key": "double-ended-saddle-search/lj38/optim", "template": "optim/de-lj38", "branch": "cells/optim/saddle-pt-repro", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38--optim-de-lj38.dat", "cause": "optim_pes_counter"},
    {"key": "double-ended-saddle-search/lj38_2ts_separation/optim", "template": "optim/de-2ts", "branch": "cells/optim/saddle-pt-repro", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38_2ts_separation--optim-de-2ts.dat", "cause": "optim_pes_counter"},
    {"key": "double-ended-saddle-search/pt-island/eon-ci-neb", "template": "eon-svn/de-pt-ci-neb", "branch": "main", "path": "results/campaign/eon-svn-de-pt-ci-neb.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-ci-swdneb", "template": "eon-svn/de-pt-ci-swdneb", "branch": "main", "path": "results/campaign/eon-svn-de-pt-ci-swdneb.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-dimer", "template": "eon-svn/de-pt-dimer", "branch": "main", "path": "results/campaign/eon-svn-de-pt-dimer.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-lanczos", "template": "eon-svn/de-pt-lanczos", "branch": "main", "path": "results/campaign/eon-svn-de-pt-lanczos.dat", "cause": "pt_lanczos"},
    {"key": "double-ended-saddle-search/pt-island/eon-lanczos-improved", "template": "eon-svn/de-pt-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-de-pt-lanczos-improved.dat", "cause": "pt_lanczos"},
    {"key": "double-ended-saddle-search/pt-island/optim", "template": "optim/de-pt", "branch": "cells/optim/saddle-pt-repro", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-pt-island--optim-de-pt.dat", "cause": "optim_pt_units"},
    # neb per-image means already stored in the 3.5.0 files
    {"key": "neb/Pt-heptamer/eon-lbfgs", "template": "eon-lbfgs", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-lbfgs.dat"},
    {"key": "neb/Pt-heptamer/eon-cg", "template": "eon-cg", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-cg.dat", "cause": "heptamer_cg"},
    {"key": "neb/Pt-heptamer/eon-fire", "template": "eon-fire", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-fire.dat"},
    {"key": "neb/Pt-heptamer/eon-qm", "template": "eon-qm", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-qm.dat"},
    # saddle search
    {"key": "saddle-search/pt-island/eon-dimer", "template": "eon-svn/saddle-pt-dimer", "branch": "main", "path": "results/campaign/eon-3.5.0-saddle-pt-classic.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "cause": "eon_saddle_count"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50", "template": "softsaddle/saddle-davidson-sm50", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-davidson-sm50.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm50", "template": "softsaddle/saddle-lanczos-sm50", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-lanczos-sm50.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm0", "template": "softsaddle/saddle-lanczos-sm0", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-lanczos-sm0.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    # globalopt and htst and aladatom rates
    {"key": "globalopt/lj38/eon-basinhopping", "template": "eon-client/bh-lj38", "branch": "entries/eon-svn-lane-a", "path": "results/campaign/eon-client-bh-lj38.dat"},
    {"key": "globalopt/lj38/pele-basinhopping", "template": "pele/lj38-basinhopping", "branch": "cells/pele/go", "path": "results/campaign/pele-b7081b93-lj38-basinhopping.dat", "cause": "pele_bh_defaults"},
    {"key": "globalopt/lj38/gmin-nosym", "template": "gmin-paper/lj38-nosym", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj38-nosym.dat"},
    {"key": "globalopt/lj38/gmin-csm", "template": "gmin/lj38-csm", "branch": "cells/gmin/go", "path": "results/campaign/gmin-df8da599-lj38-csm.dat", "cause": "gmin_csm", "ranked_path": "results/campaign/gmin-df8da599-lj38-csm.dat", "ranked_branch": "cells/gmin/go"},
    {"key": "globalopt/lj38/gmin-symmetrise", "template": "gmin-paper/lj38-symmetrise", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj38-symmetrise.dat"},
    {"key": "globalopt/lj75/gmin", "template": "gmin-paper/lj75", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj75.dat"},
    {"key": "globalopt/lj98/gmin-nosym", "template": "gmin-paper/lj98-nosym", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj98-nosym.dat", "metrics": ["energy_calls_mean", "basin_hopping_steps_mean"], "cause": "gmin_nosym_calls"},
    {"key": "globalopt/lj98/gmin-symmetrise", "template": "gmin-paper/lj98-symmetrise", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj98-symmetrise.dat", "metrics": ["energy_calls_mean", "basin_hopping_steps_mean"], "cause": "gmin_quench_count", "ranked_path": "results/campaign/gmin-02aaf45245af-lj98-bh.dat", "ranked_branch": "cells/gmin/go"},
    {"key": "globalopt/lj150/gmin-nosym", "template": "gmin-paper/lj150-nosym", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj150-nosym.dat", "metrics": ["energy_calls_mean", "basin_hopping_steps_mean"], "cause": "gmin_quench_count"},
    {"key": "globalopt/lj150/gmin-symmetrise", "template": "gmin-paper/lj150-symmetrise", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-lj150-symmetrise.dat", "metrics": ["energy_calls_mean", "basin_hopping_steps_mean"], "cause": "gmin_quench_count"},
    {"key": "globalopt/tip4p10/gmin", "template": "gmin-paper/tip4p10", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p10.dat"},
    {"key": "globalopt/tip4p10/gmin-tbp", "template": "gmin-paper/tip4p10-tbp", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p10-tbp.dat"},
    {"key": "globalopt/tip4p15/gmin", "template": "gmin-paper/tip4p15", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p15.dat", "cause": "gmin_tip4p15"},
    {"key": "globalopt/tip4p15/gmin-tbp", "template": "gmin-paper/tip4p15-tbp", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p15-tbp.dat"},
    {"key": "globalopt/tip4p20/gmin", "template": "gmin-paper/tip4p20", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p20.dat", "cause": "gmin_tip4p20"},
    {"key": "globalopt/tip4p20/gmin-tbp", "template": "gmin-paper/tip4p20-tbp", "branch": "cells/gmin/go-paper", "path": "results/campaign/gmin-paper-tip4p20-tbp.dat"},
    {"key": "hTST/Al/eon-dimer-cg", "template": "eon-svn/htst-al-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-al-dimer.dat", "kind": "htst", "cause": "htst_al"},
    {"key": "hTST/Pt/eon-dimer-cg", "template": "eon-svn/htst-pt-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-pt-dimer.dat", "kind": "htst"},
]


CAUSE_TEXT = {
    "lj38_ts_starts": (
        "The 2014-stop mean is 207.51 against the archive 192. "
        "Archive run.sh unpacks coords-xyz.tar.gz (200 xyz, max pair about 4.4). "
        "The harness problem points at coords-con.tar.gz (max pair about 37). "
        "Kabsch RMS is about 6.33, so the sets are not the same geometry. "
        "Neither set has a pair past cutoff 40, and both drivers use RMS tolerance 1e-3/sqrt(114). "
        "The starting set is the gap. The archive row is this 2014-stop mean. The harness-stop mean is 254.48."
    ),
    "lj38_ts_starts_cd": (
        "The 2014-stop mean is 263.52 against the archive 254. The same xyz-versus-con starting set as ts-ffd. "
        "findTS_local.py for the CD row does not pass orthogZeroEigs; the harness template sets orthog translations-rotations. "
        "The archive row is this 2014-stop mean. The harness-stop mean is 335.06."
    ),
    "optim_lj38_ts": (
        "Archive run.sh unpacks coords-xyz.tar.gz. The harness uses the problem coords-con set (Kabsch RMS about 6.33). "
        "The odata preamble is not in the archive snapshot. The template uses updates 4. The mean is 489 against 145."
    ),
    "pt_lanczos": (
        "The svn template uses the archived Lanczos keys (10 iterations, tolerance 0.05) and shifted Morse::force. "
        "Dimer and climbing-image on that kernel match the archive counts. Lanczos does not, and the failure count is 0 against 5 or 8."
    ),
    "ase_morse_script": (
        "ASE 3.29 with the archived Morse script (L-BFGS memory 20, maxstep 0.2, alpha 25; FIRE maxstep 0.2; PreconLBFGS Exp A 3) "
        "scores 32, 113, and 29 against 53.8, 147.47, and 34. "
        "Constructor-default L-BFGS scores 53, next to 53.8, but that is not the archived alpha and memory. "
        "eOn Morse L-BFGS and FIRE on the same bulk file match the archive, so the geometry set is not the split."
    ),
    "optim_pt_units": (
        "archive run.sh divides coordinates by r0 = 2.897 and scales rho and rcut into those units. "
        "The harness keeps the problem Morse in eV/Angstrom on the unscaled con file."
    ),
    "optim_pes_counter": (
        "The template note records the 2014 mean as OPTIM's own energy-plus-gradient calls. "
        "The harness score is pes.evaluate on the harness potential."
    ),
    "optim_morse_memory": (
        "comments.txt sets L-BFGS memory 100 and maximum move 2.0. That run scores 27. "
        "OPTIM defaults (updates 4) score 46, which is the archived integer, but that is not the comment's setup. "
        "The odata preamble was not archived."
    ),
    "optim_rr_defaults": (
        "The Rayleigh-Ritz template uses updates 4. The archived first-order mean is 25 and the second-order mean is 46. "
        "This default scores 61, so it is not either archived order."
    ),
    "pele_fire": "pele FIRE on LJ38 scores 1264 with 7 failures. The archive is 709 with 0 failures. The M100 L-BFGS row on the same build matches the archive, so this is the FIRE step, not the LJ kernel.",
    "pele_morse": "pele Morse L-BFGS scores 92. The archive is 51. eOn Morse L-BFGS on the same bulk file matches the archive at 52, so the gap is the pele memory and step, not the shared Morse parameters.",
    "morse_sd": "eOn Morse steepest descent scores 7038 with 19 failures. The archive is 196 with 0. Morse L-BFGS, CG, and FIRE on the same client match the archive, so the kernel is not the split.",
    "improved_unset": "The r2549 template turns improved on. The 2014 ini left that key unset. The means differ.",
    "sella_trust": "Sella 2.6.0 with the archived trust-region numbers (eta, gamma, delta0, rho, sigma) is the parameter_source=archive row. The mean differs from the 2014 row. The constructor-default row is a different setup.",
    "heptamer_cg": "The 3.5.0 file already stores the per-image mean, 142 against the archive 120. L-BFGS, FIRE, and QM on the same 8-image spring-5 climb match the archive, so the image count is not the split.",
    "de_exclusion": "The 3.5.0 doubly-nudged file is the per-band mean under the 2500-call exclusion (708, 5 failures). The archive mean is 861. The client total is a different counter and differs as well.",
    "eon_saddle_count": (
        "The classic dimer campaign (improved false, the 2014 key) has fraction_connected 0.692 against 0.746, "
        "force calls 6.97e6 against 3.78e6, and unique saddles 91.7 against 109.2. "
        "The archive dispersion on the force calls is 5.27e5. "
        "This eOn AKMC counter is not the harness RMSD link test."
    ),
    "saddle_novelty": (
        "The harness link and novelty test uses Cartesian RMSD 0.5 over the full island. "
        "Distinct saddles collapse: benchmark_unique_saddles is 0, 0.7, and 0.5 against the archive 115, 143, and 130. "
        "benchmark_fraction_connected stays near 1 while the v2 connected fraction is about 0. "
        "The per-search cap of 5064 then puts every repetition on the PAR-2 cap."
    ),
    "pele_bh_defaults": (
        "The archived driver is the dangling path bin/js850/globalopt.py. "
        "The harness uses pele LJCluster defaults (temperature 1, step 0.6, LBFGS memory 4) on unshifted cutoff 40. "
        "The mean is 688720 against 521595."
    ),
    "htst_al": "The r2549 Al dimer with the archived rotation string (CG, max 50, tol 1) differs on barrier_force_calls and rate_force_calls by about 30 percent. Pt dimer CG with max 10 matches all four hTST counts.",
    "akmc_vs_langevin": (
        "The archive rows are Langevin parallel replica. The harness rows are AKMC. "
        "On the aladatom campaign, eon/aladatom-akmc true_rate 3.597591452e9 is 14.1 times 2.542941e8, "
        "and eon/standard-aladatom-akmc true_rate 3.80613e7 is 45.5 times 8.359e5."
    ),
    "gmin_csm": (
        "The 06 Aug 2014 file names code wales-27058. That binary is not published. "
        "GMIN df8da599 has SYMMETRISECSM, so the ranked row runs on df8da599. "
        "The deck is the archive data file: TEMPERATURE 3.0, NEWRESTART 30, AVOID 1.25 18, "
        "STEP 0.43 0.52, SLOPPYCONV 0.002, PERMDIST 0.005, MAXIT 1000 1000, STEPS 90000000 1.0, "
        "SAVE 1, SORT, and SYMMETRISECSM 1 CI 1.0D-6 30 1 50. RANSEED is unset. "
        "The cap and MAX_NPCALL are problems/globalopt/lj38.toml convergence.max_force_calls, 2000000."
    ),
    "gmin_quench_count": (
        "The 03 Oct 2022 files and the serial df8da599 run use the same data file and SDPRND seed 0. "
        "Calls per quench match. The run takes more quenches, so the energy-call mean and the step mean move together. "
        "nfailed is 0. The archived lj98 and lj150 run.sh loops to 200 and names a directory that is not in the tarball. "
        "The tarball has 100 numbered coords files, which is the sample. "
        "The lj98 symmetrise harness rows are GMIN df8da599 on that deck."
    ),
    "gmin_tip4p15": (
        "tip4p10 basin-hopping and temperature basin-paving, and tip4p15 temperature basin-paving, "
        "are close to the archive on this serial binary. tip4p15 basin-hopping is 580721 against 670645. "
        "The data file is the archive file, including MAXIT 1.D4 and STEPS 1.D7, and RANSEED is 100."
    ),
    "gmin_tip4p20": (
        "tip4p20 temperature basin-paving is 28089000 against 25566700. "
        "Basin-hopping is 32513200 against 28596700. The median is 19448700 against 19467100. "
        "Start 32 hits at 317150614 calls. The 09 May 2014 maximum is 141910000. nfailed is 0. "
        "The data file is the archive file, including STEP 0 0 1.41, STEPS 4.D7, RADIUS 1.D1, and RANSEED 100."
    ),
    "gmin_nosym_calls": (
        "The 03 Oct 2022 nosym file and the serial df8da599 run use the same data file: STEPS 10000000 1.0, "
        "TEMPERATURE 0.44, NEWRESTART 200, AVOID 3.0 20, SAVE 5, and no RANSEED. nfailed is 0. "
        "Quenches are 49658 against 43820. Function calls are 6791676 against 59885724. "
        "Calls per quench are 137 against 1367. Symmetrise on this serial binary keeps about 195 calls per quench."
    ),
}


def row_id(row: dict) -> str:
    return f"{row['suite']}/{row['problem']}/{row['entry']}"


LOCAL_BRANCHES = {"main", "cells/gmin/go", "cells/gmin/go-paper"}
FINDINGS_PATH = "oracle/findings.jsonl"
FORBIDDEN = (
    "reproduced",
    "reproduce",
    "reproduction ledger",
    "Wilson",
    "10% band",
    "10 percent",
    "standard error",
    "standard-error",
    "not_rerun",
    "not rerun",
    "legacy",
    "paper baseline",
    "paper_baseline",
)


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").split()[0])
    except ValueError:
        return None


def parse_dat(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2 and parts[0] not in fields:
            fields[parts[0]] = parts[1].strip()
    return fields


def metrics_for(fields: dict[str, str], kind: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    if kind == "htst":
        return ["barrier_force_calls", "barrier_jobs", "rate_force_calls", "rate_jobs"]
    if "true_rate" in fields:
        return ["true_rate"]
    if "avg_energy" in fields:
        return ["avg_energy", "min_energy", "max_energy"]
    if "mean_epoch" in fields:
        return ["mean_epoch"]
    if "barrier_force_calls" in fields:
        return ["barrier_force_calls", "barrier_jobs", "rate_force_calls", "rate_jobs"]
    return ["force_calls"]


def collect_values(
    archived_fields: dict[str, str],
    campaign_fields: dict[str, str],
    names: list[str],
    value_from: dict[str, str] | None,
) -> list[dict]:
    values = []
    renaming = value_from or {}
    for name in names:
        archive = _float(archived_fields.get(name))
        source = renaming.get(name, name)
        harness = _float(campaign_fields.get(source))
        if archive is None or harness is None:
            continue
        values.append({"metric": name, "archive": archive, "harness": harness})
    return values


def screen_fields(text: str) -> dict[str, str]:
    calls = []
    harness = []
    failed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        calls.append(float(row["stop2014_calls"]))
        harness.append(float(row["force_calls"]))
        if not row.get("stop2014_success", False):
            failed += 1
    count = len(calls)
    mean = sum(calls) / count
    harness_mean = sum(harness) / count
    return {
        "force_calls": f"{mean:.6g}",
        "nfailed": str(failed),
        "n": str(count),
        "harness_stop_mean": f"{harness_mean:.6g}",
        "metric": "stop2014_calls",
    }


def read_campaign(root: Path, branch: str, path: str) -> str:
    local = root / path
    if branch in LOCAL_BRANCHES and local.is_file():
        return local.read_text(encoding="utf-8")
    if branch == "main":
        return local.read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"origin/{branch}:{path}"],
        cwd=root,
        text=True,
    )


def _sha(branch: str) -> str:
    if branch == "cells/pele/ts" and "cells/pele/ts" not in SHA:
        SHA[branch] = subprocess.check_output(
            ["git", "rev-parse", "origin/cells/pele/ts"],
            text=True,
        ).strip()
    return SHA[branch]


def load_ledger(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _local_fields(root: Path, beside: dict) -> dict[str, str] | None:
    path = root / str(beside.get("path") or "")
    branch = str(beside.get("branch") or "")
    if path.is_file() and branch in LOCAL_BRANCHES and beside.get("kind", "dat") == "dat":
        return parse_dat(path.read_text(encoding="utf-8"))
    return None


def campaign_mismatch(root: Path) -> list[str]:
    """Stored harness numbers must equal the campaign file on disk."""
    errors = []
    for row in load_ledger(root / FINDINGS_PATH):
        beside = row.get("beside") or {}
        if not beside:
            continue
        fields = _local_fields(root, beside)
        if fields is None:
            continue
        ident = row_id(row)
        for item in row.get("values") or []:
            live = _float(fields.get(item["metric"]))
            if live is None:
                continue
            if live != item["harness"]:
                errors.append(
                    f"{ident} {item['metric']} stored {item['harness']} and the campaign file has {live}"
                )
    return errors


def _git_blob_size(root: Path, path: Path) -> int | None:
    rel = path.relative_to(root).as_posix()
    try:
        out = subprocess.check_output(
            ["git", "cat-file", "-s", f":{rel}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out.strip())


def _lfs_tracked(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    try:
        out = subprocess.check_output(
            ["git", "check-attr", "filter", "--", rel],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out.strip().endswith("filter: lfs")


def large_endpoint_files(root: Path, limit: int = 1_000_000) -> list[str]:
    """Endpoint archives stored as git blobs instead of an LFS pointer.

    The committed size is the blob in the index. A path matched by
    git-LFS is allowed while it is still untracked, because ``git add``
    stores the pointer. A loose blob over ``limit`` fails.
    """
    errors = []
    campaign = root / "results" / "campaign"
    if not campaign.is_dir():
        return errors
    for path in sorted(campaign.rglob("*.npz")):
        if "endpoint" not in path.name:
            continue
        committed = _git_blob_size(root, path)
        if committed is not None:
            size = committed
        elif _lfs_tracked(root, path):
            continue
        else:
            size = path.stat().st_size
        if size > limit:
            errors.append(f"{path.relative_to(root)} is {size} bytes")
    return errors


def _forbidden(text: str) -> list[str]:
    found = []
    for phrase in FORBIDDEN:
        if phrase in text:
            found.append(phrase)
    return found


def findings_errors(root: Path) -> list[str]:
    ledger_path = root / FINDINGS_PATH
    if not ledger_path.is_file():
        return [f"{FINDINGS_PATH} is missing"]
    from .check import held_back_errors, load_oracle

    oracle = load_oracle(root / "oracle" / "entries.jsonl")
    stored = load_ledger(ledger_path)
    errors = []
    if len(stored) != len(oracle):
        errors.append(f"findings have {len(stored)} rows and the oracle has {len(oracle)}")
    if [row_id(row) for row in stored] != [row_id(row) for row in oracle]:
        errors.append("findings are not in oracle order")
    fresh = build_rows(root)
    for left, right in zip(stored, fresh):
        if left != right:
            errors.append(f"{row_id(left)} stored finding does not match the campaign files")
            break
    errors.extend(campaign_mismatch(root))
    errors.extend(held_back_errors(root))
    blob = ledger_path.read_text(encoding="utf-8")
    for phrase in _forbidden(blob):
        errors.append(f"findings contain {phrase}")
    note = root / "docs" / "findings.md"
    if note.is_file():
        for phrase in _forbidden(note.read_text(encoding="utf-8")):
            errors.append(f"docs/findings.md contains {phrase}")
    else:
        errors.append("docs/findings.md is missing")
    errors.extend(large_endpoint_files(root))
    return errors


def _gap(values: list[dict], failure: dict | None) -> str:
    parts = []
    for item in values:
        archive = item["archive"]
        harness = item["harness"]
        if archive == harness:
            continue
        if archive:
            relative = abs(harness - archive) / abs(archive)
            parts.append(f"{item['metric']} {harness:g} vs {archive:g} ({relative:.1%})")
        else:
            parts.append(f"{item['metric']} {harness:g} vs {archive:g}")
    if failure and failure["archive"] != failure["harness"]:
        parts.append(f"nfailed {failure['harness']:g} vs {failure['archive']:g}")
    return "; ".join(parts)


def _drop_none(row: dict) -> dict:
    return {key: value for key, value in row.items() if value is not None}


def _ranked_block(text: str) -> dict | None:
    fields = parse_dat(text)
    if "par2_mean" not in fields and "n" not in fields:
        return None
    n = _float(fields.get("n"))
    failed = _float(fields.get("nfailed") or fields.get("v2_nfailed"))
    successes = None if n is None or failed is None else n - failed
    code = fields.get("code", "")
    version = fields.get("code_version", "")
    block = _drop_none(
        {
            "label": f"{code} {version}".strip(),
            "n": n,
            "successes": successes,
            "nfailed": failed,
            "par2_mean": _float(fields.get("par2_mean")),
            "mean": _float(fields.get("force_calls")),
            "median": _float(fields.get("median") or fields.get("par2_median")),
            "min": _float(fields.get("force_calls_min")),
            "max": _float(fields.get("force_calls_max")),
            "cap": _float(fields.get("cap")),
        }
    )
    return block or None


def build_rows(root: Path) -> list[dict]:
    from .check import load_oracle

    oracle = {row_id(row): row for row in load_oracle(root / "oracle" / "entries.jsonl")}
    matches = {item["key"]: item for item in MATCHES}
    built = []
    for ident, entry in oracle.items():
        fields = entry["fields"]
        match = matches.get(ident)
        sample = SAMPLE_N.get(f"{entry['suite']}/{entry['problem']}")
        record = {
            "suite": entry["suite"],
            "problem": entry["problem"],
            "entry": entry["entry"],
            "archive_path": entry["path"],
            "commit": entry["commit"],
            "in_paper": ident in PAPER,
            "source": "softsaddle-2019" if entry["entry"].startswith("adaptive_HSphere") else ("paper" if ident in PAPER else "archive"),
            "archived_setup": {
                "code": fields.get("code", ""),
                "algorithm": fields.get("algorithm", ""),
                "code_version": fields.get("code_version", ""),
            },
            "beside": None,
            "values": [],
            "failure_counts": None,
            "gap": "",
            "finding_id": "",
            "finding": "",
        }
        paper = PAPER_METRIC.get(ident)
        if paper:
            record["table_metric"] = {"name": paper[0], "value": paper[1]}
        if match is None:
            if ident.startswith("accelerated-md/"):
                record["finding_id"] = "akmc_vs_langevin"
                record["finding"] = CAUSE_TEXT["akmc_vs_langevin"]
            built.append(record)
            continue
        kind = match.get("kind", "dat")
        try:
            text = read_campaign(root, match["branch"], match["path"])
        except (OSError, subprocess.CalledProcessError):
            named = match.get("cause", "")
            if named:
                record["finding_id"] = named
                record["finding"] = CAUSE_TEXT.get(named, "")
            built.append(record)
            continue
        campaign_fields = screen_fields(text) if kind == "screen" else parse_dat(text)
        names = metrics_for(fields, kind, match.get("metrics"))
        values = collect_values(fields, campaign_fields, names, match.get("value_from"))
        failure = None
        archive_failures = _float(fields.get("nfailed"))
        campaign_failures = _float(campaign_fields.get("nfailed"))
        if archive_failures is not None and campaign_failures is not None:
            failure = {"archive": archive_failures, "harness": campaign_failures}
            if sample:
                failure["sample"] = sample
        named = match.get("cause", "")
        record["values"] = values
        record["failure_counts"] = failure
        record["gap"] = _gap(values, failure)
        record["finding_id"] = named
        record["finding"] = CAUSE_TEXT.get(named, "") if named else ""
        beside = {
            "template": match["template"],
            "branch": match["branch"],
            "sha": _sha(match["branch"]),
            "path": match["path"],
            "kind": kind,
            "metric": campaign_fields.get("metric", names[0] if names else "force_calls"),
        }
        if kind == "screen":
            beside["stop2014_calls"] = _float(campaign_fields.get("force_calls"))
            beside["harness_stop_mean"] = _float(campaign_fields.get("harness_stop_mean"))
            beside["metric"] = "stop2014_calls"
        if match.get("extra"):
            beside.update(match["extra"])
        ranked_path = match.get("ranked_path")
        if ranked_path:
            try:
                ranked_text = read_campaign(root, match.get("ranked_branch", match["branch"]), ranked_path)
            except (OSError, subprocess.CalledProcessError):
                ranked_text = ""
            ranked = _ranked_block(ranked_text) if ranked_text else None
            if ranked:
                ranked["path"] = ranked_path
                problem_file = root / "problems" / entry["suite"] / f"{entry['problem']}.toml"
                stated_cap = None
                if problem_file.is_file():
                    import tomllib

                    stated = tomllib.loads(problem_file.read_text(encoding="utf-8"))
                    stated_cap = (stated.get("convergence") or {}).get("max_force_calls")
                if stated_cap not in (None, ""):
                    ranked["cap_source"] = (
                        f"problems/{entry['suite']}/{entry['problem']}.toml convergence.max_force_calls"
                    )
                else:
                    ranked["cap_source"] = (
                        f"problems/{entry['suite']}/{entry['problem']}.toml states no max_force_calls; "
                        f"template failure_cap {int(ranked['cap']) if ranked.get('cap') is not None else ''}"
                    )
                beside["ranked"] = ranked
        record["beside"] = beside
        built.append(record)
    return built


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.6g}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def render_markdown(rows: list[dict]) -> str:
    with_file = sum(1 for row in rows if row.get("beside"))
    without_file = len(rows) - with_file
    lines = [
        "# Findings",
        "",
        "Archive commit `0f26b796`. The archived numbers are the archived rows. A campaign file beside a row records both numbers and the gap. There is no pass/fail window.",
        "",
        f"{with_file} archive rows have a campaign file beside them. {without_file} archive rows have no campaign file. {len(rows)} archive rows.",
        "",
        "The pele transition-state archive score is the 2014 stop: break at the first RMS below tolerance, then one eigenvector search, transverse screen calls subtracted. Pt-island ts-ffd is 88.55 against 88. The ranked harness-stop means are 111.27, 254.48, and 335.06 and are a different stop.",
        "",
        "OPTIM LJ38 L-BFGS tuned: the archived integer is 176 and the unrounded harness mean is 175.537.",
        "",
        "EON FIRE in JCTC Table 1 is 645. The archive file and the later run are both 731.",
        "",
        "Table 10 prints EON Lanczos 165. The archive file is 325. CI-NEB(3) prints 3 failures; the archive file has 0.",
        "",
        "Endpoint minima are free-atom float32 deltas against the reactant, compressed. A file over 1 MB is a git-LFS pointer, not a loose blob.",
        "",
        "The 2014 GMIN CSM binary wales-27058 is not published. GMIN df8da599 has SYMMETRISECSM, so that row runs on df8da599.",
        "",
    ]
    previous = ""
    for row in rows:
        suite = row["suite"]
        if suite != previous:
            if previous:
                lines.append("")
            previous = suite
            lines.append(f"## {suite}")
            lines.append("")
            lines.append("| Row | Archive | Campaign | Gap |")
            lines.append("| --- | --- | --- | --- |")
        ident = f"{row['problem']}/{row['entry']}"
        if row["values"]:
            archive = ", ".join(f"{item['metric']} {_fmt(item['archive'])}" for item in row["values"])
            campaign = ", ".join(f"{_fmt(item['harness'])}" for item in row["values"])
        else:
            archive = "—"
            campaign = "—"
        gap = row.get("gap") or ""
        if row.get("finding_id"):
            gap = f"{gap}; {row['finding_id']}" if gap else row["finding_id"]
        lines.append(f"| {ident} | {archive} | {campaign} | {gap} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for row in rows:
        if row.get("finding"):
            lines.append(f"- {row['suite']}/{row['problem']}/{row['entry']}: {row['finding']}")
    lines.append("")
    return "\n".join(lines)


def write_ledger(root: Path) -> list[dict]:
    rows = build_rows(root)
    path = root / FINDINGS_PATH
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    (root / "docs" / "findings.md").write_text(render_markdown(rows), encoding="utf-8")
    old = root / "oracle" / "reproduction.jsonl"
    if old.is_file():
        old.unlink()
    old_doc = root / "docs" / "reproduction.md"
    if old_doc.is_file():
        old_doc.unlink()
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = write_ledger(root)
    with_file = sum(1 for row in rows if row.get("beside"))
    print({"beside": with_file, "archive_only": len(rows) - with_file, "rows": len(rows)})
    for row in rows:
        if row.get("gap"):
            print(f"{row_id(row)} {row.get('finding_id', '')} {row['gap']}")


if __name__ == "__main__":
    main()
