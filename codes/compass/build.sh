#!/usr/bin/env bash
# Check out COMPASS at the registry pin. compass.py is imported from the
# checkout. ASE 3.29 moved the cell filters to ase.filters; the runner
# aliases them back in optbench.engines.compass_search._import_compass.
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
dest=${COMPASS_SRC:-$(ob_root)/src/compass}
checkout compass "$dest"
test -f "$dest/compass.py"
