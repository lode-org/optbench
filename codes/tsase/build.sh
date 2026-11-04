#!/usr/bin/env bash
# Check out TSASE at the registry pin. The runner loads tsase/neb and the
# solid-state dimer from the checkout. The Fortran calculators are not built.
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
dest=${TSASE_SRC:-$(ob_root)/src/tsase}
checkout tsase "$dest"
test -f "$dest/tsase/neb/ssneb.py"
