#!/usr/bin/env bash
# Build libartn.so at the registry pin. The runner loads lib/libartn.so and
# interface/pypARTn.py from this tree (ARTN_LIB and ARTN_INTERFACE).
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
dest=${ARTN_SRC:-$(ob_root)/src/artn}
checkout artn "$dest"
cd "$dest"
./configure
make lib
test -f lib/libartn.so
test -f interface/pypARTn.py
