#!/usr/bin/env bash
# Install libdlfind at the registry pin into its own venv. The runner puts
# that venv's site-packages on sys.path (DLFIND_SITE). The Fortran library is
# built by scikit-build with the compilers named in checkout.sh.
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
src=${DLFIND_SRC:-$(ob_root)/src/libdlfind}
venv=${DLFIND_VENV:-$(ob_root)/venv-dlfind}
checkout dlfind "$src"
"${HARNESS_PYTHON:-python3}" -m venv "$venv"
read -r -a keep <<< "$(harness_pins)"
"$venv/bin/pip" install -q "${keep[@]}" "$src"
"$venv/bin/python" -c 'import libdlfind'
