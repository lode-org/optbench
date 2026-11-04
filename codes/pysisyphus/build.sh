#!/usr/bin/env bash
# Install pysisyphus at the registry pin into its own venv. The runner puts
# that venv's site-packages first on sys.path (PYSIS_SITE). The clone has no
# tag, so the installed package reports 0.1.dev1.
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
src=${PYSIS_SRC:-$(ob_root)/src/pysisyphus}
venv=${PYSIS_VENV:-$(ob_root)/venv-pysis}
checkout pysisyphus "$src"
"${HARNESS_PYTHON:-python3}" -m venv "$venv"
read -r -a keep <<< "$(harness_pins)"
"$venv/bin/pip" install -q "${keep[@]}" "$src"
"$venv/bin/python" -c 'import pysisyphus'
