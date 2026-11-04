#!/usr/bin/env bash
# Build the OPTIM target from the Wales svn.tar.bz2.
# usage: build.sh [DEST] [BUILD]
# OPTIM/source/CMakeLists.txt compiles the ASE socket in
# INTERFACES/pysocket, which optbench.engines.optim_search drives.
# The upstream URL rolls daily, and this script fetches the latest snapshot
# into DEST. WALES_TARBALL names a local copy to read instead.
# OPTIM does not print the SVNREV it was built from, so this script writes
# the tree's OPTIM/source/SVNREV and the tarball sha256 beside the binary.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source "$here/../checkout.sh"
dest=${1:-$(ob_root)/src/wales-full}
build=${2:-$dest/build-optim}
tarball=$(wales_tarball "$dest")
sha=$(wales_sha256 "$tarball")
wales_extract "$tarball" "$dest"
revision=$(wales_svnrev "$dest" OPTIM)

cmake -S "$dest/wales/OPTIM/source" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILER_SWITCH=gfortran \
  -DWITH_ALIGN=OFF \
  -DWITH_OPENMP=OFF \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_Fortran_COMPILER="$FC"
cmake --build "$build" -j"$(nproc)" --target OPTIM
echo "$revision" > "$build/SVNREV"
wales_stamp "$build" "$sha"
echo "built $build/OPTIM from svn.tar.bz2 sha256 $sha, SVNREV $revision"
