#!/usr/bin/env bash
# Build liblammps.so at the registry pin with the packages the minimizer and
# NEB runners load. The Python module is python/lammps in the same tree
# (LAMMPS_LIB and LAMMPS_PYTHON).
set -euo pipefail

source "$(cd "$(dirname "$0")/.." && pwd)/checkout.sh"
dest=${LAMMPS_SRC:-$(ob_root)/src/lammps}
checkout lammps "$dest"
cmake -S "$dest/cmake" -B "$dest/build" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="$CC" \
  -D CMAKE_CXX_COMPILER="$CXX" \
  -D CMAKE_Fortran_COMPILER="$FC" \
  -D BUILD_SHARED_LIBS=on \
  -D BUILD_MPI=off \
  -D BUILD_OMP=off \
  -D PKG_PYTHON=on \
  -D PKG_EXTRA-PAIR=on \
  -D PKG_MANYBODY=on \
  -D PKG_REPLICA=on \
  -D PKG_PHONON=on \
  -D Python_EXECUTABLE="$(command -v "${HARNESS_PYTHON:-python3}")"
cmake --build "$dest/build" -j"$(nproc)"
test -f "$dest/build/liblammps.so"
