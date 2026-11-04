#!/usr/bin/env bash
# Build libgminlbfgs.so from the Wales svn.tar.bz2.
# The objects are source/mymylbfgs.F90 and source/blas_wrappers.F90.
# stubs.f90 and driver.f90 are the OptBench callback.
# The upstream URL rolls daily, and this script fetches the latest snapshot.
# WALES_TARBALL names a local copy to read instead.
# driver.f90 includes svnrev.inc, which holds the extracted GMIN/source/SVNREV
# and the tarball sha256, so gmin_revision() and gmin_tarball_sha256() name
# the bytes this script compiled.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source "$here/../checkout.sh"
prefix=${PREFIX:-/opt/gmin}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tarball=$(wales_tarball "$work")
sha=$(wales_sha256 "$tarball")
tar -xjf "$tarball" -C "$work" \
  ./wales/GMIN/source/mymylbfgs.F90 \
  ./wales/GMIN/source/blas_wrappers.F90 \
  ./wales/GMIN/source/SVNREV
got=$(wales_svnrev "$work" GMIN)

cp "$here/stubs.f90" "$here/driver.f90" "$work/"
{
  printf '   character(len=*), parameter :: svnrev = "%s"\n' "$got"
  printf '   character(len=*), parameter :: tarball_sha256 = "%s"\n' "$sha"
} > "$work/svnrev.inc"
flags=(-fPIC -O2 -fmax-stack-var-size=32768 -ffree-line-length-none)
(
  cd "$work"
  "$FC" -c "${flags[@]}" -std=f2008 stubs.f90
  "$FC" -c "${flags[@]}" wales/GMIN/source/blas_wrappers.F90
  "$FC" -c "${flags[@]}" wales/GMIN/source/mymylbfgs.F90
  "$FC" -c "${flags[@]}" driver.f90
  "$FC" -shared -Wl,--no-undefined -o libgminlbfgs.so \
    stubs.o blas_wrappers.o mymylbfgs.o driver.o -lblas
)
cp "$work/libgminlbfgs.so" "$here/libgminlbfgs.so"
if mkdir -p "$prefix/lib" 2>/dev/null && cp "$here/libgminlbfgs.so" "$prefix/lib/libgminlbfgs.so"; then
  echo "installed $prefix/lib/libgminlbfgs.so"
else
  sudo mkdir -p "$prefix/lib"
  sudo cp "$here/libgminlbfgs.so" "$prefix/lib/libgminlbfgs.so"
  echo "installed $prefix/lib/libgminlbfgs.so"
fi
echo "libgminlbfgs.so is svn.tar.bz2 sha256 $sha, SVNREV $got"
