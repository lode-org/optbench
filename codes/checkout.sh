# Sourced by codes/*/build.sh.
#
# checkout <code id> <dest> fetches the registry commit of <code id> into
# <dest> and refuses any other HEAD.
#
# The image default cc and c++ can be a clang that finds no libstdc++
# ("cannot find -lstdc++"). Every C and C++ build here names the GNU
# compilers, and Fortran builds use gfortran.
export CC=gcc
export CXX=g++
export FC=gfortran
export F90=gfortran
export F77=gfortran

ob_root() {
  if [ -n "${TMPDIR-}" ]; then
    printf '%s' "$TMPDIR"
  elif [ -n "${TMP-}" ]; then
    printf '%s' "$TMP"
  elif [ -n "${TEMP-}" ]; then
    printf '%s' "$TEMP"
  else
    python3 -c 'import tempfile; print(tempfile.gettempdir(), end="")'
  fi
}

codes_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pin() {
  python3 "$codes_dir/pin.py" "$1" "$2"
}

checkout() {
  local code=$1 dest=$2 url commit
  url=$(pin "$code" repository)
  commit=$(pin "$code" commit)
  if [[ -d "$dest/.git" ]] && [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]]; then
    echo "$dest is $commit"
    return
  fi
  rm -rf "$dest"
  git init -q "$dest"
  git -C "$dest" remote add origin "$url"
  git -C "$dest" fetch -q --depth 1 origin "$commit"
  git -C "$dest" checkout -q FETCH_HEAD
  if [[ "$(git -C "$dest" rev-parse HEAD)" != "$commit" ]]; then
    echo "$dest HEAD is not $commit" >&2
    return 1
  fi
  echo "$dest is $commit"
}

# The venv site-packages go on the harness sys.path, so the venv keeps the
# numpy and scipy the harness already imported.
harness_pins() {
  "${HARNESS_PYTHON:-python3}" -c 'import numpy, scipy; print(f"numpy=={numpy.__version__} scipy=={scipy.__version__}")'
}
