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

# The gmin row URL is a Wales svn.tar.bz2 snapshot that rolls daily, and no
# sha256 is pinned. wales_tarball <dir> fetches whatever the URL serves to
# <dir>/svn.tar.bz2 and prints that path. WALES_TARBALL names a local copy to
# read instead, and then nothing is fetched. Either way the tarball sha256 goes
# to stderr, and each build script writes it beside what it builds.
wales_tarball() {
  local tarball
  if [[ -n "${WALES_TARBALL:-}" ]]; then
    tarball=$WALES_TARBALL
    if [[ ! -f "$tarball" ]]; then
      echo "WALES_TARBALL=$tarball is not a file" >&2
      return 1
    fi
    echo "svn.tar.bz2 read from WALES_TARBALL" >&2
  else
    mkdir -p "$1" || return 1
    tarball=$1/svn.tar.bz2
    curl -fsSL -o "$tarball" "$(pin gmin repository)" || return 1
    echo "svn.tar.bz2 fetched from the gmin row URL" >&2
  fi
  echo "svn.tar.bz2 sha256 $(wales_sha256 "$tarball")" >&2
  echo "$tarball"
}

wales_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

# wales_stamp <dir> <sha256> writes <dir>/svn.tar.bz2.sha256.
wales_stamp() {
  printf '%s  svn.tar.bz2\n' "$2" > "$1/svn.tar.bz2.sha256"
}

# wales_extract <tarball> <dest> extracts <tarball> into <dest> and stamps
# <dest> with its sha256. A <dest> stamped with that sha256 is reused, and any
# other <dest>/wales is removed first, so one tree never mixes two snapshots.
wales_extract() {
  local sha
  sha=$(wales_sha256 "$1")
  if [[ -d "$2/wales" && -f "$2/svn.tar.bz2.sha256" ]] && [[ "$(cut -d' ' -f1 "$2/svn.tar.bz2.sha256")" == "$sha" ]]; then
    return
  fi
  rm -rf "$2/wales" "$2/svn.tar.bz2.sha256"
  mkdir -p "$2"
  tar -xjf "$1" -C "$2"
  wales_stamp "$2" "$sha"
}

# wales_svnrev <tree> <program> prints <tree>/wales/<program>/source/SVNREV.
wales_svnrev() {
  local got
  got=$(tr -d '[:space:]' < "$1/wales/$2/source/SVNREV") || return 1
  if [[ -z "$got" ]]; then
    echo "$1/wales/$2/source/SVNREV is empty" >&2
    return 1
  fi
  echo "$got"
}

# The venv site-packages go on the harness sys.path, so the venv keeps the
# numpy and scipy the harness already imported.
harness_pins() {
  "${HARNESS_PYTHON:-python3}" -c 'import numpy, scipy; print(f"numpy=={numpy.__version__} scipy=={scipy.__version__}")'
}
