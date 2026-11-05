#!/usr/bin/env bash
# Fetch eOn SVN r2549 one file per request and check it against the manifest.
# A whole-tree svn export of r2549 stops on HTTP 403 (E175013) at a
# different file on each try. A request for one file at the revision does not.
# A POTCAR is a VASP pseudopotential file under the VASP license, so the
# manifest's POTCAR rows are left out unless --with-potcar is given.
# usage: fetch.sh [--with-potcar] DEST [MANIFEST]
set -euo pipefail

usage="usage: fetch.sh [--with-potcar] DEST [MANIFEST]"
potcar=0
if [[ ${1:-} == --with-potcar ]]; then
  potcar=1
  shift
fi
here=$(cd "$(dirname "$0")" && pwd)
dest=${1:?$usage}
manifest=${2:-$here/r2549.manifest}
base=${EON_SVN_URL:-https://theory.cm.utexas.edu/svn/eon/!svn/bc/2549}

digest() { sha256sum "$1" | cut -d' ' -f1; }

fetched=0
kept=0
skipped=0
while read -r sha size mode path; do
  if [[ $potcar == 0 && ${path##*/} == POTCAR ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  file=$dest/$path
  if [[ -f $file && $(stat -c %s "$file") == "$size" && $(digest "$file") == "$sha" ]]; then
    chmod "$mode" "$file"
    kept=$((kept + 1))
    continue
  fi
  mkdir -p "$(dirname "$file")"
  curl -fsS --retry 5 --retry-all-errors --retry-delay 2 -o "$file.part" "$base/$path"
  got_size=$(stat -c %s "$file.part")
  got_sha=$(digest "$file.part")
  if [[ $got_size != "$size" || $got_sha != "$sha" ]]; then
    rm -f "$file.part"
    echo "$path: $got_size bytes, sha256 $got_sha; the manifest has $size bytes, sha256 $sha" >&2
    exit 1
  fi
  mv "$file.part" "$file"
  chmod "$mode" "$file"
  fetched=$((fetched + 1))
done < <(grep -v '^#' "$manifest")
echo "fetched $fetched, already present $kept, POTCAR left out $skipped, from $manifest"
