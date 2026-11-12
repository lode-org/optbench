#!/bin/sh
# Build the pinned OGOLEM commit and the harness backend next to it.
# Usage: codes/ogolem/build.sh [dest]
# dest defaults to "$HOME/opt/ogolem/<commit>".
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
commit=cce35a376d929852ab1e80e98105c0d13d784b8d
dest=${1:-"$HOME/opt/ogolem/$commit"}
src=${OGOLEM_SRC:-"$HOME/opt/ogolem-src"}
java_home=${JAVA_HOME:-"$HOME/opt/jdk-25"}
gradle=${GRADLE:-"$HOME/opt/gradle-9.1.0/bin/gradle"}

if [ ! -x "$java_home/bin/javac" ]; then
  echo "javac is missing under $java_home" >&2
  exit 1
fi
if [ ! -x "$gradle" ]; then
  echo "gradle is missing at $gradle" >&2
  exit 1
fi

if [ ! -d "$src/.git" ]; then
  git clone https://github.com/ogolem/ogolem.git "$src"
fi
git -C "$src" fetch --depth 1 origin "$commit"
git -C "$src" checkout --force --detach "$commit"
git -C "$src" apply --check "$root/codes/ogolem/BackendFactory.patch"
git -C "$src" apply "$root/codes/ogolem/BackendFactory.patch"

JAVA_HOME="$java_home" "$gradle" -p "$src" bootJar -x test -x spotbugsMain -x spotlessCheck -x spotlessJava -x spotlessApply --no-daemon

jar=$(find "$src/build/libs" -name 'ogolem-snapshot.jar' -type f | head -n 1)
classes="$src/build/classes/java/main"
if [ -z "$jar" ] || [ ! -d "$classes/org/ogolem/core" ]; then
  echo "bootJar did not write the snapshot jar and its classes" >&2
  exit 1
fi

mkdir -p "$dest/classes"
# The snapshot is a Spring Boot fat jar. javac cannot see BOOT-INF/classes through -cp.
"$java_home/bin/javac" --enable-preview --release 25 -cp "$classes" -d "$dest/classes" \
  "$root/codes/ogolem/OptbenchBackend.java"
"$java_home/bin/jar" --create --file "$dest/optbench-backend.jar" -C "$dest/classes" .
cp "$jar" "$dest/ogolem-snapshot.jar"
rm -rf "$dest/exploded"
mkdir -p "$dest/exploded"
(
  cd "$dest/exploded"
  "$java_home/bin/jar" xf "$dest/ogolem-snapshot.jar" BOOT-INF
)
printf '%s\n' "$commit" > "$dest/COMMIT"
echo "$dest"
