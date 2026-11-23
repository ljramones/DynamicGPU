#!/bin/sh
set -eu
module_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
arch=$(uname -m)
out="$module_dir/target/classes/native/macos-$arch"
mkdir -p "$out"
xcrun clang -std=c11 -fno-objc-arc -fblocks -dynamiclib -mmacosx-version-min=13.0 \
  -framework Foundation -framework Metal -Wl,-install_name,@rpath/libdynamis_metal.dylib \
  "$module_dir/src/main/native/dynamis_metal.m" -o "$out/libdynamis_metal.dylib"
# Production shader packaging requires Xcode's optional Metal toolchain.
if xcrun --find metal >/dev/null 2>&1; then
  xcrun -sdk macosx metal -mmacosx-version-min=13.0 -c "$module_dir/../dynamis-gpu-test/src/main/resources/compute/vector_add.metal" -o "$module_dir/target/vector_add.air"
  xcrun -sdk macosx metallib "$module_dir/target/vector_add.air" -o "$out/vector_add.metallib"
elif [ "${DYNAMIS_METAL_REQUIRE_OFFLINE:-0}" = 1 ]; then
  echo 'Offline Metal compiler required; install Xcode Metal toolchain.' >&2
  exit 1
else
  echo 'Metal compiler absent: bridge built; hardware tests use runtime MSL compilation.' >&2
fi
