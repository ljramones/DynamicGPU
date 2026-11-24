#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for kernel in vector_add fp_strict; do
  glslangValidator -V --target-env vulkan1.1 --source-entrypoint main -e "$kernel" \
    "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.comp" \
    -o "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.spv"
  # Optional additional validation when the SPIR-V tools are installed.
  if command -v spirv-val >/dev/null 2>&1; then
    spirv-val --target-env vulkan1.1 "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.spv"
  fi
done
