#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for kernel in vector_add fp_strict error_report; do
  glslangValidator -V --target-env vulkan1.1 --source-entrypoint main -e "$kernel" \
    -I"$repo/dynamis-gpu-api/src/main/resources/org/dynamisengine/gpu/api/compute/shader" \
    "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.comp" \
    -o "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.spv"
  # Optional additional validation when the SPIR-V tools are installed.
  if command -v spirv-val >/dev/null 2>&1; then
    spirv-val --target-env vulkan1.1 "$repo/dynamis-gpu-test/src/main/resources/compute/$kernel.spv"
  fi
done
# The device feature probe ships in dynamis-gpu-api (ComputeDevice runs it).
probe="$repo/dynamis-gpu-api/src/main/resources/org/dynamisengine/gpu/api/compute/spi/fma_probe"
glslangValidator -V --target-env vulkan1.1 --source-entrypoint main -e fma_probe "$probe.comp" -o "$probe.spv"
if command -v spirv-val >/dev/null 2>&1; then
  spirv-val --target-env vulkan1.1 "$probe.spv"
fi
