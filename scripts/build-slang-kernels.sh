#!/bin/sh
# Compile DynamisGPU's own Slang kernels with the dynamis-gpu-shaders tool (needs slangc; see ShaderBuild).
# The build uses the committed outputs; rerun this after editing a .slang file.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
mvn -q -o -pl dynamis-gpu-shaders compile 2>/dev/null || mvn -q -pl dynamis-gpu-shaders compile
src="$repo/dynamis-gpu-api/src/main/slang"
java -cp dynamis-gpu-shaders/target/classes org.dynamisengine.gpu.shaders.ShaderBuild \
  --src "$src" --kernels "$src/kernels.list" \
  --out "$repo/dynamis-gpu-api/src/main/resources/org/dynamisengine/gpu/api/compute/spi/kernels"
