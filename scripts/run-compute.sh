#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -lt 2 ]; then echo 'Usage: sh scripts/run-compute.sh example|benchmark metal|vulkan [arguments...]' >&2; exit 1; fi
case "$1" in
  example) main=org.dynamisengine.gpu.bench.compute.ComputeExampleMain ;;
  benchmark) main=org.dynamisengine.gpu.bench.compute.ComputeBenchmarkMain ;;
  *) echo 'Choose example or benchmark' >&2; exit 1 ;;
esac
shift
cd "$repo"
# Run mvn install first: this command resolves the reactor's published local artifacts.
mvn -q -pl dynamis-gpu-bench dependency:build-classpath -Dmdep.outputFile=target/compute-classpath.txt
exec java --enable-native-access=ALL-UNNAMED --enable-preview --add-modules jdk.incubator.vector \
  "-Ddynamisgpu.vk.loader=${DYNAMIS_VULKAN_LOADER:-}" \
  -cp "$repo/dynamis-gpu-bench/target/classes:$(cat "$repo/dynamis-gpu-bench/target/compute-classpath.txt")" "$main" "$@"
