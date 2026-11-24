#!/bin/sh
# Run the Vulkan conformance suite and the primitives' hardware tests on Mesa's lavapipe (a CPU Vulkan driver) in a
# Linux container: the same driver CI uses, runnable on a Mac. Needs Docker. The repository is copied into a temp
# directory (the container's Linux build must not mix with your target/ folders); ~/.m2 is mounted so local-only
# artifacts (dynamis-parent, Vectrix 3.x) resolve.
#     sh scripts/test-lavapipe.sh            # dynamis-gpu-vulkan and dynamis-gpu-primitives
#     sh scripts/test-lavapipe.sh -pl dynamis-gpu-vulkan -am -Dtest=VulkanComputeTest -Dsurefire.failIfNoSpecifiedTests=false
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/src"
(cd "$repo" && tar --exclude=./target --exclude='*/target' --exclude=./.git -cf - .) | (cd "$work/src" && tar -xf -)
if [ $# -eq 0 ]; then set -- -pl dynamis-gpu-vulkan,dynamis-gpu-primitives -am; fi
cat > "$work/run.sh" <<SCRIPT
set -e
apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends mesa-vulkan-drivers libvulkan1 vulkan-tools maven >/dev/null 2>&1
ICD=\$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
VK_DRIVER_FILES=\$ICD vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|apiVersion|driverName' | head -3
LOADER=\$(ls /usr/lib/*-linux-gnu/libvulkan.so.1 | head -1)
cd /work/src
status=0
VK_DRIVER_FILES=\$ICD mvn -B -q -Pvulkan-hardware -Ddynamisgpu.vk.loader=\$LOADER $* test || status=\$?
grep -h 'Tests run' */target/surefire-reports/*Vulkan*.txt 2>/dev/null || true
exit \$status
SCRIPT
docker run --rm -v "$work:/work" -v "$HOME/.m2:/root/.m2" eclipse-temurin:25-jdk-noble sh /work/run.sh
