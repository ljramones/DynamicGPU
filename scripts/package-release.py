#!/usr/bin/env python3
"""Package an already-staged file Maven repository for a GitHub release."""
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET

version, repository_arg = sys.argv[1:]
repository = Path(repository_arg).resolve()
out = repository.parent
artifacts = ["dynamisgpu-parent", "dynamis-gpu-api", "dynamis-gpu-test",
             "dynamis-gpu-vulkan", "dynamis-gpu-metal"]
for artifact in artifacts:
    folder = repository / "org/dynamisengine" / artifact / version
    required = [folder / f"{artifact}-{version}.pom"]
    if artifact != "dynamisgpu-parent":
        required += [folder / f"{artifact}-{version}{suffix}.jar"
                     for suffix in ("", "-sources", "-javadoc")]
    for file in required:
        if not file.is_file():
            raise SystemExit(f"Missing release artifact: {file}")
    pom = ET.parse(required[0]).getroot()
    own_version = pom.find("{*}version")
    if own_version is None:
        own_version = pom.find("{*}parent/{*}version")
    if own_version is None or own_version.text != version:
        raise SystemExit(f"Mismatched version in {required[0]}")
if (repository / "org/dynamisengine/dynamis-gpu-bench").exists():
    raise SystemExit("Internal benchmark module must not be published")
metal = repository / "org/dynamisengine/dynamis-gpu-metal" / version / f"dynamis-gpu-metal-{version}.jar"
with zipfile.ZipFile(metal) as jar:
    natives = [n for n in jar.namelist() if n.startswith("native/macos-") and n.endswith("/libdynamis_metal.dylib")]
    if not natives:
        raise SystemExit("Metal JAR has no native bridge; use the metal-native profile")

readme = f"""DynamisGPU {version}

This Maven repository bundle is distributed through GitHub, not Maven Central.
Requires JDK 25. Native Metal bridge included: {', '.join(natives)}.

Install the versioned artifacts into your local Maven repository:

    sh install.sh

Then use org.dynamisengine:dynamis-gpu-api:{version} and the selected backend:
org.dynamisengine:dynamis-gpu-metal:{version} or
org.dynamisengine:dynamis-gpu-vulkan:{version}.
The dynamis-gpu-test artifact contains optional mocks and conformance tests.
Existing graphics contracts remain available. No benchmark artifact is published.

Alternatively configure this directory's repository/ folder as a Maven file
repository. External dependencies resolve normally from Maven Central.

Run Java with --enable-native-access=ALL-UNNAMED on the class path.
Metal uses native Metal directly; Vulkan on macOS needs the SDK loader and driver.
The initial native Metal binary is for macOS arm64 (deployment target 13.0;
verified on macOS 27.0.1). Offline metallib generation and other platforms are
unverified. Consumers supply their own kernel bundles; MSL source compilation is
available for development. See docs/compute-runtime.md in tag v{version}.

Runtime MSL compilation on macOS 15+ uses safe/precise math and disables implicit
multiply-add contraction. MoltenVK fast math is disabled by the Vulkan provider;
MVK_CONFIG_FAST_MATH_ENABLED=0 is not required. Older macOS and offline metallibs
retain their documented numerical limits. See the versioned runtime guide.

Sources, Javadocs, POMs and repository checksums are included. The parent dependency
org.dynamisengine:dynamis-parent:1.0.1 remains unchanged and is on Maven Central.
"""
install = '''#!/bin/sh
set -eu
bundle=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$bundle"
version=VERSION
for artifact in dynamisgpu-parent dynamis-gpu-api dynamis-gpu-test dynamis-gpu-vulkan dynamis-gpu-metal; do
  base="$bundle/repository/org/dynamisengine/$artifact/$version/$artifact-$version"
  if [ "$artifact" = dynamisgpu-parent ]; then
    mvn org.apache.maven.plugins:maven-install-plugin:3.2.0:install-file \\
      "-Dfile=$base.pom" "-DpomFile=$base.pom" "$@"
  else
    mvn org.apache.maven.plugins:maven-install-plugin:3.2.0:install-file \\
      "-Dfile=$base.jar" "-DpomFile=$base.pom" \\
      "-Dsources=$base-sources.jar" "-Djavadoc=$base-javadoc.jar" "$@"
  fi
done
'''.replace("VERSION", version)
archive = out / f"dynamisgpu-{version}-maven.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    prefix = f"dynamisgpu-{version}/"
    bundle.writestr(prefix + "README.txt", readme)
    bundle.writestr(prefix + "install.sh", install)
    for file in sorted(repository.rglob("*")):
        if file.is_file():
            bundle.write(file, prefix + "repository/" + file.relative_to(repository).as_posix())

# Convenient individual runtime downloads; the ZIP contains their POMs and all attachments.
assets = [archive]
for artifact in artifacts[1:]:
    source = repository / "org/dynamisengine" / artifact / version / f"{artifact}-{version}.jar"
    destination = out / source.name
    shutil.copy2(source, destination)
    assets.append(destination)
(out / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}\n" for file in assets))
print(f"Packaged {archive}; native bridge: {', '.join(natives)}")
print(f"Checksums: {out / 'SHA256SUMS'}")
