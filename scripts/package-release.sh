#!/bin/sh
# Build, test, install locally, and stage a downloadable Maven repository.
# The inherited Central release profile is explicitly disabled.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
version=$(python3 -c 'import xml.etree.ElementTree as E; print(E.parse("pom.xml").getroot().find("{*}version").text)')
repository="$repo/target/release/repository"
mvn -P'!release,metal-native,metal-hardware,vulkan-hardware' clean deploy \
  -DdeployAtEnd=true "-DaltDeploymentRepository=github-release::file://$repository" "$@"
python3 scripts/package-release.py "$version" "$repository"
