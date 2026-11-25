#!/bin/sh
set -eu
cd "$(dirname "$0")"
vectrix_version=${1:-4.1.0}
vectrix_repo=${MAVEN_REPO:-$HOME/.m2/repository}/org/dynamisengine
classpath="$vectrix_repo/vectrix-ml/$vectrix_version/vectrix-ml-$vectrix_version.jar:$vectrix_repo/vectrix-core/$vectrix_version/vectrix-core-$vectrix_version.jar"
exec java -Xmx512m -cp "$classpath" KvCodecComparison.java
