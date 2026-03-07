#!/usr/bin/env bash
set -euo pipefail

echo "==> Building DynamisGPU"
mvn clean install

echo "==> Build complete"
