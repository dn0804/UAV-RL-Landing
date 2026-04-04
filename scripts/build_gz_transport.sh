#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────
# Build gz_transport_py — minimal Python bindings for gz-transport12.
# Installs the .so into the active Python's site-packages.
#
# Prerequisites (installed by Dockerfile):
#   apt: python3-pybind11 pybind11-dev python3-dev
#        libgz-transport12-dev libgz-msgs9-dev
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$REPO_ROOT/gz_transport_py"

echo "=== Building gz_transport_py ==="

cd "$SRC_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# Find the built .so
SO_FILE=$(find . -name "gz_transport_py*.so" | head -1)
if [ -z "$SO_FILE" ]; then
    echo "[ERROR] Build succeeded but .so not found"
    exit 1
fi

# Install to site-packages
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp "$SO_FILE" "$SITE_PACKAGES/"
echo "Installed: $SITE_PACKAGES/$(basename "$SO_FILE")"

# Verify
python3 -c "import gz_transport_py; print('gz_transport_py OK')"
