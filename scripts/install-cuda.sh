#!/usr/bin/env bash
# Install the CUDA 13.4.3 compiler toolkit for aarch64 (sbsa) into a folder, without sudo.
#
#   scripts/install-cuda.sh [PREFIX]        default PREFIX: $HOME/cuda-13.4
#
# Why: ninfer's Spark kernels need CUDA 13.4. DGX OS ships 13.0, and a 13.0 build aborts on any
# prompt longer than a few dozen tokens (cudaErrorIllegalInstruction). The 13.4 toolkit builds
# binaries that run on the stock driver (tested with 580.178.04). Nothing system-wide changes.
#
# It downloads NVIDIA's redistributable archives (about 100 MB), checks their sha256 against
# NVIDIA's manifest and unpacks them into PREFIX/toolkit.
set -euo pipefail

PREFIX=${1:-$HOME/cuda-13.4}
BASE=https://developer.download.nvidia.com/compute/cuda/redist
MANIFEST=redistrib_13.4.3.json
COMPONENTS="cccl cuda_crt cuda_cudart cuda_culibos cuda_cupti cuda_nvcc cuda_nvtx cuda_profiler_api libnvfatbin libnvptxcompiler libnvvm"

if [ -x "$PREFIX/toolkit/bin/nvcc" ]; then
    echo "already installed: $("$PREFIX/toolkit/bin/nvcc" --version | tail -1)"
    exit 0
fi
[ "$(uname -m)" = aarch64 ] || { echo "this script installs the aarch64 (sbsa) toolkit; found $(uname -m)" >&2; exit 1; }

mkdir -p "$PREFIX/dl" "$PREFIX/toolkit"
cd "$PREFIX/dl"
curl -fsSLO "$BASE/$MANIFEST"

for component in $COMPONENTS; do
    read -r path sha < <(python3 - "$MANIFEST" "$component" <<'EOF'
import json, sys
entry = json.load(open(sys.argv[1]))[sys.argv[2]]["linux-sbsa"]
print(entry["relative_path"], entry["sha256"])
EOF
    )
    file=$(basename "$path")
    [ -f "$file" ] || curl -fsSLO "$BASE/$path"
    echo "$sha  $file" | sha256sum -c --quiet
    tar -xJf "$file" -C "$PREFIX/toolkit" --strip-components=1
    echo "unpacked $file"
done

cd "$PREFIX/toolkit"
[ -e lib64 ] || ln -s lib lib64
mkdir -p targets/sbsa-linux
[ -e targets/sbsa-linux/lib ] || ln -s ../../lib targets/sbsa-linux/lib
[ -e targets/sbsa-linux/include ] || ln -s ../../include targets/sbsa-linux/include

"$PREFIX/toolkit/bin/nvcc" --version | tail -1
echo "CUDA toolkit ready in $PREFIX/toolkit"
