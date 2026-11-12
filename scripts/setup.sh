#!/usr/bin/env bash
# Fetch ninfer at the pinned commit, apply this repository's patches and build it for the DGX Spark.
#
#   scripts/setup.sh [--no-build]
#
# Environment:
#   NINFER_DIR   where the engine source goes      (default: ./ninfer)
#   CUDA_HOME    CUDA 13.4 toolkit                 (default: $HOME/cuda-13.4/toolkit)
#   JOBS         parallel compile jobs             (default: 4; each can take 2-3 GiB of memory)
#
# A full build compiles about 420 CUDA sources and takes a while. Run it with no model loaded,
# or lower JOBS: the Spark's memory is shared between the compiler and any running engine.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
UPSTREAM=https://github.com/Neroued/ninfer.git
PIN=13ebb92f6f1d56b676e5300cb19555b2ba24053a
NINFER_DIR=${NINFER_DIR:-$ROOT/ninfer}
CUDA_HOME=${CUDA_HOME:-$HOME/cuda-13.4/toolkit}
JOBS=${JOBS:-4}

if [ ! -d "$NINFER_DIR/.git" ]; then
    git clone "$UPSTREAM" "$NINFER_DIR"
fi
cd "$NINFER_DIR"

if git rev-parse -q --verify refs/heads/flash-next-spark >/dev/null; then
    echo "branch flash-next-spark exists already; leaving the source as it is"
    git checkout -q flash-next-spark
else
    git fetch -q origin "$PIN" 2>/dev/null || git fetch -q origin
    git checkout -q -b flash-next-spark "$PIN"
    git -c user.name="${GIT_AUTHOR_NAME:-setup}" -c user.email="${GIT_AUTHOR_EMAIL:-setup@localhost}" \
        am -q "$ROOT"/patches/*.patch
    echo "applied $(ls "$ROOT"/patches/*.patch | wc -l) patches on ${PIN:0:7}"
fi

[ "${1:-}" = --no-build ] && exit 0

[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc in $CUDA_HOME/bin; run scripts/install-cuda.sh first" >&2; exit 1; }
"$CUDA_HOME/bin/nvcc" --version | grep -q "release 13\.[4-9]" ||
    echo "warning: ninfer's Spark kernels need CUDA 13.4 or newer; found: $("$CUDA_HOME/bin/nvcc" --version | tail -1)" >&2

export PATH=$CUDA_HOME/bin:$PATH
cmake --preset dgx-spark-release -B build/spark \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME"
cmake --build build/spark -j"$JOBS"
echo "built: $NINFER_DIR/build/spark/apps/ninfer-serve"
