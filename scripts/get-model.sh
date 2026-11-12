#!/usr/bin/env bash
# Download the Flash Next artifact and the DFlash2 drafter, and join them into one file set.
#
#   scripts/get-model.sh
#
# Environment:
#   MODELS_DIR   where everything goes             (default: ./models; needs about 105 GiB)
#   NINFER_DIR   patched engine source             (default: ./ninfer; scripts/setup.sh makes it)
#
# Result: MODELS_DIR/flash-next-dflash2-q8/qwen3_8_flash_next.ninfer and config/local.toml
# pointing at it. The joined set hard-links the artifact's files and adds one 0.8 GB file with
# the drafter in the Spark's 8-bit layout, so it must be on the same filesystem as the artifact.
# Needs: the Hugging Face CLI (`hf`, from `pip install -U huggingface_hub`) and numpy.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS_DIR=${MODELS_DIR:-$ROOT/models}
NINFER_DIR=${NINFER_DIR:-$ROOT/ninfer}
ARTIFACT=neroued/Qwen3.8-Flash-Next-NInfer
DRAFTER=incoai/Qwen3.8-Flash-Next-DFlash2
OUT=$MODELS_DIR/flash-next-dflash2-q8
ENTRY=qwen3_8_flash_next.ninfer

command -v hf >/dev/null || { echo "the Hugging Face CLI is missing: pip install -U huggingface_hub" >&2; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "numpy is missing: pip install numpy" >&2; exit 1; }
[ -d "$NINFER_DIR/tools/artifact" ] || { echo "no engine source in $NINFER_DIR; run scripts/setup.sh first" >&2; exit 1; }

mkdir -p "$MODELS_DIR"
hf download "$ARTIFACT" --local-dir "$MODELS_DIR/Qwen3.8-Flash-Next-NInfer"
hf download "$DRAFTER" config.json model.safetensors --local-dir "$MODELS_DIR/Qwen3.8-Flash-Next-DFlash2"

if [ -f "$OUT/$ENTRY" ]; then
    echo "file set exists already: $OUT"
else
    python3 "$ROOT/tools/graft_dflash2.py" --q8 "$NINFER_DIR" \
        "$MODELS_DIR/Qwen3.8-Flash-Next-NInfer/$ENTRY" "$MODELS_DIR/Qwen3.8-Flash-Next-DFlash2" "$OUT"
fi

sed "s|^path = .*|path = \"$OUT/$ENTRY\"|" "$ROOT/config/flash-next-spark.toml" > "$ROOT/config/local.toml"
echo "wrote config/local.toml; start the server with scripts/serve.sh"
