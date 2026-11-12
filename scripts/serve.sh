#!/usr/bin/env bash
# Start Qwen3.8 Flash Next with the DFlash2 drafter, in the foreground.
#
#   scripts/serve.sh [extra ninfer-serve flags]      e.g. scripts/serve.sh --port 8888
#
# Environment:
#   CONFIG       server configuration             (default: config/local.toml)
#   NINFER_DIR   patched and built engine         (default: ./ninfer)
#   NINFER_FLASH_*  see docs/configuration.md; the values below are the measured defaults and
#                   anything already set in the environment wins.
#
# Do not start it while another large engine holds the GPU's memory: the model needs about
# 100 GiB of the Spark's 121.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
NINFER_DIR=${NINFER_DIR:-$ROOT/ninfer}
CONFIG=${CONFIG:-$ROOT/config/local.toml}
SERVER=$NINFER_DIR/build/spark/apps/ninfer-serve

[ -x "$SERVER" ] || { echo "no server binary at $SERVER; run scripts/setup.sh" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "no configuration at $CONFIG; run scripts/get-model.sh" >&2; exit 1; }

# DFlash2 drafter instead of MTP steps.
export NINFER_FLASH_DFLASH2=${NINFER_FLASH_DFLASH2:-1}
# Verify only the drafts that pay off: fixed cost of a round in verify columns, and the power
# applied to the drafter's confidence (2 is the calibrated value).
export NINFER_FLASH_DFLASH2_AUTO=${NINFER_FLASH_DFLASH2_AUTO:-17}
export NINFER_FLASH_DFLASH2_AUTO_POWER=${NINFER_FLASH_DFLASH2_AUTO_POWER:-2}
# Copy extension: continue a round with the tokens after an earlier stretch that the newest
# 8 tokens repeat (needs draft_tokens above 7 in the configuration).
export NINFER_FLASH_DFLASH2_COPY=${NINFER_FLASH_DFLASH2_COPY:-8}
# Output head: screen the vocabulary with an 8-bit copy, compute only the possible top 20 exactly.
export NINFER_FLASH_HEAD_SCREEN=${NINFER_FLASH_HEAD_SCREEN:-1}

# 12 GiB of context store leaves about 20 GiB free; "auto" takes all the memory there is.
exec "$SERVER" --config "$CONFIG" --context-capacity "${CONTEXT_CAPACITY:-12GiB}" "$@"
