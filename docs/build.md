# Build and run

## Requirements

| | |
|---|---|
| Machine | NVIDIA DGX Spark (GB10, aarch64, 121 GiB unified memory) |
| System | Ubuntu 24.04 as shipped; tested with NVIDIA driver 580.178.04 |
| Compiler | GCC 13, CMake 3.28 or newer, Ninja |
| CUDA | 13.4 toolkit for compiling (`scripts/install-cuda.sh`); the system's 13.0 runtime and driver stay |
| Disk | about 110 GiB: model 101, drafter 1.1 + 0.8, build 3 |
| Python | 3.10 or newer with `numpy` and the Hugging Face CLI, only for `scripts/get-model.sh` |

Packages:

```bash
sudo apt install build-essential cmake ninja-build pkg-config git curl xz-utils \
     libjpeg-dev libhwloc-dev liburing-dev \
     libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
pip install -U huggingface_hub numpy
```

## 1. CUDA 13.4 compiler

```bash
scripts/install-cuda.sh            # into ~/cuda-13.4, or pass another folder
```

ninfer's Spark kernels need the CUDA 13.4 compiler. A build with the system's CUDA 13.0 compiles and loads,
then aborts on any prompt longer than a few dozen tokens with `cudaErrorIllegalInstruction`. The script
downloads NVIDIA's redistributable compiler archives (about 100 MB), verifies them against NVIDIA's manifest
and unpacks them into a folder of your own. It does not touch the system CUDA, and the binaries it builds run
on the stock driver.

## 2. Engine

```bash
scripts/setup.sh                   # JOBS=2 scripts/setup.sh on a busy machine
```

This clones ninfer, checks out `13ebb92`, applies `patches/*.patch` with `git am` onto a branch called
`flash-next-spark`, configures the `dgx-spark-release` preset with the 13.4 compiler and builds
`ninfer/build/spark/apps/ninfer-serve`.

- The build compiles about 420 CUDA sources. Each compile job can take 2 to 3 GiB, and on the Spark that
  memory is shared with any model that is loaded. Build with nothing large running, or lower `JOBS`.
- `scripts/setup.sh --no-build` only prepares the source.
- To apply the patches to a checkout you already have: `git checkout 13ebb92 && git am /path/to/patches/*.patch`.

## 3. Model

```bash
scripts/get-model.sh               # MODELS_DIR=/data/models scripts/get-model.sh
```

It downloads ninfer's Flash Next artifact
([`neroued/Qwen3.8-Flash-Next-NInfer`](https://huggingface.co/neroued/Qwen3.8-Flash-Next-NInfer), 101 GiB) and
the drafter ([`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
1.1 GiB), then runs `tools/graft_dflash2.py --q8` to join them:

```
models/flash-next-dflash2-q8/
  qwen3_8_flash_next.ninfer              new entry file: old payload, longer directory
  qwen3_8_flash_next.ninfer.part-0001    hard links to the artifact's parts
  ...
  qwen3_8_flash_next.ninfer.part-0004    the drafter, 776 MB, Spark 8-bit layout
```

The original artifact is only read. Because of the hard links both folders must be on the same filesystem and
the joined set costs about 33 GiB of new disk (the entry file is rewritten; the parts are shared). Without
`--q8` the script stores the drafter as BF16 (1.2 GB), which is the layout patch 0003 alone understands.

## 4. Serve

```bash
scripts/serve.sh                   # foreground; Ctrl-C stops it and prints the statistics
scripts/serve.sh --port 8888 --host 0.0.0.0
```

Ready after about 95 seconds. The API is ninfer's OpenAI-compatible server: `/v1/chat/completions`,
`/v1/models`, `/health`. To keep it running, wrap the script in a systemd user unit or `tmux`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `cudaErrorIllegalInstruction` on the first longer prompt | Built with CUDA 13.0. Rebuild with the 13.4 toolkit |
| The server is killed while loading, or the machine stalls | Another model holds the memory. The Spark has room for one large engine |
| `requires --draft-tokens in [1,15]` | `draft_tokens` outside the range the patches allow |
| `DFlash2 drafter left off` at start | The file set has no drafter (run `get-model.sh`), or drafts are disabled |
| `output head screen left off (not this head)` | The model is not Flash Next (the screen is built for its 248,320 × 2,560 head) |
| Start takes minutes | The first start after a reboot reads 74 GiB from disk; later starts come from the page cache |
| Lower speed than the tables | Something else is using the GPU or the memory bus; thinking is on (the tables are with thinking off) |
