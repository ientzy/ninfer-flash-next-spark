# Results

All numbers are from one DGX Spark (GB10, 121 GiB unified memory, Ubuntu 24.04, driver 580.178.04, CUDA 13.4.3
toolkit), November 2026, with the engine alone on the machine, thinking off, fp8 KV cache. Speeds are the
server's own decode rate in tokens per second. Each number is a single run; repeated runs differ by about 5%.

## Against stock ninfer

Six kinds of text, greedy, 384 tokens, one stream:

| Text | Stock ninfer, MTP 3 | Stock ninfer, MTP 5 | This build |
|---|---:|---:|---:|
| No drafts at all | 26.0 | 26.0 | |
| Story | 39.3 | 34.5 | **42.3** |
| Explanation | 49.1 | 50.4 | **61.6** |
| New code | 56.9 | 61.7 | **81.3** |
| Thinking | 59.5 | 63.6 | **92.8** |
| JSON from notes | 64.7 | 80.4 | **120.3** |
| Edit a file | 63.5 | 76.3 | **157.8** |

- *Stock ninfer* is [Neroued/ninfer](https://github.com/Neroued/ninfer) at `13ebb92` with its Flash Next
  artifact and optimized draft head.
- *This build* is all nine patches with the defaults of `scripts/serve.sh`.

Unchanged from stock ninfer: prefill 2,613 tok/s at 55K tokens, and weights load in 30 s.

## Step by step

The bench prompts of `bench/bench.py speed` (500 tokens, greedy, one stream). Each row adds one patch.

| Build | New code | Explanation | Story | File edit | JSON |
|---|---:|---:|---:|---:|---:|
| MTP, 7 drafts | 67 | 46 | 30 | 84 | 96 |
| DFlash2 drafter, BF16, 7 drafts | 77.5 | 45.5 | 32.8 | 106.5 | 107.5 |
| + Spark 8-bit drafter layout | 81.8 | 46.6 | 34.3 | 108.9 | 109.8 |
| + automatic width | 78.6 | 48.9 | 39.3 | 109.2 | 110.4 |
| + window 15 and copy extension | 82.6 | 56.8 | 40.3 | 135.3 | 110.9 |
| + output head screen | **86.3** | **60.0** | **42.6** | **139.0** | **114.4** |

The MTP row comes from the six-text probe above, so compare it loosely. A text repeated exactly
(`bench.py copy`): 113 tok/s with 7 drafts, 181 with the window of 15, 185 with the head screen.

## Several streams

N greedy requests at once (the five bench prompts, cycled), total tokens per second over the whole run
including prefill:

| Streams | MTP 7 | DFlash2, fixed 7 | DFlash2, automatic (window 7) | This build (window 15) |
|---:|---:|---:|---:|---:|
| 1 | 64 | 80 | 77 to 79 | 82 |
| 2 | 66 | 72 | 82 | 81 |
| 4 | 80 | 87 | 94 | 92 to 97 |
| 8 | 118 | 122 | 133 to 141 | 125 to 131 |

With 12 requests queued on 8 slots: about 150 tok/s. The window of 15 costs about 4% at 8 streams (the layout
is twice as wide); set `draft_tokens = 7` in the configuration if you mostly serve many streams.

Per stream at 8 streams (code, explanation, story, file edit, JSON, then code, explanation, story again):

| | Total | Per stream |
|---|---:|---|
| Fixed 7 drafts | 121 | 29 20 18 41 39 31 21 17 |
| Own draft count per stream (window 7) | 133 to 141 | 33 24 20 46 49 34 23 20 |

## Output head screen: speed and exactness

Same build, screen off against on, one stream:

| Text | Off | On |
|---|---:|---:|
| New code, greedy | 82.6 | 86.3 |
| Explanation, greedy | 56.8 | 60.0 |
| Story, greedy | 40.3 | 42.6 |
| File edit | 135.3 | 139.0 |
| JSON | 110.9 | 114.4 |
| Exact copy | 180.7 | 184.5 |
| Code, temperature 0.7, top-p 0.95, top-k 20 | 72.9 | 76.3 |
| Explanation, same sampling | 51.1 | 53.4 |
| Story, temperature 1.0 | 36.8 | 38.9 |
| Story, temperature 0.8, min-p 0.05, top-k 12 | 38.1 | 40.5 |
| Greedy with presence penalty (full head) | 38.8 | 38.8 |
| Sampled with frequency penalty (full head) | 45.7 | 45.7 |
| JSON mode (full head) | 56.6 | 56.8 |

Every one of these texts is identical off and on. Round time 67.6 → 64.9 ms.

Check mode (`NINFER_FLASH_HEAD_SCREEN=check`: both heads run, every verify position is compared):

| | |
|---|---:|
| Positions compared | 6,858 |
| Top-20 tokens the screen left out | 0 |
| Positions with a different first token | 0 |
| Top-20 values with a different last bit | 2 |
| Tokens computed exactly per position | about 52 of 248,077 |

## Repeating the measurements

```bash
scripts/serve.sh &                       # wait for "ready"
python3 bench/bench.py speed mine        # five prompts
python3 bench/bench.py copy
python3 bench/bench.py streams 1 2 4 8
python3 bench/bench.py edge mine         # sampling, penalties, JSON mode
python3 bench/bench.py mixed 6           # free and JSON-constrained requests together
```

To compare a switch, run the server twice and compare the saved runs:

```bash
NINFER_FLASH_HEAD_SCREEN=0 scripts/serve.sh &   # then: bench.py speed off; bench.py edge off; stop it
scripts/serve.sh &                              # then: bench.py speed on;  bench.py edge on
python3 bench/bench.py compare off on
python3 bench/bench.py compare-edge off on
```

For the exactness check, start with `NINFER_FLASH_HEAD_SCREEN=check`, run any of the above, stop the server
with Ctrl-C and read the `head screen check` line it prints. When the server stops it also prints the rounds,
drafts kept, widths chosen and the mean round time.

Profiling used `nsys launch --trace=cuda,nvtx --cuda-graph-trace=node` around the server and
`nsys stats -r cuda_gpu_kern_sum` on the result.
