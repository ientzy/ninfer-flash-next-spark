# Configuration

Two places: the server's TOML file (ninfer's own options) and environment variables (everything these patches
add). `scripts/serve.sh` sets the variables to the measured defaults unless they are already set.

## TOML (`config/flash-next-spark.toml`)

Only the settings that matter here; `ninfer-serve --help` and ninfer's `ninfer.example.toml` list the rest.

| Setting | Value | Notes |
|---|---|---|
| `speculative.backend` | `"mtp"` | Stays `mtp`: the drafter replaces MTP rounds when its variable is set |
| `speculative.draft_tokens` | `15` | The verify window, 1 to 15. The drafter drafts at most 7; more only helps with copy extension. Use `7` if you mostly serve many streams |
| `speculative.proposal_head` | `"optimized"` | ninfer's own setting |
| `engine.max_concurrency` | `8` | Streams served at once, 1 to 8 |
| `engine.kv_dtype` | `"fp8"` | What everything here was measured with |
| `engine.cuda_graph` | `true` | Keep on: drafter rounds run on captured graphs and were only measured this way |
| `server.host`, `server.port` | `127.0.0.1`, `8080` | `0.0.0.0` to serve the network |

`scripts/serve.sh` passes `--context-capacity 12GiB` (override with `CONTEXT_CAPACITY`). `auto` gives the
context store all remaining memory, which leaves the machine about 3 GiB.

## Environment variables

### Drafter

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_FLASH_DFLASH2` | unset | Any value turns the DFlash2 drafter on. Unset: plain MTP |
| `NINFER_FLASH_DFLASH2_AUTO` | `0` | Automatic draft width. The value is the fixed cost of a round in verify columns; `17` on the Spark. `0`: always verify the full window |
| `NINFER_FLASH_DFLASH2_AUTO_POWER` | `1` | Power applied to the drafter's confidence. `2` is the calibrated value |
| `NINFER_FLASH_DFLASH2_AUTO_ROW` | `3` | Extra cost per stream in a round with several streams |
| `NINFER_FLASH_DFLASH2_AUTO_ROWS` | `1` | `0`: rounds with several streams always verify the full window |
| `NINFER_FLASH_DFLASH2_COPY` | `0` | Copy extension. The value is how many newest tokens must repeat earlier text (2 to 64); `8` is the measured choice. Needs `draft_tokens` above 7 |
| `NINFER_FLASH_DFLASH2_MIN_CONTEXT` | `1` | Tokens a stream must have seen before it drafts (below that it uses MTP) |

### Output head

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_FLASH_HEAD_SCREEN` | unset | `1`: screened output head. `check`: run both heads, use the full one, and count differences (slower; for validation). `0` or unset: full head |

### From the first two patches

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_PROMPT_LOOKUP` | `0` | Prompt-lookup rounds on the MTP path: suffix length, 2 to 64. Not needed with the drafter and copy extension |

## What the server prints

At start:

```
ninfer: DFlash2 drafter on, 7 draft tokens in a window of 15, width automatic
ninfer: output head screen on
```

When it stops:

```
ninfer: DFlash2 rounds 2580, MTP fallbacks 8
ninfer: DFlash2 drafts 9939, kept 5204, predicted kept 5417; widths 1:695 2:805 3:570 ...
ninfer: DFlash2 copy extensions 42, tokens 336
ninfer: DFlash2 round ms by rows 1:64.9 2:75.9 ... 8:178.0
ninfer: head screen rounds 2297, full head rounds 298, columns 3782, tokens computed per column 53.2
```

- `MTP fallbacks`: rounds that used MTP because a stream had no drafter context yet.
- `kept` against `predicted kept`: how well the automatic width's estimate matches; they should be close.
- `widths`: how often each draft count was chosen.
- `full head rounds`: rounds with a grammar mask, a penalty or more than 16 columns.
