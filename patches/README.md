# Patches

For [Neroued/ninfer](https://github.com/Neroued/ninfer) at
[`13ebb92`](https://github.com/Neroued/ninfer/commit/13ebb92f6f1d56b676e5300cb19555b2ba24053a). Apply in order
with `git am` (`scripts/setup.sh` does it). Each builds and runs on its own with the ones before it.

| # | Patch | What it adds | Switch |
|---|---|---|---|
| 1 | `0001-…-MTP-draft-windows-up-to-10` | MTP draft windows up to 10 on Flash Next | `--draft-tokens` |
| 2 | `0002-…-prompt-lookup-rounds` | Draft by copying what followed an earlier match (MTP path) | `NINFER_PROMPT_LOOKUP` |
| 3 | `0003-…-DFlash2-drafter-rounds` | The DFlash2 drafter for hidden size 2560; BF16 drafter, one stream | `NINFER_FLASH_DFLASH2` |
| 4 | `0004-…-Spark-8-bit-drafter-layout` | 8-bit drafter weights, shared projections, three fused kernels | file set made with `--q8` |
| 5 | `0005-…-automatic-draft-width` | Verify only the drafts that pay off | `NINFER_FLASH_DFLASH2_AUTO` |
| 6 | `0006-…-several-streams` | Drafter state per stream, batched proposals, drafting after a cached prefix | |
| 7 | `0007-…-each-stream-verifies-its-own-draft-count` | Own draft count per stream; expert layers only on columns in use | `NINFER_FLASH_DFLASH2_AUTO_ROW` |
| 8 | `0008-…-windows-up-to-15-with-copy-extension` | Verify windows up to 15, copy hint from the session, cheaper drafter top-16 | `NINFER_FLASH_DFLASH2_COPY` |
| 9 | `0009-…-output-head-with-an-8-bit-screen` | 8-bit screen of the output head with exact top 20 | `NINFER_FLASH_HEAD_SCREEN` |

Mostly touched: `src/models/qwen4_exp/` (program, session, execution), `src/ops/flash_dflash/`,
`src/ops/head_screen/`, and new shape tables under `src/ops/linear/projection/{bf16,q8}/`.

A workflow in this repository checks on every push that the series still applies to the pinned commit.
Moving to a newer ninfer means rebasing the series; expect conflicts in `src/models/qwen4_exp/program/`.
