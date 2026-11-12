# How it works

Speculative decoding in one paragraph: a cheap drafter proposes the next few tokens, the full model checks all
of them in one forward pass (a *round*), and the longest correct prefix is kept plus one token of the model's
own. A round costs about the same whatever the text, so speed is *tokens kept per round* divided by *time per
round*. Everything below moves one of those two numbers.

On the Spark a round of Qwen3.8 Flash Next takes about 65 ms. Stock ninfer drafts with the model's MTP head,
3 tokens a round, so it tops out at 4 tokens a round.

## Where a round's time goes

Measured with `nsys` on one stream (65 ms a round, the GPU busy for 64 of them):

| Part | ms a round |
|---|---:|
| Expert layers (grouped NVFP4 experts, shared expert, routing) | about 25 |
| Other target layers (attention, norms, state records) | about 14 |
| GDN input projections (8-bit, 42 MB a layer, 36 layers) | 7.8 |
| Output head (BF16, 1.27 GB) | 5.8 |
| Drafter | about 4 |
| One MTP step kept in a drafter round | 2.3 |

The large kernels already run at the speed of the memory. This Spark reads 239 GB/s at best (ninfer's own
bandwidth probe); the output head reads at 222 GB/s and the GDN projections at 194 to 206. Under load the GPU
draws 20 W at 48 °C with no throttling: it waits on memory, not on compute. A round is about 12 GB of weights
read at memory speed, so better kernels could gain under 5%. The gains had to come from reading less per round
or keeping more tokens per round.

## 1. The DFlash2 drafter (patch 0003)

[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) is a small
block-diffusion drafter: it reads hidden states of the target at five layers and proposes a block of tokens in
one pass. ninfer already runs DFlash2 drafters for other models, but its fused kernels are specialized for
hidden sizes 5120 and 4096; Flash Next is 2560.

- The drafter runs on new linear shapes for hidden 2560 and small glue kernels (`src/ops/flash_dflash`),
  reusing ninfer's attention and selector operators.
- ninfer stores a drafter inside the model's file set, and adding one the stock way means reconverting 460 GiB
  of source checkpoints. `tools/graft_dflash2.py` instead writes a new entry file and one extra part holding
  the drafter, with hard links to the existing parts.
- With `NINFER_FLASH_DFLASH2` unset, the same files run plain MTP.

At 7 drafts this is +4 to +19% over MTP at 7 drafts.

## 2. Spark 8-bit layout for the drafter (patch 0004)

A proposal is bounded by reading the drafter's weights. `graft_dflash2.py --q8` stores the large matrices as
`q8_g32_fp16` (8-bit with a scale per 32 columns, 0.5% relative error; 776 MB instead of 1.2 GB) and puts
projections that read the same input into one weight: `[key; value; query]` and `[gate; up]`. Three fused
kernels do the rest: context K/V store, Q/K/V split with norm and rotation, packed SwiGLU.
+2 to +5.5%, with the same drafts kept.

## 3. Automatic draft width (patch 0005)

Verifying 7 drafts when only the first 2 are likely right wastes 5 columns of the round. With `p_i` the
drafter's probability for draft `i`, the round verifies the count `e` that maximizes expected tokens per cost:

```
(1 + sum over j <= e of prod over i <= j of p_i^power) / (cost + e + 1)
```

`cost` is the fixed part of a round in verify columns (17 on the Spark: about 44 ms fixed over 2.6 ms a
column). `power` corrects the drafter's optimism; calibrated on the bench, power 1 predicted 3,959 kept drafts
against 3,298 actually kept, power 2 predicted 3,213 against 3,265. Each width is its own captured CUDA graph.
Prose gains the most (+15%), since its drafts are the least reliable.

## 4. Several streams (patches 0006, 0007)

- Each stream has its own drafter state, and all rows of a round draft in one batched pass.
- A stream that resumes after a cached prompt prefix drafts from the context it has, instead of falling back to
  MTP for the next 2,048 tokens.
- **Each stream verifies its own number of drafts.** Measured first: at 8 rows a round costs 200 ms with 64
  columns and 121 ms with 16, but 125 ms with the full 64-column layout when only the *expert layers* run on 16.
  Nearly all the cost of a column is the expert layers. So several rows keep the full layout and the expert
  layers run only on the columns in use (gather, experts, scatter). The round takes every row's first draft,
  then the remaining drafts best first, and stops at the best expected tokens per cost.

A stream of easy text (a copy) no longer pays for a neighbour writing prose: at 8 streams it runs at 46 tok/s
against 29 with one window shared by the round.

## 5. Windows of 15 and copy extension (patch 0008)

The drafter drafts 7 tokens. When the model is repeating text it has already seen (renaming a parameter across
a file, quoting, returning a document with a small change), more than 7 can be right.

With `NINFER_FLASH_DFLASH2_COPY=8` the session finds the earlier stretch of the sequence that the newest 8
tokens repeat and sends the tokens that followed it as a hint. If the drafter's 7 drafts equal the first 7 of
the hint, the round continues with the hint's next tokens, up to 15. The hint keeps following its source
across a changed token or two (a rename): the position is kept from the last round and may shift by up to two
tokens while 12 of the newest 16 tokens agree.

A wrong hint costs nothing, because the drafter has to agree first, and every token is still verified. A match
length of 4 gave false extensions on JSON; 8 did not.

Exact copy 113 → 181 tok/s, file edit 111 → 136. The price: twice the verify graphs (start takes 95 s instead
of 55) and about 4% at 8 streams, where the layout is twice as wide.

The same patch stops a drafter round from projecting the unused MTP head and makes the drafter's top-16 run in
32 segments per column.

## 6. Output head screen (patch 0009)

The output head is a 248,320 × 2,560 BF16 matrix, 1.27 GB, read in full every round (5.3 ms). But ninfer's
sampling never looks past the 20 largest logits: `top_k` is capped at 20, and `top_p` / `min_p` work inside
those. So the round only needs the top 20 of each position, exactly.

1. At start the program makes an 8-bit copy of the head (`q8_g32_fp16`, 0.68 GB) and records each row's error
   norm `|w - w8|`.
2. Each round projects the copy (2.7 ms), giving approximate logits `a`.
3. By Cauchy–Schwarz the true logit of row `v` is within `|w_v - w8_v| · |h|` of `a_v`, plus a margin for BF16
   rounding. A row whose upper bound is below a value that at least 20 rows surely reach cannot be in the top
   20.
4. The rows that remain, about 50 a position, get their exact dot product from the BF16 head. All other rows
   are published as -1e30. Accept and sampling are unchanged.

Why not the model's 4-bit MTP head, which is the same matrix and is already loaded: its error is 11% a row,
the bound comes to 13 logits, and every token stays a candidate. With 8 bits and a scale per 32 columns the
error is 0.55% and the bound is about 0.65 logits.

The screen is used when no row of the round has a grammar mask or a presence/frequency penalty (those can lift
a token from outside the top 20) and the round has at most 16 columns. Other rounds run the full head.

A round goes from 67.6 to 64.9 ms, +3 to 6%. A published value is the FP32 dot product rounded to BF16, summed
in another order than the full head's kernel, so its last bit can differ; in the check run that happened for 2
of about 137,000 top-20 values and never changed a first token.

## Prompt lookup and longer MTP windows (patches 0001, 0002)

The first two patches predate the drafter: MTP windows up to 10, and prompt-lookup rounds
(`NINFER_PROMPT_LOOKUP=16`) that draft by copying what followed an earlier match. They remain as the base the
drafter rounds ride on; with the drafter on, copy extension does the lookup's job.

## What was measured and not built

- Skipping the drafter while a copy runs would save about 4 ms of such a round.
- Windows wider than 15 would lift exact copies further and cost more start-up time and multi-stream speed.
- Tuning kernels: under 5% in total, see above.
