# head_dim and perf: KDA, and the other attention families

Two questions from Jet Jiang, 2026-09-11, on behalf of a customer co-designing
Ling 3 MAX (training just started, using our co-design playbook):

1. They saw no significant perf gain from changing head_dim on their KDA model.
   Is that expected?
2. Have we studied the head_dim effect on the recent linear or sparse attentions?

**Yes to the first, and it is expected for a reason that generalises: at
moderate sequence length every attention family we can price shows exactly
1.000x, because the whole attention component hides inside the FSDP comm
envelope. At MAX scale the envelope is deeper, not shallower.** Answer to the
second is below: the head_dim law differs per family, and linear attention runs
the opposite direction from softmax.

Modelled with perfsim. Scripts `scripts/kda_headdim_perfsim.py` (KDA depth) and
`scripts/headdim_across_attention.py` (cross-family). Ratios only. perfsim
assumes perfect overlap and prices the linear-attention cumsum as an MXU node;
on our OLMoE3 it predicts 1339 ms against 1752 ms measured, and puts KDA at 27%
of step against 6% measured. See `olmoe3-perf-report.md`.

**Sharing note.** The customer-facing results below use only open-weight presets
(Qwen3.5-397B, Qwen3-Coder-480B, DeepSeek-V3, gpt-oss-120B) and our own OLMoE3.
`kimi_k3`, `deepseek_v4_pro` and `qwen3_8_2p4t_a95b` are WATCH or PRIVATE in
`perfsim/PUBLIC_MANIFEST.py` and are deliberately not used, since Ant is a
competitor to those labs.

## The three structural terms

Everything below follows from which of these dominates. Sweeping head_dim at a
**fixed total head width**, so `num_heads x head_dim` is invariant and neither
the projections nor the parameter count move:

| term | scaling | direction as you add heads |
|---|---|---|
| QK^T and AV matmuls | `num_heads x seq^2 x head_dim` = `seq^2 x W` | invariant |
| score tensor, softmax and mask | `num_heads x seq^2` | grows |
| linear-attention recurrent state | `num_heads x d_k x d_v` = `W_k x W_v / num_heads` | shrinks |

So softmax-family attention gets **more** expensive as you split into more,
narrower heads, and a linear-attention recurrence gets **cheaper**. Anything
that caps the score length, a sliding window or a sparse top-k selection, kills
the `seq^2` term and drives the sensitivity toward zero.

## Cross-family result

Training step time, head_dim swept at fixed total head width, `vs base` for the
0.5x head_dim / 2x heads arm and the 2.0x head_dim / 0.5x heads arm.

| family | seq 8192 | seq 32768 | seq 131072 |
|---|---|---|---|
| linear KDA (ours, 3.5B active, 128 dev) | 1.000x / 1.000x | **0.723x** / 1.554x | **0.739x** / 1.523x |
| linear GDN (Qwen3.5-397B, 512 dev) | 1.000x / 1.000x | 1.000x / 1.111x | **0.939x** / 1.122x |
| softmax GQA (Qwen3-Coder-480B, 512 dev) | 1.000x / 1.000x | 1.698x / **0.651x** | 1.905x / **0.548x** |
| MLA (DeepSeek-V3, 512 dev) | 1.011x / n/a | 1.693x / n/a | 1.891x / n/a |
| sliding window (gpt-oss-120B, 50% SWA, 128 dev) | 1.000x / 1.000x | 1.549x / 0.868x | 1.871x / 0.565x |
| sparse top-k 2048 (proxy, 512 dev) | 1.000x / 1.000x | 1.087x / 1.000x | 1.129x / 0.936x |

Read the bold cells: the two linear families want **more** heads, every
softmax-derived family wants **fewer**. MLA cannot express the 2.0x arm because
it already has a single latent KV head.

Three things fall out.

**At seq 8192 every family is 1.000x.** All six, both directions. This is the
direct answer to the customer's observation, and it is not KDA-specific.

**Sparse selection nearly removes the sensitivity.** Capping the score length at
2048 takes the softmax swing at 131072 from 1.905x / 0.548x down to 1.129x /
0.936x, because only the residual `heads x seq x K` term survives. If Ling 3 MAX
pairs KDA with a sparse or windowed full-attention layer, the softmax half stops
fighting the linear half, and the net head_dim effect gets smaller still.

**Hybrids dilute it.** Qwen3.5-397B is a linear/softmax hybrid, so its linear
arm moves only 0.939x where our linear-heavy OLMoE3 moves 0.739x. At 131072 it
has 20.3 s of softmax attention against 9.4 s of linear, and the two respond in
opposite directions.

## Why seq 8192 is flat: the comm envelope

At seq 8192 on our OLMoE3, `KDA ms + exposed comm ms` is constant to four figures
across a 5.4x swing in KDA time, and the step is exactly 1339.4 ms every time.

| arm | KDA ms | exposed comm ms | sum | step ms |
|---|---|---|---|---|
| 4 heads (512, 1024) | 678.7 | 353.1 | 1031.8 | 1339.4 |
| base 8 heads (256, 512) | 362.3 | 669.6 | 1031.9 | 1339.4 |
| 16 heads (128, 256) | 204.0 | 827.8 | 1031.8 | 1339.4 |
| 32 heads (64, 128) | 124.9 | 906.9 | 1031.8 | 1339.4 |

There is about 1032 ms of FSDP expert-weight traffic per step, fixed regardless
of head shape, and the attention component hides underneath it. You could take
KDA to zero and the step would not move.

**This gets worse at MAX scale, which is the relevant point for Ling 3.** Comm
as a share of step at seq 8192, same pure-FSDP plan:

| model | comm ms | step ms | comm share |
|---|---|---|---|
| OLMoE3 3.5B active | 669.6 | 1339.4 | 50% |
| Qwen3-Coder 480B | 5312.8 | 8093.8 | 66% |
| Qwen3.5 397B | 4697.2 | 6628.7 | 71% |
| DeepSeek-V3 | 8554.9 | 11878.7 | 72% |

Bigger model, deeper shadow, less head_dim sensitivity. A MAX-scale model is the
worst place to expect a head_dim win.

The envelope is fixed per optimizer step while the attention cost is linear (or
quadratic) in tokens per step, so the shadow ends once attention exceeds it. For
our OLMoE3 that is between **16K and 24K tokens per device per step**. Past it
the same change is worth up to 1.7x, and the fast arms simply fall back onto the
comm floor rather than going below it.

## The mechanism, from perfsim's own nodes

Dominant KDA node is `LinAttn_StateUpdate_fwd`, **VPU-bound** (`t_vpu_us` is the
driver, not HBM and not MXU), 512 invocations, `ops_per_element = 3584` at base.
Forward ms per layer at seq 32768, against the `1/num_heads` prediction:

| arm | StateUpdate fwd ms | vs base | predicted |
|---|---|---|---|
| 4 heads (512, 1024) | 53.16 | 1.98x | 2.00x |
| base 8 heads (256, 512) | 26.79 | 1.00x | 1.00x |
| 16 heads (128, 256) | 13.60 | 0.51x | 0.50x |
| 32 heads (64, 128) | 7.01 | 0.26x | 0.25x |

Softmax at the same fixed total width, compute-bound rather than VPU-bound, goes
the other way: 3.18 / 6.35 / 12.71 ms at 8 / 16 / 32 heads, exactly `num_heads`.

Within KDA, **`value_head_dim` is the sensitive knob and `key_head_dim` is not**,
because the state is `d_k x d_v` per head but the value side also drives the
output projection. At seq 32768, holding heads at 8:

| change | seq 8192 | seq 32768 |
|---|---|---|
| `value_head_dim` 512 to 1024 (expand_v 4) | 1.330x | 2.712x |
| `value_head_dim` 512 to 256 (expand_v 1) | 1.000x | 0.586x |
| `key_head_dim` 256 to 128 | 1.000x | 0.991x |
| `key_head_dim` 256 to 512 | 1.000x | 1.017x |

These are not iso-param; they are included because they are the change people
actually make when they say "we changed head_dim".

## Decode, for completeness

Batch 64, tp=4 chosen so the base's 4 KV heads shard exactly. Step ms.

| arm | ctx 4096 | ctx 8192 | ctx 32768 | ctx 131072 |
|---|---|---|---|---|
| base | 10.1 | 10.6 | 13.4 | 24.9 |
| KDA 32 heads (64, 128) | 0.929x | 0.932x | 0.947x | 0.971x |
| KDA 4 heads (512, 1024) | 1.095x | 1.090x | 1.071x | 1.038x |
| attn 16h x 128 | 1.000x | 1.000x | 1.000x | 1.000x |
| attn 4h x 512 | 1.046x | 1.089x | 1.283x | 1.613x |

KDA head_dim moves decode by at most 7%, and the effect **shrinks** with context,
because the KDA state is context-independent (flat 1.0 to 2.7 ms at every
context) while `KV_cache_read` grows 0.5 to 15.3 ms.

**The one decode effect worth acting on is divisibility, not head_dim.** Softmax
head_dim at fixed width is exactly 1.000x as long as the KV heads divide the TP
degree. When they do not, perfsim clamps `kv_heads_dev = 1` and the cache
replicates: `attn 4h x 512` has 2 KV heads, cannot split across tp=4, and pays
1.61x at 128K. The same thing hits the *base* config at tp=8, where 4 KV heads
do not shard across 8 devices, making an unrelated arm look like a 0.633x win.
That is a sharding artifact, and it disappears at tp=4. Worth asking whether the
customer's serving TP degree divides their KV head count, since it is a real 2x
and easy to misattribute.

## Caveats

Pure FSDP, `tp=1`, `ep=1`, `pp=1`. Real MAX-scale training uses expert and
tensor parallelism, which changes the size of the comm envelope. The structural
laws (`1/num_heads` for linear, `num_heads` for softmax) are
parallelism-independent; the depth of the shadow is not, so the "1.000x at seq
8192" result should be restated against their actual parallelism plan before it
is quoted as a number rather than as a mechanism.

perfsim does not model sparse top-k selection in training. Its DSA indexer
fields (`index_n_heads`, `index_head_dim`, `index_topk`) are informational only
(`plugins/models/dsv4_pro.py`), and `compress_ratios` applies at decode only. The
sparse row above is therefore a **proxy**: a score-length cap of 2048 on every
layer, which is FLOP-equivalent to selecting 2048 keys for the attention body.
It omits the indexer cost, which is a separate head set that head_dim does not
touch.

## What to ask them

Tokens per device per step is the single most informative number, since the
answer flips between 16K and 24K for a linear-attention model of this class.
Then: what fraction of their step is exposed collective time (if they are
comm-bound, no attention-shape change will show up at all), whether they held
total head width fixed or just changed head_dim (the latter moves parameter
count and is a different experiment), whether the full-attention layers are
dense, windowed, or sparse, and their serving TP degree against KV head count.
