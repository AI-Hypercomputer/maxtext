# OLMo 3.5 partner family: perfsim estimates on Ironwood

Estimates for the four rungs of
`allenai/OLMo-core@codex/partner-model-family-20260914`, produced with
`scripts/olmo35_perfsim.py`. The MaxText implementation of these models matches
the reference logits to 1.2e-05 relative (`tests/unit/olmo35_vs_reference_test.py`),
so the geometry priced here is the geometry that runs.

perfsim has no preset for this family. Each rung is built from the shipped
`olmoe3_3p5b` preset with the family's geometry: head_dim 128, latent = d_model/2
(`moe_input_bottleneck` 2.0), expert hidden = d_model, 512 experts top-16, a
full-attention layer every 8th block, and a dense first block at 8 x d_model.

**Read these as an upper bound.** perfsim assumes perfect compute/communication
overlap and roofline-perfect GEMMs. On our measured OLMoE3-3p5b it predicts
1.339 s against 1.752 s measured at the same operating point, so it runs about
**1.7x optimistic**. Divide the MFU column by roughly 1.7 for a planning number.

## Training

Global batch held at **16,777,216 tokens** (2048 sequences at seq 8192) for every
rung. That is the one training number the spec actually validates (Tiny's hero
recipe); batch is TODO for the other three, and holding it constant is what makes
the rungs comparable. Per-device batch therefore falls as the slice grows:
16 / 16 / 4 / 1.

| rung | active | total | slice | dev | ep | step s | TF/s/dev | MFU | tok/s/dev | kda% | moe% | comm% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 0.79B | 12.5B | v7x 4x4x4 | 128 | 1 | 1.672 | 435 | 37.7% | 78,371 | 50.8% | 42.7% | 0.0% |
| small | 3.78B | 72.2B | v7x 4x4x4 | 128 | 1 | 7.785 | 495 | 42.9% | 16,836 | 57.5% | 38.2% | 0.0% |
| medium | 15.42B | 322.6B | v7x 4x8x8 | 512 | 1 | 5.855 | 677 | 58.7% | 5,596 | 51.0% | 44.0% | 1.1% |
| large | 62.13B | 1310.2B | v7x 8x8x16 | 2048 | 1 | 20.997 | 192 | 16.7% | 390 | 10.6% | 18.9% | 69.5% |
| large | | | | | **32** | **5.191** | **776** | **67.3%** | **1,578** | 43.1% | 39.5% | 13.7% |

Three things fall out.

**Only Large needs expert parallelism.** Tiny, small and medium reach comm 0.0%
to 1.1% under plain FSDP: at a 16.7M-token batch there is enough compute per step
to bury the expert-weight all-gather completely, and adding `ep` only buys
all-to-all traffic it does not need. Large cannot do this, and the reason is
arithmetic rather than architectural: 16.7M tokens over 2048 devices is already
`per_device_batch_size = 1`, so the batch cannot grow to cover 1.31T parameters
of all-gather. Expert parallelism is what rescues it, worth **4.0x** (21.0 s to
5.19 s, comm 69.5% to 13.7%).

**This is a batch-size effect, not a property of the architecture.** At
`per_device_batch_size = 1` every rung looks comm-bound (tiny 50.0% comm, small
58.3%, medium 64.1%) and expert parallelism looks like a 2.6x to 3.8x win across
the board. That conclusion is an artifact of the small batch. Tiny at pdb=1 is
14.5% MFU and at pdb=16 is 37.7%, with comm going 50.0% to 0.0% and ep=1 beating
ep=2 once the batch is real. Any parallelism recommendation for this family has
to state the batch it was measured at.

| tiny, 128 devices | ep=1 | ep=2 | ep=8 |
|---|---|---|---|
| pdb 1 (1.05M tokens) | 0.272 s, 14.5%, comm 50.0% | **0.126 s, 31.2%** | 0.137 s, 28.6% |
| pdb 4 (4.19M tokens) | **0.441 s, 35.7%, comm 0.0%** | 0.445 s, 35.2% | 0.522 s, 29.9% |
| pdb 16 (16.8M tokens) | **1.672 s, 37.7%, comm 0.0%** | 1.741 s, 36.0% | 2.069 s, 30.2% |

**KDA is the largest single component once comm is hidden**, at 43% to 58% of the
step. That is perfsim pricing the delta-rule cumsum as an MXU node, which our own
profiling says is pessimistic by roughly 4x (perfsim puts KDA at 27% of the
OLMoE3 step against 6% measured, after the kernel fix). Treat the kda column as
"where perfsim thinks the time is", not as a profile. It does mean the KDA kernel
is the component whose real cost most needs measuring on this family, since the
estimate is least trustworthy there.

Per-device throughput falls roughly as active parameters rise, as it should:
78,371 / 16,836 / 5,596 / 1,578 tokens/s/device against active params of
0.79 / 3.78 / 15.42 / 62.13 B. Cross-checking `6 x active x tokens` against the
reported FLOPs agrees within the remat overhead at every rung, which is the
evidence that the perfsim geometry is configured correctly.

## Decode

Batch 64, context 8192. `tp` always divides `num_kv_heads`, otherwise the KV
cache silently replicates and costs up to 2x (see `kda-headdim-answer.md`). `ep`
spreads the routed experts so the resident weights fit.

| rung | tp | ep | dev | step ms | tok/s | tok/s/dev | kda ms | attn ms | moe ms |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 4 | 1 | 4 | 2.43 | 26,355 | 6,589 | 0.54 | 0.16 | 1.51 |
| small | 8 | 1 | 8 | 6.89 | 9,285 | 1,161 | 1.37 | 0.42 | 4.46 |
| medium | 4 | 8 | 32 | 12.04 | 5,316 | 166 | 3.73 | 1.94 | 5.54 |
| large | 8 | 8 | 64 | 20.11 | 3,182 | 50 | 5.19 | 2.45 | 11.25 |

Decode per-device throughput falls **132x** from tiny to large while active
parameters rise only 78x, because decode is bound by resident weight bandwidth
and that scales with **total** parameters (105x), not active. This is the cost of
holding experts and top-k fixed while growing capacity: Large stores 1.31T
parameters, about 2.62 TB in bf16 before KV cache, and every one of them has to
be resident somewhere. The MoE column is 47% to 56% of each decode step and is
almost entirely weight movement.

## Measured on hardware

**Ironwood was not obtainable.** `cloud-tpu-multipod-dev` had zero RUNNING tpu7x
instances; the flex-start pools on `tpu7x-cluster-flex` went into Backoff with
"Scale-up timed out after 15m" and the tpu7x nodes that lived there a week ago
have been replaced by v6e. In `cloud-tpu-shared-capacity`, the only two tpu7x
nodes (`bodaborg-tpu7x-sps`) are held by a live SPS Pathways service, and the one
cluster with spare Ironwood (`bodaborg-tpu7x-reh-sc`) denies pod create. So the
Ironwood table above stays an estimate.

What could be measured is **v6e (Trillium), 8 chips, bf16 peak 918 TFLOP/s/chip**,
running the real MaxText olmo35 path. Two limits forced a proxy. Full
`olmo35-tiny` does not fit: 47.75 GB of HLO temporaries against 31.24 GB
available, and still 31.45 GB at seq 2048. The per-device activation working set
at seq 8192 with full remat is about 26 GB on top of a 22 GB parameter and
optimizer shard, which is precisely why this family targets Ironwood's 95 GiB.
So the numbers below are a **depth-reduced proxy** at the exact per-layer
geometry, 8 layers instead of 16, which preserves the 1-in-8 full-attention ratio
and every GEMM shape.

| config | step s | TF/s/dev | MFU | tok/s/dev |
|---|---|---|---|---|
| 8 layers, seq 8192, KDA unfused | 0.620 | 32.4 | 3.5% | 13,225 |
| 8 layers, seq 8192, KDA fused | 0.591 | 33.9 | 3.7% | 13,850 |
| 4 layers, seq 8192, KDA unfused | 0.619 | 19.9 | 2.2% | 13,229 |
| 4 layers, seq 8192, KDA fused | 0.592 | 20.9 | 2.3% | 13,840 |
| 8 layers, seq 4096, KDA unfused | 0.440 | 22.6 | 2.5% | 9,300 |
| 8 layers, seq 2048, KDA unfused | 0.356 | 13.9 | 1.5% | 5,756 |

**This measures a regime nobody would train in, and the headline 3.5% must not be
compared against the estimates above.** The batch here is 8 devices x 1 x 8192 =
**65,536 tokens**, against the spec recipe's **16,777,216**, a factor of 256. The
evidence that the measurement is overhead-bound rather than architecture-bound:

Wall time does not depend on depth. Halving the layers from 8 to 4 leaves the
step at 0.619 s against 0.620 s unfused, and 0.592 s against 0.591 s fused, while
the analytic FLOPs fall by 1.6x. The layers are contributing almost nothing to
wall time.

The fused KDA kernel buys 5%. If the delta rule were the bottleneck, turning on
`use_tokamax_kda` would move much more than 0.620 s to 0.591 s. It does confirm
the fused kernel works at this family's `dk=128 / dv=256`, which was previously
untested.

Step time against sequence is 0.356 / 0.440 / 0.620 s at 2048 / 4096 / 8192,
implying a fixed floor near 0.27 s, about 44% of the step.

All three point the same way. At 65,536 tokens each of the 512 experts receives
`8192 x 16 / 512 = 256` rows per device, so every expert GEMM is launch-bound,
and the per-step fixed costs (embedding, the 100k-vocab LM head, collectives,
host sync) dominate. Raising the batch is the fix and is not possible here:
`per_device_batch_size=2` OOMs even at 4 layers.

This is the same effect the estimates section documents, seen on real silicon at
a more extreme operating point. perfsim puts tiny at 14.5% MFU at pdb=1 and 37.7%
at pdb=16; the v6e run sits below even the pdb=1 point because it has 8 devices
rather than 128 and a third of Ironwood's memory bandwidth per FLOP. The measured
numbers therefore corroborate the **direction** of the batch-size finding and
say nothing about the absolute Ironwood MFU.

## Caveats

Pure FSDP on the training side apart from the expert-parallel sweep, no tensor,
context or pipeline parallelism, and no attempt to tune the mesh per rung. The
slices are chosen so the weights plus fp32 master and Adam moments (14 bytes per
parameter) fit, not because they are the right production topologies.

Medium and Large are untrained proposals in the spec, with batch, peak LR, token
budget, device count and recomputation all marked TODO. Their step times are
priced at an assumed batch, so they answer "what would this cost per token at the
Tiny recipe's batch", not "what will this run at".

`use_fp8_gemm` is off everywhere, matching the spec's `fp8_or_mxfp8: False`.
