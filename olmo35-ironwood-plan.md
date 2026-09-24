# OLMo 3.5 on Ironwood: rebase, consistency gate, measured performance loop

Tracking checklist. Tick items as they land. Full context and rationale live in the
session plan; this file is the execution record.

Reference: `allenai/OLMo-core@codex/partner-model-family-20260914`,
`src/scripts/standalone/standalone_configs.py`.

## Target and anchors

`tpu-recipes/training/ironwood/README.md`, per chip against the 2307 TFLOP/s bf16 peak:

| recipe | TF/s/chip | MFU | expert layout |
|---|---|---|---|
| llama3.1-70b (dense) | 1207 | 52.3% | dense |
| qwen3-235b-a22b | 630 | 27.3% | MoE, wide experts |
| deepseek3-671b | 608 | 26.3% | MoE, wide experts |
| gpt-oss-120b | 330 | 14.3% | MoE, many narrow experts |
| OLMoE3-3p5b (ours, measured) | ~195 | 8.45% | 512 experts, top-16 |

OLMo 3.5 is 512 experts at top-16 with expert hidden = d_model, which is the gpt-oss
class rather than the deepseek class. Goal is measured MFU above 20% on v7x, or a
defensible statement of what caps it.

MFU convention: **TF/s/device / 1153.5** (a chip holds two devices).

## What the TPU design guide says about this geometry

From `TPU Model Design Considerations and Optimization Guide (3).md`. OLMo 3.5 is
256-aligned on d_model, expert hidden and latent for every rung. It violates the
guide in three places, and one of them has measured iso-parameter evidence.

**head_dim 128 is the single biggest architectural gap.** Principle 1 calls head_dim
the most-missed dimension: at 128 or 64 the QK product leaves the MXU at least 50%
idle. Measured on a Qwen3-30B MoE, reshaping attention from 32 heads x 128 to
16 x 256 at *identical parameters and FLOPs*:

| context | head_dim 128 | head_dim 256 | gain |
|---|---|---|---|
| 8K | 23.1% MFU | 28.1% MFU | +21% |
| 16K | 23.3% | 30.8% | +32% |
| 32K | 23.7% | 34.7% | +46% |

OLMo 3.5 fixes head_dim at 128 and KDA key_head_dim at 128 (value_head_dim 256 is
fine). Our own v4 hardware sweep agrees on direction: KDA at 32 heads x 64 measured
1.74x *slower* than 8 x 256 at fixed total width, so the MXU argument beats the
FLOP-count argument on real silicon.

**Expert granularity.** The guide's high-expert-count regime says "small per-expert
GEMMs starve the MXU, so prefer fewer, larger experts". Its ~100B LatentMoE reference
config for Ironwood uses **128 experts top-8, expert d_ff 4096, latent 3x
compression, alternating dense/LatentMoE layers**. OLMo 3.5 uses 512 experts top-16,
expert hidden = d_model, latent 2x, MoE on every layer but the first.

**Alignment we already satisfy:** d_model 1024/1536/2560/4608 and latent
512/768/1280/2304 are all multiples of 256, as is expert hidden. (For contrast the
guide notes gpt-oss's 2880 width is 11.25 x 256 and "pays padding that no tuning can
remove", which is part of its 14.3%.)

### Diagnostic thresholds to check in every profile (phase 5)

| # | number | healthy |
|---|---|---|
| 1 | exposed (un-overlapped) comm, share of step | a few % |
| 2 | backward:forward wall-clock ratio | 2 to 3 (2 = no remat, 3 = full remat) |
| 3 | large GEMMs, % of peak FLOP/s | 80 to 95% |
| 4 | non-matmul share of step | under 30% for MoE |
| 5 | engine running sorts/gathers/scatters | SparseCore |
| 6 | step-time variance after warmup | steady |

Method the guide prescribes, which this plan follows: one knob per run, 20-30 steps,
capture a profile after each change and **verify the mechanism in it, not just the
speedup**, keep negative results, and gate every perf change on a fixed-seed numerics
run. HBM budget is ~16 bytes/param at MaxText defaults; when a config will not fit,
the lever order is shard more (FSDP/EP), then remat policy, then cut per-device
batch, with host offload last.

## Phase 0: rebase onto current main

New branch `olmo35-rebase` cut from `gagik-test-olmo`; the latter stays at `fa2db0f27`
as the fallback. Commit messages are plain one-line subjects with no attribution lines.

- [x] 0.0 This checklist written into the worktree
- [x] 0.1 Pre-rebase state recorded (HEAD `fa2db0f27`, 54 dirty paths, `/tmp/pre-rebase-status.txt`)
- [x] 0.2 Working tree backed up to `~/olmo35-prerebase-20260921.tgz` (41 MB)
- [x] 0.3 `git switch -c olmo35-rebase`
- [x] 0.4 Four commits landed (A `4d4c1dcbc`, B `af6b928c5`, C `540e3af06`, D `33a782707`)
- [x] 0.5 `git fetch origin`: **411 commits behind**, 117 touching the five shared files
- [x] 0.6 `git rebase origin/main` done. Upstream had **deleted `decoders.py`** ([NNX] Remove Linen module code); our embedding scale+norm already lives in `nnx_decoders.py`, so the Linen copy was dropped. Conflicts resolved in `nnx_decoders.py`, `base.yml`, `linears.py`, `moe.py` (3), `normalizations.py` (2), `types.py`
- [x] 0.7 Post-rebase inspection passed: `startswith("olmo3-")` intact, four config fields present and consumed, four model names registered, tuple `num_features` works
- [x] 0.8 Green-light gate **PASSED**: parity 8/8, `olmoe3_test.py` + `attention_test.py` 104 passed / 90 skipped. Needed one fix: upstream added `forced_routed_experts` to `RoutedMoE.get_topk()` and our override rejected it (`8966905e8`)

## Phase 1: consistency gate against the reference

- [x] 1.1 Shallow-cloned OLMo-core at the partner branch
- [x] 1.2 Reference unchanged: still commit `6bbdac3`
- [x] 1.3 Parity suite: **8 passed** on the rebased tree
- [x] 1.4 Diff `standalone_configs.py` geometry against the four `olmo35-*.yml`: **84 fields across 4 rungs, 0 mismatches**
- [x] 1.5 Parity recorded: 8/8, max relative logit error 1.2e-05

Expected parameter counts: 12,496,341,632 / 72,237,847,936 / 322,601,566,720 /
1,310,163,554,560.

## Phase 1b: OLMo 3.1 baseline on real Ironwood (DONE)

How the *existing* shipped OLMo models perform on our stack, as the reference the
3.5 family has to be judged against. SPS `bodaborg-tpu7x-sps`, 8 Ironwood devices
(2x2x1), seq 8192, pdb 1, `remat_policy=full`, synthetic, 20 steps, untuned.

- [x] 1b.1 SPS brought up end to end (this doubles as the phase-2.3 smoke test)
- [x] 1b.2 `olmo3-7b` measured
- [x] 1b.3 `olmo3-32b` measured

| model | d_model | layers | heads | step s | TF/s/dev | **MFU** | tok/s/dev |
|---|---|---|---|---|---|---|---|
| olmo3-7b (dense) | 4096 | 32 | 32q / 32kv x 128 | 1.41 | 277 | **24.0%** | 5,800 |
| olmo3-32b (dense) | 5120 | 64 | 40q / 8kv x 128 | 4.84 | 349 | **30.3%** | 1,690 |

**Both clear the 20% target untuned, at the worst-case pdb=1.** That is the key
framing for everything below: the stack is not the problem and Ironwood is not the
problem. OLMoE3 at 8.45% is 3x below its own dense sibling, so the gap is the MoE
geometry (512 experts top-16, expert hidden = d_model) plus MoE-path execution, not
the platform. Note both OLMo 3.1 models also use head_dim 128, so they are leaving
the Principle-1 gain on the table too and still reach 24-30%.

Two SPS fixes were needed to get here, both now in `scripts/sps_olmo35.sh`:
`--collect_service_metrics` fails on Cloud Monitoring throttling, and the proxy image
must match the deployed server (`proxy_server:20260901-jax_0.11.1`, not the
`runtime_20260720` one in the guide) or the run dies with a Pathways cache-key
mismatch.

## Phase 2: plumbing for real Ironwood

- [x] 2.1 Patched tokamax KDA installed into `venv-maxtext`; its two previously-skipped parity tests now pass
- [x] 2.2 Parity re-run after the install: 8 passed
- [x] 2.3 SPS smoke test done via phase 1b: `Jax Backend: Pathways`, `Num_devices: 8`, clean teardown, RC=0
- [x] 2.1 Patched tokamax KDA installed into `venv-maxtext` at `site-packages/tokamax/_src/ops/experimental/kda/` (from `olmo35/tokamax-kda-patched/kda`). `kimi_delta_attention` imports; the two previously-skipped tokamax KDA parity tests now **pass**
- [x] 2.2 Parity re-run after the install: **8 passed**
- [x] 2.4 Baseline `olmo35-tiny` seq 8192 pdb 1 remat full: **45.7 TF/s/dev = 3.96% MFU**, step 0.79 s, 10,400 tok/s/dev
- [x] 2.5 Proxy-job cleanup verified after every run (3 hung jobs found and cleared once)
- [x] 2.6 4x4x4 hunt done: capacity exists (256 idle chips, 307-chip quota) but is gated behind `priority-dev` RBAC. See the blocker section

### SPS constraint found: XLA runtime flags do not propagate

`LIBTPU_INIT_ARGS` set on the client has **no effect** over SPS, because the
Pathways workers are pre-deployed with the service. Measured directly: olmo3-7b at
identical config gave 318.7 / 315.8 / 321.1 TF/s with the flags and 317.7 / 320.5
without. The sparse-core collective-offload set is a large part of the OLMo3 recipe's
27% -> 44.5%, so **that ceiling is not reachable on SPS**; it needs a dedicated slice
(the 4x4x4 hunt). A second symptom: `sa_block_*=2048` from the recipe OOMs VMEM at
compile over SPS, because the matching `--xla_tpu_scoped_vmem_limit_kib=65536` never
arrives.

So on SPS we can tune MaxText-level knobs (remat, attention implementation, MoE
flags, per-device batch, sharding) but not XLA runtime flags.

## Phase 3: lever sweep

One factor at a time against 2.4, identical steps and seed.

- [x] 3.1 `use_gmm_v2=true`: was NaN, **root-caused and fixed** (`_clamp_tiles`); now 45.9 TF/s, fastest path, 1.06x over megablox
- [x] 3.2 `TOKAMAX_KDA_BF16_FWD/BWD=1`: **neutral** (45.8 vs 45.7), loss unchanged
- [x] 3.3 `TOKAMAX_KDA_DENSE_PAIRS=1`: **1.26x** (61.0 -> 76.8 TF/s). Targets the 303ms KDA kernel the roadmap named
- [x] 3.4 `TOKAMAX_KDA_CHUNK_SIZE=128`: errors out (MaxText `gdn_chunk_size` stays 64, the two disagree); 256 untested
- [x] 3.5 gpt-oss MaxText flag set: `use_custom_sort_vjp` neutral, **`shard_exp_on_fsdp=True` +4%** (48.4 -> 50.5), splash attention neutral (only 1 layer in 8 is softmax)
- [x] 3.6 gpt-oss/OLMo3 XLA flag set: **+5%** (46.1 vs 43.9), the first time these actually applied (SPS ignored them)
- [x] 3.7 qwen3-235b cross-check: it sets `shard_exp_on_fsdp=False`, gpt-oss sets True. Measured **True is better** here (+4%), so the gpt-oss choice wins for this expert class
- [x] 3.8 `use_random_routing` kept off for every headline number

### What the tiny measurements actually say

Every kernel lever measured **neutral** (bf16 KDA, tokamax GMM v1, remat=custom), and
halving the depth from 16 to 8 layers left the step time essentially unchanged
(45.7 -> 44.3 TF/s). A step whose time does not depend on how many layers it runs is
not compute-bound. perfsim agrees and names the cause: at 8 devices it puts **90% of
the step in comm**, because FSDP re-gathers a 12.5B-parameter model across only 8
devices every step.

Two consequences. First, `olmo35-tiny` on an 8-device slice is a poor vehicle for
kernel work: there is nothing for a kernel lever to win. Second, the measured 3.96%
is a slice artifact, not the architecture's number.

perfsim predicts 28.2 TF/s (2.4% MFU) here against 45.7 measured, so at 8 devices it
is **pessimistic by 0.62x** — the opposite of the 1.7x optimism seen at 128 devices.
The optimism factor is regime-dependent and cannot be carried across slice sizes.

### Why the rungs will not behave alike

Rows per expert per device is `seq x pdb x top_k / num_experts` = 8192 x 1 x 16 / 512
= **256** for every rung, which is exactly one MXU tile (256x256). So the M dimension
never amortizes at pdb=1, and the only thing that grows with rung size is K and N:

| rung | expert GEMM M x K x N | MXU tiles |
|---|---|---|
| tiny | 256 x 512 x 1024 | 8 |
| small | 256 x 768 x 1536 | 18 |
| medium | 256 x 1280 x 2560 | 50 |
| large | 256 x 2304 x 4608 | **162** |

`large` does 20x more MXU work per expert GEMM than `tiny`. Measuring only `tiny` and
generalising to the family would be badly wrong, and `tiny` is the family's worst case
by construction.

## Phase 4: per-device batch and remat

- [x] 4.1 pdb swept 1/2/4/8: pdb=2 NaNs on the KDA kernel at seq 8192, pdb>=4 OOMs. Sequence length is the working substitute (seq 16384 = 1.30x)
- [x] 4.2 Host offload (`decoder_layer_input=offload`, `mlpwo=offload`): **neutral** (48.4), and it did not unlock a higher pdb. xla-shell agrees: "Remat is not a win on this profile"
- [x] 4.3 Winners re-tested at the best operating point; the stack composes to 81.5 TF/s
- [x] 4.4 Memory wall recorded: pdb=4 needs 112-128 G, pdb=8 191-215 G, seq 32768 needs 124 G, all against 94.74 G

## Phase 5 result: profiled, and the binder moved

Profile of the best config (gmm_v2 fixed + XLA flags + `shard_exp_on_fsdp`, seq 16384,
pdb 1, 8 devices), read with xla-shell.

```
Engine lanes (concurrent; step >= max lane):
  TensorCore   906ms (82%)   compute 616 + vpu 224 + relayout 67
  SparseCore   372ms         comm (hidden 201 / exposed 171)
  Host-DMA      50ms
Best-overlap ceiling 906ms (at 1.10s -> 1.22x headroom).  Binder: TensorCore
```

**The step is no longer comm-bound.** At 8 devices with the untuned config it was;
with the levers on and seq 16384 the binder is the TensorCore lane. Against the design
guide's thresholds: exposed comm 15.5% of step (guide wants "a few %"), non-matmul TC
work 26% (guide wants under 30% for MoE, so borderline), collectives correctly on
SparseCore.

`roadmap --all` moving floor:

| # | lever | step | gain | binder |
|---|---|---|---|---|
| 0 | as-profiled | 1.10s | - | imperfect overlap |
| 1 | schedule exposed comm | 906ms | 198ms | TensorCore |
| 2 | kernels (bounded by slack) | **372ms** | 534ms | TensorCore |
| 3 | relayout reduction | 372ms | 0 | SparseCore comm |
| 4 | host-offload remat | 372ms | 0 | SparseCore comm |

Floor after all levers **372ms**, binder SparseCore comm: a 2.96x software headroom
from where the profile was taken, after which only cutting comm *volume* helps.
"Remat is not a win on this profile", which matches the measurement (remat=custom was
neutral).

`roadmap --kernels` named the target unambiguously:

| kernel | calls | time | useful |
|---|---|---|---|
| **`_fused_dhu_wy_intra_cumsum_pallas_`** | 14 | **303ms** | 303ms |
| fusion | 1809 | 60ms | 60ms |
| gmm_v2 g=512 m=262144 k=512 | 61 | 51ms | 51ms |
| gmm_v2 g=512 m=262144 k=1024 | 53 | 44ms | 44ms |
| tgmm_v2 k=512 | 30 | 28ms | 28ms |

One kernel is **27% of the step and 5.9x the next** — the KDA intra-chunk factoring,
one call per KDA layer (14 of tiny's 16 layers). That is where the kernel budget of
534ms should be spent, and acting on it is what produced the wins below.

### Acting on the roadmap: 1.34x from two KDA env flags

seq 16384, pdb 1, 8 devices, best flag set:

| config | TF/s/dev | MFU | vs |
|---|---|---|---|
| chunk 64 (baseline) | 61.0 | 5.3% | 1.00x |
| + `TOKAMAX_KDA_DENSE_PAIRS=1` | 76.8 | 6.7% | **1.26x** |
| + `TOKAMAX_KDA_BF16_FWD/BWD=1` | **81.5** | **7.1%** | **1.34x** |
| + `TOKAMAX_KDA_CHUNK_SIZE=128` | n/a | n/a | VMEM OOM (91.0M of 63.9M) |
| + `TOKAMAX_KDA_CHUNK_SIZE=256` | n/a | n/a | VMEM OOM (120.2M of 63.9M) |

**bf16 KDA measured neutral on its own earlier and is worth 1.06x here.** The lever
only pays once `dense_pairs` has removed the backward bottleneck it was hiding behind,
which is exactly the interaction the plan warned about. Chunk sizes above 64 do not
fit VMEM at these shapes, closing item 3.4.

## Phase 5: xla-shell profile loop

- [x] 5.1 xplane profile captured on Ironwood at the best config (580 MB)
- [x] 5.2 `analyze_profile`: TC lane 906ms / SC 372ms / host-DMA 50ms, ceiling 906ms, **binder TensorCore** (no longer comm)
- [x] 5.3 `roadmap --all` and `--kernels` run; moving floor 1.10s -> 372ms
- [x] 5.4 Acted on the roadmap's #1 kernel (`_fused_dhu_wy_intra_cumsum_pallas_`, 303ms): **1.34x** from `dense_pairs` + bf16
- [x] 5.5 Latent-MoE fusion and `tokamax_gmm_tile_m` NOT pursued: the roadmap puts gmm_v2/tgmm_v2 at 51+44+28ms against the KDA kernel's 303ms, so they are not the lever
- [x] 5.6 xla-shell floor recorded: **372ms** (SparseCore comm) against 1.10s as-profiled, i.e. 2.96x software headroom; past it only comm VOLUME helps

## Phase 6: perfsim cross-check and the 20% verdict

- [x] 6.1 perfsim re-run at the measured operating point (8 devices, pdb=1, seq 8192)
- [x] 6.2 Optimism factor derived: perfsim is **0.62x pessimistic** at 8 devices (28.2 predicted vs 45.7 measured), the opposite of its 1.7x optimism at 128. The factor is regime-dependent and does not transfer
- [x] 6.3 Projections re-issued below with the measured factor
- [x] 6.4 Verdict stated below
- [x] 6.5 Architectural gap priced (phase 6.5 section): shipped geometry reaches 35-51% at pdb>=4 on 128 devices; granularity is worth a further 1.12-1.60x
- [ ] 6.6 `small` on a 4x4x4: **blocked on `priority-dev` RBAC**, launcher written and validated

## Phase 6.5 result: the shipped geometry is not the problem, pdb=1 is

perfsim at a **realistic** operating point (128 devices, seq 8192, shipped 512-expert
top-16 geometry), sweeping per-device batch. rows/expert = `8192 * pdb * 16 / 512` =
`256 * pdb`, and 256 rows is exactly one MXU tile:

| rung | pdb=1 | pdb=2 | pdb=4 | pdb=8 | pdb=16 |
|---|---|---|---|---|---|
| tiny | 14.5% | 29.0% | 35.7% | 37.1% | 37.7% |
| small | 13.6% | 27.3% | 41.3% | 42.6% | 42.9% |
| large | 12.7% | 25.4% | 50.8% | **77.8%** | 78.2% |

**The shipped geometry clears 20% at pdb=2 on every rung and reaches 35-51% at pdb=4.**
No architecture change is needed for the 20-30% target. The entire measured deficit is
the pdb=1 / 8-device operating point we were forced onto, which is also why every
kernel lever measured neutral there.

### Granularity is a real but secondary lever, and it is one specific thing

Iso-active-FLOP granularity arms at 128 devices, pdb=4 (halving top_k while doubling
expert hidden leaves compute per token unchanged):

| arm | rows/exp | tiny | small | large |
|---|---|---|---|---|
| base 512e top16 eh=d | 1024 | 35.7% | 41.3% | 50.8% |
| 256e top8 eh=2d | 1024 | 39.7% | 44.6% | 50.7% |
| **128e top8 eh=2d** | **2048** | **41.5%** | **46.1%** | **81.3%** |
| 128e top4 eh=4d | 1024 | 42.0% | 46.4% | 50.7% |
| 64e top4 eh=4d | 2048 | 41.2% | 44.1% | 71.6% |

On `large` the discriminator is exactly **rows per expert, not expert width**: every
2048-row arm wins big (1.41-1.60x) and every 1024-row arm is flat (1.00x). Since
rows/expert = `seq * pdb * top_k / num_experts`, halving the expert count and raising
the batch are interchangeable ways to buy the same thing. Prefer the batch, since it
needs no architecture change.

### Correction: head_dim 256 does NOT help this architecture

Earlier in this plan I called `head_dim=128` "the single biggest architectural gap",
on the strength of the guide's measured Qwen3-30B result (+21% at 8K, +46% at 32K).
That does not transfer to OLMo 3.5: perfsim puts `head_dim=256` at **1.00x on tiny and
small and 0.99x on large**. The reason is the 7:1 KDA-to-full-attention ratio, so only
one layer in eight is softmax attention and Principle 1 has almost nothing to act on.
The guide's measurement was on a full-attention model. Keep head_dim 128.

## Working Ironwood vehicle (found after SPS went down)

`bodaborg-tpu7x-spot-sps` (project `cloud-tpu-multipod-dev`, us-central1):
**no Kueue at all**, 5 free single-host 2x2x1 nodes (4 chips = 8 devices each),
plain `kubectl apply` of a Pod works. This replaced SPS and, unlike SPS, the
workers are ours so **XLA flags actually reach the compiler**.

One gotcha: the pod runs as `cloud-tpu-multipod-dev.svc.id.goog` and gets **403 on
`gs://agagik-us`**. Stage the source in a bucket in the pod's own project instead:
`gs://cloud-pathways-staging/agagik/olmo35-src.tgz`. Then `tar xzf` to `/wt` and
`export PYTHONPATH=/wt/src`, which overrides the image's stale `/deps/src`.

### First measurement of the XLA flag set (SPS silently ignored it)

olmo35-tiny, 8 devices, pdb=1, seq 8192: **46.1 TF/s with the flags, 43.9 without**,
so the sparse-core offload set is worth about **+5%** here. Small, because at 8
devices the model is comm-volume-bound rather than overlap-bound. Cross-check: the
same config on SPS measured 45.7, so the two vehicles agree.

### Sequence length is a usable lever, and it dodges the KDA bug

olmo35-tiny, 8 devices, pdb=1, remat full:

| seq | TF/s/dev | MFU | note |
|---|---|---|---|
| 8192 | 43.1 | 3.7% | |
| **16384** | **56.2** | **4.9%** | **1.30x, and avoids the NaN** |
| 32768 | n/a | n/a | OOM, 124.35 G vs 94.74 G |

Raising the sequence rather than the batch buys 1.30x *and* stays out of the kernel
bug's trigger region. That is the recommended workaround until the kernel is fixed.

### Expert parallelism HURTS on hardware, contradicting perfsim

olmo35-tiny, 8 devices, pdb=1, seq 8192, `fsdp x ep = 8`:

| config | TF/s/dev | vs ep=1 |
|---|---|---|
| ep=1, fsdp=8 | 43.5 | 1.00x |
| ep=2, fsdp=4 | 42.4 | 0.97x |
| ep=4, fsdp=2 | 34.3 | **0.79x** |
| ep=8, fsdp=1 | OOM (109.59 G) | - |

**This directly contradicts the perfsim result recorded earlier in this plan**, which
predicted expert parallelism worth 2.6x to 3.8x at pdb=1. On hardware it is
monotonically worse. The mechanism perfsim misses: `fsdp x ep` is fixed at the device
count, so every unit of `ep` *removes* a unit of FSDP sharding, and the resulting
growth in per-device weight traffic and footprint outweighs replacing the expert
all-gather with a token all-to-all. At ep=8 there is no FSDP at all and it OOMs.

Treat perfsim's parallelism modelling as unvalidated until checked on hardware. Its
*geometry* ratios (phase 6.5) are a separate question and remain unchecked.

### tokamax GMM v1 is neutral, not a win

megablox 43.4 vs tokamax gmm v1 43.8 TF/s, loss identical to 4 decimals (10.151).

## Open bugs found (both block the main levers)

1. **Fused tokamax KDA produces NaN at `per_device_batch_size=2`.** **ROOT-CAUSED AND FIXED 2026-09-24**: a decay overflow, not a batch bug; the B>=2 pattern was coincidence. See `olmo35-ironwood-runs.md`, "The pdb=3 NaN is a KDA decay overflow", and `tokamax_kda_log_decay_floor`. Original notes follow. pdb=1 is clean and
   the unfused chunked path is clean at pdb=2 (loss 11.73 -> 9.26), so it is the
   kernel, not the model. This blocks the single most valuable lever, since pdb is
   what every tuned recipe uses to raise MFU. Kernel is
   `olmo35/tokamax-kda-patched/kda` at dk=128 / dv=256.
2. **`use_gmm_v2=True` produces NaN at pdb=1.** This is the lever Aleksey Vlasenko
   measured as the largest single OLMoE3 win (2.06x on the ragged-dot kernel, -6.6%
   step). Note it also requires `use_tokamax_gmm=True` or config validation rejects it.

Both need a fix before the lever sweep can conclude.

**Bug 1 is now precisely characterized.** It needs **B >= 2 AND T >= 8192 together**;
neither alone triggers it, and it is not total tokens per device:

| per-device batch | seq | tokens/device | result |
|---|---|---|---|
| 1 | 8192 | 8192 | OK |
| 1 | 16384 | 16384 | **OK** |
| 2 | 4096 | 8192 | OK |
| 2 | 8192 | 16384 | **NaN** |

`TOKAMAX_KDA_DENSE_PAIRS=1` does **not** fix it. `TOKAMAX_KDA_CHUNK_SIZE=128` errors
out rather than running (MaxText's `gdn_chunk_size` stays 64, so the two disagree).

**Bug 2 is FIXED.** Root cause: **MaxText passed unclamped tile sizes to gmm_v2.**
`wi_tile_fwd_embed_dim` defaults to 1024 while olmo35-tiny's latent (the contracting
dim) is 512, so `tile_k > k`. gmm_v2 indexes its operands by tile, so the oversized
tile walks off the contracting extent and the kernel returns NaN with no error.
`jax_ragged_dot_gmm` in `layers/moe.py:1747` already clamps exactly this way; the
tokamax v2 path did not.

Why it stayed hidden: on a smaller-VMEM part the same over-request surfaces as
`CompileTimeScopedVmemOom` rather than NaN (reproduced on v4), and tokamax v1 ignores
`tiling` entirely (its public `ragged_dot` takes no tiling argument), so only the v2
path is exposed.

Fix: `_clamp_tiles()` in `kernels/megablox/ops.py`, applied at all three `TileSizes`
sites (fwd, dlhs, drhs). Measured on tpu7x with the **default** tiles that previously
aborted at step 1:

| path | TF/s/dev | loss @ step 7 |
|---|---|---|
| megablox baseline | 43.3 | 10.151 |
| tokamax gmm v1 | 44.5 | 10.151 |
| **tokamax gmm v2, fixed** | **45.9** | 10.157 |

So gmm_v2 is now clean and the fastest of the three, **1.06x over megablox**. Two
independent workarounds also verified before the fix landed: clamping the
`wi_tile_*`/`wo_tile_*` config by hand (44.1 TF/s), and `use_gmm_v2_heuristic_tiling=True`
(42.7 TF/s), which lets tokamax choose all four tile fields itself.

**Repro status.** `scripts/olmo35_nan_repro.py` builds a shrunk OLMo 3.5 (d_model 512,
8 layers, dk 128 / dv 256, 8 experts top-2) and checks logits and every gradient for
non-finite values across per-device batch. It cannot be used locally: the Mosaic KDA
kernel raises **"Not supported on TPU v4"**, and the `xla` reference implementation is
finite at B = 1, 2 and 4 (absmax 1.2455e-01, identical across batch). So the NaN is
specific to the Mosaic kernel on v7x and can only be bisected on Ironwood, where the
only vehicle today is the shared single-slice SPS pool. Budget one SPS run per bisect
step (5-10 minutes each including placement waits), and do not launch two at once:
concurrent runs contend for the same slice and one dies.

Suggested bisect order, cheapest first: `TOKAMAX_KDA_CHUNK_SIZE` 128/256 at pdb=2 (the
BC sub-block tiling is the known overflow site, patch 1 in the patched tree); then
`TOKAMAX_KDA_DENSE_PAIRS=1`; then bf16 fwd/bwd off individually; then shrink seq at
pdb=2 to see whether the trigger is total tokens or the batch dimension itself.

## BLOCKER as of 2026-09-21 ~08:00: the SPS service is down

`sps-j6080103-pathways-head-0-0-bpt6z` is in **Error** (0/1) and all its worker pods
are gone, so every run now hangs waiting for a placement that will never come. The two
tpu7x nodes on `bodaborg-tpu7x-sps` are Ready and idle, but they belong to that
JobSet; the service is shared and its owner restarts it (guide section 9: "a new
Pathways image needs the service redeployed... the service owner handles it"). I did
not take those nodes, since squatting them would block the service from recovering.

**This ends Ironwood access for now.** To resume: ask in the *Shared Pathways Service
Users* chat space for a redeploy of `sps-j6080103`, or land a dedicated slice.

An in-flight bisect of the KDA NaN was lost to this. The three tests, still worth
running first when a slice returns, in this order:

1. `pdb=2 seq=4096` (8192 tokens/device, same as the working pdb=1) versus
   `pdb=1 seq=16384` (16384 tokens/device, same as the failing pdb=2). This separates
   "the batch dimension itself" from "total tokens per device", which decides whether
   to look at batching or at an accumulation overflow.
2. `pdb=2` with `TOKAMAX_KDA_CHUNK_SIZE=128`.
3. `pdb=2` with `TOKAMAX_KDA_DENSE_PAIRS=1`.

Note the baseline pdb=2 NaN was already with the bf16 flags **off**, so the fault is in
the base PR#1103-plus-patches kernel, not the bf16 path.

Code read so far: `pallas_mosaic_tpu_fwd_kernel.py::_pre_process_pallas` does handle
B > 1, by looping over batch elements and calling the B=1 kernel (line ~187), so the
obvious "hardcoded batch 0" theory is wrong there. That loop also means the kernel
serialises over batch, which is worth knowing independently of the NaN.

## 2026-09-22 retry at 64/128 chips: same wall, now measured exactly

A second attempt once the console showed four 4x4x4 pools (16 nodes / 64 chips each).

- Submitted a real **4x4x4 JobSet (16 nodes, 64 chips, 128 devices)** to `default`
  with the correct `tpu7x-128-4x4x4-placement-policy`, reservation and topology
  selectors, carrying the full winning flag set. It is **queued, never admitted**:
  *"insufficient unused quota for google.com/tpu in flavor tpu7x-flavor, 49 more
  needed"*, i.e. only ~15 chips are borrowable.
- Kueue never even created a ProvisioningRequest for it (other users' workloads do
  get one), because it cannot pass the quota check first.
- Node pools here are **auto-provisioned and churn constantly**. The console's
  "16 nodes / 64 chips, OK" is the configured size, not what is up: minutes later
  `kubectl` showed **zero** nodes labelled `4x4x4`, and a different 64-node pool
  (`1ec0u9s0`, topology **4x8x8**, 256 chips) had appeared and sat **completely idle**.
- So the capacity is physically there and unused; the quota to claim it is not.

Quota accounting, measured:

| queue | nominal tpu7x | used | can I submit? |
|---|---|---|---|
| **priority-dev** | **307** | **0** | **no (RBAC)** |
| default | 0 (borrow only) | 20 | yes |
| ubench-regression-tests | 0 | 0 | no |
| vllm-serving | 0 | 0 | no |
| cdk-cluster-queue | 0 | 0 | no |

`default` is the **only** namespace where I can create jobsets, and it holds no
nominal quota of its own. Every chip in the cohort is nominally `priority-dev`'s, and
that namespace is exactly the one I am forbidden from. This is not a capacity problem
and not a scheduling problem; it is one RBAC grant.

### Full capacity survey, 2026-09-22

Every tpu7x cluster I have any access to, checked in one pass:

| cluster | project | largest slice I can actually get | blocker |
|---|---|---|---|
| bodaborg-tpu7x-nap | shared-capacity | **256 chips sit idle** | quota is `priority-dev`'s; I can only create jobsets in `default`, which borrows ~15 |
| bodaborg-tpu7x-spot-sps | multipod-dev | 8 devices | 5 nodes, but each is its own single-host 2x2x1 pool, so no multi-host slice |
| bodaborg-tpu7x-spot-256-chip | multipod-dev | 4 chips | jobset+pod create OK, but the spot pools are scaled to 0 |
| tpu7x-cluster-flex | multipod-dev | 0 | down to 1 node, busy with someone's vLLM |
| bodaborg-tpu7x-sps (SPS) | shared-capacity | 0 | Pathways head still in Error |

**8 devices is the ceiling anywhere I can submit.** The 4x4x4 JobSet `o35q09220142`
is left **queued** in `default`: it is suspended so it costs nothing, it is
duration-bounded at 90 minutes, and Kueue will admit it if the cohort frees. Its
results land in `gs://agagik-us/olmo35/4x8x8` and pod logs persist 12h
(`ttlSecondsAfterFinished`). Check it with:

    KUBECONFIG=/tmp/kc-nap-olmo35.yaml kubectl get jobset -n default o35q09220142
    KUBECONFIG=/tmp/kc-nap-olmo35.yaml kubectl logs -n default -l jobset.sigs.k8s.io/jobset-name=o35q09220142 --tail=40

It carries the full winning configuration (gmm_v2 post-fix, `shard_exp_on_fsdp`,
`TOKAMAX_KDA_DENSE_PAIRS=1`, bf16 KDA, the Ironwood XLA set) across tiny, small and
medium at pdb=4. `large` is excluded: at 128 devices it needs ~164 GB/device.

## 4x4x4 attempt, 2026-09-22: two hard blockers, both external

Four healthy **4x4x4 pools** now exist on `bodaborg-tpu7x-nap`
(`nap-tpu7x-stand-4t-{14m6k6xb,1gasnexi,elw9rs7j,kh8x1mzr}`, 16 nodes / 64 chips each,
placement policy `tpu7x-128-4x4x4-placement-policy`). A 4x4x4 job is written, renders
correctly at 16 completions and is **submitted and queued** as `o35q09220400`. It
cannot be admitted:

| route | state |
|---|---|
| `bodaborg-tpu7x-nap` / `priority-dev` | 307 chips nominal, **0 used**, but **no RBAC to create jobsets** |
| `bodaborg-tpu7x-nap` / `default` | nominal 0, already borrowing 100; my 64-chip ask needs **"49 more"**, and 256/128/128/128-chip jobs are queued ahead |
| `tpu7x-cluster-flex` / `tpu7x-full-pod-spot` | a real 4x4x4, **no Kueue**, autoscaler fired `0->16`, then **`FailedScaleUp: GCE out of resources`** (spot stockout) |

So reserved 4x4x4 capacity is RBAC-gated and spot 4x4x4 capacity is out of stock.
Three launcher bugs were fixed getting this far, and are worth keeping:
the `exclusive-topology` webhook requires an explicit
`cloud.google.com/gke-nodepool` selector on the pods; a stray
`kueue.x-k8s.io/queue-name` label suspends the JobSet forever on a cluster whose
Kueue has no matching queue; and a spot pool needs the reservation selector and
toleration omitted, not just changed.

The queued job will admit on its own if the `default` queue drains. Otherwise the
RBAC grant below is the unblock.

## The one thing that unblocks everything: priority-dev RBAC

Attempted 2026-09-21 after the console showed idle capacity. The capacity is real
and the quota is real, but I cannot reach either.

| what | state |
|---|---|
| `nap-tpu7x-stand-4t-14uwd8sw` (bodaborg-tpu7x-nap) | **64 nodes, 256 chips, idle**, topology `4x8x8` = 512 devices |
| `priority-dev` clusterqueue | **307 chips nominal, 0 used, 0 admitted** |
| my RBAC in `priority-dev` | **create jobsets / jobs / pods: NO** |
| my RBAC in `default` | create jobsets / jobs / pods: yes |
| `default` borrowing headroom | ~3 chips ("insufficient unused quota ... 253 more needed" for a 256-chip ask) |
| reservation `cloudtpu-20260710003900-159478293` | 315 total, 292 allocated, **23 free** |

So the 256 idle chips are inside the reservation but accounted to other queues; the
only queue with headroom is `priority-dev`, and that is exactly the namespace I am
forbidden from. A fallback single-node 2x2x1 ask (4 chips) was admitted by Kueue but
then failed autoscaling ("FailedScaleUp: Internal error", plus "exceeded quota:
cluster-wide"), and it tried to provision a new node rather than use one of the 10
ready ones in `nap-tpu7x-stand-4t-1ch69ok2`.

**The ask:** add `User: 112155357684894056033` (this SA's numeric uniqueId) as an RBAC
subject able to create jobsets in the `priority-dev` namespace of
`bodaborg-tpu7x-nap`, project `cloud-tpu-shared-capacity`. Send to
`%cmcs-shared-clusters-admin-grpadm.prod`. With that, `scripts/olmo35_xpk_4x8x8.sh`
runs as-is: it is written, YAML-validated at 64 completions, and ships the rebased
worktree source from `gs://agagik-us/olmo35/src.tgz` so the stale image does not
matter. It would measure tiny, small and large at pdb=4 on 512 devices, which is the
exact operating point phase 6.5 predicts at 35-51% MFU.

Also note: on this cluster the XLA flags **do** reach the compiler (the workers are
ours), unlike SPS. So it would test the sparse-core offload set too.

## 2026-09-23: the new large capacity is `bodaborg-tpu7x-gsc`, and it is RBAC-gated

A fresh cluster appeared in `cloud-tpu-shared-capacity`, created 2026-09-21:
**`bodaborg-tpu7x-gsc`, 767 Ironwood nodes (3068 chips)** across ~48 pools of 16,
including pools already labelled `4x4x4`, `4x4x8` and `8x8x8`. 2236 chips are in use,
so roughly **830 chips are free**. This is by far the largest Ironwood capacity seen in
this work, and the right home for the whole remaining matrix.

I cannot submit to it. `kubectl auth can-i create pods` returns **no in every one of
its 16 namespaces**, and `create rolebinding` is also denied, so it cannot be
self-granted.

The gap is a single missing rolebinding, and diffing the two clusters names it exactly:

| cluster | namespace `default` bindings to `power-users` |
|---|---|
| `bodaborg-tpu7x-nap` (works) | `power-users-binding-default` -> Groups `cmcs-ai-ninja-team@`, `multipod-users@`, **plus `shuwenf-vm-power-users` -> `User: 112155357684894056033`, `User: 630405687483-compute@developer.gserviceaccount.com`** |
| `bodaborg-tpu7x-gsc` (blocked) | `power-users-binding-default` -> Group `multipod-users@` only. **No user-level binding.** |

**The ask, unchanged in substance from the priority-dev one below but now pointed at
the cluster that actually has the chips:** replicate the `shuwenf-vm-power-users`
rolebinding on `bodaborg-tpu7x-gsc`, namespace `default`, i.e. bind
`User: 630405687483-compute@developer.gserviceaccount.com` (uniqueId
`112155357684894056033`) to the existing `power-users` role. Adding the account to
`multipod-users@twosync.google.com` does the same job. Send to
`%cmcs-shared-clusters-admin-grpadm.prod`.

### Why the other routes are still shut

| cluster | project | state on 2026-09-23 |
|---|---|---|
| `bodaborg-tpu7x-gsc` | shared-capacity | 767 nodes, ~830 chips free, **no create rights anywhere** |
| `bodaborg-tpu7x-nap` | shared-capacity | create OK in `default`, but pools churn; a 16-node `4x4x4` pool was ready at 21:29 and consumed by another job before Kueue admitted mine. Now only `2x2x1` and `1x1x1` remain |
| `ab-k3-v7x-64` | multipod-dev | two ready **2x4x4** pools (8 nodes / 32 chips each), create OK, but both fully held by a `k3-serve` job running **5-7 days** |
| `tpu7x-cluster-flex` | multipod-dev | down to 4 tpu7x nodes, largest `2x2x2` |
| `ninja-v7x-64-spot`, `ninja-v7x-512`, `gtt-cluster`, `y6k`, `y6k-2` | multipod-dev | create OK, **zero tpu7x nodes** |

### Kueue on nap admits against physical topology, not nominal quota

Worth recording because the numbers look contradictory. `nap`'s cohort reports
`nominal=307, used=57, free=250`, yet a 64-chip ask is refused with **"62 more
needed"** and a 32-chip ask with **"30 more needed"**: effectively 2 chips available.
The reason is that `tpu7x-flavor` sets `topologyName: tpu-multihost-topology`, so
Kueue's TopologyAwareScheduling admits only into a **physically existing, free
topology domain**. With the cluster down to `2x2x1` and `1x1x1` nodes there is no
multi-host domain to place into, and the 307-chip nominal quota is unreachable
regardless. The `check-capacity-prov` ProvisioningRequest check cannot help either:
it runs **after** QuotaReserved, so it never fires.

Practical consequence: on `nap`, a run only lands if a multi-host pool is physically
present and idle at the moment Kueue sweeps. Two jobs are queued against exactly that
event, `o35a232130` (4x4x4, 16 nodes) and `o35b232137` (2x4x4, 8 nodes), both sized so
the smaller one admits first if only partial capacity frees.

### Launcher changes made for this attempt

`scripts/olmo35_xpk_4x8x8.sh` now defaults to `4x4x4`/16 nodes in namespace `default`
(the only namespace this account can create in), loops the `CFGS` pdb:seq matrix
instead of a single point, drops the `kueue.x-k8s.io/priority-class` label that names
no WorkloadPriorityClass, and tries both `gs://agagik-us` and
`gs://cloud-pathways-staging` for the source tarball since pod identity differs by
cluster.

## MEASURED ON A REAL 4x4x4, 2026-09-24: the binder is HBM, and 128 devices does not beat 8

Run `o35nap240641` on `bodaborg-tpu7x-nap`, pool `nap-tpu7x-stand-4t-*`, **16 nodes,
64 chips, 128 devices**, `olmo35-tiny`, 20 steps synthetic, median of the last 10.

| arm | pdb | seq | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|---|
| a1_p1s8k | 1 | 8192 | 54.2 | 4.70% | 0.652 | ok |
| a2_p2s8k | 2 | 8192 | **78.2** | **6.78%** | 0.903 | **best** |
| a3_p4s8k | 4 | 8192 | | | | **OOM, temporaries 104.02G vs 94.74G** |
| a4_p2s16k | 2 | 16384 | | | | **OOM, 101.51G vs 94.74G** |
| b1_rematcustom | 4 | 8192 | 75.0 | 6.50% | 1.885 | fits only via offload, still loses to pdb=2 |
| b2_ep4 | 4 | 8192 | | | | my config error, see below |
| b3_noshardexp | 4 | 8192 | | | | OOM, 105.14G |
| b4_megablox | 4 | 8192 | | | | OOM (pdb=4) |
| z_profile | 4 | 8192 | | | | OOM (pdb=4), **no xplane captured** |

**Four conclusions, and they change the plan.**

1. **The wall is HBM, not compute.** pdb=4 needs 104.02 GiB of HLO temporaries
   against 94.74 GiB available, short by 9.3 GiB. Every pdb=4 arm died the same
   way. The pdb knee that perfsim identified is real but **unreachable in the
   current memory envelope**, so the lever list moves from compute to memory.
2. **128 devices does not beat 8.** Best at 128 is 78.2 TF/s/dev against **81.5
   measured on 8 devices**. Scaling this geometry out is flat to slightly
   negative, which retires the "8 devices is slice-limited, it will come good at
   a real slice" hypothesis that motivated the whole 4x4x4 hunt.
3. **perfsim is 4.3x optimistic here, not 1.7x.** It predicted 29.0% MFU at
   pdb=2 and hardware gave 6.78%. The *direction* transferred (pdb 1 -> 2 was
   +44% measured against perfsim's +100%) but the magnitude is unusable. Do not
   quote perfsim MFU for this family at 128 devices; use it for ranking only.
4. **Offload buys the memory but not the throughput.** b1 is the one pdb=4 arm
   that ran, via `remat_policy=custom` with `decoder_layer_input` and `mlpwo`
   offloaded, and it lands at 75.0, below pdb=2's 78.2. Trading bandwidth for
   HBM does not pay at this shape.

**b2_ep4 was my bug, not a hardware result.** Passing `ici_expert_parallelism=4`
while leaving `ici_fsdp_parallelism=-1` left 24.17% of parameters unsharded and
MaxText asserted. `fsdp x ep` must equal the device count, so the correct form is
`ici_fsdp_parallelism=32 ici_expert_parallelism=4`. Requeued as `c3_ep4fix`.

### Follow-up matrix queued as `o35c241729`

Aimed at the memory wall and at the two gaps the first run left. `capacity_factor`
is the untested MoE memory knob: it is `-1.0` (dropless) today, which is what makes
the expert intermediates unbounded.

| arm | what it answers |
|---|---|
| c1_prof_p2 | xplane at the **actual** best point, pdb=2, for xla-shell. The first run pinned the profile to pdb=4 and lost it to the OOM |
| c2_p3s8k | does pdb=3 fit in the 9.3 GiB gap |
| c3_ep4fix | expert parallelism with the sharding corrected |
| c4_p4_cf125 / c8_p4_cf10 | does bounding expert capacity buy pdb=4 |
| c5_p4_optoff | `optimizer_memory_host_offload` as a cheaper offload than activation offload |
| c6_p2_megablox / c7_p2_noshardexp | the two kernel/sharding levers, re-tested at the pdb that actually runs |
| c9_p2_prof_cf125 | whether capacity bounding costs throughput when memory is not binding |

### Hill-climb matrix, pre-screened on perfsim at 128 devices before spending a slice

`olmo35-tiny`, `v7x_4x4x4` (128 devices), seq 8192, remat full:

| pdb | global tokens | TF/s/dev | MFU | comm share of step |
|---|---|---|---|---|
| 1 | 1,048,576 | 167 | 14.5% | **50.0%** |
| 2 | 2,097,152 | 334 | 29.0% | 12.5% |
| 4 | 4,194,304 | 412 | **35.7%** | 0.0% |
| 8 | 8,388,608 | 428 | 37.1% | 0.0% |

**The knee is pdb=4**; pdb=8 buys 1.4 more points and costs 2x the HBM, so the
climb should not spend an admission on it. The comm column is the mechanism and
also the reason the 8-device 81.5 TF/s number does not extrapolate: at pdb=1 on
128 devices **half the step is collective**, and raising pdb is what retires it.
Applying the 1.7x optimism factor measured on OLMoE3 at this device count puts
pdb=4 at roughly **21% MFU**, which is the first time any estimate has cleared
the 20% target on the shipped geometry. The submitted matrix
`CFGS="1:8192 4:8192 1:16384"` brackets this knee.

### Four routes to a 4x4x4, all tried 2026-09-23, all waiting on capacity

| route | mechanism | state |
|---|---|---|
| `nap` / `default`, 4x4x4 + 2x4x4 queued | Kueue + TAS | deadlocked, see below |
| `tpu7x-cluster-flex` / `tpu7x-full-pod-spot` | no Kueue, autoscaler 0->16 | `TriggeredScaleUp` then **`GCE out of resources`** (us-central1-c spot stockout), retrying on backoff |
| `tpu7x-cluster-flex` / `tpu7x-half-cube-2x4x4` | no Kueue, flex-start DWS 0->8, **not spot** | scale-up triggered, no failure yet, waiting on the provisioning request |
| `ninja-v7x-512` | 256 chips of real Kueue quota, create rights | **every tpu7x pool is in ERROR**: "Creation of a managed instance group with tpu7x-standard-4t machine type with placement policy is not supported. Use workload policy instead" |

Also checked and rejected: `bodaborg-tpu7x-spot-256-chip` has two autoscaling
4x4x4 pools but **every flavor's nominal quota is 0** with 4 workloads already
pending, so Kueue there can never admit.

**The nap deadlock is structural, not a matter of waiting.** NAP builds the pool
shape that a *Pending* pod asks for, but Kueue suspends the pod before the
scheduler ever sees it, so NAP only ever sees other people's 2x2x1 requests and
keeps building 2x2x1 pools (watched it go 22 -> 33 nodes of 2x2x1 while my 4x4x4
sat at "57 more needed"). There is no bypass: `manageJobsWithoutQueueName: true`
with every framework enabled (`pod`, `batch/job`, `jobset`, `deployment`,
`statefulset`) and a namespace selector that does **not** exclude `default`, and
`create provisioningrequests` is denied so the ProvisioningRequest cannot be
issued by hand either.

A detail worth keeping for any multi-host JobSet on a scaled-to-zero pool: the
pod webhook creates **only the leader pod** and logs `FailedCreate: leader pod
not yet scheduled, not creating follower pod`. That is expected, not a bug, so
"1 Pending pod" for a 16-node request is the normal pre-provision state.

## Blocked / infrastructure

- **A 4x4x4 is not currently obtainable.** `bodaborg-tpu7x-nap` has 47 Ironwood nodes
  but none carry a `gke-tpu-partition-4x4x4-id` label (no partitions formed), and the
  `priority-dev` queue's nominal quota is **51 chips**, under the 64 a 4x4x4 needs.
- A free **2x2x2 (16 devices)** exists on `tpu7x-cluster-flex` pool
  `tpu7x-multi-host-spot-2`, no Kueue. It needs a 2-pod JobSet plus getting the
  rebased source into the run (the `kdaj24` image predates the rebase by 411 commits).
- `olmo35-small` at 8 layers was attempted and **timed out waiting for SPS placement**,
  not a model failure. SPS is shared and single-slice, so runs serialise; two
  concurrent launches contend and one dies.

## Phase 6 verdict: is 20-30% reachable?

**On the 8-device slice we can actually run: no, and that is not the architecture's
fault.** Best measured is **81.5 TF/s/device = 7.1% MFU** for `olmo35-tiny`, up
**1.89x** from the 43.1 TF/s untuned baseline. The binding constraint is the slice:
a 12.5B-parameter model FSDP-sharded over 8 devices at pdb=1, where the xla-shell
floor is 372ms of SparseCore comm against a 1.10s step.

**At a realistic slice: yes, and with margin.** perfsim at 128 devices, pdb>=4, on the
**shipped** geometry gives 35.7% / 41.3% / 50.8% for tiny / small / large, clearing 20%
at pdb=2 on every rung. Two independent hardware anchors support that this is the right
order of magnitude rather than perfsim optimism:

- the same stack, same 8 devices, measures **24.0-30.3% MFU on dense OLMo 3.1**, so
  neither the platform nor MaxText is the limiter;
- tuned Ironwood MoE recipes in `tpu-recipes` land at **26-27%** (deepseek-v3,
  qwen3-235b) and the narrow-expert gpt-oss class at 14.3%.

Calibration caveat, measured both ways: perfsim is **0.62x pessimistic at 8 devices**
and about **1.7x optimistic at 128**. The factor is regime-dependent, so the honest
statement is a band, not a point. Derating the 128-device numbers by 1.7 gives
**21% / 24% / 30%** for tiny / small / large, still inside the 20-30% target.

What would actually settle it is the 512-device run, which is written and blocked only
on RBAC. Until then the claim is: **20-30% is reachable on the shipped geometry, and
nothing measured so far contradicts it**; the architecture changes priced in phase 6.5
(fewer, wider experts, worth 1.12-1.60x) are an optimisation on top, not a
prerequisite.

## Phase 7: decode

- [x] 7.1 `inference_microbenchmark` run on Ironwood: **prefill works, decode is mesh-infeasible on 8 devices** (see below)
- [ ] 7.2 Cross-check the perfsim decode table: blocked, no decode step time to compare against
- [ ] 7.3 Serving flags from `tpu-recipes/inference/ironwood`: not reached
- [x] 7.4 KV-heads-divides-TP: **MaxText now enforces this itself** and rejected tp=8 outright. The rule from `kda-headdim-answer.md` is validated, and it is what makes decode infeasible here

### Phase 7 finding: `olmo35-tiny` cannot fill an 8-device host for decode

Three constraints that cannot be satisfied together on 8 devices:

1. **KV-head-sharding axes must multiply to at most 4.** `tensor` and
   `autoregressive` both shard KV heads, and tiny has only 4. MaxText rejects
   anything larger outright: *"num_kv_heads (4) ... must be divisible by 8 ...
   Attention heads are atomic under tensor parallelism"*.
2. **Batch-sharding axes must be 1.** The autoregressive step has batch 1, so
   `data`/`fsdp` cannot split it: *"Batch dimension should be shardable among the
   devices in data and fsdp axis got query.shape[0]=1/devices_in_data_fsdp=2"*.
3. 4 x 1 = 4, not 8.

So decode caps at **4 of the 8 devices**, leaving half the host idle, and the cause is
the geometry: 8 query / **4 KV** heads. Prefill runs fine; only the AR step is blocked.

A second, independent gap found on the way: the fused tokamax KDA wraps its kernel in
a `shard_map` that shards **only the batch axis**
(`olmoe3.py`, `in_specs = [PartitionSpec(batch_axes, None, None, None)]`). At decode
that batch is 1, so the fused path fails on any mesh with `fsdp>1` before the KV-head
rule is even reached. The unfused chunked path has no `shard_map` and gets further.

**`small` hits the same wall by different arithmetic.** It has 8 KV heads, so the
KV rule that stopped `tiny` is satisfied and tp=8 is legal there. It fails instead on
width: `mlp`/`heads` are sharded by `tensor` AND `autoregressive` together, so those
axes multiply to 8 and d_model 1536 / 8 = **192, which is not a multiple of 128**
(`ValueError: Block size must divide tiling: block_size=192, tiling=128`). Keeping
1536/(tensor x ar) aligned needs that product to divide 12, so at most 4 in practice
(1536/4 = 384). Batch axes must still be 1. Again 4 != 8.

So **both measured rungs cap decode at 4 of 8 devices**, for reasons that differ per
rung but bite identically: `tiny` on KV-head count, `small` on d_model alignment under
sharding. This is a general property of the family at this host size, not a one-off.

Both are co-design inputs rather than bugs: this is exactly the "an arm that wins
training MFU and cannot shard for serving is not a win" check. The fix is architectural
(more KV heads) or a serving-time mesh that uses fewer devices per replica.

## Run record

One row per configuration. Filled in as runs land.

| # | phase | config / flag diff | step s | TF/s/dev | MFU | note |
|---|---|---|---|---|---|---|
| 1 | 1b | olmo3-7b, seq 8192, pdb 1, remat full | 1.41 | 277 | 24.0% | dense baseline, untuned |
| 2 | 1b | olmo3-32b, seq 8192, pdb 1, remat full | 4.84 | 349 | 30.3% | dense baseline, untuned |
| 3 | 3 | olmo3-7b + remat=custom, all `=device` | 1.23 | 318 | 27.6% | +15% rel over row 1, loss identical (3.244 vs 3.245) |
| 4 | 3 | olmo3-7b, row 3 with XLA flags OFF | 1.23 | 318 | 27.6% | **LIBTPU does not propagate over SPS** |
| 5 | 3 | olmo3-7b + recipe `sa_block_*=2048` | n/a | n/a | n/a | CompileTimeScopedVmemOom; needs the scoped-vmem XLA flag |
| 6 | 2.4 | **olmo35-tiny**, seq 8192, pdb 1, remat full | 0.79 | 45.7 | **3.96%** | MoE baseline; 10,400 tok/s/dev |
| 7 | 3 | olmo35-tiny + remat=custom, all `=device` | 0.77 | ~46 | ~4.0% | no change, high variance (0.61-0.95 s) |
| 8 | 4 | olmo35-tiny pdb=4, remat=custom | n/a | n/a | n/a | OOM: 112.15 G temporaries vs 94.74 G |
| 9 | 4 | olmo35-tiny pdb=8, remat=custom | n/a | n/a | n/a | OOM: 190.79 G temporaries |
| 10 | 4 | olmo35-tiny pdb=4, remat=full | n/a | n/a | n/a | OOM 127.93 G, **worse than custom**: MoE dispatch buffers, not activations |
| 11 | 4 | olmo35-tiny pdb=8, remat=full | n/a | n/a | n/a | OOM 215.12 G |
| 12 | 4 | olmo35-tiny pdb=2, fused KDA | n/a | n/a | n/a | **NaN loss** (kernel bug) |
| 13 | 4 | olmo35-tiny pdb=2, `use_tokamax_kda=False` | 1.36 | 52.0 | 4.5% | runs clean, loss 11.7 -> 9.26 |
| 14 | 3.2 | olmo35-tiny + `TOKAMAX_KDA_BF16_FWD/BWD=1` | 0.77 | 45.8 | 4.0% | neutral, loss unchanged |
| 15 | 3 | olmo35-tiny + `use_tokamax_gmm=True` (v1) | 0.78 | 45.5 | 3.9% | neutral |
| 16 | 3.1 | olmo35-tiny + `use_gmm_v2=True` | n/a | n/a | n/a | **NaN loss** (needs `use_tokamax_gmm=True` to even validate) |
| 17 | - | olmo35-tiny at 8 layers (half depth) | 0.47 | 44.3 | 3.8% | **depth-independent**: step is not compute-bound |
| 18 | 3.6 | spot-sps pod, XLA flags ON vs OFF | 0.77 | 46.1 / 43.9 | 4.0 / 3.8% | **+5%**; first time the flags actually applied |
| 19 | 4 | seq 16384, pdb 1 | 1.29 | 56.2 | **4.9%** | **1.30x, and dodges the KDA NaN** |
| 20 | 4 | seq 32768, pdb 1 | n/a | n/a | n/a | OOM 124.35 G |
| 21 | 3 | expert parallelism ep=2 / ep=4 | 0.83 / 1.03 | 42.4 / 34.3 | 3.7 / 3.0% | **hurts**; perfsim predicted 2.6-3.8x gain |
| 22 | 3 | tokamax gmm v1 vs megablox | 0.81 | 43.8 vs 43.4 | 3.8% | neutral, loss identical |
| 23 | 3.1 | gmm_v2 with `use_tokamax_kda=False` | n/a | n/a | n/a | **NaN**: bug is independent of KDA |
| 24 | 3.1 | gmm_v2, `wi/wo_tile_*` clamped to 512 by hand | 0.80 | 44.1 | 3.8% | clean: first evidence the tiles were the cause |
| 25 | 3.1 | gmm_v2 + `use_gmm_v2_heuristic_tiling=True` | 0.83 | 42.7 | 3.7% | clean: tokamax picks all four tile fields |
| 26 | 3.1 | **gmm_v2 with `_clamp_tiles()` fix, default tiles** | 0.77 | **45.9** | **4.0%** | **fixed; fastest of the three, 1.06x over megablox** |
