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
- [ ] 1.4 Diff `standalone_configs.py --model-size all` geometry against the four `olmo35-*.yml` field by field
- [ ] 1.5 Record parity numbers as the gate for everything downstream

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

- [ ] 2.1 Install patched tokamax KDA from `olmo35/tokamax-kda-patched/kda` into `venv-maxtext` (0.0.13 ships `gmm_v2` but no `kda`; upstream `kda` is chunk 64 only, no bf16 switches)
- [ ] 2.2 Re-run parity after the install (KDA is on the logits path)
- [x] 2.3 SPS smoke test done via phase 1b: `Jax Backend: Pathways`, `Num_devices: 8`, clean teardown, RC=0
- [x] 2.1 Patched tokamax KDA installed into `venv-maxtext` at `site-packages/tokamax/_src/ops/experimental/kda/` (from `olmo35/tokamax-kda-patched/kda`). `kimi_delta_attention` imports; the two previously-skipped tokamax KDA parity tests now **pass**
- [x] 2.2 Parity re-run after the install: **8 passed**
- [x] 2.4 Baseline `olmo35-tiny` seq 8192 pdb 1 remat full: **45.7 TF/s/dev = 3.96% MFU**, step 0.79 s, 10,400 tok/s/dev
- [ ] 2.5 No leftover `isc-proxy-agagik-*` jobs
- [ ] 2.6 Start the 4x4x4 hunt for `small` (needs >= 64 chips)

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

- [ ] 3.1 `use_gmm_v2=true` (prior: 2.06x kernel, -6.6% step)
- [ ] 3.2 `TOKAMAX_KDA_BF16_FWD=1` + `TOKAMAX_KDA_BF16_BWD=1` (grad parity 2.5e-3)
- [ ] 3.3 `TOKAMAX_KDA_DENSE_PAIRS=1`
- [ ] 3.4 `TOKAMAX_KDA_CHUNK_SIZE` 64 / 128 / 256
- [ ] 3.5 gpt-oss-120b MaxText flag set (same expert class)
- [ ] 3.6 gpt-oss-120b XLA flag set (sparse-core offload; prior 1.93x for 2-SC)
- [ ] 3.7 Cross-check qwen3-235b where it disagrees (`megablox=False`, `shard_exp_on_fsdp=False`)
- [ ] 3.8 Keep `use_random_routing` off for headline numbers

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

- [ ] 4.1 Sweep pdb 1 / 2 / 3 at seq 8192
- [ ] 4.2 Buy batch with `decoder_layer_input=offload`, `mlpwo=offload`
- [ ] 4.3 Re-test phase-3 winners at the best pdb
- [ ] 4.4 Record the memory wall (OOM pdb and reported temporaries)

## Phase 5: xla-shell profile loop

- [ ] 5.1 Capture xplane profile of the best config
- [ ] 5.2 `analyze_profile`: lanes, ceiling, binder, verdict
- [ ] 5.3 `roadmap --all --json`, then `--collective | --kernels | --remat | --relayout`
- [ ] 5.4 Fix in roadmap order, re-measure, stop when the floor stops moving
- [ ] 5.5 Kernel interventions only if the roadmap points at them (latent-MoE fusion prototype, `tokamax_gmm_tile_m`)
- [ ] 5.6 Record the xla-shell floor next to the measured step time

## Phase 6: perfsim cross-check and the 20% verdict

- [ ] 6.1 Re-run `scripts/olmo35_perfsim.py` at the measured operating point
- [ ] 6.2 Derive the true optimism factor (replaces the 1.7x borrowed from OLMoE3)
- [ ] 6.3 Re-issue medium and large projections with that factor
- [ ] 6.4 State the 20% verdict for `tiny` and name the binding constraint
- [ ] 6.5 If below 20%, price the architectural gap: sweep experts, `top_k`, expert width
- [ ] 6.6 If the 4x4x4 landed, repeat 2.4-5.6 for `small`

## Open bugs found (both block the main levers)

1. **Fused tokamax KDA produces NaN at `per_device_batch_size=2`.** pdb=1 is clean and
   the unfused chunked path is clean at pdb=2 (loss 11.73 -> 9.26), so it is the
   kernel, not the model. This blocks the single most valuable lever, since pdb is
   what every tuned recipe uses to raise MFU. Kernel is
   `olmo35/tokamax-kda-patched/kda` at dk=128 / dv=256.
2. **`use_gmm_v2=True` produces NaN at pdb=1.** This is the lever Aleksey Vlasenko
   measured as the largest single OLMoE3 win (2.06x on the ragged-dot kernel, -6.6%
   step). Note it also requires `use_tokamax_gmm=True` or config validation rejects it.

Both need a minimal repro and a fix before the lever sweep can conclude.

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

## Phase 7: decode

- [ ] 7.1 `inference_microbenchmark` for `tiny` on SPS
- [ ] 7.2 Cross-check the perfsim decode table
- [ ] 7.3 Serving flags from `tpu-recipes/inference/ironwood`
- [ ] 7.4 Hold KV-heads-divides-TP (violating it replicates the cache, up to 2x)

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
