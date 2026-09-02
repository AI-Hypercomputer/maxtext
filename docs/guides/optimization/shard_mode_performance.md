<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# `shard_mode: explicit` vs `shard_mode: auto` — a measured performance analysis

**Status:** investigation report. Measured 2026-08-31 on a TPU v5p slice (4 JAX devices / 8 TensorCores,
95.7 GiB HBM per device), JAX 0.11.1, MaxText branch `chengnuojin-trainer-fix` @ `671cec44d`.
Revised 2026-09-02 with the shipped fix (`lm_head_weight_grad_in_kernel_order`) and two negative
results; see §8 item 1 and §8 item 2. Where a claim was superseded the original is kept and marked,
because two of the corrections reverse a conclusion that looked solid at the time.

`shard_mode` (`src/maxtext/configs/base.yml:545`) selects how MaxText expresses sharding:

- **`auto`** — `jax.lax.with_sharding_constraint`. A *hint*. GSPMD/Shardy is free to propagate it
  backwards into producers, refine it, sink it, or drop it.
- **`explicit`** — `jax.sharding.reshard` on an all-`AxisType.Explicit` mesh (JAX sharding-in-types).
  A *typed conversion*. The sharding becomes part of the value's type and the transfer is emitted at
  exactly that program point.

The entire switch is one branch, `maybe_shard_with_name` at `src/maxtext/utils/sharding.py:125-129`:

```python
if shard_mode == ShardMode.EXPLICIT:
    return reshard(inputs, named_sharding)
else:
    return jax.lax.with_sharding_constraint(inputs, named_sharding)
```

plus mesh construction at `src/maxtext/utils/maxtext_utils.py:2426-2429`, which stamps *every* mesh
axis `Explicit` or *every* axis `Auto` — there is no partial adoption.

______________________________________________________________________

## 1. Executive summary

01. **The root cause of the regression is an optimization barrier, not sharding policy.** Under
    explicit, Shardy materializes a `custom-call(custom_call_target="Sharding")` on every pinned
    intermediate. On llama2 that is **65 such calls under auto vs 1,276 under explicit** (qwen3:
    65 vs 1,494), and the pre-optimization jaxpr is otherwise near-identical (42 dots and 49
    transposes in both; every non-`Sharding` opcode delta ≤ +33). The custom-call is a semantic
    no-op, but it sits *between* the unembedding weight-gradient `dot` and its `transpose`, and XLA's
    algebraic simplifier will not fold `transpose(dot(A,B)) → dot(B,A)` across it. Census on the
    dp4 module: **auto has 17 adjacent (foldable) transpose-of-dot pairs, explicit has 0.**

02. **Two concrete costs follow from that one barrier.**

    - **A lost reduce-scatter.** Under auto the vocab weight-gradient `all-reduce` feeds a
      `dynamic-slice`, and XLA rewrites the pair into a real reduce-scatter. Under explicit the
      gradient is transposed, the FSDP shard dim becomes minor-most, the pattern no longer matches,
      and a plain full all-reduce is emitted — **2× the wire bytes for that tensor** (1.5·D vs
      0.75·D on a 4-way ring; 49.15 MB vs 24.58 MB per device on llama2).
    - **Six relayout copies per step.** `entry_computation_layout` is identical in both modes and
      the optimizer state is donated, so the `{0,1}`-pinned Adam chain must be converted in and out:
      three copies for `logits_dense.kernel` / `mu` / `nu` on the way in, three on the way out.
      196.6 MB/step on llama2, up to 1.57 GB/step at emb 4096. **Added copy bytes separate the sign
      of the effect perfectly**: the only three configs of 25 that add ~zero (gemma, gemma2, tp4) are
      exactly the three that get faster; all 22 that add ≥ 66 MB regress.

03. **Collective *bytes* are not identical — the byte counter was wrong.** A naive counter
    reads the `all-reduce` *inside* auto's `kind=kCustom, calls=%all-reduce-scatter.N` fusion at its
    full pre-slice shape, so it reports "unchanged". `after_codegen` shows the truth:
    `reduce-scatter` lines are **llama2 12 → 4, mistral 12 → 4, mixtral 12 → 6, qwen3 10 → 4**.
    The same fusion is booked as FUSION rather than COLLECTIVE by the xplane classifier, so a large
    part of the apparent "collective time doubled" is reclassification. Everywhere this document
    previously said "identical bytes", read "identical *static* byte counts, which is an artifact".

04. **The penalty does not simply amortize with scale.** Its *instruction count* is fixed (always the
    same six copies); its *cost* is not, and the ladder reads **non-monotone**: w512 +4.80% → w2048
    +0.96% → w4096 **+1.49%**. The largest absolute penalty in the whole 28-run sweep (**+1111 µs**)
    is at the *largest* width. Regressing step delta on measured copy time over 12 configs gives
    `Δ = −8.7 + 1.37·copy_µs`, **R² = 0.449** — the copies explain about half the variance.

    > *Correction (2026-09-02).* "Non-monotone" conflates two regimes. w512 is the only rung below
    > XLA's 512-byte minor-most AR→RS gate (§4.2), so it pays mechanism A **and** mechanism B; w2048
    > and w4096 pay only B. Within the above-gate regime the ladder is monotone increasing in width
    > (+0.96% → +1.49%), which is what the relayout-copy model predicts. The practical consequence is
    > sharper than the original claim: **the emb-512 numbers throughout this document overstate what
    > a production-geometry job sees**, and any fix validated only at emb 512 is validated on an
    > artifact. See §8 item 1.

05. **The 16-layer near-zero is cancellation, not amortization.** At d16 the six copies still cost a
    measured **345.2 µs/step**; they are masked by an unrelated **−242 µs** win inside the layer
    scan (and a −390 µs `concatenate` swing). Pooling all four d16 runs per mode gives **+70.7 µs =
    +0.088%, t = 4.5** — a small but statistically significant penalty, not noise.

06. **Tensor parallelism is the only win, and for a different reason entirely.** llama2-tp4's
    −0.61% headline is inflated ~3× by module-launch lead time; the true device-side effect is
    **−0.21%** (−9.3 µs, reproduced independently on TPU:1). The mechanism is not the vocab tensor
    at all — it is the RMSNorm **scale** reshard at `normalizations.py:34-51`, which lets explicit
    write the optimizer update straight into the donated parameter buffers and saves auto's 14
    buffer-aliasing copies.

07. **One headline number in the matrix is a harness artifact, not a model property.** gemma3's
    +4.49% happens because the shrink sets `base_num_decoder_layers=4`, which is *less than*
    `len(GEMMA3_ATTENTION_PATTERN) == 6`, so `scan_length = 4 // 6 = 0` (`decoders.py:1341`) and the
    whole decoder is **unrolled** into `layers_remainder`. gemma3 is the only model in the sweep
    compiled with zero `while` loops, and unrolling is what makes explicit lose. Re-running the
    same config at more layers: **L4 +4.54% → L6 +1.87% → L12 −0.58%.** Nothing about local/global
    attention, the 262144 vocab, or multimodal code is involved.

08. **`reduced=` / `unreduced=` PartitionSpec tags are a real explicit-only capability but account
    for 0.0% of every number in this document.** `moe.py` contains zero occurrences; the tags live
    only in `deepseek_batchsplit.py` and `gradient_accumulation.py`, behind a gate
    (`gradient_accumulation.py:77`) that requires `data > 1`. Every measured run had `data=1` and
    `gradient_accumulation_steps=1`. Nothing in the MoE path participates in any measured delta.

09. **Explicit forfeits capabilities**: `check_vma` (which `base.yml:704` calls a performance win) is
    hard-rejected, `fused_qkv=True` is a trace-time error, context parallelism on the
    `dot_product`/GPU-flash paths is a trace-time error, and `P.UNCONSTRAINED` is illegal.

10. **There is a fix, it is measured, and it ships here off by default.** The barrier itself cannot be
    removed — it comes from JAX's `lower_with_sharding_in_types`, which annotates every
    sharding-in-types primitive on an all-Explicit mesh, not from any MaxText pin (§4.5; the earlier
    "stop pinning the logits output" recommendation was tested and produces a 0-line HLO diff, and
    so does eliding the redundant barriers — §8 item 2). What can be removed is the *need* for the
    fold: write the LM head's weight gradient by hand so it contracts straight into the kernel's
    stored axis order. `lm_head_weight_grad_in_kernel_order: true` does that. It recovers the
    reduce-scatter of item 02 — the emitted `%all-reduce-scatter` is byte-for-byte auto's — and
    drives the six relayout copies to zero, without touching the stored kernel, the checkpoint, or
    initialization. Median across seven configs: explicit alone **+4.59%**, with the flag
    **−0.57%**. At production geometry, where there was nearly nothing to recover, it is neutral
    (**+0.03%** and **−0.57%**). See §8 item 1.

    > *Superseded (2026-09-02).* This item previously recommended `lm_head_kernel_transposed`, which
    > stores the kernel as `[vocab, embed]`. That flag reaches the same layout but is slower on four
    > of seven configs, has no checkpoint-conversion path, and — because `jax.random` draws a
    > different matrix from the same seed at the flipped shape — perturbs the loss ~35× more than
    > explicit's own noise. It has been removed rather than shipped alongside a strictly better flag.

**Practical guidance:** at realistic depth/sequence/batch the penalty falls to a few tenths of a
percent and capability should decide the choice. But it does **not** reliably vanish with *width*,
and under pure data parallelism it is +10%. On an untied model where you have committed to explicit,
set `lm_head_weight_grad_in_kernel_order: true` (§8) rather than assuming scale will hide it; it is
checkpoint-compatible and inert under `auto`, so it costs nothing to leave on.

______________________________________________________________________

## 2. Scope: which models are actually onboarded

`src/maxtext/configs/types.py:4142-4168` gates explicit mode to a whitelist:

```python
supported_decoders = {
    "simple",
    "simple_mlp",
    "llama2",
    "deepseek",
    "mistral",
    "mixtral",
    "qwen3",
    "qwen3_moe",
    "qwen3_custom_moe",
    "gemma",
    "gemma2",
    "gemma3",
}
```

plus `use_multimodal` is rejected. Every real model in that set was measured. Exactly one shipped
config defaults to explicit: `src/maxtext/configs/models/deepseek3-671b-batchsplit.yml:61`.

______________________________________________________________________

## 3. Method

### Harness

Purpose-built for this study and not checked in; §9 gives the exact command line. Four pieces:

| piece                   | what it does                                                                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| run driver              | runs one (model, mode) with `--xla_dump_to` + `profiler=xplane`, serialized under `flock`                                                     |
| xplane decoder          | dependency-free XSpace protobuf reader; the installed `tensorboard_plugin_profile` cannot decode xplane on this host                          |
| collective byte counter | counts collectives by kind, bytes from dtype × shape — **see the caveat below, this one under-reports**                                       |
| HLO differ              | signature-bucketed instruction diff (opcode + layout-stripped shape + `dimensions`), since instruction names are unstable across compilations |

### Timing source

MaxText does not log per-step device time, so step time is taken from the xprof `XLA Modules`
line: the median duration of the `jit_train_step` module events on `/device:TPU:0`. With
`skip_first_n_steps_for_profiler=3, profiler_steps=3` there are exactly 3 such events per run.
Device-op time and the collective / fusion / data-movement split come from the `XLA Ops` and
`Async XLA Ops` lines, summed over all 8 TensorCores and all 3 profiled steps — so device-op totals
are larger than wall step time and should be read as *ratios*, not as wall clock.

### Noise floor

Same config (llama2 decoder, emb 2048, mlp 8192, 16 layers, seq 1024, pdbs 1, fsdp 4), 3 independent
runs per mode:

|          |   run 1 |   run 2 |   run 3 |       mean |            stdev |
| -------- | ------: | ------: | ------: | ---------: | ---------------: |
| auto     | 80366.1 | 80394.8 | 80419.1 | 80393.3 µs | 26.5 µs (0.033%) |
| explicit | 80483.6 | 80477.7 | 80466.0 | 80475.8 µs |  9.0 µs (0.011%) |

Largest within-mode spread: 53 µs = **0.066%**. Treat |Δ| below ~0.15% as noise on a *single* pair;
with 4 runs per mode pooled, a +0.09% effect is resolvable (§5.3).

**Cross-validation against a second, independent metric.** On the 21 device-bound configs where the
xprof module span and MaxText's own host-side step time can both be read, the two agree with
**r = 0.96**, and **17 of 21 are slower under explicit on both**. deepseek measures +8.8% device /
+7.1% host. The direction of every headline in this document survives the metric change; only the
sub-1% magnitudes are metric-sensitive (see caveat 3 below).

### Measurement caveats — read these before trusting any number here

1. **A static collective byte total is not a measure of communication volume.** The byte counter
   used here had two independent defects:

   - It sums the *body* of auto's `kind=kCustom, calls=%all-reduce-scatter.N` fusion, so it reads
     the pre-slice `all-reduce` shape and assigns a reduce-scatter and a full all-reduce of the same
     operand the same bytes — even though they differ by exactly 2× on the wire.
   - It is **trip-count-unweighted**: every collective is counted once regardless of its enclosing
     `while`. `scale/d16` (16 layers) and `scale/w2048` (4 layers) report *byte-for-byte identical*
     totals (628,146,216 B auto) while true dynamic volume is **4.20 GB vs 1.34 GB**.

   Use `grep -c reduce-scatter *after_codegen*` instead, and weight by trip count. Once weighted,
   aggregate per-step collective volume really is nearly mode-invariant across all 25 configs
   (|Δ| ≤ 0.70%, max −0.700% at d16) — but that is *aggregate* parity, not per-class parity:
   collective-permute volume **drops 50%** under explicit in the five scale-ladder configs
   (2 → 1 `collective-permute-start(bf16[48,8192])`, `source_target_pairs={{0,1},{1,2},{2,3}}`),
   while gemma2 gains a real extra in-loop `all-gather bf16[1,8,128,512] dimensions={3}` worth
   +4 MiB/step. This also means **collective instruction *count* is an anti-signal**: explicit's
   lower count reflects auto's wins being fused away.

2. **The xplane classifier books `fusion.N` as FUSION even when `calls=%all-reduce-scatter.N`.**
   Part of every "collective % of device time" jump in §5.1 is this reclassification, not new
   communication. Quantified in §4.4.

3. **`train_step_ns_median` includes a variable module-launch lead gap.** Measured at 30–88 µs
   across the 3 profiled steps on TPU:0 — larger than some of the deltas being reported. For
   sub-1% results, cross-check against the op span (first op start → last op end) or the TPU:1
   module span. This matters for tp4 (§5.2) and nothing else in this document.

4. **Shrinking a model can change its compilation structure, not just its size.** gemma3 at
   4 layers takes a completely different code path from gemma3 at 6+ layers (§4.7). **Always check
   `while`-loop count parity between the two dumps before comparing them** — every other model in
   this sweep compiled to `while = 2` or `6/7` in both modes; gemma3 compiled to `while = 0`. A
   shrink that disables the scan is not a proxy for the real model.

5. **The persistent compilation cache silently suppresses HLO dumps.** MaxText defaults
   `jax_cache_dir: "~/jax_cache"` (`base.yml:534`). A cache hit skips compilation, so
   `--xla_dump_to` produces nothing. The harness sets `jax_cache_dir=` to force compilation. Timing
   is unaffected (the executable is identical), but a sweep that trusts `status: ok` alone will
   silently produce empty dumps.

6. **MaxText's own `dump_hlo` config crashes on a local `base_output_directory`.**
   `src/maxtext/utils/gcs_utils.py:89-116` `upload_dump()` calls `parse_gcs_bucket_and_prefix()`
   unconditionally, which raises `IndexError: string index out of range` on a non-`gs://` path — and
   it fires *after* training but *before* the xplane profile is flushed, so you lose the profile.
   The harness drives the dump through `XLA_FLAGS` instead. **This is a real robustness bug.**

### Measured HBM bandwidth

The relayout copies in §4.3 are bandwidth-bound and were used to calibrate the platform:
**2.29–2.38 TB/s** effective across the four ≥ 65 MB cases, i.e. **83–86% of the v5p 2765 GB/s
spec**. The small (16 MB) llama2 copies read back at an implied 3.64 TB/s — above spec — which means
they are partly cached or partly overlapped and their profiled durations *understate* their isolated
cost. Only the ≥ 65 MB cases give a trustworthy reading.

______________________________________________________________________

## 4. The dominant mechanism

Chain, in causal order: **Sharding custom-call barrier → blocked `transpose(dot)` fold → transposed
vocab gradient → (a) lost reduce-scatter and (b) six relayout copies.**

### 4.1 Root cause: the `Sharding` custom-call is an optimization barrier

MaxText builds the unembedding as a separate `dense_general` with
`kernel_axes=("embed_vocab", "vocab")` at `src/maxtext/layers/decoders.py:831-844`, so the parameter
is `[emb, V]`. `base.yml:561` maps `embed_vocab → fsdp` and `:560` maps `vocab → [tensor, ...]`; at
fsdp=4 / tensor=1 the kernel is sharded 4 ways on its **contracting** (embed) axis and replicated on
vocab. JAX's `_dot_general_transpose_rhs` computes `dW` as a dot that naturally produces `[V, emb]`
followed by an explicit transpose back to `[emb, V]`. **Both modes emit exactly that pair** in
`before_optimizations`:

```
# auto — llama2, lines 2273-2274: dot and transpose are ADJACENT
%dot_general.80 = bf16[32000,512]{1,0} dot(%convert_element_type.147, %transpose.85),
    lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
%transpose.86  = bf16[512,32000]{0,1} transpose(%dot_general.80), dimensions={1,0}

# explicit — qwen3, lines 3446-3449: a Sharding custom-call is INTERPOSED
%dot_general.190 = bf16[151936,512]{1,0} dot(...)
%dot_general.191 = bf16[151936,512]{1,0} custom-call(%dot_general.190),
    custom_call_target="Sharding", sharding={devices=[1,4]<=[4]},
    xla.sdy.sharding=<[{}, {"fsdp"}]>
%transpose.216 = bf16[512,151936]{0,1} transpose(%dot_general.191), dimensions={1,0}
%transpose.217 = bf16[512,151936]{1,0} custom-call(%transpose.216),
    custom_call_target="Sharding", sharding={devices=[4,1]<=[4]}
```

llama2's explicit dump is byte-for-byte the same pattern at V=32000.

The custom-call is semantically a no-op. Its only effect is to stop the algebraic simplifier. The
result is visible in `after_optimizations` as an **unfolded operand order**:

```
qwen3/auto:3721     ROOT %convolution.78 = bf16[512,151936]{1,0:T(8,128)(2,1)} convolution(%fusion.270, %fusion.298), dim_labels=fb_io->bf
qwen3/explicit:3735 ROOT %convolution.78 = bf16[151936,512]{1,0:T(8,128)(2,1)} convolution(%fusion.336, %fusion.285), dim_labels=fb_io->bf
                                           ^^^^^^^^^^^^^^ operands swapped, transpose not folded back
```

Same for llama2 (`bf16[512,32000]` vs `bf16[32000,512]`), mistral, mixtral, deepseek
(`bf16[512,129280]` vs `bf16[129280,512]`) and qwen3_moe.

Scale of the barrier, module-wide:

|                   | Sharding custom-calls | non-`Sharding` instructions | adjacent foldable transpose-of-dot pairs |
| ----------------- | --------------------: | --------------------------: | ---------------------------------------: |
| auto (llama2)     |                    65 |                       1,836 | **17** (+3 already behind a custom-call) |
| explicit (llama2) |                 1,276 |                       1,950 |      **0** (all 17 behind a custom-call) |
| auto (qwen3)      |                    65 |                           — |                                        — |
| explicit (qwen3)  |                 1,494 |                           — |                                        — |

Every other opcode delta between the two pre-optimization modules is ≤ +33 (`add` +33,
`multiply` +28, `constant` +22, `broadcast` +22). The two programs are the same program plus 1,200
barriers.

**Honesty about causality:** this has *not* been proven by rebuilding XLA with the barrier removed.
The evidence is structural and statistical — the modules differ almost only by the custom-calls,
17/17 foldable pairs are adjacent under auto and 0/17 under explicit, and exactly those dots come
out folded under auto and unfolded under explicit. An alternative reading is that XLA's layout
assignment simply made a worse entry-layout choice under explicit (it chose `f32[512,32000]{1,0}`
entry while pinning the internal chain to `{0,1}`; under tp4 it chose `{0,1}` entry and paid
nothing). **Both readings point at the same fix.**

### 4.2 Consequence A — the lost reduce-scatter

Under auto, XLA:TPU pattern-matches the FSDP weight-gradient `all-reduce` feeding a
`dynamic-slice(partition-id*128, 0)` and rewrites it into a genuine reduce-scatter, packaged as a
custom fusion:

```
llama2/auto: %fusion.12 = bf16[128,32000]{1,0} fusion(%fusion.226), kind=kCustom, calls=%all-reduce-scatter.5
  body: %all-reduce.96 = bf16[512,32000] all-reduce(%input.5), replica_groups={{0,1,2,3}},
                         frontend_attributes={from-cross-replica-sharding="true"}
        ROOT %dynamic-slice.111 = bf16[128,32000] dynamic-slice(%all-reduce.96, %multiply.222, %constant.718)
```

Under explicit the gradient is `[V, emb]`, so the shard dim (512 → 128) is minor-most, the rewrite
cannot match, and a plain (combiner-tupled) all-reduce is emitted:

```
llama2/explicit:   %all-reduce.118 = (bf16[512], bf16[32000,512]) all-reduce(%copy-done.146, %fusion.242), channel_id=58
deepseek/explicit: %all-reduce.56  = bf16[129280,512] all-reduce(%fusion.440), channel_id=136
```

`after_codegen` reduce-scatter line counts (2 lines per fusion):

| model   | auto | explicit |
| ------- | ---: | -------: |
| llama2  |   12 |        4 |
| mistral |   12 |        4 |
| mixtral |   12 |        6 |
| qwen3   |   10 |        4 |

On a 4-way ring a reduce-scatter moves `(N−1)/N · D` per device and an all-reduce moves
`2(N−1)/N · D`. For llama2's 32.77 MB vocab gradient that is **24.58 MB vs 49.15 MB** — the wire
traffic for that one tensor genuinely doubles.

Three important limits on this half of the mechanism:

- **It is threshold-dependent, not universal.** At emb ≥ 2048 **both** modes form
  `%all-reduce-scatter.7`. qwen3 (V=151936, 77.79 MB/shard) has a bare all-reduce either way —
  `%all-reduce.25 bf16[512,151936]` 1700.5 µs/step auto vs `bf16[151936,512]` 1695.0 µs/step
  explicit, a 0.3% wash. So **the collective half applies to llama2 / mistral / mixtral / deepseek
  at emb 512 only** — 9 of the 25 measured configs (emb 512 × fsdp ≥ 4 × untied vocab); the relayout
  half (§4.3) is universal.

  Two successive explanations of *why* it bites there have been tested and discarded, so be careful
  with this paragraph's history. It is **not** "the FSDP axis became minor so reduce-scatter is
  impossible": the emb-2048 explicit dump contains
  `%all-reduce-scatter.7 (input.7: bf16[32000,2048]) → bf16[32000,512]` with
  `from-cross-replica-sharding="true"`, so XLA does reduce-scatter along dim 1 of the transposed
  shape. Nor is it the **all-reduce combiner** (`xla_tpu_ars_combiner_threshold_in_bytes`), which is
  what an earlier revision of this section claimed. The actual gate is in
  `tpu-all-reduce-scatter-fusion`: it refuses an AR→DS rewrite when the **per-shard extent of the
  minor-most dimension is below ~512 bytes** — one 128-lane × 32-bit vreg row — independent of
  dtype. Under `[V, emb]` at emb 512 the per-shard minor extent is 128 elements of bf16 = 256 bytes,
  under the bar; at emb 2048 it is 512 × 2 = 1024 bytes, over it. That is the whole emb-512-only
  story.

- **There is a flag, and it is a diagnostic rather than a fix.**
  `--xla_tpu_relayout_group_size_threshold_for_reduce_scatter` (default `INT64_MAX`, i.e. "never
  insert a relayout in order to enable a reduce-scatter"). Setting it to `1` recovers **every** lost
  reduce-scatter in the real MaxText module, which makes it a clean per-config isolation of
  mechanism A. Its sign is model-dependent — deepseek3-tiny 9885.2 → 9710.4 µs (−1.77%, 21.9% of the
  penalty), llama2 −0.5% — so it is an experiment, never a default.

- **It is recoverable, and that is not obvious from the paragraph above.** "The gradient is `[V, emb]`,
  so the shard dim is minor-most and the rewrite cannot match" describes the *symptom*, and an
  earlier revision treated it as terminal for explicit mode. It is not: the orientation is a
  consequence of the unfoldable transpose, so emitting the gradient dot in kernel order removes it at
  the source. With `lm_head_weight_grad_in_kernel_order: true` on llama2/shrink/fsdp4 the explicit
  dump gains a third reduce-scatter fusion, `%all-reduce-scatter.2`, which is byte-for-byte auto's
  `%all-reduce-scatter.5` — `bf16[512,32000] → bf16[128,32000]`, layout `{1,0:T(8,128)(2,1)}`,
  dim-0 dynamic-slice. `lm_head_kernel_transposed` does **not** recover it; it only fixes the
  relayout copies of §4.3. That difference is the whole reason the shipped flag is the `custom_vjp`
  one (§8 item 1).

- **The reduce-scatter is worth less than the bucket in §4.4 suggests.** It halves wire bytes but
  achieves only 91.7–99.3 GB/s versus 130.1–138.8 GB/s for the all-reduce, so the win is 0.66–0.76×
  time, not the ideal 0.50×. The causal control — `--xla_tpu_enable_all_reduce_scatter_fusion=false`
  applied to **auto**, which removes the reduce-scatter while changing nothing else — costs auto only
  **+47.2 µs** on llama2, ≤28% of that config's +168 µs penalty. §4.4's xprof bucket puts the same
  quantity at +126 µs, so **that bucket over-attributes mechanism A by ~2.7×** on llama2; the
  difference is time that overlaps with other work rather than adding to the critical path.

### 4.3 Consequence B — six relayout copies per step (universal)

`entry_computation_layout` is **identical** in both modes (llama2: 6× `f32[128,32000]{1,0:T(8,128)}`

- 6× `f32[32000,128]{1,0:T(8,128)}`) and the optimizer state is donated / input-output-aliased. But
  the `{0,1}` layout from the unfolded transpose propagates through the whole Adam update fusion. So
  explicit must convert `{1,0}→{0,1}` on the way in and `{0,1}→{1,0}` on the way out, for exactly
  three tensors — the `logits_dense` kernel and its Adam `mu` and `nu`:

```
%copy.264 = f32[128,32000]{0,1:T(8,128)} copy(%param.100)  op_name="state['optimizer']['opt_state'][0]['mu']['decoder']['logits_dense']['kernel'].value"
%copy.265 = f32[128,32000]{0,1:T(8,128)} copy(%param.101)  op_name="...['nu']['decoder']['logits_dense']['kernel'].value"
%copy.267 = f32[128,32000]{0,1:T(8,128)} copy(%param.55)   op_name="state['model']['decoder']['logits_dense']['kernel'].value"
%copy.268 = f32[128,32000]{1,0:T(8,128)} copy(%multiply_reduce_fusion.4#1)
%copy.269 = f32[128,32000]{1,0:T(8,128)} copy(%multiply_reduce_fusion.4#3)
%copy.270 = f32[128,32000]{1,0:T(8,128)} copy(%multiply_reduce_fusion.4#2)
```

Auto has **zero** of these in qwen3 / deepseek / qwen3_moe, and exactly **one** in llama2 / mistral /
mixtral. Layout census on llama2, same logical tensor:

|          | `f32[128,32000]{1,0}` | `f32[128,32000]{0,1}` |
| -------- | --------------------: | --------------------: |
| auto     |                    51 |                 **0** |
| explicit |                    18 |                **37** |

These are genuine data movement, not metadata. Both layouts are exactly tile-divisible under
`T(8,128)` (for `{1,0}` on `[128,V]`: 128/8 = 16, 32000/128 = 250; for `{0,1}`: 32000/8 = 4000,
128/128 = 1), so there is no padding component — it is a pure transposing gather at HBM speed. Cost
is **linear in `(d_model / n_fsdp_shards) × vocab_size`** and independent of depth, sequence length,
batch size and MoE-ness. Each copy touches the buffer twice (read + write):

| config                                        | bytes/step | predicted @ 2.37 TB/s |       measured |
| --------------------------------------------- | ---------: | --------------------: | -------------: |
| llama2 / mistral / mixtral (emb 512, V 32000) |   196.6 MB |                 83 µs |      60.2 µs\* |
| deepseek (emb 512, V 129280)                  |   794.3 MB |                335 µs |       336.2 µs |
| qwen3 (emb 512, V 151936)                     |   933.5 MB |                394 µs |       393.7 µs |
| w2048 / d16 / dp4 (emb 2048, V 32000)         |   786.4 MB |                332 µs | 345.1–351.5 µs |
| w4096 (emb 4096, V 32000)                     |   1.573 GB |                664 µs |       659.2 µs |

\* the 16 MB llama2 buffers are small enough to partially overlap; see the bandwidth caveat in §3.

**Added entry-copy bytes separate the sign of the effect perfectly across all 25 configs.** This is
the strongest single generalization in the study: the only three configs that add ~zero copy bytes
(gemma +2.1 MB, gemma2 +2.1 MB, tp4 −0.1 MB) are **exactly** the three non-regressions, and all 22
configs that add ≥ 66 MB regress. No exceptions in either direction.

The critical observation is that the copies do **not** slow the arithmetic — there is an internal
control in the same module. On dp4, the `token_embedder` Adam fusion (same 65.5 MB tensor, same
3 outputs) costs 177,290 ns vs 177,393 ns (0.06% apart), and the `logits_dense` Adam fusion itself
costs 165,019 ns vs 164,015 ns (0.6% apart). **The `{0,1}` layout costs nothing arithmetically; the
copies are purely additive work.**

### 4.4 xprof attribution, with the reclassification separated out

llama2, device-op time summed over 8 cores × 3 steps:

| class         |                  auto |              explicit |          Δ |  Δ per step |
| ------------- | --------------------: | --------------------: | ---------: | ----------: |
| collective    | 2,761,490 ns (13.76%) | 4,358,489 ns (21.20%) | +1,596,999 |     +532 µs |
| data-movement |    307,767 ns (1.53%) |    531,344 ns (2.58%) |   +223,577 |    +74.5 µs |
| fusion        | 6,321,712 ns (31.50%) | 5,067,125 ns (24.65%) | −1,254,587 |     −418 µs |
| other         |         10,678,565 ns |         10,602,167 ns |    −76,397 |      −25 µs |
| **total**     |         20,069,534 ns |         20,559,125 ns |   +489,591 | **+163 µs** |

Measured wall delta is +173 µs/step (4040.3 → 4213.6 µs); the device-op sum predicts +163 µs.

**But the collective row above is misleading.** Auto's reduce-scatter fusions are classified as
FUSION, which is simultaneously why "collective" rises and "fusion" falls. Re-doing the split on
leaf events and separating the two real mechanisms gives an exact 7-bucket decomposition
(each row sums to the measured window delta to the nanosecond):

| model      |   Δ step | vocab RS→AR | 6 relayout copies | all other comm | other data-mvmt | non-comm fusion | matmul/other |    idle | [RS→AR]+[copies] |
| ---------- | -------: | ----------: | ----------------: | -------------: | --------------: | --------------: | -----------: | ------: | ---------------: |
| llama2     | +173,242 |    +126,270 |           +59,776 |        −64,005 |         +28,102 |         +32,478 |         +386 |  −9,765 |       **107.4%** |
| mistral    | +213,052 |    +127,686 |           +73,665 |        −39,648 |         +25,960 |          +9,410 |       −1,629 | +17,608 |        **94.5%** |
| qwen3      | +334,399 |     −12,873 |          +394,852 |        −67,218 |         +31,761 |         −26,873 |       +6,435 |  +8,313 |       **114.2%** |
| deepseek   | +802,952 |    +347,521 |          +335,215 |        +12,135 |         +30,230 |         +61,710 |      −23,160 | +39,301 |        **85.0%** |
| mixtral    | +300,282 |    +128,674 |           +61,129 |              — |               — |               — |            — |       — |        **63.2%** |
| qwen3_moe  | +440,442 |           — |                 — |              — |               — |               — |            — |       — |        **85.4%** |
| qwen3_cmoe | +389,645 |           — |                 — |              — |               — |               — |            — |       — |       **102.0%** |

(ns, per profiled step window on `/device:TPU:0`.)

Two mechanisms account for **85–114%** of the regression in 6 of the 7 affected shrunk models
(mixtral 63%).

> **Caveat added after a causal control was run.** The `vocab RS→AR` column above is an
> *attribution*, not a measurement: it sums xprof durations for the collective that changed shape.
> Disabling the rewrite directly on **auto** (`--xla_tpu_enable_all_reduce_scatter_fusion=false`,
> which removes the reduce-scatter and touches nothing else) costs llama2 only **+47.2 µs**, against
> this table's +126,270 ns for the same quantity — the bucket over-states mechanism A by **~2.7×**,
> because much of that collective time overlaps other work instead of extending the critical path.
> The relayout-copy column does not have this problem: §4.3's independent bytes-÷-bandwidth
> prediction lands within a few percent of the measured value on 4 of 5 configs. Read the
> `[RS→AR]+[copies]` percentages as an upper bound, and the copies as the dominant term.

Of the *apparent* collective-time increase, **75–100% is reclassification** in every
model; the residual real communication delta is +62,265 (llama2), +88,039 (mistral), +359,656
(deepseek), +100,890 (mixtral) — and **negative** in 6 of 10 models (−80,090 qwen3, −142,763 gemma2,
−125,924 gemma3, −54,346 gemma, −53,401 qwen3_moe, −23,283 qwen3_cmoe).

Four alternative explanations were tested and **all four refuted**:

| hypothesis                                      | verdict                            | evidence                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| explicit overlaps less / reshards serialize     | **refuted**                        | occupancy (leaf union / window) 96.30→96.68, 97.57→97.27, 98.96→98.91, 97.35→97.16; idle delta is −5.6% to +4.9% of the step delta. The core is not idling more, it is executing more work.                                                                                                                                                                                                                                |
| tupled all-reduces hide latency worse           | **refuted, opposite sign**         | llama2 in-loop per iteration: auto = 4-operand AR (2.099 MB) 33,462 ns + five AR-scatter fusions (5.243 MB) 87,798 ns = 121,260 ns for 7.081 MB (58.4 GB/s). explicit = 7-operand AR (5.245 MB) 75,461 ns + two AR-scatter fusions (2.097 MB) 34,840 ns = 110,301 ns for 9.440 MB (85.6 GB/s). **Explicit moves +33% bytes in −9% time.** Same sign for mistral (−69,745 ns) and qwen3 (−66,514 ns). Bigger tuples *help*. |
| the transposed collective is slower on the wire | **refuted**                        | qwen3's `all-reduce.28` is the identical collective in both modes, differing only by transposition: `bf16[512,151936]` 1,702,101 ns vs `bf16[151936,512]` 1,689,229 ns = **−0.76%**, inside noise. All-gather totals: qwen3 +0.06%, deepseek +0.06%, llama2 −2.1%, mixtral −0.5%. **Transposing a collective is free on the wire**; the cost is the relayout copies it induces around it.                                  |
| MoE / `unreduced` machinery                     | **refuted — not exercised at all** | see §6.1                                                                                                                                                                                                                                                                                                                                                                                                                   |

### 4.5 Where the barrier comes from — *not* from MaxText's pin

**This section originally blamed MaxText's `out_sharding` pin. That was measured and is wrong.**
The pin is real, but removing it changes nothing: gating it on `shard_mode` the way
`norm_out_sharding` is gated produces a **0-line diff** in the explicit `after_optimizations` dump,
and all six relayout copies survive. Of the 1,276 `Sharding` custom-calls in the explicit llama2
module, **0** originate from the `out_sharding=` argument. Keep reading for what does cause them;
the description of the pin below is retained because it is what a reader will find in the code.

`src/maxtext/layers/decoders.py:805-810` computes an `out_sharding` for the logits
**unconditionally** (unlike `norm_out_sharding` two lines above, which *is* gated on shard_mode):

```python
if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
    out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
else:
    out_sharding = create_sharding(
        self.mesh,
        ("activation_embed_and_logits_batch", "activation_length", "activation_vocab"),
    )
```

and passes it into `linears.dense_general` at `decoders.py:843`.
`src/maxtext/layers/linears.py:309-311` is where it becomes explicit-only:

```python
# out_sharding should be None for auto mesh axis
if self.shard_mode != ShardMode.EXPLICIT:
    out_sharding = None
```

The real source is one level down, in JAX. On an all-`AxisType.Explicit` mesh,
`lower_with_sharding_in_types` (`jax/_src/interpreters/mlir.py:3170-3190`) annotates the output of
**every** sharding-in-types primitive — ~60 of them — with a `Sharding` custom-call. It early-returns
only for an empty, all-manual or all-auto mesh; there is no "the result sharding already equals the
propagated one, skip it" short-circuit. So the barrier between the gradient `dot_general` and its
`transpose` is emitted whether or not MaxText asks for anything. (Correspondingly,
`_dot_general_lower` at `jax/_src/lax/lax.py:6266-6292` accepts `out_sharding` and never reads it —
only `_dot_general_sharding_rule` uses it, to override the output aval.)

Two independent controls pin this down:

- One redundant `jax.lax.with_sharding_constraint` inserted between the dot and the transpose on an
  **auto** mesh reproduces the explicit artifact byte for byte.
- `auto` plus `--xla_disable_hlo_passes=algsimp` reproduces explicit's structure exactly.

So the causal chain is: *any* barrier between the two ops → `algsimp` cannot fold
`transpose(dot(A,B)) → dot(B,A)` → the weight gradient materializes as `[V, emb]`. The fold window is
gone for good by the time the partitioner runs (`algsimp` sits in `simplification-1` at pass 0008-0009;
`xla-partitioner` / `shardy-xla` at 0012-0016), so nothing later can recover it.

That is why the fix in §8 attacks the **orientation** rather than the barrier: the barrier is
structural to sharding-in-types, but a kernel already stored as `[V, emb]` needs no fold in the
first place.

This is still an instance of the general pattern recorded as finding [55] in the accompanying code
review — *pinning an intermediate blocks XLA from propagating the constraint backwards into the
producer* — but here the pin is JAX's, not MaxText's, and it manifests through the **algebraic
simplifier and layout assignment** rather than through sharding propagation.

### 4.6 The natural control: tied embeddings

gemma and gemma2 are **faster** under explicit (−1.39%, −0.66%). Both set
`logits_via_embedding: true`, so `apply_output_head` takes the `attend_on_embedding` branch
(`decoders.py:813-829`) and **never reaches `logits_dense` at all**. There is no separate vocab
weight-gradient dot, no reduce-scatter to lose, and no `f32[emb/4, V]` optimizer chain to relayout.

This is the cleanest evidence for the mechanism: **remove the untied vocab projection and the sign
of the effect flips.** Across the 7 non-tied matrix models the mean regression is +4.5%.

**The all-gather dimension flip that replaces it is free.** All three gemmas show

```
auto:      %all-gather.30 = bf16[262144,512]{1,0:T(8,128)(2,1)} all-gather(%convert_element_type.862), dimensions={1}
explicit:  %all-gather.29 = bf16[512,262144]{0,1:T(8,128)(2,1)} all-gather(%convert_bitcast_fusion),  dimensions={0}
```

HLO layout is minor-to-major, so `{1,0}` on `[262144,512]` and `{0,1}` on `[512,262144]` both put
512 minor. Same physical `(262144, 512)` grid, same `T(8,128)(2,1)` tiling, and the gather axis is
the physical-minor 512 axis (exactly one 128-lane tile per device) either way. It is a pure logical
relabel, and it measures that way — per-step durations over 3 profiled steps:

| model  |                        auto |                    explicit |                     Δ |
| ------ | --------------------------: | --------------------------: | --------------------: |
| gemma3 | 1225.1 / 1223.3 / 1224.8 µs | 1223.7 / 1225.1 / 1224.9 µs | **+0.2 µs (+0.016%)** |
| gemma  | 1206.6 / 1206.1 / 1206.3 µs | 1204.7 / 1207.6 / 1204.9 µs |      −0.6 µs (−0.05%) |
| gemma2 |                   1205.7 µs |                   1204.0 µs |      −1.7 µs (−0.14%) |

**But the flip does cost something one instruction upstream, and this is a universal tied-embedding
tax.** `src/maxtext/layers/embeddings.py:235-240` does
`jnp.asarray(embedding_table, jnp.bfloat16).T` inside `attend_on_embedding`. Under auto XLA folds
the `.T` into the dot's dimension numbers (`dim_labels=bf_oi->bf`), so **one** convert serves both
consumers — `%convert_element_type.862 = bf16[262144,128]` feeds both the input `%gather_fusion` and
the tied-output `%all-gather.30`. Under explicit the transposed value is a distinct typed aval, so
XLA emits `dim_labels=bf_io->bf` and needs a second, differently-shaped value:

```
%convert_element_type.396 = bf16[262144,128]{1,0} convert(%param.193)        [81.8 µs] -> %gather_fusion
%convert_bitcast_fusion   = bf16[128,262144]{0,1} fusion(%param.193)         [82.3 µs] -> %all-gather.29
    fused body: convert(f32[262144,128]) -> ROOT bitcast(bf16[128,262144])
```

The transpose really is a bitcast (free); the **duplicated f32→bf16 conversion of the 128 MB shard
is not**: +82.3 µs (gemma3), +80.6 µs (gemma), +81.2 µs (gemma2) — about 0.75% of a step, in
duplicated HBM traffic, paid by every tied-embedding model including the two that win overall.

What actually carries gemma/gemma2's win is one memory-bound op: the tied-embedding logits matmul,
**−116.6 µs (gemma)** and **−93.4 µs (gemma2)**, i.e. 69% and 78% of their entire net gain. Full
device-op ledger: fusion −314.9 / −433.9 µs against all-reduce + copy penalties of +162.7 / +316.5,
netting −169.1 / −119.7 µs of device-op delta versus measured step deltas of −150.7 / −92.9 µs.

**Unexplained, and flagged rather than asserted:** that same logits matmul is **+47.1 µs in gemma3**
(auto max 772.7 µs < explicit min 788.4 µs, so it is real and not overlap). The HLO transformation
is byte-for-byte the same shape in all three gemmas — rhs `[V,E]{1,0} bf_oi->bf` becomes
`[E,V]{0,1} bf_io->bf`, same physical operand, same `bf16[1024,V]{1,0}` output, same fused epilogue.
The most likely cause is HBM contention (gemma3/explicit has 132 MB/step of extra layout-copy
traffic competing with a 750 µs memory-bound op), but that cannot be proven from a 3-step profile.
It is 10% of gemma3's delta and 69–78% of gemma/gemma2's gain, so it is not a small caveat.

### 4.7 gemma3's +4.49% is an artifact of the shrink, and it reverses at real depth

gemma3 has `logits_via_embedding: true`, identical `bf16[262144,512]` convolutions in both modes,
and **zero vocab relayout copies**. Its communication goes *down* by 125,924 ns. The unembedding
mechanism explains **0% of gemma3**. What it has instead is a structural change nothing else in the
sweep has: **+1243 HLO instructions** (11,209 → 12,452). For comparison, every other model's
instruction delta is between −64 and +71.

**Root cause: the shrunk model never gets scanned.**
`src/maxtext/models/gemma3.py:39-46` defines `GEMMA3_ATTENTION_PATTERN` with length 6, and
`src/maxtext/layers/decoders.py:1341` computes `scan_length = num_layers // 6`. At
`base_num_decoder_layers=4` that is **0**, the main scan is skipped entirely, and all four layers
fall into the unscanned `layers_remainder` block (`decoders.py:1386-1391`). Every parameter in the
gemma3 dumps is named `state['model']['decoder']['layers_remainder']['layers_N'][...]` — 262 such
references in auto, 298 in explicit, and zero scanned-layer references. **gemma3 is the only model
in the matrix compiled with `while = 0`;** gemma and gemma2 both scan normally.

Unrolling costs two things that only bite under explicit:

1. **Remat stops being CSE'd.** `remat_policy` defaults to `'full'` (`base.yml:374`). When the block
   is unrolled, the recomputed forward and the original forward live in the *same* HLO computation,
   so XLA can CSE them — and under auto it does: only **61 of 11,209** instructions still carry
   `rematted_computation` metadata. Under explicit the interposed Sharding custom-calls prevent it:
   **1,013 of 12,452**. That is **952 of the +1243 extra instructions (76.6%)**. Directly
   observable: `splash_mha_fwd_segmented_residuals` appears 4× in auto (= 4 layers) but **7×** in
   explicit (4 forward + 3 recomputed), 91.7 → 158.3 µs; `convolution` goes 34 → 54 with 16 of the
   20 extra tagged `rematted_computation`. Device time of remat-tagged top-level ops: **auto 0.0 µs
   (n=0), explicit 212.9 µs (n=75)**.
2. **Every parameter gets its own entry/exit layout copy.** Explicit picks `{0,1}` for 1,202 tensors
   (auto: 475; auto prefers `{1,0}` 1,724× vs explicit 1,250×). Module entry/exit buffers are
   `{1,0}` and donated, so each mismatch is a real copy — and the counts match the unroll exactly:
   48× `f32[128,1024]` (4 layers × {wi_0, wi_1} × {param, mu, nu} × {in, out}), 36× `f32[128,8,128]`
   (q/k/v), 24× `f32[8,128,128]`, 24× `f32[1024,128]` (wo). ~66 MB extra copied per step
   (132 MB HBM traffic) = the observed **+132.6 µs**. Inside a `while` body the loop-carried buffers
   have one fixed layout, so this tax does not exist for gemma/gemma2 — or for gemma3 at ≥ 6 layers.

**The decisive experiment.** Same harness, same flags, only `base_num_decoder_layers` changed:

| layers | structure                  |       auto |   explicit |                      Δ | remat-tagged (auto/expl) | Δ instructions |
| -----: | -------------------------- | ---------: | ---------: | ---------------------: | -----------------------: | -------------: |
|      4 | `scan_length=0`, `while=0` | 10659.8 µs | 11144.3 µs | **+484.4 µs (+4.54%)** |                61 / 1059 |          +1243 |
|      6 | `scan_length=1`            | 12827.8 µs | 13067.8 µs |     +240.1 µs (+1.87%) |              1943 / 1952 |            −71 |
|     12 | `scan_length=2`, `while=2` | 18093.2 µs | 17987.6 µs | **−105.6 µs (−0.58%)** |              2519 / 2552 |            −18 |

L4 reproduces the original +4.49% to within 0.05 pp. The remat asymmetry that drives item 1
disappears at exactly the point the scan appears, and so does the instruction delta. Opcode ledger
across the ladder:

```
  fusion delta      +115.7 (L4)  ->  -193.3 (L6)  ->  -596.7 (L12)
  custom-call delta  +70.4 (L4)  ->    +2.7 (L6)  ->   -16.6 (L12)
  copy delta        +132.6 (L4)  ->  +146.2 (L6)  ->   +59.8 (L12)
  all-reduce delta  +112.4 (L4)  ->  +218.6 (L6)  ->  +378.7 (L12)   <- universal tax, GROWS with depth
```

Attribution of the +486.4 µs at L4 (per-opcode ledger sums to +442.5 µs = 91.0%; residual ~44 µs is
inter-op scheduling gaps): un-CSE'd remat **279.5 µs (57%)**, entry/exit layout copies
**132.6 µs (27%)**, gradient all-reduce repack **112.4 µs (23%)**, duplicated embedding convert
**82.3 µs (17%)**, logits matmul **47.1 µs (10%)**, offset by a **−192 µs** win on weight-gradient
dot relabels. Items 1 and 2 (412 µs, 85% of the delta) exist **only** because `scan_length == 0`.

Three things the +4.49% is *not*, all checked directly:

- **Not multimodal.** `use_multimodal` defaults false (`base.yml:1250`). Neither dump contains a
  single vision tensor — no 1152 or 4304 hidden dims, no 14×14 patch convolution, zero ops with
  image/vision/patch metadata.
- **Not local/global alternation.** With 4 layers, `get_attention_type(0..3)` returns
  `GEMMA3_ATTENTION_PATTERN[0..3]` = `LOCAL_SLIDING` for all four — the shrunk model has **zero**
  global layers, so the alternation never happens. Both dumps carry the same two `s32[1,512,512]`
  mask constants, and the per-layer FSDP weight all-gathers (`all-gather.60-.83`, `bf16[512,8,128]`,
  6 per layer) are 24 in auto vs 25 in explicit and *cheaper* under explicit (245.3 → 223.7 µs). It
  is the pattern **length** (6 > 4), not the alternation, that matters.
- **Not the 262144 vocab.** gemma and gemma2 have the same tied-embedding structure, the same 256 MB
  all-gather and the same 268 MB gradient all-reduce — and the vocab path is precisely where they
  *win*.

**Practical consequence: do not benchmark gemma3 below 6 layers, and always compare `while`-loop
counts before comparing dumps.** The one durable gemma3 finding is the last ledger row: the
gradient all-reduce repack is a real tax that **grows with depth** (+112 → +219 → +379 µs), even
though the total flips negative because the fusion win grows faster.

______________________________________________________________________

## 5. Results

### 5.1 Onboarded-model matrix

Shrunk to fit a v5p-8 while preserving the structural features that drive sharding (MoE, GQA, MLA,
local/global attention): emb 512, mlp 1024, 4 layers, 8 heads, head_dim 128, seq 1024, pdbs 1,
fsdp 4. These are *deliberately small* — see §5.3 before drawing conclusions from the magnitudes.

| model                  | tied emb | auto µs | explicit µs |     Δ step |                 mechanism coverage |
| ---------------------- | :------: | ------: | ----------: | ---------: | ---------------------------------: |
| `deepseek3-tiny`       |    no    |    9088 |        9890 | **+8.84%** |                                85% |
| `mistral-7b`           |    no    |    3890 |        4103 | **+5.48%** |                                95% |
| `gemma3-4b` ⚠          | **yes**  |   10831 |       11317 | **+4.49%** | 0% — **shrink artifact, see §4.7** |
| `llama2-7b`            |    no    |    4040 |        4214 | **+4.29%** |                               107% |
| `qwen3-8b`             |    no    |    8264 |        8598 | **+4.05%** |                               114% |
| `qwen3-30b-a3b`        |    no    |   14484 |       14924 | **+3.04%** |                                85% |
| `mixtral-8x7b`         |    no    |   10105 |       10405 | **+2.97%** |                                63% |
| `qwen3-custom-30b-a3b` |    no    |   14488 |       14878 | **+2.69%** |                               102% |
| `gemma2-2b`            | **yes**  |   14160 |       14067 | **−0.66%** |                          n/a (win) |
| `gemma-2b`             | **yes**  |   10839 |       10688 | **−1.39%** |                          n/a (win) |

"Mechanism coverage" = fraction of the step delta attributed to [lost reduce-scatter] + \[six relayout
copies\] by the per-op decomposition in §4.4.

⚠ **gemma3's row does not belong in this table.** At 4 layers gemma3 compiles with `while = 0`
(§4.7) — a different code path from the model at any realistic depth. Re-run at 12 layers it is
**−0.58%**. Read the matrix as 9 models plus one methodology finding.

> **Removed from this table:** the "Δ collective bytes" and "collective count" columns that appeared
> in the first revision of this document. Both are artifacts — see §3 caveats 1 and 2. Explicit's
> *lower* collective count reflects auto's reduce-scatters being counted as all-reduces inside their
> fusions, and the "≈ 0% byte delta" is the byte counter reading pre-slice shapes.

### 5.2 Parallelism strategy (llama2 / mixtral, same shrunk model, 4 chips cut different ways)

| config                     | auto µs | explicit µs |                   Δ step | note                                 |
| -------------------------- | ------: | ----------: | -----------------------: | ------------------------------------ |
| `ici_data_parallelism=4`   |    4442 |        4892 |              **+10.12%** | worst case                           |
| `ici_fsdp_parallelism=4`   |    4059 |        4220 |                   +3.96% | the default                          |
| `fsdp=2 × tensor=2`        |    4335 |        4463 |                   +2.95% | same six copies, as `f32[256,16000]` |
| `mixtral fsdp=4`           |   10112 |       10405 |                   +2.89% | 72% attributed                       |
| `mixtral expert=4`         |   10732 |       10987 |                   +2.38% | 98% attributed                       |
| `fsdp=4, seq 4096`         |   12058 |       12134 |                   +0.63% |                                      |
| `ici_tensor_parallelism=4` |    4563 |        4535 | **−0.61%** (true −0.21%) | only win; different mechanism        |

**Pure DP is the worst case and it is almost pure copy overhead.** With no FSDP the vocab parameter
is *fully replicated*, so the transposing copies operate on `f32[512,32000]` = 65.5 MB each. The six
copies measure **350,463 ns of the +449,471 ns delta = 78.0%**. Widening to the whole optimizer
tail: **+380,640 ns = 84.7%** of the delta occurs *after* the gradient all-reduce finishes, and
92.1% of that tail delta is the six copies. The remaining 15.3% is pre-tail and is *not* relayout:
+35,861 ns is a single **4-byte `s32[]` all-reduce** (`all-reduce.9`) whose duration is pure
inter-core wait, and +27,378 ns is the splash-attention forward Pallas kernel (identical shapes,
identical bytes, most plausibly scheduling skew).

Note the counter-intuitive detail: explicit's *non-vocab* copies are **cheaper** — per-layer
`copy.283-291` = 73,501 ns vs auto's `copy.199-203` + `copy_dynamic-update-slice_fusion.6-8` =
107,610 ns. The instruction-count delta ("copy 27 → 35") overstates the story; only 6 of the 8 matter
(441.5 MB vs 44.1 MB of copy output bytes — a 10× byte increase behind a 1.3× count increase).

**Pure TP is the only win, and it has nothing to do with the vocab tensor.** Two corrections to the
obvious reading:

- **Magnitude.** The −0.61% headline is inflated ~3× by the module-launch lead gap (§3 caveat 3).
  Measured on the op span, or on TPU:1's module span, the delta is **−9.2 to −9.3 µs = −0.21%**.
  TPU:1 medians: auto 4,475,492 ns vs explicit 4,466,266 ns. Still a genuine win, still the only one,
  but not 0.6%.

- **Mechanism.** It is the RMSNorm **scale** parameter and its AdamW state. Under TP=4 the scale is
  a scanned parameter (`param_scan_axis=1`) whose *entry* layout is replicated in both modes
  (`f32[512,4]{1,0} parameter(23), sharding={replicated}` — byte-identical). But the activation fed
  to the norm is `tensor`-sharded on its last axis, so the VJP w.r.t. `scale` produces a
  `tensor`-sharded `[128]` cotangent that must become replicated. Somebody has to all-gather it; the
  modes differ only in **where**.

  *Explicit* gathers inside the scan. `src/maxtext/layers/normalizations.py:34-51`
  `_align_scale_with_normalized_axis`, gated at `:116-117` on `ShardMode.EXPLICIT`, fires because
  `scale_spec[-1] is None != activation_spec[-1] == 'tensor'`:

  ```python
  return jax.sharding.reshard(scale, jax.sharding.PartitionSpec(activation_spec[-1]))
  ```

  visible in the pre-opt dump as
  `%reshard.37 = bf16[512]{0} custom-call(%add.53), custom_call_target="Sharding", sharding={devices=[4]<=[4]}, metadata={op_name="reshard"}`
  (plus `%reshard.25`, `.96`, `.119` in the remat body and `.147` for `decoder_norm`). The gather is
  pinned at that program point — inside the backward scan, in bf16.

  *Auto* has **zero** resharded scales (65 `Sharding` custom-calls, none with `op_name="reshard"`).
  GSPMD keeps the cotangent `tensor`-sharded and propagates that through the *entire* elementwise
  optimizer chain, doing AdamW on `f32[128,4]` shards — then has to honour the mandated replicated
  entry/exit layout at the module boundary, so it sinks **nine f32 all-gathers into the epilogue**,
  back-to-back, with nothing to overlap them with. And because a value produced by a collective
  cannot be written in place into a donated input buffer, copy-insertion adds **5 prologue + 9
  epilogue copies** that explicit does not need.

  Phase decomposition of the median step (split at the scan `while` boundaries):

  ```
    pre-while   -5,633 ns   (auto's 5 donated-param copies measure 5,540 ns -> 98% attributed)
    while      +41,647 ns   (explicit's 8 in-loop bf16 all-gathers = 34,325 ns;
                             bitcast_convert_fusion.7/.8 +4,173; dynamic_update_slice.194/.195 10,688)
    post-while -45,300 ns   (auto's 9 epilogue all-gathers 37,654 ns + 9 epilogue copies 10,268 ns)
    ------------------------
    net         -9,286 ns
  ```

  Three honest caveats. (i) Explicit's in-scan gathers are **not** well overlapped — 34,325 of the
  +41,647 ns while-loop regression is the gathers themselves. Explicit wins by moving a serial cost
  from 100%-exposed to ~80%-exposed, which is a weaker story than "explicit enables overlap".
  (ii) Explicit does strictly **more** arithmetic — it runs AdamW on replicated `f32[512,4]` instead
  of `f32[128,4]`, 4× redundant FLOPs — and only wins because at 2 KB these fusions are
  tile-quantized anyway (2,238 ns replicated vs 4,109 ns sharded). **At a real embedding width that
  sign could flip.** (iii) "Explicit moves fewer bytes" (−52,224 B of 63.1 MB = 0.08%) is true but
  causally irrelevant: at 1–8 KB messages the ICI cost is pure per-message latency, ~4.2 µs either
  way.

  So GSPMD's choice here is a mild, real, ~0.2% mistake — defensible in isolation (sharded Adam =
  4× fewer FLOPs, sink-the-collective-to-the-boundary is standard) — not evidence that GSPMD is
  broadly worse under TP.

### 5.3 Scale — the ladder is non-monotone

llama2 decoder, fsdp 4, varying one dimension at a time:

| config                          | auto µs | explicit µs |                Δ step | measured copy cost | copy / Δ |
| ------------------------------- | ------: | ----------: | --------------------: | -----------------: | -------: |
| emb 512, 4 layers, seq 1k       |    4008 |        4200 |  **+4.80%** (+192 µs) |            60.2 µs |      32% |
| emb 2048, 4 layers, seq 1k      |   22399 |       22614 |      +0.96% (+216 µs) |           351.5 µs |     163% |
| emb 4096, 4 layers, seq 1k      |   72210 |       73283 | **+1.49%** (+1073 µs) |           659.2 µs |      61% |
| emb 2048, 4 layers, **seq 4k**  |   45492 |       46286 |      +1.74% (+794 µs) |             351 µs |      44% |
| emb 2048, 4 layers, **seq 8k**  |   86953 |       87229 |      +0.32% (+276 µs) |             340 µs |     123% |
| emb 2048, 4 layers, **pdbs 4**  |   41233 |       41599 |      +0.89% (+366 µs) |             351 µs |      96% |
| emb 2048, **16 layers**, seq 1k |   80398 |       80434 |   **+0.04%** (+36 µs) |           345.2 µs |     972% |
| mixtral emb 2048, 4 layers      |   96076 |       96620 |      +0.57% (+544 µs) |             364 µs |      67% |

Read that last column carefully. **The copies are always there and always cost roughly what
bandwidth says they should — but how much of that reaches wall clock varies from 32% to 972%.**

Three conclusions, replacing the "it amortizes" claim in the first revision of this document:

1. **The width ladder reads non-monotone — because it spans two regimes.** +4.80% → +0.96% →
   **+1.49%** along width; +0.96% → +1.74% → +0.32% along sequence. The largest *absolute* penalty
   in the entire 28-run sweep (**+1111 µs** against a 2-run auto mean of 72,171 µs) is at the
   **largest** width. A fixed-cost model calibrated at w512 (+192 µs) predicts d16 = +0.24% and
   w4096 = +0.27%; measured are +0.088% and +1.54%, i.e. **wrong by 2.7× low and 5.7× high**.

   *Correction (2026-09-02):* a single-mechanism model was never going to fit this, and the reason is
   structural rather than statistical. **w512 is the only width below XLA's 512-byte minor-most AR→RS
   gate** (§4.2: bf16 needs `emb/n_fsdp ≥ 256`, and 512/4 = 128). It therefore pays mechanism A *and*
   mechanism B; w2048 and w4096 pay only B. Split that way the ladder is monotone within each regime
   — the above-gate rungs rise +0.96% → +1.49% with width, exactly as the relayout-copy model says
   they should, and the below-gate rung is a different program. **Read every emb-512 number in this
   document as an upper bound that production geometry does not reach.**

2. **d16's near-zero is cancellation, not absence.** The copies still cost 345.2 ± 0.6 µs (4 runs).
   They are offset by an unrelated **−242 µs** win inside the layer scan and a **−390.5 µs**
   `concatenate` swing (auto 778.1 µs → explicit 387.6 µs). Had the loop been neutral, d16 would
   read +313 µs = **+0.39%**, ~4× the headline. Pooling all 4 runs per mode (rep1-3 + scale/d16,
   stdev ~22 µs each) gives **+70.7 µs = +0.088%, t = 4.5** — significant, not noise.

3. **A second, variable-sign mechanism lives inside the scan.** The out-of-loop budget is fully
   explained (copies are 110% of the d16 out-of-loop delta), but the in-loop residual is **−242 µs
   at d16, +477 µs at s4k, +409 µs at w4096** — no consistent sign. Its signature is an operand
   orientation flip in the attention/MLP backward: auto materializes 2× `bf16[528,8192]` + 1×
   `bf16[8192,512]` where explicit materializes 1× + 2× (the 528 = 512 embed shard + 16-row halo,
   which drags in `concatenate bf16[576,8192]` and `collective-permute bf16[48,8192]`).

Regressing step delta on measured copy time over 12 configs: **`Δ = −8.7 + 1.37·copy_µs`,
R² = 0.449**, per-config residual −274 to +467 µs (stdev 240 µs). Mean copy cost 330.5 µs vs mean
delta 443.0 µs — so on *average* the copies explain essentially the whole penalty, but they predict
any individual config only about half the time.

**What does amortize, in relative terms** — the copy component alone: 1.51% → 1.57% → 0.91% along
width; 1.57% → 0.43% along depth (L4→L16); 1.57% → 0.77% → 0.39% along seq (1k→4k→8k); 1.57% → 0.85%
along batch. So it amortizes over depth, sequence and batch, and only weakly over width.

Extrapolating to a real llama2-7B (emb 4096, 32 layers, seq 4096, 4-way FSDP): the copy cost stays
around 659 µs on a step that scales to roughly 2.5 s, i.e. **~0.03%**. That extrapolation is one
order of magnitude beyond anything measured here and should be treated as an estimate, not a result.

______________________________________________________________________

## 6. What explicit sharding buys you

These come from a systematic read of the sharding core, the model layers, the MoE path, and the
test suite (125 candidate findings, 20 upheld under adversarial verification).

### 6.1 `reduced=` / `unreduced=` PartitionSpec tags — a real capability, unexercised here

**First, the disclaimer, because it is easy to misread this section:** these tags account for
**0.0% of every measured number in this document.** `src/maxtext/layers/moe.py` contains **zero**
occurrences of `reduced` or `unreduced` (its only two `shard_mode` branches are an `output_sharding`
at `:391-395` and a router-weight scatter `out_sharding` at `:2717-2721`). The tags exist in exactly
two places — `src/maxtext/models/deepseek_batchsplit.py:231-254, 323-378, 436-452` (model
`deepseek3-671b-batchsplit`, never run here) and `src/maxtext/utils/gradient_accumulation.py:198-209`
— and both are gated at `gradient_accumulation.py:77`:

```python
data_parallel_active = (
    config.shard_mode == ShardMode.EXPLICIT and param_mesh.shape.get("data", 1) > 1
)
```

Every matrix / parallelism / scale run used `gradient_accumulation_steps: 1` (`base.yml:986`) and a
pure fsdp4 or ep4 mesh (`data = 1`), so this was **never True**. There are no fewer and no smaller
gradient all-reduces under explicit anywhere in the data.

**What the capability actually is.** JAX's `_check_unreduced` (`shard_map.py:457-465`) rejects
`reduced=`/`unreduced=` tags unless *every* mesh axis is `AxisType.Explicit`, which makes them
categorically unavailable under auto. `_unmentioned2` computes
`name_set = _spec_to_vma(spec) | spec.unreduced`, and `to_ct_spec()` swaps reduced ↔ unreduced
(`partition_spec.py:238-242`). A `reduced={'data'}` in-spec removes `data` from `shard_map`'s
defensive `psum`, leaving the cotangent partial — which lets the all-reduce be **hoisted out of a
gradient-accumulation scan**. N per-microbatch all-reduces of `E × M × H` floats collapse to one.

`deepseek3-671b-batchsplit.yml:61` is `shard_mode: explicit` for exactly this reason:
`deepseek_batchsplit.gather_weights` (`deepseek_batchsplit.py:125-259`) runs
`jax.lax.pcast(w, axis_name='data', to='reduced')` and `jax.lax.all_gather(..., to='reduced')` inside
a `shard_map` whose `out_specs` carry `reduced={'data','fsdp','expert'}`, with the backward
counterpart at `:263-384` carrying `unreduced={...}`. **This config cannot trace under auto.** It is
not a preference; it is a compile-or-die requirement, and it is the fullest expression of what
explicit mode is for.

Caveat: on this branch the tag is largely dead code outside `deepseek_batchsplit`. Several helpers
still index/iterate `PartitionSpec` directly (`sharding.py:193`, `:242`, `:446-449`), which raises
once a spec carries a tag — `P.__getitem__`/`__iter__` are disabled for tagged specs
(`jax/_src/partition_spec.py:164-176`). Only `_truncate_pspec` (`sharding.py:398-406`) was made
tag-safe. This is why the optimization has been added and reverted twice.

### 6.2 Larger, merged collectives (the count reduction is *not* the win)

Explicit tuples gradient all-reduces: 11 → 7 on llama2, 11 → 8 on gemma/qwen3/mistral, 16 → 10 on
gemma2, 25 → 13 on gemma3.

**Do not read the lower count as a win.** As §3 caveat 1 explains, most of the "missing" all-reduces
are the ones auto had fused into reduce-scatters; the counter is an anti-signal. Similarly deepseek
is the one model where explicit has *more* collectives (51 → 52, all-reduce 11 → 12) and that extra
one is a **benefit**: `%all-reduce-scatter.4` producing `bf16[128,8,128]`, and the pair
auto `all-reduce.143` 27.2 µs → explicit `all-reduce.165` 18.3 µs is **8.9 µs/step faster**.

Where tupling *is* a genuine win, it is measurable: llama2's in-loop tuple moves **+33.3% wire bytes
in −9.0% time** (85.6 GB/s vs 58.4 GB/s), −43,835 ns over 4 iterations; mistral −69,745 ns; qwen3
−66,514 ns. Today that win is swamped by §4.

**But bigger tuples are not universally better, and the counter-example scales with depth.** On
gemma3 the repack is a cost: auto has 25 all-reduces totalling 297,815,400 B (one 33-element tuple
`all-reduce.195` = 8,408,064 B at 105.8 µs, plus 20 standalone 1 MB ARs); explicit has 13 totalling
297,815,516 B (one 45-element tuple `all-reduce.212` = 20,990,976 B at 249.7 µs, plus 8 standalone).
Total bytes differ by 116 B (0.00004%), but the tuple grew **2.50× in bytes and 2.36× in time
(+143.9 µs)** — more traffic pulled into a single blocking collective with nothing left to overlap
it. And unlike everything else in §4.7, this tax **grows with depth**: +112.4 µs at L4, +218.6 at
L6, +378.7 at L12. Caveat: 20 of auto's 25 and 8 of explicit's 13 all-reduces never appear in the
profile's `XLA Ops` line at all; the omission is symmetric, but the ledger only accounts for the
5 timed collectives in each run.

### 6.3 Errors instead of silent pessimization

A mismatched ZeRO-1 layout is a hard type error under explicit; under auto GSPMD "papers over it with
an extra collective" (`src/maxtext/configs/types.py:4548-4560`, the repo's own words). Explicit turns
a class of silent performance bugs into compile failures.

______________________________________________________________________

## 7. What explicit sharding costs you

### 7.1 Capabilities that are hard-blocked

| capability                                       | status under explicit                                                                                           | evidence                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `check_vma=True`                                 | **rejected by MaxText** (`types.py:3089-3090`) although JAX 0.11.1 accepts it on an Explicit mesh               | `base.yml:704` calls it "recommended for improved performance" |
| `fused_qkv=True`                                 | **trace-time `ShardingTypeError`** — `attentions.py:1252` is the one projection that forwards no `out_sharding` | no validator catches it; you get a raw JAX error               |
| context parallelism on `dot_product` / GPU flash | **trace-time error** — `qk_product` puts the same `context` axis on both `t` and `s` (`attention_op.py:2749`)   | TPU splash escapes via `shard_map` manual axes                 |
| `P.UNCONSTRAINED`                                | **illegal** when no axis is Auto; the pipeline substitutes `None`, which means *replicate*, the opposite        | `pipeline.py` placeholder swap                                 |
| partial adoption                                 | **impossible** — mesh axis types are all-or-nothing (`maxtext_utils.py:2426-2429`)                              |                                                                |

`check_vma` is the most consequential: it gates the varying-manual-axes fast path in megablox
(`ops.py:296`, `ops.py:665`, `backend.py:528`, `backend.py:789`) on exactly the MoE models that are
on the explicit whitelist. Note the validator is also incomplete — `pipeline.py:385`, `:392`, `:1186`
hardcode `check_vma=True` regardless of config, and nothing forbids
`ici_pipeline_parallelism > 1` with explicit.

### 7.2 Silent losses

- **`with_sharding_constraint` becomes a zero-HLO no-op assert** under an all-Explicit mesh. Any
  sharding hint not routed through `maybe_shard_with_name` simply vanishes. Confirmed live sites:
  `maxtext_utils_nnx.py:175` (init-time), `attention_op.py:2567-2579` and `:2619-2693`
  (decode/prefill), `kvcache.py:532/541/602/604/846/847`, `qwen3.py:632`, `diloco_sharding.py:78`.
  Training is unaffected (those branches are gated on non-train model modes), but decode and prefill
  under explicit are running without their sharding hints.
- **Pinned intermediates can multiply matmul FLOPs.** Reproduced standalone: `x = P('fsdp',None,None) [8,256,512]`, `w` replicated `[512,2048]`, `einsum('bse,ef->bsf')`, result constrained to
  `P('fsdp','tensor',None)`. Auto emitted `f32[4,64,512] fusion(... dynamic-slice ...)` — it sliced
  the *operand* before the matmul. Explicit emitted `f32[1024,2048] ynn_fusion(...)` then sliced
  after: **4× the per-device matmul FLOPs** on that op. This is the same class of defect as §4.1.
- **`mistral`/`qwen3` pin the MLP intermediate with `activation_norm_length`** while
  `llama2`/`gemma`/`gemma2`/`gemma3` use `activation_length`. Under flax's first-come mesh-axis
  consumption these resolve differently — `activation_norm_length` ends up **replicating** the length
  axis. Under context parallelism that is an extra all-gather per `wi` projection per layer (2/layer
  for gated MLPs) plus the backward reduce-scatter. At `context=1` they coincide, which is why the
  matrix above does not show it.

### 7.3 No parity guarantee exists

Both golden suites run **auto only**. `tests/unit/sharding_compare_test.py:124-135` never sets
`shard_mode`; neither does the case table at `tests/utils/sharding_dump.py:39-82` nor the generator
at `tests/utils/run_sharding_dump.py:66-88`. The HLO regression test
`tests/integration/hlo_diff_test.py:105-136` parametrizes `compile_topology` and layer count only.

**Consequence:** every divergence in this document is invisible to CI. A change that doubled
collectives under explicit but left the loss curve intact would pass the entire test tree. The repo
already has the machinery (`collective_lines()` in `tests/utils/hlo_test_utils.py`) — it is simply
not pointed at explicit mode.

### 7.4 A dead validation gate

`validate_shard_mode(shard_mode, decoder_block, quantization)` rejects explicit + quantization and
whitelists only `{simple, simple_mlp, llama2}`. Its only caller is
`pyconfig_deprecated.validate_keys`, reachable only from the legacy `_HyperParameters.__init__`. The
live entry point (`pyconfig.initialize` → `_initialize_pydantic` → `types.MaxTextConfig`) never runs
it, and `types.py` says nothing about quantization + `shard_mode`. So **`shard_mode=explicit` +
`quantization=int8` now passes validation and compiles**, even though the AQT/Qwix paths were never
onboarded.

______________________________________________________________________

## 8. Recommendations

### For users choosing a mode today

- **Tied-embedding models (`logits_via_embedding: true`):** explicit is already as fast or faster —
  including gemma3, once it is deep enough to scan (−0.58% at 12 layers). Nothing to wait for.
- **gemma3 specifically:** never benchmark it below 6 layers. `scan_length = num_layers // 6`, so a
  4-layer proxy unrolls the whole decoder and measures a different program (§4.7).
- **Untied dense models at production depth/sequence/batch:** the penalty is a few tenths of a
  percent. Choose on capability. If you need `check_vma`, `fused_qkv`, or context parallelism on
  non-TPU attention, you must use `auto`.
- **Wide models (emb ≥ 4096):** do *not* assume scale hides the cost — it is the configuration with
  the largest absolute penalty measured. Profile before committing.
- **MoE with gradient accumulation and `data > 1`:** explicit is the only mode that can express
  deferred expert-weight all-reduce. This is the case explicit exists for — though note that path
  is unexercised by anything measured here, so its benefit is a code-level claim, not a measurement.
- **Pure data parallelism:** prefer `auto`, or set `lm_head_weight_grad_in_kernel_order: true` if you
  are committed to explicit — that closes the whole +9.3% gap (§8 item 1). +10% is not noise.
- **Any untied model on explicit:** `lm_head_weight_grad_in_kernel_order: true` is checkpoint-safe
  and inert under `auto`, so the only reason not to set it is that you have measured it neutral for
  your geometry and prefer fewer knobs.
- **Pure tensor parallelism:** explicit is mildly better (−0.2%), for the reason in §5.2. The
  advantage may not survive at a real embedding width.
- **Small models / debugging runs:** explicit costs a few percent; usually irrelevant, but do not
  benchmark a 4-layer proxy and extrapolate.

### For MaxText, ranked by value

01. **Emit the untied LM head's weight gradient in kernel order — `lm_head_weight_grad_in_kernel_order`
    (§4.1, §4.2, §4.3, §4.5).**

    > Two earlier revisions of this item have been withdrawn. The first recommended "stop pinning the
    > logits output under explicit": **measured false**, gating the `out_sharding` produces a 0-line
    > HLO diff, because the barriers are JAX's and not MaxText's (§4.5). The second recommended
    > `lm_head_kernel_transposed`, storing the kernel as `[vocab, embed]`: that one works, but it is
    > dominated on every axis by what follows, so it has been removed rather than shipped beside it.

    Since the barrier cannot be removed, remove the *need* for the fold. Autodiff's default transpose
    rule builds the weight gradient as `transpose(dot(g, inputs))` and relies on `algsimp` folding the
    transpose back into the dot; under explicit that fold never happens. A hand-written `custom_vjp`
    contracts `dk` straight into the kernel's own axis order instead, so there is no transpose to
    fold. `_dot_general_kernel_ordered_grad` in `linears.py` implements it, `DenseGeneral`'s
    `weight_grad_in_kernel_order` gates it, and `lm_head_weight_grad_in_kernel_order: false` in
    `base.yml` wires it to the untied head. It self-disables under `auto` (where the fold does happen)
    and under quantization (where it does not own the dot).

    **Why this and not the transposed kernel.** They reach the same layout, but only the `custom_vjp`
    recovers the *collective*. On llama2/shrink/fsdp4 the kernel-order dump gains
    `%all-reduce-scatter.2`, byte-for-byte identical to auto's `%all-reduce-scatter.5`
    (`bf16[512,32000] → bf16[128,32000]`, layout `{1,0:T(8,128)(2,1)}`, dim-0 dynamic-slice); the
    transposed kernel does not. It matches the transposed kernel's copy reduction as well
    (35 → 29 data-movement ops). And it leaves the stored array untouched, so there is no checkpoint
    conversion and no change to initialization.

    **Measured**, v5p, 4 chips, xprof `train_step_ns` median on `/device:TPU:0`, 56 runs, all four
    arms back-to-back per config so within-row deltas carry no session drift
    (`A` auto, `B` explicit, `C` explicit + transposed kernel, `D` explicit + kernel-order gradient):

    | config   |     A µs |     B µs |     C µs |     D µs |   B−A |   C−A |   D−A |   D−C | rep spread |
    | -------- | -------: | -------: | -------: | -------: | ----: | ----: | ----: | ----: | ---------: |
    | deepseek |   9081.0 |   9883.5 |   9603.2 |   9216.8 | +8.84 | +5.75 | +1.50 | −4.02 |      0.56% |
    | llama2   |   4027.4 |   4223.8 |   4136.2 |   3995.3 | +4.88 | +2.70 | −0.80 | −3.41 |      0.79% |
    | mistral  |   3905.2 |   4084.6 |   4014.4 |   3853.0 | +4.59 | +2.80 | −1.34 | −4.02 |      0.97% |
    | mixtral  |  10104.5 |  10407.2 |  10320.4 |  10129.8 | +3.00 | +2.14 | +0.25 | −1.85 |      0.12% |
    | qwen3    |   8261.8 |   8608.6 |   8188.9 |   8214.8 | +4.20 | −0.88 | −0.57 | +0.32 |      0.51% |
    | *w2048*  |  14972.1 |  15215.1 |  14872.8 |  14886.3 | +1.62 | −0.66 | −0.57 | +0.09 |      0.04% |
    | *prod*   | 196039.4 | 196558.5 | 196145.3 | 196101.9 | +0.26 | +0.05 | +0.03 | −0.02 |      0.03% |

    The first five rows are the shrunk `emb 512` proxies; the two italic rows are `emb 2048`
    (`w2048` = 4 layers / seq 1k, `prod` = 16 layers / seq 4k / 16 query heads).

    **Read the two halves of that table differently — this is the part that is easy to get wrong.**

    - The five emb-512 rows go from a **+4.59% median penalty to −0.57%**, D beating auto outright on
      three of five. That is real, but it is *the artifact regime*: emb 512 / fsdp 4 puts 256 bytes
      per shard on the minor-most dimension, below XLA's 512-byte AR→RS gate (§4.2), so mechanism A
      is active only there. No production job has that geometry. **A fix validated only on these rows
      is validated on a benchmark artifact.**
    - The two above-gate rows are the actual ship criterion, and the bar there is *no regression*, not
      a win: **`prod` +0.03%** against a 0.03% rep spread, **`w2048` −0.57%**. There was almost
      nothing left to recover, and the flag does not cost anything to have recovered it. That is what
      licenses turning it on; the emb-512 column is what makes it worth having the switch at all.
    - **D vs C across all seven: median −1.85%, worst +0.32%.** D wins by 1.85–4.02% on the four
      configs where C left the most on the table, and ties C within noise on the two where C already
      won. There is no configuration measured here where C is the better choice.

    **Numerical equivalence** — 20 training steps against the `auto` baseline, same seed, max absolute
    loss deviation (relative in parentheses):

    | config  |       B (explicit) |         C (transposed) |   D (kernel-order) |
    | ------- | -----------------: | ---------------------: | -----------------: |
    | gemma2  | 6.41e-04 (4.9e-05) |                      — |                  — |
    | llama2  | 2.62e-04 (2.4e-05) | **9.42e-03 (8.7e-04)** | 2.52e-04 (2.3e-05) |
    | mixtral | 1.41e-04 (1.3e-05) | **7.93e-03 (7.3e-04)** | 9.63e-05 (8.9e-06) |

    D sits at explicit's own noise level — on mixtral slightly *below* it — because reordering a
    contraction only reassociates a floating-point sum. C is ~35× worse for a reason that is not
    rounding at all: at the flipped kernel shape `jax.random` draws a different matrix from the same
    seed, so C trains a differently-initialized model. That is survivable for a fresh run and fatal
    for resuming one.

    **Acceptance tests**, in order of how much they tell you:

    1. `%all-reduce-scatter` with post-slice shape `bf16[emb/n_fsdp, vocab]` present in the explicit
       `after_codegen` dump — the mechanism, and the one thing only this flag delivers.
    2. Zero `copy | f32[*,vocab]` instructions in the explicit `after_optimizations` dump.
    3. Step time at **production geometry** within noise of `auto`. Do not gate on an emb-512 proxy in
       either direction.

    Note that raw `reduce-scatter` *counts* are a poor acceptance test on their own — the causal
    control in §4.2 values that mechanism at ≤28% of the penalty.

02. **Do not bother eliding redundant `reshard` barriers — tested, worth exactly zero.** The obvious
    companion fix to item 1 is to stop emitting a barrier when the reshard is a no-op (the value's
    aval already carries the requested sharding), on the theory that fewer barriers means more folds.
    It was implemented and measured, and it does not work:

    | config        | barriers off → on | optimized ops | differing ops |
    | ------------- | ----------------: | ------------: | ------------- |
    | llama2 shrink |       1254 → 1201 |   2889 → 2889 | **none**      |
    | prod          |       1258 → 1205 |   2942 → 2942 | **none**      |
    | deepseek      |       3537 → 3402 |   9642 → 9642 | **none**      |

    The predicate is correct — verified on CPU, where it collapses N redundant reshards to the
    irreducible 2 — but it only reaches ~4% of the barriers in a real module, and the optimized
    executable is **op-identical** in all three cases. XLA already discards no-op `Sharding`
    custom-calls; the ones that matter are the ones on real reshards, which by definition cannot be
    elided. Recorded here so the next person does not spend the week: **the barrier count is not the
    lever, the gradient orientation is.**

03. **Add `shard_mode` to the golden suites (§7.3).** Until CI compares modes, every regression here
    can silently return. `tests/integration/hlo_diff_test.py` is the cheapest place to start; assert
    on `after_codegen` reduce-scatter counts, not on static collective byte totals.

04. **Any HLO byte-counter used to compare modes must descend into
    `kind=kCustom, calls=%all-reduce-scatter.*` fusions** and report the post-slice shape. Counting
    the `all-reduce` inside the fusion at its pre-slice shape reports auto's wins as if they were
    losses, which is how the "identical bytes" claim survived a full sweep here.

05. **Re-examine the `check_vma` ban (§7.1).** JAX 0.11.1 accepts `check_vma=True` on an all-Explicit
    mesh; the restriction appears to be MaxText-imposed. Lifting it would give the MoE models on the
    explicit whitelist back a documented optimization.

06. **Validate `fused_qkv` + explicit and context-parallel + explicit** in `types.py` so users get a
    MaxText message instead of a raw JAX trace error.

07. **Fix `gcs_utils.upload_dump()`** to no-op (or write locally) for non-`gs://` paths — today it
    crashes after training and destroys the profile.

08. **Make explicit mode not defeat remat CSE on unrolled blocks (§4.7).** With `remat_policy: full`
    and no scan, auto CSEs the recomputed forward against the original (61 remat-tagged instructions
    of 11,209); explicit cannot (1,013 of 12,452), costing 279.5 µs. This bites any config where a
    block is unrolled — gemma3 below 6 layers today, but also `scan_layers: false` runs generally.

09. **Deduplicate the tied-embedding bf16 cast (§4.6).** `embeddings.py:235-240`'s `.T` makes explicit
    materialize the 128 MB embedding shard in bf16 **twice** (+80–82 µs on every gemma). Hoisting the
    cast above the transpose, or reusing one converted value for both consumers, is a pure win and is
    independent of everything else in this document.

10. **Investigate the tuple-repack all-reduce tax (§6.2)** — the one measured explicit cost that grows
    with depth (+112 → +219 → +379 µs from 4 to 12 layers on gemma3).

11. **Delete or rewire `validate_shard_mode`** (§7.4) — as dead code it actively misinforms.

______________________________________________________________________

## 9. Reproducibility

The measurement harness was a set of throwaway scripts and is not checked in; it is short enough to
restate. Every run was:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name=<model> shard_mode=<auto|explicit> \
  steps=12 per_device_batch_size=1 max_target_length=1024 \
  dataset_type=synthetic enable_checkpointing=False \
  skip_first_n_steps_for_profiler=3 profiler_steps=3 profiler=xplane profile_cleanly=True \
  jax_cache_dir= \
  override_model_config=True base_emb_dim=512 base_mlp_dim=1024 base_num_decoder_layers=4 \
  base_num_query_heads=8 base_num_kv_heads=8 head_dim=128
```

with `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_module_re=.*train_step.*"` for the HLO. Two notes
that cost time to discover:

- Drive the dump through `XLA_FLAGS`, **not** MaxText's `dump_hlo` config — that hook unconditionally
  uploads to GCS and raises on a local `base_output_directory`, aborting the run before the xplane
  profile is flushed (§8 item 6).
- Clear `jax_cache_dir`. A persistent-cache hit skips compilation and therefore suppresses the HLO
  dump, while producing an identical executable and step time.

Timings are `train_step` durations read off `/device:TPU:0` in the `.xplane.pb`, median of the 3
profiled steps; HLO facts are from `*after_optimizations.txt` and `*after_codegen*`. The TPU is a
serial resource, so runs were serialized with `flock`. 12 steps, profile steps 4-6, synthetic data,
checkpointing off, throughout.

### Limitations, stated plainly

- **The barrier→simplifier causal link is inferred, not proven.** No XLA rebuild with the
  `Sharding` custom-call suppressed was performed. See §4.1 for the alternative layout-assignment
  reading; both imply the same MaxText-side fix, but the XLA-side attribution should be confirmed
  before filing an XLA bug.
- **4 chips, single host.** No cross-host ICI/DCN behaviour is exercised. The collective costs here
  are intra-slice. Conclusions about *relayout* cost should hold; conclusions about *collective
  latency* may not scale to multi-slice. In particular the reduce-scatter loss (§4.2) is
  combiner-threshold-dependent and its 2× wire-byte penalty is a 4-way-ring figure.
- **Shrunk models, and one of them shrank into a different program.** Real parameter counts do not
  fit a v5p-8. §5.1's magnitudes are proxies, not production numbers; §5.3 exists to bound the
  extrapolation and its answer is "partly, and non-monotonically". gemma3 (§4.7) is the cautionary
  case: its shrink disabled the scan, and its headline number reverses sign at real depth. The other
  nine models compiled to the same `while` structure in both modes, which is the check that caught it.
- **The unembedding mechanism is the well-evidenced one.** The in-scan variable-sign residual
  (§5.3 item 3), mixtral's 28% unattributed remainder, gemma3's logits-matmul divergence (§4.6,
  ±100 µs with no mechanistic explanation), and the context-parallelism MLP spec divergence (§7.2)
  are identified but not fully attributed.
- **The gemma3 layer ladder is 1 run per point**, not 3 — enough to establish the sign reversal
  (+484 µs → −106 µs dwarfs the noise floor) but not to quantify L6/L12 precisely.
- **3 profiled steps per run** for most configs (4 runs per mode only for the d16 point). Adequate
  given the 0.07% noise floor, but not a large sample — and the module-span lead gap (§3 caveat 3)
  puts a floor of a few tenths of a percent on any single-pair comparison.
