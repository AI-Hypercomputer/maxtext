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
Revised 2026-09-03: the whole matrix re-measured at above-gate geometry (§5.4), the LM-head flag
turned **on by default** where it can act, a second flag added for the dense-layer gradients (§4.8),
and a mode-independent 1.4% win found and shipped along the way (§5.5).
Revised again 2026-09-03 after a depth sweep: **every "explicit wins" row in this document sits at a
scanned-stack depth where an unrelated XLA layout pathology inflates both modes by ~25% (§4.9)**.
With those depths excluded explicit needs *both* kernel-order flags to reach parity, so
`dense_weight_grad_in_kernel_order` now also defaults on under explicit (§4.8), and both flags are
now implemented with `jax.sharding.auto_axes` rather than a hand-written `custom_vjp` (§8 item 01).
The eight-model matrix was then re-run on that build at healthy depths: **§5.6 supersedes §5.4 and
is the table to quote.**

`shard_mode` (`src/maxtext/configs/base.yml:554`) selects how MaxText expresses sharding:

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

    > *Revised (2026-09-03).* "Off by default" no longer holds. Re-measured across all eight models
    > at above-gate geometry the flag has **never lost to `auto`** — on any model, at any depth
    > (L12/L14/L16), at any vocab (4096 or 32000) — while explicit without it is +3.25% on qwen3.
    > It is therefore now tri-state and resolves to **on** wherever it can act: `shard_mode: explicit` and an untied head. Writing it out is only needed to reproduce a measurement.

    > *Revised again (2026-09-03).* The mechanism is unchanged; the implementation is not. The
    > hand-written `custom_vjp` has been replaced by a `jax.sharding.auto_axes` region around the
    > forward dot, which drops the barrier for the length of that one dot and lets Shardy
    > re-propagate the operands' shardings exactly as `shard_mode: auto` would — so XLA performs the
    > fold itself instead of MaxText hand-writing around it. Same step times to within 0.15 pp on all
    > five models measured both ways, 114 fewer lines, gradients now **bit-identical** to the default
    > rule rather than agreeing to rounding, and it works at all 17 call sites including MLA
    > (§8 item 01).

11. **With that default, explicit is faster than `auto` on five of eight onboarded models and inside
    the run-to-run spread on a sixth.** The one number a user needs is §5.4: at emb 2048 / mlp 8192 /
    16 layers, with nothing in the config but `shard_mode`, the median across the eight models is
    **−0.330%** and the worst case is gemma3 at **+0.803%** — down from a +3.25% worst case before
    this revision. Both remaining non-wins (gemma3, deepseek) are fixed by a second flag,
    `dense_weight_grad_in_kernel_order`, which ships default-off because its sign is model-dependent
    (§4.8): it is worth 0.90 pp on gemma3 and 0.16 pp on deepseek, and costs 0.06–0.60% on the other
    six. With those two per-model settings applied, no model regresses by more than 0.041%.

    > *Superseded (2026-09-03).* Both halves of this item are contaminated by §4.9. **Sixteen layers
    > is one of the depths where the layout pathology fires**, and that pathology is worth ~25% of
    > step time — enough that scheduling noise inside it swamps a 0.4% sharding effect and, on five
    > of eight models, flips its sign. Re-measuring llama2 at L16 with the pathology removed
    > (`param_scan_axis: 0`) turns the "explicit wins by 0.494%" of §5.4 into **explicit loses by
    > 1.482%**, and the dense flag takes it to **+0.006%**. Across 15 (model, depth) pairs at
    > *healthy* depths the LM-head flag alone leaves a median **+0.463%** penalty (worst +1.038% on
    > gemma2) while both flags together give a median **+0.005%** and a full range of
    > **−0.183% … +0.134%**. So the second flag is not a two-model carve-out at all — it is the other
    > half of the fix, it now defaults on under explicit too, and the honest summary of both is
    > *parity with `auto`, reliably*, rather than a win. The eight-model matrix re-run on that build
    > at healthy depths is **§5.6**: median **+0.000%**, worst **+0.133%**, and the only two rows
    > outside their own rep spread are `gemma-2b` (−0.685%) and `gemma3-4b` (−0.374%), both wins.

12. **The largest single win found here is not a sharding fix at all.** `YarnRotaryEmbedding` built
    its whole `[max_position_embeddings, head_dim/2]` frequency table inside every layer of every
    step in order to read `max_target_length` rows out of it — `f32[163840, 32]` on deepseek, 0.6%
    of it used. Computing the needed rows directly from `position` is **−1.4% of step time, equally
    in both modes** (§5.5). It is called out here because it was invisible to every mode-comparison
    in this document until a detector was pointed at ops whose output is ≥ 8× their input: A/B
    diffing two modes cannot find work that both modes do.

    > *Superseded (2026-09-02).* This item previously recommended `lm_head_kernel_transposed`, which
    > stores the kernel as `[vocab, embed]`. That flag reaches the same layout but is slower on four
    > of seven configs, has no checkpoint-conversion path, and — because `jax.random` draws a
    > different matrix from the same seed at the flipped shape — perturbs the loss ~35× more than
    > explicit's own noise. It has been removed rather than shipped alongside a strictly better flag.

13. **The biggest number found in this whole investigation is not a sharding number either: a
    scanned stack whose length is 8, 16, 24 (or 2 or 4) costs ~25% of step time, in both modes.**
    `param_scan_axis: 1` stacks each parameter as `[in, L, out]`. At those lengths XLA assigns the
    gradient stack layout `{2,0,1}` instead of `{2,1,0}`, so the per-iteration
    `dynamic-update-slice` writes a degenerate `T(1,128)`-tiled slice: **8.16 ms of a 36.5 ms step**
    on llama2 at L8. Walking depth 2→24 the ns/layer trend is flat at ~3.3 M except at exactly those
    lengths, where it jumps 25–34%. `param_scan_axis: 0` removes it (**L8 −23.6%, L16 −24.9%**) and
    costs 0.18–0.28% at healthy depths. It survives at production geometry (+14.1% at L8, +11.5% at
    L16). See §4.9 — and note that **every benchmark in this document before §4.9 used a stack length
    of 4, 8 or 16**, so it is the most consequential methodology finding here.

**Practical guidance:** as of this revision, `shard_mode: explicit` with nothing else written out is
**at parity** with `auto` on all eight onboarded models at realistic geometry and healthy depth —
median **+0.000%**, worst case **+0.133%** against a 0.16% rep spread, and two models (`gemma-2b`
−0.685%, `gemma3-4b` −0.374%) where explicit wins outright (§5.6; §4.8 gives the same picture over
15 (model, depth) pairs, median +0.005%, worst +0.134%). Choose on capability (§6, §7) and treat
performance as a non-issue rather than a tax. Both kernel-order flags now default on where they can
act; nothing needs to be written out. Two things to still verify for yourself: **pure data
parallelism** was +10% before the LM-head fix and has not been re-measured, and if your
`base_num_decoder_layers` is a multiple of 8 you are paying §4.9's ~25% in both modes and should try
`param_scan_axis: 0`.

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

> *Correction (2026-09-03).* The all-reduce repack tax does not survive the move to above-gate
> geometry. Re-measured at emb 2048 / mlp 8192 / L18, gemma3's `all-reduce` opcode delta is
> **+41.4 µs of 2.52 s of device-op time** across the whole profile (84 ops in both modes) — three
> orders of magnitude below the emb-512 figure as a share of step. It was a small-tensor effect:
> at emb 512 the repacked tuple is latency-bound, and merging it removes the only work that was
> available to overlap it. Read the ladder above as a property of the shrink, not of gemma3.
> What *does* survive at real width is §4.8.

### 4.8 Consequence C — the dense-layer weight gradient's layout is un-pinned, and that cuts both ways

§4.2 and §4.3 are about *one* dot, the untied LM head, and they are about a transpose that XLA
declines to fold. There is a second, broader effect on every other `DenseGeneral` in the model, and
it is not a missed optimization at all — it is a **layout choice that explicit hands to XLA and
`auto` does not**.

Under `auto` the weight gradient's layout is derived from the stored parameter's, because GSPMD
propagates the constraint through. Under explicit the `Sharding` custom-calls sit between the two,
layout assignment cannot propagate across them, and XLA picks whatever layout the gradient `dot`
itself prefers — `{1,2,0}` for an `f32[1, embed_shard, mlp]` MLP kernel gradient, i.e. **embed
minor-most**, the transposed orientation. Whether that costs or pays depends entirely on what
consumes the gradient, and the sign is not the same for all models.

The evidence is a single opcode. Device-op totals for `copy`, d16 geometry (emb 2048, mlp 8192),
3 profiled steps summed over all eight core planes, one rep:

| model (d16)               |               `auto` |               `explicit` | `explicit` + dense flag |
| ------------------------- | -------------------: | -----------------------: | ----------------------: |
| llama2 (scan stack 16)    | 1128 ops / 27,694 µs | 1896 ops / **26,234 µs** |    1128 ops / 27,682 µs |
| mistral (16)              |        1116 / 27,850 |        1884 / **26,298** |           1116 / 27,849 |
| qwen3 (16)                |        1116 / 31,180 |        1884 / **29,556** |           1116 / 31,771 |
| gemma3-L18 (scan stack 3) |      792 / **2,425** |            2304 / 10,110 |             792 / 2,391 |

Two things to read off that table.

**The flag is a become-`auto` switch for this opcode, exactly.** Turning
`dense_weight_grad_in_kernel_order` on reproduces `auto`'s copy count *op for op* on all four models
— 1128 vs 1128 on llama2, 792 vs 792 on gemma3 — and its cost to within 1.4% (llama2 −0.04%,
mistral −0.01%, gemma3 −1.4%, qwen3 +1.9%). It is not "removing a redundant copy"; it is declining
XLA's alternative layout.

**On the deep-scan models XLA's alternative layout is the better one.** llama2/mistral/qwen3 relayout
the MLP kernel gradient in *both* modes — the target is the optimizer's `f32[1,512,8192]{2,0,1:T(1,128)}`
— and the only difference is the source:

```
auto      f32[1,512,8192]{2,0,1:T(1,128)} copy(f32[1,512,8192]{2,1,0:T(8,128)} %get-tuple-element)
                                                            384 ops, 10,154.5 us   (26.4 us each)
explicit  f32[1,512,8192]{2,0,1:T(1,128)} copy(f32[1,512,8192]{1,2,0:T(8,128)} %convert_bitcast_fusion)
                                                            384 ops,  7,769.8 us   (20.2 us each)
```

Same count, same destination, **24% cheaper per copy** from the transposed source. Explicit nets
−1,460 µs on llama2's `copy` opcode even after paying for two extra attention-kernel families
(+739 µs on `f32[1,512,8,128]`, +174 µs on `f32[1,8,128,512]`). Forcing kernel order gives that back.

**On the short-scan models `auto` does no relayout at all, so explicit's choice is pure cost.**
gemma3-L18 (`scan_length = 18 // 6 = 3`) and deepseek's length-1 dense-layer scan have no
`f32[1,512,8192]`-class copy under `auto` — its entire `copy` budget is 2,425 µs of unrelated
attention-mask and bf16-cast traffic. Explicit adds **1,512 copies costing +7,685 µs**, in four
families, all of them the same "gradient came out transposed" signature:

```
+4,434.6 us  432 ops  f32[1,512,8192]{2,1,0}  copy(f32[1,512,8192]{1,2,0} %convert_bitcast_fusion)   MLP wi
+2,254.7 us  216 ops  f32[1,8192,512]{2,1,0}  copy(f32[1,8192,512]{1,2,0} %get-tuple-element)        MLP wo
+  809.0 us  648 ops  f32[1,512,8,128]{3,2,0,1} copy(f32[1,512,8,128]{1,3,2,0} ...)                  q/k/v
+  215.4 us  216 ops  f32[1,8,128,512]{3,2,0,1} copy(f32[1,8,128,512]{2,3,1,0} ...)                  out proj
```

Those four families are 60% of gemma3-L18's whole explicit penalty once the aggregate `while` row
(which double-counts its own body) is excluded: +7,685 of +12,832 µs. The flag deletes all four and
the `while` row flips from +8,897 µs to −2,705 µs.

**So the discriminator is the scanned stack depth, and it is a proxy, not a mechanism.** The six
models where the flag loses all have a 16-deep gradient stack; the two where it wins have stacks of
3 and 1. The plausible reading is that a 16-deep stack forces a relayout anyway (the optimizer wants
`T(1,128)` on a tensor that the scan writes with `T(8,128)`), so the gradient's own layout is free
to be chosen for the dot; at depth 3 or 1 no relayout is needed and any choice other than the
parameter's is wasted. But that is inference from layouts, not a proven causal chain, and two points
below the threshold against six above is not enough to fit one. **That is why the flag ships
default-off and the LM-head flag does not** — the LM-head transpose is a missed fold with one sign,
this is a layout trade with two.

> *Superseded (2026-09-03) — and the paragraph above contains its own refutation.* "A 16-deep stack
> forces a relayout anyway, because the optimizer wants `T(1,128)` on a tensor the scan writes with
> `T(8,128)`" is not a property of 16-deep stacks. It is §4.9: the degenerate layout XLA picks for a
> stack whose length is a multiple of 8, in **both** sharding modes, at a cost of ~25% of step time.
> Every "the flag loses here" data point sat inside it. Read on.

#### The resolution: measure at a depth where nothing else is happening

Re-running the same comparison at depths that are *not* multiples of 8 — where the `dynamic-update-slice`
is a well-tiled `{2,1,0}` write and the step is 25% cheaper — reverses the conclusion. Fifteen
(model, depth) pairs, d16 geometry, 2 reps each, medians, both columns against that config's own
`auto`:

| model      |   L | `explicit` (LM-head flag only) | `explicit` + dense flag |
| ---------- | --: | -----------------------------: | ----------------------: |
| llama2     |  12 |                        +0.622% |                 −0.016% |
| llama2     |  14 |                        +0.391% |                 +0.002% |
| llama2     |  18 |                        +0.429% |                 +0.005% |
| llama2     |  20 |                        +0.463% |                 +0.005% |
| mistral    |  12 |                        +0.598% |                 +0.051% |
| mistral    |  14 |                        +0.919% |                 −0.007% |
| qwen3      |  12 |                        +0.354% |                 +0.044% |
| qwen3      |  14 |                        +0.397% |                 +0.052% |
| qwen3      |  18 |                        +0.534% |                 +0.008% |
| gemma2     |  12 |                        +0.950% |                 −0.095% |
| gemma2     |  14 |                        +1.038% |                 −0.183% |
| gemma3     |  18 |                        +0.803% |                 −0.101% |
| mixtral    |  12 |                        −0.115% |                 +0.134% |
| mixtral    |  14 |                        −0.053% |                 +0.052% |
| deepseek   |  12 |                        −0.348% |                 +0.061% |
| **median** |     |                    **+0.463%** |             **+0.005%** |

**Range: `explicit` alone −0.348% … +1.038%; with the dense flag −0.183% … +0.134%.** The flag is
not a two-model carve-out and it is not a coin flip — it is a *variance eliminator*. It costs at
most 0.19 pp anywhere and it removes a penalty that is otherwise present on 12 of 15 configs. The
one model where it gives something up, mixtral, gives up 0.06–0.25 pp; the six models where the
original table said it "costs 0.06–0.60%" were reading scheduling noise inside the §4.9 pathology.

The two flags are also **independent and additive**, which is the check that they are two instances
of one mechanism rather than one effect measured twice. At L18, all four combinations, against the
same `auto`:

| config                       |  llama2 L18 |   qwen3 L18 |
| ---------------------------- | ----------: | ----------: |
| `explicit`, neither flag     |     +1.098% |     +5.579% |
| dense flag only              |     +0.621% |     +4.853% |
| LM-head flag only            |     +0.429% |     +0.534% |
| **both flags** (the default) | **+0.005%** | **+0.008%** |

qwen3 is the clean case: its untied 151,936-entry head dominates, so the LM-head flag is worth 5 pp
and the dense flag 0.7 pp; on llama2 the two are the same size. Neither subsumes the other, and the
sum of the two individual gains predicts the joint one to within 0.03 pp on llama2 and 0.2 pp on
qwen3. **Both flags therefore now resolve to on under `shard_mode: explicit`.**

One disclosure that does not fit the pattern and is not being papered over: **on mixtral the
LM-head flag forfeits a reproducible ~1.4%.** Explicit with *neither* flag reads −1.436% / −1.448% /
−1.522% at L12 / L14 / L16 against `auto`, and turning the LM-head flag on lands it back on `auto`'s
schedule (−0.115% / −0.053% / −0.012%). That is an MoE-scan scheduling windfall, not an LM-head
effect — shrinking mixtral's vocab 8× leaves the spread unchanged (§8 item 01) — and deepseek, the
other MoE model, has the *opposite* sign (the LM-head flag takes it from +0.947% to +0.176% at L8).
One model in one direction is not a rule to encode in a default, so the default stands and the cost
is stated here. A mixtral user who has measured this may set `lm_head_weight_grad_in_kernel_order: false`.

______________________________________________________________________

### 4.9 Not a sharding effect at all: `param_scan_axis: 1` degenerates when the stack length is a multiple of 8

This section exists because it invalidates measurements elsewhere in this document, not because it
has anything to do with `shard_mode`. The effect is **identical in both modes** — it was found by
sweeping `auto` alone.

Walk llama2 down the depth ladder at d16 geometry (emb 2048, mlp 8192, seq 1024, pdbs 1, fsdp 4),
2 reps per point, and step time is not proportional to depth:

|      L |   `auto` ns |      ns/layer |  `explicit` | `explicit` + dense |
| -----: | ----------: | ------------: | ----------: | -----------------: |
|  **2** |  11,007,529 | **5,503,764** | **−0.902%** |            −0.050% |
|      3 |  12,494,770 |     4,164,923 |     +0.027% |            +0.002% |
|  **4** |  19,376,215 | **4,844,054** | **−0.517%** |            +0.083% |
|      5 |  18,731,819 |     3,746,364 |     +0.465% |            +0.074% |
|      6 |  21,837,597 |     3,639,600 |     +0.343% |            +0.037% |
|      7 |  24,953,852 |     3,564,836 |     +0.330% |            +0.018% |
|  **8** |  36,536,639 | **4,567,080** | **−0.444%** |            −0.034% |
|      9 |  31,091,571 |     3,454,619 |     +0.342% |            +0.004% |
|     10 |  34,169,574 |     3,416,957 |     +0.409% |            +0.004% |
|     12 |  40,345,816 |     3,362,151 |     +0.622% |            −0.016% |
|     14 |  46,583,332 |     3,327,381 |     +0.391% |            +0.002% |
| **16** |  70,094,537 | **4,380,909** | **−0.494%** |                  — |
|     18 |  58,852,896 |     3,269,605 |     +0.429% |            +0.005% |
|     20 |  65,004,700 |     3,250,235 |     +0.463% |            +0.005% |
| **24** | 104,650,808 | **4,360,450** | **−0.542%** |            −0.024% |

Two patterns, and they are the same pattern. The ns/layer column decays smoothly from 4.16 M at L3
to 3.25 M at L20 — **except** at L2, L4, L8, L16 and L24, where it is 25–34% above the trend. And
those five depths are **exactly** the five where `explicit` "beats" `auto`. Everywhere else explicit
is +0.33% to +0.62%, monotonically, with a rep spread under 0.1%.

**The HLO says what it is.** Diffing the L8 and L9 `auto` dumps (`opdiff.py`), the scanned gradient
stack changes layout:

```
L9   f32[9,512,8192]{2,1,0:T(8,128)}      <- normal, row-major, 8x128 tiles
L8   f32[8,512,8192]{2,0,1:T(8,128)}      <- middle dimension made minor
```

and with it every per-iteration write into that stack:

```
L8   dynamic-update-slice(f32[8,512,8192]{2,0,1}, f32[1,512,8192]{2,0,1:T(1,128)S(1)} %copy, ...)
                            three families, 34,480 + 31,730 + 31,676 us = 97,886 us / profile
                            = 8.16 ms of a 36.5 ms step = 22% of the step
```

`T(1,128)` is a degenerate tile — one row per tile instead of eight — so the write moves 8× the
minimum number of tiles. The same three families are present, at the same cost, in the `explicit`
and `explicit`+flags dumps: this is XLA's layout assignment reacting to the stack's leading
dimension, and sharding mode has no input to it.

**`param_scan_axis: 0` removes it.** That config stacks parameters as `[L, in, out]` instead of
`[in, L, out]`:

| llama2, d16 | `param_scan_axis: 1` | `param_scan_axis: 0` |            Δ |
| ----------- | -------------------: | -------------------: | -----------: |
| L8          |           36,535,103 |           27,917,743 | **−23.587%** |
| L9          |           31,091,571 |           31,003,302 |      −0.284% |
| L16         |           70,096,329 |           52,622,643 | **−24.928%** |
| L18         |           58,852,896 |           58,747,091 |      −0.180% |

So it is worth ~24% at the bad depths and ~0.2% at the good ones — the two are the same measurement,
because `param_scan_axis: 0`'s L8 and L16 sit on the healthy ns/layer trend that `param_scan_axis: 1`
only reaches at non-multiples of 8. It is **not** an artifact of the shrunk geometry either. At
`prod` (16 query heads, seq 4096) the same jump is there:

| llama2, prod |     step ns |  ns/layer | vs the next depth up |
| ------------ | ----------: | --------: | -------------------: |
| L8           |  79,070,204 | 9,883,776 |          **+14.07%** |
| L9           |  77,978,783 | 8,664,309 |                    — |
| L16          | 151,238,509 | 9,452,407 |          **+11.45%** |
| L18          | 152,662,296 | 8,481,239 |                    — |

**What this costs the rest of this document.** Every headline comparison before this section was run
at L4 (§5.1, §5.2), L8 (deepseek) or L16 (§5.4, §4.8) — all inside the pathology. The clearest
correction is §5.4's llama2 row. Re-run at L16 with the pathology removed:

| llama2, L16, `param_scan_axis: 0` |    step ns |   vs `auto` |
| --------------------------------- | ---------: | ----------: |
| `auto`                            | 52,622,643 |           — |
| `explicit` (LM-head flag only)    | 53,402,334 | **+1.482%** |
| `explicit` + dense flag           | 52,625,548 | **+0.006%** |

§5.4 reports the same model at the same depth as **−0.494% for explicit**. The sign is opposite. A
0.4% sharding effect measured on top of a 25% layout artifact is measuring the artifact's scheduling
noise, and this is why the flag-default conclusion changed (§4.8).

This has not been chased further. Whether `param_scan_axis: 0` should become the MaxText default is
a question about checkpoint layout, optimizer-state layout and every model config, not a question
about sharding, and it wants its own investigation (§8 item 03).

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

> **This table is superseded by §5.6** and is kept only because §4 refers to its dumps. Every row is
> below XLA's minor-most AR→RS gate (§4.2) and none has the LM-head fix, so it measures the worst
> case of a mode that no longer exists. Quote §5.6 instead.

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

### 5.4 The table to quote: all eight models, above the gate, with nothing written out but `shard_mode`

Everything above this point measures a mode that no longer exists — emb 512 (below XLA's AR→RS
gate, §4.2/§5.3), and with `lm_head_weight_grad_in_kernel_order` off. This is the same sweep re-run
at **emb 2048 / mlp 8192 / 16 layers** (gemma3 at 18, a multiple of its 6-long attention pattern;
deepseek at 8), seq 1024, pdbs 1, fsdp 4, on one build, 54 runs, 3 reps per arm, medians. The
`explicit` column has **nothing in the config but `shard_mode: explicit`** — it is what a user gets.

| model                | tied |   `auto` ns | `explicit` ns |           Δ | rep spread A/B | + dense flag |
| -------------------- | :--: | ----------: | ------------: | ----------: | -------------: | -----------: |
| `mistral-7b`         |  no  |  69,990,530 |    69,604,164 | **−0.552%** |  0.01% / 0.04% |            — |
| `qwen3-8b`           |  no  |  83,941,679 |    83,490,576 | **−0.537%** |  0.08% / 0.13% |            — |
| `llama2-7b`          |  no  |  70,094,537 |    69,748,502 | **−0.494%** |  0.03% / 0.02% |            — |
| `gemma2-2b`          | yes  | 171,538,409 |   170,719,766 | **−0.477%** |  0.05% / 0.04% |            — |
| `gemma-2b`           | yes  |  88,434,470 |    88,272,771 | **−0.183%** |  0.01% / 0.01% |            — |
| `mixtral-8x7b`       |  no  | 353,006,821 |   353,151,437 |     +0.041% |  0.03% / 0.07% |            — |
| `deepseek3-16b` (L8) |  no  | 230,885,239 |   231,292,724 |     +0.176% |  0.02% / 0.02% |  **+0.021%** |
| `gemma3-4b` (L18)    | yes  |  88,235,432 |    88,944,184 |     +0.803% |  0.28% / 0.05% |  **−0.101%** |

**Median −0.330%. Explicit is faster on five of eight and inside the rep spread on a sixth.** The
two models where it is not — deepseek and gemma3 — are exactly the two that want
`dense_weight_grad_in_kernel_order` (§4.8), and with that flag set the worst regression across all
eight models is **mixtral's +0.041%, which is inside mixtral's own 0.07% rep spread**.

Compare against where this started. The same eight models under `explicit` before any of the fixes
in this document: qwen3 **+3.25%**, gemma3-L16 **+2.02%**, deepseek **+0.92%**. The change is
almost entirely the LM-head default (§8 item 01) plus, on deepseek, the YaRN fix (§5.5).

Four things to keep in mind before quoting this table:

1. **It is one geometry and one mesh.** Four chips, fsdp 4, seq 1024, pdbs 1. It is above the AR→RS
   gate, which is the point, but it is not a production model and not a production topology.
2. **Rows under ~0.2% are ties, not wins.** The rep spreads are in the table for exactly that
   reason; mixtral's +0.041% and deepseek's +0.021% are reported as parity.
3. **gemma3's depth must be a multiple of 6.** At L16 the same sweep reads **+2.020%** for
   `explicit` and +0.017% with the dense flag, because 16 // 6 = 2 leaves four layers unrolled
   alongside the scan (§4.7). L18 is the honest number.
4. **The dense flag is not a general recommendation.** On the six models not shown in its column it
   costs 0.06–0.60% (§4.8, §8 item 02).

> **Superseded (2026-09-03): read this table only for its `auto` column.** Sixteen layers is one of
> the depths where §4.9's layout pathology fires, and it is worth ~25% of step time in both modes.
> Six of these eight rows are at L16 (gemma3 is at L18 and deepseek at L8 — L8 is also inside the
> pathology, so only gemma3's row is clean). Re-running llama2 at L16 with `param_scan_axis: 0`
> gives `explicit` **+1.482%**, not the −0.494% above; the caveat 4 note about the dense flag costing
> 0.06–0.60% is the same artifact. The replacement table, at a healthy depth and with both flags
> defaulted on, is **§5.6**.

### 5.5 A mode-independent 1.4%: the YaRN frequency table was rebuilt inside every layer

Not a `shard_mode` finding — it was found while looking for one, it helps both modes equally, and it
is the largest single win in this document.

`YarnRotaryEmbedding.freqs_cis` was a `@property` that materialized the whole
`[max_position_embeddings, head_dim/2]` complex table and then indexed the `max_target_length` rows
the step actually reads. The table is *traced*, not a constant, so XLA cannot hoist or fold it: it is
rebuilt on every call site. On deepseek that is `f32[163840, 32]` — 5.24 M rows-worth of `cos`/`sin`
per build, of which 1024 rows (0.6%) are used — and the call site is inside the scanned layer.

Per 3-step profile on `deepseek3-16b`, d16 geometry, `auto` arm:

```
  (f32[163840,32], f32[163840,32]) fusion(f32[32] %multiply_add_fusion), kind=kLoop
                                              96 ops   35,842.7 us     <- building the table
  f32[1024,32] fusion(f32[163840,32] %gte, s32[1024] %broadcast_clamp_fusion), kind=kCustom
                                             168 ops      854.8 us     <- reading 0.6% of it
```

Both drop to **zero** ops post-fix. With the buffer traffic they pulled along the whole-profile
device-op total falls 6,386,779 → 6,250,836 µs (**−2.13%**), of which −63,315 µs is `copy-start` and
−33,664 µs is the `while` aggregate.

The fix (`embeddings.py`) is to compute the rows directly from `position` rather than build a table
to index — row `p` is `exp(1j · p · corrected_freqs)`, so `freqs_cis_at(position)` is one
`einsum("bs,h->bsh")` and a `cis`. The full table is kept as a member for reference and is what the
unit test compares against, bit for bit.

A second, smaller part of the same fix: writing `exp(1j·θ)` as `complex(cos θ, sin θ)` rather than
`jnp.exp(1j·θ)`. They are bit-identical, but the latter goes through XLA's overflow-safe complex
`exponential` expansion, which evaluates `exp(real(1j·θ)) == exp(0)` over the whole tensor — and
under explicit it evaluates it **twice**, because the `Sharding` custom-call on the broadcast `1j`
stops the simplifier commuting the constant to the right, so the reassociation that lets CSE merge
the two `exponential`s never fires. This is the one part of the finding that is mode-specific.

Measured (deepseek d16 L8, median of 3–8 reps, per arm, same build before and after):

| arm                       |         before |          after |       Δ |
| ------------------------- | -------------: | -------------: | ------: |
| `auto`                    | 234,120,034 ns | 230,891,079 ns | −1.379% |
| `explicit`                |    236,217,057 |    233,015,668 | −1.355% |
| `explicit` + LM-head flag |    234,508,220 |    231,270,774 | −1.381% |
| `explicit` + both flags   |    234,227,659 |    230,918,121 | −1.413% |

**The win is the same size in every arm**, which is the check that it is not a sharding effect. Its
one consequence for this document is on deepseek's *margin*: `explicit` + both flags moves from
+0.046% to **+0.012%** against `auto`, because the fix removes slightly more from explicit than from
auto.

Scope: this reaches every model with a `YarnRotaryEmbedding` and a `max_position_embeddings` much
larger than `max_target_length` — deepseek2/3 (163,840) most of all. `LlamaVisionRotaryEmbedding`
was deliberately left alone: it consumes the whole table by design, and `llama4` is not on the
explicit whitelist anyway.

### 5.6 The table to quote — eight models, shipped defaults, at a depth where §4.9 is not firing

This replaces §5.4. Same eight models, same geometry (emb 2048 / mlp 8192 / seq 1024 / pdbs 1 /
fsdp 4, four chips), same protocol (3 reps per arm, medians), re-run on the build this document
describes. Two things changed: every model now sits at a scanned-stack length that is **not** a
multiple of 8 (§4.9), and the `explicit` column has **nothing in the config but
`shard_mode: explicit`** — both `lm_head_weight_grad_in_kernel_order` and
`dense_weight_grad_in_kernel_order` resolve to on by themselves (§8 items 01–02). Sorted by margin.

| model           | layers | tied |   `auto` ns | `explicit` ns |           Δ | rep spread A/B |
| --------------- | -----: | :--: | ----------: | ------------: | ----------: | -------------: |
| `gemma-2b`      |     18 | yes  |  78,786,920 |    78,247,099 | **−0.685%** |  0.08% / 0.06% |
| `gemma3-4b`     |     18 | yes  |  88,335,850 |    88,005,221 | **−0.374%** |  0.21% / 0.31% |
| `mixtral-8x7b`  |     14 |  no  | 309,329,429 |   309,088,916 |     −0.078% |  0.08% / 0.07% |
| `llama2-7b`     |     18 |  no  |  58,858,748 |    58,857,796 |     −0.002% |  0.02% / 0.04% |
| `gemma2-2b`     |     18 | yes  | 150,702,912 |   150,706,999 |     +0.003% |  0.06% / 0.02% |
| `mistral-7b`    |     18 |  no  |  58,432,480 |    58,443,356 |     +0.019% |  0.02% / 0.01% |
| `qwen3-8b`      |     18 |  no  |  72,851,689 |    72,914,405 |     +0.086% |  0.18% / 0.24% |
| `deepseek2-16b` |     12 |  no  | 360,257,035 |   360,735,487 |     +0.133% |  0.16% / 0.05% |

**Median +0.000%. Worst case +0.133%, against a 0.16% rep spread on the same row.** Six of the eight
rows are inside their own rep spread in both directions — that is the definition of a tie. The two
that are not are `gemma-2b` and `gemma3-4b`, and `explicit` wins both.

The three models this revision set out to fix, before and after (before = §5.4, `explicit` with only
the LM-head flag defaulted on, at the depth §5.4 used):

| model           | §5.4 (LM-head flag only) | §5.6 (both flags, healthy depth) |
| --------------- | -----------------------: | -------------------------------: |
| `gemma3-4b`     |                  +0.803% |                      **−0.374%** |
| `mixtral-8x7b`  |                  +0.041% |                      **−0.078%** |
| `deepseek2-16b` |                  +0.176% |                      **+0.133%** |

gemma3 is the one where the flag does real work: 1.18 pp, and it changes the sign. mixtral and
deepseek were already inside noise in §5.4 and are still inside noise here; what the dense flag buys
on those two is not a win but the absence of a tail — across the 15 healthy (model, depth) pairs in
§4.8 the worst `explicit` row goes from +1.038% to +0.134% when it is on.

Four caveats before quoting this:

1. **mixtral is at 14 layers and deepseek at 12, not 18.** At 18 both run out of HBM at this
   geometry — deepseek asks for 43.56 G of HLO temporaries against 31.24 G available — so the sweep
   fails outright rather than producing a slow number. 14 and 12 are both healthy stack lengths
   under §4.9, which is the property that matters; §4.8 measures the same two models at the same two
   depths with all four flag combinations if you want the decomposition.
2. **gemma3's depth must be a multiple of 6** (§4.7): `scan_length = num_layers // 6`, and at any
   other depth the remainder is unrolled alongside the scan and dominates the comparison.
3. **It is one geometry and one mesh.** Four chips, fsdp 4, seq 1024, pdbs 1. Above the AR→RS gate
   (§4.2/§5.3), which is the point, but not a production model and not a production topology.
4. **Rows under ~0.15% are ties, not results.** The rep spreads are in the table for that reason.

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

### 6.3 The rotated reduce-scatter: explicit's largest and most stable win

The single biggest opcode-level difference in explicit's favour at d16 geometry is one that no
earlier revision of this document noticed, because the shrink was too small to produce it.

Under `auto`, the MLP weight-gradient reduce-scatter lands on a scatter dimension that is not
shard-aligned, and XLA implements the fixup as a **ring of `collective-permute`s** — 480 instances
of

```
(bf16[24,8192], bf16[24,8192], u32[], u32[]) collective-permute-start(bf16[24,8192] %slice),
    source_target_pairs={{0,1},{1,2},{2,3}}
```

per 3-step profile. Explicit emits **zero to fifteen** of them:

| model (d16) | `auto` count / span | `explicit` count / span |
| ----------- | ------------------: | ----------------------: |
| mistral     |     495 / 29,360 µs |             15 / 753 µs |
| qwen3       |     480 / 14,571 µs |                0 / 0 µs |
| llama2      |     495 / 13,397 µs |             15 / 122 µs |
| gemma3-L18  |     540 / 15,165 µs |          270 / 8,636 µs |

Read the spans with §3 caveat 1 in mind — `collective-permute-start` is an async span and overlaps
compute, so this is not 29 ms off mistral's clock. The **count** is the hard fact: explicit deletes
480 of 495 of these ops, in every untied model measured, at every rep. It is the largest term on
explicit's side of the ledger and the most stable, and it is the reason mistral and qwen3 come out
ahead once the LM-head transpose is out of the way.

### 6.4 Errors instead of silent pessimization

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

- **Most models, most geometries: just set `shard_mode: explicit` and write nothing else.** Both
  kernel-order flags now default on where they can act, and with them explicit is at parity with
  `auto` on every onboarded model at every healthy depth measured: median **+0.000%** and worst case
  **+0.133%** across all eight models on the shipped defaults (§5.6), median +0.005% across the 15
  (model, depth) pairs of §4.8. Choose on capability.
- **Do not write `dense_weight_grad_in_kernel_order` out any more.** The "gemma3 and deepseek only"
  advice this line used to carry was an artifact of benchmarking at 16 layers (§4.9); the flag is
  now on by default under explicit and there is no model it should be turned off for.
- **`base_num_decoder_layers` a multiple of 8?** You are paying ~25% of step time to an XLA layout
  pathology, in *either* mode, and `param_scan_axis: 0` recovers it (§4.9). Unrelated to
  `shard_mode`, larger than everything else in this document, and not yet validated as a default —
  measure it on your config.
- **mixtral only:** explicit with **neither** kernel-order flag is ~1.4% faster than `auto` at
  L12–L16, and the shipped default gives that up to land on parity (−0.078% at L14, §5.6). If you
  run mixtral and can measure it, `lm_head_weight_grad_in_kernel_order: false` is worth trying; no
  other model behaves this way, and deepseek behaves the opposite way (§4.8).
- **gemma3 specifically:** never benchmark it below 6 layers, and prefer a multiple of 6.
  `scan_length = num_layers // 6`, so a 4-layer proxy unrolls the whole decoder and measures a
  different program (§4.7), and any remainder is unrolled alongside the scan.
- **Wide models (emb ≥ 4096):** the +1.49% at emb 4096 in §5.3 predates the LM-head flag and has not
  been re-measured with it on. Profile before committing.
- **MoE with gradient accumulation and `data > 1`:** explicit is the only mode that can express
  deferred expert-weight all-reduce. This is the case explicit exists for — though note that path
  is unexercised by anything measured here, so its benefit is a code-level claim, not a measurement.
- **Pure data parallelism:** the +10% in §5.2 predates the flag, and the flag's mechanism is exactly
  the one that dominated there (§8 item 1 measured the same fix closing the whole +9.3% gap). It
  should now be a non-issue, but it has not been re-measured; verify rather than assume.
- **If you need `check_vma`, `fused_qkv`, or context parallelism on non-TPU attention:** you must
  use `auto`. These are hard blocks, not slowdowns (§7.1).
- **Pure tensor parallelism:** explicit is mildly better (−0.2%), for the reason in §5.2. The
  advantage may not survive at a real embedding width.
- **Small models / debugging runs:** do not benchmark a 4-layer proxy and extrapolate — at emb 512
  the penalty is a benchmark artifact in both directions (§5.3).

### For MaxText, ranked by value

01. **Emit the untied LM head's weight gradient in kernel order — `lm_head_weight_grad_in_kernel_order`
    (§4.1, §4.2, §4.3, §4.5).**

    > Two earlier revisions of this item have been withdrawn. The first recommended "stop pinning the
    > logits output under explicit": **measured false**, gating the `out_sharding` produces a 0-line
    > HLO diff, because the barriers are JAX's and not MaxText's (§4.5). The second recommended
    > `lm_head_kernel_transposed`, storing the kernel as `[vocab, embed]`: that one works, but it is
    > dominated on every axis by what follows, so it has been removed rather than shipped beside it.

    Since the barrier cannot be removed globally, remove it *locally*, for the length of the one dot
    that needs the fold. Autodiff's default transpose rule builds the weight gradient as
    `transpose(dot(g, inputs))` and relies on `algsimp` folding the transpose back into the dot;
    under explicit that fold never happens because a `Sharding` custom-call sits between the two.
    Tracing the forward dot inside a `jax.sharding.auto_axes` region takes its operands' shardings
    out of the type system for that dot only — Shardy then re-propagates them exactly as
    `shard_mode: auto` does — so the barrier is never emitted there, XLA folds the transpose itself,
    and the gradient comes out in the kernel's stored order. `_dot_general_in_auto_axes` in
    `linears.py` is the whole implementation, `DenseGeneral`'s `weight_grad_in_kernel_order` gates
    it, and `lm_head_weight_grad_in_kernel_order` in `base.yml` wires it to the untied head. It
    self-disables outside an explicit mesh (where there is no barrier) and under quantization (where
    it does not own the dot).

    > *Implementation note (2026-09-03).* This shipped first as a hand-written `custom_vjp`
    > (`_dot_general_kernel_ordered_grad`) that contracted `dk` straight into the kernel's axis order,
    > so that no transpose was ever emitted. The `auto_axes` region reaches the same place by letting
    > XLA do the fold, and is better on four counts: it is **114 lines shorter**, its gradients are
    > **bit-identical** to the default rule rather than agreeing to rounding (it reassociates
    > nothing — the numerical-equivalence table below becomes vacuous for column D), it has no
    > batch-dimension restriction so it reaches all 17 call sites including MLA and the MoE shared
    > MLP, and it cannot drift from JAX's transpose rule because it *is* JAX's transpose rule. The
    > two were measured against each other on all five models at both flag settings and agree to
    > within **0.15 pp** everywhere (llama2 −0.494 vs −0.449, qwen3 −0.537 vs −0.443, mixtral +0.041
    > vs +0.083, gemma3 +0.803 vs +0.958, deepseek +0.176 vs +0.202). The `custom_vjp` has been
    > deleted.

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

    **Update (2026-09-03): the flag is now on by default where it can act.** The table above is
    seven configs at two geometries; the case for a default needed the flag to be safe on *every*
    model, and it is:

    | model (d16, above gate) | `explicit` | `explicit` + flag |
    | ----------------------- | ---------: | ----------------: |
    | qwen3-8b                |    +3.252% |       **−0.405%** |
    | deepseek3-16b (L8)      |    +0.920% |       **+0.164%** |
    | llama2-7b               |    −0.061% |       **−0.468%** |
    | mistral-7b              |    −0.186% |       **−0.548%** |
    | mixtral-8x7b            |    −1.393% |           +0.009% |

    Plus two controls on mixtral, the one model where the flag does not help: at **L12 / L14 / L16**
    it reads +0.120% / −0.151% / −0.038%, oscillating around zero rather than trending; and
    shrinking the vocab **8× (32000 → 4096)** leaves the spread unchanged (`explicit` −1.580% vs
    −1.441%, with-flag −0.081% at both). Mixtral's −1.393% is therefore an MoE-scan scheduling
    windfall that has nothing to do with the LM head's size, and the flag costs it nothing — it
    lands on `auto`'s schedule rather than losing to it. **Across every model, depth and vocab
    measured, the flag has never been worse than `auto`.** So it resolves to on for an untied model
    under `shard_mode: explicit` and off everywhere else; `base.yml` writes `None` and
    `resolve_lm_head_weight_grad_in_kernel_order` in `types.py` does the resolution, rejecting an
    explicit `true` on a tied head rather than silently ignoring it.

02. **Ship `dense_weight_grad_in_kernel_order` — on by default under explicit (§4.8).** The same
    region wired to every other `DenseGeneral`: the three `MlpBlock` projections
    (`linears.py:650,670,689`), the four attention projections (`attentions.py:700,737,779,835`) and
    the nine MLA / Indexer projections (`attention_mla.py:141,157,177,814,830,852,868,893`).

    Unlike the LM-head flag this one is **not** a missed optimization with one sign. Under explicit
    XLA is free to choose the dense weight gradient's layout, because the `Sharding` custom-calls
    decouple it from the stored parameter's; the flag declines that freedom and reproduces `auto`'s
    layout op-for-op. Measured at d16, on top of the LM-head default:

    | model      | default | + dense flag |            |
    | ---------- | ------: | -----------: | ---------- |
    | gemma3-L18 | +0.690% |  **−0.267%** | **use it** |
    | deepseek   | +0.164% |  **−0.003%** | **use it** |
    | mixtral    | +0.009% |      +0.065% | leave off  |
    | gemma2-2b  | −0.518% |      −0.327% | leave off  |
    | gemma-2b   | −0.213% |      +0.165% | leave off  |
    | llama2     | −0.468% |      +0.013% | leave off  |
    | mistral    | −0.548% |      −0.049% | leave off  |
    | qwen3      | −0.405% |      +0.192% | leave off  |

    (That is the one build carrying both arms for all eight models, so it is quoted whole rather
    than mixed with §5.4's. The later, cleaner build of §5.4 reproduces both recommendations at the
    same magnitude: gemma3 +0.803% → **−0.101%**, deepseek +0.176% → **+0.021%**.)

    The discriminator is the scanned stack depth — the two winners have gradient stacks of 3 and 1,
    the six losers all have 16 — but that is a proxy read off layouts, not a proven mechanism, and
    two points below the line against six above is not enough to fit a threshold on. Encoding it as
    a default would be fitting an XLA layout heuristic. **Leave it off, document the two models, and
    revisit if a third data point ever lands between 3 and 16.**

    > *Reversed (2026-09-03).* The "discriminator" was §4.9. Stack depth 16 is one of the lengths
    > where `param_scan_axis: 1` degenerates and the step gets 25% more expensive in **both** modes;
    > the six models that "lost" were all measured there, and what was being read as a layout trade
    > was scheduling noise on top of an artifact. Measured at depths that are not multiples of 8 —
    > 15 (model, depth) pairs across seven models — the flag never costs more than 0.19 pp and
    > removes a +0.35% to +1.04% penalty on 12 of 15 (§4.8, resolution table). It is therefore now
    > tri-state like its LM-head counterpart and resolves to **on** under `shard_mode: explicit`;
    > `resolve_dense_weight_grad_in_kernel_order` in `types.py` does the resolution. There is no
    > tied-head carve-out here — every model has these projections.
    >
    > The two flags are independent and additive: at L18, explicit costs +1.098% (llama2) / +5.579%
    > (qwen3) with neither, +0.621% / +4.853% with this one alone, +0.429% / +0.534% with the LM-head
    > flag alone, and **+0.005% / +0.008%** with both.
    >
    > One model pays for the *LM-head* default: mixtral is −1.4% against `auto` with **neither** flag
    > at L12/L14/L16 and lands on parity with both. That is an MoE-scan scheduling windfall unrelated
    > to the head (an 8× vocab shrink leaves it unchanged) and deepseek's MoE has the opposite sign,
    > so it is disclosed rather than encoded. A mixtral user who measures it can set
    > `lm_head_weight_grad_in_kernel_order: false`.

03. **Investigate `param_scan_axis: 0` as the default — it is worth ~25% at any depth that is a
    multiple of 8, in both modes (§4.9).** This is not a sharding item and it is the largest number
    in this document. `param_scan_axis: 1` stacks scanned parameters as `[in, L, out]`; when `L` is
    a multiple of 8 (also at L2 and L4) XLA lays the gradient stack out `{2,0,1}` and every
    per-iteration `dynamic-update-slice` writes a `T(1,128)`-tiled slice — one row per tile.
    Measured on llama2 at d16: **L8 −23.6%, L16 −24.9%** from flipping to `param_scan_axis: 0`,
    against −0.18…−0.28% at L9/L18 where the pathology is absent. It persists at `prod` geometry
    (+14.1% / +11.5%). The ns/layer trend across L2…L24 is flat except at exactly those depths.

    What is *not* established: whether `param_scan_axis: 0` is safe as a global default. It changes
    the on-disk parameter-stack layout, so it interacts with checkpoint compatibility and with every
    optimizer-state layout, and it has only been measured on one model family, one geometry and one
    chip count. The tractable next steps are (a) reproduce the ns/layer discontinuity on a second
    model family and a second topology, (b) check whether the degenerate layout is reachable through
    an XLA layout hint instead of a config change, and (c) work out whether a checkpoint written at
    one `param_scan_axis` can be read at the other. **Until then, the actionable advice is narrower:
    do not benchmark anything at a layer count that is a multiple of 8, and if your production depth
    is one, try the flag.**

    The one thing this item settles immediately is a measurement rule for the rest of this document:
    §5.1/§5.2 (L4), §5.4 (L16, plus deepseek at L8) and §4.8's original table (L16) are all inside
    the pathology, and a sub-1% sharding delta measured there is measuring the artifact's scheduling
    noise. §4.8 and §5.6 are the depths where that is not true.

04. **Do not build a lookup table you index once (§5.5) — and do not expect an A/B to find it.**
    `YarnRotaryEmbedding.freqs_cis` materialized `f32[163840, 32]` inside every layer of every step
    to read 1024 rows out of it; computing the rows straight from `position` is **−1.4% of step
    time in both modes**. It survived a full mode-comparison sweep precisely because both modes paid
    it equally. The detector that found it is cheap and worth keeping: flag any op whose output
    element count is ≥ 8× its largest input's. Run over all eight models' profiles it reproduced
    this hit at exactly the reported magnitude and found **nothing else above ~0.06%** — the
    `timescale` `@property` family, the splash-attention masks (numpy at trace time),
    `generate_attention_mask`'s iota masks (`dot_product` path only, not selected on TPU),
    `PositionalEmbedding._compute_embeddings` (once per step, outside the scan), the MoE one-hots,
    `DeepSeekV4RotaryEmbedding.inv_freq` and all of `src/maxtext/utils/` are clean. The residual
    `broadcast(f32[] 0)` at 0.8–1.2% of step is JAX's `lax.scan` `ys` output buffers, not a MaxText
    bug.

05. **Do not bother eliding redundant `reshard` barriers — tested, worth exactly zero.** The obvious
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

06. **Add `shard_mode` to the golden suites (§7.3).** Until CI compares modes, every regression here
    can silently return. `tests/integration/hlo_diff_test.py` is the cheapest place to start; assert
    on `after_codegen` reduce-scatter counts, not on static collective byte totals.

07. **Any HLO byte-counter used to compare modes must descend into
    `kind=kCustom, calls=%all-reduce-scatter.*` fusions** and report the post-slice shape. Counting
    the `all-reduce` inside the fusion at its pre-slice shape reports auto's wins as if they were
    losses, which is how the "identical bytes" claim survived a full sweep here.

08. **Re-examine the `check_vma` ban (§7.1).** JAX 0.11.1 accepts `check_vma=True` on an all-Explicit
    mesh; the restriction appears to be MaxText-imposed. Lifting it would give the MoE models on the
    explicit whitelist back a documented optimization.

09. **Validate `fused_qkv` + explicit and context-parallel + explicit** in `types.py` so users get a
    MaxText message instead of a raw JAX trace error.

10. **Fix `gcs_utils.upload_dump()`** to no-op (or write locally) for non-`gs://` paths — today it
    crashes after training and destroys the profile.

11. **Make explicit mode not defeat remat CSE on unrolled blocks (§4.7).** With `remat_policy: full`
    and no scan, auto CSEs the recomputed forward against the original (61 remat-tagged instructions
    of 11,209); explicit cannot (1,013 of 12,452), costing 279.5 µs. This bites any config where a
    block is unrolled — gemma3 below 6 layers today, but also `scan_layers: false` runs generally.

12. **Deduplicate the tied-embedding bf16 cast (§4.6).** `embeddings.py:235-240`'s `.T` makes explicit
    materialize the 128 MB embedding shard in bf16 **twice** (+80–82 µs on every gemma). Hoisting the
    cast above the transpose, or reusing one converted value for both consumers, is a pure win and is
    independent of everything else in this document.

13. **Investigate the tuple-repack all-reduce tax (§6.2)** — the one measured explicit cost that grows
    with depth (+112 → +219 → +379 µs from 4 to 12 layers on gemma3).

14. **Delete or rewire `validate_shard_mode`** (§7.4) — as dead code it actively misinforms.

### Levers that were tried and do not work

Recorded so the next person does not spend the time. Each was implemented or measured, not merely
reasoned about, and each survived an adversarial attempt to rescue it.

| candidate                                                                                                                                                                                                                   | verdict                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Stop pinning the logits output under explicit                                                                                                                                                                               | 0-line HLO diff. The barriers are JAX's, not MaxText's (§4.5).                                                                                                                                                                                                                                                     |
| Elide no-op `reshard` barriers                                                                                                                                                                                              | Op-identical executable on all three configs (item 05).                                                                                                                                                                                                                                                            |
| `lm_head_kernel_transposed` (store the kernel `[vocab, embed]`)                                                                                                                                                             | Works, but slower on 4 of 7 configs, no checkpoint path, and re-initializes the model. Removed (item 01).                                                                                                                                                                                                          |
| Reshard the RMSNorm scale under explicit outside TP                                                                                                                                                                         | Implemented and reverted — negative on every non-TP config.                                                                                                                                                                                                                                                        |
| Sliced/prefetched MoE expert weights via MSA                                                                                                                                                                                | No measurable effect; the expert all-gathers are already overlapped.                                                                                                                                                                                                                                               |
| Remove the rotated-reduce-scatter `collective-permute` fixup                                                                                                                                                                | Backwards: it is an `auto`-mode op that explicit already deletes (§6.3). There was nothing on the explicit side to remove, and it is not caused by `_permuted_sharding`.                                                                                                                                           |
| `param_scan_axis: 1` + XLA tiling as the dense-flag discriminator                                                                                                                                                           | Correct as stated — every model uses the same value, so it cannot separate models — but the conclusion drawn from it was wrong. It separates *depths*, and that is the whole discriminator: §4.9, worth ~25% of step time in both modes.                                                                           |
| 3-D kernels / MLA low-rank / gemma3's attention pattern / contracting-axis-vs-FSDP, as the dense-flag discriminator                                                                                                         | None separate the sign. gemma3-L18's win is carried by plain 2-D `[embed, mlp]` `MlpBlock` kernels — the same sites that lose on llama2 — and deepseek's by its length-1 dense-layer scan (4,640 µs of `f32[512,1,8192]` copies) plus 500 µs on the MoE shared MLP, not by MLA (MLA's 4-D gradients move ~170 µs). |
| Rematerializing RoPE sin/cos in the backward pass                                                                                                                                                                           | Mode-neutral: identical to 0.1 µs in both arms.                                                                                                                                                                                                                                                                    |
| `LlamaVisionRotaryEmbedding`'s full-table build                                                                                                                                                                             | Consumes the whole table by design, and `llama4` is not on the explicit whitelist — unreachable.                                                                                                                                                                                                                   |
| Deduplicating the out-projection all-gather                                                                                                                                                                                 | The "extra" all-gather is the same op scheduled differently; no bytes to remove.                                                                                                                                                                                                                                   |
| The gemma2 padded-reduce-scatter `concatenate`                                                                                                                                                                              | Sign flips by rep; within noise.                                                                                                                                                                                                                                                                                   |
| deepseek's two remaining duplicated dense-layer weight all-gathers (`bf16[2048,1,8,192]` 371 µs / 12 ops, `bf16[2048,1,576]` 189 µs / 12 ops, both `%copy-done` duplicating an existing `%convert_element_type` all-gather) | Real, explicit-only, and worth **0.009%**. Not worth chasing.                                                                                                                                                                                                                                                      |

A related null result worth stating positively: a systematic hunt for *any* remaining explicit-only
cost on llama2 / mistral / qwen3 at d16 found **nothing above ~0.10% of step**. The largest positive
is llama2's scan gradient-stack `dynamic-update-slice` at +68 µs/step (+0.098%), and it has an
identical signature and op count (960 / 960) in both arms with a sign that flips by model — i.e. it
is scheduling around a shared op, not an explicit-mode defect.

> *Postscript (2026-09-03).* That last sentence is right about the sharding question and badly
> understated about everything else. The shared op it dismisses is §4.9's degenerate
> `dynamic-update-slice`, and at d16/L16 it is not +68 µs of *delta* but **8.2 ms of step**, ~22% of
> the whole thing, paid identically by both modes. A/B diffing two modes cannot see a cost both
> modes pay — the same blind spot that hid the YaRN table (§5.5). Both were found only by looking at
> one mode's absolute profile, which is now the second entry in a very short list of things a
> mode-comparison sweep structurally cannot find.

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
- The geometry above is the *shrink* used by §5.1–§5.2 and by §4's dumps. Everything from §4.8
  onward — including §5.6 — uses `base_emb_dim=2048 base_mlp_dim=8192 base_num_query_heads=8 base_num_kv_heads=8 head_dim=128` (referred to as **d16**) with `base_num_decoder_layers` varied
  per row, plus `base_moe_mlp_dim=8192` on mixtral. §5.6 writes out nothing else but `shard_mode`.

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

Added with the 2026-09-03 revision:

- **§5.4 is one geometry.** emb 2048 / mlp 8192 / 16 layers / seq 1024 / pdbs 1 / fsdp 4 on four
  chips. It is above the AR→RS gate, which is what makes it the right table to quote, but it is not
  a production model and it is not a production mesh. Nothing here was re-measured at emb 4096,
  seq ≥ 4096, pure DP, pure TP or multi-host with the new default on — the §5.2 and §5.3 rows are
  all pre-flag and should be read as upper bounds.

- **3 reps per arm, medians compared.** Several §5.4 deltas (mixtral +0.009%, deepseek's +0.012%
  over 8 reps) are inside the noise floor and are reported as ties, not wins. Only the ≥ 0.2% rows
  should be read as directional.

- **The dense flag's discriminator is fitted to eight points**, with two on one side of the
  threshold and six on the other, and it is a layout proxy rather than a proven mechanism (§4.8).
  It is documented as a per-model recommendation for exactly the two models it was measured on.

  > *Withdrawn (2026-09-03).* There is no discriminator; all eight points were measured inside
  > §4.9's layout pathology. See §4.8's resolution table.

- **The YaRN finding was measured on deepseek only.** It is the only model in the sweep whose
  `max_position_embeddings` (163,840) dwarfs `max_target_length` by enough for the table build to
  dominate; the fix is unconditional, so other YaRN models get whatever their ratio is worth, which
  was not measured.

- **`deepseek3-671b-batchsplit` newly resolves the LM-head flag to `true`** and was verified by
  inspection only — the flag's permutation is the identity for a 2-D `[embed_vocab, vocab]` kernel,
  and the `reduced=`/`unreduced=` tags live on decoder-layer weight gradients inside
  `deepseek_batchsplit.py`'s `shard_map`s, never on the LM head. That config does not fit on the
  hardware used here, so it has not been run.

  > *Amended (2026-09-03).* It now resolves **both** flags to `true`. The dense flag reaches the
  > decoder-layer projections, which is where `deepseek_batchsplit.py`'s `reduced=`/`unreduced=`
  > tags live. Those tags are applied to gradients inside a `shard_map`, and the `auto_axes` region
  > wraps only the forward `dot_general` in `DenseGeneral` — a different program point — so the two
  > should not interact. That reasoning has not been run on hardware either.

Added with the second 2026-09-03 revision:

- **§4.9's pathology is characterized on one model family.** llama2 at d16 and at `prod`, plus the
  consequence visible in gemma3's and deepseek's short scans. The rule "stack length a multiple of 8
  (also 2 and 4)" is read off 15 depths on one model; it has not been checked on a second topology,
  a second chip generation, or a non-Adam optimizer, and no XLA-side explanation for *why* layout
  assignment picks `{2,0,1}` at those lengths has been established.
- **The dense flag's new default rests on 15 (model, depth) pairs at one geometry.** Seven models,
  depths 12–20, d16, 2 reps each. Its worst measured cost is +0.134% (mixtral L12). It has not been
  measured at `prod`, at emb 4096, under TP or DP, or on any depth above 20.
- **The `auto_axes` and `custom_vjp` implementations were compared at L16** — inside the pathology.
  Their agreement (≤0.15 pp on five models, both flag settings) is therefore an agreement between
  two noisy measurements; the claim it supports is "the swap is not a regression", not "the swap is
  worth +0.05%". The bit-exactness claim is separate and is asserted by unit test, not by benchmark.
- **mixtral's −1.4% no-flag win is unexplained.** It reproduces at L12, L14 and L16, survives an 8×
  vocab shrink, and has the opposite sign on the other MoE model in the sweep. It is disclosed as a
  cost of the shipped default rather than attributed.
- **§5.6's two MoE rows are at different depths from the other six.** mixtral is at 14 layers and
  deepseek at 12 because at 18 both exceed HBM at this geometry (deepseek asks 43.56 G of HLO
  temporaries against 31.24 G available). Both are healthy stack lengths under §4.9, so the
  comparison is sound, but the eight rows are not all the same program size and the table should not
  be read as a cross-model ranking — only each row against its own `auto`.
- **Nothing in §5.6 is multi-host.** Every number in this document is a single 4-chip v5e host. The
  collective topology that explicit's `reduced=`/`unreduced=` tags exist to exploit (§6.1) only
  appears at scale, and so does the failure mode that would most plausibly break parity.
