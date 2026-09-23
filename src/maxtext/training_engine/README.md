# `MaxTextTrainingEngine`: standalone validation and benchmark harness

## What this is for

In the Trellis / Tunix RL stack MaxText stops being a training loop. It becomes `MaxTextTrainingEngine`, a
stateful library the orchestrator drives one call at a time: `fwd_bwd(micro_batch)` per micro-batch, then
`update()`, with validation, log-prob scoring and weight sync as separate calls the orchestrator makes when it
chooses. Every Qwen3.5-397B-A17B trainer number so far came from `train.py` instead: one fused `lax.scan` over
gradient accumulation, MaxText's own cross-entropy loss, random routing.

Before the 1,024-chip vLLM sampler is attached, the trainer half has to be proven on its own, at the cells it will
run: 128 v7x chips (1:8 against the sampler) and 256 (1:4), sequence 65,536, GBS 1,024. That means answering, per
cell:

1. **Does it fit in HBM** with AdamW and the RL loss (Tunix GRPO), at micro-batch 64 and the 32 fallback, with
   router replay?
2. **How fast is a step**, in tokens/s/chip and MFU, against the targets: at least 1,610 tokens/s/chip on 256
   chips, so training hides behind a 163 s rollout, and progress from 2,048.8 toward about 3,220 on 128?
3. **Do the read-only paths work at that cell**: a validation pass that moves no training state, forward-only
   log-prob scoring (reference KL, sampler agreement), and weight-sync staging?

And the engine needs a **standalone validation method** the orchestrator can call, rather than validation
buried inside a loop. That is `run_eval` (below).

## How this differs from `experimental/maxtext_engine/train.py`

`src/maxtext/experimental/maxtext_engine/train.py` (Surbhi Jain, #5269) is a *training driver*: it trains a
model through the engine the way `train.py` does, with MaxText's data pipeline and checkpointing. This harness is
a *measuring instrument* for the engine in its RL-trainer role. Both call the same `fwd_bwd` x G + `update`.

| | `experimental/maxtext_engine/train.py` | `training_engine/engine_benchmark.py` |
|---|---|---|
| purpose | train a model through the engine | size, time and check the engine at a cell, before a sampler exists |
| loss | MaxText cross-entropy | cross-entropy, or Tunix `grpo_loss_fn` through `TunixMaxTextAdapter`, optionally chunked |
| data | MaxText input pipeline (dataset, tokenizer) | synthetic micro-batches built on device at the exact RL shapes: `RLTrainerPayload` with prompt/completion split, sampler log-probs, router-replay `routed_experts` |
| checkpoints | restores and saves every step | none |
| what it reports | the engine's loss log | step time until the step's outputs exist; tokens/s per device and chip, TFLOP/s and MFU on `train.py`'s accounting; a JSON report |
| correctness gates | none | finite loss and gradient norm; update not skipped; no kernel recompile in the timed window; exit code |
| validation, scoring, weight sync | no | `run_eval`, forward-only log-prob scoring through `fwd_only`, `prepare_weight_sync` |
| memory without a TPU | no | `--mode=aot`: every kernel, eval included, compiled for the topology on a CPU host; per-kernel memory and device peak |
| profiling | the engine's profiler, counted in micro-steps | whole optimizer steps, traced after the timed window |
| cell presets | no | `scripts/qwen35_397b_engine_benchmark.sh`, carrying the benchmark job's flags |

**Why this is still needed.** None of the three questions above can be answered with `train.py`: it has no GRPO
loss, reports no throughput, cannot check memory without a slice, and never exercises validation, scoring or
weight sync. Run it as the plain SFT baseline ("does the engine train the 397B at this cell at all"); run this
harness for the numbers the RL trainer is held to.

## Files

| file | what it is |
|---|---|
| `engine_benchmark.py` | The harness. `--mode=aot` compiles every engine kernel for a TPU topology on a CPU host and reports memory; `--mode=run` executes on a slice and reports throughput. `--loss_type=sft` or `grpo`. |
| `scripts/qwen35_397b_engine_benchmark.sh` | Launcher for the 397B at the benchmarked v7x cells, carrying the `train.py` job's flags verbatim. |
| `maxtext_engine.py` | Adds `run_eval()`, the standalone validation entry point, plus `compile_eval_kernel()` and `input_sharding()`. |
| `maxtext_engine_compile.py` | The ahead-of-time engine: Tunix-adapter support (so the GRPO loss can be compiled) and a fix that lets it compile Qwen3.5 at all. |
| `../../../tests/post_training/unit/maxtext_engine_profiling_test.py` | Profiling tests updated for the profiler switch (#5316). |
| `../../../tests/post_training/unit/engine_benchmark_test.py` | Tests for the harness, including end-to-end runs on a 4-device CPU mesh. |

## Quick start

### Ahead of time, on a CPU host (no TPU needed)
```bash
# From the repo root, in an environment with MaxText's post-training dependencies (tunix).
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128 aot grpo --report_path=/tmp/aot.json
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128 aot grpo --router_replay
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128 aot grpo --compute_logps_chunk_size=2048
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128-mbs32 aot grpo
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 256 aot sft
# Any MaxText key after the cell replaces the preset, e.g. the GDN remat setting (see Verified results):
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128 aot grpo gdn=remat gdn_conv=remat
```
The launcher passes the benchmark job's 63 MaxText flags unchanged (this branch accepts `use_gdn_kernel`, `gdn`
and `gdn_conv`). A 397B arm took 8 to 14 minutes on a 160-vCPU host (measured `compile_s`: about 460-800 s). It compiles `fwd_bwd`, `fwd_bwd_accum`, `update`
and `eval` one at a time, so a kernel that does not fit still leaves the others' numbers and the compiler's own
out-of-memory figure in the report.

On a shared host, run each compile in its own mount namespace with a private `/tmp`
(`unshare -rm bash -c 'mount -t tmpfs tmpfs /tmp; ...'`) and unset `JAX_PLATFORMS`: libtpu takes a lockfile in
`/tmp`, and a TPU topology cannot be built under `JAX_PLATFORMS=cpu`. AOT mode also sets
`ALLOW_MULTIPLE_LIBTPU_LOAD=1` for itself, which is safe because compiling touches no device.

### Live, on the TPU slice
```bash
bash src/maxtext/training_engine/scripts/qwen35_397b_engine_benchmark.sh 128 run grpo \
  --steps=5 --warmup_steps=1 --eval_batches=2 --logprob_batches=2 --weight_sync \
  --target_tokens_per_sec_per_chip=3220 --report_path=gs://<bucket>/engine/128_grpo.json \
  base_output_directory=gs://<bucket>/engine
```
Run the same command on every host of the slice, as for `train.py`. It exits non-zero if any check fails.
To trace, add `--profile_steps=1`. It traces whole optimizer steps *after* the timed ones, with every
micro-step and the update annotated. Leave MaxText's own profiler off (`profiler=""`, the default). In the
engine it counts micro-steps, so `profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=1` would
trace micro-step 3, not step 3. The harness refuses to run with both on.

A launch that already runs `train.py` with these MaxText flags can switch to the harness by running the module
`maxtext.training_engine.engine_benchmark` instead, with the harness flags (`--mode=run --loss_type=grpo ...`)
in front of the MaxText ones.

### CPU smoke test (a tiny Qwen3.5, minutes)
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
python3 -m maxtext.training_engine.engine_benchmark --mode=run --loss_type=grpo --router_replay --steps=3 \
  --prompt_length=16 --eval_batches=2 --logprob_batches=2 \
  src/maxtext/configs/base.yml model_name=qwen3.5-397b-a17b override_model_config=true use_multimodal=false \
  base_emb_dim=64 base_num_decoder_layers=4 base_num_query_heads=4 base_num_kv_heads=2 head_dim=32 \
  "mrope_section=[2,1,1]" base_mlp_dim=32 base_moe_mlp_dim=32 num_experts=8 num_experts_per_tok=2 \
  gdn_key_head_dim=16 gdn_value_head_dim=16 gdn_num_key_heads=2 gdn_num_value_heads=4 gdn_chunk_size=16 \
  sparse_matmul=false megablox=false attention=dot_product dtype=float32 weight_dtype=float32 scan_layers=true \
  max_target_length=64 ici_fsdp_parallelism=4 per_device_batch_size=1 gradient_accumulation_steps=2 \
  opt_type=adamw learning_rate=1e-4 warmup_steps_fraction=0.0 enable_checkpointing=false skip_jax_distributed_system=true
```
Every GDN, attention and MoE layer type of the real model is present, and so is the router-replay path.

## The cells

| cell | chips (topology) | mesh (cp-as-ep, TP1) | micro-batch x accumulation | tokens/device/micro-batch |
|---|---|---|---|---|
| `128` | 128 v7x (`tpu7x-256`, 256 devices) | FSDP32 x CP4 x EP2 | 64 x 16 = GBS 1024 | 16,384 |
| `128-mbs32` | 128 v7x | **FSDP16 x CP8 x EP2** | 32 x 32 = GBS 1024 | 8,192 |
| `256` | 256 v7x (`tpu7x-512`, 512 devices) | FSDP64 x CP4 x EP2 | 128 x 8 = GBS 1024 | 16,384 |

**Micro-batch 32 cannot run on FSDP32 x EP2.** Under `cp-as-ep` the MoE's `shard_map` shards the micro-batch
over `('fsdp', 'expert')` (`sparse_matmul_route_and_compute`, measured). So FSDP32 x EP2 needs a multiple of
64, and at 32 it fails with `... axis sizes that are not evenly divisible`. `128-mbs32` keeps EP2 and halves FSDP.

## What it measures, and how

**Throughput uses `train.py`'s accounting.** `tokens/s/device = seq x per_device_batch_size x
gradient_accumulation_steps / step_time` and `TFLOP/s/device = calculate_tflops_training_per_device /
step_time`. These are the helpers `train.py`'s metric logger uses, and both already count every micro-batch.
Per-chip figures multiply by the devices per chip, read off the devices' chip coordinates (2 on v7x). This
reproduces both measured `train.py` numbers exactly: 128 chips at 255.9 s/step gives **2,048.8** tokens/s/chip,
and 256 chips at 141.8 s/step gives **1,849** (`ThroughputAccountingTest`). MFU is per-chip TFLOP/s over the
bf16 dense peak per chip (public Cloud TPU specs, overridable with `--peak_tflops_per_chip`). Tokens count
prompt and completion alike, as `train.py` counts every position.

**A step's time runs until that step's own outputs exist.** The clock stops when the metrics buffer (every
micro-batch's loss, and the gradient norm the update returned) is ready, not when the calls return. Kernels are
compiled before step 0 (`engine.compile`). `--warmup_steps` leading steps are left out, and the report gives
median, mean, min and max over the rest. Micro-batches are built on device up front (`--distinct_micro_batches`,
default 2) and cycled, so host-side generation is never on the clock.

**A refused update fails the run.** With `skip_step_on_nan` (on by default) the engine skips an update whose
gradients are non-finite and records `step_skipped`. A skipped step still takes the time of a step, so
`updates_applied` fails if any step reports one.

**Nothing may recompile inside the timed window.** Every XLA backend compile is logged with its function name
and the phase it happened in. If a kernel that `engine.compile` built compiles again during the timed steps,
`no_kernel_recompile_in_timed_steps` fails. A retrace in every step would otherwise look exactly like a slow
kernel. One-off helper compiles (metric reads) are reported, not failed.

**AOT memory is reported per kernel and per device.** `resident = argument + output - alias + temp` comes from
the compiler's memory analysis. That analysis only sees a kernel's own arguments, so the report adds what stays
in HBM beside each kernel: the optimizer moments (arguments of `update` only), the gradient accumulator (beside
an eval that runs mid-step) and the other pooled micro-batches. That sum is `device_peak_gib`, and `fits`
compares its maximum with `--hbm_gib_per_device`. XLA gates compilation on temporaries alone, so "it compiled"
is not "it fits". GRPO's MBS-32 `fwd_bwd` compiled with 107.32 GiB of temporaries against a 94.74 GiB limit.

## The GRPO path

The engine is built the way the RL trainer builds it: wrapped in `TunixMaxTextAdapter`, trained with Tunix's
`algo_core.grpo_loss_fn`, and given the same four-key model-input function (`train_example`, `algo_config`,
`pad_id`, `eos_id`) that Tunix's orchestrator installs over RPC. Each micro-batch is a Tunix `RLTrainerPayload`
shaped as `PaddedBatchAssembler` emits it:

| field | shape | content |
|---|---|---|
| `prompt_ids` / `prompt_mask` | `[B, P]` | random tokens that are neither pad nor eos / ones |
| `completion_ids` / `completion_mask` | `[B, T-P]` | same |
| `advantages` | `[B]` | standard normal |
| `old_per_token_logps` (default on) | `[B, T-P]` | `-log(vocab)`, which is what a randomly initialized model assigns, so the importance ratio sits near 1 as it does on-policy |
| `ref_per_token_logps` (only if `--grpo_beta` != 0) | `[B, T-P]` | same |
| `routed_experts` (`--router_replay`) | `[B, T, layers, top_k]` int32 | `top_k` distinct experts per token and layer, spread as random routing spreads them |

`P` is `--prompt_length` (default 4096, so 4096 + 61440 = 65536). Defaults follow MLPerf's fixed values: PPO
clip 0.2 / 0.28, KL coefficient 0, temperature 1.0. Arrays are built shard by shard on the engine's input
sharding (`engine.input_sharding`), which is how a multi-host driver has to build them, and each shard is seeded
by its global coordinates.

**Chunk the log-probs: unchunked GRPO pays for the full logits.** Unchunked, `grpo_loss_fn` reads the full
`[tokens, 248,320]` logits. The decoder returns none under vocab tiling (`layers/nnx_decoders.py`), so the
harness refuses `num_vocab_tiling > 1` there and the launcher sets 1 for GRPO. Those logits cost about 8-10 GiB
per device at 16,384 tokens/device (measured below). `--compute_logps_chunk_size=N` passes Tunix's
`compute_logps_chunk_size`: the model returns hidden states (`skip_lm_head`) and `compute_final_logits` projects
them N tokens at a time. That brings GRPO's memory down to SFT's, and it works under vocab tiling too. The loss is
unchanged: the tests compare chunked and unchunked step-0 loss and gradient norm on identical data. Tunix's
orchestrator does not send this key today.

## Read-only entry points

- **`engine.run_eval(eval_ds) -> dict`** is new: the standalone validation method. Tunix's
  `TrainerWorker.run_eval` calls a trainer's `run_eval` when it has one, so the orchestrator decides when
  validation happens. It wraps `eval_step` over the stream in `eval_context` and returns every metric the loss
  produced, including aux metrics such as GRPO's `kl` that are not on the logging list, plus `eval_batches`.
  Values are reduced exactly as the engine logs them: each micro-batch's value, then averaged. It is forward
  only; no optimizer state, gradient accumulator, micro-step count or step counter moves, so it can run between
  two `fwd_bwd` calls of an unfinished step.
- `compile_eval_kernel()` compiles the eval kernel ahead of time, so AOT covers validation as well as training.
- The harness scores log-probs forward-only through `engine.fwd_only` with Tunix's `compute_per_token_logps`,
  the reference-KL / sampler-agreement pass, and exercises `prepare_weight_sync` / `release_weight_sync`.

## Checks and exit status

| check | fails when |
|---|---|
| `loss_finite` / `grad_norm_finite` | any step's loss or gradient norm is missing or non-finite |
| `updates_applied` | any step's update was skipped (`step_skipped`) |
| `no_kernel_recompile_in_timed_steps` | an engine kernel compiled again after `engine.compile` |
| `run_eval` | the pass changed training state, lost batches, or returned a non-finite loss |
| `logprob_scoring` | scored log-probs are non-finite or not `[B, completion]` |
| `weight_sync` | staging raised or registered no variables |

Any FAIL makes the process exit non-zero. The JSON report (`--report_path`, local or `gs://`) holds the cell,
per-step times, losses and gradient norms, throughput, compiles per phase, check results, and in AOT mode the
per-kernel memory and device peaks.

## Engine changes

1. `maxtext_engine.py`: `run_eval`; `eval_context` records the pass's reduced metrics; `compile_eval_kernel`;
   `input_sharding`.
2. `metrics.py`: `process_metrics(..., only_logged=False)` reduces every metric, not just the logged ones.
3. `maxtext_engine_compile.py`: `AbstractMaxTextEngine` accepts the Tunix adapter. Its graph-only trace now runs
   with no mesh in context. With a mesh, flax's `nnx.eval_shape` re-derived every variable's sharding from
   logical names through its own lookup, which neither drops a mesh axis claimed twice (Qwen3.5's scanned MLP
   weights put `fsdp_transpose` under both `embed` and `mlp`) nor tolerates an unmapped name (`norm` under
   `cp-as-ep`). The **upstream AOT engine could not compile Qwen3.5 at all**, under either rule set. The xaot
   parity suite (abstract engine vs a live engine that stepped) passes identically before and after the fix.
4. `maxtext_engine_profiling_test.py`: Mohit Khatwani's #5316 made the engine's profiler switch on only when
   `profiler` is set. Before it, `profiler_steps` alone (default 5) traced micro-steps 1-5 of every run; the
   harness had measured that independently as a 7.3 s step among 0.8 s ones. #5316 left the two window tests
   setting `profiler_steps` without `profiler`, so they fail on `atwigg/mlperf`. They now set
   `profiler="xplane"`, and a new test pins that the default config never profiles.

## Verified results

Everything below ran on a 160-vCPU CPU host (jax 0.11.1, flax 0.12.9, libtpu 0.0.46, tunix `main` @ `20c2baa8`)
against this branch on `atwigg/mlperf` @ `024eb4b9d`. **Nothing here has run on a TPU yet**: the live throughput
numbers against the 1,610 / 3,220 tokens/s/chip targets still need a slice, using the commands above.

### Test suites

| suite | this branch | `atwigg/mlperf` alone |
|---|---|---|
| `maxtext_engine_test.py` | 65 passed, 2 failed | 62 passed, 2 failed |
| `engine_benchmark_test.py` | **28 passed** (arithmetic, compile log, payloads, device peak, OOM parsing, 8 live CPU runs, 3 AOT compiles) | not present |
| constructor, data-parallel, packing, profiling, xaot engine suites | **40 passed**, 9 subtests | 37 passed, 2 failed, 9 subtests |

The two `maxtext_engine_test.py` failures fail identically without this branch, and neither touches its code:
`test_gradient_norm_is_recorded_every_step` asserts that `step_skipped` is absent, but #5346 now records it every
update; `test_maybe_register_pathways_persistence` needs the Pathways persistence handler, which this host's
environment does not provide. The two profiling failures on `atwigg/mlperf` alone are the window tests #5316 left
behind, fixed here.

**Every new test has been shown to fail.** Seven mutations, each breaking one mechanism on a scratch copy of this
tree, were each caught by their test: `eval_context` keeping stale metrics; `eval_context` reducing only logged
metrics; `process_metrics` ignoring `only_logged`; the profiler switch reverted; the harness's profiler guard keyed
on `profiler_steps`; the skipped-update check removed; the chunk size not passed to the loss. The harness's gates
also fire on inputs built to trip them: a learning rate of 1e30 fails `loss_finite`, `grad_norm_finite` and
`updates_applied`; unchunked GRPO under vocab tiling is refused; a new input shape shows up as a kernel recompile;
`--hbm_gib_per_device=1e-6` returns `fits: false`. All changed files pass `pyink --pyink-indentation=2
--line-length=122` and pylint with the repo's `pylintrc`.

On a small Qwen3.5 (GDN, attention and MoE layers, router replay; 4 CPU devices) `--mode=run` compiles all four
kernels before step 0, runs with no compile of any kernel in the timed window, and passes `run_eval`,
`logprob_scoring` and `weight_sync`. Chunked and unchunked GRPO give the same step-0 loss and gradient norm.

### AOT at the production cells (GiB per device, HBM 94.74)

Run through the launcher, so with the benchmark job's 63 flags. An OOM row gives the compiler's own temporaries
figure, the only number XLA reports for a kernel it refused. "Device peak" is resident plus what stays in HBM beside
the kernel: 8.67 GiB of optimizer moments at 128 chips (4.34 at 256), and for eval also the gradient accumulator.
Arms named `chunk` use `--compute_logps_chunk_size=2048`.

**The job's flags as they are** (`gdn=device gdn_conv=device`):

| arm | cell | loss | kernel | arg | temp | resident | device peak | verdict |
|---|---|---|---|---|---|---|---|---|
| nb_A1_sft_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | fwd_bwd | | 136.27 | | | OOM (compiler) |
| nb_A1_sft_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | fwd_bwd_accum | | 141.54 | | | OOM (compiler) |
| nb_A1_sft_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A1_sft_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | eval | 5.78 | 30.30 | 36.20 | 50.65 | fits |
| nb_A2_grpo_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd | | 144.11 | | | OOM (compiler) |
| nb_A2_grpo_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd_accum | | 149.88 | | | OOM (compiler) |
| nb_A2_grpo_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A2_grpo_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | eval | 5.78 | 39.37 | 45.15 | 59.59 | fits |
| nb_A2c_grpo_chunk_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd | | 135.64 | | | OOM (compiler) |
| nb_A2c_grpo_chunk_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd_accum | | 141.41 | | | OOM (compiler) |
| nb_A2c_grpo_chunk_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A2c_grpo_chunk_128 | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | eval | 5.78 | 30.31 | 36.09 | 50.53 | fits |
| nb_A4b_sft_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | sft | fwd_bwd | | 94.83 | | | OOM (compiler) |
| nb_A4b_sft_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | sft | fwd_bwd_accum | 11.56 | 107.94 | 119.55 | 128.22 | does not fit |
| nb_A4b_sft_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | sft | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A4b_sft_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | sft | eval | 5.78 | 26.89 | 32.73 | 47.17 | fits |
| nb_A5b_grpo_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | fwd_bwd | 5.78 | 107.32 | 118.88 | 127.54 | does not fit |
| nb_A5b_grpo_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | fwd_bwd_accum | 11.56 | 108.65 | 120.21 | 128.88 | does not fit |
| nb_A5b_grpo_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A5b_grpo_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | eval | 5.78 | 28.57 | 34.35 | 48.79 | fits |
| nb_A5c_grpo_chunk_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | fwd_bwd | 5.78 | 111.67 | 123.23 | 131.90 | does not fit |
| nb_A5c_grpo_chunk_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | fwd_bwd_accum | 11.56 | 108.83 | 120.39 | 129.05 | does not fit |
| nb_A5c_grpo_chunk_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_A5c_grpo_chunk_128mbs32 | 128 ch, fsdp16 cp8 ep2, MBS 32 x 32 | grpo | eval | 5.78 | 26.60 | 32.38 | 46.82 | fits |
| nb_B1_sft_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | sft | fwd_bwd | | 129.09 | | | OOM (compiler) |
| nb_B1_sft_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | sft | fwd_bwd_accum | | 131.98 | | | OOM (compiler) |
| nb_B1_sft_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | sft | update | 10.13 | 0.02 | 10.15 | 10.15 | fits |
| nb_B1_sft_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | sft | eval | 2.89 | 29.51 | 32.53 | 39.76 | fits |
| nb_B2_grpo_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd | | 136.42 | | | OOM (compiler) |
| nb_B2_grpo_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd_accum | | 139.31 | | | OOM (compiler) |
| nb_B2_grpo_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | update | 10.13 | 0.02 | 10.15 | 10.15 | fits |
| nb_B2_grpo_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | eval | 2.89 | 39.36 | 42.26 | 49.49 | fits |
| nb_B2c_grpo_chunk_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd | | 128.46 | | | OOM (compiler) |
| nb_B2c_grpo_chunk_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd_accum | | 131.35 | | | OOM (compiler) |
| nb_B2c_grpo_chunk_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | update | 10.13 | 0.02 | 10.15 | 10.15 | fits |
| nb_B2c_grpo_chunk_256 | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | eval | 2.89 | 29.51 | 32.41 | 39.64 | fits |

**The same, with only `gdn=remat gdn_conv=remat`:**

| arm | cell | loss | kernel | arg | temp | resident | device peak | verdict |
|---|---|---|---|---|---|---|---|---|
| nb_R1_sft_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | fwd_bwd | 5.78 | 97.21 | 108.90 | 117.56 | does not fit |
| nb_R1_sft_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | fwd_bwd_accum | 11.56 | 97.43 | 109.12 | 117.78 | does not fit |
| nb_R1_sft_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_R1_sft_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | sft | eval | 5.78 | 30.30 | 36.20 | 50.65 | fits |
| nb_R2c_grpo_chunk_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd | 5.78 | 95.75 | 107.30 | 115.97 | does not fit |
| nb_R2c_grpo_chunk_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | fwd_bwd_accum | 11.56 | 97.43 | 108.99 | 117.66 | does not fit |
| nb_R2c_grpo_chunk_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | update | 20.22 | 0.02 | 20.25 | 20.25 | fits |
| nb_R2c_grpo_chunk_128_gdnremat | 128 ch, fsdp32 cp4 ep2, MBS 64 x 16 | grpo | eval | 5.78 | 30.31 | 36.09 | 50.53 | fits |
| nb_R3c_grpo_chunk_256_gdnremat | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd | 2.89 | 94.50 | 100.28 | 104.62 | does not fit |
| nb_R3c_grpo_chunk_256_gdnremat | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | fwd_bwd_accum | 5.79 | 100.68 | 106.47 | 110.81 | does not fit |
| nb_R3c_grpo_chunk_256_gdnremat | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | update | 10.13 | 0.02 | 10.15 | 10.15 | fits |
| nb_R3c_grpo_chunk_256_gdnremat | 256 ch, fsdp64 cp4 ep2, MBS 128 x 8 | grpo | eval | 2.89 | 29.51 | 32.41 | 39.64 | fits |

`train_compile` (the fused `train.py` step) at the exact 128-chip cell: **temp 150.22 with `gdn=device`, 95.27 with
`gdn=remat`**, the only difference between the two runs.

### What the numbers say

- **Chunked log-probs remove GRPO's memory surcharge.** Unchunked, GRPO costs 7-8 GiB more than SFT on the training
  kernels and 9-10 GiB more on eval, at 16,384 tokens/device: the full `[tokens, 248,320]` logits. Chunked, it is
  within 1.5 GiB of SFT on every kernel, at both chip counts and with either GDN setting. Tunix's orchestrator does
  not send `compute_logps_chunk_size` today; turning it on in the Trellis path is the GRPO memory lever.
- **`gdn=device gdn_conv=device` costs about 40-55 GiB per device on this branch.** Keeping the GDN residuals on
  device instead of rematerializing them moves `train_compile` from 95.27 to 150.22 GiB of temporaries and the
  engine's training kernels by about 40 GiB. On an older tree carrying the GDN-kernel changes the two settings gave
  identical memory, so the same flags now mean something else. Check `gdn` / `gdn_conv` before running the
  benchmark job's flags on this branch.
- **Even the best configuration is over the limit in AOT**: chunked GRPO with `gdn=remat` peaks at 117.7 GiB per
  device at 128 chips and 110.8 at 256. AOT on this host already disagreed with hardware by at least 20 GiB at this
  cell (below, and Open items), so read this as "measure it on a slice", not as a verdict.
- **Router replay is free in memory** (within 0.7 GiB on every kernel, measured at these cells on the `main`-based
  version of this branch), and the validation pass fits everywhere (eval device peak at most 59.6 GiB, including
  the moments and a live accumulator).
- **XLA's compile check is not a fit verdict.** It gates on temporaries alone, and not consistently: kernels above
  94.74 GiB of temporaries compiled in several arms here. Read `device_peak_gib`.
- FLOPs from XLA's cost analysis are not quoted for chunked arms: the chunked logits run in a loop, which the cost
  analysis does not count per iteration.

### Pre-registration scorecard

Written down and md5-pinned before each wave launched. **11 HIT / 4 MISS** on this base.

| id | prediction | measured | |
|---|---|---|---|
| P0 | `atwigg/mlperf`'s two profiling-window tests fail | fail | HIT |
| P1 | `train_compile` with 63 flags still OOMs, temp 100 ±6 | OOM, temp 150.22 | HIT (verdict), MISS (size) |
| P2 | SFT MBS 64 fwd_bwd 90.6 ±5, accum 96 ±5 | 136.27, 141.54 | MISS |
| P3 | GRPO - SFT: fwd_bwd +[2, 10], eval +[7, 11] | +7.84, +9.07 | 2 HIT |
| P4 | chunked GRPO within ±2 of SFT, every kernel, 128 chips | -0.63 / -0.13 / 0 / +0.01 | HIT |
| P5 | MBS 32 SFT accum 95 ±15; chunked within ±2 of it | 107.94; +0.89 | 2 HIT |
| P6 | chunked GRPO within ±2 of SFT, 256 chips | -0.63 / -0.63 / 0 / 0 | HIT |
| P7 | chunked and unchunked step-0 loss agree | agree | HIT |
| P8 | `gdn=remat` SFT at 81 / 86 ±8, both compile | 97.21 / 97.43, both compile | HIT (compile), MISS (size) |
| P9 | `gdn=remat` chunked GRPO within ±2 of SFT | -1.46 / 0 / 0 / +0.01 | HIT |
| P10 | `gdn=remat` chunked GRPO at 256: 73 / 76 ±8 | 94.50 / 100.68 | MISS |

The misses are all absolute sizes; every relative prediction (GRPO vs SFT, chunked vs unchunked) hit.

## Open items

1. **Run it on a TPU.** `128 run grpo` and `256 run grpo` with `--compute_logps_chunk_size=2048 --steps=5
   --eval_batches=2 --logprob_batches=2 --weight_sync`. That run gives the step time, tokens/s/chip and MFU against
   the 1,610 / 3,220 targets, whether the cell fits on hardware, and the host-memory check for `prepare_weight_sync`
   at 397B, where unscanning 60 layers is the known risk.
2. **Settle `gdn` / `gdn_conv` for this branch.** `device` costs about 40-55 GiB per device here and nothing on the
   older tree the benchmark job's flags came from. Decide the setting before running those flags on this branch.
3. **Turn on chunked log-probs in the Trellis path.** Tunix's orchestrator sends `grpo_loss_fn` four keys and no
   `compute_logps_chunk_size`, so production GRPO pays the full-logits cost this harness measures at 7-10 GiB.
4. **Explain the AOT-vs-hardware gap before trusting an absolute fit verdict.** On `main`, AOT of the exact
   128-chip `train.py` job gave 108.01 GiB of temporaries while the job ran on hardware. Cheapest first: diff the
   job's resolved `Config param` lines against the AOT host's, then the libtpu version, then the code revision.
5. Two `maxtext_engine_test.py` tests fail on `atwigg/mlperf` with or without this branch (see Test suites):
   one still asserts the absence of `step_skipped`, one needs the Pathways persistence handler.
6. `eval_step` traces a *recompiled* eval kernel without the mesh and logical axis rules (`fwd_bwd` and `update`
   enter `_sharding_ctx`; `eval_step` does not). The fix is on the separate, unmerged `engine-eval-sharding-ctx`
   branch. The harness does not hit it: it compiles the eval kernel up front, under the rules, and its compile log
   shows no kernel compile in the eval phase.
7. Eval does not force `enable_dropout=False` on the Tunix-adapter path. It is moot at `dropout_rate: 0.0`, which
   every config here uses.
8. Synthetic `routed_experts` are uniform; a real rollout's routing is imbalanced, which changes the MoE's buffer
   use but not the input's size.
