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

# RL trainer bench

This directory times the RL trainer program of a Tunix GRPO e2e run on its own. The bench runs the exact trainer
program the e2e run compiles (the same MaxText flags, libtpu flags, mesh, GRPO loss and chunked log-prob pass, hence
the same compile key) on one trainer slice, under the same Pathways setup, with a synthetic micro-batch whose MoE
routing load is calibrated to a production run. There is no orchestrator, no rollout engines and no weight sync. A
variant (a MaxText flag, a libtpu flag, a slice size) can therefore be A/B tested in one short job on one slice, and
its profile read, without the rest of the RL system. The bench is standalone and launches no e2e job: its input,
`LAUNCH_RENDER`, is the DRY_RUN output of the full launcher, from which the renderer takes only the trainer JobSet and
the trainer and orchestrator argv.

## Why a separate harness

MaxText's standalone engine script (`python3 -m maxtext.experimental.maxtext_engine.train`) runs MaxText's
pre-training cross-entropy on MaxText's synthetic loader. That is a different program from the RL trainer: the GRPO
loss and its vocabulary-chunked `compute_per_token_logps` pass are missing, the batch is laid out differently, and the
optimizer step is driven by MaxText rather than by Tunix's `TrainerWorker`. Timings from it do not transfer to the RL
step, and a variant that compiles there may not compile in the RL program (or the other way round).

The bench therefore drives the trainer through Tunix's own code paths. `bench_driver.py` builds the trainer worker with
`run_trainer_node._parse_args` and `_create_maxtext_trainer_factory` from the e2e trainer's argv, builds the
`GRPOAdapter`, `BatchConfig` and batch assembler with `run_deepswe_dist._parse_args` and `_build_algo` from the e2e
orchestrator's argv, applies `configure_worker`'s loss and model-input functions, compiles against Tunix's
TRAINER_WARMUP_COMPILE payload, and then runs `STEPS` optimizer steps of `MICRO_STEPS` `fwd_bwd` calls plus one
`update`, as `DistributedRLEngine.train_step` drives them.

## Requirements

The driver imports Tunix (`tunix.experimental.examples.common.run_trainer_node`,
`tunix.experimental.examples.deepswe_dist.run_deepswe_dist`, `trainer_warmup`, `batch_assembly`, `trainer_worker`)
and MaxText, so it must run in an image whose Tunix and MaxText match the e2e run you want to reproduce; by default
the bench keeps the client image of the e2e render. The renderer and the unit tests need only Python 3 with PyYAML
and NumPy. Since most of what the driver touches is Tunix code, it may belong in Tunix in the long run; we leave that
to the team.

## Files

| file | what it does |
|---|---|
| `bench_driver.py` | the bench itself; runs as the Pathways client of the trainer slice |
| `render_bench.py`, `render_bench.sh` | turn an e2e DRY_RUN render into the bench JobSet, with variant knobs |
| `structdiff_render.py` | structured diff of two bench renders (YAML leaves, MaxText flags, libtpu flags, argv) |
| `diff_config.py` | diff of the resolved MaxText `Config param` lines of two logs (e2e trainer vs bench) |
| `aot_trainer.py`, `aot_bench.sh`, `aot_summarize.py` | ahead-of-time compile and memory report on a CPU host |
| `check_payload_routing.py` | CPU check of the realistic payload at the 397B production shapes (needs Tunix) |
| `check_numerics_cpu.py` | CPU check of the numerics knobs with Tunix's GRPO loss (needs Tunix and MaxText) |
| `hlo_collective_schedule.py` | static overlap summary of the collectives in a scheduled HLO dump |
| `tests/` | CPU unit tests of the renderer, the payload builder and the driver's step loop |

## How to use it

### 1. Render your own e2e launch with DRY_RUN

Start from the launch you want to reproduce, with the environment you would launch it with, and ask the team launcher
for a dry run. For the MLPerf recipe in Tunix this is

```bash
DRY_RUN=true bash tunix/experimental/examples/recipes/mlperf_gbs1024_submission.sh start > launch-dryrun.yaml
```

The output is a stream of JobSets (orchestrator, trainer, one per rollout engine), possibly interleaved with
launcher log lines. The renderer picks the trainer and the orchestrator JobSet by the `--process_main=` of their
`main` container and ignores everything else. If you already have the two JobSets as separate files, pass them with
`--train-yaml` and `--orch-yaml` instead. The DRY_RUN output describes your environment (namespace, images,
buckets), so keep it local. In our runs the dry run still switched the current kubectl context and printed (without
executing) the delete commands of a stop, so check your context afterwards.

### 2. Render the bench JobSet

```bash
LAUNCH_RENDER=launch-dryrun.yaml NAME=tb-base K8S_NAMESPACE=<namespace> OUTPUT_PREFIX=<output-prefix> \
  bash render_bench.sh                   # writes rendered/tb-base.yaml and nothing else
```

The render keeps every scheduling field of the e2e trainer JobSet (Kueue labels and annotations, node selectors,
tolerations, reservations, priority class, service account, the Pathways resource manager, proxy, worker and sidecar
containers with their images, args and env). It changes the names, sets `failurePolicy.maxRestarts` to 0 (so a
failure does not rerun the compile), and replaces the client command: the same inline env prefix as the e2e trainer
(MAXTEXT_EXTRA_FLAGS, LIBTPU_INIT_ARGS and the rest) runs `bench_driver.py`, written into the pod by the command itself,
with the e2e trainer argv and orchestrator argv as `TRAINER_ARGV` and `ORCH_ARGV`. In the trainer argv,
`--maxtext_ckpt_path` is cleared (random initialization is the one intended difference), `--maxtext_output_directory`
becomes `<output-prefix>/<NAME>`, `--worker_id` becomes `NAME`, and the gRPC and discovery flags are dropped. The
Tunix fail-fast wrapper is dropped as well, since it rewrites a clean exit into a restart.

The knobs, all environment variables of `render_bench.sh` (the Python script takes the same as flags):

`IMAGE` replaces the client image; the Pathways containers keep the e2e render's. `STEPS` (default 3) is the number of
optimizer steps and `MICRO_STEPS` the `fwd_bwd` calls per step. Set `MICRO_STEPS` to what your e2e run does per
optimizer step (the orchestrator logs one train call per micro-batch); the default of 9 (5 with
`TRAINER_SLICE=4x8x8`) is an example calibration from our 397B runs, where about 1,152 trajectories per step packed
into 9 micro-batches of 64 rows on a 4x4x8 trainer.

`EXTRA_MAXTEXT_FLAGS="k=v ..."` appends MaxText flags (MaxText keys are last-wins), and
`REPLACE_MAXTEXT_FLAGS="k=v ..."` replaces an existing key in place (it fails unless the key is there exactly once),
which keeps one value per key in the rendered flag set, e.g. `REPLACE_MAXTEXT_FLAGS="context=remat"`.
`SET_TRAINER_FLAGS` and `SET_ORCH_FLAGS` do the same for the trainer and orchestrator argv, e.g.
`train_micro_batch_size=128` in both. `COMPUTE_LOGPS_CHUNK_SIZE` sets the trainer's `--compute_logps_chunk_size`.

`LIBTPU_EXTRA="--flag=v ..."` adds or overrides libtpu flags and `LIBTPU_DROP="--flag ..."` removes them by name. Both
are written to the three places the e2e render carries libtpu flags: the client's LIBTPU_INIT_ARGS, the
`pathways-proxy` `--xla_*` args (where the programs are compiled) and the `pw-node` workers' LIBTPU_INIT_ARGS (which
keep their worker-only extras).

`TRAINER_SLICE=<topology>` (e.g. `4x8x8`) re-targets the trainer at another slice of the same accelerator and fills
it with FSDP, keeping expert, context and tensor parallelism. It changes the resource manager's `--instance_type`,
the `pw-node` completions and parallelism (hosts = chips / `google.com/tpu` per pod), the slice-topology annotation,
the trainer's `--mesh_fsdp` and the orchestrator's `--trainer_fsdp`. The last one sets the packed micro-batch rows
(fsdp x expert), so the number of micro-steps for the same token mass changes too; set `MICRO_STEPS` and
`FINAL_PAD_FRACTION` to keep the real rows per step comparable.

`PROFILER_SKIP=<n> PROFILER_STEPS=<m>` sets the trainer's `--skip_first_n_profiler_steps` and `--profiler_steps`,
which Tunix maps to MaxText's `profiler=xplane`, `skip_first_n_steps_for_profiler` and `profiler_steps`. MaxText's
`training_engine/micro_step_profiler.py` counts `fwd_bwd` calls from 0 over the whole run (the compile does not
count), opens the trace before call `n` and closes it after call `n + m - 1`. So with `MICRO_STEPS=9`,
`PROFILER_SKIP=16 PROFILER_STEPS=3` traces calls 16, 17 and 18: the second-to-last and last micro-batch of step 1,
step 1's `update`, and the first micro-batch of step 2. That is the window to use for the last micro-batch and the
optimizer update; `STEPS` must be large enough for the window to close. Profiling options (chips and SparseCores to
trace) are MaxText flags and go in `EXTRA_MAXTEXT_FLAGS`.

`DRIVER_ENV="K=V ..."` passes knobs to the driver; see the next section.

To compare two renders, or a render with the e2e trainer JobSet it came from:

```bash
python3 structdiff_render.py rendered/tb-base.yaml rendered/tb-variant.yaml
```

It prints only differences, by YAML leaf and by flag (MaxText flags, libtpu flags, trainer argv, orchestrator argv,
driver env), so it is quick to confirm that a variant differs in the intended fields only.

### 3. Apply

```bash
kubectl --context <context> -n <namespace> apply --dry-run=server -f rendered/tb-base.yaml   # optional check
kubectl --context <context> -n <namespace> apply -f rendered/tb-base.yaml
kubectl --context <context> -n <namespace> logs -f -c main \
  -l jobset.sigs.k8s.io/jobset-name=tb-base,jobset.sigs.k8s.io/replicatedjob-name=proc 2>&1 | tee -a tb-base.log
kubectl --context <context> -n <namespace> delete jobset tb-base                              # when done
```

### 4. Read the result

The client log carries one line per optimizer step:

```
completed step: <n>, seconds: <s>, micro_steps: <k>, seconds/micro_step: <s/k>, fwd_bwd dispatch s: [...]
```

The step time runs from the first `fwd_bwd` dispatch until the step's device work has drained (the driver waits on
the engine's throttler), and includes the `update`. Step 0 includes first execution, and any step that overlaps the
profiler window carries the profiler's start and stop cost, so compare steps that are neither. After each step the
driver also logs `STEP_LOSS` (pooled and per-micro-batch loss and denominators, gradient norm) and `STEP_METRICS`
(every trainer metric, per-micro-batch ones as lists), read after the step's timing so that they do not affect it.

`python3 diff_config.py <e2e trainer log> <bench log>` compares the resolved MaxText `Config param` lines of the two
runs and prints the keys that differ; we expect only the run name, the output directory and the checkpoint path to
differ.

The profile goes under `<output-prefix>/<NAME>/tensorboard/` in the usual `plugins/profile/<timestamp>/` layout of
`jax.profiler`.

## The payload

Tunix's TRAINER_WARMUP_COMPILE payload fixes the shapes, dtypes, fields and per-row segment count of the packed
micro-batch, which is what the compile needs. As a timing payload it is degenerate for a MoE model: at the 397B shapes
it was about 97% padding in our runs (measured with `check_payload_routing.py`), every token carries the same id and every routed expert is UNSET, so with router replay every
token falls back to the router with identical inputs and lands on the same experts.

`PAYLOAD=realistic` (the default) keeps the warm-up payload's structure and rebuilds its contents with
`realistic_payload`. Real tokens get seeded ids in [256, vocab). Each row's tail is padding, a fraction
`PAD_FRACTION` (default 0.07) of the row, written as the packer writes it (segment id 0, pad id, zero masks, UNSET
routing). Real tokens get replayed top-k routing with distinct experts, so MaxText replays them rather than routing
them. With `ROUTING=probe` (the default) each layer draws from a pool whose max expert-shard load ratio (under the
EP=8 cp-as-ep layout) is calibrated by bisection to a per-layer target, with a seeded hot shard per layer.

The default targets, `PROBE_REGULAR_MEAN_RATIO` in `bench_driver.py`, were calibrated from a Qwen3.5-397B-A17B GRPO
production run (the per-layer mean over its 65 regular micro-batches). For another model or another phase of
training, pass your own table with `ROUTING_TARGETS="r0,r1,..."` (one ratio per MoE layer, in the same probe order),
or stretch the default with `ROUTING_SCALE` (targets become `1 + scale * (target - 1)`). `ROUTING=uniform` draws
uniform top-k instead. `EP_SIZE` (default 8) sets the expert-shard count the calibration assumes.

The step's last micro-batch gets the same treatment plus `FINAL_PAD_FRACTION` (default 0.25) of its rows entirely
padded, since the e2e run's last micro-batch of a step is a partially filled tail, and its `fwd_bwd` is called with
`apply_optimizer=True` as in `DistributedRLEngine.train_step`. `PAYLOAD_SEED` seeds the regular payload and seed + 1
the final one. Before the steps the driver logs one `PAYLOAD_ROUTING` line per payload with the offline per-layer
ratio of the replayed tokens; the runtime ratio, which includes the padding's router-chosen experts, is in the
engine's `MOE_DROP_AUDIT` lines with the MaxText flag `log_moe_drops=true`. Building the two payloads at the 397B
shapes took about two minutes each of host time in our runs, before the first step. `PAYLOAD=warmup` restores the warm-up payload.

## Numerics mode

With the e2e orchestrator flags, the GRPO loss gates sequences on the gap between the rollout log-probs and the
trainer's (`--seq_logprob_error_threshold`, `--truncated_importance_sampling_*`). The payload's rollout log-probs are
a synthetic constant, far from what a randomly initialized trainer assigns, so both gates drop every sequence, the
loss denominator is 0 and the gradient is exactly 0 (with `step_skipped` 0). This does not affect timing, but it hides
numerics differences. `LOGP_GATES=off` strips the gate flags from the orchestrator argv, so every sequence stays in
the loss, and `ADVANTAGES=random` gives each segment a seeded N(0, 1) advantage while keeping token ids, segment cuts
and routing byte-identical. With these two, the gradient norm of two variants (e.g. a MoE padding flag on and off) is
a meaningful comparison; the loss value itself is a function of the payload alone when the ratio is 1.

`check_numerics_cpu.py --launch-render launch-dryrun.yaml` reproduces this on CPU with Tunix's GRPO loss and a
one-parameter stand-in for the model. The stand-in replaces `compute_per_token_logps`, so it is tied to that
function's signature in the Tunix version it runs against.

## Step boundary: STAGE_BEFORE_UPDATE

`DRIVER_ENV="STAGE_BEFORE_UPDATE=on"` (default `off`; any other value raises) times the step boundary of a trainer that
stages the next step's first micro-batch before the previous step's update, as Tunix's `TrainerWorker.stage_fwd_bwd`
does before `apply_deferred_update` (google/tunix#2877). After step k's last `fwd_bwd` and before its `update`, the
driver hands step k+1's first micro-batch to the engine's `prepare_fwd_bwd_input`, which maps it to loss inputs and
places it on `fwd_bwd`'s input shardings while the device still computes step k's last micro-batch. The driver then
calls `wait_for_device` (it returns how long that micro-batch still ran) and runs the update, the drain and the
metrics read as with the switch off; step k+1's first `fwd_bwd` takes the prepared input. The trainer sees the same
micro-batches, in the same order, with the same `apply_optimizer` flags and one `update` per step. Each staged
boundary logs, after step k's `completed step` line:

```
STAGE_BEFORE_UPDATE step=<k> staged=bench_<k+1>_0 stage_s=<host seconds> device_wait_s=<seconds>
```

A `device_wait_s` well above `stage_s` means the copy was issued while the device was still busy.
`prepare_fwd_bwd_input` and `wait_for_device` come with AI-Hypercomputer/maxtext#5718. With an engine that lacks them
the driver raises a `RuntimeError` naming them right after the trainer worker initializes, before the compile; the
switch off needs nothing new. A trainer worker that runs the update inside the step's last `fwd_bwd` (Tunix's fused
update, google/tunix#2719) leaves no update to stage before, so with it the driver raises at the first boundary.

Why it shows on this bench: with replayed routing a micro-batch's input is several GB (`routed_experts` is
`[rows, seq, layers, top_k]`), and with the switch off step k+1's first micro-batch is copied to the device after the
drain, while the device is idle. In one same-pod A/B on the 397B recipe (128 TPU v7x chips, `MICRO_STEPS=9`, steps
1-4), the step's second `fwd_bwd` call took 7.5-7.8 s longer than a regular one with the switch off and about the same
as a regular one with it on, and the step went from 107.7 s to 100.0 s; the regular micro-batches and the step's tail
(last micro-batch, update and drain) did not change.

## Ahead-of-time memory check

```bash
LAUNCH_RENDER=launch-dryrun.yaml AOT_IMAGE=<image> [TRAINER_SLICE=4x8x8] bash aot_bench.sh
python3 aot_summarize.py logs/aot-bench-<time>.log
```

`aot_bench.sh` takes the trainer's MaxText flags, `--xla_*` libtpu flags and mesh from the e2e render
(`render_bench.py --print-aot-env`, with the same variant knobs as `render_bench.sh`) and runs `aot_trainer.py` in the
image on a CPU host against a virtual topology. `aot_trainer.py` builds the MaxText argv as the Tunix trainer node does
and calls `training_engine/maxtext_engine_compile.py` with the GRPO loss, packed micro-batches and router replay; its
recipe constants (sequence lengths, optimizer, GRPO config) are the 397B recipe's and need editing for another
recipe. The report gives each kernel's temporaries, which is what XLA checks against `--xla_tpu_max_hbm_size_mib`.
A static compile cannot see runtime fragmentation, staging copies made by weight sync, or kernels that only compile
at runtime, so treat the result as a screen before spending a slice.

## Tests

```bash
python3 -m pytest src/maxtext/experimental/rl_trainer_bench/tests
```

The tests run on CPU with PyYAML and NumPy only: the renderer on a hand-written DRY_RUN fixture
(`tests/testdata/launch_dryrun_fixture.yaml`), and the payload builder and metric helpers on a small stand-in payload.
`tests/stage_before_update_test.py` runs the driver's `main()` against stand-ins for Pathways, JAX, Tunix and the
training engine: with `STAGE_BEFORE_UPDATE` off the engine sees the original loop's calls, with it on the same
micro-batches in the same order with one update per step, and an engine without the two methods is refused.
They live next to the code, so the repository's default `testpaths = tests` does not collect them; run them by path.
`check_payload_routing.py` and `check_numerics_cpu.py` are the heavier checks at the production shapes and need a
matching Tunix.

## Caveats

The weights are randomly initialized, so router decisions that the payload does not replay (the padding's, and those
of fully padded rows) follow random weights rather than trained ones. The payload is synthetic but calibrated: its
replayed routing follows a measured per-layer load profile, while its segment lengths, prompt and completion split and
padding are chosen by the knobs above. A variant whose effect depends on the load imbalance of real rollouts needs a
confirming e2e run.

The profiler counts `fwd_bwd` calls, so its window is in micro-steps; a window that spans an optimizer step boundary
includes the `update`. After each `update` the bench drains the step and reads the metrics on the host before
dispatching the next step, so a trace across the boundary holds a host-side gap that the e2e trainer does not have.

Weight sync to the rollout engines and checkpoint saves do not run, so their memory and time are not part of the bench.
The AOT check is static (see above) and compiles some kernels with CPU fallbacks, so its peak is an estimate best used
to compare variants with each other.
