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

# MPMD pipeline parallelism on Shardy MPMD and Pathways (experimental)

Pipeline parallelism in which every stage is its own XLA program on its own mesh. One training step, traced from
MaxText's embedding, decoder layers and output head, is partitioned by
[Shardy MPMD](https://github.com/openxla/shardy/tree/main/shardy/integrations/python/jax/mpmd) (OSS PartIR:MPMD) into
an IFRT IR program: a fragment per stage, microbatch and pass, plus the cross-mesh transfers, in the order of the
pipeline schedule (1F1B, GPipe, ZB-H1, ZB-H2, ...). Pathways runs it, one execute per step. The program compiles only
through Pathways' IFRT proxy; a CPU host lowers it without a TPU. **Status: experimental.**

## Environment

| Component | Tested with | Note |
|---|---|---|
| MaxText | `c034bc02f` (2026-11-25) | the 2026-11 results below were measured on `4b5b24c92` (2026-11-24); the October table states its own |
| Python | 3.12 | |
| jax, jaxlib | 0.12.0.1 | pin both: the Shardy MPMD frontend must match jaxlib (next table) |
| libtpu | 0.0.51, MaxText's minimum, from the install below; 0.0.50.1 also run, with the same losses | not loaded by a Pathways client (`JAX_PLATFORMS=proxy`) or on CPU: the Pathways servers compile |
| pathwaysutils | 0.1.14 | MaxText's minimum, from the install below |
| shardy-mpmd (Shardy MPMD's frontend) | Shardy `38ab99f4` | on neither PyPI nor the jax wheels; installed below |
| Pathways server and proxy images | `20261117-jax_0.12.0.1` | builds from 2026-11-18 on reject jaxlib 0.12.0.1's IFRT IR ("Failed to parse IFRT IR module string"); plain `jax.jit` runs on them too |
| GKE launcher | xpk | or a gcluster built from cluster-toolkit's `develop`: see [GKE](#pathways-on-gke) |

| jaxlib | Shardy commit (`sdy_mpmd_versions.SHARDY_COMMITS`) | Run |
|---|---|---|
| 0.12.0.1 | `38ab99f4d252c59d491899a694fe8cbadfeccec3` | everything here from November 2026 |
| 0.12.1 | `57fffc23dfdf139b80013ad681701cee59c22f96` | not run |
| 0.11.1 | `2832731619ffb4bcc718faa4aa8054214a68c969` | the October results' implementation; MaxText now needs jax 0.12.0.1 or newer |

Install MaxText from source as its [install guide](../../../../../docs/install_maxtext.md#from-source) describes, pin
jax, and add the frontend from the installed jaxlib's Shardy commit, without dependencies (`sdy_mpmd_compat.py` warns
when pip recorded another commit):

```bash
uv pip install -e .[tpu] --resolution=lowest && install_tpu_pre_train_extra_deps
uv pip install jax==0.12.0.1 jaxlib==0.12.0.1
uv pip install --no-deps "shardy-mpmd @ git+https://github.com/openxla/shardy@38ab99f4d252c59d491899a694fe8cbadfeccec3#subdirectory=shardy/integrations/python/jax/mpmd"
```

### Pathways in docker on a TPU VM

[`local_pathways.sh`](local_pathways.sh) runs the resource manager, one worker (the VM's chips) and the IFRT proxy from
the public images `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/{server,proxy_server}`, as a GKE Pathways job's pods
run them. It needs docker and a `gs://` scratch location the VM's service account can write. `--instance_type` is
`tpu7x:2x2x1` on a TPU7x VM with 4 chips (`tpu7x-8`), `tpuv6e:2x2` on a v6e-4; `--tag` picks the images;
`--xla_flags_file` (one `--key=value` a line) gives XLA / libtpu flags to all three servers. The worker holds the TPU
until `down`; it exits whenever a client disconnects and restarts, so each new client waits about 10 s.

### Pathways on GKE

Create a Pathways cluster and install Cluster Toolkit (`gcluster`) as MaxText's
[Pathways guide](../../../../../docs/run_maxtext/run_maxtext_via_pathways.md) describes, and build an image with this
MaxText tree ([Build MaxText](../../../../../docs/tutorials/build_maxtext.md)) and the `shardy-mpmd` install above.
[`launch.py`](launch.py) checks the decoder block, the stages and slices, the workload name and the microbatch split,
and prints `gcluster job config set` and `gcluster job submit --pathways`, whose head pod runs the entry point
(`--submit` runs them; `--launcher=xpk` prints `xpk workload create-pathways`).

- **gcluster v1.108.0 and v1.109.0 rerun a finished Pathways job and end it `Failed`, and no release has the fix yet**
  (v1.109.0 is the newest on 2026-11-26). They give the workers `restartPolicy: Never`, and a worker exits when its
  client disconnects: when the entry point ends, even successfully, the JobSet restarts and runs it a second time. The
  fix, cluster-toolkit [PR #6739](https://github.com/GoogleCloudPlatform/cluster-toolkit/pull/6739), was merged into
  `develop` on 2026-11-25: use a gcluster built from `develop`, or `--launcher=xpk`. The launcher logs this warning.
- `--xla_flags_file` and `--extra_xla_flags=key=value` go to the proxy, which compiles the TPU programs, and to the
  resource manager and the workers (gcluster's `--pathways-{proxy,server,worker}-args`, xpk's
  `--custom-pathways-{proxy-server,server,worker}-args`).
- `--pathways_tag` (default `20261117-jax_0.12.0.1`) must match the image's jax. `--module=parity` or
  `--module=train_compile` run the other entry points; arguments after `--` go to the entry point.

## Quick start

```bash
# CPU only: lower one step on 8 fake devices and write /tmp/pp4.{stablehlo,mpmd,ifrt}.mlir.
JAX_PLATFORMS=cpu python3 -m maxtext.experimental.mpmd.partir.train_compile --pp=4 --num_microbatches=4 \
  --per_device_batch_size=4 --model_name=deepseek3-tiny --pipeline_schedule=1F1B --ir_dump_path=/tmp/pp4

# A TPU VM: train, check MPMD against one SPMD program, and log each compiled fragment's memory and cost.
bash src/maxtext/experimental/mpmd/partir/local_pathways.sh up --gcs_scratch_location=gs://<bucket>/pathways \
  --instance_type=tpu7x:2x2x1
export JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
python3 -m maxtext.experimental.mpmd.partir.train --pp=4 --num_microbatches=4 --per_device_batch_size=4 \
  --model_name=simple
python3 -m maxtext.experimental.mpmd.partir.parity --pp=4 --num_microbatches=4 --per_device_batch_size=4 \
  --model_name=simple --steps=3 --maxtext_overrides=dtype=float32
python3 -m maxtext.experimental.mpmd.partir.train_compile --compile_topology=local --pp=4 --num_microbatches=4 \
  --per_device_batch_size=4 --model_name=deepseek3-tiny
bash src/maxtext/experimental/mpmd/partir/local_pathways.sh down

# GKE, 2 x tpu7x-4x4x4: the llama3.1-8b PP=4 run of the results. Prints the xpk command; --submit runs it.
python3 -m maxtext.experimental.mpmd.partir.launch --cluster=<cluster> --project=<project> --zone=<zone> \
  --docker_image=<image> --pathways_gcs_location=gs://<bucket>/pathways --device_type=tpu7x-4x4x4 --num_slices=2 \
  --workload=mpmd-llama-pp4 --pp=4 --num_microbatches=4 --per_device_batch_size=4 --model_name=llama3.1-8b \
  --max_target_length=1024 --learning_rate=1e-5 --warmup_steps=10 --steps=20 --launcher=xpk
# The same model and tokens a step on MaxText's own SPMD pipeline: prints its settings and trainer command.
python3 -m maxtext.experimental.mpmd.partir.spmd_baseline --device_type=tpu7x-4x4x4 --num_slices=2 --pp=4 \
  --num_microbatches=4 --per_device_batch_size=4 --model_name=llama3.1-8b --max_target_length=1024 --learning_rate=1e-5
```

`train` logs one `MPMD_RESULT {json}` line (`--result_json` writes it), stops with exit 1 at a NaN loss, and traces
with `--enable_profile --profile_dir=gs://...`. Each stage's devices run FSDP unless `--fsdp` and `--expert` say
otherwise. `spmd_baseline` also prints pure FSDP (`--pure_fsdp`) and, with `--cluster`, the job. Decoder blocks:
`simple_mlp` (`--model_name=simple`), `llama2`, `mistral`, `mixtral`, `qwen3`, `qwen3_moe`, `deepseek` (e.g.
llama3.1-8b, mistral-7b, mixtral-8x7b, qwen3-8b, qwen3-30b-a3b, deepseek2-16b, deepseek3-tiny; qwen3-235b-a22b with
`--maxtext_overrides=load_balance_loss_weight=0`). `--num_layers` defaults to 8 for every model; named models keep
their dimensions, while `simple` and deepseek3-tiny take `--emb_dim`, `--mlp_dim` and `--vocab_size` (512, 2048,
16384).

## Schedules

`--pipeline_schedule` is `1F1B` by default, or `GPipe`, `GPipeBut1F1BForLastMesh`, or a zero-bubble schedule
([Qi et al.](https://arxiv.org/abs/2401.10241)): `ZB-H1`, or `ZB-H2` (`ZB-H2-ZERO`; `ZB-H2-HALF` and `ZB-H2-FULL`
assume a transfer takes half or all of a stage's compute, and start more forwards early). For the zero-bubble
schedules the port sets Shardy's `mpmd_split_bwd_fragments`: each backward fragment that sends an activation gradient
upstream becomes B_a, which computes that gradient, and B_w, which computes the weight gradients and runs where the
stage would otherwise wait. The first stage's backward stays whole (it sends nothing upstream); the last stage runs
B_a merged with its forward, as 1F1B's last stage runs the whole backward; MaxText's MoE layers compute the routed
experts' weight gradients in B_a. ZB-H1 runs as many forwards ahead as 1F1B and also holds B_w's inputs until B_w
runs; ZB-H2 runs up to 2(P - s) - 1 forwards ahead on stage s (1F1B: P - s) to also fill the warm-up, 1.75 times
1F1B's activations on the first of 4 stages. `train_compile --ir_dump_path` shows the fragments; `pipeline_test.py`
checks the split and how many forwards each zero-bubble schedule runs ahead.

## Benchmark results

jax 0.12.0.1, Shardy `38ab99f4`, Pathways `20261117-jax_0.12.0.1`, synthetic data, 1F1B, 4 microbatches,
`--per_device_batch_size=4`, FSDP over each stage; median step time of the timed steps; measured on 2026-11-25. The
quick-start commands reran on 2026-11-26, on MaxText `4b5b24c92` and `c034bc02f`, with the same losses and parity
figures, bit for bit.

**One TPU7x VM** (tpu7x-8: 4 chips, 8 devices; Pathways in docker), PP=4 x FSDP=2, 2 warm-up steps:

| Model (8 layers) | Sequence | Timed steps | Step time | TFLOP/s/device | Loss, first to last step |
|---|---|---|---|---|---|
| `simple` | 512 | 8 | 7.11 ms | (toy) | 10.217 to 2.772 |
| deepseek3-tiny | 512 | 8 | 17.72 ms | (toy) | 10.200 to 2.919 |
| deepseek3-tiny with mHC (`mhc_expansion_rate=4`) | 512 | 8 | 36.88 ms | (toy) | 10.190 to 2.718 |
| llama3.1-8b, `--learning_rate=1e-5` | 1024 | 10 | 70.19 ms | 201.7 | 12.27 to 0.27 |

Through the proxy, tiny models' step times follow the host's load: `simple` and deepseek3-tiny reran at 8.07 and
15.06 ms on 2026-11-26, and the mHC program took 45.0 ms at a load average of 65 to 135.

**2 x tpu7x-4x4x4** (256 devices; one xpk-rendered Pathways JobSet whose head ran `train.run_training` for the four
runs in one client session, not through `launch.py`), 10 warm-up and 20 timed steps, no stage straddling a slice:

| Model (8 layers) | Stages | Devices a stage | Tokens a step | Step time | TFLOP/s/device |
|---|---|---|---|---|---|
| llama3.1-8b, sequence 1024, `--learning_rate=1e-5` | PP=4, 2 a slice | 64 | 262,144 | 159.0 ms | 89.0 |
| llama3.1-8b, sequence 1024, `--learning_rate=1e-5` | PP=2, 1 a slice | 128 | 524,288 | 231.8 ms | 122.1 |
| deepseek3-tiny, sequence 512 | PP=4, 2 a slice | 64 | 131,072 | 49.9 ms | (toy) |
| deepseek3-tiny, sequence 512 | PP=2, 1 a slice | 128 | 262,144 | 44.3 ms | (toy) |

deepseek3-tiny took about 0.3 s a step until it settled by its 14th step. No MaxText SPMD step time was measured.

**Schedules** (2026-11-26): llama3.1-8b cut to 8 layers, sequence 1024, `--learning_rate=1e-5`, 8 microbatches,
`--per_device_batch_size=8`; median step time (TFLOP/s/device). On the TPU7x VM PP=4 x FSDP=2 with 3 warm-up and 20
timed steps, on 2 x tpu7x-4x4x4 as above:

| Schedule | TPU7x VM | the VM, layers 2,2,3,1 | 2 x tpu7x-4x4x4, PP=4 | PP=4, layers 2,2,3,1 | PP=2 |
|---|---|---|---|---|---|
| 1F1B | 105.6 ms (268) | 103.4 ms (274) | 243.7 ms (116) | 248.9 ms (114) | 402.7 ms (141) |
| GPipe | 124.2 ms (228) | 108.0 ms (262) | not run | not run | not run |
| ZB-H1 | 96.6 ms (293) | 88.6 ms (320) | 217.4 ms (130) | 195.9 ms (145) | 326.3 ms (174) |
| ZB-H2 | 97.2 ms (291) | 82.8 ms (342) | 212.2 ms (133) | 194.6 ms (146) | 319.4 ms (177) |
| ZB-H2-HALF | 107.2 ms (264) | 94.0 ms (301) | 249.2 ms (114) | not run | 330.5 ms (171) |
| ZB-H2-FULL | 105.7 ms (268) | 91.9 ms (308) | not run | not run | not run |

All schedules give llama the same losses (on the VM, 12.268 to 0.0235 over 23 steps). ZB-H1 and ZB-H2 save 8 to 13%
of 1F1B's step at PP=4 and 19 to 21% at PP=2. The output head, about 2.3 layers of FLOPs, makes the last stage the
slowest; with 2,2,3,1 layers (`--stage_layer_counts`) the stages are closer, and the saving grows to 14 to 22%. In a
trace of the VM, ZB-H1's B_a halves the middle stages' backward (2.0 against 4.1 ms), so each gradient reaches the
stage upstream sooner, and the last stage runs its last four B_w (14.5 ms) while the gradient drains back through the
other stages. On the VM `simple` and deepseek3-tiny run fragments of 0.1 to 0.3 ms, so their steps are bound by
dispatch and follow the host's load: the zero-bubble schedules ranged from 24% faster to 55% slower across sessions.
On 2 x tpu7x-4x4x4, deepseek3-tiny at PP=4 took 72.0 ms with 1F1B, 66.7 with ZB-H1, 55.6 with ZB-H2 and 54.6 with
ZB-H2-HALF.

**Parity** (`parity.py`, float32, PP=4 x FSDP=2, 3 steps, on the TPU7x VM; the second row on 2026-11-26): MPMD
against one unsharded SPMD program from the same parameters and data. Each cell is the worst step or tensor / its
bound, max(tolerance, 4 x the noise floor), the floor being the SPMD program with another microbatch split:

| Model and settings | Loss | Step-1 gradients (rel. L2) | Parameters (rel. L2) |
|---|---|---|---|
| `simple` (the quick start's command) | 1.7e-5 / 1e-4 | 7.4e-4 / 1.0e-2 | 1.3e-3 / 1.0e-2 |
| `simple`, `matmul_precision=highest`, `JAX_DEFAULT_MATMUL_PRECISION=highest` | 9.3e-8 / 1e-4 | 3.2e-7 / 1e-2 | 2.9e-6 / 1e-2 |
| deepseek3-tiny, every expert routed, `matmul_precision=highest` | 1.4e-4 / 1.1e-3 | 4.2e-3 / 1.1e-2 | 6.5e-3 / 2.4e-2 |
| the same with mHC (`mhc_expansion_rate=4`), floor at 8 microbatches | 3.1e-5 / 1.1e-4 | 3.3e-2 / 4.7e-2 | 8.7e-3 / 3.2e-2 |
| the second row with 8 microbatches, ZB-H1 and each ZB-H2 (2026-11-26) | 1.1e-7 / 1e-4 | 3.5e-7 / 1e-2 | 2.6e-6 / 1e-2 |
| the third row with 8 microbatches, ZB-H1 and ZB-H2, floor at 16 (2026-11-26) | 1.4e-4 / 5.5e-4 | 5.2e-3 / 1.4e-2 | 5.7e-3 / 2.2e-2 |

All passed. The zero-bubble schedules give the same numbers as each other, and as 1F1B at the same settings to two
digits; the last row's default floor, 4 unsharded microbatches of 2,048 tokens, halts the TPU in MaxText's GMM kernel
(as 2,048-token microbatches with mHC do). With top-8 routing, deepseek3-tiny fails on its router gradients only
(10.5% against 1%): near-tied expert scores round to another expert. mHC's mixing scalars have near-zero step-1
gradients that move with the microbatch split, so they pass only against a wide floor.

**October 2026, the earlier, unpublished implementation** this package was ported from (jax 0.11.1, Shardy `2832731`,
Pathways `20261013-jax_0.11.1`, an October MaxText; v5p, MFU against 459 TFLOP/s; not repeated with this package):

| Hardware | Configuration (1F1B unless noted) | Step time | TFLOP/s/device (MFU) |
|---|---|---|---|
| v5p-64 | llama3.1-8b (32 layers), PP=4 x FSDP=8, 8 microbatches of 8 x 2048 tokens; ZB-H1 | 1337 ms; 1113 ms | 142.9 (31.1%); 171.7 (37.4%) |
| v5p-32 | llama3.1-8b, PP=4 x FSDP=4, 8 microbatches of 4 x 2048: MPMD; MaxText's SPMD pipeline at the same tokens | 1351 ms; 1641 ms | 141 (30.8%); 116.4 |
| 2 x v5p-32 | the v5p-64 run with one stage boundary over DCN (one v5p-64: 1337 ms) | 1349 ms (+0.9%) | 141.6 (30.8%) |
| 4 x v5p-16 | the v5p-64 run with one stage a slice, 3 boundaries over DCN (one v5p-64: 1339 ms) | 1388 ms (+3.7%) | 137.6 (30.0%) |
| v5p-128 | llama3.1-70b (80 layers), PP=8 x FSDP=8, `--remat`, 16 microbatches of 8 x 2048 | 12.69 s | 137.2 (29.9%) |
| v5p-64 | mixtral-8x7b, PP=4 x FSDP=8, `--remat`; deepseek2-16b, 6/7/7/7 layers, `--expert=8 --ragged_buffer_factor=2` | 3258 ms; 996 ms | 98.2 (21.4%); 64.0 |

## Tests

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 -m pytest tests/unit/experimental/mpmd/partir
```

With 4 or 8 devices: 129 passed, 1 skipped with shardy-mpmd; 116 passed, 14 skipped without it (the lowering, merge-rule
and shim tests need the frontend); tests that need 4 devices skip below that. MaxText CI's `cpu-unit` job runs them with
4 devices, without shardy-mpmd and the marker `cpu_only and not post_training and not integration_test`: 116 passed, 13
skipped. `pathways_train_test.py` (`tpu_only`) trains on Pathways and skips elsewhere: run it with shardy-mpmd and
`JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000` against `local_pathways.sh`.

## Known limitations

- **The MPMD step trains on the token cross-entropy alone**, with a constant-learning-rate `optax.adamw`
  (`--learning_rate`, default 1e-3; llama3.1-8b needs 1e-5) on synthetic data: not MaxText's optimizer, schedule or
  input pipeline. Settings that add to MaxText's model or loss are refused, each naming the override that turns it off:
  an MTP block, a sinusoidal position embedding, the MoE load-balance loss, the z-loss, routing-bias updates and the
  indexer loss.
- **Refused rather than trained as another model:** other decoder blocks, m3 models, tied embeddings (pass
  `--maxtext_overrides=logits_via_embedding=false`), learned position embeddings, hyper-connections outside `deepseek`,
  quantization, `shard_mode=explicit`, `prefetch_fsdp_weights`, the output-head variants `config._check_buildable`
  lists, and remat policies that offload to the host. YaRN RoPE (DeepSeek, Kimi) becomes real RoPE, in `spmd_baseline`
  too: Shardy MPMD asserts on its `complex64` table.
- **OSS Shardy MPMD lacks** five of its ten boolean partitioning options (jaxlib's driver ignores them), tying an
  output's layout to an input's (the state's shardings are pinned on both sides of `mpmd.jit`), `lowered.ifrt_ir_module`
  (`pipeline.pin_ifrt_transfers_to_producers` edits `lowered.partitioning_result`), AOT compilation for a TPU topology,
  serialized programs and custom schedule rules. The wrap-around and circular schedules are refused: each stage holds
  one contiguous chunk of layers.
- **Pathways:** `ifrt_ir_program_memory_stats` is unimplemented through the proxy, so `train_compile` logs each
  fragment's memory instead of the program's; a worker's out-of-memory reaches the client as a lost connection; traces
  need a `gs://` directory and about 1 s after the trace starts (`--profile_delay_s`).
- **MoE:** expert gradients under FSDP are reduce-scattered once a step by patching the private
  `jax._src.shard_map._unmentioned2` (`models_test.py` fails if jax renames it); a large MoE needs
  `--ragged_buffer_factor=2`, or its routed-token buffers are sized for the worst case.
- `launch.py`'s commands have not run on a cluster; shardy-mpmd is not in MaxText's requirements.
