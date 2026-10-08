# olmoe3-3p5b on one TPU v4-128: config, flags, kernels and run script

Model: `src/maxtext/configs/models/olmoe3-3p5b.yml` (OLMo 3.5 small: 62.9B total, 3.48B active, 30 layers =
24 Kimi Delta Attention + 6 full attention, 1 dense + 29 latent-MoE layers, 512 experts top-16, latent 768, one
shared expert, EMo document-pool routing on). Hardware: one v4-128 slice (4x4x4, 64 chips, 16 hosts); v4 is a
megacore chip, so each chip is one JAX device with 32 GiB HBM (31.75 GiB usable with MEGACORE_DENSE) and 275 TF/s
bf16.

## Results

Median of steps 10-19 of a 20-step run, synthetic data. MFU uses the FLOPs counted from the real parameter shapes
(`scripts/olmoe3_flops_check.py`: 85.17 TF per chip per step at 4k, 172.81 at 8k); MaxText's logged TF/s is within
1.3% of that; the reference report's count (91.14 / 184.75) is 7% higher (see "FLOP accounting" in `olmoe3-3p5b-v4-best-configs.md`).

| config | seq | step s | MFU | MaxText log MFU | tok/s/chip | run, arm |
|---|---|---|---|---|---|---|
| **best, 8k** | 8192 | **3.095** (repeats 3.103, 3.106, 3.113) | **20.3%** | 20.0% | 2,647 | o3v4r081817 `r0_s8k_k3_ng_sr` |
| best, 4k | 4096 | 1.520 (repeat 1.523) | 20.4% | 20.1% | 2,695 | o3v4o081736 `o0_s4k_m0_ng_sr` |
| 8k, `F32_KDA_STATE=1` (strictly loss-neutral) | 8192 | 3.167 | 19.8% | 19.6% | 2,587 | o3v4w082116 `q1_s8k_ours_f32` |
| 4k, `F32_KDA_STATE=1` | 4096 | 1.558 | 19.9% | 19.6% | 2,629 | o3v4w082116 `q0_s4k_ours_f32` |
| 8k, `SAFE_BUFFER=1` (ragged buffer 1.25) | 8192 | 3.223 | 19.5% | | 2,542 | o3v4s081838 `s1_s8k_k3_rbf125` |
| 4k, `SAFE_BUFFER=1` | 4096 | 1.581 | 19.6% | | 2,591 | o3v4s081838 `s0_s4k_o0_rbf125` |

Every run's source tree, arms, JobSet, logs and profiles: `gs://agagik-us/olmo35/v4/runs/<run>/`.

## Batch size

| item | value |
|---|---|
| `per_device_batch_size` | **1** (one sequence per chip; v4 has one device per chip) |
| global batch | 64 sequences per step |
| tokens per step | 262,144 at 4k, 524,288 at 8k |
| gradient accumulation | none |
| `per_device_batch_size=2` | slower per token or out of memory in the earlier (scanned-layer) configs, z-series; not re-measured since unrolling freed memory |

## Run script

`src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4_128.sh` runs the best config on every worker of
the slice. Its flags equal the measured best arms (checked flag by flag against `benchmarks/olmoe3_v4/best_arms.txt`;
the only extra, `decoder_layer_input=device`, is the base.yml default).

    SEQ_LEN=8192 bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4_128.sh       # best 8k
    SEQ_LEN=4096 bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4_128.sh       # best 4k
    SEQ_LEN=8192 F32_KDA_STATE=1 bash .../run_olmoe3_3p5b_v4_128.sh     # strict precision
    SEQ_LEN=8192 SAFE_BUFFER=1 PROFILE=1 bash .../run_olmoe3_3p5b_v4_128.sh
    DRY_RUN=1 SEQ_LEN=8192 bash .../run_olmoe3_3p5b_v4_128.sh           # print env and flags only

Knobs: `SEQ_LEN` (4096 or 8192), `F32_KDA_STATE`, `SAFE_BUFFER`, `PROFILE`, `STEPS`, `RUN_NAME`, `OUTPUT_DIR`,
`DATASET_TYPE`; extra MaxText flags can follow on the command line.

On the GKE cluster used here (`v4-128-bodaborg-us-central2-b`, project `cloud-tpu-multipod-dev`) the measured runs
went through `scripts/olmoe3_v4_launch.sh <letter> benchmarks/olmoe3_v4/best_arms.txt`, which renders a JobSet
(16 pods, image `gcr.io/cloud-tpu-multipod-dev/agagik-olmoe3:kdaj24`, this source tree first on `PYTHONPATH`)
via `scripts/olmo35_xpk_4x8x8.sh`; `scripts/olmoe3_v4_harvest.sh <run>` copies logs, results and profiles off the
leader pod.

## Environment and libtpu flags

| setting | 4k | 8k | why |
|---|---|---|---|
| `TPU_MEGACORE=MEGACORE_DENSE` | yes | yes | +1 GiB HBM; at 8k it is worth 521 ms |
| `--xla_tpu_enable_async_ragged_all_to_all=true` | yes | yes | one token chunk's all-to-all overlaps another chunk's expert GEMMs (-107 ms at 4k, -494 ms at 8k) |
| `--xla_enable_async_all_gather=false` | yes | **no** | with MEGACORE_DENSE XLA turns the weight all-gathers async, +50 ms at 4k; at 8k they must stay async (+404 ms if forced sync) |
| `--xla_tpu_scoped_vmem_limit_kib=16384` | yes | yes | v4 has 16 MiB VMEM |
| `--xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_spmd_rng_bit_generator_unsafe=true` | yes | yes | |

## MaxText flags

| group | flags | notes |
|---|---|---|
| model, precision | `model_name=olmoe3-3p5b override_model_config=True dtype=bfloat16 weight_dtype=float32` | f32 master weights and Adam state; bf16 weights do not train (loss 10.32 vs 8.49 at step 19) |
| batch | `per_device_batch_size=1 max_target_length=4096 \| 8192` | |
| layout | `ici_expert_parallelism=64 ici_fsdp_parallelism=1 shard_exp_on_fsdp=False` | 8 experts per chip; no expert-weight all-gathers or FSDP gradient reductions |
| layers, remat | `scan_layers=False olmoe3_per_layer_remat=True remat_policy=custom` | unrolled: -35 / -90 ms, and compile-time temporaries fall from 27.5 to 11.5 GB (4k), which pays for the saves below |
| saved activations | `moe_routing=device moe_x_sorted=device moe_combine=device decoder_layer_input=device`; 4k adds `moe_mlpwi_0=device`, 8k adds `kda_wy=device` | the bwd reruns neither the dispatch nor the combine all-to-all; 4k also skips the gate GMM recompute |
| expert GEMMs | `sparse_matmul=True megablox=True use_tokamax_gmm=False use_gmm_v2=False capacity_factor=-1` | tokamax kernels need TPU generation >= 6 |
| megablox tiles (m, k, n) | wi fwd 512, 768, 896; wi dlhs 512, 1792, 384; wi drhs 512, 768, 896; wo fwd 512, 1792, 384; wo dlhs 512, 768, 896; wo drhs 512, 896, 384 (`wi_tile_*`, `wo_tile_*`) | two n-tiles per GMM: megablox parallelizes only the n axis, so one n-tile runs on one of v4's two TensorCores (-201 ms at 4k) |
| routing | `moe_lean_routing=True moe_topk_pallas=True emo_threshold_by_bisection=True ragged_buffer_factor=1.125` | `SAFE_BUFFER=1` sets 1.25 |
| dispatch | `moe_a2a_expert_major=True`; `moe_a2a_token_chunks=2` (4k) / `4` (8k) | routing runs once over the whole sequence, then the all-to-all MoE runs per chunk; keeps every gather under the 128 MiB CMEM |
| KDA | `use_tokamax_kda=False kda_chunked_impl=subblock gdn_chunk_size=128 kda_fused_input_proj=True gdn_state_dtype=bfloat16 kda_conv_in_compute_dtype=True` | `F32_KDA_STATE=1` sets `gdn_state_dtype=float32 kda_conv_in_compute_dtype=False` |
| LM head | 4k `num_vocab_tiling=1`; 8k `num_vocab_tiling=4 vocab_tiling_ag_once=True` | |
| metrics, benchmark | `norm_metrics=grad synthetic_data_reuse_batch=True` | logging and synthetic-data only; training is unchanged (3.103 s at 8k without them) |

## Kernels

No custom kernel is needed for these numbers. The Pallas kernels are MaxText's and JAX's existing ones; the new
work is at the JAX level (layout, scheduling, saves, exact rewrites).

| part (8k ms per step, profile o2) | kernel |
|---|---|
| routed experts, 29 layers x 3 GEMMs fwd + bwd (565) | **megablox** Pallas grouped matmul (`kernels/megablox`, unchanged), with the megacore tile shapes above |
| KDA, 24 layers (507 elementwise + 326 matmul) | **no Pallas**: pure-JAX sub-block chunked delta rule (`_delta_rule_chunked_subblock` in `models/olmoe3.py`) lowered by XLA; 16-row sub-blocks keep the intra-chunk decay on the MXU without overflow, a block-doubling inverse with a hand-written VJP, the state scan an XLA while loop. Tokamax KDA does not run on v4 |
| full attention, 6 layers (94) | **splash attention** (JAX Pallas, segmented fwd, dq, dkv) |
| router top-16 (6) | **Pallas top-k** (`kernels/topk.py`) |
| dispatch and combine (334 exposed) | XLA async ragged all-to-all, expert-major layout, 4 token chunks, paired custom VJP |
| dense projections (346), LM head (162), router / SwiGLU / combine (200) | XLA |

`kernels/kda_scan.py` holds a Pallas KDA state-scan kernel with a hand-written backward (2.6x faster in isolation)
behind `kda_pallas_scan`; it is off because the full model was 41-51 ms slower with it (operand relayouts around
the custom call).

## Code in this branch (all off by default unless noted)

| change | where | exact? |
|---|---|---|
| sub-block KDA (`kda_chunked_impl=subblock`, `kda_sub_block`), fused KDA input projection (`kda_fused_input_proj`), `kda_wy` remat name, Pallas state scan (`kda_pallas_scan`), fused-projection reduce-scatter (`kda_fused_proj_reduce_scatter`) | `models/olmoe3.py`, `kernels/kda_scan.py` | sub-block matches the exact scan to 2e-5 (f32), gradients to 1e-4 |
| `tokamax_kda_resets_in_gate` (packed-document resets folded into the gate for tokamax KDA) | `models/olmoe3.py` | matches the jnp reference |
| `moe_a2a_token_chunks` (route once, dispatch per chunk); EMo mask skipped for forced experts | `layers/moe.py`, `models/olmoe3.py` | yes (`test_a2a_token_chunks_match_unchunked`) |
| `moe_a2a_expert_major`, paired all-to-all VJP | `layers/moe.py` | yes (simulated and 4-device tests, bit-identical gradients) |
| dense top-k weights and group sizes, sort-free local permute (under `moe_lean_routing`) | `layers/moe.py` | yes (`test_block_transpose_indices_match_local_permute`) |
| `moe_x_sorted` / `moe_combine` remat names | `layers/moe.py`, `configs/types.py` | yes |
| `norm_metrics`, `synthetic_data_reuse_batch` | `trainers/pre_train/train.py`, `input_pipeline/synthetic_data_processing.py` | training unchanged |
| run script, launch / harvest / analysis scripts, arms files | `src/maxtext/trainers/pre_train/scripts/olmo/`, `scripts/olmoe3_v4_*`, `benchmarks/olmoe3_v4/` | |

Tests: `JAX_PLATFORMS=cpu pytest tests/unit/olmoe3_test.py tests/unit/moe_test.py` (99 passed, 56 TPU-only skipped).

## Cautions

| item | detail |
|---|---|
| `ragged_buffer_factor=1.125` | drops overflow tokens of experts hotter than 1.125x the mean; probe real data with `log_required_ragged_buffer_factor=true`, else use `SAFE_BUFFER=1` |
| bf16 KDA state and conv | a precision change: no loss difference beyond run-to-run spread in 20-step runs (4k 7.534 vs 7.525, 8k 8.336 vs 8.289 at step 19), but no long real-data A/B yet; `F32_KDA_STATE=1` is the strictly neutral setting |
| synthetic data | `synthetic_data_reuse_batch` applies only to `dataset_type=synthetic` |
| memory | the 8k config sits close to the limit: adding `moe_mlpwi_0` at 8k needs 31.86 G of 31.75 G |

More: `olmoe3-3p5b-v4-best-configs.md` (bests, what each lever is worth, FLOP accounting),
`olmoe3-3p5b-v4-runs.md` (every series).
