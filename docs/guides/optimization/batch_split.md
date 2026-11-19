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

# Generalized batch-split scheduling

Under Expert Parallelism (`ici_expert_parallelism > 1`) at small per-device batch sizes (such as
`per_device_batch_size=1` or `2`), each MoE layer issues two token collectives across the EP axes:
a token **dispatch** (`all-gather`) before the local expert grouped matmuls (GMMs) and a token
**combine** (`reduce-scatter`) after them. Even when FSDP weight collectives are hidden across
layers by `prefetch_fsdp_weights=True`, a single-stream layer schedule leaves these intra-layer EP
token collectives exposed on the critical path.

`enable_batch_split=True` (`enable_batch_split: false` by default) splits each device's batch (or
sequence when sharded across `context` at `per_device_batch_size=1`) into two microbatches (`mb0`
and `mb1`) and interleaves them across an 8-phase pipeline inside a custom `jax.custom_vjp` scan
(`src/maxtext/layers/batch_split.py` for homogeneous decoder stacks and
`src/maxtext/layers/batch_split_inhomogeneous.py` for inhomogeneous composite blocks). Each
microbatch's SparseCore EP token collective is overlapped with the other microbatch's TensorCore
attention or expert compute.

## 8-phase interleaved schedule

Wrapping `mb0`'s next-layer attention into the end of layer *k* gives `mb1`'s token combine a
TensorCore compute phase to hide under at every layer boundary:

| Phase | Microbatch 0 (`mb0`)                          | Microbatch 1 (`mb1`)           | Overlap target                                   |
| ----- | --------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| 1     | Pre-MoE norm + router metadata                | —                              | Fast TensorCore setup                            |
| 2     | **EP dispatch (`SC0`)**                       | **Attention / MLA (`TC`)**     | `mb0` dispatch hidden under `mb1` attention      |
| 3     | —                                             | Pre-MoE norm + router metadata | Fast TensorCore setup                            |
| 4     | **Routed experts (`TC`)**                     | **EP dispatch (`SC0`)**        | `mb1` dispatch hidden under `mb0` expert GMMs    |
| 5     | **EP combine (`SC0`)** + shared expert (`TC`) | **Routed experts (`TC`)**      | `mb0` combine hidden under `mb1` expert GMMs     |
| 6     | Post-MoE residual                             | —                              | Fast residual / norm                             |
| 7     | **Next layer (*k*+1) Attention / MLA (`TC`)** | **EP combine (`SC0`)**         | `mb1` combine hidden under `mb0` next-layer attn |
| 8     | —                                             | Post-MoE residual              | Fast residual / norm                             |

The backward pass executes the exact reverse 8-phase schedule, overlapping backward EP
`combine_bwd` (`all-gather`) and `dispatch_bwd` (`reduce-scatter`) on **SparseCore 0 (`SC0`)** with
backward attention (`splash_mha_dkv` / `dq`) and expert `dlhs` / `drhs` GMMs on TensorCore, while
cross-layer FSDP weight `all-gather` and deferred weight-gradient `reduce-scatter` run concurrently
on **SparseCore 1 (`SC1`)**.

### Key compiler and SparseCore mechanisms

| Mechanism                                              | What it does                                                                                                                                                                                                                                                               |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dual-SparseCore queue separation (`SC0` vs. `SC1`)** | Pins intra-layer EP token dispatch/combine collectives (including rowwise FP8 `combine_bwd` scales reshaped to 2D `(B, S)` so trailing dim `S >= 128`) to `SC0`, keeping `SC1` dedicated to cross-layer FSDP weight prefetch and gradient reductions.                      |
| **Compute-epoch post-scheduler overlap**               | Extends `fsdp_prefetch_overlap_transform` with `use_compute_epochs=True` when `batch_split` is active, ordering shared-queue SparseCore starts by `kCustomCall` compute phase with `done_slot` tie-breaking so collectives hoist ahead of SplashAttention and GMM kernels. |
| **In-place microbatch weight-gradient accumulation**   | Accumulates `mb0` and `mb1` routed-expert weight gradients in place inside `gmm_v2` (`moe_accumulate_wi_dlhs=True` / `moe_3d_dlhs_partial_sum=True`) and reduces replicated router gradients asynchronously on `SC1`.                                                      |
| **Selective `remat_policy` preservation**              | Traces per-phase forward/backward `ClosedJaxpr`s under the configured `remat_policy` (`full`, `minimal`, or `custom` device/offload checkpoints), stripping input-passthrough residuals so saved activations are not duplicated.                                           |

## Supported model architectures

1. **Homogeneous MoE stacks (`src/maxtext/layers/batch_split.py`):**
   - **DeepSeek-V3 / DeepSeek-V3.2** (`deepseek3-671b`, `deepseek3.2-671b` with MLA, optional DSA
     `use_indexer=True` when `indexer_loss_scaling_factor == 0.0`, and optional Multi-Token
     Prediction `mtp_num_layers >= 1`)
   - **Kimi-K2** (`kimi-k2-1t`)
   - **Qwen3 MoE** (`qwen3-235b-a22b`, `qwen3-30b-a3b`, with or without shared experts)
   - **Mixtral** (`mixtral-8x7b`, `mixtral-8x22b`)
2. **Inhomogeneous composite blocks (`src/maxtext/layers/batch_split_inhomogeneous.py`):**
   - **Gemma 4 MoE** (`gemma4-26b`, `[5 Sliding + 1 Global]` attention sub-layers per block)
   - **DeepSeek-V4** (`deepseek4-284b`, `deepseek4-1600b`, alternating `[HCA, CSA]` with 4D
     manifold-constrained hyper-connections `mhc_expansion_rate > 1`)
   - **GPT-OSS** (`gpt-oss-20b`, `gpt-oss-120b`, alternating `[Sliding, Global]` attention with sink
     logits)
   - **Qwen3.5 / Qwen3.8** (`Qwen3NextScannableBlock`, `[N-1 GatedDeltaNet + 1 FullAttention]`)

When a configuration is outside the supported scope (for example dense-only models, inference / KV
cache decoding, `scan_layers=False`, or `per_device_batch_size` producing an odd local microbatch
axis without sequence sharding), `batch_split_supported()` logs the reason and transparently falls
back to the standard scan or `prefetch_fsdp_weights` path.

## When to enable `enable_batch_split`

- **Enable (`enable_batch_split=True`)** on **Expert-Parallel MoE workloads (`ici_expert_parallelism >= 4`)**
  with standard self-attention or MLA (homogeneous DeepSeek-V3 / Kimi-K2, or inhomogeneous
  sliding/global attention models such as Gemma 4 MoE and DeepSeek-V4) where SparseCore EP token
  collectives are exposed and FSDP uses orthogonal ICI axes (for example `EP` over `(X, Y, core)` or
  `(Z, core)` on `SC0` and `FSDP` on the remaining torus axes on `SC1`).
- **Keep disabled (`enable_batch_split=False`, default)** on **hybrid linear-attention recipes**
  such as Qwen3.5 / Qwen3.8 (`[3 GatedDeltaNet + 1 FullAttention]`) when FSDP weight traffic already
  saturates shared chip ICI links or sequential chunk recurrences dominate TensorCore time: splitting
  into half-sized microbatches runs sequential Gated DeltaNet `conv1d` / `triangular_solve`
  recurrences twice per sub-layer without free ICI slack.

## Quick-start command (DeepSeek-V3 on TPU7x)

```bash
export LIBTPU_INIT_ARGS="\
  --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000 \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_enable_offloading_copy_to_sparsecore=false \
  --xla_tpu_enable_offloading_gather_to_sparsecore=false \
  --xla_tpu_scoped_vmem_limit_kib=65472"

python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name=deepseek3-671b \
  per_device_batch_size=1 \
  max_target_length=4096 \
  scan_layers=True \
  shard_mode=explicit \
  enable_batch_split=True \
  prefetch_fsdp_weights=True \
  prefetch_fsdp_defer_grad_reduce_scatter=True \
  moe_flat_fsdp_weights=True \
  use_ring_of_experts=True \
  moe_topk_before_ep_all_gather=True \
  moe_pin_sparse_core_ep_all_gathers=True \
  moe_ep_all_gather_sparse_core_id=0 \
  use_tokamax_gmm=True \
  use_gmm_v2=True \
  moe_gmm_v2_dlhs_transpose_rhs=True \
  moe_accumulate_wi_dlhs=True \
  moe_3d_dlhs_partial_sum=True
```

## Benchmark results (64 TPU7x chips, `4x4x4`)

Matched A/B step-time comparisons (`enable_batch_split=False` vs. `enable_batch_split=True`,
`max_target_length=4096`, `prefetch_fsdp_weights=True`, `prefetch_fsdp_defer_grad_reduce_scatter=True`)
on 64 TPU7x chips (`128` TensorCores):

| Model & configuration                                                                    | `enable_batch_split=False` | `enable_batch_split=True` | Step-time delta        |
| ---------------------------------------------------------------------------------------- | -------------------------- | ------------------------- | ---------------------- |
| **DeepSeek-V3 671B** (15L + 1 MTP, BF16, `pdbs=1`, `EP=32` on `SC0`, `FSDP=4` on `SC1`)  | 1115.3 ms                  | **975.0 ms**              | **−140.3 ms (−12.6%)** |
| **Gemma 4 26B MoE** (30L `[5 Sliding + 1 Global]`, FP8, `pdbs=2`, `DP=16, FSDP=2, EP=4`) | 2155.9 ms                  | **1935.2 ms**             | **−220.8 ms (−10.2%)** |
| **DeepSeek-V3 671B** (15L + 1 MTP, FP8, `pdbs=1`, `EP=32` on `SC0`, `FSDP=4` on `SC1`)   | 965.5 ms (947.0 ms chunk2) | **887.5 ms**              | **−78.0 ms (−8.1%)**   |
| **DeepSeek-V4 284B** (12L `[1 HCA + 1 CSA]` + 4D mHC, FP8, `pdbs=2`, `FSDP=16, EP=8`)    | 3483.4 ms                  | **3402.9 ms**             | **−80.4 ms (−2.3%)**   |
| **Kimi-K2 1T** (9L, `E=384`, FP8, `pdbs=2`, `FSDP=16, EP=8`, `moe_flat_fsdp_weights`)    | 1043.1 ms                  | **1039.3 ms**             | **−3.7 ms (−0.4%)**    |
| **Mixtral-8x22B** (16L, `shared_experts=0`, FP8, `pdbs=2`, `FSDP=32, EP=4`)              | 1772.1 ms                  | **1770.2 ms**             | **−1.9 ms (−0.1%)**    |

### Full 61-layer + 1-MTP DeepSeek-V3 671B scaling (`128` to `1,024` TPU7x chips)

Matched A/B steady-state step times (`per_device_batch_size=1`, `max_target_length=4096`, `61`
decoder layers + `1` MTP layer, `remat_policy=custom`, `moe_flat_fsdp_weights=True`,
`moe_accumulate_wi_dlhs=True`, `moe_3d_dlhs_partial_sum=True`):

| Slice chips (`topology`)                                     | Precision | `enable_batch_split=False` | `enable_batch_split=True` | Step-time delta         |
| ------------------------------------------------------------ | --------- | -------------------------- | ------------------------- | ----------------------- |
| **128** (`1 × 4x4x8`, `EP=16` on `SC0`, `FSDP=16` on `SC1`)  | BF16      | 4463.2 ms                  | **4089.7 ms**             | **−373.5 ms (−8.4%)**   |
| **128** (`1 × 4x4x8`, `EP=16` on `SC0`, `FSDP=16` on `SC1`)  | FP8       | 3799.5 ms (3726.2 chunk2)  | **3706.8 ms**             | **−92.6 ms (−2.4%)**    |
| **256** (`1 × 4x8x8`, `EP=16` on `SC0`, `FSDP=32` on `SC1`)  | BF16      | 4598.2 ms                  | **4326.7 ms**             | **−271.6 ms (−5.9%)**   |
| **512** (`1 × 4x4x32`, `EP=32` on `SC0`, `FSDP=32` on `SC1`) | BF16      | 5000.5 ms                  | **4617.8 ms**             | **−382.8 ms (−7.7%)**   |
| **512** (`1 × 4x4x32`, `EP=32` on `SC0`, `FSDP=32` on `SC1`) | FP8       | 4337.5 ms (4151.5 chunk2)  | **4147.1 ms**             | **−190.4 ms (−4.4%)**   |
| **1024** (`2 × 4x4x32`, `DCN_DP=2`, `EP=32, FSDP=32`)        | BF16      | 9340.4 ms                  | **7142.0 ms**             | **−2198.5 ms (−23.5%)** |
