# Benchmarking & tuning guide

This tutorial guides you through setting up benchmarks and performing performance tuning in MaxText, **focusing on key aspects** like how to set up benchmarks, choose the right config and tuning the benchmark performance.

## How to setup benchmark

Setting up effective benchmarks is crucial for accurate performance tuning. Here's how to approach it:

### Benchmark with synthetic data and repeated batches

To efficiently benchmark MaxText without large, real datasets, use synthetic data to eliminate input and pipeline bottlenecks.

set `dataset_type` to "synthetic" and `reuse_example_batch` to 1.

### Start with arithmetic intensity analysis

Begin your benchmarking efforts by performing an arithmetic intensity analysis. This fundamental step helps you determine the ideal batch size and sharding strategy for your specific hardware and workload. Based on your analysis and hardware configuration, select the sharding approach that yields the best results.

Arithmetic intensity is calculated as the ratio of floating-point operations (FLOPs) to memory(bytes) or communication(bytes).

- **Arithmetic Intensity = FLOPs / Bytes**

This metric helps determine whether a computation is MXU-bound (high arithmetic intensity) or memory-bound/communication-bound (low arithmetic intensity).

[This sharding document](sharding.md) illustrates various sharding strategies and their roofline analysis, through AI analysis.

## Metrics for benchmark analysis

For benchmarking purposes, we collect the step time for training. This step time is then used to calculate MFU and throughputs, which provide insights into the utilization achieved for each benchmark workload.

- **MFU = flops_train_step / step_time / peak HW FLOPS**
- **Throughput = global tokens / step_time / number of devices**

More detailed are explained in [](performance-metrics).

## Tuning benchmark performance

### Tuning Remat policy

Rematerialization (remat) is a technique used to reduce memory consumption during model training. It works by recomputing activations during the backward pass instead of storing them in memory during the forward pass. This can be particularly beneficial for large models where memory is a bottleneck.

When tuning remat policy, the goal is to find a balance between memory savings and computational overhead. Recomputing activations consumes FLOPs, so aggressive rematerialization can increase training time. However, if memory is the limiting factor, remat can enable larger batch sizes or model sizes, ultimately leading to faster training or the ability to train models that would otherwise be impossible.

If there is enough host HBM memory, activations can be offloaded to host memory to save the recompute.

MaxText offers both preset and granular re-materialization policies, allowing you to tailor them to your specific requirements.

**Provided Remat policies**

Remat policies can be chosen from: `minimal_with_context`, `minimal`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_dot_except_mlp`, `save_qkv_proj`, `qkv_proj_offloaded`, `custom`, `minimal_offloaded`, `save_out_proj` and `full`.

These options offer a trade-off between speed (fastest to slowest) and HBM usage (highest to lowest)

`minimal_with_context` consumes the most HBM memory, while `full` signifies minimal checkpointing, with everything being rematerialized. [More explanation and latest support](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/decoders.py#L287)

**Custom policy**

To use a custom policy, set `remat_policy` to `custom` and specify the layers in the decode module as `offload`, `device`, or `remat`.

- `offload`: The tensor is offloaded to the CPU host.
- `device`: The activation remains on the TPU device.
- `Remat`: Rematerialization is performed during the backward pass.

**Automatic remat policy search with the Estimator**

Finding the optimal remat policy and batch size manually can be time-consuming. MaxText provides an **Estimator** tool (`estimator.py`) that automates this search using [Ahead-of-Time (AOT) compilation](../monitoring_and_debugging/features_and_diagnostics.md#ahead-of-time-compilation-aot). It leverages `train_compile` to test whether a given configuration causes an Out-Of-Memory (OOM) error *without* requiring the target hardware.

The estimator supports two modes:

1. **Search both batch size and remat policy** (when `per_device_batch_size` is *not* provided): It finds the Pareto frontier of batch size vs. remat policy by iterating through policies from full remat to full device, using binary search for the largest non-OOM batch size at each step.
2. **Search remat policy only** (when `per_device_batch_size` *is* provided): It finds the least aggressive (fastest) remat policy that fits in memory for the given fixed batch size.

*Mode 1 example: Search both batch size and remat policy (Llama 3.1 405B on tpu7x-1024)*

```bash
python -m maxtext.utils.estimator \
  maxtext/configs/base.yml \
  compile_topology=tpu7x-1024 \
  compile_topology_num_slices=1 \
  model_name=llama3.1-405b \
  max_target_length=32768 \
  ici_context_parallelism=8 \
  ici_fsdp_parallelism=-1 \
  log_config=False \
  write_estimator_result=False
```

*Mode 2 example: Search best remat policy for a fixed batch size (DeepSeek3 671B on v5p-1024)*

```bash
python3 -m maxtext.utils.estimator maxtext/configs/base.yml \
  model_name=deepseek3-671b \
  compile_topology=v5p-1024 \
  compile_topology_num_slices=1 \
  ici_fsdp_parallelism=512 \
  per_device_batch_size=2.0 \
  dtype=bfloat16 \
  weight_dtype=float32 \
  max_target_length=8192 \
  log_config=False \
  write_estimator_result=False \
  decoder_layer_input=offload
```

Key options:

- `write_estimator_result=True`: Writes runnable training commands to `remat_commands_from_estimator.txt`.
- `write_estimator_result=False` (default): Prints results to stdout only.
- You can pin specific tensor remat actions (e.g., `context=offload`) to constrain the search space.

*Advanced example: Search remat policy with XLA tuning flags (DeepSeek3 671B on tpu7x-512)*

For production workloads you often want to combine the estimator with XLA compiler tuning flags for SparseCore offloading, latency-hiding scheduling, and other optimizations. Set these via `LIBTPU_INIT_ARGS` before invoking the estimator:

```bash
export LIBTPU_INIT_ARGS=" \
  --xla_tpu_dvfs_p_state=7 \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false "

python3 -m maxtext.utils.estimator maxtext/configs/base.yml \
  compile_topology=tpu7x-512 \
  compile_topology_num_slices=1 \
  run_name=${WORKLOAD_NAME} \
  skip_jax_distributed_system=true \
  dtype=bfloat16 \
  per_device_batch_size=4.0 \
  model_name=deepseek3-671b \
  remat_policy=custom \
  decoder_layer_input=device \
  mu_dtype=bfloat16 \
  grad_dtype=bfloat16 \
  ici_fsdp_parallelism=128 \
  ici_expert_parallelism=4 \
  dataset_type=synthetic \
  dataset_path=gs://max-datasets-rogue \
  opt_type=adamw \
  steps=20 \
  sa_use_fused_bwd_kernel=true \
  use_max_logit_estimate=-1 \
  cost_estimate_flops_fwd=5000000000000 \
  cost_estimate_flops_bwd=5000000000000 \
  float32_weight_sum=False \
  megablox=true \
  sparse_matmul=true \
  use_tokamax_gmm=false \
  use_tokamax_splash=true \
  max_target_length=4096 \
  use_random_routing=true \
  use_ring_of_experts=true \
  use_ragged_sort=true \
  tokenizer_path=assets/tokenizer.mistral-v3 \
  base_output_directory=${BASE_OUTPUT_DIR} \
  merge_gating_gmm=false
```

This example fixes `per_device_batch_size=4.0` so the estimator runs in **Mode 2** (policy-only search), finding the least aggressive remat policy that fits the DeepSeek3 671B model on a tpu7x-512 pod. The XLA flags enable SparseCore collective offloading and latency-hiding scheduling, which affect compilation memory layout and thus the OOM boundary.

### Low precision training

MaxText supports quantization via QWIX. To enable this, set `use_qwix_quantization=true`.

Different quantization recipes are available, including` "int8", "fp8", "fp8_full", "fp8_gpu", and "fp8_nanoo"`.

For v6e and earlier generation TPUs, use the "int8" recipe. For v7x and later generation TPUs, use "fp8_full". GPUs should use “fp8_gpu” for NVIDIA and "nanoo_fp8" for AMD.

See [](quantization-doc).

### Choose sharding strategy

Sharding is crucial for optimizing model performance. MaxText offers various sharding strategies and hybrid options, including FSDP, TP, EP, CP, and PP, which can be configured through your MaxText settings.
[This document](sharding.md) illustrates in detail how sharding works in maxtext and chooses the right sharding config for your workload.

### Performance tuning on custom Pallas call

[Tune-jax](https://github.com/rdyro/tune-jax) offers a tuning tool for Pallas kernels on both GPU and TPU. Users can tune custom kernels for specific model configurations by providing input shapes and defining a tuning search space.

Usage:

```
import functools
from tune_jax import tune

@functools.partial(tune, hyperparams={'block_q': [256, 512, 1024], 'block_k': [8, 16]})
def my_pallas_function(...):
```

This example code will benchmark `my_pallas_function` across all combinations of `block_q` and `block_k`, automatically handling any compilation failures.

### Communication overlaps and tuning

There are two methods for asynchronous collective offloading:

1. Offload Collectives to Sparse Core:

   This method is recommended for v7x. To enable it, set the following flags from [link](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py#L70):

- `ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR`
- `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER`
- `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER`
- `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE`

2. Overlap Collective Using Continuation Fusion:

   This method is recommended for v5p and v6e. To enable it, set the following flags ([link](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py#L39)):

- `CF_FOR_ALL_GATHER`
- `CF_FOR_ALL_REDUCE`

Those XLA can be set via `LIBTPU_INIT_ARGS`

### Scoped Vmem tuning

`scoped_vmem` can be tuned using `xla_tpu_scoped_vmem_limit_kib`. The hardware limitations for vmem are 64M for v5e, 128M for v6e, and 64M for v7x.

This can be set via `LIBTPU_INIT_ARGS`, range can be 0 through hardware limit. For example:

```
LIBTPU_INIT_ARGS="---xla_tpu_scoped_vmem_limit_kib=65536"
```
