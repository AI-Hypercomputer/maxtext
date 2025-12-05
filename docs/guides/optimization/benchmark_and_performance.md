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

*   **Arithmetic Intensity = FLOPs / Bytes**

This metric helps determine whether a computation is MXU-bound (high arithmetic intensity) or memory-bound/communication-bound (low arithmetic intensity).

[This sharding document](sharding_on_TPUs) illustrates various sharding strategies and their roofline analysis, through AI analysis.

## Metrics for benchmark analysis

For benchmarking purposes, we collect the step time for training. This step time is then used to calculate MFU and throughputs, which provide insights into the utilization achieved for each benchmark workload.

*   **MFU = flops_train_step / step_time / peak HW FLOPS**
*   **Throughput = global tokens / step_time / number of devices**

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

`minimal_with_context` consumes the most HBM memory, while `full` signifies minimal checkpointing, with everything being rematerialized. [More explanation and latest support](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/decoders.py#L287)

**Custom policy**

To use a custom policy, set `remat_policy` to `custom` and specify the layers in the decode module as `offload`, `device`, or `remat`.

- `offload`: The tensor is offloaded to the CPU host.
- `device`: The activation remains on the TPU device.
- `Remat`: Rematerialization is performed during the backward pass.

### Low precision training

MaxText supports quantization via QWIX. To enable this, set `use_qwix_quantization=true`.

Different quantization recipes are available, including` "int8", "fp8", "fp8_full", "fp8_gpu", and "fp8_nanoo"`.

For v6e and earlier generation TPUs, use the "int8" recipe. For v7x and later generation TPUs, use "fp8_full". GPUs should use “fp8_gpu” for NVIDIA and "nanoo_fp8" for AMD.

See [](quantization).

### Choose sharding strategy

Sharding is crucial for optimizing model performance. MaxText offers various sharding strategies and hybrid options, including FSDP, TP, EP, CP, and PP, which can be configured through your MaxText settings.
[This document](sharding_on_TPUs) illustrates in detail how sharding works in maxtext and chooses the right sharding config for your workload.
[This document](sharding_on_TPUs) illustrates in detail how sharding works in maxtext and chooses the write sharding config for your workload.

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

    This method is recommended for v7x. To enable it, set the following flags from [[link](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py#L70)]:

*   `ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR`
*   `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER`
*   `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER`
*   `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE`

 2. Overlap Collective Using Continuation Fusion:**

    This method is recommended for v5p and v6e. To enable it, set the following flags [[link](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py#L39)]:

*   `CF_FOR_ALL_GATHER`
*   `CF_FOR_ALL_REDUCE`

Those XLA can be set via `LIBTPU_INIT_ARGS`

### Scoped Vmem tuning

`scoped_vmem` can be tuned using `xla_tpu_scoped_vmem_limit_kib`. The hardware limitations for vmem are 64M for v5e, 128M for v6e, and 64M for v7x.

This can be set via `LIBTPU_INIT_ARGS`, range can be 0 through hardware limit. For example:

```
LIBTPU_INIT_ARGS="---xla_tpu_scoped_vmem_limit_kib=65536"
```
