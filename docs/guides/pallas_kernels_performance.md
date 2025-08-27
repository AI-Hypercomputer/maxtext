# Performance optimizations with Pallas Kernels

## Introduction

While JAX and the XLA compiler provide excellent out-of-the-box performance, writing custom kernels with Pallas, a JAX extension for GPUs and TPUs, can unlock even greater speed. Pallas allows for lower-level control over hardware execution, which is especially beneficial for memory-bound operations. This guide provides an overview of writing and integrating custom Pallas kernels to optimize MaxText's performance.

## Benefits of Using Pallas

While the XLA compiler is highly effective, Pallas provides a powerful tool for expert users to unlock additional performance in specific scenarios. Key benefits include:

* **Operator Fusion**: Manually fuse multiple operations into a single kernel to reduce launch overhead and memory I/O, especially when XLA's automatic fusion is suboptimal.
* **Hardware-Specific Optimizations**: Tailor code to the specific architecture of your GPU or TPU, leveraging features that the compiler might not fully utilize. This includes fine-grained control over memory access and parallelism.
* **Improved Memory Access Patterns**: With Pallas, you can optimize how data is read from and written to memory to maximize bandwidth and reduce latency.
* **Algorithmic Optimizations**: Implement novel algorithms not available in standard libraries, such as flash attention, which combines the benefits of the above points to significantly speed up attention and save memory.

## Getting Started with Pallas Kernels

### Writing a Custom Kernel

Pallas kernels are Python functions decorated with `@pallas.kernel` that use JAX APIs to operate on `Ref` objects, which are references to JAX arrays. Inside a kernel, you have direct control over reading from and writing to these references.

**Example Kernel:**

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

@pl.kernel
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x = x_ref[:]
  y = y_ref[:]
  o_ref[:] = x + y

def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
  )(x, y)
```

For a deeper dive, see the [Pallas Quickstart guide](https://jax.readthedocs.io/en/latest/pallas/quickstart.html).

### Integration Steps

* **Define the kernel**: Write the Pallas kernel function.
* **Create a wrapper**: Write a function that invokes the kernel using `pallas_call`.
* **Replace in MaxText**: Substitute the original JAX operation in the MaxText code with your new wrapper function.


### Advanced Topics

#### Understanding Pallas on TPU

On TPUs, Pallas kernels are compiled using **Mosaic**. When a Pallas kernel is called, its inputs are loaded from High Bandwidth Memory (HBM) into the TPU's fast, on-chip **Vector Memory (VMEM)**. The computations within the kernel are performed on data residing in VMEM.

TPUs have two main compute units:

* **Vector Unit (VPU)**: Executes element-wise operations.
* **Matrix Unit (MXU)**: Specialized for large matrix multiplications and convolutions.

Understanding this memory hierarchy and the different compute units is key to writing high-performance kernels. For more information, refer to the JAX documentation on [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

#### Kernel Tuning and Pipelining

Tuning is essential for achieving optimal performance. A key technique is **pipelining**, which hides memory access latency by overlapping computation with data transfers from HBM to VMEM. Pallas offers two main ways to implement this.

##### Pipelining via Grid Iteration

Instead of an explicit loop inside a kernel, you can define an iteration space using the `grid` argument in `pallas_call`. Pallas unrolls this grid into a series of kernel invocations. The Pallas runtime, especially with a `pltpu.PrefetchScalarGridSpec`, schedules these invocations and their memory transfers to overlap, effectively creating a pipeline. This is the approach used in MaxText kernels like [`megablox/gmm.py`](MaxText/kernels/megablox/gmm.py).

* **When to use**: This method is simpler and often sufficient when loop iterations are independent and don't require complex state to be carried between them.

##### Pipelining with `pallas.for_loop`

For more complex scenarios, you can use an explicit `pallas.for_loop` inside your kernel. This is not a standard Python loop; it's a specific instruction to the compiler to create a pipelined schedule. It gives you fine-grained control to manage the state carried between iterations, which is necessary for more advanced pipelining patterns.

* **When to use**: This is necessary when you need to manage complex state between loop iterations or have more intricate dependencies that can't be expressed easily through the grid.

For more information, refer to the JAX documentation on [Pallas TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).

#### Block Sizes & Other Parameters

* **Block Sizes**: The `BlockSpec` defines data tiling and is critical for performance. Start with a baseline (e.g., `(1024, 1024)` on TPUs), experiment with powers of two, and profile to find the optimal size for your hardware.
* **Grid Size**: The `grid` parameter sets the number of kernel instances. Tune it to balance parallelism and overhead.
* **Memory Spaces**: Use VMEM for intermediate results to reduce latency.
* **Compiler Parameters**: The `compiler_params` argument can pass additional flags to the Mosaic compiler for specific optimizations.


### Distributed Execution with `shard_map`

Pallas kernels can be seamlessly used in distributed environments with `jax.experimental.shard_map`. This allows you to run your custom kernels across a `Mesh` of devices, which is essential for large-scale training with MaxText. When using `shard_map`, you specify how your data is partitioned across the devices using the `in_specs` and `out_specs` arguments. For more details, see the JAX documentation on [distributed Pallas kernels](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html).


### Pallas Kernels in MaxText

MaxText includes several Pallas kernels that serve as excellent references:

* **Splash Attention**: `MaxText/layers/attention_op.py` (utilizes Jax's [splash\_attention\_kernel](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py))
* **Ragged Attention**: `MaxText/kernels/ragged_attention.py`
* **MegaBlox**: `MaxText/kernels/megablox/gmm.py`