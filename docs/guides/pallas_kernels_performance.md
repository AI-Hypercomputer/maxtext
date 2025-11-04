<!--
 Copyright 2023–2025 Google LLC

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

# Performance optimizations with Pallas kernels

New to Pallas? Start with the [official docs](https://docs.jax.dev/en/latest/pallas/index.html).

While JAX and the XLA compiler provide excellent out-of-the-box performance, some scenarios demand the next level of optimization. **Pallas** is a JAX extension for TPUs and GPUs that gives expert users fine-grained control over how kernels execute on accelerator hardware. When you know something about your problem’s structure that the general-purpose compiler cannot infer, you can often translate that knowledge into choices that outperform the default lowering. For example, you can explicitly manage **cache locality** through tiling, handle **data sparsity** in workloads like Mixture-of-Experts, or orchestrate **matrix unit overlap** with memory transfers through manual pipelining.

This guide explains **when** to consider Pallas, a **workflow** for developing and tuning kernels, and how Pallas is **used in MaxText** today.

## 🧠 The Pallas mindset: when to write a custom kernel

Think in **roofline** terms ([All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)) and in terms of **structure the compiler can’t see**:

* **Roofline framing.** Is your op **compute-limited** (MXU at or near peak) or **bandwidth-limited** (HBM↔on-chip transfers dominate)? Pallas tends to shine when you can reduce bandwidth pressure or avoid wasted work via better tiling and scheduling.
* **Compiler invisibles.** Irregular sparsity, ragged batch shapes, non-contiguous memory access, and domain-specific invariants are all signals that a custom kernel could help.

**Know when XLA is enough.** Before writing a custom kernel, always [profile your baseline](#1-high-level-profiling). If a standard operation (like a dense [`jnp.matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html)) is already performing well, the XLA compiler is doing its job. In these cases, a Pallas kernel will increase code complexity and maintenance burden with minimal performance improvement.

**When maintainability wins.** Pallas kernels are lower-level and harder to debug. If gains are small, prefer the simpler path.

**Autodiff note.** Pallas kernels require writing the autodiff rule manually (e.g., with [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html)), since unlike other transforms such as [`shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html),
it is very difficult to automatically infer the dual of the memory pipeline.

## 💡 Use Cases

### 1. Irregular compute (MoE, ragged activations)

For dense, regular GEMMs, XLA’s libraries are hard to beat. The exception is **Mixture-of-Experts (MoE)** MLPs with **ragged token→expert layouts** (some tokens routed to different experts; shapes are irregular). Zero-padding to make dense tensors wastes FLOPs; a custom kernel can operate only on the actually-selected tokens.

* In MaxText, we use Grouped Matrix Multiplication (GMM) via **Megablox** to compute per-expert matmuls on ragged batches. Precomputed metadata (e.g., token→expert indices and ranges) guides the grouped computation and avoids work on padded regions.

**Note:** *Megablox* is an efficient, non-capped MoE implementation in JAX. *Megablocks* refers to the equivalent PyTorch implementation. See [arXiv:2211.15841](https://arxiv.org/abs/2211.15841) for more details.

### 2. Memory-Access-Bound work (attention)

Attention kernels are classically **bandwidth-limited** if you materialize the full \[L,L\] score matrix. A Pallas kernel can block **Q/K/V** into tiles that fit on-chip and perform **online softmax accumulation**, never storing the massive intermediate.

* MaxText uses a Pallas attention kernel for training (Flash/Splash-style) and **paged/ragged** attention for inference to efficiently fetch KV cache pages and handle non-contiguous layouts.

## 🛠️ Pallas kernels in MaxText

To maximize performance, MaxText uses custom Pallas kernels for memory-bandwidth-bound or structurally irregular operations that a general-purpose compiler cannot optimize as effectively. Below are the key kernels we use. **Note**: Examples evolve; treat this list as guidance.

* **Training Attention (Flash/Splash-style):** This kernel is the default for training Transformer models in MaxText, such as DeepSeek, Gemma and Llama. It avoids creating the large \[L,L\] attention matrix to save memory, processing data in smaller, tiled chunks with online softmax accumulation.
  * [`src/MaxText/kernels/splash_attention_kernel.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/kernels/splash_attention_kernel.py)
* **Serving Attention (Paged & Ragged):** For high-throughput inference, this kernel efficiently fetches non-contiguous "pages" of the KV cache from memory. It is a key optimization for our serving stack and is used for models running on MaxText's inference engine.
  * [`src/MaxText/inference/paged_attention.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/inference/paged_attention.py)
  * [`src/MaxText/inference/paged_attention_kernel_v2.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/inference/paged_attention_kernel_v2.py)
* **MoE Grouped Matmul (Megablox GMM):** Sparse/irregular grouped GEMMs driven by host-built metadata.

  >  This is an efficient computation method for Mixture-of-Experts (MoE) models like DeepSeek, Llama 4, Mixtral and Qwen-MoE.  In MoE, each token is processed by only a few "experts," which is inefficient for standard matrix multiplication. Megablox solves this by having the CPU (**host**) first create a routing plan (**metadata**) that assigns tokens to experts. The accelerator (**device**) then uses this plan to perform many small, dense matrix multiplications in parallel (**Grouped Matrix Multiplication**), avoiding wasted work on unused experts.
  * [`src/MaxText/kernels/megablox/gmm.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/kernels/megablox/gmm.py)

  **Note:** Megablox accelerates the grouped **matmul**; **routing/gating** is separate code ([`src/MaxText/layers/moe.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/moe.py)).

## 🔧 The Pallas optimization workflow: code → profile → tune → repeat

### 1. High-level profiling

Give the kernel a clear name in traces and capture a profile. Always use [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html) when timing your operations.

``` python
import jax
from jax import profiler

def my_op(...):
    # This name shows up in Perfetto/TensorBoard traces
    with jax.named_scope("my_custom_kernel"):
        out = my_kernel_wrapper(...)
    return out

# Capture a Perfetto/TensorBoard trace
with profiler.trace("/tmp/tb_profile"):
    y = my_op(x)
    # Stabilize timing for accurate measurement
    jax.block_until_ready(y)
```

### 2. Deeper compiler view (optional)

For hard cases, inspect compiler dumps (e.g., LLO) to understand schedules, memory moves, and resource usage. Keep this as an advanced tool—most tuning decisions come from traces and microbenchmarks.

### 3. Systematic tuning

Performance is a function of interacting hyperparameters, chiefly block shapes (via [`BlockSpec`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.BlockSpec.html)). Build a small test script (a "harness") to systematically run the kernel with different block sizes. **Record the throughput and latency** for each run, and let data, not rules of thumb, pick the winners.
For a more automated approach, consider using libraries like [tune-jax](https://github.com/rdyro/tune-jax).

## ⚙️ Understanding TPU memory & compute

Pallas exposes the underlying hardware primitives for you to control.

* **HBM:** High-Bandwidth Memory (standard device memory).
* **VMEM:** On-chip vector SRAM for array tiles; your kernel primarily reads/writes VMEM refs.
* **SMEM:** On-chip scalar SRAM for control/metadata (e.g., counters, small tables).
* **Semaphores:** Available for advanced async/barrier patterns in manual pipelines.
* **MXU:** The Matrix Unit, optimized for large block GEMMs/convolutions.
* **VPU:** The Vector Processing Unit, used for elementwise/vector work.

**Alignment & Constraints:** Respect TPU BlockSpec constraints (divisibility/shape rules for trailing dimensions and supported block shapes). Start with tile shapes that fit in VMEM and meet these requirements, then sweep different sizes to find the optimum. Let profiling guide you; don't assume powers of two are always best.

## 🧱 Core Pallas design patterns

These are the common techniques used in MaxText's Pallas kernels.

* **Tiling & Blocking:** Move just a tile that fits on-chip, compute on it, and write it back.
* **Explicit Pipelining:** Overlap HBM↔VMEM loads with compute to hide latency (e.g., double-buffering).
* **Online Accumulation:** Combine partial results as you go; don’t materialize huge intermediate arrays.
* **Auxiliary Metadata:** Precompute control tables (e.g., token-to-expert ranges) and keep them in fast scalar memory.
* **Compute↔Communication Overlap:** In distributed runs, overlap local work with cross-device traffic when possible.

## ✍️ Writing & integrating a kernel

A Pallas kernel is a Python function that operates on Refs (references to array tiles). When this function is passed to [`pl.pallas_call`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html), it will be compiled and scheduled by Pallas.

### Example 1: Minimal elementwise add

Shows the kernel/ref pattern used by `pallas_call`.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_vectors_kernel(x_ref, y_ref, o_ref):
    o_ref[:] = x_ref[:] + y_ref[:]

def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    assert x.shape == y.shape
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
    )(x, y)
```

### Example 2: Blocked 2D add with BlockSpec

This example shows how to map a grid of blocks over larger arrays.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def tile_add_kernel(x_ref, y_ref, o_ref):
    # Operate on the tile slices handed in by BlockSpecs (already in VMEM on TPU).
    o_ref[:, :] = x_ref[:, :] + y_ref[:, :]

def tile_add(x: jax.Array, y: jax.Array) -> jax.Array:
    assert x.shape == y.shape and x.ndim == 2
    B0 = min(128, x.shape[0])  # Example choice; tune this with a sweep
    B1 = x.shape[1]            # Full width tile (for illustration)

    # Map program id (tile index) -> tile origin in the full (HBM) array.
    # NOTE: The runtime advances origins by `block_shape`, so `i` is already a tile
    # index. Do NOT multiply by B0 here.
    in_out_spec = pl.BlockSpec(
        block_shape=(B0, B1),
        index_map=lambda i: (i, 0),
    )

    # Grid is implied by data & block shape (use ceiling-division helper).
    grid = (pl.cdiv(x.shape[0], B0),)
    # Note: grid size can also be computed dynamically at runtime.

    return pl.pallas_call(
        tile_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[in_out_spec, in_out_spec],
        out_specs=in_out_spec,
        grid=grid,
    )(x, y)
```

**Tip:** In practice, you’ll **sweep** `(B0, B1)`-i.e., try a small grid of tile sizes and pick the fastest. Focus tuning on block shapes; treat grid as derived.

## ⏩ Pipelining best practices (TPU)

Prefer `pl.pallas_call` with scratch buffers allocated in the appropriate memory space (VMEM/SMEM) and use multi-buffering to overlap HBM loads with compute. Advanced pipelining to consider: custom prefetch block order via a scalar prefetch grid (for details see [here](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)), which lets you control block execution order based on runtime values.


## 🌐 Distributed execution

Dispatch a kernel on multiple devices with `jax.shard_map`. It’s usually simpler and more maintainable than in-kernel cross-device communication. While Pallas supports low-level comms, `shard_map` is the right first choice for multi-device parallelism, and you can **communicate with `shard_map` collectives** when needed.

## 🐞 Debugging tips

* Use `interpret=True` in `pallas_call` to run the kernel body in a Python interpreter backend, simulating device execution on CPU without lowering through XLA.
* Start with a tiny problem size and assert on invariants inside the kernel.
* Add `jax.named_scope` liberally so kernels are easy to spot in performance traces.

## ✅ Putting it all together (checklist)

1. **Profile** the baseline using `named_scope` and `block_until_ready`.
2. **Tile arrays into smaller chunks using BlockSpecs** (virtually always necessary, even for simple kernels).
3. Build a **sweep harness** for block shapes (and optionally scalar prefetch grid choices).
4. **Validate** end-to-end performance in the model, not just microbenchmarks.
5. Consider **maintainability** and guard the new kernel with tests.
6. Consider applying **`jax.vmap`** to a Pallas kernel to simplify implementation; think of it as prepending grid dimensions automatically.

## 📚 References

* **Pallas Docs & Quickstart:** [docs.jax.dev/en/latest/pallas/index.html](https://docs.jax.dev/en/latest/pallas/index.html)
* **JAX Profiling Guides:** [jax.readthedocs.io/en/latest/profiling.html](https://jax.readthedocs.io/en/latest/profiling.html)
* **Manual Parallelism (shard_map):** [docs.jax.dev/en/latest/notebooks/shard_map.html](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
* **Distributed Pallas on TPU:** [docs.jax.dev/en/latest/pallas/tpu/distributed.html](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
