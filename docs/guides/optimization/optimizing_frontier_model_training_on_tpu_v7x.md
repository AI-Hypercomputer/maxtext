<!--
 Copyright 2024 Google LLC

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

# Optimizing frontier model training on TPU v7x (Ironwood)

In this page, we share the exact optimization techniques ML performance engineers use at Google, so you can maximize Ironwood’s performance right away. For a deeper dive into the hardware, check out the [TPU cloud documentation](https://docs.cloud.google.com/tpu/docs/tpu7x).

## Components of training performance optimization

Let’s take a deeper look at the various components of Ironwood’s architecture that you need to understand to tune model training performance.

### Taking advantage of the architecture

#### Utilizing the memory hierarchy

Managing data movement between Ironwood’s multi-tiered memory system is a crucial element of managing performance. Ironwood features high-bandwidth memory (HBM), vector memory (VMEM), and host memory with the following characteristics:

* **HBM:** Each chip features 192 GB of HBM, a 6x increase over Trillium. Peak bandwidth is 7.38 TB/s. While vast, HBM can still be a bottleneck for memory-bound vector operations or inefficient data access.  
* **Vector Memory (VMEM):** VMEM is a smaller, on-chip SRAM with significantly higher bandwidth to the MXU than HBM. It acts as a high-speed scratchpad for custom kernels.  
* **Host memory and PCIe:** Each set of four TPU chips connects to a CPU host via PCIe. The host’s main memory can be used for offloading activations or optimizer states to free up HBM.

**Interconnect fabric and arithmetic intensity**

Arithmetic intensity (AI) is the ratio of peak FLOPs to communication bandwidth.  
For Ironwood, the One Dimensional AI is very high, approximately **11,500**. This means it can perform a large amount of computation for each byte of data moved. When tuning performance, focus on minimizing or hiding data movement so that the MXUs aren’t left idle waiting for data. For more on arithmetic intensity and benchmarking, see the [Benchmarking & tuning guide](benchmark_and_performance.md)

#### Utilizing SparseCore

SparseCore is a unique component of TPUs, a processing unit engineered for high-performance acceleration of workloads that involve irregular, sparse memory access and computation. One of the ways you can utilize SparseCore for large-scale model training on Ironwood is to offload collective computation to it. This allows collective communication operations (like All-Gather or Reduce-Scatter) to execute asynchronously with the main computations happening on the TensorCores. Using [specific XLA flags](https://github.com/AI-Hypercomputer/maxtext/blob/c0abc4c0c0a98e02413d7b6c669927d013467045/benchmarks/xla_flags_library.py#L70-L116) enables this offloading for the most common collectives.

#### Design for architectural alignment

Achieving peak performance on specialized hardware such as Ironwood requires designing hardware-friendly model architectures. Performance tuning starts with model definition, as architectural choices can set permanent performance limits. For practical instructions on adjusting these configurations, see [Customizing model configs for TPUs](custom_model.md)

* **Architectural specification:** The Ironwood MXU is a 256x256 systolic array, which is efficient when the contracting dimension is equal to a multiple of 256\.  
* **MXU utilization:** Models whose head dimensions are a multiple of 256 will be able to utilize the MXU fully and see high Model Flops Utilization (MFU) on the attention blocks. For models with head\_dim 128 or 64, for the QK product in flash attention, we see 50% or more underutilization of MXU, and recommend using other techniques to compensate for it.

### Balancing compute and memory utilization

The next challenge in performance optimization is managing the trade-off between compute and memory efficiency. This involves selecting appropriate sharding strategies and techniques like activation rematerialization to optimize resource use.

#### Finding optimal sharding strategy

Choosing the right sharding strategy is essential. A guiding principle is to select the simplest strategy that meets memory constraints, as this typically minimizes communication overhead. Before selecting a strategy, perform a **roofline analysis** to determine whether a given computation is limited by compute, memory bandwidth, or interconnect bandwidth.

*For a comprehensive overview of how to apply these strategies in MaxText, refer to the [Sharding on TPUs](sharding.md) guide. Below are Ironwood-specific considerations:*

* **Fully Sharded Data Parallelism (FSDP):** This is the preferred strategy for large-scale model training that exceeds the memory capacity of a single chip. FSDP shards the model’s weights, gradients, and optimizer states. Increasing the per-device batch size improves efficiency by introducing more compute, which can hide the latency of the All-Gather operations it introduces.  
* **Tensor Parallelism (TP):** TP shards individual tensors. Ironwood’s high AI (11.5k) requires an MLP dimension greater than 46k (for TP degree 4\) to be viable over ICI. Most open source models like Llama3 70B (MLP dimension 28,672) and Qwen 2.5 7B (MLP dimension 18,944) fall short, and using TP here would result in the system becoming communication-bound.  
* **Expert Parallelism (EP):** This can be a helpful sharding strategy for training Mixture of Experts (MoE) models. EP shards the “expert” layers across a set of devices (a device contains only a subset of experts), and an All-to-All communication collective is used to route tokens to their designated expert device.  
* **Context Parallelism (CP):** CP is **essential for long sequence lengths**. It shards the sequence dimension of activation tensors, allowing for a *fractional* per-device batch size. Because CP introduces more communication than FSDP, the rule of thumb is to use the **minimum degree of CP necessary**.

#### Activation rematerialization

Rematerialization reduces HBM footprint by discarding activations and recomputing them during the backward pass. While it saves significant amounts of memory, it incurs \~25-30% additional FLOPs.

MaxText provides granular control over these trade-offs via the `remat_policy` flag. Beyond presets like **full** (maximizes memory savings) and **minimal** (maximizes training speed), users can implement **custom policies**. This allows you to specify behavior for individual layers:

* **device:** Store the activation in HBM.  
* **remat:** Recompute the activation during the backward pass.  
* **offload:** Move the activation to CPU host memory via PCIe to free up HBM without the compute cost of recomputation.

### Leveraging kernels optimized for Ironwood

While architecture provides the foundation, achieving maximum performance requires optimizing the computational routines themselves. 

#### Leveraging Tokamax kernels

To address hardware-specific bottlenecks, we recommend utilizing Tokamax, a high-performance JAX kernels library, with many highly optimized TPU kernels. *For more details on writing, profiling, and tuning custom kernels, refer to the [Optimizing with Pallas kernels](pallas_kernels_performance.md) guide.*

* **Splash Attention:** Used as the primary attention implementation to eliminate the HBM bottleneck of standard attention and use the most efficient attention implementation on TPUs.  
* **Megablox Grouped Matrix Multiplication (GMM):** For MoE workloads, Megablox efficiently handles grouped matrix multiplications by computing over the ragged activations representation.  
* **Empirical tuning with tune-jax:** The Tokamax library has [utilities](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/utils/tuning/tpu/README.md) that use `tune-jax` to perform empirical searches for optimal block sizes. Default kernel tile sizes are often suboptimal; tuning allows choosing hardware friendly VMEM tile sizes (as well as other hyperparameters) to maximize hardware utilization.

#### Memory pipeline tuning

Kernel performance, like flash attention, depends on the selected tile sizes in the kernel, whose size is limited by the total available VMEM (on-chip SRAM). Ironwood chips have 64 MB of VMEM per tensorcore, which can be split between the current scope (scoped VMEM) and future weight prefetch. Increasing the VMEM reserved for the current scope allows increasing the tile sizes used by the kernel, potentially removing memory stalls and increasing kernel performance (for example, `block_q`, `block_k`). You can control the scoped VMEM size by setting `xla_tpu_scoped_vmem_limit_kib` (in `LIBTPU_INIT_ARGS`).

 Further, experimenting with this setting allows you to explore kernel performance as well as end-to-end performance limits. Optimizing scoped VMEM size improves custom Pallas kernel performance.

## Case studies: Detailed optimization profiles

We ran pre-training benchmarks for both custom models and common OSS models on Ironwood. We conducted these benchmarks using a 4x4x4 configuration (64 chips) to evaluate performance across the 3D Torus topology. Let’s take a look at the results.

### Case study: Dense LLM (< 20B parameters) – short context (8k)

In this regime, the workload is primarily compute-bound. The objective is to keep the MXUs fully saturated and minimize TensorCore idle time.

* SparseCore offload: By offloading communication collectives to the SparseCore, we freed TensorCores to focus on MXU operations and achieved near-perfect overlap between communication and computation. *Result: 22% decrease in step time.*  
* Sharding with FSDP: FSDP gave us the best performance as it is designed to overlap communication with computation more efficiently for a model of this size.  
* Splash Attention and kernel tuning: We replaced standard attention with Splash Attention. Because default block sizes often lead to either memory stalls or poor compute units overlap, we used `tune-jax` to find the exact “sweet spot” for Ironwood’s SRAM. *Result: 12% decrease in step time.*

### Case study: Dense LLM (< 20B parameters) – long context (128k)

At a context length of 128k, activation memory grows with sequence length, making out-of-memory (OOM) errors the primary hurdle.

* SparseCore Offload: Offloading All-Gather and Reduce-Scatter operations ensured that the communication required for TP and CP did not stall the MXUs. *Result: 5% reduction in step time.*  
* Hybrid Parallelism (FSDP16 \+ TP2 \+ CP2): To handle a full batch, we utilized a hybrid approach of CP2 and TP2. We chose TP2 specifically to align the workload with Ironwood’s dual-chiplet architecture. This allows frequent communications to occur over the internal die-to-die (D2D) interface — which is 6x faster than the standard ICI. *Result: 4% performance improvement compared to using CP4 alone.*  
* Max logits estimate: The Tokamax Splash Attention kernel was optimized by setting a value for `max_logit_const` (in MaxText, configured via `use_max_logits_estimate`). This replaces the reduction calculation of the max logit during the softmax operation, reducing computations and synchronization overhead. *Result: 4% reduction in step time.*

### Case study: MoE 110B – short context (8k)

Training a 110B MoE model introduces unique structural inefficiencies because tokens are routed to specific “experts,” creating “ragged” batches.

* SparseCore offload: We leveraged SparseCore offloading to handle the heavy communication requirements of the MoE architecture without stalling the MXUs. *Result: 15% decrease in step time.*  
* Sharding using FSDP: We experimented with a hybrid approach of EP and FSDP, but the All-to-All collective used in EP caused a large bottleneck. We ultimately got the best performance using FSDP for this model.  
* Tokamax GMM kernel: We employed Megablox because it performs only the necessary work for each expert using parallel dense GEMMs, without wasteful padding. Using `tune-jax` further optimized the tiling strategy. *Result: 10% decrease in step time.*

## Get started

7th generation Ironwood TPUs are available for your frontier model training workloads. To learn more and get started:

* **Explore [Tutorials](../../tutorials.md):** Access our pre training tutorials for a hands-on experience training a model in JAX  
* **Experiment with [Tokamax](https://github.com/openxla/tokamax/tree/main) kernels**: Use our high-performance JAX and Pallas kernels library to address hardware-specific bottlenecks and optimize attention and MoE workloads.  
* **Deploy optimized training recipes**: Use these [ready-to-use optimized recipes](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/ironwood/README.md) to understand techniques used to run common OSS models on Ironwood efficiently.
