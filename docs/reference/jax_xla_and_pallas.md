# JAX, XLA, and Pallas for MaxText Users

This document serves as a guide for MaxText users to understand MaxText’s core components. To move beyond basic model training and truly leverage the power of MaxText for ambitious research and production goals, a deep understanding of its foundational technology stack—JAX, XLA, and Pallas—is crucial.

MaxText's core design proposition is to provide a high-performance, massively scalable, yet simple Large Language Model (LLM) codebase written in pure Python/JAX, which is a direct result of building on this specific stack. The "simplicity" and "optimization-free" nature of the user-facing code is achieved by delegating immense complexity and optimization work to the underlying compiler technologies.

The stack can be understood as three layers:

1. JAX as the expressive Python-native programming model
2. XLA as the powerful backend compiler that unlocks hardware speed
3. Pallas as the expert's toolkit for writing custom, high-performance kernels for novel operations.

The following table provides a high-level overview to scaffold understanding before a more detailed exploration.

| Technology | Role in MaxText | Key Benefit for LLM Training |
| :---- | :---- | :---- |
| JAX | Programming Model & Transformations | Enables scalable, composable, and differentiable model definitions in pure Python. |
| XLA | High-Performance Compiler | Automatically fuses operations and compiles JAX code into optimized machine code for TPUs/GPUs. |
| Pallas | Custom Kernel Language | Allows for hand-tuned, hardware-specific kernels for peak performance on novel operations (e.g., MoE, custom attention). |

## 1\. JAX: The High-Performance Engine of MaxText

This section details JAX, focusing on the features that dictate MaxText's architecture and enable its remarkable scalability.

### 1.1. What is JAX?

JAX is a Python library designed for high-performance numerical computing and large-scale machine learning. It provides a NumPy-like API, which makes it familiar and easy to adopt for anyone in the scientific Python ecosystem.

However, JAX is much more than an accelerated version of NumPy. Its core power comes from an extensible system of composable function transformations, such as jit for just-in-time compilation, grad for automatic differentiation, built on top of a distributed array type, such that compute follows data. MaxText is fundamentally an application built by composing these powerful transformations.

### 1.2. The Pure Function Paradigm: A Core Design Constraint

JAX transformations are designed to operate on *pure functions*—functions that have no side effects and whose output depends solely on their explicit inputs. This is a critical concept that profoundly influences MaxText's architecture. In a typical MaxText train\_step, the function does not modify a global model state. Instead, it takes the current state (model parameters, optimizer state) as inputs and returns a completely new, updated state as its output.

This functional purity has a direct and important consequence for how users interact with the codebase. Since JAX requires pure functions for its transformations to work reliably, the model's state (like its weights) cannot be implicitly stored within objects but must be explicitly passed into and returned from the functions that manipulate them. This paradigm makes complex, object-oriented abstractions with hidden internal state—common in other frameworks—feel unnatural and difficult to implement in a "JAX-native" way.

### 1.3. Core JAX Transformations in MaxText

#### **1.3.1.** jax.jit**: From Python to High-Performance Code**

The jax.jit (Just-In-Time compilation) transformation is a cornerstone of JAX's performance. It takes a standard Python function and compiles it into an optimized computation graph using the XLA compiler backend. The first time this function is called with inputs of a specific shape and type, JAX "traces" the operations to build the graph, which is then compiled and cached. All subsequent calls with matching input structures will execute the highly optimized, pre-compiled machine code directly, bypassing the overhead of the Python interpreter for each operation.

In MaxText, the entire train\_step function is wrapped in jax.jit. This is the primary mechanism that transforms the readable, NumPy-like Python code defining the forward pass, loss calculation, and backward pass into a single, highly efficient program that can be executed on a TPU or GPU.

#### **1.3.2.** jax.grad**: Powering Training with Automatic Differentiation**

Training neural networks relies on gradient-based optimization. JAX provides the jax.grad transformation, which takes a numerical function (like a loss function) and returns a new function that computes its gradient. It supports reverse-mode differentiation, also known as backpropagation, which is essential for efficiently training deep neural networks.

Within the MaxText train\_step, jax.grad is applied to the loss function. This automatically generates the function that performs the entire backward pass, calculating the gradients of the loss with respect to all trainable model parameters. This new gradient function is then seamlessly composed with jax.jit to be compiled into the final, optimized training step.

#### **1.3.3. SPMD: Scaling from One Chip to Thousands**

Modern LLM training requires scaling across many accelerator devices. The dominant paradigm for this is SPMD (Single-Program, Multiple-Data), where a single program is executed in parallel on multiple devices, with each device operating on a different slice (or shard) of the data.

JAX provides powerful, high-level abstractions for SPMD programming. vmap provides automatic vectorization of functions over a batch and is used in numerous places in the MaxText codebase. While historically JAX users called jax.pmap as a direct way to map a function across devices, MaxText supports the more modern and flexible approach of operating on distributed arrays. This involves defining a logical Mesh of devices (e.g., a 2D grid) and using sharding annotations (PartitionSpecs) to declaratively state how data arrays should be distributed across this mesh. When a function with these sharded inputs is JIT-compiled, JAX automatically transforms the single-device program into a multi-device SPMD program. It transparently inserts all necessary inter-device communication collectives, such as all-reduce for summing gradients across data-parallel devices.

This unified scalability model is a key advantage of the JAX ecosystem. Training large LLMs requires different types of parallelism—data parallelism (splitting the batch), tensor parallelism (splitting a single matrix multiplication), and pipeline parallelism (splitting layers across devices). In many frameworks, implementing these requires different APIs, libraries, or coding patterns, which adds significant complexity. In JAX, all these forms of parallelism can be expressed through the single, unified concept of sharding tensors over a logical device mesh. For data parallelism, one shards the batch dimension of the input data. For tensor parallelism, one shards the weight matrices along their feature or output dimensions.

MaxText leverages this unification to great effect. The core model code remains largely agnostic to the parallelism strategy. Scalability is controlled primarily by changing the level of each kind of parallelism  in configuration files. This abstraction is a primary reason MaxText can be described as both "simple" and "massively scalable," as the immense complexity of distributed execution is handled by JAX and the XLA compiler, rather than the user.

### 1.4. Composability: The JAX Superpower

The true power of JAX lies in the arbitrary composability of its transformations. A MaxText training step is a prime example of this principle in action. A function is first created by applying jax.grad to the loss function, and this new gradient function is then transformed by jax.jit along with sharding annotations to make it compiled and parallel.

This composability allows for clean, declarative code. Instead of writing complex, imperative loops for batching, manual gradient calculations, and explicit communication calls, developers declare *what* they want (e.g., a JIT-compiled, parallel, gradient computation), and JAX figures out *how* to execute it efficiently. This is fundamental to maintaining a readable and "forkable" codebase for a process as complex as distributed LLM training.

## 2\. XLA: The Silent Optimizer Unleashing Hardware Speed

This section demystifies XLA, showing how it enables MaxText's high performance without requiring manual, low-level tuning from the user.

### 2.1. What is XLA (Accelerated Linear Algebra)?

XLA is an open-source, domain-specific compiler for linear algebra. It is not a user-facing library but a powerful backend compiler that takes computation graphs from frameworks like JAX. It then optimizes and compiles these graphs into highly efficient machine code tailored for specific hardware including TPUs and GPUs. In the MaxText stack, XLA is the engine that jax.jit uses under the hood. When a MaxText function is JIT-compiled, JAX traces the operations to build an intermediate representation, which is then handed to XLA for optimization and native code generation.

### 2.2. The Magic of Operator Fusion

Fusion is widely considered XLA's single most important optimization. In a standard, "eager execution" model, each mathematical operation (e.g., a multiplication, then an addition, then a sum) is dispatched and executed sequentially. Each step requires launching a separate kernel and reading inputs from and writing intermediate results back to the accelerator's main memory (e.g., High Bandwidth Memory), a process that is often a major performance bottleneck.

XLA avoids this by analyzing the entire computation graph and "fusing" multiple sequential operations into a single, custom-generated kernel. This fused kernel can perform the entire sequence of calculations without ever writing the intermediate results to main memory. Instead, it keeps these values in the accelerator's much faster on-chip registers or shared memory, only reading the initial inputs and writing the final output. For example, a function like jnp.sum(x \+ y \* z) would normally be three separate kernel launches. XLA can fuse this into a single kernel, eliminating two slow memory round-trips and dramatically increasing performance. This is precisely the kind of optimization MaxText relies on for the performance of its core transformer blocks.

### 2.3. How MaxText Relies on XLA

The power of the XLA compiler is the primary source of MaxText's "optimization-free" user experience. Other high-performance LLM frameworks have traditionally relied on developers writing custom, low-level CUDA kernels for performance-critical operations to achieve peak hardware utilization. This requires specialized expertise and makes the code less portable and harder to modify.

MaxText's philosophy is to avoid this manual, low-level optimization where possible and instead write in pure Python/JAX. The claim that it is "optimization-free" for the user is predicated on the XLA compiler being capable of performing these critical low-level optimizations, like operator fusion, automatically. The user writes simple, readable Python code, and XLA is responsible for generating the high-performance machine code. This design choice makes MaxText simpler to modify and more portable across different XLA-supported backends (TPU, GPU) than a framework with hand-written kernels. The performance of MaxText on any given hardware platform is a direct reflection of the maturity and quality of the XLA compiler for that platform.

#### Practical Application: Ahead-of-Time (AOT) Compilation

A practical challenge with Just-In-Time compilation is that it occurs on the first run, which can cause a long startup delay, especially on a large cluster where every device needs to compile the program. To address this, MaxText provides a train\_compile.py utility. This tool allows a user to run the XLA compilation process *ahead of time* on a single machine (even one with only a CPU) for a specific target hardware configuration.

The resulting compiled artifact can then be saved and loaded by all machines in the training cluster. This workflow dramatically reduces job startup times and allows for faster debugging cycles by catching potential compilation errors, such as Out-of-Memory (OOM) conditions, during the offline AOT step rather than on the full, expensive cluster. This is a practical and powerful workflow built entirely around leveraging the capabilities of the XLA compiler.

## 3\. Pallas: Expert-Level Control for Cutting-Edge Performance

This section introduces Pallas as the advanced tool for pushing performance beyond what automatic compilation can achieve, explaining its crucial role in keeping MaxText state-of-the-art.

### 3.1. What is Pallas?

Pallas is an extension to JAX that enables developers to write custom, hardware-specific kernels for GPUs and TPUs directly within a Pythonic environment. It functions as a JAX-native kernel language. Under the hood, Pallas translates its high-level kernel definitions into low-level intermediate languages like Triton for GPUs or Mosaic for TPUs, but it abstracts away the need for the user to write in those specialized languages directly.

Pallas is effectively an "escape hatch" from the fully automated world of XLA. It is designed for situations where developers need fine-grained control over memory access patterns, data tiling, and hardware parallelism to achieve performance that the general-purpose XLA compiler cannot automatically discover.

### 3.2. Why Custom Kernels Matter for Modern LLMs

The field of LLM research moves incredibly fast, with new architectural components being invented constantly. Techniques like FlashAttention and various Mixture-of-Experts (MoE) routing strategies often have complex, data-dependent memory access patterns that are not well-suited for a general-purpose compiler like XLA to optimize automatically. For instance, the I/O-aware algorithm of FlashAttention requires explicit management of on-chip SRAM, a level of control that is beyond XLA's intended scope.

A framework that relies *only* on a general compiler would inevitably lag behind, unable to efficiently implement these new, state-of-the-art techniques. Pallas provides the necessary bridge between compiler-driven simplicity and cutting-edge research. It allows MaxText developers to implement highly optimized kernels for these specific, performance-critical new components. This creates a powerful hybrid development model: the vast majority of the model benefits from the simplicity and automation of standard JAX/XLA, while the most novel and performance-sensitive parts are accelerated with hand-tuned Pallas kernels. This allows MaxText to remain both relatively simple and at the performance frontier.

### 3.3. Pallas in the MaxText Ecosystem

It is important to understand that most MaxText users will *consume* Pallas kernels, not write them. When a user enables a feature in the MaxText configuration, such as a specific attention mechanism or MoE, they are often transparently opting into a Pallas-optimized implementation under the hood.

#### Concrete Example 1: Mixture-of-Experts (MoE) on TPU

MoE models are computationally dominated by block-sparse matrix multiplications, a pattern that is notoriously difficult for general compilers to optimize efficiently. To accelerate MoE training, Google has open-sourced Pallas kernels specifically optimized for block-sparse matrix multiplication. When a user trains a Mixtral-style model in MaxText, they are leveraging these highly tuned kernels to achieve peak performance in the expert layers.

#### Concrete Example 2: Fused Attention ("Splash Attention")

Standard self-attention is a memory-bandwidth-bound operation, making it a key target for optimization. Fused attention kernels like FlashAttention have become the industry standard for overcoming this bottleneck. MaxText implements its own custom, high-performance fused attention kernel, referred to as "splash attention," which is enabled by default during training to provide a faster alternative. While the implementation details are part of the MaxText codebase, the pattern of creating custom, named kernels for critical operations like attention is exactly the problem Pallas is designed to solve, allowing MaxText to offer a highly efficient attention implementation without waiting for the general XLA compiler to natively support such complex fusions.

## Conclusion

The relationship between these technologies creates a layered journey from user code to hardware execution:

1. **The User writes in MaxText:** A developer interacts with high-level concepts in Python, modifying model layers in a familiar style or adjusting parallelism strategies in YAML configuration files. For certain critical operations, they may be using a Pallas kernel without even knowing it, simply by enabling a feature.
2. **JAX provides the language:** This Python code is interpreted by JAX, which provides the NumPy API and the powerful jit, grad, and SPMD transformations. These transformations allow the user to express a complex intent—"I want to run a parallelized, differentiable training step"—in a declarative way.
3. **XLA does the heavy lifting:** JAX hands a graph of the program to the XLA compiler. XLA then performs optimizations like operator fusion and memory layout management to create machine code that runs incredibly fast on the target accelerator.
4. **Pallas provides the sharp edge:** For the few, highly specialized operations where XLA's automated heroism isn't enough to match state-of-the-art performance, a pre-written Pallas kernel provides a hand-tuned, optimal implementation, ensuring MaxText stays at the cutting edge.

The "simplicity" of MaxText is not in having the fewest lines of code, but in its *conceptual consistency* and its reliance on a powerful, unified stack to handle the immense underlying complexity of high-performance, distributed computing.

By understanding this stack, a user is no longer just running a script; they are a developer who understands the principles behind MaxText's performance and scalability. This knowledge equips them to diagnose performance issues, intelligently customize the architecture for novel research, and confidently push the boundaries of what is possible with large language models.
