# JAX, XLA, and Pallas reference

This document serves as a guide for MaxText users to understand the components at the core of MaxText. To move beyond basic model training and truly leverage the power of MaxText for ambitious research and production goals, a deep understanding of its foundational technology technologies — JAX, XLA, and Pallas — is crucial.

MaxText's core design proposition is to provide a high-performance, massively scalable, yet simple Large Language Model (LLM) codebase written in pure Python/JAX, which is a direct result of building on this specific stack. The simplicity of the user-facing code is achieved by delegating immense complexity and optimization work to the underlying compiler.

MaxText builds on the following core technologies:

1. [JAX](https://docs.jax.dev/en/latest/), for writing high-level numerical code
2. [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html), the kernel language of JAX
3. [XLA](https://openxla.org/xla), the compiler for JAX code
4. Mosaic, the Pallas compiler

The following table provides a high-level overview to scaffold understanding before a more detailed exploration.

| Technology | Role in MaxText | Key Benefit for LLM Training |
| :---- | :---- | :---- |
| JAX | Programming Model & Transformations | Enables scalable, composable, and differentiable model definitions in pure Python. |
| JAX Pallas | Custom Kernel Language | Allows for hand-tuned, hardware-specific kernels for peak performance on novel operations (e.g., MoE, custom attention). |
| XLA | JAX Compiler | Automatically fuses operations and compiles HLO code, emitted by JAX into optimized LLO machine code for TPUs/GPUs. |
| Mosaic | Pallas Compiler | Compiles the Mosaic IR code emitted by JAX Pallas into LLO |

## 1. JAX: the high-performance engine of MaxText

This section details JAX, focusing on the features that dictate MaxText's architecture and enable its remarkable scalability.

### 1.1. What is JAX?

JAX is a Python library designed for high-performance numerical computing and large-scale machine learning. It provides a NumPy-like API, which makes it familiar and easy to adopt for anyone in the scientific Python ecosystem.

However, JAX is much more than an accelerated version of NumPy. Its core power comes from an extensible system of composable function transformations, such as jit for just-in-time compilation, grad for automatic differentiation, built on top of a distributed array type, such that compute follows data. MaxText is fundamentally an application built by composing these powerful transformations.

### 1.2. The pure function paradigm: a core design constraint

JAX transformations are designed to operate on *pure functions*—functions that have no side effects and whose output depends solely on their explicit inputs. This is a critical concept that profoundly influences MaxText's architecture. In a typical MaxText `train_step`, the function does not modify a global model state. Instead, it takes the current state (model parameters, optimizer state) as inputs and returns a completely new, updated state as its output.

This functional purity has a direct and important consequence for how users interact with the codebase. Since JAX requires pure functions for its transformations to work reliably, the model's state (like its weights) cannot be implicitly stored within objects but must be explicitly passed into and returned from the functions that manipulate them. This paradigm makes complex, object-oriented abstractions with hidden internal state—common in other frameworks—feel unnatural and difficult to implement in a "JAX-native" way. Fortunately MaxText is built on top of Flax's [NNX](https://flax.readthedocs.io/en/stable/), which handles a lot of the complexity for the user, presenting more of an object oriented interface.

### 1.3. Core JAX transformations in MaxText

#### 1.3.1. jax.jit: from Python to high-performance code

The [`jax.jit`](https://docs.jax.dev/en/latest/jit-compilation.html) (Just-In-Time compilation) transformation is a cornerstone of JAX's performance. It takes a standard Python function and compiles it into an optimized computation graph using the XLA compiler backend. The first time this function is called with inputs of a specific shape and type, JAX "traces" the operations to build the graph, which is then compiled and cached. All subsequent calls with matching input structures will execute the highly optimized, pre-compiled machine code directly, bypassing the overhead of the Python interpreter for each operation.

In MaxText, the entire `train_step` function is wrapped in `jax.jit`. This is the primary mechanism that transforms the readable, NumPy-like Python code defining the forward pass, loss calculation, and backward pass into a single, highly efficient program that can be executed on a TPU or GPU.

#### 1.3.2. jax.grad: powering training with automatic differentiation

Training neural networks relies on gradient-based optimization. JAX provides the [`jax.grad`](https://docs.jax.dev/en/latest/automatic-differentiation.html) transformation, which takes a numerical function (like a loss function) and returns a new function that computes its gradient. It supports reverse-mode differentiation, also known as backpropagation, which is essential for efficiently training deep neural networks.

Within the MaxText `train_step`, `jax.grad` is applied to the loss function. This automatically generates the function that performs the entire backward pass, calculating the gradients of the loss with respect to all trainable model parameters. This new gradient function is then seamlessly composed with `jax.jit` to be compiled into the final, optimized training step.

#### 1.3.3. SPMD: scaling from one chip to thousands

Modern LLM training requires scaling across many accelerator devices. The dominant paradigm for this is SPMD (Single-Program, Multiple-Data), where a single program is executed in parallel on multiple devices, with each device operating on a different slice (or shard) of the data.

JAX provides powerful, high-level abstractions for SPMD programming. [`jax.vmap`](https://docs.jax.dev/en/latest/automatic-vectorization.html#id2) provides automatic vectorization of functions over a batch and is used in numerous places in the MaxText codebase. While JAX users can use `jax.shard_map` for maximum control over parallelization, MaxText more typically uses the higher-level approach of operating on distributed arrays. This involves defining a logical Mesh of devices (e.g., a 2D grid) and using sharding annotations ([PartitionSpecs](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec)) to declaratively state how data arrays should be distributed across this mesh. When a function with these sharded inputs is JIT-compiled, JAX automatically transforms the single-device program into a multi-device SPMD program. It transparently inserts all necessary inter-device communication collectives, such as all-reduce for summing gradients across data-parallel devices.

This unified scalability model is a key advantage of the JAX ecosystem. Training large LLMs requires different types of parallelism—data parallelism (splitting the batch), tensor parallelism (splitting a single matrix multiplication), and pipeline parallelism (splitting layers across devices). In many frameworks, implementing these requires different APIs, libraries, or coding patterns, which adds significant complexity. In JAX, all these forms of parallelism can be expressed through the single, unified concept of sharding tensors over a logical device mesh. For data parallelism, one shards the batch dimension of the input data. For tensor parallelism, one shards the weight matrices along their feature or output dimensions.

MaxText leverages this unification to great effect. The core model code remains largely agnostic to the parallelism strategy. Scalability is controlled primarily by changing the level of each kind of parallelism  in configuration files. This abstraction is a primary reason MaxText can be described as both "simple" and "massively scalable," as the immense complexity of distributed execution is handled by JAX and the XLA compiler, rather than the user.

### 1.4. Composability: the JAX superpower

The true power of JAX lies in the arbitrary composability of its transformations. A MaxText training step is a prime example of this principle in action. A function is first created by applying `jax.grad` to the loss function, and this new gradient function is then transformed by `jax.jit` along with sharding annotations to make it compiled and parallel.

This composability allows for clean, declarative code. Instead of writing complex, imperative loops for batching, manual gradient calculations, and explicit communication calls, developers declare *what* they want (e.g., a JIT-compiled, parallel, gradient computation), and JAX figures out *how* to execute it efficiently. This is fundamental to maintaining a readable and "forkable" codebase for a process as complex as distributed LLM training.

## 2. JAX Pallas: expert-level control for cutting-edge performance

This section introduces Pallas as the advanced tool for pushing performance beyond what automatic compilation can achieve, explaining its crucial role in keeping MaxText state-of-the-art.

### 2.1. What is Pallas?

Pallas is the part of JAX that enables developers to write custom, hardware-specific kernels for GPUs and TPUs directly within a Pythonic environment. It functions as a JAX-native kernel language.

Pallas is effectively an "escape hatch" from the fully automated world of XLA. It is designed for situations where developers need fine-grained control over memory access patterns, data tiling, and hardware parallelism to achieve performance that the general-purpose XLA compiler cannot automatically discover.

### 2.2. Why custom kernels matter for modern LLMs

The field of LLM research moves incredibly fast, with new architectural components being invented constantly. Techniques like FlashAttention and various Mixture-of-Experts (MoE) routing strategies often have complex, data-dependent memory access patterns that can benefit from careful tuning.

A framework that relies *only* on a general compiler would inevitably lag behind, unable to efficiently implement these new, state-of-the-art techniques. Pallas provides the necessary bridge between compiler-driven simplicity and cutting-edge research. It allows MaxText developers to implement highly optimized kernels for these specific, performance-critical new components. This creates a powerful hybrid development model: the vast majority of the model benefits from the simplicity and automation of standard JAX/XLA, while the most novel and performance-sensitive parts are accelerated with hand-tuned Pallas kernels. This allows MaxText to remain both relatively simple and at the performance frontier.

### 2.3. Pallas in the MaxText ecosystem

It is important to understand that most MaxText users will *consume* Pallas kernels, not write them. When a user enables a feature in the MaxText configuration, such as a specific attention mechanism or MoE, they are often transparently opting into a Pallas-optimized implementation under the hood.

#### Concrete example 1: Mixture-of-Experts (MoE) on TPU

MoE models are computationally dominated by block-sparse matrix multiplications, a pattern that is notoriously difficult for general compilers to optimize efficiently. To accelerate MoE training, Google has open-sourced Pallas kernels specifically optimized for block-sparse matrix multiplication. When a user trains a Mixtral-style model in MaxText, they are leveraging these highly tuned kernels to achieve peak performance in the expert layers.

#### Concrete example 2: fused attention ("Splash Attention")

Standard self-attention is a memory-bandwidth-bound operation, making it a key target for optimization. Fused attention kernels like FlashAttention have become the industry standard for overcoming this bottleneck. MaxText uses a high-performance fused attention kernel, referred to as "splash attention," which is enabled by default during training to provide a faster alternative. While the implementation details are part of the MaxText codebase, the pattern of creating custom, named kernels for critical operations like attention is exactly the problem Pallas is designed to solve, allowing MaxText to offer a highly efficient attention implementation without waiting for the general XLA compiler to natively support such complex fusions.

## 3. XLA

This section demystifies XLA, showing how it enables MaxText's high performance without requiring manual, low-level tuning from the user.

### 3.1. What is XLA (Accelerated Linear Algebra)?

XLA is an open-source, domain-specific compiler for linear algebra. It is not a user-facing library but a powerful backend compiler that takes computation graphs from frameworks like JAX. It then optimizes and compiles these graphs into highly efficient machine code tailored for specific hardware including TPUs and GPUs. In the MaxText stack, XLA is the engine that `jax.jit` uses under the hood. When a MaxText function is JIT-compiled, JAX traces the operations to build an intermediate representation, which is then handed to XLA for optimization and native code generation.

### 3.2. Operator fusion

In a standard, "eager execution" model, each mathematical operation (e.g., a multiplication, then an addition, then a sum) is dispatched and executed sequentially. Each step requires launching a separate block of code and reading inputs from and writing intermediate results back to the accelerator's main memory (e.g., High Bandwidth Memory), a process that is often a major performance bottleneck.

XLA avoids this by analyzing the entire computation graph and "fusing" multiple sequential operations into a single operation which can perform the entire sequence of calculations without ever writing the intermediate results to HBM.

### 3.3. Overlapping compute and communication

Training a model requires applying the amount of compute required to train the model to a desired level as fast as possible. Accordingly, we want the accelerator to be executing computation as close to 100% of the time as can be achieved. Since compute is performed on data, this implies that data should be available for that computation and -- since data moves slowly relative to compute on modern accelerators -- this can become a bottleneck. Fortunately, during compilation, XLA is able to identify opportunities to overlap loading data required for future computations while executing the current computation, saving that otherwise wasted time.

### 3.4. How MaxText relies on XLA

The power of the XLA compiler is the primary source of MaxText's user experience. Other high-performance LLM frameworks have traditionally relied on developers writing custom, low-level CUDA kernels for performance-critical operations to achieve peak hardware utilization. This requires specialized expertise and makes the code less portable and harder to modify.

MaxText's philosophy is to avoid this manual, low-level optimization where possible and instead write in pure Python/JAX. This is predicated on the XLA compiler being capable of performing these critical low-level optimizations, like operator fusion, automatically. The user writes simple, readable Python code, and XLA is responsible for generating the high-performance machine code. This design choice makes MaxText simpler to modify and more portable across different XLA-supported backends (TPU, GPU) than a framework with hand-written kernels. The performance of MaxText on any given hardware platform is a direct reflection of the maturity and quality of the XLA compiler for that platform.

## 4. Mosaic

Similar to XLA's compilation of high-level JAX code, Mosaic is a compiler for Pallas kernels. While JAX is lowered to HLO which XLA optimizes into LLO, Pallas kernels are lowered to Mosaic IR, which the Mosaic compiler compiles to LLO. This LLO is then knitted into the LLO of the JAX code.

## Conclusion

The relationship between these technologies creates a layered journey from user code to hardware execution:

1. **The User writes in MaxText:** A developer interacts with high-level concepts in Python, modifying model layers in a familiar style or adjusting parallelism strategies in YAML configuration files. For certain critical operations, they may be using a Pallas kernel without even knowing it, simply by enabling a feature.
2. **JAX provides the language:** This Python code is interpreted by JAX, which provides the NumPy API and the powerful jit, grad, and SPMD transformations. These transformations allow the user to express a complex intent — "I want to run a parallelized, differentiable training step"—in a declarative way.
3. **Pallas provides the sharp edge:** For the few, highly specialized operations where compilation isn't enough to match state-of-the-art performance, a pre-written Pallas kernel provides a hand-tuned, optimal implementation, ensuring MaxText stays at the cutting edge.
4. **XLA and Mosaic do the heavy lifting:** JAX hands a graph of the program to the XLA compiler. XLA then performs optimizations like operator fusion and memory layout management to create machine code that runs incredibly fast on the target accelerator. For Pallas kernels, Mosaic does the same and the results of both compilers are combined into the final program.

The "simplicity" of MaxText is not in having the fewest lines of code, but in its *conceptual consistency* and its reliance on a powerful, unified stack to handle the immense underlying complexity of high-performance, distributed computing.

By understanding this stack, a user is no longer just running a script; they are a developer who understands the principles behind MaxText's performance and scalability. This knowledge equips them to diagnose performance issues, intelligently customize the architecture for novel research, and confidently push the boundaries of what is possible with large language models.
