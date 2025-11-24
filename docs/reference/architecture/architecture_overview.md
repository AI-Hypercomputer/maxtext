# Architecture overview

## The MaxText philosophy

The architecture of MaxText is guided by a distinct and deliberate philosophy that prioritizes accessibility and scalability by deeply leveraging the power of the XLA compiler. This approach marks a strategic departure from frameworks that rely on extensive manual optimization. Instead, MaxText achieves its goals through a pure Python/JAX implementation that trusts the underlying compiler to handle the complexities of hardware optimization. Only for the most performance-critical operations, such as custom attention mechanisms or Mixture-of-Experts (MoE) routing, does MaxText use custom kernels written in  Pallas.

## Trusting the compiler

The MaxText framework is intentionally written as much as possible in pure Python and JAX, offloading the burden of performance optimization to the XLA (Accelerated Linear Algebra) compiler. This allows MaxText to achieve high Model FLOPs Utilization (MFU) and scale from a single host to tens of thousands of accelerator chips without requiring developers to write low-level, hardware-specific code.

This philosophy stands in stark contrast to alternative high-performance frameworks which lean heavily on custom accelerator-specific kernels. MaxText abstracts much of this layer away, relying on kernels just for the most complex, performance-sensitive code. By relying on XLA, the same high-level Python/JAX codebase can be efficiently compiled to target diverse hardware platforms, including both Google Cloud TPUs and NVIDIA GPUs, a key advantage for portability.

The practical application of this principle is evident throughout the codebase, most notably in the use of JAX's `jit` (just-in-time) compilation decorator. A core function, such as the `train_step`, is defined in Python and then wrapped with [`@jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html). This simple decorator instructs JAX to trace the function, convert it into its own intermediate representation, and then pass it to the XLA compiler. XLA performs a host of advanced optimizations—such as operator fusion, memory layout optimization, and parallelization—to generate highly efficient machine code tailored for the specific accelerator hardware.

For example, the functional training step in [`train.py`](https://github.com/AI-Hypercomputer/maxtext/blob/01c7137d4e13878e38baae44dc99e588eaa50a70/src/MaxText/train.py#L193) is compiled using `jax.jit` before being executed in the main training loop.

```py
# A simplified representation of the JIT compilation in MaxText/train.py
p_train_step = jax.jit(
  functional_train,
  in_shardings=in_shard_train,
  out_shardings=out_shard_train,
  static_argnums=static_argnums_train,
  donate_argnums=donate_argnums_train,
)
```

This code snippet demonstrates how much of the complexity of optimizing a training step, including forward pass, loss calculation, and gradient computation, is handed off to the compiler. The developer works with high-level Python and JAX primitives, while the compiler manages the low-level performance details. This strategic decision to trade the fine-grained control of custom kernels for the automated optimization and hardware portability of a compiler is central to MaxText's identity. It shifts the cognitive load from the end-user (the ML engineer) to the compiler development team, making high-performance computing more accessible to a broader audience proficient in Python.

## The control plane: configuration and orchestration

The control plane of MaxText provides a structured yet flexible interface for users to define, configure, and launch training and inference jobs. It is designed to scale with the user's needs, offering simple command-line execution for local development and sophisticated orchestration tools for production-level runs on large-scale clusters. This system is centered around a primary YAML configuration file and a tiered set of execution scripts.

### `base.yml`: the central configuration hub

Every MaxText job is governed by the same base YAML configuration file ([`src/MaxText/configs/base.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/01c7137d4e13878e38baae44dc99e588eaa50a70/src/MaxText/configs/base.yml)) with model-specific details and overrides passed through a second config (e.g. [`src/MaxText/configs/models/deepseek3-671b.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/01c7137d4e13878e38baae44dc99e588eaa50a70/src/MaxText/configs/models/deepseek3-671b.yml)). Finally, experiment-specific settings are passed on the command line. The contents of these together comprise all the hyperparameters and settings that define a run:

* Model architecture: Defines the core transformer structure, with parameters like `model_name` (e.g., 'llama2-7b'), `global_parameter_scale` for size, `base_emb_dim`, `base_num_heads`, the type of attention mechanism, and `quantization` settings (e.g., 'int8').
* Training and optimization: Controls the training process with settings like `steps`, `learning_rate`, optimizer parameters such as `adam_b1`, and the `per_device_batch_size`.
* Data pipeline: Specifies the data source via `dataset_type` ('tfds', 'grain', 'hf'), the `dataset_path` on Cloud Storage, and Hugging Face-specific parameters like `hf_path` and `hf_train_files`.
* Hardware and parallelism: Defines the physical and logical device layout with `ici_parallelism` (intra-chip interconnect), `dcn_parallelism` (data center network), and `compile_topology` for ahead-of-time compilation.
* Checkpointing and logging: Manages run artifacts with `enable_checkpointing`, `async_checkpointing`, the `base_output_directory` in a Cloud Storage bucket, and a unique `run_name`.


A critical feature of this system is its flexibility. While `base.yml` provides the default values, any parameter can be overridden at runtime via command-line arguments. This allows for easy scripting of experiments and hyperparameter sweeps without needing to modify the configuration file for every run. At the same time, reproducibility can of course be maintained, by storing command line overrides in .sh files.

### Simple local or distributed execution

MaxText can be executed trivially on a single TPU VM host and surprisingly easily on multi-host setups.

* Single-host development: This is the simplest entry point, designed for running MaxText on a single TPU VM (e.g., v5p-8) or a single GPU machine. It is ideal for initial setup, dependency installation, and small-scale debugging or experimentation.

* GKE with XPK (recommended for production): This is the most scalable and robust method for running MaxText. It leverages the Accelerated Processing Kit (XPK) on Google Kubernetes Engine (GKE). XPK is an orchestration tool that standardizes best practices for large-scale ML jobs. It decouples the provisioning of compute capacity from the execution of the training job, allowing for more efficient resource management. This approach is recommended for production-grade training and serving due to its scalability, fault tolerance, and integration with the broader Google Cloud ecosystem.


### Summary

The table below summarizes some of the most critical parameters in base.yml and the components of the architecture they control, serving as a quick reference for configuring a MaxText run.

| Parameter | Module(s) Affected | Description |
| :---- | :---- | :---- |
| model\_name | models.py, train.py | Selects the transformer architecture as specified in the corresponding model config file (e.g., 'llama2-7b'). |
| per\_device\_batch\_size | train.py, input\_pipeline.py | Sets the local batch size per accelerator chip. |
| ici\_parallelism, dcn\_parallelism | max\_utils.py, train.py | Defines the device mesh shape for intra-chip and data center network parallelism. |
| dataset\_type | input\_pipeline.py | Specifies the data loader backend ('tfds', 'grain', 'hf'). |
| enable\_checkpointing | checkpointing.py, train.py | Enables or disables saving model state. |
| async\_checkpointing | checkpointing.py, train.py | If True, saves checkpoints without blocking the training loop. |
| quantization | layers.py, optimizers.py | Enables quantization, e.g., 'int8' for AQT or Qwix. |
| compile\_topology | train\_compile.py | Specifies the target hardware topology for AOT compilation. |

## Core architectural components

MaxText is constructed from a set of modular Python components. Each module is responsible for one part of a distinct aspect of the LLM lifecycle, from model definition and data loading to state persistence and distributed setup.

### Model definition

MaxText's model implementations are captured in a set of shared and model-specific modules. The core transformer is defined in `MaxText/layers/models.py`, the transformer decoder in `decoders.py` and a model-specific `DecoderLayer` such as `deepseek.py` implements the core of a given model. Shared modules such as `embeddings.py` and `attentions.py` capture the inner layer building blocks used by most or all models, with some occasional awareness of the model context in which they operate (this balance of sharing code vs. the increased need for model-specific branching that can entail is a balancing act we're continuously revisiting).

The typical model comprises a decoder-only autoregressive transformer, but MaxText also supports multi-modal models such as Llama 4 and Gemma 3, and as such the transformer module (in `models.py`) makes use of a Vision Encoder where appropriate.

While the base model implementations are typically simple, MaxText is equipped to handle a wide range of advanced, industry-standard features necessary for state-of-the-art performance and efficiency:

* Mixture-of-Experts (MoE): MaxText provides native support for sparse MoE models, such as DeepSeek. This includes efficient "dropping" and "dropless" MoE implementations leveraging the MegaBlox [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel, which can be enabled via configuration flags.

* Advanced attention mechanisms: The architecture is not limited to standard self-attention. It supports variants like Grouped-Query Attention (GQA), Multi-Query Attention (MQA) and Multi-headed Latent Attention (MLA). Since, like MoE, attention can be a performance hot-spot in transformers, attention is typically implemented in [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernels, with Splash (Sparse, Flash) Attention being the default for training.

* Quantization: The framework seamlessly integrates with Google's Accurate Quantized Training (AQT) and Qwix libraries. Quantization logic is applied at the layer level.


The modularity of this design is clearly demonstrated by third-party extensions. For instance, the NVIDIA maxtext-jaxpp fork was able to add support for pipeline parallelism by inserting jaxpp.pipeline\_enter\_stage hooks directly into the \_\_call\_\_ method of the Decoder class, a testament to the codebase's modularity and extensibility.

### Data ingestion (`input_pipeline.py`)

[The data ingestion pipeline](data-input-pipeline) is a critical component for performance at scale. In MaxText, the main training loop interfaces with the data pipeline through the create\_data\_iterator function, which is called from train.py. This function acts as a facade, abstracting the specific data loading implementation from the rest of the training logic.

MaxText supports three primary data loading backends:

1. Hugging Face datasets: For streaming data directly from the Hugging Face Hub.
2. TFDS (TensorFlow Datasets): For using datasets in the TFRecord format.
3. Grain: A data loading library optimized for large-scale, distributed environments.


While all three are supported, MaxText recommends the use of Grain, particularly for multi-host training scenarios. The rationale stems from performance and determinism considerations, at which Grain excels. Grain uses a data format called ArrayRecord, which supports efficient random access by index. This allows for true global shuffling of data across all hosts and eliminates the performance bottleneck associated with sequential reading.

### State management and persistence (`checkpointing.py`)

MaxText's state management and persistence layer is built on [Orbax](https://orbax.readthedocs.io/en/latest/), a flexible and powerful open-source checkpointing library for JAX applications. The core logic is encapsulated within the
checkpointing.py module, which provides a comprehensive suite of tools for saving and loading training state with high performance and resilience.

The central function is create\_orbax\_checkpoint\_manager, which configures and returns an Orbax CheckpointManager instance. This manager handles the core checkpointing operations and is configured with several key features:

* Asynchronous checkpointing: By setting the `async_checkpointing` flag to true, users can enable non-blocking checkpoint saves. This is a critical performance optimization. The training loop can proceed with the next step on the accelerators while the CPU on each host handles the process of serializing the previous step's state and writing it to Google Cloud Storage. This effectively hides the I/O latency of checkpointing and maximizes accelerator utilization.
* Flexible state restoration: The `load_state_if_possible` function implements a sophisticated, prioritized logic for resuming a run. When a job starts, it first attempts to find and load a full checkpoint from the current run's output directory. If that fails, it checks if a path to a full state checkpoint from a different run has been provided via the `load_full_state_from_path` argument. If that also fails, it looks for a parameter-only checkpoint (without training/optimizer state) specified by `load_parameters_from_path`.
* Emergency and replicated checkpointing: For maximum resilience and rapid job resumption in large-scale, production environments like GKE, the module includes support for advanced Orbax features.


A fundamental aspect of the MaxText workflow is the conversion of checkpoints between different formats. Scripts are provided to handle both ingestion and egress of model weights:

* Ingestion: Utilities like convert\_gemma\_chkpt.py and llama\_or\_mistral\_ckpt.py are used to transform checkpoints from standard frameworks (e.g., Hugging Face PyTorch) into the native MaxText Orbax format, which includes the full PyTree structure required for training.
* Preparation for inference: Conversely, the generate\_param\_only\_checkpoint.py script serves a crucial role in the path to deployment. It takes a full training checkpoint (which contains model parameters, optimizer state, and other metadata) and strips it down to only the essential model parameters. This script also performs a critical transformation from the "scanned" format used during training (an optimization where layers are stacked into a single tensor for efficient compilation) to the "unscanned" format required for autoregressive decoding. The resulting lightweight, parameter-only checkpoint is optimized for use with the decode.py script or for deployment with the JetStream inference engine.
* There also exist conversion scripts to convert weights to Hugging Face, e.g. `llama_mistral_mixtral_orbax_to_hf.py`


### Utilities and distributed setup (`max_utils.py`)

The `max_utils.py` module serves as a collection of common helper functions used across the MaxText codebase, but its most critical function is to abstract away the initialization of the JAX distributed environment.

The `maybe_initialize_jax_distributed_system` function is one example of this abstraction. This single function encapsulates the logic required to correctly call `jax.distributed.initialize()` in various deployment scenarios. It inspects the configuration and environment to determine the correct initialization parameters, handling cases for:

* Different hardware types, such as `gpu_multiprocess`.
* Configurations involving asynchronous checkpointing and multi-controller setups, which have specific distributed system requirements.
* Specialized environments like GKE with emergency checkpointing enabled. In this scenario, the JAX process ID and the coordinator's network address are not known beforehand but are written to a file by the GKE orchestrator. The function contains logic to poll for this file and parse the necessary information to initialize the distributed system correctly.


By centralizing this complex, environment-dependent logic into a single utility function, MaxText keeps the main training script cleaner and shields the end-user from the low-level details of distributed system bootstrapping.

In addition to distributed setup, the module provides other essential utilities, such as a function to calculate the total number of parameters in a model's PyTree, helpers for creating and managing TensorBoard summary writers for logging, and the implementation of a stabilized cross-entropy loss function.

## Scaling and performance optimization

MaxText is engineered from the ground up to deliver state-of-the-art performance and to scale efficiently to massive accelerator clusters comprising tens of thousands of chips. This is achieved through a combination of JAX's parallelism features, deep reliance on the XLA compiler for hardware-specific optimization, and the integration of advanced techniques like quantized training.

### Parallelism via JAX distributed arrays

The foundation of MaxText's scaling strategy is JAX's `jit` transformation, which allows for the automatic distribution of computations across a logical grid, or "mesh," of accelerator devices. The shape and dimensions of this mesh are defined by the user in command line overrides to the base.yml configuration file through parameters like `ici_fsdp_parallelism` (for devices connected by high-speed Inter-Chip Interconnect) or `dcn_data_parallelism` (for devices connected across the Data Center Network).

This logical mesh abstraction enables the implementation of the standard parallelism strategies required for training large language models:

* Data parallelism (DP): The simplest form, where the entire model is replicated on each device (or group of devices), and the global data batch is split among the replicas.
* Fully sharded data parallelism (FSDP): An optimization over DP where the model's parameters, gradients, and optimizer states are sharded (split) across the data-parallel replicas, significantly reducing the memory footprint on each device.
* Tensor parallelism (TP): A model parallelism technique where individual operations within a transformer layer (such as large matrix multiplications) are split across multiple devices within a replica.
* Pipeline parallelism (PP): splitting multiple stages of the network (groups of layers) across devices

In MaxText, these strategies are implemented by annotating the model's PyTrees (the nested Python structures of arrays that hold the parameters and state) with sharding specifications. This is done using Flax's partitioning utilities, such as nn\_partitioning. These annotations provide requirements and hints to the compiler, telling it how each tensor should be distributed across the axes of the device mesh. The compiler then generates the appropriate collective communication operations (e.g., all-reduce, all-gather) needed to execute the parallel computation correctly and efficiently.

For more information on sharding see [our sharding documentation](sharding_on_TPUs).

### Hardware abstraction and performance via XLA

As established previously, the XLA compiler is the linchpin of MaxText's performance and portability. It acts as a powerful abstraction layer, taking the hardware-agnostic computation graph generated by JAX and compiling it into highly optimized, device-specific machine code. This allows the same MaxText codebase to run with high performance on both Google TPUs and NVIDIA GPUs.

The effectiveness of this compiler-centric approach is validated by impressive performance results. Google has successfully used MaxText to run a single training job across a cluster of 50,944 Cloud TPU v5e chips, demonstrating near-linear scaling. The framework consistently achieves high Model FLOPs Utilization (MFU) across various models and hardware configurations. For example, public benchmarks show Llama2-70B achieving 65% MFU on a v5p-128 pod and Mixtral 8x7B achieving 54.89% MFU on the same hardware.

Performance can be further tuned by setting specific XLA flags in the configuration scripts. These flags can enable or disable specific compiler passes, such as those that combine multiple collective communication operations (e.g., `xla_gpu_enable_all_gather_combine_by_dim` and `xla_gpu_enable_reduce_scatter_combine_by_dim`) to reduce network overhead and improve throughput.

### Quantization for throughput boost

One of the most significant performance levers available in MaxText is the integration of Google's Accurate Quantized Training (AQT) and Qwix libraries. These enable training with reduced numerical precision, reducing memory requirements and often increasing FLOPS, while maintaining model quality and convergence characteristics that are very close to the full-precision baseline.

Integration into MaxText is seamless for the user. Quantization can be enabled by simply setting, for example, `quantization: 'int8'` in the configuration file. This flag activates quantization-aware layers (defined in
[`src/MaxText/layers/quantizations.py`](https://github.com/AI-Hypercomputer/maxtext/blob/db7b85be153e6b7ca387a8d02c991f9d35bae6bd/src/MaxText/layers/quantizations.py)) that are applied to the relevant dense layers within the model's Flax definition. The quantization library handles the complexities of simulating quantization during the forward and backward passes, allowing the model to learn weights that are robust to the reduced precision.

## The ecosystem: interoperability and advanced features

MaxText is not merely a standalone training framework; it is a central component within Google's open-source AI ecosystem. Its architecture is designed for interoperability, supporting a seamless workflow from ingesting popular open models to deploying them for high-throughput inference. Furthermore, it includes a suite of advanced diagnostic tools essential for debugging and optimizing performance at a massive scale.

### A hub for open-source model training

A primary strategic goal of MaxText is to serve as a high-performance platform for training and fine-tuning the world's most popular open-source LLMs on Google's advanced AI infrastructure. The framework maintains support for a wide and actively growing list of model families, including Google's Gemma, Meta's Llama, Alibaba's Qwen, DeepSeek, Mistral AI's Mistral and Mixtral, Open AI's gpt-oss, and Moonshot AI's Kimi K2.

The critical technology enabling this strategy is the suite of checkpoint conversion scripts included with the repository. These scripts act as bridges, allowing users to import standard model weights from their original frameworks (which are often PyTorch-based) into the MaxText/Orbax format required for training with JAX. Utilities like `llama_or_mistral_ckpt.py` and `convert_gemma_chkpt.py` handle the complex task of remapping weight names and structures, making it straightforward for users to begin a fine-tuning run with a state-of-the-art pretrained model.

### Diagnostics for debugging at scale

Debugging performance issues in a distributed system with thousands of accelerators is a notoriously difficult challenge. MaxText incorporates several built-in diagnostic features designed to provide visibility into the system's behavior at scale.

* Stack trace collection: To diagnose program hangs or faults, users can set `collect_stack_trace: True` in the configuration. This feature will periodically dump the Python stack traces from all worker processes. The traces can be directed to the console for immediate inspection or, more scalably, uploaded to Cloud Logging, where they can be aggregated and queried to identify misbehaving nodes.
* HLO dumping: For deep, low-level performance analysis, MaxText allows users to dump the XLA High-Level Optimizer (HLO) graph. By setting the `dump_hlo` flag, the compiled graph for a specific training step can be saved to a local directory or uploaded to Cloud Storage. This HLO representation is invaluable for compiler engineers and advanced users who need to understand exactly how XLA is interpreting and optimizing the model, making it possible to debug subtle performance regressions or compiler-related issues.
* Goodput monitoring: The framework integrates with the ml-goodput-measurement library, which provides a more holistic view of job efficiency than simple TFLOPs calculations. This allows for the tracking of metrics that capture overall "goodput," accounting for factors like data loading time, compilation overhead, and idle time, giving a truer picture of end-to-end performance.
