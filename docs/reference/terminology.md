# Terminology

This page provides definitions for technical terms and concepts used throughout the MaxText codebase and documentation. Terms are organized by category for easy reference.

## Table of Contents

- [Performance Metrics](#performance-metrics)
- [Hardware & Interconnect](#hardware--interconnect)
- [Parallelism Strategies](#parallelism-strategies)
- [Attention Mechanisms](#attention-mechanisms)
- [Positional Encodings](#positional-encodings)
- [Model Architecture](#model-architecture)
- [Mixture of Experts (MoE)](#mixture-of-experts-moe)
- [Quantization](#quantization)
- [Memory Optimization](#memory-optimization)
- [Checkpointing](#checkpointing)
- [JAX & XLA Concepts](#jax--xla-concepts)
- [Training & Optimization](#training--optimization)
- [Data Pipeline](#data-pipeline)
- [Inference](#inference)
- [Configuration & Infrastructure](#configuration--infrastructure)

---

## Performance Metrics

**FLOP (Floating Point Operation):**
A single floating-point arithmetic operation (addition, multiplication, etc.).

**FLOPS (Floating Point Operations):**
Plural form of FLOP.

**FLOP/s:**
Floating Point Operations Per Second, measuring computational throughput.

**MFU (Model FLOP/s Utilization):**
The ratio of actual model FLOPS to peak hardware FLOPS. A key metric for measuring training efficiency. MaxText calculates this by dividing theoretical model flops per step by measured step time times peak hardware flops/s. Higher MFU indicates better hardware utilization.

**Goodput:**
The metric measuring productive time spent on training progress proportional to total time spent by the workload. Includes forward/backward passes and optimizer updates.

**Badput:**
Time spent on non-productive activities like initialization, data loading, checkpointing overhead, disruptions, and wasted progress.

**Step Time:**
Time taken to complete one training step (forward pass + backward pass + parameter update).

**Tokens per Second:**
Throughput metric measuring how many tokens are processed per second during training or inference.

---

## Hardware & Interconnect

**TPU (Tensor Processing Unit):**
Google's custom-designed AI accelerator chips optimized for machine learning workloads.

**GPU (Graphics Processing Unit):**
NVIDIA GPUs supported by MaxText for training and inference.

**ICI (Interchip-Interconnect):**
High-speed interconnect within a TPU slice or pod, connecting chips on the same physical rack. Provides much higher bandwidth than DCN.

**DCN (Data Center Network):**
Network connecting different TPU slices or pods across the data center. Lower bandwidth than ICI.

**HBM (High Bandwidth Memory):**
Built with DRAM technology, the main memory on each chip (typically 16-96 GiB depending on accelerator generation).

**VMEM (Vector Memory):**
Built with SRAM technology, on-chip memory cache (typically measured in MiB). Much faster but smaller than HBM.

**PCIe (Peripheral Component Interconnect Express):**
How TPUs communicate with the CPU host.

**Slice:**
A group of TPU chips connected via ICI (up to 256 chips for v5e/trillium, 8k-9k for v4/v5p).

**NVL Domain (NVLink Domain):**
For GPUs, a group of chips connected via high-speed NVLink (can span multiple hosts in Grace Blackwell).

**Pod:**
Collection of TPU chips forming a compute cluster.

**Node:**
Physical server in a cluster.

**Host:**
CPU/server coordinating accelerators.

---

## Parallelism Strategies

**SPMD (Single-Program, Multiple-Data):**
Programming paradigm where a single program executes in parallel on multiple devices with different data. The foundation of MaxText's distributed execution model.

**DP (Data Parallelism):**
Simplest parallelization strategy where each chip works on different batch data. Gradients are synchronized across all devices during the backward pass.

**FSDP (Fully Sharded Data Parallelism):**
Also called ZeRO-3, shards model weights, gradients, and optimizer states across devices to save memory. Each device only stores a portion of the full model.

**FSDP Transpose:**
Variant of FSDP that shards weights on the larger `mlp_dim` dimension instead of `embed_dim`, useful when `embed_dim` is larger than `mlp_dim`.

**TP (Tensor Parallelism):**
Shards activations along feature dimensions (embed/mlp) instead of batch dimension. Communicates activations vs weights compared to data parallelism.

**TP Transpose:**
Shards feed forward weights along `embed_dim` instead of `mlp_dim`, useful when `mlp_dim < embed_dim`.

**TP Sequence (Tensor Sequence Parallelism):**
Similar to TP but shards initial FF activations on sequence dimension instead of model dimension.

**SP (Sequence Parallelism):**
Shards layer inputs and FF activations along sequence dimension, shards QKV along head dimension for attention computation.

**CP (Context Parallelism):**
Similar to FSDP but shards sequence dimension of activations, enabling smaller per-device batch size with large sequences.

**EP (Expert Parallelism):**
Shards expert FF computation (weights and activations) by expert in MoE models. Essential for scaling large MoE architectures.

**PP (Pipeline Parallelism):**
Shards weights and computation by layers, splits batch into microbatches that flow through pipeline stages.

**Stage:**
Group of layers assigned to a device in pipeline parallelism.

**Layers per Pipeline Stage:**
Number of consecutive layers grouped in each pipeline stage.

**Pipeline Iteration:**
One complete pass through all pipeline stages.

**Bubble:**
Idle time in pipeline parallelism while waiting for work from previous stages.

**gPipe:**
Pipeline parallelism style that breaks batch into microbatches with bubble time (idle periods when stages wait for data).

**Circular Pipeline:**
Pipeline strategy where layers "wrap" back around to reduce bubble time and improve efficiency.

**2D TP:**
Combination of TP and TP Transpose for more efficient inference decoding, especially beneficial for large embedding dimensions.

**Hybrid Parallelism:**
Flexible combinations of FSDP, TP, EP, DP, PP, and CP strategies to optimize training efficiency across different model architectures and hardware configurations.

---

## Attention Mechanisms

**Attention:**
Core mechanism in transformers for modeling relationships between tokens in a sequence.

**Self-Attention:**
Attention where queries, keys, and values all come from the same input sequence.

**Causal Attention:**
Attention mask ensuring tokens only attend to previous positions, required for autoregressive generation.

**MHA (Multi-Head Attention):**
Standard attention with separate heads for queries, keys, and values, allowing the model to jointly attend to information from different representation subspaces.

**MQA (Multi-Query Attention):**
All query heads share a single set of key-value heads, reducing KV cache size by number of heads.

**GQA (Grouped-Query Attention):**
Variant where multiple query heads share the same key-value heads, providing a middle ground between MHA and MQA. Reduces KV cache size while maintaining more capacity than MQA.

**MLA (Multi-Head Latent Attention):**
DeepSeek's compressed attention mechanism using low-rank projections to significantly reduce KV cache memory requirements.

**Flash Attention:**
Memory-efficient fused attention kernel that avoids materializing the full attention matrix, significantly reducing memory usage and improving speed.

**Splash Attention:**
MaxText's Pallas-based fused attention kernel (Sparse, Flash) used as default for training on TPUs. Provides optimizations for both sparse and dense attention patterns.

**Sliding-Window Attention (SWA):**
Local attention limited to a fixed window size, reducing computational complexity for long sequences.

**Local-Global Interleaved Attention:**
Architecture alternating between local attention (sliding window) and global attention across layers.

**Chunk Attention:**
Attention divided into chunks for memory efficiency, useful for very long sequences.

**Paged Attention:**
Inference-optimized attention that efficiently fetches non-contiguous KV cache pages from memory, used in vLLM.

**Ragged Attention:**
Handles non-contiguous memory layouts in attention computation, useful for variable-length sequences.

**Attention Sink:**
Technique to maintain attention stability for very long contexts by preserving initial tokens in the KV cache.

**QK-Norm:**
Query-Key normalization technique used in Gemma 3 and Llama 4 to improve training stability.

---

## Positional Encodings

**RoPE (Rotary Position Embedding):**
Positional encoding that rotates query and key embeddings based on position. Used in Llama, Gemma, DeepSeek, and most modern LLMs. Enables better length extrapolation.

**YaRN (Yet another RoPE extensioN):**
Modified RoPE technique for better long-context scaling, adjusting frequency scaling based on wavelength.

**Absolute Position Encoding:**
Traditional fixed positional embeddings added to input tokens, used in original Transformer and GPT models.

---

## Model Architecture

**Transformer:**
Core architecture based on self-attention mechanisms, used as the foundation for modern LLMs.

**Decoder-Only:**
Transformer architecture with only decoder blocks, standard for autoregressive language models like GPT, Llama, and Gemma.

**Encoder-Decoder:**
Transformer architecture with both encoder and decoder components, used in sequence-to-sequence models.

**DecoderBlock:**
Single transformer layer containing attention and feedforward sublayers with residual connections and normalization.

**DecoderLayer:**
Model-specific implementation of decoder block (e.g., `llama2_layer`, `deepseek_layer`, `gemma_layer`).

**Embedding Layer:**
Converts discrete token IDs to dense vector representations in the model's hidden dimension.

**Unembedding Layer:**
Final layer mapping hidden states to vocabulary logits for next-token prediction.

**MLP (Multi-Layer Perceptron) / FFN (FeedForward Network):**
Feedforward network in transformer blocks, typically with two linear layers and an activation function. Often the most compute-intensive part of the model.

**SwiGLU:**
Activation function combining Swish and GLU (Gated Linear Unit), used in Llama, Gemma, and many modern LLMs. Requires 3 weight matrices instead of 2.

**GELU (Gaussian Error Linear Unit):**
Smooth activation function used in some models like GPT-3.

**RMSNorm (Root Mean Square Layer Normalization):**
Lighter alternative to LayerNorm that only normalizes by the root mean square, omitting mean centering. Used in Llama, Gemma, and most modern LLMs.

**LayerNorm (Layer Normalization):**
Normalization technique that normalizes across the feature dimension.

**Attention Bias:**
Learnable bias added to query, key, value projections in attention.

**Logits:**
Raw prediction scores before softmax in the output layer, representing unnormalized log probabilities.

**Soft Capping (Logit Soft Capping):**
Technique to limit logit values using tanh for numerical stability and improved training dynamics, used in Gemma 2 and 3.

**Scan Layers:**
Using `jax.lax.scan` to iterate over transformer layers, compiling them into a single optimized kernel for better memory and speed.

**Vision Encoder:**
Component processing image inputs in multimodal models, typically based on Vision Transformer (ViT) architecture.

**Multimodal Projector:**
Maps vision features from the vision encoder to the language model's embedding space.

---

## Mixture of Experts (MoE)

**MoE (Mixture of Experts):**
Architecture with multiple expert networks (FFN layers), where a router selectively activates only a subset of experts per token. Provides more model capacity with less computation.

**Router:**
Gating mechanism that assigns tokens to experts based on learned routing probabilities.

**Top-K Routing:**
Router selects the K experts with highest routing scores for each token.

**Shared Experts:**
Experts that are always activated regardless of routing decisions, used in DeepSeek-V3 to ensure a common knowledge base.

**Routed Experts:**
Experts that are activated based on routing decisions, providing specialized processing.

**Load Balancing Loss:**
Auxiliary loss encouraging even expert utilization to prevent some experts from being underused.

**Capacity Factor:**
Controls the maximum number of tokens each expert can process, used to limit memory and computation.

**Dropless MoE:**
MoE without capacity limits that processes all routed tokens, avoiding token dropping at the cost of variable computation.

**Sparse MoE:**
Only selected experts are activated per token, not all experts.

**Megablox:**
Efficient MoE implementation using grouped matrix multiplication (GMM) for handling ragged batches of tokens routed to different experts.

**GMM (Grouped Matrix Multiplication):**
Pallas kernel for efficient sparse/irregular grouped GEMMs in MoE, handling different batch sizes per expert.

**Expert Capacity:**
Maximum number of tokens each expert can process in a given batch.

**EP (Expert Parallelism):**
Sharding strategy that distributes experts across devices.

**Routing Groups:**
Grouping mechanism for routing in large MoE models to improve load balancing.

---

## Quantization

**Quantization:**
Reducing numerical precision from FP32/BF16 to lower-bit formats like INT8/FP8 to reduce memory and increase speed.

**QAT (Quantization-Aware Training):**
Training with simulated quantization effects, inserting fake quantization operations into the forward pass.

**PTQ (Post-Training Quantization):**
Quantizing weights and activations after training completion.

**AQT (Accurate Quantized Training):**
Google's quantization library for JAX, providing tools for QAT. [Documentation](https://github.com/google/aqt)

**Qwix:**
Recommended successor to AQT with a non-intrusive API for quantization in JAX. [Documentation](https://qwix.readthedocs.io/en/latest/)

**INT8:**
8-bit integer quantization format, reducing memory by 2-4x compared to FP32/BF16.

**FP8:**
8-bit floating-point format for quantization, maintaining better dynamic range than INT8.

**BF16 (Brain Float 16):**
16-bit floating point format with same exponent range as FP32 but reduced mantissa, offering a good balance between precision and memory.

**FP32:**
32-bit floating point, full precision format.

**Dynamic Range Quantization:**
Quantization with per-batch or per-tensor statistics for scaling factors.

**Static Quantization:**
Fixed quantization parameters determined during calibration.

**Calibration:**
Process of determining optimal quantization scaling factors using representative data.

---

## Memory Optimization

**Gradient Accumulation:**
Splitting batch into micro-batches and accumulating gradients before updating parameters, enabling larger effective batch sizes with limited memory.

**Gradient Accumulation Steps:**
Number of micro-batches to accumulate before performing a parameter update.

**Rematerialization (Remat) / Activation Checkpointing:**
Recomputing activations in backward pass instead of storing them, trading compute for memory.

**Remat Policy:**
Strategy for which activations to save vs recompute. Options include (from fastest/highest HBM to slowest/lowest HBM): `full`, `save_out_proj`, `save_qkv_proj`, `save_dot_except_mlp`, `save_dot_except_mlpwi`, `save_dot_with_context_except_mlp`, `minimal`, `minimal_with_context`, `minimal_offloaded`, `qkv_proj_offloaded`, and `custom`.

**Tiling / Chunking:**
Partitioning large tensors into smaller blocks processed sequentially to reduce peak memory usage.

**Vocabulary Tiling:**
Tiling the final unembedding layer to avoid materializing the full logits tensor (vocab_size × sequence_length × batch_size), controlled by `num_vocab_tiling` config.

**Offloading:**
Moving optimizer state or parameters to host CPU memory to free accelerator HBM.

**Optimizer Memory Host Offload:**
Storing optimizer state (momentum, variance) on CPU memory instead of accelerator memory.

**Parameter Memory Host Offload:**
Storing model parameters on CPU memory, loading them to accelerator only when needed.

**ZeRO-1 Sharding:**
Sharding only optimizer state across devices, not weights or gradients.

**ZeRO-3 Sharding:**
Same as FSDP, sharding weights, gradients, and optimizer state across all devices.

---

## Checkpointing

**Checkpoint:**
Saved model state including parameters, optimizer state, and other metadata (step count, etc.).

**Training Checkpoint:**
Full checkpoint with both parameters and optimizer state, used to resume training.

**Inference Checkpoint / Param-Only Checkpoint:**
Checkpoint containing only model parameters without optimizer state, used for inference or fine-tuning.

**Stacked Checkpoint / Scanned Format:**
Checkpoint with parameters stacked for `jax.lax.scan`, providing memory-efficient loading.

**Unstacked Checkpoint / Unscanned Format:**
Parameters stored layer-by-layer, standard format for public models and cross-framework compatibility.

**Orbax:**
JAX checkpointing library used by MaxText, providing flexible and efficient checkpoint management. [Document](https://orbax.readthedocs.io/en/latest/index.html)

**Async Checkpointing:**
Non-blocking checkpoint saves allowing training to continue while checkpoint is being written.

**Sync Checkpointing:**
Blocking checkpoint saves that pause training until checkpoint is written.

**Emergency Checkpoint:**
Local checkpoint saved for fast recovery after interruption, typically stored on local disk.

**Multi-Tier Checkpointing:**
Experimental Orbax feature combining periodic GCS saves with local emergency checkpoints.

**Checkpoint Conversion:**
Transforming checkpoints between formats (e.g., HuggingFace PyTorch ↔ MaxText Orbax).

**OCDBT (Optimized Cloud Database):**
Storage format for checkpoints optimized for cloud storage.

**Zarr3:**
Storage format for checkpoints using the Zarr v3 specification.

---

## JAX & XLA Concepts

**JAX:**
Python library for high-performance numerical computing with composable function transformations (grad, jit, vmap, pmap). [Documentation](https://docs.jax.dev/en/latest/index.html)

**XLA (Accelerated Linear Algebra):**
Domain-specific compiler for linear algebra that compiles JAX code to optimized machine code for TPUs, GPUs, and CPUs.

**JIT (Just-In-Time) Compilation:**
Compiling Python functions to optimized machine code at runtime using `jax.jit`.

**AOT (Ahead-of-Time) Compilation:**
Pre-compiling for specific hardware topology before execution, useful for reproducible performance.

**Pallas:**
JAX extension for writing custom TPU/GPU kernels with fine-grained control over memory and computation. [Documentation](https://docs.jax.dev/en/latest/pallas/index.html)

**Mosaic:**
Compiler for Pallas kernels, converting Mosaic IR to low-level optimized code.

**HLO (High-Level Optimizer):**
XLA's intermediate representation for computation graphs.

**LLO (Low-Level Optimizer):**
XLA's low-level intermediate representation closer to machine code.

**jax.grad:**
Automatic differentiation transformation for computing gradients.

**jax.vmap:**
Automatic vectorization transformation that vectorizes a function along batch dimensions.

**jax.scan:**
JAX function for efficient sequential operations, compiling loops into optimized kernels.

**shard_map:**
JAX API for explicit SPMD control with manual sharding specifications.

**Pure Function:**
Function with no side effects where output depends only on inputs, required for JAX transformations.

**PyTree:**
Nested Python structure (dicts, lists, tuples) of arrays, used to hold parameters and state in JAX.

**Mesh:**
Logical grid of accelerator devices for distributed computation, defined with named axes.

**PartitionSpec:**
Sharding specification describing how arrays are distributed across mesh dimensions.

**Logical Axes:**
Named axes in tensor shapes that map to physical mesh axes for sharding (e.g., `activation_batch`, `activation_embed`).

**Physical Axes:**
Actual device mesh dimensions (e.g., `data`, `fsdp`, `tensor`, `expert`, `stage`).

**Operator Fusion:**
XLA optimization combining multiple operations into a single kernel to reduce memory traffic.

**Collective Operations:**
Communication primitives for distributed computing. See [XLA Operation Semantics](https://openxla.org/xla/operation_semantics) for details.

**All-Reduce:**
Sum values across all devices and broadcast the result to all devices. [Details](https://openxla.org/xla/operation_semantics#allreduce)

**All-Gather:**
Gather sharded data from all devices and concatenate. [Details](https://openxla.org/xla/operation_semantics#allgather)

**Reduce-Scatter:**
Reduce across devices then shard the result. [Details](https://openxla.org/xla/operation_semantics#reducescatter)

**All-to-All:**
General permutation communication pattern for resharding data. [Details](https://openxla.org/xla/operation_semantics#alltoall)

**Collective Matmul:**
Fused operation overlapping matrix multiplication computation with communication.

---

## Training & Optimization

**Forward Pass:**
Computing model predictions from inputs by propagating through the network.

**Backward Pass:**
Computing gradients via backpropagation algorithm.

**Backpropagation:**
Algorithm for computing gradients through neural network layers.

**Training Step:**
One complete cycle of forward pass + backward pass + parameter update.

**Microbatch:**
Smaller batch when using gradient accumulation or pipeline parallelism.

**Global Batch Size:**
Total effective batch size across all devices and gradient accumulation steps.

**Per-Device Batch Size:**
Batch size processed on each individual device.

**Learning Rate:**
Step size for gradient descent optimization, controlling how much parameters change per update.

**Optimizer:**
Algorithm for updating parameters based on gradients (Adam, SGD, Adafactor, etc.).

**Adam:**
Adaptive moment estimation optimizer using first and second moment estimates.

**AdamW:**
Adam optimizer with decoupled weight decay, used as default in many modern LLM training recipes including Llama.

**Adafactor:**
Memory-efficient variant of Adam that reduces optimizer state size.

**SGD (Stochastic Gradient Descent):**
Basic optimization algorithm that updates parameters using gradients computed on mini-batches of data.

**Momentum:**
Accumulated gradient history for optimization, helping accelerate convergence.

**Weight Decay:**
Regularization technique penalizing large weight values, equivalent to L2 regularization.

**Gradient Clipping:**
Limiting gradient magnitude for training stability, preventing exploding gradients.

**Warmup:**
Gradually increasing learning rate from zero at training start to improve stability.

**Learning Rate Schedule:**
Time-varying learning rate strategy (constant, cosine decay, linear decay, etc.).

**Pre-training:**
Training model from scratch on large corpus of text.

**Post-training:**
Further training after pre-training, including SFT, RLHF, DPO, etc.

**DPO (Direct Preference Optimization):**
Alignment technique that directly optimizes the policy model using preference data without requiring a separate reward model. Simpler alternative to RLHF.

**Fine-tuning:**
Adapting pre-trained model to specific task with smaller learning rate.

**SFT (Supervised Fine-Tuning):**
Fine-tuning on labeled instruction datasets to improve model's instruction-following capabilities.

**RL (Reinforcement Learning):**
Training via reward-based optimization, used for aligning models with human preferences.

**GRPO (Group Relative Policy Optimization):**
RL algorithm for LLMs that is a variant of PPO without requiring a separate value model.

**PPO (Proximal Policy Optimization):**
RL algorithm with policy constraints to prevent large policy updates.

**Reward Model:**
Model that scores generation quality for RL training.

**Policy Model:**
Model being trained with RL, learning to maximize rewards.

**Reference Model:**
Frozen copy of policy model used for KL divergence penalty in RL to prevent policy from deviating too far.

**Multi-Token Prediction (MTP):**
Auxiliary loss predicting multiple future tokens simultaneously to enhance training efficiency, used in DeepSeek-V3.

**Knowledge Distillation:**
Training smaller student model to mimic larger teacher model's outputs.

**LoRA (Low-Rank Adaptation):**
Parameter-efficient fine-tuning method that adds small trainable low-rank matrices to frozen weights.

**RLHF (Reinforcement Learning from Human Feedback):**
Training methodology using human preference feedback to align model behavior with human values, typically involving a reward model and policy optimization.

---

## Data Pipeline

**Data Input Pipeline:**
System for loading and preprocessing training data efficiently.

**Grain:**
Recommended data loading library for MaxText, optimized for distributed training with JAX.

**TFDS (TensorFlow Datasets):**
Data loading using TensorFlow's dataset catalog and TFRecord format.

**HuggingFace Datasets:**
Data loading from HuggingFace Hub using the datasets library.

**TFRecord:**
Sequential-access TensorFlow data format.

**ArrayRecord:**
Random-access data format recommended for multi-host training, providing better shuffling.

**Parquet:**
Columnar data format with sequential access.

**Tokenization:**
Converting text strings to token IDs using a vocabulary.

**Sequence Length:**
Number of tokens in a sequence, including padding.

**Sequence Packing:**
Combining multiple short sequences into one batch element to improve efficiency and reduce padding waste.

**Padding Batch:**
Empty batch generated in distributed training when some hosts run out of data before others, used to keep all hosts synchronized during training steps.

**Data Sharding:**
Splitting dataset across multiple hosts/workers to enable parallel data loading.

**Global Shuffle:**
Shuffle data throughout the entire dataset and dataset shards for the best randomization. Requires file formats that support random access.

**Hierarchical Shuffle:**
Shuffle data locally by randomizing the file order, interleaving, and then shuffling in a limited buffer. Used to approximate global shuffle when the file format only supports sequential access.

**Chat Template:**
Template for formatting conversational data with special tokens for different roles (user, assistant, system).

**Synthetic Dataset:**
Artificially generated data for testing and benchmarking, useful for reproducible performance testing.

---

## Inference

**Inference:**
Using trained model for predictions on new inputs.

**Autoregressive Decoding:**
Generating one token at a time conditioned on all previous tokens.

**KV Cache:**
Cached key-value pairs from attention to avoid recomputing them for previous tokens during generation.

**Prefill:**
Initial processing of prompt to populate KV cache with context.

**Decode:**
Autoregressive generation of new tokens one at a time.

**Decode Length:**
Number of tokens to generate in decode phase.

**vLLM:**
Efficient inference engine with paged attention, memory optimization, and continuous batching.

**JetStream:**
MaxText's inference serving stack built on top of MaxEngine.

**MaxEngine:**
MaxText's inference server implementation providing high-throughput serving.

**Pathways:**
Google's orchestration system for large-scale multi-host inference and training.

**Disaggregated Serving:**
Inference architecture separating prefill and decode stages across different slices for better resource utilization.

**Chunked Prefill:**
Breaking long prompts into smaller chunks for processing, reducing memory requirements during prefill phase.

**Batch Size (Inference):**
Number of concurrent sequences processed in inference.

**Throughput:**
Tokens generated per second in inference.

**Latency:**
Time from request to completion in inference.

**Beam Search:**
Decoding strategy exploring multiple hypotheses simultaneously.

**Sampling:**
Random token selection based on probability distribution.

**Temperature:**
Parameter controlling randomness in sampling (lower = more deterministic, higher = more random).

**Top-P (Nucleus) Sampling:**
Sampling from smallest set of tokens with cumulative probability >= p.

**Top-K Sampling:**
Sampling from the K tokens with highest probabilities.

**Greedy Decoding:**
Always selecting the token with highest probability, deterministic decoding strategy.

**Composite Sampling:**
Multi-stage sampling strategy combining top-k, top-p, and temperature-based weighted sampling.

---

## Configuration & Infrastructure

**base.yml:**
Main configuration file with default parameters for MaxText.

**Run Name:**
Unique identifier for a training run, used in logging and checkpointing.

**Model Name:**
Identifier for model architecture (llama3-8b, gemma2-27b, deepseek-v3, etc.).

**Dataset Type:**
Data loading backend choice (grain, tfds, hf, synthetic).

**Dataset Path:**
Location of training data for TFDS pipeline (`dataset_path` flag). For Grain, use `grain_train_files`. For HuggingFace, use `hf_path` or `hf_train_files`.

**Output Directory:**
Base directory where checkpoints and logs are saved.

**Steps:**
Total number of training steps to run.

**Checkpoint Period:**
Frequency of checkpoint saves (in steps).

**Logging Interval:**
Frequency of metric logging (in steps).

**Enable Checkpointing:**
Boolean flag to enable/disable checkpointing.

**Hardware Type:**
Target hardware (tpu, gpu, cpu).

**Compile Topology:**
Hardware layout specification for AOT compilation (e.g., `v5e-256`, `h100-128`).

**GCS (Google Cloud Storage):**
Cloud object storage for checkpoints and datasets.

**GKE (Google Kubernetes Engine):**
Kubernetes service for running distributed ML jobs.

**XPK (Accelerated Processing Kit):**
Tool for managing ML workloads on GKE with TPU/GPU support.

**TPU VM:**
Virtual machine with attached TPUs for training.

**Multi-Host:**
Setup with multiple physical machines connected via network.

**Single-Host:**
Setup on one physical machine with multiple accelerators.

**Worker:**
Context-dependent term with different meanings in MaxText. See Grain Worker, XPK Worker, and Pathways Worker for specific definitions.

**Grain Worker:**
Process for parallel data loading in the Grain data pipeline.

**XPK Worker:**
Kubernetes worker node running ML workloads in XPK clusters.

**Pathways Worker:**
Process executing computation on devices in Pathways orchestration.

**Replica:**
Copy of model/computation in data parallel setup.

**Rank:**
Unique ID assigned to each worker/process in distributed training (0 to num_workers-1). Used to coordinate operations across multiple accelerators.

**Shardy:**
XLA backend for SPMD sharding (default in JAX 0.7.0+), replacing GSPMD.

**GSPMD (General SPMD):**
Older XLA backend for SPMD compilation, being deprecated in favor of Shardy.

**Single Controller:**
Pathways feature allowing coordination of multiple subslices from a single controller process.

**Subslice:**
Portion of a full TPU pod/slice, e.g., 8x8 subgrid (64 chips) of a 16x16 trillium pod.

**Custom Mesh:**
User-defined device mesh topology for specialized parallelism strategies (e.g., hybrid_ring).

**SentencePiece:**
Tokenizer library supporting subword tokenization, used in Llama and Gemma models. [GitHub](https://github.com/google/sentencepiece)

**Tiktoken:**
OpenAI's tokenizer library, used in GPT models. [GitHub](https://github.com/openai/tiktoken)

**BOS (Beginning of Sequence):**
Special token marking the start of a sequence.

**EOS (End of Sequence):**
Special token marking the end of a sequence.

---

## Debugging & Monitoring

**XProf:**
Profiler for XLA programs, providing detailed performance analysis.

**TensorBoard:**
Visualization tool for training metrics, loss curves, and other logs.

**Perfetto:**
Performance tracing tool for low-level profiling.

**Stack Trace Collection:**
Capturing call stacks for debugging hangs and deadlocks.

**HLO Dump:**
Saving XLA computation graph (HLO) for analysis and debugging.

**Cloud Logging:**
Google Cloud log aggregation service for distributed job logs.

**Metrics Logger:**
System recording training metrics (loss, learning rate, step time, etc.).

**Step Time Deviation:**
Variance from ideal step time, indicating potential issues.

**Hang Detection:**
Detecting when workload stops progressing, often due to deadlocks or communication issues.

**Diagnostic Tools:**
Suite of debugging capabilities including stack traces, HLO dumps, profiling, etc.

---

## Additional Concepts

**AI (Arithmetic Intensity):**
Ratio of compute FLOPs to memory/communication bytes, determining if workload is compute-bound or memory-bound.

**Roofline Model:**
Performance model showing achievable performance based on arithmetic intensity and hardware limits.

**Compute Bound:**
Computation time exceeds communication/memory access time.

**Memory Bandwidth Bound:**
Communication or memory access dominates over computation.

**Flax:**
Neural network library built on JAX, providing high-level abstractions. [Documentation](https://flax.readthedocs.io/en/stable/)

**Linen:**
Original Flax API, being gradually replaced by NNX. [Documentation](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/index.html)

**Optax:**
JAX optimization library providing various optimizers and learning rate schedules. [Reference](https://github.com/google-deepmind/optax)

**Tunix:**
JAX library for post-training techniques (SFT, GRPO) integrated with MaxText. [GitHub](https://github.com/google/tunix)

**MoBA (Mixture of Block Attention):**
Technique that applies the principles of Mixture of Experts (MoE) to the attention mechanism, combining different attention patterns (full, sliding window) with learned routing for efficiency. [Paper](https://arxiv.org/abs/2502.13189)

**GDN (Gated Delta Net):**
Architecture component in Qwen3-Next using gated delta rule for sequence modeling with 1D convolution.

**Prefix Caching:**
Caching mechanism that reuses KV cache for common prompt prefixes across multiple requests.

**NNX:**
New Flax API with object-oriented interface, replacing the older Linen API. [Documentation](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)

**SafeTensors:**
Storage format for model weights providing safer and faster loading than pickle.

**Vertex AI Tensorboard:**
Google Cloud managed TensorBoard service for experiment tracking and visualization.

**Prometheus:**
Monitoring system for collecting and querying metrics, used in MaxEngine for serving metrics.

**XLA Flags:**
Configuration options for XLA compiler behavior, controlling optimizations and debugging output.

**Nsys:**
NVIDIA profiling tool for analyzing GPU performance.

**Stack Trace:**
Record of active function calls at a point in time, useful for debugging hangs and deadlocks.

**Checkify:**
JAX tool for runtime error checking and debugging.

**Heartbeat Metric:**
Periodic signal indicating workload is still running, used for monitoring job health.

---

For more information, please refer to:
- [MaxText Documentation](https://maxtext.readthedocs.io/)
- [MaxText GitHub Repository](https://github.com/AI-Hypercomputer/maxtext)
