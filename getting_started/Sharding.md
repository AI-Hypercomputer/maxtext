# Sharding
Choosing efficient sharding strategies is key to achieving good performance, especially at scale. In general there are other related knobs to optimize performance - you should make use of all your HBM (by tuning batch size and rematerialization policies), but here we discuss the various sharding strategies we support in maxtext.

When considering different sharding strategies, the main concern is the amount of communication executed between chips. Different sharding strategies will require different patterns of communication - how often communication is needed and the amount of data needed to communicate. A very helpful tool to understand the performance implications of these communications is **arithmetic intensity** - which roughly gives the ratio of useful computation to the communication cost. We highly recommend understanding arithmetic intensity if you are serious about optimizing performance - we recommend both the [“Jax Train your LLM”](https://jax-ml.github.io/scaling-book/sharding/) document and a MaxText HighPerformanceLLM [class](https://github.com/rwitten/HighPerfLLMs2024) (specifically classes 1-4). We briefly describe how to compute arithmetic intensities, and then explain the various sharding strategies we support in maxtext below, starting with some notation.

## Table of Contents

- [Sharding notation](#sharding-notation)
- [Arithmetic Intensity: whirlwind introduction example](#arithmetic-intensity-whirlwind-introduction-example)
- [Arithmetic Intensity: Mixed sharding strategies](#arithmetic-intensity-mixed-sharding-strategies)
- [Code implementation of sharding in MaxText](#code-implementation-of-sharding-in-maxtext)
- [Hierarchical Mesh](#hierarchical-mesh)
- [Data Parallelism (DP)](#data-parallelism-dp)
- [Fully Sharded Data Parallelism (FSDP)](#fully-sharded-data-parallelism-fsdp)
- [Fully Sharded Data Parallelism Transpose (FSDP Transpose)](#fully-sharded-data-parallelism-transpose)
- [Context Parallelism (CP)](#context-parallelism-cp)
- [Sequence Parallelism (SP)](#sequence-parallelism-sp)
- [Tensor Parallelism (TP)](#tensor-parallelism-tp)
- [Tensor Sequence Parallelism](#tensor-sequence-parallelism)
- [Tensor Parallelism Transpose (TP Transpose)](#tensor-parallelism-transpose-tp-transpose)
- [Expert Parallelism (EP)](#expert-parallelism-ep)
- [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
- [Context Autoregressive](#context-autoregressive)
- [Autoregressive](#autoregressive)

# Sharding notation

We illustrate our sharding notation with an example matmul:

$$B_xE  \times  EM = B_xM$$

Where B, E and M are names of dimensions and a subscript denotes sharding. For example, $B_xE$ is a 2-dimensional matrix sharded along the $B$ dimension, using the $x$ mesh axis. Dimensions without a subscript are not sharded.
This example is of standard data parallelism, only the batch dimension is sharded. Note that $B$ refers to the batch dimension, $B_x$ to the local shard of this dimension, whereas we use $\left|B\right|$ and $\left|B_x\right|$ to refer to the lengths of single axes, and $\left|x\right|$ as the degree of sharding on the x axis, e.g. $\left|B_x\right| = \left|B\right|/\left|x\right|$. We drop this $\left|\cdot\right|$ notation when there is a product to reduce clutter, e.g. we use $BEM_x$ instead of $\left|B\right|\left|E\right|\left|M_x\right|$.

We illustrate this notation on model parallelism as well:

$BM_x \times M_xE = BE \rightarrow \text{Reduce-Scatter (RS) over x} \rightarrow BE_x$

Explanation: Both the activations ($BM$) and weights ($ME$) are sharded on the M dimension. Thus each device is able to perform the matmul locally with its shard of the $M_x$ dimension, the local result is of the right global shape ($BE$) but is only a partial result - it needs to be summed with the other shards to get the full result. This is achieved with a reduce scatter (which does the summation and additionally shards the activations). Note that some flavors of tensor parallelism call for an all reduce instead a reduce scatter, but generally in maxtext we use a reduce scatter here.

## Axis labels
| Symbol | Description                                                                       |
| :----- | :-------------------------------------------------------------------------------- |
| $B$      | batch (either in tokens or sequences) |
| $S$      | sequence                                                                          |
| $E$      | emb_dim (aka model dim)                                                                           |
| $M$      | mlp_dim  (aka intermediate dim)                                                                         |
| $X$      | expert

Note for the feed forward computation the batch and sequence dimensions act the same and thus we use only one $B$ axis (which you can think of as a token batch dimension, a reshaping of batch and sequence into one axis), but for context and sequence parallelism they act differently and thus we use both a $B$ and $S$ dimension and the $B$ dimension is really batch in sequences. For example a matmul with an explicit sequence dimension might look like

$$BSE \times EM = BSM$$

But for arithmetic intensity roofline analysis purposes the $B$ and $S$ axis act as one, and generally we omit the $S$ axis except for when its needed (context/sequence parallelism), thus we only write

$$BE \times EM = BM$$

We recognize this overloads the definition of $B$ but for arithmetic intensity purposes the only batch size that matters is batch in tokens - which imagines combining the $B$ and $S$ axis into one.

# Arithmetic Intensity whirlwind introduction example

Arithmetic Intensity has a simple definition
```
Arithmetic Intensity:= Flops / Comms
```
We will see why this is a useful definition by walking through an example.

We want to be compute bound (because there is a fixed amount of compute to perform), which means we want the compute to take longer than the communication. Consider the above example (model parallelism aka tensor parallelism)

$$ BM_x \times M_xE = BE \text{ (partial result)} \rightarrow \text{RS over x} \rightarrow BE_x $$

The compute is $BM_x \times M_xE = BE$ matmul, which takes $2BM_xE$ flops (you can think of this as $\left|B\right| * \left|E\right|$ dot products each of length $\left|M_x\right|$, thus there are $BEM_x$ multiplications and additions to perform.

**Compute time** = Flops / compute speed = $2BEM_x$ / compute speed

The required communication is the RS from $BE$ to $BE_x$. It turns out an optimal reduce scatter algorithm in `bf16` would take $2BE$ bytes communicated per device

**Comm time** = comms bytes / comm speed = $2BE$ bytes / comm speed

We want to be compute bound, so we want:

```
Compute time > Communication time
Compute Flops / compute speed > Comm bytes / comm speed
```

Arithmetic Intensity simplifies and generalizes this analysis by re-arranging this inequality to put everything about the model on one side, and everything about the hardware on the other:

```
Compute Flops / Comm bytes > Compute Speed / comm speed
Operation Arithmetic Intensity > Hardware Arithmetic Intensity
```

The LHS (Compute Flops / Comm bytes) is the “Operation” or “Model” arithmetic intensity, whereas the RHS (Compute Speed / comm speed) is the hardware arithmetic intensity. This re-arrangement has a huge benefit in that it separates model from hardware - the operational intensity is independent of the hardware. Note however that arithmetic has this funky unity of flops/byte - intuitively you can think of this as the amount of flops unlocked by communicating a certain amount of bytes.

Operation Arithmetic Intensity for this example: $2BM_xE$ flops / $2BE$ bytes = $\left|M_x\right|$ flops/byte

Hardware Arithmetic Intensity: Compute speed / comm speed

Example hardware for trillium (See https://cloud.google.com/tpu/docs/v6e), compute speed = $917$ TFLOPs, and comm speed of 1 ICI axis is $180$ GB/s so the ratio $917 * 10^12 / 180 * 10^ 9 = 5100$. Thus we would need $\left|M_x\right| > 5100$ (Operational AI > Hardware AI) to be compute bound for this operation. This is an example of key insights that arithmetic intensity gives us - it tells us we need a large $\left|M\right|$ to achieve high utilization for model parallelism because the operational intensity is proportional to $\left|M\right|$.

# Arithmetic Intensity: Mixed sharding strategies

When we use multiple sharding strategies together it seems intractable to keep track of all of the compute vs communication ratios. However it turns out (not obvious at first), that the arithmetic intensity analysis of a “pure” sharding strategy generalizes to when it's used in a mix. For instance, if we added data parallelism to the above tensor parallelism example then  the batch dimension $B$ would also be sharded by a new mesh axes $y$. Both the compute and communication would decrease by this sharding factor $\left|y\right|$, and thus the ratio of compute to comms for tensor parallelism would remain the same ($\left|M\right|\left|x\right|$, independent of $\left|y\right|$). Concretely this would look like

$$B_yM_x \times M_xE = B_yE \rightarrow \text{RS over x } \rightarrow B_yE_x  $$

**Compute:** = $2B_yM_xE$ Flops

**TP comms (RS)** = $2B_yE$ bytes

**Ratio (Arithmetic Intensity)** = $\left|M_x\right|$ Flops/byte

This "independence" of sharding strategies is true for the main four parallelisms (data, model (tensor), pipeline, and expert). Note that data, fsdp, context and sequence parallelism are all roughly the same for the purpose of
arithmetic intensity analysis since they shard the batch, as we will illustrate in the individual sections below. In addition both data and pipeline parallelism (microbatches) shard the batch which decreases the HBM arithmetic intensity.

# Code implementation of sharding in MaxText

Sharding in maxtext is split into 3 layers

* **Physical** mesh axes (e.g. `data`, `fsdp`, `tensor`) defined [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/configs/base.yml#L269)

    * Mesh is created via [create_device_mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/max_utils.py#L576-L580)

    * Mesh given names in train.py via [Mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/train.py#L594)

* **Logical** axes which map a meaningful axes name to physical axes defined [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/configs/base.yml#L270)

    * E.g. logical axes `activation_batch` is sharded by the physical axes of `data` and `fsdp` (among others) since those sharding strategies shard the batch. `Activation_batch` is a common axis among most activation tensors. Note that if we use `data_parallelism=4` and `fsdp_parallelism=2`, then the `activation_batch` dimension will get sharded over both, e.g. $4*2=8$ ways.

* **Individual tensors** have sharding constraints - generally specified by logical rules

    * Example for weights using `kernel_axes` in `MlpBlock` [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/linears.py#L240) which in turns relies on flax’s param argument `nn.with_logical_partitioning` [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/linears.py#L135)

    * For activations we use `nn.with_logical_constraint` to give sharding hints for the compiler - here is an [example](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/llama2.py#L85). Sharding hints for the activations is not strictly necessary but the compiler may do funky/inefficient things without these hints.

# Hierarchical Mesh

Constructing a hierarchical mesh and specifying shardings is very similar to a “flat” mesh except we use the nice API [create_hybrid_device_mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/max_utils.py#L558) and specify both the degree of lower level faster network (e.g. `TPU ICI`) and higher level slower network (e.g. `DCN`) separately. For example if we want to use 4x fsdp parallelism over `ICI` and 2x data parallelism over `DCN` then we specify

```
mesh = mesh_utils.create_hybrid_device_mesh(
    (1,4), # (1 data, 4 fsdp) over ICI
    (2,1), # (2 data, 1 fsdp) over DCN
    devices,
)
```

For TPUs this two level hierarchy is (within-slice, across slices) using (ICI, DCN). For `v5e` and `trillium` there are at most 256 chips within a slice, whereas for `v4`, `v5p`, and the upcoming `ironwood` can span up to 8k/9k chips within a slice.

For GPUs this two level hierarchy is (within NVL domain, across NVL Domains) using (NVLink, DCN). Starting with  Grace Blackwell chips these NVL domains can span multiple hosts (e.g. 72 hosts or 576 chips). 

XLA will perform efficient hierarchical
collectives (all-gather, all-reduces, reduce-scatters) that communicate the minimal amount of information over the slower upper layer of the network. See the [Data Parallel Hierarchal Section](#dp-arithmetic-intensity-hierarchical) for an analysis of these communications.

# Data Parallelism (DP)

The simplest parallelization is data parallelization. Each chip works on a different batch of data, and the forward pass is embarrassingly parallel. No communication is needed in the forward pass. The gradients are synchronized in the backward pass (averaged or summed) - which is typically achieved with an all reduce.

## DP Arithmetic Intensity (Dense)

Roughly approximate the entire backward pass:

**Compute**: $4 * \text{local batch} * \text{params}$

We saw above that each matmul performs $2 * \text{local batch} * \text{params}$ flops, it turns out that the backward pass requires twice as many flops as the forward pass. We don't derive this here but highly recommend reading these [slides](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec6.pdf) from University of Toronto to explain the mathematics and implementation of backprop.

**Communicate**: All reduce size of params (`bf16`) : $4 * \text{params}$ (`2*` since `bf16`, another `2*` since an optimal all reduce algorithm turns out to require two passes of communicating data (generally a reduce scatter followed by an all-gather))

**Ratio (arithmetic intensity)**: `local_batch`

## DP Arithmetic Intensity (Sparse)

For an MoE architecture, we can imagine the `batch` axis is reshaped into `[batch_per_expert, expert]`, where the

`batch_per_expert` * `expert` = `batch` * `expert_per_token`

 e.g. the original activations have grown by a factor of `expert_per_token` and after reshaping the new batch axis is:
 
 `batch_per_expert` = `batch` * (`expert_per_token`/`expert`) = `batch` / `sparsity`

We denote the local `batch_per_expert` with $\beta$ and analyze an MoE feedfoward matmul to calculate arithmetic intensity:

$$\beta EX \times EMX = \beta MX$$

**Compute:** $4\beta EMX$ Flops (2x in backward pass)

**Comms:** All Reduce Gradient of size $EMX$: $4EMX$ bytes

**Ratio (arithmetic intensity):** $\left|\beta\right| = \text{local batch} / \text{sparsity}$ 

## DP Arithmetic Intensity (Hierarchical)

For a hierarchal mesh (TPU: within slice ICI, across slice DCN, GPU: within NVL domain, across NVL Domains), only one set of gradients need to be communicated
across the slower network per slice/NVL Domain (as opposed to one set per chip). This is generally achieved for us automatically by the XLA compiler:

Reduce Scatter grads on fast network $\rightarrow$ All Reduce across slow $\rightarrow$ All Gather on faster network

We can compute the arithmetic intensity of these cross slice/NVL Domain comms by imagining the chips forming a slice or NVL Domain as one "super chip". This "super chip" processes all of the tokens within its domain, but it only 
has to share one copy of the gradients to its super chip neighbors.

If the local per device batch size is `local batch`, then we can imagine each "super chip" has a batch of

`super batch` = `# devices per slice` * `local batch`

We can then perform the same arithmetic intensity analysis as before, and indeed get the same result:

**Compute (per super chip):** $4 * \text{super batch} * \text{params}$ flops

**Comms (per super chip):** All reduce params $\rightarrow 4 * \text{params}$ bytes

**Ratio (arithmetic intensity):** $\text{super batch } (\text{super batch} / \text{sparsity} \text{ for sparse models})$

This illustrates there are more than one way to calculate arithmetic intensity - we could also derive the same expression 
from the chip level as long as we are consistent for the compute and comms - either both the compute and comms should be at the super chip level, or both should be at the regular chip level.

# Fully Sharded Data Parallelism (FSDP)

Similar to data parallelism, except the model weights are also sharded to save memory. Generally the weights must get all-gathered before computation.

In addition to the weights all-gathering, the gradient communications are synchronized in the backward pass similar to DP (optimally will be synchronized with a reduce scatter which is 2x faster than an all-reduce, but only certain sizes of weight matrices allow for this efficient reduce scatter operation). The arithmetic intensity of this grad comm is thus either the same or 2x better than in the DP case, which has an arithmetic intensity of local_batch.

Fully sharded data parallelism (aka zero3) is used when the full model weights do not fit into HBM memory and thus they should be sharded as well. Generally we recommend using FSDP on TPU ICI or GPU NVLINK and DP across slices for TPUs or across hosts for NVLINK, although for large models even more FSDP may be required.

## FSDP Arithmetic Intensity

Approximate a typical weight @ activation = activation matmul:

Start with activations sharded like $B_xE$ and weights sharded like $E_xM$ (it doesn't matter which axis of weights is sharded). We must first All Gather (AG) the weights 

$$E_xM \rightarrow \text{AG } x \rightarrow  EM$$


**Compute**: $B_xE \times EM = B_xM$

This takes $2B_xEM$ flops

Note that $B$ is the global batch (unsharded), whereas $B_x$ is the `local_batch`.


**Communicate**: All gather params $EM$ in (`bf16`): $2EM$ bytes

**Ratio (arithmetic intensity)** $B_x$ = `local_batch` flops/byte (`local_batch` / `sparsity` for sparse)

The `sparsity` factor for sparse models shows up for the same reason as derived in the [DP Sparse Section](#dp-arithmetic-intensity-sparse)

*Note*: You may notice that in the DP arithmetic intensity we analyzed the *entire* backward pass whereas here we analyzed a single matmul. Both approaches should give the same answer, it is useful to understand both ways. Certain shardings are easier to analyze with a global view, whereas others are better analyzed with a local view, it is useful to practice switching between them.

# Fully Sharded Data Parallelism (transpose)

This is nearly identical to FSDP above except we choose to shard the main feedforward weights on the larger mlp dim instead of embed dim. This can be useful when the embed dim cannot be sharded further or does not have enough powers of 2 for efficient reduce scatter algorithms on TPUs. You may try swapping between `FSDP` and `FSDP_transpose`, their performance should be very similar, but one may offer a ~1% MFU improvement.

# Context Parallelism (CP)

Context parallelism is similar to FSDP except we shard the sequence dimension of activations instead of batch to allow for smaller batch dimensions (correspondingly smaller per device batch, including fractional per device batch sizes). A smaller per device batch dimension is often  needed for large sequence lengths so that the activations fit into memory. Also a smaller per device batch size is needed so that the global token count (global batch size) stays under some desired global batch size limit for optimal training - generally smaller global batch sizes can achieve better losses given a fixed number of total tokens (e.g. Llama3 used 16M global batch in tokens, DeepSeek uses 61M).

Care needs to be taken to shard the sequence dimension for attention - only the queries are sharded by sequence, the keys and values need to be all-gathered to perform the full computation. Additionally if we naively shard the sequence dimension then the attention computation is not evenly distributed due to the lower triangular causal mask - shards corresponding to later queries have more non-zero mask and thus become the bottleneck. Instead we “stripe” the inputs, so that the first shard has the first and last chunk of the sequence, the second shard has the second and second to last, etc. This striping is done on the initial data inputs (instead of every layer), so it is a small cost.

Note in general there are many flavors of CP such as ring attention, which in theory can hide all of the comms (as opposed to this implementation where the KV all gathers are probably exposed). This all gather is relatively cheap so we have implementd this flavor for now, a good trade-off of complexity and performance.

Currently Context Parallelism is only supported for GPUs (Sequence parallelism below is supported on TPUs). We plan to land context parallelism on TPUs shortly.

## CP Arithmetic Intensity

The main communications are the same as FSDP (all gather weights and synchronize gradients), with an arithmetic intensity of `local_batch` / `sparsity`.

The extra cost of all gathering of keys and values is small, especially for long sequences, analyzed below assuming group query attention:

**Compute**: Attention - `4 * batch * seq_len^2 * query_heads * head_dim/|CP|`

**Communicate (KV all gather)**: All-gather keys and values  - `4 * batch * seq_len * kv_heads * head_dim`

**Ratio**: `seq_len * query_heads / (kv_heads * |CP|)`

# Sequence Parallelism (SP)

Sequence parallelism is very similar to context parallelism - we shard the layer inputs and feed forward activations along the sequence dimension. The difference is for attention - we shard the queries, keys, and values along the head dimension instead of sequence dimension (this is fairly MaxText specific, you might not see this in other codebases). This is because the head dimension is easy to shard on for attention (it is not a contracting dimension), and thus can be more efficient than context parallelism as long as there are enough heads. Both sequence parallelism and tensor parallelism shard the heads, so we are constrained by `tensor_parallelism * sequence_parallelism < kv_heads`. E.g. if there are only 8 `kv_heads` as for llama3 and we use `tensor_parallelism=8`, then we cannot use any `sequence_parallelism` (e.g. `sequence_parallelism=1`)

Sequence parallelism is currently only supported with TPUs attention kernel, for GPUs we recommend context parallelism above.

## SP Arithmetic Intensity ##

The main communications are the same as `FSDP` (all gather weights and synchronize gradients), with an arithmetic intensity of `local_batch` / `sparsity`

### SP Extra A2A cost ###
Sequence parallelism has an additional cost of transferring the sharding from sequence to heads (and back again) for attention. This is executed via and all-to-all which are generally cheap operations, analyzed below:

**Compute**: Attention (`4 * batch * seq_len^2 * heads * head_dim \ |SP|`)

**Communicate:** A2A QKV activations and output activations (roughly `4 * batch * seq_len * heads * head_dim`)

**Ratio (Arithmetic Intensity)**: Proportional to `seq_len / |SP|`

The exact ratio depends on MHA vs GQA, how many kv heads there are and the efficiency of an all-to-all on the given hardware.

# Tensor Parallelism (TP)

Shard the activations along the feature dimensions (e.g. model or `embed` dimension and intermediate or `mlp` dimension) instead of the batch dimension. Tensor parallelism communicates the activations as opposed to the weights as in DP/FSDP. Tensor parallelism can be used to replace some amount of DP/FSDP when the batch size is small and/or when the model is large (when the `mlp` dim is large). Tensor parallelism is needed to run with small batches, such as fraction `per_device_batch_size` < 1. For instance if we use `TP=4` then we can use the rest with FSDP and set `per_device_batch_size=0.25` since the `global_batch = per_device_batch_size * TP * FSDP = 0.25 * 4 * FSDP = FSDP`, and this is shardable among `FSDP` devices (each device will get a shard of `FSDP/FSDP = 1` of the batch axis in this case). For the attention activations (query, key, value), we shard the heads on `TP` since that is the easiest dimension to shard on and use an attention kernel like flash attention (the heads are not a contracting dimension during the attention computation).

## TP Arithmetic Intensity

Analyze one pattern of TP as given above

$$ BM_x \times M_xE = BE \text{ (local partial result) } \rightarrow \text{ Reduce-Scatter (RS) } x \rightarrow BE_x $$

**Compute:** $2BM_xE$ Flops

**Communicate:** Reduce scatter  $BE$ (`bf16`): $2BE$ bytes

**Ratio (arithmetic intensity)**

$\left|M_x\right| = \left|M\right|/\left|TP\right|$

Note this is one pattern of TP where the contracting dimension is sharded. By contrast for the initial feed forward matmul the non-contracting weight dimension is sharded:

$$BE_x \times EM_x \rightarrow \text{AG activations over } x\rightarrow BE \times EM_x = BM_x$$

This is the same amount of compute, and also the same amount of communication - again activations of $BE$ are communicated, but in this case it is an initial all-gathering instead of secondary all-reduce. Ideally these activations (all-gather or reduce scatter) can be overlapped with the compute by the XLA compiler - an idea called a **collective matmul**. This is fairly challenging for the compiler since the comms and compute do depend on each other - to achieve overlap the computation and communication have to be chunked into smaller pieces and pipelined.

# Tensor Sequence Parallelism

This sharding strategy is very similar to tensor parallelism, except we shard the initial feed forward (FF) activations on the  sequence dimension as opposed to the model dimension. The activations have to get all-gathered at the start of the FF and reduce-scattered at the end, but it's the same amount of total comms, just a different axis (see above analysis for TP). The intermediate activations of shape [batch, sequence, mlp] are still sharded by mlp (since the weights are sharded on mlp). The benefits are explained in more detail in this [paper](https://arxiv.org/pdf/2205.05198), TL;DR is that all-reduces for small normalizations are not needed since the feature dimension is not sharded with `TP sequence` as opposed to when its sharded with regular `TP`. This is generally recommended for GPUs over tensor parallelism. See [PR #1136](https://github.com/AI-Hypercomputer/maxtext/pull/1136) which introduces this parallelism.

## Tensor Sequence Arithmetic Intensity

Near identical to tensor parallelism above except a different axis gets all-gathered and reduce-scattered on:  thus `MLP/TP`

# Tensor Parallelism Transpose (TP Transpose)

Similar to tensor parallelism, but instead of sharding the feed forward weights along the `mlp_dim`, shard them along the `embed_dim`. This will require communicating activations of the `mlp_dim` as opposed to the `embed_dim`, and thus is useful when the `mlp_dim` < `embed_dim` which is unusual but is true for some models such as DeepSeek V3.

`TP` and `TP transpose` can be used together called "2D TP" which can be more efficient than using purely one of them for inference decoding, although this is still a work in progress/largely untested.

## TP Transpose Arithmetic Intensity

This is really just swapping $E$ and $M$ of the TP analysis above, but we will include it here:

$$BE_x \times E_xM = BM_x$$

**Compute:** $2BE_xM$ FLOPS

**Communicate:** Reduce scatter  $BM$ (`bf16`): $2BM$ bytes

**Ratio (arithmetic intensity):** $\left|E_x\right|=\left|E\right|/\left|TP\right|$

# Expert Parallelism (EP)

Shard expert feed forward computation (both weights and activations) by expert!

The feedforward layer is the only one that has experts - for this layer we shard the weights and the activations on the experts dimensions by `EP`. For attention operations (including projections) the `EP` dimension acts like `FSDP`. This is the default choice by MaxText. There is an option for `EP` to act like `CP` in training. We may implement more options in the future where instead `EP` could act like `DP` or `SP` as well.

When using dropless strategies you may want to ensure that the shards are balanced. The balance can be improved by using less `EP` so that each shard is averaged over more experts. For instance imagine a scenario where expert 1 gets 10x more tokens routed to it than the rest. If `EP = # experts = 64`  than we will get terrible performance waiting for this one expert to finish its computation which is 3x slower. However if we set `EP = 1/4 * # experts` than the EP rank with expert 1 will have 4 experts, so we will have `3 + 1 + 1 + 1 = 6` compute to do compared to the average of `1 + 1 + 1 + 1 = 4`, a ratio of `6/4 = 1.5x` slower, which is a huge improvement over the `3x` slower.

## EP Arithmetic Intensity

An all-to-all (A2A) is needed to move between data sharding (fsdp) prior to the feed forward and the expert sharding during the feed forward. We denote $X$ as the expert tensor axis, and keep $x$ as the mesh axes

**Compute**

Analyze only 1 feed forward matmul

$$ BEX_x \times EMX_x = BMX_x $$

$$ 2BEX_x \text{ Flops} $$

**Communicate**

$$ B_xEX \rightarrow A2A \rightarrow BEX_x $$

Ideally this `A2A` only requires moving around $BEX_x$ elements per shard, but it depends on if the hardware is connected with an all to all network (true for `GPUs` and `TPU DCN` but not for `TPU ICI`)

With a true all-to-all network this takes $2BEX_x$ bytes. Over TPU ICI, an all-to-all is instead as costly as `1/4` of all gathering the entire activation as nicely drawn [here](https://jax-ml.github.io/scaling-book/sharding/#our-final-communication-primitive-the-alltoall) in jax's sharding doc.

**Ratio (arithmetic intensity)**: $2BEMX_x / 2BEX_x = \left|M\right|$

Note: The batch $B$ cancels in above arithmetic intensity - the batch dimension is present in both the compute and communication since we are communicating activations so cancels from the arithmetic intensity ratio regardless of how it is shaped (e.g.`batch` or `batch_per_exp`)

# Pipeline Parallelism (PP)

Shard the weights and computation by layers. There are many flavors of pipelining, MaxText current supports `gPipe` and `circular pipelines`, which are discussed below

## Why Pipeline Parallelism?

Pipeline parallelism is generally needed when the `per_device_batch` size is too small for data parallelism to be efficient. Recall above the arithmetic intensity of data parallelism is given by the `local_batch/sparsity`, so when this becomes too small then the communications associated with data parallelism will be very costly. This occurs either for very sparse models (e.g. DeepSeek), or when scaling to a large number of chips and maintaining a fixed global batch size (and thus the per device batch size is small).

## gPipe

gPipe style pipelining ([reference](https://arxiv.org/abs/1811.06965)) shards layers across stages, where each stage can have multiple layers. E.g. if there are four stages and twelve layers, stage 0 will perform layers 0, 1, and 2, then pass the results to stage 1 which will perform layers 3, 4, and 5, etc. Naively implemented this isn’t parallel since stage 1 has to wait for stage 0 to finish, however we can break the batch into microbatches to enable parallelism. E.g. as stage 1 works on microbatch 0, stage 0 can start working on a new microbatch 1. There is still a “bubble” - an amount of time each stage is idle while either waiting for the first microbatch or once it has finished all of its microbatches. This “bubble” time goes down with the amount of microbatches:

`Bubble = (PP - 1) / (Microbatches + PP - 1)`

## Circular Pipelining

Circular pipelining also shards layers across stages, but the layers “wrap” back around. E.g. if we have 24 layers, 4 stages, and 2 repeats, then stage 0 will perform layers 0, 1, 2 and also layers 12, 13, 14. Stage 1 will perform layers 3, 4, 5 and also 15, 16, 17 etc. This pattern helps to reduce the bubble: stage 1 is able to start its set of layers earlier (only need to wait for a microbatch to finish 3 layers instead of 6 since there are two repeats).

`Bubble = (PP - 1) / (repeats * Microbatches + PP - 1)`

There is a tradeoff of using many `repeats` - more `repeats` creates a schedule with a smaller bubble, however it also requires more `PP` comms between stages. The limiting case `repeats=1` is a gPipe schedule with minimal communication overhead, but maximal bubble. Ideally the `PP` comms are overlapped as long as there is enough compute, however achieving overlap is a challenging problem for the compiler. To break the data dependency of the circular transfer (last stage to first), the number of microbatches must exceed the number of stages, and thus we generally recommend `num_pipeline_microbatches = 2 * PP`.

## Other Pipeline Schedules

We are actively investing in Multiple Program Multiple Data (`MPMD`) style jax to support fancier pipeline schedules such as 1F1B and dualpipe which can achieve smaller bubbles while using less `PP` comms. Currently we only support `gPipe` and `circular pipelines`.

## PP + FSDP/DP

Pipelining and FSDP/DP interactions have to be considered together to achieve optimal performance. Generally we want to reduce the gradients across DP replicas only once outside of the pipeline loop as opposed to every microbatch (we want the gradient reduction performed locally across microbatches first and only once across DP replicas). We rely on the XLA compiler for this optimization. Similarly for FSDP we want to all-gather the weights across FSDP only once before the pipeline loop as opposed to every microbatch - we have implemented this in maxtext with `pipeline_fsdp_ag_once` and generally recommend this with small batch sizes. However this comes with a huge memory cost - the weights and gradients are not sharded by FSDP, and thus a significant amount of other sharding (PP, EP, TP) must be used. This is roughly equivalent  0-1 sharding, FSDP only shards the optimizer state, not the weights and gradients.

## PP Arithmetic Intensity

The arithmetic intensity is a bit harder to define for PP, and depends on the pipeline flavor. We analyze the circular pipeline below.

**Compute**

One stage worth. A stage can consist of multiple layers, if `layers_per_pipeline_stage > 1`. Each layer generally is a combination of a fully connected feed forward block and an attention block. Let's ignore attention since it's generally significantly smaller than the `FF` (for sequence length of `8k`). A typical `FF` has 3 matmuls (2 in for silu, 1 out), for a total of $6BME$. Thus there are `layers_per_pipeline_stage * 6 * B * M * E` flops

**Communicate**

The layer outputs between stages of size $BE$. These are collectively permuted (stage 0 &rarr; 1 &rarr; 2 &rarr; 3 &rarr; 0). Our current implementation of pipelining also rotates the inputs to stage 0 around so there are two collective permutes per stage, so $4BE$ bytes per stage.

**Ratio (arithmetic intensity)**

`3/2 * layers_per_pipeline_stage * M * experts_per_token`

Note that for MoE models, this arithmetic intensity grows by a factor of `experts_per_token` since the compute grows by this factor, but the communication is independent of this factor.

# Context Autoregressive

Context Autoregressive shards the KV cache on the sequence dimension. It shards feed forward layer by experts for both activations and weights. This is used for inference only, see [inference.yml](https://github.com/AI-Hypercomputer/maxtext/blob/353a45d57eb1f1cc02e5c8d9e7b18eaf634d7edc/MaxText/configs/inference.yml#L4) for the modified logical axis rules for inference.

# Autoregressive

Autoregressive shards weights, but not activations. This is used for inference only. See [inference.yml](https://github.com/AI-Hypercomputer/maxtext/blob/353a45d57eb1f1cc02e5c8d9e7b18eaf634d7edc/MaxText/configs/inference.yml#L4) for the modified logical axis rules for inference.