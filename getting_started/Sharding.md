Understanding sharding strategies is key to acheiving good performance, especially at scale. In general there are other related knobs to optimize performance - you should make use of all your HBM (by tuning batch size and rematerialization policies), but here we discuss the various sharding strategies we support in maxtext.

When considering different sharding strategies, the main concern is the amount of communication executed between chips. Different sharding strategies will require different patterns of communication - how often communication is needed and the amount of data needed to communicate. A very helpful tool to understand the performance implications of these communications is arithmetic intensity - which roughly gives the ratio of useful computation to the communication cost. We highly recommend understanding arithmetic intensity if you are serious about optimizing performance - we recommend both the [“Jax Train your LLM”](https://jax-ml.github.io/scaling-book/sharding/) document and a MaxText HighPerformanceLLM [class](https://github.com/rwitten/HighPerfLLMs2024) (specifically classes 1-4). We briefly describe how to compute arithmetic intensities, and then explain the various sharding strategies we support in maxtext below, starting with some notation:

# Sharding notation: 
We illustrate our sharding notation with an example matmul:

B<sub>x</sub>E @ EM = B<sub>x</sub>M

This denotes that the Batch axis `B` is sharded on the mesh axes `x`, whereas the other dimensions are not sharded. This example is of standard data parallelism, only the batch dimension is sharded. We illustrate this notation on model parallelism as well:

BM<sub>x</sub> @ M<sub>x</sub>E = BE (local partial result) -> Reduce-Scatter (RS) over x -> BE<sub>x</sub>

Explanation: Both the activations (`BM`) and weights (`ME`) are sharded on the M dimension. Thus each device is able to perform the matmul locally with its shard of the M<sub>x</sub> dimension, the local result is of the right global shape (`BE`) but is only a partial result - it needs to be summed with the other shards to get the full result. This is achieved with a reduce scatter (which does the summation and additionally shards the activations).

# Arithmetic Intensity whirlwind introduction example:
Arithmetic intensity is a key tool for understanding performance. We want to be compute bound (because there is a fixed amount of compute to perform), which means we want the compute to take longer than the communication. Consider the above example (model parallelism aka tensor parallelism)

BM<sub>x</sub> @ M<sub>x</sub>E = BE (partial result) -> RS over x -> BE<sub>x</sub>

The compute is BM<sub>x</sub> @ M<sub>x</sub>E = BE matmul, which takes `2 * B * M_x * E` flops (you can think of this as `B * E` dot products each of length `M_x`, thus there are `B * E * M_x` multiplications and additions to perform.

**Compute time** = Flops / compute speed = `2 * B * E * M_x` / compute speed

The required communication is the RS from `BE` to `BE_x`. It turns out an optimal reduce scatter algorithm in bf16 would take `BE * 2` bytes communicated per device 

**Comm time** = comms bytes / comm speed = `2 * B * E` bytes / comm speed

We want to be compute bound, so we want 

```
Compute time > Communication time
Compute Flops / compute speed > Comm bytes / comm speed
```

Arithmetic Intensity simplifies and generlizes this analysis by re-arranging this inequality to put everything about the model on one side, and everything about the hardware on the other: 
```
Compute Flops / Comm bytes > Compute Speed / comm speed
Operation Arithmetic Intensity > Hardware Arithmetic Intensity
```

The LHS (Compute Flops / Comm bytes) of this is the “Operation” or “Model” arithmetic intensity, whereas the RHS (Compute Speed / comm speed) is the hardware arithmetic intensity. This re-arrangement has a huge benefit in that it separates model from hardware - the operational intensity is independent of the hardware. Note however that arithmetic has this funky unity of flops/byte - intuitvely you can think of this as the amount of flops unlocked by communicating a certain amount of bytes.

Operation Arithmetic Intensity for this example: `2 * B * M_x * E` flops / `2 * B * E` bytes = `M_x`

Hardware Arithmetic Intensity: Compute speed / comm speed

Example hardware for trillium, compute speed = `917` TFLOPs, and comm speed of 1 ICI axis is `180` GB/s so the ratio `917 * 10^12 / 180 * 10^ 9 = 5100`. Thus we would need `M_x > 5100` (Operational AI > Hardware AI) to be compute bound for this operation (Note `M_x = M/|x|`, the degree of sharding). This is an example of key insights that arithmetic intensity gives us - it tells us we need a large `M` dimension to achieve high utilization for model parallelism because the operational intensity is proportaionl to `M`.

# Arithmetic Intensity: Mixed sharding strategies
When we use multiple sharding strategies together it seems intractable to keep track of all of the compute vs communication ratios. However it turns out (not obvious at first), that the arithmetic intensity analysis of a “pure” sharding strategy generalizes to when it's used in a mix. For instance if we added data parallelism to the above tensor parallelism example then  the batch dimension `B` would also be sharded by a new mesh axes `y`. Both the compute and communication would decrease by this sharding factor `|y|`, and thus the ratio of compute to comms for tensor parallelism would remain the same (`M/|x|`, independent of `y`). Concretely this would look like

B<sub>y</sub>M<sub>x</sub> @ M<sub>x</sub>E = B<sub>y</sub>E &rarr; RS over x &rarr; B<sub>y</sub> E<sub>x</sub>   

**Compute** = `2 * B_y * M_x * E` Flops

**TP comms (RS)** = `2 * B_y * E` bytes

**Ratio (Arithmetic Intensity)** = `M_x`


# Code implementation of sharding in MaxText
Sharding in maxtext is split into 3 layers

* **Physical** mesh axes (e.g. `data`, `fsdp`, `tensor`) defined [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/configs/base.yml#L269)
  * Mesh is created via [create_device_mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/max_utils.py#L576-L580)
  * Mesh given names in train.py via [Mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/train.py#L594)
* **Logical** axes which maps a semantically meaning axes name to physical axes defined [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/configs/base.yml#L270)
  * E.g. logical axes `activation_batch` is sharded by the physical axes of `data` and `fsdp` (among others) since those sharding strategies shard the batch. `Activation_batch` is a common axis among most activation tensors. Note that if we use `data_parallelism=4` and `fsdp_parallelism=2`, then the `activation_batch` dimension will get sharded over both, e.g. `4*2=8` ways.
* **Individual tensors** have sharding constraints - generally specified by logical rules
  * Example for weights using `kernel_axes` in `MlpBlock` [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/linears.py#L240) which in turns relies on flax’s param argument `nn.with_logical_partitioning` [here](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/linears.py#L135)
  * For activations we use `nn.with_logical_constraint` to give sharding hints for the compiler - here is an [example](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/layers/llama2.py#L85). Sharding hints for the activations is not strictly necessary but the compiler may do funky/inefficient things without these hints. 

# Hierarchical Mesh (multislice for TPUs, multihost for GPUs)
Constructing a hierarchical mesh and specifying shardings is very similar to a “flat” mesh except we use the nice API [create_hybrid_device_mesh](https://github.com/AI-Hypercomputer/maxtext/blob/f269268bd622f6d2f40d38632ede7a7834a6024e/MaxText/max_utils.py#L558) and specify both the degree of DCN and ICI separately. E.g. if we want to use 2x data parallelism over DCN and 4x fsdp parallelism over ICI then we we specify 
```
      mesh = mesh_utils.create_hybrid_device_mesh(
          (1,4), # (1 data, 4 fsdp)
          (2,1), # (2 data, 1 fsdp)
          devices,
      )
```

# Data Parallelism (DP)

The simplest parallelization is data parallelization. Each chip works on a different batch of data, and the forward pass is embarrassingly parallel. No communication is needed in the forward pass. The gradients are synchronized in the backward pass (averaged or summed) - which is typically achieved with an all reduce.

## DP Arithmetic Intensity

Roughly approximate the entire backward pass:

**Compute** `4 * local_batch * params`

We saw above that each matmul performs `2 * batch * params` flops, in turns out that the backward pass requires twice as many flops as the forward pass. We don't derive this here but highly recommend reading these [slides](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec6.pdf) from university of tornto to explain the mathematics and implementation of backprop

**Communicate**: All reduce size of params (`bf16`) : `4 * params` (`2*` since `bf16`, another `2*` since an optimal all reduce algorithm turns out to require two passes of communicating data)

**Ratio (arithmetic intensity)**: `local_batch`

Note: this analysis ignores attention however this approximation `local_batch` is still accurate for attention (would require a separate analysis to see why)

# Fully Sharded Data Parallelism (FSDP)
Similar to data parallelism, except the model weights are also sharded to save memory. Generally the weights must get all-gathered before computation.

In addition to the weights all-gathering the gradient communications are synchronized in the backward pass similar to DP (optimally will be synchronized with a reduce scatter which is 2x faster than an all-reduce, but only certain sizes of weight matrixes allow for this efficient reduce scatter operation). The arithmetic intensity of this grad comm is thus either the same or 2x better than in the DP case, which has an arithmetic intensity of local_batch.

Fully sharded data parallelism (aka zero3) is used when the full model weights do not fit into HBM memory and thus they should be sharded as well. Generally we recommend using FSDP on TPU ICI or GPU NVLINK and DP across slices for TPUs or across hosts for NVLINK (for sparse models anyway)

 ## FSDP Arithmetic Intensity:

 Approximate a typical weight @ activation = activation matmul:
 
 Start with activations sharded like `B_xE` and weights sharded like `E_xM` (it doesn't matter which axis of weights is sharded). We must first all gather the weights
 `E_xM` &rarr; AG over `x` `EM`. Note that `B` is the global batch (unsharded), whereas `B_x` is the `local_batch`.

 **Compute**: B<sub>x</sub>E @ EM = B<sub>x</sub>M
`2 * B_x * E * M` flops

 **Communicate**: All gather params `EM` in (`bf16`): `2 * EM` bytes
 
 **Ratio (arithmetic intensity)** `B_x` = `local_batch` flops/byte

 *Note*: You may notice that in the DP arithmetic intensity we analyzed the *entire* backward pass whereas here we analzed a single matmul. Both approaches should give the same answer, it is useful to understand both ways. Certain shardings are easier to analyze with a global view, whereas others are better analyzed with a local view, it is useful to practice switching between them. The local view is closer to what is really happening on the chip, but is sometimes harder to analyze.

# Fully Sharded Data Parallelism (transpose)
This is nearly identical to FSDP above except we choose the shard the main feedforward weights on the larger mlp dim instead of embed dim. This can be useful when the embed dim cannot be sharded further or does not have enough powers of 2 for efficient reduce scatter algorithms on TPUs. You may try swapping between `FSDP` and `FSDP_transpose`, their performance should be very similar, but one may offer a ~1% MFU improvement.

# Context Parallelism (CP)
Context parallelism is similar to FSDP except we shard the sequence dimension of activations instead of batch to allow for smaller batch dimensions (correspondingly smaller per device batch, including fractional per device batch sizes). A smaller per device batch dimension is often  needed for large sequence lengths so that the activations fit into memory. Also a smaller per device batch size is needed so that the global token count (global batch size) stays under some desired global batch size limit for optimal training - generally smaller global batch sizes can achieve better loses given a fixed number of total tokens (e.g. Llama3 used 16M global batch in tokens, DeepSeek uses 61M).

Care needs to be taken to shard the sequence dimension for attention - only the queries are sharded by sequence, the keys and values need to be all-gathered to perform the full computation. Additionally if we naively shard the sequence dimension than the attention computation is not evenly distributed due to the lower triangular casual mask - shards corresponding to later queries have more non-zero mask and thus become the bottleneck. Instead we “stripe” the inputs, so that the first shard has the first and last chunk of the sequence, the second shard has the second and second to last, etc. This striping is done on the initial data inputs (instead of every layer), so it is a small cost.

Currently Context Parallelism is only supported for GPUs (Sequence parallelism below is supported on TPUs). We plan to land context parallelism on TPUs shortly.

## CP Arithmetic Intensity: 
The main communications are the same as FSDP (all gather weights and synchronize gradients), with an arithmetic intensity of `local_batch`.

The extra cost of all gathering of keys and values is small, especially for long sequence, analyzed below assuming group query attention:

**Compute**: Attention (`4 * batch * seq_len^2 * query_heads * head_dim`)

**Communicate (KV all gather)**: All-gather keys and values  (`4 * batch * seq_len * kv_heads * head_dim`)

**Ratio**: `seq_len * query_heads / kv_heads`

# Sequence Parallelism
Sequence parallelism is very similar to context parallelism - we shard the layer inputs and feed forward activations along the sequence dimension. The difference is for attention - we shard the queries, keys, and values along the head dimension instead of sequence dimension. This is because the head dimension is easy to shard on for attention (it is not a contracting dimension), and thus can be more efficient than context parallelism as long as there are enough heads. Both sequence parallelism and tensor parallelism shard the heads, so we are constrained by `tensor_parallelism * sequence_parallelism < kv_heads`. E.g. if there are only 8 `kv_heads` as for llama3 and we use `tensor_parallelism=8`, then we cannot use any `sequence_parallelism` (e.g. `sequence_parallelism=1`) 

Sequence parallelism is currently only supported with TPUs attention kernel, for GPUs we recommend context parallelism above.

**Arithmetic Intensity**

The main communications are the same as `FSDP` (all gather weights and synchronize gradients), with an arithmetic intensity of `local_batch`

Sequence parallelism has an additional cost of transfering the sharding from sequence to heads (and back again) for attention. This is executed via and all-to-all which are generally cheap operations, analyzed below:
The extra all-to-all as an arithmetic intensity propriotnal to sequence dimension:

**Compute**: Attention (`4 * batch * seq_len^2 * heads * head_dim`)

**Communicate** (All-to-all): all-to-all qkv activations and output activations (roughly `4 * batch * seq * heads * head_dim`)

**Ratio (Arithmetic Intensity)**: Proprtional to `seq_len`
The exact ratio depends on MHA vs GQA, how many kv heads there are and the efficiency of an all-to-all on the given hardware.