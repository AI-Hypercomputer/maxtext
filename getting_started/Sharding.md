Understanding sharding strategies is key to acheiving good performance, especially at scale. In general there are other related knobs to optimize performance - you should always make use of all your HBM (batch size, rematerialization policy), but here we discuss the various sharding strategies we support in maxtext.

When considering different sharding strategies, the main concern is the amount of communication executed between chips. Different sharding strategies will require different patterns of communication - how often communication is needed and the amount of data needed to communicate. A very helpful tool to understand the performance implications of these communications is arithmetic intensity - which roughly gives the ratio of useful computation to the communication cost. We highly recommend understanding arithmetic intensity if you are serious about optimizing performance - we recommend both the l [“Jax Train your LLM”](https://jax-ml.github.io/scaling-book/sharding/) and a MaxText HighPerformanceLLM [class](https://github.com/rwitten/HighPerfLLMs2024) (specifically classes 1-4). We briefly describe how to compute arithmetic intensities, and then explain the various sharding strategies we support in maxtext below, starting with some notation:

# Sharding notation: 
We illustrate our sharding notation with an example:

B<sub>x</sub>E @ EM = B<sub>x</sub>M

This denotes that the Batch axis `B` is sharded on the mesh axes `x`, whereas the other dimensions are not sharded. This example is of standard data parallelism, only the batch dimension is sharded. We illustrate this notation on model parallelism as well:

BM<sub>x</sub> @ M<sub>x</sub>E = BE (partial result) -> Reduce-Scatter (RS) over x -> BE<sub>x</sub>

Explanation: Both the activations (`BM`) and weights (`ME`) are sharded on the M dimension. Thus each device is able to perform the matmul locally with its shard of the M<sub>x</sub> dimension, the resultant local result is of the right global shape (`BE`) but is only a partial result - it needs to be summed with the other shards to get the full result. This is achieved with a reduce scatter (which does the summation and additionally shards the activations).

# Arithmetic Intensity whirlwind introduction example:
Arithmetic intensity is a key tool for understanding performance. We want to be compute bound (because there is a fixed amount of compute to perform), which means we want the compute to take longer than the communication. Consider the above example (model parallelism aka tensor parallelism)

BM<sub>x</sub> @ M<sub>x</sub>E = BE (partial result) -> RS over x -> BE<sub>x</sub>

The compute is BM<sub>x</sub> @ M<sub>x</sub>E = BE matmul, which takes `2 * B * M_x * E` flops (you can think of this as `B * E` dot products each of length `M_x`, thus there are `B * E * M_x` multiplications and additions to perform.

**Compute time** = Flops / compute speed = `2 * B * E * M_x` / compute speed

The required communication is the RS from BE to BE_x. It turns out an optimal reduce scatter algorithm in bf16 would take `BE * 2` bytes communicated per device 

**Comm time** = comms bytes / comm speed = `2 * B * E` bytes / comm speed

We want to be compute bound, so we want 

```
Compute time > comm time
Compute Flops / compute speed > Comm bytes / comm speed
```

Arithmetic Intensity simplifies and generlizes this analysis by re-arranging this inequality to put everything about the model on one side, and everything about the hardware on the other: 
```
Compute Flops / Comm bytes > Compute Speed / comm speed
Operation Arithmetic Intensity > Hardware Arithmetic Intensity
```

The LHS (Compute Flops / Comm bytes) of this is the “Operation” or “Model” arithmetic intensity, whereas the RHS (Compute Speed / comm speed) is the hardware arithmetic intensity. This re-arrangement has a huge benefit in that it separates model from hardware - the operational intensity is independent of the hardware.

Operation Arithmetic Intensity: `2 * B * M_x * E` flops / `2 * B * E` bytes = `M_x`
Hardware Arithmetic Intensity: Compute speed / comm speed
Example hardware for trillium, compute speed = 917 TFLOPs, and comm speed of 1 ICI axis is 180 GB/s so the ratio `917 * 10^12 / 180 * 10^ 9 = 5100`. Thus we would need `M_x > 5100` to be compute bound for this operation (Note `M_x = M/|x|`, the degree of sharding). This is an example of key insights that arithmetic intensity gives us - it tells us we need a large `M` dimension to achieve high utilization for model parallelism.
