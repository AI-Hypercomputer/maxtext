# Performance Metrics

## MFU

Model Flops Utilization (MFU) is one of the most commonly used metrics to summarize performance, along with step time or tokens per second.

### Definition




$$ MFU:= \frac{\text{model flops/s}}{\text{peak hardware flops/s}} $$




Model flops are the theoretical number of flops needed to execute the computation (both forward and backward passes for training). This is generally easy to calculate/estimate theoretically since the model is mostly performing matmuls, and so we can just sum up the flops of each matmul. For example a $[A,B] \times [B,C] = [A,C]$ matmul has $2ABC$ flops.

### MaxText Calculating + Reporting 
In MaxText, we calculate MFU by calculating the theoretical number of flops and measuring step time.

We calculate theoretical model flops per step as described above, summing all of the matmuls needed in detail, see [calculate_tflops_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/maxtext_utils.py#L297)
We divide this by the measured step time (measured via python `time.time()`), to get the Model Flops per second that we print each step [`per_device_tflops_seconds`](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/metric_logger.py#L193-L194). You can get the MFU by dividing this number by the peak tflops of the hardware (e,g. $918e^{12}$ FLOPS/s for Trillium)

 $$ MFU = \frac{\text{model flops/s}}{\text{peak hardware flops/s}} = \frac{\text{theoretical model flops}}{\text{measured step time} \times \text{peak hardware flops/s}}$$

This is also equivalent to the ratio of the theoretically optimal step time and measured step time:

$$\text{theoretically optimal step time} = \frac{\text{theoretical model flops}}{\text{peak hardware flops/s }} $$

$$\frac{\text{theoretically optimal step}}{\text{measured step time}} = \frac{\text{theoretical model flops}}{\text{measured step time} \times \text{peak hardware flops/s }} = MFU$$


### Causal Attention
Due to causality only half of the (query, key) pairs need to be computed, those with query_idx >= key_idx. This accounts for the fact only prior tokens can be used to predict future ones. Prior to https://github.com/AI-Hypercomputer/maxtext/pull/1988 MaxText did not account for sparsity for theoretical flops, and used

Attention Flops ~= 4 * sequence^2 * batch * heads * head_dim


When accounting for causal masking, this should be halved

Attention Flops ~= 2 * sequence^2 * batch * heads * head_dim

Which maxtext now uses since this [PR/1988](https://github.com/AI-Hypercomputer/maxtext/pull/1988)

Note that

$$ \text{Total Flops} =  \text{Attention (quadratic in sequence) + Non-attention  (linear)}$$ 

Thus the distinction between causal vs non causal flops is particularly important for long sequence when the attention flops dominate / are a significant fraction of the total flops. For 8k sequence length, the attention flops are generally around 10% of total flops (depending on exact model dims), whereas for 128k seq, the attention flops may be around 90%. Note however the attention flops also vary by attention type, e.g. sliding window flops are not quadratic in sequence, but are only linear in both sequence length and window length. We updated our model flops calculation to account for sliding window attention and chunked attention in [PR 2009](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR 2030](https://github.com/AI-Hypercomputer/maxtext/pull/2030).



### Why MFU
MFU is a very useful metric to understand your systems performance, but like step time or tokens/s, there are pros and cons of summarizing the system’s performance to a single number.

**Pros**
* Clearly shows room left to improve, how much more the hardware is capable of. (e.g. 25% MFU means its possible to get 4x more performance and 4x smaller step times). Note that achieving 100% is not practical due to many factors, but MFU score effectively shows how much room is left for optimization.
* Generalizable across hardwares, model, configs (e.g. batch sizes)

**Cons**
* Care needs to be token to compare MFU across codebases that the model flops calculation are identcail (e.g. was causality taken into account in both code bases?)

Step time, tokens/s, and MFU all can be used to calculate how long training will take (e.g. how long will it take to train my model on $T$ tokens given $C$ chips?)



$$\begin{align*}
\text{training time} &= \text{step time} \times \text{num steps} \\
                     &= \frac{T tokens}{\text{measured tokens per second per chip} \times C} \\
                     &= \frac{\text{theoretical flops to train T tokens}}{\text{MFU} \times C \times \text{chip peak speed}}
\end{align*}$$


This shows any of step time, tokens/s or MFU can be used to determine how long training will take and are proportionally (or inversely prortionally) related. MFU is most useful to compare across different models/hardwares and while optimizing performance, wheres step time or tokens/second may be more useful when these are fixed.

## Why not Hardware Flops?

Hardware (e.g., XLA reported) FLOPs do not accurately reflect computation efficiency as they depend on the program / implementation, not just on the model and its inherent computations (higher hardware FLOPs does not necessarily mean less room for improvement). For example, they includes remat and potentially auxilliary operations (such as reshaping for dropless moe [here](https://github.com/AI-Hypercomputer/maxtext/blob/4b6142950aff5d9ba42d830efc5ce4c4ac9d4135/MaxText/layers/moe.py#L1267)), which are an implementation detail and not part of the model. In addition, XLA reported FLOPs may not be accurate with pallas kernels. Hardware flops utilization is not (inversely) proprtional to step time as opposed to MFU, since hardware flops can change with implementation details like remat policies. 