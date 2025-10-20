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

(performance-metrics)=
# Performance metrics

## MFU

Model Flops Utilization (MFU) is one of the most commonly used metrics to summarize performance, along with step time or tokens per second.

### Definition

Model FLOPs are the floating point operations required to perform model computations regardless of implementation or hardware limitations. 
For training, this corresponds to the operations in a single forward and backward pass (one model step).

$$ MFU:= \frac{\text{model flops/s}}{\text{peak hardware flops/s}} $$

Model flops are generally easy to calculate/estimate theoretically since the model is mostly performing matmuls, and so we can just sum up the flops of each matmul. For example a $[A,B] \times [B,C] = [A,C]$ matmul has $2ABC$ flops. Hence, to calculate the observed model flops/s we can sum up the theoretical flops required in a training step of the model and then divide by the measured step time (in seconds).

 $$ MFU = \frac{\text{model flops/s}}{\text{peak hardware flops/s}} = \frac{\text{theoretical model flops per step}}{\text{measured step time} \times \text{peak hardware flops/s}}$$

Furthermore, since

$$
\frac{\text{theoretical model flops per step}}
     {\text{peak hardware flops/s}}
= \text{theoretically optimal step time}
$$

we also get that:

$$
MFU = \frac{\text{theoretically optimal step time}}
           {\text{measured step time}}
$$

Finally, we can also look at throughput utilization. In each training step, the model processes $(batch_size x seq_length)$ tokens. Since the (optimal or measured) number of tokens per second is just the number of tokens per step divided by step time (optimal or measured, respectively), we get that:

$$
MFU = \frac{\text{theoretically optimal step time}}
           {\text{measured step time}} = \frac{\text{number of tokens per step / optimal tokens/s}}
           {\text{number of tokens per step / measured tokens/s}} = \frac{\text{measured tokens/s}}
           {\text{optimal tokens/s}}
$$

Hence, MFU is the fraction of peak hardware performance actually utilized by the model, and can be expressed in different units — step time, throughput, or raw flops/s.

### MaxText calculating + reporting
In MaxText, we sum all of the matmuls performed in one step, see [calculate_tflops_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/fafdeaa14183a8f5ca7b9f7b7542ce1655237574/src/MaxText/maxtext_utils.py#L454)
and divide it by the measured (via python `time.time()`) step time. In each step we print the resulting Model Flops per second [`per_device_tflops_per_sec`](https://github.com/AI-Hypercomputer/maxtext/blob/fafdeaa14183a8f5ca7b9f7b7542ce1655237574/src/MaxText/metric_logger.py#L211-L213). One can calculate the MFU by dividing this number by the peak tflops of the hardware (e.g., $918e^{12}$ FLOPS/s for Trillium).

### Causal attention
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
* Clearly shows room left to improve, how much more the hardware is capable of. (e.g. 25% MFU means it's possible to get 4x more performance and 4x smaller step times). Note that achieving 100% is not practical due to many factors, but MFU score effectively shows how much room is left for optimization.
* Generalizable across hardwares, model, configs (e.g. batch sizes)

**Cons**
* Care needs to be token to compare MFU across codebases that the model flops calculation are identcail (e.g. was causality taken into account in both code bases?)

Step time, tokens/s, and MFU all can be used to calculate how long training will take (e.g. how long will it take to train my model on $T$ tokens given $C$ chips?)

$$\begin{align*}
\text{training time} &= \text{step time} \times \text{num steps} \\
                     &= \frac{T tokens}{\text{measured tokens per second per chip} \times C} \\
                     &= \frac{\text{theoretical flops to train T tokens}}{\text{MFU} \times C \times \text{chip peak speed}}
\end{align*}$$

This shows any of step time, tokens/s or MFU can be used to determine how long training will take and are proportionally (or inversely proportionally) related. MFU is most useful to compare across different models/hardwares and while optimizing performance, whereas step time or tokens/second may be more useful when these are fixed.

## Why not hardware flops?

Hardware (e.g., XLA reported) FLOPs do not accurately reflect computation efficiency as they depend on the program / implementation, not just on the model and its inherent computations (higher hardware FLOPs does not necessarily mean less room for improvement). For example, they include remat and potentially auxilliary operations (such as reshaping for dropping moe [here](https://github.com/AI-Hypercomputer/maxtext/blob/fafdeaa14183a8f5ca7b9f7b7542ce1655237574/src/MaxText/layers/moe.py#L1544)), which are an implementation detail and not part of the model. In addition, XLA reported FLOPs may not be accurate with pallas kernels. Hardware flops utilization is not (inversely) proportional to step time as opposed to MFU, since hardware flops can change with implementation details like remat policies.
