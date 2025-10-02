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

# Quantization

MaxText supports quantization via both the [AQT](https://github.com/google/aqt) and [Qwix](https://github.com/google/qwix) libraries. Qwix is the recommended approach, providing a non-intrusive way to apply various quantization techniques, including Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).

## Why quantize?

*   **Reduced model size**: Lower precision numbers require less storage, making models easier to store and deploy.
*   **Faster inference**: Operations on lower-precision data are computationally less expensive, which can lead to faster inference times.
*   **Lower memory usage**: Reduced precision for weights and activations decreases the memory footprint, allowing for the deployment of larger models on hardware with limited memory.

## Quantizing using AQT

Jax supports AQT. You can read more about AQT on this [Google Cloud blog](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e).
You can turn on the quantization by adding the following flag `--quantization` and passing one of the following values:

- 'int8' for dynamic range quantization using 8-bits
- 'int8w' for weights only quantization using 8-bits
- 'int4w' for weights only quantization using 4-bits
- 'intmp' for mixed precision weight only quantization based on config file
- 'fp8' for 8-bit floating-point GeMMs on NVIDIA GPUs.



## How QAT works with Qwix

The core idea behind QAT is to insert "fake quantization" operations into the model's computation graph. During the training forward pass, these operations simulate the effect of quantizing weights and activations to a lower precision. For the backward pass, Qwix uses the Straight-Through Estimator (STE) to approximate the gradients, allowing the model to learn effectively despite the non-differentiable nature of quantization.

## Using Qwix in MaxText

You can enable quantization in MaxText by setting flags in your configuration file (e.g., `base.yml`) or via the command line.

### Configuration flags

*   `use_qwix_quantization`: Must be set to `True` to enable quantization using the Qwix library.
*   `quantization`: Specifies the type of quantization to apply. Common options include:
    *   `"int8"`: 8-bit integer quantization.
    *   `"fp8"`: 8-bit floating-point quantization.
    *   `"fp8_full"`: FP8 quantization with static scaling.
    *   `"fp8_gpu"`: FP8 for NVIDIA GPUs.
    *   `"fp8_nanoo"`: FP8 for AMD MI300/MI325 GPUs.
*   `quantization_calibration_method`: The calibration method for weights and activations (e.g., `"absmax"`).

### Example command

Here is an example of how to run a training job with int8 quantization enabled via Qwix:

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml ... use_qwix_quantization=True quantization='int8'
```

## The Qwix interception API

MaxText integrates Qwix using its powerful and non-intrusive Interception API. This approach allows you to enable QAT for your models without modifying the original model source code. You don't need to manually replace `nn.Dense` with `QuantizedDense` or other quantized layer types.

Instead, you define a set of quantization rules externally. Qwix then uses a context manager to "intercept" the creation of standard Flax/NNX layers during model initialization and dynamically replaces the layers with their QAT-enabled versions on the fly.

A quantization rule can be defined as follows:

```python
rule = [qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.int8,
          act_qtype=jnp.int8,
          bwd_qtype=jnp.int8,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
     )]
```

**`QtRule` parameters**:

*   `module_path`: A regex to match the layers to which this rule should be applied.
*   `weight_qtype`: The target quantization type for weights (e.g., `jnp.int8`).
*   `act_qtype`: The target quantization type for activations.
*   `bwd_qtype`: The quantization type for the backward pass.
*   `op_names`: The operations to be quantized (e.g., `"dot_general"`).

This rule is then used within a `QtProvider` to quantize the model automatically:

```python
model = qwix.quantize_model(model, qwix.QtProvider(rule))
```
