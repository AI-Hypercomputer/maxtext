<!--
 Copyright 2024-2025 Google LLC

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

(quantization)=

# Quantization

Quantization in deep learning is the process of reducing the precision of numbers used to represent a model's weights and/or activations. Instead of using higher-precision floating-point formats like 32-bit floats (`float32`) or 16-bit brain floats (`bfloat16`), quantization maps these values to lower-precision numerical formats, most commonly 8-bit integers (`int8`) or floats (`fp8`).

MaxText supports quantization via both the [AQT](https://github.com/google/aqt) and [Qwix](https://github.com/google/qwix) libraries. Qwix is the recommended approach, providing a non-intrusive way to apply Quantized Training (QT).

## Why use quantization?

The drive to use lower-precision formats like `int8` or `fp8` stems from significant performance advantages:

**Faster computation**: Hardware accelerators like TPUs and GPUs often have specialized instructions for integer arithmetic. Operations on lower-precision data like `int8` or `fp8` can be significantly faster than on BF16 or FP32. For example, matrix multiplications with these formats can often be 2x or more faster on hardware supporting native lower-precision tensor cores.

**Reduced memory footprint**: Storing weights and activations in `int8` or `fp8` requires 2x less memory compared to `bfloat16`. This reduces:

- **HBM usage**: Less memory is needed on the accelerator itself.
- **Communication costs**: Less data needs to be transferred between memory and compute units, or across devices in distributed training, which makes these transfers faster and consumes less bandwidth.
- **Reduced power consumption**: Lower precision operations and reduced memory access lead to less energy usage, which is crucial for deploying models on edge devices and for sustainable AI.

The primary trade-off with quantization is between the model accuracy and computational performance:

- Reduced Dynamic Range & Precision: Lower-precision formats like `int8` or `fp8` can represent a much smaller range of values and with less precision than BF16. This can be problematic for models with wide distributions of weights or activations, potentially clipping large values or losing fine-grained details.
- Impact on Gradients: Gradients during backpropagation can have very different, often wider, distributions than weights or activations, making them more sensitive to quantization errors.
- Convergence Issues: The approximations introduced by quantization can sometimes hinder the model's ability to converge during training.

To overcome the challenges of quantization, libraries like Google's Accurate Quantized Training (AQT) and its successor Qwix (used in MaxText) employ a suite of advanced techniques. These methods ensure that models can be trained with low-precision arithmetic without significant loss in accuracy and with stable convergence.

## How Quantized Training (QT) works with Qwix

Quantized Training (QT) incorporates the effects of quantization into the training loop. This allows the model to learn and adapt to the reduced precision of quantized weights and activations.

Here’s how it works:

1. **Forward Pass**: During the forward pass, high-precision weights and activations are converted to a lower-precision format. This step simulates the information loss that occurs during quantization. The model then performs its computations using these lower-precision representations before they are converted back to a higher precision for the rest of the network. This process forces the model to become robust to the noise and reduced range of quantized values.

2. **Backward Pass**: Standard backpropagation cannot flow through the non-differentiable quantization operations (like rounding). To solve this, QT uses the **Straight-Through Estimator (STE)**. The STE essentially "ignores" the non-differentiable quantization step during the backward pass, passing the gradients through as if the operation was an identity function. This allows the high-precision weights to be updated based on the loss, enabling the model to learn effectively.

By integrating the quantization simulation directly into the training, the model learns to minimize the impact of precision loss, resulting in a more accurate quantized model.

## Using Quantization in MaxText

You can enable quantization in MaxText by setting flags in your configuration file (e.g., `base.yml`) or via the command line. MaxText supports two quantization libraries: Qwix (recommended) and AQT.

### Configuration Flags

The primary flags to control quantization are:

- `use_qwix_quantization`: A boolean flag.
  - Set to `True` to enable quantization using the Qwix library.
  - Set to `False` (or omit) to use the AQT library if `quantization` is set.
- `quantization`: A string that specifies the type of quantization to apply. The accepted values depend on whether you are using Qwix or AQT.
- `quantization_calibration_method`: The calibration method for weights and activations (e.g., `"absmax"`). This is mainly for Qwix.

### Qwix Quantization (Recommended)

To use Qwix, you must set `use_qwix_quantization=True`. Qwix is a powerful and non-intrusive library for Quantized Training.

#### `quantization` values for Qwix

Common options for the `quantization` flag when using Qwix include:

- `"int8"`: 8-bit integer quantization.
- `"fp8"`: 8-bit floating-point quantization.
- `"fp8_full"`: FP8 quantization with static scaling.
- `"fp8_gpu"`: FP8 for NVIDIA GPUs.
- `"fp8_nanoo"`: FP8 for AMD MI300/MI325 GPUs.

#### Example command for Qwix

Here is an example of how to run a training job with int8 quantization enabled via Qwix:

```bash
python3 -m MaxText.train src/maxtext/configs/base.yml run_name=$YOUR_JOB_NAME base_output_directory=gs://<my-bucket> dataset_type=synthetic use_qwix_quantization=true quantization='int8'
```

#### The Qwix Interception API

MaxText integrates Qwix using its powerful and non-intrusive Interception API. This approach allows you to enable QAT for your models without modifying the original model source code. You don't need to manually replace `nn.Dense` with `QuantizedDense` or other quantized layer types.

Instead, you define a set of quantization rules externally. Qwix then uses a context manager to "intercept" the creation of standard Flax/NNX layers during model initialization and dynamically replaces the layers with their QAT-enabled versions on the fly.

A quantization rule can be defined as follows:

```python
rule = [
    qwix.QtRule(
        module_path="decoder/.*layers.*",
        weight_qtype=jnp.int8,
        act_qtype=jnp.int8,
        bwd_qtype=jnp.int8,
        bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
        op_names=("dot_general",),
    )
]
```

**`QtRule` parameters**:

- `module_path`: A regex to match the layers to which this rule should be applied.
- `weight_qtype`: The target quantization type for weights (e.g., `jnp.int8`).
- `act_qtype`: The target quantization type for activations.
- `bwd_qtype`: The quantization type for the backward pass.
- `op_names`: The operations to be quantized (e.g., `"dot_general"`).

This rule is then used within a `QtProvider` to quantize the model automatically:

```python
model = qwix.quantize_model(model, qwix.QtProvider(rule))
```

### AQT Quantization

If `use_qwix_quantization` is `False` or not set, you can still apply quantization using the AQT library by setting the `quantization` flag. You can read more about AQT on this [Google Cloud blog](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e).

#### `quantization` values for AQT

When using AQT, you can pass one of the following values to the `quantization` flag:

- 'int8' for dynamic range quantization using 8-bits
- 'int8w' for weights only quantization using 8-bits
- 'int4w' for weights only quantization using 4-bits
- 'intmp' for mixed precision weight only quantization based on config file
- 'fp8' for 8-bit floating-point GeMMs on NVIDIA GPUs.

#### Example command for AQT

```bash
python3 -m MaxText.train src/maxtext/configs/base.yml run_name=$YOUR_JOB_NAME base_output_directory=gs://<my-bucket> dataset_type=synthetic use_qwix_quantization=false quantization='int8'
```

Note that `use_qwix_quantization` is not set to `True`.

For further reading, please refer to the [Qwix Read the Docs website](https://qwix.readthedocs.io/en/latest/get_started.html#).

## DeepSeek V3 Fine-tuning FP8 Recipe

To improve the performance of DeepSeek V3 fine-tuning, we developed a custom recipe optimized for FP8 throughput. The method prioritizes specific compute-intensive and bandwidth-heavy components while preserving training stability through a fine-grained scaling strategy.

### Quantization Scope

To realize these gains, the recipe employs a w8a8g8 (8-bit weights, activations and gradients) strategy targeting three primary areas:

- Megablox Kernels: Specifically the `gmm` and `tgmm` operations.

- Attention Projections: Utilizing convolution fusion.

- Communication: Specifically the weight All-Gathers.

### FP8 Recipe

- Rounding: rounding to nearest even
- Precision
  - Activations and weights: e4m3fn
  - Gradients: e5m2
- Scaling granularity: per-axis
- Scaling mode:
  - static for weights and activations
  - dynamic for gradients

### Convergence

To validate this recipe, we utilized MaxText following the MLPerf Training framework by MLCommons to ensure a reproducible and standardized evaluation. Using the C4 dataset (loaded via TFDS) as the reference corpus, we tracked convergence by monitoring validation loss on a held-out split. This aligns with MLPerf’s time-to-quality principle, where the primary metric is the speed at which the model achieves target quality.

For this specific case, we derived our training duration from the MLPerf 405B benchmark, targeting roughly 2–3 billion tokens after resuming from a checkpoint. In our configuration, we executed 300 steps with a sequence length of 4096 and a global batch size of 2048, resulting in a total of approximately 2.5 billion tokens.

### Performance Sensitivity

Please note that the FP8 benefits are highly sensitive to model parameters, the efficiency of the BF16 baseline, and hardware utilization; consequently, results will vary when this recipe is applied to other models. Any variance in these factors shifts the ratio of compute-bound to memory-bound operations, directly altering the potential gains.
