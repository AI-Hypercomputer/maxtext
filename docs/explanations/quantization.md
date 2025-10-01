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

## What is quantization?
Quantization in deep learning is the process of reducing the precision of numbers used to represent a model's weights and/or activations. Instead of using higher-precision floating-point formats like 32-bit floats (`float32`) or 16-bit brain floats (`bfloat16`), quantization maps these values to lower-precision numerical formats, most commonly 8-bit integers (`int8`).

The core idea is to represent the wide range of floating-point numbers using a smaller set of discrete integer values. This conversion typically involves:

* Scaling: Determining a scale factor to map the original range of floating-point values to the target integer range (e.g., -128 to 127 for `int8`).
* Zero-Point: Finding an integer value that maps to the real value 0.0.
* Rounding: Converting the scaled floating-point numbers to the nearest integers.
The relationship can be expressed as: `real_value = scale * (quantized_value - zero_point)`.

## Why use quantization? 

### Primary benefits
The drive to use lower-precision formats like `int8` or `fp8` stems from significant performance advantages:

**Faster computation**: Hardware accelerators like TPUs and GPUs often have specialized instructions for integer arithmetic. Operations on `int8` data can be significantly faster than on BF16 or FP32. For example, `int8` matrix multiplications can often be 2x or more faster on hardware supporting native `int8` tensor cores.

**Reduced memory footprint**: Storing weights and activations in `int8` requires 2x less memory compared to `bfloat16`. This reduces:
- **HBM usage**: Less memory is needed on the accelerator itself.
- **Communication costs**: Transferring data between memory and compute units, or across devices in distributed training, becomes faster and consumes less bandwidth.

### Potential challenges: Trade offs

The primary trade-off with quantization is a potential loss of model accuracy or issues with training convergence:

* Reduced Dynamic Range & Precision: `int8` can represent a much smaller range of values and with less precision than BF16. This can be problematic for models with wide distributions of weights or activations, potentially clipping large values or losing fine-grained details.
* Impact on Gradients: Gradients during backpropagation can have very different, often wider, distributions than weights or activations, making them more sensitive to quantization errors.
* Convergence Issues: The approximations introduced by quantization can sometimes hinder the model's ability to converge during training.

### Mitigating Trade-offs: Techniques in AQT & Qwix

To overcome the challenges of quantization, libraries like Google's Accurate Quantized Training (AQT) and its successor Qwix (used in MaxText) employ a suite of advanced techniques. These methods ensure that models can be trained with low-precision arithmetic without significant loss in accuracy and with stable convergence.

**Quantized Training**: Instead of quantizing a model after training is complete (Post-Training Quantization - PTQ), this approach integrates quantization directly into the training loop. By simulating quantization effects during training, the model learns to adapt its weights to be robust to precision loss. This not only helps maintain high accuracy but also aims to accelerate the training process itself by using faster, low-precision computations. Qwix employs actual quantized operations in the forward pass and optionally in the backward pass, using a Straight-Through Estimator (STE) to allow gradients to flow through the non-differentiable quantization steps.

**Granular Scaling**: To better represent tensors with a wide dynamic range, scaling factors can be applied at different levels of granularity.
- **Per-tensor scaling**: Uses a single scale factor for the entire tensor. This is simple but can be imprecise if the tensor contains values with vastly different magnitudes.
- **Per-axis / per-channel scaling**: Uses a different scale factor for each channel or axis of a tensor. For example, in a weight matrix, this allows each output channel to have its own optimal scaling, better accommodating variations.
- **Sub-channel / tiled quantization**: Takes granularity a step further by applying quantization to smaller blocks or tiles within a tensor (e.g., 1x128 for activations, 128x128 for weights). This provides even finer control, as seen in modern approaches like DeepSeek-V3. Qwix also supports this technique.

**Calibration**: To find the most effective scaling factors, a calibration process is used. This involves analyzing the distribution of weights and activations, typically on a representative sample of data, to determine the optimal range (min/max values) for quantization. Good calibration minimizes information loss by ensuring the full integer range is used effectively.

**Stochastic Rounding**: Standard rounding (e.g., "round to nearest") can introduce systematic bias, especially when dealing with the small numerical updates in gradients. Stochastic rounding mitigates this by rounding a value up or down probabilistically based on its fractional part. On average, this creates an unbiased representation, which is critical for stable backpropagation, especially when using `int8` for gradients.

**Flexible Configuration**: AQT and Qwix provide fine-grained control over the quantization process. Users can specify what gets quantized (weights, activations), the target precision (`int8`, FP8, etc.), and whether to apply it to the forward pass, backward pass (gradients), or both. This flexibility allows for mixed-precision training, where highly sensitive parts of the model can remain in higher precision while still reaping the performance benefits of quantizing less critical parts.

For further reading, please refer to the [Qwix Read the Docs website](https://qwix.readthedocs.io/en/latest/get_started.html#).


