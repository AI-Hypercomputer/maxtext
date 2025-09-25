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
Quantization in deep learning is the process of reducing the precision of numbers used to represent a model's weights and/or activations. Instead of using higher-precision floating-point formats like 32-bit floats (fp32) or 16-bit brain floats (bfloat16), quantization maps these values to lower-precision numerical formats, most commonly 8-bit integers (int8).

The core idea is to represent the wide range of floating-point numbers using a smaller set of discrete integer values. This conversion typically involves:

* Scaling: Determining a scale factor to map the original range of floating-point values to the target integer range (e.g., -128 to 127 for INT8).
* Zero-Point: Finding an integer value that maps to the real value 0.0.
* Rounding: Converting the scaled floating-point numbers to the nearest integers.
The relationship can be expressed as: `real_value = scale * (quantized_value - zero_point)`.

## Why use quantization? 

### Primary benefits
The drive to use lower-precision formats like INT8 stems from significant performance advantages:

**Faster computation**: Hardware accelerators like TPUs and GPUs often have specialized instructions for integer arithmetic. Operations on INT8 data can be significantly faster than on BF16 or FP32. For example, INT8 matrix multiplications can often be 2x or more faster on hardware supporting native INT8 tensor cores.

**Reduced memory footprint**: Storing weights and activations in INT8 requires 2x less memory compared to BF16. This reduces:
- **HBM usage**: Less memory is needed on the accelerator itself.
- **Communication costs**: Transferring data between memory and compute units, or across devices in distributed training, becomes faster and consumes less bandwidth.

### Potential challenges: Trade offs

The primary trade-off with quantization is a potential loss of model accuracy or issues with training convergence:

* Reduced Dynamic Range & Precision: INT8 can represent a much smaller range of values and with less precision than BF16. This can be problematic for models with wide distributions of weights or activations, potentially clipping large values or losing fine-grained details.
* Impact on Gradients: Gradients during backpropagation can have very different, often wider, distributions than weights or activations, making them more sensitive to quantization errors (INT8 versus FP8 comparison).
* Convergence Issues: The approximations introduced by quantization can sometimes hinder the model's ability to converge during training.

### Mitigating Trade-offs: Techniques in AQT & Qwix

Google's Accurate Quantized Training (AQT) library and the newer Qwix library (which MaxText is moving to) employ several techniques to minimize accuracy loss and ensure stable training:

**Quantization Aware Training (QAT)**: Instead of just quantizing a model after training (Post-Training Quantization - PTQ), QAT simulates the effect of quantization during the training process. "Fake quantization" operations are inserted into the model graph, so the model learns to be robust to the precision loss. Qwix uses this approach, employing a Straight-Through Estimator (STE) for backpropagation (MaxText <-> Qwix Integration).

**Per-axis / per-tensor scaling**: Instead of using a single scale factor for an entire tensor, AQT and Qwix can use different scale factors for different parts of a tensor. For example, per-channel scaling for weights allows for a much tighter range for each output channel, better accommodating variations in magnitude.

**Calibration**: Before or during training, a calibration process can be used to determine the optimal range (min/max values) for quantization, often using a representative dataset to minimize information loss (Intro to Quantization).

**Stochastic rounding**: Particularly important for gradients, stochastic rounding rounds a value up or down probabilistically based on its fractional part. This helps to maintain an unbiased representation on average, which can be critical for backpropagation, especially in INT8 (INT8 versus FP8 comparison).

**Flexible configuration**: AQT and Qwix allow fine-grained control over what gets quantized, to what precision (INT8, FP8, etc.), and whether to quantize the forward pass, backward pass (gradients), or both (AQT : Accurate Quantized Training). This allows for mixed-precision training where sensitive parts remain in higher precision.

**Sub-channel / tiled quantization**: For even finer control, quantization can be applied to blocks or tiles within a tensor, as seen in DeepSeek-V3's approach, which uses 1x128 tiling for activations and 128x128 for weights (MaxText: DeepSeek style Quantization Recipe). Qwix also supports sub-channel quantization.

