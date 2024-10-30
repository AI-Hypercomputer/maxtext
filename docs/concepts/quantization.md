# Quantization

Accurated Quantized Training is another technique that maps a subset of matrix multiplications in the training step to int8 to boost training efficiency.

Jax supports AQT quantization. You can read more about AQT quantization on this [Google Cloud blog](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e).
You can turn on the quantization by adding the following flag `--quantization` and passing one of the following values:

- 'int8' for dynamic range quantization using 8-bits
- 'int8w' for weights only quantization using 8-bits
- 'int4w' for weights only quantization using 4-bits
- 'intmp' for mixed precision weight only quantization based on config file
- 'fp8' for 8-bit floating-point GeMMs on NVIDIA GPUs.

```{figure} quantization.png

EMFU measured using MaxText 128b, context length 2048, trained with synthetic data, using Cloud TPU v5e-256. Measured as of April, 2024.
```
