
# Optimized models tiering

For each of the TPU platforms listed below, we present a list of optimized models[^1] [^2] for pre-training. If you’re getting started with MaxText, or want to push performance, we recommend choosing a Gold model, with an accompanying pre-training recipe.

- **Gold Tier**: Fully Optimized Models certified to run with maximum efficiency on Cloud TPUs. They are thoroughly refined for the highest possible performance, making them ideal for production-critical workloads requiring peak throughput.

- **Silver Tier**: High Performance Models that are well-optimized to deliver high, reliable performance on Cloud TPUs. They are effective for most use cases but may offer opportunities for expert tuning to achieve peak (Gold Tier) performance.

## Trillium (v6e)

### Gold

| Model | Recipe | Benchmark Configuration | MFU | Approx tokens/sec/device |
| :--- | :--- | :--- | :--- | :--- |
| Llama 2 70B | [Link](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama2-70B-MaxText) | 256, BF16, SL=4096 | 43.8% | 900 |
| Llama 3.1 8B | [Link](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-256) | 256 Chips, BF16, SL=8192 | 45.46% | 7,207 |
| Llama 3.1 70B | [Link](https://github.com/AI-Hypercomputer/maxtext/blob/92e59fdf547421f647590087f50fea5729da42d8/benchmarks/maxtext_trillium_model_configs.py#L959) | 256 Chips, BF16, SL=8192 | 50.33% | 960 |

### Silver

| Model | Recipe | Benchmark Configuration | MFU | Approx tokens/sec/device |
| :--- | :--- | :--- | :--- | :--- |
| Llama 3.1 405B | [Link](https://github.com/AI-Hypercomputer/maxtext/blob/5e6a7caff904f67fa654fc0ae983a16156bc21f8/benchmarks/maxtext_trillium_model_configs.py#L723) | 256 Chips, BF16, SL=8192 | 38.55% | 123 |
| Mixtral 8X7B | [Link](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x7B-MaxText) | 256 Chips, BF16, SL=4096 | 35.23% | 3,899 |
| Mixtral 8X22B | [Link](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x22B-MaxText) | 256 Chips, BF16, SL=4096 | 36.2% | 1,326 |

## v5p

### Gold

| Model | Recipe | Benchmark Configuration | MFU | Approx tokens/sec/device |
| :--- | :--- | :--- | :--- | :--- |
| Llama 2 70B | [Link](https://github.com/AI-Hypercomputer/maxtext/blob/92e59fdf547421f647590087f50fea5729da42d8/benchmarks/maxtext_v5p_model_configs.py#L156) | 512 Chips, BF16, SL=4096 | 65.4% | 692 |

### Silver

| Model | Recipe | Benchmark Configuration | MFU | Approx tokens/sec/device |
| :--- | :--- | :--- | :--- | :--- |
| Mixtral 8X7B | [Link](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Mixtral-8X7B-Maxtext) | 256 Chips(8x4x4), bf16, SL=4096 | 52.56% | 2,909 |

[^1]:  Performance results are subject to variations based on system configuration, software versions, and other factors. These benchmarks represent point-in-time measurements under specific conditions.
[^2]:  Some older TFLOPS/s results are impacted by an updated calculation for causal attention ([PR #1988](https://github.com/AI-Hypercomputer/maxtext/pull/1988)), which halves the attention FLOPs. This change particularly affects configurations with large sequence lengths. For more details, please refer to the [performance metrics guide](https://maxtext.readthedocs.io/en/latest/reference/performance_metrics.html).
