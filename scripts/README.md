# MaxText Diverse Beam Search Benchmarking

This suite provides an automated pipeline to evaluate the **Quality** and **Diversity** of MaxText sampling algorithms. It is specifically designed to measure the impact of the **Diverse Beam Search (DBS)** implementation.

## Prerequisites

Ensure you have the evaluation dependencies installed:

```bash
pip install evaluate sacrebleu datasets tqdm
```

## Usage

The script is portable across branches. Run it on `master` for baseline results or on feature branch for diversity testing.

### 1. Mock Mode (Logic Validation)
Verify the scoring and data pipeline without loading heavy model weights:

```bash
python3 scripts/benchmark_bleu.py src/maxtext/configs/base.yml \
    --mock \
    max_dataset_examples=10 \
    skip_jax_distributed_system=True
```

### 2. Standard Baseline (Upstream)
To measure standard Beam Search performance:

```bash
python3 scripts/benchmark_bleu.py src/maxtext/configs/base.yml \
    model_name=gemma2-2b \
    load_parameters_path=/path/to/checkpoints \
    decode_sampling_strategy=greedy \
    max_dataset_examples=100 \
    skip_jax_distributed_system=True
```

### 3. Diverse Beam Search Benchmark
To measure diversity using the custom implementation:

```bash
python3 scripts/benchmark_bleu.py src/maxtext/configs/base.yml \
    model_name=gemma2-2b \
    load_parameters_path=/path/to/checkpoints \
    decode_sampling_strategy=diverse_beam_search \
    decode_num_beams=4 \
    decode_num_beam_groups=2 \
    decode_diversity_penalty=0.5 \
    max_dataset_examples=100 \
    skip_jax_distributed_system=True
```

## Metrics Explained

- **BLEU-4 (Quality):** Measures correlation between the top-1 beam and the reference summary.
- **Self-BLEU (Diversity):** Measures pairwise correlation among all generated beams for a single prompt. A lower Self-BLEU score indicates that the algorithm is producing more diverse candidate sequences.

## Configuration Reference

- `max_dataset_examples`: Sample limit (default: 100).
- `skip_jax_distributed_system=True`: Required for single-host/VM execution.
- `decode_diversity_penalty`: Higher values force the model to explore distinct groups.
