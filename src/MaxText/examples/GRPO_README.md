# GRPO Demo - Unified Training Interface

This directory contains a unified interface for running GRPO (Group Relative Policy Optimization) training demos across different model sizes and configurations.

## Overview

Previously, there were separate demo scripts for different model configurations:
- `grpo_llama3_1_8b_demo.py` - Single host 8B model
- `grpo_llama3_1_8b_demo_pw.py` - Pathways-based 8B model  
- `grpo_llama3_1_70b_demo_pw.py` - Pathways-based 70B model

These have been consolidated into a single **unified CLI script** (`grpo_demo.py`) that works with the new **grpo.yml** configuration file.

## New Structure

### Configuration File
`src/MaxText/configs/grpo.yml`
- Contains common GRPO parameters
- Can be overridden via CLI arguments
- Consolidates dataset, training, and GRPO-specific settings

### Unified CLI Script
`src/MaxText/examples/grpo_demo.py`
- Single entry point for all GRPO demos
- Supports both single-host and multi-host (Pathways) setups
- Provides intuitive CLI arguments
- Automatically generates proper config for training and inference

## Usage Examples

### Llama3.1-8B (Single Host)

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --steps=100
```

### Llama3.1-70B with Pathways (Multi-Host)

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-70b \
  --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=gs://path/to/output \
  --hf_access_token=$HF_TOKEN \
  --use_pathways \
  --inference_devices_per_replica=4 \
  --inference_replicas=4 \
  --ici_fsdp_parallelism=16 \
  --steps=100
```

### Custom Dataset

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --hf_path=custom/dataset \
  --hf_data_split=train \
  --steps=100
```

### With Custom GRPO Parameters

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --num_generations=4 \
  --grpo_beta=0.04 \
  --grpo_epsilon=0.15 \
  --learning_rate=5e-6 \
  --steps=200
```

## CLI Arguments

### Required Arguments

- `--model_name`: Model name (e.g., llama3.1-8b, llama3.1-70b)
- `--tokenizer_path`: HuggingFace tokenizer path
- `--load_parameters_path`: Path to model checkpoint (local or gs://)
- `--base_output_directory`: Base output directory for logs and checkpoints

### Dataset Arguments

- `--hf_access_token`: HuggingFace access token (can use $HF_TOKEN env var)
- `--hf_path`: HuggingFace dataset path (default: gsm8k)
- `--hf_data_split`: Dataset split (default: main)
- `--hf_data_files`: Dataset files (default: train)

### Training Arguments

- `--steps`: Number of training steps (default: 100)
- `--per_device_batch_size`: Per device batch size (default: 1)
- `--learning_rate`: Learning rate (default: 3e-6)
- `--run_name`: Custom run name for the experiment

### GRPO-Specific Arguments

- `--num_generations`: Number of generations per prompt (default: 2)
- `--grpo_beta`: KL divergence penalty coefficient (default: 0.08)
- `--grpo_epsilon`: Clipping value for stable updates (default: 0.2)

### Sequence Length Arguments

- `--max_prefill_predict_length`: Maximum prompt length (default: 256)
- `--max_target_length`: Maximum total sequence length (default: 768)

### Multi-Host/Pathways Arguments

- `--use_pathways`: Enable Pathways for multi-host training
- `--inference_devices_per_replica`: Devices per inference replica (default: 4)
- `--inference_replicas`: Number of inference replicas (default: 1)

### Parallelism Arguments

- `--ici_fsdp_parallelism`: FSDP parallelism (-1 for auto)
- `--ici_tensor_parallelism`: Tensor parallelism (-1 for auto)

### Other Arguments

- `--profiler`: Profiler to use (default: xplane)
- `--checkpoint_period`: Checkpoint saving period (default: 50)
- `--config_file`: Optional custom config file (overrides grpo.yml)

## Migration Guide

### From Individual Demo Scripts

**Old way:**
```python
# Editing grpo_llama3_1_8b_demo.py directly
MODEL_NAME = "llama3.1-8b"
TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"
# ... many hardcoded parameters
```

**New way:**
```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  # ... all parameters via CLI
```

### Benefits

1. **Single Script**: One script for all model sizes and configurations
2. **No Code Editing**: All parameters configurable via CLI
3. **Better Defaults**: Common parameters in `grpo.yml`
4. **Easier Testing**: Quickly test different configurations
5. **CI/CD Friendly**: Easy to integrate into automated workflows

## Configuration Files

### grpo.yml
Main configuration file with sensible defaults for GRPO demos. Override any parameter via CLI.

Location: `src/MaxText/configs/grpo.yml`

### grpo.yml and grpo_inference.yml
Low-level configuration files used by the GRPO trainer. Generally, you don't need to modify these directly.

Location: `src/MaxText/experimental/rl/`

## Advanced Usage

### Using a Custom Config File

If you have a custom configuration:

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --config_file=/path/to/custom_config.yml \
  --model_name=llama3.1-8b \
  # ... other args
```

### Environment Variables

You can set these environment variables:
- `HF_TOKEN`: HuggingFace access token (alternative to `--hf_access_token`)

## Troubleshooting

### Common Issues

1. **HF_TOKEN not set**: Make sure to either set the environment variable or pass `--hf_access_token`

2. **Pathways configuration**: For multi-host setups, ensure:
   - `--use_pathways` is set
   - `--inference_devices_per_replica` and `--inference_replicas` are configured correctly
   - The total number of devices is sufficient

3. **Memory issues**: Try reducing:
   - `--per_device_batch_size`
   - `--max_target_length`
   - `--num_generations`

## Contributing

When adding new features or model support:
1. Add sensible defaults to `grpo.yml`
2. Add CLI arguments to `grpo_demo.py` if needed
3. Update this README with examples

## See Also

- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [MaxText Documentation](../../../docs/)
- [Tunix Library](https://github.com/google/tunix)

