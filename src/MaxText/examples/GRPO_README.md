# GRPO Demo - Unified Interface

This directory contains the unified GRPO (Group Relative Policy Optimization) demo interface that consolidates the common logic from individual demo scripts. The interface is **model-agnostic** and supports any model (Llama, Qwen, etc.).

## Structure

- **`grpo_demo.py`** - Simple CLI interface for running GRPO training
- **`grpo_demo_trainer.py`** - Core GRPO training logic (in `experimental/rl/`)
- **`grpo.yml`** - Unified model-agnostic configuration file (in `configs/`)

## Usage

### Llama Models

```bash
# Llama3.1-8B (single host)
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --steps=100

# Llama3.1-70B with Pathways
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-70b \
  --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=gs://path/to/output \
  --hf_access_token=$HF_TOKEN \
  --use_pathways=true \
  --steps=100
```

### Qwen Models

```bash
# Qwen2.5-7B (single host)
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=qwen2.5-7b \
  --tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --steps=100

# Qwen2.5-72B with Pathways
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=qwen2.5-72b \
  --tokenizer_path=Qwen/Qwen2.5-72B-Instruct \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=gs://path/to/output \
  --hf_access_token=$HF_TOKEN \
  --use_pathways=true \
  --steps=100
```

### Custom Dataset

```bash
# Any model with custom HuggingFace dataset
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=your-model \
  --tokenizer_path=your/tokenizer \
  --load_parameters_path=gs://path/to/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --hf_path=custom/dataset \
  --steps=100
```

## Key Features

### GRPO-Specific Components

The unified interface includes all essential GRPO components:

1. **Reward Functions**:
   - `match_format_exactly` - Rewards exact format matching
   - `match_format_approximately` - Rewards partial format matching  
   - `check_answer` - Rewards correct answers
   - `check_numbers` - Rewards correct numerical answers

2. **Model Loading**:
   - Reference model (for KL divergence)
   - Policy model (for training)
   - Proper device allocation for multi-host setups

3. **Dataset Processing**:
   - GSM8K math reasoning dataset
   - Special token formatting for reasoning tasks
   - Batch processing for training and evaluation

4. **Training Configuration**:
   - GRPO-specific hyperparameters (beta, epsilon, num_generations)
   - Optimizer setup with warmup and cosine decay
   - Checkpointing and metrics logging

### Device Allocation

The system automatically handles device allocation:

- **Single Host**: Uses all available devices
- **Multi-Host**: Splits devices between training and inference
- **Pathways**: Full multi-host support with proper mesh setup

### Configuration

The `grpo.yml` config file provides sensible defaults for:

- GRPO hyperparameters
- Training loop configuration
- Dataset processing
- Checkpointing settings
- Performance optimizations

## Migration from Individual Demos

The old individual demo files (`grpo_llama3_1_8b_demo.py`, etc.) are now deprecated. To migrate:

1. **Replace model-specific scripts** with the unified `grpo_demo.py`
2. **Use CLI arguments** instead of hardcoded parameters
3. **Leverage `grpo.yml`** for common configuration
4. **Customize via CLI** for model-specific needs

## Examples

### Llama3.1-8B Training

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-8b \
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  --load_parameters_path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --steps=10
```

### Qwen2.5-7B Training

```bash
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=qwen2.5-7b \
  --tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
  --load_parameters_path=gs://path/to/qwen/checkpoint \
  --base_output_directory=/tmp/grpo_output \
  --hf_access_token=$HF_TOKEN \
  --steps=100
```

### Large Models with Pathways

```bash
# Llama3.1-70B with Pathways
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=llama3.1-70b \
  --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  --load_parameters_path=gs://path/to/70b/checkpoint \
  --base_output_directory=gs://path/to/output \
  --hf_access_token=$HF_TOKEN \
  --use_pathways=true \
  --inference_devices_per_replica=8 \
  --inference_replicas=2 \
  --steps=100

# Qwen2.5-72B with Pathways
python3 src/MaxText/examples/grpo_demo.py \
  --model_name=qwen2.5-72b \
  --tokenizer_path=Qwen/Qwen2.5-72B-Instruct \
  --load_parameters_path=gs://path/to/qwen72b/checkpoint \
  --base_output_directory=gs://path/to/output \
  --hf_access_token=$HF_TOKEN \
  --use_pathways=true \
  --inference_devices_per_replica=8 \
  --inference_replicas=2 \
  --steps=100
```

## Contributing

When adding new features:

1. **Add CLI arguments** to `grpo_demo.py`
2. **Update `grpo.yml`** with new configuration options
3. **Extend `grpo_demo_trainer.py`** with new logic
4. **Update this README** with usage examples

## Dependencies

The GRPO demo requires:

- MaxText core dependencies
- Tunix library for RL
- vLLM for efficient inference
- HuggingFace datasets and tokenizers
- JAX/Flax for model training