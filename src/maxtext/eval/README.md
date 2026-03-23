# MaxText Model Evaluation Framework

A vLLM-native evaluation framework for MaxText models.

## Quick Start

### Run eval from a MaxText checkpoint

```bash
python -m maxtext.eval.runner.eval_runner \
  --config src/maxtext/eval/configs/mmlu.yml \
  --base_config src/maxtext/configs/post_train/rl.yml \
  --checkpoint_path gs://<bucket>/run/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path gs://<bucket>/run/hf/ \
  --base_output_directory gs://<bucket>/ \
  --run_name my_run \
  --hf_token $HF_TOKEN
```

The runner will:
1. Convert the MaxText checkpoint to HuggingFace format (skipped if `config.json` already exists at `hf_path`).
2. Start a vLLM-TPU server pointed at `hf_path`.
3. Warm up the server.
4. Dispatch evaluation requests concurrently.
5. Score responses.
6. Write results to `{base_output_directory}/{run_name}/eval_results/`.

### Run eval from an existing HuggingFace model (no checkpoint conversion)

Use `--skip_conversion` and point `--hf_path` directly at a local or GCS HuggingFace model directory.

```bash
python -m maxtext.eval.runner.eval_runner \
  --config src/maxtext/eval/configs/mmlu.yml \
  --hf_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_name tinyllama \
  --base_output_directory /tmp/eval_test/ \
  --run_name smoke_test \
  --skip_conversion \
  --num_samples 20 \
  --tensor_parallel_size 1
```

## HuggingFace Token

Llama, Gemma, and other gated models require a HuggingFace token.

Pass the token in order of preference:

1. `--hf_token` CLI flag, this will forwarded to the vLLM server subprocess, checkpoint conversion, and tokenizer loading.
2. Export a `HF_TOKEN` environment variable, this will be picked up automatically if `--hf_token` is not set.

```bash
# Pass in token.
python -m maxtext.eval.runner.eval_runner ... --hf_token hf_...

# Or set env var.
export HF_TOKEN=hf_...
python -m maxtext.eval.runner.eval_runner ...
```

## Configuration

1. `base_eval.yml`: Shared defaults (temperature, concurrency, server params)
2. `benchmark.yml`: Benchmark-specific overrides (max_tokens, concurrency)
3. MaxText base config `base.yml`: Common configs used to derive max_model_len, max_tokens, results path
4. Eval CLI flags: Can override everything

### CLI flags details

| Flag | Description |
|---|---|
| `--config` | Path to benchmark YAML |
| `--base_config` | Path to MaxText config |
| `--checkpoint_path` | MaxText orbax checkpoint path (`…/0/items`) |
| `--hf_path` | HuggingFace model directory |
| `--model_name` | MaxText model name (e.g. `llama3.1-8b`) |
| `--base_output_directory` | GCS or local base directory for results |
| `--run_name` | Run identifier, used in results path |
| `--hf_token` | HuggingFace token for gated models |
| `--num_samples` | Limit number of eval samples |
| `--skip_conversion` | Skip checkpoint conversion and use `hf_path` instead |
| `--tensor_parallel_size` | Number of chips for vLLM tensor parallelism |
| `--max_num_batched_tokens` | vLLM scheduler tokens per step |
| `--max_num_seqs` | vLLM max concurrent sequences |


## Adding a New Benchmark

1. Implement `BenchmarkDataset` in `src/maxtext/eval/datasets/`:

```python
from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

class MyDataset(BenchmarkDataset):
    name = "my_benchmark"

    def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
        ...  # load dataset, build prompts, return SampleRequest list
```

2. Register it in `src/maxtext/eval/datasets/registry.py`:

```python
from maxtext.eval.datasets.my_dataset import MyDataset
DATASET_REGISTRY["my_benchmark"] = MyDataset
```

3. Add a scorer in `src/maxtext/eval/scoring/` and register it in
   `src/maxtext/eval/scoring/registry.py`. Scorers can delegate to
   `eval.vllm.benchmark_utils` functions or implement new logic.
