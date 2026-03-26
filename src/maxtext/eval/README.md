# MaxText Model Evaluation Framework

A vLLM-native evaluation framework for MaxText models.

## Quick Start

### eval_runner: With MaxText checkpoint

```bash
python -m maxtext.eval.runner.eval_runner \
  --config src/maxtext/eval/configs/mlperf.yml \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --base_output_directory gs://<bucket>/ \
  --run_name mlperf_eval_run \
  --hf_token $HF_TOKEN
```

### eval_runner: With HF model

Use `--hf_mode` with a public HF model to test the framework
without any MaxText checkpoint.

```bash
python -m maxtext.eval.runner.eval_runner \
  --config src/maxtext/eval/configs/mlperf.yml \
  --hf_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_name tinyllama \
  --base_output_directory /tmp/eval_test/ \
  --run_name smoke_test \
  --hf_mode \
  --num_samples 20 \
  --tensor_parallel_size 1
```

### lm_eval_runner

Uses lm-evaluation-harness with loglikelihood scoring.

Requires: `pip install "lm_eval[api]"`

```bash
python -m maxtext.eval.runner.lm_eval_runner \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --tasks mmlu gpqa \
  --base_output_directory gs://<bucket>/ \
  --run_name my_run \
  --max_model_len 8192 \
  --tensor_parallel_size 4 \
  --hf_token $HF_TOKEN
```

### evalchemy_runner

Uses [mlfoundations/evalchemy](https://github.com/mlfoundations/evalchemy), which
extends lm-evaluation-harness with chat-completions-based benchmarks. Imports
`evalchemy` for task registration then drives evaluation via `lm_eval.simple_evaluate`.

Requires: `pip install evalchemy`

```bash
python -m maxtext.eval.runner.evalchemy_runner \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --tasks ifeval math500 gpqa_diamond \
  --base_output_directory gs://<bucket>/ \
  --run_name my_run \
  --max_model_len 8192 \
  --tensor_parallel_size 4 \
  --hf_token $HF_TOKEN
```

## HuggingFace Token

Llama, Gemma, and other gated models require a HuggingFace token. You must
also have accepted the model license on huggingface.co.

In the `MaxTextForCausalLM` mode, the token is only needed to
download the tokenizer, not model weights.

Pass the token in order of preference:

1. `--hf_token` — forwarded to the server and tokenizer loading.
2. `HF_TOKEN` environment variable (picked up automatically if `--hf_token` is not set).

```bash
# Pass hf_token.
python -m maxtext.eval.runner.eval_runner ... --hf_token hf_...

# Or export env variable.
export HF_TOKEN=hf_...
python -m maxtext.eval.runner.eval_runner ...
```

### Configuration (eval_runner)

| Flag | Description |
|---|---|
| `--config` | Path to benchmark YAML. |
| `--base_config` | Path to MaxText config |
| `--checkpoint_path` | MaxText orbax checkpoint. Enables MaxTextForCausalLM mode. |
| `--hf_path` | HF model ID or tokenizer dir. |
| `--model_name` | MaxText model name (e.g. `llama3.1-8b`) |
| `--base_output_directory` | GCS or local base directory for results |
| `--run_name` | Run name, used in results path |
| `--hf_token` | HuggingFace token for gated models |
| `--num_samples` | Limit number of eval samples |
| `--hf_mode` | Force HF safetensors mode (disables MaxTextForCausalLM mode) |
| `--tensor_parallel_size` | vLLM tensor parallelism |
| `--max_num_batched_tokens` | vLLM scheduler tokens per step |
| `--max_num_seqs` | vLLM max concurrent sequences (KV cache cap) |

### Configuration (lm_eval_runner)

| Flag | Description |
|---|---|
| `--checkpoint_path` | MaxText orbax checkpoint. Enables MaxTextForCausalLM mode. |
| `--model_name` | MaxText model name |
| `--hf_path` | HF model ID for tokenizer |
| `--tasks` | Space-separated lm-eval task names (e.g. `mmlu gpqa`) |
| `--base_output_directory` | GCS or local base directory for results |
| `--run_name` | Run name |
| `--max_model_len` | vLLM max context length |
| `--tensor_parallel_size` | Number of chips |
| `--num_fewshot` | Few-shot examples per task (default: 0) |
| `--num_samples` | Limit samples per task (default: full dataset) |
| `--hf_token` | HuggingFace token for gated models |
| `--hf_mode` | Force HF safetensors mode |

### Configuration (evalchemy_runner)

| Flag | Description |
|---|---|
| `--checkpoint_path` | MaxText orbax checkpoint. Enables MaxTextForCausalLM mode. |
| `--model_name` | MaxText model name |
| `--hf_path` | HF model ID for tokenizer |
| `--tasks` | Space-separated task names from the table above |
| `--base_output_directory` | GCS or local base directory for results |
| `--run_name` | Run name |
| `--max_model_len` | vLLM max context length |
| `--tensor_parallel_size` | Number of chips |
| `--num_fewshot` | Few-shot examples per task (default: 0) |
| `--num_samples` | Limit samples per task (default: full dataset) |
| `--hf_token` | HuggingFace token for gated models |
| `--hf_mode` | Force HF safetensors mode |

## Adding a New Benchmark

For custom datasets not covered by lm-eval or evalchemy:

1. Implement `BenchmarkDataset` in `src/maxtext/eval/datasets/`:

```python
from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

class MyDataset(BenchmarkDataset):
    name = "my_benchmark"

    def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
        # load dataset, build prompts, return SampleRequest list
```

2. Register it in `src/maxtext/eval/datasets/registry.py`:

```python
from maxtext.eval.datasets.my_dataset import MyDataset
DATASET_REGISTRY["my_benchmark"] = MyDataset
```

3. Add a scorer in `src/maxtext/eval/scoring/` and register it in
   `src/maxtext/eval/scoring/registry.py`.
