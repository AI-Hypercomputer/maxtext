# MaxText vLLM Eval Framework

A vLLM-native evaluation framework for MaxText models supporting harness-based eval (lm-eval, evalchemy) and custom datasets.

## Quick Start

All runners share a single entry point:

```bash
python -m maxtext.eval.runner.run --runner <eval|lm_eval|evalchemy> [flags]
```

### Custom dataset (MLPerf OpenOrca, ROUGE scoring, Other)

```bash
python -m maxtext.eval.runner.run \
  --runner eval \
  --config src/maxtext/eval/configs/mlperf.yml \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --base_output_directory gs://<bucket>/ \
  --run_name eval_run \
  --max_model_len 8192 \
  --hf_token $HF_TOKEN
```

HF safetensors mode (no MaxText checkpoint):

```bash
python -m maxtext.eval.runner.run \
  --runner eval \
  --config src/maxtext/eval/configs/mlperf.yml \
  --hf_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_name tinyllama \
  --base_output_directory gs://<bucket>/ \
  --run_name eval_test \
  --hf_mode \
  --num_samples 20 \
  --max_model_len 2048 \
  --tensor_parallel_size 1
```

### LM Eval

Requires: `pip install "lm_eval[api]"`

```bash
python -m maxtext.eval.runner.run \
  --runner lm_eval \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name qwen3-30b-a3b \
  --hf_path Qwen/Qwen3-30B-A3B \
  --tasks gsm8k \
  --base_output_directory gs://<bucket>/ \
  --run_name my_run \
  --max_model_len 8192 \
  --tensor_parallel_size 8 \
  --expert_parallel_size 8 \
  --hf_token $HF_TOKEN
```

### Evalchemy

Requires: `pip install git+https://github.com/mlfoundations/evalchemy.git`

```bash
python -m maxtext.eval.runner.run \
  --runner evalchemy \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --tasks ifeval math500 gpqa_diamond \
  --base_output_directory gs://<bucket>/ \
  --run_name eval_run \
  --max_model_len 8192 \
  --tensor_parallel_size 4 \
  --hf_token $HF_TOKEN
```

## Common Flags

| Flag | Description |
|---|---|
| `--checkpoint_path` | MaxText Orbax checkpoint path. Enables `MaxTextForCausalLM` mode. |
| `--model_name` | MaxText model name (e.g. `llama3.1-8b`) |
| `--hf_path` | HF model ID or local path |
| `--max_model_len` | vLLM max context length. |
| `--tensor_parallel_size` | Chips per model replica |
| `--expert_parallel_size` | Chips for the expert mesh axis |
| `--data_parallel_size` | Number of model replicas |
| `--hbm_memory_utilization` | Fraction of HBM reserved for KV cache |
| `--hf_token` | HF token (or set `HF_TOKEN` env var) |
| `--hf_mode` | HF safetensors mode, no MaxText checkpoint loading |
| `--server_host` / `--server_port` | vLLM server address (default: localhost:8000) |
| `--max_num_batched_tokens` | vLLM tokens per scheduler step |
| `--max_num_seqs` | vLLM max concurrent sequences |
| `--gcs_results_path` | GCS path to upload results JSON |
| `--log_level` | Logging verbosity (default: INFO) |

 Custom `eval` specific:

| Flag | Description |
|---|---|
| `--config` | Benchmark YAML config (required) |
| `--num_samples` | Limit eval samples |
| `--max_tokens` | Max tokens per generation |
| `--temperature` | Sampling temperature (default: 0.0) |
| `--concurrency` | HTTP request concurrency (default: 64) |

Harness `lm_eval` / `evalchemy` specific:

| Flag | Description |
|---|---|
| `--tasks` | Space-separated task names |
| `--num_fewshot` | Few-shot examples per task (default: 0) |
| `--num_samples` | Limit samples per task (default: full dataset) |

## Eval on RL Checkpoints



Example (Qwen3-30B-A3B, v6e-8):

```bash
STEP=244
MODEL=qwen3-30b-a3b
HF_PATH=Qwen/Qwen3-30B-A3B
CHECKPOINT=gs://<bucket>/run/checkpoints/actor/${STEP}/model_params
OUTPUT=gs://<bucket>/eval/

python -m maxtext.eval.runner.run \
  --runner lm_eval \
  --checkpoint_path ${CHECKPOINT} \
  --model_name ${MODEL} \
  --hf_path ${HF_PATH} \
  --tasks gsm8k \
  --base_output_directory ${OUTPUT} \
  --run_name rl_${MODEL}_step${STEP} \
  --max_model_len 4096 \
  --tensor_parallel_size 8 \
  --expert_parallel_size 8 \
  --num_samples 20 \
  --hf_token $HF_TOKEN
```


## Adding a Custom Benchmark

1. Implement `BenchmarkDataset` in `src/maxtext/eval/datasets/`:

```python
from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

class MyDataset(BenchmarkDataset):
    name = "my_benchmark"

    def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
        # load dataset, build prompts, return SampleRequest list
```

2. Register in `src/maxtext/eval/datasets/registry.py`:

```python
from maxtext.eval.datasets.my_dataset import MyDataset
DATASET_REGISTRY["my_benchmark"] = MyDataset
```

3. Add a scorer in `src/maxtext/eval/scoring/` and register it in `src/maxtext/eval/scoring/registry.py`.
