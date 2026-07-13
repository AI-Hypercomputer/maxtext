# MaxText vLLM Eval Framework

A vLLM-native evaluation framework for MaxText models supporting harness-based eval (lm-eval, evalchemy) and custom datasets.

## Quick Start

All runners share a single entry point:

```bash
python -m maxtext.eval.runner.run --runner <eval|lm_eval|evalchemy|simple_evals> [flags]
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

### Simple Evals (OpenAI simple-evals)

Runs OpenAI's [simple-evals](https://github.com/openai/simple-evals) benchmarks
against the vLLM server's OpenAI-compatible `/v1/chat/completions` endpoint.

Requires: `pip install aiohttp openai jinja2 numpy pandas scipy requests tqdm`

Phase 1 supports grader-free evals only: `mmlu`, `gpqa`, `gsm8k`, `drop`, `mgsm`,
`mgsm_en`, `aime2024`, `aime2025`. Grader-dependent evals (math, simpleqa, browsecomp,
healthbench) need an LLM grader endpoint and are not yet supported.

`mmlu`, `gpqa`, `drop`, `mgsm` are based on the vendored implementations from
[openai/simple-evals](https://github.com/openai/simple-evals); `gsm8k` and
`aime2024`/`aime2025` are not part of upstream simple-evals and are
MaxText-authored (`maxtext.eval.native_evals`) following the same
grader-free Eval conventions. `mgsm` matches upstream's 11-language suite;
`mgsm_en` is the explicitly named English-only variant.

```bash
python -m maxtext.eval.runner.run \
  --runner simple_evals \
  --checkpoint_path gs://<bucket>/checkpoints/0/items \
  --model_name llama3.1-8b \
  --hf_path meta-llama/Llama-3.1-8B-Instruct \
  --tasks mmlu gpqa gsm8k drop mgsm aime2024 aime2025 \
  --base_output_directory gs://<bucket>/ \
  --run_name simple_evals_run \
  --max_model_len 8192 \
  --tensor_parallel_size 4 \
  --num_samples 50 \
  --hf_token $HF_TOKEN
```

Simple-evals specific flags:

| Flag | Description |
|---|---|
| `--tasks` | Space-separated task names (`mmlu`, `gpqa`, `gsm8k`, `drop`, `mgsm`, `mgsm_en`, `aime2024`, `aime2025`). |
| `--num_samples` | Limit examples per task; for MGSM variants this is per language (None = full dataset). |
| `--n_repeats` | Repeats per example (defaults: upstream GPQA=4, AIME=1; forced to 1 with `--num_samples`). |
| `--max_tokens` | Max tokens per generation (default: 2048). |
| `--temperature` | Sampling temperature (upstream default: 0.5). |
| `--concurrency` | Task worker count and maximum in-flight requests. Omit for automatic selection from CPU count, accelerator count, `max_num_seqs`, and chat batch capacity. This is the only request-pressure setting users normally need to tune. |
| `--log-debug-info` | Write a timestamp-matched `.debug.txt` report beside the result JSON. The report includes request/token throughput, latency percentiles, retry and terminal-error rates, output truncation, answer-extraction failures, confidence intervals, and bounded samples of final, reasoning, and raw model output. Reasoning and raw output remain diagnostic-only and are never graded. |
| `--continue-on-request-error` | Score terminal request failures as zero instead of aborting. This is independent of debug logging. |

Debug logging is observational and does not change request-failure behavior.
When `--continue-on-request-error` is enabled, failed requests count as zero in
official accuracy. The debug report also shows diagnostic accuracy excluding
infrastructure failures; that diagnostic must not be used as the published
benchmark score.

Before a long TPU benchmark, run the one-shot chat-path diagnostic with the
same model/server settings. For an 8-chip v6e GPT-OSS deployment in HF mode:

```bash
python -m maxtext.eval.runner.debug_simple_evals \
  --model_name gpt-oss-20b \
  --hf_path openai/gpt-oss-20b \
  --hf_mode \
  --base_output_directory /tmp \
  --run_name simple_evals_debug \
  --max_model_len 32768 \
  --tensor_parallel_size 8 \
  --debug_max_tokens 1024 \
  --reasoning_effort high \
  --debug_output /tmp/simple_evals_tpu_debug.json
```

The script launches the normal in-process server and checks Harmony prompt
rendering, final/reasoning/raw separation, forced analysis-only truncation,
diagnostic opt-in, token accounting, and two-request batch response ordering.
It prints `PASS` and exits zero only when every applicable check succeeds;
otherwise it writes the failure to the JSON report and exits nonzero. When
testing a MaxText checkpoint, pass the same `--checkpoint_path` and
`--model_name` used by the benchmark instead of `--hf_mode`.

### XProf Profiling (inference bottleneck analysis)

Boots the same in-process vLLM server as the eval runners and captures one
xplane trace per workload phase — `prefill` (long prompt, 1 output token),
`decode` (single-stream generation), and `batch` (concurrent requests through
the chat batching queue) — so prefill-, decode-, and host-bound bottlenecks
can be separated in XProf. Each phase runs once untraced first so XLA
compilation is excluded from the trace.

```bash
python -m maxtext.eval.runner.run \
  --runner profile \
  --model_name gpt-oss-20b \
  --hf_path openai/gpt-oss-20b \
  --base_output_directory gs://<bucket>/ \
  --run_name profile_run \
  --max_model_len 8192 \
  --tensor_parallel_size 8 \
  --hf_token $HF_TOKEN

# View the traces:
pip install xprof
xprof --logdir gs://<bucket>/profile_run/eval_results/xprof --port 8791
```

Profiling specific flags:

| Flag | Description |
|---|---|
| `--profile_dir` | Trace output dir (default: `{base_output_directory}/{run_name}/eval_results/xprof`). |
| `--prefill_prompt_words` | Approximate prompt length for the prefill phase (default: 1024). |
| `--decode_prompt_words` | Approximate prompt length for decode/batch (default: 32). Sweep it across separate runs to test KV-length sensitivity; the summary reports actual token counts. |
| `--decode_tokens` | Tokens generated per request in the decode/batch phases (default: 256). |
| `--batch_concurrency` | Concurrent requests in the batch phase (default: 32). Must not exceed `--concurrency` after automatic/explicit resolution. |
| `--reasoning_effort` | `low`/`medium`/`high`, for reasoning models like gpt-oss. |

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
