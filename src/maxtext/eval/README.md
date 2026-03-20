# MaxText Model Evaluation Framework

A vLLM-native evaluation framework for MaxText models. 

## Quick Start

### Post-training eval (standalone)

```bash
# Run MMLU on a MaxText checkpoint
python -m maxtext.eval.runner.eval_runner \
  --config src/maxtext/eval/configs/mmlu.yml \
  --base_config src/maxtext/configs/post_train/rl.yml \
  --base_output_directory gs://<gcs_bucket>/ \
  --run_name eval_run \
  --checkpoint_path gs://<gcs_bucket>/checkpoint/0/items \
  --model_name llama3.1-8b \
  --hf_path gs://<gcs_bucket>/hf/
```

Results are written to `{base_output_directory}/{run_name}/eval_results/`.


The runner will:
1. Convert the MaxText checkpoint to huggingface format (skipped if already exists).
2. Start a vLLM-TPU server.
3. Warmup the server.
4. Dispatch evaluation requests.
5. Score responses.
6. Write a results to `{base_output_directory}/{run_name}/eval_results/`

### Throughput benchmarking (vllm tpu-inference benchmark_serving)

The vllm tpu-inference `benchmark_serving.py` from
[vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference/tree/main/scripts/vllm/benchmarking)
can be run directly against any running vLLM server:

```bash
python -m maxtext.eval.vllm.benchmark_serving \
  --model llama3.1-8b \
  --dataset-name mmlu \
  --dataset-path /path/to/mmlu/test/ \
  --num-prompts 500 \
  --run-eval
```

### Adding a new benchmark

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

3. Add a scorer wrapper in `src/maxtext/eval/scoring/` that delegates to either a vllm tpu-inference
   `benchmark_utils` function or a new implementation, then register
   it in `src/maxtext/eval/scoring/registry.py`.
