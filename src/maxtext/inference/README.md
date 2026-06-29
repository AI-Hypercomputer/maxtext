<!--
 # Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# maxtext.inference

This subpackage provides high-performance inference capabilities for large language models on TPU and GPU hardware.

Currently, vLLM is the recommended inference engine, providing optimized execution and batching. The alternative
inference engines are deprecated and will be removed in future releases.

See [Inference on MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/tutorials/inference.html) for a tutorial on
using vLLM with MaxText models.

## Components

### `vllm_decode.py`

Offline inference script that runs a single prompt through a MaxText model using vLLM as the execution backend. Useful
for verifying checkpoint correctness and quickly testing model outputs before deploying an online server.

**Two execution modes**, selected via the `--use_tunix` flag:

- **Default (`--use_tunix=False`)**: loads the model through the MaxText vLLM adapter plugin (`MaxTextForCausalLM`) and
  runs inference with vLLM's `LLM` API. Supports tensor parallelism, data parallelism, expert parallelism (MoE), and HBM
  utilization control.

- **Tunix (`--use_tunix=True`)**: loads the model directly with `model_creation_utils.from_pretrained`, wraps it in
  `TunixMaxTextAdapter`, and runs inference via `VllmRollout`.

## Other utilities

The following utilities are provided for advanced inference use-cases, but are not required for standard offline or
online inference.

### `decode.py`

CLI utility for single-stream or batched inference on a model.

### `decode_multi.py`

CLI utility for interleaved prefill/generate with multiple concurrent streams.

### `offline_engine.py`

Batch inference engine for high-throughput offline processing. Handles input padding and sorting, prefill packing, and
continuous batching (interleaved prefill/decode with background detokenization) with structured completion tracking.

### `kvcache.py`

Key-value cache implementation with per-layer management, quantization support (INT4, INT8, FP8), configurable
quantization axes, and cache transformation utilities.

### `inference_utils.py`

Common utilities for MaxText inference, including sampling and log probability calculations.

### `inference_microbenchmark.py` / `inference_microbenchmark_sweep.py`

Benchmarking tools for measuring prefill and autoregressive decode performance with configurable parameter sweeps.

## Subdirectories

### `maxengine/` (legacy)

Core inference engine and serving infrastructure. Supports prefill/autoregressive generation, KV-cache quantization,
sampling strategies, prefix caching, and multimodal inputs. This submodule is kept for legacy purposes. Inference
through vLLM is now preferred.

### `mlperf/`

MLPerf inference benchmark implementation with offline/server scenarios, dataset preparation, accuracy evaluation, and
quantization checkpoint generation. See [mlperf/README.md](mlperf/README.md).

### `gpu/`

Baseline benchmarking scripts with fixed parameters for GPU (H100). See [gpu/README.md](gpu/README.md).

### `jetstream_pathways/`

Contains a script to build MaxText + JetStream + Pathways Server image. See [jetstream_pathways/README.md](jetstream_pathways/README.md).

### `scripts/`

Helper scripts for model sharding, decode orchestration, and analysis notebooks.
