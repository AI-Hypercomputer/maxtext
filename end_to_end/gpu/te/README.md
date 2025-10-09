# MaxText + Transformer Engine E2E Benchmarking

This directory contains scripts for testing MaxText with Transformer Engine (TE) integration across different parallelization strategies and quantization recipes.

Requirements:
- NVIDIA MaxText image with installed Transformer Engine (TE). Suggested to use the latest version of `ghcr.io/nvidia/jax:maxtext`.
- `test-maxtext.sh` script which is available in the suggested image. Otherwise, you can get it (here)[https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/test-maxtext.sh].
- NVIDIA GPU(s) with compute capability 9.0 or higher for FP8 quantization, 10.0 or higher for MXFP8 quantization.

## Quick Start

### 1. Run Individual Tests

#### MaxText Baseline with FP8
```bash
MAXTEXT_DIR=/path/to/maxtext bash test-maxtext.sh --data-parallel=1 --tensor-sequence-parallel=1 --fsdp=1 --quantization=fp8 --model llama3.1-8b --steps 100
```

#### TE with DelayedScaling FP8
```bash
MAXTEXT_DIR=/path/to/maxtext bash test-maxtext.sh --data-parallel=1 --tensor-sequence-parallel=1 --fsdp=1 --quantization=te_fp8_delayedscaling --model llama3.1-8b --steps 100
```

#### TE with CurrentScaling FP8
```bash
MAXTEXT_DIR=/path/to/maxtext bash test-maxtext.sh --data-parallel=1 --tensor-sequence-parallel=1 --fsdp=1 --quantization=te_fp8_currentscaling --model llama3.1-8b --steps 100
```

#### TE with MXFP8 Block Scaling
```bash
MAXTEXT_DIR=/path/to/maxtext bash test-maxtext.sh --data-parallel=1 --tensor-sequence-parallel=1 --fsdp=1 --quantization=te_mxfp8 --model llama3.1-8b --steps 100
```

#### Enable Profiling/Tracing
Add profiling arguments to collect XPlane traces (only the last step is traced):
```bash
MAXTEXT_DIR=/path/to/maxtext bash test-maxtext.sh --data-parallel=1 --tensor-sequence-parallel=1 --fsdp=1 --quantization=te_fp8_delayedscaling --model llama3.1-8b --steps 100 --additional-args="profiler=xplane skip_first_n_steps_for_profiler=99 profiler_steps=1"
```

### 2. Run Comprehensive Benchmarking

The `run_single_node_model_parallel.sh` script automatically tests all quantization recipes across multiple parallelization strategies:

#### Basic Usage
```bash
bash run_single_node_model_parallel.sh --model llama3.1-8b --steps 100
```

#### With Tracing Enabled
```bash
bash run_single_node_model_parallel.sh --model llama3.1-8b --steps 100 --trace true
```

#### Collecting traces with custom number of decoder layers
```bash
bash run_single_node_model_parallel.sh --model llama3.1-8b --steps 100 --trace true --num-decoder-layers 4
```

#### Skip Single GPU Tests
```bash
bash run_single_node_model_parallel.sh --model llama3.1-8b --steps 100 --single-gpu-run false
```