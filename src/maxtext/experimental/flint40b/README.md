# Pure-JAX Flint 40B-A2.5B Trainer (64-Chip TPU v7x, 8K to 256K Context)

Standalone pure-JAX + Pallas trainer for the open-weight
[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) 40B-A2.5B
mixture-of-experts model on a 64-chip TPU v7x slice (`tpu7x:4x4x4`: 128
TensorCores, 16 hosts), with tuned recipes from 8K to 256K context. With no
flags, the Python entrypoints run the optimized 256K recipe (the same settings
as `recipes/flint_64c_256k.sh`).

## Model

| Setting | Value |
| :--- | :--- |
| Parameters | 39.79B total, 2.52B active per token |
| Layers | 49: 2 dense + 47 MoE |
| Attention | GQA (32 query / 4 KV heads, `head_dim=128`) with QK RMSNorm |
| Global / local layers | 14 global causal layers without RoPE (layers 0-1, every 4th MoE layer, and the last layer); 35 sliding-window layers (`sliding_window=513`) with RoPE (`theta=1e4`) |
| Hidden size / dense MLP | 2,048 / 6,144 |
| MoE | 256 routed experts (intermediate 512) with an FP32 sigmoid top-6 router, plus 1 shared expert (intermediate 1,024) |
| Vocabulary / max context | 150,144 / 262,144 |

## How the 256K recipe uses 128 TensorCores

*   **Mesh `(fsdp, cp, ep) = (32, 4, 1)`**: the 32 sequences per step are
    sharded over `fsdp`, and each 262,144-token sequence is split over `cp=4`
    (65,536 tokens per TensorCore). Weights are sharded over all 128
    TensorCores.
*   **Context parallelism**: a load-balanced sequence layout
    (`use_lb_cp_layout`) gives every CP rank the same causal attention work;
    sliding-window layers only exchange a small K/V halo with the neighboring CP
    rank (`splash_use_local_halo_cp`).
*   **`global_selective` remat**: saves the global-layer K/V and attention
    outputs (plus router maps and the pre-attention norm output) and recomputes
    everything else in the backward pass.
*   **FP8 matmuls** for dense and MoE projections, with FP32 master weights
    (`use_fp8`).
*   **MoE kernels**: Pallas grouped matmuls (`moe_impl=pallas_gmm_v2`) and
    SparseCore token sorting and gather-reduce (`use_sc_counting_sort`,
    `use_sc_moe_gather_reduce`).
*   **Collectives**: FSDP all-gathers and reduce-scatters are offloaded to the
    SparseCores (`sc_collective_mode=dual_sc` plus the tuned compiler flags).

## Install

Install MaxText from source (see `docs/install_maxtext.md`), from the repository
root:

```bash
uv pip install -e .[tpu] --resolution=lowest
install_tpu_pre_train_extra_deps
# PyTorch (CPU) is only needed for the unit tests under tests/:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start (default 256K recipe)

```bash
# AOT compile for tpu7x:4x4x4 plus an HBM/HLO audit. Runs on one host with
# libtpu installed and writes aot_summary.json under --out_dir.
python3 -m maxtext.experimental.flint40b.aot_compile

# Train + benchmark. Run the same command on all 16 hosts of the slice (for
# example via XPK, JobSet, or `gcloud compute tpus tpu-vm ssh --worker=all`).
python3 -m maxtext.experimental.flint40b.benchmark_train \
  --output_dir=gs://YOUR_BUCKET/flint40b/256k
```

`benchmark_train` runs 18 steps on synthetic data by default (4 warmup steps,
then 3 steps profiled with `jax.profiler`) and writes `benchmark_summary.json`,
`metrics.jsonl`, and profiler traces to `--output_dir`.

To train on pre-tokenized C4, add `--dataset_type=c4` and
`--c4_data_dir=gs://YOUR_BUCKET/c4_tokenized`. The value can be a directory, a
single file, or a glob with wildcards in the final path component only (for
example `gs://YOUR_BUCKET/c4_tokenized/*.array_record`). The shards hold
serialized `tf.train.Example` records, each with a fixed-length int64 `text`
token feature. Each training row concatenates `seq_len / record_len`
consecutive records into one sequence.

## Recipes

`recipes/flint_64c_base.sh` is the shared runner (8K defaults). Every other
recipe sets the sequence length, global batch size (GBS), and mesh, and the base
script derives the remaining settings (remat policy, FP8, attention block
sizes). All recipes use gradient accumulation 1 and FP8 matmuls. PDB = GBS / 128
sequences per TensorCore.

| Recipe | Seq len | GBS | PDB | Mesh `(fsdp, cp, ep)` | Remat |
| :--- | ---: | ---: | ---: | :--- | :--- |
| `flint_64c_base.sh` | 8,192 | 1024 | 8 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_8k.sh` | 8,192 | 1024 | 8 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_8k_pdb8.sh` | 8,192 | 1024 | 8 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_8k_pdb4.sh` | 8,192 | 512 | 4 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_8k_pdb2.sh` | 8,192 | 256 | 2 | `(128, 1, 1)` | `selective` |
| `flint_64c_8k_pdb2_ep8.sh` | 8,192 | 256 | 2 | `(16, 1, 8)` | `global_selective` |
| `flint_64c_8k_pdb1.sh` | 8,192 | 128 | 1 | `(128, 1, 1)` | `selective` |
| `flint_64c_16k.sh` | 16,384 | 512 | 4 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_16k_pdb4.sh` | 16,384 | 512 | 4 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_16k_pdb2.sh` | 16,384 | 256 | 2 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_16k_pdb1.sh` | 16,384 | 128 | 1 | `(128, 1, 1)` | `selective` |
| `flint_64c_16k_pdb05.sh` | 16,384 | 64 | 0.5 | `(64, 2, 1)` | `selective` |
| `flint_64c_32k.sh` | 32,768 | 256 | 2 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_32k_pdb2.sh` | 32,768 | 256 | 2 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_32k_pdb1.sh` | 32,768 | 128 | 1 | `(128, 1, 1)` | `global_selective` |
| `flint_64c_32k_pdb05.sh` | 32,768 | 64 | 0.5 | `(32, 4, 1)` | `global_selective` |
| `flint_64c_64k.sh` | 65,536 | 128 | 1 | `(16, 8, 1)` | `global_selective` |
| `flint_64c_64k_pdb1.sh` | 65,536 | 128 | 1 | `(16, 8, 1)` | `global_selective` |
| `flint_64c_64k_pdb05.sh` | 65,536 | 64 | 0.5 | `(16, 8, 1)` | `global_selective` |
| `flint_64c_64k_pdb025.sh` | 65,536 | 32 | 0.25 | `(16, 8, 1)` | `global_selective` |
| `flint_64c_128k.sh` | 131,072 | 64 | 0.5 | `(64, 2, 1)` | `global_selective` |
| `flint_64c_128k_pdb05.sh` | 131,072 | 64 | 0.5 | `(64, 2, 1)` | `global_selective` |
| `flint_64c_128k_pdb025.sh` | 131,072 | 32 | 0.25 | `(32, 4, 1)` | `global_selective` |
| `flint_64c_128k_pdb0125.sh` | 131,072 | 16 | 0.125 | `(16, 8, 1)` | `global_selective` |
| `flint_64c_256k.sh` | 262,144 | 32 | 0.25 | `(32, 4, 1)` | `global_selective` |
| `flint_64c_256k_pdb025.sh` | 262,144 | 32 | 0.25 | `(32, 4, 1)` | `global_selective` |
| `flint_64c_256k_pdb0125.sh` | 262,144 | 16 | 0.125 | `(16, 8, 1)` | `global_selective` |

Run the same command on every host (paths are relative to the MaxText root):

```bash
R=src/maxtext/experimental/flint40b/recipes
bash $R/flint_64c_256k.sh             # train + benchmark
bash $R/flint_64c_256k.sh --aot       # AOT compile + HBM/HLO audit
bash $R/flint_64c_256k.sh --dry-run   # print the resolved settings
C4_DATA_DIR=gs://YOUR_BUCKET/c4_tokenized bash $R/flint_64c_256k.sh --data=c4
```

Useful environment variables: `OUTPUT_DIR`, `C4_DATA_DIR`, `STEPS`, `JOB_NAME`,
`AOT_OUT_DIR`, `EXTRA_CONFIG_OVERRIDES_JSON` (a JSON object of extra config
overrides), `ENABLE_OPTIMIZED_STACK=false` (unoptimized baseline), and
`EXTRA_LIBTPU_ARGS`.

### Compiler flags

The entrypoints set `LIBTPU_INIT_ARGS` themselves and replace any existing
value. They use `--xla_tpu_scoped_vmem_limit_kib=65536`, then the tuned TPU v7x
flags from `flint_config.OPTIMIZED_V7X_LIBTPU_ARGS` (turn these off with
`--optimized_libtpu_args=false` or `OPTIMIZED_LIBTPU_ARGS=false`), then
`--extra_libtpu_args` (`EXTRA_LIBTPU_ARGS` in the recipes). The same flags are
also passed to the XLA compiler, so extra flags must be XLA compiler options.

## Measured performance

Measured in November 2026 on a 64-chip TPU v7x slice (`tpu7x:4x4x4`); results
vary with JAX and libtpu versions. Step time is the median of the steady-state
steps. All rows use pre-tokenized C4 except the synthetic row.

| Recipe | Seq len | GBS | Tokens/step | Step time | Tokens/s/chip | eMFU | Peak HBM/TensorCore (AOT) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flint_64c_8k_pdb1.sh` | 8,192 | 128 | 1,048,576 | 0.85 s | 19,281.0 | 15.68% | 59.0 GiB |
| `flint_64c_8k.sh` | 8,192 | 1024 | 8,388,608 | 3.86 s | 33,989.8 | 27.65% | 84.0 GiB |
| `flint_64c_16k.sh` | 16,384 | 512 | 8,388,608 | 4.30 s | 30,497.3 | 28.55% | 84.0 GiB |
| `flint_64c_32k.sh` | 32,768 | 256 | 8,388,608 | 5.17 s | 25,372.1 | 29.96% | 84.1 GiB |
| `flint_64c_64k.sh` | 65,536 | 128 | 8,388,608 | 7.07 s | 18,546.8 | 30.97% | 85.3 GiB |
| `flint_64c_128k.sh` | 131,072 | 64 | 8,388,608 | 10.29 s | 12,741.7 | 33.73% | 82.4 GiB |
| `flint_64c_256k.sh` | 262,144 | 32 | 8,388,608 | 17.29 s | 7,579.9 | 34.88% | 82.3 GiB |
| `flint_64c_256k.sh` (synthetic) | 262,144 | 32 | 8,388,608 | 17.35 s | 7,555.6 | 34.77% | 82.3 GiB |

eMFU = model FLOPs per step / (step time x peak FLOP/s). Model FLOPs come from
`flint_config.calculate_tflops_per_device` (6 x tokens x active matmul
parameters, plus causal attention FLOPs). Peak is the TPU v7x dense BF16 peak of
2,307 TFLOP/s per chip (1,153.5 per TensorCore); FP8 matmuls are counted against
the BF16 peak.

## Tests

CPU tests (8 simulated devices), from the MaxText repository root:

```bash
T=src/maxtext/experimental/flint40b/tests
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python3 -m pytest \
  $T/flint_config_and_flops_test.py $T/flint_data_test.py \
  $T/flint_checkpoint_test.py $T/flint_hf_parity_cpu_test.py
```

`flint_aot_compile_test.py` compiles full 64-chip programs offline; it needs
`libtpu` and about 28 GiB of host memory.

The TPU tests need a host with 4 TPU v7x chips. Set `LIBTPU_INIT_ARGS` per test:

| Test | `LIBTPU_INIT_ARGS` |
| :--- | :--- |
| `flint_attention_tpu_test.py` | `--xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_collective_permute=ENABLED` |
| `flint_experts_tpu_test.py`, `flint_sparse_layer_tpu_test.py` | `--xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true` |
| `flint_model_tpu_test.py` | The previous row's flags plus `--xla_enable_async_collective_permute=ENABLED --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false` |
| `flint_layer_perf_test.py` | `--xla_tpu_scoped_vmem_limit_kib=65472` |

## Files

| File | Contents |
| :--- | :--- |
| `flint_config.py` | Config, validation, recipe overrides, tuned compiler flags, parameter and FLOP counts |
| `flint_model.py`, `flint_layers_scan.py` | `shard_map` model forward and gradient, scanned 4-layer MoE superblocks |
| `flint_dense_layer.py`, `flint_sparse_layer.py`, `flint_mlp.py` | Dense and MoE decoder layers, SwiGLU MLP |
| `flint_attention.py`, `flint_attention_prologue.py`, `flint_splash_kernel.py` | Hybrid GQA attention, context parallelism, Pallas splash attention kernels |
| `flint_router.py`, `flint_experts.py`, `flint_gmm_v2.py`, `flint_moe_kernels.py` | MoE router, experts, Pallas grouped matmuls, SparseCore kernels |
| `flint_loss.py`, `flint_train.py` | Chunked FP32 cross-entropy, AdamW train step |
| `flint_sharding.py`, `flint_types.py`, `ops.py` | 3D `(fsdp, cp, ep)` mesh and sharding, PyTree types, shared ops |
| `data.py`, `flint_checkpoint.py` | C4 ArrayRecord and synthetic data, HuggingFace safetensors checkpoints |
| `aot_compile.py`, `benchmark_train.py` | AOT compile and audit, multi-host trainer |
| `recipes/`, `tests/` | Shell recipes and unit tests |
