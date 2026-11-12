# Lineage DeepSeek-V3 671B Recipes (64-Chip TPU v7x, 8K and 16K Context)

Lineage implementation of DeepSeek-V3 671B (+ 1 Multi-Token Prediction layer,
685B total parameters) with tuned single-slice 64-chip TPU v7x (`tpu7x:4x4x4`:
128 TensorCores, 16 hosts) pre-training recipes at 8K and 16K sequence lengths.

## Model

| Setting | Value |
| :--- | :--- |
| Parameters | 685B total (671B decoder + 1 MTP layer), ~37B active per token |
| Layers | 61 decoder (3 dense + 58 sparse MoE) + 1 sparse MTP layer (62 MLA blocks, 59 MoE blocks) |
| Attention | Multi-Head Latent Attention (128 heads, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`) with YaRN RoPE |
| Hidden size / dense MLP | 7,168 / 18,432 (SwiGLU) |
| MoE | 256 routed experts (`moe_mlp_dim=2048`), FP32 sigmoid top-8 device-limited routing (8 groups, top-4) + loss-free routing bias + sequence aux loss, plus 1 shared expert (`moe_mlp_dim=2048`) |
| Vocabulary | 129,280 (untied embedding and LM head; MTP reuses decoder input embeddings) |
| Optimizer & precision | AdamW with FP32 master weights, BF16 first moment (`mu`), FP32 second moment (`nu`), and FP8 matmuls |

## How the 64-chip recipes fit 685B parameters at 8K and 16K

*   **2D `ep8_fsdp64` sharding (`ici_parallelism=[1,4,4,4,2]`)**: places expert
    parallelism (`EP=8`, 32 routed experts per shard) on `(z, core)`, shards
    dense, MLA, shared-expert, router, and MTP weights across all 64 chips / 128
    TensorCores, splits attention heads and sequence over intra-chip `core=2`,
    and shards the 128-sequence global batch (`per_device_batch_size=1.0`,
    `gradient_accumulation_steps=1`) across all 64 chips with no cross-chip
    context parallelism (`CP=1`).
*   **Memory management**: combines streaming per-layer weight casting and
    quantization (`lineage_stream_param_cast=true`), dense and MTP checkpointing
    (`lineage_remat_dense_layers=true`, `lineage_remat_mtp=true`), chunked FP32
    LM head + cross-entropy (`lineage_loss_chunk_tokens=2048` at 8K, `4096` at
    16K), and selective async pinned-host offloading of scanned sparse-layer MLA
    (`orig_x`, `kv_down`, `splash_out`) and MoE (`out_proj_out`) activation
    residuals while keeping `q_down`, `context`, and reduced weight gradients in
    HBM (`lineage_offload_w_grad=false`) with
    `--xla_tpu_scheduler_percent_shared_memory_limit=100`, fitting within
    ~93.1 GiB per TensorCore.
*   **FP8 quantization (`fp8_full_all_splash_qk_e3_combine_per_hop` +
    `lineage_fp8_splash_out=true`)**: quantizes routed-expert Pallas grouped
    matmuls, MLA projections (2D `out_proj`), shared-expert and dense MLPs,
    Splash attention `QK^T` (with prescaled base-2 softmax), forward Splash
    attention output residuals, and the SparseCore per-hop MoE combine
    reduce-scatter.
*   **Long-context Pallas kernels**: tiles `mla_qkv_up` in VMEM with zero
    spills, runs causal Splash MLA with `2048 x 2048` blocks
    (`sa_block_kv_compute=1024`), 3-step `dQ` reduction, and diagonal-band
    skipping, and sizes dropless MoE buffers with `capacity_factor=1.25`.

## Install

Install MaxText from source (see `docs/install_maxtext.md`), from the repository
root:

```bash
uv pip install -e .[tpu] --resolution=lowest
install_tpu_pre_train_extra_deps
```

## Quick start

Run the same command on all 16 hosts of a `tpu7x:4x4x4` slice (for example via
XPK, JobSet, or `gcloud compute tpus tpu-vm ssh --worker=all`), from the MaxText
repository root:

```bash
R=src/maxtext/experimental/lineage/recipes

# Synthetic-data benchmark + profiler capture (8K or 16K)
OUTPUT_DIR=gs://YOUR_BUCKET/lineage/8k  bash $R/dsv3_64c_8k.sh
OUTPUT_DIR=gs://YOUR_BUCKET/lineage/16k bash $R/dsv3_64c_16k.sh

# Pre-tokenized C4 TFDS training (with optional Orbax checkpoint restore)
DATASET_PATH=gs://YOUR_BUCKET/c4_tfds \
LOAD_PARAMETERS_PATH=gs://YOUR_BUCKET/dsv3_ckpt/0/items \
OUTPUT_DIR=gs://YOUR_BUCKET/lineage/16k_c4 \
STEPS=61 EVAL_INTERVAL=10 \
  bash $R/dsv3_64c_16k.sh --data=c4

# Offline AOT compile + HBM/cost analysis on a single host with libtpu
bash $R/dsv3_64c_8k.sh --aot
bash $R/dsv3_64c_16k.sh --aot

# Print resolved LIBTPU_INIT_ARGS and full MaxText command without running
bash $R/dsv3_64c_16k.sh --dry-run
```

Any extra `key=value` arguments passed after the script name are appended to the
MaxText command and override recipe defaults. Useful environment variables:
`OUTPUT_DIR`, `DATASET_PATH`, `DATASET_NAME`, `EVAL_DATASET_NAME`,
`LOAD_PARAMETERS_PATH`, `STEPS`, `LEARNING_RATE`, `EVAL_INTERVAL`, `PROFILER`,
`PIPELINE_MEMORY_HOST_TRANSFERS`, `BF16_PROPAGATION`, and `EXTRA_LIBTPU_ARGS`.

## Recipes

`recipes/dsv3_64c_base.sh` is the shared runner (8K defaults); `dsv3_64c_8k.sh`
and `dsv3_64c_16k.sh` set the tuned 8K and 16K configurations. All recipes use
`gradient_accumulation_steps=1`, `per_device_batch_size=1.0` (`GBS=128` across
128 TensorCores), `ici_parallelism=[1,4,4,4,2]`, and `CP=1`, `EP=8`.

| Recipe | Seq len | PDB | GBS | Tokens/step | Layout | Loss chunk | Compiler flag overrides |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `dsv3_64c_8k.sh` (C4) | 8,192 | 1.0 | 128 | 1,048,576 | `ep8_fsdp64` | 2,048 | `--xla_tpu_pipeline_memory_host_transfers=false` |
| `dsv3_64c_8k.sh` (synthetic) | 8,192 | 1.0 | 128 | 1,048,576 | `ep8_fsdp64` | 2,048 | `--xla_tpu_pipeline_memory_host_transfers=true` |
| `dsv3_64c_16k.sh` | 16,384 | 1.0 | 128 | 2,097,152 | `ep8_fsdp64` | 4,096 | `--xla_tpu_pipeline_memory_host_transfers=false --xla_jf_bf16_propagation=false` |

### Compiler flags

`recipes/dsv3_64c_base.sh` exports `LIBTPU_INIT_ARGS` and passes matching
`compile_xla_flags` (so both live training and `--aot` compilation see the same
XLA compiler options), using the base flags from
`src/maxtext/configs/models/deepseek3-671b-lineage.yml`,
`--xla_tpu_scheduler_percent_shared_memory_limit=100` (so XLA's async
latency-hiding scheduler continues prefetching offloaded residuals and FSDP
all-gathers when `q_down` and `context` stay in HBM above the default 95% cap),
plus any sequence-level overrides and `EXTRA_LIBTPU_ARGS`. Do not pass
`--xla_tpu_aggressive_opt_barrier_removal`; Lineage's backward pass relies on
explicit optimization barriers for overlap scheduling.

At 8K, `PIPELINE_MEMORY_HOST_TRANSFERS=auto` resolves to `false` on real C4 data
(`4.24 s` steady step) and `true` on synthetic data (which avoids extra
host-to-device staging wait on unsharded synthetic batches). At 16K,
`BF16_PROPAGATION=false` (`--xla_jf_bf16_propagation=false`) lowers peak HBM by
~7.9 GiB per TensorCore so `q_down`, `context`, and reduced sparse weight
gradients stay in HBM within the `94.7 GiB` TensorCore limit (`93.1 GiB` peak).

An alternative 8K configuration (`LINEAGE_LAYOUT_4X4X4=default
LINEAGE_LOSS_CHUNK_TOKENS=8192 BF16_PROPAGATION=false
PIPELINE_MEMORY_HOST_TRANSFERS=false`, measured with 5-field MLA offload) reaches
`4.37 s` (`3,752.1 tokens/s/chip`, `47.47%` BF16 MFU, `88.2 GiB` peak
HBM/TensorCore).

## Measured performance

Measured on a single 64-chip TPU v7x slice (`tpu7x:4x4x4`, 128 TensorCores);
results vary with JAX and `libtpu` versions. Unoptimized 64-chip baselines are
included for comparison.

| Configuration | Data | Seq len | PDB / GBS | Tokens/step | Step time | Tokens/s/chip | BF16 MFU | Peak HBM/TC |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8K baseline | C4 | 8,192 | 1.0 / 128 | 1,048,576 | 6.28 s | 2,607.7 | 32.99% | 89.9 GiB |
| **`dsv3_64c_8k.sh` (steady / profiled)** | **C4** | **8,192** | **1.0 / 128** | **1,048,576** | **4.24 s / 4.26 s** | **3,861.7 / 3,848.2** | **48.85% / 48.68%** | **92.7 GiB** |
| 16K baseline (`CP=2`) | C4 | 16,384 | 0.5 / 64 | 1,048,576 | 8.36 s | 1,959.3 | 30.09% | 88.4 GiB |
| 16K baseline (`CP=1`) | C4 | 16,384 | 1.0 / 128 | 2,097,152 | 12.40 s | 2,642.9 | 40.58% | 89.9 GiB |
| **`dsv3_64c_16k.sh` (steady / profiled)** | **C4** | **16,384** | **1.0 / 128** | **2,097,152** | **9.21 s / 9.23 s** | **3,559.0 / 3,550.0** | **54.65% / 54.51%** | **93.1 GiB** |

With full 5-field MLA residual offloading (`LINEAGE_OFFLOAD_MLA_FIELDS=orig_x,q_down,kv_down,context,splash_out`,
default 95% scheduler memory cap), `dsv3_64c_8k.sh` measured `4.35 s` steady on
C4 (`4.88 s` C4 / `4.92 s` synthetic with `PIPELINE_MEMORY_HOST_TRANSFERS=true`)
and `dsv3_64c_16k.sh` measured `9.31 s` steady on C4 (`9.49 s` on synthetic
data, `88.7 GiB` peak HBM/TC).

*   **MFU accounting**: `BF16 MFU = model_flops_per_step / (step_time * 64 *
    2,307 TFLOP/s)` (`306.03 PFLOP/step` at 8K `GBS=128`; `742.95 PFLOP/step` at
    16K `GBS=128`, counting `6 * N_active * B * T` active matmul FLOPs plus
    exact causal attention FLOPs against the TPU v7x dense BF16 peak of `2,307
    TFLOP/s/chip`; divide by 2 for utilization relative to the `4,614
    TFLOP/s/chip` FP8 peak).
*   **Convergence verification**: verified over 61 C4 training steps from a
    pretrained DeepSeek-V3 checkpoint (`learning_rate=2.4e-5`, evaluated every
    10 steps). At 8K, the recipe matches baseline validation loss within ~1 step
    (2-run mean `3.7407` vs `3.7278` baseline at step 60; `3.7260` and `3.7554`
    across runs). At 16K, the recipe reaches `3.6034` vs `3.6371` for the `CP=1`
    baseline (~3.3 steps ahead). Retaining `q_down` and `context` in HBM with
    `--xla_tpu_scheduler_percent_shared_memory_limit=100` is numerically
    lossless and produces identical training loss against same-commit controls.
