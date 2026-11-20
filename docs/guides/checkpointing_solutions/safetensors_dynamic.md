(safetensors-dynamic)=

# Dynamic SafeTensors loading

This guide explains how to use `source_checkpoint_layout="safetensors_dynamic"` to load Hugging Face (HF) `.safetensors` checkpoints directly into sharded MaxText weights at startup—transforming tensor layouts and stacking layers or experts on device without an offline [checkpoint conversion](checkpoint-conversion) step or an intermediate Orbax copy in Google Cloud Storage (GCS).

It is supported across **pre-training** (`train.py`), **post-training** (`from_pretrained` in RL, SFT, DPO, and Distillation with explicit `teacher_overrides.load_parameters_path`), and **inference and evaluation** (`MaxEngine`, `vLLM` via `vllm_decode.py --use_tunix=True` or unpadded `MaxTextForCausalLM`, and `forward_pass_logit_checker`).

## Comparison with other checkpointing methods

| Method                                                                         | How it works                                                                                      | Extra GCS storage                                                                                                                           | Peak HBM during load                                                     |
| :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
| **Offline `to_maxtext` + Orbax**                                               | Convert `.safetensors` offline on CPU, write Orbax to GCS, then restore Orbax at startup          | 2x (HF SafeTensors + Orbax checkpoint)                                                                                                      | Target MaxText weights                                                   |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=False`, default)**     | Read `.safetensors` sharded directly onto devices, then transform and stack in memory             | None initially (0x duplicate Orbax copy until first save; `hf://` caches raw shards under `<base_output_directory>/hf_cache/<org>_<repo>/`) | Whole HF checkpoint (sharded) + MaxText weights (~2x weights)            |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=True`, experimental)** | Stream `.safetensors` in bounded layer chunks and place directly into preallocated target weights | None initially (0x duplicate Orbax copy until first save)                                                                                   | MaxText target weights + bounded streaming read buffer (~8 GiB per host) |

Use `safetensors_dynamic` for cold starts from Hugging Face `.safetensors` weights, evaluation, post-training, and large models where single-host CPU `to_maxtext.py` hits host RAM limits or duplicate multi-terabyte Orbax copies are undesirable. Across **175** paired cold-start configurations on TPU `v7x`, `v6e`, `v5p`, and `v5e`, `safetensors_dynamic` was faster in **168 of 175** runs (**1.8x** median end-to-end speedup; **4.1x** excluding HF Hub download), and loaded **7 of 7** tested `>200B` models up to `3.2 TB` (`2.4x–3.2x` faster end-to-end where `to_maxtext.py` succeeded). For repeated restarts when a pre-converted Orbax checkpoint already exists, restoring with `source_checkpoint_layout="orbax"` was faster in **147 of 175** configurations (median `2.9x`), except for `24 of 40` scanned configurations of models `>=30 GB` and `qwen3.5-397b-a17b` unscanned on `v7x`.

## Prerequisites

- MaxText must be installed in a Python virtual environment (or container image). For installation instructions, refer to the [installation documentation](install-from-source).
- The target model architecture must be supported by MaxText checkpoint conversion (see [Checkpoint Conversion Utilities](checkpoint-conversion) and [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py) for `model_name` and `HF_IDS` entries).
- Multi-controller JAX (McJAX) must be used (`enable_single_controller=False`). Pathways single-controller workloads are not supported with `safetensors_dynamic`.
- Set the `HF_TOKEN` environment variable (recommended over the deprecated `hf_access_token` CLI flag) if loading gated models from the Hugging Face Hub, and provide a `gs://` `base_output_directory` (without a trailing slash) for Hub shard caching.

## Loading from a GCS or local directory

### Setup environment

```bash
export MODEL="llama3.1-70b"
export HF_CKPT_PATH="gs://my-bucket/hf_checkpoints/Llama-3.1-70B/"
export BASE_OUTPUT_DIRECTORY="gs://my-bucket/output"
export RUN_NAME="my_dynamic_run"
```

### Run training

Point `load_parameters_path` to a directory containing `.safetensors` files (on `gs://` or local disk) and set `source_checkpoint_layout="safetensors_dynamic"`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=${MODEL?} \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path=${HF_CKPT_PATH?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    weight_dtype=bfloat16
```

### What happens on restart

At startup, MaxText checks the run's own checkpoint directory (`${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/`) first and only uses `safetensors_dynamic` when no checkpoint exists for `run_name`. With `enable_checkpointing=True` and the default `save_checkpoint_on_start=True`, MaxText writes an Orbax checkpoint at step 0, so restarting with the same `run_name` resumes from that Orbax checkpoint and ignores `load_parameters_path` (pointing `load_parameters_path` at a new `.safetensors` directory while reusing `run_name` has no effect, and Orbax storage is created from step 0 onward).

## Loading directly from Hugging Face Hub

When `load_parameters_path` is a Hugging Face repository ID (`hf://<org>/<repo>` or `<org>/<repo>`) or is left unset, Host 0 downloads any missing `.safetensors` shards (plus `config.json` when `safetensors_dynamic_streaming=True`) into `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` (for example, `hf_cache/meta-llama_Llama-3.1-8B-Instruct/`) using a parallel worker pool while all other hosts wait until caching finishes. Blobs already present in that GCS directory are skipped, so subsequent runs sharing the same `base_output_directory` and repository are a cache hit and do not re-download.

If `load_parameters_path` is left unset, MaxText resolves the exact `model_name` key (preserving suffixes such as `-Instruct`) via `HF_IDS` in [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py).

### Setup environment

```bash
export MODEL="llama3.1-8b-Instruct"
export HF_REPO="hf://meta-llama/Llama-3.1-8B-Instruct"
export BASE_OUTPUT_DIRECTORY="gs://my-bucket/output"  # Must use gs:// without trailing /
export RUN_NAME="my_hf_run"
export HF_TOKEN="<your-huggingface-token>"  # Required for gated repos
```

### Run training

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=${MODEL?} \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path=${HF_REPO?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    weight_dtype=bfloat16
```

> **Note:** To use the default `HF_IDS` repository for `model_name`, leave `load_parameters_path` unset on the CLI instead of passing `hf://...`. Passing `load_parameters_path=""` explicitly on the CLI fails config validation because the CLI parser converts empty values to `None`.

## Streaming mode (experimental)

On very large models where holding both the sharded Hugging Face tensors and the converted MaxText weights simultaneously would exceed device HBM (or where `tf.io.gfile` read-ahead exceeds host RAM on fine-grained MoE models), set `safetensors_dynamic_streaming=True`. MaxText preallocates target arrays and streams decoder layers incrementally in bounded per-host chunks (`~4 GiB/host`).

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=qwen3.5-397b-a17b \
    source_checkpoint_layout="safetensors_dynamic" \
    safetensors_dynamic_streaming=True \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Qwen3.5-397B-A17B/" \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    weight_dtype=bfloat16
```

## Inference and logit verification

Both `model_creation_utils.from_pretrained` and the [forward pass logit checker](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/utils/forward_pass_logit_checker.py) support `source_checkpoint_layout="safetensors_dynamic"`:

```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    model_name=gemma3-4b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/gemma-3-4b-it/" \
    weight_dtype=bfloat16 \
    enable_checkpointing=false \
    --run_hf_model=True \
    --hf_model_path="google/gemma-3-4b-it" \
    --max_kl_div=0.015
```

## Quantized DeepSeek checkpoints (`FP8` and `MXFP4`)

Official Hugging Face checkpoints for large DeepSeek models (`DeepSeek-V3`, `DeepSeek-V3.2`, and `DeepSeek-V4`) store block-quantized `FP8` (`*_scale_inv`) or hybrid `FP8` + packed `MXFP4` (`F8_E8M0` `*.scale`) weights, whereas MaxText's `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-284b`, and `deepseek4-1600b` models expect `BF16` (or `FP32`) weights:

- **Current workflow:** Convert quantized shards offline to `BF16` SafeTensors using `src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py`, then point `load_parameters_path` at the `BF16` GCS directory. With `safetensors_dynamic_streaming=True`, `hf_streaming_load` also supports `None`-mapped synthesized weights (used by `deepseek4-*`) and reads `quantization_config` from the checkpoint's `config.json` so `BF16` exports do not trigger unscaled-FP8 errors.
- **On-the-fly dequantization (in review):** Direct shard-level `FP8` and `MXFP4` -> `BF16` dequantization is tracked in open [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) (default Orbax path) and [PR #5802](https://github.com/AI-Hypercomputer/maxtext/pull/5802) (`DeepSeek-V4.1-Flash` on the streaming path).

## How it works

Instead of downloading an entire checkpoint onto a single CPU host, converting it sequentially in host RAM, and writing a second Orbax copy to GCS, `safetensors_dynamic` splits loading across all hosts in a cluster and converts weights directly on accelerators:

### 1. Weights-only target tree and mapping resolution

- At startup, `checkpointing.load_state_if_possible` (or `model_creation_utils.from_pretrained`) passes only the parameter leaves (`_abstract_params` or NNX parameter state, excluding optimizer state) into `load_safetensors_dynamic_state`.
- MaxText resolves the exact `model_name` key (preserving `-Instruct` suffixes for `HF_IDS` lookups) against `HF_MODEL_CONFIGS`, `PARAM_MAPPING`, and `HOOK_FNS` to build the Hugging Face-to-MaxText key map and per-tensor transform hooks. When `safetensors_dynamic_streaming=True`, `config.json` in the checkpoint directory overrides stock `quantization_config`.
- Any older multimodal checkpoint prefixes (for example legacy Gemma 3 `language_model.model.` or `vision_tower.` keys) are normalized transparently to match current `PARAM_MAPPING` keys.

### 2. Optional Host-0 Hugging Face Hub → GCS caching

When `load_parameters_path` points to a Hub repo (`hf://<org>/<repo>`, `<org>/<repo>`, or is left unset to use `HF_IDS[model_name]`):

- **Single-host API discovery**: Only Host 0 (`jax.process_index() == 0`) lists remote files via `HfFileSystem` (`*.safetensors`, plus `config.json` when `safetensors_dynamic_streaming=True`). Avoiding per-host HF API calls prevents HTTP `429` rate limiting on `128+`-host jobs.
- **Parallel cache population**: Host 0 lists existing objects under `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/`, skips any blobs already present, and downloads only missing files using a `32`-worker spawned process pool (`32 MB` upload chunks, up to `5 s` random startup jitter, and up to `5` retries per file with `15 s` backoff).
- **Cluster barrier**: All other hosts wait on `jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")` before reading from GCS.

### 3. Default path (`safetensors_dynamic_streaming=False`): Distributed contiguous reads and in-memory transforms

- **Contiguous leading-axis sharded I/O**: SafeTensors stores tensors in row-major order. On multi-process runs, `_leading_axis_shardings` shards each tensor along **axis `0` only** over a 2D `("process", "local")` mesh so every host reads a single contiguous byte range per tensor. Smaller tensors that cannot divide across all devices are sharded across processes only (then replicated locally) or replicated.
- **Eliminating GCS over-reads**: Multi-process reads pass `SafetensorsOptions(max_over_read_ratio=1.0)` (preventing Orbax from falling back to full-span reads on MoE files with `>1,024` tensors) and temporarily set `gcsfs.GCSFileSystem.default_block_size = 0` to disable `gcsfs`'s `5 MiB` read-ahead buffer during small ranged reads.
- **Refcounted hooks, stacking, and chunked resharding**:
  - As `transform_hf_state_to_mt_state` processes each target key, a `remaining_uses` counter immediately pops each HF tensor from memory after its final use (handling HF tensors shared across multiple leaves, such as fused gate/up projections split into `wi_0` and `wi_1`).
  - Hooks run on local device shards whenever possible; if a NumPy-only hook hits a tensor split across hosts, MaxText falls back to `1 GiB` sliced cross-host gather/scatter.
  - Decoder layer and MoE expert lists are concatenated via balanced binary-tree stacking (`_binary_chunked_stack`) to avoid quadratic memory growth, preserving variable collections (`MoEBiasVar`, `Tid2EidVar`) in their expected dtype (for example FP32 router biases), and arrays larger than `1 GiB` are resharded in chunks to bound peak HBM.
  - Finally, `_raise_on_weight_mismatch` verifies every expected parameter leaf in `abstract_params` was materialized.

### 4. Streaming path (`safetensors_dynamic_streaming=True`, experimental)

- **Single-pass headers & `64`-thread `SafetensorsReader`**: Parses `.safetensors` headers once on initialization and reads slices directly onto local TPU chips via `32 MiB` ranged GCS requests across `64` threads per host (including `F8_E8M0` scale dtypes and zero-padded row reads for hooks marked `pads_rows_to_target`). Groups of small sharded tensors whose per-chip slice would be under `4 MiB` (such as `16 MB` MoE experts) are read whole across devices and rearranged over ICI.
- **Double-buffered layer-by-layer writes**: Decoder layers (including `segments_<k>-layers_<j>` groups) and `None`-mapped synthesized weights are packed into `~4 GiB/host` read calls (`READ_BYTES_PER_HOST`). While the current chunk is transformed on device and written in place into preallocated target arrays via jitted `dynamic_update_slice` (bounding in-flight writes to `256`), a background thread prefetches the next chunk onto host/device memory, keeping transient overhead near two read chunks (`~8 GiB/host`).

## HF Hub-to-GCS cache download time

When loading from Hugging Face Hub for the first time, **Host 0** transfers shards from Hugging Face Hub into `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` before multi-host loading begins. On subsequent runs sharing the same `base_output_directory` and repo (or when pointing `load_parameters_path` directly at an existing `gs://` directory), existing blobs are skipped and download time is **0 s**.

| Model               | Checkpoint Size | Cold HF→GCS Download (s) | `safetensors_dynamic` Load + Transform (s) | Cold E2E Total (s)  | Download Share of Cold Startup |
| :------------------ | :-------------- | :----------------------- | :----------------------------------------- | :------------------ | :----------------------------- |
| `llama3.1-8b`       | 16.1 GB         | 62.91                    | 28.52                                      | 91.43               | 68.8%                          |
| `qwen3-30b-a3b`     | 61.1 GB         | 78.82                    | 129.52                                     | 208.34              | 37.8%                          |
| `qwen3-32b`         | 65.5 GB         | 107.24                   | 88.92                                      | 196.16              | 54.7%                          |
| `mixtral-8x7b`      | 93.4 GB         | 239.75                   | 63.12                                      | 302.87              | 79.2%                          |
| `llama3.1-70b`      | 141.1 GB        | 197.80                   | 104.98                                     | 302.78              | 65.3%                          |
| `gpt-oss-120b`      | 233.7 GB        | 350.33                   | 390.69                                     | 741.02              | 47.3%                          |
| `qwen3-235b-a22b`   | 470.4 GB        | 583.31                   | 697.13 (unscanned) / 757.50 (scanned)      | 1,280.44 / 1,340.81 | 45.6% / 43.5%                  |
| `qwen3.5-397b-a17b` | 795.5 GB        | 814.53                   | 954.34 (unscanned)                         | 1,768.87            | 46.0%                          |
| `qwen3-480b-a35b`   | 961.0 GB        | 1,046.14                 | 718.65 (unscanned) / 757.70 (scanned)      | 1,764.79 / 1,803.84 | 59.3% / 58.0%                  |

> **Tip:** On large multi-host TPU allocations, pre-populating `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` (or a regional `gs://` SafeTensors directory) before launching multi-slice training avoids spending cluster-wide accelerator time waiting on Host 0's HF Hub download.

## Performance and tested models

Because all hosts read shards concurrently over GCS and transform weights on accelerators without serializing a full Orbax checkpoint first, `safetensors_dynamic` cuts cold-start latency across single-slice and multi-slice topologies.

### Parity summary

- **Two-slice bit-exact parity (`<=120B` suite):** Across `44` registered `<=120B` models with valid `to_maxtext` baselines on TPU `v6e`/`v5e`/`CPU`, **42 of 44** produced bit-exact logits (`forward_pass_logit_checker --atol=0 --rtol=0`) against offline `to_maxtext` checkpoints (the remaining `2` differed only by `float32`/`bfloat16` cast rounding and passed HF golden-logit checks within `max KL <= 0.017`).
- **Training loss and checkpoint parity:** 10-step `float32` C4 training on `llama3.1-8b` (`v6e-32`) produced identical step-by-step losses, and automated 4-way E2E checks across `1x`/`2x` `v5p` verified `0.0` maximum absolute difference across weights, step-0/step-1 Orbax checkpoints, and C4 losses on `qwen3-0.6b`, `gemma2-2b`, `qwen3-30b-a3b`, and `llama3.1-70b`.

### Cross-accelerator summary (`v7x`, `v6e`, `v5p`, `v5e`, `v4`, `CPU`, `GPU`)

Across **175** paired `<=120B` and `>200B` comparisons spanning `v7x`, `v6e`, `v5p`, and `v5e` (`21` models, `0.6B–1.6T`), `safetensors_dynamic` was faster end-to-end in **168 of 175** configurations (median **1.8x** E2E, **4.1x** excluding HF download):

| Accelerator       | Hardware tested                                         | Key benchmark findings                                                                                                                                                              |
| :---------------- | :------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TPU `v7x`**     | `1–8` hosts (`1 TB` host RAM); `1–2x tpu7x-128`         | `<=120B` faster E2E in `53–54/56` pairs across sweeps (`1.7–1.8x` median E2E, `3.1–3.2x` excl. DL); loaded `7/7` `>200B` real BF16 models (`2.4–3.2x` faster where `to_maxtext` ran). |
| **TPU `v6e`**     | `1–64` hosts (`760.5–1,522 GB` RAM/host)                | Faster E2E in `55/56` `<=120B` pairs (`1.6x` median E2E, `4.8x` excl. DL); loaded `7/7` `>200B` models (`deepseek3-671b` in `363.0 s`, `deepseek4-1600b` in `649.1 s` unscanned). |
| **TPU `v5p`**     | `1–16` hosts (`473.3 GB` RAM/host)                      | Faster E2E in `47/54` `<=120B` pairs (`1.5x` median E2E, `2.1x` excl. DL); loaded `7/7` `>200B` (`0` OOMs vs. `0/7` for `to_maxtext`). Layer streaming loads `397B` in `115.0 s` (`6.9 GB/s` on `v5p-64`) and `284B` in `163.5 s` (`3.5 GB/s` on `v5p-32`). |
| **TPU `v5e`**     | `2x v5e-32`, `v5e-256` (`64h`), `2x v5e-256` (`128h`)   | Two-slice parity (`42/44` bit-exact); with `gcsfs`, loads `235B–671B` in `229.9–474.8 s` (`53.9–80.8 GB` host RAM) on 64 hosts and `1.6T` unscanned in `1,245.2 s` on 128 hosts. |
| **TPU `v4`**      | `v4-8` (`1` host, `4` chips)                            | `SafetensorsReader` streams `Qwen3-8B` in `10.8–11.9 s` and `DeepSeek-V2-Lite` in `28.5–33.2 s` (`~2.3 GB/s` vs. `~1.1 GB/s` Orbax).                                               |
| **CPU**           | Single-host CPU VM                                      | **7/7** tested small/multimodal models (`0.6B–9B`) match `to_maxtext` bit-exact (`--atol=0 --rtol=0`) and HF within `max KL <= 2.9e-4`.                                             |
| **GPU (`H100`/`B200`)** | Not benchmarked                                   | Uses standard JAX `Mesh` / `NamedSharding` and `sync_global_devices`; GPU validation tracked separately.                                                                            |

### Representative models `<=120B` (TPU `v7x`, 2 hosts / 16 devices)

| Model           | Size (GB) | Config (Scan, Devices) | HF→GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (s) | E2E Speedup | Excl. Download Speedup |
| :-------------- | --------: | :--------------------- | ---------: | :------------------------------ | :----------------------------- | ----------: | ---------------------: |
| `llama3.1-8b`   |      16.1 | Scanned, `16`          |       90.9 | 83.0 + 23.5                     | **35.0**                       |       1.57x |                  3.04x |
| `qwen3-30b-a3b` |      61.1 | Scanned, `16`          |      181.0 | 187.2 + 636.9                   | **425.2**                      |       1.66x |                  1.94x |
| `mixtral-8x7b`  |      93.4 | Scanned, `16`          |      287.6 | 222.6 + 901.5                   | **98.7**                       |       3.65x |                 11.38x |
| `llama3.1-70b`  |     141.1 | Scanned, `16`          |      397.4 | 890.5 + 395.8                   | **110.6** (`72.9 + 37.4`)      |       3.31x |                 11.63x |
| `gpt-oss-120b`  |     233.7 | Unscanned, `16`        |      618.0 | 756.6 + 203.6                   | **405.0** (`133.4 + 271.1`)    |       1.54x |                  2.37x |

### Large models `>200B` (TPU `v7x`, 1–8 hosts / 8–64 devices)

On `>200B` models, single-host CPU `to_maxtext` failed or exceeded single-host RAM (`1,014.5 GB`) on all scanned runs and all models `>=480B`, whereas `safetensors_dynamic` loaded **7 of 7** real `BF16` GCS checkpoints up to **3.2 TB**:

| Model                | Size (GB) | Config (Scan, Hosts / Devices) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Outcome / Speedup (E2E / Excl. DL) |
| :------------------- | --------: | :----------------------------- | :------------------------------ | :----------------------------------------------- | :--------------------------------- |
| `qwen3-235b-a22b`    |     470.4 | Scanned, `1h / 8`              | Exceeds single-host RAM         | **830.7 s** (`718.2 + 108.9`)                    | Only `safetensors_dynamic` loads   |
| `qwen3-235b-a22b`    |     470.4 | Unscanned, `1h / 8`            | 3,043.0 + 685.5                 | **733.0 s** (`676.4 + 52.6`)                     | **2.56x** / **5.09x**              |
| `deepseek4-284b`\*   |     581.9 | Scanned, `2h / 16`             | OOM-killed (single-host RAM)    | **963.9 s** (`856.3 + 105.2`)                    | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Scanned, `2h / 16`             | Exceeds single-host RAM         | **1,456.5 s** (with PR #6353)                    | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Unscanned, `2h / 16`           | 7,877.3 + 2,065.8 (with PR #6354) | **1,120.9 s**                                  | **3.22x** / **8.87x**              |
| `qwen3-480b-a35b`    |     961.0 | Scanned, `2h / 16`             | Exceeds single-host RAM         | **1,109.6 s**                                    | Only `safetensors_dynamic` loads   |
| `deepseek3-671b`\*   |   1,369.0 | Scanned, `4h / 32`             | OOM-killed (`~990 GB`)          | **1,319.5 s** (`1,123.3 + 191.6`)                | Only `safetensors_dynamic` loads   |
| `deepseek3.2-671b`\* |   1,370.7 | Scanned, `4h / 32`             | OOM-killed (`~990 GB`)          | **1,529.7 s** (`1,323.2 + 201.9`)                | Only `safetensors_dynamic` loads   |
| `deepseek4-1600b`\*  |   3,200.0 | Scanned, `8h / 64`             | OOM-killed (single-host RAM)    | **2,492.8 s** (`2,175.5 + 312.1`)                | Only `safetensors_dynamic` loads   |

*\* DeepSeek rows load dequantized `BF16` SafeTensors from regional GCS (see [Quantized DeepSeek checkpoints](#quantized-deepseek-checkpoints-fp8-and-mxfp4)). End-to-end GCS load + 2-step C4 training on real `BF16` weights was additionally verified on `1–2` `tpu7x-128` slices (`16–32` hosts) for `llama3.1-405b` (`36 s` read / `25 s` transform), `qwen3.5-397b-a17b` (`51 s` / `587 s`), `qwen3-480b-a35b` (`191 s` / `68 s`), `deepseek4-284b` (`296 s` / `150 s`), and `deepseek3-671b` (`252 s` / `48 s`).*

## MaxText configuration

| Flag                                            | Description                                                                                                                                                                                           | Type                            | Default     |
| :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------ | :---------- |
| `source_checkpoint_layout`                      | Specifies the source checkpoint format to load. Set to `"safetensors_dynamic"` to load and convert Hugging Face `.safetensors` weights directly at startup.                                           | `string`                        | `"orbax"`   |
| `safetensors_dynamic_streaming`                 | Experimental. When `True` with `source_checkpoint_layout="safetensors_dynamic"`, streams decoder layers sequentially to limit transient HBM usage.                                                    | `boolean`                       | `False`     |
| `load_parameters_path`                          | Path to a `gs://` or local `.safetensors` directory, `hf://<org>/<repo>`, or `<org>/<repo>`. Leave unset to use `HF_IDS[model_name]`; passing `load_parameters_path=""` on the CLI fails validation.  | `string`                        | `""`        |
| `base_output_directory`                         | Output directory for training artifacts. Must use a `gs://` prefix (without a trailing `/`) when caching weights from Hugging Face Hub (`hf_cache/`).                                                 | `string`                        | `""`        |
| `hf_access_token`                               | Hugging Face authentication token used by Host 0 when downloading from private or gated Hugging Face repositories. Prefer setting the `HF_TOKEN` environment variable.                                | `string` or `None`              | `""`        |
| `weight_dtype`                                  | Target weight dtype in MaxText (for example `"bfloat16"`). We recommend `"bfloat16"` when loading BF16 Hugging Face checkpoints to reduce memory usage.                                               | `string`                        | `"float32"` |
| `scan_layers`                                   | Controls scanned (`True`) or unscanned (`False`) decoder layer layout (including nested `-local_layers` block scans).                                                                                 | `boolean`                       | `True`      |
| `enable_single_controller`                      | Enables Pathways single-controller execution. Must remain `False` for `safetensors_dynamic`.                                                                                                          | `boolean`                       | `False`     |
| `use_m3_model`                                  | Not supported with `safetensors_dynamic`.                                                                                                                                                             | `boolean`                       | `False`     |
| `moe_flat_fsdp_weights` / `prefuse_moe_weights` | Not supported with `safetensors_dynamic` (keep `False`).                                                                                                                                              | `boolean`                       | `False`     |
| `checkpoint_conversion_fn`                      | Used only by `source_checkpoint_layout="safetensors"`; ignored by `"safetensors_dynamic"`.                                                                                                            | `string`, `callable`, or `None` | `None`      |
| `enable_single_replica_ckpt_restoring`          | Single-replica restore optimization for Orbax checkpoints; not used by `safetensors_dynamic`.                                                                                                         | `boolean`                       | `False`     |

## Troubleshooting

- **Pathways error (`enable_single_controller=True`)**: `source_checkpoint_layout="safetensors_dynamic"` raises a `ValueError` on Pathways. Either run on McJAX (`enable_single_controller=False`) or convert the checkpoint offline with `to_maxtext.py` and load it with `source_checkpoint_layout="orbax"`.
- **Missing `gs://` prefix or trailing `/` on `base_output_directory` for HF Hub downloads**: Set `base_output_directory="gs://<bucket>/output"` (without a trailing slash; a trailing `/` produces `gs://.../output//hf_cache/...`) so Host 0 can stage shards to shared GCS storage, or download `.safetensors` files to a local/GCS directory beforehand.
- **High host RAM on fine-grained MoE models when TensorFlow is installed**: With `safetensors_dynamic_streaming=False`, `etils.epath` defaults to `tf.io.gfile` (`64 MiB` read-ahead buffers per open file) when TensorFlow is installed. Set `export EPATH_USE_TF=false` so `epath` routes `gs://` reads through `gcsfs`, or set `safetensors_dynamic_streaming=True`.
- **Quantized DeepSeek checkpoints (`*_scale_inv` or `F8_E8M0` `*.scale`) fail or produce degraded outputs**: Convert shards offline to `BF16` with `standalone_scripts/deepseek_dequantize.py` (for `deepseek4-*`, enable `safetensors_dynamic_streaming=True` for `None`-key support), or track open [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) / [PR #5802](https://github.com/AI-Hypercomputer/maxtext/pull/5802).
- **Changes to `load_parameters_path` ignored on restart**: If an Orbax checkpoint already exists under `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/` (created at step 0 when `enable_checkpointing=True` and `save_checkpoint_on_start=True`), restarts resume from Orbax and ignore `load_parameters_path`. Use a new `run_name` or clear the run's checkpoint directory to reload from `.safetensors`.
- **Weight-mismatch errors on startup or under `vLLM`**: Verify that `model_name` (including suffixes such as `-Instruct`), `scan_layers`, `use_multimodal`, and any model overrides match the Hugging Face checkpoint, keep `moe_flat_fsdp_weights=False` and `prefuse_moe_weights=False`, and under vLLM use `vllm_decode.py --use_tunix=True` (or unpadded `MaxTextForCausalLM`).
