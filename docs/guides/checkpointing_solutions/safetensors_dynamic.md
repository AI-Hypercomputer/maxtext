<!--
 Copyright 2023-2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Dynamic SafeTensors Loading (`safetensors_dynamic`)

`source_checkpoint_layout="safetensors_dynamic"` loads Hugging Face (HF) `.safetensors` checkpoints **directly into sharded MaxText weights in host and accelerator memory at startup**, without running offline `to_maxtext` conversion or writing an intermediate Orbax checkpoint to GCS.

It is supported across **pre-training** (`train.py` -> `load_state_if_possible`), **post-training** (`from_pretrained` in RL, SFT, DPO, Distillation with explicit `teacher_overrides.load_parameters_path`, and `MaxTextTrainingEngine`), and **inference and evaluation** (`MaxEngine`, `vLLM` via `vllm_decode.py --use_tunix=True` or unpadded `MaxTextForCausalLM`, `forward_pass_logit_checker`, and `zero_gen_logits_checker`), scaling from single-host CPU/TPU runs up to **128+ hosts** and **1.6T-parameter (`3.2 TB`)** checkpoints.

## Why use `safetensors_dynamic`?

| Dimension                               | Offline `to_maxtext` + Orbax                                                                                                                                                                                                                                                             | `safetensors_dynamic` (on-the-fly)                                                                            |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **Workflow**                            | Download HF -> run single-host CPU `to_maxtext` -> write Orbax to GCS -> restore Orbax                                                                                                                                                                                                   | Point `load_parameters_path` at `gs://`, local dir, or `hf://` -> load directly into MaxText                  |
| **GCS storage**                         | Requires `2x` storage (HF SafeTensors + converted Orbax copy)                                                                                                                                                                                                                            | Requires **`0x` extra checkpoint storage** (`hf://` caches raw `.safetensors` once)                           |
| **Single-host RAM limit**               | Exceeds single-host RAM at `>=141 GB` on `v5e` (`203 GB`), `>=234 GB` scanned / `>=470 GB` unscanned on `v5p` (`473 GB`), `>=470 GB` scanned / `>=582 GB` unscanned on `v6e` (`761 GB`), and `>=470 GB` scanned / `582 GB` (`deepseek4-284b`) and `>=961 GB` unscanned on `v7x` (`1 TB`) | Shards reads across all hosts along axis `0`; verified up to **`3.2 TB` (`deepseek4-1600b`)**                 |
| **Startup latency (cold / single run)** | Unscanned CPU conversion accounts for **75–99%** (`95%` median) of non-download startup time                                                                                                                                                                                             | **`1.5–1.8x` median speedup** (`<=120B`, `2.1–4.8x` excl. download) and **`1.85–3.22x`** (`>200B`) end-to-end |
| **Repeated runs on small / mid models** | Pre-converted unscanned Orbax restore alone is faster (`2.0–7.4x` platform medians, `~3.5x` pooled) for repeated `<=120B` restarts                                                                                                                                                       | Best for large models (`>200B`), cold HF experiments, evals, post-training, and storage-free workflows        |

## Quick start

### 1. Load from a GCS or local SafeTensors directory

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_dynamic_run \
    model_name=llama3.1-70b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Llama-3.1-70B/" \
    base_output_directory="gs://my-bucket/output"
```

> **Note:** Loaded weights are cast to `weight_dtype` (default `"float32"` in `base.yml`). Pass `weight_dtype=bfloat16` to keep BF16 checkpoints in BF16 and halve weight HBM usage.

### 2. Load directly from Hugging Face Hub (`hf://` or auto-lookup)

When loading from the Hugging Face Hub, **Host 0** lists (`HfFileSystem.glob`) and uploads missing `.safetensors` files in parallel (`32` spawned workers, `32 MiB` chunks) into `<base_output_directory>/hf_cache/<org>_<repo>/`, then synchronizes all hosts via `jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")`.

```bash
export HF_TOKEN=<your-token>  # Required for gated repos such as Llama and Gemma.

python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_hf_run \
    model_name=llama3.1-8b-Instruct \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="hf://meta-llama/Llama-3.1-8B-Instruct" \
    base_output_directory="gs://my-bucket/output"
```

If `load_parameters_path` is omitted, MaxText looks up the **exact** `model_name` (preserving `-Instruct` suffixes) in `HF_IDS`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_auto_hf_run \
    model_name=qwen3-8b \
    source_checkpoint_layout="safetensors_dynamic" \
    base_output_directory="gs://my-bucket/output"
```

### 3. Inference and logit verification (`from_pretrained`)

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

### 4. Quantized checkpoints (DeepSeek `FP8` and `MXFP4`)

Official Hugging Face checkpoints for large DeepSeek models (`deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3.2`, `deepseek-ai/DeepSeek-V4-Flash`, and `deepseek-ai/DeepSeek-V4-Pro`) publish quantized `FP8` weights (`F8_E4M3` + `F32` `*_scale_inv` `128x128` block scales on V3/V3.2) or hybrid `FP8` + packed `MXFP4` (`I8`, two 4-bit `E2M1` values per byte) weights with `F8_E8M0` `*.scale` tensors on V4, whereas MaxText's `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-284b`, and `deepseek4-1600b` models expect `BF16` (or `FP32`) weights:

- **Current workflow (`main`):** Neither `to_maxtext` nor `safetensors_dynamic` dequantizes `FP8`/`MXFP4` weights on the fly yet (`safetensors_dynamic` skips `*_scale_inv` scales on raw `FP8` and rejects `F8_E8M0` headers, while `to_maxtext` leaves scales unapplied and fails reshaping packed `I8` `MXFP4` bytes). First convert the HF shards offline to `BF16` SafeTensors using `src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py` (`581.9 GB` for `284B`, `1.37 TB` for `671B` / `3.2-671B`, `3.20 TB` for `1600B`), then point `load_parameters_path` at the `BF16` directory. On `main` today, `deepseek3-671b` and `deepseek3.2-671b` load directly from those `BF16` directories, while `deepseek4-284b` and `deepseek4-1600b` additionally require open [PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591) or [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) for synthesized `None` mapping keys (`mhc_norm-scale`) in `hf_streaming_load`.
- **On-the-fly dequantization (under review):** [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) adds direct per-shard `FP8` and `MXFP4` -> `BF16` dequantization, `F8_E8M0` scale reads, and `None`-key support inside `safetensors_dynamic` (see also open [PR #5806](https://github.com/AI-Hypercomputer/maxtext/pull/5806) for `F8_E8M0` + `quantization_config` plumbing and [PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591) for `None` mapping support), eliminating offline `BF16` staging and reducing GCS reads by `2.0–3.7x`.

## Configuration reference and constraints

| Flag                                            | Requirement / behavior                                                                                                                                                                                  | Type                            | Default                     |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------ | :-------------------------- |
| `source_checkpoint_layout`                      | Set to `"safetensors_dynamic"` for on-the-fly HF SafeTensors loading.                                                                                                                                   | `string`                        | `"orbax"`                   |
| `load_parameters_path`                          | `gs://` directory, local directory on every host, `hf://<org>/<repo>`, or `<org>/<repo>`. Omit this flag to use `HF_IDS[model_name]` (exact `model_name` match; do not pass an empty `""` on the CLI).  | `string`                        | `""` (unset)                |
| `base_output_directory`                         | **Must start with `gs://`** (without a trailing slash) when downloading from HF Hub (`hf://` or `HF_IDS` auto-lookup); shards cache under `<base_output_directory>/hf_cache/<org>_<repo>/`.             | `string`                        | `""`                        |
| `HF_TOKEN` / `hf_access_token`                  | Required for gated Hugging Face repositories (such as Llama and Gemma). Prefer the `HF_TOKEN` environment variable (`hf_access_token` CLI flag is deprecated).                                          | `string` or `None`              | `None` (`""` in `base.yml`) |
| `weight_dtype`                                  | Loaded tensors are cast to each target leaf's `weight_dtype` (pass `bfloat16` to preserve BF16 checkpoint memory footprints; also supports `float32`).                                                  | `string`                        | `"float32"`                 |
| `scan_layers`                                   | Works in both `scan_layers=True` (scanned) and `scan_layers=False` (unscanned) modes, plus nested `-local_layers` block scans (such as Gemma 4 and Qwen3-Next).                                         | `boolean`                       | `True`                      |
| `enable_checkpointing`                          | Works with `enable_checkpointing=False` (`jax.distributed` is initialized automatically on multi-host runs). When `True`, resuming an interrupted `run_name` restores its saved Orbax checkpoint first. | `boolean`                       | `True` (`types.py`)         |
| `enable_single_controller`                      | **Not supported.** Config validation rejects `source_checkpoint_layout="safetensors_dynamic"` with `enable_single_controller=True` (Pathways single-controller).                                        | `boolean`                       | `False`                     |
| `use_m3_model`                                  | **Not supported.** M3 models (`src/maxtext/m3/`) are incompatible with `safetensors_dynamic` (explicit error guard under review in [PR #5594](https://github.com/AI-Hypercomputer/maxtext/pull/5594)).  | `boolean`                       | `False`                     |
| `moe_flat_fsdp_weights` / `prefuse_moe_weights` | **Not supported.** On-load 2D MoE FSDP flattening and gate/up `wi` pre-fusion apply only to Orbax restores.                                                                                             | `boolean`                       | `False`                     |
| `checkpoint_conversion_fn`                      | Used only by `source_checkpoint_layout="safetensors"`; ignored by `"safetensors_dynamic"`, which uses `PARAM_MAPPING` and `HOOK_FNS` automatically.                                                     | `string`, `callable`, or `None` | `None`                      |
| `enable_single_replica_ckpt_restoring`          | Applies only to Orbax checkpoint restores; unused by `safetensors_dynamic`.                                                                                                                             | `boolean`                       | `False`                     |

## Architecture overview

Implemented in `src/maxtext/checkpoint_conversion/utils/` (`load_dynamic.py`, `hf_streaming_load.py`, `safetensors_reader.py`, and `tensor_handling.py`), execution proceeds through five stages:

1. **Phase 0 — Host-0 HF Hub cache (optional):** Only `jax.process_index() == 0` queries the HF API (`HfFileSystem.glob`) and uploads missing `.safetensors` shards into `<base_output_directory>/hf_cache/<org>_<repo>/` using `32` spawned workers (`32 MiB` chunks), preventing cluster-wide HTTP `429` throttling on `128+` host jobs. All hosts wait on `sync_global_devices("dynamic_hf_download_complete")`.
2. **Phase 1 — Weights-only target tree and mappings:** Callers pass weights only (`_abstract_params` / NNX `State`), avoiding optimizer-state key flattening issues. `get_hf_config_and_mappings` strips any `"-Instruct"` suffix to look up `HF_MODEL_CONFIGS`, `PARAM_MAPPING`, and `HOOK_FNS` (while `HF_IDS` preserves `"-Instruct"` for repository lookup).
3. **Phase 2 — Streaming plan and parallel header inspection:** `build_plan` resolves layer and expert stack axes (`single_stack_axis`, `stacked_axes`, and `MoEBiasVar` / `Tid2EidVar` variable collections) and checks that stacked targets are completely filled. `SafetensorsReader` then parses `.safetensors` headers in parallel across `64` threads, `rename_legacy_hf_keys` normalizes legacy multimodal prefixes (such as Gemma 3's `language_model.model.*`), `_check_sources` verifies every mapped HF tensor exists, and `group_into_read_calls` packs whole decoder layers into read calls of ~`4 GiB` per host (`READ_BYTES_PER_HOST`).
4. **Phase 3 — Contiguous leading-axis ranged reads:** Each HF tensor is split along axis `0` (`load_split_factor = gcd(dim 0, num_devices)`, or `shared_split_factor` across multi-input hooks) so every host fetches contiguous row-major byte ranges without strided over-reads or cross-host collectives (tensors `< 1 MiB` or coprime to device count are replicated). Neighboring ranges within `1 MiB` are merged and fetched via `32 MiB` ranged reads across `64` threads per host directly into HBM.
5. **Phase 4 — On-device transforms, in-place writes, and verification:** A background thread prefetches the next read call while `PARAM_MAPPING` hooks run on the accelerators via `_apply_hook_fns_any_host` (with `1 GiB` sliced `process_allgather` fallback for NumPy-only hooks). A jitted writer casts each tensor to `weight_dtype` and updates preallocated stacked weights in place via `dynamic_update_slice`, followed by `_raise_on_weight_mismatch` completeness checks.

### Active optimizations under review

- **On-the-fly `FP8` and `MXFP4` DeepSeek dequantization ([PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398)):** Streams `F8_E8M0` scales and dequantizes `FP8` (`128x128` block scales) and packed `MXFP4` (`I8` nibbles + `1x32` `F8_E8M0` scales) -> `BF16` per shard on TPU via `jax.shard_map` with zero cross-host collectives, cutting GCS reads by `2.0–3.7x` across DeepSeek-V3, V3.2, and V4.
- **Synthesized `None` mappings ([PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591)) and `F8_E8M0` / quantization-config plumbing ([PR #5806](https://github.com/AI-Hypercomputer/maxtext/pull/5806)):** Supports synthesized `None`-mapped weights (such as DeepSeek-V4 `mhc_norm-scale`) in `hf_streaming_load`, `F8_E8M0` scale reads, zero-padded row splits, and forwarding `quantization_config` to hooks.
- **Stacked whole-tensor reads for small per-expert tensors ([PR #5589](https://github.com/AI-Hypercomputer/maxtext/pull/5589)), `from_pretrained` / M3 guards ([PR #5594](https://github.com/AI-Hypercomputer/maxtext/pull/5594)), and streaming opt-in flag ([PR #6410](https://github.com/AI-Hypercomputer/maxtext/pull/6410)):** Reads groups of small per-expert tensors whole (`>=4 MiB`) before unpacking over ICI, preserves native `lora_a` / `lora_b` buffers in `from_pretrained`, rejects incompatible `use_m3_model=True` upfront (bundled in rollup [PR #6223](https://github.com/AI-Hypercomputer/maxtext/pull/6223)), and evaluates gating layer-streaming behind `safetensors_dynamic_streaming`.

## Correctness and parity results

- **Two-slice bit-exact parity (`<=120B` suite):** Across the **47** `<=120B` models registered in `PARAM_MAPPING` at validation time, 44 had valid `to_maxtext` baselines (excluding `weaver-mini` and `weaver-max`, where `to_maxtext` fails on a missing `norm.weight` key, and duplicate alias `qwen3.5-35b-fp8`):
  - **42 / 44 models** matched `to_maxtext` logits **bit-exact** (`forward_pass_logit_checker --atol=0 --rtol=0`).
  - **2 / 44 models** (`qwen3.5-35b-a3b-fp8`, `gemma2-27b`) passed against HF (`max KL <= 0.017`) and differed from `to_maxtext` only by `float32`/`bfloat16` cast rounding.
- **Single-slice TPU `v6e-32` and CPU logit parity:** On CPU, all **7** tested small and multimodal models (`qwen3-0.6b`, `qwen2.5-1.5b`, `gemma2-2b`, `gemma3-4b`, `gemma4-e2b`, `qwen3.5-9b`, `qwen3-vl-2b`) matched `to_maxtext` logits bit-exact (`--atol=0 --rtol=0`) and HF golden logits within `max KL <= 2.9e-4` (`<= 8.6e-5` on 6 of 7; `1.4e-7` on `qwen3-0.6b`). On TPU `v6e-32` (`8` hosts), **8 of 9** tested models matched `to_maxtext` bit-exact and `qwen3.5-35b-a3b-fp8` matched HF within `max KL 1.6e-2`.
- **Training loss and multi-slice checkpoint parity:** **10-step `float32` training on C4** (`llama3.1-8b` on `TPU v6e-32`, 8 hosts) produced identical `learning/loss` at every step compared with training from an offline `to_maxtext` Orbax checkpoint. Automated 4-way comparisons across `1x`/`2x` `v5p-8`, `v5p-16`, and `v5p-32` (`dcn_data_parallelism=2`) verified `0.0` maximum absolute difference across in-memory weights, step-0 and step-1 Orbax checkpoints, and C4 losses on `qwen3-0.6b`, `gemma2-2b`, `qwen3-30b-a3b`, and `llama3.1-70b`.
- **Large-model (`>140B`) validation:** Forward-pass token probes (`"Paris is the capital of"` -> `" France"`, `"Monday, Tuesday, Wednesday,"` -> `" Thursday"`) and 2-step C4 training passed across `mixtral-8x22b` (`1.952`), `qwen3-235b-a22b` (`2.356`), `llama3.1-405b` (`1.744`), `qwen3.5-397b-a17b` (`3.602`; also `max KL 0.00157` vs. HF on `v5p-64`), `qwen3-480b-a35b` (`2.443`), and dequantized `BF16` copies of `deepseek4-284b` (`2.661`) and `deepseek3-671b` (`2.153`); `deepseek3.2-671b` (`1.37 TB`) and `deepseek4-1600b` (`3.20 TB`) `BF16` checkpoints were validated for multi-host weight loading across `v7x`, `v6e`, `v5p`, and `v5e` (`deepseek4-*` requires open [PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591) or [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) on `main` for `None` mapping keys).

## Benchmarks by accelerator type

### Accelerator coverage summary

| Accelerator             | Hardware profile tested                                                                                          | Models & scope evaluated                                                                                               | Key findings                                                                                                                                                                                                                                        |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TPU `v7x` (`tpu7x`)** | 1–8 hosts (`4` chips / `8` devices, `~814 GB` HBM, `1,014.5 GB` RAM/host); `1–2x tpu7x-128` (`16–32` hosts)      | **21 models (`0.6B–1.6T`)**, scanned & unscanned (`56` pairs `<=120B`, `7` models `>200B` real BF16 GCS)               | Faster E2E in **53–54/56** `<=120B` pairs across two sweeps (median **1.7–1.8x** E2E, **3.1–3.2x** excl. download). Loaded **7/7** models `>200B` (vs. **2/7** for `to_maxtext`), **2.39–3.22x** faster E2E (**4.35–8.87x** excl. download).        |
| **TPU `v6e`**           | 1–64 hosts (`32 GiB` HBM/chip; `8` chips / `1,522 GB` RAM on `2x4`, `4` chips / `760.5 GB` RAM on `4x4`–`16x16`) | **21 models (`0.6B–1.6T`)** `to_maxtext` comparison + `v6e-32` parity & 10-step `float32` C4 training                  | Faster E2E in **55/56** `<=120B` pairs (median **1.6x** E2E, **4.8x** excl. download). Loaded **7/7** models `>200B` (vs. **1/7** for `to_maxtext`), including `671B` in **363 s** and `1.6T` (`3.20 TB`) in **649 s** unscanned on `1x16x16`.      |
| **TPU `v5p`**           | 1–16 hosts (`4` chips/host, `95 GiB` HBM/chip, `473.3 GB` RAM/host): `2x2x1` through `4x4x4`                     | **21 models (`0.6B–1.6T`)** `to_maxtext` comparison (`164` rows) + 4-way E2E parity + layer-streaming (`284B`, `397B`) | Faster E2E in **47/54** `<=120B` pairs (median **1.5x** E2E, **2.1x** excl. download). Loaded **7/7** models `>200B` (**0 OOMs**, vs. **0/7** for `to_maxtext`). Layer streaming loads `397B` (`792.7 GB`) in **115.0 s** (`6.9 GB/s`) on `v5p-64`. |
| **TPU `v5e`**           | `2x v5e-32` (`16`h), `v5e-256` (`64`h), `2x v5e-256` (`128`h); `16 GiB` HBM/chip, `202.6 GB` RAM/host            | **2-slice `PARAM_MAPPING` parity (`42/44` bit-exact)** + `0.6B–1.6T` multi-host GCS loads                              | On MoE runs with `gcsfs`, loads `235B–671B` in **230–638 s** (`54–81 GB` host RAM) on 64 hosts and `1.6T` (`3.20 TB`) unscanned in **1,245 s** (`102.3 GB` RAM) on 128 hosts.                                                                       |
| **TPU `v4`**            | `v4-8` (`1` host, `4` chips)                                                                                     | `SafetensorsReader` layer-streaming throughput & step-0 loss (`Qwen3-8B`, `DeepSeek-V2-Lite`)                          | Streams `Qwen3-8B` in **10.8–11.9 s** and `DeepSeek-V2-Lite` in **28.5–33.2 s** (`~2.3 GB/s` GCS reader vs. `~1.1 GB/s` Orbax SafeTensors), matching step-0 loss.                                                                                   |
| **CPU**                 | Single-host CPU VM                                                                                               | Small & multimodal parity (`0.6B–9B`) and unit parity tests                                                            | **7/7** bit-exact (`--atol=0 --rtol=0`) vs. `to_maxtext` and `max KL <= 2.9e-4` vs. HF (`<= 8.6e-5` on 6 of 7).                                                                                                                                     |
| **GPU (`H100`/`B200`)** | Not yet benchmarked                                                                                              | Out of scope in current TPU validation suite                                                                           | Uses standard JAX `Mesh` / `NamedSharding` and `sync_global_devices`; GPU validation tracked separately.                                                                                                                                            |

> **Benchmark methodology note:** In the comparison tables below, **E2E** includes cold `HF -> GCS` download (`Step 1`), while **Excl. DL** compares `to_maxtext` (single-host CPU convert + Orbax load) against `safetensors_dynamic` load, excluding `to_maxtext` local shard staging from both ratios. The 21-model paired matrices in §1–4 measured the two-stage Orbax-backed loader (`Read + Transform`, whereas §3.C and §5 measure the `SafetensorsReader` layer-streaming pipeline; total load times also include `~0.3–6 s` of header planning and barrier sync). `to_maxtext` runs capped `checkpoint_storage_concurrent_gb` (`20–45 GB`, open [PR #6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354)), and `gpt-oss-120b` / `qwen3.5-397b-a17b` scanned runs included open [PR #6353](https://github.com/AI-Hypercomputer/maxtext/pull/6353). DeepSeek rows marked `*` load dequantized `BF16` SafeTensors from regional GCS (see [Quantized checkpoints](#4-quantized-checkpoints-deepseek-fp8-and-mxfp4)).

### 1. TPU `v7x` (`tpu7x`) benchmarks

Host hardware: `4` chips (`8` TPU devices), `~814 GB` usable HBM, and `1,014.5 GB` host RAM per host. Across all 14 models `<=120B` (`56` paired configurations across 1 and 2 slices, run across two sweeps), `safetensors_dynamic` was faster E2E in **53–54 / 56** runs (median **1.7–1.8x** E2E, **3.1–3.2x** excl. download):

#### A. Representative models `<=120B` (TPU `v7x`, 2 hosts / 16 devices)

| Model           | Size (GB) | Config (Scan, Topology (Devices)) | HF -> GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM (GB) | Speedup (E2E / Excl. DL) |
| :-------------- | --------: | :-------------------------------- | ------------: | :------------------------------ | :----------------------------------------------- | -----------------: | :----------------------- |
| `qwen3-0.6b`    |       1.5 | Scanned, `2x2x2x1` (16)           |          33.7 | 19.4 + 3.0                      | **21.6 s**                                       |               73.0 | **1.01x** / **1.03x**    |
| `qwen3-0.6b`    |       1.5 | Unscanned, `2x2x2x1` (16)         |          33.7 | 70.6 + 3.6                      | **18.4 s**                                       |               72.6 | **2.07x** / **4.03x**    |
| `llama3.1-8b`   |      16.1 | Scanned, `2x2x2x1` (16)           |          90.9 | 83.0 + 23.5                     | **35.0 s**                                       |               67.5 | **1.57x** / **3.04x**    |
| `llama3.1-8b`   |      16.1 | Unscanned, `2x2x2x1` (16)         |          90.9 | 113.6 + 14.9                    | **33.8 s**                                       |               67.5 | **1.76x** / **3.80x**    |
| `qwen3-30b-a3b` |      61.1 | Scanned, `2x2x2x1` (16)           |         181.0 | 187.2 + 636.9                   | **425.2 s**                                      |              156.6 | **1.66x** / **1.94x**    |
| `qwen3-30b-a3b` |      61.1 | Unscanned, `2x2x2x1` (16)         |         181.0 | 318.6 + 39.4                    | **381.0 s**                                      |              156.4 | 0.96x / 0.94x            |
| `mixtral-8x7b`  |      93.4 | Scanned, `2x2x2x1` (16)           |         287.6 | 222.6 + 901.5                   | **98.7 s**                                       |               71.6 | **3.65x** / **11.38x**   |
| `mixtral-8x7b`  |      93.4 | Unscanned, `2x2x2x1` (16)         |         287.6 | 318.8 + 48.7                    | **71.5 s**                                       |               73.3 | **1.82x** / **5.14x**    |
| `llama3.1-70b`  |     141.1 | Scanned, `2x2x2x1` (16)           |         397.4 | 890.5 + 395.8                   | **110.6 s** (72.9 + 37.4)                        |               73.3 | **3.31x** / **11.63x**   |
| `llama3.1-70b`  |     141.1 | Unscanned, `2x2x2x1` (16)         |         397.4 | 628.0 + 53.2                    | **87.7 s** (72.6 + 14.8)                         |               71.2 | **2.22x** / **7.77x**    |
| `gpt-oss-120b`  |     233.7 | Scanned, `2x2x2x1` (16)           |         618.0 | 662.1 + 1,225.7                 | **460.2 s** (139.9 + 318.0)                      |               71.3 | **2.32x** / **4.10x**    |
| `gpt-oss-120b`  |     233.7 | Unscanned, `2x2x2x1` (16)         |         618.0 | 756.6 + 203.6                   | **405.0 s** (133.4 + 271.1)                      |               70.2 | **1.54x** / **2.37x**    |

*Note: On 1 host (`1x2x2x1`), `qwen3-30b-a3b` loads in `146.9 s` scanned (`3.02x` E2E) and `124.8 s` unscanned (`1.80x` E2E); the 2-host `381.0 s` read above reflects `tf.io.gfile` `64 MiB` read-ahead across `18,432` per-expert tensors without `gcsfs` (compare `v5p` below at `126.7 s` / `2.38x` E2E with `gcsfs`).*

#### B. Large models `>200B` (TPU `v7x`, 1–8 hosts / 8–64 devices)

Offline `to_maxtext` succeeded on only **2 of 7** models (`qwen3-235b-a22b` unscanned and `qwen3.5-397b-a17b` unscanned, the latter using open [PR #6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354)), failing or exceeding single-host RAM on all scanned `>200B` runs, `deepseek4-284b` unscanned (`~990 GB`), and all models `>=480B`. `safetensors_dynamic` loaded **7 of 7** models:

| Model                | Size (GB) | Config (Scan, Topology (Hosts / Devices)) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM / HBM (GB) | Outcome / Speedup                  |
| :------------------- | --------: | :---------------------------------------- | :------------------------------ | :----------------------------------------------- | :----------------------- | :--------------------------------- |
| `qwen3-235b-a22b`    |     470.4 | Scanned, `1x2x2x1` (1h / 8)               | Exceeds single-host RAM         | **830.7 s** (718.2 + 108.9)                      | 392.0 / 80.2             | Only `safetensors_dynamic` loads   |
| `qwen3-235b-a22b`    |     470.4 | Unscanned, `1x2x2x1` (1h / 8)             | 3,043.0 + 685.5                 | **733.0 s** (676.4 + 52.6)                       | 391.6 / 59.2             | **2.56x** E2E / **5.09x** excl. DL |
| `deepseek4-284b`\*   |     581.9 | Scanned, `2x2x2x1` (2h / 16)              | OOM-killed (single-host RAM)    | **963.9 s** (856.3 + 105.2)                      | 291.2 / 45.6             | Only `safetensors_dynamic` loads   |
| `deepseek4-284b`\*   |     581.9 | Unscanned, `2x2x2x1` (2h / 16)            | OOM-killed (single-host RAM)    | **855.2 s** (788.6 + 63.1)                       | 294.1 / 39.7             | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Scanned, `1x2x2x2` (2h / 16)              | Exceeds single-host RAM         | **1,456.5 s**                                    | 52.9 / 69.2              | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Unscanned, `1x2x2x2` (2h / 16)            | 7,877.3 + 2,065.8               | **1,120.9 s**                                    | 61.8 / 55.0              | **3.22x** E2E / **8.87x** excl. DL |
| `qwen3-480b-a35b`    |     961.0 | Scanned, `2x2x2x1` (2h / 16)              | Exceeds single-host RAM         | **1,109.6 s**                                    | 414.9 / 81.6             | Only `safetensors_dynamic` loads   |
| `qwen3-480b-a35b`    |     961.0 | Unscanned, `2x2x2x1` (2h / 16)            | Exceeds single-host RAM         | **1,060.2 s**                                    | 416.6 / 62.1             | Only `safetensors_dynamic` loads   |
| `deepseek3-671b`\*   |   1,369.0 | Scanned, `2x2x2x2` (4h / 32)              | OOM-killed (~990 GB)            | **1,319.5 s** (1,123.3 + 191.6)                  | 352.8 / 57.4             | Only `safetensors_dynamic` loads   |
| `deepseek3-671b`\*   |   1,369.0 | Unscanned, `2x2x2x2` (4h / 32)            | OOM-killed (~990 GB)            | **1,209.7 s** (1,085.4 + 120.2)                  | 354.1 / 44.8             | Only `safetensors_dynamic` loads   |
| `deepseek3.2-671b`\* |   1,370.7 | Scanned, `2x2x2x2` (4h / 32)              | OOM-killed (~990 GB)            | **1,529.7 s** (1,323.2 + 201.9)                  | 354.0 / 57.5             | Only `safetensors_dynamic` loads   |
| `deepseek3.2-671b`\* |   1,370.7 | Unscanned, `2x2x2x2` (4h / 32)            | OOM-killed (~990 GB)            | **1,235.7 s** (1,107.1 + 124.4)                  | 354.5 / 44.8             | Only `safetensors_dynamic` loads   |
| `deepseek4-1600b`\*  |   3,200.0 | Scanned, `1x2x4x4` (8h / 64)              | OOM-killed (single-host RAM)    | **2,492.8 s** (2,175.5 + 312.1)                  | 400.4 / 66.3             | Only `safetensors_dynamic` loads   |
| `deepseek4-1600b`\*  |   3,200.0 | Unscanned, `1x2x4x4` (8h / 64)            | OOM-killed (single-host RAM)    | **2,203.2 s** (2,028.8 + 168.0)                  | 398.4 / 58.8             | Only `safetensors_dynamic` loads   |

*End-to-end GCS load + 2-step C4 training on real `BF16` weights was additionally verified on `1–2` `tpu7x-128` slices (`16–32` hosts): `llama3.1-405b` (`36 s` read / `25 s` transform, step-0 loss `1.744`), `qwen3.5-397b-a17b` (`51 s` / `587 s`, loss `3.602`), `qwen3-480b-a35b` (`191 s` / `68 s`, loss `2.443`), `deepseek4-284b`\* (`296 s` / `150 s`, loss `2.661`), and `deepseek3-671b`\* (`252 s` / `48 s`, loss `2.153`).*

### 2. TPU `v6e` benchmarks

Host hardware: `32 GiB` HBM per chip on `ct6e-standard-8t` (`8` chips/host, `1,522 GB` host RAM on `2x4`) and `ct6e-standard-4t` (`4` chips/host, `760.5 GB` host RAM on `4x4`, `8x8`, and `16x16`). Across all `56` paired `<=120B` configurations, `safetensors_dynamic` was faster E2E in **55 / 56** runs (median **1.6x** E2E, **4.8x** excl. download):

#### A. Representative models `<=120B` (TPU `v6e`, 16 chips)

| Model           | Size (GB) | Config (Scan, Topology (Devices)) | HF -> GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM (GB) | Speedup (E2E / Excl. DL) |
| :-------------- | --------: | :-------------------------------- | ------------: | :------------------------------ | :----------------------------------------------- | -----------------: | :----------------------- |
| `qwen3-0.6b`    |       1.5 | Scanned, `2x2x4` (16)             |          91.4 | 15.1 + 2.0                      | **13.0 s** (4.9 + 7.7)                           |               77.5 | **1.04x** / **1.31x**    |
| `qwen3-0.6b`    |       1.5 | Unscanned, `2x2x4` (16)           |          91.4 | 205.8 + 2.7                     | **8.4 s** (4.7 + 3.2)                            |               76.3 | **3.00x** / **24.86x**   |
| `llama3.1-8b`   |      16.1 | Scanned, `2x2x4` (16)             |         299.5 | 56.2 + 15.6                     | **24.1 s** (9.1 + 14.7)                          |               71.3 | **1.15x** / **2.98x**    |
| `llama3.1-8b`   |      16.1 | Unscanned, `2x2x4` (16)           |         299.5 | 226.3 + 8.3                     | **16.8 s** (9.0 + 7.3)                           |               71.3 | **1.69x** / **13.93x**   |
| `qwen3-30b-a3b` |      61.1 | Scanned, `2x2x4` (16)             |         245.4 | 156.4 + 434.2                   | **314.6 s** (268.5 + 43.6)                       |              161.0 | **1.49x** / **1.88x**    |
| `qwen3-30b-a3b` |      61.1 | Unscanned, `2x2x4` (16)           |         245.4 | 493.4 + 27.7                    | **305.5 s** (271.6 + 30.9)                       |              164.5 | **1.39x** / **1.71x**    |
| `mixtral-8x7b`  |      93.4 | Scanned, `2x2x4` (16)             |         587.8 | 165.3 + 619.6                   | **59.4 s** (41.4 + 17.5)                         |               74.0 | **2.12x** / **13.20x**   |
| `mixtral-8x7b`  |      93.4 | Unscanned, `2x2x4` (16)           |         587.8 | 378.5 + 38.0                    | **54.9 s** (42.8 + 11.3)                         |               74.5 | **1.56x** / **7.59x**    |
| `llama3.1-70b`  |     141.1 | Scanned, `2x2x4` (16)             |       1,038.4 | 642.8 + 306.9                   | **88.0 s** (58.3 + 29.2)                         |               75.4 | **1.77x** / **10.79x**   |
| `llama3.1-70b`  |     141.1 | Unscanned, `2x2x4` (16)           |       1,038.4 | 885.8 + 42.2                    | **72.9 s** (57.7 + 14.4)                         |               75.3 | **1.77x** / **12.73x**   |
| `gpt-oss-120b`  |     233.7 | Scanned, `1x4x4` (16)             |       1,029.1 | 480.9 + 453.3                   | **232.1 s** (47.7 + 183.8)                       |               36.2 | **1.56x** / **4.03x**    |
| `gpt-oss-120b`  |     233.7 | Unscanned, `1x4x4` (16)           |       1,029.1 | 856.5 + 70.7                    | **191.2 s** (48.5 + 140.8)                       |               35.6 | **1.60x** / **4.85x**    |

#### B. Large models `>200B` (TPU `v6e`, 16–64 hosts / 64–256 chips)

Single-host `to_maxtext` converted only `qwen3-235b-a22b` unscanned and exceeded single-host RAM on all larger configurations. `safetensors_dynamic` loaded **7 of 7** models, loading `deepseek3-671b`\* (`1.37 TB`) in **363.0 s** and `deepseek4-1600b`\* (`3.20 TB`) in **649.1 s** unscanned on `1x16x16` (`64` hosts, with `gcsfs`):

| Model                | Size (GB) | Config (Scan, Topology (Hosts / Devices)) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM / HBM (GB) | Outcome / Speedup                  |
| :------------------- | --------: | :---------------------------------------- | :------------------------------ | :----------------------------------------------- | :----------------------- | :--------------------------------- |
| `qwen3-235b-a22b`    |     470.4 | Scanned, `1x8x8` (16h / 64)               | Exceeds single-host RAM         | **1,301.8 s** (1,223.7 + 76.0)                   | 215.4 / 10.1             | Only `safetensors_dynamic` loads   |
| `qwen3-235b-a22b`    |     470.4 | Unscanned, `1x8x8` (16h / 64)             | 3,494.8 + 368.9                 | **1,170.2 s** (1,126.4 + 41.5)                   | 216.0 / 7.4              | **1.85x** E2E / **3.30x** excl. DL |
| `deepseek4-284b`\*   |     581.9 | Scanned, `1x8x8` (16h / 64)               | OOM-killed (single-host RAM)    | **1,227.8 s** (1,151.9 + 73.8)                   | 172.2 / 14.8             | Only `safetensors_dynamic` loads   |
| `deepseek4-284b`\*   |     581.9 | Unscanned, `1x8x8` (16h / 64)             | OOM-killed (single-host RAM)    | **1,148.5 s** (1,107.2 + 39.0)                   | 171.0 / 12.2             | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Scanned, `1x8x8` (16h / 64)               | Exceeds single-host RAM         | **903.0 s**                                      | 131.1 / 19.2             | Only `safetensors_dynamic` loads   |
| `qwen3.5-397b-a17b`  |     795.5 | Unscanned, `1x8x8` (16h / 64)             | Exceeds single-host RAM         | **592.0 s**                                      | 131.4 / 16.5             | Only `safetensors_dynamic` loads   |
| `qwen3-480b-a35b`    |     961.0 | Scanned, `1x8x8` (16h / 64)               | Exceeds single-host RAM         | **1,173.6 s**                                    | 158.8 / 20.4             | Only `safetensors_dynamic` loads   |
| `qwen3-480b-a35b`    |     961.0 | Unscanned, `1x8x8` (16h / 64)             | Exceeds single-host RAM         | **1,099.7 s**                                    | 158.1 / 16.9             | Only `safetensors_dynamic` loads   |
| `deepseek3-671b`\*   |   1,369.0 | Scanned, `1x16x16` (64h / 256)            | OOM-killed (single-host RAM)    | **779.4 s** (293.4 + 483.3)                      | 41.5 / 7.7               | Only `safetensors_dynamic` loads   |
| `deepseek3-671b`\*   |   1,369.0 | Unscanned, `1x16x16` (64h / 256)          | OOM-killed (single-host RAM)    | **363.0 s** (275.3 + 85.1)                       | 41.7 / 7.5               | Only `safetensors_dynamic` loads   |
| `deepseek3.2-671b`\* |   1,370.7 | Scanned, `1x16x16` (64h / 256)            | OOM-killed (single-host RAM)    | **781.1 s** (277.5 + 500.9)                      | 41.8 / 7.7               | Only `safetensors_dynamic` loads   |
| `deepseek3.2-671b`\* |   1,370.7 | Unscanned, `1x16x16` (64h / 256)          | OOM-killed (single-host RAM)    | **376.2 s** (282.9 + 90.6)                       | 41.8 / 7.5               | Only `safetensors_dynamic` loads   |
| `deepseek4-1600b`\*  |   3,200.0 | Scanned, `1x16x16` (64h / 256)            | OOM-killed (single-host RAM)    | **1,381.3 s** (482.6 + 894.3)                    | 61.5 / 30.2              | Only `safetensors_dynamic` loads   |
| `deepseek4-1600b`\*  |   3,200.0 | Unscanned, `1x16x16` (64h / 256)          | OOM-killed (single-host RAM)    | **649.1 s** (518.6 + 125.8)                      | 61.5 / 21.1              | Only `safetensors_dynamic` loads   |

### 3. TPU `v5p` benchmarks

Host hardware (`ct5p-hightpu-4t`): `4` chips (`4` TPU devices), `95 GiB` HBM per chip, and `473.3 GB` host RAM per host. Across `54` paired `<=120B` configurations (with `gcsfs`), `safetensors_dynamic` was faster E2E in **47 / 54** runs (median **1.5x** E2E, **2.1x** excl. download) and completed **82 / 82** `0.6B–1.6T` configurations with **0 OOMs**:

#### A. Representative models `<=120B` (TPU `v5p`, 2 hosts / 8 chips)

| Model           | Size (GB) | Config (Scan, Topology (Devices)) | HF -> GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM (GB) | Speedup (E2E / Excl. DL)         |
| :-------------- | --------: | :-------------------------------- | ------------: | :------------------------------ | :----------------------------------------------- | -----------------: | :------------------------------- |
| `qwen3-0.6b`    |       1.5 | Scanned, `2x2x2x1` (8)            |          26.3 | 18.0 + 1.6                      | **14.5 s** (4.8 + 9.4)                           |               52.8 | **1.13x** / **1.36x**            |
| `qwen3-0.6b`    |       1.5 | Unscanned, `2x2x2x1` (8)          |          26.3 | 120.9 + 2.5                     | **7.4 s** (4.4 + 2.7)                            |               52.7 | **4.43x** / **16.59x**           |
| `llama3.1-8b`   |      16.1 | Scanned, `2x2x2x1` (8)            |          85.2 | 83.6 + 10.7                     | **50.7 s** (33.2 + 17.1)                         |               57.4 | **1.32x** / **1.86x**            |
| `llama3.1-8b`   |      16.1 | Unscanned, `2x2x2x1` (8)          |          85.2 | 144.0 + 8.1                     | **40.4 s** (33.0 + 6.9)                          |               57.3 | **1.89x** / **3.77x**            |
| `qwen3-30b-a3b` |      61.1 | Scanned, `2x2x2x1` (8)            |         109.3 | 205.2 + 262.8                   | **140.2 s** (102.2 + 36.7)                       |               58.6 | **2.31x** / **3.34x**            |
| `qwen3-30b-a3b` |      61.1 | Unscanned, `2x2x2x1` (8)          |         109.3 | 432.1 + 20.2                    | **126.7 s** (102.4 + 23.5)                       |               58.9 | **2.38x** / **3.57x**            |
| `mixtral-8x7b`  |      93.4 | Scanned, `2x2x2x1` (8)            |         323.8 | 212.5 + 294.9                   | **205.3 s** (185.5 + 19.4)                       |               58.7 | **1.57x** / **2.47x**            |
| `mixtral-8x7b`  |      93.4 | Unscanned, `2x2x2x1` (8)          |         323.8 | 403.8 + 29.7                    | **197.7 s** (186.3 + 10.9)                       |               58.8 | **1.45x** / **2.19x**            |
| `llama3.1-70b`  |     141.1 | Scanned, `2x2x2x1` (8)            |         237.5 | 1,018.8 + 173.0                 | **310.0 s** (276.1 + 33.7)                       |               68.8 | **2.61x** / **3.84x**            |
| `llama3.1-70b`  |     141.1 | Unscanned, `2x2x2x1` (8)          |         237.5 | 911.6 + 34.2                    | **291.0 s** (278.3 + 12.3)                       |               68.8 | **2.24x** / **3.25x**            |
| `gpt-oss-120b`  |     233.7 | Scanned, `2x2x2x1` (8)            |         496.0 | Exceeds single-host RAM         | **823.7 s** (451.8 + 370.0)                      |               65.3 | Only `safetensors_dynamic` loads |
| `gpt-oss-120b`  |     233.7 | Unscanned, `2x2x2x1` (8)          |         496.0 | 1,768.2 + 71.2                  | **836.4 s** (453.8 + 382.5)                      |               64.9 | **1.75x** / **2.20x**            |

#### B. Large models `>200B` (TPU `v5p`, 2–16 hosts / 8–64 chips)

None of the 7 models `>200B` (`470.4 GB` to `3.20 TB`) fit in a single `473.3 GB` `v5p` host for offline `to_maxtext` conversion, whereas `safetensors_dynamic` loaded **7 of 7** models at `35.6–183.4 GB` peak host RAM:

| Model                | Size (GB) | Config (Scan, Topology (Hosts / Devices)) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM / HBM (GB) | Outcome / Speedup                |
| :------------------- | --------: | :---------------------------------------- | :------------------------------ | :----------------------------------------------- | :----------------------- | :------------------------------- |
| `qwen3-235b-a22b`    |     470.4 | Scanned, `2x2x2x1` (2h / 8)               | Exceeds single-host RAM         | **1,428.8 s** (1,343.9 + 83.2)                   | 183.3 / 80.2             | Only `safetensors_dynamic` loads |
| `qwen3-235b-a22b`    |     470.4 | Unscanned, `2x2x2x1` (2h / 8)             | Exceeds single-host RAM         | **1,286.9 s** (1,235.4 + 47.4)                   | 183.4 / 59.2             | Only `safetensors_dynamic` loads |
| `deepseek4-284b`\*   |     581.9 | Scanned, `1x2x2x4` (4h / 16)              | Exceeds single-host RAM         | **1,009.8 s** (929.7 + 77.9)                     | 96.0 / 45.6              | Only `safetensors_dynamic` loads |
| `deepseek4-284b`\*   |     581.9 | Unscanned, `1x2x2x4` (4h / 16)            | Exceeds single-host RAM         | **1,053.1 s** (1,010.5 + 40.3)                   | 96.0 / 39.7              | Only `safetensors_dynamic` loads |
| `qwen3.5-397b-a17b`  |     795.5 | Scanned, `1x2x4x4` (8h / 32)              | Exceeds single-host RAM         | **1,296.2 s** (397.0 + 898.6)                    | 36.6 / 34.7              | Only `safetensors_dynamic` loads |
| `qwen3.5-397b-a17b`  |     795.5 | Unscanned, `1x2x4x4` (8h / 32)            | Exceeds single-host RAM         | **1,158.4 s** (396.0 + 761.8)                    | 35.6 / 28.8              | Only `safetensors_dynamic` loads |
| `qwen3-480b-a35b`    |     961.0 | Scanned, `1x2x4x4` (8h / 32)              | Exceeds single-host RAM         | **880.6 s** (680.9 + 197.7)                      | 74.0 / 40.9              | Only `safetensors_dynamic` loads |
| `qwen3-480b-a35b`    |     961.0 | Unscanned, `1x2x4x4` (8h / 32)            | Exceeds single-host RAM         | **850.8 s** (685.9 + 162.8)                      | 74.2 / 32.0              | Only `safetensors_dynamic` loads |
| `deepseek3-671b`\*   |   1,369.0 | Scanned, `1x2x4x4` (8h / 32)              | Exceeds single-host RAM         | **1,586.8 s** (1,365.7 + 218.4)                  | 129.8 / 57.4             | Only `safetensors_dynamic` loads |
| `deepseek3-671b`\*   |   1,369.0 | Unscanned, `1x2x4x4` (8h / 32)            | Exceeds single-host RAM         | **1,426.5 s** (1,256.5 + 167.6)                  | 129.8 / 44.8             | Only `safetensors_dynamic` loads |
| `deepseek3.2-671b`\* |   1,370.7 | Scanned, `2x2x2x4` (8h / 32)              | Exceeds single-host RAM         | **1,377.6 s** (1,216.5 + 158.4)                  | 164.8 / 57.5             | Only `safetensors_dynamic` loads |
| `deepseek3.2-671b`\* |   1,370.7 | Unscanned, `2x2x2x4` (8h / 32)            | Exceeds single-host RAM         | **1,383.0 s** (1,271.0 + 109.3)                  | 164.9 / 44.8             | Only `safetensors_dynamic` loads |
| `deepseek4-1600b`\*  |   3,200.0 | Scanned, `1x4x4x4` (16h / 64)             | Exceeds single-host RAM         | **2,675.0 s** (2,484.5 + 186.1)                  | 174.9 / 66.3             | Only `safetensors_dynamic` loads |
| `deepseek4-1600b`\*  |   3,200.0 | Unscanned, `1x4x4x4` (16h / 64)           | Exceeds single-host RAM         | **2,540.0 s** (2,430.0 + 105.3)                  | 175.4 / 58.8             | Only `safetensors_dynamic` loads |

#### C. Layer-streaming throughput on TPU `v5p` (`SafetensorsReader`)

With the layer-streaming loader (including stacked small-tensor reads in open [PR #5589](https://github.com/AI-Hypercomputer/maxtext/pull/5589); loaded language-model weights excluding unused `mtp.*` / `model.visual.*` shards):

- `DeepSeek-V4 284B`\* (`568.7 GB` BF16) on `v5p-32` (`4h / 16 chips`): **163.5 s** (`3.5 GB/s`), peak HBM **40.8 GB/chip**; forward-pass logit check `max KL 0.340` (top-1 match, with open [PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591)).
- `Qwen3.5-397B-A17B` (`792.7 GB` BF16) on `v5p-64` (`8h / 32 chips`): **115.0 s** (`6.9 GB/s`), peak HBM **28.6 GB/chip** (`149.8 s` / `5.3 GB/s` / `54.2 GB` HBM/chip on `v5p-32`, load-only); `max KL 0.00157`, top-10 overlap `9/10`.

### 4. TPU `v5e` benchmarks (`v5e-256` 64 hosts and `2x v5e-256` 128 hosts)

TPU `v5e-256` hosts provide `16 GiB` HBM/chip (`202.6 GB` host RAM). Leading-axis contiguous sharded reads cut `mixtral-8x22b` read time on `v5e-256` from `726 s` (`60.3x` over-read) to **14 s**, and routing `gs://` reads through `gcsfs` (instead of `tf.io.gfile`'s `64 MiB` read-ahead) on the fine-grained MoE runs below kept per-host RAM within `53.9–102.3 GB` up to `3.20 TB` (`79.0–171.9 GB` on default `tf.io.gfile` rows):

| Model                | Size (GB) | Config (Scan, Topology (Hosts / Chips)) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load (Read + Transform, s) | Peak Host RAM / HBM (GB) | Notes                                                     |
| :------------------- | --------: | :-------------------------------------- | :------------------------------ | :----------------------------------------------- | :----------------------- | :-------------------------------------------------------- |
| `llama3.1-8b`        |      16.1 | Scanned, `1x16x16` (64h / 256)          | 61.5 + 10.3                     | **56.0 s** (26.4 + 29.2)                         | 79.0 / 0.4               | 3-rep median                                              |
| `qwen3-30b-a3b`      |      61.1 | Scanned, `1x16x16` (64h / 256)          | 157.6 + 727.6                   | **118.7 s** (67.0 + 50.8)                        | 61.5 / 0.4               | With `gcsfs` (host OOM >170 GB under `tf.io.gfile`)       |
| `qwen3-30b-a3b`      |      61.1 | Unscanned, `1x16x16` (64h / 256)        | 848.8 + 92.8                    | **90.2 s** (56.4 + 32.4)                         | 61.6 / 0.3               | With `gcsfs` (10.4x faster excl. download)                |
| `llama3.1-70b`       |     141.1 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **117.1 s** (78.1 + 38.5)                        | 110.8 / 0.7              | Single-host `to_maxtext` exceeds 203 GB RAM               |
| `gpt-oss-120b`       |     233.7 | Scanned, `1x16x16` (64h / 256)          | Exceeds single-host RAM         | **308.9 s**                                      | 77.2 / 7.1               | Single-host `to_maxtext` exceeds 203 GB RAM               |
| `qwen3-235b-a22b`    |     470.4 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **229.9 s** (155.5 + 72.3)                       | 53.9 / 2.3               | With `gcsfs`                                              |
| `deepseek4-284b`\*   |     581.9 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **313.8 s** (224.5 + 87.0)                       | 68.5 / 5.3               | Dequantized BF16 shards, with `gcsfs`                     |
| `qwen3.5-397b-a17b`  |     795.5 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **1,029.5 s** (337.7 + 691.2)                    | 171.9 / 6.8              | Default `tf.io.gfile`; top next-token probe verified      |
| `qwen3-480b-a35b`    |     961.0 | Scanned, `1x16x16` (64h / 256)          | Exceeds single-host RAM         | **433.4 s** (203.6 + 227.9)                      | 74.4 / 6.5               | With `gcsfs` (74.4 GB/host vs. >203 GB OOM)               |
| `deepseek3-671b`\*   |   1,369.0 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **452.4 s** (338.2 + 111.5)                      | 80.8 / 7.5               | Dequantized BF16 shards, with `gcsfs`                     |
| `deepseek3.2-671b`\* |   1,370.7 | Unscanned, `1x16x16` (64h / 256)        | Exceeds single-host RAM         | **474.8 s** (355.1 + 117.0)                      | 80.8 / 7.5               | Dequantized BF16 shards, with `gcsfs`                     |
| `deepseek4-1600b`\*  |   3,200.0 | Unscanned, `2x16x16` (128h / 512)       | Exceeds single-host RAM         | **1,245.2 s** (848.2 + 392.3)                    | 102.3 / 14.8             | Dequantized BF16 shards (scanned exceeds 16 GiB/chip HBM) |

### 5. TPU `v4`, CPU, and GPU summary

- **TPU `v4-8` (`1` host, `4` chips):** `SafetensorsReader` ranged reads achieve `~2.3 GB/s` end-to-end (`~2.8 GB/s` plain parallel ranged reads) vs. `~1.1 GB/s` for Orbax SafeTensors on `DeepSeek-V2-Lite`, streaming `Qwen3-8B` in **10.8–11.9 s** and `DeepSeek-V2-Lite` in **28.5–33.2 s** with matching step-0 training loss.
- **CPU:** Validated on `qwen3-0.6b`, `qwen2.5-1.5b`, `gemma2-2b`, `gemma3-4b`, `gemma4-e2b`, `qwen3.5-9b`, and `qwen3-vl-2b`, confirming bit-exact (`--atol=0 --rtol=0`) logits vs. `to_maxtext` and `max KL <= 2.9e-4` vs. HF (`<= 8.6e-5` on 6 of 7; `1.4e-7` on `qwen3-0.6b`).
- **GPUs (`H100` / `B200` / CUDA):** Not benchmarked in the current TPU validation suite. Multi-process `safetensors_dynamic` relies on standard JAX `Mesh` / `NamedSharding` and `sync_global_devices`, so GPU functionality and throughput remain to be validated separately.

## Troubleshooting

- **Quantized DeepSeek checkpoints (`FP8` `*_scale_inv` or `MXFP4` `F8_E8M0` `*.scale`) fail header parsing (`Unsupported SafeTensors dtype F8_E8M0`), fail on `None` mapping keys, or produce degraded outputs:** Official DeepSeek-V3, V3.2, and V4 HF repos store block-quantized weights. Convert shards offline to `BF16` with `standalone_scripts/deepseek_dequantize.py` (`deepseek3-671b` and `deepseek3.2-671b` load directly on `main`; `deepseek4-284b` and `deepseek4-1600b` also require open [PR #5591](https://github.com/AI-Hypercomputer/maxtext/pull/5591) or [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) for `None` mapping keys), or track [PR #6398](https://github.com/AI-Hypercomputer/maxtext/pull/6398) for direct on-the-fly dequantization (see [Quantized checkpoints](#4-quantized-checkpoints-deepseek-fp8-and-mxfp4)).
- **`ValueError: base_output_directory with gs:// prefix is required for huggingface downloads.`:** Set `base_output_directory="gs://..."` (without a trailing slash; a trailing `/` causes `No .safetensors files in gs://.../hf_cache/`) for HF Hub downloads, or verify that a local `load_parameters_path` directory exists on every host (missing local paths are treated as HF repo IDs).
- **Config validation fails on `load_parameters_path=""`:** Omit `load_parameters_path` entirely on the CLI to use `HF_IDS[model_name]` (passing an empty `load_parameters_path=` parses as `None`).
- **`-Instruct` models load base weights instead of instruction-tuned weights:** Pass the exact `model_name` (such as `llama3.1-8b-Instruct`) rather than stripping the `-Instruct` suffix so `HF_IDS` resolves the instruction-tuned repository.
- **High host RAM or slow GCS reads on fine-grained MoE models when TensorFlow is installed:** `etils.epath` defaults to `tf.io.gfile` (`64 MiB` read-ahead buffers) when TensorFlow is installed; set `export EPATH_USE_TF=false` so `epath` routes `gs://` reads through `gcsfs`.
- **Out of HBM during load:** Pass `weight_dtype=bfloat16` (default is `"float32"`) or increase the number of TPU devices.
- **`Checkpoint does not match the model`, missing `lora_a` / `lora_b` buffers in `from_pretrained`, or shape mismatches under vLLM:** Check `use_multimodal`, keep `moe_flat_fsdp_weights=False` and `prefuse_moe_weights=False`, track open [PR #5594](https://github.com/AI-Hypercomputer/maxtext/pull/5594) if passing native `enable_lora=True` directly to `from_pretrained`, and under vLLM use `vllm_decode.py --use_tunix=True` (or set `"source_checkpoint_layout": "safetensors_dynamic"` in `additional_config["maxtext_config"]` without KV-head replication or padded MoE MLP dimensions).
- **`enable_single_controller=True` error:** Pathways single-controller mode is rejected at config validation (`types.py`); run `safetensors_dynamic` under multi-controller JAX or pre-convert checkpoints with `to_maxtext` (see [Checkpoint Conversion Utilities](convert_checkpoint.md)).
