# b/586746548 — [MaxText][safetensors_dynamic] Support on-the-fly FP8 / MXFP4 -> BF16 dequantization when loading quantized Hugging Face DeepSeek checkpoints

* **Buganizer URL:** http://b/586746548
* **Component:** `1174383` (`Cloud Platform > Compute > CMCS > Cloud TPU/GPUs > Multipod`)
* **Type / Priority / Assignee:** `FEATURE_REQUEST` / `P2` / `mesakh@google.com`

---

## Summary

Official Hugging Face checkpoints for large DeepSeek models (`deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3.2`, `deepseek-ai/DeepSeek-V4-Pro`) are published **only in quantized `FP8` (`E4M3` + `_scale_inv`) and/or packed `MXFP4` (`E2M1` + `.scale`) formats**, whereas MaxText's DeepSeek model definitions (`deepseek4-284b`, `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-1600b`) expect **`BF16`** weight leaves.

Today, neither `safetensors_dynamic` nor `to_maxtext` dequantizes `FP8` / `MXFP4` weights during load, requiring a separate offline CPU conversion step (`src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py`, https://paste.googleplex.com/6755399437104149) that writes a second `582 GB – 3.2 TB` `BF16` `.safetensors` checkpoint to GCS before MaxText can load it.

We should support **in-memory, on-the-fly `FP8` / `MXFP4` → `BF16` dequantization** directly inside `safetensors_dynamic` (right after reading the raw quantized HF tensors from GCS/HF, during the transform step).

## Tested Revision & Configs (Large DeepSeek Models)

* **MaxText base revision tested:** Git commit `5b6773c9b` (`maxtext-v0.2.6`, includes cl/1017682327 / PR #5828 `Fix safetensors_dynamic load timeout on many hosts`, **prior** to the `HF streaming` CLs cl/1020481604, cl/1020513918, cl/1023046858).
* **Models & raw vs. `BF16` checkpoint sizes:**
  * `deepseek4-284b` (`deepseek-ai/DeepSeek-V4-Flash`): `159.6 GB` raw HF (`FP8` + `MXFP4` + `.scale`) → `581.9 GB` `BF16` (`46` shards)
  * `deepseek3-671b` (`deepseek-ai/DeepSeek-V3`): `688.6 GB` raw HF (`FP8` + `_scale_inv`) → `1,369.0 GB` `BF16` (`163` shards)
  * `deepseek3.2-671b` (`deepseek-ai/DeepSeek-V3.2`): `689.5 GB` raw HF (`FP8` + `_scale_inv`) → `1,370.7 GB` `BF16` (`163` shards)
  * `deepseek4-1600b` (`deepseek-ai/DeepSeek-V4-Pro`): `864.7 GB` raw HF (`FP8` + `MXFP4` + `.scale`) → `3,197.7 GB` `BF16`
* **Exact MaxText `pyconfig` flags used in experiments:**

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  run_name=benchmark_safetensors_dynamic \
  model_name=deepseek3-671b \                # {deepseek4-284b, deepseek3-671b, deepseek3.2-671b, deepseek4-1600b}
  scan_layers=True \                         # tested both True (scanned) and False (unscanned)
  hardware=tpu \
  weight_dtype=bfloat16 \
  ici_fsdp_parallelism=-1 \
  num_slices=1 \                             # 1 or 2 (with dcn_fsdp_parallelism=2 when num_slices=2)
  enable_checkpointing=False \
  async_checkpointing=False \
  per_device_batch_size=1 \
  dataset_type=synthetic \
  source_checkpoint_layout=safetensors_dynamic \
  load_parameters_path="gs://<bucket>/hf_cache/deepseek3-671b/" \
  base_output_directory="gs://<bucket>/output/"
```

* **TPU topologies tested:**
  * **TPU `v7x`:** `1x 2x2x1` / `2x 2x2x1` (`284B`), `1x 2x2x4` / `2x 2x2x2` (`671B`, `3.2-671B`), `1x 2x4x4` (`1600B`)
  * **TPU `v5p`:** `1x 2x2x4` / `2x 2x2x2` (`284B`), `1x 2x4x4` / `2x 2x2x4` (`671B`, `3.2-671B`), `1x 4x4x4` (`1600B`)
  * **TPU `v6e`:** `1x 8x8` (`284B`), `1x 16x16` (`671B`, `3.2-671B`, `1600B`)
  * **TPU `v5e`:** `1x 16x16` (`284B`, `671B`, `3.2-671B`), `2x 16x16` (`1600B`)

## Current vs. Expected Behavior

* **Current behavior:**
  * **Case 1 — Loading raw quantized HF shards (`load_parameters_path` pointing at raw HF `FP8` / `MXFP4` shards or `hf://deepseek-ai/...`):**
    * `DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING` in `src/maxtext/checkpoint_conversion/utils/param_mapping.py` maps only `*.weight` keys and ignores companion `*_scale_inv` (`128x128` block `FP8` scales) and `*.scale` (`MXFP4` block scales).
    * Packed 4-bit `MXFP4` `int8`/`uint8` tensors (two `E2M1` nibbles per byte → half column width) fail shape checks, and `float8_e4m3fn` tensors are not multiplied by `_scale_inv`.
    * Note: `src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_ckpt.py` already performs in-memory `int4` dequantization for Kimi-K2 (`dequantize_pack_quantized_int4`), but raises `ValueError("fp8 checkpoint is not supported.")` on `_scale_inv`.
  * **Case 2 — Loading offline-dequantized `BF16` shards (`load_parameters_path` pointing at `hf_cache_bf16/`):**
    * Works on `v7x` (`14/14` runs `OK`, `256s – 1,691s`), and works on `64–128` host `v5e` (`168s – 356s` up to `3.2 TB`) when paired with PR #6319 (`gcsfs` path to avoid `tf.io.gfile` `64 MiB` read-ahead amplification across ~54k unfused MoE expert keys), **but** requires pre-converting and storing `582 GB – 3.2 TB` of `BF16` `.safetensors` in GCS first.
* **Expected behavior:**
  * `safetensors_dynamic` detects quantized DeepSeek HF checkpoints (`FP8` / `MXFP4`), streams the sharded quantized weights + companion scale tensors directly from GCS / Hugging Face, and dequantizes them to `BF16` on-the-fly in memory (on TPU/JAX) during the load/transform step.

## Impact & Benefits

* **Eliminates offline conversion & duplicate GCS storage:** Removes the multi-hour offline CPU dequantization step and saves `582 GB – 3.2 TB` of duplicate `BF16` `.safetensors` storage per region.
* **`2.0x – 3.7x` less GCS network traffic during `safetensors_dynamic` load:**
  * `deepseek3-671b` / `deepseek3.2-671b`: reads **`689 GB`** (`FP8`) instead of **`1,370 GB`** (`BF16`) — **`2.0x` less GCS I/O**.
  * `deepseek4-284b`: reads **`160 GB`** (`FP8`+`FP4`) instead of **`582 GB`** (`BF16`) — **`3.6x` less GCS I/O**.
  * `deepseek4-1600b`: reads **`865 GB`** (`FP8`+`FP4`) instead of **`3,200 GB`** (`BF16`) — **`3.7x` less GCS I/O**, while also reducing per-host GCS read buffer RAM.

## Proposed Implementation Approach

1. **Composite `(weight, scale)` mapping:** Update DeepSeek `PARAM_MAPPING` / loader metadata to pair quantized `*.weight` keys with their `*_scale_inv` (V3 / V3.2) or `*.scale` (V4) scale keys when present in the safetensors index.
2. **Sharded weight + replicated scale reads:** In `_leading_axis_shardings` (`src/maxtext/checkpoint_conversion/utils/load_dynamic.py`), keep axis-0 sharding across hosts for `weight`, and replicate the tiny `scale` tensor (`PartitionSpec()`, `<1%` of weight bytes, avoiding axis-0 divisibility issues when `M // 128 < num_hosts` on `64+` host slices).
3. **In-flight JAX `jnp` dequantization hook:** Port `unpack_fp4` (`E2M1` LUT) + blockwise `(q * scale).astype(jnp.bfloat16)` from `src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py` into a JAX transform hook executed before layer/expert stacking, freeing raw quantized buffers immediately after dequantization.
