Hi Abhinav, while running the large DeepSeek models (`284B`, `671B`, `3.2-671B`, `1600B`), we found that Hugging Face's weight dtype doesn't match what MaxText requires. Here is a quick summary of what we found and our synthetic vs. real-weight test results:

**1. Hugging Face vs. MaxText Weight Dtype**
- **Hugging Face checkpoints:** Published **only in quantized `FP8` / `FP4` (+ scale tensors)** — official `BF16` weights are not available on Hugging Face.
- **MaxText requirement:** Both `to_maxtext` and `safetensors_dynamic` expect **`BF16`** weights and do not dequantize `FP8` / `FP4` during load.
- **Required step:** Quantized Hugging Face weights must first be converted offline from **`FP8` / `FP4` → `BF16`** (`582 GB` to `3.2 TB`) — e.g., via MaxText's standalone `deepseek_dequantize.py` script ([gpaste](https://paste.googleplex.com/6755399437104149) | [GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py)) — and stored in GCS before loading into MaxText.

**2. First Test (Synthetic `BF16`) vs. Second Test (Real `BF16` from GCS)**
- **First test — Synthetic `BF16` weights:**
  - **Weight source:** Dummy zero-filled `BF16` `.safetensors` files generated directly on each TPU host's local disk from the DeepSeek model config and tensor shapes (not downloaded from Hugging Face; `0` bytes read from GCS).
  - **Result:** Validated TPU HBM capacity and XLA sharding/transforms up to `3.2 TB`, **but** understated end-to-end load time and did not measure real GCS network I/O or GCS client host RAM.
- **Second test — Real `BF16` weights from GCS:**
  - **Weight source:**
    - **`284B`, `671B`, `3.2-671B` (`582 GB – 1.37 TB`):** Copied pre-converted `BF16` `.safetensors` checkpoints from the MaxText team's shared GCS bucket (`gs://maxtext-deepseek/`) into each benchmark region's GCS bucket.
    - **`1600B` (`3.2 TB`):** Downloaded the quantized `FP8` / `FP4` checkpoint (`865 GB`) from Hugging Face, dequantized it offline to `BF16` (`3.2 TB`) using `deepseek_dequantize.py` ([gpaste](https://paste.googleplex.com/6755399437104149)), and staged it in regional GCS.
  - **Results (streaming full `BF16` weights from regional GCS via `safetensors_dynamic`):**
    - **TPU `v7x` (`8–64` devices):** Loaded all DeepSeek models (`284B` to `1600B`, scanned and unscanned) cleanly in **`256s – 1,691s`**.
    - **Large multi-host slices (`v5e` / `v6e`, `64–128` hosts):** Streaming ~54k unfused MoE expert tensors from GCS exposed `64 MiB` read-ahead buffering in the default `tf.io.gfile` reader (causing host RAM OOM on `192 GiB` `v5e` hosts and slow reads on `v6e`).
    - **With the `gcsfs` GCS reader fix:** Eliminates read-ahead amplification and loads real `BF16` DeepSeek models up to **`3.2 TB` (`1600B`)** in **`168s – 356s`** on `v5e` using only **`25 – 53 GB`** peak host RAM (`v6e` and `v5p` benchmarks currently running).
