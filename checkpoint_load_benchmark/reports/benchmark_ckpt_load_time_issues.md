# MaxText Checkpoint Load-Time Benchmark: Issues, OOM Test Logs, Failure Root-Cause Analysis, and Upstream Fixes

This document records:
1. **Part 1 — Runtime Failures & Cluster OOM Tests Encountered During the 48-Configuration Benchmark (`0.6B` to `1600B`)**: Every crash, OOM (`Exit Code 137`), or configuration failure encountered while running the 48-configuration matrix (`12 models × 2 scan modes × 2 slice counts`), which specific model and configuration triggered it, its exact error message / cluster OOM log, and the fix applied in [`benchmark_ckpt_load_time.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time.py) and on branch `fix/safetensors-dynamic-multislice-hooks`.
2. **Part 2 — Upstream MaxText Code & Performance Issues Uncovered**: Existing bugs and scaling bottlenecks in MaxText's `safetensors_dynamic` and `to_maxtext` pipelines—along with concrete code-level fixes.
3. **Part 3 — Exact Commands Used to Run the Experiments**: Both the top-level benchmark harness commands and the underlying MaxText CLI commands for each step (`single-slice` and `multi-slice`).

---

## Part 1: Runtime Failures & Cluster OOM Tests (`0.6B` to `1600B`)

### 1.1 Summary Matrix of Experiment Failures, Cluster OOM Logs, and Fixes

| Issue ID | Affected Model(s) | Affected Configuration (`scan_mode`, `num_slices`) | Pipeline & Step Where It Failed | Exact Error / Failure Symptom | Root Cause Summary | Fix Applied / Resolution |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-1** | **`gemma2-2b`**<br>*(Also affects `qwen3-*` when HF vocab $\neq$ MaxText vocab)* | • `scanned`, `num_slices=2`<br>• `unscanned`, `num_slices=2` | **`safetensor` Step 2**<br>(`[3/3]` Transform HF State to MaxText State) | `RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices is not possible.` | `pad_hf_embedding_layer` / `pad_embedding_layer` in [`param_mapping.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/param_mapping.py#L489-L510) allocates a host NumPy array `np.zeros(target_shape)` and copies a multi-host `jax.Array` into it. | Replaced NumPy array slicing with pure JAX `jnp.pad` on branch `fix/safetensors-dynamic-multislice-hooks`. |
| **F-2 & F-3** | **`qwen3-32b`** *(65.5 GB)*,<br>**`llama3.1-70b`** *(141.1 GB)* | • `scanned`, `num_slices=1 & 2` | **`to_maxtext` Step 3**<br>(Load Orbax Checkpoint) | **Pod `OOMKilled` (Exit Code 137)** (`concurrent_gb=96`) vs `ValueError: Requested more bytes than we reserved space for: 37580963840 > 20000000000` (`llama3.1-70b`) | Scanned MLP leaves are `16.78 GB` (`qwen3-32b` `[5120, 64, 25600]`) and **`37.58 GB`** (`llama3.1-70b` `[8192, 80, 28672]`). Orbax `LimitInFlightBytes` requires `concurrent_gb` $\ge$ largest single leaf per process (`38 GB` on `1-slice`, `20 GB` on `2-slice`). | Set `checkpoint_storage_concurrent_gb=38` for `llama3.1-70b` `1-slice` (`20` on `2-slice`, `16` on `unscanned`). |
| **F-4** | **`qwen3-32b`**, **`qwen3-30b-a3b`**, **`mixtral-8x7b`**, **`llama3.1-70b`**, **`gpt-oss-120b`** | • `scanned`, `num_slices=1 & 2` | **`safetensor` Step 2**<br>(`[3/3]` `_binary_chunked_stack`) | Excessive host RAM peak (`>230 GiB`) during recursive layer/expert stacking | [`_binary_chunked_stack`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/tensor_handling.py#L91-L111) retained references to intermediate chunk lists while recursively stacking 32–80 layers or 128 experts. | Pop and free intermediate tensors immediately as each chunk is stacked on branch `fix/safetensors-dynamic-multislice-hooks`. |
| **F-5** | **`llama3.1-70b`** *(141.1 GB)* & **`gpt-oss-120b`** *(233.7 GB)* | • `llama3.1-70b`: `scanned, num_slices=1`<br>• `gpt-oss-120b`: `num_slices=2` | **`safetensor` Step 2**<br>(`[2/3]` GCS Load) | **`OOMKilled (Exit Code 137)`** at `263.04/270.25 GB` on `llama3.1-70b` with `orbax-checkpoint 0.11.x` | Base image's `orbax-checkpoint 0.11.x` `SafetensorsLayout` reads entire `.safetensors` files into memory without byte-range coalescing or in-flight byte limiting. | Upgraded `orbax-checkpoint` to `0.12.7`, reducing `llama3.1-70b` `scanned, 1-slice` Peak RAM from `263.04 GB` (OOM) to `225.16 GB` (Succeeded in `1,012.45s`). |
| **F-6** | **`gpt-oss-120b`** *(233.66 GB MoE, 73 shards)* | • `to_maxtext` Step 2 (`scanned` & `unscanned`)<br>• `safetensor` (`num_slices=1`, `scanned` & `unscanned`) | **`to_maxtext` Step 2** & **`safetensor` Step 2 (`1-slice`)** | **`OOMKilled (Exit Code 137)`** at `251.69/251.69 GB` (`to_maxtext`) and `253.06/261.99 GB` (`safetensor` `1-slice`) | `233.66 GB` BF16 checkpoint exceeds single-host RAM (`251.69–261.99 GB`) when converting (`to_maxtext`) or loading onto a single host (`num_slices=1`). | **Multi-slice `safetensor` (`num_slices=2`) succeeded** in both `scanned` (`3,061.14s`, `206.09 GB/slice`) and `unscanned` (`3,004.55s`, `133.86 GB/slice`) by sharding across 2 slices (`8` devices). |
| **F-7** | **`deepseek3.2-671b`**, **`deepseek4-284b`**, **`deepseek4-1600b`** | • `hardware=cpu` (`scanned` & `unscanned`) | **`to_maxtext` Step 2** & **`safetensor` Step 2** | `ValueError: Sparse indexer with mla is only supported with dot_product attention or flash attention with tokamax splash.` | `deepseek3.2-671b.yml` and `deepseek4-*.yml` set `use_sparse_indexer: True` with `attention_type: mla`, which fails validation when `hardware=cpu` leaves `attention="autoselected"`. | Pass `attention=dot_product` when running `deepseek3.2` or `deepseek4` with `hardware=cpu`. |
| **F-8** | **`deepseek4-284b`** *(581.9 GB)*,<br>**`deepseek3-671b`** *(1,365.3 GB)*,<br>**`deepseek3.2-671b`** *(1,367.0 GB)*,<br>**`deepseek4-1600b`** *(3,197.7 GB)* | • `scanned` & `unscanned`<br>• `num_slices=1 & 2` (All 16 configs) | **`to_maxtext` Step 2** & **`safetensor` Step 2 (`1 & 2 slices`)** | **`OOMKilled (Exit Code 137)`** at `240.32–246.10 / 270.25 GB` | Model BF16 footprints (`581.89 GB` to `3,197.68 GB`) exceed 1-host (`270.25 GB`) and 2-host (`540.50 GB`) RAM. Single scanned expert leaves alone are `434.87 GB` (`671B`) and `541.17 GB` (`1600B`). | Requires `num_slices >= 4` (`284B`), `>= 8` (`671B`), and `>= 16` (`1600B`) with `safetensors_dynamic`. |
| **F-9** | **`deepseek3-671b`**, **`deepseek3.2-671b`**, **`deepseek4-*`** | • `num_slices=2` (`35,023–73,167` HF tensors) | **`safetensor` Step 2**<br>(`[2/3]` `SafetensorsCheckpointHandler`) | Multi-minute CPU hang generating `~245M` Python byte-run tuples before reading from disk/GCS | Sharding 2D `.weight` tensors (`[2048, 7168]`) along `axis=1` across 8 devices creates `2,048` non-contiguous strided byte runs per device per tensor across `30,000+` expert tensors. | Updated `_make_sharded_spec` to shard along leading `axis=0`, reducing each tensor to `1` contiguous byte run per device (`>10m -> <45s`). |

---

### 1.2 Detailed Breakdown of Cluster OOM Tests (`gpt-oss-120b` and `deepseek3 / 3.2 / 4`)

#### OOM Test 1: `gpt-oss-120b` (`unsloth/gpt-oss-120b-BF16`, `233.66 GB` BF16, 73 Shards) — Single-Host OOM vs. Multi-Slice `safetensors_dynamic` Success

- **1. `to_maxtext` Step 2 (`scanned`, 41 PyTree leaves) — `OOMKilled (Exit Code 137)` at `73% (30/41 leaves)`**:
  ```text
  Transforming weights: 100%|██████████| 41/41 [00:00<00:00, 128019.22param/s, RAM: 6.4/251.7GB (2.5%)]
  INFO:absl:Starting checkpoint save (loading weights just-in-time)...
  INFO:absl:Shard weights across 16 devices
   71%|███████   | 29/41 [10:34<02:25, 12.14s/it]
   73%|███████▎  | 30/41 [10:47<02:14, 12.27s/it]
  [STEP FAILED]: Worker phase 'to_maxtext_step2' failed with exit code 137 (OOMKilled / Exit Code 137) at 73% (30/41 leaves, 647s) (Peak RAM: 251.69/251.69 GB).
  ```
- **2. `to_maxtext` Step 2 (`unscanned`, 687 PyTree leaves) — `OOMKilled (Exit Code 137)` at `100%` during `save_finalize`**:
  ```text
  100%|██████████| 687/687 [04:51<00:00,  2.36it/s]
  INFO:absl:Donelazily loading and sharding weights. Saving to Orbax...
  [STEP FAILED]: Worker phase 'to_maxtext_step2' failed with exit code 137 (OOMKilled / Exit Code 137) (Peak RAM: 251.69/251.69 GB).
  ```
- **3. `safetensors_dynamic` Single-Slice (`num_slices=1`) — `OOMKilled (Exit Code 137)`**:
  - `scanned, num_slices=1`: `Worker phase 'safetensors_dynamic' failed with exit code -9 (OOMKilled / Exit Code 137) (Peak RAM: 253.06/261.99 GB).`
  - `unscanned, num_slices=1`: `Worker phase 'safetensors_dynamic' failed with exit code -9 (OOMKilled / Exit Code 137) (Peak RAM: 253.01/261.99 GB).`
- **4. `safetensors_dynamic` Multi-Slice (`num_slices=2`, 8 Global Devices Across 2 Cluster Nodes) — BOTH SUCCEEDED**:
  - `scanned, num_slices=2`:
    ```text
    [safetensor] Completed: (1) HF->GCS=1061.72s, (2) safetensors & convert to MaxText=1999.42s (total=3061.14s, 41 parameter leaves, num_slices=2, global_devices=8)
    INFO:watchdog:Worker Peak Host RAM: 206.09/261.99 GB
    ```
  - `unscanned, num_slices=2`:
    ```text
    [safetensor] Completed: (1) HF->GCS=0.70s (1061.72s cold), (2) safetensors & convert to MaxText=1942.83s (total=3004.55s, 687 parameter leaves, num_slices=2, global_devices=8)
    INFO:watchdog:Worker Peak Host RAM: 133.86/261.99 GB
    ```

---

#### OOM Test 2: DeepSeek 3 (`deepseek3-671b`), DeepSeek 3.2 (`deepseek3.2-671b`), and DeepSeek 4 (`deepseek4-284b`, `deepseek4-1600b`) on `bodaborg-v5p-nap` (`270.25 GB` RAM / Node)

All 4 ultra-large DeepSeek models were executed on `bodaborg-v5p-nap` across all 6 phases (`to_maxtext` Step 2 `scanned`/`unscanned`, and `safetensors_dynamic` `scanned`/`unscanned` × `num_slices=1`/`2`):

| Model | BF16 Size | HF Tensors | `to_maxtext` Step 2 (`scanned`) OOM Point & Peak RAM | `to_maxtext` Step 2 (`unscanned`) OOM Point & Peak RAM | `safetensors_dynamic` (`1-Slice` `scanned` / `unscanned`) Peak RAM | `safetensors_dynamic` (`2-Slice` `scanned` / `unscanned`) Peak RAM |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **`deepseek4-284b`** | `581.89 GB` | `35,023` | **`48% (39/82 leaves)`**<br>`Peak RAM: 245.42/270.25 GB` | **`21% (408/1,930 leaves)`**<br>`Peak RAM: 245.33/270.25 GB` | `245.75 GB` / `245.95 GB`<br>*(init: `1.29s` / `74.16s`)* | `245.29 GB` / `245.29 GB`<br>*(init: `1.34s` / `74.02s`)* |
| **`deepseek3-671b`** | `1,365.27 GB` | `46,181` | **`50% (16/32 leaves)`**<br>`Peak RAM: 245.40/270.25 GB` | **`16% (163/1,025 leaves)`**<br>`Peak RAM: 245.31/270.25 GB` | `245.10 GB` / `245.88 GB`<br>*(init: `1.88s` / `74.73s`)* | `245.35 GB` / `245.33 GB`<br>*(init: `1.95s` / `74.66s`)* |
| **`deepseek3.2-671b`** | `1,367.00 GB` | `46,491` | **`50% (21/42 leaves)`**<br>`Peak RAM: 245.56/270.25 GB` | **`15% (198/1,330 leaves)`**<br>`Peak RAM: 245.43/270.25 GB` | `246.10 GB` / `245.90 GB`<br>*(init: `1.93s` / `102.89s`)* | `245.19 GB` / `245.46 GB`<br>*(init: `1.98s` / `102.20s`)* |
| **`deepseek4-1600b`** | `3,197.68 GB` | `73,167` | **`49% (112/228 leaves)`**<br>`Peak RAM: 245.65/270.25 GB` | **`6% (168/2,748 leaves)`**<br>`Peak RAM: 245.96/270.25 GB` | `245.11 GB` / `245.86 GB`<br>*(init: `4.83s` / `325.35s`)* | `245.51 GB` / `240.32 GB`<br>*(init: `4.83s` / `318.48s`)* |

- **Representative Cluster OOM Log (`deepseek3.2-671b` `to_maxtext` Step 2 `scanned` & `unscanned`)**:
  ```text
  # scanned (42 leaves):
   48%|████▊     | 20/42 [00:20<00:29,  1.33s/it]
   50%|█████     | 21/42 [00:24<00:46,  2.22s/it]
  [OOM_WATCHDOG] Host RAM exhausted! Peak RAM Usage: 245.44/270.25 GB (90.8%, MemAvailable=24.81 GB < 25.00 GB threshold). Sending SIGKILL (OOMKilled / exit code 137) to worker PID 150.
  INFO:watchdog:Subprocess Peak Host RAM: 245.56/270.25 GB

  # unscanned (1,330 leaves):
   15%|█▍        | 197/1330 [06:09<2:14:52,  7.14s/it]
   15%|█▍        | 198/1330 [06:32<3:42:37, 11.80s/it]
  [OOM_WATCHDOG] Host RAM exhausted! Peak RAM Usage: 245.51/270.25 GB (90.8%, MemAvailable=24.73 GB < 25.00 GB threshold). Sending SIGKILL (OOMKilled / exit code 137) to worker PID 357.
  INFO:watchdog:Subprocess Peak Host RAM: 245.43/270.25 GB
  ```
- **Representative Cluster Config Error Log (`deepseek3.2-671b` without `attention=dot_product` on `hardware=cpu`)**:
  ```text
  File "/mnt/shm/workspace/src/maxtext/configs/pyconfig.py", line 682, in validate_AND_update_keys
    raise ValueError(
  ValueError: Sparse indexer with mla is only supported with dot_product attention or flash attention with tokamax splash.
  ```

---

## Part 2: Current Upstream MaxText Issues & Performance Bottlenecks Uncovered

### 2.1 Issue U-1: Multi-Host `jax.Array` to NumPy Crash in `pad_hf_embedding_layer` / `pad_embedding_layer` (`num_slices > 1`)
- **Location**: [`GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN.pad_hf_embedding_layer`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/param_mapping.py#L489-L510) and [`QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN.pad_embedding_layer`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/param_mapping.py#L764-L777).
- **Root Cause**: Allocating `padded_tensor = np.zeros(target_shape)` and assigning `padded_tensor[:source_vocab_size, :] = input_tensor` fails when `input_tensor` is a multi-host `jax.Array` spanning remote slices (`RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices is not possible`).
- **Fix**: Use `jnp.pad(input_tensor, ((0, target_shape[0] - source_vocab_size), (0, 0)))` when `isinstance(input_tensor, jax.Array)` (committed on branch `fix/safetensors-dynamic-multislice-hooks`).

### 2.2 Issue U-2: `1.6x–1.8x` Safetensors Strided Byte Over-Read & `245M` Byte-Run Explosion in `construct_maximal_shardings`
- **Location**: [`sharding_utils.construct_maximal_shardings`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/sharding_utils.py#L47-L69).
- **Root Cause**: Sharding 2D `.safetensors` tensors `[dim_0, dim_1]` along `axis=1` when `dim_1 > dim_0` creates `dim_0` non-contiguous strided byte spans per device in row-major `.safetensors` files. On `qwen3-30b-a3b` / `qwen3-32b`, this causes a `1.6x–1.8x` GCS byte over-read; on `deepseek3-671b` (`46,181` tensors) and `deepseek4-1600b` (`73,167` tensors), generating `2,048` byte runs × `8` devices × `30,000+` expert tensors creates `>245M` Python tuples before any file I/O begins.
- **Fix**: Always prefer sharding along leading contiguous **`axis=0`** whenever `shape[0] % num_devices == 0`.

### 2.3 Issue U-3: XLA SPMD Involuntary Full Rematerialization When Resharding `1D all_devices` $\to$ `2D (dcn_data, ici_fsdp)`
- **Location**: [`load_sharded_hf_state`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L180-L192) and [`transform_hf_state_to_mt_state`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/tensor_handling.py#L150-L218).
- **Root Cause**: Resharding from 1D 8-way maximal sharding (`{devices=[8,1]<=[8]}`) to a 2D data-parallel + FSDP mesh (`{devices=[1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}`) forces XLA SPMD to replicate each tensor across slices before partitioning (`util.cc:175 [SPMD] Involuntary full rematerialization`). Conversely, when using 2-slice FSDP sharding (`dcn_fsdp_parallelism=2, ici_fsdp_parallelism=4` on `llama3.1-70b` and `gpt-oss-120b`), `[3/3]` transform stays fast (`76.57s–91.72s` on `llama3.1-70b` and `92.26s–163.14s` on `gpt-oss-120b`).

### 2.4 Issue U-4: Native FP8 (`F8_E4M3` + `weight_scale_inv`) Support in `safetensors_dynamic` for `deepseek3-671b` & `deepseek3.2-671b`
- **Location**: [`load_sharded_hf_state`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L180-L192).
- **Root Cause**: Official `deepseek-ai/DeepSeek-V3` and `deepseek-ai/DeepSeek-V3.2` HuggingFace repositories store weights in `F8_E4M3` with companion `*.weight_scale_inv` (`F32`) tensors (`163` shards, `~685 GB` FP8 $\to$ `1,365–1,367 GB` BF16). `safetensors_dynamic` currently assumes `.safetensors` weights can be read directly as `bfloat16` without block-wise scale multiplication.

---

## Part 3: Commands Used to Run the Experiments

```bash
# Full 12-Model / 48-Configuration Benchmark & Cluster OOM Suite:
PYTHONPATH=src python3 benchmark_ckpt_load_time.py \
  --models qwen3-0.6b gemma2-2b deepseek2-16b qwen3-30b-a3b qwen3-32b \
           mixtral-8x7b llama3.1-70b gpt-oss-120b \
           deepseek3-671b deepseek3.2-671b deepseek4-284b deepseek4-1600b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2 \
  --hardware cpu \
  --weight-dtype bfloat16 \
  --reuse-model-hf-cache \
  --allow-step-failure \
  --gcs-base gs://mesa-maxtext/benchmark_ckpt_load_time \
  --run-id 20261023_032224 \
  --csv-report benchmark_ckpt_load_time_summary.csv
```
