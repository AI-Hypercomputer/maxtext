# Turnkey Team Reproducibility Guide (`docs/checkpoint-load-benchmark`)

This guide provides self-contained, copy-paste commands to reproduce the **MaxText Checkpoint Load-Time Benchmark v2** harness and the three upstream checkpoint-loading PRs integrated on `docs/checkpoint-load-benchmark`—ranging from a **3-minute local CPU run** (no TPU or GCS required) to **live multi-slice TPU parity and large-model benchmarks on GKE**.

---

## What Is Included on `docs/checkpoint-load-benchmark`

| Component | Summary | Key Files |
| :--- | :--- | :--- |
| **Benchmark `v2` 4-Platform Suite** | End-to-end and warm checkpoint load harness (`to_maxtext` vs `safetensors_dynamic`) across **TPU v7x, v6e, v5p, and v5e** (`516` planned `v2` rows across `21` models from `0.6B` to `1600B`, plus real-weight `BF16` DeepSeek runs; see [`reports/benchmark_v2_summary.md`](reports/benchmark_v2_summary.md)). | `benchmark_ckpt_load_time.py`, `bench/`, `scripts/` |
| **PR #6398** (`feat/safetensors-dynamic-deepseek-dequant`) | On-the-fly block-scaled `FP8` (`*_scale_inv` / `*.scale`) and packed `MXFP4` (`*.scale`, including `F8_E8M0` scales via `SafetensorsReader` fallback) $\to$ `bfloat16` dequantization in both default (`load_sharded_hf_state` + `transform_hf_state_to_mt_state`) and streaming (`hf_streaming_load`) `safetensors_dynamic` modes. | `src/maxtext/checkpoint_conversion/utils/{dequantize,hf_streaming_load,load_dynamic,safetensors_reader}.py` |
| **PR #6353** (`fix/safetensors-dynamic-composite-hbm-peak`) | Stacks composite leaves one by one (`num_parts`), joins reshard slices in pairs (`_concatenate_pieces`), and frees consumed `jax.Array` buffers immediately (`_delete_inputs` with zero-copy `pinned` tracking) to eliminate HBM peaks on large composite/MoE models (`qwen3.5-397b-a17b`, `gpt-oss-120b`, `gemma4`, `glm5.2`). | `src/maxtext/checkpoint_conversion/utils/{tensor_handling,load_dynamic}.py` |
| **PR #6354** (`fix/to-maxtext-orbax-save-concurrent-gb`) | Forwards `config.checkpoint_storage_concurrent_gb` to `create_orbax_checkpoint_manager` in `save_weights_to_checkpoint`, capping Orbax write buffers to prevent host RAM OOMs during offline `to_maxtext` conversions. | `src/maxtext/checkpoint_conversion/utils/utils.py`, `tests/unit/to_maxtext_test.py` |
| **Turnkey Repro Scripts** | Parameterized CPU, GCS shard-streaming, and single/multi-slice GKE TPU verification tools with zero hardcoded user paths. | `checkpoint_load_benchmark/scripts/repro/` |

---

## Prerequisites

Clone the repository, check out `docs/checkpoint-load-benchmark`, and activate a standard MaxText Python 3.12 virtual environment:

```bash
git fetch origin docs/checkpoint-load-benchmark
git checkout docs/checkpoint-load-benchmark
# Activate your MaxText Python 3.12 virtualenv before running the commands below.
```

---

## Tier 1 — Local CPU Reproduction (~3–4 min, No TPU or GCS Needed)

Runs all unit tests and reduced-dimension end-to-end model parity checks on 4 simulated host CPU JAX devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`):

```bash
bash checkpoint_load_benchmark/scripts/repro/run_cpu_repro.sh
```

### What It Runs & Expected Output

1. **`[Step 1/4]` PR #6398, PR #6353 & PR #6354 Unit Tests (`295 passed`):**
   - `tests/unit/deepseek_dequantize_dynamic_test.py` (`FP8` / `MXFP4` / `F8_E8M0` math, `SafetensorsReader` fallback, `transform_hf_state_to_mt_state` & `hf_streaming_load` parity)
   - `tests/unit/hf_streaming_load_test.py` & `tests/unit/safetensors_reader_test.py`
   - `tests/unit/to_maxtext_test.py` (`checkpoint_storage_concurrent_gb` forwarding & default `96` GB behavior)
   - `tests/unit/checkpointing_test.py -k "not pathways"` (per-leaf composite stacking, pairwise resharding, buffer deletion & HF Hub `config.json` caching)
2. **`[Step 2/4]` Benchmark `v2` Harness Unit Tests (`checkpoint_load_benchmark/tests/`, `16` test modules):**
   - Validates `v7x` / `v6e` / `v5p` / `v5e` platform profiles, K8s JobSet & Step 1/1b manifest generation, tarball provenance, and Sheet `v2` schema validators.
3. **`[Step 3/4]` End-to-End CPU Model Parity & Buffer-Hygiene (`verify_e2e_cpu_models.py`):**
   - **Reduced-dim `deepseek4-284b`** (`FP8` + `MXFP4` + `F8_E8M0`): `from_pretrained` weights and deterministic forward-pass logits match offline `BF16` bit-for-bit across `scan_layers={True, False}` $\times$ `streaming={False, True}`, while `transform_hf_state_to_mt_state` frees `100%` of raw quantized/scale `jax.Array` buffers.
   - **Reduced-dim `gemma4-26b`** (`scan_layers={True, False}`): shared `experts.gate_up_proj` composite splitting leaves `0` deleted restored buffers and matches `hf_streaming_load` bit-for-bit.
   - **Reduced-dim `glm5.2-744b`** (`scan_layers=True` and `scan_layers=False` with `prune_shared={True, False}`): `IndexShare` zero-copy buffer pinning leaves `0` deleted restored buffers and matches `hf_streaming_load` bit-for-bit.
4. **`[Step 4/4]` Cross-Model `PARAM_MAPPING` Safety Audit (`audit_all_param_mappings.py`):**
   - Audits every `(model, scan_layers)` configuration in `PARAM_MAPPING` (`120+` configurations), confirming `0` false companion-scale collisions on `BF16` metadata and `0` canonical-key mutations in `rename_legacy_hf_keys`.

Expected final banner:

```text
========================================================================
 [PASS] All 4 local CPU reproduction suites completed with 0 failures!
========================================================================
```

---

## Tier 2 — Real GCS Shard Streaming Check (~1 min on CPU with GCS Access)

Streams real production `DeepSeek-V3` (`671B`, shard 1: `FP8` + `F32` `*_scale_inv`, including the non-divisible `576x7168` MLA `kv_a_proj_with_mqa.weight`) and `DeepSeek-V4` (`284B`, shard 5: `FP8` + `MXFP4` `I8` + `F8_E8M0` `*.scale`) directly from GCS on CPU and compares every leaf bit-for-bit against the pre-converted offline `BF16` GCS shards:

```bash
PYTHONPATH=$PWD/src:$PWD/checkpoint_load_benchmark \
  python3 checkpoint_load_benchmark/scripts/repro/verify_gcs_shards_streaming.py \
  --gcs-root gs://mesa-maxtext-europe-west4/mesa-bench-v2
```

> **Note:** You can also pass `--gcs-root gs://mesa-maxtext-us-central1/mesa-bench-v2` or `--gcs-root gs://mesa-maxtext-southamerica-west1/mesa-bench-v2` to stream from the closest regional bucket. Expected final line: `[PASS] All real GCS production shard streaming checks matched offline BF16 bit-for-bit.`

---

## Tier 3 — Live TPU Parity & Regression Reproduction on GKE (~15–25 min)

All GKE reproduction jobs use **user-scoped JobSet names (`--job-prefix "${USER}-repro"`)** and **personal GCS prefixes (`repro-${USER}/`)** so teammates never collide with each other or overwrite canonical `mesa-bench-v2` artifacts.

### 1. Select Your Target GKE TPU Cluster

| Platform | Cluster (`--platform`) | Region | Project | Regional Data Root (`--gcs-data-root`) |
| :--- | :--- | :--- | :--- | :--- |
| **`v7x`** (Default) | `bodaborg-tpu7x-nap` | `us-central1` | `cloud-tpu-shared-capacity` | `gs://mesa-maxtext-us-central1/mesa-bench-v2` |
| **`v5p`** | `bodaborg-v5p-nap` | `europe-west4` | `cloud-tpu-shared-capacity` | `gs://mesa-maxtext-europe-west4/mesa-bench-v2` |
| **`v6e`** | `bodaborg-v6e-nap` | `southamerica-west1` | `cloud-tpu-shared-capacity` | `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2` |
| **`v5e`** | `v5e-256-bodaborg-europe-west4` | `europe-west4` | `tpu-prod-env-multipod` | `gs://mesa-maxtext-europe-west4/mesa-bench-v2` |

Connect `kubectl` (example for `bodaborg-tpu7x-nap`):

```bash
gcloud container clusters get-credentials bodaborg-tpu7x-nap \
  --region us-central1 \
  --project cloud-tpu-shared-capacity
```

### 2. Build & Upload Your Committed Code Tarball

`scripts/build_code_tarball.py` archives `HEAD` (including `src/` and `checkpoint_load_benchmark/scripts/repro/`) with `CODE_VERSION.json` git provenance and uploads `maxtext_lance_prs.tar.gz` to your personal GCS code directory:

```bash
export REPRO_PREFIX="${USER}-repro"
export GCS_CODE_DIR="gs://mesa-maxtext/repro-${USER}/code"
export GCS_OUT_DIR="gs://mesa-maxtext-us-central1/repro-${USER}/results"

python3 checkpoint_load_benchmark/scripts/build_code_tarball.py \
  --out "/tmp/code_tarball_${USER}" \
  --upload "${GCS_CODE_DIR}"
```

*(For a local unpushed branch test, build with `--allow-unpushed --out "/tmp/code_tarball_${USER}"` and upload `/tmp/code_tarball_${USER}/maxtext_lance_prs.tar.gz` to `${GCS_CODE_DIR}/maxtext_lance_prs.tar.gz` via `gcloud storage cp`.)*

### 3. Generate & Apply the Single-Slice and Multi-Slice JobSets

Generate both JobSet manifests (`${REPRO_PREFIX}-reg.yaml` for 1-slice `2x2x1` 4-model regression and `${REPRO_PREFIX}-dsv.yaml` for 4-slice `4x 2x2x1` = 32-device full `deepseek4-284b` / `deepseek3-671b` shard-digest parity):

```bash
python3 checkpoint_load_benchmark/scripts/repro/build_tpu_verify_jobsets.py \
  --platform v7x \
  --job-prefix "${REPRO_PREFIX}" \
  --multislice-suite dsv4 \
  --gcs-code-dir "${GCS_CODE_DIR}" \
  --gcs-out-dir "${GCS_OUT_DIR}" \
  --out-dir "/tmp/repro_jobsets_${USER}"

kubectl apply -f "/tmp/repro_jobsets_${USER}/${REPRO_PREFIX}-reg.yaml"
kubectl apply -f "/tmp/repro_jobsets_${USER}/${REPRO_PREFIX}-dsv.yaml"
```

*(Pass `--multislice-suite both` to verify both `deepseek4-284b` and `deepseek3-671b` in the `4x 2x2x1` JobSet.)*

### 4. Monitor, Inspect JSON Results & Clean Up

```bash
# Watch JobSet & pod status
kubectl get jobsets "${REPRO_PREFIX}-reg" "${REPRO_PREFIX}-dsv"
kubectl logs -f "jobs/${REPRO_PREFIX}-reg-slice0-0"
kubectl logs -f "jobs/${REPRO_PREFIX}-dsv-slice0-0"

# Inspect summary JSON payloads written by rank 0
gcloud storage cat "${GCS_OUT_DIR}/${REPRO_PREFIX}-reg_result.json"
gcloud storage cat "${GCS_OUT_DIR}/${REPRO_PREFIX}-dsv_result.json"

# Always delete finished JobSets once complete
kubectl delete jobset "${REPRO_PREFIX}-reg" "${REPRO_PREFIX}-dsv"
```

### Reference TPU v7x (`bodaborg-tpu7x-nap`) Timings & Parity Results

| Suite & Topology | Model | `safetensors_dynamic` Default | `safetensors_dynamic` Streaming | Offline `BF16` Streaming | Peak HBM / Device | Parity Verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **1-Host `2x2x1` (`8` TPU v7x devices)** | `qwen3-0.6b` | `29.2s` | `8.2s` | — | `1.6 GB` | `100%` bit-exact (`default == streaming`) |
| **1-Host `2x2x1` (`8` TPU v7x devices)** | `gemma3-4b` | `52.8s` | `16.2s` | — | `4.5 GB` | `100%` bit-exact (`default == streaming`) |
| **1-Host `2x2x1` (`8` TPU v7x devices)** | `deepseek2-16b` | `298.2s` | `48.3s` | — | `9.2 GB` | `100%` bit-exact (`default == streaming`) |
| **1-Host `2x2x1` (`8` TPU v7x devices)** | `gpt-oss-20b` | `352.1s` | `55.7s` | — | `13.8 GB` | `100%` bit-exact (`default == streaming`) |
| **4-Host `4x 2x2x1` (`32` TPU v7x devices)** | `deepseek4-284b` (`FP8` + `MXFP4` + `F8_E8M0`) | — | **`480.1s`** | **`449.8s`** | **`23.6 GB`** (vs `23.6 GB` `BF16`) | **`224 / 224` weight leaves byte-exact** (`blake2b` digests across all 32 devices) |

---

## Tier 4 — Reproducing Benchmark `v2` Rows & PR #6353 / #6354 Large-Model Runs

> **Guardrail:** Always pass a personal `--run-id` / `--gcs-base` (or `--experiment-name "repro-${USER}"`) so canonical `mesa-bench-v2` results and Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) are never overwritten.

### A. Single-Command Smoke JobSets (`scripts/make_smoke_jobset.py`)

`make_smoke_jobset.py` generates an isolated single-slice JobSet writing under `<data_root>/smoke/run_<run_id>/` while reusing the existing regional `hf_cache/` (`--reuse-model-hf-cache`):

```bash
# Example 1 (PR #6353): Verify scanned composite HBM peak fix on gpt-oss-120b or qwen3.5-397b-a17b
python3 checkpoint_load_benchmark/scripts/make_smoke_jobset.py \
  --platform v7x \
  --topology 2x2x2 \
  --model qwen3.5-397b-a17b \
  --run-id "${USER}-pr6353" \
  --scan-modes scanned unscanned \
  --pipelines safetensors_dynamic \
  --regional \
  --code-dir "${GCS_CODE_DIR}" \
  --out "/tmp/smoke_qwen397b_${USER}.yaml"

kubectl apply -f "/tmp/smoke_qwen397b_${USER}.yaml"
```

```bash
# Example 2 (PR #6354 + Benchmark v2): Run both to_maxtext and safetensors_dynamic on a v2 row
python3 checkpoint_load_benchmark/scripts/make_smoke_jobset.py \
  --platform v7x \
  --topology 2x2x1 \
  --model qwen3-30b-a3b \
  --run-id "${USER}-v2row" \
  --scan-modes scanned unscanned \
  --pipelines both \
  --regional \
  --code-dir "${GCS_CODE_DIR}" \
  --out "/tmp/smoke_qwen30b_${USER}.yaml"

kubectl apply -f "/tmp/smoke_qwen30b_${USER}.yaml"
```

### B. Direct Harness Execution with a Personal GCS Output Prefix

Inside an interactive TPU VM or pod, invoke `benchmark_ckpt_load_time.py` with a personal `--gcs-base` (`gs://.../repro-${USER}`) and `--run-id` (or pass `--experiment-name "repro-${USER}"` to `scripts/generate_v2_manifests.py`):

```bash
PYTHONPATH=$PWD/src:$PWD/checkpoint_load_benchmark \
  python3 -u checkpoint_load_benchmark/benchmark_ckpt_load_time.py \
  --models llama3.1-70b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 \
  --num-hosts-per-slice 1 \
  --devices-per-slice 8 \
  --hardware tpu \
  --platform v7x \
  --slice-topology 2x2x1 \
  --gcs-base "gs://mesa-maxtext-us-central1/repro-${USER}" \
  --run-id "${USER}-repro-70b" \
  --local-work-dir "/ramdisk/bench_${USER}" \
  --csv-report "/tmp/repro_70b_${USER}.csv" \
  --verify-checksums
```

To test PR #6354's Orbax write-buffer cap directly during a standalone `to_maxtext` conversion on host RAM-constrained machines, pass `checkpoint_storage_concurrent_gb=<gb>` (e.g. `checkpoint_storage_concurrent_gb=40`) on the `maxtext.checkpoint_conversion.to_maxtext` CLI.
