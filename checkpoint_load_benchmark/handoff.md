# Handoff: Checkpoint Load-Time Benchmark on TPU v7 (`bodaborg-tpu7x-nap`)

> **Snapshot:** 2026-10-28 ~15:05 UTC · **Owner:** mesakh · **Branch:** `docs/checkpoint-load-benchmark` (pushed; pods run the copies in `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`)
>
> The numbers below come from the per-phase result JSONs in GCS at snapshot time. Pods were still running, so
> re-check §6 before acting. **Per-test progress table:** [`reports/tpu7x_priority1_progress.md`](reports/tpu7x_priority1_progress.md). To refresh the status tables, run
> [`scripts/summarize_tpu7x_results.py`](scripts/summarize_tpu7x_results.py) (see §9).

---

## 0. TL;DR

| Workstream | Status |
| :--- | :--- |
| **P1 · Exp 1A WITH Lance's 3 PRs** (`safetensors_dynamic`) | ✅ Done. 235B 4/4, 284B 4/4, 480B 2/2 that fit, 397B unscanned · 2 (1,311.9 s) and scanned · 4 (1,399.7 s), 671B on 4 slices (649.4 s / 531.0 s). 397B scanned on 1–2 slices runs out of HBM, also with `hbmfix` (§7.19) |
| **P1 · Exp 1A BEFORE** (`origin/main`) | ✅ Done. Every BEFORE run **timed out at 1800 s while still loading** (235B · 1 and · 2, 284B · 1 and · 2, 397B unscanned · 2, 671B · 4), except 480B, which hits `KeyError`. With the PRs, the same loads take 353–708 s |
| **P1 · Exp 1B** (`to_maxtext` vs `safetensors_dynamic`) | `to_maxtext` S2 can't run on one tpu7x host for scanned 235B/397B/480B, unscanned 480B, and 284B (§7.16). Unscanned 235B converted in **5,150.1 s** (shards streamed); its Orbax loads are running. The unscanned 397B conversion is ~27% done, with loads to follow (ETA ~17:45 UTC) |
| **P1 completion run** | GCS task queue (`configs/p1_completion_tasks.json`, 29 tasks). All tasks are done except the two unscanned conversion chains (235B, 397B). Workers left: `s3` (235B), `s8` (397B), standby `s1`, pairs `i` and `k` for the 2-slice loads |
| **P2 · 16-model suite** | ✅ Done (`p2_leftovers` and re-runs). `gpt-oss-120b` scanned `safetensors_dynamic` runs out of HBM (U-6); unscanned loads. `llama3.1-70b` scanned · 2 Orbax load needed `concurrent_gb=38` (T-13). Re-ran the `qwen3-14b` and `qwen3-32b` scanned · 1 outliers (U-7) |
| **Reports** | Written: `reports/tpu7x_large_model_200b_experiment_report.md`, `tpu7x_benchmark_ckpt_load_time_{summary.md,summary.csv,report.md,issues.md}`, plus the live `tpu7x_priority1_progress.md`. To do: fill in the 235B/397B unscanned `to_maxtext` results when they land (re-run `scripts/build_tpu7x_tables.py --md-into`), then clean up the cluster |

---

## 1. Goal & experiment design

- **Priority 1 (>200B models)**: `qwen3-235b-a22b`, `deepseek4-284b`, `qwen3.5-397b-a17b`, `qwen3-480b-a35b`, `deepseek3-671b`.
  - **Exp 1A**: `safetensors_dynamic` **BEFORE** (`origin/main`) vs **WITH** Lance Wang's 3 PRs (#5706, #5828, #5705).
  - **Exp 1B**: `to_maxtext` (S1 HF→GCS, S2 CPU conversion to an Orbax checkpoint, S3 Orbax load into HBM) vs `safetensors_dynamic` (S1 HF→GCS, S2 sharded safetensors load and in-memory conversion).
- **Priority 2**: rerun the earlier 16-model `to_maxtext` vs `safetensors_dynamic` benchmark on tpu7x (WITH the PRs).
- Every model runs in 4 configs: {`scanned`, `unscanned`} × {1 slice, 2 slices}.
- **Deliverables**: separate `reports/tpu7x_*` files (the existing v5p reports stay untouched), plus scripts and docs in this folder that reproduce every result.

---

## 2. Code & branches

| Ref | Commit | Contents |
| :--- | :--- | :--- |
| `origin/main` | `f596f5369` | Baseline for the BEFORE runs |
| `origin/fix/safetensors-dynamic-lance-prs` | `335d5e880` | `origin/main` plus Lance's PRs: `df2eacf0d` (#5706: composite keys, multi-host hooks, legacy keys), `1a89e9dfc` (#5705: `safetensors_dynamic` outside the training restore), `1f06b304d` + `335d5e880` (#5828: leading-axis sharded reads, inner MoE stacks kept in the read layout) |
| `origin/docs/checkpoint-load-benchmark` | `ef8488fec` | The above plus this benchmark suite (pushed) |

**On this branch** (on top of `ef8488fec`; the pods run the GCS copies uploaded by `run_tpu7x_cluster.sh`):

| Path | Contents |
| :--- | :--- |
| `benchmark_ckpt_load_time.py` | tpu7x support, zlib-compressed remote payload, env filtering that keeps `VBAR_CONTROL_SERVICE_URL`, SIGABRT teardown tolerance, phase watchdog with `TIMEOUT` labels, `--skip-safetensors`, `--tm-convert-only`, RAM-aware staging and fail-fast RAM checks for `to_maxtext` (§7.15, §7.16), shared-cache reuse when all shards are present (§7.17) |
| `bootstrap_tpu7x.py`, `generate_tpu7x_manifests.py`, `scripts/run_tpu7x_cluster.sh` | Pod bootstrap (trees: Lance's PRs, `origin/main`, `hbmfix`), manifests (P1/P2 pairs, P1-completion pair `g`, daemon controllers, recovery), launcher |
| `scripts/tpu7x_task_queue.py`, `scripts/tpu7x_daemon_controller.py`, `configs/p1_completion_tasks.json` | P1 completion queue (README "Finishing Priority 1"); workers sync the harness before each task, and tasks can be limited to newer pods with the `/tmp/maxtext_stagefix` tree (§7.16) |
| `scripts/summarize_tpu7x_results.py` | Status matrix from per-phase result JSONs |
| `patches/hbmfix.patch`, `patches/hbmfix/src/...` | HBM-peak fix for composite keys (§7.6). Same 63 passing and 13 failing unit tests as HEAD in a local CPU venv (the 13 failures are from the old local JAX); `src/` itself stays equal to Lance's PRs |
| Not committed: `docs/plans/`, `run_e2e_matrix_xpk.{py,sh}`, `vllm_decoder_issue.md` | Unrelated to this work |

Code bundles verified against git: `code/maxtext_lance_prs.tar.gz` `src/` = this branch's `src/` (`ef8488fec`); `code/checkpoint_conversion_origin_main.tar.gz` = `f596f5369:src/maxtext/checkpoint_conversion`.

**What the BEFORE tree contains:** `/tmp/maxtext_origin_main` is the Lance-PRs tree with `src/maxtext/checkpoint_conversion` replaced by the `f596f5369` copy. Lance's PRs also touch 3 files outside that directory (`utils/globals.py`: extra `HF_IDS`; `utils/max_utils.py`: `_uses_checkpoints` for `jax.distributed` init; `utils/model_creation_utils.py`: `from_pretrained` support). None of them is on the measured path: the harness calls `jax.distributed.initialize` itself and calls `checkpointing.load_state_if_possible(source_checkpoint_layout="safetensors_dynamic")` directly (`checkpointing.py` is unchanged by the PRs).

---

## 3. Infrastructure (what the experiments ran on)

| Item | Value |
| :--- | :--- |
| Cluster | `bodaborg-tpu7x-nap` (kube context `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap`), project `cloud-tpu-shared-capacity`, zone `us-central1-c` |
| Accelerator | TPU v7 (Ironwood) `tpu7x`; machine type `tpu7x-standard-4t` |
| One slice | 1 node, topology **`2x2x1`** = **4 chips = 8 JAX TPU devices** (2 TensorCores per chip), `google.com/tpu: 4` |
| TPU HBM | **813.8 GB per slice** (8 × 101.725 GB); **1,627.6 GB on 2 slices** |
| Host | **1,014.5 GB RAM** (944.83 GiB) and 224 vCPUs per node; RAM-backed `emptyDir` mounted at `/ramdisk` |
| Capacity | **Spot** (`cloud.google.com/gke-spot: "true"`, `tpu7x.gke.io/capacity: spot`), admitted through Kueue queue `default`. Pods sit in `SchedulingGated` until Kueue admits them. |
| 1-slice runs | Single process, 8 devices, `ici_fsdp_parallelism=-1` (mesh `fsdp=8`) |
| 2-slice runs | **Multi-slice**: 2 single-host slices over DCN with `jax.distributed.initialize` (2 processes). Slice 0 runs the harness and coordinates; slice 1 runs a daemon on TCP `55555`. Models ≥70B use `dcn_fsdp_parallelism=2` (mesh `fsdp=16`); models ≤47B use `dcn_data_parallelism=2` (mesh `data=2 × fsdp=8`). |
| Container images | Bootstrap pods: `gcr.io/cloud-tpu-shared-capacity/yaoyuchen-runner:ds41-2026-10-22-16-03-48` (JAX 0.11.1). Pair `d` only: `us-central1-docker.pkg.dev/cloud-tpu-shared-capacity/bodaborg-images/maxtext-base-image:latest` (§7.11). |
| HF token | k8s secret `mesakh-hf-token` (key `HF_TOKEN`); locally in `hf_token.txt`. **Never print it.** |
| Permissions | No `pods/exec` for our account, so inspect runs through `kubectl logs` and GCS only |

**Capacity of each P1 model (BF16 weights vs. HBM):**

| Model | BF16 size / shards | 1 slice (813.8 GB) | 2 slices (1,627.6 GB) |
| :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 470.38 GB / 118 | ✅ fits | ✅ fits |
| `deepseek4-284b` | 581.89 GB / 146 | ✅ fits | ✅ fits |
| `qwen3.5-397b-a17b` | 795.48 GB / 94 | ❌ 97.7% of HBM, no headroom left for the transform | ⚠️ weights fit; the scanned transform peak OOMs (§7.6) |
| `qwen3-480b-a35b` | 961.03 GB / 241 | ❌ exceeds HBM | ✅ fits |
| `deepseek3-671b` | 1,342.35 GB (staged sparse copy: 1,365.27 GB / 343) | ❌ exceeds HBM | ❌ host OOM-killed during the load; ✅ loads on 4 slices (690 GB peak host RAM per host) |

---

## 4. How the runs are orchestrated

- **Pod pair `<id>`**: `mesakh-tpu7x-<id>0` (`ROLE=runner`, slice 0, runs `RUNNER_SCRIPT`), `mesakh-tpu7x-<id>1` (`ROLE=daemon`, slice 1, `--remote-slice-daemon --daemon-port 55555`), and the headless Service `mesakh-w1-<id>`.
  [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py) generates them and [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) deploys them.
  - P1 pairs: `a` = 235B, `b` = 397B, `c` = 480B, `f` = 284B + 671B. Each runs 3 stages: WITH PRs → BEFORE (`origin/main`, watchdog `PHASE_TIMEOUT_S`: 360 s when these pairs were deployed, 900 s in the current generator) → full TM vs ST comparison.
  - P2 pairs: `d` = 10 small and medium models, `e` = `qwen3-32b`, `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b`.
- **What [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py) does at pod start** (it is downloaded from `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`):
  1. Installs missing Python packages from `site_packages_overlay.tar.gz`, without shadowing the image's jax, flax, or orbax.
  2. Builds `/tmp/maxtext` from `maxtext_lance_prs.tar.gz` (the Lance-PRs tree), then overwrites `benchmark_ckpt_load_time.py` with the latest copy from GCS.
  3. Builds `/tmp/maxtext_origin_main` by copying that tree and overlaying `checkpoint_conversion_origin_main.tar.gz`. The BEFORE runs use this tree.
  4. **Patches the code by string replacement.** In both trees, `load_dynamic.py` gets single-GET HF→GCS streaming with 16 workers. In `/tmp/maxtext` only, `tensor_handling.py` and `load_dynamic.py` get the older HBM patch: free reshard/stack intermediates and split composite leaves before the reshard.
  5. Builds `/tmp/maxtext_hbmfix`: a copy of the patched `/tmp/maxtext` whose `load_dynamic.py` and `tensor_handling.py` are replaced by the §7.6 fix from `gs://…/code/` (same files as `patches/hbmfix/`), then re-patched for single-GET streaming. Only queue tasks labeled `hbmfix` use this tree.
  > ⚠️ **So the "WITH PRs" numbers are Lance's PRs plus the step-4 string patches**, not the bare PRs. I checked that every patch pattern matches the PR code (so each patch applies). The BEFORE tree gets only the single-GET streaming patch, which affects Step 1 (HF→GCS) and not the timed S2 load.
- **Recovery controller `mesakh-ctrl-c1`** (a `python:3.11-slim` pod on a CPU node). After spot preemptions, it reuses the two surviving daemon pods: `c1` (10.224.42.10) acts as slice 0 and `f1` (10.224.75.10) as slice 1. It first syncs the harness from GCS to both, then runs this queue on `c1`:
  ```bash
  export PYTHONPATH=/tmp/maxtext/src:/tmp/maxtext HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp KEEP_HF_CACHE=1
  COMMON="--devices-per-slice 8 --hardware tpu --remote-slice-addrs 10.224.75.10:55555 --run-id tpu7x_200b --reuse-model-hf-cache --skip-completed --allow-step-failure"
  # 1) qwen3.5-397b WITH PRs, 2 slices                                     -> results/p1_st_lance_b_s2.csv   [DONE: failed, §7.6]
  python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models qwen3.5-397b-a17b \
    --scan-modes scanned unscanned --num-slices-list 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \
    --local-work-dir /tmp/p1_lance_prs --csv-report /tmp/p1_st_lance_b_s2.csv --skip-to-maxtext $COMMON
  # 2) DeepSeek BEFORE (origin/main), 1+2 slices, 360 s watchdog            -> results/p1_origin_main_f.csv   [DONE: inconclusive, §7.5]
  PYTHONPATH=/tmp/maxtext_origin_main/src:/tmp/maxtext_origin_main STAGE_SPARSE_DEEPSEEK_OOM=1 PHASE_TIMEOUT_S=360 \
  SHARED_HF_CACHE_BASE_DIR=gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/safetensors_dynamic \
  python3 -u /tmp/maxtext_origin_main/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models deepseek4-284b deepseek3-671b \
    --scan-modes scanned unscanned --num-slices-list 1 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main \
    --local-work-dir /tmp/p1_origin_main --csv-report /tmp/p1_origin_main_f.csv --skip-to-maxtext $COMMON
  # 3) qwen3-235b full to_maxtext + safetensors_dynamic, 1+2 slices        -> results/p1_full_a.csv          [INTERRUPTED ~09:01 UTC, §7.14]
  python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models qwen3-235b-a22b \
    --scan-modes scanned unscanned --num-slices-list 1 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \
    --local-work-dir /tmp/p1_lance_prs --csv-report /tmp/p1_full_a.csv $COMMON
  ```
- **Harness rules that shape the result tables:**
  - If `scanned` fails with OOM or timeout, the harness **skips `unscanned`** for that slice count.
  - `--skip-completed` reuses result JSONs from GCS. FAILED results are **re-run by default**; with `SKIP_FAILED_RESULTS=1` they are skipped instead (§7.7).

---

## 5. Results (snapshot)

### 5.1 / 5.2 P1 · Exp 1A and Exp 1B

The per-test P1 tables (summary by model, the row-by-row `to_maxtext` vs `safetensor` table, and all tests) are kept current in
[`reports/tpu7x_priority1_progress.md`](reports/tpu7x_priority1_progress.md). Highlights are in §0. Each P1 model's cold Step 1 time is 583.3 s (235B), 814.5 s (397B), 1,046.1 s (480B, `to_maxtext` S1 measured separately: 1,135.5 s), or 0 for the synthetic DeepSeek files.

### 5.3 P2 · 16-model suite (WITH PRs)

| Model | 4/4 configs done? | Missing |
| :--- | :--- | :--- |
| `qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `llama3.1-8b`, `gemma2-2b`, `gemma3-4b`, `deepseek2-16b`, `gpt-oss-20b`, `mixtral-8x7b` | ✅ | – |
| `qwen3-32b` | ⚠️ | TM S3 for scanned · 2 slices |
| `qwen3-30b-a3b` | ⚠️ | unscanned: TM S2 failed (§7.8), so TM S3 is missing for both slice counts; ST is missing for unscanned · 2 slices |
| `llama3.1-70b` | ⏳ | Running on pair `e` (TM S2 scanned). Its unscanned TM S1 was killed earlier and will be retried (§7.9). |
| `gpt-oss-120b` | ⏳ | Queued on pair `e` (and, as a duplicate, on pair `d`, §7.11) |
| `deepseek4-284b`, `deepseek3-671b` | ST only | Covered by the P1 runs (§5.1); TM not run |

<details>
<summary>Full P2 matrix (generated by <code>scripts/summarize_tpu7x_results.py</code>)</summary>

| Model | Config | TM S1 HF→GCS | TM S2 convert | TM S3 load Orbax | ST S2 load+convert | ST peak host RAM (GB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-0.6b` | scanned, 1 slice | 16.0s | 19.4s | 2.5s | 16.1s | 26.8 |
| `qwen3-0.6b` | unscanned, 1 slice | 16.0s | 66.8s | 3.3s | 6.5s | 26.8 |
| `qwen3-0.6b` | scanned, 2 slices | (shared) | (shared) | 3.3s | 16.1s | 73.6 |
| `qwen3-0.6b` | unscanned, 2 slices | (shared) | (shared) | 3.4s | 9.5s | 73.4 |
| `qwen3-4b` | scanned, 1 slice | 51.9s | 62.2s | 14.2s | 24.7s | 31.0 |
| `qwen3-4b` | unscanned, 1 slice | 51.9s | 95.9s | 9.7s | 14.4s | 31.1 |
| `qwen3-4b` | scanned, 2 slices | (shared) | (shared) | 18.9s | 22.8s | 69.6 |
| `qwen3-4b` | unscanned, 2 slices | (shared) | (shared) | 9.8s | 15.0s | 70.7 |
| `qwen3-8b` | scanned, 1 slice | 54.3s | 96.5s | 21.6s | 43.1s | 33.4 |
| `qwen3-8b` | unscanned, 1 slice | 54.3s | 119.9s | 13.3s | 28.2s | 33.6 |
| `qwen3-8b` | scanned, 2 slices | (shared) | (shared) | 22.5s | 34.0s | 68.2 |
| `qwen3-8b` | unscanned, 2 slices | (shared) | (shared) | 11.4s | 22.3s | 67.5 |
| `qwen3-14b` | scanned, 1 slice | 51.3s | 145.9s | 48.9s | 342.7s ⚠️ | 34.3 |
| `qwen3-14b` | unscanned, 1 slice | 51.3s | 144.7s | 19.7s | 40.3s | 34.5 |
| `qwen3-14b` | scanned, 2 slices | (shared) | (shared) | 61.9s | 69.0s | 68.8 |
| `qwen3-14b` | unscanned, 2 slices | (shared) | (shared) | 19.0s | 57.9s | 68.0 |
| `qwen3-32b` | scanned, 1 slice | 107.2s | 381.5s ⚠️ | 203.7s ⚠️ | 377.7s ⚠️ | 33.9 |
| `qwen3-32b` | unscanned, 1 slice | 107.2s | 208.0s | 36.2s | 77.6s | 35.0 |
| `qwen3-32b` | scanned, 2 slices | (shared) | (shared) | ⏳ | 88.9s | 70.3 |
| `qwen3-32b` | unscanned, 2 slices | (shared) | (shared) | 30.4s | 74.7s | 70.3 |
| `llama3.1-8b` | scanned, 1 slice | 62.9s | 89.6s | 22.0s | 51.8s | 33.0 |
| `llama3.1-8b` | unscanned, 1 slice | 62.9s | 105.3s | 14.3s | 36.7s | 37.0 |
| `llama3.1-8b` | scanned, 2 slices | (shared) | (shared) | 17.9s | 28.5s | 67.6 |
| `llama3.1-8b` | unscanned, 2 slices | (shared) | (shared) | 12.5s | 18.1s | 68.6 |
| `llama3.1-70b` | scanned, 1 slice | 197.8s | ⏳ | ⏳ | ⏳ | - |
| `llama3.1-70b` | unscanned, 1 slice | ❌ killed (-9) | ⏳ | ⏳ | ⏳ | - |
| `llama3.1-70b` | scanned, 2 slices | (shared) | (shared) | ⏳ | ⏳ | - |
| `llama3.1-70b` | unscanned, 2 slices | (shared) | (shared) | ⏳ | ⏳ | - |
| `gemma2-2b` | scanned, 1 slice | 169.4s | 54.4s | 5.3s | 28.0s | 35.4 |
| `gemma2-2b` | unscanned, 1 slice | 169.4s | 91.2s | 7.8s | 19.9s | 35.8 |
| `gemma2-2b` | scanned, 2 slices | (shared) | (shared) | 6.7s | 24.9s | 72.6 |
| `gemma2-2b` | unscanned, 2 slices | (shared) | (shared) | 8.5s | 17.9s | 72.5 |
| `gemma3-4b` | scanned, 1 slice | 55.8s | 56.4s | 7.8s | 24.3s | 33.5 |
| `gemma3-4b` | unscanned, 1 slice | 55.8s | 108.9s | 9.9s | 17.8s | 33.4 |
| `gemma3-4b` | scanned, 2 slices | (shared) | (shared) | 7.0s | 24.1s | 80.6 |
| `gemma3-4b` | unscanned, 2 slices | (shared) | (shared) | 8.1s | 21.2s | 81.5 |
| `deepseek2-16b` | scanned, 1 slice | 137.3s | 76.7s | 94.4s | 75.0s | 38.4 |
| `deepseek2-16b` | unscanned, 1 slice | 137.3s | 147.2s | 18.2s | 58.4s | 42.4 |
| `deepseek2-16b` | scanned, 2 slices | (shared) | (shared) | 112.5s | 125.5s | 115.5 |
| `deepseek2-16b` | unscanned, 2 slices | (shared) | (shared) | 17.6s | 115.0s | 116.7 |
| `gpt-oss-20b` | scanned, 1 slice | 93.1s | 102.6s | 63.3s | 127.5s | 39.2 |
| `gpt-oss-20b` | unscanned, 1 slice | 93.1s | 185.8s | 12.6s | 78.3s | 37.2 |
| `gpt-oss-20b` | scanned, 2 slices | (shared) | (shared) | 58.3s | 111.9s | 67.1 |
| `gpt-oss-20b` | unscanned, 2 slices | (shared) | (shared) | 12.2s | 104.5s | 68.3 |
| `qwen3-30b-a3b` | scanned, 1 slice | 78.8s | 151.6s | 386.1s | 129.5s | 61.4 |
| `qwen3-30b-a3b` | unscanned, 1 slice | 78.8s | ❌ exit 1 | ⏳ | 110.7s | 61.0 |
| `qwen3-30b-a3b` | scanned, 2 slices | (shared) | (shared) | 449.3s | 385.5s | 156.1 |
| `qwen3-30b-a3b` | unscanned, 2 slices | (shared) | (shared) | ⏳ | ⏳ | - |
| `mixtral-8x7b` | scanned, 1 slice | 239.7s | 134.1s | 533.6s | 126.9s | 39.6 |
| `mixtral-8x7b` | unscanned, 1 slice | 239.7s | 208.3s | 32.0s | 115.6s | 39.1 |
| `mixtral-8x7b` | scanned, 2 slices | (shared) | (shared) | 792.5s | 63.1s | 74.7 |
| `mixtral-8x7b` | unscanned, 2 slices | (shared) | (shared) | 47.7s | 56.6s | 69.6 |

⚠️ marks outliers to re-run before they go into a report (§8).

</details>

**Early findings to verify in the reports:**
- For **scanned MoE** models, the Orbax restore (TM S3) is slow compared with ST: `mixtral-8x7b` takes 533.6 s / 792.5 s vs. ST 126.9 s / 63.1 s; `qwen3-30b-a3b` takes 386.1 s / 449.3 s.
- For small dense models, TM S3 is faster than ST S2, but TM needs S1 + S2 first (an offline conversion of minutes per model).

---

## 6. Live cluster state (at snapshot)

Manifests: `tpu7x_p1_completion_scaleout.yaml` (pairs `i`, `j`, singles `s1`–`s3`, 4-slice group `m`) and `tpu7x_p1_completion_scaleout2.yaml` (pair `k`, singles `s4`–`s8`), both from `generate_tpu7x_manifests.py`. All are queue workers (`scripts/tpu7x_task_queue.py`) on spot `2x2x1` nodes. Pods started after 13:31 UTC (`s1`, `s3`, `s8`, `m0`, `m3`) have the `/tmp/maxtext_stagefix` tree and sync the harness before every task (§7.16).

| Worker | Pods | Slices | Task at snapshot (15:05 UTC) |
| :--- | :--- | :---: | :--- |
| `s3` | `mesakh-tpu7x-s3` | 1 | `tm_235b_unscanned_s1`: conversion done (5,150.1 s), 1-slice Orbax load running |
| `s8` | `mesakh-tpu7x-s8` | 1 | `tm_397b_unscanned_s1`: conversion ~27% (streams shards; ETA ~17:15 UTC, then save and load) |
| `s1` | `mesakh-tpu7x-s1` | 1 | idle standby with the `stagefix` tree (can take over `s3`/`s8` tasks) |
| `i`, `k` | `mesakh-tpu7x-{i,k}{0,1}` | 2 | idle until `tm_235b_unscanned_s2` / `tm_397b_unscanned_s2` become claimable |

Pending: `tm_235b_unscanned_s2`, `tm_397b_unscanned_s2`. Every other task is done. Group `m`, pair `j`, and singles `s2`, `s4`–`s7` were deleted after their work finished.

Gone (all recorded in §7.14 and §7.18):
- `c1`, `f1`, `g0`, then `g1` at 13:32 (Kueue `The workload is deactivated`); `mesakh-ctrl-c1` is `Completed`, `mesakh-ctrl-g1` and `h1` were deleted. Delete `mesakh-ctrl-c1` at cleanup.
- `s1` (evicted at 944 GB host memory) and `s3` (OOM-killed): full staging for the 235B conversion (§7.16); both recreated.
- `m3` and `s7`: spot node shutdown at 13:42. `m3` and `m0` were recreated; `s7` was not needed.
- `s8` first landed on node `gke-tpu-fd9dfbd0-2jfc`, whose kubelet stopped responding; recreated elsewhere. `s5` deleted while idle to free a node for `m0`.
- P2 pair `e`: evicted earlier (its leftovers are in `p2_leftovers`). P2 pair `d`: deleted (misconfigured, §7.11).

---

## 7. Known issues & findings

1. **tpu7x TPU init needs `VBAR_CONTROL_SERVICE_URL`.** GKE injects `VBAR_CONTROL_SERVICE_URL=unix:///var/run/tpu-plugin/vbar.sock`. Filtering out `*_SERVICE_*` env vars drops it, and TPU init then fails with `Failed to connect to [::]:8353`. *Fixed*: the harness now strips only `*_SERVICE_HOST`, `*_SERVICE_PORT*`, and `*_TCP*`. Two stored records predate the fix: `qwen3.5-397b-a17b` WITH scanned · 1 and an older `deepseek3-671b` `origin/main` scanned · 2. Re-run them.
2. **Linux `MAX_ARG_STRLEN` (131,072 bytes).** The harness ships its own source to the remote daemon as one argument, which grew past the limit (`[Errno 7] Argument list too long`). *Fixed*: zlib then base64, about 28 KB.
3. **SIGABRT (-6) at multi-slice interpreter teardown.** *Fixed*: the run is not marked failed when its result JSON is already complete.
4. **Watchdog kills are recorded as `OOMKilled / Exit Code 137`.** The harness labels every `-9` exit as OOM (`benchmark_ckpt_load_time.py` ~L2276/L2417), including SIGKILLs from `PHASE_TIMEOUT_S`. Every BEFORE `-9` above had a peak host RAM of ≤194 GB out of 1,014.5 GB, so they were timeouts, not OOMs. *Fixed for new runs*: the harness now records `TIMEOUT` when its watchdog fired. Records written before the fix still say OOM; reclassify them from the `[TIMEOUT_WATCHDOG]` lines in the run output when writing the reports.
5. **A watchdog shorter than the WITH-PRs time is inconclusive.** The DeepSeek BEFORE runs were killed at 360 s, but with the PRs they need 353–453 s. Likewise, `qwen3.5-397b-a17b` scanned · 1 took 947.7 s just to load with the PRs. **Re-run BEFORE with `PHASE_TIMEOUT_S=1800`** or more. The original P1 manifests (deployed 06:00 UTC) used `PHASE_TIMEOUT_S=360` for all BEFORE runs, so `qwen3-235b-a22b` (WITH: 697–792 s) is inconclusive too; the generator was later changed to 900 s.
6. **`qwen3.5-397b-a17b` scanned · 2-slice HBM OOM in the composite `(wi_0, wi_1)` path.**
   - **What happens:** each layer's `mlp.experts.gate_up_proj` is 512 × 2048 × 4096 BF16, about 8.6 GB. 45 local layers are stacked into one composite target `(…, 2)` of about 387 GB, about 24.2 GB per device across 16 devices. It is then sliced into `wi_0` and `wi_1` (about 12.1 GB per device each, which is the failed `11.25G` allocation) while the rest of the model already fills HBM.
   - **What changed:** the fix (`patches/hbmfix/`, deployed on the pods as the `/tmp/maxtext_hbmfix` tree and run as `st_397b_scanned_s2_hbmfix`) does three things:
     - Frees each consumed HF input right after its hook, but only tensors popped from `hf_state` on their last use. Tensors shared by several leaves are kept.
     - Frees stack, `astype`, and `moveaxis` intermediates as soon as their successor is ready.
     - Makes `_reshard(..., delete_input=)` opt-in, passed only where the input is a known intermediate.
   - **Caveat:** the composite split itself still briefly holds `loaded` plus both parts. If that is still too much, split at the leaf level instead: split each layer into `wi_0`/`wi_1` and stack the two separately.
   - **Fastest path to a 397B data point:** run unscanned on 2 slices. Per-layer composites are about 0.5 GB per device, and `qwen3-480b-a35b` unscanned · 2 (961 GB) already succeeded. *Done at 13:12 UTC: 1,311.9 s (load 515.9 s + transform 795.7 s), peak host RAM 98.7 GB.*
7. **`SKIP_FAILED_RESULTS` docs were inverted** (now fixed in the README and the harness docstring). In code, with `--skip-completed`, FAILED results are re-run by default, and `SKIP_FAILED_RESULTS=1` skips them. The P1 pair manifests and **pair `d`** set it to `1`.
8. **`qwen3-30b-a3b` unscanned TM S2 failed with `HFValidationError`.** The local staging dir `/tmp/p2_lance_prs/hf_local/qwen3-30b-a3b` was missing (an older harness staging path), so `to_maxtext` treated the path as a repo id. The current harness re-stages into `/ramdisk/hf_local/`, so a plain retry should work.
9. **`llama3.1-70b` unscanned TM S1 was killed (-9) at 21.6 GB RAM during the HF download** (07:45, an external kill). Pair `e` doesn't set `SKIP_FAILED_RESULTS`, so it will retry this automatically.
10. **The DeepSeek P1 numbers use `STAGE_SPARSE_DEEPSEEK_OOM=1`**: synthetic sparse BF16 safetensors on local disk, with no HF→GCS step. They measure load/transform mechanics and capacity, not GCS read bandwidth, so don't compare them directly with the Qwen GCS loads.
11. **Pair `d` is inconsistent with the other pairs:**
    - It uses a different image and runs the pushed branch (`git clone`) without the bootstrap patches.
    - It sets `SKIP_FAILED_RESULTS=1`, so it would **not** retry `qwen3-30b-a3b` unscanned.
    - It queues `gpt-oss-120b`, which pair `e` also runs. The two would race on the same GCS result paths.
12. **P2 outliers:** `qwen3-14b` scanned · 1 ST took 342.7 s (298.4 s in the transform, vs. 69.0 s on 2 slices). `qwen3-32b` scanned · 1 is 3–5× slower than unscanned in every phase. Re-run both before reporting.
13. **`run_tpu7x_cluster.sh --cleanup`** deletes the pods labeled `app=mesakh-tpu7x-bench`, every `mesakh-w<k>-<id>` daemon Service (they carry no labels), and `mesakh-ctrl-c1` (created before the controllers got the label). `--status` still only lists the labeled pods.
14. **Kueue can evict running pods.** At ~09:01 UTC, `c1` was stopped after about 3 h with event `EvictedDueToDeactivated: The workload is deactivated`. Our workloads set no `maximumExecutionTimeSeconds`, so the cause isn't visible to us (likely a cluster-level deactivation or quota action). Assume any pod can disappear: keep queues short, rely on the per-phase result JSONs in GCS (`--skip-completed` resumes from them), and don't depend on a 2-pod pairing surviving.
    - Root cause confirmed later: the `default` ClusterQueue has **0 nominal tpu7x quota** and borrows all its chips from the cohort, so `priority-dev` reclaims them (`Preempted ... due to reclamation within the cohort`). Spot nodes also go NotReady. This is why the P1 completion run uses the GCS task queue.
15. **Scanned `to_maxtext` conversion thrashed when streaming shards.** For very large models, the harness used to stage only index/config files and stream shards from GCS through a 3-shard LRU cache. A scanned MoE leaf stacks each expert across all layers, so every leaf reads every shard. With a 3-shard cache, the checkpoint was re-downloaded once per expert: the 235B scanned conversion ran 2 h at leaf 1/15.
    - *Fixed*: the harness stages all shards when they fit within 55% of `/ramdisk` free space (235B: 470 GB). For scanned conversions that don't fit (397B: 795 GB, 480B: 961 GB on a 1,014.5 GB host whose boot disk is only 100 GB), it fails fast with the reason. Unscanned leaves read one layer at a time, so streaming stays for them.
    - Re-staging inside Step 2 is now recorded as `hf_to_local_restage_s`.
16. **`to_maxtext` S2 needs host RAM for the whole converted state.** `to_maxtext` materializes every leaf (lazy loading only delays it) and keeps the full converted state in host RAM until the Orbax save, so it needs about the checkpoint size again on top of any staged shards. A tpu7x node has 1,014.5 GB RAM, 986.7 GB allocatable.
    - **Evidence:** the 235B scanned conversion with all 470 GB staged in `/ramdisk` was OOM-killed at leaf 4/15 (13:23 UTC). The unscanned one was evicted at 944 GB container memory (13:32 UTC). `deepseek4-284b` S2 (both scan modes) was OOM-killed at 989.6 GB.
    - **Fix in the harness:** stage on `/ramdisk` only if `free − checkpoint − 60 GB ≥ checkpoint`, otherwise stream shards (unscanned) or fail fast with the reason (scanned). Also fail fast when the converted state alone exceeds 93% of host RAM (480B unscanned: 960 GB vs ~943 GB).
    - **Result:** TM S2 can't run on one host for scanned 235B/397B/480B, unscanned 480B, and 284B. Unscanned 235B and 397B stream and are running.
    - **Deployment:** queue workers run the harness they downloaded at pod start. So the affected tasks name the tree `/tmp/maxtext_stagefix`, a symlink that only pods bootstrapped after the fix have, and older pods never claim them. `tpu7x_task_queue.py` now also syncs the harness from GCS before each task (pods started after 13:31 UTC), so later harness fixes need no gating. The 2-slice TM load tasks set `SKIP_FAILED_RESULTS=1`, so an old pod never re-runs a failed conversion.
17. **Two workers re-downloaded the 397B checkpoint from HF Hub into the shared GCS cache** while other runs read it (13:26–13:38 UTC).
    - **Cause:** TM Step 1 reused the shared cache only if the scanned 1-slice `safetensors_dynamic` record had a cold Step 1 time. For 397B, that record is an infra failure with `hf_to_gcs_s=0`.
    - **What I did:** stopped both workers by replacing their task locks with a far-future heartbeat (worker `hold-coordinator`), which made their workers drop the tasks (LOST LOCK).
    - **Fix:** TM Step 1 now reuses the cache when all shards are present. The cold time recorded with it is the largest Step 1 time recorded for the model, or 0 if only warm checks (under 1 s) exist (397B: use 814.53 s from `results/p1_st_lance_b.csv`).
    - **Impact on other runs:** an interrupted upload never replaces an object, and the re-uploaded bytes are identical. The concurrent `hbmfix` load completed and failed later in HBM, for the reason in §7.19. The BEFORE 397B load (`origin/main`) overlapped the re-download and never left its load phase before the 1800 s timeout. With the PRs, that load takes 515.9 s, so the overlap can't explain the gap; the reports still mention it.
18. **More capacity loss during the scale-out:**
    - Kueue deactivated `g1` at 13:32 UTC (`The workload is deactivated`, same as `c1` in §7.14), so its daemon controller and slice 1 (`h1`) were deleted.
    - Spot node shutdowns killed `m3` and `s7` at 13:42 UTC. `m3`'s loss also failed the BEFORE 671B 4-slice run, which the queue released for retry (attempt 1/3). Its stored `OOMKilled` label comes from the local slice being killed after the remote slice died; it is not a real OOM.
    - Node `gke-tpu-fd9dfbd0-2jfc` stopped answering kubelet calls.
    - **Recovery:** reset the dead worker's lock heartbeat to 0, then recreate the pod from the scale-out manifest. A runner pod resolves its daemons' IPs once at start, so after a daemon is recreated, recreate the runner too (`m0`).
19. **`hbmfix` did not fix `qwen3.5-397b-a17b` scanned · 2.** The load finished (469.7 s), then the split of the stacked `(wi_0, wi_1)` composite failed at the same allocation (11.25 GiB needed, 2.77 GiB free): the composite (~22.5 GiB per device) is still alive while its first half is created. Next fix: split each layer's `gate_up_proj` before stacking, so only one target leaf is built at a time. Alternatively, move the composite to host before splitting. Meanwhile, `st_397b_scanned_s4` measures scanned 397B with 4 slices of HBM headroom.

---

## 8. Next steps (in order)

1. **Let the P1 queue drain.** Watch `queue/p1/done/` and the locks (§9). If a worker is evicted, reset its lock heartbeat to 0 and recreate the pod from `tpu7x_p1_completion_scaleout.yaml` or `tpu7x_p1_completion_scaleout2.yaml` (from `generate_tpu7x_manifests.py`). For an N-slice group, recreate the runner after any daemon. Finished tasks are never re-run, and interrupted ones resume from their last finished phase.
2. **Check the remaining results:**
   - unscanned 235B and 397B conversions, and their 1- and 2-slice Orbax loads (the only 1B comparisons possible for >200B on one host)
   - the BEFORE 1800 s re-runs (235B, 284B, 671B on 4 slices)
   - `st_397b_scanned_s4`
   - `p2_leftovers`
3. **Optional follow-up:** the leaf-level `(wi_0, wi_1)` split for `qwen3.5-397b-a17b` scanned · 2 (§7.19), as a separate patch and run.
4. **Review failed phases** with `summarize_tpu7x_results.py`. Re-queue any task whose phases failed for infrastructure reasons by deleting its `done/<id>.json`.
5. **Write the reports** (leave the v5p reports untouched):
   - `reports/tpu7x_large_model_200b_experiment_report.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_summary.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_report.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_issues.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_summary.csv`
6. **Clean up the cluster** (zero pods left):
   ```bash
   CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
   ./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --cleanup   # pods with app=mesakh-tpu7x-bench, all mesakh-w<k>-<id> services, mesakh-ctrl-c1
   kubectl --context=$CTX get pods,svc | grep mesakh                     # expect no output
   ```
7. **Commit and push** further changes to `docs/checkpoint-load-benchmark`. Keep `src/` equal to Lance's PRs; the HBM fix lives in `patches/hbmfix` and would go upstream as its own PR.

## 9. Command cheat sheet

```bash
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
kubectl --context=$CTX get pods -o wide | grep mesakh
kubectl --context=$CTX logs --since=15m mesakh-ctrl-g1 | grep -v "Constructed sharding" | tail -n 40
kubectl --context=$CTX logs --tail=40 mesakh-tpu7x-m0

# Queue progress: finished tasks, and running tasks with their worker
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/done/
gcloud storage cat 'gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/locks/*.json'   # {"worker": ..., "heartbeat": <epoch s>}
# A heartbeat older than ~2 min means the worker is gone; another worker takes the task over after 20 min.

# Result CSVs written by the runner scripts
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/results/

# Status matrix from the per-phase result JSONs (works while runs are in progress)
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/logs /tmp/tpu7x_results/p2 --exclude='.*\.log$'
python3 checkpoint_load_benchmark/scripts/summarize_tpu7x_results.py /tmp/tpu7x_results/p2
#   P1: .../p1_lance_prs/run_tpu7x_200b/logs and .../p1_origin_main/run_tpu7x_200b/logs

# Upload harness and bootstrap changes to GCS (pods fetch them at start; ctrl-c1 fetches them when its queue starts)
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --upload-only
```

## 10. Where things are

| What | Location |
| :--- | :--- |
| Harness | [`benchmark_ckpt_load_time.py`](benchmark_ckpt_load_time.py) (the pods run the copy in `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`) |
| Pod bootstrap / manifests / launcher | [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py), [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py), [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) |
| Code tarballs used by the bootstrap | `gs://mesa-maxtext/mesakh_tpu7x_bench/code/{maxtext_lance_prs,checkpoint_conversion_origin_main,site_packages_overlay}.tar.gz` |
| P1 WITH PRs (HF cache, Orbax checkpoints, logs) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/{safetensors_dynamic,to_maxtext,logs}/` |
| P1 BEFORE (`origin/main`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main/run_tpu7x_200b/logs/` (360 s watchdog, inconclusive) and `.../p1_origin_main/run_tpu7x_200b_t1800/logs/` (1800 s re-runs) |
| P1 WITH PRs + `hbmfix` | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_hbmfix/logs/` |
| P1 completion queue | `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/{tasks.json,locks/,done/,attempts/}` |
| P2 | `gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/{safetensors_dynamic,to_maxtext,logs}/` |
| Per-phase result JSON | `<run dir>/logs/<model>/<scanned\|unscanned>[_slices2\|_slices4]/<phase>_result.json` (+ `<phase>.log`, `<phase>_slice1.log`) |
| Result CSVs | `gs://mesa-maxtext/mesakh_tpu7x_bench/results/{p1_st_lance_*,p1_origin_main_*,p1_full_*,p2_full_*}.csv` |
| v5p reports (keep untouched) | [`reports/benchmark_ckpt_load_time_*`](reports/) |
| tpu7x reports (to write) | `reports/tpu7x_*` |
