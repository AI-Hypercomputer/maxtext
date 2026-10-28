# Handoff: Checkpoint Load-Time Benchmark on TPU v7 (`bodaborg-tpu7x-nap`)

> **Snapshot:** 2026-10-28 ~12:50 UTC · **Owner:** mesakh · **Branch:** `docs/checkpoint-load-benchmark` (pushed; pods run the copies in `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`)
>
> The numbers below come from the per-phase result JSONs in GCS at snapshot time. Pods were still running, so
> re-check §6 before acting. To refresh the status tables, run
> [`scripts/summarize_tpu7x_results.py`](scripts/summarize_tpu7x_results.py) (see §9).

---

## 0. TL;DR

| Workstream | Status |
| :--- | :--- |
| **P1 · Exp 1A WITH Lance's 3 PRs** (`safetensors_dynamic`) | `qwen3-235b-a22b` 4/4 ✅ · `deepseek4-284b` 4/4 ✅ (synthetic sparse files, see §7.10) · `qwen3-480b-a35b` 2/4 ✅ (1 slice can't hold it) · `qwen3.5-397b-a17b` 0/4 ❌ (HBM OOM in the scanned 2-slice transform, fix written but not deployed; unscanned · 2 not run yet) · `deepseek3-671b` 0/4 ❌ (exceeds 1- and 2-slice capacity) |
| **P1 · Exp 1A BEFORE** (`origin/main`) | No config succeeded. `qwen3-480b-a35b` hits `KeyError` (no mapping before #5706). Everything else was **killed by the watchdog**. That proves a failure only for `qwen3-235b-a22b` (killed at 900 s; with the PRs it takes 697–792 s). **The DeepSeek runs killed at 360 s prove nothing**, because with the PRs they take 353–453 s (§7.5). |
| **P1 · Exp 1B** (`to_maxtext` vs `safetensors_dynamic`) | Only `to_maxtext` Step 1 is measured so far (235B: 583.3 s, 480B: 1,135.5 s). Two conversion attempts for 235B scanned were lost: one to an eviction, and one that ran 2 h at leaf 1/15 because shards were streamed through a 3-shard LRU cache (§7.15). The harness now stages the full checkpoint when it fits on the host. **Scanned conversion of 397B/480B does not fit one tpu7x host**, so it now fails fast with a recorded reason; unscanned conversion streams per layer and works. |
| **P1 completion run** | Started 10:30 UTC as a GCS task queue (`configs/p1_completion_tasks.json`, 23 tasks; see README "Finishing Priority 1") drained by preemption-tolerant workers. At snapshot, worker `g1`+`h1` (2 slices, driven by `mesakh-ctrl-g1`) is the only one alive (pair `g`'s runner and `f1` were evicted within minutes of starting). Progress: `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/done/`. |
| **P2 · 16-model suite** | 10 of the 14 P2-only models are complete in all 4 configs. Gaps: `qwen3-32b` (1 cell), `qwen3-30b-a3b` (4 cells), `llama3.1-70b` (running), `gpt-oss-120b` (queued). `deepseek4-284b` and `deepseek3-671b` come from the P1 runs (`safetensors_dynamic` only). |
| **Not started** | The 5 `reports/tpu7x_*` files, unit tests for the new memory fix, commit/push, cluster cleanup |

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
| `benchmark_ckpt_load_time.py` | tpu7x support, zlib-compressed remote payload, env filtering that keeps `VBAR_CONTROL_SERVICE_URL`, SIGABRT teardown tolerance, phase watchdog with `TIMEOUT` labels, `--skip-safetensors`, `--tm-convert-only`, full staging for scanned `to_maxtext` (§7.15) |
| `bootstrap_tpu7x.py`, `generate_tpu7x_manifests.py`, `scripts/run_tpu7x_cluster.sh` | Pod bootstrap (trees: Lance's PRs, `origin/main`, `hbmfix`), manifests (P1/P2 pairs, P1-completion pair `g`, daemon controllers, recovery), launcher |
| `scripts/tpu7x_task_queue.py`, `scripts/tpu7x_daemon_controller.py`, `configs/p1_completion_tasks.json` | P1 completion queue (README "Finishing Priority 1") |
| `scripts/summarize_tpu7x_results.py` | Status matrix from per-phase result JSONs |
| `patches/hbmfix.patch`, `patches/hbmfix/src/...` | HBM-peak fix for composite keys (§7.6). Same 63 passing and 13 failing unit tests as HEAD in a local CPU venv (the 13 failures are from the old local JAX); `src/` itself stays equal to Lance's PRs |
| Not committed: `docs/plans/`, `run_e2e_matrix_xpk.{py,sh}`, `vllm_decoder_issue.md` | Unrelated to this work |

Code bundles verified against git: `code/maxtext_lance_prs.tar.gz` `src/` = this branch's `src/` (`ef8488fec`); `code/checkpoint_conversion_origin_main.tar.gz` = `f596f5369:src/maxtext/checkpoint_conversion`.

--- | :--- | :--- |
| `checkpoint_load_benchmark/benchmark_ckpt_load_time.py` | tpu7x support, zlib-compressed remote payload, env filtering that keeps `VBAR_CONTROL_SERVICE_URL`, SIGABRT teardown tolerance, phase watchdog | Synced to `gs://mesa-maxtext/mesakh_tpu7x_bench/code/` (what the pods run) |
| `README.md`, `configs/model_tiers.env` | tpu7x documentation | Docs |
| `bootstrap_tpu7x.py`, `generate_tpu7x_manifests.py`, `scripts/run_tpu7x_cluster.sh` | tpu7x pod bootstrap, manifest generator, launcher (new) | Used by the running pods |
| `scripts/summarize_tpu7x_results.py`, `handoff.md` | Status matrix tool and this doc (new) | Docs/tools |
| `src/maxtext/checkpoint_conversion/utils/{load_dynamic.py,tensor_handling.py}` | HBM-peak reductions for the `qwen3.5-397b-a17b` composite `(wi_0, wi_1)` path (§7.6) | ⚠️ **Compile-checked only. Not unit-tested and not deployed.** |
| `docs/plans/`, `run_e2e_matrix_xpk.{py,sh}`, `vllm_decoder_issue.md` | Unrelated to this work | **Do not commit with this work** |

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
| `deepseek3-671b` | 1,342.35 GB (staged sparse copy: 1,365.27 GB / 343) | ❌ exceeds HBM | ❌ host OOM-killed during the load; needs ≥4 slices |

---

## 4. How the runs are orchestrated

- **Pod pair `<id>`**: `mesakh-tpu7x-<id>0` (`ROLE=runner`, slice 0, runs `RUNNER_SCRIPT`), `mesakh-tpu7x-<id>1` (`ROLE=daemon`, slice 1, `--remote-slice-daemon --daemon-port 55555`), and the headless Service `mesakh-w1-<id>`.
  [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py) generates them and [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) deploys them.
  - P1 pairs: `a` = 235B, `b` = 397B, `c` = 480B, `f` = 284B + 671B. Each runs 3 stages: WITH PRs → BEFORE (`origin/main`, watchdog `PHASE_TIMEOUT_S=900`) → full TM vs ST comparison.
  - P2 pairs: `d` = 10 small and medium models, `e` = `qwen3-32b`, `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b`.
- **What [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py) does at pod start** (it is downloaded from `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`):
  1. Installs missing Python packages from `site_packages_overlay.tar.gz`, without shadowing the image's jax, flax, or orbax.
  2. Builds `/tmp/maxtext` from `maxtext_lance_prs.tar.gz` (the Lance-PRs tree), then overwrites `benchmark_ckpt_load_time.py` with the latest copy from GCS.
  3. Builds `/tmp/maxtext_origin_main` by copying that tree and overlaying `checkpoint_conversion_origin_main.tar.gz`. The BEFORE runs use this tree.
  4. **Patches the code by string replacement.** In both trees, `load_dynamic.py` gets single-GET HF→GCS streaming with 16 workers. In `/tmp/maxtext` only, `tensor_handling.py` and `load_dynamic.py` get the older HBM patch: free reshard/stack intermediates and split composite leaves before the reshard.
  > ⚠️ **The bootstrap does not use `gs://…/code/load_dynamic.py` or `tensor_handling.py`.** Uploading those files deploys nothing. To deploy the §7.6 fix, make the bootstrap download them into `/tmp/maxtext/src/maxtext/checkpoint_conversion/utils/` (and drop the old string patches), or rebuild `maxtext_lance_prs.tar.gz`.
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

### 5.1 P1 · Exp 1A: BEFORE (`origin/main`) vs WITH Lance's 3 PRs (`safetensors_dynamic`)

ST S2 = sharded load + in-memory conversion (the load and transform split is in parentheses). Cold HF→GCS Step 1 is measured once per model and listed under the model name.

| Model | Config | BEFORE (`origin/main`) | WITH PRs: ST S2 (load + transform) | Peak host RAM |
| :--- | :--- | :--- | :--- | ---: |
| `qwen3-235b-a22b`<br>Step 1: 583.3 s | scanned · 1 | ❌ killed by the 900 s watchdog during the load | **757.5 s** (674.8 + 82.7) | 391.8 GB |
| | unscanned · 1 | skipped (harness rule) | **697.1 s** (661.8 + 35.2) | 392.7 GB |
| | scanned · 2 | ❌ killed by the 900 s watchdog | **792.4 s** (708.0 + 83.6) | 255.9 GB |
| | unscanned · 2 | skipped | **749.0 s** (704.9 + 44.0) | 254.6 GB |
| `deepseek4-284b`<br>(synthetic sparse local files) | scanned · 1 | ❌ killed by the 360 s watchdog ⚠️ inconclusive | **453.3 s** (366.3 + 86.8) | 963.4 GB |
| | unscanned · 1 | skipped | **414.1 s** (377.0 + 37.0) | 980.3 GB |
| | scanned · 2 | ❌ killed by the 360 s watchdog ⚠️ inconclusive | **440.7 s** (352.9 + 87.6) | 822.6 GB |
| | unscanned · 2 | skipped | **353.0 s** (306.3 + 46.5) | 822.6 GB |
| `qwen3.5-397b-a17b`<br>Step 1: 814.5 s | scanned · 1 | ❌ killed by the watchdog | ❌ An earlier attempt finished the load (947.7 s), then hit TPU HBM OOM in the transform. The stored record is a later infra failure (TPU init, §7.1), so it needs a clean re-run. | – |
| | unscanned · 1 | skipped | not run | – |
| | scanned · 2 | ❌ killed by the watchdog | ❌ Load finished (483.8 s); the transform failed with `RESOURCE_EXHAUSTED: allocate 11.25G, 2.77G free` (§7.6) | 161.2 GB |
| | unscanned · 2 | skipped | **not run yet: expected to fit, do this next** (§8) | – |
| `qwen3-480b-a35b`<br>Step 1: 1,046.1 s | scanned · 1 | ❌ `KeyError: 'qwen3-480b-a35b'` | ❌ TPU HBM OOM (961 GB > 813.8 GB) | – |
| | unscanned · 1 | skipped | skipped | – |
| | scanned · 2 | ❌ `KeyError` | **1,067.9 s** (942.0 + 125.8) | 415.7 GB |
| | unscanned · 2 | skipped | **992.2 s** (907.7 + 84.3) | 415.5 GB |
| `deepseek3-671b`<br>(synthetic sparse local files) | scanned · 1 | ❌ killed by the 360 s watchdog | ❌ exceeds 1-slice HBM (1,342 GB > 813.8 GB) | – |
| | scanned · 2 | ❌ killed by the 360 s watchdog | ❌ host OOM-killed (exit 137) at a sampled peak of 742.1 GB | – |
| | unscanned · 1/2 | skipped | skipped | – |

### 5.2 P1 · Exp 1B: `to_maxtext` vs `safetensors_dynamic`

| Model | TM S1 HF→GCS | TM S2 convert (CPU) | TM S3 Orbax load | Notes |
| :--- | ---: | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 583.3 s | ❌ **killed by the eviction** (scanned, at the start of the Orbax save, ~09:01 UTC); no partial checkpoint in GCS | ⏳ | Relaunch queue item 3 from §4 (it writes `results/p1_full_a.csv`) |
| `qwen3-480b-a35b` | 1,135.5 s | ⏳ | ⏳ | S2 runs on one host with 1,014.5 GB RAM; untested at 961 GB |
| `deepseek4-284b`, `qwen3.5-397b-a17b` | ⏳ | ⏳ | ⏳ | Not scheduled yet |
| `deepseek3-671b` | – | – | – | Can't be loaded on 2 slices (§3), so there's nothing to compare |

The `results/p1_full_{a,b,c}.csv` files in GCS are currently **all zeros**: placeholders from pairs that were preempted before stage 3.

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

| Pod | Node / IP | Role | State / doing now |
| :--- | :--- | :--- | :--- |
| `mesakh-ctrl-g1` | CPU node (`cpu-np`) | Controller: runs one queue task per command on daemon `g1`, as worker `g1` with `--max-slices 2` | Running |
| `mesakh-tpu7x-g1` | `gke-tpu-1c328440-8976` / 10.224.9.7 | Slice 0 (daemon of pair `g`, whose runner `g0` was evicted at 10:32) | Running `st_397b_unscanned_s2` (`safetensors_dynamic`, 397B unscanned, 2 slices) since 12:48 UTC |
| `mesakh-tpu7x-h1` | `gke-tpu-687aaa3e-qk54` / 10.224.75.11 | Slice 1 daemon for `g1` (service `mesakh-w1-h`) | Running |

Gone since the previous snapshot:
- `c1`, `f1` and `g0`: evicted by Kueue cohort reclamation or spot node loss.
- P2 pair `e`: evicted. It finished most of `llama3.1-70b`, but scanned 2-slice TM S3 failed with exit 1, and `gpt-oss-120b` is only partly done. Both models were added to the queue task `p2_leftovers`.
- P2 pair `d`: deleted (misconfigured, §7.11). Its models are also in `p2_leftovers`.

Queue status at snapshot: `tm_235b_scanned_s1` was interrupted on purpose (lock replaced, §7.15) and will re-run with full staging. All other tasks are pending.

---

## 7. Known issues & findings

1. **tpu7x TPU init needs `VBAR_CONTROL_SERVICE_URL`.** GKE injects `VBAR_CONTROL_SERVICE_URL=unix:///var/run/tpu-plugin/vbar.sock`. Filtering out `*_SERVICE_*` env vars drops it, and TPU init then fails with `Failed to connect to [::]:8353`. *Fixed*: the harness now strips only `*_SERVICE_HOST`, `*_SERVICE_PORT*`, and `*_TCP*`. Two stored records predate the fix: `qwen3.5-397b-a17b` WITH scanned · 1 and an older `deepseek3-671b` `origin/main` scanned · 2. Re-run them.
2. **Linux `MAX_ARG_STRLEN` (131,072 bytes).** The harness ships its own source to the remote daemon as one argument, which grew past the limit (`[Errno 7] Argument list too long`). *Fixed*: zlib then base64, about 28 KB.
3. **SIGABRT (-6) at multi-slice interpreter teardown.** *Fixed*: the run is not marked failed when its result JSON is already complete.
4. **Watchdog kills are recorded as `OOMKilled / Exit Code 137`.** The harness labels every `-9` exit as OOM (`benchmark_ckpt_load_time.py` ~L2276/L2417), including SIGKILLs from `PHASE_TIMEOUT_S`. Every BEFORE `-9` above had a peak host RAM of ≤194 GB out of 1,014.5 GB, so they were timeouts, not OOMs. **TODO**: record `TIMEOUT (<N>s)` from the watchdog state, and reclassify existing records from the `[TIMEOUT_WATCHDOG]` lines in the run output.
5. **A watchdog shorter than the WITH-PRs time is inconclusive.** The DeepSeek BEFORE runs were killed at 360 s, but with the PRs they need 353–453 s. Likewise, `qwen3.5-397b-a17b` scanned · 1 took 947.7 s just to load with the PRs. **Re-run BEFORE with `PHASE_TIMEOUT_S=1800`** or more. For `qwen3-235b-a22b`, 900 s vs. 697–792 s is a fair comparison, but it only shows that `origin/main` is "> 900 s".
6. **`qwen3.5-397b-a17b` scanned · 2-slice HBM OOM in the composite `(wi_0, wi_1)` path.**
   - **What happens:** each layer's `mlp.experts.gate_up_proj` is 512 × 2048 × 4096 BF16, about 8.6 GB. 45 local layers are stacked into one composite target `(…, 2)` of about 387 GB, about 24.2 GB per device across 16 devices. It is then sliced into `wi_0` and `wi_1` (about 12.1 GB per device each, which is the failed `11.25G` allocation) while the rest of the model already fills HBM.
   - **What changed:** the uncommitted fix does three things:
     - Frees each consumed HF input right after its hook, but only tensors popped from `hf_state` on their last use. Tensors shared by several leaves are kept.
     - Frees stack, `astype`, and `moveaxis` intermediates as soon as their successor is ready.
     - Makes `_reshard(..., delete_input=)` opt-in, passed only where the input is a known intermediate.
   - **Caveat:** the composite split itself still briefly holds `loaded` plus both parts. If that is still too much, split at the leaf level instead: split each layer into `wi_0`/`wi_1` and stack the two separately.
   - **Fastest path to a 397B data point:** run unscanned on 2 slices. Per-layer composites are about 0.5 GB per device, and `qwen3-480b-a35b` unscanned · 2 (961 GB) already succeeded.
7. **`SKIP_FAILED_RESULTS` docs were inverted** (now fixed in the README and the harness docstring). In code, with `--skip-completed`, FAILED results are re-run by default, and `SKIP_FAILED_RESULTS=1` skips them. The P1 pair manifests and **pair `d`** set it to `1`.
8. **`qwen3-30b-a3b` unscanned TM S2 failed with `HFValidationError`.** The local staging dir `/tmp/p2_lance_prs/hf_local/qwen3-30b-a3b` was missing (an older harness staging path), so `to_maxtext` treated the path as a repo id. The current harness re-stages into `/ramdisk/hf_local/`, so a plain retry should work.
9. **`llama3.1-70b` unscanned TM S1 was killed (-9) at 21.6 GB RAM during the HF download** (07:45, an external kill). Pair `e` doesn't set `SKIP_FAILED_RESULTS`, so it will retry this automatically.
10. **The DeepSeek P1 numbers use `STAGE_SPARSE_DEEPSEEK_OOM=1`**: synthetic sparse BF16 safetensors on local disk, with no HF→GCS step. They measure load/transform mechanics and capacity, not GCS read bandwidth, so don't compare them directly with the Qwen GCS loads.
11. **Pair `d` is inconsistent with the other pairs:**
    - It uses a different image and runs the pushed branch (`git clone`) without the bootstrap patches.
    - It sets `SKIP_FAILED_RESULTS=1`, so it would **not** retry `qwen3-30b-a3b` unscanned.
    - It queues `gpt-oss-120b`, which pair `e` also runs. The two would race on the same GCS result paths.
12. **P2 outliers:** `qwen3-14b` scanned · 1 ST took 342.7 s (298.4 s in the transform, vs. 69.0 s on 2 slices). `qwen3-32b` scanned · 1 is 3–5× slower than unscanned in every phase. Re-run both before reporting.
13. **`run_tpu7x_cluster.sh --status` and `--cleanup` only match the label `app=mesakh-tpu7x-bench`** (plus the `mesakh-w1-*` services). They don't touch `mesakh-ctrl-c1` or `d0/d1`, which carry different labels.
14. **Kueue can evict running pods.** At ~09:01 UTC, `c1` was stopped after about 3 h with event `EvictedDueToDeactivated: The workload is deactivated`. Our workloads set no `maximumExecutionTimeSeconds`, so the cause isn't visible to us (likely a cluster-level deactivation or quota action). Assume any pod can disappear: keep queues short, rely on the per-phase result JSONs in GCS (`--skip-completed` resumes from them), and don't depend on a 2-pod pairing surviving.
    - Root cause confirmed later: the `default` ClusterQueue has **0 nominal tpu7x quota** and borrows all its chips from the cohort, so `priority-dev` reclaims them (`Preempted ... due to reclamation within the cohort`). Spot nodes also go NotReady. This is why the P1 completion run uses the GCS task queue.
15. **Scanned `to_maxtext` conversion thrashed when streaming shards.** For very large models, the harness used to stage only index/config files and stream shards from GCS through a 3-shard LRU cache. A scanned MoE leaf stacks each expert across all layers, so every leaf reads every shard. With a 3-shard cache, the checkpoint was re-downloaded once per expert: the 235B scanned conversion ran 2 h at leaf 1/15.
    - *Fixed*: the harness stages all shards when they fit within 55% of `/ramdisk` free space (235B: 470 GB). For scanned conversions that don't fit (397B: 795 GB, 480B: 961 GB on a 1,014.5 GB host whose boot disk is only 100 GB), it fails fast with the reason. Unscanned leaves read one layer at a time, so streaming stays for them.
    - Re-staging inside Step 2 is now recorded as `hf_to_local_restage_s`.

---

## 8. Next steps (in order)

1. **Let the P1 queue drain.** Watch `queue/p1/done/` and the controller logs (§9). If workers get evicted, relaunch them:
   `run_tpu7x_cluster.sh --priority p1-completion` re-uploads the code and deploys pair `g`, or apply `tpu7x_p1_completion_recovery.yaml` to drive a surviving daemon. Finished tasks are never re-run, and interrupted ones resume from their last finished phase.
2. **Check the `hbmfix` result** (`st_397b_scanned_s2_hbmfix`, run dir `run_tpu7x_200b_hbmfix`). If 397B scanned still OOMs on 2 slices, split composite keys at the leaf level (§7.6).
3. **Review interrupted or failed phases** with `summarize_tpu7x_results.py`. Any task whose phases failed for infrastructure reasons gets re-queued by deleting its `done/<id>.json`.
4. **Write the reports** (leave the v5p reports untouched):
   - `reports/tpu7x_large_model_200b_experiment_report.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_summary.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_report.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_issues.md`
   - `reports/tpu7x_benchmark_ckpt_load_time_summary.csv`
5. **Clean up the cluster** (zero pods left):
   ```bash
   CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
   ./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --cleanup   # app=mesakh-tpu7x-bench: pairs, daemons, controllers
   kubectl --context=$CTX delete pod mesakh-ctrl-c1 --ignore-not-found
   kubectl --context=$CTX get pods | grep mesakh                         # expect no output
   ```
6. **Commit and push** further changes to `docs/checkpoint-load-benchmark`. Keep `src/` equal to Lance's PRs; the HBM fix lives in `patches/hbmfix` and would go upstream as its own PR.

## 9. Command cheat sheet

```bash
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
kubectl --context=$CTX get pods -o wide | grep mesakh
kubectl --context=$CTX logs --since=15m mesakh-ctrl-c1 | grep -v "Constructed sharding" | tail -n 40
kubectl --context=$CTX logs --tail=40 mesakh-tpu7x-e0

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
| P1 BEFORE (`origin/main`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main/run_tpu7x_200b/logs/` |
| P2 | `gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/{safetensors_dynamic,to_maxtext,logs}/` |
| Per-phase result JSON | `<run dir>/logs/<model>/<scanned\|unscanned>[_slices2]/<phase>_result.json` (+ `<phase>.log`, `<phase>_slice1.log`) |
| Result CSVs | `gs://mesa-maxtext/mesakh_tpu7x_bench/results/{p1_st_lance_*,p1_origin_main_*,p1_full_*,p2_full_*}.csv` |
| v5p reports (keep untouched) | [`reports/benchmark_ckpt_load_time_*`](reports/) |
| tpu7x reports (to write) | `reports/tpu7x_*` |
