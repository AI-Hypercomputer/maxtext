# MaxText Checkpoint Load-Time Benchmark Suite (`to_maxtext` vs. `safetensors_dynamic`)

A self-contained, reproducible benchmark suite for comparing end-to-end and warm checkpoint load times, sub-phase breakdowns, and peak Host RAM / TPU HBM utilization between MaxText's offline Orbax conversion pipeline (**`to_maxtext`**) and on-the-fly distributed SafeTensors loading (**`safetensors_dynamic`**) across:
- **Scan Modes**: `scanned` (`scan_layers=True`) and `unscanned` (`scan_layers=False`)
- **Slice Topologies**: **Single-Slice (`num_slices=1`)** and **Multi-Slice (`num_slices=2+`)**
- **Hardware Generations**:
  - **TPU v5p (`bodaborg-v5p-nap`, `ct5p-hightpu-4t`, `2x2x1`)**: `4` TPU devices/slice (`368 GB` HBM, `~270 GB` Host RAM per slice; `8` devices / `736 GB` HBM / `~540 GB` Host RAM on 2 slices)
  - **TPU v7 / Ironwood (`bodaborg-tpu7x-nap`, `tpu7x-standard-4t`, `2x2x1`)**: `8` JAX TPU cores (`4` dual-core chips) per slice (`813.80 GB` HBM, `1,014.50 GB` Host RAM per slice; `16` TPU cores / `1,627.60 GB` HBM / `2,029.00 GB` Host RAM on 2 slices)
- **Model Scales (`0.6B` to `1600B`)**:
  - **Priority 1 (`>200B` Models on `bodaborg-tpu7x-nap`)**: `qwen3-235b-a22b` (235B), `deepseek4-284b` (284B), `qwen3.5-397b-a17b` (397B), `qwen3-480b-a35b` (480B), `deepseek3-671b` (671B)
  - **Priority 2 (`16-Model` Suite on `bodaborg-tpu7x-nap`)**: `qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`, `llama3.1-8b`, `llama3.1-70b`, `gemma2-2b`, `gemma3-4b`, `deepseek2-16b`, `gpt-oss-20b`, `qwen3-30b-a3b`, `mixtral-8x7b`, `gpt-oss-120b`, `deepseek4-284b`, `deepseek3-671b`
  - **Original `v5p` 12-Model Tier Matrix (`bodaborg-v5p-nap`)**: 48 configurations across Tiers 0–3 (`0.6B` to `1600B`)

---

## 1. Directory & Script Structure

```text
checkpoint_load_benchmark/
├── README.md                                        # Complete documentation & reproduction guide
├── handoff.md                                       # Current progress, live cluster state, known issues & next steps
├── benchmark_ckpt_load_time.py                      # Unified benchmark harness, multi-slice coordinator & remote slice daemon
├── bootstrap_tpu7x.py                               # Pod bootstrap: code trees (Lance's PRs, origin/main, hbmfix) & single-GET HF->GCS streamer
├── generate_tpu7x_manifests.py                      # Kubernetes manifest generator for bodaborg-tpu7x-nap (P1, P2, P1-completion workers)
├── configs/
│   ├── model_tiers.env                              # Model tier definitions (v5p Tiers 0-3 + tpu7x Priority 1 & Priority 2) & cluster defaults
│   └── p1_completion_tasks.json                     # Task list of the Priority 1 completion queue (one harness invocation per task)
├── patches/
│   ├── hbmfix.patch                                 # HBM-peak fix for composite keys (qwen3.5 wi_0/wi_1), as a diff against src/
│   └── hbmfix/src/maxtext/checkpoint_conversion/utils/  # The patched load_dynamic.py & tensor_handling.py (staged as /tmp/maxtext_hbmfix)
├── scripts/
│   ├── common.sh                                    # Shared environment setup, PYTHONPATH, HF token & Python interpreter resolution
│   ├── run_single_slice.sh                          # Run Single-Slice (num_slices=1) benchmarks
│   ├── run_multi_slice_local.sh                     # Run Multi-Slice (num_slices=2+) benchmarks on a single host
│   ├── run_multi_slice_cluster.sh                   # Deploy & run 1-slice and 2-slice jobs on GKE TPU v5p (bodaborg-v5p-nap)
│   ├── run_tpu7x_cluster.sh                         # Deploy & run Priority 1 (>200B) & Priority 2 (16 models) on GKE TPU v7 (bodaborg-tpu7x-nap)
│   ├── tpu7x_task_queue.py                          # Preemption-tolerant GCS task queue worker (claims tasks via atomic GCS locks)
│   ├── tpu7x_daemon_controller.py                   # CPU-pod controller that runs queue tasks on a daemon pod whose runner was evicted
│   ├── summarize_tpu7x_results.py                   # Markdown status matrix from per-phase result JSONs (works on partial runs)
│   ├── build_tpu7x_tables.py                        # tpu7x summary CSV + row-by-row to_maxtext vs safetensor table (merges re-runs)
│   └── run_full_matrix.sh                           # Reproduce the 48-configuration v5p benchmark matrix by tier (0, 1, 2, 3, all)
├── k8s/
│   └── multislice_benchmark_job.yaml.tmpl           # Kubernetes Service + 2-Node Coordinator/Worker Job template for GKE TPU v5p-8
└── reports/
    ├── team_sync_and_200b_experiment_plan.md        # Team sync, Lance Wang's 3 PRs (#5706, #5828, #5705) & >200B cluster capacity guide
    ├── benchmark_ckpt_load_time_summary.md          # [v5p] Executive 48-config summary (two-row side-by-side comparison table)
    ├── benchmark_ckpt_load_time_report.md           # [v5p] Detailed 48-config engineering report with phase breakdowns
    ├── benchmark_ckpt_load_time_issues.md           # [v5p] Root-cause analysis, cluster OOM logs, and upstream fixes
    ├── benchmark_ckpt_load_time_summary.csv         # [v5p] Raw 48-row benchmark metrics CSV
    ├── tpu7x_large_model_200b_experiment_report.md  # [tpu7x] Priority 1 (>200B models: Before vs. After Lance's PRs + to_maxtext vs. safetensors_dynamic)
    ├── tpu7x_benchmark_ckpt_load_time_summary.md    # [tpu7x] Executive summary on bodaborg-tpu7x-nap (two-row side-by-side comparison table)
    ├── tpu7x_benchmark_ckpt_load_time_report.md     # [tpu7x] Detailed engineering report on bodaborg-tpu7x-nap with phase breakdowns
    ├── tpu7x_benchmark_ckpt_load_time_issues.md     # [tpu7x] HBM/RAM capacity boundaries, multi-host timeout analysis, and fixes on tpu7x
    └── tpu7x_benchmark_ckpt_load_time_summary.csv   # [tpu7x] Raw benchmark metrics CSV on bodaborg-tpu7x-nap
```

---

## 2. Benchmark Pipelines & Experiments Compared

### 2.1 Checkpoint Loading Pipelines (`to_maxtext` vs. `safetensors_dynamic`)

| Pipeline | Steps Measured | Description |
| :--- | :--- | :--- |
| **`to_maxtext`**<br>*(3-Step Offline Orbax)* | **Step 1**: `HF -> GCS` *(+ local stage)*<br>**Step 2**: `Convert to Orbax checkpoint`<br>**Step 3**: `Load the Orbax checkpoint` | Downloads HuggingFace `.safetensors`, converts weights offline on a single CPU host via `to_maxtext.py` into a GCS Orbax checkpoint, then restores the Orbax checkpoint into the MaxText mesh via `checkpointing.load_state_if_possible(source_checkpoint_layout="orbax")`. |
| **`safetensor`**<br>*(`safetensors_dynamic`, 2-Step Direct)* | **Step 1**: `HF -> GCS`<br>**Step 2**: `Safetensors & convert to MaxText`<br>*(sub-phases: `[1/3]` mappings, `[2/3]` sharded GCS load, `[3/3]` in-memory transform)* | Streams HuggingFace `.safetensors` shards to GCS `hf_cache`, reads byte-ranged shards directly into the distributed mesh across slices, and transforms/stacks parameters in memory via `checkpointing.load_state_if_possible(source_checkpoint_layout="safetensors_dynamic")`. |

### 2.2 Priority 1 (`>200B`) Experiment Structure on `bodaborg-tpu7x-nap`
Each Priority 1 pod pair executes a 3-stage pipeline defined in [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py):
1. **Step 1 — `safetensors_dynamic` WITH Lance's 3 PRs (`#5706 + #5828 + #5705`)**: Runs `scanned` and `unscanned` across `num_slices=1` (`8` TPU cores) and `num_slices=2` (`16` TPU cores) using `/tmp/maxtext` (`origin/docs/checkpoint-load-benchmark`).
2. **Step 2 — `safetensors_dynamic` BEFORE Lance's 3 PRs (`origin/main`)**: Runs the same `>200B` models against `/tmp/maxtext_origin_main` (`f596f5369`) reusing the populated GCS `hf_cache` (`SHARED_HF_CACHE_BASE_DIR`) with a per-phase watchdog (`PHASE_TIMEOUT_S`: the original P1 pairs were deployed with `360`, the generator now uses `900`, and the completion queue re-runs BEFORE at `1800`) to capture `origin/main` failures (`KeyError` for models without a mapping, e.g. `qwen3-480b-a35b`; slow full-file multi-host reads that exceed the watchdog). A watchdog limit below the WITH-PRs load time is inconclusive (see [`handoff.md`](handoff.md)).
3. **Step 3 — Full `to_maxtext` vs. `safetensors_dynamic` Comparison**: Runs `to_maxtext` Steps 1, 2, and 3 alongside the completed `safetensors_dynamic` runs (`--skip-completed`) and writes consolidated CSV/JSON results to `gs://mesa-maxtext/mesakh_tpu7x_bench/results/`.

---

## 3. Hardware Specifications & Slice Topologies

| Spec / Dimension | `bodaborg-v5p-nap` (TPU v5p) | `bodaborg-tpu7x-nap` (TPU v7 / Ironwood) |
| :--- | :--- | :--- |
| **GKE Cluster Context** | `gke_cloud-tpu-shared-capacity_europe-west4_bodaborg-v5p-nap` | `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` |
| **GCE Machine Type** | `ct5p-hightpu-4t` (`v5p-8`) | `tpu7x-standard-4t` (`tpu7x`) |
| **Slice Topology (`cloud.google.com/gke-tpu-topology`)** | `2x2x1` (`4` chips / `4` JAX devices per slice) | `2x2x1` (`4` physical chips / `8` JAX TPU cores per slice) |
| **1 Slice (`num_slices=1`) TPU HBM** | **`368.00 GB`** (`4 × 92 GB`) | **`813.80 GB`** (`8 cores × 101.725 GB/core`) |
| **1 Slice (`num_slices=1`) Host CPU RAM** | **`~270.00 GB`** | **`1,014.50 GB`** (`944.83 GiB`, `224` vCPUs) |
| **2 Slices (`num_slices=2`) TPU HBM** | **`736.00 GB`** (`8` TPU devices across 2 hosts) | **`1,627.60 GB`** (`16` JAX TPU cores across 2 hosts) |
| **2 Slices (`num_slices=2`) Host CPU RAM** | **`~540.00 GB`** (`2 × 270 GB`) | **`2,029.00 GB`** (`1,889.66 GiB`, `2 × 1,014.50 GB`) |

### 3.1 Single-Slice (`num_slices=1`) Execution
- Runs a single worker process (`process_count=1`) with `4` devices on `v5p-8` (`--devices-per-slice 4`) or `8` TPU cores on `tpu7x` (`--devices-per-slice 8`).
- Passes MaxText flags: `skip_jax_distributed_system=True`, `num_slices=1`, `ici_fsdp_parallelism=-1`.

### 3.2 Multi-Slice (`num_slices=2`) Execution
1. **JAX Multi-Controller Distributed Initialization**:
   Every slice worker (`slice_id = 0 .. num_slices - 1`) initializes the JAX distributed runtime before creating the hybrid mesh:
   ```python
   jax.distributed.initialize(
       coordinator_address=args.coordinator_address,
       num_processes=args.num_slices,
       process_id=args.slice_id,
   )
   ```
2. **MaxText Hybrid Mesh & DCN Parallelism Flags**:
   - Passes `skip_jax_distributed_system=False`, `num_slices=<N>`, and `ici_fsdp_parallelism=-1`.
   - Automatically selects (or overrides via `--dcn-parallelism-axis {auto,data,fsdp}`) the DCN axis via [`resolve_dcn_axis`](benchmark_ckpt_load_time.py#L74-L94):
     - **Data Parallelism Across Slices (`dcn_data_parallelism=<N>`)**: Default for `<=47B` models (`qwen3-0.6b..32b`, `llama3.1-8b`, `gemma2-2b`, `gemma3-4b`, `deepseek2-16b`, `gpt-oss-20b`, `qwen3-30b-a3b`, `mixtral-8x7b`).
     - **FSDP Sharding Across Slices (`dcn_fsdp_parallelism=<N>`)**: Default for `>=70B` models (`llama3.1-70b`, `gpt-oss-120b`, `qwen3-235b-a22b`, `deepseek4-284b`, `qwen3.5-397b-a17b`, `qwen3-480b-a35b`, `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-1600b`) so parameters are sharded across all global devices (`16` TPU cores on 2 `tpu7x` slices = `(1, 1, 1, 16, ...)` mesh).

---

## 4. Prerequisites

1. **Branch**: Ensure you are on branch `docs/checkpoint-load-benchmark` (or `fix/safetensors-dynamic-lance-prs` for pure code changes), which includes cherry-picked fixes from Lance Wang's 3 PRs:
   ```bash
   git checkout docs/checkpoint-load-benchmark
   ```
   - **PR #5706**: Fix `safetensors_dynamic` loading for `gpt-oss`, `Qwen3.5`, `Qwen3-Next`, and `Gemma` (composite keys, multi-host hooks, legacy keys)
   - **PR #5828**: Fix `safetensors_dynamic` load timeout on multi-hosts (leading-axis sharded reads, keep inner MoE stacks in read layout until final reshard)
   - **PR #5705**: Support `safetensors_dynamic` in `from_pretrained` and with checkpointing off
2. **Dependencies**: Ensure `orbax-checkpoint >= 0.12.7` is installed:
   ```bash
   pip install --upgrade "orbax-checkpoint>=0.12.7"
   ```
3. **HuggingFace Token & GCS Access**:
   - Export `HF_TOKEN=<your_token>` or place it in `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/hf_token.txt`.
   - Ensure read/write access to `gs://mesa-maxtext/benchmark_ckpt_load_time` (`v5p`) and `gs://mesa-maxtext/mesakh_tpu7x_bench` (`tpu7x`).

---

## 5. Quick Start & Reproduction Commands

### 5.1 Run Priority 1 (`>200B`) & Priority 2 (`16 Models`) on `bodaborg-tpu7x-nap` (`tpu7x`)
Use [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) to sync [`benchmark_ckpt_load_time.py`](benchmark_ckpt_load_time.py) and [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py) to GCS, generate the Kubernetes manifests via [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py), and deploy the 2-slice pod pairs:

```bash
# Deploy both Priority 1 (>200B models: Pairs a, b, c, f) and Priority 2 (16-model suite: Pairs d, e):
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --priority all

# Deploy only Priority 1 (>200B models: 235B, 284B, 397B, 480B, 671B):
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --priority p1

# Deploy only Priority 2 (16-model benchmark suite):
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --priority p2

# Deploy or restart a single 2-slice pod pair (e.g. Pair b for qwen3.5-397b-a17b):
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --pair b

# Check live pod status on bodaborg-tpu7x-nap:
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --status

# Clean up all pods and services on bodaborg-tpu7x-nap after completion:
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --cleanup
```

#### Finishing Priority 1 with the GCS task queue (`--priority p1-completion`)
Spot preemptions and Kueue reclamation evict pods mid-run (our queue has no nominal TPU quota and borrows everything),
so the remaining Priority 1 work runs as a queue that any number of workers drain together:
- **Tasks**: [`configs/p1_completion_tasks.json`](configs/p1_completion_tasks.json), uploaded to
  `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/tasks.json`. Each task is one harness invocation with its code tree
  (`/tmp/maxtext` = Lance's PRs, `/tmp/maxtext_origin_main`, `/tmp/maxtext_hbmfix`), env and slice count.
- **Workers**: [`scripts/tpu7x_task_queue.py`](scripts/tpu7x_task_queue.py) claims a task with an atomic GCS lock
  (`if_generation_match=0`), refreshes a heartbeat, and writes `queue/p1/done/<id>.json`. A lock without heartbeat for
  20 min is taken over, and `--skip-completed` resumes the task from its last finished phase. Interrupted runs (non-zero
  exit, or slice-1 daemon gone) are released for retry (3 attempts).
- **Deploy**: `./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --priority p1-completion` uploads the code and
  task list and deploys pair `g`. When a runner pod is evicted, its surviving daemon keeps working through
  [`scripts/tpu7x_daemon_controller.py`](scripts/tpu7x_daemon_controller.py) (a CPU pod sending one task per daemon
  command, as each command is capped at 4 h); `generate_tpu7x_manifests.py` writes such recovery manifests
  (`tpu7x_p1_completion_recovery.yaml`: daemon `h1` + controller `ctrl-g1` driving `g1`).
- **Progress**: `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/{locks,done,attempts}/` and
  [`scripts/summarize_tpu7x_results.py`](scripts/summarize_tpu7x_results.py).
- **Scale-out** (`tpu7x_p1_completion_scaleout.yaml`, written by `generate_tpu7x_manifests.py`): more workers on the
  same queue when spot `2x2x1` capacity is idle. It adds 2-slice pairs `i`, `j`, single-slice workers `s1`–`s3` (for
  the multi-hour `to_maxtext` CPU conversions, so no slice-1 node idles), and a **4-slice group `m`** (runner `m0` +
  daemons `m1`–`m3`, `--min-slices 4`) for `deepseek3-671b`. That model's 1.34 TB BF16 needs 32 TPU cores / 3.26 TB HBM.
  A worker uses as many slices as it has reachable daemons (`--max-slices`, `--min-slices`). A second wave
  (`tpu7x_p1_completion_scaleout2.yaml`: pair `k`, singles `s4`–`s8`) starts the remaining single-slice tasks at once.
  Workers exit when no task they can run is left, which frees their nodes.
- **Code variants**: runs labeled `hbmfix` (`--run-id tpu7x_200b_hbmfix`) use [`patches/hbmfix`](patches/hbmfix.patch).
  The other WITH-PRs runs use `src/` from this branch (Lance's 3 PRs) **plus the string patches applied by
  [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py)**: single-GET HF→GCS streaming (Step 1 only), and, for the Step 2 HBM
  peak, freeing reshard/stack intermediates and splitting composite leaves before the reshard. BEFORE runs
  (`origin/main` `checkpoint_conversion`) get only the streaming patch.

### 5.2 Run a Single-Slice Benchmark (`num_slices=1`)
```bash
./checkpoint_load_benchmark/scripts/run_single_slice.sh \
  --models qwen3-0.6b gemma2-2b \
  --scan-modes scanned unscanned
```

### 5.3 Run a Multi-Slice Benchmark Locally (`num_slices=2` or `1 2`)
```bash
./checkpoint_load_benchmark/scripts/run_multi_slice_local.sh \
  --models qwen3-0.6b gemma2-2b deepseek2-16b qwen3-30b-a3b qwen3-32b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2
```

### 5.4 Run Single-Slice & Multi-Slice on the GKE TPU v5p Cluster (`bodaborg-v5p-nap`)
```bash
# Tier 1 (47B-70B) & Tier 2 (120B OOM boundary) across 1 and 2 slices on 2 GKE v5p-8 nodes:
./checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh \
  --models mixtral-8x7b llama3.1-70b gpt-oss-120b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2

# Tier 3 (DeepSeek 284B-1600B OOM test suite) across 1 and 2 slices on 2 GKE v5p-8 nodes:
./checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh \
  --models deepseek4-284b deepseek3-671b deepseek3.2-671b deepseek4-1600b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2 \
  --stage-sparse-oom
```

### 5.5 Reproduce the 48-Configuration `v5p` Matrix by Tier
```bash
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 0
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 1 --use-cluster
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 2 --use-cluster
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 3 --use-cluster
```

---

## 6. Direct Python CLI & Environment Variable Reference

You can invoke [`benchmark_ckpt_load_time.py`](benchmark_ckpt_load_time.py) directly from any directory:

```bash
python3 checkpoint_load_benchmark/benchmark_ckpt_load_time.py \
  --models qwen3-235b-a22b deepseek4-284b qwen3.5-397b-a17b qwen3-480b-a35b deepseek3-671b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2 \
  --dcn-parallelism-axis auto \
  --devices-per-slice 8 \
  --hardware tpu \
  --weight-dtype bfloat16 \
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \
  --run-id tpu7x_200b \
  --reuse-model-hf-cache \
  --skip-completed \
  --allow-step-failure \
  --csv-report checkpoint_load_benchmark/reports/tpu7x_benchmark_ckpt_load_time_summary.csv
```

### 6.1 CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--models` | `qwen3-0.6b` | Space-separated list of MaxText model names to benchmark. |
| `--scan-modes` | `scanned unscanned` | Layer scan modes (`scanned` = `scan_layers=True`, `unscanned` = `scan_layers=False`). |
| `--num-slices-list` | `1` | Space-separated slice counts to test (`1` for single-slice, `2` for 2-slice, `1 2` for both). |
| `--dcn-parallelism-axis` | `auto` | Multi-slice DCN axis: `auto` (`fsdp` for `>=70B`, `data` for `<=47B`), `fsdp`, or `data`. |
| `--devices-per-slice` | `4` | Devices per slice (`4` for `v5p-8`, `8` for `tpu7x`). |
| `--hardware` | `auto` | Target backend: `auto`, `tpu`, or `cpu`. |
| `--remote-slice-addrs` | `None` *(or `$REMOTE_SLICE1_ADDR`)* | Optional `host:port` endpoint(s) of remote slice daemon(s) for multi-node cluster execution. |
| `--remote-slice-daemon` | `False` | Run as a TCP worker daemon listening on `--daemon-port` (default `55555`) on a companion slice node. |
| `--reuse-model-hf-cache` | `False` | Measure cold `HF -> GCS` once per model and reuse the populated GCS cache for subsequent runs. |
| `--skip-to-maxtext` | `False` | Run only the `safetensors_dynamic` pipeline (used in Priority 1A before/after PR comparison). |
| `--skip-safetensors` | `False` | Run only the `to_maxtext` pipeline (e.g. to add `to_maxtext` to configurations whose `safetensors_dynamic` runs are done). |
| `--tm-convert-only` | `False` | Run only `to_maxtext` Steps 1–2 (HF→GCS + CPU conversion), skipping Step 3 and `safetensors_dynamic`: converts on one slice a checkpoint that only fits in HBM on two. |
| `--skip-completed` | `False` | Reuse existing per-phase JSON results from `--local-work-dir` or GCS `logs/`. |
| `--allow-step-failure` | `False` | Record OOM (`Exit Code 137` / `pjrt.OOM`) or step failures in JSON/CSV and continue remaining configurations. |

### 6.2 Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `KEEP_HF_CACHE` | `0` | Set to `1` to keep the shared GCS `hf_cache` after `safetensors_dynamic` finishes so subsequent runs or branches can reuse it. |
| `SKIP_FAILED_RESULTS` | `0` | With `--skip-completed`, cached results with `status == "FAILED"` are **re-run by default**. Set to `1` to also treat FAILED results as done and skip them (e.g. to not retry known OOMs). |
| `SHARED_HF_CACHE_BASE_DIR` | `""` | Override the GCS `safetensors_dynamic` base path to reuse an already-populated `hf_cache` (e.g., when benchmarking `origin/main` against `p1_lance_prs`). |
| `REUSE_SHARED_ST_GCS_CACHE` | `0` | Set to `1` so `to_maxtext` Step 1 stages local `.safetensors` files directly from the `safetensors_dynamic` GCS `hf_cache` instead of re-downloading from HuggingFace Hub. |
| `PHASE_TIMEOUT_S` | `0` *(no timeout)* | Maximum wall-clock seconds per worker phase. The watchdog SIGKILLs the worker, so the result JSON currently records `exit code -9 (OOMKilled / Exit Code 137)`; the `[TIMEOUT_WATCHDOG]` line in the run output is what tells a timeout apart from a real host OOM. Set it above the expected load time, or the result is inconclusive. |
| `STAGE_SPARSE_DEEPSEEK_OOM` | `0` | Set to `1` to synthesize sparse BF16 `.safetensors` files on local disk for ultra-large DeepSeek models (`284B`, `671B`, `1600B`) when testing memory/HBM capacity boundaries. |

---

## 7. Benchmark Reports & Results (`reports/`)

### 7.1 TPU v7 (`bodaborg-tpu7x-nap`, `tpu7x-standard-4t`) Reports
- **[Priority 1 (`>200B`) Progress: Every Test and Its Status](reports/tpu7x_priority1_progress.md)** *(live tracker, updated with the completion queue)*
- **[Priority 1 (`>200B` Models): Before vs. After Lance's 3 PRs & `to_maxtext` vs. `safetensors_dynamic`](reports/tpu7x_large_model_200b_experiment_report.md)**
- **[`tpu7x` Executive Summary (Two-Row Side-by-Side Comparison Table)](reports/tpu7x_benchmark_ckpt_load_time_summary.md)**
- **[`tpu7x` Full Technical Report (Sub-Phase Timings, Scaling & HBM/RAM Analysis)](reports/tpu7x_benchmark_ckpt_load_time_report.md)**
- **[`tpu7x` Issues, Capacity Boundaries & Multi-Host Fixes](reports/tpu7x_benchmark_ckpt_load_time_issues.md)**
- **[`tpu7x` Consolidated Raw Metrics CSV](reports/tpu7x_benchmark_ckpt_load_time_summary.csv)**

### 7.2 TPU v5p (`bodaborg-v5p-nap`, `ct5p-hightpu-4t`) Reports
- **[Team Sync, PR Cherry-Picks (`#5706`, `#5828`, `#5705`) & `>200B` Cluster Capacity Guide](reports/team_sync_and_200b_experiment_plan.md)**
- **[`v5p` Executive Summary (48 Configurations, Two-Row Side-by-Side Table)](reports/benchmark_ckpt_load_time_summary.md)**
- **[`v5p` Full Technical Report (Architecture, Sub-Phase Timings, Scaling & OOM Analysis)](reports/benchmark_ckpt_load_time_report.md)**
- **[`v5p` Issues, Cluster OOM Logs & Upstream Fixes (`F-1`–`F-9`, `U-1`–`U-4`)](reports/benchmark_ckpt_load_time_issues.md)**
- **[`v5p` Consolidated Raw Metrics CSV (48 Rows)](reports/benchmark_ckpt_load_time_summary.csv)**
