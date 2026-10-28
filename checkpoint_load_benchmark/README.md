# MaxText Checkpoint Load-Time Benchmark Suite (`to_maxtext` vs. `safetensors_dynamic`)

A self-contained, reproducible benchmark suite for comparing end-to-end and warm checkpoint load times between MaxText's offline Orbax conversion pipeline (**`to_maxtext`**) and on-the-fly distributed SafeTensors loading (**`safetensors_dynamic`**) across:
- **Scan Modes**: `scanned` (`scan_layers=True`) and `unscanned` (`scan_layers=False`)
- **Slice Topologies**: **Single-Slice (`num_slices=1`, 4 devices)** and **Multi-Slice (`num_slices=2+`, 8+ global devices)**
- **Model Scales (`0.6B` to `1600B`)**: 12 models across 4 tiers (48 total configurations)

---

## 1. Directory Structure

```text
checkpoint_load_benchmark/
├── README.md                                  # Complete documentation & reproduction guide
├── benchmark_ckpt_load_time.py                # Main benchmark harness, multi-slice coordinator & remote slice daemon
├── configs/
│   └── model_tiers.env                        # Model tier definitions (Tiers 0-3) & default runtime/cluster settings
├── scripts/
│   ├── common.sh                              # Shared environment setup, PYTHONPATH, HF token & interpreter resolution
│   ├── run_single_slice.sh                    # Run Single-Slice (num_slices=1) benchmarks
│   ├── run_multi_slice_local.sh               # Run Multi-Slice (num_slices=2+) benchmarks on a single host
│   ├── run_multi_slice_cluster.sh             # Deploy & run 1-slice and 2-slice jobs on GKE TPU cluster (bodaborg-v5p-nap)
│   └── run_full_matrix.sh                     # Reproduce the 48-configuration benchmark matrix by tier (0, 1, 2, 3, all)
├── k8s/
│   └── multislice_benchmark_job.yaml.tmpl     # Kubernetes Service + 2-Node Coordinator/Worker Job template for GKE TPU v5p-8
└── reports/
    ├── benchmark_ckpt_load_time_summary.md    # Executive 48-config summary (two-row side-by-side comparison table)
    ├── benchmark_ckpt_load_time_report.md     # Detailed 48-config engineering report with phase breakdowns
    ├── benchmark_ckpt_load_time_issues.md     # Root-cause analysis, cluster OOM logs, and upstream fixes
    └── benchmark_ckpt_load_time_summary.csv   # Raw 48-row benchmark metrics CSV
```

---

## 2. Benchmark Pipelines Compared

| Pipeline | Steps Measured | Description |
| :--- | :--- | :--- |
| **`to_maxtext`**<br>*(3-Step Offline Orbax)* | **Step 1**: `HF -> GCS` *(+ local stage)*<br>**Step 2**: `Convert to Orbax checkpoint`<br>**Step 3**: `Load the Orbax checkpoint` | Downloads HuggingFace `.safetensors`, converts weights offline on a single CPU host via `to_maxtext.py` into a GCS Orbax checkpoint, then restores the Orbax checkpoint into the MaxText mesh via `checkpointing.load_state_if_possible(source_checkpoint_layout="orbax")`. |
| **`safetensor`**<br>*(`safetensors_dynamic`, 2-Step Direct)* | **Step 1**: `HF -> GCS`<br>**Step 2**: `Safetensors & convert to MaxText`<br>*(sub-phases: `[1/3]` mappings, `[2/3]` sharded GCS load, `[3/3]` in-memory transform)* | Streams HuggingFace `.safetensors` shards to GCS `hf_cache`, reads byte-ranged shards directly into the distributed mesh across slices, and transforms/stacks parameters in memory via `checkpointing.load_state_if_possible(source_checkpoint_layout="safetensors_dynamic")`. |

---

## 3. How Single-Slice and Multi-Slice Execution Work

### 3.1 Single-Slice (`num_slices=1`)
Enabled with `--num-slices-list 1` (or [`scripts/run_single_slice.sh`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/scripts/run_single_slice.sh)):
- Runs a single worker process (`process_count=1`) with **4 devices** (`v5p-8` TPU slice or `XLA_FLAGS=--xla_force_host_platform_device_count=4` on CPU).
- Passes MaxText flags:
  ```text
  skip_jax_distributed_system=True
  num_slices=1
  ici_fsdp_parallelism=-1
  ```

### 3.2 Multi-Slice (`num_slices >= 2`)
Enabled with `--num-slices-list 2` or `--num-slices-list 1 2` (via [`scripts/run_multi_slice_local.sh`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/scripts/run_multi_slice_local.sh) or [`scripts/run_multi_slice_cluster.sh`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh)):
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
   - Automatically selects (or overrides via `--dcn-parallelism-axis {auto,data,fsdp}`) the DCN axis:
     - **Data Parallelism Across Slices (`dcn_data_parallelism=<N>`)**: Default for `<=47B` models (`qwen3-*`, `gemma2-2b`, `deepseek2-16b`, `mixtral-8x7b`) to benchmark multi-slice data-parallel replication.
     - **FSDP Sharding Across Slices (`dcn_fsdp_parallelism=<N>`)**: Default for `>=70B` models (`llama3.1-70b`, `gpt-oss-120b`, `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-284b`, `deepseek4-1600b`) so parameters are sharded across all `4 * N` devices across slices—allowing models like `gpt-oss-120b` (`233.66 GB` BF16) that OOM on 1 slice (`253 GB` RAM) to succeed on 2 slices (`133.86–206.09 GB` RAM/slice).
3. **Two Multi-Slice Execution Modes**:
   - **Mode A — Local Multi-Process Multi-Slice** ([`scripts/run_multi_slice_local.sh`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/scripts/run_multi_slice_local.sh)): Spawns `N` local slice worker processes (`slice_id=0..N-1`) coordinating over `127.0.0.1:<port>` with `process_is_granule=True` in `create_hybrid_device_mesh` (`4` devices/slice $\times$ `2` slices = `8` global devices).
   - **Mode B — Multi-Node GKE TPU Cluster Multi-Slice** ([`scripts/run_multi_slice_cluster.sh`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh)): Deploys 2 physical `v5p-8` nodes on GKE (`bodaborg-v5p-nap`). Node 1 runs `benchmark_ckpt_load_time.py --remote-slice-daemon --daemon-port 55555`, and Node 0 coordinates across `REMOTE_SLICE1_ADDR=<slice1_pod_ip>:55555`, giving `2 x 270 GB = 540 GB` aggregate host RAM and `2 x 368 GB = 736 GB` aggregate TPU HBM.

---

## 4. Prerequisites

1. **Branch**: Ensure you are on branch `docs/checkpoint-load-benchmark` (which includes cherry-picked fixes from `#5706`, `#5828`, and `#5705`):
   ```bash
   git checkout docs/checkpoint-load-benchmark
   ```
   - **PR #5706**: Fix `safetensors_dynamic` loading for `gpt-oss`, `Qwen3.5`, `Qwen3-Next`, and `Gemma` (composite keys, multi-host hooks, legacy keys)
   - **PR #5828**: Fix `safetensors_dynamic` load timeout on multi-hosts (leading-axis sharded reads, keep inner MoE stacks in read layout until final reshard)
   - **PR #5705**: Support `safetensors_dynamic` in `from_pretrained` and with checkpointing off
2. **Dependencies**: Ensure `orbax-checkpoint >= 0.12.7` is installed (required to avoid whole-file `.safetensors` RAM spikes on `>=70B` models):
   ```bash
   pip install --upgrade "orbax-checkpoint>=0.12.7"
   ```
3. **HuggingFace Token & GCS Access**:
   - Export `HF_TOKEN=<your_token>` or place it in `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/hf_token.txt`.
   - Ensure read/write access to the GCS benchmark bucket (default: `gs://mesa-maxtext/benchmark_ckpt_load_time`).

---

## 5. Quick Start & Reproduction Commands

### 5.1 Run a Single-Slice Benchmark (`num_slices=1`)
```bash
./checkpoint_load_benchmark/scripts/run_single_slice.sh \
  --models qwen3-0.6b gemma2-2b \
  --scan-modes scanned unscanned
```

### 5.2 Run a Multi-Slice Benchmark Locally (`num_slices=2` or `1 2`)
```bash
# Run 2-slice only:
./checkpoint_load_benchmark/scripts/run_multi_slice_local.sh \
  --models qwen3-0.6b gemma2-2b \
  --scan-modes scanned unscanned \
  --num-slices-list 2

# Run both 1-slice and 2-slice side-by-side:
./checkpoint_load_benchmark/scripts/run_multi_slice_local.sh \
  --models qwen3-0.6b gemma2-2b deepseek2-16b qwen3-30b-a3b qwen3-32b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2
```

### 5.3 Run Single-Slice & Multi-Slice on the GKE TPU Cluster (`bodaborg-v5p-nap`)
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

### 5.4 Reproduce the Full 48-Configuration Matrix by Tier
```bash
# Tier 0 (<=32B: qwen3-0.6b, gemma2-2b, deepseek2-16b, qwen3-30b-a3b, qwen3-32b):
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 0

# Tier 1 (47B-70B: mixtral-8x7b, llama3.1-70b) on GKE cluster:
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 1 --use-cluster

# Tier 2 (120B OOM boundary: gpt-oss-120b) on GKE cluster:
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 2 --use-cluster

# Tier 3 (284B-1600B OOM test: deepseek4-284b, deepseek3-671b, deepseek3.2-671b, deepseek4-1600b) on GKE cluster:
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 3 --use-cluster

# Regenerate the consolidated 48-row CSV table from an existing run ID without re-running completed steps:
./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier all --run-id 20261023_032224 --skip-completed
```

---

## 6. Direct Python CLI Reference

You can also invoke [`benchmark_ckpt_load_time.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/benchmark_ckpt_load_time.py) directly from any directory:

```bash
python3 checkpoint_load_benchmark/benchmark_ckpt_load_time.py \
  --models qwen3-0.6b llama3.1-70b gpt-oss-120b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2 \
  --dcn-parallelism-axis auto \
  --devices-per-slice 4 \
  --hardware auto \
  --weight-dtype bfloat16 \
  --gcs-base gs://mesa-maxtext/benchmark_ckpt_load_time \
  --run-id 20261023_032224 \
  --reuse-model-hf-cache \
  --skip-completed \
  --allow-step-failure \
  --csv-report checkpoint_load_benchmark/reports/benchmark_ckpt_load_time_summary.csv
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--models` | `qwen3-0.6b` | Space-separated list of MaxText model names to benchmark. |
| `--scan-modes` | `scanned unscanned` | Layer scan modes (`scanned` = `scan_layers=True`, `unscanned` = `scan_layers=False`). |
| `--num-slices-list` | `1` | Space-separated slice counts to test (e.g., `1` for single-slice, `2` for 2-slice, `1 2` for both). |
| `--dcn-parallelism-axis` | `auto` | Multi-slice DCN axis: `auto` (`fsdp` for `>=70B`, `data` for `<=47B`), `fsdp`, or `data`. |
| `--devices-per-slice` | `4` | Devices per slice when simulating CPU devices (`4` matches `v5p-8`). |
| `--remote-slice-addrs` | `None` *(or `$REMOTE_SLICE1_ADDR`)* | Optional `host:port` endpoint(s) of remote slice daemon(s) for multi-node cluster execution. |
| `--remote-slice-daemon` | `False` | Run as a TCP worker daemon listening on `--daemon-port` (default `55555`) on a companion slice node. |
| `--reuse-model-hf-cache` | `False` | Measure cold `HF -> GCS` once per model and reuse the populated GCS cache for subsequent runs. |
| `--skip-completed` | `False` | Reuse existing per-phase JSON results from `--local-work-dir` or GCS `logs/`. |
| `--allow-step-failure` | `False` | Record OOM (`Exit Code 137`) or step failures in JSON/CSV and continue remaining configurations. |

---

## 7. Benchmark Reports & Results (`reports/`)

- **[Team Sync, PR Cherry-Picks (`#5706`, `#5828`, `#5705`) & `>200B` Cluster Capacity Guide](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/team_sync_and_200b_experiment_plan.md)**
- **[Executive Summary (48 Configurations, Two-Row Side-by-Side Table)](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/benchmark_ckpt_load_time_summary.md)**
- **[Full Technical Report (Architecture, Sub-Phase Timings, Scaling & OOM Analysis)](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/benchmark_ckpt_load_time_report.md)**
- **[Issues, Cluster OOM Logs & Upstream Fixes (`F-1`–`F-9`, `U-1`–`U-4`)](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/benchmark_ckpt_load_time_issues.md)**
- **[Consolidated Raw Metrics CSV (48 Rows)](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/benchmark_ckpt_load_time_summary.csv)**
