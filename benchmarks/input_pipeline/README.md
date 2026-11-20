# MaxText input pipeline GCS QPS benchmark (PyGrain + ArrayRecord)

`benchmark_qps.py` is a standalone CPU benchmark for the real MaxText Grain
input pipeline: ArrayRecord global shuffle, SentencePiece tokenization,
`FirstFit` packing, and optional `mp_prefetch` workers. For each simulated host
it measures **actual** GCS requests, bytes, step time, and sample order. It
derives no QPS or byte count from formulas or per-record constants.

Use this tool to establish a stock GCS baseline or run A/B comparisons against
a custom Grain or ArrayRecord build.

## What it measures

| What | How |
|---|---|
| Multi-host | Each simulated host (`--host_counts=1,4,8`) is a separate spawned OS process with its own C++ GCS client, Grain worker pool, and trace files. Hosts start simultaneously from a barrier and, by default, step in lockstep. |
| GCS requests, live `gs://` (`--backend=gcs`) | `google-cloud-cpp` client tracing: `GOOGLE_CLOUD_CPP_ENABLE_CLOG=yes` and `CLOUD_STORAGE_ENABLE_TRACING=raw-client,http,rpc,rpc-streams`. Each process, including Grain workers, writes `host<H>.pid<P>.clog`. The parser counts one request per REST `ReadObject() << ReadObjectRangeRequest={...}` line, or per gRPC `ReadObject() << bucket: ... object: ...` line. Retries and resumes count as separate requests. It attaches HTTP status (`200`/`206`/`429`/`503`), received bytes, and cancellation from the curl trace, and gRPC status and `Cancel` from the RPC stream trace. |
| GCS requests, hermetic (`--backend=emulator`) | One `InstrumentedGCSServer` per host records every request served. `--emulator_client_trace` also parses client traces so they can be checked against the server (the e2e test asserts equal counts). |
| GCSFuse or local disk (`--backend=posix`) | `/proc/<pid>/io` for the host process tree, NIC RX, optional GCSFuse Prometheus `gcs_request_count` (`--gcsfuse_metrics_url`), and Cloud Monitoring. |
| Phases | Each host reports Unix-time boundaries for `init` (listing, record counting, iterator build), `first_batch`, `warmup`, and `steady` (per step). Requests are assigned to phases by timestamp. |
| QPS | `unique_examples_per_step` and `steady_gets_per_unique_example` (normalizes batch-size and sharding comparisons), per-phase GET counts, average QPS, **peak 1-second QPS** (sliding window), startup (`init` + `first_batch`) peak 1 s QPS, steady GETs per step, and outcome counts (`ok`, `throttled`, `unavailable`, `cancelled`). |
| Network and process | NIC RX/TX bytes from `/proc/net/dev` (loopback reported separately). Per-host process-tree RSS, open FDs, threads, and `rchar`/`syscr`. |
| Cloud Monitoring | `--cloud_monitoring_project=P` sums `storage.googleapis.com/api/request_count` for the bucket over the traced pass window, grouped by method and response code. Requires `google-cloud-monitoring`. The metric is minute-granular and delayed (`--cloud_monitoring_delay_s`). |
| Timing without tracing overhead | `--measure_mode=paired` (default) runs an **untraced timing pass** (step data-wait p50/p90/p99, `examples_per_s_per_host` over the steady wall clock, and `data_wait_examples_per_s_per_host` pipeline capacity) and a **traced QPS pass** with the same config. `timing` and `trace` run only one of them. |
| Sample-order parity | Each host hashes every yielded batch (keys, dtypes, shapes, bytes) with SHA-256. Runs with the same host count, data sharding, worker count (Grain packs per `mp_prefetch` worker), batch, sequence length, steps, and seed must match the first run in their group (`BASELINE`/`MATCH`/`MISMATCH`). |

## Environment and GCE VM setup (live GCS)

Run live `--backend=gcs` benchmarks on a GCE VM in the **same region and zone**
as your GCS bucket(s). Cross-region workstations add RTT latency, and corporate
proxies or context-aware mTLS can cause `google-cloud-cpp` requests to fail with
HTTP `401`.

### 1. Create a VM

```bash
export PROJECT=YOUR_GCP_PROJECT
export REGION=us-central1
export ZONE=us-central1-a
export VM=gcs-qps-bench

gcloud compute instances create "$VM" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=n2-standard-32 \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform
```

Ensure the VM's service account has `roles/storage.objectViewer` on the dataset
bucket(s) (and `roles/monitoring.viewer` on `$PROJECT` if using
`--cloud_monitoring_project`).

### 2. Install MaxText and Python dependencies

```bash
gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE"

sudo apt-get update && sudo apt-get install -y python3.12-venv git
git clone https://github.com/AI-Hypercomputer/maxtext.git ~/maxtext
cd ~/maxtext

python3.12 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install -e . \
  grain==0.2.19 \
  array-record==0.8.6 \
  sentencepiece \
  google-cloud-storage \
  google-cloud-monitoring

# Verify versions (PyPI array-record 0.8.6 includes google-cloud-cpp tracing hooks).
python3 -c "import array_record, grain, jax, sentencepiece; print(grain.__version__, jax.__version__, sentencepiece.__version__)"
```

Always raise the open file descriptor limit in your benchmark shell before
running multi-host or multi-worker benchmarks:

```bash
ulimit -n 65536
```

### 3. Hermetic smoke check (no GCS traffic)

Run the local emulator first to confirm multiprocessing, `mp_prefetch` workers,
client trace parsing, and SHA-256 parity all work in your environment:

```bash
cd ~/maxtext
source ~/venv/bin/activate
ulimit -n 65536

JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps \
  --backend=emulator \
  --host_counts=2 \
  --grain_worker_count=0,2 \
  --steps=5 \
  --emulator_client_trace \
  --output_json=/tmp/qps_emulator.json \
  --output_markdown=/tmp/qps_emulator.md
```

Verify that `client_trace` and `emulator` report identical `GETs total` for
every run and that `digests_identical_across_passes` is `true` in the JSON
output.

## Running baseline benchmarks (stock Grain + ArrayRecord)

The commands below run the unmodified MaxText Grain pipeline (`every_host`
startup counting, `per_host` data sharding, `prefetch_buffer_size=500`,
`grain_num_threads=16`, `FirstFit` packing to 4,096 tokens) against C4
`en/3.0.1` ArrayRecord shards (1,024 train shards, ~420 MB each,
356,075–356,076 records/shard).

```bash
cd ~/maxtext
source ~/venv/bin/activate
ulimit -n 65536

export OUT=~/bench_baseline
export BUCKET=gs://YOUR_DATASET_BUCKET
export C4_PREFIX=maxtext-dataset/array-record/c4/en/3.0.1
export FULL="$BUCKET/$C4_PREFIX/c4-train.array_record-*"
mkdir -p "$OUT"

QPS() {
  local name=$1; shift
  JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps \
    --backend=gcs \
    --per_host_batch_size=16 \
    --max_target_length=4096 \
    --grain_num_threads=16 \
    --measure_mode=paired \
    --trace_root="$OUT/traces_$name" \
    --output_json="$OUT/$name.json" \
    --output_markdown="$OUT/$name.md" \
    "$@" 2>&1 | tee "$OUT/$name.log"
}

# 1. Quick single-host smoke check (unpaced, W=0):
QPS baseline_h1_w0 --train_files="$FULL" --host_counts=1 \
  --grain_worker_count=0 --warmup_steps=1 --steps=3

# 2. Single-host with 4 Grain workers (0.4 s/step pacing):
QPS baseline_h1_w4 --train_files="$FULL" --host_counts=1 \
  --grain_worker_count=4 --warmup_steps=1 --steps=5 --target_step_time_s=0.4

# 3. Multi-host with 4 Grain workers per host (20 processes; fits in 128 GB):
QPS baseline_h4_w4 --train_files="$FULL" --host_counts=4 \
  --grain_worker_count=4 --warmup_steps=1 --steps=5 --target_step_time_s=0.4

# 4. 8 simulated hosts (W=0 fits in 128 GB; use 256 GB+ RAM for H=8, W>=4):
QPS baseline_h8_w0 --train_files="$FULL" --host_counts=8 \
  --grain_worker_count=0 --warmup_steps=1 --steps=5 --target_step_time_s=0.4
```

### Reference baseline numbers (stock `array-record 0.8.6`, `grain 0.2.19`, `jax 0.12.1`)

Measured on `n2-standard-32` (`us-central1-a`), C4 `en/3.0.1` (1,024 shards,
~420 MB/shard), stock HTTP/JSON client. With `array-record 0.8.6`, each shard
open issues **2 GETs** (C4's ~4–6 MiB footer exceeds the 1 MiB default
`footer_prefetch_bytes`), and most open-ended range reads are cancelled
mid-stream (~60–63% of total GETs in the short runs below; ~93% in longer
`W=16` runs).

| Config (1,024 shards) | Steps (pace) | Init s | 1st batch s | Step p50 / p99 s | Init GETs | Total GETs | Steady GETs/step | Startup peak 1s QPS | Peak 1s QPS | Bytes MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `H1 / W0` | 3 (unpaced) | 4.07 | 17.96 | 11.097 / 17.457 | 2,048 | 3,961 | 339.3 | 64 | 64 | 4,766 |
| `H1 / W4` | 5 (0.4 s) | 3.34 | 17.03 | 0.0001 / 12.225 | 2,048 | 6,304 | 287.8 | 89 | 97 | 7,120 |
| `H4 / W4` | 5 (0.4 s) | 11.20 | 23.35 | 0.0001 / 15.616 | 8,192 | 26,191 | 1,268.6 | 341 | 363 | 29,297 |
| `H8 / W0` | 5 (0.4 s) | 22.45 | 19.39 | 6.943 / 12.491 | 16,384 | 35,213 | 2,377.0 | 512 | 512 | 40,626 |

## A/B testing a custom Grain or ArrayRecord build

To compare a new Grain feature or ArrayRecord wheel against the stock baseline
in the same venv:

```bash
# 1. Run the baseline commands above (outputs saved under ~/bench_baseline/).
# 2. Install your candidate package without perturbing other pins:
pip install --no-deps --force-reinstall /path/to/grain-*.whl
# Or from a local checkout:
# pip install --no-deps -e /path/to/grain

# 3. Re-run the exact same commands into a new output directory:
export OUT=~/bench_candidate
mkdir -p "$OUT"
# ...re-run the same QPS commands...

# 4. Compare Markdown/JSON QPS, bytes, step-time percentiles, and `digests`
#    (matching per-host SHA-256 digests prove bitwise-identical sample order).
```

## Core CLI flags, sizing, and outputs

### Core flags

List-valued flags are expanded as a Cartesian product (comma `,` for numbers,
pipe `|` for strings).

| Flag | Default | Purpose |
|---|---|---|
| `--backend` | `emulator` | `gcs` (live `gs://`), `emulator` (local HTTP), or `posix` (GCSFuse / local disk). |
| `--train_files` | `""` | `gs://` or local glob (required for `gcs`/`posix`). |
| `--host_counts` | `1` | Simulated host counts, e.g. `1,4,8`. |
| `--grain_worker_count` | `0` | Grain `mp_prefetch` workers per host, e.g. `0,2,4,8,16`. |
| `--grain_prefetch_buffer_size` | `500` | Grain `ReadOptions.prefetch_buffer_size`. |
| `--grain_num_threads` | `16` | Grain `ReadOptions.num_threads`. |
| `--per_host_batch_size` | `4` | Packed examples loaded per host per step. |
| `--max_target_length` | `2048` | Sequence length for `FirstFit` packing. |
| `--warmup_steps` / `--steps` | `2` / `10` | Warmup and measured steady-state steps. |
| `--target_step_time_s` | `0.0` | Optional sleep to emulate accelerator step pacing (e.g. `0.4`). |
| `--measure_mode` | `paired` | `paired` (untraced timing + traced QPS), `timing`, or `trace`. |
| `--trace_root` / `--keep_traces` | `""` / off | Directory for per-process `.clog` traces (auto-cleaned unless `--keep_traces`). |
| `--cloud_monitoring_project` | `""` | Optional Cloud Monitoring `api/request_count` cross-check. |
| `--output_json` / `--output_markdown` | `""` | Paths to write full JSON and rendered Markdown reports. |

### Operational sizing and caveats

- **Memory (`H x W`):** Each `mp_prefetch` worker opens its own
  `ArrayRecordReader` handles (~2.7 GB RSS per worker on 1,024 C4 shards). On a
  128 GB VM (`n2-standard-32`), `H=4, W=4` (20 processes) and `H=8, W=0` (8
  processes) fit comfortably, whereas `H=8, W=4` (40 processes) OOMs. For `W=16`
  on 128 GB, use a 100-shard subset
  (`c4-train.array_record-000??-of-01024`) or a 256 GB+ VM.
- **Open FDs:** Run `ulimit -n 65536` in every shell running multi-host or
  multi-worker benchmarks.
- **Trace disk usage (`--trace_root`):** Traced passes write per-process
  `host<H>.pid<P>.clog` files (~2–6 GB per pass on large sweeps). Trace
  directories are automatically deleted after parsing unless `--keep_traces` is
  set; if keeping traces, point `--trace_root` at a disk with >=50 GB free.
- **Cloud Monitoring delay:** `--cloud_monitoring_project=PROJECT` sleeps
  `--cloud_monitoring_delay_s` (default `240s`) after each traced pass so
  minute-granular `storage.googleapis.com/api/request_count` metrics settle.
  Omit it during fast iteration.

### Outputs and unit tests

The JSON report (`--output_json`) contains, per run, the config, timing summary,
per-source request summaries (`emulator`, `client_trace`), NIC deltas, per-host
process stats and SHA-256 digests, and parity status. The Markdown output
(`--output_markdown`) renders the timing, QPS, and parity tables.

```bash
python3 -m pytest -q \
  benchmarks/input_pipeline/gcs_trace_test.py \
  benchmarks/input_pipeline/qps_runner_test.py \
  benchmarks/input_pipeline/benchmark_qps_cli_test.py \
  benchmarks/input_pipeline/benchmark_e2e_test.py
```

---

## Optional: GCS best-practice sweeps (BP-1 to BP-8)

> You can skip this section if you are only benchmarking the stock pipeline or
> comparing a custom Grain/ArrayRecord build against the baseline above.

In addition to stock baseline benchmarking, `benchmark_qps.py` includes sweep
flags for evaluating GCS QPS best-practice mitigations. Every mode preserves
Grain's global shuffle (the harness never partitions files per host and never
block-shuffles). All flags except `--grain_reader_options` (BP-2) and
`--modality=multimodal` work directly on MaxText `main` with PyPI
`sentencepiece`.

### Sweep axes

| BP | Flag | Values |
|---|---|---|
| BP-1 | `--startup_counting` | `every_host` (every host opens every shard to count records), `sharded` (each host counts 1/H of the shards, integer counts are exchanged, and Grain `FileInstruction`s are built), `manifest` (pre-indexed `{shard_path: num_records}` JSON via `--manifest_path`, so no shard opens). `--write_manifest=PATH --train_files=GLOB` creates a manifest. |
| BP-2 | `--grain_reader_options` | ArrayRecord reader options, e.g. `footer_prefetch_bytes:1M\|footer_prefetch_bytes:8M` (requires `get_datasets` reader-option pass-through). |
| BP-3 | `--startup_jitter_max_s` | Deterministic per-host startup delay in `[0, max)`, e.g. `0,10`. |
| BP-4 | `--grain_worker_count` | `0,2,4,8,16`. |
| BP-5 | `--grain_prefetch_buffer_size` | `16,32,128,500`. Compare startup peak 1 s QPS with steady step p99. |
| BP-7 | `--train_files` | `gs://standard/...\|gs://hns/...\|gs://zonal/...` (`--backend=gcs`) or a GCSFuse mount path (`--backend=posix`). Note: Zonal Rapid buckets do not support server-side rewrite (`HTTP 400`); seed them with `gcloud storage cp --daisy-chain`. |
| BP-8 | `--data_sharding`, `--hosts_per_replica`, `--per_replica_full_batch` | `per_host` (MaxText default: `dataloading_host_index = host`) vs. `per_replica` (all hosts in a replica read the same slice). By default each `per_replica` host loads the whole replica batch (`hosts_per_replica` x `--per_host_batch_size`), keeping the global batch of unique examples fixed while duplicating reads `hosts_per_replica`x. |

### Sweep commands

Using the `QPS` helper, `$FULL`, and `$OUT` from the baseline section above:

```bash
export SUB="$BUCKET/$C4_PREFIX/c4-train.array_record-000??-of-01024"  # 100 shards

# Write BP-1 manifests once:
JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps \
  --write_manifest="$OUT/c4_manifest.json" --train_files="$FULL"
JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps \
  --write_manifest="$OUT/c4_sub_manifest.json" --train_files="$SUB"

# BP-1: startup record counting (full 1,024 shards):
QPS bp1_h1 --train_files="$FULL" --host_counts=1 --startup_counting=every_host,sharded,manifest \
  --manifest_path="$OUT/c4_manifest.json" --grain_worker_count=4 --warmup_steps=1 --steps=5 --target_step_time_s=0.4
QPS bp1_h4 --train_files="$FULL" --host_counts=4 --startup_counting=every_host,sharded,manifest \
  --manifest_path="$OUT/c4_manifest.json" --grain_worker_count=4 --warmup_steps=1 --steps=5 --target_step_time_s=0.4
QPS bp1_h8 --train_files="$FULL" --host_counts=8 --startup_counting=every_host,sharded,manifest \
  --manifest_path="$OUT/c4_manifest.json" --grain_worker_count=0 --warmup_steps=1 --steps=5 --target_step_time_s=0.4

# BP-3: deterministic startup jitter (full 1,024 shards, H=8, W=0):
QPS bp3 --train_files="$FULL" --host_counts=8 --startup_counting=every_host \
  --startup_jitter_max_s=0,10 --grain_worker_count=0 --warmup_steps=1 --steps=5 --target_step_time_s=0.4

# BP-4: mp_prefetch worker sweep (100-shard subset, H=1, unpaced):
QPS bp4 --train_files="$SUB" --host_counts=1 --startup_counting=manifest \
  --manifest_path="$OUT/c4_sub_manifest.json" --grain_worker_count=0,2,4,8,16 --warmup_steps=2 --steps=12

# BP-5: prefetch_buffer_size sweep (100-shard subset, H=1, W=16, 0.4 s/step):
QPS bp5 --train_files="$SUB" --host_counts=1 --startup_counting=manifest \
  --manifest_path="$OUT/c4_sub_manifest.json" --grain_worker_count=16 \
  --grain_prefetch_buffer_size=16,32,128,500 --warmup_steps=3 --steps=20 --target_step_time_s=0.4

# BP-8: per-host vs. per-replica sharding (full 1,024 shards, H=8, R=8, W=0, 16 unique ex/step):
QPS bp8 --train_files="$FULL" --host_counts=8 --data_sharding=per_host,per_replica \
  --hosts_per_replica=8 --per_replica_full_batch=true --per_host_batch_size=2 \
  --startup_counting=manifest --manifest_path="$OUT/c4_manifest.json" \
  --grain_worker_count=0 --warmup_steps=1 --steps=5 --target_step_time_s=0.4

# BP-7: storage tiers (100-shard subset, H=1, W=16, pf=500, 0.4 s/step):
#   Run separately per bucket with its matching manifest (or pass --train_files="STD|HNS|RAPID" with every_host):
#   QPS bp7_hns --train_files="gs://YOUR_HNS_BUCKET/$C4_PREFIX/c4-train.array_record-000??-of-01024" \
#     --host_counts=1 --startup_counting=manifest --manifest_path="$OUT/hns_sub_manifest.json" \
#     --grain_worker_count=16 --warmup_steps=3 --steps=20 --target_step_time_s=0.4
#   For GCSFuse (--backend=posix), mount the bucket and run:
#   JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps --backend=posix \
#     --train_files="/mnt/gcsfuse/$C4_PREFIX/c4-train.array_record-000??-of-01024" \
#     --per_host_batch_size=16 --max_target_length=4096 --grain_num_threads=16 \
#     --host_counts=1 --startup_counting=every_host --grain_worker_count=16 --warmup_steps=3 --steps=20

# BP-2 (optional; requires get_datasets reader-option pass-through):
#   QPS bp2 --train_files="$FULL" --host_counts=1 --startup_counting=manifest \
#     --manifest_path="$OUT/c4_manifest.json" \
#     --grain_reader_options='footer_prefetch_bytes:1M|footer_prefetch_bytes:8M' \
#     --grain_worker_count=4 --warmup_steps=2 --steps=10 --target_step_time_s=0.4
```

### Reference sweep results (`array-record 0.8.6`, `grain 0.2.19`, `jax 0.12.1`)

All order-preserving comparisons below reported `MATCH` on per-host SHA-256
digests.

#### BP-1 — Startup record counting (1,024 shards)

| Hosts / W | Counting | Init s | Init GETs | GETs total | Bytes MiB | Startup peak 1s QPS | Parity |
|---|---|---|---|---|---|---|---|
| H1 / W4 | `every_host` | 3.34 | 2,048 | 6,304 | 7,120 | 89 | BASELINE |
| H1 / W4 | `sharded` | 9.66 | 2,048 | 6,288 | 7,102 | 91 | MATCH |
| H1 / W4 | `manifest` | 0.17 | 0 | 4,287 | 4,137 | 92 | MATCH |
| H4 / W4 | `every_host` | 11.20 | 8,192 | 26,191 | 29,297 | 341 | BASELINE |
| H4 / W4 | `sharded` | 3.35 | 2,048 | 20,427 | 20,582 | 342 | MATCH |
| H4 / W4 | `manifest` | 0.19 | 0 | 18,424 | 17,601 | 343 | MATCH |
| H8 / W0 | `every_host` | 22.45 | 16,384 | 35,213 | 40,626 | 512 | BASELINE |
| H8 / W0 | `sharded` | 3.24 | 2,048 | 20,877 | 19,507 | 310 | MATCH |
| H8 / W0 | `manifest` | 0.19 | 0 | 18,845 | 16,501 | 181 | MATCH |

#### BP-3 — Startup jitter (1,024 shards, H8, W0, `every_host`)

| Jitter | Init s | Init GETs | GETs total | Startup peak 1s QPS | Parity |
|---|---|---|---|---|---|
| 0 s | 22.91 | 16,384 | 35,213 | 512 | BASELINE |
| 0–10 s | 16.77 | 16,384 | 35,620 | 277 (−46%) | MATCH |

#### BP-4 — `mp_prefetch` workers (100-shard subset, H1, unpaced, 12 steps)

| W | 1st batch s | Step p50 / p99 s | Examples/s (wall) | GETs total | `first_batch` GETs | Startup peak 1s QPS | Peak 1s QPS |
|---|---|---|---|---|---|---|---|
| 0 | 11.09 | 0.433 / 0.593 | 37.1 | 2,357 | 388 | 70 | 241 |
| 2 | 11.81 | 0.092 / 0.634 | 73.9 | 3,333 | 670 | 114 | 445 |
| 4 | 10.85 | 0.0001 / 1.351 | 64.6 | 5,402 | 1,303 | 197 | 736 |
| 8 | 11.18 | 0.0001 / 1.696 | 78.7 | 10,659 | 2,994 | 406 | 1,247 |
| 16 | 11.97 | 0.0001 / 0.060 | (prefetch-served) | 19,835 | 5,372 | 721 | 1,597 |

#### BP-5 — `prefetch_buffer_size` (100-shard subset, H1, W16, 0.4 s/step)

| Buffer | 1st batch s | Step p50 / p99 s | GETs total | Steady GETs/step | Startup peak 1s QPS | Peak 1s QPS | Parity |
|---|---|---|---|---|---|---|---|
| 16 | 11.93 | 0.0001 / 0.0002 | 15,824 | 484.4 | 629 | 1,503 | BASELINE |
| 32 | 11.56 | 0.0001 / 0.0002 | 16,080 | 491.0 | 640 | 1,492 | MATCH |
| 128 | 11.79 | 0.0001 / 0.0002 | 17,616 | 537.8 | 768 | 1,545 | MATCH |
| 500 | 11.70 | 0.0001 / 0.0001 | 23,192 | 585.2 | 785 | 1,611 | MATCH |

#### BP-7 — Storage tiers (100-shard subset, H1, W16, pf=500, 0.4 s/step)

| Tier | 1st batch s | Step p90 / p99 s | GETs total | Startup peak 1s QPS | Peak 1s QPS | Cancelled | Digest |
|---|---|---|---|---|---|---|---|
| Standard regional | 11.70 | 0.0001 / 0.0001 | 23,192 | 785 | 1,611 | 21,592 | `3a41b255d9cc` |
| Regional + HNS | 12.81 | 0.0001 / 0.0002 | 23,206 | 773 | 1,611 | 21,606 | `3a41b255d9cc` |
| Zonal Rapid | 44.09 | 0.0002 / 0.0002 | 23,208 | 795 | 1,615 | 21,608 | `3a41b255d9cc` |
| GCSFuse 3.13.1 | 5.78 | 0.107 / 1.017 | n/a | n/a | n/a | n/a | `3a41b255d9cc` |

#### BP-8 — Per-host vs. per-replica sharding (1,024 shards, H8, R8, W0, 16 unique ex/step)

| Sharding | 1st batch s | Step p50 / p99 s | GETs total | Steady GETs / unique ex | Bytes MiB | Peak 1s QPS | Distinct streams |
|---|---|---|---|---|---|---|---|
| `per_host` | 3.24 | 0.83 / 2.16 | 3,413 | 26.2 | 3,420 | 185 | 8 |
| `per_replica` | 17.59 | 8.31 / 15.08 | 19,586 | 155.4 | 16,983 | 398 | 1 (x8 identical) |
