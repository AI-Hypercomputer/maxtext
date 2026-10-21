# MaxText Input Pipeline GCS QPS Benchmark (`PyGrain` + `ArrayRecord`)

This benchmark measures and compares the Google Cloud Storage (GCS) HTTP QPS, byte read amplification, and step latency of MaxText's `PyGrain` + `ArrayRecord` input pipelines on CPU VMs (`JAX_PLATFORMS=cpu`) without requiring TPUs.

## What Is Measured (Zero Heuristics)

1. **Modalities**:
   - **`text` (`pretrain_preprocessing_pipeline`)**: Variable-length log-normal documents packed with `FirstFitPackIterDataset` into `max_target_length` sequences.
   - **`multimodal` (`vision_sft_preprocessing_pipeline`)**: High-resolution encoded RGB image payloads (`896x896` JPEG/PNG) + prompt/response pairs processed with `gemma3-4b` vision preprocessing.
2. **Storage Backends (Both Measured Directly from Real C++ Execution)**:
   - **`direct_gcs` (`gs://...`)**: Direct GCS reads via C++ `riegeli::gcs::GcsReader` and `google-cloud-cpp` against `InstrumentedGCSServer`. Every HTTP request is registered immediately upon arrival and deterministically classified by comparing its `start_offset` against the shard's exact `RiegeliPostscript` `footer_offset`:
     - `init_probe_shard_open`: `ArrayRecordDataSource.__init__()` opening all shards to probe `num_records()` and immediately closing them.
     - `lazy_worker_shard_open`: `_ensure_reader_exists()` lazily re-opening shard readers inside worker processes on first touch.
     - `record_read`: Actual user data chunk reads (`[0, footer_offset)`).
   - **`gcsfuse` (`/path/to/mount/...`)**: POSIX file descriptor reads via C++ `riegeli::FdReader`, traced live across the process tree (including `mp_prefetch` workers) using `PosixGCSFuseTracer` (`strace -f -qq -s 0 -ttt -y -e trace=pread64`) and replayed through a stateful GCSFuse streaming read-ahead (`--gcsfuse_read_ahead_kb`) and local file-cache state machine.
3. **Phase Separation (`init` $\rightarrow$ `first_batch` $\rightarrow$ `warmup` $\rightarrow$ `steady`)**:
   - Separates cold startup and lazy shard-discovery warmup (`--warmup_steps`) from measured steady-state steps (`--steps`), reporting both **True Steady-State Record-Read QPS** (`true_record_read_gets_per_step_per_host`) and **Early-Step QPS (including Lazy Shard Opens)** (`total_gets_per_step_per_host`).

## Quick Start (2,048-Host Scale Emulation on CPU VM)

```bash
cd /usr/local/google/home/mohitkhatwani/qps_optimizer/maxtext

JAX_PLATFORMS=cpu PYTHONPATH=.:src /usr/local/google/home/mohitkhatwani/qps_optimizer/.venv/bin/python \
  -m benchmarks.input_pipeline.benchmark_qps \
  --data_dir=/tmp/maxtext_qps_bench_data \
  --modalities=text,multimodal \
  --backends=direct_gcs,gcsfuse \
  --num_shards=64 \
  --text_records_per_shard=256 \
  --multimodal_records_per_shard=64 \
  --emulated_host_count=2048 \
  --concurrent_emulated_hosts=4 \
  --per_host_batch_size=4 \
  --warmup_steps=5 \
  --steps=10 \
  --grain_worker_count=4 \
  --output_json=/usr/local/google/home/mohitkhatwani/qps_optimizer/benchmark_2k_hosts_4workers_v2.json
```
