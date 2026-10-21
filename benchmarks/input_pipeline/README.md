# MaxText Input Pipeline GCS QPS Benchmark (`PyGrain` + `ArrayRecord`)

This benchmark measures and compares the Google Cloud Storage (GCS) HTTP QPS,
byte read amplification, and step latency of MaxText's `PyGrain` + `ArrayRecord`
input pipelines on CPU VMs (`JAX_PLATFORMS=cpu`) without requiring TPUs.

## What Is Measured (Zero Heuristics)

1.  **Modalities**:
    *   **`text` (`pretrain_preprocessing_pipeline`)**: Variable-length
        log-normal documents packed with `FirstFitPackIterDataset` into
        `max_target_length` sequences.
    *   **`multimodal` (`vision_sft_preprocessing_pipeline`)**: High-resolution
        encoded RGB image payloads (`896x896` JPEG/PNG) + prompt/response pairs
        processed with `gemma3-4b` vision preprocessing.
2.  **Storage Backends (Both Measured Directly from Real C++ Execution)**:
    *   **`direct_gcs` (`gs://...`)**: Direct GCS reads via C++
        `riegeli::gcs::GcsReader` and `google-cloud-cpp` against
        `InstrumentedGCSServer`. Every HTTP request is registered immediately
        upon arrival and deterministically classified by comparing its
        `start_offset` against the shard's exact `RiegeliPostscript`
        `footer_offset`:
        *   `init_probe_shard_open`: `ArrayRecordDataSource.__init__()` opening
            all shards to probe `num_records()` and immediately closing them.
        *   `lazy_worker_shard_open`: `_ensure_reader_exists()` lazily
            re-opening shard readers inside worker processes on first touch.
        *   `record_read`: Actual user data chunk reads (`[0, footer_offset)`).
    *   **`gcsfuse` (`/path/to/mount/...`)**: POSIX file descriptor reads via
        C++ `riegeli::FdReader`, traced live across the process tree (including
        `mp_prefetch` workers) using `PosixGCSFuseTracer` (`strace -f -qq -s 0
        -ttt -y -e trace=pread64`) and replayed through a stateful GCSFuse
        streaming read-ahead (`--gcsfuse_read_ahead_kb`) and local file-cache
        state machine.
3.  **Phase Separation (`init` -> `first_batch` -> `warmup` -> `steady`) & Peak QPS**:
    *   Separates cold startup and lazy shard-discovery warmup
        (`--warmup_steps`) from measured steady-state steps (`--steps`),
        reporting:
        *   **True Steady-State Record-Read QPS**
            (`true_record_read_gets_per_step_per_host`)
        *   **Early-Step QPS (including Lazy Shard Opens)**
            (`total_gets_per_step_per_host`)
        *   **Peak Step QPS (`cluster_peak_step_qps_at_tpu_step_time`)**: Peak
            single-step GET burst across all active steps (`first_batch`,
            `warmup`, `steady`), capturing the prefetch-buffer fill spike
            controlled by `--prefetch_batch_size` and
            `--grain_prefetch_buffer_size`.
        *   **Peak 1-Second Sliding-Window QPS (`cluster_peak_1s_qps`)**: Exact
            maximum GET count within any 1.0-second sliding window across the
            entire run.

## Quick Start (2,048-Host Scale Emulation on CPU VM)

```bash
JAX_PLATFORMS=cpu python3 -m benchmarks.input_pipeline.benchmark_qps \
  --data_dir=/tmp/maxtext_qps_bench_data \
  --modalities=text,multimodal \
  --backends=direct_gcs,gcsfuse \
  --num_shards=64 \
  --text_records_per_shard=256 \
  --multimodal_records_per_shard=64 \
  --emulated_host_count=2048 \
  --concurrent_emulated_hosts=4 \
  --per_host_batch_size=4 \
  --prefetch_batch_size=2 \
  --warmup_steps=5 \
  --steps=10 \
  --grain_worker_count=4 \
  --output_json=/tmp/benchmark_2k_hosts_4workers.json
```
