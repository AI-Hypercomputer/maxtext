---
name: maxtext-goodput
description: >-
  Run and monitor MaxText training jobs with ML Goodput Measurement
  (`ml-goodput-measurement`). Use when enabling Goodput and Badput recording or
  monitoring in MaxText, configuring cumulative or rolling-window Goodput flags,
  or inspecting Goodput outputs in training logs, Google Cloud Logs Explorer,
  Metrics Explorer, and Google Cloud Monitoring Dashboards.
---

# MaxText Goodput Measurement

Reference docs:
- [`ml-goodput-measurement` README](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/README.md)
- [MaxText Goodput Monitoring Guide](../../../docs/guides/monitoring_and_debugging/monitor_goodput.md)
- Implementation: [`src/maxtext/common/goodput.py`](../../../src/maxtext/common/goodput.py) and [`src/maxtext/configs/base.yml`](../../../src/maxtext/configs/base.yml)

---

## 1. Overview & Metrics

- **Goodput** is the percentage of a training job's wall-clock time spent on productive training steps.
- **Badput** is the remaining wall-clock time spent on non-training overhead (initialization, compilation, synchronous data loading, blocking checkpoint saves/restores, disruptions, and custom events).

In MaxText, Goodput measurement runs on the lead host (`jax.process_index() == 0`) in two parts:
1. **Record (`enable_goodput_recording=true`)**: `GoodputRecorder` writes start/end timestamps for job, step, and overhead events to Google Cloud Logging under `goodput_<run_name>`.
2. **Monitor (`monitor_goodput=true`)**: `GoodputMonitor` runs background uploaders that read Cloud Logging via `GoodputCalculator` and export cumulative and rolling-window metrics to **TensorBoard** and **Google Cloud Monitoring (GCM)**.

### Why Use Goodput Library Instead of a Vanilla Timer?

- **Survives crashes & preemptions**: Persists timestamps to Cloud Logging to measure cluster recovery downtime (`INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION`) and retroactively reclassify steps lost after a rollback (`WASTED_PROGRESS_FROM_DISRUPTION`).
- **Separates blocking vs. async I/O**: Counts only synchronous stalls (`DATA_LOADING_SYNC`, `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` via Orbax Cloud Logger) without penalizing overlapped prefetching or background checkpointing.
- **Isolates compilation & step anomalies**: Automatically separates first-step JAX/XLA compilation (`PROGRAM_STARTUP`) from steady-state steps and tracks `workload/step_time_deviation`.
- **Full wall-clock attribution & rolling windows**: Reconciles 100% of job lifetime into standardized `BadputType` buckets across both cumulative and sliding windows (1h, 1d, 3d, 5d).
- **Zero training-loop overhead**: Logs asynchronously from host 0 only and computes/exports metrics out-of-process to TensorBoard and Google Cloud Monitoring.

### Google Cloud Monitoring (GCM) Metrics

Full details: [Access Metrics on Google Cloud Monitoring](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/README.md#access-metrics-on-google-cloud-monitoring).

| Metric | GCM Metric (`compute.googleapis.com/...`) | PromQL Name (`compute_googleapis_com:...`) | Description |
| :--- | :--- | :--- | :--- |
| **Cumulative Goodput** | `workload/goodput_time` | `workload_goodput_time` | Cumulative productive training time (s), labeled by `goodput_source`. |
| **Cumulative Badput** | `workload/badput_time` | `workload_badput_time` | Cumulative overhead time (s), labeled by `badput_source`. |
| **Rolling Window Goodput** | `workload/interval_goodput` | `workload_interval_goodput` | Productive time over each rolling window (`window_size`). |
| **Rolling Window Badput** | `workload/interval_badput` | `workload_interval_badput` | Overhead time over each rolling window (`window_size`), labeled by `badput_source`. |
| **Total Elapsed Time** | `workload/total_elapsed_time` | `workload_total_elapsed_time` | Total wall-clock elapsed time since job start (s). |
| **Max Productive Steps** | `workload/max_productive_steps` | `workload_max_productive_steps` | Highest completed productive step count. |
| **Disruption Count** | `workload/disruptions` | `workload_disruptions` | Cumulative count of job disruptions. |
| **Step Time Deviation** | `workload/step_time_deviation` | `workload_step_time_deviation` | Step duration deviation from ideal step time. |
| **Ideal Step Time** | `workload/performance` | `workload_performance` | Configured or computed ideal step time (s). |

### Badput Breakdown Categories (`BadputType`)

Full details: [Retrieve Badput Breakdown](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/README.md#retrieve-badput-breakdown).

- `TPU_INITIALIZATION`: Device discovery, slice initialization, and driver/buffer setup.
- `TRAINING_PREP`: Checkpoint manager creation, mesh/model initialization, and optimizer setup.
- `PROGRAM_STARTUP`: JAX tracing, XLA compilation, and runtime startup.
- `DATA_LOADING_SYNC`: Blocking batch fetch time (`DATA_LOADING_ASYNC` tracks overlapped non-blocking loading without penalizing Goodput).
- `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` / `UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME`: Blocking checkpoint save/restore time (requires `enable_checkpoint_cloud_logger=true` with Orbax).
- `WASTED_PROGRESS_FROM_DISRUPTION` / `INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION`: Productive steps lost since the last checkpoint and infrastructure recovery time after a disruption.
- `CUSTOM_BADPUT_EVENTS` / `OTHER`: Blocking non-training events (e.g., eval, SDC checks) and uncategorized overhead.

---

## 2. Run Pre-Training with Goodput Enabled

### Prerequisites

- **Unique `run_name` per run**: Always use a fresh `run_name` (or `goodput_job_name`) for each new experiment so `goodput_<run_name>` does not mix logs from earlier runs into cumulative Goodput.
- **Cloud Logging access scope**: Nodes need Cloud Logging read/write access (`https://www.googleapis.com/auth/cloud-platform`).

### Sample Script

Follow the [MaxText Pre-Training on TPUs tutorial](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/run_maxtext/run_maxtext_localhost.html#training-on-tpus) (`llama3-8b`) and append the Goodput flags:

```bash
python3 -m maxtext.trainers.pre_train.train \
  model_name=llama3-8b \
  run_name=${YOUR_JOB_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=synthetic \
  steps=10 \
  enable_goodput_recording=true \
  monitor_goodput=true \
  enable_rolling_window_goodput=true \
  enable_checkpoint_cloud_logger=true
```


### Key Flags (`src/maxtext/configs/base.yml`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `enable_goodput_recording` | `false` | Record job, step, and overhead timestamps to Cloud Logging (`goodput_<run_name>`). |
| `monitor_goodput` | `false` | Start background `GoodputMonitor` uploader on host 0 (TensorBoard + GCM). |
| `enable_rolling_window_goodput` | `true` | Upload rolling-window Goodput/Badput to GCM when `monitor_goodput=true`. |
| `rolling_windows_seconds` | `[3600, 86400, 259200, 432000]` | Rolling window sizes in seconds (1h, 1d, 3d, 5d). |
| `enable_checkpoint_cloud_logger` | `false` | Enable Orbax structured Cloud Logger for checkpointing Badput breakdown. |
| `goodput_upload_interval_seconds` | `30` | Upload interval (s) for cumulative and rolling-window metrics. |
| `enable_pathways_goodput` | `false` | Set to `true` when running on Pathways. |

---

## 3. Where to Find the Output

### A. Training Log

Verify that the background uploaders started, published metrics, and flushed on exit:
- Startup: `Started Cumulative Goodput upload to Tensorboard & GCM in the background!` and `Started Rolling Window Goodput upload to Tensorboard & GCM in the background!`
- Periodic upload: `gcp_metrics.py ... 'Sent Goodput metrics to GCM Monitoring.'`
- Shutdown: `Flushed final metrics and safe exited from rolling window Goodput monitoring.` and `Flushed final metrics and safe exited from cumulative Goodput monitoring.`

### B. Google Cloud Logs Explorer

Open **Logging > Logs Explorer** (`https://console.cloud.google.com/logs/query`) and query `goodput_<RUN_NAME>`:

```sql
logName=("projects/<PROJECT_ID>/logs/goodput_<RUN_NAME>")
timestamp >= "2026-09-25T17:00:00-07:00"
AND timestamp <= "2026-09-25T19:00:00-07:00"
```

Each entry's `jsonPayload` contains the recorded event timestamps (`job_start_time`, `tpu_init_start_time`, `training_prep_start_time`, `data_loading_start_time`, `step_start_time`, `step_count`, `job_end_time`).

### C. Google Cloud Metrics Explorer

Open **Monitoring > Metrics Explorer** (`https://console.cloud.google.com/monitoring/metrics-explorer`), switch to **PromQL**, and filter by `workload_id="<RUN_NAME>"`:

```promql
label_replace(compute_googleapis_com:workload_goodput_time{workload_id="<RUN_NAME>"}, "m", "goodput_time", "", "")
or label_replace(compute_googleapis_com:workload_badput_time{workload_id="<RUN_NAME>"}, "m", "badput_time", "", "")
or label_replace(compute_googleapis_com:workload_total_elapsed_time{workload_id="<RUN_NAME>"}, "m", "total_elapsed_time", "", "")
or label_replace(compute_googleapis_com:workload_max_productive_steps{workload_id="<RUN_NAME>"}, "m", "max_productive_steps", "", "")
or label_replace(compute_googleapis_com:workload_disruptions{workload_id="<RUN_NAME>"}, "m", "disruptions", "", "")
or label_replace(compute_googleapis_com:workload_step_time_deviation{workload_id="<RUN_NAME>"}, "m", "step_time_deviation", "", "")
or label_replace(compute_googleapis_com:workload_performance{workload_id="<RUN_NAME>"}, "m", "performance", "", "")
or label_replace(compute_googleapis_com:workload_interval_goodput{workload_id="<RUN_NAME>"}, "m", "interval_goodput", "", "")
or label_replace(compute_googleapis_com:workload_interval_badput{workload_id="<RUN_NAME>"}, "m", "interval_badput", "", "")
```

### D. Google Cloud Monitoring Dashboard

1. In **Monitoring > Dashboards** (`https://console.cloud.google.com/monitoring/dashboards`), import [`ml_goodput_measurement/dashboards/goodput_dashboard.json`](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/ml_goodput_measurement/dashboards/goodput_dashboard.json).
2. Filter the dashboard by `workload_id: <RUN_NAME>` over the run's time range to view Cumulative and Rolling Window Goodput/Badput breakdown charts.
