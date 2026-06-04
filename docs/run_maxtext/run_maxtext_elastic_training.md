<!--
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

(run-elastic-training)=

# Elastic training with Pathways

This guide shows how to run **elastic training** on a multi-slice TPU cluster: training that survives a slice failure *in-process*, without restarting the job. You launch a Qwen3 0.6B run across several TPU slices with Pathways, lose a slice mid-run, and watch training recover from the last checkpoint on the same controller.

```{important}
This guide is a **demonstration of the elastic training mechanism**, not a production recipe. It uses a small model (Qwen3 0.6B) and synthetic data so you can see recovery happen on a short run, then tear everything down. The exact slice counts, timeouts, and checkpoint cadence here are illustrative; tune them for your own model and hardware. Treat it as a starting point to understand the feature, not a configuration to copy verbatim into a long-running job.
```

## What is elastic training?

Large model training runs across many TPU slices. When one slice fails (a hardware fault, a preemption, a network blip), the default outcome is that the whole job crashes and restarts from scratch, losing the XLA compilation time plus everything since the last checkpoint.

Elastic training keeps the training process alive instead. Three components make that possible:

- **Pathways** orchestrates training across the slices. Its Resource Manager detects when a slice goes down and reports it to the training process.
- **MaxText** wraps the training loop with `elastic_retry`. When Pathways reports a failure, it catches the exception *inside the same Python process*, cleans up, and restarts training without exiting.
- **Orbax** handles checkpointing. Each checkpoint writes to GCS and creates a `commit_success` marker only after all data is flushed, so a checkpoint interrupted mid-write has no marker and is safely discarded on recovery.

Because the controller process never exits, the expensive XLA recompile is skipped and recovery is fast.

```{note}
This demo shows recovery via *checkpoint restore* on a fixed mesh: when a slice is lost, Pathways waits for a replacement, then all slices restore from the last committed checkpoint. It does **not** show elastic *degradation* (continuing on fewer slices at reduced throughput), which requires dynamic mesh resize and is not covered here.
```

## 1. Prerequisites

This guide assumes you already have a **Pathways-enabled GKE cluster** created with `xpk`, and a MaxText Docker image in your Artifact Registry. If you don't:

1. **Install XPK and create a Pathways GKE cluster.** Follow [Running MaxText with XPK](run_maxtext_via_xpk.md) and the [Pathways & XPK cluster guide](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster#xpk). Cluster creation and management is out of scope for this page.
2. **Build and upload the MaxText Docker image.** See [Build MaxText](../build_maxtext.md).

```{note}
If you installed `xpk` inside a Python virtual environment (`venv`), reactivate it (e.g., `source <VENV_NAME>/bin/activate`) in any new terminal before running `xpk` commands, or you will hit a `Command xpk not found` error.
```

## 2. Environment configuration

Set these environment variables in your shell. Replace the placeholders with your own values.

```bash
# -- Google Cloud Configuration --
export PROJECT_ID=<GCP project ID>
export ZONE=<GCP location>        # e.g., 'us-central1'
export GKE_CLUSTER=<cluster name> # your Pathways-enabled cluster

# -- Workload Configuration --
# Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores/periods).
export RUN_NAME="elastic-qwen3-$(date +%Y%m%d-%H%M%S)"

# TPU type and slice count. For supported types see src/maxtext/utils/accelerator_to_spec_map.py.
export TPU_TYPE="v5litepod-16"  # one slice = 16 v5e chips
export NUM_SLICES=3             # total slices in the run

# -- MaxText & Storage Configuration --
export BASE_OUTPUT_DIRECTORY=<gcs bucket path>  # e.g., gs://my-bucket/maxtext-runs
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/<your maxtext image>"
```

## 3. Launch the elastic workload

Submit the run with `xpk workload create-pathways`. Two sets of flags make it elastic:

- **On the `xpk` side**, `--elastic-slices` tells Pathways how many slices the workload is allowed to lose and keep going, and `--max-slice-restarts` caps how many times a slice's workers may be restarted.
- **On the MaxText side** (inside `--command`), `elastic_enabled=true` turns on the `elastic_retry` wrapper, and `enable_single_controller=True` runs training through Pathways. `checkpoint_period` is kept small so a recovery rewinds only a little.

```bash
xpk workload create-pathways \
  --workload=${RUN_NAME?} \
  --cluster=${GKE_CLUSTER?} \
  --project=${PROJECT_ID?} \
  --zone=${ZONE?} \
  --tpu-type=${TPU_TYPE?} \
  --num-slices=${NUM_SLICES?} \
  --docker-image=${DOCKER_IMAGE?} \
  --elastic-slices=1 \
  --max-slice-restarts=10 \
  --command="python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=${RUN_NAME?} \
    model_name=qwen3-0.6b \
    dataset_type=synthetic \
    per_device_batch_size=1 \
    max_target_length=2048 \
    attention=flash \
    remat_policy=full \
    steps=5000 \
    enable_checkpointing=true \
    checkpoint_period=100 \
    enable_single_controller=True \
    elastic_enabled=true \
    elastic_timeout_seconds=300 \
    elastic_max_retries=10"
```

```{note}
`--elastic-slices=1` means the run tolerates losing **one** slice at a time out of `${NUM_SLICES}`. Keep `--max-slice-restarts` and the MaxText `elastic_max_retries` consistent with how many failures you want to ride out.
```

```{warning}
**Do not enable profiling in an elastic run.** An elastic event (a slice going down and recovering) in the middle of a profile is not supported, so this example leaves the profiler off (`profiler` is unset). Profile a separate, non-elastic run if you need performance traces.
```

### Watch training start

List the workload and follow its logs through the Cloud Console (**Kubernetes Engine → Workloads →** your run **→ Logs**), or:

```bash
xpk workload list --cluster=${GKE_CLUSTER?} --project=${PROJECT_ID?} --zone=${ZONE?}
```

After XLA compilation (a couple of minutes) you should see elastic training enabled and a steady stream of steps:

```
Elastic utils: Elastic training enabled.
Elastic Retry Enabled
completed step: 8, seconds: 0.159, TFLOP/s/device: 43.430, loss: 220.774
completed step: 9, seconds: 0.166, TFLOP/s/device: 41.524, loss: 217.296
```

Let it run until the step counter passes the first checkpoint (here, step ~130, so `checkpoint_period=100` has committed once) before you inject a failure, so there is a complete checkpoint to recover from.

## 4. Simulate a slice failure

To see recovery, remove a worker on one slice. Connect to the cluster and delete a worker pod immediately (`--grace-period=0 --force`), so it does not drain gracefully. This mimics an abrupt hardware failure rather than a clean shutdown:

```bash
gcloud container clusters get-credentials ${GKE_CLUSTER?} --location ${ZONE?} --project ${PROJECT_ID?}

# Pick a worker pod on one slice and remove it immediately.
WORKER=$(kubectl get pods -o name | grep "${RUN_NAME?}" | grep worker | head -1)
kubectl delete ${WORKER?} --grace-period=0 --force
```

```{warning}
This deliberate pod deletion is only for observing recovery in this demo. Do not remove pods this way against a real training job.
```

## 5. Verify in-process recovery

Recovery shows up in the **same controller log** you were already watching, which is the point: the controller process never exited. Within seconds of the termination you should see Pathways report the slice down and `elastic_retry` restore the last committed checkpoint:

```
Slice down event detected. Retrying.
Found commit_success file. Keeping gs://.../checkpoints/100/.
Elastic attempt 2 out of 10
Restoring checkpoint from gs://.../checkpoints/100.
completed step: 101, ...
```

The step counter dropping (for example `150 -> 101`) is the rewind to the last committed checkpoint. Training then continues from there on the same controller, with no JobSet restart. That is the whole point of elastic training: a slice failure became a short rewind instead of a full job restart.

## 6. Clean up

Delete the workload to stop the meter. TPU slices are expensive, so don't skip this.

```bash
xpk workload delete --workload=${RUN_NAME?} --cluster=${GKE_CLUSTER?} --project=${PROJECT_ID?} --zone=${ZONE?}
```

If you created the cluster only for this demo, delete it too (see the [XPK documentation](https://github.com/AI-Hypercomputer/xpk) for `xpk cluster delete`).

## Going further

- **The elastic flags** are documented in `src/maxtext/configs/base.yml`: `elastic_enabled`, `elastic_timeout_seconds`, `elastic_max_retries`, plus `enable_single_controller` (runs training through Pathways) and `checkpoint_period`.
- **A larger model** changes the checkpoint size that streams through Pathways during recovery; size the controller and adjust `checkpoint_period` accordingly.
- **Custom Pathways server args** can be passed through `xpk` with `--custom-pathways-proxy-server-args` if you need finer control than `--elastic-slices` exposes.

## More information

- [Running MaxText with XPK](run_maxtext_via_xpk.md)
- [Running MaxText via Pathways](run_maxtext_via_pathways.md)
- [Pathways on Cloud documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
