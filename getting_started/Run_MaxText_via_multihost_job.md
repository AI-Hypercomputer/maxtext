<!--
 Copyright 2023 Google LLC

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

## Getting Started: `multihost_job.py` - Production Jobs On Multiple Slices

The workflow using `multihost_job.py` is optimized for long running experiments, providing resiliency against hardware failure and avoiding long running ssh connections. Its latency is much higher than `multihost_runner.py` because it needs to provision new capacity each time. The `multihost_job.py` script ends once the request to create the TPUs is issued. Logs are written both to gcloud in real time and also sent to GCS at the end of the job.

The `multihost_job.py` script:

* Copies your code to your GCS bucket
* Spins up specified TPU VM(s) via CQR
* Directs the TPU's to download then run that code. Because this logic is within the CQR's startup script, if there hardware is interrupted, the job will be rescheduled and resumed.
* Logs to gcloud, and additionally sends the logs to GCS at the job end
* Delete the TPUs and QR at the end of the job.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can
either be a TPUVM or not. If your runner machine is a TPUVM, it needs service account roles that grant it permission to create queued resources and has write access to GCS, such as the `TPU ADMIN` and `STORAGE ADMIN` roles. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone.
    Set your gcloud config, see https://cloud.google.com/sdk/gcloud/reference/config for more.
    ```
    PROJECT=<project>
    ```

    ```
    ZONE=<zone>
    ```

    ```
    gcloud config set project $PROJECT
    gcloud config set compute/zone $ZONE
    ```

3. Link to a GCS bucket.
    Create a bucket if you don't already have one, see: https://cloud.google.com/storage/docs/creating-buckets for instructions to create one. Once you've identified your bucket:

    ```
    BUCKET_NAME=<your-bucket>
    ```

4. Run your training job.

    *** IMPORTANT *** `multihost_job` creates a request for new capacity for each run! You cannot use this tool on existing capacity, instead we recommend `multihost_runner` for this purpose.

    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    ```
    RUN_NAME=$YOUR_JOB_NAME # You may set this to any unique name for a fresh run.
    python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --CQR_EXTRA_ARGS="--reserved" --COMMAND="bash setup.sh && python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME"
    ```

    We tell `multihost_job` to target the `reserved` pool by  by including `--reserved` as extra arguments to the CQR request, but you may instead target the `on-demand` pool by removing the `--CQR_EXTRA_ARGS` flag (on-demand is default), or the pre-emptible pool with `--CQR_EXTRA_ARGS="--best-effort"`, which may be necessary if your reservation is full.

5. View the job's logs in cloud logging.

    The link to your job's cloud logging is printed at the end of `multihost_job` output. Additionally logs are saved to GCS when your job finishes, and this bucket's URL is also printed by `multihost_job`.

