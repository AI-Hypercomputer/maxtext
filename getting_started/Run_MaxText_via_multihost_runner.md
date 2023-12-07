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

## Getting Started: `multihost_runner.py` - Quick Experiments on Multiple Hosts (or Multiple Slices)

This workflow using `multihost_runner.py` is optimized for quick experiments, repeatedly re-using the same TPUs. Because the `multihost_runner.py` script depends on long-lived `ssh` connections, we do not recommend it for any long-running jobs.

We call the `runner` machine the one that `multihost_runner.py` is called from. This script will `ssh` into TPUVM `worker` machines that are found from the `--TPU_PREFIX` flag, and must be different than the runner machine.
If the runner machine is a cloud VM, it must be in the same project as the workers.

The `multihost_runner.py` script:
* Distributes your code by recursively copying the current state of the chosen directory to multiple worker TPUVM.
* Runs the code on the workers
* Logs and monitors the processes' error statuses and brings the logs back to the runner machine.

Although there are several steps below, most are for the initial setup. Once setup you can continually make changes to your code and re-run your code with only step 5.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can
either be a TPUVM or not, but it cannot be one of the workers. If your runner machine is a TPUVM, it needs service account roles that grant it permission to create queued resources and ssh into them, such as the `TPU ADMIN` role. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone, and ssh keys.

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

    Create ssh keys for gcloud, we recommend leaving a blank password (hit enter twice after running the below command). If you are prompted that the the file already exists you can choose not to overwrite by selecting "n".
    ```
    ssh-keygen -f ~/.ssh/google_compute_engine
    ```

3. Create your instances via Queued Resource (QR).
    Choose names for your TPUs and QR:
    ```
    TPU_PREFIX=$YOUR_TPU_NAME # Use new names when you create new TPUs
    QR_ID=$TPU_PREFIX # Convenient to re-use the node names, but can be different
    ```
    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    Create a multislice environment of nodes using create queued resources
    ```
    gcloud alpha compute tpus queued-resources create $QR_ID --accelerator-type=v4-8 --runtime-version=tpu-ubuntu2204-base --node-count=$NODE_COUNT --node-prefix=$TPU_PREFIX  --reserved
    ```
    We target the `reserved` pool above, but you may instead target the `on-demand` pool by omitting this flag,
    or target pre-emptible capacity with the `--best-effort` flag, which may be necessary if your reservation is full.

    You have to wait for the QR to become `ACTIVE` (as opposed to `ACCEPTED` or `PROVISIONING`) which corresponds to the worker nodes becoming `READY` (as opposed to `CREATING`). This may take a minute or two and can be checked via
    ```
    gcloud alpha compute tpus queued-resources list --filter=$QR_ID
    ```
4. Install dependencies.
    Install the dependencies of `train.py` on each worker using `multihost_runner.py`:
    ```
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh"
    ```
    If you are running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_IP=true`.

5. Run your training job.

    Set a RUN_NAME for your job:
    ```
    RUN_NAME=$YOUR_JOB_NAME # You may set this to any unique name for a fresh run.
    ```
    Set config values for `base_output_directory` and `dataset_path` in `configs/base.yml` if not set already.
    ```
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME"
    ```
    If you are running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_IP=true`.

6. Clean up TPUs and QR when finished.

     ```
    gcloud alpha compute tpus queued-resources delete $QR_ID --force --async
    ```

    The `--force` flag deletes both the queued resources and the TPU VMs, without it only a `SUSPENDED` queued resource whose TPUs have already been deleted can itself be deleted. We highly recommend the `--async` flag since deleting the TPUs and QR will take a minute or two.

