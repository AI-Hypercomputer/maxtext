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

# Overview

xpk (Accelerated Processing Kit, pronounced x-p-k,) is a software tool to help
Cloud developers to orchestrate training jobs on accelerators such as TPUs and
GPUs on GKE. xpk handles the "multihost pods" of TPUs and GPUs (HGX H100) as
first class citizens.

xpk decouples provisioning capacity from running jobs. There are two structures:
clusters (provisioned VMs) and workloads (training jobs). Clusters represent the
physical resources you have available. Workloads represent training jobs -- at
any time some of these will be completed, others will be running and some will
be queued, waiting for cluster resources to become available.

The ideal workflow starts by provisioning the clusters for all of the ML
hardware you have reserved. Then, without re-provisioning, submit jobs as
needed. By eliminating the need for re-provisioning between jobs, using Docker
containers with pre-installed dependencies and cross-ahead of time compilation,
these queued jobs run with minimal start times. Further, because workloads
return the hardware back to the shared pool when they complete, developers can
achieve better use of finite hardware resources. And automated tests can run
overnight while resources tend to be underutilized.

# Example usages:

To get started, be sure to set your GCP Project and Zone as usual via `gcloud
config set`.

Below are reference commands. A typical journey starts with a `Cluster Create`
followed by many `Workload Create`s. To understand the state of the system you
might want to use `Cluster List` or `Workload List` commands. Finally, you can
cleanup with a `Cluster Delete`.

*   Cluster Create (provision capacity):

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --host-maintenance-interval=PERIODIC --num-slices=4
    ```

    Cluster Create can be called again with the same `--cluster name` to modify
    the number of slices or retry failed steps.

    For example, if a user creates a cluster with 4 slices:

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --host-maintenance-interval=PERIODIC --num-slices=4
    ```

    and recreates the cluster with 8 slices. The command will rerun to create 4
    new slices:

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --host-maintenance-interval=PERIODIC --num-slices=8
    ```

    and recreates the cluster with 6 slices. The command will rerun to delete 2
    slices. The command will warn the user when deleting slices.
    Use `--force` to skip prompts.

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --host-maintenance-interval=PERIODIC --num-slices=8

    # Skip delete prompts using --force.

    python3 xpk/xpk.py cluster create --force \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --host-maintenance-interval=PERIODIC --num-slices=8

    ```


*   Cluster Delete (deprovision capacity):

    ```shell
    python3 xpk/xpk.py cluster delete \
    --cluster xpk-test
    ```

*   Cluster List (see provisioned capacity):

    ```shell
    python3 xpk/xpk.py cluster list
    ```

*   Cluster Describe (see capacity):

    ```shell
    python3 xpk/xpk.py cluster describe \
    --cluster xpk-test
    ```

*   Cluster Cacheimage (enables faster start times):

    ```shell
    python3 xpk/xpk.py cluster cacheimage \
    --cluster xpk-test --docker-image gcr.io/your_docker_image
    ```

*   Workload Create (submit training job):

    ```shell
    python3 xpk/xpk.py workload create \
    --workload xpk-test-workload --command "echo goodbye" --cluster \
    xpk-test --tpu-type=v5litepod-16
    ```

*   Workload Delete (delete training job):

    ```shell
    python3 xpk/xpk.py workload delete \
    --workload xpk-test-workload --cluster xpk-test
    ```

*   Workload List (see training jobs):

    ```shell
    python3 xpk/xpk.py workload list \
    --cluster xpk-test
    ```

# More advanced facts:

* Workload create accepts a --docker-name and --docker-image.
By using custom images you can achieve very fast boots and hence very fast
feedback.

* Workload create accepts a --env-file flag to allow specifying the container's
environment from a file. Usage is the same as Docker's
[--env-file flag](https://docs.docker.com/engine/reference/commandline/run/#env)
