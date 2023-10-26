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

# XPK for Large Scale (>1k VMs)

Follow user instructions in [xpk-large-scale-guide.sh](xpk-large-scale-guide.sh)
to use xpk for a GKE cluster greater than 1000 VMs. Run these steps to set up a
GKE cluster with large scale training and high throughput support with XPK, and
run jobs with XPK. We recommend you manually copy commands per step and verify
the outputs of each step.

# Example usages:

To get started, be sure to set your GCP Project and Zone as usual via `gcloud
config set`.

Below are reference commands. A typical journey starts with a `Cluster Create`
followed by many `Workload Create`s. To understand the state of the system you
might want to use `Cluster List` or `Workload List` commands. Finally, you can
cleanup with a `Cluster Delete`.

## Cluster Create
*   Cluster Create (provision on-demand capacity):

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --num-slices=4
    ```


*   Cluster Create (provision reserved capacity):

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-256 \
    --num-slices=2 \
    --custom-tpu-nodepool-arguments="--reservation-affinity=specific --reservation=RESERVATION_ID"
    ```

*   Cluster Create can be called again with the same `--cluster name` to modify
    the number of slices or retry failed steps.

    For example, if a user creates a cluster with 4 slices:

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --num-slices=4
    ```

    and recreates the cluster with 8 slices. The command will rerun to create 4
    new slices:

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --num-slices=8
    ```

    and recreates the cluster with 6 slices. The command will rerun to delete 2
    slices. The command will warn the user when deleting slices.
    Use `--force` to skip prompts.

    ```shell
    python3 xpk/xpk.py cluster create \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --num-slices=6

    # Skip delete prompts using --force.

    python3 xpk/xpk.py cluster create --force \
    --cluster xpk-test --tpu-type=v5litepod-16 \
    --num-slices=6

    ```
## Cluster Delete
*   Cluster Delete (deprovision capacity):

    ```shell
    python3 xpk/xpk.py cluster delete \
    --cluster xpk-test
    ```
## Cluster List
*   Cluster List (see provisioned capacity):

    ```shell
    python3 xpk/xpk.py cluster list
    ```
## Cluster Describe
*   Cluster Describe (see capacity):

    ```shell
    python3 xpk/xpk.py cluster describe \
    --cluster xpk-test
    ```

## Cluster Cacheimage
*   Cluster Cacheimage (enables faster start times):

    ```shell
    python3 xpk/xpk.py cluster cacheimage \
    --cluster xpk-test --docker-image gcr.io/your_docker_image
    ```

## Workload Create
*   Workload Create (submit training job):

    ```shell
    python3 xpk/xpk.py workload create \
    --workload xpk-test-workload --command "echo goodbye" --cluster \
    xpk-test --tpu-type=v5litepod-16
    ```

### Workload Priority and Preemption
* Set the priority level of your workload with `--priority=LEVEL`

  We have five priorities defined: [`verylow`, `low`, `medium`, `high`, `very-high-non-preempt`].
  The default priority is `medium`.

  Priority determines:

  1. Order of queued jobs.

      Queued jobs are ordered by
      `verylow` < `low` < `medium` < `high` <  `very-high-non-preempt`

  2. Preemption of lower priority workloads.

      A higher priority job will `evict` lower priority jobs.
      Evicted jobs are brought back to the queue and will re-hydrate appropriately.

      Only `very-high-non-preempt` will not preempt other jobs. `very-high-non-preempt` will enter the queue with the
      highest priority and wait its turn.

  #### General Example:
  ```shell
  python3 xpk/xpk.py workload create \
  --workload xpk-test-medium-workload --command "echo goodbye" --cluster \
  xpk-test --tpu-type=v5litepod-16 --priority=medium
  ```

## Workload Delete
*   Workload Delete (delete training job):

    ```shell
    python3 xpk/xpk.py workload delete \
    --workload xpk-test-workload --cluster xpk-test
    ```

## Workload List
*   Workload List (see training jobs):

    ```shell
    python3 xpk/xpk.py workload list \
    --cluster xpk-test
    ```

# How to add docker images to a xpk workload

The default behavior is `xpk workload create` will layer the local directory (`--script-dir`) into
the base docker image (`--base-docker-image`) and run the workload command.
If you don't want this layering behavior, you can directly use `--docker-image`. Do not mix arguments from the two flows in the same command.

## Recommended / Default Docker Flow: `--base-docker-image` and `--script-dir`
This flow pulls the `--script-dir` into the `--base-docker-image` and runs the new docker image.

* The below arguments are optional by default. xpk will pull the local
  directory with a generic base docker image.

  - `--base-docker-image` sets the base image that xpk will start with.

  - `--script-dir` sets which directory to pull into the image. This defaults to the current working directory.

  See `python3 xpk/xpk.py workload create --help` for more info.

* Example with defaults which pulls the local directory into the base image:
  ```shell
  echo -e '#!/bin/bash \n echo "Hello world from a test script!"' > test.sh
  python3 xpk/xpk.py workload create --cluster xpk-test \
  --workload xpk-test-workload-base-image --command "bash test.sh" \
  --tpu-type=v5litepod-16 --num-slices=1
  ```

* Recommended Flow For Normal Sized Jobs (fewer than 10k accelerators):
  ```shell
  python3 xpk/xpk.py workload create --cluster xpk-test \
  --workload xpk-test-workload-base-image --command "bash custom_script.sh" \
  --base-docker-image=gcr.io/your_dependencies_docker_image \
  --tpu-type=v5litepod-16 --num-slices=1
  ```

## Optional Direct Docker Image Configuration: `--docker-image`
If a user wants to directly set the docker image used and not layer in the
current working directory, set `--docker-image` to the image to be use in the
workload.

* Running with `--docker-image`:
  ```shell
  python3 xpk/xpk.py workload create --cluster xpk-test \
  --workload xpk-test-workload-base-image --command "bash test.sh" \
  --tpu-type=v5litepod-16 --num-slices=1 --docker-image=gcr.io/your_docker_image
  ```

* Recommended Flow For Large Sized Jobs (more than 10k accelerators):
  ```shell
  python3 xpk/xpk.py cluster cacheimage \
  --cluster xpk-test --docker-image gcr.io/your_docker_image
  # Run workload create with the same image.
  python3 xpk/xpk.py workload create --cluster xpk-test \
  --workload xpk-test-workload-base-image --command "bash test.sh" \
  --tpu-type=v5litepod-16 --num-slices=1 --docker-image=gcr.io/your_docker_image
  ```

# More advanced facts:

* Workload create accepts a --docker-name and --docker-image.
By using custom images you can achieve very fast boots and hence very fast
feedback.

* Workload create accepts a --env-file flag to allow specifying the container's
environment from a file. Usage is the same as Docker's
[--env-file flag](https://docs.docker.com/engine/reference/commandline/run/#env)


# Troubleshooting

## `Invalid machine type` for CPUs.
XPK will create a regional GKE cluster. If you see issues like

```shell
Invalid machine type e2-standard-32 in zone $ZONE_NAME
```

Please select a CPU type that exists in all zones in the region.

```shell
# Find CPU Types supported in zones.
gcloud compute machine-types list --zones=$ZONE_LIST
# Adjust default cpu machine type.
python3 xpk/xpk.py cluster create --cluster-cpu-machine-type=CPU_TYPE ...
```
