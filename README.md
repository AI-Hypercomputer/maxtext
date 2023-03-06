# Overview

MaxText is a **high performance**, **arbitrarily scalable**, **open-source**, **simple**, **easily forkable**, **well-tested**, **batteries included** LLM written in pure Python/Jax and targeting Google Cloud TPUs. MaxText achieves [TK] model-flop utilization and scales from single host to very large clusters (tested on up to [TK] chips) while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs. If you're additionally interested in contributing to the community, need support or just want to get in touch, [learn more](#contributions-and-bug-reports).

# Table of Contents

* [Getting Started](#getting-started)
* [Runtime Performance Results](#runtime-performance-results)
* [Full Training Results](#full-training-results)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)
* [Contributions and Bug Reports](#contributions-and-bug-reports)

# Getting Started

There are three recommended patterns for running MaxText. You can run locally, run on a cluster experimentally or spawn a production-style that is managed by Google Compute Engine. We recommend starting with Local Development, moving to Cluster Experimentation for some ad hoc development and ultimately running your long running jobs with Queued Resources.

## Getting Started: Local Development

Local development is the faster and most convenient way to run MaxText. However, it doesn't scale to multiple hosts.

1. [Create and SSH to the single-host TPU of your choice.](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud) We recommend a `v4-8`.
2. Clone MaxText onto that TPUVM.
3. Within the root directory of that `git` repo, install dependencies by running:
```
bash setup.sh
```
4. After installation completes, run training with the command:
```
python3 MaxText/train.py MaxText/configs/base.yml run_name=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
```

5. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml run_name=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
```
Be aware, these decodings will be random. To get high quality decodings you need pass in a checkpoint, typically via the `load_parameters_path` argument.

## Getting Started: Cluster Experimentation

This workflow using `multihost_runner.py` is optimized for quick experiments. Because the `multihost_runner.py` script depends on long-lived `ssh` connections, we do not recommend it for any long-running jobs.

We call the `runner` machine the one that `multihost_runner.py` is called from. This script will `ssh` into TPUVM `worker` machines that are found from the `--TPU_PREFIX` flag, and must be different than the runner machine.
If the runner machine is a cloud VM, it must be in the same project as the workers.

The `multihost_runner.py` script:
* Distributes your code to multiple worker TPUVM's
* Runs the code on the workers
* Logs and monitors the processes' error statuses and brings the logs back to the runner machine.

Although there are several steps below, most are for the initial setup. Once setup you can continually make changes to your code and re-run your code with only step 5.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can 
either be a TPUVM or not, but it cannot be one of the workers. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone, gcloud permissions and ssh keys.

    Authorize gcloud to access the Cloud Platform with Google user credentials
    ```
    gcloud auth login
    ```

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

    Create ssh keys for gcloud, we recommend leaving a blank password (hit enter twice after running the below command)
    ```
    ssh-keygen -f ~/.ssh/google_compute_engine 
    ```
    
3. Create your instances via Queued Resource (QR). 
    Choose names for your TPUs and QR:
    ```
    TPU_PREFIX=${USER}_$(date +%Y-%m-%d-%H-%M-%S) # Use new names when you create new TPUs
    QR_ID=$TPU_PREFIX # Convenient to re-use the node names, but can be different
    ```
    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    Create a multislice environment of nodes using create queued resources
    ```
    gcloud alpha compute tpus queued-resources create $QR_ID --accelerator-type=v4-8 --runtime-version=tpu-vm-v4-base --node-count=$NODE_COUNT --node-prefix=$TPU_PREFIX  --reserved
    ```
    You have to wait to for the QR to become `ACTIVE` (as opposed to `ACCEPTED` or `PROVISIONING`) which corresponds to the worker nodes becoming `READY` (as opposed to `CREATING`). This may take a minute or two and can be checked via
    ```
    gcloud alpha compute tpus queued-resources list --filter=$TPU_PREFIX 
    ```
4. Install dependencies. 
    ```
    pip3 install absl-py # Dependency of multihost_runner.py
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh" # Dependencies of MaxText/train.py
    ```

5. Run your training job.

    Set a RUN_NAME for your job:
    ```
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S) # You may set this to any unique name for a fresh run.
    ```
    ```
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME dcn_data_parallelism=$NODE_COUNT"
    ```

6. Clean up TPUs and QR when finished.

    There is ongoing work to simplify this step into a single command, but for now it is split into two main steps:

    a. Delete the nodes

    ```
    for ((i=0; i<$NODE_COUNT; i++))
    do
        curl -X DELETE -H "Authorization: Bearer $(gcloud auth print-access-token)" https://tpu.googleapis.com/v2alpha1/projects/$PROJECT/locations/$ZONE/nodes/${TPU_PREFIX}-$i
    done
    ```
    b. Delete the QR, first waiting for the nodes the to be deleted (~15 seconds). You should see the status of the QR will change from `ACTIVE` to `SUSPENDING` to `SUSPENDED` as the nodes are deleted (the QR must be `SUSPENDED` to be deletable), which can be checked via

    ```
    gcloud alpha compute tpus queued-resources list --filter=$QR_ID
    ```
    When the QR is in state `SUSPENDED`, delete it.

    ```
    gcloud alpha compute tpus queued-resources delete $QR_ID
    ```

## Getting Started: Production Jobs Via Queued Resource

This workflow using `multihost_job.py` is optimized for long running experiments. This script avoids any `ssh` connection, and logs are written to cloud logging.

The `multihost_job.py` script:

* Copies your code to your GCS bucket
* Spins up specified TPUVM(s) via CQR
* Directs the TPU's to download then run that code
* Logs locally to each worker TPU, sending these logs to GCS at the job end
* Delete the TPUs at the end of the job.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can 
either be a TPUVM or not. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone, and gcloud permissions. 
     Authorize gcloud to access the Cloud Platform with Google user credentials
    ```
    gcloud auth login
    ```

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

    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    ```
    pip3 install absl-py # Dependency of multihost_job.py
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S) # You may set this to any unique name for a fresh run.
    python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --COMMAND="bash setup.sh && python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME dcn_data_parallelism=$NODE_COUNT"
    ```
5. View the job's logs in cloud logging. 

    The link to your job's logs is printed at the end of the multihost_job output.

6. Cleanup the QR when finished.

    There is ongoing work to automate this step.

    When your job is finished `multihost_job.py` will delete the TPUs for you. However you still need to delete the QR. You can check that your job is done because the QR will no longer be `ACTIVE`, but instead in
    state `SUSPENDED` since the nodes have been deleted.
    ```
    gcloud alpha compute tpus queued-resources list --filter=$RUN_NAME # You can remove the filter to list all QR in your project 
    ```
    Then delete the QR

    ```
    gcloud alpha compute tpus queued-resources delete $RUN_NAME
    ```


# Runtime Performance Results

TK

# Full Training Results

See [Experiments](Experiments.md).


# Comparison to Alternatives

MaxText is heavily inspired by [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT), elegant standalone GPT implementations written in PyTorch and targeting Nvidia GPUs. MaxText is more complex but achieves higher model-flop utilization [TK], is massively scalable and implements a key-value cache for efficient auto-regressive decoding.

MaxText is most similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), a very well tuned LLM implementation targeting Nvidia GPUs. The two implementations achieve comparable MFUs. The difference in the codebases highlights the different programming strategies. MaxText is pure Python, relying heavily on the XLA compiler to achieve high performance. By contrast, Megatron-LM is a mix of Python and CUDA, relying on well-optimized CUDA kernels to achieve high performance.

MaxText is also comparable to [Pax](https://github.com/google/paxml). Like Pax, MaxText provides high-performance and scalable implementations of LLMs. Pax focuses on enabling powerful configuration parameters, enabling developers to change the model by editing config parameters. By contrast, MaxText is a simple, concrete implementation of an LLM that encourages users to extend by forking and directly editing the source code. The right choice depends on your project's priorities.

# Development

Whether you are forking MaxText for your own needs or intending to contribute back to the community, we wanted to offer simple testing recipes.

To run unit tests and lint, simply run:
```
bash unit_test_and_lint.sh
```

The full suite of end-to-end tests is in `end_to_end/`. We run them with a nightly cadence.

# Contributions and Bug Reports

We welcome contributions and bug reports!
* We're focused on continuing to make MaxText align to its [values](#overview) and welcome pull requests to improve simplicity, scalability and performance. Read the [development](#development) section for more context.
* To file a bug, use Github Issues.
* If you want to chat, join our public [Google Chat Room](TK).


