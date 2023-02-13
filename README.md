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

This workflow is optimized for quick experiments. Because the `multihost_runner.py` script depends on long-lived `ssh` connections, we do not recommend it for any long-running jobs.

The `multihost_runner.py` script manages distributing code to multiple worker TPUVM's, running the code, monitoring the processes' error statuses and bringing the logs back to the host machine.

1. [TK] Set your project, zone, tpu name, ssh keys.
2. [TK] Create your instances via QR.
3. Install dependencies. 
```
python3 multihost_runner.py --TPU_PREFIX=<tpu_prefix> --COMMAND="bash setup.sh"
```
If you aren't running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_TPU=false`.

4. Run your training job.
```
python3 multihost_runner.py --TPU_PREFIX=<tpu_prefix> --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=${USER}_$(date +%Y-%m-%d-%H-%M-%S)"
```
If you aren't running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_TPU=false`.

## Getting Started: Production Jobs Via Queued Resource
For Matt to fill out.
1. [TK] Set your project, zone, tpu name, ssh keys.
2. ????
3. ????

# Runtime Performance Results

TK

# Full Training Results

TK -- this will get cut for private preview.

If you're making a change likely to effect performance, please compare your run to the "reference" and make sure you're
doing better. Assuming you're doing better, merge your change and update the reference.
```
tensorboard --logdir=gs://max-experiments/rwitten_2023-01-20-01:02:08/tensorboard/
```

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


