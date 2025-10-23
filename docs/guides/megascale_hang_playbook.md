# Megascale Hang Playbook

Cloud TPU Multislice environments are composed of multiple TPU slices that communicate over the Data Center Network (DCN). Multislice workloads use Megascale collectives to communicate over DCN. This guide discusses how to identify and diagnose sources of performance issues such as slowness, hangs or errors in a multi-slice job driven by Megascale.

## Hangs

A Megascale *hang* occurs when a worker has waited on a Megascale communication operation for a set timeout period. In this situation, you will see a Megascale `HANG_DETECTED` message in your Cloud TPU logs. Because Megascale is responsible for communicating over DCN, the hang is generally first detected and reported by Megascale. However, this does not mean the error has anything to do with Megascale. Most often, hang detection is a symptom of a problem in another part of the system.

In this way, we can think of the Megascale `HANG_DETECTED` message as a catch-all indication that your workload is not progressing properly. This could be caused by Customer-owned software, Google-owned software, or an issue with the hardware itself. This guide will help you determine which layer is causing the workload issue.

Much of this guide is geared towards providing Google with the right data to help debug your workload issues.

## Before You Start

1. Use `JAX` 0.6 or up, and enable JAX distributed service. This version of JAX contains additional logging that can help identify which workers are experiencing issues.
2. Generate an HLO dump using the `--xla_dump_to` flag when initializing your workload. This is discussed in the [XLA Documentation](https://openxla.org/xla/hlo_dumps).
3. Run your workload with stack traces enabled. XPK users should follow the [XPK-specific instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#collect-stack-traces). Note the `--deploy-stacktrace-sidecar` flag when running the XPK workload command.
4. Set `--vmodule=real_program_continuator=1` to enable verbose logging for the TPU program execution status.

## Locate the Megascale Hang Detected Error

```{note}
The following message was added in JAX 0.4.33. Make sure you are using a version at least this new.
```

When Megascale detects a hang in your workload, the first step is to locate the `Megascale detects a hang` message in your logs. This is generally found in worker 0/slice 0 of your coordinator logs.

This message will often provide the potential cause of a hang. Please provide Google with the relevant logging information.

## Common Issues

### 1. Inconsistent TPU Programs

Occasionally, different programs can run on TPU workers within the same system. This can lead to errors. Search your logs for a message like the following:

```
Megascale detects a hang that is likely caused by inconsistent TPU programs. This can be caused by some workers running with different JIT functions or a bug in the XLA compiler. Please inspect the HLO dumps to confirm the root cause.

Example hosts that have different HLO fingerprints: ...

Full error digest:
  Potential cause: INCONSISTENT_TPU_PROGRAM
  Potential culprit workers: <host_name>
  TPU stats:
    <host_name>: <pc>
  TPU program fingerprints:
    <host_name>: <fingerprint>
```

If you see this log, we need to inspect the HLO dumps. Use the following steps to [collect HLO dumps](https://openxla.org/xla/hlo_dumps), and then send them to Google.

### 2. Bad TPU Chip or Data Pipeline Stall

The following error message indicates that there is either a faulty TPU, or the workers are stalling on the input pipeline.

```
Megascale detects a hang that is likely caused by bad TPU chips on the following hosts. Please remove the hosts from the fleet and restart the workload. If problem persists please contact Megascale XLA team.
The host that have bad TPUs are: <host>
Full error digest:
  Potential cause: Bad TPU chips
  Potential culprit workers: <host>
  TPU states:
    Module: jit_loss_and_grad
    Fingerprint: <fingerprint>
    Launch ID: <id>
      <tag>:<pc>(<hlo>): <host>
```

If the TPU listed in the log shows a non-zero program counter, it is very likely that the TPU is faulty and causing the hang. Restart the job and check if the same TPU shows up as an outlier. If the TPU repeatedly shows up as an outlier, send the information to Google so we can remove the host from the fleet.

If the logged TPU shows a program counter of 0, it is likely that the TPU is waiting on input. We can attempt to confirm the worker is hung during the input pipeline using the stack trace library found in the [cloud-tpu-diagnostics package](https://pypi.org/project/cloud-tpu-diagnostics/).

XPK users should follow the [XPK-specific instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#collect-stack-traces) to emit stack traces. Note the `--deploy-stacktrace-sidecar` flag when running the XPK workload command.

Customers can then query Cloud Logging for the stack trace logs from the outlier TPU. The stack trace log will help users determine where in the Python code the program was during the hang.

### 3. Unknown Error

The following error indicates the MXLA runtime cannot determine the cause of the hangs. Send Google an HLO dump (see how [here](https://openxla.org/xla/hlo_dumps)) for further debugging.

```
Megascale detects a hang but cannot determine the root cause. Please inspect the full digest below.
```

### 4. Checkpoint File Sizes Too Small

If your checkpoint file sizes are too small and frequent, there is a risk of GCS issues. We recommend using [Orbax checkpointing](https://orbax.readthedocs.io/en/latest/index.html) which writes 200MB files by default.

### 5. Training code uses MXLA to synchronize for a I/O operation

Training code might use [MXLA collectives](https://openxla.org/xla) as a global barrier to sync across devices.

In case one of the hosts is taking more than the time set in `--megascale_graph_hang_threshold` (default is 5 minutes) then MXLA timeout will expire. Depending on the setting, MXLA will either block the training or will crash and restart it. If the latter, then a series of restarts will occur until the I/O operation completes within the timeout period.

As an example, this could occur when using an `all-reduce` to synchronize at the end of checkpoint loading. One possible solution would be to switch this collective barrier to use the JAX distributed `wait_at_barrier` API instead:

```python
jax._src.distributed.global_state.client.wait_at_barrier(...)
```

for worker synchronization.

## Share HLO Dump for debugging purposes

After [creating an HLO Dump](https://openxla.org/xla/hlo_dumps), you can share it with Google using your preferred method. For example, you can move the HLO dump to GCS using a command line like this on each VM:

```
gcloud storage cp -r /tmp/xla_dump gs://<bucket_location>
```

When sharing the HLO dump, you will need to give Google permission to access the GCS bucket. A Google user can then download the HLO graph using `gsutil`.
