# Optimization 1: Multihost recommended network settings
We included all the recommended network settings in [rto_setup.sh](https://github.com/google/maxtext/blob/main/rto_setup.sh). 

[preflight.sh](https://github.com/google/maxtext/blob/main/preflight.sh) will help you apply them based on GCE or GKE platform.

Before you run ML workload on Multihost with GCE or GKE, simply apply `bash preflight.sh PLATFORM=[GCE or GKE]` to leverage the best DCN network performance.

Here is an example for GCE:
```
bash preflight.sh PLATFORM=GCE && python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```

Here is an example for GKE:
```
bash preflight.sh PLATFORM=GKE && python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```

# Optimization 2: Numa binding (You can only apply this to v4 and v5p)
NUMA binding is recommended for enhanced performance, as it reduces memory latency and maximizes data throughput, ensuring that your high-performance applications operate more efficiently and effectively.

For GCE, 
[preflight.sh](https://github.com/google/maxtext/blob/main/preflight.sh) will help you install `numactl` dependency, so you can use it directly, here is an example:

```
bash preflight.sh PLATFORM=GCE && numactl --membind 0 --cpunodebind=0 python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```

For GKE,
`numactl` should be built into your docker image from [maxtext_dependencies.Dockerfile](https://github.com/google/maxtext/blob/main/maxtext_dependencies.Dockerfile), so you can use it directly if you built the maxtext docker image. Here is an example

```
bash preflight.sh PLATFORM=GKE && numactl --membind 0 --cpunodebind=0 python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```

1. `numactl`: This is the command-line tool used for controlling NUMA policy for processes or shared memory. It's particularly useful on multi-socket systems where memory locality can impact performance.
2. `--membind 0`: This option binds the memory allocation of the process to node 0. It means the process will allocate memory only from the memory of node 0. If node 0's memory is exhausted and more is required, the process will fail rather than using memory from other nodes.
3. `--cpunodebind=0`: This option binds the process to the CPUs of node 0. The process will only run on the CPUs of node 0, not on CPUs of other nodes. This can improve performance by ensuring that the process runs on CPUs that are close to the memory it is using.
