# Sharding

Maxtext supports the following sharding mechanisms:

- Distributed Data Parallelism
- Tensor Parallelism
- Fully Sharded Data Parallel
- Sequence Parallel

They are covered in the following parameters. These are the default values from base.yml. Use the following sharding parameters for setting on a single TPU Slice or a GPU Slice.

```
ici_data_parallelism: 1
ici_fsdp_parallelism: -1 # recommended ICI axis to be auto-sharded
ici_fsdp_transpose_parallelism: 1
ici_sequence_parallelism: 1
ici_tensor_parallelism: 1
```

Following sharding values dictate how training will happen across multiple TPU Pods.

```
dcn_data_parallelism: -1  # recommended DCN axis to be auto-sharded
dcn_fsdp_parallelism: 1
dcn_fsdp_transpose_parallelism: 1
dcn_sequence_parallelism: 1  # never recommended
dcn_tensor_parallelism: 1 # never recommended
dcn_pipeline_parallelism: 1
dcn_expert_parallelism: 1
dcn_autoregressive_parallelism: 1 # never recommended
```
