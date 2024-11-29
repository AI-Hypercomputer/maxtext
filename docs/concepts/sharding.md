# Sharding

Maxtext supports the following sharding mechanisms:

- Distributed Data Parallelism
    - This is arguably the simplest parallelization strategy, where each device
      can run the forward pass independently on a different set of data. The
      devices must communicate the gradients during the backward pass. This
      strategy works best with large per device batch sizes, and is suitable for
      slower networks since it doesn't require much communication.
- Fully Sharded Data Parallelism
    - Similar to data parallelism each device computes on a different set of
      data. However additionally the optimizer state is sharded across devices,
      which allows larger models to fit in this distributed memory. However now
      the weights need to be all-gathered during the forward pass. This strategy
      works best with large per device batch sizes.
- Tensor Parallelism
    - Each device has the same data, but is responsible for computing a
      different set of features. For the feed forward layer, this requires all
      gathering the activations, performing the computations, and then
      reduce-scattering the output, similar to the
      [megatron strategy](https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf).
      Ideally these communications can be overlapped with the compute in a
      pattern called a "collective matmul". This strategy works best for large
      models (large intermediate or "mlp" dim), and is often used when the per
      device batch size is small (which is where pure FSDP would not work well).
      In MaxText we shard the heads by the tensor parallel axis for the
      attention ops, since the heads act like a batch dimension it is easy to
      use with efficient attention kernels such as flash attention.
- Sequence Parallelism
    - Sequence parallelism as implemented in MaxText is similar to fully sharded
      data parallelism. The optimizer state is sharded just like as in FSDP, and
      we still shard the tokens, but on the "sequence" dimension instead of the
      "batch" dimension. However for the attention component we shard the heads
      by the sequence axis for the same reason as TP above - heads act like
      batch dimension in the attention ops. Transition from sharding on sequence
      to heads requires an all-to-all which should be cheap. Sequence
      parallelism has strictly more communications than FSDP because of this
      all-to-all, however it allows for a fractional per device batch size since
      we shard the sequence dimension instead of the batch dimension. A
      fractional per device batch size is needed to remain within memory limits
      for longer sequence lengths.
- Pipeline parallelism
    - Pipeline parallelism shards the optimizer state and computation by layers.
      In MaxText we have implemented a "circular" pipeline which is able to
      achieve smaller pipeline "bubbles" (idle time). Users can tradeoff bubble
      versus communications by setting the layers per stage, more layers per
      stage -> less communications required between layers, but also a larger
      bubble due to fewer repeats. Pipeline parallelism is useful when the
      gradient comms of data parallelism across the slower network cannot be
      hidden, which generally occurs with "strong scaling" (fixed global batch
      size of say 8M or 16M tokens) and a large number of "pods" or slower
      network data parallel replicas. Pipeline parallelism is most useful for
      large models when run on a huge cluster which drives the per device batch
      size (per pod batch size) small.
- Expert parallelism
    - Expert parallelism is specific to MoE models. It shards the optimizer
      state and computation by experts for the MoE feedforward component. The
      attention component is shared across experts, and thus in MaxText the
      expert parallelism axis acts like FSDP in the attention layer. Moving
      between this expert sharding and FSDP sharding requires an all-to-all,
      which is generally cheap, and thus expert parallelism is often used in any
      MoE configuration. However currently in MaxText we only support expert
      parallelism with a dropping strategy (dropping tokens that exceed an
      "expert capacity"), we are still improving the EP integrations.

These mechanisms are covered in the following parameters. These are the default
values from `base.yml`. Use the following sharding parameters for setting on a
single TPU Slice or a GPU Slice.

```
ici_data_parallelism: 1
ici_fsdp_parallelism: -1 # recommended ICI axis to be auto-sharded
ici_fsdp_transpose_parallelism: 1
ici_sequence_parallelism: 1
ici_tensor_parallelism: 1
```

The following sharding values dictate how training will happen across multiple TPU Pods.

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

## Sharding implementation details

You may think of the sharding in the maxtext codebase as split into three levels
1. The physical mesh where e.g. `ici_fsdp_parallelism` is used - see [`create_device_mesh`](https://github.com/AI-Hypercomputer/maxtext/blob/e7c4824ee9cc13fd6db863796bbe7696b03eb448/MaxText/max_utils.py#L363)
2. The logical names, with physical <-> logical mappings [here](https://github.com/AI-Hypercomputer/maxtext/blob/e7c4824ee9cc13fd6db863796bbe7696b03eb448/MaxText/configs/base.yml#L211-L248)
3. Individual tensors which will use logical names, here is one [example](https://github.com/AI-Hypercomputer/maxtext/blob/e7c4824ee9cc13fd6db863796bbe7696b03eb448/MaxText/layers/linears.py#L243)

Following this example we see the first axis is sharded by logical name "embed". This logical name maps the physical names "fsdp, fsdp_transpose, sequence, expert", thus this axes will get sharded by the product of these specified parallelisms. E.g. if `ici_fsdp_parallelism=4` and `ici_sequence_parallelism=2` then this array axis will get sharded 8 ways.

This example showed a "kernel_axes" which is used to define a weight matrix. For activations we use shardings hints for the compiler such as `nn.with_logical_constraint` (example [here](https://github.com/AI-Hypercomputer/maxtext/blob/e7c4824ee9cc13fd6db863796bbe7696b03eb448/MaxText/layers/linears.py#L261)). This will generally shard the activations according to these constraints, but the compiler occasionally chooses a different sharding other that what we specified for these activations.
