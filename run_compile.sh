#!/bin/bash
export BASE_IMAGE="gcr.io/tpu-prod-env-one-vm/chengnuojin_google_com_ragged_sort@sha256:5a5014a362f453f29c8b89902d024ee2c89c53000b74904504ae3b01d3567a54"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_dvfs_p_state=7 \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false \
  --xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true "

docker run --rm \
  -v /home/mattdavidow_google_com/maxtext:/deps \
  -e JAX_PLATFORMS=cpu \
  -e LIBTPU_INIT_ARGS="${XLA_FLAGS}" \
  -e PYTHONPATH=/deps/src \
  "${BASE_IMAGE}" \
  python3 -m maxtext.trainers.pre_train.train_compile \
  src/maxtext/configs/base.yml \
  compile_topology=tpu7x-128 \
  compile_topology_num_slices=1 \
  model_name=deepseek3-671b \
  override_model_config=True \
  base_num_decoder_layers=10 \
  weight_dtype=bfloat16 \
  opt_type=sgd \
  steps=2 \
  ici_fsdp_parallelism=32 \
  ici_expert_parallelism=4 \
  ici_data_parallelism=1 \
  ici_pipeline_parallelism=1 \
  per_device_batch_size=4.0 \
  max_target_length=4096 \
  dataset_type=synthetic \
  attention=flash \
  sparse_matmul=true \
  megablox=true \
  use_tokamax_splash=true \
  use_random_routing=true \
  use_ring_of_experts=true \
  use_ragged_sort=true \
  debug_sharding=True
