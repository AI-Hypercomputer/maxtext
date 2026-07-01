# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Maps MaxText checkpoint config onto the Orbax v1 Context + training policies.

This module is the single place that translates MaxText's flat checkpoint flags
into those objects. It builds configuration only.

It replaces a scattered set of v0 constructs:
  * ``PyTreeCheckpointHandler(use_ocdbt=, use_zarr3=, *_concurrent_gb=)``  ->
  ``ctx.array`` / ``ctx.memory``
  * per-leaf ``SaveArgs(chunk_byte_size=)`` + ``ocdbt_target_data_file_size``
  -> ``ctx.array.saving``
  * ``SingleReplicaArrayHandler`` registration                              ->
  ``ctx.array.loading.use_load_and_broadcast``
  * ``ocp.pathways.register_type_handlers(...)``                            ->
  ``ctx.pathways.checkpointing_impl``
  * ``CheckpointManagerOptions(save_decision_policy=, preservation_policy=)`` ->
  the policy builders below
"""

from orbax.checkpoint import pathways as ocp_pathways
from orbax.checkpoint import v1 as ocp


_BYTES_PER_GB = 1024**3

# Matches the v0 SingleReplicaArrayHandler broadcast limit (1000 MB) that
# MaxText uses when restoring a single replica and broadcasting to the rest.
_SINGLE_REPLICA_BROADCAST_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1000


def build_save_decision_policy(
    *,
    save_interval_steps: int,
    enable_continuous_checkpointing: bool = False,
    enable_autocheckpoint: bool = False,
):
  """Builds the v1 SaveDecisionPolicy.

  - continuous: save as often as possible (async-friendly).
  - autocheckpoint: save on preemption OR at the fixed interval.
  - otherwise: save at the fixed interval.
  """
  policies = ocp.training.save_decision_policies
  if enable_continuous_checkpointing:
    return policies.ContinuousCheckpointingPolicy()
  if enable_autocheckpoint:
    return policies.AnySavePolicy([
        policies.PreemptionCheckpointingPolicy(),
        policies.FixedIntervalPolicy(save_interval_steps),
    ])
  return policies.FixedIntervalPolicy(interval=save_interval_steps)


def build_preservation_policy(*, max_num_checkpoints_to_keep: int):
  """Builds the v1 PreservationPolicy (keep the latest N checkpoints)."""
  return ocp.training.preservation_policies.LatestN(
      max_num_checkpoints_to_keep
  )


def build_context(
    *,
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    ocdbt_target_data_file_size_bytes: int | None = None,
    checkpoint_storage_concurrent_gb: int | None = None,
    async_timeout_secs: int | None = None,
    todelete_full_path: str | None = None,
    todelete_subdir: str | None = None,
    enable_single_replica_ckpt_restoring: bool = False,
    replica_axis_index: int = 0,
    colocated_python_checkpointing: bool = False,
    partial_load: bool = False,
    checkpoint_layout: ocp.options.CheckpointLayout | None = None,
) -> ocp.Context:
  """Builds an Orbax v1 ``Context`` from MaxText checkpoint flags.

  The returned Context is unfrozen (its options are mutable until it is entered
  via ``with ctx:``); callers pass it to ``ocp_v1.training.Checkpointer``, which
  applies it to every save/load.

  Args:
    use_ocdbt: Use OCDBT storage format.
    use_zarr3: Use Zarr3 storage format.
    ocdbt_target_data_file_size_bytes: Target OCDBT data-file size; also used as
      the per-array ``chunk_byte_size`` (matching the v0 ``SaveArgs`` value).
    checkpoint_storage_concurrent_gb: Concurrent IO budget in GB; applied to
      both write and read as a byte limit (v0 used one value for both).
    async_timeout_secs: Async save timeout; ``None`` keeps the Orbax default.
    todelete_full_path: GCS soft-delete path.
    todelete_subdir: Subdirectory renaming hook for deletions.
    enable_single_replica_ckpt_restoring: Restore on one replica and broadcast
      to the rest (replaces the v0 ``SingleReplicaArrayHandler``).
    replica_axis_index: Mesh axis separating replicas for load-and-broadcast.
    colocated_python_checkpointing: Use Pathways colocated-python checkpointing.
    partial_load: Restore only the keys present in the abstract tree (the v1
      equivalent of v0 ``partial_restore=True``).
    checkpoint_layout: On-disk layout (``ORBAX`` or ``SAFETENSORS``) for
      loading.

  Returns:
    A configured, unfrozen ``ocp_v1.Context``.
  """
  ctx = ocp.Context()

  # Array storage format + file sizing.
  ctx.array.saving.use_ocdbt = use_ocdbt
  ctx.array.saving.use_zarr3 = use_zarr3
  if ocdbt_target_data_file_size_bytes is not None:
    ctx.array.saving.ocdbt_target_data_file_size = (
        ocdbt_target_data_file_size_bytes
    )
    ctx.array.saving.storage_options.chunk_byte_size = (
        ocdbt_target_data_file_size_bytes
    )

  # Concurrent IO budget: v0 GB -> v1 bytes, applied to both directions.
  if checkpoint_storage_concurrent_gb is not None:
    concurrent_bytes = checkpoint_storage_concurrent_gb * _BYTES_PER_GB
    ctx.memory.write_concurrent_bytes = concurrent_bytes
    ctx.memory.read_concurrent_bytes = concurrent_bytes

  if async_timeout_secs is not None:
    ctx.asynchronous.timeout_secs = async_timeout_secs

  if todelete_full_path is not None:
    ctx.deletion.gcs_deletion_options.todelete_full_path = todelete_full_path

  if todelete_subdir is not None:
    # copybara:strip_begin(reason='CNS is Google-internal')
    ctx.deletion.cns_deletion_options.todelete_subdir = todelete_subdir
    # copybara:strip_end
    # copybara:insert_begin(reason='External version does not support todelete_subdir')
    # raise ValueError(
    #     "Renaming to subdirectory before deleting (todelete_subdir) is"
    #     " unsupported in this environment."
    # )
    # copybara:insert_end

  # Single-replica restore (load on one replica, broadcast to the others).
  if enable_single_replica_ckpt_restoring:
    ctx.array.loading.use_load_and_broadcast = True
    broadcast = ctx.array.loading.load_and_broadcast_options
    broadcast.replica_axis_index = replica_axis_index
    broadcast.broadcast_memory_limit_bytes = (
        _SINGLE_REPLICA_BROADCAST_MEMORY_LIMIT_BYTES
    )

  # Pathways colocated-python checkpointing.
  if colocated_python_checkpointing:
    ctx.pathways.checkpointing_impl = (
        ocp_pathways.CheckpointingImpl.from_options(
            use_colocated_python=True,
        )
    )

  if partial_load:
    ctx.pytree.loading.partial_load = True

  if checkpoint_layout is not None:
    ctx.checkpoint_layout = checkpoint_layout

  return ctx
