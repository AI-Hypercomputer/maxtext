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

"""Sharding specs, axis mapping and physical mesh construction for DSv3.

The Lineage DSv3 layers shard weights and activations over five *logical*
axes:

* `dcn`: data parallelism across slices (weights replicated, batch split).
* `fsdp_attention`: FSDP axes of the attention/dense weights and the batch axis
  of the activations.
* `attention`: head sharding of the MLA projections and the sequence axis of
  the activations outside attention.
* `expert`: expert parallelism of the routed experts.
* `fsdp_moe`: FSDP axes of the MoE weights.

An `axis_mapping` translates these names to physical mesh axes
(`ops.physical_pspec`). Physical names (`"z"`, `"dcn"`) may also appear in a
spec directly; `ops.physical_pspec` passes unknown names through.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
from typing import Any

import jax
from jax.experimental import mesh_utils
import numpy as np

from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops

P = jax.sharding.PartitionSpec
AxisMapping = Mapping[str, str | tuple[str, ...]]

DCN_AXIS = "dcn"
# Logical axis names consumed by the Lineage layers.
LOGICAL_AXES = ("dcn", "attention", "fsdp_attention", "expert", "fsdp_moe")
# Physical FSDP axis of the 2-D attention/dense weights (`dsv3_perf_test`).
WEIGHT_FSDP_AXIS = "z"

MLA_WEIGHT_SHARDINGS = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, WEIGHT_FSDP_AXIS, None),
    # As in dsv3_perf_test; `fit_mla_up_shardings` moves the FSDP axis of the
    # up-projections to their LoRA dim when z does not divide the head dim.
    q_up=P(None, None, "attention", WEIGHT_FSDP_AXIS),
    q_norm_scale=P(None, None),
    kv_down=P(None, WEIGHT_FSDP_AXIS, None),
    k_up=P(None, None, "attention", WEIGHT_FSDP_AXIS),
    v_up=P(None, None, "attention", WEIGHT_FSDP_AXIS),
    kv_norm_scale=P(None, None),
    out=P(None, "attention", None, WEIGHT_FSDP_AXIS),
)

DECODER_WEIGHT_SHARDINGS = dsv3_types.DSv3WeightsPytree(
    dense=dsv3_types.DSv3DenseLayerWeightsPytree(
        pre_attn_norm_scale=P(None, None),
        mla=MLA_WEIGHT_SHARDINGS,
        post_attn_norm_scale=P(None, None),
        mlp=dsv3_types.DSv3MLPWeightsPytree(
            gate_0=P(None, None, "fsdp_moe"),
            gate_1=P(None, None, "fsdp_moe"),
            linear=P(None, "fsdp_moe", None),
        ),
    ),
    sparse=dsv3_types.DSv3SparseLayerWeightsPytree(
        pre_attn_norm_scale=P(None, None),
        mla=MLA_WEIGHT_SHARDINGS,
        post_attn_norm_scale=P(None, None),
        moe=dsv3_types.DSv3MoEWeightsPytree(
            router=dsv3_types.DSv3MoERouterWeightsPytree(
                kernel=P(),
                bias=P(),
            ),
            routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
                gate=P(None, "expert", "fsdp_moe", None),
                # The routed `linear` ([layer, expert, hidden, embed]) is
                # sharded on its hidden dim, as in dsv3_perf_test. Sharding it
                # on the embed dim instead makes the in-loop all-gather over
                # `fsdp_moe` land in a transposed layout that costs a relayout
                # copy at every use and before the backward reduce-scatter.
                linear=P(None, "expert", "fsdp_moe", None),
            ),
            shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
                gate_0=P(None, None, "fsdp_moe"),
                gate_1=P(None, None, "fsdp_moe"),
                linear=P(None, "fsdp_moe", None),
            ),
        ),
    ),
)

# An MTP layer is one unscanned DeepSeek MoE layer behind the `eh_proj`
# projection, so its sparse weights reuse the stack specs minus the layer axis.
MTP_WEIGHT_SHARDINGS = dsv3_types.DSv3MTPWeightsPytree(
    ehproj=dsv3_types.DSv3EHProjWeightsPytree(
        enorm_scale=P(None),
        hnorm_scale=P(None),
        # [2 * emb, emb]. FSDP only, as in dsv3_perf_test: 14336 rows do not
        # divide over a whole 4x4x128 slice (fsdp_attention x attention = 4096).
        eh_proj=P(WEIGHT_FSDP_AXIS, None),
    ),
    sparse=jax.tree.map(
        lambda spec: P(*spec[1:]),
        DECODER_WEIGHT_SHARDINGS.sparse,
        is_leaf=lambda x: isinstance(x, P),
    ),
)

# Embedding table `[vocab, emb]`, final norm scale `[emb]` and LM head kernel
# `[emb, vocab]`: sharded on the embedding dim over every ICI axis, matching
# MaxText's `embed_vocab` logical axis for this recipe. `fit_embed_shardings`
# drops leading axes on meshes too large for the embedding dim.
EMBED_TABLE_SHARDING = P(None, ("fsdp_attention", "attention"))
FINAL_NORM_SCALE_SHARDING = P(None)
HEAD_KERNEL_SHARDING = P(("fsdp_attention", "attention"), None)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DSv3WeightShardings:
  """Logical PartitionSpecs of every DSv3 parameter group.

  Attributes:
    decoder: Specs of the scanned dense and sparse decoder stacks.
    mtp: Specs of one (unscanned) MTP layer.
    embed_table: Spec of the token embedding table `[vocab, emb]`.
    final_norm_scale: Spec of the final RMSNorm scale `[emb]`, also used for the
      per-depth MTP final norm scales.
    head_kernel: Spec of the LM head kernel `[emb, vocab]`.
  """

  decoder: dsv3_types.DSv3WeightsPytree[Any] = dataclasses.field(default_factory=lambda: DECODER_WEIGHT_SHARDINGS)
  mtp: dsv3_types.DSv3MTPWeightsPytree[Any] = dataclasses.field(default_factory=lambda: MTP_WEIGHT_SHARDINGS)
  embed_table: P = EMBED_TABLE_SHARDING
  final_norm_scale: P = FINAL_NORM_SCALE_SHARDING
  head_kernel: P = HEAD_KERNEL_SHARDING


@dataclasses.dataclass(frozen=True)
class DSv3ShardingConfig:
  """Axis mapping plus the logical specs of all weights and activations.

  Attributes:
    axis_mapping: Lineage logical-to-physical axis mapping.
    weights: Logical specs of every parameter group.
    activation: Spec of `[batch, seq, emb]` activations (embeddings, hidden
      states, logits).
    segment_ids: Spec of `[batch, seq]` segment ids entering the layers.
    token: Spec of per-token `[batch, seq]` arrays (losses, token ids).
    yarn_freqs: Spec of the `[batch, seq, 1, rope]` YaRN frequencies.
  """

  axis_mapping: AxisMapping
  weights: DSv3WeightShardings
  activation: P
  segment_ids: P
  token: P
  yarn_freqs: P


def _move_up_fsdp_to_lora(node: Any, fields: Sequence[str], fsdp_axis: str) -> Any:
  """Puts the FSDP axis of `fields` on their LoRA dim in an MLA spec pytree."""
  if not isinstance(node, dsv3_types.DSv3MLAWeightsPytree):
    return node
  updates = {}
  for field in fields:
    spec = getattr(node, field)
    if spec is None:
      continue
    layer_axes = (None,) * (len(spec) - 3)  # Empty for an MTP layer.
    updates[field] = P(*layer_axes, fsdp_axis, "attention", None)
  return dataclasses.replace(node, **updates)


def fit_mla_up_shardings(
    specs: Any,
    mesh_shape: Mapping[str, int],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    v_head_dim: int,
    fsdp_axis: str = WEIGHT_FSDP_AXIS,
) -> Any:
  """Moves an up-projection's FSDP axis to its LoRA dim if it does not divide.

  `q_up` (`[layer, q_lora, head, qk_head_dim + rope_head_dim]`), `k_up`
  (`[layer, kv_lora, head, qk_head_dim]`) and `v_up` (`[layer, kv_lora, head,
  v_head_dim]`) shard their trailing head dim over FSDP, as in dsv3_perf_test.
  When the FSDP axis does not divide that dim (e.g. z = 128 for q_up's 192, z =
  256 for k_up's 128), it moves to the LoRA dim instead.

  Args:
    specs: Weight PartitionSpecs (`DECODER_WEIGHT_SHARDINGS`,
      `MTP_WEIGHT_SHARDINGS` or a `DSv3WeightShardings`).
    mesh_shape: Mesh axis sizes, `mesh.shape`.
    qk_head_dim: Non-positional q/k head dim.
    rope_head_dim: RoPE head dim.
    v_head_dim: Value head dim.
    fsdp_axis: Physical FSDP axis of the up-projections.

  Returns:
    `specs`, with the FSDP axes moved to the LoRA dims where needed.
  """
  head_dims = {
      "q_up": qk_head_dim + rope_head_dim,
      "k_up": qk_head_dim,
      "v_up": v_head_dim,
  }
  fields = tuple(field for field, dim in head_dims.items() if dim % mesh_shape[fsdp_axis] != 0)
  if not fields:
    return specs
  if isinstance(specs, DSv3WeightShardings):
    return dataclasses.replace(
        specs,
        decoder=fit_mla_up_shardings(
            specs.decoder,
            mesh_shape,
            qk_head_dim=qk_head_dim,
            rope_head_dim=rope_head_dim,
            v_head_dim=v_head_dim,
            fsdp_axis=fsdp_axis,
        ),
        mtp=fit_mla_up_shardings(
            specs.mtp,
            mesh_shape,
            qk_head_dim=qk_head_dim,
            rope_head_dim=rope_head_dim,
            v_head_dim=v_head_dim,
            fsdp_axis=fsdp_axis,
        ),
    )
  return jax.tree.map(
      lambda node: _move_up_fsdp_to_lora(node, fields, fsdp_axis),
      specs,
      is_leaf=lambda x: isinstance(x, (dsv3_types.DSv3MLAWeightsPytree, P)),
  )


def _fit_spec_dim(
    spec: P,
    dim_index: int,
    dim: int,
    mesh_shape: Mapping[str, int],
    axis_mapping: AxisMapping,
) -> P:
  """Drops leading mesh axes of one spec entry until they divide `dim`.

  Args:
    spec: Logical PartitionSpec.
    dim_index: Index of the entry to fit.
    dim: Size of that array dimension.
    mesh_shape: Mesh axis sizes, `mesh.shape`.
    axis_mapping: Lineage logical-to-physical axis mapping.

  Returns:
    `spec` itself if its physical axes already divide `dim`; otherwise the
    physical spec with the largest trailing run of those axes that divides it.
  """
  physical = ops.physical_pspec(spec, axis_mapping)
  entry = physical[dim_index]
  axes = (entry,) if isinstance(entry, str) else tuple(entry or ())
  kept = axes
  while dim % math.prod(mesh_shape[axis] for axis in kept) != 0:
    kept = kept[1:]  # Terminates: the empty product is 1.
  if kept == axes:
    return spec
  entries = list(physical)
  entries[dim_index] = _to_axis_tuple(kept) if kept else None
  return P(*entries)


def fit_embed_shardings(
    specs: DSv3WeightShardings,
    mesh_shape: Mapping[str, int],
    axis_mapping: AxisMapping,
    *,
    emb_dim: int,
) -> DSv3WeightShardings:
  """Fits the embedding-dim sharding of the embedding table and LM head.

  Both shard the embedding dim over every ICI axis, which does not divide it on
  large meshes (7168 = 2^10 * 7 over more than 1024 devices). There, the
  leading mesh axes are dropped until the rest divides `emb_dim`; the weights
  are replicated over the dropped axes.

  Args:
    specs: Weight PartitionSpecs.
    mesh_shape: Mesh axis sizes, `mesh.shape`.
    axis_mapping: Lineage logical-to-physical axis mapping.
    emb_dim: The embedding dim.

  Returns:
    `specs`, with `embed_table` and `head_kernel` fitted.
  """
  return dataclasses.replace(
      specs,
      embed_table=_fit_spec_dim(specs.embed_table, 1, emb_dim, mesh_shape, axis_mapping),
      head_kernel=_fit_spec_dim(specs.head_kernel, 0, emb_dim, mesh_shape, axis_mapping),
  )


def _to_axis_tuple(v: str | Sequence[str]) -> str | tuple[str, ...]:
  if isinstance(v, str):
    return v
  return v[0] if len(v) == 1 else tuple(v)


def build_axis_mapping(rules: Mapping[str, str | Sequence[str]], *, has_dcn: bool) -> AxisMapping:
  """Builds the Lineage logical-to-physical axis mapping from MaxText rules.

  Args:
    rules: MaxText logical axis rules (`dict(cfg.logical_axis_rules)`). Reads
      `activation_length` (-> `attention`), `activation_batch_attn` (->
      `fsdp_attention`), `exp` (-> `expert`) and `embed_moe` (-> `fsdp_moe`).
    has_dcn: Whether the mesh carries a leading `dcn` axis. If so, the logical
      `dcn` axis maps to it; otherwise it maps to the empty tuple, which
      `ops.physical_pspec` / `ops.collect_along_axis` treat as a no-op.

  Returns:
    The axis mapping.
  """
  return {
      "dcn": DCN_AXIS if has_dcn else (),
      "attention": _to_axis_tuple(rules["activation_length"]),
      "fsdp_attention": _to_axis_tuple(rules["activation_batch_attn"]),
      "expert": _to_axis_tuple(rules["exp"]),
      "fsdp_moe": _to_axis_tuple(rules["embed_moe"]),
  }


def dcn_batch_axis(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis: str | tuple[str, ...],
) -> str | tuple[str, ...]:
  """Prefixes a logical batch axis with the physical DCN mesh axis, if any.

  Multi-slice runs place one slice per index of a leading `dcn` mesh axis and
  use it for pure data parallelism: activations split their batch across
  slices while weights stay replicated. Only the *activation* specs handed to
  Lineage pick up the axis; `axis_mapping` keeps `fsdp_attention` / `expert` /
  `fsdp_moe` confined to the ICI axes of a single slice.

  Args:
    mesh: The device mesh.
    axis: Logical batch axis name(s).

  Returns:
    `axis`, prefixed with `dcn` when the mesh has that axis.
  """
  if DCN_AXIS not in mesh.axis_names:
    return axis
  axes = (axis,) if isinstance(axis, str) else tuple(axis)
  return (DCN_AXIS,) + axes


def activation_sharding(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> P:
  """Logical spec of `[batch, seq, emb]` activations entering the layers."""
  return P(dcn_batch_axis(mesh, "fsdp_attention"), "attention", None)


def segment_ids_sharding(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> P:
  """Logical spec of `[batch, seq]` segment ids entering the layers."""
  return P(dcn_batch_axis(mesh, "fsdp_attention"), None)


def token_sharding(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> P:
  """Logical spec of per-token `[batch, seq]` arrays (losses, token ids)."""
  return P(dcn_batch_axis(mesh, "fsdp_attention"), "attention")


def yarn_freqs_sharding(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> P:
  """Logical spec of the `[batch, seq, 1, rope]` YaRN frequencies."""
  return P(dcn_batch_axis(mesh, "fsdp_attention"), None, None)


def sharding_config(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: AxisMapping,
    *,
    emb_dim: int,
    qk_head_dim: int,
    rope_head_dim: int,
    v_head_dim: int,
    weights: DSv3WeightShardings | None = None,
) -> DSv3ShardingConfig:
  """Assembles the `DSv3ShardingConfig` for `mesh`.

  Args:
    mesh: The device mesh.
    axis_mapping: Lineage logical-to-physical axis mapping (see
      `build_axis_mapping`).
    emb_dim: Embedding dim (for `fit_embed_shardings`).
    qk_head_dim: Non-positional q/k head dim (for `fit_mla_up_shardings`).
    rope_head_dim: RoPE head dim (for `fit_mla_up_shardings`).
    v_head_dim: Value head dim (for `fit_mla_up_shardings`).
    weights: Weight specs to start from; defaults to the dsv3_perf_test specs.

  Returns:
    The sharding config.

  Raises:
    ValueError: If `axis_mapping` lacks a logical axis.
  """
  missing = [axis for axis in LOGICAL_AXES if axis not in axis_mapping]
  if missing:
    raise ValueError(f"axis_mapping lacks logical axes {missing}.")
  fitted_weights = fit_mla_up_shardings(
      weights if weights is not None else DSv3WeightShardings(),
      mesh.shape,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      v_head_dim=v_head_dim,
  )
  fitted_weights = fit_embed_shardings(fitted_weights, mesh.shape, axis_mapping, emb_dim=emb_dim)
  return DSv3ShardingConfig(
      axis_mapping=dict(axis_mapping),
      weights=fitted_weights,
      activation=activation_sharding(mesh),
      segment_ids=segment_ids_sharding(mesh),
      token=token_sharding(mesh),
      yarn_freqs=yarn_freqs_sharding(mesh),
  )


def named_shardings(
    specs: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: AxisMapping,
) -> Any:
  """Resolves a pytree of logical PartitionSpecs to physical NamedShardings."""
  return jax.tree.map(
      lambda spec: jax.sharding.NamedSharding(mesh, ops.physical_pspec(spec, axis_mapping)),
      specs,
      is_leaf=lambda x: isinstance(x, P),
  )


def reshard(
    tree: Any,
    specs: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: AxisMapping,
) -> Any:
  """Reshards `tree` to the physical layout of the Lineage logical `specs`."""
  return jax.tree.map(jax.reshard, tree, named_shardings(specs, mesh, axis_mapping))


def _physical_coords(device: Any) -> tuple[int, ...]:
  return (*getattr(device, "coords", ()), getattr(device, "core_on_chip", 0))


def devices_by_slice(devices: Sequence[Any]) -> list[list[Any]]:
  """Groups devices by slice, ordered by ascending slice index."""
  by_slice: dict[int, list[Any]] = {}
  for device in devices:
    by_slice.setdefault(getattr(device, "slice_index", 0), []).append(device)
  return [by_slice[slice_index] for slice_index in sorted(by_slice)]


def _physical_slice_grid(slice_devices: Sequence[Any]) -> np.ndarray:
  """Sorts one slice's devices into an (x, y, z, core) grid."""
  coords_list = [_physical_coords(device) for device in slice_devices]
  if not coords_list or any(len(coords) != 4 for coords in coords_list):
    raise ValueError("Devices do not expose 3D TPU physical coordinates (x, y, z, core).")
  physical_shape = tuple(max(coords[axis] for coords in coords_list) + 1 for axis in range(4))
  if math.prod(physical_shape) != len(slice_devices):
    raise ValueError(f"Device count {len(slice_devices)} does not fill physical shape" f" {list(physical_shape)}.")
  return np.array(sorted(slice_devices, key=_physical_coords), dtype=object).reshape(physical_shape)


def create_physical_device_mesh(ici_parallelism: Sequence[int], devices: Sequence[Any]) -> np.ndarray:
  """Maps the logical mesh one-to-one onto the physical TPU torus.

  Supports both 4-entry `(x, y, z, core)` single-slice meshes and 5-entry
  `(dcn, x, y, z, core)` meshes (with a leading `1`), stacking per-slice
  physical grids along the leading DCN axis.

  Args:
    ici_parallelism: Per-axis mesh sizes, `(x, y, z, core)` or `(1, x, y, z,
      core)`.
    devices: The devices to arrange.

  Returns:
    The device array of the mesh.

  Raises:
    ValueError: If the devices do not form the requested physical topology.
  """
  ici = tuple(ici_parallelism)
  has_dcn_axis = len(ici) == 5
  if has_dcn_axis:
    if ici[0] != 1:
      raise ValueError(
          "create_physical_device_mesh requires the leading (DCN) entry of"
          f" ici_parallelism {list(ici)} to be 1; the DCN axis is sized by"
          " dcn_parallelism."
      )
    ici = ici[1:]
  elif len(ici) != 4:
    raise ValueError(
        "create_physical_device_mesh requires ici_parallelism ordered"
        f" (x, y, z, core) or (dcn, x, y, z, core), got {list(ici)}."
    )

  slices = devices_by_slice(devices)
  if len(slices) > 1 and not has_dcn_axis:
    raise ValueError(
        f"create_physical_device_mesh got devices from {len(slices)} slices"
        " but the mesh has no DCN axis; use mesh_axes ['dcn', 'x', 'y', 'z',"
        " 'core'] with ici_parallelism [1, x, y, z, core]."
    )

  grids = [_physical_slice_grid(s) for s in slices]
  for grid in grids:
    if ici != grid.shape:
      raise ValueError(
          "create_physical_device_mesh requires ici_parallelism"
          f" {list(ici)} to equal the physical (x, y, z, core)"
          f" topology {list(grid.shape)}."
      )
  if not has_dcn_axis:
    return grids[0]
  return np.stack(grids, axis=0)


def create_device_mesh(
    ici_parallelism: Sequence[int],
    devices: Sequence[Any],
    dcn_parallelism: Sequence[int] | None = None,
    allow_split_physical_axes: bool | None = None,
    *,
    log_fn: Callable[[str], Any] | None = None,
) -> np.ndarray:
  """Maps logical mesh axis i one-to-one onto physical axis i, with fallback.

  Args:
    ici_parallelism: Per-axis mesh sizes within a slice.
    devices: The devices to arrange.
    dcn_parallelism: Per-axis DCN sizes for multi-slice meshes, or None.
    allow_split_physical_axes: Forwarded to `jax.experimental.mesh_utils` on
      fallback.
    log_fn: Optional logger for the fallback message.

  Returns:
    The device array of the mesh.

  Raises:
    ValueError: If a multi-slice request does not use the `dcn`-leading mesh.
  """
  if dcn_parallelism is not None:
    expected = [len(devices_by_slice(devices))] + [1] * (len(dcn_parallelism) - 1)
    if len(ici_parallelism) != 5 or list(dcn_parallelism) != expected:
      raise ValueError(
          "Lineage across several slices requires mesh_axes"
          " ['dcn', 'x', 'y', 'z', 'core'] with ici_parallelism"
          f" [1, x, y, z, core] and dcn_parallelism {expected}; got"
          f" ici_parallelism={list(ici_parallelism)} and"
          f" dcn_parallelism={list(dcn_parallelism)}."
      )
  try:
    return create_physical_device_mesh(ici_parallelism, devices)
  except ValueError as e:
    if log_fn is not None:
      log_fn(f"Lineage: {e} Falling back to jax mesh_utils device assignment.")
    if dcn_parallelism is not None and len(devices_by_slice(devices)) > 1:
      return mesh_utils.create_hybrid_device_mesh(
          ici_parallelism,
          dcn_parallelism,
          devices,
          allow_split_physical_axes=bool(allow_split_physical_axes),
      )
    if allow_split_physical_axes is not None:
      return mesh_utils.create_device_mesh(
          ici_parallelism,
          devices,
          allow_split_physical_axes=allow_split_physical_axes,
      )
    return mesh_utils.create_device_mesh(ici_parallelism, devices)
