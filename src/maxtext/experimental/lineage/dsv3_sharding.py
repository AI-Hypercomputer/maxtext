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

`sharding_config` takes only the mesh: `TOPOLOGY_SHARDINGS` maps each per-slice
`(x, y, z)` topology to its axis mapping and to the physical PartitionSpecs of
every weight.
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DSv3WeightShardings:
  """Physical PartitionSpecs of every DSv3 parameter group.

  Attributes:
    decoder: Specs of the scanned dense and sparse decoder stacks.
    mtp: Specs of one (unscanned) MTP layer.
    embed_table: Spec of the token embedding table `[vocab, emb]`.
    final_norm_scale: Spec of the final RMSNorm scale `[emb]`, also used for the
      per-depth MTP final norm scales.
    head_kernel: Spec of the LM head kernel `[emb, vocab]`.
  """

  decoder: dsv3_types.DSv3WeightsPytree[Any]
  mtp: dsv3_types.DSv3MTPWeightsPytree[Any]
  embed_table: P
  final_norm_scale: P
  head_kernel: P


@dataclasses.dataclass(frozen=True)
class DSv3ShardingConfig:
  """Axis mapping plus the specs of all weights and activations.

  Attributes:
    axis_mapping: Lineage logical-to-physical axis mapping.
    weights: Physical specs of every parameter group.
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


@dataclasses.dataclass(frozen=True)
class TopologySharding:
  """Shardings of one per-slice topology, an entry of `TOPOLOGY_SHARDINGS`.

  Attributes:
    axis_mapping: Lineage logical-to-physical axis mapping of every logical axis
      but `dcn`, which `sharding_config` adds.
    weights: Physical specs of every parameter group.
  """

  axis_mapping: AxisMapping
  weights: DSv3WeightShardings


def drop_layer_axis(specs: Any) -> Any:
  """Drops the leading (layer) entry of every PartitionSpec in `specs`.

  An MTP layer is one unscanned DeepSeek MoE layer behind the `eh_proj`
  projection, so its sparse weights use the decoder's sparse specs minus the
  layer axis.

  Args:
    specs: A pytree of PartitionSpecs of scanned weights.

  Returns:
    `specs`, with the leading entry of every PartitionSpec dropped.
  """
  return jax.tree.map(lambda spec: P(*spec[1:]), specs, is_leaf=lambda x: isinstance(x, P))


# 8x8x8: MLA heads on core, EP over (x, core), FSDP over (y, z).
_MLA_8X8X8 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, ("y", "z"), None),
    # q_up carries its FSDP axes on its q_lora dim: 64 ways divide its 192-wide
    # head dim but not the 32-wide even/odd rope halves that
    # `init_dsv3_mla_weights` slices out of it, and JAX cannot slice a sharded
    # dim into pieces its mesh axes do not divide.
    q_up=P(None, ("y", "z"), "core", None),
    q_norm_scale=P(None, None),
    kv_down=P(None, ("y", "z"), None),
    k_up=P(None, None, "core", ("y", "z")),
    v_up=P(None, None, "core", ("y", "z")),
    kv_norm_scale=P(None, None),
    out=P(None, "core", None, ("y", "z")),
)
_SPARSE_8X8X8 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None),
    mla=_MLA_8X8X8,
    post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core"), ("y", "z"), None),
            # The routed `linear` ([layer, expert, hidden, embed]) is sharded on
            # its hidden dim, as in dsv3_perf_test. Sharding it on the embed dim
            # instead makes the in-loop all-gather over `fsdp_moe` land in a
            # transposed layout that costs a relayout copy at every use and
            # before the backward reduce-scatter.
            linear=P(None, ("x", "core"), ("y", "z"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("y", "z")),
            gate_1=P(None, None, ("y", "z")),
            linear=P(None, ("y", "z"), None),
        ),
    ),
)

# 8x16x16: the 8x8x8 layout, except where 256-way FSDP over (y, z) does not fit.
_MLA_8X16X16 = dsv3_types.DSv3MLAWeightsPytree(
    # The down-projections are FSDP-sharded over z only: 256 ways would leave
    # 7168 / 256 = 28-row shards of the embedding dim, not whole 8-row TPU
    # tiles, and XLA cannot lower the SparseCore all-gather of such shards.
    q_down=P(None, "z", None),
    # The up-projections carry their FSDP axes on their LoRA dim: 256 ways
    # divide none of their head dims.
    q_up=P(None, ("y", "z"), "core", None),
    q_norm_scale=P(None, None),
    kv_down=P(None, "z", None),
    k_up=P(None, ("y", "z"), "core", None),
    v_up=P(None, ("y", "z"), "core", None),
    kv_norm_scale=P(None, None),
    out=P(None, "core", None, ("y", "z")),
)
_SPARSE_8X16X16 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None),
    mla=_MLA_8X16X16,
    post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            # 256-way FSDP over (y, z) would leave 7168 / 256 = 28-row gate
            # shards, not whole 8-row TPU tiles: the SparseCore all-gather
            # lands them with the embedding dim outermost, and XLA relayouts
            # every gathered gate and gate gradient. y shards the expert dim
            # instead, after the expert axes, splitting the 16 local experts
            # into one per device, and z leaves 448-row gate shards.
            # `ops.collect_along_axis` gathers both dims with one all-gather.
            gate=P(None, ("x", "core", "y"), "z", None),
            # On its hidden dim, as on 8x8x8, with gate's expert axes so both
            # routed weights share one expert layout.
            linear=P(None, ("x", "core", "y"), "z", None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("y", "z")),
            gate_1=P(None, None, ("y", "z")),
            linear=P(None, ("y", "z"), None),
        ),
    ),
)

# 8x16x32: the 8x16x16 layout, except where 512-way FSDP over (y, z) does not
# fit whole 8-row TPU tiles.
_SPARSE_8X16X32 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None),
    mla=_MLA_8X16X16,
    post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core", "y"), "z", None),
            linear=P(None, ("x", "core", "y"), "z", None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            # 512-way FSDP over (y, z) would leave 2048 / 512 = 4-row shards
            # (0.5 tiles), not whole 8-row TPU tiles. Shard over z only
            # (64-row shards, 8 tiles).
            gate_0=P(None, None, "z"),
            gate_1=P(None, None, "z"),
            linear=P(None, "z", None),
        ),
    ),
)

# 8x32x32: 16384-way ICI with EP over (x, core) and 1024-way FSDP over (y, z).
# Routed expert weights merge `(expert, rows) -> expert * rows` (`[L, E*rows,
# cols]`) and shard all four `(x, core, y, z)` axes on that single row axis so
# every shard is an exact T(8, 128) tile (`[112, 4096]` gate, `[32, 7168]`
# linear) and SparseCore issues one unified 2D `(y, z)` collective.
_MLA_8X32X32 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, "z", None),
    q_up=P(None, "z", "core", None),
    q_norm_scale=P(None, None),
    kv_down=P(None, "z", None),
    k_up=P(None, "z", "core", None),
    v_up=P(None, "z", "core", None),
    kv_norm_scale=P(None, None),
    out=P(None, "core", None, "z"),
)
_SPARSE_8X32X32 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None),
    mla=_MLA_8X32X32,
    post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core", "y", "z"), None),
            linear=P(None, ("x", "core", "y", "z"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, "z", None),
            gate_1=P(None, "z", None),
            linear=P(None, "z", None),
        ),
    ),
)

# 4x8x8: the 8x8x8 layout with EP over (z, core) and FSDP over (x, y), so each
# device keeps 16 experts and the collectives match 8x8x8's. The slice must be
# a plain (untwisted) torus, which the launchers request explicitly.
_MLA_4X8X8 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, ("x", "y"), None),
    # On its q_lora dim, as on 8x8x8.
    q_up=P(None, ("x", "y"), "core", None),
    q_norm_scale=P(None, None),
    kv_down=P(None, ("x", "y"), None),
    k_up=P(None, None, "core", ("x", "y")),
    v_up=P(None, None, "core", ("x", "y")),
    kv_norm_scale=P(None, None),
    out=P(None, "core", None, ("x", "y")),
)
_SPARSE_4X8X8 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None),
    mla=_MLA_4X8X8,
    post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("z", "core"), ("x", "y"), None),
            # On its hidden dim, as on 8x8x8.
            linear=P(None, ("z", "core"), ("x", "y"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("x", "y")),
            gate_1=P(None, None, ("x", "y")),
            linear=P(None, ("x", "y"), None),
        ),
    ),
)

# 4x4x8: the 4x8x8 layout, except q_up carries its 16-way FSDP over (x, y) on
# its 192-wide head dim (as k_up and v_up do): 16 ways divide the 32-wide
# even/odd rope halves in `init_dsv3_mla_weights`, whereas sharding q_lora
# (1536 / 16 = 96) pads the 128-wide minor TPU tile dim by 33%.
_MLA_4X4X8 = dataclasses.replace(_MLA_4X8X8, q_up=P(None, None, "core", ("x", "y")))
_SPARSE_4X4X8 = dataclasses.replace(_SPARSE_4X8X8, mla=_MLA_4X4X8)

# Per-slice `(x, y, z)` topology -> shardings. A multi-slice mesh uses the entry
# of its slices; `sharding_config` adds data parallelism over `dcn`.
TOPOLOGY_SHARDINGS: dict[tuple[int, int, int], TopologySharding] = {
    (4, 4, 8): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("z", "core"),
            "fsdp_moe": ("x", "y"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_4X4X8,
                    post_attn_norm_scale=P(None, None),
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, None, ("x", "y")),
                        gate_1=P(None, None, ("x", "y")),
                        linear=P(None, ("x", "y"), None),
                    ),
                ),
                sparse=_SPARSE_4X4X8,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    eh_proj=P(("x", "y"), None),
                ),
                sparse=drop_layer_axis(_SPARSE_4X4X8),
            ),
            # Over every ICI axis, as on 8x8x8: 256 ways divide the 7168-wide
            # embedding dim.
            embed_table=P(None, ("x", "y", "z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("x", "y", "z", "core"), None),
        ),
    ),
    (4, 8, 8): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("z", "core"),
            "fsdp_moe": ("x", "y"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_4X8X8,
                    post_attn_norm_scale=P(None, None),
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, None, ("x", "y")),
                        gate_1=P(None, None, ("x", "y")),
                        linear=P(None, ("x", "y"), None),
                    ),
                ),
                sparse=_SPARSE_4X8X8,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    eh_proj=P(("x", "y"), None),
                ),
                sparse=drop_layer_axis(_SPARSE_4X8X8),
            ),
            # Over every ICI axis, as on 8x8x8: 512 ways divide the 7168-wide
            # embedding dim.
            embed_table=P(None, ("x", "y", "z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("x", "y", "z", "core"), None),
        ),
    ),
    (8, 8, 8): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("x", "core"),
            "fsdp_moe": ("y", "z"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_8X8X8,
                    post_attn_norm_scale=P(None, None),
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, None, ("y", "z")),
                        gate_1=P(None, None, ("y", "z")),
                        linear=P(None, ("y", "z"), None),
                    ),
                ),
                sparse=_SPARSE_8X8X8,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    # [2 * emb, emb]: FSDP only, as in dsv3_perf_test.
                    eh_proj=P(("y", "z"), None),
                ),
                sparse=drop_layer_axis(_SPARSE_8X8X8),
            ),
            # The embedding table `[vocab, emb]` and LM head kernel `[emb,
            # vocab]` are sharded on the embedding dim over every ICI axis,
            # matching MaxText's `embed_vocab` logical axis for this recipe.
            embed_table=P(None, ("x", "y", "z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("x", "y", "z", "core"), None),
        ),
    ),
    (8, 16, 16): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("x", "core"),
            "fsdp_moe": ("y", "z"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_8X16X16,
                    post_attn_norm_scale=P(None, None),
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, None, ("y", "z")),
                        gate_1=P(None, None, ("y", "z")),
                        linear=P(None, ("y", "z"), None),
                    ),
                ),
                sparse=_SPARSE_8X16X16,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    eh_proj=P(("y", "z"), None),
                ),
                sparse=drop_layer_axis(_SPARSE_8X16X16),
            ),
            # Over every ICI axis but x: 4096 ways do not divide the 7168-wide
            # embedding dim.
            embed_table=P(None, ("y", "z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("y", "z", "core"), None),
        ),
    ),
    (8, 16, 32): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("x", "core"),
            "fsdp_moe": ("y", "z"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_8X16X16,
                    post_attn_norm_scale=P(None, None),
                    # 512-way FSDP over (y, z) would leave 18432 / 512 = 36-row
                    # shards (4.5 tiles), not whole 8-row TPU tiles. Shard over
                    # z only (576-row shards, 72 tiles).
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, None, "z"),
                        gate_1=P(None, None, "z"),
                        linear=P(None, "z", None),
                    ),
                ),
                sparse=_SPARSE_8X16X32,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    eh_proj=P(("y", "z"), None),
                ),
                sparse=drop_layer_axis(_SPARSE_8X16X32),
            ),
            # Over every ICI axis but x: 8192 ways do not divide the 7168-wide
            # embedding dim. 1024 ways over (y, z, core) divide it (7 rows).
            embed_table=P(None, ("y", "z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("y", "z", "core"), None),
        ),
    ),
    (8, 32, 32): TopologySharding(
        axis_mapping={
            "attention": "core",
            "fsdp_attention": ("x", "y", "z"),
            "expert": ("x", "core"),
            "fsdp_moe": ("y", "z"),
        },
        weights=DSv3WeightShardings(
            decoder=dsv3_types.DSv3WeightsPytree(
                dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                    pre_attn_norm_scale=P(None, None),
                    mla=_MLA_8X32X32,
                    post_attn_norm_scale=P(None, None),
                    mlp=dsv3_types.DSv3MLPWeightsPytree(
                        gate_0=P(None, "z", None),
                        gate_1=P(None, "z", None),
                        linear=P(None, "z", None),
                    ),
                ),
                sparse=_SPARSE_8X32X32,
            ),
            mtp=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=P(None),
                    hnorm_scale=P(None),
                    eh_proj=P("z", None),
                ),
                sparse=drop_layer_axis(_SPARSE_8X32X32),
            ),
            # 64 ways over (z, core) leave 7168 / 64 = 112 = 14 * 8-row tiles.
            embed_table=P(None, ("z", "core")),
            final_norm_scale=P(None),
            head_kernel=P(("z", "core"), None),
        ),
    ),
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
) -> DSv3ShardingConfig:
  """Looks up the `DSv3ShardingConfig` of `mesh`'s per-slice topology.

  Args:
    mesh: The device mesh: `x`, `y`, `z` and `core` axes, plus a leading `dcn`
      axis across slices if any.

  Returns:
    The sharding config of the `TOPOLOGY_SHARDINGS` entry for the mesh's
    `(x, y, z)` shape. Its axis mapping maps `dcn` to the `dcn` mesh axis, or
    to the empty tuple (a no-op) if the mesh has none.

  Raises:
    ValueError: If `TOPOLOGY_SHARDINGS` has no entry for that shape.
  """
  topology = (mesh.shape["x"], mesh.shape["y"], mesh.shape["z"])
  entry = TOPOLOGY_SHARDINGS.get(topology)
  if entry is None:
    raise ValueError(f"No DSv3 shardings for per-slice topology {topology}; supported:" f" {sorted(TOPOLOGY_SHARDINGS)}.")
  return DSv3ShardingConfig(
      axis_mapping={
          "dcn": DCN_AXIS if DCN_AXIS in mesh.axis_names else (),
          **entry.axis_mapping,
      },
      weights=entry.weights,
      activation=activation_sharding(mesh),
      segment_ids=segment_ids_sharding(mesh),
      token=token_sharding(mesh),
      yarn_freqs=yarn_freqs_sharding(mesh),
  )


def _axis_names(v: str | Sequence[str]) -> tuple[str, ...]:
  return (v,) if isinstance(v, str) else tuple(v)


def _to_axis_tuple(v: str | Sequence[str]) -> str | tuple[str, ...]:
  if isinstance(v, str):
    return v
  return v[0] if len(v) == 1 else tuple(v)


# MLA weights whose head dim is sharded over `attention`, and its negative
# dimension index (same in the scanned decoder stacks and in an unscanned MTP
# layer).
_MLA_HEAD_DIMS = (("q_up", -2), ("k_up", -2), ("v_up", -2), ("out", -3))


def _is_mla_or_spec(x: Any) -> bool:
  return isinstance(x, (dsv3_types.DSv3MLAWeightsPytree, P))


def _reshard_mla_heads(node: Any, head_axis: str | tuple[str, ...]) -> Any:
  """Replaces the head-dim entry of the head-sharded MLA specs with `head_axis`."""
  if not isinstance(node, dsv3_types.DSv3MLAWeightsPytree):
    return node
  head_axes = _axis_names(head_axis)
  updates = {}
  for field, dim in _MLA_HEAD_DIMS:
    spec = getattr(node, field)
    if spec is None:
      continue
    entries = list(spec)
    idx = len(entries) + dim
    other_axes = set()
    for i, entry in enumerate(entries):
      if i != idx and entry is not None:
        other_axes.update(_axis_names(entry))
    overlap = [ax for ax in head_axes if ax in other_axes]
    if overlap:
      raise ValueError(
          f"The activation layout's `attention` axes {head_axis} overlap with"
          f" the MLA `{field}` FSDP axes {entries}: {overlap}."
      )
    entries[idx] = head_axis
    updates[field] = P(*entries)
  return dataclasses.replace(node, **updates)


def activation_resharded_config(
    base: DSv3ShardingConfig,
    activation_axis_mapping: AxisMapping,
) -> DSv3ShardingConfig:
  """A sharding config with new activation axes and (almost) the same weights.

  Used to evaluate with a different activation layout than training, e.g. a
  fractional batch with the expert axes acting as context parallelism:
  `attention` (the sequence axis outside attention and the MLA head axis inside
  it) and `fsdp_attention` (the batch axis) change, while `expert` and
  `fsdp_moe` must not.

  Every weight keeps the physical layout of `base`, except the MLA projections
  whose head dim follows `attention`. The new `attention` axes must start with
  `base`'s (e.g. `core` -> `(core, x)`), so every device's heads are a subset of
  its training heads and resharding the parameters of `base` to this config
  only slices the head dim locally, without communication. In particular, the
  embedding table and LM head keep `base`'s axis order.

  Args:
    base: The training sharding config.
    activation_axis_mapping: Logical-to-physical axis mapping of the new layout.

  Returns:
    The sharding config of the new layout.

  Raises:
    ValueError: If `activation_axis_mapping` lacks a logical axis, changes
      `dcn`, `expert` or `fsdp_moe`, does not shard `attention` over `base`'s
      `attention` axes as a prefix, or overlaps with the MLA FSDP axes.
  """
  missing = [axis for axis in LOGICAL_AXES if axis not in activation_axis_mapping]
  if missing:
    raise ValueError(f"activation_axis_mapping lacks logical axes {missing}.")
  for axis in ("dcn", "expert", "fsdp_moe"):
    if _to_axis_tuple(_axis_names(activation_axis_mapping[axis])) != (
        _to_axis_tuple(_axis_names(base.axis_mapping[axis]))
    ):
      raise ValueError(
          f"The activation layout must keep `{axis}`:" f" {activation_axis_mapping[axis]} != {base.axis_mapping[axis]}."
      )
  base_attention = _axis_names(base.axis_mapping["attention"])
  new_attention = _axis_names(activation_axis_mapping["attention"])
  if new_attention[: len(base_attention)] != base_attention:
    # Each device's heads must be a subset of its training heads, i.e. the
    # training axes must be the major (leading) axes of the new head sharding.
    raise ValueError(
        "The activation layout must shard `attention` over the training axes"
        f" {base.axis_mapping['attention']} as a prefix, got"
        f" {activation_axis_mapping['attention']}."
    )
  head_axis = _to_axis_tuple(new_attention)
  weights = dataclasses.replace(
      base.weights,
      decoder=jax.tree.map(
          lambda node: _reshard_mla_heads(node, head_axis),
          base.weights.decoder,
          is_leaf=_is_mla_or_spec,
      ),
      mtp=jax.tree.map(
          lambda node: _reshard_mla_heads(node, head_axis),
          base.weights.mtp,
          is_leaf=_is_mla_or_spec,
      ),
  )
  return dataclasses.replace(
      base,
      axis_mapping={axis: _to_axis_tuple(_axis_names(activation_axis_mapping[axis])) for axis in LOGICAL_AXES},
      weights=weights,
  )


def expert_axes_as_cp_axis_mapping(axis_mapping: AxisMapping) -> AxisMapping:
  """The axis mapping that uses the expert axes as context parallelism.

  For evaluating a fractional batch (fewer rows than batch shards): the expert
  axes that do not already shard `attention` move from the batch
  (`fsdp_attention`) to the sequence and MLA heads (`attention`), appended
  after the training `attention` axes. Outside attention each device then holds
  a shorter sequence slice of fewer rows; inside MLA the low-rank q/kv latents
  are all-gathered along the longer `attention` group as in training, and each
  device holds the full sequence for fewer heads. `expert` and `fsdp_moe` are
  unchanged, so the MoE keeps its expert parallelism and FSDP.

  E.g. the recipe layout `attention=core`, `fsdp_attention=(x, y, z)`,
  `expert=(x, core)` becomes `attention=(core, x)`, `fsdp_attention=(y, z)`.

  Args:
    axis_mapping: The training axis mapping.

  Returns:
    The new axis mapping, for `activation_resharded_config`.

  Raises:
    ValueError: If an expert axis that does not shard `attention` does not shard
      `fsdp_attention` either, i.e. moving it would not shorten the batch.
  """
  attention = _axis_names(axis_mapping["attention"])
  fsdp_attention = _axis_names(axis_mapping["fsdp_attention"])
  moved = tuple(ax for ax in _axis_names(axis_mapping["expert"]) if ax not in attention)
  missing = [ax for ax in moved if ax not in fsdp_attention]
  if missing:
    raise ValueError(
        f"Expert axes {missing} shard neither `attention` {attention} nor"
        f" `fsdp_attention` {fsdp_attention}; they cannot become context"
        " parallelism."
    )
  return {
      **axis_mapping,
      "attention": _to_axis_tuple(attention + moved),
      "fsdp_attention": _to_axis_tuple(tuple(ax for ax in fsdp_attention if ax not in moved)),
  }


def expert_axes_as_cp_config(base: DSv3ShardingConfig) -> DSv3ShardingConfig:
  """`base` with the expert axes as context parallelism for the activations.

  The eval layout of a fractional eval batch: the activations follow
  `expert_axes_as_cp_axis_mapping`, the weights keep `base`'s physical layout
  (only the MLA head dim is sliced further, locally; see
  `activation_resharded_config`).

  Batch split needs an even local batch on every batch shard, so the eval batch
  needs at least 2 rows per shard of the new `fsdp_attention` (and `dcn`) axes,
  e.g. 2 * |y * z| rows on an [x, y, z, core] mesh with the recipe layout.

  Args:
    base: The training sharding config.

  Returns:
    The eval sharding config.
  """
  return activation_resharded_config(base, expert_axes_as_cp_axis_mapping(base.axis_mapping))


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
