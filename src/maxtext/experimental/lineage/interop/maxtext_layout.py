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

"""Conversion between MaxText's and Lineage's DSv3 parameter layouts.

MaxText stores the DeepSeek-V3 parameters of its `Transformer` as a nested
dict of Flax parameter names (`decoder/dense_layers/self_attention/wq_a/kernel`,
...; see `maxtext_abstract_param_tree`). Compared to Lineage's
`dsv3_model.DSv3Params` / `dsv3_model.DSv3MutableState`:

*   the scanned decoder stacks keep their layer axis at `param_scan_axis` (dim 1
    by default), including the 1-D norm scales;
*   the RoPE channels of `wq_b` and `wkv_a` are interleaved
    (`rope_interleave: true`), where Lineage's `dsv3_mla.rotate_half` pairs
    channel `i` with channel `i + R/2`;
*   `wkv_b` fuses the key and value up-projections, which Lineage keeps apart
    (`k_up`, `v_up`);
*   the routed experts' `wi_0` / `wi_1` are separate, where Lineage fuses them
    into one `gate` matrix;
*   the routed gate bias is `[num_experts, layers]` and lives among the
    parameters, where Lineage keeps it `[layers, num_experts]` in
    `DSv3MutableState`.

`maxtext_to_lineage_params` is the whole-tree conversion that MaxText's
adapter runs once when restoring a MaxText-layout checkpoint
(`load_parameters_path`); `lineage_to_maxtext_params` is its inverse. Both are
pure and sharding-agnostic: `convert_maxtext_params` runs the former under
`jax.sharding.auto_axes` into Lineage's shardings, since slicing a sharded axis
is not allowed under explicit sharding. `maxtext_abstract_param_tree` is the
Orbax restore target: the MaxText tree of `jax.ShapeDtypeStruct`s with
shardings derived from `dsv3_model.dsv3_param_shardings`.
"""

from collections.abc import Mapping
import contextlib
import dataclasses
import functools
import math
from typing import Any, cast

import jax
import jax.numpy as jnp

from maxtext.experimental.lineage import dsv3_config
from maxtext.experimental.lineage import dsv3_embed
from maxtext.experimental.lineage import dsv3_model
from maxtext.experimental.lineage import dsv3_mtp_block
from maxtext.experimental.lineage import dsv3_types

P = jax.sharding.PartitionSpec

# MaxText's `param_scan_axis` default: the layer axis of scanned stacks.
PARAM_SCAN_AXIS = 1

Tree = dict[str, Any]


def _unbox(x: Any) -> Any:
  """Unwraps Flax / NNX variable boxes."""
  while hasattr(x, "value"):
    x = x.value
  return x


def _arr(x: jax.Array | None) -> jax.Array:
  """Narrows an optional pytree leaf to an array."""
  if x is None:
    raise ValueError("Expected an array leaf, got None.")
  return x


def _deinterleave_rope(w: jax.Array, rope_head_dim: int) -> jax.Array:
  """Trailing RoPE channels: interleaved `(2i, 2i+1)` -> split-half `(i, i+R/2)`."""
  if rope_head_dim <= 0:
    return w
  if w.shape[-1] < rope_head_dim:
    raise ValueError(f"Cannot split {rope_head_dim} RoPE channels off a trailing axis of" f" size {w.shape[-1]}.")
  nope = w[..., :-rope_head_dim]
  rope = w[..., -rope_head_dim:]
  return jnp.concatenate([nope, rope[..., 0::2], rope[..., 1::2]], axis=-1)


def _interleave_rope(w: jax.Array, rope_head_dim: int) -> jax.Array:
  """Inverse of `_deinterleave_rope`."""
  if rope_head_dim <= 0:
    return w
  nope = w[..., :-rope_head_dim]
  rope = w[..., -rope_head_dim:]
  half = rope_head_dim // 2
  interleaved = jnp.stack([rope[..., :half], rope[..., half:]], axis=-1)
  return jnp.concatenate([nope, interleaved.reshape(*rope.shape[:-1], rope_head_dim)], axis=-1)


def _to_lineage_leaf(x: Any, *, param_scan_axis: int, dtype: Any) -> jax.Array:
  """Unboxes, casts and moves the layer axis of a (scanned) MaxText leaf to 0."""
  x = jnp.asarray(_unbox(x), dtype)
  if 0 < param_scan_axis < x.ndim:
    x = jnp.moveaxis(x, param_scan_axis, 0)
  return x


def _to_maxtext_leaf(x: Any, *, param_scan_axis: int) -> jax.Array:
  """Moves the layer axis of a (scanned) Lineage leaf back to `param_scan_axis`."""
  x = jnp.asarray(x)
  if 0 < param_scan_axis < x.ndim:
    x = jnp.moveaxis(x, 0, param_scan_axis)
  return x


def _mla_from_maxtext(
    attn: Mapping[str, Any], leaf, *, qk_head_dim: int, rope_head_dim: int
) -> dsv3_types.DSv3MLAWeightsPytree:
  kv_up = leaf(attn["wkv_b"]["kernel"])
  return dsv3_types.DSv3MLAWeightsPytree(
      q_down=leaf(attn["wq_a"]["kernel"]),
      q_up=_deinterleave_rope(leaf(attn["wq_b"]["kernel"]), rope_head_dim),
      q_norm_scale=leaf(attn["q_norm"]["scale"]),
      kv_down=_deinterleave_rope(leaf(attn["wkv_a"]["kernel"]), rope_head_dim),
      k_up=kv_up[..., :qk_head_dim],
      v_up=kv_up[..., qk_head_dim:],
      kv_norm_scale=leaf(attn["kv_norm"]["scale"]),
      out=leaf(attn["out"]["kernel"]),
  )


def _mla_to_maxtext(mla: dsv3_types.DSv3MLAWeightsPytree, leaf, *, rope_head_dim: int) -> Tree:
  return {
      "wq_a": {"kernel": leaf(mla.q_down)},
      "wq_b": {"kernel": leaf(_interleave_rope(_arr(mla.q_up), rope_head_dim))},
      "q_norm": {"scale": leaf(mla.q_norm_scale)},
      "wkv_a": {"kernel": leaf(_interleave_rope(_arr(mla.kv_down), rope_head_dim))},
      "kv_norm": {"scale": leaf(mla.kv_norm_scale)},
      "wkv_b": {"kernel": leaf(jnp.concatenate([_arr(mla.k_up), _arr(mla.v_up)], axis=-1))},
      "out": {"kernel": leaf(mla.out)},
  }


def _dense_layer_from_maxtext(
    layer: Mapping[str, Any], leaf, *, qk_head_dim: int, rope_head_dim: int
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  mlp = layer["mlp"]
  return dsv3_types.DSv3DenseLayerWeightsPytree(
      pre_attn_norm_scale=leaf(layer["pre_self_attention_layer_norm"]["scale"]),
      mla=_mla_from_maxtext(
          layer["self_attention"],
          leaf,
          qk_head_dim=qk_head_dim,
          rope_head_dim=rope_head_dim,
      ),
      post_attn_norm_scale=leaf(layer["post_self_attention_layer_norm"]["scale"]),
      mlp=dsv3_types.DSv3MLPWeightsPytree(
          gate_0=leaf(mlp["wi_0"]["kernel"]),
          gate_1=leaf(mlp["wi_1"]["kernel"]),
          linear=leaf(mlp["wo"]["kernel"]),
      ),
  )


def _dense_layer_to_maxtext(w: dsv3_types.DSv3DenseLayerWeightsPytree, leaf, *, rope_head_dim: int) -> Tree:
  return {
      "pre_self_attention_layer_norm": {"scale": leaf(w.pre_attn_norm_scale)},
      "self_attention": _mla_to_maxtext(w.mla, leaf, rope_head_dim=rope_head_dim),
      "post_self_attention_layer_norm": {"scale": leaf(w.post_attn_norm_scale)},
      "mlp": {
          "wi_0": {"kernel": leaf(w.mlp.gate_0)},
          "wi_1": {"kernel": leaf(w.mlp.gate_1)},
          "wo": {"kernel": leaf(w.mlp.linear)},
      },
  }


def _sparse_layer_from_maxtext(
    layer: Mapping[str, Any],
    leaf,
    bias_leaf,
    *,
    qk_head_dim: int,
    rope_head_dim: int,
) -> tuple[dsv3_types.DSv3SparseLayerWeightsPytree, jax.Array]:
  """Returns the layer weights (router bias None) and the router bias."""
  ds_moe = layer["DeepSeekMoeBlock_0"]
  moe_block = ds_moe["MoeBlock_0"]
  shared = ds_moe["shared_experts"]
  if "wi" in moe_block:
    gate = leaf(moe_block["wi"])
  else:
    gate = jnp.concatenate([leaf(moe_block["wi_0"]), leaf(moe_block["wi_1"])], axis=-1)
  weights = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=leaf(layer["pre_self_attention_layer_norm"]["scale"]),
      mla=_mla_from_maxtext(
          layer["self_attention"],
          leaf,
          qk_head_dim=qk_head_dim,
          rope_head_dim=rope_head_dim,
      ),
      post_attn_norm_scale=leaf(layer["post_self_attention_layer_norm"]["scale"]),
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=leaf(moe_block["gate"]["kernel"]), bias=None),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=gate, linear=leaf(moe_block["wo"])),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
              gate_0=leaf(shared["wi_0"]["kernel"]),
              gate_1=leaf(shared["wi_1"]["kernel"]),
              linear=leaf(shared["wo"]["kernel"]),
          ),
      ),
  )
  return weights, bias_leaf(moe_block["gate"]["bias"])


def _sparse_layer_to_maxtext(
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    bias: jax.Array,
    leaf,
    *,
    rope_head_dim: int,
) -> Tree:
  """Inverse of `_sparse_layer_from_maxtext`: one sparse stack in MaxText names."""
  gate = _arr(w.moe.routed.gate)
  hidden = gate.shape[-1] // 2
  # In MaxText NNX (`nnx_scan.create_scanned_layers`) and on-disk Orbax
  # checkpoints, `gate.bias` (`MoEBiasVar`, non-`Param`) is stacked at axis 0
  # (`[num_layers, num_experts]`) regardless of `param_scan_axis`.
  return {
      "pre_self_attention_layer_norm": {"scale": leaf(w.pre_attn_norm_scale)},
      "self_attention": _mla_to_maxtext(w.mla, leaf, rope_head_dim=rope_head_dim),
      "post_self_attention_layer_norm": {"scale": leaf(w.post_attn_norm_scale)},
      "DeepSeekMoeBlock_0": {
          "MoeBlock_0": {
              "gate": {"kernel": leaf(w.moe.router.kernel), "bias": _arr(bias)},
              "wi_0": leaf(gate[..., :hidden]),
              "wi_1": leaf(gate[..., hidden:]),
              "wo": leaf(w.moe.routed.linear),
          },
          "shared_experts": {
              "wi_0": {"kernel": leaf(w.moe.shared.gate_0)},
              "wi_1": {"kernel": leaf(w.moe.shared.gate_1)},
              "wo": {"kernel": leaf(w.moe.shared.linear)},
          },
      },
  }


def maxtext_to_lineage_params(
    tree: Mapping[str, Any],
    cfg: dsv3_config.DSv3Config,
    *,
    param_scan_axis: int = PARAM_SCAN_AXIS,
) -> tuple[
    dsv3_model.DSv3Params[dsv3_types.ArrayType],
    dsv3_model.DSv3MutableState[dsv3_types.ArrayType],
]:
  """Converts a MaxText-layout parameter tree to Lineage layout.

  Pure and sharding-agnostic; see `convert_maxtext_params` for the sharded
  whole-tree conversion.

  Args:
    tree: MaxText parameters as nested dicts (the `params` collection of the
      `Transformer`: `token_embedder`, `decoder`, `mtp_block`), with array,
      boxed-variable or abstract leaves.
    cfg: Model config. Weights are cast to `cfg.model.weight_dtype` and the
      router biases to `cfg.model.router_bias_dtype`.
    param_scan_axis: MaxText's `param_scan_axis`: the layer axis of the leaves
      of the scanned decoder stacks.

  Returns:
    The parameters and the state (router biases from the tree; identity expert
    permutations when `cfg.kernels.expert_permutation` is enabled).
  """
  m = cfg.model
  scanned = functools.partial(_to_lineage_leaf, param_scan_axis=param_scan_axis, dtype=m.weight_dtype)
  unscanned = functools.partial(_to_lineage_leaf, param_scan_axis=0, dtype=m.weight_dtype)
  unscanned_bias = functools.partial(_to_lineage_leaf, param_scan_axis=0, dtype=m.router_bias_dtype)
  dims = dict(qk_head_dim=m.qk_head_dim, rope_head_dim=m.rope_head_dim)

  decoder = tree["decoder"]
  sparse, router_bias = _sparse_layer_from_maxtext(decoder["moe_layers"], scanned, unscanned_bias, **dims)
  mtp = []
  mtp_router_bias = []
  for k in range(1, m.num_mtp_layers + 1):
    layer = tree["mtp_block"][f"mtp_layer_{k}"]
    prefix = f"mtp_{k}"
    mtp_sparse, bias = _sparse_layer_from_maxtext(layer[f"{prefix}_transformer_layer"], unscanned, unscanned_bias, **dims)
    mtp.append(
        dsv3_mtp_block.DSv3MTPDepthWeightsPytree(
            layer=dsv3_types.DSv3MTPWeightsPytree(
                ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                    enorm_scale=unscanned(layer[f"{prefix}_embedding_norm"]["scale"]),
                    hnorm_scale=unscanned(layer[f"{prefix}_hidden_state_norm"]["scale"]),
                    eh_proj=unscanned(layer[f"{prefix}_projection"]["kernel"]),
                ),
                sparse=mtp_sparse,
            ),
            final_norm_scale=unscanned(layer[f"{prefix}_final_norm"]["scale"]),
        )
    )
    mtp_router_bias.append(bias)

  params = dsv3_model.DSv3Params[dsv3_types.ArrayType](
      embed=dsv3_embed.DSv3EmbedWeightsPytree(table=unscanned(tree["token_embedder"]["embedding"])),
      decoder=dsv3_types.DSv3WeightsPytree(
          dense=_dense_layer_from_maxtext(decoder["dense_layers"], scanned, **dims),
          sparse=sparse,
      ),
      mtp=tuple(mtp),
      head=dsv3_embed.DSv3HeadWeightsPytree(
          final_norm_scale=unscanned(decoder["decoder_norm"]["scale"]),
          kernel=(None if m.tied_head else unscanned(decoder["logits_dense"]["kernel"])),
      ),
  )
  expert_permutations, mtp_expert_permutations = dsv3_model.identity_expert_permutations(cfg)
  state = dsv3_model.DSv3MutableState[dsv3_types.ArrayType](
      router_bias=router_bias,
      mtp_router_bias=tuple(mtp_router_bias),
      expert_permutations=expert_permutations,
      mtp_expert_permutations=mtp_expert_permutations,
  )
  return params, state


def lineage_to_maxtext_params(
    params: dsv3_model.DSv3Params,
    state: dsv3_model.DSv3MutableState,
    cfg: dsv3_config.DSv3Config,
    *,
    param_scan_axis: int = PARAM_SCAN_AXIS,
) -> Tree:
  """Inverse of `maxtext_to_lineage_params` (expert permutations are dropped).

  Args:
    params: Lineage-layout parameters.
    state: The state; only the router biases are used.
    cfg: Model config.
    param_scan_axis: MaxText's `param_scan_axis`.

  Returns:
    The MaxText-layout parameter tree (nested dicts of arrays).
  """
  m = cfg.model
  scanned = functools.partial(_to_maxtext_leaf, param_scan_axis=param_scan_axis)
  unscanned = functools.partial(_to_maxtext_leaf, param_scan_axis=0)
  rope = dict(rope_head_dim=m.rope_head_dim)
  assert state.router_bias is not None
  if len(params.mtp) != len(state.mtp_router_bias):
    raise ValueError(f"Got {len(state.mtp_router_bias)} MTP router biases for" f" {len(params.mtp)} MTP depths.")

  decoder = {
      "decoder_norm": {"scale": unscanned(params.head.final_norm_scale)},
      "dense_layers": _dense_layer_to_maxtext(params.decoder.dense, scanned, **rope),
      "moe_layers": _sparse_layer_to_maxtext(params.decoder.sparse, state.router_bias, scanned, **rope),
  }
  if params.head.kernel is not None:
    decoder["logits_dense"] = {"kernel": unscanned(params.head.kernel)}
  tree: Tree = {
      "token_embedder": {"embedding": unscanned(params.embed.table)},
      "decoder": decoder,
  }
  if params.mtp:
    mtp_block = {}
    for k, (depth, bias) in enumerate(zip(params.mtp, state.mtp_router_bias), start=1):
      prefix = f"mtp_{k}"
      ehproj = depth.layer.ehproj
      mtp_block[f"mtp_layer_{k}"] = {
          f"{prefix}_embedding_norm": {"scale": unscanned(ehproj.enorm_scale)},
          f"{prefix}_hidden_state_norm": {"scale": unscanned(ehproj.hnorm_scale)},
          f"{prefix}_projection": {"kernel": unscanned(ehproj.eh_proj)},
          f"{prefix}_transformer_layer": _sparse_layer_to_maxtext(depth.layer.sparse, bias, unscanned, **rope),
          f"{prefix}_final_norm": {"scale": unscanned(depth.final_norm_scale)},
      }
    tree["mtp_block"] = mtp_block
  return tree


def restore_sharding(
    shape: tuple[int, ...],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> jax.sharding.NamedSharding:
  """Greedily shards `shape` from dim 0 onward over `mesh.axis_names`.

  Args:
    shape: Global array shape.
    mesh: The device mesh.

  Returns:
    A `NamedSharding` that assigns each mesh axis to the earliest array
    dimension whose remaining size is a multiple of that axis's size.
  """
  remaining = list(mesh.axis_names)
  spec: list[str | tuple[str, ...] | None] = [None] * len(shape)
  for i, dim in enumerate(shape):
    assigned: list[str] = []
    rem_dim = dim
    still_remaining: list[str] = []
    for axis in remaining:
      axis_size = mesh.shape[axis]
      if rem_dim % axis_size == 0:
        assigned.append(axis)
        rem_dim //= axis_size
      else:
        still_remaining.append(axis)
    remaining = still_remaining
    if len(assigned) == 1:
      spec[i] = assigned[0]
    elif len(assigned) > 1:
      spec[i] = tuple(assigned)
  if not any(spec):
    return jax.sharding.NamedSharding(mesh, P())
  return jax.sharding.NamedSharding(mesh, P(*spec))


@contextlib.contextmanager
def _mesh_context(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh):
  if isinstance(mesh, jax.sharding.AbstractMesh):
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())):
      with jax.sharding.use_abstract_mesh(mesh):
        yield
  else:
    with jax.set_mesh(mesh):
      yield


def _rebind_mesh(tree: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> Any:
  """Rebinds `NamedSharding`s from `eval_shape`'s abstract mesh to `mesh`."""

  def _rebind(a):
    if not isinstance(getattr(a, "sharding", None), jax.sharding.NamedSharding):
      return a
    sharding = jax.sharding.NamedSharding(mesh, a.sharding.spec)
    return jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding)

  return jax.tree.map(_rebind, tree)


def _move_mla_trailing_sharding_to_lora(a: Any) -> Any:
  """Moves any trailing-axis sharding of an MLA up-projection to its LoRA axis.

  In Lineage layout, `q_up`, `k_up` and `v_up` shard their trailing head-dim
  axis over `WEIGHT_FSDP_AXIS` (`"z"`). In MaxText layout, `_interleave_rope`
  and the `k_up`/`v_up` concatenation into `wkv_b` act on that trailing axis, so
  the restore target instead shards the LoRA axis (`axis -3`, which becomes
  `dim 0` after moving `param_scan_axis`).

  Args:
    a: Abstract up-projection leaf (`jax.ShapeDtypeStruct`) or None.

  Returns:
    `a` with its trailing-axis sharding moved to axis -3 when divisible.
  """
  if a is None:
    return None
  sharding = a.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return a
  mesh = sharding.mesh
  spec = list(sharding.spec) + [None] * (a.ndim - len(sharding.spec))
  trailing = spec[-1]
  if trailing is None:
    return a
  spec[-1] = None
  axes = (trailing,) if isinstance(trailing, str) else tuple(trailing)
  factor = math.prod(mesh.shape[axis] for axis in axes)
  if spec[-3] is None and a.shape[-3] % factor == 0:
    spec[-3] = trailing
  return jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=jax.sharding.NamedSharding(mesh, P(*spec)))


def _prepare_mla_for_maxtext_restore(node: Any) -> Any:
  if not isinstance(node, dsv3_types.DSv3MLAWeightsPytree):
    return node
  return dataclasses.replace(
      node,
      q_up=_move_mla_trailing_sharding_to_lora(node.q_up),
      k_up=_move_mla_trailing_sharding_to_lora(node.k_up),
      v_up=_move_mla_trailing_sharding_to_lora(node.v_up),
  )


def lineage_abstract_params(
    cfg: dsv3_config.DSv3Config,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> tuple[Any, Any]:
  """`init_dsv3_params`' outputs as `jax.ShapeDtypeStruct`s (with shardings)."""
  with _mesh_context(mesh):
    out = jax.eval_shape(lambda: dsv3_model.init_dsv3_params(jax.random.key(0), cfg, cast(jax.sharding.Mesh, mesh)))
  shardings = dsv3_model.dsv3_param_shardings(cfg, cast(jax.sharding.Mesh, mesh))
  return jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
      out,
      shardings,
  )


def maxtext_abstract_param_tree(
    cfg: dsv3_config.DSv3Config,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    *,
    param_scan_axis: int = PARAM_SCAN_AXIS,
) -> Tree:
  """The MaxText-layout parameter tree as an Orbax restore target.

  Derives each leaf's `NamedSharding` from `lineage_abstract_params(cfg, mesh)`
  (with MLA up-projection trailing-axis sharding moved to the LoRA axis) so
  that on-disk zarr3 chunks along `dim 0` (notably `wi_0`, `wi_1` and `wo`
  sharded over the expert axes) are read without cross-host amplification and
  already sit on their target Lineage devices before `convert_maxtext_params`.

  Args:
    cfg: Model config.
    mesh: The device mesh.
    param_scan_axis: MaxText's `param_scan_axis`.

  Returns:
    Nested dicts of `jax.ShapeDtypeStruct` with MaxText's names, global shapes,
    dtypes (`weight_dtype`; `router_bias_dtype` for the routed gate bias) and
    `NamedSharding`s.
  """
  params, state = lineage_abstract_params(cfg, mesh)
  params = jax.tree.map(
      _prepare_mla_for_maxtext_restore,
      params,
      is_leaf=lambda x: isinstance(x, dsv3_types.DSv3MLAWeightsPytree),
  )
  with _mesh_context(mesh):
    tree = jax.eval_shape(
        functools.partial(lineage_to_maxtext_params, cfg=cfg, param_scan_axis=param_scan_axis),
        params,
        state,
    )
  return _rebind_mesh(tree, mesh)


def convert_maxtext_params(
    tree: Mapping[str, Any],
    cfg: dsv3_config.DSv3Config,
    mesh: jax.sharding.Mesh,
    *,
    param_scan_axis: int = PARAM_SCAN_AXIS,
) -> tuple[
    dsv3_model.DSv3Params[dsv3_types.ArrayType],
    dsv3_model.DSv3MutableState[dsv3_types.ArrayType],
]:
  """`maxtext_to_lineage_params` as one jitted program into Lineage's shardings.

  The conversion runs with the mesh axes in `Auto` mode (`jax.sharding.
  auto_axes`): under explicit sharding, slicing the RoPE channels or the fused
  `wkv_b` along a sharded axis is not allowed. Its outputs are the
  `dsv3_model.dsv3_param_shardings` of `cfg`.

  Args:
    tree: MaxText parameters (see `maxtext_to_lineage_params`), e.g. restored
      into `maxtext_abstract_param_tree`.
    cfg: Model config.
    mesh: The device mesh.
    param_scan_axis: MaxText's `param_scan_axis`.

  Returns:
    The parameters and the state, sharded as `dsv3_param_shardings(cfg, mesh)`.
  """
  out_shardings = dsv3_model.dsv3_param_shardings(cfg, mesh)
  convert = jax.sharding.auto_axes(
      functools.partial(maxtext_to_lineage_params, cfg=cfg, param_scan_axis=param_scan_axis),
      out_sharding=out_shardings,
  )
  with jax.set_mesh(mesh):
    return jax.jit(convert)(jax.tree.map(_unbox, tree))
