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

"""Model-level API of the Lineage DSv3 model.

Composes the Lineage building blocks (token embedding, decoder stacks, MTP
block, final norm, LM head and losses) into the training objective of
MaxText's DeepSeek-V3 recipe (`train.loss_fn` plus the `use_lineage` decoder
and MTP branches), as pure functions of explicit parameter / state pytrees and
a `DSv3Config`. Framework adapters only map their configs and parameter
containers onto these functions; nothing below them reads a framework config.

Parameters are stored in Lineage layout, so a step performs no weight layout
translation. The only per-step weight work is the cast of the decoder / MTP
layer weights from `weight_dtype` to the compute `dtype`, as in MaxText.

The mesh must use Explicit sharding axes: inputs are placed with `jax.reshard`
and `out_sharding`, as the Lineage layers expect.
"""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3
from maxtext.experimental.lineage import dsv3_config
from maxtext.experimental.lineage import dsv3_embed
from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_loss
from maxtext.experimental.lineage import dsv3_mla
from maxtext.experimental.lineage import dsv3_mtp
from maxtext.experimental.lineage import dsv3_mtp_block
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage.gmmv2 import expert_mlp

P = jax.sharding.PartitionSpec
T = TypeVar(
    "T",
    dsv3_types.ArrayType,
    dsv3_types.ShardingType,
    dsv3_types.AxisNameType,
)

# Output spec of the Splash attention kernel: heads over `attention`.
SPLASH_KERNEL_OUT_SPEC = P("attention", None)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3Params(Generic[T]):
  """Trainable parameters of the DSv3 model, in Lineage layout.

  Attributes:
    embed: Token embedding table `[vocab, emb]`.
    decoder: Stacked dense and sparse decoder layer weights. The routed gate
      biases (`sparse.moe.router.bias`) are None here; they are not trained by
      gradient and live in `DSv3MutableState.router_bias`.
    mtp: Weights of the MTP depths `1..K`, in order (`DSv3MTPDepthWeightsPytree`
      each, no layer axis). Their router biases are None as well.
    head: Final norm scale and LM head kernel (None when `tied_head`).
  """

  embed: dsv3_embed.DSv3EmbedWeightsPytree[T] = dataclasses.field(default_factory=dsv3_embed.DSv3EmbedWeightsPytree[T])
  decoder: dsv3_types.DSv3WeightsPytree[T] = dataclasses.field(default_factory=dsv3_types.DSv3WeightsPytree[T])
  mtp: tuple[dsv3_mtp_block.DSv3MTPDepthWeightsPytree[T], ...] = ()
  head: dsv3_embed.DSv3HeadWeightsPytree[T] = dataclasses.field(default_factory=dsv3_embed.DSv3HeadWeightsPytree[T])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MutableState(Generic[T]):
  """Non-differentiated state that a training step updates.

  Attributes:
    router_bias: Routed gate biases of the sparse decoder layers,
      `[num_sparse_layers, num_experts]` in `router_bias_dtype`.
    mtp_router_bias: Routed gate bias `[num_experts]` of every MTP depth.
    expert_permutations: Expert permutations `[num_sparse_layers, num_experts]`
      (int32) of the sparse decoder layers for the next step, or None when
      expert shuffling is off (`expert_permutation == "none"`).
    mtp_expert_permutations: Expert permutation `[num_experts]` of every MTP
      depth; empty when expert shuffling is off.
  """

  router_bias: T | None = None
  mtp_router_bias: tuple[T, ...] = ()
  expert_permutations: T | None = None
  mtp_expert_permutations: tuple[T, ...] = ()


class DSv3Batch(NamedTuple):
  """One training / eval batch, as MaxText's data fields.

  Attributes:
    inputs: Input token ids `[batch, seq]` (`inputs`).
    targets: Target token ids `[batch, seq]` (`targets`).
    positions: Token positions `[batch, seq]` (`inputs_position`).
    segment_ids: Packed-document segment ids `[batch, seq]`
      (`inputs_segmentation`), or None when unpacked.
    target_mask: Target segmentation `[batch, seq]` (`targets_segmentation`);
      zero marks padding.
  """

  inputs: jax.Array
  targets: jax.Array
  positions: jax.Array
  segment_ids: jax.Array | None
  target_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3ForwardOutputs:
  """Outputs of `dsv3_forward`.

  Attributes:
    hidden: Pre-final-norm hidden states `[batch, seq, emb]` of the decoder.
    token_embeddings: Embeddings `[batch, seq, emb]` of the input tokens, in the
      compute dtype.
    router_aux: Router aux of the sparse decoder layers (local, unreduced).
  """

  hidden: jax.Array
  token_embeddings: jax.Array
  router_aux: dsv3_types.DSv3RouterAux[jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3LossAux:
  """Auxiliary outputs of `dsv3_loss_and_aux`.

  Attributes:
    xent_sum: Sum of the main cross entropy (incl. z-loss) over valid tokens.
    z_loss_sum: Sum of the z-loss over valid tokens.
    total_weights: Number of valid target tokens (int32).
    moe_lb_loss: MoE load balance loss added to the objective (0.0 when
      disabled); sum of the per-layer Megatron seq_aux losses over the decoder
      and MTP layers, or the mean of the Switch-style losses.
    mtp_loss: MTP loss added to the objective; 0.0 in eval or without MTP.
    router_aux: Router aux of the sparse decoder layers.
    mtp_router_aux: Router aux of every MTP depth.
    router_bias_updates: Loss-free load balancing updates `[num_sparse_layers,
      num_experts]` applied in `new_state`; None in eval or when disabled.
    mtp_router_bias_updates: The updates `[num_experts]` of every MTP depth;
      empty in eval or when disabled.
    new_state: The state for the next step: biases plus updates and the next
      expert permutations in training; `state` unchanged in eval.
    logits: Main logits `[batch, seq, vocab]` in eval, else None.
    mtp_preds: fp32 argmax predictions `[1, batch, seq]` of the
      `mtp_eval_target_module` depth in eval, else None.
    mtp_mask: Validity mask `[1, batch, seq]` of `mtp_preds`, else None.
  """

  xent_sum: jax.Array
  z_loss_sum: jax.Array
  total_weights: jax.Array
  moe_lb_loss: jax.Array | float
  mtp_loss: jax.Array | float
  router_aux: dsv3_types.DSv3RouterAux[jax.Array]
  mtp_router_aux: tuple[dsv3_types.DSv3RouterAux[jax.Array], ...]
  router_bias_updates: jax.Array | None
  mtp_router_bias_updates: tuple[jax.Array, ...]
  new_state: DSv3MutableState[dsv3_types.ArrayType]
  logits: jax.Array | None
  mtp_preds: jax.Array | None
  mtp_mask: jax.Array | None


def _named_sharding(spec: P, cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
  return jax.sharding.NamedSharding(mesh, ops.physical_pspec(spec, cfg.sharding.axis_mapping))


def _cast(tree: Any, dtype: jax.typing.DTypeLike) -> Any:
  """Casts every array leaf of `tree` to `dtype`."""
  return jax.tree.map(lambda x: jnp.asarray(x, dtype), tree)


def with_router_bias(
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    bias: jax.Array | None,
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  """Returns `w` with `moe.router.bias` set to `bias`."""
  return dataclasses.replace(
      w,
      moe=dataclasses.replace(w.moe, router=dataclasses.replace(w.moe.router, bias=bias)),
  )


def init_kernels(
    positions: jax.Array,
    cfg: dsv3_config.DSv3Config,
    mesh: jax.sharding.Mesh,
) -> tuple[Any, Any]:
  """Builds the YaRN frequencies and the Splash attention kernel of a step.

  Args:
    positions: Token positions `[batch, seq]`.
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    `(yarn_freqs, splash_kernel)` for the decoder and MTP layers.
  """
  m, k, s = cfg.model, cfg.kernels, cfg.sharding
  yarn_freqs = dsv3_mla.get_yarn_freqs(
      positions,
      rope_head_dim=m.rope_head_dim,
      rope_theta=m.rope_theta,
      max_position_embeddings=m.max_position_embeddings,
      original_max_position_embeddings=m.original_max_position_embeddings,
      beta_fast=m.beta_fast,
      beta_slow=m.beta_slow,
      rope_factor=m.rope_factor,
      out_pspec=s.yarn_freqs,
      mesh=mesh,
      axis_mapping=s.axis_mapping,
      dtype=m.dtype,
  )
  splash_kernel = dsv3_mla.init_splash_kernel(
      sa_block_q=k.sa_block_q,
      sa_block_kv=k.sa_block_kv,
      sa_block_kv_compute=k.sa_block_kv_compute,
      sa_block_q_dkv=k.sa_block_q_dkv,
      sa_block_kv_dkv=k.sa_block_kv_dkv,
      sa_block_kv_dkv_compute=k.sa_block_kv_dkv_compute,
      sa_q_layout=k.sa_q_layout,
      sa_k_layout=k.sa_k_layout,
      sa_v_layout=k.sa_v_layout,
      max_target_length=k.max_target_length,
      num_query_heads=m.num_query_heads,
      kernel_out_spec=SPLASH_KERNEL_OUT_SPEC,
      mesh=mesh,
      axis_mapping=s.axis_mapping,
      qk_diag_skip=k.qk_diag_skip,
      qk_diag_grid=k.qk_diag_grid,
      sv_diag_skip=k.sv_diag_skip,
  )
  return yarn_freqs, splash_kernel


def dsv3_layer_kwargs(cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """Keyword arguments shared by `dsv3.dsv3` and `dsv3_mtp.dsv3_mtp_layer`.

  Args:
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    The keyword arguments (everything but the activations, weights, kernels,
    segment ids and expert permutations).
  """
  m, k = cfg.model, cfg.kernels
  return dict(
      num_experts=m.num_experts,
      num_experts_per_tok=m.num_experts_per_tok,
      routed_scaling_factor=float(m.routed_scaling_factor),
      n_routing_groups=m.n_routing_groups,
      topk_routing_group=m.topk_routing_group,
      topk_in_group=m.topk_in_group,
      qk_head_dim=m.qk_head_dim,
      rope_head_dim=m.rope_head_dim,
      num_query_heads=m.num_query_heads,
      mscale=float(m.mscale),
      kv_lora_rank=m.kv_lora_rank,
      max_position_embeddings=m.max_position_embeddings,
      original_max_position_embeddings=m.original_max_position_embeddings,
      rope_factor=int(m.rope_factor),
      mesh=mesh,
      norm_fn=functools.partial(ops.rms_norm, epsilon=m.norm_epsilon),
      rope_fn=dsv3_mla.yarn,
      gmm_fn=expert_mlp.make_gmm_fn(emb_dim=m.emb_dim),
      axis_mapping=cfg.sharding.axis_mapping,
      expert_axis_name="expert",
      capacity_factor=float(k.capacity_factor),
      router_dtype=m.router_dtype,
      ragged_buffer_factor=float(k.ragged_buffer_factor),
      quant=k.quant,
  )


def identity_expert_permutations(
    cfg: dsv3_config.DSv3Config,
) -> tuple[jax.Array | None, tuple[jax.Array, ...]]:
  """The (unsharded) identity expert permutations of the first step.

  Args:
    cfg: Model config.

  Returns:
    `(expert_permutations, mtp_expert_permutations)` of `DSv3MutableState`:
    identity permutations (int32) when `cfg.kernels.expert_permutation` is
    not `"none"`, else `(None, ())`.
  """
  m = cfg.model
  if cfg.kernels.expert_permutation == "none":
    return None, ()
  identity = jnp.arange(m.num_experts, dtype=jnp.int32)
  return (
      jnp.tile(identity, (m.num_sparse_layers, 1)),
      (identity,) * m.num_mtp_layers,
  )


def init_expert_permutations(
    cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh
) -> tuple[jax.Array | None, tuple[jax.Array, ...]]:
  """`identity_expert_permutations`, replicated over `mesh`.

  Args:
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    `(expert_permutations, mtp_expert_permutations)` of `DSv3MutableState`.
  """
  expert_permutations, mtp_expert_permutations = identity_expert_permutations(cfg)
  if expert_permutations is None:
    return None, ()
  replicated = jax.sharding.NamedSharding(mesh, P())
  return (
      jax.reshard(expert_permutations, replicated),
      tuple(jax.reshard(p, replicated) for p in mtp_expert_permutations),
  )


def dsv3_param_shardings(cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh) -> tuple[
    DSv3Params[dsv3_types.ShardingType],
    DSv3MutableState[dsv3_types.ShardingType],
]:
  """The physical `NamedSharding`s of `init_dsv3_params`' outputs.

  Args:
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    Pytrees of `NamedSharding` with the structure of the parameters and the
    state (router biases and expert permutations replicated).
  """
  m, s = cfg.model, cfg.sharding
  specs = s.weights

  def named(spec: P) -> jax.sharding.NamedSharding:
    return _named_sharding(spec, cfg, mesh)

  def named_tree(spec_tree: Any) -> Any:
    return jax.tree.map(named, spec_tree, is_leaf=lambda x: isinstance(x, P))

  decoder = named_tree(specs.decoder)
  mtp_layer = named_tree(specs.mtp)
  params = DSv3Params[dsv3_types.ShardingType](
      embed=dsv3_embed.DSv3EmbedWeightsPytree(table=named(specs.embed_table)),
      decoder=dataclasses.replace(decoder, sparse=with_router_bias(decoder.sparse, None)),
      mtp=tuple(
          dsv3_mtp_block.DSv3MTPDepthWeightsPytree(
              layer=dataclasses.replace(mtp_layer, sparse=with_router_bias(mtp_layer.sparse, None)),
              final_norm_scale=named(specs.final_norm_scale),
          )
          for _ in range(m.num_mtp_layers)
      ),
      head=dsv3_embed.DSv3HeadWeightsPytree(
          final_norm_scale=named(specs.final_norm_scale),
          kernel=None if m.tied_head else named(specs.head_kernel),
      ),
  )
  replicated = jax.sharding.NamedSharding(mesh, P())
  shuffling = cfg.kernels.expert_permutation != "none"
  state = DSv3MutableState[dsv3_types.ShardingType](
      router_bias=replicated,
      mtp_router_bias=(replicated,) * m.num_mtp_layers,
      expert_permutations=replicated if shuffling else None,
      mtp_expert_permutations=((replicated,) * m.num_mtp_layers if shuffling else ()),
  )
  return params, state


def init_dsv3_params(
    rng: jax.Array,
    cfg: dsv3_config.DSv3Config,
    mesh: jax.sharding.Mesh,
) -> tuple[DSv3Params[dsv3_types.ArrayType], DSv3MutableState[dsv3_types.ArrayType]]:
  """Initializes random parameters and the initial state.

  Weights are drawn with the Lineage default initializers (the embedding table
  with MaxText's `normal(stddev=1.0)`), in `weight_dtype`, sharded as
  `cfg.sharding.weights`. Router biases start at zero in `router_bias_dtype`
  and expert permutations at the identity.

  Args:
    rng: A PRNG key.
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    The parameters and the state.
  """
  m, s = cfg.model, cfg.sharding
  specs = s.weights
  embed_key, decoder_key, head_key, mtp_key = jax.random.split(rng, 4)

  embed = dsv3_embed.init_dsv3_embed_weights(
      embed_key,
      m.vocab_size,
      m.emb_dim,
      mesh,
      s.axis_mapping,
      weight_dtype=m.weight_dtype,
      out_shardings=dsv3_embed.DSv3EmbedWeightsPytree(table=specs.embed_table),
  )
  decoder = dsv3_types.init_dsv3_weights(
      decoder_key,
      m.emb_dim,
      m.q_lora_rank,
      m.kv_lora_rank,
      m.num_query_heads,
      m.num_kv_heads,
      m.rope_head_dim,
      m.qk_head_dim,
      m.v_head_dim,
      m.mlp_dim,
      m.num_experts,
      m.moe_mlp_dim,
      mesh,
      s.axis_mapping,
      weight_dtype=m.weight_dtype,
      num_dense_layers=m.num_dense_layers,
      num_sparse_layers=m.num_sparse_layers,
      out_shardings=specs.decoder,
  )
  head = dsv3_embed.init_dsv3_head_weights(
      head_key,
      m.vocab_size,
      m.emb_dim,
      mesh,
      s.axis_mapping,
      weight_dtype=m.weight_dtype,
      tied_head=m.tied_head,
      out_shardings=dsv3_embed.DSv3HeadWeightsPytree(final_norm_scale=specs.final_norm_scale, kernel=specs.head_kernel),
  )

  replicated = jax.sharding.NamedSharding(mesh, P())
  mtp = []
  mtp_router_bias = []
  for depth_key in jax.random.split(mtp_key, m.num_mtp_layers):
    layer_key, scale_key = jax.random.split(depth_key)
    layer = dsv3_types.init_dsv3_mtp_weights(
        layer_key,
        m.emb_dim,
        m.q_lora_rank,
        m.kv_lora_rank,
        m.num_query_heads,
        m.num_kv_heads,
        m.rope_head_dim,
        m.qk_head_dim,
        m.v_head_dim,
        m.num_experts,
        m.moe_mlp_dim,
        mesh,
        s.axis_mapping,
        weight_dtype=m.weight_dtype,
        num_layers=1,
        out_shardings=specs.mtp,
    )
    assert layer.sparse.moe.router.bias is not None
    mtp_router_bias.append(
        jax.reshard(
            layer.sparse.moe.router.bias.astype(m.router_bias_dtype),
            replicated,
        )
    )
    mtp.append(
        dsv3_mtp_block.DSv3MTPDepthWeightsPytree(
            layer=dataclasses.replace(layer, sparse=with_router_bias(layer.sparse, None)),
            final_norm_scale=dsv3_types.default_scale_init(
                scale_key,
                (m.emb_dim,),
                m.weight_dtype,
                out_sharding=_named_sharding(specs.final_norm_scale, cfg, mesh),
            ),
        )
    )

  assert decoder.sparse.moe.router.bias is not None
  router_bias = jax.reshard(decoder.sparse.moe.router.bias.astype(m.router_bias_dtype), replicated)
  params = DSv3Params[dsv3_types.ArrayType](
      embed=embed,
      decoder=dataclasses.replace(decoder, sparse=with_router_bias(decoder.sparse, None)),
      mtp=tuple(mtp),
      head=head,
  )

  expert_permutations, mtp_expert_permutations = init_expert_permutations(cfg, mesh)
  state = DSv3MutableState[dsv3_types.ArrayType](
      router_bias=router_bias,
      mtp_router_bias=tuple(mtp_router_bias),
      expert_permutations=expert_permutations,
      mtp_expert_permutations=mtp_expert_permutations,
  )
  return params, state


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_forward(
    params: DSv3Params,
    state: DSv3MutableState,
    batch: DSv3Batch,
    cfg: dsv3_config.DSv3Config,
    *,
    mesh: jax.sharding.Mesh,
    kernels: tuple[Any, Any] | None = None,
) -> DSv3ForwardOutputs:
  """Runs the token embedding and the decoder stacks.

  Args:
    params: Model parameters.
    state: Mutable state (router biases, expert permutations).
    batch: The batch.
    cfg: Model config.
    mesh: The device mesh.
    kernels: `init_kernels(...)` of this batch, or None to build them.

  Returns:
    The pre-final-norm hidden states, the token embeddings and the decoder
    router aux.
  """
  m, k, s = cfg.model, cfg.kernels, cfg.sharding
  shuffling = k.expert_permutation != "none"
  if shuffling != (state.expert_permutations is not None):
    raise ValueError(
        f"expert_permutation={k.expert_permutation!r} but" f" state.expert_permutations is {state.expert_permutations}."
    )
  assert params.embed.table is not None
  with jax.named_scope("embed"):
    token_embeddings = jax.reshard(
        dsv3_embed.embed_tokens_collected(
            jax.reshard(batch.inputs, _named_sharding(s.token, cfg, mesh)),
            params.embed.table,
            dtype=m.dtype,
            iota_embed=m.iota_embed,
        ),
        _named_sharding(s.activation, cfg, mesh),
    )
  segment_ids = None
  if batch.segment_ids is not None:
    segment_ids = jax.reshard(batch.segment_ids, _named_sharding(s.segment_ids, cfg, mesh))
  if kernels is None:
    kernels = init_kernels(batch.positions, cfg, mesh)
  yarn_freqs, splash_kernel = kernels

  assert state.router_bias is not None
  decoder = _cast(params.decoder, m.dtype)
  decoder = dataclasses.replace(decoder, sparse=with_router_bias(decoder.sparse, state.router_bias))
  with jax.named_scope("decoder"):
    hidden, router_aux = dsv3.dsv3(
        token_embeddings,
        decoder,
        yarn_freqs,
        splash_kernel,
        segment_ids=segment_ids,
        max_async_overlap_transform=k.max_async_overlap_transform,
        expert_permutations=state.expert_permutations,
        **dsv3_layer_kwargs(cfg, mesh),
    )
  return DSv3ForwardOutputs(hidden=hidden, token_embeddings=token_embeddings, router_aux=router_aux)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_lm_head(
    params: DSv3Params,
    normed: jt.Num[jax.Array, "B T D"],
    cfg: dsv3_config.DSv3Config,
    *,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "B T V"]:
  """Applies the (shared) LM head to normalized hidden states.

  Args:
    params: Model parameters.
    normed: Normalized hidden states `[batch, seq, emb]`.
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    The logits `[batch, seq, vocab]` in `cfg.model.logits_dtype`.
  """
  m = cfg.model
  out_sharding = _named_sharding(cfg.sharding.activation, cfg, mesh)
  if m.tied_head:
    assert params.embed.table is not None
    return dsv3_embed.tied_lm_head(
        normed,
        params.embed.table,
        dtype=m.dtype,
        dot_in_fp32=m.head_dot_in_fp32,
        cast_logits_to_fp32=m.cast_logits_to_fp32,
        out_sharding=out_sharding,
    )
  assert params.head.kernel is not None
  return dsv3_embed.lm_head(
      normed,
      params.head.kernel,
      dtype=m.dtype,
      dot_in_fp32=m.head_dot_in_fp32,
      cast_logits_to_fp32=m.cast_logits_to_fp32,
      out_sharding=out_sharding,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_logits(
    params: DSv3Params,
    hidden: jt.Num[jax.Array, "B T D"],
    cfg: dsv3_config.DSv3Config,
    *,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "B T V"]:
  """Applies the final norm and the LM head to the decoder output.

  Args:
    params: Model parameters.
    hidden: Pre-final-norm hidden states `[batch, seq, emb]`.
    cfg: Model config.
    mesh: The device mesh.

  Returns:
    The logits `[batch, seq, vocab]`.
  """
  assert params.head.final_norm_scale is not None
  normed = dsv3_embed.final_norm(hidden, params.head.final_norm_scale, epsilon=cfg.model.norm_epsilon)
  return dsv3_lm_head(params, normed, cfg, mesh=mesh)


def _moe_lb_loss(parts: Sequence[jax.Array], *, megatron: bool) -> jax.Array:
  """Combines per-module load balance losses as MaxText's `train.loss_fn`."""
  lb_losses = jnp.concatenate([jnp.ravel(p) for p in parts])
  return jnp.sum(lb_losses) if megatron else jnp.mean(lb_losses)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_loss_and_aux(
    params: DSv3Params,
    state: DSv3MutableState,
    batch: DSv3Batch,
    cfg: dsv3_config.DSv3Config,
    *,
    mesh: jax.sharding.Mesh,
    train: bool,
) -> tuple[jax.Array, DSv3LossAux]:
  """Computes the training objective of MaxText's DeepSeek-V3 recipe.

  `loss = xent_sum / (total_weights + EPS) [+ mtp_loss if train] + moe_lb_loss`
  with the terms added in that order, as `train.loss_fn`. The MTP block runs in
  eval as well (its load balance loss is part of `moe_lb_loss` there too); only
  its loss term is train-only.

  Args:
    params: Model parameters.
    state: Mutable state (router biases, expert permutations).
    batch: The batch.
    cfg: Model config.
    mesh: The device mesh.
    train: Training step (adds the MTP loss; computes the routed-bias updates
      and the next expert permutations into `aux.new_state`).

  Returns:
    The scalar loss and the auxiliary outputs.
  """
  m, k, t, s = cfg.model, cfg.kernels, cfg.training, cfg.sharding
  token_sharding = _named_sharding(s.token, cfg, mesh)
  replicated = jax.sharding.NamedSharding(mesh, P())
  # Per-token arrays share the logits' batch / sequence layout.
  inputs = jax.reshard(batch.inputs, token_sharding)
  targets = jax.reshard(batch.targets, token_sharding)
  target_mask = jax.reshard(batch.target_mask, token_sharding)
  token_segment_ids = None
  if batch.segment_ids is not None:
    token_segment_ids = jax.reshard(batch.segment_ids, token_sharding)

  kernels = init_kernels(batch.positions, cfg, mesh)
  fwd = dsv3_forward(params, state, batch, cfg, mesh=mesh, kernels=kernels)
  logits = dsv3_logits(params, fwd.hidden, cfg, mesh=mesh)
  with jax.named_scope("main_loss"):
    sums = dsv3_loss.masked_token_loss(
        logits,
        targets,
        target_mask,
        z_loss_multiplier=t.z_loss_multiplier,
        token_sharding=token_sharding,
    )
  assert sums.xent is not None
  assert sums.z_loss is not None
  assert sums.weights is not None
  loss = sums.xent / (sums.weights + dsv3_loss.EPS)

  router_aux_kwargs = dict(
      num_experts=m.num_experts,
      num_experts_per_tok=m.num_experts_per_tok,
      load_balance_loss_weight=t.load_balance_loss_weight,
      megatron_seq_aux_loss=t.megatron_seq_aux_loss,
      routed_bias_update_rate=t.routed_bias_update_rate if train else 0.0,
  )
  lb_loss, bias_updates = dsv3_loss.moe_router_aux_outputs(fwd.router_aux, **router_aux_kwargs)
  lb_parts = [] if lb_loss is None else [lb_loss]

  mtp_loss: jax.Array | float = 0.0
  mtp_outputs = None
  mtp_bias_updates = []
  if params.mtp:
    if len(state.mtp_router_bias) != len(params.mtp):
      raise ValueError(f"Got {len(state.mtp_router_bias)} MTP router biases for" f" {len(params.mtp)} MTP depths.")
    segment_ids = None
    if batch.segment_ids is not None:
      segment_ids = jax.reshard(batch.segment_ids, _named_sharding(s.segment_ids, cfg, mesh))
    yarn_freqs, splash_kernel = kernels
    layer_kwargs = dsv3_layer_kwargs(cfg, mesh)
    depth_weights = []
    for w, bias in zip(params.mtp, state.mtp_router_bias):
      layer = _cast(w.layer, m.dtype)
      depth_weights.append(
          dsv3_mtp_block.DSv3MTPDepthWeightsPytree(
              layer=dataclasses.replace(layer, sparse=with_router_bias(layer.sparse, bias)),
              final_norm_scale=w.final_norm_scale,
          )
      )

    def embed_fn(tokens):
      assert params.embed.table is not None
      return jax.reshard(
          dsv3_embed.embed_tokens_collected(
              tokens,
              params.embed.table,
              dtype=m.dtype,
              iota_embed=m.iota_embed,
          ),
          _named_sharding(s.activation, cfg, mesh),
      )

    def layer_fn(h_prev, emb, w, expert_permutation):
      return dsv3_mtp.dsv3_mtp_layer(
          h_prev,
          emb,
          w,
          yarn_freqs,
          splash_kernel,
          segment_ids=segment_ids,
          expert_permutations=expert_permutation,
          **layer_kwargs,
      )

    main_token_embeddings = token_zero_embedding = None
    if t.mtp_reuse_input_embedding:
      assert params.embed.table is not None
      main_token_embeddings = fwd.token_embeddings
      token_zero_embedding = dsv3_embed.embed_single_token(params.embed.table, 0, dtype=m.dtype, iota_embed=m.iota_embed)
    with jax.named_scope("mtp"):
      mtp_outputs = dsv3_mtp_block.dsv3_mtp_block(
          fwd.hidden,
          inputs,
          targets,
          target_mask,
          token_segment_ids,
          depth_weights,
          embed_fn=embed_fn,
          layer_fn=layer_fn,
          head_fn=functools.partial(dsv3_lm_head, params, cfg=cfg, mesh=mesh),
          dtype=m.dtype,
          norm_epsilon=m.norm_epsilon,
          compute_losses=train,
          eval_target_module=0 if train else t.mtp_eval_target_module,
          expert_permutations=state.mtp_expert_permutations or None,
          main_token_embeddings=main_token_embeddings,
          token_zero_embedding=token_zero_embedding,
      )
    if train:
      mtp_loss = dsv3_mtp_block.mtp_loss(mtp_outputs, scaling_factor=t.mtp_loss_scaling_factor)
      loss = loss + mtp_loss
    for aux in mtp_outputs.aux:
      depth_lb_loss, depth_bias_updates = dsv3_loss.moe_router_aux_outputs(aux, **router_aux_kwargs)
      if depth_lb_loss is not None:
        lb_parts.append(depth_lb_loss)
      if depth_bias_updates is not None:
        mtp_bias_updates.append(depth_bias_updates)

  moe_lb_loss: jax.Array | float = 0.0
  if lb_parts:
    moe_lb_loss = _moe_lb_loss(lb_parts, megatron=t.megatron_seq_aux_loss)
    loss = loss + moe_lb_loss

  new_state = state
  if train:
    assert state.router_bias is not None
    # The layers read the biases and permutations replicated.
    router_bias = state.router_bias
    if bias_updates is not None:
      router_bias = jax.reshard((router_bias + bias_updates).astype(router_bias.dtype), replicated)
    mtp_router_bias = state.mtp_router_bias
    if mtp_bias_updates:
      mtp_router_bias = tuple(
          jax.reshard((b + u).astype(b.dtype), replicated) for b, u in zip(state.mtp_router_bias, mtp_bias_updates)
      )
    expert_permutations = state.expert_permutations
    mtp_expert_permutations = state.mtp_expert_permutations
    if k.expert_permutation != "none":
      num_shards = dsv3_expert_shuffle.num_expert_shards(mesh, "expert", s.axis_mapping)
      next_permutation = functools.partial(
          dsv3_expert_shuffle.next_expert_permutation,
          k.expert_permutation,
          num_shards=num_shards,
      )
      expert_permutations = jax.reshard(next_permutation(fwd.router_aux), replicated)
      if mtp_outputs is not None:
        mtp_expert_permutations = tuple(jax.reshard(next_permutation(aux), replicated) for aux in mtp_outputs.aux)
    new_state = DSv3MutableState[dsv3_types.ArrayType](
        router_bias=router_bias,
        mtp_router_bias=mtp_router_bias,
        expert_permutations=expert_permutations,
        mtp_expert_permutations=mtp_expert_permutations,
    )

  aux = DSv3LossAux(
      xent_sum=sums.xent,
      z_loss_sum=sums.z_loss,
      total_weights=sums.weights,
      moe_lb_loss=moe_lb_loss,
      mtp_loss=mtp_loss,
      router_aux=fwd.router_aux,
      mtp_router_aux=() if mtp_outputs is None else tuple(mtp_outputs.aux),
      router_bias_updates=bias_updates,
      mtp_router_bias_updates=tuple(mtp_bias_updates),
      new_state=new_state,
      logits=None if train else logits,
      mtp_preds=None if mtp_outputs is None else mtp_outputs.preds,
      mtp_mask=None if mtp_outputs is None else mtp_outputs.mask,
  )
  return loss, aux
