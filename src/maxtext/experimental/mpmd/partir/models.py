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
"""A pipelineable transformer assembled from native MaxText NNX layers.

The model is deliberately split into one :func:`mpmd.named_computation` per
*stage-assignable unit* (``embed``, ``layer0`` ... ``layerN``, ``unembed``).
PartIR MPMD then maps those names onto meshes using the assignment produced by
:mod:`mesh_utils`, and derives the pipeline schedule from the data dependencies
between them.

The single most important property of this module is that the forward pass is
written **once** and parameterised by a `wrap` strategy:

  * :data:`NAMED_COMPUTATION` wraps each unit in ``mpmd.named_computation``,
    producing a pipelined MPMD program;
  * :data:`UNWRAPPED` is the identity, producing an ordinary single-program
    (SPMD) model.

Both therefore execute *exactly* the same arithmetic, in the same order, on the
same parameters, which is what makes the numerical-parity check in
``parity.py`` meaningful rather than a comparison of two similar-looking but
independently written models.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import functools
from typing import Any, TypeVar

from flax import nnx
from flax.core import spmd as flax_spmd
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax._src import shard_map as jax_shard_map  # pylint: disable=protected-access
from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import mhc
from maxtext.layers import nnx_decoders
from maxtext.layers.embeddings import Embed
from maxtext.layers.linears import DenseGeneral
from maxtext.utils import max_logging
import numpy as np

from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import mesh_utils

_F = TypeVar("_F", bound=Callable[..., Any])

# The decoder blocks this model builds; `config.build_maxtext_config` refuses
# the others (see `hparams.SUPPORTED_DECODER_BLOCKS`).
SUPPORTED_DECODER_BLOCKS = frozenset(DecoderBlockType(block) for block in hparams_lib.SUPPORTED_DECODER_BLOCKS)

# Default unsharded target-dtype parameter footprint budget (2 GiB on 16 GiB
# TPU v5e / CPU) for which FSDP weights are pre-cast and all-gathered once
# before the microbatch loop. Stages on higher-HBM accelerators (8 GiB on 32 GiB
# TPU v6e, 24 GiB on 95 GiB TPU v7x and v5p) scale this budget automatically via
# `_stage_pregather_budget_bytes`. A v5p gets its 24 GiB through Pathways, whose
# devices report "TPU v5p"; libtpu alone reports "TPU v5", which keeps 2 GiB.
_DEFAULT_UNIT_PREGATHER_BYTES = 2 * 1024**3
_MAX_UNIT_PREGATHER_BYTES = _DEFAULT_UNIT_PREGATHER_BYTES
_V6E_STAGE_PREGATHER_BYTES = 8 * 1024**3
_V7X_STAGE_PREGATHER_BYTES = 24 * 1024**3

# A `Wrapper` turns a stage's python function into the function that will
# actually be traced, given the stage's name.
Wrapper = Callable[[str], Callable[[_F], _F]]


def NAMED_COMPUTATION(name: str) -> Callable[[_F], _F]:  # pylint: disable=invalid-name
  """Wraps a stage in an MPMD named computation (pipelined execution)."""
  # Imported here, so that the SPMD half of this module (and its tests) runs
  # without Shardy MPMD installed.
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import mpmd  # pylint: disable=import-outside-toplevel

  return mpmd.named_computation(name=name)


def UNWRAPPED(name: str) -> Callable[[_F], _F]:  # pylint: disable=invalid-name,unused-argument
  """Leaves a stage untouched (single-program / SPMD execution)."""
  return lambda fn: fn


class _NNXDecoderLayerFactory:
  """Adapter that delegates single-layer construction to ``NNXDecoder``.

  Calling ``NNXDecoder.get_decoder_layers`` and
  ``NNXDecoder._init_sequential_generic`` directly instantiates every supported
  ``DecoderBlockType`` with the exact per-layer arguments (attention types,
  NoPE/MLA flags, MoE interleaving) that MaxText's own unscanned decoder uses,
  while only allocating and drawing RNGs for the requested ``layer_index``.
  ``NNXDecoder.get_remat_policy`` resolves the layers' remat policy, from
  ``remat_policy_override``. Some of these are private ``NNXDecoder`` methods:
  ``models_test`` builds every supported block, and resolves every
  ``hparams.REMAT_POLICIES`` entry, through them, so a change of their protocol
  fails there.
  """

  def __init__(self, config: Any, mesh: Mesh | None, layer_index: int, remat_policy_override: str | None = None):
    self.config = config
    self.mesh = mesh
    self.quant = None
    self.model_mode = MODEL_MODE_TRAIN
    self.remat_policy_override = remat_policy_override
    self._layer_index = layer_index
    self.built_layer: nnx.Module | None = None

  def _effective_remat_policy(self) -> str | None:
    return nnx_decoders.NNXDecoder._effective_remat_policy(self)  # pylint: disable=protected-access

  def minimal_policy(self, with_context: bool = False, with_quantization: bool = False) -> Any:
    return nnx_decoders.NNXDecoder.minimal_policy(self, with_context=with_context, with_quantization=with_quantization)

  def remat_policy(self) -> Any:
    """Returns MaxText's ``jax.checkpoint`` policy for ``remat_policy_override`` (None for ``full``)."""
    return nnx_decoders.NNXDecoder.get_remat_policy(self)

  def _create_single_layer(
      self,
      decoder_layer_class: type[nnx.Module],
      rngs: nnx.Rngs,
      **kwargs: Any,
  ) -> nnx.Module:
    return nnx_decoders.NNXDecoder._create_single_layer(  # pylint: disable=protected-access
        self, decoder_layer_class, rngs, **kwargs
    )

  def _create_and_register_layer(
      self,
      layer_cls: type[nnx.Module],
      rngs: nnx.Rngs,
      base_name: str,
      i: int,
      **layer_kwargs: Any,
  ) -> nnx.Module | None:
    """Instantiates only the requested layer index ``self._layer_index``."""
    del base_name
    if i != self._layer_index:
      return None
    self.built_layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
    return self.built_layer

  def instantiate_layer(
      self,
      block_cls: type[nnx.Module],
      rngs: nnx.Rngs,
      **extra_kwargs: Any,
  ) -> nnx.Module:
    """Directly instantiates a single layer of ``block_cls``."""
    self.built_layer = self._create_single_layer(block_cls, rngs, **extra_kwargs)
    return self.built_layer


@dataclasses.dataclass(frozen=True)
class PipelineGraphDef:
  """The static (non-array) half of a split :class:`PipelinedTransformer`.

  ``nnx.split`` separates a module into a ``GraphDef`` (structure, closed-over
  python values) and ``State`` (the arrays). Only the arrays may cross an MPMD
  computation boundary, so the model is split once up front and the graphdefs
  are captured as trace-time constants.

  Attributes:
    config: The MaxText config the model was built from.
    embed_def: GraphDef of the token embedder.
    embed_rest: Non-parameter state of the token embedder.
    layer_defs: Per-layer GraphDefs, ordered from input to output.
    layer_rests: Per-layer non-parameter state.
    unembed_def: GraphDef of the output head (final norm + projection).
    unembed_rest: Non-parameter state of the output projection.
    is_deepseek: Whether the decoder blocks are DeepSeek layers.
    remat_policy: The policy each decoder layer is checkpointed with
      (``Hparams.mpmd_remat_policy``: ``--remat_policy`` under ``--remat``, else
      ``none``). Not MaxText's ``remat_policy``, which pipeline parallelism keeps
      from being ``none`` (``config.build_maxtext_config``).
    stage_layer_counts: Decoder layers per stage (``--stage_layer_counts``);
      empty for a uniform split.
    fsdp_size: Size of the stage mesh's ``fsdp`` axis.
    mesh: Concrete stage mesh used to resolve parameter shardings inside stages.
  """

  config: Any
  embed_def: nnx.GraphDef
  embed_rest: nnx.State
  layer_defs: tuple[nnx.GraphDef, ...]
  layer_rests: tuple[nnx.State, ...]
  unembed_def: nnx.GraphDef
  unembed_rest: nnx.State
  is_deepseek: bool
  # Required, unlike the two below: a caller that forgot them would silently lose remat or uneven stages.
  remat_policy: str
  stage_layer_counts: tuple[int, ...]
  fsdp_size: int = 1
  mesh: Mesh | None = None

  @property
  def num_layers(self) -> int:
    """Returns the number of decoder layers in the model."""
    return len(self.layer_defs)

  @property
  def data_parallel_size(self) -> int:
    """Returns how many ways a microbatch is sharded inside a stage.

    This is the product of the stage mesh's ``fsdp`` and ``expert`` axes (the
    ``expert`` axis also shards the batch; see :mod:`mesh_utils`).
    """
    if self.mesh is None:
      return max(self.fsdp_size, 1)
    return max(mesh_utils.data_parallel_size(self.mesh), 1)


class OutputHead(nnx.Module):
  """MaxText's output head (``NNXDecoder.apply_output_head``) as one unit.

  The final norm (``decoder_norm``, the norm class MaxText's decoder picks for
  the decoder block: RMSNorm for most) and then the vocabulary projection
  (``logits_dense``). Without the norm the model was not MaxText's: its logits
  were unnormalised, and initial losses on the fixed synthetic batch were 24.8
  for llama3.1-8b (MaxText's own trainer: 12.3) and 21 for deepseek2-16b,
  against ln(vocab) of about 11.8. Dropout, which MaxText applies after the
  norm, is off in every config here (``config._DETERMINISM_OVERRIDES``).

  Only the untied head: ``config.build_maxtext_config`` refuses models with
  ``logits_via_embedding`` (tied embeddings, whose MaxText head also applies
  ``normalize_embedding_logits`` and ``final_logits_soft_cap``) unless
  ``--maxtext_overrides=logits_via_embedding=false`` asks for the untied
  variant, which this is. MaxText's default ``cast_logits_to_fp32`` is the
  loss's own float32 cast, and ``config._check_buildable`` refuses the head
  settings this does not mirror (``lm_head_fsdp_as_vocab_tp``, fp8 weights, ...).
  """

  def __init__(self, config: Any, mesh: Mesh, *, rngs: nnx.Rngs):
    norm_cls = nnx_decoders.NNXDecoder.get_norm_layer(
        _NNXDecoderLayerFactory(config, mesh, layer_index=-1),
        num_features=config.emb_dim,
        rngs=rngs,
    )
    self.decoder_norm = norm_cls(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
    )
    # As NNXDecoder builds it, except for the kernel's logical axes ("embed",
    # not "embed_vocab": the FSDP sharding here keys on "embed").
    self.logits_dense = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.vocab_size,
        kernel_axes=("embed", "vocab"),
        dtype=_unembed_output_dtype(config),
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        use_bias=False,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    return self.logits_dense(self.decoder_norm(hidden_states))


class PipelinedTransformer(nnx.Module):
  """A decoder-only transformer whose layers are individually addressable.

  Builds the decoder blocks in :data:`SUPPORTED_DECODER_BLOCKS` by delegating
  per-layer class lookup and keyword-argument resolution to MaxText's
  ``NNXDecoder``.
  """

  def __init__(
      self,
      config: Any,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
      remat_policy: str,
      stage_layer_counts: tuple[int, ...],
  ):
    """Builds the model.

    Args:
      config: An initialised MaxText config (see :mod:`config`).
      mesh: The mesh used to resolve MaxText's logical parameter shardings. This
        is the *global* sharding mesh, not a per-stage mesh: PartIR re-shards
        parameters onto their stage mesh when it partitions.
      rngs: Initialisation RNGs.
      remat_policy: See :attr:`PipelineGraphDef.remat_policy`.
      stage_layer_counts: See :attr:`PipelineGraphDef.stage_layer_counts`.
    """
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    self.stage_layer_counts = tuple(stage_layer_counts)

    self.token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        config=config,
        mesh=mesh,
        dtype=config.dtype,
        rngs=rngs,
    )

    # `config.decoder_block` is a `DecoderBlockType`, not a string: pyconfig
    # converts the YAML value with `DecoderBlockType(final_dict[...])`. And
    # `DecoderBlockType` is a plain `enum.Enum`, not a `str` mixin, so
    # `config.decoder_block == "deepseek"` is *always* False. Comparing against
    # the string meant every `--model_name=deepseek3-*` run silently built
    # eight dense `SimpleMlpDecoderLayer`s while MaxText -- which compares
    # against the enum properly -- charged the run DeepSeek's analytic FLOP
    # count, reporting an impossible 427 TFLOP/s/device.
    self.is_deepseek = config.decoder_block == DecoderBlockType.DEEPSEEK
    first_num_dense_layers = config.first_num_dense_layers

    # `nnx.List` (rather than a plain python list) so the layers are part of
    # the module graph and are picked up by `nnx.split`.
    self.layers = nnx.List()
    for layer_index in range(config.num_decoder_layers):
      self.layers.append(self._make_layer(config, mesh, rngs, layer_index, first_num_dense_layers))

    self.unembed = OutputHead(config, mesh, rngs=rngs)

  @staticmethod
  def _build_decoder_layer(
      config: Any,
      mesh: Mesh,
      rngs: nnx.Rngs,
      layer_index: int,
      first_num_dense_layers: int,
  ) -> nnx.Module:
    """Instantiates the decoder block for `layer_index`.

    Delegates class lookup and per-layer kwarg resolution to MaxText's
    :class:`nnx_decoders.NNXDecoder`.

    Raises:
      ValueError: If `config.decoder_block` is not one of
        :data:`SUPPORTED_DECODER_BLOCKS`.
    """
    decoder_block = config.decoder_block
    if decoder_block not in SUPPORTED_DECODER_BLOCKS:
      raise ValueError(
          f"Unsupported decoder_block {decoder_block!r}. Expected one of"
          f" {sorted(b.value for b in SUPPORTED_DECODER_BLOCKS)}; anything else"
          " would silently benchmark a different architecture than the one"
          " MaxText charges FLOPs for."
      )
    factory = _NNXDecoderLayerFactory(config, mesh, layer_index)
    if decoder_block == DecoderBlockType.DEEPSEEK:
      dense_cls, moe_cls = nnx_decoders.NNXDecoder.get_decoder_layers(factory)
      layer_cls = dense_cls if layer_index < first_num_dense_layers else moe_cls
      return factory.instantiate_layer(layer_cls, rngs, layer_idx=layer_index)
    try:
      decoder_block_classes = nnx_decoders.NNXDecoder.get_decoder_layers(factory)
    except NotImplementedError as exc:
      raise ValueError(f"Unsupported decoder_block {decoder_block!r}: {exc}") from exc
    nnx_decoders.NNXDecoder._init_sequential_generic(  # pylint: disable=protected-access
        factory, decoder_block_classes, rngs
    )
    if factory.built_layer is None:
      return factory.instantiate_layer(decoder_block_classes[0], rngs, layer_idx=layer_index)
    return factory.built_layer

  def _make_layer(
      self,
      config: Any,
      mesh: Mesh,
      rngs: nnx.Rngs,
      layer_index: int,
      first_num_dense_layers: int,
  ) -> nnx.Module:
    return self._build_decoder_layer(config, mesh, rngs, layer_index, first_num_dense_layers)

  def split(self) -> tuple[PipelineGraphDef, dict[str, nnx.State]]:
    """Separates the model into a static graphdef and a per-unit param dict.

    Returns:
      A tuple ``(graphdef, params)`` where `params` is keyed by the same names
      used for the MPMD named computations (``embed``, ``layer0``, ...,
      ``unembed``), so that ``params[name]`` is exactly the state the
      computation called `name` consumes.
    """
    embed_def, embed_params, embed_rest = nnx.split(self.token_embedder, nnx.Param, ...)
    params: dict[str, nnx.State] = {mesh_utils.EMBED_NAME: embed_params}

    layer_defs = []
    layer_rests = []
    for layer_index, layer in enumerate(self.layers):
      layer_def, layer_params, layer_rest = nnx.split(layer, nnx.Param, ...)
      layer_defs.append(layer_def)
      layer_rests.append(layer_rest)
      params[mesh_utils.layer_name(layer_index)] = layer_params

    unembed_def, unembed_params, unembed_rest = nnx.split(self.unembed, nnx.Param, ...)
    params[mesh_utils.UNEMBED_NAME] = unembed_params

    fsdp_size = int(self.mesh.shape.get(mesh_utils.FSDP_AXIS, 1))
    graphdef = PipelineGraphDef(
        config=self.config,
        embed_def=embed_def,
        embed_rest=embed_rest,
        layer_defs=tuple(layer_defs),
        layer_rests=tuple(layer_rests),
        unembed_def=unembed_def,
        unembed_rest=unembed_rest,
        is_deepseek=self.is_deepseek,
        fsdp_size=fsdp_size,
        mesh=self.mesh,
        remat_policy=self.remat_policy,
        stage_layer_counts=self.stage_layer_counts,
    )
    return graphdef, params

  @classmethod
  def init_split(
      cls,
      config: Any,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
      unit_shard_fn: Callable[[str, nnx.State], nnx.State] | None = None,
      remat_policy: str,
      stage_layer_counts: tuple[int, ...],
  ) -> tuple[PipelineGraphDef, dict[str, nnx.State]]:
    """Builds and splits the model one unit at a time, optionally sharding each.

    Draws from `rngs` in the exact same sequence as ``PipelinedTransformer(
    config, mesh, rngs=rngs, ...).split()``, while allowing `unit_shard_fn` to move
    each layer's parameters onto its target stage mesh and FSDP sharding before
    the next layer is allocated. This prevents OOMing ``device 0`` when
    initialising multi-billion-parameter models across many pipeline stages.

    Args:
      config: Initialised MaxText config.
      mesh: Concrete stage mesh used to resolve logical axis rules.
      rngs: Initialisation RNGs.
      unit_shard_fn: Optional callback ``(unit_name, unit_params) ->
        sharded_unit_params`` invoked immediately after each unit is split.
      remat_policy: See :attr:`PipelineGraphDef.remat_policy`.
      stage_layer_counts: See :attr:`PipelineGraphDef.stage_layer_counts`.

    Returns:
      ``(graphdef, params)`` identical in structure and values to :meth:`split`.
    """
    shard_fn = unit_shard_fn if unit_shard_fn is not None else (lambda _name, p: p)

    def shard_unit(name: str, module: nnx.Module, unit_params: nnx.State):
      sharded = shard_fn(name, unit_params)
      # The GraphDef can keep `module` alive: MaxText's Attention passes
      # `kernel_init=query_init`, a closure over the module, and the GraphDef
      # holds it as a static attribute. Pointing the module at the sharded
      # arrays lets the unsharded ones go; they were left on device 0 (5 GiB
      # for llama3.1-8b, 45 GiB for llama3.1-70b, on one stage-0 chip).
      nnx.update(module, sharded)
      return sharded

    token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        config=config,
        mesh=mesh,
        dtype=config.dtype,
        rngs=rngs,
    )
    embed_def, embed_params, embed_rest = nnx.split(token_embedder, nnx.Param, ...)
    params: dict[str, nnx.State] = {
        mesh_utils.EMBED_NAME: shard_unit(mesh_utils.EMBED_NAME, token_embedder, embed_params)
    }
    del token_embedder, embed_params

    is_deepseek = config.decoder_block == DecoderBlockType.DEEPSEEK
    first_num_dense_layers = config.first_num_dense_layers

    layer_defs = []
    layer_rests = []
    for layer_index in range(config.num_decoder_layers):
      layer = cls._build_decoder_layer(config, mesh, rngs, layer_index, first_num_dense_layers)
      layer_def, layer_params, layer_rest = nnx.split(layer, nnx.Param, ...)
      layer_defs.append(layer_def)
      layer_rests.append(layer_rest)
      unit_name = mesh_utils.layer_name(layer_index)
      params[unit_name] = shard_unit(unit_name, layer, layer_params)
      del layer, layer_params

    unembed = OutputHead(config, mesh, rngs=rngs)
    unembed_def, unembed_params, unembed_rest = nnx.split(unembed, nnx.Param, ...)
    params[mesh_utils.UNEMBED_NAME] = shard_unit(mesh_utils.UNEMBED_NAME, unembed, unembed_params)
    del unembed, unembed_params

    fsdp_size = int(mesh.shape.get(mesh_utils.FSDP_AXIS, 1))
    graphdef = PipelineGraphDef(
        config=config,
        embed_def=embed_def,
        embed_rest=embed_rest,
        layer_defs=tuple(layer_defs),
        layer_rests=tuple(layer_rests),
        unembed_def=unembed_def,
        unembed_rest=unembed_rest,
        is_deepseek=is_deepseek,
        fsdp_size=fsdp_size,
        mesh=mesh,
        remat_policy=remat_policy,
        stage_layer_counts=tuple(stage_layer_counts),
    )
    return graphdef, params


def _is_nnx_variable(node: Any) -> bool:
  """Whether `node` is an NNX variable (a parameter with its logical-axis metadata)."""
  return isinstance(node, nnx.Variable)


def _unembed_output_dtype(config: Any) -> Any:
  """Returns the vocabulary projection's dtype, as MaxText's ``logits_dense``."""
  if config.logits_dot_in_fp32:
    return jnp.float32
  return config.dtype


def _is_embed_logical_axis(axis_entry: Any) -> bool:
  """Returns whether a MaxText logical axis entry represents an FSDP embed dim."""
  if isinstance(axis_entry, str):
    return axis_entry.startswith("embed")
  if isinstance(axis_entry, (tuple, list)):
    return any(isinstance(a, str) and a.startswith("embed") for a in axis_entry)
  return False


# MaxText logical axis names that denote the *expert* dimension of a routed
# MoE weight (`RoutedMoE.wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")`).
_EXPERT_LOGICAL_AXES = frozenset({"exp", "expert_only"})


def _is_expert_logical_axis(axis_entry: Any) -> bool:
  """Returns whether a MaxText logical axis entry is the MoE expert dim."""
  if isinstance(axis_entry, str):
    return axis_entry in _EXPERT_LOGICAL_AXES
  if isinstance(axis_entry, (tuple, list)):
    return any(isinstance(a, str) and a in _EXPERT_LOGICAL_AXES for a in axis_entry)
  return False


def _extract_logical_axes(node: Any) -> tuple[Any, ...] | None:
  """Extracts MaxText logical axis names from an NNX Variable or VariableState."""
  metadata = node.get_metadata() if hasattr(node, "get_metadata") else {}
  raw = (
      metadata.get("out_sharding")
      or metadata.get("sharding_names")
      or metadata.get("sharding")
      or getattr(node, "out_sharding", None)
      or getattr(node, "sharding", None)
  )
  if raw is None or isinstance(raw, (NamedSharding, P)):
    return None
  if isinstance(raw, str):
    return (raw,)
  if isinstance(raw, (tuple, list)):
    return tuple(raw)
  return None


def _axes_entry(axes: tuple[str, ...]) -> Any:
  """Returns a PartitionSpec entry for `axes` (a name, a tuple, or None)."""
  if not axes:
    return None
  return axes[0] if len(axes) == 1 else tuple(axes)


@dataclasses.dataclass(frozen=True)
class LeafSharding:
  """How one parameter leaf is laid out at rest and while it is being used.

  Attributes:
    sharded: The at-rest sharding of the leaf (what the optimizer state holds).
      ``embed*`` dims are split over the stage's FSDP axes, and the ``exp`` dim
      of routed-MoE weights is split over the ``expert`` axis.
    gathered: The sharding the forward pass consumes. FSDP axes are gathered
      (every device sees the full ``embed`` dim), but the ``expert`` dim stays
      partitioned: with expert parallelism each device only ever runs its own
        experts, so there is nothing to gather.
    expert_sharded: Whether the leaf's expert dimension is partitioned over the
      ``expert`` mesh axis.
  """

  sharded: NamedSharding
  gathered: NamedSharding
  expert_sharded: bool = False

  @property
  def needs_gather(self) -> bool:
    """Returns whether using the leaf requires an FSDP all-gather."""
    return self.sharded.spec != self.gathered.spec


def leaf_shardings(
    mesh: Mesh,
    leaf: Any,
    fsdp_size: int,
    *,
    logical_axes: tuple[Any, ...] | None = None,
) -> LeafSharding:
  """Computes the at-rest and gathered shardings of `leaf` on a stage mesh.

  When `logical_axes` metadata from an :class:`nnx.Param` is available:

    * the dim annotated ``exp`` (the expert dim of ``RoutedMoE.wi_0`` /
      ``wi_1`` / ``wo``) is partitioned over the mesh's ``expert`` axis, if
      any. This is *expert parallelism*: each device owns whole experts and
      tokens travel to them, so these weights are never gathered;
    * the dim annotated ``embed*`` (such as axis 1 of ``MlpBlock.wo`` or
      ``RoutedMoE.wi_0`` and axis 2 of ``Attention.out`` or ``RoutedMoE.wo``)
      is FSDP-sharded over the data-parallel axes not already used by the
      expert dim -- ``fsdp`` for expert weights, ``('fsdp', 'expert')`` for
      everything else, exactly as MaxText's ``embed`` rule maps onto
      ``['fsdp', ..., 'expert']``. Sharding axis 0 of output projections or
      the expert dim of MoE tensors over FSDP devices would instead split
      their contracting / expert dimension, forcing cross-device ``bfloat16``
      reductions or all-to-all weight reshards.

  When no ``logical_axes`` are present (e.g. batch tensors), axis 0 is sharded
  over every data-parallel axis if divisible, else replicated.

  Args:
    mesh: The (possibly abstract) stage mesh the sharding refers to.
    leaf: The array (or ``ShapeDtypeStruct``) to shard.
    fsdp_size: Number of FSDP shards (the size of the ``fsdp`` axis when the
      mesh has one).
    logical_axes: Optional MaxText logical axis tuple from ``nnx.Param``.

  Returns:
    The :class:`LeafSharding` of the leaf.
  """
  shape = tuple(getattr(leaf, "shape", ()))
  if len(shape) < 2:
    replicated = NamedSharding(mesh, P())
    return LeafSharding(sharded=replicated, gathered=replicated)

  mesh_shape = dict(getattr(mesh, "shape", {}))
  dp_axes = mesh_utils.data_parallel_axes(mesh)
  axis_size = {
      name: (max(fsdp_size, 1) if name == mesh_utils.FSDP_AXIS else int(mesh_shape.get(name, 1))) for name in dp_axes
  }
  dp_axes = tuple(name for name in dp_axes if axis_size[name] > 1)

  if logical_axes:
    spec: list[Any] = [None] * len(shape)
    expert_size = axis_size.get(mesh_utils.EXPERT_AXIS, 1)
    expert_dim = None
    if expert_size > 1:
      for dim_index, axis_entry in enumerate(logical_axes):
        if dim_index < len(shape) and _is_expert_logical_axis(axis_entry) and shape[dim_index] % expert_size == 0:
          spec[dim_index] = mesh_utils.EXPERT_AXIS
          expert_dim = dim_index
          break
    gathered_spec = list(spec)

    gather_axes = tuple(name for name in dp_axes if name != mesh_utils.EXPERT_AXIS or expert_dim is None)
    gather_size = int(np.prod([axis_size[name] for name in gather_axes]))
    if gather_axes and gather_size > 1:
      for dim_index, axis_entry in enumerate(logical_axes):
        if (
            dim_index < len(shape)
            and dim_index != expert_dim
            and _is_embed_logical_axis(axis_entry)
            and shape[dim_index] % gather_size == 0
        ):
          spec[dim_index] = _axes_entry(gather_axes)
          break
    return LeafSharding(
        sharded=NamedSharding(mesh, P(*spec)),
        gathered=NamedSharding(mesh, P(*gathered_spec)),
        expert_sharded=expert_dim is not None,
    )

  dp_size = int(np.prod([axis_size[name] for name in dp_axes])) if dp_axes else 1
  if dp_axes and shape[0] % dp_size == 0:
    sharded = NamedSharding(mesh, P(_axes_entry(dp_axes)))
  else:
    sharded = NamedSharding(mesh, P())
  return LeafSharding(sharded=sharded, gathered=sharded)


def shard_leaf_on_fsdp(
    mesh: Mesh,
    leaf: Any,
    fsdp_size: int,
    *,
    logical_axes: tuple[Any, ...] | None = None,
) -> NamedSharding:
  """Returns the at-rest sharding of `leaf`; see :func:`leaf_shardings`."""
  return leaf_shardings(mesh, leaf, fsdp_size, logical_axes=logical_axes).sharded


def shard_tree_on_fsdp(
    mesh: Mesh,
    tree: Any,
    fsdp_size: int,
    *,
    device_put: bool = False,
) -> Any:
  """Computes (or applies) FSDP shardings across a pytree, preserving NNX axes."""

  def _map_node(node: Any) -> Any:
    logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None

    def _map_leaf(leaf: Any) -> Any:
      sharding = shard_leaf_on_fsdp(mesh, leaf, fsdp_size, logical_axes=logical_axes)
      return jax.device_put(leaf, sharding) if device_put else sharding

    return jax.tree.map(_map_leaf, node)

  return jax.tree.map(_map_node, tree, is_leaf=_is_nnx_variable)


def _has_fsdp_axis(spec: P) -> bool:
  """Returns whether `spec` partitions any dimension over ``fsdp``."""
  for axis in spec:
    if axis == mesh_utils.FSDP_AXIS:
      return True
    if isinstance(axis, tuple) and mesh_utils.FSDP_AXIS in axis:
      return True
  return False


def _remove_fsdp_from_spec(spec: P) -> P:
  """Returns `spec` with ``fsdp`` replaced by replication (`None`)."""
  new_axes = []
  for axis in spec:
    if axis == mesh_utils.FSDP_AXIS:
      new_axes.append(None)
    elif isinstance(axis, tuple):
      filtered = tuple(a for a in axis if a != mesh_utils.FSDP_AXIS)
      new_axes.append(filtered if filtered else None)
    else:
      new_axes.append(axis)
  return P(*new_axes)


def _target_param_dtype(config: Any) -> jnp.dtype:
  """Returns the compute dtype a parameter leaf is cast to in forward (MaxText's ``dtype``)."""
  return jnp.dtype(config.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _bind_pregathered(
    sharding: NamedSharding,
    sharded_dtype: Any,
    w_sharded: jax.Array,
    w_gathered: jax.Array,
) -> jax.Array:
  """Substitutes pre-cast, FSDP-gathered weights in forward & remat passes.

  Forward and ``jax.checkpoint`` rematerialisation return `w_gathered` directly
  with zero tensor residuals, avoiding repeated ``float32 -> bfloat16`` casts
  and FSDP ``all-gather``s on every microbatch. In the backward pass, the
  cotangent `g_gathered` (`bfloat16`) is constrained to `sharding` so XLA fuses
  the gradient reduction into a ``bfloat16`` ``reduce-scatter`` before casting
  only the FSDP shard back to `sharded_dtype` (`float32`).
  """
  del sharding, sharded_dtype, w_sharded
  return w_gathered


def _bind_pregathered_fwd(
    sharding: NamedSharding,
    sharded_dtype: Any,
    w_sharded: jax.Array,
    w_gathered: jax.Array,
) -> tuple[jax.Array, None]:
  del sharding, sharded_dtype, w_sharded
  return w_gathered, None


def _bind_pregathered_bwd(
    sharding: NamedSharding,
    sharded_dtype: Any,
    unused_res: None,
    g_gathered: jax.Array,
) -> tuple[jax.Array, None]:
  g_sharded = jax.lax.with_sharding_constraint(g_gathered, sharding)
  return (g_sharded.astype(sharded_dtype), None)


_bind_pregathered.defvjp(_bind_pregathered_fwd, _bind_pregathered_bwd)


def _pregather_candidate(
    mesh: Mesh,
    leaf: Any,
    fsdp_size: int,
    logical_axes: tuple[Any, ...] | None,
    target_dtype: jnp.dtype,
) -> LeafSharding | None:
  """Returns the leaf's shardings if it should be pre-cast / pre-gathered."""
  if getattr(leaf, "ndim", 0) < 2:
    return None
  shardings = leaf_shardings(mesh, leaf, fsdp_size, logical_axes=logical_axes)
  if shardings.needs_gather or (shardings.expert_sharded and jnp.dtype(leaf.dtype) != target_dtype):
    return shardings
  return None


def _resident_bytes(mesh: Mesh, leaf: Any, sharding: NamedSharding, itemsize: int) -> int:
  """Returns the per-device byte footprint of `leaf` laid out as `sharding`."""
  mesh_shape = dict(getattr(mesh, "shape", {}))
  num_shards = 1
  for entry in sharding.spec:
    axes = (entry,) if isinstance(entry, str) else tuple(entry or ())
    for axis in axes:
      num_shards *= int(mesh_shape.get(axis, 1))
  return int(np.prod(leaf.shape)) * itemsize // max(num_shards, 1)


def _unit_pregather_bytes(
    mesh: Mesh,
    unit_params: nnx.State,
    fsdp_size: int,
    config: Any,
) -> int:
  """Returns the per-device target-dtype footprint of pre-gathered weights."""
  total_bytes = 0

  leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(unit_params, is_leaf=_is_nnx_variable)
  for _, node in leaves_with_path:
    logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None
    target_dtype = _target_param_dtype(config)
    for leaf in jax.tree_util.tree_leaves(node):
      shardings = _pregather_candidate(mesh, leaf, fsdp_size, logical_axes, target_dtype)
      if shardings is not None:
        total_bytes += _resident_bytes(mesh, leaf, shardings.gathered, target_dtype.itemsize)

  return total_bytes


def _stage_pregather_budget_bytes(mesh: Mesh | None) -> int:
  """Returns the per-stage HBM budget in bytes for FSDP weight pregathering.

  Scales the budget with the target accelerator's HBM capacity (24 GiB on 95 GiB
  TPU v7x and v5p, 8 GiB on 32 GiB TPU v6e, 2 GiB on 16 GiB TPU v5e / CPU),
  unless a test sets :data:`_MAX_UNIT_PREGATHER_BYTES`. A v5p matches the
  ``v5p`` tag through Pathways, whose devices report ``TPU v5p``; libtpu alone
  reports ``TPU v5``.

  Args:
    mesh: Optional JAX device mesh used to inspect accelerator `device_kind`.
  """
  if _MAX_UNIT_PREGATHER_BYTES != _DEFAULT_UNIT_PREGATHER_BYTES:
    return _MAX_UNIT_PREGATHER_BYTES
  if mesh is not None and getattr(mesh, "devices", None) is not None:
    flat_devices = np.asarray(mesh.devices).flat
    if len(flat_devices) > 0:
      kind = str(getattr(flat_devices[0], "device_kind", "")).lower()
      if any(tag in kind for tag in ("v7", "tpu7", "v5p")):
        return _V7X_STAGE_PREGATHER_BYTES
      if any(tag in kind for tag in ("v6", "trillium")):
        return _V6E_STAGE_PREGATHER_BYTES
  return _MAX_UNIT_PREGATHER_BYTES


def pregather_params(
    graphdef: PipelineGraphDef,
    params: dict[str, nnx.State],
    *,
    microbatch_rows: int,
    wrap: Wrapper = NAMED_COMPUTATION,
) -> dict[str, Any] | None:
  """Pre-casts and all-gathers FSDP-sharded stage weights once per step.

  In PartIR MPMD, microbatches are unrolled across separate ``mpmd.call``
  invocations (`fwd_call0..N`, `bwd_call0..N`). Because XLA cannot hoist loop-
  invariant weight casts or FSDP ``all-gather``s across ``mpmd.call`` or
  ``jax.checkpoint`` boundaries, passing FSDP-sharded ``float32`` weights
  directly into each microbatch causes XLA to execute ``convert(f32 -> bf16)``
  and ``all-gather(bf16)`` `2 * num_microbatches` times per step (once per
  forward microbatch and once per rematerialised backward microbatch).

  Wrapping each stage's weight cast and FSDP all-gather in a dedicated pre-loop
  ``@wrap(pre_stage{s})`` computation with a distinct ``stage_id`` prevents
  Shardy's ``MergeUserDefinedFragmentsIntoSchedulingUnitsPass`` from merging
  the weight all-gather into ``fwd_call0``. Because ``pre_stage{s}`` depends
  only on stage ``s``'s own ``state.params``, IFRT dispatches all stages'
  weight all-gathers in parallel at ``t = 0`` before ``fwd_call0`` begins,
  eliminating the ``(num_stages - 1) * T_AG`` warmup serialization bubble while
  reusing the gathered ``bfloat16`` weights across all forward and backward
  microbatches.

  Applies to ``embed`` (stage 0), ``layer0..layer_{N-1}``, and ``unembed``
  (stage ``num_stages - 1``). Pre-gathering is enabled on stages whose
  cumulative unsharded weight footprint fits within
  :func:`_stage_pregather_budget_bytes`.

  Args:
    graphdef: Static model structure.
    params: Per-unit parameter state.
    microbatch_rows: Rows of each microbatch. Unless the stage's data-parallel
      devices share them (:func:`_microbatch_shards_in_stage`), no 3D expert
      weight whose reduce-scatter would defer
      (:func:`_should_defer_bwd_reduce_scatter`) is gathered, so none defers.
    wrap: Stage wrapping strategy.

  Returns:
    A dict mapping unit names to their pre-cast, FSDP-gathered parameter
    trees, or ``None`` when FSDP pregathering is not applicable.
  """
  if wrap is UNWRAPPED or graphdef.data_parallel_size <= 1 or graphdef.mesh is None:
    return None

  num_stages = max(1, graphdef.config.ici_pipeline_parallelism)
  stage_budget = _stage_pregather_budget_bytes(graphdef.mesh)
  if stage_budget <= 0:
    return None

  unit_stages = _unit_stages(graphdef, params)

  unit_bytes = {
      name: _unit_pregather_bytes(
          graphdef.mesh,
          params[name],
          graphdef.fsdp_size,
          graphdef.config,
      )
      for name, _ in unit_stages
  }
  stage_bytes = [0] * num_stages
  for name, stage_idx in unit_stages:
    nbytes = unit_bytes[name]
    if 0 < nbytes <= stage_budget:
      stage_bytes[stage_idx] += nbytes

  pregathered: dict[str, Any] = {name: None for name, _ in unit_stages}
  any_pregathered = False

  # Which units fit the budget, decided before anything is gathered.
  gathered_units = {
      name
      for name, stage_idx in unit_stages
      if 0 < stage_bytes[stage_idx] <= stage_budget and 0 < unit_bytes[name] <= stage_budget
  }
  # `defer_fsdp_shard_map_psum` removes the per-microbatch FSDP psum of every
  # 3D expert weight at once, and only gathered expert weights get the deferred
  # psum_scatter that replaces it. So the expert weights are gathered (and
  # their reduce-scatter deferred) for every unit holding them, or for none.
  # The expert gradients of the units left out used to keep one device's
  # tokens: mixtral-8x7b's 2.7 GiB MoE layers miss a 2 GiB budget while embed
  # and unembed fit, and on a v5p (24 GiB) --stage_layer_counts=7,9,8,8 leaves
  # out stage 1 (24.3 GiB) alone. Nor are they gathered for a microbatch the
  # stage's data-parallel devices cannot share: MaxText's MoE shard_map then
  # runs all of it on every device, each holds the whole expert gradient, and
  # the deferred psum_scatter added up fsdp copies of it (4x at FSDP=4 with the
  # default batch: 2-row microbatches); the shard_map's own psum is right then.
  deferrable = _deferrable_units(graphdef, params)
  gather_experts = (
      bool(deferrable) and deferrable <= gathered_units and _microbatch_shards_in_stage(graphdef, microbatch_rows)
  )

  def _gather_unit(lp):
    def _gather_node(path, node):
      logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None
      target_dtype = _target_param_dtype(graphdef.config)

      def _gather_leaf(leaf):
        if not gather_experts and _should_defer_bwd_reduce_scatter(
            leaf,
            shard_leaf_on_fsdp(
                graphdef.mesh,
                leaf,
                graphdef.fsdp_size,
                logical_axes=logical_axes,
            ),
            graphdef.config,
            logical_axes=logical_axes,
        ):
          return None
        shardings = _pregather_candidate(
            graphdef.mesh,
            leaf,
            graphdef.fsdp_size,
            logical_axes,
            target_dtype,
        )
        if shardings is None:
          return None
        return jax.lax.with_sharding_constraint(leaf.astype(target_dtype), shardings.gathered)

      return jax.tree.map(_gather_leaf, node)

    return jax.tree_util.tree_map_with_path(_gather_node, lp, is_leaf=_is_nnx_variable)

  for stage_idx in range(num_stages):
    stage_unit_names = [name for name, s in unit_stages if s == stage_idx and name in gathered_units]
    if not stage_unit_names:
      continue

    any_pregathered = True
    pre_name = mesh_utils.pre_stage_name(stage_idx)
    stage_params = {uname: params[uname] for uname in stage_unit_names}

    def _pregather_stage(sp):
      return {uname: _gather_unit(ulp) for uname, ulp in sp.items()}

    pregathered.update(wrap(pre_name)(_pregather_stage)(stage_params))

  return pregathered if any_pregathered else None


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _cast_sharded(
    sharding: NamedSharding,
    sharded_dtype: Any,
    target_dtype: Any,
    w_sharded: jax.Array,
) -> jax.Array:
  """Casts an FSDP-sharded parameter to `target_dtype` before `all-gather`."""
  del sharded_dtype
  return jax.lax.with_sharding_constraint(w_sharded.astype(target_dtype), sharding)


def _cast_sharded_fwd(
    sharding: NamedSharding,
    sharded_dtype: Any,
    target_dtype: Any,
    w_sharded: jax.Array,
) -> tuple[jax.Array, None]:
  return (
      _cast_sharded(sharding, sharded_dtype, target_dtype, w_sharded),
      None,
  )


def _cast_sharded_bwd(
    sharding: NamedSharding,
    sharded_dtype: Any,
    target_dtype: Any,
    unused_res: None,
    g_target: jax.Array,
) -> tuple[jax.Array]:
  g_sharded = jax.lax.with_sharding_constraint(g_target.astype(target_dtype), sharding)
  return (g_sharded.astype(sharded_dtype),)


_cast_sharded.defvjp(_cast_sharded_fwd, _cast_sharded_bwd)


def _cast_sharded_layer_params(
    layer_params: nnx.State,
    mesh: Mesh | None,
    fsdp_size: int,
    config: Any,
) -> nnx.State:
  """Casts FSDP-sharded parameters to compute dtype while keeping them sharded."""
  if mesh is None or mesh_utils.data_parallel_size(mesh) <= 1:
    return layer_params

  def _cast_node(path: tuple[Any, ...], node: Any) -> Any:
    logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None
    target_dtype = _target_param_dtype(config)

    def _cast_leaf(leaf: Any) -> Any:
      if getattr(leaf, "ndim", 0) < 2 or leaf.dtype == target_dtype:
        return leaf
      shardings = leaf_shardings(mesh, leaf, fsdp_size, logical_axes=logical_axes)
      if not shardings.needs_gather:
        return leaf
      return _cast_sharded(shardings.sharded, leaf.dtype, target_dtype, leaf)

    return jax.tree.map(_cast_leaf, node)

  return jax.tree_util.tree_map_with_path(_cast_node, layer_params, is_leaf=_is_nnx_variable)


def _should_defer_bwd_reduce_scatter(
    leaf: Any,
    sharding: NamedSharding,
    config: Any,
    *,
    logical_axes: tuple[Any, ...] | None = None,
) -> bool:
  """Returns whether a pre-gathered weight should defer FSDP reduce-scatter.

  Never when this jax lacks the private function the deferral patches
  (:func:`can_defer_moe_reduce_scatter`): every expert gradient is then reduced
  inside its microbatch, as MaxText's shard_map transpose does by itself.
  """
  is_moe_expert_3d = logical_axes is None or (len(logical_axes) == 3 and logical_axes[0] == "exp")
  return (
      can_defer_moe_reduce_scatter()
      and getattr(leaf, "ndim", 0) == 3
      and is_moe_expert_3d
      and config.sparse_matmul
      and not config.check_vma
      and _has_fsdp_axis(sharding.spec)
  )


def _deferrable_units(graphdef: PipelineGraphDef, params: dict[str, nnx.State]) -> set[str]:
  """Names of the units holding a 3D expert weight whose reduce-scatter defers.

  See :func:`_should_defer_bwd_reduce_scatter`; empty without FSDP, without
  sparse_matmul, or with check_vma.
  """
  if graphdef.mesh is None or graphdef.fsdp_size <= 1 or not graphdef.config.sparse_matmul or graphdef.config.check_vma:
    return set()
  names = set()
  for name, unit_params in params.items():
    for node in jax.tree_util.tree_leaves(unit_params, is_leaf=_is_nnx_variable):
      logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None
      for leaf in jax.tree_util.tree_leaves(node):
        sharding = shard_leaf_on_fsdp(graphdef.mesh, leaf, graphdef.fsdp_size, logical_axes=logical_axes)
        if _should_defer_bwd_reduce_scatter(leaf, sharding, graphdef.config, logical_axes=logical_axes):
          names.add(name)
  return names


def _defers_moe_reduce_scatter(
    graphdef: PipelineGraphDef,
    params: dict[str, nnx.State],
    pregathered_params: dict[str, Any] | None,
) -> bool:
  """Whether this step defers 3D expert gradients' FSDP reduce-scatter.

  It does when the deferrable units' expert weights are pre-gathered:
  :func:`pregather_params` gathers them for all such units or for none (see
  there), because :func:`defer_fsdp_shard_map_psum` removes the per-microbatch
  psum for every 3D expert weight at once.
  """
  if pregathered_params is None:
    return False
  return any(
      pregathered_params.get(name) is not None
      and _build_unit_deferred_meta(graphdef, params[name], pregathered_params[name])[1]
      for name in _deferrable_units(graphdef, params)
  )


def build_grad_carry(
    graphdef: PipelineGraphDef,
    params: dict[str, nnx.State],
    pregathered_params: dict[str, Any] | None,
) -> dict[str, nnx.State]:
  """Replaces deferred 3D MoE leaves with `pregathered_params` (`bfloat16`).

  For 3D MoE weights whose backward FSDP ``reduce-scatter`` is deferred to
  :func:`reduce_scatter_deferred_grads`, the per-microbatch backward pass
  produces unsharded ``bfloat16`` gradients (`15 GiB`/stage). Using the
  already-gathered ``bfloat16`` weight leaves from `pregathered_params` as the
  differentiation target and VJP carry keeps the microbatch gradient accumulator
  in ``bfloat16`` (`15 GiB`/stage instead of `30 GiB`/stage in ``float32``),
  avoiding a `15 GiB` zero-initialized accumulator on microbatch 0, per-
  microbatch ``bfloat16 -> float32`` casts, and post-loop ``float32 ->
  bfloat16`` truncation before ``psum_scatter``.

  Args:
    graphdef: Static model structure.
    params: Sharded model parameters tree.
    pregathered_params: Pre-gathered parameter tree from
      :func:`pregather_params`.

  Returns:
    A parameter PyTree with deferred 3D MoE leaves replaced by their
    ``bfloat16`` pre-gathered counterparts.
  """
  if (
      pregathered_params is None
      or graphdef.fsdp_size <= 1
      or graphdef.mesh is None
      or not graphdef.config.sparse_matmul
      or graphdef.config.check_vma
  ):
    return params

  out_carry: dict[str, nnx.State] = {}

  for name, unit_params in params.items():
    unit_preg = pregathered_params.get(name)
    if unit_preg is None:
      out_carry[name] = unit_params
      continue

    def _select_node(param_node, preg_node):
      if preg_node is None:
        return param_node
      logical_axes = _extract_logical_axes(param_node) if _is_nnx_variable(param_node) else None

      def _select_leaf(param_leaf, preg_leaf):
        if preg_leaf is None:
          return param_leaf
        sharding = shard_leaf_on_fsdp(
            graphdef.mesh,
            param_leaf,
            graphdef.fsdp_size,
            logical_axes=logical_axes,
        )
        if _should_defer_bwd_reduce_scatter(
            param_leaf,
            sharding,
            graphdef.config,
            logical_axes=logical_axes,
        ):
          return preg_leaf
        return param_leaf

      return jax.tree.map(
          _select_leaf,
          param_node,
          preg_node,
          is_leaf=lambda x: x is None,
      )

    out_carry[name] = jax.tree.map(
        _select_node,
        unit_params,
        unit_preg,
        is_leaf=lambda x: _is_nnx_variable(x) or x is None,
    )

  return out_carry


def can_defer_moe_reduce_scatter() -> bool:
  """Whether this jax has the private function :func:`suppress_fsdp_psum_of_3d_inputs` patches.

  The deferred expert-gradient reduce-scatter replaces
  ``jax._src.shard_map._unmentioned2``, which the transpose of a
  ``check_vma=False`` shard_map looks up at call time. Without it the step
  reduces every expert gradient inside its microbatch (correct, but once per
  microbatch), and ``models_test`` fails, so that a rename in jax is noticed.
  """
  available = callable(getattr(jax_shard_map, "_unmentioned2", None))
  if not available:
    _warn_no_deferral()
  return available


@functools.cache
def _warn_no_deferral() -> None:
  max_logging.warning(
      "jax._src.shard_map._unmentioned2 is gone: MoE expert gradients are reduce-scattered once per microbatch"
      " instead of once per step."
  )


@contextlib.contextmanager
def suppress_fsdp_psum_of_3d_inputs():
  """Drops the ``fsdp`` psum the transpose of a ``check_vma=False`` shard_map adds to 3D inputs.

  During VJP transposition, ``jax._src.shard_map._shard_map_transpose_fancy``
  psums the cotangent of every input whose ``in_specs`` entry does not mention a
  manual mesh axis over that axis (``_unmentioned2``). Inside this context, a
  3D input (a routed-MoE expert weight) keeps its per-device partial gradient
  over ``fsdp`` instead. This patches private JAX; see
  :func:`can_defer_moe_reduce_scatter`.

  Yields:
    Nothing; the patch is undone on exit.
  """
  orig_unmentioned2 = jax_shard_map._unmentioned2  # pylint: disable=protected-access

  def _deferred_unmentioned2(mesh, spec, manual_axes):
    axes = orig_unmentioned2(mesh, spec, manual_axes)
    if len(spec) == 3 and mesh_utils.FSDP_AXIS in axes:
      return [a for a in axes if a != mesh_utils.FSDP_AXIS]
    return axes

  jax_shard_map._unmentioned2 = _deferred_unmentioned2  # pylint: disable=protected-access
  try:
    yield
  finally:
    jax_shard_map._unmentioned2 = orig_unmentioned2  # pylint: disable=protected-access


@contextlib.contextmanager
def defer_fsdp_shard_map_psum(
    graphdef: PipelineGraphDef,
    pregathered_params: dict[str, Any] | None,
    params: dict[str, nnx.State],
):
  """Suppresses per-microbatch FSDP psum on 3D MoE weights inside shard_map.

  MaxText's ``RoutedMoE.sparse_matmul_route_and_compute`` wraps each MoE layer
  in ``@jax.shard_map(..., in_specs=(..., w0_pspec, w1_pspec, wo_pspec, ...),
  check_vma=False)`` where ``w0_pspec`` is ``P(None, None, None)``. During VJP
  transposition the shard_map inserts a defensive ``lax.psum(..., "fsdp")`` on
  every microbatch for any input whose spec does not mention ``"fsdp"``, forcing
  an N-microbatch redundant FSDP ``reduce-scatter`` inside the microbatch loop.
  :func:`suppress_fsdp_psum_of_3d_inputs` keeps each device's partial expert
  gradient unreduced during the microbatch loop so
  :func:`reduce_scatter_deferred_grads` can execute a single ``psum_scatter``
  once per step after the microbatch loop.

  Only when the step defers (:func:`_defers_moe_reduce_scatter`): the patch
  applies to every 3D expert weight's shard_map, so every one of them must be
  reduced later by :func:`reduce_scatter_deferred_grads`.

  Args:
    graphdef: Static model structure.
    pregathered_params: Pre-gathered parameter tree from
      :func:`pregather_params`.
    params: The step's parameters (to see which expert weights were gathered).

  Yields:
    Nothing; the patch, if any, is undone on exit.
  """
  if (
      pregathered_params is None
      or graphdef.fsdp_size <= 1
      or not graphdef.config.sparse_matmul
      or graphdef.config.check_vma
      or not _defers_moe_reduce_scatter(graphdef, params, pregathered_params)
  ):
    yield
    return
  with suppress_fsdp_psum_of_3d_inputs():
    yield


def _unit_stages(
    graphdef: PipelineGraphDef,
    units: dict[str, Any],
) -> list[tuple[str, int]]:
  """Maps each model unit in ``units`` to its contiguous pipeline stage index."""
  num_stages = max(1, graphdef.config.ici_pipeline_parallelism)
  stage_layer_counts = graphdef.stage_layer_counts
  if stage_layer_counts or (graphdef.num_layers >= num_stages and graphdef.num_layers % num_stages == 0):
    layer_stages = mesh_utils.layer_stage_indices(
        num_layers=graphdef.num_layers,
        num_stages=num_stages,
        stage_layer_counts=stage_layer_counts,
    )
  else:
    layers_per_stage = max(1, graphdef.num_layers // num_stages)
    layer_stages = [min(num_stages - 1, i // layers_per_stage) for i in range(graphdef.num_layers)]
  unit_stages: list[tuple[str, int]] = []
  seen: set[str] = set()
  if mesh_utils.EMBED_NAME in units:
    unit_stages.append((mesh_utils.EMBED_NAME, 0))
    seen.add(mesh_utils.EMBED_NAME)
  for i, stage_idx in enumerate(layer_stages):
    name = mesh_utils.layer_name(i)
    if name in units:
      unit_stages.append((name, stage_idx))
      seen.add(name)
  if mesh_utils.UNEMBED_NAME in units:
    unit_stages.append((mesh_utils.UNEMBED_NAME, num_stages - 1))
    seen.add(mesh_utils.UNEMBED_NAME)
  for name in units:
    if name not in seen:
      unit_stages.append((name, 0))
  return unit_stages


@dataclasses.dataclass(frozen=True)
class _DeferredGradMeta:
  """Static metadata for deferred 3D MoE gradient reduce-scatter."""

  sharding_spec: P
  no_fsdp_spec: P
  fsdp_dim: int
  target_dtype: Any
  out_dtype: Any


def _build_unit_deferred_meta(
    graphdef: PipelineGraphDef,
    unit_params: nnx.State | None,
    unit_preg: Any,
) -> tuple[Any, bool]:
  """Builds a static metadata tree (no JAX arrays) for deferred 3D MoE grads."""
  if (
      unit_params is None
      or unit_preg is None
      or graphdef.fsdp_size <= 1
      or graphdef.mesh is None
      or not graphdef.config.sparse_matmul
      or graphdef.config.check_vma
  ):
    return None, False

  found_deferred = False

  def _meta_node(path, param_node, preg_node):
    nonlocal found_deferred
    if preg_node is None:
      return jax.tree.map(lambda _: None, param_node)
    logical_axes = _extract_logical_axes(param_node) if _is_nnx_variable(param_node) else None
    target_dtype = _target_param_dtype(graphdef.config)

    def _meta_leaf(param_leaf, preg_leaf):
      nonlocal found_deferred
      if preg_leaf is None:
        return None
      sharding = shard_leaf_on_fsdp(
          graphdef.mesh,
          param_leaf,
          graphdef.fsdp_size,
          logical_axes=logical_axes,
      )
      if not _should_defer_bwd_reduce_scatter(
          param_leaf,
          sharding,
          graphdef.config,
          logical_axes=logical_axes,
      ):
        return None
      fsdp_dim = next(
          i
          for i, a in enumerate(sharding.spec)
          if a == mesh_utils.FSDP_AXIS or (isinstance(a, tuple) and mesh_utils.FSDP_AXIS in a)
      )
      found_deferred = True
      return _DeferredGradMeta(
          sharding_spec=sharding.spec,
          no_fsdp_spec=_remove_fsdp_from_spec(sharding.spec),
          fsdp_dim=fsdp_dim,
          target_dtype=target_dtype,
          out_dtype=param_leaf.dtype,
      )

    return jax.tree.map(
        _meta_leaf,
        param_node,
        preg_node,
        is_leaf=lambda x: x is None,
    )

  meta_tree = jax.tree_util.tree_map_with_path(
      _meta_node,
      unit_params,
      unit_preg,
      is_leaf=lambda x: _is_nnx_variable(x) or x is None,
  )
  return (meta_tree if found_deferred else None), found_deferred


def reduce_scatter_deferred_grads(
    graphdef: PipelineGraphDef,
    params: dict[str, nnx.State],
    pregathered_params: dict[str, Any] | None,
    grads: dict[str, nnx.State],
    *,
    num_microbatches: int = 1,
    wrap: Wrapper = NAMED_COMPUTATION,
) -> dict[str, nnx.State]:
  """Finalizes accumulated gradients once per step after the microbatch loop.

  Performs the deferred ``bfloat16`` FSDP ``psum_scatter`` on 3D MoE expert
  weight gradients and scales all gradients by ``1 / num_microbatches``.
  Under MPMD (`wrap is NAMED_COMPUTATION`), downstream stages (`stage_idx > 0`)
  wrap their post-loop gradient finalization in a dedicated
  ``@wrap(post_m{s})`` user fragment taking only that stage's accumulated
  gradients as traced inputs (using precomputed static
  :class:`_DeferredGradMeta`
  leaves so ``params`` and ``pregathered_params`` are never captured as traced
  arrays). This prevents Shardy's ``MergeInferredFragmentsPass`` from merging
  post-loop gradient reduce-scatter and AdamW updates into the last microbatch's
  backward fragment (``bwd_call_{M-1}``) on downstream stages, allowing them to
  emit ``dX`` upstream immediately and execute ``reduce_scatter + AdamW`` in
  parallel during cooldown. Because Stage 0 (`stage_idx == 0`) has no upstream
  stage waiting on ``dX`` when ``num_stages > 1``, its gradient finalization is
  left unwrapped when it has deferred reduce-scatters, so
  ``MergeInferredFragmentsPass`` fuses Stage 0's ``psum_scatter`` and AdamW
  updates into ``stage0_bwd_call_{M-1}``, allowing XLA's
  ``LatencyHidingScheduler`` to overlap Stage 0's SparseCore reduce-scatters
  with TensorCore backward compute across layers. Without one there is nothing
  to overlap, and the fused update only raises that program's memory: on
  deepseek2-16b with expert parallelism (no FSDP) on a v5p-64, stage 0's HBM
  was full and ``stage0_bwd_call_7`` with the update failed to load every step
  (``RuntimeProgramAllocationFailure``, 6.1 GB scoped), the runtime
  defragmented HBM, and the loss went NaN. Stage 0 then gets ``post_m0`` like
  the others.

  Args:
    graphdef: Static structure produced by :meth:`PipelinedTransformer.split`.
    params: Per-unit parameter state tree.
    pregathered_params: Optional pre-gathered ``bfloat16`` parameter tree.
    grads: Accumulated per-unit gradient state tree.
    num_microbatches: Number of microbatches accumulated before scaling.
    wrap: Stage wrapping strategy.

  Returns:
    The finalized per-unit ``float32`` gradient tree.
  """
  unit_metas: dict[str, Any] = {}
  unit_has_deferred: dict[str, bool] = {}
  for name in grads:
    unit_preg = pregathered_params.get(name) if pregathered_params is not None else None
    meta_tree, has_def = _build_unit_deferred_meta(graphdef, params.get(name), unit_preg)
    unit_metas[name] = meta_tree
    unit_has_deferred[name] = has_def

  def _finalize_unit_grads(
      unit_grads: nnx.State,
      unit_meta: Any,
  ) -> nnx.State:
    if unit_meta is None:
      if num_microbatches != 1:
        return jax.tree.map(lambda g: g / num_microbatches, unit_grads)
      return unit_grads

    def _finalize_leaf(g_leaf: jax.Array, meta: _DeferredGradMeta | None):
      if meta is not None:

        @functools.partial(
            jax.shard_map,
            mesh=graphdef.mesh,
            in_specs=(meta.no_fsdp_spec,),
            out_specs=meta.sharding_spec,
            check_vma=False,
        )
        def _psum_scatter(g):
          return jax.lax.psum_scatter(
              g.astype(meta.target_dtype),
              mesh_utils.FSDP_AXIS,
              scatter_dimension=meta.fsdp_dim,
              tiled=True,
          ).astype(meta.out_dtype)

        g_leaf = _psum_scatter(g_leaf)
      if num_microbatches != 1:
        g_leaf = g_leaf / num_microbatches
      return g_leaf

    return jax.tree.map(_finalize_leaf, unit_grads, unit_meta)

  if wrap is UNWRAPPED:
    return {name: _finalize_unit_grads(ug, unit_metas.get(name)) for name, ug in grads.items()}

  num_stages = max(1, graphdef.config.ici_pipeline_parallelism)
  unit_stages = _unit_stages(graphdef, grads)
  out_grads = dict(grads)

  for stage_idx in range(num_stages):
    stage_names = [name for name, s in unit_stages if s == stage_idx]
    if not stage_names:
      continue
    stage_has_deferred = any(unit_has_deferred.get(name, False) for name in stage_names)
    if num_microbatches == 1 and not stage_has_deferred:
      continue
    if stage_idx == 0 and num_stages > 1 and stage_has_deferred:
      for uname in stage_names:
        out_grads[uname] = _finalize_unit_grads(grads[uname], unit_metas.get(uname))
      continue
    post_name = mesh_utils.post_stage_name(stage_idx)
    stage_grads = {name: grads[name] for name in stage_names}
    stage_metas = {name: unit_metas.get(name) for name in stage_names}

    def _finalize_stage(sg, smetas=stage_metas):
      return {uname: _finalize_unit_grads(ugrad, smetas[uname]) for uname, ugrad in sg.items()}

    out_grads.update(wrap(post_name)(_finalize_stage)(stage_grads))

  return out_grads


def _bind_layer_params(
    layer_params: nnx.State,
    layer_pregathered: Any,
    mesh: Mesh | None,
    fsdp_size: int,
    config: Any = None,
) -> nnx.State:
  """Binds pre-gathered layer weights to `layer_params` via `_bind_pregathered`."""
  if layer_pregathered is None or mesh is None or mesh_utils.data_parallel_size(mesh) <= 1:
    return layer_params

  def _bind_node(node: Any, preg_node: Any) -> Any:
    if preg_node is None:
      return node
    logical_axes = _extract_logical_axes(node) if _is_nnx_variable(node) else None

    def _bind_leaf(leaf: Any, preg_leaf: Any) -> Any:
      if preg_leaf is None:
        return leaf
      sharding = shard_leaf_on_fsdp(mesh, leaf, fsdp_size, logical_axes=logical_axes)
      if config is not None and _should_defer_bwd_reduce_scatter(leaf, sharding, config, logical_axes=logical_axes):
        bwd_sharding = NamedSharding(mesh, _remove_fsdp_from_spec(sharding.spec))
      else:
        bwd_sharding = sharding
      return _bind_pregathered(bwd_sharding, leaf.dtype, leaf, preg_leaf)

    return jax.tree.map(_bind_leaf, node, preg_node, is_leaf=lambda x: x is None)

  return jax.tree.map(
      _bind_node,
      layer_params,
      layer_pregathered,
      is_leaf=lambda x: _is_nnx_variable(x) or x is None,
  )


def _microbatch_shards_in_stage(graphdef: PipelineGraphDef, rows: int) -> bool:
  """Whether a microbatch of `rows` rows is split over the stage's devices.

  It is when MaxText's logical axis rules apply (:func:`_logical_axis_context`):
  the stage has several data-parallel devices (``fsdp`` x ``expert``) and
  `rows` is a multiple of their number. Otherwise every device of the stage
  runs the whole microbatch.
  """
  dp_size = graphdef.data_parallel_size
  return bool(graphdef.config.logical_axis_rules) and dp_size > 1 and rows > 0 and rows % dp_size == 0


def _logical_axis_context(
    graphdef: PipelineGraphDef,
    tokens: Any,
    wrap: Wrapper,
) -> contextlib.AbstractContextManager[None]:
  """Enters MaxText's logical axis rules when the microbatch shards in-stage.

  MaxText's DeepSeek layers, Splash Attention and RoutedMoE read the active
  Flax ``logical_axis_rules`` context to map logical batch axes
  (``activation_batch``, ``activation_batch_attn``, ``activation_batch_moe``)
  onto the stage mesh's ``fsdp`` / ``expert`` axes, and the ``exp`` weight axis
  onto ``expert``. Without this context ``get_logical_axis_rules()`` returns
  ``()``, causing every activation and every ``shard_map`` inside
  ``AttentionOp`` and ``RoutedMoE`` to replicate the entire microbatch across
  all devices in the stage.

  Args:
    graphdef: Static structure produced by :meth:`PipelinedTransformer.split`.
    tokens: The microbatch ``inputs`` tensor.
    wrap: Stage wrapping strategy.

  Returns:
    A context manager that activates ``config.logical_axis_rules`` when
    pipelined across multiple data-parallel devices per stage and the
    microbatch batch dimension is divisible by their number.
  """
  batch_dim = tokens.shape[0] if getattr(tokens, "ndim", 0) > 0 else 0
  if wrap is UNWRAPPED or not _microbatch_shards_in_stage(graphdef, batch_dim):
    return contextlib.nullcontext()
  return flax_spmd.logical_axis_rules(graphdef.config.logical_axis_rules)


def _resolve_remat_policy(config: Any, policy_name: str) -> tuple[bool, Any]:
  """Resolves the MPMD model's remat policy into ``(enabled, jax_policy)``.

  The policy is :attr:`PipelineGraphDef.remat_policy` (``--remat_policy``, or
  ``none`` without ``--remat``), not MaxText's ``remat_policy``, which
  ``config.build_maxtext_config`` must keep at a policy MaxText accepts with
  pipeline parallelism. MaxText's own resolver (``NNXDecoder.get_remat_policy``)
  turns it into the policy, so each layer saves exactly the named
  intermediates MaxText's decoder saves under that name.

  Args:
    config: The MaxText config the model was built from.
    policy_name: One of ``hparams.REMAT_POLICIES``, or ``none``.

  Returns:
    ``(use_remat, checkpoint_policy)`` where ``checkpoint_policy`` is passed to
    :func:`jax.checkpoint`.
  """
  if policy_name == "none":
    return False, None
  return True, _NNXDecoderLayerFactory(config, None, layer_index=-1, remat_policy_override=policy_name).remat_policy()


def _ensure_jax_rest(rest_tree: Any) -> Any:
  """Converts abstract numpy array placeholders in `rest_tree` to JAX arrays."""

  def _to_jax(leaf):
    if isinstance(leaf, np.ndarray):
      return jnp.zeros(leaf.shape, dtype=leaf.dtype)
    return leaf

  return jax.tree.map(_to_jax, rest_tree)


def forward(
    graphdef: PipelineGraphDef,
    params: dict[str, nnx.State],
    batch: dict[str, Any],
    *,
    wrap: Wrapper = NAMED_COMPUTATION,
    pregathered_params: dict[str, Any] | None = None,
) -> Any:
  """Runs the forward pass, returning logits.

  Args:
    graphdef: Static structure produced by :meth:`PipelinedTransformer.split`.
    params: Per-unit parameter state, keyed as in ``split()``.
    batch: Must contain ``inputs``; may contain ``inputs_position`` and
      ``inputs_segmentation``.
    wrap: Stage wrapping strategy. :data:`NAMED_COMPUTATION` (the default)
      produces a pipelined MPMD program; :data:`UNWRAPPED` produces the
      equivalent single-program model.
    pregathered_params: Optional pre-cast, FSDP-gathered layer parameter trees
      produced by :func:`pregather_params`.

  Returns:
    Logits of shape ``[batch, length, vocab_size]``.
  """
  tokens = batch["inputs"]
  positions = batch.get("inputs_position")
  segmentation = batch.get("inputs_segmentation")
  use_remat, _ = _resolve_remat_policy(graphdef.config, graphdef.remat_policy)
  # DeepSeek's manifold-constrained hyper-connections (mHC): the layers carry
  # `mhc_expansion_rate` residual streams, which MaxText's decoder expands from
  # the embedding and sums back before the final norm (`NNXDecoder`); so do the
  # embed and unembed computations here.
  expand_streams, reduce_streams = mhc.get_functions(graphdef.config.mhc_expansion_rate)
  uses_mhc = graphdef.config.mhc_expansion_rate > 1

  with _logical_axis_context(graphdef, tokens, wrap):

    embed_pregathered = pregathered_params.get(mesh_utils.EMBED_NAME) if pregathered_params is not None else None

    @wrap(mesh_utils.EMBED_NAME)
    def apply_embed(embed_params, embed_preg, token_ids):
      bound_params = _bind_layer_params(embed_params, embed_preg, graphdef.mesh, graphdef.fsdp_size)

      def _embed_body(p, tok):
        if wrap is not UNWRAPPED and embed_preg is None:
          p = _cast_sharded_layer_params(p, graphdef.mesh, graphdef.fsdp_size, graphdef.config)
        with _logical_axis_context(graphdef, tok, wrap):
          module = nnx.merge(graphdef.embed_def, p, _ensure_jax_rest(graphdef.embed_rest))
          y = module(tok, model_mode=MODEL_MODE_TRAIN)
          # [batch, length, emb] -> [batch, length, streams, emb]
          return expand_streams(y) if uses_mhc else y

      # Always checkpoint iota embedding in MPMD so JAX AD never stashes
      # the `[micro_batch, seq_len, vocab_size]` one-hot tensor in HBM.
      if use_remat or (wrap is not UNWRAPPED and graphdef.config.use_iota_embed):
        return jax.checkpoint(_embed_body)(bound_params, token_ids)
      return _embed_body(bound_params, token_ids)

    hidden = apply_embed(params[mesh_utils.EMBED_NAME], embed_pregathered, tokens)

    def _run_layer(p, layer_preg, h, seg, pos, layer_def, layer_rest):
      if wrap is not UNWRAPPED and layer_preg is None:
        p = _cast_sharded_layer_params(p, graphdef.mesh, graphdef.fsdp_size, graphdef.config)
      with _logical_axis_context(graphdef, h, wrap):
        module = nnx.merge(layer_def, p, _ensure_jax_rest(layer_rest))
        if graphdef.config.decoder_block == DecoderBlockType.SIMPLE_MLP:
          outputs = module(h, positions=pos, segmentation=seg, deterministic=True, model_mode=MODEL_MODE_TRAIN)
        else:
          outputs = module(
              h,
              decoder_segment_ids=seg,
              decoder_positions=pos,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
          )
        # Layers may return `(hidden_states, kv_cache)`.
        return outputs[0] if isinstance(outputs, tuple) else outputs

    for layer_index in range(graphdef.num_layers):
      name = mesh_utils.layer_name(layer_index)
      layer_pregathered = pregathered_params.get(name) if pregathered_params is not None else None
      # Resolved for each layer: a policy object shared by every layer's
      # jax.checkpoint changes how jax numbers the saved residuals in the
      # StableHLO (not the partitioned program).
      layer_use_remat, layer_remat_policy = _resolve_remat_policy(graphdef.config, graphdef.remat_policy)

      # `layer_def`/`layer_rest` are bound as default arguments rather than
      # captured, so each closure keeps its own layer instead of the last one.
      def apply_layer(
          layer_params,
          layer_preg,
          hidden_states,
          segment_ids,
          token_positions,
          *,
          layer_def=graphdef.layer_defs[layer_index],
          layer_rest=graphdef.layer_rests[layer_index],
          use_layer_ckpt=layer_use_remat,
          layer_ckpt_policy=layer_remat_policy,
      ):
        bound_params = _bind_layer_params(
            layer_params,
            layer_preg,
            graphdef.mesh,
            graphdef.fsdp_size,
            graphdef.config,
        )

        def _layer_body(p, h, seg, pos):
          return _run_layer(p, layer_preg, h, seg, pos, layer_def, layer_rest)

        if use_layer_ckpt:
          return jax.checkpoint(_layer_body, policy=layer_ckpt_policy)(
              bound_params, hidden_states, segment_ids, token_positions
          )
        return _layer_body(bound_params, hidden_states, segment_ids, token_positions)

      hidden = wrap(name)(apply_layer)(params[name], layer_pregathered, hidden, segmentation, positions)

    unembed_pregathered = pregathered_params.get(mesh_utils.UNEMBED_NAME) if pregathered_params is not None else None

    @wrap(mesh_utils.UNEMBED_NAME)
    def apply_unembed(unembed_params, unembed_preg, hidden_states):
      bound_params = _bind_layer_params(unembed_params, unembed_preg, graphdef.mesh, graphdef.fsdp_size)
      if wrap is not UNWRAPPED and unembed_preg is None:
        bound_params = _cast_sharded_layer_params(bound_params, graphdef.mesh, graphdef.fsdp_size, graphdef.config)
      with _logical_axis_context(graphdef, hidden_states, wrap):
        if uses_mhc:
          # [batch, length, streams, emb] -> [batch, length, emb]
          hidden_states = reduce_streams(hidden_states)
        module = nnx.merge(graphdef.unembed_def, bound_params, graphdef.unembed_rest)
        return module(hidden_states)

    return apply_unembed(params[mesh_utils.UNEMBED_NAME], unembed_pregathered, hidden)
