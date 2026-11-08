# Copyright 2023–2026 Google LLC
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

"""The jitted programs of the layerwise gates, shared by the harness (concrete inputs) and the AOT checker.

Every program runs the production modules unchanged. Intermediates for per-op localization are recorded by
`capture_calls`, which wraps the named module methods so that each returns its real output and records it; the
gated output is therefore always the production forward, never a re-implementation.
"""

# pylint: disable=too-many-positional-arguments,missing-function-docstring

import contextlib
import dataclasses
import functools
import hashlib
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core.spmd import logical_axis_rules
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import deepseek41, deepseek41_vision
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import maxtext_utils
from tests.utils.deepseek_v41_layerwise import lw_model, lw_routing

from maxtext.layers import (  # isort: skip
    attention_deepseek41,
    embeddings,
    encoders,
    engram_deepseek41,
    moe,
    nnx_decoders,
    nnx_scan,
    normalizations,
)

# Module methods whose outputs `capture_calls` records for per-op localization, in forward order.
DECODER_CAPTURE_POINTS: tuple[tuple[type, str], ...] = (
    (engram_deepseek41.DeepSeek41Engram, "__call__"),
    (deepseek41.DeepSeek41DecoderLayer, "hc_mixes"),
    (normalizations.RMSNorm, "__call__"),
    (attention_deepseek41.DeepSeek41Compressor, "__call__"),
    (attention_deepseek41.DeepSeek41Indexer, "__call__"),
    (attention_deepseek41.DeepSeek41Attention, "__call__"),
    (moe.GateLogit, "__call__"),
    (moe.RoutedMoE, "__call__"),
    (moe.RoutedAndSharedMoE, "__call__"),
)
DECODER_STATIC_CAPTURE_POINTS: tuple[str, ...] = ("hc_pre", "hc_post")
VISION_CAPTURE_POINTS: tuple[tuple[type, str], ...] = (
    (deepseek41_vision.DeepSeek41VisionEncoder, "embed_patches"),
    (deepseek41_vision.DeepSeek41VisionBlock, "__call__"),
    (deepseek41_vision.DeepSeek41VisionEncoder, "__call__"),
    (deepseek41_vision.DeepSeek41VisionProjector, "align"),
    (deepseek41_vision.DeepSeek41VisionProjector, "__call__"),
)


@contextlib.contextmanager
def capture_calls(
    root: Any, points: Sequence[tuple[type, str]], static_points: Sequence[str]
) -> Iterator[dict[str, Any]]:
  """Records the outputs of `points` (class, method) called on modules inside `root`.

  Keys are "<module path in root>:<method>#<call index>" for instance methods and
  "DeepSeek41DecoderLayer.<name>#<call index>" for the static `hc_pre` / `hc_post`; the call index counts calls of
  the same module and method, so every recorded output keeps its own key.
  """
  paths = {
      id(node): "/".join(str(p) for p in path) for path, node in nnx.iter_graph(root) if isinstance(node, nnx.Module)
  }
  store: dict[str, Any] = {}
  calls: dict[str, int] = {}

  def _put(prefix: str, value: Any) -> None:
    index = calls[prefix] if prefix in calls else 0
    calls[prefix] = index + 1
    store[f"{prefix}#{index}"] = value

  def _wrap(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
      out = original(self, *args, **kwargs)
      if id(self) in paths:
        _put(f"{paths[id(self)]}:{method_name}", out)
      return out

    return wrapped

  def _wrap_static(name: str, original: Callable[..., Any]) -> Any:
    def wrapped(*args, **kwargs):
      out = original(*args, **kwargs)
      _put(f"DeepSeek41DecoderLayer.{name}", out)
      return out

    return staticmethod(wrapped)

  with contextlib.ExitStack() as stack:
    for cls, method_name in points:
      stack.enter_context(mock.patch.object(cls, method_name, _wrap(method_name, getattr(cls, method_name))))
    for name in static_points:
      original = getattr(deepseek41.DeepSeek41DecoderLayer, name)
      stack.enter_context(mock.patch.object(deepseek41.DeepSeek41DecoderLayer, name, _wrap_static(name, original)))
    yield store


def layer_build_fn(config: Any, mesh: Mesh, layer_id: int) -> Callable[[], deepseek41.DeepSeek41DecoderLayer]:
  """Constructor of real layer `layer_id` (the spec `build_layer_specs` gives it in the full model)."""
  spec = deepseek41.build_layer_specs(config)[layer_id]
  return lambda: deepseek41.DeepSeek41DecoderLayer(
      config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, spec=spec, quant=None, rngs=nnx.Rngs(params=layer_id)
  )


class L1Modules(nnx.Module):
  """The vision encoder and token embedder exactly as `models.Transformer` builds them (models.py:131-143)."""

  def __init__(self, config: Any, mesh: Mesh, *, rngs: nnx.Rngs):
    self.token_embedder = embeddings.Embed(
        mesh=mesh,
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        dtype=config.dtype,
        attend_dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=config,
        rngs=rngs,
    )
    self.vision_encoder = encoders.VisionEncoder(config=config, mesh=mesh, rngs=rngs)


def l1_build_fn(config: Any, mesh: Mesh) -> Callable[[], L1Modules]:
  return lambda: L1Modules(config, mesh, rngs=nnx.Rngs(params=0))


def layer_reads(config: Any, layer_id: int) -> tuple[str, ...]:
  """Shared-state keys layer `layer_id` reads from earlier layers (the canonical `_group_reads_and_writes`)."""
  specs = deepseek41.build_layer_specs(config)
  reads, _ = deepseek41._group_reads_and_writes(specs, (layer_id,))  # pylint: disable=protected-access
  return tuple(key for key in deepseek41.SHARED_STATE_KEYS if key in reads)


def layer_writes(config: Any, layer_id: int) -> tuple[str, ...]:
  """Shared-state keys layer `layer_id` writes for later layers (the canonical `_group_reads_and_writes`)."""
  specs = deepseek41.build_layer_specs(config)
  _, writes = deepseek41._group_reads_and_writes(specs, (layer_id,))  # pylint: disable=protected-access
  return tuple(key for key in deepseek41.SHARED_STATE_KEYS if key in writes)


def abstract_carry(config: Any, layer_id: int, batch: int, seq: int, sharding: NamedSharding) -> dict[str, Any]:
  """ShapeDtypeStructs of the carry layer `layer_id` consumes (h, pre_mix and the shared state it reads)."""
  spec = deepseek41.build_layer_specs(config)[layer_id]
  hc, dim = config.mhc_expansion_rate, config.emb_dim
  carry = {
      "h": jax.ShapeDtypeStruct((batch, seq, hc, dim), jnp.dtype(config.dtype), sharding=sharding),
      "pre_mix": jax.ShapeDtypeStruct((batch, seq, hc), jnp.float32, sharding=sharding),
  }
  reads = layer_reads(config, layer_id)
  if reads:
    shapes = jax.eval_shape(
        lambda: deepseek41.add_placeholders(
            {"h": jnp.zeros((batch, seq, hc, dim), config.dtype)}, reads, spec.compress_ratio, config
        )
    )
    for key in reads:
      carry[key] = jax.ShapeDtypeStruct(shapes[key].shape, shapes[key].dtype, sharding=sharding)
  return carry


def _pop_naturals(layer: Any) -> dict[str, jax.Array]:
  """The natural top-k (and on candidate sources the raw indexer scores) sowed during the last call of `layer`."""
  inter = nnx.to_pure_dict(nnx.pop(layer, nnx.Intermediate))
  out = {"moe_topk": inter["mlp"]["MoeBlock_0"][lw_routing.NATURAL_MOE_TOPK][0]}
  if layer.spec.is_index_source:
    out["indexer_mask"] = inter["self_attention"]["indexer"][lw_routing.NATURAL_INDEXER_MASK][0]
  if layer.spec.is_candidate_source:
    out["indexer_raw_scores"] = inter["self_attention"]["indexer"][lw_routing.INDEXER_RAW_SCORES][0]
  return out


def as_layer_replay(layer: Any, replay: Mapping[str, jax.Array]) -> lw_routing.LayerReplay:
  """The `LayerReplay` of `layer` from a replay dict; the indexer entry is read exactly for index sources."""
  indexer_mask = replay["indexer_mask"] if layer.spec.is_index_source else None
  return lw_routing.LayerReplay(moe_topk=replay["moe_topk"], indexer_mask=indexer_mask)


@dataclasses.dataclass(frozen=True)
class GroupProgram:
  """Static description of a teacher-forced group of consecutive real layers."""

  config: Any
  mesh: Mesh
  layer_ids: tuple[int, ...]
  graphdefs: tuple[Any, ...]
  engram_slices: tuple[lw_model.EngramSlice | None, ...]

  def merge_layer(self, i: int, weights: nnx.State, rest: nnx.State) -> deepseek41.DeepSeek41DecoderLayer:
    layer = lw_model.merge_fresh(self.graphdefs[i], weights, rest)
    if self.engram_slices[i] is not None:
      lw_model.install_sliced_hasher(layer, self.engram_slices[i])
    return layer


def group_forward(
    program: GroupProgram,
    weights: Sequence[nnx.State],
    rest: Sequence[nnx.State],
    carry: Mapping[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    image_mask: jax.Array,
    replays: Sequence[Mapping[str, jax.Array]],
    *,
    decoder_segment_ids: jax.Array | None = None,
    capture: bool,
) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]], list[dict[str, Any]]]:
  """Runs the group's layers in order from the teacher-forced `carry`, with replayed routing.

  Must be traced inside `lw_routing.replayed_routing()`. `replays[i]` holds "moe_topk" and, exactly when layer i is
  an index source, "indexer_mask".

  Returns:
    Per layer: the output {"h", "pre_mix", and the shared-state keys the layer writes}, the sowed natural top-k, and
    (when `capture`) the captured intermediates.
  """
  outs, naturals, captured = [], [], []
  state = dict(carry)
  for i, _ in enumerate(program.layer_ids):
    layer = program.merge_layer(i, weights[i], rest[i])
    lw_routing.attach_replay(layer, as_layer_replay(layer, replays[i]))
    with contextlib.ExitStack() as stack:
      store = (
          stack.enter_context(capture_calls(layer, DECODER_CAPTURE_POINTS, DECODER_STATIC_CAPTURE_POINTS))
          if capture
          else None
      )
      state = layer(state, positions, tokens, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask)
    written = layer_writes(program.config, program.layer_ids[i])
    outs.append({"h": state["h"], "pre_mix": state["pre_mix"], **{key: state[key] for key in written}})
    naturals.append(_pop_naturals(layer))
    captured.append(store if capture else {})
  return outs, naturals, captured


def objective_term(value: jax.Array, cotangent: jax.Array) -> jax.Array:
  """<value, cotangent> in the wider of value's dtype and float32 (float32 for bf16 / fp32 runs; float64 in the fp64
  parity tests, where a float32 objective would round the cotangent to ~1e-7 relative)."""
  work = jnp.promote_types(value.dtype, jnp.float32)
  return jnp.sum(value.astype(work) * cotangent.astype(work))


def layer_vjp(
    program: GroupProgram,
    weights: nnx.State,
    rest: nnx.State,
    carry: Mapping[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    image_mask: jax.Array,
    replay: Mapping[str, jax.Array],
    cotangents: Mapping[str, jax.Array],
    *,
    decoder_segment_ids: jax.Array | None = None,
) -> tuple[nnx.State, jax.Array, jax.Array]:
  """Gradients of the L4 objective of the program's single layer w.r.t. (weights, h_in, pre_mix_in).

  Objective: <h_out, cotangents["h"]> + <pre_mix_out, cotangents["pre_mix"]> (the only path to the pre rows of the
  hc_ffn mix), plus <indexer raw scores, cotangents["indexer_scores"]> exactly when the layer is an index source (the
  indexer's top-k is discrete, so its weights get a gradient only through this term).
  Routing is replayed; must be traced inside `lw_routing.replayed_routing()`.
  """
  if len(program.layer_ids) != 1:
    raise ValueError(f"layer_vjp takes a single-layer program, got layers {program.layer_ids}")
  shared = {k: v for k, v in carry.items() if k not in ("h", "pre_mix")}

  def loss(w: nnx.State, h: jax.Array, pre_mix: jax.Array) -> jax.Array:
    layer = program.merge_layer(0, w, rest)
    lw_routing.attach_replay(layer, as_layer_replay(layer, replay))
    out = layer(
        {**shared, "h": h, "pre_mix": pre_mix},
        positions,
        tokens,
        decoder_segment_ids=decoder_segment_ids,
        image_mask=image_mask,
    )
    inter = nnx.to_pure_dict(nnx.pop(layer, nnx.Intermediate))
    h_out = out["h"]
    pre_mix_out = out["pre_mix"]
    if decoder_segment_ids is not None:
      valid_h = (decoder_segment_ids != 0)[:, :, None, None] if h_out.ndim == 4 else (decoder_segment_ids != 0)[..., None]
      valid_pre = (decoder_segment_ids != 0)[..., None]
      h_out = jnp.where(valid_h, h_out, 0.0)
      pre_mix_out = jnp.where(valid_pre, pre_mix_out, 0.0)
    total = objective_term(h_out, cotangents["h"]) + objective_term(pre_mix_out, cotangents["pre_mix"])
    if layer.spec.is_index_source:
      raw = inter["self_attention"]["indexer"][lw_routing.INDEXER_RAW_SCORES][0]
      if decoder_segment_ids is not None:
        raw_mask = (
            (decoder_segment_ids != 0)[:, None, :, None] if raw.ndim == 4 else (decoder_segment_ids != 0)[..., None]
        )
        raw = jnp.where(raw_mask, raw, 0.0)
      total = total + objective_term(raw, cotangents["indexer_scores"])
    return total

  return jax.grad(loss, argnums=(0, 1, 2))(weights, carry["h"], carry["pre_mix"])


class DecoderApplier:
  """Gives the production layer appliers of `NNXDecoder` the only attribute they read from the decoder: `config`.

  `_apply_layers_sequentially` (scanned stacks) and `_apply_layer_with_remat` (unscanned remat) are borrowed
  unchanged, so the L3 programs run exactly the scan / remat code of the full model (nnx_decoders.py:1099-1311).
  """

  get_remat_policy = nnx_decoders.NNXDecoder.get_remat_policy
  minimal_policy = nnx_decoders.NNXDecoder.minimal_policy
  _apply_layers_sequentially = nnx_decoders.NNXDecoder._apply_layers_sequentially  # pylint: disable=protected-access
  _apply_layer_with_remat = nnx_decoders.NNXDecoder._apply_layer_with_remat  # pylint: disable=protected-access

  def __init__(self, config: Any, mesh: Any = None):
    self.config = config
    self.mesh = mesh


def check_repeat_signatures(config: Any, group_layer_ids: Sequence[Sequence[int]]) -> None:
  """Raises unless every repeat has the same per-layer `LayerSpec.signature()` (what a scanned stack requires)."""
  specs = deepseek41.build_layer_specs(config)
  signatures = {tuple(specs[i].signature() for i in group) for group in group_layer_ids}
  if len(signatures) != 1:
    raise ValueError(f"Scanned repeats {group_layer_ids} have different layer signatures {signatures}")


def scanned_block_build_fn(config: Any, mesh: Mesh, group_layer_ids: Sequence[Sequence[int]]) -> Callable[[], Any]:
  """Constructor of one scanned `DeepSeek41Block` stack whose repeats are `group_layer_ids` (identical signatures).

  Uses the same `nnx_scan.create_scanned_layers` call as `NNXDecoder._init_deepseek41_layers` (nnx_decoders.py:645-659).
  """
  check_repeat_signatures(config, group_layer_ids)
  specs = deepseek41.build_layer_specs(config)
  per_layer_remat = getattr(config, "ds41_per_layer_remat", True) and config.remat_policy != "none"
  applier = DecoderApplier(config, mesh)
  first = tuple(specs[i] for i in group_layer_ids[0])
  block_params = inspect.signature(deepseek41.DeepSeek41Block.__init__).parameters

  def build() -> Any:
    def _make_block(layer_rngs):
      block_kwargs: dict[str, Any] = {
          "config": config,
          "mesh": mesh,
          "model_mode": MODEL_MODE_TRAIN,
          "specs": first,
          "quant": None,
      }
      remat_fn = applier.get_remat_policy() if per_layer_remat else None
      if "remat_policy_fn" in block_params:
        block_kwargs["remat_policy_fn"] = remat_fn
      elif "remat_policy" in block_params:
        block_kwargs["remat_policy"] = remat_fn

      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(config)
      if "prevent_cse_in_remat" in block_params:
        block_kwargs["prevent_cse_in_remat"] = prevent_cse
      elif "prevent_cse" in block_params:
        block_kwargs["prevent_cse"] = prevent_cse

      if "apply_internal_remat" in block_params:
        block_kwargs["apply_internal_remat"] = per_layer_remat

      return deepseek41.DeepSeek41Block(
          **block_kwargs,
          rngs=layer_rngs,
      )

    return nnx_scan.create_scanned_layers(
        _make_block,
        length=len(group_layer_ids),
        param_scan_axis=config.param_scan_axis,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(params=0),
    )

  return build


def stack_axis(config: Any, *, is_bias: bool, stacked_ndim: int) -> int:
  """Axis of the repeat dimension in a stacked block leaf (the one owner of this rule): `moe.MoEBiasVar` leaves at 0,
  `nnx.Param` leaves at `config.param_scan_axis` when the stacked leaf has that axis, else 0."""
  if is_bias:
    return 0
  return config.param_scan_axis if stacked_ndim > config.param_scan_axis else 0


def stack_group_weights(config: Any, per_repeat: Sequence[nnx.State], abstract: nnx.State) -> nnx.State:
  """Stacks per-repeat block weights (`nnx.State({"layers_<j>": layer weights})`) into the scanned layout.

  As `copy_full_multimodal_model` does: `nnx.Param` leaves stack on `config.param_scan_axis`, `moe.MoEBiasVar` leaves
  on axis 0. The result takes the shardings of the scanned stack's abstract weights.
  """

  def _stack_params(*leaves: jax.Array) -> jax.Array:
    stacked = jnp.stack(leaves, axis=0)
    return jnp.moveaxis(stacked, 0, stack_axis(config, is_bias=False, stacked_ndim=stacked.ndim))

  params = jax.tree.map(_stack_params, *[nnx.filter_state(w, nnx.Param) for w in per_repeat])
  biases = jax.tree.map(
      lambda *leaves: jnp.stack(leaves, axis=0), *[nnx.filter_state(w, moe.MoEBiasVar) for w in per_repeat]
  )
  target = jax.tree.map(lambda x: x, abstract)
  nnx.replace_by_pure_dict(target, nnx.to_pure_dict(nnx.merge_state(params, biases)))
  return jax.tree.map(lambda x, a: jax.device_put(x, a.sharding), target, abstract)


@dataclasses.dataclass(frozen=True)
class ScanGeometry:
  """Real layers of the scanned L3 / L2 program: unscanned prefix, scanned repeats, unscanned suffix."""

  prefix: tuple[int, ...]
  repeats: tuple[tuple[int, ...], ...]
  suffix: tuple[int, ...]

  @property
  def layer_ids(self) -> tuple[int, ...]:
    return self.prefix + tuple(i for group in self.repeats for i in group) + self.suffix

  def validate(self, config: Any) -> None:
    """Raises unless the layers are consecutive, there are >= 2 repeats, and prefix / suffix are non-empty."""
    ids = self.layer_ids
    if ids != tuple(range(ids[0], ids[0] + len(ids))):
      raise ValueError(f"Scan geometry layers {ids} are not consecutive")
    if len(self.repeats) < 2 or not self.prefix or not self.suffix:
      raise ValueError(f"Scan geometry needs prefix + >= 2 scanned repeats + suffix, got {self}")
    check_repeat_signatures(config, self.repeats)


def chain_unscanned(
    program: GroupProgram,
    weights: Sequence[nnx.State],
    rest: Sequence[nnx.State],
    carry: Mapping[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    image_mask: jax.Array,
    *,
    remat: bool,
    decoder_segment_ids: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Runs the program's layers unscanned with natural routing, as `_apply_deepseek41_layers` does (:695-723).

  With `remat`, each layer goes through the production `NNXDecoder._apply_layer_with_remat` with the config's
  remat policy; otherwise layers are called directly.
  """
  applier = DecoderApplier(program.config, getattr(program, "mesh", None))
  state = dict(carry)
  for i, _ in enumerate(program.layer_ids):
    layer = program.merge_layer(i, weights[i], rest[i])
    if remat:
      state = applier._apply_layer_with_remat(  # pylint: disable=protected-access
          layer,
          state,
          applier.get_remat_policy(),
          maxtext_utils.should_prevent_cse_in_remat(program.config),
          decoder_positions=positions,
          decoder_input_tokens=tokens,
          decoder_segment_ids=decoder_segment_ids,
          image_mask=image_mask,
      )
    else:
      state = layer(state, positions, tokens, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask)
  return {"h": state["h"], "pre_mix": state["pre_mix"]}


@dataclasses.dataclass(frozen=True)
class ScannedProgram:
  """The prefix layers, one scanned `DeepSeek41Block` stack over the repeats, and the suffix layers."""

  config: Any
  geometry: ScanGeometry
  prefix: GroupProgram
  stack_graphdef: Any
  suffix: GroupProgram


def chain_scanned(
    program: ScannedProgram,
    weights: Mapping[str, Any],
    rest: Mapping[str, Any],
    carry: Mapping[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    image_mask: jax.Array,
    *,
    decoder_segment_ids: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Prefix layers, then the stack through the production `NNXDecoder._apply_layers_sequentially` (:682-692),
  then the suffix layers; natural routing throughout.

  `weights` / `rest` hold "prefix" and "suffix" (lists per layer) and "stack" (the stacked block state).
  """
  full = dict(carry)
  for i, _ in enumerate(program.prefix.layer_ids):
    layer = program.prefix.merge_layer(i, weights["prefix"][i], rest["prefix"][i])
    full = layer(full, positions, tokens, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask)
  stack = lw_model.merge_fresh(program.stack_graphdef, weights["stack"], rest["stack"])
  mesh = getattr(program.prefix, "mesh", getattr(program.suffix, "mesh", None))
  full, _, _ = DecoderApplier(program.config, mesh)._apply_layers_sequentially(  # pylint: disable=protected-access
      stack,
      full,
      positions,
      tokens,
      decoder_segment_ids,
      length=len(program.geometry.repeats),
      metadata_axis_name="layers",
      skip_block_remat=getattr(stack, "apply_internal_remat", getattr(program.config, "remat_policy", "none") != "none"),
      image_mask=image_mask,
  )
  for i, _ in enumerate(program.suffix.layer_ids):
    layer = program.suffix.merge_layer(i, weights["suffix"][i], rest["suffix"][i])
    full = layer(full, positions, tokens, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask)
  return {"h": full["h"], "pre_mix": full["pre_mix"]}


def l1_forward(
    graphdef: Any, weights: nnx.State, rest: nnx.State, tokens: jax.Array, patches: jax.Array, *, capture: bool
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
  """Embeds tokens, encodes the image(s) and merges the span exactly as the multimodal model does.

  Follows models.py:249-255 (vision encoder + bidirectional mask) and nnx_decoders.py:1558-1605 (token embedding and
  `merge_mm_embeddings`), then `deepseek41.initial_carry` (nnx_decoders.py:674).
  """
  module = lw_model.merge_fresh(graphdef, weights, rest)
  config = module.vision_encoder.config
  with contextlib.ExitStack() as stack:
    store = stack.enter_context(capture_calls(module, VISION_CAPTURE_POINTS, static_points=())) if capture else None
    image_embeddings, _ = module.vision_encoder(input_images=patches, deterministic=True)
  image_mask = mm_processor.get_bidirectional_mask_vision(config, tokens, is_video=False)
  text = module.token_embedder(tokens.astype("int32"), model_mode=MODEL_MODE_TRAIN)
  merged = mm_utils.merge_mm_embeddings(
      text_embeddings=text, multimodal_embeddings=image_embeddings, mask=image_mask, token_masks=None
  )
  carry = deepseek41.initial_carry(merged, config.mhc_expansion_rate)
  outputs = {
      "span": image_embeddings,
      "image_mask": image_mask,
      "merged": merged,
      "h0": carry["h"],
      "pre_mix0": carry["pre_mix"],
  }
  return outputs, (store if capture else {})


def replicated(mesh: Mesh) -> NamedSharding:
  return NamedSharding(mesh, PartitionSpec())


# Program names: the AOT checker records each compiled program under one, and the harness requires that record.
L1_FORWARD_PROGRAM = "L1.forward"
L1_VJP_PROGRAM = "L4.vision"


def group_program_name(layer_ids: Sequence[int]) -> str:
  return f"L2.group[{','.join(map(str, layer_ids))}]"


def layer_vjp_program_name(layer_id: int) -> str:
  return f"L4.layer[{layer_id}]"


def chain_program_name(mode: str) -> str:
  return f"L3.chain.{mode}"


# Jit entry points. The harness and the AOT checker both call `jax.jit` on exactly these functions; each enters the
# replayed routing (where it applies) and the config's logical axis rules while it is traced.


def group_forward_entry(program: GroupProgram, *, capture: bool) -> Callable[..., Any]:
  def entry(weights, rest, carry, positions, tokens, image_mask, replays, decoder_segment_ids=None):
    with lw_routing.replayed_routing(), logical_axis_rules(program.config.logical_axis_rules):
      return group_forward(
          program,
          weights,
          rest,
          carry,
          positions,
          tokens,
          image_mask,
          replays,
          decoder_segment_ids=decoder_segment_ids,
          capture=capture,
      )

  return entry


def layer_vjp_entry(program: GroupProgram) -> Callable[..., Any]:
  def entry(weights, rest, carry, positions, tokens, image_mask, replay, cotangents, decoder_segment_ids=None):
    with lw_routing.replayed_routing(), logical_axis_rules(program.config.logical_axis_rules):
      return layer_vjp(
          program,
          weights,
          rest,
          carry,
          positions,
          tokens,
          image_mask,
          replay,
          cotangents,
          decoder_segment_ids=decoder_segment_ids,
      )

  return entry


def chain_entry(forward: Callable[..., dict[str, jax.Array]], config: Any) -> Callable[..., Any]:
  """Forward outputs and d<h_out, cotangent_h>/d(weights, h_in, pre_mix_in) of a natural-routing chain forward."""

  def entry(weights, rest, carry, positions, tokens, image_mask, cotangent_h):
    shared = {k: v for k, v in carry.items() if k not in ("h", "pre_mix")}

    def loss(w, h, pre_mix):
      out = forward(w, rest, {**shared, "h": h, "pre_mix": pre_mix}, positions, tokens, image_mask)
      return objective_term(out["h"], cotangent_h), out

    with logical_axis_rules(config.logical_axis_rules):
      (_, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(weights, carry["h"], carry["pre_mix"])
    return out, grads

  return entry


def replayed_chain_entry(program: GroupProgram) -> Callable[..., Any]:
  """`chain_entry` (forward outputs and d<h_out, cotangent_h>/d(weights, h_in, pre_mix_in)) of the group with
  replayed routing (`replays` as in `group_forward`); the L3 EP comparison."""

  def entry(weights, rest, carry, positions, tokens, image_mask, replays, cotangent_h):
    def forward(w, r, c, p, t, m):
      outs, _, _ = group_forward(program, w, r, c, p, t, m, replays, capture=False)
      return outs[-1]

    with lw_routing.replayed_routing():
      return chain_entry(forward, program.config)(weights, rest, carry, positions, tokens, image_mask, cotangent_h)

  return entry


def l1_forward_entry(graphdef: Any, config: Any, *, capture: bool) -> Callable[..., Any]:
  def entry(weights, rest, tokens, patches):
    with logical_axis_rules(config.logical_axis_rules):
      return l1_forward(graphdef, weights, rest, tokens, patches, capture=capture)

  return entry


def l1_vjp_entry(graphdef: Any, config: Any) -> Callable[..., Any]:
  """d<merged embeddings, cotangent>/d(weights, patches); needs `freeze_vision_encoder_params=False` for the ViT."""
  if config.freeze_vision_encoder_params:
    raise ValueError("l1_vjp_entry needs freeze_vision_encoder_params=False, else the ViT gets no gradient")

  def entry(weights, rest, tokens, patches, cotangent_merged):
    def loss(w, x):
      outputs, _ = l1_forward(graphdef, w, rest, tokens, x, capture=False)
      return objective_term(outputs["merged"], cotangent_merged)

    with logical_axis_rules(config.logical_axis_rules):
      return jax.grad(loss, argnums=(0, 1))(weights, patches)

  return entry


# sha256 of `inspect.getsource(deepseek41.DeepSeek41DecoderLayer.hc_post)` at eda64008d78a7d043006f74edc48cb575b570db0.
ORIGINAL_HC_POST_SHA256 = "d2f6c830aae894e1473e4eb8170eb3449945fd30480325e5b253ad6a39a2f2d3"


def hc_post_source_sha256() -> str:
  return hashlib.sha256(inspect.getsource(deepseek41.DeepSeek41DecoderLayer.hc_post).encode("utf-8")).hexdigest()


def default_precision_hc_post(x: jax.Array, residual: jax.Array, post: jax.Array, comb: jax.Array) -> jax.Array:
  """`DeepSeek41DecoderLayer.hc_post` (deepseek41.py:261-269) with `precision=DEFAULT` instead of `HIGHEST`.

  The math is identical; only the matmul precision of the residual mix changes, which is the defect class a missing
  `precision=` causes on the TPU MXU (tier-b negative control).
  """
  mixed = jnp.einsum(
      "bsjk,bsjd->bskd",
      comb,
      residual.astype(jnp.float32),
      precision=jax.lax.Precision.DEFAULT,
  )
  return (post[..., None] * x.astype(jnp.float32)[:, :, None, :] + mixed).astype(x.dtype)


@contextlib.contextmanager
def hc_post_default_precision() -> Iterator[None]:
  """Swaps `DeepSeek41DecoderLayer.hc_post` for `default_precision_hc_post` (raises if the original changed)."""
  actual = hc_post_source_sha256()
  if actual != ORIGINAL_HC_POST_SHA256:
    raise AssertionError(f"DeepSeek41DecoderLayer.hc_post changed (sha256 {actual} != {ORIGINAL_HC_POST_SHA256})")
  with mock.patch.object(deepseek41.DeepSeek41DecoderLayer, "hc_post", staticmethod(default_precision_hc_post)):
    yield
