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

"""Converters that translate MaxText trainer weights into vLLM rollout weights."""

import abc
import dataclasses
import gc
import logging
import os
import re
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Union, Any, Dict, Optional, Mapping, Tuple
from flax import traverse_util, nnx
from maxtext.integration.vllm.convert_utils import (
    _align_per_axis,
    _apply_dtype_cast,
    _bulk_align_and_unstack,
    _device_ids,
    _fuse_and_unstack_moe,
    _get_n_shards,
    _jit_unstack,
    _scanned_sharding_from_per_layer,
    _sharding_summary,
)

_MOE_MLP_WEIGHTS = frozenset({"wi_0", "wi_1", "wo", "wi"})


def _malloc_trim() -> None:
  try:
    import ctypes  # pylint: disable=g-import-not-at-top
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:
    pass


# ==========================================
# 1. Operations
# ==========================================
class Operation(abc.ABC):
  """Base class for weight transformation operations."""

  @abc.abstractmethod
  def __call__(self, tensors: List[Any], **kwargs) -> Any:
    pass


class Concatenate(Operation):
  """Concatenates input tensors along a given dimension."""

  def __init__(self, dim: int):
    self.dim = dim

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(*ts):
      return jnp.concatenate(ts, axis=self.dim)

    return _f(*tensors)


class Transpose(Operation):
  """Transposes the input tensor along specified axes."""

  def __init__(self, axes):
    self.axes = axes

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      return jnp.transpose(t, self.axes)

    return _f(tensors[0])


class TransposeUnstack(Operation):
  """Transposes the input tensor and unstacks along axis 0."""

  def __init__(self, axes):
    self.axes = axes

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      return jnp.transpose(t, self.axes)

    return list(jnp.unstack(_f(tensors[0]), axis=0))


class AttentionOut(Operation):
  """Transforms and unstacks attention output projection weights."""

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      # t shape: [heads?, num_layers, head_dim, d_model]
      o = jnp.transpose(t, (1, 3, 0, 2))
      return o.reshape(o.shape[0], o.shape[1], -1)

    return list(jnp.unstack(_f(tensors[0]), axis=0))


class AttentionQKV(Operation):
  """Combines and transforms Q, K, V projections into interleaved QKV format."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _f(q_in, k_in, v_in):
      q = jnp.transpose(q_in, (1, 0, 2, 3))
      k = jnp.transpose(k_in, (1, 0, 2, 3))
      v = jnp.transpose(v_in, (1, 0, 2, 3))

      num_layers, d_model, num_q_heads, head_dim = q.shape
      num_kv_heads = k.shape[2]
      actual_tp = min(tp, num_kv_heads)

      kv_per_tp = num_kv_heads // actual_tp
      q_per_tp = num_q_heads // actual_tp

      q_by_tp = q.reshape(num_layers, d_model, actual_tp, q_per_tp, head_dim)
      k_by_tp = k.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)
      v_by_tp = v.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)

      qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
      qkv_flat = qkv_by_tp.reshape(num_layers, d_model, -1)
      qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))
      return qkv_proj

    res = _f(tensors[0], tensors[1], tensors[2])
    return list(jnp.unstack(res, axis=0))


class MoEExpertDown(Operation):
  """Transposes MoE expert down projection weights across layer axis."""

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t_in):
      # t_in is shape [num_experts, num_layers, d_inner, d_model]
      # transpose (1, 0, 2, 3) makes it [num_layers, num_experts, d_inner, d_model]
      return jnp.transpose(t_in, (1, 0, 2, 3))

    t_transposed = _f(tensors[0])
    return list(jnp.unstack(t_transposed, axis=0))


class MoEFuseGateUp(Operation):
  """Fuses separate gate (wi_0) and up (wi_1) MoE projections with TP chunking."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _fuse_all(wi_0, wi_1):
      wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))
      wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

      def _fuse_single(w0, w1):
        w0 = jnp.transpose(w0, (0, 2, 1))
        w1 = jnp.transpose(w1, (0, 2, 1))
        num_experts, d_inner, d_model = w0.shape
        chunk_size = d_inner // tp
        padded_chunk_size = ((chunk_size + 127) // 128) * 128
        pad_amount = padded_chunk_size - chunk_size
        gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
        if pad_amount > 0:
          gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
        return jnp.transpose(res, (0, 2, 1))

      return jax.vmap(_fuse_single)(wi_0, wi_1)

    fused = _fuse_all(tensors[0], tensors[1])
    return list(jnp.unstack(fused, axis=0))


class MoEFuseGateUpPrefused(Operation):
  """Restructures pre-fused gate/up MoE projections with TP chunking."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _fuse_all(wi):
      wi = jnp.transpose(wi, (1, 0, 2, 3))

      def _fuse_single(w):
        w = jnp.transpose(w, (0, 2, 1))
        num_experts, double_d_inner, d_model = w.shape
        d_inner = double_d_inner // 2
        w0 = w[:, :d_inner, :]
        w1 = w[:, d_inner:, :]
        chunk_size = d_inner // tp
        padded_chunk_size = ((chunk_size + 127) // 128) * 128
        pad_amount = padded_chunk_size - chunk_size
        gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
        if pad_amount > 0:
          gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
        return jnp.transpose(res, (0, 2, 1))

      return jax.vmap(_fuse_single)(wi)

    fused = _fuse_all(tensors[0])
    return list(jnp.unstack(fused, axis=0))


class Identity(Operation):
  """Returns the input tensor unmodified."""

  def __call__(self, tensors, **kwargs):
    return tensors[0]


# ==========================================
# 2. Rule
# ==========================================
class Rule:
  """Unified rule format for converting weights."""

  def __init__(self, source_patterns: Union[str, List[str]], target_pattern: str, operations: List[Operation] = None):
    if isinstance(source_patterns, str):
      self.source_patterns = [source_patterns]
    else:
      self.source_patterns = source_patterns
    self.target_pattern = target_pattern
    self.operations = operations or []


# ==========================================
# 3. Engine
# ==========================================
class WeightConverter:
  """Converts MaxText trainer weights into the rollout engine's layout.

  Two modes, selected by `rules`:

  1. `rules=None` -- **direct MaxText-to-MaxText**. vLLM is running the MaxText
     model itself (`MaxTextForCausalLM`), so no name translation is needed and
     the differences are purely structural: the trainer scans decoder layers
     while the rollout unrolls them, and the rollout pre-fuses MoE `wi_0`/`wi_1`
     into a padded `wi`. This is the replacement for Tunix's
     `transfer_state_directly`. Requires `config`; `prefuse_moe_weights` is
     assumed true on the rollout side.

  2. `rules=[Rule(...), ...]` -- **standalone torchax conversion**. vLLM is
     running its own HuggingFace-shaped model, so weights are renamed and
     restructured (QKV fusion with GQA interleave, MoE gate+up fusion into
     `w13_weight`, layer-norm and lm-head transposes) per the rule table in
     `MODEL_TO_CONVERSION_RULES`.

  An empty rule list is rejected: it would silently convert nothing and leave
  the rollout on its randomly-initialized dummy weights. Pass `None` to mean
  "direct".

  `debug` (direct mode only) logs the device placement of every operand and
  barriers between conversion steps, so a device or sharding failure names the
  parameter that caused it. It serializes the sync; keep it off in steady
  state.
  """

  def __init__(
      self,
      rules: Optional[List[Rule]] = None,
      tp: int = 1,
      num_kv_heads: Optional[int] = None,
      head_dim: Optional[int] = None,
      config: Any = None,
      trainer_config: Any = None,
      rollout_backend: str = "maxtext",
      moe_fused_layout: Optional[str] = None,
      allow_unused_source_keys: Tuple[str, ...] = (),
      debug: bool = False,
      prefuse_moe_weights: Optional[bool] = None,
      target_dtype: Optional[Any] = None,
  ):
    config = trainer_config if config is None else config
    if rules is not None and not rules:
      raise ValueError(
          "WeightConverter(rules=[]) would convert nothing and leave the "
          "rollout on dummy weights. Pass rules=None for the direct "
          "MaxText-to-MaxText path, or a non-empty rule list."
      )
    self.rules = rules
    self.tp = tp
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    # Read by the rollout engine to decide whether to trace the reshard
    # step that runs after conversion.
    self.debug = debug
    self.rollout_backend = rollout_backend

    self._direct: Optional["MaxTextToMaxTextConverter"] = None
    if rollout_backend == "maxtext" and rules is None:
      if config is None:
        raise ValueError(
            "WeightConverter(rules=None) needs `config` to derive the "
            "scanned-to-unrolled layer plan (num_decoder_layers, "
            "inhomogeneous_layer_cycle_interval, param_scan_axis)."
        )
      self._direct = MaxTextToMaxTextConverter(
          config=config,
          moe_fused_layout=(moe_fused_layout or MoEFusedLayout.PER_SHARD_INTERLEAVE),
          allow_unused_source_keys=allow_unused_source_keys,
          debug=debug,
          prefuse_moe_weights=prefuse_moe_weights,
          target_dtype=target_dtype,
      )
      logging.info("WeightConverter: direct MaxText-to-MaxText mode (debug=%s).", debug)
    else:
      if self.rules is None and config is not None:
        model_name = getattr(config, "model_name", "")
        if model_name in MODEL_TO_CONVERSION_RULES and MODEL_TO_CONVERSION_RULES[model_name] is not None:
          self.rules = MODEL_TO_CONVERSION_RULES[model_name]
      logging.info(
          "WeightConverter: torchax rule mode (tp=%d, %d rules).",
          self.tp,
          len(self.rules) if self.rules else 0,
      )

  def convert(self, src_pytree: Any, target_state: Any = None) -> Any:
    """Converts source weights pytree into target format using rules or direct converter."""
    if self.rollout_backend == "maxtext" and self.rules is None:
      return self._direct.convert(src_pytree, target_state=target_state)

    if self.rules is None:
      raise ValueError("WeightConverter in torchax mode requires conversion rules.")

    flat_src = traverse_util.flatten_dict(_to_pure_dict(src_pytree), sep=".")
    gc.collect()

    # HuggingFace-shaped target: rename and restructure per the rule table.
    result = {}
    unfired = []
    for rule in self.rules:
      tensors = [flat_src.pop(src_pat) for src_pat in rule.source_patterns if src_pat in flat_src]
      if not tensors:
        # A rule can legitimately not fire (e.g. the `wi` rule when the
        # trainer stores split `wi_0`/`wi_1`), but a table where *many*
        # rules miss means the source layout has drifted.
        unfired.append(rule.target_pattern)
        continue

      out = tensors
      for op in rule.operations:
        out = op(out, tp=self.tp)
        if not isinstance(out, list) and op != rule.operations[-1]:
          out = [out]

      if isinstance(out, list) and len(out) > 1 and "{}" in rule.target_pattern:
        for i, tensor in enumerate(out):
          result[rule.target_pattern.format(i)] = tensor
      elif isinstance(out, list) and len(out) == 1:
        result[rule.target_pattern] = out[0]
      else:
        result[rule.target_pattern] = out

      del out, tensors
      gc.collect()

    if not result:
      raise ValueError(
          "No conversion rule matched the trainer state; the rollout would "
          f"keep its dummy weights. Rule targets: {unfired}"
      )
    if unfired:
      logging.info("Conversion rules that did not fire: %s", unfired)

    return _rekey_to_target(result, target_state)


# ==========================================
# 4. Registries and Builders
# ==========================================
# Rule tables for the torchax (HuggingFace-target) mode, replacing the legacy
# transfer_state_with_mappings(). `None` means the model has no HF-target rules
# and is only supported on the direct MaxText-to-MaxText path.
MODEL_TO_CONVERSION_RULES = {
    "qwen3": None,
    "qwen35_moe": None,
    "qwen3_moe": [
        Rule(["base.token_embedder.embedding"], "vllm_model.model.embed_tokens.weight"),
        Rule(["base.decoder.decoder_norm.scale"], "vllm_model.model.norm.weight"),
        Rule(["base.decoder.logits_dense.kernel"], "vllm_model.lm_head.weight", [Transpose(axes=(1, 0))]),
        Rule(
            ["base.decoder.layers.pre_self_attention_layer_norm.scale"],
            "vllm_model.model.layers.{}.input_layernorm.weight",
            [TransposeUnstack(axes=(1, 0))],
        ),
        Rule(
            ["base.decoder.layers.post_self_attention_layer_norm.scale"],
            "vllm_model.model.layers.{}.post_attention_layernorm.weight",
            [TransposeUnstack(axes=(1, 0))],
        ),
        Rule(
            ["base.decoder.layers.self_attention.out.kernel"],
            "vllm_model.model.layers.{}.self_attn.o_proj.weight",
            [AttentionOut()],
        ),
        Rule(
            [
                "base.decoder.layers.self_attention.query.kernel",
                "base.decoder.layers.self_attention.key.kernel",
                "base.decoder.layers.self_attention.value.kernel",
            ],
            "vllm_model.model.layers.{}.self_attn.qkv_proj.weight",
            [AttentionQKV()],
        ),
        Rule(
            ["base.decoder.layers.self_attention.query_norm.scale"],
            "vllm_model.model.layers.{}.self_attn.q_norm.weight",
            [TransposeUnstack(axes=(1, 0))],
        ),
        Rule(
            ["base.decoder.layers.self_attention.key_norm.scale"],
            "vllm_model.model.layers.{}.self_attn.k_norm.weight",
            [TransposeUnstack(axes=(1, 0))],
        ),
        Rule(
            ["base.decoder.layers.moe_block.gate.kernel"],
            "vllm_model.model.layers.{}.mlp.gate.weight",
            [TransposeUnstack(axes=(1, 2, 0))],
        ),
        Rule(["base.decoder.layers.moe_block.wo"], "vllm_model.model.layers.{}.mlp.experts.w2_weight", [MoEExpertDown()]),
        Rule(
            ["base.decoder.layers.moe_block.wi_0", "base.decoder.layers.moe_block.wi_1"],
            "vllm_model.model.layers.{}.mlp.experts.w13_weight",
            [MoEFuseGateUp()],
        ),
        Rule(
            ["base.decoder.layers.moe_block.wi"],
            "vllm_model.model.layers.{}.mlp.experts.w13_weight",
            [MoEFuseGateUpPrefused()],
        ),
    ],
}
# Backward compatibility alias
_MODEL_TO_CONVERSION_RULES = MODEL_TO_CONVERSION_RULES


# ==========================================
# 5. MaxText -> MaxText structural converter
# ==========================================
# Used when vLLM runs the MaxText model itself (`MaxTextForCausalLM`), i.e. the
# "direct / no-op mappings" sync path. Both trees are MaxText, so no name
# translation is needed; the only structural differences are:
#   1. the trainer scans decoder layers while the rollout unrolls them
#      (`scan_layers: false` in configs/inference/vllm.yml), and
#   2. the rollout may pre-fuse MoE `wi_0`/`wi_1` into a single `wi`
#      (`prefuse_moe_weights`) and pad the intermediate dim for GMM_v2.

# Matches an unrolled layer attribute, e.g. "layers_7" or "layer_7".
_UNROLLED_LAYER_TOKEN = re.compile(r"^layers?_(\d+)$")

# Fallback filter for runtime state rather than transferable weights, used when
# the target state has been flattened past its nnx variable types. Prefix
# matching (not exact) so it also covers `cached_prefill_*` / `cached_ar_*`.
_NON_WEIGHT_PATH_PREFIXES = ("cache", "rngs", "intermediates")


class MoEFusedLayout:
  """How the rollout expects a pre-fused MoE `wi` kernel to be laid out.

  `PER_SHARD_INTERLEAVE` places each tensor-parallel shard's gate chunk
  immediately before its up chunk, zero-padding each chunk to the GMM_v2
  lane requirement. `CONCAT` is a plain `[gate | up]` split on the last axis.

  Which one is correct depends on how `moe.py` hands `wi` to the kernel: with
  `prefuse_moe_weights=True` and `attention="vllm_rpa"` the kernel receives
  `wi` untouched (moe.py:3100) rather than via the `wi[..., :n] / wi[..., n:]`
  split (moe.py:3103), so the layout is the kernel's, not MaxText's. This is
  resolved empirically -- see `validate_converter.py`.
  """

  PER_SHARD_INTERLEAVE = "per_shard_interleave"
  CONCAT = "concat"


@dataclasses.dataclass(frozen=True)
class _PlanEntry:
  """One target leaf and how to produce it from the source tree."""

  target_key: Tuple[Any, ...]
  source_keys: Tuple[Tuple[Any, ...], ...]
  # Index along the scan axis, or None when the source is already per-layer.
  slice_index: Optional[int]
  op: str  # "identity" | "slice" | "fuse_moe"


@dataclasses.dataclass(frozen=True)
class _PlanGroup:
  """Every target leaf produced from one source parameter, in one operation.

  The unrolled rollout reads N layers out of a single scanned trainer
  parameter. Executing that per target leaf means N slices, N alignments and
  -- for MoE -- N separate fuse programs against the same operand. Grouping
  lets the whole scanned tensor be cast, aligned and fused once, with the
  per-layer arrays falling out of a single unstack.

  `targets` is ordered by `slice_index`, but is *not* assumed dense: a
  trainer may legitimately have blocks that no rollout layer reads, so the
  slice index is carried explicitly rather than inferred from position.
  """

  source_keys: Tuple[Tuple[Any, ...], ...]
  op: str  # "identity" | "slice" | "fuse_moe"
  # (slice_index, target_key) pairs. slice_index is None only for "identity".
  targets: Tuple[Tuple[Optional[int], Tuple[Any, ...]], ...]
  # Dot-joined source path, precomputed for diagnostics and for the
  # leaf-name check inside `_align_per_axis`.
  source_path: str


def _group_plan(plan: List[_PlanEntry]) -> List[_PlanGroup]:
  """Collapses per-leaf plan entries into one group per source parameter.

  Entries are grouped by `(source_keys, op)`. Target shape is deliberately
  *not* part of the key: every rollout layer reading a given scanned source
  must share a shape, and if two did not, the bulk unstack would produce the
  wrong shape for one of them -- which `convert()`'s post-execution shape
  check catches loudly rather than silently writing incorrect weights.
  """
  grouped: Dict[Tuple[Any, ...], List[_PlanEntry]] = {}
  order: List[Tuple[Any, ...]] = []
  for entry in plan:
    key = (entry.source_keys, entry.op)
    if key not in grouped:
      grouped[key] = []
      order.append(key)
    grouped[key].append(entry)

  groups: List[_PlanGroup] = []
  for key in order:
    entries = grouped[key]
    source_keys, op = key
    groups.append(
        _PlanGroup(
            source_keys=source_keys,
            op=op,
            targets=tuple(
                (e.slice_index, e.target_key)
                # `identity` entries carry no index; -1 keeps the sort key
                # totally ordered so None is never compared against an int
                # (or against another None).
                for e in sorted(
                    entries,
                    key=lambda e: -1 if e.slice_index is None else e.slice_index,
                )
            ),
            source_path=".".join(map(str, source_keys[0])),
        )
    )
  return groups


class WeightConverterError(Exception):
  """Base class for all weight converter exceptions."""


class ConversionPlanError(WeightConverterError, ValueError):
  """Raised when the source and target trees cannot be fully reconciled."""


class UnscanShapeMismatch(WeightConverterError, ValueError):
  """Raised when an unscan shape mismatch is detected."""


class WeightSyncBindError(WeightConverterError, RuntimeError):
  """Raised when parameter binding to Raiden synchronizer fails."""


def _is_non_weight_path(key_tuple: Tuple[Any, ...]) -> bool:
  return any(isinstance(part, str) and part.lstrip("_").startswith(_NON_WEIGHT_PATH_PREFIXES) for part in key_tuple)


def _non_param_paths(state: Any) -> frozenset:
  """Paths in `state` held by an nnx variable that is not a Param.

  Preferred over `_is_non_weight_path`: KV caches, RNG state and intermediates
  are distinguished by their variable type, which is exact, where a name-based
  filter is guesswork. Returns an empty set when the target has already been
  reduced to plain arrays, in which case the name filter is all we have.
  """
  flat_state = getattr(state, "flat_state", None)
  if flat_state is None or not callable(flat_state):
    return frozenset()
  return frozenset(tuple(path) for path, var in flat_state() if not isinstance(var, nnx.Param))


def _find_unrolled_layer(key_tuple: Tuple[Any, ...]) -> Tuple[int, int, int]:
  """Locates an unrolled layer index in a target key.

  Handles both spellings MaxText produces: a fused attribute name
  (`("decoder", "layers_7", ...)`, from `_create_and_register_layer`) and a
  container plus index (`("decoder", "layers", 7, ...)`, from `nnx.List`).

  Returns:
    `(layer_index, prefix_len, suffix_start)` where the source key is rebuilt
    as `key[:prefix_len] + <scanned container> + key[suffix_start:]`, or
    `(-1, -1, -1)` when the key is not inside a decoder layer.
  """
  for i, part in enumerate(key_tuple):
    if isinstance(part, str):
      match = _UNROLLED_LAYER_TOKEN.match(part)
      if match:
        return int(match.group(1)), i, i + 1
      if part.isdigit() and i > 0 and key_tuple[i - 1] in ("layers", "layer"):
        return int(part), i - 1, i + 1
    elif isinstance(part, int) and i > 0 and key_tuple[i - 1] in ("layers", "layer"):
      return part, i - 1, i + 1
  return -1, -1, -1


def _summarize_tree_placement(flat: Mapping[Tuple[Any, ...], Any]) -> str:
  """Device-id span covering every leaf of a flattened tree.

  Used by the debug path to state up front which half of a split Pathways
  cluster each tree lives on, so a per-entry log line that falls outside its
  tree's span stands out.
  """
  ids = set()
  for v in flat.values():
    ids.update(_device_ids(v))
  if not ids:
    return "[unknown]"
  return f"[{min(ids)}..{max(ids)}]({len(ids)})"


class MaxTextToMaxTextConverter:
  """Structural converter for the direct (no-op mappings) sync path.

  The mapping between the scanned trainer tree and the unrolled rollout tree
  is fully determined by the model config, so the plan is derived once and
  reused for every subsequent weight sync.

  Execution is **grouped by source parameter**, not by target leaf. All N
  rollout layers that read one scanned trainer parameter are produced by a
  single cast + align + (for MoE) fuse, with the per-layer arrays falling out
  of one unstack. Running per target leaf instead would re-slice the same
  operand N times and, for MoE, dispatch N independent fuse programs -- on a
  48-layer model that is hundreds of extra dispatches and N sets of padded
  intermediates rather than one.

  Nothing rollout-side is read from `config`: the rollout's shapes, shardings
  and dtypes all come from `target_state` at convert time. That matters
  because the two configs genuinely disagree -- the rollout overrides
  `prefuse_moe_weights` and has `padded_base_moe_mlp_dim` injected by
  `maxtext_vllm_adapter/adapter.py` -- so deriving target properties from a
  config would silently skew. MoE fusion is instead triggered by the target
  tree carrying a `wi` leaf, and padding falls out of aligning to the target
  shape.

  Args:
    config: The MaxText config describing the **trainer's** layer structure
      (`num_decoder_layers`, `inhomogeneous_layer_cycle_interval`,
      `param_scan_axis`). The layer count and cycle are shared by both sides.
    moe_fused_layout: See `MoEFusedLayout`.
    allow_unused_source_keys: Source leaves permitted to have no target, as
      dot-joined substrings (e.g. a vision tower present only in the trainer).
    debug: Logs the placement of every operand and forces each conversion
      group to complete before the next is dispatched. Off by default because
      the barrier serializes what is otherwise a pipelined sync; turn it on
      to attribute a device/sharding failure to a specific parameter. The
      barrier is per *source parameter*, which is the granularity that
      matters -- shape and sharding are properties of the source, so every
      layer reading it fails or succeeds together.
  """

  def __init__(
      self,
      config: Any,
      moe_fused_layout: str = MoEFusedLayout.PER_SHARD_INTERLEAVE,
      allow_unused_source_keys: Tuple[str, ...] = (),
      debug: bool = False,
      prefuse_moe_weights: Optional[bool] = None,
      target_dtype: Optional[Any] = None,
  ):
    self.config = config
    self.moe_fused_layout = moe_fused_layout
    self.allow_unused_source_keys = allow_unused_source_keys
    self.debug = debug
    self.prefuse_moe_weights = (
        prefuse_moe_weights
        if prefuse_moe_weights is not None
        else getattr(config, "prefuse_moe_weights", False)
    )
    self.padded_base_moe_mlp_dim = getattr(config, "padded_base_moe_mlp_dim", None)
    self.target_dtype = target_dtype if target_dtype is not None else getattr(config, "weight_dtype", None)

    self.cycle = int(getattr(config, "inhomogeneous_layer_cycle_interval", 1) or 1)
    self.num_decoder_layers = int(config.num_decoder_layers)
    if self.num_decoder_layers % self.cycle:
      raise ConversionPlanError(
          f"num_decoder_layers ({self.num_decoder_layers}) is not a multiple of "
          f"inhomogeneous_layer_cycle_interval ({self.cycle}); the scanned "
          "block structure cannot be unrolled unambiguously."
      )
    self.num_blocks = self.num_decoder_layers // self.cycle
    self.scan_axis = int(getattr(config, "param_scan_axis", 1))

    self._plan: Optional[List[_PlanEntry]] = None
    self._groups: Optional[List[_PlanGroup]] = None

    logging.info(
        "MaxTextToMaxTextConverter: %d layers, cycle=%d, %d scanned blocks, "
        "scan_axis=%d, moe_fused_layout=%s, prefuse_moe=%s, padded_moe_dim=%s",
        self.num_decoder_layers,
        self.cycle,
        self.num_blocks,
        self.scan_axis,
        self.moe_fused_layout,
        self.prefuse_moe_weights,
        self.padded_base_moe_mlp_dim,
    )

  def _resolve_target_dtype(self):
    if self.target_dtype is None:
      return None
    if isinstance(self.target_dtype, str):
      if self.target_dtype in ("bfloat16", "bf16"):
        return jnp.bfloat16
      if self.target_dtype in ("float32", "fp32"):
        return jnp.float32
      return jnp.dtype(self.target_dtype)
    return self.target_dtype

  # -------------------------------------------------------------- #
  # Plan construction
  # -------------------------------------------------------------- #
  def _scanned_candidates(
      self, key_tuple: Tuple[Any, ...], prefix_len: int, suffix_start: int, slot: int
  ) -> List[Tuple[Any, ...]]:
    """Source keys that could hold the scanned form of a target layer key."""
    prefix, suffix = key_tuple[:prefix_len], key_tuple[suffix_start:]
    if self.cycle > 1:
      # Inhomogeneous: `Qwen3_5ScannableBlock` holds `layer_0..layer_{C-1}`.
      return [
          prefix + ("layers", f"layer_{slot}") + suffix,
          prefix + ("layers", str(slot)) + suffix,
          prefix + ("layers", slot) + suffix,
      ]
    # Homogeneous: a single scanned `layers` container.
    return [prefix + ("layers",) + suffix]

  def _build_target_free_plan(
      self,
      src_flat: Mapping[Tuple[Any, ...], Any],
  ) -> List[_PlanEntry]:
    """Builds the conversion plan directly from source keys and config without target state."""
    plan: List[_PlanEntry] = []
    consumed_wi_1 = set()

    for src_key in src_flat:
      if _is_non_weight_path(src_key):
        continue
      if src_key in consumed_wi_1:
        continue

      if "layers" not in src_key:
        plan.append(_PlanEntry(src_key, (src_key,), None, "identity"))
        continue

      idx = src_key.index("layers")
      prefix = src_key[:idx]
      rest = src_key[idx + 1 :]

      if self.cycle == 1:
        # Homogeneous: ("decoder", "layers", "self_attention", "query", "kernel")
        is_wi_0 = bool(rest and rest[-1] == "wi_0")
        wi_1_key = src_key[:-1] + ("wi_1",) if is_wi_0 else None
        fuse_moe = self.prefuse_moe_weights and is_wi_0 and (wi_1_key in src_flat)

        if fuse_moe:
          consumed_wi_1.add(wi_1_key)
          for i in range(self.num_decoder_layers):
            tgt_key = prefix + (f"layers_{i}",) + rest[:-1] + ("wi",)
            plan.append(_PlanEntry(tgt_key, (src_key, wi_1_key), i, "fuse_moe"))
        elif self.prefuse_moe_weights and rest and rest[-1] == "wi_1" and (src_key[:-1] + ("wi_0",) in src_flat):
          continue
        else:
          for i in range(self.num_decoder_layers):
            tgt_key = prefix + (f"layers_{i}",) + rest
            plan.append(_PlanEntry(tgt_key, (src_key,), i, "slice"))

      else:
        # Inhomogeneous hybrid cycle: ("decoder", "layers", "layer_0", "input_layernorm", "scale")
        slot_token = rest[0]
        if isinstance(slot_token, str) and slot_token.startswith("layer_"):
          slot = int(slot_token[6:])
        elif isinstance(slot_token, str) and slot_token.isdigit():
          slot = int(slot_token)
        elif isinstance(slot_token, int):
          slot = slot_token
        else:
          raise ConversionPlanError(f"Unexpected slot token {slot_token!r} in key {src_key}")

        suffix = rest[1:]
        is_wi_0 = bool(suffix and suffix[-1] == "wi_0")
        wi_1_key = src_key[:-1] + ("wi_1",) if is_wi_0 else None
        fuse_moe = self.prefuse_moe_weights and is_wi_0 and (wi_1_key in src_flat)

        if fuse_moe:
          consumed_wi_1.add(wi_1_key)
          for b in range(self.num_blocks):
            global_idx = b * self.cycle + slot
            tgt_key = prefix + (f"layers_{global_idx}",) + suffix[:-1] + ("wi",)
            plan.append(_PlanEntry(tgt_key, (src_key, wi_1_key), b, "fuse_moe"))
        elif self.prefuse_moe_weights and suffix and suffix[-1] == "wi_1" and (src_key[:-1] + ("wi_0",) in src_flat):
          continue
        else:
          for b in range(self.num_blocks):
            global_idx = b * self.cycle + slot
            tgt_key = prefix + (f"layers_{global_idx}",) + suffix
            plan.append(_PlanEntry(tgt_key, (src_key,), b, "slice"))

    return plan

  def _build_plan(
      self,
      src_flat: Mapping[Tuple[Any, ...], Any],
      tgt_flat: Mapping[Tuple[Any, ...], Any],
      skip_paths: frozenset = frozenset(),
  ) -> List[_PlanEntry]:
    """Builds the conversion plan mapping source keys to target keys."""
    plan: List[_PlanEntry] = []
    unmatched: List[Tuple[Any, ...]] = []
    consumed: set = set()

    for tgt_key in tgt_flat:
      if tgt_key in skip_paths or _is_non_weight_path(tgt_key):
        continue

      # 1. Identical path on both sides (embeddings, final norm, lm head,
      #    and any layer already stored unscanned).
      if tgt_key in src_flat:
        plan.append(_PlanEntry(tgt_key, (tgt_key,), None, "identity"))
        consumed.add(tgt_key)
        continue

      layer_idx, prefix_len, suffix_start = _find_unrolled_layer(tgt_key)
      if layer_idx < 0:
        unmatched.append(tgt_key)
        continue
      if layer_idx >= self.num_decoder_layers:
        raise ConversionPlanError(
            f"Target key {'.'.join(map(str, tgt_key))} refers to layer "
            f"{layer_idx}, beyond num_decoder_layers={self.num_decoder_layers}."
        )

      slot, block = layer_idx % self.cycle, layer_idx // self.cycle
      candidates = self._scanned_candidates(tgt_key, prefix_len, suffix_start, slot)

      # 2. Scanned source, sliced at this block index.
      matched = next((c for c in candidates if c in src_flat), None)
      if matched is not None:
        plan.append(_PlanEntry(tgt_key, (matched,), block, "slice"))
        consumed.add(matched)
        continue

      # 3. Rollout pre-fuses MoE `wi`; trainer still has `wi_0`/`wi_1`.
      if tgt_key and tgt_key[-1] == "wi":
        for cand in candidates:
          wi_0, wi_1 = cand[:-1] + ("wi_0",), cand[:-1] + ("wi_1",)
          if wi_0 in src_flat and wi_1 in src_flat:
            plan.append(_PlanEntry(tgt_key, (wi_0, wi_1), block, "fuse_moe"))
            consumed.update((wi_0, wi_1))
            break
        else:
          unmatched.append(tgt_key)
        continue

      unmatched.append(tgt_key)

    self._validate_plan(src_flat, tgt_flat, unmatched, consumed)
    return plan

  def _validate_plan(self, src_flat, tgt_flat, unmatched, consumed) -> None:
    """Fails loudly rather than leaving rollout weights at their dummy values."""
    if unmatched:
      shown = "\n  ".join(".".join(map(str, k)) for k in sorted(unmatched)[:40])
      raise ConversionPlanError(
          f"{len(unmatched)} rollout parameter(s) have no source in the trainer "
          f"state and would keep vLLM's random dummy weights:\n  {shown}" + ("\n  ..." if len(unmatched) > 40 else "")
      )

    unused = [
        k
        for k in src_flat
        if k not in consumed
        and not _is_non_weight_path(k)
        and not any(m in ".".join(map(str, k)) for m in self.allow_unused_source_keys)
    ]
    if unused:
      shown = "\n  ".join(".".join(map(str, k)) for k in sorted(unused)[:40])
      raise ConversionPlanError(
          f"{len(unused)} trainer parameter(s) were not transferred to the "
          f"rollout. This usually means a layer was renamed on one side:\n  {shown}"
          + ("\n  ..." if len(unused) > 40 else "")
          + "\nPass `allow_unused_source_keys` if the omission is intentional."
      )

    logging.info(
        "MaxTextToMaxTextConverter: plan covers %d rollout parameters from " "%d trainer parameters.",
        len(tgt_flat),
        len(consumed),
    )

  # -------------------------------------------------------------- #
  # Plan execution
  # -------------------------------------------------------------- #
  def _fuse_moe_bulk(self, wi_0, wi_1, tgt_val, key_path: str):
    """Fuses the *scanned* gate/up kernels, returning one array per block."""
    tgt_shape = tgt_val.shape
    tgt_fused_axis = len(tgt_shape) - 1
    scan_fused_axis = tgt_fused_axis if tgt_fused_axis < self.scan_axis else tgt_fused_axis + 1

    if self.moe_fused_layout == MoEFusedLayout.PER_SHARD_INTERLEAVE:
      return _fuse_and_unstack_moe(
          wi_0,
          wi_1,
          self.scan_axis,
          _get_n_shards(tgt_val, tgt_fused_axis),
          tgt_shape,
          scan_fused_axis,
          tgt_fused_axis,
      )

    if self.moe_fused_layout == MoEFusedLayout.CONCAT:
      half = tgt_shape[tgt_fused_axis] // 2
      half_shape = tgt_shape[:tgt_fused_axis] + (half,) + tgt_shape[tgt_fused_axis + 1 :]
      scanned_half_shape = half_shape[: self.scan_axis] + (wi_0.shape[self.scan_axis],) + half_shape[self.scan_axis :]
      scanned_sharding = _scanned_sharding_from_per_layer(getattr(tgt_val, "sharding", None), self.scan_axis)
      padded = [_align_per_axis(w, scanned_half_shape, scanned_sharding, key_path) for w in (wi_0, wi_1)]
      fused = jnp.concatenate(padded, axis=scan_fused_axis)
      return _jit_unstack(fused, self.scan_axis)

    raise ConversionPlanError(f"Unknown moe_fused_layout: {self.moe_fused_layout!r}")

  def _slice_bulk_target_free(self, val: Any, path: str):
    last_key = path.split(".")[-1]
    if isinstance(val, jax.ShapeDtypeStruct):
      unrolled_shape = list(val.shape[: self.scan_axis] + val.shape[self.scan_axis + 1 :])
      if last_key in _MOE_MLP_WEIGHTS and self.padded_base_moe_mlp_dim is not None:
        if last_key == "wo":
          if self.padded_base_moe_mlp_dim > unrolled_shape[1]:
            unrolled_shape[1] = self.padded_base_moe_mlp_dim
        elif last_key in ("wi_0", "wi_1", "wi"):
          if self.padded_base_moe_mlp_dim > unrolled_shape[-1]:
            unrolled_shape[-1] = self.padded_base_moe_mlp_dim
      return tuple(jax.ShapeDtypeStruct(tuple(unrolled_shape), val.dtype) for _ in range(val.shape[self.scan_axis]))

    if last_key in _MOE_MLP_WEIGHTS and self.padded_base_moe_mlp_dim is not None:
      if last_key == "wo":
        intermediate_axis = 2
        if self.padded_base_moe_mlp_dim > val.shape[intermediate_axis]:
          pad_amount = self.padded_base_moe_mlp_dim - val.shape[intermediate_axis]
          pad_spec = [(0, 0)] * val.ndim
          pad_spec[intermediate_axis] = (0, pad_amount)
          val = jnp.pad(val, pad_spec)
      elif last_key in ("wi_0", "wi_1"):
        intermediate_axis = len(val.shape) - 1
        if self.padded_base_moe_mlp_dim > val.shape[intermediate_axis]:
          pad_amount = self.padded_base_moe_mlp_dim - val.shape[intermediate_axis]
          pad_spec = [(0, 0)] * val.ndim
          pad_spec[intermediate_axis] = (0, pad_amount)
          val = jnp.pad(val, pad_spec)

    return _jit_unstack(val, self.scan_axis)

  def _fuse_moe_bulk_target_free(self, wi_0: Any, wi_1: Any, path: str):
    unpadded_dim = wi_0.shape[-1]
    target_intermediate = (
        self.padded_base_moe_mlp_dim
        if (self.padded_base_moe_mlp_dim is not None and self.padded_base_moe_mlp_dim > unpadded_dim)
        else unpadded_dim
    )
    if isinstance(wi_0, jax.ShapeDtypeStruct):
      fused_shape = (wi_0.shape[0], wi_0.shape[2], 2 * target_intermediate)
      return tuple(jax.ShapeDtypeStruct(fused_shape, wi_0.dtype) for _ in range(wi_0.shape[self.scan_axis]))

    tgt_shape = (wi_0.shape[0], wi_0.shape[2], 2 * target_intermediate)
    tgt_fused_axis = len(tgt_shape) - 1
    scan_fused_axis = tgt_fused_axis if tgt_fused_axis < self.scan_axis else tgt_fused_axis + 1

    if self.moe_fused_layout == MoEFusedLayout.PER_SHARD_INTERLEAVE:
      n_shards = _get_n_shards(wi_0, scan_fused_axis)
      return _fuse_and_unstack_moe(
          wi_0,
          wi_1,
          self.scan_axis,
          n_shards,
          tgt_shape,
          scan_fused_axis,
          tgt_fused_axis,
      )

    if self.moe_fused_layout == MoEFusedLayout.CONCAT:
      if target_intermediate > unpadded_dim:
        pad_spec = [(0, 0)] * wi_0.ndim
        pad_spec[-1] = (0, target_intermediate - unpadded_dim)
        wi_0 = jnp.pad(wi_0, pad_spec)
        wi_1 = jnp.pad(wi_1, pad_spec)
      fused = jnp.concatenate([wi_0, wi_1], axis=scan_fused_axis)
      return _jit_unstack(fused, self.scan_axis)

    raise ConversionPlanError(f"Unknown moe_fused_layout: {self.moe_fused_layout!r}")

  def _execute_group_target_free(self, group: _PlanGroup, src_flat):
    path = group.source_path
    target_dtype = self._resolve_target_dtype()
    is_pathways = bool("proxy" in os.environ.get("JAX_PLATFORMS", "") and os.environ.get("JAX_BACKEND_TARGET"))

    if group.op == "identity":
      raw_val = src_flat[group.source_keys[0]]
      if isinstance(raw_val, jax.ShapeDtypeStruct):
        val = _apply_dtype_cast(raw_val, target_dtype, path)
        return [(tgt_key, val) for _, tgt_key in group.targets]
      if is_pathways:
        np_val = np.asarray(jax.device_get(raw_val))
        if target_dtype is not None:
          np_val = np_val.astype(target_dtype)
        cpu = jax.local_devices(backend="cpu")[0]
        val = jax.device_put(np_val, cpu)
        del np_val
        return [(tgt_key, val) for _, tgt_key in group.targets]
      val = _apply_dtype_cast(raw_val, target_dtype, path)
      return [(tgt_key, val) for _, tgt_key in group.targets]

    if any(idx is None for idx, _ in group.targets):
      raise ConversionPlanError(
          f"Plan group for {path} has op={group.op!r} but a target with no "
          "scan index; only 'identity' targets may omit one."
      )

    if group.op == "fuse_moe":
      raw_0 = src_flat[group.source_keys[0]]
      raw_1 = src_flat[group.source_keys[1]]
      if isinstance(raw_0, jax.ShapeDtypeStruct):
        wi_0, wi_1 = (_apply_dtype_cast(raw_0, target_dtype, path), _apply_dtype_cast(raw_1, target_dtype, path))
        self._check_scan_axis(wi_0, path)
        per_block = self._fuse_moe_bulk_target_free(wi_0, wi_1, path)
        return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

      if is_pathways:
        wi_0 = np.asarray(jax.device_get(raw_0))
        wi_1 = np.asarray(jax.device_get(raw_1))
        if target_dtype is not None:
          wi_0 = wi_0.astype(target_dtype)
          wi_1 = wi_1.astype(target_dtype)
        self._check_scan_axis(wi_0, path)
        unpadded_dim = wi_0.shape[-1]
        target_intermediate = (
            self.padded_base_moe_mlp_dim
            if (self.padded_base_moe_mlp_dim is not None and self.padded_base_moe_mlp_dim > unpadded_dim)
            else unpadded_dim
        )
        tgt_shape = (wi_0.shape[0], wi_0.shape[2], 2 * target_intermediate)
        tgt_fused_axis = len(tgt_shape) - 1
        scan_fused_axis = tgt_fused_axis if tgt_fused_axis < self.scan_axis else tgt_fused_axis + 1

        if self.moe_fused_layout == MoEFusedLayout.PER_SHARD_INTERLEAVE:
          n_shards = 1
          target_half_dim = target_intermediate
          current_total_size = wi_0.shape[scan_fused_axis]
          chunk_size = current_total_size // n_shards
          target_chunk_size = target_half_dim // n_shards
          pad_amount = target_chunk_size - chunk_size
          if pad_amount > 0:
            pad_spec = [(0, 0)] * wi_0.ndim
            pad_spec[scan_fused_axis] = (0, pad_amount)
            wi_0 = np.pad(wi_0, pad_spec)
            wi_1 = np.pad(wi_1, pad_spec)
          fused = np.concatenate([wi_0, wi_1], axis=scan_fused_axis)
        elif self.moe_fused_layout == MoEFusedLayout.CONCAT:
          if target_intermediate > unpadded_dim:
            pad_spec = [(0, 0)] * wi_0.ndim
            pad_spec[-1] = (0, target_intermediate - unpadded_dim)
            wi_0 = np.pad(wi_0, pad_spec)
            wi_1 = np.pad(wi_1, pad_spec)
          fused = np.concatenate([wi_0, wi_1], axis=scan_fused_axis)
        else:
          raise ConversionPlanError(f"Unknown moe_fused_layout: {self.moe_fused_layout!r}")

        cpu = jax.local_devices(backend="cpu")[0]
        per_block = tuple(
            jax.device_put(np.ascontiguousarray(fused.take(indices=i, axis=self.scan_axis)), cpu)
            for i in range(self.num_blocks)
        )
        del fused, wi_0, wi_1
        return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

      wi_0, wi_1 = (_apply_dtype_cast(raw_0, target_dtype, path), _apply_dtype_cast(raw_1, target_dtype, path))
      self._check_scan_axis(wi_0, path)
      per_block = self._fuse_moe_bulk_target_free(wi_0, wi_1, path)
      return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

    # group.op == "slice"
    raw_val = src_flat[group.source_keys[0]]
    if isinstance(raw_val, jax.ShapeDtypeStruct):
      val = _apply_dtype_cast(raw_val, target_dtype, path)
      self._check_scan_axis(val, path)
      per_block = self._slice_bulk_target_free(val, path)
      return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

    if is_pathways:
      np_val = np.asarray(jax.device_get(raw_val))
      if target_dtype is not None:
        np_val = np_val.astype(target_dtype)
      self._check_scan_axis(np_val, path)
      last_key = path.split(".")[-1]
      if last_key in _MOE_MLP_WEIGHTS and self.padded_base_moe_mlp_dim is not None:
        if last_key == "wo":
          intermediate_axis = 2
          if self.padded_base_moe_mlp_dim > np_val.shape[intermediate_axis]:
            pad_amount = self.padded_base_moe_mlp_dim - np_val.shape[intermediate_axis]
            pad_spec = [(0, 0)] * np_val.ndim
            pad_spec[intermediate_axis] = (0, pad_amount)
            np_val = np.pad(np_val, pad_spec)
        elif last_key in ("wi_0", "wi_1", "wi"):
          intermediate_axis = len(np_val.shape) - 1
          if self.padded_base_moe_mlp_dim > np_val.shape[intermediate_axis]:
            pad_amount = self.padded_base_moe_mlp_dim - np_val.shape[intermediate_axis]
            pad_spec = [(0, 0)] * np_val.ndim
            pad_spec[intermediate_axis] = (0, pad_amount)
            np_val = np.pad(np_val, pad_spec)
      cpu = jax.local_devices(backend="cpu")[0]
      per_block = tuple(
          jax.device_put(np.ascontiguousarray(np_val.take(indices=i, axis=self.scan_axis)), cpu)
          for i in range(self.num_blocks)
      )
      del np_val
      return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

    val = _apply_dtype_cast(raw_val, target_dtype, path)
    self._check_scan_axis(val, path)
    per_block = self._slice_bulk_target_free(val, path)
    return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

  def _execute_group(self, group: _PlanGroup, src_flat, tgt_flat):
    """Produces every target leaf in `group`. Returns (target_key, array) pairs."""
    first_tgt = tgt_flat[group.targets[0][1]]
    path = group.source_path

    if group.op == "identity":
      val = _apply_dtype_cast(src_flat[group.source_keys[0]], first_tgt.dtype, path)
      out = _align_per_axis(val, first_tgt.shape, getattr(first_tgt, "sharding", None), path)
      return [(tgt_key, out) for _, tgt_key in group.targets]

    if any(idx is None for idx, _ in group.targets):
      raise ConversionPlanError(
          f"Plan group for {path} has op={group.op!r} but a target with no "
          "scan index; only 'identity' targets may omit one."
      )

    if group.op == "fuse_moe":
      wi_0, wi_1 = (_apply_dtype_cast(src_flat[k], first_tgt.dtype, path) for k in group.source_keys)
      self._check_scan_axis(wi_0, path)
      per_block = self._fuse_moe_bulk(wi_0, wi_1, first_tgt, path)
    else:  # "slice"
      val = _apply_dtype_cast(src_flat[group.source_keys[0]], first_tgt.dtype, path)
      self._check_scan_axis(val, path)
      per_block = _bulk_align_and_unstack(val, self.scan_axis, first_tgt, path)

    return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]

  def _check_scan_axis(self, val, path: str) -> None:
    if val.shape[self.scan_axis] != self.num_blocks:
      raise ConversionPlanError(
          f"Expected {self.num_blocks} scanned blocks on axis "
          f"{self.scan_axis} of {path}, found shape {val.shape}. Check "
          "param_scan_axis and inhomogeneous_layer_cycle_interval."
      )

  # -------------------------------------------------------------- #
  # Public API
  # -------------------------------------------------------------- #
  def convert(self, src_pytree: Any, target_state: Any = None) -> Dict[str, Any]:
    """Returns a nested dict of rollout weights, keyed by target paths.

    Pure: neither `src_pytree` nor `target_state` is mutated. Leaves are wrapped in nnx.Param.
    Note on memory lifecycle: `src_flat.pop()` frees internal flattened dict references
    as `result` is constructed to prevent dictionary growth overhead, but the caller's
    `src_pytree` retains full-tree references until `convert()` returns and the caller
    rebinds or drops its handle.
    """
    src_flat = traverse_util.flatten_dict(_to_pure_dict(src_pytree))
    src_flat, _ = _strip_root(src_flat, "base")

    if target_state is None:
      if self._plan is None:
        self._plan = self._build_target_free_plan(src_flat)
        self._groups = _group_plan(self._plan)

      result: Dict[Tuple[Any, ...], Any] = {}
      for group in self._groups:
        outs = self._execute_group_target_free(group, src_flat)
        # Drop processed source entries from src_flat to reduce dict overhead
        for k in group.source_keys:
          src_flat.pop(k, None)
        for tgt_key, out in outs:
          result[tgt_key] = out
        del outs

      del src_flat
      gc.collect()
      nested = traverse_util.unflatten_dict(result)
      del result
      gc.collect()
      _malloc_trim()
      return jax.tree_util.tree_map(
          lambda x: nnx.Param(x) if not isinstance(x, (nnx.Param, nnx.Variable)) else x,
          nested,
      )

    # Read variable types before purifying to plain arrays loses them.
    skip_paths = _non_param_paths(target_state)
    tgt_flat = traverse_util.flatten_dict(_to_pure_dict(target_state))

    tgt_flat, tgt_root = _strip_root(tgt_flat, "model")
    if tgt_root:
      depth = len(tgt_root)
      skip_paths = frozenset(p[depth:] for p in skip_paths if p[:depth] == tgt_root)

    if self._plan is None:
      self._plan = self._build_plan(src_flat, tgt_flat, skip_paths)
      self._groups = _group_plan(self._plan)

    if self.debug:
      src_span = _summarize_tree_placement(src_flat)
      tgt_span = _summarize_tree_placement(tgt_flat)
      logging.info(
          "weight_sync_debug: converting %d entries in %d source groups | "
          "source tree on devices %s | target tree on devices %s",
          len(self._plan),
          len(self._groups),
          src_span,
          tgt_span,
      )

    result: Dict[Tuple[Any, ...], Any] = {}
    for group in self._groups:
      if self.debug:
        for k in group.source_keys:
          logging.info(
              "weight_sync_debug: op=%s source=%s (%d targets) | src %s | tgt %s",
              group.op,
              ".".join(map(str, k)),
              len(group.targets),
              _sharding_summary(src_flat[k]),
              _sharding_summary(tgt_flat[group.targets[0][1]]),
          )
        try:
          outs = self._execute_group(group, src_flat, tgt_flat)
          jax.block_until_ready([out for _, out in outs])
        except BaseException:
          logging.exception(
              "weight_sync_debug: FAILED on op=%s sources=%s -> %d "
              "targets (first: %s). Output of the preceding groups "
              "completed successfully.",
              group.op,
              ", ".join(".".join(map(str, k)) for k in group.source_keys),
              len(group.targets),
              ".".join(map(str, group.targets[0][1])),
          )
          raise
        logging.info(
            "weight_sync_debug: ok op=%s source=%s -> %d outputs, first %s",
            group.op,
            group.source_path,
            len(outs),
            _sharding_summary(outs[0][1]),
        )
      else:
        outs = self._execute_group(group, src_flat, tgt_flat)

      for k in group.source_keys:
        src_flat.pop(k, None)

      for tgt_key, out in outs:
        tgt_val = tgt_flat[tgt_key]
        if hasattr(out, "shape") and hasattr(tgt_val, "shape") and out.shape != tgt_val.shape:
          raise ConversionPlanError(
              f"Shape mismatch after conversion for "
              f"{'.'.join(map(str, tgt_key))}: produced {out.shape}, "
              f"rollout expects {tgt_val.shape}."
          )
        result[tgt_root + tgt_key] = out
      del outs

    del src_flat, tgt_flat
    gc.collect()
    nested = traverse_util.unflatten_dict(result)
    del result
    gc.collect()
    _malloc_trim()
    return jax.tree_util.tree_map(
        lambda x: nnx.Param(x) if not isinstance(x, (nnx.Param, nnx.Variable)) else x,
        nested,
    )


def _rekey_to_target(flat_dotted: Dict[str, Any], target_state: Any) -> Dict[str, Any]:
  """Re-keys dotted converter output onto the target state's own key form.

  The rollout engine only assigns what a converter returns; matching its key
  encoding is therefore the converter's job, not the engine's. It matters
  because the two rollout targets are shaped differently: `MaxTextForCausalLM`
  exposes a nested `nnx.State`, while the native vLLM JAX models expose a flat
  dict whose keys are the dotted paths themselves. Joining the target's own
  flattened keys with '.' normalizes both.
  """
  if target_state is None:
    return traverse_util.unflatten_dict(flat_dotted, sep=".")

  target_flat = traverse_util.flatten_dict(_to_pure_dict(target_state))
  by_dotted = {".".join(str(p) for p in key): key for key in target_flat}

  matched, unmatched = {}, []
  for dotted, value in flat_dotted.items():
    key = by_dotted.get(dotted)
    if key is None:
      unmatched.append(dotted)
    else:
      matched[key] = value

  if unmatched:
    logging.warning(
        "%d converted tensors have no counterpart in the rollout state and " "were dropped, e.g. %s",
        len(unmatched),
        sorted(unmatched)[:5],
    )
  return traverse_util.unflatten_dict(matched)


def _to_pure_dict(tree: Any) -> Mapping[str, Any]:
  """Unwraps nnx.State / FrozenDict / nnx.Variable into plain nested dicts."""
  if hasattr(tree, "to_pure_dict"):
    tree = tree.to_pure_dict()
  if hasattr(tree, "unfreeze"):
    tree = tree.unfreeze()
  if hasattr(tree, "items"):
    return {k: _to_pure_dict(v) for k, v in tree.items()}
  if hasattr(tree, "value"):
    return tree.value
  return tree


def _strip_root(flat: Mapping[Tuple[Any, ...], Any], root: str) -> Tuple[Dict[Tuple[Any, ...], Any], Tuple[Any, ...]]:
  """Removes leading `root` levels shared by every key; returns them too."""
  stripped, depth = dict(flat), 0
  while stripped and all(k and k[0] == root for k in stripped):
    stripped = {k[1:]: v for k, v in stripped.items()}
    depth += 1
  return stripped, (root,) * depth
