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

"""Flax NNX adapter bridging m3 causal language models to Tunix trainers and rollouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import dataclasses
import functools
import inspect
import re
import sys
from typing import Any, NamedTuple, cast

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import checkpoint as m3_checkpoint


_UNSCANNED_SEQ_RE = re.compile(r"^base\.decoder\.layers\.layers\.(\d+)\.(.+)$")
_UNSCANNED_LIST_RE = re.compile(r"^base\.decoder\.layers\.(\d+)\.(.+)$")
_UNSCANNED_LEGACY_RE = re.compile(r"^base\.decoder\.layers_(\d+)\.(.+)$")


class M3RolloutCache(NamedTuple):
  """Stateful KV-cache PyTree pairing model KV buffers with physical write offset."""

  kv_cache: Any
  write_index: jax.Array


def segment_ids_from_attention_mask(
    attention_mask: jax.Array,
    input_tokens: jax.Array,
    *,
    allow_kv_cache: bool = False,
) -> jax.Array:
  """Recovers per-token non-pad segment IDs `[B, kv_len]` from Tunix's 3D causal mask.

  Tunix constructs causal masks via `input_mask[..., None, :] * tril`, so the
  final query row carries the per-key validity mask without query-side zeroing.
  Shape validation runs statically on shapes and therefore fails closed inside
  `jax.jit` and `nnx.jit` as well as eagerly.
  """
  if input_tokens.ndim != 2:
    raise ValueError(f"input_tokens must be rank 2 [batch, seq_len], got shape {input_tokens.shape}")
  if allow_kv_cache:
    if (
        attention_mask.ndim != 3
        or attention_mask.shape[0] != input_tokens.shape[0]
        or attention_mask.shape[1] != input_tokens.shape[1]
        or attention_mask.shape[2] < input_tokens.shape[1]
    ):
      raise ValueError(
          f"KV-cache attention_mask has shape {attention_mask.shape}, expected "
          f"({input_tokens.shape[0]}, {input_tokens.shape[1]}, >= {input_tokens.shape[1]})"
      )
  else:
    expected = (*input_tokens.shape, input_tokens.shape[1])
    if attention_mask.shape != expected:
      raise ValueError(f"attention_mask has shape {attention_mask.shape}, expected {expected}")
  return attention_mask[:, -1, :].astype(jnp.int32)


def _extract_declared_attrs(spec_obj: Any) -> dict[str, Any]:
  """Extracts declared field and property values from a model spec object without silent fallbacks."""
  if spec_obj is None:
    return {}
  if isinstance(spec_obj, Mapping):
    return dict(spec_obj)
  values: dict[str, Any] = {}
  if dataclasses.is_dataclass(spec_obj):
    for field_info in dataclasses.fields(spec_obj):
      values[field_info.name] = getattr(spec_obj, field_info.name)
    for base_cls in type(spec_obj).__mro__:
      for name, member in vars(base_cls).items():
        if name not in values and isinstance(member, property):
          values[name] = getattr(spec_obj, name)
    return values
  get_keys_fn = getattr(type(spec_obj), "get_keys", None)
  if callable(get_keys_fn):
    return dict(spec_obj.get_keys())
  model_dump_fn = getattr(type(spec_obj), "model_dump", None)
  if callable(model_dump_fn):
    dumped = dict(spec_obj.model_dump())
    extra = getattr(spec_obj, "__pydantic_extra__", None)
    if isinstance(extra, Mapping):
      dumped.update(extra)
    return dumped
  obj_vars = vars(spec_obj) if hasattr(spec_obj, "__dict__") else {}
  if "_pydantic_config" in obj_vars and obj_vars["_pydantic_config"] is not None:
    pyd_obj = obj_vars["_pydantic_config"]
    if hasattr(pyd_obj, "model_dump"):
      return dict(pyd_obj.model_dump())
  return dict(obj_vars)


def _first_present_int(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
  """Returns the first positive integer among `keys` present in `attrs`, else `None`."""
  for key in keys:
    if key in attrs and attrs[key] is not None:
      val = int(attrs[key])
      if val > 0:
        return val
  return None


@dataclasses.dataclass(frozen=True)
class RolloutCacheSpec:
  """Hashable KV-cache geometry exposed when rollout or inference callers inspect `actor.config`."""

  num_layers: int
  num_kv_heads: int
  head_dim: int
  vocab_size: int | None = None

  def __post_init__(self) -> None:
    for field_name in ("num_layers", "num_kv_heads", "head_dim"):
      value = getattr(self, field_name)
      if not isinstance(value, int) or value <= 0:
        raise ValueError(f"RolloutCacheSpec.{field_name} must be a positive int, got {value!r}")
    if self.vocab_size is not None and (not isinstance(self.vocab_size, int) or self.vocab_size <= 0):
      raise ValueError(f"RolloutCacheSpec.vocab_size must be a positive int or None, got {self.vocab_size!r}")

  @classmethod
  def from_model(cls, base_model: nnx.Module) -> RolloutCacheSpec:
    """Derives cache geometry from an m3 model's typed configuration object."""
    model_vars = vars(base_model) if hasattr(base_model, "__dict__") else {}
    spec_obj = model_vars.get("config")
    attrs = _extract_declared_attrs(spec_obj)
    if not attrs:
      raise ValueError(
          "Cannot derive RolloutCacheSpec because base_model.config is None or empty; "
          "pass an explicit `rollout_cache_spec=RolloutCacheSpec(...)`."
      )

    num_layers = _first_present_int(
        attrs, ("num_layers", "num_decoder_layers", "base_num_decoder_layers", "num_hidden_layers")
    )
    num_kv_heads = _first_present_int(attrs, ("num_kv_heads", "base_num_kv_heads", "num_key_value_heads"))
    head_dim = _first_present_int(attrs, ("head_dim", "qk_head_dim", "qk_nope_head_dim"))
    if head_dim is None:
      emb_dim = _first_present_int(attrs, ("emb_dim", "hidden_size"))
      num_q_heads = _first_present_int(attrs, ("num_query_heads", "base_num_query_heads", "num_attention_heads"))
      if emb_dim is not None and num_q_heads is not None and emb_dim % num_q_heads == 0:
        head_dim = emb_dim // num_q_heads
    vocab_size = _first_present_int(attrs, ("vocab_size",))

    if num_layers is None or num_kv_heads is None or head_dim is None:
      raise ValueError(
          "Cannot derive (num_layers, num_kv_heads, head_dim) from base_model.config "
          f"(resolved num_layers={num_layers}, num_kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}); pass an explicit `rollout_cache_spec`."
      )
    return cls(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
    )


@functools.cache
def _inspect_unbound_callable(
    func: Any,
) -> tuple[frozenset[str], frozenset[str], bool]:
  """Caches `(named_params, required_params, has_var_keyword)` for an unbound method."""
  sig = inspect.signature(func)
  named: list[str] = []
  required: list[str] = []
  has_var_kw = False
  for idx, (name, param) in enumerate(sig.parameters.items()):
    if idx == 0 and name == "self":
      continue
    if param.kind == inspect.Parameter.VAR_KEYWORD:
      has_var_kw = True
      continue
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
      continue
    named.append(name)
    if param.default is inspect.Parameter.empty:
      required.append(name)
  return frozenset(named), frozenset(required), has_var_kw


def _state_to_pure_arrays(node: Any) -> Any:
  """Converts an `nnx.State` or mapping tree into nested dicts of array leaves."""
  if isinstance(node, nnx.Variable):
    raw = node.get_raw_value()
    return raw if isinstance(raw, jax.ShapeDtypeStruct) else node.get_value()
  if isinstance(node, nnx.State):
    return _state_to_pure_arrays(nnx.to_pure_dict(node))
  if hasattr(node, "to_pure_dict"):
    return _state_to_pure_arrays(node.to_pure_dict())
  if isinstance(node, Mapping):
    return {k: _state_to_pure_arrays(v) for k, v in node.items()}
  if isinstance(node, tuple):
    if hasattr(node, "_fields"):
      return type(node)(*(_state_to_pure_arrays(v) for v in node))
    return tuple(_state_to_pure_arrays(v) for v in node)
  if isinstance(node, list):
    return [_state_to_pure_arrays(v) for v in node]
  return node


def _flat_param_shapes(param_state: nnx.State) -> dict[str, tuple[int, ...]]:
  """Returns `{dot_key: shape}` for all non-RNG leaves in `param_state`."""
  shapes: dict[str, tuple[int, ...]] = {}
  for path, leaf in nnx.to_flat_state(param_state):
    dot_key = ".".join(str(part) for part in path)
    if "rng" in dot_key:
      continue
    arr = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    raw_shape = getattr(arr, "shape", ())
    shapes[dot_key] = tuple(int(dim) for dim in raw_shape)
  return shapes


def _discover_unscanned_decoder_layers(
    param_keys: Iterable[str],
) -> tuple[str, tuple[int, ...]] | None:
  """Detects unscanned `base.decoder.layers*` prefix template and sorted layer indices."""
  for pattern, template in (
      (_UNSCANNED_SEQ_RE, "base.decoder.layers.layers.{idx}."),
      (_UNSCANNED_LIST_RE, "base.decoder.layers.{idx}."),
      (_UNSCANNED_LEGACY_RE, "base.decoder.layers_{idx}."),
  ):
    indices = sorted({int(m.group(1)) for key in param_keys if (m := pattern.match(key)) is not None})
    if indices:
      return template, tuple(indices)
  return None


def _strip_lora_suffix(param_key: str) -> str:
  """Strips a trailing `_lora_a` or `_lora_b` suffix when present."""
  for suffix in ("_lora_a", "_lora_b"):
    if param_key.endswith(suffix):
      return param_key[: -len(suffix)]
  return param_key


def _expand_unscanned_mapping_entries(
    raw: Mapping[str, Any],
    unscanned_info: tuple[str, tuple[int, ...]] | None,
    *,
    strip_layer_sharding: bool,
) -> dict[str, Any]:
  """Expands generalized `base.decoder.layers.<leaf>` keys onto concrete unscanned layer keys."""
  if unscanned_info is None:
    return dict(raw)
  prefix_tmpl, layer_indices = unscanned_info
  expanded: dict[str, Any] = {}
  scanned_prefix = "base.decoder.layers."
  for src_key, spec_val in raw.items():
    if (
        src_key.startswith(scanned_prefix)
        and not src_key.startswith("base.decoder.layers.layers.")
        and not _UNSCANNED_LIST_RE.match(src_key)
    ):
      suffix = src_key[len(scanned_prefix) :]
      for idx in layer_indices:
        concrete_src = f"{prefix_tmpl.format(idx=idx)}{suffix}"
        if isinstance(spec_val, tuple) and len(spec_val) == 2 and isinstance(spec_val[0], str):
          hf_target, sharding_spec = spec_val
          concrete_tgt = hf_target.replace("*", str(idx))
          if strip_layer_sharding and isinstance(sharding_spec, tuple):
            concrete_sharding = tuple(axis for axis in sharding_spec if axis != "layer")
          else:
            concrete_sharding = sharding_spec
          expanded[concrete_src] = (concrete_tgt, concrete_sharding)
        else:
          expanded[concrete_src] = spec_val
    else:
      expanded[src_key] = spec_val
  return expanded


def _build_legacy_maxtext_rollout_mappings(
    param_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[str, tuple[None, ...]]]:
  """Builds 1-to-1 `base.* -> model.*` mappings translating unscanned m3 layers to `layers_<i>`."""
  mappings: dict[str, tuple[str, tuple[None, ...]]] = {}
  for src_key, shape in param_shapes.items():
    unsharded = (None,) * len(shape)
    m_seq = _UNSCANNED_SEQ_RE.match(src_key)
    if m_seq is not None:
      idx, suffix = m_seq.group(1), m_seq.group(2)
      mappings[src_key] = (f"model.decoder.layers_{idx}.{suffix}", unsharded)
      continue
    m_list = _UNSCANNED_LIST_RE.match(src_key)
    if m_list is not None:
      idx, suffix = m_list.group(1), m_list.group(2)
      mappings[src_key] = (f"model.decoder.layers_{idx}.{suffix}", unsharded)
      continue
    if src_key.startswith("base."):
      mappings[src_key] = (f"model.{src_key[len('base.'):]}", unsharded)
    else:
      mappings[src_key] = (f"model.{src_key}", unsharded)
  return mappings


def _call_mapping_member(source: Any, names: tuple[str, ...], backend: str | None) -> dict[str, Any]:
  """Resolves a mapping dictionary or method on `source`, returning `{}` when absent."""
  if source is None:
    return {}
  if isinstance(source, Mapping):
    for name in names:
      if name in source and source[name] is not None:
        val = source[name]
        if callable(val):
          sig = inspect.signature(val)
          return dict(val(backend) if sig.parameters else val())
        return dict(val)
    return {}
  for name in names:
    member = getattr(source, name, None)
    if member is None:
      continue
    if callable(member):
      sig = inspect.signature(member)
      res = member(backend) if sig.parameters else member()
      return dict(res) if res is not None else {}
    return dict(member)
  return {}


def _bind_variable_meshes(
    base_model: nnx.Module,
    explicit_mesh: jax.sharding.Mesh | None = None,
) -> None:
  """Binds `mesh` metadata onto `out_sharding` variables so `nnx.Optimizer` works outside `set_mesh`."""
  for var in jax.tree_util.tree_leaves(nnx.state(base_model), is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if not isinstance(var, nnx.Variable):
      continue
    meta = var.get_metadata()
    if not meta.get("out_sharding") or meta.get("mesh") is not None:
      continue
    target_mesh = explicit_mesh
    if target_mesh is None:
      raw_val = var.get_raw_value()
      val_sharding = getattr(raw_val, "sharding", None)
      cand_mesh = getattr(val_sharding, "mesh", None)
      if isinstance(cand_mesh, jax.sharding.Mesh) and not cand_mesh.empty:
        target_mesh = cand_mesh
    if target_mesh is not None:
      var.set_metadata(mesh=target_mesh)


class M3TunixAdapter(nnx.Module):
  """First-class Flax NNX adapter bridging m3 causal LM models to Tunix and vLLM.

  Resolves the three m3 post-training / RL interface frictions without Linen
  monkeypatches (`jax.lax.with_sharding_constraint` / `jax.lax.top_k` stay
  unmodified) and without mutating `model.__class__`:

  1. Parameter hierarchy (`F3`): wraps `base_model` under `self.base` so
     `nnx.state(adapter, nnx.Param)` exposes the `"base"` root expected by Tunix
     `transfer_state_directly`, `MappingConfig`, and RL checkpoints while
     `nnx.state(adapter.base, nnx.Param)` preserves 100% unmodified m3 paths.
  2. Config contract (`F13`): keeps `self.config = None` by default (matching
     `MaxTextTrainingEngine` + Tunix + vLLM direct sync, even when
     `base_model.config` is `None` or a typed m3 dataclass), while optionally
     exposing a hashable frozen `RolloutCacheSpec` when `expose_rollout_config=True`.
  3. Call & mask plumbing (`F15`): translates Tunix `(input_tokens, positions,
     cache, attention_mask, segment_ids, skip_lm_head, forced_routed_experts)`
     calls into m3 forward conventions with strict `segment_ids` precedence and
     returns `(logits_or_hidden_states, cache_or_none)` 2-tuples.
  """

  def __init__(
      self,
      base_model: nnx.Module,
      *,
      pad_id: int | None = None,
      image_token_id: int | None = None,
      use_no_op_mappings: bool = True,
      use_standalone_mappings: bool = True,
      mapping_config: Mapping[str, Any] | Any | None = None,
      model_name: str | None = None,
      rollout_cache_spec: RolloutCacheSpec | None = None,
      expose_rollout_config: bool = False,
      filter_missing_source_keys: bool = True,
      require_complete_mappings: bool | None = None,
      unscanned_direct_sync_target: str = "legacy_maxtext",
      mesh: jax.sharding.Mesh | None = None,
  ):
    if not isinstance(base_model, nnx.Module):
      raise TypeError(f"M3TunixAdapter requires an nnx.Module base_model, got {type(base_model).__name__}")
    if pad_id is not None and (isinstance(pad_id, bool) or not isinstance(pad_id, int)):
      raise TypeError(f"pad_id must be an int or None, got {type(pad_id).__name__}")
    if image_token_id is not None and (isinstance(image_token_id, bool) or not isinstance(image_token_id, int)):
      raise TypeError(f"image_token_id must be an int or None, got {type(image_token_id).__name__}")
    if unscanned_direct_sync_target not in ("legacy_maxtext", "m3"):
      raise ValueError(
          f"unscanned_direct_sync_target must be 'legacy_maxtext' or 'm3', got {unscanned_direct_sync_target!r}"
      )

    _bind_variable_meshes(base_model, explicit_mesh=mesh)
    self.base = base_model
    if rollout_cache_spec is not None:
      self.config: RolloutCacheSpec | None = rollout_cache_spec
    elif expose_rollout_config:
      self.config = RolloutCacheSpec.from_model(base_model)
    else:
      self.config = None

    self._pad_id = pad_id
    self._image_token_id = image_token_id
    self._model_name = model_name
    self.use_no_op_mappings = bool(use_no_op_mappings)
    self._filter_missing_source_keys = bool(filter_missing_source_keys)
    self._unscanned_direct_sync_target = unscanned_direct_sync_target
    self._require_complete_mappings = (
        (mapping_config is None) if require_complete_mappings is None else bool(require_complete_mappings)
    )

    if self.use_no_op_mappings:
      self._mapping_source: Any = None
    elif mapping_config is not None:
      self._mapping_source = dict(mapping_config) if isinstance(mapping_config, Mapping) else mapping_config
      # Eagerly validate explicit mapping configuration so empty/mismatched maps fail at construction.
      self.to_hf_mappings()
    else:
      resolved_name = model_name
      if resolved_name is None:
        base_vars = vars(base_model) if hasattr(base_model, "__dict__") else {}
        attrs = _extract_declared_attrs(base_vars.get("config"))
        raw_name = attrs.get("model_name")
        if isinstance(raw_name, str) and raw_name:
          resolved_name = raw_name
      if not resolved_name:
        raise ValueError(
            "use_no_op_mappings=False requires an explicit `mapping_config` or `model_name` "
            "when base_model.config does not declare `model_name`."
        )
      # Lazy import avoids loading transformers / HF registries during pretrain or direct-sync init.
      from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=import-outside-toplevel
      from maxtext.integration.tunix.utils import VllmWeightMapping  # pylint: disable=import-outside-toplevel

      if resolved_name not in HF_MODEL_CONFIGS:
        raise ValueError(
            f"Model {resolved_name!r} has no entry in HF_MODEL_CONFIGS; "
            "pass `use_no_op_mappings=True` for direct weight sync or provide `mapping_config`."
        )
      self._mapping_source = VllmWeightMapping(
          resolved_name,
          HF_MODEL_CONFIGS[resolved_name].to_dict(),
          use_standalone_mappings,
      )
      # Eagerly verify that VllmWeightMapping covers 100% of base_model's parameters.
      self.to_hf_mappings()

  @property
  def pad_id(self) -> int | None:
    """Configured padding token ID used as lowest-priority segment mask fallback."""
    return self._pad_id

  @property
  def num_embed(self) -> int:
    """Vocabulary size exposed for rollout samplers inspecting `transformer.num_embed`."""
    if self.config is not None and self.config.vocab_size is not None:
      return self.config.vocab_size
    base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
    attrs = _extract_declared_attrs(base_vars.get("config"))
    vocab_size = _first_present_int(attrs, ("vocab_size",))
    if vocab_size is None:
      raise AttributeError(f"{type(self.base).__name__} does not declare a positive `vocab_size`.")
    return vocab_size

  def init_cache(
      self,
      batch_size: int,
      max_seq_len: int,
      dtype: Any = None,
  ) -> M3RolloutCache:
    """Allocates a stateful `M3RolloutCache` tracking KV buffers and physical write offset."""
    if not isinstance(batch_size, int) or batch_size <= 0:
      raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
      raise ValueError(f"max_seq_len must be a positive int, got {max_seq_len!r}")

    base_cls = type(self.base)
    model_init_fn = getattr(base_cls, "init_kv_cache", None)
    if callable(model_init_fn):
      kv = cast(Any, self.base).init_kv_cache(batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype)
      return M3RolloutCache(kv_cache=kv, write_index=jnp.zeros((), dtype=jnp.int32))

    fwd_details = getattr(base_cls, "forward_with_details", None)
    mod = sys.modules.get(base_cls.__module__)
    mod_init_fn = getattr(mod, "init_kv_cache", None) if mod is not None else None
    base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
    base_cfg = base_vars.get("config")
    if callable(fwd_details) and callable(mod_init_fn) and base_cfg is not None:
      det_params, _, _ = _inspect_unbound_callable(fwd_details)
      if "kv_cache" in det_params and "cache_position" in det_params:
        init_params, _, _ = _inspect_unbound_callable(mod_init_fn)
        init_kw: dict[str, Any] = {"dtype": dtype}
        if "mesh" in init_params and "mesh" in base_vars:
          init_kw["mesh"] = base_vars["mesh"]
        if "sharding" in init_params and "sharding" in base_vars:
          init_kw["sharding"] = base_vars["sharding"]
        kv = mod_init_fn(base_cfg, batch_size, max_seq_len, **init_kw)
        return M3RolloutCache(kv_cache=kv, write_index=jnp.zeros((), dtype=jnp.int32))

    raise NotImplementedError(
        f"{base_cls.__name__} does not implement external KV-cache allocation (`init_kv_cache`); "
        "use `VllmRollout` or an m3 model with stateful KV-cache support."
    )

  def resolve_segment_ids(
      self,
      input_tokens: jax.Array,
      *,
      segment_ids: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      cache: Any = None,
  ) -> jax.Array | None:
    """Resolves per-token segment IDs in strict priority order.

    Priority: `segment_ids` > `decoder_segment_ids` > `attention_mask` > `pad_id`.
    """
    resolved: jax.Array | None = None
    if segment_ids is not None:
      resolved = jnp.asarray(segment_ids, dtype=jnp.int32)
    elif decoder_segment_ids is not None:
      resolved = jnp.asarray(decoder_segment_ids, dtype=jnp.int32)
    elif attention_mask is not None:
      resolved = segment_ids_from_attention_mask(attention_mask, input_tokens, allow_kv_cache=cache is not None)
    elif self._pad_id is not None and cache is None:
      resolved = (input_tokens != self._pad_id).astype(jnp.int32)

    if resolved is not None:
      if resolved.ndim != 2 or resolved.shape[0] != input_tokens.shape[0]:
        raise ValueError(
            f"segment_ids has shape {resolved.shape}, expected batch-matched 2D " f"array ({input_tokens.shape[0]}, ...)"
        )
      if cache is None and resolved.shape != input_tokens.shape:
        raise ValueError(f"segment_ids has shape {resolved.shape}, expected {input_tokens.shape}")
    return resolved

  def _build_common_call_kwargs(
      self,
      *,
      call_params: frozenset[str],
      enable_dropout: bool,
      images: jax.Array | None,
      forced_routed_experts: jax.Array | None,
      extra_kwargs: Mapping[str, Any],
  ) -> dict[str, Any]:
    """Builds validated model keyword arguments without silent kwarg drops."""
    forwarded: dict[str, Any] = {}
    if "enable_dropout" in call_params:
      forwarded["enable_dropout"] = enable_dropout
    elif "deterministic" in call_params:
      forwarded["deterministic"] = not enable_dropout
    elif enable_dropout:
      raise NotImplementedError(
          f"{type(self.base).__name__} does not accept `enable_dropout` or `deterministic`; cannot enable dropout."
      )

    if images is not None:
      if "encoder_images" in call_params:
        forwarded["encoder_images"] = images
      elif "images" in call_params:
        forwarded["images"] = images
      else:
        raise NotImplementedError(f"{type(self.base).__name__} does not accept multimodal image inputs.")

    if forced_routed_experts is not None:
      if "forced_routed_experts" in call_params:
        forwarded["forced_routed_experts"] = forced_routed_experts
      else:
        raise NotImplementedError(f"{type(self.base).__name__} does not support `forced_routed_experts` router replay.")

    for key, value in extra_kwargs.items():
      forwarded[key] = value
    return forwarded

  def _resolve_image_token_id(self, cfg_attrs: Mapping[str, Any]) -> int:
    """Resolves multimodal image placeholder token ID from adapter, model config, or HF_MODEL_CONFIGS."""
    if self._image_token_id is not None:
      return self._image_token_id
    cfg_token_id = _first_present_int(cfg_attrs, ("image_token_id", "image_pad_token_id"))
    if cfg_token_id is not None:
      return cfg_token_id
    resolved_name = self._model_name
    if resolved_name is None:
      raw_name = cfg_attrs.get("model_name")
      if isinstance(raw_name, str) and raw_name:
        resolved_name = raw_name
    if resolved_name:
      # Lazy import avoids loading transformers / HF registries during text-only pretrain or rollout init.
      from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=import-outside-toplevel

      if resolved_name in HF_MODEL_CONFIGS:
        hf_cfg = HF_MODEL_CONFIGS[resolved_name]
        hf_token_id = getattr(hf_cfg, "image_token_id", None)
        if isinstance(hf_token_id, int) and not isinstance(hf_token_id, bool):
          return hf_token_id
    raise ValueError(
        f"{type(self.base).__name__} multimodal skip_lm_head requires `image_token_id` "
        "on M3TunixAdapter, base_model.config, or HF_MODEL_CONFIGS[model_name]."
    )

  def _forward_hidden_states(
      self,
      input_tokens: jax.Array,
      positions: jax.Array,
      resolved_segment_ids: jax.Array | None,
      *,
      cache: Any,
      enable_dropout: bool,
      images: jax.Array | None,
      forced_routed_experts: jax.Array | None,
      extra_kwargs: Mapping[str, Any],
  ) -> jax.Array:
    """Computes pre-head hidden states `[B, L, D]` when `skip_lm_head=True`."""
    if cache is not None:
      raise NotImplementedError(
          "skip_lm_head=True is for chunked log-probability evaluation and cannot be combined with KV cache."
      )
    base_cls = type(self.base)
    call_params, _, _ = _inspect_unbound_callable(base_cls.__call__)
    common_kw = self._build_common_call_kwargs(
        call_params=call_params,
        enable_dropout=enable_dropout,
        images=images,
        forced_routed_experts=forced_routed_experts,
        extra_kwargs=extra_kwargs,
    )

    if "skip_lm_head" in call_params:
      raw = self.base(
          decoder_input_tokens=input_tokens,
          decoder_positions=positions,
          decoder_segment_ids=resolved_segment_ids,
          skip_lm_head=True,
          **common_kw,
      )
      return raw[0] if isinstance(raw, tuple) else raw

    fwd_with_aux = getattr(base_cls, "forward_with_aux", None)
    if callable(fwd_with_aux):
      aux_params, _, _ = _inspect_unbound_callable(fwd_with_aux)
      if "compute_logits" in aux_params:
        aux_kw = self._build_common_call_kwargs(
            call_params=aux_params,
            enable_dropout=enable_dropout,
            images=images,
            forced_routed_experts=forced_routed_experts,
            extra_kwargs=extra_kwargs,
        )
        out = cast(Any, self.base).forward_with_aux(
            decoder_input_tokens=input_tokens,
            decoder_positions=positions,
            decoder_segment_ids=resolved_segment_ids,
            compute_logits=False,
            return_aux=False,
            **aux_kw,
        )
        return out.hidden_states

    vocab_tiling_fn = getattr(base_cls, "logits_from_hidden_states_for_vocab_tiling", None)
    base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
    decoder = base_vars.get("decoder")
    token_embedder = base_vars.get("token_embedder")
    vision_encoder = base_vars.get("vision_encoder")
    if callable(vocab_tiling_fn) and decoder is not None and token_embedder is not None:
      dec_vars = vars(decoder) if hasattr(decoder, "__dict__") else {}
      layers = dec_vars.get("layers")
      if images is None and forced_routed_experts is None and not extra_kwargs and layers is not None:
        hidden = token_embedder(input_tokens)
        deterministic = not enable_dropout
        if callable(layers):
          return layers(
              hidden,
              decoder_segment_ids=resolved_segment_ids,
              decoder_positions=positions,
              deterministic=deterministic,
          )
        if not isinstance(layers, Iterable):
          raise NotImplementedError(f"{base_cls.__name__} skip_lm_head path received non-iterable layers.")
        for layer in layers:
          hidden = layer(
              hidden,
              decoder_segment_ids=resolved_segment_ids,
              decoder_positions=positions,
              deterministic=deterministic,
          )
        return hidden
      if images is not None and callable(vision_encoder) and layers is not None and forced_routed_experts is None:
        extra_kw_mut = dict(extra_kwargs)
        image_masks = extra_kw_mut.pop("encoder_image_masks", None)
        video_grid_thw = extra_kw_mut.pop("encoder_video_grid_thw", None)
        if extra_kw_mut:
          raise NotImplementedError(
              f"{base_cls.__name__} multimodal skip_lm_head path received unsupported kwargs: {sorted(extra_kw_mut)}"
          )
        layers_seq = vars(layers).get("layers") if isinstance(layers, nnx.Sequential) else layers
        if not isinstance(layers_seq, Iterable):
          raise NotImplementedError(
              f"{base_cls.__name__} multimodal skip_lm_head path does not support scanned layers; "
              "set scan_layers=False."
          )
        mod = sys.modules.get(base_cls.__module__)
        merge_fn = getattr(mod, "merge_multimodal_embeddings", None) if mod is not None else None
        deepstack_fn = getattr(mod, "deepstack_process", None) if mod is not None else None
        compact_fn = getattr(mod, "_compact_visual_features", None) if mod is not None else None
        if not callable(merge_fn):
          raise NotImplementedError(f"{base_cls.__name__} multimodal skip_lm_head could not locate merge helper.")
        cfg_attrs = _extract_declared_attrs(base_vars.get("config"))
        image_pad_id = self._resolve_image_token_id(cfg_attrs)
        vis_embeds, deepstack_embeds = vision_encoder(
            input_images=images,
            input_masks=image_masks,
            video_grid_thw=video_grid_thw,
        )
        feature_mask = None
        if image_masks is not None:
          t_patch = int(cfg_attrs["temporal_patch_size_for_vit"])
          s_patch = int(cfg_attrs["patch_size_for_vit"])
          s_merge = int(cfg_attrs["spatial_merge_size_for_vit"])
          patch_elems = t_patch * (s_patch**2)
          patch_mask = image_masks.reshape(vis_embeds.shape[0], -1, patch_elems).any(axis=-1)
          feature_mask = patch_mask.reshape(vis_embeds.shape[0], -1, s_merge**2).any(axis=-1)
        visual_mask = input_tokens == image_pad_id
        if feature_mask is not None:
          valid_count = feature_mask.sum(axis=1, keepdims=True)
          visual_mask = visual_mask & (jnp.cumsum(visual_mask, axis=1) <= valid_count)
        hidden = merge_fn(
            text_embeddings=token_embedder(input_tokens),
            multimodal_embeddings=vis_embeds,
            mask=visual_mask,
            feature_mask=feature_mask,
        )
        deterministic = not enable_dropout
        for lyr_idx, layer in enumerate(layers_seq):
          hidden = layer(
              hidden,
              decoder_segment_ids=resolved_segment_ids,
              decoder_positions=positions,
              deterministic=deterministic,
          )
          if deepstack_embeds is not None and lyr_idx < len(deepstack_embeds) and callable(deepstack_fn):
            feat = deepstack_embeds[lyr_idx]
            if feat is not None and feature_mask is not None and callable(compact_fn):
              feat = compact_fn(feat, feature_mask)
            if feat is not None:
              hidden = deepstack_fn(hidden, visual_mask, feat)
        return hidden
      raise NotImplementedError(f"{base_cls.__name__} skip_lm_head path does not support multimodal or extra kwargs.")

    fwd_details = getattr(base_cls, "forward_with_details", None)
    if callable(fwd_details):
      det_params, _, _ = _inspect_unbound_callable(fwd_details)
      if images is not None or forced_routed_experts is not None or extra_kwargs:
        raise NotImplementedError(
            f"{base_cls.__name__} forward_with_details skip_lm_head path received unsupported kwargs."
        )
      det_kw: dict[str, Any] = {"deterministic": not enable_dropout}
      if "segment_ids" in det_params:
        det_kw["segment_ids"] = resolved_segment_ids
      out = cast(Any, self.base).forward_with_details(input_tokens, positions, **det_kw)
      return out.hidden_states

    lm_head = base_vars.get("lm_head")
    if decoder is not None and callable(decoder) and callable(lm_head):
      if images is not None or forced_routed_experts is not None or extra_kwargs:
        raise NotImplementedError(f"{base_cls.__name__} decoder skip_lm_head path received unsupported kwargs.")
      dec_params, _, _ = _inspect_unbound_callable(type(decoder).__call__)
      dec_kw: dict[str, Any] = {}
      if "return_aux" in dec_params:
        dec_kw["return_aux"] = False
      if "return_pre_mixer" in dec_params:
        dec_kw["return_pre_mixer"] = False
      hidden = decoder(input_tokens, positions, resolved_segment_ids, **dec_kw)
      return hidden[0] if isinstance(hidden, tuple) else hidden

    raise NotImplementedError(
        f"{base_cls.__name__} does not expose a pre-LM-head hidden-state hook for skip_lm_head=True."
    )

  def compute_final_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Projects pre-head hidden states `[..., Emb]` to vocabulary logits `[..., Vocab]`."""
    base_cls = type(self.base)
    vocab_tiling_fn = getattr(base_cls, "logits_from_hidden_states_for_vocab_tiling", None)
    if callable(vocab_tiling_fn):
      vt_params, vt_required, _ = _inspect_unbound_callable(vocab_tiling_fn)
      vt_kw: dict[str, Any] = {}
      if "deterministic" in vt_params:
        vt_kw["deterministic"] = True
      if "model_mode" in vt_required:
        base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
        vt_kw["model_mode"] = base_vars.get("model_mode", "train")
      return cast(Any, self.base).logits_from_hidden_states_for_vocab_tiling(hidden_states, **vt_kw)

    base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
    lm_head = base_vars.get("lm_head")
    embed_tokens = base_vars.get("embed_tokens")
    spec_attrs = _extract_declared_attrs(base_vars.get("config"))

    if callable(lm_head):
      raw_logits = lm_head(hidden_states)
    elif embed_tokens is not None and spec_attrs.get("tie_word_embeddings", False):
      emb_weight = embed_tokens.embedding[...].astype(hidden_states.dtype)
      precision = base_vars.get("precision")
      raw_logits = jnp.einsum("...h,vh->...v", hidden_states, emb_weight, precision=precision)
    else:
      raise NotImplementedError(
          f"{base_cls.__name__} does not implement `logits_from_hidden_states_for_vocab_tiling` or `lm_head`."
      )

    sharding_obj = base_vars.get("sharding")
    if callable(sharding_obj) and hasattr(sharding_obj, "constrain") and raw_logits.ndim == 3:
      raw_logits = sharding_obj.constrain(raw_logits, "logits", ("batch", "length", "vocab"))

    logits_scaling = spec_attrs.get("logits_scaling")
    if logits_scaling is not None and float(logits_scaling) != 1.0:
      raw_logits = raw_logits / jnp.asarray(logits_scaling, dtype=raw_logits.dtype)
    return raw_logits

  def _validate_positions_shape(self, input_tokens: jax.Array, positions: jax.Array) -> None:
    """Validates `positions` shape against `input_tokens` (including 3D M-RoPE `(B, L, 3)` / `(B, L, 1)`)."""
    if positions.shape == input_tokens.shape:
      return
    base_vars = vars(self.base) if hasattr(self.base, "__dict__") else {}
    attrs = _extract_declared_attrs(base_vars.get("config"))
    mrope = attrs.get("mrope_section")
    has_mrope = isinstance(mrope, (tuple, list)) and len(mrope) == 3
    call_params, _, _ = _inspect_unbound_callable(type(self.base).__call__)
    supports_3d_mrope = has_mrope and ("encoder_images" in call_params or "images" in call_params)
    if supports_3d_mrope and positions.shape in ((*input_tokens.shape, 3), (*input_tokens.shape, 1)):
      return
    raise ValueError(f"positions has shape {positions.shape}, expected {input_tokens.shape}")

  # pylint: disable=too-many-positional-arguments
  def __call__(
      self,
      input_tokens: jax.Array,
      positions: jax.Array | None = None,
      cache: Any = None,
      attention_mask: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      output_hidden_states: bool = False,
      forced_routed_experts: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      skip_lm_head: bool = False,
      *,
      images: jax.Array | None = None,
      enable_dropout: bool = False,
      **kwargs: Any,
  ) -> tuple[jax.Array, Any]:
    """Executes m3 model under Tunix's trainer and rollout call signature.

    Returns `(logits, updated_cache_or_none)` (or `(hidden_states, None)` when
    `skip_lm_head=True`).
    """
    del output_hidden_states
    if input_tokens.ndim != 2:
      raise ValueError(f"input_tokens must be rank 2 [batch, seq_len], got shape {input_tokens.shape}")
    if positions is None:
      positions = jnp.broadcast_to(
          jnp.arange(input_tokens.shape[1], dtype=jnp.int32)[None, :],
          input_tokens.shape,
      )
    else:
      self._validate_positions_shape(input_tokens, positions)

    resolved_segment_ids = self.resolve_segment_ids(
        input_tokens,
        segment_ids=segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        attention_mask=attention_mask,
        cache=cache,
    )

    if skip_lm_head:
      hidden = self._forward_hidden_states(
          input_tokens,
          positions,
          resolved_segment_ids,
          cache=cache,
          enable_dropout=enable_dropout,
          images=images,
          forced_routed_experts=forced_routed_experts,
          extra_kwargs=kwargs,
      )
      return hidden, None

    base_cls = type(self.base)
    call_params, _, _ = _inspect_unbound_callable(base_cls.__call__)
    is_rollout_cache = isinstance(cache, M3RolloutCache)
    raw_cache = cache.kv_cache if is_rollout_cache else cache

    if cache is not None:
      fwd_details = getattr(base_cls, "forward_with_details", None)
      if callable(fwd_details):
        det_params, _, _ = _inspect_unbound_callable(fwd_details)
        if "kv_cache" in det_params:
          if images is not None or forced_routed_experts is not None:
            raise NotImplementedError(
                f"{base_cls.__name__} KV-cache forward_with_details path does not support multimodal or router replay."
            )
          if isinstance(raw_cache, Mapping):
            raise TypeError(
                f"{base_cls.__name__} requires a typed KV cache allocated via `adapter.init_cache(...)`, got dict."
            )
          extra_det = dict(kwargs)
          explicit_cache_pos = extra_det.pop("cache_position", None)
          if explicit_cache_pos is not None:
            cache_pos: Any = explicit_cache_pos
          elif is_rollout_cache:
            cache_pos = cache.write_index
          else:
            raise ValueError(
                f"{base_cls.__name__} received a raw KV cache without `cache_position`, which would overwrite "
                "cache slot 0; allocate via `adapter.init_cache(...)` or pass `cache_position` explicitly."
            )
          det_kw: dict[str, Any] = {
              "kv_cache": raw_cache,
              "deterministic": not enable_dropout,
          }
          if "cache_position" in det_params:
            det_kw["cache_position"] = cache_pos
          if "segment_ids" in det_params:
            det_kw["segment_ids"] = resolved_segment_ids
          for key, value in extra_det.items():
            det_kw[key] = value
          out = cast(Any, self.base).forward_with_details(input_tokens, positions, **det_kw)
          if out.kv_cache is None:
            raise NotImplementedError(f"{base_cls.__name__}.forward_with_details returned None for `kv_cache`.")
          if is_rollout_cache:
            next_index = jnp.asarray(cache_pos, dtype=jnp.int32) + jnp.asarray(input_tokens.shape[1], dtype=jnp.int32)
            return out.logits, M3RolloutCache(kv_cache=out.kv_cache, write_index=next_index)
          return out.logits, out.kv_cache

    forwarded = self._build_common_call_kwargs(
        call_params=call_params,
        enable_dropout=enable_dropout,
        images=images,
        forced_routed_experts=forced_routed_experts,
        extra_kwargs=kwargs,
    )
    if cache is not None:
      if "kv_cache" in call_params:
        forwarded["kv_cache"] = raw_cache
      elif "cache" in call_params:
        forwarded["cache"] = raw_cache
      else:
        raise NotImplementedError(f"{base_cls.__name__} does not accept a KV `cache` argument.")

    outputs = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        decoder_segment_ids=resolved_segment_ids,
        **forwarded,
    )
    updated_cache: Any = None
    if isinstance(outputs, tuple):
      logits_out = outputs[0]
      updated_cache = outputs[1] if cache is not None and len(outputs) >= 2 else None
    elif hasattr(outputs, "logits"):
      logits_out = outputs.logits
      out_vars = vars(outputs) if hasattr(outputs, "__dict__") else {}
      updated_cache = out_vars.get("kv_cache") if cache is not None else None
    else:
      logits_out = outputs

    if cache is not None:
      if updated_cache is None:
        raise NotImplementedError(f"{base_cls.__name__} ignored non-None `cache` and did not return an updated KV cache.")
      if is_rollout_cache:
        next_index = cache.write_index + jnp.asarray(input_tokens.shape[1], dtype=jnp.int32)
        return logits_out, M3RolloutCache(kv_cache=updated_cache, write_index=next_index)
      return logits_out, updated_cache
    return logits_out, None

  def param_state(self) -> nnx.State:
    """Returns `nnx.Param` state rooted under `"base"` for Tunix trainers and `RLCluster`."""
    return nnx.state(self, nnx.Param)

  def base_param_state(self) -> nnx.State:
    """Returns unwrapped m3 `nnx.Param` state matching `MaxTextTrainingEngine`."""
    return nnx.state(self.base, nnx.Param)

  def to_rollout_pure_dict(
      self,
      state: Any = None,
      *,
      legacy_layer_names: bool = True,
  ) -> dict[str, Any]:
    """Exports `"base"`-prefixed pure array tree for rollout weight synchronization.

    When `legacy_layer_names=True`, unscanned m3 `decoder/layers/layers/<i>`
    containers are renamed to `decoder/layers_<i>` without copying array buffers.
    """
    src = self.param_state() if state is None else state
    pure = _state_to_pure_arrays(src)
    if not isinstance(pure, dict):
      raise TypeError(f"Expected dict state tree, got {type(pure).__name__}")
    has_base = "base" in pure and isinstance(pure["base"], Mapping)
    inner = dict(pure["base"]) if has_base else dict(pure)
    if legacy_layer_names:
      inner = m3_checkpoint.to_legacy_layer_names(inner)
    return {"base": inner}

  def to_hf_mappings(self, backend: str | None = None) -> dict[str, Any]:
    """Returns parameter-key mappings for rollout weight synchronization."""
    param_shapes = _flat_param_shapes(self.param_state())
    base_keys = {k for k in param_shapes if _strip_lora_suffix(k) == k}
    lora_keys = set(param_shapes.keys()) - base_keys
    unscanned_info = _discover_unscanned_decoder_layers(base_keys)

    if self.use_no_op_mappings:
      if self._unscanned_direct_sync_target == "legacy_maxtext" and unscanned_info is not None:
        return _build_legacy_maxtext_rollout_mappings(param_shapes)
      return {}

    raw = _call_mapping_member(self._mapping_source, ("to_hf_mappings", "to_hf_mapping"), backend)
    if not raw:
      raise ValueError(f"No vLLM/HF weight mappings found for {type(self.base).__name__}.")

    expanded = _expand_unscanned_mapping_entries(raw, unscanned_info, strip_layer_sharding=True)
    if self._filter_missing_source_keys:
      filtered = {k: v for k, v in expanded.items() if k in base_keys}
    else:
      filtered = expanded

    if not filtered:
      raise ValueError(
          f"Resolved to_hf_mappings() for {type(self.base).__name__} is empty; "
          "none of the mapping source keys match base_model parameters."
      )
    if self._require_complete_mappings:
      unmapped_base = sorted(base_keys - set(filtered.keys()))
      if unmapped_base:
        raise ValueError(
            f"Incomplete vLLM weight mapping for {type(self.base).__name__}: "
            f"{len(unmapped_base)} of {len(base_keys)} base parameter(s) are unmapped "
            f"(first unmapped: {unmapped_base[:3]}); pass `use_no_op_mappings=True` or a complete `mapping_config`."
        )
      if lora_keys:
        lora_map = self.lora_to_hf_mappings(backend)
        unmapped_lora = sorted(lora_keys - set(lora_map.keys()))
        if unmapped_lora:
          raise ValueError(
              f"Incomplete vLLM LoRA weight mapping for {type(self.base).__name__}: "
              f"{len(unmapped_lora)} of {len(lora_keys)} LoRA parameter(s) are unmapped "
              f"(first unmapped: {unmapped_lora[:3]}); pass `use_no_op_mappings=True` or a complete `mapping_config`."
          )
    return filtered

  def lora_to_hf_mappings(self, backend: str | None = None) -> dict[str, Any]:
    """Returns LoRA parameter-key mappings for Hugging Face / vLLM rollout transfer."""
    if self.use_no_op_mappings:
      return {}
    raw = _call_mapping_member(self._mapping_source, ("lora_to_hf_mappings",), backend)
    if not raw:
      return {}
    param_shapes = _flat_param_shapes(self.param_state())
    base_keys = {k for k in param_shapes if _strip_lora_suffix(k) == k}
    lora_keys = set(param_shapes.keys()) - base_keys
    unscanned_info = _discover_unscanned_decoder_layers(base_keys)
    expanded = _expand_unscanned_mapping_entries(raw, unscanned_info, strip_layer_sharding=True)
    if self._filter_missing_source_keys:
      if lora_keys:
        return {k: v for k, v in expanded.items() if k in lora_keys}
      return {k: v for k, v in expanded.items() if _strip_lora_suffix(k) in base_keys}
    return expanded

  def to_hf_transpose_keys(self, backend: str | None = None) -> dict[str, tuple[int, ...]]:
    """Returns parameter transpose rules for Hugging Face / vLLM rollout transfer."""
    if self.use_no_op_mappings:
      return {}
    raw = _call_mapping_member(self._mapping_source, ("to_hf_transpose_keys",), backend)
    if not raw:
      return {}
    param_shapes = _flat_param_shapes(self.param_state())
    unscanned_info = _discover_unscanned_decoder_layers(param_shapes.keys())
    return _expand_unscanned_mapping_entries(raw, unscanned_info, strip_layer_sharding=False)

  def lora_to_hf_transpose_keys(self, backend: str | None = None) -> dict[str, tuple[int, ...]]:
    """Returns LoRA transpose rules for Hugging Face / vLLM rollout transfer."""
    if self.use_no_op_mappings:
      return {}
    raw = _call_mapping_member(self._mapping_source, ("lora_to_hf_transpose_keys",), backend)
    if not raw:
      return {}
    param_shapes = _flat_param_shapes(self.param_state())
    unscanned_info = _discover_unscanned_decoder_layers(param_shapes.keys())
    return _expand_unscanned_mapping_entries(raw, unscanned_info, strip_layer_sharding=False)

  def to_hf_hook_fns(self, backend: str | None = None) -> dict[str, Callable[..., Any]]:
    """Returns hook functions applied during Hugging Face / vLLM rollout transfer."""
    if self.use_no_op_mappings:
      return {}
    raw = _call_mapping_member(self._mapping_source, ("to_hf_hook_fns",), backend)
    if not raw:
      return {}
    param_shapes = _flat_param_shapes(self.param_state())
    base_keys = {k for k in param_shapes if _strip_lora_suffix(k) == k}
    unscanned_info = _discover_unscanned_decoder_layers(base_keys)
    expanded = _expand_unscanned_mapping_entries(raw, unscanned_info, strip_layer_sharding=False)
    if self._filter_missing_source_keys:
      return {k: v for k, v in expanded.items() if k in param_shapes or _strip_lora_suffix(k) in base_keys}
    return expanded


def wrap_m3_for_tunix(
    base_model: nnx.Module,
    *,
    pad_id: int | None = None,
    use_no_op_mappings: bool = True,
    **kwargs: Any,
) -> M3TunixAdapter:
  """Convenience factory wrapping an m3 causal LM in `M3TunixAdapter`."""
  return M3TunixAdapter(
      base_model=base_model,
      pad_id=pad_id,
      use_no_op_mappings=use_no_op_mappings,
      **kwargs,
  )
