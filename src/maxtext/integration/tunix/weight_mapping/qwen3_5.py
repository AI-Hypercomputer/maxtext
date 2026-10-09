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

"""Weight mapping from MaxText's Qwen3.5 (text) model to tpu-inference's vLLM (torchax) Qwen3.5.

Unlike the dense mappings in this package, the target here is not a JAX/nnx model but the
vLLM (PyTorch) implementation that tpu-inference runs through torchax. That runner keeps
its weights in an internal, tensor-parallelism-dependent layout, so the contract used
here is vLLM's *canonical* parameter layout (what ``model.named_parameters()`` holds on
a single GPU at TP=1):

  * linear kernels are ``[out, in]``;
  * ``self_attn.qkv_proj.weight`` is ``[q | k | v]`` with each KV head once (``q`` carries
    the attention output gate, i.e. ``num_heads * 2 * head_dim`` rows);
  * ``linear_attn.in_proj_qkvz.weight`` is ``[q | k | v | z]``, ``in_proj_ba.weight`` is
    ``[b | a]`` (HF / vLLM order, not MaxText's per-key-head interleaving);
  * ``conv1d.weight`` is ``[channels, 1, kernel]``;
  * routed experts are ``w13_weight = [E, 2F, D]`` (gate rows first) and
    ``w2_weight = [E, D, F]``; the shared expert is ``gate_up_proj = [2F, D]``.

tpu-inference (``VllmModelWrapper.load_canonical_weights``) converts canonical arrays into
its internal layout, so this mapping needs no knowledge of TP size, KV-head replication
or the MoE backend. Tunix drives it from ``VllmSampler.update_params``.

Because Tunix's key mapping is one-to-one, every MaxText parameter that has to be *fused*
or *reordered* is handled in ``preprocess_src_state`` (which returns a flat dict of
canonical arrays keyed by MaxText-side names), and ``to_hf_mapping`` is then a pure
rename. Both scanned (inhomogeneous ``layers.layer_{b}`` blocks) and unscanned
(``layers_{i}``) MaxText parameter trees are accepted.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import jax
import jax.numpy as jnp


# Regexes matched against the canonical vLLM parameter names. `(\d+)` captures the layer
# index that fills the `*` of the MaxText-side key; the optional `language_model.` prefix
# covers both `Qwen3_5MoeForConditionalGeneration` and the text-only `...ForCausalLM`.
_LM = r"(?:language_model\.)?"
_LAYER = _LM + r"model\.layers\.(\d+)\."


def _tgt(rest: str) -> str:
  return _LAYER + rest


@dataclass
class QWEN3_5_VLLM_MAPPING:
  """Mapping MaxText Qwen3.5 weights to tpu-inference vLLM (torchax) Qwen3.5 weights."""

  @staticmethod
  def to_hf_mapping():
    """Maps MaxText-side keys (after `preprocess_src_state`) to canonical vLLM parameter names.

    The second tuple element is the sharding hint Tunix expects; no entry carries a
    `layer` axis because `preprocess_src_state` already unstacks scanned layers.
    """
    return {
        "base.token_embedder.embedding": (_LM + r"model\.embed_tokens\.weight", ("model", None)),
        "base.decoder.decoder_norm.scale": (_LM + r"model\.norm\.weight", (None,)),
        "base.decoder.logits_dense.kernel": (_LM + r"lm_head\.weight", ("model", None)),
        # Per-layer norms
        "base.decoder.layers.*.input_layernorm.scale": (_tgt(r"input_layernorm\.weight"), (None,)),
        "base.decoder.layers.*.post_attention_layernorm.scale": (
            _tgt(r"post_attention_layernorm\.weight"),
            (None,),
        ),
        # Full (gated) attention layers
        "base.decoder.layers.*.attention.qkv_proj": (_tgt(r"self_attn\.qkv_proj\.weight"), ("model", None)),
        "base.decoder.layers.*.attention.o_proj": (_tgt(r"self_attn\.o_proj\.weight"), (None, "model")),
        "base.decoder.layers.*.attention.query_norm.scale": (_tgt(r"self_attn\.q_norm\.weight"), (None,)),
        "base.decoder.layers.*.attention.key_norm.scale": (_tgt(r"self_attn\.k_norm\.weight"), (None,)),
        # Gated DeltaNet (linear attention) layers
        "base.decoder.layers.*.attention.in_proj_qkvz": (_tgt(r"linear_attn\.in_proj_qkvz\.weight"), ("model", None)),
        "base.decoder.layers.*.attention.in_proj_ba": (_tgt(r"linear_attn\.in_proj_ba\.weight"), ("model", None)),
        "base.decoder.layers.*.attention.conv1d": (_tgt(r"linear_attn\.conv1d\.weight"), (None, None, None)),
        "base.decoder.layers.*.attention.A_log": (_tgt(r"linear_attn\.A_log"), (None,)),
        "base.decoder.layers.*.attention.dt_bias": (_tgt(r"linear_attn\.dt_bias"), (None,)),
        "base.decoder.layers.*.attention.norm": (_tgt(r"linear_attn\.norm\.weight"), (None,)),
        "base.decoder.layers.*.attention.gdn_out_proj": (_tgt(r"linear_attn\.out_proj\.weight"), (None, "model")),
        # MoE
        "base.decoder.layers.*.mlp.gate": (_tgt(r"mlp\.gate\.weight"), (None, None)),
        "base.decoder.layers.*.mlp.experts.w13": (
            _tgt(r"mlp\.experts(?:\.routed_experts)?\.w13_weight"),
            ("expert", "model", None),
        ),
        "base.decoder.layers.*.mlp.experts.w2": (
            _tgt(r"mlp\.experts(?:\.routed_experts)?\.w2_weight"),
            ("expert", None, "model"),
        ),
        "base.decoder.layers.*.mlp.shared_expert.gate_up_proj": (
            _tgt(r"mlp\.shared_expert\.gate_up_proj\.weight"),
            ("model", None),
        ),
        "base.decoder.layers.*.mlp.shared_expert.down_proj": (
            _tgt(r"mlp\.shared_expert\.down_proj\.weight"),
            (None, "model"),
        ),
        "base.decoder.layers.*.mlp.shared_expert_gate": (_tgt(r"mlp\.shared_expert_gate\.weight"), (None, None)),
    }

  @staticmethod
  def to_hf_hook_fns():
    """All layout transformations happen in `preprocess_src_state`."""
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def preprocess_src_state(hf_config: Optional[Mapping[str, Any]] = None) -> Callable[[Any], Dict[str, jax.Array]]:
    """Returns the function that turns a MaxText state into canonical vLLM arrays.

    Args:
      hf_config: optional HF config dict (``text_config`` is honoured) used for the GDN
        head geometry. Without it the geometry is inferred from parameter shapes,
        assuming ``linear_key_head_dim == linear_value_head_dim`` (true for Qwen3.5).
    """
    return lambda state: qwen3_5_maxtext_to_vllm_canonical(state, hf_config)


# ----------------------------------------------------------------------------------- #
# Canonicalization
# ----------------------------------------------------------------------------------- #

_UNSCANNED = re.compile(r"^decoder\.layers_(\d+)\.(.+)$")
_SCANNED_BLOCK = re.compile(r"^decoder\.layers\.layer_(\d+)\.(.+)$")
_SCAN_AXIS = 1  # MaxText stacks scanned layer parameters on axis 1.


def _flatten(state: Any) -> Dict[str, jax.Array]:
  """Flattens an nnx.State / dict / pytree of MaxText params to {dotted_key: array}."""
  if hasattr(state, "flat_state"):
    items = state.flat_state()
  elif isinstance(state, Mapping) and all(isinstance(k, str) for k in state):
    items = state.items()
  else:
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    items = [(tuple(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k))) for k in path), v) for path, v in leaves]
  out = {}
  for key, val in items:
    key = key if isinstance(key, str) else ".".join(str(k) for k in key)
    val = getattr(val, "value", val)
    if val is None or not hasattr(val, "shape"):
      continue
    out[key] = val
  return out


def _normalize_key(key: str) -> Optional[str]:
  """Strips wrapper prefixes (`base.`, `params.params.`, ...) down to the MaxText param path."""
  for anchor in ("decoder.", "token_embedder."):
    idx = key.find(anchor)
    if idx >= 0:
      return key[idx:]
  return None


def _gdn_geometry(hf_config: Optional[Mapping[str, Any]], layer: Mapping[str, jax.Array]):
  """(H_k, H_v, D_k, D_v) of the Gated DeltaNet block."""
  if hf_config:
    cfg = hf_config.get("text_config", hf_config)
    keys = ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    if all(k in cfg for k in keys):
      return tuple(int(cfg[k]) for k in keys)
  h_v = layer["attention.A_log"].shape[0]
  d_v = layer["attention.norm.rms_norm.scale"].shape[0]
  qkvz_out = layer["attention.in_proj_qkvz.kernel"].shape[1]
  d_k = d_v
  h_k = (qkvz_out - 2 * h_v * d_v) // (2 * d_k)
  assert h_k * d_k * 2 + 2 * h_v * d_v == qkvz_out, (
      f"cannot infer GDN geometry from in_proj_qkvz {layer['attention.in_proj_qkvz.kernel'].shape}"
  )
  return h_k, h_v, d_k, d_v


def _canonicalize_layer(layer: Mapping[str, jax.Array], hf_config) -> Dict[str, jax.Array]:
  """MaxText per-layer params (keys relative to the layer) -> canonical vLLM arrays."""
  out: Dict[str, jax.Array] = {}

  out["input_layernorm.scale"] = layer["input_layernorm.scale"]
  out["post_attention_layernorm.scale"] = layer["post_attention_layernorm.scale"]

  if "attention.in_proj_qkvz.kernel" in layer:
    # --- Gated DeltaNet. MaxText stores in_proj_qkvz per key head as
    # [q_h | k_h | v_h(0..V/K) | z_h(0..V/K)] and in_proj_ba as [b_h | a_h]; vLLM wants
    # [Q | K | V | Z] and [B | A] (see QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN in
    # checkpoint_conversion/utils/param_mapping.py).
    h_k, h_v, d_k, d_v = _gdn_geometry(hf_config, layer)
    v_per_k = h_v // h_k
    qkvz = layer["attention.in_proj_qkvz.kernel"]  # [D, H_k * block]
    d_model = qkvz.shape[0]
    block = 2 * d_k + 2 * v_per_k * d_v
    t = jnp.transpose(qkvz).reshape(h_k, block, d_model)
    q = t[:, :d_k].reshape(h_k * d_k, d_model)
    k = t[:, d_k : 2 * d_k].reshape(h_k * d_k, d_model)
    v = t[:, 2 * d_k : 2 * d_k + v_per_k * d_v].reshape(h_v * d_v, d_model)
    z = t[:, 2 * d_k + v_per_k * d_v :].reshape(h_v * d_v, d_model)
    out["attention.in_proj_qkvz"] = jnp.concatenate([q, k, v, z], axis=0)

    ba = jnp.transpose(layer["attention.in_proj_ba.kernel"]).reshape(h_k, 2 * v_per_k, d_model)
    b = ba[:, :v_per_k].reshape(h_v, d_model)
    a = ba[:, v_per_k:].reshape(h_v, d_model)
    out["attention.in_proj_ba"] = jnp.concatenate([b, a], axis=0)

    out["attention.conv1d"] = jnp.transpose(layer["attention.conv1d.kernel"], (2, 1, 0))  # [K,1,C] -> [C,1,K]
    out["attention.A_log"] = layer["attention.A_log"]
    out["attention.dt_bias"] = layer["attention.dt_bias"]
    out["attention.norm"] = layer["attention.norm.rms_norm.scale"]
    out["attention.gdn_out_proj"] = jnp.transpose(layer["attention.out_proj.kernel"])
  else:
    # --- Full attention with output gate: MaxText query kernel is [D, H, 2*Dh] (q and
    # gate per head, HF order), key/value are [D, H_kv, Dh].
    q = layer["attention.attention.query.kernel"]
    k = layer["attention.attention.key.kernel"]
    v = layer["attention.attention.value.kernel"]
    d_model = q.shape[0]
    qkv = [jnp.transpose(x.reshape(d_model, -1)) for x in (q, k, v)]
    out["attention.qkv_proj"] = jnp.concatenate(qkv, axis=0)
    out["attention.o_proj"] = jnp.transpose(layer["attention.attention.out.kernel"])
    out["attention.query_norm.scale"] = layer["attention.attention.query_norm.scale"]
    out["attention.key_norm.scale"] = layer["attention.attention.key_norm.scale"]

  # --- MoE: MaxText experts are [E, D, F] / [E, F, D]; vLLM wants [E, 2F, D] / [E, D, F].
  out["mlp.gate"] = jnp.transpose(layer["mlp.routed_experts.gate.kernel"])
  w13 = jnp.concatenate([layer["mlp.routed_experts.wi_0"], layer["mlp.routed_experts.wi_1"]], axis=-1)
  out["mlp.experts.w13"] = jnp.swapaxes(w13, 1, 2)
  out["mlp.experts.w2"] = jnp.swapaxes(layer["mlp.routed_experts.wo"], 1, 2)
  gate_up = jnp.concatenate([layer["mlp.shared_expert.wi_0.kernel"], layer["mlp.shared_expert.wi_1.kernel"]], axis=-1)
  out["mlp.shared_expert.gate_up_proj"] = jnp.transpose(gate_up)
  out["mlp.shared_expert.down_proj"] = jnp.transpose(layer["mlp.shared_expert.wo.kernel"])
  out["mlp.shared_expert_gate"] = jnp.transpose(layer["mlp.shared_expert_gate.kernel"])
  return out


def _mesh_of(arrays) -> Optional[jax.sharding.Mesh]:
  """The mesh the (jax) source arrays live on, or None for host/numpy inputs."""
  for arr in arrays:
    sharding = getattr(arr, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
      return sharding.mesh
  return None


def _even_sharding(mesh: jax.sharding.Mesh, shape) -> jax.sharding.NamedSharding:
  """Shards the first dimension divisible by the device count over the whole mesh (else replicates)."""
  n = mesh.size
  spec = [None] * len(shape)
  for i, dim in enumerate(shape):
    if dim % n == 0 and dim >= n:
      spec[i] = tuple(mesh.axis_names)
      break
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))


def _materialize(fn: Callable[[Any], Dict[str, jax.Array]], args: Any) -> Dict[str, jax.Array]:
  """Runs `fn(args)` under jit with every output evenly sharded on the source mesh.

  Eager execution lets XLA pick the output shardings, and reshapes/transposes of
  fsdp-sharded kernels readily come back replicated -- a full canonical copy per
  device does not fit next to the trainer and sampler weights.
  """
  mesh = _mesh_of(jax.tree_util.tree_leaves(args))
  if mesh is None:
    return fn(args)
  shapes = jax.eval_shape(fn, args)
  out_shardings = {k: _even_sharding(mesh, v.shape) for k, v in shapes.items()}
  return jax.jit(fn, out_shardings=out_shardings)(args)


def qwen3_5_maxtext_to_vllm_canonical(state: Any, hf_config: Optional[Mapping[str, Any]] = None) -> Dict[str, jax.Array]:
  """Converts a MaxText Qwen3.5 param tree into canonical vLLM arrays.

  Returns a flat dict keyed by the MaxText-side names used in `to_hf_mapping`
  (``base.decoder.layers.{i}....``), one entry per (unstacked) layer.
  """
  flat = _flatten(state)
  layers: Dict[int, Dict[str, jax.Array]] = {}
  globals_: Dict[str, jax.Array] = {}
  scanned_blocks: Dict[int, Dict[str, jax.Array]] = {}

  for key, val in flat.items():
    norm = _normalize_key(key)
    if norm is None:
      continue
    if m := _UNSCANNED.match(norm):
      layers.setdefault(int(m.group(1)), {})[m.group(2)] = val
    elif m := _SCANNED_BLOCK.match(norm):
      scanned_blocks.setdefault(int(m.group(1)), {})[m.group(2)] = val
    elif norm.startswith("decoder.layers"):
      raise NotImplementedError(f"unexpected MaxText layer parameter layout: {key}")
    else:
      globals_[norm] = val

  if scanned_blocks:
    # Inhomogeneous scan: block b holds layers b, b+n, b+2n, ... stacked on _SCAN_AXIS.
    n_blocks = max(scanned_blocks) + 1
    for b, params in scanned_blocks.items():
      n_rep = next(iter(params.values())).shape[_SCAN_AXIS]
      for j in range(n_rep):
        layers[b + n_blocks * j] = {k: jnp.take(v, j, axis=_SCAN_AXIS) for k, v in params.items()}

  out: Dict[str, jax.Array] = {}
  canonicalize = lambda params: _canonicalize_layer(params, hf_config)  # pylint: disable=unnecessary-lambda-assignment
  for i, params in sorted(layers.items()):
    for name, arr in _materialize(canonicalize, params).items():
      out[f"base.decoder.layers.{i}.{name}"] = arr

  out["base.token_embedder.embedding"] = globals_["token_embedder.embedding"]
  out["base.decoder.decoder_norm.scale"] = globals_["decoder.decoder_norm.scale"]
  out.update(
      _materialize(
          lambda g: {"base.decoder.logits_dense.kernel": jnp.transpose(g["decoder.logits_dense.kernel"])},
          {"decoder.logits_dense.kernel": globals_["decoder.logits_dense.kernel"]},
      )
  )
  return out
