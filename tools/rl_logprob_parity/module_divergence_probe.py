#!/usr/bin/env python3
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
"""Layer-by-layer, module-by-module value divergence probe between vLLM Sampler and MaxText Trainer.

Provides:
  1. `ModuleProbeTap`: Non-invasive `jax.debug.callback` context manager that captures
     per-module intermediate tensors in-situ during live Sampler and Trainer forward passes.
  2. `run_isolated_trainer_replay`: Teacher-forced single-module replay feeding the Sampler's
     exact captured inputs (and forced router indices) into each Trainer submodule in isolation.
  3. `compare_module_probes`: Dual divergence analysis reporting both Isolated (single-module)
     and Cumulative (end-to-end) value divergence, router top-k agreement, module-family
     attribution, and root-cause bottleneck ranking.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import contextlib
import inspect
import json
import os
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


LAYER_MODULES = (
    "layer_in",
    "input_layernorm",
    "attn.in_proj_qkvz",
    "attn.in_proj_ba",
    "attn.query",
    "attn.key",
    "attn.value",
    "attn.out",
    "post_attn_residual",
    "post_attention_layernorm",
    "mlp.gate_logits",
    "mlp.router_topk",
    "mlp.selected_experts",
    "mlp.routed_experts",
    "mlp.shared_expert",
    "mlp.shared_expert_gate",
    "mlp.out",
    "layer_out",
)

BOUNDARY_MODULES = (
    "token_embedder",
    "decoder_norm",
    "logits_head",
    "logits_summary",
    "logits_top1",
)


def _pick_span_positions(lo: int, hi: int, budget: int, anchors: Sequence[int]) -> set[int]:
  """Selects up to `budget` indices in `[lo, hi)` prioritizing `anchors` then uniform spacing."""
  if budget <= 0 or hi <= lo:
    return set()
  result = set(sorted({int(a) for a in anchors if lo <= a < hi})[:budget])
  if len(result) < budget:
    for x in np.linspace(lo, hi - 1, num=budget * 2, dtype=np.int32):
      if len(result) >= budget:
        break
      result.add(int(x))
  return result


def select_probe_positions(prompt_len: int, gen_len: int, max_tokens: int = 64) -> np.ndarray:
  """Selects a canonical, sorted 1D int32 array of token indices in `[0, prompt_len + gen_len)`."""
  p_len, g_len = max(0, int(prompt_len)), max(0, int(gen_len))
  total = p_len + g_len
  if total == 0:
    return np.zeros((0,), dtype=np.int32)
  if max_tokens <= 0 or total <= max_tokens:
    return np.arange(total, dtype=np.int32)

  if g_len == 0 or p_len == 0:
    p_budget, g_budget = (max_tokens, 0) if g_len == 0 else (0, max_tokens)
  else:
    g_budget = min(g_len, max(4, max_tokens // 2))
    p_budget = min(p_len, max_tokens - g_budget)
    g_budget = min(g_len, max_tokens - p_budget)

  p_anchors = (0, 1, 2, 3, 255, 256, 511, 512, 2047, 2048, p_len - 4, p_len - 3, p_len - 2, p_len - 1)
  g_anchors = (p_len, p_len + 1, p_len + 2, p_len + 3, p_len + 255, p_len + 256, total - 4, total - 3, total - 2, total - 1)
  chosen = _pick_span_positions(0, p_len, p_budget, p_anchors) | _pick_span_positions(p_len, total, g_budget, g_anchors)
  return np.array(sorted(chosen)[:max_tokens], dtype=np.int32)


def parse_probe_layers(spec: str | Sequence[int] | None) -> set[int] | None:
  """Parses a `--probe-layers` specification (`'all'`, `'0,1,2-4,39'`, or a sequence of ints)."""
  if spec is None:
    return None
  if not isinstance(spec, str):
    return {int(i) for i in spec}
  s = spec.strip().lower()
  if not s or s == "all":
    return None
  out: set[int] = set()
  for part in (p.strip() for p in s.split(",")):
    if "-" in part:
      lo_s, hi_s = part.split("-", 1)
      out.update(range(int(lo_s), int(hi_s) + 1))
    elif part.isdigit():
      out.add(int(part))
  return out


# --------------------------------------------------------- Decoder Discovery & Layer Slicing


def discover_decoder_structure(model: Any) -> tuple[Any, int, bool, int, list[tuple[int, Any]]]:
  """Returns `(decoder, num_layers, is_scanned, cycle_interval, sublayers)`."""
  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  cfg = getattr(decoder, "config", None)
  cfg_num_layers = int(getattr(cfg, "num_decoder_layers", 0) or 0)

  unscanned = [
      (i, getattr(decoder, f"layers_{i}"))
      for i in range(max(cfg_num_layers, 128))
      if getattr(decoder, f"layers_{i}", None) is not None
  ]
  if unscanned:
    return decoder, (cfg_num_layers or len(unscanned)), False, 1, unscanned

  scanned = getattr(decoder, "layers", None)
  if scanned is not None:
    cycle_len = int(getattr(cfg, "inhomogeneous_layer_cycle_interval", 0) or 0)
    sublayers = [
        (i, getattr(scanned, f"layer_{i}"))
        for i in range(max(cycle_len, 16))
        if getattr(scanned, f"layer_{i}", None) is not None
    ]
    if not sublayers:
      local_layers = [lyr for lyr in (getattr(scanned, "local_layers", None) or ()) if lyr is not None]
      if getattr(scanned, "global_layer", None) is not None:
        local_layers.append(scanned.global_layer)
      sublayers = list(enumerate(local_layers))
    if sublayers:
      c_interval = len(sublayers)
      return decoder, (cfg_num_layers or c_interval), True, c_interval, sublayers

  return decoder, max(cfg_num_layers, 1), False, 1, []


def get_layer_for_index(model: Any, layer_idx: int) -> Any | None:
  """Returns an unscanned single-layer module for global `layer_idx` (slicing scan axis if needed)."""
  decoder, num_layers, is_scanned, cycle_interval, sublayers = discover_decoder_structure(model)
  if not is_scanned:
    return getattr(decoder, f"layers_{layer_idx}", None)
  sublayer = dict(sublayers).get(layer_idx % max(1, cycle_interval))
  scan_length = max(1, num_layers // max(1, cycle_interval))
  if sublayer is None or scan_length <= 1:
    return sublayer

  scan_idx = layer_idx // max(1, cycle_interval)
  scan_axis = int(getattr(getattr(decoder, "config", None), "param_scan_axis", 0) or 0)
  graphdef, state = nnx.split(sublayer)

  def _slice_leaf(x):
    if hasattr(x, "shape") and x.ndim >= 2:
      if scan_axis < x.ndim and x.shape[scan_axis] == scan_length:
        return jnp.take(x, scan_idx, axis=scan_axis)
      if x.shape[0] == scan_length:
        return jnp.take(x, scan_idx, axis=0)
    return x

  return nnx.merge(graphdef, jax.tree.map(_slice_leaf, state))


def _resolve_top_k(routed: Any, decoder_cfg: Any = None) -> int:
  routed_cfg = getattr(routed, "config", None)
  val = getattr(routed, "num_experts_per_tok", getattr(routed_cfg, "num_experts_per_tok", None))
  return int(val or getattr(decoder_cfg, "num_experts_per_tok", 8) or 8)


def _iter_layer_submodules(layer_self: Any):
  """Yields `(submodule_obj, probe_name)` for all tapped submodules within a decoder layer."""
  for attr in ("input_layernorm", "post_attention_layernorm"):
    if getattr(layer_self, attr, None) is not None:
      yield getattr(layer_self, attr), attr

  attn = getattr(layer_self, "attention", None)
  if attn is not None:
    yield attn, "attn.out"
    for attr in ("in_proj_qkvz", "in_proj_ba"):
      if getattr(attn, attr, None) is not None:
        yield getattr(attn, attr), f"attn.{attr}"
    attn_inner = getattr(attn, "attention", attn)
    for attr in ("query", "key", "value"):
      if getattr(attn_inner, attr, None) is not None:
        yield getattr(attn_inner, attr), f"attn.{attr}"

  mlp = getattr(layer_self, "mlp", None)
  if mlp is not None:
    yield mlp, "mlp.out"
    routed = getattr(mlp, "routed_experts", None)
    if routed is not None:
      yield routed, "mlp.routed_experts"
      if getattr(routed, "gate", None) is not None:
        yield getattr(routed, "gate"), "mlp.gate_logits"
    for attr in ("shared_expert", "shared_expert_gate"):
      if getattr(mlp, attr, None) is not None:
        yield getattr(mlp, attr), f"mlp.{attr}"


# --------------------------------------------------------- Mesh & Boundary Output Helpers


def _to_mesh_array(arr: Any, dtype: Any, mesh: Any = None) -> jax.Array:
  out = jnp.asarray(arr, dtype=dtype)
  if mesh is not None and getattr(getattr(mesh, "devices", None), "size", 1) > 1:
    with contextlib.suppress(Exception):
      out = jax.device_put(out, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
  return out


def _try_call_primary(fn: Any, *args, **kwargs) -> jax.Array | None:
  """Calls `fn` if callable, unwrapping `(primary, ...)` tuples and suppressing runtime errors."""
  if not callable(fn):
    return None
  with contextlib.suppress(Exception):
    res = fn(*args, **kwargs)
    return res[0] if isinstance(res, tuple) else res
  return None


@contextlib.contextmanager
def _model_mesh_context(decoder: Any, base: Any, cfg: Any):
  mesh = getattr(decoder, "mesh", getattr(base, "mesh", None))
  rules = getattr(cfg, "logical_axis_rules", None) if cfg is not None else None
  with (
      jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext(),
      nnx.logical_axis_rules(rules) if rules else contextlib.nullcontext(),
  ):
    yield mesh


def _compute_logits_summary_and_head(
    logits_3d: jax.Array,
    target_tokens_2d: jax.Array | np.ndarray | None = None,
    temperature: float = 1.0,
    prompt_len: int | None = None,
    probe_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes `(logits_head, logits_summary, logits_top1)` for `[1, N_probe, V]` logits."""
  logits_f32 = jnp.asarray(logits_3d, dtype=jnp.float32)[0]
  n_tok, vocab_size = logits_f32.shape
  if temperature not in (0.0, 1.0):
    if prompt_len is not None and probe_positions is not None:
      pos_arr = jnp.asarray(probe_positions, dtype=jnp.int32)
      scaled = logits_f32 / jnp.where(pos_arr >= (prompt_len - 1), jnp.float32(temperature), jnp.float32(1.0))[:, None]
    else:
      scaled = logits_f32 / jnp.float32(temperature)
  else:
    scaled = logits_f32

  logits_head = np.asarray(scaled[:, : min(256, int(vocab_size))], dtype=np.float32)
  lse = jax.nn.logsumexp(scaled, axis=-1)
  max_logit = jnp.max(scaled, axis=-1)
  top1_id = np.asarray(jnp.argmax(scaled, axis=-1), dtype=np.int32)

  if target_tokens_2d is not None:
    tgt_ids = jnp.clip(jnp.asarray(target_tokens_2d, dtype=jnp.int32).reshape(-1)[:n_tok], 0, vocab_size - 1)
    tgt_logit = jnp.take_along_axis(scaled, tgt_ids[:, None], axis=-1)[:, 0]
  else:
    tgt_logit = max_logit
  summary = np.stack([np.asarray(x, dtype=np.float32) for x in (tgt_logit, lse, tgt_logit - lse, max_logit)], axis=-1)
  return logits_head, summary, top1_id


def _forward_logits_head(decoder: Any, base: Any, cfg: Any, embedder: Any, norm_out: jax.Array, raw_last_h: jax.Array | None = None) -> jax.Array | None:
  """Runs the decoder output projection (`logits_dense` or `apply_output_head`) to produce 3D logits."""
  logits_dense_fn = getattr(decoder, "logits_dense", None)
  apply_head_fn = getattr(decoder, "apply_output_head", None)
  if callable(logits_dense_fn) and not getattr(cfg, "logits_via_embedding", False):
    logits_3d = logits_dense_fn(norm_out)
    return logits_3d.astype(jnp.float32) if getattr(cfg, "cast_logits_to_fp32", True) else logits_3d
  if callable(apply_head_fn):
    model_mode = getattr(base, "model_mode", "train")
    if "normalize_y" in inspect.signature(apply_head_fn).parameters:
      return apply_head_fn(embedder, norm_out, True, model_mode, normalize_y=False)
    if raw_last_h is not None:
      return apply_head_fn(embedder, raw_last_h, True, model_mode)
  return None


def capture_output_head_probes(
    model: Any,
    probes_dict: dict[str, np.ndarray],
    *,
    num_layers: int,
    target_next_tokens: np.ndarray | None = None,
    prompt_tokens_probe: np.ndarray | None = None,
    temperature: float = 1.0,
    prompt_len: int | None = None,
    probe_positions: np.ndarray | None = None,
    input_probes: dict[str, np.ndarray] | None = None,
    b_pad: int = 1,
) -> None:
  """Populates `token_embedder`, `decoder_norm`, `logits_head`, `logits_summary`, `logits_top1` in `probes_dict`."""
  src = probes_dict if input_probes is None else input_probes
  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  embedder = getattr(base, "token_embedder", getattr(decoder, "token_embedder", None))
  cfg = getattr(decoder, "config", None)
  model_dtype = getattr(cfg, "dtype", jnp.bfloat16) if cfg is not None else jnp.bfloat16

  with _model_mesh_context(decoder, base, cfg) as mesh:
    if "token_embedder" not in probes_dict:
      if prompt_tokens_probe is not None and callable(embedder):
        tok_in = _to_mesh_array(np.repeat(np.asarray(prompt_tokens_probe, dtype=np.int32)[None, :], max(1, b_pad), axis=0), jnp.int32, mesh)
        emb_0 = _try_call_primary(embedder, tok_in)
        if emb_0 is not None:
          probes_dict["token_embedder"] = np.asarray(emb_0[0], dtype=np.float32)
      for fallback_key in ("token_embedder", "layer_0.layer_in"):
        if "token_embedder" not in probes_dict and fallback_key in src:
          probes_dict["token_embedder"] = np.array(src[fallback_key], dtype=np.float32, copy=True)

    last_key = f"layer_{num_layers - 1}.layer_out"
    last_h = _safe_input_3d(src[last_key], model_dtype, mesh, b_pad) if last_key in src else None
    decoder_norm_fn = getattr(decoder, "decoder_norm", None)
    norm_out = None
    if last_h is not None:
      if callable(decoder_norm_fn):
        norm_out = _try_call_primary(decoder_norm_fn, last_h)
        if norm_out is not None:
          probes_dict["decoder_norm"] = np.asarray(norm_out[0], dtype=np.float32)
      else:
        norm_out = last_h

    head_in = _safe_input_3d(src["decoder_norm"], model_dtype, mesh, b_pad) if (input_probes is not None and "decoder_norm" in src) else norm_out
    if head_in is not None:
      logits_3d = _forward_logits_head(decoder, base, cfg, embedder, head_in, raw_last_h=last_h)
      if logits_3d is not None:
        l_head, l_sum, l_top1 = _compute_logits_summary_and_head(logits_3d[:1], target_next_tokens, temperature, prompt_len, probe_positions)
        probes_dict["logits_head"], probes_dict["logits_summary"], probes_dict["logits_top1"] = l_head, l_sum, l_top1


# --------------------------------------------------------- In-Situ Module Probe Tap


class ModuleProbeTap:
  """Context manager that hooks decoder layers and submodules via `jax.debug.callback`."""

  def __init__(
      self,
      model: Any,
      *,
      role: str,
      probe_positions: np.ndarray,
      prompt_len: int,
      gen_len: int,
      packed_seq0_row: int = 0,
      packed_seq0_positions: np.ndarray | None = None,
      trainer_microbatch_idx: int = 0,
      layer_indices: set[int] | Sequence[int] | None = None,
      recording_enabled: bool = True,
  ):
    self.model = model
    self.role = role
    self.probe_positions = np.asarray(probe_positions, dtype=np.int32)
    self.prompt_len = int(prompt_len)
    self.gen_len = int(gen_len)
    self.total_seq_len = self.prompt_len + self.gen_len
    self.packed_seq0_row = int(packed_seq0_row)
    self.packed_seq0_positions = (
        np.asarray(packed_seq0_positions, dtype=np.int32) if packed_seq0_positions is not None else None
    )
    self.trainer_microbatch_idx = int(trainer_microbatch_idx)
    self.layer_indices: set[int] | None = {int(i) for i in layer_indices} if layer_indices is not None else None
    self.recording_enabled = bool(recording_enabled)

    self.decoder, self.num_layers, self.is_scanned, self.cycle_interval, self.sublayers = (
        discover_decoder_structure(model)
    )
    self.scan_length = max(1, self.num_layers // max(1, self.cycle_interval)) if self.is_scanned else 1
    self._trainer_gather_cols = (
        np.asarray(self.packed_seq0_positions[self.probe_positions], dtype=np.int32)
        if (self.packed_seq0_positions is not None and self.probe_positions.size > 0)
        else self.probe_positions
    )

    self._buffers: dict[str, np.ndarray] = {}
    self._filled_slots: dict[str, np.ndarray] = {}
    self._scan_call_counts: dict[tuple[int, str, int], int] = collections.defaultdict(int)
    self._trainer_layer_call_counts: dict[tuple[int, str], int] = collections.defaultdict(int)
    self._patched_classes: list[tuple[type, str, Any]] = []
    self._active_batch_shape: tuple[int, int] = (1, 1)
    self._active_forced_experts: Any = None
    self._active_seq_bounds: tuple[Any, Any, Any, Any] = (
        jnp.int32(0),
        jnp.int32(self.total_seq_len),
        jnp.int32(0),
        jnp.int32(self.total_seq_len),
    )

  def start_recording(self, reset: bool = True) -> None:
    if reset:
      self._buffers.clear()
      self._filled_slots.clear()
      self._scan_call_counts.clear()
      self._trainer_layer_call_counts.clear()
    self.recording_enabled = True

  def stop_recording(self) -> None:
    self.recording_enabled = False

  def _resolve_global_layer(self, sublayer_idx: int, module_name: str, seq_start: int) -> int:
    if not self.is_scanned:
      return int(sublayer_idx)
    key = (int(sublayer_idx), module_name, int(seq_start))
    step = self._scan_call_counts[key] % self.scan_length
    self._scan_call_counts[key] += 1
    return step * self.cycle_interval + int(sublayer_idx)

  def _host_record_trainer(self, arr_np: np.ndarray, sublayer_idx_np: np.ndarray, module_name: str) -> None:
    if not self.recording_enabled or self.probe_positions.size == 0:
      return
    layer_idx = self._resolve_global_layer(int(sublayer_idx_np), module_name, 0)
    call_idx = self._trainer_layer_call_counts[(layer_idx, module_name)]
    self._trainer_layer_call_counts[(layer_idx, module_name)] = call_idx + 1
    if call_idx != self.trainer_microbatch_idx:
      return
    if self.layer_indices is not None and layer_idx not in self.layer_indices:
      return
    out_dtype = np.int32 if np.issubdtype(arr_np.dtype, np.integer) else np.float32
    self._buffers[f"layer_{layer_idx}.{module_name}"] = np.asarray(arr_np, dtype=out_dtype).copy()

  def _host_record_sampler(
      self,
      arr_np: np.ndarray,
      sublayer_idx_np: np.ndarray,
      q_start_np: np.ndarray,
      q_end_np: np.ndarray,
      seq_start_np: np.ndarray,
      seq_end_np: np.ndarray,
      max_rows: int,
      module_name: str,
  ) -> None:
    if not self.recording_enabled or self.probe_positions.size == 0:
      return
    q_start, q_end, seq_start, seq_end = int(q_start_np), int(q_end_np), int(seq_start_np), int(seq_end_np)
    if q_end <= q_start or seq_end <= seq_start:
      return
    layer_idx = self._resolve_global_layer(int(sublayer_idx_np), module_name, seq_start)
    if self.layer_indices is not None and layer_idx not in self.layer_indices:
      return
    key = f"layer_{layer_idx}.{module_name}"
    out_dtype = np.int32 if np.issubdtype(arr_np.dtype, np.integer) else np.float32
    buf = self._buffers.get(key)
    if buf is None:
      fill_val = -1 if out_dtype == np.int32 else np.nan
      buf = np.full((len(self.probe_positions), *arr_np.shape[1:]), fill_val, dtype=out_dtype)
      self._buffers[key] = buf
    filled = self._filled_slots.setdefault(key, np.zeros(len(self.probe_positions), dtype=bool))

    for slot, logical_pos in enumerate(self.probe_positions.tolist()):
      if not filled[slot] and seq_start <= logical_pos < seq_end and 0 <= q_start + (logical_pos - seq_start) < max_rows:
        buf[slot] = np.asarray(arr_np[slot], dtype=out_dtype)
        filled[slot] = True

  def _normalize_3d(self, x: jax.Array) -> jax.Array:
    b_dim, s_dim = self._active_batch_shape
    if x.ndim == 2 and x.shape[0] == b_dim * s_dim:
      return jnp.reshape(x, (b_dim, s_dim, x.shape[1]))
    if x.ndim == 2:
      return x[:, None, :]
    return jnp.reshape(x, (x.shape[0], x.shape[1], -1)) if x.ndim > 3 else x

  def _tap(self, x: jax.Array | None, sublayer_idx: int, module_name: str) -> None:
    if x is None or self.probe_positions.size == 0:
      return
    x3 = self._normalize_3d(jnp.asarray(x))
    out_jnp_dtype = jnp.int32 if jnp.issubdtype(x3.dtype, jnp.integer) else jnp.float32
    sub_idx_arr = jnp.int32(sublayer_idx)

    if self.role == "trainer":
      row_idx = min(self.packed_seq0_row, max(0, x3.shape[0] - 1))
      cols = jnp.clip(jnp.asarray(self._trainer_gather_cols, dtype=jnp.int32), 0, max(0, x3.shape[1] - 1))
      sliced = jnp.take(x3[row_idx], cols, axis=0).astype(out_jnp_dtype)
      jax.debug.callback(
          lambda a, s, _m=module_name: self._host_record_trainer(a, s, _m),
          sliced,
          sub_idx_arr,
          ordered=self.is_scanned,
      )
      return

    q_start, q_end, seq_start, seq_end = self._active_seq_bounds
    seq0_2d = x3[0] if (x3.shape[1] > 1 and x3.shape[0] >= 1) else jnp.reshape(x3[:, 0], (x3.shape[0], -1))
    max_rows_int = max(1, int(seq0_2d.shape[0]))
    probe_pos_jnp = jnp.asarray(self.probe_positions, dtype=jnp.int32)
    row_indices = jnp.clip(q_start + (probe_pos_jnp - seq_start), 0, max_rows_int - 1)
    sliced_sa = jnp.take(seq0_2d, row_indices, axis=0).astype(out_jnp_dtype)
    should_emit = (q_end > q_start) & jnp.any((probe_pos_jnp >= seq_start) & (probe_pos_jnp < seq_end))

    def _do_emit(_):
      jax.debug.callback(
          lambda a, s, qs, qe, ss, se, _mr=max_rows_int, _m=module_name: self._host_record_sampler(
              a, s, qs, qe, ss, se, _mr, _m
          ),
          sliced_sa,
          sub_idx_arr,
          q_start,
          q_end,
          seq_start,
          seq_end,
          ordered=self.is_scanned,
      )
      return jnp.int32(0)

    jax.lax.cond(should_emit, _do_emit, lambda _: jnp.int32(0), operand=None)

  def _extract_sampler_seq_bounds(
      self, inputs: jax.Array, attention_metadata: Any
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    meta = (
        next(iter(attention_metadata.values()))
        if isinstance(attention_metadata, dict) and attention_metadata
        else attention_metadata
    )
    q_loc = getattr(meta, "query_start_loc", None) if meta is not None else None
    s_lens = getattr(meta, "seq_lens", None) if meta is not None else None
    if q_loc is not None and s_lens is not None:
      q_loc_1d = jnp.reshape(jnp.asarray(q_loc, dtype=jnp.int32), (-1,))
      s_lens_1d = jnp.reshape(jnp.asarray(s_lens, dtype=jnp.int32), (-1,))
      q_start = q_loc_1d[0]
      q_end = q_loc_1d[1] if q_loc_1d.shape[0] > 1 else jnp.int32(inputs.shape[0])
      seq_end = s_lens_1d[0]
      return q_start, q_end, seq_end - (q_end - q_start), seq_end
    n_tok = jnp.int32(inputs.shape[1] if inputs.ndim >= 3 and inputs.shape[1] > 1 else inputs.shape[0])
    return jnp.int32(0), n_tok, jnp.int32(0), n_tok

  def _patch_method(self, cls: type, method_name: str, wrapper_factory: Any) -> None:
    if any(c is cls and m == method_name for c, m, _ in self._patched_classes):
      return
    orig = getattr(cls, method_name, None)
    if orig is not None:
      self._patched_classes.append((cls, method_name, orig))
      setattr(cls, method_name, wrapper_factory(orig))

  def _tag_child_modules(self, layer_self: Any, sub_idx: int) -> list[tuple[Any, str]]:
    tagged: list[tuple[Any, str]] = []
    cfg = getattr(self.decoder, "config", None)
    routed = getattr(getattr(layer_self, "mlp", None), "routed_experts", None)
    for obj, name in _iter_layer_submodules(layer_self):
      with contextlib.suppress(Exception):
        object.__setattr__(obj, "_probe_tag", (sub_idx, name))
        tagged.append((obj, "_probe_tag"))
      if name == "mlp.gate_logits" and routed is not None:
        with contextlib.suppress(Exception):
          object.__setattr__(obj, "_probe_top_k", _resolve_top_k(routed, cfg))
          tagged.append((obj, "_probe_top_k"))
    return tagged

  def __enter__(self) -> ModuleProbeTap:
    tap = self
    layer_classes = {type(lyr) for _, lyr in self.sublayers}
    submodule_classes = {type(sub_obj) for _, lyr in self.sublayers for sub_obj, _ in _iter_layer_submodules(lyr)}

    def _make_submodule_wrapper(orig_call):
      def _wrapped_sub(mod_self, *args, **kwargs):
        tag = getattr(mod_self, "_probe_tag", None)
        if tag is None:
          return orig_call(mod_self, *args, **kwargs)
        sub_idx, mod_name = tag
        if mod_name == "post_attention_layernorm" and args:
          tap._tap(args[0], sub_idx, "post_attn_residual")
        out = orig_call(mod_self, *args, **kwargs)
        primary = out[0] if isinstance(out, tuple) else out
        if mod_name == "mlp.gate_logits":
          tap._tap(primary, sub_idx, "mlp.gate_logits")
          k_top = min(int(getattr(mod_self, "_probe_top_k", 8) or 8), int(primary.shape[-1]))
          _, router_topk = jax.lax.top_k(jax.nn.softmax(primary.astype(jnp.float32), axis=-1), k_top)
          tap._tap(router_topk.astype(jnp.int32), sub_idx, "mlp.router_topk")
          if tap._active_forced_experts is None:
            tap._tap(router_topk.astype(jnp.int32), sub_idx, "mlp.selected_experts")
        elif mod_name == "mlp.routed_experts":
          if kwargs.get("forced_routed_experts") is not None:
            tap._tap(jnp.asarray(kwargs["forced_routed_experts"], dtype=jnp.int32), sub_idx, "mlp.selected_experts")
          tap._tap(primary, sub_idx, "mlp.routed_experts")
        elif mod_name == "mlp.shared_expert_gate":
          tap._tap(jax.nn.sigmoid(primary.astype(jnp.float32)), sub_idx, "mlp.shared_expert_gate")
        else:
          tap._tap(primary, sub_idx, mod_name)
        return out

      return _wrapped_sub

    for sub_cls in submodule_classes - layer_classes:
      self._patch_method(sub_cls, "__call__", _make_submodule_wrapper)

    def _make_layer_wrapper(orig_layer_call):
      param_names = list(inspect.signature(orig_layer_call).parameters.keys())[1:]

      def _wrapped_layer(layer_self, *args, **kwargs):
        bound = {param_names[idx]: val for idx, val in enumerate(args) if idx < len(param_names)}
        bound.update(kwargs)

        raw_in = bound.get("inputs", args[0] if args else None)
        x_in = raw_in[0] if isinstance(raw_in, tuple) else raw_in
        sub_idx = int(getattr(layer_self, "layer_idx", 0) or 0)
        if x_in is not None and x_in.ndim >= 2:
          tap._active_batch_shape = (int(x_in.shape[0]), int(x_in.shape[1]) if x_in.ndim >= 3 else 1)
        tap._active_forced_experts = bound.get("forced_routed_experts")

        if tap.role == "sampler" and x_in is not None:
          tap._active_seq_bounds = tap._extract_sampler_seq_bounds(x_in, bound.get("attention_metadata"))

        tap._tap(x_in, sub_idx, "layer_in")
        tagged = tap._tag_child_modules(layer_self, sub_idx)
        try:
          out = orig_layer_call(layer_self, *args, **kwargs)
        finally:
          for obj, attr in tagged:
            with contextlib.suppress(Exception):
              object.__delattr__(obj, attr)
        tap._tap(out[0] if isinstance(out, tuple) else out, sub_idx, "layer_out")
        return out

      return _wrapped_layer

    for lyr_cls in layer_classes:
      self._patch_method(lyr_cls, "__call__", _make_layer_wrapper)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    for cls, method_name, orig in reversed(self._patched_classes):
      setattr(cls, method_name, orig)
    self._patched_classes.clear()

  def get_probes(
      self,
      *,
      target_next_tokens: np.ndarray | None = None,
      prompt_tokens_probe: np.ndarray | None = None,
      temperature: float = 1.0,
  ) -> dict[str, np.ndarray]:
    """Returns all captured module tensors plus boundary `token_embedder`, `decoder_norm`, and `logits_*`."""
    out = {k: np.array(v, copy=True) for k, v in self._buffers.items()}
    capture_output_head_probes(
        self.model,
        out,
        num_layers=self.num_layers,
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=temperature,
        prompt_len=self.prompt_len,
        probe_positions=self.probe_positions,
    )
    out["probe_positions"] = np.array(self.probe_positions, dtype=np.int32)
    out["prompt_len"] = np.int32(self.prompt_len)
    out["gen_len"] = np.int32(self.gen_len)
    out["num_layers"] = np.int32(self.num_layers)
    return out


# --------------------------------------------------------- Isolated Submodule Replay


def _pad_seq_3d(arr: jax.Array, target_len: int) -> jax.Array:
  pad_n = target_len - int(arr.shape[1])
  return arr if pad_n <= 0 else jnp.pad(arr, ((0, 0), (0, pad_n), (0, 0)))


def _safe_input_3d(np_arr: np.ndarray, dtype: Any, mesh: Any = None, b_size: int = 1) -> jax.Array:
  clean = np.nan_to_num(np.asarray(np_arr, dtype=np.float32), nan=0.0)
  return _to_mesh_array(np.repeat(clean[None, :, :], max(1, b_size), axis=0), dtype, mesh)


def _replay_layer_attention(
    attn: Any, sa_norm1: jax.Array, p: str, iso: dict[str, np.ndarray], *, k_pad: int, mesh: Any, is_contiguous_prefix: bool
) -> jax.Array | None:
  """Replays attention sub-projections (`in_proj_qkvz`, `in_proj_ba`, `query`, `key`, `value`) and `attn.out`."""
  if attn is None:
    return None
  b_pad, k_orig = int(sa_norm1.shape[0]), int(sa_norm1.shape[1])
  for proj_name in ("in_proj_qkvz", "in_proj_ba"):
    proj_out = _try_call_primary(getattr(attn, proj_name, None), sa_norm1)
    if proj_out is not None:
      iso[f"{p}.attn.{proj_name}"] = np.asarray(proj_out[0], dtype=np.float32)

  attn_inner = getattr(attn, "attention", attn)
  for proj_name in ("query", "key", "value"):
    proj_out = _try_call_primary(getattr(attn_inner, proj_name, None), sa_norm1)
    if proj_out is not None:
      iso[f"{p}.attn.{proj_name}"] = np.asarray(jnp.reshape(proj_out[0], (k_orig, -1)), dtype=np.float32)

  if not (is_contiguous_prefix and callable(attn)):
    return None
  sa_norm1_pad = _pad_seq_3d(sa_norm1, k_pad)
  seg_np, pos_np = np.zeros((b_pad, k_pad), dtype=np.int32), np.zeros((b_pad, k_pad), dtype=np.int32)
  seg_np[:, :k_orig] = 1
  pos_np[:, :k_orig] = np.arange(k_orig, dtype=np.int32)[None, :]
  seg_ids, pos_ids = _to_mesh_array(seg_np, jnp.int32, mesh), _to_mesh_array(pos_np, jnp.int32, mesh)
  if "decoder_positions" in inspect.signature(attn.__call__).parameters:
    attn_full = _try_call_primary(attn, sa_norm1_pad, seg_ids, pos_ids, True, "train")
  else:
    attn_full = _try_call_primary(attn, sa_norm1_pad, model_mode="train", decoder_segment_ids=seg_ids)
  if attn_full is None:
    return None
  iso[f"{p}.attn.out"] = np.asarray(attn_full[0, :k_orig], dtype=np.float32)
  return attn_full[:, :k_orig]


def _replay_layer_mlp(
    mlp: Any, sa_norm2: jax.Array, p: str, sampler_probes: dict[str, np.ndarray], iso: dict[str, np.ndarray], *, cfg: Any, k_pad: int, mesh: Any
) -> jax.Array | None:
  """Replays router gate, routed experts, shared expert, shared expert gate, and combined `mlp.out`."""
  if mlp is None:
    return None
  b_pad, k_orig = int(sa_norm2.shape[0]), int(sa_norm2.shape[1])
  model_dtype = getattr(cfg, "dtype", jnp.bfloat16) if cfg is not None else jnp.bfloat16
  float32_wsum = bool(getattr(cfg, "float32_weight_sum", True)) if cfg is not None else True
  sa_norm2_pad = _pad_seq_3d(sa_norm2, k_pad)
  routed = getattr(mlp, "routed_experts", None)
  k_top = _resolve_top_k(routed, cfg) if routed is not None else 8
  raw_sel_exp = (
      np.asarray(sampler_probes[f"{p}.mlp.selected_experts"], dtype=np.int32)[:, :k_top]
      if f"{p}.mlp.selected_experts" in sampler_probes
      else None
  )
  sa_sel_exp_pad = (
      _pad_seq_3d(_to_mesh_array(np.repeat(np.maximum(0, raw_sel_exp)[None, :, :], b_pad, axis=0), jnp.int32, mesh), k_pad)
      if raw_sel_exp is not None
      else None
  )
  iso_routed_out = None
  if routed is not None:
    g_logits = _try_call_primary(getattr(routed, "gate", None), sa_norm2)
    if g_logits is not None:
      iso[f"{p}.mlp.gate_logits"] = np.asarray(g_logits[0], dtype=np.float32)
      _, r_topk = jax.lax.top_k(jax.nn.softmax(g_logits.astype(jnp.float32), axis=-1), min(k_top, int(g_logits.shape[-1])))
      iso[f"{p}.mlp.router_topk"] = np.asarray(r_topk[0], dtype=np.int32)
      iso[f"{p}.mlp.selected_experts"] = raw_sel_exp.copy() if raw_sel_exp is not None else np.asarray(r_topk[0], dtype=np.int32)
    r_full = _try_call_primary(routed, sa_norm2_pad, forced_routed_experts=sa_sel_exp_pad)
    if r_full is not None:
      iso_routed_out = r_full[:, :k_orig]
      iso[f"{p}.mlp.routed_experts"] = np.asarray(iso_routed_out[0], dtype=np.float32)

  shared_fn = getattr(mlp, "shared_expert", None)
  shared_kw = {"deterministic": True} if callable(shared_fn) and "deterministic" in inspect.signature(shared_fn.__call__).parameters else {}
  iso_shared_out = _try_call_primary(shared_fn, sa_norm2, **shared_kw)
  if iso_shared_out is not None:
    iso[f"{p}.mlp.shared_expert"] = np.asarray(iso_shared_out[0], dtype=np.float32)

  sg_logits = _try_call_primary(getattr(mlp, "shared_expert_gate", None), sa_norm2)
  iso_shared_gate = jax.nn.sigmoid(sg_logits.astype(jnp.float32)) if sg_logits is not None else None
  if iso_shared_gate is not None:
    iso[f"{p}.mlp.shared_expert_gate"] = np.asarray(iso_shared_gate[0], dtype=np.float32)

  iso_mlp_out = None
  if iso_routed_out is not None and iso_shared_out is not None and iso_shared_gate is not None:
    if float32_wsum:
      iso_mlp_out = (iso_routed_out.astype(jnp.float32) + iso_shared_gate * iso_shared_out.astype(jnp.float32)).astype(model_dtype)
    else:
      iso_mlp_out = iso_routed_out + iso_shared_gate.astype(model_dtype) * iso_shared_out
  elif iso_routed_out is not None and shared_fn is None:
    iso_mlp_out = iso_routed_out
  else:
    m_full = _try_call_primary(mlp, sa_norm2_pad.astype(model_dtype), deterministic=True, forced_routed_experts=sa_sel_exp_pad)
    if m_full is not None:
      iso_mlp_out = m_full[:, :k_orig]

  if iso_mlp_out is not None:
    iso[f"{p}.mlp.out"] = np.asarray(iso_mlp_out[0], dtype=np.float32)
  return iso_mlp_out


def _replay_single_layer(
    tr_layer: Any, lyr_idx: int, sampler_probes: dict[str, np.ndarray], iso: dict[str, np.ndarray], *, cfg: Any, mesh: Any, b_pad: int, is_tpu: bool, is_contiguous_prefix: bool
) -> None:
  """Replays a single Trainer decoder layer submodule-by-submodule on Sampler's captured inputs."""
  p = f"layer_{lyr_idx}"
  model_dtype = getattr(cfg, "dtype", jnp.bfloat16) if cfg is not None else jnp.bfloat16
  gate_dtype = jnp.float32 if (bool(getattr(cfg, "float32_gate_logits", True)) if cfg is not None else True) else model_dtype
  raw_layer_in = np.asarray(sampler_probes[f"{p}.layer_in"], dtype=np.float32)
  sa_layer_in = _safe_input_3d(raw_layer_in, model_dtype, mesh, b_pad)
  k_orig = int(sa_layer_in.shape[1])
  k_pad = max(256, ((k_orig + 255) // 256) * 256) if is_tpu and (k_orig % 256 != 0) else k_orig
  iso[f"{p}.layer_in"] = raw_layer_in.copy()

  def _sa_or_iso(key: str, fallback: jax.Array, dtype: Any = model_dtype) -> jax.Array:
    return _safe_input_3d(sampler_probes[key], dtype, mesh, b_pad) if key in sampler_probes else fallback.astype(dtype)

  iso_norm1 = _try_call_primary(getattr(tr_layer, "input_layernorm", None), sa_layer_in)
  if iso_norm1 is not None:
    iso[f"{p}.input_layernorm"] = np.asarray(iso_norm1[0], dtype=np.float32)
  else:
    iso_norm1 = sa_layer_in

  sa_norm1 = _sa_or_iso(f"{p}.input_layernorm", iso_norm1)
  iso_attn_out = _replay_layer_attention(
      getattr(tr_layer, "attention", None), sa_norm1, p, iso, k_pad=k_pad, mesh=mesh, is_contiguous_prefix=is_contiguous_prefix
  )
  if iso_attn_out is None and f"{p}.attn.out" in sampler_probes:
    iso_attn_out = _safe_input_3d(sampler_probes[f"{p}.attn.out"], model_dtype, mesh, b_pad)
    iso[f"{p}.attn.out"] = np.asarray(sampler_probes[f"{p}.attn.out"], dtype=np.float32).copy()

  if iso_attn_out is not None:
    iso_post_attn_res = (sa_layer_in + iso_attn_out.astype(sa_layer_in.dtype)).astype(model_dtype)
    iso[f"{p}.post_attn_residual"] = np.asarray(iso_post_attn_res[0], dtype=np.float32)
  else:
    iso_post_attn_res = sa_layer_in

  sa_post_attn_res = _sa_or_iso(f"{p}.post_attn_residual", iso_post_attn_res)
  post_ln = getattr(tr_layer, "post_attention_layernorm", None)
  iso_norm2 = _try_call_primary(post_ln, sa_post_attn_res)
  if iso_norm2 is not None:
    iso[f"{p}.post_attention_layernorm"] = np.asarray(iso_norm2[0], dtype=np.float32)
  else:
    iso_norm2 = sa_post_attn_res

  norm2_dtype = getattr(post_ln, "dtype", None) or getattr(iso_norm2, "dtype", gate_dtype)
  sa_norm2 = _sa_or_iso(f"{p}.post_attention_layernorm", iso_norm2, dtype=norm2_dtype)
  iso_mlp_out = _replay_layer_mlp(
      getattr(tr_layer, "mlp", None), sa_norm2, p, sampler_probes, iso, cfg=cfg, k_pad=k_pad, mesh=mesh
  )
  if iso_mlp_out is not None:
    iso_layer_out = (sa_post_attn_res + iso_mlp_out.astype(sa_post_attn_res.dtype)).astype(model_dtype)
    iso[f"{p}.layer_out"] = np.asarray(iso_layer_out[0], dtype=np.float32)


def run_isolated_trainer_replay(
    trainer_model: Any,
    sampler_probes: dict[str, np.ndarray],
    *,
    target_next_tokens: np.ndarray | None = None,
    prompt_tokens_probe: np.ndarray | None = None,
    temperature: float = 1.0,
    layer_indices: set[int] | Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
  """Runs each Trainer submodule in isolation on the Sampler's exact captured inputs."""
  base = getattr(trainer_model, "base", trainer_model)
  decoder, num_layers, _, _, _ = discover_decoder_structure(trainer_model)
  cfg = getattr(decoder, "config", None)
  allowed_layers = {int(i) for i in layer_indices} if layer_indices is not None else None
  probe_positions = np.asarray(sampler_probes.get("probe_positions", []), dtype=np.int32)
  prompt_len = int(sampler_probes.get("prompt_len", 0))
  is_contiguous_prefix = probe_positions.size > 0 and np.array_equal(probe_positions, np.arange(probe_positions.size, dtype=np.int32))

  iso: dict[str, np.ndarray] = {
      "probe_positions": probe_positions,
      "prompt_len": np.int32(prompt_len),
      "gen_len": np.int32(sampler_probes.get("gen_len", 0)),
      "num_layers": np.int32(num_layers),
  }
  with _model_mesh_context(decoder, base, cfg) as mesh:
    is_tpu = jax.default_backend() == "tpu"
    b_pad = 1
    if is_tpu and mesh is not None and hasattr(mesh, "shape"):
      for ax_name, ax_size in mesh.shape.items():
        if ax_name not in ("tensor", "model", "mlp", "heads", "kv"):
          b_pad *= int(ax_size)
      b_pad = max(1, b_pad)

    for lyr_idx in range(num_layers):
      if (allowed_layers is not None and lyr_idx not in allowed_layers) or f"layer_{lyr_idx}.layer_in" not in sampler_probes:
        continue
      tr_layer = get_layer_for_index(trainer_model, lyr_idx)
      if tr_layer is not None:
        _replay_single_layer(
            tr_layer, lyr_idx, sampler_probes, iso, cfg=cfg, mesh=mesh, b_pad=b_pad, is_tpu=is_tpu, is_contiguous_prefix=is_contiguous_prefix
        )

  capture_output_head_probes(
      trainer_model,
      iso,
      num_layers=num_layers,
      target_next_tokens=target_next_tokens,
      prompt_tokens_probe=prompt_tokens_probe,
      temperature=temperature,
      prompt_len=prompt_len,
      probe_positions=probe_positions,
      input_probes=sampler_probes,
      b_pad=b_pad,
  )
  return iso


# --------------------------------------------------------- Divergence Metrics & Reporting


def compute_tensor_divergence(ref_sa: np.ndarray, cand_tr: np.ndarray) -> dict[str, float]:
  """Computes numerical divergence metrics `cand_tr - ref_sa` over valid finite positions."""
  sa, tr = np.asarray(ref_sa, dtype=np.float64), np.asarray(cand_tr, dtype=np.float64)
  min_rows = min(sa.shape[0], tr.shape[0])
  sa, tr = sa[:min_rows], tr[:min_rows]
  if sa.ndim == 2 and tr.ndim == 2 and sa.shape[1] != tr.shape[1]:
    min_d = min(sa.shape[1], tr.shape[1])
    sa, tr = sa[:, :min_d], tr[:, :min_d]

  valid = np.isfinite(sa) & np.isfinite(tr)
  if not np.any(valid):
    return {
        "rel_l2": 0.0,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "median_abs": 0.0,
        "p99_abs": 0.0,
        "cos_sim": 1.0,
        "signed_bias": 0.0,
        "n_elems": 0,
    }

  sa_v, tr_v = sa[valid], tr[valid]
  diff = tr_v - sa_v
  abs_diff = np.abs(diff)
  sa_norm, tr_norm, diff_norm = float(np.linalg.norm(sa_v)), float(np.linalg.norm(tr_v)), float(np.linalg.norm(diff))
  denom = sa_norm * tr_norm
  cos_sim = 1.0 if diff_norm == 0.0 else (float(np.dot(sa_v, tr_v) / denom) if denom > 1e-24 else 1.0)

  return {
      "rel_l2": float(diff_norm / max(sa_norm, 1e-12)),
      "max_abs": float(np.max(abs_diff)),
      "mean_abs": float(np.mean(abs_diff)),
      "median_abs": float(np.median(abs_diff)),
      "p99_abs": float(np.percentile(abs_diff, 99)),
      "cos_sim": float(np.clip(cos_sim, -1.0, 1.0)),
      "signed_bias": float(np.mean(diff)),
      "n_elems": int(sa_v.size),
  }


def compute_router_agreement(sa_topk: np.ndarray, tr_topk: np.ndarray) -> dict[str, float]:
  """Computes top-1 exact match and top-k set Jaccard/overlap agreement between `[N, K]` expert index arrays."""
  sa, tr = np.asarray(sa_topk, dtype=np.int32), np.asarray(tr_topk, dtype=np.int32)
  n = min(sa.shape[0], tr.shape[0])
  sa, tr = sa[:n], tr[:n]
  valid_rows = (sa[:, 0] >= 0) & (tr[:, 0] >= 0) if sa.ndim == 2 and sa.size > 0 else np.zeros((0,), dtype=bool)
  sa, tr = sa[valid_rows], tr[valid_rows]
  if sa.shape[0] == 0:
    return {"top1_agree": 1.0, "topk_overlap": 1.0, "topk_jaccard": 1.0}

  k = max(sa.shape[1], 1)
  overlaps, jaccards = [], []
  for r in range(sa.shape[0]):
    s_set, t_set = {int(x) for x in sa[r] if x >= 0}, {int(x) for x in tr[r] if x >= 0}
    inter = len(s_set & t_set)
    overlaps.append(inter / k)
    jaccards.append(inter / max(len(s_set | t_set), 1))
  return {
      "top1_agree": float(np.mean(sa[:, 0] == tr[:, 0])),
      "topk_overlap": float(np.mean(overlaps)),
      "topk_jaccard": float(np.mean(jaccards)),
  }


def _summarize_module_families(family_buckets: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
  summary: dict[str, dict[str, float]] = {}
  for fam, recs in family_buckets.items():
    iso_rels = [r["isolated"]["rel_l2"] for r in recs if r["isolated"] is not None]
    iso_maxs = [r["isolated"]["max_abs"] for r in recs if r["isolated"] is not None]
    cum_rels = [r["cumulative"]["rel_l2"] for r in recs]
    cum_maxs = [r["cumulative"]["max_abs"] for r in recs]
    cum_coss = [r["cumulative"]["cos_sim"] for r in recs]
    biases = [r["cumulative"]["signed_bias"] for r in recs]
    summary[fam] = {
        "count": len(recs),
        "iso_rel_l2_mean": float(np.mean(iso_rels)) if iso_rels else 0.0,
        "iso_rel_l2_max": float(np.max(iso_rels)) if iso_rels else 0.0,
        "iso_max_abs": float(np.max(iso_maxs)) if iso_maxs else 0.0,
        "cum_rel_l2_mean": float(np.mean(cum_rels)) if cum_rels else 0.0,
        "cum_rel_l2_max": float(np.max(cum_rels)) if cum_rels else 0.0,
        "cum_max_abs": float(np.max(cum_maxs)) if cum_maxs else 0.0,
        "cum_cos_sim_min": float(np.min(cum_coss)) if cum_coss else 1.0,
        "signed_bias_mean": float(np.mean(biases)) if biases else 0.0,
    }
  return summary


def _print_divergence_report(
    tag: str,
    probe_positions: np.ndarray,
    prompt_len: int,
    num_layers: int,
    records: list[dict[str, Any]],
    visible_layers: set[int],
    family_summary: dict[str, dict[str, float]],
    top_isolated: list[dict[str, Any]],
    logprob_div: dict[str, Any] | None,
    logits_top1_agree: float | None,
) -> None:
  """Prints the formatted ASCII table for module-by-module divergence."""
  n_p = int(np.sum(probe_positions < prompt_len)) if probe_positions.size else 0
  n_d = int(np.sum(probe_positions >= prompt_len)) if probe_positions.size else 0
  print(f"\n==================== MODULE-BY-MODULE VALUE DIVERGENCE: {tag} ====================")
  print(f"  Probed Tokens: total={len(probe_positions)} (prompt={n_p}, decode={n_d}) | Layers={num_layers}")
  print(
      f"  {'Lyr':>3} | {'Module':<25} | {'Iso RelL2':>10} | {'Iso Max|d|':>10} | "
      f"{'Cum RelL2':>10} | {'Cum Max|d|':>10} | {'Cum CosSim':>10} | {'SignedBias':>11} | {'RouterTop1':>10}"
  )
  print("  " + "-" * 116)

  for r in records:
    lyr = r["layer"]
    if lyr is not None and lyr not in visible_layers:
      continue
    lyr_str = f"{lyr:02d}" if lyr is not None else "--"
    c_m, i_m = r["cumulative"], r["isolated"]
    iso_r_str = f"{i_m['rel_l2']:10.3e}" if i_m is not None else f"{'N/A':>10}"
    iso_m_str = f"{i_m['max_abs']:10.3e}" if i_m is not None else f"{'N/A':>10}"
    r_info = r["router"]
    r_src = (r_info["iso"] if r_info.get("iso") is not None else r_info["cum"]) if r_info is not None else None
    r_str = f"{r_src['top1_agree']:9.1%}" if r_src is not None else f"{'-':>10}"
    print(
        f"  {lyr_str:>3} | {r['module']:<25} | {iso_r_str} | {iso_m_str} | "
        f"{c_m['rel_l2']:10.3e} | {c_m['max_abs']:10.3e} | {c_m['cos_sim']:10.6f} | "
        f"{c_m['signed_bias']:+11.3e} | {r_str:>10}"
    )

  print("\n  --- Module Family Attribution Summary (Across All Layers) ---")
  print(
      f"  {'Module Family':<25} | {'Count':>5} | {'IsoMeanRelL2':>12} | {'IsoMaxRelL2':>12} | "
      f"{'CumMeanRelL2':>12} | {'CumMaxRelL2':>12} | {'MinCosSim':>10}"
  )
  print("  " + "-" * 102)
  for fam, fs in family_summary.items():
    print(
        f"  {fam:<25} | {fs['count']:>5d} | {fs['iso_rel_l2_mean']:12.3e} | {fs['iso_rel_l2_max']:12.3e} | "
        f"{fs['cum_rel_l2_mean']:12.3e} | {fs['cum_rel_l2_max']:12.3e} | {fs['cum_cos_sim_min']:10.6f}"
    )

  if top_isolated:
    print("\n  --- Top-5 Isolated Single-Module Divergence Bottlenecks (m_trainer(x_sampler) vs y_sampler) ---")
    for idx, r in enumerate(top_isolated, 1):
      im, cm = r["isolated"], r["cumulative"]
      l_tag = f"layer_{r['layer']:02d}" if r["layer"] is not None else "boundary"
      print(
          f"    {idx}. {l_tag}.{r['module']:<24} iso_rel_l2={im['rel_l2']:.4e}  iso_max|d|={im['max_abs']:.4e}  "
          f"(cum_rel_l2={cm['rel_l2']:.4e}, cos_sim={im['cos_sim']:.6f})"
      )

  if logprob_div is not None:
    lp_c, lp_i = logprob_div["cumulative"], logprob_div["isolated"]
    iso_lp_str = f"{lp_i['mean_abs']:.4e} (bias={lp_i['signed_bias']:+.4e})" if lp_i else "N/A"
    t1_str = f"{logits_top1_agree:.2%}" if logits_top1_agree is not None else "N/A"
    print(
        "\n  --- Output Head Logprob Impact ---\n"
        f"    Isolated Head |dlogp| mean  : {iso_lp_str}\n"
        f"    Cumulative    |dlogp| mean  : {lp_c['mean_abs']:.4e} "
        f"(max={lp_c['max_abs']:.4e}, signed_bias={lp_c['signed_bias']:+.4e})\n"
        f"    Probed Token Top-1 Agreement: {t1_str}"
    )
  print("=" * 84 + "\n", flush=True)


def compare_module_probes(
    sampler_probes: dict[str, np.ndarray],
    trainer_cum_probes: dict[str, np.ndarray],
    trainer_iso_probes: dict[str, np.ndarray] | None = None,
    *,
    out_dir: str | None = None,
    tag: str = "Sampler vs Trainer",
    probe_layers: str = "all",
) -> dict[str, Any]:
  """Compares Sampler vs Trainer module activations (both Isolated and Cumulative) and prints report."""
  probe_positions = np.asarray(sampler_probes.get("probe_positions", []), dtype=np.int32)
  prompt_len = int(sampler_probes.get("prompt_len", 0))
  num_layers = int(sampler_probes.get("num_layers", trainer_cum_probes.get("num_layers", 0)))
  visible_layers = parse_probe_layers(probe_layers) or set(range(num_layers))

  records: list[dict[str, Any]] = []
  family_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

  def _eval_key(key: str, layer_idx: int | None, mod_name: str):
    if key not in sampler_probes or key not in trainer_cum_probes:
      return
    sa_arr = sampler_probes[key]
    if np.issubdtype(sa_arr.dtype, np.integer):
      return
    cum_div = compute_tensor_divergence(sa_arr, trainer_cum_probes[key])
    iso_div = (
        compute_tensor_divergence(sa_arr, trainer_iso_probes[key])
        if (trainer_iso_probes is not None and key in trainer_iso_probes)
        else None
    )

    router_info = None
    if layer_idx is not None and mod_name in ("mlp.gate_logits", "mlp.routed_experts"):
      r_key, s_key = f"layer_{layer_idx}.mlp.router_topk", f"layer_{layer_idx}.mlp.selected_experts"
      sa_r = sampler_probes.get(r_key, sampler_probes.get(s_key))
      tr_cum_r = trainer_cum_probes.get(r_key, trainer_cum_probes.get(s_key))
      tr_iso_r = trainer_iso_probes.get(r_key, trainer_iso_probes.get(s_key)) if trainer_iso_probes else None
      if sa_r is not None and tr_cum_r is not None:
        router_info = {
            "cum": compute_router_agreement(sa_r, tr_cum_r),
            "iso": compute_router_agreement(sa_r, tr_iso_r) if tr_iso_r is not None else None,
        }

    rec = {
        "key": key,
        "layer": layer_idx,
        "module": mod_name,
        "shape": list(sa_arr.shape),
        "cumulative": cum_div,
        "isolated": iso_div,
        "router": router_info,
    }
    records.append(rec)
    family_buckets[mod_name].append(rec)

  _eval_key("token_embedder", None, "token_embedder")
  for lyr in range(num_layers):
    if lyr in visible_layers:
      for mod_name in LAYER_MODULES:
        if mod_name not in ("mlp.router_topk", "mlp.selected_experts"):
          _eval_key(f"layer_{lyr}.{mod_name}", lyr, mod_name)
  for b_mod in ("decoder_norm", "logits_head", "logits_summary"):
    _eval_key(b_mod, None, b_mod)

  family_summary = _summarize_module_families(family_buckets)
  internal_recs = [r for r in records if r["module"] not in ("logits_summary", "logits_head", "layer_in")]
  top_isolated = sorted(
      [r for r in internal_recs if r["isolated"] is not None], key=lambda r: r["isolated"]["rel_l2"], reverse=True
  )[:5]
  top_cumulative = sorted(internal_recs, key=lambda r: r["cumulative"]["rel_l2"], reverse=True)[:5]

  logits_top1_agree = None
  if "logits_top1" in sampler_probes and "logits_top1" in trainer_cum_probes:
    sa_t1, tr_t1 = sampler_probes["logits_top1"], trainer_cum_probes["logits_top1"]
    n_t1 = min(len(sa_t1), len(tr_t1))
    logits_top1_agree = float(np.mean(sa_t1[:n_t1] == tr_t1[:n_t1])) if n_t1 > 0 else 1.0

  logprob_div = None
  if "logits_summary" in sampler_probes and "logits_summary" in trainer_cum_probes:
    sa_lp, tr_cum_lp = sampler_probes["logits_summary"][:, 2], trainer_cum_probes["logits_summary"][:, 2]
    tr_iso_lp = (
        trainer_iso_probes["logits_summary"][:, 2]
        if (trainer_iso_probes and "logits_summary" in trainer_iso_probes)
        else None
    )
    logprob_div = {
        "cumulative": compute_tensor_divergence(sa_lp, tr_cum_lp),
        "isolated": compute_tensor_divergence(sa_lp, tr_iso_lp) if tr_iso_lp is not None else None,
        "top1_agree": logits_top1_agree,
    }

  _print_divergence_report(
      tag,
      probe_positions,
      prompt_len,
      num_layers,
      records,
      visible_layers,
      family_summary,
      top_isolated,
      logprob_div,
      logits_top1_agree,
  )

  report = {
      "tag": tag,
      "num_layers": num_layers,
      "probe_positions": probe_positions.tolist(),
      "prompt_len": prompt_len,
      "records": records,
      "family_summary": family_summary,
      "top_isolated_bottlenecks": [
          {
              "key": r["key"],
              "layer": r["layer"],
              "module": r["module"],
              "iso_rel_l2": r["isolated"]["rel_l2"],
              "iso_max_abs": r["isolated"]["max_abs"],
              "cum_rel_l2": r["cumulative"]["rel_l2"],
          }
          for r in top_isolated
      ],
      "top_cumulative_bottlenecks": [
          {
              "key": r["key"],
              "layer": r["layer"],
              "module": r["module"],
              "cum_rel_l2": r["cumulative"]["rel_l2"],
              "cum_max_abs": r["cumulative"]["max_abs"],
          }
          for r in top_cumulative
      ],
      "logprob_divergence": logprob_div,
  }

  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "module_divergence_report.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)

  return report
