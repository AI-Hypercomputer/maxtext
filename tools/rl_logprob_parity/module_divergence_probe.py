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
     per-module intermediate tensors in-situ during live Sampler (chunked prefill + decode)
     and Trainer (2D unpacked or 1D sequence-packed, scanned or unscanned) forward passes.
  2. `run_isolated_trainer_replay`: Teacher-forced single-module replay feeding the Sampler's
     exact captured inputs (and forced router indices) into each Trainer submodule in isolation.
  3. `compare_module_probes`: Dual divergence analysis reporting both Isolated (single-module)
     and Cumulative (end-to-end) value divergence, router top-k agreement, module-family
     attribution, and root-cause bottleneck ranking.
"""

from __future__ import annotations

import collections
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


def select_probe_positions(
    prompt_len: int,
    gen_len: int,
    max_probe_tokens: int = 64,
    *,
    max_tokens: int | None = None,
) -> np.ndarray:
  """Selects a canonical, sorted 1D int32 array of token indices in `[0, prompt_len + gen_len)`.

  When `prompt_len + gen_len <= max_probe_tokens` (or `max_probe_tokens <= 0`), selects all
  token positions. Otherwise, selects a boundary-aware subset spanning prompt start,
  block/chunk boundaries (256, 512, 2048), prompt->decode transition, and decode rollout steps.
  """
  if max_tokens is not None:
    max_probe_tokens = int(max_tokens)
  total = max(0, int(prompt_len) + int(gen_len))
  if total == 0:
    return np.zeros((0,), dtype=np.int32)
  if max_probe_tokens <= 0 or total <= max_probe_tokens:
    return np.arange(total, dtype=np.int32)

  p_len, g_len = int(prompt_len), int(gen_len)
  if g_len <= 0:
    p_budget, g_budget = max_probe_tokens, 0
  elif p_len <= 0:
    p_budget, g_budget = 0, max_probe_tokens
  else:
    g_budget = min(g_len, max(4, max_probe_tokens // 2))
    p_budget = min(p_len, max_probe_tokens - g_budget)
    if p_budget + g_budget < max_probe_tokens:
      g_budget = min(g_len, max_probe_tokens - p_budget)

  chosen: set[int] = set()
  if p_budget > 0 and p_len > 0:
    anchors = [0, 1, 2, 3, 255, 256, 511, 512, 2047, 2048, p_len - 4, p_len - 3, p_len - 2, p_len - 1]
    for a in anchors:
      if 0 <= a < p_len and len(chosen) < p_budget:
        chosen.add(int(a))
    if len(chosen) < p_budget:
      for x in np.linspace(0, p_len - 1, num=p_budget * 2, dtype=np.int32):
        if len(chosen) >= p_budget:
          break
        chosen.add(int(x))

  if g_budget > 0 and g_len > 0:
    g_chosen: set[int] = set()
    g_anchors = [
        p_len,
        p_len + 1,
        p_len + 2,
        p_len + 3,
        p_len + 255,
        p_len + 256,
        total - 4,
        total - 3,
        total - 2,
        total - 1,
    ]
    for a in g_anchors:
      if p_len <= a < total and len(g_chosen) < g_budget:
        g_chosen.add(int(a))
    if len(g_chosen) < g_budget:
      for x in np.linspace(p_len, total - 1, num=g_budget * 2, dtype=np.int32):
        if len(g_chosen) >= g_budget:
          break
        g_chosen.add(int(x))
    chosen.update(g_chosen)

  return np.array(sorted(chosen)[:max_probe_tokens], dtype=np.int32)


def discover_decoder_structure(model) -> tuple[Any, int, bool, int, list[tuple[int, Any]]]:
  """Returns `(decoder, num_layers, is_scanned, cycle_interval, sublayers)`."""
  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  cfg = getattr(decoder, "config", None)
  cfg_num_layers = int(getattr(cfg, "num_decoder_layers", 0) or 0)

  unscanned = []
  max_check = max(cfg_num_layers, 128)
  for i in range(max_check):
    lyr = getattr(decoder, f"layers_{i}", None)
    if lyr is not None:
      unscanned.append((i, lyr))
  if unscanned:
    num_layers = cfg_num_layers or len(unscanned)
    return decoder, num_layers, False, 1, unscanned

  scanned = getattr(decoder, "layers", None)
  if scanned is not None:
    cycle_sublayers = []
    cycle_len = int(getattr(cfg, "inhomogeneous_layer_cycle_interval", 0) or 0)
    for i in range(max(cycle_len, 16)):
      lyr = getattr(scanned, f"layer_{i}", None)
      if lyr is not None:
        cycle_sublayers.append((i, lyr))
    if cycle_sublayers:
      c_interval = len(cycle_sublayers)
      num_layers = cfg_num_layers or c_interval
      return decoder, num_layers, True, c_interval, cycle_sublayers

    local_layers = list(getattr(scanned, "local_layers", None) or ())
    global_layer = getattr(scanned, "global_layer", None)
    if local_layers or global_layer is not None:
      combined = [(idx, lyr) for idx, lyr in enumerate(local_layers) if lyr is not None]
      if global_layer is not None:
        combined.append((len(combined), global_layer))
      c_interval = max(1, len(combined))
      num_layers = cfg_num_layers or c_interval
      return decoder, num_layers, True, c_interval, combined

  return decoder, max(cfg_num_layers, 1), False, 1, []


def _extract_single_layer_from_scanned(sublayer: nnx.Module, scan_idx: int, scan_length: int, param_scan_axis: int = 0) -> nnx.Module:
  """Slices a single unscanned layer instance at `scan_idx` from a scanned NNX block sublayer."""
  if scan_length <= 1:
    return sublayer
  graphdef, state = nnx.split(sublayer)

  def _slice_leaf(x):
    if hasattr(x, "shape") and x.ndim >= 2:
      if param_scan_axis < x.ndim and x.shape[param_scan_axis] == scan_length:
        return jnp.take(x, scan_idx, axis=param_scan_axis)
      if x.shape[0] == scan_length:
        return jnp.take(x, scan_idx, axis=0)
    return x

  sliced_state = jax.tree.map(_slice_leaf, state)
  return nnx.merge(graphdef, sliced_state)


def get_layer_for_index(model, layer_idx: int) -> Any | None:
  """Returns an unscanned single-layer module for global `layer_idx` (slicing scan axis if needed)."""
  decoder, num_layers, is_scanned, cycle_interval, sublayers = discover_decoder_structure(model)
  if not is_scanned:
    return getattr(decoder, f"layers_{layer_idx}", None)
  if not sublayers:
    return None
  sub_map = dict(sublayers)
  sub_idx = layer_idx % cycle_interval
  sublayer = sub_map.get(sub_idx)
  if sublayer is None:
    return None
  scan_length = max(1, num_layers // max(1, cycle_interval))
  scan_idx = layer_idx // max(1, cycle_interval)
  cfg = getattr(decoder, "config", None)
  param_scan_axis = int(getattr(cfg, "param_scan_axis", 0) or 0)
  return _extract_single_layer_from_scanned(sublayer, scan_idx=scan_idx, scan_length=scan_length, param_scan_axis=param_scan_axis)


def _compute_logits_summary_and_head(
    logits_3d: jax.Array,
    target_tokens_2d: jax.Array | None = None,
    logits_slice_dim: int = 256,
    temperature: float = 1.0,
    prompt_len: int | None = None,
    probe_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes `(logits_head, logits_summary, logits_top1)` for `[1, N_probe, V]` logits."""
  logits_f32 = jnp.asarray(logits_3d, dtype=jnp.float32)[0]  # [N_probe, V]
  n_tok, vocab_size = logits_f32.shape
  if temperature not in (0.0, 1.0):
    if prompt_len is not None and probe_positions is not None:
      pos_arr = jnp.asarray(probe_positions, dtype=jnp.int32)
      temp_scale = jnp.where(pos_arr >= (prompt_len - 1), jnp.float32(temperature), jnp.float32(1.0))[:, None]
      scaled_logits = logits_f32 / temp_scale
    else:
      scaled_logits = logits_f32 / jnp.float32(temperature)
  else:
    scaled_logits = logits_f32

  head_dim = min(int(logits_slice_dim), int(vocab_size))
  logits_head = np.asarray(scaled_logits[:, :head_dim], dtype=np.float32)
  lse = jax.nn.logsumexp(scaled_logits, axis=-1)
  max_logit = jnp.max(scaled_logits, axis=-1)
  top1_id = np.asarray(jnp.argmax(scaled_logits, axis=-1), dtype=np.int32)

  if target_tokens_2d is not None:
    tgt_ids = jnp.clip(jnp.asarray(target_tokens_2d, dtype=jnp.int32).reshape(-1)[:n_tok], 0, vocab_size - 1)
    tgt_logit = jnp.take_along_axis(scaled_logits, tgt_ids[:, None], axis=-1)[:, 0]
  else:
    tgt_logit = max_logit
  logprob = tgt_logit - lse
  summary = np.stack(
      [
          np.asarray(tgt_logit, dtype=np.float32),
          np.asarray(lse, dtype=np.float32),
          np.asarray(logprob, dtype=np.float32),
          np.asarray(max_logit, dtype=np.float32),
      ],
      axis=-1,
  )
  return logits_head, summary, top1_id


def _to_mesh_array(arr: Any, dtype: Any, mesh: Any) -> jax.Array:
  out = jnp.asarray(arr, dtype=dtype)
  if mesh is not None and getattr(mesh, "devices", None) is not None and getattr(mesh.devices, "size", 1) > 1:
    with contextlib.suppress(Exception):
      out = jax.device_put(out, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
  return out


@contextlib.contextmanager
def _model_mesh_context(decoder: Any, base: Any, cfg: Any):
  mesh = getattr(decoder, "mesh", getattr(base, "mesh", None))
  rules = getattr(cfg, "logical_axis_rules", None) if cfg is not None else None
  mesh_ctx = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
  rules_ctx = nnx.logical_axis_rules(rules) if rules else contextlib.nullcontext()
  with mesh_ctx, rules_ctx:
    yield mesh


def capture_output_head_probes(
    model,
    probes_dict: dict[str, np.ndarray],
    *,
    num_layers: int,
    target_next_tokens: np.ndarray | None = None,
    prompt_tokens_probe: np.ndarray | None = None,
    logits_slice_dim: int = 256,
    temperature: float = 1.0,
    prompt_len: int | None = None,
    probe_positions: np.ndarray | None = None,
) -> None:
  """Populates `token_embedder`, `decoder_norm`, `logits_head`, `logits_summary`, `logits_top1` in `probes_dict`."""
  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  embedder = getattr(base, "token_embedder", getattr(decoder, "token_embedder", None))
  cfg = getattr(decoder, "config", None)

  with _model_mesh_context(decoder, base, cfg) as mesh:
    if "token_embedder" not in probes_dict:
      if prompt_tokens_probe is not None and embedder is not None and callable(embedder):
        with contextlib.suppress(Exception):
          tok_in = _to_mesh_array(np.asarray(prompt_tokens_probe, dtype=np.int32)[None, :], jnp.int32, mesh)
          emb = embedder(tok_in)
          probes_dict["token_embedder"] = np.asarray(emb[0], dtype=np.float32)
      if "token_embedder" not in probes_dict and "layer_0.layer_in" in probes_dict:
        probes_dict["token_embedder"] = np.array(probes_dict["layer_0.layer_in"], dtype=np.float32, copy=True)

    last_key = f"layer_{num_layers - 1}.layer_out"
    if last_key not in probes_dict:
      return
    model_dtype = getattr(cfg, "dtype", jnp.bfloat16) if cfg is not None else jnp.bfloat16
    last_h = _to_mesh_array(np.asarray(probes_dict[last_key])[None, :, :], model_dtype, mesh)

    decoder_norm_fn = getattr(decoder, "decoder_norm", None)
    if callable(decoder_norm_fn):
      norm_out = decoder_norm_fn(last_h)
      probes_dict["decoder_norm"] = np.asarray(norm_out[0], dtype=np.float32)
    else:
      norm_out = last_h

    logits_3d = None
    logits_dense_fn = getattr(decoder, "logits_dense", None)
    apply_head_fn = getattr(decoder, "apply_output_head", None)
    if callable(logits_dense_fn) and not getattr(cfg, "logits_via_embedding", False):
      logits_3d = logits_dense_fn(norm_out)
      if getattr(cfg, "cast_logits_to_fp32", True):
        logits_3d = logits_3d.astype(jnp.float32)
    elif callable(apply_head_fn):
      model_mode = getattr(base, "model_mode", "train")
      sig = inspect.signature(apply_head_fn)
      if "normalize_y" in sig.parameters:
        logits_3d = apply_head_fn(embedder, norm_out, True, model_mode, normalize_y=False)
      else:
        logits_3d = apply_head_fn(embedder, last_h, True, model_mode)

    if logits_3d is not None:
      l_head, l_sum, l_top1 = _compute_logits_summary_and_head(
          logits_3d,
          target_tokens_2d=target_next_tokens,
          logits_slice_dim=logits_slice_dim,
          temperature=temperature,
          prompt_len=prompt_len,
          probe_positions=probe_positions,
      )
      probes_dict["logits_head"] = l_head
      probes_dict["logits_summary"] = l_sum
      probes_dict["logits_top1"] = l_top1


class ModuleProbeTap:
  """Context manager that hooks decoder layers and submodules via `jax.debug.callback`.

  Normalizes captured activations to `[N_probe, D]` for Sequence 0 across both:
    * Sampler (`role='sampler'`): flattened `[num_batched_tokens, 1, D]` across chunked prefill
      and token-by-token decode steps (using `attention_metadata.query_start_loc` and `seq_lens`,
      or `decoder_positions` / 2D fallback).
    * Trainer (`role='trainer'`): `[micro_batch, seq_len, D]` (both unpacked 2D and 1D
      sequence-packed `segment_ids`, and both unscanned and scanned `Qwen3_5ScannableBlock`).
  """

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

    self.decoder, self.num_layers, self.is_scanned, self.cycle_interval, self.sublayers = discover_decoder_structure(model)
    self.scan_length = max(1, self.num_layers // max(1, self.cycle_interval)) if self.is_scanned else 1

    self._pos_to_slot: dict[int, int] = {int(pos): idx for idx, pos in enumerate(self.probe_positions)}
    self._min_probe_pos = int(np.min(self.probe_positions)) if self.probe_positions.size else 0
    self._max_probe_pos = int(np.max(self.probe_positions)) if self.probe_positions.size else -1

    if self.packed_seq0_positions is not None and self.probe_positions.size > 0:
      self._trainer_gather_cols = np.asarray(self.packed_seq0_positions[self.probe_positions], dtype=np.int32)
    else:
      self._trainer_gather_cols = self.probe_positions

    self._buffers: dict[str, np.ndarray] = {}
    self._scan_call_counts: dict[tuple[int, str, int], int] = collections.defaultdict(int)
    self._patched_classes: list[tuple[type, str, Any]] = []
    self._active_sublayer_idx: int = 0
    self._active_batch_shape: tuple[int, int] = (1, 1)
    self._active_forced_experts: Any = None
    self._active_seq_bounds: tuple[Any, Any, Any, Any] = (
        jnp.int32(0),
        jnp.int32(self.total_seq_len),
        jnp.int32(0),
        jnp.int32(self.total_seq_len),
    )
    self._trainer_layer_call_counts: dict[tuple[int, str], int] = collections.defaultdict(int)
    self._filled_slots: dict[str, np.ndarray] = {}

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
    key = f"layer_{layer_idx}.{module_name}"
    out_dtype = np.int32 if np.issubdtype(arr_np.dtype, np.integer) else np.float32
    self._buffers[key] = np.asarray(arr_np, dtype=out_dtype).copy()

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
    q_start, q_end = int(q_start_np), int(q_end_np)
    seq_start, seq_end = int(seq_start_np), int(seq_end_np)
    q_len = q_end - q_start
    if q_len <= 0 or seq_end <= seq_start:
      return
    layer_idx = self._resolve_global_layer(int(sublayer_idx_np), module_name, seq_start)
    if self.layer_indices is not None and layer_idx not in self.layer_indices:
      return
    key = f"layer_{layer_idx}.{module_name}"
    out_dtype = np.int32 if np.issubdtype(arr_np.dtype, np.integer) else np.float32
    feat_shape = arr_np.shape[1:]
    buf = self._buffers.get(key)
    if buf is None:
      fill_val = -1 if out_dtype == np.int32 else np.nan
      buf = np.full((len(self.probe_positions), *feat_shape), fill_val, dtype=out_dtype)
      self._buffers[key] = buf
    filled = self._filled_slots.get(key)
    if filled is None:
      filled = np.zeros(len(self.probe_positions), dtype=bool)
      self._filled_slots[key] = filled

    for slot, logical_pos in enumerate(self.probe_positions.tolist()):
      if not filled[slot] and seq_start <= logical_pos < seq_end:
        row_idx = q_start + (logical_pos - seq_start)
        if 0 <= row_idx < max_rows:
          buf[slot] = np.asarray(arr_np[slot], dtype=out_dtype)
          filled[slot] = True

  def _normalize_3d(self, x: jax.Array) -> jax.Array:
    b_dim, s_dim = self._active_batch_shape
    if x.ndim == 2 and x.shape[0] == b_dim * s_dim:
      return jnp.reshape(x, (b_dim, s_dim, x.shape[1]))
    if x.ndim == 2:
      return x[:, None, :]
    if x.ndim > 3:
      return jnp.reshape(x, (x.shape[0], x.shape[1], -1))
    return x

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
    else:
      q_start, q_end, seq_start, seq_end = self._active_seq_bounds
      if x3.shape[1] > 1 and x3.shape[0] >= 1:
        seq0_2d = x3[0]
      else:
        seq0_2d = jnp.reshape(x3[:, 0], (x3.shape[0], -1))

      max_rows_int = max(1, int(seq0_2d.shape[0]))
      probe_pos_jnp = jnp.asarray(self.probe_positions, dtype=jnp.int32)
      row_indices = jnp.clip(q_start + (probe_pos_jnp - seq_start), 0, max_rows_int - 1)
      sliced_sa = jnp.take(seq0_2d, row_indices, axis=0).astype(out_jnp_dtype)
      has_probe_pos = jnp.any((probe_pos_jnp >= seq_start) & (probe_pos_jnp < seq_end))
      should_emit = (q_end > q_start) & has_probe_pos

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
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array | None,
      attention_metadata: Any,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    meta = attention_metadata
    if isinstance(meta, dict) and meta:
      meta = next(iter(meta.values()))
    q_loc = getattr(meta, "query_start_loc", None) if meta is not None else None
    s_lens = getattr(meta, "seq_lens", None) if meta is not None else None
    if q_loc is not None and s_lens is not None:
      q_loc_1d = jnp.reshape(jnp.asarray(q_loc, dtype=jnp.int32), (-1,))
      s_lens_1d = jnp.reshape(jnp.asarray(s_lens, dtype=jnp.int32), (-1,))
      q_start = q_loc_1d[0]
      q_end = q_loc_1d[1] if q_loc_1d.shape[0] > 1 else jnp.int32(inputs.shape[0])
      seq_end = s_lens_1d[0]
      seq_start = seq_end - (q_end - q_start)
      return q_start, q_end, seq_start, seq_end

    if inputs.ndim >= 3 and inputs.shape[1] > 1:
      s_len = jnp.int32(inputs.shape[1])
      return jnp.int32(0), s_len, jnp.int32(0), s_len

    if decoder_positions is not None:
      pos_1d = jnp.reshape(jnp.asarray(decoder_positions, dtype=jnp.int32), (-1,))
      n_tok = jnp.int32(inputs.shape[0])
      start_p = pos_1d[0]
      return jnp.int32(0), n_tok, start_p, start_p + n_tok

    n_tok = jnp.int32(inputs.shape[0])
    return jnp.int32(0), n_tok, jnp.int32(0), n_tok

  def _patch_method(self, cls: type, method_name: str, wrapper_factory) -> None:
    if any(c is cls and m == method_name for c, m, _ in self._patched_classes):
      return
    orig = getattr(cls, method_name, None)
    if orig is None:
      return
    self._patched_classes.append((cls, method_name, orig))
    setattr(cls, method_name, wrapper_factory(orig))

  def _tag_child_modules(self, layer_self: Any, sub_idx: int) -> list[tuple[Any, str]]:
    tagged = []

    def _set_tag(obj, name: str):
      if obj is not None:
        with contextlib.suppress(Exception):
          object.__setattr__(obj, "_probe_tag", (sub_idx, name))
          tagged.append((obj, "_probe_tag"))

    _set_tag(getattr(layer_self, "input_layernorm", None), "input_layernorm")
    _set_tag(getattr(layer_self, "post_attention_layernorm", None), "post_attention_layernorm")

    attn = getattr(layer_self, "attention", None)
    if attn is not None:
      _set_tag(attn, "attn.out")
      _set_tag(getattr(attn, "in_proj_qkvz", None), "attn.in_proj_qkvz")
      _set_tag(getattr(attn, "in_proj_ba", None), "attn.in_proj_ba")
      attn_inner = getattr(attn, "attention", attn)
      _set_tag(getattr(attn_inner, "query", None), "attn.query")
      _set_tag(getattr(attn_inner, "key", None), "attn.key")
      _set_tag(getattr(attn_inner, "value", None), "attn.value")

    mlp = getattr(layer_self, "mlp", None)
    if mlp is not None:
      _set_tag(mlp, "mlp.out")
      routed = getattr(mlp, "routed_experts", None)
      if routed is not None:
        _set_tag(routed, "mlp.routed_experts")
        gate_obj = getattr(routed, "gate", None)
        _set_tag(gate_obj, "mlp.gate_logits")
        if gate_obj is not None:
          top_k_val = int(
              getattr(
                  routed,
                  "num_experts_per_tok",
                  getattr(
                      getattr(routed, "config", None),
                      "num_experts_per_tok",
                      getattr(getattr(self.decoder, "config", None), "num_experts_per_tok", 8),
                  ),
              )
              or 8
          )
          with contextlib.suppress(Exception):
            object.__setattr__(gate_obj, "_probe_top_k", top_k_val)
            tagged.append((gate_obj, "_probe_top_k"))
      _set_tag(getattr(mlp, "shared_expert", None), "mlp.shared_expert")
      _set_tag(getattr(mlp, "shared_expert_gate", None), "mlp.shared_expert_gate")
    return tagged

  def __enter__(self) -> ModuleProbeTap:
    tap = self
    layer_classes: set[type] = set()
    submodule_classes: set[type] = set()

    for _, lyr in self.sublayers:
      layer_classes.add(type(lyr))
      for attr in ("input_layernorm", "post_attention_layernorm", "attention", "mlp"):
        sub = getattr(lyr, attr, None)
        if sub is not None:
          submodule_classes.add(type(sub))
      attn = getattr(lyr, "attention", None)
      if attn is not None:
        for attr in ("in_proj_qkvz", "in_proj_ba"):
          p = getattr(attn, attr, None)
          if p is not None:
            submodule_classes.add(type(p))
        attn_inner = getattr(attn, "attention", attn)
        for attr in ("query", "key", "value"):
          p = getattr(attn_inner, attr, None)
          if p is not None:
            submodule_classes.add(type(p))
      mlp = getattr(lyr, "mlp", None)
      if mlp is not None:
        routed = getattr(mlp, "routed_experts", None)
        if routed is not None:
          submodule_classes.add(type(routed))
          gate = getattr(routed, "gate", None)
          if gate is not None:
            submodule_classes.add(type(gate))
        for attr in ("shared_expert", "shared_expert_gate"):
          s = getattr(mlp, attr, None)
          if s is not None:
            submodule_classes.add(type(s))

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
          k_top = int(
              getattr(
                  mod_self,
                  "_probe_top_k",
                  getattr(getattr(mod_self, "config", None), "num_experts_per_tok", 8),
              )
              or 8
          )
          k_top = min(k_top, int(primary.shape[-1]))
          _, router_topk = jax.lax.top_k(jax.nn.softmax(primary.astype(jnp.float32), axis=-1), k_top)
          tap._tap(router_topk.astype(jnp.int32), sub_idx, "mlp.router_topk")
          if tap._active_forced_experts is None:
            tap._tap(router_topk.astype(jnp.int32), sub_idx, "mlp.selected_experts")
        elif mod_name == "mlp.routed_experts":
          fre = kwargs.get("forced_routed_experts")
          if fre is not None:
            tap._tap(jnp.asarray(fre, dtype=jnp.int32), sub_idx, "mlp.selected_experts")
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
      sig = inspect.signature(orig_layer_call)
      param_names = list(sig.parameters.keys())[1:]  # skip self

      def _wrapped_layer(layer_self, *args, **kwargs):
        bound = {}
        for idx, val in enumerate(args):
          if idx < len(param_names):
            bound[param_names[idx]] = val
        bound.update(kwargs)

        raw_in = bound.get("inputs", args[0] if args else None)
        x_in = raw_in[0] if isinstance(raw_in, tuple) else raw_in
        sub_idx = int(getattr(layer_self, "layer_idx", 0) or 0)
        tap._active_sublayer_idx = sub_idx
        if x_in is not None and x_in.ndim >= 2:
          tap._active_batch_shape = (int(x_in.shape[0]), int(x_in.shape[1]) if x_in.ndim >= 3 else 1)
        tap._active_forced_experts = bound.get("forced_routed_experts")

        if tap.role == "sampler" and x_in is not None:
          tap._active_seq_bounds = tap._extract_sampler_seq_bounds(
              x_in,
              bound.get("decoder_positions"),
              bound.get("attention_metadata"),
          )

        tap._tap(x_in, sub_idx, "layer_in")
        tagged = tap._tag_child_modules(layer_self, sub_idx)
        try:
          out = orig_layer_call(layer_self, *args, **kwargs)
        finally:
          for obj, attr in tagged:
            with contextlib.suppress(Exception):
              object.__delattr__(obj, attr)
        layer_out = out[0] if isinstance(out, tuple) else out
        tap._tap(layer_out, sub_idx, "layer_out")
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
      logits_slice_dim: int = 256,
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
        logits_slice_dim=logits_slice_dim,
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


def run_isolated_trainer_replay(
    trainer_model: Any,
    sampler_probes: dict[str, np.ndarray],
    *,
    target_next_tokens: np.ndarray | None = None,
    prompt_tokens_probe: np.ndarray | None = None,
    logits_slice_dim: int = 256,
    temperature: float = 1.0,
    layer_indices: set[int] | Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
  """Runs each Trainer submodule in isolation on the Sampler's exact captured inputs.

  For every layer `l` and submodule `m`, computes `m_trainer(x_sampler)` so that single-module
  kernel/quantization differences (`fused_moe` vs `gmm_v2`, `vllm_rpa` vs `flash`, GDN paged vs
  chunked) are measured with zero upstream error accumulation.
  """
  base = getattr(trainer_model, "base", trainer_model)
  decoder, num_layers, _, _, _ = discover_decoder_structure(trainer_model)
  cfg = getattr(decoder, "config", None)
  model_dtype = getattr(cfg, "dtype", jnp.bfloat16) if cfg is not None else jnp.bfloat16
  float32_gate = bool(getattr(cfg, "float32_gate_logits", True)) if cfg is not None else True
  float32_wsum = bool(getattr(cfg, "float32_weight_sum", True)) if cfg is not None else True
  gate_dtype = jnp.float32 if float32_gate else model_dtype
  allowed_layers = {int(i) for i in layer_indices} if layer_indices is not None else None

  probe_positions = np.asarray(sampler_probes.get("probe_positions", []), dtype=np.int32)
  prompt_len = int(sampler_probes.get("prompt_len", 0))
  is_contiguous_prefix = (
      probe_positions.size > 0
      and np.array_equal(probe_positions, np.arange(probe_positions.size, dtype=np.int32))
  )

  iso: dict[str, np.ndarray] = {
      "probe_positions": probe_positions,
      "prompt_len": np.int32(prompt_len),
      "gen_len": np.int32(sampler_probes.get("gen_len", 0)),
      "num_layers": np.int32(num_layers),
  }

  def _pad_seq_3d(arr: jax.Array, target_len: int) -> jax.Array:
    pad_n = target_len - int(arr.shape[1])
    if pad_n <= 0:
      return arr
    return jnp.pad(arr, ((0, 0), (0, pad_n), (0, 0)))

  def _safe_input_3d(np_arr: np.ndarray, dtype: Any, mesh: Any, b_size: int = 1) -> jax.Array:
    clean = np.nan_to_num(np.asarray(np_arr, dtype=np.float32), nan=0.0)
    tiled = np.repeat(clean[None, :, :], max(1, b_size), axis=0)
    return _to_mesh_array(tiled, dtype, mesh)

  jitted_mod_cache: dict[Any, Any] = {}

  def _call_mod(mod: Any, *args, **kwargs):
    if not is_tpu or not isinstance(mod, nnx.Module):
      return mod(*args, **kwargs)
    try:
      graphdef, state = nnx.split(mod)
      state_sig = tuple((getattr(x, "shape", None), getattr(x, "dtype", None)) for x in jax.tree.leaves(state))
      arg_sig = tuple((getattr(x, "shape", None), getattr(x, "dtype", None)) for x in args)
      kw_names = tuple(sorted(kwargs.keys()))
      kw_vals = tuple(kwargs[k] for k in kw_names)
      static_mask = tuple(isinstance(v, (bool, int, float, str, type(None))) for v in kw_vals)
      static_kws = tuple((k, v) for k, v, is_s in zip(kw_names, kw_vals, static_mask) if is_s)
      dyn_kw_names = tuple(k for k, is_s in zip(kw_names, static_mask) if not is_s)
      dyn_kw_vals = tuple(v for v, is_s in zip(kw_vals, static_mask) if not is_s)
      cache_key = (type(mod), state_sig, arg_sig, static_kws, dyn_kw_names)
      jitted = jitted_mod_cache.get(cache_key)
      if jitted is None:

        @jax.jit
        def _fn(st, a_tuple, d_tuple, _gdef=graphdef):
          m = nnx.merge(_gdef, st)
          call_kw = dict(static_kws)
          for k, v in zip(dyn_kw_names, d_tuple):
            call_kw[k] = v
          return m(*a_tuple, **call_kw)

        jitted = _fn
        jitted_mod_cache[cache_key] = jitted
      return jitted(state, args, dyn_kw_vals)
    except Exception:
      return mod(*args, **kwargs)

  with _model_mesh_context(decoder, base, cfg) as mesh:
    is_tpu = jax.default_backend() == "tpu"
    b_pad = 1
    if is_tpu and mesh is not None and hasattr(mesh, "shape"):
      for ax_name, ax_size in mesh.shape.items():
        if ax_name not in ("tensor", "model", "mlp", "heads", "kv"):
          b_pad *= int(ax_size)
      b_pad = max(1, b_pad)

    embedder = getattr(base, "token_embedder", getattr(decoder, "token_embedder", None))
    if prompt_tokens_probe is not None and embedder is not None and callable(embedder):
      with contextlib.suppress(Exception):
        tok_in = _to_mesh_array(np.repeat(np.asarray(prompt_tokens_probe, dtype=np.int32)[None, :], b_pad, axis=0), jnp.int32, mesh)
        emb = _call_mod(embedder, tok_in)
        iso["token_embedder"] = np.asarray(emb[0], dtype=np.float32)
    if "token_embedder" not in iso and "token_embedder" in sampler_probes:
      iso["token_embedder"] = np.array(sampler_probes["token_embedder"], dtype=np.float32, copy=True)

    for lyr_idx in range(num_layers):
      if allowed_layers is not None and lyr_idx not in allowed_layers:
        continue
      p = f"layer_{lyr_idx}"
      if f"{p}.layer_in" not in sampler_probes:
        continue
      tr_layer = get_layer_for_index(trainer_model, lyr_idx)
      if tr_layer is None:
        continue

      raw_layer_in = np.asarray(sampler_probes[f"{p}.layer_in"], dtype=np.float32)
      sa_layer_in = _safe_input_3d(raw_layer_in, model_dtype, mesh, b_pad)
      k_orig = int(sa_layer_in.shape[1])
      k_pad = max(256, ((k_orig + 255) // 256) * 256) if is_tpu and (k_orig % 256 != 0) else k_orig
      iso[f"{p}.layer_in"] = raw_layer_in.copy()

      # 1. input_layernorm (fed Sampler's layer_in)
      in_ln = getattr(tr_layer, "input_layernorm", None)
      if callable(in_ln):
        iso_norm1 = _call_mod(in_ln, sa_layer_in)
        iso[f"{p}.input_layernorm"] = np.asarray(iso_norm1[0], dtype=np.float32)
      else:
        iso_norm1 = sa_layer_in

      # 2. Attention sub-projections & core (fed Sampler's input_layernorm)
      sa_norm1 = (
          _safe_input_3d(sampler_probes[f"{p}.input_layernorm"], model_dtype, mesh, b_pad)
          if f"{p}.input_layernorm" in sampler_probes
          else iso_norm1
      )
      attn = getattr(tr_layer, "attention", None)
      iso_attn_out = None
      if attn is not None:
        for proj_name in ("in_proj_qkvz", "in_proj_ba"):
          proj_fn = getattr(attn, proj_name, None)
          if callable(proj_fn):
            with contextlib.suppress(Exception):
              proj_out = _call_mod(proj_fn, sa_norm1)
              iso[f"{p}.attn.{proj_name}"] = np.asarray(proj_out[0], dtype=np.float32)

        attn_inner = getattr(attn, "attention", attn)
        for proj_name in ("query", "key", "value"):
          proj_fn = getattr(attn_inner, proj_name, None)
          if callable(proj_fn):
            with contextlib.suppress(Exception):
              proj_out = _call_mod(proj_fn, sa_norm1)
              iso[f"{p}.attn.{proj_name}"] = np.asarray(
                  jnp.reshape(proj_out[0], (sa_norm1.shape[1], -1)), dtype=np.float32
              )

        if is_contiguous_prefix and callable(attn):
          with contextlib.suppress(Exception):
            sa_norm1_pad = _pad_seq_3d(sa_norm1, k_pad)
            seg_np = np.zeros((b_pad, k_pad), dtype=np.int32)
            seg_np[:, :k_orig] = 1
            pos_np = np.zeros((b_pad, k_pad), dtype=np.int32)
            pos_np[:, :k_orig] = np.arange(k_orig, dtype=np.int32)[None, :]
            seg_ids = _to_mesh_array(seg_np, jnp.int32, mesh)
            pos_ids = _to_mesh_array(pos_np, jnp.int32, mesh)
            sig = inspect.signature(attn.__call__)
            if "decoder_positions" in sig.parameters:
              attn_res = _call_mod(attn, sa_norm1_pad, seg_ids, pos_ids, True, "train")
            else:
              attn_res = _call_mod(attn, sa_norm1_pad, model_mode="train", decoder_segment_ids=seg_ids)
            attn_full = attn_res[0] if isinstance(attn_res, tuple) else attn_res
            iso_attn_out = attn_full[:, :k_orig]
            iso[f"{p}.attn.out"] = np.asarray(iso_attn_out[0], dtype=np.float32)

      if iso_attn_out is None and f"{p}.attn.out" in sampler_probes:
        iso_attn_out = _safe_input_3d(sampler_probes[f"{p}.attn.out"], model_dtype, mesh, b_pad)
        iso[f"{p}.attn.out"] = np.asarray(sampler_probes[f"{p}.attn.out"], dtype=np.float32).copy()

      # 3. post_attn_residual & post_attention_layernorm
      if iso_attn_out is not None:
        iso_post_attn_res = (sa_layer_in + iso_attn_out.astype(sa_layer_in.dtype)).astype(model_dtype)
        iso[f"{p}.post_attn_residual"] = np.asarray(iso_post_attn_res[0], dtype=np.float32)
      else:
        iso_post_attn_res = sa_layer_in

      sa_post_attn_res = (
          _safe_input_3d(sampler_probes[f"{p}.post_attn_residual"], model_dtype, mesh, b_pad)
          if f"{p}.post_attn_residual" in sampler_probes
          else iso_post_attn_res
      )
      post_ln = getattr(tr_layer, "post_attention_layernorm", None)
      if callable(post_ln):
        iso_norm2 = _call_mod(post_ln, sa_post_attn_res)
        iso[f"{p}.post_attention_layernorm"] = np.asarray(iso_norm2[0], dtype=np.float32)
      else:
        iso_norm2 = sa_post_attn_res

      # 4. MLP: Router Gate, Routed Experts, Shared Expert, Shared Expert Gate
      sa_norm2 = (
          _safe_input_3d(sampler_probes[f"{p}.post_attention_layernorm"], model_dtype, mesh, b_pad)
          if f"{p}.post_attention_layernorm" in sampler_probes
          else iso_norm2.astype(model_dtype)
      )
      sa_norm2_pad = _pad_seq_3d(sa_norm2, k_pad)
      mlp = getattr(tr_layer, "mlp", None)
      iso_mlp_out = None
      if mlp is not None:
        routed = getattr(mlp, "routed_experts", None)
        k_top = int(
            getattr(
                routed,
                "num_experts_per_tok",
                getattr(getattr(routed, "config", None), "num_experts_per_tok", getattr(cfg, "num_experts_per_tok", 8)),
            )
            or 8
        )
        raw_sel_exp = (
            np.asarray(sampler_probes[f"{p}.mlp.selected_experts"], dtype=np.int32)[:, :k_top]
            if f"{p}.mlp.selected_experts" in sampler_probes
            else None
        )
        sa_sel_exp = (
            _to_mesh_array(np.repeat(np.maximum(0, raw_sel_exp)[None, :, :], b_pad, axis=0), jnp.int32, mesh)
            if raw_sel_exp is not None
            else None
        )
        sa_sel_exp_pad = _pad_seq_3d(sa_sel_exp, k_pad) if sa_sel_exp is not None else None
        iso_routed_out = None
        if routed is not None:
          gate_fn = getattr(routed, "gate", None)
          if callable(gate_fn):
            with contextlib.suppress(Exception):
              g_res = _call_mod(gate_fn, sa_norm2)
              g_logits = g_res[0] if isinstance(g_res, tuple) else g_res
              iso[f"{p}.mlp.gate_logits"] = np.asarray(g_logits[0], dtype=np.float32)
              eff_k = min(k_top, int(g_logits.shape[-1]))
              _, r_topk = jax.lax.top_k(jax.nn.softmax(g_logits.astype(jnp.float32), axis=-1), eff_k)
              iso[f"{p}.mlp.router_topk"] = np.asarray(r_topk[0], dtype=np.int32)
              iso[f"{p}.mlp.selected_experts"] = (
                  raw_sel_exp.copy() if raw_sel_exp is not None else np.asarray(r_topk[0], dtype=np.int32)
              )
          if callable(routed):
            with contextlib.suppress(Exception):
              r_res = _call_mod(routed, sa_norm2_pad, forced_routed_experts=sa_sel_exp_pad)
              r_full = r_res[0] if isinstance(r_res, tuple) else r_res
              iso_routed_out = r_full[:, :k_orig]
              iso[f"{p}.mlp.routed_experts"] = np.asarray(iso_routed_out[0], dtype=np.float32)

        shared_fn = getattr(mlp, "shared_expert", None)
        shared_gate_fn = getattr(mlp, "shared_expert_gate", None)
        iso_shared_out = None
        iso_shared_gate_prob = None
        if callable(shared_fn):
          with contextlib.suppress(Exception):
            sig_s = inspect.signature(shared_fn.__call__)
            s_res = (
                _call_mod(shared_fn, sa_norm2, deterministic=True)
                if "deterministic" in sig_s.parameters
                else _call_mod(shared_fn, sa_norm2)
            )
            iso_shared_out = s_res[0] if isinstance(s_res, tuple) else s_res
            iso[f"{p}.mlp.shared_expert"] = np.asarray(iso_shared_out[0], dtype=np.float32)

        if callable(shared_gate_fn):
          with contextlib.suppress(Exception):
            sg_logits = _call_mod(shared_gate_fn, sa_norm2)
            sg_logits = sg_logits[0] if isinstance(sg_logits, tuple) else sg_logits
            iso_shared_gate_prob = jax.nn.sigmoid(sg_logits.astype(jnp.float32))
            iso[f"{p}.mlp.shared_expert_gate"] = np.asarray(iso_shared_gate_prob[0], dtype=np.float32)

        if iso_routed_out is not None and iso_shared_out is not None and iso_shared_gate_prob is not None:
          if float32_wsum:
            iso_mlp_out = (
                iso_routed_out.astype(jnp.float32) + iso_shared_gate_prob * iso_shared_out.astype(jnp.float32)
            ).astype(model_dtype)
          else:
            iso_mlp_out = iso_routed_out + iso_shared_gate_prob.astype(model_dtype) * iso_shared_out
          iso[f"{p}.mlp.out"] = np.asarray(iso_mlp_out[0], dtype=np.float32)
        elif iso_routed_out is not None and shared_fn is None:
          iso_mlp_out = iso_routed_out
          iso[f"{p}.mlp.out"] = np.asarray(iso_mlp_out[0], dtype=np.float32)
        elif callable(mlp):
          with contextlib.suppress(Exception):
            m_res = _call_mod(mlp, sa_norm2_pad.astype(model_dtype), deterministic=True, forced_routed_experts=sa_sel_exp_pad)
            m_full = m_res[0] if isinstance(m_res, tuple) else m_res
            iso_mlp_out = m_full[:, :k_orig]
            iso[f"{p}.mlp.out"] = np.asarray(iso_mlp_out[0], dtype=np.float32)

      if iso_mlp_out is not None:
        iso_layer_out = (sa_post_attn_res + iso_mlp_out.astype(sa_post_attn_res.dtype)).astype(model_dtype)
        iso[f"{p}.layer_out"] = np.asarray(iso_layer_out[0], dtype=np.float32)

    # 5. Boundary output head in isolation:
    #    - decoder_norm fed Sampler's last layer_out
    #    - logits_dense fed Sampler's decoder_norm
    last_layer_key = f"layer_{num_layers - 1}.layer_out"
    decoder_norm_fn = getattr(decoder, "decoder_norm", None)
    if last_layer_key in sampler_probes and callable(decoder_norm_fn):
      sa_last = _safe_input_3d(sampler_probes[last_layer_key], model_dtype, mesh, b_pad)
      iso_dec_norm = _call_mod(decoder_norm_fn, sa_last)
      iso["decoder_norm"] = np.asarray(iso_dec_norm[0], dtype=np.float32)

    sa_dec_norm = (
        _safe_input_3d(sampler_probes["decoder_norm"], model_dtype, mesh, b_pad)
        if "decoder_norm" in sampler_probes
        else (_safe_input_3d(iso["decoder_norm"], model_dtype, mesh, b_pad) if "decoder_norm" in iso else None)
    )
    if sa_dec_norm is not None:
      logits_3d = None
      logits_dense_fn = getattr(decoder, "logits_dense", None)
      apply_head_fn = getattr(decoder, "apply_output_head", None)
      if callable(logits_dense_fn) and not getattr(cfg, "logits_via_embedding", False):
        logits_3d = _call_mod(logits_dense_fn, sa_dec_norm)
        if getattr(cfg, "cast_logits_to_fp32", True):
          logits_3d = logits_3d.astype(jnp.float32)
      elif callable(apply_head_fn):
        sig = inspect.signature(apply_head_fn)
        if "normalize_y" in sig.parameters:
          logits_3d = apply_head_fn(embedder, sa_dec_norm, True, "train", normalize_y=False)
        elif last_layer_key in sampler_probes:
          logits_3d = apply_head_fn(
              embedder,
              _safe_input_3d(sampler_probes[last_layer_key], model_dtype, mesh, b_pad),
              True,
              "train",
          )
      if logits_3d is not None:
        l_head, l_sum, l_top1 = _compute_logits_summary_and_head(
            logits_3d[:1],
            target_tokens_2d=target_next_tokens,
            logits_slice_dim=logits_slice_dim,
            temperature=temperature,
            prompt_len=prompt_len,
            probe_positions=probe_positions,
        )
        iso["logits_head"] = l_head
        iso["logits_summary"] = l_sum
        iso["logits_top1"] = l_top1

  return iso


# --------------------------------------------------------- Divergence Metrics & Reporting


def _align_feature_shapes(sa: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  if sa.shape == tr.shape or sa.ndim != 2 or tr.ndim != 2:
    return sa, tr
  k = sa.shape[0]
  d_sa, d_tr = sa.shape[1], tr.shape[1]
  if d_tr > d_sa and d_tr % d_sa == 0:
    rep = d_tr // d_sa
    for hd in (256, 128, 64, 32, d_sa):
      if d_sa % hd == 0:
        n_heads = d_sa // hd
        tr_4d = tr.reshape(k, n_heads, rep, hd)
        if np.allclose(np.nan_to_num(tr_4d[:, :, 0, :]), np.nan_to_num(tr_4d[:, :, -1, :]), atol=1e-3):
          return sa, tr_4d[:, :, 0, :].reshape(k, d_sa)
    return sa, tr.reshape(k, d_sa, rep)[..., 0]
  if d_sa > d_tr and d_sa % d_tr == 0:
    tr_aligned, sa_aligned = _align_feature_shapes(tr, sa)
    return sa_aligned, tr_aligned
  min_d = min(d_sa, d_tr)
  return sa[:, :min_d], tr[:, :min_d]


def compute_tensor_divergence(ref_sa: np.ndarray, cand_tr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
  """Computes numerical divergence metrics `cand_tr - ref_sa` over valid finite positions."""
  sa = np.asarray(ref_sa, dtype=np.float64)
  tr = np.asarray(cand_tr, dtype=np.float64)
  min_rows = min(sa.shape[0], tr.shape[0])
  sa, tr = sa[:min_rows], tr[:min_rows]
  sa, tr = _align_feature_shapes(sa, tr)
  if mask is not None:
    m = np.asarray(mask[:min_rows], dtype=bool)
    sa, tr = sa[m], tr[m]

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

  sa_v = sa[valid]
  tr_v = tr[valid]
  diff = tr_v - sa_v
  abs_diff = np.abs(diff)

  sa_norm = float(np.linalg.norm(sa_v))
  tr_norm = float(np.linalg.norm(tr_v))
  diff_norm = float(np.linalg.norm(diff))
  rel_l2 = diff_norm / max(sa_norm, 1e-12)
  denom = sa_norm * tr_norm
  cos_sim = 1.0 if diff_norm == 0.0 else (float(np.dot(sa_v, tr_v) / denom) if denom > 1e-24 else 1.0)
  cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

  return {
      "rel_l2": float(rel_l2),
      "max_abs": float(np.max(abs_diff)),
      "mean_abs": float(np.mean(abs_diff)),
      "median_abs": float(np.median(abs_diff)),
      "p99_abs": float(np.percentile(abs_diff, 99)),
      "cos_sim": cos_sim,
      "signed_bias": float(np.mean(diff)),
      "n_elems": int(sa_v.size),
  }


def compute_router_agreement(sa_topk: np.ndarray, tr_topk: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
  """Computes top-1 exact match and top-k set Jaccard/overlap agreement between `[N, K]` expert index arrays."""
  sa = np.asarray(sa_topk, dtype=np.int32)
  tr = np.asarray(tr_topk, dtype=np.int32)
  n = min(sa.shape[0], tr.shape[0])
  sa, tr = sa[:n], tr[:n]
  if mask is not None:
    m = np.asarray(mask[:n], dtype=bool)
    sa, tr = sa[m], tr[m]
  valid_rows = (sa[:, 0] >= 0) & (tr[:, 0] >= 0) if sa.ndim == 2 and sa.size > 0 else np.zeros((0,), dtype=bool)
  sa, tr = sa[valid_rows], tr[valid_rows]
  if sa.shape[0] == 0:
    return {"top1_agree": 1.0, "topk_overlap": 1.0, "topk_jaccard": 1.0}

  top1_agree = float(np.mean(sa[:, 0] == tr[:, 0]))
  k = sa.shape[1]
  overlaps = []
  jaccards = []
  for r in range(sa.shape[0]):
    s_set = set(int(x) for x in sa[r] if x >= 0)
    t_set = set(int(x) for x in tr[r] if x >= 0)
    inter = len(s_set & t_set)
    union = len(s_set | t_set)
    overlaps.append(inter / max(k, 1))
    jaccards.append(inter / max(union, 1))
  return {
      "top1_agree": top1_agree,
      "topk_overlap": float(np.mean(overlaps)),
      "topk_jaccard": float(np.mean(jaccards)),
  }


def _module_family(mod_name: str) -> str:
  if mod_name in ("token_embedder", "decoder_norm", "logits_head", "logits_summary"):
    return mod_name
  if mod_name.startswith("attn."):
    return mod_name
  if mod_name.startswith("mlp."):
    return mod_name
  return mod_name


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

  prompt_mask = probe_positions < prompt_len if probe_positions.size else None
  decode_mask = probe_positions >= prompt_len if probe_positions.size else None

  if probe_layers == "all" or not probe_layers:
    visible_layers = set(range(num_layers))
  else:
    visible_layers = {int(x.strip()) for x in probe_layers.split(",") if x.strip().isdigit()}

  records: list[dict[str, Any]] = []
  family_Buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

  def _eval_key(key: str, layer_idx: int | None, mod_name: str):
    if key not in sampler_probes or key not in trainer_cum_probes:
      return
    sa_arr = sampler_probes[key]
    tr_cum_arr = trainer_cum_probes[key]
    if np.issubdtype(sa_arr.dtype, np.integer):
      return
    cum_all = compute_tensor_divergence(sa_arr, tr_cum_arr)
    cum_prompt = compute_tensor_divergence(sa_arr, tr_cum_arr, mask=prompt_mask) if prompt_mask is not None and np.any(prompt_mask) else None
    cum_decode = compute_tensor_divergence(sa_arr, tr_cum_arr, mask=decode_mask) if decode_mask is not None and np.any(decode_mask) else None

    if trainer_iso_probes is not None and key in trainer_iso_probes:
      tr_iso_arr = trainer_iso_probes[key]
      iso_all = compute_tensor_divergence(sa_arr, tr_iso_arr)
      iso_prompt = compute_tensor_divergence(sa_arr, tr_iso_arr, mask=prompt_mask) if prompt_mask is not None and np.any(prompt_mask) else None
      iso_decode = compute_tensor_divergence(sa_arr, tr_iso_arr, mask=decode_mask) if decode_mask is not None and np.any(decode_mask) else None
    else:
      iso_all = iso_prompt = iso_decode = None

    router_info = None
    if layer_idx is not None and mod_name in ("mlp.gate_logits", "mlp.routed_experts"):
      r_key = f"layer_{layer_idx}.mlp.router_topk"
      s_key = f"layer_{layer_idx}.mlp.selected_experts"
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
        "cumulative": {"all": cum_all, "prompt": cum_prompt, "decode": cum_decode},
        "isolated": {"all": iso_all, "prompt": iso_prompt, "decode": iso_decode} if iso_all is not None else None,
        "router": router_info,
    }
    records.append(rec)
    family_Buckets[_module_family(mod_name)].append(rec)

  _eval_key("token_embedder", None, "token_embedder")
  for lyr in range(num_layers):
    for mod_name in LAYER_MODULES:
      if mod_name in ("mlp.router_topk", "mlp.selected_experts"):
        continue
      _eval_key(f"layer_{lyr}.{mod_name}", lyr, mod_name)
  for b_mod in ("decoder_norm", "logits_head", "logits_summary"):
    _eval_key(b_mod, None, b_mod)

  # Summarize by module family
  family_summary: dict[str, dict[str, float]] = {}
  for fam, recs in family_Buckets.items():
    iso_rels = [r["isolated"]["all"]["rel_l2"] for r in recs if r["isolated"] is not None]
    iso_maxs = [r["isolated"]["all"]["max_abs"] for r in recs if r["isolated"] is not None]
    cum_rels = [r["cumulative"]["all"]["rel_l2"] for r in recs]
    cum_maxs = [r["cumulative"]["all"]["max_abs"] for r in recs]
    cum_coss = [r["cumulative"]["all"]["cos_sim"] for r in recs]
    biases = [r["cumulative"]["all"]["signed_bias"] for r in recs]
    family_summary[fam] = {
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

  # Rank worst isolated & cumulative single-module divergences (excluding raw logits_summary scale)
  internal_recs = [r for r in records if r["module"] not in ("logits_summary", "logits_head", "layer_in")]
  top_isolated = sorted(
      [r for r in internal_recs if r["isolated"] is not None],
      key=lambda r: r["isolated"]["all"]["rel_l2"],
      reverse=True,
  )[:5]
  top_cumulative = sorted(
      internal_recs,
      key=lambda r: r["cumulative"]["all"]["rel_l2"],
      reverse=True,
  )[:5]

  logits_top1_agree = None
  if "logits_top1" in sampler_probes and "logits_top1" in trainer_cum_probes:
    sa_t1 = sampler_probes["logits_top1"]
    tr_t1 = trainer_cum_probes["logits_top1"]
    n_t1 = min(len(sa_t1), len(tr_t1))
    logits_top1_agree = float(np.mean(sa_t1[:n_t1] == tr_t1[:n_t1])) if n_t1 > 0 else 1.0

  logprob_div = None
  if "logits_summary" in sampler_probes and "logits_summary" in trainer_cum_probes:
    sa_lp = sampler_probes["logits_summary"][:, 2]
    tr_cum_lp = trainer_cum_probes["logits_summary"][:, 2]
    tr_iso_lp = trainer_iso_probes["logits_summary"][:, 2] if (trainer_iso_probes and "logits_summary" in trainer_iso_probes) else None
    logprob_div = {
        "cumulative": compute_tensor_divergence(sa_lp, tr_cum_lp),
        "isolated": compute_tensor_divergence(sa_lp, tr_iso_lp) if tr_iso_lp is not None else None,
        "top1_agree": logits_top1_agree,
    }

  # Print formatted report
  n_p = int(np.sum(prompt_mask)) if prompt_mask is not None else 0
  n_d = int(np.sum(decode_mask)) if decode_mask is not None else 0
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
    c_m = r["cumulative"]["all"]
    i_m = r["isolated"]["all"] if r["isolated"] is not None else None
    iso_r_str = f"{i_m['rel_l2']:10.3e}" if i_m is not None else f"{'N/A':>10}"
    iso_m_str = f"{i_m['max_abs']:10.3e}" if i_m is not None else f"{'N/A':>10}"
    r_info = r["router"]
    if r_info is not None:
      r_src = r_info["iso"] if r_info.get("iso") is not None else r_info["cum"]
      r_str = f"{r_src['top1_agree']:9.1%}"
    else:
      r_str = f"{'-':>10}"
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
      im = r["isolated"]["all"]
      cm = r["cumulative"]["all"]
      l_tag = f"layer_{r['layer']:02d}" if r["layer"] is not None else "boundary"
      print(
          f"    {idx}. {l_tag}.{r['module']:<24} iso_rel_l2={im['rel_l2']:.4e}  iso_max|d|={im['max_abs']:.4e}  "
          f"(cum_rel_l2={cm['rel_l2']:.4e}, cos_sim={im['cos_sim']:.6f})"
      )

  if logprob_div is not None:
    lp_c = logprob_div["cumulative"]
    lp_i = logprob_div["isolated"]
    iso_lp_str = f"{lp_i['mean_abs']:.4e} (bias={lp_i['signed_bias']:+.4e})" if lp_i else "N/A"
    t1_str = f"{logits_top1_agree:.2%}" if logits_top1_agree is not None else "N/A"
    print(
        f"\n  --- Output Head Logprob Impact ---\n"
        f"    Isolated Head |dlogp| mean  : {iso_lp_str}\n"
        f"    Cumulative    |dlogp| mean  : {lp_c['mean_abs']:.4e} (max={lp_c['max_abs']:.4e}, signed_bias={lp_c['signed_bias']:+.4e})\n"
        f"    Probed Token Top-1 Agreement: {t1_str}"
    )
  print("=" * 84 + "\n", flush=True)

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
              "iso_rel_l2": r["isolated"]["all"]["rel_l2"],
              "iso_max_abs": r["isolated"]["all"]["max_abs"],
              "cum_rel_l2": r["cumulative"]["all"]["rel_l2"],
          }
          for r in top_isolated
      ],
      "top_cumulative_bottlenecks": [
          {
              "key": r["key"],
              "layer": r["layer"],
              "module": r["module"],
              "cum_rel_l2": r["cumulative"]["all"]["rel_l2"],
              "cum_max_abs": r["cumulative"]["all"]["max_abs"],
          }
          for r in top_cumulative
      ],
      "logprob_divergence": logprob_div,
  }

  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "module_divergence_report.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    npz_payload = {
        "probe_positions": probe_positions,
        "layers": np.array([-1 if r["layer"] is None else r["layer"] for r in records], dtype=np.int32),
        "modules": np.array([r["module"] for r in records]),
        "cum_rel_l2": np.array([r["cumulative"]["all"]["rel_l2"] for r in records], dtype=np.float32),
        "cum_max_abs": np.array([r["cumulative"]["all"]["max_abs"] for r in records], dtype=np.float32),
        "cum_cos_sim": np.array([r["cumulative"]["all"]["cos_sim"] for r in records], dtype=np.float32),
        "cum_signed_bias": np.array([r["cumulative"]["all"]["signed_bias"] for r in records], dtype=np.float32),
        "iso_rel_l2": np.array(
            [r["isolated"]["all"]["rel_l2"] if r["isolated"] is not None else np.nan for r in records], dtype=np.float32
        ),
        "iso_max_abs": np.array(
            [r["isolated"]["all"]["max_abs"] if r["isolated"] is not None else np.nan for r in records], dtype=np.float32
        ),
    }
    np.savez(os.path.join(out_dir, "module_divergence_metrics.npz"), **npz_payload)

  return report


def parse_probe_layers(spec: str | Sequence[int] | None) -> set[int] | None:
  """Parses a `--probe-layers` specification (`'all'`, `'0,1,2,3,39'`, or a sequence of ints)."""
  if spec is None:
    return None
  if isinstance(spec, str):
    s = spec.strip().lower()
    if not s or s == "all":
      return None
    out: set[int] = set()
    for part in s.split(","):
      part = part.strip()
      if not part:
        continue
      if "-" in part:
        lo_s, hi_s = part.split("-", 1)
        out.update(range(int(lo_s), int(hi_s) + 1))
      else:
        out.add(int(part))
    return out
  return {int(i) for i in spec}


def sync_trainer_weights_to_sampler(
    trainer_model: Any,
    sampler_or_model: Any,
    *,
    scan_axis: int | None = None,
    delete_dst_buffers: bool = False,
) -> None:
  """Synchronizes Trainer weights into Sampler via Tunix's `transfer_state_directly`.

  Handles MoE weight fusion (`wi_0` + `wi_1` -> `wi`), KV head repetition (e.g.
  `trainer_tp=2` `base_num_kv_heads=2` -> `sampler_tp=1`), scanned layer unrolling,
  and dtype casting, matching the production weight sync in `mlperf_base.sh`.
  """
  from tunix.generate import utils as tunix_gen_utils

  src_state = nnx.state(trainer_model) if isinstance(trainer_model, nnx.Module) else trainer_model
  if hasattr(sampler_or_model, "load_checkpoint") and hasattr(sampler_or_model, "transformer_state"):
    sampler_or_model.load_checkpoint(src_state)
    return

  decoder, _, _, _, _ = discover_decoder_structure(trainer_model) if isinstance(trainer_model, nnx.Module) else (None, 0, False, 1, [])
  if scan_axis is None:
    cfg = getattr(decoder, "config", None) if decoder is not None else None
    scan_axis = int(getattr(cfg, "param_scan_axis", 0) if cfg is not None and hasattr(cfg, "param_scan_axis") else 1)

  dst_state = nnx.state(sampler_or_model)
  tunix_gen_utils.transfer_state_directly(
      src_state=src_state,
      dst_state=dst_state,
      reshard_fn=lambda source, target: jax.tree.map(lambda x, _: x, source, target),
      scan_axis=scan_axis,
      delete_dst_buffers=delete_dst_buffers,
  )
  nnx.update(sampler_or_model, dst_state)

