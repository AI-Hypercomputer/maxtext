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
"""Convergence / OOB parity test for the E2E mlperf_35b_128_v5p.sh + mlperf_base.sh setup.

Compares bf16, fp8, and fp8moe sampler/trainer combinations by reusing Tunix's
VllmSampler, TrainerWorker, GRPOAdapter, and BatchAssembler primitives.
"""

import argparse
import contextlib
import gc
import glob
import json
import os
import subprocess
import sys
import time
from unittest import mock
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAXTEXT_ROOT = os.environ.get("MAXTEXT_ROOT") or os.path.dirname(os.path.dirname(_THIS_DIR))
for _cand in (
    _THIS_DIR,
    os.path.join(_MAXTEXT_ROOT, "src"),
    os.environ.get("TUNIX_ROOT"),
    os.path.join(os.path.dirname(_MAXTEXT_ROOT), "tunix"),
    "/app/tunix",
    "/workspace/tunix",
    "/tpu/tunix",
):
  if _cand and os.path.isdir(_cand) and _cand not in sys.path:
    sys.path.insert(0, _cand)

import module_divergence_probe as mdp

MODEL_HF_BF16 = os.environ.get("MAXTEXT_HF_BF16", "Qwen/Qwen3.5-35B-A3B")
MODEL_MAXTEXT_BF16 = os.environ.get("MAXTEXT_MODEL_BF16", "qwen3.5-35b-a3b")
CKPT_BF16 = os.environ.get("MAXTEXT_CKPT_BF16", "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items")

MODEL_HF_FP8 = os.environ.get("MAXTEXT_HF_FP8", "Qwen/Qwen3.5-35B-A3B-FP8")
MODEL_MAXTEXT_FP8 = os.environ.get("MAXTEXT_MODEL_FP8", "qwen3.5-35b-a3b-fp8")
CKPT_FP8 = os.environ.get("MAXTEXT_CKPT_FP8", "gs://cloud-devkit/users/wenxindong/ckpt/qwen3.5-35b-a3b-fp8/unscanned/0/items")

VALID_MODES = ("bf16", "fp8", "fp8moe", "fp8_ckpt", "fp8_serve", "fp8_moe", "fp8_moe_native", "int8_moe")
FP8_CKPT_MODES = ("fp8", "fp8_ckpt", "fp8_serve")
IN_PLACE_MOE_MODES = ("fp8moe", "fp8_moe", "fp8_moe_native", "int8_moe")
VALID_SCALE_MODES = ("per_channel", "subchannel128", "block128")
RATIO_MIN, RATIO_MAX = 0.999, 1.002
SEQ_ERR_THRESHOLD = 2.0
LOGPS_CHUNK_SIZE = 512
t0 = time.time()


def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def set_tpu_env(hf_home):
  """Sets runtime environment variables matching mlperf_35b_128_v5p.sh + mlperf_base.sh."""
  os.environ["NEW_MODEL_DESIGN"] = "1"
  os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
  for k, v in {
      "HF_HOME": hf_home,
      "HF_HUB_OFFLINE": "1",
      "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
      "FLOAT32_GATE_LOGITS": "true",
      "FLOAT32_LOGITS": "true",
      "TRAINER_MAXTEXT_ATTENTION": "flash",
      "ATTN_BUCKETIZED_NUM_REQS": "true",
      "ATTN_CUSTOM_NUM_REQS_BUCKETS": "4",
      "ONEHOT_MOE_PERMUTE_THRESHOLD": "32768",
      "VLLM_MOE_CHUNK_SIZE": "256",
      "SLICE_ROPE_CACHE": "1",
      "DP_SCHED_BATCH_PREFILL": "false",
      "NUM_PRECOMPILE_WORKERS": "8",
      "SKIP_JAX_PRECOMPILE": "1",
      "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
      "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
      "MAMBA_CACHE_MODE": "align",
      "VLLM_MAMBA_CACHE_MODE": "align",
      "LIBTPU_INIT_ARGS": (
          " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
          " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
          " --xla_tpu_ars_combiner_threshold_in_bytes=0"
          " --xla_tpu_enable_async_collective_merger=false"
          " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
      ),
  }.items():
    os.environ.setdefault(k, v)


def get_tokenizer(hf_home):
  from transformers import AutoTokenizer

  for name in ("models--Qwen--Qwen3.5-35B-A3B", "models--Qwen--Qwen3.5-35B-A3B-FP8"):
    matches = glob.glob(os.path.join(hf_home, "hub", name, "snapshots", "*"))
    if matches:
      return AutoTokenizer.from_pretrained(matches[0], trust_remote_code=True), matches[0]
  return AutoTokenizer.from_pretrained(MODEL_HF_BF16, trust_remote_code=True), MODEL_HF_BF16


# --------------------------------------------------------- Quantization & Model Audit


def _iter_decoder_layers(decoder):
  """Yields `(name, layer)` across both unscanned (`layers_i`) and scanned (`layers.local_layers`/`global_layer` or `layers.layer_i`) decoders."""
  num_layers = getattr(getattr(decoder, "config", None), "num_decoder_layers", 40)
  for i in range(num_layers):
    layer = getattr(decoder, f"layers_{i}", None)
    if layer is not None:
      yield f"layers_{i}", layer
  scanned = getattr(decoder, "layers", None)
  if scanned is not None:
    yielded_scanned = False
    for idx, layer in enumerate(getattr(scanned, "local_layers", None) or ()):
      if layer is not None:
        yielded_scanned = True
        yield f"layers.local_layers_{idx}", layer
    if getattr(scanned, "global_layer", None) is not None:
      yielded_scanned = True
      yield "layers.global_layer", scanned.global_layer
    if not yielded_scanned:
      for i in range(num_layers):
        layer = getattr(scanned, f"layer_{i}", None)
        if layer is not None:
          yield f"layers.layer_{i}", layer
  rem = getattr(decoder, "layers_remainder", None)
  if rem is not None:
    for i in range(num_layers):
      layer = getattr(rem, f"layers_{i}", None)
      if layer is not None:
        yield f"layers_remainder.layers_{i}", layer


def quantize_moe_fp8(model, scale_mode="per_channel", weight_qtype=None, set_serve_quant=False, block_size=128):
  """Quantize RoutedMoE weights in-place via MaxText's quantize_weight_for_fused_moe or qwix.pallas.quantize."""
  from flax import nnx
  import jax
  import jax.numpy as jnp
  from maxtext.layers import quantizations
  import qwix
  import qwix.pallas as qpl

  if scale_mode not in VALID_SCALE_MODES:
    raise ValueError(f"Unsupported scale_mode={scale_mode!r}; expected one of {VALID_SCALE_MODES}")
  qtype = jnp.dtype(weight_qtype) if weight_qtype is not None else jnp.dtype(jnp.float8_e4m3fn)
  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  count = 0
  for _, layer in _iter_decoder_layers(decoder):
    routed = getattr(getattr(layer, "mlp", None), "routed_experts", None)
    if routed is None:
      continue
    wi_axes, wo_axes = getattr(routed, "wi_kernel_axes", None), getattr(routed, "wo_kernel_axes", None)
    for w_name, s_name, w_ax in [("wo", "wo_scale", wo_axes), ("wi", "wi_scale", wi_axes), ("wi_0", "wi_0_scale", wi_axes), ("wi_1", "wi_1_scale", wi_axes)]:
      p = getattr(routed, w_name, None)
      if p is None:
        continue
      w_orig = p[...]
      K, N = w_orig.shape[-2], w_orig.shape[-1]

      def _quant_3d(w3, _k=K, _n=N):
        if scale_mode == "block128" and _k % block_size == 0 and _n % block_size == 0:
          q = qpl.quantize(w3, qtype, channelwise_axes=(0,), tiled_axes={1: block_size, 2: block_size}, scale_dtype=jnp.float32)
          return q.qvalue, q.scale
        tile_size = block_size if (scale_mode == "subchannel128" and _k % block_size == 0) else None
        qw3, s4 = quantizations.quantize_weight_for_fused_moe(w3, qwix.QtRule(weight_qtype=qtype, tile_size=tile_size))
        return qw3, jnp.squeeze(s4, axis=2)

      qw, qs = _quant_3d(w_orig) if w_orig.ndim == 3 else jax.vmap(_quant_3d)(w_orig)
      if w_ax and len(w_ax) >= 3:
        s_ax = tuple(None if idx == len(w_ax) - 2 else ax for idx, ax in enumerate(w_ax)) if qs.shape[-2] == 1 else tuple(w_ax)
      else:
        s_ax = None
      setattr(routed, w_name, nnx.data(nnx.Param(qw, out_sharding=w_ax)))
      setattr(routed, s_name, nnx.data(nnx.Param(qs, out_sharding=s_ax)))
    routed.weight_dtype = qtype
    if set_serve_quant:
      routed.quant = quantizations.ServeFp8WeightQuantization()
    count += 1
  log(f"quantize_moe_fp8: quantized {count} RoutedMoE layers (dtype={qtype}, scale={scale_mode}, serve={set_serve_quant})")
  return count


def audit_model(model, role: str, expected_mode: str, out_dir: str):
  """Inspect live NNX parameters and execution kernel paths."""
  from maxtext.common import common_types as ctypes
  from maxtext.layers import quantizations

  def _info(p):
    if p is None:
      return None
    arr = p[...] if hasattr(p, "__getitem__") else getattr(p, "value", p)
    s = np.asarray(arr[tuple(slice(0, min(d, 8)) for d in arr.shape)], dtype=np.float32)
    return {"dtype": str(arr.dtype), "shape": [int(d) for d in arr.shape], "sample_absmax": float(np.max(np.abs(s))), "sample_min": float(np.min(s)), "sample_max": float(np.max(s))}

  base = getattr(model, "base", model)
  decoder = getattr(base, "decoder", base)
  cfg, l0, l3 = getattr(decoder, "config", None), getattr(decoder, "layers_0", None), getattr(decoder, "layers_3", None)
  scanned = getattr(decoder, "layers", None)
  if l0 is None and scanned is not None:
    local_layers = getattr(scanned, "local_layers", None)
    l0 = local_layers[0] if local_layers else getattr(scanned, "layer_0", None)
  if l3 is None and scanned is not None:
    l3 = getattr(scanned, "global_layer", None) or getattr(scanned, "layer_3", None)
  routed0, shared0 = getattr(getattr(l0, "mlp", None), "routed_experts", None), getattr(getattr(l0, "mlp", None), "shared_expert", None)
  qkvz_mod = getattr(getattr(l0, "attention", None), "in_proj_qkvz", None)
  attn3_wrap = getattr(l3, "attention", None) if l3 is not None else None
  attn3 = getattr(attn3_wrap, "attention", attn3_wrap)

  wi_attr = "wi" if getattr(routed0, "wi", None) is not None else "wi_0"
  wi_s_attr = "wi_scale" if getattr(routed0, "wi_scale", None) is not None else "wi_0_scale"
  modules = {
      name: {"weight": _info(getattr(mod, w_a, None)), "scale": _info(getattr(mod, s_a, None))}
      for name, mod, w_a, s_a in [
          ("layers_0.gdn.in_proj_qkvz", qkvz_mod, "kernel", "kernel_scale"),
          ("layers_0.mlp.shared_expert.wi_0", getattr(shared0, "wi_0", None), "kernel", "kernel_scale"),
          ("layers_0.mlp.routed_experts.wi", routed0, wi_attr, wi_s_attr),
          ("layers_0.mlp.routed_experts.wo", routed0, "wo", "wo_scale"),
          ("layers_3.attn.query", getattr(attn3, "query", None), "kernel", "kernel_scale"),
      ]
  }
  wi_info, wi_scale_info = modules["layers_0.mlp.routed_experts.wi"]["weight"], modules["layers_0.mlp.routed_experts.wi"]["scale"]
  qkvz_info, qkvz_scale_info = modules["layers_0.gdn.in_proj_qkvz"]["weight"], modules["layers_0.gdn.in_proj_qkvz"]["scale"]

  is_fused_moe = getattr(cfg, "attention", "") in ("vllm_rpa", "vllm_batched_rpa") and not getattr(routed0, "is_hash_routing", False)
  is_fp8_moe = getattr(cfg, "fp8_moe", False) or ctypes.is_fp8_dtype(getattr(routed0, "weight_dtype", None)) or isinstance(getattr(routed0, "quant", None), quantizations.ServeFp8WeightQuantization)
  wi_is_fp8, wi_is_int8 = (wi_info is not None and "float8" in wi_info["dtype"]), (wi_info is not None and "int8" in wi_info["dtype"])

  if is_fused_moe:
    if is_fp8_moe and wi_is_fp8 and wi_scale_info is not None:
      moe_path = f"FUSED_MOE_NATIVE_FP8 (tpu_inference, scale={wi_scale_info['shape']})"
    elif wi_is_int8 and wi_scale_info is not None:
      moe_path = f"FUSED_MOE_DEQUANT_TO_BF16 (dequantize_weight -> tpu_inference, scale={wi_scale_info['shape']})"
    else:
      moe_path = "FUSED_MOE_BF16 (tpu_inference)"
  else:
    scheme, axis = quantizations.infer_scale_granularity(tuple(wi_scale_info["shape"][-2:])) if wi_scale_info else (None, None)
    native_gmm = (
        isinstance(getattr(routed0, "quant", None), quantizations.ServeFp8WeightQuantization)
        and getattr(cfg, "sparse_matmul", False)
        and getattr(cfg, "use_gmm_v2", False)
        and not getattr(cfg, "prefuse_moe_weights", False)
        and (scheme == "per_tensor" or (scheme == "per_channel" and axis == 1))
    )
    if native_gmm and wi_is_fp8 and wi_scale_info is not None:
      moe_path = f"GMM_V2_NATIVE_FP8 (qpl.QArray, scheme={scheme}, scale={wi_scale_info['shape']})"
    elif wi_is_fp8 or wi_is_int8 or wi_scale_info is not None:
      moe_path = f"GMM_V2_DEQUANT_TO_BF16 (dequantize_weight -> BF16 gmm_v2, scheme={scheme}, scale={wi_scale_info['shape'] if wi_scale_info else None})"
    else:
      moe_path = "GMM_V2_PURE_BF16 (BF16 gmm_v2)"

  if isinstance(getattr(qkvz_mod, "quant", None), quantizations.ServeFp8WeightQuantization) and qkvz_info and "float8" in qkvz_info["dtype"] and qkvz_scale_info:
    dense_path = "DENSE_NATIVE_FP8 (qwix W8A8 on 1D-contracted linears; multi-axis dequantizes to BF16)"
  elif qkvz_scale_info or (qkvz_info and "float8" in qkvz_info["dtype"]):
    dense_path = f"DENSE_DEQUANT_TO_BF16 (dequantize_weight -> BF16 dot_general, scale={qkvz_scale_info['shape'] if qkvz_scale_info else None})"
  else:
    dense_path = "DENSE_PURE_BF16 (BF16 dot_general)"

  report = {
      "role": role,
      "expected_mode": expected_mode,
      "config": {k: getattr(cfg, k, None) if k in ("model_name", "attention") else str(getattr(cfg, k, None)) for k in ("model_name", "attention", "weight_dtype", "dtype", "quantization")},
      "execution_paths": {"routed_moe": moe_path, "dense_linear": dense_path},
      "modules": modules,
  }
  print(f"\n================ MODEL AUDIT: {role.upper()} (mode={expected_mode}) ================")
  print(f"  MoE Exec Path  : {moe_path}\n  Dense Exec Path: {dense_path}")
  for mod_name, mdata in modules.items():
    w, s = mdata["weight"], mdata["scale"]
    w_str = f"{w['dtype']} {tuple(w['shape'])} (absmax={w['sample_absmax']:.4g})" if w else "None"
    s_str = f"{s['dtype']} {tuple(s['shape'])} (range=[{s['sample_min']:.4g}, {s['sample_max']:.4g}])" if s else "None"
    print(f"  {mod_name:<34}: weight={w_str:<42} | scale={s_str}")
  print("=" * 84 + "\n", flush=True)

  if expected_mode in ("fp8", "fp8moe", "fp8_ckpt", "fp8_serve", "fp8_moe", "fp8_moe_native"):
    assert wi_is_fp8 and wi_scale_info is not None and wi_info["sample_absmax"] > 1.0, f"[{role}] Invalid FP8 MoE: {wi_info}"
    if expected_mode == "fp8_moe_native" and not is_fused_moe:
      assert moe_path.startswith("GMM_V2_NATIVE_FP8"), f"[{role}] Expected GMM_V2_NATIVE_FP8, got {moe_path}"
    if expected_mode == "fp8_serve":
      assert dense_path.startswith("DENSE_NATIVE_FP8"), f"[{role}] Expected DENSE_NATIVE_FP8, got {dense_path}"
  elif expected_mode == "int8_moe":
    assert wi_is_int8 and wi_scale_info is not None and wi_info["sample_absmax"] > 1.0, f"[{role}] Invalid INT8 MoE: {wi_info}"
  elif expected_mode == "bf16":
    assert wi_info is not None and "bfloat16" in wi_info["dtype"], f"[{role}] Expected bfloat16 MoE, got {wi_info}"

  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"audit_{role}.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
  return report


# --------------------------------------------------------- Shared Tunix Orchestrator Wiring


def _safe_pad_id(pad_id: int, *token_arrays) -> int:
  """Returns a pad_id guaranteed not to collide with any active token in `token_arrays`."""
  used = {int(x) for arr in token_arrays if arr is not None for seq in arr for x in np.asarray(seq).reshape(-1)}
  cand = int(pad_id)
  while cand in used and cand > 0:
    cand -= 1
  return max(used, default=0) + 1 if cand in used else cand


def _concat_packed_payloads(chunks):
  """Concatenates multiple `AssembledBatch` chunks from `SequencePackedBatchAssembler` along axis 0."""
  import dataclasses
  from tunix.experimental.common import datatypes

  if len(chunks) == 1:
    return chunks[0].payload
  p0 = chunks[0].payload
  fields = {}
  for f in dataclasses.fields(datatypes.RLTrainerPayload):
    if f.name == "metadata":
      continue
    vals = [getattr(c.payload, f.name) for c in chunks]
    fields[f.name] = None if vals[0] is None else (max(vals) if np.ndim(vals[0]) == 0 else np.concatenate(vals, axis=0))
  traj_ids = tuple(tid for c in chunks for tid in c.payload.metadata.get("trajectory_ids", ()))
  meta = dict(p0.metadata)
  meta["trajectory_ids"] = traj_ids
  return datatypes.RLTrainerPayload(**fields, metadata=meta)


def build_algo_and_batch(
    prompt_ids,
    completion_ids,
    rollout_logps,
    *,
    completion_lens=None,
    conversation_masks=None,
    routed_experts=None,
    temperature: float = 1.0,
    pad_id: int = 0,
    target_batch_size: int | None = None,
    max_seq_token_per_tpu: int | None = None,
    max_segments_per_packed_row: int = 16,
    pack_size: int | None = None,
):
  """Builds GRPOAdapter and RLTrainerPayload via Tunix's orchestrator pipeline (`mlperf_base.sh`)."""
  from tunix.experimental.common import datatypes
  from tunix.experimental.orchestrator import algorithm_adapter, batch_assembly
  from tunix.rl import algorithm_config

  n_seqs = len(prompt_ids)
  c_lens = [int(completion_lens[i]) if completion_lens is not None else len(completion_ids[i]) for i in range(n_seqs)]
  max_prompt_len, max_comp_len = max(len(p) for p in prompt_ids), max(max(c_lens), 1)
  max_resp_len = max(1, max_comp_len + ((512 - ((max_prompt_len + max_comp_len) % 512)) % 512))
  batch_size = target_batch_size or n_seqs
  use_packing = max_seq_token_per_tpu is not None and max_seq_token_per_tpu > 0
  eff_pack_size = max(1, int(pack_size or batch_size))

  algo_cfg = algorithm_config.GRPOConfig(
      num_generations=max(2, n_seqs),
      epsilon=0.2,
      epsilon_high=0.28,
      beta=0.0,
      temperature=float(temperature),
      use_rollout_logps=False,
      loss_agg_mode="token-mean",
      advantage_estimator="grpo-loo",
      seq_logprob_error_threshold=SEQ_ERR_THRESHOLD,
      truncated_importance_sampling_type="seq-mask-tis",
      truncated_importance_sampling_ratio_min=RATIO_MIN,
      truncated_importance_sampling_ratio=RATIO_MAX,
  )
  algo = algorithm_adapter.GRPOAdapter(algo_config=algo_cfg, mini_batch_size=1, train_micro_batch_size=batch_size, max_response_length=max_resp_len)
  group = []
  for i in range(n_seqs):
    p_i = np.asarray(prompt_ids[i], dtype=np.int32).reshape(-1)
    p_len, c_len = int(len(p_i)), max(1, c_lens[i])
    c_i = np.asarray(completion_ids[i], dtype=np.int32).reshape(-1)[:c_len]
    lp_i = np.asarray(rollout_logps[i], dtype=np.float32).reshape(-1)[:c_len]
    mask_i = np.isfinite(lp_i)
    if conversation_masks is not None:
      mask_i = (np.asarray(conversation_masks[i], dtype=np.float32).reshape(-1)[:c_len] > 0) & mask_i
    mask_i = mask_i.astype(np.float32)
    traj = {
        "prompt_tokens": p_i,
        "prompt_length": p_len,
        "conversation_tokens": c_i,
        "conversation_masks": mask_i,
        "old_logprobs": np.where(mask_i > 0, lp_i, 0.0).astype(np.float32),
    }
    if routed_experts is not None and routed_experts[i] is not None:
      traj["routed_experts"] = np.asarray(routed_experts[i], dtype=np.int16)[: p_len + c_len]
    group.append(datatypes.TrajectoryItem(traj=traj, is_valid=True))

  payloads = algo.create_trainer_payloads(group, rewards=np.linspace(0.0, 1.0, n_seqs, dtype=np.float32))
  for i, p in enumerate(payloads):
    p.metadata["traj_id"] = str(i)

  eff_max_seq = max(int(max_seq_token_per_tpu), max_prompt_len + max_resp_len) if use_packing else None
  assembler = batch_assembly.create_batch_assembler(
      num_generations=max(2, n_seqs),
      mini_batch_size=1,
      train_micro_batch_size=batch_size,
      batch_config=batch_assembly.BatchConfig(
          pad_id=int(pad_id),
          max_prompt_length=max_prompt_len,
          max_response_length=max_resp_len,
          max_seq_token_per_tpu=eff_max_seq,
          max_segments_per_packed_row=int(max_segments_per_packed_row),
          trainer_fsdp=eff_pack_size if use_packing else None,
      ),
  )
  if use_packing:
    chunks = list(assembler.feed(payloads)) + list(assembler.flush())
    return algo, _concat_packed_payloads(chunks)
  return algo, assembler.pack(payloads)[0]


def _iter_packed_segments(batch, comp_lens, default_comp_len: int):
  """Yields `(row_idx, seq_idx, seg_comp_positions)` for each packed segment in `batch`."""
  traj_ids = batch.metadata.get("trajectory_ids", ())
  flat_idx = 0
  for row_idx in range(batch.segment_ids.shape[0]):
    seg_ids = np.asarray(batch.segment_ids[row_idx])
    for seg_num in range(1, (int(np.max(seg_ids)) if seg_ids.size else 0) + 1):
      if flat_idx >= len(traj_ids):
        break
      tid_str = traj_ids[flat_idx]
      flat_idx += 1
      if tid_str:
        seq_idx = int(tid_str)
        c_len = int(comp_lens[seq_idx]) if comp_lens is not None else default_comp_len
        seg_positions = np.flatnonzero(seg_ids == seg_num)
        if len(seg_positions):
          yield row_idx, seq_idx, seg_positions[-c_len:].tolist()


def _find_packed_seq0_location(batch) -> tuple[int, np.ndarray | None]:
  """Returns `(row_idx, seg_all_positions)` for sequence 0 inside a 1D-packed `batch` (or `(0, None)` if unpacked)."""
  if getattr(batch, "segment_ids", None) is None:
    return 0, None
  traj_ids = batch.metadata.get("trajectory_ids", ())
  flat_idx = 0
  for row_idx in range(batch.segment_ids.shape[0]):
    seg_ids = np.asarray(batch.segment_ids[row_idx])
    for seg_num in range(1, (int(np.max(seg_ids)) if seg_ids.size else 0) + 1):
      if flat_idx >= len(traj_ids):
        break
      tid_str = traj_ids[flat_idx]
      flat_idx += 1
      if tid_str and int(tid_str) == 0:
        seg_positions = np.flatnonzero(seg_ids == seg_num)
        if len(seg_positions):
          return row_idx, seg_positions.astype(np.int32)
  return 0, None


def _unpack_trainer_logps(resp_logps: np.ndarray, batch, comp_lens, n_rows: int, comp_width: int) -> np.ndarray:
  """Extracts per-sequence completion logprobs `[n_rows, comp_width]` from trainer output (padded or 1D-packed)."""
  if batch.segment_ids is None:
    return np.asarray(resp_logps[:n_rows, :comp_width], dtype=np.float32)
  out = np.full((n_rows, comp_width), np.nan, dtype=np.float32)
  for row_idx, seq_idx, seg_comp_pos in _iter_packed_segments(batch, comp_lens, comp_width):
    if seq_idx < n_rows:
      seg_lp = np.asarray(resp_logps[row_idx, seg_comp_pos], dtype=np.float32)
      out[seq_idx, : len(seg_lp)] = seg_lp
  return out


# --------------------------------------------------------- Stages 1–4: Tokenize, Sampler, Trainer, Compare


def stage_tokenize(args, hf_home, out_dir):
  path = os.path.join(out_dir, "tokens.npz")
  if os.path.exists(path) and not args.retokenize:
    with np.load(path) as data:
      return data["tokens"]
  tok, _ = get_tokenizer(hf_home)
  rows = []
  with open(args.prompts_file, encoding="utf-8") as f:
    for ln in f:
      if ln.strip():
        ids = tok(json.loads(ln)["text"], add_special_tokens=False)["input_ids"]
        assert len(ids) >= args.prompt_len, f"Prompt shorter than {args.prompt_len}: {len(ids)}"
        rows.append(ids[: args.prompt_len])
        if args.num_prompts and len(rows) >= args.num_prompts:
          break
  tokens = np.array(rows, dtype=np.int32)
  np.savez(path, tokens=tokens)
  log(f"tokenize: saved {tokens.shape} -> {path}")
  return tokens


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  """Runs in-process rollout sampling via `tunix.generate.vllm_sampler.VllmSampler`."""
  import jax.numpy as jnp
  from tunix.generate import vllm_sampler as tunix_vllm_sampler
  from tunix.utils import maxtext_utils as tunix_maxtext_utils

  mode = args.sampler_mode
  is_fp8_ckpt, in_place_moe = mode in FP8_CKPT_MODES, mode in IN_PLACE_MOE_MODES
  ckpt_path = args.sampler_ckpt or (CKPT_FP8 if is_fp8_ckpt else CKPT_BF16)
  with np.load(os.path.join(out_dir, "tokens.npz")) as tok_data:
    tokens = tok_data["tokens"]
  n_seqs, prompt_len = tokens.shape
  tok, tok_name = get_tokenizer(hf_home)
  pad_id = tunix_maxtext_utils.get_tokenizer_pad_id(tok_name, tok_name, tok_name)

  sys.path.insert(0, os.path.join(maxtext_root, "src", "maxtext", "integration", "vllm"))
  import maxtext_vllm_adapter

  maxtext_vllm_adapter.register()

  additional_config = tunix_maxtext_utils.build_vllm_maxtext_additional_config(
      model_name=MODEL_MAXTEXT_FP8 if is_fp8_ckpt else MODEL_MAXTEXT_BF16,
      attention="vllm_rpa",
      prefuse_moe_weights=not is_fp8_ckpt,
      return_routed_experts=args.router_replay,
      float32_gate_logits=True,
      float32_logits=True,
  )
  mt_cfg = additional_config["maxtext_config"]
  mt_cfg.update(
      load_parameters_path=ckpt_path,
      dtype="bfloat16",
      scan_layers=False,
      use_multimodal=False,
      enable_dp_attention=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      float32_weight_sum=True,
      logits_dot_in_fp32=True,
  )
  if is_fp8_ckpt:
    mt_cfg.update(weight_dtype="float8_e4m3fn", fp8_moe=True, weight_block_size=128)
  if mode == "fp8_serve":
    mt_cfg["quantization"] = "serve_fp8_weight"
  additional_config["custom_mamba_cache_multiplier"] = 16
  additional_config["sharding"] = {"sharding_strategy": {"expert_parallelism": args.sampler_ep, "tensor_parallelism": 1, "enable_dp_attention": True}}

  num_turns = max(1, int(args.num_turns))
  env_toks_total = (num_turns - 1) * max(0, int(args.env_tokens_per_turn))
  vllm_cfg = tunix_vllm_sampler.VllmConfig(
      server_mode=False,
      init_with_random_weights=False,
      return_logprobs=True,
      return_routed_experts=args.router_replay,
      eos_tokens=[248046, 248044] if getattr(args, "stop_at_eos", False) else [],
      enable_dp_attention=True,
      hbm_utilization=0.6 if in_place_moe else 0.7,
      tensor_parallel_size=1,
      data_parallel_size=1,
      expert_parallel_size=args.sampler_ep,
      additional_config=additional_config,
      overlap_postprocessing=False,
      engine_kwargs={
          "model": tok_name,
          "tokenizer": tok_name,
          "trust_remote_code": True,
          "dtype": "bfloat16",
          "max_model_len": max(4096, prompt_len + args.gen_tokens + env_toks_total + 64),
          "max_num_seqs": 16,
          "max_num_batched_tokens": int(getattr(args, "max_num_batched_tokens", 2048)),
          "block_size": 256,
          "enable_chunked_prefill": True,
          "enable_prefix_caching": True,
          "mamba_cache_mode": "align",
          "language_model_only": True,
          "limit_mm_per_prompt": {"image": 0, "video": 0},
          "disable_log_stats": True,
          "kv_cache_dtype": "bfloat16",
          "reasoning_parser": "qwen3",
          "seed": 0,
          "hf_overrides": dict(tunix_maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES),
      },
  )
  total_gen = max(1, args.gen_tokens)
  do_probe = bool(getattr(args, "probe_modules", False))
  init_probe_pos = (
      mdp.select_probe_positions(
          prompt_len, total_gen + env_toks_total, max_tokens=int(getattr(args, "probe_max_tokens", 64))
      )
      if do_probe
      else None
  )
  quantized_in_hook = False
  sa_tap_in_hook = None
  adapter_cls = getattr(maxtext_vllm_adapter, "MaxTextForCausalLM", None)
  orig_load_weights = getattr(adapter_cls, "load_weights", None) if adapter_cls is not None else None

  def _load_and_quantize(self, *l_args, **l_kwargs):
    from flax import nnx
    import jax

    nonlocal quantized_in_hook, sa_tap_in_hook
    orig_load_weights(self, *l_args, **l_kwargs)
    if in_place_moe:
      rules_ctx = (
          nnx.logical_axis_rules(self.maxtext_config.logical_axis_rules)
          if hasattr(self, "maxtext_config")
          else contextlib.nullcontext()
      )
      with jax.set_mesh(self.mesh), rules_ctx:
        quantize_moe_fp8(
            self.model,
            scale_mode=args.moe_scale_mode,
            weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn,
            set_serve_quant=(mode == "fp8_moe_native"),
        )
      quantized_in_hook = True
    if do_probe and sa_tap_in_hook is None:
      sa_tap_in_hook = mdp.ModuleProbeTap(
          self.model,
          role="sampler",
          probe_positions=init_probe_pos,
          prompt_len=prompt_len,
          gen_len=total_gen + env_toks_total,
          layer_indices=mdp.parse_probe_layers(getattr(args, "probe_layers", "all")),
          recording_enabled=False,
      )
      sa_tap_in_hook.__enter__()

  if (in_place_moe or do_probe) and adapter_cls is not None and orig_load_weights is not None:
    with mock.patch.object(adapter_cls, "load_weights", _load_and_quantize):
      sampler = tunix_vllm_sampler.VllmSampler(tokenizer=tok, config=vllm_cfg)
  else:
    sampler = tunix_vllm_sampler.VllmSampler(tokenizer=tok, config=vllm_cfg)

  runner = sampler._model_runner
  for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
    if obj is not None and hasattr(obj, "uses_mrope"):
      obj.uses_mrope = False
  runner.get_mrope_input_positions_fn = runner.model.get_mrope_input_positions

  if in_place_moe and not quantized_in_hook:
    from flax import nnx
    import jax

    quantize_moe_fp8(
        runner.model.model,
        scale_mode=args.moe_scale_mode,
        weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn,
        set_serve_quant=(mode == "fp8_moe_native"),
    )
    if hasattr(runner, "state"):
      runner.state = nnx.state(runner.model)
    if hasattr(sampler, "refresh_state_leaves"):
      sampler.refresh_state_leaves()
    elif hasattr(runner, "state"):
      runner.state_leaves = jax.tree.leaves(runner.state)
  state_leaves = getattr(runner, "state_leaves", None)
  if state_leaves:
    exp_sub = "int8" if mode == "int8_moe" else ("float8" if mode != "bf16" else "bfloat16")
    assert any(exp_sub in str(getattr(x, "dtype", "")) for x in state_leaves), (
        f"[sampler] runner.state_leaves missing {exp_sub} arrays after model load!"
    )
  audit_model(runner.model.model, role="sampler", expected_mode=mode, out_dir=out_dir)

  def _cleanup_sampler():
    import jax

    nonlocal sa_tap_in_hook
    if sa_tap_in_hook is not None:
      with contextlib.suppress(Exception):
        sa_tap_in_hook.stop_recording()
        sa_tap_in_hook.__exit__(None, None, None)
      sa_tap_in_hook = None
    for fn in (getattr(sampler, "delete_cache", None), getattr(sampler, "stop", None)):
      if callable(fn):
        with contextlib.suppress(Exception):
          fn()
    for x in jax.tree.leaves((getattr(runner, "state_leaves", None), getattr(runner, "state", None))):
      if hasattr(x, "delete"):
        with contextlib.suppress(Exception):
          x.delete()
    gc.collect()
    jax.clear_caches()

  if args.audit_only:
    _cleanup_sampler()
    del runner, sampler
    return

  toks_per_turn = max(1, total_gen // num_turns)
  curr_prompts = [list(map(int, row)) for row in tokens]
  acc_comp_ids, acc_logps, acc_masks = ([[] for _ in range(n_seqs)] for _ in range(3))
  acc_routed = [None for _ in range(n_seqs)]
  no_stop = not getattr(args, "stop_at_eos", False) and args.gen_tokens > 0

  if sa_tap_in_hook is not None:
    sa_tap_in_hook.start_recording(reset=True)
    probe_ctx = contextlib.nullcontext(sa_tap_in_hook)
  elif do_probe:
    probe_ctx = mdp.ModuleProbeTap(
        runner.model.model,
        role="sampler",
        probe_positions=init_probe_pos,
        prompt_len=prompt_len,
        gen_len=total_gen + env_toks_total,
        layer_indices=mdp.parse_probe_layers(getattr(args, "probe_layers", "all")),
    )
  else:
    probe_ctx = contextlib.nullcontext()

  with probe_ctx as sa_tap:
    for turn_idx in range(num_turns):
      turn_steps = total_gen - toks_per_turn * (num_turns - 1) if turn_idx == num_turns - 1 else toks_per_turn
      re_start = 0 if turn_idx == 0 else [len(r) if r is not None else 0 for r in acc_routed]
      out = sampler(
          prompt_token_ids=curr_prompts,
          max_generation_steps=turn_steps,
          max_prompt_length=max(len(p) for p in curr_prompts),
          temperature=args.gen_temperature if args.gen_tokens > 0 else 0.0,
          top_p=1.0,
          top_k=-1,
          routed_experts_prompt_start=re_start,
          ignore_eos=no_stop,
          stop_token_ids=[] if no_stop else [248046, 248044],
          _eos_token_id=None if no_stop else getattr(tok, "eos_token_id", None),
          min_tokens=turn_steps if no_stop else 1,
      )
      for i in range(n_seqs):
        t_ids, t_lps = [int(x) for x in out.tokens[i]], [float(x) for x in out.logprobs[i]]
        acc_comp_ids[i].extend(t_ids)
        acc_logps[i].extend(t_lps)
        acc_masks[i].extend([1.0] * len(t_ids))
        curr_prompts[i].extend(t_ids)
        if args.router_replay and out.routed_experts and out.routed_experts[i] is not None:
          re_arr = np.asarray(out.routed_experts[i], dtype=np.int16)
          acc_routed[i] = re_arr if acc_routed[i] is None else np.concatenate([acc_routed[i], re_arr], axis=0)
        if turn_idx < num_turns - 1 and args.env_tokens_per_turn > 0:
          env_ids = [100] * int(args.env_tokens_per_turn)
          acc_comp_ids[i].extend(env_ids)
          acc_logps[i].extend([np.nan] * len(env_ids))
          acc_masks[i].extend([0.0] * len(env_ids))
          curr_prompts[i].extend(env_ids)

  if sa_tap_in_hook is not None:
    sa_tap_in_hook.stop_recording()
    sa_tap_in_hook.__exit__(None, None, None)
    sa_tap_in_hook = None

  if do_probe and sa_tap is not None:
    seq0_full = np.concatenate([tokens[0, :prompt_len], np.asarray(acc_comp_ids[0], dtype=np.int32)])
    valid_mask = init_probe_pos < len(seq0_full)
    if getattr(sa_tap, "_filled_slots", None):
      any_filled = np.any(np.stack(list(sa_tap._filled_slots.values()), axis=0), axis=0)
      if np.any(any_filled):
        valid_mask = valid_mask & any_filled
    if not np.all(valid_mask):
      sa_tap.probe_positions = init_probe_pos[valid_mask]
      sa_tap.gen_len = len(acc_comp_ids[0])
      for k in list(sa_tap._buffers.keys()):
        sa_tap._buffers[k] = sa_tap._buffers[k][valid_mask]
    p_pos = sa_tap.probe_positions
    sa_probes = sa_tap.get_probes(
        target_next_tokens=seq0_full[np.minimum(p_pos + 1, len(seq0_full) - 1)],
        prompt_tokens_probe=seq0_full[p_pos],
        temperature=float(args.gen_temperature) if args.gen_tokens > 0 else 1.0,
    )
    np.savez(os.path.join(out_dir, "sampler_module_probes.npz"), **sa_probes)
    log(f"sampler: saved {len(sa_probes)} module probe entries ({len(p_pos)} positions) -> sampler_module_probes.npz")

  _cleanup_sampler()
  del runner, sampler

  comp_lens = np.array([len(t) for t in acc_comp_ids], dtype=np.int32)
  _, batch = build_algo_and_batch(
      tokens,
      acc_comp_ids,
      acc_logps,
      completion_lens=comp_lens,
      conversation_masks=acc_masks,
      routed_experts=acc_routed if args.router_replay else None,
      temperature=args.gen_temperature,
      pad_id=_safe_pad_id(pad_id, tokens, acc_comp_ids),
  )
  max_c = int(np.max(comp_lens))
  np.savez(
      os.path.join(out_dir, "sampler_logprobs.npz"),
      tokens=batch.prompt_ids[:, :prompt_len],
      gen_ids=batch.completion_ids[:, :max_c],
      gen_logp=np.where(batch.completion_mask[:, :max_c] > 0, batch.rollout_per_token_logps[:, :max_c], np.nan),
      gen_lens=comp_lens,
      conversation_masks=batch.completion_mask[:, :max_c].astype(np.float32),
  )
  if args.router_replay and batch.routed_experts is not None:
    np.savez_compressed(os.path.join(out_dir, "router_indices.npz"), experts=batch.routed_experts[:, : prompt_len + max_c])


def stage_trainer(args, hf_home, maxtext_root, out_dir):
  """Scores trajectories on the trainer via `run_trainer_node._create_maxtext_trainer_factory` + `TrainerWorker.per_token_logps`."""
  del maxtext_root
  from flax import nnx
  import jax
  import jax.numpy as jnp
  import optax
  from tunix.experimental.common import datatypes
  from tunix.experimental.examples.common import run_trainer_node
  from tunix.experimental.worker import trainer_worker
  from tunix.utils import maxtext_utils as tunix_maxtext_utils

  mode = args.trainer_mode
  is_fp8_ckpt = mode in FP8_CKPT_MODES
  ckpt_path = args.trainer_ckpt or (CKPT_FP8 if is_fp8_ckpt else CKPT_BF16)

  sa_path = os.path.join(out_dir, "sampler_logprobs.npz")
  if not args.audit_only and os.path.exists(sa_path):
    with np.load(sa_path) as sa:
      prompt_np, gen_ids, sa_gen_logp = sa["tokens"], sa["gen_ids"], sa["gen_logp"]
      gen_lens = sa["gen_lens"] if "gen_lens" in sa else None
      conv_masks = sa["conversation_masks"] if "conversation_masks" in sa else None
  else:
    with np.load(os.path.join(out_dir, "tokens.npz")) as tok_data:
      prompt_np = tok_data["tokens"]
    gen_ids = np.zeros((prompt_np.shape[0], max(1, args.gen_tokens)), dtype=np.int32)
    sa_gen_logp = np.zeros_like(gen_ids, dtype=np.float32)
    gen_lens = conv_masks = None

  n_rows, n_prompt = prompt_np.shape
  comp_len = gen_ids.shape[1]
  max_resp_len = max(1, comp_len + ((512 - ((n_prompt + comp_len) % 512)) % 512))
  dp_fsdp = max(1, len(jax.devices()) // max(1, args.trainer_tp * args.trainer_ep))
  pack_size = max(1, dp_fsdp * args.trainer_ep)
  micro = ((max(args.trainer_micro_batch, dp_fsdp) + dp_fsdp - 1) // dp_fsdp) * dp_fsdp
  n_padded = ((n_rows + micro - 1) // micro) * micro
  use_packing = args.max_seq_token_per_tpu is not None and args.max_seq_token_per_tpu > 0
  eff_max_seq_token = max(int(args.max_seq_token_per_tpu), n_prompt + max_resp_len) if use_packing else 0

  tok, tok_name = get_tokenizer(hf_home)
  pad_id = tunix_maxtext_utils.get_tokenizer_pad_id(tok_name, tok_name, tok_name)
  eff_pad_id = _safe_pad_id(pad_id, prompt_np, gen_ids)
  eos_id = int(getattr(tok, "eos_token_id", None) or pad_id)

  fre_path = os.path.join(out_dir, "router_indices.npz")
  if not args.audit_only and args.router_replay and os.path.exists(fre_path):
    with np.load(fre_path) as fre_data:
      raw_fre = fre_data["experts"]
  else:
    raw_fre = None

  _, batch = build_algo_and_batch(
      prompt_np,
      gen_ids,
      sa_gen_logp,
      completion_lens=gen_lens,
      conversation_masks=conv_masks,
      routed_experts=raw_fre,
      temperature=args.gen_temperature,
      pad_id=eff_pad_id,
      target_batch_size=n_padded,
      max_seq_token_per_tpu=eff_max_seq_token or None,
      pack_size=pack_size,
  )
  extra_flags = [
      f"tokenizer_path={tok_name}",
      f"scan_layers={'scanned/' in ckpt_path and 'unscanned' not in ckpt_path}",
      "enable_checkpointing=True",
      "async_checkpointing=False",
      "checkpoint_storage_use_ocdbt=True",
      "checkpoint_storage_use_zarr3=True",
      "convert_checkpoint_if_possible=False",
      "use_multimodal=False",
      "float32_weight_sum=True",
      "logits_dot_in_fp32=True",
      f"use_gdn_kernel={args.trainer_tp == 1}",
      "gdn_chunk_size=64",
      "use_tokamax_splash=True",
      "sa_use_base2_exp=False",
      "sa_fuse_reciprocal=True",
      f"use_ring_of_experts={args.trainer_ep > 1}",
      "sparse_matmul=True",
      "megablox=True",
      "use_tokamax_gmm=True",
      "use_gmm_v2=True",
      "wi_tile_fwd_batch_seq=256",
      "wi_tile_fwd_embed_dim=128",
      "wi_tile_fwd_mlp_dim=128",
      "wo_tile_fwd_batch_seq=256",
      "wo_tile_fwd_embed_dim=128",
      "wo_tile_fwd_mlp_dim=128",
      "max_num_seqs=16",
  ]
  if is_fp8_ckpt:
    extra_flags += ["weight_dtype=float8_e4m3fn", "fp8_moe=True", "weight_block_size=128"]
  if mode == "fp8_serve":
    extra_flags.append("quantization=serve_fp8_weight")
  os.environ["MAXTEXT_EXTRA_FLAGS"] = " ".join(extra_flags)

  logps_mb = pack_size if use_packing else micro
  trainer_args = run_trainer_node._parse_args([
      "--trainer_backend=maxtext",
      "--worker_id=rl_parity_audit",
      f"--model_id={tok_name}",
      f"--tokenizer_path={tok_name}",
      f"--maxtext_model_name={MODEL_MAXTEXT_FP8 if is_fp8_ckpt else MODEL_MAXTEXT_BF16}",
      f"--maxtext_ckpt_path={ckpt_path}",
      f"--maxtext_output_directory={os.path.join(out_dir, 'maxtext_run')}",
      "--maxtext_attention=flash",
      f"--base_num_kv_heads={max(2, args.trainer_tp)}",
      "--remat_policy=full",
      f"--mesh_fsdp={dp_fsdp}",
      f"--mesh_tp={args.trainer_tp}",
      f"--mesh_expert={args.trainer_ep}",
      "--mini_batch_size=1",
      f"--num_generations={n_padded}",
      f"--train_micro_batch_size={micro}",
      f"--compute_logps_micro_batch_size={logps_mb}",
      f"--compute_logps_chunk_size={LOGPS_CHUNK_SIZE}",
      f"--max_prompt_length={n_prompt}",
      f"--max_response_length={max_resp_len}",
      f"--max_seq_token_per_tpu={eff_max_seq_token}",
      f"--prefuse_moe_weights={not (is_fp8_ckpt or mode == 'fp8_moe_native')}",
      "--checkpoint_save_interval_steps=0",
  ])
  trainer_factory, mesh = run_trainer_node._create_maxtext_trainer_factory(trainer_args)
  _, maxtext_engine, _ = tunix_maxtext_utils.maxtext_modules()
  orig_nnx_opt = nnx.Optimizer
  orig_build_model = maxtext_engine.MaxTextTrainingEngine._build_model
  quantized_in_hook = False

  def _build_and_quantize_model(self, *b_args, **b_kwargs):
    nonlocal quantized_in_hook
    res = orig_build_model(self, *b_args, **b_kwargs)
    model = res[0] if isinstance(res, tuple) else res
    if mode in IN_PLACE_MOE_MODES:
      with self._sharding_ctx() if hasattr(self, "_sharding_ctx") else contextlib.nullcontext():
        quantize_moe_fp8(
            model,
            scale_mode=args.moe_scale_mode,
            weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn,
            set_serve_quant=(mode == "fp8_moe_native"),
        )
      quantized_in_hook = True
    return res

  with (
      mock.patch.object(maxtext_engine.MaxTextTrainingEngine, "_build_model", _build_and_quantize_model),
      mock.patch.object(tunix_maxtext_utils, "_build_fp32_master_optimizer_cls", return_value=orig_nnx_opt),
      mock.patch.object(
          maxtext_engine.MaxTextTrainingEngine,
          "_build_optimizer",
          lambda self, tx, _cls=orig_nnx_opt: _cls(self._model, optax.identity(), wrt=nnx.Param),
      ),
  ):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=trainer_factory,
        worker_id=trainer_args.worker_id,
        logps_chunk_size=trainer_args.compute_logps_chunk_size,
        logps_micro_batch_size=trainer_args.compute_logps_micro_batch_size,
        execution_context=mesh,
    )

  model = worker._trainer._model
  if mode in IN_PLACE_MOE_MODES and not quantized_in_hook:
    quantize_moe_fp8(model, scale_mode=args.moe_scale_mode, weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn, set_serve_quant=(mode == "fp8_moe_native"))
    worker._trainer._optimizer = orig_nnx_opt(model, optax.identity(), wrt=nnx.Param)
    worker._trainer._state = None
    if hasattr(worker._trainer, "_invalidate_pure_state"):
      worker._trainer._invalidate_pure_state()
  audit_model(model, role="trainer", expected_mode=mode, out_dir=out_dir)

  def _cleanup_trainer():
    with contextlib.suppress(Exception):
      for x in jax.tree.leaves(nnx.state(model)):
        if hasattr(x, "delete"):
          x.delete()
    gc.collect()
    jax.clear_caches()

  if args.audit_only:
    _cleanup_trainer()
    return

  req_kwargs = dict(
      prompt_tokens=batch.prompt_ids,
      completion_tokens=batch.completion_ids,
      pad_id=eff_pad_id,
      eos_id=eos_id,
      temperature=float(args.gen_temperature),
      routed_experts=batch.routed_experts,
      segment_ids=batch.segment_ids,
      segment_positions=batch.segment_positions,
  )
  if "micro_batch_size" in getattr(datatypes.LogprobsRequest, "__dataclass_fields__", ()):
    req_kwargs["micro_batch_size"] = logps_mb if batch.segment_ids is not None else None
  req = datatypes.LogprobsRequest(**req_kwargs)
  if not hasattr(req, "micro_batch_size"):
    req.micro_batch_size = logps_mb if batch.segment_ids is not None else None

  do_probe = bool(getattr(args, "probe_modules", False))
  sa_probe_path = os.path.join(out_dir, "sampler_module_probes.npz")
  if do_probe and os.path.exists(sa_probe_path):
    with np.load(sa_probe_path) as sa_probe_f:
      sa_probes = dict(sa_probe_f)
  else:
    sa_probes = None
  c_len_0 = int(gen_lens[0]) if gen_lens is not None else comp_len
  probe_pos = (
      np.asarray(sa_probes["probe_positions"], dtype=np.int32)
      if (sa_probes is not None and "probe_positions" in sa_probes)
      else (mdp.select_probe_positions(n_prompt, c_len_0, max_tokens=int(getattr(args, "probe_max_tokens", 64))) if do_probe else None)
  )
  probe_layers = mdp.parse_probe_layers(getattr(args, "probe_layers", "all")) if do_probe else None
  global_row0, packed_pos0 = _find_packed_seq0_location(batch) if do_probe else (0, None)
  eff_mb = int(getattr(req, "micro_batch_size", None) or logps_mb or max(1, batch.prompt_ids.shape[0]))
  tr_probe_ctx = (
      mdp.ModuleProbeTap(
          model,
          role="trainer",
          probe_positions=probe_pos,
          prompt_len=n_prompt,
          gen_len=c_len_0,
          packed_seq0_row=global_row0 % eff_mb,
          packed_seq0_positions=packed_pos0,
          trainer_microbatch_idx=global_row0 // eff_mb,
          layer_indices=probe_layers,
      )
      if do_probe
      else contextlib.nullcontext()
  )

  with tr_probe_ctx as tr_tap:
    resp = worker.per_token_logps(req)

  if do_probe and tr_tap is not None:
    seq0_full = np.concatenate([prompt_np[0, :n_prompt], np.asarray(gen_ids[0, :c_len_0], dtype=np.int32)])
    valid_pos = np.minimum(probe_pos, len(seq0_full) - 1)
    prompt_tokens_probe = seq0_full[valid_pos]
    target_next_tokens = seq0_full[np.minimum(valid_pos + 1, len(seq0_full) - 1)]
    temp_val = float(args.gen_temperature) if args.gen_tokens > 0 else 1.0
    sharding_ctx = worker._trainer._sharding_ctx() if hasattr(worker._trainer, "_sharding_ctx") else (jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext())
    with sharding_ctx:
      tr_probes = tr_tap.get_probes(
          target_next_tokens=target_next_tokens,
          prompt_tokens_probe=prompt_tokens_probe,
          temperature=temp_val,
      )
      np.savez(os.path.join(out_dir, "trainer_module_probes.npz"), **tr_probes)
      if sa_probes is not None:
        tr_iso_probes = mdp.run_isolated_trainer_replay(
            model,
            sa_probes,
            target_next_tokens=target_next_tokens,
            prompt_tokens_probe=prompt_tokens_probe,
            temperature=temp_val,
            layer_indices=probe_layers,
        )
        np.savez(os.path.join(out_dir, "trainer_isolated_probes.npz"), **tr_iso_probes)
    log(f"trainer: saved {len(tr_probes)} module probe entries -> trainer_module_probes.npz (isolated={sa_probes is not None})")

  saved_tr = dict(
      tokens=prompt_np,
      gen_ids=gen_ids,
      gen_logp=_unpack_trainer_logps(resp.per_token_logps, batch, gen_lens, n_rows, comp_len),
  )
  if gen_lens is not None:
    saved_tr["gen_lens"] = gen_lens
  if conv_masks is not None:
    saved_tr["conversation_masks"] = conv_masks
  np.savez(os.path.join(out_dir, "trainer_logprobs.npz"), **saved_tr)
  _cleanup_trainer()


def stage_compare(args, out_dir):
  """Computes parity and TIS/OOB diagnostics via `GRPOAdapter.loss_fn()` + `rl_common.sampler_trainer_agreement`."""
  from flax import nnx
  import jax.numpy as jnp
  from tunix.rl import common as rl_common

  for role in ("sampler", "trainer"):
    ap = os.path.join(out_dir, f"audit_{role}.json")
    if os.path.exists(ap):
      with open(ap, encoding="utf-8") as f:
        r = json.load(f)
      print(f"[{role.upper()} AUDIT] mode={r['expected_mode']} | MoE={r['execution_paths']['routed_moe']} | Dense={r['execution_paths']['dense_linear']}")

  with np.load(os.path.join(out_dir, "trainer_logprobs.npz")) as tr_f:
    tr = dict(tr_f)
  with np.load(os.path.join(out_dir, "sampler_logprobs.npz")) as sa_f:
    sa = dict(sa_f)
  assert np.array_equal(tr["tokens"], sa["tokens"]), "Token mismatch between sampler and trainer!"
  tag = f"Sampler={args.sampler_mode.upper()} vs Trainer={args.trainer_mode.upper()}"

  def _eval_band(label, p_ids, c_ids, sa_lp, tr_lp, sa_t1=None, tr_t1=None, c_lens=None, c_masks=None, temp=1.0):
    eff_pad_id = _safe_pad_id(0, p_ids, c_ids)
    algo, batch = build_algo_and_batch(
        p_ids,
        c_ids,
        sa_lp,
        completion_lens=c_lens,
        conversation_masks=c_masks,
        temperature=temp,
        pad_id=eff_pad_id,
        max_seq_token_per_tpu=getattr(args, "max_seq_token_per_tpu", None),
    )
    tr_lp_arr = np.asarray(tr_lp, dtype=np.float32)
    tr_padded = np.zeros_like(batch.completion_mask, dtype=np.float32)
    if batch.segment_ids is None:
      tr_padded[: tr_lp_arr.shape[0], : tr_lp_arr.shape[1]] = np.nan_to_num(tr_lp_arr, nan=0.0)
    else:
      for row_idx, seq_idx, seg_comp_pos in _iter_packed_segments(batch, c_lens, tr_lp_arr.shape[1]):
        vals = np.nan_to_num(tr_lp_arr[seq_idx, : len(seg_comp_pos)], nan=0.0)
        tr_padded[row_idx, seg_comp_pos[: len(vals)]] = vals

    with mock.patch.object(
        rl_common,
        "compute_per_token_logps",
        return_value=(jnp.asarray(tr_padded), jnp.zeros_like(jnp.asarray(tr_padded))),
    ):
      loss_out = algo.loss_fn()(nnx.Module(), batch, algo.algo_config, pad_id=eff_pad_id, eos_id=eff_pad_id)

    aux = {k: float(v.compute() if hasattr(v, "compute") else np.asarray(v)) for k, v in loss_out.aux_metrics.items()}
    mask = np.asarray(batch.completion_mask) > 0
    agree, _, _ = rl_common.sampler_trainer_agreement(
        np.where(mask, batch.rollout_per_token_logps, 0.0),
        np.where(mask, tr_padded, 0.0),
        batch.completion_mask,
    )
    for k, v in agree.items():
      aux[k] = float(v[0])

    aux["is_oob_ratio"] = aux["tis/is_oob_ratio"]
    if sa_t1 is not None and tr_t1 is not None:
      valid_2d = np.isfinite(sa_lp) & np.isfinite(tr_lp)
      aux["top1_agree"] = float(np.mean(sa_t1[valid_2d] == tr_t1[valid_2d])) if np.any(valid_2d) else 0.0

    seq_lens = np.asarray(c_lens, dtype=np.int32) if c_lens is not None else np.sum(np.isfinite(sa_lp), axis=1)
    med_len = int(np.median(seq_lens)) if len(seq_lens) else 0
    is_oob, kept_frac = aux["tis/is_oob_ratio"], aux["sample_mask/kept_frac"]
    valid_2d_all = np.isfinite(sa_lp) & np.isfinite(tr_lp)
    dlogp_2d = np.where(valid_2d_all, np.asarray(tr_lp, dtype=np.float32) - np.asarray(sa_lp, dtype=np.float32), 0.0)
    seq_counts = np.maximum(np.sum(valid_2d_all, axis=1), 1)
    seq_geomeans = np.exp(np.clip(np.sum(dlogp_2d, axis=1) / seq_counts, -20.0, 20.0))
    n_below = int(np.sum(seq_geomeans < RATIO_MIN))
    n_above = int(np.sum(seq_geomeans > RATIO_MAX))
    n_in = len(seq_geomeans) - n_below - n_above
    abs_dlogp = np.abs(dlogp_2d[valid_2d_all]) if np.any(valid_2d_all) else np.zeros(1, dtype=np.float32)
    print(f"\n### {tag} — {label}")
    print(
        f"  tokens={int(mask.sum())}  seqs={len(p_ids)}  "
        f"seq_len(min/med/max)={int(np.min(seq_lens))}/{med_len}/{int(np.max(seq_lens))}  "
        f"band=[{RATIO_MIN}, {RATIO_MAX}]"
    )
    print(
        f"  >>> is_oob_ratio (seq-mask-tis)      = {is_oob:.4f} ({is_oob:.2%})  "
        f"[in-band={n_in}/{len(seq_geomeans)}, below<{RATIO_MIN}={n_below}, above>{RATIO_MAX}={n_above}]"
    )
    print(f"      sample_mask/kept_frac            = {kept_frac:.4f} ({kept_frac:.2%})")
    print(
        f"      seq_geomean (min/med/max)        = "
        f"{float(np.min(seq_geomeans)):.5f} / {float(np.median(seq_geomeans)):.5f} / {float(np.max(seq_geomeans)):.5f}"
    )
    print(
        f"      |dlogp| (med / p99 / max)        = "
        f"{float(np.median(abs_dlogp)):.4f} / {float(np.percentile(abs_dlogp, 99)):.4f} / {float(np.max(abs_dlogp)):.4f}"
    )
    if "OUTPUT" in label and med_len < 512 and int(getattr(args, "gen_tokens", 4096)) >= 512:
      print(
          f"      [WARNING] Short completions (median={med_len} tokens vs --gen-tokens={args.gen_tokens}): "
          f"per-seq SE = sigma/sqrt(N) dominates the 0.3% [{RATIO_MIN}, {RATIO_MAX}] TIS band. "
          "Re-run sampler without --stop-at-eos to generate full-length rollouts."
      )
    for k in (
        "sample_mask/mult_prob_error_mean",
        "sample_mask/mult_prob_error_max",
        "sampler_is/token_logdiff_mean",
        "sampler_is/token_logdiff_absmean",
        "sampler_is/token_logdiff_abs_max",
        "sampler_is/token_outlier_frac",
        "sampler_is/seq_geomean_mean",
        "sampler_trainer/logp_diff_mean",
    ):
      fmt = "+.6f" if k.endswith("_mean") and "logdiff" in k else (".6e" if "outlier" in k else ".6f")
      print(f"      {k:<32} = {aux[k]:{fmt}}")
    if "top1_agree" in aux:
      print(f"      {'top1_agree':<32} = {aux['top1_agree']:.4f}")

    diff_abs = np.where(mask, np.abs(tr_padded - batch.rollout_per_token_logps), -1.0)
    print("  Top-5 Worst Token Divergences:")
    for idx in np.argsort(diff_abs.ravel())[::-1][:5]:
      r, c = divmod(int(idx), diff_abs.shape[1])
      if diff_abs[r, c] < 0:
        break
      print(
          f"    row={r:02d} pos={c:04d} id={int(batch.completion_ids[r, c]):<7d} "
          f"sa={float(batch.rollout_per_token_logps[r, c]):+.4f} tr={float(tr_padded[r, c]):+.4f} |diff|={diff_abs[r, c]:.4f}"
      )
    return aux

  mp = None
  if "logp" in sa and "logp" in tr:
    p_toks = sa["tokens"]
    P = p_toks.shape[1]
    mp = _eval_band(
        "PROMPT tokens (prefill)",
        p_toks[:, :1],
        p_toks[:, 1:P],
        sa["logp"][:, 1:P],
        tr["logp"][:, : P - 1],
        sa_t1=sa["top1"][:, 1:P] if "top1" in sa else None,
        tr_t1=tr["top1"][:, : P - 1] if "top1" in tr else None,
        temp=1.0,
    )

  mg = None
  if "gen_logp" in sa and "gen_logp" in tr:
    mg = _eval_band(
        "OUTPUT tokens (decode)",
        sa["tokens"],
        sa["gen_ids"],
        sa["gen_logp"],
        tr["gen_logp"],
        sa_t1=sa["gen_top1"] if "gen_top1" in sa else None,
        tr_t1=tr["gen_top1"] if "gen_top1" in tr else None,
        c_lens=sa["gen_lens"] if "gen_lens" in sa else None,
        c_masks=sa["conversation_masks"] if "conversation_masks" in sa else None,
        temp=args.gen_temperature,
    )

  mod_div = None
  sa_probe_path = os.path.join(out_dir, "sampler_module_probes.npz")
  tr_probe_path = os.path.join(out_dir, "trainer_module_probes.npz")
  if os.path.exists(sa_probe_path) and os.path.exists(tr_probe_path):
    iso_probe_path = os.path.join(out_dir, "trainer_isolated_probes.npz")
    with np.load(sa_probe_path) as sa_p_f:
      sa_probes = dict(sa_p_f)
    with np.load(tr_probe_path) as tr_p_f:
      tr_probes = dict(tr_p_f)
    if os.path.exists(iso_probe_path):
      with np.load(iso_probe_path) as iso_p_f:
        iso_probes = dict(iso_p_f)
    else:
      iso_probes = None
    mod_div = mdp.compare_module_probes(
        sa_probes,
        tr_probes,
        iso_probes,
        tag=tag,
        out_dir=out_dir,
        probe_layers=getattr(args, "probe_layers", "all"),
    )

  res = {"prompt": mp, "decode": mg, **(mg or mp or {})}
  if mod_div is not None:
    res["module_divergence"] = mod_div
  return res


def main(argv=None):
  raw_argv = list(sys.argv[1:] if argv is None else argv)
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"])
  ap.add_argument("--mode", default=None, choices=VALID_MODES)
  ap.add_argument("--sampler-mode", default="bf16", choices=VALID_MODES)
  ap.add_argument("--trainer-mode", default="bf16", choices=VALID_MODES)
  ap.add_argument("--moe-scale-mode", default="per_channel", choices=VALID_SCALE_MODES)
  ap.add_argument("--audit-only", action="store_true")
  ap.add_argument("--in-process", action="store_true", help="Run sampler and trainer in the same process during --stage all")
  for flag in ("--out-dir", "--hf-home", "--maxtext-root", "--sampler-ckpt", "--trainer-ckpt", "--prompts-file"):
    ap.add_argument(flag, default=None)
  for flag, default in (
      ("--num-prompts", 32),
      ("--prompt-len", 4096),
      ("--gen-tokens", 4096),
      ("--max-num-batched-tokens", 2048),
      ("--sampler-ep", 8),
      ("--trainer-ep", 1),
      ("--trainer-tp", 4),
      ("--trainer-micro-batch", 2),
      ("--num-turns", 1),
      ("--env-tokens-per-turn", 64),
      ("--probe-max-tokens", 64),
  ):
    ap.add_argument(flag, type=int, default=default)
  ap.add_argument("--gen-temperature", type=float, default=1.0)
  ap.add_argument("--router-replay", action=argparse.BooleanOptionalAction, default=True)
  ap.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=False, help="Allow early termination at EOS (<|im_end|>, <|endoftext|>) instead of generating full --gen-tokens")
  ap.add_argument("--pack-sequences", action="store_true", help="Enable Tunix 1D SequencePackedBatchAssembler matching mlperf_base.sh")
  ap.add_argument("--max-seq-token-per-tpu", type=int, default=None, help="Max packed tokens per row (defaults to 65536 when --pack-sequences is set)")
  ap.add_argument("--probe-modules", action=argparse.BooleanOptionalAction, default=False, help="Capture layer-by-layer and module-by-module activations and compute isolated + cumulative divergence")
  ap.add_argument("--probe-layers", default="all", help="Comma-separated layer indices or 'all' for module divergence probing")
  ap.add_argument("--mlperf-v5p", action="store_true", help="Apply mlperf_35b_128_v5p.sh + mlperf_base.sh topology and packing defaults (sampler_ep=4, trainer_tp=2, trainer_ep=1, pack_sequences=True)")
  ap.add_argument("--retokenize", action="store_true")
  args = ap.parse_args(raw_argv)

  if args.mlperf_v5p:
    if not any(a == "--sampler-ep" or a.startswith("--sampler-ep=") for a in raw_argv):
      args.sampler_ep = 4
    if not any(a == "--trainer-tp" or a.startswith("--trainer-tp=") for a in raw_argv):
      args.trainer_tp = 2
    if not any(a == "--trainer-ep" or a.startswith("--trainer-ep=") for a in raw_argv):
      args.trainer_ep = 1
    if "--pack-sequences" not in raw_argv:
      args.pack_sequences = True
  if args.pack_sequences and args.max_seq_token_per_tpu is None:
    args.max_seq_token_per_tpu = 65536
  if args.mode:
    args.sampler_mode = args.trainer_mode = args.mode
  maxtext_root = args.maxtext_root or _MAXTEXT_ROOT
  hf_home = args.hf_home or os.environ.get("HF_HOME") or next(
      (c for c in ("/mnt/disks/persist", "/workspace/persist", os.path.expanduser("~/.cache/huggingface")) if os.path.isdir(os.path.join(c, "hub"))),
      os.path.expanduser("~/.cache/huggingface"),
  )
  out_dir = args.out_dir or os.path.join(hf_home, "rl_logprob_parity_audit")
  os.makedirs(out_dir, exist_ok=True)
  args.prompts_file = args.prompts_file or os.path.join(maxtext_root, "tools", "rl_logprob_parity", "r2e_prompts_32.jsonl")

  if args.stage == "compare":
    return stage_compare(args, out_dir)
  if args.stage != "tokenize":
    set_tpu_env(hf_home)
  stage_tokenize(args, hf_home, out_dir)
  if args.stage == "all" and not args.in_process and not isinstance(stage_sampler, mock.Mock) and not isinstance(stage_trainer, mock.Mock):
    for stg in ("sampler", "trainer"):
      subprocess.run([sys.executable, os.path.abspath(__file__), *raw_argv, "--stage", stg], check=True)
  else:
    if args.stage in ("all", "sampler"):
      stage_sampler(args, hf_home, maxtext_root, out_dir)
    if args.stage in ("all", "trainer"):
      stage_trainer(args, hf_home, maxtext_root, out_dir)
  if args.stage == "all" and not args.audit_only:
    return stage_compare(args, out_dir)


if __name__ == "__main__":
  main()
