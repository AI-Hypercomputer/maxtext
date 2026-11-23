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

"""PyTorch reference parity & Mini-Model-Verification for Inkling 2 Mini.

Suite 1 (`Inkling2SubmoduleAndSyntheticParityTest`):
  - Exact sub-module forward & VJP backward parity against
    `transformers.models.inkling2_loop` (`BlockAttentionResidual`,
    `MessageAttentionResidual`, `MessageGate`, `ShortConvolution`,
    `DynamicShortConvolution`, `SlidingAttention`, `FullAttention`,
    `SparseMoeBlock`, all 4 `DecoderLayer` variants, and full
    `Inkling2LoopForCausalLM` for both `scan_layers=False` and `True`).

Suite 2 (`Inkling2TinyCheckpointVerificationTest`):
  - Full Mini-Model-Verification on `hf-internal-testing/tiny-inkling2-mini`
    (real mixed NVFP4 layers 0..13 + MXFP8 layers 14..17 + BF16/FP32 checkpoint)
    converted through MaxText's real `to_maxtext` pipeline:
    - Boundary inputs (`seq_len=1`, non-tile lengths, `seq_len > sliding_window`,
      `pos > log_scaling_n_floor`, right/left padding invariance, packed multi-doc
      zero-leakage, vocab boundary IDs `0` and `unpadded_vocab_size - 1`).
    - Logits parity (`scan_layers=False` and `True`, cosine >= 0.9999, KL <= 1e-4)
      plus comparison against HF's published golden logits.
    - Decomposed CE & SFT completion-only loss parity (`pre_train.loss_fn`).
    - Per-leaf backward VJP gradient parity (every `nnx.Param` + `dx`).
    - `PerturbSpec` negative controls.
    - 3-step AdamW optimizer + Orbax save/restore continuity vs PyTorch.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from typing import Callable
import unittest

import pytest

try:
  import torch
  import torch.nn.functional as F
  from transformers import AutoTokenizer
  from transformers.models.inkling2_loop import (
      configuration_inkling2_loop,
      modeling_inkling2_loop,
  )

  HAS_TORCH = True
  HAS_INKLING2_HF = True
except ImportError:
  HAS_TORCH = False
  HAS_INKLING2_HF = False

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_INKLING2_HF, reason="torch or transformers>=5.24.0 (inkling2_loop) not available"),
]

# pylint: disable=wrong-import-position
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
import orbax.checkpoint as ocp

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

# pylint: enable=wrong-import-position

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
_DEFAULT_TINY_CKPT_DIR = os.environ.get("INKLING2_TINY_CKPT_DIR", "")

# Published HF CI first-step logits[:5] for `hf-internal-testing/tiny-inkling2-mini`
# on "Explain in one sentence why the sky is usually blue." (`reasoning_effort="none"`),
# from `tests/models/inkling2_loop/test_modeling_inkling2_loop.py`.
_HF_TINY_PUBLISHED_GOLDEN_LOGITS = np.array(
    [1.9270305633544922, 0.287100613117218, 3.4533743858337402, 2.711331605911255, 0.878654956817627],
    dtype=np.float32,
)

_ATOL = 1e-4
_RTOL = 1e-4


@dataclasses.dataclass(frozen=True)
class PerturbSpec:
  """Single programmatic perturbation negative control."""

  name: str
  apply: Callable[[nnx.Module], nnx.Module]


def _t2j(tensor: torch.Tensor) -> jnp.ndarray:
  return jnp.asarray(tensor.detach().cpu().float().numpy(), dtype=jnp.float32)


def _assert_close(actual, expected, atol=_ATOL, rtol=_RTOL, msg=""):
  actual_np = np.asarray(actual, dtype=np.float32)
  expected_np = (
      expected.detach().cpu().float().numpy()
      if isinstance(expected, torch.Tensor)
      else np.asarray(expected, dtype=np.float32)
  )
  np.testing.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol, err_msg=msg)


def _cosine_and_kl(logits_mt: np.ndarray, logits_pt: np.ndarray) -> tuple[float, float]:
  """Returns mean tokenwise cosine similarity and KL(PT || MT) over logits."""
  a = logits_mt.astype(np.float64).reshape(-1, logits_mt.shape[-1])
  b = logits_pt.astype(np.float64).reshape(-1, logits_pt.shape[-1])
  dot = np.sum(a * b, axis=-1)
  cos = float(np.mean(dot / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)))

  def _log_softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))
    return x - lse

  log_p = _log_softmax(b)
  log_q = _log_softmax(a)
  p = np.exp(log_p)
  kl = float(np.mean(np.sum(p * (log_p - log_q), axis=-1)))
  return cos, kl


def _randomize_pt_module_(module: torch.nn.Module, seed: int = 0, std: float = 0.08):
  """Randomizes all PT parameters & router biases so no zero/identity branch collapses."""
  gen = torch.Generator().manual_seed(seed)
  norm_suffixes = (
      "embed_norm.weight",
      "input_layernorm.weight",
      "q_input_layernorm.weight",
      "k_input_layernorm.weight",
      "v_input_layernorm.weight",
      "post_attention_layernorm.weight",
      "norm.weight",
      "norm.scale",
  )
  with torch.no_grad():
    for name, p in module.named_parameters():
      noise = std * torch.randn(p.shape, generator=gen, dtype=torch.float32)
      if name.endswith(norm_suffixes):
        p.copy_((1.0 + noise).to(p.dtype))
      else:
        p.copy_(noise.to(p.dtype))
    for name, b in module.named_buffers():
      if "e_score_correction_bias" in name:
        b.copy_((std * torch.randn(b.shape, generator=gen, dtype=torch.float32)).to(b.dtype))


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg, devices=jax.devices()[:1])
  return Mesh(devices_array, cfg.mesh_axes)


def _copy_layer_weights_from_pt(pt_layer, mt_layer, is_full: bool, is_loop: bool, has_attn_input: bool):
  """Copies weights from an HF Inkling2Loop decoder layer into MaxText Inkling2DecoderLayer."""
  with torch.no_grad():
    if has_attn_input:
      mt_layer.attention_input.query.set_value(_t2j(pt_layer.attention_input.query))
    mt_layer.mlp_input.query.set_value(_t2j(pt_layer.mlp_input.query))

    if is_full:
      mt_layer.q_input_layernorm.scale.set_value(_t2j(pt_layer.q_input_layernorm.weight))
      mt_layer.k_input_layernorm.scale.set_value(_t2j(pt_layer.k_input_layernorm.weight))
      mt_layer.v_input_layernorm.scale.set_value(_t2j(pt_layer.v_input_layernorm.weight))
    else:
      mt_layer.input_layernorm.scale.set_value(_t2j(pt_layer.input_layernorm.weight))
    mt_layer.post_attention_layernorm.scale.set_value(_t2j(pt_layer.post_attention_layernorm.weight))

    # Attention weights
    pa = pt_layer.self_attn
    ma = mt_layer.self_attention
    hq = pa.num_heads
    hkv = pa.num_key_value_heads
    d = pa.head_dim
    h = pa.config.hidden_size
    dr = pa.config.d_rel
    k_conv = pa.config.conv_kernel_size

    ma.query.kernel.set_value(_t2j(pa.q_proj.linear.weight).T.reshape(h, hq, d))
    ma.key.kernel.set_value(_t2j(pa.k_proj.linear.weight).T.reshape(h, hkv, d))
    ma.value.kernel.set_value(_t2j(pa.v_proj.linear.weight).T.reshape(h, hkv, d))
    ma.rel_proj.kernel.set_value(_t2j(pa.relative_position_bias.linear.weight).T.reshape(h, hq, dr))
    ma.out.kernel.set_value(_t2j(pa.o_proj.linear.weight).T.reshape(hq, d, h))
    ma.gate_attn.kernel.set_value(_t2j(pa.o_proj.gate_proj.weight).T.reshape(h, hq, d))
    ma.sink_proj.kernel.set_value(_t2j(pa.sink_proj.weight).T)
    ma.q_l2_clip_scale.set_value(_t2j(pa.q_proj.norm.scale))
    ma.rel_l2_clip_scale.set_value(_t2j(pa.relative_position_bias.norm.scale))
    ma.rel_logits_proj.set_value(_t2j(pa.relative_position_bias.proj))

    if is_full:
      ma.k_sconv.static_w.set_value(_t2j(pa.k_proj.sconv.static_weight).reshape(k_conv, hkv, d))
      ma.k_sconv.weight_proj.kernel.set_value(_t2j(pa.k_proj.sconv.weight_proj.weight).T.reshape(h, hkv, k_conv))
      ma.v_sconv.static_w.set_value(_t2j(pa.v_proj.sconv.static_weight).reshape(k_conv, hkv, d))
      ma.v_sconv.weight_proj.kernel.set_value(_t2j(pa.v_proj.sconv.weight_proj.weight).T.reshape(h, hkv, k_conv))
    else:
      ma.k_sconv.kernel.set_value(_t2j(pa.k_proj.sconv.conv1d.weight.squeeze(1)).T.reshape(k_conv, hkv, d))
      ma.v_sconv.kernel.set_value(_t2j(pa.v_proj.sconv.conv1d.weight.squeeze(1)).T.reshape(k_conv, hkv, d))

    mt_layer.post_attention_convolution.kernel.set_value(
        _t2j(pt_layer.post_attention_convolution.conv1d.weight.squeeze(1)).T
    )
    mt_layer.post_mlp_convolution.kernel.set_value(_t2j(pt_layer.post_mlp_convolution.conv1d.weight.squeeze(1)).T)

    if is_loop:
      for attr in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
        pm = getattr(pt_layer, attr)
        mm = getattr(mt_layer, attr)
        mm.down_proj.kernel.set_value(_t2j(pm.down_proj.weight).T)
        mm.forget_gate_proj.kernel.set_value(_t2j(pm.forget_gate_proj.weight).T)
        mm.message_gate_proj.kernel.set_value(_t2j(pm.message_gate_proj.weight).T)

    # MoE weights: HF gate_up_proj is [E, 2*I, H] with first I = gate, second I = up.
    mt_layer.mlp.gate.kernel.set_value(_t2j(pt_layer.mlp.gate.weight).T)
    mt_layer.mlp.gate.bias.set_value(_t2j(pt_layer.mlp.gate.e_score_correction_bias))
    w_gate, w_up = pt_layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
    mt_layer.mlp.wi_0.set_value(_t2j(w_gate).transpose(0, 2, 1))
    mt_layer.mlp.wi_1.set_value(_t2j(w_up).transpose(0, 2, 1))
    mt_layer.mlp.wo.set_value(_t2j(pt_layer.mlp.experts.down_proj).transpose(0, 2, 1))


def _copy_full_causal_lm_from_pt(pt_model, mt_model, cfg):
  """Copies all weights from HF Inkling2LoopForCausalLM to unscanned MaxText Transformer."""
  with torch.no_grad():
    mt_model.token_embedder.embedding.set_value(_t2j(pt_model.model.embed_tokens.weight))
    mt_model.decoder.embed_norm.scale.set_value(_t2j(pt_model.model.embed_tokens.embed_norm.weight))
    mt_model.decoder.final_input.query.set_value(_t2j(pt_model.model.final_input.query))
    mt_model.decoder.message_input.query.set_value(_t2j(pt_model.model.message_input.query))
    mt_model.decoder.decoder_norm.scale.set_value(_t2j(pt_model.model.norm.weight))

    # Pad lm_head from unpadded_vocab_size to vocab_size with zeros if needed.
    lm_w = _t2j(pt_model.lm_head.weight).T  # [H, unpadded_vocab]
    if lm_w.shape[1] < cfg.vocab_size:
      pad = jnp.zeros((lm_w.shape[0], cfg.vocab_size - lm_w.shape[1]), dtype=lm_w.dtype)
      lm_w = jnp.concatenate([lm_w, pad], axis=1)
    mt_model.decoder.logits_dense.kernel.set_value(lm_w)

    block_size = cfg.attn_res_block_size
    for i, pt_layer in enumerate(pt_model.model.layers):
      mt_layer = getattr(mt_model.decoder, f"layers_{i}")
      is_full = (i + 1) % block_size == 0
      is_loop = cfg.inkling2_core_start <= i < cfg.inkling2_core_end
      _copy_layer_weights_from_pt(pt_layer, mt_layer, is_full=is_full, is_loop=is_loop, has_attn_input=i > 0)


def _copy_unscanned_to_scanned(unscanned_model, scanned_model, scanned_cfg):
  """Stacks unscanned Inkling2 weights into the scanned model layout."""
  block_size = scanned_cfg.attn_res_block_size
  core_start = scanned_cfg.inkling2_core_start
  core_end = scanned_cfg.inkling2_core_end
  num_core_blocks = (core_end - core_start) // block_size

  var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
  src = {tuple(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(unscanned_model, var_filter))}
  dst_state = nnx.state(scanned_model, var_filter)
  new_flat = []
  for path, var in nnx.to_flat_state(dst_state):
    key = tuple(map(str, path))
    if "scanned_blocks" in key:
      idx = key.index("scanned_blocks")
      sub_idx = int(key[idx + 1].rsplit("_", 1)[-1])
      slices = [
          src[key[:idx] + (f"layers_{core_start + b * block_size + sub_idx}",) + key[idx + 2 :]].get_value()
          for b in range(num_core_blocks)
      ]
      axis = scanned_cfg.param_scan_axis if isinstance(var, nnx.Param) else 0
      value = jnp.stack(slices, axis=axis)
    else:
      value = src[key].get_value()
    new_flat.append((path, var.replace(value)))
  nnx.update(scanned_model, nnx.from_flat_state(new_flat))


def _map_pt_grad_to_mt_key(pt_name: str, cfg) -> tuple[tuple[str, ...], Callable[[np.ndarray], np.ndarray]]:
  """Maps a PyTorch `Inkling2LoopForCausalLM` parameter name to unscanned MaxText Param key + transform."""
  h = cfg.emb_dim
  hq = cfg.num_query_heads
  hkv = cfg.num_kv_heads
  d = cfg.head_dim
  dr = cfg.inkling2_d_rel
  k_conv = cfg.short_conv_kernel_size

  if pt_name == "model.embed_tokens.weight":
    return ("token_embedder", "embedding"), lambda x: x
  if pt_name == "model.embed_tokens.embed_norm.weight":
    return ("decoder", "embed_norm", "scale"), lambda x: x
  if pt_name == "model.final_input.query":
    return ("decoder", "final_input", "query"), lambda x: x
  if pt_name == "model.message_input.query":
    return ("decoder", "message_input", "query"), lambda x: x
  if pt_name == "model.norm.weight":
    return ("decoder", "decoder_norm", "scale"), lambda x: x
  if pt_name == "lm_head.weight":
    return ("decoder", "logits_dense", "kernel"), lambda x: x.T

  parts = pt_name.split(".")
  assert parts[:2] == ["model", "layers"], f"Unexpected PT param: {pt_name}"
  layer_idx = int(parts[2])
  rest = ".".join(parts[3:])
  prefix = ("decoder", f"layers_{layer_idx}")

  direct_map = {
      "attention_input.query": (prefix + ("attention_input", "query"), lambda x: x),
      "mlp_input.query": (prefix + ("mlp_input", "query"), lambda x: x),
      "input_layernorm.weight": (prefix + ("input_layernorm", "scale"), lambda x: x),
      "q_input_layernorm.weight": (prefix + ("q_input_layernorm", "scale"), lambda x: x),
      "k_input_layernorm.weight": (prefix + ("k_input_layernorm", "scale"), lambda x: x),
      "v_input_layernorm.weight": (prefix + ("v_input_layernorm", "scale"), lambda x: x),
      "post_attention_layernorm.weight": (prefix + ("post_attention_layernorm", "scale"), lambda x: x),
      "self_attn.q_proj.linear.weight": (prefix + ("self_attention", "query", "kernel"), lambda x: x.T.reshape(h, hq, d)),
      "self_attn.k_proj.linear.weight": (prefix + ("self_attention", "key", "kernel"), lambda x: x.T.reshape(h, hkv, d)),
      "self_attn.v_proj.linear.weight": (
          prefix + ("self_attention", "value", "kernel"),
          lambda x: x.T.reshape(h, hkv, d),
      ),
      "self_attn.relative_position_bias.linear.weight": (
          prefix + ("self_attention", "rel_proj", "kernel"),
          lambda x: x.T.reshape(h, hq, dr),
      ),
      "self_attn.o_proj.linear.weight": (prefix + ("self_attention", "out", "kernel"), lambda x: x.T.reshape(hq, d, h)),
      "self_attn.o_proj.gate_proj.weight": (
          prefix + ("self_attention", "gate_attn", "kernel"),
          lambda x: x.T.reshape(h, hq, d),
      ),
      "self_attn.sink_proj.weight": (prefix + ("self_attention", "sink_proj", "kernel"), lambda x: x.T),
      "self_attn.q_proj.norm.scale": (prefix + ("self_attention", "q_l2_clip_scale"), lambda x: x),
      "self_attn.relative_position_bias.norm.scale": (prefix + ("self_attention", "rel_l2_clip_scale"), lambda x: x),
      "self_attn.relative_position_bias.proj": (prefix + ("self_attention", "rel_logits_proj"), lambda x: x),
      "self_attn.k_proj.sconv.conv1d.weight": (
          prefix + ("self_attention", "k_sconv", "kernel"),
          lambda x: np.squeeze(x, axis=1).T.reshape(k_conv, hkv, d),
      ),
      "self_attn.v_proj.sconv.conv1d.weight": (
          prefix + ("self_attention", "v_sconv", "kernel"),
          lambda x: np.squeeze(x, axis=1).T.reshape(k_conv, hkv, d),
      ),
      "self_attn.k_proj.sconv.static_weight": (
          prefix + ("self_attention", "k_sconv", "static_w"),
          lambda x: x.reshape(k_conv, hkv, d),
      ),
      "self_attn.k_proj.sconv.weight_proj.weight": (
          prefix + ("self_attention", "k_sconv", "weight_proj", "kernel"),
          lambda x: x.T.reshape(h, hkv, k_conv),
      ),
      "self_attn.v_proj.sconv.static_weight": (
          prefix + ("self_attention", "v_sconv", "static_w"),
          lambda x: x.reshape(k_conv, hkv, d),
      ),
      "self_attn.v_proj.sconv.weight_proj.weight": (
          prefix + ("self_attention", "v_sconv", "weight_proj", "kernel"),
          lambda x: x.T.reshape(h, hkv, k_conv),
      ),
      "post_attention_convolution.conv1d.weight": (
          prefix + ("post_attention_convolution", "kernel"),
          lambda x: np.squeeze(x, axis=1).T,
      ),
      "post_mlp_convolution.conv1d.weight": (
          prefix + ("post_mlp_convolution", "kernel"),
          lambda x: np.squeeze(x, axis=1).T,
      ),
      "mlp.gate.weight": (prefix + ("mlp", "gate", "kernel"), lambda x: x.T),
      "mlp.experts.down_proj": (prefix + ("mlp", "wo"), lambda x: np.transpose(x, (0, 2, 1))),
  }
  for gate_attr in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
    direct_map[f"{gate_attr}.down_proj.weight"] = (prefix + (gate_attr, "down_proj", "kernel"), lambda x: x.T)
    direct_map[f"{gate_attr}.forget_gate_proj.weight"] = (
        prefix + (gate_attr, "forget_gate_proj", "kernel"),
        lambda x: x.T,
    )
    direct_map[f"{gate_attr}.message_gate_proj.weight"] = (
        prefix + (gate_attr, "message_gate_proj", "kernel"),
        lambda x: x.T,
    )
  if rest in direct_map:
    return direct_map[rest]
  raise KeyError(f"Unmapped PT parameter: {pt_name}")


def _pt_grads_to_mt_dict(pt_model, cfg) -> dict[tuple[str, ...], np.ndarray]:
  """Converts every PyTorch `.grad` tensor to the unscanned MaxText flat Param dict."""
  out: dict[tuple[str, ...], np.ndarray] = {}
  for name, p in pt_model.named_parameters():
    assert p.grad is not None, f"Missing PT grad for {name}"
    g_np = p.grad.detach().cpu().float().numpy()
    if name.endswith("mlp.experts.gate_up_proj"):
      layer_idx = int(name.split(".")[2])
      mid = g_np.shape[1] // 2
      out[("decoder", f"layers_{layer_idx}", "mlp", "wi_0")] = np.transpose(g_np[:, :mid, :], (0, 2, 1))
      out[("decoder", f"layers_{layer_idx}", "mlp", "wi_1")] = np.transpose(g_np[:, mid:, :], (0, 2, 1))
    else:
      mt_key, fn = _map_pt_grad_to_mt_key(name, cfg)
      arr = fn(g_np)
      if mt_key == ("decoder", "logits_dense", "kernel") and arr.shape[1] < cfg.vocab_size:
        arr = np.pad(arr, ((0, 0), (0, cfg.vocab_size - arr.shape[1])), mode="constant")
      out[mt_key] = arr
  return out


def _perturb_param_leaf(model: nnx.Module, suffix: str, target_rel_norm: float = 1e-2) -> nnx.Module:
  """Returns a cloned model with the first matching Param leaf perturbed by `target_rel_norm`."""
  graphdef, state = nnx.split(model)
  flat = list(nnx.to_flat_state(state))
  matched = False
  rng = np.random.RandomState(314)
  new_flat = []
  for path, var in flat:
    key = "/".join(str(p) for p in path)
    if not matched and isinstance(var, nnx.Param) and key.endswith(suffix):
      val = var.get_value()
      direction = jnp.asarray(rng.randn(*val.shape), dtype=jnp.float32)
      base_norm = jnp.maximum(jnp.linalg.norm(val.astype(jnp.float32)), 1.0)
      dir_norm = jnp.maximum(jnp.linalg.norm(direction), 1e-6)
      delta = direction * (target_rel_norm * base_norm / dir_norm)
      new_flat.append((path, var.replace((val.astype(jnp.float32) + delta).astype(val.dtype))))
      matched = True
    else:
      new_flat.append((path, var))
  if not matched:
    raise KeyError(f"No Param leaf matched suffix {suffix!r}")
  return nnx.merge(graphdef, nnx.from_flat_state(new_flat))


# ==============================================================================
# Suite 1: Submodule & Synthetic End-to-End Parity vs HF PyTorch
# ==============================================================================


class Inkling2SubmoduleAndSyntheticParityTest(unittest.TestCase):
  """Exact submodule & full-model FP32 parity against `transformers.models.inkling2_loop`."""

  _EMB = 64
  _HQ = 4
  _HKV = 2
  _HEAD_DIM = 16
  _MOE_INTER = 48
  _EXPERTS = 8
  _TOP_K = 2
  _VOCAB = 96
  _UNPADDED_VOCAB = 88
  _NUM_LAYERS = 8  # 2 core blocks: prelude 0..1, core 2..5 (2 blocks of 2), coda 6..7
  _BLOCK_SIZE = 2
  _CORE_START = 2
  _CORE_END = 6
  _NUM_PASSES = 3
  _GATE_RANK = 16
  _D_REL = 8
  _REL_EXTENT = 16
  _SLIDING_WINDOW = 6
  _LOG_FLOOR = 4
  _SEQ_LEN = 10

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._prev_matmul = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.pt_cfg = configuration_inkling2_loop.Inkling2LoopTextConfig.from_dict(
        {
            "vocab_size": cls._VOCAB,
            "unpadded_vocab_size": cls._UNPADDED_VOCAB,
            "hidden_size": cls._EMB,
            "num_hidden_layers": cls._NUM_LAYERS,
            "num_attention_heads": cls._HQ,
            "num_key_value_heads": cls._HKV,
            "head_dim": cls._HEAD_DIM,
            "swa_num_attention_heads": cls._HQ,
            "swa_num_key_value_heads": cls._HKV,
            "swa_head_dim": cls._HEAD_DIM,
            "sliding_window_size": cls._SLIDING_WINDOW,
            "d_rel": cls._D_REL,
            "rel_extent": cls._REL_EXTENT,
            "log_scaling_n_floor": cls._LOG_FLOOR,
            "log_scaling_alpha": 0.1,
            "layer_types": [
                "hybrid" if (i + 1) % cls._BLOCK_SIZE == 0 else "hybrid_sliding" for i in range(cls._NUM_LAYERS)
            ],
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "conv_kernel_size": 4,
            "moe_intermediate_size": cls._MOE_INTER,
            "n_routed_experts": cls._EXPERTS,
            "num_experts_per_tok": cls._TOP_K,
            "route_scale": 8.0,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "n_layers_per_attn_res_block": cls._BLOCK_SIZE,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "final_logit_softcapping": 30.0,
            "scoring_func": "sigmoid",
            "loop_layer_index_start": cls._CORE_START,
            "loop_layer_index_end": cls._CORE_END,
            "loop_num_passes": cls._NUM_PASSES,
            "loop_gate_rank": cls._GATE_RANK,
            "attn_implementation": "eager",
        }
    )
    cls.mt_cfg = cls._make_mt_config(scan_layers=False)
    cls.mesh = _make_mesh(cls.mt_cfg)

  @classmethod
  def tearDownClass(cls):
    jax.config.update("jax_default_matmul_precision", cls._prev_matmul)
    super().tearDownClass()

  @classmethod
  def _make_mt_config(cls, scan_layers=False, **overrides):
    """Builds a toy MaxText Inkling2 config matching `cls.pt_cfg`."""
    kwargs = {
        "model_name": "inkling2-mini",
        "override_model_config": True,
        "base_emb_dim": cls._EMB,
        "base_num_query_heads": cls._HQ,
        "base_num_kv_heads": cls._HKV,
        "head_dim": cls._HEAD_DIM,
        "base_mlp_dim": cls._MOE_INTER,
        "base_moe_mlp_dim": cls._MOE_INTER,
        "base_num_decoder_layers": cls._NUM_LAYERS,
        "attn_res_block_size": cls._BLOCK_SIZE,
        "inkling2_core_start": cls._CORE_START,
        "inkling2_core_end": cls._CORE_END,
        "inkling2_num_passes": cls._NUM_PASSES,
        "inkling2_message_gate_rank": cls._GATE_RANK,
        "inkling2_d_rel": cls._D_REL,
        "inkling2_rel_extent": cls._REL_EXTENT,
        "sliding_window_size": cls._SLIDING_WINDOW,
        "short_conv_kernel_size": 4,
        "inkling2_log_scaling_n_floor": float(cls._LOG_FLOOR),
        "inkling2_log_scaling_alpha": 0.1,
        "vocab_size": cls._VOCAB,
        "unpadded_vocab_size": cls._UNPADDED_VOCAB,
        "num_experts": cls._EXPERTS,
        "num_experts_per_tok": cls._TOP_K,
        "shared_experts": 0,
        "sparse_matmul": False,
        "megablox": False,
        "capacity_factor": -1,
        "per_device_batch_size": 2,
        "max_target_length": cls._SEQ_LEN,
        "max_prefill_predict_length": 4,
        "scan_layers": scan_layers,
        "remat_policy": "none",
        "packing": False,
        "enable_checkpointing": False,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "dropout_rate": 0.0,
        "skip_jax_distributed_system": True,
        "hardware": "cpu",
    }
    kwargs.update(overrides)
    return pyconfig.initialize(["", _BASE_CONFIG_PATH], **kwargs)

  def _build_pair(self, seed: int = 10):
    pt_model = modeling_inkling2_loop.Inkling2LoopForCausalLM(self.pt_cfg).float().eval()
    _randomize_pt_module_(pt_model, seed=seed)
    mt_model = model_creation_utils.from_config(
        self.mt_cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
    )
    _copy_full_causal_lm_from_pt(pt_model, mt_model, self.mt_cfg)
    return pt_model, mt_model

  def test_block_attn_res_and_message_attn_res_forward_and_vjp(self):
    """Exact parity for 1-query & 3-query BlockAttnRes and 4-query MessageAttnRes."""
    pt_model, mt_model = self._build_pair(seed=1)
    rng = np.random.RandomState(2)
    b, t, h = 2, 6, self._EMB
    blocks_np = rng.randn(b, t, 2, h).astype(np.float32)
    partial_np = rng.randn(b, t, h).astype(np.float32)

    # 1-query BlockAttnRes (layer 2 sliding)
    pt_res1 = pt_model.model.layers[2].attention_input
    mt_res1 = mt_model.decoder.layers_2.attention_input
    dy1 = rng.randn(b, t, h).astype(np.float32)

    pt_b1 = torch.tensor(blocks_np, requires_grad=True)
    pt_p1 = torch.tensor(partial_np, requires_grad=True)
    out_pt1 = pt_res1(pt_b1, pt_p1)[0]
    (out_pt1 * torch.tensor(dy1)).sum().backward()

    def _mt_fn1(blk, par):
      out = mt_res1(blk, par)
      return jnp.sum(out * jnp.asarray(dy1)), out

    (_, out_mt1), (gb1, gp1) = jax.value_and_grad(_mt_fn1, argnums=(0, 1), has_aux=True)(
        jnp.asarray(blocks_np), jnp.asarray(partial_np)
    )
    _assert_close(out_mt1, out_pt1, msg="BlockAttnRes(1Q) forward")
    _assert_close(gb1, pt_b1.grad, msg="BlockAttnRes(1Q) d_blocks")
    _assert_close(gp1, pt_p1.grad, msg="BlockAttnRes(1Q) d_partial")

    # 3-query BlockAttnRes (layer 1 full)
    pt_res3 = pt_model.model.layers[1].attention_input
    mt_res3 = mt_model.decoder.layers_1.attention_input
    dy3 = [rng.randn(b, t, h).astype(np.float32) for _ in range(3)]

    pt_b3 = torch.tensor(blocks_np, requires_grad=True)
    pt_p3 = torch.tensor(partial_np, requires_grad=True)
    outs_pt3 = pt_res3(pt_b3, pt_p3)
    sum((o * torch.tensor(d)).sum() for o, d in zip(outs_pt3, dy3)).backward()

    def _mt_fn3(blk, par):
      outs = mt_res3(blk, par)
      return sum(jnp.sum(o * jnp.asarray(d)) for o, d in zip(outs, dy3)), outs

    (_, outs_mt3), (gb3, gp3) = jax.value_and_grad(_mt_fn3, argnums=(0, 1), has_aux=True)(
        jnp.asarray(blocks_np), jnp.asarray(partial_np)
    )
    for q_idx in range(3):
      _assert_close(outs_mt3[q_idx], outs_pt3[q_idx], msg=f"BlockAttnRes(3Q) forward stream {q_idx}")
    _assert_close(gb3, pt_b3.grad, msg="BlockAttnRes(3Q) d_blocks")
    _assert_close(gp3, pt_p3.grad, msg="BlockAttnRes(3Q) d_partial")

    # MessageAttnRes across all core layers (2 core blocks -> prev_core_blocks has 1 block + prev_partial)
    pt_msg = pt_model.model.message_input
    mt_msg = mt_model.decoder.message_input
    prev_blk_np = rng.randn(b, t, 1, h).astype(np.float32)
    prev_par_np = rng.randn(b, t, h).astype(np.float32)
    n_core = self._CORE_END - self._CORE_START
    dy_msg = rng.randn(b, t, n_core, 4, h).astype(np.float32)

    pt_mb = torch.tensor(prev_blk_np, requires_grad=True)
    pt_mp = torch.tensor(prev_par_np, requires_grad=True)
    pt_msgs = torch.stack([pt_msg(pt_mb, pt_mp, idx) for idx in range(n_core)], dim=2)
    (pt_msgs * torch.tensor(dy_msg)).sum().backward()

    def _mt_msg_fn(blk, par):
      out = mt_msg(blk, par)
      return jnp.sum(out * jnp.asarray(dy_msg)), out

    (_, mt_msgs), (gmb, gmp) = jax.value_and_grad(_mt_msg_fn, argnums=(0, 1), has_aux=True)(
        jnp.asarray(prev_blk_np), jnp.asarray(prev_par_np)
    )
    _assert_close(mt_msgs, pt_msgs, msg="MessageAttnRes forward")
    _assert_close(gmb, pt_mb.grad, msg="MessageAttnRes d_blocks")
    _assert_close(gmp, pt_mp.grad, msg="MessageAttnRes d_partial")

  def test_message_gate_forward_and_vjp(self):
    """Exact parity for rank-R `Inkling2MessageGate`."""
    pt_model, mt_model = self._build_pair(seed=2)
    pt_gate = pt_model.model.layers[2].q_message_gate
    mt_gate = mt_model.decoder.layers_2.q_message_gate

    rng = np.random.RandomState(3)
    b, t, h = 2, 6, self._EMB
    dest_np = rng.randn(b, t, h).astype(np.float32)
    msg_np = rng.randn(b, t, h).astype(np.float32)
    dy_np = rng.randn(b, t, h).astype(np.float32)

    pt_d = torch.tensor(dest_np, requires_grad=True)
    pt_m = torch.tensor(msg_np, requires_grad=True)
    out_pt = pt_gate(pt_d, pt_m)
    (out_pt * torch.tensor(dy_np)).sum().backward()

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

      def _mt_fn(d_arr, m_arr):
        out = mt_gate(d_arr, m_arr)
        return jnp.sum(out * jnp.asarray(dy_np)), out

      (_, out_mt), (gd, gm) = jax.value_and_grad(_mt_fn, argnums=(0, 1), has_aux=True)(
          jnp.asarray(dest_np), jnp.asarray(msg_np)
      )
    _assert_close(out_mt, out_pt, msg="MessageGate forward")
    _assert_close(gd, pt_d.grad, msg="MessageGate d_dest")
    _assert_close(gm, pt_m.grad, msg="MessageGate d_msg")

  def test_static_and_dynamic_short_convolutions_with_packed_segments(self):
    """Verifies static (w[3]=lag0, /sqrt(5)) & dynamic (static_w[0]=lag0, /3) sconv + packing."""
    pt_model, mt_model = self._build_pair(seed=3)
    rng = np.random.RandomState(4)
    b, t = 2, 10
    seg_np = np.array([[1, 1, 1, 1, 2, 2, 2, 2, 0, 0], [1, 1, 2, 2, 2, 2, 2, 0, 0, 0]], dtype=np.int32)
    conv_mask_pt = torch.tensor(seg_np != 0, dtype=torch.bool)

    # 1) Static post_attention_convolution (unpacked & packed doc2 vs unpacked doc2)
    pt_sconv = pt_model.model.layers[0].post_attention_convolution
    mt_sconv = mt_model.decoder.layers_0.post_attention_convolution
    x_np = rng.randn(b, t, self._EMB).astype(np.float32)
    # Run PT separately on each valid doc to get exact packed ground truth.
    with torch.no_grad():
      pt_doc1 = pt_sconv(torch.tensor(x_np[0:1, :4]))
      pt_doc2 = pt_sconv(torch.tensor(x_np[0:1, 4:8]))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):
      mt_out, _ = mt_sconv(jnp.asarray(x_np), decoder_segment_ids=jnp.asarray(seg_np))
    _assert_close(mt_out[0:1, :4], pt_doc1, msg="Static sconv packed doc1")
    _assert_close(mt_out[0:1, 4:8], pt_doc2, msg="Static sconv packed doc2 (zero cross-doc leak)")

    # 2) Dynamic k_sconv on full layer 1 (reversed lag order static_w[0]=lag0, /3, seq_idx isolation)
    pt_dconv = pt_model.model.layers[1].self_attn.k_proj.sconv
    mt_dconv = mt_model.decoder.layers_1.self_attention.k_sconv
    kv_x_np = rng.randn(b, t, self._HKV, self._HEAD_DIM).astype(np.float32)
    dyn_in_np = rng.randn(b, t, self._EMB).astype(np.float32)
    dy_kv = rng.randn(b, t, self._HKV, self._HEAD_DIM).astype(np.float32)

    pt_kv_t = torch.tensor(kv_x_np.reshape(b, t, -1), requires_grad=True)
    pt_dyn_t = torch.tensor(dyn_in_np, requires_grad=True)
    # Note: PT DynamicShortConvolution supports seq_idx directly!
    out_pt_d = pt_dconv(
        pt_kv_t, pt_dyn_t, conv_mask=conv_mask_pt, seq_idx=torch.tensor(seg_np, dtype=torch.int32)
    ).reshape(b, t, self._HKV, self._HEAD_DIM)
    # Zero out padded output positions before VJP comparison.
    valid_mask_pt = conv_mask_pt[:, :, None, None].float()
    ((out_pt_d * valid_mask_pt) * torch.tensor(dy_kv)).sum().backward()

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

      def _mt_d_fn(kv_arr, dyn_arr):
        out, _ = mt_dconv(kv_arr, dyn_arr, decoder_segment_ids=jnp.asarray(seg_np))
        mask = (jnp.asarray(seg_np) != 0)[:, :, None, None].astype(jnp.float32)
        return jnp.sum((out * mask) * jnp.asarray(dy_kv)), out

      (_, out_mt_d), (g_kv, g_dyn) = jax.value_and_grad(_mt_d_fn, argnums=(0, 1), has_aux=True)(
          jnp.asarray(kv_x_np), jnp.asarray(dyn_in_np)
      )
    _assert_close(out_mt_d[seg_np != 0], out_pt_d.detach().numpy()[seg_np != 0], msg="Dynamic sconv forward")
    _assert_close(
        g_kv[seg_np != 0],
        pt_kv_t.grad.reshape(b, t, self._HKV, self._HEAD_DIM).numpy()[seg_np != 0],
        msg="Dynamic sconv d_kv",
    )
    _assert_close(g_dyn[seg_np != 0], pt_dyn_t.grad.numpy()[seg_np != 0], msg="Dynamic sconv d_dyn")

  def test_moe_block_softcapped_swiglu_parity(self):
    """Exact parity for RoutedMoE (SiTU gate=4, up=25, sigmoid, bias, route_scale=8)."""
    pt_model, mt_model = self._build_pair(seed=4)
    pt_moe = pt_model.model.layers[0].mlp
    mt_moe = mt_model.decoder.layers_0.mlp
    rng = np.random.RandomState(5)
    x_np = rng.randn(2, 6, self._EMB).astype(np.float32)
    dy_np = rng.randn(2, 6, self._EMB).astype(np.float32)

    pt_x = torch.tensor(x_np, requires_grad=True)
    out_pt = pt_moe(pt_x)
    (out_pt * torch.tensor(dy_np)).sum().backward()

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

      def _mt_moe_fn(x_arr):
        out, _, _ = mt_moe(x_arr)
        return jnp.sum(out * jnp.asarray(dy_np)), out

      (_, out_mt), gx = jax.value_and_grad(_mt_moe_fn, has_aux=True)(jnp.asarray(x_np))
    _assert_close(out_mt, out_pt, msg="RoutedMoE SiTU forward")
    _assert_close(gx, pt_x.grad, msg="RoutedMoE SiTU dx")

  def test_sliding_and_full_attention_forward_and_vjp(self):
    """Isolated parity for `Inkling2Attention` (sliding + full with log-scaling & 3 streams)."""
    pt_model, mt_model = self._build_pair(seed=5)
    rng = np.random.RandomState(6)
    b, t, h = 2, self._SEQ_LEN, self._EMB  # t=10 > sliding_window=6 & > log_floor=4!
    q_np = rng.randn(b, t, h).astype(np.float32)
    k_np = rng.randn(b, t, h).astype(np.float32)
    v_np = rng.randn(b, t, h).astype(np.float32)
    dy_np = rng.randn(b, t, h).astype(np.float32)
    pos_np = np.broadcast_to(np.arange(t, dtype=np.int32)[None, :], (b, t))
    seg_np = np.ones((b, t), dtype=np.int32)

    q_idx = np.arange(t)[:, None]
    kv_idx = np.arange(t)[None, :]
    rel_dist_pt = torch.tensor(q_idx - kv_idx, dtype=torch.long)
    full_mask_pt = torch.tensor(np.where(q_idx >= kv_idx, 0.0, np.finfo(np.float32).min), dtype=torch.float32)[None, None]
    swa_mask_pt = torch.tensor(
        np.where((q_idx >= kv_idx) & ((q_idx - kv_idx) < self._SLIDING_WINDOW), 0.0, np.finfo(np.float32).min),
        dtype=torch.float32,
    )[None, None]
    pos_pt = torch.tensor(pos_np, dtype=torch.long)

    # 1) Sliding attention (layer 0) with 3 distinct streams
    pt_swa = pt_model.model.layers[0].self_attn
    mt_swa = mt_model.decoder.layers_0.self_attention
    pt_q = torch.tensor(q_np, requires_grad=True)
    pt_k = torch.tensor(k_np, requires_grad=True)
    pt_v = torch.tensor(v_np, requires_grad=True)
    out_pt_swa, _ = pt_swa(
        hidden_states=pt_q,
        key_hidden_states=pt_k,
        value_hidden_states=pt_v,
        attention_mask=swa_mask_pt,
        relative_distance=rel_dist_pt,
        position_ids=pos_pt,
    )
    (out_pt_swa * torch.tensor(dy_np)).sum().backward()

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

      def _mt_swa_fn(qa, ka, va):
        out, _ = mt_swa(qa, ka, va, decoder_segment_ids=jnp.asarray(seg_np), decoder_positions=jnp.asarray(pos_np))
        return jnp.sum(out * jnp.asarray(dy_np)), out

      (_, out_mt_swa), (gq_s, gk_s, gv_s) = jax.value_and_grad(_mt_swa_fn, argnums=(0, 1, 2), has_aux=True)(
          jnp.asarray(q_np), jnp.asarray(k_np), jnp.asarray(v_np)
      )
    _assert_close(out_mt_swa, out_pt_swa, msg="SlidingAttention forward")
    _assert_close(gq_s, pt_q.grad, msg="SlidingAttention dq")
    _assert_close(gk_s, pt_k.grad, msg="SlidingAttention dk")
    _assert_close(gv_s, pt_v.grad, msg="SlidingAttention dv")

    # 2) Full attention (layer 1) with dynamic K/V sconv + LogScale (pos 0..9 crosses log_floor=4!)
    pt_full = pt_model.model.layers[1].self_attn
    mt_full = mt_model.decoder.layers_1.self_attention
    pt_qf = torch.tensor(q_np, requires_grad=True)
    pt_kf = torch.tensor(k_np, requires_grad=True)
    pt_vf = torch.tensor(v_np, requires_grad=True)
    out_pt_full, _ = pt_full(
        hidden_states=pt_qf,
        key_hidden_states=pt_kf,
        value_hidden_states=pt_vf,
        attention_mask=full_mask_pt,
        relative_distance=rel_dist_pt,
        position_ids=pos_pt,
    )
    (out_pt_full * torch.tensor(dy_np)).sum().backward()

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

      def _mt_full_fn(qa, ka, va):
        out, _ = mt_full(qa, ka, va, decoder_segment_ids=jnp.asarray(seg_np), decoder_positions=jnp.asarray(pos_np))
        return jnp.sum(out * jnp.asarray(dy_np)), out

      (_, out_mt_full), (gq_f, gk_f, gv_f) = jax.value_and_grad(_mt_full_fn, argnums=(0, 1, 2), has_aux=True)(
          jnp.asarray(q_np), jnp.asarray(k_np), jnp.asarray(v_np)
      )
    _assert_close(out_mt_full, out_pt_full, msg="FullAttention forward")
    _assert_close(gq_f, pt_qf.grad, msg="FullAttention dq")
    _assert_close(gk_f, pt_kf.grad, msg="FullAttention dk")
    _assert_close(gv_f, pt_vf.grad, msg="FullAttention dv")

  def test_all_four_decoder_layer_variants_forward_and_vjp(self):
    """Isolated parity for all 4 DecoderLayer variants (sliding/full x non-loop/loop + boundaries)."""
    pt_model, mt_model = self._build_pair(seed=6)
    rng = np.random.RandomState(7)
    b, t, h = 2, self._SEQ_LEN, self._EMB
    pos_np = np.broadcast_to(np.arange(t, dtype=np.int32)[None, :], (b, t))
    seg_np = np.ones((b, t), dtype=np.int32)
    q_idx = np.arange(t)[:, None]
    kv_idx = np.arange(t)[None, :]
    rel_dist_pt = torch.tensor(q_idx - kv_idx, dtype=torch.long)
    full_mask_pt = torch.tensor(np.where(q_idx >= kv_idx, 0.0, np.finfo(np.float32).min), dtype=torch.float32)[None, None]
    swa_mask_pt = torch.tensor(
        np.where((q_idx >= kv_idx) & ((q_idx - kv_idx) < self._SLIDING_WINDOW), 0.0, np.finfo(np.float32).min),
        dtype=torch.float32,
    )[None, None]
    pos_pt = torch.tensor(pos_np, dtype=torch.long)

    # (layer_idx, num_blocks_in, is_full, is_loop, with_messages)
    cases = [
        (0, 0, False, False, False),  # Layer 0: sliding non-loop, boundary, attention_input=None
        (1, 1, True, False, False),  # Layer 1: full non-loop, non-boundary
        (2, 1, False, True, True),  # Layer 2: sliding loop, boundary, WITH messages
        (3, 2, True, True, True),  # Layer 3: full loop, non-boundary, WITH messages
    ]
    for layer_idx, n_blocks, is_full, is_loop, with_msgs in cases:
      with self.subTest(layer=layer_idx, is_full=is_full, is_loop=is_loop):
        x_np = rng.randn(b, t, h).astype(np.float32)
        blk_np = rng.randn(b, t, n_blocks, h).astype(np.float32)
        msg_np = rng.randn(b, t, 4, h).astype(np.float32) if with_msgs else None
        dy_np = rng.randn(b, t, h).astype(np.float32)

        pt_layer = pt_model.model.layers[layer_idx]
        mt_layer = getattr(mt_model.decoder, f"layers_{layer_idx}")

        pt_x = torch.tensor(x_np, requires_grad=True)
        pt_b = torch.tensor(blk_np, requires_grad=True)
        pt_kwargs = {
            "attention_mask": full_mask_pt if is_full else swa_mask_pt,
            "relative_distance": rel_dist_pt,
            "position_ids": pos_pt,
        }
        if with_msgs:
          pt_m = torch.tensor(msg_np, requires_grad=True)
          out_pt = pt_layer(pt_x, pt_b, pt_m, **pt_kwargs)
        else:
          pt_m = None
          out_pt = pt_layer(pt_x, pt_b, **pt_kwargs)
        (out_pt * torch.tensor(dy_np)).sum().backward()

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):

          def _mt_layer_fn(xa, ba, ma, _layer=mt_layer, _dy=dy_np):
            h_out, b_out, _, _ = _layer(
                xa,
                ba,
                jnp.asarray(seg_np),
                jnp.asarray(pos_np),
                True,
                MODEL_MODE_TRAIN,
                layer_messages=ma,
            )
            return jnp.sum(h_out * jnp.asarray(_dy)), (h_out, b_out)

          if with_msgs:
            (_, (out_mt, blk_out_mt)), (gx, gb, gm) = jax.value_and_grad(_mt_layer_fn, argnums=(0, 1, 2), has_aux=True)(
                jnp.asarray(x_np), jnp.asarray(blk_np), jnp.asarray(msg_np)
            )
            _assert_close(gm, pt_m.grad, atol=2e-4, rtol=2e-4, msg=f"L{layer_idx} d_messages")
          else:
            (_, (out_mt, blk_out_mt)), (gx, gb) = jax.value_and_grad(_mt_layer_fn, argnums=(0, 1), has_aux=True)(
                jnp.asarray(x_np), jnp.asarray(blk_np), None
            )

        expected_blocks = n_blocks + (1 if layer_idx % self._BLOCK_SIZE == 0 else 0)
        self.assertEqual(blk_out_mt.shape[2], expected_blocks)
        _assert_close(out_mt, out_pt, atol=2e-4, rtol=2e-4, msg=f"L{layer_idx} forward")
        _assert_close(gx, pt_x.grad, atol=2e-4, rtol=2e-4, msg=f"L{layer_idx} dx")
        if n_blocks > 0:
          _assert_close(gb, pt_b.grad, atol=2e-4, rtol=2e-4, msg=f"L{layer_idx} d_blocks")

  def test_full_synthetic_model_forward_loss_vjp_and_perturb_controls(self):
    """Full 8-layer (2-core-block, 3-pass = 16 logical visits) parity + VJP + PerturbSpec."""
    pt_model, mt_u = self._build_pair(seed=10)
    mt_cfg_s = self._make_mt_config(scan_layers=True)
    mt_s = model_creation_utils.from_config(mt_cfg_s, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _copy_unscanned_to_scanned(mt_u, mt_s, mt_cfg_s)

    rng = np.random.RandomState(12)
    b, t = 2, self._SEQ_LEN  # t=10 > sliding_window=6 AND > log_scaling_floor=4!
    tokens_np = rng.randint(1, self._UNPADDED_VOCAB, size=(b, t)).astype(np.int32)
    targets_np = rng.randint(1, self._UNPADDED_VOCAB, size=(b, t)).astype(np.int32)
    pos_np = np.broadcast_to(np.arange(t, dtype=np.int32)[None, :], (b, t))
    seg_np = np.ones((b, t), dtype=np.int32)

    # PyTorch forward + CE loss + backward (including dx wrt inputs_embeds)
    pt_model.zero_grad()
    pt_inputs_embeds = (
        pt_model.model.embed_tokens(torch.tensor(tokens_np, dtype=torch.long)).detach().requires_grad_(True)
    )
    pt_out = pt_model(inputs_embeds=pt_inputs_embeds)
    pt_logits = pt_out.logits  # [B, T, unpadded_vocab]
    pt_loss = F.cross_entropy(
        pt_logits.reshape(-1, self._UNPADDED_VOCAB), torch.tensor(targets_np, dtype=torch.long).reshape(-1)
    )
    pt_loss.backward()
    pt_dx = pt_inputs_embeds.grad.detach().cpu().numpy()

    # Also compute full PT param grads from input_ids so embed_tokens.weight gets grad too.
    pt_model.zero_grad()
    pt_logits_ids = pt_model(input_ids=torch.tensor(tokens_np, dtype=torch.long)).logits
    pt_loss_ids = F.cross_entropy(
        pt_logits_ids.reshape(-1, self._UNPADDED_VOCAB), torch.tensor(targets_np, dtype=torch.long).reshape(-1)
    )
    pt_loss_ids.backward()
    pt_grads = _pt_grads_to_mt_dict(pt_model, self.mt_cfg)

    # MaxText unscanned & scanned forward + train.loss_fn + VJP grads
    batch_data = {
        "inputs": jnp.asarray(tokens_np),
        "inputs_position": jnp.asarray(pos_np),
        "inputs_segmentation": jnp.asarray(seg_np),
        "targets": jnp.asarray(targets_np),
        "targets_position": jnp.asarray(pos_np),
        "targets_segmentation": jnp.asarray(seg_np),
    }

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):
      mt_logits_u = mt_u(
          decoder_input_tokens=jnp.asarray(tokens_np),
          decoder_positions=jnp.asarray(pos_np),
          decoder_segment_ids=jnp.asarray(seg_np),
          model_mode=MODEL_MODE_TRAIN,
          enable_dropout=False,
      )
      (mt_loss_u, _), mt_grads_u = nnx.value_and_grad(
          pre_train.loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
      )(mt_u, self.mt_cfg, dict(batch_data), None, None, is_train=True)

      # Input embedding gradient `dx` parity (after embed_norm, matching pt_inputs_embeds)
      def _loss_from_emb(emb):
        logits = mt_u(
            decoder_input_tokens=jnp.asarray(tokens_np),
            decoder_positions=jnp.asarray(pos_np),
            decoder_segment_ids=jnp.asarray(seg_np),
            decoder_input_embeddings=emb,
            model_mode=MODEL_MODE_TRAIN,
            enable_dropout=False,
        )
        log_probs = jax.nn.log_softmax(logits[..., : self._UNPADDED_VOCAB].astype(jnp.float32), axis=-1)
        return -jnp.mean(jnp.take_along_axis(log_probs, jnp.asarray(targets_np)[..., None], axis=-1))

      mt_dx = jax.grad(_loss_from_emb)(_t2j(pt_inputs_embeds))

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(mt_cfg_s.logical_axis_rules):
      mt_logits_s = mt_s(
          decoder_input_tokens=jnp.asarray(tokens_np),
          decoder_positions=jnp.asarray(pos_np),
          decoder_segment_ids=jnp.asarray(seg_np),
          model_mode=MODEL_MODE_TRAIN,
          enable_dropout=False,
      )

    v = self._UNPADDED_VOCAB
    cos_u, kl_u = _cosine_and_kl(np.asarray(mt_logits_u[..., :v]), pt_logits_ids.detach().numpy())
    cos_s, kl_s = _cosine_and_kl(np.asarray(mt_logits_s[..., :v]), pt_logits_ids.detach().numpy())
    self.assertGreaterEqual(cos_u, 0.99999, f"Unscanned cosine too low: {cos_u}")
    self.assertLessEqual(kl_u, 1e-6, f"Unscanned KL too high: {kl_u}")
    self.assertGreaterEqual(cos_s, 0.99999, f"Scanned cosine too low: {cos_s}")
    self.assertLessEqual(kl_s, 1e-6, f"Scanned KL too high: {kl_s}")
    _assert_close(mt_logits_u[..., :v], pt_logits_ids, msg="Full synthetic unscanned logits")
    _assert_close(mt_logits_s[..., :v], pt_logits_ids, msg="Full synthetic scanned logits")
    _assert_close(mt_loss_u, pt_loss_ids, msg="Full synthetic train.loss_fn CE parity")
    _assert_close(mt_dx, pt_dx, atol=2e-4, rtol=2e-4, msg="Input embedding dx parity")

    # Per-leaf VJP gradient parity for EVERY nnx.Param!
    mt_grad_dict = {
        tuple(map(str, p)): np.asarray(val.get_value(), dtype=np.float32) for p, val in nnx.to_flat_state(mt_grads_u)
    }
    self.assertEqual(set(mt_grad_dict.keys()), set(pt_grads.keys()), "Param gradient key bijection failed")
    for key, g_pt in pt_grads.items():
      g_mt = mt_grad_dict[key]
      self.assertTrue(np.all(np.isfinite(g_mt)), f"Non-finite grad at {key}")
      ref_norm = float(np.linalg.norm(g_pt))
      rel_l2 = float(np.linalg.norm(g_mt - g_pt)) / max(ref_norm, 1e-8)
      self.assertLess(rel_l2, 1e-3, f"Grad rel L2 mismatch at {key}: rel_l2={rel_l2:.3e}")
      _assert_close(g_mt, g_pt, atol=5e-4, rtol=5e-4, msg=f"Grad mismatch at {key}")

    # PerturbSpec negative controls: perturbing any critical Inkling2 weight MUST trip parity!
    perturb_specs = [
        PerturbSpec("q_l2_clip_scale", lambda m: _perturb_param_leaf(m, "self_attention/q_l2_clip_scale")),
        PerturbSpec("message_input.query", lambda m: _perturb_param_leaf(m, "message_input/query")),
        PerturbSpec("forget_gate_proj", lambda m: _perturb_param_leaf(m, "q_message_gate/forget_gate_proj/kernel")),
        PerturbSpec("rel_logits_proj", lambda m: _perturb_param_leaf(m, "self_attention/rel_logits_proj")),
        PerturbSpec("k_sconv_static", lambda m: _perturb_param_leaf(m, "self_attention/k_sconv/kernel")),
        PerturbSpec("k_sconv_dynamic", lambda m: _perturb_param_leaf(m, "self_attention/k_sconv/static_w")),
        PerturbSpec("attention_input.query", lambda m: _perturb_param_leaf(m, "attention_input/query")),
        PerturbSpec("mlp.gate.kernel", lambda m: _perturb_param_leaf(m, "mlp/gate/kernel")),
    ]
    for spec in perturb_specs:
      with self.subTest(perturb=spec.name):
        perturbed = spec.apply(mt_u)
        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.mt_cfg.logical_axis_rules):
          bad_logits = np.asarray(
              perturbed(
                  decoder_input_tokens=jnp.asarray(tokens_np),
                  decoder_positions=jnp.asarray(pos_np),
                  decoder_segment_ids=jnp.asarray(seg_np),
                  model_mode=MODEL_MODE_TRAIN,
                  enable_dropout=False,
              )
          )
        with self.assertRaises(AssertionError, msg=f"PerturbSpec {spec.name!r} failed to trip parity!"):
          _assert_close(bad_logits[..., :v], pt_logits_ids, atol=1e-4, rtol=1e-4)


# ==============================================================================
# Suite 2: Full Mini-Model-Verification on `hf-internal-testing/tiny-inkling2-mini`
# ==============================================================================


@unittest.skipUnless(
    bool(_DEFAULT_TINY_CKPT_DIR) and os.path.isdir(_DEFAULT_TINY_CKPT_DIR),
    "Set INKLING2_TINY_CKPT_DIR to a local hf-internal-testing/tiny-inkling2-mini checkout to run",
)
class Inkling2TinyCheckpointVerificationTest(unittest.TestCase):
  """End-to-end Mini-Model-Verification on real `hf-internal-testing/tiny-inkling2-mini`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._prev_matmul = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    cls._tmpdir = tempfile.mkdtemp(prefix="inkling2_tiny_verify_")

    with open(os.path.join(_DEFAULT_TINY_CKPT_DIR, "config.json"), "rt", encoding="utf-8") as f:
      raw_cfg = json.load(f)
    cls.tc = raw_cfg["text_config"]
    cls.unpadded_vocab = cls.tc["unpadded_vocab_size"]
    cls.vocab_size = cls.tc["vocab_size"]

    cls.tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TINY_CKPT_DIR)
    cls.pt_model = modeling_inkling2_loop.Inkling2LoopForCausalLM.from_pretrained(
        _DEFAULT_TINY_CKPT_DIR, dtype=torch.float32
    ).eval()

    cls.mt_cfg_unscanned = cls._build_tiny_mt_cfg(scan_layers=False)
    cls.mt_cfg_scanned = cls._build_tiny_mt_cfg(scan_layers=True)
    cls.mesh = _make_mesh(cls.mt_cfg_unscanned)

    # Convert real tiny-inkling2-mini (NVFP4 L0-13 + MXFP8 L14-17) via production `to_maxtext.main`!
    cls.ckpt_dir_unscanned = os.path.join(cls._tmpdir, "orbax_unscanned")
    cls.ckpt_dir_scanned = os.path.join(cls._tmpdir, "orbax_scanned")
    cls._run_to_maxtext(cls.ckpt_dir_unscanned, scan_layers=False)
    cls._run_to_maxtext(cls.ckpt_dir_scanned, scan_layers=True)

    cls.mt_unscanned = cls._load_orbax_model(cls.mt_cfg_unscanned, cls.ckpt_dir_unscanned)
    cls.mt_scanned = cls._load_orbax_model(cls.mt_cfg_scanned, cls.ckpt_dir_scanned)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._tmpdir, ignore_errors=True)
    jax.config.update("jax_default_matmul_precision", cls._prev_matmul)
    super().tearDownClass()

  @classmethod
  def _tiny_overrides(cls, scan_layers: bool = False, **extra):
    """Returns MaxText config overrides matching `tiny-inkling2-mini/config.json`."""
    tc = cls.tc
    lc = tc["loop_config"]
    overrides = {
        "model_name": "inkling2-mini",
        "override_model_config": True,
        "base_emb_dim": tc["hidden_size"],
        "base_num_query_heads": tc["num_attention_heads"],
        "base_num_kv_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "base_mlp_dim": tc["intermediate_size"],
        "base_moe_mlp_dim": tc["intermediate_size"],
        "base_num_decoder_layers": tc["num_hidden_layers"],
        "attn_res_block_size": tc["n_layers_per_attn_res_block"],
        "inkling2_core_start": lc["core_start"],
        "inkling2_core_end": lc["core_end"],
        "inkling2_num_passes": lc["num_passes"],
        "inkling2_message_gate_rank": lc["gate_rank"],
        "inkling2_d_rel": tc["d_rel"],
        "inkling2_rel_extent": tc["rel_extent"],
        "sliding_window_size": tc["sliding_window_size"],
        "short_conv_kernel_size": tc["sconv_kernel_size"],
        "inkling2_log_scaling_n_floor": float(tc["log_scaling_n_floor"]),
        "inkling2_log_scaling_alpha": float(tc["log_scaling_alpha"]),
        "vocab_size": tc["vocab_size"],
        "unpadded_vocab_size": tc["unpadded_vocab_size"],
        "num_experts": tc["n_routed_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "shared_experts": tc["n_shared_experts"],
        "sparse_matmul": False,
        "megablox": False,
        "capacity_factor": -1,
        "per_device_batch_size": 1,
        "max_target_length": 256,
        "max_prefill_predict_length": 32,
        "scan_layers": scan_layers,
        "remat_policy": "none",
        "packing": False,
        "enable_checkpointing": False,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "dropout_rate": 0.0,
        "skip_jax_distributed_system": True,
        "hardware": "cpu",
    }
    overrides.update(extra)
    return overrides

  @classmethod
  def _build_tiny_mt_cfg(cls, scan_layers: bool = False, **extra):
    return pyconfig.initialize(["", _BASE_CONFIG_PATH], **cls._tiny_overrides(scan_layers=scan_layers, **extra))

  @classmethod
  def _run_to_maxtext(cls, out_dir: str, scan_layers: bool):
    """Converts local `tiny-inkling2-mini` safetensors into Orbax via `to_maxtext.main`."""
    ov = cls._tiny_overrides(scan_layers=scan_layers)
    argv = [
        "to_maxtext.py",
        _BASE_CONFIG_PATH,
        f"base_output_directory={out_dir}",
        "run_name=convert",
    ]
    for k, v in ov.items():
      if isinstance(v, bool):
        argv.append(f"{k}={'true' if v else 'false'}")
      else:
        argv.append(f"{k}={v}")
    to_maxtext.main(
        args=argv,
        lazy_load_tensors=True,
        eager_load_method="safetensors",
        hf_model_path=_DEFAULT_TINY_CKPT_DIR,
        save_dtype="float32",
        simulated_cpu_devices_count=1,
    )

  @classmethod
  def _load_orbax_model(cls, cfg, out_dir: str):
    """Builds a fresh MaxText Transformer and populates it from the converted Orbax checkpoint."""
    model = model_creation_utils.from_config(cfg, mesh=cls.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    ckpt_items = os.path.join(out_dir, "0", "items")
    if not os.path.isdir(ckpt_items):
      ckpt_items = os.path.join(out_dir, "convert", "checkpoints", "0", "items")
    checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            use_ocdbt=cfg.checkpoint_storage_use_ocdbt,
            use_zarr3=cfg.checkpoint_storage_use_zarr3,
        )
    )
    restored = checkpointer.restore(ckpt_items)
    top = restored.get("params", restored)
    params_tree = top.get("params", top) if isinstance(top, dict) else top
    bias_tree = top.get("MoEBiasVar", restored.get("MoEBiasVar", {})) if isinstance(top, dict) else {}
    # Update Param and MoEBiasVar leaves from restored Orbax tree.
    var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
    state = nnx.state(model, var_filter)
    new_flat = []
    for path, var in nnx.to_flat_state(state):
      cur = bias_tree if isinstance(var, moe.MoEBiasVar) else params_tree
      for part in path:
        cur = cur[str(part)]
      arr = jnp.asarray(cur, dtype=var.get_value().dtype)
      new_flat.append((path, var.replace(arr)))
    nnx.update(model, nnx.from_flat_state(new_flat))
    return model

  def _mt_logits(self, model, cfg, tokens, positions=None, segment_ids=None):
    b, t = tokens.shape
    if positions is None:
      positions = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      return np.asarray(
          model(
              decoder_input_tokens=tokens,
              decoder_positions=positions,
              decoder_segment_ids=segment_ids,
              model_mode=MODEL_MODE_TRAIN,
              enable_dropout=False,
          ),
          dtype=np.float32,
      )

  def test_golden_prompt_logits_unscanned_and_scanned_match_hf(self):
    """Real `tiny-inkling2-mini` logits match PyTorch FP32 & HF published golden."""
    messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort="none",
    )
    ids_pt = inputs["input_ids"]
    ids_j = jnp.asarray(ids_pt.numpy(), dtype=jnp.int32)

    with torch.no_grad():
      logits_pt = self.pt_model(input_ids=ids_pt).logits.numpy()

    logits_u = self._mt_logits(self.mt_unscanned, self.mt_cfg_unscanned, ids_j)[..., : self.unpadded_vocab]
    logits_s = self._mt_logits(self.mt_scanned, self.mt_cfg_scanned, ids_j)[..., : self.unpadded_vocab]

    for tag, mt_l in (("unscanned", logits_u), ("scanned", logits_s)):
      cos, kl = _cosine_and_kl(mt_l, logits_pt)
      self.assertGreaterEqual(cos, 0.9999, f"[{tag}] cosine={cos}")
      self.assertLessEqual(kl, 1e-4, f"[{tag}] kl={kl}")
      np.testing.assert_allclose(mt_l, logits_pt, atol=5e-3, rtol=1e-3, err_msg=f"[{tag}] logits vs PT FP32")
      # Also cross-check against HF's published CUDA BF16 golden logits[:5].
      np.testing.assert_allclose(
          mt_l[0, -1, :5],
          _HF_TINY_PUBLISHED_GOLDEN_LOGITS,
          atol=0.1,
          rtol=0.0,
          err_msg=f"[{tag}] first-step logits[:5] vs HF published golden",
      )

  def test_boundary_inputs_padding_packing_and_long_positions(self):
    """Tests seq_len=1, non-tile, >sliding_window, >log_scaling_n_floor, padding, packing, vocab bounds."""
    v = self.unpadded_vocab

    # 1) seq_len=1, non-tile seq_len=7, vocab boundary IDs (0 and unpadded_vocab_size - 1)
    for name, toks_np in (
        ("seq1", np.array([[42]], dtype=np.int32)),
        ("vocab_bounds", np.array([[0, 1, v - 2, v - 1, 17, 100, 5000]], dtype=np.int32)),
    ):
      with self.subTest(case=name):
        with torch.no_grad():
          pt_l = self.pt_model(input_ids=torch.tensor(toks_np, dtype=torch.long)).logits.numpy()
        mt_l = self._mt_logits(self.mt_unscanned, self.mt_cfg_unscanned, jnp.asarray(toks_np))[..., :v]
        cos, kl = _cosine_and_kl(mt_l, pt_l)
        self.assertGreaterEqual(cos, 0.9999)
        self.assertLessEqual(kl, 1e-4)

    # 2) Long sequence (> sliding_window=128) AND high positions (> log_scaling_n_floor=8192)
    rng = np.random.RandomState(202)
    long_len = 140
    long_toks = rng.randint(1, v, size=(1, long_len)).astype(np.int32)
    high_pos = (np.arange(long_len, dtype=np.int32) + 8180)[None, :]  # crosses 8192!
    with torch.no_grad():
      pt_long = self.pt_model(
          input_ids=torch.tensor(long_toks, dtype=torch.long),
          position_ids=torch.tensor(high_pos, dtype=torch.long),
      ).logits.numpy()
    mt_long = self._mt_logits(
        self.mt_unscanned, self.mt_cfg_unscanned, jnp.asarray(long_toks), positions=jnp.asarray(high_pos)
    )[..., :v]
    cos, kl = _cosine_and_kl(mt_long, pt_long)
    self.assertGreaterEqual(cos, 0.9999, f"Long/high-pos cosine={cos}")
    self.assertLessEqual(kl, 1e-4, f"Long/high-pos kl={kl}")

    # 3) Right-padding & left-padding invariance + batch vs single-row parity (mean_batch_diff <= 1e-4)
    doc1 = np.array([[11, 22, 33, 44, 55]], dtype=np.int32)
    doc2 = np.array([[66, 77, 88]], dtype=np.int32)
    with torch.no_grad():
      pt_d1 = self.pt_model(input_ids=torch.tensor(doc1, dtype=torch.long)).logits.numpy()
      pt_d2 = self.pt_model(input_ids=torch.tensor(doc2, dtype=torch.long)).logits.numpy()

    # Right-padded batch (row 0 full, row 1 right-padded) and left-padded row
    rp_toks = jnp.array([[11, 22, 33, 44, 55], [66, 77, 88, 999, 1234]], dtype=jnp.int32)
    rp_pos = jnp.array([[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]], dtype=jnp.int32)
    rp_seg = jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=jnp.int32)
    mt_rp = self._mt_logits(self.mt_unscanned, self.mt_cfg_unscanned, rp_toks, rp_pos, rp_seg)[..., :v]
    self.assertLessEqual(float(np.mean(np.abs(mt_rp[0:1, :5] - pt_d1))), 1e-3)
    self.assertLessEqual(float(np.mean(np.abs(mt_rp[1:2, :3] - pt_d2))), 1e-3)

    lp_toks = jnp.array([[999, 1234, 66, 77, 88]], dtype=jnp.int32)
    lp_pos = jnp.array([[0, 0, 0, 1, 2]], dtype=jnp.int32)
    lp_seg = jnp.array([[0, 0, 1, 1, 1]], dtype=jnp.int32)
    mt_lp = self._mt_logits(self.mt_unscanned, self.mt_cfg_unscanned, lp_toks, lp_pos, lp_seg)[..., :v]
    self.assertLessEqual(float(np.mean(np.abs(mt_lp[:, 2:5] - pt_d2))), 1e-3)
    self.assertLessEqual(float(np.mean(np.abs(mt_lp[:, 2:5] - mt_rp[1:2, :3]))), 1e-4)

    # 4) Packed multi-doc [1,1,1,1,1, 2,2,2, 0,0] vs unpacked PyTorch single docs
    packed_toks = jnp.array([[11, 22, 33, 44, 55, 66, 77, 88, 0, 0]], dtype=jnp.int32)
    packed_pos = jnp.array([[0, 1, 2, 3, 4, 0, 1, 2, 0, 0]], dtype=jnp.int32)
    packed_seg = jnp.array([[1, 1, 1, 1, 1, 2, 2, 2, 0, 0]], dtype=jnp.int32)
    mt_pk = self._mt_logits(self.mt_unscanned, self.mt_cfg_unscanned, packed_toks, packed_pos, packed_seg)[..., :v]
    self.assertTrue(np.all(np.isfinite(mt_pk)))
    np.testing.assert_allclose(mt_pk[:, :5], pt_d1, atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(mt_pk[:, 5:8], pt_d2, atol=5e-3, rtol=1e-3)

  def test_decomposed_loss_sft_vjp_and_perturb_controls_on_tiny(self):
    """Pure CE, SFT completion-only loss, per-leaf VJP grads, and PerturbSpec on tiny checkpoint."""
    v = self.unpadded_vocab
    seq_len = 8
    toks_np = np.array([[15, 42, 108, 7, 501, 23, 99, 4]], dtype=np.int32)
    tgts_np = np.array([[42, 108, 7, 501, 23, 99, 4, 19]], dtype=np.int32)
    pos_np = np.arange(seq_len, dtype=np.int32)[None, :]
    seg_np = np.ones((1, seq_len), dtype=np.int32)
    # SFT mask: prompt tokens 0..3 masked out (0), completion tokens 4..6 active (1), tail 7 pad (0).
    sft_tgt_seg = np.array([[0, 0, 0, 0, 1, 1, 1, 0]], dtype=np.int32)

    # Pure CE + VJP
    self.pt_model.zero_grad()
    pt_logits = self.pt_model(input_ids=torch.tensor(toks_np, dtype=torch.long)).logits
    pt_ce = F.cross_entropy(pt_logits.reshape(-1, v), torch.tensor(tgts_np, dtype=torch.long).reshape(-1))
    pt_ce.backward()
    pt_grads = _pt_grads_to_mt_dict(self.pt_model, self.mt_cfg_unscanned)

    # SFT completion-only CE in PyTorch
    pt_sft_labels = np.where(sft_tgt_seg == 1, tgts_np, -100)
    with torch.no_grad():
      pt_sft_loss = F.cross_entropy(
          pt_logits.detach().reshape(-1, v), torch.tensor(pt_sft_labels, dtype=torch.long).reshape(-1), ignore_index=-100
      )

    cfg = self.mt_cfg_unscanned
    batch_ce = {
        "inputs": jnp.asarray(toks_np),
        "inputs_position": jnp.asarray(pos_np),
        "inputs_segmentation": jnp.asarray(seg_np),
        "targets": jnp.asarray(tgts_np),
        "targets_position": jnp.asarray(pos_np),
        "targets_segmentation": jnp.asarray(seg_np),
    }
    batch_sft = dict(batch_ce, targets_segmentation=jnp.asarray(sft_tgt_seg))

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      (mt_ce, aux_ce), mt_grads = nnx.value_and_grad(
          pre_train.loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
      )(self.mt_unscanned, cfg, dict(batch_ce), None, None, is_train=True)
      mt_sft, aux_sft = pre_train.loss_fn(self.mt_unscanned, cfg, dict(batch_sft), None, None, is_train=True)

    _assert_close(mt_ce, pt_ce, atol=1e-3, rtol=1e-3, msg="tiny pure CE loss mismatch")
    _assert_close(mt_sft, pt_sft_loss, atol=1e-3, rtol=1e-3, msg="tiny SFT completion-only loss mismatch")
    self.assertAlmostEqual(float(aux_ce["total_weights"]), float(seq_len), places=5)
    self.assertAlmostEqual(float(aux_sft["total_weights"]), 3.0, places=5)

    # Every nnx.Param gradient matches PyTorch!
    mt_grad_dict = {
        tuple(map(str, p)): np.asarray(val.get_value(), dtype=np.float32) for p, val in nnx.to_flat_state(mt_grads)
    }
    self.assertEqual(set(mt_grad_dict.keys()), set(pt_grads.keys()))
    for key, g_pt in pt_grads.items():
      g_mt = mt_grad_dict[key]
      self.assertTrue(np.all(np.isfinite(g_mt)), f"Non-finite grad at {key}")
      ref_norm = float(np.linalg.norm(g_pt))
      if ref_norm > 1e-6:
        rel_l2 = float(np.linalg.norm(g_mt - g_pt)) / ref_norm
        self.assertLess(rel_l2, 2e-3, f"tiny grad rel_l2={rel_l2:.3e} at {key}")

    # PerturbSpec negative controls on real tiny weights (compared against clean MaxText logits).
    # Note: `tiny-inkling2-mini` has `num_core_blocks == 1`, so `message_input/query` is a 1-element softmax
    # (tested on 2-core-block synthetic above); here we perturb `final_input/query` (4-block pooling) and
    # active core/full layer parameters.
    clean_mt_l = self._mt_logits(self.mt_unscanned, cfg, jnp.asarray(toks_np))[..., :v]
    for suffix in (
        "layers_6/self_attention/q_l2_clip_scale",
        "final_input/query",
        "layers_6/q_message_gate/forget_gate_proj/kernel",
        "layers_6/self_attention/rel_logits_proj",
        "layers_6/self_attention/k_sconv/kernel",
        "layers_11/self_attention/k_sconv/static_w",
    ):
      with self.subTest(perturb=suffix):
        perturbed = _perturb_param_leaf(self.mt_unscanned, suffix, target_rel_norm=0.2)
        bad_l = self._mt_logits(perturbed, cfg, jnp.asarray(toks_np))[..., :v]
        with self.assertRaises(AssertionError, msg=f"PerturbSpec {suffix!r} failed to trip on tiny!"):
          np.testing.assert_allclose(bad_l, clean_mt_l, atol=1e-4, rtol=1e-4)

  def test_three_step_adamw_and_orbax_checkpoint_continuity_vs_pytorch(self):
    """3-step AdamW updates match PyTorch; Orbax save/restore after step 2 is bitwise identical."""
    cfg = self.mt_cfg_unscanned
    v = self.unpadded_vocab
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    wd = 0.01

    # Fresh copies so class fixtures stay untouched.
    pt_m = modeling_inkling2_loop.Inkling2LoopForCausalLM.from_pretrained(
        _DEFAULT_TINY_CKPT_DIR, dtype=torch.float32
    ).train()
    pt_opt = torch.optim.AdamW(pt_m.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)

    mt_m = self._load_orbax_model(cfg, self.ckpt_dir_unscanned)
    tx = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=wd)
    opt_state = tx.init(nnx.state(mt_m, nnx.Param))

    rng = np.random.RandomState(404)
    batches = [
        (rng.randint(1, v, size=(1, 6)).astype(np.int32), rng.randint(1, v, size=(1, 6)).astype(np.int32))
        for _ in range(3)
    ]

    def _mt_step(model, o_state, toks_np, tgts_np):
      pos = jnp.arange(toks_np.shape[1], dtype=jnp.int32)[None, :]
      seg = jnp.ones_like(pos)
      batch = {
          "inputs": jnp.asarray(toks_np),
          "inputs_position": pos,
          "inputs_segmentation": seg,
          "targets": jnp.asarray(tgts_np),
          "targets_position": pos,
          "targets_segmentation": seg,
      }
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        (loss, _), grads = nnx.value_and_grad(pre_train.loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(
            model, cfg, batch, None, None, is_train=True
        )
      params = nnx.state(model, nnx.Param)
      updates, new_o_state = tx.update(grads, o_state, params)
      new_params = optax.apply_updates(params, updates)
      nnx.update(model, new_params)
      return float(loss), new_o_state

    resume_dir = os.path.join(self._tmpdir, "step2_resume")
    checkpointer = ocp.StandardCheckpointer()
    step2_saved_model = None
    step2_saved_opt = None

    for step_idx, (toks_np, tgts_np) in enumerate(batches):
      # PyTorch step
      pt_opt.zero_grad()
      pt_l = pt_m(input_ids=torch.tensor(toks_np, dtype=torch.long)).logits
      pt_loss = F.cross_entropy(pt_l.reshape(-1, v), torch.tensor(tgts_np, dtype=torch.long).reshape(-1))
      pt_loss.backward()
      pt_opt.step()

      # MaxText step
      mt_loss, opt_state = _mt_step(mt_m, opt_state, toks_np, tgts_np)
      self.assertAlmostEqual(mt_loss, float(pt_loss.item()), delta=2e-3, msg=f"AdamW step {step_idx} loss mismatch")

      # Save Orbax checkpoint after step 2 (step_idx == 1) to test save->restore continuity on step 3.
      if step_idx == 1:
        save_payload = {
            "params": jax.tree.map(lambda x: x, nnx.to_pure_dict(nnx.state(mt_m, nnx.Param))),
            "opt_state": opt_state,
        }
        checkpointer.save(resume_dir, save_payload)
        checkpointer.wait_until_finished()

        restored = checkpointer.restore(resume_dir, target=save_payload)
        step2_saved_model = self._load_orbax_model(cfg, self.ckpt_dir_unscanned)
        restored_state = nnx.state(step2_saved_model, nnx.Param)
        nnx.replace_by_pure_dict(restored_state, restored["params"])
        nnx.update(step2_saved_model, restored_state)
        step2_saved_opt = restored["opt_state"]

    # Step 3 from restored Orbax state MUST be bitwise/near-bitwise identical to uninterrupted step 3!
    mt_loss_resumed, _ = _mt_step(step2_saved_model, step2_saved_opt, *batches[2])
    self.assertAlmostEqual(mt_loss_resumed, mt_loss, places=6, msg="Orbax save->restore step-3 loss drift")


if __name__ == "__main__":
  unittest.main()
