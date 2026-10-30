# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and parity tests for GLM-5.3-Flash checkpoint conversion, dequant, and Orbax roundtrip."""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import collections
from collections.abc import Mapping
import contextlib
import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import tracemalloc
from typing import Any
import unittest
from unittest import mock
import weakref

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import pytest
import tensorstore as ts

try:
  from maxtext.m3.models.glm5_3_flash import create_mini_slice
  from maxtext.m3.models.glm5_3_flash import glm5_ckpt
  import safetensors.torch
  import torch
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

try:
  jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError as e:
  if "before backends are initialized" not in str(e):
    raise


def _make_cfg(
    num_hidden_layers: int = 7,
    scan_layers: bool = True,
    dtype: str = "float32",
    weight_dtype: str = "float32",
):
  """Constructs matching mini Glm5Config and HF Glm5NextTextConfig."""
  kws = (
      "hidden_size=64 intermediate_size=128 moe_intermediate_size=32"
      " n_routed_experts=8 num_experts_per_tok=2 q_lora_rank=32 kv_lora_rank=16"
      " num_attention_heads=4 num_key_value_heads=4 qk_nope_head_dim=16"
      " v_head_dim=16 index_n_heads=8 index_head_dim=16 index_topk=8"
      " linear_num_key_heads=4 linear_num_value_heads=4"
      " linear_key_head_dim=16 linear_value_head_dim=16"
  )
  args = {k: int(v) for k, v in (s.split("=") for s in kws.split())}
  args.update(
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      kda_implementation="xla",
      vocab_size=256,
      dtype=dtype,
      weight_dtype=weight_dtype,
  )
  c = glm5_config.Glm5Config.mini_config(**args)
  hf_args = {x: args[x] for x in args if not x.startswith("linear_") and x != "qk_head_dim"}
  hf_k = (
      "vocab_size pad_token_id num_hidden_layers _attn_implementation"
      " index_kpool linear_num_heads linear_head_dim linear_conv_kernel_dim"
  ).split()
  hf_args.update(zip(hf_k, (256, 0, num_hidden_layers, "eager", 4, 4, 16, 4)))
  hf_args["mlp_layer_types"] = ["dense"] * min(3, num_hidden_layers) + ["sparse"] * max(0, num_hidden_layers - 3)
  hf_args["layer_types"], hf_args["indexer_types"] = list(c.layer_types), list(c.indexer_types)
  return c, (hf_cfg.Glm5NextTextConfig(**hf_args) if HAS_TORCH else None)


def _init_hf(hf_m: torch.nn.Module, hf_head: torch.nn.Module) -> None:
  """Initializes HF backbone and lm_head with deterministic random weights."""
  with torch.no_grad():
    for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers()):
      if p.is_floating_point():
        if "layernorm" in n or "norm" in n:
          if n.endswith(".bias"):
            p.zero_()
          else:
            p.fill_(1.0)
        elif "A_log" in n:
          p.copy_(torch.log(torch.rand_like(p) + 0.1))
        elif "dt_bias" in n:
          p.uniform_(-2.0, -0.5)
        else:
          p.normal_(0.0, 0.02)
    hf_head.weight.normal_(0.0, 0.02)


def _hf_forward(hf_m: torch.nn.Module, hf_head: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
  orig = hf_model.chunk_kimi_delta_attention
  hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
  try:
    return hf_head(hf_m(input_ids=ids).last_hidden_state)
  finally:
    hf_model.chunk_kimi_delta_attention = orig


def _hf_module_state_to_hub(
    state_dict: Mapping[str, torch.Tensor],
    *,
    num_experts: int,
    moe_intermediate_size: int,
) -> dict[str, torch.Tensor]:
  """Inverts transformers glm5_next conversion mapping to Hub safetensors format.

  Authority: transformers conversion_mapping.py:535-580.
  Converts PyTorch Glm5NextTextModel state_dict keys and fused tensors into
  the
  exact flat Hub safetensors layout under `model.language_model.` and
  `lm_head.`.
  """
  hub_dict: dict[str, torch.Tensor] = {}
  for orig_k, tensor in state_dict.items():
    if orig_k == "lm_head.weight":
      hub_dict["lm_head.weight"] = tensor
      continue
    k = orig_k[len("model.language_model.") :] if orig_k.startswith("model.language_model.") else orig_k

    if k.endswith(".self_attn.conv1d.weight"):
      prefix = k[: -len("self_attn.conv1d.weight")]
      q_t, k_t, v_t = torch.chunk(tensor, 3, dim=0)
      hub_dict[f"model.language_model.{prefix}self_attn.q_conv1d.weight"] = q_t
      hub_dict[f"model.language_model.{prefix}self_attn.k_conv1d.weight"] = k_t
      hub_dict[f"model.language_model.{prefix}self_attn.v_conv1d.weight"] = v_t
      continue

    if k.endswith(".mlp.experts.gate_up_proj"):
      prefix = k[: -len("mlp.experts.gate_up_proj")]
      for e in range(num_experts):
        ge, up_e = torch.split(tensor[e], moe_intermediate_size, dim=0)
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.gate_proj.weight"] = ge
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.up_proj.weight"] = up_e
      continue

    if k.endswith(".mlp.experts.down_proj"):
      prefix = k[: -len("mlp.experts.down_proj")]
      for e in range(num_experts):
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.down_proj.weight"] = tensor[e]
      continue

    k = k.replace("self_attn.forget_gate.f_a_proj.", "self_attn.f_a_proj.")
    k = k.replace("self_attn.forget_gate.f_b_proj.", "self_attn.f_b_proj.")
    k = k.replace("self_attn.forget_gate.dt_bias", "self_attn.dt_bias")
    k = k.replace("self_attn.forget_gate.A_log", "self_attn.A_log")

    k = k.replace("attn_hc.fn", "hc_attn_fn")
    k = k.replace("attn_hc.base", "hc_attn_base")
    k = k.replace("attn_hc.scale", "hc_attn_scale")
    k = k.replace("ffn_hc.fn", "hc_ffn_fn")
    k = k.replace("ffn_hc.base", "hc_ffn_base")
    k = k.replace("ffn_hc.scale", "hc_ffn_scale")

    hub_dict[f"model.language_model.{k}"] = tensor
  return hub_dict


@contextlib.contextmanager
def _fp64_mode():
  """Context manager promoting float32 casts in JAX and Torch/HF to float64."""
  orig_x64 = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(jnp, "float32", jnp.float64))
      if HAS_TORCH:
        stack.enter_context(mock.patch.object(torch, "float32", torch.float64))
        stack.enter_context(mock.patch.object(torch.Tensor, "float", torch.Tensor.double))
        stack.enter_context(
            mock.patch.object(
                hf_moe_integ,
                "_can_use_grouped_mm",
                lambda *args, **kwargs: False,
            )
        )
      yield
  finally:
    jax.config.update("jax_enable_x64", orig_x64)


def _path_to_tuple(path: tuple[Any, ...]) -> tuple[str | int, ...]:
  """Normalizes JAX tree path entries to a tuple of str | int."""
  out = []
  for k in path:
    if isinstance(k, jax.tree_util.DictKey):
      out.append(k.key)
    elif isinstance(k, jax.tree_util.GetAttrKey):
      out.append(k.name)
    elif isinstance(k, jax.tree_util.SequenceKey):
      out.append(k.idx)
    elif isinstance(k, (str, int)):
      out.append(k)
    else:
      raise TypeError(f"Unsupported key path element: {k!r} ({type(k)})")
  return tuple(out)


class _SafeOpenTracker:
  """Wraps safetensors.safe_open handles to observe shard lifecycle and tensor loads."""

  def __init__(self, handle, *, on_get, on_enter, on_exit):
    self._h = handle
    self._on_get = on_get
    self._on_enter = on_enter
    self._on_exit = on_exit

  def __enter__(self):
    self._on_enter()
    self._h.__enter__()
    return self

  def __exit__(self, *a):
    self._on_exit()
    return self._h.__exit__(*a)

  def get_tensor(self, k):
    t = self._h.get_tensor(k)
    self._on_get(t, k)
    return t

  def get_slice(self, k):
    return self._h.get_slice(k)

  def keys(self):
    return self._h.keys()


def _write_config_json(cfg_path: str, cfg: glm5_config.Glm5Config) -> None:
  """Writes non-CLI Glm5Config fields to a JSON configuration file."""
  with open(cfg_path, "w", encoding="utf-8") as fd:
    json.dump(
        {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "q_lora_rank": cfg.q_lora_rank,
            "kv_lora_rank": cfg.kv_lora_rank,
            "qk_nope_head_dim": cfg.qk_nope_head_dim,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "v_head_dim": cfg.v_head_dim,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "index_n_heads": cfg.index_n_heads,
            "index_head_dim": cfg.index_head_dim,
            "index_topk": cfg.index_topk,
            "first_k_dense_replace": cfg.first_k_dense_replace,
            "kda_implementation": cfg.kda_implementation,
            "layer_types": list(cfg.layer_types),
            "indexer_types": list(cfg.indexer_types),
        },
        fd,
    )


def _run_with_tracker(
    hf_dir: str,
    obx_dir: str,
    c: glm5_config.Glm5Config,
    *,
    on_get: Any,
    on_enter: Any,
    on_exit: Any,
) -> None:
  """Runs convert_hf_checkpoint_to_orbax with a _SafeOpenTracker hook on safe_open."""
  orig_open = safetensors.safe_open

  def side_fn(*a, **kw):
    return _SafeOpenTracker(
        orig_open(*a, **kw),
        on_get=on_get,
        on_enter=on_enter,
        on_exit=on_exit,
    )

  # Mock safe_open to track tensor memory usage.
  with mock.patch.object(safetensors, "safe_open", side_effect=side_fn):
    glm5_ckpt.convert_hf_checkpoint_to_orbax(
        hf_dir,
        obx_dir,
        c,
        ignore_mtp_layer=False,
        slice_to_config_layers=False,
        num_workers=1,
        max_inflight_bytes=64 * 1024 * 1024,
    )


def _write_sharded_ckpt(
    hf_dir: str, c: glm5_config.Glm5Config, *, num_shards: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
  """Writes synthetic sharded safetensors files matching Glm5Config."""
  abs_m = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False))
  b_dt = torch.float64 if c.weight_dtype == "float64" else torch.float32
  hub: dict[str, torch.Tensor] = {}
  for s in glm5_ckpt.iter_conversion_specs(c, include_mtp=c.num_nextn_predict_layers > 0):
    sh = glm5_ckpt.resolve_target_var(abs_m, s.target_path).value.shape
    sh = sh[1:] if s.stack_index is not None else sh
    dt = torch.float32 if any(s.hf_keys[0].endswith(x) for x in ("A_log", "dt_bias", "conv1d.weight")) else b_dt
    if s.kind == "direct":
      k_sh = sh
    elif s.kind == "transpose_2d":
      k_sh = (sh[1], sh[0])
    elif s.kind == "moe_gate_up":
      k_sh = (c.moe_intermediate_size, c.hidden_size)
    elif s.kind == "moe_down":
      k_sh = (c.hidden_size, c.moe_intermediate_size)
    else:
      raise ValueError(f"Unknown spec kind: {s.kind!r}")
    for k in s.hf_keys:
      hub[k] = torch.randn(k_sh, dtype=dt)
  if c.weight_dtype == "float64":
    hub["model.language_model.embed_tokens.weight"][0, 0] = 1.0 + 1e-12
  fp8_k = "model.language_model.layers.0.mlp.gate_proj.weight"
  sc_k = f"{fp8_k}_scale_inv"
  w_src = hub[fp8_k].to(torch.float32)
  m, n = w_src.shape
  sb_m, sb_n = math.ceil(m / 128), math.ceil(n / 128)
  s_inv = torch.linspace(0.5, 2.0, steps=sb_m * sb_n, dtype=torch.float32).reshape(sb_m, sb_n)
  exp_s = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
  hub[fp8_k], hub[sc_k] = (w_src / exp_s).to(torch.float8_e4m3fn), s_inv
  wmap = {fp8_k: "m-1.safetensors", sc_k: f"m-{min(2, num_shards)}.safetensors"}
  for idx, k in enumerate(sorted(k for k in hub if k not in (fp8_k, sc_k))):
    wmap[k] = f"m-{idx % num_shards + 1}.safetensors"
  for sf in sorted(set(wmap.values())):
    safetensors.torch.save_file(
        {k: hub[k] for k, v in wmap.items() if v == sf},
        os.path.join(hf_dir, sf),
    )
  with open(
      os.path.join(hf_dir, "model.safetensors.index.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump({"weight_map": wmap}, f)
  return hub, exp_s


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5CkptTest(parameterized.TestCase):
  """Tests for dequantize_fp8_block128, load_hf_state_into_model, and checkpoint utils."""

  def setUp(self):
    super().setUp()
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _assert_e3(self, a, b, name=""):
    """Asserts parameter parity using E3 relative Frobenius and cosine metrics."""
    a_f, b_f = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    ref_inf, norm_a, norm_b = (
        float(np.max(np.abs(b_f))),
        float(np.linalg.norm(a_f)),
        float(np.linalg.norm(b_f)),
    )
    if norm_b == 0.0:
      self.assertEqual(norm_a, 0.0, msg=f"{name}: expected zero grad")
      return
    self.assertGreater(norm_a, 0.0, msg=f"{name}: actual norm is zero")
    diff = float(np.linalg.norm(a_f - b_f))
    cos = float(np.sum(a_f * b_f) / (norm_a * norm_b))
    np.testing.assert_allclose(a_f, b_f, atol=1e-5 * max(1.0, ref_inf), rtol=1e-5, err_msg=name)
    self.assertLessEqual(diff, 1e-5 * max(1.0, norm_b), msg=f"{name} diff={diff}")
    self.assertGreater(cos, 0.99999, msg=f"{name} cos={cos}")

  def test_dequantize_fp8_block128(self):
    """Tests 2D and 3D block-128 dequantization, float8 validation, and shape mismatch guards."""
    w2d = torch.randn(150, 200).to(torch.float8_e4m3fn)
    s2d = np.random.randn(2, 2).astype(np.float32)
    out2d = glm5_ckpt.dequantize_fp8_block128(w2d, s2d, target_dtype=jnp.float32)
    self.assertIsInstance(out2d, np.ndarray)
    self.assertEqual(out2d.dtype, np.float32)
    self.assertEqual(out2d.shape, (150, 200))
    exp2d = w2d.to(torch.float32).numpy() * np.repeat(np.repeat(s2d, 128, axis=0), 128, axis=1)[:150, :200]
    np.testing.assert_allclose(np.array(out2d), exp2d, atol=1e-5, rtol=1e-5)

    w3d = torch.randn(4, 150, 200).to(torch.float8_e4m3fn)
    s3d = np.random.randn(4, 2, 2).astype(np.float32)
    self.assertEqual(
        glm5_ckpt.dequantize_fp8_block128(w3d, s3d, target_dtype=jnp.float32).shape,
        (4, 150, 200),
    )

    for bad_w, bad_s, pat in (
        (w2d, np.ones((3, 2)), "scale_inv shape"),
        (
            torch.ones((1, 2, 3, 4)).to(torch.float8_e4m3fn),
            np.ones((1, 2)),
            "Weight must be 2D or 3D",
        ),
    ):
      with self.assertRaisesRegex(ValueError, pat):
        glm5_ckpt.dequantize_fp8_block128(bad_w, bad_s, target_dtype=jnp.float32)

    # Dequantization with torch.bfloat16 scale_inv succeeds cleanly
    s2d_bf16 = torch.from_numpy(s2d).to(torch.bfloat16)
    out2d_bf16 = glm5_ckpt.dequantize_fp8_block128(w2d, s2d_bf16, target_dtype=jnp.float32)
    exp2d_bf16 = (
        w2d.to(torch.float32).numpy()
        * np.repeat(
            np.repeat(s2d_bf16.to(torch.float32).numpy(), 128, axis=0),
            128,
            axis=1,
        )[:150, :200]
    )
    np.testing.assert_allclose(np.array(out2d_bf16), exp2d_bf16, atol=1e-5, rtol=1e-5)

    # Rejection of float8_e5m2 tensor (only float8_e4m3fn is accepted)
    w_e5m2 = torch.randn(150, 200).to(torch.float8_e5m2)
    with self.assertRaisesRegex(ValueError, "Expected float8_e4m3fn weight"):
      glm5_ckpt.dequantize_fp8_block128(w_e5m2, s2d, target_dtype=jnp.float32)

    # Rejection of non-float8 tensor
    non_fp8_w = np.random.randn(150, 200).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Expected float8_e4m3fn weight"):
      glm5_ckpt.dequantize_fp8_block128(non_fp8_w, s2d, target_dtype=jnp.float32)

    # Rejection of float8_e4m3fnuz tensors
    with self.assertRaisesRegex(ValueError, "Expected float8_e4m3fn weight"):
      glm5_ckpt.dequantize_fp8_block128(
          torch.zeros((128, 128), dtype=torch.float8_e4m3fnuz),
          s2d,
          target_dtype=jnp.float32,
      )
    with self.assertRaisesRegex(ValueError, "Expected float8_e4m3fn weight"):
      glm5_ckpt.dequantize_fp8_block128(
          jnp.zeros((128, 128), dtype=jnp.float8_e4m3fnuz),
          s2d,
          target_dtype=jnp.float32,
      )

    # Rejection of dtype-less object raises TypeError
    with self.assertRaises(TypeError):
      glm5_ckpt.dequantize_fp8_block128(object(), s2d, target_dtype=jnp.float32)

  def test_dequantize_fp8_block128_dtype_owner_and_fp64_without_x64(self):
    """Verifies fp64 precision is preserved without x64 enabled and unsupported dtype raises."""
    w2d = torch.randn(150, 200).to(torch.float8_e4m3fn)
    s2d = np.random.randn(2, 2).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Unsupported target_dtype"):
      glm5_ckpt.dequantize_fp8_block128(w2d, s2d, target_dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "Unsupported target_dtype"):
      glm5_ckpt._to_target_np_array(w2d, np.int32)

    orig_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
      s64 = np.full((2, 2), 1.0 + 1e-12, dtype=np.float64)
      out64 = glm5_ckpt.dequantize_fp8_block128(w2d, torch.from_numpy(s64), target_dtype=np.float64)
      self.assertIsInstance(out64, np.ndarray)
      self.assertEqual(out64.dtype, np.float64)
      expected_64 = w2d.to(torch.float64).numpy() * np.repeat(np.repeat(s64, 128, axis=0), 128, axis=1)[:150, :200]
      np.testing.assert_array_equal(out64, expected_64)
      truncated_fp32 = (
          w2d.to(torch.float32).numpy()
          * np.repeat(
              np.repeat(s64.astype(np.float32), 128, axis=0),
              128,
              axis=1,
          )[:150, :200]
      ).astype(np.float64)
      self.assertFalse(np.array_equal(out64, truncated_fp32))
    finally:
      jax.config.update("jax_enable_x64", orig_x64)

    with tempfile.TemporaryDirectory() as st_dir:
      st_path = os.path.join(st_dir, "test.safetensors")
      safetensors.torch.save_file(
          {
              "w32": torch.randn(16, 32, dtype=torch.float32),
              "w64": torch.randn(16, 32, dtype=torch.float64),
          },
          st_path,
      )
      with safetensors.safe_open(st_path, framework="pt", device="cpu") as f:
        arr32 = glm5_ckpt._to_target_np_array(f.get_tensor("w32"), np.float32)
        arr64 = glm5_ckpt._to_target_np_array(f.get_tensor("w64"), np.float64)
      self.assertTrue(arr32.flags.owndata)
      self.assertTrue(arr64.flags.owndata)
      for view in (arr32, arr32.T, arr64, arr64.T):
        b = view
        while isinstance(b, np.ndarray):
          b = b.base
        self.assertIsNone(b)

  @parameterized.parameters(
      ("float32", False),
      ("float32", True),
      ("bfloat16", False),
      ("bfloat16", True),
      ("float64", False),
      ("float64", True),
  )
  def test_iter_conversion_specs_eval_shape_bidirectional(self, w_dt, scan_layers):
    """Verifies bidirectional set equality and shape/dtype parity with abstract model."""
    cm = _fp64_mode() if w_dt == "float64" else contextlib.nullcontext()
    with cm:
      cfg_dt, hf_c_dt = _make_cfg(
          num_hidden_layers=7,
          scan_layers=scan_layers,
          dtype=w_dt,
          weight_dtype=w_dt,
      )
      cfg_dt = dataclasses.replace(cfg_dt, num_nextn_predict_layers=1)
      hf_m = hf_model.Glm5NextTextModel(hf_c_dt)
      hf_head = torch.nn.Linear(hf_c_dt.hidden_size, hf_c_dt.vocab_size, bias=False)
      raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight})
      hub_state = _hf_module_state_to_hub(
          raw_state,
          num_experts=cfg_dt.n_routed_experts,
          moe_intermediate_size=cfg_dt.moe_intermediate_size,
      )
      hub_shapes = {k: tuple(v.shape) for k, v in hub_state.items()}
      mtp_indices = (glm5_config.MTP_LAYER_INDEX, 46) if glm5_config.MTP_LAYER_INDEX != 46 else (46,)
      l3_prefix = "model.language_model.layers.3."
      for idx in mtp_indices:
        mtp_prefix = f"model.language_model.layers.{idx}."
        for k, s in list(hub_shapes.items()):
          if k.startswith(l3_prefix):
            hub_shapes[f"{mtp_prefix}{k[len(l3_prefix) :]}"] = s
        for norm_name in (
            "enorm",
            "hnorm",
            "input_layernorm",
            "post_attention_layernorm",
        ):
          hub_shapes[f"{mtp_prefix}{norm_name}.weight"] = (cfg_dt.hidden_size,)
        hub_shapes[f"{mtp_prefix}shared_head.norm.weight"] = (cfg_dt.hidden_size,)
        hub_shapes[f"{mtp_prefix}eh_proj.weight"] = (
            cfg_dt.hidden_size,
            2 * cfg_dt.hidden_size,
        )

      abs_m = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(cfg_dt, rngs=nnx.Rngs(0), place_on_devices=False))
      abstract_leaves = jax.tree_util.tree_leaves_with_path(
          nnx.state(abs_m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)),
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      abstract_leaf_map = {_path_to_tuple(p): v for p, v in abstract_leaves}
      specs = glm5_ckpt.iter_conversion_specs(cfg_dt, include_mtp=True)
      self.assertEqual(
          set(s.target_path for s in specs),
          set(abstract_leaf_map.keys()),
      )

      def mock_get(key: str, dt: Any) -> jax.Array:
        return jnp.zeros(hub_shapes[key], dtype=dt)

      kda_suffixes = (
          ("self_attn", "A_log"),
          ("self_attn", "dt_bias"),
          ("self_attn", "q_conv1d", "weight"),
          ("self_attn", "k_conv1d", "weight"),
          ("self_attn", "v_conv1d", "weight"),
      )
      for spec in specs:
        model_var = glm5_ckpt.resolve_target_var(abs_m, spec.target_path)
        self.assertEqual(model_var.value.dtype, spec.target_dtype)
        if spec.stack_index is not None:
          self.assertEqual(model_var.value.shape[0], spec.num_super_blocks)
          unstacked_shape = model_var.value.shape[1:]
        else:
          unstacked_shape = model_var.value.shape

        transformed = glm5_ckpt._transform_for_spec(spec, mock_get)
        self.assertEqual(transformed.shape, unstacked_shape)
        self.assertEqual(transformed.dtype, spec.target_dtype)

        if any(spec.target_path[-len(sfx) :] == sfx for sfx in kda_suffixes):
          self.assertEqual(
              spec.target_dtype,
              jnp.float32,
              msg=f"KDA param {spec.target_path} not float32 in mode {w_dt}",
          )

  def test_assign_dtype_mismatch_raises_and_kda_fp32_preserved(self):
    """Verifies _assign raises ValueError on dtype mismatch and KDA fp32 is preserved."""
    # (a) _assign raises ValueError on dtype mismatch
    c_f32, hf_c_f32 = _make_cfg(
        num_hidden_layers=5,
        scan_layers=False,
        dtype="float32",
        weight_dtype="float32",
    )
    hf_m_f32 = hf_model.Glm5NextTextModel(hf_c_f32).to(torch.float32)
    hf_head_f32 = torch.nn.Linear(hf_c_f32.hidden_size, hf_c_f32.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m_f32, hf_head_f32)
    raw_state_f32 = dict(
        hf_m_f32.state_dict(),
        **{"lm_head.weight": hf_head_f32.weight.detach().clone()},
    )
    bias_val_f32 = torch.randn(c_f32.n_routed_experts, dtype=torch.float32)
    hf_m_f32.layers[3].mlp.gate.e_score_correction_bias.copy_(bias_val_f32)
    raw_state_f32["layers.3.mlp.gate.e_score_correction_bias"] = bias_val_f32
    hub_state_f32 = _hf_module_state_to_hub(
        raw_state_f32,
        num_experts=c_f32.n_routed_experts,
        moe_intermediate_size=c_f32.moe_intermediate_size,
    )
    jax_m_f32 = modeling_glm5_3_flash.Glm5Model(c_f32, rngs=nnx.Rngs(0), place_on_devices=False)
    real_specs = glm5_ckpt.iter_conversion_specs(c_f32, include_mtp=False)
    mocked_specs = [
        dataclasses.replace(s, target_dtype=jnp.bfloat16) if s.target_path == ("norm", "weight") else s
        for s in real_specs
    ]
    # Scoped mock.patch to verify ValueError when spec dtype mismatches model variable dtype
    with mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=mocked_specs):
      with self.assertRaisesRegex(
          ValueError,
          r"Dtype mismatch for model\.language_model\.norm\.weight",
      ):
        glm5_ckpt.load_hf_state_into_model(jax_m_f32, hub_state_f32, ignore_mtp_layer=True)

    # (b) Value-level test that A_log stays bit-exact fp32 under weight_dtype="bfloat16"
    c_bf16, hf_c_bf16 = _make_cfg(
        num_hidden_layers=5,
        scan_layers=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    hf_m_bf16 = hf_model.Glm5NextTextModel(hf_c_bf16).to(torch.bfloat16)
    hf_head_bf16 = torch.nn.Linear(hf_c_bf16.hidden_size, hf_c_bf16.vocab_size, bias=False).to(torch.bfloat16)
    _init_hf(hf_m_bf16, hf_head_bf16)
    raw_state_bf16 = dict(
        hf_m_bf16.state_dict(),
        **{"lm_head.weight": hf_head_bf16.weight.detach().clone()},
    )
    bias_val_bf16 = torch.randn(c_bf16.n_routed_experts, dtype=torch.bfloat16)
    hf_m_bf16.layers[3].mlp.gate.e_score_correction_bias.copy_(bias_val_bf16)
    raw_state_bf16["layers.3.mlp.gate.e_score_correction_bias"] = bias_val_bf16
    hub_state_bf16 = _hf_module_state_to_hub(
        raw_state_bf16,
        num_experts=c_bf16.n_routed_experts,
        moe_intermediate_size=c_bf16.moe_intermediate_size,
    )
    a_log_k = "model.language_model.layers.0.self_attn.A_log"
    expected_fp32_np = np.full(hub_state_bf16[a_log_k].shape, -1.2345679, dtype=np.float32)
    hub_state_bf16[a_log_k] = torch.from_numpy(expected_fp32_np)

    jax_m_bf16 = modeling_glm5_3_flash.Glm5Model(c_bf16, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m_bf16, hub_state_bf16, ignore_mtp_layer=True)

    actual_a_log = np.asarray(jax_m_bf16.prefix_layers[0].self_attn.A_log.value)
    self.assertEqual(jax_m_bf16.prefix_layers[0].self_attn.A_log.value.dtype, jnp.float32)
    self.assertTrue(np.array_equal(actual_a_log, expected_fp32_np))
    bf16_truncated = np.asarray(expected_fp32_np, dtype=jnp.bfloat16).astype(np.float32)
    self.assertFalse(np.array_equal(actual_a_log, bf16_truncated))

  def test_extract_mini_slice_keys(self):
    """Tests filtering of index weight_map for mini-slice and MTP layers."""
    weight_map = {f"model.language_model.layers.{l}.input_layernorm.weight": (f"s{l}.safetensors") for l in range(46)}
    for k in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ):
      weight_map[k] = "s0.safetensors"

    filtered = create_mini_slice.extract_mini_slice_keys(
        weight_map,
        num_hidden_layers=5,
        include_mtp_layer=glm5_config.MTP_LAYER_INDEX,
    )
    self.assertTrue(
        all(
            f"model.language_model.layers.{l}.input_layernorm.weight" in filtered
            for l in (0, 1, 2, 3, 4, glm5_config.MTP_LAYER_INDEX)
        )
    )
    self.assertNotIn("model.language_model.layers.5.input_layernorm.weight", filtered)
    for bad_layers, pat in (
        (0, "num_hidden_layers must be positive"),
        (-1, "num_hidden_layers must be positive"),
        (50, "zero matching keys"),
    ):
      with self.assertRaisesRegex(ValueError, pat):
        create_mini_slice.extract_mini_slice_keys(weight_map, num_hidden_layers=bad_layers, include_mtp_layer=None)
    with self.assertRaisesRegex(ValueError, "No lm_head keys found"):
      create_mini_slice.extract_mini_slice_keys(
          {k: v for k, v in weight_map.items() if not k.startswith("lm_head")},
          5,
          include_mtp_layer=None,
      )

  @parameterized.parameters((5, False), (13, True))
  def test_load_hf_state_into_model_parity(self, num_layers, scan_layers):
    """Verifies load_hf_state_into_model with FP8 scales, per-expert keys, and bias restoration."""
    c, hf_c = _make_cfg(num_hidden_layers=num_layers, scan_layers=scan_layers)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})

    bias_val = torch.randn(c.n_routed_experts, dtype=torch.float32)
    moe_l = 3
    hf_m.layers[moe_l].mlp.gate.e_score_correction_bias.copy_(bias_val)
    raw_state[f"layers.{moe_l}.mlp.gate.e_score_correction_bias"] = bias_val

    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )

    w0 = hub_state["model.language_model.layers.0.mlp.gate_proj.weight"]
    m, n = w0.shape
    s_inv = torch.full((math.ceil(m / 128), math.ceil(n / 128)), 2.0, dtype=torch.float32)
    exp_s = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
    w0_fp8 = (w0 / exp_s).to(torch.float8_e4m3fn)
    w0_dequant = w0_fp8.to(torch.float32) * exp_s
    hf_m.layers[0].mlp.gate_proj.weight.data.copy_(w0_dequant)
    raw_state["layers.0.mlp.gate_proj.weight"] = w0_dequant
    hub_state["model.language_model.layers.0.mlp.gate_proj.weight"] = w0_fp8
    hub_state["model.language_model.layers.0.mlp.gate_proj.weight_scale_inv"] = s_inv

    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)

    gate = jax_m.suffix_layers[0].mlp.gate if num_layers == 5 else jax_m.super_block.slot_0.mlp.gate
    restored_bias = gate.e_score_correction_bias.value if num_layers == 5 else gate.e_score_correction_bias.value[0]
    np.testing.assert_allclose(np.array(restored_bias), bias_val.numpy(), atol=1e-5, rtol=1e-5)

    seq_len = 16
    ids = torch.randint(1, 200, (1, seq_len), dtype=torch.long)
    cot = torch.randn(1, seq_len, hf_c.vocab_size, dtype=torch.float32)
    tokens = jnp.array(ids.numpy())

    hf_logits = _hf_forward(hf_m, hf_head, ids)
    (hf_logits * cot).sum().backward()

    out_aux = jax_m.forward_with_aux(tokens)
    self._assert_e3(out_aux.logits, hf_logits.detach().cpu().numpy(), "forward_logits")

    cot_jax = jnp.array(cot.detach().cpu().numpy())
    _, grads = nnx.value_and_grad(lambda m: jnp.sum(m(tokens) * cot_jax))(jax_m)

    hf_grad_state = {
        n: (p.grad.detach().clone() if hasattr(p, "grad") and p.grad is not None else torch.zeros_like(p))
        for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers())
    }
    hf_grad_state["lm_head.weight"] = (
        hf_head.weight.grad.detach().clone() if hf_head.weight.grad is not None else torch.zeros_like(hf_head.weight)
    )
    hub_grad_state = _hf_module_state_to_hub(
        hf_grad_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )

    jax_grad_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_grad_m, hub_grad_state, ignore_mtp_layer=True)

    leaves_jax = jax.tree.leaves(nnx.state(grads, nnx.Param))
    leaves_hf = jax.tree.leaves(nnx.state(jax_grad_m, nnx.Param))
    self.assertEqual(len(leaves_jax), len(leaves_hf))
    for gj, gh in zip(leaves_jax, leaves_hf, strict=True):
      self._assert_e3(gj, gh, "param_grad")
    self.assertEmpty(dict(nnx.state(grads, modeling_glm5_3_flash.Glm5RouterBias)))

  def test_fp64_ckpt_load_parity(self):
    """Verifies fp64 checkpoint loading forward logits and gradient parity at <= 1e-10."""
    c, hf_c = _make_cfg(
        num_hidden_layers=5,
        scan_layers=True,
        dtype="float64",
        weight_dtype="float64",
    )
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float64)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float64)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})

    bias_val = torch.randn(c.n_routed_experts, dtype=torch.float64) * 0.02
    moe_l = 3
    hf_m.layers[moe_l].mlp.gate.e_score_correction_bias.copy_(bias_val)
    raw_state[f"layers.{moe_l}.mlp.gate.e_score_correction_bias"] = bias_val

    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )

    seq_len = 16
    ids = torch.randint(1, 200, (1, seq_len), dtype=torch.long)
    cot = torch.randn(1, seq_len, hf_c.vocab_size, dtype=torch.float64)

    with _fp64_mode():
      tokens = jnp.array(ids.numpy())
      cot_jax = jnp.array(cot.detach().cpu().numpy(), dtype=jnp.float64)
      jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)

      hf_logits = _hf_forward(hf_m, hf_head, ids)
      (hf_logits * cot).sum().backward()

      out_aux = jax_m.forward_with_aux(tokens)
      act_l, exp_l = (
          np.asarray(out_aux.logits, dtype=np.float64),
          hf_logits.detach().cpu().numpy(),
      )
      l_frob = float(np.linalg.norm(act_l - exp_l)) / float(np.linalg.norm(exp_l))
      self.assertLessEqual(l_frob, 1e-10, f"fp64 logits frob_rel: {l_frob}")

      _, grads = nnx.value_and_grad(lambda m: jnp.sum(m(tokens) * cot_jax))(jax_m)

      hf_grad_state = {
          n: (p.grad.detach().clone() if hasattr(p, "grad") and p.grad is not None else torch.zeros_like(p))
          for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers())
      }
      hf_grad_state["lm_head.weight"] = (
          hf_head.weight.grad.detach().clone() if hf_head.weight.grad is not None else torch.zeros_like(hf_head.weight)
      )
      hub_grad_state = _hf_module_state_to_hub(
          hf_grad_state,
          num_experts=c.n_routed_experts,
          moe_intermediate_size=c.moe_intermediate_size,
      )

      jax_grad_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.load_hf_state_into_model(jax_grad_m, hub_grad_state, ignore_mtp_layer=True)

      leaves_jax = jax.tree.leaves(nnx.state(grads, nnx.Param))
      leaves_hf = jax.tree.leaves(nnx.state(jax_grad_m, nnx.Param))
      self.assertEqual(len(leaves_jax), len(leaves_hf))
      for gj, gh in zip(leaves_jax, leaves_hf, strict=True):
        aj, ah = np.asarray(gj, dtype=np.float64), np.asarray(gh, dtype=np.float64)
        norm_h = float(np.linalg.norm(ah))
        if norm_h == 0.0:
          self.assertEqual(float(np.linalg.norm(aj)), 0.0)
        else:
          self.assertGreater(float(np.linalg.norm(aj)), 0.0)
          diff = float(np.linalg.norm(aj - ah)) / norm_h
          self.assertLessEqual(diff, 1e-10)

  def test_strict_key_coverage_and_unknown_keys(self):
    """Verifies strict key validation rejects missing/unknown keys and respects ignore_mtp_layer."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(
        hf_m.state_dict(),
        **{"lm_head.weight": torch.randn(hf_c.vocab_size, hf_c.hidden_size)},
    )
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)

    # Missing required key raises
    missing_key_state = {k: v for k, v in hub_state.items() if k != "model.language_model.norm.weight"}
    with self.assertRaisesRegex(ValueError, "Required parameter missing"):
      glm5_ckpt.load_hf_state_into_model(jax_m, missing_key_state, ignore_mtp_layer=True)

    # Unknown key raises
    with self.assertRaisesRegex(ValueError, "Unknown HF key prefix"):
      glm5_ckpt.load_hf_state_into_model(jax_m, dict(hub_state, bogus=torch.zeros(1)), ignore_mtp_layer=True)

    # Non-hub prefix (e.g. without model.language_model.) raises ValueError
    non_hub_state = dict(raw_state)
    with self.assertRaisesRegex(ValueError, "Unknown HF key prefix"):
      glm5_ckpt.load_hf_state_into_model(jax_m, non_hub_state, ignore_mtp_layer=True)

    # Alternate prefix (e.g. model.layers... instead of model.language_model.layers...) raises ValueError
    old_prefix_state = {
        (f"model.{k[len('model.language_model.'):]}" if k.startswith("model.language_model.") else k): v
        for k, v in hub_state.items()
    }
    with self.assertRaisesRegex(ValueError, "Unknown HF key prefix"):
      glm5_ckpt.load_hf_state_into_model(jax_m, old_prefix_state, ignore_mtp_layer=True)

    # MTP layer handling
    mtp_k = f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}.input_layernorm.weight"
    mtp_state = dict(hub_state, **{mtp_k: torch.zeros(c.hidden_size)})
    with self.assertRaisesRegex(
        ValueError,
        "ignore_mtp_layer=False requires model.mtp to be initialized",
    ):
      glm5_ckpt.load_hf_state_into_model(jax_m, mtp_state, ignore_mtp_layer=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, mtp_state, ignore_mtp_layer=True)

    # Bogus unmapped key
    bogus_state = dict(
        hub_state,
        **{"model.language_model.layers.0.bogus_weight": torch.zeros(c.hidden_size)},
    )
    with self.assertRaisesRegex(ValueError, "Unmapped key in hf_tensors"):
      glm5_ckpt.load_hf_state_into_model(jax_m, bogus_state, ignore_mtp_layer=True)

  def test_fp8_scale_mismatch_validation(self):
    """Verifies FP8 tensor without scale_inv and non-FP8 tensor with scale_inv raise ValueError."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(
        hf_m.state_dict(),
        **{"lm_head.weight": torch.randn(hf_c.vocab_size, hf_c.hidden_size)},
    )
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)

    # 1. FP8 tensor without *_scale_inv raises ValueError
    fp8_no_scale = dict(hub_state)
    fp8_no_scale["model.language_model.embed_tokens.weight"] = hub_state["model.language_model.embed_tokens.weight"].to(
        torch.float8_e4m3fn
    )
    with self.assertRaisesRegex(ValueError, "missing scale inverse"):
      glm5_ckpt.load_hf_state_into_model(jax_m, fp8_no_scale, ignore_mtp_layer=True)

    # 2. Non-FP8 tensor with *_scale_inv raises ValueError
    non_fp8_with_scale = dict(hub_state)
    non_fp8_with_scale["model.language_model.embed_tokens.weight_scale_inv"] = torch.ones(
        (math.ceil(c.vocab_size / 128), math.ceil(c.hidden_size / 128)),
        dtype=torch.float32,
    )
    with self.assertRaisesRegex(ValueError, "not float8"):
      glm5_ckpt.load_hf_state_into_model(jax_m, non_fp8_with_scale, ignore_mtp_layer=True)

    # 3. Positive liveness check: valid FP8 tensor with matching scale_inv succeeds
    valid_fp8_state = dict(hub_state)
    w = hub_state["model.language_model.embed_tokens.weight"]
    m, n = w.shape
    valid_fp8_state["model.language_model.embed_tokens.weight"] = w.to(torch.float8_e4m3fn)
    valid_fp8_state["model.language_model.embed_tokens.weight_scale_inv"] = torch.ones(
        (math.ceil(m / 128), math.ceil(n / 128)), dtype=torch.float32
    )
    glm5_ckpt.load_hf_state_into_model(jax_m, valid_fp8_state, ignore_mtp_layer=True)

  def test_convert_hf_checkpoint_to_orbax_roundtrip(self):
    """Verifies convert_hf_checkpoint_to_orbax and restoration parity."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )

    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      orbax_dir = os.path.join(tmp_dir, "orbax_ckpt")
      safetensors.torch.save_file(hub_state, f"{hf_dir}/model.safetensors")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          orbax_dir,
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )
      restored_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(restored_m, orbax_dir, mesh=None)

      ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
      hf_logits = _hf_forward(hf_m, hf_head, ids).detach().cpu().numpy()
      np.testing.assert_allclose(
          hf_logits,
          np.array(restored_m(jnp.array(ids.numpy()))),
          atol=1e-4,
          rtol=1e-4,
      )

  def test_write_mini_slice_checkpoint(self):
    """Verifies write_mini_slice_checkpoint with multi-shard index, MTP layer, and Orbax load."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    hub_state[f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}.input_layernorm.weight"] = torch.randn(
        c.hidden_size
    )
    hub_state["model.language_model.layers.10.input_layernorm.weight"] = torch.randn(c.hidden_size)

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst, tempfile.TemporaryDirectory() as tmp:
      obx = os.path.join(tmp, "orbax_ckpt")
      keys = list(hub_state)
      s1_keys, s2_keys = set(keys[::2]), set(keys[1::2])
      safetensors.torch.save_file(
          {k: hub_state[k] for k in s1_keys},
          os.path.join(src, "model-00001-of-00002.safetensors"),
      )
      safetensors.torch.save_file(
          {k: hub_state[k] for k in s2_keys},
          os.path.join(src, "model-00002-of-00002.safetensors"),
      )

      weight_map = {k: "model-00001-of-00002.safetensors" for k in s1_keys}
      weight_map.update({k: "model-00002-of-00002.safetensors" for k in s2_keys})
      with open(
          os.path.join(src, "model.safetensors.index.json"),
          "w",
          encoding="utf-8",
      ) as f:
        json.dump({"weight_map": weight_map}, f)
      # Also populate text_config subdict and tokenizer files in source directory
      cfg_dict = {
          "num_hidden_layers": 46,
          "text_config": {
              "num_hidden_layers": 46,
              "layer_types": ["mla"] * 46,
              "mlp_layer_types": ["sparse"] * 46,
              "indexer_types": ["kpool"] * 46,
              "linear_attn_config": {
                  "kda_layers": [0, 1, 2, 4, 5, 6],
                  "full_attn_layers": [3, 7],
              },
          },
      }
      with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f)
      for tok_f in (
          "tokenizer.json",
          "tokenizer_config.json",
          "special_tokens_map.json",
          "generation_config.json",
      ):
        with open(os.path.join(src, tok_f), "w", encoding="utf-8") as f:
          f.write(f'{{"file": "{tok_f}"}}')

      sliced_map = create_mini_slice.write_mini_slice_checkpoint(
          src,
          dst,
          num_hidden_layers=5,
          include_mtp_layer=glm5_config.MTP_LAYER_INDEX,
      )
      self.assertTrue(os.path.exists(os.path.join(dst, "model.safetensors")))
      self.assertTrue(os.path.exists(os.path.join(dst, "model.safetensors.index.json")))
      with open(os.path.join(dst, "config.json"), "r", encoding="utf-8") as f:
        dst_cfg = json.load(f)
        self.assertEqual(dst_cfg["num_hidden_layers"], 5)
        self.assertEqual(dst_cfg["text_config"]["num_hidden_layers"], 5)
        self.assertEqual(len(dst_cfg["text_config"]["layer_types"]), 5)
        self.assertEqual(len(dst_cfg["text_config"]["mlp_layer_types"]), 5)
        self.assertEqual(len(dst_cfg["text_config"]["indexer_types"]), 5)
        self.assertEqual(
            dst_cfg["text_config"]["linear_attn_config"]["kda_layers"],
            [0, 1, 2, 4],
        )
        self.assertEqual(
            dst_cfg["text_config"]["linear_attn_config"]["full_attn_layers"],
            [3],
        )
      for tok_f in (
          "tokenizer.json",
          "tokenizer_config.json",
          "special_tokens_map.json",
          "generation_config.json",
      ):
        self.assertTrue(os.path.exists(os.path.join(dst, tok_f)))

      self.assertIn(
          f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}.input_layernorm.weight",
          sliced_map,
      )
      self.assertNotIn("model.language_model.layers.10.input_layernorm.weight", sliced_map)

      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          dst,
          obx,
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )
      restored_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(restored_m, obx, mesh=None)

      ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
      hf_logits = _hf_forward(hf_m, hf_head, ids).detach().cpu().numpy()
      np.testing.assert_allclose(
          hf_logits,
          np.array(restored_m(jnp.array(ids.numpy()))),
          atol=1e-4,
          rtol=1e-4,
      )

  def test_unpopulated_parameter_slot_raises(self):
    """Verifies that an unpopulated model parameter slot raises ValueError."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    # Positive liveness match: normal full load succeeds cleanly
    glm5_ckpt.load_hf_state_into_model(jax_m, dict(hub_state), ignore_mtp_layer=True)

    # Negative check: mock iter_conversion_specs to omit norm.weight
    all_specs = glm5_ckpt.iter_conversion_specs(c, include_mtp=False)
    specs_without_norm = [s for s in all_specs if s.target_path != ("norm", "weight")]
    bad_hub_state = dict(hub_state)
    del bad_hub_state["model.language_model.norm.weight"]
    with mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=specs_without_norm):
      with self.assertRaisesRegex(ValueError, "Unpopulated model parameter slot"):
        glm5_ckpt.load_hf_state_into_model(jax_m, bad_hub_state, ignore_mtp_layer=True)

  def test_assign_shape_mismatch_raises(self):
    """Verifies strict shape validation in _assign raises ValueError on shape mismatch."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    # Replace norm.weight with bad shape
    bad_hub_state = dict(hub_state)
    bad_hub_state["model.language_model.norm.weight"] = torch.randn(c.hidden_size + 8)
    with self.assertRaisesRegex(ValueError, "Shape mismatch for model.language_model.norm.weight"):
      glm5_ckpt.load_hf_state_into_model(jax_m, bad_hub_state, ignore_mtp_layer=True)

  def test_visual_prefix_keys_ignored(self):
    """Verifies multimodal visual keys (model.visual.* and visual.*) are ignored cleanly."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    hub_state["model.visual.patch_embed.proj.weight"] = torch.randn(64, 3, 14, 14)
    hub_state["visual.blocks.0.attn.qkv.weight"] = torch.randn(64, 64)
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    # Should not raise "Unknown HF key prefix" or "Unmapped key in hf_tensors"
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)

  def test_convert_hf_checkpoint_slice_to_config_layers(self):
    """Verifies slice_to_config_layers opt-in flag in convert_hf_checkpoint_to_orbax."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    # Add an unmapped layer 10 key
    hub_state["model.language_model.layers.10.input_layernorm.weight"] = torch.randn(c.hidden_size)

    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      safetensors.torch.save_file(hub_state, os.path.join(hf_dir, "model.safetensors"))
      weight_map = {k: "model.safetensors" for k in hub_state}
      with open(
          os.path.join(hf_dir, "model.safetensors.index.json"),
          "w",
          encoding="utf-8",
      ) as f:
        json.dump({"weight_map": weight_map}, f)

      orbax_dir1 = os.path.join(tmp_dir, "orbax1")
      # With slice_to_config_layers=False: raises ValueError for unmapped key
      with self.assertRaisesRegex(ValueError, "Unmapped key in hf_tensors"):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            orbax_dir1,
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      # With slice_to_config_layers=True: extracts mini slice and converts cleanly
      orbax_dir2 = os.path.join(tmp_dir, "orbax2")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          orbax_dir2,
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=True,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )
      self.assertTrue(os.path.exists(orbax_dir2))

  def test_convert_hf_checkpoint_to_orbax_enforces_mini_slice_scope(self):
    """Verifies _convert_hf_checkpoint_to_orbax_in_memory rejects checkpoints exceeding mini-slice bounds."""
    c, _ = _make_cfg(
        num_hidden_layers=glm5_ckpt.MAX_IN_MEMORY_CONVERSION_LAYERS + 1,
        scan_layers=False,
    )
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      orbax_dir = os.path.join(tmp_dir, "orbax_ckpt")
      with self.assertRaisesRegex(
          ValueError,
          "scoped to mini-slice checkpoints with at most",
      ):
        glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
            hf_dir,
            orbax_dir,
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
        )

    # Test MAX_IN_MEMORY_CONVERSION_PARAMS header check
    c_ok, hf_c_ok = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c_ok).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c_ok.hidden_size, hf_c_ok.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(
        hf_m.state_dict(),
        **{"lm_head.weight": hf_head.weight.detach().clone()},
    )
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c_ok.n_routed_experts,
        moe_intermediate_size=c_ok.moe_intermediate_size,
    )
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      safetensors.torch.save_file(hub_state, os.path.join(hf_dir, "model.safetensors"))
      orbax_dir = os.path.join(tmp_dir, "orbax_ckpt2")
      with mock.patch.object(glm5_ckpt, "MAX_IN_MEMORY_CONVERSION_PARAMS", 10):
        with self.assertRaisesRegex(
            ValueError,
            "scoped to mini-slice checkpoints with at most 10 parameters",
        ):
          glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
              hf_dir,
              orbax_dir,
              c_ok,
              ignore_mtp_layer=True,
              slice_to_config_layers=False,
          )

  def test_iter_conversion_specs_validation(self):
    """Verifies iter_conversion_specs requires keyword-only include_mtp and validates MTP layers."""
    c, _ = _make_cfg(num_hidden_layers=5, scan_layers=False)
    # 1. Missing include_mtp raises TypeError
    with self.assertRaises(TypeError):
      glm5_ckpt.iter_conversion_specs(c)  # pylint: disable=missing-kwoa

    # 2. include_mtp=True with num_nextn_predict_layers=0 raises ValueError
    c_no_mtp = dataclasses.replace(c, num_nextn_predict_layers=0)
    with self.assertRaisesRegex(
        ValueError,
        "include_mtp=True requires config.num_nextn_predict_layers > 0",
    ):
      glm5_ckpt.iter_conversion_specs(c_no_mtp, include_mtp=True)

    # 3. include_mtp=False succeeds
    specs = glm5_ckpt.iter_conversion_specs(c, include_mtp=False)
    self.assertNotEmpty(specs)
    self.assertFalse(any("mtp" in s.target_path for s in specs))
    for s in specs:
      for hk in s.hf_keys:
        self.assertNotIn("..", hk)

    # 4. moe_gate_up kind present for MoE layers
    moe_specs = [s for s in specs if s.kind == "moe_gate_up"]
    self.assertNotEmpty(moe_specs)

    # 5. include_mtp=True emits MTP specs (T37)
    c_mtp = dataclasses.replace(c, num_nextn_predict_layers=1)
    specs_mtp = glm5_ckpt.iter_conversion_specs(c_mtp, include_mtp=True)
    self.assertTrue(any("mtp" in s.target_path for s in specs_mtp))

  def test_iter_conversion_specs_use_indexer_false(self):
    """Verifies iter_conversion_specs with use_indexer=False emits zero indexer specs (T35)."""
    c, _ = _make_cfg(num_hidden_layers=5, scan_layers=False)
    c_no_idx = dataclasses.replace(c, use_indexer=False)
    specs = glm5_ckpt.iter_conversion_specs(c_no_idx, include_mtp=False)
    self.assertFalse(any("indexer" in s.target_path for s in specs))

  def test_load_hf_state_bf16_preserves_fp32_params(self):
    """Verifies weight_dtype='bfloat16' retains float32 for sensitive parameters (T33)."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False, weight_dtype="bfloat16")
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)

    # 1. KDA float32 parameters
    l0 = jax_m.prefix_layers[0]
    self.assertEqual(l0.self_attn.A_log.dtype, jnp.float32)
    self.assertEqual(l0.self_attn.dt_bias.dtype, jnp.float32)
    self.assertEqual(l0.self_attn.q_conv1d.weight.dtype, jnp.float32)
    self.assertEqual(l0.self_attn.k_conv1d.weight.dtype, jnp.float32)
    self.assertEqual(l0.self_attn.v_conv1d.weight.dtype, jnp.float32)

    # 2. MoE router bias parameter
    l3 = jax_m.suffix_layers[0]
    self.assertEqual(l3.mlp.gate.e_score_correction_bias.dtype, jnp.float32)

  def test_create_mini_slice_unrecognized_key_and_mtp_filtering(self):
    """Verifies extract_mini_slice_keys rejects unknown keys and write trims MTP/layers (T36)."""
    # 1. extract_mini_slice_keys rejects unrecognized key
    bad_map = {
        "model.language_model.embed_tokens.weight": "shard-1",
        "model.language_model.norm.weight": "shard-1",
        "lm_head.weight": "shard-1",
        "model.language_model.layers.0.weight": "shard-1",
        "model.language_model.some.unknown.parameter": "shard-1",
    }
    with self.assertRaisesRegex(ValueError, "Unrecognized HF key in weight_map"):
      create_mini_slice.extract_mini_slice_keys(bad_map, num_hidden_layers=1, include_mtp_layer=None)

    # 2. write_mini_slice_checkpoint trims config.json
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
      cfg_dict = {
          "num_hidden_layers": 45,
          "num_nextn_predict_layers": 1,
          "kda_layers": [0, 1, 2, 4, 5, 6, 7],
          "full_attn_layers": [3, 8],
          "text_config": {
              "num_hidden_layers": 45,
              "num_nextn_predict_layers": 1,
              "kda_layers": [0, 1, 2, 4, 5, 6, 7],
              "full_attn_layers": [3, 8],
          },
      }
      with open(os.path.join(src_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f)
      good_map = {
          "model.language_model.embed_tokens.weight": "shard1.safetensors",
          "model.language_model.norm.weight": "shard1.safetensors",
          "lm_head.weight": "shard1.safetensors",
          "model.language_model.layers.0.weight": "shard1.safetensors",
          "model.language_model.layers.1.weight": "shard1.safetensors",
      }
      with open(
          os.path.join(src_dir, "model.safetensors.index.json"),
          "w",
          encoding="utf-8",
      ) as f:
        json.dump({"weight_map": good_map}, f)
      dummy_tensors = {k: torch.zeros((2, 2)) for k in good_map}
      safetensors.torch.save_file(dummy_tensors, os.path.join(src_dir, "shard1.safetensors"))

      create_mini_slice.write_mini_slice_checkpoint(src_dir, dst_dir, num_hidden_layers=2, include_mtp_layer=None)
      with open(os.path.join(dst_dir, "config.json"), "r", encoding="utf-8") as f:
        out_cfg = json.load(f)
      self.assertEqual(out_cfg["num_hidden_layers"], 2)
      self.assertEqual(out_cfg["num_nextn_predict_layers"], 0)
      self.assertEqual(out_cfg["kda_layers"], [0, 1])
      self.assertEqual(out_cfg["full_attn_layers"], [])
      self.assertEqual(out_cfg["text_config"]["num_hidden_layers"], 2)
      self.assertEqual(out_cfg["text_config"]["num_nextn_predict_layers"], 0)
      self.assertEqual(out_cfg["text_config"]["kda_layers"], [0, 1])
      self.assertEqual(out_cfg["text_config"]["full_attn_layers"], [])

  def test_lm_head_scale_inv_not_aliased(self):
    """Verifies lm_head.weight_scale_inv is not aliased to lm_head.weight."""
    c, hf_c = _make_cfg(num_hidden_layers=5, scan_layers=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _init_hf(hf_m, hf_head)
    raw_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})
    hub_state = _hf_module_state_to_hub(
        raw_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    w_lm = hub_state["lm_head.weight"]
    m, n = w_lm.shape
    s_inv = torch.full((math.ceil(m / 128), math.ceil(n / 128)), 1.5, dtype=torch.float32)
    exp_s = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
    w_fp8 = (w_lm / exp_s).to(torch.float8_e4m3fn)
    hub_state["lm_head.weight"] = w_fp8
    hub_state["lm_head.weight_scale_inv"] = s_inv

    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)
    expected_dequant = (w_fp8.to(torch.float32) * exp_s).numpy().T
    np.testing.assert_allclose(
        np.array(jax_m.lm_head.kernel.value),
        expected_dequant,
        atol=1e-5,
        rtol=1e-5,
    )

  def test_restore_orbax_checkpoint_dtype_mismatch_raises(self):
    """Verifies restore_orbax_checkpoint raises ValueError when dtype mismatches or changes."""
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      obx_dir = os.path.join(tmp_dir, "obx")
      c = _make_cfg(5, scan_layers=False, dtype="float64", weight_dtype="float64")[0]
      with _fp64_mode():
        _write_sharded_ckpt(hf_dir, c, num_shards=2)
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            obx_dir,
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )
        m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      with self.assertRaisesRegex(ValueError, "Restored leaf dtype changed during conversion"):
        glm5_ckpt.restore_orbax_checkpoint(m, obx_dir, mesh=None)
      m_fp32 = modeling_glm5_3_flash.Glm5Model(
          _make_cfg(5, scan_layers=False)[0],
          rngs=nnx.Rngs(0),
          place_on_devices=False,
      )
      with _fp64_mode():
        with self.assertRaisesRegex(ValueError, "does not match model variable dtype"):
          glm5_ckpt.restore_orbax_checkpoint(m_fp32, obx_dir, mesh=None)

  def test_streaming_bit_identical_to_oracle_and_hf_source(self):
    """Verifies streaming bit-identity to source HF tensors against the in-memory reference oracle."""
    with _fp64_mode(), tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      c = dataclasses.replace(
          _make_cfg(7, scan_layers=False, dtype="float64", weight_dtype="float64")[0],
          num_nextn_predict_layers=1,
      )
      hub_state, exp_s = _write_sharded_ckpt(hf_dir, c, num_shards=3)
      s_dir, i_dir = os.path.join(tmp_dir, "s"), os.path.join(tmp_dir, "i")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          s_dir,
          c,
          ignore_mtp_layer=False,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )
      # Mock MAX_IN_MEMORY_CONVERSION_LAYERS to bypass layer limit for oracle test.
      with mock.patch.object(glm5_ckpt, "MAX_IN_MEMORY_CONVERSION_LAYERS", 8):
        glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
            hf_dir,
            i_dir,
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
        )
      for scan_mode in (False, True):
        cfg = dataclasses.replace(c, scan_layers=scan_mode)
        ms = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
        mi = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
        glm5_ckpt.restore_orbax_checkpoint(ms, s_dir, mesh=None)
        glm5_ckpt.restore_orbax_checkpoint(mi, i_dir, mesh=None)
        for sl, il in zip(jax.tree.leaves(nnx.state(ms)), jax.tree.leaves(nnx.state(mi))):
          self.assertEqual(sl.dtype, il.dtype)
          self.assertTrue(jnp.array_equal(sl, il))
      self.assertEqual(ms.embed_tokens.embedding.value[0, 0], 1.0 + 1e-12)
      np.testing.assert_array_equal(
          ms.embed_tokens.embedding.value,
          hub_state["model.language_model.embed_tokens.weight"].numpy(),
      )
      np.testing.assert_array_equal(
          ms.super_block.slot_0.mlp.experts.gate_up_proj.value[0, 0, :, 0, :],
          hub_state["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"].numpy().T,
      )
      w0 = hub_state["model.language_model.layers.0.mlp.gate_proj.weight"].to(torch.float64)
      np.testing.assert_array_equal(
          ms.prefix_layers[0].mlp.gate_proj.kernel.value,
          (w0 * exp_s.to(torch.float64)).numpy().T,
      )

  def test_streaming_shard_handle_lru_bound(self):
    """Verifies at most _MAX_CONCURRENT_OPEN_SHARDS handles are concurrently open."""
    with _fp64_mode(), tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      c = dataclasses.replace(
          _make_cfg(7, scan_layers=False, dtype="float64", weight_dtype="float64")[0],
          num_nextn_predict_layers=1,
      )
      _write_sharded_ckpt(hf_dir, c, num_shards=3)
      active, max_open = [0], [0]

      def _on_enter():
        active[0] += 1
        max_open[0] = max(max_open[0], active[0])

      _run_with_tracker(
          hf_dir,
          os.path.join(tmp_dir, "obx"),
          c,
          on_get=lambda *a: None,
          on_enter=_on_enter,
          on_exit=lambda: active.__setitem__(0, active[0] - 1),
      )
      self.assertLessEqual(max_open[0], glm5_ckpt._MAX_CONCURRENT_OPEN_SHARDS)
      self.assertEqual(active[0], 0)

  def test_streaming_single_load_per_key(self):
    """Verifies each tensor key in the checkpoint is loaded from shard exactly once."""
    with _fp64_mode(), tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      c = dataclasses.replace(
          _make_cfg(7, scan_layers=False, dtype="float64", weight_dtype="float64")[0],
          num_nextn_predict_layers=1,
      )
      _write_sharded_ckpt(hf_dir, c, num_shards=3)
      key_load_counts: dict[str, int] = collections.defaultdict(int)
      _run_with_tracker(
          hf_dir,
          os.path.join(tmp_dir, "obx"),
          c,
          on_get=lambda _, k: key_load_counts.__setitem__(k, key_load_counts[k] + 1),
          on_enter=lambda: None,
          on_exit=lambda: None,
      )
      self.assertNotEmpty(key_load_counts)
      self.assertEqual(max(key_load_counts.values()), 1)

  @parameterized.named_parameters(
      ("float64_scanned", "float64", torch.float64 if HAS_TORCH else None, 2048, True),
      ("bfloat16_unscanned", "bfloat16", torch.bfloat16 if HAS_TORCH else None, 2048, False),
  )
  def test_streaming_peak_memory_bound_and_positive_control(
      self,
      target_dtype: str,
      shard_dtype: torch.dtype,
      vocab_size: int,
      scan_layers: bool,
  ):
    """Verifies bounded peak host memory, open-shard accounting, and positive control.

    The expected streaming peak memory bound is defined as:
        expected_bound = (two_shards_bytes + single_unit_bytes + slack_bytes) <
        pure_dict_bytes

    Terms justification:
      - two_shards_bytes: Maximum byte size of _MAX_CONCURRENT_OPEN_SHARDS = 2
        concurrently open safetensors shard files. The file sizes of open shards
        are accounted (not measured resident RSS, since OS mmap pages may not be
        resident in physical RAM), and strictly enforced by asserting
        active_open_shards[0] <= glm5_ckpt._MAX_CONCURRENT_OPEN_SHARDS (2) at
        every sample point.
      - single_unit_bytes: Transient working memory headroom for converting,
        transposing, or dequantizing a single unstacked leaf or single MoE
        expert chunk,
        multiplied by 2 to account for transient source + converted slice
        buffers during
        a single step.
      - slack_bytes: 15% allowance (int(0.15 * pure_dict_bytes)) covering Python
        dict/tuple/spec structures, object headers, and tracemalloc bookkeeping
        overhead.

    The test executes under both float64 and bfloat16 configurations and both
    scanned and unscanned layouts, asserting peak_streaming <= expected_bound <
    pure_dict_bytes,
    and verifies that retained shards trigger the positive control.
    """

    def _write_local_sharded_ckpt(
        dest_hf_dir: str,
        model_cfg: glm5_config.Glm5Config,
        shard_dt: torch.dtype,
        num_shards: int,
    ) -> None:
      abs_m = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(model_cfg, rngs=nnx.Rngs(0), place_on_devices=False))
      hub: dict[str, torch.Tensor] = {}
      for s in glm5_ckpt.iter_conversion_specs(model_cfg, include_mtp=model_cfg.num_nextn_predict_layers > 0):
        sh = glm5_ckpt.resolve_target_var(abs_m, s.target_path).value.shape
        sh = sh[1:] if s.stack_index is not None else sh
        dt = torch.float32 if any(s.hf_keys[0].endswith(x) for x in ("A_log", "dt_bias", "conv1d.weight")) else shard_dt
        if s.kind == "direct":
          k_sh = sh
        elif s.kind == "transpose_2d":
          k_sh = (sh[1], sh[0])
        elif s.kind == "moe_gate_up":
          k_sh = (model_cfg.moe_intermediate_size, model_cfg.hidden_size)
        elif s.kind == "moe_down":
          k_sh = (model_cfg.hidden_size, model_cfg.moe_intermediate_size)
        else:
          raise ValueError(f"Unknown spec kind: {s.kind!r}")
        for k in s.hf_keys:
          hub[k] = torch.randn(k_sh, dtype=dt)
      if model_cfg.weight_dtype == "float64":
        hub["model.language_model.embed_tokens.weight"][0, 0] = 1.0 + 1e-12
      fp8_k = "model.language_model.layers.0.mlp.gate_proj.weight"
      sc_k = f"{fp8_k}_scale_inv"
      w_src = hub[fp8_k].to(torch.float32)
      m, n = w_src.shape
      sb_m, sb_n = math.ceil(m / 128), math.ceil(n / 128)
      s_inv = torch.linspace(0.5, 2.0, steps=sb_m * sb_n, dtype=torch.float32).reshape(sb_m, sb_n)
      exp_s = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
      hub[fp8_k], hub[sc_k] = (w_src / exp_s).to(torch.float8_e4m3fn), s_inv
      wmap = {
          fp8_k: "m-1.safetensors",
          sc_k: f"m-{min(2, num_shards)}.safetensors",
      }
      for idx, k in enumerate(sorted(k for k in hub if k not in (fp8_k, sc_k))):
        wmap[k] = f"m-{idx % num_shards + 1}.safetensors"
      for sf in sorted(set(wmap.values())):
        safetensors.torch.save_file(
            {k: hub[k] for k, v in wmap.items() if v == sf},
            os.path.join(dest_hf_dir, sf),
        )
      with open(
          os.path.join(dest_hf_dir, "model.safetensors.index.json"),
          "w",
          encoding="utf-8",
      ) as f:
        json.dump({"weight_map": wmap}, f)

    with (
        _fp64_mode() if target_dtype == "float64" else contextlib.nullcontext()
    ), tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      c = dataclasses.replace(
          _make_cfg(
              11,
              scan_layers=scan_layers,
              dtype=target_dtype,
              weight_dtype=target_dtype,
          )[0],
          hidden_size=512,
          intermediate_size=1024,
          moe_intermediate_size=128,
          n_routed_experts=24,
          vocab_size=vocab_size,
          num_nextn_predict_layers=1,
      )
      _write_local_sharded_ckpt(hf_dir, c, shard_dtype, 12)
      abs_m = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False))
      pure_leaves = jax.tree.leaves(nnx.state(abs_m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      pure_dict_bytes = sum(int(np.prod(v.shape)) * np.dtype(v.dtype).itemsize for v in pure_leaves)
      self.assertGreaterEqual(pure_dict_bytes, 32 * 1024 * 1024)
      single_unit_bytes = 16_777_216 if target_dtype == "bfloat16" else 26_214_400

      shards = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
      shard_sizes = sorted(os.path.getsize(os.path.join(hf_dir, f)) for f in shards)
      total_shard_bytes = sum(shard_sizes)
      two_shards_bytes = sum(shard_sizes[-2:])
      slack_bytes = int(0.15 * pure_dict_bytes)
      expected_bound = two_shards_bytes + single_unit_bytes + slack_bytes
      self.assertLess(expected_bound, pure_dict_bytes)
      self.assertLess(
          two_shards_bytes + single_unit_bytes + slack_bytes,
          total_shard_bytes,
          "fixture degenerate: positive control cannot exceed the bound",
      )
      orig_open = safetensors.safe_open

      def _run(out_name: str, *, preload_all: bool) -> int:
        tracemalloc.start()
        tracemalloc.reset_peak()
        peak_combined: int = 0
        live: list[tuple[weakref.ref[Any], int, int]] = []
        active_open_shards: list[int] = [0]
        preload_all_flag: list[bool] = [preload_all]

        def _sample_peak() -> None:
          nonlocal peak_combined
          if not preload_all_flag[0]:
            self.assertLessEqual(active_open_shards[0], glm5_ckpt._MAX_CONCURRENT_OPEN_SHARDS)
          live[:] = [(r, ptr, sz) for r, ptr, sz in live if r() is not None]
          live_bytes = sum({ptr: sz for _, ptr, sz in live}.values())
          traced_cur, traced_peak = tracemalloc.get_traced_memory()
          peak_combined = max(
              peak_combined,
              traced_cur + live_bytes,
              traced_peak + live_bytes,
          )

        def _track_tensor(t: torch.Tensor, _: str) -> None:
          live.append(
              (
                  weakref.ref(t),
                  t.untyped_storage().data_ptr(),
                  t.numel() * t.element_size(),
              )
          )
          _sample_peak()
          orig_clone = t.clone

          def _clone_hook(*a, **kw):
            cloned = orig_clone(*a, **kw)
            live.append(
                (
                    weakref.ref(cloned),
                    cloned.untyped_storage().data_ptr(),
                    cloned.numel() * cloned.element_size(),
                )
            )
            _sample_peak()
            return cloned

          t.clone = _clone_hook

        def _on_enter() -> None:
          active_open_shards[0] += 1

        def _on_exit() -> None:
          active_open_shards[0] -= 1

        def side_fn(*a, **kw):
          return _SafeOpenTracker(
              orig_open(*a, **kw),
              on_get=_track_tensor,
              on_enter=_on_enter,
              on_exit=_on_exit,
          )

        orig_to = torch.Tensor.to

        def _hooked_to(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
          res_t = orig_to(tensor_self, *args, **kwargs)
          _track_tensor(res_t, "to")
          return res_t

        out_path = os.path.join(tmp_dir, out_name)
        with (
            mock.patch.object(safetensors, "safe_open", side_effect=side_fn),
            mock.patch.object(torch.Tensor, "to", _hooked_to),
        ):
          retained = []
          if preload_all:
            for sf in shards:
              with safetensors.safe_open(os.path.join(hf_dir, sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                  retained.append(f.get_tensor(k))
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              hf_dir,
              out_path,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=single_unit_bytes,
          )
          self.assertEqual(active_open_shards[0], 0)
          del retained
        tracemalloc.stop()
        if not preload_all:
          self.assertTrue(ocp.path.step.is_path_finalized(out_path))
          restored = ocp.StandardCheckpointer().restore(out_path)
          restored_leaves = jax.tree.leaves(restored)
          for leaf in restored_leaves:
            self.assertIsInstance(leaf, np.ndarray)
            b = leaf.base
            while isinstance(b, np.ndarray):
              b = b.base
            self.assertNotIsInstance(b, torch.Tensor)
          self.assertEqual(sum(leaf.nbytes for leaf in restored_leaves), pure_dict_bytes)
        return peak_combined

      peak_streaming = _run("obx1", preload_all=False)
      self.assertLessEqual(peak_streaming, expected_bound)
      self.assertLess(peak_streaming, pure_dict_bytes)
      peak_preload = _run("obx2", preload_all=True)
      self.assertGreater(peak_preload, expected_bound)

  @parameterized.parameters(False, True)
  def test_streaming_zarr_zstd_compressor_and_chunk_metadata(self, scan_layers: bool):
    """Verifies Zarr zstd compressor and chunking metadata across scanned and unscanned modes."""
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      c = dataclasses.replace(
          _make_cfg(11, scan_layers=scan_layers)[0],
          n_routed_experts=24,
          num_nextn_predict_layers=1,
      )
      _write_sharded_ckpt(hf_dir, c, num_shards=3)
      obx_dir = os.path.join(tmp_dir, "obx")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          obx_dir,
          c,
          ignore_mtp_layer=False,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )
      self.assertTrue(ocp.path.step.is_path_finalized(obx_dir))
      self.assertTrue(os.path.isfile(os.path.join(obx_dir, "_METADATA")))
      self.assertTrue(os.path.isfile(os.path.join(obx_dir, "_CHECKPOINT_METADATA")))

      ts_context = ocp.serialization.tensorstore_utils.get_ts_context(use_ocdbt=True)

      def _open_array(param_name: str) -> Any:
        read_spec = ocp._src.serialization.tensorstore_utils.ArrayReadSpec(
            obx_dir,
            param_name,
            use_zarr3=False,
            use_ocdbt=True,
        )
        return ts.open(
            ts.Spec(read_spec.json),
            open=True,
            context=ts_context,
        ).result()

      expert_chunk = math.gcd(glm5_ckpt._MAX_MOE_EXPERT_CHUNK_SIZE, c.n_routed_experts)

      store_moe = _open_array("super_block.slot_0.mlp.experts.gate_up_proj")
      moe_meta = store_moe.spec().to_json()["metadata"]
      self.assertEqual(moe_meta["compressor"], {"id": "zstd", "level": 1})
      self.assertEqual(
          tuple(moe_meta["shape"]),
          (2, 24, c.hidden_size, 2, c.moe_intermediate_size),
      )
      self.assertEqual(
          tuple(moe_meta["chunks"]),
          (1, expert_chunk, c.hidden_size, 2, c.moe_intermediate_size),
      )
      self.assertLess(moe_meta["chunks"][0], moe_meta["shape"][0])
      self.assertLess(moe_meta["chunks"][1], moe_meta["shape"][1])

      store_mtp_moe = _open_array("mtp.mlp.experts.gate_up_proj")
      mtp_moe_meta = store_mtp_moe.spec().to_json()["metadata"]
      self.assertEqual(mtp_moe_meta["compressor"], {"id": "zstd", "level": 1})
      self.assertEqual(
          tuple(mtp_moe_meta["shape"]),
          (24, c.hidden_size, 2, c.moe_intermediate_size),
      )
      self.assertEqual(
          tuple(mtp_moe_meta["chunks"]),
          (expert_chunk, c.hidden_size, 2, c.moe_intermediate_size),
      )
      self.assertLess(mtp_moe_meta["chunks"][0], mtp_moe_meta["shape"][0])

      store_attn = _open_array("super_block.slot_0.self_attn.q_a_proj.kernel")
      attn_meta = store_attn.spec().to_json()["metadata"]
      self.assertEqual(attn_meta["compressor"], {"id": "zstd", "level": 1})
      self.assertEqual(attn_meta["shape"][0], 2)
      self.assertEqual(tuple(attn_meta["chunks"]), (1, *attn_meta["shape"][1:]))
      self.assertLess(attn_meta["chunks"][0], attn_meta["shape"][0])

      store_embed = _open_array("embed_tokens.embedding")
      embed_meta = store_embed.spec().to_json()["metadata"]
      self.assertEqual(embed_meta["compressor"], {"id": "zstd", "level": 1})
      self.assertEqual(tuple(embed_meta["chunks"]), tuple(embed_meta["shape"]))

      restored_obx = ocp.StandardCheckpointer().restore(obx_dir)
      if scan_layers:
        in_mem_dir = os.path.join(tmp_dir, "in_mem")
        # Mock MAX_IN_MEMORY_CONVERSION_LAYERS to bypass the 6-layer cap for the 11-layer test oracle.
        with mock.patch.object(glm5_ckpt, "MAX_IN_MEMORY_CONVERSION_LAYERS", 12):
          glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
              hf_dir,
              in_mem_dir,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
          )
        restored_in_mem = ocp.StandardCheckpointer().restore(in_mem_dir)
        jax.tree.map(np.testing.assert_array_equal, restored_obx, restored_in_mem)

      if not scan_layers:
        orig_write_spec = ocp._src.serialization.tensorstore_utils.ArrayWriteSpec

        def _write_spec_0124(*args, **kwargs):
          spec_obj = orig_write_spec(*args, **kwargs)
          spec_obj._json_spec["metadata"]["compressor"] = {"id": "zstd"}
          return spec_obj

        obx_0124_dir = os.path.join(tmp_dir, "obx_0124")
        # Simulate orbax-checkpoint 0.12.4 ArrayWriteSpec emitting {"id": "zstd"} without "level".
        with mock.patch.object(
            ocp._src.serialization.tensorstore_utils,
            "ArrayWriteSpec",
            side_effect=_write_spec_0124,
        ):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              hf_dir,
              obx_0124_dir,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )
        read_spec_0124 = ocp._src.serialization.tensorstore_utils.ArrayReadSpec(
            obx_0124_dir,
            "embed_tokens.embedding",
            use_zarr3=False,
            use_ocdbt=True,
        )
        store_0124 = ts.open(
            ts.Spec(read_spec_0124.json),
            open=True,
            context=ts_context,
        ).result()
        self.assertEqual(
            store_0124.spec().to_json()["metadata"]["compressor"],
            {"id": "zstd", "level": 1},
        )
        restored_0124 = ocp.StandardCheckpointer().restore(obx_0124_dir)
        jax.tree.map(np.testing.assert_array_equal, restored_0124, restored_obx)

        def _write_spec_bad_comp(*args, **kwargs):
          spec_obj = orig_write_spec(*args, **kwargs)
          spec_obj._json_spec["metadata"]["compressor"] = {
              "id": "zstd",
              "level": 3,
          }
          return spec_obj

        # Verify non-standard compressor dicts (e.g. level != 1) still raise ValueError.
        with mock.patch.object(
            ocp._src.serialization.tensorstore_utils,
            "ArrayWriteSpec",
            side_effect=_write_spec_bad_comp,
        ):
          with self.assertRaisesRegex(ValueError, "Expected default Orbax zstd level 1 compressor"):
            glm5_ckpt.convert_hf_checkpoint_to_orbax(
                hf_dir,
                os.path.join(tmp_dir, "obx_bad_comp"),
                c,
                ignore_mtp_layer=False,
                slice_to_config_layers=False,
                num_workers=1,
                max_inflight_bytes=64 * 1024 * 1024,
            )

  def test_streaming_mid_run_failure_leaves_no_finalized_checkpoint(self):
    """Verifies mid-run failure leaves no finalized checkpoint or lingering temporary directory."""
    c = dataclasses.replace(
        _make_cfg(7, scan_layers=False)[0],
        num_nextn_predict_layers=1,
    )

    # (a) Mid-run failure during merge_ocdbt_per_process_files
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      _write_sharded_ckpt(hf_dir, c, num_shards=3)
      obx_dir = os.path.join(tmp_dir, "obx")
      with mock.patch.object(
          ocp.type_handlers,
          "merge_ocdbt_per_process_files",
          side_effect=RuntimeError("simulated ocdbt merge failure"),
      ):
        with self.assertRaisesRegex(RuntimeError, "simulated ocdbt merge failure"):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              hf_dir,
              obx_dir,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )
      self.assertFalse(ocp.path.step.is_path_finalized(obx_dir))
      self.assertEmpty([d for d in os.listdir(tmp_dir) if ".orbax-checkpoint-tmp" in d])
      with self.assertRaises((FileNotFoundError, ValueError)):
        ocp.StandardCheckpointer().restore(obx_dir)

    # (b) Mid-run failure during tensor streaming after earlier writes succeeded
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      _write_sharded_ckpt(hf_dir, c, num_shards=3)
      obx_dir = os.path.join(tmp_dir, "obx")
      # Corrupt index so that lm_head.weight points to an absent shard file
      index_path = os.path.join(hf_dir, "model.safetensors.index.json")
      with open(index_path, "r", encoding="utf-8") as f:
        idx_data = json.load(f)
      idx_data["weight_map"]["lm_head.weight"] = "absent.safetensors"
      with open(index_path, "w", encoding="utf-8") as f:
        json.dump(idx_data, f)

      with self.assertRaises((FileNotFoundError, ValueError)):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            obx_dir,
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )
      self.assertFalse(ocp.path.step.is_path_finalized(obx_dir))
      self.assertEmpty([d for d in os.listdir(tmp_dir) if ".orbax-checkpoint-tmp" in d])
      with self.assertRaises((FileNotFoundError, ValueError)):
        ocp.StandardCheckpointer().restore(obx_dir)

  def test_streamed_checkpoint_metadata_matches_standard_checkpointer(self):
    """Verifies streamed OCDBT checkpoint metadata matches StandardCheckpointer reference."""
    c, _ = _make_cfg(
        num_hidden_layers=3,
        scan_layers=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    c = dataclasses.replace(c, first_k_dense_replace=1)
    with tempfile.TemporaryDirectory() as tmpdir:
      hf_dir = os.path.join(tmpdir, "hf")
      os.makedirs(hf_dir)
      _write_sharded_ckpt(hf_dir, c, num_shards=2)

      streamed_out = os.path.join(tmpdir, "streamed_out")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          streamed_out,
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )

      ref_out = os.path.join(tmpdir, "ref_out")
      m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(m, streamed_out, mesh=None)
      glm5_ckpt.save_orbax_checkpoint(m, ref_out)

      checkpointer = ocp.StandardCheckpointer()
      streamed_meta = checkpointer.metadata(streamed_out).item_metadata.tree
      ref_meta = checkpointer.metadata(ref_out).item_metadata.tree

      self.assertEqual(jax.tree.structure(streamed_meta), jax.tree.structure(ref_meta))
      flat_streamed, _ = jax.tree.flatten(streamed_meta)
      flat_ref, _ = jax.tree.flatten(ref_meta)
      for s_meta, r_meta in zip(flat_streamed, flat_ref, strict=True):
        self.assertEqual(s_meta.shape, r_meta.shape)
        self.assertEqual(s_meta.dtype, r_meta.dtype)

      streamed_ckpt_meta_path = epath.Path(streamed_out) / "_CHECKPOINT_METADATA"
      ref_ckpt_meta_path = epath.Path(ref_out) / "_CHECKPOINT_METADATA"
      streamed_ckpt_meta = json.loads(streamed_ckpt_meta_path.read_text())
      ref_ckpt_meta = json.loads(ref_ckpt_meta_path.read_text())
      self.assertEqual(
          streamed_ckpt_meta["item_handlers"],
          ref_ckpt_meta["item_handlers"],
      )

  def test_streaming_error_branches(self):
    """Verifies streaming conversion error branches for discovery, MTP, and key mapping."""
    c = _make_cfg(7, scan_layers=False)[0]
    idx_fn = "model.safetensors.index.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
      for name, files, exc, regex in [
          (
              "bad",
              [("model-00001.safetensors", "d")],
              FileNotFoundError,
              "Expected model.safetensors.index.json",
          ),
          (
              "empty",
              [],
              FileNotFoundError,
              "Expected model.safetensors.index.json",
          ),
          (
              "m",
              [("model.safetensors", ""), ("m1.safetensors", "")],
              FileNotFoundError,
              "Multiple safetensors files",
          ),
          (
              "miss",
              [(idx_fn, '{"weight_map": {"x": "a"}}')],
              FileNotFoundError,
              "Missing referenced safetensors shards",
          ),
          ("idx", [(idx_fn, '{"foo": 1}')], ValueError, "missing 'weight_map'"),
      ]:
        d = os.path.join(tmp_dir, name)
        os.makedirs(d)
        for fn, cnt in files:
          with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
            f.write(cnt)
        with self.assertRaisesRegex(exc, regex):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              d,
              os.path.join(tmp_dir, "o"),
              c,
              ignore_mtp_layer=True,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )

      d_err = os.path.join(tmp_dir, "err")
      os.makedirs(d_err)
      hub, _ = _write_sharded_ckpt(d_err, c, num_shards=1)
      c_mtp = dataclasses.replace(c, num_nextn_predict_layers=1)
      with self.assertRaisesRegex(
          ValueError,
          "ignore_mtp_layer=True cannot be used when" " config.num_nextn_predict_layers > 0",
      ):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_err,
            os.path.join(tmp_dir, "o"),
            c_mtp,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )
      with self.assertRaisesRegex(
          ValueError,
          "ignore_mtp_layer=False requires config.num_nextn_predict_layers > 0",
      ):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_err,
            os.path.join(tmp_dir, "o"),
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      # Test shape mismatch validation against model structure
      d_shape = os.path.join(tmp_dir, "shape_err")
      os.makedirs(d_shape)
      _write_sharded_ckpt(d_shape, c, num_shards=1)
      c_wrong_hidden = dataclasses.replace(c, hidden_size=c.hidden_size * 2)
      with self.assertRaisesRegex(ValueError, "Shape mismatch for"):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_shape,
            os.path.join(tmp_dir, "o_shape"),
            c_wrong_hidden,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      specs = glm5_ckpt.iter_conversion_specs(c, include_mtp=False)
      # Mock iter_conversion_specs to duplicate keys and trigger Duplicate write error.
      with (
          mock.patch.object(
              glm5_ckpt,
              "iter_conversion_specs",
              return_value=[specs[0], *specs],
          ),
          self.assertRaisesRegex(ValueError, "Duplicate write to parameter slot"),
      ):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_err,
            os.path.join(tmp_dir, "o"),
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      sb_idx = next(i for i, s in enumerate(specs) if s.stack_index == 0)
      specs_sb = [
          *specs[:sb_idx],
          dataclasses.replace(specs[sb_idx], num_super_blocks=2),
          *specs[sb_idx + 1 :],
      ]
      # Mock iter_conversion_specs to simulate missing populated slot.
      with (
          mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=specs_sb),
          self.assertRaisesRegex(ValueError, "Unpopulated model parameter slot"),
      ):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_err,
            os.path.join(tmp_dir, "o"),
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      k_emb = "model.language_model.embed_tokens.weight"
      k_gp = "model.language_model.layers.0.mlp.gate_proj.weight"
      k_sci = f"{k_gp}_scale_inv"
      k_head = "model.language_model.lm_head.weight"
      k_norm = "model.language_model.norm.weight"

      specs_unconsumed = [
          dataclasses.replace(specs[0], hf_keys=(*specs[0].hf_keys, k_norm)),
          *[s for s in specs[1:] if s.target_path != ("norm", "weight")],
      ]
      # Mock iter_conversion_specs to simulate a valid_hf_keys entry that is never consumed.
      with (
          mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=specs_unconsumed),
          self.assertRaisesRegex(ValueError, "Unmapped key in hf_tensors"),
      ):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            d_err,
            os.path.join(tmp_dir, "o"),
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=64 * 1024 * 1024,
        )

      safetensors.torch.save_file({}, os.path.join(d_err, "absent.safetensors"))

      def _assert_err(bad_dict, regex, bad_map):
        safetensors.torch.save_file(bad_dict, os.path.join(d_err, "m.safetensors"))
        actual_map = bad_map if bad_map is not None else {k: "m.safetensors" for k in bad_dict}
        with open(
            os.path.join(d_err, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as f:
          json.dump({"weight_map": actual_map}, f)
        with self.assertRaisesRegex(ValueError, regex):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              d_err,
              os.path.join(tmp_dir, "o"),
              c,
              ignore_mtp_layer=True,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )

      for bad_hub, regex, bad_map in [
          (
              {k: v for k, v in hub.items() if k != k_emb},
              "Required parameter missing",
              None,
          ),
          (
              {k: v for k, v in hub.items() if k != k_sci},
              "missing scale inverse",
              None,
          ),
          (
              {**hub, k_gp: torch.randn(c.intermediate_size, c.hidden_size)},
              "present but weight .* is not float8",
              None,
          ),
          (
              {**hub, k_head: hub["lm_head.weight"].clone()},
              "Duplicate normalized key found",
              None,
          ),
          (
              {**hub, "model.language_model.extra.weight": torch.randn(2, 2)},
              "Unmapped key in hf_tensors",
              None,
          ),
          (
              hub,
              "is absent from shard file",
              {
                  **{k: "m.safetensors" for k in hub},
                  k_norm: "absent.safetensors",
              },
          ),
      ]:
        _assert_err(bad_hub, regex, bad_map)

      # 1. moe_gate_up expert shape mismatch (_run_work_unit in glm5_ckpt.py)
      with tempfile.TemporaryDirectory() as tmp_hf_dir:
        c_moe = dataclasses.replace(_make_cfg(7, scan_layers=False)[0], num_nextn_predict_layers=1)
        _write_sharded_ckpt(tmp_hf_dir, c_moe, num_shards=1)
        wrong_t = torch.randn(
            c_moe.moe_intermediate_size + 1,
            c_moe.hidden_size,
            dtype=torch.bfloat16,
        )
        safetensors.torch.save_file(
            {"model.language_model.layers.3.mlp.experts.0.gate_proj.weight": (wrong_t)},
            os.path.join(tmp_hf_dir, "wrong.safetensors"),
        )
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
          idx = json.load(f)
        idx["weight_map"]["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"] = "wrong.safetensors"
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as f:
          json.dump(idx, f)
        with self.assertRaisesRegex(ValueError, "Shape mismatch for .*gate_up_proj.* expert 0"):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              tmp_hf_dir,
              os.path.join(tmp_dir, "obx_err1"),
              c_moe,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )

      # 2. moe_down expert shape mismatch (_run_work_unit in glm5_ckpt.py)
      with tempfile.TemporaryDirectory() as tmp_hf_dir:
        _write_sharded_ckpt(tmp_hf_dir, c_moe, num_shards=1)
        wrong_t = torch.randn(
            c_moe.hidden_size + 1,
            c_moe.moe_intermediate_size,
            dtype=torch.bfloat16,
        )
        safetensors.torch.save_file(
            {"model.language_model.layers.3.mlp.experts.0.down_proj.weight": (wrong_t)},
            os.path.join(tmp_hf_dir, "wrong.safetensors"),
        )
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
          idx = json.load(f)
        idx["weight_map"]["model.language_model.layers.3.mlp.experts.0.down_proj.weight"] = "wrong.safetensors"
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as f:
          json.dump(idx, f)
        with self.assertRaisesRegex(ValueError, "Shape mismatch for .*down_proj.* expert 0"):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              tmp_hf_dir,
              os.path.join(tmp_dir, "obx_err2"),
              c_moe,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )

      # 3. Stacked direct/transpose_2d slot 1 shape mismatch (_run_work_unit in glm5_ckpt.py)
      with tempfile.TemporaryDirectory() as tmp_hf_dir:
        c11 = dataclasses.replace(_make_cfg(11, scan_layers=False)[0], num_nextn_predict_layers=1)
        _write_sharded_ckpt(tmp_hf_dir, c11, num_shards=1)
        wrong_t = torch.randn(c11.hidden_size + 1, dtype=torch.bfloat16)
        safetensors.torch.save_file(
            {"model.language_model.layers.7.input_layernorm.weight": wrong_t},
            os.path.join(tmp_hf_dir, "wrong.safetensors"),
        )
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
          idx = json.load(f)
        idx["weight_map"]["model.language_model.layers.7.input_layernorm.weight"] = "wrong.safetensors"
        with open(
            os.path.join(tmp_hf_dir, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as f:
          json.dump(idx, f)
        with self.assertRaisesRegex(ValueError, "Shape mismatch for .*input_layernorm.* \\(slot 1\\)"):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              tmp_hf_dir,
              os.path.join(tmp_dir, "obx_err3"),
              c11,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=1,
              max_inflight_bytes=64 * 1024 * 1024,
          )

  def test_assign_b_idx_bounds_raises(self):
    """Verifies that _assign bounds checks work and kwargs are required."""
    c, _ = _make_cfg(num_hidden_layers=7, scan_layers=True)
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)

    with self.assertRaises(TypeError):
      glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory({}, "/tmp/out", c)  # pylint: disable=missing-kwoa
    with self.assertRaises(TypeError):
      glm5_ckpt.convert_hf_checkpoint_to_orbax(  # pylint: disable=missing-kwoa
          "/tmp/in", "/tmp/out", c, num_workers=1, max_inflight_bytes=100
      )
    with self.assertRaises(TypeError):
      glm5_ckpt.restore_orbax_checkpoint(jax_m, "/tmp/out")  # pylint: disable=missing-kwoa

    with self.assertRaises(TypeError):
      glm5_ckpt.convert_hf_checkpoint_to_orbax(  # pylint: disable=missing-kwoa
          "/tmp/in",
          "/tmp/out",
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=False,
      )
    for bad_nw in (0, -1, True):
      with self.assertRaisesRegex(ValueError, "num_workers must be an integer >= 1"):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            "/tmp/in",
            "/tmp/out",
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=bad_nw,
            max_inflight_bytes=1024,
        )
    for bad_mib in (0, -1, True):
      with self.assertRaisesRegex(ValueError, "max_inflight_bytes must be an integer >= 1"):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            "/tmp/in",
            "/tmp/out",
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=1,
            max_inflight_bytes=bad_mib,
        )

    real_specs = glm5_ckpt.iter_conversion_specs(c, include_mtp=False)
    self.assertTrue(any(s.stack_index is not None for s in real_specs))

    def mock_transform(spec, get_fn):
      del get_fn
      v = glm5_ckpt.resolve_target_var(jax_m, spec.target_path).value
      shape = v.shape[1:] if spec.target_path[0] == "super_block" else v.shape
      return jnp.zeros(shape, dtype=spec.target_dtype)

    def _test_with_mutated_specs(mutate_fn, regex):
      mutated_specs = [mutate_fn(s) for s in real_specs]
      # Mock iter_conversion_specs solely to inject out-of-range bounds for testing _assign error paths.
      with mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=mutated_specs):
        with mock.patch.object(glm5_ckpt, "_transform_for_spec", side_effect=mock_transform):
          with self.assertRaisesRegex(ValueError, regex):
            glm5_ckpt.load_hf_state_into_model(jax_m, {}, ignore_mtp_layer=True)

    # Positive liveness call with unmutated specs to verify success on the 7-layer scanned model
    with mock.patch.object(glm5_ckpt, "_transform_for_spec", side_effect=mock_transform):
      glm5_ckpt.load_hf_state_into_model(jax_m, {}, ignore_mtp_layer=True)

    # Negative check: omit one stacked spec to test the unpopulated slot error
    stacked_spec = next(s for s in real_specs if s.stack_index is not None)
    omitted_specs = [s for s in real_specs if s is not stacked_spec]
    # Mock iter_conversion_specs solely to omit a spec and test the unpopulated slot error.
    with mock.patch.object(glm5_ckpt, "iter_conversion_specs", return_value=omitted_specs):
      with mock.patch.object(glm5_ckpt, "_transform_for_spec", side_effect=mock_transform):
        with self.assertRaisesRegex(ValueError, "Unpopulated model parameter slot"):
          glm5_ckpt.load_hf_state_into_model(jax_m, {}, ignore_mtp_layer=True)

    # 1. stack_index=2 on stacked specs -> out of bounds
    _test_with_mutated_specs(
        lambda s: dataclasses.replace(s, stack_index=2) if s.stack_index is not None else s,
        "out of bounds",
    )

    # 2. stack_index=-1 on stacked specs -> out of bounds
    _test_with_mutated_specs(
        lambda s: dataclasses.replace(s, stack_index=-1) if s.stack_index is not None else s,
        "out of bounds",
    )

    # 3. stack_index=1 on unstacked specs -> b_idx must be 0
    _test_with_mutated_specs(
        lambda s: dataclasses.replace(s, stack_index=1) if s.stack_index is None else s,
        "b_idx must be 0 when is_stacked=False",
    )

    # 4. stack_index=None on stacked specs -> requires a non-None stack_index
    _test_with_mutated_specs(
        lambda s: dataclasses.replace(s, stack_index=None) if s.stack_index is not None else s,
        "requires a non-None stack_index",
    )

  def test_restore_orbax_checkpoint_sharded(self):
    """Verifies restoring an Orbax checkpoint with different sharding branch configurations."""
    if len(jax.devices()) < 4 and "TEST_SRCDIR" not in os.environ:
      self.skipTest("Requires at least 4 JAX devices.")
    c, _ = _make_cfg(num_hidden_layers=7, scan_layers=True)
    self.assertGreaterEqual(len(jax.devices()), 4)
    devices = np.array(jax.devices()[:4]).reshape((1, 4, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))

    with tempfile.TemporaryDirectory() as tmp_dir:
      obx_dir = os.path.join(tmp_dir, "obx")
      with jax.sharding.set_mesh(mesh):
        src_m = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(0))
      glm5_ckpt.save_orbax_checkpoint(src_m, obx_dir)

      # Branch 2: model.sharding is not None, mesh=None
      with jax.sharding.set_mesh(mesh):
        dst_m = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(1))
        glm5_ckpt.restore_orbax_checkpoint(dst_m, obx_dir, mesh=None)

      pure_src = nnx.to_pure_dict(nnx.state(src_m))
      pure_dst = nnx.to_pure_dict(nnx.state(dst_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_dst)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(dl.sharding.mesh, mesh)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      for var in jax.tree.leaves(nnx.state(dst_m), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        if (
            isinstance(var, nnx.Variable)
            and var.has_metadata("out_sharding")
            and var.get_metadata("out_sharding") is not None
        ):
          out_sh = var.get_metadata("out_sharding")
          spec = out_sh.spec if isinstance(out_sh, jax.sharding.NamedSharding) else out_sh
          self.assertEqual(
              var.value.sharding,
              jax.sharding.NamedSharding(mesh, spec),
          )

      self.assertEqual(
          dst_m.super_block.slot_0.attn_hc.fn.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "fsdp", None)),
      )
      exp_var = dst_m.super_block.slot_0.mlp.experts.gate_up_proj
      out_sh = exp_var.get_metadata("out_sharding")
      spec = out_sh.spec if isinstance(out_sh, jax.sharding.NamedSharding) else out_sh
      self.assertEqual(
          exp_var.value.sharding,
          jax.sharding.NamedSharding(mesh, spec),
      )
      self.assertEqual(
          exp_var.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      self.assertEqual(
          dst_m.prefix_layers[0].attn_hc.scale.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
      )
      self.assertEqual(
          dst_m.super_block.slot_0.attn_hc.scale.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None)),
      )

      # Branch with mesh != None raises ValueError
      with self.assertRaisesRegex(
          ValueError,
          "Passing mesh to restore_orbax_checkpoint is not supported",
      ):
        glm5_ckpt.restore_orbax_checkpoint(dst_m, obx_dir, mesh=mesh)

      # Restore into unplaced model (place_on_devices=False, mesh=mesh)
      with jax.sharding.set_mesh(mesh):
        dst_unplaced = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=False, rngs=nnx.Rngs(2))
      glm5_ckpt.restore_orbax_checkpoint(dst_unplaced, obx_dir, mesh=None)
      self.assertEqual(
          dst_unplaced.super_block.slot_0.mlp.experts.gate_up_proj.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      pure_unplaced = nnx.to_pure_dict(nnx.state(dst_unplaced))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_unplaced)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      # Restore into abstract model (nnx.eval_shape)
      with jax.sharding.set_mesh(mesh):
        abs_m = nnx.eval_shape(
            lambda: modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), mesh=mesh, place_on_devices=False)
        )
      glm5_ckpt.restore_orbax_checkpoint(abs_m, obx_dir, mesh=None)
      self.assertEqual(
          abs_m.super_block.slot_0.mlp.experts.gate_up_proj.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      pure_abs = nnx.to_pure_dict(nnx.state(abs_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_abs)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      # Unsharded restore: model.sharding is None, mesh=None (CPU host sharding)
      host_m = modeling_glm5_3_flash.Glm5Model(c, place_on_devices=False, rngs=nnx.Rngs(3))
      with self.assertRaisesRegex(
          ValueError,
          "Passing mesh to restore_orbax_checkpoint is not supported",
      ):
        glm5_ckpt.restore_orbax_checkpoint(host_m, obx_dir, mesh=mesh)
      glm5_ckpt.restore_orbax_checkpoint(host_m, obx_dir, mesh=None)
      pure_host = nnx.to_pure_dict(nnx.state(host_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_host)):
        self.assertIsInstance(dl.sharding, jax.sharding.SingleDeviceSharding)
        self.assertEqual(dl.sharding.device_set, {jax.local_devices(backend="cpu")[0]})
        np.testing.assert_array_equal(sl, dl)

      # Error guard: model.sharding is set, but model.mesh is None
      bad_m = modeling_glm5_3_flash.Glm5Model(c, place_on_devices=False, rngs=nnx.Rngs(4))
      bad_m.sharding = jax.sharding.PartitionSpec()
      bad_m.mesh = None
      with self.assertRaisesRegex(ValueError, "model.sharding is set, but model.mesh is None or empty"):
        glm5_ckpt.restore_orbax_checkpoint(bad_m, obx_dir, mesh=None)

      # Error guard: mesh=None with model.sharding=None when model leaves already carry NamedSharding
      with jax.sharding.set_mesh(mesh):
        unsharded_with_shards = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(5))
      unsharded_with_shards.sharding = None
      with self.assertRaisesRegex(
          ValueError,
          "cannot be used when model leaves already carry NamedSharding",
      ):
        glm5_ckpt.restore_orbax_checkpoint(unsharded_with_shards, obx_dir, mesh=None)

  def test_cli_arg_parser_missing_required_flags_raises(self):
    """Verifies that the CLI parser raises SystemExit when required flags are missing."""
    parser = glm5_ckpt.build_arg_parser()
    valid_flag_pairs = [
        ("--hf_checkpoint_dir", "a"),
        ("--orbax_output_dir", "b"),
        ("--weight_dtype", "bfloat16"),
        ("--ignore_mtp_layer", "true"),
        ("--slice_to_config_layers", "false"),
        ("--num_hidden_layers", "45"),
        ("--num_nextn_predict_layers", "1"),
        ("--config_json", "none"),
        ("--num_workers", "2"),
        ("--max_inflight_bytes", "64MiB"),
    ]
    all_args = [arg for pair in valid_flag_pairs for arg in pair]
    parsed = parser.parse_args(all_args)
    self.assertEqual(parsed.num_hidden_layers, 45)
    self.assertEqual(parsed.num_nextn_predict_layers, 1)

    for i in range(0, len(all_args), 2):
      args_missing_one = all_args[:i] + all_args[i + 2 :]
      with self.assertRaises(SystemExit) as cm:
        parser.parse_args(args_missing_one)
      self.assertNotEqual(cm.exception.code, 0)

  def test_cli_canonical_and_custom_config_construction(self):
    """Verifies Glm5Config construction with explicit CLI overrides."""
    parser = glm5_ckpt.build_arg_parser()
    base_argv = [
        "--hf_checkpoint_dir",
        "a",
        "--orbax_output_dir",
        "b",
        "--weight_dtype",
        "bfloat16",
        "--ignore_mtp_layer",
        "true",
        "--slice_to_config_layers",
        "false",
        "--num_hidden_layers",
        "45",
        "--num_nextn_predict_layers",
        "1",
        "--config_json",
        "none",
        "--num_workers",
        "2",
        "--max_inflight_bytes",
        "64MiB",
    ]
    args = parser.parse_args(base_argv)
    cfg = glm5_ckpt.create_glm5_config_from_cli_args(args)
    self.assertEqual(cfg.num_hidden_layers, 45)
    self.assertEqual(cfg.num_nextn_predict_layers, 1)
    self.assertEqual(cfg.scan_layers, True)
    self.assertEqual(cfg.dtype, "bfloat16")
    self.assertEqual(cfg.weight_dtype, "bfloat16")
    self.assertEqual(cfg.hidden_size, 4096)
    self.assertEqual(cfg.n_routed_experts, 288)
    self.assertEqual(cfg.remat_policy, "full")
    self.assertEqual(len(cfg.layer_types), 45)

    args.num_hidden_layers = 5
    args.num_nextn_predict_layers = 0
    cfg5 = glm5_ckpt.create_glm5_config_from_cli_args(args)
    self.assertEqual(cfg5.num_hidden_layers, 5)
    self.assertEqual(cfg5.num_nextn_predict_layers, 0)
    self.assertEqual(cfg5.remat_policy, "full")
    self.assertEqual(len(cfg5.layer_types), 5)

    with self.assertRaises(SystemExit):
      parser.parse_args(base_argv + ["--dtype", "float32"])

    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_f = os.path.join(tmpdir, "cfg.json")
      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump({"layer_types": ["linear_attention", "full_attention"]}, fd)
      args.config_json = cfg_f
      args.num_hidden_layers = 2
      cfg_custom = glm5_ckpt.create_glm5_config_from_cli_args(args)
      self.assertIsInstance(cfg_custom.layer_types, tuple)
      self.assertEqual(cfg_custom.layer_types, ("linear_attention", "full_attention"))

  def test_cli_flag_validation_invalid_values_raises(self):
    """Tests CLI flag validation raises exact expected errors."""

    def _parse(update_dict):
      args_dict = {
          "hf_checkpoint_dir": "a",
          "orbax_output_dir": "b",
          "weight_dtype": "bfloat16",
          "ignore_mtp_layer": "true",
          "slice_to_config_layers": "false",
          "num_hidden_layers": 45,
          "num_nextn_predict_layers": 1,
          "config_json": "none",
      }
      args_dict.update(update_dict)
      return argparse.Namespace(**args_dict)

    with self.assertRaisesRegex(ValueError, "Unsupported --weight_dtype"):
      glm5_ckpt.create_glm5_config_from_cli_args(_parse({"weight_dtype": "int8"}))
    with self.assertRaisesRegex(ValueError, "--num_hidden_layers must be >= 1"):
      glm5_ckpt.create_glm5_config_from_cli_args(_parse({"num_hidden_layers": 0}))
    with self.assertRaisesRegex(ValueError, "--num_hidden_layers must be >= 1"):
      glm5_ckpt.create_glm5_config_from_cli_args(_parse({"num_hidden_layers": -5}))
    with self.assertRaisesRegex(ValueError, "--num_nextn_predict_layers must be 0 or 1"):
      glm5_ckpt.create_glm5_config_from_cli_args(_parse({"num_nextn_predict_layers": 2}))

    def _run(argv):
      glm5_ckpt.run_cli(argv)

    with tempfile.TemporaryDirectory() as tmpdir:
      base_argv = [
          "--hf_checkpoint_dir",
          tmpdir,
          "--orbax_output_dir",
          os.path.join(tmpdir, "out"),
          "--weight_dtype",
          "bfloat16",
          "--ignore_mtp_layer",
          "true",
          "--slice_to_config_layers",
          "false",
          "--num_hidden_layers",
          "45",
          "--num_nextn_predict_layers",
          "0",
          "--config_json",
          "none",
          "--num_workers",
          "2",
          "--max_inflight_bytes",
          "64MiB",
      ]

      with self.assertRaises(FileNotFoundError):
        _run(base_argv)

      argv = list(base_argv)
      argv[argv.index("--num_workers") + 1] = "0"
      with self.assertRaisesRegex(ValueError, "--num_workers must be >= 1"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--max_inflight_bytes") + 1] = "0"
      with self.assertRaisesRegex(ValueError, "max_inflight_bytes must be >= 1"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--max_inflight_bytes") + 1] = "bad"
      with self.assertRaisesRegex(ValueError, "Invalid format for max_inflight_bytes"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--max_inflight_bytes") + 1] = "0MiB"
      with self.assertRaisesRegex(ValueError, "max_inflight_bytes must be >= 1"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--max_inflight_bytes") + 1] = "badMiB"
      with self.assertRaisesRegex(ValueError, "Invalid numeric part in max_inflight_bytes"):
        _run(argv)

      argv = list(base_argv)
      argv[1] = os.path.join(tmpdir, "missing")
      with self.assertRaisesRegex(FileNotFoundError, "HF checkpoint directory not found"):
        _run(argv)

      argv = list(base_argv)
      argv[3] = "  "
      with self.assertRaisesRegex(ValueError, "--orbax_output_dir must be a non-empty path"):
        _run(argv)

      f = os.path.join(tmpdir, "file")
      with open(f, "w", encoding="utf-8") as fd:
        fd.write("a")
      argv = list(base_argv)
      argv[3] = f
      with self.assertRaisesRegex(FileExistsError, "Refusing to write into existing non-empty"):
        _run(argv)

      os.makedirs(os.path.join(tmpdir, "dir1", "file1"))
      argv = list(base_argv)
      argv[3] = os.path.join(tmpdir, "dir1")
      with self.assertRaisesRegex(FileExistsError, "Refusing to write into existing non-empty"):
        _run(argv)

      argv = list(base_argv)
      argv[3] = "s3://my-bucket/out"
      with self.assertRaisesRegex(ValueError, "Unsupported URI scheme in --orbax_output_dir"):
        _run(argv)

      argv = list(base_argv)
      argv[3] = "http://example.com/out"
      with self.assertRaisesRegex(ValueError, "Unsupported URI scheme in --orbax_output_dir"):
        _run(argv)

      argv = list(base_argv)
      argv[1] = "gs://my-bucket/hf"
      with self.assertRaisesRegex(ValueError, "Unsupported URI scheme in --hf_checkpoint_dir"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--config_json") + 1] = "gs://my-bucket/cfg.json"
      with self.assertRaisesRegex(ValueError, "Unsupported URI scheme in --config_json"):
        _run(argv)

      with epath.testing.mock_epath(
          exists=lambda orig, p: True,
          isdir=lambda orig, p: True,
          listdir=lambda orig, p: ["0"],
      ):
        argv = list(base_argv)
        argv[3] = "gs://test-bucket/existing-non-empty"
        with self.assertRaisesRegex(
            FileExistsError,
            "Refusing to write into existing non-empty --orbax_output_dir",
        ):
          _run(argv)

      with epath.testing.mock_epath(exists=lambda orig, p: True, isdir=lambda orig, p: False):
        argv = list(base_argv)
        argv[3] = "gs://test-bucket/existing-file"
        with self.assertRaisesRegex(
            FileExistsError,
            "Refusing to write into existing non-empty --orbax_output_dir",
        ):
          _run(argv)

      with epath.testing.mock_epath(
          exists=lambda orig, p: True,
          isdir=lambda orig, p: True,
          listdir=lambda orig, p: [],
      ):
        argv = list(base_argv)
        argv[3] = "gs://test-bucket/empty-dir"
        with self.assertRaisesRegex(FileNotFoundError, "Expected model.safetensors.index.json"):
          _run(argv)

      argv = list(base_argv)
      argv[argv.index("--ignore_mtp_layer") + 1] = "true"
      argv[argv.index("--num_nextn_predict_layers") + 1] = "1"
      with self.assertRaisesRegex(ValueError, "ignore_mtp_layer=True cannot be used when"):
        _run(argv)

      argv = list(base_argv)
      argv[argv.index("--ignore_mtp_layer") + 1] = "false"
      argv[argv.index("--num_nextn_predict_layers") + 1] = "0"
      with self.assertRaisesRegex(ValueError, "ignore_mtp_layer=False requires"):
        _run(argv)

      cfg_f = os.path.join(tmpdir, "cfg.json")
      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump({"num_hidden_layers": 5}, fd)
      argv = list(base_argv)
      argv[argv.index("--config_json") + 1] = cfg_f
      with self.assertRaisesRegex(ValueError, "config_json must not specify CLI-owned keys"):
        _run(argv)

      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump({"unknown_field": 123}, fd)
      with self.assertRaisesRegex(ValueError, "Unknown Glm5Config keys in config_json"):
        _run(argv)

      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump({}, fd)
      with self.assertRaisesRegex(ValueError, "config_json must contain a non-empty JSON object"):
        _run(argv)

      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump({"layer_types": "bad"}, fd)
      with self.assertRaisesRegex(TypeError, "Expected JSON list for"):
        _run(argv)

      for flag in [
          "--ignore_mtp_layer",
          "--slice_to_config_layers",
      ]:
        for invalid_bool in ["1", "0", "yes", "maybe", ""]:
          argv = list(base_argv)
          argv[argv.index(flag) + 1] = invalid_bool
          with self.assertRaisesRegex(ValueError, "Invalid boolean value"):
            _run(argv)

      argv = list(base_argv)
      argv[argv.index("--config_json") + 1] = os.path.join(tmpdir, "nonexistent_cfg.json")
      with self.assertRaisesRegex(FileNotFoundError, "config_json file not found"):
        _run(argv)

      with open(cfg_f, "w", encoding="utf-8") as fd:
        json.dump([1, 2, 3], fd)
      argv = list(base_argv)
      argv[argv.index("--config_json") + 1] = cfg_f
      with self.assertRaisesRegex(ValueError, "config_json must contain a non-empty JSON object"):
        _run(argv)

  def test_cli_end_to_end_synthetic_conversion(self):
    """End-to-end synthetic conversion tests."""
    c, _ = _make_cfg(
        num_hidden_layers=7,
        scan_layers=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    c = dataclasses.replace(c, num_nextn_predict_layers=1)

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_dir = os.path.join(tmpdir, "hf")
      os.makedirs(hf_dir)

      _write_sharded_ckpt(hf_dir, c, num_shards=3)

      cfg_f = os.path.join(tmpdir, "cfg.json")
      _write_config_json(cfg_f, c)

      obx_dir = os.path.join(tmpdir, "obx")
      obx_dir_ref = os.path.join(tmpdir, "obx_ref")

      argv = [
          "convert_glm5_ckpt",
          "--hf_checkpoint_dir",
          hf_dir,
          "--orbax_output_dir",
          obx_dir,
          "--weight_dtype",
          "bfloat16",
          "--ignore_mtp_layer",
          "false",
          "--slice_to_config_layers",
          "false",
          "--num_hidden_layers",
          "7",
          "--num_nextn_predict_layers",
          "1",
          "--config_json",
          cfg_f,
          "--num_workers",
          "2",
          "--max_inflight_bytes",
          "64MiB",
      ]

      with mock.patch.object(sys, "argv", argv):
        glm5_ckpt.main()

      # The in-memory oracle is test-only and capped at 6 layers by default;
      # the production streaming CLI path has no such limit.
      with mock.patch.object(glm5_ckpt, "MAX_IN_MEMORY_CONVERSION_LAYERS", 8):
        glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
            hf_dir,
            obx_dir_ref,
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
        )
      self.assertEqual(glm5_ckpt.MAX_IN_MEMORY_CONVERSION_LAYERS, 6)

      raw_ckpt = ocp.StandardCheckpointer().restore(obx_dir)
      self.assertIn("super_block", raw_ckpt)

      for scan_mode in (True, False):
        c_mode = dataclasses.replace(c, scan_layers=scan_mode)
        m = modeling_glm5_3_flash.Glm5Model(c_mode, rngs=nnx.Rngs(0), place_on_devices=False)
        glm5_ckpt.restore_orbax_checkpoint(m, obx_dir, mesh=None)

        m_ref = modeling_glm5_3_flash.Glm5Model(c_mode, rngs=nnx.Rngs(0), place_on_devices=False)
        glm5_ckpt.restore_orbax_checkpoint(m_ref, obx_dir_ref, mesh=None)

        if scan_mode:
          self.assertIsNotNone(m.super_block)
        else:
          self.assertIsNotNone(m.prefix_layers)
          self.assertIsNotNone(m.suffix_layers)

        state_m = nnx.to_pure_dict(nnx.state(m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
        state_ref = nnx.to_pure_dict(nnx.state(m_ref, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))

        flat_m, _ = jax.tree_util.tree_flatten_with_path(state_m)
        flat_ref, _ = jax.tree_util.tree_flatten_with_path(state_ref)
        self.assertEqual([p for p, _ in flat_m], [p for p, _ in flat_ref])
        for (_, g), (_, r) in zip(flat_m, flat_ref, strict=True):
          np.testing.assert_array_equal(g, r)

  def test_cli_end_to_end_sliced_conversion(self):
    """End-to-end sliced conversion tests."""
    c, _ = _make_cfg(
        num_hidden_layers=7,
        scan_layers=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    c = dataclasses.replace(c, num_nextn_predict_layers=1)

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_dir = os.path.join(tmpdir, "hf")
      os.makedirs(hf_dir)

      _write_sharded_ckpt(hf_dir, c, num_shards=3)

      cfg_f = os.path.join(tmpdir, "cfg.json")
      c5, _ = _make_cfg(
          num_hidden_layers=5,
          scan_layers=False,
          dtype="bfloat16",
          weight_dtype="bfloat16",
      )
      _write_config_json(cfg_f, c5)

      obx_dir = os.path.join(tmpdir, "obx")
      obx_dir_ref = os.path.join(tmpdir, "obx_ref")

      argv = [
          "convert_glm5_ckpt",
          "--hf_checkpoint_dir",
          hf_dir,
          "--orbax_output_dir",
          obx_dir,
          "--weight_dtype",
          "bfloat16",
          "--ignore_mtp_layer",
          "true",
          "--slice_to_config_layers",
          "true",
          "--num_hidden_layers",
          "5",
          "--num_nextn_predict_layers",
          "0",
          "--config_json",
          cfg_f,
          "--num_workers",
          "2",
          "--max_inflight_bytes",
          "64MiB",
      ]

      with mock.patch.object(sys, "argv", argv):
        glm5_ckpt.main()

      glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
          hf_dir,
          obx_dir_ref,
          c5,
          ignore_mtp_layer=True,
          slice_to_config_layers=True,
      )

      m = modeling_glm5_3_flash.Glm5Model(c5, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(m, obx_dir, mesh=None)

      m_ref = modeling_glm5_3_flash.Glm5Model(c5, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(m_ref, obx_dir_ref, mesh=None)

      leaves_m = jax.tree.leaves(nnx.state(m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      leaves_ref = jax.tree.leaves(nnx.state(m_ref, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      self.assertEqual(len(leaves_m), len(leaves_ref))
      for g, r in zip(leaves_m, leaves_ref, strict=True):
        np.testing.assert_array_equal(g, r)

  def test_cli_standalone_entrypoint_subprocess(self):
    c, _ = _make_cfg(
        num_hidden_layers=4,
        scan_layers=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      hf_dir = os.path.join(tmpdir, "hf")
      os.makedirs(hf_dir)
      _write_sharded_ckpt(hf_dir, c, num_shards=2)

      cfg_f = os.path.join(tmpdir, "cfg.json")
      _write_config_json(cfg_f, c)

      obx_dir = os.path.join(tmpdir, "obx")
      obx_dir_ref = os.path.join(tmpdir, "obx_ref")

      pkg_dir = os.path.dirname(os.path.abspath(glm5_ckpt.__file__))
      cli_bin = os.path.join(pkg_dir, "convert_glm5_ckpt")
      cli_py = os.path.join(pkg_dir, "convert_glm5_ckpt.py")

      cli_args = [
          "--hf_checkpoint_dir",
          hf_dir,
          "--orbax_output_dir",
          obx_dir,
          "--weight_dtype",
          "bfloat16",
          "--ignore_mtp_layer",
          "true",
          "--slice_to_config_layers",
          "false",
          "--num_hidden_layers",
          "4",
          "--num_nextn_predict_layers",
          "0",
          "--config_json",
          cfg_f,
          "--num_workers",
          "2",
          "--max_inflight_bytes",
          "64MiB",
      ]

      if os.path.isfile(cli_bin):
        cmd = [cli_bin, *cli_args]
      else:
        self.assertNotIn(
            "TEST_SRCDIR",
            os.environ,
            f"convert_glm5_ckpt binary not found at: {cli_bin}",
        )
        self.assertTrue(
            os.path.isfile(cli_py),
            f"convert_glm5_ckpt script not found at: {cli_py}",
        )
        cmd = [sys.executable, cli_py, *cli_args]

      env = os.environ.copy()
      env["JAX_PLATFORMS"] = "cpu"
      env.pop("JAX_PLATFORM_NAME", None)
      res = subprocess.run(
          cmd,
          env=env,
          capture_output=True,
          text=True,
          timeout=120,
          check=False,
      )
      self.assertEqual(res.returncode, 0, msg=res.stderr)

      glm5_ckpt._convert_hf_checkpoint_to_orbax_in_memory(
          hf_dir,
          obx_dir_ref,
          c,
          ignore_mtp_layer=True,
          slice_to_config_layers=False,
      )

      m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(m, obx_dir, mesh=None)

      m_ref = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.restore_orbax_checkpoint(m_ref, obx_dir_ref, mesh=None)

      leaves_m = jax.tree.leaves(nnx.state(m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      leaves_ref = jax.tree.leaves(nnx.state(m_ref, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      self.assertEqual(len(leaves_m), len(leaves_ref))
      for g, r in zip(leaves_m, leaves_ref, strict=True):
        np.testing.assert_array_equal(g, r)

  @parameterized.named_parameters(
      ("float32_scanned", "float32", True),
      ("bfloat16_unscanned", "bfloat16", False),
  )
  def test_streaming_peak_memory_bound_parallel(self, target_dtype: str, scan_layers: bool):
    """Verifies parallel peak memory bounds and concurrent shard LRU limits."""
    c = dataclasses.replace(
        _make_cfg(
            11,
            scan_layers=scan_layers,
            dtype=target_dtype,
            weight_dtype=target_dtype,
        )[0],
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=128,
        n_routed_experts=24,
        vocab_size=1024,
        num_nextn_predict_layers=1,
    )
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      _write_sharded_ckpt(hf_dir, c, num_shards=12)

      abs_m = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False))
      specs = glm5_ckpt.iter_conversion_specs(c, include_mtp=c.num_nextn_predict_layers > 0)

      dt = np.dtype(jnp.dtype(target_dtype))
      target_itemsize = dt.itemsize
      compute_itemsize = glm5_ckpt._TARGET_TO_INTERMEDIATE_DTYPE[dt][1].itemsize
      max_non_moe_leaf_bytes = max(
          int(
              np.prod(
                  glm5_ckpt.resolve_target_var(abs_m, s.target_path).value.shape[1:]
                  if s.stack_index is not None
                  else glm5_ckpt.resolve_target_var(abs_m, s.target_path).value.shape
              )
          )
          * (
              (compute_itemsize + 3 * target_itemsize)
              if dt == np.dtype(ml_dtypes.bfloat16)
              else 3 * np.dtype(s.target_dtype).itemsize
          )
          for s in specs
          if s.kind in ("direct", "transpose_2d")
      )
      expert_chunk = math.gcd(glm5_ckpt._MAX_MOE_EXPERT_CHUNK_SIZE, c.n_routed_experts) if c.n_routed_experts > 0 else 1
      moe_chunk_bytes = (
          3 * expert_chunk * c.hidden_size * 2 * c.moe_intermediate_size * target_itemsize
          + 2 * c.hidden_size * c.moe_intermediate_size * compute_itemsize
      )
      fp8_gate_bytes = (
          3 * c.hidden_size * c.intermediate_size * target_itemsize
          + 2 * c.hidden_size * c.intermediate_size * compute_itemsize
      )
      single_unit_bytes = 10_485_760 if target_dtype == "bfloat16" else 26_214_400
      largest_work_unit_bytes = max(max_non_moe_leaf_bytes, moe_chunk_bytes, fp8_gate_bytes)

      max_inflight_bytes = 4 * single_unit_bytes

      active_open_shards = [0]
      max_open_shards = [0]
      lock = threading.Lock()

      def _on_enter():
        with lock:
          active_open_shards[0] += 1
          max_open_shards[0] = max(max_open_shards[0], active_open_shards[0])

      def _on_exit():
        with lock:
          active_open_shards[0] -= 1

      live_torch_bytes = [0]
      peak_torch_bytes = [0]
      torch_lock = threading.Lock()

      def _on_tensor_deleted(nbytes: int) -> None:
        with torch_lock:
          live_torch_bytes[0] -= nbytes

      def _track_tensor(t: Any, _k: str) -> None:
        if not isinstance(t, torch.Tensor):
          return
        nbytes = t.numel() * t.element_size()
        with torch_lock:
          live_torch_bytes[0] += nbytes
          if live_torch_bytes[0] > peak_torch_bytes[0]:
            peak_torch_bytes[0] = live_torch_bytes[0]
        fin = weakref.finalize(t, _on_tensor_deleted, nbytes)
        orig_clone = t.clone

        def _clone_with_finalizer(*a, **kw):
          cloned = orig_clone(*a, **kw)
          if fin.detach() is not None:
            weakref.finalize(cloned, _on_tensor_deleted, nbytes)
          return cloned

        t.clone = _clone_with_finalizer

      orig_open = safetensors.safe_open

      def side_fn(*a, **kw):
        return _SafeOpenTracker(
            orig_open(*a, **kw),
            on_get=_track_tensor,
            on_enter=_on_enter,
            on_exit=_on_exit,
        )

      obx_dir = os.path.join(tmp_dir, "obx")
      with mock.patch.object(safetensors, "safe_open", side_effect=side_fn):
        with self.assertRaisesRegex(ValueError, "smaller than the largest single work unit"):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              hf_dir,
              obx_dir,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=4,
              max_inflight_bytes=largest_work_unit_bytes - 1,
          )

        captured_sem = []
        orig_sem_cls = glm5_ckpt._MemoryBudgetSemaphore

        def _capturing_sem(*a, **kw):
          s = orig_sem_cls(*a, **kw)
          captured_sem.append(s)
          return s

        tracemalloc.start()
        tracemalloc.reset_peak()
        with mock.patch.object(glm5_ckpt, "_MemoryBudgetSemaphore", side_effect=_capturing_sem):
          glm5_ckpt.convert_hf_checkpoint_to_orbax(
              hf_dir,
              obx_dir,
              c,
              ignore_mtp_layer=False,
              slice_to_config_layers=False,
              num_workers=4,
              max_inflight_bytes=max_inflight_bytes,
          )
        _, peak_streaming = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(captured_sem), 1)
        self.assertGreaterEqual(captured_sem[0].max_concurrent_acquisitions, 2)
        self.assertLessEqual(captured_sem[0].peak_bytes, max_inflight_bytes)

      self.assertLessEqual(max_open_shards[0], glm5_ckpt._MAX_CONCURRENT_OPEN_SHARDS)
      self.assertEqual(active_open_shards[0], 0)

      shards = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
      shard_sizes = sorted(os.path.getsize(os.path.join(hf_dir, f)) for f in shards)
      two_shards_bytes = sum(shard_sizes[-2:])

      pure_leaves = jax.tree.leaves(nnx.state(abs_m, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)))
      pure_dict_bytes = sum(int(np.prod(v.shape)) * np.dtype(v.dtype).itemsize for v in pure_leaves)
      slack_bytes = int(0.10 * pure_dict_bytes)

      self.assertLessEqual(
          peak_streaming + peak_torch_bytes[0],
          max_inflight_bytes + two_shards_bytes + slack_bytes,
      )
      self.assertLess(max_inflight_bytes + two_shards_bytes + slack_bytes, pure_dict_bytes)

      # Positive control: bypassing semaphore with tight_budget = largest_work_unit_bytes
      # causes concurrent work-unit bytes to exceed tight_budget.
      tight_budget = largest_work_unit_bytes
      current_concurrent_bytes = [0]
      unbounded_peak_bytes = [0]
      bypass_lock = threading.Lock()

      class _BypassSemaphore:
        """No-op semaphore stub that tracks peak concurrent bytes without blocking."""

        def __init__(self, _budget):
          pass

        def acquire(self, nbytes):
          with bypass_lock:
            current_concurrent_bytes[0] += nbytes
            unbounded_peak_bytes[0] = max(unbounded_peak_bytes[0], current_concurrent_bytes[0])

        def release(self, nbytes):
          with bypass_lock:
            current_concurrent_bytes[0] -= nbytes

        @property
        def max_concurrent_acquisitions(self):
          return 4

        @property
        def peak_bytes(self):
          return unbounded_peak_bytes[0]

      obx_bypass = os.path.join(tmp_dir, "obx_bypass")
      with mock.patch.object(glm5_ckpt, "_MemoryBudgetSemaphore", _BypassSemaphore):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            obx_bypass,
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
            num_workers=4,
            max_inflight_bytes=tight_budget,
        )
      self.assertGreater(unbounded_peak_bytes[0], tight_budget)

      obx_seq = os.path.join(tmp_dir, "obx_seq")
      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          obx_seq,
          c,
          ignore_mtp_layer=False,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=max_inflight_bytes,
      )

      restored_par = ocp.StandardCheckpointer().restore(obx_dir)
      restored_seq = ocp.StandardCheckpointer().restore(obx_seq)
      jax.tree.map(np.testing.assert_array_equal, restored_par, restored_seq)

  def test_memory_budget_semaphore(self):
    """Verifies _MemoryBudgetSemaphore validation and deterministic multi-thread blocking."""
    with self.assertRaisesRegex(ValueError, "max_bytes must be positive"):
      glm5_ckpt._MemoryBudgetSemaphore(0)
    sem = glm5_ckpt._MemoryBudgetSemaphore(200)
    with self.assertRaisesRegex(ValueError, "exceeds total memory budget"):
      sem.acquire(201)
    with self.assertRaisesRegex(ValueError, "must be > 0"):
      sem.acquire(0)

    sem.acquire(120)
    sem.acquire(80)
    self.assertEqual(sem._current_bytes, 200)

    entering_wait = threading.Event()
    w3_acquired = threading.Event()
    orig_wait = sem._cond.wait

    def _tracked_wait(*a, **kw):
      entering_wait.set()
      return orig_wait(*a, **kw)

    # Mock sem._cond.wait solely to observe deterministic entry into condition wait without sleeps.
    with mock.patch.object(sem._cond, "wait", side_effect=_tracked_wait):
      t3 = threading.Thread(target=lambda: (sem.acquire(50), w3_acquired.set()))
      t3.start()
      self.assertTrue(entering_wait.wait(timeout=10))
      with sem._cond:
        self.assertFalse(w3_acquired.is_set())
        self.assertEqual(sem._current_bytes, 200)
      entering_wait.clear()
      sem.release(30)
      self.assertTrue(entering_wait.wait(timeout=10))
      with sem._cond:
        self.assertFalse(w3_acquired.is_set())
        self.assertEqual(sem._current_bytes, 170)
      sem.release(20)
      t3.join(timeout=10)
      self.assertFalse(t3.is_alive())
      self.assertTrue(w3_acquired.is_set())
      self.assertEqual(sem._current_bytes, 200)
      self.assertEqual(sem._peak_bytes, 200)
    sem.release(200)
    self.assertEqual(sem._current_bytes, 0)

  def test_thread_safe_shard_lru_invalid_max_open(self):
    """Verifies _ThreadSafeShardLRU max_open validation and deterministic lease contention."""
    with self.assertRaisesRegex(ValueError, "max_open must be positive"):
      glm5_ckpt._ThreadSafeShardLRU(0)
    with self.assertRaisesRegex(ValueError, "max_open must be positive"):
      glm5_ckpt._ThreadSafeShardLRU(-1)

    with tempfile.TemporaryDirectory() as tmp_dir:
      shard_paths = []
      for idx in range(3):
        sp = os.path.join(tmp_dir, f"s{idx}.safetensors")
        safetensors.torch.save_file({f"k{idx}": torch.ones((4,), dtype=torch.float32)}, sp)
        shard_paths.append(sp)

      lru = glm5_ckpt._ThreadSafeShardLRU(2)
      leased_two = threading.Barrier(3)
      release_s0 = threading.Event()
      release_s1 = threading.Event()
      t2_waiting = threading.Event()
      t2_done = threading.Event()
      orig_open = safetensors.safe_open

      def _tracked_open(sp, *a, **kw):
        h = orig_open(sp, *a, **kw)
        if sp == shard_paths[0]:
          return _SafeOpenTracker(
              h,
              on_get=lambda *_: (
                  leased_two.wait(timeout=10),
                  release_s0.wait(timeout=10),
              ),
              on_enter=lambda: None,
              on_exit=lambda: None,
          )
        if sp == shard_paths[1]:
          return _SafeOpenTracker(
              h,
              on_get=lambda *_: (
                  leased_two.wait(timeout=10),
                  release_s1.wait(timeout=10),
              ),
              on_enter=lambda: None,
              on_exit=lambda: None,
          )
        return h

      orig_wait = lru._cond.wait

      def _on_wait(*a, **kw):
        t2_waiting.set()
        return orig_wait(*a, **kw)

      # Mock safe_open and lru._cond.wait to deterministically hold 2 active shard leases while a 3rd thread blocks.
      with (
          mock.patch.object(safetensors, "safe_open", side_effect=_tracked_open),
          mock.patch.object(lru._cond, "wait", side_effect=_on_wait),
      ):
        t0 = threading.Thread(target=lambda: lru.load_tensor(shard_paths[0], "k0", "s0.safetensors"))
        t1 = threading.Thread(target=lambda: lru.load_tensor(shard_paths[1], "k1", "s1.safetensors"))
        t0.start()
        t1.start()
        leased_two.wait(timeout=10)

        t2 = threading.Thread(
            target=lambda: (
                lru.load_tensor(shard_paths[2], "k2", "s2.safetensors"),
                t2_done.set(),
            )
        )
        t2.start()
        self.assertTrue(t2_waiting.wait(timeout=10))
        self.assertFalse(t2_done.is_set())

        release_s0.set()
        self.assertTrue(t2_done.wait(timeout=10))
        release_s1.set()
        t0.join(timeout=10)
        t1.join(timeout=10)
        t2.join(timeout=10)
        lru.close_all()

  def test_streaming_parallel_completion_order_determinism(self):
    """Verifies parallel conversion is bit-exact deterministic when later slots and expert chunks finish first."""
    c = dataclasses.replace(
        _make_cfg(11, scan_layers=True, dtype="bfloat16", weight_dtype="bfloat16")[0],
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        n_routed_experts=24,
        vocab_size=512,
        num_nextn_predict_layers=1,
    )
    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as tmp_dir:
      _write_sharded_ckpt(hf_dir, c, num_shards=3)

      k_slot0 = "model.language_model.layers.3.input_layernorm.weight"
      k_slot1 = "model.language_model.layers.7.input_layernorm.weight"
      k_chunk0 = "model.language_model.layers.3.mlp.experts.0.down_proj.weight"
      k_chunk1 = "model.language_model.layers.3.mlp.experts.8.down_proj.weight"
      slot1_done = threading.Event()
      chunk1_done = threading.Event()
      order_log: list[str] = []
      lock = threading.Lock()
      orig_load = glm5_ckpt._ThreadSafeShardLRU.load_tensor

      def _delayed_load(self_lru, shard_path, orig_k, shard_file):
        res = orig_load(self_lru, shard_path, orig_k, shard_file)
        if orig_k == k_slot0:
          if not slot1_done.wait(timeout=15):
            raise RuntimeError(f"Timed out waiting for {k_slot1} before {k_slot0}")
        elif orig_k == k_chunk0:
          if not chunk1_done.wait(timeout=15):
            raise RuntimeError(f"Timed out waiting for {k_chunk1} before {k_chunk0}")
        with lock:
          order_log.append(orig_k)
        if orig_k == k_slot1:
          slot1_done.set()
        elif orig_k == k_chunk1:
          chunk1_done.set()
        return res

      obx1 = os.path.join(tmp_dir, "obx1")
      obx2 = os.path.join(tmp_dir, "obx2")
      # Mock load_tensor to deterministically force slot 1 and expert chunk 1
      # to complete before slot 0 and chunk 0 after shard lease release.
      with mock.patch.object(glm5_ckpt._ThreadSafeShardLRU, "load_tensor", _delayed_load):
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            obx1,
            c,
            ignore_mtp_layer=False,
            slice_to_config_layers=False,
            num_workers=4,
            max_inflight_bytes=64 * 1024 * 1024,
        )
      self.assertLess(order_log.index(k_slot1), order_log.index(k_slot0))
      self.assertLess(order_log.index(k_chunk1), order_log.index(k_chunk0))

      glm5_ckpt.convert_hf_checkpoint_to_orbax(
          hf_dir,
          obx2,
          c,
          ignore_mtp_layer=False,
          slice_to_config_layers=False,
          num_workers=1,
          max_inflight_bytes=64 * 1024 * 1024,
      )

      r1 = glm5_ckpt.ocp.StandardCheckpointer().restore(obx1)
      r2 = glm5_ckpt.ocp.StandardCheckpointer().restore(obx2)
      jax.tree.map(np.testing.assert_array_equal, r1, r2)

      m1 = glm5_ckpt.ocp.StandardCheckpointer().metadata(obx1).item_metadata.tree
      m2 = glm5_ckpt.ocp.StandardCheckpointer().metadata(obx2).item_metadata.tree

      def _check_meta(a, b):
        self.assertEqual(a, b)
        self.assertEqual(a.storage.chunk_shape, b.storage.chunk_shape)

      jax.tree.map(_check_meta, m1, m2)

  def test_thread_safe_shard_lru_exception_propagation_and_cleanup(self):
    """Verifies that _ThreadSafeShardLRU propagates original exceptions under concurrent load and conversion cleans up."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      bad_shard = os.path.join(tmp_dir, "bad.safetensors")
      with open(bad_shard, "wb") as f:
        f.write(b"short")

      lru = glm5_ckpt._ThreadSafeShardLRU(2)
      exceptions = []

      def _worker():
        try:
          lru.load_tensor(bad_shard, "k", "bad.safetensors")
        except Exception as exc:  # pylint: disable=broad-except
          exceptions.append(exc)

      threads = [threading.Thread(target=_worker) for _ in range(4)]
      for t in threads:
        t.start()
      for t in threads:
        t.join(timeout=10)
        self.assertFalse(t.is_alive())

      self.assertEqual(len(exceptions), 4)
      for exc in exceptions:
        self.assertNotIsInstance(exc, RuntimeError)
        self.assertTrue("HeaderTooSmall" in str(exc) or "Error while deserializing header" in str(exc))
      lru.close_all()

      c, _ = _make_cfg(num_hidden_layers=3, scan_layers=False)
      hf_dir = os.path.join(tmp_dir, "hf")
      os.makedirs(hf_dir, exist_ok=True)
      _write_sharded_ckpt(hf_dir, c, num_shards=2)
      shards = [f for f in os.listdir(hf_dir) if f.endswith(".safetensors")]
      with open(os.path.join(hf_dir, shards[0]), "wb") as f:
        f.write(b"corrupt")
      obx_dir = os.path.join(tmp_dir, "obx_fail")
      with self.assertRaises(Exception) as cm:
        glm5_ckpt.convert_hf_checkpoint_to_orbax(
            hf_dir,
            obx_dir,
            c,
            ignore_mtp_layer=True,
            slice_to_config_layers=False,
            num_workers=2,
            max_inflight_bytes=64 * 1024 * 1024,
        )
      self.assertNotIsInstance(cm.exception, RuntimeError)
      self.assertTrue("HeaderTooSmall" in str(cm.exception) or "Error while deserializing header" in str(cm.exception))
      self.assertFalse(os.path.exists(obx_dir))


if __name__ == "__main__":
  absltest.main()
