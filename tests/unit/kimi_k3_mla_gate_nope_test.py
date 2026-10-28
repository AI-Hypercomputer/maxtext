#  Copyright 2023-2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the MLA sigmoid output gate and NoPE (skip-RoPE) support used by Kimi-K3.

The hermetic tests run everywhere. The HF parity test additionally needs torch,
transformers, einops and the untracked Kimi-K3 HF reference checkout, and skips
cleanly when any of them is missing.
"""

import contextlib
import importlib.util
import sys
import unittest

from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, requires_kimi_k3_reference
from tests.utils.test_helpers import get_test_config_path

_HF_DEPS_AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "einops"))

# Small dims: parity does not need production sizes.
BATCH = 2
SEQ_LEN = 8
HIDDEN = 256
NUM_HEADS = 4
QK_NOPE_HEAD_DIM = 32
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 32
Q_LORA_RANK = 64
KV_LORA_RANK = 32
RMS_NORM_EPS = 1e-5

# Shared submodules of the MLA layer (everything except the optional g_proj).
_SHARED_SUBMODULES = ("wq_a", "q_norm", "wq_b", "wkv_a", "kv_norm", "wkv_b", "out")


def _make_config():
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="kimi_k3_mla_gate_nope_test",
      model_name="default",
      attention="dot_product",
      attention_type="mla",
      base_emb_dim=HIDDEN,
      emb_dim=HIDDEN,
      num_query_heads=NUM_HEADS,
      num_kv_heads=NUM_HEADS,
      head_dim=V_HEAD_DIM,
      q_lora_rank=Q_LORA_RANK,
      kv_lora_rank=KV_LORA_RANK,
      qk_nope_head_dim=QK_NOPE_HEAD_DIM,
      qk_rope_head_dim=QK_ROPE_HEAD_DIM,
      v_head_dim=V_HEAD_DIM,
      mla_naive_kvcache=True,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      normalization_layer_epsilon=RMS_NORM_EPS,
      max_target_length=SEQ_LEN,
      max_prefill_predict_length=SEQ_LEN,
      per_device_batch_size=BATCH,
      rope_max_timescale=500000.0,
      max_position_embeddings=1048576,
      original_max_position_embeddings=1048576,
      enable_checkpointing=False,
  )


def _make_mla(cfg, mesh, *, use_output_gate, is_nope_layer, seed=0):
  return attention_mla.MLA(
      config=cfg,
      num_query_heads=NUM_HEADS,
      num_kv_heads=NUM_HEADS,
      head_dim=V_HEAD_DIM,
      inputs_q_shape=(BATCH, SEQ_LEN, HIDDEN),
      inputs_kv_shape=(BATCH, SEQ_LEN, HIDDEN),
      max_target_length=SEQ_LEN,
      max_prefill_predict_length=SEQ_LEN,
      mesh=mesh,
      attention_kernel="dot_product",
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      attention_type="mla",
      is_nope_layer=is_nope_layer,
      mla_use_output_gate=use_output_gate,
      q_lora_rank=Q_LORA_RANK,
      kv_lora_rank=KV_LORA_RANK,
      qk_nope_head_dim=QK_NOPE_HEAD_DIM,
      qk_rope_head_dim=QK_ROPE_HEAD_DIM,
      v_head_dim=V_HEAD_DIM,
      max_position_embeddings=1048576,
      original_max_position_embeddings=1048576,
      mscale=1.0,
      rope_factor=40.0,
      rngs=nnx.Rngs(params=seed, dropout=seed),
  )


def _copy_shared_weights(src, dst):
  for name in _SHARED_SUBMODULES:
    nnx.update(getattr(dst, name), nnx.state(getattr(src, name)))


def _run(mla, x, positions):
  segment_ids = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)
  out, _ = mla(
      inputs_q=x,
      inputs_kv=x,
      inputs_positions=positions,
      decoder_segment_ids=segment_ids,
      model_mode="train",
  )
  return np.asarray(out)


def _arange_positions():
  return jnp.broadcast_to(jnp.arange(SEQ_LEN, dtype=jnp.int32), (BATCH, SEQ_LEN))


class MlaGateNopeTest(unittest.TestCase):
  """Hermetic numerical tests for the MLA output gate and NoPE path."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.cfg), self.cfg.mesh_axes)
    self.x = jnp.asarray(np.random.default_rng(0).standard_normal((BATCH, SEQ_LEN, HIDDEN)), dtype=jnp.float32)

  def test_gate_disabled_by_default(self):
    mla = _make_mla(self.cfg, self.mesh, use_output_gate=False, is_nope_layer=False)
    self.assertIsNone(mla.g_proj)

  def test_zero_gate_weights_halve_output(self):
    """With g_proj == 0, sigmoid(0) = 0.5 and o_proj is linear, so the output is exactly halved."""
    ungated = _make_mla(self.cfg, self.mesh, use_output_gate=False, is_nope_layer=True)
    gated = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=True)
    _copy_shared_weights(ungated, gated)
    gated.g_proj.kernel[...] = jnp.zeros_like(gated.g_proj.kernel[...])
    self.assertEqual(gated.g_proj.kernel[...].shape, (HIDDEN, NUM_HEADS, V_HEAD_DIM))

    positions = _arange_positions()
    np.testing.assert_allclose(
        _run(gated, self.x, positions), 0.5 * _run(ungated, self.x, positions), rtol=1e-5, atol=1e-6
    )

  def test_nope_layer_skips_rope_at_nonzero_positions(self):
    """NoPE output is position-independent and equals the RoPE output at position 0 (where RoPE is identity)."""
    rope = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=False)
    nope = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=True)
    _copy_shared_weights(rope, nope)
    nnx.update(nope.g_proj, nnx.state(rope.g_proj))

    positions = _arange_positions()
    zeros = jnp.zeros_like(positions)
    nope_out = _run(nope, self.x, positions)

    np.testing.assert_allclose(nope_out, _run(nope, self.x, positions + 100), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(nope_out, _run(rope, self.x, zeros), rtol=1e-5, atol=1e-6)
    # Sanity check: with RoPE enabled, non-zero positions do change the output.
    self.assertGreater(np.max(np.abs(_run(rope, self.x, positions) - nope_out)), 1e-3)


@contextlib.contextmanager
def _hf_reference_modules():
  """Imports the HF reference modules, restoring sys.path / sys.modules afterwards."""
  saved_modules = set(sys.modules)
  sys.path.insert(0, KIMI_K3_REFERENCE_DIR)
  try:
    from configuration_kimi_k3 import KimiLinearConfig  # pylint: disable=import-outside-toplevel,import-error
    from modeling_kimi_linear import KimiMLAAttention  # pylint: disable=import-outside-toplevel,import-error

    yield KimiLinearConfig, KimiMLAAttention
  finally:
    sys.path.remove(KIMI_K3_REFERENCE_DIR)
    for name in ("configuration_kimi_k3", "modeling_kimi_linear"):
      if name not in saved_modules:
        sys.modules.pop(name, None)


@unittest.skipUnless(_HF_DEPS_AVAILABLE, "torch, transformers and einops are required for HF parity")
@requires_kimi_k3_reference
class MlaGateNopeVsReferenceTest(unittest.TestCase):
  """Parity of gated NoPE MLA against the HF `KimiMLAAttention` reference."""

  layer_prefix = "params-decoder-layers_3-self_attention"

  def test_gated_nope_mla_parity(self):
    import torch  # pylint: disable=import-outside-toplevel
    from tests.utils.kimi_k3_conversion_utils import (  # pylint: disable=import-outside-toplevel
        convert_pytorch_module_to_maxtext_params,
    )

    with _hf_reference_modules() as (kimi_linear_config, kimi_mla_attention):
      pt_config = kimi_linear_config(
          hidden_size=HIDDEN,
          num_attention_heads=NUM_HEADS,
          num_key_value_heads=NUM_HEADS,
          q_lora_rank=Q_LORA_RANK,
          kv_lora_rank=KV_LORA_RANK,
          qk_nope_head_dim=QK_NOPE_HEAD_DIM,
          qk_rope_head_dim=QK_ROPE_HEAD_DIM,
          v_head_dim=V_HEAD_DIM,
          mla_use_nope=True,
          mla_use_output_gate=True,
          rms_norm_eps=RMS_NORM_EPS,
      )
      pt_config._attn_implementation = "eager"  # pylint: disable=protected-access
      torch.manual_seed(42)
      pt_mla = kimi_mla_attention(pt_config, layer_idx=3).eval()
      # Non-trivial RMSNorm scales so the norm mapping is exercised too.
      with torch.no_grad():
        pt_mla.q_a_layernorm.weight.uniform_(0.5, 1.5)
        pt_mla.kv_a_layernorm.weight.uniform_(0.5, 1.5)

      pt_input = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float32)
      causal_mask = torch.triu(torch.full((BATCH, 1, SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1)
      with torch.no_grad():
        pt_out = pt_mla(pt_input, attention_mask=causal_mask)
      pt_out = (pt_out[0] if isinstance(pt_out, tuple) else pt_out).numpy()

    # Flat dict consumed by PARAM_MAPPING / HOOK_FNS["kimi-k3"]; layer 3 is the full-attention (MLA) layer.
    mapping_config = {
        "num_hidden_layers": 4,
        "first_k_dense_replace": 4,
        "full_attn_layers": [3],
        "hidden_size": HIDDEN,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "rms_norm_eps": RMS_NORM_EPS,
    }
    params = convert_pytorch_module_to_maxtext_params(
        pt_mla.state_dict(), mapping_config, None, prefix_filter=self.layer_prefix
    )
    expected = {
        "wq_a": "kernel",
        "q_norm": "scale",
        "wq_b": "kernel",
        "wkv_a": "kernel",
        "kv_norm": "scale",
        "wkv_b": "kernel",
        "g_proj": "kernel",
        "out": "kernel",
    }
    missing = [
        f"{self.layer_prefix}-{m}-{p}" for m, p in expected.items() if f"{self.layer_prefix}-{m}-{p}" not in params
    ]
    self.assertFalse(missing, f"Unconverted params: {missing}")

    cfg = _make_config()
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    nope = _make_mla(cfg, mesh, use_output_gate=True, is_nope_layer=True)
    rope = _make_mla(cfg, mesh, use_output_gate=True, is_nope_layer=False)
    for mla in (nope, rope):
      for module, param in expected.items():
        getattr(getattr(mla, module), param)[...] = jnp.asarray(params[f"{self.layer_prefix}-{module}-{param}"])

    x = jnp.asarray(pt_input.numpy())
    positions = _arange_positions()
    np.testing.assert_allclose(_run(nope, x, positions), pt_out, rtol=1e-4, atol=1e-4)
    # Negative check: applying RoPE at non-zero positions must diverge from the NoPE reference.
    self.assertGreater(np.max(np.abs(_run(rope, x, positions) - pt_out)), 1e-3)


if __name__ == "__main__":
  unittest.main()
