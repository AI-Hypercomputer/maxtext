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
import importlib
import importlib.util
import sys
import types
import unittest

from flax import nnx
import jax
from jax import test_util as jtu
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params
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

# MaxText param prefix of the full-attention (MLA) layer in `_MAPPING_CONFIG`.
_LAYER_PREFIX = "params-decoder-layers_3-self_attention"

# Flat dict consumed by PARAM_MAPPING / HOOK_FNS["kimi-k3"]; layer 3 is the full-attention (MLA) layer.
_MAPPING_CONFIG = {
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

# MaxText MLA submodule -> parameter name, for every weight of the gated MLA layer.
_MLA_PARAMS = {
    "wq_a": "kernel",
    "q_norm": "scale",
    "wq_b": "kernel",
    "wkv_a": "kernel",
    "kv_norm": "scale",
    "wkv_b": "kernel",
    "g_proj": "kernel",
    "out": "kernel",
}


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


def _make_mla(cfg, mesh, *, use_output_gate, is_nope_layer, seed=0, max_position_embeddings=1048576):
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
      max_position_embeddings=max_position_embeddings,
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
    self.assertIsNone(getattr(mla, "g_proj", None))

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

  def test_nope_layer_ignores_yarn_mscale(self):
    """YaRN mscale only rescales rotary layers; NoPE layers keep qk_head_dim**-0.5 (HF `KimiMLAAttention`)."""
    base_scale = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5
    nope = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=True, max_position_embeddings=4 * 1048576)
    rope = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=False, max_position_embeddings=4 * 1048576)
    _run(nope, self.x, _arange_positions())  # The query projection recomputes the scale.
    _run(rope, self.x, _arange_positions())
    self.assertAlmostEqual(nope.softmax_scale, base_scale, places=7)
    self.assertGreater(rope.softmax_scale, base_scale * 1.5)


# --- Hermetic NumPy oracle of the HF `KimiMLAAttention` forward (gated, NoPE). ---


def _random_hf_state_dict(seed=0):
  """Random HF-layout (`nn.Linear` [out, in]) weights for a gated NoPE `KimiMLAAttention`."""
  rng = np.random.default_rng(seed)
  qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

  def linear(out_dim, in_dim):
    return (rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)).astype(np.float32)

  return {
      "q_a_proj.weight": linear(Q_LORA_RANK, HIDDEN),
      "q_a_layernorm.weight": rng.uniform(0.5, 1.5, Q_LORA_RANK).astype(np.float32),
      "q_b_proj.weight": linear(NUM_HEADS * qk_head_dim, Q_LORA_RANK),
      "kv_a_proj_with_mqa.weight": linear(KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN),
      "kv_a_layernorm.weight": rng.uniform(0.5, 1.5, KV_LORA_RANK).astype(np.float32),
      "kv_b_proj.weight": linear(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
      "g_proj.weight": linear(NUM_HEADS * V_HEAD_DIM, HIDDEN),
      "o_proj.weight": linear(HIDDEN, NUM_HEADS * V_HEAD_DIM),
  }


def _oracle_gated_nope_mla(sd, x, xp=np):
  """Mirrors `KimiMLAAttention.forward` (eager, causal, no RoPE) in HF weight layout.

  `xp` is `numpy` for the forward oracle or `jax.numpy` to differentiate the same math.
  """
  b, t, _ = x.shape
  qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

  def rms_norm(v, w):
    return w * (v * (1.0 / xp.sqrt(xp.mean(v * v, axis=-1, keepdims=True) + RMS_NORM_EPS)))

  q = rms_norm(x @ sd["q_a_proj.weight"].T, sd["q_a_layernorm.weight"]) @ sd["q_b_proj.weight"].T
  q = q.reshape(b, t, NUM_HEADS, qk_head_dim)
  compressed_kv = x @ sd["kv_a_proj_with_mqa.weight"].T
  k_pass, k_rot = compressed_kv[..., :KV_LORA_RANK], compressed_kv[..., KV_LORA_RANK:]
  kv = rms_norm(k_pass, sd["kv_a_layernorm.weight"]) @ sd["kv_b_proj.weight"].T
  kv = kv.reshape(b, t, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
  k_nope, value = kv[..., :QK_NOPE_HEAD_DIM], kv[..., QK_NOPE_HEAD_DIM:]
  k_rot = xp.broadcast_to(k_rot[:, :, None, :], (b, t, NUM_HEADS, QK_ROPE_HEAD_DIM))
  key = xp.concatenate([k_nope, k_rot], axis=-1)

  scores = xp.einsum("bqhd,bkhd->bhqk", q, key) * qk_head_dim**-0.5
  scores = xp.where(np.tril(np.ones((t, t), dtype=bool)), scores, -1e30)
  scores = scores - xp.max(scores, axis=-1, keepdims=True)
  probs = xp.exp(scores)
  probs = probs / xp.sum(probs, axis=-1, keepdims=True)
  attn = xp.einsum("bhqk,bkhd->bqhd", probs, value).reshape(b, t, NUM_HEADS * V_HEAD_DIM)

  gate = 1.0 / (1.0 + xp.exp(-(x @ sd["g_proj.weight"].T)))
  return (attn * gate) @ sd["o_proj.weight"].T


def _to_maxtext(hf_tensors):
  """HF-layout tensors -> MaxText params via the production PARAM_MAPPING / HOOK_FNS["kimi-k3"]."""
  return convert_pytorch_module_to_maxtext_params(hf_tensors, _MAPPING_CONFIG, None, prefix_filter=_LAYER_PREFIX)


class MlaGateNopeOracleTest(unittest.TestCase):
  """CI-runnable parity of gated NoPE MLA (forward, VJP and `g_proj` layout) against a NumPy HF oracle."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.cfg), self.cfg.mesh_axes)
    self.hf_sd = _random_hf_state_dict()
    self.x = np.random.default_rng(1).standard_normal((BATCH, SEQ_LEN, HIDDEN)).astype(np.float32)
    self.mla = _make_mla(self.cfg, self.mesh, use_output_gate=True, is_nope_layer=True)
    params = _to_maxtext(self.hf_sd)
    for module, param in _MLA_PARAMS.items():
      getattr(getattr(self.mla, module), param)[...] = jnp.asarray(params[f"{_LAYER_PREFIX}-{module}-{param}"])

  def test_g_proj_hook_roundtrip(self):
    """HF `g_proj.weight` [H*V, D] <-> MaxText kernel [D, H, V], in both hook directions."""
    key = f"{_LAYER_PREFIX}-g_proj-kernel"
    to_mt = HOOK_FNS["kimi-k3"](_MAPPING_CONFIG, None, scan_layers=False, saving_to_hf=False)[key]
    to_hf = HOOK_FNS["kimi-k3"](_MAPPING_CONFIG, None, scan_layers=False, saving_to_hf=True)[key]
    hf_w = self.hf_sd["g_proj.weight"]

    mt_w = to_mt(hf_w, (HIDDEN, NUM_HEADS, V_HEAD_DIM))
    self.assertEqual(mt_w.shape, (HIDDEN, NUM_HEADS, V_HEAD_DIM))
    np.testing.assert_array_equal(to_mt(hf_w), mt_w)  # Built-in target shape agrees.
    # Element-wise layout: MaxText [d, h, v] == HF [h * V + v, d].
    np.testing.assert_array_equal(mt_w, hf_w.reshape(NUM_HEADS, V_HEAD_DIM, HIDDEN).transpose(2, 0, 1))
    # Same projection: x @ W.T (HF) == einsum(x, kernel) (MaxText DenseGeneral).
    np.testing.assert_allclose(
        np.einsum("btd,dhv->bthv", self.x, mt_w).reshape(BATCH, SEQ_LEN, -1), self.x @ hf_w.T, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(to_hf(mt_w, hf_w.shape), hf_w)

  def test_forward_matches_oracle(self):
    out = _run(self.mla, jnp.asarray(self.x), _arange_positions())
    np.testing.assert_allclose(out, _oracle_gated_nope_mla(self.hf_sd, self.x), rtol=1e-4, atol=1e-5)

  def test_vjp_matches_oracle(self):
    """Input and weight gradients match the oracle's, with weight gradients mapped through the same hooks."""
    cotangent = jnp.asarray(np.random.default_rng(2).standard_normal((BATCH, SEQ_LEN, HIDDEN)), dtype=jnp.float32)
    positions = _arange_positions()
    segment_ids = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)
    graphdef, params, rest = nnx.split(self.mla, nnx.Param, ...)

    def maxtext_loss(params, x):
      out, _ = nnx.merge(graphdef, params, rest)(
          inputs_q=x, inputs_kv=x, inputs_positions=positions, decoder_segment_ids=segment_ids, model_mode="train"
      )
      return jnp.sum(out * cotangent)

    def oracle_loss(sd, x):
      return jnp.sum(_oracle_gated_nope_mla(sd, x, xp=jnp) * cotangent)

    mt_param_grads, mt_x_grad = jax.grad(maxtext_loss, argnums=(0, 1))(params, jnp.asarray(self.x))
    hf_sd = {k: jnp.asarray(v) for k, v in self.hf_sd.items()}
    hf_param_grads, hf_x_grad = jax.grad(oracle_loss, argnums=(0, 1))(hf_sd, jnp.asarray(self.x))

    np.testing.assert_allclose(mt_x_grad, hf_x_grad, rtol=1e-4, atol=1e-5)
    expected = _to_maxtext({k: np.asarray(v) for k, v in hf_param_grads.items()})
    for module, param in _MLA_PARAMS.items():
      with self.subTest(param=f"{module}.{param}"):
        np.testing.assert_allclose(
            mt_param_grads[module][param][...], expected[f"{_LAYER_PREFIX}-{module}-{param}"], rtol=1e-4, atol=1e-5
        )

  def test_gate_gradients_numerically(self):
    """Autodiff of the gated output matches finite differences (w.r.t. the input)."""
    positions = _arange_positions()
    segment_ids = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)

    graphdef, state = nnx.split(self.mla)

    def f(x):
      out, _ = nnx.merge(graphdef, state)(
          inputs_q=x, inputs_kv=x, inputs_positions=positions, decoder_segment_ids=segment_ids, model_mode="train"
      )
      return out

    # Loose tolerance: float32 central differences through softmax/RMSNorm are only good to ~2%; the exact
    # gradient comparison is `test_vjp_matches_oracle`.
    jtu.check_grads(f, (jnp.asarray(self.x),), order=1, modes=("rev",), atol=5e-2, rtol=5e-2)


_HF_PACKAGE = "_kimi_k3_hf_ref"


@contextlib.contextmanager
def _hf_reference_modules():
  """Imports the HF reference modules, restoring sys.path / sys.modules afterwards.

  Upstream `modeling_kimi_linear.py` uses a package-relative import (`from .configuration_kimi_k3
  import ...`, standard for `trust_remote_code`), so the checkout is imported as a synthetic package.
  The directory is also put on `sys.path` so checkouts edited to use a top-level import still load.
  """
  saved_modules = set(sys.modules)
  package = types.ModuleType(_HF_PACKAGE)
  package.__path__ = [KIMI_K3_REFERENCE_DIR]
  sys.modules[_HF_PACKAGE] = package
  sys.path.insert(0, KIMI_K3_REFERENCE_DIR)
  try:
    modeling = importlib.import_module(f"{_HF_PACKAGE}.modeling_kimi_linear")
    yield modeling.KimiLinearConfig, modeling.KimiMLAAttention
  finally:
    sys.path.remove(KIMI_K3_REFERENCE_DIR)
    for name in list(sys.modules):
      top_level = name.split(".")[0]
      if top_level in (_HF_PACKAGE, "configuration_kimi_k3", "modeling_kimi_linear") and name not in saved_modules:
        sys.modules.pop(name, None)


@unittest.skipUnless(_HF_DEPS_AVAILABLE, "torch, transformers and einops are required for HF parity")
@requires_kimi_k3_reference
class MlaGateNopeVsReferenceTest(unittest.TestCase):
  """Parity of gated NoPE MLA against the HF `KimiMLAAttention` reference."""

  layer_prefix = _LAYER_PREFIX

  def test_gated_nope_mla_parity(self):
    import torch  # pylint: disable=import-outside-toplevel

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

    params = convert_pytorch_module_to_maxtext_params(
        pt_mla.state_dict(), _MAPPING_CONFIG, None, prefix_filter=self.layer_prefix
    )
    expected = _MLA_PARAMS
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
