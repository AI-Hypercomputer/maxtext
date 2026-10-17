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

"""Nemotron3 (Nemotron-H) MaxText modules vs transformers' NemotronH on tiny random fp32 models.

The reference is `transformers.models.nemotron_h` (>= 5.17: its chunked SSD scan sums the inter-chunk
recurrence over the source chunk, matching the CUDA kernels, and it clamps dt to
`(time_step_min, inf)`). It differs from the Nemotron-3-Nano-Omni checkpoint's bundled modeling file
only where fp32 cannot tell them apart (the bundled RMSNorm casts to the input dtype before scaling,
which is a no-op in fp32) or where the bundled CPU path is wrong for sequences longer than one chunk
(the inter-chunk scan), so these checks use the library classes directly. MaxText's mamba_time_step_limit
is set to the reference's clamp. HF has no packed-document reset, so packed rows are compared against
each document run on its own.
"""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.models.nemotron_h import modeling_nemotron_h as nemotron_h
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils
from tests.utils.test_helpers import get_test_config_path

PATTERN = "ME*"
BATCH, SEQ, CHUNK = 4, 20, 8  # SEQ crosses two chunk boundaries (8 and 16).
HIDDEN, VOCAB = 32, 64
MAMBA_HEADS, MAMBA_HEAD_DIM, N_GROUPS, STATE, CONV_K = 4, 8, 2, 8, 4
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
EXPERTS, TOP_K, MOE_DIM, SHARED_EXPERTS = 8, 2, 16, 2
TIME_STEP_MIN = 0.01
EPS = 1e-5
# Both sides run fp32 on CPU (matmul precision highest).
RTOL = ATOL = 1e-5


def hf_config():
  return NemotronHConfig(
      vocab_size=VOCAB,
      hidden_size=HIDDEN,
      hybrid_override_pattern=PATTERN,
      num_attention_heads=Q_HEADS,
      num_key_value_heads=KV_HEADS,
      head_dim=HEAD_DIM,
      mamba_num_heads=MAMBA_HEADS,
      mamba_head_dim=MAMBA_HEAD_DIM,
      n_groups=N_GROUPS,
      ssm_state_size=STATE,
      conv_kernel=CONV_K,
      chunk_size=CHUNK,
      time_step_min=TIME_STEP_MIN,
      use_mamba_kernels=False,
      n_routed_experts=EXPERTS,
      num_experts_per_tok=TOP_K,
      moe_intermediate_size=MOE_DIM,
      moe_shared_expert_intermediate_size=SHARED_EXPERTS * MOE_DIM,
      routed_scaling_factor=2.5,
      n_group=1,
      topk_group=1,
      norm_topk_prob=True,
      layer_norm_epsilon=EPS,
      tie_word_embeddings=False,
      attn_implementation="eager",
  )


def maxtext_config(**overrides):
  """Tiny Nemotron3 config; override_model_config lets every key below win over the model yml."""
  kwargs = {
      "run_name": "nemotron3_layers_test",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "nemotron3-nano-omni-30b-a3b",
      "override_model_config": True,
      "decoder_block": "nemotron3",
      "scan_layers": False,
      "base_emb_dim": HIDDEN,
      "base_num_decoder_layers": len(PATTERN),
      "hybrid_override_pattern": PATTERN,
      "vocab_size": VOCAB,
      "normalization_layer_epsilon": EPS,
      "logits_via_embedding": False,
      "enable_dropout": False,
      "base_num_query_heads": Q_HEADS,
      "base_num_kv_heads": KV_HEADS,
      "head_dim": HEAD_DIM,
      "attention_bias": False,
      "use_qk_norm": False,
      "float32_qk_product": False,
      "float32_logits": True,
      "mamba_num_heads": MAMBA_HEADS,
      "mamba_head_dim": MAMBA_HEAD_DIM,
      "mamba_ssm_state_size": STATE,
      "mamba_n_groups": N_GROUPS,
      "mamba_conv_kernel": CONV_K,
      "mamba_conv_bias": True,
      "mamba_chunk_size": CHUNK,
      "mamba_time_step_limit": [TIME_STEP_MIN, float("inf")],
      "num_experts": EXPERTS,
      "num_experts_per_tok": TOP_K,
      "base_moe_mlp_dim": MOE_DIM,
      "base_mlp_dim": MOE_DIM,
      "shared_experts": SHARED_EXPERTS,
      "mlp_activations": ["relu2"],
      "mlp_bias": False,
      "routed_score_func": "sigmoid",
      "routed_bias": True,
      "routed_scaling_factor": 2.5,
      "norm_topk_prob": True,
      "n_routing_groups": 1,
      "topk_routing_group": 1,
      "float32_gate_logits": True,
      "float32_weight_sum": True,
      "routed_bias_update_rate": 0.0,
      "load_balance_loss_weight": 0.0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "attention": "dot_product",
      "sparse_matmul": True,
      "megablox": False,
      # Global batch BATCH on any CPU device count (CI forces 4 host devices).
      "per_device_batch_size": BATCH / jax.device_count(),
      "max_target_length": SEQ,
      "max_prefill_predict_length": SEQ,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def hf_model(seed=0):
  """NemotronHForCausalLM with random weights, non-trivial norms, spread router biases and dt below the clamp."""
  torch.manual_seed(seed)
  model = nemotron_h.NemotronHForCausalLM(hf_config()).eval()
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("norm.weight") or name.endswith("norm_f.weight"):
        p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=gen))
      elif name.endswith("A_log"):
        p.copy_(torch.log(torch.linspace(0.5, 4.0, p.numel())))
      elif name.endswith("dt_bias"):
        # softplus(-6) ~ 2.5e-3 < TIME_STEP_MIN: the most negative heads hit the dt clamp.
        p.copy_(torch.linspace(-6.0, 1.0, p.numel()))
      elif name.endswith(".D"):
        p.copy_(torch.randn(p.shape, generator=gen))
      else:
        fan_in = p.shape[-1] if p.ndim > 1 else 1
        p.copy_(torch.randn(p.shape, generator=gen) / np.sqrt(fan_in))
    for layer in model.model.layers:
      if layer.block_type == "moe":
        # Spread over the sigmoid range so the bias changes the top-k without ties.
        bias = torch.linspace(-0.4, 0.4, EXPERTS)[torch.randperm(EXPERTS, generator=gen)]
        layer.mixer.gate.e_score_correction_bias.copy_(bias)
  return model


def _np(t):
  return t.detach().to(torch.float32).numpy()


def maxtext_model(hf, model_mode=MODEL_MODE_TRAIN, **overrides):
  """MaxText Transformer with the HF weights mapped explicitly (HF Linear [out, in] -> MaxText [in, out])."""
  cfg = maxtext_config(**overrides)
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, model_mode=model_mode)
  model = model_creation_utils.create_model(cfg, mesh, model_mode=model_mode, rngs=rngs)
  sd = {k: _np(v) for k, v in hf.state_dict().items()}

  def put(var, value):
    assert tuple(var.get_value().shape) == value.shape, (var.get_value().shape, value.shape)
    var.set_value(jnp.asarray(value, var.get_value().dtype))

  put(model.token_embedder.embedding, sd["model.embeddings.weight"])
  put(model.decoder.decoder_norm.scale, sd["model.norm_f.weight"])
  put(model.decoder.logits_dense.kernel, sd["lm_head.weight"].T)
  for i, block in enumerate(PATTERN):
    layer, p = getattr(model.decoder, f"layers_{i}"), f"model.layers.{i}."
    put(layer.norm.scale, sd[p + "norm.weight"])
    mixer = layer.mixer
    if block == "M":
      put(mixer.in_proj.kernel, sd[p + "mixer.in_proj.weight"].T)
      put(mixer.conv_weight, sd[p + "mixer.conv1d.weight"][:, 0, :].T)
      put(mixer.conv_bias, sd[p + "mixer.conv1d.bias"])
      for name in ("dt_bias", "A_log", "D"):
        put(getattr(mixer, name), sd[p + f"mixer.{name}"])
      put(mixer.norm.scale, sd[p + "mixer.norm.weight"])
      put(mixer.out_proj.kernel, sd[p + "mixer.out_proj.weight"].T)
    elif block == "*":
      put(mixer.query.kernel, sd[p + "mixer.q_proj.weight"].T.reshape(HIDDEN, Q_HEADS, HEAD_DIM))
      put(mixer.key.kernel, sd[p + "mixer.k_proj.weight"].T.reshape(HIDDEN, KV_HEADS, HEAD_DIM))
      put(mixer.value.kernel, sd[p + "mixer.v_proj.weight"].T.reshape(HIDDEN, KV_HEADS, HEAD_DIM))
      put(mixer.out.kernel, sd[p + "mixer.o_proj.weight"].T.reshape(Q_HEADS, HEAD_DIM, HIDDEN))
    else:
      routed = mixer.routed_moe
      put(routed.gate.kernel, sd[p + "mixer.gate.weight"].T)
      put(routed.gate.bias, sd[p + "mixer.gate.e_score_correction_bias"])
      put(routed.wi_0, sd[p + "mixer.experts.up_proj"].transpose(0, 2, 1))
      put(routed.wo, sd[p + "mixer.experts.down_proj"].transpose(0, 2, 1))
      put(mixer.shared_experts.wi.kernel, sd[p + "mixer.shared_experts.up_proj.weight"].T)
      put(mixer.shared_experts.wo.kernel, sd[p + "mixer.shared_experts.down_proj.weight"].T)
  return model


def _inputs(seed=0):
  rng = np.random.default_rng(seed)
  ids = rng.integers(0, VOCAB, size=(BATCH, SEQ)).astype(np.int32)
  x = rng.normal(size=(BATCH, SEQ, HIDDEN)).astype(np.float32)
  return ids, x


def _positions(n):
  return np.broadcast_to(np.arange(n, dtype=np.int32), (BATCH, n))


def _logits(model, ids, **kwargs):
  n = ids.shape[1]
  seg = np.ones((BATCH, n), np.int32)
  return model(ids, _positions(n), decoder_segment_ids=seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN, **kwargs)


class _RecordRouting:
  """Records (and optionally replaces) the HF router's top-k for one MoE block."""

  def __init__(self, gate, forced=None):
    self.gate, self.forced, self.indices = gate, forced, None
    self.original = gate.forward

  def __enter__(self):
    def forward(hidden_states):
      logits, weights, indices = self.original(hidden_states)
      if self.forced is not None:
        indices = torch.as_tensor(self.forced.reshape(-1, TOP_K), dtype=torch.long)
        scores = logits.sigmoid().gather(1, indices)
        weights = scores / (scores.sum(-1, keepdim=True) + 1e-20) * self.gate.routed_scaling_factor
      self.indices = indices.numpy().reshape(BATCH, -1, TOP_K)
      return logits, weights, indices

    self.gate.forward = forward
    return self

  def __exit__(self, *exc):
    self.gate.forward = self.original


class Nemotron3LayersTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch.set_float32_matmul_precision("highest")
    cls.hf = hf_model()
    cls.mt = maxtext_model(cls.hf)
    cls.ids, cls.x = _inputs()
    cls.moe_gate = cls.hf.model.layers[PATTERN.index("E")].mixer.gate

  def assert_close(self, actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), rtol=rtol, atol=atol)

  def test_gated_rmsnorm(self):
    hf_norm = self.hf.model.layers[0].mixer.norm
    self.assertGreater(float(hf_norm.weight.std()), 0.1)
    y = self.x
    z = np.random.default_rng(1).normal(size=y.shape).astype(np.float32)
    expected = _np(hf_norm(torch.from_numpy(y), torch.from_numpy(z)))
    self.assert_close(self.mt.decoder.layers_0.mixer.norm(jnp.asarray(y), jnp.asarray(z)), expected)
    self.assert_close(self.mt.decoder.layers_0.mixer.norm(jnp.asarray(y)), _np(hf_norm(torch.from_numpy(y))))

  def test_gated_rmsnorm_output_dtype(self):
    norm = self.mt.decoder.layers_0.mixer.norm
    out = norm(jnp.asarray(self.x, jnp.bfloat16), jnp.asarray(self.x, jnp.bfloat16))
    self.assertEqual(norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_mamba_mixer_multi_chunk(self):
    hf_mixer = self.hf.model.layers[0].mixer
    with torch.no_grad():
      dt = torch.nn.functional.softplus(hf_mixer.in_proj(torch.from_numpy(self.x))[..., -MAMBA_HEADS:] + hf_mixer.dt_bias)
    self.assertGreater(int((dt < TIME_STEP_MIN).sum()), 0, "dt clamp must be exercised")
    expected = _np(hf_mixer(torch.from_numpy(self.x)))
    got = self.mt.decoder.layers_0.mixer(jnp.asarray(self.x), decoder_segment_ids=None, model_mode=MODEL_MODE_TRAIN)
    self.assert_close(got, expected)

  def test_mamba_mixer_packed_and_right_padded(self):
    hf_mixer = self.hf.model.layers[0].mixer
    # Row 0 packs documents of 11 and 9 tokens (each crosses a chunk boundary); row 1 is 17 tokens + 3 pads.
    seg = np.ones((BATCH, SEQ), np.int32)
    seg[0] = [1] * 11 + [2] * 9
    seg[1] = [1] * 17 + [0] * 3
    got = np.asarray(
        self.mt.decoder.layers_0.mixer(jnp.asarray(self.x), decoder_segment_ids=jnp.asarray(seg), model_mode="train")
    )
    for row, start, end in ((0, 0, 11), (0, 11, 20), (1, 0, 17)):
      expected = _np(hf_mixer(torch.from_numpy(self.x[row : row + 1, start:end])))[0]
      self.assert_close(got[row, start:end], expected)

  def test_decoder_forward(self):
    expected = _np(self.hf(torch.from_numpy(self.ids).long(), use_cache=False).logits)
    self.assert_close(_logits(self.mt, self.ids), expected)

  def test_decoder_backward(self):
    cot = np.random.default_rng(2).normal(size=(BATCH, SEQ, VOCAB)).astype(np.float32)
    self.hf.zero_grad()
    logits = self.hf(torch.from_numpy(self.ids).long(), use_cache=False).logits
    (logits * torch.from_numpy(cot)).sum().backward()
    hf_grads = {k: _np(p.grad) for k, p in self.hf.named_parameters() if p.grad is not None}

    def loss(model):
      return jnp.sum(_logits(model, self.ids) * cot)

    grads = nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Param))(self.mt)
    layer0, layer1, layer2 = grads.decoder.layers_0, grads.decoder.layers_1, grads.decoder.layers_2
    pairs = {
        "embedding": (grads.token_embedder.embedding, hf_grads["model.embeddings.weight"]),
        "decoder_norm": (grads.decoder.decoder_norm.scale, hf_grads["model.norm_f.weight"]),
        "mamba in_proj": (layer0.mixer.in_proj.kernel, hf_grads["model.layers.0.mixer.in_proj.weight"].T),
        "mamba conv": (layer0.mixer.conv_weight, hf_grads["model.layers.0.mixer.conv1d.weight"][:, 0, :].T),
        "mamba A_log": (layer0.mixer.A_log, hf_grads["model.layers.0.mixer.A_log"]),
        "mamba dt_bias": (layer0.mixer.dt_bias, hf_grads["model.layers.0.mixer.dt_bias"]),
        "mamba D": (layer0.mixer.D, hf_grads["model.layers.0.mixer.D"]),
        "gated norm": (layer0.mixer.norm.scale, hf_grads["model.layers.0.mixer.norm.weight"]),
        "router": (layer1.mixer.MoeBlock_0.gate.kernel, hf_grads["model.layers.1.mixer.gate.weight"].T),
        "experts up": (
            layer1.mixer.MoeBlock_0.wi_0,
            hf_grads["model.layers.1.mixer.experts.up_proj"].transpose(0, 2, 1),
        ),
        "attention q": (
            layer2.mixer.query.kernel,
            hf_grads["model.layers.2.mixer.q_proj.weight"].T.reshape(HIDDEN, Q_HEADS, HEAD_DIM),
        ),
    }
    for name, (got, expected) in pairs.items():
      with self.subTest(name):
        got = np.asarray(got.get_value())
        self.assertTrue(np.all(np.isfinite(got)))
        # Gradients sum over all positions and vocab entries, so allow a relative 1e-4.
        self.assert_close(got, expected, rtol=1e-4, atol=1e-4 * float(np.max(np.abs(expected))))

  def test_forced_routing(self):
    with _RecordRouting(self.moe_gate) as rec:
      expected = _np(self.hf(torch.from_numpy(self.ids).long(), use_cache=False).logits)
    hf_top_k = rec.indices.astype(np.int32)
    free = np.asarray(_logits(self.mt, self.ids))
    forced = np.asarray(_logits(self.mt, self.ids, forced_routed_experts=jnp.asarray(hf_top_k)))
    self.assert_close(forced, expected)
    np.testing.assert_array_equal(forced, free)
    # Replay a different expert set on both sides: weights come from the unbiased scores of the forced experts.
    other = ((hf_top_k + TOP_K) % EXPERTS).astype(np.int32)
    with _RecordRouting(self.moe_gate, forced=other):
      expected_other = _np(self.hf(torch.from_numpy(self.ids).long(), use_cache=False).logits)
    got_other = _logits(self.mt, self.ids, forced_routed_experts=jnp.asarray(other))
    self.assert_close(got_other, expected_other)
    self.assertGreater(float(np.max(np.abs(expected_other - expected))), 1e-3)

  def test_prefill_then_decode_matches_full_forward(self):
    prefill = 12
    ar = maxtext_model(self.hf, model_mode=MODEL_MODE_AUTOREGRESSIVE, max_prefill_predict_length=prefill)
    full = np.asarray(_logits(self.mt, self.ids))
    pos = _positions(SEQ)
    steps = [
        ar(
            self.ids[:, :prefill],
            pos[:, :prefill],
            decoder_segment_ids=np.ones((BATCH, prefill), np.int32),
            enable_dropout=False,
            model_mode=MODEL_MODE_PREFILL,
        )[:, -1]
    ]
    for t in range(prefill, SEQ):
      out = ar(
          self.ids[:, t : t + 1],
          pos[:, t : t + 1],
          decoder_segment_ids=None,
          enable_dropout=False,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )
      steps.append(out[:, -1])
    self.assert_close(np.stack([np.asarray(s) for s in steps], axis=1), full[:, prefill - 1 :])


if __name__ == "__main__":
  unittest.main()
