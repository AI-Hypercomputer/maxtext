# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests validating Hy3 (Tencent Hunyuan V3) MaxText components against PyTorch references.

The reference modules are imported directly from `transformers.models.hy_v3`
rather than re-derived here: `transformers` ships a native Hy3 implementation,
and a hand-written copy risks encoding the same misreading of the architecture
that a MaxText port might contain, in which case the comparison passes while
both sides are wrong. Hy3's router is the concrete example -- the expert-choice
bias is added *after* the sigmoid and is used only to pick the top-k experts,
while the returned weights come from the *unbiased* sigmoid scores. Importing
the reference also means an upstream fix shows up here on the next
`transformers` bump instead of silently diverging.

Everything runs at toy dimensions on random weights, so no checkpoint download
is needed. Weights are initialized on the PyTorch side and copied into the JAX
modules so both start from an identical state.
"""

import os
import unittest

import pytest

try:
  import torch
  from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
  from transformers.models.hy_v3.modeling_hy_v3 import (
      HYV3MLP as TorchHy3MLP,
      HYV3MoE as TorchHy3MoE,
      HYV3DecoderLayer as TorchHy3DecoderLayer,
      HYV3RotaryEmbedding as TorchHy3RotaryEmbedding,
  )

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

# torch is not part of the base MaxText requirements, and `hy_v3` only exists in
# recent `transformers` releases; skip rather than fail collection without them.
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch or transformers.models.hy_v3 not available")

# pylint: disable=wrong-import-position
from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.models import hy3
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT


# ==============================================================================
# Helpers
# ==============================================================================


def to_jax(pt_tensor):
  """Converts a PyTorch tensor to a JAX array."""
  return jnp.asarray(pt_tensor.detach().cpu().numpy())


def init_normal_(module, std=0.02):
  """Fills every parameter and buffer of a module with N(0, std) values.

  `HYV3TopKRouter` and `HYV3Experts` allocate their weights with `torch.empty`,
  which leaves uninitialized memory behind; comparing against that produces
  NaN/Inf drift rather than a meaningful parity signal.
  """
  with torch.no_grad():
    for tensor in list(module.parameters()) + list(module.buffers()):
      if tensor.is_floating_point():
        tensor.normal_(mean=0.0, std=std)


def copy_mlp_weights(pt_mlp, jax_mlp):
  """Copies an `HYV3MLP`'s weights into a MaxText `MlpBlock`.

  HF `nn.Linear` stores kernels as `[out, in]` while MaxText stores `[in, out]`,
  and MaxText names the SwiGLU projections wi_0 (gate), wi_1 (up) and wo (down).
  """
  jax_mlp.wi_0.kernel.value = to_jax(pt_mlp.gate_proj.weight.t())
  jax_mlp.wi_1.kernel.value = to_jax(pt_mlp.up_proj.weight.t())
  jax_mlp.wo.kernel.value = to_jax(pt_mlp.down_proj.weight.t())


def copy_attention_weights(pt_attn, jax_attn, hidden_size, num_heads, num_kv_heads, head_dim):
  """Copies an `HYV3Attention`'s weights into a MaxText `Attention`.

  On top of the `[out, in]` -> `[in, out]` transpose, MaxText keeps the head
  axis separate: `[in, heads, head_dim]` for q/k/v and `[heads, head_dim, out]`
  for the output projection, where HF keeps `heads * head_dim` flattened. HF
  splits that flat axis head-major (`.view(*shape, -1, head_dim)`), so a plain
  reshape lines the two up.
  """
  jax_attn.query.kernel.value = to_jax(pt_attn.q_proj.weight.t().reshape(hidden_size, num_heads, head_dim))
  jax_attn.key.kernel.value = to_jax(pt_attn.k_proj.weight.t().reshape(hidden_size, num_kv_heads, head_dim))
  jax_attn.value.kernel.value = to_jax(pt_attn.v_proj.weight.t().reshape(hidden_size, num_kv_heads, head_dim))
  jax_attn.out.kernel.value = to_jax(pt_attn.o_proj.weight.t().reshape(num_heads, head_dim, hidden_size))
  jax_attn.query_norm.scale.value = to_jax(pt_attn.q_norm.weight)
  jax_attn.key_norm.scale.value = to_jax(pt_attn.k_norm.weight)


def copy_moe_weights(pt_moe, jax_moe):
  """Copies an `HYV3MoE` block's weights into a MaxText `RoutedAndSharedMoE`."""
  routed = jax_moe.MoeBlock_0

  # Router. HF holds the gate weight as `[num_experts, hidden]`, MaxText as
  # `[hidden, num_experts]`. HF keeps the expert-choice bias outside the gate
  # matmul as `e_score_correction_bias` and adds it after the sigmoid; MaxText's
  # `GateLogit` adds its `bias` at that same point, so they map 1:1.
  routed.gate.kernel.value = to_jax(pt_moe.gate.weight.t())
  routed.gate.bias.value = to_jax(pt_moe.e_score_correction_bias)

  # Routed experts. HF packs gate and up into a single
  # `[num_experts, 2 * moe_intermediate, hidden]` tensor split by `chunk(2, -1)`
  # after the matmul (so gate first, then up) and keeps down as
  # `[num_experts, hidden, moe_intermediate]`. MaxText keeps three stacks in the
  # `[in, out]` orientation.
  gate_up_proj = pt_moe.experts.gate_up_proj
  moe_intermediate = gate_up_proj.shape[1] // 2
  routed.wi_0.value = to_jax(gate_up_proj[:, :moe_intermediate, :].transpose(1, 2))
  routed.wi_1.value = to_jax(gate_up_proj[:, moe_intermediate:, :].transpose(1, 2))
  routed.wo.value = to_jax(pt_moe.experts.down_proj.transpose(1, 2))

  copy_mlp_weights(pt_moe.shared_experts, jax_moe.shared_experts)


# ==============================================================================
# Test Suite
# ==============================================================================


class Hy3VsReferenceTest(unittest.TestCase):
  """Unit tests comparing MaxText Hy3 components against the HF PyTorch reference."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 128
    self.head_dim = 32
    self.num_heads = 4
    self.num_kv_heads = 2
    self.mlp_dim = 256
    self.moe_mlp_dim = 128
    self.num_experts = 4
    self.num_experts_per_tok = 2

    # Base test configuration for MaxText (must be initialized first before JAX operations)
    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    self.jax_config = pyconfig.initialize(
        ["", base_config_path],
        model_name="hy3-tiny",
        override_model_config=True,
        base_emb_dim=self.hidden_size,
        head_dim=self.head_dim,
        base_num_query_heads=self.num_heads,
        base_num_kv_heads=self.num_kv_heads,
        base_mlp_dim=self.mlp_dim,
        base_moe_mlp_dim=self.moe_mlp_dim,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        shared_experts=1,
        routed_score_func="sigmoid",
        routed_bias=True,
        use_qk_norm=True,
        attention="dot_product",
        matmul_precision="highest",
        max_target_length=self.seq_len,
        dtype="float32",
        weight_dtype="float32",
        float32_logits=True,
        float32_qk_product=True,
        # The HF router runs its matmul and sigmoid in fp32 regardless of model dtype.
        float32_gate_logits=True,
        # megablox and the sparse ragged-dot path are TPU kernels; use the dense
        # path (with no token dropping) so this runs anywhere.
        megablox=False,
        sparse_matmul=False,
        capacity_factor=-1.0,
        dropout_rate=0.0,
    )

    # JAX mesh
    devices_array = maxtext_utils.create_device_mesh(self.jax_config)
    self.mesh = Mesh(devices_array, self.jax_config.mesh_axes)

    # Initial inputs
    self.pt_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    self.pt_positions = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1)
    self.jax_input = to_jax(self.pt_input)
    self.jax_positions = to_jax(self.pt_positions)
    self.jax_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

  def _hf_config(self, mlp_layer_types):
    """Builds the HF config from the MaxText config so the two cannot drift apart.

    `mlp_layer_types` names the dense/sparse pattern per layer; it is how HF
    expresses what MaxText calls `first_num_dense_layers`.
    """
    cfg = self.jax_config
    # `HYV3Config`'s fields are declared as annotations consumed by huggingface_hub's
    # `@strict` decorator, which pylint cannot resolve into an __init__ signature.
    hf_config = HYV3Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.emb_dim,
        intermediate_size=cfg.mlp_dim,
        moe_intermediate_size=cfg.moe_mlp_dim,
        num_hidden_layers=len(mlp_layer_types),
        num_attention_heads=cfg.num_query_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.max_target_length,
        rms_norm_eps=cfg.normalization_layer_epsilon,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        num_shared_experts=cfg.shared_experts,
        router_scaling_factor=cfg.routed_scaling_factor,
        mlp_layer_types=mlp_layer_types,
        rope_parameters={"rope_type": "default", "rope_theta": float(cfg.rope_max_timescale)},
    )
    # MaxText's `attention="dot_product"` is an unfused reference kernel; match it.
    hf_config._attn_implementation = "eager"  # pylint: disable=protected-access
    return hf_config

  def _run_pt_layer(self, pt_layer, hf_config):
    """Runs an `HYV3DecoderLayer` under a causal mask, returning its output."""
    rotary_emb = TorchHy3RotaryEmbedding(hf_config)
    position_embeddings = rotary_emb(self.pt_input, self.pt_positions)
    causal_mask = torch.full((self.seq_len, self.seq_len), torch.finfo(torch.float32).min, dtype=torch.float32).triu(
        diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return pt_layer(
        self.pt_input,
        attention_mask=causal_mask,
        position_ids=self.pt_positions,
        position_embeddings=position_embeddings,
    )

  def _run_jax_layer(self, jax_layer):
    """Runs a MaxText Hy3 decoder layer, unwrapping the `(output, kv_cache)` tuple."""
    jax_out = jax_layer(
        inputs=self.jax_input,
        decoder_segment_ids=self.jax_segment_ids,
        decoder_positions=self.jax_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return jax_out[0] if isinstance(jax_out, tuple) else jax_out

  def test_hy3_dense_mlp_vs_reference(self):
    """Validates Hy3's dense MLP (SwiGLU) against the HF reference."""
    hf_config = self._hf_config(["dense"])
    pt_mlp = TorchHy3MLP(hf_config)
    pt_out = pt_mlp(self.pt_input)

    jax_mlp = linears.MlpBlock(
        in_features=self.hidden_size,
        intermediate_dim=self.mlp_dim,
        activations=["silu", "linear"],
        intermediate_dropout_rate=0.0,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        config=self.jax_config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )
    copy_mlp_weights(pt_mlp, jax_mlp)

    jax_out = jax_mlp(self.jax_input, deterministic=True)

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)

  def test_hy3_moe_block_vs_reference(self):
    """Validates the Hy3 MoE block (sigmoid+bias router, shared expert) against the HF reference.

    This is the piece that does not follow from Qwen/DeepSeek naming: the bias
    is added post-sigmoid and only steers expert selection, the gathered weights
    are renormalized over the selected experts and then scaled by
    `router_scaling_factor`, and the shared expert is added to the routed sum.
    """
    hf_config = self._hf_config(["sparse"])
    pt_moe = TorchHy3MoE(hf_config)
    init_normal_(pt_moe)
    pt_out = pt_moe(self.pt_input)

    jax_moe = moe.RoutedAndSharedMoE(
        config=self.jax_config,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(self.jax_config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    copy_moe_weights(pt_moe, jax_moe)

    jax_out, _, _ = jax_moe(self.jax_input)

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)

  def test_hy3_dense_layer_vs_reference(self):
    """Validates a full `Hy3DenseLayer` forward pass against the HF reference."""
    hf_config = self._hf_config(["dense"])
    pt_layer = TorchHy3DecoderLayer(hf_config, layer_idx=0)
    pt_out = self._run_pt_layer(pt_layer, hf_config)

    jax_layer = hy3.Hy3DenseLayer(
        config=self.jax_config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
        layer_idx=0,
    )
    jax_layer.pre_self_attention_layer_norm.scale.value = to_jax(pt_layer.input_layernorm.weight)
    jax_layer.post_self_attention_layer_norm.scale.value = to_jax(pt_layer.post_attention_layernorm.weight)
    copy_attention_weights(
        pt_layer.self_attn,
        jax_layer.self_attention,
        self.hidden_size,
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
    )
    copy_mlp_weights(pt_layer.mlp, jax_layer.mlp)

    jax_out = self._run_jax_layer(jax_layer)

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)

  def test_hy3_moe_layer_vs_reference(self):
    """Validates a full `Hy3MoELayer` forward pass against the HF reference."""
    # Layer 0 is dense and layer 1 is sparse, mirroring `first_num_dense_layers=1`.
    hf_config = self._hf_config(["dense", "sparse"])
    pt_layer = TorchHy3DecoderLayer(hf_config, layer_idx=1)
    init_normal_(pt_layer.mlp)
    pt_out = self._run_pt_layer(pt_layer, hf_config)

    jax_layer = hy3.Hy3MoELayer(
        config=self.jax_config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
        layer_idx=1,
    )
    jax_layer.pre_self_attention_layer_norm.scale.value = to_jax(pt_layer.input_layernorm.weight)
    jax_layer.post_self_attention_layer_norm.scale.value = to_jax(pt_layer.post_attention_layernorm.weight)
    copy_attention_weights(
        pt_layer.self_attn,
        jax_layer.self_attention,
        self.hidden_size,
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
    )
    copy_moe_weights(pt_layer.mlp, jax_layer.Hy3MoeBlock_0)

    jax_out = self._run_jax_layer(jax_layer)

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
