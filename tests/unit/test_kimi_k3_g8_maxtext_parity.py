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

"""Numerical parity of the *MaxText* Kimi-K3 stack against the PyTorch reference.

g1-g4 prove each leaf module (KDA, MLA, dense MLP, latent MoE, AttnRes pooling) and g6
proves the Linen test oracle; neither ever runs `maxtext.models.kimi_k3.KimiK3DecoderLayer`
or `NNXDecoder._apply_kimi_layers` against PyTorch. This suite closes that gap at toy
scale (hidden 64, 4 experts, 8 layers in two AttnRes blocks of 4), always loading the
PyTorch weights through the production `PARAM_MAPPING["kimi-k3"]`:

  * per-layer: KDA+dense (layer 0), KDA+MoE (layer 1), MLA+MoE (layer 3), and a
    mid-stack block boundary (layer 4) with a non-empty highway;
  * per-block: all 8 layers run sequentially, checking `(prefix_sum, block_residual)`
    after every layer;
  * full model: `Transformer` logits vs `KimiLinearForCausalLM`.

See `tests/utils/kimi_k3_parity_utils.py` for why the reference's KDA path is patched.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.utils.kimi_k3_parity_utils import (
    TinyKimiK3Spec,
    causal_mask_pt,
    convert_hf_layer_state_dict,
    convert_hf_model_state_dict,
    hf_block_residual_to_maxtext,
    init_hf_uninitialized_params,
    load_hf_reference,
    load_params_into_nnx,
    make_hf_config,
    make_maxtext_config,
    make_mesh,
    positions_and_segments,
    unpatch_hf_reference,
)
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference

_RTOL = 1e-4
_ATOL = 1e-4


class KimiK3FullAttnLayerResolutionTest(unittest.TestCase):
  """Index-base contract of `param_mapping._resolve_kimi_k3_full_attn_layers`."""

  def test_hf_linear_attn_config_is_one_indexed_and_authoritative(self):
    cfg = {"num_hidden_layers": 8, "full_attn_layers": [0], "linear_attn_config": {"full_attn_layers": [4, 8]}}
    self.assertEqual(param_mapping._resolve_kimi_k3_full_attn_layers(cfg), {3, 7})  # pylint: disable=protected-access

  def test_top_level_is_zero_indexed(self):
    cfg = {"num_hidden_layers": 4, "full_attn_layers": [1]}
    self.assertEqual(param_mapping._resolve_kimi_k3_full_attn_layers(cfg), {1})  # pylint: disable=protected-access

  def test_out_of_range_is_rejected(self):
    with self.assertRaises(ValueError):
      param_mapping._resolve_kimi_k3_full_attn_layers(  # pylint: disable=protected-access
          {"num_hidden_layers": 4, "full_attn_layers": [4]}
      )
    with self.assertRaises(ValueError):
      param_mapping._resolve_kimi_k3_full_attn_layers(  # pylint: disable=protected-access
          {"num_hidden_layers": 4, "linear_attn_config": {"full_attn_layers": [0]}}
      )

  def test_mapping_and_hooks_agree_on_layer_types(self):
    """Layer 3 must be MLA in both the name map and the hook map for an HF-style config."""
    cfg = {"num_hidden_layers": 4, "num_experts": 2, "linear_attn_config": {"full_attn_layers": [4]}}
    mapping = param_mapping.PARAM_MAPPING["kimi-k3"](cfg, None, scan_layers=False)
    hooks = param_mapping.HOOK_FNS["kimi-k3"](cfg, None, scan_layers=False, saving_to_hf=False)
    self.assertIn("params-decoder-layers_3-self_attention-wq_a-kernel", mapping)
    self.assertNotIn("params-decoder-layers_3-self_attention-q_proj-kernel", mapping)
    self.assertIn("params-decoder-layers_3-self_attention-wq_b-kernel", hooks)
    self.assertIn("params-decoder-layers_2-self_attention-q_conv1d-kernel", hooks)


@requires_kimi_k3_reference
class KimiK3MaxTextParityTest(unittest.TestCase):
  """MaxText `KimiK3DecoderLayer` / decoder vs PyTorch `KimiDecoderLayer` / CausalLM."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)
    cls.spec = TinyKimiK3Spec()
    cls.pt_cfg = make_hf_config(cls.spec, cls.hf_config_mod)
    cls.mt_cfg = make_maxtext_config(cls.spec)
    cls.mesh = make_mesh(cls.mt_cfg)

  @classmethod
  def tearDownClass(cls):
    # The KDA patch is process-global; restore it so collection order cannot decide
    # which kernel another test file ends up running against.
    unpatch_hf_reference()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    np.random.seed(42)

  # ---------------------------------------------------------------------------
  # helpers
  # ---------------------------------------------------------------------------
  def _pt_layer(self, layer_idx: int):
    return init_hf_uninitialized_params(
        self.hf_model_mod.KimiDecoderLayer(self.pt_cfg, layer_idx=layer_idx), seed=layer_idx
    ).eval()

  def _mt_layer(self, layer_idx: int, pt_layer) -> kimi_k3.KimiK3DecoderLayer:
    """Builds the MaxText layer for `layer_idx` and loads `pt_layer`'s weights into it."""
    layer = kimi_k3.KimiK3DecoderLayer(
        config=self.mt_cfg,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
        layer_idx=layer_idx,
        is_linear_attn=not self.spec.is_full_attn(layer_idx),
        is_moe=self.spec.is_moe(layer_idx),
    )
    converted = convert_hf_layer_state_dict(pt_layer, layer_idx, self.spec)
    load_params_into_nnx(layer, converted, prefix=f"params-decoder-layers_{layer_idx}")
    return layer

  def _run_pt_layer(self, pt_layer, x_np: np.ndarray, b_np: np.ndarray):
    """HF layer forward. KDA layers must see `attention_mask=None` on the CPU path."""
    B, S, D = x_np.shape
    mask = None if pt_layer.is_linear_attn else causal_mask_pt(S)
    with torch.no_grad():
      h, b = pt_layer(
          torch.from_numpy(x_np),
          attention_mask=mask,
          block_residual=torch.from_numpy(b_np.reshape(B * S, -1, D)),
      )
    return h.numpy(), hf_block_residual_to_maxtext(b, B, S)

  def _run_mt_layer(self, mt_layer, x_np: np.ndarray, b_np: np.ndarray):
    """Runs a single MaxText layer forward pass."""
    B, S, _ = x_np.shape
    positions, segment_ids = positions_and_segments(B, S)
    h, b, _ = mt_layer(
        jnp.asarray(x_np),
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        block_residual=jnp.asarray(b_np),
    )
    return np.asarray(h), np.asarray(b)

  def _random_inputs(self, num_blocks: int):
    B, S, D = self.spec.batch_size, self.spec.seq_len, self.spec.hidden_size
    x = np.random.randn(B, S, D).astype(np.float32)
    b = np.random.randn(B, S, num_blocks, D).astype(np.float32)
    return x, b

  def _assert_layer_parity(self, layer_idx: int, num_blocks_in: int, expected_blocks_out: int):
    """Asserts numerical parity for a layer between PyTorch and MaxText."""
    pt_layer = self._pt_layer(layer_idx)
    mt_layer = self._mt_layer(layer_idx, pt_layer)
    x, b = self._random_inputs(num_blocks_in)

    h_pt, b_pt = self._run_pt_layer(pt_layer, x, b)
    h_mt, b_mt = self._run_mt_layer(mt_layer, x, b)

    self.assertEqual(b_pt.shape[-2], expected_blocks_out)
    self.assertEqual(b_mt.shape, b_pt.shape)
    np.testing.assert_allclose(b_mt, b_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {layer_idx} block_residual")
    np.testing.assert_allclose(h_mt, h_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {layer_idx} prefix_sum")

  # ---------------------------------------------------------------------------
  # per-layer parity
  # ---------------------------------------------------------------------------
  def test_layer0_kda_dense_opens_first_block(self):
    """Layer 0: KDA + dense MLP, empty highway in, one checkpointed block out."""
    self._assert_layer_parity(layer_idx=0, num_blocks_in=0, expected_blocks_out=1)

  def test_layer1_kda_moe_with_highway(self):
    """Layer 1: KDA + latent MoE, pooling over one existing block, no boundary."""
    self._assert_layer_parity(layer_idx=1, num_blocks_in=1, expected_blocks_out=1)

  def test_layer3_mla_moe_with_highway(self):
    """Layer 3: gated NoPE MLA + latent MoE, pooling over one existing block."""
    self.assertTrue(self.spec.is_full_attn(3))
    self._assert_layer_parity(layer_idx=3, num_blocks_in=1, expected_blocks_out=1)

  def test_layer4_kda_moe_mid_stack_block_boundary(self):
    """Layer 4 (4 % block_size == 0): pre-attention pooling, then a new block is pushed."""
    self.assertEqual(4 % self.spec.attn_res_block_size, 0)
    self._assert_layer_parity(layer_idx=4, num_blocks_in=1, expected_blocks_out=2)

  # ---------------------------------------------------------------------------
  # per-block parity: whole highway across two AttnRes blocks
  # ---------------------------------------------------------------------------
  def test_attn_res_highway_across_all_layers(self):
    """Runs all 8 layers sequentially and checks the carry after every layer."""
    B, S, D = self.spec.batch_size, self.spec.seq_len, self.spec.hidden_size
    x = np.random.randn(B, S, D).astype(np.float32)

    pt_layers = [self._pt_layer(i) for i in range(self.spec.num_layers)]
    mt_layers = [self._mt_layer(i, pt_layers[i]) for i in range(self.spec.num_layers)]

    h_pt, b_pt = x, np.zeros((B, S, 0, D), np.float32)
    h_mt, b_mt = x, np.zeros((B, S, 0, D), np.float32)
    for i in range(self.spec.num_layers):
      h_pt, b_pt = self._run_pt_layer(pt_layers[i], h_pt, b_pt)
      h_mt, b_mt = self._run_mt_layer(mt_layers[i], h_mt, b_mt)
      with self.subTest(layer=i):
        expected_blocks = i // self.spec.attn_res_block_size + 1
        self.assertEqual(b_pt.shape[-2], expected_blocks)
        self.assertEqual(b_mt.shape, b_pt.shape)
        np.testing.assert_allclose(b_mt, b_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {i} block_residual")
        np.testing.assert_allclose(h_mt, h_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {i} prefix_sum")
      # Feed each side its own output (true sequential execution), but both sides
      # started from identical inputs so drift is what we are measuring.

  # ---------------------------------------------------------------------------
  # full model logits
  # ---------------------------------------------------------------------------
  def test_full_model_logits(self):
    """`Transformer` logits vs `KimiLinearForCausalLM`, weights via PARAM_MAPPING."""
    pt_model = init_hf_uninitialized_params(self.hf_model_mod.KimiLinearForCausalLM(self.pt_cfg)).eval()
    mt_model = model_creation_utils.from_config(
        self.mt_cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
    )
    converted = convert_hf_model_state_dict(pt_model, self.spec)
    loaded = load_params_into_nnx(mt_model, converted, prefix="params")
    self.assertGreater(len(loaded), 0)

    B, S = self.spec.batch_size, self.spec.seq_len
    tokens = np.random.randint(0, self.spec.vocab_size, size=(B, S)).astype(np.int32)
    with torch.no_grad():
      logits_pt = pt_model(input_ids=torch.from_numpy(tokens).long()).logits.numpy()

    positions, _ = positions_and_segments(B, S)
    logits_mt = np.asarray(mt_model(jnp.asarray(tokens), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))

    self.assertEqual(logits_mt.shape, logits_pt.shape)
    np.testing.assert_allclose(logits_mt, logits_pt, rtol=_RTOL, atol=_ATOL)


if __name__ == "__main__":
  unittest.main()
