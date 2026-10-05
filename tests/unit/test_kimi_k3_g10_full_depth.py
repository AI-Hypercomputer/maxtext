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

"""Full-*depth* Kimi-K3 parity: the real 93-layer layout at toy width.

g8 proves numerical parity of the MaxText stack against PyTorch, but only for the
8-layer toy layout (two AttnRes blocks of 4, MLA at layers 3 and 7). The real
checkpoint test (`tests/utils/kimi_k3_real_ckpt.py`) proves the conversion
pipeline on real weights, but only for a 4-layer truncation — less than half of one
AttnRes block. Neither exercises the structural edges of the released architecture:

  * `attn_res_block_size=12` over 93 layers -> 8 blocks, the last one a *partial*
    9-layer tail (layers 84-92) that is pooled by the final output projection;
  * the mid-stack block boundaries at layers 12, 24, ..., 84;
  * the back-to-back full-attention layers 91 and 92, which break the otherwise
    regular 3xKDA + 1xMLA cycle.

Those are all pure layout/plumbing properties: they need the real *number* of layers
and the real layer-type pattern, but not the real *width*. This suite therefore runs
the released layout (93 layers, `full_attn_layers=[3, 7, ..., 91, 92]`, block size 12,
layer 0 dense) at toy width (hidden 64, 4 experts), loading PyTorch weights through the
production `PARAM_MAPPING["kimi-k3"]` exactly as g8 does.

Cost: seconds of CPU, no checkpoint, no accelerator.
"""

import math
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils.kimi_k3_parity_utils import (
    TinyKimiK3Spec,
    causal_mask_pt,
    convert_hf_layer_state_dict,
    convert_hf_model_state_dict,
    hf_block_residual_to_maxtext,
    load_hf_reference,
    load_params_into_nnx,
    make_hf_config,
    make_maxtext_config,
    make_mesh,
    positions_and_segments,
    unpatch_hf_reference,
)
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference

_KIMI_K3_YML = os.path.join(MAXTEXT_CONFIGS_DIR, "models", "kimi-k3.yml")

# The released layout: 23 cycles of (3x KDA + 1x MLA) -> MLA at 0-indexed 3, 7, ..., 91,
# plus a 93rd layer (index 92) that is also full attention.
_REAL_NUM_LAYERS = 93
_REAL_FULL_ATTN_LAYERS = tuple(range(3, _REAL_NUM_LAYERS - 1, 4)) + (_REAL_NUM_LAYERS - 1,)
_REAL_ATTN_RES_BLOCK_SIZE = 12
_REAL_FIRST_K_DENSE_REPLACE = 1

# Toy width, real depth. batch 1 / seq 8 keeps 93 eager layers cheap.
FULL_DEPTH_SPEC = TinyKimiK3Spec(
    num_layers=_REAL_NUM_LAYERS,
    full_attn_layers=_REAL_FULL_ATTN_LAYERS,
    attn_res_block_size=_REAL_ATTN_RES_BLOCK_SIZE,
    first_k_dense_replace=_REAL_FIRST_K_DENSE_REPLACE,
    batch_size=1,
    seq_len=8,
)

# Tolerances are calibrated from the measured error curve at this depth, not inherited
# from g8 (whose 1e-4 is an 8-layer number). Under the controlled init below, the
# per-layer absolute error accumulates smoothly with depth and plateaus -- there is no
# step change at any layer or AttnRes block boundary:
#
#     layer  0: abs 1.1e-05   layer 38: abs 1.4e-04
#     layer 68: abs 4.3e-04   layer 92: abs 3.9e-04   (|h| ~ 4.8 throughout)
#
# so 1e-3 leaves ~2x headroom over the worst observed value. A regression in any layer
# type, block boundary or the partial tail moves these by orders of magnitude, not 2x.
_RTOL = 1e-4
_ATOL = 1e-3
# Logits are compared *relatively*: after 93 layers the absolute scale of the logits is
# an emergent property of the init, so a fixed atol would be meaningless. Measured:
# max|diff| = 1.21e-03 against |logits|max = 1.71, i.e. rel = 7.1e-04. The logit
# divergence is ~9x the hidden-state divergence because the LM head sums 64 terms of
# alternating sign, so the relative error of the difference is amplified. 5e-3 keeps a
# ~7x margin; the exact top-1 match asserted below is the sharp check.
_LOGITS_REL_TOL = 5e-3


def init_pt_model_for_depth(model: torch.nn.Module, *, seed: int = 1234, gain: float = 0.5) -> None:
  """Deterministic, well-conditioned init for a 93-layer stack.

  Neither of HF's own initializations is usable at this depth:

    * constructing bare `KimiDecoderLayer`s skips `post_init()`, leaving every
      `nn.Linear` at torch's default (std ~ 1/sqrt(fan_in) = 0.125 at hidden 64). The
      residual stream then grows layer over layer until the SiTU activation's
      `beta=25` branch overflows, and the stack returns NaN around layer 40;
    * `post_init()` (std 0.02) makes the opposite problem: sublayer outputs are so
      small that driving a layer standalone can produce an exactly-zero KDA query,
      and the L2 normalization inside the KDA kernel then evaluates 0/0 -> NaN.

  The parity comparison is fair for *any* weights, because MaxText is loaded from
  these exact tensors through `PARAM_MAPPING["kimi-k3"]`. All that is required is a
  forward pass that is numerically well conditioned, so we choose the weights:

    * norm scales -> 1.0, the only value for which RMSNorm is a no-op;
    * projections -> N(0, gain/sqrt(fan_in)), so each sublayer's contribution is a
      fraction `gain` of its (RMS-normalized) input. The residual stream then grows
      like sqrt(depth) rather than exponentially, and pre-activations stay O(1), well
      clear of the `exp(25 x)` overflow;
    * `A_log` -> log of decay rates spread over [1, 8], the regime the released
      checkpoint uses (`g = gate_lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`);
    * conv kernels -> non-degenerate, so the KDA query is never identically zero.
  """
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("norm") or name.endswith("norm.weight") or "layernorm" in name:
        p.fill_(1.0)
      elif name.endswith("A_log"):
        p.copy_(torch.log(torch.linspace(1.0, 8.0, p.numel())).reshape(p.shape))
      elif name.endswith("dt_bias"):
        p.uniform_(-0.5, 0.5, generator=gen)
      elif name.endswith("e_score_correction_bias"):
        p.zero_()
      elif p.ndim >= 2:
        p.normal_(0.0, gain / math.sqrt(p.shape[-1]), generator=gen)
      else:
        p.normal_(0.0, 0.02, generator=gen)


class KimiK3ReleasedLayoutTest(unittest.TestCase):
  """The toy spec must mirror `configs/models/kimi-k3.yml`, or this suite proves nothing."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(_KIMI_K3_YML, "rt", encoding="utf8") as f:
      cls.yml = yaml.safe_load(f)

  def test_spec_matches_released_yml(self):
    self.assertEqual(self.yml["base_num_decoder_layers"], FULL_DEPTH_SPEC.num_layers)
    self.assertEqual(self.yml["full_attn_layers"], list(FULL_DEPTH_SPEC.full_attn_layers))
    self.assertEqual(self.yml["attn_res_block_size"], FULL_DEPTH_SPEC.attn_res_block_size)
    self.assertEqual(self.yml["first_k_dense_replace"], FULL_DEPTH_SPEC.first_k_dense_replace)

  def test_full_attn_layers_are_consistent_with_the_cycle_fields(self):
    """`full_attn_layers` must be the 3xKDA+1xMLA cycle, plus the trailing extra MLA."""
    interval = self.yml["inhomogeneous_layer_cycle_interval"]
    num_cycles = self.yml["num_cycles"]
    self.assertEqual(interval, self.yml["kda_layers_per_cycle"] + self.yml["mla_layers_per_cycle"])
    cycled = [c * interval + interval - 1 for c in range(num_cycles)]
    self.assertEqual(self.yml["full_attn_layers"], cycled + [self.yml["base_num_decoder_layers"] - 1])
    # 23 cycles cover 92 layers; layer 92 is the odd one out and is full attention.
    self.assertEqual(num_cycles * interval, self.yml["base_num_decoder_layers"] - 1)

  def test_last_attn_res_block_is_partial(self):
    """93 layers / block 12 -> 8 blocks, the last covering only 9 layers."""
    n, bs = FULL_DEPTH_SPEC.num_layers, FULL_DEPTH_SPEC.attn_res_block_size
    self.assertEqual((n + bs - 1) // bs, 8)
    self.assertEqual(n % bs, 9)


@requires_kimi_k3_reference
class KimiK3FullDepthParityTest(unittest.TestCase):
  """MaxText vs PyTorch over all 93 layers of the released layout, at toy width."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.manual_seed(42)
    np.random.seed(42)
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)
    cls.spec = FULL_DEPTH_SPEC
    cls.pt_cfg = make_hf_config(cls.spec, cls.hf_config_mod)
    cls.mt_cfg = make_maxtext_config(cls.spec)
    cls.mesh = make_mesh(cls.mt_cfg)
    # Built once and shared: constructing the model and 93 MaxText layers dominates
    # the runtime, and both tests below want the same weights.
    cls._pt_model = None
    cls._pt_layers = None
    cls._mt_layers = None

  @classmethod
  def tearDownClass(cls):
    # The KDA patch is process-global; restore it so collection order cannot decide
    # which kernel another test file ends up running against.
    unpatch_hf_reference()
    cls._pt_model = cls._pt_layers = cls._mt_layers = None
    super().tearDownClass()

  # ---------------------------------------------------------------------------
  # helpers (mirrors of g8's, kept local so g8 stays untouched)
  # ---------------------------------------------------------------------------
  @classmethod
  def _pt_causal_lm(cls):
    """The single reference model, under the controlled init (see module docstring)."""
    if cls._pt_model is None:
      model = cls.hf_model_mod.KimiLinearForCausalLM(cls.pt_cfg).eval()
      init_pt_model_for_depth(model)
      cls._pt_model = model
    return cls._pt_model

  @classmethod
  def _layers(cls):
    """Lazily builds and caches the paired 93-layer stacks."""
    if cls._pt_layers is None:
      pt_layers = [layer.eval() for layer in cls._pt_causal_lm().model.layers]
      mt_layers = []
      for i, pt_layer in enumerate(pt_layers):
        mt_layer = kimi_k3.KimiK3DecoderLayer(
            config=cls.mt_cfg,
            model_mode=MODEL_MODE_TRAIN,
            mesh=cls.mesh,
            rngs=nnx.Rngs(0),
            layer_idx=i,
            is_linear_attn=not cls.spec.is_full_attn(i),
            is_moe=cls.spec.is_moe(i),
        )
        converted = convert_hf_layer_state_dict(pt_layer, i, cls.spec)
        load_params_into_nnx(mt_layer, converted, prefix=f"params-decoder-layers_{i}")
        mt_layers.append(mt_layer)
      cls._pt_layers, cls._mt_layers = pt_layers, mt_layers
    return cls._pt_layers, cls._mt_layers

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

  def _zero_inputs(self):
    B, S, D = self.spec.batch_size, self.spec.seq_len, self.spec.hidden_size
    x = np.random.randn(B, S, D).astype(np.float32)
    return x, np.zeros((B, S, 0, D), np.float32)

  # ---------------------------------------------------------------------------
  # structure: what the reference does to the AttnRes highway over 93 layers
  # ---------------------------------------------------------------------------
  def test_block_structure_at_full_depth(self):
    """The PyTorch reference must open a new block every 12 layers, ending with 8."""
    pt_layers, _ = self._layers()
    h, b = self._zero_inputs()
    for i in range(self.spec.num_layers):
      h, b = self._run_pt_layer(pt_layers[i], h, b)  # pylint: disable=unsubscriptable-object
      with self.subTest(layer=i):
        self.assertEqual(b.shape[-2], i // self.spec.attn_res_block_size + 1)
        self.assertTrue(np.isfinite(h).all(), f"non-finite prefix_sum at layer {i}")
    self.assertEqual(b.shape[-2], 8)

  # ---------------------------------------------------------------------------
  # numerics: MaxText matches PyTorch at every one of the 93 layers
  # ---------------------------------------------------------------------------
  def test_full_depth_highway_layerwise(self):
    """Runs both stacks layer by layer; names the first layer that diverges."""
    pt_layers, mt_layers = self._layers()
    x, b0 = self._zero_inputs()
    h_pt, b_pt = x, b0
    h_mt, b_mt = x, b0
    for i in range(self.spec.num_layers):
      h_pt, b_pt = self._run_pt_layer(pt_layers[i], h_pt, b_pt)  # pylint: disable=unsubscriptable-object
      h_mt, b_mt = self._run_mt_layer(mt_layers[i], h_mt, b_mt)  # pylint: disable=unsubscriptable-object
      with self.subTest(layer=i, is_full_attn=self.spec.is_full_attn(i)):
        self.assertEqual(b_mt.shape, b_pt.shape)
        np.testing.assert_allclose(b_mt, b_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {i} block_residual")
        np.testing.assert_allclose(h_mt, h_pt, rtol=_RTOL, atol=_ATOL, err_msg=f"layer {i} prefix_sum")

  # ---------------------------------------------------------------------------
  # end to end: embed -> 93 layers -> final norm -> output AttnRes pooling -> head
  # ---------------------------------------------------------------------------
  def test_full_depth_model_logits(self):
    """`Transformer` logits vs `KimiLinearForCausalLM`, weights via PARAM_MAPPING."""
    pt_model = self._pt_causal_lm()
    mt_model = model_creation_utils.from_config(
        self.mt_cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
    )
    converted = convert_hf_model_state_dict(pt_model, self.spec)
    loaded = load_params_into_nnx(mt_model, converted, prefix="params")
    self.assertGreater(len(loaded), 0)

    B, S = self.spec.batch_size, self.spec.seq_len
    tokens = np.random.randint(0, self.spec.vocab_size, size=(B, S)).astype(np.int32)
    with torch.no_grad():
      logits_pt = pt_model(input_ids=torch.from_numpy(tokens).long()).logits.numpy()  # pylint: disable=not-callable

    positions, _ = positions_and_segments(B, S)
    logits_mt = np.asarray(mt_model(jnp.asarray(tokens), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))

    self.assertEqual(logits_mt.shape, logits_pt.shape)
    self.assertTrue(np.isfinite(logits_mt).all())

    scale = max(float(np.abs(logits_pt).max()), 1e-6)
    max_abs_diff = float(np.abs(logits_mt - logits_pt).max())
    rel = max_abs_diff / scale
    msg = f"93 layers: max|diff|={max_abs_diff:.3e}, |logits|max={scale:.3e}, rel={rel:.3e}"
    self.assertLess(rel, _LOGITS_REL_TOL, msg)
    np.testing.assert_array_equal(logits_mt.argmax(-1), logits_pt.argmax(-1), err_msg=f"top-1 mismatch ({msg})")


if __name__ == "__main__":
  unittest.main()
