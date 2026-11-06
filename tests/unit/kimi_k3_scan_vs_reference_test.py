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

"""Scanned Kimi-K3 (`scan_layers=True`) logits vs the Hugging Face reference, at toy width.

The weights reach the scanned model through the scanned checkpoint conversion
(`PARAM_MAPPING["kimi-k3"](..., scan_layers=True)` run through `to_maxtext`'s own loaders,
i.e. HF state dict -> stacked `scanned_blocks` params), not by re-stacking unscanned params,
so the conversion is what is under test. Each case also runs the unscanned model on the
unscanned conversion of the same weights and checks scanned == unscanned tightly.

Cases (all with the reference's controlled init from `kimi_k3_full_depth_vs_reference_test`):
  * the released 93-layer layout (prefix 0..3, 22 cycles, suffix [92]; block commits every
    12 layers, i.e. inside the scanned region);
  * 13 layers, full_attn_layers=[3, 7, 11, 12] (prefix 0..3, 2 cycles, suffix [12]) with
    attn_res_block_size 5 (commits inside the cycles, pending-slot path) and 2 (several
    commits per cycle, per-sub-layer buffer path).

Parity criteria are the unscanned full-depth test's: max|diff| / max|logits| < 5e-3 and an
exact top-1 match at every position. Skips unless torch, transformers and einops are
installed and the HF reference is checked out at `kimi-k3-hf-reference/`.
"""

import importlib.util
import unittest

from absl.testing import parameterized
import pytest

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.unit import kimi_k3_full_depth_vs_reference_test as full_depth
from tests.utils.kimi_k3_conversion_utils import convert_hf_state_dict_via_to_maxtext, load_checkpoint_params_into_nnx
from tests.utils.kimi_k3_parity_utils import (
    TinyKimiK3Spec,
    convert_hf_model_state_dict,
    load_hf_reference,
    load_params_into_nnx,
    make_hf_config,
    make_maxtext_config,
    positions_and_segments,
    unpatch_hf_reference,
)
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference

# pylint: enable=wrong-import-position

_HF_DEPS_AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("transformers", "einops"))

_LOGITS_REL_TOL = 5e-3  # same as the unscanned full-depth logits test
# Scanned vs unscanned on identical weights only differs by fp32 summation order.
_SCAN_VS_UNSCAN_REL_TOL = 1e-4

_SPECS = {
    "released_93_layers": full_depth.FULL_DEPTH_SPEC,
    "13_layers_block5": TinyKimiK3Spec(
        num_layers=13, full_attn_layers=(3, 7, 11, 12), attn_res_block_size=5, batch_size=1, seq_len=8
    ),
    "13_layers_block2": TinyKimiK3Spec(
        num_layers=13, full_attn_layers=(3, 7, 11, 12), attn_res_block_size=2, batch_size=1, seq_len=8
    ),
}


def _stats(logits_mt: np.ndarray, logits_pt: np.ndarray) -> dict:
  diff = np.abs(logits_mt - logits_pt)
  scale = max(float(np.abs(logits_pt).max()), 1e-6)
  flat_mt, flat_pt = logits_mt.ravel().astype(np.float64), logits_pt.ravel().astype(np.float64)
  return {
      "max_abs": float(diff.max()),
      "rel": float(diff.max()) / scale,
      "top1": int((logits_mt.argmax(-1) == logits_pt.argmax(-1)).sum()),
      "positions": int(np.prod(logits_pt.shape[:-1])),
      "cosine": float(flat_mt @ flat_pt / (np.linalg.norm(flat_mt) * np.linalg.norm(flat_pt))),
  }


@requires_kimi_k3_reference
@unittest.skipUnless(_HF_DEPS_AVAILABLE, "transformers and einops are required by the HF reference")
class KimiK3ScannedVsReferenceTest(parameterized.TestCase):
  """Scanned MaxText (weights via the scanned conversion) vs `KimiLinearForCausalLM`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._prev_matmul_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)

  @classmethod
  def tearDownClass(cls):
    unpatch_hf_reference()
    jax.config.update("jax_default_matmul_precision", cls._prev_matmul_precision)
    super().tearDownClass()

  def _mt_logits(self, cfg, load_fn, tokens):
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    self.assertGreater(len(load_fn(model)), 0)
    self.assertEqual(hasattr(model.decoder, "scanned_blocks"), bool(cfg.scan_layers))
    positions, _ = positions_and_segments(*tokens.shape)
    return np.asarray(model(jnp.asarray(tokens), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))

  @parameterized.named_parameters(*((name, name) for name in _SPECS))
  def test_scanned_logits_vs_reference(self, name):
    spec = _SPECS[name]
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    pt_model = self.hf_model_mod.KimiLinearForCausalLM(make_hf_config(spec, self.hf_config_mod)).eval()
    full_depth.init_pt_model_for_depth(pt_model)
    tokens = rng.integers(0, spec.vocab_size, size=(spec.batch_size, spec.seq_len)).astype(np.int32)
    with torch.no_grad():
      logits_pt = pt_model(input_ids=torch.from_numpy(tokens).long()).logits.numpy()  # pylint: disable=not-callable

    scanned_cfg = make_maxtext_config(spec, scan_layers=True)
    layout = kimi_k3.kimi_scan_layout(scanned_cfg)
    self.assertGreater(layout.num_blocks, 0)
    model_info, _ = to_maxtext.get_maxtext_model_info(scanned_cfg)
    converted = convert_hf_state_dict_via_to_maxtext(
        pt_model.state_dict(), spec.mapping_config(), scanned_cfg, {k: v[1] for k, v in model_info.items()}
    )
    logits_scan = self._mt_logits(scanned_cfg, lambda m: load_checkpoint_params_into_nnx(m, converted), tokens)

    unscanned_converted = convert_hf_model_state_dict(pt_model, spec)
    logits_unscan = self._mt_logits(
        make_maxtext_config(spec), lambda m: load_params_into_nnx(m, unscanned_converted, prefix="params"), tokens
    )

    vs_ref, vs_unscan = _stats(logits_scan, logits_pt), _stats(logits_scan, logits_unscan)
    unscan_vs_ref = _stats(logits_unscan, logits_pt)
    print(
        f"[{name}] layout={tuple(layout)[:3]} scanned vs HF: max|diff|={vs_ref['max_abs']:.3e} rel={vs_ref['rel']:.3e} "
        f"top-1 {vs_ref['top1']}/{vs_ref['positions']} cosine={vs_ref['cosine']:.7f} | unscanned vs HF: "
        f"max|diff|={unscan_vs_ref['max_abs']:.3e} rel={unscan_vs_ref['rel']:.3e} | scanned vs unscanned: "
        f"max|diff|={vs_unscan['max_abs']:.3e} rel={vs_unscan['rel']:.3e} cosine={vs_unscan['cosine']:.9f}",
        flush=True,
    )

    self.assertEqual(logits_scan.shape, logits_pt.shape)
    self.assertTrue(np.isfinite(logits_scan).all())
    self.assertLess(vs_ref["rel"], _LOGITS_REL_TOL, f"scanned vs HF: {vs_ref}")
    np.testing.assert_array_equal(logits_scan.argmax(-1), logits_pt.argmax(-1), err_msg=f"top-1 mismatch {vs_ref}")
    self.assertLess(vs_unscan["rel"], _SCAN_VS_UNSCAN_REL_TOL, f"scanned vs unscanned: {vs_unscan}")
    np.testing.assert_array_equal(logits_scan.argmax(-1), logits_unscan.argmax(-1))


if __name__ == "__main__":
  unittest.main()
