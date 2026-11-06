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

"""Training-FLOPs tests for Qwen3.8-Flash-Next (`m3/models/qwen3_8_flash_next/qwen3_8_flops.py`).

The goldens are anchored outside the implementation: active parameters are counted from the
instantiated NNX module tree, attention FLOPs are derived from first principles, and the
non-embedding active-parameter count is checked against the published model card.
"""

from typing import Any

from absl.testing import absltest
from flax import nnx
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.m3.models.qwen3_8_flash_next import qwen3_8_flops
from maxtext.utils import maxtext_utils
from tests.utils import test_helpers
import numpy as np

# Non-embedding activated params (billions) from the model card ("180B total / 6B activated");
# excludes token embedding, lm_head, and the MTP draft block.
PUBLISHED_ACTIVE_PARAMS_B = 6.0


def _initialize_model_config(model_name: str, **overrides: Any):
  """Initializes a config via `base.yml model_name=...`, as a real run does.

  m3 dispatch (`MODEL_REGISTRY`, `MODEL_FLOPS_REGISTRY`) keys on `config.model_name`, which stays at its
  default when a model YAML is loaded directly as the config file.
  """
  config_args = [None, test_helpers.get_test_config_path(), f"model_name={model_name}", "run_name=test"]
  config_args.append("skip_jax_distributed_system=True")
  return pyconfig.initialize(config_args, enable_checkpointing=False, override_model_config=True, **overrides)


def _count_active_params(cfg) -> int:
  """Active (per-token) parameters from the instantiated NNX module tree, independent of the FLOPs code.

  Counts every nnx.Param leaf of the abstract model except embedding-lookup tables (token_embedder,
  PLE ngram_embedding -- gathers, 0 dense FLOPs), and counts routed-expert tensors at the activated
  fraction num_experts_per_tok / num_experts. Norm scales, A_log, dt_bias are included (negligible).
  """
  model = nnx.eval_shape(
      lambda: modeling_qwen3_8_flash_next.create_qwen3_8_flash_next_model(cfg, rngs=nnx.Rngs(params=0))
  )
  dense, routed = 0, 0
  for path, leaf in nnx.state(model, nnx.Param).flat_state():
    keys = tuple(str(getattr(k, "key", k)) for k in path)
    shape = leaf.value.shape if hasattr(leaf, "value") else leaf.shape
    n = int(np.prod(shape))
    if "token_embedder" in keys or "ngram_embedding" in keys:
      continue
    if "experts" in keys:
      routed += n
    else:
      dense += n
  return dense + routed * cfg.num_experts_per_tok // cfg.num_experts


def _golden_attn_tflops(cfg) -> float:
  """Non-weight (attention) training TFLOPs from first principles, independent of the FLOPs code.

  Per forward pass:
  - GDN layers: the chunked delta rule touches a [D_k, D_v] state per value head four times per token
    (KK^T, VK^T, gated state update, SQ), 2 FLOPs each -> 8 * H_v * D_k * D_v per token.
  - QSA layers: each query attends to min(i + 1, budget) keys (causal prefix until the top-k budget is hit);
    QK^T and PV cost 2 FLOPs per attended pair per head per head dim. The indexer scores every query against
    S / block_size compressed key blocks (causal half) and reduces over its heads.
  - PLE: elementwise gated mixing over the 4-stream hidden state.
  Training multiplies the forward count by 3 (forward + backward).
  """
  B, S = cfg.per_device_batch_size, cfg.max_target_length
  num_qsa_layers = cfg.num_decoder_layers // cfg.inhomogeneous_layer_cycle_interval
  num_gdn_layers = cfg.num_decoder_layers - num_qsa_layers
  gdn_core = 8 * B * S * cfg.gdn_num_value_heads * cfg.gdn_key_head_dim * cfg.gdn_value_head_dim
  budget = min(cfg.index_select_budget, S)
  attended_pairs = sum(min(i + 1, budget) for i in range(S))
  qsa_sparse = 4 * B * attended_pairs * cfg.num_query_heads * cfg.head_dim
  num_blocks = S / cfg.index_select_block_size
  indexer = B * S * num_blocks * cfg.index_n_heads * (cfg.index_head_dim + 1)
  ple = 2 * B * S * cfg.hc_count * cfg.emb_dim if cfg.use_ple_layer else 0
  # The MTP draft block runs one more QSA layer per MTP step.
  forward = num_gdn_layers * gdn_core + (num_qsa_layers + cfg.mtp_num_layers) * (qsa_sparse + indexer) + ple
  return 3 * forward / 1e12


class Qwen3_8FlopsTest(absltest.TestCase):
  """Anchors the Qwen3.8-Flash-Next FLOPs calculation against implementation-independent goldens."""

  def assertFlopsAlmostEqual(self, flops1, flops2, rel_tol=5e-2):
    """Asserts that two FLOPs values are equal within 5% relative tolerance."""
    self.assertLessEqual(
        abs(flops1 - flops2) / max(abs(flops1), abs(flops2)),
        rel_tol,
        f"FLOPs values are not equal: {flops1} != {flops2} (rel_tol={rel_tol:.2e})",
    )

  def test_qwen3_8_flash_next_flops(self):
    """6*B*S*P weight check (P from the module tree) plus a first-principles attention check on the full config."""
    for mtp_num_layers in (0, 1):
      with self.subTest(mtp_num_layers=mtp_num_layers):
        cfg = _initialize_model_config(
            "qwen3.8-flash-next",
            max_target_length=4096,
            per_device_batch_size=1,
            mtp_num_layers=mtp_num_layers,
        )
        B, S = cfg.per_device_batch_size, cfg.max_target_length
        active_params = _count_active_params(cfg)
        # The MTP draft block re-runs the shared lm_head on its own hidden states; the weights are counted once in P
        # but the [B*S, E] x [E, V] matmul executes once per MTP layer in addition to the main head.
        extra_unembed_params = mtp_num_layers * cfg.vocab_size * cfg.emb_dim
        golden_weight_tflops = 6 * B * S * (active_params + extra_unembed_params) / 1e12
        golden_attn_tflops = _golden_attn_tflops(cfg)
        total, weight, attn = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
        self.assertFlopsAlmostEqual(weight, golden_weight_tflops)
        self.assertFlopsAlmostEqual(attn, golden_attn_tflops)
        self.assertFlopsAlmostEqual(total, golden_weight_tflops + golden_attn_tflops)

  def test_qwen3_8_flash_next_published_active_params(self):
    """Anchor Qwen3.8-Flash-Next non-embedding active params against the published 6.0B model card."""
    cfg = _initialize_model_config(
        "qwen3.8-flash-next",
        max_target_length=1,
        per_device_batch_size=1,
        mtp_num_layers=0,
    )
    _, weight_tflops, _ = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    non_lookup_params = weight_tflops * 1e12 / 6  # B=S=1 -> 6 * params
    # Drop lm_head (token-embedding lookup is already 0 FLOPs).
    non_embedding_params = non_lookup_params - cfg.vocab_size * cfg.emb_dim
    expected = PUBLISHED_ACTIVE_PARAMS_B * 1e9
    self.assertAlmostEqual(non_embedding_params, expected, delta=0.02 * expected)

  def test_qwen3_8_flash_next_uses_m3_flops_registry(self):
    """Both flash-next names route `calculate_tflops_training_per_device` through the m3 FLOPs registry."""
    for model_name in ("qwen3.8-flash-next", "qwen3.8-flash-next-mini"):
      with self.subTest(model_name=model_name):
        self.assertIs(
            m3_models.MODEL_FLOPS_REGISTRY[model_name], qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device
        )
        cfg = _initialize_model_config(model_name, max_target_length=64, per_device_batch_size=1)
        self.assertEqual(
            maxtext_utils.calculate_tflops_training_per_device(cfg, log=False),
            qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device(cfg, log=False),
        )


if __name__ == "__main__":
  absltest.main()
