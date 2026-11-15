# Copyright 2026 Google LLC
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

"""Guards the MoE `wo` zero-pad behaviour of the RL rollout weight sync.

RL syncs trainer weights into the vLLM rollout through Tunix's
`_align_per_axis`, which zero-pads MoE MLP weights and repeats everything
else. Tunix's key set omits `wo`, so a `wo` whose intermediate dim the rollout
padded for GMM_v2 took the repeat branch and crashed the job (gemma4-26b
704 -> 1024, qwen3-30b-a3b 768 -> 1024). `maxtext_vllm_rollout` closes the gap
at import time.

These tests assert the resulting *behaviour*, not who fixed it, so they keep
passing once the fix lands upstream in Tunix and the workaround is deleted --
but they fail if the workaround is deleted while Tunix is still missing `wo`.
"""

import os

# Must precede the first JAX import; `_align_per_axis` jits its pad helper.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest  # pylint: disable=wrong-import-position
from types import SimpleNamespace  # pylint: disable=wrong-import-position
from unittest import mock  # pylint: disable=wrong-import-position

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

# Imported for its side effect: the module patches Tunix's MoE key set at
# import time, exactly as it does in production via train_rl.py.
from maxtext.integration.vllm import maxtext_vllm_rollout  # pylint: disable=wrong-import-position
from maxtext.integration.vllm.convert_utils import MOE_MLP_WEIGHTS  # pylint: disable=wrong-import-position
from tunix.generate import utils as tunix_utils  # pylint: disable=wrong-import-position

pytestmark = pytest.mark.post_training

# The production shapes at 1/64 scale. What matters is that the pad is not an
# integer multiple of the source dim (1024 % 704 != 0, like 16 % 11 != 0), so
# the repeat branch cannot express it and raises instead of silently
# double-counting.
EXPERTS = 2
EMB = 4
SRC_MOE_DIM = 11
TGT_MOE_DIM = 16


def _wo(dim):
  """A `wo`-shaped ramp: (experts, moe_intermediate, emb)."""
  size = EXPERTS * dim * EMB
  return np.arange(size, dtype=np.float32).reshape(EXPERTS, dim, EMB)


class TunixMoeZeroPadTest(unittest.TestCase):
  """`wo` must reach the rollout zero-padded on its contracting axis."""

  @pytest.mark.cpu_only
  def test_padded_wo_is_zero_padded_not_repeated(self):
    src = _wo(SRC_MOE_DIM)
    tgt_shape = (EXPERTS, TGT_MOE_DIM, EMB)

    out = np.asarray(
        tunix_utils._align_per_axis(  # pylint: disable=protected-access
            src, tgt_shape, None, "decoder.layers_0.moe_block.wo"
        )
    )

    self.assertEqual(out.shape, tgt_shape)
    np.testing.assert_array_equal(out[:, :SRC_MOE_DIM, :], src)
    np.testing.assert_array_equal(out[:, SRC_MOE_DIM:, :], np.zeros((EXPERTS, TGT_MOE_DIM - SRC_MOE_DIM, EMB)))

  @pytest.mark.cpu_only
  def test_non_moe_key_still_takes_the_repeat_branch(self):
    """Negative control: the zero-pad above comes from the key, not the shape.

    Without it, a test that only checked `wo` could pass because Tunix started
    zero-padding everything.
    """
    with self.assertRaises(tunix_utils.ShapeMismatchError):
      tunix_utils._align_per_axis(  # pylint: disable=protected-access
          _wo(SRC_MOE_DIM), (EXPERTS, TGT_MOE_DIM, EMB), None, "decoder.layers_0.self_attention.out"
      )

  @pytest.mark.cpu_only
  def test_tunix_recognizes_every_maxtext_moe_weight(self):
    """Cheap, readable restatement of the invariant the two tests above probe."""
    self.assertTrue(
        MOE_MLP_WEIGHTS.issubset(tunix_utils._MOE_MLP_WEIGHTS),  # pylint: disable=protected-access
        f"Tunix zero-pads {sorted(tunix_utils._MOE_MLP_WEIGHTS)}, "  # pylint: disable=protected-access
        f"missing {sorted(MOE_MLP_WEIGHTS - tunix_utils._MOE_MLP_WEIGHTS)}",  # pylint: disable=protected-access
    )

  @pytest.mark.cpu_only
  def test_prepare_direct_sync_enables_prefuse_for_tp1_moe_and_honors_opt_out(self):
    """TP=1 MoE defaults prefuse_moe_weights=True, respects explicit False, and TP>1 forces True."""
    out_tp1 = maxtext_vllm_rollout.prepare_direct_sync_additional_config(
        None, direct_maxtext_sync=True, num_experts=256, tensor_parallel_size=1
    )
    self.assertTrue(out_tp1["maxtext_config"]["prefuse_moe_weights"])

    out_tp1_opt_out = maxtext_vllm_rollout.prepare_direct_sync_additional_config(
        {"maxtext_config": {"prefuse_moe_weights": False}},
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=1,
    )
    self.assertFalse(out_tp1_opt_out["maxtext_config"]["prefuse_moe_weights"])

    out_tp4_forced = maxtext_vllm_rollout.prepare_direct_sync_additional_config(
        {"maxtext_config": {"prefuse_moe_weights": False}},
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=4,
    )
    self.assertTrue(out_tp4_forced["maxtext_config"]["prefuse_moe_weights"])

  @pytest.mark.cpu_only
  def test_delete_target_buffers_preserves_pathways_aliased_and_origin_tracked_buffers(self):
    """When unsafe_buffer_pointer is unavailable (Pathways), direct and origin-tracked aliases are preserved."""

    class _FakeArray:

      def __init__(self):
        self.shape = (4,)
        self.deleted = False

      def is_deleted(self):
        return self.deleted

      def delete(self):
        self.deleted = True

    src_direct = _FakeArray()
    src_origin = _FakeArray()
    tgt_origin = _FakeArray()
    src_updated = _FakeArray()
    tgt_stale = _FakeArray()
    old_src = _FakeArray()

    maxtext_vllm_rollout._SYNCED_TGT_TO_SRC_ID[id(tgt_origin)] = id(src_origin)  # pylint: disable=protected-access
    maxtext_vllm_rollout._SYNCED_TGT_TO_SRC_ID[id(tgt_stale)] = id(old_src)  # pylint: disable=protected-access
    try:
      with mock.patch.object(tunix_utils, "_collect_src_buffer_ids", return_value=None):
        tunix_utils._delete_target_buffers(  # pylint: disable=protected-access
            spec_flat={("a",): src_direct, ("b",): tgt_origin, ("c",): tgt_stale},
            src_flat={("a",): src_direct, ("b",): src_origin, ("c",): src_updated},
        )
      self.assertFalse(src_direct.deleted)
      self.assertFalse(tgt_origin.deleted)
      self.assertTrue(tgt_stale.deleted)
    finally:
      maxtext_vllm_rollout._SYNCED_TGT_TO_SRC_ID.pop(id(tgt_origin), None)  # pylint: disable=protected-access
      maxtext_vllm_rollout._SYNCED_TGT_TO_SRC_ID.pop(id(tgt_stale), None)  # pylint: disable=protected-access

  @pytest.mark.cpu_only
  def test_cond_apply_sampling_transforms_exact_parity_and_cond_gating(self):
    """Patched _apply_sampling_transforms matches reference bitwise across all regimes and gates via lax.cond."""

    def _ref_topk_mask(logits, top_k, replace_val=-1e12):
      # Simple deterministic mask keeping entries >= kth largest per row.
      sorted_vals = jnp.sort(logits, axis=-1)[:, ::-1]
      idx = jnp.clip(top_k - 1, 0, logits.shape[-1] - 1)
      thresh = sorted_vals[jnp.arange(logits.shape[0]), idx][:, None]
      return jnp.where(logits >= thresh, logits, replace_val)

    def _ref_topp_mask(logits, top_p, replace_val=-1e12):
      probs = jax.nn.softmax(logits, axis=-1)
      order = jnp.argsort(-probs, axis=-1)
      sorted_probs = jnp.take_along_axis(probs, order, axis=-1)
      cumprobs = jnp.cumsum(sorted_probs, axis=-1)
      keep_sorted = (cumprobs - sorted_probs) < top_p[:, None]
      inv_order = jnp.argsort(order, axis=-1)
      keep = jnp.take_along_axis(keep_sorted, inv_order, axis=-1)
      return jnp.where(keep, logits, replace_val)

    def _orig_transforms(logits, md):
      if not md.do_sampling:
        return logits
      logits = logits / md.temperature.astype(logits.dtype)[:, None]
      logits = jnp.where((md.top_k > 0)[:, None], _ref_topk_mask(logits, md.top_k, replace_val=-1e12), logits)
      logits = jnp.where((md.top_p < 1.0)[:, None], _ref_topp_mask(logits, md.top_p, replace_val=-1e12), logits)
      return logits

    fake_mod = SimpleNamespace(
        _apply_sampling_transforms=_orig_transforms,
        topk_mask=_ref_topk_mask,
        topp_mask=_ref_topp_mask,
    )
    self.assertTrue(maxtext_vllm_rollout._patch_tpu_inference_sampling_transforms(fake_mod))  # pylint: disable=protected-access
    self.assertFalse(maxtext_vllm_rollout._patch_tpu_inference_sampling_transforms(fake_mod))  # pylint: disable=protected-access

    logits = jax.random.normal(jax.random.PRNGKey(0), (4, 16), dtype=jnp.float32)
    for do_sampling, top_k, top_p in [
        (False, [-1, -1, -1, -1], [1.0, 1.0, 1.0, 1.0]),
        (True, [-1, -1, -1, -1], [1.0, 1.0, 1.0, 1.0]),
        (True, [4, 4, 4, 4], [1.0, 1.0, 1.0, 1.0]),
        (True, [-1, -1, -1, -1], [0.8, 0.8, 0.8, 0.8]),
        (True, [5, -1, 3, 0], [1.0, 0.7, 0.9, 1.0]),
    ]:
      md = SimpleNamespace(
          do_sampling=do_sampling,
          temperature=jnp.array([0.9, 0.7, 1.2, 0.8], dtype=jnp.float32),
          top_k=jnp.array(top_k, dtype=jnp.int32),
          top_p=jnp.array(top_p, dtype=jnp.float32),
      )
      ref_out = _orig_transforms(logits, md)
      got_out = jax.jit(lambda x, t, k, p: fake_mod._apply_sampling_transforms(  # pylint: disable=protected-access
          x, SimpleNamespace(do_sampling=do_sampling, temperature=t, top_k=k, top_p=p)
      ))(logits, md.temperature, md.top_k, md.top_p)
      np.testing.assert_array_equal(np.asarray(got_out), np.asarray(ref_out))

    jaxpr_text = str(
        jax.make_jaxpr(
            lambda x, t, k, p: fake_mod._apply_sampling_transforms(  # pylint: disable=protected-access
                x, SimpleNamespace(do_sampling=True, temperature=t, top_k=k, top_p=p)
            )
        )(logits, jnp.ones((4,), jnp.float32), jnp.full((4,), -1, jnp.int32), jnp.ones((4,), jnp.float32))
    )
    self.assertIn("cond[", jaxpr_text)

  @pytest.mark.cpu_only
  def test_sampler_update_params_skips_redundant_sync_when_source_buffers_unchanged(self):
    """Repeated update_params with identical source buffers (e.g. Pre-RL eval after init) is a safe no-op."""
    sampler = object.__new__(maxtext_vllm_rollout.MaxTextVllmSampler)
    sampler._direct_maxtext_sync = True  # pylint: disable=protected-access
    sampler._is_gemma = False  # pylint: disable=protected-access
    sampler._last_synced_src_ids = None  # pylint: disable=protected-access
    sampler._last_synced_tgt_ids = None  # pylint: disable=protected-access
    sampler._tracked_tgt_ids = []  # pylint: disable=protected-access

    w0 = jnp.ones((2, 2), dtype=jnp.float32)
    w1 = jnp.ones((2, 2), dtype=jnp.float32) * 2.0
    state_holder = {"model": {"w": jnp.zeros((2, 2), dtype=jnp.float32)}}
    calls = []

    def _fake_super_update(_self, updated_weights, _filter_types=None):
      calls.append(updated_weights)
      state_holder["model"]["w"] = jnp.array(updated_weights["base"]["w"])

    with (
        mock.patch.object(
            maxtext_vllm_rollout.VllmSampler, "transformer_state", new_callable=mock.PropertyMock, return_value=state_holder
        ),
        mock.patch.object(maxtext_vllm_rollout.VllmSampler, "update_params", _fake_super_update),
    ):
      sampler.update_params({"base": {"w": w0}})
      self.assertEqual(len(calls), 1)
      # Second sync with exact same unchanged buffer object must short-circuit cleanly.
      sampler.update_params({"base": {"w": w0}})
      self.assertEqual(len(calls), 1)
      # New buffer object (post-step or restored checkpoint) must trigger full sync.
      sampler.update_params({"base": {"w": w1}})
      self.assertEqual(len(calls), 2)


if __name__ == "__main__":
  unittest.main()
