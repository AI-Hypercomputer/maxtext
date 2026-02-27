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

"""Tests for QK-Clip utilities."""

import unittest
import sys
import pytest
from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.layers import attention_mla
from maxtext.utils import maxtext_utils
from maxtext.utils.qk_clip_utils import apply_qk_clip, calculate_max_logit_metric

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path


class QKClipTest(unittest.TestCase):

  def _get_config_and_state(self, threshold, nope_dim, params_dict, attention_type="mla"):
    """Helper to create mock Config and State objects."""
    Config = namedtuple("Config", ["qk_clip_threshold", "qk_nope_head_dim", "attention_type"])
    config = Config(qk_clip_threshold=threshold, qk_nope_head_dim=nope_dim, attention_type=attention_type)

    State = namedtuple("State", ["params", "replace"])
    state = State(params=params_dict, replace=lambda params: State(params, None))
    return config, state

  def test_raises_error_for_non_mla(self):
    """Verifies that non-MLA attention types raise ValueError."""
    params = {}  # Params don't matter for this check
    config, state = self._get_config_and_state(
        threshold=10.0, nope_dim=4, params_dict=params, attention_type="dot_product"
    )
    intermediates = {}

    with self.assertRaisesRegex(ValueError, "QK-Clip is only supported for MLA attention"):
      apply_qk_clip(state, intermediates, config)

  def test_apply_qk_clip_logic(self):
    """Tests QK Clip math and application logic on CPU with random weights."""
    # 1. Setup Mock Data with random weights
    rng = jax.random.PRNGKey(0)
    rng_q, rng_kv = jax.random.split(rng)
    wq_b = jax.random.normal(rng_q, (2, 2, 6))  # [rank, heads, dim]
    wkv_b = jax.random.normal(rng_kv, (2, 2, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}, "wkv_b": {"kernel": wkv_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # 2. Setup Mock Intermediates
    # Head 0: max_logit = 20.0 (>10.0) -> Gamma = 0.5
    # Head 1: max_logit = 5.0  (<10.0) -> Gamma = 1.0
    max_logits = jnp.array([[20.0, 5.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    # 3. Run Apply Clip
    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]
    new_wkv = new_state.params["decoder"]["layers_0"]["self_attention"]["wkv_b"]["kernel"]

    # 4. Verify Results against original random weights
    # Head 0: Scale = 0.5. W_c * sqrt(0.5), W_r * 0.5
    self.assertTrue(jnp.allclose(new_wq[:, 0, :4], wq_b[:, 0, :4] * jnp.sqrt(0.5)))
    self.assertTrue(jnp.allclose(new_wq[:, 0, 4:], wq_b[:, 0, 4:] * 0.5))
    self.assertTrue(jnp.allclose(new_wkv[:, 0, :4], wkv_b[:, 0, :4] * jnp.sqrt(0.5)))

    # Head 1: Scale = 1.0. No change.
    self.assertTrue(jnp.allclose(new_wq[:, 1, :], wq_b[:, 1, :]))
    self.assertTrue(jnp.allclose(new_wkv[:, 1, :], wkv_b[:, 1, :]))

  def test_verify_per_head_clipping(self):
    """Explicitly verifies that clipping is applied independently per head."""
    # Setup 3 Heads:
    # Head 0: 40.0 (Way above threshold) -> Scale 0.25
    # Head 1: 9.9  (Just below threshold) -> Scale 1.0
    # Head 2: 1.0  (Way below threshold)  -> Scale 1.0
    rng = jax.random.PRNGKey(1)
    wq_b = jax.random.normal(rng, (1, 3, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    max_logits = jnp.array([[40.0, 9.9, 1.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    # Head 0: Scale 0.25. W_qc = original * sqrt(0.25)
    self.assertTrue(jnp.allclose(new_wq[0, 0, 0], wq_b[0, 0, 0] * jnp.sqrt(0.25)))
    # Head 1: Unchanged
    self.assertTrue(jnp.allclose(new_wq[0, 1, 0], wq_b[0, 1, 0]))
    # Head 2: Unchanged
    self.assertTrue(jnp.allclose(new_wq[0, 2, 0], wq_b[0, 2, 0]))

  def test_no_clipping_when_below_threshold(self):
    """Verifies that weights are unchanged when max_logits < tau."""
    rng = jax.random.PRNGKey(2)
    wq_b = jax.random.normal(rng, (2, 1, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=100.0, nope_dim=4, params_dict=params)

    # Max logits = 50.0 (Below threshold 100.0)
    max_logits = jnp.array([[50.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    # Assert exact equality
    self.assertTrue(jnp.array_equal(new_wq, wq_b))

  def test_shared_keys_are_untouched(self):
    """Verifies that wkv_a (Shared Key) is ignored by the clipper."""
    rng = jax.random.PRNGKey(3)
    rng_a, rng_b = jax.random.split(rng)
    wkv_a = jax.random.normal(rng_a, (2, 1, 6))
    wkv_b = jax.random.normal(rng_b, (2, 1, 6))
    params = {
        "decoder": {
            "layers_0": {
                "self_attention": {
                    "wkv_a": {"kernel": wkv_a},  # Should be ignored
                    "wkv_b": {"kernel": wkv_b},  # Should be clipped
                }
            }
        }
    }
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # Trigger clipping with high logits
    max_logits = jnp.array([[100.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wkv_a = new_state.params["decoder"]["layers_0"]["self_attention"]["wkv_a"]["kernel"]

    # Assert wkv_a is completely unchanged
    self.assertTrue(jnp.array_equal(new_wkv_a, wkv_a))

  def test_resilience_to_missing_stats(self):
    """Verifies that code handles layers without max_logits gracefully."""
    rng = jax.random.PRNGKey(4)
    wq_b = jax.random.normal(rng, (2, 1, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # Intermediates dict is empty
    intermediates = {}

    # Should not crash, should return original params
    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    self.assertTrue(jnp.array_equal(new_wq, wq_b))


class MaxLogitsMetricTest(unittest.TestCase):
  """Tests for the max logit calculation functions."""

  def test_dot_product_max_logits(self):
    """Verifies max_logits calculation logic used in AttentionOp."""
    # [Batch=1, Heads=1, Len_Q=2, Dim=4]
    q = jnp.array([[[[10.0, 0, 0, 0], [0, 10.0, 0, 0]]]])
    k = jnp.array([[[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]]])

    # Standard Einsum for dot product attention
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    # Max logits over Q and K dimensions
    computed_max = jnp.max(logits, axis=(-2, -1))

    self.assertEqual(computed_max.shape, (1, 1))
    self.assertEqual(computed_max[0, 0], 10.0)

  def test_calculate_max_logit_metric(self):
    """Verifies global max logit calculation across multiple layers."""
    # Simulating a batch size of 1 with 2 heads
    # Layer 0: Max logit across heads/batch is 20.0
    layer_0_logits = (jnp.array([[10.0, 20.0]]),)
    # Layer 1: Max logit across heads/batch is 50.0
    layer_1_logits = (jnp.array([[5.0, 50.0]]),)

    intermediate_outputs = {
        "decoder": {
            "layers_0": {"self_attention": {"max_logits": layer_0_logits}},
            "layers_1": {"self_attention": {"max_logits": layer_1_logits}},
            # Layer 2 has no stats, logic should handle this
            "layers_2": {"self_attention": {}},
        }
    }

    global_max = calculate_max_logit_metric(intermediate_outputs)
    self.assertEqual(global_max, 50.0)

  def test_calculate_max_logit_metric_empty(self):
    """Verifies behavior when no logits are present in intermediates."""
    intermediate_outputs = {"decoder": {"layers_0": {}}}
    global_max = calculate_max_logit_metric(intermediate_outputs)
    self.assertIsNone(global_max)


class QKClipMLATest(unittest.TestCase):
  """End-to-End integration tests using the actual AttentionMLA layer."""

  # Config for MLA with QK-Clip enabled
  config_arguments = {
      "run_name": "test_qk_clip",
      "model_name": "default",
      "metrics_file": "",
      "base_output_directory": "",
      # MLA Architecture Overrides
      "attention_type": "mla",
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "qk_rope_head_dim": 8,
      "v_head_dim": 32,
      "num_query_heads": 4,
      # MLA implementation requires equal heads
      "num_kv_heads": 4,
      # FIX: Ensure emb_dim is set here so it propagates during init
      "emb_dim": 128,  # Matches: 4 heads * 32 v_dim = 128
      "base_emb_dim": 128,
      "max_target_length": 128,
      "dropout_rate": 0.0,
      # Block sizes for Splash/Flash attention
      # Must be 128 to satisfy Tokamax bkv_compute requirements
      "sa_block_q": 128,
      "sa_block_kv": 128,
      "sa_block_kv_compute": 128,
      "sa_block_q_dkv": 128,
      "sa_block_kv_dkv": 128,
      "sa_block_kv_dkv_compute": 128,
      # QK Clip Settings
      "use_qk_clip": True,
      "qk_clip_threshold": 10.0,
      # Minimal Training Args
      "per_device_batch_size": 1.0,
      "steps": 1,
  }

  def setUp(self):
    """Initializes the configuration for each test using pyconfig."""
    super().setUp()
    if not is_decoupled():
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)

    args = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(args, **self.config_arguments)

  def _run_mla_pass(self, config_overrides=None):
    """Initializes MLA, runs forward pass, and returns variables & intermediates."""
    run_config_args = self.config_arguments.copy()
    if config_overrides:
      run_config_args.update(config_overrides)

    # Ensure emb_dim is enforced in args before init (config object is immutable)
    run_config_args["emb_dim"] = 128

    args = [sys.argv[0], get_test_config_path()]
    config = pyconfig.initialize(args, **run_config_args)

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    batch_size = jax.device_count()
    seq_len = 128

    # Force alignment: 4 heads * 32 dim = 128
    forced_v_head_dim = 32
    forced_emb_dim = 128

    rngs = nnx.Rngs(0)
    x = jax.random.normal(rngs.params(), (batch_size, seq_len, forced_emb_dim))

    attention_kernel = "flash" if (config.use_jax_splash or config.use_tokamax_splash) else "dot_product"

    inputs_q_shape = x.shape
    inputs_kv_shape = x.shape

    # Initialize MLA
    model = attention_mla.MLA(
        config=config,
        num_query_heads=4,
        num_kv_heads=4,
        # head_dim must match v_head_dim because the base Attention class
        # uses it to initialize the output projection size.
        head_dim=forced_v_head_dim,
        v_head_dim=forced_v_head_dim,
        max_target_length=128,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        mesh=mesh,
        # Explicitly pass LoRA ranks
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        rngs=rngs,
    )

    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))

    # Run forward pass
    # model call returns (output, kv_cache) - it does NOT return intermediates
    _ = model(x, x, positions, None)

    # Retrieve sowed intermediates from the model state
    state = nnx.state(model)
    intermediates = state.to_pure_dict()

    params = nnx.state(model, nnx.Param)

    return params, intermediates, config

  def test_mla_dot_product_integration(self):
    """Verifies standard Dot Product MLA correctly sows logits for clipping."""
    # Ensure splash flags are off
    params, intermediates, config = self._run_mla_pass({"use_jax_splash": False, "use_tokamax_splash": False})

    # Hoist logits for verification if they are buried in attention_op
    logits = intermediates.get("attention_op", {}).get("max_logits", None)
    if logits is None:
      logits = intermediates.get("max_logits", None)

    self.assertIsNotNone(logits, "MLA (Dot Product) failed to sow max_logits in intermediates")

    State = namedtuple("State", ["params", "replace"])

    def replace_fn(params=None, **kwargs):
      return State(params=params, replace=replace_fn)

    train_state = State(params=params, replace=replace_fn)

    apply_qk_clip(train_state, intermediates, config)

  @pytest.mark.tpu_only
  def test_mla_splash_integration(self):
    """Verifies Splash Attention MLA setup correctly sows logits."""
    # Use 'use_tokamax_splash' for stats collection support
    params, intermediates, config = self._run_mla_pass({"use_jax_splash": False, "use_tokamax_splash": True})

    logits = intermediates.get("attention_op", {}).get("max_logits", None)
    self.assertIsNotNone(logits, "MLA (Splash) failed to sow max_logits in intermediates")

    State = namedtuple("State", ["params", "replace"])

    def replace_fn(params=None, **kwargs):
      return State(params=params, replace=replace_fn)

    train_state = State(params=params, replace=replace_fn)

    apply_qk_clip(train_state, intermediates, config)

  def _run_mla_pass_with_inputs(self, config_overrides, kernel_name, x, positions):
    """Helper to run a pass with specific inputs/config."""
    run_config_args = self.config_arguments.copy()
    run_config_args.update(config_overrides)

    # Ensure emb_dim is enforced in args before init
    run_config_args["emb_dim"] = 128

    args = [sys.argv[0], get_test_config_path()]
    config = pyconfig.initialize(args, **run_config_args)

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Initialize
    model = attention_mla.MLA(
        config=config,
        num_query_heads=4,
        num_kv_heads=4,
        head_dim=32,
        v_head_dim=32,
        max_target_length=128,
        attention_kernel=kernel_name,
        inputs_q_shape=x.shape,
        inputs_kv_shape=x.shape,
        mesh=mesh,
        # Explicitly pass LoRA ranks
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        rngs=nnx.Rngs(0),
    )

    # Run
    _ = model(x, x, positions, None)

    # Capture
    state = nnx.state(model)
    intermediates = state.to_pure_dict()
    params = nnx.state(model, nnx.Param)

    return params, intermediates, config

  @pytest.mark.tpu_only
  def test_cross_check_dot_vs_splash(self):
    """Cross-checks Max Logits and QK Clip results between Dot Product and Splash attention."""

    test_threshold = 1.0

    config_overrides = {
        "emb_dim": 128,
        "base_emb_dim": 128,
        "v_head_dim": 32,
        "num_query_heads": 4,
        "num_kv_heads": 4,
        "sa_block_q": 128,
        "sa_block_kv": 128,
        "sa_block_kv_compute": 128,
        "qk_clip_threshold": test_threshold,
        "per_device_batch_size": 1.0,
    }

    batch_size = jax.device_count()
    seq_len = 128
    emb_dim = 128

    rngs = nnx.Rngs(0)
    # Use larger initialization to ensure large logits (so clipping actually triggers)
    x = jax.random.normal(rngs.params(), (batch_size, seq_len, emb_dim)) * 2.0
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))

    # Helper to run a pass with a specific kernel
    def run_pass(kernel_name, use_tokamax):
      overrides = config_overrides.copy()
      overrides["use_jax_splash"] = False
      overrides["use_tokamax_splash"] = use_tokamax

      params, intermediates, cfg = self._run_mla_pass_with_inputs(overrides, kernel_name, x, positions)

      # Extract max_logits from the standalone MLA layer intermediates
      # Correct path for unit test: 'attention_op' submodule
      layer_stats = intermediates["attention_op"]["max_logits"][0]
      return params, layer_stats, cfg

    # 2. Run Dot Product Pass
    params_dot, logits_dot, cfg_dot = run_pass("dot_product", use_tokamax=False)

    # 3. Run Splash (Tokamax) Pass
    params_splash, logits_splash, cfg_splash = run_pass("flash", use_tokamax=True)

    # 4. Compare Max Logits
    self.assertEqual(logits_dot.shape, logits_splash.shape)

    np.testing.assert_allclose(
        logits_dot, logits_splash, rtol=1e-3, atol=1e-2, err_msg="Max Logits differ between Dot Product and Splash!"
    )

    # 5. Perform QK Clip & Compare Weights
    State = namedtuple("State", ["params", "replace"])

    def replace_fn(params=None, **kwargs):
      return State(params=params, replace=replace_fn)

    # Re-run Dot to get full objects for clipping
    _, intermediates_dot, _ = self._run_mla_pass_with_inputs(
        {**config_overrides, "use_tokamax_splash": False}, "dot_product", x, positions
    )

    state_dot = State(params=params_dot, replace=replace_fn)
    new_state_dot = apply_qk_clip(state_dot, intermediates_dot, cfg_dot)

    # Re-run Splash to get full objects for clipping
    _, intermediates_splash, _ = self._run_mla_pass_with_inputs(
        {**config_overrides, "use_tokamax_splash": True}, "flash", x, positions
    )

    state_splash = State(params=params_splash, replace=replace_fn)
    new_state_splash = apply_qk_clip(state_splash, intermediates_splash, cfg_splash)

    # 6. Compare Clipped Weights
    wq_dot = new_state_dot.params["wq_b"]["kernel"]
    wq_splash = new_state_splash.params["wq_b"]["kernel"]

    np.testing.assert_allclose(
        wq_dot,
        wq_splash,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Clipped weights differ! QK Clip logic inconsistent between kernels.",
    )


if __name__ == "__main__":
  unittest.main()
