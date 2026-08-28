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

"""Tests for context parallelism in the Gated Delta Net (Qwen3-Next / Qwen3.5)."""

import functools
import re
import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models.qwen3 import (
    Qwen3NextGatedDeltaNet,
    jax_chunk_gated_delta_rule,
    jax_chunk_gated_delta_rule_cp,
)
from maxtext.utils import maxtext_utils

from tests.utils.test_helpers import get_test_config_path


def _layer_forward(layer, x):
  """Module-level so `jax.jit` does not close over a loop variable."""
  return layer(x, model_mode=MODEL_MODE_TRAIN)[0]


def _cp_size() -> int:
  """Largest power-of-two context-parallel degree the local topology supports."""
  n = jax.device_count()
  size = 1
  while size * 2 <= n:
    size *= 2
  return size


@pytest.mark.tpu_only
class GatedDeltaRuleContextParallelKernelTest(unittest.TestCase):
  """`jax_chunk_gated_delta_rule_cp` must match the single-device recurrence."""

  B, S, H, K, V, C = 2, 512, 4, 128, 128, 64

  def setUp(self):
    super().setUp()
    self.cp = _cp_size()
    if self.cp < 2:
      self.skipTest("context parallelism needs at least 2 devices")
    rng = np.random.default_rng(0)

    def mk(*shape):
      return jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)

    self.q = mk(self.B, self.S, self.H, self.K)
    self.k = mk(self.B, self.S, self.H, self.K)
    self.v = mk(self.B, self.S, self.H, self.V)
    # `g` is a log-decay (negative) and `beta` a sigmoid output, matching the layer.
    self.g = -jnp.exp(mk(self.B, self.S, self.H) * 0.3) * 0.05
    self.beta = jax.nn.sigmoid(mk(self.B, self.S, self.H))
    self.h0 = jnp.zeros((self.B, self.H, self.K, self.V), jnp.float32)
    self.mesh = Mesh(np.array(jax.devices()[: self.cp]).reshape(self.cp), ("context",))

  def _cp_call(self, q, k, v, g, beta, h0):
    """Runs the context-parallel kernel under a `shard_map` over the `context` axis."""
    fn = jax.shard_map(
        lambda *a: jax_chunk_gated_delta_rule_cp(
            *a,
            cp_axis_name="context",
            chunk_size=self.C,
            compute_dtype=jnp.float32,
            use_qk_norm_in_gdn=True,
        ),
        mesh=self.mesh,
        in_specs=(
            P(None, "context", None, None),
            P(None, "context", None, None),
            P(None, "context", None, None),
            P(None, "context", None),
            P(None, "context", None),
            P(),
        ),
        out_specs=(P(None, "context", None, None), P()),
        check_vma=False,
    )
    return fn(q, k, v, g, beta, h0)

  def _ref_call(self, q, k, v, g, beta, h0):
    return jax_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=self.C, initial_state=h0, use_qk_norm_in_gdn=True, compute_dtype=jnp.float32
    )

  def test_forward_matches_single_device(self):
    ref_o, ref_h = jax.jit(self._ref_call)(self.q, self.k, self.v, self.g, self.beta, self.h0)
    with jax.set_mesh(self.mesh):
      cp_o, cp_h = jax.jit(self._cp_call)(self.q, self.k, self.v, self.g, self.beta, self.h0)
    np.testing.assert_allclose(cp_o, ref_o, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(cp_h, ref_h, rtol=2e-4, atol=2e-4)

  def test_backward_matches_single_device(self):
    def loss(fn, *args):
      o, _ = fn(*args, self.h0)
      return jnp.sum(o * jnp.sin(o))

    args = (self.q, self.k, self.v, self.g, self.beta)
    with jax.set_mesh(self.mesh):
      ref_g = jax.grad(functools.partial(loss, self._ref_call), argnums=(0, 1, 2, 3, 4))(*args)
      cp_g = jax.grad(functools.partial(loss, self._cp_call), argnums=(0, 1, 2, 3, 4))(*args)
    for name, a, b in zip("query key value g beta".split(), ref_g, cp_g):
      np.testing.assert_allclose(b, a, rtol=5e-3, atol=5e-4, err_msg=f"gradient mismatch for {name}")

  def test_cross_rank_payload_is_sequence_length_independent(self):
    """The only cross-rank tensors are the (M, U) affine maps, sized by head dims.

    This is the whole point of the scheme: softmax-attention context parallelism
    exchanges O(sequence_length) key/value tensors, whereas the delta rule collapses
    each rank's shard into a fixed-size affine map.
    """
    # Matches `%foo = f32[8,4,128,128]{...} all-gather(...)`, ignoring bitcast aliases.
    pattern = re.compile(r"=\s*\w+\[([0-9,]+)\][^=]*\ball-gather(?:-start)?\(")
    sizes = {}
    for repeat in (1, 2, 4):
      tiled = [jnp.concatenate([x] * repeat, axis=1) for x in (self.q, self.k, self.v, self.g, self.beta)]
      with jax.set_mesh(self.mesh):
        hlo = jax.jit(self._cp_call).lower(*tiled, self.h0).compile().as_text()
      total = 0
      for line in hlo.splitlines():
        match = pattern.search(line)
        if match:
          total += int(np.prod([int(d) for d in match.group(1).split(",")]))
      self.assertGreater(total, 0, "expected the cross-rank prefix exchange to appear in the HLO")
      sizes[repeat * self.S] = total
    self.assertEqual(
        len(set(sizes.values())),
        1,
        f"cross-rank all-gather volume must not grow with sequence length, got {sizes}",
    )
    # cp * batch * heads * k_dim * (k_dim for M + v_dim for U), in elements.
    expected = self.cp * self.B * self.H * self.K * (self.K + self.V)
    self.assertEqual(next(iter(sizes.values())), expected)


@pytest.mark.tpu_only
class GatedDeltaNetLayerContextParallelTest(unittest.TestCase):
  """The full GDN layer must be invariant to the context-parallel degree."""

  def setUp(self):
    super().setUp()
    self.cp = _cp_size()
    if self.cp < 2:
      self.skipTest("context parallelism needs at least 2 devices")

  def _build(self, cp_degree):
    """Builds a small GDN layer on a mesh with the requested context-parallel degree."""
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="gdn_cp_test",
        enable_checkpointing=False,
        max_prefill_predict_length=32,
        max_target_length=256,
        base_emb_dim=128,
        gdn_num_value_heads=4,
        gdn_num_key_heads=4,
        gdn_key_head_dim=32,
        gdn_value_head_dim=32,
        gdn_conv_kernel_dim=4,
        gdn_chunk_size=16,
        ici_context_parallelism=cp_degree,
        context_parallel_load_balance=False,
        dtype="float32",
        weight_dtype="float32",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    layer = Qwen3NextGatedDeltaNet(
        config=cfg,
        inputs_shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim),
        mesh=mesh,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )
    return cfg, mesh, layer

  def test_layer_output_matches_without_context_parallelism(self):
    outs = {}
    for cp_degree in (1, self.cp):
      cfg, mesh, layer = self._build(cp_degree)
      lnx = jax.random.normal(
          jax.random.PRNGKey(7),
          (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim),
          dtype=jnp.float32,
      )
      forward = functools.partial(_layer_forward, layer)
      with jax.set_mesh(mesh):
        out = jax.jit(forward)(lnx)
      outs[cp_degree] = np.asarray(out)
    np.testing.assert_allclose(outs[self.cp], outs[1], rtol=2e-4, atol=2e-4)

  def test_load_balancing_is_rejected(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="gdn_cp_lb_test",
        enable_checkpointing=False,
        max_target_length=256,
        base_emb_dim=128,
        gdn_num_value_heads=4,
        gdn_num_key_heads=4,
        gdn_key_head_dim=32,
        gdn_value_head_dim=32,
        gdn_chunk_size=16,
        ici_context_parallelism=self.cp,
        context_parallel_load_balance=True,
        dtype="float32",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    layer = Qwen3NextGatedDeltaNet(
        config=cfg,
        inputs_shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim),
        mesh=mesh,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )
    lnx = jnp.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim), jnp.float32)
    with self.assertRaisesRegex(ValueError, "context_parallel_load_balance"):
      with jax.set_mesh(mesh):
        layer(lnx, model_mode=MODEL_MODE_TRAIN)


if __name__ == "__main__":
  unittest.main()
