# Copyright 2025-2026 Google LLC
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

"""Interpret-mode numerics tests for the fused LM-head + cross-entropy kernel."""

import os

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

# pylint: disable=g-import-not-at-top,wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import fused_ce
from maxtext.utils import max_utils


def _inputs(t, d, v, seed=0, mask_frac=0.25):
  k = jax.random.split(jax.random.PRNGKey(seed), 5)
  x = jax.random.normal(k[0], (t, d), jnp.float32).astype(jnp.bfloat16)
  w = jax.random.normal(k[1], (d, v), jnp.float32) * (2.0 / np.sqrt(d))
  targets = jax.random.randint(k[2], (t,), 0, v, jnp.int32)
  weights = (jax.random.uniform(k[3], (t,)) > mask_frac).astype(jnp.float32)
  return x, w, targets, weights


def _round_bf16(a):
  return a.astype(jnp.bfloat16).astype(jnp.float32)


def _rel_err(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _small_cfg(**kw):
  base = dict(
      block_t_fwd=64,
      block_v_fwd=128,
      block_t_dx=64,
      block_v_dx=256,
      block_t_dw=32,
      block_v_dw=128,
      block_t_bwd=64,
      block_v_bwd=64,
      interpret=True,
  )
  base.update(kw)
  return fused_ce.FusedCEConfig(**base)


class FusedCETest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("aligned", 128, 128, 512, 0.0, True),
      ("ragged_t_v", 200, 128, 1000, 0.0, True),
      ("ragged_zloss", 136, 256, 777, 1e-4, True),
      ("ragged_zloss_dw_tn", 136, 256, 777, 1e-4, False),
      ("ragged_zloss_bwd_fused", 136, 256, 777, 1e-4, "fused"),
      ("aligned_bwd_fused", 128, 128, 512, 0.0, "fused"),
  )
  def test_matches_reference(self, t, d, v, z_loss, dw_mode):
    x, w, targets, weights = _inputs(t, d, v)
    if dw_mode == "fused":
      cfg = _small_cfg(bwd_fused=True, block_t_bwd=32, block_v_bwd=128)
    else:
      cfg = _small_cfg(bwd_fused=False, dw_transposed=dw_mode)

    def fused_obj(x, w):
      loss, z = fused_ce.fused_linear_cross_entropy(x, w, targets, weights, z_loss=z_loss, cfg=cfg)
      return jnp.sum(loss), (loss, z)

    def ref_obj(x, w):
      # f32 golden reference on the bf16-rounded operands the kernel consumes.
      loss, z = fused_ce.reference_linear_cross_entropy(
          x.astype(jnp.float32), _round_bf16(w), targets, weights, z_loss=z_loss, compute_dtype=jnp.float32
      )
      return jnp.sum(loss), (loss, z)

    (f_sum, (f_loss, f_z)), (f_dx, f_dw) = jax.value_and_grad(fused_obj, argnums=(0, 1), has_aux=True)(x, w)
    (r_sum, (r_loss, r_z)), (r_dx, r_dw) = jax.value_and_grad(ref_obj, argnums=(0, 1), has_aux=True)(x, w)

    self.assertEqual(f_dx.dtype, x.dtype)
    self.assertEqual(f_dw.dtype, jnp.float32)
    self.assertEqual(f_dw.shape, w.shape)
    np.testing.assert_allclose(f_loss, r_loss, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(f_z, r_z, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_sum, r_sum, rtol=1e-5)
    # Masked tokens contribute neither loss nor gradient.
    self.assertTrue(np.all(np.asarray(f_loss)[np.asarray(weights) == 0] == 0))
    self.assertTrue(np.all(np.asarray(f_dx, np.float32)[np.asarray(weights) == 0] == 0))
    self.assertLess(_rel_err(f_dx, r_dx), 1e-2)
    self.assertLess(_rel_err(f_dw, r_dw), 1e-2)
    np.testing.assert_allclose(np.asarray(f_dw), np.asarray(r_dw), rtol=5e-2, atol=1e-2)

  def test_matches_maxtext_cross_entropy_semantics(self):
    """Compares against max_utils.cross_entropy_with_logits on f32 logits (incl. z-loss grad)."""
    t, d, v, z_loss = 96, 128, 640, 3e-4
    x, w, targets, weights = _inputs(t, d, v, seed=3)
    cfg = _small_cfg()

    def fused_obj(x, w):
      loss, z = fused_ce.fused_linear_cross_entropy(x, w, targets, weights, z_loss=z_loss, cfg=cfg)
      return jnp.sum(loss), jnp.sum(z)

    def maxtext_obj(x, w):
      logits = jnp.dot(x, w.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
      xent, z = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(targets, v), z_loss=z_loss)
      return jnp.sum(xent * weights), jnp.sum(z * weights)

    (f_sum, f_z), (f_dx, f_dw) = jax.value_and_grad(fused_obj, argnums=(0, 1), has_aux=True)(x, w)
    (m_sum, m_z), (m_dx, m_dw) = jax.value_and_grad(maxtext_obj, argnums=(0, 1), has_aux=True)(x, w)
    np.testing.assert_allclose(f_sum, m_sum, rtol=1e-5)
    np.testing.assert_allclose(f_z, m_z, rtol=1e-5)
    # maxtext's path rounds dlogits / dw to bf16, so compare in relative norm.
    self.assertLess(_rel_err(f_dx, m_dx), 1e-2)
    self.assertLess(_rel_err(f_dw, m_dw), 1e-2)

  def test_lse_cotangent_and_batch_dims(self):
    """Arbitrary (lse, target_logit) cotangents and [B, S] leading dims."""
    b, s, d, v = 2, 48, 128, 384
    x, w, targets, _ = _inputs(b * s, d, v, seed=7)
    x = x.reshape(b, s, d)
    targets = targets.reshape(b, s)
    ka, kb = jax.random.split(jax.random.PRNGKey(11))
    ca = jax.random.normal(ka, (b, s))
    cb = jax.random.normal(kb, (b, s))
    cfg = _small_cfg()

    def fused_obj(x, w):
      lse, tl = fused_ce.linear_lse_and_target_logit(x, w, targets, cfg)
      return jnp.sum(ca * lse + cb * tl)

    def ref_obj(x, w):
      logits = jnp.einsum("bsd,dv->bsv", x.astype(jnp.float32), _round_bf16(w))
      lse = jax.scipy.special.logsumexp(logits, axis=-1)
      tl = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
      return jnp.sum(ca * lse + cb * tl)

    f_val, (f_dx, f_dw) = jax.value_and_grad(fused_obj, argnums=(0, 1))(x, w)
    r_val, (r_dx, r_dw) = jax.value_and_grad(ref_obj, argnums=(0, 1))(x, w)
    np.testing.assert_allclose(f_val, r_val, rtol=1e-4, atol=1e-3)
    self.assertLess(_rel_err(f_dx, r_dx), 1e-2)
    self.assertLess(_rel_err(f_dw, r_dw), 1e-2)

  @parameterized.named_parameters(
      ("fsdp_data_tokens_fsdp_head", ("data", "fsdp"), ("fsdp", None)),
      ("fsdp_tokens_head_on_both", "fsdp", ("fsdp", "data")),
  )
  def test_sharded_matches_unsharded(self, batch_axes, w_axes):
    """shard_map wrapper: tokens split over mesh axes, head all-gathered inside."""
    if len(jax.devices()) < 8:
      self.skipTest("needs 8 devices")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:8]).reshape(2, 4), ("data", "fsdp"))
    b, s, d, v = 8, 32, 128, 512
    x, w, targets, weights = _inputs(b * s, d, v, seed=5)
    x, targets, weights = x.reshape(b, s, d), targets.reshape(b, s), weights.reshape(b, s)
    cfg = _small_cfg()
    token_spec = jax.sharding.PartitionSpec(batch_axes, None)
    w_spec = jax.sharding.PartitionSpec(*w_axes)

    def sharded_obj(x, w):
      loss, z = fused_ce.sharded_fused_linear_cross_entropy(
          x, w, targets, weights, mesh=mesh, token_spec=token_spec, w_spec=w_spec, z_loss=1e-4, cfg=cfg
      )
      return jnp.sum(loss), jnp.sum(z)

    def local_obj(x, w):
      loss, z = fused_ce.fused_linear_cross_entropy(x, w, targets, weights, z_loss=1e-4, cfg=cfg)
      return jnp.sum(loss), jnp.sum(z)

    xs = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(batch_axes, None, None)))
    ws = jax.device_put(w, jax.sharding.NamedSharding(mesh, w_spec))
    (s_sum, s_z), (s_dx, s_dw) = jax.jit(jax.value_and_grad(sharded_obj, argnums=(0, 1), has_aux=True))(xs, ws)
    (l_sum, l_z), (l_dx, l_dw) = jax.value_and_grad(local_obj, argnums=(0, 1), has_aux=True)(x, w)
    np.testing.assert_allclose(s_sum, l_sum, rtol=1e-5)
    np.testing.assert_allclose(s_z, l_z, rtol=1e-5)
    self.assertLess(_rel_err(s_dx, l_dx), 1e-3)
    self.assertLess(_rel_err(s_dw, l_dw), 1e-5)


if __name__ == "__main__":
  absltest.main()
