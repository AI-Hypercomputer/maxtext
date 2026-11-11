# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-device GLM-5.3-Flash tests that stay in cpu-unit.

The rest of these tests' modules live in tests/integration/m3/. These two need 4 JAX devices, and only the
cpu-unit CI flavor sets --xla_force_host_platform_device_count=4, so they stay here. The config helpers below
mirror the ones in tests/integration/m3/glm5_3_flash_ckpt_test.py and glm5_3_flash_kda_test.py.
"""

from __future__ import annotations

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.glm5_3_flash_config import Glm5Sharding
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5KdaAttention
import numpy as np
import pytest

try:
  from maxtext.m3.models.glm5_3_flash import glm5_ckpt
  import torch

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

try:
  jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError as e:
  if "before backends are initialized" not in str(e):
    raise


def _make_cfg(
    num_hidden_layers: int = 7, scan_layers: bool = True, dtype: str = "float32", weight_dtype: str = "float32"
):
  """Constructs the mini Glm5Config used by glm5_3_flash_ckpt_test.py."""
  kws = (
      "hidden_size=64 intermediate_size=128 moe_intermediate_size=32"
      " n_routed_experts=8 num_experts_per_tok=2 q_lora_rank=32 kv_lora_rank=16"
      " num_attention_heads=4 num_key_value_heads=4 qk_nope_head_dim=16"
      " v_head_dim=16 index_n_heads=8 index_head_dim=16 index_topk=8"
      " linear_num_key_heads=4 linear_num_value_heads=4"
      " linear_key_head_dim=16 linear_value_head_dim=16"
  )
  args = {k: int(v) for k, v in (s.split("=") for s in kws.split())}
  args.update(
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      kda_implementation="xla",
      vocab_size=256,
      dtype=dtype,
      weight_dtype=weight_dtype,
  )
  return glm5_config.Glm5Config.mini_config(**args)


class Glm5CkptMultiDeviceTest(parameterized.TestCase):
  """Orbax restore with 4-device sharding (from glm5_3_flash_ckpt_test.py)."""

  def setUp(self):
    super().setUp()
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def test_restore_orbax_checkpoint_sharded(self):
    """Verifies restoring an Orbax checkpoint with different sharding branch configurations."""
    if len(jax.devices()) < 4 and "TEST_SRCDIR" not in os.environ:
      self.skipTest("Requires at least 4 JAX devices.")
    c = _make_cfg(num_hidden_layers=7, scan_layers=True)
    self.assertGreaterEqual(len(jax.devices()), 4)
    devices = np.array(jax.devices()[:4]).reshape((1, 4, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))

    with tempfile.TemporaryDirectory() as tmp_dir:
      obx_dir = os.path.join(tmp_dir, "obx")
      with jax.sharding.set_mesh(mesh):
        src_m = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(0))
      glm5_ckpt.save_orbax_checkpoint(src_m, obx_dir)

      # Branch 2: model.sharding is not None, mesh=None
      with jax.sharding.set_mesh(mesh):
        dst_m = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(1))
        glm5_ckpt.restore_orbax_checkpoint(dst_m, obx_dir, mesh=None)

      pure_src = nnx.to_pure_dict(nnx.state(src_m))
      pure_dst = nnx.to_pure_dict(nnx.state(dst_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_dst)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(dl.sharding.mesh, mesh)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      for var in jax.tree.leaves(nnx.state(dst_m), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        if (
            isinstance(var, nnx.Variable)
            and var.has_metadata("out_sharding")
            and var.get_metadata("out_sharding") is not None
        ):
          out_sh = var.get_metadata("out_sharding")
          spec = out_sh.spec if isinstance(out_sh, jax.sharding.NamedSharding) else out_sh
          self.assertEqual(
              var.value.sharding,
              jax.sharding.NamedSharding(mesh, spec),
          )

      self.assertEqual(
          dst_m.super_block.slot_0.attn_hc.fn.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "fsdp", None)),
      )
      exp_var = dst_m.super_block.slot_0.mlp.experts.gate_up_proj
      out_sh = exp_var.get_metadata("out_sharding")
      spec = out_sh.spec if isinstance(out_sh, jax.sharding.NamedSharding) else out_sh
      self.assertEqual(
          exp_var.value.sharding,
          jax.sharding.NamedSharding(mesh, spec),
      )
      self.assertEqual(
          exp_var.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      self.assertEqual(
          dst_m.prefix_layers[0].attn_hc.scale.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
      )
      self.assertEqual(
          dst_m.super_block.slot_0.attn_hc.scale.value.sharding,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None)),
      )

      # Branch with mesh != None raises ValueError
      with self.assertRaisesRegex(
          ValueError,
          "Passing mesh to restore_orbax_checkpoint is not supported",
      ):
        glm5_ckpt.restore_orbax_checkpoint(dst_m, obx_dir, mesh=mesh)

      # Restore into unplaced model (place_on_devices=False, mesh=mesh)
      with jax.sharding.set_mesh(mesh):
        dst_unplaced = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=False, rngs=nnx.Rngs(2))
      glm5_ckpt.restore_orbax_checkpoint(dst_unplaced, obx_dir, mesh=None)
      self.assertEqual(
          dst_unplaced.super_block.slot_0.mlp.experts.gate_up_proj.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      pure_unplaced = nnx.to_pure_dict(nnx.state(dst_unplaced))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_unplaced)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      # Restore into abstract model (nnx.eval_shape)
      with jax.sharding.set_mesh(mesh):
        abs_m = nnx.eval_shape(
            lambda: modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), mesh=mesh, place_on_devices=False)
        )
      glm5_ckpt.restore_orbax_checkpoint(abs_m, obx_dir, mesh=None)
      self.assertEqual(
          abs_m.super_block.slot_0.mlp.experts.gate_up_proj.value.sharding,
          jax.sharding.NamedSharding(
              mesh,
              jax.sharding.PartitionSpec(None, "expert", "fsdp", None, None),
          ),
      )
      pure_abs = nnx.to_pure_dict(nnx.state(abs_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_abs)):
        self.assertIsInstance(dl.sharding, jax.sharding.NamedSharding)
        self.assertEqual(len(dl.sharding.device_set), 4)
        np.testing.assert_array_equal(sl, dl)

      # Unsharded restore: model.sharding is None, mesh=None (CPU host sharding)
      host_m = modeling_glm5_3_flash.Glm5Model(c, place_on_devices=False, rngs=nnx.Rngs(3))
      with self.assertRaisesRegex(
          ValueError,
          "Passing mesh to restore_orbax_checkpoint is not supported",
      ):
        glm5_ckpt.restore_orbax_checkpoint(host_m, obx_dir, mesh=mesh)
      glm5_ckpt.restore_orbax_checkpoint(host_m, obx_dir, mesh=None)
      pure_host = nnx.to_pure_dict(nnx.state(host_m))
      for sl, dl in zip(jax.tree.leaves(pure_src), jax.tree.leaves(pure_host)):
        self.assertIsInstance(dl.sharding, jax.sharding.SingleDeviceSharding)
        self.assertEqual(dl.sharding.device_set, {jax.local_devices(backend="cpu")[0]})
        np.testing.assert_array_equal(sl, dl)

      # Error guard: model.sharding is set, but model.mesh is None
      bad_m = modeling_glm5_3_flash.Glm5Model(c, place_on_devices=False, rngs=nnx.Rngs(4))
      bad_m.sharding = jax.sharding.PartitionSpec()
      bad_m.mesh = None
      with self.assertRaisesRegex(ValueError, "model.sharding is set, but model.mesh is None or empty"):
        glm5_ckpt.restore_orbax_checkpoint(bad_m, obx_dir, mesh=None)

      # Error guard: mesh=None with model.sharding=None when model leaves already carry NamedSharding
      with jax.sharding.set_mesh(mesh):
        unsharded_with_shards = modeling_glm5_3_flash.Glm5Model(c, mesh=mesh, place_on_devices=True, rngs=nnx.Rngs(5))
      unsharded_with_shards.sharding = None
      with self.assertRaisesRegex(
          ValueError,
          "cannot be used when model leaves already carry NamedSharding",
      ):
        glm5_ckpt.restore_orbax_checkpoint(unsharded_with_shards, obx_dir, mesh=None)


class Glm5KdaMultiDeviceTest(parameterized.TestCase):
  """Sharded vs unsharded Glm5KdaAttention parity on 4 devices (from glm5_3_flash_kda_test.py)."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _mini_cfg(self, kda_implementation="xla", dtype="float32", weight_dtype=None, **kwargs) -> Glm5Config:
    w_dtype = weight_dtype if weight_dtype is not None else dtype
    return Glm5Config.mini_config(
        hidden_size=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        kda_implementation=kda_implementation,
        dtype=dtype,
        weight_dtype=w_dtype,
        **kwargs,
    )

  def test_kda_multi_device_sharding_parity(self):
    """Verifies numerical parity between sharded and unsharded Glm5KdaAttention forward and backward."""
    if len(jax.devices()) < 4:
      if "TEST_SRCDIR" in os.environ:
        self.fail("Expected >= 4 JAX devices under Blaze XLA_FLAGS, got" f" {len(jax.devices())}")
      self.skipTest(f"Requires >= 4 JAX devices, got {len(jax.devices())}")
    mesh = m3_sharding.create_mesh(
        ici={"data": 2, "fsdp": 1, "tensor": 2, "expert": 1},
        devices=jax.devices()[:4],
    )
    sharding = Glm5Sharding.from_mesh(mesh)
    cfg = self._mini_cfg()

    layer_ref = Glm5KdaAttention(cfg, rngs=nnx.Rngs(0))
    rng = np.random.RandomState(42)
    for cv in (layer_ref.q_conv1d, layer_ref.k_conv1d, layer_ref.v_conv1d):
      cv.weight[...] = jnp.array(rng.randn(*cv.weight[...].shape).astype(np.float32))
    layer_ref.A_log[...] = jnp.array(rng.randn(*layer_ref.A_log[...].shape).astype(np.float32))
    layer_ref.dt_bias[...] = jnp.array(rng.randn(*layer_ref.dt_bias[...].shape).astype(np.float32))

    with jax.sharding.set_mesh(mesh):
      layer_sharded = Glm5KdaAttention(cfg, sharding=sharding, rngs=nnx.Rngs(0))
      ref_dict = dict(nnx.state(layer_ref, nnx.Param).flat_state())
      sharded_state = nnx.state(layer_sharded, nnx.Param)
      sharded_dict = dict(sharded_state.flat_state())
      for path, ref_var in ref_dict.items():
        target_var = sharded_dict[path]
        target_shd = target_var.sharding if target_var.sharding is not None else jax.sharding.NamedSharding(mesh, P())
        target_var[...] = jax.device_put(ref_var[...], target_shd)
      nnx.update(layer_sharded, sharded_state)

    x_np = rng.randn(4, 8, 64).astype(np.float32)
    cot_np = rng.randn(4, 8, 64).astype(np.float32)
    x = jnp.array(x_np)
    cot = jnp.array(cot_np)

    def loss(m, inp, c):
      return jnp.sum(m(inp) * c)

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    _, (g_ref, gx_ref) = vg(layer_ref, x, cot)
    out_ref = layer_ref(x)

    x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, P(("data", "fsdp"), None, None)))
    cot_sharded = jax.device_put(cot, jax.sharding.NamedSharding(mesh, P(("data", "fsdp"), None, None)))

    with jax.sharding.set_mesh(mesh):
      vg_sharded = jax.jit(vg)
      _, (g_sharded, gx_sharded) = vg_sharded(layer_sharded, x_sharded, cot_sharded)
      out_sharded = jax.jit(lambda m, inp: m(inp))(layer_sharded, x_sharded)

    def _assert_frob_rel(actual, ref, name, tol=1e-5):
      a_np = np.array(actual, dtype=np.float64)
      r_np = np.array(ref, dtype=np.float64)
      self.assertTrue(np.all(np.isfinite(a_np)), f"Non-finite actual: {name}")
      self.assertTrue(np.all(np.isfinite(r_np)), f"Non-finite ref: {name}")
      denom = max(float(np.linalg.norm(r_np)), 1e-12)
      frob_rel = float(np.linalg.norm(a_np - r_np)) / denom
      self.assertLessEqual(frob_rel, tol, f"{name} frob_rel={frob_rel} > {tol}")

    _assert_frob_rel(out_sharded, out_ref, "output")
    _assert_frob_rel(gx_sharded, gx_ref, "gx")
    g_ref_flat = dict(nnx.state(g_ref, nnx.Param).flat_state())
    g_sharded_flat = dict(nnx.state(g_sharded, nnx.Param).flat_state())
    self.assertLen(g_ref_flat, 15)
    self.assertEqual(set(g_ref_flat.keys()), set(g_sharded_flat.keys()))
    for path, ref_var in g_ref_flat.items():
      _assert_frob_rel(g_sharded_flat[path][...], ref_var[...], ".".join(map(str, path)))

    # Positive check: verify gradients are non-trivial
    self.assertGreater(float(np.linalg.norm(np.array(g_ref.A_log[...]))), 1e-4)
    self.assertGreater(float(np.linalg.norm(np.array(g_ref.dt_bias[...]))), 1e-4)
    self.assertGreater(float(np.linalg.norm(np.array(gx_ref))), 1e-4)


if __name__ == "__main__":
  absltest.main()
