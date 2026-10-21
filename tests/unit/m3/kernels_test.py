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

"""Tests for `maxtext.m3.core.kernels`.

CPU: the legacy-config translation, the specs a kernel object carries, tile
selection, and the plain-XLA backends. TPU: every fused backend against the
reference arithmetic on an 8-chip mesh that shards both the batch (fsdp=4)
and the heads / expert MLP dimension (tensor=2), standalone and through a
model, including under the scan and remat transforms.
"""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core import attention, kernels, moe
from maxtext.m3.core.sharding import DefaultSharding, TensorType, create_mesh
from maxtext.m3.infra import remat, scan
from tests.utils import m3_test_utils as u

_SA_FIELDS = {
    "sa_block_q": 512,
    "sa_block_kv": 512,
    "sa_block_kv_compute": 512,
    "sa_block_q_dkv": 512,
    "sa_block_kv_dkv": 512,
    "sa_block_kv_dkv_compute": 512,
    "sa_block_q_dq": 512,
    "sa_block_kv_dq": 512,
    "sa_use_fused_bwd_kernel": False,
}


def _config(**overrides):
  fields = {"attention": "flash", "attention_type": "global", "use_tokamax_splash": False, **_SA_FIELDS}
  return SimpleNamespace(**{**fields, **overrides})


class AttentionKernelConfigTest(parameterized.TestCase):

  def test_dot_product_is_the_reference(self):
    self.assertIs(
        kernels.attention_from_config(_config(attention="dot_product"), DefaultSharding()), kernels.XLA_ATTENTION
    )

  def test_flash_resolves_mesh_axes_from_the_sharding(self):
    kernel = kernels.attention_from_config(_config(sa_block_q=256), DefaultSharding())
    self.assertEqual(kernel.backend, "splash")
    # DefaultSharding: batch over data x fsdp x expert, heads over tensor, length and head_dim replicated.
    self.assertEqual(kernel.batch, ("data", "fsdp", "expert"))
    self.assertEqual(kernel.heads, "tensor")
    self.assertEqual(kernel.kv_heads, "tensor")
    self.assertIsNone(kernel.length)
    self.assertEqual(kernel.blocks.block_q, 256)
    self.assertEqual(kernel.blocks.block_kv, 512)

  def test_tokamax_flag_selects_tokamax(self):
    self.assertEqual(
        kernels.attention_from_config(_config(use_tokamax_splash=True), DefaultSharding()).backend, "tokamax"
    )

  def test_autoselected_is_the_reference_off_tpu(self):
    """This test is CPU-only (unmarked), so `autoselected` must come out as the reference here."""
    self.assertEqual(kernels.attention_from_config(_config(attention="autoselected"), DefaultSharding()).backend, "xla")

  @parameterized.named_parameters(
      ("gpu_kernel", {"attention": "cudnn_flash_te"}, "m3 supports attention in"),
      ("sliding_window", {"attention_type": "local_sliding"}, "full causal attention"),
  )
  def test_unsupported_choices_raise(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      kernels.attention_from_config(_config(**overrides), DefaultSharding())

  def test_kernel_object_is_static_module_state(self):
    """Hashable and equal by value, which is what lets Flax treat it as graph metadata."""
    a = kernels.attention_from_config(_config(), DefaultSharding())
    b = kernels.attention_from_config(_config(), DefaultSharding())
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))
    self.assertNotEqual(a, kernels.attention_from_config(_config(sa_block_q=128), DefaultSharding()))

  def test_blocks_clamp_to_the_sequence(self):
    blocks = kernels.SplashBlocks().resolve(q_length=128, kv_length=128)
    self.assertEqual((blocks.block_q, blocks.block_kv, blocks.block_q_dq), (128, 128, 128))
    fused = kernels.SplashBlocks(use_fused_bwd_kernel=True).resolve(q_length=2048, kv_length=2048)
    self.assertEqual((fused.block_q, fused.block_q_dq, fused.block_kv_dq), (512, None, None))

  def test_fused_attention_refuses_the_xla_kernel(self):
    x = jnp.zeros((1, 4, 1, 8))
    with self.assertRaisesRegex(ValueError, "No fused attention backend"):
      kernels.fused_attention(x, x, x, segment_ids=None, kernel=kernels.XLA_ATTENTION)


class GroupedMatmulTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    keys = jax.random.split(jax.random.key(0), 2)
    self.lhs = jax.random.normal(keys[0], (256, 128), jnp.bfloat16)
    self.rhs = jax.random.normal(keys[1], (4, 128, 128), jnp.bfloat16)
    self.group_sizes = jnp.array([100, 0, 28, 128], jnp.int32)

  def test_ragged_dot_backend_is_jax_lax(self):
    out = kernels.grouped_matmul(self.lhs, self.rhs, self.group_sizes, backend="ragged_dot")
    expected = jax.lax.ragged_dot(self.lhs, self.rhs, self.group_sizes, preferred_element_type=jnp.float32)
    self.assertEqual(out.dtype, jnp.float32)
    np.testing.assert_array_equal(out, expected)

  def test_unknown_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown grouped matmul backend"):
      kernels.grouped_matmul(self.lhs, self.rhs, self.group_sizes, backend="cublas")

  def test_megablox_tiles_divide_the_problem(self):
    self.assertEqual(kernels._megablox_tiling(16384, 2048, 768), (512, 1024, 256))  # pylint: disable=protected-access
    self.assertEqual(kernels._megablox_tiling(24, 128, 128), (8, 128, 128))  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      kernels._megablox_tiling(512, 64, 128)  # pylint: disable=protected-access

  def test_megablox_runs_in_the_interpreter_off_tpu(self):
    """The Pallas path is exercised on CPU too, through `shard_map` on a one-device mesh."""
    mesh = create_mesh(devices=jax.devices()[:1])
    executor = moe.Executor("megablox", batch=("data", "fsdp", "expert"), experts="expert", mlp="tensor")
    keys = jax.random.split(jax.random.key(1), 5)
    num_experts, embed, mlp = 4, 128, 128
    x = jax.random.normal(keys[0], (2, 8, embed), jnp.float32)
    weights, indices = jax.lax.top_k(jax.nn.softmax(jax.random.normal(keys[1], (2, 8, num_experts))), 2)
    projections = (
        jax.random.normal(keys[2], (num_experts, embed, mlp)) * 0.05,
        jax.random.normal(keys[3], (num_experts, embed, mlp)) * 0.05,
        jax.random.normal(keys[4], (num_experts, mlp, embed)) * 0.05,
    )
    with jax.set_mesh(mesh):
      out = jax.jit(lambda *a: moe.routed_experts(*a, executor=executor))(x, weights, indices, *projections)
      expected = moe.routed_experts(x, weights, indices, *projections, executor=moe.DENSE)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# TPU
# --------------------------------------------------------------------------- #


def _attention_inputs(key, *, batch=8, length=512, heads=8, kv_heads=4, head_dim=128, value_dim=128, segments=True):
  """Scaled bf16 inputs, two packed segments per row when `segments`."""
  keys = jax.random.split(key, 3)
  q = jax.random.normal(keys[0], (batch, length, heads, head_dim), jnp.bfloat16) * head_dim**-0.5
  k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), jnp.bfloat16)
  v = jax.random.normal(keys[2], (batch, length, kv_heads, value_dim), jnp.bfloat16)
  if segments:
    segment_ids = jnp.broadcast_to((jnp.arange(length) >= length // 2).astype(jnp.int32), (batch, length))
    positions = jnp.broadcast_to(jnp.arange(length) % (length // 2), (batch, length))
  else:
    segment_ids = None
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
  return q, k, v, positions, segment_ids


def _max_abs(a, b):
  return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


@pytest.mark.tpu_only
class FusedAttentionTest(parameterized.TestCase):
  """Fused backends against the float32 reference, on a mesh sharding batch and heads."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(ici={"fsdp": 4, "tensor": 2})
    self.enter_context(jax.set_mesh(self.mesh))
    self.sharding = DefaultSharding()

  def _kernel(self, backend):
    return kernels.attention_from_config(_config(use_tokamax_splash=backend == "tokamax"), self.sharding)

  def _place(self, q, k, v, positions, segment_ids):
    """Shards the inputs the way a model's activation constraints would."""

    def act(name, axes):
      return NamedSharding(self.mesh, self.sharding(name, axes, TensorType.ACTIVATION))

    q = jax.device_put(q, act("query", ("batch", "length", "heads", "head_dim")))
    k = jax.device_put(k, act("key", ("batch", "length", "kv_heads", "head_dim")))
    v = jax.device_put(v, act("value", ("batch", "length", "kv_heads", "head_dim")))
    positions = jax.device_put(positions, act("positions", ("batch", "length")))
    if segment_ids is not None:
      segment_ids = jax.device_put(segment_ids, act("segment_ids", ("batch", "length")))
    return q, k, v, positions, segment_ids

  @parameterized.product(backend=("splash", "tokamax"), segments=(True, False), value_dim=(128, 64))
  def test_forward_matches_reference(self, backend, segments, value_dim):
    """Fused output is as close to the float32 reference as the bf16 reference itself is."""
    q, k, v, positions, segment_ids = self._place(
        *_attention_inputs(jax.random.key(0), segments=segments, value_dim=value_dim)
    )

    def run(kernel, dtype):
      def f(q, k, v, positions, segment_ids):
        return attention.dot_product_attention(q, k, v, positions=positions, segment_ids=segment_ids, kernel=kernel)

      return jax.jit(f)(q.astype(dtype), k.astype(dtype), v.astype(dtype), positions, segment_ids)

    exact = run(kernels.XLA_ATTENTION, jnp.float32)
    bf16_reference = run(kernels.XLA_ATTENTION, jnp.bfloat16)
    fused = run(self._kernel(backend), jnp.bfloat16)
    self.assertEqual(fused.shape, (8, 512, 8, value_dim))
    self.assertEqual(fused.dtype, jnp.bfloat16)
    self.assertLessEqual(_max_abs(fused, exact), 2 * _max_abs(bf16_reference, exact) + 1e-3)

  @parameterized.named_parameters(("splash", "splash"), ("tokamax", "tokamax"))
  def test_gradients_match_reference(self, backend):
    q, k, v, positions, segment_ids = self._place(*_attention_inputs(jax.random.key(1)))

    def loss(kernel, dtype):
      def f(q, k, v):
        out = attention.dot_product_attention(q, k, v, positions=positions, segment_ids=segment_ids, kernel=kernel)
        return jnp.sum(out.astype(jnp.float32) ** 2)

      return jax.jit(jax.grad(f, argnums=(0, 1, 2)))(q.astype(dtype), k.astype(dtype), v.astype(dtype))

    exact = loss(kernels.XLA_ATTENTION, jnp.float32)
    bf16_reference = loss(kernels.XLA_ATTENTION, jnp.bfloat16)
    fused = loss(self._kernel(backend), jnp.bfloat16)
    for name, g_fused, g_ref, g_exact in zip("qkv", fused, bf16_reference, exact):
      self.assertLessEqual(_max_abs(g_fused, g_exact), 2 * _max_abs(g_ref, g_exact) + 1e-2, name)

  def test_segments_are_respected(self):
    """A fused kernel with segment ids equals the reference run segment by segment."""
    q, k, v, positions, segment_ids = self._place(*_attention_inputs(jax.random.key(2), segments=True))
    splash = self._kernel("splash")

    @jax.jit
    def run(q, k, v, positions, segment_ids):
      fused = attention.dot_product_attention(q, k, v, positions=positions, segment_ids=segment_ids, kernel=splash)
      unsegmented = attention.dot_product_attention(q, k, v, positions=positions, segment_ids=None, kernel=splash)
      reference = attention.dot_product_attention(q, k, v, positions=positions, segment_ids=segment_ids)
      return fused, unsegmented, reference

    fused, unsegmented, reference = run(q, k, v, positions, segment_ids)
    half = q.shape[1] // 2
    # Without segment ids the second half would see the first; with them it must not.
    self.assertGreater(_max_abs(fused[:, half:], unsegmented[:, half:]), 1e-2)
    self.assertLess(_max_abs(fused, reference), 5e-2)


@pytest.mark.tpu_only
class PallasExecutorTest(parameterized.TestCase):
  """`megablox` and `tokamax` executors against `ragged_dot`, with FSDP-sharded tokens and a tensor-sharded MLP."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(ici={"fsdp": 4, "tensor": 2})
    self.enter_context(jax.set_mesh(self.mesh))
    s = DefaultSharding()
    keys = jax.random.split(jax.random.key(0), 5)
    self.num_experts, top_k, embed, mlp = 8, 2, 256, 256
    tokens = NamedSharding(self.mesh, s("x", ("batch", "length", "embed"), TensorType.ACTIVATION))
    self.x = jax.device_put(jax.random.normal(keys[0], (8, 128, embed), jnp.bfloat16), tokens)
    weights, indices = jax.lax.top_k(jax.nn.softmax(jax.random.normal(keys[1], (8, 128, self.num_experts))), top_k)
    self.weights = jax.device_put(weights.astype(jnp.bfloat16), tokens)
    self.indices = jax.device_put(indices, tokens)
    wi = NamedSharding(self.mesh, s("wi_0", ("experts", "expert_embed", "expert_mlp")))
    wo = NamedSharding(self.mesh, s("wo", ("experts", "expert_mlp", "expert_embed")))
    self.projections = (
        jax.device_put(jax.random.normal(keys[2], (self.num_experts, embed, mlp), jnp.bfloat16) * 0.05, wi),
        jax.device_put(jax.random.normal(keys[3], (self.num_experts, embed, mlp), jnp.bfloat16) * 0.05, wi),
        jax.device_put(jax.random.normal(keys[4], (self.num_experts, mlp, embed), jnp.bfloat16) * 0.05, wo),
    )
    self.config = SimpleNamespace(sparse_matmul=True, megablox=True, use_tokamax_gmm=False, capacity_factor=-1.0)

  def _run(self, executor, x=None, projections=None):
    x = self.x if x is None else x
    projections = self.projections if projections is None else projections

    def f(x, weights, indices, projections):
      return moe.routed_experts(x, weights, indices, *projections, executor=executor)

    return jax.jit(f)(x, self.weights, self.indices, projections)

  @parameterized.named_parameters(("megablox", False), ("tokamax", True))
  def test_matches_ragged_dot(self, use_tokamax_gmm):
    executor = moe.executor_from_config(
        SimpleNamespace(**{**vars(self.config), "use_tokamax_gmm": use_tokamax_gmm}), DefaultSharding()
    )
    self.assertEqual(executor.kind, "tokamax" if use_tokamax_gmm else "megablox")
    out = self._run(executor)
    reference = self._run(moe.RAGGED_DOT)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(out.sharding.spec, self.x.sharding.spec)
    np.testing.assert_allclose(out.astype(jnp.float32), reference.astype(jnp.float32), rtol=5e-2, atol=5e-2)

  @parameterized.named_parameters(("megablox", False), ("tokamax", True))
  def test_gradients_match_ragged_dot(self, use_tokamax_gmm):
    executor = moe.executor_from_config(
        SimpleNamespace(**{**vars(self.config), "use_tokamax_gmm": use_tokamax_gmm}), DefaultSharding()
    )

    def grads(executor):
      def f(x, weights, indices, projections):
        out = moe.routed_experts(x, weights, indices, *projections, executor=executor)
        return jnp.sum(out.astype(jnp.float32) ** 2)

      return jax.jit(jax.grad(f, argnums=(0, 3)))(self.x, self.weights, self.indices, self.projections)

    g_out, g_ref = grads(executor), grads(moe.RAGGED_DOT)
    for a, b in zip(jax.tree.leaves(g_out), jax.tree.leaves(g_ref)):
      scale = float(jnp.max(jnp.abs(b.astype(jnp.float32)))) + 1e-6
      self.assertLess(_max_abs(a, b) / scale, 5e-2)

  def test_expert_parallelism_is_refused(self):
    executor = moe.Executor("megablox", batch=("data", "expert"), experts="fsdp", mlp="tensor")
    with self.assertRaisesRegex(ValueError, "expert parallelism"):
      self._run(executor)


@pytest.mark.tpu_only
class KernelsInModelsTest(absltest.TestCase):
  """Config-selected kernels through real models, on the 8-chip mesh, with the transforms applied."""

  # Kernel-friendly dimensions: lane-dense head and expert-MLP sizes, a sequence
  # longer than one splash block, and a batch that divides over fsdp=4.
  _OVERRIDES = {"head_dim": 128, "max_target_length": 256, "per_device_batch_size": 1}
  _MOE_OVERRIDES = {**_OVERRIDES, "base_emb_dim": 256, "base_moe_mlp_dim": 256}

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(ici={"fsdp": 4, "tensor": 2})
    self.enter_context(jax.set_mesh(self.mesh))

  def _batch(self, config):
    return u.random_batch(config, batch=config.global_batch_size_to_train_on, length=config.max_target_length)

  def _loss_and_grads(self, model, batch, config):
    return nnx.jit(lambda m, b: nnx.value_and_grad(lambda m: u.m3_loss(m, b, config))(m))(model, batch)

  def _assert_close(self, loss_a, grads_a, loss_b, grads_b, rtol):
    self.assertLess(abs(float(loss_a) - float(loss_b)) / abs(float(loss_b)), rtol)
    flat_a, flat_b = u.flat_params(grads_a), u.flat_params(grads_b)
    self.assertEqual(sorted(flat_a), sorted(flat_b))
    for path in flat_a:
      scale = float(jnp.max(jnp.abs(flat_b[path].astype(jnp.float32)))) + 1e-6
      self.assertLess(_max_abs(flat_a[path], flat_b[path]) / scale, 0.1, path)

  def test_flash_attention_matches_dot_product_through_qwen3(self):
    reference_config = u.tiny_config("qwen3-0.6b", attention="dot_product", **self._OVERRIDES)
    flash_config = u.tiny_config("qwen3-0.6b", attention="flash", **self._OVERRIDES)
    reference = m3_models.create_model(reference_config, self.mesh, rngs=nnx.Rngs(0))
    flash = m3_models.create_model(flash_config, self.mesh, rngs=nnx.Rngs(0))
    # Same seed, same parameters; only the kernel differs.
    u.assert_params_match(self, reference, flash)
    backends = {node.attention_kernel.backend for _, node in nnx.iter_graph(flash) if hasattr(node, "attention_kernel")}
    self.assertEqual(backends, {"splash"})

    batch = self._batch(flash_config)
    loss_ref, grads_ref = self._loss_and_grads(reference, batch, reference_config)
    loss_flash, grads_flash = self._loss_and_grads(flash, batch, flash_config)
    self._assert_close(loss_flash, grads_flash, loss_ref, grads_ref, rtol=2e-2)

  def test_flash_attention_survives_scan_and_remat(self):
    config = u.tiny_config("qwen3-0.6b", attention="flash", scan_layers=True, remat_policy="minimal", **self._OVERRIDES)
    plain = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    transformed = scan.maybe_apply_scan(
        remat.maybe_apply_remat(m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0)), config), config
    )
    batch = self._batch(config)
    loss_plain, _ = self._loss_and_grads(plain, batch, config)
    loss_scanned, grads_scanned = self._loss_and_grads(transformed, batch, config)
    # Scan stacks the per-layer parameters, so compare the loss and check gradient flow rather than paths.
    self.assertLess(abs(float(loss_plain) - float(loss_scanned)) / abs(float(loss_plain)), 1e-3)
    self.assertTrue(all(float(jnp.max(jnp.abs(g))) > 0.0 for g in u.flat_params(grads_scanned).values()))

  def test_megablox_matches_dense_through_qwen3_moe(self):
    dense_config = u.tiny_config("qwen3-30b-a3b", sparse_matmul=False, **self._MOE_OVERRIDES)
    megablox_config = u.tiny_config("qwen3-30b-a3b", sparse_matmul=True, megablox=True, **self._MOE_OVERRIDES)
    dense = m3_models.create_model(dense_config, self.mesh, rngs=nnx.Rngs(0))
    megablox = m3_models.create_model(megablox_config, self.mesh, rngs=nnx.Rngs(0))
    u.assert_params_match(self, dense, megablox)
    kinds = {node.executor.kind for _, node in nnx.iter_graph(megablox) if hasattr(node, "executor")}
    self.assertEqual(kinds, {"megablox"})

    batch = self._batch(megablox_config)
    loss_dense, grads_dense = self._loss_and_grads(dense, batch, dense_config)
    loss_megablox, grads_megablox = self._loss_and_grads(megablox, batch, megablox_config)
    self._assert_close(loss_megablox, grads_megablox, loss_dense, grads_dense, rtol=2e-2)


if __name__ == "__main__":
  absltest.main()
