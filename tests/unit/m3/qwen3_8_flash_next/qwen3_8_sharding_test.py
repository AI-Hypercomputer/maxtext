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

"""Sharding of Qwen3.8-Flash-Next: annotations, placement on a mesh, and unchanged numerics.

`sharding_test.py` covers the `DefaultSharding` rules in isolation. This file checks that the model uses them: every
parameter carries a resolved `out_sharding`, parameters land on the mesh as the rules say, the scanned cycle keeps the
placement, and a model built on a mesh computes what the same weights compute on one device. Multi-device cases need
four devices (CI runs the CPU suite with `--xla_force_host_platform_device_count=4`).
"""

import math
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.m3 import models as m3_models  # pylint: disable=no-name-in-module
from maxtext.m3.core.sharding import create_mesh, DefaultSharding, TensorType  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
import numpy as np

_DENSE_FSDP = ("fsdp", "fsdp_transpose", "expert")
_EXPERT_FSDP = ("fsdp", "fsdp_transpose")
_TP = ("tensor", "tensor_sequence")


def _mini_config(**overrides):
  """`qwen3.8-flash-next-mini`: one four-layer cycle (three GDN layers, one QSA layer), PLE, and an MTP block."""
  kwargs = {"scan_layers": False, "dtype": "float32", "weight_dtype": "float32", **overrides}
  return pyconfig.initialize(
      [None, f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml"],
      model_name="qwen3.8-flash-next-mini",
      override_model_config=True,
      packing=False,
      run_name="qwen3_8_sharding_test",
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
      **kwargs,
  )


def _params(model):
  """`{path: variable}` for every `nnx.Param`."""
  return {"/".join(map(str, path)): var for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _shard_shape(var):
  value = var[...]
  return value.sharding.shard_shape(value.shape)


def _shard_fraction(var) -> float:
  return math.prod(_shard_shape(var)) / var[...].size


class _DataOnlyBatch(DefaultSharding):
  """A strategy that puts the batch on `data` alone."""

  def map_axis(self, axis, tensor_name, tensor_type):
    if axis == "batch" and tensor_type == TensorType.ACTIVATION:
      return "data"
    return super().map_axis(axis, tensor_name, tensor_type)


class AnnotationTest(absltest.TestCase):
  """Construction on one device: every tensor resolves, and resolves to the axes the strategy names."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])

  def test_every_parameter_carries_a_resolved_sharding(self):
    for scan_layers in (False, True):
      model = m3_models.create_model(_mini_config(scan_layers=scan_layers), self.mesh, rngs=nnx.Rngs(0))
      params = _params(model)
      self.assertNotEmpty(params)
      for path, var in params.items():
        spec = var.get_metadata().get("out_sharding")
        self.assertIsInstance(spec, P, f"{path} has no resolved out_sharding: {var.get_metadata()}")
        self.assertLen(spec, var[...].ndim, path)
        self.assertEqual(var[...].sharding.spec, spec, f"{path} is not placed as its out_sharding says")
        # Placement ends with construction, so variables rebuilt from this metadata need no mesh.
        self.assertFalse(var.get_metadata().get("eager_sharding"), f"{path} still re-places on rebuild")

  def test_weights_name_the_strategy_axes(self):
    params = _params(m3_models.create_model(_mini_config(), self.mesh, rngs=nnx.Rngs(0)))
    gdn, qsa, ple = "decoder/layers_0", "decoder/layers_3/self_attn", "decoder/layers_1/ple"
    expected = {
        "decoder/token_embedder/embedding": P(_TP, _DENSE_FSDP),
        "lm_head/kernel": P(_DENSE_FSDP, _TP),
        # Hyper-connections: the concatenated stream dimension is the model dimension.
        f"{gdn}/attn_hyper_connection/input_mix_weight_down/kernel": P(_DENSE_FSDP, None),
        f"{gdn}/attn_hyper_connection/input_mix_weight_up/kernel": P(None, _DENSE_FSDP),
        f"{gdn}/attn_hyper_connection/block_inject_weight/kernel": P(_DENSE_FSDP, None),
        f"{gdn}/attn_hyper_connection/hc_norm/scale": P(None),
        # Gated DeltaNet: FSDP only, the fused projections have no head-major dimension for `tensor`.
        f"{gdn}/linear_attn/in_proj_qkv/kernel": P(_DENSE_FSDP, None),
        f"{gdn}/linear_attn/in_proj_z/kernel": P(_DENSE_FSDP, None),
        f"{gdn}/linear_attn/conv1d/kernel": P(None, None, _DENSE_FSDP),
        f"{gdn}/linear_attn/out_proj/kernel": P(None, _DENSE_FSDP),
        f"{gdn}/linear_attn/A_log": P(None),
        # Sparse attention: heads over `tensor`, the indexer over FSDP.
        f"{qsa}/q_proj/kernel": P(_DENSE_FSDP, _TP),
        f"{qsa}/k_proj/kernel": P(_DENSE_FSDP, _TP),
        f"{qsa}/o_proj/kernel": P(_TP, _DENSE_FSDP),
        f"{qsa}/indexer/index_qk_proj/kernel": P(_DENSE_FSDP, None),
        f"{qsa}/q_norm/scale": P(None),
        # MoE: routed experts over `expert`, 4D gate_up_proj and down_proj over `_TP`, shared expert like a dense MLP.
        f"{gdn}/mlp/experts/gate_up_proj": P("expert", _EXPERT_FSDP, None, _TP),
        f"{gdn}/mlp/experts/down_proj": P("expert", _TP, _EXPERT_FSDP),
        f"{gdn}/mlp/gate/linear/kernel": P(_DENSE_FSDP, None),
        f"{gdn}/mlp/shared_expert/gate_proj/kernel": P(_DENSE_FSDP, _TP),
        f"{gdn}/mlp/shared_expert/down_proj/kernel": P(_TP, _DENSE_FSDP),
        # PLE: the hashed n-gram table shards over its rows.
        f"{ple}/ple_embedding/ngram_embedding/embedding": P(_DENSE_FSDP, None),
        f"{ple}/conv1d/kernel": P(None, None, _DENSE_FSDP),
        f"{ple}/key_proj/kernel": P(_DENSE_FSDP, None),
        "mtp/fc_hidden/kernel": P(_DENSE_FSDP, None),
    }
    for path, spec in expected.items():
      self.assertEqual(params[path].get_metadata()["out_sharding"], spec, path)

  def test_scanned_cycle_leaves_the_cycle_axis_unsharded(self):
    config = _mini_config(scan_layers=True, base_num_decoder_layers=8)
    params = _params(m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0)))
    cycle = "decoder/cycles/gdn_0"
    kernel = params[f"{cycle}/linear_attn/in_proj_qkv/kernel"]
    self.assertEqual(kernel[...].shape[0], 2)
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(None, _DENSE_FSDP, None))
    self.assertEqual(
        params[f"{cycle}/mlp/experts/gate_up_proj"].get_metadata()["out_sharding"],
        P(None, "expert", _EXPERT_FSDP, None, _TP),
    )
    self.assertEqual(
        params[f"{cycle}/mlp/experts/down_proj"].get_metadata()["out_sharding"], P(None, "expert", _TP, _EXPERT_FSDP)
    )
    self.assertEqual(params[f"{cycle}/linear_attn/norm/scale"].get_metadata()["out_sharding"], P(None, None))
    # The PLE layer is hoisted out of the cycle and keeps its unstacked spec.
    self.assertEqual(
        params["decoder/ple/ple_embedding/ngram_embedding/embedding"].get_metadata()["out_sharding"], P(_DENSE_FSDP, None)
    )

  def test_construction_without_a_mesh_records_the_logical_spec(self):
    """`create_model(config, mesh=None)` (unit tests, single-device decode) keeps the metadata and applies nothing."""
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    params = _params(m3_models.create_model(_mini_config(), mesh=None, rngs=nnx.Rngs(0)))
    for path, var in params.items():
      self.assertLen(var.get_metadata()["out_sharding"], var[...].ndim, path)
    self.assertEqual(
        params["decoder/layers_0/linear_attn/in_proj_qkv/kernel"].get_metadata()["out_sharding"], P(_DENSE_FSDP, None)
    )


class PlacementTest(absltest.TestCase):
  """Construction on a multi-device mesh: where the bytes go, and that the numbers do not move."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest(f"needs 4 devices, found {jax.device_count()}")
    self.devices = jax.devices()[:4]

  def _model(self, mesh, **overrides):
    return m3_models.create_model(_mini_config(**overrides), mesh, rngs=nnx.Rngs(0))

  def test_fsdp_splits_every_weight_matrix_and_replicates_norms(self):
    params = _params(self._model(create_mesh(ici={"fsdp": -1}, devices=self.devices)))
    for path, var in params.items():
      if var[...].ndim == 1:
        self.assertEqual(_shard_fraction(var), 1.0, path)
      else:
        self.assertAlmostEqual(_shard_fraction(var), 1 / 4, msg=path)

  def test_tensor_parallelism_splits_heads_mlp_and_vocab(self):
    params = _params(self._model(create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)))
    self.assertEqual(_shard_shape(params["decoder/layers_3/self_attn/q_proj/kernel"]), (128, 256))
    self.assertEqual(_shard_shape(params["decoder/layers_3/self_attn/o_proj/kernel"]), (128, 128))
    self.assertEqual(_shard_shape(params["decoder/layers_0/mlp/shared_expert/gate_proj/kernel"]), (128, 32))
    self.assertEqual(_shard_shape(params["decoder/layers_0/mlp/experts/gate_up_proj"]), (8, 128, 2, 32))
    self.assertEqual(_shard_shape(params["decoder/layers_0/mlp/experts/down_proj"]), (8, 32, 128))
    self.assertEqual(_shard_shape(params["decoder/token_embedder/embedding"]), (512, 128))
    self.assertEqual(_shard_shape(params["lm_head/kernel"]), (128, 512))
    # Gated DeltaNet has no `tensor` dimension, so it stays FSDP-only.
    self.assertEqual(_shard_shape(params["decoder/layers_0/linear_attn/in_proj_qkv/kernel"]), (128, 512))

  def test_expert_axis_splits_the_routed_experts(self):
    params = _params(self._model(create_mesh(ici={"fsdp": -1, "expert": 2}, devices=self.devices)))
    self.assertEqual(_shard_shape(params["decoder/layers_0/mlp/experts/gate_up_proj"]), (4, 128, 2, 64))
    self.assertEqual(_shard_shape(params["decoder/layers_0/mlp/experts/down_proj"]), (4, 64, 128))
    # A dense weight spends the `expert` axis on its model dimension.
    self.assertEqual(_shard_shape(params["decoder/layers_0/linear_attn/in_proj_qkv/kernel"]), (64, 512))

  def test_scan_keeps_the_placement(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    params = _params(self._model(mesh, scan_layers=True, base_num_decoder_layers=8))
    kernel = params["decoder/cycles/gdn_0/linear_attn/in_proj_qkv/kernel"]
    self.assertEqual(_shard_shape(kernel), (2, 64, 512))
    self.assertAlmostEqual(_shard_fraction(params["decoder/ple/ple_embedding/ngram_embedding/embedding"]), 1 / 4)

  def test_dimension_the_mesh_cannot_divide_stays_replicated(self):
    """Six experts do not divide over `expert=4`: that dimension is left whole instead of failing construction."""
    mesh = create_mesh(ici={"expert": -1}, devices=self.devices)
    with jax.set_mesh(mesh):
      experts = modeling_qwen3_8_flash_next.Qwen3_8Experts(
          num_experts=6, emb_dim=64, moe_mlp_dim=32, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(0)
      )
    self.assertEqual(experts.gate_up_proj.get_metadata()["out_sharding"], P(None, _EXPERT_FSDP, None, _TP))
    self.assertEqual(_shard_shape(experts.gate_up_proj), (6, 64, 2, 32))

  def test_sharded_model_matches_single_device(self):
    """The annotations move bytes, not numbers: loss and gradients match the same weights on one device.

    FSDP and tensor parallelism are checked on separate meshes. On a mesh that shards the batch over only some of its
    axes (`fsdp=2, tensor=2`), the kernel gradient of a depthwise convolution comes back doubled with JAX 0.11.1 on
    CPU, with or without these annotations, so that combination cannot be compared here.
    """
    plain = m3_models.create_model(_mini_config(), mesh=None, rngs=nnx.Rngs(0))
    plain_params = _params(plain)
    tokens = jax.random.randint(jax.random.key(1), (4, 16), 4, 1024, dtype=jnp.int32)
    targets = jnp.roll(tokens, -1, axis=-1)
    segment_ids = jnp.ones((4, 16), dtype=jnp.int32).at[:, 8:].set(2)

    def loss_fn(model):
      loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          model, tokens, targets, segment_ids=segment_ids, router_aux_coeff=0.01
      )
      return loss

    for ici in ({"fsdp": -1}, {"tensor": -1}):
      mesh = create_mesh(ici=ici, devices=self.devices)
      sharded = self._model(mesh)
      for path, var in _params(sharded).items():
        plain_params[path].set_value(jnp.asarray(np.asarray(var[...])))
      with jax.set_mesh(mesh):
        loss_sharded, grads_sharded = nnx.jit(nnx.value_and_grad(loss_fn))(sharded)
      loss_plain, grads_plain = nnx.jit(nnx.value_and_grad(loss_fn))(plain)
      np.testing.assert_allclose(loss_sharded, loss_plain, rtol=1e-5, err_msg=str(ici))
      grads_plain = {"/".join(map(str, path)): var[...] for path, var in nnx.to_flat_state(grads_plain)}
      for path, var in nnx.to_flat_state(grads_sharded):
        name = "/".join(map(str, path))
        # Partitioned and single-device programs accumulate in different orders, so allow fp32 noise.
        np.testing.assert_allclose(var[...], grads_plain[name], atol=1e-4, rtol=1e-4, err_msg=f"{ici} {name}")

  def test_gradients_match_with_inputs_sharded_on_batch_and_sequence(self):
    """Inputs placed the way the training loop places them, batch on `fsdp` and sequence on `context`."""
    mesh = create_mesh(ici={"fsdp": 2, "context": 2}, devices=self.devices)
    sharded = self._model(mesh)
    plain = m3_models.create_model(_mini_config(), mesh=None, rngs=nnx.Rngs(0))
    plain_params = _params(plain)
    for path, var in _params(sharded).items():
      plain_params[path].set_value(jnp.asarray(np.asarray(var[...])))
    tokens = jax.random.randint(jax.random.key(1), (4, 32), 4, 1024, dtype=jnp.int32)
    segment_ids = jnp.ones((4, 32), dtype=jnp.int32)

    def loss_fn(model, tokens, segment_ids):
      loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          model, tokens, jnp.roll(tokens, -1, axis=-1), segment_ids=segment_ids, router_aux_coeff=0.01
      )
      return loss

    loss_plain, grads_plain = nnx.jit(nnx.value_and_grad(loss_fn))(plain, tokens, segment_ids)
    placed = jax.sharding.NamedSharding(mesh, P("fsdp", "context"))
    with jax.set_mesh(mesh):
      loss_sharded, grads_sharded = nnx.jit(nnx.value_and_grad(loss_fn))(
          sharded, jax.device_put(tokens, placed), jax.device_put(segment_ids, placed)
      )
    np.testing.assert_allclose(loss_sharded, loss_plain, rtol=1e-5)
    grads_plain = {"/".join(map(str, path)): var[...] for path, var in nnx.to_flat_state(grads_plain)}
    self.assertGreater(float(jnp.linalg.norm(grads_plain["mtp/fc_hidden/kernel"])), 0.0)
    for path, var in nnx.to_flat_state(grads_sharded):
      name = "/".join(map(str, path))
      np.testing.assert_allclose(var[...], grads_plain[name], atol=1e-4, rtol=1e-4, err_msg=name)

  def test_batch_axes_follow_the_strategy(self):
    """Activations are constrained through the strategy's `batch` axis, not a hard-coded list of mesh axes."""
    mesh = create_mesh(ici={"data": 2, "fsdp": 2}, devices=self.devices)
    with jax.set_mesh(mesh):
      self.assertEqual(modeling_qwen3_8_flash_next.get_batch_mesh_axes(), ("data", "fsdp"))
      self.assertEqual(modeling_qwen3_8_flash_next.get_batch_mesh_axes(_DataOnlyBatch()), ("data",))

      def constrained(batch, sharding):
        fn = lambda x: modeling_qwen3_8_flash_next.shard_batch(x, sharding)  # pylint: disable=unnecessary-lambda-assignment
        return "sharding_constraint" in str(jax.make_jaxpr(fn)(jnp.zeros((batch, 16))))

      # Two rows divide over `data=2` but not over `data=2 x fsdp=2`.
      self.assertTrue(constrained(4, None))
      self.assertFalse(constrained(2, None))
      self.assertTrue(constrained(2, _DataOnlyBatch()))

  def test_legacy_driver_sees_the_annotations(self):
    """`create_nnx_abstract_model` (the `from_pretrained` / training-engine path) places parameters by the specs."""
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    _, abstract = model_creation_utils.create_nnx_abstract_model(_mini_config(), mesh)
    state = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abstract, nnx.Param))}
    kernel = state["decoder/layers_0/linear_attn/in_proj_qkv/kernel"]
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(_DENSE_FSDP, None))
    self.assertEqual(kernel.get_value().sharding.spec, P(_DENSE_FSDP, None))
    self.assertEqual(
        state["decoder/layers_0/mlp/experts/gate_up_proj"].get_value().sharding.spec, P("expert", _EXPERT_FSDP, None, _TP)
    )
    self.assertEqual(state["decoder/layers_0/linear_attn/norm/scale"].get_value().sharding.spec, P(None))


def _grads(fn, *args):
  """`(value, flat gradients)` of `fn` over an NNX model, under `nnx.jit`."""
  value, grads = nnx.jit(nnx.value_and_grad(fn))(*args)
  return value, {"/".join(map(str, path)): var[...] for path, var in nnx.to_flat_state(grads)}


class SequenceShardingTest(absltest.TestCase):
  """The sequence stays sharded through the model, on `context` and, with EP = CP, on `expert`."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest(f"needs 4 devices, found {jax.device_count()}")
    self.devices = jax.devices()[:4]

  def test_length_axes_follow_the_strategy(self):
    ecp = DefaultSharding(expert_context_parallel=True)
    with jax.set_mesh(create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)):
      self.assertEqual(modeling_qwen3_8_flash_next.get_length_mesh_axes(), ())
      self.assertEqual(modeling_qwen3_8_flash_next.get_batch_mesh_axes(), ("fsdp", "expert"))
      self.assertEqual(modeling_qwen3_8_flash_next.get_length_mesh_axes(ecp), ("expert",))
      self.assertEqual(modeling_qwen3_8_flash_next.get_batch_mesh_axes(ecp), ("fsdp",))

      def placed(shape, sharding):
        out = jax.jit(lambda x: modeling_qwen3_8_flash_next.shard_batch(x * 2.0, sharding))(jnp.ones(shape))
        return out.sharding.shard_shape(out.shape)

      self.assertEqual(placed((4, 16, 8), None), (1, 16, 8))
      self.assertEqual(placed((4, 16, 8), ecp), (2, 8, 8))
      # A sequence the shards cannot divide stays whole; the batch is still split.
      self.assertEqual(placed((4, 15, 8), ecp), (2, 15, 8))
      # A batch the shards cannot divide stays whole; the sequence is still split.
      self.assertEqual(placed((3, 16, 8), ecp), (3, 8, 8))

  def test_context_parallel_delta_rule_matches_the_unsharded_rule(self):
    """Values and gradients of the sharded recurrence equal the single-device chunked rule."""
    b, s, h, d_k, d_v, chunk = 2, 64, 2, 8, 12, 8
    keys = jax.random.split(jax.random.key(0), 5)
    inputs = (
        jax.random.normal(keys[0], (b, s, h, d_k)),
        jax.random.normal(keys[1], (b, s, h, d_k)),
        jax.random.normal(keys[2], (b, s, h, d_v)),
        -jax.nn.softplus(jax.random.normal(keys[3], (b, s, h))),
        jax.nn.sigmoid(jax.random.normal(keys[4], (b, s, h))),
    )
    cotangent = jax.random.normal(jax.random.key(1), (b, s, h, d_v))

    def unsharded(*args):
      out, _ = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(*args, chunk_size=chunk)
      return jnp.sum(out * cotangent), out

    (_, ref_out), ref_grads = jax.jit(jax.value_and_grad(unsharded, argnums=(0, 1, 2, 3, 4), has_aux=True))(*inputs)

    for ici, batch_axes, length_axes in (
        ({"context": 4}, (), ("context",)),
        ({"fsdp": 2, "context": 2}, ("fsdp",), ("context",)),
        ({"context": 2, "expert": 2}, (), ("context", "expert")),
    ):
      mesh = create_mesh(ici=ici, devices=self.devices)

      def sharded(*args, batch_axes=batch_axes, length_axes=length_axes):
        out = modeling_qwen3_8_flash_next.context_parallel_chunk_gated_delta_rule(
            *args, batch_axes=batch_axes, length_axes=length_axes, chunk_size=chunk
        )
        return jnp.sum(out * cotangent), out

      with jax.set_mesh(mesh):
        (_, out), grads = jax.jit(jax.value_and_grad(sharded, argnums=(0, 1, 2, 3, 4), has_aux=True))(*inputs)
        self.assertEqual(out.sharding.shard_shape(out.shape)[1], s // 4 if not batch_axes else s // 2, str(ici))
        with self.assertRaisesRegex(ValueError, "multiple of"):
          # 4 or 2 sequence shards of 12 tokens are not whole chunks of 8.
          sharded(*(x[:, : 12 * math.prod(mesh.shape[ax] for ax in length_axes)] for x in inputs))
      np.testing.assert_allclose(out, ref_out, atol=1e-5, rtol=1e-4, err_msg=str(ici))
      for got, ref in zip(grads, ref_grads):
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4, err_msg=str(ici))

  def _compare_with_single_device(self, ici, tokens, *, devices=None, atol=1e-4, rtol=1e-4, **overrides):
    """Loss and gradients of a model built on `ici` against the same weights on one device."""
    plain = m3_models.create_model(_mini_config(**overrides), mesh=None, rngs=nnx.Rngs(0))
    plain_params = _params(plain)
    targets = jnp.roll(tokens, -1, axis=-1)
    segment_ids = jnp.ones(tokens.shape, dtype=jnp.int32).at[:, tokens.shape[1] // 2 :].set(2)

    def loss_fn(model):
      loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          model, tokens, targets, segment_ids=segment_ids, router_aux_coeff=0.01
      )
      return loss

    mesh = create_mesh(ici=ici, devices=devices or self.devices)
    sharded = m3_models.create_model(_mini_config(**overrides), mesh, rngs=nnx.Rngs(0))
    for path, var in _params(sharded).items():
      plain_params[path].set_value(jnp.asarray(np.asarray(var[...])))
    with jax.set_mesh(mesh):
      loss_sharded, grads_sharded = _grads(loss_fn, sharded)
    loss_plain, grads_plain = _grads(loss_fn, plain)
    np.testing.assert_allclose(loss_sharded, loss_plain, rtol=1e-5, err_msg=str(ici))
    for name, grad in grads_sharded.items():
      np.testing.assert_allclose(grad, grads_plain[name], atol=atol, rtol=rtol, err_msg=f"{ici} {name}")

  def test_sequence_sharded_model_matches_single_device(self):
    """Sharding the sequence, over `context` or (EP = CP) over `expert`, moves bytes, not numbers.

    64 tokens are a whole number of 16-token Gated DeltaNet chunks per shard, so the recurrence takes the
    context-parallel rule; the mock asserts that it does.
    """
    tokens = jax.random.randint(jax.random.key(1), (4, 64), 4, 1024, dtype=jnp.int32)
    rule = modeling_qwen3_8_flash_next.context_parallel_chunk_gated_delta_rule
    for ici, overrides in (
        ({"context": -1}, {}),
        ({"expert": -1}, {"expert_context_parallel": True}),
        ({"fsdp": 2, "expert": 2}, {"expert_context_parallel": True}),
    ):
      with mock.patch.object(modeling_qwen3_8_flash_next, "context_parallel_chunk_gated_delta_rule", wraps=rule) as cp:
        self._compare_with_single_device(ici, tokens, **overrides)
      self.assertEqual(cp.call_count, 3, f"{ici}: one call per Gated DeltaNet layer")
      length_axes = {call.kwargs["length_axes"] for call in cp.call_args_list}
      self.assertEqual(length_axes, {("context",) if "context" in ici else ("expert",)}, str(ici))

  def test_ragged_sequence_falls_back_to_the_unsharded_rule(self):
    """48 tokens over four shards are 12 each, not whole 16-token chunks: the recurrence gathers the sequence."""
    tokens = jax.random.randint(jax.random.key(1), (4, 48), 4, 1024, dtype=jnp.int32)
    rule = modeling_qwen3_8_flash_next.context_parallel_chunk_gated_delta_rule
    with mock.patch.object(modeling_qwen3_8_flash_next, "context_parallel_chunk_gated_delta_rule", wraps=rule) as cp:
      self._compare_with_single_device({"context": -1}, tokens)
    self.assertEqual(cp.call_count, 0)

  def test_flash_attention_keeps_queries_on_their_sequence_shard(self):
    """The Splash kernel runs per sequence shard, with the kernel's block tables split like the queries."""
    tokens = jax.random.randint(jax.random.key(1), (2, 256), 4, 1024, dtype=jnp.int32)
    splash = modeling_qwen3_8_flash_next.tokamax_splash_kernel
    with mock.patch.object(splash, "make_splash_mha", wraps=splash.make_splash_mha) as make_kernel:
      self._compare_with_single_device(
          {"context": 2},
          tokens,
          devices=self.devices[:2],
          qsa_attention_kernel="flash",
          index_select_budget=256,
      )
    self.assertIn(2, [call.kwargs.get("q_seq_shards") for call in make_kernel.call_args_list])

  def test_moe_preserves_3d_sequence_sharding_and_logical_axis_rules(self):
    """Qwen3_8SparseMoeBlock passes 3D (B, S, D) tokens to routed_experts_fused so executor.length stays ('context', 'expert')."""
    cfg_default = _mini_config(expert_context_parallel=False)
    cfg_epcp = _mini_config(expert_context_parallel=True)
    rules_default = dict(cfg_default.logical_axis_rules)
    rules_epcp = dict(cfg_epcp.logical_axis_rules)
    self.assertIn("expert", rules_default["activation_batch"])
    self.assertNotIn("expert", rules_epcp["activation_batch"])
    self.assertIn("expert", rules_epcp["activation_length"])

    tokens = jax.random.randint(jax.random.key(1), (2, 32), 4, 1024, dtype=jnp.int32)
    moe_mod = modeling_qwen3_8_flash_next.m3_moe
    with mock.patch.object(moe_mod, "routed_experts_fused", side_effect=lambda x, *a, **kw: x) as mocked_moe:
      mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
      model = m3_models.create_model(cfg_epcp, mesh, rngs=nnx.Rngs(0))
      with jax.set_mesh(mesh):
        out = model(tokens)
      self.assertEqual(out.shape, (2, 32, cfg_epcp.vocab_size))
    self.assertGreater(mocked_moe.call_count, 0)
    for call in mocked_moe.call_args_list:
      tok_arg = call.args[0]
      exec_cfg = call.kwargs["executor"]
      self.assertEqual(tok_arg.shape[:2], (2, 32))
      self.assertEqual(exec_cfg.length, ("context", "expert"))
      self.assertEqual(exec_cfg.batch, ("data", "fsdp", "fsdp_transpose"))

  def test_context_parallel_causal_conv1d_matches_unsharded(self):
    """Causal conv1d halo exchange (`ppermute` / `all_gather`) matches single-device `nnx.Conv(padding='CAUSAL')`."""
    b, s, c, k = 4, 32, 16, 4
    x = jax.random.normal(jax.random.key(0), (b, s, c), dtype=jnp.float32)
    kernel = jax.random.normal(jax.random.key(1), (k, 1, c), dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(2), (b, s, c), dtype=jnp.float32)

    def unsharded_fn(x_in, k_in):
      padded = jnp.pad(x_in, ((0, 0), (k - 1, 0), (0, 0)))
      out = jax.lax.conv_general_dilated(
          lhs=padded,
          rhs=k_in,
          window_strides=(1,),
          padding="VALID",
          dimension_numbers=("NWC", "WIO", "NWC"),
          feature_group_count=c,
      )
      return jnp.sum(out * cotangent), out

    (_, ref_out), (ref_dx, ref_dk) = jax.jit(jax.value_and_grad(unsharded_fn, argnums=(0, 1), has_aux=True))(x, kernel)

    for ici, batch_axes, length_axes, sharding in (
        ({"context": 4}, (), ("context",), DefaultSharding()),
        ({"fsdp": 2, "expert": 2}, ("fsdp",), ("expert",), DefaultSharding(expert_context_parallel=True)),
        ({"context": 2, "expert": 2}, (), ("context", "expert"), DefaultSharding(expert_context_parallel=True)),
    ):
      mesh = create_mesh(ici=ici, devices=self.devices)

      def sharded_fn(x_in, k_in, batch_axes=batch_axes, length_axes=length_axes, sharding=sharding):
        out = modeling_qwen3_8_flash_next._context_parallel_causal_conv1d(  # pylint: disable=protected-access
            x_in,
            k_in,
            batch_axes=batch_axes,
            length_axes=length_axes,
            sharding=sharding,
            dtype=jnp.float32,
        )
        return jnp.sum(out * cotangent), out

      with jax.set_mesh(mesh):
        (_, out), (dx, dk) = jax.jit(jax.value_and_grad(sharded_fn, argnums=(0, 1), has_aux=True))(x, kernel)
      np.testing.assert_allclose(out, ref_out, atol=1e-5, rtol=1e-5, err_msg=str(ici))
      np.testing.assert_allclose(dx, ref_dx, atol=1e-5, rtol=1e-5, err_msg=str(ici))
      np.testing.assert_allclose(dk, ref_dk, atol=1e-5, rtol=1e-5, err_msg=str(ici))

  def test_vocab_tiling_matches_untiled_loss_and_gradients(self):
    """`num_vocab_tiling > 1` produces identical loss and gradients to `num_vocab_tiling = 1` on a sharded mesh."""
    tokens = jax.random.randint(jax.random.key(3), (2, 32), 4, 1024, dtype=jnp.int32)
    targets = jnp.roll(tokens, -1, axis=-1)
    segment_ids = jnp.ones(tokens.shape, dtype=jnp.int32).at[:, 16:].set(2)
    mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)

    untiled_cfg = _mini_config(
        base_num_decoder_layers=1, use_ple_layer=False, expert_context_parallel=True, num_vocab_tiling=1
    )
    tiled_cfg = _mini_config(
        base_num_decoder_layers=1, use_ple_layer=False, expert_context_parallel=True, num_vocab_tiling=2
    )
    untiled_model = m3_models.create_model(untiled_cfg, mesh, rngs=nnx.Rngs(0))
    tiled_model = m3_models.create_model(tiled_cfg, mesh, rngs=nnx.Rngs(0))
    tiled_params = _params(tiled_model)
    for path, var in _params(untiled_model).items():
      tiled_params[path].set_value(var[...])

    def loss_fn(model):
      loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          model, tokens, targets, segment_ids=segment_ids, router_aux_coeff=0.01
      )
      return loss

    with jax.set_mesh(mesh):
      loss_untiled, grads_untiled = _grads(loss_fn, untiled_model)
      loss_tiled, grads_tiled = _grads(loss_fn, tiled_model)
      # Also verify the train.py sow-interception path when decoder_target_tokens is passed.
      ret = tiled_model(tokens, decoder_target_tokens=targets)
      self.assertIsNone(ret)
      sowed = tiled_model.decoder.hidden_states.get_value()[-1]
      self.assertEqual(sowed.shape, (2, 32, tiled_cfg.emb_dim))
      tile_logits = tiled_model.logits_from_hidden_states_for_vocab_tiling(sowed[:, :8, :])
      self.assertEqual(tile_logits.shape, (2, 8, tiled_cfg.vocab_size))

    np.testing.assert_allclose(loss_tiled, loss_untiled, rtol=1e-5)
    self.assertGreater(float(jnp.linalg.norm(grads_tiled["lm_head/kernel"])), 0.0)
    self.assertGreater(float(jnp.linalg.norm(grads_tiled["mtp/fc_hidden/kernel"])), 0.0)
    for name, grad in grads_tiled.items():
      np.testing.assert_allclose(grad, grads_untiled[name], atol=1e-4, rtol=1e-4, err_msg=name)


if __name__ == "__main__":
  absltest.main()
