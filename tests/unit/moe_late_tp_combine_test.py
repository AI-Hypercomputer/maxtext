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
"""Tests for the `moe_late_tp_combine` routed-MoE sharding flag."""

import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.extend.core import ClosedJaxpr, Jaxpr
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import logical_to_mesh_axes
from tests.utils.test_helpers import get_test_config_path

_COLLECTIVE_PRIMITIVES = (
    "all_gather",
    "all_to_all",
    "ppermute",
    "psum",
    "psum_scatter",
    "ragged_all_to_all",
    "reduce_scatter",
)


def _build_config(late_tp_combine, use_ring_of_experts):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name=f"moe_late_tp_combine_{late_tp_combine}_roe_{use_ring_of_experts}",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=64,
      base_mlp_dim=64,
      base_moe_mlp_dim=64,
      dtype="float32",
      weight_dtype="float32",
      megablox=True,
      sparse_matmul=True,
      per_device_batch_size=2,  # TODO(b/450900273): sharding error if pdbs=1
      max_target_length=64,
      ici_expert_parallelism=2,
      ici_tensor_parallelism=4,
      use_ring_of_experts=use_ring_of_experts,
      moe_late_tp_combine=late_tp_combine,
      float32_gate_logits=True,
  )


def _build_model(cfg, mesh):
  return moe.get_routed_moe(
      name="MoeBlock",
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.mlp_dim,
      dtype=cfg.dtype,
  )


def _zeroed_params(model, hidden_states):
  """Builds params without executing ragged_all_to_all, which XLA:CPU does not lower."""

  def init():
    return model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, hidden_states)

  return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), jax.eval_shape(init)["params"])


def _scan_jaxpr(jaxpr, collectives, shard_map_out_specs):
  """Collects (primitive, axis_names) for every collective and every shard_map output spec."""
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in _COLLECTIVE_PRIMITIVES:
      axis_name = eqn.params.get("axis_name", ())
      collectives.append((eqn.primitive.name, (axis_name,) if isinstance(axis_name, str) else tuple(axis_name)))
    elif eqn.primitive.name == "shard_map":
      shard_map_out_specs.append(eqn.params["out_specs"][0])
    for param in jax.tree_util.tree_leaves(eqn.params, is_leaf=lambda p: isinstance(p, (Jaxpr, ClosedJaxpr))):
      param = param.jaxpr if isinstance(param, ClosedJaxpr) else param
      if isinstance(param, Jaxpr):
        _scan_jaxpr(param, collectives, shard_map_out_specs)


class MoeLateTpCombineTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() != 8:
      self.skipTest(f"needs 8 devices for expert=2 x tensor=4, got {jax.device_count()}")

  def _loss_and_grad(self, cfg, variables, hidden_states):
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = _build_model(cfg, mesh)

    def loss_fn(params, x):
      out, lb_loss, _ = model.apply({"params": params}, x)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      if variables is None:
        variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, hidden_states)
      out = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))(variables["params"], hidden_states)
    return variables, out

  def test_parity_off_vs_on(self):
    """Checks loss and gradient parity through the CPU-compatible ring-of-experts path."""
    cfg_off = _build_config(late_tp_combine=False, use_ring_of_experts=True)
    cfg_on = _build_config(late_tp_combine=True, use_ring_of_experts=True)
    hidden_states = jax.random.uniform(
        jax.random.PRNGKey(2345),
        (int(cfg_off.per_device_batch_size) * jax.device_count(), cfg_off.max_target_length, cfg_off.base_emb_dim),
        dtype=jnp.float32,
    )

    variables, (loss_off, grads_off) = self._loss_and_grad(cfg_off, None, hidden_states)
    _, (loss_on, grads_on) = self._loss_and_grad(cfg_on, variables, hidden_states)

    self.assertAlmostEqual(float(loss_on), float(loss_off), delta=1e-5 * max(1.0, abs(float(loss_off))))
    leaves_off, treedef = jax.tree_util.tree_flatten(grads_off)
    leaves_on, treedef_on = jax.tree_util.tree_flatten(grads_on)
    self.assertEqual(treedef, treedef_on)
    for i, (g_off, g_on) in enumerate(zip(leaves_off, leaves_on)):
      self.assertEqual(g_off.shape, g_on.shape, f"grad shape mismatch at leaf {i}")
      diff = float(jnp.max(jnp.abs(g_on - g_off)))
      scale = float(jnp.max(jnp.abs(g_off))) or 1.0
      self.assertLess(diff / scale, 1e-4, f"grad mismatch at leaf {i}: max abs diff {diff}")

  def test_no_tensor_axis_collectives_in_moe_body(self):
    """Checks production sharding by tracing the ragged_all_to_all path."""
    hidden_states = jnp.zeros((2 * jax.device_count(), 64, 64), dtype=jnp.float32)
    collectives, out_specs = {}, {}
    for late_tp_combine in (False, True):
      cfg = _build_config(late_tp_combine=late_tp_combine, use_ring_of_experts=False)
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = _build_model(cfg, mesh)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        params = _zeroed_params(model, hidden_states)
        jaxpr = jax.make_jaxpr(model.apply)({"params": params}, hidden_states)
      collectives[late_tp_combine], out_specs[late_tp_combine] = [], []
      _scan_jaxpr(jaxpr.jaxpr, collectives[late_tp_combine], out_specs[late_tp_combine])

    tensor_only_off = [c for c in collectives[False] if c[1] == ("tensor",)]
    tensor_only_on = [c for c in collectives[True] if c[1] == ("tensor",)]
    self.assertTrue(tensor_only_off, "expected a tensor-axis collective in the MoE body with the flag off")
    self.assertFalse(tensor_only_on, f"tensor-axis collectives remain with the flag on: {tensor_only_on}")

    self.assertTrue(
        any(c[1] == ("expert", "tensor") for c in collectives[True]),
        f"expected expert*tensor dispatch collectives, got {collectives[True]}",
    )

    self.assertEqual(out_specs[False], [P("expert", None, "tensor")])
    self.assertEqual(out_specs[True], [P("expert", "tensor", None)])

  def test_expert_weights_sharded_over_expert_and_tensor(self):
    cfg = _build_config(late_tp_combine=True, use_ring_of_experts=False)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      block = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          rngs=nnx.Rngs(params=0),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )
      self.assertEqual(block.wi_kernel_axes, ("exp_tp", "embed_moe", None))
      self.assertEqual(block.wo_kernel_axes, ("exp_tp", None, "embed_moe"))
      self.assertEqual(block.get_tensor_parallelism_size(), 1)
      self.assertEqual(block.get_expert_parallelism_size(), 8)
      self.assertEqual(
          logical_to_mesh_axes(("exp_tp", None, None), mesh=mesh, rules=cfg.logical_axis_rules)[0],
          ("expert", "tensor"),
      )


if __name__ == "__main__":
  unittest.main()
