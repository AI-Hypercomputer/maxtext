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

"""Small CPU tests for true-loss scheduling, gradient bookkeeping, and fusion."""

# Closures over loop variables are traced and consumed within the same iteration.
# pylint: disable=missing-function-docstring,cell-var-from-loop,unnecessary-lambda-assignment

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.ad_checkpoint import checkpoint_name

from maxtext.experimental.dualpipe_schedule import (
    _layer_passes,
    _reduce_aux,
    _slot_tables,
    make_training_schedule,
)


def prefix_apply(params, data):
  return params["embedding"][data["inputs"]] * params["prefix_scale"]


def layer_apply(params, hidden, state, positions, segments):
  del positions, segments
  return jnp.tanh(hidden @ params["kernel"] + params["bias"]) * state["gain"], {}


def loss_apply(params, hidden, data):
  # Tied input/output embedding exercises shared boundary-parameter gradients.
  logits = hidden @ params["embedding"].T + params["head_bias"]
  logprobs = jax.nn.log_softmax(logits)
  token_loss = -jnp.take_along_axis(logprobs, data["targets"][..., None], axis=-1)[..., 0]
  mask = data["targets_segmentation"] != 0
  loss = jnp.sum(jnp.where(mask, token_loss, 0))
  return loss, {"total_loss": loss, "total_weights": jnp.sum(mask)}


def make_inputs(layers, microbatches):
  keys = jax.random.split(jax.random.key(17), 5)
  layer_params = {
      "kernel": jax.random.normal(keys[0], (layers, 4, 4)) * 0.2,
      "bias": jax.random.normal(keys[1], (layers, 4)) * 0.1,
  }
  layer_state = {"gain": jnp.full((layers, 4), 0.9)}
  boundary_params = {
      "embedding": jax.random.normal(keys[2], (7, 4)) * 0.2,
      "prefix_scale": jnp.full((4,), 1.1),
      "head_bias": jnp.arange(7, dtype=jnp.float32) * 0.01,
  }
  data_shape = (microbatches, 2, 3)
  data = {
      "inputs": jax.random.randint(keys[3], data_shape, 0, 7),
      "inputs_position": jnp.broadcast_to(jnp.arange(3), data_shape),
      "inputs_segmentation": jnp.ones(data_shape, dtype=jnp.int32),
      "targets": jax.random.randint(keys[4], data_shape, 0, 7),
      # Unequal valid-token counts exercise unnormalized accumulation.
      "targets_segmentation": (jnp.arange(microbatches * 6).reshape(data_shape) % 3 != 0).astype(jnp.int32),
  }
  return layer_params, layer_state, boundary_params, data


def reference_loss(layer_params, boundary_params, layer_state, data):
  loss_total = jnp.float32(0)
  weights_total = jnp.int32(0)
  for microbatch in range(data["inputs"].shape[0]):
    one = jax.tree.map(lambda x: x[microbatch], data)
    hidden = prefix_apply(boundary_params, one)
    for layer in range(layer_params["kernel"].shape[0]):
      hidden, _ = layer_apply(
          jax.tree.map(lambda x: x[layer], layer_params),
          hidden,
          jax.tree.map(lambda x: x[layer], layer_state),
          one["inputs_position"],
          one["inputs_segmentation"],
      )
    loss, aux = loss_apply(boundary_params, hidden, one)
    loss_total += loss
    weights_total += aux["total_weights"]
  return loss_total, {"total_loss": loss_total, "total_weights": weights_total}


def expert_layer_apply(params, hidden, state, positions, segments):
  """A tiny soft-routed expert layer with a different parameter/residual tree."""
  del positions, segments
  expert_hidden = jnp.tanh(jnp.einsum("...d,edh->...eh", hidden, params["up"]))
  expert_outputs = jnp.einsum("...eh,ehd->...ed", expert_hidden, params["down"])
  routing = jax.nn.softmax(hidden @ params["router"], axis=-1)
  return jnp.sum(expert_outputs * routing[..., None], axis=-2) * state["scale"], {}


def named_layer_apply(params, hidden, state, positions, segments):
  """layer_apply with its pre-activation tagged for remat policies."""
  del positions, segments
  preactivation = checkpoint_name(hidden @ params["kernel"] + params["bias"], "query_proj")
  return jnp.tanh(preactivation) * state["gain"], {}


def named_expert_layer_apply(params, hidden, state, positions, segments):
  """expert_layer_apply with its expert hidden activations tagged for remat policies."""
  del positions, segments
  expert_hidden = checkpoint_name(jnp.tanh(jnp.einsum("...d,edh->...eh", hidden, params["up"])), "mlpwi")
  expert_outputs = jnp.einsum("...eh,ehd->...ed", expert_hidden, params["down"])
  routing = jax.nn.softmax(hidden @ params["router"], axis=-1)
  return jnp.sum(expert_outputs * routing[..., None], axis=-2) * state["scale"], {}


def make_grouped_inputs(lengths, microbatches):
  _, _, boundary, data = make_inputs(1, microbatches)
  functions, params, states = [], [], []
  for index, length in enumerate(lengths):
    if index % 2 == 0:
      weights, state, _, _ = make_inputs(length, microbatches)
      functions.append(layer_apply)
    else:
      keys = jax.random.split(jax.random.key(31 + index), 3)
      weights = {
          "up": jax.random.normal(keys[0], (length, 2, 4, 6)) * 0.2,
          "down": jax.random.normal(keys[1], (length, 2, 6, 4)) * 0.2,
          "router": jax.random.normal(keys[2], (length, 4, 2)) * 0.2,
      }
      state = {"scale": jnp.full((length,), 1.1)}
      functions.append(expert_layer_apply)
    params.append(weights)
    states.append(state)
  return tuple(functions), (tuple(params), tuple(states), boundary, data)


def grouped_reference_loss(functions, params, boundary, states, data):
  loss_sum, weights_sum = jnp.float32(0), jnp.int32(0)
  for microbatch in range(data["inputs"].shape[0]):
    one = jax.tree.map(lambda x: x[microbatch], data)
    hidden = prefix_apply(boundary, one)
    for apply, group_params, state in zip(functions, params, states):
      for layer in range(jax.tree.leaves(group_params)[0].shape[0]):
        hidden, _ = apply(
            jax.tree.map(lambda x: x[layer], group_params),
            hidden,
            jax.tree.map(lambda x: x[layer], state),
            one["inputs_position"],
            one["inputs_segmentation"],
        )
    loss, aux = loss_apply(boundary, hidden, one)
    loss_sum += loss
    weights_sum += aux["total_weights"]
  return loss_sum, {"total_loss": loss_sum, "total_weights": weights_sum}


class DenseTrainingScheduleTest(unittest.TestCase):

  def assert_tree_allclose(self, actual, expected):
    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-5, atol=2e-6)

  def test_loss_and_all_gradients_match_full_autodiff(self):
    for layers in (1, 3):
      for microbatches in (1, 2, 3):
        args = make_inputs(layers, microbatches)
        params, state, boundary, data = args
        (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference_loss, argnums=(0, 1), has_aux=True)(
            params, boundary, state, data
        )
        with self.subTest(layers=layers, microbatches=microbatches):
          step = make_training_schedule(prefix_apply, (layer_apply,), loss_apply)
          actual = jax.jit(step)((params,), (state,), boundary, data)
          self.assert_tree_allclose(actual, (loss, aux, (layer_grads,), boundary_grads))

  def test_empty_mask_produces_zero_sums(self):
    params, state, boundary, data = make_inputs(3, 3)
    data["targets_segmentation"] = jnp.zeros_like(data["targets_segmentation"])
    step = make_training_schedule(prefix_apply, (layer_apply,), loss_apply)
    result = jax.jit(step)((params,), (state,), boundary, data)
    for leaf in jax.tree.leaves(result):
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_full_remat_matches_full_autodiff(self):
    args = make_inputs(3, 3)
    params, state, boundary, data = args
    (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference_loss, argnums=(0, 1), has_aux=True)(
        params, boundary, state, data
    )
    rematted_layer = jax.checkpoint(layer_apply, policy=jax.checkpoint_policies.nothing_saveable)
    step = make_training_schedule(prefix_apply, (rematted_layer,), loss_apply)
    actual = jax.jit(step)((params,), (state,), boundary, data)
    self.assert_tree_allclose(actual, (loss, aux, (layer_grads,), boundary_grads))

  def test_bfloat16_gradients_accumulate_in_float32(self):
    params, state, boundary, data = make_inputs(3, 3)
    params, state, boundary = jax.tree.map(lambda x: x.astype(jnp.bfloat16), (params, state, boundary))
    step = make_training_schedule(prefix_apply, (layer_apply,), loss_apply)
    _, _, layer_grads, boundary_grads = jax.jit(step)((params,), (state,), boundary, data)
    for leaf in jax.tree.leaves((layer_grads, boundary_grads)):
      self.assertEqual(leaf.dtype, jnp.float32)

  def test_backward_and_forward_share_inner_scan(self):
    step = make_training_schedule(prefix_apply, (layer_apply,), loss_apply)
    params, state, boundary, data = make_inputs(3, 3)
    graph = jax.make_jaxpr(step)((params,), (state,), boundary, data)

    def scans(jaxpr):
      for equation in jaxpr.eqns:
        if equation.primitive.name == "scan":
          body = equation.params["jaxpr"].jaxpr
          yield equation.params["length"], body
          yield from scans(body)

    # Microbatches 1 and 2 form one period-2 pair, so the steady scan runs once over two fused steps.
    steady = [
        eq.params["jaxpr"].jaxpr
        for eq in graph.jaxpr.eqns
        if eq.primitive.name == "scan" and "steady_Bi_Fnext" in str(eq.source_info.name_stack)
    ]
    self.assertEqual(len(steady), 1)
    layer_bodies = [body for length, body in scans(steady[0]) if length == 3]
    self.assertEqual(len(layer_bodies), 2)
    for body in layer_bodies:
      scopes = [str(eq.source_info.name_stack).split("/") for eq in body.eqns]
      self.assertTrue(any("backward" in scope for scope in scopes))
      self.assertTrue(any("forward" in scope for scope in scopes))

  def test_heterogeneous_groups_match_full_autodiff(self):
    for lengths in ((1, 3), (3, 1), (2, 2), (1, 2, 1), (3,)):
      for microbatches in (1, 2, 3):
        functions, args = make_grouped_inputs(lengths, microbatches)
        params, states, boundary, data = args
        reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
        (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(
            params, boundary
        )
        with self.subTest(lengths=lengths, microbatches=microbatches):
          step = make_training_schedule(prefix_apply, functions, loss_apply)
          self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_full_remat_flag_matches_full_autodiff(self):
    for lengths in ((1, 3), (3, 1), (2, 2), (1, 2, 1), (3,), (3, 7)):
      for microbatches in (1, 2, 3, 4, 5):
        functions, args = make_grouped_inputs(lengths, microbatches)
        params, states, boundary, data = args
        reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
        (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(
            params, boundary
        )
        rematted = tuple(jax.checkpoint(f, policy=jax.checkpoint_policies.nothing_saveable) for f in functions)
        with self.subTest(lengths=lengths, microbatches=microbatches):
          step = make_training_schedule(prefix_apply, rematted, loss_apply)
          self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_custom_remat_policies_match_full_autodiff(self):
    policies = {
        "save_attention": jax.checkpoint_policies.save_only_these_names("query_proj"),
        "save_experts": jax.checkpoint_policies.save_only_these_names("mlpwi"),
        "save_both": jax.checkpoint_policies.save_only_these_names("query_proj", "mlpwi"),
        "save_nothing": jax.checkpoint_policies.nothing_saveable,
    }
    named = {layer_apply: named_layer_apply, expert_layer_apply: named_expert_layer_apply}
    for lengths in ((1, 3), (3, 1), (2, 2), (1, 2, 1), (3, 7)):
      for microbatches in (1, 2, 3, 5):
        functions, args = make_grouped_inputs(lengths, microbatches)
        params, states, boundary, data = args
        reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
        expected = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(params, boundary)
        (loss, aux), (layer_grads, boundary_grads) = expected
        for name, policy in policies.items():
          rematted = tuple(jax.checkpoint(named[f], policy=policy) for f in functions)
          with self.subTest(lengths=lengths, microbatches=microbatches, policy=name):
            step = make_training_schedule(prefix_apply, rematted, loss_apply)
            self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_saved_residuals_are_exactly_the_policy_names(self):
    weights, state, _, data = make_inputs(1, 1)
    weights, state = jax.tree.map(lambda x: x[0], (weights, state))
    hidden = jnp.ones((2, 3, 4))
    positions, segments = data["inputs_position"][0], data["inputs_segmentation"][0]
    expected = {
        # Backward of x @ W needs x: the hidden input is kept, nothing else.
        "nothing": [(2, 3, 4)],
        # The tagged pre-activation is kept in addition to the hidden input.
        "query_proj": [(2, 3, 4), (2, 3, 4)],
    }
    policies = {
        "nothing": jax.checkpoint_policies.nothing_saveable,
        "query_proj": jax.checkpoint_policies.save_only_these_names("query_proj"),
    }
    for name, policy in policies.items():
      forward, backward = _layer_passes(jax.checkpoint(named_layer_apply, policy=policy))
      out, saved, _ = jax.jit(forward)(weights, hidden, state, positions, segments)
      with self.subTest(policy=name):
        self.assertEqual([x.shape for x in saved], expected[name])
        # Relies on jax.vjp's pullback leaves keeping input identity: hidden is recognized as an input.
        self.assertTrue(forward.layouts[-1].needs_hidden)
        if name == "nothing":
          self.assertEqual(sum(forward.layouts[-1].kept), 0)
        # Weights/state are re-read from the model at backward time, never saved.
        self.assertTrue(all(x.shape != weights["kernel"].shape for x in saved))
        dweights, dhidden = jax.jit(backward)(weights, state, positions, segments, saved, jnp.ones_like(out))
        want = jax.vjp(lambda w, x: named_layer_apply(w, x, state, positions, segments)[0], weights, hidden)[1](
            jnp.ones_like(out)
        )
        self.assert_tree_allclose((dweights, dhidden), want)

  def test_residuals_the_backward_never_reads_are_not_saved(self):
    # A custom_vjp whose forward rule saves an extra residual that its backward ignores. jax.vjp's
    # pullback still closes over it (the scan path would prune it); the schedule must not store it.
    @jax.custom_vjp
    def op(x, w):
      return jnp.tanh(x @ w)

    def op_fwd(x, w):
      y = jnp.tanh(x @ w)
      return y, (x, w, y, checkpoint_name(jnp.exp(x) @ w, "big_unused"))

    def op_bwd(res, dy):
      x, w, y, _ = res
      dz = dy * (1 - y * y)
      return dz @ w.T, x.reshape(-1, x.shape[-1]).T @ dz.reshape(-1, dz.shape[-1])

    op.defvjp(op_fwd, op_bwd)

    def apply(params, hidden, state, positions, segments):
      del state, positions, segments
      return op(hidden, params["kernel"]), {}

    weights = {"kernel": jax.random.normal(jax.random.key(1), (4, 4)) * 0.2}
    hidden = jax.random.normal(jax.random.key(2), (2, 3, 4))
    args = (weights, hidden, {}, jnp.zeros((2, 3), jnp.int32), jnp.ones((2, 3), jnp.int32))
    policy = jax.checkpoint_policies.save_only_these_names("big_unused")
    forward, backward = _layer_passes(jax.checkpoint(apply, policy=policy))
    out, saved, _ = jax.jit(forward)(*args)
    layout = forward.layouts[0]
    pruned = [s is None and not k for s, k in zip(layout.sources, layout.kept)]
    self.assertEqual(sum(pruned), 1, "exactly the unread saved residual should be pruned")
    self.assertEqual(len(saved), sum(layout.kept) + int(layout.needs_hidden))
    dweights, dhidden = jax.jit(backward)(weights, {}, args[3], args[4], saved, jnp.ones_like(out))
    want = jax.vjp(lambda w, x: apply(w, x, {}, None, None)[0], weights, hidden)[1](jnp.ones_like(out))
    self.assert_tree_allclose((dweights, dhidden), want)

  def test_residual_analysis_degrades_when_private_jax_module_is_missing(self):
    # If the private partial-eval module cannot be imported, every residual is kept (correct, just not pruned).
    import sys  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from maxtext.experimental.dualpipe_schedule import _used_pullback_leaves  # pylint: disable=import-outside-toplevel

    leaves = [jnp.ones((2, 3)), jnp.ones((4,))]
    with mock.patch.dict(sys.modules, {"jax._src.interpreters.partial_eval": None}):
      used = _used_pullback_leaves(jax.tree.structure(leaves), leaves, jnp.ones((2, 3)))
    self.assertEqual(used, (True, True))

  def test_slot_tables_never_overwrite_unread_residuals(self):
    for layer_pool, microbatches in (([0] * 10, 4), ([0] * 3 + [1] * 7, 4), ([0, 1, 1, 0], 5), ([0] * 1 + [1] * 3, 1)):
      tables, sizes = _slot_tables(microbatches, layer_pool, max(layer_pool) + 1)
      layers = len(layer_pool)
      holder = {}  # (pool, slot) -> (microbatch, layer) still to be read

      def write(m, layer):
        key = (layer_pool[layer], int(tables[layer_pool[layer], m, layer]))
        self.assertNotIn(key, holder, f"slot {key} overwritten before it was read")
        self.assertLess(key[1], sizes[key[0]])
        holder[key] = (m, layer)

      def read(m, layer):
        key = (layer_pool[layer], int(tables[layer_pool[layer], m, layer]))
        self.assertEqual(holder.pop(key), (m, layer))

      for layer in range(layers):
        write(0, layer)
      for m in range(1, microbatches):
        for k in range(layers):
          read(m - 1, layers - 1 - k)
          write(m, k)
      for layer in reversed(range(layers)):
        read(microbatches - 1, layer)

  def test_slot_pool_sizes(self):
    # Full remat: one shared pool, exactly L slots, each fused write reuses the slot just read.
    tables, sizes = _slot_tables(4, [0] * 10, 1)
    self.assertEqual(sizes, [10])
    for m in range(1, 4):
      for k in range(10):
        self.assertEqual(tables[0, m, k], tables[0, m - 1, 9 - k])
    # Separate dense(3)/MoE(7) pools: the old microbatch's dense residuals are read
    # last while the new ones are written first, so the dense pool needs 2x3 slots.
    _, sizes = _slot_tables(4, [0] * 3 + [1] * 7, 2)
    self.assertEqual(sizes, [6, 7])

  @pytest.mark.gpu_only
  @unittest.skipUnless(jax.default_backend() == "gpu", "host offload needs a GPU backend")
  def test_offloaded_residuals_stay_on_host(self):
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["query_proj"],
        names_which_can_be_offloaded=["mlpwi"],
        offload_src="device",
        offload_dst="pinned_host",
    )
    functions, args = make_grouped_inputs((2, 3), 3)
    params, states, boundary, data = args
    reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
    (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(
        params, boundary
    )
    named = {layer_apply: named_layer_apply, expert_layer_apply: named_expert_layer_apply}
    rematted = tuple(jax.checkpoint(named[f], policy=policy) for f in functions)
    step = make_training_schedule(prefix_apply, rematted, loss_apply)
    self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))
    hlo = jax.jit(step).lower(*args).compile().as_text()
    # The offloaded expert activations [L-slots, 2, 3, 2, 6] live in host memory space S(5).
    self.assertRegex(hlo, r"f32\[\d+,2,3,2,6\]\{[0-9,]+:S\(5\)\}")

  def test_heterogeneous_segments_keep_backward_forward_together(self):
    functions, args = make_grouped_inputs((1, 3), 3)
    graph = jax.make_jaxpr(make_training_schedule(prefix_apply, functions, loss_apply))(*args)
    steady = [
        eq
        for eq in graph.jaxpr.eqns
        if eq.primitive.name == "scan" and "steady_Bi_Fnext" in str(eq.source_info.name_stack)
    ]
    self.assertEqual(len(steady), 1)
    # One period-2 pair (microbatches 1 and 2), each fused step split into the same three segments.
    self.assertEqual(steady[0].params["length"], 1)
    segments = [eq for eq in steady[0].params["jaxpr"].jaxpr.eqns if eq.primitive.name == "scan"]
    self.assertEqual([eq.params["length"] for eq in segments], [1, 2, 1, 1, 2, 1])
    for segment in segments:
      scopes = [str(eq.source_info.name_stack).split("/") for eq in segment.params["jaxpr"].jaxpr.eqns]
      self.assertTrue(any("backward" in scope for scope in scopes))
      self.assertTrue(any("forward" in scope for scope in scopes))

  def test_heterogeneous_full_remat_preserves_aux_and_gradients(self):
    functions, (params, states, boundary, data) = make_grouped_inputs((1, 3), 3)
    data["inputs_position"] += jnp.arange(3)[:, None, None] * 10
    for index, (length, state) in enumerate(zip((1, 3), states)):
      state["received"] = jnp.arange(length, dtype=jnp.int32) + index
      state["capacity"] = jnp.full((length,), 15 - index, jnp.int32)

    def with_metrics(apply):
      def layer(params, hidden, state, positions, segments):
        received = state["received"] + jnp.max(positions)
        hidden, _ = apply(params, hidden, state, positions, segments)
        return hidden, {
            "te_moe_capacity_overflow": received > state["capacity"],
            "te_moe_max_total_recv_tokens": received,
            "te_moe_recv_capacity_per_rank": state["capacity"],
        }

      return jax.checkpoint(layer, policy=jax.checkpoint_policies.nothing_saveable)

    reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
    (loss, expected_aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(
        params, boundary
    )
    expected_aux.update(
        te_moe_capacity_overflow=jnp.bool_(True),
        te_moe_max_total_recv_tokens=jnp.int32(25),
        te_moe_recv_capacity_per_rank=jnp.int32(14),
    )
    step = make_training_schedule(prefix_apply, tuple(with_metrics(apply) for apply in functions), loss_apply)
    actual = jax.jit(step)(params, states, boundary, data)
    self.assert_tree_allclose(actual, (loss, expected_aux, layer_grads, boundary_grads))

  def test_empty_groups_must_be_omitted(self):
    functions, args = make_grouped_inputs((0,), 1)
    step = make_training_schedule(prefix_apply, functions, loss_apply)
    with self.assertRaisesRegex(ValueError, "omit empty groups"):
      jax.jit(step)(*args)

  def test_fixed_metric_reductions(self):
    reduced = _reduce_aux(
        {
            "te_moe_capacity_overflow": jnp.array([False, True, False]),
            "te_moe_max_total_recv_tokens": jnp.array([7, 13, 5]),
            "te_moe_recv_capacity_per_rank": jnp.array([12, 16, 14]),
            "total_loss": jnp.array([1.0, 2.0, 3.0]),
            "total_weights": jnp.array([2, 4, 1]),
        }
    )
    self.assert_tree_allclose(
        reduced,
        {
            "te_moe_capacity_overflow": jnp.bool_(True),
            "te_moe_max_total_recv_tokens": jnp.int32(13),
            "te_moe_recv_capacity_per_rank": jnp.int32(12),
            "total_loss": jnp.float32(6),
            "total_weights": jnp.int32(7),
        },
    )
    self.assertEqual(reduced["te_moe_capacity_overflow"].dtype, jnp.bool_)
    self.assertEqual(_reduce_aux({}), {})

  def test_groups_require_tuples(self):
    with self.assertRaisesRegex(TypeError, "tuple"):
      make_training_schedule(prefix_apply, layer_apply, loss_apply)
    params, state, boundary, data = make_inputs(1, 1)
    step = make_training_schedule(prefix_apply, (layer_apply,), loss_apply)
    with self.assertRaisesRegex(ValueError, "parameter/state tuples"):
      jax.jit(step)(params, state, boundary, data)


if __name__ == "__main__":
  unittest.main()
