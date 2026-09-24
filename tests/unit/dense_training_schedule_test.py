"""Small CPU tests for true-loss scheduling, gradient bookkeeping, and fusion."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.dense_training_schedule import make_training_schedule


def prefix_apply(params, data):
  return params["embedding"][data["inputs"]] * params["prefix_scale"]


def layer_apply(params, hidden, state, positions, segments):
  del positions, segments
  return jnp.tanh(hidden @ params["kernel"] + params["bias"]) * state["gain"]


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
      hidden = layer_apply(
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
  return jnp.sum(expert_outputs * routing[..., None], axis=-2) * state["scale"]


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
        hidden = apply(
            jax.tree.map(lambda x: x[layer], group_params), hidden,
            jax.tree.map(lambda x: x[layer], state), one["inputs_position"], one["inputs_segmentation"],
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
        for schedule in ("serial", "dual_pipe"):
          with self.subTest(layers=layers, microbatches=microbatches, schedule=schedule):
            step = make_training_schedule(prefix_apply, layer_apply, loss_apply, schedule)
            actual = jax.jit(step)(*args)
            self.assert_tree_allclose(actual, (loss, aux, layer_grads, boundary_grads))

  def test_empty_mask_produces_zero_sums(self):
    params, state, boundary, data = make_inputs(3, 3)
    data["targets_segmentation"] = jnp.zeros_like(data["targets_segmentation"])
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    result = jax.jit(step)(params, state, boundary, data)
    for leaf in jax.tree.leaves(result):
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_full_remat_matches_full_autodiff(self):
    args = make_inputs(3, 3)
    params, state, boundary, data = args
    (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference_loss, argnums=(0, 1), has_aux=True)(
        params, boundary, state, data
    )
    rematted_layer = jax.checkpoint(layer_apply, policy=jax.checkpoint_policies.nothing_saveable)
    step = make_training_schedule(prefix_apply, rematted_layer, loss_apply)
    self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_bfloat16_gradients_accumulate_in_float32(self):
    params, state, boundary, data = make_inputs(3, 3)
    params, state, boundary = jax.tree.map(lambda x: x.astype(jnp.bfloat16), (params, state, boundary))
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    _, _, layer_grads, boundary_grads = jax.jit(step)(params, state, boundary, data)
    for leaf in jax.tree.leaves((layer_grads, boundary_grads)):
      self.assertEqual(leaf.dtype, jnp.float32)

  def test_backward_and_forward_share_inner_scan(self):
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    graph = jax.make_jaxpr(step)(*make_inputs(3, 3))

    def scans(jaxpr):
      for equation in jaxpr.eqns:
        if equation.primitive.name == "scan":
          body = equation.params["jaxpr"].jaxpr
          yield equation.params["length"], body
          yield from scans(body)

    steady_bodies = [body for length, body in scans(graph.jaxpr) if length == 2]
    self.assertEqual(len(steady_bodies), 1)
    layer_bodies = [body for length, body in scans(steady_bodies[0]) if length == 3]
    self.assertEqual(len(layer_bodies), 1)
    scopes = [str(eq.source_info.name_stack).split("/") for eq in layer_bodies[0].eqns]
    self.assertTrue(any("backward" in scope for scope in scopes))
    self.assertTrue(any("forward" in scope for scope in scopes))

  def test_heterogeneous_groups_match_full_autodiff(self):
    for lengths in ((1, 3), (3, 1), (2, 2), (1, 2, 1), (3,)):
      for microbatches in (1, 2, 3):
        functions, args = make_grouped_inputs(lengths, microbatches)
        params, states, boundary, data = args
        reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
        (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(params, boundary)
        for schedule in ("serial", "dual_pipe"):
          with self.subTest(lengths=lengths, microbatches=microbatches, schedule=schedule):
            step = make_training_schedule(prefix_apply, functions, loss_apply, schedule)
            self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_heterogeneous_segments_keep_backward_forward_together(self):
    functions, args = make_grouped_inputs((1, 3), 3)
    graph = jax.make_jaxpr(make_training_schedule(prefix_apply, functions, loss_apply))(*args)
    steady = [
        eq for eq in graph.jaxpr.eqns
        if eq.primitive.name == "scan" and "steady_Bi_Fnext" in str(eq.source_info.name_stack)
    ]
    self.assertEqual(len(steady), 1)
    self.assertEqual(steady[0].params["length"], 2)
    segments = [eq for eq in steady[0].params["jaxpr"].jaxpr.eqns if eq.primitive.name == "scan"]
    self.assertEqual([eq.params["length"] for eq in segments], [1, 2, 1])
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
        hidden = apply(params, hidden, state, positions, segments)
        return hidden, {"overflow": received > state["capacity"], "received": received, "capacity": state["capacity"]}
      return jax.checkpoint(layer, policy=jax.checkpoint_policies.nothing_saveable)

    def reduce_metrics(aux):
      reducers = {"overflow": jnp.any, "received": jnp.max, "capacity": jnp.min}
      return {key: reducers.get(key, jnp.sum)(value, axis=0) for key, value in aux.items()}

    reference = lambda p, b: grouped_reference_loss(functions, p, b, states, data)
    (loss, expected_aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference, argnums=(0, 1), has_aux=True)(params, boundary)
    expected_aux.update(overflow=jnp.bool_(True), received=jnp.int32(25), capacity=jnp.int32(14))
    for schedule in ("serial", "dual_pipe"):
      with self.subTest(schedule=schedule):
        step = make_training_schedule(
            prefix_apply, tuple(with_metrics(apply) for apply in functions), loss_apply, schedule,
            layer_has_aux=True, reduce_aux=reduce_metrics,
        )
        actual = jax.jit(step)(params, states, boundary, data)
        self.assert_tree_allclose(actual, (loss, expected_aux, layer_grads, boundary_grads))

  def test_empty_groups_must_be_omitted(self):
    functions, args = make_grouped_inputs((0,), 1)
    step = make_training_schedule(prefix_apply, functions, loss_apply)
    with self.assertRaisesRegex(ValueError, "omit empty groups"):
      jax.jit(step)(*args)


if __name__ == "__main__":
  unittest.main()
