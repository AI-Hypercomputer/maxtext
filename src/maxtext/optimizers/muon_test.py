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

"""Tests for the Muon optimizer in muon.py."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from maxtext.optimizers import muon as _muon
import numpy as np
import optax

UNSPECIFIED = object()


def get_updates(
    params,
    muon_weight_dimension_numbers=UNSPECIFIED,
    preconditioning="frobenius",
):
  """Computes optimizer updates for test parameters."""
  if muon_weight_dimension_numbers is UNSPECIFIED:
    opt = _muon.muon(learning_rate=1e-3, preconditioning=preconditioning)
  else:
    opt = _muon.muon(
        learning_rate=1e-3,
        preconditioning=preconditioning,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
    )
  state = opt.init(params)
  # assume loss = 1/2 * sum(params ** 2)
  grad = params
  updates, state = opt.update(grad, state, params=params)
  return updates, state


def _setup_mixed_tensor_target_complex(dtype):
  """Complex version of _common_test._setup_mixed_tensor_target."""
  initial_params = jnp.zeros((2, 2), dtype=dtype)
  final_params = jnp.array(
      [[1.0 + 2.0j, 0.0], [-1.0 + 1.0j, 1.0 - 3.0j]],
      dtype=dtype,
  )

  def obj_fn(params):
    return jnp.sum(jnp.abs(params - final_params) ** 2)

  return initial_params, final_params, obj_fn


class MuonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "2d_tuple_axes",
          "input_shape": (2, 3),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1),
          "expected_flat_shape": (1, 1, 2, 3),
      },
      {
          "testcase_name": "3d_batch_axis",
          "input_shape": (2, 3, 4),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=2),
          "expected_flat_shape": (1, 3, 2, 4),
      },
      {
          "testcase_name": "3d_negative_axes_indices",
          "input_shape": (2, 3, 4),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=-3, output_axis=-1),
          "expected_flat_shape": (1, 3, 2, 4),
      },
      {
          "testcase_name": "4d_multiple_batch_axes",
          "input_shape": (2, 3, 4, 5),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3),
          "expected_flat_shape": (1, 6, 4, 5),
      },
      {
          "testcase_name": "4d_multiple_output_axes",
          "input_shape": (2, 3, 4, 5),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=(0, 3)),
          "expected_flat_shape": (1, 3, 4, 10),
      },
  )
  def test_reshape_inverse(self, input_shape, dim_nums, expected_flat_shape):
    x = jnp.arange(math.prod(input_shape), dtype=jnp.float32).reshape(input_shape)
    reshape_fn, inverse_fn, _ = _muon.get_reshape_fns(x, dim_nums)
    reshaped_x = reshape_fn(x)
    reconstructed_x = inverse_fn(reshaped_x)
    # Check flat shape (sharded_batch, unsharded_batch, reduction, output)
    self.assertEqual(reshaped_x.shape, expected_flat_shape)
    # Check inverse shape and value
    self.assertEqual(reconstructed_x.shape, x.shape)
    chex.assert_trees_all_close(reconstructed_x, x)

  @parameterized.named_parameters(("frobenius", "frobenius"), ("aol", "aol"), ("schatten", "schatten"))
  def test_callable_weight_dim_nums(self, preconditioning):
    # Case 1: a dim nums for all weights, no matter if they're muon.
    def weight_dim_nums_fn(params):
      def fn_(x):
        return _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None

      return jax.tree.map(fn_, params)

    opt = _muon.muon(
        learning_rate=1e-3,
        preconditioning=preconditioning,
        muon_weight_dimension_numbers=weight_dim_nums_fn,
    )
    params = {"w1": jnp.ones((10, 10)), "w2": jnp.ones((2, 10))}
    state = opt.init(params)
    _, _ = opt.update(params, state, params=params)

    # Case 2: a None inserted for parameters that are not muon.
    def weight_dim_nums_fn(params):  # pylint: disable=function-redefined
      del params
      return {"w1": _muon.MuonDimensionNumbers(), "w2": None}

    opt = _muon.muon(
        learning_rate=1e-3,
        preconditioning=preconditioning,
        muon_weight_dimension_numbers=weight_dim_nums_fn,
    )
    params = {"w1": jnp.ones((10, 10)), "w2": jnp.ones((2, 10))}
    state = opt.init(params)
    _, _ = opt.update(params, state, params=params)

  @parameterized.named_parameters(("frobenius", "frobenius"), ("aol", "aol"), ("schatten", "schatten"))
  def test_reshape_update_for_square_parameter_matches_muon_without_dim_nums(self, preconditioning):
    # Use 2D parameter (10, 10) with no dim nums as groundtruth
    key = jax.random.key(0)
    params_sq = {"w": jax.random.normal(key, shape=(10, 10))}
    updates_sq, _ = get_updates(params_sq, preconditioning=preconditioning)
    # Test: 2D parameter (10, 10) with trivial dim nums
    dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1)}
    reshape_updates_sq, _ = get_updates(
        params_sq,
        preconditioning=preconditioning,
        muon_weight_dimension_numbers=dim_nums,
    )
    chex.assert_trees_all_close(updates_sq, reshape_updates_sq, rtol=1e-8, atol=1e-8)

  @parameterized.named_parameters(("frobenius", "frobenius"), ("aol", "aol"), ("schatten", "schatten"))
  def test_reshape_and_update_single_param(self, preconditioning):
    # Use 2D parameter (10, 12) with no dimension numbers as groundtruth
    key = jax.random.key(0)
    w = jax.random.normal(key, shape=(10, 12))
    params = {"w": w}
    updates, _ = get_updates(params, preconditioning=preconditioning)

    with self.subTest("2D with dimension numbers, (10, 12)"):
      # Test 1: 2D with dimension numbers, (10, 12)
      params = {"w": w}
      dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1)}
      reshape_updates, _ = get_updates(
          params,
          preconditioning=preconditioning,
          muon_weight_dimension_numbers=dim_nums,
      )
      chex.assert_trees_all_close(updates, reshape_updates, rtol=1e-8, atol=1e-8)

    with self.subTest("4D with dim nums, (10, 12) -> (4, 1, 10, 3)"):
      # Test 2: 4D with dim nums, (10, 12) -> (4, 1, 10, 3)
      def reshape_fn_1(x):
        return x.reshape(10, 3, 1, 4).transpose(3, 2, 0, 1)

      reshape_params = {"w": reshape_fn_1(w)}
      dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=(2,), output_axis=(0, 3))}
      reshape_updates, _ = get_updates(
          reshape_params,
          preconditioning=preconditioning,
          muon_weight_dimension_numbers=dim_nums,
      )
      chex.assert_trees_all_close(
          jax.tree.map(reshape_fn_1, updates),
          reshape_updates,
          rtol=1e-8,
          atol=1e-8,
      )

    with self.subTest("4D with dim_nums, (10, 12) -> (5, 12, 1, 2)"):
      # Test 3: 4D with dim_nums, (10, 12) -> (5, 12, 1, 2)
      def reshape_fn_2(x):
        return x.reshape(2, 1, 5, 12).transpose(2, 3, 1, 0)

      reshape_params = {"w": reshape_fn_2(w)}
      dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=(0, 3), output_axis=(1,))}
      reshape_updates, _ = get_updates(
          reshape_params,
          preconditioning=preconditioning,
          muon_weight_dimension_numbers=dim_nums,
      )
      chex.assert_trees_all_close(
          jax.tree.map(reshape_fn_2, updates),
          reshape_updates,
          rtol=1e-5,
          atol=1e-5,
      )

  @parameterized.named_parameters(("frobenius", "frobenius"), ("aol", "aol"), ("schatten", "schatten"))
  def test_dim_nums_combinations(self, preconditioning):
    def get_muon_mu(state):
      return state[0]["muon"][0][0][1]

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=(1,), output_axis=(2,))

    # Test 1: full dim_nums
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((2, 3, 4))}
    dim_nums = {"w1": dim_num, "w2": dim_num}
    _, state = get_updates(params, dim_nums, preconditioning=preconditioning)
    self.assertNotIsInstance(get_muon_mu(state)["w1"], optax.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"], optax.MaskedNode)

    # Test 2: no dim_nums
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((3, 4))}
    _, state = get_updates(params, preconditioning=preconditioning)
    self.assertIsInstance(get_muon_mu(state)["w1"], optax.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"], optax.MaskedNode)

    # Test 3: partial dim_nums with none
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((2, 3, 4))}
    dim_nums = {"w1": None, "w2": dim_num}
    _, state = get_updates(params, dim_nums, preconditioning=preconditioning)
    self.assertIsInstance(get_muon_mu(state)["w1"], optax.MaskedNode)

    # Test 4: prefix None, full dim_nums
    params = {
        "w1": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
        "w2": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
    }
    dim_num = _muon.MuonDimensionNumbers()
    dim_nums = {"w1": None, "w2": {"a": dim_num, "b": None}}
    _, state = get_updates(params, dim_nums, preconditioning=preconditioning)
    state_structure = jax.tree.structure(get_muon_mu(state), is_leaf=lambda x: isinstance(x, optax.MaskedNode))
    self.assertEqual(state_structure, jax.tree.structure(params))
    self.assertIsInstance(get_muon_mu(state)["w1"]["a"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w1"]["b"], optax.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"]["a"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["b"], optax.MaskedNode)

    # Test 5: prefix None and dim_nums
    params = {
        "w1": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
        "w2": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
    }
    dim_num = _muon.MuonDimensionNumbers()
    dim_nums = {"w1": dim_num, "w2": None}
    _, state = get_updates(params, dim_nums)
    state_structure = jax.tree.structure(get_muon_mu(state), is_leaf=lambda x: isinstance(x, optax.MaskedNode))
    self.assertEqual(state_structure, jax.tree.structure(params))
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["a"], optax.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["b"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["a"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["b"], optax.MaskedNode)

  def test_newton_schulz(self):
    """Test that Newton--Schulz orhogonalizes/unitiarizes correctly."""
    mat_real = jax.random.normal(jax.random.key(0), (4, 3), dtype=jnp.float32)
    mat_complex = jax.random.normal(jax.random.key(0), (4, 3), dtype=jnp.complex64)
    ns_coeffs = jnp.array([2.0, -1.5, 0.5])

    if jax.default_backend() == "tpu":
      atol, rtol = 1e-2, 1e-2
    else:
      atol, rtol = 1e-5, 1e-5

    precond_fn = _muon.get_precond_fn("frobenius", 1e-12)
    ns_step_fn = _muon.NS_ITERATORS["frobenius"]

    # For real matrices, Newton--Schulz should produce an orthonormal matrix
    mat_real_orth = _muon.orthogonalize(
        mat_real,
        ns_coeffs,
        ns_steps=20,
        precond_fn=precond_fn,
        ns_step_fn=ns_step_fn,
        dim_nums=_muon.MuonDimensionNumbers(0, 1),
    )

    gram_real = mat_real_orth.T @ mat_real_orth
    with self.subTest("Real Newton--Schulz produces an orthonormal matrix"):
      msg = "Real Newton–Schulz did not orthogonalize correctly.\nGram:" f" \n{gram_real}"
      np.testing.assert_allclose(
          gram_real,
          jnp.eye(mat_real.shape[1]),
          atol=atol,
          rtol=rtol,
          err_msg=msg,
      )
    # For complex matrices, Newton--Schulz should produce a unitary matrix
    mat_complex_orth = _muon.orthogonalize(
        mat_complex,
        ns_coeffs,
        ns_steps=10,
        precond_fn=precond_fn,
        ns_step_fn=ns_step_fn,
        dim_nums=_muon.MuonDimensionNumbers(0, 1),
    )

    gram_complex = mat_complex_orth.conj().T @ mat_complex_orth
    with self.subTest("Complex Newton--Schulz produces a unitary matrix"):
      msg = "Complex Newton–Schulz did not produce a unitary matrix." f"\nGram:\n{gram_complex}"
      np.testing.assert_allclose(
          gram_complex,
          jnp.eye(mat_complex.shape[1]),
          atol=atol,
          rtol=rtol,
          err_msg=msg,
      )

    with self.subTest("Output shape is preserved for complex matrices"):
      # Check that the output shape is preserved.
      self.assertEqual(mat_complex_orth.shape, mat_complex.shape)

  @parameterized.product(
      target=(_setup_mixed_tensor_target_complex,),
      dtype=("complex64",),
      adaptive=(True, False),
      nesterov=(True, False),
  )
  def test_complex_mixed_target(self, target, dtype, adaptive, nesterov):
    """Test Muon optimizer on a complex mixed tensor optimization target."""
    dtype = getattr(jnp, dtype)

    opt = _muon.muon(learning_rate=1e-2, adaptive=adaptive, nesterov=nesterov, mu_dtype=dtype)
    initial_params, final_params, obj_fn = target(dtype)

    @jax.jit
    def step(params, state):
      _, updates = jax.value_and_grad(obj_fn)(params)
      updates = jax.tree.map(jnp.conj, updates)
      updates, state = opt.update(updates, state, params)
      params = optax.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)

    def f(params_state, _):
      return step(*params_state), None

    (params, _), _ = jax.lax.scan(f, (params, state), length=1000)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @parameterized.named_parameters(
      ("standard_square", "frobenius", (100, 100)),
      ("standard_tall", "frobenius", (100, 50)),
      ("standard_wide", "frobenius", (50, 100)),
      ("aol_square", "aol", (100, 100)),
      ("aol_tall", "aol", (100, 50)),
      ("aol_wide", "aol", (50, 100)),
  )
  def test_muon_orthogonalization_modes(self, preconditioning, shape):
    """Tests that Muon runs and produces near-orthogonal updates."""
    key = jax.random.key(42)
    params = {"w": jax.random.normal(key, shape)}

    # Use LR=1.0 and no weight decay to inspect the raw orthogonalization.
    opt = _muon.muon(learning_rate=1.0, weight_decay=0.0, preconditioning=preconditioning)
    state = opt.init(params)
    updates, _ = opt.update(params, state, params=params)

    # Check shape preservation
    self.assertEqual(updates["w"].shape, shape)

    # Check Near-Orthogonality (Spectral Norm Constraint)
    if shape[0] == shape[1]:
      s = jnp.linalg.svd(updates["w"], compute_uv=False)
      max_s = jnp.max(s)
      min_s = jnp.min(s)
      self.assertLess(max_s, 2.0, msg=f"Max singular value {max_s} too high")
      self.assertGreater(min_s, 0.1, msg=f"Min singular value {min_s} too low")

  def test_aol_numerical_difference(self):
    """Ensures that AOL=True produces different updates than Standard Muon."""
    params = {"w": jnp.eye(8) * 2.0}

    def _get_updates(preconditioning, **kwargs):
      opt = _muon.muon(learning_rate=0.1, preconditioning=preconditioning, **kwargs)
      updates, _ = opt.update(params, opt.init(params), params)
      for leaf in jax.tree_util.tree_leaves(updates):
        self.assertFalse(jnp.isnan(leaf).any(), f"Found NaN values in {preconditioning}")
      return updates

    u_std = _get_updates("frobenius")
    u_aol = _get_updates("aol", ns_coeffs="dion")
    u_schatten = _get_updates("schatten", ns_coeffs="dion")

    with self.assertRaises(AssertionError, msg="AOL should differ from Frobenius and Schatten"):
      chex.assert_trees_all_close(u_std, u_aol)
      chex.assert_trees_all_close(u_schatten, u_aol)

  @parameterized.named_parameters(("frobenius", "frobenius"), ("aol", "aol"), ("schatten", "schatten"))
  def test_orthogonality(self, preconditioning):
    """Ensures that updates satisfy approximate orthogonality (U^T U ≈ I)."""
    params = {"w": jnp.eye(8) * 2.0}
    opt = _muon.muon(learning_rate=0.1, preconditioning=preconditioning)
    updates, _ = opt.update(params, opt.init(params), params)
    w_update = updates["w"]

    for leaf in jax.tree_util.tree_leaves(updates):
      self.assertFalse(jnp.isnan(leaf).any(), "Found NaN values in updates")

    # Check Orthogonality
    gram = jnp.dot(w_update.T, w_update)
    gram = gram / jnp.max(gram)
    ortho_error = jnp.linalg.norm(gram - jnp.eye(gram.shape[0]))
    self.assertLess(ortho_error, 1e-3, f"Orthogonality error too high: {ortho_error}")

  @parameterized.parameters(True, False)
  def test_use_all_to_all_option(self, use_all_to_all):
    """Ensures use_all_to_all option runs cleanly on multi-dimensional tensors."""
    key = jax.random.key(0)
    devices = jax.devices()
    if len(devices) >= 2:
      mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), axis_names=("x", "y"))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None, None, None))
      w = jax.device_put(jax.random.normal(key, (2, 3, 4, 5)), sharding)
      dim_nums = _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
    else:
      w = jax.random.normal(key, (2, 3, 4, 5))
      dim_nums = _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3)

    params = {"w": w}
    opt = _muon.muon(
        learning_rate=0.1,
        muon_weight_dimension_numbers={"w": dim_nums},
        use_all_to_all=use_all_to_all,
    )
    state = opt.init(params)
    updates, _ = opt.update(params, state, params)
    self.assertEqual(updates["w"].shape, (2, 3, 4, 5))
    self.assertFalse(jnp.isnan(updates["w"]).any())
    self.assertFalse(jnp.isinf(updates["w"]).any())

  def test_ns_coeffs_schedule_slicing(self):
    """Passing more coeffs than ns_steps should slice the trailing ns_steps."""
    params = {"w": jnp.ones((4, 4))}
    coeffs_12 = jnp.ones((12, 3))
    opt = _muon.scale_by_muon(ns_coeffs=coeffs_12, ns_steps=10)
    state = opt.init(params)
    self.assertEqual(state.ns_coeffs.shape, (10, 3))

  def test_ns_coeffs_insufficient_raises(self):
    """Passing fewer coeffs than ns_steps must raise ValueError."""
    params = {"w": jnp.ones((4, 4))}
    coeffs_3 = jnp.ones((3, 3))
    opt = _muon.scale_by_muon(ns_coeffs=coeffs_3, ns_steps=5)
    with self.assertRaisesRegex(ValueError, "Not enough coeffs to perform 5 steps"):
      opt.init(params)


if __name__ == "__main__":
  absltest.main()
