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
import re

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


class MuonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("2d_tuple_axes", (2, 3), _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1)),
      ("3d_batch_axis", (2, 3, 4), _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=2)),
      ("3d_negative_axes_indices", (2, 3, 4), _muon.MuonDimensionNumbers(reduction_axis=-3, output_axis=-1)),
      ("4d_multiple_batch_axes", (2, 3, 4, 5), _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3)),
      ("4d_multiple_output_axes", (2, 3, 4, 5), _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=(0, 3))),
  )
  def test_reshape_inverse(self, input_shape, dim_nums):
    x = jnp.arange(math.prod(input_shape), dtype=jnp.float32).reshape(input_shape)
    reshape_fn, inverse_fn, _ = _muon.get_reshape_fns(x, dim_nums)
    reshaped_x = reshape_fn(x)
    reconstructed_x = inverse_fn(reshaped_x)
    self.assertEqual(reshaped_x.ndim, 3)
    self.assertLessEqual(reshaped_x.shape[1], reshaped_x.shape[2])
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
      return {"w1": _muon.ShardedMuonDimensionNumbers(), "w2": None}

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
    dim_nums = {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=0, output_axis=1)}
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
      dim_nums = {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=0, output_axis=1)}
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
      dim_nums = {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=(2,), output_axis=(0, 3))}
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
      dim_nums = {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=(0, 3), output_axis=(1,))}
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=(1,), output_axis=(2,))

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
    dim_num = _muon.ShardedMuonDimensionNumbers()
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
    dim_num = _muon.ShardedMuonDimensionNumbers()
    dim_nums = {"w1": dim_num, "w2": None}
    _, state = get_updates(params, dim_nums)
    state_structure = jax.tree.structure(get_muon_mu(state), is_leaf=lambda x: isinstance(x, optax.MaskedNode))
    self.assertEqual(state_structure, jax.tree.structure(params))
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["a"], optax.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["b"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["a"], optax.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["b"], optax.MaskedNode)

  def test_newton_schulz(self):
    """Test that Newton--Schulz orthogonalizes matrices accurately."""
    mat = jax.random.normal(jax.random.key(0), (4, 3), dtype=jnp.float32)
    ns_coeffs = jnp.array([2.0, -1.5, 0.5])

    if jax.default_backend() in ("tpu", "proxy"):
      atol, rtol = 1e-2, 1e-2
    else:
      atol, rtol = 1e-5, 1e-5

    precond_fn = _muon.get_precond_fn("frobenius", 1e-12)
    ns_step_fn = _muon.NS_ITERATORS["frobenius"]

    mat_orth = _muon.orthogonalize(
        mat,
        ns_coeffs,
        ns_steps=20,
        precond_fn=precond_fn,
        ns_step_fn=ns_step_fn,
        dim_nums=_muon.ShardedMuonDimensionNumbers(0, 1),
    )

    gram = mat_orth.T @ mat_orth
    msg = f"Newton–Schulz did not orthogonalize accurately.\nGram:\n{gram}"
    np.testing.assert_allclose(
        gram,
        jnp.eye(mat.shape[1]),
        atol=atol,
        rtol=rtol,
        err_msg=msg,
    )

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
    tol = 1e-2 if jax.default_backend() in ("tpu", "proxy") else 1e-3
    self.assertLess(ortho_error, tol, f"Orthogonality error too high: {ortho_error}")

  @parameterized.parameters(True, False)
  def test_use_all_to_all_option(self, use_all_to_all):
    """Ensures use_all_to_all option runs cleanly on multi-dimensional tensors."""
    key = jax.random.key(0)
    devices = jax.devices()
    if len(devices) >= 2:
      mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), axis_names=("x", "y"))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None, None, None))
      w = jax.device_put(jax.random.normal(key, (2, 3, 4, 5)), sharding)
      dim_nums = _muon.ShardedMuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
    else:
      w = jax.random.normal(key, (2, 3, 4, 5))
      dim_nums = _muon.ShardedMuonDimensionNumbers(reduction_axis=2, output_axis=3)

    params = {"w": w}
    opt = _muon.muon(
        learning_rate=0.1,
        muon_weight_dimension_numbers={"w": dim_nums},
        use_all_to_all=use_all_to_all,
    )
    state = opt.init(params)

    @jax.jit
    def step(p, s):
      return opt.update(p, s, p)

    updates, _ = step(params, state)
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

  def test_split_output_chunks_validation(self):
    """Validates split_output_chunks preconditions on rank, divisibility, and count."""
    w = jnp.ones((4, 8, 10))
    for bad_chunks in (0, -1, True, 2.0):
      bad_dn = _muon.ShardedMuonDimensionNumbers(
          reduction_axis=-2, output_axis=-1, split_output_chunks=bad_chunks  # type: ignore[arg-type]
      )
      with self.assertRaisesRegex(ValueError, "split_output_chunks must be an int >= 1"):
        _muon.orthogonalize(
            w,
            ns_coeffs=jnp.asarray((3.4445, -4.7750, 2.0315)),
            ns_steps=1,
            precond_fn=_muon.get_precond_fn("frobenius"),
            ns_step_fn=_muon.NS_ITERATORS["frobenius"],
            dim_nums=bad_dn,
        )
      with self.assertRaisesRegex(ValueError, "split_output_chunks must be an int >= 1"):
        tx = _muon.scale_by_muon(weight_dimension_numbers={"w": bad_dn})
        tx.update({"w": w}, tx.init({"w": w}), {"w": w})
      with self.assertRaisesRegex(ValueError, "split_output_chunks must be an int >= 1"):
        get_updates({"w": w}, {"w": bad_dn})
    with self.assertRaisesRegex(ValueError, "not divisible by split_output_chunks=3"):
      get_updates(
          {"w": w},
          {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=-2, output_axis=-1, split_output_chunks=3)},
      )
    with self.assertRaisesRegex(ValueError, "requires a single output_axis"):
      get_updates(
          {"w": jnp.ones((2, 4, 6))},
          {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=0, output_axis=(1, 2), split_output_chunks=2)},
      )

  @parameterized.product(
      shape=((4, 8, 12), (4, 2, 8, 12)),
      consistent_rms=(None, 0.2),
      adaptive=(False, True),
      use_all_to_all=(True, False),
  )
  def test_split_output_chunks_matches_separate_projections(self, shape, consistent_rms, adaptive, use_all_to_all):
    """split_output_chunks=2 matches orthogonalizing and scaling [w0 | w1] as two independent leaves."""
    k1, k2, k3 = jax.random.split(jax.random.key(17), 3)
    w = jax.random.normal(k1, shape, dtype=jnp.float32)
    g_step0 = jax.random.normal(k2, shape, dtype=jnp.float32)
    g_step1 = jax.random.normal(k3, shape, dtype=jnp.float32)
    half = shape[-1] // 2

    fused_dim_nums = {
        "w": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), split_output_chunks=2)
    }
    unfused_dim_nums = {
        "w0": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,)),
        "w1": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,)),
    }
    naive_dim_nums = {
        "w": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), split_output_chunks=1)
    }

    common_kwargs = {
        "learning_rate": 1e-2,
        "consistent_rms": consistent_rms,
        "adaptive": adaptive,
        "use_all_to_all": use_all_to_all,
        "weight_decay": 0.05,
    }
    opt_fused = _muon.muon(muon_weight_dimension_numbers=fused_dim_nums, **common_kwargs)
    opt_unfused = _muon.muon(muon_weight_dimension_numbers=unfused_dim_nums, **common_kwargs)
    opt_naive = _muon.muon(muon_weight_dimension_numbers=naive_dim_nums, **common_kwargs)

    p_fused = {"w": w}
    p_unfused = {"w0": w[..., :half], "w1": w[..., half:]}
    s_fused = opt_fused.init(p_fused)
    s_unfused = opt_unfused.init(p_unfused)
    s_naive = opt_naive.init(p_fused)

    u_naive, _ = opt_naive.update({"w": g_step0}, s_naive, p_fused)

    for step_idx, g in enumerate((g_step0, g_step1)):
      u_fused, s_fused = opt_fused.update({"w": g}, s_fused, p_fused)
      u_unfused, s_unfused = opt_unfused.update({"w0": g[..., :half], "w1": g[..., half:]}, s_unfused, p_unfused)
      expected = jnp.concatenate([u_unfused["w0"], u_unfused["w1"]], axis=-1)
      np.testing.assert_allclose(u_fused["w"], expected, rtol=1e-5, atol=1e-5)
      if step_idx == 0:
        self.assertGreater(float(jnp.max(jnp.abs(u_fused["w"] - u_naive["w"]))), 1e-3)
      p_fused = optax.apply_updates(p_fused, u_fused)
      p_unfused = optax.apply_updates(p_unfused, u_unfused)

  @parameterized.product(
      explicit_axes=(False, True),
      tp_size=(1, 2),
      use_all_to_all=(True, False),
  )
  def test_split_output_chunks_sharded_meshes(self, explicit_axes, tp_size, use_all_to_all):
    """Verifies split_output_chunks=2 across size-1 and size-2 output mesh axes (Auto and Explicit)."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("Requires at least 4 devices for sharded Muon mesh tests.")

    axis_type = jax.sharding.AxisType.Explicit if explicit_axes else jax.sharding.AxisType.Auto
    if tp_size == 1:
      mesh = jax.sharding.Mesh(
          np.array(devices[:4]).reshape((2, 2, 1)),
          axis_names=("expert", "fsdp", "tensor"),
          axis_types=(axis_type, axis_type, axis_type),
      )
      pspec = jax.sharding.PartitionSpec("expert", "fsdp", "tensor")
    else:
      mesh = jax.sharding.Mesh(
          np.array(devices[:4]).reshape((2, 2)),
          axis_names=("expert", "tensor"),
          axis_types=(axis_type, axis_type),
      )
      pspec = jax.sharding.PartitionSpec("expert", None, "tensor")

    sharding = jax.sharding.NamedSharding(mesh, pspec)
    k1, k2 = jax.random.split(jax.random.key(29))
    w_host = jax.random.normal(k1, (4, 8, 12), dtype=jnp.float32)
    g_host = jax.random.normal(k2, (4, 8, 12), dtype=jnp.float32)

    w_fused = jax.device_put(w_host, sharding)
    g_fused = jax.device_put(g_host, sharding)
    w0 = jax.device_put(w_host[..., :6], sharding)
    w1 = jax.device_put(w_host[..., 6:], sharding)
    g0 = jax.device_put(g_host[..., :6], sharding)
    g1 = jax.device_put(g_host[..., 6:], sharding)

    for adaptive in (False, True):
      opt_fused = _muon.muon(
          learning_rate=1e-2,
          consistent_rms=0.2,
          adaptive=adaptive,
          use_all_to_all=use_all_to_all,
          muon_weight_dimension_numbers={
              "w": _muon.ShardedMuonDimensionNumbers(
                  reduction_axis=(-2,),
                  output_axis=(-1,),
                  sharding=sharding,
                  split_output_chunks=2,
              )
          },
      )
      opt_unfused = _muon.muon(
          learning_rate=1e-2,
          consistent_rms=0.2,
          adaptive=adaptive,
          use_all_to_all=use_all_to_all,
          muon_weight_dimension_numbers={
              "w0": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
              "w1": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
          },
      )

      with jax.set_mesh(mesh):
        s_fused = opt_fused.init({"w": w_fused})
        s_unfused = opt_unfused.init({"w0": w0, "w1": w1})
        u_fused, _ = jax.jit(lambda g, s, p, o=opt_fused: o.update(g, s, p))({"w": g_fused}, s_fused, {"w": w_fused})
        u_unfused, _ = jax.jit(lambda g, s, p, o=opt_unfused: o.update(g, s, p))(
            {"w0": g0, "w1": g1}, s_unfused, {"w0": w0, "w1": w1}
        )

      self.assertEqual(u_fused["w"].sharding, sharding)
      u_fused_host = np.asarray(jax.device_get(u_fused["w"]))
      u0_host = np.asarray(jax.device_get(u_unfused["w0"]))
      u1_host = np.asarray(jax.device_get(u_unfused["w1"]))
      np.testing.assert_allclose(u_fused_host, np.concatenate([u0_host, u1_host], axis=-1), rtol=1e-5, atol=1e-5)

  @parameterized.product(
      explicit_axes=[False, True],
      num_experts=[1, 2, 4],
      matrix_shape=[(16, 8), (8, 16)],
  )
  def test_split_output_chunks_unscanned_fsdp_gt_chunks_skips_padding(self, explicit_axes, num_experts, matrix_shape):
    """Unscanned MoE `wi` with `split_output_chunks=2` and `fsdp=4` avoids padding and replicates once."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("Requires at least 4 devices for sharded Muon mesh tests.")

    k_dim, n_dim = matrix_shape
    axis_type = jax.sharding.AxisType.Explicit if explicit_axes else jax.sharding.AxisType.Auto
    mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape((1, 4, 1)),
        axis_names=("expert", "fsdp", "tensor"),
        axis_types=(axis_type, axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("expert", "fsdp", "tensor"))
    k1, k2 = jax.random.split(jax.random.key(31))
    w_host = jax.random.normal(k1, (num_experts, k_dim, 2 * n_dim), dtype=jnp.float32)
    g_host = jax.random.normal(k2, (num_experts, k_dim, 2 * n_dim), dtype=jnp.float32)

    w_fused = jax.device_put(w_host, sharding)
    g_fused = jax.device_put(g_host, sharding)
    w0 = jax.device_put(w_host[..., :n_dim], sharding)
    w1 = jax.device_put(w_host[..., n_dim:], sharding)
    g0 = jax.device_put(g_host[..., :n_dim], sharding)
    g1 = jax.device_put(g_host[..., n_dim:], sharding)

    opt_fused = _muon.muon(
        learning_rate=1e-2,
        consistent_rms=0.2,
        use_all_to_all=True,
        muon_weight_dimension_numbers={
            "w": _muon.ShardedMuonDimensionNumbers(
                reduction_axis=(-2,),
                output_axis=(-1,),
                sharding=sharding,
                split_output_chunks=2,
            )
        },
    )
    opt_unfused = _muon.muon(
        learning_rate=1e-2,
        consistent_rms=0.2,
        use_all_to_all=True,
        muon_weight_dimension_numbers={
            "w0": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
            "w1": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
        },
    )

    with jax.set_mesh(mesh):
      s_fused = opt_fused.init({"w": w_fused})
      s_unfused = opt_unfused.init({"w0": w0, "w1": w1})
      jitted_fused = jax.jit(opt_fused.update)
      lowered_fused = jitted_fused.lower({"w": g_fused}, s_fused, {"w": w_fused})
      hlo_text = lowered_fused.as_text()
      self.assertNotIn("concatenate", hlo_text)
      self.assertNotIn("stablehlo.pad", hlo_text)
      if num_experts == 1:
        compiled_hlo = lowered_fused.compile().as_text()
        all_gather_count = len(re.findall(r"=\s*[^\n]*\ball-gather\(", compiled_hlo))
        all_reduce_count = len(re.findall(r"=\s*[^\n]*\ball-reduce\(", compiled_hlo))
        self.assertEqual(all_gather_count, 1, msg=compiled_hlo)
        self.assertEqual(all_reduce_count, 0, msg=compiled_hlo)
      u_fused, _ = jitted_fused({"w": g_fused}, s_fused, {"w": w_fused})
      u_unfused, _ = jax.jit(opt_unfused.update)({"w0": g0, "w1": g1}, s_unfused, {"w0": w0, "w1": w1})

    self.assertEqual(u_fused["w"].sharding, sharding)
    u_fused_host = np.asarray(jax.device_get(u_fused["w"]))
    u0_host = np.asarray(jax.device_get(u_unfused["w0"]))
    u1_host = np.asarray(jax.device_get(u_unfused["w1"]))
    np.testing.assert_allclose(u_fused_host, np.concatenate([u0_host, u1_host], axis=-1), rtol=1e-5, atol=1e-5)

  @parameterized.product(explicit_axes=[False, True], matrix_shape=[(16, 8), (8, 16)])
  def test_split_output_chunks_unscanned_fsdp_gt_chunks_spectral(self, explicit_axes, matrix_shape):
    """Single-local-expert replication supports spectral preconditioning (SVD) under Auto and Explicit sharding."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("Requires at least 4 devices for sharded Muon mesh tests.")

    k_dim, n_dim = matrix_shape
    axis_type = jax.sharding.AxisType.Explicit if explicit_axes else jax.sharding.AxisType.Auto
    mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape((1, 4, 1)),
        axis_names=("expert", "fsdp", "tensor"),
        axis_types=(axis_type, axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("expert", "fsdp", "tensor"))
    k1, k2 = jax.random.split(jax.random.key(37))
    w_host = jax.random.normal(k1, (1, k_dim, 2 * n_dim), dtype=jnp.float32)
    g_host = jax.random.normal(k2, (1, k_dim, 2 * n_dim), dtype=jnp.float32)

    w_fused = jax.device_put(w_host, sharding)
    g_fused = jax.device_put(g_host, sharding)
    w0 = jax.device_put(w_host[..., :n_dim], sharding)
    w1 = jax.device_put(w_host[..., n_dim:], sharding)
    g0 = jax.device_put(g_host[..., :n_dim], sharding)
    g1 = jax.device_put(g_host[..., n_dim:], sharding)

    opt_fused = _muon.muon(
        learning_rate=1e-2,
        consistent_rms=0.2,
        preconditioning="spectral",
        use_all_to_all=True,
        muon_weight_dimension_numbers={
            "w": _muon.ShardedMuonDimensionNumbers(
                reduction_axis=(-2,),
                output_axis=(-1,),
                sharding=sharding,
                split_output_chunks=2,
            )
        },
    )
    opt_unfused = _muon.muon(
        learning_rate=1e-2,
        consistent_rms=0.2,
        preconditioning="spectral",
        use_all_to_all=True,
        muon_weight_dimension_numbers={
            "w0": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
            "w1": _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), sharding=sharding),
        },
    )

    with jax.set_mesh(mesh):
      s_fused = opt_fused.init({"w": w_fused})
      s_unfused = opt_unfused.init({"w0": w0, "w1": w1})
      u_fused, _ = jax.jit(opt_fused.update)({"w": g_fused}, s_fused, {"w": w_fused})
      u_unfused, _ = jax.jit(opt_unfused.update)({"w0": g0, "w1": g1}, s_unfused, {"w0": w0, "w1": w1})

    self.assertEqual(u_fused["w"].sharding, sharding)
    u_fused_host = np.asarray(jax.device_get(u_fused["w"]))
    u0_host = np.asarray(jax.device_get(u_unfused["w0"]))
    u1_host = np.asarray(jax.device_get(u_unfused["w1"]))
    np.testing.assert_allclose(u_fused_host, np.concatenate([u0_host, u1_host], axis=-1), rtol=1e-5, atol=1e-5)

  def test_adaptive_dual_norm_preserves_masked_nodes(self):
    """adaptive=True skips optax.MaskedNode leaves when wrapped in optax.masked."""
    params = {"w": jnp.ones((2, 8, 12), dtype=jnp.float32), "frozen": jnp.ones((4,), dtype=jnp.float32)}
    grads = {
        "w": jax.random.normal(jax.random.key(41), (2, 8, 12), dtype=jnp.float32),
        "frozen": jnp.full((4,), 3.0, dtype=jnp.float32),
    }
    dn = _muon.ShardedMuonDimensionNumbers(reduction_axis=(-2,), output_axis=(-1,), split_output_chunks=2)
    tx = optax.masked(
        _muon.scale_by_muon(adaptive=True, weight_dimension_numbers={"w": dn, "frozen": None}),
        mask={"w": True, "frozen": False},
    )
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    np.testing.assert_array_equal(updates["frozen"], grads["frozen"])
    self.assertEqual(updates["w"].shape, grads["w"].shape)


if __name__ == "__main__":
  absltest.main()
