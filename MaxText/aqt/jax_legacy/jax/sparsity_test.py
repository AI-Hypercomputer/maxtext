# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for sparsity."""

import dataclasses
import typing

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax_legacy.jax import sparsity
from aqt.jax_legacy.jax.flax import struct as flax_struct
from aqt.jax_legacy.jax.sparsity import SparseHParams
from aqt.jax_legacy.jax.sparsity import Sparsity
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import numpy as np


dataclass = (
    flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
)


class SparsityTest(parameterized.TestCase):

  def init_model(
      self,
      update_mask,
      apply_mask,
      unstruct_sparsity,
      structure_decay=False,
      num_update_sparsity=0,
      mask_decay_weight=0.0,
      prune_rate=(2, 4),
      sparse_ste=False,
  ):
    rng = random.PRNGKey(0)
    self.inputs = jnp.array([[3, 4, 6], [1, 2, 1]])
    if unstruct_sparsity:
      sparsity_hparams = SparseHParams(
          type='UNSTRUCTURED',
          prune_rate=0.2,
          structure_decay=structure_decay,
          mask_decay_weight=mask_decay_weight,
          sparse_ste=sparse_ste,
      )
    else:
      sparsity_hparams = SparseHParams(
          type='STRUCTURED_NM',
          prune_rate=prune_rate,
          structure_decay=structure_decay,
          mask_decay_weight=mask_decay_weight,
          sparse_ste=sparse_ste,
      )
    sparsity_module = Sparsity(sparsity_hparams=sparsity_hparams)
    init_mask = sparsity_module.init(
        rng,
        self.inputs,
        update_mask=update_mask,
        apply_mask=apply_mask,
        num_update_sparsity=num_update_sparsity,
    )
    return sparsity_module, init_mask

  @parameterized.named_parameters(('unstruct', True), ('struct', False))
  def test_init(self, unstruct_sparsity):
    _, init_state = self.init_model(False, False, unstruct_sparsity)
    init_state_mask = init_state['sparsity']['mask']
    np.testing.assert_array_equal(
        init_state_mask, [[True, True, True], [True, True, True]]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='initial_structure_decay',
          num_update_sparsity=0,
          out=[[3, 4, 6, 8], [1, 2, 1, 4]],
          mask=[[True, True, True, True], [True, True, True, True]],
      ),
      dict(
          testcase_name='first_iteration_structure_decay',
          num_update_sparsity=1,
          out=[[0, 4, 6, 8], [0, 2, 1, 4]],
          mask=[[False, True, True, True], [False, True, True, True]],
      ),
      dict(
          testcase_name='second_iteration_structure_decay',
          num_update_sparsity=2,
          out=[[0, 0, 0, 8], [0, 0, 0, 4]],
          mask=[[False, False, False, True], [False, False, False, True]],
      ),
      dict(
          testcase_name='third_iteration_structure_decay',
          num_update_sparsity=3,
          out=[[0, 0, 0, 8], [0, 0, 0, 4]],
          mask=[[False, False, False, True], [False, False, False, True]],
      ),
  )
  def test_structure_decay(self, num_update_sparsity, out, mask):
    model, init_state = self.init_model(
        update_mask=False,
        apply_mask=False,
        unstruct_sparsity=False,
        structure_decay=True,
        prune_rate=(4, 4))
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    model_out, state_0 = model.apply(
        init_state,
        inputs,
        update_mask=True,
        apply_mask=True,
        num_update_sparsity=num_update_sparsity,
        mutable='sparsity')
    np.testing.assert_array_equal(model_out, out)
    np.testing.assert_array_equal(state_0['sparsity']['mask'], mask)

  @parameterized.named_parameters(
      dict(
          testcase_name='initial_mask_decay',
          num_update_sparsity=0,
          out=[[3, 4, 6, 8], [1, 2, 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
      dict(
          testcase_name='first_iteration_mask_decay',
          num_update_sparsity=1,
          out=[[0.9 * 3, 0.9 * 4, 6, 8], [0.9 * 1, 2, 0.9 * 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
      dict(
          testcase_name='second_iteration_mask_decay',
          num_update_sparsity=2,
          out=[[0.8 * 3, 0.8 * 4, 6, 8], [0.8 * 1, 2, 0.8 * 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
      dict(
          testcase_name='third_iteration_mask_decay',
          num_update_sparsity=3,
          out=[[0.7 * 3, 0.7 * 4, 6, 8], [0.7 * 1, 2, 0.7 * 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
      dict(
          testcase_name='ninth_iteration_mask_decay',
          num_update_sparsity=9,
          out=[[0.1 * 3, 0.1 * 4, 6, 8], [0.1 * 1, 2, 0.1 * 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
      dict(
          testcase_name='tenth_iteration_mask_decay',
          num_update_sparsity=10,
          out=[[0.0 * 3, 0.0 * 4, 6, 8], [0.0 * 1, 2, 0.0 * 1, 4]],
          mask=[[False, False, True, True], [False, True, False, True]]),
  )
  def test_mask_decay(self, num_update_sparsity, out, mask):
    model, init_state = self.init_model(
        update_mask=False,
        apply_mask=False,
        unstruct_sparsity=False,
        structure_decay=False,
        mask_decay_weight=0.1,
        prune_rate=(2, 4))
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    model_out, state_0 = model.apply(
        init_state,
        inputs,
        update_mask=True,
        apply_mask=True,
        num_update_sparsity=num_update_sparsity,
        mutable='sparsity')
    np.testing.assert_allclose(model_out, out)
    np.testing.assert_array_equal(state_0['sparsity']['mask'], mask)

  # TODO(ayazdan): Add a more general test for other forms of sparsity.
  def test_sorting_network_sparse_2_4(self):
    key = random.PRNGKey(0)
    inputs = random.normal(key, (2, 5, 16))
    print(inputs)
    # n = 4
    # k = 2
    m = 4
    n = 2
    alpha = 1.0
    # JAX Top-K Implementation.
    xs = jnp.stack(
        jnp.split(inputs, int(jnp.shape(inputs)[-1] / m), axis=-1), axis=0)
    topk = jax.lax.top_k(xs, k=n)
    mask = jax.numpy.sum(
        jax.nn.one_hot(topk[1],
                       jnp.shape(xs)[-1]), axis=len(jnp.shape(inputs)))
    topk_filter = jnp.where(
        jnp.equal(mask, 0.), jnp.full(list(jnp.shape(mask)), alpha - 1), mask)
    filtered = jnp.multiply(xs, topk_filter)
    ys = jnp.concatenate(
        jnp.moveaxis(filtered, 0, 0), axis=len(jnp.shape(inputs)) - 1)
    print(ys)
    ys_hat = sparsity.prune_2_4(inputs)
    print(ys_hat)
    np.testing.assert_array_equal(ys, ys_hat)

  @parameterized.named_parameters(
      dict(
          testcase_name='row_wise_pruning',
          order='R',
          exp_output=[[0, 0, 3, 4, 0, 0, 7, 8],
                      [0, 0, 11, 12, 0, 0, 15, 16],
                      [0, 0, 19, 20, 0, 0, 23, 24],
                      [0, 0, 27, 28, 0, 0, 31, 32]]),
      dict(
          testcase_name='column_wise_pruning',
          order='C',
          exp_output=[[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [17, 18, 19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30, 31, 32]]))
  def test_column_row_pruning(self, order, exp_output):
    inputs = jnp.reshape(jnp.arange(1, 33), (4, 8))
    output = sparsity.prune_inputs_n_m(inputs, n=2, m=4, order=order)
    np.testing.assert_array_equal(output, exp_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='column_wise_pruning',
          order='C',
          exp_output=[[[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]],
                      [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [25, 26, 27, 28],
                       [29, 30, 31, 32]],
                      [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [41, 42, 43, 44],
                       [45, 46, 47, 48]],
                      [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [57, 58, 59, 60],
                       [61, 62, 63, 64]]]))
  def test_3d_column_pruning(self, order, exp_output):
    inputs = jnp.reshape(jnp.arange(1, 65), (4, 4, 4))
    output = sparsity.prune_inputs_n_m(inputs, n=2, m=4, order=order)
    np.testing.assert_array_equal(output, exp_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='offset_zero',
          offset=0,
          exp_output=[[0, 0, 3, 14, 0, 16, 0, 18, 0, 20],
                      [0, 12, 13, 0, 15, 0, 17, 0, 19, 0]]),
      dict(
          testcase_name='offset_positive',
          offset=5,
          exp_output=[[0, 0, 0, 14, 0, 16, 0, 18, 0, 20],
                      [11, 12, 13, 0, 15, 0, 17, 0, 19, 0]]))
  def test_n_m_row_wise_sparsity_with_offset(self, offset, exp_output):
    inputs = jnp.array([[1, 2, 3, 14, 5, 16, 7, 18, 9, 20],
                        [11, 12, 13, 4, 15, 6, 17, 8, 19, 10]])

    output = sparsity.prune_inputs_n_m(inputs, n=2, m=4, offset=offset)
    np.testing.assert_array_equal(output, exp_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='update_mask_apply_mask',
          update_mask=True,
          apply_mask=True),
      dict(
          testcase_name='no_update_mask_apply_mask',
          update_mask=False,
          apply_mask=True),
      dict(
          testcase_name='update_mask_no_apply_mask',
          update_mask=True,
          apply_mask=False),
      dict(
          testcase_name='no_update_mask_no_apply_mask',
          update_mask=False,
          apply_mask=False),
  )
  def test_sr_ste_fwd_pass(self, update_mask, apply_mask):
    rng = random.PRNGKey(0)
    sparsity_hparams = SparseHParams(
        type='STRUCTURED_NM',
        prune_rate=(2, 4),
        sparse_ste=True,
        structure_decay=False,
        mask_decay_weight=0.0,
    )
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    sparsity_module = Sparsity(sparsity_hparams=sparsity_hparams)
    init_state = sparsity_module.init(
        rng,
        inputs,
        update_mask=False,
        apply_mask=True,
        num_update_sparsity=0.0)
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    out, state_0 = sparsity_module.apply(
        init_state,
        inputs,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity')
    state_0_mask = state_0['sparsity']['mask']
    if not apply_mask or not update_mask:
      np.testing.assert_array_equal(out, inputs)
    else:
      np.testing.assert_array_equal(out, [[0, 0, 6, 8], [0, 2, 0, 4]])
    if update_mask:
      np.testing.assert_array_equal(
          state_0_mask,
          [[False, False, True, True], [False, True, False, True]])
    else:
      np.testing.assert_array_equal(
          state_0_mask, [[True, True, True, True], [True, True, True, True]])

    inputs2 = jnp.array([[2., 3., 1., 1.], [2., -6., 1., 5.]])
    out, state = sparsity_module.apply(
        state_0,
        inputs2,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity')
    state_mask = state['sparsity']['mask']
    if not apply_mask or not update_mask:
      np.testing.assert_array_equal(out, inputs2)
    else:
      np.testing.assert_array_equal(out, [[2., 3., 0., 0.], [0., -6., 0., 5.]])
    if update_mask:
      np.testing.assert_array_equal(
          state_mask, [[True, True, False, False], [False, True, False, True]])
    else:
      np.testing.assert_array_equal(
          state_mask, [[True, True, True, True], [True, True, True, True]])

  def test_sr_ste_bwd_pass(self):

    class SingleLayer(nn.Module):

      @dataclass
      class HParams:
        sparsity: SparseHParams

      hparams: HParams

      @nn.compact
      def __call__(
          self,
          inputs: jnp.ndarray,
          update_mask: bool,
          apply_mask: bool
      ) -> jnp.ndarray:
        kernel = self.param('kernel', nn.initializers.ones, (1, 4))
        kernel = Sparsity(
            sparsity_hparams=self.hparams.sparsity, name='weight_sparsity')(
                kernel,
                update_mask=update_mask,
                apply_mask=apply_mask,
                num_update_sparsity=0)
        return jnp.multiply(inputs, kernel)

    rng = random.PRNGKey(0)
    layer_kwargs = {}
    layer_kwargs['hparams'] = SingleLayer.HParams(
        sparsity=SparseHParams(
            type='STRUCTURED_NM',
            prune_rate=(2, 4),
            sparse_ste=True,
            structure_decay=False,
            mask_decay_weight=0.0,
        ))

    inputs = jnp.array([[2., 3., -5., 6.]])

    def loss_fn(params, state):
      del state
      model = SingleLayer(**layer_kwargs)
      y, updated_state = model.apply({
          'params': params,
      },
                                     inputs,
                                     update_mask=True,
                                     apply_mask=True,
                                     mutable=True)
      total_loss = jnp.sum(y)
      return total_loss, updated_state

    @jax.jit
    def update_params(params, grads):
      params = jax.tree_util.tree_map(lambda p, g: p - g, params, grads)
      return params

    module = SingleLayer(**layer_kwargs)
    variables = module.init(
        rng, jnp.zeros(inputs.shape), update_mask=False, apply_mask=False)
    state, params = flax.core.pop(variables, 'params')
    del variables
    for _ in range(10):
      # At each iteration, the pruned weights are multiplied with
      # ste_weight = 0.0002 and added to corresponding gradients.
      # In this simple example, gradients are simply the inputs to the network.
      (_, state), grads = jax.value_and_grad(
          loss_fn, has_aux=True, allow_int=True)(params, state)
      np.testing.assert_allclose(
          grads['kernel'], inputs + 0.0002 * jnp.multiply(
              ~state['sparsity']['weight_sparsity']['mask'], params['kernel']))
      params = update_params(params, grads)


# TODO(shivaniagrawal): Add tests for struct sparsity as well.

  @parameterized.named_parameters(
      dict(
          testcase_name='update_mask_apply_mask',
          update_mask=True,
          apply_mask=True,
      ),
      dict(
          testcase_name='no_update_mask_apply_mask',
          update_mask=False,
          apply_mask=True),
      dict(
          testcase_name='update_mask_no_apply_mask',
          update_mask=True,
          apply_mask=False),
      dict(
          testcase_name='no_update_mask_no_apply_mask',
          update_mask=False,
          apply_mask=False),
  )
  def test_mask(self, update_mask, apply_mask):
    model, init_state = self.init_model(
        update_mask, False, unstruct_sparsity=True)
    out, state_0 = model.apply(
        init_state,
        self.inputs,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity')

    state_0_mask = state_0['sparsity']['mask']
    if not apply_mask or not update_mask:
      np.testing.assert_array_equal(out, self.inputs)
    else:
      np.testing.assert_array_equal(out, [[3, 4, 6], [0, 2, 1]])
    if update_mask:
      np.testing.assert_array_equal(state_0_mask,
                                    [[True, True, True], [False, True, True]])
    else:
      np.testing.assert_array_equal(state_0_mask,
                                    [[True, True, True], [True, True, True]])

    inputs2 = jnp.array([[2, 3, 1], [5, 4, 2]])
    out, state = model.apply(
        state_0,
        inputs2,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity')
    state_mask = state['sparsity']['mask']
    if not apply_mask or not update_mask:
      np.testing.assert_array_equal(out, inputs2)
    else:
      np.testing.assert_array_equal(out, [[2, 3, 0], [5, 4, 2]])
    if update_mask:
      np.testing.assert_array_equal(state_mask,
                                    [[True, True, False], [True, True, True]])
    else:
      np.testing.assert_array_equal(state_mask,
                                    [[True, True, True], [True, True, True]])


class PruningParamsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(sparse_type='STRUCTURED_NM', prune_rate=0.1),
      dict(sparse_type='UNSTRUCTURED', prune_rate=(4, 1)))
  def test_invalid_params(self, sparse_type, prune_rate):
    with self.assertRaisesRegex(
        AssertionError, 'prune rate should be either None for no pruning'):
      sparsity.SparseHParams(type=sparse_type, prune_rate=prune_rate)

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          mask_decay_weight=-0.1),
      dict(sparse_type='UNSTRUCTURED', prune_rate=0.2, mask_decay_weight=-0.1))
  def test_invalid_mask_decay_weight(self, sparse_type, prune_rate,
                                     mask_decay_weight):
    with self.assertRaisesRegex(AssertionError,
                                '.* `mask_decay_weight` must be positive.'):
      sparsity.SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          mask_decay_weight=mask_decay_weight)

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          sparse_ste=True,
          mask_decay_weight=0.1),
      dict(
          sparse_type='UNSTRUCTURED',
          prune_rate=0.2,
          sparse_ste=True,
          mask_decay_weight=0.1))
  def test_invalid_sparse_ste_with_non_zero_mask_decay_weight(
      self, sparse_type, prune_rate, sparse_ste, mask_decay_weight):
    with self.assertRaisesRegex(ValueError,
                                'SR-STE only works with non-decaying mask.'):
      sparsity.SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          mask_decay_weight=mask_decay_weight)

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          sparse_ste=True,
          structure_decay=True),
      dict(
          sparse_type='UNSTRUCTURED',
          prune_rate=0.2,
          sparse_ste=True,
          structure_decay=True))
  def test_invalid_sparse_ste_with_structure_decay(self, sparse_type,
                                                   prune_rate, sparse_ste,
                                                   structure_decay):
    with self.assertRaisesRegex(
        ValueError,
        'SR-STE only works with non-decaying sparse structure.'):
      sparsity.SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          structure_decay=structure_decay)

  @parameterized.parameters(
      dict(sparse_type='UNSTRUCTURED', prune_rate=0.2, sparse_ste=True))
  def test_invalid_sparse_ste_with_unstructured_sparsity(
      self, sparse_type, prune_rate, sparse_ste):
    with self.assertRaisesRegex(ValueError,
                                'SR-STE only works with structured sparsity.'):
      sparsity.SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          sparse_ste=sparse_ste)

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          error_msg='must be lower than prune_rate'),
      dict(
          sparse_type='UNSTRUCTURED',
          prune_rate=2.5,
          error_msg='sparsity ratio can not be > 1, provided prune_rate'))
  def test_invalid_prune_rate(self, sparse_type, prune_rate, error_msg):
    sparsity_hparams = sparsity.SparseHParams(
        type=sparse_type, prune_rate=prune_rate)

    inputs = jnp.arange(12)
    with self.assertRaisesRegex(AssertionError, error_msg):
      sparsity.get_sparsity_mask(inputs, sparsity_hparams)


class PruningFunctionalityTest(parameterized.TestCase):

  def test_pruning_mask(self):
    # Total number of parameters = 20
    inputs = jnp.array(np.random.rand(10, 2))
    mask = sparsity.get_pruning_unstruct_mask(inputs, prune_rate=0.1)
    self.assertEqual(jnp.sum(mask), 18)

  def test_smallest_largest_magnitude_mask(self):
    # Total number of parameters = 20
    inputs = jnp.array(np.arange(20))
    mask = sparsity.get_pruning_unstruct_mask(
        inputs, smallest=True, prune_rate=0.1)
    self.assertFalse(mask[0])
    self.assertFalse(mask[1])

    mask = sparsity.get_pruning_unstruct_mask(
        inputs, smallest=False, prune_rate=0.1)
    self.assertFalse(mask[-1])
    self.assertFalse(mask[-2], 0)

  def test_prune_inputs_unstruct(self):
    # Total number of parameters = 20
    inputs = jnp.array(np.random.rand(10, 2))
    prune_input = sparsity.prune_inputs_unstruct(inputs)
    self.assertEqual(jnp.sum(prune_input == 0), 2)

  def test_smallest_largest_magnitude_prune_unstruct(self):
    # Total number of parameters = 20
    inputs = jnp.array(np.arange(20))
    prune_input = sparsity.prune_inputs_unstruct(inputs, prune_rate=0.1)
    self.assertEqual(prune_input[0], 0)
    self.assertEqual(prune_input[1], 0)

    prune_input = sparsity.prune_inputs_unstruct(
        inputs, prune_rate=0.1, smallest=False)
    self.assertEqual(prune_input[-1], 0)
    self.assertEqual(prune_input[-2], 0)

  def test_prune_inputs_n_m(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    out = sparsity.prune_inputs_n_m(inputs, n=1, m=4)
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])
    # Only 20 non-zero elements must exist after pruning.
    self.assertEqual(out[out != 0].shape[0], 20)
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(out != 0, axis=2).flatten()))

  def test_order_not_valid(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    with self.assertRaises(ValueError):
      _ = sparsity.prune_inputs_n_m(inputs, n=1, m=4, order='X')

  def test_n_m_pruning_mask(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    mask = sparsity.get_pruning_n_m_mask(inputs, n=1, m=4)
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(mask == 1, axis=2).flatten()))


if __name__ == '__main__':
  absltest.main()
