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

"""Tests for aqt_matmul."""

from typing import Iterable

from absl.testing import absltest
from aqt.common import aqt_config
from aqt.jax import aqt_dot_general
from aqt.jax import aqt_ops
from aqt.jax import aqt_tensor
from aqt.test import aqt_matmul_test_base
from flax import linen as nn
import jax
import jax.numpy as jnp

# pylint: disable=g-long-lambda

dot_general = aqt_dot_general.injectable_dot_general


class Matmul(nn.Module):
  config: aqt_config.AqtMatmulConfig
  lhs_shape: Iterable[int]
  rhs_shape: Iterable[int]

  def setup(self):
    self.lhs_quantizer = aqt_tensor.TensorQuantizer(
        list(self.lhs_shape), self.config.lhs)
    self.rhs_quantizer = aqt_tensor.TensorQuantizer(
        list(self.rhs_shape), self.config.rhs)

  def __call__(self, lhs, rhs, train=True):
    return aqt_ops.aqt_matmul(
        lhs,
        rhs,
        dot_general=dot_general(self.lhs_quantizer, self.rhs_quantizer, train))

  def update_lhs(self, x, weight, event_count):
    self.lhs_quantizer.update(x, weight, event_count)

  def update_rhs(self, x, weight, event_count):
    self.rhs_quantizer.update(x, weight, event_count)


class AqtMatmulTest(aqt_matmul_test_base.MatmulTest):

  _matmul_state = {}

  def constant(self, x):
    return jnp.array(x, dtype=jnp.float32)

  def matmul(self, config, lhs_shape, rhs_shape, name="aqt"):
    mm = Matmul(config, lhs_shape, rhs_shape, name=name)
    state = mm.init(
        jax.random.PRNGKey(0), jnp.zeros(lhs_shape), jnp.zeros(rhs_shape))
    self._matmul_state[name] = state

    return mm

  def matmul_apply(self, mm, lhs, rhs, train=True, keep_stats=False):
    event_count = 0
    lhs_sample = jnp.zeros_like(lhs) if keep_stats else lhs
    lhs_weight = jnp.ones_like(lhs) if keep_stats else None
    rhs_sample = jnp.zeros_like(rhs) if keep_stats else rhs
    rhs_weight = jnp.ones_like(rhs) if keep_stats else None

    _, state = mm.apply(
        self._matmul_state[mm.name],
        lhs_sample,
        lhs_weight,
        event_count,
        method=mm.update_lhs,
        mutable=True)
    _, state = mm.apply(
        state,
        rhs_sample,
        rhs_weight,
        event_count,
        method=mm.update_rhs,
        mutable=True)
    self._matmul_state[mm.name] = state

    return mm.apply(self._matmul_state[mm.name], lhs, rhs, train)

  def matmul_unquantized(self, lhs, rhs):
    lhs = self.constant(lhs)
    rhs = self.constant(rhs)

    return jnp.matmul(lhs, rhs)

  def gradients(self, fwd_func, x, w, reduce_sum=False):
    if reduce_sum:
      fwd_func = lambda x, w: jnp.sum(fwd_func(x, w)**2)
    y, vjp_fn = jax.vjp(fwd_func, x, w)
    return vjp_fn(jnp.ones(y.shape))[0]


if __name__ == "__main__":
  absltest.main()
