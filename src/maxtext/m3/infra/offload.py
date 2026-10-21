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

"""Host offload of optimizer state, the last step of the composition order.

AdamW keeps two float32 moments per parameter -- more HBM than the bf16
parameters themselves. `HostOffloadedOptimizer` parks them in pinned host
memory between steps and moves them onto the device only for the duration of
`update`. XLA schedules the two transfers inside the jitted step, so from the
loop's point of view this is an ordinary `nnx.Optimizer`.

Moving memory kinds inside a traced function goes through
`jax.device_put(x, jax.memory.Space.*)`; outside tracing, through the array's
own sharding `with_memory_kind`. `to_memory` picks whichever applies.

Verified on a v6e-8 that the loss trajectory matches the un-offloaded
optimizer step for step (`OffloadTest` in `tests/unit/m3/infra_test.py`). Activation
offload is a remat policy (`minimal_offloaded`, `custom`), not this module;
parameter offload is not implemented.
"""

from typing import Any

from flax import nnx
import jax
import optax

_SPACES = {
    "pinned_host": jax.memory.Space.Host,
    "device": jax.memory.Space.Device,
}


def to_memory(tree: Any, kind: str) -> Any:
  """Returns `tree` with every array in memory `kind` (`"pinned_host"` or `"device"`).

  Works on `nnx.Variable` leaves (their metadata is kept) as well as bare
  arrays, traced or concrete.
  """

  def move(x):
    if isinstance(x, nnx.Variable):
      return x.replace(move(x[...]))
    if isinstance(x, jax.core.Tracer):
      return jax.device_put(x, _SPACES[kind])
    return jax.device_put(x, x.sharding.with_memory_kind(kind))

  return jax.tree.map(move, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


class HostOffloadedOptimizer(nnx.Optimizer):
  """An `nnx.Optimizer` whose state lives in pinned host memory between steps."""

  def __init__(self, model: nnx.Module, tx: optax.GradientTransformation, wrt=nnx.Param):
    super().__init__(model, tx, wrt=wrt)
    self.opt_state = to_memory(self.opt_state, "pinned_host")

  def update(self, model: nnx.Module, grads: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
    self.opt_state = to_memory(self.opt_state, "device")
    super().update(model, grads, **kwargs)
    self.opt_state = to_memory(self.opt_state, "pinned_host")


def make_optimizer(model: nnx.Module, tx: optax.GradientTransformation, config) -> nnx.Optimizer:
  """Builds the optimizer, host-offloaded when `config.optimizer_memory_host_offload` is set."""
  if config.optimizer_memory_host_offload:
    return HostOffloadedOptimizer(model, tx, wrt=nnx.Param)
  return nnx.Optimizer(model, tx, wrt=nnx.Param)
