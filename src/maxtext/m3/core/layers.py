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

"""Container for a model's repeated layers.

Models keep their repeated blocks in a `Layers` and run them with
`carry = self.layers(carry, **kwargs)`. That one indirection is what lets the
post-construction transforms in `maxtext.m3.infra` act on every model without
model-specific code:

  * `infra.remat` sets `Layers.remat`, and the container checkpoints each
    layer call itself. No wrapper module, so parameter paths do not change.
  * `infra.scan` replaces a `Layers` (or a run of identical layers inside one)
    with a `ScannedLayers` that stacks the parameters and runs `jax.lax.scan`.

Parameter paths are `layers/<i>/...`, the layer index being an ordinary
`nnx.List` segment.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

from flax import nnx


@dataclasses.dataclass(frozen=True)
class Remat:
  """How a container checkpoints each layer call. Set by `maxtext.m3.infra.remat`.

  Attributes:
    policy: A `jax.checkpoint` policy. `None` is JAX's default and saves
      nothing -- full rematerialization. Named policies come from
      `jax.checkpoint_policies` and refer to the `checkpoint_name` tags that
      model files put on their activations.
  """

  policy: Callable[..., bool] | None = None


class Layers(nnx.List):
  """An `nnx.List` of layers that applies them in order when called.

  `remat` is a plain attribute rather than constructor state so that a
  transform can set it after the model is built. It lives in the graphdef, not
  the state, and so survives `nnx.split` / `nnx.merge` and `nnx.jit`.
  """

  remat: Remat | None = None

  def __call__(self, carry: Any, **kwargs: Any) -> Any:
    """Returns `carry` passed through every layer: `carry = layer(carry, **kwargs)`.

    `carry` is whatever the layers agree on -- a hidden-states array for dense
    decoders, a `(hidden, aux_loss)` pair for MoE decoders. `kwargs` (positions,
    segment ids, ...) are the same for every layer.
    """
    for layer in self:
      carry = apply_layer(layer, carry, self.remat, **kwargs)
    return carry


def apply_layer(layer: nnx.Module, carry: Any, remat: Remat | None, **kwargs: Any) -> Any:
  """Calls `layer(carry, **kwargs)`, under `jax.checkpoint` if `remat` is set.

  The per-layer kwargs are closed over rather than passed through the
  checkpointed function: they are the same arrays for every layer and
  `jax.checkpoint` treats closed-over tracers as inputs, so nothing is lost
  and the checkpointed signature stays `(layer, carry)`.
  """
  if remat is None:
    return layer(carry, **kwargs)
  return nnx.remat(lambda l, c: l(c, **kwargs), policy=remat.policy)(layer, carry)
