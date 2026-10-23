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

"""Callable layer container used by the model-independent scan transform.

Models run repeated blocks with `carry = self.layers(carry, **kwargs)`.
`infra.scan` can replace this container with stacked, scanned layers.
Unscanned parameter paths use ordinary `nnx.List` indices: `layers/<i>/...`.
"""

from typing import Any

from flax import nnx


class Layers(nnx.List):
  """An `nnx.List` that applies its layers in order."""

  def __call__(self, carry: Any, **kwargs: Any) -> Any:
    """Passes `carry` through each layer, sharing `kwargs` across layers."""
    for layer in self:
      carry = layer(carry, **kwargs)
    return carry
