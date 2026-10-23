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

"""Shared output types.

There are no model ABCs here. NNX modules are duck-typed, and the one thing
every m3 language model promises its drivers is the shape of what `__call__`
returns, which a `NamedTuple` states more precisely than a base class would.
"""

from typing import NamedTuple

import jax


class CausalLMOutput(NamedTuple):
  """What an m3 language model returns from `__call__`.

  Attributes:
    logits: `(batch, length, vocab)` next-token logits.
    aux_loss: A scalar auxiliary loss already multiplied by its weight, or
      `None` for models that have none. MoE routers' load-balance losses come
      out this way. Returning it explicitly, rather than `sow`ing it as an
      `nnx.Intermediate`, keeps the model's non-parameter state fixed across
      the forward pass, which the training engine depends on (see
      `m3/README.md`). `maxtext.m3.train.loss` adds it to the primary loss;
      a driver that ignores it trains without load balancing.
  """

  logits: jax.Array
  aux_loss: jax.Array | None = None
