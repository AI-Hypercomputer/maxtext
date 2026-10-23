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

"""Training loss: next-token cross entropy plus whatever the model reports as auxiliary.

One function for every m3 model, with no model-type branching: a dense model
returns `aux_loss=None` and gets plain cross entropy; an MoE model returns its
already-weighted load-balance loss and gets it added. The sum is the single
scalar that is differentiated, which is the only way an auxiliary term
contributes gradient under the training engine (see `m3/README.md`).
"""

from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax


class LossBreakdown(NamedTuple):
  """What `loss_fn` reports alongside the differentiated total."""

  cross_entropy: jax.Array
  aux_loss: jax.Array


def cross_entropy(logits: jax.Array, targets: jax.Array, weights: jax.Array | None = None) -> jax.Array:
  """Mean next-token cross entropy in float32, optionally weighted per token.

  `weights` (0 for padding, 1 otherwise) scales each token's loss and the
  normalizer, so padded positions neither contribute nor dilute.
  """
  losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
  if weights is None:
    return jnp.mean(losses)
  weights = weights.astype(jnp.float32)
  return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> tuple[jax.Array, LossBreakdown]:
  """Total loss for one batch, and its parts.

  `batch` carries `inputs`, `targets`, `positions`, and optionally
  `segment_ids` (packed sequences) and `weights` (loss mask). Suitable as the
  function handed to `nnx.value_and_grad(..., has_aux=True)`.
  """
  output = model(batch["inputs"], batch["positions"], batch.get("segment_ids"))
  xent = cross_entropy(output.logits, batch["targets"], batch.get("weights"))
  aux = jnp.zeros((), jnp.float32) if output.aux_loss is None else output.aux_loss.astype(jnp.float32)
  return xent + aux, LossBreakdown(cross_entropy=xent, aux_loss=aux)
