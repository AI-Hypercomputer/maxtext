# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and optimizer helpers for GLM-5.3-Flash."""

from collections.abc import Mapping
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.training_engine import abstract_engine
import optax

_GLM5_LOSS_BATCH_KEYS = (
    "inputs",
    "targets",
    "targets_segmentation",
    "targets_position",
    "inputs_segmentation",
    "inputs_position",
)


def _is_router_bias(v: Any) -> bool:
  """Returns True if the variable is a Glm5RouterBias."""
  return isinstance(v, modeling_glm5_3_flash.Glm5RouterBias)


def _get_param_label(path: tuple[Any, ...], v: Any) -> str:
  """Labels a parameter as 'frozen' if indexer or router bias, else 'train'."""
  if _is_router_bias(v):
    return "frozen"
  path_str = jax.tree_util.keystr(path)
  if "indexer" in path_str or "e_score_correction_bias" in path_str:
    return "frozen"
  return "train"


def glm5_engine_loss_fn(
    model: modeling_glm5_3_flash.Glm5Model,
    config: Any,
    data: Mapping[str, Any],
    dropout_rng: Any = None,
    params: Any = None,
    *,
    is_train: bool,
) -> tuple[abstract_engine.WeightedMetric, dict[str, jax.Array]]:
  """Engine loss function adapter conforming to MaxTextTrainingEngine signature."""
  del config, dropout_rng, params
  filtered_batch = {k: data[k] for k in _GLM5_LOSS_BATCH_KEYS if k in data}
  total_loss, aux = modeling_glm5_3_flash.glm5_loss_fn(model, is_train=is_train, **filtered_batch)
  return (
      abstract_engine.WeightedMetric(
          unreduced_sum=(total_loss * aux["total_weights"]).astype(jnp.float32),
          denominator=aux["total_weights"],
      ),
      aux,
  )


def create_glm5_frozen_mask(params: Any) -> Any:
  """Returns a boolean PyTree where True marks frozen parameters (indexer and router bias)."""
  return jax.tree_util.tree_map(lambda is_train: not is_train, modeling_glm5_3_flash.trainable_mask(params))


def create_glm5_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    **adamw_kwargs: Any,
) -> optax.GradientTransformation:
  """Creates an AdamW optimizer for GLM-5.3-Flash.

  Masks out Glm5RouterBias (e_score_correction_bias) and indexer parameters
  so weight decay NEVER decays zero-gradient parameters when weight_decay > 0.

  Args:
    learning_rate: Required learning rate.
    weight_decay: Required weight decay rate.
    **adamw_kwargs: Additional keyword arguments forwarded to optax.adamw.

  Returns:
    An optax.GradientTransformation using optax.multi_transform.
  """
  if not isinstance(learning_rate, (float, int)):
    raise TypeError(f"learning_rate must be a float or int, got {type(learning_rate)}")
  if not isinstance(weight_decay, (float, int)):
    raise TypeError(f"weight_decay must be a float or int, got {type(weight_decay)}")
  base_tx = optax.adamw(
      learning_rate=float(learning_rate),
      weight_decay=float(weight_decay),
      **adamw_kwargs,
  )
  return optax.multi_transform(
      {"train": base_tx, "frozen": optax.set_to_zero()},
      param_labels=lambda params: jax.tree_util.tree_map_with_path(
          _get_param_label,
          params,
          is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)),
      ),
  )


def configure_glm5_engine(engine: Any) -> Any:
  """Configures a MaxTextTrainingEngine for GLM-5.3-Flash training.

  Attaches GLM-5.3-Flash's custom loss function adapter (with auxiliary
  metrics),
  clears gen_model_input_fn, and freezes indexer and router bias parameters in
  the optimizer.

  Args:
    engine: A MaxTextTrainingEngine instance.

  Returns:
    The configured engine instance.
  """
  engine.with_loss_fn(
      glm5_engine_loss_fn,
      has_aux=True,
  )
  engine.with_gen_model_input_fn(None)

  # pylint: disable=protected-access
  base_tx = engine._optimizer.tx
  masked_tx = optax.multi_transform(
      {"train": base_tx, "frozen": optax.set_to_zero()},
      param_labels=lambda params: jax.tree_util.tree_map_with_path(
          _get_param_label,
          params,
          is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)),
      ),
  )
  engine._optimizer = engine._build_optimizer(masked_tx)
  engine._tx = masked_tx
  # pylint: enable=protected-access
  return engine


def train_step(
    model: modeling_glm5_3_flash.Glm5Model,
    optimizer: nnx.Optimizer,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Executes a single forward, backward, and optimizer update step.

  Contract:
    `decoder_target_tokens` and `decoder_target_mask` are next-token shifted
    targets (`tokens[:, 1:]` aligned with `tokens[:, :-1]`), matching
    MaxText's `loss_fn` convention.

  Args:
    model: The Glm5Model instance being trained.
    optimizer: The nnx.Optimizer instance.
    tokens: Input token ids of shape [batch_size, seq_len].
    decoder_target_tokens: Next-token shifted targets [batch_size, seq_len].
    decoder_target_mask: Next-token shifted loss mask [batch_size, seq_len].
    decoder_segment_ids: Optional input segmentation [batch_size, seq_len].
    decoder_positions: Optional input positions [batch_size, seq_len].

  Returns:
    (loss, aux_metrics) tuple where loss is a float32 scalar.
  """

  def loss_fn(
      mod: modeling_glm5_3_flash.Glm5Model,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    return modeling_glm5_3_flash.compute_glm5_loss(
        model=mod,
        tokens=tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_target_mask=decoder_target_mask,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        is_train=True,
    )

  (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
  optimizer.update(model, grads)
  return loss, aux
