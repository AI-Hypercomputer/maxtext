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

"""Loss function and optimizer mask for GLM-5.3-Flash training.

Note on Sequence Packing (F16):
Host-side sequence packing validation is enforced by:
- Glm5Config.__post_init__ raising ValueError when packing=True
  (src/maxtext/m3/models/glm5_3_flash/glm5_config.py:136-137)
- glm5_config.validate_no_packing(decoder_segment_ids)
  (src/maxtext/m3/models/glm5_3_flash/glm5_config.py:35-41)
- MaxText data processing pipelines honoring config.packing
  (src/maxtext/input_pipeline/data_processing_utils.py:93,
   src/maxtext/input_pipeline/tfds_data_processing.py:241,
   src/maxtext/input_pipeline/hf_data_processing.py:550)
"""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.glm5_3_flash import glm5_moe
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import optax


def glm5_loss_fn(
    model: modeling_glm5_3_flash.Glm5Model,
    *,
    inputs: jax.Array,
    targets: jax.Array,
    targets_segmentation: jax.Array,
    # Custom gen_model_input_fn or unsegmented TrainerPayload batches may omit
    # inputs_segmentation and inputs_position (src/maxtext/training_engine/maxtext_engine.py:1155-1165, 1331-1344).
    # Because GLM-5.3-Flash is strictly NoPE (qk_rope_head_dim=0) and unpacked (packing=False),
    # decoder_positions is unused and decoder_segment_ids=None treats all input tokens as unpadded.
    inputs_segmentation: jax.Array | None = None,
    inputs_position: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes masked token cross-entropy loss and folds MoE load-balancing loss.

  Compatible with MaxTextTrainingEngine's loss_fn(model,
  **gen_model_input_fn(payload))
  convention (src/maxtext/training_engine/maxtext_engine.py:480-486, 870, 1165,
  1283).

  Args:
    model: The Glm5Model instance being trained.
    inputs: Input token ids of shape [batch_size, seq_len].
    targets: Target token ids of shape [batch_size, seq_len].
    targets_segmentation: Required segmentation mask of shape [batch_size,
      seq_len] where non-zero positions are valid target tokens and zero
      positions are padding.
    inputs_segmentation: Optional input segmentation mask of shape [batch_size,
      seq_len]. May be omitted by unsegmented TrainerPayload batches
      (src/maxtext/training_engine/maxtext_engine.py:1155-1165, 1331-1344); None
      treats all tokens as unpadded since packing is unsupported.
    inputs_position: Optional input positions of shape [batch_size, seq_len].
      May be omitted by custom gen_model_input_fn pipelines; unused because
      GLM-5.3-Flash is strictly NoPE.

  Returns:
    (total_loss, aux_metrics) tuple where aux_metrics contains 'ce_loss',
    'moe_lb_loss', 'xent_sum', and 'total_weights'.
  """
  if inputs is None:
    raise ValueError("inputs must be provided as a non-None array.")
  if targets is None:
    raise ValueError("targets must be provided as a non-None array.")
  if targets_segmentation is None:
    raise ValueError("targets_segmentation must be provided as a non-None array.")

  inputs_arr = jnp.asarray(inputs)
  targets_arr = jnp.asarray(targets)
  targets_seg_arr = jnp.asarray(targets_segmentation)

  if targets_seg_arr.shape != targets_arr.shape:
    raise ValueError(f"targets_segmentation shape {targets_seg_arr.shape} != targets shape {targets_arr.shape}.")

  decoder_segment_ids = jnp.asarray(inputs_segmentation) if inputs_segmentation is not None else None
  decoder_positions = jnp.asarray(inputs_position) if inputs_position is not None else None

  out = model.forward_with_aux(
      inputs_arr,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
  )
  logits = out.logits.astype(jnp.float32)

  if targets_arr.shape != logits.shape[:2]:
    raise ValueError(f"targets shape {targets_arr.shape} does not match logits batch/seq shape {logits.shape[:2]}.")

  mask = (targets_seg_arr != 0).astype(jnp.float32)
  token_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets_arr)
  masked_losses = token_losses * mask
  xent_sum = jnp.sum(masked_losses)
  total_weights = jnp.sum(mask)
  ce_loss = xent_sum / jnp.maximum(total_weights, 1.0)

  moe_lb_loss = out.aux["moe_lb_loss"]
  total_loss = ce_loss + model.config.moe_lb_loss_weight * moe_lb_loss

  aux_metrics = {
      "ce_loss": ce_loss,
      "moe_lb_loss": moe_lb_loss,
      "xent_sum": xent_sum,
      "total_weights": total_weights,
  }

  if model.mtp is not None:
    h_i = out.aux["final_hidden_states"]
    mtp_targets = jnp.roll(targets_arr, -1, axis=-1)
    valid_mask = mask != 0
    mtp_valid_mask = valid_mask & jnp.concatenate(
        [valid_mask[:, 1:], jnp.zeros((valid_mask.shape[0], 1), dtype=jnp.bool_)], axis=-1
    )
    mtp_logits, mtp_router_logits = model.mtp(
        h_i,
        targets_arr,
        embed_tokens=model.embed_tokens,
        lm_head=model.lm_head,
        decoder_segment_ids=decoder_segment_ids,
    )
    mtp_token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=mtp_logits.astype(jnp.float32), labels=mtp_targets
    )
    mtp_mask_f32 = mtp_valid_mask.astype(jnp.float32)
    mtp_total_weights = jnp.sum(mtp_mask_f32)
    mtp_ce_loss = jnp.sum(mtp_token_losses * mtp_mask_f32) / jnp.maximum(mtp_total_weights, 1.0)

    flat_mtp_router = [mtp_router_logits.reshape(-1, mtp_router_logits.shape[-1])]
    mtp_moe_lb_loss = glm5_moe.compute_moe_load_balancing_loss(
        flat_mtp_router,
        model.config.n_routed_experts,
        model.config.num_experts_per_tok,
        attention_mask=valid_mask,
    )
    total_loss = (
        total_loss
        + model.config.mtp_loss_scaling_factor * mtp_ce_loss
        + model.config.moe_lb_loss_weight * mtp_moe_lb_loss
    )
    aux_metrics["mtp_loss"] = mtp_ce_loss
    aux_metrics["mtp_moe_lb_loss"] = mtp_moe_lb_loss

  return total_loss, aux_metrics


def trainable_mask(model_or_params: Any) -> Any:
  """Returns a boolean PyTree matching the input parameter/update tree (False on indexer.*)."""
  tree = nnx.state(model_or_params, nnx.Param) if isinstance(model_or_params, nnx.Module) else model_or_params

  def _is_trainable(path: tuple[Any, ...], leaf: Any) -> bool:
    del leaf
    return "indexer" not in jax.tree_util.keystr(path)

  return jax.tree_util.tree_map_with_path(
      _is_trainable,
      tree,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
