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

"""Train utils for training a ResNet model on the ImageNet dataset.

The data is loaded using tensorflow_datasets.
The function create_model() can accept an hparam file as input, which will
  determine the model's size, training parameters, quantization precisions, etc.
"""

import dataclasses
import functools
from typing import Any, Mapping
from aqt.jax_legacy.jax.imagenet import input_pipeline
from aqt.jax_legacy.jax.imagenet import models
from aqt.jax_legacy.jax.imagenet import pokebnn

import flax
from flax import jax_utils
from flax import struct
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
from jax import lax
import jax.nn
import jax.numpy as jnp
import optax


def create_model(
    key,
    batch_size,
    image_size,
    model_dtype,
    hparams,
    train=True,
    is_teacher=True,  # create vanilla resnet by default
    **kwargs):
  """Creates the ResNet model using hparams."""
  input_shape = (batch_size, image_size, image_size, 3)
  if is_teacher:  # create teacher model
    model = models.create_resnet(hparams, model_dtype, train, **kwargs)
  else:  # create PokeBNN model
    model = pokebnn.create_pokebnn(hparams=hparams, train=train, **kwargs)
  init_state = model.init(key, jnp.zeros(input_shape, dtype=model_dtype))
  return model, init_state


def cross_entropy_loss(logits, labels):
  logits = jax.nn.log_softmax(logits)
  return -jnp.sum(
      common_utils.onehot(labels, num_classes=1000) * logits) / labels.size


def kl_div_loss(logits, teacher_logits):
  # Attention: logits are *not* log_softmaxed
  # Attention: teacher logits are probability distributions (softmaxed)
  log_prob_x = jax.nn.log_softmax(logits)  # default axis is -1
  # prob_y = jax.nn.softmax(teacher_logits)  # default axis is -1
  return -jnp.sum(teacher_logits * log_prob_x) / logits.shape[0]


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def train_step(model, state, batch, hparams, update_bounds, quantize_weights,
               learning_rate_fn, teacher):
  """Perform a single training step."""

  # TODO(yichi): update the quantize_weights flag in dynamic_context
  dynamic_context = dataclasses.replace(
      model.dynamic_context,
      update_bounds=update_bounds,
      quantize_weights=quantize_weights)

  model = dataclasses.replace(model, dynamic_context=dynamic_context)

  def loss_fn(params):
    """loss function used for training."""
    variables = {'params': params, **state.mutable_vars}
    logits, new_model_state = model.apply(
        variables, batch['image'], mutable=['batch_stats', 'get_bounds'])
    # TODO(yichi): use the checkpoint and the 8-bit model to compute logits
    teacher_logits = teacher['model'](teacher['variables'], batch['image'],
                                      batch['label'])
    # TODO(yichi): replace cross_entropy_loss with KL div loss
    loss = kl_div_loss(logits, teacher_logits)
    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = hparams.weight_decay
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grad = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = lax.pmean(grad, axis_name='batch')
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  updates, new_opt_state = state.tx.update(grad, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  new_state = state.replace(
      step=step+1,
      params=new_params,
      mutable_vars=new_model_state,
      opt_state=new_opt_state,
      dynamic_scale=dynamic_scale)

  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and the old optimizer
    # state should be restored.
    new_state = new_state.replace(
        opt_state=jax.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


# pylint: disable=missing-function-docstring
def eval_step(model, state, batch, quantize_weights):
  # TODO(yichi): update the quantize_weights flag in dynamic_context
  dynamic_context = dataclasses.replace(
      model.dynamic_context, quantize_weights=quantize_weights)
  model = dataclasses.replace(model, dynamic_context=dynamic_context)

  variables = {'params': state.params, **state.mutable_vars}
  model = dataclasses.replace(model, train=False)
  logits = model.apply(variables, batch['image'], mutable=False)
  return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(batch_size, data_dir, image_size, dtype, train, cache):
  """Creates input data iterator."""
  ds = input_pipeline.load_split(
      batch_size,
      data_dir=data_dir,
      image_size=image_size,
      dtype=dtype,
      train=train,
      cache=cache)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


class TrainState(struct.PyTreeNode):
  """Train state keeping track of step, parameters, optimizer state, and etc."""
  step: int
  params: flax.core.FrozenDict[str, Any]
  mutable_vars: Mapping[str, Any]
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState
  dynamic_scale: dynamic_scale_lib.DynamicScale

  @classmethod
  def create(cls, *, params, tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    return cls(
        step=0,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

  new_mutable_vars = state.mutable_vars.copy(
      {'batch_stats': avg(state.mutable_vars['batch_stats'])})
  return state.replace(mutable_vars=new_mutable_vars)
