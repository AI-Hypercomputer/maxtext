"""Training loop and Decoding of the model."""

import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from absl import app
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import numpy as np
import optax

from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.maps import Mesh

from layers import Transformer
from config import T5Config
from input_pipeline import get_datasets
import temperature_sampler

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]


# -----------------------------------------------------------------------------
# Flax TrainState with mutable variables field
# -----------------------------------------------------------------------------
# TODO(levskaya): upstream this field to the main TrainState.
class MutableTrainState(train_state.TrainState):
    mutables: Optional[flax.core.FrozenDict[str, Any]]


# -----------------------------------------------------------------------------
# Data Iterators
# -----------------------------------------------------------------------------

# A generator should return (input, target, segments, positions)
def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None),] * x.ndim
  slices[axis] = slice(0, -1)
  padded = np.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[tuple(slices)]

def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted

def data_generator(train_ds):
  train_iter = iter(train_ds)
  while True:
    data = jax.tree_util.tree_map(np.asarray, next(train_iter))
    yield (
      shift_inputs(data['inputs'], data['inputs_segmentation']),
      data['inputs'],
      data['inputs_segmentation'],
      data['inputs_position']
    )


# Helpers
# -----------------------------------------------------------------------------

# learning rate scheduling
def rsqrt_schedule(init_value: float, shift: int = 0):
  def schedule(count):
    return init_value * (count + shift)**-.5 * shift**.5
  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps
          ),
      rsqrt_schedule(
        init_value=learning_rate,
        shift=warmup_steps),
    ],
    boundaries=[warmup_steps])


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------

def init_train_state(model, tx, config, key):
  # We pass in "static" objects like model, tx, config as JAX compares them by
  # object hash, and instantiating them inside causes pjit top-level annotations
  # to fail to match as pytree prefixes if we re-instantiate.
  input_shape = (
      len(jax.devices()) * config.per_device_batch_size,
      config.max_target_length
  )
  model_vars = model.init({'params': key, 'dropout': key},
                          jnp.ones(input_shape),
                          jnp.ones(input_shape))
  state = MutableTrainState.create(
      apply_fn=model.apply,
      params=model_vars['params'],
      tx=tx,
      mutables=None)
  return state


def train_step(model, state, apply_args, dropout_rng):
  inputs, targets, segments, positions = apply_args
  rng1, rng2 = jax.random.split(dropout_rng)

  def loss_fn(params):
    logits = model.apply({'params': params},
                         inputs,
                         targets,
                         segments,
                         positions,
                         rngs={'dropout': rng1})
    # TODO: is optax xent as good as custom T5X one?
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    # Mask out paddings at the end of each example.
    xent = xent * (segments != 0)
    # TODO: mask out the prompt if training prefix-LM
    return jnp.sum(xent), logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, _), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)

  metrics = {'loss': loss}

  return new_state, metrics, rng2


def predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  """Predict language model on a batch."""
  # NOTE: wtf are we adding inputs.shape[2:] here?  it's almost always empty??
  target_shape = (inputs.shape[0], config.max_predict_length) + inputs.shape[2:]

  initial_variables = model.init(
    jax.random.PRNGKey(0),
    jnp.ones(target_shape, config.dtype),
    None,
    enable_dropout=False,
    decode=True,
    max_decode_length=config.max_predict_length
  )
  cache = initial_variables["cache"]

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = model.apply(
        {
            "params": state.params,
            "cache": flat_cache
        },
        flat_ids,
        None,
        enable_dropout=False,
        decode=True,
        max_decode_length=config.max_predict_length,
        mutable=["cache"])
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # search over possible sequences given input encoding.
  seqs = temperature_sampler.temperature_sample(
      inputs,
      cache,
      tokens_ids_to_logits,
      rngkey,
      temperature=config.sampling_temperature,
      topk=config.sampling_top_k,
      eos_token=config.eos_id)

  return seqs


# Train Loop
# ---------------------------------------------------------------------------

def train_loop(
  config,
  state=None,
  ckpt_path='~/flaxformer/lm1b'):

  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(0), 2)

  # Set up datasets.
  train_ds, eval_ds, predict_ds, sp_tokenizer = get_datasets(
      n_devices=jax.local_device_count(),
      config=config,
      vocab_path=None)
  data_gen = data_generator(train_ds)

  # Model and Optimizer definition
  model = Transformer(config)
  tx = optax.adam(
    create_learning_rate_schedule(
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps))

  # Mesh definition
  mesh = Mesh(mesh_utils.create_device_mesh(config.mesh_shape), config.mesh_axes)

  # Abstract initialization
  init_fn = functools.partial(init_train_state, model, tx, config)
  abstract_state = jax.eval_shape(init_fn, init_rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)

  print(f'devices {jax.devices()}')

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
      state = pjit(
          init_fn,
          in_axis_resources=None,
          out_axis_resources=state_mesh_annotations
      )(init_rng)

  # Checkpoint Restoration
  # TODO: we shouldn't run full init compilation when we need to load ckpt.
  if config.restore_checkpoints:
    state = checkpoints.restore_checkpoint(ckpt_path, state)

  data_pspec = nn.logical_to_mesh(P('batch'))

  p_train_step = pjit(
    train_step,
    in_axis_resources=(state_mesh_annotations,
                       data_pspec,
                       None),
    out_axis_resources=(state_mesh_annotations, None, None),
    static_argnums=(0,))

  # TODO: add held-out eval step.

  p_predict_step = pjit(
      functools.partial(predict_step, model=model, config=config),
      in_axis_resources=(data_pspec,
                        state_mesh_annotations,
                        None),
      out_axis_resources=None
  )

  # Prediction helpers.

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == config.eos_id) + 1].astype(np.int32)
    return sp_tokenizer.detokenize(valid_toks).numpy().decode("utf-8")

  def encode_strings(strs, max_len):
    tokenized_batch = np.zeros((len(strs), max_len), np.int32)
    for i, s in enumerate(strs):
      toks = sp_tokenizer.tokenize(s).numpy()
      # Remove EOS token in prompt.
      tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
    return tokenized_batch

  tokenized_prompts = encode_strings(
      [config.prompt], config.max_predict_length)

  # Main Loop

  for step in np.arange(config.steps):

    example_batch = next(data_gen)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(model, state, example_batch, nextrng)

    if step % config.log_period == 0:
      print(step, metrics)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        seqs = p_predict_step(tokenized_prompts, state, nextrng)
        print(decode_tokens(seqs[0]))

    if step % config.save_period == 0:
      if config.save_checkpoints and step != 0:
        checkpoints.save_checkpoint(
            ckpt_path, state, step=step, keep=1000, overwrite=True)

  return state, model


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = T5Config()
  state, model = train_loop(config)


if __name__ == '__main__':
  app.run(main)
