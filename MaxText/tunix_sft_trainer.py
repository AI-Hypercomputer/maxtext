from typing import Sequence

from absl import app

from flax import nnx
import jax
import optax
import jax.numpy as jnp
from tunix.sft import peft_trainer
from flax.nnx import bridge

import MaxText as mt

import datetime
import os
from typing import Sequence

from absl import app

import numpy as np

import tensorflow as tf

import jax


from MaxText import checkpointing
from MaxText import exceptions
from MaxText import max_utils
from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import profiler
from MaxText import pyconfig
from MaxText import train_utils
from MaxText.data_loader import DataLoader
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    validate_train_config,
)


def dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
      "input_tokens": x.input_tokens,
      "input_mask": x.input_mask,
      "positions": jnp.arange(x.input_tokens.shape[1]),
      "attention_mask": jnp.ones_like(x.input_tokens),
  }


def dummy_datasets(batch_size: int, repeat: int = 1):
  # (num_batch, batch_size, seq_len)
  dummy_input = jnp.arange(128).reshape((-1, batch_size, 16))
  return [
      peft_trainer.TrainingInput(input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)) for x in dummy_input
  ] * repeat


def assert_not_equal(path, x, y):
  jnp.testing.assert_(jnp.any(jnp.not_equal(x, y)), msg=f"Unexpected match at path: {path}")


def gen_tunix_config(mt_config):
  ## TODO: Add config
  return peft_trainer.TrainingConfig(
      eval_every_n_steps=2,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_directory,
  )


def basic_training(mt_config):
  linen_model = mt.from_pretrained(mt_config)
  # hooks = mt.sft_hooks.TrainingHooks(mt_config)

  rngs = nnx.Rngs(0)
  # FIXME: what should be the shape of this
  x = jnp.ones((1, 32))
  model = bridge.ToNNX(linen_model, rngs).lazy_init(x)
  original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

  # FIXME: move to real datasets
  eval_ds = train_ds = dummy_datasets(batch_size=4)
  print("DATASET: ", train_ds)

  tunix_config = gen_tunix_config(mt_config)
  trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), tunix_config, hooks)
  trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

  with linen_model.mesh:
    trainer.train(train_ds, eval_ds)

  variables = nnx.state(model, nnx.Param)

  jax.tree.map_with_path(assert_not_equal, original_variables, variables)
  # assert trainer._metrics_logger.get_metric('perplexity', 'train') > 0
  # assert trainer._metrics_logger.get_metric('perplexity', 'eval') > 0
  # assert trainer._train_steps > 0
  # assert len(trainer._metrics_logger.get_metric_history('perplexity', 'train')) == trainer._train_steps


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(mt_config)
  os.environ["TFDS_DATA_DIR"] = mt_config.dataset_path

  ## TODO: FIXME
  # maybe_monitor_goodput(config)
  # recorder = create_goodput_recorder(config)
  # with maybe_record_goodput(recorder, GoodputEvent.JOB):
  #  train_loop(config, recorder)

  basic_training(mt_config)


if __name__ == "__main__":
  app.run(main)
