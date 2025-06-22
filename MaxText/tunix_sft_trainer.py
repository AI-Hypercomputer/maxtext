from typing import Sequence
from functools import partial

import absl
from flax import nnx
import jax
import optax
import jax.numpy as jnp
from tunix import peft_trainer
from flax.nnx import bridge

import MaxText as mt


def dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
    'input_tokens': x.input_tokens,
    'input_mask': x.input_mask,
    'positions': jnp.arange(x.input_tokens.shape[1]),
    'attention_mask': jnp.ones_like(x.input_tokens),
  }


def dummy_datasets(batch_size: int, repeat: int = 1):
  # (num_batch, batch_size, seq_len)
  dummy_input = jnp.arange(128).reshape((-1, batch_size, 16))
  return [
    peft_trainer.TrainingInput(
      input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
    )
    for x in dummy_input
  ] * repeat



def assert_not_equal(path, x, y):
  jnp.testing.assert_(
    jnp.any(jnp.not_equal(x, y)), msg=f'Unexpected match at path: {path}'
  )


def gen_tunix_config(mt_config):
  # FIXME: build from our config
  return peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)


def basic_training(mt_config):
  # TODO: should this really return the mesh, or should that be a separate call that happens first
  linen_model, mesh, init_rng, *_ = mt.from_pretrained(mt_config)
  hooks = mt.sft_hooks.TrainingHooks(mt_config)
  # FIXME: Anisha's notebook calls setup_decode_state. Can we just not do that and assume that NNX handles it

  rngs = nnx.Rngs(0)
  # FIXME: what should be the shape of this
  x = jnp.ones((1, 32))
  model = bridge.ToNNX(linen_model, rngs).lazy_init(x)
  original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

  # FIXME: move to real datasets
  eval_ds = train_ds = dummy_datasets(batch_size=4)

  tunix_config = gen_tunix_config(mt_config)
  trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), tunix_config, hooks)
  trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

  with mesh:
    trainer.train(train_ds, eval_ds)

  variables = nnx.state(model, nnx.Param)

  jax.tree.map_with_path(assert_not_equal, original_variables, variables)
  assert trainer._metrics_logger.get_metric('perplexity', 'train') > 0
  assert trainer._metrics_logger.get_metric('perplexity', 'eval') > 0
  assert trainer._train_steps > 0
  assert len(trainer._metrics_logger.get_metric_history('perplexity', 'train')) == trainer._train_steps


def main(argv: Sequence[str]) -> None:
  mt_config = mt.pyconfig.initialize(argv)
  basic_training(mt_config)


if __name__ == "__main__":
  absl.app.run(main)
