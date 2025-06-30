#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
ATTENTION: This unit test should only be run on TPU v4-8. The test
may fail on different versions like v5p-8, v6e-8

TODO: b/413146740 - Match logits on other TPU versions

Runs GRPO trainer unit test correctness with golden logits generated
  from maxtext/MaxText/scratch_code/generate_grpo_golden_logits.py

Usage:
  pytest MaxText/tests/grpo_trainer_correctness_test.py
"""

from collections.abc import Callable
import functools
import os
import sys
import subprocess
import unittest

import pytest

import jsonlines

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

import transformers

from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import Array
from MaxText.experimental.rl.grpo_trainer import compute_log_probs, grpo_loss_fn, _merge_grpo_state, generate_completions
from MaxText.globals import PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText import optimizers

import pathwaysutils

def get_golden_data(config):
  """Get the golden data for GrpoTrainer from maxtext/MaxText/scratch_code/generate_grpo_golden_logits.py."""
  input_golden_data_path = os.path.join(PKG_DIR, "test_assets", f"golden_data_grpo_{config.model_name}.jsonl")
  print(f"Loading {input_golden_data_path}")
  with jsonlines.open(input_golden_data_path, "r") as reader:
    return next(iter(reader))


def setup_maxtext_model(config):
  """setup maxtext model"""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  quant = quantizations.configure_quantization(config)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  maxtext_model = models.Transformer(config=config, mesh=mesh, quant=quant)
  state, state_mesh_annotations = maxtext_utils.setup_decode_state(maxtext_model, config, init_rng, mesh, None)
  state_mesh_shardings = nn.logical_to_mesh_sharding(state_mesh_annotations, mesh, config.logical_axis_rules)
  data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  reference_params = jax.tree.map(jnp.copy, state.params["params"])
  state = _merge_grpo_state(state, reference_params)
  return maxtext_model, state, reference_params, init_rng, state_mesh_shardings, data_sharding


def prepare_maxtext_inputs(input_str, tokenizer_model):
  """prepare maxtext inputs"""
  prompt = tokenizer_model.encode(input_str)
  input_ids = jnp.pad(
      jnp.tile(jnp.concat([jnp.array(prompt), jnp.array(prompt)], axis=-1), (4, 1)),
      ((0, 0), (0, 4)),
      constant_values=tokenizer_model.pad_token_type_id,
  )  # pad some tokens at the end of input prompt
  input_segmentation = (input_ids > 0).astype(jnp.int32)
  input_position = jnp.where(input_segmentation, jnp.arange(input_segmentation.shape[1]), 0)
  completion_segmentation = jnp.tile(
      jnp.pad(jnp.array([0] * len(prompt) + [1] * len(prompt)), (0, input_ids.shape[1] - 2 * len(prompt))), (4, 1)
  )
  return input_ids, input_segmentation, input_position, completion_segmentation


class GrpoTrainerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    command = [
        "gsutil",
        "cp",
        "-r",
        "gs://maxtext-dataset/hf/llama3.1-tokenizer",
        os.path.join(os.path.dirname(PKG_DIR), "assets", ""),
    ]
    exit_code = subprocess.call(command, cwd=os.path.dirname(PKG_DIR))
    if exit_code != 0:
      raise ValueError(f"{command} failed with exit code: {exit_code}")
    self.config = pyconfig.initialize(
        [None, "MaxText/experimental/rl/grpo_trainer_test.yml"],
        run_name="unit_test_grpo_trainer",
        tokenizer_path=os.path.join(os.path.dirname(PKG_DIR), "assets", "llama3.1-tokenizer"),
        enable_checkpointing=False,
    )
    self.config_inference = pyconfig.initialize(
        [None, "MaxText/experimental/rl/grpo_trainer_test.yml"],
        run_name="unit_test_grpo_trainer_inference",
        tokenizer_path=os.path.join(os.path.dirname(PKG_DIR), "assets", "llama3.1-tokenizer"),
        enable_checkpointing=False,
        ici_tensor_parallelism=4,
        per_device_batch_size=self.config.per_device_batch_size * self.config.num_generations,
    )
    self.rtol = 1e-05
    self.atol = 1e-08
    self.rng = jax.random.PRNGKey(self.config.init_weights_seed)
    self.tokenizer_model = transformers.AutoTokenizer.from_pretrained(
        self.config.tokenizer_path,
        add_bos_token=self.config.add_bos,
        add_eos_token=self.config.add_eos,
        legacy=False,
        padding_side="left",
    )
    self.tokenizer_model.add_special_tokens({"pad_token": "<pad>"})

  @pytest.mark.skip(reason="Logit output test fragile, failing on jax upgrade to 0.6.2 - see b/425997645")
  @pytest.mark.tpu_only  # ATTENTION: Only run on TPU V4-8
  def test_grpo_trainer_correctness(self):
    # Get the expected (golden) data.
    golden_data = get_golden_data(self.config)
    # Initialize the model and related objects.
    maxtext_model, state, reference_params, rng, state_mesh_shardings, data_sharding = setup_maxtext_model(self.config)
    # Prepare inputs for the model.
    input_ids, input_segmentation, input_position, completion_segmentation = prepare_maxtext_inputs(
        self.config.prompt, self.tokenizer_model
    )
    # Obtain per-token logits.
    maxtext_per_token_logps, _ = compute_log_probs(
        maxtext_model,
        state.params,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
        self.config,
        is_train=False,
        rngs=self.rng,
    )
    jax.debug.print("maxtext_per_token_logps={maxtext_per_token_logps}", maxtext_per_token_logps=maxtext_per_token_logps)
    jax.debug.print(
        "golden_per_token_logps={golden_per_token_logps}",
        golden_per_token_logps=golden_data["maxtext_per_token_logps_no_ckpt_loading"],
    )
    golden_maxtext_logits = np.array(golden_data["maxtext_per_token_logps_no_ckpt_loading"])
    self.assertTrue(jnp.all(np.array(golden_data["input_ids"]) == np.array(input_ids[0])))
    self.assertTrue(
        jax.numpy.allclose(
            maxtext_per_token_logps[0],
            golden_maxtext_logits,
            rtol=float(self.rtol),
            atol=float(self.atol),
            equal_nan=False,
        )
    )
    max_diff = np.max(np.abs(np.subtract(maxtext_per_token_logps[0], golden_maxtext_logits)))
    print("Max numerical difference:", max_diff)

    # Create the data dictionary required for computing the loss.
    data = {
        "prompt_completions": input_ids,
        "prompt_completions_position": input_position,
        "prompt_completions_segmentation": input_segmentation,
        "ar_completions_segmentation": completion_segmentation,
    }
    # Compute the loss and auxiliary values.
    maxtext_loss, aux = grpo_loss_fn(maxtext_model, self.config, data, rng, state.params, reference_params)

    # Assert that the computed loss and auxiliary averages match the golden data.
    self.assertEqual(maxtext_loss.tolist(), golden_data["maxtext_loss"])
    self.assertEqual(aux.avg_kl.tolist(), golden_data["avg_kl"])
    self.assertEqual(aux.avg_advantage.tolist(), golden_data["avg_advantage"])

    # Generate completions using the inference engine.
    engine = maxengine.MaxEngine(self.config_inference)
    _ = engine.load_params(rng)
    prompt_tokens = self.tokenizer_model.encode(self.config_inference.prompt)
    prompt = jnp.pad(
        jnp.tile(jnp.array(prompt_tokens), (4, 1)),
        ((0, 0), (0, 4)),
        constant_values=self.tokenizer_model.pad_token_type_id,
    )
    prompt_true_length = jnp.array([len(prompt_tokens)] * 4)
    engine_data = {"prompt": prompt, "prompt_true_length": prompt_true_length}
    p_generate_completions: Callable[[dict, dict, Array], Array] = jax.jit(
        functools.partial(generate_completions, self.config, self.tokenizer_model, engine),
        in_shardings=(data_sharding, state_mesh_shardings.params, None),
        out_shardings=data_sharding,
        donate_argnums=(0,),
    )
    # pylint: disable=not-callable
    engine_data = p_generate_completions(engine_data, {"params": state.params["params"]}, rng)
    # Assert that the generated completions match the golden reference.
    self.assertEqual(engine_data["prompt_completions"][0].tolist(), golden_data["generated_completions"])

class ReshardTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    pathwaysutils.initialize()
    self.training_config, self.inference_config = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(self.training_config.init_weights_seed)
  
  def init_pyconfig(self, **kwargs):
    """Initialize MaxText pyconfig."""
    init_kwargs = {
        "run_name": "test",
        # Parallelism
        "per_device_batch_size": 1,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": -1,
        # Model
        "model_name": "gemma2-2b",
        "skip_jax_distributed_system": True,
        "enable_checkpointing": False,
    } | kwargs
    training_config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    inference_config = pyconfig.initialize(
      [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml"), "ici_fsdp_parallelism=1", "ici_tensor_parallelism=4",],
        **init_kwargs,
    )
    return training_config, inference_config
  
  def init_named_sharded_params(self, rng_key, mesh):
    """
    Create a nested param tree matching your structure, with arrays initialized
    randomly and sharded using NamedSharding and PartitionSpec.
    """
    key = rng_key

    def make(name, shape, dtype, pspec):
      nonlocal key
      key, sub = jax.random.split(key)
      arr = jax.random.normal(sub, shape, dtype=dtype)
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*pspec))
      return jax.device_put(arr, sharding)

    return {
      'decoder': {
          'decoder_norm': {
              'scale': make(
                  'decoder_norm/scale',
                  (2304,),
                  jnp.float32,
                  pspec=(('tensor', 'tensor_transpose', 'tensor_sequence'),)
              ),
          },
          'layers': {
              'mlp_local': {
                  'wi_0': {'kernel': make('mlp_local/wi_0/kernel',
                                            (2304,13,9216), jnp.float32,
                                            pspec=(('fsdp','sequence','tensor_transpose','context','expert'),
                                                  'stage',
                                                  ('fsdp_transpose','tensor','tensor_sequence','autoregressive')))},
              },
              # Norms
              **{
                  'post_ffw_norm_global': {
                      'scale': make(f'post_ffw_norm_global/scale',
                                    (2304,13), jnp.float32,
                                    pspec=(('tensor','tensor_transpose','tensor_sequence'),('stage')))
                  }
              },
              # Self-Attention
              'self_attention_global': {
                    'self_attention_global': {
                        'kernel': make(f'self_attention_global/key/kernel',
                                        (2304,13,8,256), jnp.float32,
                                        pspec=(('fsdp','fsdp_transpose','sequence','context','expert'),
                                              'stage',
                                              ('tensor','tensor_transpose','tensor_sequence'),
                                              'autoregressive'))
                    },
              },
          },
      },
      'token_embedder': {
        'embedding': make(
          'token_embedder/embedding',
          (256128,2304), jnp.float32,
          pspec=(('tensor','tensor_transpose','tensor_sequence','autoregressive'),
          ('fsdp','fsdp_transpose','sequence','context','expert'))
        )
      }
    }
  
  def test_pathways_reshard(self):
    # assume the test runs on 32 v5p devices
    training_devices = jax.devices()[16:]
    training_devices = maxtext_utils.create_device_mesh(self.training_config, devices=training_devices)
    training_mesh = Mesh(training_devices, self.training_config.mesh_axes)
    inference_devices = jax.devices()[:4]
    inference_devices = maxtext_utils.create_device_mesh(config=self.inference_config, devices=inference_devices)
    inference_mesh = jax.sharding.Mesh(inference_devices, self.inference_config.mesh_axes)
    # model = models.Transformer(config=self.training_config, mesh=training_mesh, quant=None)
    # inference_model = models.Transformer(self.inference_config, inference_mesh, quant=None)
    # tx = optimizers.get_optimizer(self.training_config, 0.01)
    # state, _, state_mesh_shardings, _ = maxtext_utils.setup_training_state(
    #     model, None, tx, self.training_config, self.rng, training_mesh, None
    # )
    # inference_state_mesh_shardings = maxtext_utils.get_abstract_state(inference_model, tx, self.inference_config, self.rng, inference_mesh, is_training=False)[2]
    
    def list_params_with_paths(params):
      """
      Return a list of (string path, array) entries from a pytree of parameters.
      """
      path_vals, _ = jax.tree_util.tree_flatten_with_path(params)
      result = []
      for keypath, leaf in path_vals:
          # Convert KeyPath to human-readable string
          readable = jax.tree_util.keystr(keypath)
          result.append((readable, leaf))
      return result

    # # Usage example
    # pairs = list_params_with_paths(inference_state_mesh_shardings.params)
    # for path, arr in pairs:
    #   print(f"{path}: sharding={arr}")
    params = self.init_named_sharded_params(self.rng, training_mesh)
    def apply_resharding(params, mesh):
      """
      Shard each array in a pytree according to spec_map.

      - params: nested pytree (e.g. state.params)
      - mesh: JAX mesh with named axes
      """
      def fn(path, leaf):
        spec = leaf.sharding.spec
        if spec is None:
          return leaf
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        with (
          jax.transfer_guard_device_to_host("disallow_explicit"),
          jax.transfer_guard_host_to_device("disallow_explicit"),
        ):
          return jax.device_put(leaf, sharding)

      return jax.tree_util.tree_map_with_path(fn, params)
    resharded_params = apply_resharding(params, inference_mesh)


if __name__ == "__main__":
  unittest.main()
