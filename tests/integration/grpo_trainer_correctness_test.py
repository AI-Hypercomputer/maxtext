# Copyright 2023–2025 Google LLC
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

"""
ATTENTION: This unit test should only be run on TPU v4-8. The test
may fail on different versions like v5p-8, v6e-8

TODO: b/413146740 - Match logits on other TPU versions

Runs GRPO trainer unit test correctness with golden logits generated
  from maxtext/tests/assets/logits_generation/generate_grpo_golden_logits.py

Usage:
  pytest tests/integration/grpo_trainer_correctness_test.py
"""

import functools
import os
import subprocess
import sys
import unittest
import pytest
import jsonlines
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn

import transformers

import MaxText as mt
from MaxText import maxengine
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.experimental.rl.grpo_trainer import grpo_loss_fn, _merge_grpo_state, setup_train_loop
from MaxText.experimental.rl.grpo_utils import compute_log_probs
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT, MAXTEXT_TEST_ASSETS_ROOT
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.experimental.rl import grpo_utils
from maxtext.inference import offline_engine
from maxtext.inference.offline_engine import InputData
from maxtext.utils import maxtext_utils

# This test is for serving pathways via offline_engine and maxengine.
pytestmark = [pytest.mark.external_training]


def get_golden_data(config):
  """Get the golden data for GrpoTrainer from tests/assets/logits_generation/generate_grpo_golden_logits.py."""
  input_golden_data_path = os.path.join(
      MAXTEXT_TEST_ASSETS_ROOT,
      "golden_logits",
      f"golden_data_grpo_{config.model_name}.jsonl",
  )
  print(f"Loading {input_golden_data_path}")
  with jsonlines.open(input_golden_data_path, "r") as reader:
    return next(iter(reader))


def setup_maxtext_model(config, mesh):
  """setup maxtext model"""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  quant = quantizations.configure_quantization(config)

  if config.pure_nnx:
    # NNX has a different function to init the training state.
    raise NotImplementedError("Pure NNX support has not been implemented yet.")
  else:
    maxtext_model = models.transformer_as_linen(config=config, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    init_state_fn = functools.partial(maxtext_utils.init_initial_state, maxtext_model, None, config, False, init_rng)
  state, state_mesh_annotations = maxtext_utils.setup_decode_state(config, mesh, None, init_state_fn)
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
        os.path.join(MAXTEXT_ASSETS_ROOT, ""),
    ]
    exit_code = subprocess.call(command, cwd=os.path.dirname(MAXTEXT_PKG_DIR))
    if exit_code != 0:
      raise ValueError(f"{command} failed with exit code: {exit_code}")
    self.config = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo_trainer_test.yml")],
        run_name="unit_test_grpo_trainer",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "llama3.1-tokenizer"),
        enable_checkpointing=False,
        train_data_columns="prompt",
    )
    self.config_inference = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo_trainer_test.yml")],
        run_name="unit_test_grpo_trainer_inference",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "llama3.1-tokenizer"),
        enable_checkpointing=False,
        ici_tensor_parallelism=4,
        per_device_batch_size=self.config.per_device_batch_size * self.config.rl["num_generations"],
    )
    self.model = mt.from_config(self.config)
    self.inference_model = mt.from_config(self.config_inference)
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
    devices_array = maxtext_utils.create_device_mesh(self.config_inference)
    self.mesh = Mesh(devices_array, self.config_inference.mesh_axes)
    self.tokenizer_model.add_special_tokens({"pad_token": "<pad>"})
    self.inference_engine = offline_engine.OfflineEngine(
        config=self.config_inference,
        mesh=self.inference_model.mesh,
    )

  @pytest.mark.skip(reason="Logit output test fragile, failing on jax upgrade to 0.6.2 - see b/425997645")
  @pytest.mark.integration_test
  @pytest.mark.tpu_only  # ATTENTION: Only run on TPU V4-8
  def test_grpo_trainer_correctness(self):
    # Get the expected (golden) data.
    golden_data = get_golden_data(self.config)
    # Initialize the model and related objects.
    maxtext_model, state, reference_params, rng, _, _ = setup_maxtext_model(self.config, self.mesh)
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

    input_data = []
    for i, d in enumerate(engine_data[self.config.train_data_columns]):
      input_data.append(
          InputData(
              id=f"input_{i}",
              tokens=np.array(d),
              true_length=np.array(data[f"{self.config.train_data_columns}_true_length"][i])[0],
          )
      )

    results = self.inference_engine.batch_inference(input_data)

    # Assert that the generated completions match the golden reference.
    self.assertEqual(results[0]["prompt_completions"].tolist(), golden_data["generated_completions"])


class ReshardingTest(unittest.TestCase):
  """Tests for the resharding functions in max_utils.py"""

  def init_pyconfig(self, config_file, **kwargs):
    """init pyconfig"""
    init_kwargs = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "dataset_type": "hf",
        "hf_path": "trl-lib/tldr",
        "tokenizer_type": "huggingface",
        "tokenizer_path": "google/gemma-2-2b-it",
        "attention": "dot_product",
        "model_name": "gemma2-2b",
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", config_file)],
        **init_kwargs,
    )
    return config

  @pytest.mark.skip(reason="This test only runs on multihost cluster with pathways backend")
  @pytest.mark.tpu_only
  def test_pw_reshard_pytree(self):
    """Test that reshard_pytree correctly reshards a PyTree."""
    # Create a mesh of 16 devices, laid out as 16x1.
    source_config = self.init_pyconfig(
        "grpo.yml", ici_fsdp_parallelism=16, inference_replicas=4, inference_devices_per_replica=4
    )

    # Create a second mesh of 16 devices for inference
    dst_config = self.init_pyconfig("grpo_inference.yml", ici_data_parallelism=4, ici_tensor_parallelism=4)

    _, _, _, dest_sharding, *_, state = setup_train_loop(source_config, dst_config, None)
    pytree = state.params

    # Reshard the PyTree from the source to the destination sharding.
    # Applying transfer guards to prevent implicit transfers through controller.
    with (
        jax.transfer_guard_device_to_host("disallow_explicit"),
        jax.transfer_guard_host_to_device("disallow_explicit"),
    ):
      resharded_pytree = grpo_utils.reshard_pytree(pytree, dest_sharding.params)

    # Check that the output has the correct sharding.
    jax.tree_util.tree_map(
        lambda x, y: self.assertEqual(x.sharding, y),
        resharded_pytree,
        dest_sharding.params,
    )


if __name__ == "__main__":
  unittest.main()
