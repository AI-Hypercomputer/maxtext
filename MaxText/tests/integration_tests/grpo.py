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

import jax
import os
import unittest
from layers import models
from layers import initializers
import jax.numpy as jnp
import numpy as np

import pyconfig
import max_utils
from jax.sharding import Mesh
import flax.linen as nn
from typing import Tuple
import sys
import common_types
import torch
# from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(maxtext_parent_dir)

from experimental.rl.grpo_trainer import compute_log_probs


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer



class GRPOTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
      [None, "experimental/rl/grpo.yml"],
      run_name="grpo_test",
      model_name="llama3.1-8b",
      enable_checkpointing=True,
      # load_parameters_path="gs://maxtext-model-checkpoints/gemma2-2b-it/2025-02-20-18-01/scanned/0/items",
      load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
      max_target_length=32,
      per_device_batch_size=1,
      max_prefill_predict_length=16,
      dataset_type="synthetic",
      dtype="float32",
      matmul_precision="high",
      logits_dot_in_fp32=True,
    )
    self.rng = jax.random.PRNGKey(42)
    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.model = models.Transformer(config=self.cfg, mesh=mesh, quant=None)
    self.state, _ = max_utils.setup_decode_state(self.model, self.cfg, self.rng, mesh, None)
    self.tokenizer_model = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        add_bos_token=self.cfg.add_bos,
        add_eos_token=self.cfg.add_eos,
        model_max_length=self.cfg.max_target_length,
        legacy=False,
        token=self.cfg.hf_access_token,
        padding_side="left",
    )
    self.tokenizer_model.add_special_tokens({'pad_token': '<pad>'})
    self.input_str = "Hello world this is a test"

    # Define the reward function, which rewards completions that are close to 20 characters
    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    training_args = GRPOConfig(
      output_dir="test-grpo-trainer",
      logging_steps=1,
      per_device_train_batch_size=4,
      num_generations=4,
      max_prompt_length=16,
      max_completion_length=16
    )

    self.trainer = GRPOTrainer(
      model="meta-llama/Llama-3.1-8B",
      processing_class=self.tokenizer_model,
      reward_funcs=reward_len,
      args=training_args,
      train_dataset=load_dataset("trl-lib/tldr", split="train"),
    )
    self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-3.1-8B",
      torch_dtype=torch.float32,
    )

  # def test_inference(self):
  #   """ This is to test GRPO input pipeline and generate prompts
  #   """

  # def test_logits(self):
  #   def _prepare_inputs():
  #     input_ids = jnp.tile(jnp.array(self.tokenizer_model.encode(self.input_str)), (4, 1))
  #     input_segmentation =  (input_ids > 0).astype(jnp.int32)
  #     input_position = jnp.tile(jnp.arange(input_ids.shape[1]), (4, 1))

  #     return input_ids, input_segmentation, input_position

  #   inputs, inputs_segmentation, inputs_position = _prepare_inputs()
  #   logits, _ = self.model.apply(
  #     self.state.params,
  #     inputs,
  #     inputs_position,
  #     decoder_segment_ids = inputs_segmentation,
  #     enable_dropout=False,
  #     rngs=self.rng,
  #     mutable="intermediates",
  #   )
  #   logits = np.asarray(logits)
  #   hf_logits = self.hf_model(input_ids = torch.tensor(inputs.tolist()), attention_mask = torch.tensor(inputs_segmentation.tolist())).logits.detach().numpy()
  #   print(f"Max Diff {np.max(np.abs(logits - hf_logits))}")
  #   self.assertTrue(jax.numpy.allclose(hf_logits, logits, rtol=1e-2, atol=2e-1, equal_nan=False))


  # def test_logps(self):
  #   def _prepare_maxtext_inputs():
  #     prompt = self.tokenizer_model.encode(self.input_str)
  #     input_ids = jnp.pad(jnp.tile(jnp.concat([jnp.array(prompt),jnp.array(prompt)], axis = -1), (4, 1)), ((0, 0),(0, 4)), constant_values=self.tokenizer_model.pad_token_type_id) # pad some tokens at the end of input prompt
  #     input_segmentation = (input_ids > 0).astype(jnp.int32)
  #     input_position = jnp.where(input_segmentation, jnp.arange(input_segmentation.shape[1]), 0)
  #     completion_segmentation = jnp.tile(jnp.pad(jnp.array([0] * len(prompt) + [1] * len(prompt)), (0, input_ids.shape[1] - 2 * len(prompt))), (4,1))
  #     return input_ids, input_segmentation, input_position, completion_segmentation

  #   def _prepare_trl_inputs():
  #     prompt = self.tokenizer_model.encode(self.input_str)
  #     logits_to_keep = len(prompt)
  #     prompt = torch.tensor([prompt], dtype=torch.int64)
  #     input_ids = torch.cat([prompt, prompt], axis = -1)
  #     attention_mask = (input_ids > 0).int()
  #     return input_ids, attention_mask, logits_to_keep

  #   hf_input_ids, attention_mask, logits_to_keep = _prepare_trl_inputs()
  #   hf_per_token_logps = self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep) # pylint: disable=protected-access
    
  #   input_ids, input_segmentation, input_position, completion_segmentation = _prepare_maxtext_inputs()
  #   maxtext_per_token_logps, _ = compute_log_probs(self.model, self.state.params, input_ids, input_position, input_segmentation, completion_segmentation, self.cfg, is_train=False)
  #   print(f"Max Diff {np.max(np.abs(np.trim_zeros(np.asarray(maxtext_per_token_logps)[0]) - hf_per_token_logps.detach().numpy()[0]))}")
  #   self.assertTrue(jax.numpy.allclose(np.trim_zeros(np.asarray(maxtext_per_token_logps)[0]), hf_per_token_logps.detach().numpy()[0], rtol=1e-2, atol=1e-2, equal_nan=False))

  def test_loss_kl_div(self):
    def _prepare_trl_inputs():
      return [{"prompt": self.input_str, "completion": self.input_str}]
    inputs = _prepare_trl_inputs()
    
    inputs_for_loss = self.trainer._prepare_inputs(inputs)
    breakpoint()
    
    