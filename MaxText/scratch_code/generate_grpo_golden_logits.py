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

"""Script to get golden data for GRPOTrainer in TRL. Currently hardcoded for llama3.1-8b

Usage:

python3 -m MaxText.scratch_code.generate_grpo_golden_logits
"""

import argparse
import jax
import os
import unittest

import jsonlines
from MaxText.layers import models
from MaxText.layers import initializers
import jax.numpy as jnp
import numpy as np

import MaxText.pyconfig as pyconfig
import MaxText.max_utils as max_utils
from jax.sharding import Mesh
import flax.linen as nn
from typing import Tuple
import sys
import MaxText.common_types as common_types
import torch
# from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

from datasets import load_dataset

from MaxText.experimental.rl.grpo_trainer import compute_log_probs, grpo_loss_fn, _merge_grpo_state, generate_completions
from MaxText.tests.grpo_trainer_correctness_test import _prepare_maxtext_inputs
from MaxText import maxengine
from MaxText.globals import PKG_DIR

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer


class GRPOTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, "MaxText/experimental/rl/grpo.yml"],
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
    self.cfg_no_ckpt_loading = pyconfig.initialize(
        [None, "MaxText/experimental/rl/grpo.yml"],
        run_name="grpo_test",
        model_name="llama3.1-8b",
        enable_checkpointing=True,
        # load_parameters_path="gs://maxtext-model-checkpoints/gemma2-2b-it/2025-02-20-18-01/scanned/0/items",
        # load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
        max_target_length=32,
        per_device_batch_size=1,
        max_prefill_predict_length=16,
        dataset_type="synthetic",
        dtype="float32",
        matmul_precision="high",
        logits_dot_in_fp32=True,
        base_num_decoder_layers=8,
        base_emb_dim=1024,
    )
    self.cfg_no_ckpt_loading_inference = pyconfig.initialize(
        [None, "MaxText/experimental/rl/grpo.yml"],
        run_name="grpo_test",
        model_name="llama3.1-8b",
        enable_checkpointing=True,
        # load_parameters_path="gs://maxtext-model-checkpoints/gemma2-2b-it/2025-02-20-18-01/scanned/0/items",
        # load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
        max_target_length=32,
        max_prefill_predict_length=16,
        dataset_type="synthetic",
        dtype="float32",
        matmul_precision="high",
        logits_dot_in_fp32=True,
        ici_tensor_parallelism=4,
        per_device_batch_size=self.cfg_no_ckpt_loading.per_device_batch_size * self.cfg_no_ckpt_loading.num_generations,
    )
    self.rng = jax.random.PRNGKey(42)
    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    # With checkpoint
    self.model = models.Transformer(config=self.cfg, mesh=mesh, quant=None)
    self.state, _ = max_utils.setup_decode_state(self.model, self.cfg, self.rng, mesh, None)
    # Without checkpoint
    self.model_no_ckpt_loading = models.Transformer(config=self.cfg_no_ckpt_loading, mesh=mesh, quant=None)
    self.state_no_ckpt_loading, _ = max_utils.setup_decode_state(
        self.model_no_ckpt_loading, self.cfg_no_ckpt_loading, self.rng, mesh, None
    )

    self.tokenizer_model = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        add_bos_token=self.cfg.add_bos,
        add_eos_token=self.cfg.add_eos,
        model_max_length=self.cfg.max_target_length,
        legacy=False,
        token=self.cfg.hf_access_token,
        padding_side="left",
    )
    self.tokenizer_model.add_special_tokens({"pad_token": "<pad>"})
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
        max_completion_length=16,
        temperature=0.0,
    )
    self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float32,
    )
    self.trainer = GRPOTrainer(
        model=self.hf_model,
        processing_class=self.tokenizer_model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=load_dataset("trl-lib/tldr", split="train"),
    )

  def _prepare_trl_inputs(self):
    tokenized_inputs = self.tokenizer_model([self.input_str], return_tensors="pt")
    input_ids = torch.cat((tokenized_inputs["input_ids"], tokenized_inputs["input_ids"]), axis=-1)
    attention_mask = torch.cat((tokenized_inputs["attention_mask"], tokenized_inputs["attention_mask"]), axis=-1)
    logits_to_keep = tokenized_inputs["input_ids"].size()[1]
    return input_ids, attention_mask, logits_to_keep

  def test_w_trl_and_write_golden_data(self):
    # def _prepare_trl_inputs():
    #   return [{"prompt": self.input_str, "completion": self.input_str}]
    # inputs = _prepare_trl_inputs()
    hf_input_ids, attention_mask, logits_to_keep = self._prepare_trl_inputs()
    # hf_inputs = self.trainer._prepare_inputs([{'prompt':self.input_str}]*4)

    completions = [{"prompt": self.input_str}] * 4
    rewards = torch.tensor([self.trainer.reward_funcs[0](completion) for completion in completions], dtype=torch.float32)
    # Compute grouped-wise rewards
    mean_grouped_rewards = rewards.view(-1, self.trainer.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, self.trainer.num_generations).std(dim=1)

    # Normalize the rewards to compute the advantages
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.trainer.num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.trainer.num_generations, dim=0)
    # since we are using the same completion, so advantages = 0 for every sequence
    # but we can keep it this way since our on-policy implementation
    # gets average advantage which becomes zero anyway
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    inputs = {
        "prompt_ids": hf_input_ids[:, :logits_to_keep],
        "prompt_mask": attention_mask[:, :logits_to_keep],
        "completion_ids": hf_input_ids[:, logits_to_keep:],
        "completion_mask": attention_mask[:, logits_to_keep:],
        # using the same model as the ref model,
        # which is equivalent of step 0 of GRPO training when
        # the on-policy params are the same as the ref model
        "ref_per_token_logps": self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep),  # pylint: disable=protected-access
        # using only one advantage because we have just one sequence
        "advantages": advantages[0][0].unsqueeze(0),
    }
    hf_loss = self.trainer.compute_loss(self.hf_model, inputs)

    hf_per_token_logps = self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep)  # pylint: disable=protected-access

    input_ids, input_segmentation, input_position, completion_segmentation = self._prepare_maxtext_inputs()
    maxtext_per_token_logps, _ = compute_log_probs(
        self.model,
        self.state.params,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
        self.cfg,
        is_train=False,
    )

    reference_params = jax.tree.map(jnp.copy, self.state.params["params"])
    self.state = _merge_grpo_state(self.state, reference_params)

    reference_params_no_ckpt_loading = jax.tree.map(jnp.copy, self.state_no_ckpt_loading.params["params"])
    self.state_no_ckpt_loading = _merge_grpo_state(self.state_no_ckpt_loading, reference_params_no_ckpt_loading)

    data = {
        "prompt_completions": input_ids,
        "prompt_completions_position": input_position,
        "prompt_completions_segmentation": input_segmentation,
        "ar_completions_segmentation": completion_segmentation,
    }
    maxtext_loss, aux = grpo_loss_fn(self.model, self.cfg, data, self.rng, self.state.params, reference_params)
    self.assertEqual(self.trainer._metrics["train"]["kl"][0], aux["avg_kl"].tolist())
    self.assertEqual(hf_loss.item(), maxtext_loss.tolist())
    # since this is on-policy
    self.assertEqual(aux["avg_advantage"].tolist(), 0.0)
    # since we are at step 0
    maxtext_per_token_logps, _ = compute_log_probs(
        self.model,
        self.state.params,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
        self.cfg,
        is_train=False,
    )
    maxtext_per_token_logps_ref, _ = compute_log_probs(
        self.model,
        {"params": reference_params},
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
        self.cfg,
        is_train=False,
    )
    self.assertTrue(
        jax.numpy.allclose(
            np.trim_zeros(np.asarray(maxtext_per_token_logps)[0]),
            np.trim_zeros(np.asarray(maxtext_per_token_logps_ref)[0]),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=False,
        )
    )
    # Now that we have ensured that the MaxText implementation is correct
    # let us create a MaxText model without the checkpoint and save the logits

    maxtext_per_token_logps_no_ckpt_loading, _ = compute_log_probs(
        self.model_no_ckpt_loading,
        self.state_no_ckpt_loading.params,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
        self.cfg_no_ckpt_loading,
        is_train=False,
    )

    # Convert logits to fp32
    maxtext_per_token_logps_no_ckpt_loading = maxtext_per_token_logps_no_ckpt_loading.cpu().numpy().astype("float32")

    maxtext_loss, aux = grpo_loss_fn(
        self.model_no_ckpt_loading,
        self.cfg_no_ckpt_loading,
        data,
        self.rng,
        self.state_no_ckpt_loading.params,
        reference_params_no_ckpt_loading,
    )

    engine = maxengine.MaxEngine(self.config_inference)
    generated_completions = generate_completions(
        self.cfg_no_ckpt_loading,
        self.tokenizer_model,
        engine,
        self.tokenizer_model.encode(self.input_str),
        self.state_no_ckpt_loading.params["params"],
        self.rng,
    )

    data_to_save = {
        "maxtext_loss": maxtext_loss.tolist(),
        "generated_completions": generated_completions,
        "maxtext_per_token_logps_no_ckpt_loading": maxtext_per_token_logps_no_ckpt_loading.tolist()[
            0
        ],  # Convert numpy array to list for JSON serialization
        "avg_kl": aux["avg_kl"].tolist(),
    }
    model_output_path = os.path.join(
        os.path.dirname(PKG_DIR), "MaxText", "test_assets", f"golden_data_grpo_{self.cfg.model_name}.jsonl"
    )
    with jsonlines.open(model_output_path, "w") as f:
      f.write(data_to_save)


if __name__ == "__main__":
  unittest.main()
