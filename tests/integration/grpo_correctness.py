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
"""GRPO correctness tests"""

import os
import unittest

from datasets import load_dataset
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.experimental.rl.grpo_trainer import _merge_grpo_state, grpo_loss_fn
from maxtext.experimental.rl.grpo_utils import compute_log_probs
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.models import models
from maxtext.utils import maxtext_utils
import numpy as np
import pytest
import torch
import transformers
# from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer

pytestmark = [pytest.mark.external_training]  # uses pre-generated checkpoint


class GRPOTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo.yml")],
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
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
    self.state, _ = maxtext_utils.setup_decode_state(self.model, self.cfg, self.rng, mesh, None)
    self.tokenizer_model = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        add_bos_token=False,
        add_eos_token=False,
        token=self.cfg.hf_access_token,
    )
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

  def _prepare_maxtext_inputs(self):
    """prepare maxtext inputs"""
    prompt = self.tokenizer_model.encode(self.input_str)
    input_ids = jnp.pad(
        jnp.tile(jnp.concat([jnp.array(prompt), jnp.array(prompt)], axis=-1), (4, 1)),
        ((0, 0), (0, 4)),
        constant_values=0,
    )  # pad some tokens at the end of input prompt
    input_segmentation = (input_ids > 0).astype(jnp.int32)
    input_position = jnp.where(input_segmentation, jnp.arange(input_segmentation.shape[1]), 0)
    completion_segmentation = jnp.tile(
        jnp.pad(
            jnp.array([0] * len(prompt) + [1] * len(prompt)),
            (0, input_ids.shape[1] - 2 * len(prompt)),
        ),
        (4, 1),
    )
    return (
        input_ids,
        input_segmentation,
        input_position,
        completion_segmentation,
    )

  def _prepare_trl_inputs(self):
    """Prepare TRL inputs."""
    tokenized_inputs = self.tokenizer_model([self.input_str], return_tensors="pt")
    input_ids = torch.cat((tokenized_inputs["input_ids"], tokenized_inputs["input_ids"]), axis=-1)
    attention_mask = torch.cat(
        (
            tokenized_inputs["attention_mask"],
            tokenized_inputs["attention_mask"],
        ),
        axis=-1,
    )
    logits_to_keep = tokenized_inputs["input_ids"].size()[1]
    return input_ids, attention_mask, logits_to_keep

  def test_logits(self):
    def _prepare_inputs():
      input_ids = jnp.tile(jnp.array(self.tokenizer_model.encode(self.input_str)), (4, 1))
      input_segmentation = (input_ids > 0).astype(jnp.int32)
      input_position = jnp.tile(jnp.arange(input_ids.shape[1]), (4, 1))

      return input_ids, input_segmentation, input_position

    inputs, inputs_segmentation, inputs_position = _prepare_inputs()
    logits, _ = self.model.apply(
        self.state.params,
        inputs,
        inputs_position,
        decoder_segment_ids=inputs_segmentation,
        enable_dropout=False,
        rngs=self.rng,
        mutable="intermediates",
    )

    hf_logits = (
        self.hf_model(
            input_ids=torch.tensor(inputs.tolist()),
            attention_mask=torch.tensor(inputs_segmentation.tolist()),
        )
        .logits.detach()
        .numpy()
    )
    print(f"Max Diff {np.max(np.abs(logits - hf_logits))}")
    self.assertTrue(jax.numpy.allclose(hf_logits, logits, rtol=1e-2, atol=2e-1, equal_nan=False))

  def test_logps(self):

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
    hf_input_ids, attention_mask, logits_to_keep = self._prepare_trl_inputs()
    with torch.no_grad():
      hf_per_token_logps = self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep)  # pylint: disable=protected-access

    print(
        "Max Diff",
        np.max(np.abs(np.trim_zeros(np.asarray(maxtext_per_token_logps)[0]) - hf_per_token_logps.detach().numpy()[0])),
    )
    self.assertTrue(
        jax.numpy.allclose(
            np.trim_zeros(np.asarray(maxtext_per_token_logps)[0]),
            hf_per_token_logps.detach().numpy()[0],
            rtol=1e-2,
            atol=1e-2,
            equal_nan=False,
        )
    )

  def test_loss_kl_div(self):
    # def _prepare_trl_inputs():
    #   return [{"prompt": self.input_str, "completion": self.input_str}]
    # inputs = _prepare_trl_inputs()
    hf_input_ids, attention_mask, logits_to_keep = self._prepare_trl_inputs()
    # hf_inputs = self.trainer._prepare_inputs([{'prompt':self.input_str}]*4)

    completions = [{"prompt": self.input_str}] * 4
    rewards = torch.tensor(
        [self.trainer.reward_funcs[0](completion) for completion in completions],
        dtype=torch.float32,
    )
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
        # pylint: disable=protected-access
        "ref_per_token_logps": self.trainer._get_per_token_logps(
            self.hf_model, hf_input_ids, attention_mask, logits_to_keep
        ),
        # using only one advantage because we have just one sequence
        "advantages": advantages[0][0].unsqueeze(0),
    }
    hf_loss = self.trainer.compute_loss(self.hf_model, inputs)

    self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep)  # pylint: disable=protected-access

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
    data = {
        "prompt_completions": input_ids,
        "prompt_completions_position": input_position,
        "prompt_completions_segmentation": input_segmentation,
        "ar_completions_segmentation": completion_segmentation,
    }
    maxtext_loss, aux = grpo_loss_fn(
        self.model,
        self.cfg,
        data,
        self.rng,
        self.state.params,
        reference_params,
    )
    self.assertEqual(self.trainer._metrics["train"]["kl"][0], aux.avg_kl.tolist())  # pylint: disable=protected-access
    self.assertEqual(hf_loss.item(), maxtext_loss.tolist())
    # since this is on-policy
    self.assertEqual(aux.avg_advantage.tolist(), 0.0)
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
