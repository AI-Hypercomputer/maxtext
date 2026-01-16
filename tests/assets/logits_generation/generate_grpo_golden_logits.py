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

"""Script to get golden data for GRPOTrainer in TRL. Currently hardcoded for llama3.1-8b

Usage:

python3 -m tests.assets.logits_generation.generate_grpo_golden_logits
"""

import functools
import os
import unittest
from collections.abc import Callable

import jsonlines

import numpy as np

import jax.numpy as jnp
import jax
from jax.sharding import Mesh

from flax import linen as nn

import torch

# from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

import transformers

from datasets import load_dataset

from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import Array, MODEL_MODE_TRAIN
from MaxText.experimental.rl.grpo_trainer import grpo_loss_fn, _merge_grpo_state, generate_completions
from MaxText.experimental.rl.grpo_utils import compute_log_probs
from MaxText.globals import MAXTEXT_TEST_ASSETS_ROOT, MAXTEXT_PKG_DIR
from MaxText.layers import models

from tests.grpo_trainer_correctness_test import prepare_maxtext_inputs


class GRPOTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo_trainer_test.yml")],
        model_name="llama3.1-8b",
        run_name="generate_grpo_test_data",
        load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
        enable_checkpointing=True,
    )
    self.cfg_no_ckpt_loading = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo_trainer_test.yml")],
        run_name="generate_grpo_test_data_no_ckpt_loading",
        enable_checkpointing=False,
    )
    self.cfg_no_ckpt_loading_inference = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "experimental", "rl", "grpo_trainer_test.yml")],
        run_name="generate_grpo_test_data_no_ckpt_loading_inference",
        enable_checkpointing=False,
        ici_tensor_parallelism=4,
        per_device_batch_size=self.cfg_no_ckpt_loading.per_device_batch_size * self.cfg_no_ckpt_loading.num_generations,
    )
    self.rng = jax.random.key(self.cfg.init_weights_seed)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    # With checkpoint
    self.model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
    self.state, state_mesh_annotations = maxtext_utils.setup_decode_state(self.model, self.cfg, self.rng, mesh, None)
    self.state_mesh_shardings = nn.logical_to_mesh_sharding(state_mesh_annotations, mesh, self.cfg.logical_axis_rules)
    self.data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
    # Without checkpoint
    self.model_no_ckpt_loading = models.transformer_as_linen(
        config=self.cfg_no_ckpt_loading, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN
    )
    self.state_no_ckpt_loading, _ = maxtext_utils.setup_decode_state(
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
    tokenized_inputs = self.tokenizer_model([self.cfg.prompt], return_tensors="pt")
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

    completions = [{"prompt": self.cfg.prompt}] * 4
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
        # pylint: disable=protected-access
        "ref_per_token_logps": self.trainer._get_per_token_logps(
            self.hf_model, hf_input_ids, attention_mask, logits_to_keep
        ),  # pylint: disable=protected-access
        # using only one advantage because we have just one sequence
        "advantages": advantages[0][0].unsqueeze(0),
    }
    hf_loss = self.trainer.compute_loss(self.hf_model, inputs)

    self.trainer._get_per_token_logps(self.hf_model, hf_input_ids, attention_mask, logits_to_keep)  # pylint: disable=protected-access

    input_ids, input_segmentation, input_position, completion_segmentation = prepare_maxtext_inputs(
        self.cfg.prompt, self.tokenizer_model
    )
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
    # pylint: disable=protected-access
    self.assertEqual(self.trainer._metrics["train"]["kl"][0], aux.avg_kl.tolist())
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
        rngs=self.rng,
    )

    maxtext_loss, aux = grpo_loss_fn(
        self.model_no_ckpt_loading,
        self.cfg_no_ckpt_loading,
        data,
        self.rng,
        self.state_no_ckpt_loading.params,
        reference_params_no_ckpt_loading,
    )

    engine = maxengine.MaxEngine(self.cfg_no_ckpt_loading_inference)
    _ = engine.load_params(self.rng)
    prompt_tokens = self.tokenizer_model.encode(self.cfg_no_ckpt_loading_inference.prompt)
    prompt = jnp.pad(
        jnp.tile(jnp.array(prompt_tokens), (4, 1)),
        ((0, 0), (0, 4)),
        constant_values=self.tokenizer_model.pad_token_type_id,
    )
    prompt_true_length = jnp.array([len(prompt_tokens)] * 4)
    engine_data = {"prompt": prompt, "prompt_true_length": prompt_true_length}
    p_generate_completions: Callable[[dict, dict, Array], Array] = jax.jit(
        functools.partial(generate_completions, self.cfg, self.tokenizer_model, engine),
        in_shardings=(self.data_sharding, self.state_mesh_shardings.params, None),
        out_shardings=self.data_sharding,
        donate_argnums=(0,),
    )
    # pylint: disable=not-callable
    engine_data = p_generate_completions(engine_data, {"params": self.state_no_ckpt_loading.params["params"]}, self.rng)
    data_to_save = {
        "maxtext_loss": maxtext_loss.tolist(),
        "input_ids": input_ids[0].tolist(),
        "generated_completions": engine_data["prompt_completions"][0].tolist(),
        "maxtext_per_token_logps_no_ckpt_loading": maxtext_per_token_logps_no_ckpt_loading.tolist()[
            0
        ],  # Convert numpy array to list for JSON serialization
        "avg_kl": aux.avg_kl.tolist(),
        "avg_advantage": aux.avg_advantage.tolist(),
    }
    model_output_path = os.path.join(
        MAXTEXT_TEST_ASSETS_ROOT,
        f"golden_data_grpo_{self.cfg_no_ckpt_loading.model_name}.jsonl",
    )
    with jsonlines.open(model_output_path, "w") as f:
      f.write(data_to_save)


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  unittest.main()
