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

from collections.abc import Callable
import functools
import os
import unittest

from datasets import load_dataset
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import jsonlines
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR, MAXTEXT_TEST_ASSETS_ROOT
from maxtext.common.common_types import Array
from maxtext.experimental.rl.grpo_trainer import generate_completions, grpo_loss_fn_nnx
from maxtext.experimental.rl.grpo_utils import compute_log_probs_nnx
from maxtext.inference.maxengine import maxengine
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.post_training.integration.grpo_trainer_correctness_test import prepare_maxtext_inputs
import numpy as np
import torch
import transformers
# from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def _setup_model(config, mesh, rng):
  """Builds the model and a frozen reference clone.

  Returns (model, reference_model). The model carries its own params (from_pretrained
  loads the checkpoint or inits) and the reference is a clone of the policy.
  """
  model = model_creation_utils.from_pretrained(config, mesh=mesh, rng_key=rng)
  return model, nnx.clone(model)


def _logps(config, model, ids, pos, seg, comp_seg):
  """Policy per-token log-probs."""
  return compute_log_probs_nnx(model, ids, pos, seg, comp_seg, config, is_train=False)


def _reference_logps(config, reference_model, ids, pos, seg, comp_seg):
  """Reference per-token log-probs, using the cloned reference model."""
  return compute_log_probs_nnx(reference_model, ids, pos, seg, comp_seg, config, is_train=False)


def _grpo_loss(config, model, reference_model, data, rng):
  """GRPO loss, using the cloned reference model."""
  return grpo_loss_fn_nnx(model, config, data, rng, None, reference_model)


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
    self.mesh = mesh
    # With checkpoint
    self.model, self.reference_model = _setup_model(self.cfg, mesh, self.rng)
    self.data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
    # Without checkpoint
    self.model_no_ckpt_loading, self.reference_model_no_ckpt_loading = _setup_model(
        self.cfg_no_ckpt_loading, mesh, self.rng
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
    maxtext_per_token_logps, _ = _logps(
        self.cfg, self.model, input_ids, input_position, input_segmentation, completion_segmentation
    )

    data = {
        "prompt_completions": input_ids,
        "prompt_completions_position": input_position,
        "prompt_completions_segmentation": input_segmentation,
        "ar_completions_segmentation": completion_segmentation,
    }
    maxtext_loss, aux = _grpo_loss(self.cfg, self.model, self.reference_model, data, self.rng)
    # pylint: disable=protected-access
    self.assertEqual(self.trainer._metrics["train"]["kl"][0], aux.avg_kl.tolist())
    self.assertEqual(hf_loss.item(), maxtext_loss.tolist())
    # since this is on-policy
    self.assertEqual(aux.avg_advantage.tolist(), 0.0)
    # since we are at step 0
    maxtext_per_token_logps, _ = _logps(
        self.cfg, self.model, input_ids, input_position, input_segmentation, completion_segmentation
    )
    maxtext_per_token_logps_ref, _ = _reference_logps(
        self.cfg,
        self.reference_model,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
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

    maxtext_per_token_logps_no_ckpt_loading, _ = _logps(
        self.cfg_no_ckpt_loading,
        self.model_no_ckpt_loading,
        input_ids,
        input_position,
        input_segmentation,
        completion_segmentation,
    )

    maxtext_loss, aux = _grpo_loss(
        self.cfg_no_ckpt_loading,
        self.model_no_ckpt_loading,
        self.reference_model_no_ckpt_loading,
        data,
        self.rng,
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
    # Params live on the model; the inference engine reads them directly.
    gen_params = nnx.state(self.model_no_ckpt_loading, nnx.Param)
    gen_param_shardings = jax.tree.map(lambda _: jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec()), gen_params)
    p_generate_completions: Callable[[dict, dict, Array], Array] = jax.jit(
        functools.partial(generate_completions, self.cfg, self.tokenizer_model, engine),
        in_shardings=(self.data_sharding, gen_param_shardings, None),
        out_shardings=self.data_sharding,
        donate_argnums=(0,),
    )
    # pylint: disable=not-callable
    engine_data = p_generate_completions(engine_data, gen_params, self.rng)
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
