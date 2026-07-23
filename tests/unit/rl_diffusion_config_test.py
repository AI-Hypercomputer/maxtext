# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration tests for default-off block-diffusion RL."""

from absl.testing import absltest
from absl.testing import parameterized

from maxtext.configs import types


def _diffusion_rl_config(**overrides):
  """Builds a valid default diffusion-RL configuration for tests."""
  values = {
      "training_objective": "block_diffusion",
      "attention_type": "block_diffusion",
      "attention": "dot_product",
      "hardware": "cpu",
      "packing": False,
      "block_diffusion_mask_id": 31,
      "vocab_size": 32,
      "decode_sampling_top_k": -1,
      "decode_sampling_nucleus_p": 1.0,
      "eval_sampling_strategy": "greedy",
      "generation_configs": {"greedy": {"eval_temperature": 0.01, "eval_top_k": 1, "eval_top_p": 1.0}},
      "rl": {"diffusion_rollout": True},
  }
  values.update(overrides)
  return types.RLConfig(**values)


class RLDiffusionConfigTest(parameterized.TestCase):

  def test_ar_defaults_are_unchanged(self):
    config = types.RLConfig()

    self.assertFalse(config.rl.diffusion_rollout)
    self.assertEqual(config.training_objective, "causal_lm")

  def test_accepts_explicit_denoising_trace_contract(self):
    config = _diffusion_rl_config()

    self.assertTrue(config.rl.diffusion_rollout)
    self.assertEqual(config.rl.diffusion_score_mode, "denoising_trace")

  def test_accepts_qwen_model_attention_fields(self):
    config = types.RLConfig(use_qk_norm=True)

    self.assertTrue(config.use_qk_norm)
    self.assertFalse(config.use_mrope)
    self.assertGreater(config.emb_dim, 0)
    self.assertGreater(config.num_decoder_layers, 0)

  def test_rejects_block_diffusion_objective_without_diffusion_rollout(self):
    with self.assertRaisesRegex(ValueError, "diffusion_rollout=True"):
      types.RLConfig(
          training_objective="block_diffusion",
          attention_type="block_diffusion",
          packing=False,
          dataset_type="hf",
          hf_path="parquet",
          block_diffusion_mask_id=31,
          vocab_size=32,
      )

  @parameterized.named_parameters(
      ("causal_objective", {"training_objective": "causal_lm"}, "training_objective='block_diffusion'"),
      ("top_k", {"decode_sampling_top_k": 8}, "decode_sampling_top_k=-1"),
      ("top_p", {"decode_sampling_nucleus_p": 0.9}, "decode_sampling_nucleus_p=1.0"),
      ("short_denoise", {"rl": {"diffusion_rollout": True, "diffusion_max_denoise_steps": 3}}, "at least"),
      ("duplicate_stop", {"rl": {"diffusion_rollout": True, "diffusion_stop_token_ids": [5, 5]}}, "unique IDs"),
      ("mask_is_stop", {"rl": {"diffusion_rollout": True, "diffusion_stop_token_ids": [31]}}, "mask token"),
      ("agentic", {"rl": {"diffusion_rollout": True, "use_agentic_rollout": True}}, "agentic"),
      ("multi_controller", {"cluster": {"use_pathways": False}}, "Pathways"),
      ("stop_strings", {"stop_strings": ["stop"]}, "token stop IDs"),
      (
          "filtered_eval",
          {
              "eval_sampling_strategy": "standard",
              "generation_configs": {"standard": {"eval_temperature": 0.7, "eval_top_k": 50, "eval_top_p": 0.95}},
          },
          "evaluation supports",
      ),
  )
  def test_rejects_incompatible_modes(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      _diffusion_rl_config(**overrides)


if __name__ == "__main__":
  absltest.main()
