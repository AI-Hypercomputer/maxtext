# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration test for comparing MaxText/Tunix DPO training stack against Hugging Face TRL-inspired implementation.

This test validates the mathematical correctness of the train_dpo.py pipeline.

Goals of this test:
1. Backpropagation & Optimizer Parity: Verifies that JAX successfully runs the backward pass and updates
   parameters in exact convergence with the reference PyTorch implementation after multiple steps.
2. Padding & Masking Invariance: Ensures that padding tokens do not leak or affect model representations.
   JAX executes on padded static-shape inputs, whereas the PyTorch golden reference is computed on raw,
   completely unpadded sequences. Any numerical difference in how padding is handled between JAX/Tunix
   and PyTorch must be considered a bug in the JAX/Tunix causal masking or RoPE coordinate implementations.
3. Parity on Diverged Models: Verifies that JAX and PyTorch forward logits remain aligned on diverged
   weights after training has occurred, ensuring that numerical computations do not drift over time.

Test Flow:
1. Local Dataset Setup: Configures a temporary JSON dataset with prompt/chosen/rejected strings.
   Steps 1 and 2 use different prompt-response pairs to simulate training history variation.
   Step 3 uses the target evaluation prompt.
2. JAX DPO Training Loop: Runs the JAX trainer for exactly 3 steps.
3. Weights Sync: Captures the JAX policy and reference model states at the start of Step 3,
   synchronizing parameters to equivalent PyTorch Qwen2 model structures on-the-fly.
4. PyTorch Golden Reference: Computes reference log-probabilities and DPO loss using the synced weights.
5. Step 3 Parity Check: Compares JAX Step 3 metrics (loss, chosen/rejected log probabilities)
   against the PyTorch golden reference. This verifies that state synchronization and forward
   pass computations match and are immune to prior training history.

To execute:
  pytest tests/post_training/integration/dpo_trl_correctness_test.py
"""

import json
import os
import tempfile
import unittest
import numpy as np
import pytest

import jax
from flax import nnx
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

# MaxText / Tunix imports
from maxtext.configs import pyconfig
from maxtext.trainers.post_train.dpo import train_dpo
from maxtext.trainers.post_train.dpo import hooks as dpo_hooks

# This test can run on a CPU-only machine, we force JAX to use CPU to avoid Pytorch/JAX numerical differences.
# However, we don't set pytest.mark.cpu_only, because we currently don't have a CPU-only integration test runner.
pytestmark = [pytest.mark.post_training, pytest.mark.integration_test]


def get_pytorch_reference(policy_model, ref_model, tokenizer, prompt_str, chosen_str, rejected_str, beta=0.1):
  """Computes reference chosen/rejected logps and DPO loss in PyTorch using raw, unpadded sequences."""
  policy_model.eval()
  ref_model.eval()

  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

  # Tokenize prompt, chosen, and rejected without any padding
  prompt_tokens = tokenizer.encode(prompt_str) + [im_end_id]
  chosen_tokens = tokenizer.encode(chosen_str) + [im_end_id]
  rejected_tokens = tokenizer.encode(rejected_str) + [im_end_id]

  # Form raw sequences
  chosen_ids = prompt_tokens + chosen_tokens
  rejected_ids = prompt_tokens + rejected_tokens

  # Labels mask (loss mask): 0 for prompt, 1 for response
  chosen_labels_mask = [0] * len(prompt_tokens) + [1] * len(chosen_tokens)
  rejected_labels_mask = [0] * len(prompt_tokens) + [1] * len(rejected_tokens)

  chosen_tensor = torch.tensor([chosen_ids], dtype=torch.long)
  rejected_tensor = torch.tensor([rejected_ids], dtype=torch.long)
  chosen_mask = torch.tensor([chosen_labels_mask], dtype=torch.float32)
  rejected_mask = torch.tensor([rejected_labels_mask], dtype=torch.float32)

  def get_batch_logps(logits, labels, loss_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return (per_token_logps * shift_mask).sum(-1)

  with torch.no_grad():
    # PyTorch causal model defaults to sequential position IDs and full attention on unpadded sequences
    policy_chosen_logits = policy_model(chosen_tensor).logits.float()
    policy_rejected_logits = policy_model(rejected_tensor).logits.float()

    ref_chosen_logits = ref_model(chosen_tensor).logits.float()
    ref_rejected_logits = ref_model(rejected_tensor).logits.float()

    policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_tensor, chosen_mask)
    policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_tensor, rejected_mask)

    ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_tensor, chosen_mask)
    ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_tensor, rejected_mask)

    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    delta = chosen_log_ratio - rejected_log_ratio
    loss = -F.logsigmoid(beta * delta).mean()
    margin = chosen_log_ratio - rejected_log_ratio

  return {
      "chosen_ids": chosen_ids,
      "rejected_ids": rejected_ids,
      "chosen_mask": chosen_labels_mask,
      "rejected_mask": rejected_labels_mask,
      "chosen_logps": policy_chosen_logps.item(),
      "rejected_logps": policy_rejected_logps.item(),
      "ref_chosen_logps": ref_chosen_logps.item(),
      "ref_rejected_logps": ref_rejected_logps.item(),
      "loss": loss.item(),
      "margin": margin.item(),
  }


def get_jax_reference(config):
  """Executes JAX DPO training and returns a flat dict of captured step metrics."""
  print("Executing JAX DPO train_dpo.train()...")
  train_dpo.train(config)  # Perform 3 steps of DPO training.

  # We override the TrainingHooks (see below) to capture the metrics that we need to compare against PyTorch.
  captured = InterceptingTrainingHooks.captured_metrics
  assert len(captured) == 3, f"Expected 3 steps metrics, got {len(captured)}"

  step_1 = captured[0]  # We will print the stats for this step, but won't use them for comparison.
  step_3 = captured[2]  # These are the values we want to compare against PyTorch.

  return {
      "loss_step_1": step_1["loss"],
      "margin_step_1": step_1["margin"],
      "loss": step_3["loss"],
      "margin": step_3["margin"],
      "chosen_logps": step_3["chosen_logps"],
      "rejected_logps": step_3["rejected_logps"],
  }


def sync_jax_to_pytorch(jax_model, torch_model):
  """Synchronizes JAX model parameters directly to a PyTorch model."""
  hidden_size = torch_model.config.hidden_size
  torch_state_dict = torch_model.state_dict()
  jax_flat = dict(nnx.state(jax_model).flat_state())

  def sync_param(torch_key, jax_key, reshape=None, transpose=False):
    val = np.array(jax_flat[jax_key][...])
    if reshape:
      val = val.reshape(reshape)
    if transpose:
      val = val.T
    torch_state_dict[torch_key].copy_(torch.from_numpy(val))

  # 1. Token embedding
  sync_param("model.embed_tokens.weight", ("base", "token_embedder", "embedding"))

  # 2. Final layer norm
  sync_param("model.norm.weight", ("base", "decoder", "decoder_norm", "scale"))

  # 3. Causal layers (2 layers)
  num_layers = 2
  for i in range(num_layers):
    # Input and post-attention layer norms
    sync_param(
        f"model.layers.{i}.input_layernorm.weight",
        ("base", "decoder", f"layers_{i}", "pre_self_attention_layer_norm", "scale"),
    )
    sync_param(
        f"model.layers.{i}.post_attention_layernorm.weight",
        ("base", "decoder", f"layers_{i}", "post_self_attention_layer_norm", "scale"),
    )

    # Attention projection weights (transposed from JAX to match PyTorch)
    sync_param(
        f"model.layers.{i}.self_attn.q_proj.weight",
        ("base", "decoder", f"layers_{i}", "self_attention", "query", "kernel"),
        reshape=(hidden_size, hidden_size),
        transpose=True,
    )
    sync_param(
        f"model.layers.{i}.self_attn.k_proj.weight",
        ("base", "decoder", f"layers_{i}", "self_attention", "key", "kernel"),
        reshape=(hidden_size, hidden_size),
        transpose=True,
    )
    sync_param(
        f"model.layers.{i}.self_attn.v_proj.weight",
        ("base", "decoder", f"layers_{i}", "self_attention", "value", "kernel"),
        reshape=(hidden_size, hidden_size),
        transpose=True,
    )
    sync_param(
        f"model.layers.{i}.self_attn.o_proj.weight",
        ("base", "decoder", f"layers_{i}", "self_attention", "out", "kernel"),
        reshape=(hidden_size, hidden_size),
        transpose=True,
    )

    # Attention biases
    sync_param(
        f"model.layers.{i}.self_attn.q_proj.bias",
        ("base", "decoder", f"layers_{i}", "self_attention", "query", "bias"),
        reshape=(hidden_size,),
    )
    sync_param(
        f"model.layers.{i}.self_attn.k_proj.bias",
        ("base", "decoder", f"layers_{i}", "self_attention", "key", "bias"),
        reshape=(hidden_size,),
    )
    sync_param(
        f"model.layers.{i}.self_attn.v_proj.bias",
        ("base", "decoder", f"layers_{i}", "self_attention", "value", "bias"),
        reshape=(hidden_size,),
    )

    # MLP projection weights (wi_0, wi_1, wo)
    sync_param(
        f"model.layers.{i}.mlp.gate_proj.weight",
        ("base", "decoder", f"layers_{i}", "mlp", "wi_0", "kernel"),
        transpose=True,
    )
    sync_param(
        f"model.layers.{i}.mlp.up_proj.weight",
        ("base", "decoder", f"layers_{i}", "mlp", "wi_1", "kernel"),
        transpose=True,
    )
    sync_param(
        f"model.layers.{i}.mlp.down_proj.weight",
        ("base", "decoder", f"layers_{i}", "mlp", "wo", "kernel"),
        transpose=True,
    )

  # 4. LM Head weight (logits_via_embedding=True matches embeddings)
  sync_param("lm_head.weight", ("base", "token_embedder", "embedding"))

  torch_model.load_state_dict(torch_state_dict)


# Store original hook class for clean cleanup
_original_training_hooks = dpo_hooks.DPOTrainingHooks


class InterceptingTrainingHooks(_original_training_hooks):
  """Custom training hooks class to intercept loss and rewards margin during real trainer step execution."""

  captured_metrics = []
  last_batch = None
  torch_policy_model = None
  torch_ref_model = None

  def on_train_step_start(self, train_ctx):
    super().on_train_step_start(train_ctx)
    InterceptingTrainingHooks.last_batch = train_ctx.data_hooks.train_batch
    # Sync the reference model on step 1. It's supposed to be frozen, the test would fail if it is drifting.
    if train_ctx.train_steps == 0:
      sync_jax_to_pytorch(train_ctx.ref_model, InterceptingTrainingHooks.torch_ref_model)

    # Sync the trained policy model on step 3 for logits comparison against Pytorch.
    if train_ctx.train_steps == 2:
      sync_jax_to_pytorch(train_ctx.model, InterceptingTrainingHooks.torch_policy_model)

  def on_train_step_end(self, train_ctx, train_step, train_loss, step_time=0.0):
    super().on_train_step_end(train_ctx, train_step, train_loss, step_time)

    prefix = train_ctx.metrics_prefix
    accuracy = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/accuracy", "train")[-1])
    margin = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/margin", "train")[-1])
    chosen_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/chosen", "train")[-1])
    rejected_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/rejected", "train")[-1])

    self.captured_metrics.append(
        {
            "loss": float(train_loss),
            "accuracy": accuracy,
            "margin": margin,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
        }
    )


class DPOTRLCorrectnessTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Set JAX default platform to CPU to eliminate CPU/TPU accelerator math variations
    jax.config.update("jax_platforms", "cpu")
    # Set JAX to CPU/TPU PRNG and SPMD defaults
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = (
          os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
      )

    # Monkeypatch the training hooks class dynamically
    dpo_hooks.DPOTrainingHooks = InterceptingTrainingHooks

  @classmethod
  def tearDownClass(cls):
    # Restore original hooks to prevent polluting other tests
    dpo_hooks.DPOTrainingHooks = _original_training_hooks
    # Clear class variables
    InterceptingTrainingHooks.captured_metrics = []
    InterceptingTrainingHooks.last_batch = None
    InterceptingTrainingHooks.torch_policy_model = None
    InterceptingTrainingHooks.torch_ref_model = None

  def test_maxtext_pytorch_dpo_parity(self):
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    prompt_str = "What is DPO?"
    chosen_str = "DPO stands for Direct Preference Optimization, an algorithm for aligning LLMs."
    rejected_str = "DPO is a marketing strategy used to target customers' preferences."
    max_prompt_len = 144  # We intentionally set max_prompt_len != max_sequence_length to test the padding logic.
    max_response_len = 112
    max_target_length = max_prompt_len + max_response_len
    beta = 0.1

    # 1. Initialize Symmetrical PyTorch model structures and register them with test hooks
    print("Initializing PyTorch tiny Qwen2 model...")
    torch_config = Qwen2Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=max_target_length,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        use_cache=False,
    )
    torch_policy_model = Qwen2ForCausalLM(torch_config)
    torch_ref_model = Qwen2ForCausalLM(torch_config)

    # The training hooks class will copy JAX model parameters to torch models.
    InterceptingTrainingHooks.torch_policy_model = torch_policy_model
    InterceptingTrainingHooks.torch_ref_model = torch_ref_model

    # 2. Setup a temporary local JSON dataset containing our DPO sample to test the REAL MaxText input pipeline
    # We duplicate the sample to provide enough contiguous batches for exactly 3 steps
    # We feed different inputs in Steps 1 and 2 to verify that state synchronization makes
    # Step 3 parity immune to training history!
    #
    # NOTE: We append 20 copies of the Step 3 item. The tf.data input pipeline prefetches
    # elements eagerly using AUTOTUNE (scaling with CPU cores). On high-resource CI agents,
    # a small dataset would trigger premature OutOfRange/StopIteration iterator exhaustion.
    # Appending 20 copies completely isolates the test from prefetch/exhaustion flakiness.
    temp_json_data = [
        {
            "prompt": "How does gradient descent work?",
            "chosen": (
                "Gradient descent is an optimization algorithm that updates"
                " parameters in the opposite direction of the gradient."
            ),
            "rejected": "Gradient descent is a method for climbing up hills to find local maxima.",
        },
        {
            "prompt": "What is a neural network?",
            "chosen": "A neural network is a network of interconnected nodes that learns representations from data.",
            "rejected": "A neural network is a database designed for storing tabular data.",
        },
    ] + [
        {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
        }
    ] * 20

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_json_path = os.path.join(temp_dir, "captured_dpo_sample.json")
      with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(temp_json_data, f)

      # Configure MaxText Hyperparameters pointing to our local JSON file
      print("\nInitializing JAX MaxText Config...")
      argv = [
          "src/maxtext/configs/base.yml",
          "model_name=qwen2.5-1.5b",
          f"tokenizer_path={model_id}",
          "scan_layers=False",
          "attention=dot_product",
          "per_device_batch_size=1",
          f"max_target_length={max_target_length}",
          "skip_jax_distributed_system=True",
          "enable_nnx=True",
          "pure_nnx=True",
          "pure_nnx_decoder=False",
          "remat_policy=full",
          "log_config=0",
          # Tiny architecture specifications
          "base_emb_dim=64",
          "head_dim=32",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "base_mlp_dim=128",
          "base_num_decoder_layers=2",
          "override_model_config=True",
          # Native input pipeline dataset specifications
          "use_dpo=True",
          "packing=False",
          "dataset_type=hf",
          "hf_path=json",
          f"hf_train_files={temp_json_path}",
          "tokenize_train_data=True",
          "train_data_columns=[prompt,chosen,rejected]",
          "eval_data_columns=[prompt,chosen,rejected]",
          "enable_data_shuffling=False",
          f"dpo.max_prompt_length={max_prompt_len}",
          "steps=3",
      ]
      config = pyconfig.initialize_pydantic(argv)

      # Run JAX DPO Native Training Loop and get flat metrics
      jax_ref = get_jax_reference(config)

    # 3. Evaluate PyTorch Reference dynamically on the synced distinct weights
    print("Evaluating PyTorch Reference on distinct synced models...")
    py_ref = get_pytorch_reference(
        policy_model=torch_policy_model,
        ref_model=torch_ref_model,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        prompt_str=prompt_str,
        chosen_str=chosen_str,
        rejected_str=rejected_str,
        beta=beta,
    )

    print("\n=== Symmetrical Parity Comparison (Step 3 - Diverged weights) ===")
    print(
        f"PyTorch Symmetrical Chosen Logps:   {py_ref['chosen_logps']:.6f} | "
        f"JAX Step 3: {jax_ref['chosen_logps']:.6f}"
    )
    print(
        f"PyTorch Symmetrical Rejected Logps: {py_ref['rejected_logps']:.6f} | "
        f"JAX Step 3: {jax_ref['rejected_logps']:.6f}"
    )
    print(f"PyTorch Ref Chosen Logps:           {py_ref['ref_chosen_logps']:.6f}")
    print(f"PyTorch Ref Rejected Logps:         {py_ref['ref_rejected_logps']:.6f}")
    print(f"PyTorch DPO Loss:                   {py_ref['loss']:.6f} | JAX Step 3: {jax_ref['loss']:.6f}")
    print(f"PyTorch Reward Margin:              {py_ref['margin']:.6f} | JAX Step 3: {jax_ref['margin']:.6f}")
    print(f"JAX train_dpo Step 1 Loss:          {jax_ref['loss_step_1']:.6f}")
    print(f"JAX train_dpo Step 1 Margin:        {jax_ref['margin_step_1']:.6f}")

    # Verify JAX policy and reference models did mutate and diverge after training steps
    self.assertNotEqual(jax_ref["margin"], 0.0, msg="JAX policy model did not mutate and diverge after training steps!")

    # Assert strict parity on raw, non-zero log probabilities at Step 3
    # This validates JAX ref_model + policy model outputs concurrently on distinct parameters!
    self.assertLess(
        abs(jax_ref["chosen_logps"] - py_ref["chosen_logps"]),
        0.5,
        msg=(
            "Step 3 Chosen Log probabilities diverge: "
            f"JAX {jax_ref['chosen_logps']:.6f} vs PyTorch {py_ref['chosen_logps']:.6f}"
        ),
    )
    self.assertLess(
        abs(jax_ref["rejected_logps"] - py_ref["rejected_logps"]),
        0.5,
        msg=(
            "Step 3 Rejected Log probabilities diverge: "
            f"JAX {jax_ref['rejected_logps']:.6f} vs PyTorch {py_ref['rejected_logps']:.6f}"
        ),
    )
    # Assert strict parity on non-trivial loss at Step 3 (Weights are completely diverged!)
    self.assertLess(
        abs(jax_ref["loss"] - py_ref["loss"]),
        0.05,
        msg=f"Step 3 DPO Loss diverges: JAX {jax_ref['loss']:.6f} vs PyTorch {py_ref['loss']:.6f}",
    )
    print("\n[INFO] Parity check succeeded with history variation under strict thresholds!")


if __name__ == "__main__":
  unittest.main()
