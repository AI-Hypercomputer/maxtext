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

"""PyTorch reference functions and weight synchronization helpers for DPO/ORPO integration tests.

Note: The integration tests validate parity by comparing JAX and PyTorch/TRL on an
identical, miniaturized 2-layer Qwen2 model architecture.
- The JAX model configuration is defined in `tests/post_training/integration/dpo_correctness_base.py`
  (via `_build_jax_config` which overrides model dimensions to a tiny 2-layer shape).
- The PyTorch model configuration is defined here in `create_pytorch_config` and is
  designed to be structurally identical to the JAX model to allow direct parameter
  synchronization and logit comparison.
"""

import numpy as np
from flax import nnx
import torch
from transformers import Qwen2Config
from trl import DPOConfig, DPOTrainer
from datasets import Dataset


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


def get_pytorch_reference(
    policy_model,
    ref_model,
    tokenizer,
    prompt_str,
    chosen_str,
    rejected_str,
    beta=0.1,
    tokenize_together=False,
):
  # pylint: disable=too-many-positional-arguments
  """Computes reference chosen/rejected logps and loss in PyTorch using TRL trainers."""
  policy_model.eval()
  if ref_model is not None:
    ref_model.eval()

  # Set up the tokenizer based on JAX's tokenize_together formatting.
  # tokenize_together=True (2-column format) has no EOS in the middle of prompt.
  # tokenize_together=False (3-column format) has EOS in the middle.
  tokenizer.add_eos_token = False
  if not tokenize_together:
    prompt_str = prompt_str + tokenizer.eos_token

  # Build Dataset
  dataset = Dataset.from_list(
      [
          {
              "prompt": prompt_str,
              "chosen": chosen_str,
              "rejected": rejected_str,
          }
      ]
  )

  training_args = DPOConfig(
      output_dir="/tmp/trl_ref",
      beta=beta,
      max_length=256,
      use_cpu=True,
      remove_unused_columns=False,
  )
  trainer = DPOTrainer(
      model=policy_model,
      ref_model=ref_model,
      args=training_args,
      train_dataset=dataset,
      processing_class=tokenizer,
  )
  dataloader = trainer.get_train_dataloader()
  batch = next(iter(dataloader))
  device = torch.device("cpu")
  batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

  with torch.no_grad():
    loss = trainer.compute_loss(policy_model, batch)

  # Extract logps and ref_logps from "eval" key in trainer._metrics
  # pylint: disable=protected-access
  metrics = trainer._metrics["eval"]
  chosen_logps = metrics["logps/chosen"][0]
  rejected_logps = metrics["logps/rejected"][0]

  # Reconstruct ref_chosen_logps and ref_rejected_logps:
  ref_chosen_logps = chosen_logps - (metrics["rewards/chosen"][0] / beta)
  ref_rejected_logps = rejected_logps - (metrics["rewards/rejected"][0] / beta)

  # Margin:
  margin = metrics["rewards/margins"][0] / beta

  return {
      "chosen_logps": chosen_logps,
      "rejected_logps": rejected_logps,
      "ref_chosen_logps": ref_chosen_logps,
      "ref_rejected_logps": ref_rejected_logps,
      "loss": loss.item(),
      "margin": margin,
  }


def create_pytorch_config(max_target_length: int) -> Qwen2Config:
  """Helper to create a symmetrical PyTorch tiny Qwen2 model configuration.

  This configuration must be kept structurally identical to the JAX model configuration
  defined in `tests/post_training/integration/dpo_correctness_base.py` (specifically
  the tiny architecture overrides in `_build_jax_config`: base_emb_dim=64,
  base_num_decoder_layers=2, etc.).
  """
  return Qwen2Config(
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
