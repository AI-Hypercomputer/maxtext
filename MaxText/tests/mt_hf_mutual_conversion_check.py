"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

from MaxText.utils.ckpt_conversion.utils.hf_utils import (
    # check_arrays_match,
    convert_jax_weight_to_torch,
)
from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.globals import PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations

hf_token = os.environ.get("HF_AUTH_TOKEN")

"""
This script is to compare the logits of maxtext checkpoint and Huggingface checkpoint.
Used to verify the checkpoint conversion (to_huggingface.py and to_maxtext.py)

It loads the HF checkpoint and a maxtext checkpoint, and:
    1. runs a foward pass of a MaxText model and a HF model
    2. compares their output logits for a given input
    3. compares the predicted token sequences
    
Extra Requirements:
    torch
    huggingface_hub
    transformers
    accelerate
    tabulate
"""


def get_top_k_tokens_scores(logits_tensor, tokenizer_instance, k=10, description=""):
  """Get the top-k tokens and their scores from a given logits tensor."""
  max_logging.log(f"\n--- {description} top {k} tokens ---")
  collected_tokens = []
  tokens = []
  topk_results = torch.topk(logits_tensor[0], k=k)
  for i in range(k):
    tok_id = topk_results.indices[i].item()
    score = topk_results.values[i].item()
    tok = tokenizer_instance.decode(tok_id)
    collected_tokens.append({"id": int(tok_id), "token": tok.strip(), "score": float(score)})
    tokens.append({"id": int(tok_id), "token": tok.strip(), "score": float(score)})

  # Prepare data for tabulate: a list of lists
  table_data = [[d["id"], d["token"], d["score"]] for d in collected_tokens]
  max_logging.log(tabulate(table_data, headers=["Token ID", "Token", "Score"], tablefmt="orgtbl"))
  return tokens


def compare_top_tokens(converted_tokens, golden_tokens):
  """
  Compares two lists of top tokens and calculates similarity metrics.

  Args:
      converted_tokens: top tokens from the converted model.
      golden_tokens:  top tokens from the golden model.
  """
  # Extract the sets of token IDs for comparison
  converted_ids = {token["id"] for token in converted_tokens}
  golden_ids = {token["id"] for token in golden_tokens}

  # --- Metric 1: Overlap Count & Jaccard Similarity ---
  intersection = converted_ids.intersection(golden_ids)
  union = converted_ids.union(golden_ids)

  overlap_count = len(intersection)
  jaccard_similarity = overlap_count / len(union) if union else 0.0

  # --- Metric 2: Rank Agreement ---
  rank_matches = 0
  min_len = min(len(converted_tokens), len(golden_tokens))
  for i in range(min_len):
    if converted_tokens[i]["id"] == golden_tokens[i]["id"]:
      rank_matches += 1

  rank_agreement = (rank_matches / min_len) * 100 if min_len > 0 else 0.0

  metrics = {
      "overlap_count": f"{overlap_count}/{min_len}",
      "jaccard_similarity": jaccard_similarity,
      "rank_agreement_percentage": rank_agreement,
  }

  max_logging.log("\n--- Similarity Metrics of Top Tokens ---")
  table = [[key, value] for key, value in metrics.items()]
  max_logging.log(tabulate(table, headers=["Metric", "Value"], tablefmt="orgtbl"))


def check_kl_divergence(model_logits, golden_logits, atol=0.02):
  """
  Calculates KL divergence D_KL(P_golden || Q_model) over a batch of sequences.

  Args:
      model_logits: Logits from the converted model (Batch, SeqLen, VocabSize).
      golden_logits: Logits from the golden model (Batch, SeqLen, VocabSize).
      token_size: The number of vocabulary entries to consider for the comparison.
                  (Effectively vocab_size_to_compare).
  """
  # 1. Select the relevant vocabulary slice from the logits.
  token_size = min(model_logits.shape[2], golden_logits.shape[2])
  model_logits_sliced = model_logits[..., :token_size]
  golden_logits_sliced = golden_logits[..., :token_size]

  # 2. Reshape
  b, s, v = model_logits_sliced.shape
  model_logits_reshaped = model_logits_sliced.view(b * s, v)
  golden_logits_reshaped = golden_logits_sliced.view(b * s, v)

  # 3. Get the probability distributions.
  golden_probabilities = F.softmax(golden_logits_reshaped, dim=-1)
  model_log_probabilities = F.log_softmax(model_logits_reshaped, dim=-1)

  # 4. Calculate avg KL divergence for all token distributions.
  # use 'batchmean'; the sum of the KL divergences for each token in the batch
  # and then divides by the number of tokens (b * s)
  kl_div_value = F.kl_div(
      input=model_log_probabilities,
      target=golden_probabilities,
      reduction="batchmean",  # Use 'batchmean' for the average KL per token.
      log_target=False,
  )

  max_logging.log(f"\nAverage KL divergence per token (D_KL(P_golden || Q_model)): {kl_div_value.item():.6f}")

  # To find the max KL divergence for any single token in the set
  # use reduction='none'.
  kl_divs_per_token = F.kl_div(
      input=model_log_probabilities, target=golden_probabilities, reduction="none", log_target=False
  ).sum(
      dim=-1
  )  # Sum over the vocab dim to get a single KL value per token

  max_kl_div = kl_divs_per_token.max()
  max_logging.log(f"\nMax KL divergence for a single token in the set: {max_kl_div.item():.6f}")

  assert max_kl_div < atol, f"KL divergence values {max_kl_div.item():.6f} exceed the threshold {atol}"


def run_prompts(args: argparse.Namespace, additional_maxtext_overrides: list) -> None:
  """
  Args:
      - hf_model_id (str): HF model ID for the HF checkpoint.
      - maxtext_checkpoint_path (str): Path to the MaxText checkpoint.
      - maxtext_base_config_path (str): Path to MaxText base configuration.
      - maxtext_model_name (str): Name of the MaxText model.
      - max_kl_div (float): Maximum allowed KL divergence.
      - additional_maxtext_overrides (list): List of MaxText configuration overrides.
  """
  # 1. Load Golden HF Model and Tokenizer
  hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, torch_dtype=torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)  # Use this for both

  # 2. Load MaxText Model and Parameters
  maxtext_argv = [""]  # Placeholder for script name
  maxtext_argv.append(args.maxtext_base_config_path)

  if args.maxtext_model_name:
    additional_maxtext_overrides.append(f"model_name={args.maxtext_model_name}")
  additional_maxtext_overrides.append(f"load_parameters_path={args.maxtext_checkpoint_path}")
  additional_maxtext_overrides.append("per_device_batch_size=1")  # Match batch size for comparison
  additional_maxtext_overrides.append("max_prefill_predict_length=16")
  additional_maxtext_overrides.append("max_target_length=32")
  additional_maxtext_overrides.append("scan_layers=False")
  maxtext_argv.extend(additional_maxtext_overrides)
  config = pyconfig.initialize(maxtext_argv)

  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  init_rng, rng1 = jax.random.split(init_rng)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  maxtext_model = models.Transformer(config, mesh, quant=quant)
  maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng1, mesh, None)

  prompts = ["I love to", "Today is a", "What is the"]
  for input_text in prompts:
    max_logging.log(f"\n--- Prompt: {input_text} ---")

    # Tokenize for HF
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, max_length=config.max_target_length, truncation=True)
    actual_seq_len = inputs["input_ids"].shape[1]
    # actual_seq_len = 4

    # Tokenize for MaxText
    mt_ids = jnp.asarray(inputs["input_ids"], dtype=jnp.int32)

    if mt_ids.shape[0] != config.global_batch_size_to_train_on:  # Ensure batch size matches
      mt_ids = jnp.repeat(mt_ids, config.global_batch_size_to_train_on // mt_ids.shape[0], axis=0)

    s = (config.global_batch_size_to_train_on, config.max_target_length)
    mt_decoder_segment_ids_full = jnp.zeros(s, dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR

    mt_decoder_segment_ids = mt_decoder_segment_ids_full[:, :actual_seq_len]

    # Create full decoder positions up to max_target_length
    mt_decoder_positions_full = jnp.stack(
        [jnp.arange(config.max_target_length, dtype=jnp.int32) for _ in range(config.global_batch_size_to_train_on)]
    )
    mt_decoder_positions = mt_decoder_positions_full[:, :actual_seq_len]
    # max_logging.log(f"MaxText input shapes: ids={mt_ids.shape}, "
    #                 f"decoder_positions={mt_decoder_positions.shape}, "
    #                 f"decoder_segment_ids={mt_decoder_segment_ids.shape}")

    # --- HF Forward Pass ---
    with torch.no_grad():
      hf_logits_torch = hf_model(**inputs).logits

    # --- MaxText Forward Pass ---
    mt_logits_jax = maxtext_model.apply(
        maxtext_state.params,
        mt_ids,
        mt_decoder_positions,
        mt_decoder_segment_ids,
        enable_dropout=False,
        rngs={"aqt": init_rng},
    )
    mt_logits_jax_sliced = mt_logits_jax[:, :actual_seq_len, :]
    mt_logits_torch = convert_jax_weight_to_torch(mt_logits_jax_sliced)

    # --- Compare logits for the last token prediction ---
    hf_last_token_logits = hf_logits_torch[:, -1, :]
    mt_last_token_logits = mt_logits_torch[:, -1, :]  # MaxText output already sliced to actual_seq_len

    tokens_maxtext = get_top_k_tokens_scores(mt_last_token_logits, tokenizer, k=10, description="MaxText model")
    tokens_hf = get_top_k_tokens_scores(hf_last_token_logits, tokenizer, k=10, description="HF model")
    compare_top_tokens(converted_tokens=tokens_maxtext, golden_tokens=tokens_hf)

    # --- Compare all logits in the sequence (for the first batch item) ---
    # Unsqueeze to add batch dimension for check_kl_divergence: [1, seq, vocab]
    check_kl_divergence(mt_logits_torch[0].unsqueeze(0), hf_logits_torch[0].unsqueeze(0), atol=args.max_kl_div)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verify HuggingFace checkpoints converted from MaxText.")
  parser.add_argument(
      "--hf_model_id",
      type=str,
      default="google/gemma-2-2b",
  )
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      default="gemma2-2b",
  )
  parser.add_argument("--maxtext_checkpoint_path", type=str, default=os.path.expanduser("~/.mt_output/0/items"))
  parser.add_argument(
      "--maxtext_base_config_path",
      type=str,
      default=os.path.join(PKG_DIR, "configs", "base.yml"),
  )
  parser.add_argument("--max_kl_div", type=float, default=0.02, help="Maximum allowed KL divergence between model logits.")

  parsed_args, unknown_args = parser.parse_known_args()

  maxtext_config_overrides = []
  for arg in unknown_args:
    maxtext_config_overrides.append(arg.lstrip("-"))  # pyconfig expects "key=value"

  run_prompts(parsed_args, maxtext_config_overrides)
