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
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

from MaxText.utils.ckpt_conversion.utils.hf_utils import (
    # check_predicted_tokens_match,
    check_arrays_match,
)
from MaxText import max_logging
# Read Hugging Face token from environment variable
hf_token = os.environ.get("HF_AUTH_TOKEN")

"""
This script is to verify HuggingFace (HF) checkpoints that have been converted from MaxText. 

It loads the converted HF model and a "golden" (reference) HF model, and:
    1. runs a foward pass of the converted ckpt model
    2. compares their weights, output logits for a given input
    3. Compare the predicted token sequences
    
Extra Requirements:
    torch
    huggingface_hub
    transformers
    accelerate
    tabulate
"""


def get_all_modules(model):
  """Get all weights names from a HF model."""
  modules = []
  for name, _ in model.named_modules():
    if name and hasattr(model.get_submodule(name), "weight"):
      modules.append(name)
  return modules


def check_weights_match(model, golden_model, tol=0.1):
  """Compare weights between two HF models."""
  modules = get_all_modules(golden_model)

  for module in modules:
    golden_weights = golden_model.get_submodule(module).state_dict()["weight"]
    model_weight = model.get_submodule(module).state_dict()["weight"]
    check_arrays_match(golden_weights, model_weight, tol)


def get_logits(inputs, model, golden_model):
  """Get logits from two HF models for comparison."""
  logits = model(**inputs, output_hidden_states=True).logits
  golden_logits = golden_model(**inputs, output_hidden_states=True).logits

  return logits, golden_logits


def get_top_k_tokens_scores(logits_tensor, tokenizer_instance, k=10, description=""):
  """Get the top-k tokens and their scores from a given logits tensor."""
  max_logging.log(f"\n--- {description} top {k} tokens ---")
  collected_tokens = []
  tokens = []
  # Ensure logits_tensor is on CPU for operations like topk and item()
  logits_tensor = logits_tensor.cpu()
  topk_results = torch.topk(logits_tensor[0, -1], k=k)
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


def run_prompts(args: argparse.Namespace) -> None:
  """
  Args:
      - golden_model_id (str): HF model ID for the golden model.
      - hf_checkpoint_path (str): Path to the converted HF checkpoint.
      - max_kl_div (float): Maximum allowed KL divergence.
  """
  golden_model = AutoModelForCausalLM.from_pretrained(args.golden_model_id, torch_dtype=torch.bfloat16)
  golden_tokenizer = AutoTokenizer.from_pretrained(args.golden_model_id)

  tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint_path)
  model, _ = AutoModelForCausalLM.from_pretrained(
      args.hf_checkpoint_path, trust_remote_code=True, torch_dtype=torch.bfloat16, output_loading_info=True
  )

  # max_logging.log(loading_info)

  prompts = ["I love to", "Today is a", "What is the"]
  for input_text in prompts:
    max_logging.log(f"\n--- Prompt: {input_text} ---")
    inputs = tokenizer(input_text, return_tensors="pt")
    # --- Generate Output ---
    with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    # --- Decode and Print ---
    max_logging.log(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    # --- Compare tokens ---
    model_logits, golden_model_logits = get_logits(inputs, model, golden_model)
    tokens = get_top_k_tokens_scores(model_logits, tokenizer, k=10, description="converted model")
    golden_tokens = get_top_k_tokens_scores(golden_model_logits, golden_tokenizer, k=10, description="golden model")
    compare_top_tokens(converted_tokens=tokens, golden_tokens=golden_tokens)

    check_kl_divergence(model_logits, golden_model_logits, atol=args.max_kl_div)

  """
  if the model's structure is exactly the same as the golden model (layers, vocab_size, etc.), 
  you can check more weights details using the following steps:

  check_weights_match(model, golden_model)

  # Check logits from the first 5 tokens match
  check_arrays_match(model_logits[0, :5, :], golden_model_logits[0, :5, :], atol=0.2)

  check_predicted_tokens_match(model_logits, golden_model_logits)
  """


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verify HuggingFace checkpoints converted from MaxText.")
  parser.add_argument(
      "--golden_model_id",
      type=str,
      default="google/gemma-2-2b-it",
      help="The HuggingFace model ID for the golden/reference model.",
  )
  parser.add_argument(
      "--hf_checkpoint_path",
      type=str,
      default=os.path.expanduser("~/.hf_output/"),
      help="Path to the converted HuggingFace checkpoint directory.",
  )
  parser.add_argument("--max_kl_div", type=float, default=0.02, help="Maximum allowed KL divergence between model logits.")

  parsed_args = parser.parse_args()

  run_prompts(parsed_args)
