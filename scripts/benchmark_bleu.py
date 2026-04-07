# Copyright 2023–2026 Google LLC
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

"""Standalone script to benchmark BLEU and Self-BLEU for DBS vs Baseline."""

import os
import sys
import jax
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from maxtext.configs import pyconfig
# Deferred import for maxengine to allow mock mode without JetStream
# from maxtext.inference.maxengine import maxengine

def calculate_self_bleu(generated_beams, bleu_metric):
  """Calculates diversity by measuring pairwise BLEU among generated beams.
  Higher Self-BLEU means LESS diversity.
  """
  if len(generated_beams) <= 1:
    return 1.0
  
  scores = []
  for i in range(len(generated_beams)):
    prediction = generated_beams[i]
    # We provide all other beams as valid references for the current beam
    references = [[generated_beams[j] for j in range(len(generated_beams)) if i != j]]
    # Prediction: str, Reference: list[list[str]]
    res = bleu_metric.compute(predictions=[prediction], references=references)
    scores.append(res['bleu'])
  return float(np.mean(scores))

def main():
  # 1. Initialize MaxText Config
  # Usage: python3 scripts/benchmark_bleu.py src/maxtext/configs/base.yml ...
  is_mock = "--mock" in sys.argv
  if is_mock:
    sys.argv.remove("--mock")
  
  # Manually extract max_dataset_examples because it's not in the core MaxText types.py
  max_examples = 100
  for arg in sys.argv[:]:
    if arg.startswith("max_dataset_examples="):
      max_examples = int(arg.split("=")[1])
      sys.argv.remove(arg)
      break
  
  config = pyconfig.initialize(sys.argv)
  
  if is_mock:
    print(f"RUNNING IN MOCK MODE (No model loading) - Limit: {max_examples}")
    engine = None
    params = None
  else:
    from maxtext.inference.maxengine import maxengine
    # Note: MaxEngine initialization sets up the JAX mesh and loaded model
    engine = maxengine.MaxEngine(config)
  
  # 2. Load Dataset (CNN/DailyMail)
  print(f"Loading CNN/DailyMail dataset (test split) - Goal: {max_examples} samples...")
  # Use streaming=True to avoid downloading the whole dataset at once
  dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
  
  # Max examples was extracted manually above
  limit = max_examples
    
  # 3. Load BLEU metric
  print("Loading BLEU metric...")
  try:
    bleu = evaluate.load("bleu")
  except Exception as e: # pylint: disable=broad-exception-caught
    print(f"Could not load BLEU metric: {e}. Ensure 'evaluate' and 'sacrebleu' are installed.")
    return

  # 4. Initialize Model Params
  if not is_mock:
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)
    params = engine.load_params(rng_load)
  else:
    rng = jax.random.PRNGKey(1234)
    params = None
  
  results = {
      "bleu": [],
      "self_bleu": []
  }

  print(f"Starting Benchmark on {limit} examples...")
  for i, example in enumerate(tqdm(dataset, total=limit)):
    if i >= limit:
      break
      
    # Prepare prompt
    article = example['article']
    # MaxText expects a single string prompt or pre-tokenized batch.
    # We use a simple wrap here.
    prompt = f"Summarize the following article:\n{article}\n\nSummary:"
    reference = example['highlights']

    # 5. Generate with Engine
    if is_mock:
      # Return multiple dummy beams for the mock to test Self-BLEU
      import random
      ref_words = reference.split()
      beams = []
      for _ in range(config.decode_num_beams):
        # Generate slightly different "beams" by shuffling or sampling
        beam = " ".join(random.sample(ref_words, min(len(ref_words), 12)))
        beams.append(beam)
      output_text_or_beams = beams
    else:
      metadata = engine.get_tokenizer()
      tokenizer_model = engine.build_tokenizer(metadata)
      
      # We perform a simplified inference call sequence similar to decode.py
      # but adapted for multiple beam output if needed.
      
      rng, rng_gen = jax.random.split(rng)
      # Note: MaxEngine.generate returns (new_state, result_tokens)
      
      # Helper to get one string result
      def get_inference_result(text_input):
         # This is a simplified wrapper around the engine's decode logic
         tokens, true_length = tokenizer_model.encode(text_input, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
         
         # Prefill
         prefill_res, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
         
         # Insert
         decode_state = engine.init_decode_state(rng_gen)
         decode_state = engine.insert(prefill_res, decode_state, slot=0)
         
         # Generate
         steps = range(config.max_prefill_predict_length, config.max_target_length)
         all_tokens = [first_token.get_result_at_slot(0).tokens.item()]
         
         for _ in steps:
           decode_state, sampled_tokens = engine.generate(params, decode_state)
           all_tokens.append(sampled_tokens.get_result_at_slot(0).tokens.item())
           if all_tokens[-1] == tokenizer_model.eos_id:
             break
             
         return tokenizer_model.decode(all_tokens)

      output_text_or_beams = get_inference_result(prompt)
    
    # Handle both single string (greedy/beam-search-top1) and multiple beams
    if isinstance(output_text_or_beams, list):
       top_prediction = output_text_or_beams[0]
       beams = output_text_or_beams
    else:
       top_prediction = output_text_or_beams
       beams = [output_text_or_beams]

    # 6. Score Top-1 BLEU
    score = bleu.compute(predictions=[top_prediction], references=[[reference]])
    results["bleu"].append(score['bleu'])
    
    # 7. Score Self-BLEU if multiple beams were generated
    if len(beams) > 1:
       sb = calculate_self_bleu(beams, bleu)
       results["self_bleu"].append(sb)

  # 7. Print Final Metrics
  print("\n" + "="*40)
  print(f"BENCHMARK RESULTS")
  print(f"Strategy: {config.decode_sampling_strategy}")
  print(f"Num Examples: {len(results['bleu'])}")
  print(f"Avg BLEU-4 (Quality): {np.mean(results['bleu']):.4f}")
  if results["self_bleu"]:
    print(f"Avg Self-BLEU (Diversity): {np.mean(results['self_bleu']):.4f}")
    print(" (Lower = More Diverse)")
  print("="*40)

if __name__ == "__main__":
  main()
