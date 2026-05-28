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

"""
Script to compare SFT baseline vs. DPO qualitative decodings.
Includes automated checks for output similarity and semantic steerability.
"""

import re
import sys


def parse_responses(log_path):
  """Parses prompt decodings from the inference log file."""
  responses = {}
  # Match pattern: Input `PROMPT` -> `RESPONSE`
  pattern = re.compile(r"Input `([^`]+)` -> `([^`]+)`", re.DOTALL)

  try:
    with open(log_path, "r", encoding="utf-8") as f:
      content = f.read()
  except FileNotFoundError:
    print(f"Error: Response log file '{log_path}' not found.", file=sys.stderr)
    return responses

  matches = pattern.findall(content)
  for prompt, response in matches:
    prompt = prompt.strip()
    # Escape vertical bars for markdown table formatting
    response = response.strip().replace("\n", "<br>").replace("|", "\\|")
    responses[prompt] = response

  return responses


def get_jaccard_similarity(s1, s2):
  """Computes Jaccard similarity of unique words between two responses."""
  # Normalize text (lowercase, remove basic punctuation)
  words1 = set(re.findall(r"\b\w+\b", s1.lower()))
  words2 = set(re.findall(r"\b\w+\b", s2.lower()))

  union = words1.union(words2)
  if not union:
    return 1.0
  return len(words1.intersection(words2)) / len(union)


def main():
  sft_file = "quals/logs/sft_responses.log"
  dpo_file = "quals/logs/dpo_responses.log"

  sft_res = parse_responses(sft_file)
  dpo_res = parse_responses(dpo_file)

  prompts = [
      "Explain the concept of Direct Preference Optimization in simple terms.",
      "Write a short story about a robot learning to cook.",
      "What are the pros and cons of using JAX for machine learning?",
      "How do I optimize a MaxText training run on TPU v4-8?",
      "Give me a recipe for a healthy vegetarian dinner.",
  ]

  print("# Qualitative Decoding Comparison & Automated Validation")
  print("\n| Prompt | SFT Baseline Response | DPO Fine-Tuned Response | Similarity | Check Status |")
  print("| --- | --- | --- | --- | --- |")

  failed_checks = 0
  total_similarity = 0.0
  evaluated_prompts = 0

  for prompt in prompts:
    sft_out = sft_res.get(prompt, "N/A")
    dpo_out = dpo_res.get(prompt, "N/A")
    p_esc = prompt.replace("|", "\\|")

    if sft_out == "N/A" or dpo_out == "N/A":
      print(f"| {p_esc} | {sft_out} | {dpo_out} | N/A | Missing Output |")
      failed_checks += 1
      continue

    similarity = get_jaccard_similarity(sft_out, dpo_out)
    total_similarity += similarity
    evaluated_prompts += 1

    status = "Passed"
    # Check if responses are identical or too similar (e.g., > 0.90 Jaccard similarity)
    if similarity > 0.90:
      status = "FAILED: Identical Output"
      failed_checks += 1
    elif prompt == "Explain the concept of Direct Preference Optimization in simple terms.":
      # Specific steerability check for DPO concept definition
      dpo_lower = dpo_out.lower()
      # DPO response should NOT look like a marketing strategy
      if "marketing strategy" in dpo_lower or "audience" in dpo_lower or "campaign" in dpo_lower:
        status = "FAILED: SFT Marketing Hallucination Retained"
        failed_checks += 1
      # DPO response should have mathematical optimization keywords
      elif not any(w in dpo_lower for w in ["optimization", "preference", "loss", "align", "feedback"]):
        status = "FAILED: Missing Alignment Terminology"
        failed_checks += 1

    print(f"| {p_esc} | {sft_out[:150]}... | {dpo_out[:150]}... | {similarity:.2%} | {status} |")

  avg_similarity = total_similarity / evaluated_prompts if evaluated_prompts > 0 else 1.0
  print("\n**Summary Evaluation:**")
  print(f"- Average Response Word Similarity: {avg_similarity:.2%}")
  print(f"- Failed Validation Checks: {failed_checks}/{len(prompts)}")

  # Strict qualification failure
  if avg_similarity > 0.95:
    print("\nCRITICAL ERROR: SFT and DPO outputs have >95% word similarity.", file=sys.stderr)
    print("DPO training yielded zero behavioral alignment progress.", file=sys.stderr)
    sys.exit(3)

  if failed_checks > 0:
    print(f"\nCRITICAL ERROR: {failed_checks} qualitative validation checks failed.", file=sys.stderr)
    sys.exit(4)

  print("\nSUCCESS: Qualitative behavioral alignment successfully verified.")
  sys.exit(0)


if __name__ == "__main__":
  main()
