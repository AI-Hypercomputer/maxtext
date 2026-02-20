# Copyright 2025 Google LLC
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


"""
An evaluation gemini call to assess the generated-code vs. human-written code
"""

from maxtext.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template
from maxtext.experimental.agent.ckpt_conversion_agent.base import BaseAgent

import argparse
import zlib
import bz2
import lzma
import os.path

# file pattern, [0] is groud truth, [1] is generated code


def compressed_size(data: bytes, method: str = "gzip") -> int:
  """
  Compress `data` using the specified method and return the length of the compressed bytes.
  method: 'gzip' | 'bz2' | 'lzma'
  """
  if method == "gzip":
    return len(zlib.compress(data))
  elif method == "bz2":
    return len(bz2.compress(data))
  elif method == "lzma":
    return len(lzma.compress(data))
  else:
    raise ValueError(f"Unknown compression method: {method}")


def estimate_kolmogorov(filepath: str) -> dict:
  """
  Read the file at `filepath` and return a dict of compression-based complexity estimates.
  """
  with open(filepath, "rb") as f:
    data = f.read()
  results = {
      "original_size": len(data),
      "gzip_size": compressed_size(data, "gzip"),
      "bz2_size": compressed_size(data, "bz2"),
      "lzma_size": compressed_size(data, "lzma"),
  }
  # You could average or take the minimum as a final estimate:
  results["approx_k_complexity"] = min(results["gzip_size"], results["bz2_size"], results["lzma_size"])
  return results


def main():
  parser = argparse.ArgumentParser(
      description="Gemini evaluate the agent code implementation against human-written ground truth code"
  )
  parser.add_argument("--files", nargs=2, help="Paths to code files to analyze.")
  parser.add_argument("--api_key", type=str, help="API key.")
  parser.add_argument("--dir_path", type=str, help="Directory path.")

  args = parser.parse_args()

  baseAgent = BaseAgent(api_key=args.api_key)
  dir_path = args.dir_path

  prompt_templates = {
      "eval": load_prompt_template(f"{dir_path}/prompts/rate_outputs.txt"),
      "pitfalls": load_prompt_template(f"{dir_path}/prompts/04_pitfalls.txt"),
  }

  # # Evaluation 1: Estimate complexity for each file
  # estimates = {}
  # for path in args.files:
  #     if not os.path.isfile(path):
  #         print(f"Error: file not found: {path}", file=sys.stderr)
  #         sys.exit(1)
  #     estimates[path] = estimate_kolmogorov(path)

  # # Display results
  # for path, stats in estimates.items():
  #     print(f"\nFile: {path}")
  #     print(f"  Original size (bytes):     {stats['original_size']}")
  #     print(f"  Gzip compressed size:      {stats['gzip_size']}")
  #     print(f"  BZ2 compressed size:       {stats['bz2_size']}")
  #     print(f"  LZMA compressed size:      {stats['lzma_size']}")
  #     print(f"  Estimated K-complexity:    {stats['approx_k_complexity']}  (min of above)\n")

  with open(os.path.join(dir_path, args.files[0]), "rt", encoding="utf8") as f:
    ground_truth = f.read()
  with open(os.path.join(dir_path, args.files[1]), "rt", encoding="utf8") as f:
    dsl_chain = f.read()

  prompt = prompt_templates["eval"].format(
      ground_truth=ground_truth,
      dsl_chain=dsl_chain,
  )
  analysis = baseAgent.generate_text(prompt)
  print(analysis)


if __name__ == "__main__":
  main()
