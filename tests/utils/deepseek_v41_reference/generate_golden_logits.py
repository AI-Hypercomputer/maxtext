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

"""Generates golden logits with the plain-PyTorch DeepSeek-V4.1 reference, for forward_pass_logit_checker.

Works on the released (FP8 / FP4 block-quantized) checkpoint as well as unquantized ones: weights are dequantized
to bf16 on load and, with --fp32_compute, every matmul runs in float32. The output pickle holds one dict per prompt
with `prompt`, `tokens` and `logits` ([seq, vocab] float32), the format `--golden_logits_path` reads.

  python -m tests.utils.deepseek_v41_reference.generate_golden_logits \
      --ckpt_dir /dev/shm/DeepSeek-V4.1-Flash --output golden.pkl --fp32_compute [--index_topk 4 ...]
"""

import argparse
import pickle
import time

import torch

from . import runner

# The prompts forward_pass_logit_checker uses with --run_hf_model (kept in sync with build_long_prompt there).
_LONG_PROMPT_PASSAGE = (
    "The city of Lyon sits at the confluence of the Rhone and Saone rivers in eastern France. "
    "Founded as a Roman colony in 43 BC, it served for centuries as the capital of the Gauls. "
    "Its silk industry flourished in the sixteenth century, drawing weavers from across Europe. "
    "The old town preserves narrow covered passageways known as traboules, built so that silk "
    "merchants could carry bolts of fabric between workshops without exposing them to rain. "
)
LONG_PROMPT = (
    _LONG_PROMPT_PASSAGE * 5 + "\n\nBased on the passage above, the traboules of Lyon were originally built so that"
)
PROMPTS = ["I love to", "Today is a", "What is the", LONG_PROMPT]


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--ckpt_dir", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="weight storage dtype")
  parser.add_argument("--fp32_compute", action="store_true")
  parser.add_argument("--max_seq_len", type=int, default=1024)
  parser.add_argument("--index_topk", type=int, default=None)
  parser.add_argument("--candidate_topk_blocks", type=int, default=None)
  parser.add_argument("--threads", type=int, default=0)
  parser.add_argument(
      "--extra_topk_output",
      default="",
      help="Also write goldens with index_topk=4 / candidate_topk_blocks=8 here, reusing the loaded model.",
  )
  args = parser.parse_args()

  if args.threads:
    torch.set_num_threads(args.threads)
  overrides = {
      k: v for k, v in (("index_topk", args.index_topk), ("candidate_topk_blocks", args.candidate_topk_blocks)) if v
  }
  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
  t0 = time.time()
  model, margs = runner.load_reference_model(
      args.ckpt_dir,
      dtype=args.dtype,
      max_seq_len=args.max_seq_len,
      tokenizer=tokenizer,
      fp32_compute=args.fp32_compute,
      **overrides,
  )
  print(
      f"loaded reference in {time.time() - t0:.0f}s: index_topk={margs.index_topk} "
      f"candidate_topk_blocks={margs.candidate_topk_blocks}",
      flush=True,
  )

  run_prompts(model, tokenizer, args.output)
  if args.extra_topk_output:
    for layer in model.layers:
      if layer.attn.indexer is not None:
        layer.attn.indexer.index_topk = 4
        layer.attn.indexer.candidate_topk_blocks = 8
    print("index_topk=4 candidate_topk_blocks=8", flush=True)
    run_prompts(model, tokenizer, args.extra_topk_output)


def run_prompts(model, tokenizer, output):
  out = []
  for prompt in PROMPTS:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    t0 = time.time()
    logits = model(ids, full_logits=True)[0].float()
    top = logits[-1].topk(5)
    print(
        f"{ids.shape[1]} tokens in {time.time() - t0:.0f}s; last-token top-5 "
        f"{tokenizer.convert_ids_to_tokens(top.indices)}",
        flush=True,
    )
    out.append({"prompt": prompt, "tokens": ids[0].numpy(), "logits": logits.numpy()})
  with open(output, "wb") as f:
    pickle.dump(out, f)
  print(f"wrote {output}", flush=True)


if __name__ == "__main__":
  main()
