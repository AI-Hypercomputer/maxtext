"""Decode the original text-only four-layer HF Kimi model on GPU."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-path", default="/dev/shm/hf_mini/kimi_k3_4layers_original_gpu")
  parser.add_argument("--output", type=Path, help="Optional JSON result path")
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("Use a GPU VM with the original CUDA model dependencies.")
  config = json.loads((Path(args.model_path) / "config.json").read_text())
  if config.get("num_hidden_layers") != 4:
    raise ValueError("Expected a validated text-only checkpoint with exactly four layers.")
  tokenizer = AutoTokenizer.from_pretrained(
      args.model_path,
      trust_remote_code=True,
      local_files_only=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
      args.model_path,
      trust_remote_code=True,
      local_files_only=True,
      dtype="auto",
      device_map="auto",
  ).eval()
  if any(p.device.type != "cuda" for p in model.parameters()):
    raise RuntimeError("All model parameters must reside on GPU; use sufficient GPU memory.")
  inputs = tokenizer("I love to", return_tensors="pt")
  inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
  prompt_ids = inputs["input_ids"][0].tolist()
  with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
        eos_token_id=None,
        forced_eos_token_id=None,
        stop_strings=None,
    )
  ids = output[0].tolist()
  generated = ids[len(prompt_ids) :]
  if len(generated) != 16:
    raise RuntimeError(f"Expected 16 generated tokens, got {len(generated)}")
  result = {
      "prompt": "I love to",
      "input_ids": prompt_ids,
      "generated_token_ids": generated,
      "full_generated_token_ids": ids,
      "full_text": tokenizer.decode(ids),
  }
  text = json.dumps(result, indent=2, ensure_ascii=False)
  print(text)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


if __name__ == "__main__":
  main()
