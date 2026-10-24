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

"""Generates HF golden logits + greedy decode for Ming-3.0-audio-tiny (audio + text -> text).

Runs the publisher's remote code on CPU (fla kernels replaced, see ming_hf_reference.py)
and writes a pickle consumable by `tests.utils.forward_pass_logit_checker`:

  [{"prompt", "formatted_prompt", "tokens", "logits", "audio_values",
    "audio_embeds", "generated_ids", "generated_text"}]

Usage:
  HF_HOME=/dev/shm/$USER python3 tests/utils/ming/generate_golden.py \
      --model_path=/dev/shm/hf_mini/ming-3.0-audio-tiny_4layers \
      --audio_path=/path/to/introduce_ant_group.wav \
      --prompt="Please answer the question in the audio." \
      --output=/dev/shm/$USER/ming/golden_mini.pkl --max_new_tokens=16
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch

from tests.utils.ming import ming_hf_reference


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default=ming_hf_reference.MING_HF_MODEL_ID)
  parser.add_argument("--audio_path", required=True)
  parser.add_argument("--prompt", default="Please answer the question in the audio.")
  parser.add_argument("--output", required=True)
  parser.add_argument("--max_new_tokens", type=int, default=16)
  parser.add_argument("--no_audio", action="store_true", help="text-only prompt")
  args = parser.parse_args()

  torch.set_grad_enabled(False)
  t0 = time.time()
  processor = ming_hf_reference.load_hf_processor(args.model_path)
  model = ming_hf_reference.load_hf_model(args.model_path, dtype=torch.float32)
  print(f"loaded model in {time.time() - t0:.1f}s")
  print("rope inv_freq[:4]:", model.model.rotary_emb.inv_freq[:4].tolist())

  content = []
  if not args.no_audio:
    content.append({"type": "audio", "audio": args.audio_path})
  content.append({"type": "text", "text": args.prompt})
  messages = [{"role": "user", "content": content}]

  if args.no_audio:
    formatted = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor.tokenizer(formatted, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = dict(inputs)
    audio_values = None
  else:
    formatted = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = dict(processor(messages))
    inputs["audio_feats"] = inputs["audio_feats"].float()
    audio_values = inputs["audio_feats"].numpy()

  input_ids = inputs["input_ids"]
  print("formatted prompt:", repr(formatted))
  print("num tokens:", input_ids.shape[1], "audio tokens:", int((input_ids == model.config.audio_token_id).sum()))

  audio_embeds = None
  if audio_values is not None:
    audio_embeds = model._get_audio_features(inputs["audio_feats"], inputs["audio_feats_lengths"]).float().numpy()  # pylint: disable=protected-access

  t0 = time.time()
  out = model(**inputs, use_cache=False)
  logits = out.logits.float().numpy()[0]
  print(f"forward in {time.time() - t0:.1f}s, logits {logits.shape}")

  t0 = time.time()
  gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
  gen_ids = gen[0, input_ids.shape[1] :].tolist()
  gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
  print(f"generate in {time.time() - t0:.1f}s")
  print("HF generated ids:", gen_ids)
  print("HF generated text:", repr(gen_text))

  record = {
      "prompt": args.prompt,
      "formatted_prompt": formatted,
      "tokens": input_ids[0].numpy().astype(np.int32),
      "logits": logits.astype(np.float32),
      "audio_values": audio_values,
      "audio_embeds": audio_embeds,
      "generated_ids": gen_ids,
      "generated_text": gen_text,
  }
  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump([record], f)
  print("saved", args.output)


if __name__ == "__main__":
  main()
