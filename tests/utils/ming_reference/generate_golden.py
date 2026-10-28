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

"""Generates CPU HF golden logits + greedy decode for inclusionAI/Ming-3.0-audio-tiny.

The publisher's remote code calls GPU-only flash-linear-attention (fla) kernels, so on CPU
`chunk_kda` / `fused_recurrent_kda` are rebound to fla's naive recurrent reference and
`FusedRMSNormGated` to a float32 PyTorch RMSNorm. RoPE `inv_freq` buffers are recomputed after
loading (transformers>=5 leaves these non-persistent buffers uninitialized). Requires torch,
transformers, flash-linear-attention, openai-whisper and torchaudio.

Writes a pickle consumable by `tests.utils.forward_pass_logit_checker`:
  [{"prompt", "formatted_prompt", "tokens", "logits", "audio_values", "generated_ids", "generated_text"}]

Optionally slices a real-weight mini subset first (`--mini_dir`, `--num_layers`): the first 4
text layers (KDA+dense, KDA+MoE, KDA+MoE, MLA+MoE) form the minimal representative cycle; the
audio encoder/projector, embeddings, final norm and lm_head are kept, the MTP layer is dropped.

Example (mini 4-layer subset):
  HF_HOME=/dev/shm/$USER/huggingface python3 tests/utils/ming_audio/generate_golden.py \
      --mini_dir=/dev/shm/$USER/ming/mini --num_layers=4 \
      --audio_path=tests/assets/test_audio.wav --output=/dev/shm/$USER/ming/golden.pkl --max_new_tokens=16

Then convert `--hf_model_path=/dev/shm/$USER/ming/mini` with `maxtext.checkpoint_conversion.to_maxtext`
and run `tests.utils.forward_pass_logit_checker` with `model_name=ming-3.0-audio-tiny
base_num_decoder_layers=4 --golden_logits_path=/dev/shm/$USER/ming/golden.pkl` in fp32 with
`sparse_matmul=false`. Omit `--audio_path` for a text-only golden (`use_audio=false use_multimodal=false`).
"""

import argparse
import json
import os
import pickle
import re
import shutil
import sys

import numpy as np
import torch
from torch import nn

MING_HF_MODEL_ID = "inclusionAI/Ming-3.0-audio-tiny"


class TorchRMSNormGated(nn.Module):
  """`fla.modules.FusedRMSNormGated` reference: rmsnorm(x) * weight * act(g), float32 math."""

  def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kwargs):
    del kwargs
    super().__init__()
    self.eps = eps
    self.activation = activation
    self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

  def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    del residual, prenorm, residual_in_fp32
    xf, gf = x.float(), g.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.weight is not None:
      y = y * self.weight.float()
    y = y * torch.sigmoid(gf) if self.activation == "sigmoid" else y * gf * torch.sigmoid(gf)
    return y.to(x.dtype)


def torch_kda(  # pylint: disable=too-many-positional-arguments
    q,
    k,
    v,
    g,
    beta,
    A_log=None,  # pylint: disable=invalid-name
    dt_bias=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    lower_bound=None,
    cu_seqlens=None,
    scale=None,
    **kwargs,
):
  """CPU replacement for `fla.ops.kda.chunk_kda` / `fused_recurrent_kda` built from fla's naive references."""
  from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate  # pylint: disable=import-outside-toplevel
  from fla.ops.kda.naive import naive_recurrent_kda  # pylint: disable=import-outside-toplevel

  del kwargs  # e.g. safe_gate: only selects a kernel variant, same math.
  assert cu_seqlens is None, "varlen is not supported by the CPU reference"
  if use_gate_in_kernel:
    if lower_bound is not None:
      g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)
    else:
      g = naive_kda_gate(g, A_log, dt_bias)
  if use_qk_l2norm_in_kernel:  # fla l2norm: x / sqrt(sum(x^2) + 1e-6)
    q, k = (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + 1e-6) for x in (q, k))
  return naive_recurrent_kda(
      q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
  )


def load_hf_model(model_path):
  """Loads the full audio+text HF model on CPU in float32 with the fla kernels replaced."""
  from transformers.dynamic_module_utils import get_class_from_dynamic_module  # pylint: disable=import-outside-toplevel

  audio_cls = get_class_from_dynamic_module(
      "modeling_bailing_moe_v3_audio.BailingMoeV3AudioForConditionalGeneration", model_path
  )
  text_mod = sys.modules[sys.modules[audio_cls.__module__].BailingMoeV3Model.__module__]
  text_mod.FusedRMSNormGated = TorchRMSNormGated
  text_mod.chunk_kda = torch_kda
  text_mod.fused_recurrent_kda = torch_kda
  # Use the patched remote class directly so from_pretrained cannot re-import an unpatched copy.
  model = audio_cls.from_pretrained(
      model_path, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager"
  ).eval()
  for module in model.modules():
    if type(module).__name__ == "BailingMoeV3RotaryEmbedding":
      module.inv_freq, module.attention_scaling = module.rope_init_fn(module.config, None)
      module.original_inv_freq = module.inv_freq
  return model


def slice_mini_checkpoint(model_id, output, num_layers):
  """Saves config/tokenizer/remote code and the first `num_layers` text layers of `model_id` to `output`."""
  from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel
  from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

  if os.path.isdir(model_id):
    src = model_id
  else:
    src = snapshot_download(model_id, allow_patterns=["*.json", "*.py", "*.jinja", "model*.safetensors"])
  os.makedirs(output, exist_ok=True)
  for f in os.listdir(src):
    if f.endswith((".json", ".py", ".jinja")) and f != "model.safetensors.index.json":
      shutil.copy(os.path.join(src, f), os.path.join(output, f))

  with open(os.path.join(src, "model.safetensors.index.json"), encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]

  def keep(key):
    m = re.match(r"model\.layers\.(\d+)\.", key)
    return m is None or int(m.group(1)) < num_layers

  keys = [k for k in weight_map if keep(k)]
  tensors = {}
  for shard in sorted({weight_map[k] for k in keys}):
    with safe_open(os.path.join(src, shard), "pt") as sf:
      tensors.update({k: sf.get_tensor(k).contiguous() for k in keys if weight_map[k] == shard})
  save_file(tensors, os.path.join(output, "model.safetensors"), metadata={"format": "pt"})

  with open(os.path.join(output, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
  cfg["text_config"]["num_hidden_layers"] = num_layers
  cfg["text_config"]["num_nextn_predict_layers"] = 0
  with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
  print(f"Saved {len(tensors)} tensors with {num_layers} text layers to {output}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default=MING_HF_MODEL_ID, help="HF model ID or local checkpoint dir.")
  parser.add_argument("--mini_dir", default=None, help="If set, slice a mini subset of --model_path here and use it.")
  parser.add_argument("--num_layers", type=int, default=4, help="Text layers kept with --mini_dir.")
  parser.add_argument("--audio_path", default=None, help="Audio file; omit for a text-only prompt.")
  parser.add_argument("--prompt", default="Please answer the question in the audio.")
  parser.add_argument("--output", required=True)
  parser.add_argument("--max_new_tokens", type=int, default=16)
  args = parser.parse_args()

  model_path = args.model_path
  if args.mini_dir:
    slice_mini_checkpoint(model_path, args.mini_dir, args.num_layers)
    model_path = args.mini_dir

  from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel

  torch.set_grad_enabled(False)
  processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
  model = load_hf_model(model_path)

  content = [{"type": "audio", "audio": args.audio_path}] if args.audio_path else []
  messages = [{"role": "user", "content": content + [{"type": "text", "text": args.prompt}]}]
  formatted = processor.tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
  )
  audio_values = None
  if args.audio_path:
    inputs = dict(processor(messages))
    inputs["audio_feats"] = inputs["audio_feats"].float()
    audio_values = inputs["audio_feats"].numpy()
  else:
    inputs = dict(processor.tokenizer(formatted, return_tensors="pt"))
    inputs.pop("token_type_ids", None)
  input_ids = inputs["input_ids"]
  print("formatted prompt:", repr(formatted))
  print("num tokens:", input_ids.shape[1], "audio tokens:", int((input_ids == model.config.audio_token_id).sum()))

  logits = model(**inputs, use_cache=False).logits.float().numpy()[0]
  gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
  gen_ids = gen[0, input_ids.shape[1] :].tolist()
  gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
  print("HF generated ids:", gen_ids)
  print("HF generated text:", repr(gen_text))

  record = {
      "prompt": args.prompt,
      "formatted_prompt": formatted,
      "tokens": input_ids[0].numpy().astype(np.int32),
      "logits": logits.astype(np.float32),
      "audio_values": audio_values,
      "generated_ids": gen_ids,
      "generated_text": gen_text,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump([record], f)
  print("saved", args.output)


if __name__ == "__main__":
  main()
