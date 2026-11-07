# Copyright 2026 Google LLC
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

"""HF PyTorch reference for the Qwen3-Omni talker: replays a MaxText talker debug dump and compares codes.

MaxText `decode.py enable_audio_output=true` writes `<audio_output_path>.talker_debug.npz` holding the exact talker
prefill inputs (projected thinker states), trailing text embeddings, MRoPE positions and the generated codes.
This script loads the real HF talker + code2wav weights (safetensors shards 13-15 only), runs
`Qwen3OmniMoeTalkerForConditionalGeneration.generate` greedily on the same inputs, decodes a waveform, and
reports how many code frames agree.

Example:
  HF_HOME=/dev/shm/hengtaoguo python tests/utils/qwen3_omni_talker_hf_reference.py \
      --dump=/dev/shm/hengtaoguo/omni_text.wav.talker_debug.npz --output=/dev/shm/hengtaoguo/omni_text_hf.wav
"""

import argparse
import json
import time
import wave

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def load_prefixed_state(model_id, prefixes, dtype):
  """Loads all tensors whose name starts with one of `prefixes` from the HF safetensors shards."""
  index_path = hf_hub_download(model_id, "model.safetensors.index.json")
  with open(index_path, encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  shards = {}
  for name, shard in weight_map.items():
    if name.startswith(prefixes):
      shards.setdefault(shard, []).append(name)
  state = {}
  for shard, names in sorted(shards.items()):
    path = hf_hub_download(model_id, shard)
    with safe_open(path, framework="pt", device="cpu") as f:
      for name in names:
        state[name] = f.get_tensor(name).to(dtype)
  return fuse_moe_experts(state)


def fuse_moe_experts(state):
  """Converts per-expert `experts.{e}.{gate,up,down}_proj.weight` into the fused 3-D tensors transformers>=5 expects.

  `experts.gate_up_proj`: `[E, 2I, H]` (gate rows first), `experts.down_proj`: `[E, H, I]`.
  """
  import re  # pylint: disable=import-outside-toplevel
  from collections import defaultdict  # pylint: disable=import-outside-toplevel

  pattern = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
  grouped = defaultdict(dict)
  for name in list(state.keys()):
    match = pattern.match(name)
    if match:
      prefix, idx, kind = match.groups()
      grouped[prefix][(int(idx), kind)] = state.pop(name)
  for prefix, tensors in grouped.items():
    num_experts = max(i for i, _ in tensors) + 1
    gate_up = torch.stack(
        [torch.cat([tensors[(e, "gate_proj")], tensors[(e, "up_proj")]], dim=0) for e in range(num_experts)]
    )
    down = torch.stack([tensors[(e, "down_proj")] for e in range(num_experts)])
    state[f"{prefix}.gate_up_proj"] = gate_up
    state[f"{prefix}.down_proj"] = down
  return state


def write_wav(path, waveform, sample_rate=24000):
  pcm = (np.clip(waveform, -1, 1) * 32767.0).astype(np.int16)
  with wave.open(path, "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes(pcm.tobytes())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dump", required=True, help="MaxText *.talker_debug.npz file")
  parser.add_argument("--output", required=True, help="Output WAV path for the HF reference")
  parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  parser.add_argument("--max_new_tokens", type=int, default=0, help="0 = same number of frames MaxText produced + 8")
  parser.add_argument("--speaker", default="ethan")
  args = parser.parse_args()
  torch.manual_seed(0)
  dtype = getattr(torch, args.dtype)

  dump = np.load(args.dump)
  mt_codes = dump["codes"]  # [num_code_groups, T]
  inputs_embeds = torch.from_numpy(dump["inputs_embeds"]).to(dtype)
  trailing = torch.from_numpy(dump["trailing_text_hidden"]).to(dtype)
  tts_pad = torch.from_numpy(dump["tts_pad_embed"]).to(dtype)
  input_ids = torch.from_numpy(dump["input_ids"]).long()
  position_ids = torch.from_numpy(dump["position_ids"]).long().permute(2, 0, 1)  # (3, B, L)
  rope_deltas = torch.from_numpy(dump["rope_deltas"]).long()

  config = Qwen3OmniMoeConfig.from_pretrained(MODEL_ID)
  talker_cfg = config.talker_config
  talker_cfg.text_config._attn_implementation = "eager"  # pylint: disable=protected-access
  talker_cfg.code_predictor_config._attn_implementation = "eager"  # pylint: disable=protected-access
  config.code2wav_config._attn_implementation = "eager"  # pylint: disable=protected-access

  start = time.time()
  state = load_prefixed_state(MODEL_ID, ("talker.", "code2wav."), dtype)
  talker = Qwen3OmniMoeTalkerForConditionalGeneration(talker_cfg).to(dtype).eval()
  missing, unexpected = talker.load_state_dict(
      {k[len("talker.") :]: v for k, v in state.items() if k.startswith("talker.")}, strict=False
  )
  print(f"talker load: missing={len(missing)} unexpected={len(unexpected)}")
  if missing or unexpected:
    raise RuntimeError(f"talker state dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
  code2wav = Qwen3OmniMoeCode2Wav(config.code2wav_config).to(dtype).eval()
  missing, unexpected = code2wav.load_state_dict(
      {k[len("code2wav.") :]: v for k, v in state.items() if k.startswith("code2wav.")}, strict=False
  )
  print(f"code2wav load: missing={len(missing)} unexpected={len(unexpected)} ({time.time() - start:.1f}s)")

  # Replay the exact MaxText positions (text-only prompts are plain arange; video prompts carry 3-D grids).
  talker.get_rope_index = lambda *a, **k: (position_ids, rope_deltas)  # type: ignore[method-assign]

  suppressed = [
      i
      for i in range(talker_cfg.text_config.vocab_size - 1024, talker_cfg.text_config.vocab_size)
      if i != talker_cfg.codec_eos_token_id
  ]
  max_new_tokens = args.max_new_tokens or (mt_codes.shape[-1] + 8)
  start = time.time()
  with torch.no_grad():
    result = talker.generate(
        inputs_embeds=inputs_embeds,
        trailing_text_hidden=trailing,
        tts_pad_embed=tts_pad,
        talker_input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=talker_cfg.codec_eos_token_id,
        repetition_penalty=1.05,
        suppress_tokens=suppressed,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
  print(f"HF talker generate: {result.sequences.shape[-1]} tokens in {time.time() - start:.1f}s")
  hf_codes = torch.stack([hid[-1] for hid in result.hidden_states if hid[-1] is not None], dim=1).transpose(1, 2)
  hf_codes = hf_codes[0].numpy()  # [num_code_groups, T]
  print(f"HF codes: {hf_codes.shape}, MaxText codes: {mt_codes.shape}")
  common = min(hf_codes.shape[-1], mt_codes.shape[-1])
  code0_match = (hf_codes[0, :common] == mt_codes[0, :common]).mean() if common else float("nan")
  frame_match = (hf_codes[:, :common] == mt_codes[:, :common]).all(axis=0).mean() if common else float("nan")
  first_mismatch = next((i for i in range(common) if hf_codes[0, i] != mt_codes[0, i]), None)
  print(
      f"code0 agreement over {common} frames: {code0_match:.3f}; full-frame agreement: {frame_match:.3f}; "
      f"first code0 mismatch at frame {first_mismatch}"
  )

  with torch.no_grad():
    wav = code2wav.chunked_decode(torch.from_numpy(hf_codes)[None].long(), chunk_size=300, left_context_size=25)
  wav = wav[0, 0].float().numpy()
  write_wav(args.output, wav)
  print(f"wrote {wav.shape[0] / 24000:.2f}s HF reference audio to {args.output}")
  np.save(args.output + ".codes.npy", hf_codes)


if __name__ == "__main__":
  main()
