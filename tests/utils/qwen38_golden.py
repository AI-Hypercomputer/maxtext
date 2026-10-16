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

"""Real-weight HF golden generation with demand-loaded original PLE rows."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextNGramEmbedding
from maxtext.checkpoint_conversion.utils.safetensors_range import RemoteSafetensors


class DemandPLE(nn.Module):
  """Reference-only embedding lookup: every value comes from the source table."""

  def __init__(self, manifest):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(1, 160), requires_grad=False)
    self.manifest = manifest
    self.rows = {}

  def forward(self, indices):
    """Fetch missing original rows and return the requested reference values."""
    missing = sorted(set(indices.reshape(-1).tolist()) - self.rows.keys())
    grouped = {}
    for row in missing:
      grouped.setdefault(row // 2500012, []).append(row)

    def fetch(item):
      shard, rows = item
      name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
      spec = self.manifest["table_tensors"][name]
      reader = RemoteSafetensors.__new__(RemoteSafetensors)
      reader.url = (
          f'https://huggingface.co/{self.manifest["model_id"]}/resolve/{self.manifest["revision"]}/{spec["file"]}'
      )
      reader.header = self.manifest["headers"][spec["file"]]["header"]
      reader.data_start = self.manifest["headers"][spec["file"]]["data_start"]
      payload = reader.read_rows(name, [row % 2500012 for row in rows])
      values = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(-1, 160).clone()
      return dict(zip(rows, values))

    with ThreadPoolExecutor(max_workers=8) as pool:
      for result in pool.map(fetch, grouped.items()):
        self.rows.update(result)
    return torch.stack([self.rows[row] for row in indices.reshape(-1).tolist()]).reshape(*indices.shape, 160).float()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mini", default="/dev/shm/hf_mini/qwen3.8-flash-next_4layers")
  parser.add_argument("--output", default="/dev/shm/qwen38_bringup")
  parser.add_argument("--decode-tokens", type=int, default=16)
  parser.add_argument("--augment-padding-only", action="store_true")
  args = parser.parse_args()
  mini = Path(args.mini)
  output = Path(args.output)
  torch.set_num_threads(16)
  cfg = Qwen4ExpTextConfig(**json.loads((mini / "config.json").read_text())["text_config"])
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  if args.augment_padding_only:
    with torch.device("meta"):
      embedding = Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, 1, 0)
    index = json.loads((mini / "model.safetensors.index.json").read_text())
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    for name in ["layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"]:
      with safe_open(mini / index["weight_map"][prefix + name], framework="pt") as f:
        setattr(embedding, name, f.get_tensor(prefix + name))
    demand = DemandPLE(json.loads((mini / "ple_source.json").read_text()))
    with safe_open(mini / "ple_compact.safetensors", framework="pt") as f:
      demand.rows = dict(zip(f.get_tensor(prefix + "row_ids").tolist(), f.get_tensor(prefix + "ngram_embedding.weight")))
    embedding.ngram_embedding = demand
    token_sequences = [
        json.loads(line)["tokens"] for line in (output / "golden_mini_logits.jsonl").read_text().splitlines()
    ]
    decode = json.loads((output / "golden_decode.json").read_text())["tokens"]
    token_sequences.extend(decode[:n] for n in range(3, len(decode) + 1))
    for tokens in token_sequences:
      embedding(torch.tensor([tokens + [0] * (32 - len(tokens))]), None)
      embedding(torch.tensor([tokens + [cfg.eos_token_id] * (32 - len(tokens))]), None)
    row_ids = sorted(demand.rows)
    save_file(
        {
            prefix + "ngram_embedding.weight": torch.stack([demand.rows[row] for row in row_ids]),
            prefix + "row_ids": torch.tensor(row_ids),
        },
        mini / "ple_compact.safetensors",
    )
    (output / "fixture.json").write_text(
        json.dumps({"compact_rows": len(row_ids), "revision": demand.manifest["revision"], "mini": str(mini)}, indent=2)
    )
    print("COMPACT PLE ROWS WITH PADDING", len(row_ids), flush=True)
    return
  with torch.device("meta"):
    model = Qwen4ExpForCausalLM(cfg)
  index = json.loads((mini / "model.safetensors.index.json").read_text())
  state = {}
  for name, file in index["weight_map"].items():
    if name.endswith(("ple_embedding.ngram_embedding.weight", "ple_embedding.row_ids")):
      continue
    with safe_open(mini / file, framework="pt") as f:
      tensor = f.get_tensor(name)
      state[name.replace("model.language_model.", "model.")] = tensor.float() if tensor.is_floating_point() else tensor
  missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
  expected = ["model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
  if missing != expected or unexpected:
    raise ValueError(f"Unexpected missing/extra HF weights: {missing}, {unexpected}")
  del state
  demand = DemandPLE(json.loads((mini / "ple_source.json").read_text()))
  model.model.layers[1].ple.ple_embedding.ngram_embedding = demand
  # Meta initialization leaves the nonpersistent rotary frequency buffer on meta.
  rotary = model.model.rotary_emb
  rotary.inv_freq, rotary.attention_scaling = rotary.compute_default_rope_parameters(cfg)
  rotary.original_inv_freq = rotary.inv_freq.clone()
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(mini)
  prompts = ["I love to", "The capital of France is", "Hello world!"]
  output.mkdir(parents=True, exist_ok=True)
  with (output / "golden_mini_logits.jsonl").open("w") as f, torch.no_grad():
    for prompt in prompts:
      ids = tokenizer(prompt, return_tensors="pt").input_ids
      logits = model(ids, use_cache=False).logits[0]
      f.write(json.dumps({"prompt": prompt, "tokens": ids[0].tolist(), "logits": logits.tolist()}) + "\n")
      print("golden", repr(prompt), ids.tolist(), "rows", len(demand.rows), flush=True)
    ids = tokenizer(prompts[0], return_tensors="pt").input_ids
    tokens = ids.clone()
    cache = None
    for step in range(args.decode_tokens):
      result = model(ids, past_key_values=cache, use_cache=True)
      cache = result.past_key_values
      ids = result.logits[:, -1:].argmax(-1)
      tokens = torch.cat([tokens, ids], dim=1)
      print("decode", step, ids.tolist(), flush=True)
    (output / "golden_decode.json").write_text(
        json.dumps({"prompt": prompts[0], "tokens": tokens[0].tolist(), "text": tokenizer.decode(tokens[0])}, indent=2)
    )
    # Include the checker padding tokens (0) and EOS boundary cases in the row fixture.
    embedding = model.model.layers[1].ple.ple_embedding
    embedding(torch.tensor([[0] * 32, [cfg.eos_token_id] * 32, [2, 3, cfg.eos_token_id, 7, 8] + [0] * 27]), None)
  row_ids = sorted(demand.rows)
  prefix = "model.language_model.layers.1.ple.ple_embedding."
  save_file(
      {
          prefix + "ngram_embedding.weight": torch.stack([demand.rows[row] for row in row_ids]),
          prefix + "row_ids": torch.tensor(row_ids),
      },
      mini / "ple_compact.safetensors",
  )
  index["weight_map"].update(
      {prefix + "ngram_embedding.weight": "ple_compact.safetensors", prefix + "row_ids": "ple_compact.safetensors"}
  )
  (mini / "model.safetensors.index.json").write_text(json.dumps(index))
  (output / "fixture.json").write_text(
      json.dumps({"compact_rows": len(row_ids), "revision": demand.manifest["revision"], "mini": str(mini)}, indent=2)
  )
  print("COMPACT PLE ROWS", len(row_ids), "bytes", len(row_ids) * 320, flush=True)


if __name__ == "__main__":
  main()
