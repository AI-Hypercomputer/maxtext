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

r"""Builds a mini GLM-5.3-Flash HF checkpoint and its golden logits.

The mini checkpoint uses the exact upstream `zai-org/GLM-5.3-Flash` config.json with only:
  * fewer decoder layers (13 = 3 dense KDA + 1 DSA + 2x[3 KDA + 1 DSA] + 1 trailing KDA),
    truncating every per-layer list (`layer_types`, `mlp_layer_types`, `indexer_types`,
    `linear_attn_config.kda_layers`, `linear_attn_config.full_attn_layers`) to match;
  * fewer routed experts (16 instead of 288; `num_experts_per_tok` stays 8);
  * a small `index_topk` (8 = 2 k-pools of 4) so short prompts still exercise sparse selection.

The weights are randomly initialized with the HF reference (`transformers.models.glm5_next`, used
unmodified), constant-initialized tensors (norm weights, biases, k-pool gate/ape, A_log, ...) are
perturbed so mapping bugs on them are observable, and the result is written in exactly the
upstream on-disk layout: same tensor names, same per-tensor dtypes, and FP8 (e4m3, 128x128 blocks,
float32 `weight_scale_inv`) for exactly the tensors that are FP8 upstream. Vision and MTP weights
are omitted (text-only bring-up).

Usage (run with a transformers build that contains `glm5_next`):

  # 1. Build the checkpoint.
  python tests/assets/logits_generation/glm5_next_mini_checkpoint.py create \
      --output-dir /dev/shm/glm5.3-flash-mini

  # 2. Generate golden logits from the saved checkpoint (loads it back through HF).
  python tests/assets/logits_generation/glm5_next_mini_checkpoint.py golden \
      --checkpoint-dir /dev/shm/glm5.3-flash-mini \
      --output-path /dev/shm/golden_glm5.3-flash-mini.jsonl
"""

import argparse
import copy
import json
import math
import os
import re
import shutil
import struct
import tempfile
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ID = "zai-org/GLM-5.3-Flash"
NUM_LAYERS = 13
NUM_EXPERTS = 16
INDEX_TOPK = 8
FP8_BLOCK = 128
FP8_MAX = 448.0
SHARD_BYTES = 2 * 1024**3

AUX_FILES = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

# ~400 tokens: longer than the KDA chunk size (64) and far longer than index_topk (8), so both the
# chunked KDA recurrence and the k-pool sparse selection (pools + tail) are exercised.
DEFAULT_PROMPT = (
    "The history of mathematics spans thousands of years, beginning with counting and simple arithmetic "
    "used by early civilizations for trade, astronomy, and construction. The Babylonians developed a "
    "base-sixty number system and solved quadratic equations, while Egyptian scribes computed areas and "
    "volumes for the pyramids. Greek mathematicians such as Euclid, Archimedes, and Apollonius introduced "
    "rigorous proofs, axiomatic geometry, and early ideas of limits. In India, mathematicians formalized "
    "zero as a number and refined the decimal place-value system, which later spread through the Islamic "
    "world, where scholars like al-Khwarizmi wrote foundational texts on algebra. During the European "
    "Renaissance, symbolic notation matured, and the seventeenth century saw Newton and Leibniz "
    "independently invent calculus, transforming physics and engineering. The eighteenth and nineteenth "
    "centuries brought Euler, Gauss, Riemann, and Cauchy, who deepened analysis, number theory, and "
    "complex functions, while Galois and Abel uncovered the structure behind polynomial equations. In the "
    "twentieth century, Hilbert posed influential problems, Godel proved his incompleteness theorems, and "
    "Turing formalized computation, laying the groundwork for computer science. Today, mathematics "
    "continues to expand through fields such as topology, category theory, cryptography, and machine "
    "learning, where linear algebra, probability, and optimization drive modern artificial intelligence "
    "systems that can read, write, translate, and reason about the world. Researchers now train enormous "
    "neural networks on clusters of accelerators, carefully checking that every implementation of a model "
    "produces the same numbers as its reference, layer by layer and logit by logit, before trusting it."
)


def _upstream_config() -> dict:
  with open(hf_hub_download(REPO_ID, "config.json"), encoding="utf-8") as f:
    return json.load(f)


def mini_config(config: dict) -> dict:
  """Applies the only allowed edits (layers, experts, index_topk) to the upstream config."""
  config = copy.deepcopy(config)
  tc = config["text_config"]
  tc["num_hidden_layers"] = NUM_LAYERS
  for key in ("layer_types", "mlp_layer_types", "indexer_types"):
    tc[key] = tc[key][:NUM_LAYERS]
  lac = tc["linear_attn_config"]
  lac["kda_layers"] = [i for i in lac["kda_layers"] if i < NUM_LAYERS]
  lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i < NUM_LAYERS]
  tc["n_routed_experts"] = NUM_EXPERTS
  assert tc["num_experts_per_tok"] < NUM_EXPERTS
  tc["index_topk"] = INDEX_TOPK
  assert INDEX_TOPK % tc["index_kpool"] == 0
  return config


def _read_remote_header(filename: str) -> dict:
  url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
  req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
  with urllib.request.urlopen(req) as r:
    n = struct.unpack("<Q", r.read())[0]
  req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
  with urllib.request.urlopen(req) as r:
    header = json.loads(r.read())
  header.pop("__metadata__", None)
  return header


def _keep_upstream_key(key: str) -> bool:
  """Whether an upstream tensor name belongs to the mini (text-only, no MTP) checkpoint."""
  if key.startswith("model.visual."):
    return False
  m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
  if m and int(m.group(1)) >= NUM_LAYERS:
    return False
  m = re.search(r"\.experts\.(\d+)\.", key)
  if m and int(m.group(1)) >= NUM_EXPERTS:
    return False
  return True


def upstream_tensor_specs() -> dict:
  """Returns {name: (dtype, shape)} for every upstream tensor kept in the mini checkpoint."""
  with open(hf_hub_download(REPO_ID, "model.safetensors.index.json"), encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  wanted = {k: v for k, v in weight_map.items() if _keep_upstream_key(k)}
  specs = {}
  for filename in sorted(set(wanted.values())):
    for k, v in _read_remote_header(filename).items():
      if k in wanted:
        specs[k] = (v["dtype"], list(v["shape"]))
  missing = set(wanted) - set(specs)
  assert not missing, f"missing upstream headers for {sorted(missing)[:5]}"
  return specs


def expected_mini_shape(name: str, upstream_shape: list) -> list:
  """Upstream shape with the routed-expert dimension shrunk."""
  if name.endswith("mlp.gate.weight") or name.endswith("mlp.gate.e_score_correction_bias"):
    return [NUM_EXPERTS] + upstream_shape[1:]
  return upstream_shape


@torch.no_grad()
def perturb_constant_init(model: torch.nn.Module, seed: int) -> None:
  """Gives constant-initialized tensors random values so that mapping bugs are observable."""
  gen = torch.Generator().manual_seed(seed)

  def randn_like(p, std, mean=0.0):
    return (mean + std * torch.randn(p.shape, generator=gen)).to(p.dtype)

  for name, p in model.named_parameters():
    if name.startswith("model.visual"):
      continue
    if re.search(r"(layernorm|norm)\.weight$", name):  # RMSNorms, gated o_norm, indexer k_norm LayerNorm
      p.copy_(randn_like(p, 0.1, 1.0))
    elif name.endswith("k_norm.bias"):
      p.copy_(randn_like(p, 0.1))
    elif name.endswith("index_kpool_compress_ape"):
      p.copy_(randn_like(p, 0.5))
    elif name.endswith("index_kpool_compress_gate"):
      p.copy_(randn_like(p, 0.02))
    elif name.endswith("forget_gate.A_log"):
      p.copy_(randn_like(p, 0.5))
    elif re.search(r"_hc\.base$", name):
      p.copy_(randn_like(p, 0.1))
    elif re.search(r"_hc\.scale$", name):
      p.copy_(randn_like(p, 0.1, 1.0))
  for name, b in model.named_buffers():
    if name.endswith("e_score_correction_bias"):
      b.copy_(randn_like(b, 0.05))


def quantize_fp8_blockwise(w: torch.Tensor):
  """DeepSeek-style 128x128 block FP8 quantization. Returns (fp8 weight, float32 scale_inv)."""
  w = w.float()
  out_dim, in_dim = w.shape
  so, si = math.ceil(out_dim / FP8_BLOCK), math.ceil(in_dim / FP8_BLOCK)
  padded = torch.zeros(so * FP8_BLOCK, si * FP8_BLOCK)
  padded[:out_dim, :in_dim] = w
  blocks = padded.view(so, FP8_BLOCK, si, FP8_BLOCK)
  amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
  scale = amax / FP8_MAX
  q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(so * FP8_BLOCK, si * FP8_BLOCK)
  q = q[:out_dim, :in_dim].contiguous().to(torch.float8_e4m3fn)
  return q, scale.contiguous()


_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def create(output_dir: str, seed: int) -> None:
  """Builds the mini checkpoint in `output_dir`."""
  from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration  # pylint: disable=import-outside-toplevel

  upstream = _upstream_config()
  config_dict = mini_config(upstream)
  specs = upstream_tensor_specs()

  # Initialize unquantized (we quantize ourselves to mirror the upstream FP8 layout exactly).
  init_dict = {k: v for k, v in config_dict.items() if k != "quantization_config"}
  config = Glm5NextConfig.from_dict(init_dict)
  torch.manual_seed(seed)
  model = Glm5NextForConditionalGeneration._from_config(config, dtype=torch.float32)  # pylint: disable=protected-access
  model.eval()
  perturb_constant_init(model, seed)

  # save_pretrained reverses the HF weight conversions, giving upstream on-disk tensor names.
  with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(output_dir))) as tmp:
    model.save_pretrained(tmp, max_shard_size="100GB")
    del model
    tensors = {}
    for fn in sorted(os.listdir(tmp)):
      if fn.endswith(".safetensors"):
        with safe_open(os.path.join(tmp, fn), framework="pt") as f:
          for k in f.keys():
            if not k.startswith("model.visual."):
              tensors[k] = f.get_tensor(k)

  # Cast/quantize every tensor to its upstream dtype and validate names and shapes.
  out = {}
  for name, t in tensors.items():
    scale_name = name + "_scale_inv"
    assert name in specs, f"unexpected tensor not present upstream: {name}"
    dtype, shape = specs[name]
    assert list(t.shape) == expected_mini_shape(name, shape), (name, list(t.shape), shape)
    if dtype == "F8_E4M3":
      assert scale_name in specs, scale_name
      q, s = quantize_fp8_blockwise(t)
      assert list(s.shape) == specs[scale_name][1], (scale_name, list(s.shape), specs[scale_name][1])
      out[name], out[scale_name] = q, s
    else:
      out[name] = t.to(_DTYPES[dtype]).contiguous()
  missing = set(specs) - set(out)
  assert not missing, f"tensors missing from mini checkpoint: {sorted(missing)[:10]}"

  # Write shards + index in the upstream layout.
  os.makedirs(output_dir, exist_ok=True)
  shards, current, size = [], {}, 0
  for name in sorted(out):
    nbytes = out[name].numel() * out[name].element_size()
    if current and size + nbytes > SHARD_BYTES:
      shards.append(current)
      current, size = {}, 0
    current[name] = out[name]
    size += nbytes
  shards.append(current)
  weight_map, total = {}, 0
  for i, shard in enumerate(shards):
    fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(shard, os.path.join(output_dir, fn), metadata={"format": "pt"})
    for k, v in shard.items():
      weight_map[k] = fn
      total += v.numel() * v.element_size()
  with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
  with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_dict, f, indent=2)
  for fn in AUX_FILES:
    shutil.copy(hf_hub_download(REPO_ID, fn), os.path.join(output_dir, fn))
  print(f"Wrote {len(out)} tensors ({total / 1e9:.2f} GB) in {len(shards)} shards to {output_dir}")


def golden(checkpoint_dir: str, output_path: str, prompt: str, index_topks=None) -> None:
  """Loads a GLM-5.3 checkpoint through the HF reference and saves float32 golden logits.

  Works for both the mini and the full checkpoint. With `index_topks`, the model is loaded once and one golden
  file is written per value (`<output_path stem>_topk<k>.jsonl`), overriding each DSA indexer's `index_topk`
  attribute (the HF code is unchanged); without it, the checkpoint config's value is used.
  """
  from transformers import AutoTokenizer, FineGrainedFP8Config, Glm5NextForConditionalGeneration  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
  model = Glm5NextForConditionalGeneration.from_pretrained(
      checkpoint_dir,
      dtype=torch.float32,
      quantization_config=FineGrainedFP8Config(dequantize=True),
      attn_implementation="eager",
  )
  # Tensors that go through HF's weight-conversion mapping (merged conv1d, renamed forget_gate.*, hc fn) are not
  # cast by `dtype=` and stay bf16; upcast everything losslessly so the reference runs fully in float32.
  model = model.float()
  model.eval()
  input_ids = tokenizer.encode(prompt, return_tensors="pt")
  indexers = [
      layer.self_attn.indexer
      for layer in model.model.language_model.layers
      if getattr(layer.self_attn, "indexer", None) is not None
  ]
  runs = [(None, output_path)]
  if index_topks:
    stem = output_path[: -len(".jsonl")] if output_path.endswith(".jsonl") else output_path
    runs = [(k, f"{stem}_topk{k}.jsonl") for k in index_topks]
  for topk, path in runs:
    if topk is not None:
      for indexer in indexers:
        assert topk % indexer.index_kpool == 0, (topk, indexer.index_kpool)
        indexer.index_topk = topk
    with torch.no_grad():
      logits = model(input_ids=input_ids).logits[0].float().numpy()
    print(f"index_topk={topk} tokens={input_ids.shape[1]} logits={logits.shape} finite={np.isfinite(logits).all()}")
    with open(path, "w", encoding="utf-8") as f:
      f.write(json.dumps({"prompt": prompt, "tokens": input_ids[0].tolist(), "logits": logits.tolist()}) + "\n")
    print(f"Wrote golden logits to {path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)
  p_create = sub.add_parser("create")
  p_create.add_argument("--output-dir", required=True)
  p_create.add_argument("--seed", type=int, default=0)
  p_golden = sub.add_parser("golden")
  p_golden.add_argument("--checkpoint-dir", required=True)
  p_golden.add_argument("--output-path", required=True)
  p_golden.add_argument("--prompt", default=DEFAULT_PROMPT)
  p_golden.add_argument(
      "--index-topk", type=int, nargs="*", default=None, help="Write one golden file per DSA index_topk value."
  )
  args = parser.parse_args()
  if args.cmd == "create":
    create(args.output_dir, args.seed)
  else:
    golden(args.checkpoint_dir, args.output_path, args.prompt, args.index_topk)


if __name__ == "__main__":
  main()
