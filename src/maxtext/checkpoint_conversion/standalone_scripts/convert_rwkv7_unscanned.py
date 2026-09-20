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

"""Converts a BlinkDL RWKV-7 `.pth` checkpoint to MaxText's unscanned layout.

BlinkDL publishes RWKV-7 as a raw PyTorch state dict (no HuggingFace config),
keyed like `blocks.3.att.receptance.weight`. The mapping below is the single
source of truth for RWKV-7 weight layout and is exercised by the correctness
tests, so a rename in the model must break conversion loudly.

Usage:
  python convert_rwkv7_unscanned.py \
      --checkpoint_path=rwkv7-g1d-0.1b-20260129-ctx8192.pth \
      --output_dir=/tmp/rwkv7-0.1b-maxtext

  then run MaxText with `load_parameters_path=/tmp/rwkv7-0.1b-maxtext/0/items`
  plus the config overrides the script prints.
"""

import argparse
import dataclasses

import numpy as np

from maxtext.inference.inference_utils import str2bool


@dataclasses.dataclass(frozen=True)
class Rwkv7Dims:
  """Architecture dimensions, all inferred from the checkpoint itself."""

  num_layers: int
  emb_dim: int
  mlp_dim: int
  head_size: int
  vocab_size: int
  decay_lora_rank: int
  iclr_lora_rank: int
  value_lora_rank: int
  gate_lora_rank: int

  @property
  def num_heads(self) -> int:
    return self.emb_dim // self.head_size

  def as_config_overrides(self) -> list[str]:
    """MaxText config flags describing this checkpoint."""
    return [
        f"base_emb_dim={self.emb_dim}",
        f"base_mlp_dim={self.mlp_dim}",
        f"base_num_decoder_layers={self.num_layers}",
        f"base_num_query_heads={self.num_heads}",
        f"base_num_kv_heads={self.num_heads}",
        f"head_dim={self.head_size}",
        f"vocab_size={self.vocab_size}",
        f"rwkv7_head_size={self.head_size}",
        f"rwkv7_decay_lora_rank={self.decay_lora_rank}",
        f"rwkv7_iclr_lora_rank={self.iclr_lora_rank}",
        f"rwkv7_value_lora_rank={self.value_lora_rank}",
        f"rwkv7_gate_lora_rank={self.gate_lora_rank}",
    ]


def infer_dims(state_dict) -> Rwkv7Dims:
  """Reads architecture dimensions out of a raw RWKV-7 state dict."""
  vocab_size, emb_dim = state_dict["emb.weight"].shape
  return Rwkv7Dims(
      num_layers=1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
      emb_dim=emb_dim,
      mlp_dim=state_dict["blocks.0.ffn.key.weight"].shape[0],
      head_size=state_dict["blocks.0.att.r_k"].shape[1],
      vocab_size=vocab_size,
      decay_lora_rank=state_dict["blocks.0.att.w1"].shape[1],
      iclr_lora_rank=state_dict["blocks.0.att.a1"].shape[1],
      value_lora_rank=state_dict["blocks.0.att.v1"].shape[1],
      gate_lora_rank=state_dict["blocks.0.att.g1"].shape[1],
  )


def _array(value) -> np.ndarray:
  """Accepts torch tensors or numpy arrays; always returns float32 numpy."""
  if hasattr(value, "detach"):
    value = value.detach().cpu().float().numpy()
  return np.asarray(value, np.float32)


def _norm(state_dict, prefix):
  return {"scale": _array(state_dict[f"{prefix}.weight"]), "bias": _array(state_dict[f"{prefix}.bias"])}


def _linear(state_dict, prefix):
  # PyTorch nn.Linear stores (out, in); MaxText DenseGeneral kernels are (in, out).
  return {"kernel": _array(state_dict[f"{prefix}.weight"]).T}


def _time_mix(state_dict, prefix):
  """Maps one `blocks.N.att` subtree."""

  def get(name):
    return _array(state_dict[f"{prefix}.{name}"])

  # The token-shift and bias parameters ship as (1, 1, emb_dim) so they broadcast
  # against (batch, seq, emb) activations; MaxText stores them flat and reshapes
  # where needed, which also broadcasts correctly against single-token decode.
  params = {
      name: get(name).reshape(-1) for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a")
  }
  params.update({name: get(name) for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2", "r_k")})
  params.update({name: _linear(state_dict, f"{prefix}.{name}") for name in ("receptance", "key", "value", "output")})
  params["ln_x_scale"] = _array(state_dict[f"{prefix}.ln_x.weight"])
  params["ln_x_bias"] = _array(state_dict[f"{prefix}.ln_x.bias"])
  return params


def _channel_mix(state_dict, prefix):
  return {
      "x_k": _array(state_dict[f"{prefix}.x_k"]).reshape(-1),
      "key": _linear(state_dict, f"{prefix}.key"),
      "value": _linear(state_dict, f"{prefix}.value"),
  }


def convert_rwkv7_params(state_dict) -> dict:
  """Maps a raw RWKV-7 state dict onto MaxText's unscanned NNX parameter tree.

  Args:
    state_dict: BlinkDL-format state dict (torch tensors or numpy arrays).

  Returns:
    A nested dict matching `maxtext.models.models.Transformer`'s `nnx.Param`
    tree for `decoder_block=rwkv7`, `scan_layers=false`.
  """
  dims = infer_dims(state_dict)
  layers = {}
  for lyr in range(dims.num_layers):
    block = f"blocks.{lyr}"
    layer = {
        "ln1": _norm(state_dict, f"{block}.ln1"),
        "ln2": _norm(state_dict, f"{block}.ln2"),
        "att": _time_mix(state_dict, f"{block}.att"),
        "ffn": _channel_mix(state_dict, f"{block}.ffn"),
    }
    if lyr == 0:
      # Only layer 0 normalizes the embedding output.
      layer["ln0"] = _norm(state_dict, f"{block}.ln0")
    layers[f"layers_{lyr}"] = layer

  return {
      "token_embedder": {"embedding": _array(state_dict["emb.weight"])},
      "decoder": {
          **layers,
          "decoder_norm": _norm(state_dict, "ln_out"),
          "logits_dense": _linear(state_dict, "head"),
      },
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint_path", required=True, help="Path to the BlinkDL .pth file.")
  parser.add_argument("--output_dir", required=True, help="Directory to write the Orbax checkpoint to.")
  parser.add_argument("--use_ocdbt", type=str2bool, default=True)
  parser.add_argument("--use_zarr3", type=str2bool, default=True)
  args = parser.parse_args(argv)

  import torch  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint  # pylint: disable=import-outside-toplevel

  state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
  dims = infer_dims(state_dict)
  print(f"RWKV-7 checkpoint dims: {dims}")
  print("MaxText config overrides:\n  " + "\n  ".join(dims.as_config_overrides()))

  # The same writer as MaxText's other converters: step 0 of a checkpoint
  # manager, with the weights under `params/params`, which is the layout
  # `load_params_from_path` restores from `<output_dir>/0/items`.
  save_weights_to_checkpoint(args.output_dir, convert_rwkv7_params(state_dict), 1, args.use_ocdbt, args.use_zarr3)
  print(f"Wrote {args.output_dir}; load it with load_parameters_path={args.output_dir.rstrip('/')}/0/items")


if __name__ == "__main__":
  main()
