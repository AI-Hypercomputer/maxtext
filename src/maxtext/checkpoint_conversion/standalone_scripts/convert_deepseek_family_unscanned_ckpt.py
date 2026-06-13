# Copyright 2023–2025 Google LLC
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

r"""Convert weights from a DeepSeek style model to a MaxText one.

Example cmd:

python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_unscanned_ckpt \
    --base_model_path <path/to/meta/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek2-16b

Pass --low_memory true to keep peak host RAM at O(one tensor) instead of
O(2x model size): tensors are read from the safetensors shards one at a time
and converted weights are staged in disk-backed memmaps (under TMPDIR) until
the checkpoint is written. Recommended for trillion-parameter checkpoints such
as kimi-k2.6-text, which otherwise needs ~2.5 TB of host RAM.
"""

# pylint: disable=line-too-long
# pylint: disable=unsupported-assignment-operation
# pytype: disable=unsupported-operands

import argparse
import pathlib
import os
import gc
import logging
import shutil
import tempfile
import absl

import numpy as np
import torch
import psutil
from tqdm import tqdm

from maxtext.checkpoint_conversion.standalone_scripts import convert_deepseek_family_ckpt as ds_ckpt
from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging
from safetensors import safe_open

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


class _LazyShardLoader:
  """Dict-like loader that reads one tensor at a time from safetensors shards.

  Building the index only reads the shard headers (`safe_open(...).keys()`), so
  no tensor data is resident up front. Each `__getitem__` loads (and, for
  compressed int4 experts, dequantizes) exactly one tensor from its source shard
  and keeps no reference to it, so peak memory during conversion stays at
  O(one tensor) instead of O(whole model). See
  https://github.com/AI-Hypercomputer/maxtext/issues/4071.
  """

  def __init__(self, ckpt_paths, model_params):
    base_num_decoder_layers = model_params["num_layers"]
    first_num_dense_layers = model_params["first_num_dense_layers"]
    num_experts = model_params["num_experts"]
    is_compressed = bool(model_params.get("compressed_int4", False))
    hf_key_prefix = model_params.get("hf_key_prefix", "")

    def _normalize(raw_key):
      if not hf_key_prefix:
        return raw_key
      if raw_key.startswith(hf_key_prefix):
        return raw_key[len(hf_key_prefix) :]
      return None

    self._handles = {}
    # mapped MaxText key -> (shard path, raw key within the shard, whether the
    # raw key is the base name of a packed-int4 triple to dequantize on access).
    self._index = {}
    for i, ckpt_path in enumerate(ckpt_paths):
      max_logging.log(f"Indexing checkpoint {i+1} of {len(ckpt_paths)} ...")
      f = self._open(ckpt_path)
      for raw_key in f.keys():
        key = _normalize(raw_key)
        if key is None:
          continue

        if is_compressed and key.endswith(".weight_packed"):
          base = key[: -len(".weight_packed")]
          hf_key = base + ".weight"
          parts = hf_key.split(".")
          layer = int(parts[2]) if "layers" in hf_key else 0
          if not ds_ckpt.is_key_allowed(hf_key, ds_ckpt.MTP_KEYS_TO_SKIP):
            continue
          mapped_key = ds_ckpt.hf_to_maxtext_mapping(
              layer, num_experts, first_num_dense_layers, base_num_decoder_layers
          ).get(hf_key)
          if not mapped_key:
            continue
          self._index[mapped_key] = (ckpt_path, raw_key[: -len(".weight_packed")], True)
          continue
        if is_compressed and key.endswith((".weight_scale", ".weight_shape")):
          continue

        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key else 0
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if ds_ckpt.is_key_allowed(key, ds_ckpt.MTP_KEYS_TO_SKIP):
          mapped_key = ds_ckpt.hf_to_maxtext_mapping(
              layer, num_experts, first_num_dense_layers, base_num_decoder_layers
          ).get(key)
          if mapped_key:
            self._index[mapped_key] = (ckpt_path, raw_key, False)
          else:
            # This catches keys that are allowed but missing from the mapping dictionary
            max_logging.log(f"Debug: Allowed key '{key}' (layer {layer}) has no mapping in hf_to_maxtext_mapping.")

  def _open(self, ckpt_path):
    if ckpt_path not in self._handles:
      self._handles[ckpt_path] = safe_open(ckpt_path, framework="pt", device="cpu")
    return self._handles[ckpt_path]

  def __getitem__(self, mapped_key) -> torch.Tensor:
    ckpt_path, raw_key, is_packed_int4 = self._index[mapped_key]
    f = self._open(ckpt_path)
    if is_packed_int4:
      return ds_ckpt.dequantize_pack_quantized_int4(
          f.get_tensor(raw_key + ".weight_packed"),
          f.get_tensor(raw_key + ".weight_scale"),
          f.get_tensor(raw_key + ".weight_shape").tolist(),
      )
    return f.get_tensor(raw_key)


class _LeafSpiller:
  """Spills converted weight leaves to .npy memmaps in a scratch directory.

  `spill` writes an array to disk and returns it reopened as a read-only memmap,
  so its pages are clean and the OS can evict them under memory pressure.
  `empty` pre-allocates a zero-initialized leaf on disk for incremental filling
  (e.g. per-expert stacking); `seal` flushes such a leaf and reopens it
  read-only.
  """

  def __init__(self, spill_dir):
    self._spill_dir = spill_dir
    self._num_leaves = 0

  def empty(self, shape, dtype) -> np.memmap:
    """Allocates a zero-initialized writable memmap leaf on disk."""
    leaf_path = os.path.join(self._spill_dir, f"leaf_{self._num_leaves:05d}.npy")
    self._num_leaves += 1
    return np.lib.format.open_memmap(leaf_path, mode="w+", dtype=dtype, shape=shape)

  def seal(self, leaf) -> np.memmap:
    """Flushes a writable memmap leaf and reopens it read-only."""
    leaf.flush()
    return np.lib.format.open_memmap(leaf.filename, mode="r")

  def spill(self, leaf) -> np.memmap:
    """Writes an in-memory array to disk and returns a read-only memmap view."""
    out = self.empty(leaf.shape, leaf.dtype)
    out[...] = leaf
    return self.seal(out)


def _keep(leaf):
  """Identity leaf placement used when low-memory spilling is disabled."""
  return leaf


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info, spill_dir=None) -> dict:
  """Convert Huggingface Checkpoint to Jax.

  If spill_dir is set, converted leaves are staged in disk-backed memmaps under
  that directory instead of host RAM (low-memory mode).
  """
  base_num_decoder_layers = model_params["num_layers"]
  first_num_dense_layers = model_params["first_num_dense_layers"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_emb_dim = model_params["base_emb_dim"]
  num_experts = model_params["num_experts"]
  q_lora_rank = model_params["q_lora_rank"]
  kv_lora_rank = model_params["kv_lora_rank"]
  qk_nope_head_dim = model_params["qk_nope_head_dim"]
  qk_rope_head_dim = model_params["qk_rope_head_dim"]
  v_head_dim = model_params["v_head_dim"]

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = _LazyShardLoader(ckpt_paths, model_params)
  if spill_dir is None:
    to_leaf, alloc_leaf, seal_leaf = _keep, np.zeros, _keep
  else:
    spiller = _LeafSpiller(spill_dir)
    to_leaf, alloc_leaf, seal_leaf = spiller.spill, spiller.empty, spiller.seal

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  jax_weights["decoder"]["decoder_norm"]["scale"] = to_leaf(chkpt_vars["decoder_norm.scale"].to(torch.float16).numpy())
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")
  jax_weights["decoder"]["logits_dense"]["kernel"] = to_leaf(
      chkpt_vars["logits_dense.kernel"].to(torch.float16).numpy().transpose()
  )
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")
  jax_weights["token_embedder"]["embedding"] = to_leaf(chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy())
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  layers = {
      "dense_layers": first_num_dense_layers,
      "moe_layers": base_num_decoder_layers - first_num_dense_layers,
  }
  # self attention and normalization ###############################################
  max_logging.log("Processing self attention and normalization in dense layer")
  for layer_key, layer_value in layers.items():
    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
      layer_name = f"{layer_key}_{layer_idx}"
      if layer_key == "dense_layers":
        jax_weights["decoder"].update(
            {
                layer_name: {
                    "mlp": {
                        "wi_0": {"kernel": None},
                        "wi_1": {"kernel": None},
                        "wo": {"kernel": None},
                    },
                    "self_attention": {
                        "kv_norm": {"scale": None},
                        "wkv_a": {"kernel": None},
                        "wkv_b": {"kernel": None},
                        "out": {"kernel": None},
                    },
                    "pre_self_attention_layer_norm": {"scale": None},
                    "post_self_attention_layer_norm": {"scale": None},
                },
            }
        )
      else:
        jax_weights["decoder"].update(
            {
                layer_name: {
                    "DeepSeekMoeBlock_0": {
                        "MoeBlock_0": {
                            "wi_0": None,
                            "wi_1": None,
                            "wo": None,
                            "gate": {"kernel": None},
                        },
                        "shared_experts": {
                            "wi_0": {"kernel": None},
                            "wi_1": {"kernel": None},
                            "wo": {"kernel": None},
                        },
                    },
                    "self_attention": {
                        "kv_norm": {"scale": None},
                        "wkv_a": {"kernel": None},
                        "wkv_b": {"kernel": None},
                        "out": {"kernel": None},
                    },
                    "pre_self_attention_layer_norm": {"scale": None},
                    "post_self_attention_layer_norm": {"scale": None},
                },
            }
        )
      self_attention = jax_weights["decoder"][layer_name]["self_attention"]
      pre_self_attention_layer_norm = jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"]
      post_self_attention_layer_norm = jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"]

      pre_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      post_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      kv_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.kv_norm.scale"].to(torch.float16).numpy().transpose()
      wkv_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_a.kernel"].to(torch.float16).numpy().transpose()
      wkv_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_b.kernel"].to(torch.float16).numpy().transpose()
      out = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.out.kernel"].to(torch.float16).numpy().transpose()
      if q_lora_rank != 0:
        q_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.q_norm.scale"].to(torch.float16).numpy()
        wq_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_a.kernel"].to(torch.float16).numpy().transpose()
        wq_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_b.kernel"].to(torch.float16).numpy().transpose()
      else:
        query = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"].to(torch.float16).numpy().transpose()

      # reshape to match maxtext
      wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
      out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
      if q_lora_rank != 0:
        wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
      else:
        query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

      if q_lora_rank != 0:
        self_attention.update(
            {
                "q_norm": {"scale": None},
                "wq_a": {"kernel": None},
                "wq_b": {"kernel": None},
            }
        )
      else:
        self_attention.update({"query": {"kernel": None}})

      self_attention["kv_norm"]["scale"] = to_leaf(kv_norm)
      self_attention["wkv_a"]["kernel"] = to_leaf(wkv_a)
      self_attention["wkv_b"]["kernel"] = to_leaf(wkv_b)
      self_attention["out"]["kernel"] = to_leaf(out)
      pre_self_attention_layer_norm["scale"] = to_leaf(pre_self_attention)
      post_self_attention_layer_norm["scale"] = to_leaf(post_self_attention)
      if q_lora_rank != 0:
        self_attention["q_norm"]["scale"] = to_leaf(q_norm)
        self_attention["wq_a"]["kernel"] = to_leaf(wq_a)
        self_attention["wq_b"]["kernel"] = to_leaf(wq_b)
      else:
        self_attention["query"]["kernel"] = to_leaf(query)

      jax_weights["decoder"][layer_name]["self_attention"] = self_attention
      jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
      jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"] = post_self_attention_layer_norm

  # layer weights ################################################
  max_logging.log("Processing layer weights")
  for layer_key, layer_value in layers.items():
    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
      if layer_key == "dense_layers":
        layer_name = f"{layer_key}_{layer_idx}"
        mlp = jax_weights["decoder"][layer_name]["mlp"]
        wi_0 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"].to(torch.float16).numpy().transpose()
        wi_1 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"].to(torch.float16).numpy().transpose()
        wo = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"].to(torch.float16).numpy().transpose()
        mlp["wi_0"]["kernel"] = to_leaf(wi_0)
        mlp["wi_1"]["kernel"] = to_leaf(wi_1)
        mlp["wo"]["kernel"] = to_leaf(wo)
        jax_weights["decoder"][layer_name]["mlp"] = mlp
      else:
        layer_name = f"{layer_key}_{layer_idx}"
        moe = jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]
        if q_lora_rank != 0:
          gate_bias = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
        gate = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_0 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_1 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wo = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wo.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )

        if q_lora_rank != 0:
          moe["MoeBlock_0"]["gate"]["bias"] = to_leaf(gate_bias)
        moe["MoeBlock_0"]["gate"]["kernel"] = to_leaf(gate)
        moe["shared_experts"]["wi_0"]["kernel"] = to_leaf(shared_wi_0)
        moe["shared_experts"]["wi_1"]["kernel"] = to_leaf(shared_wi_1)
        moe["shared_experts"]["wo"]["kernel"] = to_leaf(shared_wo)

        for k in tqdm(range(num_experts), desc="experts", leave=False):
          wi_0 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wi_1 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wo = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )

          if moe["MoeBlock_0"]["wi_0"] is None:
            stack_shape = (num_experts,)
            moe["MoeBlock_0"]["wi_0"] = alloc_leaf(stack_shape + wi_0.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wi_1"] = alloc_leaf(stack_shape + wi_1.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wo"] = alloc_leaf(stack_shape + wo.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wi_0"][k, ...] = wi_0
          moe["MoeBlock_0"]["wi_1"][k, ...] = wi_1
          moe["MoeBlock_0"]["wo"][k, ...] = wo

        moe["MoeBlock_0"]["wi_0"] = seal_leaf(moe["MoeBlock_0"]["wi_0"])
        moe["MoeBlock_0"]["wi_1"] = seal_leaf(moe["MoeBlock_0"]["wi_1"])
        moe["MoeBlock_0"]["wo"] = seal_leaf(moe["MoeBlock_0"]["wo"])
        jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"] = moe

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def _convert_to_jax_weights(base_model_path, model_size, mem_info, spill_dir=None) -> dict:
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText.

  Args:
      base_model_path: Path to the Hugging Face model checkpoint.
      model_size: Model size key in MODEL_PARAMS_DICT.
      mem_info: A process instance used for memory tracking.
      spill_dir: Optional scratch directory; if set, converted leaves are staged
          in disk-backed memmaps under it instead of host RAM (low-memory mode).

  Returns:
      The converted JAX weights.
  """
  model_params = ds_ckpt.MODEL_PARAMS_DICT[model_size]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info, spill_dir)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert and save DeepSeek model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--simulated_cpu_devices_count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  parser.add_argument(
      "--low_memory",
      type=str2bool,
      required=False,
      default=False,
      help="Stage converted weights in disk-backed memmaps under TMPDIR instead of host RAM, keeping peak RSS at "
      "O(one tensor) instead of O(2x model size). Needs free disk for one fp16 copy of the model; the checkpoint is "
      "saved without simulated-device sharding (the saved checkpoint is identical and topology-independent).",
  )
  args = parser.parse_args()

  if args.model_size not in ds_ckpt.MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  mem_info = psutil.Process()
  spill_dir = None
  device_count = args.simulated_cpu_devices_count
  if args.low_memory:
    spill_dir = tempfile.mkdtemp(prefix="convert_unscanned_spill_")
    max_logging.log(f"low_memory: staging converted weights in {spill_dir} (set TMPDIR to control its location)")
    if device_count > 1:
      # Simulated-device sharding would re-materialize every leaf in host RAM.
      max_logging.log("low_memory: saving without simulated-device sharding so weights stay on disk")
      device_count = 1
  try:
    save_weights_to_checkpoint(
        args.maxtext_model_path,
        _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info, spill_dir),
        device_count,
        args.use_ocdbt,
        args.use_zarr3,
    )
  finally:
    if spill_dir is not None:
      shutil.rmtree(spill_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
