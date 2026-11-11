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

"""Bidirectional checkpoint converter between OLMo-core multimodal DCP and MaxText Orbax.

Supports three conversion modes for OLMo 3.5 hybrid KDA+FA LatentMoE + SigLIP2 models:

1. ``olmo_core_to_maxtext``:
   Converts an ``OLMo-core`` DCP checkpoint directory (either text-only ``step870229`` or full
   multimodal ``step3500`` / ``step22320``, handling ``module.*``, ``frozen_model.*``, and
   ``model.*`` entries) into a MaxText Orbax parameter checkpoint at ``<output_path>/0/items``.
2. ``bridge_init``:
   Builds a Stage 1 (``8K Bridge``) multimodal MaxText checkpoint from an existing text-only
   checkpoint (either ``OLMo-core`` DCP or MaxText Orbax ``items``), resetting the 6 special
   image-token embedding rows (``[100278..100283]``) via ``InitMethod.normal.init_embeddings``
   and attaching the 25-block HuggingFace ``SigLIP2-SO400M/14-384`` tower + ``attention_meanq``
   SwiGLU ``VisionConnector``.
3. ``maxtext_to_olmo_core``:
   Exports a scanned or unscanned MaxText Orbax parameter checkpoint back to ``OLMo-core`` DCP
   format (``model_and_optim/`` with ``model.<key>`` entries + ``config.json``).
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
import torch

from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.models import olmoe3_vision

log = logging.getLogger(__name__)

OLMO_CORE_CONFIG_FILE = "olmo_core_config.json"
DEFAULT_IMAGE_TOKEN_IDS: Tuple[int, ...] = (100278, 100279, 100280, 100281, 100282, 100283)
DEFAULT_VISION_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_VISION_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
DEFAULT_HF_CACHE_DIR = os.environ.get("HF_HOME") or os.path.join(
    os.path.expanduser("~"), ".cache", "hf_cache"
)


def _flatten_tree(tree: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
  out: Dict[str, Any] = {}
  for k, v in tree.items():
    path = f"{prefix}/{k}" if prefix else str(k)
    if isinstance(v, Mapping):
      out.update(_flatten_tree(v, path))
    else:
      out[path] = v
  return out


def _unflatten_tree(flat: Mapping[str, Any]) -> Dict[str, Any]:
  out: Dict[str, Any] = {}
  for path, v in flat.items():
    node = out
    *parents, leaf = path.split("/")
    for p in parents:
      node = node.setdefault(p, {})
    node[leaf] = v
  return out


def is_multimodal_olmo_config(experiment_config: Mapping[str, Any]) -> bool:
  model_cfg = experiment_config.get("model", {})
  return isinstance(model_cfg, Mapping) and "lm" in model_cfg and "vision" in model_cfg


def vision_and_connector_shapes(
    vision_cfg: Mapping[str, Any], connector_cfg: Mapping[str, Any]
) -> Dict[str, Tuple[int, ...]]:
  """Returns canonical ``vision.*`` and ``connector.*`` parameter shapes from OLMo-core config dicts."""
  emb_v = int(vision_cfg.get("image_emb_dim", 1152))
  patch = int(vision_cfg.get("image_patch_size", 14))
  num_pos = int(vision_cfg.get("image_num_pos", 729))
  n_layers = int(vision_cfg.get("image_num_layers", 25))
  n_heads = int(vision_cfg.get("image_num_heads", 16))
  n_kv = int(vision_cfg.get("image_num_key_value_heads", n_heads))
  head_dim = int(vision_cfg.get("image_head_dim", emb_v // n_heads))
  mlp_v = int(vision_cfg.get("image_mlp_dim", 4304))

  q_dim_v = n_heads * head_dim
  kv_dim_v = n_kv * head_dim

  shapes: Dict[str, Tuple[int, ...]] = {
      "vision.patch_embedding.weight": (emb_v, 3 * patch * patch),
      "vision.patch_embedding.bias": (emb_v,),
      "vision.positional_embedding": (num_pos, emb_v),
  }
  for i in range(n_layers):
    p = f"vision.blocks.{i}"
    shapes[f"{p}.attn_norm.weight"] = (emb_v,)
    shapes[f"{p}.attn_norm.bias"] = (emb_v,)
    shapes[f"{p}.ffn_norm.weight"] = (emb_v,)
    shapes[f"{p}.ffn_norm.bias"] = (emb_v,)
    shapes[f"{p}.attn.wq.weight"] = (q_dim_v, emb_v)
    shapes[f"{p}.attn.wq.bias"] = (q_dim_v,)
    shapes[f"{p}.attn.wk.weight"] = (kv_dim_v, emb_v)
    shapes[f"{p}.attn.wk.bias"] = (kv_dim_v,)
    shapes[f"{p}.attn.wv.weight"] = (kv_dim_v, emb_v)
    shapes[f"{p}.attn.wv.bias"] = (kv_dim_v,)
    shapes[f"{p}.attn.wo.weight"] = (emb_v, q_dim_v)
    shapes[f"{p}.attn.wo.bias"] = (emb_v,)
    shapes[f"{p}.ffn.w1.weight"] = (mlp_v, emb_v)
    shapes[f"{p}.ffn.w1.bias"] = (mlp_v,)
    shapes[f"{p}.ffn.w2.weight"] = (emb_v, mlp_v)
    shapes[f"{p}.ffn.w2.bias"] = (emb_v,)

  num_in = int(connector_cfg.get("num_input_layers", 2))
  c_emb = int(connector_cfg.get("image_emb_dim", emb_v))
  c_heads = int(connector_cfg.get("image_num_heads", n_heads))
  c_kv = int(connector_cfg.get("image_num_key_value_heads", c_heads))
  c_hd = int(connector_cfg.get("image_head_dim", head_dim))
  c_in = num_in * c_emb
  c_q = c_heads * c_hd
  c_kvd = c_kv * c_hd
  c_mlp = int(connector_cfg.get("mlp_hidden_size") or (4 * c_emb))
  c_out = int(connector_cfg["output_dim"])

  shapes.update({
      "connector.pooling.wq.weight": (c_q, c_in),
      "connector.pooling.wq.bias": (c_q,),
      "connector.pooling.wk.weight": (c_kvd, c_in),
      "connector.pooling.wk.bias": (c_kvd,),
      "connector.pooling.wv.weight": (c_kvd, c_in),
      "connector.pooling.wv.bias": (c_kvd,),
      "connector.pooling.wo.weight": (c_emb, c_q),
      "connector.pooling.wo.bias": (c_emb,),
      "connector.projector.w1.weight": (c_mlp, c_emb),
      "connector.projector.w2.weight": (c_out, c_mlp),
      "connector.projector.w3.weight": (c_mlp, c_emb),
  })
  return shapes


def _is_dcp_weight_key(k: str) -> bool:
  if k.startswith("__"):
    return False
  if k.endswith((".exp_avg", ".exp_avg_sq", ".step")):
    return False
  return True


def _resolve_dcp_key_for_canonical(
    canonical_key: str, metadata_keys: set[str], *, is_lm: bool
) -> Optional[str]:
  if is_lm:
    candidates = [
        f"model.lm.{canonical_key}",
        f"model.{canonical_key}",
        f"module.lm.{canonical_key}.main",
        f"module.{canonical_key}.main",
        f"frozen_model.lm.{canonical_key}",
        f"frozen_model.{canonical_key}",
    ]
  else:
    candidates = [
        f"model.{canonical_key}",
        f"module.{canonical_key}.main",
        f"frozen_model.{canonical_key}",
    ]
  for cand in candidates:
    if cand in metadata_keys:
      return cand
  return None


def load_olmo_core_checkpoint_weights(
    checkpoint_dir: str, experiment_config: Mapping[str, Any]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
  """Loads unsharded FP32 LM and Vision+Connector state dicts from an OLMo-core DCP checkpoint.

  Handles trainable flattened ``module.{lm,vision,connector}.<key>.main`` tensors, frozen
  unflattened ``frozen_model.{lm,vision,connector}.<key>`` tensors, native text ``module.<key>.main``
  tensors, and exported ``model.<key>`` checkpoints.
  """
  from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys  # type: ignore
  from olmo_core.io import join_path  # type: ignore
  from olmo_core.nn.maxtext.config import get_maxtext_config  # type: ignore
  from olmo_core.nn.maxtext.convert import olmo_core_shapes  # type: ignore

  multimodal = is_multimodal_olmo_config(experiment_config)
  lm_dict = experiment_config["model"]["lm"] if multimodal else experiment_config["model"]
  mt_lm_cfg = get_maxtext_config(lm_dict)
  lm_shapes = olmo_core_shapes(mt_lm_cfg)
  vis_shapes: Dict[str, Tuple[int, ...]] = {}
  if multimodal:
    vis_shapes = vision_and_connector_shapes(
        experiment_config["model"]["vision"], experiment_config["model"]["connector"]
    )

  model_and_optim = join_path(checkpoint_dir, "model_and_optim")
  metadata = get_checkpoint_metadata(model_and_optim).state_dict_metadata
  weight_keys = {k for k in metadata if _is_dcp_weight_key(k)}

  key_map: Dict[str, str] = {}
  missing: List[str] = []
  for k in lm_shapes:
    resolved = _resolve_dcp_key_for_canonical(k, weight_keys, is_lm=True)
    if resolved is None:
      missing.append(f"lm.{k}" if multimodal else k)
    else:
      key_map[k] = resolved
  for k in vis_shapes:
    resolved = _resolve_dcp_key_for_canonical(k, weight_keys, is_lm=False)
    if resolved is None:
      missing.append(k)
    else:
      key_map[k] = resolved

  if missing:
    raise KeyError(f"{model_and_optim} is missing {len(missing)} expected parameters: {missing[:10]}")

  used_dcp_keys = set(key_map.values())
  extra = sorted(weight_keys - used_dcp_keys)
  if extra:
    raise KeyError(f"{model_and_optim} has {len(extra)} unexpected weight keys: {extra[:10]}")

  all_shapes = {**lm_shapes, **vis_shapes}
  for canonical, shape in all_shapes.items():
    dcp_k = key_map[canonical]
    stored_shape = tuple(metadata[dcp_k].size)  # type: ignore[union-attr]
    if stored_shape not in (shape, (int(np.prod(shape)),)):
      raise ValueError(f"{dcp_k}: stored shape {stored_shape} does not match expected {shape}")

  ordered_canonical = list(all_shapes.keys())
  ordered_dcp = [key_map[k] for k in ordered_canonical]
  log.info("Loading %d FP32 parameter tensors from %s", len(ordered_dcp), model_and_optim)
  loaded_tensors = list(load_keys(model_and_optim, ordered_dcp))

  lm_state: Dict[str, torch.Tensor] = {}
  vis_state: Dict[str, torch.Tensor] = {}
  for idx, canonical in enumerate(ordered_canonical):
    tensor = loaded_tensors[idx]
    loaded_tensors[idx] = None  # type: ignore[call-overload]
    if tensor.dtype != torch.float32:
      tensor = tensor.float()
    reshaped = tensor.reshape(all_shapes[canonical]).contiguous()
    del tensor
    if canonical in lm_shapes:
      lm_state[canonical] = reshaped
    else:
      vis_state[canonical] = reshaped
  del loaded_tensors
  gc.collect()
  try:
    import ctypes  # type: ignore

    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:
    pass

  return lm_state, vis_state


def load_maxtext_params_any(items_dir: str) -> Dict[str, torch.Tensor]:
  """Loads a MaxText Orbax ``items`` directory (auto-detecting OCDBT/Zarr3 vs non-OCDBT) as FP32."""
  path = epath.Path(items_dir)
  has_ocdbt = (path / "manifest.ocdbt").exists() or (path / "ocdbt.process_0").exists()
  has_zarr3 = (path / "zarr.json").exists() or has_ocdbt

  last_err: Optional[Exception] = None
  for use_ocdbt, use_zarr3 in [(has_ocdbt, has_zarr3), (True, True), (False, False)]:
    try:
      ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3))
      tree = ckptr.metadata(path).item_metadata.tree
      if "params" not in tree:
        raise KeyError(f"{items_dir} has no 'params' entry; found {sorted(tree)}")
      sub_tree = {"params": tree["params"]}
      mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("x",))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      restore_args = jax.tree_util.tree_map(
          lambda _: ocp.ArrayRestoreArgs(sharding=sharding, dtype=np.float32),
          sub_tree,
          is_leaf=lambda x: hasattr(x, "shape"),
      )
      restored = ckptr.restore(
          path,
          args=ocp.args.PyTreeRestore(item=sub_tree, restore_args=restore_args, partial_restore=True),
      )
      params = restored["params"]["params"] if "params" in restored["params"] else restored["params"]
      return {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in _flatten_tree(params).items()}
    except Exception as err:  # pylint: disable=broad-except
      last_err = err
  raise RuntimeError(f"Failed to restore MaxText params from {items_dir}: {last_err}") from last_err


def _infer_olmoe3_n_layers_from_unscanned(params: Mapping[str, Any]) -> int:
  from olmo_core.nn.maxtext.convert import split_layer_key  # type: ignore

  max_layer = -1
  for k in params:
    split = split_layer_key(k, "decoder/layers_", "/")
    if split is not None:
      max_layer = max(max_layer, split[0])
  if max_layer < 0:
    raise ValueError("Could not infer decoder layer count from unscanned MaxText parameters.")
  return max_layer + 1


def _build_default_multimodal_config(
    lm_config_dict: Optional[Mapping[str, Any]],
    *,
    d_model: int,
    vocab_size: int = 100352,
    n_layers: int = 40,
    vision_num_layers: int = 25,
    connector_mlp_hidden_size: int = 9728,
    image_patch_token_id: int = 100280,
) -> Dict[str, Any]:
  vision_dict = {
      "name": "siglip",
      "image_default_input_size": [378, 378],
      "image_patch_size": 14,
      "image_emb_dim": 1152,
      "image_num_heads": 16,
      "image_num_key_value_heads": 16,
      "image_num_layers": vision_num_layers,
      "image_head_dim": 72,
      "image_mlp_dim": 4304,
      "image_mlp_activations": "gelu_pytorch_tanh",
      "image_num_pos": 729,
      "image_norm_eps": 1e-6,
      "attention_dropout": 0.0,
      "residual_dropout": 0.0,
      "use_cls_token": False,
      "patch_embedding_bias": True,
      "use_pre_ln": False,
      "block": {"name": "standard", "_CLASS_": "olmo_core.nn.vision.config.VisionBlockConfig"},
      "initializer_range": 0.02,
      "dtype": "float32",
      "_CLASS_": "olmo_core.nn.vision.config.VisionEncoderConfig",
  }
  connector_dict = {
      "image_emb_dim": 1152,
      "image_num_heads": 16,
      "image_num_key_value_heads": 16,
      "image_head_dim": 72,
      "output_dim": d_model,
      "num_input_layers": 2,
      "pooling_type": "attention_meanq",
      "pooling_attention_mask": True,
      "projector_type": "mlp",
      "mlp_hidden_size": connector_mlp_hidden_size,
      "initializer_range": 0.02,
      "dtype": "float32",
      "_CLASS_": "olmo_core.nn.vision.connector.VisionConnectorConfig",
  }
  if lm_config_dict is None:
    lm_config_dict = {
        "d_model": d_model,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
    }
  return {
      "model": {
          "lm": copy.deepcopy(dict(lm_config_dict)),
          "vision": vision_dict,
          "connector": connector_dict,
          "image_patch_token_id": image_patch_token_id,
          "vit_layers": [24, 18],
          "sync_vit_crops": True,
          "loss_chunk_size": 4096,
          "compile_loss": False,
          "_CLASS_": "olmo_core.nn.vision.multimodal.MultimodalLMConfig",
      }
  }


def reset_image_embedding_rows(
    embedding_weight: torch.Tensor,
    image_token_ids: Sequence[int] = DEFAULT_IMAGE_TOKEN_IDS,
    *,
    seed: int = 6198,
    std: float = 0.02,
) -> torch.Tensor:
  """Re-initializes image-special token rows in FP32 matching ``MultimodalOLMoDDPTrainModule.reset_image_token_rows``."""
  from olmo_core.nn.transformer.init import InitMethod  # type: ignore

  vocab_size, d_model = int(embedding_weight.shape[0]), int(embedding_weight.shape[1])
  ids = [int(i) for i in image_token_ids]
  if not ids or len(set(ids)) != len(ids):
    raise ValueError(f"image_token_ids must be non-empty unique IDs, got {ids}")
  if min(ids) < 0 or max(ids) >= vocab_size:
    raise ValueError(f"image_token_ids {ids} out of range [0, {vocab_size})")

  gen = torch.Generator(device="cpu").manual_seed(int(seed))
  emb_rows = torch.nn.Embedding(len(ids), d_model, device="cpu", dtype=torch.float32)
  InitMethod.normal.init_embeddings(
      emb_rows,
      d_model=d_model,
      embed_scale=math.sqrt(d_model),
      std=float(std),
      generator=gen,
  )
  out = embedding_weight.detach().clone().to(torch.float32)
  row_idx = torch.tensor(ids, device="cpu", dtype=torch.long)
  out.index_copy_(0, row_idx, emb_rows.weight)
  return out


def build_bridge_vision_and_connector_state(
    *,
    d_model: int,
    vision_model_id: str = DEFAULT_VISION_MODEL_ID,
    vision_revision: Optional[str] = DEFAULT_VISION_REVISION,
    hf_cache_dir: Optional[str] = DEFAULT_HF_CACHE_DIR,
    vision_num_layers: int = 25,
    connector_mlp_hidden_size: int = 9728,
    seed: int = 6198,
    initializer_range: float = 0.02,
    skip_hf_download_for_test: bool = False,
) -> Dict[str, torch.Tensor]:
  """Loads 25-block HF SigLIP2 weights and initializes the OLMo 3.5 VisionConnector in FP32."""
  from olmo_core.config import DType as OlmoDType  # type: ignore
  from olmo_core.nn.vision import siglip_state_dict_to_vision_encoder  # type: ignore
  from olmo_core.nn.vision.config import VisionEncoderConfig  # type: ignore
  from olmo_core.nn.vision.connector import (  # type: ignore
      ImagePoolingType,
      ImageProjectorType,
      VisionConnectorConfig,
  )

  vit_cfg = VisionEncoderConfig.siglip2_so400m_patch14_378()
  vit_cfg.image_num_layers = int(vision_num_layers)
  vit_cfg.dtype = OlmoDType.float32

  if skip_hf_download_for_test:
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(int(seed) + 1)
      pt_vit = vit_cfg.build(init_device="cpu")
    vision_state = {k: v.detach().clone().to(torch.float32) for k, v in pt_vit.state_dict().items()}
  else:
    from transformers import SiglipVisionModel  # type: ignore

    hf_vit = SiglipVisionModel.from_pretrained(
        vision_model_id,
        revision=vision_revision,
        cache_dir=hf_cache_dir,
        dtype=torch.float32,
    )
    vision_state = siglip_state_dict_to_vision_encoder(
        hf_vit.state_dict(), n_blocks=int(vision_num_layers)
    )

  con_cfg = VisionConnectorConfig.from_vision_encoder(
      vit_cfg,
      output_dim=int(d_model),
      num_input_layers=2,
      pooling_type=ImagePoolingType.attention_meanq,
      pooling_attention_mask=True,
      projector_type=ImageProjectorType.mlp,
      mlp_hidden_size=int(connector_mlp_hidden_size),
      initializer_range=float(initializer_range),
  )
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(int(seed))
    pt_con = con_cfg.build(init_device="cpu")

  combined: Dict[str, torch.Tensor] = {}
  for k, v in vision_state.items():
    combined[f"vision.{k}"] = v.detach().clone().to(torch.float32)
  for k, v in pt_con.state_dict().items():
    combined[f"connector.{k}"] = v.detach().clone().to(torch.float32)
  return combined


def _convert_vision_to_maxtext_with_validation(
    vis_state: Mapping[str, torch.Tensor], *, num_layers: int, validate: bool
) -> Dict[str, torch.Tensor]:
  np_vis = {k: v.detach().cpu().numpy() for k, v in vis_state.items()}
  mt_vis_np = olmoe3_vision.olmo_core_vision_to_maxtext_params(np_vis, num_layers=num_layers)
  if validate:
    rt_np = olmoe3_vision.maxtext_vision_to_olmo_core_params(mt_vis_np, num_layers=num_layers)
    if set(rt_np.keys()) != set(np_vis.keys()):
      raise AssertionError(
          f"Vision/Connector key mismatch on round-trip: {set(rt_np) ^ set(np_vis)}"
      )
    max_diff = max(float(np.max(np.abs(rt_np[k] - np_vis[k]))) for k in np_vis)
    if max_diff != 0.0:
      raise AssertionError(f"Non-zero Vision/Connector round-trip diff: {max_diff}")
  return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in mt_vis_np.items()}


def _scan_params_consume(
    params: Dict[str, torch.Tensor], layout: Any, n_layers: int
) -> Dict[str, torch.Tensor]:
  """Memory-frugal scan_params that consumes input entries incrementally to avoid 2x RAM spikes."""
  from olmo_core.nn.maxtext.convert import split_layer_key  # type: ignore

  if n_layers % layout.cycle:
    raise ValueError(f"n_layers={n_layers} isn't a multiple of the scan cycle ({layout.cycle})")
  n_unrolled = 1 if layout.unrolled is not None else 0
  n_stacked = n_layers // layout.cycle - n_unrolled
  out: Dict[str, torch.Tensor] = {}
  per_slot: Dict[str, list] = {}
  for key in list(params.keys()):
    value = params.pop(key)
    split = split_layer_key(key, "decoder/layers_", "/")
    if split is None:
      out[key] = value
      continue
    layer, rest = split
    cycle, slot = divmod(layer, layout.cycle)
    if cycle < n_unrolled:
      out[f"{layout.unrolled}/{layout.slot}{slot}/{rest}"] = value
    else:
      stacked_key = f"{layout.stacked}/{layout.slot}{slot}/{rest}"
      per_slot.setdefault(stacked_key, [None] * n_stacked)[cycle - n_unrolled] = value
  for key in list(per_slot.keys()):
    values = per_slot.pop(key)
    if any(v is None for v in values):
      raise KeyError(f"{key} is missing from some cycles")
    out[key] = torch.stack(values, dim=1)
    values.clear()
  gc.collect()
  return out


def _save_maxtext_and_config(
    output_path: str,
    mt_params: Dict[str, torch.Tensor],
    experiment_config: Mapping[str, Any],
    *,
    use_ocdbt: bool = False,
    use_zarr3: bool = False,
    checkpoint_storage_concurrent_gb: int = 2,
) -> None:
  from flax.training import train_state  # type: ignore
  from olmo_core.io import normalize_path  # type: ignore
  from olmo_core.nn.maxtext.checkpoint import write_json  # type: ignore
  from orbax.checkpoint import v1 as ocp_v1  # type: ignore
  from maxtext.common import checkpoint_context, checkpointing  # type: ignore
  from maxtext.utils import gcs_utils  # type: ignore

  nested: Dict[str, Any] = {}
  for path in list(mt_params.keys()):
    v = mt_params.pop(path)
    arr = np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32)
    del v
    node = nested
    *parents, leaf = path.split("/")
    for p in parents:
      node = node.setdefault(p, {})
    node[leaf] = arr
  gc.collect()
  validated_path = gcs_utils.mkdir_and_check_permissions(output_path)
  ctx = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      ocdbt_target_data_file_size_bytes=256 * 1024 * 1024,
      checkpoint_storage_concurrent_gb=int(checkpoint_storage_concurrent_gb),
      enable_continuous_checkpointing=True,
      partial_load=True,
  )
  # Avoid 2x host RAM spike from Orbax NumpyHandler deepcopying multi-GB host arrays on synchronous save.
  ctx.memory.deepcopy_host_arrays = False
  ctx.memory.transfer_concurrent_bytes = int(checkpoint_storage_concurrent_gb) * 10**9
  checkpoint_manager = ocp_v1.training.Checkpointer(
      validated_path,
      context=ctx,
      save_decision_policy=checkpoint_context.build_save_decision_policy(
          save_interval_steps=1,
          enable_continuous_checkpointing=True,
      ),
      preservation_policy=checkpoint_context.build_preservation_policy(max_to_keep=10),
  )
  checkpoint_manager.use_async = False  # type: ignore[attr-defined]
  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": nested}, tx=None, opt_state={}  # type: ignore
  )
  checkpointing.save_checkpoint(checkpoint_manager, 0, state_new)
  checkpointing.wait_until_finished(checkpoint_manager)
  del state_new, nested, checkpoint_manager
  gc.collect()
  try:
    import ctypes  # type: ignore

    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:
    pass
  cfg_dst = f"{normalize_path(output_path).rstrip('/')}/{OLMO_CORE_CONFIG_FILE}"
  write_json(cfg_dst, dict(experiment_config), save_overwrite=True)


def convert_olmo_core_to_maxtext(
    checkpoint_path: str,
    output_path: str,
    *,
    olmo_config_path: Optional[str] = None,
    scan_layers: bool = True,
    validate: bool = True,
    use_ocdbt: bool = False,
    use_zarr3: bool = False,
) -> Dict[str, Any]:
  """Converts an OLMo-core text or multimodal DCP checkpoint into MaxText Orbax ``<output_path>/0/items``."""
  from olmo_core.nn.maxtext.checkpoint import (  # type: ignore
      load_olmo_core_config,
      read_json,
      unscan_params,
  )
  from olmo_core.nn.maxtext.config import get_maxtext_config  # type: ignore
  from olmo_core.nn.maxtext.convert import _convert  # type: ignore

  exp_cfg = (
      read_json(olmo_config_path)
      if olmo_config_path is not None
      else load_olmo_core_config(checkpoint_path)
  )
  multimodal = is_multimodal_olmo_config(exp_cfg)
  lm_cfg_dict = exp_cfg["model"]["lm"] if multimodal else exp_cfg["model"]
  mt_lm_cfg = get_maxtext_config(lm_cfg_dict)

  lm_state, vis_state = load_olmo_core_checkpoint_weights(checkpoint_path, exp_cfg)
  n_lm_tensors = len(lm_state)
  n_vis_tensors = len(vis_state)
  total_lm_bytes = sum(t.numel() * t.element_size() for t in lm_state.values())
  # Keep reference copy only on small (<512 MiB) test models; on multi-GiB models `_convert(..., consume=True, round_trip=True)`
  # already validates exact 0.0 round-trip per tensor while freeing `lm_state` in place.
  lm_ref = {k: v.clone() for k, v in lm_state.items()} if (validate and scan_layers and total_lm_bytes < 512 * 1024 * 1024) else None

  mt_lm_params = _convert(mt_lm_cfg, lm_state, to_maxtext=True, consume=True, round_trip=validate)
  lm_state.clear()
  del lm_state
  gc.collect()

  mt_vis_params: Dict[str, torch.Tensor] = {}
  if multimodal:
    n_vit_layers = int(exp_cfg["model"]["vision"].get("image_num_layers", 25))
    mt_vis_params = _convert_vision_to_maxtext_with_validation(
        vis_state, num_layers=n_vit_layers, validate=validate
    )
    vis_state.clear()
  del vis_state
  gc.collect()

  if scan_layers:
    mt_lm_params = _scan_params_consume(mt_lm_params, mt_lm_cfg.scan_layout, mt_lm_cfg.n_layers)
    if lm_ref is not None:
      unscanned_rt = unscan_params(mt_lm_params, mt_lm_cfg.scan_layout)
      rt_lm = _convert(mt_lm_cfg, unscanned_rt, to_maxtext=False, consume=False, round_trip=False)
      max_scan_diff = max(float((rt_lm[k] - lm_ref[k]).abs().max().item()) for k in lm_ref)
      if max_scan_diff != 0.0:
        raise AssertionError(f"Non-zero scanned LM round-trip diff: {max_scan_diff}")
      del lm_ref, unscanned_rt, rt_lm

  mt_all = mt_lm_params
  mt_all.update(mt_vis_params)
  mt_vis_params.clear()
  n_maxtext_leaves = len(mt_all)
  _save_maxtext_and_config(
      output_path, mt_all, exp_cfg, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
  )
  return {
      "mode": "olmo_core_to_maxtext",
      "multimodal": multimodal,
      "scan_layers": scan_layers,
      "olmo_lm_tensors": n_lm_tensors,
      "olmo_vision_connector_tensors": n_vis_tensors,
      "olmo_total_tensors": n_lm_tensors + n_vis_tensors,
      "maxtext_leaves": n_maxtext_leaves,
      "roundtrip_max_abs_diff": 0.0 if validate else None,
      "output_items_path": f"{output_path.rstrip('/')}/0/items",
  }


def init_bridge_maxtext_checkpoint(
    output_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    maxtext_items_path: Optional[str] = None,
    olmo_config_path: Optional[str] = None,
    vision_model_id: str = DEFAULT_VISION_MODEL_ID,
    vision_revision: Optional[str] = DEFAULT_VISION_REVISION,
    hf_cache_dir: Optional[str] = DEFAULT_HF_CACHE_DIR,
    vision_num_layers: int = 25,
    connector_mlp_hidden_size: int = 9728,
    image_token_ids: Sequence[int] = DEFAULT_IMAGE_TOKEN_IDS,
    seed: int = 6198,
    embedding_init_std: float = 0.02,
    scan_cycle: int = 8,
    scan_layers: bool = True,
    validate: bool = True,
    use_ocdbt: bool = False,
    use_zarr3: bool = False,
    skip_hf_download_for_test: bool = False,
) -> Dict[str, Any]:
  """Initializes a Stage 1 Bridge MaxText checkpoint from an OLMo-core or MaxText text checkpoint."""
  from olmo_core.nn.maxtext.checkpoint import (  # type: ignore
      is_scanned,
      load_olmo_core_config,
      read_json,
      unscan_params,
  )
  from olmo_core.nn.maxtext.config import ScanLayout, get_maxtext_config  # type: ignore
  from olmo_core.nn.maxtext.convert import _convert  # type: ignore

  if (checkpoint_path is None) == (maxtext_items_path is None):
    raise ValueError("Provide exactly one of checkpoint_path (OLMo-core DCP) or maxtext_items_path (MaxText Orbax).")

  lm_cfg_dict: Optional[Dict[str, Any]] = None
  scan_layout = ScanLayout(
      cycle=int(scan_cycle),
      stacked="decoder/scanned_blocks",
      slot="layer_",
      unrolled="decoder/layers_0",
  )

  if checkpoint_path is not None:
    base_cfg = (
        read_json(olmo_config_path)
        if olmo_config_path is not None
        else load_olmo_core_config(checkpoint_path)
    )
    if is_multimodal_olmo_config(base_cfg):
      raise ValueError(f"{checkpoint_path} is already a multimodal checkpoint; use --mode olmo_core_to_maxtext.")
    lm_cfg_dict = dict(base_cfg["model"])
    mt_lm_cfg = get_maxtext_config(lm_cfg_dict)
    lm_state, _ = load_olmo_core_checkpoint_weights(checkpoint_path, base_cfg)
    mt_lm_params = _convert(mt_lm_cfg, lm_state, to_maxtext=True, consume=True, round_trip=validate)
    lm_state.clear()
    del lm_state
    gc.collect()
    scan_layout = mt_lm_cfg.scan_layout
    n_layers = mt_lm_cfg.n_layers
    d_model = mt_lm_cfg.d_model
    vocab_size = mt_lm_cfg.vocab_size
    embedding_init_std = float(
        lm_cfg_dict.get("embedding_init_std")
        if lm_cfg_dict.get("embedding_init_std") is not None
        else lm_cfg_dict.get("init_std", embedding_init_std)
    )
  else:
    assert maxtext_items_path is not None
    raw_mt = load_maxtext_params_any(maxtext_items_path)
    mt_lm_params = {k: raw_mt.pop(k) for k in list(raw_mt.keys()) if not k.startswith("vision_encoder/")}
    raw_mt.clear()
    del raw_mt
    gc.collect()
    if is_scanned(mt_lm_params, scan_layout):
      mt_lm_params = unscan_params(mt_lm_params, scan_layout)
    n_layers = _infer_olmoe3_n_layers_from_unscanned(mt_lm_params)
    emb_shape = mt_lm_params["token_embedder/embedding"].shape
    vocab_size, d_model = int(emb_shape[0]), int(emb_shape[1])
    if olmo_config_path is not None:
      loaded_cfg = read_json(olmo_config_path)
      lm_cfg_dict = dict(loaded_cfg["model"]["lm"] if is_multimodal_olmo_config(loaded_cfg) else loaded_cfg["model"])

  # Reset the 6 special <im_*> input embedding rows deterministically (without touching logits_dense).
  mt_lm_params["token_embedder/embedding"] = reset_image_embedding_rows(
      mt_lm_params["token_embedder/embedding"],
      image_token_ids=image_token_ids,
      seed=seed,
      std=embedding_init_std,
  )

  vis_conn_state = build_bridge_vision_and_connector_state(
      d_model=d_model,
      vision_model_id=vision_model_id,
      vision_revision=vision_revision,
      hf_cache_dir=hf_cache_dir,
      vision_num_layers=vision_num_layers,
      connector_mlp_hidden_size=connector_mlp_hidden_size,
      seed=seed,
      skip_hf_download_for_test=skip_hf_download_for_test,
  )
  n_vis_conn = len(vis_conn_state)
  mt_vis_params = _convert_vision_to_maxtext_with_validation(
      vis_conn_state, num_layers=vision_num_layers, validate=validate
  )
  vis_conn_state.clear()
  del vis_conn_state

  if scan_layers:
    mt_lm_params = _scan_params_consume(mt_lm_params, scan_layout, n_layers)

  mm_exp_cfg = _build_default_multimodal_config(
      lm_cfg_dict,
      d_model=d_model,
      vocab_size=vocab_size,
      n_layers=n_layers,
      vision_num_layers=vision_num_layers,
      connector_mlp_hidden_size=connector_mlp_hidden_size,
      image_patch_token_id=int(image_token_ids[2]) if len(image_token_ids) > 2 else 100280,
  )
  mt_all = mt_lm_params
  mt_all.update(mt_vis_params)
  mt_vis_params.clear()
  n_maxtext_leaves = len(mt_all)
  _save_maxtext_and_config(
      output_path, mt_all, mm_exp_cfg, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
  )
  return {
      "mode": "bridge_init",
      "d_model": d_model,
      "vocab_size": vocab_size,
      "n_layers": n_layers,
      "scan_layers": scan_layers,
      "vision_num_layers": vision_num_layers,
      "connector_mlp_hidden_size": connector_mlp_hidden_size,
      "reset_image_token_ids": list(image_token_ids),
      "vision_connector_tensors": n_vis_conn,
      "maxtext_leaves": n_maxtext_leaves,
      "roundtrip_max_abs_diff": 0.0 if validate else None,
      "output_items_path": f"{output_path.rstrip('/')}/0/items",
  }


def convert_maxtext_to_olmo_core(
    maxtext_items_path: str,
    output_path: str,
    *,
    olmo_config_path: Optional[str] = None,
    validate: bool = True,
    save_overwrite: bool = True,
) -> Dict[str, Any]:
  """Exports a scanned or unscanned MaxText Orbax checkpoint back to OLMo-core DCP ``model.*`` format."""
  from olmo_core.nn.maxtext.checkpoint import (  # type: ignore
      is_scanned,
      read_json,
      save_olmo_core_weights,
      unscan_params,
  )
  from olmo_core.nn.maxtext.config import get_maxtext_config  # type: ignore
  from olmo_core.nn.maxtext.convert import _convert  # type: ignore
  from olmo_core.nn.maxtext.convert_checkpoint import load_olmo_core_config_for_maxtext  # type: ignore

  exp_cfg = (
      read_json(olmo_config_path)
      if olmo_config_path is not None
      else load_olmo_core_config_for_maxtext(maxtext_items_path)
  )
  multimodal = is_multimodal_olmo_config(exp_cfg)
  lm_cfg_dict = exp_cfg["model"]["lm"] if multimodal else exp_cfg["model"]
  mt_lm_cfg = get_maxtext_config(lm_cfg_dict)

  mt_all = load_maxtext_params_any(maxtext_items_path)
  mt_vis = {k: mt_all.pop(k) for k in list(mt_all.keys()) if k.startswith("vision_encoder/")}
  mt_lm = mt_all

  if is_scanned(mt_lm, mt_lm_cfg.scan_layout):
    mt_lm = unscan_params(mt_lm, mt_lm_cfg.scan_layout)
    gc.collect()

  lm_state = _convert(mt_lm_cfg, mt_lm, to_maxtext=False, consume=True, round_trip=validate)
  mt_lm.clear()
  gc.collect()
  olmo_state: Dict[str, torch.Tensor] = {}
  if multimodal:
    if not mt_vis:
      raise KeyError(f"Expected vision_encoder/* parameters in multimodal MaxText checkpoint {maxtext_items_path}")
    n_vit_layers = int(exp_cfg["model"]["vision"].get("image_num_layers", 25))
    mt_vis_np = {k: v.detach().cpu().numpy() for k, v in mt_vis.items()}
    vis_np = olmoe3_vision.maxtext_vision_to_olmo_core_params(mt_vis_np, num_layers=n_vit_layers)
    if validate:
      rt_mt = olmoe3_vision.olmo_core_vision_to_maxtext_params(vis_np, num_layers=n_vit_layers)
      max_diff = max(float(np.max(np.abs(rt_mt[k] - mt_vis_np[k]))) for k in mt_vis_np)
      if max_diff != 0.0:
        raise AssertionError(f"Non-zero MaxText->OLMo-core Vision/Connector round-trip diff: {max_diff}")
    for k in list(lm_state.keys()):
      olmo_state[f"lm.{k}"] = lm_state.pop(k)
    for k, v in vis_np.items():
      olmo_state[k] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
  else:
    olmo_state = lm_state

  save_olmo_core_weights(output_path, olmo_state, exp_cfg, save_overwrite=save_overwrite)
  n_total = len(olmo_state)
  olmo_state.clear()
  gc.collect()
  return {
      "mode": "maxtext_to_olmo_core",
      "multimodal": multimodal,
      "olmo_total_tensors": n_total,
      "roundtrip_max_abs_diff": 0.0 if validate else None,
      "output_path": output_path,
  }


def _parse_bool(v: str) -> bool:
  if isinstance(v, bool):
    return v
  low = v.strip().lower()
  if low in ("1", "true", "yes", "y"):
    return True
  if low in ("0", "false", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean string, got {v!r}")


def main(argv: Optional[Sequence[str]] = None) -> None:
  parser = argparse.ArgumentParser(
      description="Convert OLMo 3.5 text or multimodal DCP checkpoints <-> MaxText Orbax."
  )
  parser.add_argument(
      "--mode",
      required=True,
      choices=("olmo_core_to_maxtext", "bridge_init", "maxtext_to_olmo_core"),
      help="Conversion mode.",
  )
  parser.add_argument("--checkpoint_path", default=None, help="Input OLMo-core DCP step directory.")
  parser.add_argument("--maxtext_items_path", default=None, help="Input MaxText Orbax .../items directory.")
  parser.add_argument("--olmo_config_path", default=None, help="Optional override path to OLMo-core config.json.")
  parser.add_argument("--output_path", required=True, help="Destination checkpoint directory (local or gs://).")
  parser.add_argument("--vision_model_id", default=DEFAULT_VISION_MODEL_ID)
  parser.add_argument("--vision_revision", default=DEFAULT_VISION_REVISION)
  parser.add_argument("--hf_cache_dir", default=DEFAULT_HF_CACHE_DIR)
  parser.add_argument("--vision_num_layers", type=int, default=25)
  parser.add_argument("--connector_mlp_hidden_size", type=int, default=9728)
  parser.add_argument(
      "--image_token_ids",
      default=",".join(str(i) for i in DEFAULT_IMAGE_TOKEN_IDS),
      help="Comma-separated image special token IDs to reset in bridge_init mode.",
  )
  parser.add_argument("--seed", type=int, default=6198)
  parser.add_argument("--embedding_init_std", type=float, default=0.02)
  parser.add_argument("--scan_cycle", type=int, default=8)
  parser.add_argument("--scan_layers", type=_parse_bool, default=True)
  parser.add_argument("--validate", type=_parse_bool, default=True)
  parser.add_argument("--use_ocdbt", type=_parse_bool, default=False)
  parser.add_argument("--use_zarr3", type=_parse_bool, default=False)
  parser.add_argument("--save_overwrite", type=_parse_bool, default=True)
  args = parser.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  if args.mode == "olmo_core_to_maxtext":
    if not args.checkpoint_path:
      parser.error("--checkpoint_path is required for --mode olmo_core_to_maxtext")
    summary = convert_olmo_core_to_maxtext(
        args.checkpoint_path,
        args.output_path,
        olmo_config_path=args.olmo_config_path,
        scan_layers=args.scan_layers,
        validate=args.validate,
        use_ocdbt=args.use_ocdbt,
        use_zarr3=args.use_zarr3,
    )
  elif args.mode == "bridge_init":
    ids = [int(x.strip()) for x in args.image_token_ids.split(",") if x.strip()]
    summary = init_bridge_maxtext_checkpoint(
        args.output_path,
        checkpoint_path=args.checkpoint_path,
        maxtext_items_path=args.maxtext_items_path,
        olmo_config_path=args.olmo_config_path,
        vision_model_id=args.vision_model_id,
        vision_revision=args.vision_revision,
        hf_cache_dir=args.hf_cache_dir,
        vision_num_layers=args.vision_num_layers,
        connector_mlp_hidden_size=args.connector_mlp_hidden_size,
        image_token_ids=ids,
        seed=args.seed,
        embedding_init_std=args.embedding_init_std,
        scan_cycle=args.scan_cycle,
        scan_layers=args.scan_layers,
        validate=args.validate,
        use_ocdbt=args.use_ocdbt,
        use_zarr3=args.use_zarr3,
    )
  else:
    if not args.maxtext_items_path:
      parser.error("--maxtext_items_path is required for --mode maxtext_to_olmo_core")
    summary = convert_maxtext_to_olmo_core(
        args.maxtext_items_path,
        args.output_path,
        olmo_config_path=args.olmo_config_path,
        validate=args.validate,
        save_overwrite=args.save_overwrite,
    )

  print(summary)


if __name__ == "__main__":
  main()
