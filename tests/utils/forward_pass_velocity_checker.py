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

"""Numerical forward-pass velocity checker for converted Weaver-Nano Orbax checkpoints.

Validates the converted MaxText ``WeaverOmniTransformer`` Orbax checkpoint against
the golden PyTorch reference by feeding identical text prompt tokens ``input_ids``,
noisy latents ``x_t``, and diffusion timestep ``t=500.0``, and comparing the
predicted velocity vectors ``v_pred``.

Usage:

1. Mini 1-layer parity check (max_abs_diff <= 1e-4):
  python3 -m tests.utils.forward_pass_velocity_checker src/maxtext/configs/base.yml \\
      model_name=weaver-mini-diffuser override_model_config=True \\
      base_num_decoder_layers=1 base_emb_dim=64 base_mlp_dim=128 \\
      base_num_query_heads=4 base_num_kv_heads=2 head_dim=16 vocab_size=256 \\
      mrope_section=[4,2,2] scan_layers=False dtype=float32 weight_dtype=float32 \\
      skip_jax_distributed_system=True \\
      --atol=1e-4 --timestep=500.0

2. Full 36-layer parity check (cosine_sim >= 0.9999, max_abs_diff <= 5e-4):
  python3 -m tests.utils.forward_pass_velocity_checker src/maxtext/configs/base.yml \\
      model_name=weaver-mini-diffuser override_model_config=True \\
      base_num_decoder_layers=36 base_emb_dim=64 base_mlp_dim=128 \\
      base_num_query_heads=4 base_num_kv_heads=2 head_dim=16 vocab_size=256 \\
      mrope_section=[4,2,2] scan_layers=False dtype=float32 weight_dtype=float32 \\
      skip_jax_distributed_system=True \\
      --atol=5e-4 --min_cosine_sim=0.9999 --timestep=500.0
"""

from collections.abc import Mapping
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any
from unittest import mock

import absl.logging
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.configs import pyconfig
from maxtext.models import weaver
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
import numpy as np
import orbax.checkpoint as ocp
from safetensors.numpy import save_file as save_safetensors

try:
  import torch
except ImportError:  # pragma: no cover - optional in JAX/TPU-only virtualenvs
  torch = None  # type: ignore[assignment]

absl.logging.set_verbosity(absl.logging.INFO)

_VELOCITY_GOLDEN_FILENAME = "weaver_velocity_golden_data.npz"
_DEFAULT_GCS_BUCKET = "gs://maxtext-test-assets"


def str2bool(v: str | bool) -> bool:
  """Converts a string or boolean CLI flag to bool."""
  if isinstance(v, bool):
    return v
  val = v.strip().lower()
  if val in ("yes", "true", "t", "y", "1"):
    return True
  if val in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}.")


def resolve_golden_velocity_path(golden_path: str = "") -> str:
  """Locates or downloads the golden velocity `.npz` asset from GCS."""
  if golden_path:
    if os.path.exists(golden_path):
      return golden_path
    if golden_path.startswith("gs://"):
      local_target = os.path.join("/tmp", os.path.basename(golden_path))
      if not os.path.exists(local_target):
        subprocess.run(
            ["gcloud", "storage", "cp", golden_path, local_target],
            check=True,
            timeout=60,
        )
      return local_target
    raise FileNotFoundError(f"Specified --golden_velocity_path does not exist: {golden_path}")

  tmp_path = f"/tmp/{_VELOCITY_GOLDEN_FILENAME}"
  if os.path.exists(tmp_path):
    return tmp_path

  gcs_uri = f"{_DEFAULT_GCS_BUCKET}/{_VELOCITY_GOLDEN_FILENAME}"
  max_logging.log(f"Downloading golden velocity asset from {gcs_uri} to {tmp_path}...")
  subprocess.run(
      ["gcloud", "storage", "cp", gcs_uri, tmp_path],
      check=True,
      timeout=60,
  )
  return tmp_path


def build_hf_config_from_maxtext_config(config: Any) -> dict[str, Any]:
  """Builds a Weaver HuggingFace/Diffusers config dictionary matching `config`."""
  weaver_cfg = weaver.WeaverConfig.from_maxtext_config(config)
  return {
      "architectures": ["WeaverOmniTransformer2DModel"],
      "model_type": "weaver_omni",
      "num_hidden_layers": weaver_cfg.num_hidden_layers,
      "hidden_size": weaver_cfg.hidden_size,
      "num_attention_heads": weaver_cfg.num_attention_heads,
      "num_key_value_heads": weaver_cfg.num_key_value_heads,
      "head_dim": weaver_cfg.head_dim,
      "intermediate_size": weaver_cfg.intermediate_size,
      "vocab_size": weaver_cfg.vocab_size,
      "in_channels": weaver_cfg.latent_channels,
      "patch_size": weaver_cfg.patch_size,
      "time_embed_in_channels": weaver_cfg.time_embed_in_channels,
      "timestep_scale": weaver_cfg.timestep_scale,
      "rms_norm_eps": weaver_cfg.rms_norm_eps,
      "hidden_act": weaver_cfg.hidden_act,
      "attention_bias": weaver_cfg.attention_bias,
      "qk_norm_for_text": weaver_cfg.qk_norm_for_text,
      "qk_norm_for_diffusion": weaver_cfg.qk_norm_for_diffusion,
      "use_und_k_norm_for_gen": weaver_cfg.use_und_k_norm_for_gen,
      "mrope_section": list(weaver_cfg.mrope_section),
      "rope_theta": weaver_cfg.rope_theta,
  }


def extract_hf_weights_for_layers(
    golden_data: Mapping[str, np.ndarray],
    num_layers: int,
) -> dict[str, np.ndarray]:
  """Extracts Diffusers/HF safetensors weights sliced to `num_layers` from golden data."""
  valid_layer_prefixes = tuple(f"layers.{i}." for i in range(num_layers))
  hf_weights: dict[str, np.ndarray] = {}
  for key in golden_data.keys():
    if not key.startswith("weights/"):
      continue
    hf_key = key[len("weights/") :]
    if hf_key.startswith("layers.") and not hf_key.startswith(valid_layer_prefixes):
      continue
    hf_weights[hf_key] = np.asarray(golden_data[key], dtype=np.float32)
  if not hf_weights:
    raise ValueError("No 'weights/*' entries found in golden velocity archive.")
  return hf_weights


def convert_golden_weights_to_orbax(
    config: Any,
    golden_data: Mapping[str, np.ndarray],
    orbax_output_dir: str,
    hf_model_path: str = "",
) -> str:
  """Converts golden safetensors weights to a MaxText Orbax checkpoint via `to_maxtext`."""
  weaver_cfg = weaver.WeaverConfig.from_maxtext_config(config)
  num_layers = weaver_cfg.num_hidden_layers
  hf_cfg = build_hf_config_from_maxtext_config(config)

  with tempfile.TemporaryDirectory() as tmp_hf_dir:
    if hf_model_path and os.path.isdir(hf_model_path):
      source_hf_dir = hf_model_path
    else:
      hf_weights = extract_hf_weights_for_layers(golden_data, num_layers=num_layers)
      save_safetensors(hf_weights, os.path.join(tmp_hf_dir, "model.safetensors"))
      with open(os.path.join(tmp_hf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2)
      source_hf_dir = tmp_hf_dir

    mock_hf_cfg_obj = mock.Mock()
    mock_hf_cfg_obj.to_dict.return_value = hf_cfg

    mrope_str = "[" + ",".join(str(x) for x in weaver_cfg.mrope_section) + "]"
    save_dtype_str = "bfloat16" if weaver_cfg.weight_dtype == jnp.bfloat16 else "float32"
    dtype_str = "bfloat16" if weaver_cfg.dtype == jnp.bfloat16 else "float32"

    conv_args = [
        "",
        os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml"),
        f"model_name={config.model_name}",
        "override_model_config=True",
        f"base_output_directory={orbax_output_dir}",
        "run_name=velocity_checker_conversion",
        "hardware=cpu",
        "skip_jax_distributed_system=True",
        f"scan_layers={weaver_cfg.scan_layers}",
        f"param_scan_axis={weaver_cfg.param_scan_axis}",
        f"base_num_decoder_layers={weaver_cfg.num_hidden_layers}",
        f"base_emb_dim={weaver_cfg.hidden_size}",
        f"base_num_query_heads={weaver_cfg.num_attention_heads}",
        f"base_num_kv_heads={weaver_cfg.num_key_value_heads}",
        f"head_dim={weaver_cfg.head_dim}",
        f"base_mlp_dim={weaver_cfg.intermediate_size}",
        f"vocab_size={weaver_cfg.vocab_size}",
        f"mrope_section={mrope_str}",
        f"weight_dtype={save_dtype_str}",
        f"dtype={dtype_str}",
    ]

    with mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {config.model_name: mock_hf_cfg_obj}):
      to_maxtext.main(
          args=conv_args,
          lazy_load_tensors=False,
          eager_load_method="safetensors",
          hf_model_path=source_hf_dir,
          save_dtype=save_dtype_str,
          simulated_cpu_devices_count=1,
      )

  ckpt_items_path = os.path.join(orbax_output_dir, "0", "items")
  if not os.path.exists(ckpt_items_path):
    raise FileNotFoundError(f"Expected converted Orbax checkpoint at {ckpt_items_path}")
  return ckpt_items_path


def load_orbax_weights_into_weaver_model(
    model: weaver.WeaverOmniTransformer,
    checkpoint_path: str,
) -> int:
  """Restores an Orbax checkpoint from `checkpoint_path` into `model`.

  Supports both `scan_layers=False` and `scan_layers=True`, including cross-loading
  between scanned and unscanned Orbax checkpoints.

  Args:
    model: Target `WeaverOmniTransformer` instance.
    checkpoint_path: Path to Orbax checkpoint `0/items` directory.

  Returns:
    Number of parameter leaves loaded into `model`.
  """
  checkpointer = ocp.PyTreeCheckpointer()
  restored = checkpointer.restore(checkpoint_path)
  extracted_mt = conversion_utils.detect_and_extract_checkpoint(restored)

  pure_params = nnx.state(model, nnx.Param).to_pure_dict()
  flat_leaves, treedef = jax.tree_util.tree_flatten_with_path({"params": pure_params})
  updated_leaves = []
  num_layers = model.num_hidden_layers
  scan_axis = model.param_scan_axis

  for path_tuple, leaf in flat_leaves:
    mt_key = "-".join(conversion_utils.param_key_parts_from_path(path_tuple))
    target_shape = leaf.shape
    target_dtype = leaf.dtype

    if mt_key in extracted_mt:
      arr = np.asarray(extracted_mt[mt_key])
    elif model.scan_layers and mt_key.startswith("params-scanned_layers-"):
      subkey = mt_key[len("params-scanned_layers-") :]
      per_layer_arrays = [np.asarray(extracted_mt[f"params-layers_{i}-{subkey}"]) for i in range(num_layers)]
      stack_axis = 0 if per_layer_arrays[0].ndim == 1 else scan_axis
      arr = np.stack(per_layer_arrays, axis=stack_axis)
    elif not model.scan_layers and mt_key.startswith("params-layers_"):
      rest = mt_key[len("params-layers_") :]
      layer_idx_str, subkey = rest.split("-", 1)
      layer_idx = int(layer_idx_str)
      scanned_arr = np.asarray(extracted_mt[f"params-scanned_layers-{subkey}"])
      slice_axis = 0 if scanned_arr.ndim == 2 else scan_axis
      arr = np.take(scanned_arr, indices=layer_idx, axis=slice_axis)
    else:
      raise KeyError(f"Parameter '{mt_key}' not found in Orbax checkpoint at {checkpoint_path}.")

    if np.isnan(arr).any() or np.isinf(arr).any():
      raise ValueError(f"NaN or Inf detected in restored Orbax parameter '{mt_key}'.")

    if arr.shape != target_shape:
      raise ValueError(f"Shape mismatch for parameter '{mt_key}': Orbax has {arr.shape}, model expects {target_shape}.")

    updated_leaves.append(jnp.asarray(arr, dtype=target_dtype))

  restored_tree = jax.tree_util.tree_unflatten(treedef, updated_leaves)
  nnx.update(model, restored_tree["params"])
  return len(updated_leaves)


def _np_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """NumPy float32 RMSNorm matching the PyTorch reference implementation."""
  x_f32 = x.astype(np.float32)
  var = np.mean(np.square(x_f32), axis=-1, keepdims=True)
  normed = x_f32 / np.sqrt(var + eps)
  return (normed * weight.astype(np.float32)).astype(x.dtype)


def _np_silu(x: np.ndarray) -> np.ndarray:
  """NumPy float32 SiLU matching `torch.nn.functional.silu`."""
  x_f32 = x.astype(np.float32)
  return (x_f32 / (1.0 + np.exp(-x_f32))).astype(x.dtype)


def _np_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
  """NumPy linear projection matching `torch.nn.functional.linear(x, weight, bias)`."""
  out = np.matmul(x.astype(np.float32), weight.astype(np.float32).T)
  if bias is not None:
    out = out + bias.astype(np.float32)
  return out.astype(x.dtype)


def _np_rotate_half(x: np.ndarray) -> np.ndarray:
  half = x.shape[-1] // 2
  return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _np_apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
  cos_u = np.expand_dims(cos.astype(np.float32), axis=1)
  sin_u = np.expand_dims(sin.astype(np.float32), axis=1)
  x_f32 = x.astype(np.float32)
  return ((x_f32 * cos_u) + (_np_rotate_half(x_f32) * sin_u)).astype(x.dtype)


def _np_softmax(scores: np.ndarray) -> np.ndarray:
  scores_f32 = scores.astype(np.float32)
  shifted = scores_f32 - np.max(scores_f32, axis=-1, keepdims=True)
  exp_scores = np.exp(shifted)
  return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


class PyTorchWeaverReferenceModel:
  """Reference model for WeaverOmniTransformer velocity verification using HF/PyTorch weights."""

  def __init__(self, hf_cfg: Mapping[str, Any]):
    self.hidden_size = int(hf_cfg["hidden_size"])
    self.num_heads = int(hf_cfg["num_attention_heads"])
    self.num_kv_heads = int(hf_cfg["num_key_value_heads"])
    self.head_dim = int(hf_cfg["head_dim"])
    self.intermediate_size = int(hf_cfg["intermediate_size"])
    self.num_layers = int(hf_cfg["num_hidden_layers"])
    self.vocab_size = int(hf_cfg["vocab_size"])
    self.latent_channels = int(hf_cfg.get("in_channels", 48))
    self.patch_size = int(hf_cfg.get("patch_size", 2))
    self.patch_dim = self.patch_size * self.patch_size * self.latent_channels
    self.time_embed_in_channels = int(hf_cfg.get("time_embed_in_channels", 256))
    self.timestep_scale = float(hf_cfg.get("timestep_scale", 0.001))
    self.rms_norm_eps = float(hf_cfg.get("rms_norm_eps", 1e-6))
    self.hidden_act = str(hf_cfg.get("hidden_act", "silu"))
    self.mrope_section = tuple(int(x) for x in hf_cfg.get("mrope_section", [24, 20, 20]))
    self.rope_theta = float(hf_cfg.get("rope_theta", 5000000.0))
    self.hf_weights: dict[str, np.ndarray] = {}

  def load_hf_weights(self, hf_weights: Mapping[str, np.ndarray]) -> None:
    """Loads Diffusers/HF safetensors dictionary (`(out_dim, in_dim)` layout) into the reference model."""
    self.hf_weights = {k: np.asarray(v, dtype=np.float32) for k, v in hf_weights.items()}

  def eval(self) -> "PyTorchWeaverReferenceModel":
    """No-op evaluation mode toggle for API compatibility with `torch.nn.Module`."""
    return self

  def _compute_time_embed_np(self, timesteps: np.ndarray) -> np.ndarray:
    """Computes continuous sinusoidal timestep embeddings projected through a 2-layer SiLU MLP."""
    scaled = timesteps.astype(np.float32) * self.timestep_scale
    half_dim = self.time_embed_in_channels // 2
    exponent = -math.log(10000.0) * np.arange(0, half_dim, dtype=np.float32) / half_dim
    freqs = scaled[..., None] * np.exp(exponent)
    sin_emb = np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1)
    hidden = _np_silu(
        _np_linear(
            sin_emb,
            self.hf_weights["time_embedder.linear_1.weight"],
            self.hf_weights["time_embedder.linear_1.bias"],
        )
    )
    return _np_linear(
        hidden,
        self.hf_weights["time_embedder.linear_2.weight"],
        self.hf_weights["time_embedder.linear_2.bias"],
    )

  def _compute_mrope_cos_sin_np(
      self,
      *,
      bsz: int,
      s_und: int,
      t_lat: int,
      h_patch: int,
      w_patch: int,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes 3D M-RoPE cosine/sine tables partitioned into understanding and generation tokens."""
    pos_3d_jax = weaver.build_weaver_3d_position_ids(bsz, s_und, t_lat, h_patch, w_patch)
    cos_jax, sin_jax = weaver.compute_3d_mrope_cos_sin(
        pos_3d_jax,
        head_dim=self.head_dim,
        mrope_section=self.mrope_section,
        rope_theta=self.rope_theta,
    )
    cos_all = np.asarray(cos_jax, dtype=np.float32)
    sin_all = np.asarray(sin_jax, dtype=np.float32)
    s_gen = t_lat * h_patch * w_patch
    sample_len = s_und + s_gen
    und_idx, gen_idx = [], []
    for b in range(bsz):
      base = b * sample_len
      und_idx.append(np.arange(base, base + s_und))
      gen_idx.append(np.arange(base + s_und, base + sample_len))
    u_i = np.concatenate(und_idx, axis=0)
    g_i = np.concatenate(gen_idx, axis=0)
    return cos_all[u_i], sin_all[u_i], cos_all[g_i], sin_all[g_i]

  def _mlp_np(self, x: np.ndarray, prefix: str) -> np.ndarray:
    """Runs SwiGLU or ReLU2 MLP using un-transposed PyTorch `(out_dim, in_dim)` weights."""
    up = _np_linear(x, self.hf_weights[f"{prefix}.up_proj.weight"])
    if self.hidden_act == "relu2":
      hidden = np.square(np.maximum(up, 0.0))
    else:
      gate = _np_linear(x, self.hf_weights[f"{prefix}.gate_proj.weight"])
      hidden = _np_silu(gate) * up
    return _np_linear(hidden, self.hf_weights[f"{prefix}.down_proj.weight"])

  def forward(
      self,
      input_ids: Any,
      latents: Any,
      timesteps: Any,
  ) -> Any:
    """Runs the reference forward pass and returns predicted velocity `v_pred`."""
    input_ids_np = np.asarray(input_ids, dtype=np.int64)
    latents_np = np.asarray(latents, dtype=np.float32)
    timesteps_np = np.asarray(timesteps, dtype=np.float32)

    bsz, s_und = input_ids_np.shape
    _, c_lat, t_lat, h_lat, w_lat = latents_np.shape
    p = self.patch_size
    h_patch, w_patch = h_lat // p, w_lat // p
    s_gen = t_lat * h_patch * w_patch

    und_seq = self.hf_weights["embed_tokens.weight"][input_ids_np].reshape(bsz * s_und, self.hidden_size)

    lat_reshaped = latents_np.reshape(bsz, c_lat, t_lat, h_patch, p, w_patch, p)
    patches = np.einsum("bcthpwq->bthwpqc", lat_reshaped).reshape(bsz, s_gen, self.patch_dim)
    vis_embeds = _np_linear(patches, self.hf_weights["proj_in.weight"], self.hf_weights["proj_in.bias"])
    t_embeds = np.expand_dims(self._compute_time_embed_np(timesteps_np), axis=1)
    gen_seq = (vis_embeds + t_embeds).reshape(bsz * s_gen, self.hidden_size)

    cos_und, sin_und, cos_gen, sin_gen = self._compute_mrope_cos_sin_np(
        bsz=bsz, s_und=s_und, t_lat=t_lat, h_patch=h_patch, w_patch=w_patch
    )
    repeat_kv = self.num_heads // self.num_kv_heads
    scale = 1.0 / math.sqrt(self.head_dim)
    causal_mask = np.tril(np.ones((s_und, s_und), dtype=bool))

    for i in range(self.num_layers):
      lp = f"layers.{i}"
      u_norm = _np_rms_norm(und_seq, self.hf_weights[f"{lp}.input_layernorm.weight"], self.rms_norm_eps)
      g_norm = _np_rms_norm(gen_seq, self.hf_weights[f"{lp}.input_layernorm_moe_gen.weight"], self.rms_norm_eps)

      q_u = _np_rms_norm(
          _np_linear(u_norm, self.hf_weights[f"{lp}.self_attn.to_q.weight"]).reshape(-1, self.num_heads, self.head_dim),
          self.hf_weights[f"{lp}.self_attn.norm_q.weight"],
          self.rms_norm_eps,
      )
      k_u = _np_rms_norm(
          _np_linear(u_norm, self.hf_weights[f"{lp}.self_attn.to_k.weight"]).reshape(
              -1, self.num_kv_heads, self.head_dim
          ),
          self.hf_weights[f"{lp}.self_attn.norm_k.weight"],
          self.rms_norm_eps,
      )
      v_u = _np_linear(u_norm, self.hf_weights[f"{lp}.self_attn.to_v.weight"]).reshape(
          -1, self.num_kv_heads, self.head_dim
      )

      q_g = _np_rms_norm(
          _np_linear(g_norm, self.hf_weights[f"{lp}.self_attn.add_q_proj.weight"]).reshape(
              -1, self.num_heads, self.head_dim
          ),
          self.hf_weights[f"{lp}.self_attn.norm_added_q.weight"],
          self.rms_norm_eps,
      )
      k_g = _np_rms_norm(
          _np_linear(g_norm, self.hf_weights[f"{lp}.self_attn.add_k_proj.weight"]).reshape(
              -1, self.num_kv_heads, self.head_dim
          ),
          self.hf_weights[f"{lp}.self_attn.norm_added_k.weight"],
          self.rms_norm_eps,
      )
      v_g = _np_linear(g_norm, self.hf_weights[f"{lp}.self_attn.add_v_proj.weight"]).reshape(
          -1, self.num_kv_heads, self.head_dim
      )

      q_u = _np_apply_rope(q_u, cos_und, sin_und)
      k_u = _np_apply_rope(k_u, cos_und, sin_und)
      q_g = _np_apply_rope(q_g, cos_gen, sin_gen)
      k_g = _np_apply_rope(k_g, cos_gen, sin_gen)

      u_outs, g_outs = [], []
      for b in range(bsz):
        qu_b = q_u[b * s_und : (b + 1) * s_und]
        ku_b = np.repeat(k_u[b * s_und : (b + 1) * s_und], repeat_kv, axis=1)
        vu_b = np.repeat(v_u[b * s_und : (b + 1) * s_und], repeat_kv, axis=1)

        qg_b = q_g[b * s_gen : (b + 1) * s_gen]
        kg_b = np.repeat(k_g[b * s_gen : (b + 1) * s_gen], repeat_kv, axis=1)
        vg_b = np.repeat(v_g[b * s_gen : (b + 1) * s_gen], repeat_kv, axis=1)

        scores_u = np.einsum("qhd,khd->hqk", qu_b, ku_b) * scale
        scores_u = np.where(causal_mask[None, :, :], scores_u, -1e9)
        probs_u = _np_softmax(scores_u)
        out_u = np.einsum("hqk,khd->qhd", probs_u, vu_b).reshape(s_und, self.hidden_size)
        u_outs.append(out_u)

        k_all_b = np.concatenate([ku_b, kg_b], axis=0)
        v_all_b = np.concatenate([vu_b, vg_b], axis=0)
        scores_g = np.einsum("qhd,khd->hqk", qg_b, k_all_b) * scale
        probs_g = _np_softmax(scores_g)
        out_g = np.einsum("hqk,khd->qhd", probs_g, v_all_b).reshape(s_gen, self.hidden_size)
        g_outs.append(out_g)

      und_seq = und_seq + _np_linear(np.concatenate(u_outs, axis=0), self.hf_weights[f"{lp}.self_attn.to_out.weight"])
      gen_seq = gen_seq + _np_linear(np.concatenate(g_outs, axis=0), self.hf_weights[f"{lp}.self_attn.to_add_out.weight"])

      und_post = _np_rms_norm(und_seq, self.hf_weights[f"{lp}.post_attention_layernorm.weight"], self.rms_norm_eps)
      gen_post = _np_rms_norm(
          gen_seq, self.hf_weights[f"{lp}.post_attention_layernorm_moe_gen.weight"], self.rms_norm_eps
      )
      und_seq = und_seq + self._mlp_np(und_post, f"{lp}.mlp")
      gen_seq = gen_seq + self._mlp_np(gen_post, f"{lp}.mlp_moe_gen")

    gen_seq = _np_rms_norm(gen_seq, self.hf_weights["norm_moe_gen.weight"], self.rms_norm_eps)
    out_patches = _np_linear(gen_seq, self.hf_weights["proj_out.weight"], self.hf_weights["proj_out.bias"]).reshape(
        bsz, t_lat, h_patch, w_patch, p, p, c_lat
    )
    v_pred = np.einsum("bthwpqc->bcthpwq", out_patches).reshape(bsz, c_lat, t_lat, h_lat, w_lat)
    if torch is not None and isinstance(input_ids, torch.Tensor):
      return torch.from_numpy(v_pred)
    return v_pred

  def __call__(self, input_ids: Any, latents: Any, timesteps: Any) -> Any:
    return self.forward(input_ids, latents, timesteps)


def compute_velocity_metrics(
    v_pred_maxtext: np.ndarray | jax.Array,
    v_pred_golden: np.ndarray | jax.Array,
) -> dict[str, float]:
  """Computes numerical parity metrics between MaxText and golden velocity vectors."""
  mt_f64 = np.asarray(v_pred_maxtext, dtype=np.float64)
  gold_f64 = np.asarray(v_pred_golden, dtype=np.float64)
  if mt_f64.shape != gold_f64.shape:
    raise ValueError(f"Velocity shape mismatch: MaxText {mt_f64.shape} vs Golden {gold_f64.shape}.")

  abs_diff = np.abs(mt_f64 - gold_f64)
  max_abs_diff = float(np.max(abs_diff))
  mean_abs_diff = float(np.mean(abs_diff))
  rmse = float(np.sqrt(np.mean(np.square(mt_f64 - gold_f64))))
  max_rel_diff = float(np.max(abs_diff / (np.abs(gold_f64) + 1e-8)))

  mt_flat = mt_f64.reshape(-1)
  gold_flat = gold_f64.reshape(-1)
  denom = float(np.linalg.norm(mt_flat) * np.linalg.norm(gold_flat))
  cosine_sim = float(np.dot(mt_flat, gold_flat) / denom) if denom > 0.0 else 0.0

  return {
      "max_abs_diff": max_abs_diff,
      "mean_abs_diff": mean_abs_diff,
      "rmse": rmse,
      "max_rel_diff": max_rel_diff,
      "cosine_similarity": cosine_sim,
  }


def check_velocity_parity(
    v_pred_maxtext: np.ndarray | jax.Array,
    v_pred_golden: np.ndarray | jax.Array,
    *,
    atol: float | None = None,
    rtol: float = 1e-4,
    min_cosine_sim: float | None = None,
    description: str = "Weaver Velocity Parity",
) -> dict[str, float]:
  """Logs velocity comparison metrics and asserts acceptance thresholds."""
  mt_arr = np.asarray(v_pred_maxtext, dtype=np.float32)
  gold_arr = np.asarray(v_pred_golden, dtype=np.float32)

  assert np.all(np.isfinite(mt_arr)), "MaxText predicted velocity contains NaN or Inf values."
  assert np.all(np.isfinite(gold_arr)), "Golden reference velocity contains NaN or Inf values."

  metrics = compute_velocity_metrics(mt_arr, gold_arr)

  table_str = f"\n--- {description} ---\n"
  table_str += f"| {'Metric':<28} | {'Value':<20} |\n"
  table_str += f"|{'-' * 30}|{'-' * 22}|\n"
  table_str += f"| {'shape':<28} | {str(mt_arr.shape):<20} |\n"
  table_str += f"| {'max_abs_diff':<28} | {metrics['max_abs_diff']:<20.6e} |\n"
  table_str += f"| {'mean_abs_diff':<28} | {metrics['mean_abs_diff']:<20.6e} |\n"
  table_str += f"| {'rmse':<28} | {metrics['rmse']:<20.6e} |\n"
  table_str += f"| {'max_rel_diff':<28} | {metrics['max_rel_diff']:<20.6e} |\n"
  table_str += f"| {'cosine_similarity':<28} | {metrics['cosine_similarity']:<20.8f} |\n"
  max_logging.log(table_str)

  if min_cosine_sim is not None:
    max_logging.log(
        f"Checking cosine similarity {metrics['cosine_similarity']:.8f} >= threshold {min_cosine_sim} (rtol={rtol})..."
    )
    assert metrics["cosine_similarity"] >= float(
        min_cosine_sim
    ), f"Cosine similarity {metrics['cosine_similarity']:.8f} is below required threshold {min_cosine_sim}."

  if atol is not None:
    max_logging.log(f"Checking maximum absolute difference {metrics['max_abs_diff']:.6e} <= atol={atol}...")
    assert metrics["max_abs_diff"] <= float(
        atol
    ), f"Maximum absolute difference {metrics['max_abs_diff']:.6e} exceeds required atol={atol}."

  return metrics


def main(config: Any, test_args: argparse.Namespace) -> dict[str, float]:
  """Runs the end-to-end forward pass velocity checker for WeaverOmniTransformer."""
  weaver_cfg = weaver.WeaverConfig.from_maxtext_config(config)
  num_layers = weaver_cfg.num_hidden_layers

  # Apply default acceptance criteria when not explicitly overridden on CLI
  atol = test_args.atol
  min_cosine_sim = test_args.min_cosine_sim
  if atol is None:
    atol = 1e-4 if num_layers <= 1 else 5e-4
  if min_cosine_sim is None and num_layers >= 36:
    min_cosine_sim = 0.9999

  golden_npz_path = resolve_golden_velocity_path(test_args.golden_velocity_path)
  max_logging.log(f"Loading golden velocity reference from {golden_npz_path}...")
  golden_data = np.load(golden_npz_path)

  # Extract identical inputs: prompt token IDs, noisy latents x_t, and timestep t=500.0
  input_ids_np = np.asarray(golden_data["input_ids"], dtype=np.int32)
  x_t_np = np.asarray(golden_data["x_t" if "x_t" in golden_data else "latents"], dtype=np.float32)
  if test_args.timestep is not None:
    timesteps_np = np.full((input_ids_np.shape[0],), float(test_args.timestep), dtype=np.float32)
  elif "timesteps" in golden_data:
    timesteps_np = np.asarray(golden_data["timesteps"], dtype=np.float32)
  else:
    timesteps_np = np.full((input_ids_np.shape[0],), 500.0, dtype=np.float32)

  max_logging.log(
      f"Inputs prepared: input_ids={input_ids_np.shape}, x_t={x_t_np.shape}, "
      f"timesteps={timesteps_np.tolist()}, num_layers={num_layers}, scan_layers={weaver_cfg.scan_layers}"
  )

  # Resolve or convert the MaxText Orbax checkpoint
  load_ckpt_path = getattr(config, "load_parameters_path", "") or ""
  with contextlib.ExitStack() as stack:
    if load_ckpt_path and os.path.exists(load_ckpt_path):
      orbax_ckpt_path = load_ckpt_path
      max_logging.log(f"Using existing converted MaxText Orbax checkpoint at {orbax_ckpt_path}")
    elif load_ckpt_path and not load_ckpt_path.startswith("gs://"):
      orbax_base_dir = load_ckpt_path
      if orbax_base_dir.endswith("/0/items"):
        orbax_base_dir = orbax_base_dir[: -len("/0/items")]
      os.makedirs(orbax_base_dir, exist_ok=True)
      max_logging.log(f"Converting golden safetensors to MaxText Orbax checkpoint at {orbax_base_dir}...")
      orbax_ckpt_path = convert_golden_weights_to_orbax(
          config, golden_data, orbax_base_dir, hf_model_path=test_args.hf_model_path
      )
    else:
      temp_orbax_dir = stack.enter_context(tempfile.TemporaryDirectory())
      max_logging.log(f"Converting golden safetensors to temporary Orbax checkpoint at {temp_orbax_dir}...")
      orbax_ckpt_path = convert_golden_weights_to_orbax(
          config, golden_data, temp_orbax_dir, hf_model_path=test_args.hf_model_path
      )

    # Instantiate MaxText WeaverOmniTransformer and restore converted Orbax weights
    maxtext_model = weaver.WeaverOmniTransformer(weaver_cfg, rngs=nnx.Rngs(0))
    loaded_leaves = load_orbax_weights_into_weaver_model(maxtext_model, orbax_ckpt_path)
    max_logging.log(f"Restored {loaded_leaves} parameter tensors into MaxText WeaverOmniTransformer.")

  # Execute MaxText forward pass
  input_ids_jax = jnp.asarray(input_ids_np, dtype=jnp.int32)
  x_t_jax = jnp.asarray(x_t_np, dtype=weaver_cfg.dtype)
  timesteps_jax = jnp.asarray(timesteps_np, dtype=jnp.float32)

  v_pred_maxtext = maxtext_model(input_ids_jax, x_t_jax, timesteps_jax)
  v_pred_maxtext_np = np.asarray(v_pred_maxtext, dtype=np.float32)

  # Obtain golden PyTorch velocity reference
  golden_key = f"mini_{num_layers}layer_v_pred" if num_layers == 1 else f"full_{num_layers}layer_v_pred"
  if test_args.run_hf_model or golden_key not in golden_data:
    max_logging.log(f"Running reference model ({num_layers} layers) on identical inputs...")
    hf_cfg = build_hf_config_from_maxtext_config(config)
    hf_weights = extract_hf_weights_for_layers(golden_data, num_layers=num_layers)
    pt_model = PyTorchWeaverReferenceModel(hf_cfg)
    pt_model.load_hf_weights(hf_weights)
    pt_model.eval()
    v_pred_golden_np = np.asarray(pt_model(input_ids_np, x_t_np, timesteps_np), dtype=np.float32)
  else:
    max_logging.log(f"Using pre-generated GPU PyTorch golden velocity '{golden_key}' from {golden_npz_path}")
    v_pred_golden_np = np.asarray(golden_data[golden_key], dtype=np.float32)

  metrics = check_velocity_parity(
      v_pred_maxtext_np,
      v_pred_golden_np,
      atol=atol,
      rtol=test_args.rtol,
      min_cosine_sim=min_cosine_sim,
      description=f"Weaver {num_layers}-Layer Velocity Parity (t={timesteps_np[0]:.1f})",
  )

  if jax.process_index() == 0 and test_args.output_velocity_path:
    os.makedirs(os.path.dirname(os.path.abspath(test_args.output_velocity_path)), exist_ok=True)
    np.savez(
        test_args.output_velocity_path,
        v_pred_maxtext=v_pred_maxtext_np,
        v_pred_golden=v_pred_golden_np,
        **{k: np.array(v, dtype=np.float64) for k, v in metrics.items()},
    )
    max_logging.log(f"Saved velocity comparison outputs to {test_args.output_velocity_path}")

  max_logging.log(f"Velocity parity check PASSED for {num_layers}-layer WeaverOmniTransformer!")
  return metrics


def parse_velocity_checker_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
  """Parses velocity checker CLI flags and returns `(test_args, remaining_maxtext_args)`."""
  parser = argparse.ArgumentParser(description="Weaver forward-pass velocity numerical checker.")
  parser.add_argument("--atol", type=float, required=False, default=None, help="Maximum absolute difference threshold.")
  parser.add_argument("--rtol", type=float, required=False, default=1e-4, help="Relative tolerance.")
  parser.add_argument(
      "--min_cosine_sim",
      type=float,
      required=False,
      default=None,
      help="Minimum required cosine similarity between MaxText and golden velocity vectors.",
  )
  parser.add_argument(
      "--golden_velocity_path",
      "--golden_path",
      dest="golden_velocity_path",
      type=str,
      required=False,
      default="",
      help="Path or GCS URI to the golden velocity `.npz` archive.",
  )
  parser.add_argument(
      "--hf_model_path",
      type=str,
      required=False,
      default="",
      help="Optional directory containing HuggingFace/Diffusers safetensors checkpoint.",
  )
  parser.add_argument(
      "--run_hf_model",
      type=str2bool,
      required=False,
      default=False,
      help="Whether to execute the PyTorch reference model on-the-fly in addition to/instead of golden `.npz`.",
  )
  parser.add_argument(
      "--timestep",
      type=float,
      required=False,
      default=None,
      help="Diffusion timestep scalar t fed to both models (default: 500.0).",
  )
  parser.add_argument(
      "--output_velocity_path",
      type=str,
      required=False,
      default="",
      help="Optional path to save predicted and golden velocity arrays.",
  )
  return parser.parse_known_args(argv)


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parsed_test_args, remaining_args = parse_velocity_checker_args()
  model_args = [sys.argv[0]] + remaining_args
  cfg = pyconfig.initialize(model_args)
  main(cfg, parsed_test_args)
