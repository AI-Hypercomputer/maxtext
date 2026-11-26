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

"""CPU verification suite for Qwen3.5 layer-wise and block-wise streaming."""

import copy
import json
import os
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.models import qwen3
from maxtext.trainers.pre_train import train as mt_train
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import qwen3_5_layerwise
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax
from safetensors.torch import save_file as save_safetensors
import tensorflow as tf
import torch
from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe as hf_qwen35_cfg
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_qwen35_mod

if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  _existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
  os.environ["XLA_FLAGS"] = (
      f"{_existing_xla_flags} --xla_force_host_platform_device_count=4".strip()
  )

_DUMP_JSON = flags.DEFINE_string(
    "dump_json",
    "",
    "Optional path to write CPU and HLO verification results as JSON.",
)

_TINY_TEXT_CFG: dict[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "dtype": "bfloat16",
    "eos_token_id": 248,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "layer_types": (
        [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 2
    ),
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 8192,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_moe_text",
    "moe_intermediate_size": 384,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "norm_topk_prob": True,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "router_aux_loss_coef": 0.001,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
    "shared_expert_intermediate_size": 384,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 512,
}

_TINY_HF_DICT: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "image_token_id": 500,
    "model_type": "qwen3_5_moe",
    "text_config": _TINY_TEXT_CFG,
    "tie_word_embeddings": False,
    "video_token_id": 501,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "model_type": "qwen3_5_moe",
        "num_heads": 4,
        "num_position_embeddings": 64,
        "out_hidden_size": 256,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 499,
    "vision_start_token_id": 498,
}

_BLOCK = 128
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _diff_metrics(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
  got64 = np.asarray(got, dtype=np.float64)
  ref64 = np.asarray(ref, dtype=np.float64)
  abs_diff = np.abs(got64 - ref64)
  max_abs = float(np.max(abs_diff))
  mean_abs = float(np.mean(abs_diff))
  ref_max = max(float(np.max(np.abs(ref64))), 1e-12)
  rel_l2 = float(
      np.linalg.norm(got64 - ref64) / max(float(np.linalg.norm(ref64)), 1e-12)
  )
  cos = float(
      np.dot(got64.ravel(), ref64.ravel())
      / max(float(np.linalg.norm(got64) * np.linalg.norm(ref64)), 1e-12)
  )
  return {
      "max_abs": max_abs,
      "mean_abs": mean_abs,
      "max_rel_vs_peak": max_abs / ref_max,
      "rel_l2": rel_l2,
      "cosine_sim": cos,
      "ref_rms": float(np.sqrt(np.mean(ref64**2))),
      "got_rms": float(np.sqrt(np.mean(got64**2))),
  }


def _quantize_fp8_blockwise(
    w_f32: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
  rows, cols = w_f32.shape
  rb, cb = rows // _BLOCK, cols // _BLOCK
  r_gain = torch.linspace(0.4, 1.7, rb, dtype=torch.float32)[:, None]
  c_gain = torch.linspace(0.45, 1.65, cb, dtype=torch.float32)[None, :]
  jitter = torch.empty((rb, cb), dtype=torch.float32).uniform_(
      0.75, 1.3, generator=generator
  )
  block_mult = (r_gain * c_gain * jitter)[:, None, :, None]
  reshaped = w_f32.reshape(rb, _BLOCK, cb, _BLOCK) * block_mult
  max_abs = reshaped.abs().amax(dim=(1, 3)).clamp(min=1e-6)
  scale_inv_bf16 = (max_abs / _FP8_MAX).to(torch.bfloat16)
  scale_inv_f32 = scale_inv_bf16.to(torch.float32)[:, None, :, None]
  q_fp8 = torch.clamp(
      torch.round(reshaped / scale_inv_f32), -_FP8_MAX, _FP8_MAX
  ).to(torch.float8_e4m3fn)
  return q_fp8.reshape(rows, cols), scale_inv_bf16


def _dequantize_fp8_blockwise_np(
    q_np: np.ndarray, scale_inv_np: np.ndarray, out_dtype: torch.dtype
) -> torch.Tensor:
  """Dequantizes a 2D FP8 array with `[rb, cb]` block scales into a PyTorch tensor."""
  rows, cols = q_np.shape
  rb, cb = scale_inv_np.shape
  bm, bn = rows // rb, cols // cb
  q_f32 = torch.from_numpy(q_np.astype(np.float32)).reshape(rb, bm, cb, bn)
  s_f32 = torch.from_numpy(scale_inv_np.astype(np.float32)).reshape(
      rb, 1, cb, 1
  )
  return (q_f32 * s_f32).to(out_dtype).reshape(rows, cols)


def _synthesize_tiny_checkpoint(
    seed: int = 20261113,
) -> dict[str, torch.Tensor]:
  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  tc = _TINY_TEXT_CFG
  h_dim, vocab = int(tc["hidden_size"]), int(tc["vocab_size"])
  i_moe, i_sh = int(tc["moe_intermediate_size"]), int(
      tc["shared_expert_intermediate_size"]
  )
  num_e, num_l = int(tc["num_experts"]), int(tc["num_hidden_layers"])
  h_q, h_kv, d_head = (
      int(tc["num_attention_heads"]),
      int(tc["num_key_value_heads"]),
      int(tc["head_dim"]),
  )
  h_k, h_v, d_k, d_v = (
      int(tc["linear_num_key_heads"]),
      int(tc["linear_num_value_heads"]),
      int(tc["linear_key_head_dim"]),
      int(tc["linear_value_head_dim"]),
  )
  k_conv = int(tc["linear_conv_kernel_dim"])
  conv_dim = 2 * h_k * d_k + h_v * d_v
  tensors: dict[str, torch.Tensor] = {}

  def randn(shape, std=0.02):
    return torch.randn(shape, generator=gen, dtype=torch.float32) * std

  tensors["model.language_model.embed_tokens.weight"] = randn(
      (vocab, h_dim), 0.04
  ).to(torch.bfloat16)
  tensors["model.language_model.norm.weight"] = (
      torch.empty((h_dim,), dtype=torch.float32)
      .uniform_(-0.35, 0.35, generator=gen)
      .to(torch.bfloat16)
  )
  tensors["lm_head.weight"] = randn((vocab, h_dim), 0.03).to(torch.bfloat16)

  for i in range(num_l):
    pfx = f"model.language_model.layers.{i}"
    tensors[f"{pfx}.input_layernorm.weight"] = (
        torch.empty((h_dim,), dtype=torch.float32)
        .uniform_(-0.35, 0.35, generator=gen)
        .to(torch.bfloat16)
    )
    tensors[f"{pfx}.post_attention_layernorm.weight"] = (
        torch.empty((h_dim,), dtype=torch.float32)
        .uniform_(-0.35, 0.35, generator=gen)
        .to(torch.bfloat16)
    )
    is_full = (i + 1) % int(tc["full_attention_interval"]) == 0
    if is_full:
      q_raw = randn((h_q * d_head * 2, h_dim), 0.03)
      q_raw.view(h_q, 2 * d_head, h_dim)[:, d_head:, :] *= 4.0
      for name, raw in (
          ("q_proj", q_raw),
          ("k_proj", randn((h_kv * d_head, h_dim), 0.03)),
          ("v_proj", randn((h_kv * d_head, h_dim), 0.03)),
          ("o_proj", randn((h_dim, h_q * d_head), 0.03)),
      ):
        q_w, s_inv = _quantize_fp8_blockwise(raw, gen)
        tensors[f"{pfx}.self_attn.{name}.weight"] = q_w
        tensors[f"{pfx}.self_attn.{name}.weight_scale_inv"] = s_inv
      tensors[f"{pfx}.self_attn.q_norm.weight"] = (
          torch.empty((d_head,), dtype=torch.float32)
          .uniform_(-0.35, 0.35, generator=gen)
          .to(torch.bfloat16)
      )
      tensors[f"{pfx}.self_attn.k_norm.weight"] = (
          torch.empty((d_head,), dtype=torch.float32)
          .uniform_(-0.35, 0.35, generator=gen)
          .to(torch.bfloat16)
      )
    else:
      for name, raw in (
          ("in_proj_qkv", randn((conv_dim, h_dim), 0.03)),
          ("in_proj_z", randn((h_v * d_v, h_dim), 0.03)),
          ("out_proj", randn((h_dim, h_v * d_v), 0.03)),
      ):
        q_w, s_inv = _quantize_fp8_blockwise(raw, gen)
        tensors[f"{pfx}.linear_attn.{name}.weight"] = q_w
        tensors[f"{pfx}.linear_attn.{name}.weight_scale_inv"] = s_inv
      tensors[f"{pfx}.linear_attn.in_proj_b.weight"] = randn(
          (h_v, h_dim), 0.04
      ).to(torch.bfloat16)
      tensors[f"{pfx}.linear_attn.in_proj_a.weight"] = randn(
          (h_v, h_dim), 0.04
      ).to(torch.bfloat16)
      tensors[f"{pfx}.linear_attn.conv1d.weight"] = randn(
          (conv_dim, 1, k_conv), 0.05
      ).to(torch.bfloat16)
      a_vals = torch.empty((h_v,), dtype=torch.float32).uniform_(
          0.5, 8.0, generator=gen
      )
      tensors[f"{pfx}.linear_attn.A_log"] = torch.log(a_vals)
      tensors[f"{pfx}.linear_attn.dt_bias"] = (
          torch.empty((h_v,), dtype=torch.float32)
          .uniform_(0.5, 1.5, generator=gen)
          .to(torch.bfloat16)
      )
      tensors[f"{pfx}.linear_attn.norm.weight"] = torch.empty(
          (d_v,), dtype=torch.float32
      ).uniform_(0.65, 1.35, generator=gen)

    tensors[f"{pfx}.mlp.gate.weight"] = randn((num_e, h_dim), 0.04).to(
        torch.bfloat16
    )
    tensors[f"{pfx}.mlp.shared_expert_gate.weight"] = randn(
        (1, h_dim), 0.25
    ).to(torch.bfloat16)
    for name, raw in (
        ("gate_proj", randn((i_sh, h_dim), 0.03)),
        ("up_proj", randn((i_sh, h_dim), 0.03)),
        ("down_proj", randn((h_dim, i_sh), 0.03)),
    ):
      q_w, s_inv = _quantize_fp8_blockwise(raw, gen)
      tensors[f"{pfx}.mlp.shared_expert.{name}.weight"] = q_w
      tensors[f"{pfx}.mlp.shared_expert.{name}.weight_scale_inv"] = s_inv
    for e in range(num_e):
      for name, raw in (
          ("gate_proj", randn((i_moe, h_dim), 0.03)),
          ("up_proj", randn((i_moe, h_dim), 0.03)),
          ("down_proj", randn((h_dim, i_moe), 0.03)),
      ):
        q_w, s_inv = _quantize_fp8_blockwise(raw, gen)
        tensors[f"{pfx}.mlp.experts.{e}.{name}.weight"] = q_w
        tensors[f"{pfx}.mlp.experts.{e}.{name}.weight_scale_inv"] = s_inv
  return tensors


def _write_sharded_safetensors(
    ckpt_dir: str, tensors: dict[str, torch.Tensor]
) -> None:
  keys = sorted(tensors.keys())
  mid = len(keys) // 2
  shards = {
      "model-00001-of-00002.safetensors": {k: tensors[k] for k in keys[:mid]},
      "model-00002-of-00002.safetensors": {k: tensors[k] for k in keys[mid:]},
  }
  weight_map = {}
  total_size = 0
  for fname, shard_dict in shards.items():
    save_safetensors(shard_dict, os.path.join(ckpt_dir, fname))
    for k, v in shard_dict.items():
      weight_map[k] = fname
      total_size += v.numel() * v.element_size()
  with open(
      os.path.join(ckpt_dir, "model.safetensors.index.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        f,
        indent=2,
    )
  with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(_TINY_HF_DICT, f, indent=2)


class Qwen35LayerwiseTest(unittest.TestCase):
  """Validates `Qwen35LayerwiseVerifier` on synthetic 8-layer and optional 397B weights."""

  tmpdir: tempfile.TemporaryDirectory[str]
  ckpt_dir: str
  raw_tensors: dict[str, torch.Tensor]
  report: dict[str, object]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch.set_grad_enabled(False)
    cls.tmpdir = tempfile.TemporaryDirectory()
    cls.ckpt_dir = cls.tmpdir.name
    cls.raw_tensors = _synthesize_tiny_checkpoint(seed=20261113)
    _write_sharded_safetensors(cls.ckpt_dir, cls.raw_tensors)
    cls.report = {}

  @classmethod
  def tearDownClass(cls):
    json_str = json.dumps(cls.report, indent=2, sort_keys=True)
    print(
        f"\n=== LAYERWISE_TEST_JSON_BEGIN ===\n{json_str}\n==="
        " LAYERWISE_TEST_JSON_END ===\n",
        flush=True,
    )
    if _DUMP_JSON.value:
      os.makedirs(os.path.dirname(_DUMP_JSON.value), exist_ok=True)
      with open(_DUMP_JSON.value, "w", encoding="utf-8") as f:
        f.write(json_str)
    cls.tmpdir.cleanup()
    super().tearDownClass()

  def _init_tiny_config(
      self,
      dtype: str,
      matmul_precision: str,
      scan_layers: bool,
      batch_size: int,
      seq_len: int,
  ):
    num_devices = len(jax.devices())
    per_device_bs = batch_size / num_devices
    return pyconfig.initialize([
        "",
        get_test_config_path(),
        "model_name=qwen3.5-397b-a17b-fp8",
        "override_model_config=True",
        "run_name=qwen35_layerwise_test",
        f"load_parameters_path={self.ckpt_dir}",
        "source_checkpoint_layout=safetensors_dynamic",
        f"dtype={dtype}",
        "weight_dtype=float8_e4m3fn",
        f"weight_block_size={_BLOCK}",
        f"matmul_precision={matmul_precision}",
        f"scan_layers={scan_layers}",
        "float32_gate_logits=True",
        "float32_logits=True",
        "cast_logits_to_fp32=True",
        "logits_dot_in_fp32=True",
        "remat_policy=none",
        "enable_dropout=False",
        "dropout_rate=0.0",
        "attention=dot_product",
        "megablox=False",
        "sparse_matmul=False",
        "use_multimodal=False",
        "ici_data_parallelism=2",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=1",
        "enable_checkpointing=False",
        "async_checkpointing=False",
        "skip_jax_distributed_system=True",
        f"per_device_batch_size={per_device_bs}",
        f"max_target_length={seq_len}",
        f"base_emb_dim={_TINY_TEXT_CFG['hidden_size']}",
        f"base_num_decoder_layers={_TINY_TEXT_CFG['num_hidden_layers']}",
        f"base_num_query_heads={_TINY_TEXT_CFG['num_attention_heads']}",
        f"base_num_kv_heads={_TINY_TEXT_CFG['num_key_value_heads']}",
        f"head_dim={_TINY_TEXT_CFG['head_dim']}",
        f"vocab_size={_TINY_TEXT_CFG['vocab_size']}",
        f"base_mlp_dim={_TINY_TEXT_CFG['moe_intermediate_size']}",
        f"base_moe_mlp_dim={_TINY_TEXT_CFG['moe_intermediate_size']}",
        f"num_experts={_TINY_TEXT_CFG['num_experts']}",
        f"num_experts_per_tok={_TINY_TEXT_CFG['num_experts_per_tok']}",
        "shared_experts=1",
        "norm_topk_prob=True",
        f"inhomogeneous_layer_cycle_interval={_TINY_TEXT_CFG['full_attention_interval']}",
        f"gdn_conv_kernel_dim={_TINY_TEXT_CFG['linear_conv_kernel_dim']}",
        f"gdn_key_head_dim={_TINY_TEXT_CFG['linear_key_head_dim']}",
        f"gdn_value_head_dim={_TINY_TEXT_CFG['linear_value_head_dim']}",
        f"gdn_num_key_heads={_TINY_TEXT_CFG['linear_num_key_heads']}",
        f"gdn_num_value_heads={_TINY_TEXT_CFG['linear_num_value_heads']}",
        "gdn_chunk_size=64",
        "use_mrope=True",
        "mrope_section=[11,11,10]",
        "partial_rotary_factor=0.25",
        "rope_max_timescale=10000000",
    ])

  def test_01_synthetic_8layer_chained_vs_full_scanned_model(self):
    """Verifies `Qwen35LayerwiseVerifier` (1-layer & 4-layer block streaming) matches full 8-layer scanned model."""
    rng = np.random.default_rng(101)
    total_bs = 8
    mb_bs = 4
    seq_len = 96
    inputs = rng.integers(
        1,
        int(_TINY_TEXT_CFG["vocab_size"]),
        size=(total_bs, seq_len),
        dtype=np.int32,
    )
    targets = np.roll(inputs, shift=-1, axis=1)
    seg = np.ones((total_bs, seq_len), dtype=np.int32)
    seg[:, 48:] = 2
    pos = np.concatenate(
        [np.arange(48, dtype=np.int32), np.arange(48, dtype=np.int32)]
    )[None, :]
    pos = np.broadcast_to(pos, (total_bs, seq_len)).copy()

    full_batch_np = {
        "inputs": inputs,
        "inputs_position": pos,
        "inputs_segmentation": seg,
        "targets": targets,
        "targets_position": pos,
        "targets_segmentation": seg,
    }
    microbatches_np = [
        {k: v[0:mb_bs] for k, v in full_batch_np.items()},
        {k: v[mb_bs:total_bs] for k, v in full_batch_np.items()},
    ]

    orig_run_layer = qwen3.Qwen3NextScannableBlock._run_layer

    def _recording_run_layer(
        block_self,
        layer,
        y,
        layer_kwargs,
        kv_cache=None,
        forced_routed_experts=None,
    ):
      out, updated_kv = orig_run_layer(
          block_self,
          layer,
          y,
          layer_kwargs,
          kv_cache=kv_cache,
          forced_routed_experts=forced_routed_experts,
      )
      seg_ids = layer_kwargs.get("decoder_segment_ids")
      valid_f32 = (
          (seg_ids != 0).astype(jnp.float32)
          if seg_ids is not None
          else jnp.ones(out.shape[:2], dtype=jnp.float32)
      )
      tok_ms = jnp.mean(jnp.square(out.astype(jnp.float32)), axis=-1)
      layer.sow(nnx.Intermediate, "resid_ms_sum", jnp.sum(tok_ms * valid_f32))
      return out, updated_kv

    tiny_hf_cfg_obj = hf_qwen35_cfg.Qwen3_5MoeConfig(
        **copy.deepcopy(_TINY_HF_DICT)
    )
    with (
        mock.patch.dict(
            hf_model_configs.HF_MODEL_CONFIGS,
            {"qwen3.5-397b-a17b-fp8": tiny_hf_cfg_obj},
        ),
        mock.patch.dict(
            load_dynamic.HF_MODEL_CONFIGS,
            {"qwen3.5-397b-a17b-fp8": tiny_hf_cfg_obj},
        ),
        mock.patch.object(
            qwen3.Qwen3NextScannableBlock, "_run_layer", _recording_run_layer
        ),
    ):
      # 1. Full 8-layer scanned model forward pass (`mt_train.loss_fn`)
      cfg_full = self._init_tiny_config(
          "bfloat16",
          "default",
          scan_layers=True,
          batch_size=mb_bs,
          seq_len=seq_len,
      )
      mesh = Mesh(
          maxtext_utils.create_device_mesh(cfg_full), cfg_full.mesh_axes
      )
      create_model_partial, _ = model_creation_utils.create_nnx_abstract_model(
          cfg_full, mesh
      )

      def init_state_fn():
        m = create_model_partial()
        opt = nnx.Optimizer(m, optax.sgd(0.0), wrt=nnx.Param)
        return train_state_nnx.TrainStateNNX(m, opt)

      with nn_partitioning.axis_rules(cfg_full.logical_axis_rules):
        graphdef, _ = nnx.split(nnx.eval_shape(init_state_fn))
      state, _, _, _, was_restored = maxtext_utils.setup_training_state(
          None, cfg_full, mesh, None, init_state_fn
      )
      self.assertTrue(was_restored)
      full_model = nnx.merge(graphdef, state).model

      data_sharding = sharding.get_input_data_sharding(cfg_full, mesh)
      full_loss_sum = 0.0
      full_embed_ms_sum = 0.0
      full_layer_ms_sums = np.zeros(8, dtype=np.float64)
      full_logits_ms_sum = 0.0
      full_doc1_loss_sum = 0.0
      full_doc1_tok_sum = 0.0
      full_doc2p_loss_sum = 0.0
      full_doc2p_tok_sum = 0.0

      for mb_np in microbatches_np:
        mb_dev = jax.tree.map(lambda x: jax.device_put(x, data_sharding), mb_np)
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(
            cfg_full.logical_axis_rules
        ):
          mb_loss, _ = mt_train.loss_fn(
              full_model, cfg_full, mb_dev, None, None, is_train=False
          )
          full_loss_sum += float(mb_loss)

          seg_dev = mb_dev["inputs_segmentation"]
          pos_dev = mb_dev["inputs_position"]
          tgt_dev = mb_dev["targets"]
          valid_f32 = (seg_dev != 0).astype(jnp.float32)

          y_emb = full_model.token_embedder(
              mb_dev["inputs"].astype(jnp.int32)
          ).astype(cfg_full.dtype)
          full_embed_ms_sum += float(
              jnp.sum(
                  jnp.mean(jnp.square(y_emb.astype(jnp.float32)), axis=-1)
                  * valid_f32
              )
          )

          logits = full_model(
              mb_dev["inputs"],
              pos_dev,
              decoder_segment_ids=seg_dev,
              enable_dropout=False,
          )
          _, _, inter_st, _ = nnx.split(
              full_model.decoder.layers, nnx.Param, nnx.Intermediate, ...
          )
          inter_d = nnx.to_pure_dict(inter_st)
          loc_raw = inter_d["local_layers"]["resid_ms_sum"]
          glb_raw = inter_d["global_layer"]["resid_ms_sum"]
          loc_arr = np.asarray(
              loc_raw[-1] if isinstance(loc_raw, (list, tuple)) else loc_raw,
              dtype=np.float64,
          )
          glb_arr = np.asarray(
              glb_raw[-1] if isinstance(glb_raw, (list, tuple)) else glb_raw,
              dtype=np.float64,
          )
          for blk_i in range(2):
            full_layer_ms_sums[blk_i * 4 : blk_i * 4 + 3] += loc_arr[
                blk_i
            ].reshape(3)
            full_layer_ms_sums[blk_i * 4 + 3] += float(
                glb_arr[blk_i].reshape(-1)[0]
            )

          logits_f32 = logits.astype(jnp.float32)
          full_logits_ms_sum += float(
              jnp.sum(jnp.mean(jnp.square(logits_f32), axis=-1) * valid_f32)
          )
          one_hot = jax.nn.one_hot(
              tgt_dev, cfg_full.vocab_size, dtype=logits_f32.dtype
          )
          xent, _ = max_utils.cross_entropy_with_logits(
              logits_f32, one_hot, z_loss=cfg_full.z_loss_multiplier
          )
          doc1_m = (mb_dev["targets_segmentation"] == 1).astype(jnp.float32)
          doc2p_m = (mb_dev["targets_segmentation"] >= 2).astype(jnp.float32)
          full_doc1_loss_sum += float(jnp.sum(xent * doc1_m))
          full_doc1_tok_sum += float(jnp.sum(doc1_m))
          full_doc2p_loss_sum += float(jnp.sum(xent * doc2p_m))
          full_doc2p_tok_sum += float(jnp.sum(doc2p_m))

      valid_tok = float(np.sum(seg != 0))
      full_embed_rms = float(np.sqrt(full_embed_ms_sum / valid_tok))
      full_layer_rms = [
          float(np.sqrt(ms / valid_tok)) for ms in full_layer_ms_sums
      ]
      full_logits_rms = float(np.sqrt(full_logits_ms_sum / valid_tok))
      full_eval_loss = full_loss_sum / len(microbatches_np)
      full_doc1_loss = full_doc1_loss_sum / full_doc1_tok_sum
      full_doc2p_loss = full_doc2p_loss_sum / full_doc2p_tok_sum

      # 2. Layer-by-layer and block-by-block streaming verifier across the 2
      # microbatches.
      cfg_lw = self._init_tiny_config(
          "bfloat16",
          "default",
          scan_layers=False,
          batch_size=mb_bs,
          seq_len=seq_len,
      )
      verifier = qwen3_5_layerwise.Qwen35LayerwiseVerifier(
          cfg_lw, mesh, self.ckpt_dir, total_layers=8
      )
      act_dir = os.path.join(self.ckpt_dir, "mb0_act")
      lw_res_b = verifier.run_layerwise_verification(
          microbatches_np,
          run_clean_single_doc=True,
          include_teacher_forced=True,
          save_mb0_act_dir=act_dir,
      )
      lw_res = verifier.run_layerwise_verification(
          microbatches_np,
          run_clean_single_doc=True,
          include_teacher_forced=True,
          ref_mb0_act_dir=act_dir,
      )
      merged_ab = qwen3_5_layerwise.compare_layerwise_runs(lw_res, lw_res_b)
      self.assertAlmostEqual(
          merged_ab["packed"]["loss_delta_a_minus_b"], 0.0, places=6
      )
      bw_res = verifier.run_blockwise_forward(microbatches_np)

    lw_layer_rms = [lyr["resid_rms"] for lyr in lw_res["packed"]["layers"]]
    bw_layer_rms = bw_res["layer_rms"]
    lw_head = lw_res["packed"]["head"]
    bw_head = bw_res["head"]

    max_lw_rms_diff = max(
        abs(a - b) for a, b in zip(full_layer_rms, lw_layer_rms)
    )
    max_bw_rms_diff = max(
        abs(a - b) for a, b in zip(full_layer_rms, bw_layer_rms)
    )
    lw_loss_diff = abs(full_eval_loss - lw_head["eval_loss"])
    lw_doc1_diff = abs(full_doc1_loss - lw_head["doc1_loss"])
    lw_doc2p_diff = abs(full_doc2p_loss - lw_head["doc2plus_loss"])

    type(self).report["01_synthetic_8layer_chained_vs_full_scanned"] = {
        "full_embed_rms": full_embed_rms,
        "lw_embed_rms": lw_res["packed"]["embed_rms"],
        "full_layer_rms": full_layer_rms,
        "lw_layer_rms": lw_layer_rms,
        "bw_layer_rms": bw_layer_rms,
        "max_lw_rms_diff": max_lw_rms_diff,
        "max_bw_rms_diff": max_bw_rms_diff,
        "lw_final_norm_rms": lw_head["final_norm_rms"],
        "full_logits_rms": full_logits_rms,
        "lw_logits_rms": lw_head["logits_rms"],
        "full_eval_loss": full_eval_loss,
        "lw_eval_loss": lw_head["eval_loss"],
        "bw_eval_loss": bw_head["eval_loss"],
        "lw_loss_abs_diff": lw_loss_diff,
        "full_doc1_loss": full_doc1_loss,
        "lw_doc1_loss": lw_head["doc1_loss"],
        "lw_doc1_abs_diff": lw_doc1_diff,
        "full_doc2plus_loss": full_doc2p_loss,
        "lw_doc2plus_loss": lw_head["doc2plus_loss"],
        "lw_doc2plus_abs_diff": lw_doc2p_diff,
        "clean_single_doc_eval_loss": lw_res["clean_single_doc"]["head"][
            "eval_loss"
        ],
    }

    self.assertAlmostEqual(
        full_embed_rms, lw_res["packed"]["embed_rms"], places=5
    )
    self.assertAlmostEqual(full_logits_rms, lw_head["logits_rms"], places=4)
    self.assertLess(max_lw_rms_diff, 1e-5)
    self.assertLess(max_bw_rms_diff, 1e-5)
    self.assertLess(lw_loss_diff, 1e-5)
    self.assertLess(lw_doc1_diff, 1e-5)
    self.assertLess(lw_doc2p_diff, 1e-5)

  def test_02_real_397b_streaming_and_hf_layer_parity_on_cpu(self):
    """Streams real Qwen3.5-397B-A17B-FP8 `layer_0` (GDN) & `layer_3` (FullAttn) on CPU and checks HF parity."""
    ckpt_dir = qwen3_5_layerwise.FLAGS.checkpoint_dir
    if not ckpt_dir:
      self.skipTest(
          "Skipping optional real-checkpoint HF parity test (--checkpoint_dir"
          " not set)."
      )
    try:
      ckpt_exists = tf.io.gfile.exists(
          os.path.join(ckpt_dir, "model.safetensors.index.json")
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(
          "Checkpoint access unavailable in sandboxed test runner"
          f" ({type(e).__name__})."
      )
      return
    if not ckpt_exists:
      self.skipTest(f"Real 397B checkpoint not accessible at {ckpt_dir}")

    batch_size = 4
    seq_len = 16
    cpu_num_experts = 8
    cpu_experts_per_tok = 4
    cfg = pyconfig.initialize([
        "",
        get_test_config_path(),
        "model_name=qwen3.5-397b-a17b-fp8",
        "override_model_config=True",
        "run_name=qwen35_real_397b_cpu_layerwise",
        "dtype=bfloat16",
        "weight_dtype=float8_e4m3fn",
        "weight_block_size=128",
        "matmul_precision=default",
        "scan_layers=False",
        "float32_gate_logits=True",
        "float32_logits=True",
        "cast_logits_to_fp32=True",
        "logits_dot_in_fp32=True",
        "remat_policy=none",
        "enable_dropout=False",
        "dropout_rate=0.0",
        "attention=dot_product",
        "megablox=False",
        "sparse_matmul=False",
        "use_multimodal=False",
        "ici_data_parallelism=1",
        "ici_fsdp_parallelism=4",
        "ici_expert_parallelism=1",
        "enable_checkpointing=False",
        "async_checkpointing=False",
        "skip_jax_distributed_system=True",
        "per_device_batch_size=1.0",
        f"max_target_length={seq_len}",
        "gdn_chunk_size=16",
        f"num_experts={cpu_num_experts}",
        f"num_experts_per_tok={cpu_experts_per_tok}",
    ])
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    verifier = qwen3_5_layerwise.Qwen35LayerwiseVerifier(
        cfg, mesh, ckpt_dir, total_layers=60
    )
    self.assertGreater(len(verifier.reader.weight_map), 90000)

    rng = np.random.default_rng(2026)
    h0_np = (rng.standard_normal((batch_size, seq_len, 4096)) * 0.04).astype(
        np.float32
    )
    h0_bf16_pt = torch.from_numpy(h0_np).to(torch.bfloat16)
    h0_bf16_jax = jnp.asarray(
        h0_bf16_pt.to(torch.float32).numpy(), dtype=jnp.bfloat16
    )

    seg_clean = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    pos_clean = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len)
    )

    l0_params = verifier.reader.load_single_layer_params(
        0, verifier.abs_gdn_params, cfg, total_layers=60
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      y0_clean, _ = verifier.gdn_fwd(
          l0_params, h0_bf16_jax, seg_clean, pos_clean
      )
      y0_clean_np = np.asarray(y0_clean, dtype=np.float32)

    hf_cfg_dict, _, _ = verifier.reader.get_hf_config_and_unscanned_mappings(
        cfg, total_layers=60
    )
    tc_dict = dict(hf_cfg_dict["text_config"])
    tc_dict["num_experts"] = cpu_num_experts
    tc_dict["num_experts_per_tok"] = cpu_experts_per_tok
    hf_text_cfg = hf_qwen35_cfg.Qwen3_5MoeTextConfig(**tc_dict)
    hf_text_cfg._attn_implementation = "eager"  # pylint: disable=protected-access
    hf_rotary = hf_qwen35_mod.Qwen3_5MoeTextRotaryEmbedding(hf_text_cfg)

    def _load_hf_decoder_layer(
        layer_idx: int, raw_np: dict[str, np.ndarray]
    ) -> hf_qwen35_mod.Qwen3_5MoeDecoderLayer:
      hf_lyr = (
          hf_qwen35_mod.Qwen3_5MoeDecoderLayer(hf_text_cfg, layer_idx=layer_idx)
          .to(dtype=torch.bfloat16)
          .eval()
      )
      pfx = f"model.language_model.layers.{layer_idx}."
      sd = {}
      for k, arr in raw_np.items():
        if k.endswith(".weight_scale_inv") or ".mlp.experts." in k:
          continue
        rel_k = k[len(pfx) :]
        scale_k = (
            f"{k[:-len('.weight')]}.weight_scale_inv"
            if k.endswith(".weight")
            else ""
        )
        if scale_k in raw_np:
          sd[rel_k] = _dequantize_fp8_blockwise_np(
              arr, raw_np[scale_k], torch.bfloat16
          )
        else:
          t_val = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
          if rel_k == "mlp.gate.weight" and t_val.shape[0] > cpu_num_experts:
            t_val = t_val[:cpu_num_experts, :]
          sd[rel_k] = t_val
      gate_up_list = []
      down_list = []
      for e in range(cpu_num_experts):
        epfx = f"{pfx}mlp.experts.{e}"
        g = _dequantize_fp8_blockwise_np(
            raw_np[f"{epfx}.gate_proj.weight"],
            raw_np[f"{epfx}.gate_proj.weight_scale_inv"],
            torch.bfloat16,
        )
        u = _dequantize_fp8_blockwise_np(
            raw_np[f"{epfx}.up_proj.weight"],
            raw_np[f"{epfx}.up_proj.weight_scale_inv"],
            torch.bfloat16,
        )
        d = _dequantize_fp8_blockwise_np(
            raw_np[f"{epfx}.down_proj.weight"],
            raw_np[f"{epfx}.down_proj.weight_scale_inv"],
            torch.bfloat16,
        )
        gate_up_list.append(torch.cat([g, u], dim=0))
        down_list.append(d)
      sd["mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
      sd["mlp.experts.down_proj"] = torch.stack(down_list, dim=0)
      hf_lyr.load_state_dict(sd, strict=True)
      return hf_lyr

    hf_l0 = _load_hf_decoder_layer(0, verifier.reader.last_hf_tensors)
    verifier.reader.last_hf_tensors = {}
    pos_ids_4d = (
        torch.arange(seq_len, dtype=torch.int64)
        .view(1, 1, -1)
        .expand(4, batch_size, -1)
    )
    text_pos_ids = pos_ids_4d[0]
    pos_emb = hf_rotary(h0_bf16_pt, pos_ids_4d[1:])
    mask_kwargs: dict[str, Any] = {
        "config": hf_text_cfg,
        "inputs_embeds": h0_bf16_pt,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": text_pos_ids,
    }
    causal_masks = {
        "full_attention": hf_qwen35_mod.create_causal_mask(**mask_kwargs),
        "linear_attention": hf_qwen35_mod.create_recurrent_attention_mask(
            **mask_kwargs
        ),
    }
    hf_y0_pt = hf_l0(
        h0_bf16_pt,
        position_embeddings=pos_emb,
        attention_mask=causal_masks["linear_attention"],
        position_ids=text_pos_ids,
    )
    l0_parity = _diff_metrics(y0_clean_np, hf_y0_pt.to(torch.float32).numpy())
    del l0_params, hf_l0

    l3_params = verifier.reader.load_single_layer_params(
        3, verifier.abs_attn_params, cfg, total_layers=60
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      y3_clean_chained, _ = verifier.attn_fwd(
          l3_params, y0_clean, seg_clean, pos_clean
      )
      y3_clean_chained_np = np.asarray(y3_clean_chained, dtype=np.float32)

    hf_l3 = _load_hf_decoder_layer(3, verifier.reader.last_hf_tensors)
    verifier.reader.last_hf_tensors = {}
    hf_y3_chained_pt = hf_l3(
        hf_y0_pt,
        position_embeddings=pos_emb,
        attention_mask=causal_masks["full_attention"],
        position_ids=text_pos_ids,
    )
    l3_chained_parity = _diff_metrics(
        y3_clean_chained_np, hf_y3_chained_pt.to(torch.float32).numpy()
    )
    del l3_params, hf_l3

    self.assertGreater(l0_parity["cosine_sim"], 0.999)
    self.assertLess(l0_parity["rel_l2"], 0.03)
    self.assertGreater(l3_chained_parity["cosine_sim"], 0.999)
    self.assertLess(l3_chained_parity["rel_l2"], 0.04)


if __name__ == "__main__":
  absltest.main()
