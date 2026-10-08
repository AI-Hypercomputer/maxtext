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

"""HF GlmMoeDsaForCausalLM reference parity and pure-JAX IndexShare / DS-3.2 invariance tests."""

import os
import tempfile
import unittest

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import index_share_utils
from maxtext.utils import maxtext_utils
import numpy as np
import optax
from tests.utils.test_helpers import get_test_config_path
import transformers

try:
  import torch  # type: ignore
except ImportError:
  torch = None

_HAS_HF_GLM_MOE_DSA = torch is not None and hasattr(transformers, "GlmMoeDsaForCausalLM")


def _common_glm_cli_args(
    tmpdir: str,
    *,
    model_name: str = "glm5.2-744b",
    scan_layers: bool = True,
    prune_shared_indexers: bool | None = None,
    seq_len: int = 8,
    indexer_topk: int = 4,
    indexer_n_heads: int = 16,
    indexer_loss_scaling_factor: float = 0.0,
    indexer_sparse_training: bool = False,
    routed_bias_update_rate: float = 0.0,
    max_prefill_predict_length: int = 4,
    mla_naive_kvcache: bool = True,
) -> list[str]:
  """Builds small-model CLI arguments for GLM-5.1 / GLM-5.2 with L_moe=4 != num_experts=6."""
  args = [
      "",
      get_test_config_path(),
      f"model_name={model_name}",
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      f"base_output_directory={tmpdir}",
      "run_name=glm5_ref_test",
      "enable_checkpointing=False",
      "remat_policy=none",
      f"scan_layers={scan_layers}",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "matmul_precision=highest",
      "normalization_layer_epsilon=1.0e-6",
      "base_emb_dim=32",
      "base_mlp_dim=64",
      "base_moe_mlp_dim=16",
      "base_num_decoder_layers=6",
      "first_num_dense_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "q_lora_rank=16",
      "kv_lora_rank=8",
      "qk_nope_head_dim=8",
      "qk_rope_head_dim=4",
      "v_head_dim=8",
      f"indexer_n_heads={indexer_n_heads}",
      "indexer_head_dim=8",
      f"indexer_topk={indexer_topk}",
      f"indexer_loss_scaling_factor={indexer_loss_scaling_factor}",
      f"indexer_sparse_training={indexer_sparse_training}",
      "num_experts=6",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "n_routing_groups=1",
      "topk_routing_group=1",
      f"routed_bias_update_rate={routed_bias_update_rate}",
      "vocab_size=64",
      "max_position_embeddings=128",
      "original_max_position_embeddings=128",
      f"max_prefill_predict_length={max_prefill_predict_length}",
      f"max_target_length={seq_len}",
      f"mla_naive_kvcache={mla_naive_kvcache}",
      "per_device_batch_size=1",
  ]
  if model_name == "glm5.2-744b":
    args.append("index_share_pattern=FFSSFS")
  if prune_shared_indexers is not None:
    args.append(f"prune_shared_indexers={prune_shared_indexers}")
  return args


def _create_mesh(cfg):
  devices = maxtext_utils.create_device_mesh(cfg)
  return jax.sharding.Mesh(devices, cfg.mesh_axes)


def _copy_hf_weights_to_maxtext(hf_model, mt_model, cfg) -> None:
  """Copies HuggingFace GlmMoeDsaForCausalLM weights into scanned or unscanned MaxText Transformer."""
  n_heads = cfg.base_num_query_heads
  qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
  kv_out_dim = cfg.qk_nope_head_dim + cfg.v_head_dim
  idx_heads = cfg.indexer_n_heads
  idx_dim = cfg.indexer_head_dim
  moe_dim = cfg.base_moe_mlp_dim
  first_dense = cfg.first_num_dense_layers
  num_layers = cfg.base_num_decoder_layers
  pattern = (
      index_share_utils.parse_index_share_pattern(cfg.index_share_pattern, num_layers)
      if cfg.use_index_share
      else ("F",) * num_layers
  )

  sd = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
  mt_model.token_embedder.embedding.set_value(jnp.asarray(sd["model.embed_tokens.weight"]))
  mt_model.decoder.decoder_norm.scale.set_value(jnp.asarray(sd["model.norm.weight"]))
  mt_model.decoder.logits_dense.kernel.set_value(jnp.asarray(sd["lm_head.weight"].T))

  def _attn_layer_weights(layer_idx: int) -> dict[str, np.ndarray]:
    prefix = f"model.layers.{layer_idx}.self_attn"
    layer_dict = {
        "pre_norm": sd[f"model.layers.{layer_idx}.input_layernorm.weight"],
        "post_norm": sd[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
        "wq_a": sd[f"{prefix}.q_a_proj.weight"].T,
        "q_norm": sd[f"{prefix}.q_a_layernorm.weight"],
        "wq_b": sd[f"{prefix}.q_b_proj.weight"].T.reshape(cfg.q_lora_rank, n_heads, qk_head_dim),
        "wkv_a": sd[f"{prefix}.kv_a_proj_with_mqa.weight"].T,
        "kv_norm": sd[f"{prefix}.kv_a_layernorm.weight"],
        "wkv_b": sd[f"{prefix}.kv_b_proj.weight"].T.reshape(cfg.kv_lora_rank, n_heads, kv_out_dim),
        "out": sd[f"{prefix}.o_proj.weight"].T.reshape(n_heads, cfg.v_head_dim, cfg.base_emb_dim),
    }
    donor = index_share_utils.get_donor_layer_idx(layer_idx, pattern) if cfg.use_index_share else layer_idx
    idx_prefix = f"model.layers.{donor}.self_attn.indexer"
    if f"{idx_prefix}.wq_b.weight" in sd:
      layer_dict["idx_wq_b"] = sd[f"{idx_prefix}.wq_b.weight"].T.reshape(cfg.q_lora_rank, idx_heads, idx_dim)
      layer_dict["idx_wk"] = sd[f"{idx_prefix}.wk.weight"].T
      layer_dict["idx_k_scale"] = sd[f"{idx_prefix}.k_norm.weight"]
      layer_dict["idx_k_bias"] = sd[f"{idx_prefix}.k_norm.bias"]
      layer_dict["idx_wproj"] = sd[f"{idx_prefix}.weights_proj.weight"].T
    return layer_dict

  def _set_stacked_attn(block_mod, attn_dicts: list[dict[str, np.ndarray]], axis: int) -> None:
    block_mod.pre_self_attention_layer_norm.scale.set_value(
        jnp.asarray(np.stack([d["pre_norm"] for d in attn_dicts], axis=axis))
    )
    block_mod.post_self_attention_layer_norm.scale.set_value(
        jnp.asarray(np.stack([d["post_norm"] for d in attn_dicts], axis=axis))
    )
    sa = block_mod.self_attention
    sa.wq_a.kernel.set_value(jnp.asarray(np.stack([d["wq_a"] for d in attn_dicts], axis=axis)))
    sa.q_norm.scale.set_value(jnp.asarray(np.stack([d["q_norm"] for d in attn_dicts], axis=axis)))
    sa.wq_b.kernel.set_value(jnp.asarray(np.stack([d["wq_b"] for d in attn_dicts], axis=axis)))
    sa.wkv_a.kernel.set_value(jnp.asarray(np.stack([d["wkv_a"] for d in attn_dicts], axis=axis)))
    sa.kv_norm.scale.set_value(jnp.asarray(np.stack([d["kv_norm"] for d in attn_dicts], axis=axis)))
    sa.wkv_b.kernel.set_value(jnp.asarray(np.stack([d["wkv_b"] for d in attn_dicts], axis=axis)))
    sa.out.kernel.set_value(jnp.asarray(np.stack([d["out"] for d in attn_dicts], axis=axis)))
    sa.indexer.wq_b.kernel.set_value(jnp.asarray(np.stack([d["idx_wq_b"] for d in attn_dicts], axis=axis)))
    sa.indexer.wk.kernel.set_value(jnp.asarray(np.stack([d["idx_wk"] for d in attn_dicts], axis=axis)))
    sa.indexer.k_norm.scale.set_value(jnp.asarray(np.stack([d["idx_k_scale"] for d in attn_dicts], axis=axis)))
    sa.indexer.k_norm.bias.set_value(jnp.asarray(np.stack([d["idx_k_bias"] for d in attn_dicts], axis=axis)))
    sa.indexer.weights_proj.kernel.set_value(jnp.asarray(np.stack([d["idx_wproj"] for d in attn_dicts], axis=axis)))

  if cfg.scan_layers:
    ax = cfg.param_scan_axis
    dense_range = list(range(first_dense))
    dense_mod = mt_model.decoder.dense_layers
    _set_stacked_attn(dense_mod, [_attn_layer_weights(i) for i in dense_range], ax)
    dense_mod.mlp.wi_0.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.gate_proj.weight"].T for i in dense_range], axis=ax))
    )
    dense_mod.mlp.wi_1.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.up_proj.weight"].T for i in dense_range], axis=ax))
    )
    dense_mod.mlp.wo.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.down_proj.weight"].T for i in dense_range], axis=ax))
    )

    moe_range = list(range(first_dense, num_layers))
    moe_mod = mt_model.decoder.moe_layers
    _set_stacked_attn(moe_mod, [_attn_layer_weights(i) for i in moe_range], ax)
    moe_blk = moe_mod.DeepSeekMoeBlock_0
    moe_blk.shared_experts.wi_0.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"].T for i in moe_range], axis=ax))
    )
    moe_blk.shared_experts.wi_1.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].T for i in moe_range], axis=ax))
    )
    moe_blk.shared_experts.wo.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"].T for i in moe_range], axis=ax))
    )
    moe_blk.MoeBlock_0.gate.kernel.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.gate.weight"].T for i in moe_range], axis=ax))
    )
    # MoEBiasVar is non-Param and therefore stacked on axis 0: [L_moe, num_experts].
    moe_blk.MoeBlock_0.gate.bias.set_value(
        jnp.asarray(np.stack([sd[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] for i in moe_range], axis=0))
    )
    wi0_list, wi1_list, wo_list = [], [], []
    for i in moe_range:
      gate_up = sd[f"model.layers.{i}.mlp.experts.gate_up_proj"]
      wi0_list.append(np.transpose(gate_up[:, :moe_dim, :], (0, 2, 1)))
      wi1_list.append(np.transpose(gate_up[:, moe_dim:, :], (0, 2, 1)))
      wo_list.append(np.transpose(sd[f"model.layers.{i}.mlp.experts.down_proj"], (0, 2, 1)))
    moe_blk.MoeBlock_0.wi_0.set_value(jnp.asarray(np.stack(wi0_list, axis=ax)))
    moe_blk.MoeBlock_0.wi_1.set_value(jnp.asarray(np.stack(wi1_list, axis=ax)))
    moe_blk.MoeBlock_0.wo.set_value(jnp.asarray(np.stack(wo_list, axis=ax)))
  else:
    for i in range(num_layers):
      attn_d = _attn_layer_weights(i)
      lyr_name = f"dense_layers_{i}" if i < first_dense else f"moe_layers_{i - first_dense}"
      lyr = getattr(mt_model.decoder, lyr_name)
      lyr.pre_self_attention_layer_norm.scale.set_value(jnp.asarray(attn_d["pre_norm"]))
      lyr.post_self_attention_layer_norm.scale.set_value(jnp.asarray(attn_d["post_norm"]))
      sa = lyr.self_attention
      sa.wq_a.kernel.set_value(jnp.asarray(attn_d["wq_a"]))
      sa.q_norm.scale.set_value(jnp.asarray(attn_d["q_norm"]))
      sa.wq_b.kernel.set_value(jnp.asarray(attn_d["wq_b"]))
      sa.wkv_a.kernel.set_value(jnp.asarray(attn_d["wkv_a"]))
      sa.kv_norm.scale.set_value(jnp.asarray(attn_d["kv_norm"]))
      sa.wkv_b.kernel.set_value(jnp.asarray(attn_d["wkv_b"]))
      sa.out.kernel.set_value(jnp.asarray(attn_d["out"]))
      if sa.indexer is not None:
        sa.indexer.wq_b.kernel.set_value(jnp.asarray(attn_d["idx_wq_b"]))
        sa.indexer.wk.kernel.set_value(jnp.asarray(attn_d["idx_wk"]))
        sa.indexer.k_norm.scale.set_value(jnp.asarray(attn_d["idx_k_scale"]))
        sa.indexer.k_norm.bias.set_value(jnp.asarray(attn_d["idx_k_bias"]))
        sa.indexer.weights_proj.kernel.set_value(jnp.asarray(attn_d["idx_wproj"]))
      if i < first_dense:
        lyr.mlp.wi_0.kernel.set_value(jnp.asarray(sd[f"model.layers.{i}.mlp.gate_proj.weight"].T))
        lyr.mlp.wi_1.kernel.set_value(jnp.asarray(sd[f"model.layers.{i}.mlp.up_proj.weight"].T))
        lyr.mlp.wo.kernel.set_value(jnp.asarray(sd[f"model.layers.{i}.mlp.down_proj.weight"].T))
      else:
        moe_blk = lyr.DeepSeekMoeBlock_0
        moe_blk.shared_experts.wi_0.kernel.set_value(
            jnp.asarray(sd[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"].T)
        )
        moe_blk.shared_experts.wi_1.kernel.set_value(
            jnp.asarray(sd[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].T)
        )
        moe_blk.shared_experts.wo.kernel.set_value(
            jnp.asarray(sd[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"].T)
        )
        moe_blk.MoeBlock_0.gate.kernel.set_value(jnp.asarray(sd[f"model.layers.{i}.mlp.gate.weight"].T))
        moe_blk.MoeBlock_0.gate.bias.set_value(jnp.asarray(sd[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]))
        gate_up = sd[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        moe_blk.MoeBlock_0.wi_0.set_value(jnp.asarray(np.transpose(gate_up[:, :moe_dim, :], (0, 2, 1))))
        moe_blk.MoeBlock_0.wi_1.set_value(jnp.asarray(np.transpose(gate_up[:, moe_dim:, :], (0, 2, 1))))
        moe_blk.MoeBlock_0.wo.set_value(
            jnp.asarray(np.transpose(sd[f"model.layers.{i}.mlp.experts.down_proj"], (0, 2, 1)))
        )


def _copy_scanned_to_unscanned_jax(mt_scanned, mt_unscanned, cfg) -> None:
  """Copies weights from a scanned MaxText Transformer to an unscanned Transformer (pure JAX)."""
  ax = cfg.param_scan_axis
  first_dense = cfg.first_num_dense_layers
  num_layers = cfg.base_num_decoder_layers

  mt_unscanned.token_embedder.embedding.set_value(mt_scanned.token_embedder.embedding.get_value())
  mt_unscanned.decoder.decoder_norm.scale.set_value(mt_scanned.decoder.decoder_norm.scale.get_value())
  mt_unscanned.decoder.logits_dense.kernel.set_value(mt_scanned.decoder.logits_dense.kernel.get_value())

  def _copy_attn_slice(src_block, dst_layer, local_idx: int) -> None:
    dst_layer.pre_self_attention_layer_norm.scale.set_value(
        jnp.take(src_block.pre_self_attention_layer_norm.scale.get_value(), local_idx, axis=ax)
    )
    dst_layer.post_self_attention_layer_norm.scale.set_value(
        jnp.take(src_block.post_self_attention_layer_norm.scale.get_value(), local_idx, axis=ax)
    )
    src_sa = src_block.self_attention
    dst_sa = dst_layer.self_attention
    dst_sa.wq_a.kernel.set_value(jnp.take(src_sa.wq_a.kernel.get_value(), local_idx, axis=ax))
    dst_sa.q_norm.scale.set_value(jnp.take(src_sa.q_norm.scale.get_value(), local_idx, axis=ax))
    dst_sa.wq_b.kernel.set_value(jnp.take(src_sa.wq_b.kernel.get_value(), local_idx, axis=ax))
    dst_sa.wkv_a.kernel.set_value(jnp.take(src_sa.wkv_a.kernel.get_value(), local_idx, axis=ax))
    dst_sa.kv_norm.scale.set_value(jnp.take(src_sa.kv_norm.scale.get_value(), local_idx, axis=ax))
    dst_sa.wkv_b.kernel.set_value(jnp.take(src_sa.wkv_b.kernel.get_value(), local_idx, axis=ax))
    dst_sa.out.kernel.set_value(jnp.take(src_sa.out.kernel.get_value(), local_idx, axis=ax))
    if dst_sa.indexer is not None:
      dst_sa.indexer.wq_b.kernel.set_value(jnp.take(src_sa.indexer.wq_b.kernel.get_value(), local_idx, axis=ax))
      dst_sa.indexer.wk.kernel.set_value(jnp.take(src_sa.indexer.wk.kernel.get_value(), local_idx, axis=ax))
      dst_sa.indexer.k_norm.scale.set_value(jnp.take(src_sa.indexer.k_norm.scale.get_value(), local_idx, axis=ax))
      dst_sa.indexer.k_norm.bias.set_value(jnp.take(src_sa.indexer.k_norm.bias.get_value(), local_idx, axis=ax))
      dst_sa.indexer.weights_proj.kernel.set_value(
          jnp.take(src_sa.indexer.weights_proj.kernel.get_value(), local_idx, axis=ax)
      )

  src_dense = mt_scanned.decoder.dense_layers
  for i in range(first_dense):
    dst_lyr = getattr(mt_unscanned.decoder, f"dense_layers_{i}")
    _copy_attn_slice(src_dense, dst_lyr, i)
    dst_lyr.mlp.wi_0.kernel.set_value(jnp.take(src_dense.mlp.wi_0.kernel.get_value(), i, axis=ax))
    dst_lyr.mlp.wi_1.kernel.set_value(jnp.take(src_dense.mlp.wi_1.kernel.get_value(), i, axis=ax))
    dst_lyr.mlp.wo.kernel.set_value(jnp.take(src_dense.mlp.wo.kernel.get_value(), i, axis=ax))

  src_moe = mt_scanned.decoder.moe_layers
  src_mb = src_moe.DeepSeekMoeBlock_0
  for i in range(first_dense, num_layers):
    m_idx = i - first_dense
    dst_lyr = getattr(mt_unscanned.decoder, f"moe_layers_{m_idx}")
    _copy_attn_slice(src_moe, dst_lyr, m_idx)
    dst_mb = dst_lyr.DeepSeekMoeBlock_0
    dst_mb.shared_experts.wi_0.kernel.set_value(jnp.take(src_mb.shared_experts.wi_0.kernel.get_value(), m_idx, axis=ax))
    dst_mb.shared_experts.wi_1.kernel.set_value(jnp.take(src_mb.shared_experts.wi_1.kernel.get_value(), m_idx, axis=ax))
    dst_mb.shared_experts.wo.kernel.set_value(jnp.take(src_mb.shared_experts.wo.kernel.get_value(), m_idx, axis=ax))
    dst_mb.MoeBlock_0.gate.kernel.set_value(jnp.take(src_mb.MoeBlock_0.gate.kernel.get_value(), m_idx, axis=ax))
    dst_mb.MoeBlock_0.gate.bias.set_value(jnp.take(src_mb.MoeBlock_0.gate.bias.get_value(), m_idx, axis=0))
    dst_mb.MoeBlock_0.wi_0.set_value(jnp.take(src_mb.MoeBlock_0.wi_0.get_value(), m_idx, axis=ax))
    dst_mb.MoeBlock_0.wi_1.set_value(jnp.take(src_mb.MoeBlock_0.wi_1.get_value(), m_idx, axis=ax))
    dst_mb.MoeBlock_0.wo.set_value(jnp.take(src_mb.MoeBlock_0.wo.get_value(), m_idx, axis=ax))


@unittest.skipUnless(_HAS_HF_GLM_MOE_DSA, "Requires torch and transformers.GlmMoeDsaForCausalLM")
class Glm5HfReferenceParityTest(parameterized.TestCase):
  """End-to-end forward logits parity against HuggingFace PyTorch GlmMoeDsaForCausalLM."""

  @parameterized.named_parameters(
      {
          "testcase_name": "glm51_dense_regime_seq_le_topk",
          "model_name": "glm5.1-744b",
          "use_index_share": False,
          "seq_len": 4,
          "indexer_topk": 6,
      },
      {
          "testcase_name": "glm51_sparse_regime_seq_gt_topk",
          "model_name": "glm5.1-744b",
          "use_index_share": False,
          "seq_len": 10,
          "indexer_topk": 4,
      },
      {
          "testcase_name": "glm52_index_share_dense_regime_seq_le_topk",
          "model_name": "glm5.2-744b",
          "use_index_share": True,
          "seq_len": 4,
          "indexer_topk": 6,
      },
      {
          "testcase_name": "glm52_index_share_sparse_regime_seq_gt_topk",
          "model_name": "glm5.2-744b",
          "use_index_share": True,
          "seq_len": 10,
          "indexer_topk": 4,
      },
  )
  def test_forward_logits_parity_vs_hf_glm_moe_dsa(
      self,
      model_name: str,
      use_index_share: bool,
      seq_len: int,
      indexer_topk: int,
  ):
    """Verifies scanned == unscanned (pruned) == HF GlmMoeDsaForCausalLM logits in both top-k regimes."""
    torch.manual_seed(123)
    indexer_types = ["full", "full", "shared", "shared", "full", "shared"] if use_index_share else ["full"] * 6
    rope_theta = 8000000.0 if use_index_share else 1000000.0
    hf_cfg = transformers.GlmMoeDsaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=6,
        routed_scaling_factor=2.5,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=8,
        v_head_dim=8,
        head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        first_k_dense_replace=2,
        index_n_heads=16,
        index_head_dim=8,
        index_topk=indexer_topk,
        mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
        indexer_types=indexer_types,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    hf_cfg._attn_implementation = "eager"  # pylint: disable=protected-access
    hf_model = transformers.GlmMoeDsaForCausalLM(hf_cfg).float().eval()
    with torch.no_grad():
      for layer_idx in range(2, 6):
        hf_model.model.layers[layer_idx].mlp.gate.e_score_correction_bias.copy_(
            torch.linspace(-0.15, 0.25, 6, dtype=torch.float32) * (layer_idx - 1)
        )

    if use_index_share:
      for s_idx in (2, 3, 5):
        self.assertIsNone(hf_model.model.layers[s_idx].self_attn.indexer)
      for f_idx in (0, 1, 4):
        self.assertIsNotNone(hf_model.model.layers[f_idx].self_attn.indexer)

    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_scan = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name=model_name,
              scan_layers=True,
              prune_shared_indexers=False,
              seq_len=seq_len,
              indexer_topk=indexer_topk,
              indexer_n_heads=16,
          )
      )
      cfg_unscan = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name=model_name,
              scan_layers=False,
              prune_shared_indexers=use_index_share,
              seq_len=seq_len,
              indexer_topk=indexer_topk,
              indexer_n_heads=16,
          )
      )
      mesh = _create_mesh(cfg_scan)
      batch_size = cfg_scan.global_batch_size_to_train_on
      toks_np = np.random.default_rng(77).integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)

      with torch.no_grad():
        hf_logits = hf_model(torch.from_numpy(toks_np).long()).logits.cpu().numpy()

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scan.logical_axis_rules):
        mt_scan = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=0, dropout=1),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        mt_unscan = models.Transformer(
            cfg_unscan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan),
            rngs=nnx.Rngs(params=0, dropout=1),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        if use_index_share:
          for s_idx in (2, 3, 5):
            self.assertIsNone(getattr(mt_unscan.decoder, f"moe_layers_{s_idx - 2}").self_attention.indexer)

        _copy_hf_weights_to_maxtext(hf_model, mt_scan, cfg_scan)
        _copy_hf_weights_to_maxtext(hf_model, mt_unscan, cfg_unscan)

        toks_jax = jnp.asarray(toks_np)
        pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        logits_scan = np.asarray(
            mt_scan(
                toks_jax,
                pos,
                decoder_segment_ids=seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_TRAIN,
            )
        )
        logits_unscan = np.asarray(
            mt_unscan(
                toks_jax,
                pos,
                decoder_segment_ids=seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_TRAIN,
            )
        )

      np.testing.assert_allclose(logits_scan, logits_unscan, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(logits_scan, hf_logits, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(logits_unscan, hf_logits, rtol=1e-4, atol=1e-4)


class Glm5IndexSharePureJaxTest(parameterized.TestCase):
  """Unconditional pure-JAX tests for GLM-5.2 IndexShare, MoEBiasVar, and DeepSeek-3.2 invariance."""

  @parameterized.named_parameters(
      {"testcase_name": "seq_le_topk", "seq_len": 4, "indexer_topk": 6},
      {"testcase_name": "seq_gt_topk", "seq_len": 10, "indexer_topk": 4},
  )
  def test_scanned_vs_unscanned_forward_and_backward_parity(self, seq_len: int, indexer_topk: int):
    """Verifies scanned vs unscanned (pruned) forward loss/logits and backward Param gradients."""
    with tempfile.TemporaryDirectory() as tmpdir:
      is_sparse = seq_len > indexer_topk
      idx_scale = 0.25 if is_sparse else 0.0
      cfg_scan = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=True,
              prune_shared_indexers=False,
              seq_len=seq_len,
              indexer_topk=indexer_topk,
              indexer_loss_scaling_factor=idx_scale,
              indexer_sparse_training=is_sparse,
          )
      )
      cfg_unscan = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=False,
              prune_shared_indexers=True,
              seq_len=seq_len,
              indexer_topk=indexer_topk,
              indexer_loss_scaling_factor=idx_scale,
              indexer_sparse_training=is_sparse,
          )
      )
      mesh = _create_mesh(cfg_scan)
      batch_size = cfg_scan.global_batch_size_to_train_on
      rng_np = np.random.default_rng(2026)
      inputs_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)
      targets_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scan.logical_axis_rules):
        mt_scan = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=11, dropout=12),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        # Set non-zero e_score_correction_bias on the 4 MoE layers to exercise MoEBiasVar routing.
        bias_init = jnp.asarray(rng_np.uniform(-0.2, 0.2, size=(4, 6)).astype(np.float32))
        mt_scan.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.set_value(bias_init)

        mt_unscan = models.Transformer(
            cfg_unscan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan),
            rngs=nnx.Rngs(params=99, dropout=98),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        # Pruned unscanned Shared layers {2, 3, 5} must have self_attention.indexer is None.
        for s_idx in (2, 3, 5):
          self.assertIsNone(getattr(mt_unscan.decoder, f"moe_layers_{s_idx - 2}").self_attention.indexer)
        for f_idx in (0, 1, 4):
          lyr_tok = f"dense_layers_{f_idx}" if f_idx < 2 else f"moe_layers_{f_idx - 2}"
          self.assertIsNotNone(getattr(mt_unscan.decoder, lyr_tok).self_attention.indexer)

        _copy_scanned_to_unscanned_jax(mt_scan, mt_unscan, cfg_scan)

        pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        batch = {
            "inputs": jnp.asarray(inputs_np),
            "inputs_position": pos,
            "inputs_segmentation": seg,
            "targets": jnp.asarray(targets_np),
            "targets_position": pos,
            "targets_segmentation": seg,
        }

        logits_s = np.asarray(
            mt_scan(
                batch["inputs"],
                pos,
                decoder_segment_ids=seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_TRAIN,
            )
        )
        logits_u = np.asarray(
            mt_unscan(
                batch["inputs"],
                pos,
                decoder_segment_ids=seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_TRAIN,
            )
        )
        np.testing.assert_allclose(logits_s, logits_u, rtol=1e-5, atol=1e-5)

        def _loss_and_grads(model_obj, cfg_obj):
          nnx.pop(model_obj, nnx.Intermediate)
          nnx.pop(model_obj, attention_mla.indexer_losses)
          gdef, params, rest = nnx.split(model_obj, nnx.Param, ...)

          def _fn(p):
            m = nnx.merge(gdef, p, rest, copy=True)
            loss_val, aux = pre_train.loss_fn(m, cfg_obj, batch, None, None, is_train=True)
            idx_l = aux.get("indexer_loss")
            return loss_val, (jnp.float32(0.0) if idx_l is None else idx_l)

          return jax.value_and_grad(_fn, has_aux=True)(params)

        (loss_s, idx_loss_s), grads_s = _loss_and_grads(mt_scan, cfg_scan)
        (loss_u, idx_loss_u), grads_u = _loss_and_grads(mt_unscan, cfg_unscan)

      np.testing.assert_allclose(float(loss_s), float(loss_u), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(float(idx_loss_s), float(idx_loss_u), rtol=1e-5, atol=1e-5)

      ax = cfg_scan.param_scan_axis
      np.testing.assert_allclose(
          np.asarray(grads_s["token_embedder"]["embedding"].get_value()),
          np.asarray(grads_u["token_embedder"]["embedding"].get_value()),
          rtol=1e-4,
          atol=1e-4,
      )
      np.testing.assert_allclose(
          np.asarray(grads_s["decoder"]["logits_dense"]["kernel"].get_value()),
          np.asarray(grads_u["decoder"]["logits_dense"]["kernel"].get_value()),
          rtol=1e-4,
          atol=1e-4,
      )

      # Compare attention & MLP gradients across all 6 layers and verify Shared-layer indexer grads are 0.
      for f_dense in range(2):
        np.testing.assert_allclose(
            np.take(
                np.asarray(grads_s["decoder"]["dense_layers"]["self_attention"]["out"]["kernel"].get_value()),
                f_dense,
                axis=ax,
            ),
            np.asarray(grads_u["decoder"][f"dense_layers_{f_dense}"]["self_attention"]["out"]["kernel"].get_value()),
            rtol=1e-4,
            atol=1e-4,
        )
        g_idx_s = np.take(
            np.asarray(grads_s["decoder"]["dense_layers"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()),
            f_dense,
            axis=ax,
        )
        g_idx_u = np.asarray(
            grads_u["decoder"][f"dense_layers_{f_dense}"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()
        )
        np.testing.assert_allclose(g_idx_s, g_idx_u, rtol=1e-4, atol=1e-4)
        if seq_len > indexer_topk:
          self.assertGreater(float(np.linalg.norm(g_idx_s)), 0.0)

      for m_idx in range(4):
        global_idx = m_idx + 2
        np.testing.assert_allclose(
            np.take(
                np.asarray(grads_s["decoder"]["moe_layers"]["self_attention"]["out"]["kernel"].get_value()),
                m_idx,
                axis=ax,
            ),
            np.asarray(grads_u["decoder"][f"moe_layers_{m_idx}"]["self_attention"]["out"]["kernel"].get_value()),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.take(
                np.asarray(grads_s["decoder"]["moe_layers"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"].get_value()),
                m_idx,
                axis=ax,
            ),
            np.asarray(grads_u["decoder"][f"moe_layers_{m_idx}"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"].get_value()),
            rtol=1e-4,
            atol=1e-4,
        )
        g_moe_idx_s = np.take(
            np.asarray(grads_s["decoder"]["moe_layers"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()),
            m_idx,
            axis=ax,
        )
        if global_idx == 4:
          g_moe_idx_u = np.asarray(
              grads_u["decoder"][f"moe_layers_{m_idx}"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()
          )
          np.testing.assert_allclose(g_moe_idx_s, g_moe_idx_u, rtol=1e-4, atol=1e-4)
          if seq_len > indexer_topk:
            self.assertGreater(float(np.linalg.norm(g_moe_idx_s)), 0.0)
        else:
          # Shared layers {2, 3, 5} never execute their own indexer in scan mode -> exact zero gradient.
          np.testing.assert_array_equal(g_moe_idx_s, np.zeros_like(g_moe_idx_s))

  def test_prefill_and_greedy_ar_decode_parity_and_cache_isolation(self):
    """Verifies prefill + greedy AR decode vs teacher-forced forward, cache isolation, and missing-donor fast-fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
      common_kw = {
          "model_name": "glm5.2-744b",
          "seq_len": 10,
          "indexer_topk": 4,
          "indexer_n_heads": 8,
          "max_prefill_predict_length": 4,
          "mla_naive_kvcache": False,
      }
      cfg_scan = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=True, prune_shared_indexers=False, **common_kw)
      )
      cfg_unscan_pruned = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=False, prune_shared_indexers=True, **common_kw)
      )
      cfg_unscan_retained = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=False, prune_shared_indexers=False, **common_kw)
      )
      mesh = _create_mesh(cfg_scan)
      batch_size = cfg_scan.global_batch_size_to_train_on
      prompt_np = np.arange(batch_size * 4, dtype=np.int32).reshape(batch_size, 4) % 64

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scan.logical_axis_rules):
        m_scan_ar = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=3, dropout=4),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        m_unscan_pruned_ar = models.Transformer(
            cfg_unscan_pruned,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan_pruned),
            rngs=nnx.Rngs(params=5, dropout=6),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        m_unscan_retained_ar = models.Transformer(
            cfg_unscan_retained,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan_retained),
            rngs=nnx.Rngs(params=7, dropout=8),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        m_train_tf = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=9, dropout=10),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )

        _copy_scanned_to_unscanned_jax(m_scan_ar, m_unscan_pruned_ar, cfg_scan)
        _copy_scanned_to_unscanned_jax(m_scan_ar, m_unscan_retained_ar, cfg_scan)
        nnx.update(m_train_tf, nnx.state(m_scan_ar, nnx.Param))
        m_train_tf.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.set_value(
            m_scan_ar.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.get_value()
        )

        # Fast-fail check: calling an unscanned Shared layer with cached_indexer_state=None raises ValueError.
        shared_layer = m_unscan_pruned_ar.decoder.moe_layers_0
        dummy_h = jnp.ones((batch_size, 4, cfg_scan.base_emb_dim), dtype=jnp.float32)
        dummy_pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_size, 4))
        dummy_seg = jnp.ones((batch_size, 4), dtype=jnp.int32)
        with self.assertRaisesRegex(
            ValueError, "Shared DSA layers require top-k indices from a previous full indexer layer"
        ):
          shared_layer(
              dummy_h,
              dummy_seg,
              dummy_pos,
              True,
              common_types.MODEL_MODE_TRAIN,
              cached_indexer_state=None,
          )

        def _rollout_ar(model_ar):
          prompt = jnp.asarray(prompt_np)
          p_pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_size, 4))
          p_seg = jnp.ones((batch_size, 4), dtype=jnp.int32)
          prefill_logits = model_ar(
              prompt,
              p_pos,
              decoder_segment_ids=p_seg,
              enable_dropout=False,
              model_mode=common_types.MODEL_MODE_PREFILL,
          )
          next_tok = jnp.argmax(prefill_logits[:, -1:, :], axis=-1).astype(jnp.int32)
          chunks = [prompt, next_tok]
          pred_logits = [prefill_logits[:, -1, :]]
          # Steps 4, 5 cross from kv_len=4 (<= indexer_topk) into kv_len=5, 6 (> indexer_topk).
          for step_pos in range(4, 6):
            s_pos = jnp.full((batch_size, 1), step_pos, dtype=jnp.int32)
            ar_logits = model_ar(
                next_tok,
                s_pos,
                decoder_segment_ids=None,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            )
            pred_logits.append(ar_logits[:, 0, :])
            next_tok = jnp.argmax(ar_logits[:, -1:, :], axis=-1).astype(jnp.int32)
            chunks.append(next_tok)
          full_tokens = jnp.concatenate(chunks, axis=1)  # [batch_size, 7]
          stacked_logits = np.stack([np.asarray(x) for x in pred_logits], axis=1)  # [batch_size, 3, vocab]
          return np.asarray(full_tokens), stacked_logits

        tokens_scan, logits_scan = _rollout_ar(m_scan_ar)
        tokens_pruned, logits_pruned = _rollout_ar(m_unscan_pruned_ar)
        tokens_retained, logits_retained = _rollout_ar(m_unscan_retained_ar)

        tf_input = jnp.asarray(tokens_scan[:, :6])
        tf_pos = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32), (batch_size, 6))
        tf_seg = jnp.ones((batch_size, 6), dtype=jnp.int32)
        tf_all_logits = np.asarray(
            m_train_tf(
                tf_input,
                tf_pos,
                decoder_segment_ids=tf_seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_TRAIN,
            )
        )
        tf_pred_logits = tf_all_logits[:, 3:6, :]
        tf_greedy_tokens = np.argmax(tf_pred_logits, axis=-1).astype(np.int32)

      # Verify greedy rollout tokens and step-by-step logits match across all 3 AR modes and teacher-forced train mode.
      np.testing.assert_array_equal(tokens_scan, tokens_pruned)
      np.testing.assert_array_equal(tokens_scan, tokens_retained)
      np.testing.assert_array_equal(tokens_scan[:, 4:7], tf_greedy_tokens)
      np.testing.assert_allclose(logits_scan, logits_pruned, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(logits_scan, logits_retained, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(logits_scan, tf_pred_logits, rtol=1e-4, atol=1e-4)

      # Verify IndexerKVCache_0 isolation on scanned model (`FFSSFS`: dense [0,1] & moe [2] Full; moe [0,1,3] Shared).
      dense_pk = np.asarray(m_scan_ar.decoder.dense_layers.self_attention.IndexerKVCache_0.cached_prefill_key[...])
      dense_ak = np.asarray(m_scan_ar.decoder.dense_layers.self_attention.IndexerKVCache_0.cached_ar_key[...])
      for d_idx in (0, 1):
        self.assertGreater(float(np.linalg.norm(dense_pk[d_idx])), 0.0)
        self.assertGreater(float(np.linalg.norm(dense_ak[d_idx])), 0.0)

      moe_pk = np.asarray(m_scan_ar.decoder.moe_layers.self_attention.IndexerKVCache_0.cached_prefill_key[...])
      moe_ak = np.asarray(m_scan_ar.decoder.moe_layers.self_attention.IndexerKVCache_0.cached_ar_key[...])
      self.assertGreater(float(np.linalg.norm(moe_pk[2])), 0.0)
      self.assertGreater(float(np.linalg.norm(moe_ak[2])), 0.0)
      for s_local in (0, 1, 3):
        np.testing.assert_array_equal(moe_pk[s_local], np.zeros_like(moe_pk[s_local]))
        np.testing.assert_array_equal(moe_ak[s_local], np.zeros_like(moe_ak[s_local]))

      # Verify unscanned pruned Shared layers have IndexerKVCache_0 is None, and unscanned retained stay all zeros.
      for s_local in (0, 1, 3):
        self.assertIsNone(getattr(m_unscan_pruned_ar.decoder, f"moe_layers_{s_local}").self_attention.IndexerKVCache_0)
        ret_cache = getattr(m_unscan_retained_ar.decoder, f"moe_layers_{s_local}").self_attention.IndexerKVCache_0
        self.assertIsNotNone(ret_cache)
        np.testing.assert_array_equal(
            np.asarray(ret_cache.cached_prefill_key[...]),
            np.zeros_like(np.asarray(ret_cache.cached_prefill_key[...])),
        )
        np.testing.assert_array_equal(
            np.asarray(ret_cache.cached_ar_key[...]),
            np.zeros_like(np.asarray(ret_cache.cached_ar_key[...])),
        )
      ret_full_cache = m_unscan_retained_ar.decoder.moe_layers_2.self_attention.IndexerKVCache_0
      self.assertGreater(float(np.linalg.norm(np.asarray(ret_full_cache.cached_prefill_key[...]))), 0.0)
      self.assertGreater(float(np.linalg.norm(np.asarray(ret_full_cache.cached_ar_key[...]))), 0.0)

  def test_group_normalized_indexer_loss_and_gradients(self):
    """Verifies group-normalized calculate_indexer_loss (1/served_group_size) and finite Full-layer gradients."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=False,
              prune_shared_indexers=True,
              seq_len=10,
              indexer_topk=4,
              indexer_loss_scaling_factor=0.6,
          )
      )
      mesh = _create_mesh(cfg)
      batch_size = cfg.global_batch_size_to_train_on
      seq_len = 10

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        mla_kw = {
            "config": cfg,
            "num_query_heads": cfg.num_query_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "max_target_length": seq_len,
            "mesh": mesh,
            "attention_kernel": "dot_product",
            "inputs_q_shape": (batch_size, seq_len, cfg.emb_dim),
            "inputs_kv_shape": (batch_size, seq_len, cfg.emb_dim),
            "dtype": cfg.dtype,
            "weight_dtype": cfg.weight_dtype,
            "q_lora_rank": cfg.q_lora_rank,
            "kv_lora_rank": cfg.kv_lora_rank,
            "qk_nope_head_dim": cfg.qk_nope_head_dim,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "v_head_dim": cfg.v_head_dim,
            "max_position_embeddings": cfg.max_position_embeddings,
            "original_max_position_embeddings": cfg.original_max_position_embeddings,
            "mscale": cfg.mscale,
            "rope_factor": cfg.rope_factor,
            "model_mode": common_types.MODEL_MODE_TRAIN,
            "is_shared_layer": False,
        }
        mla_g1 = attention_mla.MLA(**mla_kw, served_group_size=1, rngs=nnx.Rngs(params=21))
        mla_g3 = attention_mla.MLA(**mla_kw, served_group_size=3, rngs=nnx.Rngs(params=21))
        nnx.update(mla_g3, nnx.state(mla_g1, nnx.Param))

        x = jax.random.normal(jax.random.PRNGKey(31), (batch_size, seq_len, cfg.emb_dim), dtype=jnp.float32)
        pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        def _sum_loss(mla_mod):
          _, _, _ = mla_mod(
              x,
              x,
              pos,
              seg,
              deterministic=True,
              model_mode=common_types.MODEL_MODE_TRAIN,
              return_indexer_state=True,
          )
          return mla_mod.indexer_loss.get_value()

        loss_g1, grads_g1 = nnx.value_and_grad(_sum_loss)(mla_g1)
        loss_g3, grads_g3 = nnx.value_and_grad(_sum_loss)(mla_g3)

        self.assertTrue(np.isfinite(float(loss_g1)))
        self.assertGreater(float(loss_g1), 0.0)
        # Group size 3 scales indexer loss and its gradients by exactly 1/3 relative to group size 1.
        np.testing.assert_allclose(float(loss_g1), 3.0 * float(loss_g3), rtol=1e-5, atol=1e-6)

        for key_fn in (
            lambda g: g.indexer.wq_b.kernel.get_value(),
            lambda g: g.indexer.wk.kernel.get_value(),
            lambda g: g.indexer.k_norm.scale.get_value(),
            lambda g: g.indexer.k_norm.bias.get_value(),
            lambda g: g.indexer.weights_proj.kernel.get_value(),
        ):
          arr1 = np.asarray(key_fn(grads_g1))
          arr3 = np.asarray(key_fn(grads_g3))
          self.assertTrue(np.all(np.isfinite(arr1)))
          self.assertGreater(float(np.linalg.norm(arr1)), 0.0)
          np.testing.assert_allclose(arr1, 3.0 * arr3, rtol=1e-5, atol=1e-6)

        # Also verify in the 6-layer unscanned Transformer (`FFSSFS`) that multi-layer group-normalized
        # distillation sows finite positive indexer_loss on every served layer while routing non-zero
        # gradients exclusively into the Full donor layers {0, 1, 4}.
        mt_unscan = models.Transformer(
            cfg,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg),
            rngs=nnx.Rngs(params=41, dropout=42),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        toks = jnp.asarray(np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len) % 64)

        def _layer_mod(m, idx):
          return getattr(m.decoder, f"dense_layers_{idx}" if idx < 2 else f"moe_layers_{idx - 2}")

        def _total_indexer_loss(m):
          _ = m(toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)
          losses = [_layer_mod(m, i).self_attention.indexer_loss.get_value() for i in range(6)]
          return sum(losses)

        def _layer1_only_indexer_loss(m):
          _ = m(toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)
          return _layer_mod(m, 1).self_attention.indexer_loss.get_value()

        total_idx_loss, full_grads = nnx.value_and_grad(_total_indexer_loss)(mt_unscan)
        _, l1_only_grads = nnx.value_and_grad(_layer1_only_indexer_loss)(mt_unscan)
        self.assertTrue(np.isfinite(float(total_idx_loss)))
        self.assertGreater(float(total_idx_loss), 0.0)

        for i in range(6):
          lyr = _layer_mod(mt_unscan, i)
          self.assertTrue(hasattr(lyr.self_attention, "indexer_loss"))
          self.assertIsInstance(lyr.self_attention.indexer_loss, attention_mla.indexer_losses)
          self.assertGreater(float(lyr.self_attention.indexer_loss.get_value()), 0.0)

        for f_idx in (0, 1, 4):
          g_idx = _layer_mod(full_grads, f_idx).self_attention.indexer
          for arr in (
              np.asarray(g_idx.wq_b.kernel.get_value()),
              np.asarray(g_idx.wk.kernel.get_value()),
              np.asarray(g_idx.k_norm.scale.get_value()),
              np.asarray(g_idx.k_norm.bias.get_value()),
              np.asarray(g_idx.weights_proj.kernel.get_value()),
          ):
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertGreater(float(np.linalg.norm(arr)), 0.0)

        # Shared layers {2, 3} contribute non-zero distillation gradient back into donor Full layer 1.
        g_l1_total = np.asarray(_layer_mod(full_grads, 1).self_attention.indexer.wq_b.kernel.get_value())
        g_l1_self = np.asarray(_layer_mod(l1_only_grads, 1).self_attention.indexer.wq_b.kernel.get_value())
        self.assertGreater(float(np.max(np.abs(g_l1_total - g_l1_self))), 1e-6)

  def test_moe_bias_var_no_weight_decay_and_clean_update(self):
    """Verifies gate.bias is MoEBiasVar, skips AdamW weight decay at rate=0, and updates cleanly at rate>0."""
    # Non-collision check: glm5.3-flash uses LLAMA2 decoder_block, not DeepSeek/GLM5 routed bias.
    gate_glm5 = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=6,
        model_name="glm5.2-744b",
        decoder_block=common_types.DecoderBlockType.GLM5,
        mesh=None,
        rngs=nnx.Rngs(0),
        use_bias=True,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        bias_dtype=jnp.float32,
    )
    self.assertIsInstance(gate_glm5.bias, moe.MoEBiasVar)
    self.assertNotIsInstance(gate_glm5.bias, nnx.Param)

    gate_flash = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=6,
        model_name="glm5.3-flash",
        decoder_block=common_types.DecoderBlockType.LLAMA2,
        mesh=None,
        rngs=nnx.Rngs(0),
        use_bias=True,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
    )
    self.assertIsInstance(gate_flash.bias, nnx.Param)

    with tempfile.TemporaryDirectory() as tmpdir:
      # 1. routed_bias_update_rate == 0.0: AdamW (weight_decay=0.1) must NOT decay gate.bias!
      cfg_zero = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=True,
              seq_len=8,
              indexer_topk=4,
              routed_bias_update_rate=0.0,
          )
      )
      mesh = _create_mesh(cfg_zero)
      batch_size = cfg_zero.global_batch_size_to_train_on
      seq_len = 8
      rng_np = np.random.default_rng(5150)
      inputs_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)
      targets_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_zero.logical_axis_rules):
        mt_zero = models.Transformer(
            cfg_zero,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_zero),
            rngs=nnx.Rngs(params=51, dropout=52),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        gate_var = mt_zero.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate
        self.assertIsInstance(gate_var.bias, moe.MoEBiasVar)
        # L_moe=4 != num_experts=6 guarantees axis-0 scan stacking [4, 6] is verified unambiguously.
        self.assertEqual(gate_var.bias.get_value().shape, (4, 6))
        init_bias = np.linspace(0.5, 1.5, 24, dtype=np.float32).reshape(4, 6)
        gate_var.bias.set_value(jnp.asarray(init_bias))
        init_gate_kernel = np.asarray(gate_var.kernel.get_value()).copy()

        opt_zero = nnx.Optimizer(mt_zero, optax.adamw(learning_rate=1e-2, weight_decay=0.1), wrt=nnx.Param)
        ts_zero = train_state_nnx.TrainStateNNX(mt_zero, opt_zero)
        ts_zero_gdef, ts_zero_state = nnx.split(ts_zero)

        pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        batch = {
            "inputs": jnp.asarray(inputs_np),
            "inputs_position": pos,
            "inputs_segmentation": seg,
            "targets": jnp.asarray(targets_np),
            "targets_position": pos,
            "targets_segmentation": seg,
        }
        new_zero_state, _ = pre_train.train_step(ts_zero_gdef, cfg_zero, None, None, ts_zero_state, batch)
        updated_zero = nnx.merge(ts_zero_gdef, new_zero_state)
        after_zero_bias = np.asarray(
            updated_zero.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.get_value()
        )
        after_zero_kernel = np.asarray(
            updated_zero.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel.get_value()
        )

      np.testing.assert_array_equal(after_zero_bias, init_bias)
      self.assertGreater(float(np.max(np.abs(after_zero_kernel - init_gate_kernel))), 0.0)

      # 2. routed_bias_update_rate > 0.0: train_step updates [L_moe=4, E=6] cleanly without shape mismatch!
      rate = 0.02
      cfg_pos = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=True,
              seq_len=8,
              indexer_topk=4,
              routed_bias_update_rate=rate,
          )
      )
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_pos.logical_axis_rules):
        mt_pos = models.Transformer(
            cfg_pos,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_pos),
            rngs=nnx.Rngs(params=51, dropout=52),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        mt_pos.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.set_value(jnp.asarray(init_bias))
        opt_pos = nnx.Optimizer(mt_pos, optax.adamw(learning_rate=1e-2, weight_decay=0.1), wrt=nnx.Param)
        ts_pos = train_state_nnx.TrainStateNNX(mt_pos, opt_pos)
        ts_pos_gdef, ts_pos_state = nnx.split(ts_pos)
        new_pos_state, _ = pre_train.train_step(ts_pos_gdef, cfg_pos, None, None, ts_pos_state, batch)
        updated_pos = nnx.merge(ts_pos_gdef, new_pos_state)
        after_pos_bias = np.asarray(
            updated_pos.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.get_value()
        )

      self.assertEqual(after_pos_bias.shape, (4, 6))
      self.assertEqual(after_pos_bias.dtype, np.float32)
      self.assertTrue(np.all(np.isfinite(after_pos_bias)))
      delta = np.abs(after_pos_bias - init_bias)
      self.assertGreater(float(np.max(delta)), 0.0)
      # Every entry changes by either 0.0 (exact average load) or +/- rate (no AdamW decay contamination).
      nonzero_deltas = delta[delta > 1e-7]
      np.testing.assert_allclose(nonzero_deltas, np.full_like(nonzero_deltas, rate), rtol=1e-5, atol=1e-6)

  def test_deepseek32_use_index_share_false_invariance(self):
    """Verifies DeepSeek-3.2 (use_index_share=False) Jaxpr/scope/checkpoint invariance, bf16 mask, and mask error."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "model_name=deepseek3.2-671b",
              "override_model_config=True",
              "skip_jax_distributed_system=True",
              f"base_output_directory={tmpdir}",
              "run_name=ds32_invariance",
              "enable_checkpointing=False",
              "dtype=bfloat16",
              "weight_dtype=bfloat16",
              "attention=dot_product",
              "base_emb_dim=32",
              "base_mlp_dim=64",
              "base_moe_mlp_dim=16",
              "base_num_decoder_layers=2",
              "first_num_dense_layers=1",
              "base_num_query_heads=4",
              "base_num_kv_heads=4",
              "q_lora_rank=16",
              "kv_lora_rank=8",
              "qk_nope_head_dim=8",
              "qk_rope_head_dim=4",
              "v_head_dim=8",
              "use_indexer=True",
              "use_index_share=False",
              "indexer_n_heads=4",
              "indexer_head_dim=8",
              "indexer_topk=4",
              "vocab_size=64",
              "max_position_embeddings=128",
              "original_max_position_embeddings=64",
              "max_target_length=8",
              "per_device_batch_size=1",
          ]
      )
      self.assertFalse(cfg.use_index_share)
      self.assertFalse(cfg.prune_shared_indexers)

      mesh = _create_mesh(cfg)
      batch_size = cfg.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        mla = attention_mla.MLA(
            config=cfg,
            num_query_heads=cfg.num_query_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_target_length=8,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=(batch_size, 8, cfg.emb_dim),
            inputs_kv_shape=(batch_size, 8, cfg.emb_dim),
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            original_max_position_embeddings=cfg.original_max_position_embeddings,
            mscale=cfg.mscale,
            rope_factor=cfg.rope_factor,
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=0),
        )
        self.assertIsNone(mla._index_share_schedule)  # pylint: disable=protected-access

        # 1. Verify Jaxpr has zero glm_ scopes and zero full_layer_ / shared_layer_ checkpoint names in both regimes.
        gdef, state = nnx.split(mla)
        for seq_len in (3, 8):
          x = jnp.ones((batch_size, seq_len, cfg.emb_dim), dtype=jnp.bfloat16)
          pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
          seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

          def _call_mla(st, inp, p, s):
            mod = nnx.merge(gdef, st)
            out, _, idx_state = mod(
                inp,
                inp,
                p,
                s,
                deterministic=True,
                model_mode=common_types.MODEL_MODE_TRAIN,
                return_indexer_state=True,
            )
            self.assertIsNone(idx_state)
            return out

          jaxpr_str = str(jax.make_jaxpr(_call_mla)(state, x, pos, seg))
          self.assertNotIn("glm_", jaxpr_str)
          self.assertNotIn("full_layer_", jaxpr_str)
          self.assertNotIn("shared_layer_", jaxpr_str)

        # 2. Verify seq_len <= indexer_topk preserves bfloat16 attention_mask dtype without float32 upcast.
        short_len = 3
        x_short = jnp.ones((batch_size, short_len, cfg.emb_dim), dtype=jnp.bfloat16)
        q_low = jnp.ones((batch_size, short_len, cfg.q_lora_rank), dtype=jnp.bfloat16)
        pos_short = jnp.broadcast_to(jnp.arange(short_len, dtype=jnp.int32), (batch_size, short_len))
        bf16_mask = jnp.zeros((batch_size, short_len, short_len), dtype=jnp.bfloat16)
        idx_mask, _, _ = mla.indexer(
            inputs_q=x_short,
            low_rank_q=q_low,
            inputs_kv=x_short,
            inputs_positions=pos_short,
            attention_mask=bf16_mask,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        self.assertEqual(idx_mask.dtype, jnp.bfloat16)

        # 3. Verify unexpected attention_mask shape raises ValueError immediately.
        bad_mask = jnp.zeros((batch_size, short_len, short_len + 5), dtype=jnp.bfloat16)
        with self.assertRaisesRegex(ValueError, "Expected attention_mask last dim"):
          mla.indexer(
              inputs_q=x_short,
              low_rank_q=q_low,
              inputs_kv=x_short,
              inputs_positions=pos_short,
              attention_mask=bad_mask,
              model_mode=common_types.MODEL_MODE_TRAIN,
          )

  def test_scanned_shared_layer_indexer_cache_nan_poison_invariance(self):
    """Verifies poisoning scanned Shared-layer IndexerKVCache_0 slots with NaN after prefill leaves AR decode unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_scan = pyconfig.initialize(
          _common_glm_cli_args(
              tmpdir,
              model_name="glm5.2-744b",
              scan_layers=True,
              prune_shared_indexers=False,
              seq_len=10,
              indexer_topk=4,
              indexer_n_heads=8,
              max_prefill_predict_length=4,
              mla_naive_kvcache=False,
          )
      )
      mesh = _create_mesh(cfg_scan)
      batch_size = cfg_scan.global_batch_size_to_train_on
      prompt_np = np.arange(batch_size * 4, dtype=np.int32).reshape(batch_size, 4) % 64
      pattern = index_share_utils.parse_index_share_pattern(
          cfg_scan.index_share_pattern, cfg_scan.base_num_decoder_layers
      )
      first_dense = cfg_scan.first_num_dense_layers

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scan.logical_axis_rules):
        m_clean = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=3, dropout=4),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        m_poisoned = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=13, dropout=14),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        nnx.update(m_poisoned, nnx.state(m_clean, nnx.Param))
        m_poisoned.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.set_value(
            m_clean.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias.get_value()
        )

        prompt = jnp.asarray(prompt_np)
        p_pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_size, 4))
        p_seg = jnp.ones((batch_size, 4), dtype=jnp.int32)
        prefill_clean = np.asarray(
            m_clean(
                prompt,
                p_pos,
                decoder_segment_ids=p_seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_PREFILL,
            )
        )
        prefill_poisoned = np.asarray(
            m_poisoned(
                prompt,
                p_pos,
                decoder_segment_ids=p_seg,
                enable_dropout=False,
                model_mode=common_types.MODEL_MODE_PREFILL,
            )
        )
        self.assertTrue(np.all(np.isfinite(prefill_poisoned)))
        np.testing.assert_allclose(prefill_poisoned, prefill_clean, rtol=0.0, atol=0.0)

        # Overwrite every Shared-layer slot in IndexerKVCache_0 (both prefill and AR buffers) with NaN.
        poisoned_slots = []
        for global_idx in range(cfg_scan.base_num_decoder_layers):
          if not index_share_utils.is_shared_layer(global_idx, pattern):
            continue
          if global_idx < first_dense:
            idx_cache = m_poisoned.decoder.dense_layers.self_attention.IndexerKVCache_0
            local_idx = global_idx
          else:
            idx_cache = m_poisoned.decoder.moe_layers.self_attention.IndexerKVCache_0
            local_idx = global_idx - first_dense
          poisoned_slots.append((idx_cache, local_idx))
          for attr in ("cached_prefill_key", "cached_prefill_value", "cached_ar_key", "cached_ar_value"):
            cache_var = getattr(idx_cache, attr)
            cur_val = cache_var.get_value()
            cache_var.set_value(cur_val.at[local_idx].set(jnp.nan))
            self.assertTrue(bool(jnp.all(jnp.isnan(cache_var.get_value()[local_idx]))))
        self.assertLen(poisoned_slots, 3)

        next_tok_clean = jnp.argmax(jnp.asarray(prefill_clean)[:, -1:, :], axis=-1).astype(jnp.int32)
        next_tok_poisoned = jnp.argmax(jnp.asarray(prefill_poisoned)[:, -1:, :], axis=-1).astype(jnp.int32)
        ar_logits_clean, ar_logits_poisoned = [], []
        ar_tokens_clean, ar_tokens_poisoned = [], []

        # Steps 4, 5, 6 cross into kv_len=5, 6, 7 (> indexer_topk=4), where Full layers read IndexerKVCache_0.
        for step_pos in range(4, 7):
          s_pos = jnp.full((batch_size, 1), step_pos, dtype=jnp.int32)
          step_clean = m_clean(
              next_tok_clean,
              s_pos,
              decoder_segment_ids=None,
              enable_dropout=False,
              model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          )
          step_poisoned = m_poisoned(
              next_tok_poisoned,
              s_pos,
              decoder_segment_ids=None,
              enable_dropout=False,
              model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          )
          ar_logits_clean.append(np.asarray(step_clean[:, 0, :]))
          ar_logits_poisoned.append(np.asarray(step_poisoned[:, 0, :]))
          next_tok_clean = jnp.argmax(step_clean[:, -1:, :], axis=-1).astype(jnp.int32)
          next_tok_poisoned = jnp.argmax(step_poisoned[:, -1:, :], axis=-1).astype(jnp.int32)
          ar_tokens_clean.append(np.asarray(next_tok_clean))
          ar_tokens_poisoned.append(np.asarray(next_tok_poisoned))

        for idx_cache, local_idx in poisoned_slots:
          self.assertTrue(bool(jnp.all(jnp.isnan(idx_cache.cached_prefill_key.get_value()[local_idx]))))
          self.assertTrue(bool(jnp.all(jnp.isnan(idx_cache.cached_ar_key.get_value()[local_idx]))))

      stacked_clean = np.stack(ar_logits_clean, axis=1)
      stacked_poisoned = np.stack(ar_logits_poisoned, axis=1)
      self.assertTrue(np.all(np.isfinite(stacked_poisoned)))
      np.testing.assert_allclose(stacked_poisoned, stacked_clean, rtol=0.0, atol=0.0)
      np.testing.assert_array_equal(np.concatenate(ar_tokens_poisoned, axis=1), np.concatenate(ar_tokens_clean, axis=1))

  def test_shared_layer_indexer_loss_and_donor_gradient_routing(self):
    """Verifies S-layer KL loss positivity, donor-only grad routing (scanned & unscanned), and CPU remat compile."""
    with tempfile.TemporaryDirectory() as tmpdir:
      seq_len = 10
      common_kw = {
          "model_name": "glm5.2-744b",
          "seq_len": seq_len,
          "indexer_topk": 4,
          "indexer_loss_scaling_factor": 0.6,
          "indexer_sparse_training": True,
      }
      cfg_scan = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=True, prune_shared_indexers=False, **common_kw)
      )
      remat_args = [
          a
          for a in _common_glm_cli_args(tmpdir, scan_layers=True, prune_shared_indexers=False, **common_kw)
          if not a.startswith("remat_policy=")
      ] + ["remat_policy=full"]
      cfg_scan_remat = pyconfig.initialize(remat_args)
      cfg_unscan_pruned = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=False, prune_shared_indexers=True, **common_kw)
      )
      cfg_unscan_retained = pyconfig.initialize(
          _common_glm_cli_args(tmpdir, scan_layers=False, prune_shared_indexers=False, **common_kw)
      )
      mesh = _create_mesh(cfg_scan)
      batch_size = cfg_scan.global_batch_size_to_train_on
      ax = cfg_scan.param_scan_axis
      rng_np = np.random.default_rng(2027)
      inputs_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)
      targets_np = rng_np.integers(0, 64, size=(batch_size, seq_len), dtype=np.int32)

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scan.logical_axis_rules):
        mt_scan = models.Transformer(
            cfg_scan,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan),
            rngs=nnx.Rngs(params=61, dropout=62),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        mt_scan_remat = models.Transformer(
            cfg_scan_remat,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_scan_remat),
            rngs=nnx.Rngs(params=63, dropout=64),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        mt_unscan_pruned = models.Transformer(
            cfg_unscan_pruned,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan_pruned),
            rngs=nnx.Rngs(params=65, dropout=66),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        mt_unscan_retained = models.Transformer(
            cfg_unscan_retained,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg_unscan_retained),
            rngs=nnx.Rngs(params=67, dropout=68),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )

        nnx.update(mt_scan_remat, nnx.state(mt_scan, nnx.Param))
        _copy_scanned_to_unscanned_jax(mt_scan, mt_unscan_pruned, cfg_scan)
        _copy_scanned_to_unscanned_jax(mt_scan, mt_unscan_retained, cfg_scan)

        toks = jnp.asarray(inputs_np)
        pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        batch = {
            "inputs": toks,
            "inputs_position": pos,
            "inputs_segmentation": seg,
            "targets": jnp.asarray(targets_np),
            "targets_position": pos,
            "targets_segmentation": seg,
        }

        def _unscan_layer(m, idx: int):
          return getattr(m.decoder, f"dense_layers_{idx}" if idx < 2 else f"moe_layers_{idx - 2}")

        # 1. Verify scanned `indexer_losses` stacks per-layer scalars along axis 0 and every S layer has loss > 0.
        _ = mt_scan(toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)
        _ = mt_unscan_pruned(
            toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN
        )
        dense_losses_scan = np.asarray(mt_scan.decoder.dense_layers.self_attention.indexer_loss.get_value())
        moe_losses_scan = np.asarray(mt_scan.decoder.moe_layers.self_attention.indexer_loss.get_value())
        self.assertEqual(dense_losses_scan.shape, (2,))
        self.assertEqual(moe_losses_scan.shape, (4,))
        all_losses_scan = np.concatenate([dense_losses_scan, moe_losses_scan], axis=0)
        all_losses_unscan = np.asarray(
            [float(_unscan_layer(mt_unscan_pruned, i).self_attention.indexer_loss.get_value()) for i in range(6)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(all_losses_scan, all_losses_unscan, rtol=1e-5, atol=1e-6)
        for s_idx in (2, 3, 5):
          self.assertTrue(np.isfinite(float(all_losses_scan[s_idx])))
          self.assertGreater(float(all_losses_scan[s_idx]), 0.0)
          self.assertGreater(float(all_losses_unscan[s_idx]), 0.0)

        # 2. Differentiate ONE Shared layer's sown loss alone on both scanned and unscanned paths.
        def _single_layer_loss_and_grad_scan(model_obj, target_idx: int):
          nnx.pop(model_obj, nnx.Intermediate)
          nnx.pop(model_obj, attention_mla.indexer_losses)
          gdef, params, rest = nnx.split(model_obj, nnx.Param, ...)

          def _fn(p):
            m = nnx.merge(gdef, p, rest, copy=True)
            _ = m(toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)
            if target_idx < 2:
              return m.decoder.dense_layers.self_attention.indexer_loss.get_value()[target_idx]
            return m.decoder.moe_layers.self_attention.indexer_loss.get_value()[target_idx - 2]

          return jax.value_and_grad(_fn)(params)

        def _single_layer_loss_and_grad_unscan(model_obj, target_idx: int):
          nnx.pop(model_obj, nnx.Intermediate)
          nnx.pop(model_obj, attention_mla.indexer_losses)
          gdef, params, rest = nnx.split(model_obj, nnx.Param, ...)

          def _fn(p):
            m = nnx.merge(gdef, p, rest, copy=True)
            _ = m(toks, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)
            return _unscan_layer(m, target_idx).self_attention.indexer_loss.get_value()

          return jax.value_and_grad(_fn)(params)

        def _scan_indexer_grads(grads_tree, layer_idx: int) -> dict[str, np.ndarray]:
          grp = "dense_layers" if layer_idx < 2 else "moe_layers"
          local_i = layer_idx if layer_idx < 2 else layer_idx - 2
          idx_g = grads_tree["decoder"][grp]["self_attention"]["indexer"]
          return {
              "wq_b": np.take(np.asarray(idx_g["wq_b"]["kernel"].get_value()), local_i, axis=ax),
              "wk": np.take(np.asarray(idx_g["wk"]["kernel"].get_value()), local_i, axis=ax),
              "k_scale": np.take(np.asarray(idx_g["k_norm"]["scale"].get_value()), local_i, axis=ax),
              "k_bias": np.take(np.asarray(idx_g["k_norm"]["bias"].get_value()), local_i, axis=ax),
              "w_proj": np.take(np.asarray(idx_g["weights_proj"]["kernel"].get_value()), local_i, axis=ax),
          }

        def _unscan_indexer_grads(grads_tree, layer_idx: int) -> dict[str, np.ndarray] | None:
          lyr_key = f"dense_layers_{layer_idx}" if layer_idx < 2 else f"moe_layers_{layer_idx - 2}"
          sa_g = grads_tree["decoder"][lyr_key]["self_attention"]
          if "indexer" not in sa_g or not sa_g["indexer"]:
            return None
          idx_g = sa_g["indexer"]
          return {
              "wq_b": np.asarray(idx_g["wq_b"]["kernel"].get_value()),
              "wk": np.asarray(idx_g["wk"]["kernel"].get_value()),
              "k_scale": np.asarray(idx_g["k_norm"]["scale"].get_value()),
              "k_bias": np.asarray(idx_g["k_norm"]["bias"].get_value()),
              "w_proj": np.asarray(idx_g["weights_proj"]["kernel"].get_value()),
          }

        # Test both cross-scan-group (S layer 3 -> Dense donor 1) and intra-scan-group (S layer 5 -> MoE donor 4).
        for s_idx, donor_idx in ((3, 1), (5, 4)):
          s_loss_sc, grads_sc = _single_layer_loss_and_grad_scan(mt_scan, s_idx)
          s_loss_up, grads_up = _single_layer_loss_and_grad_unscan(mt_unscan_pruned, s_idx)
          s_loss_ur, grads_ur = _single_layer_loss_and_grad_unscan(mt_unscan_retained, s_idx)

          self.assertGreater(float(s_loss_sc), 0.0)
          self.assertGreater(float(s_loss_up), 0.0)
          self.assertGreater(float(s_loss_ur), 0.0)
          np.testing.assert_allclose(float(s_loss_sc), float(s_loss_up), rtol=1e-5, atol=1e-6)
          np.testing.assert_allclose(float(s_loss_sc), float(s_loss_ur), rtol=1e-5, atol=1e-6)

          donor_sc = _scan_indexer_grads(grads_sc, donor_idx)
          donor_up = _unscan_indexer_grads(grads_up, donor_idx)
          donor_ur = _unscan_indexer_grads(grads_ur, donor_idx)
          self.assertIsNotNone(donor_up)
          self.assertIsNotNone(donor_ur)
          assert donor_up is not None and donor_ur is not None
          for param_name in ("wq_b", "wk"):
            self.assertTrue(np.all(np.isfinite(donor_sc[param_name])))
            self.assertGreater(float(np.linalg.norm(donor_sc[param_name])), 0.0)
            self.assertGreater(float(np.linalg.norm(donor_up[param_name])), 0.0)
            self.assertGreater(float(np.linalg.norm(donor_ur[param_name])), 0.0)
            np.testing.assert_allclose(donor_sc[param_name], donor_up[param_name], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(donor_sc[param_name], donor_ur[param_name], rtol=1e-4, atol=1e-5)

          for other_idx in range(6):
            if other_idx == donor_idx:
              continue
            other_sc = _scan_indexer_grads(grads_sc, other_idx)
            for arr in other_sc.values():
              np.testing.assert_array_equal(arr, np.zeros_like(arr))
            other_ur = _unscan_indexer_grads(grads_ur, other_idx)
            self.assertIsNotNone(other_ur)
            assert other_ur is not None
            for arr in other_ur.values():
              np.testing.assert_array_equal(arr, np.zeros_like(arr))
            other_up = _unscan_indexer_grads(grads_up, other_idx)
            if other_up is not None:
              for arr in other_up.values():
                np.testing.assert_array_equal(arr, np.zeros_like(arr))

        # 3. Verify scanned + remat (`remat_policy=full`) AOT compilation and backward parity on CPU with `nnx.cond`.
        def _compile_and_run_scan_loss(model_obj, cfg_obj):
          nnx.pop(model_obj, nnx.Intermediate)
          nnx.pop(model_obj, attention_mla.indexer_losses)
          gdef, params, rest = nnx.split(model_obj, nnx.Param, ...)

          def _step(p):
            m = nnx.merge(gdef, p, rest, copy=True)
            loss_val, aux = pre_train.loss_fn(m, cfg_obj, batch, None, None, is_train=True)
            return loss_val, aux["indexer_loss"]

          compiled = jax.jit(jax.value_and_grad(_step, has_aux=True)).lower(params).compile()
          return compiled(params)

        (loss_none, idx_none), grads_none = _compile_and_run_scan_loss(mt_scan, cfg_scan)
        (loss_remat, idx_remat), grads_remat = _compile_and_run_scan_loss(mt_scan_remat, cfg_scan_remat)
        self.assertTrue(np.isfinite(float(loss_remat)))
        self.assertGreater(float(idx_remat), 0.0)
        np.testing.assert_allclose(float(loss_remat), float(loss_none), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(idx_remat), float(idx_none), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(grads_remat["decoder"]["dense_layers"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()),
            np.asarray(grads_none["decoder"]["dense_layers"]["self_attention"]["indexer"]["wq_b"]["kernel"].get_value()),
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  unittest.main()
