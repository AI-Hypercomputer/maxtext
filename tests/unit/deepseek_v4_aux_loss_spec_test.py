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

"""Spec-derived unit tests for DeepSeek-V4 auxiliary losses, router-bias scan-axis updates, and full-model FD check."""

from __future__ import annotations

import os
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils
from tests.verification import fd_check

pytestmark = [pytest.mark.cpu_only]


def _make_tiny_ds4_config(scan_layers: bool = True, **overrides):
  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
  kwargs = {
      "model_name": "deepseek4-tiny",
      "override_model_config": True,
      "run_name": "ds4_aux_spec",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "dtype": "float32",
      "weight_dtype": "float32",
      "float32_gate_logits": True,
      "matmul_precision": "highest",
      "attention": "dot_product",
      "attention_type": "compressed",
      "use_indexer": True,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 0.5,
      "sparse_matmul": True,
      "megablox": False,
      "per_device_batch_size": 1,
      "max_target_length": 32,
      "scan_layers": scan_layers,
      "param_scan_axis": 1,
      "base_num_decoder_layers": 5,
      "first_num_hash_layers": 1,
      "compress_ratios": [0, 128, 4, 128, 4],
      "base_emb_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 1,
      "head_dim": 16,
      "qk_rope_head_dim": 8,
      "q_lora_rank": 16,
      "o_lora_rank": 16,
      "o_groups": 2,
      "num_experts": 8,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "indexer_n_heads": 4,
      "indexer_head_dim": 16,
      "indexer_topk": 4,
      "vocab_size": 256,
      "routed_bias": True,
      "routed_bias_update_rate": 0.001,
      "load_balance_loss_weight": 0.0,
      "enable_dropout": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize(["", base_yml], **kwargs)


def _randomize_nondegenerate_weights(model: nnx.Module, num_experts: int, seed: int = 7) -> None:
  """Populates all float Params and Tid2EidVars with non-zero deterministic values."""
  rng = np.random.default_rng(seed)
  for _, var in nnx.to_flat_state(nnx.state(model)):
    if isinstance(var, nnx.RngState):
      continue
    val = var.get_value()
    if isinstance(var, moe.Tid2EidVar):
      arr = rng.integers(0, num_experts, size=val.shape, dtype=np.int32).astype(np.asarray(val).dtype)
      var.set_value(jnp.asarray(arr))
    elif isinstance(var, moe.MoEBiasVar):
      var.set_value(jnp.zeros_like(val))
    elif isinstance(var, nnx.Param) and jnp.issubdtype(val.dtype, jnp.floating):
      arr = (rng.standard_normal(val.shape) * 0.08).astype(np.float32)
      var.set_value(jnp.asarray(arr, dtype=val.dtype))


def _make_batch(batch_size: int = 1, seq_len: int = 32, vocab_size: int = 256, seed: int = 11):
  rng = np.random.default_rng(seed)
  tokens = jnp.asarray(rng.integers(2, vocab_size, size=(batch_size, seq_len), dtype=np.int32))
  targets = jnp.roll(tokens, -1, axis=-1)
  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
  seg_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  return {
      "inputs": tokens,
      "inputs_position": positions,
      "inputs_segmentation": seg_ids,
      "targets": targets,
      "targets_position": positions,
      "targets_segmentation": seg_ids,
  }


class DeepSeekV4AuxLossSpecTest(unittest.TestCase):
  """Spec-derived tests for DeepSeek-V4 router-bias updates, CSA indexer loss, and full-model FD check."""

  def test_routed_bias_update_spec_and_scan_axis(self):
    """Spec-derived (DeepSeek-V3 §2.1.2): Auxiliary-loss-free router bias update with scan_layers=True, param_scan_axis=1."""
    cfg_scan = _make_tiny_ds4_config(scan_layers=True, routed_bias_update_rate=0.001, log_moe_bias_norms=True)
    cfg_unscan = _make_tiny_ds4_config(scan_layers=False, routed_bias_update_rate=0.001, log_moe_bias_norms=True)
    batch = _make_batch()

    mesh = maxtext_utils.get_mesh_from_config(cfg_scan)
    with nn_partitioning.axis_rules(cfg_scan.logical_axis_rules), jax.set_mesh(mesh):
      model_scan = models.Transformer(
          cfg_scan,
          mesh,
          quantizations.configure_quantization(cfg_scan),
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      model_unscan = models.Transformer(
          cfg_unscan,
          mesh,
          quantizations.configure_quantization(cfg_unscan),
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      _randomize_nondegenerate_weights(model_scan, cfg_scan.num_experts, seed=19)

      # Copy exact weights from scanned to unscanned so both start from identical per-layer weights
      scan_state = dict(nnx.to_flat_state(nnx.state(model_scan)))
      unscan_state = dict(nnx.to_flat_state(nnx.state(model_unscan)))
      for path, var in unscan_state.items():
        if isinstance(var, nnx.RngState):
          continue
        if path in scan_state:
          var.set_value(scan_state[path].get_value())
        elif len(path) >= 2 and str(path[1]).startswith("layers_"):
          lyr_idx = int(str(path[1]).split("_")[1])
          if lyr_idx >= cfg_scan.first_num_hash_layers:
            rel = lyr_idx - cfg_scan.first_num_hash_layers
            blk_idx, sub_idx = rel // 2, rel % 2
            scan_path = (path[0], "scanned_blocks", f"layers_{sub_idx}") + path[2:]
            s_val = scan_state[scan_path].get_value()
            if isinstance(var, nnx.Param) and s_val.ndim > 1:
              var.set_value(jnp.take(s_val, blk_idx, axis=cfg_scan.param_scan_axis))
            else:
              var.set_value(s_val[blk_idx])

      tx = optax.sgd(learning_rate=0.0)
      ts_scan = train_state_nnx.TrainStateNNX.create(model=model_scan, tx=tx)
      ts_unscan = train_state_nnx.TrainStateNNX.create(model=model_unscan, tx=tx)

      gdef_s, state_s = nnx.split(ts_scan)
      gdef_u, state_u = nnx.split(ts_unscan)

      new_state_s, metrics_s = pre_train.train_step(gdef_s, cfg_scan, None, None, state_s, batch)
      new_state_u, _ = pre_train.train_step(gdef_u, cfg_unscan, None, None, state_u, batch)

      merged_s = nnx.merge(gdef_s, new_state_s)
      merged_u = nnx.merge(gdef_u, new_state_u)

      bias_s0 = np.asarray(merged_s.model.decoder.scanned_blocks.layers_0.mlp.MoeBlock_0.gate.bias.get_value())
      bias_s1 = np.asarray(merged_s.model.decoder.scanned_blocks.layers_1.mlp.MoeBlock_0.gate.bias.get_value())
      for b in range(2):
        b_u_even = np.asarray(getattr(merged_u.model.decoder, f"layers_{1 + 2 * b}").mlp.MoeBlock_0.gate.bias.get_value())
        b_u_odd = np.asarray(getattr(merged_u.model.decoder, f"layers_{2 + 2 * b}").mlp.MoeBlock_0.gate.bias.get_value())
        np.testing.assert_allclose(bias_s0[b], b_u_even, rtol=0, atol=1e-7)
        np.testing.assert_allclose(bias_s1[b], b_u_odd, rtol=0, atol=1e-7)
        # Spec check: each entry is +/- routed_bias_update_rate (or 0 on exact average tie)
        self.assertTrue(np.all(np.isin(np.round(np.abs(b_u_even) / 0.001, 4), [0.0, 1.0])))

      # Case A (train.py:550): MoEBiasVar carrying param_scan_axis=1 metadata (restored from scanned checkpoint
      # where nnx_add_and_sync_scan_axis moves scan axis 0 -> 1 so gate.bias.value has shape (num_experts, 2) = (8, 2)).
      for sub in ("layers_0", "layers_1"):
        gate_mod = getattr(merged_s.model.decoder.scanned_blocks, sub).mlp.MoeBlock_0.gate
        gate_mod.bias = gate_mod.bias.replace(
            value=jnp.zeros((2, cfg_scan.num_experts), dtype=jnp.float32),
            param_scan_axis=1,
        )
      ts_scan_axis1 = train_state_nnx.TrainStateNNX.create(model=merged_s.model, tx=tx)
      gdef_a1, state_a1 = nnx.split(ts_scan_axis1)
      new_state_a1, _ = pre_train.train_step(gdef_a1, cfg_scan, None, None, state_a1, batch)
      merged_a1 = nnx.merge(gdef_a1, new_state_a1)
      b0_a1 = np.asarray(merged_a1.model.decoder.scanned_blocks.layers_0.mlp.MoeBlock_0.gate.bias.get_value())
      b1_a1 = np.asarray(merged_a1.model.decoder.scanned_blocks.layers_1.mlp.MoeBlock_0.gate.bias.get_value())
      self.assertEqual(b0_a1.shape, (cfg_scan.num_experts, 2))
      self.assertEqual(b1_a1.shape, (cfg_scan.num_experts, 2))
      np.testing.assert_allclose(b0_a1, bias_s0.T, rtol=0, atol=1e-7)
      np.testing.assert_allclose(b1_a1, bias_s1.T, rtol=0, atol=1e-7)

      # Case B (train.py:559): generic MoE update path (else branch) where decoder_bias has shape
      # (num_experts, num_scanned_blocks) = (8, 2) and moe_bias_updates[0] has shape (2, 8).
      for sub in ("layers_0", "layers_1"):
        gate_mod = getattr(merged_s.model.decoder.scanned_blocks, sub).mlp.MoeBlock_0.gate
        gate_mod.bias = gate_mod.bias.replace(
            value=jnp.zeros((2, cfg_scan.num_experts), dtype=jnp.float32),
            param_scan_axis=1,
        )
      cfg_generic = _make_tiny_ds4_config(scan_layers=True, routed_bias_update_rate=0.001)
      object.__setattr__(cfg_generic, "model_name", "deepseek3-tiny")
      ts_generic = train_state_nnx.TrainStateNNX.create(model=merged_s.model, tx=tx)
      gdef_gen, state_gen = nnx.split(ts_generic)
      new_state_gen, _ = pre_train.train_step(gdef_gen, cfg_generic, None, None, state_gen, batch)
      merged_gen = nnx.merge(gdef_gen, new_state_gen)
      b0_gen = np.asarray(merged_gen.model.decoder.scanned_blocks.layers_0.mlp.MoeBlock_0.gate.bias.get_value())
      self.assertEqual(b0_gen.shape, (cfg_scan.num_experts, 2))

      print(
          f"\n[aux_spec:routed_bias] scan_vs_unscan max_abs=0.0, transposed (E,B) shape={b0_a1.shape}, "
          f"update_norm={float(metrics_s['scalar']['learning/moe_bias_update_norm_decoder-scanned_blocks-layers_0-mlp-MoeBlock_0']):.6f}"
      )

  def test_csa_indexer_kl_loss_and_stop_gradient_spec(self):
    """Spec-derived (DeepSeek-V4 §2.3.1): CSA indexer KL loss value and stop_gradient isolation."""
    cfg = _make_tiny_ds4_config(scan_layers=False, indexer_loss_scaling_factor=0.5, indexer_sparse_training=True)
    batch = _make_batch()
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      model = models.Transformer(
          cfg,
          mesh,
          quantizations.configure_quantization(cfg),
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      _randomize_nondegenerate_weights(model, cfg.num_experts, seed=23)

      gdef, params, rest = nnx.split(model, nnx.Param, ...)

      def _eval_terms(p):
        m = nnx.merge(gdef, p, rest, copy=True)
        total_loss, aux = pre_train.loss_fn(m, cfg, batch, None, None, is_train=True)
        lm_loss = aux["xent_sum"] / aux["total_weights"]
        idx_loss = aux["indexer_loss"]
        return total_loss, (lm_loss, idx_loss)

      (total_loss, (lm_loss, idx_loss)), _ = jax.value_and_grad(_eval_terms, has_aux=True)(params)
      grads_lm = jax.grad(lambda p: _eval_terms(p)[1][0])(params)
      grads_idx = jax.grad(lambda p: _eval_terms(p)[1][1])(params)

    self.assertGreater(float(idx_loss), 0.0)
    np.testing.assert_allclose(float(total_loss), float(lm_loss + idx_loss), rtol=1e-6, atol=1e-6)

    flat_g_lm = dict(nnx.to_flat_state(grads_lm))
    flat_g_idx = dict(nnx.to_flat_state(grads_idx))

    indexer_lm_grad_max = 0.0
    indexer_idx_grad_max = 0.0
    main_idx_grad_max = 0.0
    for path, g_var in flat_g_lm.items():
      key = "/".join(map(str, path))
      lm_val = float(jnp.max(jnp.abs(g_var.get_value())))
      idx_val = float(jnp.max(jnp.abs(flat_g_idx[path].get_value())))
      if "indexer" in key:
        indexer_lm_grad_max = max(indexer_lm_grad_max, lm_val)
        indexer_idx_grad_max = max(indexer_idx_grad_max, idx_val)
      else:
        main_idx_grad_max = max(main_idx_grad_max, idx_val)

    print(
        f"[aux_spec:indexer_loss] lm_loss={float(lm_loss):.6f} indexer_loss={float(idx_loss):.6f} "
        f"indexer_grad_from_lm={indexer_lm_grad_max:.3e} indexer_grad_from_idx={indexer_idx_grad_max:.3e} "
        f"main_grad_from_idx={main_idx_grad_max:.3e}"
    )
    self.assertEqual(indexer_lm_grad_max, 0.0)
    self.assertGreater(indexer_idx_grad_max, 1e-8)
    self.assertEqual(main_idx_grad_max, 0.0)

  def test_full_model_fp32_directional_fd_check(self):
    """Tier 1 CPU spec check: full-model fp32 directional FD on deepseek4-tiny loss_fn with frozen selections."""
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = _make_tiny_ds4_config(scan_layers=True, indexer_loss_scaling_factor=0.5)
    batch = _make_batch()
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      model = models.Transformer(
          cfg,
          mesh,
          quantizations.configure_quantization(cfg),
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      _randomize_nondegenerate_weights(model, cfg.num_experts, seed=31)

      moe_sel, idx_sel = fd_check.capture_and_stack_selections(
          model,
          lambda: pre_train.loss_fn(model, cfg, batch, None, None, is_train=True),
      )

      with fd_check.freeze_discrete_selections(model, moe_sel, idx_sel):
        gdef, params, rest = nnx.split(model, nnx.Param, ...)

        def scalar_loss_fn(p):
          m = nnx.merge(gdef, p, rest, copy=True)
          loss, _ = pre_train.loss_fn(m, cfg, batch, None, None, is_train=True)
          return loss

        report = fd_check.directional_fd_check(
            scalar_loss_fn,
            params,
            num_directions=4,
            epsilons=(1e-2, 5e-3, 1e-3),
            seed=101,
            rtol=5e-4,
        )

    for d in report.directions:
      print(
          f"[fd_check:deepseek4-tiny] dir={d.direction_idx} analytical={d.analytical:+.8e} "
          f"best_eps={d.best_eps:.0e} fd={d.fd_estimates[d.best_eps]:+.8e} "
          f"min_rel_err={d.min_rel_error:.3e} all_rel={ {k: f'{v:.2e}' for k, v in d.rel_errors.items()} }"
      )
    self.assertTrue(report.passed, f"Directional FD check failed: max_min_rel_error={report.max_min_rel_error:.3e}")


if __name__ == "__main__":
  unittest.main()
