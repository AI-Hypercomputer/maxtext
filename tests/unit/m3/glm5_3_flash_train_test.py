# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for GLM-5.3-Flash training bringup: loss, masks, remat, and training gates."""

# pylint: disable=protected-access

import contextlib
import dataclasses
import importlib
import importlib.util
import io
import os
import sys
import types
from typing import Any
import unittest

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_flops
from maxtext.m3.models.glm5_3_flash import glm5_train
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.utils import maxtext_utils
from tests.utils import glm5_test_utils
import numpy as np
import optax
import pytest

try:
  from maxtext.training_engine import maxtext_engine

  HAS_TUNIX = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TUNIX = False

try:
  HAS_TOKAMAX_KDA = "TEST_SRCDIR" in os.environ or importlib.util.find_spec("tokamax._src.ops.kda") is not None
except (ImportError, ValueError):
  HAS_TOKAMAX_KDA = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TOKAMAX_KDA, reason="tokamax KDA not available"),
]

_RUN_TPU_TRAIN_TEST = flags.DEFINE_bool("run_tpu_train_test", False, "Whether to run TPU training test.")


class _TrainConfig(glm5_config.Glm5Config, types.SimpleNamespace):
  """MaxText-style config inheriting from Glm5Config and SimpleNamespace."""


def _make_train_cfg(
    num_hidden_layers: int = 7,
    scan_layers: bool = True,
    remat_policy: str = "full",
    dtype: str = "float32",
    weight_dtype: str = "float32",
    **overrides: Any,
) -> _TrainConfig:
  """Constructs a compact Glm5Config using shared glm5_test_utils._make_cfg."""
  base_cfg, _ = glm5_test_utils._make_cfg(
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      dtype=dtype,
      weight_dtype=weight_dtype,
  )
  cfg_dict = dataclasses.asdict(base_cfg)
  cfg_dict.update(overrides)
  cfg_dict["remat_policy"] = remat_policy

  renames = {
      "emb_dim": "hidden_size",
      "num_decoder_layers": "num_hidden_layers",
      "num_query_heads": "num_attention_heads",
      "num_kv_heads": "num_key_value_heads",
      "mlp_dim": "intermediate_size",
      "moe_mlp_dim": "moe_intermediate_size",
      "first_num_dense_layers": "first_k_dense_replace",
      "num_experts": "n_routed_experts",
      "shared_experts": "n_shared_experts",
      "normalization_layer_epsilon": "rms_norm_eps",
      "indexer_n_heads": "index_n_heads",
      "indexer_head_dim": "index_head_dim",
      "indexer_topk": "index_topk",
      "routed_score_func": "scoring_func",
      "n_routing_groups": "n_group",
      "topk_routing_group": "topk_group",
      "sinkhorn_iterations": "mhc_sinkhorn_iters",
      "mtp_num_layers": "num_nextn_predict_layers",
  }
  cfg = _TrainConfig(**cfg_dict)
  for maxtext_name, glm_name in renames.items():
    object.__setattr__(cfg, maxtext_name, getattr(cfg, glm_name))
  object.__setattr__(cfg, "use_m3_model", True)
  return cfg


class Glm5TrainTest(absltest.TestCase):

  def test_m1_maxtext_utils_lazy_import_does_not_load_tokamax_kda(self):
    """Verifies importing maxtext_utils does not eagerly load tokamax KDA or glm5_3_flash."""
    saved_modules: dict[str, Any] = {}
    mt_utils_mod = maxtext_utils.__name__
    glm5_pkg = glm5_flops.__package__
    glm5_flops_mod = glm5_flops.__name__
    prefixes = (
        mt_utils_mod,
        glm5_pkg,
        "tokamax._src.ops.kda",
    )
    for k in list(sys.modules):
      if k.startswith(prefixes):
        saved_modules[k] = sys.modules.pop(k)
    try:
      importlib.import_module(mt_utils_mod)
      self.assertNotIn("tokamax._src.ops.kda", sys.modules)
      self.assertNotIn(glm5_pkg, sys.modules)
      self.assertNotIn(glm5_flops_mod, sys.modules)
    finally:
      sys.modules.update(saved_modules)

  def test_f23_remat_policy_mapping(self):
    """Verifies F23 remat_policy mapping to JAX checkpoint policies and loud rejection."""
    cfg_none = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    m_none = m3_models.create_model(cfg_none, None, rngs=nnx.Rngs(0))
    self.assertIsNone(m_none.remat_policy)

    cfg_full = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="full")
    m_full = m3_models.create_model(cfg_full, None, rngs=nnx.Rngs(0))
    self.assertEqual(m_full.remat_policy, jax.checkpoint_policies.nothing_saveable)

    m_full_opt = nnx.Optimizer(m_full, optax.adamw(1e-2), wrt=nnx.Param)
    dummy_batch = {
        "inputs": jnp.array([[1, 2, 3, 4]]),
        "targets": jnp.array([[2, 3, 4, 5]]),
        "targets_segmentation": jnp.ones((1, 4), dtype=jnp.int32),
    }
    (loss_val, _), grads = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.glm5_loss_fn(m, is_train=True, **dummy_batch),
        has_aux=True,
    )(m_full)
    m_full_opt.update(m_full, grads)
    self.assertTrue(jnp.isfinite(loss_val))

    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.create_glm5_3_flash_model(cfg_none, rngs=nnx.Rngs(0), remat_policy="none")

    missing_cfg = types.SimpleNamespace()
    with self.assertRaises(AttributeError):
      modeling_glm5_3_flash.create_glm5_3_flash_model(missing_cfg, rngs=nnx.Rngs(0))

    bad_type_cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    object.__setattr__(bad_type_cfg, "remat_policy", 123)
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.create_glm5_3_flash_model(bad_type_cfg, rngs=nnx.Rngs(0))

    for bad_policy in ("", "minimal", "save_dot_except_mlp", "bogus_policy"):
      bad_cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
      object.__setattr__(bad_cfg, "remat_policy", bad_policy)
      with self.assertRaises(ValueError):
        modeling_glm5_3_flash.create_glm5_3_flash_model(bad_cfg, rngs=nnx.Rngs(0))

  def test_remat_gradient_equality(self):
    """Verifies all-leaf parameter gradients match between remat_policy='full' and 'none' (P2-53)."""
    orig_prec = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    try:
      with glm5_test_utils._fp64_mode():
        seq_len = 8
        tokens = jnp.array([[2, 4, 6, 8, 10, 12, 14, 16]])
        target_tokens = jnp.array([[4, 6, 8, 10, 12, 14, 16, 18]])
        target_mask = jnp.ones((1, seq_len), dtype=jnp.int32)

        for scan_layers in (False, True):
          with self.subTest(scan_layers=scan_layers):
            cfg_none = _make_train_cfg(
                num_hidden_layers=5,
                scan_layers=scan_layers,
                remat_policy="none",
                dtype="float64",
                weight_dtype="float64",
            )
            cfg_full = _make_train_cfg(
                num_hidden_layers=5,
                scan_layers=scan_layers,
                remat_policy="full",
                dtype="float64",
                weight_dtype="float64",
            )
            model_none = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg_none, rngs=nnx.Rngs(42))
            model_full = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg_full, rngs=nnx.Rngs(42))
            nnx.update(model_full, nnx.state(model_none, nnx.Param))

            @nnx.jit
            def grad_fn(m):
              def loss_fn(mdl):
                loss_val, _ = modeling_glm5_3_flash.compute_glm5_loss(
                    mdl, tokens, target_tokens, target_mask, is_train=True
                )
                return loss_val

              _, grads = nnx.value_and_grad(loss_fn)(m)
              return grads

            grads_none = grad_fn(model_none)
            grads_full = grad_fn(model_full)

            leaves_none = jax.tree.leaves(grads_none)
            leaves_full = jax.tree.leaves(grads_full)
            self.assertNotEmpty(leaves_none)
            self.assertEqual(len(leaves_none), len(leaves_full))
            max_abs = 0.0
            max_rel = 0.0
            max_frob_rel = 0.0
            for idx, (g_none, g_full) in enumerate(zip(leaves_none, leaves_full, strict=True)):
              g_none_np = np.asarray(g_none)
              g_full_np = np.asarray(g_full)
              np.testing.assert_allclose(g_full_np, g_none_np, rtol=1e-12, atol=1e-12)
              abs_diff = float(np.max(np.abs(g_full_np - g_none_np)))
              max_abs = max(max_abs, abs_diff)
              if np.all(g_none_np == 0.0) and np.all(g_full_np == 0.0):
                continue
              frob_rel, max_rel_inf, _ = glm5_test_utils._assert_e3(
                  g_full_np,
                  g_none_np,
                  name=f"remat_grad_scan_{scan_layers}_leaf_{idx}",
                  is_fp64=True,
              )
              max_rel = max(max_rel, max_rel_inf)
              max_frob_rel = max(max_frob_rel, frob_rel)
            print(
                f"[REMAT_GRAD] scan_layers={scan_layers} max_abs={max_abs:.3e}"
                f" max_rel={max_rel:.3e} max_frob_rel={max_frob_rel:.3e}",
                flush=True,
            )
    finally:
      jax.config.update("jax_default_matmul_precision", orig_prec)

  def test_compute_glm5_loss_and_validation(self):
    """Verifies Row 89 contract, Row 92 shape & entry validation, and loss calculation."""
    cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(0))

    tokens = jnp.array([[10, 20, 30, 40]])
    target_tokens = jnp.array([[20, 30, 40, 50]])  # Next-token shifted
    target_mask = jnp.array([[1, 1, 0, 0]])

    # Successful call with next-token shifted targets
    total_loss, aux = modeling_glm5_3_flash.compute_glm5_loss(
        model,
        tokens=tokens,
        decoder_target_tokens=target_tokens,
        decoder_target_mask=target_mask,
        is_train=True,
    )
    self.assertTrue(jnp.isfinite(total_loss))
    self.assertEqual(total_loss.dtype, jnp.float32)
    self.assertEqual(aux["ce_loss"].dtype, jnp.float32)
    self.assertEqual(aux["moe_lb_loss"].dtype, jnp.float32)
    self.assertEqual(float(aux["total_weights"]), 2.0)

    # Verify non-unit non-zero mask values binarize identically in untiled & tiled modes
    for tiling in (1, 2):
      cfg_t = dataclasses.replace(cfg, num_vocab_tiling=tiling)
      m_t = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg_t, rngs=nnx.Rngs(0))
      loss_unit, aux_unit = modeling_glm5_3_flash.compute_glm5_loss(
          m_t,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=target_mask,
          is_train=True,
      )
      loss_scaled, aux_scaled = modeling_glm5_3_flash.compute_glm5_loss(
          m_t,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=target_mask * 5,
          is_train=True,
      )
      np.testing.assert_allclose(float(loss_unit), float(loss_scaled), rtol=1e-6)
      np.testing.assert_allclose(float(aux_unit["ce_loss"]), float(aux_scaled["ce_loss"]), rtol=1e-6)
      np.testing.assert_allclose(float(aux_unit["xent_sum"]), float(aux_scaled["xent_sum"]), rtol=1e-6)
      self.assertEqual(float(aux_scaled["total_weights"]), 2.0)

    # Missing/None arguments raise ValueError
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=None,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=target_mask,
          is_train=True,
      )
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=tokens,
          decoder_target_tokens=None,
          decoder_target_mask=target_mask,
          is_train=True,
      )
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=None,
          is_train=True,
      )

    # Row 92: Target shape mismatch raises ValueError
    bad_shape_targets = jnp.array([[20, 30, 40]])
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=tokens,
          decoder_target_tokens=bad_shape_targets,
          decoder_target_mask=target_mask,
          is_train=True,
      )
    bad_shape_mask = jnp.array([[1, 1, 0]])
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=bad_shape_mask,
          is_train=True,
      )

    # Omitting is_train or invalid is_train type raises TypeError
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.compute_glm5_loss(  # pylint: disable=missing-kwoa
          model,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=target_mask,
      )
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=tokens,
          decoder_target_tokens=target_tokens,
          decoder_target_mask=target_mask,
          is_train="not_a_bool",  # type: ignore
      )
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.glm5_loss_fn(  # pylint: disable=missing-kwoa
          model,
          inputs=tokens,
          targets=target_tokens,
          targets_segmentation=target_mask,
      )
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.glm5_loss_fn(
          model,
          inputs=tokens,
          targets=target_tokens,
          targets_segmentation=target_mask,
          is_train=123,  # type: ignore
      )

    # MaxTextTrainingEngine glm5_loss_fn compatibility wrapper
    engine_loss, engine_aux = modeling_glm5_3_flash.glm5_loss_fn(
        model,
        inputs=tokens,
        targets=target_tokens,
        targets_segmentation=target_mask,
        is_train=True,
    )
    self.assertEqual(float(engine_loss), float(total_loss))
    self.assertEqual(float(engine_aux["ce_loss"]), float(aux["ce_loss"]))

  def test_bfloat16_loss_and_metrics_are_float32(self):
    """Row 91: Verifies bfloat16 model produces float32 scalars for loss, ce_loss, and moe_lb_loss."""
    cfg_bf16 = _make_train_cfg(
        num_hidden_layers=5,
        scan_layers=False,
        remat_policy="none",
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )
    model_bf16 = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg_bf16, rngs=nnx.Rngs(123))

    tokens = jnp.array([[5, 12, 18, 22]])
    target_tokens = jnp.array([[12, 18, 22, 25]])
    target_mask = jnp.ones((1, 4), dtype=jnp.int32)

    total_loss, aux = modeling_glm5_3_flash.compute_glm5_loss(
        model_bf16,
        tokens=tokens,
        decoder_target_tokens=target_tokens,
        decoder_target_mask=target_mask,
        is_train=True,
    )
    self.assertEqual(total_loss.dtype, jnp.float32)
    self.assertEqual(aux["ce_loss"].dtype, jnp.float32)
    self.assertEqual(aux["moe_lb_loss"].dtype, jnp.float32)
    self.assertTrue(jnp.isfinite(total_loss))
    self.assertTrue(jnp.isfinite(aux["ce_loss"]))
    self.assertTrue(jnp.isfinite(aux["moe_lb_loss"]))

  def test_optax_masked_weight_decay_invariance(self):
    """Row 90: Verifies weight_decay=0.1 does not decay router bias or indexer parameters."""
    cfg = _make_train_cfg(num_hidden_layers=7, scan_layers=True, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(42))

    # Initialize router bias to non-zero values
    for _, bias_var in jax.tree.leaves_with_path(
        nnx.state(model, modeling_glm5_3_flash.Glm5RouterBias),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    ):
      bias_var.value = jnp.ones_like(bias_var.value) * 0.75

    opt_tx = glm5_train.create_glm5_optimizer(learning_rate=1e-2, weight_decay=0.1)
    optimizer = nnx.Optimizer(model, opt_tx, wrt=nnx.Param)

    # Snapshot before training step
    params_before = nnx.state(model, nnx.Param)
    indexer_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_before)
        if "indexer" in jax.tree_util.keystr(p)
    }
    self.assertNotEmpty(indexer_before)

    router_bias_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    self.assertNotEmpty(router_bias_before)

    tokens = jnp.array([[2, 4, 6, 8, 10, 12, 14, 16]])
    target_tokens = jnp.array([[4, 6, 8, 10, 12, 14, 16, 18]])
    target_mask = jnp.ones((1, 8), dtype=jnp.int32)

    loss_step, _ = glm5_train.train_step(
        model=model,
        optimizer=optimizer,
        tokens=tokens,
        decoder_target_tokens=target_tokens,
        decoder_target_mask=target_mask,
    )
    self.assertTrue(jnp.isfinite(loss_step))

    # Verify bit-for-bit identity on indexer parameters despite weight_decay > 0
    params_after = nnx.state(model, nnx.Param)
    indexer_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_after)
        if "indexer" in jax.tree_util.keystr(p)
    }
    for k, val_before in indexer_before.items():
      self.assertTrue(
          np.array_equal(val_before, indexer_after[k]),
          f"Indexer parameter drifted under weight decay: {k}",
      )

    # Verify bit-for-bit identity on router bias despite weight_decay > 0
    opt_bias = nnx.Optimizer(model, opt_tx, wrt=(nnx.Param, modeling_glm5_3_flash.Glm5RouterBias))
    zero_grads = jax.tree.map(
        jnp.zeros_like,
        nnx.state(model, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)),
    )
    opt_bias.update(model, zero_grads)
    router_bias_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    for k, val_before in router_bias_before.items():
      self.assertTrue(
          np.array_equal(val_before, router_bias_after[k]),
          f"Router bias drifted under weight decay: {k}",
      )

  def test_f16_validate_no_packing(self):
    """Verifies F16 no-packing enforcement on Glm5Config and validate_no_packing."""
    with self.assertRaises(ValueError):
      _make_train_cfg(remat_policy="none", packing=True)

    with self.assertRaises(ValueError):
      glm5_config.validate_no_packing(np.array([[1, 1, 2, 2]]))

    glm5_config.validate_no_packing(None)
    glm5_config.validate_no_packing(np.array([[1, 1, 1, 1]]))
    glm5_config.validate_no_packing(np.array([[0, 0, 1, 1]]))

  def test_phase2_multi_step_training_and_frozen_params(self):
    """Phase 2: 10 training steps on 7L model verifying loss decrease and active expert training."""
    cfg = _make_train_cfg(num_hidden_layers=7, scan_layers=True, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(42))

    tx = glm5_train.create_glm5_optimizer(learning_rate=1e-2, weight_decay=0.0)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    seq_len = 8
    np.random.seed(123)
    inputs_np = np.random.randint(1, cfg.vocab_size, (1, seq_len))
    targets_np = np.random.randint(1, cfg.vocab_size, (1, seq_len))
    batch = {
        "tokens": jnp.array(inputs_np),
        "decoder_target_tokens": jnp.array(targets_np),
        "decoder_target_mask": jnp.ones((1, seq_len), dtype=jnp.int32),
        "decoder_segment_ids": jnp.ones((1, seq_len), dtype=jnp.int32),
    }

    # Verify maxtext_engine normalization integration
    if HAS_TUNIX:
      initial_loss, initial_aux = modeling_glm5_3_flash.compute_glm5_loss(model, is_train=True, **batch)
      normalized = maxtext_engine._normalize_loss_output(  # pylint: disable=protected-access
          (initial_loss, initial_aux), has_aux=True
      )
      self.assertIn("ce_loss", normalized.aux_metrics)
      self.assertIn("moe_lb_loss", normalized.aux_metrics)

    @nnx.jit
    def step_fn(m: Any, opt: Any, b: dict[str, Any]):
      def loss_fn(mod):
        loss, aux = modeling_glm5_3_flash.compute_glm5_loss(mod, is_train=True, **b)
        return loss, aux

      (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
      opt.update(m, grads)
      return loss, aux, grads

    losses = []
    for step in range(10):
      loss, aux, grads = step_fn(model, optimizer, batch)
      self.assertTrue(jnp.isfinite(loss), f"Loss non-finite at step {step}: {loss}")
      self.assertTrue(jnp.isfinite(aux["ce_loss"]), f"CE loss non-finite at step {step}")
      self.assertTrue(
          jnp.isfinite(aux["moe_lb_loss"]),
          f"MoE LB loss non-finite at step {step}",
      )
      losses.append(float(loss))

      if step in (0, 9):
        grad_leaves = {jax.tree_util.keystr(p): np.asarray(g) for p, g in jax.tree.leaves_with_path(grads)}
        for k, g_val in grad_leaves.items():
          self.assertTrue(
              np.all(np.isfinite(g_val)),
              f"Gradient non-finite in {k} at step {step}",
          )
          if "indexer" in k:
            self.assertTrue(
                np.all(g_val == 0.0),
                f"Indexer grad non-zero in {k} at step {step}",
            )

    self.assertLess(
        losses[-1],
        losses[0],
        f"Loss did not decrease: initial={losses[0]}, final={losses[-1]}",
    )

  def test_multi_step_loss_decrease_with_scan_and_sharded_mesh(self):
    """Row 94: Verifies 5-step loss decrease with scan_layers=True and remat under Glm5Sharding mesh."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))
    _ = glm5_config.Glm5Sharding.from_mesh(mesh)

    cfg = _make_train_cfg(num_hidden_layers=7, scan_layers=True, remat_policy="full")
    with jax.sharding.set_mesh(mesh):
      model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, mesh=mesh, rngs=nnx.Rngs(2026))

      opt = glm5_train.create_glm5_optimizer(learning_rate=2e-2, weight_decay=0.0)
      optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

      tokens = jnp.array([[3, 7, 11, 15, 19, 23, 27, 31]])
      target_tokens = jnp.array([[7, 11, 15, 19, 23, 27, 31, 35]])
      target_mask = jnp.ones((1, 8), dtype=jnp.int32)

      @nnx.jit
      def jitted_step(m, opt, tok, tgt, msk):
        return glm5_train.train_step(
            model=m,
            optimizer=opt,
            tokens=tok,
            decoder_target_tokens=tgt,
            decoder_target_mask=msk,
        )

      losses = []
      for step in range(5):
        loss_val, _ = jitted_step(model, optimizer, tokens, target_tokens, target_mask)
        self.assertTrue(jnp.isfinite(loss_val), f"Loss non-finite at step {step}")
        losses.append(float(loss_val))

    self.assertLess(
        losses[-1],
        losses[0],
        f"5-step training with scan+remat did not decrease loss: {losses[0]} ->" f" {losses[-1]}",
    )

  def test_phase3_train_ce_decrease_and_disjoint_eval_finiteness(self):
    """Verifies training cross-entropy decreases while evaluation on disjoint token IDs stays finite and reflects

    updated shared parameters.
    """
    cfg = _make_train_cfg(num_hidden_layers=7, scan_layers=True, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(101))

    tx = glm5_train.create_glm5_optimizer(learning_rate=2e-2, weight_decay=0.0)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    seq_len = 8
    # Training tokens and completely distinct, non-overlapping holdout tokens
    train_tokens = np.array([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.int32)
    train_targets = np.array([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=np.int32)
    train_batch = {
        "tokens": jnp.array(train_tokens),
        "decoder_target_tokens": jnp.array(train_targets),
        "decoder_target_mask": jnp.ones((1, seq_len), dtype=jnp.int32),
    }

    holdout_tokens = np.array([[30, 31, 32, 33, 34, 35, 36, 37]], dtype=np.int32)
    holdout_targets = np.array([[31, 32, 33, 34, 35, 36, 37, 38]], dtype=np.int32)
    holdout_batch = {
        "tokens": jnp.array(holdout_tokens),
        "decoder_target_tokens": jnp.array(holdout_targets),
        "decoder_target_mask": jnp.ones((1, seq_len), dtype=jnp.int32),
    }

    def eval_ce_and_prob(m, b):
      _, aux = modeling_glm5_3_flash.compute_glm5_loss(m, is_train=False, **b)
      logits = m(b["tokens"])
      probs = jax.nn.softmax(logits, axis=-1)
      target_probs = jnp.take_along_axis(probs, b["decoder_target_tokens"][..., None], axis=-1).squeeze(-1)
      return float(aux["ce_loss"]), float(jnp.mean(target_probs))

    pre_train_ce, pre_train_prob = eval_ce_and_prob(model, train_batch)
    pre_ce, pre_prob = eval_ce_and_prob(model, holdout_batch)

    @nnx.jit
    def jitted_step(m, opt, tok, tgt, msk):
      return glm5_train.train_step(
          model=m,
          optimizer=opt,
          tokens=tok,
          decoder_target_tokens=tgt,
          decoder_target_mask=msk,
      )

    for _ in range(15):
      jitted_step(
          model,
          optimizer,
          train_batch["tokens"],
          train_batch["decoder_target_tokens"],
          train_batch["decoder_target_mask"],
      )

    post_train_ce, post_train_prob = eval_ce_and_prob(model, train_batch)
    post_ce, post_prob = eval_ce_and_prob(model, holdout_batch)
    self.assertLess(post_train_ce, pre_train_ce)
    self.assertGreater(post_train_prob, pre_train_prob)
    self.assertTrue(jnp.isfinite(post_ce))
    self.assertTrue(jnp.isfinite(post_prob))
    self.assertNotEqual(post_ce, pre_ce)
    self.assertNotEqual(post_prob, pre_prob)

  def test_vocab_tiling_loss_and_gradient_parity(self):
    """Verifies num_vocab_tiling=1, 2, and 4 produce exact loss and gradient parity."""
    cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(42))

    tokens = jnp.array([[2, 4, 6, 8, 10, 12, 14, 16]])
    target_tokens = jnp.array([[4, 6, 8, 10, 12, 14, 16, 18]])
    target_mask = jnp.array([[1, 1, 1, 1, 0, 1, 1, 0]], dtype=jnp.int32)

    # 1. Untiled (num_vocab_tiling=1)
    object.__setattr__(model.config, "num_vocab_tiling", 1)
    (loss_1, aux_1), grads_1 = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.compute_glm5_loss(m, tokens, target_tokens, target_mask, is_train=True),
        has_aux=True,
    )(model)

    # 2. Tiled with 2 chunks
    object.__setattr__(model.config, "num_vocab_tiling", 2)
    (loss_2, aux_2), grads_2 = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.compute_glm5_loss(m, tokens, target_tokens, target_mask, is_train=True),
        has_aux=True,
    )(model)

    # 3. Tiled with 4 chunks
    object.__setattr__(model.config, "num_vocab_tiling", 4)
    (loss_4, aux_4), grads_4 = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.compute_glm5_loss(m, tokens, target_tokens, target_mask, is_train=True),
        has_aux=True,
    )(model)

    # Reset
    object.__setattr__(model.config, "num_vocab_tiling", 1)

    np.testing.assert_allclose(float(loss_1), float(loss_2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(loss_1), float(loss_4), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["ce_loss"]), float(aux_2["ce_loss"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["ce_loss"]), float(aux_4["ce_loss"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["xent_sum"]), float(aux_2["xent_sum"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["xent_sum"]), float(aux_4["xent_sum"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        float(aux_1["total_weights"]),
        float(aux_2["total_weights"]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(aux_1["total_weights"]),
        float(aux_4["total_weights"]),
        rtol=1e-5,
        atol=1e-5,
    )

    g1_leaves = jax.tree.leaves(grads_1)
    g2_leaves = jax.tree.leaves(grads_2)
    for g1, g2 in zip(g1_leaves, g2_leaves):
      np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-4, atol=1e-4)

    g4_leaves = jax.tree.leaves(grads_4)
    for g1, g4 in zip(g1_leaves, g4_leaves):
      np.testing.assert_allclose(np.asarray(g1), np.asarray(g4), rtol=1e-4, atol=1e-4)

    # Indivisible tile count raises ValueError
    object.__setattr__(model.config, "num_vocab_tiling", 3)
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.compute_glm5_loss(model, tokens, target_tokens, target_mask, is_train=True)
    object.__setattr__(model.config, "num_vocab_tiling", 1)

  def test_vocab_tiling_remat_saves_no_vocab_sized_residuals(self):
    """P2b-16: Verifies scan remat in tiled cross entropy saves no vocab-sized residuals in backward pass."""
    cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(42))
    batch_size = 2
    seq_len = 8
    hidden_states = jnp.zeros((batch_size, seq_len, cfg.hidden_size), dtype=jnp.float32)
    target_tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      jax.ad_checkpoint.print_saved_residuals(
          lambda h: modeling_glm5_3_flash._compute_tiled_cross_entropy(  # pylint: disable=protected-access
              model, h, target_tokens, target_mask, num_vocab_tiling=4
          )[0],
          hidden_states,
      )
    out_str = buf.getvalue()
    # Ensure no residual tensor saved has last dimension equal to vocab_size
    self.assertNotIn(f", {cfg.vocab_size})", out_str)

  @pytest.mark.post_training
  @unittest.skipUnless(HAS_TUNIX, "tunix not available")
  def test_configure_glm5_engine_integration(self):
    """Verifies configure_glm5_engine attaches glm5_engine_loss_fn with has_aux=True and no gen_input_fn."""

    class DummyOptimizer:
      """Dummy optimizer for testing configure_glm5_engine."""

      def __init__(self, tx: Any):
        self.tx = tx

    class DummyEngine:
      """Dummy training engine for testing configure_glm5_engine."""

      def __init__(self):
        self.loss_fn = None
        self.has_aux = False
        self.gen_input_fn = None
        self._optimizer = DummyOptimizer(optax.adamw(1e-3))
        self._tx = None
        self.model = None

      def with_loss_fn(self, fn, has_aux=False):
        self.loss_fn = fn
        self.has_aux = has_aux
        return self

      def with_gen_model_input_fn(self, fn):
        self.gen_input_fn = fn
        return self

      def _build_optimizer(self, tx):
        return DummyOptimizer(tx)

    dummy_engine = DummyEngine()
    configured = glm5_train.configure_glm5_engine(dummy_engine)
    self.assertIs(configured, dummy_engine)
    self.assertIs(dummy_engine.loss_fn, glm5_train.glm5_engine_loss_fn)
    self.assertTrue(dummy_engine.has_aux)
    self.assertIsNone(dummy_engine.gen_input_fn)

    sample_batch = {
        "inputs": jnp.array([[1, 2, 3, 4]]),
        "targets": jnp.array([[2, 3, 4, 5]]),
        "targets_segmentation": jnp.ones((1, 4), dtype=jnp.int32),
        "targets_position": jnp.arange(4, dtype=jnp.int32)[None, :],
        "inputs_segmentation": jnp.ones((1, 4), dtype=jnp.int32),
        "inputs_position": jnp.arange(4, dtype=jnp.int32)[None, :],
        "extra_metadata": "pipeline_extra",
    }

    cfg = _make_train_cfg(num_hidden_layers=5, scan_layers=False, remat_policy="none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(0))

    # is_train=True
    primary_loss, aux = dummy_engine.loss_fn(model, cfg, sample_batch, None, None, is_train=True)
    normalized = maxtext_engine._normalize_loss_output(  # pylint: disable=protected-access
        (primary_loss, aux), has_aux=True
    )
    expected_total_loss, _ = modeling_glm5_3_flash.glm5_loss_fn(
        model,
        inputs=sample_batch["inputs"],
        targets=sample_batch["targets"],
        targets_segmentation=sample_batch["targets_segmentation"],
        is_train=True,
    )
    np.testing.assert_allclose(
        float(normalized.primary_loss.unreduced_sum / normalized.primary_loss.denominator),
        float(expected_total_loss),
        rtol=1e-6,
        atol=1e-6,
    )
    self.assertIn("ce_loss", normalized.aux_metrics)
    self.assertIn("moe_lb_loss", normalized.aux_metrics)
    self.assertGreater(float(aux["moe_lb_loss"]), 0.0)

    # is_train=False
    primary_loss_eval, aux_eval = dummy_engine.loss_fn(model, cfg, sample_batch, None, None, is_train=False)
    normalized_eval = maxtext_engine._normalize_loss_output(  # pylint: disable=protected-access
        (primary_loss_eval, aux_eval), has_aux=True
    )
    expected_eval_loss, expected_eval_aux = modeling_glm5_3_flash.glm5_loss_fn(
        model,
        inputs=sample_batch["inputs"],
        targets=sample_batch["targets"],
        targets_segmentation=sample_batch["targets_segmentation"],
        is_train=False,
    )
    np.testing.assert_allclose(
        float(normalized_eval.primary_loss.unreduced_sum / normalized_eval.primary_loss.denominator),
        float(expected_eval_loss),
        rtol=1e-6,
        atol=1e-6,
    )
    self.assertEqual(float(aux_eval["moe_lb_loss"]), float(expected_eval_aux["moe_lb_loss"]))
    # In CL7 (before MTP), total_loss == eval_loss in both modes
    np.testing.assert_allclose(
        float(expected_eval_loss),
        float(expected_total_loss),
        rtol=1e-6,
        atol=1e-6,
    )

  def test_exact_flop_breakdown(self):
    """Verifies exact FLOP equation with and without MTP."""
    # Independent hand derivation for Glm5Config() (B=1, S=4096, grad_accum=1, mtp_num_layers=1):
    # - Forward learnable weights:
    #   unembedding 5.19691042816 + mHC 0.33822867456 + dense FFN 3.710851743744
    #   + MoE FFN 78.333761028096 + KDA weights 38.359963533312 + MLA weights 10.582799417344
    #   + MTP weights 8.298950557696 = 144.821465382912 TFLOPs;
    #   frozen indexer weight (12 layers at 1x) = 0.734439407616 TFLOPs
    #   -> learnable_weight_tflops = 3 * 144.821465382912 + 0.734439407616 = 435.198835556352 TFLOPs
    #   (vs 436.668 TFLOPs when indexer was counted at 3x).
    # - Forward trainable attention:
    #   KDA 1.168231104512 + MLA 4.535485464576 + MTP MLA 0.412316860416 = 6.116033429504 TFLOPs;
    #   frozen indexer scoring (12 layers at 1x) = 0.207769042944 TFLOPs
    #   -> attention_tflops = 3 * 6.116033429504 + 0.207769042944 = 18.555869331456 TFLOPs
    #   (vs 18.971 TFLOPs when indexer was counted at 3x).
    # - Total training TFLOPs = 453.754704887808 (453.755 TFLOPs;
    #   vs 455.639 TFLOPs before 1.884 TFLOPs 1x-indexer correction).
    # - Forward FLOPs/token = 151.879707262976 * 1e12 / 4096 = 37.080 GFLOP/token.
    base_glm = glm5_config.Glm5Config()
    cfg_dict = {f.name: getattr(base_glm, f.name) for f in dataclasses.fields(base_glm)}
    renames = {
        "hidden_size": "emb_dim",
        "num_hidden_layers": "num_decoder_layers",
        "num_attention_heads": "num_query_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "mlp_dim",
        "moe_intermediate_size": "moe_mlp_dim",
        "first_k_dense_replace": "first_num_dense_layers",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "shared_experts",
        "rms_norm_eps": "normalization_layer_epsilon",
        "index_n_heads": "indexer_n_heads",
        "index_head_dim": "indexer_head_dim",
        "index_topk": "indexer_topk",
        "scoring_func": "routed_score_func",
        "n_group": "n_routing_groups",
        "topk_group": "topk_routing_group",
        "mhc_sinkhorn_iters": "sinkhorn_iterations",
        "num_nextn_predict_layers": "mtp_num_layers",
        "moe_lb_loss_weight": "load_balance_loss_weight",
        "mlp_clamp_limit": "mlp_activations_limit",
    }
    for glm_k, maxtext_k in renames.items():
      cfg_dict[maxtext_k] = cfg_dict[glm_k]
    cfg_dict.update(
        model_name="glm5.3-flash",
        use_m3_model=True,
        per_device_batch_size=1,
        max_target_length=4096,
        gradient_accumulation_steps=1,
    )
    cfg = types.SimpleNamespace(**cfg_dict)

    tflops, weight_tflops, attn_tflops = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg, log=False)
    self.assertAlmostEqual(tflops, 453.755, places=3)
    self.assertAlmostEqual(weight_tflops, 435.199, places=3)
    self.assertAlmostEqual(attn_tflops, 18.556, places=3)
    self.assertAlmostEqual(tflops, weight_tflops + attn_tflops, places=6)

    # Independent hand derivation of MTP delta from config dimensions:
    # - MTP forward learnable weights:
    #   eh_proj (2*B*S*2H*H = 0.274877906944) + 1x mHC (4*B*S*(4H)*24 = 0.006442450944)
    #   + 1x MLA weights (2*B*S*116654080 = 0.955630223360)
    #   + 1x MoE FFN (2*B*S*(H*288 + 9*3*H*2048) = 1.865089548288)
    #   + unembedding lm_head (2*B*S*H*154880 = 5.196910428160)
    #   = 8.298950557696 TFLOPs -> 3x training = 24.896851673088 TFLOPs.
    # - MTP forward trainable attention (1x MLA: B*S^2*64*512 = 0.412316860416 TFLOPs)
    #   -> 3x training = 1.236950581248 TFLOPs.
    # - MTP frozen indexer (1x forward-only: weight 0.061203283968 + score 0.017314086912
    #   = 0.078517370880 TFLOPs).
    # - Total MTP training delta = 24.896851673088 + 1.236950581248 + 0.078517370880
    #   = 26.212319625216 TFLOPs (26.212 TFLOPs, 5.78% of 453.755 TFLOPs, matching ~5.8% MTP share).
    # - Therefore tflops_no_mtp = 453.754704887808 - 26.212319625216 = 427.542385262592 TFLOPs (427.542 at places=3).
    cfg_no_mtp = types.SimpleNamespace(**cfg_dict)
    cfg_no_mtp.num_nextn_predict_layers = 0
    cfg_no_mtp.mtp_num_layers = 0
    tflops_no_mtp, weight_no_mtp, attn_no_mtp = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg_no_mtp, log=False)
    self.assertAlmostEqual(tflops_no_mtp, 427.542, places=3)
    self.assertGreater(tflops, tflops_no_mtp)
    self.assertGreater(weight_tflops, weight_no_mtp)
    self.assertGreater(attn_tflops, attn_no_mtp)

    # Test indexer 1.0x forward-only FLOP counting (P2-50)
    cfg_no_idx = types.SimpleNamespace(**cfg_dict)
    cfg_no_idx.use_indexer = False
    _, weight_no_idx, _ = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg_no_idx, log=False)
    b_dim = cfg.per_device_batch_size
    s_dim = cfg.max_target_length
    batch_len = b_dim * s_dim
    glm_cfg = glm5_config.Glm5Config.from_maxtext_config(cfg)
    num_mla_layers = sum(1 for lt in glm_cfg.layer_types if lt == "full_attention")
    total_mla = num_mla_layers + glm_cfg.num_nextn_predict_layers
    expected_indexer_proj = (
        2
        * batch_len
        * (
            glm_cfg.q_lora_rank * glm_cfg.index_n_heads * glm_cfg.index_head_dim
            + 2 * glm_cfg.hidden_size * glm_cfg.index_head_dim
            + glm_cfg.hidden_size * glm_cfg.index_n_heads
        )
    )
    expected_weight_delta_tflops = (1.0 * total_mla * expected_indexer_proj / 10**12) * cfg.gradient_accumulation_steps
    self.assertAlmostEqual(weight_tflops - weight_no_idx, expected_weight_delta_tflops, places=6)

  def test_glm5_3_flash_flop_calculation(self):
    """Verifies GLM-5.3-Flash FLOP equation produces positive deterministic TFLOPs."""
    self.assertIn("glm5.3-flash", m3_models.MODEL_FLOPS_REGISTRY)
    base_glm = _make_train_cfg(num_hidden_layers=45, scan_layers=True, remat_policy="none")
    cfg_dict = {f.name: getattr(base_glm, f.name) for f in dataclasses.fields(base_glm)}
    renames = {
        "hidden_size": "emb_dim",
        "num_hidden_layers": "num_decoder_layers",
        "num_attention_heads": "num_query_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "mlp_dim",
        "moe_intermediate_size": "moe_mlp_dim",
        "first_k_dense_replace": "first_num_dense_layers",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "shared_experts",
        "rms_norm_eps": "normalization_layer_epsilon",
        "index_n_heads": "indexer_n_heads",
        "index_head_dim": "indexer_head_dim",
        "index_topk": "indexer_topk",
        "scoring_func": "routed_score_func",
        "n_group": "n_routing_groups",
        "topk_group": "topk_routing_group",
        "mhc_sinkhorn_iters": "sinkhorn_iterations",
        "num_nextn_predict_layers": "mtp_num_layers",
        "moe_lb_loss_weight": "load_balance_loss_weight",
        "mlp_clamp_limit": "mlp_activations_limit",
    }
    for glm_k, maxtext_k in renames.items():
      cfg_dict[maxtext_k] = cfg_dict[glm_k]
    cfg_dict.update(
        model_name="glm5.3-flash",
        use_m3_model=True,
        per_device_batch_size=2,
        max_target_length=2048,
        gradient_accumulation_steps=1,
    )
    cfg = types.SimpleNamespace(**cfg_dict)

    tflops, weight_tflops, attn_tflops = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg, log=False)
    self.assertGreater(tflops, 0.0)
    self.assertGreater(weight_tflops, 0.0)
    self.assertGreater(attn_tflops, 0.0)
    self.assertAlmostEqual(tflops, weight_tflops + attn_tflops, places=6)

    t_tot, t_wt, t_at = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    self.assertEqual(t_tot, tflops)
    self.assertEqual(t_wt, weight_tflops)
    self.assertEqual(t_at, attn_tflops)

  def test_flop_accounting_indexer_when_seq_le_topk_and_not_always_select_tail(
      self,
  ):
    """Verifies FLOP calculation when s_dim <= index_topk and index_kpool_always_select_tail=False (T50)."""
    base_glm = glm5_config.Glm5Config()
    cfg_dict = {f.name: getattr(base_glm, f.name) for f in dataclasses.fields(base_glm)}
    renames = {
        "hidden_size": "emb_dim",
        "num_hidden_layers": "num_decoder_layers",
        "num_attention_heads": "num_query_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "mlp_dim",
        "moe_intermediate_size": "moe_mlp_dim",
        "first_k_dense_replace": "first_num_dense_layers",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "shared_experts",
        "rms_norm_eps": "normalization_layer_epsilon",
        "index_n_heads": "indexer_n_heads",
        "index_head_dim": "indexer_head_dim",
        "index_topk": "indexer_topk",
        "scoring_func": "routed_score_func",
        "n_group": "n_routing_groups",
        "topk_group": "topk_routing_group",
        "mhc_sinkhorn_iters": "sinkhorn_iterations",
        "num_nextn_predict_layers": "mtp_num_layers",
        "moe_lb_loss_weight": "load_balance_loss_weight",
        "mlp_clamp_limit": "mlp_activations_limit",
    }
    for glm_k, maxtext_k in renames.items():
      cfg_dict[maxtext_k] = cfg_dict[glm_k]
    cfg_dict.update(
        model_name="glm5.3-flash",
        use_m3_model=True,
        per_device_batch_size=1,
        max_target_length=128,  # s_dim <= index_topk (2048)
        gradient_accumulation_steps=1,
        index_kpool_always_select_tail=False,
    )
    cfg = types.SimpleNamespace(**cfg_dict)
    tflops, weight_tflops, attn_tflops = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg, log=False)
    self.assertGreater(tflops, 0.0)
    self.assertGreater(weight_tflops, 0.0)
    self.assertGreater(attn_tflops, 0.0)

    cfg_no_idx = types.SimpleNamespace(**cfg_dict)
    cfg_no_idx.use_indexer = False
    tflops_no_idx, weight_no_idx, _ = glm5_flops.calculate_glm5_3_flash_tflops_per_device(cfg_no_idx, log=False)
    self.assertGreater(weight_tflops, weight_no_idx)
    self.assertGreater(tflops, tflops_no_idx)

  def test_create_glm5_optimizer_rejects_missing_kwargs(self):
    """P2b-18: Verifies create_glm5_optimizer requires learning_rate and weight_decay keyword arguments."""
    with self.assertRaises(TypeError):
      glm5_train.create_glm5_optimizer()  # pylint: disable=missing-kwoa
    with self.assertRaises(TypeError):
      glm5_train.create_glm5_optimizer(learning_rate=1e-2)  # pylint: disable=missing-kwoa
    with self.assertRaises(TypeError):
      glm5_train.create_glm5_optimizer(weight_decay=0.1)  # pylint: disable=missing-kwoa
    with self.assertRaises(TypeError):
      glm5_train.create_glm5_optimizer(1e-2, 0.1)  # pylint: disable=missing-kwoa,too-many-function-args

  @pytest.mark.post_training
  @unittest.skipUnless(HAS_TUNIX, "tunix not available")
  def test_maxtext_training_engine_real_construction_freezes_indexer_and_router_bias(
      self,
  ):
    """P2b-18: Verifies MaxTextTrainingEngine constructs with glm5.3-flash and freezes indexer and router bias."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))

    argv = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=glm5.3-flash",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        "base_emb_dim=128",
        "base_mlp_dim=128",
        "base_moe_mlp_dim=64",
        "first_num_dense_layers=2",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=16",
        "vocab_size=256",
        "num_experts=8",
        "num_experts_per_tok=2",
        "per_device_batch_size=1",
        "max_target_length=16",
        "q_lora_rank=32",
        "kv_lora_rank=16",
        "qk_nope_head_dim=16",
        "v_head_dim=16",
        "indexer_n_heads=8",
        "indexer_head_dim=16",
        "indexer_topk=8",
        "use_m3_model=true",
        "adam_weight_decay=0.1",
        "learning_rate=1e-2",
        "warmup_steps_fraction=0.0",
        "enable_checkpointing=false",
        "use_weight_converter=false",
        f"base_output_directory={self.create_tempdir().full_path}",
        "run_name=test_run",
    ]
    py_cfg = pyconfig.initialize(argv)
    py_cfg.get_keys().update(
        kda_implementation="xla",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )

    engine = maxtext_engine.MaxTextTrainingEngine(py_cfg, mesh=mesh)

    self.assertIs(engine._loss_fn, glm5_train.glm5_engine_loss_fn)  # pylint: disable=protected-access
    self.assertIsNotNone(engine._tx)

    # Initialize router bias to non-zero values
    for _, bias_var in jax.tree.leaves_with_path(
        nnx.state(engine.model, modeling_glm5_3_flash.Glm5RouterBias),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    ):
      bias_var.value = jnp.ones_like(bias_var.value) * 0.75

    params_before = nnx.state(engine.model, nnx.Param)
    indexer_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_before)
        if "indexer" in jax.tree_util.keystr(p)
    }
    router_bias_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(engine.model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    embed_before = np.asarray(engine.model.embed_tokens.embedding.value).copy()

    batch_size = int(py_cfg.micro_batch_size_to_train_on)
    seq_len = int(py_cfg.max_target_length)
    batch = {
        "inputs": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "targets": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "targets_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "targets_position": jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_size, axis=0),
        "inputs_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "inputs_position": jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_size, axis=0),
    }

    self.assertIsInstance(engine._optimizer.opt_state.inner_states["frozen"], optax.MaskedState)
    self.assertIsInstance(
        engine._optimizer.opt_state.inner_states["frozen"].inner_state,
        optax.EmptyState,
    )

    engine.fwd_bwd(batch)
    self.assertLen(engine._cached_losses, 1)
    loss_metric = engine._cached_losses[0]
    self.assertTrue(bool(jnp.isfinite(loss_metric.unreduced_sum)))
    step = engine.update()
    self.assertEqual(step, 1)
    self.assertEqual(engine.train_step, 1)

    # 1. Indexer parameters remain bit-identical
    params_after = nnx.state(engine.model, nnx.Param)
    indexer_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_after)
        if "indexer" in jax.tree_util.keystr(p)
    }
    for k, val_before in indexer_before.items():
      np.testing.assert_array_equal(val_before, indexer_after[k])

    # 2. Router bias remains bit-identical
    router_bias_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(engine.model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    for k, val_before in router_bias_before.items():
      np.testing.assert_array_equal(val_before, router_bias_after[k])

    # 3. Trainable parameters strictly change
    embed_after = np.asarray(engine.model.embed_tokens.embedding.value)
    self.assertFalse(np.array_equal(embed_before, embed_after))

  def test_tpu_training(self):
    """Row 93: TPU training test with loud failure if backend is not TPU."""
    if not _RUN_TPU_TRAIN_TEST.value:
      self.skipTest("Pass --run_tpu_train_test=true to run TPU training test.")
    if jax.default_backend() != "tpu":
      raise RuntimeError(
          "--run_tpu_train_test=true requires jax.default_backend() == 'tpu'," f" but got '{jax.default_backend()}'."
      )
    cfg = _make_train_cfg(num_hidden_layers=7, scan_layers=True, remat_policy="full")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(0))
    opt = glm5_train.create_glm5_optimizer(learning_rate=1e-3, weight_decay=0.0)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    target_tokens = jnp.ones((1, 8), dtype=jnp.int32)
    target_mask = jnp.ones((1, 8), dtype=jnp.int32)
    loss, _ = glm5_train.train_step(model, optimizer, tokens, target_tokens, target_mask)
    self.assertTrue(jnp.isfinite(loss))


if __name__ == "__main__":
  absltest.main()
