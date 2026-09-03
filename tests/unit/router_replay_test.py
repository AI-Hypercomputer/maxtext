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
"""Tests for router replay (forced routing): unit tests for moe.py's

forced_routed_experts handling, the scan-support helpers, and end-to-end
trainer integration tests.

MaxTextTrainingEngine integration lives separately in
tests/post_training/unit/router_replay_engine_test.py, since it imports
maxtext.training_engine.maxtext_engine, which pulls in tunix -- a dependency
only installed for post-training test environments, not the pretrain-unit
environment this file runs under.
"""

import os
import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.configs.types import check_forced_routing_support
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.nnx_decoders import reshape_forced_routed_experts_for_scan
from maxtext.models import models
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import pytest


def _init_test_cfg(extra_args=(), **kwargs):
  """pyconfig.initialize with this file's common test defaults folded in.

  Only pass kwargs that actually need to differ from base.yml (or from the
  selected model's own yaml) -- e.g. omit ici_*_parallelism, enable_nnx,
  pure_nnx, pure_nnx_decoder, sparse_matmul, dtype, and scan_layers=True,
  which already match their base.yml defaults.
  """
  kwargs.setdefault("enable_checkpointing", False)
  kwargs.setdefault("log_config", False)
  kwargs.setdefault("skip_jax_distributed_system", True)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path(), *extra_args],
      **kwargs,
  )


def _tiny_qwen35_kwargs(seq_len, batch_size, num_experts, top_k, **overrides):
  """Base kwargs for a CPU-friendly, shrunk-down Qwen3.5 MoE config."""
  kwargs = {
      "override_model_config": True,
      "model_name": "qwen3.5-35b-a3b",
      "num_experts": num_experts,
      "num_experts_per_tok": top_k,
      "base_emb_dim": 256,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 256,
      "partial_rotary_factor": 0.25,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "vocab_size": 1000,
      "max_target_length": seq_len,
      "max_prefill_predict_length": seq_len,
      "per_device_batch_size": float(batch_size),
      "weight_dtype": "bfloat16",
  }
  kwargs.update(overrides)
  return kwargs


class DummyConfig:
  """Minimal stand-in for MaxTextConfig, for direct RoutedMoE method calls."""

  def __init__(self, model_name="default", decoder_block=ctypes.DecoderBlockType.DEFAULT):
    self.model_name = model_name
    self.decoder_block = decoder_block
    self.norm_topk_prob = False
    self.use_random_routing = False
    self.shard_mode = ctypes.ShardMode.AUTO
    self.routed_score_func = ""
    self.routed_scaling_factor = 2.5
    self.model_call_mode = "train"
    self.fuse_expert_scales = False
    self.n_routing_groups = -1
    self.topk_routing_group = 1


class DummyRoutedMoE:
  """Minimal stand-in for RoutedMoE, for calling its methods unbound."""

  def __init__(self, config, per_expert_scale=None):
    self.config = config
    self.dtype = jnp.float32
    self.num_experts_per_tok = 2
    self.num_experts = 3
    self.is_hash_routing = False
    self.per_expert_scale = None if per_expert_scale is None else nnx.Param(jnp.asarray(per_expert_scale))

  def _maybe_shard_with_logical(self, x, spec):
    return x

  def deepseek_scale_weights(self, weights):
    return moe.RoutedMoE.deepseek_scale_weights(self, weights)

  def deepseek_routing(self, gate_logits, pre_bias_logits):
    return moe.RoutedMoE.deepseek_routing(self, gate_logits, pre_bias_logits)

  def get_topk_indices(self, *args, **kwargs):
    return moe.RoutedMoE.get_topk_indices(self, *args, **kwargs)

  def get_topk(self, *args, **kwargs):
    return moe.RoutedMoE.get_topk(self, *args, **kwargs)


class ForcedRoutingTest(unittest.TestCase):

  def test_basic_override(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits  # Not DeepSeek
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model,
        gate_logits,
        pre_bias_logits,
        forced_routed_experts=forced_routed_experts,
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())
    # Check that weights are extracted correctly and softmaxed
    # For token 0: indices 2, 1 -> logits 3.0, 2.0 -> softmax([3.0, 2.0])
    # For token 1: indices 0, 2 -> logits 4.0, 6.0 -> softmax([4.0, 6.0])
    expected_weights = jax.nn.softmax(jnp.array([[[3.0, 2.0], [4.0, 6.0]]]).astype(jnp.float32), axis=-1)
    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_gemma4_softmax(self):
    config = DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4)
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model,
        gate_logits,
        pre_bias_logits,
        forced_routed_experts=forced_routed_experts,
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())

    # For Gemma 4, it applies softmax to gate_logits first!

    expected_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    expected_weights = jnp.take_along_axis(expected_probs, forced_routed_experts, axis=-1)

    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_reshape_and_update_weights(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    weights = jnp.array([[[0.1, 0.2], [0.3, 0.4]]])  # (1, 2, 2)
    indices = jnp.array([[[2, -1], [-1, 1]]])  # (1, 2, 2)

    update_weights = moe.RoutedMoE.reshape_and_update_weights(model, weights, indices, safe_updates=True)

    # Expected shape: (1, 2, 3) where 3 is num_experts!
    # For token 0: index 2 -> 0.1. Index -1 -> mapped to 0 but weight 0.0!
    # So for expert 0: 0.0. Expert 1: 0.0. Expert 2: 0.1.
    # For token 1: index -1 -> mapped to 0 but weight 0.0! Index 1 -> 0.4.
    # So for expert 0: 0.0. Expert 1: 0.4. Expert 2: 0.0.
    expected_update_weights = jnp.array([[[0.0, 0.0, 0.1], [0.0, 0.4, 0.0]]])

    self.assertTrue(jax.numpy.allclose(update_weights, expected_update_weights, rtol=1e-5, atol=1e-5))

  def test_reshape_and_update_weights_duplicate_indices_use_add_not_set(self):
    """A regression test for the `.set()` -> `.add()` scatter-safety fix.

    Forced-routing replay can legitimately produce duplicate expert indices
    for the same token (e.g. two padding slots, which both remap to dummy
    index 0). `.set()` has undefined behavior for duplicate scatter indices,
    so we must use `.add()`. Since both duplicate slots carry a real,
    nonzero weight here, `.add()` and `.set()` produce different, checkable
    results: `.add()` sums to the total, `.set()` would keep only one.
    """
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    # Token 0 has expert index 1 selected twice with different weights.
    weights = jnp.array([[[0.3, 0.7]]])  # (1, 1, 2)
    indices = jnp.array([[[1, 1]]])  # (1, 1, 2)

    update_weights = moe.RoutedMoE.reshape_and_update_weights(model, weights, indices, safe_updates=True)

    # `.add()` must sum both contributions at expert index 1.
    expected_update_weights = jnp.array([[[0.0, 1.0, 0.0]]])
    self.assertTrue(jax.numpy.allclose(update_weights, expected_update_weights, rtol=1e-5, atol=1e-5))


class CheckForcedRoutingSupportTest(unittest.TestCase):
  """Regression tests for the decoder_block validation gate: forced routing

  is supported (scanned or not) only for a fixed set of decoder_blocks.
  """

  def test_supported_decoder_blocks_are_allowed(self):
    for decoder_block in (
        ctypes.DecoderBlockType.QWEN3_5,
        ctypes.DecoderBlockType.MIXTRAL,
        ctypes.DecoderBlockType.GEMMA4,
    ):
      check_forced_routing_support(decoder_block)  # must not raise

  def test_unsupported_decoder_blocks_raise(self):
    for decoder_block in (
        ctypes.DecoderBlockType.QWEN3_MOE,
        ctypes.DecoderBlockType.QWEN3_NEXT,
        ctypes.DecoderBlockType.LLAMA4,
        ctypes.DecoderBlockType.ENVY,
        ctypes.DecoderBlockType.DEEPSEEK,
        ctypes.DecoderBlockType.DEEPSEEK4,
        ctypes.DecoderBlockType.DEFAULT,
    ):
      with self.assertRaises(NotImplementedError):
        check_forced_routing_support(decoder_block)


class UnsupportedConfigGuardTest(unittest.TestCase):
  """Configurations forced routing rejects must fail loudly, not silently

  drop the replayed routing and train on normally-routed tokens.
  """

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    self.seq_len, self.batch_size, self.top_k = 8, 1, 2

  def _run(self, **overrides):
    """Builds a tiny model and applies it with forced routing enabled."""
    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        **_tiny_qwen35_kwargs(
            self.seq_len,
            self.batch_size,
            4,
            self.top_k,
            run_name="test_router_replay_guard",
            base_num_decoder_layers=1,
            num_decoder_layers=1,
            **overrides,
        ),
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    positions = jnp.arange(self.seq_len, dtype=jnp.int32)[None, :]
    segmentation = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    forced = jnp.zeros((self.batch_size, self.seq_len, self.top_k), dtype=jnp.int32)

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        ids,
        positions,
        segmentation,
        enable_dropout=False,
    )
    return model.apply(
        params,
        ids,
        positions,
        segmentation,
        enable_dropout=False,
        forced_routed_experts=forced,
    )

  def test_linen_decoder_rejects_forced_routing(self):
    with self.assertRaisesRegex(NotImplementedError, "pure-NNX decoder"):
      self._run(scan_layers=False, pure_nnx_decoder=False)

  def test_linen_decoder_without_forced_routing_still_works(self):
    """The guard must not break the Linen path when replay is unused."""
    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        **_tiny_qwen35_kwargs(
            self.seq_len,
            self.batch_size,
            4,
            self.top_k,
            run_name="test_router_replay_guard_linen_ok",
            base_num_decoder_layers=1,
            num_decoder_layers=1,
            scan_layers=False,
            pure_nnx_decoder=False,
        ),
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    positions = jnp.arange(self.seq_len, dtype=jnp.int32)[None, :]
    segmentation = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        ids,
        positions,
        segmentation,
        enable_dropout=False,
    )
    self.assertIsNotNone(params)

  def test_gemma4_scanned_rejects_forced_routing(self):
    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        override_model_config=True,
        model_name="gemma4-26b",
        num_experts=4,
        num_experts_per_tok=self.top_k,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=128,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        vocab_size=1000,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        per_device_batch_size=float(self.batch_size),
        weight_dtype="bfloat16",
        run_name="test_router_replay_guard_gemma4_scan",
        base_num_decoder_layers=2,
        num_decoder_layers=2,
        scan_layers=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    positions = jnp.arange(self.seq_len, dtype=jnp.int32)[None, :]
    segmentation = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    forced = jnp.zeros((self.batch_size, self.seq_len, self.top_k), dtype=jnp.int32)
    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        ids,
        positions,
        segmentation,
        enable_dropout=False,
    )
    with self.assertRaisesRegex(NotImplementedError, "GEMMA4"):
      model.apply(
          params,
          ids,
          positions,
          segmentation,
          enable_dropout=False,
          forced_routed_experts=forced,
      )


class ReshapeForcedRoutedExpertsForScanTest(unittest.TestCase):
  """Regression tests for the scan-xs reshape used to support forced routing

  with scan_layers=True (see check_forced_routing_support in configs/types.py
  for which decoder_blocks).
  """

  def test_4d_input_preserves_per_layer_values_in_scan_order(self):
    """Homogeneous-MoE case (e.g.

    QWEN3_5): every sub-layer in the cycle is MoE, so layers_per_cycle ==
    cycle_interval and num_layers == num_decoder_layers.
    """
    num_decoder_layers = 8
    cycle_interval = 4
    scan_length = num_decoder_layers // cycle_interval
    batch, seq, top_k = 2, 3, 2

    # [batch, seq, num_layers, top_k], where every entry for layer L is L.
    layer_ids = jnp.arange(num_decoder_layers, dtype=jnp.int32)
    forced_routed_experts = jnp.broadcast_to(layer_ids[None, None, :, None], (batch, seq, num_decoder_layers, top_k))

    scanned = reshape_forced_routed_experts_for_scan(
        forced_routed_experts,
        num_layers=num_decoder_layers,
        scan_length=scan_length,
        layers_per_cycle=cycle_interval,
    )

    self.assertEqual(scanned.shape, (scan_length, cycle_interval, batch, seq, top_k))
    # Layer L must land at scanned[L // cycle_interval, L % cycle_interval],
    # since jax.lax.scan slices axis 0 (scan_length) automatically per outer
    # iteration, then the ScannableBlock's static loop indexes axis 0 of the
    # remaining [cycle_interval, ...] chunk by sub-layer position.
    for layer in range(num_decoder_layers):
      cycle_idx, sub_idx = divmod(layer, cycle_interval)
      self.assertTrue(
          (scanned[cycle_idx, sub_idx] == layer).all(),
          f"layer {layer} landed in the wrong scan slot",
      )

  def test_3d_input_broadcasts_same_routing_to_every_layer(self):
    num_layers = 4
    layers_per_cycle = 4
    scan_length = num_layers // layers_per_cycle
    batch, seq, top_k = 1, 2, 2

    forced_routed_experts = jnp.array([[[1, 3], [0, 2]]])  # [batch, seq, top_k]

    scanned = reshape_forced_routed_experts_for_scan(
        forced_routed_experts,
        num_layers=num_layers,
        scan_length=scan_length,
        layers_per_cycle=layers_per_cycle,
    )

    self.assertEqual(scanned.shape, (scan_length, layers_per_cycle, batch, seq, top_k))
    for sub_idx in range(layers_per_cycle):
      self.assertTrue((scanned[0, sub_idx] == forced_routed_experts).all())

  def test_rejects_unexpected_ndim(self):
    """Only 3D ([batch, seq, top_k]) or 4D ([batch, seq, num_layers,

    top_k]) forced_routed_experts are valid; anything else (a caller bug,
    e.g. an unbatched or over-batched array) must raise, not silently
    misroute or crash deeper in the stack.
    """
    for bad_shape in ((4,), (2, 4), (2, 4, 8, 2, 1)):  # ndim 1, 2, 5
      with self.assertRaises(ValueError):
        reshape_forced_routed_experts_for_scan(
            jnp.zeros(bad_shape, dtype=jnp.int32),
            num_layers=4,
            scan_length=1,
            layers_per_cycle=4,
        )


class TrainerRouterReplayTest(unittest.TestCase):
  """Integration tests: forced routing threaded end-to-end through

  train.loss_fn, both unscanned and scanned, across every architecture that
  supports it (see check_forced_routing_support in configs/types.py).
  """

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _assert_forced_routing_loss_finite(self, cfg, seq_len, batch_size, forced_experts, label):
    """Builds a real model for `cfg`, runs train.loss_fn with a batch carrying

    `forced_experts`, and asserts the resulting loss is finite.
    """
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    rng = jax.random.PRNGKey(42)

    tokens = jnp.array(([10, 20, 30, 40] * ((seq_len // 4) + 1))[:seq_len], dtype=jnp.int32)
    inputs = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    positions = jnp.tile(
        jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0),
        (batch_size, 1),
    )
    segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    targets = jnp.roll(inputs, -1, axis=-1)

    data_batch = {
        "inputs": inputs,
        "inputs_position": positions,
        "inputs_segmentation": segmentation,
        "targets": targets,
        "targets_segmentation": segmentation,
        "forced_routed_experts": forced_experts,
    }

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    init_params_rng, init_dropout_rng = jax.random.split(rng)
    params = model.init(
        {"params": init_params_rng, "dropout": init_dropout_rng},
        inputs,
        positions,
        segmentation,
        enable_dropout=False,
    )

    loss, aux = train.loss_fn(
        model,
        cfg,
        data_batch,
        dropout_rng=init_dropout_rng,
        params=params,
        is_train=True,
    )

    self.assertIsNotNone(loss)
    self.assertFalse(jnp.isnan(loss), "Loss must not be NaN")
    print(f"\n[Trainer Router Replay][{label}] Computed loss with forced routing" f" + padding: {loss}")
    return loss, aux

  def _loss_for_routing(self, cfg, seq_len, batch_size, forced_experts):
    """train.loss_fn for one routing, reusing fixed params so losses compare."""
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    tokens = jnp.array(([10, 20, 30, 40] * ((seq_len // 4) + 1))[:seq_len], dtype=jnp.int32)
    inputs = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    positions = jnp.tile(
        jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0),
        (batch_size, 1),
    )
    segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(42))
    params = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        inputs,
        positions,
        segmentation,
        enable_dropout=False,
    )

    data_batch = {
        "inputs": inputs,
        "inputs_position": positions,
        "inputs_segmentation": segmentation,
        "targets": jnp.roll(inputs, -1, axis=-1),
        "targets_segmentation": segmentation,
    }
    if forced_experts is not None:
      data_batch["forced_routed_experts"] = forced_experts
    loss, _ = train.loss_fn(
        model,
        cfg,
        data_batch,
        dropout_rng=dropout_rng,
        params=params,
        is_train=True,
    )
    return float(loss)

  def _assert_routing_is_load_bearing(self, cfg, seq_len, batch_size, forced_a, forced_b, label):
    """Two different replays must give two different losses.

    Every "loss is finite" assertion in this class also passes when
    forced_routed_experts is silently dropped anywhere along the chain from
    train.loss_fn down to RoutedMoE.get_topk, because the model just routes
    normally. This is the assertion that actually pins the plumbing.
    """
    loss_a = self._loss_for_routing(cfg, seq_len, batch_size, forced_a)
    loss_b = self._loss_for_routing(cfg, seq_len, batch_size, forced_b)
    loss_none = self._loss_for_routing(cfg, seq_len, batch_size, None)
    print(f"\n[Router Replay][{label}] a={loss_a} b={loss_b} unforced={loss_none}")
    self.assertNotAlmostEqual(
        loss_a,
        loss_b,
        places=4,
        msg=f"[{label}] two different replays gave the same loss",
    )
    self.assertNotAlmostEqual(
        loss_a,
        loss_none,
        places=4,
        msg=f"[{label}] replay matched the unforced loss",
    )

  def test_loss_fn_with_forced_routed_experts(self):
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    cfg = _init_test_cfg(
        extra_args=["attention=flash"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts,
            top_k,
            run_name="test_trainer_router_replay",
            base_num_decoder_layers=1,
            num_decoder_layers=1,
            scan_layers=False,
        ),
    )

    # Synthetic forced routed experts: [batch, seq_len, top_k]
    forced_experts = jnp.zeros((batch_size, seq_len, top_k), dtype=jnp.int32)
    forced_experts = forced_experts.at[:, :, 0].set(1)
    forced_experts = forced_experts.at[:, :, 1].set(3)
    # Mark the last two tokens of every sequence as padding (-1) in every
    # expert slot, exercising the padding-mask/scatter-safety/NaN-guard code
    # paths end-to-end, not just in isolated unit tests.
    forced_experts = forced_experts.at[:, -2:, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "qwen3.5")

  def _qwen35_cfg(self, seq_len, batch_size, num_experts, top_k, run_name, **overrides):
    return _init_test_cfg(
        extra_args=["attention=dot_product"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts,
            top_k,
            run_name=run_name,
            weight_dtype="float32",
            dtype="float32",
            **overrides,
        ),
    )

  def test_different_replays_give_different_losses_unscanned(self):
    seq_len, batch_size, top_k, num_experts = 8, 1, 2, 4
    cfg = self._qwen35_cfg(
        seq_len,
        batch_size,
        num_experts,
        top_k,
        "test_router_replay_differential",
        base_num_decoder_layers=1,
        num_decoder_layers=1,
        scan_layers=False,
    )
    a = jnp.tile(jnp.array([0, 1], jnp.int32), (batch_size, seq_len, 1))
    b = jnp.tile(jnp.array([2, 3], jnp.int32), (batch_size, seq_len, 1))
    self._assert_routing_is_load_bearing(cfg, seq_len, batch_size, a, b, "qwen3.5 unscanned")

  def test_per_layer_routing_is_not_broadcast_unscanned(self):
    """4D per-layer replay must land layer L's routing on layer L.

    Reversing the layer axis has to change the loss; if the slicing collapsed
    to a broadcast of one layer, both orders would agree.
    """
    seq_len, batch_size, top_k, num_experts, layers = 8, 1, 2, 4, 2
    cfg = self._qwen35_cfg(
        seq_len,
        batch_size,
        num_experts,
        top_k,
        "test_router_replay_per_layer_unscanned",
        base_num_decoder_layers=layers,
        num_decoder_layers=layers,
        scan_layers=False,
    )
    per_layer = jnp.stack(
        [
            jnp.full((batch_size, seq_len, top_k), 0, jnp.int32),
            jnp.full((batch_size, seq_len, top_k), 3, jnp.int32),
        ],
        axis=2,
    )
    self._assert_routing_is_load_bearing(
        cfg,
        seq_len,
        batch_size,
        per_layer,
        per_layer[:, :, ::-1, :],
        "qwen3.5 unscanned per-layer",
    )

  def test_per_layer_routing_is_not_broadcast_scanned(self):
    """Same, through jax.lax.scan's xs -- the path that reshapes to

    [scan_length, layers_per_cycle, ...].
    """
    seq_len, batch_size, top_k, num_experts = 8, 1, 2, 4
    cycle_interval, layers = 2, 4
    cfg = self._qwen35_cfg(
        seq_len,
        batch_size,
        num_experts,
        top_k,
        "test_router_replay_per_layer_scanned",
        base_num_decoder_layers=layers,
        num_decoder_layers=layers,
        inhomogeneous_layer_cycle_interval=cycle_interval,
    )
    layer_ids = jnp.arange(layers, dtype=jnp.int32) % num_experts
    per_layer = jnp.broadcast_to(layer_ids[None, None, :, None], (batch_size, seq_len, layers, top_k))
    self._assert_routing_is_load_bearing(
        cfg,
        seq_len,
        batch_size,
        per_layer,
        per_layer[:, :, ::-1, :],
        "qwen3.5 scanned per-layer",
    )

  def test_per_layer_routing_reaches_the_scan_remainder_block(self):
    """Layers past the last whole cycle live in a separate remainder block.

    With 3 layers and a cycle of 2 the scan covers layers 0-1 and layer 2 is
    applied afterwards, so the remainder needs its own slice of the routing.
    """
    seq_len, batch_size, top_k, num_experts = 8, 1, 2, 4
    cycle_interval, layers = 2, 3
    cfg = self._qwen35_cfg(
        seq_len,
        batch_size,
        num_experts,
        top_k,
        "test_router_replay_scan_remainder",
        base_num_decoder_layers=layers,
        num_decoder_layers=layers,
        inhomogeneous_layer_cycle_interval=cycle_interval,
    )
    # Only the trailing (remainder) layer's routing differs between a and b, so
    # the losses can only diverge if the remainder block replays its own slice.
    shared = jnp.zeros((batch_size, seq_len, layers, top_k), dtype=jnp.int32)
    a = shared.at[:, :, -1, :].set(1)
    b = shared.at[:, :, -1, :].set(3)
    self._assert_routing_is_load_bearing(cfg, seq_len, batch_size, a, b, "qwen3.5 scan remainder")

  def test_loss_fn_with_forced_routed_experts_gemma4(self):
    """Gemma4 is supported unscanned only (its scanned path raises), and it is

    the one supported block that derives weights from softmaxed gate logits
    rather than the raw ones.
    """
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        override_model_config=True,
        model_name="gemma4-26b",
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=128,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        vocab_size=1000,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        per_device_batch_size=float(batch_size),
        weight_dtype="bfloat16",
        run_name="test_trainer_router_replay_gemma4",
        base_num_decoder_layers=1,
        num_decoder_layers=1,
        scan_layers=False,
    )

    forced_experts = jnp.zeros((batch_size, seq_len, top_k), dtype=jnp.int32)
    forced_experts = forced_experts.at[:, :, 0].set(1)
    forced_experts = forced_experts.at[:, :, 1].set(3)
    forced_experts = forced_experts.at[:, -2:, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "gemma4")

  def test_loss_fn_with_forced_routed_experts_scanned_qwen3_5(self):
    """Qwen3.5 supports forced routing together with `scan_layers=True` (see

    `check_forced_routing_support` in configs/types.py for the full list of
    supported decoder_blocks). This exercises that scanned path end-to-end:
    forced_routed_experts is threaded through jax.lax.scan's xs (one slice
    per layer) instead of being broadcast.
    """
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    cycle_interval = 4
    num_layers = 2 * cycle_interval  # 2 scan iterations of one cycle each.

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts,
            top_k,
            run_name="test_trainer_router_replay_scanned_qwen3_5",
            base_num_decoder_layers=num_layers,
            num_decoder_layers=num_layers,
            inhomogeneous_layer_cycle_interval=cycle_interval,
        ),
    )

    # Synthetic forced routed experts, one distinct value per layer:
    # [batch, seq_len, num_layers, top_k].
    layer_ids = jnp.arange(num_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned qwen3.5")

  def test_loss_fn_with_forced_routed_experts_scanned_mixtral(self):
    """Mixtral has no ScannableBlock wrapper: every scan iteration is exactly

    one (always-MoE) decoder layer, so this exercises the "no per-cycle
    nesting" branch of the scan wiring (layers_per_cycle=1, squeezed).
    """
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    num_layers = 4  # 4 scan iterations of 1 layer each (cycle_interval=1).

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        run_name="test_trainer_router_replay_scanned_mixtral",
        override_model_config=True,
        base_num_decoder_layers=num_layers,
        num_decoder_layers=num_layers,
        model_name="mixtral-8x7b",
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=256,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        vocab_size=1000,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        per_device_batch_size=float(batch_size),
        weight_dtype="bfloat16",
    )

    # One distinct forced routing per layer: [batch, seq_len, num_layers, top_k].
    layer_ids = jnp.arange(num_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned mixtral")

  def test_forced_routing_reproduces_unforced_output_when_all_experts_selected(
      self,
  ):
    """Replay identity: replaying the routing the model would have chosen must

    reproduce the unforced forward pass.

    Using num_experts_per_tok == num_experts makes the selected *set* the full
    expert set no matter what the gate scores, so `arange(num_experts)` is
    exactly what the router picks. The per-slot order differs, but the combine
    is a weighted sum over experts and so is order-invariant -- meaning forced
    and unforced must agree.
    """
    seq_len, batch_size, num_experts = 8, 2, 4
    top_k = num_experts

    def build_cfg(run_name):
      return _init_test_cfg(
          extra_args=["attention=dot_product"],
          **_tiny_qwen35_kwargs(
              seq_len,
              batch_size,
              num_experts,
              top_k,
              run_name=run_name,
              base_num_decoder_layers=1,
              num_decoder_layers=1,
              scan_layers=False,
              weight_dtype="float32",
              dtype="float32",
          ),
      )

    cfg = build_cfg("test_router_replay_identity")
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    tokens = jnp.array(([10, 20, 30, 40] * ((seq_len // 4) + 1))[:seq_len], dtype=jnp.int32)
    inputs = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    positions = jnp.tile(
        jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0),
        (batch_size, 1),
    )
    segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    init_params_rng, init_dropout_rng = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(
        {"params": init_params_rng, "dropout": init_dropout_rng},
        inputs,
        positions,
        segmentation,
        enable_dropout=False,
    )

    def logits_for(forced):
      return model.apply(
          params,
          inputs,
          positions,
          segmentation,
          enable_dropout=False,
          forced_routed_experts=forced,
      )

    all_experts = jnp.broadcast_to(
        jnp.arange(num_experts, dtype=jnp.int32)[None, None, :],
        (batch_size, seq_len, top_k),
    )

    unforced_logits = logits_for(None)
    forced_logits = logits_for(all_experts)

    self.assertEqual(unforced_logits.shape, forced_logits.shape)
    self.assertTrue(
        jnp.allclose(unforced_logits, forced_logits, rtol=2e-4, atol=2e-4),
        "Replaying the router's own choice must reproduce the unforced logits;"
        " max abs diff"
        f" {float(jnp.max(jnp.abs(unforced_logits - forced_logits)))}",
    )


class GetTopkUnforcedRegressionTest(unittest.TestCase):
  """Guards that adding forced routing did not change the *unforced* path.

  Both cases below regressed when the forced-routing branch was first wrapped
  around get_topk's body: `norm_topk_prob` escaped the non-DeepSeek `else`, and
  `per_expert_scale` became nested inside `if norm_topk_prob`.
  """

  def test_deepseek_scaling_is_not_renormalized_by_norm_topk_prob(self):
    # deepseek4-284b sets decoder_block=deepseek4, routed_score_func=sqrtsoftplus
    # and norm_topk_prob=true simultaneously. deepseek_scale_weights already
    # normalizes and then applies routed_scaling_factor, so norm_topk_prob must
    # NOT run again and cancel it out.
    config = DummyConfig(
        model_name="deepseek3-671b",
        decoder_block=ctypes.DecoderBlockType.DEEPSEEK4,
    )
    config.routed_score_func = "sqrtsoftplus"
    config.norm_topk_prob = True
    config.routed_scaling_factor = 2.5
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 0.5]]])
    top_k_weights, _ = moe.RoutedMoE.get_topk(model, gate_logits, gate_logits)

    self.assertAlmostEqual(
        float(jnp.sum(top_k_weights)),
        config.routed_scaling_factor,
        places=4,
        msg=("DeepSeek weights must keep routed_scaling_factor; norm_topk_prob" " must not re-normalize them."),
    )

  def test_per_expert_scale_applies_when_norm_topk_prob_is_false(self):
    # per_expert_scale only exists for GEMMA4. It is independent of
    # norm_topk_prob and must be applied either way.
    scale = [10.0, 20.0, 30.0]
    config = DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4)
    config.norm_topk_prob = False
    model = DummyRoutedMoE(config, per_expert_scale=scale)

    gate_logits = jnp.array([[[1.0, 2.0, 0.5]]])
    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(model, gate_logits, gate_logits)

    unscaled_model = DummyRoutedMoE(DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4))
    unscaled_weights, _ = moe.RoutedMoE.get_topk(unscaled_model, gate_logits, gate_logits)
    expected = unscaled_weights * jnp.take_along_axis(jnp.asarray(scale)[None, None, :], top_k_indices, axis=-1)

    self.assertTrue(
        jnp.allclose(top_k_weights, expected, rtol=1e-5, atol=1e-5),
        "per_expert_scale must be applied even when norm_topk_prob is False.",
    )

  def test_per_expert_scale_applies_when_norm_topk_prob_is_true(self):
    scale = [10.0, 20.0, 30.0]
    config = DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4)
    config.norm_topk_prob = True
    model = DummyRoutedMoE(config, per_expert_scale=scale)

    gate_logits = jnp.array([[[1.0, 2.0, 0.5]]])
    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(model, gate_logits, gate_logits)

    # Normalization happens before scaling, so dividing the scale back out must
    # leave weights summing to 1.
    applied = jnp.take_along_axis(jnp.asarray(scale)[None, None, :], top_k_indices, axis=-1)
    self.assertAlmostEqual(float(jnp.sum(top_k_weights / applied)), 1.0, places=4)


class PaddingExcludedFromSoftmaxTest(unittest.TestCase):
  """Padding must not enter the top-k softmax denominator.

  Gathering with a -1 index wraps to the last expert, and softmaxing that
  value before masking rescales the *real* slots. With norm_topk_prob the
  renormalization happens to cancel it, but Mixtral is supported and defaults
  to norm_topk_prob=false, so the error survives there.
  """

  def _weights(
      self,
      forced,
      norm_topk_prob,
      decoder_block=ctypes.DecoderBlockType.MIXTRAL,
  ):
    """Computes top-k routing weights with forced routed experts."""
    config = DummyConfig(model_name="mixtral-8x7b", decoder_block=decoder_block)
    config.norm_topk_prob = norm_topk_prob
    config.routed_scaling_factor = 1.0
    # Expert 2 has a large logit; it is the one -1 wraps onto.
    gate_logits = jnp.array([[[0.0, 1.0, 5.0]]])
    weights, _ = moe.RoutedMoE.get_topk(
        DummyRoutedMoE(config),
        gate_logits,
        gate_logits,
        forced_routed_experts=forced,
    )
    return weights

  def test_single_real_expert_keeps_full_weight(self):
    # One real slot (expert 1) and one padding slot. The real slot is the only
    # thing routed to, so after softmax it must carry all the weight.
    weights = self._weights(jnp.array([[[1, -1]]]), norm_topk_prob=False)
    self.assertAlmostEqual(float(weights[0, 0, 0]), 1.0, places=5)
    self.assertEqual(float(weights[0, 0, 1]), 0.0)

  def test_fully_padded_token_is_zero_not_nan(self):
    weights = self._weights(jnp.array([[[-1, -1]]]), norm_topk_prob=False)
    self.assertFalse(bool(jnp.any(jnp.isnan(weights))))
    self.assertEqual(float(jnp.sum(weights)), 0.0)

  def test_unpadded_forced_routing_is_unaffected(self):
    """The masking must not perturb the no-padding case."""
    weights = self._weights(jnp.array([[[1, 2]]]), norm_topk_prob=False)
    expected = jax.nn.softmax(jnp.array([1.0, 5.0]))
    self.assertTrue(jnp.allclose(weights[0, 0], expected, rtol=1e-5, atol=1e-5))


class ForcedRoutingGradientTest(unittest.TestCase):
  """Replay must not cut the gradient to the router.

  The weights are still derived from gate_logits, so the gate must keep
  learning under replay. A stray stop_gradient or an over-broad mask would
  silently freeze the router and break RL training while every forward-only
  test stayed green.
  """

  def test_gate_logits_receive_gradient_under_forced_routing(self):
    config = DummyConfig(model_name="mixtral-8x7b", decoder_block=ctypes.DecoderBlockType.MIXTRAL)
    config.routed_scaling_factor = 1.0
    model = DummyRoutedMoE(config)
    forced = jnp.array([[[1, -1], [0, 2]]])

    def loss(gate_logits):
      weights, _ = moe.RoutedMoE.get_topk(model, gate_logits, gate_logits, forced_routed_experts=forced)
      return jnp.sum(weights**2)

    grads = jax.grad(loss)(jnp.array([[[0.0, 1.0, 5.0], [2.0, 0.5, 1.0]]]))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))
    self.assertGreater(float(jnp.max(jnp.abs(grads))), 0.0)

  def test_fully_padded_token_has_zero_gradient(self):
    config = DummyConfig(model_name="mixtral-8x7b", decoder_block=ctypes.DecoderBlockType.MIXTRAL)
    config.routed_scaling_factor = 1.0
    model = DummyRoutedMoE(config)

    def loss(gate_logits):
      weights, _ = moe.RoutedMoE.get_topk(
          model,
          gate_logits,
          gate_logits,
          forced_routed_experts=jnp.array([[[-1, -1]]]),
      )
      return jnp.sum(weights**2)

    grads = jax.grad(loss)(jnp.array([[[0.0, 1.0, 5.0]]]))
    self.assertTrue(bool(jnp.all(grads == 0.0)))


class OutOfRangeExpertIdTest(unittest.TestCase):
  """An expert id outside [0, num_experts) must behave like an unused slot.

  Before this was masked, take_along_axis clamped the gather while the scatter
  in reshape_and_update_weights dropped the index, producing a NaN loss that
  silently poisoned the whole accumulated gradient.
  """

  def _model(self):
    config = DummyConfig(model_name="mixtral-8x7b", decoder_block=ctypes.DecoderBlockType.MIXTRAL)
    config.routed_scaling_factor = 1.0
    model = DummyRoutedMoE(config)
    model.num_experts = 4
    return model

  def test_out_of_range_id_is_masked_not_nan(self):
    gate_logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])
    weights, _ = moe.RoutedMoE.get_topk(
        self._model(),
        gate_logits,
        gate_logits,
        forced_routed_experts=jnp.array([[[1, 9]]]),
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(weights))))
    # The one valid slot keeps all the weight, exactly as for a -1 slot.
    self.assertAlmostEqual(float(weights[0, 0, 0]), 1.0, places=5)
    self.assertEqual(float(weights[0, 0, 1]), 0.0)

  def test_out_of_range_matches_negative_sentinel(self):
    gate_logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])
    oob, _ = moe.RoutedMoE.get_topk(
        self._model(),
        gate_logits,
        gate_logits,
        forced_routed_experts=jnp.array([[[1, 9]]]),
    )
    pad, _ = moe.RoutedMoE.get_topk(
        self._model(),
        gate_logits,
        gate_logits,
        forced_routed_experts=jnp.array([[[1, -1]]]),
    )
    self.assertTrue(jnp.allclose(oob, pad))


class LoadBalanceUpdatePaddingTest(unittest.TestCase):
  """Padding must not be attributed to expert 0.

  jnp.bincount clips out-of-range values, so the -1 sentinel used to land on
  expert 0 and skew the routed-bias update.
  """

  def test_padding_does_not_change_the_update(self):
    with_padding = moe.calculate_load_balance_updates(jnp.array([[[-1, -1], [0, 2]]]), 3, 0.1)
    without_padding = moe.calculate_load_balance_updates(jnp.array([[[0, 2]]]), 3, 0.1)
    self.assertTrue(jnp.array_equal(with_padding, without_padding))


class GetTopkForcedRoutingParityTest(unittest.TestCase):
  """Replaying the indices the unforced path selected must give the same weights."""

  def _assert_parity(self, config):
    """Asserts replaying the unforced indices reproduces the unforced weights."""
    model = DummyRoutedMoE(config)
    gate_logits = jnp.array([[[1.0, 2.0, 0.5], [0.25, -1.0, 3.0]]])

    unforced_weights, unforced_indices = moe.RoutedMoE.get_topk(model, gate_logits, gate_logits)
    forced_weights, forced_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, gate_logits, forced_routed_experts=unforced_indices
    )

    self.assertTrue(jnp.array_equal(unforced_indices, forced_indices))
    self.assertTrue(
        jnp.allclose(unforced_weights, forced_weights, rtol=1e-5, atol=1e-5),
        f"forced weights {forced_weights} != unforced {unforced_weights}",
    )

  def test_parity_default_block(self):
    self._assert_parity(DummyConfig())

  def test_parity_default_block_with_norm_topk_prob(self):
    config = DummyConfig()
    config.norm_topk_prob = True
    self._assert_parity(config)

  def test_parity_gemma4(self):
    self._assert_parity(DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4))


class RaggedSortGuardTest(unittest.TestCase):
  """permute's standard (non-ragged) branch under forced routing.

  The ragged branch is supported too; see
  RaggedSortForcedRoutingEquivalenceTest for that one (tpu_only).
  """

  def _permute_config(self, use_ragged_sort, use_ring_of_experts):
    """Builds a DummyConfig with the attributes permute() reads."""
    config = DummyConfig()
    config.load_balance_loss_weight = 0.0
    config.routed_bias = False
    config.routed_bias_update_rate = 0.0
    config.num_experts = 3
    config.use_ragged_sort = use_ragged_sort
    config.use_ring_of_experts = use_ring_of_experts
    config.moe_use_direct_token_gather = False
    config.ragged_buffer_factor = 0.0
    return config

  def _make_model(self, config):
    model = DummyRoutedMoE(config)
    model.get_expert_parallelism_size = lambda: 1
    model.should_update_load_balance = lambda: False
    return model

  def _run_permute(self, model, forced_routed_experts):
    inputs = jnp.ones((1, 2, 4), dtype=jnp.float32)
    gate_logits = jnp.array([[[1.0, 2.0, 0.5], [0.25, -1.0, 3.0]]])
    return moe.RoutedMoE.permute(
        model,
        inputs,
        gate_logits,
        gate_logits,
        forced_routed_experts=forced_routed_experts,
    )

  def test_forced_routing_without_ragged_sort_is_allowed(self):
    model = self._make_model(self._permute_config(use_ragged_sort=False, use_ring_of_experts=True))
    forced = jnp.array([[[0, 1], [2, -1]]], dtype=jnp.int32)
    sorted_inputs = self._run_permute(model, forced)[0]
    self.assertEqual(sorted_inputs.shape[-1], 4)


class RaggedSortPaddingSentinelTest(unittest.TestCase):
  """Documents the ring_ragged_sort contract the -1 -> num_experts remap needs.

  NOTE: this asserts properties of the kernel expressions reproduced here; it
  does NOT execute maxtext's remap. Only the tpu_only equivalence test does.

  ring_ragged_sort orders slots with argsort but counts them with one_hot, and
  derives each shard's [start, end) window from a cumsum of those counts. -1
  satisfies only the counting half -- it sorts *before* expert 0 -- which
  shifts every window by the number of padding slots. num_experts satisfies
  both. These assertions mirror the kernel's own expressions
  (kernels/ragged/ragged_sort.py) so they fail if it stops holding.
  """

  NUM_EXPERTS = 3

  def _counts_and_order(self, flat_indices):
    counts = jax.nn.one_hot(flat_indices, self.NUM_EXPERTS, dtype=jnp.int32).sum(axis=0)
    return counts, jnp.argsort(flat_indices)

  def test_num_experts_sentinel_is_excluded_from_counts_and_sorts_last(self):
    real = jnp.array([0, 2, 1, 0], dtype=jnp.int32)
    padded = jnp.array([0, -1, 2, 1, -1, 0], dtype=jnp.int32)
    sentinel = jnp.where(padded < 0, self.NUM_EXPERTS, padded)

    counts, order = self._counts_and_order(sentinel)
    # Padding contributes nothing, so the counts equal the real-only counts.
    self.assertEqual(counts.tolist(), self._counts_and_order(real)[0].tolist())

    # Everything past group_offsets[num_experts] is padding, and nothing before
    # it is -- so no shard window can reach a padded slot.
    total_real = int(jnp.cumulative_sum(counts, include_initial=True)[self.NUM_EXPERTS])
    self.assertEqual(total_real, int(jnp.sum(padded >= 0)))
    reordered = padded[order]
    self.assertTrue(bool(jnp.all(reordered[:total_real] >= 0)))
    self.assertTrue(bool(jnp.all(reordered[total_real:] < 0)))

  def test_raw_negative_sentinel_would_desynchronize_the_windows(self):
    """Guards the reason for the remap: -1 breaks the offsets."""
    padded = jnp.array([0, -1, 2, 1, -1, 0], dtype=jnp.int32)
    counts, order = self._counts_and_order(padded)
    total_real = int(jnp.cumulative_sum(counts, include_initial=True)[self.NUM_EXPERTS])
    # The count is right, but padding sorts first, so the first `total_real`
    # rows are not the real ones.
    self.assertEqual(total_real, int(jnp.sum(padded >= 0)))
    self.assertFalse(bool(jnp.all(padded[order][:total_real] >= 0)))


class RaggedSortForcedRoutingEquivalenceTest(unittest.TestCase):
  """Forced routing must give the same MoE output with and without the ragged

  sort path. TPU-only: the ragged kernels require TPU, and use_ragged_sort
  additionally requires expert parallelism > 1.

  Mirrors moe_test.py::_run_ragged_sort_loss_and_grad, but drives both runs
  with a forced routing that contains -1 padding -- the case that desynchronized
  ring_ragged_sort's per-shard windows before the sentinel remap.
  """

  def _build_cfg(self, use_ragged_sort):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"router_replay_ragged_{use_ragged_sort}_test",
        enable_checkpointing=False,
        log_config=False,
        skip_jax_distributed_system=True,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=7168,  # multiple of 1024 so the kernel is fully used
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        max_target_length=128,
        float32_gate_logits=True,
        use_ragged_sort=use_ragged_sort,
        ragged_buffer_factor=-1.0,
    )

  @staticmethod
  def _build_model(cfg, mesh):
    return moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

  @pytest.mark.tpu_only
  def test_ragged_sort_matches_non_ragged_under_forced_routing(self):
    rng_model, rng_hidden = jax.random.split(jax.random.PRNGKey(2345))
    device_count = jax.device_count()

    cfg_ref = self._build_cfg(use_ragged_sort=False)
    tokens = int(cfg_ref.per_device_batch_size) * device_count
    hidden_states = jax.random.uniform(
        rng_hidden,
        (tokens, cfg_ref.max_target_length, cfg_ref.base_emb_dim),
        dtype=cfg_ref.dtype,
    )

    # Replay a fixed routing, with the tail of every sequence marked unused so
    # the -1 padding path is exercised on both sides.
    top_k = cfg_ref.num_experts_per_tok
    forced = jnp.zeros((tokens, cfg_ref.max_target_length, top_k), dtype=jnp.int32)
    forced = forced.at[:, :, 0].set(1)
    if top_k > 1:
      forced = forced.at[:, :, 1].set(3)
    forced = forced.at[:, -8:, :].set(-1)

    def run(cfg, variables=None):
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = self._build_model(cfg, mesh)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        if variables is None:
          variables = model.init(
              {"params": rng_model, "dropout": rng_model},
              hidden_states,
              forced_routed_experts=forced,
          )
        out, _, _ = model.apply(
            {"params": variables["params"]},
            hidden_states,
            forced_routed_experts=forced,
        )
      return out, variables

    out_ref, variables = run(cfg_ref)
    out_ragged, _ = run(self._build_cfg(use_ragged_sort=True), variables)

    self.assertTrue(
        jnp.allclose(
            out_ragged.astype(jnp.float32),
            out_ref.astype(jnp.float32),
            rtol=1e-2,
            atol=1e-2,
        ),
        msg=(
            "ragged sort output diverges from the non-ragged path under forced "
            "routing; max abs diff "
            f"{float(jnp.max(jnp.abs(out_ragged.astype(jnp.float32) - out_ref.astype(jnp.float32))))}"
        ),
    )


if __name__ == "__main__":
  unittest.main()
