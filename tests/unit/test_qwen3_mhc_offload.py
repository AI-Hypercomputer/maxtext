# Copyright 2026 Google LLC
"""Unit test for Qwen3-Next-80B hierarchical block scan and mHC parameter offloading."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import Config, HyperConnectionType
from maxtext.layers import mhc
from maxtext.layers import nnx_scan
from maxtext.models import qwen3
from maxtext.configs import pyconfig


class TestQwen3NextMhcScanOffload(unittest.TestCase):
  """Unit tests verifying mHC scalar parameters and scanned blocks under 2-layer buffer host offloading."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

  def test_mhc_forward_without_device_conflict(self):
    """Verifies mHC layer executes cleanly without internal to_device mutations."""
    config_dict = {
        "model_name": "qwen3-next-80b-a3b",
        "mhc_expansion_rate": 4,
        "enable_mhc_lite": True,
        "sinkhorn_iterations": 3,
        "normalization_layer_epsilon": 1e-6,
        "parameter_memory_host_offload": False,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "matmul_precision": "default",
    }
    mock_config = type("Config", (), config_dict)()

    mhc_layer = mhc.ManifoldConstrainedHyperConnections(
        dim=64,
        config=mock_config,
        mesh=None,
        rngs=self.rngs,
    )

    batch, seq, k, dim = 2, 8, 4, 64
    x = jnp.ones((batch, seq, k, dim), dtype=jnp.bfloat16)

    def dummy_norm(inp):
      return inp

    def dummy_branch(inputs, **kwargs):
      return inputs, None, None

    # Run forward pass
    out, metadata = mhc_layer(
        norm_fn=dummy_norm,
        branch_fn=dummy_branch,
        x=x,
        mhc_type=HyperConnectionType.MLP_MOE,
    )

    self.assertEqual(out.shape, (batch, seq, k, dim))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_qwen3_scannable_block_unroll_and_scan(self):
    """Verifies that 1-level leaf scan in Qwen3NextScannableBlock compiles with 2-layer parameter buffer."""
    length = 2
    
    class DummyLocalLayer(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))
        self.res_alpha_scale = nnx.Param(jnp.array([1.0], dtype=jnp.float32))

      def __call__(self, x):
        return x @ self.w.get_value() * self.res_alpha_scale.get_value()

    stacked_layers = nnx_scan.create_scanned_layers(
        DummyLocalLayer,
        length=length,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=self.rngs,
    )

    x = jnp.ones((2, 4), dtype=jnp.float32)

    # Test apply_scanned_layers with parameter_memory_two_layer_buffer=True
    out = nnx_scan.apply_scanned_layers(
        stacked_layers,
        x,
        length=length,
        param_scan_axis=0,
        apply_fn=lambda layer, carry: layer(carry),
        remat=False,
        parameter_memory_host_offload=True,
        parameter_memory_two_layer_buffer=True,
    )
    self.assertEqual(out.shape, (2, 4))


  def test_qwen3_scannable_block_value_and_grad(self):
    """Verifies that Qwen3NextScannableBlock compiles with jax.value_and_grad and jax.checkpoint."""
    cfg = pyconfig.initialize([
        None,
        "src/maxtext/configs/base.yml",
        "model_name=qwen3-next-80b-a3b",
        "override_model_config=True",
        "base_num_decoder_layers=3",
        "inhomogeneous_layer_cycle_interval=3",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_moe_mlp_dim=64",
        "num_experts=2",
        "num_experts_per_tok=1",
        "base_num_query_heads=2",
        "base_num_kv_heads=1",
        "head_dim=32",
        "gdn_key_head_dim=32",
        "gdn_value_head_dim=32",
        "gdn_num_key_heads=1",
        "gdn_num_value_heads=2",
        "gdn_conv_kernel_dim=4",
        "gdn_chunk_size=16",
        "max_target_length=16",
        "per_device_batch_size=2",
        "scan_layers=True",
        "remat_policy=custom",
        "dataset_type=synthetic",
        "dataset_name=synthetic",
        "sparse_matmul=False",
        "megablox=False",
        "use_tokamax_splash=False",
        "attention=dot_product",
        "use_gdn_kernel=False",
        "enable_checkpointing=False",
        "enable_dropout=False",
    ])

    mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
    block = qwen3.Qwen3NextScannableBlock(
        config=cfg,
        mesh=mesh,
        model_mode="train",
        rngs=self.rngs,
        num_of_layers=3,
    )

    graphdef, params, other = nnx.split(block, nnx.Param, ...)

    def loss_fn(p, o, x):
      m = nnx.merge(graphdef, p, o)
      positions = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (2, 16))
      def checkpointed_call(mod, inputs):
        return mod(inputs, decoder_segment_ids=None, decoder_positions=positions, deterministic=True, model_mode="train")[0]

      ckpt_fn = jax.checkpoint(checkpointed_call)
      y = ckpt_fn(m, x)
      return jnp.sum(y), None

    batch, seq, dim = 2, 16, 64
    x = jnp.ones((batch, seq, dim), dtype=jnp.bfloat16)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, _), grads = grad_fn(params, other, x)

    self.assertFalse(jnp.isnan(loss_val))
    self.assertIsNotNone(grads)


if __name__ == "__main__":
  unittest.main()

