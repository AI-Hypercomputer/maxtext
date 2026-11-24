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

"""Pure-JAX PyTorch-free unit tests for DeepSeek-V4.1 multimodal MaxText modules."""

from __future__ import annotations

import functools
import os
import re
import sys
import tempfile
import types as py_types
from typing import Any
import unittest

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN, MultimodalInput, VisionEncoderBlockType
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.layers import encoders
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.models import deepseek41_vision
from maxtext.multimodal import processor as mm_processor
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import gradient_accumulation
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.deepseek_v41.parity import e2e_check
from tests.utils.deepseek_v41.parity import layer_parity
from tests.utils.deepseek_v41.parity import negctl

_VALID_FULL_LEN = 36
_DOC1_LEN = 16
_DOC2_LEN = 16
_ORIG_MATMUL_PRECISION = None


def setUpModule() -> None:
  """Enforces IEEE-754 float32 matmul precision for deterministic CPU parity assertions."""
  global _ORIG_MATMUL_PRECISION  # pylint: disable=global-statement
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")


def tearDownModule() -> None:
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


def eval_multimodal_packed_and_full_logits(
    model_graphdef: Any,
    param_state: nnx.State,
    param_pure: dict[str, Any],
    rest_state: nnx.State,
    mesh: jax.sharding.Mesh,
    *,
    full_tokens: jax.Array,
    full_pos: jax.Array,
    full_seg: jax.Array,
    full_patches: jax.Array,
    packed_tokens: jax.Array,
    packed_pos: jax.Array,
    packed_seg: jax.Array,
    packed_patches: jax.Array,
) -> dict[str, np.ndarray]:
  """Evaluates full valid logits and packed `[seg=1, seg=2, pad=0]` document logits."""
  local_model = mm_testing.apply_with_pure_params(model_graphdef, param_state, param_pure, rest_state)
  with mesh:
    logits_full = local_model(
        decoder_input_tokens=full_tokens,
        decoder_positions=full_pos,
        decoder_segment_ids=full_seg,
        encoder_images=full_patches,
        enable_dropout=False,
    )
    logits_packed = local_model(
        decoder_input_tokens=packed_tokens,
        decoder_positions=packed_pos,
        decoder_segment_ids=packed_seg,
        encoder_images=packed_patches,
        enable_dropout=False,
    )
  return {
      "full_valid": np.asarray(logits_full[:, :_VALID_FULL_LEN, :], dtype=np.float64),
      "packed_doc1": np.asarray(logits_packed[:, :_DOC1_LEN, :], dtype=np.float64),
      "packed_doc2": np.asarray(logits_packed[:, _DOC1_LEN : _DOC1_LEN + _DOC2_LEN, :], dtype=np.float64),
  }


@pytest.mark.cpu_only
class DeepSeek41InitAndConfigTest(unittest.TestCase):
  """Config defaults, YAML/RL validation, FSDP sharding, activation checkpointing, and optimizer masking."""

  def test_get_shaped_batch_and_int32_dummy_init_and_forward(self) -> None:
    mesh = mm_testing.single_device_mesh()
    for grid, expected_patches, expected_span_len in (
        (mm_testing.MINI_GRID_PADDED, 35, 11),
        (mm_testing.MINI_GRID_EXACT, 54, 10),
        ((39, 39), 1521, 13 * (13 + 1) + 2),
    ):
      cfg = mm_testing.mini_vision_maxtext_config(grid=grid, dtype_mm="bfloat16")
      shaped = maxtext_utils.get_shaped_batch(cfg)
      batch_size = int(cfg.micro_batch_size_to_train_on)
      self.assertIn("images", shaped)
      self.assertNotIn("image_masks", shaped)
      self.assertEqual(shaped["images"].shape, (batch_size, 1, expected_patches, 588))
      self.assertEqual(shaped["images"].dtype, jnp.float32)

      cfg_tfds = mm_testing.mini_vision_maxtext_config(grid=grid, dtype_mm="bfloat16", dataset_type="tfds")
      shaped_tfds = maxtext_utils.get_shaped_batch(cfg_tfds)
      self.assertEqual(shaped_tfds["images"].shape, (batch_size, 1, expected_patches, 588))
      self.assertEqual(shaped_tfds["images"].dtype, jnp.float32)

      if grid != (39, 39):
        dummy_images = jnp.ones(shaped["images"].shape, dtype=jnp.int32)
        with mesh:
          wrapper = encoders.VisionEncoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0))
          spans_nnx, deep_feats_nnx = jax.jit(lambda imgs, w=wrapper: w(imgs, deterministic=True))(dummy_images)
          self.assertIsNone(deep_feats_nnx)
          self.assertEqual(spans_nnx.shape, (batch_size, 1, expected_span_len, cfg.emb_dim))
          self.assertEqual(spans_nnx.dtype, jnp.bfloat16)
          self.assertTrue(bool(jnp.all(jnp.isfinite(spans_nnx))))

          linen_enc = nnx_wrappers.ToLinen(
              encoders.VisionEncoder,
              args=(),
              kwargs={"config": cfg, "mesh": mesh},
              metadata_fn=initializers.variable_to_logically_partitioned,
          )
          variables = jax.jit(linen_enc.init)(jax.random.PRNGKey(0), dummy_images, deterministic=True)
          spans_linen, deep_feats_linen = jax.jit(linen_enc.apply)(variables, dummy_images, deterministic=True)
          self.assertIsNone(deep_feats_linen)
          self.assertEqual(spans_linen.shape, (batch_size, 1, expected_span_len, cfg.emb_dim))
          self.assertEqual(spans_linen.dtype, jnp.bfloat16)
          self.assertTrue(bool(jnp.all(jnp.isfinite(spans_linen))))

    # Positive match with config passed, paired with negative assertion when config is omitted.
    cfg_pos = mm_testing.mini_vision_maxtext_config(grid=mm_testing.MINI_GRID_PADDED)
    self.assertEqual(
        mm_processor.get_dummy_image_shape_for_init("deepseek4.1-flash", batch_size=2, config=cfg_pos),
        (2, 1, 35, 588),
    )
    with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
      mm_processor.get_dummy_image_shape_for_init("deepseek4.1-flash", batch_size=2, config=None)

  def test_yaml_configs_and_validation_and_decoder_boundary(self) -> None:
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg_text = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=ds41_yaml_text_check",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertFalse(cfg_text.use_multimodal)
      self.assertEqual(cfg_text.vision_encoder_block, VisionEncoderBlockType.DEEPSEEK41)
      self.assertEqual(cfg_text.num_hidden_layers_for_vit, 32)
      self.assertEqual(cfg_text.hidden_size_for_vit, 1024)
      self.assertEqual(cfg_text.num_attention_heads_for_vit, 16)
      self.assertEqual(cfg_text.intermediate_size_for_vit, 2816)
      self.assertEqual(cfg_text.patch_size_for_vit, 14)
      self.assertEqual(cfg_text.rope_theta_for_vit, 10000)
      self.assertEqual(cfg_text.num_channels_for_vit, 3)
      self.assertEqual(cfg_text.ds41_vision_downsample_ratio, 3)
      self.assertEqual(list(cfg_text.ds41_vision_grid), [39, 39])
      self.assertEqual(cfg_text.dtype_mm, "bfloat16")
      self.assertTrue(cfg_text.freeze_vision_encoder_params)

      cfg_mm = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=ds41_yaml_mm_check",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
              "use_multimodal=true",
          ]
      )
      self.assertTrue(cfg_mm.use_multimodal)
      self.assertEqual(cfg_mm.vision_encoder_block, VisionEncoderBlockType.DEEPSEEK41)

    # Production 1024-wide PatchEmbed kernel shape check: [588, 1024].
    mesh = mm_testing.single_device_mesh()
    cfg_prod_vit = mm_testing.mini_vision_maxtext_config(
        num_hidden_layers_for_vit=1,
        hidden_size_for_vit=1024,
        num_attention_heads_for_vit=16,
        intermediate_size_for_vit=2816,
    )
    with mesh:
      prod_vit = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg_prod_vit, rngs=nnx.Rngs(params=0))
      self.assertEqual(prod_vit.patch_embed.kernel[...].shape, (588, 1024))

    # Validation: multimodal DeepSeek-V4.1 rejects vision_encoder_block != 'deepseek41' (both explicit and default 'none').
    with self.assertRaisesRegex(ValueError, "vision_encoder_block='deepseek41'"):
      mm_testing.mini_vision_maxtext_config(use_multimodal=True, vision_encoder_block="gemma3")
    with self.assertRaisesRegex(ValueError, "vision_encoder_block='deepseek41'"):
      pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              "model_name=default",
              "decoder_block=deepseek41",
              "run_name=ds41_default_missing_vision_block",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
              "use_multimodal=true",
          ]
      )

    # Validation: ds41_vision_grid must be two positive integers.
    for bad_grid in ([39], [0, 39], [39, -1]):
      with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
        mm_testing.mini_vision_maxtext_config(ds41_vision_grid=bad_grid)

    # Text-only full Transformer has no vision_encoder params; multimodal Transformer executes end-to-end.
    cfg_mini_text = mm_testing.mini_vision_maxtext_config(
        use_multimodal=False,
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        ds41_engram_compressed_vocab_size=0,
    )
    model_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_mini_text), cfg_mini_text.mesh_axes)
    model_text = model_creation_utils.from_pretrained(cfg_mini_text, mesh=model_mesh, model_mode=MODEL_MODE_TRAIN)
    self.assertFalse(hasattr(model_text, "vision_encoder") and model_text.vision_encoder is not None)

    cfg_mini_mm = mm_testing.mini_vision_maxtext_config(
        use_multimodal=True,
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        ds41_engram_compressed_vocab_size=0,
    )
    model_mm = model_creation_utils.from_pretrained(cfg_mini_mm, mesh=model_mesh, model_mode=MODEL_MODE_TRAIN)
    self.assertIsNotNone(model_mm.vision_encoder)
    batch_size = int(cfg_mini_mm.micro_batch_size_to_train_on)
    seq_len = int(cfg_mini_mm.max_target_length)
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    tokens = tokens.at[:, 2:13].set(cfg_mini_mm.ds41_image_token_id)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    dummy_imgs = jnp.ones((batch_size, 1, 35, 588), dtype=jnp.float32)
    with model_mesh:
      logits = model_mm(tokens, positions, encoder_images=dummy_imgs, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(logits.shape, (batch_size, seq_len, cfg_mini_mm.vocab_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))

  def test_vision_encoder_remat_when_unfrozen_and_rlconfig_validation(self) -> None:
    """Verifies ViT block checkpointing when unfrozen and RLConfig multimodal DS4.1 validation."""
    cfg_remat = mm_testing.mini_vision_maxtext_config(
        use_multimodal=True,
        freeze_vision_encoder_params=False,
        remat_policy="full",
        dtype_mm="float32",
        weight_dtype="float32",
    )
    cfg_noremat = mm_testing.mini_vision_maxtext_config(
        use_multimodal=True,
        freeze_vision_encoder_params=False,
        remat_policy="none",
        dtype_mm="float32",
        weight_dtype="float32",
    )
    vit_remat = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg_remat, rngs=nnx.Rngs(params=7))
    vit_noremat = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg_noremat, rngs=nnx.Rngs(params=7))

    gdef_r, params_r, other_r = nnx.split(vit_remat, nnx.Param, nnx.Not(nnx.Param))
    gdef_n, params_n, other_n = nnx.split(vit_noremat, nnx.Param, nnx.Not(nnx.Param))
    imgs = jax.random.normal(jax.random.PRNGKey(11), (1, 1, 35, 588), dtype=jnp.float32)

    def _loss_r(ps, x):
      return jnp.mean(jnp.square(nnx.merge(gdef_r, ps, other_r, copy=True)(x)))

    def _loss_n(ps, x):
      return jnp.mean(jnp.square(nnx.merge(gdef_n, ps, other_n, copy=True)(x)))

    vg_r = jax.jit(jax.value_and_grad(_loss_r))
    vg_n = jax.jit(jax.value_and_grad(_loss_n))
    self.assertIn("remat2", str(jax.make_jaxpr(vg_r)(params_r, imgs)))
    self.assertNotIn("remat2", str(jax.make_jaxpr(vg_n)(params_n, imgs)))
    val_r, grad_r = vg_r(params_r, imgs)
    val_n, grad_n = vg_n(params_n, imgs)
    np.testing.assert_allclose(np.asarray(val_r), np.asarray(val_n), rtol=1e-5, atol=1e-6)
    for lr, ln in zip(jax.tree_util.tree_leaves(grad_r), jax.tree_util.tree_leaves(grad_n)):
      np.testing.assert_allclose(np.asarray(lr), np.asarray(ln), rtol=1e-5, atol=1e-6)

    for bad_grid in ([39], [0, 39], [39, -1]):
      with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
        types.RLConfig(model_name="deepseek4.1-flash-mini", ds41_vision_grid=bad_grid)
    with self.assertRaisesRegex(ValueError, "vision_encoder_block='deepseek41'"):
      types.RLConfig(
          model_name="deepseek4.1-flash-mini",
          use_multimodal=True,
          vision_encoder_block=common_types.VisionEncoderBlockType.GEMMA3,
      )
    with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
      types.RLConfig(
          model_name="deepseek4.1-flash-mini",
          use_multimodal=True,
          vision_encoder_block=common_types.VisionEncoderBlockType.DEEPSEEK41,
          ds41_image_token_id=0,
      )

  def test_vision_encoder_fsdp4_kernel_shardings_and_lowering(self) -> None:
    """Verifies all 7 ViT/aligner DenseGeneral kernels shard across FSDP=4 in jitted lowering."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 JAX devices")
    cfg = mm_testing.mini_vision_maxtext_config(
        grid=mm_testing.MINI_GRID_PADDED,
        ici_fsdp_parallelism=4,
        num_hidden_layers_for_vit=2,
        hidden_size_for_vit=1024,
        num_attention_heads_for_vit=16,
        intermediate_size_for_vit=2816,
    )
    fsdp_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    pspec = jax.sharding.PartitionSpec
    with fsdp_mesh, flax.core.spmd.logical_axis_rules(cfg.logical_axis_rules):
      wrapper = encoders.VisionEncoder(config=cfg, mesh=fsdp_mesh, rngs=nnx.Rngs(params=0))
      gdef, pstate, pure, ostate = mm_testing.extract_pure_params(wrapper)
      named_shardings = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(pstate, fsdp_mesh))

      expected_kernel_specs = {
          ("DeepSeek41VisionEncoder_0", "patch_embed", "kernel"): pspec(None, "fsdp"),
          ("DeepSeek41VisionEncoder_0", "blocks_0", "attn", "wqkv", "kernel"): pspec("fsdp", None),
          ("DeepSeek41VisionEncoder_0", "blocks_0", "attn", "wo", "kernel"): pspec(None, "fsdp"),
          ("DeepSeek41VisionEncoder_0", "blocks_0", "mlp", "w1", "kernel"): pspec("fsdp", None),
          ("DeepSeek41VisionEncoder_0", "blocks_0", "mlp", "w2", "kernel"): pspec(None, "fsdp"),
          ("DeepSeek41VisionProjector_0", "w1", "kernel"): pspec("fsdp", None),
          ("DeepSeek41VisionProjector_0", "w2", "kernel"): pspec(None, "fsdp"),
      }
      for key_path, want_spec in expected_kernel_specs.items():
        node = named_shardings
        for k in key_path:
          node = node[k]
        self.assertEqual(node.spec, want_spec, msg=f"Unexpected spec for {key_path}: {node.spec}")

      sharded_params = jax.tree.map(jax.device_put, pure, named_shardings)
      self.assertEqual(
          sharded_params["DeepSeek41VisionEncoder_0"]["patch_embed"]["kernel"].addressable_shards[0].data.shape,
          (588, 256),
      )

      rep_sharding = jax.sharding.NamedSharding(fsdp_mesh, pspec())

      @functools.partial(jax.jit, out_shardings=rep_sharding)
      def jit_fwd(params: dict[str, Any], x: jax.Array) -> jax.Array:
        mod = mm_testing.apply_with_pure_params(gdef, pstate, params, ostate)
        spans, _ = mod(x, deterministic=True)
        return spans

      dummy_imgs = jax.device_put(jnp.ones((2, 1, 35, 588), dtype=jnp.float32), rep_sharding)
      compiled = jit_fwd.lower(sharded_params, dummy_imgs).compile()
      compiled_param_shardings = compiled.input_shardings[0][0]
      for key_path, want_spec in expected_kernel_specs.items():
        node = compiled_param_shardings
        for k in key_path:
          node = node[k]
        self.assertEqual(node.spec, want_spec, msg=f"Compiled input sharding mismatch for {key_path}: {node.spec}")
      hlo_text = compiled.as_text()
      self.assertIn("f32[256,3072]", hlo_text)
      self.assertIn("f32[588,256]", hlo_text)
      self.assertTrue(any(op in hlo_text for op in ("all-gather", "all-reduce")))
      out = np.asarray(jit_fwd(sharded_params, dummy_imgs))
      self.assertEqual(out.shape, (2, 1, 11, cfg.emb_dim))
      self.assertTrue(bool(np.all(np.isfinite(out))))

  def test_optimizer_masks_frozen_vit_with_masked_node_and_active_bias_vl(self) -> None:
    """freeze_vision_encoder_params masks ViT AdamW mu/nu state and decay while keeping aligner and bias_vl active."""
    with tempfile.TemporaryDirectory(prefix="ds41_opt_mask_") as tmp_dir:
      cfg = mm_testing.mini_vision_maxtext_config(
          grid=mm_testing.MINI_GRID_PADDED,
          base_output_directory=tmp_dir,
          tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
          ds41_engram_compressed_vocab_size=0,
          scan_layers=False,
          freeze_vision_encoder_params=True,
          routed_bias=True,
          routed_bias_update_rate=0.05,
          opt_type="adamw",
          adam_weight_decay=0.1,
          learning_rate=1e-2,
          warmup_steps_fraction=0.0,
          learning_rate_schedule_steps=1,
          steps=1,
      )
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      tx = optimizers.get_optimizer(cfg, learning_rate_schedule=1e-2, model=model, mesh=mesh)
      state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

      opt_pure = nnx.to_pure_dict(nnx.state(state.optimizer))
      inner_states = opt_pure["opt_state"]["inner_states"]
      self.assertNotIn("frozen", inner_states)
      trainable_chain = inner_states["trainable"]["inner_state"]
      adam_state = next(v for v in trainable_chain.values() if isinstance(v, dict) and "mu" in v and "nu" in v)
      for moment_key in ("mu", "nu"):
        vit_leaves = jax.tree_util.tree_leaves(
            adam_state[moment_key]["vision_encoder"]["DeepSeek41VisionEncoder_0"],
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        self.assertTrue(vit_leaves)
        for leaf in vit_leaves:
          self.assertIsInstance(leaf, optax.MaskedNode, msg=moment_key)
        proj_mu_w1 = adam_state[moment_key]["vision_encoder"]["DeepSeek41VisionProjector_0"]["w1"]["kernel"]
        self.assertNotIsInstance(proj_mu_w1, optax.MaskedNode, msg=moment_key)

      vit_before = jax.tree_util.tree_map(
          lambda v: np.array(v, copy=True),
          nnx.to_pure_dict(nnx.state(state.model.vision_encoder.DeepSeek41VisionEncoder_0, nnx.Param)),
      )
      proj_before = np.array(state.model.vision_encoder.DeepSeek41VisionProjector_0.w1.kernel.get_value(), copy=True)
      fake_grads = jax.tree_util.tree_map(jnp.ones_like, nnx.state(state.model, nnx.Param))
      state.apply_gradients(fake_grads)
      vit_after = jax.tree_util.tree_map(
          lambda v: np.array(v, copy=True),
          nnx.to_pure_dict(nnx.state(state.model.vision_encoder.DeepSeek41VisionEncoder_0, nnx.Param)),
      )
      proj_after = np.array(state.model.vision_encoder.DeepSeek41VisionProjector_0.w1.kernel.get_value(), copy=True)
      for after_arr, before_arr in zip(jax.tree_util.tree_leaves(vit_after), jax.tree_util.tree_leaves(vit_before)):
        np.testing.assert_array_equal(after_arr, before_arr)
      self.assertFalse(np.array_equal(proj_after, proj_before))

      moe_layer = next(
          n for _, n in nnx.iter_graph(state.model.decoder) if type(n).__name__ == "GateLogit" and hasattr(n, "bias_vl")
      )
      self.assertIsNotNone(moe_layer.bias_vl)
      np.testing.assert_array_equal(
          np.asarray(moe_layer.bias_vl.get_value()), np.zeros_like(np.asarray(moe_layer.bias_vl.get_value()))
      )

  def test_vision_encoder_bfloat16_attention_dots_stay_bf16(self) -> None:
    """Verifies ViT attention contractions keep native bfloat16 operands under dtype_mm='bfloat16'."""
    cfg = mm_testing.mini_vision_maxtext_config(
        grid=mm_testing.MINI_GRID_PADDED,
        dtype_mm="bfloat16",
        weight_dtype="bfloat16",
        float32_qk_product=False,
        num_hidden_layers_for_vit=1,
    )
    mesh = mm_testing.single_device_mesh()
    with mesh:
      encoder = encoders.VisionEncoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0))
    dummy_imgs = jnp.ones((1, 1, 35, 588), dtype=jnp.bfloat16)

    @jax.jit
    def fwd(x: jax.Array) -> jax.Array:
      spans, _ = encoder(x, deterministic=True)
      return spans

    stablehlo_text = fwd.lower(dummy_imgs).as_text("stablehlo")
    dot_lines = [line for line in stablehlo_text.splitlines() if "stablehlo.dot_general" in line]
    self.assertTrue(dot_lines, msg="Expected stablehlo.dot_general ops in lowered VisionEncoder")
    f32_dots = [line for line in dot_lines if re.search(r"tensor<[^>]*xf32>", line)]
    self.assertEqual(f32_dots, [], msg=f"Unexpected float32 dot_general in bf16 ViT: {f32_dots}")
    out = fwd(dummy_imgs)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_get_shaped_batch_deepseek41_standalone(self) -> None:
    """Verifies get_shaped_batch sizes float32 DS4.1 images from global_batch_size with leading-batch sharding."""
    cfg = mm_testing.mini_vision_maxtext_config(
        use_multimodal=True,
        dataset_type="hf",
        hf_path="parquet",
        ds41_dynamic_vision_grid=True,
        ds41_vision_max_n_token=32,
        max_num_images_per_example=2,
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "context"))
    base_shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "context"))
    shaped = maxtext_utils.get_shaped_batch(cfg, batch_sharding=base_shd)
    bsz = int(cfg.global_batch_size_to_load)
    p_max = 9 * (32 - 3)
    self.assertEqual(shaped["images"].shape, (bsz, 2, p_max, 588))
    self.assertEqual(shaped["images"].dtype, jnp.float32)
    self.assertEqual(shaped["images"].sharding.spec, jax.sharding.PartitionSpec("data", None, None, None))
    self.assertEqual(shaped["image_masks"].shape, (bsz, 2))
    self.assertEqual(shaped["image_masks"].dtype, jnp.int32)
    self.assertEqual(shaped["image_masks"].sharding.spec, jax.sharding.PartitionSpec("data", None))
    self.assertEqual(shaped["image_grids"].shape, (bsz, 2, 2))
    self.assertEqual(shaped["image_grids"].dtype, jnp.int32)
    self.assertEqual(shaped["image_grids"].sharding.spec, jax.sharding.PartitionSpec("data", None, None))


@pytest.mark.cpu_only
class DeepSeek41EmbeddingMergeJaxTest(unittest.TestCase):
  """Pure-JAX tests for DS4.1 config reachability, image mask helpers, and embedding merge routing."""

  def test_config_reachability_and_validation(self) -> None:
    out_dir = os.path.join(tempfile.gettempdir(), "maxtext_mm_test_out")
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg = pyconfig.initialize(
          [sys.argv[0], "src/maxtext/configs/base.yml"],
          run_name="ds41_cfg_reachability",
          base_output_directory=out_dir,
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          model_name=model_name,
          use_multimodal=True,
          tokenizer_path=mm_testing.TOKENIZER_PATH,
      )
      self.assertEqual(cfg.ds41_image_token_id, 129264)

    custom_cfg = mm_testing.mini_vision_maxtext_config(ds41_image_token_id=129265)
    sample_tokens = jnp.array([[10, 129265, 129264, 129265]], dtype=jnp.int32)
    mask = mm_processor.get_bidirectional_mask_vision(custom_cfg, sample_tokens)
    np.testing.assert_array_equal(np.asarray(mask), np.array([[False, True, False, True]]))

    for bad_token_id in (0, -5, custom_cfg.vocab_size, custom_cfg.vocab_size + 7):
      with self.assertRaisesRegex(ValueError, "0 < ds41_image_token_id < vocab_size"):
        mm_testing.mini_vision_maxtext_config(ds41_image_token_id=bad_token_id)

    with self.assertRaisesRegex(ValueError, "video"):
      mm_processor.get_bidirectional_mask_vision(custom_cfg, sample_tokens, is_video=True)

    valid_unmm_cfg = mm_testing.mini_vision_maxtext_config(use_multimodal=False, ds41_image_token_id=129265)
    unmm_mask = mm_processor.get_bidirectional_mask_vision(valid_unmm_cfg, sample_tokens)
    np.testing.assert_array_equal(np.asarray(unmm_mask), np.array([[False, True, False, True]]))
    for bad_rt_id in (0, custom_cfg.vocab_size):
      bad_rt_cfg = mm_testing.mini_vision_maxtext_config(use_multimodal=False, ds41_image_token_id=bad_rt_id)
      with self.assertRaisesRegex(ValueError, "0 < ds41_image_token_id < vocab_size"):
        mm_processor.get_bidirectional_mask_vision(bad_rt_cfg, sample_tokens)

  def test_jax_embedding_merge_and_bias_vl_routing_without_torch(self) -> None:
    mesh = mm_testing.single_device_mesh()
    cfg = mm_testing.mini_vision_maxtext_config(
        vocab_size=mm_testing.MINI_VOCAB_SIZE,
        ds41_image_token_id=mm_testing.MINI_IMAGE_TOKEN_ID,
        ds41_engram_compressed_vocab_size=0,
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        scan_layers=False,
    )
    with mesh:
      mt_model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      tokens_j = jnp.full((2, 16), 10, dtype=jnp.int32).at[:, 2:5].set(mm_testing.MINI_IMAGE_TOKEN_ID)
      positions_j = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
      bidir_mask = mm_processor.get_bidirectional_mask_vision(cfg, tokens_j)
      spans = jnp.full((2, 1, 3, cfg.emb_dim), 0.75, dtype=jnp.float32)
      mm_in = MultimodalInput(image_embeddings=spans, bidirectional_mask=bidir_mask)
      merged = mt_model.decoder._apply_embedding(  # pylint: disable=protected-access
          mt_model.token_embedder,
          tokens_j,
          positions_j,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          multimodal_input=mm_in,
      )
      np.testing.assert_allclose(np.asarray(merged[:, 2:5, :]), np.asarray(spans[:, 0, :, :]), rtol=0.0, atol=0.0)

      gate = mt_model.decoder.layers_0.mlp.MoeBlock_0.gate
      gate.bias[...] = jnp.array([0.5, -0.5, 0.0, 0.0], dtype=jnp.float32)
      gate.bias_vl[...] = jnp.array([-0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)
      x_in = jnp.zeros((2, 16, cfg.emb_dim), dtype=jnp.float32)
      gl_txt, _ = gate(x_in, image_mask=None)
      gl_vl, _ = gate(x_in, image_mask=bidir_mask)
      self.assertFalse(bool(jnp.array_equal(gl_txt[:, 2:5, :], gl_vl[:, 2:5, :])))
      self.assertTrue(bool(jnp.array_equal(gl_txt[:, :2, :], gl_vl[:, :2, :])))


@pytest.mark.cpu_only
class DeepSeek41MoeBiasVlAndDefaultModelTest(unittest.TestCase):
  """Pure-JAX tests for separate text/VL routed-bias updates and model_name='default' GateLogit support."""

  def test_moe_load_balance_separates_text_and_vl_biases_and_default_model_name(self) -> None:
    """Default freezes bias_vl while excluding image tokens from text bias; explicit vl_rate>0 works across PP/AR."""
    base_cli = [
        sys.argv[0],
        "src/maxtext/configs/base.yml",
        "model_name=default",
        "decoder_block=deepseek41",
        "run_name=ds41_default_moe_bias_vl",
        "packing=false",
        "enable_checkpointing=false",
        "skip_jax_distributed_system=true",
        "hardware=cpu",
        "use_multimodal=true",
        "vision_encoder_block=deepseek41",
        "ds41_image_token_id=100",
        "base_emb_dim=64",
        "base_num_query_heads=4",
        "base_num_kv_heads=1",
        "head_dim=16",
        "qk_rope_head_dim=8",
        "base_mlp_dim=32",
        "base_moe_mlp_dim=32",
        "num_experts=4",
        "num_experts_per_tok=2",
        "n_routing_groups=2",
        "topk_routing_group=1",
        "shared_experts=1",
        "routed_score_func=sigmoid",
        "routed_bias=true",
        "routed_bias_update_rate=0.01",
        "megablox=false",
        "use_tokamax_gmm=false",
        "capacity_factor=-1",
        "base_num_decoder_layers=2",
        "compress_ratios=[0,0]",
        "ds41_kv_source_layers=[]",
        "ds41_index_source_layers=[]",
        "ds41_engram_layers=[]",
        "mhc_expansion_rate=2",
        "float32_gate_logits=true",
        "per_device_batch_size=1.0",
        "max_target_length=8",
    ]
    for sparse_flag in (False, True):
      sparse_arg = f"sparse_matmul={'true' if sparse_flag else 'false'}"
      # 1. Default config: ds41_routed_bias_vl_update_rate == 0.0 -> bias_vl frozen, text bias still excludes image_mask.
      cfg_default = pyconfig.initialize([*base_cli, sparse_arg])
      self.assertEqual(cfg_default.ds41_routed_bias_vl_update_rate, 0.0)
      self.assertEqual(moe.resolve_vl_bias_update_rate(cfg_default), 0.0)
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_default), cfg_default.mesh_axes)
      with jax.set_mesh(mesh):
        routed_frozen = moe.RoutedMoE(
            config=cfg_default,
            num_experts=4,
            num_experts_per_tok=2,
            mesh=mesh,
            kernel_init=moe.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=(None, None),
            intermediate_dim=32,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            rngs=nnx.Rngs(params=0),
            use_bias_vl=True,
        )
        self.assertIsInstance(routed_frozen.gate.bias, moe.MoEBiasVar)
        self.assertIsInstance(routed_frozen.gate.bias_vl, moe.MoEBiasVar)
        routed_frozen.gate.kernel.set_value(jnp.zeros_like(routed_frozen.gate.kernel.get_value()))
        routed_frozen.gate.bias.set_value(jnp.array([10.0, 9.0, -5.0, -6.0], dtype=jnp.float32))
        routed_frozen.gate.bias_vl.set_value(jnp.array([-6.0, -5.0, 9.0, 10.0], dtype=jnp.float32))

        x = jnp.ones((1, 4, 64), dtype=jnp.float32)
        img_mask = jnp.array([[False, False, True, True]], dtype=jnp.bool_)
        _, _, text_upd_frozen = routed_frozen(x, image_mask=img_mask)
        np.testing.assert_allclose(np.asarray(text_upd_frozen), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32))
        self.assertFalse(hasattr(routed_frozen, "moe_bias_vl_updates"))

      # 2. Explicit ds41_routed_bias_vl_update_rate=0.003 (!= text 0.01): updates bias_vl with 0.003 across all paths.
      cfg_vl = pyconfig.initialize([*base_cli, sparse_arg, "ds41_routed_bias_vl_update_rate=0.003"])
      self.assertAlmostEqual(moe.resolve_vl_bias_update_rate(cfg_vl), 0.003, places=7)
      with jax.set_mesh(mesh):
        routed = moe.RoutedMoE(
            config=cfg_vl,
            num_experts=4,
            num_experts_per_tok=2,
            mesh=mesh,
            kernel_init=moe.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=(None, None),
            intermediate_dim=32,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            rngs=nnx.Rngs(params=0),
            use_bias_vl=True,
        )
        routed.gate.kernel.set_value(jnp.zeros_like(routed.gate.kernel.get_value()))
        routed.gate.bias.set_value(jnp.array([10.0, 9.0, -5.0, -6.0], dtype=jnp.float32))
        routed.gate.bias_vl.set_value(jnp.array([-6.0, -5.0, 9.0, 10.0], dtype=jnp.float32))

        _, _, text_upd = routed(x, image_mask=img_mask)
        sown_vl = routed.moe_bias_vl_updates.get_value()
        vl_upd = sown_vl[-1] if isinstance(sown_vl, tuple) else sown_vl
        np.testing.assert_allclose(np.asarray(text_upd), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(vl_upd), np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32))

        all_text_mask = jnp.zeros((1, 4), dtype=jnp.bool_)
        _, _, text_only_upd = routed(x, image_mask=all_text_mask)
        sown_vl_zero = routed.moe_bias_vl_updates.get_value()
        vl_zero = sown_vl_zero[-1] if isinstance(sown_vl_zero, tuple) else sown_vl_zero
        np.testing.assert_allclose(np.asarray(text_only_upd), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(vl_zero), np.zeros((4,), dtype=np.float32))

    # 3. Pipeline-parallel decode (_decode_moe_bias_update) and deferred-AR (finalize_deferred_intermediates) use vl_rate.
    pp_stub = py_types.SimpleNamespace(
        using_pipeline_parallelism=True,
        routed_bias_update_rate=0.01,
        ds41_routed_bias_vl_update_rate=0.003,
    )
    raw_txt_counts = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
    raw_vl_counts = jnp.array([0, 0, 2, 2], dtype=jnp.int32)
    # pylint: disable=protected-access
    dec_txt = pre_train._decode_moe_bias_update(raw_txt_counts, pp_stub, rate=pp_stub.routed_bias_update_rate)
    dec_vl = pre_train._decode_moe_bias_update(raw_vl_counts, pp_stub, rate=moe.resolve_vl_bias_update_rate(pp_stub))
    # pylint: enable=protected-access
    np.testing.assert_allclose(np.asarray(dec_txt), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(dec_vl), np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32))

    ar_stub = py_types.SimpleNamespace(
        defer_small_all_reduces=True,
        gradient_accumulation_steps=1,
        routed_bias_update_rate=0.01,
        ds41_routed_bias_vl_update_rate=0.003,
    )
    deferred_inter = {
        "decoder": {
            "layers_0": {
                "mlp": {
                    "MoeBlock_0": {
                        "moe_bias_updates": raw_txt_counts[None, None, :],
                        "moe_bias_vl_updates": raw_vl_counts[None, None, :],
                    }
                }
            }
        }
    }
    finalized = moe.finalize_deferred_intermediates(deferred_inter, ar_stub)
    fin_blk = finalized["decoder"]["layers_0"]["mlp"]["MoeBlock_0"]
    np.testing.assert_allclose(
        np.asarray(fin_blk["moe_bias_updates"]), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(fin_blk["moe_bias_vl_updates"]), np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32)
    )

    # 4. gradient_accumulation_steps=2 with real NNX sow 1-tuples (counts,) finalizes int32 counts to float32 updates.
    cfg_ga2 = pyconfig.initialize(
        [
            *base_cli,
            "sparse_matmul=False",
            "gradient_accumulation_steps=2",
            "per_device_batch_size=2",
            "ds41_routed_bias_vl_update_rate=0.003",
        ]
    )
    with jax.set_mesh(mesh):
      routed_ga2 = moe.RoutedMoE(
          config=cfg_ga2,
          num_experts=4,
          num_experts_per_tok=2,
          mesh=mesh,
          kernel_init=moe.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=(None, None),
          intermediate_dim=32,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          rngs=nnx.Rngs(params=0),
          use_bias_vl=True,
      )
      routed_ga2.gate.kernel.set_value(jnp.zeros_like(routed_ga2.gate.kernel.get_value()))
      routed_ga2.gate.bias.set_value(jnp.array([10.0, 9.0, -5.0, -6.0], dtype=jnp.float32))
      routed_ga2.gate.bias_vl.set_value(jnp.array([-6.0, -5.0, 9.0, 10.0], dtype=jnp.float32))
      _, ga_params, _ = nnx.split(routed_ga2, nnx.Param, ...)
      ga_param_shardings = jax.tree.map(
          lambda _: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), ga_params
      )

      def _ga_loss_fn(mod, _cfg, batch_slice, _rng, _params, is_train=True):
        del is_train
        out, _, text_u = mod(batch_slice["inputs"], image_mask=batch_slice["img_mask"])
        if text_u is not None:
          mod.sow(nnx.Intermediate, "moe_bias_updates", text_u)
        inter = nnx.to_pure_dict(nnx.pop(mod, nnx.Intermediate))
        loss_val = jnp.sum(out**2)
        return loss_val, {
            "xent_sum": loss_val,
            "moe_lb_loss": jnp.zeros((), dtype=jnp.float32),
            "indexer_loss": jnp.zeros((), dtype=jnp.float32),
            "mtp_loss": jnp.zeros((), dtype=jnp.float32),
            "total_weights": jnp.array(4, dtype=jnp.int32),
            "intermediate_outputs": {"decoder": {"layers_0": inter}},
        }

      ga_bsz = int(cfg_ga2.global_batch_size_to_train_on)
      ga_data = {
          "inputs": jnp.ones((ga_bsz, 4, 64), dtype=jnp.float32),
          "img_mask": jnp.broadcast_to(jnp.array([[False, False, True, True]], dtype=jnp.bool_), (ga_bsz, 4)),
      }
      _, ga_aux, _ = gradient_accumulation.gradient_accumulation_loss_and_grad(
          _ga_loss_fn,
          cfg_ga2,
          routed_ga2,
          ga_params,
          ga_param_shardings,
          ga_data,
          jax.random.PRNGKey(0),
      )
      ga_l0 = ga_aux["intermediate_outputs"]["decoder"]["layers_0"]
      self.assertIsInstance(ga_l0["moe_bias_updates"], tuple)
      self.assertEqual(ga_l0["moe_bias_updates"][0].dtype, jnp.float32)
      self.assertEqual(ga_l0["moe_bias_vl_updates"][0].dtype, jnp.float32)
      np.testing.assert_allclose(
          np.asarray(ga_l0["moe_bias_updates"][0]), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32)
      )
      np.testing.assert_allclose(
          np.asarray(ga_l0["moe_bias_vl_updates"][0]), np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32)
      )

    # 5. routed_bias_update_rate=0.0 with ds41_routed_bias_vl_update_rate=0.003 updates bias_vl only (text_upd is None),
    # and ds41_routed_bias_vl_update_rate=-1.0 inherits routed_bias_update_rate.
    cfg_inherit = pyconfig.initialize([*base_cli, "sparse_matmul=False", "ds41_routed_bias_vl_update_rate=-1.0"])
    self.assertAlmostEqual(moe.resolve_vl_bias_update_rate(cfg_inherit), 0.01, places=7)
    for sparse_arg in ("sparse_matmul=False", "sparse_matmul=True"):
      cfg_vl_only = pyconfig.initialize(
          [*base_cli, sparse_arg, "routed_bias_update_rate=0.0", "ds41_routed_bias_vl_update_rate=0.003"]
      )
      with jax.set_mesh(mesh):
        routed_vlo = moe.RoutedMoE(
            config=cfg_vl_only,
            num_experts=4,
            num_experts_per_tok=2,
            mesh=mesh,
            kernel_init=moe.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=(None, None),
            intermediate_dim=32,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            rngs=nnx.Rngs(params=0),
            use_bias_vl=True,
        )
        routed_vlo.gate.kernel.set_value(jnp.zeros_like(routed_vlo.gate.kernel.get_value()))
        routed_vlo.gate.bias.set_value(jnp.array([10.0, 9.0, -5.0, -6.0], dtype=jnp.float32))
        routed_vlo.gate.bias_vl.set_value(jnp.array([-6.0, -5.0, 9.0, 10.0], dtype=jnp.float32))
        x_vlo = jnp.ones((1, 4, 64), dtype=jnp.float32)
        img_mask_vlo = jnp.array([[False, False, True, True]], dtype=jnp.bool_)
        _, _, text_vlo = routed_vlo(x_vlo, image_mask=img_mask_vlo)
        self.assertIsNone(text_vlo)
        sown_vlo = routed_vlo.moe_bias_vl_updates.get_value()
        vlo_upd = sown_vlo[-1] if isinstance(sown_vlo, tuple) else sown_vlo
        np.testing.assert_allclose(np.asarray(vlo_upd), np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32))

  def test_sparse_and_dense_moe_token_mask_with_image_mask_bias_updates(self) -> None:
    """Verifies token_mask excludes padded tokens from text bias and image bias_vl updates in dense and sparse MoE."""
    base_cli = [
        sys.argv[0],
        "src/maxtext/configs/base.yml",
        "model_name=default",
        "decoder_block=deepseek41",
        "run_name=ds41_moe_token_mask_bias_vl",
        "packing=false",
        "enable_checkpointing=false",
        "skip_jax_distributed_system=true",
        "hardware=cpu",
        "use_multimodal=true",
        "vision_encoder_block=deepseek41",
        "ds41_image_token_id=100",
        "base_emb_dim=64",
        "base_num_query_heads=4",
        "base_num_kv_heads=1",
        "head_dim=16",
        "qk_rope_head_dim=8",
        "base_mlp_dim=32",
        "base_moe_mlp_dim=32",
        "num_experts=4",
        "num_experts_per_tok=2",
        "n_routing_groups=2",
        "topk_routing_group=1",
        "shared_experts=1",
        "routed_score_func=sigmoid",
        "routed_bias=true",
        "routed_bias_update_rate=0.01",
        "megablox=false",
        "use_tokamax_gmm=false",
        "capacity_factor=-1",
        "base_num_decoder_layers=2",
        "compress_ratios=[0,0]",
        "ds41_kv_source_layers=[]",
        "ds41_index_source_layers=[]",
        "ds41_engram_layers=[]",
        "mhc_expansion_rate=2",
        "float32_gate_logits=true",
        "per_device_batch_size=1.0",
        "max_target_length=8",
    ]
    # Build a 6-token sequence where 4 padded tokens carry the opposite routing preference of the 2 valid tokens.
    # Dropping token_mask flips the majority expert count and reverses the sign of bias_updates / moe_bias_vl_updates.
    token_mask = jnp.array([[True, True, False, False, False, False]], dtype=jnp.bool_)
    image_mask = jnp.array([[False, True, True, True, False, False]], dtype=jnp.bool_)
    x = jnp.zeros((1, 6, 64), dtype=jnp.float32)
    x = x.at[:, 0, 0].set(1.0)  # valid text -> experts (0, 1)
    x = x.at[:, 1, 0].set(-1.0)  # valid image -> experts (2, 3)
    x = x.at[:, 2:4, 0].set(1.0)  # padded images -> would flip VL majority to (0, 1) if unmasked
    x = x.at[:, 4:6, 0].set(-1.0)  # padded text -> would flip text majority to (2, 3) if unmasked

    for vl_rate in (0.0, 0.003):
      dense_res: tuple[np.ndarray, np.ndarray | None] | None = None
      for sparse_flag in (False, True):
        cfg = pyconfig.initialize(
            [
                *base_cli,
                f"sparse_matmul={'true' if sparse_flag else 'false'}",
                f"ds41_routed_bias_vl_update_rate={vl_rate}",
            ]
        )
        mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
        with jax.set_mesh(mesh):
          routed = moe.RoutedMoE(
              config=cfg,
              num_experts=4,
              num_experts_per_tok=2,
              mesh=mesh,
              kernel_init=moe.nd_dense_init(1.0, "fan_in", "truncated_normal"),
              kernel_axes=(None, None),
              intermediate_dim=32,
              dtype=jnp.float32,
              weight_dtype=jnp.float32,
              rngs=nnx.Rngs(params=0),
              use_bias_vl=True,
          )
          gate_w = jnp.zeros_like(routed.gate.kernel.get_value())
          gate_w = gate_w.at[0, 0:2].set(5.0).at[0, 2:4].set(-5.0)
          routed.gate.kernel.set_value(gate_w)
          routed.gate.bias.set_value(jnp.zeros((4,), dtype=jnp.float32))
          routed.gate.bias_vl.set_value(jnp.zeros((4,), dtype=jnp.float32))

          _, _, text_upd = routed(x, image_mask=image_mask, token_mask=token_mask)
          np.testing.assert_allclose(np.asarray(text_upd), np.array([-0.01, -0.01, 0.01, 0.01], dtype=np.float32))
          if vl_rate == 0.0:
            self.assertFalse(hasattr(routed, "moe_bias_vl_updates"))
            cur = (np.asarray(text_upd), None)
          else:
            sown_vl = routed.moe_bias_vl_updates.get_value()
            vl_upd = np.asarray(sown_vl[-1] if isinstance(sown_vl, tuple) else sown_vl)
            np.testing.assert_allclose(vl_upd, np.array([0.003, 0.003, -0.003, -0.003], dtype=np.float32))
            cur = (np.asarray(text_upd), vl_upd)
          if dense_res is None:
            dense_res = cur
          else:
            np.testing.assert_allclose(cur[0], dense_res[0])
            if dense_res[1] is not None:
              np.testing.assert_allclose(cur[1], dense_res[1])


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeek41MultimodalLogitsJaxTest(unittest.TestCase):
  """Pure-JAX full-model multimodal forward, packing isolation, and scan-vs-unscan tests."""

  def test_jax_multimodal_forward_and_packed_segment_isolation_without_torch(self) -> None:
    with tempfile.TemporaryDirectory(prefix="ds41_t13_notorch_") as ckpt_dir:
      mm_testing.write_numpy_mini_mm_checkpoint(ckpt_dir, seed=41, use_multimodal=True)
      rng = np.random.default_rng(42)
      full_tokens_np = rng.integers(3, 500, size=(2, mm_testing.MINI_SEQ_LEN), dtype=np.int32)
      for b, starts in enumerate(mm_testing.MINI_DEFAULT_SPAN_STARTS):
        for s in starts:
          full_tokens_np[b, s : s + 11] = mm_testing.MINI_IMAGE_TOKEN_ID
      full_tokens = jnp.asarray(full_tokens_np, dtype=jnp.int32)
      full_pos = jnp.broadcast_to(jnp.arange(mm_testing.MINI_SEQ_LEN, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      full_seg = jnp.asarray(
          (np.arange(mm_testing.MINI_SEQ_LEN)[None, :] < _VALID_FULL_LEN).astype(np.int32)
          * np.ones((2, 1), dtype=np.int32)
      )
      full_patches = jnp.asarray(rng.standard_normal((2, 2, 35, 588), dtype=np.float32))

      packed_tokens_np = rng.integers(3, 500, size=(2, mm_testing.MINI_SEQ_LEN), dtype=np.int32)
      packed_tokens_np[:, 2:13] = mm_testing.MINI_IMAGE_TOKEN_ID
      packed_tokens_np[:, _DOC1_LEN + 2 : _DOC1_LEN + 13] = mm_testing.MINI_IMAGE_TOKEN_ID
      packed_tokens = jnp.asarray(packed_tokens_np, dtype=jnp.int32)
      pos_row = np.zeros((mm_testing.MINI_SEQ_LEN,), dtype=np.int32)
      pos_row[:_DOC1_LEN] = np.arange(_DOC1_LEN, dtype=np.int32)
      pos_row[_DOC1_LEN : _DOC1_LEN + _DOC2_LEN] = np.arange(_DOC2_LEN, dtype=np.int32)
      packed_pos = jnp.broadcast_to(jnp.asarray(pos_row, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      seg_row = np.zeros((mm_testing.MINI_SEQ_LEN,), dtype=np.int32)
      seg_row[:_DOC1_LEN] = 1
      seg_row[_DOC1_LEN : _DOC1_LEN + _DOC2_LEN] = 2
      packed_seg = jnp.broadcast_to(jnp.asarray(seg_row, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      packed_patches = full_patches

      outputs_by_scan: dict[bool, dict[str, np.ndarray]] = {}
      output_keys = ["full_valid", "packed_doc1", "packed_doc2"]
      for scan_layers in (False, True):
        model, _, mesh = mm_testing.load_mini_maxtext_from_safetensors(
            ckpt_dir,
            scan_layers=scan_layers,
            vocab_size=mm_testing.MINI_VOCAB_SIZE,
            ds41_image_token_id=mm_testing.MINI_IMAGE_TOKEN_ID,
            ds41_engram_compressed_vocab_size=0,
            tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        )
        gdef, pstate, ppure, rstate = mm_testing.extract_pure_params(model)
        primary = eval_multimodal_packed_and_full_logits(
            gdef,
            pstate,
            ppure,
            rstate,
            mesh,
            full_tokens=full_tokens,
            full_pos=full_pos,
            full_seg=full_seg,
            full_patches=full_patches,
            packed_tokens=packed_tokens,
            packed_pos=packed_pos,
            packed_seg=packed_seg,
            packed_patches=packed_patches,
        )
        outputs_by_scan[scan_layers] = primary
        if not scan_layers:
          pert_pure = negctl.perturb_one_leaf(
              ppure,
              target_path="vision_encoder/DeepSeek41VisionProjector_0/w1/kernel",
              rel_scale=1e-2,
              seed=91,
          )
          perturbed = eval_multimodal_packed_and_full_logits(
              gdef,
              pstate,
              pert_pure,
              rstate,
              mesh,
              full_tokens=full_tokens,
              full_pos=full_pos,
              full_seg=full_seg,
              full_patches=full_patches,
              packed_tokens=packed_tokens,
              packed_pos=packed_pos,
              packed_seg=packed_seg,
              packed_patches=packed_patches,
          )
          status, reasons = e2e_check.negative_run_status(
              primary,
              primary,
              perturbed,
              output_keys,
              layer_parity.FP32_TOLERANCE,
          )
          self.assertEqual(status, "FIRED", msg=f"Torch-free negctl {status}: {reasons}")

          # Cross-segment isolation: mutating segment-2 text tokens leaves segment-1 logits unchanged.
          packed_tokens_alt2_np = packed_tokens_np.copy()
          packed_tokens_alt2_np[:, _DOC1_LEN + 13 : _DOC1_LEN + _DOC2_LEN] = (
              packed_tokens_alt2_np[:, _DOC1_LEN + 13 : _DOC1_LEN + _DOC2_LEN] % 400
          ) + 50
          alt_doc2 = eval_multimodal_packed_and_full_logits(
              gdef,
              pstate,
              ppure,
              rstate,
              mesh,
              full_tokens=full_tokens,
              full_pos=full_pos,
              full_seg=full_seg,
              full_patches=full_patches,
              packed_tokens=jnp.asarray(packed_tokens_alt2_np, dtype=jnp.int32),
              packed_pos=packed_pos,
              packed_seg=packed_seg,
              packed_patches=packed_patches,
          )
          np.testing.assert_allclose(
              primary["packed_doc1"],
              alt_doc2["packed_doc1"],
              rtol=layer_parity.FP32_TOLERANCE.rtol,
              atol=layer_parity.FP32_TOLERANCE.atol,
          )
          self.assertGreater(float(np.max(np.abs(primary["packed_doc2"] - alt_doc2["packed_doc2"]))), 1e-3)

      ok_scan, _, scan_failures = e2e_check.check_outputs(
          outputs_by_scan[False],
          outputs_by_scan[True],
          output_keys,
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(ok_scan, msg=f"Scanned vs unscanned mismatch: {scan_failures}")


if __name__ == "__main__":
  unittest.main()
