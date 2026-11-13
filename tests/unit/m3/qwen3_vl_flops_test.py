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

"""Unit tests for `maxtext.m3.models.qwen3_vl.qwen3_vl_flops`."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.qwen3 import qwen3_flops
from maxtext.m3.models.qwen3_vl import qwen3_vl_flops
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _make_vl_config(output_dir: str, **overrides):
  """Builds a validated Qwen3-VL `MaxTextConfig` via `pyconfig.initialize`."""
  args = {
      "model_name": "qwen3-vl-2b",
      "use_m3_model": True,
      "override_model_config": True,
      "use_multimodal": True,
      "freeze_vision_encoder_params": True,
      "per_device_batch_size": 2,
      "max_target_length": 128,
      "base_emb_dim": 128,
      "base_mlp_dim": 256,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "vocab_size": 512,
      "normalization_layer_epsilon": 1e-6,
      "use_qk_norm": True,
      "logits_via_embedding": True,
      "normalize_embedding_logits": False,
      "rope_max_timescale": 5_000_000.0,
      "enable_dropout": False,
      "scan_layers": False,
      "dtype": "float32",
      "dtype_mm": "float32",
      "weight_dtype": "float32",
      "float32_logits": True,
      "image_size_for_vit": 64,
      "hidden_size_for_vit": 64,
      "intermediate_size_for_vit": 128,
      "num_attention_heads_for_vit": 4,
      "num_hidden_layers_for_vit": 3,
      "num_channels_for_vit": 3,
      "patch_size_for_vit": 16,
      "temporal_patch_size_for_vit": 2,
      "spatial_merge_size_for_vit": 2,
      "out_hidden_size_for_vit": 128,
      "num_position_embeddings_for_vit": 16,
      "rope_theta_for_vit": 10_000,
      "deepstack_visual_indexes_for_vit": [0, 1],
      "use_mrope": True,
      "mrope_section": [8, 4, 4],
      "gradient_accumulation_steps": 1,
      "dataset_type": "synthetic",
      "use_dpo": False,
      "enable_checkpointing": False,
      "base_output_directory": output_dir,
      "run_name": "qwen3_vl_flops_test",
  }
  args.update(overrides)
  cli = ["", get_test_config_path()]
  for key, val in args.items():
    if isinstance(val, bool):
      cli.append(f"{key}={str(val).lower()}")
    else:
      cli.append(f"{key}={val}")
  return pyconfig.initialize(cli)


class Qwen3VlFlopsTest(absltest.TestCase):
  """Tests multimodal and frozen-vision FLOP accounting for Qwen3-VL."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.enterContext(nnx.use_eager_sharding(True))

  def test_text_only_mode_matches_qwen3_tflops_exactly(self):
    cfg = _make_vl_config(self.output_dir, use_multimodal=False)
    vl_res = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(cfg, log=False)
    qwen3_res = qwen3_flops.calculate_qwen3_tflops_per_device(cfg, log=False)
    self.assertEqual(vl_res, qwen3_res)

  def test_vision_weight_fwd_flops_match_exact_instantiated_parameter_kernels(self):
    cfg = _make_vl_config(self.output_dir)
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      abstract_model = nnx.eval_shape(lambda: create_qwen3_vl_model(cfg, mesh=mesh, rngs=nnx.Rngs(0)))
    vision_params = nnx.to_flat_state(nnx.state(abstract_model.vision_encoder, nnx.Param))

    batch = cfg.per_device_batch_size
    num_patches = (cfg.image_size_for_vit // cfg.patch_size_for_vit) ** 2
    merged_tokens = num_patches // (cfg.spatial_merge_size_for_vit**2)

    expected_encoder_fwd_flops = 0
    expected_projector_fwd_flops = 0
    for path, var in vision_params:
      # Sum FLOPs over all projection/convolution `kernel` weights (excluding biases, LayerNorm scales, and pos_embed).
      if path[-1] != "kernel":
        continue
      joined = ".".join(str(p) for p in path)
      tokens = merged_tokens if ("merger" in joined) else num_patches
      num_weights = int(np.prod(var.shape))
      kernel_flops = 2 * batch * tokens * num_weights
      if path[0] == "Qwen3VLVisionEncoder_0":
        expected_encoder_fwd_flops += kernel_flops
      else:
        expected_projector_fwd_flops += kernel_flops

    enc_w, merg_w, attn_fwd_flops = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(cfg)
    self.assertEqual(enc_w, expected_encoder_fwd_flops)
    self.assertEqual(merg_w, expected_projector_fwd_flops)

    weight_fwd_flops, total_attn_fwd = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg)
    self.assertEqual(weight_fwd_flops, expected_encoder_fwd_flops + expected_projector_fwd_flops)
    expected_attn_fwd_flops = 4 * batch * (num_patches**2) * cfg.hidden_size_for_vit * cfg.num_hidden_layers_for_vit
    self.assertEqual(attn_fwd_flops, expected_attn_fwd_flops)
    self.assertEqual(total_attn_fwd, expected_attn_fwd_flops)

  def test_frozen_vs_unfrozen_multiplier_and_negative_control(self):
    text_cfg = _make_vl_config(self.output_dir, use_multimodal=False)
    text_total, text_weight, text_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(text_cfg, log=False)

    frozen_cfg = _make_vl_config(self.output_dir, use_multimodal=True, freeze_vision_encoder_params=True)
    unfrozen_cfg = _make_vl_config(self.output_dir, use_multimodal=True, freeze_vision_encoder_params=False)

    frozen_total, frozen_weight, frozen_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(frozen_cfg, log=False)
    unfrozen_total, unfrozen_weight, unfrozen_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(
        unfrozen_cfg, log=False
    )

    enc_w, merg_w, vis_a_fwd = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(frozen_cfg)

    # When frozen: `Qwen3VLVisionEncoder_0` (including DeepStack mergers) is 1x forward-only,
    # while the single `Qwen3VLVisionProjector_0` merger is trainable 3x fwd+bwd.
    self.assertAlmostEqual(frozen_weight - text_weight, 3.0 * merg_w / 10**12, places=12)
    self.assertAlmostEqual(frozen_attn - text_attn, vis_a_fwd / 10**12, places=12)
    expected_frozen_total = text_total + (enc_w + 3.0 * merg_w + vis_a_fwd) / 10**12
    self.assertAlmostEqual(frozen_total, expected_frozen_total, places=12)

    # Verify DeepStack mergers inside `Qwen3VLVisionEncoder_0` are counted as frozen (not 3x):
    no_deepstack_cfg = _make_vl_config(
        self.output_dir,
        use_multimodal=True,
        freeze_vision_encoder_params=True,
        deepstack_visual_indexes_for_vit=[],
    )
    _, no_ds_merg_w, _ = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(no_deepstack_cfg)
    self.assertEqual(merg_w, no_ds_merg_w)
    _, no_ds_frozen_weight, _ = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(no_deepstack_cfg, log=False)
    self.assertAlmostEqual(frozen_weight, no_ds_frozen_weight, places=12)

    # When unfrozen: both `Qwen3VLVisionEncoder_0` and `Qwen3VLVisionProjector_0` are 3x fwd+bwd.
    self.assertAlmostEqual(unfrozen_weight - text_weight, 3.0 * (enc_w + merg_w) / 10**12, places=12)
    self.assertAlmostEqual(unfrozen_attn - text_attn, 3.0 * vis_a_fwd / 10**12, places=12)
    expected_unfrozen_total = text_total + 3.0 * (enc_w + merg_w + vis_a_fwd) / 10**12
    self.assertAlmostEqual(unfrozen_total, expected_unfrozen_total, places=12)

    # Negative control: difference between unfrozen and frozen is 2 * enc_w + 2 * vis_a_fwd.
    self.assertAlmostEqual(unfrozen_total - frozen_total, (2.0 * enc_w + 2.0 * vis_a_fwd) / 10**12, places=12)

    # Undeclared legacy alias flags fail schema validation under `pyconfig.initialize`.
    with self.assertRaises(Exception):
      _make_vl_config(self.output_dir, freeze_vision_tower=False)

  def test_vision_tower_stop_gradient_freezes_encoder_and_deepstack_mergers_only(self):
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    tokens = jnp.array([[1, 151655, 151655, 151655, 151655, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :, None], (1, 8, 3))
    images = jax.random.normal(jax.random.key(7), (1, 3, 2, 64, 64), dtype=jnp.float32)

    def _grad_norms(freeze_encoder: bool) -> tuple[float, float, float]:
      cfg = _make_vl_config(
          self.output_dir,
          per_device_batch_size=1,
          max_target_length=8,
          freeze_vision_encoder_params=freeze_encoder,
      )
      with jax.set_mesh(mesh):
        model = create_qwen3_vl_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))

        def _loss(m):
          logits = m(
              tokens,
              decoder_positions=positions,
              encoder_images=images,
          )
          return jnp.sum(logits * jnp.arange(logits.size, dtype=jnp.float32).reshape(logits.shape))

        grads = nnx.grad(_loss)(model)
      flat_grads = nnx.to_flat_state(grads.vision_encoder)
      enc_block_max = max(
          float(jnp.max(jnp.abs(v[...])))
          for p, v in flat_grads
          if p[0] == "Qwen3VLVisionEncoder_0" and not str(p[1]).startswith("merger_")
      )
      deepstack_max = max(
          float(jnp.max(jnp.abs(v[...])))
          for p, v in flat_grads
          if p[0] == "Qwen3VLVisionEncoder_0" and str(p[1]).startswith("merger_")
      )
      proj_max = max(float(jnp.max(jnp.abs(v[...]))) for p, v in flat_grads if p[0] == "Qwen3VLVisionProjector_0")
      return enc_block_max, deepstack_max, proj_max

    frozen_enc, frozen_ds, frozen_proj = _grad_norms(freeze_encoder=True)
    unfrozen_enc, unfrozen_ds, unfrozen_proj = _grad_norms(freeze_encoder=False)

    self.assertEqual(frozen_enc, 0.0)
    self.assertEqual(frozen_ds, 0.0)
    self.assertGreater(frozen_proj, 0.0)
    self.assertGreater(unfrozen_enc, 0.0)
    self.assertGreater(unfrozen_ds, 0.0)
    self.assertGreater(unfrozen_proj, 0.0)

  def test_non_square_image_multi_image_dpo_and_validation(self):
    rect_cfg = _make_vl_config(self.output_dir, image_size_for_vit=[64, 96])
    w_rect, a_rect = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(rect_cfg)
    self.assertGreater(w_rect, 0)
    self.assertGreater(a_rect, 0)

    cfg_1img = _make_vl_config(self.output_dir, image_size_for_vit=64, max_num_images_per_example=1)
    cfg_3img = _make_vl_config(self.output_dir, image_size_for_vit=64, max_num_images_per_example=3)
    w_1, a_1 = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg_1img)
    w_3, a_3 = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg_3img)
    self.assertEqual(w_3, 3 * w_1)
    self.assertEqual(a_3, 3 * a_1)

    dpo_cfg = _make_vl_config(self.output_dir, use_dpo=True, packing=False, freeze_vision_encoder_params=True)
    dpo_total, _, _ = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(dpo_cfg, log=False)
    base_total, _, _ = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(
        _make_vl_config(self.output_dir, use_dpo=False, freeze_vision_encoder_params=True), log=False
    )
    self.assertGreater(dpo_total, base_total)

    with self.assertRaisesRegex(ValueError, "divisible"):
      qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(_make_vl_config(self.output_dir, image_size_for_vit=50))

  def test_registry_and_maxtext_utils_dispatch_for_qwen3_vl_2b_config(self):
    self.assertIs(
        m3_models.MODEL_FLOPS_REGISTRY["qwen3-vl-2b"],
        qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device,
    )
    cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-vl-2b",
            "use_m3_model=true",
            "scan_layers=false",
            "per_device_batch_size=1",
            "max_target_length=512",
            "enable_checkpointing=false",
            f"base_output_directory={self.output_dir}",
            "run_name=qwen3_vl_flops_dispatch",
        ]
    )
    total_mm, weight_mm, attn_mm = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    total_txt, _, _ = qwen3_flops.calculate_qwen3_tflops_per_device(cfg, log=False)
    self.assertGreater(total_mm, total_txt)
    enc_w, _, _ = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(cfg)
    # When frozen, total_tflops includes 1x frozen encoder weight TFLOPs while learnable weight_mm excludes it.
    self.assertAlmostEqual(total_mm, weight_mm + attn_mm + enc_w / 10**12, places=9)


if __name__ == "__main__":
  absltest.main()
