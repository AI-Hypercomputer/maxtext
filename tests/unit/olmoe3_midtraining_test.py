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

"""Unit tests for OLMo 3.5 Small 256K long-context and SigLIP2 multimodal mid-training."""

from __future__ import annotations

import os
from unittest import mock

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
).strip()

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline import olmo_data_grain
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.models import olmoe3
from maxtext.models import olmoe3_vision
from maxtext.utils import maxtext_utils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_YML = os.path.join(_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
_MIDTRAIN_256K_YML = os.path.join(
    _REPO_ROOT, "src", "maxtext", "configs", "olmo", "olmo35_small_midtrain_256k.yml"
)
_VISION_8K_YML = os.path.join(
    _REPO_ROOT, "src", "maxtext", "configs", "olmo", "olmo35_small_vision_align_8k.yml"
)


def _physical_8x8xz(z_dim: int) -> np.ndarray:
  return np.arange(8 * 8 * z_dim).reshape(8, 8, z_dim)


def _coords_8x8xz(ids: np.ndarray, z_dim: int) -> np.ndarray:
  ids = np.asarray(ids)
  return np.stack([ids // (8 * z_dim), (ids // z_dim) % 8, ids % z_dim], axis=-1)


class Olmoe3MidtrainingMeshTest(parameterized.TestCase):
  """Verifies contiguous `(8, 8, 4)` cuboid `CP * EP = 256` placement on TPU v4 `(8, 8, Z)` slices."""

  @parameterized.named_parameters(
      ("8x8x8_cp8_ep32", 8, 8, 32),
      ("8x8x8_cp16_ep16", 8, 16, 16),
      ("8x8x8_cp32_ep8", 8, 32, 8),
      ("8x8x16_cp8_ep32", 16, 8, 32),
      ("8x8x16_cp32_ep8", 16, 32, 8),
      ("8x8x48_cp8_ep32", 48, 8, 32),
      ("8x8x48_cp16_ep16", 48, 16, 16),
      ("8x8x48_cp32_ep8", 48, 32, 8),
  )
  def test_cp_times_ep_256_stays_inside_contiguous_8x8x4_cuboid(self, z_dim: int, cp: int, ep: int):
    total = 8 * 8 * z_dim
    dp = total // (cp * ep)
    mesh = maxtext_utils.tpu_8x8xz_expert_mesh(
        _physical_8x8xz(z_dim), ep, [dp, 1, cp, ep], cp_size=cp
    )
    self.assertIsNotNone(mesh)
    self.assertEqual(mesh.shape, (dp, 1, cp, ep))
    self.assertCountEqual(mesh.reshape(-1).tolist(), range(total))

    # Every `(data, fsdp)` replica's `CP x EP = 256` submesh must occupy an exact `(8, 8, 4)` torus slab.
    for d in range(dp):
      sub = mesh[d, 0].reshape(-1)
      c = _coords_8x8xz(sub, z_dim)
      extent = tuple(int(c[:, i].max() - c[:, i].min() + 1) for i in range(3))
      self.assertEqual(extent, (8, 8, 4))
      self.assertEqual(tuple(int(c[:, i].min()) % b for i, b in enumerate((8, 8, 4))), (0, 0, 0))

  def test_8x8x8_ep64_plane_override_with_cp4(self):
    with mock.patch.dict(os.environ, {"OLMOE3_EP64_SUBMESH": "8x8x1"}):
      mesh = maxtext_utils.tpu_8x8xz_expert_mesh(
          _physical_8x8xz(8), 64, [2, 1, 4, 64], cp_size=4
      )
      self.assertIsNotNone(mesh)
      self.assertEqual(mesh.shape, (2, 1, 4, 64))
      for d in range(2):
        for c_idx in range(4):
          c = _coords_8x8xz(mesh[d, 0, c_idx], 8)
          extent = tuple(int(c[:, i].max() - c[:, i].min() + 1) for i in range(3))
          self.assertEqual(extent, (8, 8, 1))


class Olmoe3VisionAndSpliceTest(parameterized.TestCase):
  """Tests SigLIP2 vision encoder, `attention_meanq` connector, `-1` masking, CP splice, and converter."""

  def _tiny_vision_cfg(self, cp: int = 1):
    return pyconfig.initialize([
        "",
        _BASE_YML,
        "run_name=ut_vis",
        "base_output_directory=/tmp/ut_vis",
        "dataset_type=synthetic",
        "model_name=olmo35-tiny-e256k8-sh2x",
        "override_model_config=True",
        "base_emb_dim=128",
        "dtype=float32",
        "dtype_mm=float32",
        "weight_dtype=float32",
        "matmul_precision=highest",
        "use_multimodal=True",
        "vision_encoder_block=olmo35",
        "image_size_for_vit=28",
        "patch_size_for_vit=14",
        "hidden_size_for_vit=32",
        "intermediate_size_for_vit=64",
        "num_attention_heads_for_vit=4",
        "num_hidden_layers_for_vit=3",
        "olmo_vit_feature_layers=[2,1]",
        "olmo_connector_mlp_hidden_size=96",
        "olmo_vit_remat_blocks=True",
        "olmo_vit_crop_chunk_size=2",
        "olmo_vit_cp_shard_crops=True",
        "olmo_image_patch_token_id=100280",
        "custom_mesh_and_rule=cp-as-ep",
        "context_parallel_load_balance=False",
        "ici_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        f"ici_context_parallelism={cp}",
        "ici_expert_parallelism=1",
        "per_device_batch_size=2.0",
        "max_target_length=32",
        "compile_topology=v4-128",
        "compile_topology_num_slices=1",
    ])

  def test_patchify_cfirst_order(self):
    # Verify (3, p, p) C-first flattening matches PyTorch NCHW unfold order.
    hwc = jnp.arange(1 * 1 * 28 * 28 * 3, dtype=jnp.float32).reshape(1, 1, 28, 28, 3)
    patches = olmoe3_vision.patchify_hwc_to_olmo_cfirst(hwc, patch_size=14)
    self.assertEqual(patches.shape, (1, 1, 4, 588))
    # Top-left patch channel 0 comes before channel 1.
    expected_p0 = np.transpose(np.asarray(hwc[0, 0, :14, :14, :]), (2, 0, 1)).reshape(-1)
    np.testing.assert_array_equal(np.asarray(patches[0, 0, 0]), expected_p0)

  def test_vision_encoder_connector_and_padding_mask_gradients(self):
    cfg = self._tiny_vision_cfg(cp=1)
    dev = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    mesh = Mesh(dev, ("data", "fsdp", "context", "expert"), axis_types=(AxisType.Auto,) * 4)

    enc = olmoe3_vision.OLMo35VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))
    con = olmoe3_vision.OLMo35VisionConnector(cfg, mesh, rngs=nnx.Rngs(params=1))

    imgs = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 28, 28, 3), dtype=jnp.float32)
    # 4 crops * 4 patches = 16 patches total; build 3 pooled tokens:
    # token 0: full 4 patches, token 1: partial (2 valid + 2 `-1`), token 2: all `-1` padding.
    pooled_idx = jnp.array(
        [
            [[0, 1, 2, 3], [4, 5, -1, -1], [-1, -1, -1, -1]],
            [[2, 3, 6, 7], [8, -1, -1, -1], [-1, -1, -1, -1]],
        ],
        dtype=jnp.int32,
    )

    def _loss(enc_m, con_m, x_img):
      feats = enc_m(x_img)
      proj = con_m(feats, pooled_idx)
      return jnp.sum(proj**2), (feats, proj)

    (loss, (feats, proj)), grads = nnx.value_and_grad(_loss, argnums=(0, 1, 2), has_aux=True)(enc, con, imgs)
    self.assertEqual(feats.shape, (2, 16, 64))
    self.assertEqual(proj.shape, (2, 3, 128))
    self.assertTrue(bool(jnp.isfinite(loss)))
    # All `-1` padding rows must produce exact zeros.
    np.testing.assert_array_equal(np.asarray(proj[:, 2, :]), np.zeros((2, 128), dtype=np.float32))
    # Active rows must be non-zero, and all parameter/image gradients must be finite.
    self.assertGreater(float(jnp.linalg.norm(proj[:, :2, :])), 0.0)
    for g_leaf in jax.tree_util.tree_leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_leaf))))

  def test_cp_sharded_image_embedding_splice_matches_cp1(self):
    cfg1 = self._tiny_vision_cfg(cp=1)
    cfg4 = self._tiny_vision_cfg(cp=4)
    b, t, n_pooled, d = 2, 32, 5, 128
    rng = np.random.default_rng(7)
    y_np = rng.standard_normal((b, t, d)).astype(np.float32)
    img_np = rng.standard_normal((b, n_pooled, d)).astype(np.float32)
    tok_np = np.full((b, t), 42, dtype=np.int32)
    # Scatter `<im_patch>` = 100280 across multiple CP shards differently per batch item.
    tok_np[0, [1, 7, 9, 18, 29]] = 100280
    tok_np[1, [0, 11, 27]] = 100280

    ref = olmoe3_vision.merge_olmo35_image_embeddings(
        jnp.asarray(y_np),
        jnp.asarray(tok_np),
        jnp.asarray(img_np),
        patch_token_id=100280,
        mesh=None,
        config=cfg1,
    )

    dev4 = np.array(jax.devices()[:4]).reshape(1, 1, 4, 1)
    mesh4 = Mesh(dev4, ("data", "fsdp", "context", "expert"), axis_types=(AxisType.Auto,) * 4)
    y_sh = jax.device_put(y_np, NamedSharding(mesh4, PartitionSpec("data", "context", None)))
    tok_sh = jax.device_put(tok_np, NamedSharding(mesh4, PartitionSpec("data", "context")))
    img_sh = jax.device_put(img_np, NamedSharding(mesh4, PartitionSpec("data", None, None)))

    with jax.set_mesh(mesh4):
      got = olmoe3_vision.merge_olmo35_image_embeddings(
          y_sh, tok_sh, img_sh, patch_token_id=100280, mesh=mesh4, config=cfg4
      )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0.0, atol=1e-6)

  def test_vision_checkpoint_converter_roundtrip(self):
    rng = np.random.default_rng(11)
    num_layers = 2
    state = {
        "vision.patch_embedding.weight": rng.standard_normal((32, 588)).astype(np.float32),
        "vision.patch_embedding.bias": rng.standard_normal((32,)).astype(np.float32),
        "vision.positional_embedding": rng.standard_normal((4, 32)).astype(np.float32),
        "connector.pooling.wq.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "connector.pooling.wq.bias": rng.standard_normal((32,)).astype(np.float32),
        "connector.pooling.wk.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "connector.pooling.wk.bias": rng.standard_normal((32,)).astype(np.float32),
        "connector.pooling.wv.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "connector.pooling.wv.bias": rng.standard_normal((32,)).astype(np.float32),
        "connector.pooling.wo.weight": rng.standard_normal((32, 32)).astype(np.float32),
        "connector.pooling.wo.bias": rng.standard_normal((32,)).astype(np.float32),
        "connector.projector.w1.weight": rng.standard_normal((96, 32)).astype(np.float32),
        "connector.projector.w2.weight": rng.standard_normal((128, 96)).astype(np.float32),
        "connector.projector.w3.weight": rng.standard_normal((96, 32)).astype(np.float32),
    }
    for i in range(num_layers):
      for n in ("attn_norm", "ffn_norm"):
        state[f"vision.blocks.{i}.{n}.weight"] = rng.standard_normal((32,)).astype(np.float32)
        state[f"vision.blocks.{i}.{n}.bias"] = rng.standard_normal((32,)).astype(np.float32)
      for p in ("wq", "wk", "wv", "wo"):
        state[f"vision.blocks.{i}.attn.{p}.weight"] = rng.standard_normal((32, 32)).astype(np.float32)
        state[f"vision.blocks.{i}.attn.{p}.bias"] = rng.standard_normal((32,)).astype(np.float32)
      state[f"vision.blocks.{i}.ffn.w1.weight"] = rng.standard_normal((64, 32)).astype(np.float32)
      state[f"vision.blocks.{i}.ffn.w1.bias"] = rng.standard_normal((64,)).astype(np.float32)
      state[f"vision.blocks.{i}.ffn.w2.weight"] = rng.standard_normal((32, 64)).astype(np.float32)
      state[f"vision.blocks.{i}.ffn.w2.bias"] = rng.standard_normal((32,)).astype(np.float32)

    mt = olmoe3_vision.olmo_core_vision_to_maxtext_params(state, num_layers=num_layers)
    rt = olmoe3_vision.maxtext_vision_to_olmo_core_params(mt, num_layers=num_layers)
    self.assertEqual(set(rt.keys()), set(state.keys()))
    for k, v in state.items():
      np.testing.assert_array_equal(rt[k], v)


class Olmoe3UlyssesKdaAndDataTest(parameterized.TestCase):
  """Tests 1D (`CP=8`) & 2D (`CP=16, 32`) Ulysses KDA fwd+bwd parity, placeholder schema, and YAML configs."""

  def _make_kda_cfg(self, cp: int):
    return pyconfig.initialize([
        "",
        _BASE_YML,
        f"run_name=ut_kda_cp{cp}",
        "base_output_directory=/tmp/ut_kda",
        "dataset_type=synthetic",
        "model_name=olmo35-tiny-e256k8-sh2x",
        "override_model_config=True",
        "base_emb_dim=128",
        "gdn_num_key_heads=8",
        "gdn_num_value_heads=8",
        "gdn_key_head_dim=16",
        "gdn_value_head_dim=16",
        "gdn_chunk_size=32",
        "kda_sub_block=16",
        "kda_chunked_impl=subblock",
        "kda_pallas_fused=False",
        "kda_flat_head_layout=False",
        "kda_fused_input_proj=True",
        "use_tokamax_kda=False",
        "dtype=float32",
        "weight_dtype=float32",
        "gdn_state_dtype=float32",
        "kda_conv_in_compute_dtype=False",
        "matmul_precision=highest",
        "kda_inverse_precision=highest",
        "custom_mesh_and_rule=cp-as-ep",
        "context_parallel_load_balance=False",
        "ici_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        f"ici_context_parallelism={cp}",
        "ici_expert_parallelism=1",
        "per_device_batch_size=1.0",
        "max_target_length=128",
        "packing=True",
        "olmoe3_segments_for_emo_only=False",
        "emo_enabled=False",
        "compile_topology=v4-128",
        "compile_topology_num_slices=1",
    ])

  def test_1d_and_2d_ulysses_kda_fwd_bwd_parity(self):
    def _eval_cp(cp: int, x_np: np.ndarray, dy_np: np.ndarray, seg_np: np.ndarray):
      cfg = self._make_kda_cfg(cp)
      devs = np.array(jax.devices()[:cp]).reshape(1, 1, cp, 1)
      mesh = Mesh(devs, ("data", "fsdp", "context", "expert"), axis_types=(AxisType.Auto,) * 4)
      with jax.set_mesh(mesh):
        mod = olmoe3.OLMoE3KimiDeltaAttention(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0))
        graphdef, state = nnx.split(mod)
        p_spec = NamedSharding(mesh, PartitionSpec("data", "context", None))
        s_spec = NamedSharding(mesh, PartitionSpec("data", "context"))
        x = jax.device_put(x_np, p_spec)
        dy = jax.device_put(dy_np, p_spec)
        seg = jax.device_put(seg_np, s_spec)

        @jax.jit
        def _run(st, inp, up, sg):
          def _loss(s, xx):
            m = nnx.merge(graphdef, s)
            y = m(xx, decoder_segment_ids=sg)
            return jnp.sum(y * up), y

          (_, y_out), (gs, gx) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(st, inp)
          return y_out, gx, gs

        y, gx, gs = _run(state, x, dy, seg)
        g_leaves = {
            jax.tree_util.keystr(p, simple=True, separator="/"): np.asarray(v)
            for p, v in jax.tree_util.tree_leaves_with_path(gs)
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating)
        }
        return np.asarray(y), np.asarray(gx), g_leaves

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 128, 128)).astype(np.float32) * 0.2
    dy_np = rng.standard_normal((2, 128, 128)).astype(np.float32) * 0.1
    seg_np = np.ones((2, 128), dtype=np.int32)
    seg_np[0, 50:] = 2
    seg_np[1, 33:90] = 2
    seg_np[1, 90:] = 3

    y1, gx1, gw1 = _eval_cp(1, x_np, dy_np, seg_np)
    self.assertGreaterEqual(len(gw1), 10)
    for cp in (8, 16, 32):
      yc, gxc, gwc = _eval_cp(cp, x_np, dy_np, seg_np)
      np.testing.assert_allclose(yc, y1, rtol=1e-5, atol=1e-5, err_msg=f"fwd mismatch at CP={cp}")
      np.testing.assert_allclose(gxc, gx1, rtol=1e-5, atol=1e-5, err_msg=f"dx mismatch at CP={cp}")
      self.assertEqual(set(gwc.keys()), set(gw1.keys()))
      for k in gw1:
        np.testing.assert_allclose(gwc[k], gw1[k], rtol=1e-4, atol=1e-5, err_msg=f"dw[{k}] mismatch at CP={cp}")

  def test_placeholder_iterator_matches_olmo_grain_and_multimodal_schema(self):
    cfg_grain = pyconfig.initialize([
        "",
        _BASE_YML,
        "run_name=ut_ph_grain",
        "base_output_directory=/tmp/ut_ph",
        "dataset_type=olmo_grain",
        "olmo_index_path=/tmp/nonexistent_index.json",
        "model_name=olmo35-tiny-e256k8-sh2x",
        "per_device_batch_size=1.0",
        "max_target_length=64",
        "compile_topology=v4-128",
        "compile_topology_num_slices=1",
    ])
    grain_batch = next(synthetic_data_processing.PlaceHolderDataIterator.get_place_holder_synthetic_data(cfg_grain))
    self.assertEqual(
        set(grain_batch.keys()),
        {"inputs", "inputs_position", "inputs_segmentation", "targets", "targets_segmentation"},
    )

    cfg_grain_mm = pyconfig.initialize([
        "",
        _BASE_YML,
        "run_name=ut_ph_grain_mm",
        "base_output_directory=/tmp/ut_ph",
        "dataset_type=olmo_grain",
        "olmo_index_path=/tmp/nonexistent_index.json",
        "model_name=olmo35-tiny-e256k8-sh2x",
        "override_model_config=True",
        "use_multimodal=True",
        "vision_encoder_block=olmo35",
        "image_size_for_vit=28",
        "patch_size_for_vit=14",
        "olmo_max_crops_per_seq=8",
        "per_device_batch_size=1.0",
        "max_target_length=64",
        "compile_topology=v4-128",
        "compile_topology_num_slices=1",
    ])
    grain_mm_batch = next(
        synthetic_data_processing.PlaceHolderDataIterator.get_place_holder_synthetic_data(cfg_grain_mm)
    )
    self.assertEqual(
        set(grain_mm_batch.keys()),
        {
            "inputs",
            "inputs_position",
            "inputs_segmentation",
            "targets",
            "targets_segmentation",
            "images",
            "pooled_patches_idx",
        },
    )
    self.assertEqual(grain_mm_batch["images"].shape[1:], (8, 4, 588))
    self.assertEqual(grain_mm_batch["pooled_patches_idx"].shape[1:], (8, 4))
    self.assertTrue(bool(np.all(grain_mm_batch["pooled_patches_idx"] == -1)))

    st_reset = olmo_data_grain.ShiftToInputsTargets(segment_eos_token_id=99, reset_positions_on_segment=True)
    mapped = st_reset.map({"tokens": np.array([10, 11, 99, 20, 21, 99, 30, 31], dtype=np.int32)})
    self.assertEqual(mapped["inputs_segmentation"].tolist(), [1, 1, 2, 2, 2, 3, 3, 3])
    self.assertEqual(mapped["inputs_position"].tolist(), [0, 1, 0, 1, 2, 0, 1, 2])

    cfg_syn = pyconfig.initialize([
        "",
        _BASE_YML,
        "run_name=ut_ph_syn",
        "base_output_directory=/tmp/ut_ph",
        "dataset_type=synthetic",
        "model_name=olmo35-tiny-e256k8-sh2x",
        "per_device_batch_size=1.0",
        "max_target_length=64",
        "compile_topology=v4-128",
        "compile_topology_num_slices=1",
    ])
    syn_batch = next(synthetic_data_processing.PlaceHolderDataIterator.get_place_holder_synthetic_data(cfg_syn))
    self.assertIn("targets_position", syn_batch)

  def test_vision_tflops_calculator_exact_connector_and_freeze_accounting(self):
    cfg_joint = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_tf_joint",
        "base_output_directory=/tmp/ut_tf",
        "dataset_type=synthetic",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
    ])
    cfg_bridge = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_tf_bridge",
        "base_output_directory=/tmp/ut_tf",
        "dataset_type=synthetic",
        "freeze_vision_encoder_params=True",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
    ])
    j_tot, j_wt, j_at = maxtext_utils.calculate_olmo35_vision_layers_tflops_per_device(cfg_joint)
    b_tot, b_wt, b_at = maxtext_utils.calculate_olmo35_vision_layers_tflops_per_device(cfg_bridge)
    self.assertAlmostEqual(j_tot, j_wt + j_at, places=9)
    self.assertAlmostEqual(b_tot, b_wt + b_at, places=9)
    # Trainable 25L SigLIP2 has strictly < 3x total TFLOPs of frozen-ViT Bridge (since Connector stays 3x in both),
    # and > 2.8x (since 25L SigLIP2 dominates Connector).
    ratio = j_tot / b_tot
    self.assertGreater(ratio, 2.8)
    self.assertLess(ratio, 3.0)

  def test_midtraining_yaml_configs_initialize_cleanly(self):
    cfg_256k_3k = pyconfig.initialize([
        "",
        _MIDTRAIN_256K_YML,
        "run_name=ut_256k_3k",
        "base_output_directory=/tmp/ut_yml",
        "dataset_type=synthetic",
        "compile_topology=v4-6144",
        "compile_topology_num_slices=1",
    ])
    self.assertEqual(cfg_256k_3k.max_target_length, 262144)
    self.assertEqual(cfg_256k_3k.global_batch_size_to_train_on, 96)
    self.assertEqual(cfg_256k_3k.global_batch_size_to_train_on * cfg_256k_3k.max_target_length, 25165824)
    self.assertFalse(cfg_256k_3k.optimizer_memory_host_offload)

    cfg_256k_512 = pyconfig.initialize([
        "",
        _MIDTRAIN_256K_YML,
        "run_name=ut_256k_512",
        "base_output_directory=/tmp/ut_yml",
        "dataset_type=synthetic",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
        "ici_data_parallelism=2",
        "ici_context_parallelism=8",
        "ici_expert_parallelism=32",
        "per_device_batch_size=0.125",
    ])
    self.assertEqual(cfg_256k_512.global_batch_size_to_train_on, 64)
    self.assertEqual(cfg_256k_512.global_batch_size_to_train_on * cfg_256k_512.max_target_length, 16777216)

    cfg_vis_8k = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_vis_8k",
        "base_output_directory=/tmp/ut_yml",
        "dataset_type=synthetic",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
    ])
    self.assertTrue(cfg_vis_8k.use_multimodal)
    self.assertEqual(cfg_vis_8k.ici_data_parallelism, 8)
    self.assertEqual(cfg_vis_8k.ici_expert_parallelism, 64)
    self.assertEqual(cfg_vis_8k.global_batch_size_to_train_on, 512)
    self.assertEqual(cfg_vis_8k.olmo_max_crops_per_seq, 64)


if __name__ == "__main__":
  absltest.main()
