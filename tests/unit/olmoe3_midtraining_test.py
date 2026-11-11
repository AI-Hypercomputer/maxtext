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

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.input_pipeline import olmo_data_grain
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.models import models
from maxtext.models import olmoe3
from maxtext.models import olmoe3_vision
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding as max_sharding

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

  def test_connector_cp_sharded_and_chunked_scan_fwd_bwd_parity(self):
    cfg1 = self._tiny_vision_cfg(cp=1)
    cfg4 = self._tiny_vision_cfg(cp=4)
    b, n_crops, p_per_crop, dim_in = 2, 8, 4, 64
    n_total = n_crops * p_per_crop
    n_pooled = 16  # divisible by cp=4 and chunk=2 -> 2 chunks of 2 per CP rank
    rng = np.random.default_rng(31)
    feats_np = rng.standard_normal((b, n_total, dim_in)).astype(np.float32) * 0.2
    dy_np = rng.standard_normal((b, n_pooled, 128)).astype(np.float32) * 0.1
    p_idx_np = rng.integers(0, n_total, size=(b, n_pooled, 4), dtype=np.int32)
    p_idx_np[:, 3, 2:] = -1
    p_idx_np[:, 7, :] = -1
    p_idx_np[:, 14, 1:] = -1

    def _eval_conn(cp: int, cfg):
      dev = np.array(jax.devices()[:cp]).reshape(1, 1, cp, 1)
      mesh = Mesh(dev, ("data", "fsdp", "context", "expert"), axis_types=(AxisType.Auto,) * 4)
      with jax.set_mesh(mesh):
        con = olmoe3_vision.OLMo35VisionConnector(cfg, mesh, rngs=nnx.Rngs(params=9))
        graphdef, state = nnx.split(con)
        f_sh = jax.device_put(feats_np, NamedSharding(mesh, PartitionSpec("data", None, None)))
        p_sh = jax.device_put(p_idx_np, NamedSharding(mesh, PartitionSpec("data", None, None)))
        dy_sh = jax.device_put(dy_np, NamedSharding(mesh, PartitionSpec("data", None, None)))

        @jax.jit
        def _step(st, ff, pp, dy):
          def _l(s, f_in):
            m = nnx.merge(graphdef, s)
            out = m(f_in, pp)
            return jnp.sum(out * dy), out

          (_, y_out), (gs, gf) = jax.value_and_grad(_l, argnums=(0, 1), has_aux=True)(st, ff)
          return y_out, gf, gs

        y, gf, gs = _step(state, f_sh, p_sh, dy_sh)
        gw = {
            jax.tree_util.keystr(k, simple=True, separator="/"): np.asarray(v)
            for k, v in jax.tree_util.tree_leaves_with_path(gs)
        }
        return np.asarray(y), np.asarray(gf), gw

    y1, gf1, gw1 = _eval_conn(1, cfg1)
    y4, gf4, gw4 = _eval_conn(4, cfg4)
    np.testing.assert_allclose(y4, y1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gf4, gf1, rtol=1e-5, atol=1e-5)
    for k in gw1:
      np.testing.assert_allclose(gw4[k], gw1[k], rtol=1e-5, atol=1e-5, err_msg=f"grad mismatch on {k}")


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
    self.assertTrue(cfg_vis_8k.olmo_segment_eos_ends_document)
    self.assertTrue(cfg_vis_8k.olmo_reset_positions_on_segment)
    self.assertTrue(cfg_vis_8k.olmo_use_loss_weights)
    self.assertTrue(cfg_vis_8k.static_loss_cotangent)
    self.assertAlmostEqual(cfg_vis_8k.z_loss_multiplier, 1e-5, places=10)
    self.assertTrue(cfg_256k_3k.olmo_segment_eos_ends_document)
    self.assertTrue(cfg_256k_3k.olmo_reset_positions_on_segment)
    self.assertTrue(cfg_256k_3k.olmo_mask_chunk_boundary_targets_only)
    self.assertEqual(list(cfg_256k_3k.frozen_parameters_mask), [])
    self.assertEqual(cfg_256k_3k.steps, 14880)
    self.assertTrue(cfg_256k_3k.olmo_loss_denominator_counts_masked_instances)
    self.assertTrue(cfg_256k_3k.static_loss_cotangent)
    self.assertAlmostEqual(cfg_256k_3k.z_loss_multiplier, 1e-5, places=10)
    self.assertEqual(list(cfg_vis_8k.frozen_parameters_mask), ["logits_dense"])
    self.assertAlmostEqual(cfg_vis_8k.learning_rate, 1e-4, places=10)
    self.assertAlmostEqual(cfg_vis_8k.olmo_connector_lr_scale, 0.2, places=10)
    self.assertAlmostEqual(cfg_vis_8k.olmo_vision_lr_scale, 0.02, places=10)
    self.assertAlmostEqual(cfg_vis_8k.olmo_embedding_lr_scale, 0.2, places=10)
    self.assertAlmostEqual(cfg_vis_8k.olmo_vision_lr_cosine_alpha_f, 0.1, places=10)
    self.assertEqual(cfg_vis_8k.olmo_connector_warmup_steps, 50)

  def test_chunk_boundary_only_target_mask_preserves_intra_chunk_eos(self):
    eos = 100257
    tokens = np.array([eos, 10, 11, eos, 12, 13, 14, 15, 16, eos, 17, 18], dtype=np.uint32)
    out = olmo_data_grain.ShiftToInputsTargets(
        segment_eos_token_id=eos,
        reset_positions_on_segment=True,
        segment_eos_ends_document=True,
        segment_chunk_length=6,
        mask_cross_segment_targets=False,
        mask_chunk_boundary_targets_only=True,
    ).map({"tokens": tokens, "instance_id": 0, "file_id": 0, "instance_mask": True})
    expected_seg = np.array([1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5], dtype=np.int32)
    expected_pos = np.array([0, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1], dtype=np.int32)
    # Only chunk edge i=5 (6-1) and sequence end i=11 are masked to 0; EOS transitions at i=0, 3, 9 stay unmasked!
    expected_tgt_seg = np.array([1, 2, 2, 2, 3, 0, 4, 4, 4, 4, 5, 0], dtype=np.int32)
    np.testing.assert_array_equal(out["inputs_segmentation"], expected_seg)
    np.testing.assert_array_equal(out["inputs_position"], expected_pos)
    np.testing.assert_array_equal(out["targets_segmentation"], expected_tgt_seg)

  def test_bridge_hold_final_lr_and_joint_per_group_schedules(self):
    from maxtext.optimizers import optimizers

    cfg_bridge = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_sched_bridge",
        "base_output_directory=/tmp/ut_sched",
        "dataset_type=synthetic",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
        "learning_rate=2.0e-4",
        "lr_schedule_type=cosine",
        "learning_rate_schedule_steps=250",
        "steps=500",
        "warmup_steps_fraction=0.4",
        "learning_rate_final_fraction=0.1",
        "olmo_hold_final_lr=True",
        "olmo_vision_lr_cosine_alpha_f=-1.0",
    ])
    sched_b = maxtext_utils.create_learning_rate_schedule(cfg_bridge)
    self.assertAlmostEqual(float(sched_b(0)), 0.0, places=9)
    self.assertAlmostEqual(float(sched_b(100)), 2.0e-4, places=9)
    self.assertAlmostEqual(float(sched_b(249)), 2.0e-5, places=9)
    self.assertAlmostEqual(float(sched_b(250)), 2.0e-5, places=9)
    self.assertAlmostEqual(float(sched_b(499)), 2.0e-5, places=9)

    cfg_joint = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_sched_joint",
        "base_output_directory=/tmp/ut_sched",
        "dataset_type=synthetic",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
        "skip_step_on_spikes=False",
    ])
    lm_sched = maxtext_utils.create_learning_rate_schedule(cfg_joint)
    self.assertAlmostEqual(float(lm_sched(100)), 1.0e-4, places=9)
    self.assertAlmostEqual(float(lm_sched(3499)), 0.0, places=9)

    tx = optimizers.get_optimizer(cfg_joint, lm_sched)
    dummy_params = {
        "decoder": {"layers": {"w": jnp.ones((4,), dtype=jnp.float32)}, "logits_dense": {"k": jnp.ones((4,), dtype=jnp.float32)}},
        "vision_encoder": {
            "OLMo35VisionConnector_0": {"w": jnp.ones((4,), dtype=jnp.float32)},
            "OLMo35VisionEncoder_0": {"w": jnp.ones((4,), dtype=jnp.float32)},
        },
        "token_embedder": {"embedding": jnp.ones((4, 4), dtype=jnp.float32)},
    }
    opt_state = tx.init(dummy_params)
    ones_grad = jax.tree_util.tree_map(jnp.ones_like, dummy_params)
    updates_0, _ = tx.update(ones_grad, opt_state, dummy_params)
    # logits_dense is frozen -> exact 0 update
    np.testing.assert_array_equal(np.asarray(updates_0["decoder"]["logits_dense"]["k"]), np.zeros((4,), dtype=np.float32))
    # At step 50: in 8K Joint (`olmo_trainable_embedding_row_range=[100278, 100284]`), `lm.embeddings` shares
    # Connector's 50-step warmup (`2.0e-5`), whereas ViT uses base 100-step warmup (`1.0e-6` = 0.5x peak).
    # When `olmo_trainable_embedding_row_range` is empty (256K MT+LC), embeddings uses base 100-step warmup.
    conn_50 = optimizers._make_olmo35_cosine_schedule(2e-5, 0.1, 50, 3500, 3500)(50)
    emb_50_joint = optimizers._make_olmo35_cosine_schedule(2e-5, 0.1, 50, 3500, 3500)(50)
    emb_50_full = optimizers._make_olmo35_cosine_schedule(2e-5, 0.1, 100, 3500, 3500)(50)
    vis_50 = optimizers._make_olmo35_cosine_schedule(2e-6, 0.1, 100, 3500, 3500)(50)
    self.assertAlmostEqual(float(conn_50), 2.0e-5, places=9)
    self.assertAlmostEqual(float(emb_50_joint), 2.0e-5, places=9)
    self.assertAlmostEqual(float(emb_50_full), 1.0e-5, places=9)
    self.assertAlmostEqual(float(vis_50), 1.0e-6, places=9)

  def test_eos_ends_document_chunk_split_and_cross_segment_mask_match_olmo_core(self):
    # Matches OLMo-core segment_ids_from_eos (eos.cumsum() - eos) + 8K chunk replay split.
    eos = 100257
    tokens = np.array([eos, 10, 11, eos, 12, 13, 14, 15, 16, eos, 17, 18], dtype=np.uint32)
    out = olmo_data_grain.ShiftToInputsTargets(
        segment_eos_token_id=eos,
        reset_positions_on_segment=True,
        segment_eos_ends_document=True,
        segment_chunk_length=6,
        mask_cross_segment_targets=True,
    ).map({"tokens": tokens, "instance_id": 0, "file_id": 0, "instance_mask": True})

    # Without chunk split: eos.cumsum() - eos = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
    # Plus chunk split at index 6 -> starts at [1, 4, 6, 10], segments = [1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5]
    expected_seg = np.array([1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5], dtype=np.int32)
    expected_pos = np.array([0, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1], dtype=np.int32)
    # Cross-segment positions i in {0 (1->2), 3 (2->3), 5 (3->4), 9 (4->5)} + trailing pad 11 masked to 0,
    # while intra-doc EOS predictions at i=2 (11->eos) and i=8 (16->eos) remain active!
    expected_tgt_seg = np.array([0, 2, 2, 0, 3, 0, 4, 4, 4, 0, 5, 0], dtype=np.int32)
    np.testing.assert_array_equal(out["inputs_segmentation"], expected_seg)
    np.testing.assert_array_equal(out["inputs_position"], expected_pos)
    np.testing.assert_array_equal(out["targets_segmentation"], expected_tgt_seg)

  def test_scalable_softmax_resets_per_packed_document_across_cp1_and_cp4(self):
    # Verify OLMoE3 Full Attention resets log(pos + 1) per packed document from decoder_segment_ids
    # (matching OLMo-core _apply_scalable_softmax(q, cu_doc_lens)), even when a custom VLM loader
    # leaves inputs_position = [0..L-1], under both CP=1 and CP=4.
    b, seq_len = 2, 64
    rng = np.random.default_rng(19)
    half_x = rng.standard_normal((b, seq_len // 2, 128)).astype(np.float32)
    x_np = np.concatenate([half_x, half_x], axis=1)
    segs_np = np.concatenate(
        [np.ones((b, seq_len // 2), np.int32), np.full((b, seq_len // 2), 2, np.int32)], axis=1
    )
    mono_pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (b, seq_len)).copy()
    reset_pos_np = np.concatenate(
        [np.arange(seq_len // 2, dtype=np.int32), np.arange(seq_len // 2, dtype=np.int32)]
    )[None, :].repeat(b, axis=0)

    def _run_cp(cp: int, pos_arr: np.ndarray):
      cfg = pyconfig.initialize([
          "",
          _BASE_YML,
          f"run_name=ut_ssmax_cp{cp}",
          "base_output_directory=/tmp/ut_ssmax",
          "dataset_type=synthetic",
          "model_name=olmo35-tiny",
          "override_model_config=True",
          "base_emb_dim=128",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=32",
          "use_scalable_softmax=True",
          "attention=dot_product",
          "dtype=float32",
          "weight_dtype=float32",
          f"max_target_length={seq_len}",
          "olmoe3_segments_for_emo_only=False",
          "olmo_reset_positions_on_segment=True",
          "ici_data_parallelism=1",
          "ici_fsdp_parallelism=1",
          f"ici_context_parallelism={cp}",
          "ici_expert_parallelism=1",
          "context_parallel_load_balance=False",
          "per_device_batch_size=2.0",
      ])
      dev = np.array(jax.devices()[:cp]).reshape(1, 1, cp, 1)
      mesh = Mesh(dev, ("data", "fsdp", "context", "expert"), axis_types=(AxisType.Auto,) * 4)
      with jax.set_mesh(mesh):
        attn = olmoe3.OLMoE3Attention(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=5))
        # Set nontrivial positive ssmax_scale so span scaling is sharply exercised!
        attn.attention.ssmax_scale.set_value(jnp.array([1.2, 0.8, 1.5, 0.5], dtype=jnp.float32))
        x_sh = jax.device_put(x_np, NamedSharding(mesh, PartitionSpec("data", "context", None)))
        seg_sh = jax.device_put(segs_np, NamedSharding(mesh, PartitionSpec("data", "context")))
        pos_sh = jax.device_put(pos_arr, NamedSharding(mesh, PartitionSpec("data", "context")))
        out, _ = jax.jit(lambda m, inp, s, p: m(inp, s, p, True, MODEL_MODE_TRAIN))(attn, x_sh, seg_sh, pos_sh)
      return np.asarray(out)

    out_cp1_mono = _run_cp(1, mono_pos_np)
    out_cp1_reset = _run_cp(1, reset_pos_np)
    out_cp4_mono = _run_cp(4, mono_pos_np)
    # 1) Monotonic 0..63 inputs_position matches pre-reset 0..31, 0..31 identically:
    np.testing.assert_allclose(out_cp1_mono, out_cp1_reset, rtol=0.0, atol=1e-6)
    # 2) Two identical 32-token packed documents produce identical attention outputs (no 2x log(pos+1) explosion!):
    np.testing.assert_allclose(out_cp1_mono[:, :32, :], out_cp1_mono[:, 32:, :], rtol=1e-5, atol=1e-5)
    # 3) CP=4 matches CP=1 identically:
    np.testing.assert_allclose(out_cp4_mono, out_cp1_mono, rtol=1e-5, atol=1e-5)

  def test_vlm_loss_weights_sharding_placeholder_and_im_patch_masking(self):
    cfg = pyconfig.initialize([
        "",
        _VISION_8K_YML,
        "run_name=ut_vlm_lw",
        "base_output_directory=/tmp/ut_lw",
        "dataset_type=synthetic",
        "max_target_length=64",
        "olmo_max_crops_per_seq=1",
        "ici_data_parallelism=1",
        "ici_expert_parallelism=1",
        "ragged_buffer_factor=-1",
        "per_device_batch_size=1.0",
        "compile_topology=v4-8",
        "compile_topology_num_slices=1",
    ])
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "expert"))
    shardings = max_sharding.get_input_data_sharding(cfg, mesh)
    shaped_batch = maxtext_utils.get_shaped_batch(cfg, batch_sharding=shardings)
    ph_batch = next(synthetic_data_processing.PlaceHolderDataIterator.get_place_holder_synthetic_data(cfg))
    syn_batch = next(synthetic_data_processing.SyntheticDataIterator(cfg, mesh))
    self.assertIn("loss_weights", shardings)
    self.assertEqual(set(shardings.keys()), set(shaped_batch.keys()))
    self.assertEqual(shaped_batch["loss_weights"].dtype, jnp.float32)
    self.assertEqual(shaped_batch["loss_weights"].shape, syn_batch["loss_weights"].shape)
    self.assertEqual(shaped_batch["loss_weights"].sharding, shardings["loss_weights"])
    self.assertEqual(set(shardings.keys()), set(ph_batch.keys()))
    self.assertEqual(set(shardings.keys()), set(syn_batch.keys()))
    self.assertEqual(syn_batch["loss_weights"].dtype, jnp.float32)
    patch_tok = int(cfg.olmo_image_patch_token_id)
    is_patch_tgt = np.asarray(syn_batch["targets"]) == patch_tok
    lw = np.asarray(syn_batch["loss_weights"])
    self.assertTrue(np.any(is_patch_tgt))
    self.assertTrue(np.any(~is_patch_tgt))
    np.testing.assert_array_equal(lw[is_patch_tgt], 0.0)
    np.testing.assert_array_equal(lw[~is_patch_tgt], 1.0)

  def test_group_loss_normalization_035_065_parity(self):
    # Verify OLMo-core MultimodalTrainModule._normalize_loss_groups parity in train.loss_fn:
    # With olmo_text_loss_group_weight=0.35 and olmo_vision_loss_group_weight=0.65, even when
    # sequence 0 (text-only) has 28 supervised tokens and sequence 1 (vision) has 6 supervised tokens,
    # total cross-entropy loss equals 0.35 * L_text + 0.65 * L_vision identically.
    def _make_cfg(w_txt: float, w_vis: float):
      return pyconfig.initialize([
          "",
          _BASE_YML,
          f"run_name=ut_gl_{w_txt}_{w_vis}",
          "base_output_directory=/tmp/ut_gl",
          "dataset_type=synthetic",
          "model_name=olmo35-tiny",
          "override_model_config=True",
          "base_emb_dim=128",
          "base_mlp_dim=256",
          "base_moe_mlp_dim=128",
          "moe_expert_input_dim=64",
          "base_num_decoder_layers=2",
          "inhomogeneous_layer_cycle_interval=2",
          "first_num_dense_layers=1",
          "num_experts=4",
          "num_experts_per_tok=2",
          "emo_enabled=False",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=32",
          "gdn_num_key_heads=4",
          "gdn_num_value_heads=4",
          "gdn_key_head_dim=32",
          "gdn_value_head_dim=32",
          "vocab_size=256",
          "max_target_length=32",
          "use_multimodal=True",
          "vision_encoder_block=olmo35",
          "image_size_for_vit=28",
          "patch_size_for_vit=14",
          "hidden_size_for_vit=64",
          "intermediate_size_for_vit=128",
          "num_attention_heads_for_vit=4",
          "num_hidden_layers_for_vit=2",
          "olmo_vit_feature_layers=[0,1]",
          "olmo_connector_mlp_hidden_size=128",
          "olmo_image_patch_token_id=250",
          "olmo_max_crops_per_seq=1",
          "olmo_max_pooled_patches_per_seq=4",
          "olmo_use_loss_weights=True",
          f"olmo_text_loss_group_weight={w_txt}",
          f"olmo_vision_loss_group_weight={w_vis}",
          "static_loss_cotangent=False",
          "z_loss_multiplier=0.0",
          "load_balance_loss_weight=0.0",
          "dtype=float32",
          "weight_dtype=float32",
          "attention=dot_product",
          "ici_data_parallelism=1",
          "ici_fsdp_parallelism=1",
          "ici_context_parallelism=1",
          "ici_expert_parallelism=1",
          "per_device_batch_size=2.0",
      ])

    cfg_mix = _make_cfg(0.35, 0.65)
    cfg_raw = _make_cfg(0.0, 0.0)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "expert"))
    with jax.set_mesh(mesh):
      model = models.Transformer(cfg_mix, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=11))
    rng = np.random.default_rng(7)
    inputs = rng.integers(1, 200, size=(2, 32), dtype=np.int32)
    inputs[1, 1:3] = 250  # 2 vision patch tokens in seq 1
    targets = rng.integers(1, 200, size=(2, 32), dtype=np.int32)
    seg = np.ones((2, 32), dtype=np.int32)
    pos = np.broadcast_to(np.arange(32, dtype=np.int32), (2, 32)).copy()
    lw = np.zeros((2, 32), dtype=np.float32)
    lw[0, :28] = 1.0  # 28 active text tokens
    lw[1, 10:16] = 1.0  # 6 active vision response tokens
    images = rng.standard_normal((2, 1, 4, 588)).astype(np.float32)
    p_idx = np.full((2, 4, 4), -1, dtype=np.int32)
    p_idx[1, :2, :] = np.array([[0, 1, 2, 3], [0, 1, -1, -1]], dtype=np.int32)
    batch = {
        "inputs": jnp.asarray(inputs),
        "targets": jnp.asarray(targets),
        "inputs_segmentation": jnp.asarray(seg),
        "targets_segmentation": jnp.asarray(seg),
        "inputs_position": jnp.asarray(pos),
        "targets_position": jnp.asarray(pos),
        "loss_weights": jnp.asarray(lw),
        "images": jnp.asarray(images),
        "pooled_patches_idx": jnp.asarray(p_idx),
    }

    with jax.set_mesh(mesh):
      loss_mix, _ = nnx.jit(lambda m, b: train.loss_fn(m, cfg_mix, b, None, None, is_train=True))(
          model, dict(batch)
      )
      # Text-only group loss L_txt (zero vision weights)
      lw_txt = lw.copy()
      lw_txt[1, :] = 0.0
      loss_txt, _ = nnx.jit(lambda m, b: train.loss_fn(m, cfg_raw, b, None, None, is_train=True))(
          model, dict(batch, loss_weights=jnp.asarray(lw_txt))
      )
      # Vision-only group loss L_vis (zero text weights)
      lw_vis = lw.copy()
      lw_vis[0, :] = 0.0
      loss_vis, _ = nnx.jit(lambda m, b: train.loss_fn(m, cfg_raw, b, None, None, is_train=True))(
          model, dict(batch, loss_weights=jnp.asarray(lw_vis))
      )

    expected = 0.35 * float(loss_txt) + 0.65 * float(loss_vis)
    self.assertAlmostEqual(float(loss_mix), expected, places=5)

  def test_checkpoint_converter_bridge_and_full_roundtrip(self):
    # Hermetic end-to-end test of `convert_olmo35_multimodal_checkpoint` covering:
    # 1) Text DCP (`module.<k>.main`) -> `bridge_init` -> MaxText Orbax (`<out>/0/items`)
    #    verifying exact reset of `<im_*>` rows while non-image embeddings and `logits_dense` stay exact.
    # 2) Full Multimodal DCP (`module.lm.<k>.main` + `frozen_model.lm.lm_head.w_out.weight` + `module.{vision,connector}.<k>.main`)
    #    -> `olmo_core_to_maxtext` (0.0 round-trip diff) -> `checkpointing.load_params_from_path` into `models.Transformer`
    #    -> `maxtext_to_olmo_core` (0.0 round-trip diff).
    import math
    import tempfile
    import torch
    from olmo_core.distributed.checkpoint import save_state_dict
    from olmo_core.nn.maxtext.checkpoint import write_json
    from olmo_core.nn.maxtext.config import get_maxtext_config
    from olmo_core.nn.maxtext.convert import olmo_core_shapes
    from maxtext.checkpoint_conversion import convert_olmo35_multimodal_checkpoint as conv
    from maxtext.common import checkpointing

    kda_mixer = {
        "type": "kimi_delta_attention",
        "n_heads": 4,
        "n_v_heads": 4,
        "head_dim": 32,
        "expand_v": 1.0,
        "allow_neg_eigval": True,
        "conv_size": 4,
        "conv_bias": False,
        "norm_eps": 1e-5,
    }
    full_mixer = {
        "type": "attention",
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 32,
        "bias": False,
        "gate": {"granularity": "elementwise", "full_precision": True},
        "qk_norm": {"name": "rms", "eps": 1e-6, "bias": False},
        "use_head_qk_norm": True,
        "scalable_softmax": True,
        "qk_norm_per_head_gains": True,
    }
    rms_norm = {"name": "rms", "eps": 1e-6, "bias": False}
    moe_kda_block = {
        "name": "moe_fused_v2",
        "sequence_mixer": kda_mixer,
        "layer_norm": rms_norm,
        "shared_experts": {"d_model": 128, "hidden_size": 128, "num_experts": 1, "bias": False, "activation": "swiglu"},
        "routed_experts": {"d_model": 64, "hidden_size": 128, "num_experts": 4, "bias": False, "activation": "swiglu"},
        "routed_experts_router": {
            "d_model": 128,
            "num_experts": 4,
            "top_k": 2,
            "bias": False,
            "normalize_expert_weights": 1.0,
            "gating_function": "softmax",
            "lb_loss_weight": 0.01,
            "z_loss_weight": 1e-5,
            "restore_weight_scale": True,
        },
        "latent_moe": {"latent_dim": 64, "bias": False, "up_proj_input_norm_enabled": False},
        "use_peri_norm": True,
        "use_pre_norm": False,
    }
    dense_kda_block = {
        "name": "moe_fused_v2",
        "sequence_mixer": kda_mixer,
        "layer_norm": rms_norm,
        "shared_experts": {"d_model": 128, "hidden_size": 256, "num_experts": 1, "bias": False, "activation": "swiglu"},
        "use_peri_norm": True,
        "use_pre_norm": False,
    }
    moe_full_block = {**moe_kda_block, "sequence_mixer": full_mixer}
    lm_cfg = {
        "name": "moe_fused_v2",
        "d_model": 128,
        "vocab_size": 256,
        "n_layers": 2,
        "embed_scale": math.sqrt(128.0),
        "tie_word_embeddings": False,
        "n_extra_vocab": 0,
        "embedding_norm": rms_norm,
        "lm_head": {"name": "default", "layer_norm": rms_norm, "bias": False},
        "block": moe_kda_block,
        "block_overrides": {"0": dense_kda_block, "1": moe_full_block},
    }
    vis_cfg = {
        "image_emb_dim": 64,
        "image_patch_size": 14,
        "image_num_pos": 4,
        "image_num_layers": 2,
        "image_num_heads": 4,
        "image_num_key_value_heads": 4,
        "image_head_dim": 16,
        "image_mlp_dim": 128,
    }
    conn_cfg = {
        "image_emb_dim": 64,
        "image_num_heads": 4,
        "image_num_key_value_heads": 4,
        "image_head_dim": 16,
        "num_input_layers": 2,
        "mlp_hidden_size": 128,
        "output_dim": 128,
    }
    mt_lm_cfg = get_maxtext_config(lm_cfg)
    lm_shapes = olmo_core_shapes(mt_lm_cfg)
    vis_shapes = conv.vision_and_connector_shapes(vis_cfg, conn_cfg)

    rng = torch.Generator().manual_seed(42)
    lm_tensors = {k: torch.randn(s, generator=rng, dtype=torch.float32) * 0.02 for k, s in lm_shapes.items()}
    vis_tensors = {k: torch.randn(s, generator=rng, dtype=torch.float32) * 0.02 for k, s in vis_shapes.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
      # 1. Text DCP -> Bridge Init (skip HF network download in unit test; test 25-layer real SigLIP2 separately)
      text_dcp = f"{tmpdir}/text_step"
      save_state_dict(
          f"{text_dcp}/model_and_optim",
          {f"module.{k}.main": v.flatten() for k, v in lm_tensors.items()},
          save_overwrite=True,
      )
      write_json(f"{text_dcp}/config.json", {"model": lm_cfg}, save_overwrite=True)

      bridge_out = f"{tmpdir}/mt_bridge"
      # Use 25 layers with skip_hf_download_for_test=True to exercise VisionEncoderConfig.siglip2_so400m_patch14_378()
      summary_b = conv.init_bridge_maxtext_checkpoint(
          bridge_out,
          checkpoint_path=text_dcp,
          vision_num_layers=2,
          connector_mlp_hidden_size=128,
          image_token_ids=(248, 249, 250, 251, 252, 253),
          seed=6198,
          scan_layers=True,
          validate=True,
          use_ocdbt=False,
          use_zarr3=False,
          skip_hf_download_for_test=True,
      )
      self.assertEqual(summary_b["roundtrip_max_abs_diff"], 0.0)
      restored_b = conv.load_maxtext_params_any(f"{bridge_out}/0/items")
      emb_b = restored_b["token_embedder/embedding"].numpy()
      orig_emb = lm_tensors["embeddings.weight"].numpy()
      np.testing.assert_allclose(emb_b[:248], orig_emb[:248], rtol=0.0, atol=0.0)
      np.testing.assert_allclose(emb_b[254:], orig_emb[254:], rtol=0.0, atol=0.0)
      self.assertGreater(float(np.max(np.abs(emb_b[248:254] - orig_emb[248:254]))), 1e-4)
      np.testing.assert_allclose(
          restored_b["decoder/logits_dense/kernel"].numpy(),
          lm_tensors["lm_head.w_out.weight"].numpy().T,
          rtol=0.0,
          atol=0.0,
      )

      # 2. Full Multimodal DCP (`module.lm.*.main` + `frozen_model.lm.lm_head.w_out.weight` + `module.{vision,connector}.*.main`)
      mm_dcp = f"{tmpdir}/mm_step3500"
      mm_state_dict = {}
      for k, v in lm_tensors.items():
        if k == "lm_head.w_out.weight":
          mm_state_dict["frozen_model.lm.lm_head.w_out.weight"] = v.clone()
        else:
          mm_state_dict[f"module.lm.{k}.main"] = v.flatten()
      for k, v in vis_tensors.items():
        mm_state_dict[f"module.{k}.main"] = v.flatten()
      mm_exp_cfg = {"model": {"lm": lm_cfg, "vision": vis_cfg, "connector": conn_cfg, "image_patch_token_id": 250}}
      save_state_dict(f"{mm_dcp}/model_and_optim", mm_state_dict, save_overwrite=True)
      write_json(f"{mm_dcp}/config.json", mm_exp_cfg, save_overwrite=True)

      mt_mm_out = f"{tmpdir}/mt_mm"
      summary_mm = conv.convert_olmo_core_to_maxtext(
          mm_dcp,
          mt_mm_out,
          scan_layers=True,
          validate=True,
          use_ocdbt=False,
          use_zarr3=False,
      )
      self.assertEqual(summary_mm["roundtrip_max_abs_diff"], 0.0)
      self.assertEqual(summary_mm["olmo_total_tensors"], len(lm_tensors) + len(vis_tensors))

      # 3. Verify `checkpointing.load_params_from_path` restores directly into MaxText `models.Transformer`!
      cfg_mt = pyconfig.initialize([
          "",
          _BASE_YML,
          "run_name=ut_ckpt_restore",
          f"base_output_directory={tmpdir}/out",
          "dataset_type=synthetic",
          "model_name=olmo35-tiny",
          "override_model_config=True",
          "base_emb_dim=128",
          "base_mlp_dim=256",
          "base_moe_mlp_dim=128",
          "moe_expert_input_dim=64",
          "base_num_decoder_layers=2",
          "inhomogeneous_layer_cycle_interval=2",
          "first_num_dense_layers=1",
          "num_experts=4",
          "num_experts_per_tok=2",
          "emo_enabled=False",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=32",
          "gdn_num_key_heads=4",
          "gdn_num_value_heads=4",
          "gdn_key_head_dim=32",
          "gdn_value_head_dim=32",
          "vocab_size=256",
          "max_target_length=32",
          "use_multimodal=True",
          "vision_encoder_block=olmo35",
          "image_size_for_vit=28",
          "patch_size_for_vit=14",
          "hidden_size_for_vit=64",
          "intermediate_size_for_vit=128",
          "num_attention_heads_for_vit=4",
          "num_hidden_layers_for_vit=2",
          "olmo_vit_feature_layers=[0,1]",
          "olmo_connector_mlp_hidden_size=128",
          "olmo_image_patch_token_id=250",
          "olmo_max_crops_per_seq=1",
          "olmo_max_pooled_patches_per_seq=4",
          "scan_layers=True",
          "dtype=float32",
          "weight_dtype=float32",
          "ici_data_parallelism=1",
          "ici_fsdp_parallelism=1",
          "ici_context_parallelism=1",
          "ici_expert_parallelism=1",
          "per_device_batch_size=1.0",
          "skip_jax_distributed_system=True",
      ])
      mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "expert"))
      from flax.core.spmd import logical_axis_rules
      from maxtext.optimizers import optimizers as mt_optimizers
      from maxtext.common import train_state_nnx
      tx = mt_optimizers.get_optimizer(cfg_mt, maxtext_utils.create_learning_rate_schedule(cfg_mt))
      def _init_fn():
        m = models.Transformer(cfg_mt, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
        return train_state_nnx.TrainStateNNX(m, nnx.Optimizer(m, tx, wrt=nnx.Param))
      unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(cfg_mt, mesh, _init_fn, is_training=True)
      with jax.set_mesh(mesh), logical_axis_rules(cfg_mt.logical_axis_rules):
        loaded_tree = checkpointing.load_params_from_path(
            f"{mt_mm_out}/0/items",
            unboxed_abstract_state.model,
            checkpoint_storage_concurrent_gb=4,
            use_ocdbt=False,
            use_zarr3=False,
        )
      flat_loaded = jax.tree_util.tree_leaves(loaded_tree.to_pure_dict())
      self.assertGreater(len(flat_loaded), 20)

      # 4. Reverse export `maxtext_to_olmo_core` and verify 0.0 round-trip diff
      rt_dcp = f"{tmpdir}/rt_olmo_dcp"
      summary_rt = conv.convert_maxtext_to_olmo_core(f"{mt_mm_out}/0/items", rt_dcp, validate=True)
      self.assertEqual(summary_rt["roundtrip_max_abs_diff"], 0.0)
      lm_rt, vis_rt = conv.load_olmo_core_checkpoint_weights(rt_dcp, mm_exp_cfg)
      for k in lm_tensors:
        np.testing.assert_allclose(lm_rt[k].numpy(), lm_tensors[k].numpy(), rtol=0.0, atol=0.0)
      for k in vis_tensors:
        np.testing.assert_allclose(vis_rt[k].numpy(), vis_tensors[k].numpy(), rtol=0.0, atol=0.0)

  def test_olmo_mm_batch_adapter_compact_and_npz_roundtrip(self):
    import tempfile
    from maxtext.input_pipeline import olmo_mm_data_processing as omm
    from maxtext.input_pipeline.input_pipeline_interface import PlaceHolderDataIterator
    from maxtext.utils import sharding as mt_sharding

    rng = np.random.default_rng(7)
    bsz, seq_in, target_seq = 4, 12, 16
    n_patches, patch_dim, pool_size = 4, 12, 4

    # Compact 3D images: seq 0 has 2 crops, seq 1 has 0 crops (text-only), seq 2 has 3 crops, seq 3 has 1 crop -> 6 total
    crop_counts = np.array([2, 0, 3, 1], dtype=np.int64)
    compact_imgs = rng.standard_normal((6, n_patches, patch_dim)).astype(np.float32)
    padded_imgs = np.zeros((bsz, 3, n_patches, patch_dim), dtype=np.float32)
    padded_imgs[0, :2] = compact_imgs[0:2]
    padded_imgs[2, :3] = compact_imgs[2:5]
    padded_imgs[3, :1] = compact_imgs[5:6]

    input_ids = rng.integers(10, 200, size=(bsz, seq_in), dtype=np.int64)
    labels = rng.integers(10, 200, size=(bsz, seq_in), dtype=np.int64)
    loss_masks = np.ones((bsz, seq_in), dtype=np.float32)
    # Mark token 4 in seq 1 as repetition-filtered (label=-100, loss_mask=1.0)
    labels[1, 4] = -100
    loss_masks[1, 4] = 1.0
    # Mark prompt tokens 0..2 as unsupervised (label=-100, loss_mask=0.0)
    labels[:, :2] = -100
    loss_masks[:, :2] = 0.0

    # Right padding on last 2 tokens of seq 3
    rtm = np.ones((bsz, seq_in), dtype=np.bool_)
    rtm[3, 10:] = False
    labels[3, 10:] = -100
    loss_masks[3, 10:] = 0.0

    doc_ids = np.zeros((bsz, seq_in), dtype=np.int64)
    doc_ids[0, 5:] = 1
    doc_ids[3, 10:] = -1

    pooled = np.full((bsz, 4, pool_size), -1, dtype=np.int64)
    pooled[0, :2] = [[0, 1, 2, 3], [4, 5, -1, -1]]
    pooled[2, :3] = [[0, 1, -1, -1], [4, 5, 6, 7], [8, -1, -1, -1]]
    pooled[3, :1] = [[0, 1, 2, 3]]

    raw_compact = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "router_token_mask": rtm,
        "document_ids": doc_ids,
        "images": compact_imgs,
        "image_crop_counts": crop_counts,
        "pooled_patches_idx": pooled,
    }
    raw_padded = dict(raw_compact, images=padded_imgs)

    mt_c = omm.olmo_core_batch_to_maxtext_batch(
        raw_compact,
        max_target_length=target_seq,
        max_crops_per_seq=4,
        max_pooled_patches_per_seq=6,
        n_patches_per_crop=n_patches,
        patch_dim=patch_dim,
        pool_size=pool_size,
        dtype_mm=jnp.bfloat16,
        reset_positions_on_segment=True,
        padding_input_token_id=100257,
        pad_token_id=100277,
    )
    mt_p = omm.olmo_core_batch_to_maxtext_batch(
        raw_padded,
        max_target_length=target_seq,
        max_crops_per_seq=4,
        max_pooled_patches_per_seq=6,
        n_patches_per_crop=n_patches,
        patch_dim=patch_dim,
        pool_size=pool_size,
        dtype_mm=jnp.bfloat16,
        reset_positions_on_segment=True,
        padding_input_token_id=100257,
        pad_token_id=100277,
    )
    for k in mt_c:
      np.testing.assert_array_equal(mt_c[k], mt_p[k], err_msg=f"Compact vs padded mismatch on {k}")

    # Verify padding_input_token_id replacement and repetition-filtered denominator preservation
    self.assertTrue(np.all(mt_c["inputs"][3, 10:] == 100257))
    self.assertTrue(np.all(mt_c["inputs"][0, 12:] == 100257))
    self.assertEqual(mt_c["targets_segmentation"][1, 4], 0)
    self.assertEqual(float(mt_c["loss_weights"][1, 4]), 1.0)
    self.assertEqual(float(mt_c["loss_weights"][1, 0]), 0.0)
    # Multi-doc segmentation reset on seq 0 at token 5
    np.testing.assert_array_equal(mt_c["inputs_segmentation"][0, :12], [1] * 5 + [2] * 7)
    np.testing.assert_array_equal(mt_c["inputs_position"][0, :12], list(range(5)) + list(range(7)))

    # Verify .npz save/load round-trip and multi-host slicing
    with tempfile.TemporaryDirectory() as tmpdir:
      p0 = os.path.join(tmpdir, "batch_00000.npz")
      p1 = os.path.join(tmpdir, "batch_00001.npz")
      omm.save_maxtext_batch_npz(p0, mt_c, metadata={"step": 0})
      mt_c2 = dict(mt_c, inputs=mt_c["inputs"] + 1)
      omm.save_maxtext_batch_npz(p1, mt_c2, metadata={"step": 1})

      loaded = omm.load_maxtext_batch_npz(
          p0, max_crops_per_seq=4, max_pooled_patches_per_seq=6, dtype_mm=jnp.bfloat16
      )
      for k in mt_c:
        np.testing.assert_array_equal(loaded[k], mt_c[k], err_msg=f"NPZ roundtrip mismatch on {k}")

      # Verify seekable byte-range reader matches fallback np.load bit-for-bit across row slices,
      # and gracefully falls back on strided slices or compressed archives.
      for sl_test in (slice(0, 1), slice(1, 3), slice(2, 4), slice(None)):
        fast_sub = omm._try_load_maxtext_batch_npz_seekable(  # pylint: disable=protected-access
            p0,
            max_crops_per_seq=4,
            max_pooled_patches_per_seq=6,
            mm_np_dtype=jnp.dtype(jnp.bfloat16),
            sl=sl_test,
            use_loss_weights=True,
            include_router_token_mask=True,
        )
        self.assertIsNotNone(fast_sub)
        with mock.patch.object(omm, "_try_load_maxtext_batch_npz_seekable", return_value=None):
          slow_sub = omm.load_maxtext_batch_npz(
              p0, max_crops_per_seq=4, max_pooled_patches_per_seq=6, dtype_mm=jnp.bfloat16, row_slice=sl_test
          )
        self.assertEqual(set(fast_sub.keys()), set(slow_sub.keys()))
        for k in slow_sub:
          np.testing.assert_array_equal(fast_sub[k], slow_sub[k], err_msg=f"Seekable vs fallback mismatch ({sl_test}, {k})")

      p0_comp = os.path.join(tmpdir, "batch_compressed.npz")
      with np.load(p0, allow_pickle=False) as z_raw:
        np.savez_compressed(p0_comp, **{k: z_raw[k] for k in z_raw.files})
      loaded_comp = omm.load_maxtext_batch_npz(
          p0_comp, max_crops_per_seq=4, max_pooled_patches_per_seq=6, dtype_mm=jnp.bfloat16, row_slice=slice(1, 3)
      )
      for k in mt_c:
        np.testing.assert_array_equal(loaded_comp[k], mt_c[k][1:3], err_msg=f"Compressed fallback mismatch on {k}")

      host1 = omm._PreExportedNpzHostLoader(  # pylint: disable=protected-access
          batches_dir=tmpdir,
          shard_index=1,
          shard_count=2,
          per_host_batch=2,
          max_crops_per_seq=4,
          max_pooled_patches_per_seq=6,
          dtype_mm=jnp.bfloat16,
          use_loss_weights=True,
          initial_batch_step=1,
      )
      b_step1 = next(host1)
      np.testing.assert_array_equal(b_step1["inputs"], mt_c2["inputs"][2:4])

    # Verify PlaceHolderDataIterator & sharding with dataset_type=olmo_mm
    cfg_ph = pyconfig.initialize([
        "",
        _BASE_YML,
        "run_name=ut_olmo_mm_ph",
        "base_output_directory=/tmp/ut_olmo_mm_ph",
        "dataset_type=olmo_mm",
        "model_name=olmo35-tiny",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "inhomogeneous_layer_cycle_interval=2",
        "use_multimodal=True",
        "vision_encoder_block=olmo35",
        "image_size_for_vit=28",
        "patch_size_for_vit=14",
        "olmo_max_crops_per_seq=4",
        "olmo_max_pooled_patches_per_seq=16",
        "olmo_use_loss_weights=True",
        "max_target_length=32",
        "per_device_batch_size=2.0",
        "ici_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        "ici_context_parallelism=1",
        "ici_expert_parallelism=1",
        "skip_jax_distributed_system=True",
    ])
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "expert"))
    ph = PlaceHolderDataIterator(cfg_ph, mesh)
    ph_batch = next(ph)
    sh_map = mt_sharding.get_input_data_sharding(cfg_ph, mesh)
    for req_key in ("inputs", "targets", "images", "pooled_patches_idx", "loss_weights", "router_token_mask"):
      self.assertIn(req_key, ph_batch)
      self.assertIn(req_key, sh_map)

    # Verify sub-1.0 per_device_batch_size keeps per_host_batch = global_load // num_procs
    # across active data-loading hosts so shard_count * per_host_batch == global_batch_size_to_train_on (never 4 // 32 = 0).
    from maxtext.input_pipeline import olmo_grain_data_processing as og_dp

    cfg_sub = pyconfig.initialize([
        "",
        _MIDTRAIN_256K_YML,
        "run_name=ut_sub_bsz",
        "base_output_directory=/tmp/ut_sub_bsz",
        "dataset_type=olmo_grain",
        "compile_topology=v4-1024",
        "compile_topology_num_slices=1",
        "ici_data_parallelism=2",
        "ici_context_parallelism=32",
        "ici_expert_parallelism=8",
        "per_device_batch_size=0.03125",
    ])
    self.assertEqual(cfg_sub.global_batch_size_to_train_on, 16)
    self.assertEqual(cfg_sub.global_batch_size_to_load, 512)
    with tempfile.TemporaryDirectory() as d_idx:
      from maxtext.input_pipeline.olmo_data import OlmoNpyFileEntry, OlmoNpyIndex

      arr_path = os.path.join(d_idx, "tok.npy")
      np.save(arr_path, np.arange(32 * 262144, dtype=np.uint32))
      idx_path = os.path.join(d_idx, "idx.json")
      OlmoNpyIndex(
          format_version="1",
          sequence_length=262144,
          dtype="uint32",
          tokenizer="test",
          files=(
              OlmoNpyFileEntry(
                  path=arr_path,
                  label="tok",
                  n_tokens=32 * 262144,
                  n_instances=32,
                  instance_offset=0,
              ),
          ),
          total_instances=32,
          total_tokens=32 * 262144,
      ).save(idx_path)
      cfg_sub_idx = pyconfig.initialize([
          "",
          _MIDTRAIN_256K_YML,
          "run_name=ut_sub_bsz2",
          "base_output_directory=/tmp/ut_sub_bsz2",
          "dataset_type=olmo_grain",
          f"olmo_index_path={idx_path}",
          "compile_topology=v4-1024",
          "compile_topology_num_slices=1",
          "ici_data_parallelism=2",
          "ici_context_parallelism=32",
          "ici_expert_parallelism=8",
          "per_device_batch_size=0.03125",
      ])
      with (
          mock.patch("jax.process_count", return_value=128),
          mock.patch("jax.process_index", return_value=2),
          mock.patch.object(og_dp, "_detect_resumed_step", return_value=0),
      ):
        ldr = og_dp._make_loader_for_host(cfg_sub_idx, process_indices=[0, 1, 2, 3], seed=0)  # pylint: disable=protected-access
        b_sub = next(iter(ldr))
        self.assertEqual(b_sub["inputs"].shape, (4, 262144))

  def test_merged_mt_lc_packed_fsl_shim_and_export_cli(self):
    import importlib.util
    import json
    import tempfile
    from maxtext.input_pipeline import olmo_mm_data_processing as omm

    try:
      from olmo_core.data.composable.instance_source import InstanceSourceConfig
      from olmo_core.data.tokenizer import TokenizerConfig
    except ImportError:
      self.skipTest("olmo_core not installed in test environment")

    omm.register_merged_mt_lc_shim()
    with tempfile.TemporaryDirectory() as tmpdir:
      tok_cfg = TokenizerConfig(vocab_size=100288, eos_token_id=100257, pad_token_id=100277)
      fsl_path = os.path.join(tmpdir, "fsl.npy")
      packed_path = os.path.join(tmpdir, "packed.npy")
      np.arange(1, 65, dtype=np.uint32).tofile(fsl_path)
      packed_arr = np.arange(100, 164, dtype=np.uint32)
      packed_arr[15::16] = 100257
      packed_arr.tofile(packed_path)

      interleaved_dict = {
          "_CLASS_": "merged_mt_lc.ProportionalInterleavedInstanceSourceConfig",
          "sources": [
              {
                  "_CLASS_": "merged_mt_lc.LegacyDataStreamInstanceSourceConfig",
                  "dataset": {
                      "_CLASS_": "olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig",
                      "paths": [fsl_path],
                      "sequence_length": 8,
                      "tokenizer": tok_cfg.as_config_dict(),
                      "work_dir": tmpdir,
                  },
                  "data_loader_seed": 42,
                  "legacy_global_batch_size": 16,
                  "output_sequence_length": 16,
                  "label": "fsl_8_to_16",
              },
              {
                  "_CLASS_": "merged_mt_lc.LegacyDataStreamInstanceSourceConfig",
                  "dataset": {
                      "_CLASS_": "olmo_core.data.numpy_dataset.NumpyPackedFSLDatasetConfig",
                      "paths": [packed_path],
                      "sequence_length": 16,
                      "tokenizer": tok_cfg.as_config_dict(),
                      "work_dir": tmpdir,
                  },
                  "data_loader_seed": 43,
                  "legacy_global_batch_size": 16,
                  "output_sequence_length": 16,
                  "label": "packed_16",
              },
          ],
          "label": "interleaved_test",
      }
      src_cfg = InstanceSourceConfig.from_dict(interleaved_dict)
      src = src_cfg.build(work_dir=tmpdir)
      self.assertGreater(len(src), 0)
      seen_chunk_mask = False
      for i in range(len(src)):
        item = src[i]
        self.assertEqual(item["input_ids"].shape, (16,))
        if "label_mask" in item and not bool(item["label_mask"][8]):
          seen_chunk_mask = True
      self.assertTrue(seen_chunk_mask)

      # Test tools/data_generation/export_olmo_mm_batches.py CLI exporter helper
      cli_path = os.path.abspath(
          os.path.join(os.path.dirname(__file__), "../../tools/data_generation/export_olmo_mm_batches.py")
      )
      spec = importlib.util.spec_from_file_location("export_olmo_mm_batches", cli_path)
      cli_mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(cli_mod)

      dummy_raw_batch = {
          "input_ids": np.arange(1, 33, dtype=np.int64).reshape(2, 16),
          "labels": np.arange(2, 34, dtype=np.int64).reshape(2, 16),
          "loss_masks": np.ones((2, 16), dtype=np.float32),
          "router_token_mask": np.ones((2, 16), dtype=np.bool_),
          "document_ids": np.zeros((2, 16), dtype=np.int64),
          "images": np.zeros((2, 4, 12), dtype=np.float32),
          "image_crop_counts": np.array([1, 1], dtype=np.int64),
          "pooled_patches_idx": np.full((2, 4, 4), -1, dtype=np.int64),
      }
      fake_raw_cfg = {
          "data_loader": {"sequence_length": 16, "pack_max_crops": 2},
          "model": {"vision": {"image_default_input_size": [4, 4], "image_patch_size": 2}},
          "train_module": {"padding_input_token_id": 100257},
      }

      class _FakeLoader:
        def __iter__(self):
          return iter([dummy_raw_batch, dummy_raw_batch])

        def reshuffle(self):
          pass

      out_dir = os.path.join(tmpdir, "exported_batches")
      with mock.patch.object(omm, "build_olmo_core_mixture_loader", return_value=(_FakeLoader(), fake_raw_cfg)) as mock_bld:
        manifest = cli_mod.export_batches(
            olmo_config_path=os.path.join(tmpdir, "dummy.json"),
            output_dir=out_dir,
            num_batches=2,
            pack_buffer_size=96,
            max_pooled_patches_per_seq=4,
        )
      mock_bld.assert_called_once()
      self.assertEqual(mock_bld.call_args.kwargs.get("pack_buffer_size"), 96)
      self.assertEqual(manifest["num_exported_batches"], 2)
      self.assertTrue(os.path.exists(os.path.join(out_dir, "batch_00000.npz")))
      self.assertTrue(os.path.exists(os.path.join(out_dir, "batch_00001.npz")))
      saved_manifest = json.loads(open(os.path.join(out_dir, "manifest.json")).read())
      self.assertEqual(saved_manifest["sequence_length"], 16)


if __name__ == "__main__":
  absltest.main()
