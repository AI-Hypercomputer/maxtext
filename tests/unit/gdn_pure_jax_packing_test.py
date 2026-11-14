# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Pure-JAX GatedDeltaNet sequence packing and block-doubling inverse."""

import functools
import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.attention import gdn_packing
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
import numpy as np


def _base_config_path() -> str:
  return os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")


class GdnPureJaxPackingTest(unittest.TestCase):
  """Verifies block-doubling inverse, segmented Conv1D, packed recurrence, and packing=False guards."""

  def test_block_doubling_unit_lower_inverse_matches_solve_triangular(self):
    rng = jax.random.PRNGKey(0)
    for chunk_size in (16, 24, 64):
      raw = jax.random.normal(rng, (2, 3, 4, chunk_size, chunk_size), dtype=jnp.float32) * (0.5 / np.sqrt(chunk_size))
      mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
      s_mat = jnp.where(mask, raw, 0.0)
      eye = jnp.broadcast_to(jnp.eye(chunk_size, dtype=jnp.float32), s_mat.shape)

      got = gdn_packing.block_doubling_unit_lower_inverse(s_mat, chunk_size)
      ref = jax.scipy.linalg.solve_triangular(eye + s_mat, eye, lower=True, unit_diagonal=True)
      np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

      prod = jnp.matmul(eye + s_mat, got, precision=jax.lax.Precision.HIGHEST)
      np.testing.assert_allclose(prod, eye, rtol=1e-5, atol=1e-5)

  def test_chunk_inverse_block_doubling_matches_solve_fwd_and_bwd(self):
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    b, s, h, d = 2, 48, 4, 16
    q = jax.random.normal(keys[0], (b, s, h, d), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (b, s, h, d), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (b, s, h, d), dtype=jnp.float32)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (b, s, h), dtype=jnp.float32))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, s, h), dtype=jnp.float32))

    def run(mode):
      def loss_fn(q_, k_, v_, g_, b_):
        out, _ = qwen3.jax_chunk_gated_delta_rule(
            q_,
            k_,
            v_,
            g_,
            b_,
            chunk_size=16,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
            chunk_inverse=mode,
        )
        return jnp.sum(out**2), out

      (_, out), grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(q, k, v, g, beta)
      return out, grads

    out_bd, grads_bd = run("block_doubling")
    out_sv, grads_sv = run("solve")
    np.testing.assert_allclose(out_bd, out_sv, rtol=1e-5, atol=1e-5)
    for g_bd, g_sv in zip(grads_bd, grads_sv):
      np.testing.assert_allclose(g_bd, g_sv, rtol=1e-4, atol=1e-4)

  def test_canonicalize_and_encode_segment_ids(self):
    raw = jnp.array([[0, 7, 7, 0, 7, 3, 3, 7, 0]], dtype=jnp.int32)
    canon = gdn_packing.canonicalize_segment_ids(raw)
    np.testing.assert_array_equal(canon, np.array([[0, 1, 1, 0, 1, 2, 2, 3, 0]], dtype=np.int32))

    enc_no_init = gdn_packing.encode_segment_ids(canon)
    np.testing.assert_array_equal(enc_no_init, np.array([[0, 1, 1, -1, 1, 2, 2, 3, -3]], dtype=np.float32))

    enc_init = gdn_packing.encode_segment_ids(canon, init_seg=jnp.array([1.0], dtype=jnp.float32))
    np.testing.assert_array_equal(enc_init, np.array([[-1, 1, 1, -1, 1, 2, 2, 3, -3]], dtype=np.float32))

  def test_segmented_conv1d_and_prefill_conv_state_extraction(self):
    k_qkv, k_w, k_state = jax.random.split(jax.random.PRNGKey(2), 3)
    dim = 12
    kernel_size = 4
    doc1 = jax.random.normal(k_qkv, (1, 5, dim), dtype=jnp.float32)
    doc2 = jax.random.normal(jax.random.PRNGKey(3), (1, 2, dim), dtype=jnp.float32)
    pad = jnp.ones((1, 2, dim), dtype=jnp.float32) * 99.0
    # Row layout: [pad(2), doc1(5), pad(2), doc2(2), pad(2)] with repeated raw segment ID 4!
    packed_qkv = jnp.concatenate([pad, doc1, pad, doc2, pad], axis=1)
    seg_ids = jnp.array([[0, 0, 4, 4, 4, 4, 4, 0, 0, 9, 9, 0, 0]], dtype=jnp.int32)
    packed_qkv_masked = jnp.where((seg_ids != 0)[..., None], packed_qkv, 0.0)
    weight = jax.random.normal(k_w, (kernel_size, 1, dim), dtype=jnp.float32)

    packed_out = qwen3.segmented_causal_depthwise_conv1d(
        packed_qkv_masked,
        weight,
        None,
        seg_ids,
        kernel_size=kernel_size,
        dtype=jnp.float32,
    )
    doc1_out = qwen3.segmented_causal_depthwise_conv1d(doc1, weight, jnp.ones((1, 5), dtype=jnp.int32), jnp.float32)
    doc2_out = qwen3.segmented_causal_depthwise_conv1d(doc2, weight, jnp.ones((1, 2), dtype=jnp.int32), jnp.float32)

    np.testing.assert_allclose(packed_out[:, 2:7, :], doc1_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(packed_out[:, 9:11, :], doc2_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(packed_out[:, :2, :], 0.0)
    np.testing.assert_array_equal(packed_out[:, 7:9, :], 0.0)
    np.testing.assert_array_equal(packed_out[:, 11:, :], 0.0)

    # Last valid document (doc2) has length 2 < kernel_size - 1 (= 3), so slot 0 of next_conv_state
    # must be zeroed (not leaked from doc1) while slots 1..2 hold doc2's two tokens.
    init_conv = jax.random.normal(k_state, (1, kernel_size - 1, dim), dtype=jnp.float32)
    next_conv = gdn_packing.segment_next_conv_state(
        init_conv, packed_qkv_masked, seg_ids, kernel_size, sequence_packing=True
    )
    np.testing.assert_array_equal(next_conv[:, 0, :], 0.0)
    np.testing.assert_allclose(next_conv[:, 1:, :], doc2, rtol=1e-6, atol=1e-6)

    # All-padding call preserves caller conv_state.
    all_pad_seg = jnp.zeros((1, 6), dtype=jnp.int32)
    preserved = gdn_packing.segment_next_conv_state(
        init_conv,
        jnp.zeros((1, 6, dim), dtype=jnp.float32),
        all_pad_seg,
        kernel_size,
        sequence_packing=True,
    )
    np.testing.assert_allclose(preserved, init_conv, rtol=1e-6, atol=1e-6)

    # Adjacent documents with extreme scale disparity must not suffer cross-document cancellation.
    doc_hi = jnp.ones((1, 4, dim), dtype=jnp.float32) * 1e6
    doc_lo = jax.random.normal(jax.random.PRNGKey(99), (1, 2, dim), dtype=jnp.float32) * 1e-3
    adj_seg = jnp.array([[1, 1, 1, 1, 2, 2]], dtype=jnp.int32)
    adj_out = qwen3.segmented_causal_depthwise_conv1d(
        jnp.concatenate([doc_hi, doc_lo], axis=1), weight, None, adj_seg, kernel_size=kernel_size, dtype=jnp.float32
    )
    doc_lo_ref = qwen3.segmented_causal_depthwise_conv1d(doc_lo, weight, jnp.ones((1, 2), dtype=jnp.int32), jnp.float32)
    np.testing.assert_allclose(adj_out[:, 4:, :], doc_lo_ref, rtol=1e-6, atol=1e-9)

  def test_packed_delta_rule_matches_per_document_single_and_cp(self):
    keys = jax.random.split(jax.random.PRNGKey(4), 16)
    h, d = 2, 8
    chunk_size = 8

    # Three documents of lengths 6, 11, 5 plus padding -> total 32 tokens (divisible by 4 CP ranks).
    lens = (6, 11, 5)
    docs = []
    for idx, doc_len in enumerate(lens):
      base = idx * 5
      docs.append(
          (
              jax.random.normal(keys[base + 0], (1, doc_len, h, d), dtype=jnp.float32),
              jax.random.normal(keys[base + 1], (1, doc_len, h, d), dtype=jnp.float32),
              jax.random.normal(keys[base + 2], (1, doc_len, h, d), dtype=jnp.float32),
              -jax.nn.softplus(jax.random.normal(keys[base + 3], (1, doc_len, h), dtype=jnp.float32)),
              jax.nn.sigmoid(jax.random.normal(keys[base + 4], (1, doc_len, h), dtype=jnp.float32)),
          )
      )
    init_h = jax.random.normal(keys[15], (1, h, d, d), dtype=jnp.float32) * 0.2

    def z_vec(n):
      return jnp.zeros((1, n, h, d), dtype=jnp.float32)

    def z_scl(n):
      return jnp.zeros((1, n, h), dtype=jnp.float32)

    # Layout (length 32): [pad(2), doc0(6), pad(3), doc1(11), pad(2), doc2(5), pad(3)]
    # Deliberately reuse raw ID 5 for doc0 and doc2 to exercise canonicalization across CP ranks!
    q_pack = jnp.concatenate(
        [
            z_vec(2),
            docs[0][0],
            z_vec(3),
            docs[1][0],
            z_vec(2),
            docs[2][0],
            z_vec(3),
        ],
        axis=1,
    )
    k_pack = jnp.concatenate(
        [
            z_vec(2),
            docs[0][1],
            z_vec(3),
            docs[1][1],
            z_vec(2),
            docs[2][1],
            z_vec(3),
        ],
        axis=1,
    )
    v_pack = jnp.concatenate(
        [
            z_vec(2),
            docs[0][2],
            z_vec(3),
            docs[1][2],
            z_vec(2),
            docs[2][2],
            z_vec(3),
        ],
        axis=1,
    )
    g_pack = jnp.concatenate(
        [
            z_scl(2),
            docs[0][3],
            z_scl(3),
            docs[1][3],
            z_scl(2),
            docs[2][3],
            z_scl(3),
        ],
        axis=1,
    )
    b_pack = jnp.concatenate(
        [
            z_scl(2),
            docs[0][4],
            z_scl(3),
            docs[1][4],
            z_scl(2),
            docs[2][4],
            z_scl(3),
        ],
        axis=1,
    )
    seg_ids = jnp.array(
        [[0] * 2 + [5] * 6 + [0] * 3 + [8] * 11 + [0] * 2 + [5] * 5 + [0] * 3],
        dtype=jnp.int32,
    )

    out_packed, final_packed = qwen3.jax_chunk_gated_delta_rule(
        q_pack,
        k_pack,
        v_pack,
        g_pack,
        b_pack,
        chunk_size=chunk_size,
        initial_state=init_h,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        segment_ids=seg_ids,
        has_initial_state=True,
    )

    out0, _ = qwen3.jax_chunk_gated_delta_rule(
        *docs[0],
        chunk_size=chunk_size,
        initial_state=init_h,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )
    out1, _ = qwen3.jax_chunk_gated_delta_rule(
        *docs[1],
        chunk_size=chunk_size,
        initial_state=None,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )
    out2, final2 = qwen3.jax_chunk_gated_delta_rule(
        *docs[2],
        chunk_size=chunk_size,
        initial_state=jnp.zeros_like(init_h),
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )

    np.testing.assert_allclose(out_packed[:, 2:8], out0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_packed[:, 11:22], out1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_packed[:, 24:29], out2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final_packed, final2, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out_packed[:, :2], 0.0)
    np.testing.assert_array_equal(out_packed[:, 8:11], 0.0)
    np.testing.assert_array_equal(out_packed[:, 29:], 0.0)

    # Verify 4-rank sequence-sharded CP matches single-device packed output, state, and gradients.
    devices = jax.devices()
    if len(devices) >= 4:
      mesh = Mesh(np.array(devices[:4]), ("context",))

      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(
              P(None, "context", None, None),
              P(None, "context", None, None),
              P(None, "context", None, None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, None, None, None),
              P(None, "context"),
          ),
          out_specs=(P(None, "context", None, None), P(None, None, None, None)),
          check_vma=False,
      )
      def cp_fn(q_, k_, v_, g_, b_, h0_, s_):
        return qwen3.jax_chunk_gated_delta_rule(
            q_,
            k_,
            v_,
            g_,
            b_,
            chunk_size=chunk_size,
            initial_state=h0_,
            use_qk_norm_in_gdn=True,
            cp_axis="context",
            compute_dtype=jnp.float32,
            segment_ids=s_,
            has_initial_state=True,
        )

      out_cp, final_cp = cp_fn(q_pack, k_pack, v_pack, g_pack, b_pack, init_h, seg_ids)
      np.testing.assert_allclose(out_cp, out_packed, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(final_cp, final_packed, rtol=1e-5, atol=1e-5)

      weights = jax.random.normal(jax.random.PRNGKey(99), out_packed.shape, dtype=jnp.float32)
      grads_ref = jax.grad(
          lambda q_, k_, v_, g_, b_, h0_: jnp.sum(
              qwen3.jax_chunk_gated_delta_rule(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  chunk_size=chunk_size,
                  initial_state=h0_,
                  use_qk_norm_in_gdn=True,
                  compute_dtype=jnp.float32,
                  segment_ids=seg_ids,
                  has_initial_state=True,
              )[0]
              * weights
          ),
          argnums=(0, 1, 2, 3, 4, 5),
      )(q_pack, k_pack, v_pack, g_pack, b_pack, init_h)
      grads_cp = jax.grad(
          lambda q_, k_, v_, g_, b_, h0_: jnp.sum(cp_fn(q_, k_, v_, g_, b_, h0_, seg_ids)[0] * weights),
          argnums=(0, 1, 2, 3, 4, 5),
      )(q_pack, k_pack, v_pack, g_pack, b_pack, init_h)
      for g_got, g_exp in zip(grads_cp, grads_ref):
        np.testing.assert_allclose(g_got, g_exp, rtol=1e-4, atol=1e-4)

  def test_qwen3_next_layer_packed_and_unpacked_fast_path(self):
    base_kwargs = {
        "run_name": "gdn_pack_test",
        "decoder_block": "qwen3_next",
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "base_emb_dim": 64,
        "gdn_key_head_dim": 16,
        "gdn_value_head_dim": 16,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 4,
        "gdn_conv_kernel_dim": 4,
        "gdn_chunk_size": 8,
        "max_target_length": 32,
        "per_device_batch_size": 1.0,
        "skip_jax_distributed_system": True,
    }
    cfg_packed = pyconfig.initialize([None, _base_config_path()], packing=True, **base_kwargs)
    devices_array = maxtext_utils.create_device_mesh(cfg_packed)
    mesh = Mesh(devices_array, cfg_packed.mesh_axes)

    layer = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_packed,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0),
    )

    k1, k2, k_pad = jax.random.split(jax.random.PRNGKey(7), 3)
    doc1 = jax.random.normal(k1, (1, 9, 64), dtype=jnp.float32)
    doc2 = jax.random.normal(k2, (1, 11, 64), dtype=jnp.float32)
    pad = jax.random.normal(k_pad, (1, 4, 64), dtype=jnp.float32) * 5.0
    packed_x = jnp.concatenate([pad[:, :2], doc1, pad[:, 2:], doc2, pad], axis=1)  # len = 28
    packed_seg = jnp.array([[0, 0] + [1] * 9 + [0, 0] + [2] * 11 + [0] * 4], dtype=jnp.int32)

    with jax.set_mesh(mesh):
      out_packed, _ = layer(packed_x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=packed_seg)
      out_doc1, _ = layer(
          doc1,
          model_mode=MODEL_MODE_TRAIN,
          decoder_segment_ids=jnp.ones((1, 9), dtype=jnp.int32),
      )
      out_doc2, _ = layer(
          doc2,
          model_mode=MODEL_MODE_TRAIN,
          decoder_segment_ids=jnp.ones((1, 11), dtype=jnp.int32),
      )

      np.testing.assert_allclose(out_packed[:, 2:11], out_doc1, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(out_packed[:, 13:24], out_doc2, rtol=1e-5, atol=1e-5)
      np.testing.assert_array_equal(out_packed[:, :2], 0.0)
      np.testing.assert_array_equal(out_packed[:, 11:13], 0.0)
      np.testing.assert_array_equal(out_packed[:, 24:], 0.0)

      # Single-sequence row with left+right padding must match identically under packing=False fast-path.
      single_x = jnp.concatenate([pad[:, :2], doc1, pad], axis=1)
      single_seg = jnp.array([[0, 0] + [1] * 9 + [0] * 4], dtype=jnp.int32)
      cfg_unpacked = pyconfig.initialize(
          [None, _base_config_path()],
          packing=False,
          **base_kwargs,
      )
      layer_unpacked = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_unpacked,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0),
      )
      nnx.update(layer_unpacked, nnx.state(layer))
      out_single_packed, _ = layer(single_x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=single_seg)
      out_single_unpacked, _ = layer_unpacked(single_x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=single_seg)
      np.testing.assert_array_equal(out_single_unpacked[:, :2], 0.0)
      np.testing.assert_allclose(out_single_unpacked[:, 2:11], out_single_packed[:, 2:11], rtol=1e-5, atol=1e-5)

      # Prefill mode with left+right padding updates cache states cleanly.
      layer_prefill = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_unpacked,
          mesh=mesh,
          model_mode=MODEL_MODE_PREFILL,
          rngs=nnx.Rngs(params=0),
          inputs_shape=single_x.shape,
      )
      nnx.update(layer_prefill, nnx.state(layer, nnx.Param))
      out_prefill, cache = layer_prefill(single_x, model_mode=MODEL_MODE_PREFILL, decoder_segment_ids=single_seg)
      self.assertIsNotNone(cache)
      np.testing.assert_allclose(out_prefill[:, 2:11], out_doc1, rtol=1e-5, atol=1e-5)

      # Explicit sharding mode must also compile and match packed outputs cleanly.
      cfg_explicit = pyconfig.initialize(
          [None, _base_config_path()],
          packing=True,
          shard_mode="explicit",
          ici_tensor_parallelism=2 if len(jax.devices()) >= 2 else 1,
          **base_kwargs,
      )
      explicit_devices = maxtext_utils.create_device_mesh(cfg_explicit)
      from jax.sharding import AxisType  # pylint: disable=import-outside-toplevel

      explicit_mesh = Mesh(
          explicit_devices, cfg_explicit.mesh_axes, axis_types=(AxisType.Explicit,) * len(cfg_explicit.mesh_axes)
      )
      with jax.set_mesh(explicit_mesh):
        layer_explicit = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_explicit,
            mesh=explicit_mesh,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=0),
        )
        out_explicit, _ = jax.jit(lambda m, x, s: m(x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=s))(
            layer_explicit, packed_x, packed_seg
        )
        np.testing.assert_allclose(out_explicit[:, 2:11], out_doc1, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_explicit[:, 13:24], out_doc2, rtol=1e-5, atol=1e-5)

  def test_packing_false_config_validations(self):
    with self.assertRaisesRegex(ValueError, "requires packing=True"):
      pyconfig.initialize(
          [
              None,
              _base_config_path(),
              "run_name=test_pack_guard",
              "decoder_block=qwen3_next",
              "packing=False",
              "dataset_type=grain",
              "grain_file_type=mmap",
              "grain_train_files=/tmp/dummy",
              "reset_attention_mask=True",
              "eval_interval=0",
              "skip_jax_distributed_system=True",
          ]
      )

    with self.assertRaisesRegex(ValueError, "requires packing=True"):
      config_types.RLConfig(
          model_name="qwen3-next-80b-a3b",
          decoder_block=config_types.DecoderBlockType.QWEN3_NEXT,
          tokenizer_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
          packing=False,
          max_seq_token_per_tpu=4096,
          max_target_length=2048,
      )


if __name__ == "__main__":
  unittest.main()
