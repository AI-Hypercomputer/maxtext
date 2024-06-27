#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for Attentions."""

import itertools
import random
import sys
import unittest

import common_types

from flax.core import freeze
import jax
import jax.numpy as jnp
import max_utils
import numpy as np
import pytest

import pyconfig

from layers import attentions

Mesh = jax.sharding.Mesh
Attention = attentions.Attention


class AttentionTest(unittest.TestCase):
  """Test for the Attention"""

  def setUp(self):
    super().setUp()
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
    )
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_kv_heads = self.cfg.num_kv_heads
    self.num_query_heads = self.cfg.num_query_heads
    self.max_target_length = self.cfg.max_target_length
    self.max_prefill_predict_length = self.cfg.max_prefill_predict_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype

    self._attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    self._attention_as_mha_generic_variable = self._attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

  def get_data(self, dtype):
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, dtype):
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(self.max_target_length, dtype=jnp.int32) for _ in range(self.global_batch_size)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((self.global_batch_size, self.max_target_length)) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions

  @pytest.mark.tpu
  def test_autoregression(self):
    prefill_length = self.cfg.max_prefill_predict_length
    decode_total_length = self.cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)

    mha_full = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mha_prefill, output_cache = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mha_prefill, mha_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      self._attention_as_mha_generic_variable.update(output_cache)
      mha_idx, output_cache = self._attention_as_mha_generic.apply(
          self._attention_as_mha_generic_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      mha_full_this_idx = mha_full[:, idx : idx + 1, :]
      self.assertTrue(mha_full_this_idx.shape == mha_idx.shape)
      self.assertTrue(jax.numpy.allclose(mha_full_this_idx, mha_idx, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu
  def test_tpu_kernel_attention_mha(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads)

  @pytest.mark.tpu
  def test_tpu_kernel_attention_gqa(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads // 2)

  @pytest.mark.tpu
  def test_tpu_kernel_attention_mqa(self):
    self.tpu_kernel_attention_helper(1)

  def tpu_kernel_attention_helper(self, num_kv_heads):
    """Test equalvant between dot_product and TPU accelerated"""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)

    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_output = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_as_mha_flash = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_flash_variable = attention_as_mha_flash.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_output = attention_as_mha_flash.apply(
        attention_as_mha_flash_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False)
    )

  @pytest.mark.tpu
  def test_dot_product_cache_axis_order(self):
    all_axis_orders = [axis_order for axis_order in itertools.permutations(range(4))]
    for axis_order in random.choices(all_axis_orders, k=4):
      self.dot_product_attention_helper(
        prefill_cache_axis_order=axis_order,
        ar_cache_axis_order=axis_order
      )
      print(f"passed test for {axis_order=}")

  def dot_product_attention_helper(self, prefill_cache_axis_order, ar_cache_axis_order):
    for compute_axis_order in [(0,1,2,3), (0,2,1,3)]:
      self._dot_product_attention(
        prefill_cache_axis_order,
        ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
      )
      print(f"passed subtest for {compute_axis_order=}")

  def _dot_product_attention(
      self,
      prefill_cache_axis_order,
      ar_cache_axis_order,
      compute_axis_order,
  ):
    """Test equalvant between different layout control in dot_product"""

    rtol, atol = 1e-02, 1e-02

    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention="dot_product",
    )
    config = pyconfig.config

    prefill_length = config.max_prefill_predict_length
    decode_total_length = config.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(config.dtype)

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    attention_w_layout = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
    )
    attention_w_layout_variable = attention_w_layout.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )
    attention_w_layout_full = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_w_layout_prefill, attention_w_layout_output_cache = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_layout_full[:, :prefill_length, :], attention_w_layout_prefill, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):

      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      
      attention_w_layout_variable.update(attention_w_layout_output_cache)
      attention_w_layout_idx, attention_w_layout_output_cache = attention_w_layout.apply(
          attention_w_layout_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_w_layout_full_this_idx = attention_w_layout_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_layout_full_this_idx.shape == attention_w_layout_idx.shape)
      self.assertTrue(jax.numpy.allclose(attention_w_layout_full_this_idx, attention_w_layout_idx, rtol=rtol, atol=atol, equal_nan=False))

  @pytest.mark.tpu
  def test_dot_product_reshape_q(self):
    for compute_axis_order in [(0,1,2,3), (0,2,1,3)]:
      self._dot_product_attention_reshape_q(
        compute_axis_order=compute_axis_order,
      )
      print(f"test passed for compute_axis_order: {compute_axis_order}")

  def _dot_product_attention_reshape_q(self, compute_axis_order):
    """Test equalvant between q and reshape q in dot_product"""

    rtol, atol = 1e-02, 1e-02

    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention="dot_product",
    )
    config = pyconfig.config

    prefill_length = config.max_prefill_predict_length
    decode_total_length = config.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(config.dtype)

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    attention_wo_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=False,
    )
    attention_wo_reshape_q_variable = attention_wo_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )

    attention_w_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=True,
    )
    attention_w_reshape_q_variable = attention_w_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )

    attention_wo_reshape_q_full = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_w_reshape_q_full = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_wo_reshape_q_prefill, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_wo_reshape_q_full[:, :prefill_length, :], attention_wo_reshape_q_prefill, equal_nan=False)
    )

    attention_w_reshape_q_prefill, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_reshape_q_full[:, :prefill_length, :], attention_w_reshape_q_prefill, equal_nan=False)
    )

    self.assertTrue(
        jax.numpy.allclose(attention_wo_reshape_q_prefill, attention_w_reshape_q_prefill, equal_nan=False)
    )
    self.assertTrue(
        jax.numpy.allclose(attention_wo_reshape_q_full[:, :prefill_length, :], attention_w_reshape_q_full[:, :prefill_length, :], equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):

      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      
      attention_wo_reshape_q_variable.update(attention_wo_reshape_q_output_cache)
      attention_wo_reshape_q_idx, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
          attention_wo_reshape_q_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_wo_reshape_q_full_this_idx = attention_wo_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_wo_reshape_q_full_this_idx.shape == attention_wo_reshape_q_idx.shape)
      self.assertTrue(jax.numpy.allclose(attention_wo_reshape_q_full_this_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False))

      attention_w_reshape_q_variable.update(attention_w_reshape_q_output_cache)
      attention_w_reshape_q_idx, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
          attention_w_reshape_q_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_w_reshape_q_full_this_idx = attention_w_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_reshape_q_full_this_idx.shape == attention_w_reshape_q_idx.shape)
      self.assertTrue(jax.numpy.allclose(attention_w_reshape_q_full_this_idx, attention_w_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False))

      self.assertTrue(jax.numpy.allclose(attention_w_reshape_q_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
