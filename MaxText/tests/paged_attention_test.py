import unittest

import jax
import jax.numpy as jnp
import flax.linen as nn
import common_types
import max_utils
import sys
import pyconfig
from jax import random
from layers.attentions import PagedAttentionOp, AttentionOp
import page_managers

Mesh = jax.sharding.Mesh

# Assuming you have your necessary imports and mock classes for Mesh, KVTensor, page_managers, etc.
# from your project. Replace the placeholders below with your actual imports.

# ... 

class PagedAttentionOpTest(unittest.TestCase):

    def setUp(self):
      # Create a mock Mesh object
      super().setUp()
      pyconfig.initialize(
          [sys.argv[0], "configs/base.yml"],
          per_device_batch_size=1.0,
          run_name="test",
          enable_checkpointing=False,
          max_target_length=128,
          max_prefill_predict_length=16,
      )
      self.seed = 0
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
      self.attention_type = self.cfg.attention_type

      self.num_pages: 2048
      self.page_size: 16 # 32, 64
      self.block_size: 512

      # Instantiate the PagedAttentionOp class with sample parameters
      # self.paged_attention_op = PagedAttentionOp(
      #     mesh=self.mesh,
      #     num_pages=128,
      #     page_size=16,
      #     max_pages_per_slot=64,
      #     max_pages_per_prefill=64,
      #     block_size=512,
      #     pages_per_compute_block=32,
      #     num_kv_heads=32,
      #     kv_head_dim_size=128,
      #     name="self_attention",
      # )
      self.paged_attention_op = PagedAttentionOp(
        mesh=self.mesh,
        num_pages=self.cfg.num_pages,
        page_size=self.cfg.page_size,
        max_pages_per_slot=self.cfg.max_target_length // self.cfg.page_size,
        max_pages_per_prefill=self.cfg.max_prefill_predict_length // self.cfg.page_size,
        block_size=self.cfg.block_size,
        pages_per_compute_block=self.cfg.block_size // self.cfg.page_size,
        num_kv_heads=self.num_kv_heads,
        kv_head_dim_size=self.head_dim,
        dtype=self.dtype,
      )
      self.attention_op = AttentionOp(
        mesh=self.mesh,
        attention_kernel="dot_product",
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        num_query_heads=self.cfg.num_query_heads,
        num_kv_heads=self.cfg.num_kv_heads,
        dtype=self.cfg.dtype,
        prefill_cache_axis_order=(0,1,2,3),
        ar_cache_axis_order=(0,1,2,3),
        compute_axis_order=(0,1,2,3),
        attention_type=self.cfg.attention_type,
      )
      self.page_state = page_managers.PageState


    # num_pages: 2048
    # page_size: 16 # 32, 64
    # block_size: 512
    
    # x = jnp.ones((16, 9))
    # model = Transformer()
    # variables = model.init(jax.random.key(0), x, method=Transformer.encode)

    # encoded = model.apply(variables, x, method=Transformer.encode)
    def test_init_or_get_kv_pages(self):
      # Apply the module with dummy inputs to initialize variables
      query = jnp.ones((2, 12, self.cfg.num_query_heads, 64))  # [batch, seq_len, num_heads, head_dim]
      key = jnp.ones((2, 12, self.cfg.num_kv_heads, 64))  # [batch, seq_len, num_kv_heads, head_dim]
      value = jnp.ones((2, 12, self.cfg.num_kv_heads, 64))
      decoder_segment_ids = None
      model_mode = common_types.MODEL_MODE_PREFILL
      params = self.paged_attention_op.init(jax.random.PRNGKey(0), query, key, value, decoder_segment_ids, model_mode, self.page_state)
      print(params)
      # attention_op = AttentionOp(mesh=self.mesh, 
      #                      attention_kernel="dot_product",
      #                      max_target_length=1024,
      #                      num_query_heads=8,
      #                      num_kv_heads=8)
      # output = self.attention_op.init_or_get_kv_pages("train")
      # print(output.shape)

      # key_pages_var = self.paged_attention_op.variables['cache']['key_pages']
      # value_pages_var = self.paged_attention_op.variables['cache']['value_pages']
      # variables = model.init(jax.random.key(0), common_types.MODEL_MODE_PREFILL, method=PagedAttentionOp.init_or_get_kv_pages)
      # results = model.apply(variables, common_types.MODEL_MODE_PREFILL, method=PagedAttentionOp.init_or_get_kv_pages)
      # Test for prefill mode
      # key_pages_var, value_pages_var = self.paged_attention_op.init_or_get_kv_pages(common_types.MODEL_MODE_PREFILL)
      # self.assertEqual(key_pages_var.value.shape, (32, 64, 16, 128)) 
      # self.assertEqual(value_pages_var.value.shape, (32, 64, 16, 128)) 

      # # Test for train/other modes (should use num_pages)
      # key_pages_var, value_pages_var = self.paged_attention_op.init_or_get_kv_pages("train") 
      # self.assertEqual(key_pages_var.value.shape, (32, 2048, 16, 128))
      # self.assertEqual(value_pages_var.value.shape, (32, 2048, 16, 128))

    def test_apply_attention_dot(self):
      key = random.PRNGKey(self.seed)
      k1, k2, k3 = random.split(key, 3)
      batch_size = 8
      seq_len = 1024
      head_dim = 128
      dtype = jnp.float32
      query = random.normal(k1, (batch_size, seq_len, self.num_query_heads, head_dim), dtype=dtype)
      key = random.normal(k2, (batch_size, seq_len, self.num_kv_heads, head_dim), dtype=dtype)
      value = random.normal(k3, (batch_size, seq_len, self.num_kv_heads, head_dim), dtype=dtype)

      paged_out, paged_max, paged_sum = self.paged_attention_op.apply_attention_dot(query, key, value, None) 
      original_out, original_max, original_sum = self.attention_op.apply_attention_dot(query, key, value, None)
      assert paged_out.shape == original_out.shape
      assert paged_max.shape == original_max.shape
      assert paged_sum.shape == original_sum.shape
      self.assertTrue(
        jax.numpy.allclose(paged_out, original_out, rtol=1e-02, atol=1e-02, equal_nan=False)
      )
      self.assertTrue(
        jax.numpy.allclose(paged_max, original_max, rtol=1e-02, atol=1e-02, equal_nan=False)
      )
      self.assertTrue(
        jax.numpy.allclose(paged_sum, original_sum, rtol=1e-02, atol=1e-02, equal_nan=False)
      )
      
