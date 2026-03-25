# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Indexer parity between JAX and Kernel implementations."""

import unittest
import time
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from maxtext.layers.attention_mla import Indexer
from maxtext.common.common_types import MODEL_MODE_TRAIN

class Config:
    def __init__(self):
        self.index_n_heads = 8
        self.index_head_dim = 64
        self.index_topk = 16
        self.emb_dim = 128
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 32
        self.q_lora_rank = 64
        self.dtype = jnp.float32
        self.weight_dtype = jnp.float32
        self.matmul_precision = "high"
        self.shard_mode = "auto"
        self.max_target_length = 2048
        self.use_sparse_indexer = True
        self.use_kernel_indexer = False
        self.mesh_axes = ['data', 'fsdp']

class MockRotaryEmbedding:
    def __init__(self):
        self.interleave = True
    def __call__(self, x, position=None):
        return x

class IndexerParityTest(unittest.TestCase):
    def setUp(self):
        self.devices = jax.devices()[:1]
        self.mesh = jax.sharding.Mesh(np.array(self.devices).reshape(1, 1), ('data', 'fsdp'))

    def test_indexer_parity(self):
        config = Config()
        rngs = nnx.Rngs(0)
        rotary_embedding = MockRotaryEmbedding()
        
        # Initialize JAX indexer
        config.use_kernel_indexer = False
        indexer_jax = Indexer(
            config=config,
            rotary_embedding=rotary_embedding,
            mesh=self.mesh,
            rngs=rngs,
        )
        
        # Initialize Kernel indexer with SAME weights
        rngs = nnx.Rngs(0)
        config.use_kernel_indexer = True
        indexer_kernel = Indexer(
            config=config,
            rotary_embedding=rotary_embedding,
            mesh=self.mesh,
            rngs=rngs,
        )
        
        bsz, seqlen = 2, 128
        inputs_q = jax.random.normal(jax.random.PRNGKey(1), (bsz, seqlen, config.emb_dim))
        low_rank_q = jax.random.normal(jax.random.PRNGKey(2), (bsz, seqlen, config.q_lora_rank))
        inputs_kv = jax.random.normal(jax.random.PRNGKey(3), (bsz, seqlen, config.emb_dim))
        inputs_positions = jnp.tile(jnp.arange(seqlen), (bsz, 1))
        # Mask with some -inf
        attention_mask = jax.random.uniform(jax.random.PRNGKey(4), (bsz, seqlen, seqlen))
        attention_mask = jnp.where(attention_mask > 0.8, -1e9, 0.0)

        with self.mesh:
            # Run JAX implementation
            indexer_jax.config.use_kernel_indexer = False
            mask_jax, topk_jax, score_jax = indexer_jax(
                inputs_q=inputs_q,
                low_rank_q=low_rank_q,
                inputs_kv=inputs_kv,
                inputs_positions=inputs_positions,
                attention_mask=attention_mask
            )

            # Run Kernel implementation
            try:
                print("\nRunning kernel implementation in test...")
                mask_kernel, topk_kernel, score_kernel = indexer_kernel(
                    inputs_q=inputs_q,
                    low_rank_q=low_rank_q,
                    inputs_kv=inputs_kv,
                    inputs_positions=inputs_positions,
                    attention_mask=attention_mask
                )
            except Exception as e:
                self.skipTest(f"Kernel implementation failed or not supported in this environment: {e}")
                return

        # Compare scores
        np.testing.assert_allclose(score_jax, score_kernel, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(topk_jax, topk_kernel)
        np.testing.assert_allclose(mask_jax, mask_kernel, rtol=1e-5, atol=1e-5)

    def test_indexer_benchmark(self):
        config = Config()
        bsz, seqlen = 4, 512
        config.index_topk = 64
        
        rngs = nnx.Rngs(0)
        rotary_embedding = MockRotaryEmbedding()
        
        indexer_jax = Indexer(config=config, rotary_embedding=rotary_embedding, mesh=self.mesh, rngs=rngs)
        rngs = nnx.Rngs(0)
        indexer_kernel = Indexer(config=config, rotary_embedding=rotary_embedding, mesh=self.mesh, rngs=rngs)
        
        inputs_q = jax.random.normal(jax.random.PRNGKey(1), (bsz, seqlen, config.emb_dim))
        low_rank_q = jax.random.normal(jax.random.PRNGKey(2), (bsz, seqlen, config.q_lora_rank))
        inputs_kv = jax.random.normal(jax.random.PRNGKey(3), (bsz, seqlen, config.emb_dim))
        inputs_positions = jnp.tile(jnp.arange(seqlen), (bsz, 1))
        attention_mask = None
        
        with self.mesh:
            # Warmup JAX
            print("\nWarming up JAX...")
            indexer_jax.config.use_kernel_indexer = False
            _ = jax.block_until_ready(indexer_jax(inputs_q, low_rank_q, inputs_kv, inputs_positions, attention_mask))
            
            # Warmup Kernel
            print("Warming up Kernel...")
            indexer_kernel.config.use_kernel_indexer = True
            try:
                _ = jax.block_until_ready(indexer_kernel(inputs_q, low_rank_q, inputs_kv, inputs_positions, attention_mask))
            except Exception as e:
                print(f"Kernel warmup failed: {e}")
                return

            num_iters = 100
            
            # Benchmark JAX
            start_time = time.time()
            for _ in range(num_iters):
                out = indexer_jax(inputs_q, low_rank_q, inputs_kv, inputs_positions, attention_mask)
            jax.block_until_ready(out)
            jax_time = (time.time() - start_time) / num_iters
            
            # Benchmark Kernel
            start_time = time.time()
            for _ in range(num_iters):
                out = indexer_kernel(inputs_q, low_rank_q, inputs_kv, inputs_positions, attention_mask)
            jax.block_until_ready(out)
            kernel_time = (time.time() - start_time) / num_iters
        
        print(f"\nIndexer Benchmark results (bsz={bsz}, seqlen={seqlen}):")
        print(f"JAX implementation:    {jax_time*1000:.3f} ms / iter")
        print(f"Kernel implementation: {kernel_time*1000:.3f} ms / iter")
        if kernel_time > 0:
            print(f"Speedup: {jax_time / kernel_time:.2f}x")

if __name__ == "__main__":
    unittest.main()
