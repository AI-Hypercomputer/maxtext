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
"""Engram tests that need neither torch nor a download, so CI runs them.

The comparisons against the PyTorch reference implementation are in engram_vs_reference_test.py, which pytest.ini
keeps out of CI. The ShortConv tests need a device per context-parallel rank; on CPU run with
XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""

import functools
import os

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.layers.engram import NgramHashMapping, ShortConv
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path

LAYER_IDS = [1, 3]


def build_hash_mapping(tokenizer):
  return NgramHashMapping(
      engram_vocab_bases=[1009, 1009],
      max_ngram_size=3,
      engram_num_heads=2,
      layer_ids=LAYER_IDS,
      tokenizer=tokenizer,
      pad_id=tokenizer.pad_token_id,
      seed=0,
  )


class NgramHashMappingTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"))
    cls.hash_mapping = build_hash_mapping(cls.tokenizer)
    cls.input_ids = np.random.default_rng(0).integers(0, len(cls.tokenizer), (4, 128))

  @parameterized.named_parameters(("cp2", 2), ("cp4", 4))
  def test_load_balanced(self, cp_size):
    """Sequences permuted for context-parallel load balancing keep each token's natural-order hashes."""
    expected = self.hash_mapping(self.input_ids)
    # The training input pipeline hands the model DUAL_CHUNK_SWAP-permuted sequences.
    input_ids = max_utils.reorder_sequence(self.input_ids, cp_size=cp_size)
    actual = self.hash_mapping(input_ids, load_balanced_cp_size=cp_size)
    for layer_id in LAYER_IDS:
      natural = max_utils.reorder_sequence(actual[layer_id], cp_size=cp_size, to_contiguous=True)
      np.testing.assert_array_equal(natural, expected[layer_id])

  def test_built_under_jit(self):
    """The training loop builds the model under jit (`initialize_state`) and runs it in later traces."""
    built = {}

    @jax.jit
    def build():
      built["hash_mapping"] = build_hash_mapping(self.tokenizer)
      return jnp.zeros(())

    build()
    # Raises UnexpectedTracerError if construction kept a jax array from the finished trace.
    actual = jax.jit(built["hash_mapping"])(self.input_ids)
    expected = self.hash_mapping(self.input_ids)
    for layer_id in LAYER_IDS:
      np.testing.assert_array_equal(actual[layer_id], expected[layer_id])


class ShortConvTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("cp2", 2, 4),
      ("cp4", 4, 4),
      ("cp2_kernel1", 2, 1),  # a width-1 kernel reads no history
  )
  def test_load_balanced(self, cp_size, kernel_size, mhc_expansion_rate=4, hidden_size=16):
    """Sequences permuted for context-parallel load balancing get natural-order outputs and gradients."""
    if jax.device_count() < cp_size:
      self.skipTest(f"Needs {cp_size} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={cp_size}).")
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="engram_test",
        enable_checkpointing=False,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        ici_context_parallelism=cp_size,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    # Dilation 3 as in Engram (max_ngram_size): each output reads 3 * (kernel_size - 1) tokens of history.
    model = ShortConv(
        cfg, hidden_size, kernel_size, 3, mhc_expansion_rate=mhc_expansion_rate, rngs=nnx.Rngs(0), mesh=mesh
    )
    # The kernel is zero-initialized, which would hide any mixing of tokens.
    kernel = model.conv.kernel
    kernel[...] = jax.random.normal(jax.random.key(0), kernel.shape, jnp.float32)
    # Input and output cotangent, shape [B, S, G, D]
    x, w = jax.random.normal(jax.random.key(1), (2, 4, 128, mhc_expansion_rate, hidden_size), jnp.float32)

    def loss_fn(model, x, w, load_balanced_cp_size):
      y = model(x, load_balanced_cp_size=load_balanced_cp_size)
      return jnp.sum(y * w), y

    grad_fn = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (_, y), (param_grads, x_grads) = grad_fn(model, x, w, 1)
    with jax.set_mesh(mesh):
      # The training input pipeline hands the model DUAL_CHUNK_SWAP-permuted sequences, sharded over context.
      # The batch is sharded too, as in training: XLA scales nnx.Conv's kernel gradient by the replica count
      # when the batch is replicated over a mesh axis while the sequence is sharded.
      x_lb, w_lb = (
          jax.device_put(max_utils.reorder_sequence(a, cp_size=cp_size), NamedSharding(mesh, P("fsdp", "context")))
          for a in (x, w)
      )
      (_, y_lb), (param_grads_lb, x_grads_lb) = grad_fn(model, x_lb, w_lb, cp_size)

    to_natural = functools.partial(max_utils.reorder_sequence, cp_size=cp_size, to_contiguous=True)
    assert_close = functools.partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-4)
    assert_close(to_natural(y_lb), y)
    assert_close(to_natural(x_grads_lb), x_grads)
    # Parameter gradients sum over all tokens, in a different order on each device.
    jax.tree.map(assert_close, param_grads_lb, param_grads)
