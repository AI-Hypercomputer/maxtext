#  Copyright 2026 Google LLC
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

"""GPU regression test for TransformerEngine sliding-window attention."""

import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import AttentionType, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils import maxtext_utils

from tests.utils.test_helpers import get_test_config_path


BATCH = 2
SEQ_LEN = 2048
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 256
WINDOW = 1024
TOL = 0.02


def _reference_attention(q, k, v):
  """Float32 dense reference with causal sliding-window masking."""
  qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
  group = q.shape[2] // k.shape[2]
  kf = jnp.repeat(kf, group, axis=2)
  vf = jnp.repeat(vf, group, axis=2)
  logits = jnp.einsum("bqhd,bkhd->bhqk", qf, kf)
  positions = jnp.arange(SEQ_LEN)
  mask = (positions[:, None] >= positions[None, :]) & (
      positions[None, :] > positions[:, None] - WINDOW
  )
  logits = jnp.where(mask[None, None], logits, -1e30)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum("bhqk,bkhd->bqhd", probs, vf)


class GpuCudnnFlashAttentionTest(unittest.TestCase):
  """Compares TE fused attention with a dense reference past the window boundary."""

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "gpu":
      self.skipTest("GPU-only test")

  @pytest.mark.gpu_only
  def test_local_sliding_head_dim_256(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="gpu_cudnn_flash_attention_test",
        enable_checkpointing=False,
        max_target_length=SEQ_LEN,
        per_device_batch_size=float(BATCH),
        head_dim=HEAD_DIM,
        attention="cudnn_flash_te",
        attention_type=AttentionType.LOCAL_SLIDING.value,
        sliding_window_size=WINDOW,
        packing=False,
        skip_jax_distributed_system=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    keys = jax.random.split(jax.random.PRNGKey(42), 4)
    q = jax.random.normal(
        keys[0], (BATCH, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM), dtype=jnp.bfloat16
    ) * (HEAD_DIM**-0.5)
    k = jax.random.normal(keys[1], (BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(
        keys[3], (BATCH, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM), dtype=jnp.bfloat16
    )
    segment_ids = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(SEQ_LEN, dtype=jnp.int32), (BATCH, SEQ_LEN))

    def reference_loss(q, k, v):
      out = _reference_attention(q, k, v)
      return jnp.sum(out * cotangent.astype(jnp.float32)), out

    def te_loss(q, k, v):
      op = AttentionOp(
          config=config,
          mesh=mesh,
          attention_kernel="cudnn_flash_te",
          max_target_length=SEQ_LEN,
          num_query_heads=NUM_Q_HEADS,
          num_kv_heads=NUM_KV_HEADS,
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=WINDOW,
          dtype=jnp.bfloat16,
          rngs=nnx.Rngs(0),
      )
      out = op(q, k, v, segment_ids, positions, MODEL_MODE_TRAIN)
      return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

    (_, reference_out), reference_grads = jax.jit(
        jax.value_and_grad(reference_loss, argnums=(0, 1, 2), has_aux=True)
    )(q, k, v)
    (_, te_out), te_grads = jax.jit(
        jax.value_and_grad(te_loss, argnums=(0, 1, 2), has_aux=True)
    )(q, k, v)

    def assert_close(reference, actual, name):
      reference = np.asarray(reference, dtype=np.float32)
      actual = np.asarray(actual, dtype=np.float32)
      scale = np.abs(reference).max()
      error = np.abs(actual - reference).max()
      self.assertLess(error, TOL * scale, f"{name}: max_err={error:.4f} ref_scale={scale:.4f}")

    assert_close(reference_out, te_out, "forward")
    for name, reference_grad, te_grad in zip(("dq", "dk", "dv"), reference_grads, te_grads):
      assert_close(reference_grad, te_grad, name)


if __name__ == "__main__":
  unittest.main()
