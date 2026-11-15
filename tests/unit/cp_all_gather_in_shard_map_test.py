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

"""Tests `cp_all_gather_in_shard_map` for all-gather context parallel flash attention.

The real `AttentionOp.tpu_flash_attention` runs on a CPU mesh; only the Pallas Splash kernel is
replaced by a dense reference with the same mask semantics (global q positions from the mask's
`q_sequence`, sharded like the real kernel's metadata; causal + segment masking).
"""

import dataclasses
import sys
import unittest
from unittest import mock

from absl.testing import parameterized
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import attention_op as attention_op_lib
from maxtext.utils import max_utils, maxtext_utils
from tests.utils.test_helpers import get_test_config_path


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _RefSplashKernel:
  q_sequence: jax.Array

  def __call__(self, q, k, v, segment_ids, sinks=None, max_logit_value=None):
    del sinks, max_logit_value
    rep = q.shape[0] // k.shape[0]
    k = jnp.repeat(k, rep, axis=0)
    v = jnp.repeat(v, rep, axis=0)
    logits = jnp.einsum("hqd,hkd->hqk", q, k, precision="highest")
    mask = self.q_sequence[:, None] >= jnp.arange(k.shape[1])[None, :]
    if segment_ids is not None:
      mask &= segment_ids.q[:, None] == segment_ids.kv[None, :]
    p = jax.nn.softmax(jnp.where(mask[None], logits, -1e30), axis=-1)
    return jnp.einsum("hqk,hkd->hqd", p, v, precision="highest")


def _fake_make_splash_mha(mask, config, q_seq_shards, **_):
  del config, q_seq_shards
  q_seq = getattr(mask, "q_sequence", None)
  if q_seq is None:
    q_seq = np.arange(mask.shape[0])
  return _RefSplashKernel(jnp.asarray(np.asarray(q_seq), dtype=jnp.int32))


class CpAllGatherInShardMapTest(parameterized.TestCase):

  SEQ = 256

  def _run(self, flag, load_balance, data, context):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="cp_all_gather_in_shard_map_test",
        enable_checkpointing=False,
        attention="flash",
        use_tokamax_splash=True,
        dtype="float32",
        ici_data_parallelism=data,
        ici_fsdp_parallelism=1,
        ici_context_parallelism=context,
        context_parallel_load_balance=load_balance,
        cp_all_gather_in_shard_map=flag,
        max_target_length=self.SEQ,
        per_device_batch_size=1,
        base_num_query_heads=4,
        base_num_kv_heads=2,
        head_dim=16,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    op = attention_op_lib.AttentionOp(
        config=cfg,
        num_query_heads=4,
        num_kv_heads=2,
        max_target_length=self.SEQ,
        mesh=mesh,
        attention_kernel="flash",
        dtype=jnp.float32,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(keys[0], (data, self.SEQ, 4, 16))
    k = jax.random.normal(keys[1], (data, self.SEQ, 2, 16))
    v = jax.random.normal(keys[2], (data, self.SEQ, 2, 16))
    dout = jax.random.normal(keys[3], (data, self.SEQ, 4, 16))
    seg = np.ones((self.SEQ,), np.int32)
    seg[19:90] = 2
    seg[90:200] = 3
    seg[200:] = 4
    seg = jnp.broadcast_to(jnp.asarray(seg), (data, self.SEQ))
    if load_balance:
      q, k, v, dout, seg = (max_utils.reorder_sequence(t, context, seq_dim=1) for t in (q, k, v, dout, seg))

    with mock.patch.object(attention_op_lib.tokamax_splash_kernel, "make_splash_mha", _fake_make_splash_mha):
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        spec = nn_partitioning.logical_to_mesh_axes(("activation_batch", "activation_length", None, None))
        seg_spec = nn_partitioning.logical_to_mesh_axes(("activation_batch", "activation_length"))
        q, k, v, dout = (jax.device_put(t, NamedSharding(mesh, spec)) for t in (q, k, v, dout))
        seg = jax.device_put(seg, NamedSharding(mesh, seg_spec))

        def loss(q, k, v):
          out, _ = op.tpu_flash_attention(q, k, v, seg)
          return jnp.sum(out * dout), out

        grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))
        (_, out), grads = grad_fn(q, k, v)
        hlo = grad_fn.lower(q, k, v).compile().as_text()
    return [np.asarray(x) for x in (out, *grads)], hlo

  @parameterized.named_parameters(
      ("cp8", False, 1, 8),
      ("cp8_load_balanced", True, 1, 8),
      ("dp2_cp4", False, 2, 4),
      ("dp2_cp4_load_balanced", True, 2, 4),
  )
  def test_matches_default_and_uses_reduce_scatter(self, load_balance, data, context):
    if jax.device_count() < data * context:
      self.skipTest(f"needs {data * context} devices")
    base, base_hlo = self._run(False, load_balance, data, context)
    new, new_hlo = self._run(True, load_balance, data, context)
    for a, b in zip(new, base):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    self.assertNotIn("reduce-scatter(", base_hlo)
    self.assertEqual(new_hlo.count("reduce-scatter("), 2)  # dK and dV


if __name__ == "__main__":
  unittest.main()
