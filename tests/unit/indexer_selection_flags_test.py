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

"""Tests for the DeepSeek-V4 CSA indexer selection flags."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import mesh_utils
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.sharding import Mesh

from maxtext import pyconfig
from maxtext.common.common_types import DEFAULT_MASK_VALUE
from maxtext.layers import initializers
from maxtext.layers.attention_compressed import DeepseekV4CSACompressor
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.nnx_decoders import compose_indexer_selection_policy

pytestmark = pytest.mark.cpu_only

BATCH = 2
SEQ = 64
RATE = 4


def make_config(**overrides):
  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "indexer_selection_flags_test",
      "enable_checkpointing": False,
      "max_target_length": 128,
      "base_emb_dim": 64,
      "head_dim": 64,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "q_lora_rank": 16,
      "indexer_n_heads": 2,
      "indexer_head_dim": 64,
      "indexer_topk": 8,
      "sliding_window_size": 8,
      "compress_ratios": [0, 0, 4, 128],
  }
  config_arguments.update(overrides)
  return pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], **config_arguments)


def make_csa_compressor(config, mesh=None):
  rotary = DeepSeekV4RotaryEmbedding(
      head_dim=config.head_dim,
      partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
      rope_theta=config.compressed_rope_max_timescale,
      fprop_dtype=config.dtype,
  )
  return DeepseekV4CSACompressor(
      config=config,
      compress_ratio=RATE,
      rotary_embedding=rotary,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs=nnx.Rngs(0),
      mesh=mesh,
  )


def make_inputs(config):
  rng = np.random.default_rng(0)
  hidden = jnp.array(rng.normal(size=(BATCH, SEQ, config.emb_dim)), dtype=jnp.float32)
  q_latent = jnp.array(rng.normal(size=(BATCH, SEQ, config.q_lora_rank)), dtype=jnp.float32)
  positions = jnp.broadcast_to(jnp.arange(SEQ)[None, :], (BATCH, SEQ))
  return hidden, q_latent, positions


def make_segment_mask():
  n_segments = 3
  ids = np.zeros((BATCH, SEQ), dtype=np.int32)
  bounds = np.linspace(0, SEQ, n_segments + 1).astype(int)
  for i in range(n_segments):
    ids[:, bounds[i] : bounds[i + 1]] = i + 1
  ids = jnp.array(ids)
  segment_mask = ids[:, :, None] == ids[:, None, :]
  additive = jnp.where(segment_mask, 0.0, DEFAULT_MASK_VALUE)
  return additive[:, :, ::RATE]


def count_top_k(closed_jaxpr):
  """Counts top_k equations in nested jaxprs."""

  def subjaxprs(v):
    if isinstance(v, ClosedJaxpr):
      yield v.jaxpr
    elif isinstance(v, Jaxpr):
      yield v
    elif isinstance(v, (list, tuple)):
      for x in v:
        yield from subjaxprs(x)
    elif isinstance(v, dict):
      for x in v.values():
        yield from subjaxprs(x)

  def walk(jaxpr):
    n = 0
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "top_k":
        n += 1
      for v in eqn.params.values():
        for sub in subjaxprs(v):
          n += walk(sub)
    return n

  return walk(closed_jaxpr.jaxpr)


class SaveSelectionPolicyTest(unittest.TestCase):

  def test_policy_composition_gating(self):
    base = jax.checkpoint_policies.save_only_these_names("query_proj")
    cfg_off = make_config()
    self.assertFalse(cfg_off.indexer_save_selection)
    self.assertFalse(cfg_off.indexer_sharded_topk)
    self.assertIs(compose_indexer_selection_policy(base, cfg_off), base)
    self.assertIsNone(compose_indexer_selection_policy(None, cfg_off))

    cfg_on = make_config(indexer_save_selection=True)
    self.assertIsNotNone(compose_indexer_selection_policy(None, cfg_on))

    offload = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=["query_proj"],
        offload_src="device",
        offload_dst="pinned_host",
    )
    combined = compose_indexer_selection_policy(offload, cfg_on)
    jaxpr = jax.make_jaxpr(lambda x: checkpoint_name(checkpoint_name(x, "query_proj"), "indexer_selection"))(jnp.ones(()))
    query_eqn, selection_eqn = jaxpr.jaxpr.eqns

    def apply_policy(policy, eqn):
      return policy(eqn.primitive, *(var.aval for var in eqn.invars), **eqn.params)

    self.assertEqual(apply_policy(combined, query_eqn), apply_policy(offload, query_eqn))
    self.assertTrue(apply_policy(combined, selection_eqn))


class SaveSelectionTest(unittest.TestCase):

  def _grad_fn_and_args(self, cfg):
    comp = make_csa_compressor(cfg)
    hidden, q_latent, positions = make_inputs(cfg)
    graphdef, state, rest = nnx.split(comp, nnx.Param, ...)
    policy = compose_indexer_selection_policy(None, cfg)

    def fwd(state_in, hidden_in, q_latent_in):
      m = nnx.merge(graphdef, state_in, rest)
      kv, mask = m(hidden_in, q_latent_in, positions, None)
      # Consume the mask the way attention does: the softmax vjp needs its primal output, so
      # without a saved selection the backward pass must recompute the whole indexer chain.
      q = hidden_in[..., : kv.shape[-1]]
      scores = jnp.einsum("bsd,bwkd->bksw", q, kv) + mask
      sink = jnp.zeros(scores.shape[:-1] + (1,), scores.dtype)
      probs = jax.nn.softmax(jnp.concatenate([scores, sink], axis=-1))
      return jnp.sum(probs * probs)

    grad_fn = jax.grad(jax.checkpoint(fwd, policy=policy), argnums=(0, 1, 2))
    return grad_fn, (state, hidden, q_latent)

  def test_grads_and_backward_topk_count(self):
    grad_off, args_off = self._grad_fn_and_args(make_config())
    grad_on, args_on = self._grad_fn_and_args(make_config(indexer_save_selection=True))

    g_off = jax.tree.leaves(grad_off(*args_off))
    g_on = jax.tree.leaves(grad_on(*args_on))
    self.assertEqual(len(g_off), len(g_on))
    for a, b in zip(g_off, g_on):
      np.testing.assert_array_equal(np.array(a), np.array(b))

    self.assertEqual(count_top_k(jax.make_jaxpr(grad_off)(*args_off)), 2)
    self.assertEqual(count_top_k(jax.make_jaxpr(grad_on)(*args_on)), 1)


class ShardedTopkTest(unittest.TestCase):

  def _run_single_device(self, packed):
    cfg_off = make_config()
    cfg_on = make_config(indexer_sharded_topk=True)
    mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("tensor",))
    comp_off = make_csa_compressor(cfg_off, mesh=mesh)
    comp_on = make_csa_compressor(cfg_on, mesh=mesh)

    hidden, q_latent, positions = make_inputs(cfg_off)
    attention_mask = make_segment_mask() if packed else None

    with nn_partitioning.axis_rules(cfg_on.logical_axis_rules), jax.set_mesh(mesh):
      sel_off = comp_off.indexer(hidden, q_latent, positions, attention_mask)
      sel_on = comp_on.indexer(hidden, q_latent, positions, attention_mask)
    self.assertEqual(sel_off.dtype, sel_on.dtype)
    np.testing.assert_array_equal(np.array(sel_off), np.array(sel_on))

  def test_single_device(self):
    self._run_single_device(packed=False)

  def test_single_device_packed(self):
    self._run_single_device(packed=True)

  def test_multidevice_tensor_mesh(self):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8").strip()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--sharded-topk-multidevice"],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _sharded_topk_multidevice_main():
  n_dev = jax.device_count()
  assert n_dev == 8, f"expected 8 forced CPU devices, got {n_dev}"
  mesh = Mesh(mesh_utils.create_device_mesh((n_dev,)), axis_names=("tensor",))
  cfg_off = make_config()
  cfg_on = make_config(indexer_sharded_topk=True)
  comp_off = make_csa_compressor(cfg_off, mesh=mesh)
  comp_on = make_csa_compressor(cfg_on, mesh=mesh)

  hidden, q_latent, positions = make_inputs(cfg_off)
  seg_mask = make_segment_mask()
  run_off = jax.jit(lambda h, q, p, m: comp_off.indexer(h, q, p, m))
  run_on = jax.jit(lambda h, q, p, m: comp_on.indexer(h, q, p, m))

  with nn_partitioning.axis_rules(cfg_on.logical_axis_rules), jax.set_mesh(mesh):
    for attention_mask in (None, seg_mask):
      sel_off = run_off(hidden, q_latent, positions, attention_mask)
      sel_on = run_on(hidden, q_latent, positions, attention_mask)
      assert sel_off.dtype == sel_on.dtype
      np.testing.assert_array_equal(np.array(sel_off), np.array(sel_on))


if __name__ == "__main__":
  if "--sharded-topk-multidevice" in sys.argv:
    _sharded_topk_multidevice_main()
  else:
    unittest.main()
