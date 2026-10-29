# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi-K3 KDA on the tokamax kernel (`kda_kernel='tokamax'`) vs the in-tree JAX chunked core.

Skipped unless the installed tokamax provides `tokamax.ops.kimi_delta_attention`.
"""

import math
import sys
import unittest

from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.layers import attention_kda
from maxtext.layers.normalizations import l2norm
from tests.utils.test_helpers import get_test_config_path

try:
  attention_kda._tokamax_kda_ops()  # pylint: disable=protected-access
  _HAS_TOKAMAX_KDA = True
except ImportError:
  _HAS_TOKAMAX_KDA = False

_LB = -5.0
_ORIG_MATMUL_PRECISION = None


def setUpModule():
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")
  jax.config.update("jax_platforms", "cpu")


def tearDownModule():
  if _ORIG_MATMUL_PRECISION is not None:
    jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


def _inputs(b, t, h, d, seed=0):
  ks = jax.random.split(jax.random.PRNGKey(seed), 7)
  return {
      "q": jax.random.normal(ks[0], (b, t, h, d)),
      "k": jax.random.normal(ks[1], (b, t, h, d)),
      "v": jax.random.normal(ks[2], (b, t, h, d)),
      "g_raw": 2.0 * jax.random.normal(ks[3], (b, t, h, d)),
      "beta": jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, h))),
      "a_log": jnp.log(jax.random.uniform(ks[5], (h,), minval=1.0, maxval=16.0)),
      "dt_bias": jax.random.uniform(ks[6], (h * d,), minval=-1.0, maxval=1.0),
  }


def _jax_core(q, k, v, g_raw, beta, a_log, dt_bias, initial_state=None):
  """The `kda_kernel='jax'` path of `KimiDeltaAttention`: explicit gate + L2-norm + chunked scan."""
  g = attention_kda.kda_gate(g_raw, a_log, dt_bias, _LB)
  q = l2norm(q, dim=-1, eps=1e-6) / math.sqrt(q.shape[-1])
  k = l2norm(k, dim=-1, eps=1e-6)
  return attention_kda.jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=initial_state)


def _tokamax_core(q, k, v, g_raw, beta, a_log, dt_bias, **kwargs):
  return attention_kda.tokamax_kimi_delta_rule(q, k, v, g_raw, beta, a_log, dt_bias, lower_bound=_LB, **kwargs)


def _rel(a, b):
  return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


@unittest.skipUnless(_HAS_TOKAMAX_KDA, "tokamax.ops.kimi_delta_attention is not available")
class TokamaxKdaCoreTest(unittest.TestCase):
  """`tokamax_kimi_delta_rule` against the in-tree chunked core."""

  def test_forward_and_gradients_match_jax_core(self):
    x = _inputs(2, 128, 3, 32)
    names = list(x)
    ref = _jax_core(**x)[0]
    w = jax.random.normal(jax.random.PRNGKey(9), ref.shape)
    grads_ref = jax.grad(lambda *a: jnp.sum(_jax_core(*a)[0] * w), argnums=tuple(range(7)))(*x.values())
    for impl in ("xla", "xla_chunked"):
      with self.subTest(implementation=impl):
        out = _tokamax_core(**x, implementation=impl)[0]
        self.assertLess(_rel(out, ref), 1e-5)
        if impl == "xla_chunked":  # the custom VJP under test
          grads = jax.grad(
              lambda *a: jnp.sum(_tokamax_core(*a, implementation="xla_chunked")[0] * w), argnums=tuple(range(7))
          )(*x.values())
          for name, g, g_ref in zip(names, grads, grads_ref):
            self.assertLess(_rel(g, g_ref), 1e-4, name)

  def test_sequence_not_multiple_of_chunk(self):
    x = _inputs(1, 100, 2, 32, seed=1)
    out = _tokamax_core(**x, implementation="xla_chunked")[0]
    self.assertEqual(out.shape, x["v"].shape)
    self.assertLess(_rel(out, _jax_core(**x)[0]), 1e-5)

  def test_initial_and_final_state_match_jax_core(self):
    x = _inputs(2, 128, 2, 32, seed=2)
    h0 = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (2, 2, 32, 32))
    ref_o, ref_h = _jax_core(**x, initial_state=h0)
    out, h = _tokamax_core(**x, implementation="xla_chunked", initial_state=h0, output_final_state=True)
    self.assertLess(_rel(out, ref_o), 1e-5)
    self.assertLess(_rel(h, ref_h), 1e-4)

  def test_segment_ids_reset_state_between_packed_documents(self):
    """A packed row [doc1 | doc2 | pad] equals running each document on its own."""
    lens = (70, 41)
    x = _inputs(1, 128, 2, 32, seed=4)
    seg = np.zeros((1, 128), np.int32)
    seg[0, : lens[0]] = 1
    seg[0, lens[0] : sum(lens)] = 2
    out = _tokamax_core(**x, implementation="xla_chunked", segment_ids=jnp.asarray(seg), max_num_segments=4)[0]
    start = 0
    for n in lens:
      piece = {
          name: (val[:, start : start + n] if val.ndim >= 3 and val.shape[1] == 128 else val) for name, val in x.items()
      }
      self.assertLess(_rel(out[:, start : start + n], _jax_core(**piece)[0]), 1e-5)
      start += n
    # Without segment ids doc2 sees doc1's state.
    unpacked = _tokamax_core(**x, implementation="xla_chunked")[0]
    self.assertGreater(_rel(unpacked[:, lens[0] : sum(lens)], out[:, lens[0] : sum(lens)]), 1e-3)


class _Cfg:
  """Minimal config object with the fields `KimiDeltaAttention` reads."""

  emb_dim, num_query_heads, head_dim = 64, 4, 16
  short_conv_kernel_size = 4
  kda_chunk_size = 64
  normalization_layer_epsilon = 1e-5
  kda_gate_lower_bound = _LB
  kda_compute_dtype = "float32"
  dtype = jnp.float32
  weight_dtype = jnp.float32
  shard_mode = ShardMode.AUTO
  matmul_precision = "highest"

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


@unittest.skipUnless(_HAS_TOKAMAX_KDA, "tokamax.ops.kimi_delta_attention is not available")
class TokamaxKdaModuleTest(unittest.TestCase):
  """`KimiDeltaAttention(kda_kernel='tokamax')` against the default JAX path, same weights."""

  def _modules(self, mesh=None, **cfg):
    jax_m = attention_kda.KimiDeltaAttention(config=_Cfg(), rngs=nnx.Rngs(params=0))
    tk_m = attention_kda.KimiDeltaAttention(
        config=_Cfg(kda_kernel="tokamax", kda_tokamax_implementation="xla_chunked", **cfg),
        mesh=mesh,
        rngs=nnx.Rngs(params=0),
    )
    return jax_m, tk_m

  def test_module_forward_and_param_grads_match(self):
    jax_m, tk_m = self._modules()
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 96, 64))
    np.testing.assert_allclose(
        np.asarray(tk_m(x, model_mode=MODEL_MODE_TRAIN)[0]),
        np.asarray(jax_m(x, model_mode=MODEL_MODE_TRAIN)[0]),
        rtol=1e-4,
        atol=1e-5,
    )

    def loss(module):
      return jnp.sum(module(x, model_mode=MODEL_MODE_TRAIN)[0] ** 2)

    g_jax = nnx.state(nnx.grad(loss)(jax_m), nnx.Param)
    g_tk = nnx.state(nnx.grad(loss)(tk_m), nnx.Param)
    for (path, grad_jax), (_, grad_tk) in zip(
        jax.tree_util.tree_leaves_with_path(g_jax), jax.tree_util.tree_leaves_with_path(g_tk)
    ):
      self.assertLess(_rel(grad_tk, grad_jax), 1e-4, jax.tree_util.keystr(path))

  def test_prefill_final_state_matches(self):
    jax_m, tk_m = self._modules()
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 64, 64))
    _, _, h_jax = jax_m(x, model_mode=MODEL_MODE_PREFILL, output_final_state=True)
    _, _, h_tk = tk_m(x, model_mode=MODEL_MODE_PREFILL, output_final_state=True)
    self.assertLess(_rel(h_tk, h_jax), 1e-4)

  def test_segment_ids_used_only_when_enabled(self):
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 64, 64))
    seg = jnp.concatenate([jnp.ones((1, 32), jnp.int32), 2 * jnp.ones((1, 32), jnp.int32)], axis=1)
    jax_m, tk_off = self._modules()
    _, tk_on = self._modules(kda_max_num_segments=4)
    y_ref = jax_m(x, model_mode=MODEL_MODE_TRAIN)[0]
    y_off = tk_off(x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=seg)[0]
    y_on = tk_on(x, model_mode=MODEL_MODE_TRAIN, decoder_segment_ids=seg)[0]
    self.assertLess(_rel(y_off, y_ref), 1e-4)  # kda_max_num_segments=0: ids ignored, like the JAX path
    self.assertLess(_rel(y_on[:, :32], y_ref[:, :32]), 1e-4)  # doc1 is unaffected by the reset
    self.assertGreater(_rel(y_on[:, 32:], y_ref[:, 32:]), 1e-3)  # doc2 no longer sees doc1's state

  def test_shard_map_path_matches_unsharded(self):
    n = len(jax.devices())
    shape = (2, 2) if n >= 4 else (1, 1)
    mesh = Mesh(np.asarray(jax.devices()[: shape[0] * shape[1]]).reshape(shape), ("data", "tensor"))
    rules = (("activation_batch", "data"), ("activation_heads", "tensor"))
    jax_m, tk_m = self._modules(mesh=mesh)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 64))
    with mesh, nn.logical_axis_rules(rules):
      y = jax.jit(lambda m, x: m(x, model_mode=MODEL_MODE_TRAIN)[0])(tk_m, x)
    self.assertLess(_rel(y, jax_m(x, model_mode=MODEL_MODE_TRAIN)[0]), 1e-4)


def _kimi_k3_config(**overrides):
  """Initializes `kimi-k3` through `pyconfig` with lightweight test overrides."""
  kwargs = {
      "run_name": "kimi_k3_kda_tokamax_test",
      "model_name": "kimi-k3",
      "override_model_config": True,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "max_target_length": 8,
      "per_device_batch_size": 1,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class TokamaxKdaPackingConfigTest(unittest.TestCase):
  """Config-time validation for `kimi_linear` sequence packing with `kda_kernel='tokamax'`."""

  def test_tokamax_packing_accepts_bounded_segments(self):
    for max_segs in (4, 8):
      with self.subTest(max_segments_per_seq=max_segs):
        cfg = _kimi_k3_config(
            packing=True,
            kda_kernel="tokamax",
            kda_max_num_segments=8,
            max_segments_per_seq=max_segs,
        )
        self.assertTrue(cfg.packing)
        self.assertEqual(cfg.max_segments_per_seq, max_segs)
        self.assertEqual(cfg.kda_max_num_segments, 8)

    cfg_grain = _kimi_k3_config(
        packing=True,
        kda_kernel="tokamax",
        kda_max_num_segments=8,
        max_segments_per_seq=8,
        dataset_type="grain",
        grain_train_files="dummy.arrayrecord",
        eval_interval=-1,
    )
    self.assertEqual(cfg_grain.dataset_type.value, "grain")

  def test_tokamax_packing_requires_tokamax_kernel_and_positive_bound(self):
    with self.assertRaisesRegex(ValueError, "kda_kernel=tokamax"):
      _kimi_k3_config(packing=True, kda_kernel="jax", kda_max_num_segments=8, max_segments_per_seq=8)
    with self.assertRaisesRegex(ValueError, "kda_max_num_segments > 0"):
      _kimi_k3_config(packing=True, kda_kernel="tokamax", kda_max_num_segments=0, max_segments_per_seq=8)

  def test_tokamax_packing_requires_bounded_max_segments_per_seq(self):
    for bad_max_segs in (-1, 0, 9):
      with self.subTest(max_segments_per_seq=bad_max_segs):
        with self.assertRaisesRegex(ValueError, "0 < max_segments_per_seq <= kda_max_num_segments"):
          _kimi_k3_config(
              packing=True,
              kda_kernel="tokamax",
              kda_max_num_segments=8,
              max_segments_per_seq=bad_max_segs,
          )

  def test_tokamax_packing_rejects_tfds_and_c4_mlperf(self):
    for dataset_type in ("tfds", "c4_mlperf"):
      with self.subTest(dataset_type=dataset_type):
        with self.assertRaisesRegex(ValueError, f"does not support dataset_type={dataset_type!r}"):
          _kimi_k3_config(
              packing=True,
              kda_kernel="tokamax",
              kda_max_num_segments=8,
              max_segments_per_seq=8,
              dataset_type=dataset_type,
              eval_interval=-1,
          )
    cfg_unpacked = _kimi_k3_config(packing=False, dataset_type="tfds", eval_interval=-1)
    self.assertFalse(cfg_unpacked.packing)


if __name__ == "__main__":
  unittest.main()
