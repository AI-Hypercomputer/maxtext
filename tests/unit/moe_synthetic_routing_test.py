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

"""CPU tests for `benchmarks.moe_synthetic_routing`."""

import gc
import tempfile
import weakref
import types
import unittest
from unittest import mock

from benchmarks.moe_synthetic_routing import synthetic_routing
from benchmarks.moe_synthetic_routing import train as bench_train
import jax
import jax.numpy as jnp
from maxtext.common import data_loader
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
import numpy as np
from tests.utils.test_helpers import get_test_config_path


class SyntheticRoutingGeneratorTest(unittest.TestCase):

  def _lognormal(self, seed=0, step=0, **kw):
    return synthetic_routing.lognormal_routing(
        batch=2,
        seq=64,
        num_layers=3,
        num_experts=32,
        top_k=4,
        sigmas=[0.5, 0.8, 1.0],
        seed=seed,
        step=step,
        threads=1,
        **kw,
    )

  def test_lognormal_shape_dtype_and_range(self):
    out = self._lognormal()
    self.assertEqual(out.shape, (2, 64, 3, 4))
    self.assertEqual(out.dtype, np.int32)
    self.assertTrue(np.all(out >= 0) and np.all(out < 32))

  def test_lognormal_topk_distinct_per_token(self):
    out = self._lognormal(row_taus="0.5", row_len=16)
    srt = np.sort(out, axis=-1)
    self.assertTrue(np.all(srt[..., 1:] != srt[..., :-1]))

  def test_lognormal_is_deterministic(self):
    np.testing.assert_array_equal(self._lognormal(seed=3, step=1), self._lognormal(seed=3, step=1))
    np.testing.assert_array_equal(
        self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
        self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
    )

  def test_lognormal_depends_on_seed_and_step(self):
    self.assertFalse(np.array_equal(self._lognormal(seed=0), self._lognormal(seed=1)))
    self.assertFalse(np.array_equal(self._lognormal(step=0), self._lognormal(step=1)))

  def test_lognormal_threads_match_serial(self):
    serial = self._lognormal(seed=5)
    threaded = synthetic_routing.lognormal_routing(2, 64, 3, 32, 4, [0.5, 0.8, 1.0], 5, 0, threads=3)
    np.testing.assert_array_equal(serial, threaded)

  def test_balanced_is_exactly_balanced(self):
    out = synthetic_routing.balanced_routing(batch=2, seq=64, num_layers=2, num_experts=32, top_k=4, step=1)
    self.assertEqual(out.shape, (2, 64, 2, 4))
    for layer in range(2):
      counts = np.bincount(out[:, :, layer, :].ravel(), minlength=32)
      self.assertTrue(np.all(counts == counts[0]))
    srt = np.sort(out, axis=-1)
    self.assertTrue(np.all(srt[..., 1:] != srt[..., :-1]))

  def test_profile_file_depth_mapping(self):
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
      prof = np.linspace(-1.0, 1.0, 5 * 8).reshape(5, 8)
      np.savez(tmp.name, log_profile=prof)
      sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
      cfg = types.SimpleNamespace(
          global_batch_size_to_load=2,
          max_target_length=16,
          num_decoder_layers=2,
          num_experts=16,
          num_experts_per_tok=2,
      )
      syn_cfg = synthetic_routing.SyntheticRoutingCfg(
          synthetic_routing="lognormal",
          synthetic_routing_pool_size=2,
          synthetic_routing_profile=tmp.name,
      )
      sr = synthetic_routing.SyntheticRouting(cfg, sharding, syn_cfg=syn_cfg)
      batch = np.asarray(sr.next())
      self.assertEqual(batch.shape, (2, 16, 2, 2))
      self.assertTrue(np.all((batch >= 0) & (batch < 16)))


class SyntheticRoutingPoolTest(unittest.TestCase):

  def _config(self, mode):
    return types.SimpleNamespace(
        synthetic_routing=mode,
        global_batch_size_to_load=2,
        max_target_length=32,
        num_decoder_layers=2,
        num_experts=16,
        num_experts_per_tok=2,
        synthetic_routing_pool_size=3,
        synthetic_routing_seed=0,
        synthetic_routing_sigma=0.5,
        synthetic_routing_sigmas="",
        synthetic_routing_row_tau=0.0,
        synthetic_routing_row_taus="",
        synthetic_routing_row_len=0,
        synthetic_routing_profile="",
    )

  def test_pool_cycles(self):
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    for mode in ("lognormal", "balanced"):
      with self.subTest(mode=mode):
        sr = synthetic_routing.SyntheticRouting(self._config(mode), sharding)
        batches = [np.asarray(sr.next()) for _ in range(4)]
        self.assertEqual(batches[0].shape, (2, 32, 2, 2))
        self.assertEqual(batches[0].dtype, np.int32)
        np.testing.assert_array_equal(batches[3], batches[0])  # pool_size=3

  def test_unknown_mode_raises(self):
    with self.assertRaises(ValueError):
      synthetic_routing.SyntheticRouting(self._config("bogus"), None)


class SyntheticRoutingWrapperTest(unittest.TestCase):

  def test_default_args_off(self):
    argv = ["train.py", get_test_config_path(), "run_name=test_run"]
    syn_cfg, rem = synthetic_routing.parse_synthetic_routing_args(argv)
    self.assertFalse(syn_cfg.enabled)
    self.assertEqual(syn_cfg.synthetic_routing, "none")
    self.assertEqual(rem, argv)

  def test_flag_and_kv_parsing_and_pyconfig_compat(self):
    argv = [
        "train.py",
        get_test_config_path(),
        "run_name=test_run",
        "enable_checkpointing=false",
        "--synthetic_routing=lognormal",
        "synthetic_routing_sigmas=0.5,0.8",
        "--synthetic_routing_row_taus",
        "1.2",
        "synthetic_routing_row_len=768",
        "--synthetic_routing_pool_size=2",
        "synthetic_routing_seed=7",
    ]
    syn_cfg, rem = synthetic_routing.parse_synthetic_routing_args(argv)
    self.assertTrue(syn_cfg.enabled)
    self.assertEqual(syn_cfg.synthetic_routing, "lognormal")
    self.assertEqual(syn_cfg.synthetic_routing_sigmas, "0.5,0.8")
    self.assertEqual(syn_cfg.synthetic_routing_row_taus, "1.2")
    self.assertEqual(syn_cfg.synthetic_routing_row_len, 768)
    self.assertEqual(syn_cfg.synthetic_routing_pool_size, 2)
    self.assertEqual(syn_cfg.synthetic_routing_seed, 7)
    self.assertEqual(rem, ["train.py", get_test_config_path(), "run_name=test_run", "enable_checkpointing=false"])
    # Remaining argv initializes MaxText pyconfig cleanly with zero core schema changes.
    cfg = pyconfig.initialize(rem)
    self.assertEqual(cfg.run_name, "test_run")

  def test_invalid_cli_options_rejected(self):
    for bad_argv in (
        ["train.py", "--synthetic_routing=bogus"],
        ["train.py", "synthetic_routing_pool_size=0"],
        ["train.py", "--synthetic_routing_unknown=1"],
    ):
      with self.subTest(bad_argv=bad_argv):
        with self.assertRaises(ValueError):
          synthetic_routing.parse_synthetic_routing_args(bad_argv)

  def test_runtime_hooks_patch_and_restore(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="hook_test",
        enable_checkpointing=False,
        per_device_batch_size=1,
        max_target_length=16,
        base_num_decoder_layers=2,
        num_experts=8,
        num_experts_per_tok=2,
        base_mlp_dim=64,
        base_moe_mlp_dim=64,
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    orig_gsb = maxtext_utils.get_shaped_batch
    orig_lnb = data_loader.DataLoader.load_next_batch

    syn_cfg = synthetic_routing.SyntheticRoutingCfg(
        synthetic_routing="balanced",
        synthetic_routing_pool_size=2,
    )
    dummy_raw = {
        "inputs": np.zeros((cfg.global_batch_size_to_load, 16), dtype=np.int32),
        "targets": np.zeros((cfg.global_batch_size_to_load, 16), dtype=np.int32),
    }

    class _DummyLoader:
      pass

    loader = _DummyLoader()
    loader.config = cfg
    loader.goodput_recorder = None
    loader.data_iterator = iter([dummy_raw, dummy_raw, dummy_raw])
    loader.last_batch = None
    loader.input_data_shardings = sharding
    loader_ref = weakref.ref(loader)
    loader.update_data_iterator = lambda: None
    loader.check_example_batch = lambda: None
    loader.load_next_batch_pre_sharding = types.MethodType(data_loader.DataLoader.load_next_batch_pre_sharding, loader)

    with bench_train.install_synthetic_routing(syn_cfg):
      shaped_train = maxtext_utils.get_shaped_batch(cfg, batch_sharding=sharding, is_eval=False)
      shaped_eval = maxtext_utils.get_shaped_batch(cfg, batch_sharding=sharding, is_eval=True)
      self.assertIn("forced_routed_experts", shaped_train)
      self.assertNotIn("forced_routed_experts", shaped_eval)
      self.assertEqual(
          shaped_train["forced_routed_experts"].shape,
          (cfg.global_batch_size_to_load, 16, cfg.num_decoder_layers, 2),
      )
      self.assertEqual(shaped_train["forced_routed_experts"].dtype, jnp.int32)

      b0 = data_loader.DataLoader.load_next_batch(loader)
      b1 = data_loader.DataLoader.load_next_batch(loader)
      b2 = data_loader.DataLoader.load_next_batch(loader)
      self.assertIn("forced_routed_experts", b0)
      self.assertEqual(b0["forced_routed_experts"].shape, shaped_train["forced_routed_experts"].shape)
      self.assertFalse(np.array_equal(np.asarray(b0["forced_routed_experts"]), np.asarray(b1["forced_routed_experts"])))
      np.testing.assert_array_equal(np.asarray(b0["forced_routed_experts"]), np.asarray(b2["forced_routed_experts"]))
      # Verify WeakKeyDictionary does not retain DataLoader instances once dropped.
      del loader
      gc.collect()
      self.assertIsNone(loader_ref())

    # Restored cleanly on exit.
    self.assertIs(maxtext_utils.get_shaped_batch, orig_gsb)
    self.assertIs(data_loader.DataLoader.load_next_batch, orig_lnb)
    self.assertNotIn("forced_routed_experts", maxtext_utils.get_shaped_batch(cfg, batch_sharding=sharding))

  def test_main_delegates_with_stripped_argv(self):
    with mock.patch.object(bench_train.pre_train, "main") as mock_pre_train_main:
      bench_train.main(["train.py", "base.yml", "steps=2", "--synthetic_routing=balanced"])
      mock_pre_train_main.assert_called_once_with(["train.py", "base.yml", "steps=2"])


if __name__ == "__main__":
  unittest.main()
