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
from flax import nnx
from flax import struct
from maxtext.common import data_loader
from maxtext.configs import pyconfig
from maxtext.layers import attentions
from maxtext.layers import moe
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
        np.testing.assert_array_equal(
            self._lognormal(seed=3, step=1), self._lognormal(seed=3, step=1)
        )
        np.testing.assert_array_equal(
            self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
            self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
        )

    def test_lognormal_depends_on_seed_and_step(self):
        self.assertFalse(
            np.array_equal(self._lognormal(seed=0), self._lognormal(seed=1))
        )
        self.assertFalse(
            np.array_equal(self._lognormal(step=0), self._lognormal(step=1))
        )

    def test_lognormal_threads_match_serial(self):
        serial = self._lognormal(seed=5)
        threaded = synthetic_routing.lognormal_routing(
            2, 64, 3, 32, 4, [0.5, 0.8, 1.0], 5, 0, threads=3
        )
        np.testing.assert_array_equal(serial, threaded)

    def test_balanced_is_exactly_balanced(self):
        out = synthetic_routing.balanced_routing(
            batch=2, seq=64, num_layers=2, num_experts=32, top_k=4, step=1
        )
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

    def test_profile_file_1d_vector_supported(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            prof_1d = np.linspace(-1.0, 1.0, 8)
            np.savez(tmp.name, log_profile=prof_1d)
            mapped_1d = synthetic_routing.depth_map_profiles(
                prof_1d, num_layers=3, num_experts=16
            )
            mapped_2d = synthetic_routing.depth_map_profiles(
                prof_1d[None, :], num_layers=3, num_experts=16
            )
            np.testing.assert_allclose(mapped_1d, mapped_2d)
            sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            cfg = types.SimpleNamespace(
                global_batch_size_to_load=2,
                max_target_length=16,
                num_decoder_layers=3,
                num_experts=16,
                num_experts_per_tok=2,
            )
            syn_cfg = synthetic_routing.SyntheticRoutingCfg(
                synthetic_routing="lognormal",
                synthetic_routing_pool_size=1,
                synthetic_routing_profile=tmp.name,
            )
            sr = synthetic_routing.SyntheticRouting(cfg, sharding, syn_cfg=syn_cfg)
            batch = np.asarray(sr.next())
            self.assertEqual(batch.shape, (2, 16, 3, 2))
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
        self.assertEqual(
            rem,
            [
                "train.py",
                get_test_config_path(),
                "run_name=test_run",
                "enable_checkpointing=false",
            ],
        )
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
        loader.load_next_batch_pre_sharding = types.MethodType(
            data_loader.DataLoader.load_next_batch_pre_sharding, loader
        )

        with bench_train.install_synthetic_routing(syn_cfg):
            shaped_train = maxtext_utils.get_shaped_batch(
                cfg, batch_sharding=sharding, is_eval=False
            )
            shaped_eval = maxtext_utils.get_shaped_batch(
                cfg, batch_sharding=sharding, is_eval=True
            )
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
            self.assertEqual(
                b0["forced_routed_experts"].shape,
                shaped_train["forced_routed_experts"].shape,
            )
            self.assertFalse(
                np.array_equal(
                    np.asarray(b0["forced_routed_experts"]),
                    np.asarray(b1["forced_routed_experts"]),
                )
            )
            np.testing.assert_array_equal(
                np.asarray(b0["forced_routed_experts"]),
                np.asarray(b2["forced_routed_experts"]),
            )
            # Verify WeakKeyDictionary does not retain DataLoader instances once dropped.
            del loader
            gc.collect()
            self.assertIsNone(loader_ref())

        # Restored cleanly on exit.
        self.assertIs(maxtext_utils.get_shaped_batch, orig_gsb)
        self.assertIs(data_loader.DataLoader.load_next_batch, orig_lnb)
        self.assertNotIn(
            "forced_routed_experts",
            maxtext_utils.get_shaped_batch(cfg, batch_sharding=sharding),
        )

    def test_forced_routing_reaches_routed_moe_and_changes_group_sizes(self):
        cfg = pyconfig.initialize(
            [None, get_test_config_path()],
            run_name="moe_reach_test",
            enable_checkpointing=False,
            per_device_batch_size=2,
            max_target_length=32,
            base_num_decoder_layers=1,
            num_experts=8,
            num_experts_per_tok=2,
            base_emb_dim=32,
            base_mlp_dim=64,
            base_moe_mlp_dim=64,
            sparse_matmul=True,
        )
        dev = jax.devices()[0]
        mesh = jax.sharding.Mesh(np.array([dev]), ("data",))
        sharding = jax.sharding.SingleDeviceSharding(dev)
        routed_moe = moe.RoutedMoE(
            config=cfg,
            num_experts=8,
            num_experts_per_tok=2,
            mesh=mesh,
            kernel_init=attentions.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=("exp", "embed_moe", "mlp_moe"),
            rngs=nnx.Rngs(params=0, dropout=1),
            intermediate_dim=64,
        )
        bsz = cfg.global_batch_size_to_load
        seq = 32
        x_in = jnp.ones((bsz, seq, 32), dtype=jnp.float32)
        # Deliberately bias router logits toward expert 0 so only forced_routed_experts can produce our target counts.
        biased_logits = (
            jnp.zeros((bsz, seq, 8), dtype=jnp.float32).at[:, :, 0].set(100.0)
        )

        def run_mode(syn_cfg):
            class _DummyLoader:
                pass

            loader = _DummyLoader()
            loader.config = cfg
            loader.goodput_recorder = None
            loader.data_iterator = iter(
                [
                    {
                        "inputs": np.zeros((bsz, seq), np.int32),
                        "targets": np.zeros((bsz, seq), np.int32),
                    }
                ]
            )
            loader.last_batch = None
            loader.input_data_shardings = sharding
            loader.update_data_iterator = lambda: None
            loader.check_example_batch = lambda: None
            loader.load_next_batch_pre_sharding = types.MethodType(
                data_loader.DataLoader.load_next_batch_pre_sharding, loader
            )
            with bench_train.install_synthetic_routing(syn_cfg):
                batch = data_loader.DataLoader.load_next_batch(loader)
            forced_l0 = batch["forced_routed_experts"][:, :, 0, :]
            _, _, _, group_sizes, _, _, _, _, _, _, _, _ = routed_moe.permute(
                x_in,
                biased_logits,
                biased_logits,
                use_custom_sort_vjp=False,
                rngs=nnx.Rngs(params=0),
                forced_routed_experts=forced_l0,
                return_expert_counts=True,
            )
            return np.asarray(forced_l0), np.asarray(group_sizes)

        forced_bal, gs_bal = run_mode(
            synthetic_routing.SyntheticRoutingCfg(
                synthetic_routing="balanced", synthetic_routing_pool_size=1
            )
        )
        np.testing.assert_array_equal(
            gs_bal, np.bincount(forced_bal.ravel(), minlength=8)
        )
        self.assertTrue(np.all(gs_bal == (bsz * seq * 2) // 8))

        forced_skew, gs_skew = run_mode(
            synthetic_routing.SyntheticRoutingCfg(
                synthetic_routing="lognormal",
                synthetic_routing_sigma=2.5,
                synthetic_routing_pool_size=1,
                synthetic_routing_seed=0,
            )
        )
        np.testing.assert_array_equal(
            gs_skew, np.bincount(forced_skew.ravel(), minlength=8)
        )
        self.assertGreaterEqual(float(gs_skew.max()) / float(gs_skew.mean()), 2.0)

    def test_bare_boolean_switch_flag_parsing(self):
        syn_mid, rem_mid = synthetic_routing.parse_synthetic_routing_args(
            ["train.py", "--synthetic_routing_overhead_control", "base.yml", "steps=2"]
        )
        self.assertTrue(syn_mid.synthetic_routing_overhead_control)
        self.assertEqual(rem_mid, ["train.py", "base.yml", "steps=2"])

        syn_end, rem_end = synthetic_routing.parse_synthetic_routing_args(
            ["train.py", "base.yml", "--synthetic_routing_overhead_control"]
        )
        self.assertTrue(syn_end.synthetic_routing_overhead_control)
        self.assertEqual(rem_end, ["train.py", "base.yml"])

        syn_false, rem_false = synthetic_routing.parse_synthetic_routing_args(
            ["train.py", "--synthetic_routing_overhead_control", "false", "base.yml"]
        )
        self.assertFalse(syn_false.synthetic_routing_overhead_control)
        self.assertEqual(rem_false, ["train.py", "base.yml"])

        with self.assertRaises(ValueError):
            synthetic_routing.parse_synthetic_routing_args(
                ["train.py", "synthetic_routing_overhead_control"]
            )

    def test_overhead_control_hook_zeroes_replica_routing_and_restores(self):
        syn_cfg, rem = synthetic_routing.parse_synthetic_routing_args(
            ["train.py", "base.yml", "--synthetic_routing_overhead_control=true"]
        )
        self.assertTrue(syn_cfg.synthetic_routing_overhead_control)
        self.assertFalse(syn_cfg.enabled)
        self.assertEqual(rem, ["train.py", "base.yml"])
        syn_off, _ = synthetic_routing.parse_synthetic_routing_args(
            ["train.py", "synthetic_routing_overhead_control=false"]
        )
        self.assertFalse(syn_off.synthetic_routing_overhead_control)
        with self.assertRaises(ValueError):
            synthetic_routing.parse_synthetic_routing_args(
                ["train.py", "--synthetic_routing_overhead_control=maybe"]
            )

        @struct.dataclass
        class _FrozenFakePlan:
            route_table: object = struct.field(pytree_node=False)
            rep_home: object = struct.field(pytree_node=False)

        fake_mrp = types.ModuleType("maxtext.layers.moe_replica_planner")

        def fake_identity_route_table(num_ep, num_experts, num_replicas):
            return ("identity", num_ep, num_experts, num_replicas)

        def orig_plan_iter(counts_src, num_ep, num_replicas, *_args, **_kwargs):
            return _FrozenFakePlan(
                route_table=("replicated", num_ep, counts_src.shape[1], num_replicas),
                rep_home="kept_rep_home",
            )

        fake_mrp.identity_route_table = fake_identity_route_table
        fake_mrp.compute_plan_iter = orig_plan_iter
        import maxtext.layers as maxtext_layers  # pylint: disable=import-outside-toplevel

        with mock.patch.dict(
            "sys.modules", {"maxtext.layers.moe_replica_planner": fake_mrp}
        ):
            with mock.patch.object(
                maxtext_layers, "moe_replica_planner", fake_mrp, create=True
            ):
                with bench_train.install_synthetic_routing(syn_cfg):
                    plan = fake_mrp.compute_plan_iter(np.zeros((4, 16), np.int32), 4, 2)
                    self.assertEqual(plan.route_table, ("identity", 4, 16, 2))
                    self.assertEqual(plan.rep_home, "kept_rep_home")
                self.assertIs(fake_mrp.compute_plan_iter, orig_plan_iter)

    def test_overhead_control_hook_with_real_replica_planner(self):
        try:
            from maxtext.layers import moe_replica_planner as mrp  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.skipTest(
                "maxtext.layers.moe_replica_planner is not present on standalone main"
            )
        orig_fn = mrp.compute_plan_iter
        counts_src = jnp.array([[50, 1, 1, 1], [40, 1, 1, 1]], dtype=jnp.int32)
        raw_plan = orig_fn(counts_src, 2, 1, 1, 2)
        self.assertFalse(
            np.array_equal(
                np.asarray(raw_plan.route_table),
                np.asarray(mrp.identity_route_table(2, 4, 1)),
            )
        )
        syn_cfg = synthetic_routing.SyntheticRoutingCfg(
            synthetic_routing_overhead_control=True
        )
        with bench_train.install_synthetic_routing(syn_cfg):
            oc_plan = mrp.compute_plan_iter(counts_src, 2, 1, 1, 2)
            self.assertIsInstance(oc_plan, mrp.ReplicaPlan)
            np.testing.assert_array_equal(
                np.asarray(oc_plan.route_table),
                np.asarray(mrp.identity_route_table(2, 4, 1)),
            )
            np.testing.assert_array_equal(
                np.asarray(oc_plan.rep_home), np.asarray(raw_plan.rep_home)
            )
        self.assertIs(mrp.compute_plan_iter, orig_fn)

    def test_partial_hook_install_failure_restores_all_originals(self):
        @struct.dataclass
        class _FrozenFakePlan:
            route_table: object = struct.field(pytree_node=False)

        fake_mrp = types.ModuleType("maxtext.layers.moe_replica_planner")
        orig_plan_iter = (
            lambda counts_src, num_ep, num_replicas, *_a, **_kw: _FrozenFakePlan(
                route_table="orig"
            )
        )
        fake_mrp.identity_route_table = (
            lambda num_ep, num_experts, num_replicas: "identity"
        )
        fake_mrp.compute_plan_iter = orig_plan_iter
        orig_gsb = maxtext_utils.get_shaped_batch
        orig_lnb = data_loader.DataLoader.load_next_batch

        import maxtext.layers as maxtext_layers  # pylint: disable=import-outside-toplevel

        syn_cfg = synthetic_routing.SyntheticRoutingCfg(
            synthetic_routing="balanced",
            synthetic_routing_overhead_control=True,
        )
        with mock.patch.dict(
            "sys.modules", {"maxtext.layers.moe_replica_planner": fake_mrp}
        ):
            with mock.patch.object(
                maxtext_layers, "moe_replica_planner", fake_mrp, create=True
            ):
                with mock.patch.object(
                    bench_train.weakref,
                    "WeakKeyDictionary",
                    side_effect=RuntimeError("boom"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        with bench_train.install_synthetic_routing(syn_cfg):
                            pass
        self.assertIs(fake_mrp.compute_plan_iter, orig_plan_iter)
        self.assertIs(maxtext_utils.get_shaped_batch, orig_gsb)
        self.assertIs(data_loader.DataLoader.load_next_batch, orig_lnb)

    def test_main_delegates_with_stripped_argv(self):
        with mock.patch.object(bench_train.pre_train, "main") as mock_pre_train_main:
            bench_train.main(
                ["train.py", "base.yml", "steps=2", "--synthetic_routing=balanced"]
            )
            mock_pre_train_main.assert_called_once_with(
                ["train.py", "base.yml", "steps=2"]
            )
        with mock.patch.object(bench_train.app, "run") as mock_app_run:
            bench_train.main(
                ["train.py", "base.yml", "--synthetic_routing=none"], use_absl_app=True
            )
            mock_app_run.assert_called_once_with(
                bench_train.pre_train.main, argv=["train.py", "base.yml"]
            )

    def test_overhead_control_missing_planner_raises_clear_import_error(self):
        syn_cfg = synthetic_routing.SyntheticRoutingCfg(
            synthetic_routing_overhead_control=True
        )
        with mock.patch.dict(
            "sys.modules", {"maxtext.layers.moe_replica_planner": None}
        ):
            with self.assertRaisesRegex(
                ImportError, "requires maxtext.layers.moe_replica_planner"
            ):
                with bench_train.install_synthetic_routing(syn_cfg):
                    pass

    def test_readme_defaults_match_synthetic_routing_cfg(self):
        import dataclasses  # pylint: disable=import-outside-toplevel
        from pathlib import Path as _Path  # pylint: disable=import-outside-toplevel

        defaults = {
            f.name: f.default
            for f in dataclasses.fields(synthetic_routing.SyntheticRoutingCfg)
        }
        readme = (
            _Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "moe_synthetic_routing"
            / "README.md"
        ).read_text()
        table_rows = {}
        for line in readme.splitlines():
            if line.startswith("| `synthetic_routing"):
                cols = [c.strip().strip("`") for c in line.strip("|").split("|")]
                table_rows[cols[0]] = cols[1]
        self.assertEqual(set(table_rows.keys()), set(defaults.keys()))
        for key, val in defaults.items():
            if isinstance(val, bool):
                expected = "true" if val else "false"
            elif isinstance(val, str):
                expected = f'"{val}"'
            else:
                expected = str(val)
            self.assertEqual(
                table_rows[key], expected, msg=f"README default mismatch for {key}"
            )


if __name__ == "__main__":
    unittest.main()
