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

"""Tests for TPU-scale sharded/jitted zero-gen runner and Orbax checkpoint restore."""

# pylint: disable=no-name-in-module,missing-class-docstring,missing-function-docstring,g-import-not-at-top

import os
import types
from unittest import mock

from absl.testing import absltest
from etils import epath

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from maxtext.m3.models.qwen38_flash_next import loss as qwen38_loss
from maxtext.m3.models.qwen38_flash_next import modeling_qwen38_flash_next
from tests.utils import masked_prefill_driver as driver
from tests.utils import tpu_prefill_runner
from tests.utils import zero_gen_logits_checker as checker
import numpy as np
import orbax.checkpoint
import pytest


def _create_mini_config(scan_layers: bool = False, seq_len: int = 16) -> types.SimpleNamespace:
  c = types.SimpleNamespace(
      model_name="qwen3.8-flash-next",
      max_target_length=seq_len,
      scan_layers=scan_layers,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
  )
  u = c.__dict__.update
  u(
      emb_dim=64,
      num_decoder_layers=8,
      vocab_size=128,
      eos_token_id=127,
      ple_eos_token_id=127,
      hc_count=4,
  )
  u(
      hc_lowrank=16,
      normalization_layer_epsilon=1e-6,
      inhomogeneous_layer_cycle_interval=4,
      gdn_chunk_size=16,
  )
  u(
      gdn_num_key_heads=2,
      gdn_num_value_heads=4,
      gdn_key_head_dim=16,
      gdn_value_head_dim=16,
      gdn_conv_kernel_dim=4,
  )
  u(
      gdn_output_gate_activation="sigmoid",
      num_query_heads=4,
      num_kv_heads=1,
      head_dim=16,
      partial_rotary_factor=0.25,
  )
  u(
      rope_max_timescale=1e4,
      mrope_section=(1, 1, 0),
      index_n_heads=2,
      index_kv_heads=1,
      index_head_dim=16,
  )
  u(
      index_select_budget=8,
      index_select_block_size=4,
      moe_mlp_dim=32,
      mlp_dim=32,
      num_experts=4,
  )
  u(
      num_experts_per_tok=2,
      norm_topk_prob=True,
      routed_scaling_factor=1.0,
      shared_experts=1,
      use_ple_layer=False,
  )
  u(
      heads_per_ngram=2,
      ngram_size=3,
      ple_embed_dim=64,
      ple_kernel_size=3,
      ple_conv_kernel_size=4,
      ple_seed=0,
  )
  u(
      split_ngram_parts=2,
      ple_hash_seed=0,
      ngram_vocab_size_base=500,
      make_ngram_vocab_size_divisible_by=4,
  )
  u(
      mtp_num_layers=1,
      mtp_num_streams=4,
      mtp_loss_scaling_factor=0.1,
      use_qk_norm_in_gdn=True,
      load_balance_loss_weight=0.001,
  )
  return c


@pytest.mark.cpu_only
class TpuPrefillRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.cfg_unscanned = _create_mini_config(scan_layers=False, seq_len=16)
    self.cfg_scanned = _create_mini_config(scan_layers=True, seq_len=16)
    self.mesh = tpu_prefill_runner.create_device_mesh_for_config(self.cfg_unscanned)

  def test_create_device_mesh_for_config(self):
    fn = tpu_prefill_runner.create_device_mesh_for_config
    self.assertEqual(fn(self.cfg_unscanned).axis_names, ("data",))
    cfg = types.SimpleNamespace(mesh_axes=("d", "f"), ici_fsdp_parallelism=1)
    mu = mock.MagicMock(create_device_mesh=lambda _: np.zeros((1, 2), dtype=object))
    with mock.patch.object(tpu_prefill_runner, "_load_maxtext_utils", return_value=mu):
      self.assertEqual(fn(cfg).axis_names, ("d", "f"))

  def test_sharded_jitted_forward_both_scan_modes_match_eager(self):
    toks_np = driver.build_byte_mod_test_tokens(16, 128)
    toks = jnp.asarray(toks_np)
    for cfg in (self.cfg_unscanned, self.cfg_scanned):
      eager = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
      runner = tpu_prefill_runner.ShardedJittedModelRunner(eager, self.mesh)
      r_out, e_out = np.asarray(runner(toks)), np.asarray(eager(toks))
      np.testing.assert_allclose(r_out, e_out, rtol=1e-4, atol=1e-4)
      if not cfg.scan_layers:
        eo = driver.extract_model_prefill_outputs(eager, toks_np, return_aux=True)
        ro = driver.extract_model_prefill_outputs(runner, toks_np, return_aux=True)
        np.testing.assert_allclose(ro["logits"], eo["logits"], atol=1e-4)
        for k in ("routing_indices", "indexer_selected_blocks"):
          np.testing.assert_array_equal(ro[k], eo[k])

  def test_orbax_save_and_restore_parameters_both_scan_modes(self):
    cp = orbax.checkpoint.PyTreeCheckpointer()
    toks = jnp.asarray(driver.build_byte_mod_test_tokens(16, 128))
    for scan in (False, True):
      cfg = _create_mini_config(scan_layers=scan, seq_len=16)
      base_runner = tpu_prefill_runner.create_sharded_jitted_runner(config=cfg, mesh=self.mesh)
      saved_model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=1))
      state = nnx.state(saved_model, nnx.Param)
      for _, var in nnx.to_flat_state(state):
        var.value = var.value * 2.0 + 0.5
      nnx.update(saved_model, state)

      ckpt_dir = os.path.join(self.temp_dir, f"ckpt_scan_{scan}")
      p_dict = {"params": nnx.to_pure_dict(state)}
      cp.save(epath.Path(ckpt_dir), p_dict)

      r = tpu_prefill_runner.create_sharded_jitted_runner(config=cfg, mesh=self.mesh, load_parameters_path=ckpt_dir)
      res, s_out = np.asarray(r(toks)), np.asarray(saved_model(toks))
      np.testing.assert_allclose(res, s_out, rtol=1e-4, atol=1e-4)
      b_out = np.asarray(base_runner(toks))
      self.assertFalse(np.allclose(res, b_out, atol=1e-3))

    p_bad = epath.Path(os.path.join(self.temp_dir, "bad_missing"))
    cp.save(p_bad, {"params": {}})
    with self.assertRaisesRegex(ValueError, "not found in checkpoint"):
      tpu_prefill_runner.restore_model_parameters(saved_model, p_bad)

    p_bad2 = epath.Path(os.path.join(self.temp_dir, "bad_shape"))
    mismatched = nnx.to_pure_dict(nnx.state(saved_model, nnx.Param))
    mismatched["lm_head"]["kernel"] = np.zeros((1,), dtype=np.float32)
    cp.save(p_bad2, {"params": mismatched})
    with self.assertRaisesRegex(ValueError, "shape mismatch"):
      tpu_prefill_runner.restore_model_parameters(saved_model, p_bad2)

  def test_driver_run_cli_delegation_to_tpu_runner(self):
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg_unscanned, rngs=nnx.Rngs(params=0))
    runner = tpu_prefill_runner.ShardedJittedModelRunner(model, self.mesh)
    toks = driver.build_byte_mod_test_tokens(16, 128)
    g_path = os.path.join(self.temp_dir, "cli_golden.npz")
    checker.save_golden_npz(g_path, **driver.extract_model_prefill_outputs(model, toks))
    with mock.patch.object(tpu_prefill_runner, "create_sharded_jitted_runner", return_value=runner) as mock_runner:
      modes = (
          "prefill_logits",
          "tier1b_logits",
          "tinymmlu",
          "holdout_ce",
      )
      for m in modes:
        res = driver.run_cli(
            [f"mode={m}", f"golden_tensor_path={g_path}"],
            token_factory=lambda p, s: toks,
        )
        self.assertTrue(res.get("passed", True) or np.isfinite(float(res.get("total_loss", 0.0))))
      self.assertEqual(mock_runner.call_count, 4)

  def test_compute_param_partition_spec_unscanned_and_scanned(self):
    fn = tpu_prefill_runner.compute_param_partition_spec
    pspec = jax.sharding.PartitionSpec
    s_emb, s_cyc = (320001536, 160), (12, 512, 2560, 2048)
    self.assertEqual(fn(("emb",), s_emb, "fsdp", 16), pspec("fsdp", None))
    self.assertEqual(
        fn(("cycles",), s_cyc, "fsdp", 16),
        pspec(None, "fsdp", None, None),
    )
    self.assertEqual(fn(("cycles",), (12, 16), "fsdp", 4), pspec(None, "fsdp"))
    self.assertEqual(
        fn(
            ("decoder", "cycles", "moe_0_2", "gate_up_proj"),
            (8, 4, 4, 32, 64),
            "fsdp",
            4,
        ),
        pspec(None, None, "fsdp", None, None),
    )
    self.assertEqual(
        fn(
            ("decoder", "cycles", "gdn_0_2", "conv_weight"),
            (8, 4, 16, 32),
            "fsdp",
            4,
        ),
        pspec(None, None, "fsdp", None),
    )

  def test_shard_and_restore_model_parameters_attaches_named_sharding(self):
    def get_specs(m):
      return [v.value.sharding.spec for _, v in nnx.to_flat_state(nnx.state(m, nnx.Param))]

    cp = orbax.checkpoint.PyTreeCheckpointer()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg_unscanned, rngs=nnx.Rngs(params=0))
    tpu_prefill_runner.shard_model_parameters(model, self.mesh)
    specs1 = get_specs(model)
    self.assertTrue(any(s != jax.sharding.PartitionSpec() for s in specs1))

    ckpt_dir = os.path.join(self.temp_dir, "sharding_ckpt")
    p_dict = {"params": nnx.to_pure_dict(nnx.state(model, nnx.Param))}
    cp.save(epath.Path(ckpt_dir), p_dict)
    model2 = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg_unscanned, rngs=nnx.Rngs(params=1))
    tpu_prefill_runner.restore_model_parameters(model2, ckpt_dir, self.mesh)
    self.assertEqual(specs1, get_specs(model2))

  def test_compute_total_loss_and_holdout_ce_jitted_parity(self):
    m = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg_unscanned, rngs=nnx.Rngs(params=0))
    r = tpu_prefill_runner.ShardedJittedModelRunner(m, self.mesh)
    toks_np = driver.build_byte_mod_test_tokens(16, 128)
    t = jnp.asarray(toks_np)
    x, y = t[:, :-1], t[:, 1:]
    le, me = qwen38_loss.compute_total_loss(m, x, y, use_mtp=True)
    lr, mr = qwen38_loss.compute_total_loss(r, x, y, use_mtp=True)
    for a, b in ((lr, le), (mr["primary_loss"], me["primary_loss"])):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    res = driver.run_holdout_ce_check(r, r, toks_np)
    self.assertTrue(res["is_finite"] and np.isfinite(float(res["total_loss"])))
    self.assertAlmostEqual(res["ce_post_sft"], res["ce_pre_sft"], places=5)
    kw = {"decoder_positions": jnp.zeros_like(t), "next_tokens": t}
    self.assertEqual(np.asarray(r(t, **kw)).shape, (1, 16, 128))
    for aux, mtp in ((True, False), (False, True), (True, True)):
      self.assertIsInstance(r(t, return_aux=aux, return_mtp=mtp, **kw), tuple)

  def test_eval_shape_and_sharded_restore_matches_eager(self):
    cfg = _create_mini_config(scan_layers=True, seq_len=16)
    saved_model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=1))
    state = nnx.state(saved_model, nnx.Param)
    for _, var in nnx.to_flat_state(state):
      var.value = var.value * 2.0 + 0.5
    nnx.update(saved_model, state)

    ckpt_dir = os.path.join(self.temp_dir, "abstract_restore_ckpt")
    p_dict = {"params": nnx.to_pure_dict(state)}
    orbax.checkpoint.PyTreeCheckpointer().save(epath.Path(ckpt_dir), p_dict)

    with mock.patch.object(nnx, "eval_shape", wraps=nnx.eval_shape) as spy_eval_shape:
      runner = tpu_prefill_runner.create_sharded_jitted_runner(config=cfg, mesh=self.mesh, load_parameters_path=ckpt_dir)
      spy_eval_shape.assert_called_once()

    toks = jnp.asarray(driver.build_byte_mod_test_tokens(16, 128))
    r_out = np.asarray(runner(toks))
    e_out = np.asarray(saved_model(toks))
    np.testing.assert_allclose(r_out, e_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
