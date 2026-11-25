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

"""Unit tests for the Tokamax GatedDeltaNet adapter (`use_tokamax_gdn`)."""

import contextlib
import dataclasses
import io
import os
import subprocess
import sys
import types
from typing import Any
import unittest
from unittest import mock

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

# pylint: disable=wrong-import-position
from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec as P
import numpy as np

from maxtext.common.common_types import (
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.attention import tokamax_gdn
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import nnx_construct_named_sharding
from tests.utils.test_helpers import get_test_config_path

# pylint: enable=wrong-import-position

try:
  tokamax_gdn._tokamax_gdn_op()  # pylint: disable=protected-access
  _HAS_TOKAMAX_GDN = True
except ImportError:
  _HAS_TOKAMAX_GDN = False

_ORIG_MATMUL_PRECISION = None
_ORIG_PLATFORMS = None


def setUpModule():
  global _ORIG_MATMUL_PRECISION, _ORIG_PLATFORMS
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  _ORIG_PLATFORMS = jax.config.jax_platforms
  jax.config.update("jax_default_matmul_precision", "highest")
  if not os.environ.get("JAX_PLATFORMS") and _ORIG_PLATFORMS is None:
    jax.config.update("jax_platforms", "cpu")


def tearDownModule():
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)
  jax.config.update("jax_platforms", _ORIG_PLATFORMS)


@dataclasses.dataclass(frozen=True)
class _FakeAbstractDevice:
  """Minimal `AbstractDevice`-compatible descriptor for CPU-only AOT TPU lowering tests."""

  device_kind: str
  num_cores: int | None
  platform: str

  def _repr(self) -> str:
    return f"device_kind={self.device_kind}, num_cores={self.num_cores}, platform={self.platform}"


def _rel_err(actual, expected) -> float:
  actual_np = np.asarray(jax.device_get(actual), dtype=np.float64)
  expected_np = np.asarray(jax.device_get(expected), dtype=np.float64)
  return float(np.max(np.abs(actual_np - expected_np)) / (np.max(np.abs(expected_np)) + 1e-30))


def _gdn_pyconfig(**overrides):
  """Initializes a lightweight `qwen3_next` config for GatedDeltaNet unit tests."""
  kwargs = {
      "run_name": "tokamax_gdn_test",
      "decoder_block": "qwen3_next",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": 32,
      "gdn_num_key_heads": 4,
      "gdn_num_value_heads": 8,
      "gdn_key_head_dim": 16,
      "gdn_value_head_dim": 16,
      "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 64,
      "use_qk_norm_in_gdn": True,
      "packing": False,
      "max_target_length": 128,
      "max_prefill_predict_length": 64,
      "per_device_batch_size": 1.0,
      "context_parallel_load_balance": False,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _run_2d_cp_mesh_parity_checks(devices) -> None:
  """Verifies `Qwen3NextGatedDeltaNet(use_tokamax_gdn=True)` on `(data=2, context=2)` for AUTO & EXPLICIT."""
  assert len(devices) >= 4, f"Expected >= 4 devices, got {len(devices)}"
  batch, seq_len, emb_dim = 2, 128, 32
  x = jax.random.normal(jax.random.PRNGKey(17), (batch, seq_len, emb_dim), dtype=jnp.float32) * 0.2
  dy = jax.random.normal(jax.random.PRNGKey(18), (batch, seq_len, emb_dim), dtype=jnp.float32)

  cfg_ref = _gdn_pyconfig(use_tokamax_gdn=False)
  with nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
    layer_ref = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_ref,
        mesh=None,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=5),
    )
    gdef_ref, st_ref = nnx.split(layer_ref)

    def ref_loss(st, inp):
      out, _ = nnx.merge(gdef_ref, st)(inp, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out * dy), out

    (_, ref_out), (ref_grads, ref_dx) = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1), has_aux=True))(st_ref, x)
  ref_out = jax.device_get(ref_out)
  ref_dx = jax.device_get(ref_dx)
  ref_grads = jax.tree_util.tree_map(jax.device_get, ref_grads)

  devs_2x2 = np.asarray(devices[:4]).reshape(2, 2)
  for cp_mode in ("auto", "head"):
    for shard_mode_str in ("auto", "explicit"):
      cfg_cp = _gdn_pyconfig(
          use_tokamax_gdn=True,
          ici_data_parallelism=2,
          ici_context_parallelism=2,
          gdn_cp_mode=cp_mode,
          shard_mode="auto",
          num_experts=1,
          sparse_matmul=False,
          megablox=False,
      )
      if shard_mode_str == "explicit":
        # Top-level config validation blocks `shard_mode='explicit'` with CP > 1 for full decoder blocks;
        # override on the initialized config to keep testing the `Qwen3NextGatedDeltaNet` submodule under EXPLICIT.
        object.__getattribute__(cfg_cp, "_flat_config")["shard_mode"] = config_types.ShardMode.EXPLICIT
      axis_types = (AxisType.Explicit, AxisType.Explicit) if shard_mode_str == "explicit" else None
      mesh = (
          Mesh(devs_2x2, ("data", "context"), axis_types=axis_types)
          if axis_types
          else Mesh(devs_2x2, ("data", "context"))
      )
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
        layer_cp = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_cp,
            mesh=mesh,
            dtype=jnp.float32,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=5),
        )
        gdef_cp, st_cp = nnx.split(layer_cp)
        st_cp = jax.device_put(st_cp, nnx_construct_named_sharding(st_cp, mesh))
        tok_sh = jax.sharding.NamedSharding(mesh, P("data", "context", None))
        x_in = jax.device_put(x, tok_sh)
        dy_in = jax.device_put(dy, tok_sh)

        def cp_loss(st, inp, cot, gdef=gdef_cp, out_sh=tok_sh if shard_mode_str == "explicit" else None):
          out, _ = nnx.merge(gdef, st)(inp, model_mode=MODEL_MODE_TRAIN, out_sharding=out_sh)
          return jnp.sum(out * cot), out

        (_, out_cp), (grads_cp, dx_cp) = jax.jit(jax.value_and_grad(cp_loss, argnums=(0, 1), has_aux=True))(
            st_cp, x_in, dy_in
        )

      tag = f"cp_mode={cp_mode}, shard_mode={shard_mode_str}"
      assert _rel_err(out_cp, ref_out) < 2e-4, f"{tag} output mismatch: {_rel_err(out_cp, ref_out)}"
      assert _rel_err(dx_cp, ref_dx) < 3e-4, f"{tag} dx mismatch: {_rel_err(dx_cp, ref_dx)}"
      seen_keys = set()
      for (path, g_got), (_, g_exp) in zip(
          jax.tree_util.tree_leaves_with_path(grads_cp),
          jax.tree_util.tree_leaves_with_path(ref_grads),
      ):
        key_str = jax.tree_util.keystr(path)
        seen_keys.add(key_str)
        err = _rel_err(g_got, g_exp)
        assert err < 3e-4, f"{tag} grad mismatch on {key_str}: {err}"
      assert any("A_log" in k for k in seen_keys), seen_keys
      assert any("dt_bias" in k for k in seen_keys), seen_keys


class TokamaxGdnConfigValidationTest(unittest.TestCase):
  """Always-running config validation tests across `pyconfig`, `MaxTextConfig`, and `RLConfig`."""

  def test_defaults_are_false(self):
    cfg = _gdn_pyconfig()
    self.assertFalse(cfg.use_tokamax_gdn)
    mt_cfg = config_types.MaxTextConfig(run_name="mt_default")
    self.assertFalse(mt_cfg.use_tokamax_gdn)
    rl_cfg = config_types.RLConfig(run_name="rl_default", model_name="qwen3-4b")
    self.assertFalse(rl_cfg.use_tokamax_gdn)

  def test_rejects_non_gdn_decoder_block(self):
    for decoder_block in ("llama2", "qwen3"):
      with self.subTest(decoder_block=decoder_block):
        with self.assertRaisesRegex(ValueError, "requires a GatedDeltaNet decoder_block"):
          _gdn_pyconfig(decoder_block=decoder_block, use_tokamax_gdn=True)
        with self.assertRaisesRegex(ValueError, "requires a GatedDeltaNet decoder_block"):
          config_types.MaxTextConfig(run_name="bad_mt_block", decoder_block=decoder_block, use_tokamax_gdn=True)
        with self.assertRaisesRegex(ValueError, "requires a GatedDeltaNet decoder_block"):
          config_types.RLConfig(
              run_name="bad_rl_block",
              model_name="qwen3-4b",
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
          )

  def test_rejects_sequence_packing_and_rl_packed_tokens(self):
    for decoder_block in ("qwen3_next", "qwen3_5"):
      with self.subTest(decoder_block=decoder_block):
        with self.assertRaisesRegex(ValueError, "does not yet support sequence packing"):
          _gdn_pyconfig(decoder_block=decoder_block, use_tokamax_gdn=True, packing=True)
        with self.assertRaisesRegex(ValueError, "does not yet support sequence packing"):
          config_types.MaxTextConfig(
              run_name="bad_mt_pack",
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
              packing=True,
              enable_checkpointing=False,
          )
        with self.assertRaisesRegex(ValueError, "does not yet support sequence packing"):
          config_types.RLConfig(
              run_name="bad_rl_pack",
              model_name="qwen3-next-80b-a3b",
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
              packing=True,
              enable_checkpointing=False,
          )
        with self.assertRaisesRegex(ValueError, "does not yet support sequence packing"):
          config_types.RLConfig(
              run_name="bad_rl_max_seq_tpu",
              model_name="qwen3-next-80b-a3b",
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
              packing=False,
              max_target_length=256,
              max_seq_token_per_tpu=256,
              enable_checkpointing=False,
          )

  def test_rejects_invalid_chunk_size_and_unaligned_target_length(self):
    for bad_chunk in (32, 128):
      with self.subTest(gdn_chunk_size=bad_chunk):
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size=64"):
          _gdn_pyconfig(use_tokamax_gdn=True, gdn_chunk_size=bad_chunk, max_target_length=256)
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size=64"):
          config_types.RLConfig(
              run_name="bad_rl_chunk",
              model_name="qwen3-next-80b-a3b",
              decoder_block="qwen3_next",
              use_tokamax_gdn=True,
              packing=False,
              gdn_chunk_size=bad_chunk,
              max_target_length=256,
              enable_checkpointing=False,
          )

    with self.assertRaisesRegex(ValueError, "divisible by gdn_chunk_size"):
      _gdn_pyconfig(use_tokamax_gdn=True, gdn_chunk_size=64, max_target_length=100)
    with self.assertRaisesRegex(ValueError, "divisible by gdn_chunk_size"):
      config_types.RLConfig(
          run_name="bad_rl_seq",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          packing=False,
          gdn_chunk_size=64,
          max_target_length=100,
          enable_checkpointing=False,
      )

  def test_context_parallel_validations_and_acceptances(self):
    # Rejects gdn_cp_mode='seq' when CP > 1.
    with self.assertRaisesRegex(ValueError, "does not yet support gdn_cp_mode='seq'"):
      _gdn_pyconfig(use_tokamax_gdn=True, ici_context_parallelism=2, gdn_cp_mode="seq")
    with self.assertRaisesRegex(ValueError, "does not yet support gdn_cp_mode='seq'"):
      config_types.RLConfig(
          run_name="bad_rl_seq_cp",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          packing=False,
          ici_context_parallelism=2,
          context_parallel_load_balance=False,
          gdn_cp_mode="seq",
          enable_checkpointing=False,
      )

    # Rejects non-divisible gdn_num_key_heads % cp != 0 in both 'auto' and 'head'.
    for cp_mode in ("auto", "head"):
      with self.subTest(cp_mode=cp_mode):
        with self.assertRaisesRegex(ValueError, "divisible by the GatedDeltaNet context-parallel size"):
          _gdn_pyconfig(
              use_tokamax_gdn=True,
              gdn_num_key_heads=6,
              gdn_num_value_heads=12,
              ici_context_parallelism=4,
              gdn_cp_mode=cp_mode,
          )
        with self.assertRaisesRegex(ValueError, "divisible by the GatedDeltaNet context-parallel size"):
          config_types.RLConfig(
              run_name="bad_rl_heads",
              model_name="qwen3-next-80b-a3b",
              decoder_block="qwen3_next",
              use_tokamax_gdn=True,
              packing=False,
              gdn_num_key_heads=6,
              gdn_num_value_heads=12,
              ici_context_parallelism=4,
              context_parallel_load_balance=False,
              gdn_cp_mode=cp_mode,
              enable_checkpointing=False,
          )

    # Rejects max_target_length not divisible by CP size (e.g. T=64, cp=16 -> 64 % 64 == 0 but 64 % 16 == 0;
    # T=192, cp=8 -> 192 % 64 == 0 and 192 % 8 == 0; T=64, cp=8 with gdn_num_key_heads=8 -> 64 % 8 == 0;
    # T=64, cp=16 with gdn_num_key_heads=32, cp=32 -> 64 % 32 == 0; T=64, cp=12 with gdn_num_key_heads=12 -> 64 % 12 != 0).
    with self.assertRaisesRegex(ValueError, "max_target_length .* to be divisible by the GatedDeltaNet"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_num_key_heads=12,
          gdn_num_value_heads=24,
          max_target_length=64,
          ici_context_parallelism=12,
      )

    # Rejects gdn_in_proj_kernel_order=True when CP > 1 under use_tokamax_gdn=True (even in 'auto').
    for cp_mode in ("auto", "head"):
      with self.subTest(kernel_order_cp_mode=cp_mode):
        with self.assertRaisesRegex(ValueError, "gdn_in_proj_kernel_order=True is incompatible with head-sharded"):
          _gdn_pyconfig(
              use_tokamax_gdn=True,
              ici_context_parallelism=2,
              gdn_cp_mode=cp_mode,
              gdn_in_proj_kernel_order=True,
          )
        with self.assertRaisesRegex(ValueError, "gdn_in_proj_kernel_order=True is incompatible with head-sharded"):
          config_types.RLConfig(
              run_name="bad_rl_kernel_order",
              model_name="qwen3-next-80b-a3b",
              decoder_block="qwen3_next",
              use_tokamax_gdn=True,
              packing=False,
              ici_context_parallelism=2,
              context_parallel_load_balance=False,
              gdn_cp_mode=cp_mode,
              gdn_in_proj_kernel_order=True,
              enable_checkpointing=False,
          )

    # Rejects combined TP * CP > gdn_num_key_heads (e.g. H_k=2 with TP=2, CP=2 -> total head shards = 4).
    with self.assertRaisesRegex(ValueError, "divisible by total GatedDeltaNet head shards"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          ici_tensor_parallelism=2,
          ici_context_parallelism=2,
      )
    with self.assertRaisesRegex(ValueError, "divisible by total GatedDeltaNet head shards"):
      config_types.RLConfig(
          run_name="bad_rl_tp_cp_heads",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          packing=False,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          ici_tensor_parallelism=2,
          ici_context_parallelism=2,
          context_parallel_load_balance=False,
          enable_checkpointing=False,
      )

    # Rejects unsupported Mosaic static input shapes/dtypes (`dtype='float16'`, `gdn_key_head_dim > 256`).
    with self.assertRaisesRegex(ValueError, "requires dtype in"):
      _gdn_pyconfig(use_tokamax_gdn=True, dtype="float16")
    with self.assertRaisesRegex(ValueError, "requires dtype in"):
      config_types.RLConfig(
          run_name="bad_rl_dtype",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          packing=False,
          dtype="float16",
          enable_checkpointing=False,
      )
    with self.assertRaisesRegex(ValueError, "requires gdn_key_head_dim in"):
      _gdn_pyconfig(use_tokamax_gdn=True, gdn_key_head_dim=512)
    with self.assertRaisesRegex(ValueError, "requires gdn_key_head_dim in"):
      config_types.RLConfig(
          run_name="bad_rl_head_dim",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          packing=False,
          gdn_key_head_dim=512,
          enable_checkpointing=False,
      )

    # Rejects shard_mode='explicit' with CP > 1 even when use_tokamax_gdn=True.
    with self.assertRaisesRegex(ValueError, "'explicit' sharding .* does not support context parallelism"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          shard_mode="explicit",
          ici_context_parallelism=2,
          num_experts=1,
          sparse_matmul=False,
          megablox=False,
      )

    # Accepts auto-inferred tensor parallelism (-1) prior to mesh resolution.
    rl_auto_tp = config_types.RLConfig(
        run_name="ok_rl_auto_tp",
        model_name="qwen3-next-80b-a3b",
        decoder_block="qwen3_next",
        use_tokamax_gdn=True,
        packing=False,
        max_target_length=256,
        ici_tensor_parallelism=-1,
        ici_context_parallelism=2,
        context_parallel_load_balance=False,
        enable_checkpointing=False,
    )
    self.assertTrue(rl_auto_tp.use_tokamax_gdn)

    # Rejects scanned custom remat host offload (`scan_layers=True` or `scan_layers_per_stage=True`) with
    # `gdn='offload'` or `gdn_states='offload'`.
    for offload_kw in (
        {"gdn_states": "offload"},
        {"gdn": "offload"},
        {"scan_layers": False, "scan_layers_per_stage": True, "gdn_states": "offload"},
    ):
      with self.subTest(offload_kw=offload_kw):
        with self.assertRaisesRegex(ValueError, "does not yet support gdn='offload' or gdn_states='offload'"):
          _gdn_pyconfig(use_tokamax_gdn=True, remat_policy="custom", **offload_kw)
        with self.assertRaisesRegex(ValueError, "does not yet support gdn='offload' or gdn_states='offload'"):
          config_types.RLConfig(
              run_name="bad_rl_scan_offload",
              model_name="qwen3-next-80b-a3b",
              decoder_block="qwen3_next",
              use_tokamax_gdn=True,
              packing=False,
              max_target_length=256,
              remat_policy="custom",
              enable_checkpointing=False,
              **offload_kw,
          )

    # Accepts cp=1 (including gdn_in_proj_kernel_order=True), cp=4 ('auto' & 'head'), scanned device remat, and
    # unscanned (`scan_layers=False`) custom remat offload.
    cfg_cp1 = _gdn_pyconfig(
        use_tokamax_gdn=True,
        gdn_in_proj_kernel_order=True,
        scan_layers=True,
        remat_policy="custom",
        gdn_states="device",
    )
    self.assertTrue(cfg_cp1.use_tokamax_gdn)
    save_dev, offload_dev = maxtext_utils.get_save_and_offload_names(cfg_cp1)
    self.assertIn("gdn_chunk_states", save_dev)
    self.assertEqual(offload_dev, [])
    for decoder_block in ("qwen3_next", "qwen3_5"):
      for cp_mode in ("auto", "head"):
        cfg_cp4 = _gdn_pyconfig(
            decoder_block=decoder_block,
            use_tokamax_gdn=True,
            gdn_num_key_heads=16,
            gdn_num_value_heads=32,
            ici_context_parallelism=4,
            max_target_length=256,
            gdn_cp_mode=cp_mode,
            scan_layers=False,
            remat_policy="custom",
            gdn="remat",
            gdn_states="offload",
        )
        self.assertTrue(cfg_cp4.use_tokamax_gdn)
        save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg_cp4)
        self.assertIn("gdn_chunk_states", offload_names)
        self.assertIn("gdn_core_attn_out", offload_names)
        self.assertNotIn("gdn_chunk_states", save_names)

      rl_ok = config_types.RLConfig(
          run_name="ok_rl_tokamax_gdn",
          model_name="qwen3-next-80b-a3b",
          decoder_block=decoder_block,
          use_tokamax_gdn=True,
          packing=False,
          max_target_length=256,
          ici_context_parallelism=4,
          context_parallel_load_balance=False,
          gdn_cp_mode="auto",
          enable_checkpointing=False,
      )
      self.assertTrue(rl_ok.use_tokamax_gdn)


class TokamaxGdnAdapterGuardTest(unittest.TestCase):
  """Always-running unit tests for adapter guards, backend dispatch, and `resolve_gdn_cp_mode`."""

  def test_constants_match_remat_expansions(self):
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_CHUNK_SIZE, 64)
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME, "gdn_chunk_states")
    self.assertIn(
        tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME,
        maxtext_utils._GDN_REMAT_EXPANSIONS["gdn_states"],  # pylint: disable=protected-access
    )
    self.assertIn(
        tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME,
        maxtext_utils._GDN_REMAT_EXPANSIONS["gdn"],  # pylint: disable=protected-access
    )

  def test_resolve_gdn_cp_mode_direct_attr_and_semantics(self):
    # Direct cfg.use_tokamax_gdn access: a bare namespace without use_tokamax_gdn raises AttributeError.
    bare_cfg: Any = types.SimpleNamespace(gdn_cp_mode="auto")
    with self.assertRaises(AttributeError):
      qwen3.resolve_gdn_cp_mode(bare_cfg, cp_size=2, num_k_heads=4)

    # Preserve existing use_tokamax_gdn=False behavior.
    cfg_false = types.SimpleNamespace(
        use_tokamax_gdn=False,
        gdn_cp_mode="auto",
        context_parallel_load_balance=False,
    )
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_false, 4, 16, prefer_head_cp=False), (False, True))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_false, 2, 16, prefer_head_cp=True), (True, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_false, 4, 16, prefer_head_cp=True), (False, True))

    # With use_tokamax_gdn=True, 'auto' and 'head' select head-CP for all cp_size <= num_k_heads, and 'seq' raises.
    for mode in ("auto", "head"):
      cfg_true = types.SimpleNamespace(
          use_tokamax_gdn=True,
          gdn_cp_mode=mode,
          context_parallel_load_balance=False,
      )
      self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_true, 1, 16), (False, False))
      self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_true, 4, 16), (True, False))
      self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_true, 16, 16), (True, False))
      self.assertEqual(
          qwen3.resolve_gdn_cp_mode(cfg_true, 4, 16, model_mode=MODEL_MODE_AUTOREGRESSIVE),
          (False, False),
      )
      with self.assertRaisesRegex(ValueError, "divisible by cp_size"):
        qwen3.resolve_gdn_cp_mode(cfg_true, 6, 16)

    cfg_seq = types.SimpleNamespace(
        use_tokamax_gdn=True,
        gdn_cp_mode="seq",
        context_parallel_load_balance=False,
    )
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_seq, 1, 16), (False, False))
    with self.assertRaisesRegex(ValueError, "does not yet support gdn_cp_mode='seq'"):
      qwen3.resolve_gdn_cp_mode(cfg_seq, 2, 16)

  def test_missing_tokamax_gated_delta_net_raises_clear_import_error(self):
    q = jnp.zeros((1, 64, 2, 16), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 64, 4, 16), dtype=jnp.bfloat16)
    g = jnp.zeros((1, 64, 4), dtype=jnp.float32)
    b = jnp.zeros((1, 64, 4), dtype=jnp.bfloat16)
    fake_ops = types.ModuleType("tokamax.ops")
    fake_pkg = types.ModuleType("tokamax")
    fake_pkg.ops = fake_ops  # type: ignore[attr-defined]
    with mock.patch.dict(sys.modules, {"tokamax": fake_pkg, "tokamax.ops": fake_ops}):
      with self.assertRaisesRegex(ImportError, "tokamax.ops.gated_delta_net"):
        tokamax_gdn._tokamax_gdn_op()  # pylint: disable=protected-access
      with self.assertRaisesRegex(ImportError, "tokamax.ops.gated_delta_net"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b)

  def test_runtime_input_and_backend_guards(self):
    q = jnp.zeros((1, 64, 2, 16), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 64, 4, 16), dtype=jnp.bfloat16)
    g = jnp.zeros((1, 64, 4), dtype=jnp.float32)
    b = jnp.zeros((1, 64, 4), dtype=jnp.bfloat16)

    with self.assertRaisesRegex(NotImplementedError, "segment_ids"):
      tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, segment_ids=jnp.ones((1, 64), dtype=jnp.int32))
    with self.assertRaisesRegex(NotImplementedError, "cp_axis"):
      tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, cp_axis="context")
    for bad_chunk in (32, 128):
      with self.assertRaisesRegex(NotImplementedError, "chunk_size=64"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, chunk_size=bad_chunk)
    for bad_len in (16, 48):
      with self.assertRaisesRegex(ValueError, "divisible by chunk_size"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q[:, :bad_len],
            q[:, :bad_len],
            v[:, :bad_len],
            g[:, :bad_len],
            b[:, :bad_len],
        )

    # Strict TPU -> "mosaic" dispatch, fast failure on offline CPU->TPU cross-compile, and GPU rejection.
    captured_kwargs = {}

    def fake_gdn(**kwargs):
      captured_kwargs.update(kwargs)
      return kwargs["value"], (kwargs["initial_state"] if kwargs["output_final_state"] else None)

    with (
        mock.patch.object(tokamax_gdn, "_tokamax_gdn_op", return_value=fake_gdn),
        mock.patch.object(jax, "default_backend", return_value="tpu"),
    ):
      # Trace through jax.jit so `query_c` is a Tracer and uses `jax.default_backend()`.
      _ = jax.jit(tokamax_gdn.tokamax_chunk_gated_delta_rule)(q, q, v, g, b)
      self.assertEqual(captured_kwargs["implementation"], "mosaic")
      self.assertFalse(captured_kwargs["use_gate_in_kernel"])
      self.assertTrue(captured_kwargs["use_beta_sigmoid_in_kernel"])
      self.assertEqual(captured_kwargs["residual_checkpoint_name"], "gdn_chunk_states")

    # Offline CPU->TPU cross-compilation (`default_backend() == "cpu"` with an active TPU mesh) fails fast instead
    # of silently lowering `"xla"` or interpret-mode loops.
    tpu7x_amesh = jax.sharding.AbstractMesh(
        (2, 4),
        ("data", "context"),
        axis_types=(AxisType.Auto, AxisType.Auto),
        abstract_device=_FakeAbstractDevice("TPU7x", 1, "tpu"),
    )
    q_spec = jax.ShapeDtypeStruct(q.shape, q.dtype)
    v_spec = jax.ShapeDtypeStruct(v.shape, v.dtype)
    g_spec = jax.ShapeDtypeStruct(g.shape, g.dtype)
    b_spec = jax.ShapeDtypeStruct(b.shape, b.dtype)
    jax.clear_caches()
    with (
        mock.patch.object(tokamax_gdn, "_tokamax_gdn_op", return_value=fake_gdn),
        mock.patch.object(jax, "default_backend", return_value="cpu"),
        jax.sharding.use_abstract_mesh(tpu7x_amesh),
    ):
      with self.assertRaisesRegex(NotImplementedError, "does not yet support offline CPU->TPU cross-compilation"):
        jax.jit(tokamax_gdn.tokamax_chunk_gated_delta_rule).lower(q_spec, q_spec, v_spec, g_spec, b_spec)

    jax.clear_caches()
    with (
        mock.patch.object(tokamax_gdn, "_tokamax_gdn_op", return_value=fake_gdn),
        mock.patch.object(jax, "default_backend", return_value="gpu"),
    ):
      with self.assertRaisesRegex(NotImplementedError, "Tokamax GDN Mosaic kernel requires TPU"):
        jax.jit(tokamax_gdn.tokamax_chunk_gated_delta_rule)(q, q, v, g, b)


@unittest.skipUnless(_HAS_TOKAMAX_GDN, "tokamax.ops.gated_delta_net is not available")
class TokamaxGdnCoreParityTest(unittest.TestCase):
  """Multi-chunk forward + backward parity of `tokamax_chunk_gated_delta_rule` vs `jax_chunk_gated_delta_rule`."""

  def _make_inputs(self, b=2, t=128, h_k=2, h_v=4, d_k=32, d_v=32, dtype=jnp.float32, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    return (
        jax.random.normal(ks[0], (b, t, h_k, d_k), dtype=dtype) * 0.3,
        jax.random.normal(ks[1], (b, t, h_k, d_k), dtype=dtype) * 0.3,
        jax.random.normal(ks[2], (b, t, h_v, d_v), dtype=dtype) * 0.3,
        -jax.nn.softplus(jax.random.normal(ks[3], (b, t, h_v), dtype=jnp.float32)),
        jax.random.normal(ks[4], (b, t, h_v), dtype=dtype),
        jax.random.normal(ks[5], (b, h_v, d_k, d_v), dtype=dtype) * 0.1,
    )

  def test_multichunk_forward_and_backward_parity(self):
    q, k, v, g, b_raw, h0 = self._make_inputs(t=128)
    w_out = jax.random.normal(jax.random.PRNGKey(99), v.shape, dtype=jnp.float32)
    w_h = jax.random.normal(jax.random.PRNGKey(100), h0.shape, dtype=jnp.float32)

    for use_qk_norm in (True, False):
      for with_state in (False, True):
        with self.subTest(use_qk_norm_in_gdn=use_qk_norm, with_initial_state=with_state):
          if not with_state:

            def ref_fn(q_i, k_i, v_i, g_i, b_i, norm=use_qk_norm):
              out, st = qwen3.jax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  jax.nn.sigmoid(b_i.astype(jnp.float32)),
                  chunk_size=64,
                  initial_state=None,
                  use_qk_norm_in_gdn=norm,
                  compute_dtype=jnp.float32,
              )
              return jnp.sum(out * w_out), (out, st)

            def tk_fn(q_i, k_i, v_i, g_i, b_i, norm=use_qk_norm):
              out, st = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  b_i,
                  chunk_size=64,
                  initial_state=None,
                  output_final_state=False,
                  use_qk_norm_in_gdn=norm,
                  compute_dtype=jnp.float32,
              )
              return jnp.sum(out * w_out), (out, st)

            (_, (ref_o, ref_st)), ref_grads = jax.value_and_grad(ref_fn, argnums=tuple(range(5)), has_aux=True)(
                q, k, v, g, b_raw
            )
            (_, (tk_o, tk_st)), tk_grads = jax.value_and_grad(tk_fn, argnums=tuple(range(5)), has_aux=True)(
                q, k, v, g, b_raw
            )
            self.assertIsNone(ref_st)
            self.assertIsNone(tk_st)
            self.assertLess(_rel_err(tk_o, ref_o), 1e-4)
            for name, g_tk, g_ref in zip(("q", "k", "v", "g", "b"), tk_grads, ref_grads):
              self.assertLess(_rel_err(g_tk, g_ref), 2e-4, name)
          else:

            def ref_state_fn(q_i, k_i, v_i, g_i, b_i, h_i, norm=use_qk_norm):
              out, st = qwen3.jax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  jax.nn.sigmoid(b_i.astype(jnp.float32)),
                  chunk_size=64,
                  initial_state=h_i,
                  use_qk_norm_in_gdn=norm,
                  compute_dtype=jnp.float32,
              )
              assert st is not None
              return jnp.sum(out * w_out) + jnp.sum(st * w_h), (out, st)

            def tk_state_fn(q_i, k_i, v_i, g_i, b_i, h_i, norm=use_qk_norm):
              out, st = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  b_i,
                  chunk_size=64,
                  initial_state=h_i,
                  output_final_state=True,
                  use_qk_norm_in_gdn=norm,
                  compute_dtype=jnp.float32,
              )
              assert st is not None
              return jnp.sum(out * w_out) + jnp.sum(st * w_h), (out, st)

            (_, (ref_o, ref_st)), ref_grads = jax.value_and_grad(ref_state_fn, argnums=tuple(range(6)), has_aux=True)(
                q, k, v, g, b_raw, h0
            )
            (_, (tk_o, tk_st)), tk_grads = jax.value_and_grad(tk_state_fn, argnums=tuple(range(6)), has_aux=True)(
                q, k, v, g, b_raw, h0
            )
            self.assertLess(_rel_err(tk_o, ref_o), 1e-4)
            self.assertLess(_rel_err(tk_st, ref_st), 1e-4)
            for name, g_tk, g_ref in zip(("q", "k", "v", "g", "b", "h0"), tk_grads, ref_grads):
              self.assertLess(_rel_err(g_tk, g_ref), 2e-4, name)


@unittest.skipUnless(_HAS_TOKAMAX_GDN, "tokamax.ops.gated_delta_net is not available")
class TokamaxGdnLayerParityTest(unittest.TestCase):
  """End-to-end `Qwen3NextGatedDeltaNet` parity across unsharded, CP meshes, padding, and prefill."""

  def _make_pair(self, **overrides):
    cfg_jax = _gdn_pyconfig(use_tokamax_gdn=False, **overrides)
    cfg_tk = _gdn_pyconfig(use_tokamax_gdn=True, **overrides)
    layer_jax = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax, mesh=None, dtype=cfg_jax.dtype, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=3)
    )
    layer_tk = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk, mesh=None, dtype=cfg_tk.dtype, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=3)
    )
    return layer_jax, layer_tk

  def test_unsharded_forward_and_all_param_grads_match(self):
    layer_jax, layer_tk = self._make_pair()
    x = jax.random.normal(jax.random.PRNGKey(11), (2, 128, 32), dtype=jnp.float32) * 0.2
    dy = jax.random.normal(jax.random.PRNGKey(12), (2, 128, 32), dtype=jnp.float32)

    def eval_layer(layer):
      gdef, st = nnx.split(layer)

      def loss(s, inp):
        out, _ = nnx.merge(gdef, s)(inp, model_mode=MODEL_MODE_TRAIN)
        return jnp.sum(out * dy), out

      (_, out), (g_st, g_x) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(st, x)
      return out, g_st, g_x

    out_jax, grads_jax, dx_jax = eval_layer(layer_jax)
    out_tk, grads_tk, dx_tk = eval_layer(layer_tk)
    self.assertLess(_rel_err(out_tk, out_jax), 1e-4)
    self.assertLess(_rel_err(dx_tk, dx_jax), 2e-4)

    expected_submodules = {"in_proj_qkvz", "in_proj_ba", "conv1d", "A_log", "dt_bias", "norm", "out_proj"}
    found_submodules = set()
    for (path, g_j), (_, g_t) in zip(
        jax.tree_util.tree_leaves_with_path(grads_jax),
        jax.tree_util.tree_leaves_with_path(grads_tk),
    ):
      key_str = jax.tree_util.keystr(path)
      for name in expected_submodules:
        if name in key_str:
          found_submodules.add(name)
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_t))), key_str)
      self.assertGreater(float(jnp.linalg.norm(g_t)), 0.0, key_str)
      self.assertLess(_rel_err(g_t, g_j), 2e-4, key_str)
    self.assertEqual(found_submodules, expected_submodules)

  def test_2d_mesh_cp_auto_and_head_in_auto_and_explicit_shard_modes(self):
    devs = jax.devices()
    if len(devs) >= 4:
      _run_2d_cp_mesh_parity_checks(devs)
    else:
      env = os.environ.copy()
      env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
      env["JAX_PLATFORMS"] = "cpu"
      res = subprocess.run(
          [sys.executable, __file__, "--run-cp-mesh-checks"],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res.returncode, 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
      self.assertIn("TOKAMAX_GDN_CP_MESH_OK", res.stdout)

  def test_left_padding_isolation_and_grad_parity(self):
    layer_jax, layer_tk = self._make_pair()
    k_real, k_pad1, k_pad2, k_dy = jax.random.split(jax.random.PRNGKey(21), 4)
    real_tokens = jax.random.normal(k_real, (2, 108, 32), dtype=jnp.float32) * 0.2
    pad_a = jax.random.normal(k_pad1, (2, 20, 32), dtype=jnp.float32) * 5.0
    pad_b = jax.random.normal(k_pad2, (2, 20, 32), dtype=jnp.float32) * -7.0
    x_a = jnp.concatenate([pad_a, real_tokens], axis=1)
    x_b = jnp.concatenate([pad_b, real_tokens], axis=1)
    seg = jnp.concatenate([jnp.zeros((2, 20), jnp.int32), jnp.ones((2, 108), jnp.int32)], axis=1)
    dy = jax.random.normal(k_dy, (2, 108, 32), dtype=jnp.float32)

    out_tk_a, _ = layer_tk(x_a, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
    out_tk_b, _ = layer_tk(x_b, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
    out_jax_a, _ = layer_jax(x_a, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)

    # Real token outputs are immune to arbitrary garbage on left-padding positions.
    np.testing.assert_allclose(np.asarray(out_tk_a[:, 20:]), np.asarray(out_tk_b[:, 20:]), rtol=1e-6, atol=1e-6)
    self.assertLess(_rel_err(out_tk_a[:, 20:], out_jax_a[:, 20:]), 1e-4)

    def padded_loss(layer):
      out, _ = layer(x_a, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out[:, 20:] * dy)

    g_jax = nnx.state(nnx.grad(padded_loss)(layer_jax), nnx.Param)
    g_tk = nnx.state(nnx.grad(padded_loss)(layer_tk), nnx.Param)
    for (path, leaf_jax), (_, leaf_tk) in zip(
        jax.tree_util.tree_leaves_with_path(g_jax),
        jax.tree_util.tree_leaves_with_path(g_tk),
    ):
      self.assertLess(_rel_err(leaf_tk, leaf_jax), 2e-4, jax.tree_util.keystr(path))

  def test_prefill_active_cache_aligned_and_fallback_paths(self):
    cfg_jax = _gdn_pyconfig(use_tokamax_gdn=False)
    cfg_tk = _gdn_pyconfig(use_tokamax_gdn=True)
    x64 = jax.random.normal(jax.random.PRNGKey(31), (2, 64, 32), dtype=jnp.float32) * 0.2
    x1 = jax.random.normal(jax.random.PRNGKey(32), (2, 1, 32), dtype=jnp.float32) * 0.2

    layer_jax = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax, inputs_shape=x64.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )
    layer_tk = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk, inputs_shape=x64.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )

    # 1. Chunk-aligned unpacked prefill (T=64) dispatches to Tokamax and updates active_cache identically.
    with mock.patch.object(
        tokamax_gdn,
        "tokamax_chunk_gated_delta_rule",
        wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
    ) as spy_tk:
      out_tk_64, cache_tk = layer_tk(x64, model_mode=MODEL_MODE_PREFILL)
      self.assertEqual(spy_tk.call_count, 1)
    out_jax_64, cache_jax = layer_jax(x64, model_mode=MODEL_MODE_PREFILL)
    self.assertLess(_rel_err(out_tk_64, out_jax_64), 1e-4)
    rec_tk, conv_tk = cache_tk.get_gdn_states()
    rec_jax, conv_jax = cache_jax.get_gdn_states()
    self.assertLess(_rel_err(rec_tk, rec_jax), 1e-4)
    self.assertLess(_rel_err(conv_tk, conv_jax), 1e-5)

    # Subsequent AR decode step (T=1) from the Tokamax-populated cache matches Pure JAX.
    ar_tk, _ = layer_tk(x1, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    ar_jax, _ = layer_jax(x1, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    self.assertLess(_rel_err(ar_tk, ar_jax), 1e-4)

    # 2. Unaligned prefill bucket (T=32) and segmented prefill cleanly use Pure JAX without error.
    x32 = x64[:, :32, :]
    layer_jax_32 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax, inputs_shape=x32.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )
    layer_tk_32 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk, inputs_shape=x32.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )
    with mock.patch.object(
        tokamax_gdn,
        "tokamax_chunk_gated_delta_rule",
        wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
    ) as spy_unaligned:
      out_tk_32, _ = layer_tk_32(x32, model_mode=MODEL_MODE_PREFILL)
      self.assertEqual(spy_unaligned.call_count, 0)
    out_jax_32, _ = layer_jax_32(x32, model_mode=MODEL_MODE_PREFILL)
    self.assertLess(_rel_err(out_tk_32, out_jax_32), 1e-5)

    seg64 = jnp.concatenate(
        [jnp.zeros((2, 8), jnp.int32), jnp.ones((2, 48), jnp.int32), jnp.zeros((2, 8), jnp.int32)], axis=1
    )
    layer_jax_seg = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax, inputs_shape=x64.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )
    layer_tk_seg = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk, inputs_shape=x64.shape, dtype=jnp.float32, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=9)
    )
    with mock.patch.object(
        tokamax_gdn,
        "tokamax_chunk_gated_delta_rule",
        wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
    ) as spy_seg:
      out_tk_seg, cache_tk_seg = layer_tk_seg(x64, decoder_segment_ids=seg64, model_mode=MODEL_MODE_PREFILL)
      self.assertEqual(spy_seg.call_count, 0)
    out_jax_seg, cache_jax_seg = layer_jax_seg(x64, decoder_segment_ids=seg64, model_mode=MODEL_MODE_PREFILL)
    self.assertLess(_rel_err(out_tk_seg, out_jax_seg), 1e-5)
    self.assertLess(_rel_err(cache_tk_seg.get_gdn_states()[0], cache_jax_seg.get_gdn_states()[0]), 1e-5)


@unittest.skipUnless(_HAS_TOKAMAX_GDN, "tokamax.ops.gated_delta_net is not available")
class TokamaxGdnRematTest(unittest.TestCase):
  """Verifies `jax.checkpoint` remat behavior and offline cross-compile guard."""

  def _tpu7x_abstract_mesh(self) -> jax.sharding.AbstractMesh:
    return jax.sharding.AbstractMesh(
        (2, 4),
        ("data", "context"),
        axis_types=(AxisType.Auto, AxisType.Auto),
        abstract_device=_FakeAbstractDevice("TPU7x", 1, "tpu"),
    )

  def test_full_remat_leaks_zero_intermediates_and_custom_policy_saves_residuals(self):
    cfg = _gdn_pyconfig(
        use_tokamax_gdn=True,
        base_emb_dim=64,
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        dtype="bfloat16",
    )
    layer = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg, mesh=None, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=13)
    )
    gdef, state = nnx.split(layer)
    x = jax.random.normal(jax.random.PRNGKey(41), (1, 128, 64), dtype=jnp.bfloat16) * 0.2

    def layer_fn(st, inp):
      out, _ = nnx.merge(gdef, st)(inp, model_mode=MODEL_MODE_TRAIN)
      return inp + out * jnp.tanh(out)

    # 1. Under save-nothing `jax.checkpoint(layer_fn)`, zero kernel intermediates leak.
    buf_full = io.StringIO()
    with contextlib.redirect_stdout(buf_full):
      ad_checkpoint.print_saved_residuals(
          lambda st, inp: jnp.sum(jax.checkpoint(layer_fn)(st, inp).astype(jnp.float32)), state, x
      )
    saved_full = [line.strip() for line in buf_full.getvalue().splitlines() if line.strip()]
    self.assertTrue(saved_full, "print_saved_residuals printed nothing")
    leaked = [line for line in saved_full if "from the argument" not in line and "from a constant" not in line]
    self.assertEqual(leaked, [])

    # 2. Under selective policy saving ["gdn_core_attn_out", "gdn_chunk_states"], `gdn_chunk_states` is saved!
    selective_policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["gdn_core_attn_out", "gdn_chunk_states"],
        names_which_can_be_offloaded=[],
        offload_src="device",
        offload_dst="pinned_host",
    )
    w_proj = jax.random.normal(jax.random.PRNGKey(42), (64, 64), dtype=jnp.bfloat16) * 0.2

    def core_step(w, inp):
      h = inp @ w
      q = jnp.repeat(h[:, :, :32].reshape(1, 128, 1, 32), 4, axis=3).reshape(1, 128, 2, 64)
      q = jnp.concatenate([q, q], axis=-1)
      v = jnp.repeat(q, 2, axis=2)
      g = -jax.nn.softplus(h[:, :, :4].astype(jnp.float32))
      b = h[:, :, 4:8]
      o, _ = tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, chunk_size=64, compute_dtype=jnp.bfloat16)
      return inp + jnp.mean(o[:, :, :2, :] * jnp.tanh(o[:, :, 2:, :]), axis=(2, 3))[..., None] * inp

    buf_core = io.StringIO()
    with contextlib.redirect_stdout(buf_core):
      ad_checkpoint.print_saved_residuals(
          lambda w, inp: jnp.sum(jax.checkpoint(core_step, policy=selective_policy)(w, inp).astype(jnp.float32)),
          w_proj,
          x,
      )
    saved_core = [line.strip() for line in buf_core.getvalue().splitlines() if line.strip()]
    self.assertTrue(any("gdn_chunk_states" in line for line in saved_core), saved_core)

    buf_sel = io.StringIO()
    with contextlib.redirect_stdout(buf_sel):
      ad_checkpoint.print_saved_residuals(
          lambda st, inp: jnp.sum(jax.checkpoint(layer_fn, policy=selective_policy)(st, inp).astype(jnp.float32)),
          state,
          x,
      )
    saved_sel = [line.strip() for line in buf_sel.getvalue().splitlines() if line.strip()]
    self.assertTrue(any("gdn_core_attn_out" in line for line in saved_sel), saved_sel)
    if jax.default_backend() == "tpu":
      self.assertTrue(any("gdn_chunk_states" in line for line in saved_sel), saved_sel)

  def test_offline_cpu_to_tpu_cross_compile_raises_not_implemented(self):
    cfg = _gdn_pyconfig(
        use_tokamax_gdn=True,
        base_emb_dim=64,
        gdn_num_key_heads=4,
        gdn_num_value_heads=8,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        dtype="bfloat16",
    )
    layer = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg, mesh=None, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=19)
    )
    gdef, state = nnx.split(layer)
    state_spec = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
    x_spec = jax.ShapeDtypeStruct((1, 128, 64), jnp.bfloat16)

    def train_loss(st, inp):
      out, _ = nnx.merge(gdef, st)(inp, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out.astype(jnp.float32))

    if jax.default_backend() == "tpu":
      hlo_tpu = jax.jit(jax.value_and_grad(train_loss, argnums=(0, 1))).lower(state_spec, x_spec).as_text()
      self.assertIn("tpu_custom_call", hlo_tpu)
      return

    with jax.sharding.use_abstract_mesh(self._tpu7x_abstract_mesh()):
      with self.assertRaisesRegex(NotImplementedError, "does not yet support offline CPU->TPU cross-compilation"):
        jax.jit(jax.value_and_grad(train_loss, argnums=(0, 1))).lower(state_spec, x_spec)


if __name__ == "__main__":
  if "--run-cp-mesh-checks" in sys.argv:
    setUpModule()
    try:
      _run_2d_cp_mesh_parity_checks(jax.devices())
      print("TOKAMAX_GDN_CP_MESH_OK")
    finally:
      tearDownModule()
  else:
    absltest.main()
