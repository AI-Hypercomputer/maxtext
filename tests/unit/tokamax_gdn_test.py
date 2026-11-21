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
import re
import subprocess
import sys
import types
from typing import Any
import unittest
from unittest import mock

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

# pylint: disable=wrong-import-position
from absl import app
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


def _eval_unsharded_ref(cfg_ref, x, dy, seg_ids=None, seed=5):
  """Evaluates unsharded Pure-JAX `Qwen3NextGatedDeltaNet` forward + backward as ground truth."""
  with nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
    layer_ref = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_ref,
        mesh=None,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=seed),
    )
    gdef_ref, st_ref = nnx.split(layer_ref)

    def ref_loss(st, inp):
      out, _ = nnx.merge(gdef_ref, st)(inp, decoder_segment_ids=seg_ids, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out * dy), out

    (_, ref_out), (ref_grads, ref_dx) = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1), has_aux=True))(st_ref, x)
  return (
      jax.device_get(ref_out),
      jax.tree_util.tree_map(jax.device_get, ref_grads),
      jax.device_get(ref_dx),
  )


def _assert_layer_parity(
    tag: str,
    out_got,
    grads_got,
    dx_got,
    ref_out,
    ref_grads,
    ref_dx,
    *,
    tol: float = 3e-4,
    grad_atol: float = 1e-4,
    grad_rtol: float = 3e-3,
) -> None:
  """Asserts forward output, input gradient, and full parameter-gradient parity."""
  assert _rel_err(out_got, ref_out) < tol, f"{tag} output mismatch: {_rel_err(out_got, ref_out)}"
  assert _rel_err(dx_got, ref_dx) < tol, f"{tag} dx mismatch: {_rel_err(dx_got, ref_dx)}"
  seen_keys = set()
  for (path, g_got), (_, g_exp) in zip(
      jax.tree_util.tree_leaves_with_path(grads_got),
      jax.tree_util.tree_leaves_with_path(ref_grads),
  ):
    key_str = jax.tree_util.keystr(path)
    seen_keys.add(key_str)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(g_got)),
        np.asarray(jax.device_get(g_exp)),
        atol=grad_atol,
        rtol=grad_rtol,
        err_msg=f"{tag} grad mismatch on {key_str}",
    )
  assert any("A_log" in k for k in seen_keys), seen_keys
  assert any("dt_bias" in k for k in seen_keys), seen_keys
  assert any("conv1d" in k for k in seen_keys), seen_keys


def _run_cp_mesh_parity_checks(devices) -> None:
  """Verifies `Qwen3NextGatedDeltaNet(use_tokamax_gdn=True)` across head-CP & 4-rank seq-CP (unpacked & packed)."""
  assert len(devices) >= 4, f"Expected >= 4 devices, got {len(devices)}"

  # 1. 2D `(data=2, context=2)` head-CP in AUTO and EXPLICIT shard modes, unpacked and packed (`seg_pspec` check).
  batch, seq_len, emb_dim = 2, 128, 32
  x_2d = jax.random.normal(jax.random.PRNGKey(17), (batch, seq_len, emb_dim), dtype=jnp.float32) * 0.2
  dy_2d = jax.random.normal(jax.random.PRNGKey(18), (batch, seq_len, emb_dim), dtype=jnp.float32)
  packed_seg_128 = jnp.array(
      [
          [1] * 40 + [0] * 8 + [2] * 52 + [3] * 20 + [0] * 8,
          [0] * 12 + [4] * 36 + [0] * 8 + [4] * 32 + [2] * 40,
      ],
      dtype=jnp.int32,
  )

  devs_2x2 = np.asarray(devices[:4]).reshape(2, 2)
  for packing, seg_ids in ((False, None), (True, packed_seg_128)):
    cfg_ref_2d = _gdn_pyconfig(use_tokamax_gdn=False, packing=packing)
    ref_out_2d, ref_grads_2d, ref_dx_2d = _eval_unsharded_ref(cfg_ref_2d, x_2d, dy_2d, seg_ids=seg_ids, seed=5)
    for cp_mode in ("auto", "head"):
      for shard_mode_str in ("auto", "explicit"):
        cfg_cp = _gdn_pyconfig(
            use_tokamax_gdn=True,
            packing=packing,
            ici_data_parallelism=2,
            ici_context_parallelism=2,
            gdn_cp_mode=cp_mode,
            shard_mode="auto",
            num_experts=1,
            sparse_matmul=False,
            megablox=False,
        )
        if shard_mode_str == "explicit":
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
          seg_sh = jax.sharding.NamedSharding(mesh, P("data", "context"))
          x_in = jax.device_put(x_2d, tok_sh)
          dy_in = jax.device_put(dy_2d, tok_sh)
          seg_in = jax.device_put(seg_ids, seg_sh) if seg_ids is not None else None

          def head_cp_loss(
              st,
              inp,
              cot,
              seg_arg,
              gdef=gdef_cp,
              out_sh=tok_sh if shard_mode_str == "explicit" else None,
          ):
            out, _ = nnx.merge(gdef, st)(
                inp,
                decoder_segment_ids=seg_arg,
                model_mode=MODEL_MODE_TRAIN,
                out_sharding=out_sh,
            )
            return jnp.sum(out * cot), out

          (_, out_cp), (grads_cp, dx_cp) = jax.jit(jax.value_and_grad(head_cp_loss, argnums=(0, 1), has_aux=True))(
              st_cp, x_in, dy_in, seg_in
          )
        _assert_layer_parity(
            f"head-2d packing={packing} cp_mode={cp_mode} shard_mode={shard_mode_str}",
            out_cp,
            grads_cp,
            dx_cp,
            ref_out_2d,
            ref_grads_2d,
            ref_dx_2d,
        )

  # 2. 1D `(context=4)` sequence-sharded CP (`BT=64` and `BT=128`, `K=128, V=128`, unpacked & packed).
  # Packed rows cover:
  #   - Row 0: doc 1 spanning rank 0 -> rank 1 boundary (`[0..199]`), exact rank-boundary reset at `t=256` (rank 2 starts
  #     doc 2), interior padding `[320..339]`, and non-adjacent reused raw doc ID `1` at `[340..511]` across rank 2 -> 3.
  #   - Row 1: leading padding on rank 0 (`[0..23]`), doc 5 spanning across an all-zero-padding rank 1 (`[128..255] == 0`)
  #     into rank 2 (`[256..319] == 5`), followed by doc 7 and trailing padding on rank 3.
  seq_cp4 = 512
  x_cp4 = jax.random.normal(jax.random.PRNGKey(27), (2, seq_cp4, 32), dtype=jnp.float32) * 0.2
  dy_cp4 = jax.random.normal(jax.random.PRNGKey(28), (2, seq_cp4, 32), dtype=jnp.float32)
  row0 = [1] * 200 + [9] * 56 + [2] * 64 + [0] * 20 + [1] * 172
  row1 = [0] * 24 + [5] * 104 + [0] * 128 + [5] * 64 + [7] * 160 + [0] * 32
  assert len(row0) == seq_cp4 and len(row1) == seq_cp4
  packed_seg_cp4 = jnp.asarray([row0, row1], dtype=jnp.int32)

  devs_4 = np.asarray(devices[:4])
  seq_cp_cases = (
      # (chunk_size, packing, cp_mode, shard_mode_str)
      (64, False, "seq", "auto"),
      (128, False, "auto", "auto"),
      (64, True, "seq", "auto"),
      (64, True, "auto", "explicit"),
      (128, True, "seq", "auto"),
  )
  ref_cache: dict[tuple[int, bool], tuple[Any, Any, Any]] = {}
  tk1_cache: dict[tuple[int, bool], tuple[Any, Any, Any]] = {}
  for chunk_size, packing, cp_mode, shard_mode_str in seq_cp_cases:
    seg_ids = packed_seg_cp4 if packing else None
    cache_key = (chunk_size, packing)
    if cache_key not in ref_cache:
      cfg_ref_seq = _gdn_pyconfig(
          use_tokamax_gdn=False,
          packing=packing,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          gdn_chunk_size=chunk_size,
          max_target_length=seq_cp4,
      )
      ref_cache[cache_key] = _eval_unsharded_ref(cfg_ref_seq, x_cp4, dy_cp4, seg_ids=seg_ids, seed=7)
      cfg_tk1_seq = _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_two_pass_bf16=True,
          packing=packing,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          gdn_chunk_size=chunk_size,
          max_target_length=seq_cp4,
      )
      tk1_cache[cache_key] = _eval_unsharded_ref(cfg_tk1_seq, x_cp4, dy_cp4, seg_ids=seg_ids, seed=7)
    ref_out_s, ref_grads_s, ref_dx_s = ref_cache[cache_key]
    tk1_out_s, tk1_grads_s, tk1_dx_s = tk1_cache[cache_key]

    cfg_seq_cp = _gdn_pyconfig(
        use_tokamax_gdn=True,
        gdn_two_pass_bf16=True,
        packing=packing,
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_chunk_size=chunk_size,
        max_target_length=seq_cp4,
        ici_context_parallelism=4,
        gdn_cp_mode=cp_mode,
        shard_mode="auto",
        num_experts=1,
        sparse_matmul=False,
        megablox=False,
    )
    if shard_mode_str == "explicit":
      object.__getattribute__(cfg_seq_cp, "_flat_config")["shard_mode"] = config_types.ShardMode.EXPLICIT
    mesh_4 = (
        Mesh(devs_4, ("context",), axis_types=(AxisType.Explicit,))
        if shard_mode_str == "explicit"
        else Mesh(devs_4, ("context",))
    )
    with jax.set_mesh(mesh_4), nn_partitioning.axis_rules(cfg_seq_cp.logical_axis_rules):
      layer_seq = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_seq_cp,
          mesh=mesh_4,
          dtype=jnp.float32,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=7),
      )
      gdef_seq, st_seq = nnx.split(layer_seq)
      st_seq = jax.device_put(st_seq, nnx_construct_named_sharding(st_seq, mesh_4))
      tok_sh_4 = jax.sharding.NamedSharding(mesh_4, P(None, "context", None))
      seg_sh_4 = jax.sharding.NamedSharding(mesh_4, P(None, "context"))
      x_s_in = jax.device_put(x_cp4, tok_sh_4)
      dy_s_in = jax.device_put(dy_cp4, tok_sh_4)
      seg_s_in = jax.device_put(seg_ids, seg_sh_4) if seg_ids is not None else None

      def seq_cp_loss(
          st,
          inp,
          cot,
          seg_arg,
          gdef=gdef_seq,
          out_sh=tok_sh_4 if shard_mode_str == "explicit" else None,
      ):
        out, _ = nnx.merge(gdef, st)(
            inp,
            decoder_segment_ids=seg_arg,
            model_mode=MODEL_MODE_TRAIN,
            out_sharding=out_sh,
        )
        return jnp.sum(out * cot), out

      (_, out_s), (grads_s, dx_s) = jax.jit(jax.value_and_grad(seq_cp_loss, argnums=(0, 1), has_aux=True))(
          st_seq, x_s_in, dy_s_in, seg_s_in
      )
    _assert_layer_parity(
        f"seq-cp4 vs pure-jax bt={chunk_size} packing={packing} cp_mode={cp_mode} shard_mode={shard_mode_str}",
        out_s,
        grads_s,
        dx_s,
        ref_out_s,
        ref_grads_s,
        ref_dx_s,
        tol=3e-4,
    )
    _assert_layer_parity(
        f"seq-cp4 vs cp1-tokamax bt={chunk_size} packing={packing} cp_mode={cp_mode} shard_mode={shard_mode_str}",
        out_s,
        grads_s,
        dx_s,
        tk1_out_s,
        tk1_grads_s,
        tk1_dx_s,
        tol=1e-4,
        grad_atol=1e-5,
        grad_rtol=1e-4,
    )


class TokamaxGdnConfigValidationTest(unittest.TestCase):
  """Always-running config validation tests across `pyconfig`, `MaxTextConfig`, and `RLConfig`."""

  def test_defaults_are_false(self):
    cfg = _gdn_pyconfig()
    self.assertFalse(cfg.use_tokamax_gdn)
    self.assertFalse(cfg.gdn_two_pass_bf16)
    mt_cfg = config_types.MaxTextConfig(run_name="mt_default")
    self.assertFalse(mt_cfg.use_tokamax_gdn)
    self.assertFalse(mt_cfg.gdn_two_pass_bf16)
    rl_cfg = config_types.RLConfig(run_name="rl_default", model_name="qwen3-4b")
    self.assertFalse(rl_cfg.use_tokamax_gdn)
    self.assertFalse(rl_cfg.gdn_two_pass_bf16)

  def test_rejects_non_gdn_decoder_block_and_standalone_two_pass_bf16(self):
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
        with self.assertRaisesRegex(ValueError, "gdn_two_pass_bf16=True requires a GatedDeltaNet decoder_block"):
          config_types.MaxTextConfig(run_name="bad_mt_2p_block", decoder_block=decoder_block, gdn_two_pass_bf16=True)

    for decoder_block in ("qwen3_next", "qwen3_5"):
      with self.subTest(two_pass_without_tokamax=decoder_block):
        with self.assertRaisesRegex(ValueError, "gdn_two_pass_bf16=True requires use_tokamax_gdn=True"):
          _gdn_pyconfig(decoder_block=decoder_block, use_tokamax_gdn=False, gdn_two_pass_bf16=True)
        with self.assertRaisesRegex(ValueError, "gdn_two_pass_bf16=True requires use_tokamax_gdn=True"):
          config_types.RLConfig(
              run_name="bad_rl_2p_off",
              model_name="qwen3-next-80b-a3b",
              decoder_block=decoder_block,
              packing=False,
              use_tokamax_gdn=False,
              gdn_two_pass_bf16=True,
              enable_checkpointing=False,
          )
        cfg_2p = _gdn_pyconfig(decoder_block=decoder_block, use_tokamax_gdn=True, gdn_two_pass_bf16=True)
        self.assertTrue(cfg_2p.gdn_two_pass_bf16)

  def test_chunk_size_and_sequence_packing_validations(self):
    # Unpacked scalar-gate GDN accepts powers of two in [16, 512].
    for valid_unpacked_cs in (16, 32, 64, 128, 256, 512):
      cfg_ok = _gdn_pyconfig(
          use_tokamax_gdn=True,
          packing=False,
          gdn_chunk_size=valid_unpacked_cs,
          max_target_length=512,
      )
      self.assertEqual(cfg_ok.gdn_chunk_size, valid_unpacked_cs)

    for bad_unpacked_cs in (8, 48, 96, 1024):
      with self.subTest(bad_unpacked_cs=bad_unpacked_cs):
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size in"):
          _gdn_pyconfig(use_tokamax_gdn=True, packing=False, gdn_chunk_size=bad_unpacked_cs, max_target_length=1024)
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size in"):
          config_types.RLConfig(
              run_name="bad_rl_unpacked_chunk",
              model_name="qwen3-next-80b-a3b",
              decoder_block="qwen3_next",
              use_tokamax_gdn=True,
              packing=False,
              gdn_chunk_size=bad_unpacked_cs,
              max_target_length=1024,
              enable_checkpointing=False,
          )

    # Packed (`packing=True` or `RLConfig` `max_seq_token_per_tpu > 0`) accepts (16, 32, 64, 128) and rejects (256, 512).
    for decoder_block in ("qwen3_next", "qwen3_5"):
      for valid_packed_cs in (16, 32, 64, 128):
        cfg_pack_ok = _gdn_pyconfig(
            decoder_block=decoder_block,
            use_tokamax_gdn=True,
            packing=True,
            gdn_chunk_size=valid_packed_cs,
            max_target_length=256,
        )
        self.assertTrue(cfg_pack_ok.packing)
        rl_pack_ok = config_types.RLConfig(
            run_name="ok_rl_pack",
            model_name="qwen3-next-80b-a3b",
            decoder_block=decoder_block,
            use_tokamax_gdn=True,
            packing=True,
            gdn_chunk_size=valid_packed_cs,
            max_target_length=256,
            max_seq_token_per_tpu=256,
            enable_checkpointing=False,
        )
        self.assertTrue(rl_pack_ok.packing)

      for bad_packed_cs in (256, 512):
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size in \\(16, 32, 64, 128\\)"):
          _gdn_pyconfig(
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
              packing=True,
              gdn_chunk_size=bad_packed_cs,
              max_target_length=512,
          )
        with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size in \\(16, 32, 64, 128\\)"):
          config_types.RLConfig(
              run_name="bad_rl_packed_cs",
              model_name="qwen3-next-80b-a3b",
              decoder_block=decoder_block,
              use_tokamax_gdn=True,
              packing=False,
              gdn_chunk_size=bad_packed_cs,
              max_target_length=512,
              max_seq_token_per_tpu=512,
              enable_checkpointing=False,
          )

    for chunk_size in (64, 128):
      with self.assertRaisesRegex(ValueError, "divisible by gdn_chunk_size"):
        _gdn_pyconfig(use_tokamax_gdn=True, gdn_chunk_size=chunk_size, max_target_length=180)
      with self.assertRaisesRegex(ValueError, "divisible by gdn_chunk_size"):
        config_types.RLConfig(
            run_name="bad_rl_seq",
            model_name="qwen3-next-80b-a3b",
            decoder_block="qwen3_next",
            use_tokamax_gdn=True,
            packing=True,
            gdn_chunk_size=chunk_size,
            max_target_length=180,
            enable_checkpointing=False,
        )

  def test_context_parallel_validations_and_acceptances(self):
    # 1. Head-CP checks (`gdn_cp_mode='head'`, plus `'auto'` at cp=2 when num_k_heads % 2 == 0).
    with self.assertRaisesRegex(ValueError, "divisible by the GatedDeltaNet context-parallel size"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_num_key_heads=6,
          gdn_num_value_heads=12,
          ici_context_parallelism=4,
          gdn_cp_mode="head",
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
          gdn_cp_mode="head",
          enable_checkpointing=False,
      )

    with self.assertRaisesRegex(ValueError, "max_target_length .* to be divisible by the GatedDeltaNet"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_num_key_heads=12,
          gdn_num_value_heads=24,
          max_target_length=64,
          ici_context_parallelism=12,
          gdn_cp_mode="head",
      )

    # `gdn_in_proj_kernel_order=True` is rejected when head-CP is resolved (`cp=2` in 'auto'/'head'),
    # and accepted when seq-CP is resolved (`cp=4` in 'auto'/'seq' with K=128, V=128).
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
    cfg_seq_kernel_order = _gdn_pyconfig(
        use_tokamax_gdn=True,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        max_target_length=256,
        ici_context_parallelism=4,
        gdn_cp_mode="auto",
        gdn_in_proj_kernel_order=True,
    )
    self.assertTrue(cfg_seq_kernel_order.gdn_in_proj_kernel_order)

    # Rejects combined TP * CP > gdn_num_key_heads when head-CP is active (`cp=2` in 'auto' or 'head').
    for cp_mode in ("auto", "head"):
      with self.assertRaisesRegex(ValueError, "divisible by total GatedDeltaNet head shards"):
        _gdn_pyconfig(
            use_tokamax_gdn=True,
            gdn_num_key_heads=2,
            gdn_num_value_heads=4,
            ici_tensor_parallelism=2,
            ici_context_parallelism=2,
            gdn_cp_mode=cp_mode,
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
            gdn_cp_mode=cp_mode,
            context_parallel_load_balance=False,
            enable_checkpointing=False,
        )

    # 2. Seq-CP validations (`gdn_cp_mode='seq'`, `'auto'` with cp=4, and `'auto'` with odd gdn_num_key_heads).
    for bad_dims in (
        {"gdn_key_head_dim": 64, "gdn_value_head_dim": 128},
        {"gdn_key_head_dim": 128, "gdn_value_head_dim": 64},
    ):
      with self.assertRaisesRegex(ValueError, "multiples of 128"):
        _gdn_pyconfig(
            use_tokamax_gdn=True,
            ici_context_parallelism=2,
            gdn_cp_mode="seq",
            max_target_length=256,
            **bad_dims,
        )
      with self.assertRaisesRegex(ValueError, "multiples of 128"):
        _gdn_pyconfig(
            use_tokamax_gdn=True,
            ici_context_parallelism=4,
            gdn_cp_mode="auto",
            max_target_length=256,
            **bad_dims,
        )
      with self.assertRaisesRegex(ValueError, "multiples of 128"):
        config_types.RLConfig(
            run_name="bad_rl_seq_cp_dim",
            model_name="qwen3-next-80b-a3b",
            decoder_block="qwen3_next",
            use_tokamax_gdn=True,
            packing=True,
            ici_context_parallelism=2,
            context_parallel_load_balance=False,
            gdn_cp_mode="seq",
            max_target_length=256,
            enable_checkpointing=False,
            **bad_dims,
        )

    # Rejects multi-axis CP (`ici_cp + dcn_cp` or `ici_cp + ici_usp`) under seq-CP.
    with self.assertRaisesRegex(ValueError, "at most one active context-parallel mesh axis"):
      config_types.MaxTextConfig(
          run_name="bad_multi_axis_dcn",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          num_slices=2,
          ici_context_parallelism=2,
          dcn_context_parallelism=2,
          context_parallel_load_balance=False,
          gdn_cp_mode="seq",
          max_target_length=512,
          enable_checkpointing=False,
      )
    with self.assertRaisesRegex(ValueError, "at most one active context-parallel mesh axis"):
      config_types.RLConfig(
          run_name="bad_multi_axis_usp",
          model_name="qwen3-next-80b-a3b",
          decoder_block="qwen3_next",
          use_tokamax_gdn=True,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          ici_context_parallelism=2,
          ici_context_usp_ulysses_parallelism=2,
          context_parallel_strategy="usp",
          context_parallel_load_balance=False,
          gdn_cp_mode="auto",
          max_target_length=512,
          enable_checkpointing=False,
      )

    # Rejects `gdn_chunk_size in (256, 512)` and `max_target_length % (cp_size * gdn_chunk_size) != 0` under seq-CP.
    with self.assertRaisesRegex(ValueError, "requires gdn_chunk_size in \\(16, 32, 64, 128\\)"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          gdn_chunk_size=256,
          max_target_length=1024,
          ici_context_parallelism=4,
          gdn_cp_mode="auto",
      )
    with self.assertRaisesRegex(ValueError, "divisible by context_parallel_size \\* gdn_chunk_size"):
      _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          gdn_chunk_size=128,
          max_target_length=256,
          ici_context_parallelism=4,
          gdn_cp_mode="seq",
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
        packing=True,
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

    # Accepts cp=1, cp=4 ('auto', 'head', and 'seq' with BT=128 + packing=True + gdn_cp_state='device'), and RLConfig.
    cfg_cp1 = _gdn_pyconfig(
        use_tokamax_gdn=True,
        gdn_two_pass_bf16=True,
        gdn_chunk_size=128,
        packing=True,
        gdn_in_proj_kernel_order=True,
        scan_layers=True,
        remat_policy="custom",
        gdn_states="device",
        gdn_cp_state="device",
    )
    self.assertTrue(cfg_cp1.use_tokamax_gdn)
    save_dev, offload_dev = maxtext_utils.get_save_and_offload_names(cfg_cp1)
    self.assertIn("gdn_chunk_states", save_dev)
    self.assertIn("gdn_cp_s_in", save_dev)
    self.assertEqual(offload_dev, [])
    for decoder_block in ("qwen3_next", "qwen3_5"):
      for cp_mode in ("auto", "head", "seq"):
        cfg_cp4 = _gdn_pyconfig(
            decoder_block=decoder_block,
            use_tokamax_gdn=True,
            gdn_two_pass_bf16=True,
            packing=True,
            gdn_num_key_heads=16,
            gdn_num_value_heads=32,
            gdn_key_head_dim=128,
            gdn_value_head_dim=128,
            gdn_chunk_size=128,
            ici_context_parallelism=4,
            max_target_length=512,
            gdn_cp_mode=cp_mode,
            scan_layers=False,
            remat_policy="custom",
            gdn="remat",
            gdn_states="offload",
            gdn_cp_state="device",
        )
        self.assertTrue(cfg_cp4.use_tokamax_gdn)
        save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg_cp4)
        self.assertIn("gdn_chunk_states", offload_names)
        self.assertIn("gdn_core_attn_out", offload_names)
        self.assertIn("gdn_cp_s_in", save_names)
        self.assertNotIn("gdn_chunk_states", save_names)

      rl_ok = config_types.RLConfig(
          run_name="ok_rl_tokamax_gdn",
          model_name="qwen3-next-80b-a3b",
          decoder_block=decoder_block,
          use_tokamax_gdn=True,
          gdn_two_pass_bf16=True,
          packing=True,
          gdn_chunk_size=128,
          max_target_length=512,
          max_seq_token_per_tpu=512,
          ici_context_parallelism=4,
          context_parallel_load_balance=False,
          gdn_cp_mode="seq",
          enable_checkpointing=False,
      )
      self.assertTrue(rl_ok.use_tokamax_gdn)
      self.assertTrue(rl_ok.gdn_two_pass_bf16)


class TokamaxGdnAdapterGuardTest(unittest.TestCase):
  """Always-running unit tests for adapter guards, backend dispatch, and `resolve_gdn_cp_mode`."""

  def test_constants_match_remat_expansions(self):
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_CHUNK_SIZE, 64)
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_UNPACKED_CHUNK_SIZES, (16, 32, 64, 128, 256, 512))
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_PACKED_OR_SEQ_CP_CHUNK_SIZES, (16, 32, 64, 128))
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME, "gdn_chunk_states")
    self.assertEqual(tokamax_gdn.TOKAMAX_GDN_CP_RESIDUAL_CHECKPOINT_NAME, "gdn_cp_s_in")
    self.assertIn(
        tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME,
        maxtext_utils._GDN_REMAT_EXPANSIONS["gdn_states"],  # pylint: disable=protected-access
    )
    self.assertIn(
        tokamax_gdn.TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME,
        maxtext_utils._GDN_REMAT_EXPANSIONS["gdn"],  # pylint: disable=protected-access
    )
    self.assertIn(
        tokamax_gdn.TOKAMAX_GDN_CP_RESIDUAL_CHECKPOINT_NAME,
        maxtext_utils._GDN_REMAT_EXPANSIONS["gdn_cp_state"],  # pylint: disable=protected-access
    )

  def test_resolve_gdn_cp_mode_unified_semantics(self):
    cfg_auto = types.SimpleNamespace(gdn_cp_mode="auto", context_parallel_load_balance=False)
    # Without prefer_head_cp (`use_tokamax_gdn=False`), 'auto' always selects seq-CP when cp_size > 1.
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 1, 16, prefer_head_cp=False), (False, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 2, 16, prefer_head_cp=False), (False, True))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 4, 16, prefer_head_cp=False), (False, True))

    # With prefer_head_cp=True (`use_tokamax_gdn=True`), 'auto' picks head-CP at cp_size == 2 when num_k_heads is even,
    # and seq-CP for cp_size > 2 or odd num_k_heads.
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 1, 16, prefer_head_cp=True), (False, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 2, 16, prefer_head_cp=True), (True, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 2, 3, prefer_head_cp=True), (False, True))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 4, 16, prefer_head_cp=True), (False, True))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_auto, 8, 16, prefer_head_cp=True), (False, True))
    self.assertEqual(
        qwen3.resolve_gdn_cp_mode(cfg_auto, 4, 16, model_mode=MODEL_MODE_AUTOREGRESSIVE, prefer_head_cp=True),
        (False, False),
    )

    cfg_head = types.SimpleNamespace(gdn_cp_mode="head", context_parallel_load_balance=False)
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_head, 1, 16, prefer_head_cp=True), (False, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_head, 4, 16, prefer_head_cp=True), (True, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_head, 16, 16, prefer_head_cp=True), (True, False))
    with self.assertRaisesRegex(ValueError, "divisible by cp_size"):
      qwen3.resolve_gdn_cp_mode(cfg_head, 6, 16, prefer_head_cp=True)

    cfg_seq = types.SimpleNamespace(gdn_cp_mode="seq", context_parallel_load_balance=False)
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_seq, 1, 16, prefer_head_cp=True), (False, False))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_seq, 2, 16, prefer_head_cp=True), (False, True))
    self.assertEqual(qwen3.resolve_gdn_cp_mode(cfg_seq, 4, 6, prefer_head_cp=True), (False, True))

  def test_missing_tokamax_symbols_raise_clear_import_errors(self):
    q = jnp.zeros((1, 64, 2, 128), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 64, 4, 128), dtype=jnp.bfloat16)
    g = jnp.zeros((1, 64, 4), dtype=jnp.float32)
    b = jnp.zeros((1, 64, 4), dtype=jnp.bfloat16)
    fake_ops = types.ModuleType("tokamax.ops")
    fake_tpu = types.ModuleType("tokamax.ops.tpu")
    fake_pkg = types.ModuleType("tokamax")
    fake_pkg.ops = fake_ops  # type: ignore[attr-defined]
    fake_ops.tpu = fake_tpu  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {
            "tokamax": fake_pkg,
            "tokamax.ops": fake_ops,
            "tokamax.ops.tpu": fake_tpu,
            "tokamax._src": None,
            "tokamax._src.ops": None,
            "tokamax._src.ops.kda": None,
            "tokamax._src.ops.kda.cp_utils": None,
        },
    ):
      with self.assertRaisesRegex(ImportError, "tokamax.ops.gated_delta_net"):
        tokamax_gdn._tokamax_gdn_op()  # pylint: disable=protected-access
      with self.assertRaisesRegex(ImportError, "tokamax.ops.gated_delta_net"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b)
      with self.assertRaisesRegex(ImportError, "GatedDeltaNetConfig"):
        tokamax_gdn._tokamax_tpu_gdn_op(chunk_size=64, two_pass_bf16=True)  # pylint: disable=protected-access
      with self.assertRaisesRegex(ImportError, "ContextParallelMetadata"):
        tokamax_gdn._tokamax_cp_metadata_cls()  # pylint: disable=protected-access

  def test_runtime_input_and_backend_guards(self):
    q = jnp.zeros((1, 64, 2, 128), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 64, 4, 128), dtype=jnp.bfloat16)
    g = jnp.zeros((1, 64, 4), dtype=jnp.float32)
    b = jnp.zeros((1, 64, 4), dtype=jnp.bfloat16)

    for bad_unpacked_chunk in (0, 24, 96, 1024):
      with self.assertRaisesRegex(NotImplementedError, "requires chunk_size in"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, chunk_size=bad_unpacked_chunk)
    q512 = jnp.zeros((1, 512, 2, 128), dtype=jnp.bfloat16)
    v512 = jnp.zeros((1, 512, 4, 128), dtype=jnp.bfloat16)
    g512 = jnp.zeros((1, 512, 4), dtype=jnp.float32)
    b512 = jnp.zeros((1, 512, 4), dtype=jnp.bfloat16)
    for bad_packed_chunk in (256, 512):
      with self.assertRaisesRegex(NotImplementedError, "packed or sequence-CP requires chunk_size in"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q512,
            q512,
            v512,
            g512,
            b512,
            chunk_size=bad_packed_chunk,
            segment_ids=jnp.ones((1, 512), dtype=jnp.int32),
        )
    for bad_len in (24, 48):
      with self.assertRaisesRegex(ValueError, "divisible by chunk_size"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q[:, :bad_len],
            q[:, :bad_len],
            v[:, :bad_len],
            g[:, :bad_len],
            b[:, :bad_len],
        )
    with self.assertRaisesRegex(ValueError, "segment_ids shape"):
      tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, segment_ids=jnp.ones((2, 64), dtype=jnp.int32))
    with self.assertRaisesRegex(ValueError, "requires `mesh` when `cp_axis` is provided"):
      tokamax_gdn.tokamax_chunk_gated_delta_rule(q, q, v, g, b, cp_axis="context")

    devs = jax.devices()
    if len(devs) >= 4:
      mesh_2x2 = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("context", "context_usp_ulysses"))
      with self.assertRaisesRegex(ValueError, "at most one active sequence-CP mesh axis"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q, q, v, g, b, mesh=mesh_2x2, cp_axis=("context", "context_usp_ulysses")
        )
      mesh_cp4 = Mesh(np.asarray(devs[:4]), ("context",))
      with self.assertRaisesRegex(NotImplementedError, "does not support `initial_state`"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q,
            q,
            v,
            g,
            b,
            mesh=mesh_cp4,
            cp_axis="context",
            initial_state=jnp.zeros((1, 4, 128, 128), dtype=jnp.bfloat16),
        )
      with self.assertRaisesRegex(NotImplementedError, "does not support `output_final_state=True`"):
        tokamax_gdn.tokamax_chunk_gated_delta_rule(
            q, q, v, g, b, mesh=mesh_cp4, cp_axis="context", output_final_state=True
        )

    # Strict TPU -> "mosaic" dispatch, `segment_ids` int32 forwarding, `ContextParallelMetadata`, and `two_pass_bf16`.
    captured_kwargs: dict[str, Any] = {}

    def fake_gdn(**kwargs):
      captured_kwargs.clear()
      captured_kwargs.update(kwargs)
      return kwargs["value"], (kwargs["initial_state"] if kwargs["output_final_state"] else None)

    with (
        mock.patch.object(tokamax_gdn, "_tokamax_gdn_op", return_value=fake_gdn),
        mock.patch.object(jax, "default_backend", return_value="tpu"),
    ):
      _ = jax.jit(lambda qi, vi, gi, bi: tokamax_gdn.tokamax_chunk_gated_delta_rule(qi, qi, vi, gi, bi, chunk_size=64))(
          q, v, g, b
      )
      self.assertEqual(captured_kwargs["implementation"], "mosaic")
      self.assertIsNone(captured_kwargs["segment_ids"])
      self.assertIsNone(captured_kwargs["context_parallel_metadata"])
      self.assertFalse(captured_kwargs["use_gate_in_kernel"])
      self.assertTrue(captured_kwargs["use_beta_sigmoid_in_kernel"])
      self.assertEqual(captured_kwargs["residual_checkpoint_name"], "gdn_chunk_states")

      seg_i64 = jnp.ones((1, 128), dtype=jnp.int32)
      q128 = jnp.zeros((1, 128, 2, 128), dtype=jnp.bfloat16)
      v128 = jnp.zeros((1, 128, 4, 128), dtype=jnp.bfloat16)
      g128 = jnp.zeros((1, 128, 4), dtype=jnp.float32)
      b128 = jnp.zeros((1, 128, 4), dtype=jnp.bfloat16)
      _ = jax.jit(
          lambda qi, vi, gi, bi, si: tokamax_gdn.tokamax_chunk_gated_delta_rule(
              qi, qi, vi, gi, bi, chunk_size=128, segment_ids=si
          )
      )(q128, v128, g128, b128, seg_i64)
      self.assertEqual(captured_kwargs["chunk_size"], 128)
      self.assertEqual(captured_kwargs["segment_ids"].dtype, jnp.int32)

      # Mosaic sequence-CP rejects head dims not divisible by 128 and builds `ContextParallelMetadata` when K=V=128.
      if len(devs) >= 2:
        mesh_cp2 = Mesh(np.asarray(devs[:2]), ("context",))
        q_bad_k = jnp.zeros((1, 64, 2, 64), dtype=jnp.bfloat16)
        v_bad_k = jnp.zeros((1, 64, 4, 128), dtype=jnp.bfloat16)
        with self.assertRaisesRegex(NotImplementedError, "multiples of 128"):
          jax.jit(
              lambda qi, vi, gi, bi: tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  qi, qi, vi, gi, bi, mesh=mesh_cp2, cp_axis="context"
              )
          )(q_bad_k, v_bad_k, g, b)

        @dataclasses.dataclass(frozen=True)
        class _FakeCpMeta:
          mesh: Any
          axis_name: str
          cp_residual_checkpoint_name: str | None = None

        with mock.patch.object(tokamax_gdn, "_tokamax_cp_metadata_cls", return_value=_FakeCpMeta):
          _ = jax.jit(
              lambda qi, vi, gi, bi: tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  qi, qi, vi, gi, bi, chunk_size=128, mesh=mesh_cp2, cp_axis=("context",)
              )
          )(q128, v128, g128, b128)
          cp_meta = captured_kwargs["context_parallel_metadata"]
          self.assertIsInstance(cp_meta, _FakeCpMeta)
          self.assertIs(cp_meta.mesh, mesh_cp2)
          self.assertEqual(cp_meta.axis_name, "context")
          self.assertEqual(cp_meta.cp_residual_checkpoint_name, "gdn_cp_s_in")

      # `two_pass_bf16=True` on Mosaic invokes `tokamax.ops.tpu.GatedDeltaNet(config=GatedDeltaNetConfig(...))`.
      captured_tpu_cfg: dict[str, Any] = {}
      captured_tpu_call: dict[str, Any] = {}

      class _FakeGdnConfig:

        def __init__(self, *, chunk_size: int, two_pass_bf16: bool):
          captured_tpu_cfg["chunk_size"] = chunk_size
          captured_tpu_cfg["two_pass_bf16"] = two_pass_bf16

      class _FakeTpuGdnOp:

        def __init__(self, *, config: Any):
          self.config = config

        def __call__(self, **kwargs):
          captured_tpu_call.clear()
          captured_tpu_call.update(kwargs)
          return kwargs["value"], None

      fake_tpu_mod = types.ModuleType("tokamax.ops.tpu")
      fake_tpu_mod.GatedDeltaNetConfig = _FakeGdnConfig  # type: ignore[attr-defined]
      fake_tpu_mod.GatedDeltaNet = _FakeTpuGdnOp  # type: ignore[attr-defined]
      fake_ops_mod = types.ModuleType("tokamax.ops")
      fake_ops_mod.gated_delta_net = fake_gdn  # type: ignore[attr-defined]
      fake_ops_mod.tpu = fake_tpu_mod  # type: ignore[attr-defined]
      fake_root = types.ModuleType("tokamax")
      fake_root.ops = fake_ops_mod  # type: ignore[attr-defined]
      with mock.patch.dict(
          sys.modules,
          {"tokamax": fake_root, "tokamax.ops": fake_ops_mod, "tokamax.ops.tpu": fake_tpu_mod},
      ):
        _ = jax.jit(
            lambda qi, vi, gi, bi: tokamax_gdn.tokamax_chunk_gated_delta_rule(
                qi, qi, vi, gi, bi, chunk_size=128, two_pass_bf16=True
            )
        )(q128, v128, g128, b128)
        self.assertEqual(captured_tpu_cfg, {"chunk_size": 128, "two_pass_bf16": True})
        self.assertEqual(captured_tpu_call["chunk_size"], 128)
        self.assertNotIn("implementation", captured_tpu_call)

    # Offline CPU->TPU cross-compilation fails fast instead of silently lowering `"xla"` or interpret-mode loops.
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

  def _make_inputs(self, b=2, t=256, h_k=2, h_v=4, d_k=32, d_v=32, dtype=jnp.float32, seed=0):
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
    q, k, v, g, b_raw, h0 = self._make_inputs(t=256)
    w_out = jax.random.normal(jax.random.PRNGKey(99), v.shape, dtype=jnp.float32)
    w_h = jax.random.normal(jax.random.PRNGKey(100), h0.shape, dtype=jnp.float32)

    for chunk_size in (64, 128):
      for use_qk_norm in (True, False):
        for with_state in (False, True):
          with self.subTest(chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm, with_initial_state=with_state):
            if not with_state:

              def ref_fn(q_i, k_i, v_i, g_i, b_i, cs=chunk_size, norm=use_qk_norm):
                out, st = qwen3.jax_chunk_gated_delta_rule(
                    q_i,
                    k_i,
                    v_i,
                    g_i,
                    jax.nn.sigmoid(b_i.astype(jnp.float32)),
                    chunk_size=cs,
                    initial_state=None,
                    use_qk_norm_in_gdn=norm,
                    compute_dtype=jnp.float32,
                )
                return jnp.sum(out * w_out), (out, st)

              def tk_fn(q_i, k_i, v_i, g_i, b_i, cs=chunk_size, norm=use_qk_norm):
                out, st = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                    q_i,
                    k_i,
                    v_i,
                    g_i,
                    b_i,
                    chunk_size=cs,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_norm_in_gdn=norm,
                    compute_dtype=jnp.float32,
                    two_pass_bf16=(cs == 128),
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

              def ref_state_fn(q_i, k_i, v_i, g_i, b_i, h_i, cs=chunk_size, norm=use_qk_norm):
                out, st = qwen3.jax_chunk_gated_delta_rule(
                    q_i,
                    k_i,
                    v_i,
                    g_i,
                    jax.nn.sigmoid(b_i.astype(jnp.float32)),
                    chunk_size=cs,
                    initial_state=h_i,
                    use_qk_norm_in_gdn=norm,
                    compute_dtype=jnp.float32,
                )
                assert st is not None
                return jnp.sum(out * w_out) + jnp.sum(st * w_h), (out, st)

              def tk_state_fn(q_i, k_i, v_i, g_i, b_i, h_i, cs=chunk_size, norm=use_qk_norm):
                out, st = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                    q_i,
                    k_i,
                    v_i,
                    g_i,
                    b_i,
                    chunk_size=cs,
                    initial_state=h_i,
                    output_final_state=True,
                    use_qk_norm_in_gdn=norm,
                    compute_dtype=jnp.float32,
                    two_pass_bf16=(cs == 128),
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

  def test_packed_multidoc_and_padding_parity(self):
    q, k, v, g, b_raw, h0 = self._make_inputs(t=256, seed=7)
    w_out = jax.random.normal(jax.random.PRNGKey(101), v.shape, dtype=jnp.float32)
    w_h = jax.random.normal(jax.random.PRNGKey(102), h0.shape, dtype=jnp.float32)
    # Row 0: multi-doc + interior 0-pad continuing doc 5 (`[5, 0, 5]`) + reused non-adjacent raw doc ID `[1, 2, 1]`.
    # Row 1: leading 0-pad, multi-doc boundaries inside and across chunks, trailing 0-pad.
    seg = jnp.asarray(
        [
            [1] * 50 + [2] * 40 + [1] * 46 + [5] * 30 + [0] * 18 + [5] * 52 + [0] * 20,
            [0] * 16 + [3] * 80 + [7] * 64 + [9] * 72 + [0] * 24,
        ],
        dtype=jnp.int32,
    )

    for chunk_size in (64, 128):
      for with_state in (False, True):
        with self.subTest(chunk_size=chunk_size, with_initial_state=with_state):
          if not with_state:

            def ref_packed(q_i, k_i, v_i, g_i, b_i, cs=chunk_size):
              out, _ = qwen3.jax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  jax.nn.sigmoid(b_i.astype(jnp.float32)),
                  chunk_size=cs,
                  initial_state=None,
                  use_qk_norm_in_gdn=True,
                  compute_dtype=jnp.float32,
                  segment_ids=seg,
                  has_initial_state=False,
              )
              return jnp.sum(out * w_out), out

            def tk_packed(q_i, k_i, v_i, g_i, b_i, cs=chunk_size):
              out, _ = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  b_i,
                  chunk_size=cs,
                  initial_state=None,
                  output_final_state=False,
                  use_qk_norm_in_gdn=True,
                  compute_dtype=jnp.float32,
                  segment_ids=seg,
              )
              return jnp.sum(out * w_out), out

            (_, ref_o), ref_g = jax.value_and_grad(ref_packed, argnums=tuple(range(5)), has_aux=True)(q, k, v, g, b_raw)
            (_, tk_o), tk_g = jax.value_and_grad(tk_packed, argnums=tuple(range(5)), has_aux=True)(q, k, v, g, b_raw)
            self.assertLess(_rel_err(tk_o, ref_o), 1e-4)
            for name, g_tk, g_ref in zip(("q", "k", "v", "g", "b"), tk_g, ref_g):
              self.assertLess(_rel_err(g_tk, g_ref), 2e-4, name)
          else:

            def ref_packed_state(q_i, k_i, v_i, g_i, b_i, h_i, cs=chunk_size):
              out, st = qwen3.jax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  jax.nn.sigmoid(b_i.astype(jnp.float32)),
                  chunk_size=cs,
                  initial_state=h_i,
                  use_qk_norm_in_gdn=True,
                  compute_dtype=jnp.float32,
                  segment_ids=seg,
                  has_initial_state=True,
              )
              assert st is not None
              return jnp.sum(out * w_out) + jnp.sum(st * w_h), (out, st)

            def tk_packed_state(q_i, k_i, v_i, g_i, b_i, h_i, cs=chunk_size):
              out, st = tokamax_gdn.tokamax_chunk_gated_delta_rule(
                  q_i,
                  k_i,
                  v_i,
                  g_i,
                  b_i,
                  chunk_size=cs,
                  initial_state=h_i,
                  output_final_state=True,
                  use_qk_norm_in_gdn=True,
                  compute_dtype=jnp.float32,
                  segment_ids=seg,
              )
              assert st is not None
              return jnp.sum(out * w_out) + jnp.sum(st * w_h), (out, st)

            (_, (ref_o, ref_st)), ref_g = jax.value_and_grad(ref_packed_state, argnums=tuple(range(6)), has_aux=True)(
                q, k, v, g, b_raw, h0
            )
            (_, (tk_o, tk_st)), tk_g = jax.value_and_grad(tk_packed_state, argnums=tuple(range(6)), has_aux=True)(
                q, k, v, g, b_raw, h0
            )
            self.assertLess(_rel_err(tk_o, ref_o), 1e-4)
            self.assertLess(_rel_err(tk_st, ref_st), 1e-4)
            for name, g_tk, g_ref in zip(("q", "k", "v", "g", "b", "h0"), tk_g, ref_g):
              self.assertLess(_rel_err(g_tk, g_ref), 2e-4, name)


@unittest.skipUnless(_HAS_TOKAMAX_GDN, "tokamax.ops.gated_delta_net is not available")
class TokamaxGdnLayerParityTest(unittest.TestCase):
  """End-to-end `Qwen3NextGatedDeltaNet` parity across unsharded, packed, CP meshes, padding, and prefill."""

  def _make_pair(self, **overrides):
    """Builds matched Pure-JAX and Tokamax `Qwen3NextGatedDeltaNet` layers sharing RNG seed."""
    jax_overrides = {k: v for k, v in overrides.items() if k != "gdn_two_pass_bf16"}
    cfg_jax = _gdn_pyconfig(use_tokamax_gdn=False, **jax_overrides)
    cfg_tk = _gdn_pyconfig(use_tokamax_gdn=True, **overrides)
    layer_jax = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax, mesh=None, dtype=cfg_jax.dtype, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=3)
    )
    layer_tk = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk, mesh=None, dtype=cfg_tk.dtype, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=3)
    )
    return layer_jax, layer_tk

  def test_unsharded_forward_and_all_param_grads_match(self):
    """Checks unsharded forward and full parameter-gradient parity at chunk sizes 64 and 128."""
    x = jax.random.normal(jax.random.PRNGKey(11), (2, 256, 32), dtype=jnp.float32) * 0.2
    dy = jax.random.normal(jax.random.PRNGKey(12), (2, 256, 32), dtype=jnp.float32)

    def eval_layer(layer):
      gdef, st = nnx.split(layer)

      def loss(s, inp):
        out, _ = nnx.merge(gdef, s)(inp, model_mode=MODEL_MODE_TRAIN)
        return jnp.sum(out * dy), out

      (_, out), (g_st, g_x) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(st, x)
      return out, g_st, g_x

    for chunk_size in (64, 128):
      with self.subTest(gdn_chunk_size=chunk_size):
        layer_jax, layer_tk = self._make_pair(
            gdn_chunk_size=chunk_size,
            gdn_two_pass_bf16=(chunk_size == 128),
            max_target_length=256,
        )
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

  def test_packed_layer_multidoc_and_pad_parity(self):
    x = jax.random.normal(jax.random.PRNGKey(15), (2, 256, 32), dtype=jnp.float32) * 0.2
    pad_noise = jax.random.normal(jax.random.PRNGKey(16), (2, 256, 32), dtype=jnp.float32) * 6.0
    seg = jnp.asarray(
        [
            [0] * 12 + [1] * 70 + [2] * 60 + [0] * 14 + [2] * 40 + [1] * 44 + [0] * 16,
            [4] * 90 + [8] * 80 + [3] * 66 + [0] * 20,
        ],
        dtype=jnp.int32,
    )
    valid = seg != 0
    x_corrupted_pads = jnp.where(valid[..., None], x, pad_noise)
    dy = jnp.where(
        valid[..., None],
        jax.random.normal(jax.random.PRNGKey(19), (2, 256, 32), dtype=jnp.float32),
        0.0,
    )

    for chunk_size in (64, 128):
      with self.subTest(gdn_chunk_size=chunk_size):
        layer_jax, layer_tk = self._make_pair(
            packing=True,
            gdn_chunk_size=chunk_size,
            gdn_two_pass_bf16=True,
            max_target_length=256,
        )
        out_tk_clean, _ = layer_tk(x, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
        out_tk_noisy, _ = layer_tk(x_corrupted_pads, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
        np.testing.assert_allclose(
            np.asarray(jnp.where(valid[..., None], out_tk_clean, 0.0)),
            np.asarray(jnp.where(valid[..., None], out_tk_noisy, 0.0)),
            rtol=1e-6,
            atol=1e-6,
        )

        def eval_packed(layer):
          gdef, st = nnx.split(layer)

          def loss(s, inp):
            out, _ = nnx.merge(gdef, s)(inp, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
            return jnp.sum(out * dy), jnp.where(valid[..., None], out, 0.0)

          (_, out_m), (g_st, g_x) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(st, x_corrupted_pads)
          return out_m, g_st, g_x

        out_jax, grads_jax, dx_jax = eval_packed(layer_jax)
        out_tk, grads_tk, dx_tk = eval_packed(layer_tk)
        self.assertLess(_rel_err(out_tk, out_jax), 1e-4)
        self.assertLess(_rel_err(dx_tk, dx_jax), 2e-4)
        for (path, leaf_jax), (_, leaf_tk) in zip(
            jax.tree_util.tree_leaves_with_path(grads_jax),
            jax.tree_util.tree_leaves_with_path(grads_tk),
        ):
          self.assertLess(_rel_err(leaf_tk, leaf_jax), 2e-4, jax.tree_util.keystr(path))

  def test_cp_mesh_head_and_seq_unpacked_and_packed_parity(self):
    devs = jax.devices()
    if len(devs) >= 4:
      _run_cp_mesh_parity_checks(devs)
    else:
      env = {k: v for k, v in os.environ.items() if k != "XML_OUTPUT_FILE" and not k.startswith("TEST_SHARD_")}
      prior_xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", env.get("XLA_FLAGS", "")).strip()
      env["XLA_FLAGS"] = f"{prior_xla_flags} --xla_force_host_platform_device_count=4".strip()
      env["JAX_PLATFORMS"] = "cpu"
      env["_TOKAMAX_GDN_RUN_CP_MESH_CHECKS"] = "1"
      repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
      env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], repo_root]) if env.get("PYTHONPATH") else repo_root
      cmd = [sys.executable, __file__] if sys.executable else [sys.argv[0]]
      res = subprocess.run(
          cmd,
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

  def test_prefill_active_cache_aligned_packed_and_fallback_paths(self):
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

    # 2. Chunk-aligned segmented/packed prefill (T=64, cp=1) ALSO dispatches to Tokamax and updates active_cache.
    seg64 = jnp.asarray(
        [
            [0] * 8 + [1] * 24 + [2] * 24 + [0] * 8,
            [1] * 56 + [0] * 8,
        ],
        dtype=jnp.int32,
    )
    cfg_jax_pack = _gdn_pyconfig(use_tokamax_gdn=False, packing=True)
    cfg_tk_pack = _gdn_pyconfig(use_tokamax_gdn=True, packing=True)
    layer_jax_seg = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_jax_pack,
        inputs_shape=x64.shape,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=9),
    )
    layer_tk_seg = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_tk_pack,
        inputs_shape=x64.shape,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=9),
    )
    with mock.patch.object(
        tokamax_gdn,
        "tokamax_chunk_gated_delta_rule",
        wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
    ) as spy_seg:
      out_tk_seg, cache_tk_seg = layer_tk_seg(x64, decoder_segment_ids=seg64, model_mode=MODEL_MODE_PREFILL)
      self.assertEqual(spy_seg.call_count, 1)
    out_jax_seg, cache_jax_seg = layer_jax_seg(x64, decoder_segment_ids=seg64, model_mode=MODEL_MODE_PREFILL)
    valid_seg = (seg64 != 0)[..., None]
    self.assertLess(_rel_err(jnp.where(valid_seg, out_tk_seg, 0.0), jnp.where(valid_seg, out_jax_seg, 0.0)), 1e-4)
    self.assertLess(_rel_err(cache_tk_seg.get_gdn_states()[0], cache_jax_seg.get_gdn_states()[0]), 1e-4)
    self.assertLess(_rel_err(cache_tk_seg.get_gdn_states()[1], cache_jax_seg.get_gdn_states()[1]), 1e-5)

    # 3. Unaligned prefill bucket (T=32) and stateful seq-CP prefill cleanly fall back to Pure JAX without error.
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

    devs = jax.devices()
    if len(devs) >= 4:
      x256 = jax.random.normal(jax.random.PRNGKey(33), (2, 256, 32), dtype=jnp.float32) * 0.2
      cfg_seq_prefill = _gdn_pyconfig(
          use_tokamax_gdn=True,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          ici_context_parallelism=4,
          gdn_cp_mode="seq",
          max_target_length=256,
          max_prefill_predict_length=256,
      )
      mesh_cp4 = Mesh(maxtext_utils.create_device_mesh(cfg_seq_prefill, devices=devs[:4]), cfg_seq_prefill.mesh_axes)
      with jax.set_mesh(mesh_cp4), nn_partitioning.axis_rules(cfg_seq_prefill.logical_axis_rules):
        layer_stateful_seq = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_seq_prefill,
            inputs_shape=x256.shape,
            mesh=mesh_cp4,
            dtype=jnp.float32,
            model_mode=MODEL_MODE_PREFILL,
            rngs=nnx.Rngs(params=9),
        )
        with mock.patch.object(
            tokamax_gdn,
            "tokamax_chunk_gated_delta_rule",
            wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
        ) as spy_seq_stateful:
          out_stateful_seq, _ = layer_stateful_seq(x256, model_mode=MODEL_MODE_PREFILL)
          self.assertEqual(spy_seq_stateful.call_count, 0)
          self.assertEqual(out_stateful_seq.shape, x256.shape)

        # 4. Multi-token `MODEL_MODE_AUTOREGRESSIVE` (`seq_len=64 > 1`, e.g. MTP / speculative decode) under CP
        # bypasses Tokamax cleanly (`use_tokamax=False`) and matches Pure JAX.
        cfg_seq_ar_jax = _gdn_pyconfig(
            use_tokamax_gdn=False,
            gdn_num_key_heads=2,
            gdn_num_value_heads=4,
            gdn_key_head_dim=128,
            gdn_value_head_dim=128,
            ici_context_parallelism=4,
            gdn_cp_mode="seq",
            max_target_length=256,
        )
        layer_ar_tk = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_seq_prefill,
            mesh=mesh_cp4,
            dtype=jnp.float32,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=9),
        )
        layer_ar_jax = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_seq_ar_jax,
            mesh=mesh_cp4,
            dtype=jnp.float32,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=9),
        )
        with mock.patch.object(
            tokamax_gdn,
            "tokamax_chunk_gated_delta_rule",
            wraps=tokamax_gdn.tokamax_chunk_gated_delta_rule,
        ) as spy_ar_mtp:
          out_ar_tk, _ = layer_ar_tk(x64, model_mode=MODEL_MODE_AUTOREGRESSIVE)
          self.assertEqual(spy_ar_mtp.call_count, 0)
        out_ar_jax, _ = layer_ar_jax(x64, model_mode=MODEL_MODE_AUTOREGRESSIVE)
        self.assertLess(_rel_err(out_ar_tk, out_ar_jax), 1e-5)


@unittest.skipUnless(_HAS_TOKAMAX_GDN, "tokamax.ops.gated_delta_net is not available")
class TokamaxGdnRematTest(unittest.TestCase):
  """Verifies `jax.checkpoint` remat behavior (`gdn_states` & `gdn_cp_state`) and offline cross-compile guard."""

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

    # 3. Sequence-CP (`cp=4`) custom remat policy with `gdn_cp_state="device"` saves CP boundary activations and matches
    # un-rematted gradients.
    devs = jax.devices()
    if len(devs) >= 4:
      mesh_cp4 = Mesh(np.asarray(devs[:4]), ("context",))
      cfg_cp_remat = _gdn_pyconfig(
          use_tokamax_gdn=True,
          packing=True,
          base_emb_dim=32,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          gdn_key_head_dim=128,
          gdn_value_head_dim=128,
          gdn_chunk_size=64,
          max_target_length=256,
          ici_context_parallelism=4,
          gdn_cp_mode="seq",
          remat_policy="custom",
          gdn="remat",
          gdn_states="remat",
          gdn_cp_state="device",
      )
      save_cp_names, offload_cp_names = maxtext_utils.get_save_and_offload_names(cfg_cp_remat)
      self.assertIn("gdn_cp_s_in", save_cp_names)
      self.assertIn("gdn_cp_conv_halo", save_cp_names)
      self.assertEqual(offload_cp_names, [])
      cp_policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=save_cp_names,
          names_which_can_be_offloaded=offload_cp_names,
          offload_src="device",
          offload_dst="pinned_host",
      )
      x_cp = jax.random.normal(jax.random.PRNGKey(45), (1, 256, 32), dtype=jnp.float32) * 0.2
      seg_cp = jnp.asarray([[1] * 100 + [2] * 120 + [0] * 36], dtype=jnp.int32)
      with jax.set_mesh(mesh_cp4), nn_partitioning.axis_rules(cfg_cp_remat.logical_axis_rules):
        layer_cp_remat = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_cp_remat,
            mesh=mesh_cp4,
            dtype=jnp.float32,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=23),
        )
        gdef_cpr, st_cpr = nnx.split(layer_cp_remat)
        st_cpr = jax.device_put(st_cpr, nnx_construct_named_sharding(st_cpr, mesh_cp4))
        x_cp_sh = jax.device_put(x_cp, jax.sharding.NamedSharding(mesh_cp4, P(None, "context", None)))
        seg_cp_sh = jax.device_put(seg_cp, jax.sharding.NamedSharding(mesh_cp4, P(None, "context")))

        def cp_step(st, inp):
          out, _ = nnx.merge(gdef_cpr, st)(inp, decoder_segment_ids=seg_cp_sh, model_mode=MODEL_MODE_TRAIN)
          return inp + out * jnp.tanh(out)

        buf_cp_full = io.StringIO()
        with contextlib.redirect_stdout(buf_cp_full):
          ad_checkpoint.print_saved_residuals(
              lambda st, inp: jnp.sum(jax.checkpoint(cp_step)(st, inp)),
              st_cpr,
              x_cp_sh,
          )
        saved_cp_full = [line.strip() for line in buf_cp_full.getvalue().splitlines() if line.strip()]
        leaked_cp = [line for line in saved_cp_full if "from the argument" not in line and "from a constant" not in line]
        self.assertEqual(leaked_cp, [])

        buf_cp = io.StringIO()
        with contextlib.redirect_stdout(buf_cp):
          ad_checkpoint.print_saved_residuals(
              lambda st, inp: jnp.sum(jax.checkpoint(cp_step, policy=cp_policy)(st, inp)),
              st_cpr,
              x_cp_sh,
          )
        saved_cp = [line.strip() for line in buf_cp.getvalue().splitlines() if line.strip()]
        saved_cp_extra = [line for line in saved_cp if "from the argument" not in line and "from a constant" not in line]
        self.assertTrue(
            any("gdn_cp_conv_halo" in line or "segmented_causal_depthwise_conv1d" in line for line in saved_cp_extra),
            saved_cp,
        )

        g_eager = jax.jit(jax.grad(lambda st, inp: jnp.sum(cp_step(st, inp)), argnums=(0, 1)))(st_cpr, x_cp_sh)
        g_remat = jax.jit(
            jax.grad(lambda st, inp: jnp.sum(jax.checkpoint(cp_step, policy=cp_policy)(st, inp)), argnums=(0, 1))
        )(st_cpr, x_cp_sh)
        for leaf_r, leaf_e in zip(jax.tree_util.tree_leaves(g_remat), jax.tree_util.tree_leaves(g_eager)):
          self.assertLess(_rel_err(leaf_r, leaf_e), 1e-5)

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


def _run_cp_mesh_child_main(unused_argv):
  setUpModule()
  try:
    _run_cp_mesh_parity_checks(jax.devices())
    print("TOKAMAX_GDN_CP_MESH_OK")
  finally:
    tearDownModule()


if __name__ == "__main__":
  if os.environ.get("_TOKAMAX_GDN_RUN_CP_MESH_CHECKS") == "1":
    app.run(_run_cp_mesh_child_main, argv=[sys.argv[0]])
  else:
    absltest.main()
