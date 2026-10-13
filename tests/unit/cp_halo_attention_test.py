# Copyright 2026 Google LLC
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
"""Tests for per-layer CP strategy resolution and halo sliding-window CP."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
import numpy as np
import pydantic
import pytest

from maxtext.common import common_types
from maxtext.common import gcloud_stub
from maxtext.configs import pyconfig
from maxtext.layers import attention_op
from maxtext.layers import attentions
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from tests.utils import attention_test_util
from tests.utils import hlo_test_utils
from tests.utils import test_helpers

_HALO_CP_TEST_CASES = (
    {
        "testcase_name": "cp2_no_load_balance",
        "ici_context_parallelism": 2,
        "context_parallel_load_balance": False,
        "packing": False,
    },
    {
        "testcase_name": "cp2_load_balance",
        "ici_context_parallelism": 2,
        "context_parallel_load_balance": True,
        "packing": False,
    },
    {
        "testcase_name": "cp2_packed",
        "ici_context_parallelism": 2,
        "context_parallel_load_balance": False,
        "packing": True,
    },
    {
        "testcase_name": "cp2_packed_load_balance",
        "ici_context_parallelism": 2,
        "context_parallel_load_balance": True,
        "packing": True,
    },
    {
        "testcase_name": "cp4_no_load_balance",
        "ici_context_parallelism": 4,
        "context_parallel_load_balance": False,
        "packing": False,
    },
    {
        "testcase_name": "cp4_load_balance",
        "ici_context_parallelism": 4,
        "context_parallel_load_balance": True,
        "packing": False,
    },
    {
        "testcase_name": "cp4_packed",
        "ici_context_parallelism": 4,
        "context_parallel_load_balance": False,
        "packing": True,
    },
    {
        "testcase_name": "cp4_packed_load_balance",
        "ici_context_parallelism": 4,
        "context_parallel_load_balance": True,
        "packing": True,
    },
)


def _count_primitives(jaxpr: Any, primitive_name: str, name_param: str | None = None) -> int:
  """Counts primitive occurrences in a jaxpr, recursing into sub-jaxprs."""
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive_name and (name_param is None or name_param in str(eqn.params.get("name", ""))):
      count += 1
    for value in eqn.params.values():
      values = value if isinstance(value, (list, tuple)) else (value,)
      for entry in values:
        entry = getattr(entry, "jaxpr", entry)
        if hasattr(entry, "eqns"):
          count += _count_primitives(entry, primitive_name, name_param)
  return count


class CpHaloConfigAndValidationTest(parameterized.TestCase):
  """Config, strategy resolution, and runtime validation tests for halo CP."""

  def test_resolve_context_parallel_strategy_overrides_only_local_sliding(self):
    config = types.SimpleNamespace(
        context_parallel_strategy="Ring",
        local_context_parallel_strategy="Halo",
    )
    self.assertEqual(
        attention_op.resolve_context_parallel_strategy(config, common_types.AttentionType.LOCAL_SLIDING),
        "halo",
    )
    self.assertEqual(
        attention_op.resolve_context_parallel_strategy(config, common_types.AttentionType.GLOBAL),
        "ring",
    )

    fallback_config = types.SimpleNamespace(
        context_parallel_strategy="ring",
        local_context_parallel_strategy="",
    )
    self.assertEqual(
        attention_op.resolve_context_parallel_strategy(fallback_config, common_types.AttentionType.LOCAL_SLIDING),
        "ring",
    )

  def test_pyconfig_accepts_local_context_parallel_strategy_halo(self):
    argv = [
        "",
        test_helpers.get_test_config_path(),
        "run_name=test_cp_halo",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "local_context_parallel_strategy=Halo",
        "sliding_window_size=128",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [mock.MagicMock(slice_index=0) for _ in range(8)]
    with mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "ring")
    self.assertEqual(config.local_context_parallel_strategy, "halo")

  def test_pyconfig_accepts_ulysses_global_with_halo_local_mqa(self):
    argv = [
        "",
        test_helpers.get_test_config_path(),
        "run_name=test_ulysses_halo",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ulysses",
        "local_context_parallel_strategy=halo",
        "context_parallel_load_balance=False",
        "sliding_window_size=128",
        "base_num_query_heads=8",
        "base_num_kv_heads=1",
        "global_num_kv_heads=4",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [mock.MagicMock(slice_index=0) for _ in range(8)]
    with mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "ulysses")
    self.assertEqual(config.local_context_parallel_strategy, "halo")

  def test_pyconfig_rejects_invalid_local_context_parallel_configs(self):
    base_args = [
        "",
        test_helpers.get_test_config_path(),
        "run_name=test_cp_halo_invalid",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "local_context_parallel_strategy=halo",
        "sliding_window_size=128",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    cases = [
        (
            ["local_context_parallel_strategy=invalid"],
            ["local_context_parallel_strategy=halo"],
            "local_context_parallel_strategy",
        ),
        (
            ["sliding_window_size=0"],
            ["sliding_window_size=128"],
            "sliding_window_size > 0",
        ),
        (
            ["use_tokamax_splash=False", "context_parallel_strategy=all_gather"],
            ["use_tokamax_splash=True", "context_parallel_strategy=ring"],
            "use_tokamax_splash=True",
        ),
        (
            [
                "context_parallel_strategy=ulysses",
                "context_parallel_load_balance=False",
                "packing=False",
                "base_num_kv_heads=1",
                "global_num_kv_heads=1",
            ],
            ["context_parallel_strategy=ring"],
            "MQA",
        ),
    ]
    mock_devices = [mock.MagicMock(slice_index=0) for _ in range(8)]
    for bad_args, args_to_remove, expected_regex in cases:
      with self.subTest(bad_args=bad_args):
        argv = [arg for arg in base_args if arg not in args_to_remove]
        argv.extend(bad_args)
        with mock.patch("jax.devices", return_value=mock_devices):
          with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
            pyconfig.initialize(argv)

  def test_kv_head_sharding_skips_ulysses_context_axis_for_local_halo_layer(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], test_helpers.get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test_kv_sharding",
        enable_checkpointing=False,
        max_target_length=128,
        sliding_window_size=128,
    )
    inputs_shape = (1, cfg.max_target_length, 16)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    local_attention = attentions.Attention(
        config=cfg,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=common_types.AttentionType.LOCAL_SLIDING,
        sliding_window_size=128,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )
    global_attention = attentions.Attention(
        config=cfg,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=common_types.AttentionType.GLOBAL,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )
    local_attention.mesh = types.SimpleNamespace(shape={"context": 4})
    global_attention.mesh = types.SimpleNamespace(shape={"context": 4})
    with mock.patch.dict(
        cfg.get_keys(),
        {
            "context_parallel_strategy": "ulysses",
            "local_context_parallel_strategy": "halo",
        },
    ):
      local_attention.init_kv_w(inputs_kv_shape=inputs_shape)
      with self.assertRaisesRegex(ValueError, r"num_kv_heads \(2\).*must be divisible by 4"):
        global_attention.init_kv_w(inputs_kv_shape=inputs_shape)

  def test_tpu_halo_flash_attention_rejects_unsupported_runtime_features(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], test_helpers.get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test_halo_runtime",
        enable_checkpointing=False,
        max_target_length=512,
        sliding_window_size=128,
        use_tokamax_splash=True,
        local_context_parallel_strategy="halo",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    op = attention_op.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=512,
        num_query_heads=4,
        num_kv_heads=2,
        attention_type=common_types.AttentionType.LOCAL_SLIDING,
        sliding_window_size=128,
    )
    self.assertEqual(op.context_parallel_strategy, "halo")
    q = jnp.zeros((1, 512, 4, cfg.head_dim), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 512, 2, cfg.head_dim), dtype=jnp.bfloat16)

    with self.assertRaisesRegex(ValueError, "train mode"):
      op.tpu_halo_flash_attention(q, k, k, None, model_mode=common_types.MODEL_MODE_PREFILL)
    with self.assertRaisesRegex(ValueError, "ragged attention"):
      op.tpu_halo_flash_attention(
          q,
          k,
          k,
          None,
          model_mode=common_types.MODEL_MODE_TRAIN,
          use_ragged_attention=True,
      )
    with self.assertRaisesRegex(ValueError, "chunked prefill"):
      op.tpu_halo_flash_attention(
          q,
          k,
          k,
          None,
          model_mode=common_types.MODEL_MODE_TRAIN,
          previous_chunk=object(),
      )
    with self.assertRaisesRegex(ValueError, "indexer"):
      op.tpu_halo_flash_attention(
          q,
          k,
          k,
          None,
          model_mode=common_types.MODEL_MODE_TRAIN,
          indexer_mask=object(),
      )
    with self.assertRaisesRegex(ValueError, "bidirectional"):
      op.tpu_halo_flash_attention(
          q,
          k,
          k,
          None,
          model_mode=common_types.MODEL_MODE_TRAIN,
          bidirectional_mask=object(),
      )

    zero_window_op = attention_op.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=512,
        num_query_heads=4,
        num_kv_heads=2,
        attention_type=common_types.AttentionType.LOCAL_SLIDING,
        sliding_window_size=0,
    )
    with self.assertRaisesRegex(ValueError, "sliding_window_size > 0"):
      zero_window_op.tpu_halo_flash_attention(q, k, k, None, model_mode=common_types.MODEL_MODE_TRAIN)

    global_op = attention_op.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=512,
        num_query_heads=4,
        num_kv_heads=2,
        attention_type=common_types.AttentionType.GLOBAL,
    )
    with self.assertRaisesRegex(ValueError, "only supported for AttentionType.LOCAL_SLIDING"):
      global_op.tpu_halo_flash_attention(q, k, k, None, model_mode=common_types.MODEL_MODE_TRAIN)


class CpHaloAttentionTest(parameterized.TestCase):
  """End-to-end TPU flash attention tests for halo sliding-window CP."""

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test_cp_halo",
      "enable_checkpointing": False,
      "max_target_length": 1024,
      "sliding_window_size": 128,
      "sa_block_q": 128,
      "sa_block_kv": 128,
      "sa_block_kv_compute": 128,
      "sa_block_q_dkv": 128,
      "sa_block_kv_dkv": 128,
      "sa_block_kv_dkv_compute": 128,
      "sa_block_q_dq": 128,
      "sa_block_kv_dq": 128,
      "local_sa_block_q": 128,
      "local_sa_block_kv": 128,
      "local_sa_block_kv_compute": 128,
      "local_sa_block_q_dkv": 128,
      "local_sa_block_kv_dkv": 128,
      "local_sa_block_kv_dkv_compute": 128,
      "local_sa_block_q_dq": 128,
      "local_sa_block_kv_dq": 128,
  }

  def setUp(self):
    super().setUp()
    if not gcloud_stub.is_decoupled():
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
    self.cfg = pyconfig.initialize(
        [sys.argv[0], test_helpers.get_test_config_path()],
        **self.config_arguments,
    )
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.max_target_length = self.cfg.max_target_length
    self.embed_dim = self.cfg.base_emb_dim

  def get_data(self, dtype):
    """Generates random input activations, segment IDs, and positions."""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )
    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng,
        (self.global_batch_size, self.max_target_length),
        0,
        self.max_target_length,
    )
    return lnx, decoder_segment_ids, decoder_positions

  def get_packed_data(self, dtype):
    """Generates packed sequence inputs with contiguous segment IDs."""
    lnx, _, _ = self.get_data(dtype)
    segment_lengths = (160, 480, 224, 160)
    segment_ids = jnp.concatenate(
        [jnp.full((length,), segment, dtype=jnp.int32) for segment, length in enumerate(segment_lengths, start=1)]
    )
    positions = jnp.concatenate([jnp.arange(length, dtype=jnp.int32) for length in segment_lengths])
    decoder_segment_ids = jnp.broadcast_to(segment_ids, (self.global_batch_size, self.max_target_length))
    decoder_positions = jnp.broadcast_to(positions, (self.global_batch_size, self.max_target_length))
    return lnx, decoder_segment_ids, decoder_positions

  def _halo_test_config(
      self,
      ici_context_parallelism,
      *,
      context_parallel_load_balance=False,
      packing=False,
  ):
    return pyconfig.initialize(
        [sys.argv[0], test_helpers.get_test_config_path()],
        **self.config_arguments,
        attention="flash",
        context_parallel_strategy="ring",
        local_context_parallel_strategy="halo",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=ici_context_parallelism,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=packing,
        dtype="float32",
    )

  def _halo_test_modules(self, cfg_cp, mesh_cp, lnx):
    """Builds the dot-product reference and halo flash attention modules."""
    attention_generic = attentions.Attention(
        config=self.cfg,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
        head_dim=cfg_cp.head_dim,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=cfg_cp.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        attention_type=common_types.AttentionType.LOCAL_SLIDING,
        sliding_window_size=cfg_cp.sliding_window_size,
        rngs=self.nnx_rng,
    )
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      attention_flash_cp = attentions.Attention(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          mesh=mesh_cp,
          attention_kernel="flash",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          attention_type=common_types.AttentionType.LOCAL_SLIDING,
          sliding_window_size=cfg_cp.sliding_window_size,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    return attention_generic, attention_flash_cp

  @parameterized.named_parameters(*_HALO_CP_TEST_CASES)
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_halo_context_parallel(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      packing,
  ):
    """Tests forward parity between dot_product and flash attention + halo CP."""
    cfg_cp = self._halo_test_config(
        ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        packing=packing,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_generic, attention_flash_cp = self._halo_test_modules(cfg_cp, mesh_cp, lnx)
    generic_output, _ = attention_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
    )
    nnx.update(attention_flash_cp, nnx.state(attention_generic))

    flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    generic_output = jax.device_get(generic_output)
    flash_cp_output = jax.device_get(flash_cp_output)

    np.testing.assert_allclose(
        generic_output,
        flash_cp_output,
        rtol=1e-02,
        atol=1e-02,
        equal_nan=False,
        err_msg=(
            "Logits from generic dot product and flash attention + halo context"
            " parallelism are not close."
            f" ici_context_parallelism={ici_context_parallelism},"
            f" context_parallel_load_balance={context_parallel_load_balance},"
            f" packing={packing}."
        ),
    )

  @parameterized.named_parameters(*_HALO_CP_TEST_CASES)
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_halo_context_parallel_grad(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      packing,
  ):
    """Tests input-gradient parity between dot_product and flash halo CP."""
    cfg_cp = self._halo_test_config(
        ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        packing=packing,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_generic, attention_flash_cp = self._halo_test_modules(cfg_cp, mesh_cp, lnx)
    nnx.update(attention_flash_cp, nnx.state(attention_generic))

    def generic_loss(inputs):
      output, _ = attention_generic(
          inputs,
          inputs,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def halo_loss(inputs):
      if context_parallel_load_balance:
        context_parallel_size = cfg_cp.ici_context_parallelism
        inputs = max_utils.reorder_sequence(inputs, cp_size=context_parallel_size)
        halo_segment_ids = max_utils.reorder_sequence(decoder_segment_ids, cp_size=context_parallel_size)
        halo_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size)
      else:
        halo_segment_ids = decoder_segment_ids
        halo_positions = decoder_positions
      output, _ = attention_flash_cp(
          inputs,
          inputs,
          decoder_segment_ids=halo_segment_ids,
          inputs_positions=halo_positions,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with (
        jax.set_mesh(mesh_cp),
        nn_partitioning.axis_rules(cfg_cp.logical_axis_rules),
    ):
      halo_grad = jax.grad(halo_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    halo_grad = jax.device_get(halo_grad)

    np.testing.assert_allclose(
        generic_grad,
        halo_grad,
        rtol=1e-02,
        atol=1e-07,
        equal_nan=False,
        err_msg=(
            "Input gradients from generic dot product and flash attention +"
            " halo context parallelism are not close."
            f" ici_context_parallelism={ici_context_parallelism},"
            f" context_parallel_load_balance={context_parallel_load_balance},"
            f" packing={packing}."
        ),
    )

  @pytest.mark.tpu_only
  def test_tpu_flash_attention_halo_hlo_uses_permute_without_sequence_all_gather(
      self,
  ):
    """Checks halo CP HLO uses collective-permute and no sequence all-gather."""
    cfg_cp = self._halo_test_config(4, context_parallel_load_balance=True, packing=True)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    _, attention_flash_cp = self._halo_test_modules(cfg_cp, mesh_cp, lnx)

    def attention_forward(x, pos, seg):
      output, _ = attention_flash_cp(
          x,
          x,
          decoder_segment_ids=seg,
          inputs_positions=pos,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      return output

    def attention_loss(x, pos, seg):
      return jnp.sum(attention_forward(x, pos, seg).astype(jnp.float32))

    hlo_texts = []
    for lowered_fn in (attention_forward, jax.grad(attention_loss)):
      with (
          jax.set_mesh(mesh_cp),
          nn_partitioning.axis_rules(cfg_cp.logical_axis_rules),
      ):
        input_sharding = NamedSharding(
            mesh_cp,
            nn_partitioning.logical_to_mesh_axes(
                ("activation_batch", "activation_length", "activation_embed"),
                nn_partitioning.get_axis_rules(),
            ),
        )
        metadata_sharding = NamedSharding(
            mesh_cp,
            nn_partitioning.logical_to_mesh_axes((None, "activation_length"), nn_partitioning.get_axis_rules()),
        )
        lowered = jax.jit(lowered_fn).lower(
            jax.device_put(lnx, input_sharding),
            jax.device_put(decoder_positions, metadata_sharding),
            jax.device_put(decoder_segment_ids, metadata_sharding),
        )
        hlo_texts.append(lowered.compile().as_text())

    sequence_lengths = (cfg_cp.max_target_length,)
    for hlo_text in hlo_texts:
      self.assertGreater(
          len(hlo_test_utils.collective_lines(hlo_text, "collective-permute")),
          0,
      )
      self.assertLen(
          hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, sequence_lengths),
          0,
      )
      self.assertLen(
          hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, sequence_lengths, dtypes=("s32",)),
          0,
      )

  def test_residual_checkpoint_name_enables_context_remat_policy(self):
    """Named halo Splash residuals let save-context remat skip fwd recompute."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 devices for a halo context mesh.")
    cfg_cp = self._halo_test_config(2, context_parallel_load_balance=True)
    mesh_cp = Mesh(maxtext_utils.create_device_mesh(cfg_cp), cfg_cp.mesh_axes)
    batch_size = cfg_cp.global_batch_size_to_train_on
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      op = attention_op.AttentionOp(
          config=cfg_cp,
          mesh=mesh_cp,
          attention_kernel="flash",
          max_target_length=cfg_cp.max_target_length,
          num_query_heads=4,
          num_kv_heads=2,
          attention_type=common_types.AttentionType.LOCAL_SLIDING,
          sliding_window_size=cfg_cp.sliding_window_size,
      )
      query = jnp.zeros(
          (batch_size, cfg_cp.max_target_length, 4, cfg_cp.head_dim),
          jnp.bfloat16,
      )
      key = jnp.zeros(
          (batch_size, cfg_cp.max_target_length, 2, cfg_cp.head_dim),
          jnp.bfloat16,
      )

      def grad_jaxpr(policy):
        remat = jax.checkpoint(
            lambda q, k, v: op.tpu_flash_attention(q, k, v, None, model_mode=common_types.MODEL_MODE_TRAIN)[0]
            .astype(jnp.float32)
            .sum(),
            policy=policy,
        )
        return jax.make_jaxpr(jax.grad(remat))(query, key, key).jaxpr

      saved = grad_jaxpr(jax.checkpoint_policies.save_only_these_names("context"))
      recomputed = grad_jaxpr(jax.checkpoint_policies.nothing_saveable)

    self.assertEqual(_count_primitives(saved, "name", "context"), 2)
    self.assertLess(
        _count_primitives(saved, "name", "context"),
        _count_primitives(recomputed, "name", "context"),
    )


if __name__ == "__main__":
  absltest.main()
