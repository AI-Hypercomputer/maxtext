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

"""Unit tests for the NNX path of generate_param_only_checkpoint.

Covers `_possibly_unroll_params_nnx` (slicing scanned NNX layers) and the
shape parity of `_save_decode_checkpoint_nnx`'s bf16 cast.
"""

import contextlib
import inspect
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from flax.training import train_state as linen_train_state

from maxtext.common import checkpointing
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.generate_param_only_checkpoint import (
    _possibly_unroll_lora_params_nnx,
    _possibly_unroll_params_nnx,
    _save_decode_checkpoint_nnx,
)
from tests.utils.test_helpers import get_test_config_path


@contextlib.contextmanager
def _orbax_checkpointer_compat():
  """Compatibility shim when running locally against orbax-checkpoint < 0.12.8 (no `logger` kwarg)."""
  orig_cls = checkpointing.ocp.training.Checkpointer
  if "logger" in inspect.signature(orig_cls.__init__).parameters:
    yield
    return

  def _compat_checkpointer(*args, **kwargs):
    if kwargs.get("logger") is None:
      kwargs.pop("logger", None)
    return orig_cls(*args, **kwargs)

  with mock.patch.object(checkpointing.ocp.training, "Checkpointer", side_effect=_compat_checkpointer):
    yield


class _ScanLayerLeaf(nnx.Module):
  """One scanned-layer kernel with leading shape `[num_layers, *]`."""

  def __init__(self, num_layers: int, in_dim: int, out_dim: int):
    self.kernel = nnx.Param(
        jnp.arange(num_layers * in_dim * out_dim, dtype=jnp.float32).reshape(num_layers, in_dim, out_dim)
    )


class _Decoder(nnx.Module):

  def __init__(self, num_layers: int):
    self.layers = _ScanLayerLeaf(num_layers, 3, 5)


class _Model(nnx.Module):

  def __init__(self, num_layers: int):
    self.decoder = _Decoder(num_layers)


def _make_split_state(num_layers: int):
  model = _Model(num_layers)
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  ts = train_state_nnx.TrainStateNNX(model, optimizer)
  _, state = nnx.split(ts)
  return state


def _make_shardings_state(state, mesh):
  """Build a sibling shardings tree where each Variable is replaced by NamedSharding(replicated)."""

  def to_named(v):
    return NamedSharding(mesh, PartitionSpec())

  return jax.tree_util.tree_map(to_named, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


class PossiblyUnrollParamsNNXTest(unittest.TestCase):

  def setUp(self):
    devices = np.array(jax.devices()).reshape(-1)
    self.mesh = Mesh(devices, ("data",))

  def test_unrolls_scanned_layers(self):
    num_layers = 3
    state = _make_split_state(num_layers)
    shardings = _make_shardings_state(state, self.mesh)

    original_kernel = np.asarray(state.model.decoder.layers.kernel[...])

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.LLAMA2,
        num_decoder_layers=num_layers,
    )

    _possibly_unroll_params_nnx(config, state, shardings, self.mesh)

    self.assertNotIn("layers", state.model.decoder)
    self.assertNotIn("layers", shardings.model.decoder)
    for i in range(num_layers):
      self.assertIn(f"layers_{i}", state.model.decoder)
      self.assertIn(f"layers_{i}", shardings.model.decoder)
      sliced = state.model.decoder[f"layers_{i}"]["kernel"][...]
      expected = jnp.take(original_kernel, i, axis=0)
      self.assertTrue(jnp.array_equal(sliced, expected))

  def test_deepseek_split(self):
    """DeepSeek decoder has separate dense/moe layer collections."""

    # Build a DeepSeek-flavored synthetic model with two scanned groups.
    class _DeepSeekDecoder(nnx.Module):

      def __init__(self):
        self.dense_layers = _ScanLayerLeaf(2, 3, 5)
        self.moe_layers = _ScanLayerLeaf(3, 3, 5)

    class _DSModel(nnx.Module):

      def __init__(self):
        self.decoder = _DeepSeekDecoder()

    model = _DSModel()
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    ts = train_state_nnx.TrainStateNNX(model, optimizer)
    _, state = nnx.split(ts)
    shardings = _make_shardings_state(state, self.mesh)

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.DEEPSEEK,
        num_decoder_layers=5,
        first_num_dense_layers=2,
    )

    _possibly_unroll_params_nnx(config, state, shardings, self.mesh)

    self.assertNotIn("dense_layers", state.model.decoder)
    self.assertNotIn("moe_layers", state.model.decoder)
    for i in range(2):
      self.assertIn(f"dense_layers_{i}", state.model.decoder)
    for i in range(3):
      self.assertIn(f"moe_layers_{i}", state.model.decoder)


class PossiblyUnrollLoraParamsNNXTest(unittest.TestCase):
  """The LoRA delta tree is single-nested (`{"decoder": {...}}`) and held in a
  Linen `TrainState` even on the NNX path — the unroll has to walk that shape."""

  def setUp(self):
    devices = np.array(jax.devices()).reshape(-1)
    self.mesh = Mesh(devices, ("data",))

  def _make_lora_state(self, num_layers: int, lora_rank: int = 4):
    """Build a synthetic LoRA delta TrainState mirroring `get_lora_abstract_state_nnx`'s output shape."""
    lora_a = jnp.arange(num_layers * 8 * lora_rank, dtype=jnp.float32).reshape(num_layers, 8, lora_rank)
    lora_b = jnp.arange(num_layers * lora_rank * 4 * 2, dtype=jnp.float32).reshape(num_layers, lora_rank, 4, 2)
    params = {
        "decoder": {
            "layers": {
                "self_attention": {
                    "query": {"lora_a.kernel": lora_a, "lora_b.kernel": lora_b},
                }
            }
        }
    }
    annotations_params = jax.tree_util.tree_map(lambda _: PartitionSpec(), params)
    state = linen_train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})
    annotations = linen_train_state.TrainState(step=0, apply_fn=None, params=annotations_params, tx=None, opt_state={})
    return state, annotations

  def test_unrolls_scanned_lora_layers(self):
    num_layers = 3
    state, annotations = self._make_lora_state(num_layers)
    original_a = np.asarray(state.params["decoder"]["layers"]["self_attention"]["query"]["lora_a.kernel"])

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.LLAMA2,
        num_decoder_layers=num_layers,
    )

    _possibly_unroll_lora_params_nnx(config, state, annotations, self.mesh)

    self.assertNotIn("layers", state.params["decoder"])
    self.assertNotIn("layers", annotations.params["decoder"])
    for i in range(num_layers):
      self.assertIn(f"layers_{i}", state.params["decoder"])
      sliced_a = state.params["decoder"][f"layers_{i}"]["self_attention"]["query"]["lora_a.kernel"]
      expected = jnp.take(original_a, i, axis=0)
      self.assertTrue(jnp.array_equal(sliced_a, expected))

  def test_unrolls_glm5_scanned_lora_layers(self):
    dense_a = jnp.arange(2 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 4)
    moe_a = jnp.arange(3 * 8 * 4, dtype=jnp.float32).reshape(3, 8, 4)
    orig_dense_a = np.asarray(dense_a)
    orig_moe_a = np.asarray(moe_a)
    params = {
        "decoder": {
            "dense_layers": {"self_attention": {"query": {"lora_a.kernel": dense_a}}},
            "moe_layers": {"self_attention": {"query": {"lora_a.kernel": moe_a}}},
        }
    }
    annotations_params = jax.tree_util.tree_map(lambda _: PartitionSpec(), params)
    state = linen_train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})
    annotations = linen_train_state.TrainState(step=0, apply_fn=None, params=annotations_params, tx=None, opt_state={})

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.GLM5,
        num_decoder_layers=5,
        first_num_dense_layers=2,
    )

    _possibly_unroll_lora_params_nnx(config, state, annotations, self.mesh)

    self.assertNotIn("dense_layers", state.params["decoder"])
    self.assertNotIn("moe_layers", state.params["decoder"])
    for i in range(2):
      np.testing.assert_array_equal(
          np.asarray(state.params["decoder"][f"dense_layers_{i}"]["self_attention"]["query"]["lora_a.kernel"]),
          orig_dense_a[i],
      )
    for i in range(3):
      np.testing.assert_array_equal(
          np.asarray(state.params["decoder"][f"moe_layers_{i}"]["self_attention"]["query"]["lora_a.kernel"]),
          orig_moe_a[i],
      )


class GLM5UnrollAndSaveNNXTest(unittest.TestCase):
  """Tests GLM-5 / GLM-5.2 unrolling, MoEBiasVar axis-0 slicing, IndexShare pruning, and decode restore."""

  def setUp(self):
    devices = np.array(jax.devices()).reshape(-1)
    self.mesh = Mesh(devices, ("data",))

  def test_glm5_moebiasvar_axis0_and_alternating_index_share_pruning(self):
    """Verifies param_scan_axis=1 Param slicing alongside axis-0 MoEBiasVar slicing and FFSSFS pruning."""

    class _KernelLeaf(nnx.Module):

      def __init__(self, arr):
        self.kernel = nnx.Param(arr)

    class _ScaleLeaf(nnx.Module):

      def __init__(self, arr):
        self.scale = nnx.Param(arr)

    class _IndexerStub(nnx.Module):

      def __init__(self, num_layers: int):
        self.wq_b = _KernelLeaf(jnp.arange(4 * num_layers * 3, dtype=jnp.float32).reshape(4, num_layers, 3))
        self.k_norm = _ScaleLeaf(jnp.linspace(1.0, 2.0, num_layers * 4, dtype=jnp.float32).reshape(4, num_layers))
        self.weights_proj = _KernelLeaf(
            jnp.linspace(0.1, 0.9, num_layers * 6, dtype=jnp.float32).reshape(3, num_layers, 2)
        )

    class _SelfAttnStub(nnx.Module):

      def __init__(self, num_layers: int):
        self.q_proj = nnx.Param(jnp.arange(3 * num_layers * 5, dtype=jnp.float32).reshape(3, num_layers, 5))
        self.indexer = _IndexerStub(num_layers)

    class _DenseScanGroup(nnx.Module):

      def __init__(self, num_layers: int):
        self.self_attention = _SelfAttnStub(num_layers)

    class _GateStub(nnx.Module):
      """Synthetic MoE router holding both an nnx.Param kernel and an axis-0 MoEBiasVar."""

      def __init__(self, num_moe_layers: int, num_experts: int):
        self.kernel = nnx.Param(
            jnp.arange(3 * num_moe_layers * num_experts, dtype=jnp.float32).reshape(3, num_moe_layers, num_experts)
        )
        # Non-Param MoEBiasVar is stacked by nnx.scan on axis 0: [num_moe_layers, num_experts].
        bias_vals = jnp.asarray(
            [[34.62297 + l + 0.01 * e for e in range(num_experts)] for l in range(num_moe_layers)],
            dtype=jnp.float32,
        )
        self.bias = moe.MoEBiasVar(bias_vals)

    class _MoeInnerStub(nnx.Module):

      def __init__(self, num_moe_layers: int, num_experts: int):
        self.gate = _GateStub(num_moe_layers, num_experts)

    class _MoeOuterStub(nnx.Module):

      def __init__(self, num_moe_layers: int, num_experts: int):
        self.MoeBlock_0 = _MoeInnerStub(num_moe_layers, num_experts)

    class _MoEScanGroup(nnx.Module):

      def __init__(self, num_moe_layers: int, num_experts: int):
        self.self_attention = _SelfAttnStub(num_moe_layers)
        self.DeepSeekMoeBlock_0 = _MoeOuterStub(num_moe_layers, num_experts)

    class _GLM5DecoderStub(nnx.Module):

      def __init__(self):
        self.dense_layers = _DenseScanGroup(2)
        self.moe_layers = _MoEScanGroup(4, 6)

    class _GLM5ModelStub(nnx.Module):

      def __init__(self):
        self.decoder = _GLM5DecoderStub()

    model = _GLM5ModelStub()
    orig_moe_bias = np.asarray(model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias[...])
    orig_moe_idx_wqb = np.asarray(model.decoder.moe_layers.self_attention.indexer.wq_b.kernel[...])

    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)
    ts_state = train_state_nnx.TrainStateNNX(model, optimizer)
    _, state = nnx.split(ts_state)
    shardings = _make_shardings_state(state, self.mesh)

    # Pattern "FFSSFS" over 6 total layers (2 dense: 0=F, 1=F; 4 MoE: 2=S, 3=S, 4=F, 5=S).
    # Notice global layer 4 (`moe_layers_2`) is Full AFTER Shared layers 2 & 3!
    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        param_scan_axis=1,
        decoder_block=DecoderBlockType.GLM5,
        num_decoder_layers=6,
        first_num_dense_layers=2,
        use_index_share=True,
        index_share_pattern="FFSSFS",
        checkpoint_dir="",
    )

    _possibly_unroll_params_nnx(config, state, shardings, self.mesh)

    self.assertNotIn("dense_layers", state.model.decoder)
    self.assertNotIn("moe_layers", state.model.decoder)

    # Dense layers 0, 1 are Full ('F') -> retain indexer
    for d_idx in (0, 1):
      self.assertIn("indexer", state.model.decoder[f"dense_layers_{d_idx}"]["self_attention"])
      self.assertIn("indexer", shardings.model.decoder[f"dense_layers_{d_idx}"]["self_attention"])

    # MoE layers: 0->S, 1->S, 2->F (after S!), 3->S
    for m_idx, is_full in ((0, False), (1, False), (2, True), (3, False)):
      moe_layer = state.model.decoder[f"moe_layers_{m_idx}"]
      moe_sharding = shardings.model.decoder[f"moe_layers_{m_idx}"]
      bias_var = moe_layer["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"]
      self.assertIsInstance(bias_var, moe.MoEBiasVar)
      self.assertEqual(bias_var[...].shape, (6,))
      np.testing.assert_array_equal(np.asarray(bias_var[...]), orig_moe_bias[m_idx])

      if is_full:
        self.assertIn("indexer", moe_layer["self_attention"])
        self.assertIn("indexer", moe_sharding["self_attention"])
        np.testing.assert_array_equal(
            np.asarray(moe_layer["self_attention"]["indexer"]["wq_b"]["kernel"][...]),
            orig_moe_idx_wqb[:, m_idx, :],
        )
      else:
        self.assertNotIn("indexer", moe_layer["self_attention"])
        self.assertNotIn("indexer", moe_sharding["self_attention"])

    # Verify _save_decode_checkpoint_nnx preserves float32 for MoEBiasVar, indexer.k_norm, and indexer.weights_proj
    saved_payloads = []
    with (
        mock.patch.object(
            checkpointing, "save_checkpoint", side_effect=lambda mgr, step, item: saved_payloads.append(item) or True
        ),
        mock.patch.object(checkpointing, "wait_until_finished"),
    ):
      _save_decode_checkpoint_nnx(config, state, checkpoint_manager=object())

    self.assertEqual(len(saved_payloads), 1)
    saved_dec = saved_payloads[0]["decoder"]
    for m_idx in range(4):
      saved_bias = saved_dec[f"moe_layers_{m_idx}"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"]["value"]
      self.assertEqual(saved_bias.dtype, jnp.float32)
      np.testing.assert_array_equal(np.asarray(saved_bias), orig_moe_bias[m_idx])

    full_moe_attn = saved_dec["moe_layers_2"]["self_attention"]
    self.assertEqual(full_moe_attn["q_proj"]["value"].dtype, jnp.bfloat16)
    self.assertEqual(full_moe_attn["indexer"]["wq_b"]["kernel"]["value"].dtype, jnp.bfloat16)
    self.assertEqual(full_moe_attn["indexer"]["k_norm"]["scale"]["value"].dtype, jnp.float32)
    self.assertEqual(full_moe_attn["indexer"]["weights_proj"]["kernel"]["value"].dtype, jnp.float32)
    self.assertNotIn("indexer", saved_dec["moe_layers_0"]["self_attention"])

  def test_glm52_roundtrip_unroll_save_and_from_pretrained_restore(self):
    """End-to-end scanned GLM-5.2 -> unroll + prune + save -> unscanned pruned from_pretrained parity."""
    with tempfile.TemporaryDirectory() as tmpdir:
      common_overrides = [
          "skip_jax_distributed_system=true",
          "model_name=glm5.2-744b",
          "override_model_config=true",
          "base_emb_dim=64",
          "base_num_query_heads=4",
          "base_num_kv_heads=4",
          "base_mlp_dim=128",
          "base_moe_mlp_dim=64",
          "base_num_decoder_layers=6",
          "first_num_dense_layers=2",
          "num_experts=4",
          "num_experts_per_tok=2",
          "q_lora_rank=32",
          "kv_lora_rank=16",
          "qk_nope_head_dim=16",
          "qk_rope_head_dim=8",
          "v_head_dim=16",
          "indexer_n_heads=4",
          "indexer_head_dim=16",
          "indexer_topk=4",
          "vocab_size=256",
          "index_share_pattern=FFSSFS",
          "per_device_batch_size=1",
          "max_target_length=8",
          "max_prefill_predict_length=4",
          "weight_dtype=bfloat16",
          "dtype=float32",
          "activations_in_float32=true",
          "matmul_precision=highest",
          "async_checkpointing=false",
      ]
      cfg_scanned = pyconfig.initialize(
          [
              None,
              get_test_config_path(),
              f"base_output_directory={tmpdir}",
              "run_name=glm52_unroll_rt",
              "scan_layers=true",
              "force_unroll=true",
              *common_overrides,
          ]
      )
      devices_array = maxtext_utils.create_device_mesh(cfg_scanned)
      mesh = Mesh(devices_array, cfg_scanned.mesh_axes)
      rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
      scanned_model = model_creation_utils.from_config(cfg_scanned, mesh=mesh, rngs=rngs)

      # Set distinct per-layer float32 routing biases so axis-0 slicing is verified end-to-end.
      expected_bias = jnp.asarray(
          [[34.62297 + 2.0 * l + 0.1 * e for e in range(4)] for l in range(4)],
          dtype=jnp.float32,
      )
      scanned_model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias[...] = expected_bias
      expected_full_moe2_wqb = np.asarray(
          scanned_model.decoder.moe_layers.self_attention.indexer.wq_b.kernel[...][:, 2, :]
      )

      batch_size = cfg_scanned.global_batch_size_to_train_on
      seq_len = cfg_scanned.max_target_length
      ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      segment_ids = jnp.full((batch_size, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR, dtype=jnp.int32)

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_scanned.logical_axis_rules):
        scanned_logits = np.asarray(scanned_model(ids, positions, segment_ids, enable_dropout=False))

      optimizer = nnx.Optimizer(scanned_model, optax.sgd(1e-3), wrt=nnx.Param)
      ts_state = train_state_nnx.TrainStateNNX(scanned_model, optimizer)
      _, state = nnx.split(ts_state)
      shardings = _make_shardings_state(state, mesh)

      _possibly_unroll_params_nnx(cfg_scanned, state, shardings, mesh)

      with _orbax_checkpointer_compat():
        ckpt_mgr = checkpointing.create_orbax_checkpoint_manager(
            cfg_scanned.checkpoint_dir,
            enable_checkpointing=True,
            use_async=False,
            save_interval_steps=1,
            use_ocdbt=cfg_scanned.checkpoint_storage_use_ocdbt,
            use_zarr3=cfg_scanned.checkpoint_storage_use_zarr3,
        )
        _save_decode_checkpoint_nnx(cfg_scanned, state, ckpt_mgr)

        ckpt_items_path = os.path.join(cfg_scanned.checkpoint_dir, "0", "items")
        cfg_unscanned = pyconfig.initialize(
            [
                None,
                get_test_config_path(),
                f"base_output_directory={tmpdir}",
                "run_name=glm52_restore_rt",
                "scan_layers=false",
                "prune_shared_indexers=true",
                f"load_parameters_path={ckpt_items_path}",
                *common_overrides,
            ]
        )
        restored_model = model_creation_utils.from_pretrained(cfg_unscanned, mesh=mesh, model_mode=MODEL_MODE_TRAIN)

      # Verify MoEBiasVar float32 values and Full vs Shared indexer presence on restored model
      for m_idx in range(4):
        restored_layer = getattr(restored_model.decoder, f"moe_layers_{m_idx}")
        restored_bias = restored_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias[...]
        self.assertEqual(restored_bias.dtype, jnp.float32)
        np.testing.assert_array_equal(np.asarray(restored_bias), np.asarray(expected_bias[m_idx]))

      self.assertIsNone(restored_model.decoder.moe_layers_0.self_attention.indexer)
      self.assertIsNone(restored_model.decoder.moe_layers_1.self_attention.indexer)
      self.assertIsNotNone(restored_model.decoder.moe_layers_2.self_attention.indexer)
      self.assertIsNone(restored_model.decoder.moe_layers_3.self_attention.indexer)
      np.testing.assert_array_equal(
          np.asarray(restored_model.decoder.moe_layers_2.self_attention.indexer.wq_b.kernel[...]),
          expected_full_moe2_wqb,
      )

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_unscanned.logical_axis_rules):
        restored_logits = np.asarray(restored_model(ids, positions, segment_ids, enable_dropout=False))

      np.testing.assert_allclose(restored_logits, scanned_logits, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
