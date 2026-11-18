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

"""Unit tests for Qwen 3.5 / Qwen MAX Lineage adapter, configs, and shared kernels."""

import os
import types
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import pytest

from maxtext.configs import pyconfig
from maxtext.experimental.lineage import lineage_adapter
from maxtext.experimental.lineage.qwen import qwen_adapter
from maxtext.kernels.gdn import wrapper as gdn_wrapper
from maxtext.kernels.ragged import ragged_gather_tc
from maxtext.kernels.ragged import ragged_sort
from maxtext.kernels.ragged import ragged_sort_tc
from tests.utils.test_helpers import get_test_config_path


@pytest.mark.cpu_only
class QwenLineageTest(unittest.TestCase):
  """CPU unit tests for Qwen Lineage configs, adapter, and forward/backward."""

  def test_shared_kernel_exports(self):
    self.assertTrue(callable(gdn_wrapper.fused_conv1d_gdn))
    self.assertTrue(callable(gdn_wrapper.fused_conv1d_gdn_states))
    self.assertTrue(callable(ragged_gather_tc.tc_scatter_1d_unique))
    self.assertTrue(callable(ragged_gather_tc.tc_gather_1d_unique))
    self.assertTrue(callable(ragged_gather_tc.packed_stable_sort_with_weights))
    self.assertTrue(callable(ragged_sort_tc.tc_unsort_row_weights))
    self.assertTrue(callable(ragged_sort.ring_ragged_unsort_down_projection))

  def test_lineage_model_configs_initialize(self):
    expected = {
        "qwen3.8-2.4t-a95b-lineage": (92, 8192, 2048),
        "qwen3.5-397b-a17b-lineage": (60, 4096, 1024),
    }
    for model_name, (layers, emb_dim, moe_dim) in expected.items():
      cfg = pyconfig.initialize([
          "train.py",
          get_test_config_path(),
          f"model_name={model_name}",
          "run_name=qwen_lineage_cfg_test",
          "hardware=cpu",
          "skip_jax_distributed_system=true",
      ])
      self.assertTrue(lineage_adapter.is_native(cfg))
      self.assertTrue(qwen_adapter.is_qwen_lineage(cfg))
      self.assertEqual(cfg.num_decoder_layers, layers)
      self.assertEqual(cfg.emb_dim, emb_dim)
      self.assertEqual(cfg.moe_mlp_dim, moe_dim)
      self.assertEqual(cfg.ici_parallelism, [32, 8, 2])

  def test_validate_supported_rejects_unsupported_options(self):
    base = dict(
        scan_layers=True,
        num_decoder_layers=8,
        inhomogeneous_layer_cycle_interval=4,
        shared_experts=1,
        quantization=None,
        gradient_accumulation_steps=1,
        use_tunix_gradient_accumulation=False,
        training_objective="causal_lm",
        use_multimodal=False,
        enable_dropout=False,
        dropout_rate=0.0,
        enable_diloco=False,
        lora=None,
        parameter_memory_host_offload=False,
    )
    qwen_adapter.validate_supported(types.SimpleNamespace(**base))

    for bad_overrides in (
        {"quantization": "fp8"},
        {"enable_dropout": True, "dropout_rate": 0.1},
        {"scan_layers": False},
        {"num_decoder_layers": 6},
        {"parameter_memory_host_offload": True},
    ):
      with self.assertRaises(ValueError):
        qwen_adapter.validate_supported(types.SimpleNamespace(**{**base, **bad_overrides}))

  def test_tiny_eval_shape_and_forward_backward(self):
    num_devices = len(jax.devices())
    if num_devices >= 4 and num_devices % 2 == 0:
      fsdp, ep = 2, num_devices // 2
    else:
      fsdp, ep = 1, num_devices

    cfg = pyconfig.initialize([
        "train.py",
        get_test_config_path(),
        "model_name=qwen3.5-397b-a17b-lineage",
        "run_name=qwen_lineage_tiny_test",
        "hardware=cpu",
        "skip_jax_distributed_system=true",
        "override_model_config=true",
        "mesh_axes=['fsdp','expert']",
        f"ici_fsdp_parallelism={fsdp}",
        f"ici_expert_parallelism={ep}",
        f"ici_parallelism=[{fsdp},{ep}]",
        "dcn_parallelism=[1,1]",
        "data_sharding=[['fsdp','expert']]",
        "base_num_decoder_layers=8",
        "base_emb_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=32",
        "vocab_size=256",
        "base_mlp_dim=64",
        "base_moe_mlp_dim=64",
        f"num_experts={max(8, ep * 4)}",
        "num_experts_per_tok=2",
        "gdn_num_key_heads=2",
        "gdn_num_value_heads=4",
        "gdn_key_head_dim=32",
        "gdn_value_head_dim=32",
        "gdn_chunk_size=16",
        "max_target_length=32",
        "per_device_batch_size=2",
        "ragged_buffer_factor=3.5",
        "lineage_quantization=none",
        "weight_dtype=float32",
        "dtype=float32",
    ])

    devices_arr = lineage_adapter.create_device_mesh(cfg.ici_parallelism, jax.devices(), config=cfg)
    mesh = Mesh(devices_arr, cfg.mesh_axes)

    abs_model = nnx.eval_shape(lambda: lineage_adapter.create_transformer(cfg, mesh, rngs=nnx.Rngs(params=0)))
    self.assertEqual(len(jax.tree.leaves(nnx.state(abs_model, nnx.Param))), 66)

    model = lineage_adapter.create_transformer(cfg, mesh, rngs=nnx.Rngs(params=0))
    global_b = num_devices * int(cfg.per_device_batch_size)
    seq_len = int(cfg.max_target_length)
    data_sharding = NamedSharding(mesh, P(("fsdp", "expert"), None))
    data = {
        "inputs": jax.device_put(
            jax.random.randint(jax.random.PRNGKey(1), (global_b, seq_len), 0, 256, jnp.int32),
            data_sharding,
        ),
        "targets": jax.device_put(
            jax.random.randint(jax.random.PRNGKey(2), (global_b, seq_len), 0, 256, jnp.int32),
            data_sharding,
        ),
        "inputs_position": jax.device_put(
            jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (global_b, seq_len)),
            data_sharding,
        ),
        "inputs_segmentation": jax.device_put(jnp.ones((global_b, seq_len), dtype=jnp.int32), data_sharding),
        "targets_segmentation": jax.device_put(jnp.ones((global_b, seq_len), dtype=jnp.int32), data_sharding),
    }

    def _loss_fn(m):
      return lineage_adapter.loss_and_aux(m, cfg, data, is_train=True)

    (loss, aux), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(model)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertEqual(float(aux["total_weights"]), float(global_b * seq_len))
    grad_leaves = jax.tree.leaves(grads)
    self.assertEqual(len(grad_leaves), 66)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves))
    self.assertIs(lineage_adapter.finalize_accumulated_step(model, aux), aux)


if __name__ == "__main__":
  unittest.main()
