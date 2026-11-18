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

"""CPU unit tests for online hot-expert replication config flags and validation."""

from absl.testing import absltest
from absl.testing import parameterized
import pydantic

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path


class ExpertReplicationConfigTest(parameterized.TestCase):
  """Tests defaults, bounds, and validator rejections of expert_replica_* flags."""

  def _init(self, **kw):
    """Initializes a Mixtral EP=2 config on a v5e-8 compile topology."""
    base_kw = {
        "run_name": "moe_expert_replication_config_test",
        "base_output_directory": "/tmp/moe_expert_replication_config_test",
        "enable_checkpointing": False,
        "compile_topology": "v5e-8",
        "compile_topology_num_slices": 1,
        "ici_expert_parallelism": 2,
        "ici_fsdp_parallelism": 4,
        "model_name": "mixtral-8x7b",
        "override_model_config": True,
        "sparse_matmul": True,
        "megablox": True,
        "use_ring_of_experts": False,
    }
    base_kw.update(kw)
    return pyconfig.initialize([None, get_test_config_path()], **base_kw)

  def test_defaults_off(self):
    cfg = self._init()
    self.assertEqual(cfg.expert_replicas_per_device, 0)
    self.assertEqual(cfg.expert_replica_max_send, 4)

  def test_flags_parse_and_max_send_upper_bound(self):
    cfg = self._init(
        expert_replicas_per_device=4,
        expert_replica_max_send=3,
    )
    self.assertEqual(cfg.expert_replicas_per_device, 4)
    self.assertEqual(cfg.expert_replica_max_send, 3)
    cfg_max = self._init(expert_replica_max_send=8)
    self.assertEqual(cfg_max.expert_replica_max_send, 8)

  def test_te_no_quant_allowed(self):
    cfg = self._init(
        expert_replicas_per_device=2,
        quantization="te_no_quant",
        te_gmm_quantization="te_no_quant",
    )
    self.assertEqual(cfg.expert_replicas_per_device, 2)
    self.assertEqual(cfg.quantization, "te_no_quant")
    self.assertEqual(cfg.te_gmm_quantization, "te_no_quant")

  @parameterized.parameters(
      ({"expert_replicas_per_device": -1},),
      ({"expert_replica_max_send": 0},),
      ({"expert_replica_max_send": 9},),
  )
  def test_invalid_bounds_rejected(self, kw):
    with self.assertRaises((pydantic.ValidationError, ValueError)):
      self._init(**kw)

  @parameterized.named_parameters(
      ("ring_of_experts", {"use_ring_of_experts": True}, "use_ring_of_experts=False"),
      ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks > 0"),
      ("mlp_bias", {"mlp_bias": True}, "mlp_bias=True"),
      ("chunk_wgrad", {"moe_accumulate_chunk_wgrad": True}, "moe_accumulate_chunk_wgrad=True"),
      (
          "dropless_fallback",
          {"moe_dropless_fallback": "layer", "use_ragged_sort": True, "ragged_buffer_factor": 1.25},
          "moe_dropless_fallback",
      ),
      ("ep_1", {"ici_expert_parallelism": 1, "ici_fsdp_parallelism": 8}, "expert parallelism > 1"),
      ("no_sparse_matmul", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("num_experts_1", {"num_experts": 1, "num_experts_per_tok": 1}, "num_experts > 1"),
      ("te_moe_block", {"te_moe_block": True}, "te_moe_block=True"),
      (
          "moe_use_fused_gmm3d",
          {
              "moe_use_fused_gmm3d": True,
              "use_tokamax_gmm": True,
              "moe_flat_fsdp_weights": True,
              "ici_expert_parallelism": 1,
              "ici_fsdp_parallelism": 8,
              "override_logical_axis_rules": True,
          },
          "moe_use_fused_gmm3d=True",
      ),
      ("fp8_moe", {"fp8_moe": True}, "FP8 or quantized MoE"),
      ("fp8_moe_fake_quant", {"fp8_moe_fake_quant": True}, "FP8 or quantized MoE"),
      ("quantization", {"quantization": "int8"}, "FP8 or quantized MoE"),
      ("te_gmm_quantization", {"te_gmm_quantization": "te_mxfp8"}, "FP8 or quantized MoE"),
      ("vllm_rpa", {"attention": "vllm_rpa"}, "does not support attention='vllm_rpa'"),
      ("vllm_batched_rpa", {"attention": "vllm_batched_rpa"}, "does not support attention='vllm_batched_rpa'"),
      ("batch_split", {"use_batch_split_schedule": True}, "does not support use_batch_split_schedule=True"),
      (
          "shard_exp_on_fsdp",
          {"shard_exp_on_fsdp": True, "shard_embed_moe_on_fsdp": False},
          "does not support shard_exp_on_fsdp=True",
      ),
      (
          "use_2d_fsdp_sharding",
          {"use_2d_fsdp_sharding": True, "shard_embed_moe_on_fsdp": False},
          "does not support use_2d_fsdp_sharding=True",
      ),
      ("moe_flat_fsdp_weights", {"moe_flat_fsdp_weights": True}, "does not support moe_flat_fsdp_weights=True"),
  )
  def test_config_validator_rejects_unsupported_combinations(self, overrides, expected_msg):
    """Verifies MaxTextConfig fails before compile on unsupported flag combinations."""
    with self.assertRaisesRegex(ValueError, expected_msg):
      self._init(expert_replicas_per_device=2, **overrides)


if __name__ == "__main__":
  absltest.main()
