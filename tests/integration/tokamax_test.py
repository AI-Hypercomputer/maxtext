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
"""Test for tokamax gmm."""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from maxtext.trainers.pre_train import train
from tests.utils.test_helpers import get_test_config_path
import pytest

train_main = train.main
gettempdir = tempfile.gettempdir


@pytest.mark.integration_test
class Train(parameterized.TestCase):
  """Smoke test for tokamax gmm.

  Similar to `train_using_ragged_dot_smoke_train.py`
  """

  def _run_smoke_train(
      self,
      *,
      quantization: str = "",
      use_gmm_v2: bool = False,
      use_gmm_v2_heuristic_tiling: bool = False,
      ici_expert_parallelism: int = 1,
      use_ring_of_experts: bool = False,
      moe_quantize_token_all_gather: bool = False,
      max_target_length: int | None = None,
  ):
    """Smoke train with small config."""
    sharding_tolerance = 0.22 if ici_expert_parallelism > 1 else 2e-2
    if max_target_length is None:
      # V1 FP8 TGMM requires the scale span to cover its 256-wide tile.
      # With only 128 tokens, it raises "subchannel_iters != 1" in the backward pass.
      max_target_length = 256 if quantization == "fp8_full" and not use_gmm_v2 else 128
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=test_smoke_train",
        # model
        "base_emb_dim=256",
        "base_num_query_heads=1",
        "base_num_kv_heads=1",
        "base_mlp_dim=256",
        "base_moe_mlp_dim=256",
        "base_num_decoder_layers=2",
        "head_dim=64",
        "decoder_block=deepseek",
        "attention_type=mla",
        "num_experts=2",
        "shared_experts=1",
        f"ici_expert_parallelism={ici_expert_parallelism}",
        f"sharding_tolerance={sharding_tolerance}",
        # tokamax gmm
        "sparse_matmul=True",
        "megablox=False",
        "use_tokamax_gmm=True",
        f"use_gmm_v2={use_gmm_v2}",
        f"use_gmm_v2_heuristic_tiling={use_gmm_v2_heuristic_tiling}",
        f"use_ring_of_experts={use_ring_of_experts}",
        # tile sizes
        "wi_tile_fwd_batch_seq=128",
        "wi_tile_fwd_embed_dim=128",
        "wi_tile_fwd_mlp_dim=128",
        "wi_tile_dlhs_batch_seq=128",
        "wi_tile_dlhs_embed_dim=128",
        "wi_tile_dlhs_mlp_dim=128",
        "wi_tile_drhs_batch_seq=128",
        "wi_tile_drhs_embed_dim=128",
        "wi_tile_drhs_mlp_dim=128",
        "wo_tile_fwd_batch_seq=128",
        "wo_tile_fwd_embed_dim=128",
        "wo_tile_fwd_mlp_dim=128",
        "wo_tile_dlhs_batch_seq=128",
        "wo_tile_dlhs_embed_dim=128",
        "wo_tile_dlhs_mlp_dim=128",
        "wo_tile_drhs_batch_seq=128",
        "wo_tile_drhs_embed_dim=128",
        "wo_tile_drhs_mlp_dim=128",
        # tokamax splash
        f"max_target_length={max_target_length}",
        "attention=flash",
        "use_tokamax_splash=False",
        # quantization
        f"quantization={quantization}",
        f"moe_quantize_token_all_gather={moe_quantize_token_all_gather}",
        "use_qwix_quantization=True",
        "weight_quantization_calibration_method=fixed,-224,224",
        "act_quantization_calibration_method=fixed,-224,224",
        "bwd_quantization_calibration_method=absmax",
        # train
        "per_device_batch_size=1",
        "dataset_type=synthetic",
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        "monitor_goodput=False",
        f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
    ]
    train_main(args)

  @parameterized.named_parameters(
      {"testcase_name": "bf16", "quantization": ""},
      {"testcase_name": "fp8", "quantization": "fp8"},  # not quantize gmm
      {"testcase_name": "fp8_full", "quantization": "fp8_full"},  # quantize gmm
  )
  @pytest.mark.tpu_only
  def test_tokamax_v1(self, quantization: str):
    self._run_smoke_train(use_gmm_v2=False, quantization=quantization)

  @parameterized.named_parameters(
      {
          "testcase_name": "bf16_ep1",
          "quantization": "",
          "ici_expert_parallelism": 1,
      },
      {
          "testcase_name": "bf16_heuristic_ep1",
          "quantization": "",
          "ici_expert_parallelism": 1,
          "use_gmm_v2_heuristic_tiling": True,
      },
      {
          "testcase_name": "fp8_full_ep1",
          "quantization": "fp8_full",
          "ici_expert_parallelism": 1,
      },
      {
          "testcase_name": "bf16_ep2",
          "quantization": "",
          "ici_expert_parallelism": 2,
      },
      {
          "testcase_name": "fp8_full_ep2",
          "quantization": "fp8_full",
          "ici_expert_parallelism": 2,
      },
  )
  @pytest.mark.tpu_only
  def test_tokamax_v2(
      self,
      quantization: str,
      ici_expert_parallelism: int,
      use_gmm_v2_heuristic_tiling: bool = False,
  ):
    self._run_smoke_train(
        use_gmm_v2=True,
        quantization=quantization,
        ici_expert_parallelism=ici_expert_parallelism,
        use_gmm_v2_heuristic_tiling=use_gmm_v2_heuristic_tiling,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "fp8_tag_ep2",
          "quantization": "fp8_full",
          "ici_expert_parallelism": 2,
      },
  )
  @pytest.mark.tpu_only
  def test_tokamax_v2_quantize_token_all_gather(
      self,
      quantization: str,
      ici_expert_parallelism: int,
  ):
    self._run_smoke_train(
        use_gmm_v2=True,
        quantization=quantization,
        ici_expert_parallelism=ici_expert_parallelism,
        use_ring_of_experts=True,
        moe_quantize_token_all_gather=True,
    )


if __name__ == "__main__":
  absltest.main()
