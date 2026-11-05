# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for `maxtext.experimental.maxtext_engine.train`."""

# pylint: disable=protected-access

import os
import re
import sys
from typing import Any

from absl.testing import absltest
from flax import nnx
import jax
from maxtext.common.data_loader import DataLoader
from maxtext.configs import pyconfig
from maxtext.experimental.maxtext_engine import train as engine_train_script
from maxtext.trainers.pre_train import train as pre_train
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest

# `maxtext.training_engine` imports `tunix` (`abstract_engine.py`), so this suite runs in the
# post-training CI container where `google-tunix` is installed.
pytestmark = [
    pytest.mark.post_training,
    pytest.mark.integration_test,
    pytest.mark.tpu_only,
]


def _qwen3_argv(run_name: str, output_dir: str, **overrides: Any) -> list[str]:
  """Builds CLI arguments for a compact `qwen3-0.6b` run on synthetic data."""
  defaults: dict[str, Any] = {
      "model_name": "qwen3-0.6b",
      "override_model_config": True,
      "dataset_type": "synthetic",
      "base_num_decoder_layers": 2,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "vocab_size": 128,
      "max_target_length": 32,
      "per_device_batch_size": 1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "grad_dtype": "float32",
      "matmul_precision": "highest",
      "sharding_tolerance": 0.3,
      "enable_checkpointing": False,
      "async_checkpointing": False,
      "checkpoint_period": 1,
      "convert_checkpoint_if_possible": False,
      "enable_tensorboard": False,
      "use_vertex_tensorboard": False,
      "enable_goodput_recording": False,
      "monitor_goodput": False,
      "enable_checkpoint_cloud_logger": False,
      "record_internal_nn_metrics": False,
      "skip_jax_distributed_system": True,
      "profiler_steps": 0,
      "log_config": False,
      "init_weights_seed": 42,
      "learning_rate": 1e-2,
      "learning_rate_schedule_steps": -1,
      "warmup_steps_fraction": 0.0,
  }
  defaults.update(overrides)
  return [
      sys.argv[0],
      get_test_config_path("base.yml"),
      f"run_name={run_name}",
      f"base_output_directory={output_dir}",
  ] + [f"{k}={v}" for k, v in defaults.items()]


def _flat_params(pure_params: dict[str, Any]) -> dict[str, np.ndarray]:
  """Returns `{path: float64_ndarray}` across every leaf of `pure_params`."""
  return {
      jax.tree_util.keystr(path): np.asarray(leaf, dtype=np.float64)
      for path, leaf in jax.tree_util.tree_leaves_with_path(pure_params)
  }


def _restore_saved_engine(argv: list[str]) -> tuple[int, dict[str, np.ndarray]]:
  """Restores the latest checkpoint for `argv` into a fresh engine and returns `(step, flat_params)`."""
  cfg = pyconfig.initialize(argv)
  mesh = maxtext_utils.get_mesh_from_config(cfg)
  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
  try:
    engine.restore_checkpoint()
    params = _flat_params(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))
    return engine.train_step, params
  finally:
    engine.close()


class MaxTextEngineTrainTest(absltest.TestCase):
  """Integration suite covering `maxtext.experimental.maxtext_engine.train`."""

  def test_micro_batch_stream_and_zero_collective_split(self):
    """Verifies strided micro-batch splitting, single-micro-step passthrough, and zero-collective HLO."""
    out_dir = self.create_tempdir().full_path
    micro_steps = 2
    # `per_device_batch_size=2` ensures `global_rows >= 4` even on 1 device, so strided `k::2`
    # rows (`[0, 2]`) differ strictly from contiguous `[:2]` rows (`[0, 1]`).
    cfg = pyconfig.initialize(
        _qwen3_argv(
            "qwen3_micro_batch_split",
            out_dir,
            steps=1,
            per_device_batch_size=2,
            gradient_accumulation_steps=micro_steps,
        )
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    global_rows = int(cfg.global_batch_size_to_load)
    seq_len = int(cfg.max_target_length)
    tokens = np.arange(global_rows * seq_len, dtype=np.int32).reshape(global_rows, seq_len) % int(cfg.vocab_size)
    global_batch = {
        "inputs": tokens,
        "targets": np.roll(tokens, -1, axis=-1),
        "inputs_position": np.tile(np.arange(seq_len, dtype=np.int32), (global_rows, 1)),
        "inputs_segmentation": np.ones((global_rows, seq_len), dtype=np.int32),
        "targets_segmentation": np.ones((global_rows, seq_len), dtype=np.int32),
    }

    loader = DataLoader(cfg, mesh, iter([global_batch]), goodput_recorder=None)
    stream = engine_train_script.micro_batch_stream(loader, micro_steps)
    for k in range(micro_steps):
      micro_batch = next(stream)
      for key, arr in global_batch.items():
        np.testing.assert_array_equal(np.asarray(micro_batch[key]), arr[k::micro_steps], err_msg=f"k={k}, key={key}")

    data_sharding = sharding.get_input_data_sharding(cfg, mesh)
    placed_batch = jax.device_put(global_batch, data_sharding)
    split_hlo = engine_train_script._micro_batch_splitter(micro_steps, data_sharding).lower(placed_batch).compile()
    collectives = re.findall(
        r"\b(all-gather|all-to-all|collective-permute|all-reduce|reduce-scatter)(?:-start)?\(",
        split_hlo.as_text(),
    )
    self.assertEmpty(collectives, "strided micro-batch split on data-sharded batch must emit zero collectives")

    single_cfg = pyconfig.initialize(
        _qwen3_argv("qwen3_single_micro_step", out_dir, steps=1, per_device_batch_size=2, gradient_accumulation_steps=1)
    )
    single_rows = int(single_cfg.global_batch_size_to_load)
    single_batch = {key: val[:single_rows] for key, val in global_batch.items()}
    single_loader = DataLoader(single_cfg, mesh, iter([single_batch]), goodput_recorder=None)
    unsplit = next(engine_train_script.micro_batch_stream(single_loader, 1))
    for key, arr in single_batch.items():
      np.testing.assert_array_equal(np.asarray(unsplit[key]), arr, err_msg=key)

  def test_train_main_matches_pre_train(self):
    """Runs `train.main()` on `qwen3-0.6b` and verifies parameter parity with `pre_train.train_loop`."""
    out_dir = self.create_tempdir().full_path
    steps = 2
    micro_steps = 2
    argv = _qwen3_argv(
        "qwen3_engine_parity",
        out_dir,
        steps=steps,
        gradient_accumulation_steps=micro_steps,
        enable_checkpointing=True,
        profiler="xplane",
        skip_first_n_steps_for_profiler=1,
        profiler_steps=1,
    )
    engine_train_script.main(argv)

    cfg = pyconfig.initialize(argv)
    if jax.process_index() == 0:
      self.assertTrue(os.path.isdir(cfg.tensorboard_dir))
      self.assertNotEmpty(os.listdir(cfg.tensorboard_dir))

    saved_step, script_params = _restore_saved_engine(argv)
    self.assertEqual(saved_step, steps)

    ref_cfg = pyconfig.initialize(
        _qwen3_argv("qwen3_pre_train_ref", out_dir, steps=steps, gradient_accumulation_steps=micro_steps)
    )
    ref_state = pre_train.train_loop(ref_cfg, recorder=None)
    self.assertEqual(int(ref_state.optimizer.step.get_value()), steps)
    _, ref_params_state, _ = nnx.split(ref_state.model, nnx.Param, ...)
    ref_params = _flat_params(nnx.to_pure_dict(ref_params_state))

    self.assertEqual(set(script_params), set(ref_params))
    for path_key, ref_val in ref_params.items():
      np.testing.assert_allclose(script_params[path_key], ref_val, rtol=1e-4, atol=1e-5, err_msg=path_key)

  def test_train_main_checkpoint_resume_and_single_microstep(self):
    """Verifies `steps=2 -> steps=4` checkpoint resume matches an uninterrupted `steps=4` run."""
    out_dir = self.create_tempdir().full_path
    common: dict[str, Any] = {
        "gradient_accumulation_steps": 1,
        "learning_rate_schedule_steps": 4,
        "learning_rate_final_fraction": 0.1,
        "enable_checkpointing": True,
    }

    phase1_argv = _qwen3_argv("qwen3_ckpt_resume", out_dir, steps=2, **common)
    engine_train_script.main(phase1_argv)
    step2, step2_params = _restore_saved_engine(phase1_argv)
    self.assertEqual(step2, 2)

    phase2_argv = _qwen3_argv("qwen3_ckpt_resume", out_dir, steps=4, **common)
    engine_train_script.main(phase2_argv)
    step4, step4_params = _restore_saved_engine(phase2_argv)
    self.assertEqual(step4, 4)
    self.assertTrue(any(not np.allclose(step4_params[k], step2_params[k]) for k in step2_params))

    uninterrupted_argv = _qwen3_argv("qwen3_uninterrupted", out_dir, steps=4, **common)
    engine_train_script.main(uninterrupted_argv)
    ref_step4, ref_step4_params = _restore_saved_engine(uninterrupted_argv)
    self.assertEqual(ref_step4, 4)

    self.assertEqual(set(step4_params), set(ref_step4_params))
    for path_key, ref_val in ref_step4_params.items():
      np.testing.assert_allclose(step4_params[path_key], ref_val, rtol=1e-5, atol=1e-6, err_msg=path_key)


if __name__ == "__main__":
  absltest.main()
