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

"""Unit tests for Workstream 3: `M3Trainer`, vocab-tiled loss, microbatch GA, `RouterBias`, and CPU `compile_aot`."""

import os
import time
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import optax

from maxtext.configs.types import AOT, DType
from maxtext.m3.configs import MeshConfig, TrainingConfig, TransformConfig
from maxtext.m3.core.data import M3PackedDataIterator
from maxtext.m3.core.interfaces import CausalLMOutput, RouterBias
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.train import __main__ as m3_train_main
from maxtext.m3.train.step import (
    accumulated_loss_and_grads,
    make_train_step,
    reshape_batch_to_microbatches,
)
from maxtext.m3.train.trainer import M3Trainer
from maxtext.trainers.pre_train import train as pretrain_train
from maxtext.trainers.pre_train import train_compile as pretrain_compile
from tests.utils.test_helpers import get_test_config_path


class TrainerAndAOTTest(absltest.TestCase):
  """End-to-end CPU verification of `M3Trainer`, loss/gradient invariants, checkpoint resume, and `compile_aot`."""

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.mesh = create_mesh(devices=jax.devices()[:1])

  def test_vocab_tiled_cross_entropy_matches_dense_forward_and_grads(self):
    for tie_emb, vocab_size in ((True, 250), (False, 250)):
      cfg = Qwen3Config.mini_config(
          vocab_size=vocab_size,
          tie_word_embeddings=tie_emb,
          compute_dtype=DType.FLOAT32,
          param_dtype=DType.FLOAT32,
      )
      batch = {
          "inputs": jnp.array([[1, 20, 130, 249], [5, 80, 190, 0]], dtype=jnp.int32),
          "targets": jnp.array([[20, 130, 249, 3], [80, 190, 240, 0]], dtype=jnp.int32),
          "positions": jnp.array([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=jnp.int32),
          "segment_ids": jnp.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32),
          "weights": jnp.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32),
      }
      with jax.set_mesh(self.mesh):
        m_dense = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(11))
        m_tiled = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(11))
        bd_dense, grads_dense = accumulated_loss_and_grads(m_dense, batch, num_vocab_tiles=1, z_loss_multiplier=1e-4)
        bd_tiled, grads_tiled = accumulated_loss_and_grads(m_tiled, batch, num_vocab_tiles=4, z_loss_multiplier=1e-4)

      np.testing.assert_allclose(float(bd_tiled.total_loss), float(bd_dense.total_loss), atol=1e-5)
      np.testing.assert_allclose(float(bd_tiled.z_loss_sum), float(bd_dense.z_loss_sum), atol=1e-5)
      for (_, g_t), (_, g_d) in zip(nnx.to_flat_state(grads_tiled), nnx.to_flat_state(grads_dense)):
        np.testing.assert_allclose(np.asarray(g_t[...]), np.asarray(g_d[...]), atol=1e-5)

  def test_microbatch_gradient_accumulation_matches_full_batch_under_asymmetric_padding(self):
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)
    # Row 0..1 dense (8 valid tokens), Row 2..3 heavily padded (only 2 valid tokens)!
    full_batch = {
        "inputs": jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 0, 0], [10, 0, 0, 0]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4, 5], [6, 7, 8, 9], [11, 0, 0, 0], [12, 0, 0, 0]], dtype=jnp.int32),
        "positions": jnp.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32),
        "segment_ids": jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32),
        "weights": jnp.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
    }
    microbatches = reshape_batch_to_microbatches(full_batch, gradient_accumulation_steps=4)

    with jax.set_mesh(self.mesh):
      m_full = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(7))
      m_ga = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(7))
      bd_full, grads_full = accumulated_loss_and_grads(m_full, full_batch, num_vocab_tiles=2)
      bd_ga, grads_ga = accumulated_loss_and_grads(m_ga, microbatches, num_vocab_tiles=2)

    np.testing.assert_allclose(float(bd_ga.total_loss), float(bd_full.total_loss), atol=1e-5)
    for (_, g_ga), (_, g_f) in zip(nnx.to_flat_state(grads_ga), nnx.to_flat_state(grads_full)):
      np.testing.assert_allclose(np.asarray(g_ga[...]), np.asarray(g_f[...]), atol=1e-5)

  def test_router_bias_updates_post_optimizer_without_affecting_ad(self):
    class _TinyMoEModel(nnx.Module):
      """Minimal MoE stub emitting `router_token_counts` on `CausalLMOutput`."""

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4, 16), dtype=jnp.float32), out_sharding=P(None, None))
        self.router_bias = RouterBias(jnp.zeros((4,), dtype=jnp.float32))

      def __call__(self, inputs, positions, **kwargs):  # pylint: disable=arguments-differ
        x = jax.nn.one_hot(inputs, 4, dtype=jnp.float32)
        logits = jnp.einsum("bsk,kv->bsv", x, self.w[...])
        counts = jnp.array([10.0, 2.0, 2.0, 2.0], dtype=jnp.float32)
        return CausalLMOutput(logits=logits, hidden_states=x, router_token_counts=counts)

    with jax.set_mesh(self.mesh):
      model = _TinyMoEModel()
      opt = nnx.Optimizer(model, optax.sgd(1e-2), wrt=nnx.Param)
      step_fn = make_train_step(router_bias_update_rate=0.05)
      batch = {
          "inputs": jnp.ones((1, 4), dtype=jnp.int32),
          "targets": jnp.ones((1, 4), dtype=jnp.int32),
          "positions": jnp.arange(4, dtype=jnp.int32)[None, :],
          "segment_ids": jnp.ones((1, 4), dtype=jnp.int32),
          "weights": jnp.ones((1, 4), dtype=jnp.float32),
      }
      _ = step_fn(model, opt, batch)
      # mean([10, 2, 2, 2]) = 4 -> sign(4 - counts) = [-1, +1, +1, +1] -> [-0.05, 0.05, 0.05, 0.05]
      np.testing.assert_allclose(
          np.asarray(model.router_bias[...]),
          np.array([-0.05, 0.05, 0.05, 0.05], dtype=np.float32),
          atol=1e-6,
      )

  def test_trainer_decreases_loss_and_avoids_per_step_graph_traversal(self):
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)
    docs = [[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(16)]

    def _run(num_steps: int):
      it = M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=0, mesh=self.mesh)
      train_cfg = TrainingConfig(
          steps=num_steps,
          max_target_length=8,
          per_device_batch_size=2.0,
          learning_rate=5e-3,
          warmup_steps_fraction=0.0,
          param_dtype=DType.FLOAT32,
          compute_dtype=DType.FLOAT32,
          transforms=TransformConfig(scan_layers=True, remat_policy="full"),
      )
      trainer = M3Trainer(model=cfg, args=train_cfg, train_iterator=it, mesh=self.mesh)
      return trainer.train()

    with (
        mock.patch.object(nnx, "state", wraps=nnx.state) as state_spy,
        mock.patch.object(nnx.graphlib, "flatten", wraps=nnx.graphlib.flatten) as flatten_spy,
    ):
      _ = _run(2)
      s2, f2 = state_spy.call_count, flatten_spy.call_count
      state_spy.reset_mock()
      flatten_spy.reset_mock()
      res6 = _run(6)
      s6, f6 = state_spy.call_count, flatten_spy.call_count

    self.assertEqual(s2, s6)
    self.assertEqual(f2, f6)
    self.assertLess(res6.step_losses[-1], res6.step_losses[0])

  def test_trainer_checkpoint_mid_epoch_resume_matches_uninterrupted_run(self):
    ckpt_base = os.path.join(self.temp_dir, "resume_run")
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)
    docs = [[(i + j) % 60 + 2 for j in range(6)] for i in range(20)]

    def _make_args(steps: int, ckpt: bool):
      return TrainingConfig(
          run_name="exp1",
          base_output_directory=ckpt_base,
          steps=steps,
          max_target_length=8,
          per_device_batch_size=2.0,
          learning_rate=1e-3,
          warmup_steps_fraction=0.0,
          param_dtype=DType.FLOAT32,
          compute_dtype=DType.FLOAT32,
          enable_checkpointing=ckpt,
          checkpoint_period=2,
          transforms=TransformConfig(scan_layers=True, remat_policy="none"),
      )

    # 1. Uninterrupted 4-step baseline
    it_ref = M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=123, mesh=self.mesh)
    trainer_ref = M3Trainer(model=cfg, args=_make_args(4, False), train_iterator=it_ref, mesh=self.mesh)
    res_ref = trainer_ref.train()

    # 2. Run 2 steps with checkpointing, then resume to step 4 in a fresh trainer
    it_part1 = M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=123, mesh=self.mesh)
    trainer_part1 = M3Trainer(model=cfg, args=_make_args(2, True), train_iterator=it_part1, mesh=self.mesh)
    res_part1 = trainer_part1.train()

    it_part2 = M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=123, mesh=self.mesh)
    trainer_part2 = M3Trainer(model=cfg, args=_make_args(4, True), train_iterator=it_part2, mesh=self.mesh)
    res_part2 = trainer_part2.train()

    combined_losses = res_part1.step_losses + res_part2.step_losses
    np.testing.assert_allclose(combined_losses, res_ref.step_losses, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(trainer_part2.model.decoder.layers.mlp.wi_0.kernel[...]),
        np.asarray(trainer_ref.model.decoder.layers.mlp.wi_0.kernel[...]),
        atol=1e-6,
    )

  def test_offline_cpu_compile_aot_on_virtual_v5e_16_topology(self):
    aot_path = os.path.join(self.temp_dir, "compiled_step.bin")
    cfg = Qwen3Config.mini_config()
    train_cfg = TrainingConfig(
        steps=10,
        max_target_length=32,
        per_device_batch_size=1.0,
        gradient_accumulation_steps=2,
        num_vocab_tiles=2,
        mesh=MeshConfig(ici={"fsdp": 8, "tensor": 2}, dcn={"data": 1}),
        aot=AOT(compile_topology="v5e-16", compile_topology_num_slices=1, compiled_trainstep_file=aot_path),
    )
    t0 = time.perf_counter()
    trainer = M3Trainer(model=cfg, args=train_cfg)
    # Zero model weights allocated on host before or after `compile_aot`!
    self.assertFalse(trainer.is_initialized)
    result = trainer.compile_aot()
    elapsed = time.perf_counter() - t0

    self.assertFalse(trainer.is_initialized)
    self.assertLess(elapsed, 90.0)
    self.assertEqual(result.mesh.size, 16)
    self.assertIsNotNone(result.memory_analysis)
    self.assertTrue(os.path.isfile(aot_path))
    self.assertGreater(os.path.getsize(aot_path), 0)

  def test_cli_entrypoints_dispatch_cleanly(self):
    # 1. Native YAML-free CLI `python -m maxtext.m3.train`
    m3_train_main.main(
        [
            "--preset",
            "qwen3-mini",
            "--steps",
            "2",
            "--max_target_length",
            "8",
            "--per_device_batch_size",
            "1",
            "--param_dtype",
            "float32",
            "--compute_dtype",
            "float32",
        ]
    )

    # 2. Existing `train.py` and `train_compile.py` 3-line `use_m3_model=true` dispatchers
    common_args = [
        "",
        get_test_config_path(),
        "model_name=qwen3-0.6b",
        "use_m3_model=true",
        "override_model_config=true",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_decoder_layers=2",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
        "per_device_batch_size=1",
        "max_target_length=8",
        "steps=2",
        "enable_checkpointing=false",
        f"base_output_directory={self.temp_dir}",
        "run_name=dispatch_smoke",
        "skip_jax_distributed_system=true",
    ]
    pretrain_train.main(common_args)
    pretrain_compile.main(common_args + ["compile_topology=v5e-16", "compile_topology_num_slices=1"])


if __name__ == "__main__":
  absltest.main()
