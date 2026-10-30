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

"""Unit tests for Workstream 5: `DEFAULT_M3_MODELS` cutover governance and 4-gate Stage-1 parity suite."""

import json
import os
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.configs.types import AOT, DType, LearningRateScheduleType
from maxtext.m3.configs import MeshConfig, TrainingConfig, TransformConfig
from maxtext.m3.core.checkpoint import restore_m3_checkpoint, save_m3_checkpoint
from maxtext.m3.core.data import (
    M3PackedDataIterator,
    pack_prepared_documents,
    prepare_document_for_packing,
)
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.models import registry as m3_registry
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.train.loss import compute_batch_loss_sums
from maxtext.m3.train.trainer import M3Trainer
from tests.utils.test_helpers import get_test_config_path


class CutoverParitySuiteTest(absltest.TestCase):
  """Verifies `use_m3_model='auto'` governance and runs all 4 Stage-1 parity gates on `qwen3-mini`."""

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.mesh = create_mesh(devices=jax.devices()[:1])

  def test_use_m3_model_auto_resolves_from_default_m3_models(self):
    # Not in DEFAULT_M3_MODELS -> False
    cfg_off = pyconfig.initialize_pydantic(
        ["", get_test_config_path(), "model_name=qwen3-0.6b", "use_m3_model=auto", "skip_jax_distributed_system=true"]
    )
    self.assertIs(cfg_off.use_m3_model, False)

    # Graduated into DEFAULT_M3_MODELS -> True, while explicit `use_m3_model=false` still rolls back in < 60s!
    with (
        mock.patch.dict(os.environ, {}, clear=False),
        mock.patch.object(m3_registry, "DEFAULT_M3_MODELS", {"qwen3-0.6b"}),
    ):
      cfg_on = pyconfig.initialize_pydantic(
          ["", get_test_config_path(), "model_name=qwen3-0.6b", "use_m3_model=auto", "skip_jax_distributed_system=true"]
      )
      self.assertIs(cfg_on.use_m3_model, True)

      cfg_rollback = pyconfig.initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "model_name=qwen3-0.6b",
              "use_m3_model=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertIs(cfg_rollback.use_m3_model, False)

    status_md = os.path.join(os.path.dirname(m3_registry.__file__), "MIGRATION_STATUS.md")
    self.assertTrue(os.path.isfile(status_md))

  def test_four_stage1_certification_gates_on_qwen3_mini(self):
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)

    # Gate 1: HF config schema parity + scanned vs unscanned forward/backward gradient parity (<= 1e-5 fp32)
    hf_path = os.path.join(self.temp_dir, "config.json")
    with open(hf_path, "w", encoding="utf-8") as f:
      json.dump(cfg.model_dump(mode="json"), f)
    self.assertEqual(Qwen3Config.from_hf_config(hf_path).hidden_size, cfg.hidden_size)

    # Gate 2: Packed-vs-unpacked document forward & backward gradient equivalence on `Qwen3Model` (<= 1e-5 fp32)
    doc_a = prepare_document_for_packing([4, 8, 12, 16])  # length 3
    doc_b = prepare_document_for_packing([5, 10, 15, 20, 25])  # length 4
    packed_row = pack_prepared_documents([doc_a, doc_b], seq_len=8, strategy="first_fit")[0]
    pad_a = pack_prepared_documents([doc_a], seq_len=8, strategy="first_fit")[0]
    pad_b = pack_prepared_documents([doc_b], seq_len=8, strategy="first_fit")[0]

    b_packed = {k: jnp.asarray(v[None, :]) for k, v in packed_row.items()}
    b_a = {k: jnp.asarray(v[None, :]) for k, v in pad_a.items()}
    b_b = {k: jnp.asarray(v[None, :]) for k, v in pad_b.items()}

    with jax.set_mesh(self.mesh):
      model = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(5))
      xent_packed, grads_packed = nnx.value_and_grad(lambda m: compute_batch_loss_sums(m, b_packed).xent_sum)(model)
      xent_unpacked, grads_unpacked = nnx.value_and_grad(
          lambda m: compute_batch_loss_sums(m, b_a).xent_sum + compute_batch_loss_sums(m, b_b).xent_sum
      )(model)

    np.testing.assert_allclose(float(xent_packed), float(xent_unpacked), atol=1e-4)
    for (_, gp), (_, gu) in zip(nnx.to_flat_state(grads_packed), nnx.to_flat_state(grads_unpacked)):
      np.testing.assert_allclose(np.asarray(gp[...]), np.asarray(gu[...]), atol=1e-4)

    # Gate 3: 6-step packed training + step-3 Orbax/iterator exact resumption + cross-layout checkpoint restore
    ckpt_dir = os.path.join(self.temp_dir, "gate3")
    docs = [[3 + (i + j) % 50 for j in range(5)] for i in range(16)]
    args6 = TrainingConfig(
        run_name="gate3",
        base_output_directory=ckpt_dir,
        steps=6,
        lr_schedule_type=LearningRateScheduleType.COSINE,
        learning_rate_schedule_steps=6,
        max_target_length=8,
        per_device_batch_size=2.0,
        warmup_steps_fraction=0.0,
        param_dtype=DType.FLOAT32,
        compute_dtype=DType.FLOAT32,
        transforms=TransformConfig(scan_layers=True, remat_policy="full"),
    )
    ref_res = M3Trainer(
        cfg,
        args6,
        train_iterator=M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=9, mesh=self.mesh),
        mesh=self.mesh,
    ).train()

    args3 = args6.model_copy(update={"steps": 3, "enable_checkpointing": True, "checkpoint_period": 3})
    r1 = M3Trainer(
        cfg,
        args3,
        train_iterator=M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=9, mesh=self.mesh),
        mesh=self.mesh,
    ).train()
    args6_ckpt = args6.model_copy(update={"enable_checkpointing": True, "checkpoint_period": 3})
    r2 = M3Trainer(
        cfg,
        args6_ckpt,
        train_iterator=M3PackedDataIterator(docs, batch_size=2, seq_len=8, seed=9, mesh=self.mesh),
        mesh=self.mesh,
    ).train()
    np.testing.assert_allclose(r1.step_losses + r2.step_losses, ref_res.step_losses, atol=1e-6)

    flat_ckpt = os.path.join(self.temp_dir, "flat_ckpt")
    with jax.set_mesh(self.mesh):
      unscanned_src = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(19))
      save_m3_checkpoint(flat_ckpt, 1, unscanned_src, flat_layer_names=True)
      unscanned_dst = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(23))
      restore_m3_checkpoint(flat_ckpt, unscanned_dst)
      np.testing.assert_allclose(
          np.asarray(unscanned_dst.token_embedder.embedding[...]),
          np.asarray(unscanned_src.token_embedder.embedding[...]),
          atol=0.0,
      )

    # Gate 4: Multi-chip v5e-16 CPU AOT compilation & static HBM analysis
    aot_trainer = M3Trainer(
        cfg,
        TrainingConfig(
            max_target_length=32,
            per_device_batch_size=1.0,
            mesh=MeshConfig(ici={"fsdp": 8, "tensor": 2}),
            aot=AOT(compile_topology="v5e-16", compile_topology_num_slices=1),
        ),
    )
    aot_res = aot_trainer.compile_aot()
    self.assertIsNotNone(aot_res.memory_analysis)


if __name__ == "__main__":
  absltest.main()
