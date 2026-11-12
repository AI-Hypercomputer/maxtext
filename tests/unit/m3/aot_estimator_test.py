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

"""Unit tests for virtual-topology CPU AOT HBM estimator and sharding goldens."""

import gc
import os
import tracemalloc
from unittest import mock
from absl.testing import absltest

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.configs import pyconfig
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import aot_estimator
from tests.utils.test_helpers import get_test_config_path


def _cpu_mesh(
    *,
    data: int = 1,
    stage: int = 1,
    fsdp: int = 2,
    fsdp_transpose: int = 1,
    context: int = 1,
    tensor: int = 2,
    tensor_sequence: int = 1,
    expert: int = 1,
) -> Mesh:
  """Builds an 8-axis m3 `Mesh` over the host CPU devices."""
  total = data * stage * fsdp * fsdp_transpose * context * tensor * tensor_sequence * expert
  devices = np.array(jax.devices()[:total]).reshape(
      (data, stage, fsdp, fsdp_transpose, context, tensor, tensor_sequence, expert)
  )
  return Mesh(devices, m3_sharding.MESH_AXIS_NAMES)


def _init_config(model_name: str, **overrides):
  """Initializes a MaxText config for unit testing."""
  kwargs = {
      "enable_single_controller": True,
      "model_name": model_name,
      "override_model_config": True,
      "per_device_batch_size": 1.0,
      "max_target_length": 64,
      "remat_policy": "none",
      "scan_layers": False,
      "enable_checkpointing": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize(["train.py", get_test_config_path("base.yml")], **kwargs)


class AOTEstimatorUnitTest(absltest.TestCase):
  """CPU mesh and virtual-TPU AOT HBM estimator and sharding-golden tests."""

  def test_topology_hbm_capacity_lookup_and_validation(self):
    gib = 1024**3
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v5p-8"), 95 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v5p-128"), 95 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v6e-8"), 32 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v6e-256"), 32 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("tpu7x-8"), 95 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("tpu7x-128"), 95 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v5e-16"), 16 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("v4-8"), 32 * gib)
    self.assertEqual(aot_estimator.get_topology_hbm_capacity_bytes("cpu"), 95 * gib)

    with self.assertRaisesRegex(ValueError, "Unsupported topology"):
      aot_estimator.get_topology_hbm_capacity_bytes("unknown-999")

    with self.assertRaisesRegex(ValueError, "num_slices must be >= 1"):
      aot_estimator.get_virtual_topology_devices("tpu7x-8", num_slices=0)

    cfg = _init_config("qwen3-0.6b", base_num_decoder_layers=1)
    with self.assertRaisesRegex(ValueError, "topology or config.compile_topology"):
      aot_estimator.get_virtual_topology_mesh(cfg, topology="")

  def test_count_hlo_collectives_sync_async_and_custom_call(self):
    synthetic_hlo = """
HloModule jit_train_step
ENTRY main {
  ROOT %all-gather.1 = bf16[8,64,1024] all-gather(%p0), channel_id=1
  %all-gather-start.2 = (bf16[4,64,1024], bf16[8,64,1024]) all-gather-start(%p1)
  %all-gather-done.2 = bf16[8,64,1024] all-gather-done(%all-gather-start.2)
  ROOT %reduce-scatter.3 = bf16[2,64,1024] reduce-scatter(%p2), channel_id=2
  %all-reduce.4 = (f32[], f32[]) all-reduce(%p3, %p4), channel_id=3
  ROOT %ragged-all-to-all.5 = bf16[512,2,128] ragged-all-to-all(%p5, %p6, %p7, %p8, %p9, %p10)
  %custom.6 = bf16[512,128] custom-call(%p5), custom_call_target="tpu_ragged_all_to_all"
}
"""
    counts = aot_estimator.count_hlo_collectives(synthetic_hlo)
    self.assertEqual(counts["all-gather"], 2)
    self.assertEqual(counts["reduce-scatter"], 1)
    self.assertEqual(counts["all-reduce"], 1)
    self.assertEqual(counts["ragged-all-to-all"], 2)

  def test_sharding_goldens_qwen3_0_6b(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=2)
    cfg = _init_config("qwen3-0.6b", base_num_decoder_layers=2)
    report = aot_estimator.trace_abstract_sharding(cfg, mesh=mesh, include_optimizer=False)

    fsdp_ax = ("fsdp", "fsdp_transpose", "context", "expert")
    tensor_ax = ("tensor", "tensor_sequence")
    golden = {
        "token_embedder.embedding": P(tensor_ax, fsdp_ax),
        "decoder.decoder_norm.scale": P(None),
        "decoder.layers.layers.0.self_attention.query.kernel": P(fsdp_ax, tensor_ax, None),
        "decoder.layers.layers.0.self_attention.key.kernel": P(fsdp_ax, tensor_ax, None),
        "decoder.layers.layers.0.self_attention.value.kernel": P(fsdp_ax, tensor_ax, None),
        "decoder.layers.layers.0.self_attention.out.kernel": P(tensor_ax, None, fsdp_ax),
        "decoder.layers.layers.0.mlp.wi_0.kernel": P(fsdp_ax, tensor_ax),
        "decoder.layers.layers.0.mlp.wi_1.kernel": P(fsdp_ax, tensor_ax),
        "decoder.layers.layers.0.mlp.wo.kernel": P(tensor_ax, fsdp_ax),
        "expert_parallelism": aot_estimator.CapabilityStatus.NOT_APPLICABLE,
    }
    statuses = aot_estimator.verify_sharding_golden(
        report,
        golden,
        check_divisibility=True,
        check_unsharded_2d_weights=True,
    )
    self.assertEqual(
        statuses["expert_parallelism"],
        aot_estimator.CapabilityStatus.NOT_APPLICABLE,
    )
    with self.assertRaisesRegex(AssertionError, "query.kernel"):
      aot_estimator.verify_sharding_golden(
          report,
          {
              "decoder.layers.layers.0.self_attention.query.kernel": P(
                  ("fsdp", "fsdp_transpose", "expert"), tensor_ax, None
              )
          },
      )

  def test_sharding_goldens_granite_3_0_1b_a400m(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=1, expert=2)
    cfg = _init_config("granite-3.0-1b-a400m", base_num_decoder_layers=2)
    report = aot_estimator.trace_abstract_sharding(cfg, mesh=mesh, include_optimizer=False)

    fsdp_exp = ("fsdp", "expert")
    golden = {
        "embed_tokens.embedding": P(None, fsdp_exp),
        "layers.self_attn.q_proj.kernel": P(None, fsdp_exp, "tensor"),
        "layers.self_attn.k_proj.kernel": P(None, fsdp_exp, "tensor"),
        "layers.self_attn.v_proj.kernel": P(None, fsdp_exp, "tensor"),
        "layers.self_attn.o_proj.kernel": P(None, "tensor", fsdp_exp),
        "layers.block_sparse_moe.router.weight": P(None, fsdp_exp, None),
        "layers.block_sparse_moe.experts.gate_up_proj": P(None, fsdp_exp, None, None, "tensor"),
        "layers.block_sparse_moe.experts.down_proj": P(None, fsdp_exp, "tensor", None),
        "expert_parallelism": aot_estimator.CapabilityStatus.SUPPORTED,
    }
    statuses = aot_estimator.verify_sharding_golden(
        report,
        golden,
        check_divisibility=True,
        check_unsharded_2d_weights=True,
        unsharded_allowlist=("layernorm", "norm"),
    )
    self.assertEqual(
        statuses["expert_parallelism"],
        aot_estimator.CapabilityStatus.SUPPORTED,
    )

  def test_sharding_goldens_qwen3_8_flash_next_mini(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=1, expert=2)
    cfg = _init_config("qwen3.8-flash-next-mini")
    report = aot_estimator.trace_abstract_sharding(cfg, mesh=mesh, include_optimizer=False)

    fsdp_ax = ("fsdp", "fsdp_transpose", "context")
    tensor_ax = ("tensor", "tensor_sequence")
    golden = {
        "decoder.layers_0.mlp.experts.gate_up_proj": P("expert", fsdp_ax, None, tensor_ax),
        "decoder.layers_0.mlp.experts.down_proj": P("expert", tensor_ax, fsdp_ax),
        "decoder.layers_3.mlp.experts.gate_up_proj": P("expert", fsdp_ax, None, tensor_ax),
        "decoder.layers_3.mlp.experts.down_proj": P("expert", tensor_ax, fsdp_ax),
        "mtp.layers_0.mlp.experts.gate_up_proj": P("expert", fsdp_ax, None, tensor_ax),
        "mtp.layers_0.mlp.experts.down_proj": P("expert", tensor_ax, fsdp_ax),
        "expert_parallelism": aot_estimator.CapabilityStatus.SUPPORTED,
    }
    statuses = aot_estimator.verify_sharding_golden(
        report,
        golden,
        check_divisibility=True,
    )
    self.assertEqual(
        statuses["expert_parallelism"],
        aot_estimator.CapabilityStatus.SUPPORTED,
    )
    with self.assertRaisesRegex(AssertionError, "gate_up_proj"):
      aot_estimator.verify_sharding_golden(
          report,
          {"decoder.layers_0.mlp.experts.gate_up_proj": P("expert", ("fsdp", "fsdp_transpose"), None, tensor_ax)},
      )

  def test_cpu_mesh_aot_hbm_estimate_and_pinned_host_offload(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=2)
    cfg_base = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=2,
        optimizer_memory_host_offload=False,
    )
    est_base = aot_estimator.compile_aot_hbm_estimate(cfg_base, mesh=mesh)
    self.assertEqual(
        est_base.resident_peak_bytes,
        est_base.argument_bytes + est_base.output_bytes - est_base.alias_bytes + est_base.temp_bytes,
    )
    self.assertGreater(est_base.resident_peak_bytes, 0)
    self.assertGreater(est_base.temp_bytes, 0)
    self.assertGreater(est_base.device_arg_bytes, 0)
    self.assertEqual(est_base.host_arg_bytes, 0)
    self.assertTrue(est_base.fits_in_hbm)
    self.assertGreater(est_base.collective_counts["all-gather"], 0)
    self.assertGreater(est_base.collective_counts["all-reduce"], 0)
    self.assertEqual(
        est_base.capabilities["optimizer_offload"],
        aot_estimator.CapabilityStatus.NOT_APPLICABLE,
    )

    cfg_offload = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=2,
        optimizer_memory_host_offload=True,
        parameter_memory_host_offload=True,
    )
    est_offload = aot_estimator.compile_aot_hbm_estimate(cfg_offload, mesh=mesh)
    self.assertGreater(est_offload.host_arg_bytes, 0)
    self.assertLess(est_offload.device_arg_bytes, est_base.device_arg_bytes)
    self.assertEqual(
        est_offload.capabilities["optimizer_offload"],
        aot_estimator.CapabilityStatus.SUPPORTED,
    )
    self.assertEqual(
        est_offload.capabilities["parameter_offload"],
        aot_estimator.CapabilityStatus.SUPPORTED,
    )

    est_fwd = aot_estimator.compile_aot_hbm_estimate(cfg_base, mesh=mesh, include_optimizer=False)
    self.assertLess(est_fwd.argument_bytes, est_base.argument_bytes)

  def test_negative_controls_sharding_and_hbm_ceiling(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=2)
    cfg = _init_config("qwen3-0.6b", base_num_decoder_layers=1)
    report = aot_estimator.trace_abstract_sharding(cfg, mesh=mesh, include_optimizer=False)

    # Negative control 1: Perturbed golden PartitionSpec must fail.
    with self.assertRaisesRegex(AssertionError, "token_embedder.embedding"):
      aot_estimator.verify_sharding_golden(
          report,
          {"token_embedder.embedding": P(None, None)},
      )

    # Negative control 2: Intentionally replicated 2D weight must fail check_unsharded_2d_weights.
    corrupted_map = dict(report.param_shardings)
    corrupted_map["decoder.layers.layers.0.mlp.wo.kernel"] = NamedSharding(mesh, P(None, None))
    with self.assertRaisesRegex(AssertionError, "fully replicated"):
      aot_estimator.verify_sharding_golden(
          corrupted_map,
          {},
          mesh=mesh,
          param_shapes=report.param_shapes,
          check_unsharded_2d_weights=True,
      )

    # Negative control 3: Indivisible tensor dimension must fail check_divisibility.
    bad_shapes = dict(report.param_shapes)
    bad_shapes["decoder.layers.layers.0.mlp.wo.kernel"] = (3071, 1024)
    with self.assertRaisesRegex(AssertionError, "not divisible"):
      aot_estimator.verify_sharding_golden(
          report.param_shardings,
          {},
          mesh=mesh,
          param_shapes=bad_shapes,
          check_divisibility=True,
      )

    # Negative control 4: Capability laundering check (dense model has no expert_parallelism).
    with self.assertRaisesRegex(AssertionError, "expert_parallelism"):
      aot_estimator.verify_sharding_golden(
          report,
          {"expert_parallelism": aot_estimator.CapabilityStatus.SUPPORTED},
      )

    # Negative control 5: Tight HBM ceiling triggers fits_in_hbm == False.
    est_oom = aot_estimator.compile_aot_hbm_estimate(
        cfg,
        mesh=mesh,
        hbm_capacity_bytes=1024,
        include_optimizer=False,
    )
    self.assertFalse(est_oom.fits_in_hbm)

    # Negative control 6: Explicit TPU Splash attention on a CPU mesh fails closed.
    cfg_splash_cpu = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=1,
        max_target_length=128,
        attention="splash",
    )
    with self.assertRaisesRegex(ValueError, "requires a TPU mesh/device"):
      aot_estimator.trace_abstract_sharding(cfg_splash_cpu, mesh=mesh, include_optimizer=False)

  def test_capabilities_separate_static_model_axes_from_active_parallelism(self):
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh_default = _cpu_mesh(fsdp=2, tensor=2)
    cfg_default = _init_config("qwen3-0.6b", base_num_decoder_layers=1)
    rep_default = aot_estimator.trace_abstract_sharding(cfg_default, mesh=mesh_default, include_optimizer=False)
    self.assertEqual(rep_default.model_capabilities["expert_parallelism"], aot_estimator.CapabilityStatus.NOT_APPLICABLE)
    self.assertEqual(
        rep_default.model_capabilities["sequence_parallel_capable"], aot_estimator.CapabilityStatus.SUPPORTED
    )
    self.assertEqual(rep_default.active_features["sequence_parallel"], aot_estimator.CapabilityStatus.NOT_APPLICABLE)
    self.assertEqual(rep_default.active_features["context_parallelism"], aot_estimator.CapabilityStatus.NOT_APPLICABLE)
    self.assertEqual(
        rep_default.active_features["tensor_sequence_parallelism"], aot_estimator.CapabilityStatus.NOT_APPLICABLE
    )

    mesh_cp = _cpu_mesh(fsdp=2, context=2, tensor=1)
    cfg_cp = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=1,
        ici_fsdp_parallelism=2,
        ici_context_parallelism=2,
        context_parallel_load_balance=False,
    )
    rep_cp = aot_estimator.trace_abstract_sharding(cfg_cp, mesh=mesh_cp, include_optimizer=False)
    self.assertEqual(rep_cp.active_features["context_parallelism"], aot_estimator.CapabilityStatus.SUPPORTED)
    self.assertEqual(rep_cp.active_features["sequence_parallel"], aot_estimator.CapabilityStatus.SUPPORTED)
    self.assertEqual(rep_cp.active_features["tensor_sequence_parallelism"], aot_estimator.CapabilityStatus.NOT_APPLICABLE)

    mesh_tsp = _cpu_mesh(fsdp=2, tensor=1, tensor_sequence=2)
    cfg_tsp = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=1,
        ici_fsdp_parallelism=2,
        ici_tensor_sequence_parallelism=2,
    )
    rep_tsp = aot_estimator.trace_abstract_sharding(cfg_tsp, mesh=mesh_tsp, include_optimizer=False)
    self.assertEqual(rep_tsp.active_features["tensor_sequence_parallelism"], aot_estimator.CapabilityStatus.SUPPORTED)
    self.assertEqual(rep_tsp.active_features["sequence_parallel"], aot_estimator.CapabilityStatus.SUPPORTED)

  def test_virtual_tpu7x_single_and_multislice_aot_estimate(self):
    if not aot_estimator.has_libtpu_topology_support():
      self.skipTest("Requires libtpu virtual topology support.")
    gc.collect()
    baseline_cpu_ids = {
        id(arr) for arr in jax.live_arrays("cpu") if np.issubdtype(arr.dtype, np.floating) and arr.nbytes >= 1024 * 1024
    }
    cfg_8 = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=2,
        compile_topology="tpu7x-8",
        compile_topology_num_slices=1,
        ici_fsdp_parallelism=4,
        ici_tensor_parallelism=2,
        optimizer_memory_host_offload=True,
    )
    est_8 = aot_estimator.compile_aot_hbm_estimate(cfg_8)
    self.assertEqual(est_8.topology, "tpu7x-8")
    self.assertEqual(est_8.num_slices, 1)
    self.assertEqual(est_8.hbm_capacity_bytes, 95 * (1024**3))
    self.assertTrue(est_8.fits_in_hbm)
    self.assertGreater(est_8.raw_xla_peak_bytes, 0)
    self.assertGreater(est_8.host_arg_bytes, 0)
    self.assertGreater(est_8.collective_counts["all-gather"], 0)
    self.assertGreater(est_8.collective_counts["reduce-scatter"], 0)

    cfg_128 = _init_config(
        "qwen3-0.6b",
        base_num_decoder_layers=2,
        compile_topology="tpu7x-128",
        compile_topology_num_slices=1,
        ici_fsdp_parallelism=64,
        ici_tensor_parallelism=2,
        optimizer_memory_host_offload=True,
    )
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
      tracemalloc.reset_peak()
    else:
      tracemalloc.start()
    base_py_bytes = tracemalloc.get_traced_memory()[0]
    try:
      est_128 = aot_estimator.compile_aot_hbm_estimate(cfg_128)
      peak_py_delta = tracemalloc.get_traced_memory()[1] - base_py_bytes
    finally:
      if not was_tracing:
        tracemalloc.stop()
    self.assertEqual(est_128.hbm_capacity_bytes, 95 * (1024**3))
    self.assertTrue(est_128.fits_in_hbm)
    self.assertLess(est_128.device_arg_bytes * 8, est_8.device_arg_bytes)
    self.assertLess(est_128.host_arg_bytes * 8, est_8.host_arg_bytes)
    # Virtual-topology AOT traces abstract state via `nnx.eval_shape`; peak host Python heap
    # allocation stays far below the ~2.15 GiB global parameter + AdamW state across 128 chips.
    global_state_bytes = (est_128.device_arg_bytes + est_128.host_arg_bytes) * 128
    self.assertLess(peak_py_delta, global_state_bytes // 4)

    cfg_v6e_multislice = _init_config(
        "granite-3.0-1b-a400m",
        base_num_decoder_layers=2,
        compile_topology="v6e-8",
        compile_topology_num_slices=2,
        dcn_data_parallelism=2,
        ici_fsdp_parallelism=2,
        ici_tensor_parallelism=2,
        ici_expert_parallelism=2,
    )
    est_v6e = aot_estimator.compile_aot_hbm_estimate(cfg_v6e_multislice)
    self.assertEqual(est_v6e.hbm_capacity_bytes, 32 * (1024**3))
    self.assertEqual(est_v6e.num_slices, 2)
    self.assertEqual(est_v6e.mesh_shape["data"], 2)
    self.assertTrue(est_v6e.fits_in_hbm)

    cfg_qwen38 = _init_config(
        "qwen3.8-flash-next-mini",
        compile_topology="tpu7x-8",
        compile_topology_num_slices=1,
        ici_fsdp_parallelism=2,
        ici_tensor_parallelism=2,
        ici_expert_parallelism=2,
    )
    est_qwen38 = aot_estimator.compile_aot_hbm_estimate(cfg_qwen38)
    self.assertGreater(est_qwen38.collective_counts["all-gather"], 0)
    self.assertGreater(est_qwen38.collective_counts["reduce-scatter"], 0)
    self.assertGreater(est_qwen38.collective_counts["all-reduce"], 0)
    self.assertGreater(est_qwen38.collective_counts["ragged-all-to-all"], 0)

    gc.collect()
    leaked_cpu_weight_arrays = [
        arr.shape
        for arr in jax.live_arrays("cpu")
        if id(arr) not in baseline_cpu_ids and np.issubdtype(arr.dtype, np.floating) and arr.nbytes >= 1024 * 1024
    ]
    self.assertEqual(leaked_cpu_weight_arrays, [])

    # Explicit TPU Splash attention on CPU host with virtual topology="tpu7x-8" (hardware="cpu", compile_topology unset).
    cfg_splash_tpu = _init_config(
        "qwen3-0.6b",
        hardware="cpu",
        base_num_decoder_layers=1,
        max_target_length=128,
        ici_fsdp_parallelism=-1,
        ici_tensor_parallelism=2,
        attention="splash",
    )
    rep_splash = aot_estimator.trace_abstract_sharding(cfg_splash_tpu, topology="tpu7x-8", include_optimizer=False)
    self.assertEqual(rep_splash.mesh_shape["fsdp"], 4)
    est_splash = aot_estimator.compile_aot_hbm_estimate(cfg_splash_tpu, topology="tpu7x-8", include_optimizer=False)
    self.assertEqual(est_splash.topology, "tpu7x-8")
    self.assertTrue(est_splash.fits_in_hbm)

  def test_host_offload_fits_in_hbm_muon_topology_and_mapping_golden_validation(self):
    """Verifies fits_in_hbm under host offload, CPU host subtraction, opt_type=muon, topology & Mapping validation."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 host CPU devices.")
    mesh = _cpu_mesh(fsdp=2, tensor=2)

    # 1. CPU fallback subtracts host_arg_bytes from resident_peak_bytes, and fits_in_hbm ignores raw_xla_peak host DRAM.
    cfg_no_off = _init_config("qwen3-0.6b", base_num_decoder_layers=2, optimizer_memory_host_offload=False)
    cfg_opt_off = _init_config("qwen3-0.6b", base_num_decoder_layers=2, optimizer_memory_host_offload=True)
    est_no = aot_estimator.compile_aot_hbm_estimate(cfg_no_off, mesh=mesh)
    est_off = aot_estimator.compile_aot_hbm_estimate(
        cfg_opt_off,
        mesh=mesh,
        hbm_capacity_bytes=est_no.resident_peak_bytes - 1024,
    )
    self.assertGreater(est_off.host_arg_bytes, 0)
    self.assertLess(est_off.resident_peak_bytes, est_no.resident_peak_bytes)
    self.assertTrue(est_off.fits_in_hbm)

    # 2. opt_type='muon' passes model and mesh into optimizers.get_optimizer cleanly (even inside an outer set_mesh).
    cfg_muon = _init_config("qwen3-0.6b", base_num_decoder_layers=2, opt_type="muon")
    with jax.set_mesh(mesh):
      rep_muon = aot_estimator.trace_abstract_sharding(cfg_muon, mesh=mesh, include_optimizer=True)
      est_muon = aot_estimator.compile_aot_hbm_estimate(cfg_muon, mesh=mesh, include_optimizer=True)
    self.assertGreater(rep_muon.device_arg_bytes, 0)
    self.assertGreater(est_muon.resident_peak_bytes, 0)

    # 3. Case-insensitive valid topologies and fail-closed invalid topology / non-MoEExecutor backend checks.
    self.assertEqual(
        aot_estimator.get_topology_hbm_capacity_bytes("TPU7X-8"),
        aot_estimator.get_topology_hbm_capacity_bytes("tpu7x-8"),
    )
    if aot_estimator.has_libtpu_topology_support():
      self.assertLen(aot_estimator.get_virtual_topology_devices("TPU7X-8"), 8)
    bad_granite_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=granite-3.0-1b-a400m",
            "use_m3_model=true",
            "override_model_config=true",
            "base_num_decoder_layers=2",
            "use_tokamax_gmm=true",
            "enable_checkpointing=false",
            f"base_output_directory={self.create_tempdir().full_path}",
            "run_name=aot_bad_granite_tokamax",
        ]
    )
    with self.assertRaisesRegex(ValueError, "MoEExecutor"):
      aot_estimator.trace_abstract_sharding(bad_granite_cfg, mesh=_cpu_mesh(fsdp=2, tensor=2))
    for bad_topo in ("", "   ", "tpu7x:2x2x1"):
      with self.assertRaises(ValueError):
        aot_estimator.get_topology_hbm_capacity_bytes(bad_topo)
      with self.assertRaises(ValueError):
        aot_estimator.get_virtual_topology_devices(bad_topo)

    # 4. verify_sharding_golden validates Mapping capabilities, catches typos, and rejects active-feature keys.
    moe_mapping = {"decoder.layers_0.mlp.experts.gate_up_proj": P("expert", "fsdp", None, "tensor")}
    with self.assertRaisesRegex(AssertionError, "expected NOT_APPLICABLE, got SUPPORTED"):
      aot_estimator.verify_sharding_golden(
          moe_mapping,
          {"expert_parallelism": aot_estimator.CapabilityStatus.NOT_APPLICABLE},
          mesh=_cpu_mesh(fsdp=2, tensor=1, expert=2),
      )
    with self.assertRaisesRegex(AssertionError, "Unknown capability key"):
      aot_estimator.verify_sharding_golden(
          moe_mapping,
          {"expert_paralellism": aot_estimator.CapabilityStatus.NOT_APPLICABLE},
          mesh=_cpu_mesh(fsdp=2, tensor=1, expert=2),
      )
    with self.assertRaisesRegex(AssertionError, "Active-feature capability"):
      aot_estimator.verify_sharding_golden(
          moe_mapping,
          {"remat": aot_estimator.CapabilityStatus.NOT_APPLICABLE},
          mesh=_cpu_mesh(fsdp=2, tensor=1, expert=2),
      )

    # 5. has_libtpu_topology_support checks find_spec("libtpu") & JAX_PLATFORMS!="proxy" without @lru_cache memoization.
    self.assertFalse(hasattr(aot_estimator.has_libtpu_topology_support, "cache_info"))
    with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "proxy"}, clear=False):
      with mock.patch.object(aot_estimator.topologies, "get_topology_desc", side_effect=AssertionError("no PJRT init")):
        self.assertFalse(aot_estimator.has_libtpu_topology_support())
    with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=False):
      with mock.patch.object(aot_estimator.importlib.util, "find_spec", return_value=object()):
        with mock.patch.object(aot_estimator.topologies, "get_topology_desc", side_effect=AssertionError("no PJRT init")):
          self.assertTrue(aot_estimator.has_libtpu_topology_support())
      with mock.patch.object(aot_estimator.importlib.util, "find_spec", return_value=None):
        self.assertFalse(aot_estimator.has_libtpu_topology_support())


if __name__ == "__main__":
  absltest.main()
