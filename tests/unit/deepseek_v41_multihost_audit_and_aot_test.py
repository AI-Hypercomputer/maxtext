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

"""Unit tests and 2-process CPU multi-host verification for DeepSeek-V4.1 audit, frozen top-k, and AOT checks."""

import collections
import itertools
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from typing import Any
import unittest
from unittest import mock

_REPO_ROOT_INIT = str(Path(__file__).resolve().parents[2])

if __name__ == "__main__":
  if "--worker_port" not in sys.argv and "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=4".strip()
  _repo_root_for_main = _REPO_ROOT_INIT
  _src_dir_for_main = str(Path(__file__).resolve().parents[2] / "src")
  if _repo_root_for_main not in sys.path:
    sys.path.insert(0, _repo_root_for_main)
  if _src_dir_for_main not in sys.path:
    sys.path.insert(1, _src_dir_for_main)

# pylint: disable=wrong-import-position
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_deepseek41
from maxtext.layers import attentions
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.models import deepseek4
from maxtext.models import deepseek41
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
import numpy as np
import pytest
import transformers

from tests.utils import deepseek_v41_frozen_topk
from tests.utils import deepseek_v41_mm_audit
from tests.utils import forward_pass_logit_checker
from tests.utils.deepseek_v41 import logit_runner
from tests.utils import linen_wrappers
from tests.utils.test_helpers import get_test_config_path

# pylint: disable=possibly-used-before-assignment
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
  from safetensors.torch import save_file  # pylint: disable=g-import-not-at-top
  import torch  # pylint: disable=g-import-not-at-top
  from tests.utils import deepseek_v41_mm_testing as mm_testing  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top

# pylint: enable=wrong-import-position

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FP32_GATE_ATOL = 1.0e-6
_BF16_GATE_TRUNCATION_MIN_DIFF = 1.0e-4


def _make_dummy_decoder_and_config(*, scan_layers: bool) -> tuple[types.SimpleNamespace, types.SimpleNamespace]:
  """Builds a lightweight 2-layer decoder stub and config supporting both `scan_layers=True` and `scan_layers=False`."""
  cfg = types.SimpleNamespace(
      num_decoder_layers=2,
      num_experts_per_tok=2,
      num_experts=8,
      indexer_topk=3,
      compress_ratios=[1, 1],
      ds41_kv_source_layers=[0, 1],
      ds41_index_source_layers=[0, 1],
      ds41_candidate_source_layer=-1,
      ds41_engram_layers=[],
      scan_layers=bool(scan_layers),
  )
  decoder = types.SimpleNamespace()
  if scan_layers:
    sub_layer = types.SimpleNamespace(
        mlp=types.SimpleNamespace(MoeBlock_0=types.SimpleNamespace()),
        self_attention=types.SimpleNamespace(indexer=types.SimpleNamespace()),
    )
    decoder.segments_0 = types.SimpleNamespace(layers_0=sub_layer)
  else:
    for layer_idx in range(2):
      sub_layer = types.SimpleNamespace(
          mlp=types.SimpleNamespace(MoeBlock_0=types.SimpleNamespace()),
          self_attention=types.SimpleNamespace(indexer=types.SimpleNamespace()),
      )
      setattr(decoder, f"layers_{layer_idx}", sub_layer)
  model = types.SimpleNamespace(decoder=decoder, mesh=None)
  return model, cfg


def _run_two_process_worker(port: int, rank: int) -> None:
  """Executes the 2-process CPU `jax.distributed.initialize` RED -> GREEN check on `rank`."""
  jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=2, process_id=rank)
  assert jax.process_count() == 2, f"Expected 2 JAX processes, got {jax.process_count()}"
  assert jax.local_device_count() == 2, f"Expected 2 local CPU devices, got {jax.local_device_count()}"

  mesh = jax.sharding.Mesh(jax.devices(), ("data",))
  replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  batch_size = 4
  local_batch = 2
  padded_seq_len = 6
  top_k = 2
  index_topk = 3

  # Expected full arrays across all 4 devices (2 processes x 2 local devices).
  full_unscanned_moe_l0 = np.arange(batch_size * padded_seq_len * top_k, dtype=np.int32).reshape(
      batch_size, padded_seq_len, top_k
  )
  full_unscanned_moe_l1 = full_unscanned_moe_l0 + 100
  full_unscanned_idx_l0 = np.arange(batch_size * padded_seq_len * index_topk, dtype=np.int32).reshape(
      batch_size, padded_seq_len, index_topk
  )
  full_unscanned_idx_l1 = full_unscanned_idx_l0 + 200

  b_slice = slice(rank * local_batch, (rank + 1) * local_batch)

  sharded_moe_l0 = multihost_utils.host_local_array_to_global_array(
      full_unscanned_moe_l0[b_slice], mesh, jax.sharding.PartitionSpec("data", None, None)
  )
  sharded_moe_l1 = multihost_utils.host_local_array_to_global_array(
      full_unscanned_moe_l1[b_slice], mesh, jax.sharding.PartitionSpec("data", None, None)
  )
  sharded_idx_l0 = multihost_utils.host_local_array_to_global_array(
      full_unscanned_idx_l0[b_slice], mesh, jax.sharding.PartitionSpec("data", None, None)
  )
  sharded_idx_l1 = multihost_utils.host_local_array_to_global_array(
      full_unscanned_idx_l1[b_slice], mesh, jax.sharding.PartitionSpec("data", None, None)
  )
  assert not sharded_moe_l0.is_fully_addressable
  assert not sharded_idx_l0.is_fully_addressable

  # 1. Prove RED: direct np.asarray(var[...], dtype=np.int32) raises RuntimeError on non-addressable jax.Array.
  red_var = nnx.Variable(sharded_moe_l0)
  raised_red = False
  try:
    _ = np.asarray(red_var[...], dtype=np.int32)
  except RuntimeError as exc:
    assert "spans non-addressable" in str(exc), f"Unexpected RuntimeError message: {exc}"
    raised_red = True
  assert raised_red, "Expected direct np.asarray(red_var[...]) to raise RuntimeError on non-addressable jax.Array"

  # 2. Prove GREEN on _safe_gather_and_numpy directly.
  gathered_direct = deepseek_v41_frozen_topk._safe_gather_and_numpy(red_var, dtype=np.int32)  # pylint: disable=protected-access
  np.testing.assert_array_equal(gathered_direct, full_unscanned_moe_l0)

  # 3. Prove GREEN on natural_routing_trace_context (scan_layers=False).
  model_unscanned, cfg_unscanned = _make_dummy_decoder_and_config(scan_layers=False)
  model_unscanned.mesh = mesh
  with deepseek_v41_frozen_topk.natural_routing_trace_context(
      model_unscanned,
      cfg_unscanned,
      sample_name="M1_unscanned",
      batch_size=batch_size,
      padded_seq_len=padded_seq_len,
      mesh=mesh,
  ) as (extract_unscanned, moe_ids_unscanned, idx_ids_unscanned):
    assert moe_ids_unscanned == [0, 1]
    assert idx_ids_unscanned == [0, 1]
    moe0_var = getattr(model_unscanned.decoder.layers_0.mlp.MoeBlock_0, "_captured_topk_indices")
    idx0_var = getattr(model_unscanned.decoder.layers_0.self_attention.indexer, "_captured_indexer_topk")
    assert moe0_var[...].sharding == replicated_sharding
    assert idx0_var[...].sharding == replicated_sharding

    # Overwrite with sharded non-addressable arrays as produced by JIT trace step.
    getattr(model_unscanned.decoder.layers_0.mlp.MoeBlock_0, "_captured_topk_indices").set_value(sharded_moe_l0)
    getattr(model_unscanned.decoder.layers_1.mlp.MoeBlock_0, "_captured_topk_indices").set_value(sharded_moe_l1)
    getattr(model_unscanned.decoder.layers_0.self_attention.indexer, "_captured_indexer_topk").set_value(sharded_idx_l0)
    getattr(model_unscanned.decoder.layers_1.self_attention.indexer, "_captured_indexer_topk").set_value(sharded_idx_l1)

    moe_out_unscanned, idx_out_unscanned = extract_unscanned()
    np.testing.assert_array_equal(moe_out_unscanned[0], full_unscanned_moe_l0)
    np.testing.assert_array_equal(moe_out_unscanned[1], full_unscanned_moe_l1)
    np.testing.assert_array_equal(idx_out_unscanned[0], full_unscanned_idx_l0)
    np.testing.assert_array_equal(idx_out_unscanned[1], full_unscanned_idx_l1)

  # 4. Prove GREEN on natural_routing_trace_context (scan_layers=True).
  full_scanned_moe = np.stack([full_unscanned_moe_l0, full_unscanned_moe_l1], axis=0)
  full_scanned_idx = np.stack([full_unscanned_idx_l0, full_unscanned_idx_l1], axis=0)
  sharded_scanned_moe = multihost_utils.host_local_array_to_global_array(
      full_scanned_moe[:, b_slice, :, :], mesh, jax.sharding.PartitionSpec(None, "data", None, None)
  )
  sharded_scanned_idx = multihost_utils.host_local_array_to_global_array(
      full_scanned_idx[:, b_slice, :, :], mesh, jax.sharding.PartitionSpec(None, "data", None, None)
  )
  assert not sharded_scanned_moe.is_fully_addressable
  assert not sharded_scanned_idx.is_fully_addressable

  model_scanned, cfg_scanned = _make_dummy_decoder_and_config(scan_layers=True)
  model_scanned.mesh = mesh
  with deepseek_v41_frozen_topk.natural_routing_trace_context(
      model_scanned,
      cfg_scanned,
      sample_name="M1_scanned",
      batch_size=batch_size,
      padded_seq_len=padded_seq_len,
      mesh=mesh,
  ) as (extract_scanned, moe_ids_scanned, idx_ids_scanned):
    assert moe_ids_scanned == [0, 1]
    assert idx_ids_scanned == [0, 1]
    scanned_moe_block = model_scanned.decoder.segments_0.layers_0.mlp.MoeBlock_0
    scanned_idx_mod = model_scanned.decoder.segments_0.layers_0.self_attention.indexer
    assert getattr(scanned_moe_block, "_captured_topk_indices")[...].sharding == replicated_sharding
    assert getattr(scanned_idx_mod, "_captured_indexer_topk")[...].sharding == replicated_sharding

    getattr(scanned_moe_block, "_captured_topk_indices").set_value(sharded_scanned_moe)
    getattr(scanned_idx_mod, "_captured_indexer_topk").set_value(sharded_scanned_idx)

    moe_out_scanned, idx_out_scanned = extract_scanned()
    np.testing.assert_array_equal(moe_out_scanned, full_scanned_moe)
    np.testing.assert_array_equal(idx_out_scanned[0], full_unscanned_idx_l0)
    np.testing.assert_array_equal(idx_out_scanned[1], full_unscanned_idx_l1)

  # 5. Verify frozen_topk_routing_context places _frozen_topk_indices and _frozen_indexer_topk on replicated sharding.
  raw_experts = np.zeros((2, 4, top_k), dtype=np.int32)
  raw_experts[:, :, 0] = 1
  raw_experts[:, :, 1] = 3
  raw_indexer_topk = np.zeros((2, 4, index_topk), dtype=np.int32)
  raw_indexer_topk[:, :, 0] = 4
  raw_indexer_topk[:, :, 1] = 5
  raw_indexer_topk[:, :, 2] = 6
  raw_indexer_layer_ids = np.array([0, 1], dtype=np.int32)

  for scan_flag, model_obj, cfg_obj in (
      (False, model_unscanned, cfg_unscanned),
      (True, model_scanned, cfg_scanned),
  ):
    with deepseek_v41_frozen_topk.frozen_topk_routing_context(
        model_obj,
        cfg_obj,
        raw_experts,
        sample_name=f"M1_frozen_{scan_flag}",
        seq_len=4,
        batch_size=batch_size,
        padded_seq_len=padded_seq_len,
        raw_indexer_topk=raw_indexer_topk,
        raw_indexer_layer_ids=raw_indexer_layer_ids,
        mesh=mesh,
    ) as num_patched:
      assert num_patched == 2
      if scan_flag:
        m_var = getattr(model_obj.decoder.segments_0.layers_0.mlp.MoeBlock_0, "_frozen_topk_indices")
        i_var = getattr(model_obj.decoder.segments_0.layers_0.self_attention.indexer, "_frozen_indexer_topk")
      else:
        m_var = getattr(model_obj.decoder.layers_0.mlp.MoeBlock_0, "_frozen_topk_indices")
        i_var = getattr(model_obj.decoder.layers_0.self_attention.indexer, "_frozen_indexer_topk")
      assert m_var[...].sharding == replicated_sharding
      assert i_var[...].sharding == replicated_sharding

  print(f"WORKER_RANK_{rank}_PASSED", flush=True)


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeekV41MultihostAuditAndAotTest(unittest.TestCase):
  """Verifies multi-host FrozenTopK process_allgather, unscanned audit coverage, and AOT HBM helpers."""

  def setUp(self) -> None:
    super().setUp()
    self._orig_matmul_precision = jax.config.jax_default_matmul_precision

  def tearDown(self) -> None:
    jax.config.update("jax_default_matmul_precision", self._orig_matmul_precision)
    super().tearDown()

  def test_two_process_cpu_non_addressable_extract_captured_red_to_green(self):
    """Spawns 2 OS processes with `jax.distributed.initialize` to prove RED -> GREEN on non-addressable arrays."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind(("127.0.0.1", 0))
      port = sock.getsockname()[1]

    env = os.environ.copy()
    existing_xla_flags = env.get("XLA_FLAGS", "")
    cleaned_flags = [
        tok for tok in existing_xla_flags.split() if not tok.startswith("--xla_force_host_platform_device_count")
    ]
    cleaned_flags.append("--xla_force_host_platform_device_count=2")
    env["XLA_FLAGS"] = " ".join(cleaned_flags)
    env["JAX_PLATFORMS"] = "cpu"
    repo_root = str(Path(__file__).resolve().parents[2])
    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}:{src_dir}:{existing_pythonpath}" if existing_pythonpath else f"{repo_root}:{src_dir}"
    )

    cmd0 = [sys.executable, __file__, "--worker_port", str(port), "--worker_rank", "0"]
    cmd1 = [sys.executable, __file__, "--worker_port", str(port), "--worker_rank", "1"]

    with (
        subprocess.Popen(cmd0, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc0,
        subprocess.Popen(cmd1, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc1,
    ):
      out0, _ = proc0.communicate(timeout=90)
      out1, _ = proc1.communicate(timeout=90)
      rc0 = proc0.returncode
      rc1 = proc1.returncode

    self.assertEqual(rc0, 0, msg=f"Rank 0 failed (rc={rc0}):\n{out0}")
    self.assertEqual(rc1, 0, msg=f"Rank 1 failed (rc={rc1}):\n{out1}")
    self.assertIn("WORKER_RANK_0_PASSED", out0)
    self.assertIn("WORKER_RANK_1_PASSED", out1)

  def test_safe_gather_and_numpy_triggers_process_allgather_when_not_fully_addressable(self):
    """Unit test and negative control for `_safe_gather_and_numpy` with a mocked non-addressable `jax.Array`."""
    expected = np.array([[[1, 2], [3, 4]]], dtype=np.int32)

    class _FakeNonAddressableJaxArray(jax.Array):
      pass

    fake_arr = mock.MagicMock(spec=_FakeNonAddressableJaxArray)
    fake_arr.is_fully_addressable = False
    fake_var = types.SimpleNamespace(value=fake_arr)

    with mock.patch.object(
        deepseek_v41_frozen_topk.multihost_utils,
        "process_allgather",
        return_value=expected,
    ) as mock_allgather:
      got = deepseek_v41_frozen_topk._safe_gather_and_numpy(fake_var, dtype=np.int32)  # pylint: disable=protected-access
      mock_allgather.assert_called_once_with(fake_arr, tiled=True)
      np.testing.assert_array_equal(got, expected)

    # Addressable JAX array must NOT invoke process_allgather.
    addressable_var = nnx.Variable(jnp.asarray(expected, dtype=jnp.int32))
    with mock.patch.object(
        deepseek_v41_frozen_topk.multihost_utils,
        "process_allgather",
    ) as mock_allgather_addr:
      got_addr = deepseek_v41_frozen_topk._safe_gather_and_numpy(addressable_var, dtype=np.int32)  # pylint: disable=protected-access
      mock_allgather_addr.assert_not_called()
      np.testing.assert_array_equal(got_addr, expected)

  def test_logit_runner_always_enforces_verify_all_text_leaves_true(self) -> None:
    """Verifies `logit_runner.main` always passes `verify_all_text_leaves=True` (scanned & unscanned)."""
    fake_mesh = mock.MagicMock(spec=jax.sharding.Mesh)
    fake_model = types.SimpleNamespace(decoder=types.SimpleNamespace(), mesh=fake_mesh)

    for scan_layers_flag in (True, False):
      cfg = types.SimpleNamespace(
          init_weights_seed=0,
          mesh_axes=("data",),
          hf_access_token="",
          tokenizer_path="",
          model_name="deepseek4.1-flash",
          load_parameters_path="/tmp/fake_hf_ckpt",
          scan_layers=scan_layers_flag,
          use_multimodal=False,
      )
      test_args = types.SimpleNamespace(
          hf_model_path="",
          trust_remote_code=True,
          run_ds41_mm_audit=True,
          hf_checkpoint_dir_for_audit="/tmp/fake_hf_ckpt",
          export_orbax_dir="",
          mm_golden_logits_path="",
          thresholds_json="",
          max_images_per_sample=4,
          summary_json="",
          enable_long_text_gate=False,
          require_frozen_topk_gate=False,
          golden_logits_path="/tmp/fake_golden.pkl",
          output_logits_path="",
          run_hf_model=False,
          token_size=None,
          skip_first_token=False,
          atol=None,
          max_kl_div=None,
      )
      with (
          mock.patch.object(
              logit_runner.maxtext_utils,
              "create_device_mesh",
              return_value=np.empty((1,), dtype=object),
          ),
          mock.patch.object(
              logit_runner.jax.sharding,
              "Mesh",
              return_value=fake_mesh,
          ),
          mock.patch.object(
              deepseek_v41_mm_audit,
              "audit_loaded_model_against_hf",
              return_value={"passed": True, "scan_layers": scan_layers_flag},
          ) as mock_audit,
          mock.patch.object(
              logit_runner,
              "load_golden_entries",
              return_value=[],
          ),
      ):
        summary = logit_runner.main(cfg, test_args, preloaded_model=fake_model)
        self.assertTrue(summary["passed"])
        mock_audit.assert_called_once_with(
            model=fake_model,
            config=cfg,
            hf_checkpoint_dir="/tmp/fake_hf_ckpt",
            mesh=fake_mesh,
            verify_all_text_leaves=True,
            export_orbax_dir=None,
        )

    # Negative control: failing audit raises DeterministicGateError.
    cfg_fail = types.SimpleNamespace(
        init_weights_seed=0,
        mesh_axes=("data",),
        hf_access_token="",
        tokenizer_path="",
        model_name="deepseek4.1-flash",
        load_parameters_path="/tmp/fake_hf_ckpt",
        scan_layers=False,
        use_multimodal=False,
    )
    with (
        mock.patch.object(
            logit_runner.maxtext_utils,
            "create_device_mesh",
            return_value=np.empty((1,), dtype=object),
        ),
        mock.patch.object(
            logit_runner.jax.sharding,
            "Mesh",
            return_value=fake_mesh,
        ),
        mock.patch.object(
            deepseek_v41_mm_audit,
            "audit_loaded_model_against_hf",
            return_value={"passed": False, "failures": ["layer_0 mismatch"]},
        ),
    ):
      with self.assertRaisesRegex(logit_runner.DeterministicGateError, "HF-to-MaxText pre-forward audit failed"):
        logit_runner.main(cfg_fail, test_args, preloaded_model=fake_model)

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference/safetensors.torch")
  def test_unscanned_audit_parallel_single_pass_speedup_on_mini_checkpoint(self) -> None:
    """Proves unscanned single-pass reads and parallel audit speedup on mini checkpoint."""
    jax.config.update("jax_default_matmul_precision", "highest")
    tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
    ref_args = multimodal.mini_model_args()
    ref_mm = multimodal.build_multimodal_model(ref_args, tokenizer=tokenizer)
    multimodal.init_mini_reference(ref_mm, seed=20261030)

    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in ref_mm.named_parameters():
      t_cpu = tensor.detach().cpu().contiguous()
      if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
        tensors[name] = t_cpu.to(torch.float32).contiguous()
      else:
        tensors[name] = t_cpu.to(torch.bfloat16).contiguous()
    for name, tensor in ref_mm.named_buffers():
      if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
        tensors[name] = tensor.detach().cpu().to(torch.float32).contiguous()
    tensors["mtp.0.norm.weight"] = torch.ones((16,), dtype=torch.bfloat16)
    tensors["mtp.0.ffn.gate.bias_vl"] = torch.zeros((4,), dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix="ds41_unscanned_speedup_") as tmp_dir:
      hf_dir = Path(tmp_dir) / "mini_hf"
      hf_dir.mkdir(parents=True, exist_ok=True)
      sorted_keys = sorted(tensors.keys())
      shard_tensors: dict[str, dict[str, torch.Tensor]] = {
          "model-00001-of-00002.safetensors": {},
          "model-00002-of-00002.safetensors": {},
      }
      weight_map: dict[str, str] = {}
      for idx, key in enumerate(sorted_keys):
        shard_name = "model-00001-of-00002.safetensors" if idx % 2 == 0 else "model-00002-of-00002.safetensors"
        shard_tensors[shard_name][key] = tensors[key]
        weight_map[key] = shard_name
      for shard_name, sub_dict in shard_tensors.items():
        save_file(sub_dict, (hf_dir / shard_name).as_posix(), metadata={"format": "pt"})
      with open(hf_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)
      with open(hf_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model_type": "deepseek_v41"}, f)

      cfg = mm_testing.mini_vision_maxtext_config(
          scan_layers=False,
          param_scan_axis=0,
          dtype="float32",
          weight_dtype="bfloat16",
          dtype_mm="float32",
          matmul_precision="highest",
          remat_policy="none",
          ds41_activation_quant=False,
          per_device_batch_size=1.0,
          ici_fsdp_parallelism=2,
          ici_tensor_parallelism=2,
          ici_expert_parallelism=1,
          max_target_length=multimodal.MINI_SEQ_LEN,
      )
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = model_creation_utils.from_config(
          cfg,
          mesh=mesh,
          rngs=nnx.Rngs(params=0),
          model_mode=MODEL_MODE_TRAIN,
      )
      mm_testing.copy_full_multimodal_model(model, ref_mm, cfg, jnp.bfloat16)
      _, abs_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh=mesh)
      _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
      _, abs_p_state, abs_b_state, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)
      p_sharded = jax.tree.map(lambda a, av: jax.device_put(a, av.sharding), p_state, abs_p_state)
      b_sharded = jax.tree.map(lambda a, av: jax.device_put(a, av.sharding), b_state, abs_b_state)
      nnx.update(model, p_sharded, b_sharded, rest_state)

      raw_counts: collections.Counter[str] = collections.Counter()
      count_lock = threading.Lock()
      orig_load_raw = deepseek_v41_mm_audit._SafetensorsShardReader.load_raw_tensor  # pylint: disable=protected-access
      orig_load_slice = (
          deepseek_v41_mm_audit._SafetensorsShardReader.load_raw_row_slice  # pylint: disable=protected-access
      )

      def _latency_load_raw(reader_self: Any, key: str) -> torch.Tensor:
        time.sleep(0.020)
        out = orig_load_raw(reader_self, key)
        with count_lock:
          raw_counts[key] += 1
        return out

      def _latency_load_slice(reader_self: Any, key: str, row_start: int, row_stop: int) -> torch.Tensor:
        time.sleep(0.020)
        out = orig_load_slice(reader_self, key, row_start, row_stop)
        with count_lock:
          raw_counts[f"{key}[{row_start}:{row_stop}]"] += 1
        return out

      with (
          mock.patch.object(
              deepseek_v41_mm_audit._SafetensorsShardReader,  # pylint: disable=protected-access
              "load_raw_tensor",
              _latency_load_raw,
          ),
          mock.patch.object(
              deepseek_v41_mm_audit._SafetensorsShardReader,  # pylint: disable=protected-access
              "load_raw_row_slice",
              _latency_load_slice,
          ),
      ):
        t0_par = time.perf_counter()
        res_par = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
            model=model,
            config=cfg,
            hf_checkpoint_dir=hf_dir,
            mesh=mesh,
            verify_all_text_leaves=True,
        )
        par_sec = time.perf_counter() - t0_par
        par_counts = dict(raw_counts)

        raw_counts.clear()
        with mock.patch.object(deepseek_v41_mm_audit, "_AUDIT_IO_MAX_WORKERS", 1):
          t0_seq = time.perf_counter()
          res_seq = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
              model=model,
              config=cfg,
              hf_checkpoint_dir=hf_dir,
              mesh=mesh,
              verify_all_text_leaves=True,
          )
          seq_sec = time.perf_counter() - t0_seq

      self.assertTrue(res_par["passed"], msg=f"Parallel unscanned audit failed: {res_par}")
      self.assertEqual(res_par, res_seq)
      self.assertEqual(res_par["local_checked_leaves"], 418)
      self.assertEqual(res_par["total_checked_shards"], 962)
      self.assertEqual(res_par["total_bit_mismatches"], 0)
      self.assertEqual(max(par_counts.values()), 1)
      for key, count in par_counts.items():
        self.assertEqual(count, 1, msg=f"HF key {key} was read {count} times across 4 shards (expected 1)")
      self.assertNotIn("layers.1.engram.embed.weight", par_counts)
      speedup = seq_sec / max(par_sec, 1e-6)
      print(f"UNSCANNED_MINI_AUDIT_TIMING: seq_sec={seq_sec:.3f}s par_sec={par_sec:.3f}s speedup={speedup:.3f}x")
      self.assertGreater(
          speedup,
          1.5,
          msg=f"Expected >1.5x speedup from parallel unscanned audit, got seq={seq_sec:.3f}s vs par={par_sec:.3f}s",
      )

  def test_ref_indexer_topk_coordinate_offset_normalization_red_to_green(self) -> None:
    """Proves RED without seq_len offset normalization and GREEN with normalization across FrozenTopK and agreement."""
    seq_len = 8
    padded_seq_len = 8
    batch_size = 2
    num_compressed = 4
    index_topk = 3
    layer_ids = np.array([0, 1], dtype=np.int32)

    # 0-based TPU compressed KV indices in [0, num_compressed), with -1 where compress_lens < index_topk.
    tpu_idx_1b = np.array(
        [
            [0, -1, -1],
            [0, -1, -1],
            [0, 1, -1],
            [0, 1, -1],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    tpu_idx_l0 = np.broadcast_to(tpu_idx_1b[None, :, :], (batch_size, padded_seq_len, index_topk)).copy()
    tpu_idx_l1 = np.broadcast_to(tpu_idx_1b[None, :, :], (batch_size, padded_seq_len, index_topk)).copy()
    tpu_indexer_map = {0: tpu_idx_l0, 1: tpu_idx_l1}

    # PyTorch reference stores valid compressed KV indices with +seq_len offset in [seq_len, seq_len + num_compressed).
    ref_idx_offset_1b = np.where(tpu_idx_1b >= 0, tpu_idx_1b + seq_len, -1).astype(np.int32)
    ref_indexer_topk_offset = np.stack([ref_idx_offset_1b, ref_idx_offset_1b], axis=0)
    self.assertTrue(bool(np.all(ref_indexer_topk_offset[ref_indexer_topk_offset >= 0] >= seq_len)))

    ref_moe = np.zeros((2, seq_len, 2), dtype=np.uint16)
    ref_moe[:, :, 0] = 1
    ref_moe[:, :, 1] = 3
    tpu_moe = np.broadcast_to(ref_moe[:, None, :, :].astype(np.int32), (2, batch_size, padded_seq_len, 2)).copy()
    ref_gate_margins = np.full((2, seq_len), 20.0 * 1.0e-6, dtype=np.float32)
    ref_gate_ulp = np.full((2, seq_len), 1.0e-6, dtype=np.float32)
    ref_idx_margins = np.full((2, seq_len), np.nan, dtype=np.float32)
    ref_idx_margins[:, 6:] = 20.0 * 1.0e-6
    ref_idx_ulp = np.full((2, seq_len), np.nan, dtype=np.float32)
    ref_idx_ulp[:, 6:] = 1.0e-6

    model_stub, cfg_stub = _make_dummy_decoder_and_config(scan_layers=False)
    idx_mod = model_stub.decoder.layers_0.self_attention.indexer
    idx_mod.index_topk = index_topk
    idx_mod.compress_ratio = 2

    dummy_x = jnp.zeros((batch_size, padded_seq_len, 4), dtype=jnp.float32)
    dummy_qr = jnp.zeros((batch_size, padded_seq_len, 4), dtype=jnp.float32)
    dummy_pos = jnp.broadcast_to(jnp.arange(padded_seq_len, dtype=jnp.int32)[None, :], (batch_size, padded_seq_len))
    dummy_shared = {"index_k": jnp.zeros((batch_size, num_compressed, 4), dtype=jnp.float32)}

    def _fake_scores_and_reachability(self_idx, x, qr, positions, shared, *, decoder_segment_ids=None):
      del self_idx, x, qr, shared, decoder_segment_ids
      compress_lens = (positions // 2) + 1
      reachable = jnp.arange(num_compressed)[None, None, :] < compress_lens[..., None]
      scores = jnp.where(
          reachable,
          jnp.arange(num_compressed, dtype=jnp.float32)[None, None, :],
          -jnp.inf,
      )
      return scores, reachable, compress_lens, num_compressed, {}

    # 1. Prove RED: without coordinate-offset normalization, _frozen_indexer_call produces an all-False mask
    #    and check_router_agreement fails with unpruned_reachable_mismatch ({8} vs {0}).
    with (
        mock.patch.object(
            deepseek_v41_frozen_topk,
            "_normalize_ref_indexer_topk_slice",
            side_effect=lambda arr, s, **kw: np.asarray(arr, dtype=np.int32),
        ),
        mock.patch.object(
            deepseek_v41_frozen_topk,
            "_compute_indexer_scores_and_reachability",
            side_effect=_fake_scores_and_reachability,
        ),
    ):
      with deepseek_v41_frozen_topk.frozen_topk_routing_context(
          model_stub,
          cfg_stub,
          ref_moe,
          sample_name="L1_red",
          seq_len=seq_len,
          batch_size=batch_size,
          padded_seq_len=padded_seq_len,
          raw_indexer_topk=ref_indexer_topk_offset,
          raw_indexer_layer_ids=layer_ids,
      ):
        mask_red, _ = attention_deepseek41.DeepSeek41Indexer.__call__(  # pylint: disable=unnecessary-dunder-call
            idx_mod, dummy_x, dummy_qr, dummy_pos, dummy_shared, decoder_segment_ids=None
        )
        self.assertEqual(int(jnp.sum(mask_red)), 0)

      ra_red = deepseek_v41_frozen_topk.check_router_agreement(
          sample_name="L1_red",
          seq_len=seq_len,
          token_types=np.full((seq_len,), -1, dtype=np.int32),
          tpu_moe_topk=tpu_moe,
          ref_topk_experts=ref_moe,
          ref_gate_margins=ref_gate_margins,
          ref_gate_boundary_ulp=ref_gate_ulp,
          tpu_indexer_topk=tpu_indexer_map,
          ref_indexer_layer_ids=layer_ids,
          ref_indexer_topk=ref_indexer_topk_offset,
          ref_indexer_margins=ref_idx_margins,
          ref_indexer_boundary_ulp=ref_idx_ulp,
          raise_on_failure=False,
      )
      self.assertFalse(ra_red["passed"])
      self.assertGreater(ra_red["high_margin_indexer_failure_count"], 0)
      self.assertEqual(ra_red["high_margin_indexer_failures"][0]["reason"], "unpruned_reachable_mismatch")
      self.assertEqual(ra_red["high_margin_indexer_failures"][0]["ref_only"], [seq_len])
      self.assertEqual(ra_red["high_margin_indexer_failures"][0]["tpu_only"], [0])

    # 2. Prove GREEN: with normalization enabled, prepare_padded_ref_indexer_topk shifts [seq_len, seq_len + C)
    #    into [0, C), _frozen_indexer_call selects the exact compressed blocks, and check_router_agreement passes!
    padded_green = deepseek_v41_frozen_topk.prepare_padded_ref_indexer_topk(
        ref_indexer_topk_offset,
        layer_ids,
        sample_name="L1_green",
        seq_len=seq_len,
        padded_seq_len=padded_seq_len,
        batch_size=batch_size,
        expected_layer_ids=[0, 1],
        index_topk=index_topk,
    )
    np.testing.assert_array_equal(padded_green[0], tpu_idx_l0)
    np.testing.assert_array_equal(padded_green[1], tpu_idx_l1)

    with mock.patch.object(
        deepseek_v41_frozen_topk,
        "_compute_indexer_scores_and_reachability",
        side_effect=_fake_scores_and_reachability,
    ):
      with deepseek_v41_frozen_topk.frozen_topk_routing_context(
          model_stub,
          cfg_stub,
          ref_moe,
          sample_name="L1_green",
          seq_len=seq_len,
          batch_size=batch_size,
          padded_seq_len=padded_seq_len,
          raw_indexer_topk=ref_indexer_topk_offset,
          raw_indexer_layer_ids=layer_ids,
      ):
        mask_green, _ = attention_deepseek41.DeepSeek41Indexer.__call__(  # pylint: disable=unnecessary-dunder-call
            idx_mod, dummy_x, dummy_qr, dummy_pos, dummy_shared, decoder_segment_ids=None
        )
        mask_green_np = np.asarray(mask_green, dtype=np.bool_)
        for pos in range(seq_len):
          expected_blocks = {int(x) for x in tpu_idx_1b[pos].tolist() if int(x) >= 0}
          actual_blocks = {int(c) for c in range(num_compressed) if bool(mask_green_np[0, pos, c])}
          self.assertEqual(actual_blocks, expected_blocks, msg=f"Mismatch at pos={pos}")

    ra_green = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="L1_green",
        seq_len=seq_len,
        token_types=np.full((seq_len,), -1, dtype=np.int32),
        tpu_moe_topk=tpu_moe,
        ref_topk_experts=ref_moe,
        ref_gate_margins=ref_gate_margins,
        ref_gate_boundary_ulp=ref_gate_ulp,
        tpu_indexer_topk=tpu_indexer_map,
        ref_indexer_layer_ids=layer_ids,
        ref_indexer_topk=ref_indexer_topk_offset,
        ref_indexer_margins=ref_idx_margins,
        ref_indexer_boundary_ulp=ref_idx_ulp,
        raise_on_failure=True,
    )
    self.assertTrue(ra_green["passed"])
    self.assertEqual(ra_green["high_margin_indexer_failure_count"], 0)
    self.assertEqual(ra_green["low_margin_indexer_flip_count"], 0)

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference/safetensors.torch")
  def test_run_frozen_topk_forward_nnx_captures_teacher_forced_pre_override_routing_red_to_green(self) -> None:
    """Proves teacher-forced pre-override capture prevents Layer 0 margin=0.0 tie from causing Layer 1 failures."""
    jax.config.update("jax_default_matmul_precision", "highest")
    tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
    ref_args = multimodal.mini_model_args()
    ref_mm = multimodal.build_multimodal_model(ref_args, tokenizer=tokenizer)
    multimodal.init_mini_reference(ref_mm, seed=20261031)

    short_seq_len = 24
    tie_pos = 5
    batch = multimodal.build_mini_vl_batch(
        ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        seq_len=short_seq_len,
        span_starts=((2,),),
        seed=2026,
    )
    tokens_np = batch.input_ids.numpy().astype(np.int32)[0]
    token_types = batch.token_types.numpy().astype(np.int32)[0]
    valid_patch_0 = batch.patches_bnpd.numpy().astype(np.float32)[0]
    padded_4_slots = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    padded_4_slots[0] = valid_patch_0[0]
    image_mask_slot1 = np.array([1, 0, 0, 0], dtype=np.int32)

    for scan_flag in (False, True):
      with self.subTest(scan_layers=scan_flag):
        cfg = mm_testing.mini_vision_maxtext_config(
            scan_layers=scan_flag,
            param_scan_axis=0,
            dtype="float32",
            weight_dtype="bfloat16",
            dtype_mm="float32",
            matmul_precision="highest",
            remat_policy="none",
            ds41_activation_quant=False,
            per_device_batch_size=1.0,
            ici_fsdp_parallelism=2,
            ici_tensor_parallelism=2,
            ici_expert_parallelism=1,
            max_target_length=multimodal.MINI_SEQ_LEN,
        )
        mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
        model = model_creation_utils.from_config(
            cfg,
            mesh=mesh,
            rngs=nnx.Rngs(params=0),
            model_mode=MODEL_MODE_TRAIN,
        )
        mm_testing.copy_full_multimodal_model(model, ref_mm, cfg, jnp.bfloat16)
        _, abs_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh=mesh)
        _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
        _, abs_p_state, abs_b_state, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)
        p_sharded = jax.tree.map(lambda a, av: jax.device_put(a, av.sharding), p_state, abs_p_state)
        b_sharded = jax.tree.map(lambda a, av: jax.device_put(a, av.sharding), b_state, abs_b_state)
        nnx.update(model, p_sharded, b_sharded, rest_state)

        # Amplify Layer 0 expert weights so a Layer 0 tie-flip at tie_pos perturbs Layer 1's input hidden state
        # enough to flip Layer 1's natural router selection when Layer 0 is NOT teacher-forced.
        if scan_flag:
          l0_moe = model.decoder.segments_0.layers_0.mlp.MoeBlock_0
        else:
          l0_moe = model.decoder.layers_0.mlp.MoeBlock_0
        for w_name in ("wi_0", "wi_1", "wo"):
          w_var = getattr(l0_moe, w_name)
          w_arr = w_var[...]
          if scan_flag:
            scale_factors = jnp.arange(1, int(w_arr.shape[1]) + 1, dtype=w_arr.dtype).reshape(
                (1, -1) + (1,) * (w_arr.ndim - 2)
            ) * jnp.asarray(25.0, dtype=w_arr.dtype)
            w_new = w_arr.at[0].set(w_arr[0] * scale_factors[0])
          else:
            scale_factors = jnp.arange(1, int(w_arr.shape[0]) + 1, dtype=w_arr.dtype).reshape(
                (-1,) + (1,) * (w_arr.ndim - 1)
            ) * jnp.asarray(25.0, dtype=w_arr.dtype)
            w_new = w_arr * scale_factors
          w_var.set_value(jax.device_put(w_new, w_arr.sharding))

        max_len = int(cfg.max_target_length)
        gbs = int(cfg.global_batch_size_to_train_on)
        ids_1 = np.zeros((1, max_len), dtype=np.int32)
        ids_1[0, :short_seq_len] = tokens_np
        ids_b = jnp.tile(jnp.asarray(ids_1, dtype=jnp.int32), (gbs, 1))
        pos_b = jnp.broadcast_to(jnp.arange(max_len, dtype=jnp.int32)[None, :], (gbs, max_len))
        seg_1 = np.zeros((1, max_len), dtype=np.int32)
        seg_1[0, :short_seq_len] = 1
        seg_b = jnp.tile(jnp.asarray(seg_1, dtype=jnp.int32), (gbs, 1))
        imgs_b = jnp.tile(jnp.asarray(padded_4_slots[None, ...], dtype=jnp.float32), (gbs, 1, 1, 1))
        masks_b = jnp.tile(jnp.asarray(image_mask_slot1[None, :], dtype=jnp.int32), (gbs, 1))

        nat_res = deepseek_v41_frozen_topk.run_natural_routing_forward_with_trace_nnx(
            model,
            cfg,
            mesh,
            sample_name="L2_sim",
            ids=ids_b,
            decoder_positions=pos_b,
            decoder_segment_ids=seg_b,
            images=imgs_b,
            image_masks=masks_b,
        )
        unforced_moe = np.asarray(nat_res.tpu_moe_topk[:, 0, :short_seq_len, :], dtype=np.uint16)
        num_layers = int(unforced_moe.shape[0])
        num_experts = int(cfg.num_experts)

        # Simulate L2 Layer 0 position tie_pos exact zero-margin tie: reference chose alt_expert instead of
        # unforced_moe[0, tie_pos, -1] at margin = 0.0.
        ref_moe = unforced_moe.copy()
        cur_l0_set = {int(x) for x in ref_moe[0, tie_pos].tolist()}
        alt_expert = next(e for e in range(num_experts - 1, -1, -1) if e not in cur_l0_set)
        ref_moe[0, tie_pos, -1] = np.uint16(alt_expert)

        sorted_idx_layers = sorted(nat_res.tpu_indexer_topk.keys())
        ref_indexer_layer_ids = np.asarray(sorted_idx_layers, dtype=np.int32)
        raw_idx_0based = np.stack(
            [
                np.asarray(nat_res.tpu_indexer_topk[lid][0, :short_seq_len, :], dtype=np.int32)
                for lid in sorted_idx_layers
            ],
            axis=0,
        )
        # Store ref_indexer_topk with PyTorch reference +seq_len offset.
        ref_indexer_topk_offset = np.where(raw_idx_0based >= 0, raw_idx_0based + short_seq_len, -1).astype(np.int32)

        # First run teacher-forced Layer 0 to record the true teacher-forced downstream Layer 1..N selections.
        _, forced_cap_init = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
            model,
            cfg,
            mesh,
            ref_moe,
            sample_name="L2_sim",
            ids=ids_b,
            decoder_positions=pos_b,
            decoder_segment_ids=seg_b,
            images=imgs_b,
            image_masks=masks_b,
            seq_len=short_seq_len,
            raw_indexer_topk=ref_indexer_topk_offset,
            raw_indexer_layer_ids=ref_indexer_layer_ids,
            return_captured=True,
        )
        ref_moe[1:] = np.asarray(forced_cap_init.tpu_moe_topk[1:, 0, :short_seq_len, :], dtype=np.uint16)
        forced_idx_0based = np.stack(
            [
                np.asarray(forced_cap_init.tpu_indexer_topk[lid][0, :short_seq_len, :], dtype=np.int32)
                for lid in sorted_idx_layers
            ],
            axis=0,
        )
        ref_indexer_topk_offset = np.where(forced_idx_0based >= 0, forced_idx_0based + short_seq_len, -1).astype(np.int32)

        # Verify that Layer 0's tie flip genuinely caused at least one downstream layer (l >= 1) to diverge in unforced!
        self.assertFalse(
            bool(np.array_equal(unforced_moe[1:], ref_moe[1:])),
            msg="Expected Layer 0 tie flip to perturb downstream unforced routing at Layer >= 1",
        )

        ref_gate_ulp = np.full((num_layers, short_seq_len), 1.0e-6, dtype=np.float32)
        ref_gate_margins = np.full((num_layers, short_seq_len), 20.0 * 1.0e-6, dtype=np.float32)
        # Exact zero-margin tie at Layer 0 position tie_pos (like L2 Layer 0 pos 364).
        ref_gate_margins[0, tie_pos] = 0.0

        n_idx_layers = int(ref_indexer_layer_ids.shape[0])
        ref_indexer_ulp = np.full((n_idx_layers, short_seq_len), 1.0e-6, dtype=np.float32)
        ref_indexer_margins = np.full((n_idx_layers, short_seq_len), 20.0 * 1.0e-6, dtype=np.float32)

        # 1. Prove RED on unforced capture: downstream layers see perturbed hidden states and fail high-margin gate!
        ra_unforced_red = deepseek_v41_frozen_topk.check_router_agreement(
            sample_name="L2_sim",
            seq_len=short_seq_len,
            token_types=token_types,
            tpu_moe_topk=nat_res.tpu_moe_topk,
            ref_topk_experts=ref_moe,
            ref_gate_margins=ref_gate_margins,
            ref_gate_boundary_ulp=ref_gate_ulp,
            tpu_indexer_topk=nat_res.tpu_indexer_topk,
            ref_indexer_layer_ids=ref_indexer_layer_ids,
            ref_indexer_topk=ref_indexer_topk_offset,
            ref_indexer_margins=ref_indexer_margins,
            ref_indexer_boundary_ulp=ref_indexer_ulp,
            raise_on_failure=False,
        )
        self.assertFalse(ra_unforced_red["passed"])
        self.assertGreater(ra_unforced_red["high_margin_moe_failure_count"], 0)
        self.assertEqual(ra_unforced_red["low_margin_moe_flip_count"], 1)
        self.assertEqual(ra_unforced_red["low_margin_moe_flips"][0]["layer"], 0)
        self.assertEqual(ra_unforced_red["low_margin_moe_flips"][0]["pos"], tie_pos)

        # 2. Prove GREEN on teacher-forced pre-override capture via run_frozen_topk_forward_nnx(..., return_captured=True):
        forced_logits, forced_captured = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
            model,
            cfg,
            mesh,
            ref_moe,
            sample_name="L2_sim",
            ids=ids_b,
            decoder_positions=pos_b,
            decoder_segment_ids=seg_b,
            images=imgs_b,
            image_masks=masks_b,
            seq_len=short_seq_len,
            raw_indexer_topk=ref_indexer_topk_offset,
            raw_indexer_layer_ids=ref_indexer_layer_ids,
            return_captured=True,
        )
        self.assertEqual(forced_logits.shape[:2], (gbs, max_len))
        ra_forced_green = deepseek_v41_frozen_topk.check_router_agreement(
            sample_name="L2_sim",
            seq_len=short_seq_len,
            token_types=token_types,
            tpu_moe_topk=forced_captured.tpu_moe_topk,
            ref_topk_experts=ref_moe,
            ref_gate_margins=ref_gate_margins,
            ref_gate_boundary_ulp=ref_gate_ulp,
            tpu_indexer_topk=forced_captured.tpu_indexer_topk,
            ref_indexer_layer_ids=ref_indexer_layer_ids,
            ref_indexer_topk=ref_indexer_topk_offset,
            ref_indexer_margins=ref_indexer_margins,
            ref_indexer_boundary_ulp=ref_indexer_ulp,
            raise_on_failure=True,
        )
        self.assertTrue(ra_forced_green["passed"])
        self.assertEqual(ra_forced_green["high_margin_moe_failure_count"], 0)
        self.assertEqual(ra_forced_green["high_margin_indexer_failure_count"], 0)
        self.assertEqual(ra_forced_green["low_margin_moe_flip_count"], 1)
        self.assertEqual(ra_forced_green["low_margin_moe_flips"][0]["layer"], 0)
        self.assertEqual(ra_forced_green["low_margin_moe_flips"][0]["pos"], tie_pos)

  def test_check_router_agreement_deduplicates_replicated_batch_slots_and_catches_slot_divergence(self) -> None:
    """Proves batch_size=8 deduplication emits 1 entry when slots 0..7 match, and reports b > 0 when slot b diverges."""
    batch_size = 8
    seq_len = 6
    num_layers = 2
    top_k = 2
    index_topk = 3
    layer_ids = np.array([0, 1], dtype=np.int32)

    ref_moe = np.zeros((num_layers, seq_len, top_k), dtype=np.uint16)
    ref_moe[:, :, 0] = 1
    ref_moe[:, :, 1] = 2
    ref_gate_ulp = np.full((num_layers, seq_len), 1.0e-6, dtype=np.float32)
    ref_gate_margins = np.full((num_layers, seq_len), 20.0 * 1.0e-6, dtype=np.float32)
    ref_gate_margins[0, 1] = 2.0 * 1.0e-6  # Low-margin position at layer 0, pos 1

    ref_idx_0based = np.broadcast_to(np.array([0, 1, 2], dtype=np.int32)[None, None, :], (2, seq_len, index_topk)).copy()
    ref_idx_offset = ref_idx_0based + seq_len
    ref_idx_ulp = np.full((2, seq_len), 1.0e-6, dtype=np.float32)
    ref_idx_margins = np.full((2, seq_len), 20.0 * 1.0e-6, dtype=np.float32)
    ref_idx_margins[1, 4] = 2.0 * 1.0e-6  # Low-margin position at indexer layer 1, pos 4

    # 1. Replicated mismatch across all 8 batch slots (slots 0..7 identical):
    tpu_moe_rep = np.broadcast_to(
        ref_moe[:, None, :, :].astype(np.int32), (num_layers, batch_size, seq_len, top_k)
    ).copy()
    tpu_moe_rep[0, :, 1, 1] = 5  # Replicated low-margin flip across all 8 slots
    tpu_moe_rep[1, :, 2, 1] = 6  # Replicated high-margin mismatch across all 8 slots

    tpu_idx_rep = {
        0: np.broadcast_to(ref_idx_0based[0, None, :, :], (batch_size, seq_len, index_topk)).copy(),
        1: np.broadcast_to(ref_idx_0based[1, None, :, :], (batch_size, seq_len, index_topk)).copy(),
    }
    tpu_idx_rep[0][:, 3, 2] = 5  # Replicated high-margin indexer mismatch across all 8 slots
    tpu_idx_rep[1][:, 4, 2] = 4  # Replicated low-margin indexer flip across all 8 slots

    res_rep = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="L2_rep8",
        seq_len=seq_len,
        token_types=np.full((seq_len,), -1, dtype=np.int32),
        tpu_moe_topk=tpu_moe_rep,
        ref_topk_experts=ref_moe,
        ref_gate_margins=ref_gate_margins,
        ref_gate_boundary_ulp=ref_gate_ulp,
        tpu_indexer_topk=tpu_idx_rep,
        ref_indexer_layer_ids=layer_ids,
        ref_indexer_topk=ref_idx_offset,
        ref_indexer_margins=ref_idx_margins,
        ref_indexer_boundary_ulp=ref_idx_ulp,
        raise_on_failure=False,
    )
    self.assertFalse(res_rep["passed"])
    # Each replicated mismatch is emitted exactly ONCE (for slot 0), not 8 times!
    self.assertEqual(res_rep["high_margin_moe_failure_count"], 1)
    self.assertEqual(res_rep["low_margin_moe_flip_count"], 1)
    self.assertEqual(res_rep["high_margin_indexer_failure_count"], 1)
    self.assertEqual(res_rep["low_margin_indexer_flip_count"], 1)
    self.assertEqual(res_rep["high_margin_moe_failures"][0]["batch_slot"], 0)
    self.assertEqual(res_rep["low_margin_moe_flips"][0]["batch_slot"], 0)
    self.assertEqual(res_rep["high_margin_indexer_failures"][0]["batch_slot"], 0)
    self.assertEqual(res_rep["low_margin_indexer_flips"][0]["batch_slot"], 0)

    # 2. Batch-invariance divergence: slot 0 matches reference, but slot 5 diverges!
    tpu_moe_div = np.broadcast_to(
        ref_moe[:, None, :, :].astype(np.int32), (num_layers, batch_size, seq_len, top_k)
    ).copy()
    tpu_moe_div[1, 5, 2, 1] = 7  # Only batch slot 5 diverges at layer 1, pos 2

    tpu_idx_div = {
        0: np.broadcast_to(ref_idx_0based[0, None, :, :], (batch_size, seq_len, index_topk)).copy(),
        1: np.broadcast_to(ref_idx_0based[1, None, :, :], (batch_size, seq_len, index_topk)).copy(),
    }
    tpu_idx_div[0][5, 3, 2] = 5  # Only batch slot 5 diverges at indexer layer 0, pos 3

    res_div = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="L2_slot5_div",
        seq_len=seq_len,
        token_types=np.full((seq_len,), -1, dtype=np.int32),
        tpu_moe_topk=tpu_moe_div,
        ref_topk_experts=ref_moe,
        ref_gate_margins=ref_gate_margins,
        ref_gate_boundary_ulp=ref_gate_ulp,
        tpu_indexer_topk=tpu_idx_div,
        ref_indexer_layer_ids=layer_ids,
        ref_indexer_topk=ref_idx_offset,
        ref_indexer_margins=ref_idx_margins,
        ref_indexer_boundary_ulp=ref_idx_ulp,
        raise_on_failure=False,
    )
    self.assertFalse(res_div["passed"])
    self.assertEqual(res_div["high_margin_moe_failure_count"], 1)
    self.assertEqual(res_div["high_margin_moe_failures"][0]["batch_slot"], 5)
    self.assertEqual(res_div["high_margin_indexer_failure_count"], 1)
    self.assertEqual(res_div["high_margin_indexer_failures"][0]["batch_slot"], 5)

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference/safetensors.torch")
  def test_audit_leaf_rep0_shards_engram_pad_multi_unique_uses_row_slice_not_full_tensor_red_to_green(
      self,
  ) -> None:
    """Proves RED when multi-unique engram_pad calls load_dequantized_tensor and GREEN with row-sliced streaming."""
    raw_rows = 100
    padded_rows = 120
    cols = 16
    hf_key = "layers.1.engram.embed.weight"

    rng = np.random.default_rng(20261031)
    raw_f32 = rng.standard_normal((raw_rows, cols), dtype=np.float32)
    raw_bf16_torch = torch.from_numpy(raw_f32).to(torch.bfloat16).contiguous()
    raw_bf16_np = raw_bf16_torch.view(torch.uint16).numpy().view(deepseek_v41_mm_audit.ml_dtypes.bfloat16)
    padded_expected = np.zeros((padded_rows, cols), dtype=raw_bf16_np.dtype)
    padded_expected[:raw_rows, :] = raw_bf16_np

    with tempfile.TemporaryDirectory(prefix="ds41_engram_pad_multi_u_") as tmp_dir:
      hf_dir = Path(tmp_dir)
      shard_name = "model-00001-of-00001.safetensors"
      save_file({hf_key: raw_bf16_torch}, (hf_dir / shard_name).as_posix(), metadata={"format": "pt"})
      with open(hf_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": {hf_key: shard_name}}, f)

      reader = deepseek_v41_mm_audit._SafetensorsShardReader(hf_dir)  # pylint: disable=protected-access
      rule = deepseek_v41_mm_audit.LeafAuditRule(
          maxtext_path="decoder/layers_1/engram/embed",
          kind="single",
          category="text",
          hf_keys=(hf_key,),
          transform="engram_pad",
          expected_mt_dtype="bfloat16",
          expected_hf_header_dtype="BF16",
          forbid_scale=False,
      )

      # 4 local replica-0 shards with num_unique = 4 distinct row slices (mimicking 4-chip v5p worker).
      row_bounds = [(0, 30), (30, 60), (60, 90), (90, 120)]
      rep0_shards = [
          types.SimpleNamespace(
              replica_id=0,
              index=(slice(r0, r1, None), slice(None, None, None)),
              data=padded_expected[r0:r1, :].copy(),
          )
          for r0, r1 in row_bounds
      ]

      row_slice_calls: list[tuple[str, int, int]] = []
      orig_load_row_slice = reader.load_dequantized_row_slice

      def _tracked_row_slice(key: str, target_dtype_str: str, row_start: int, row_stop: int) -> np.ndarray:
        row_slice_calls.append((key, int(row_start), int(row_stop)))
        return orig_load_row_slice(key, target_dtype_str, row_start, row_stop)

      def _forbid_full_tensor(key: str, target_dtype_str: str, *, use_cache: bool = True) -> np.ndarray:
        del target_dtype_str, use_cache
        raise RuntimeError(f"OOM guard: forbidden whole-tensor load_dequantized_tensor called for {key}")

      # 1. Prove RED: under the old unguarded rule.kind == "single" and num_unique > 1 behavior
      #    (simulated when _load_engram_padded_shard_2d falls back to load_dequantized_tensor), whole-tensor load fires!
      with (
          mock.patch.object(reader, "load_dequantized_tensor", side_effect=_forbid_full_tensor),
          mock.patch.object(
              deepseek_v41_mm_audit,
              "_load_engram_padded_shard_2d",
              side_effect=lambda **kw: reader.load_dequantized_tensor(
                  kw["hf_key"], kw["target_dtype_str"], use_cache=False
              ),
          ),
      ):
        with self.assertRaisesRegex(RuntimeError, "OOM guard: forbidden whole-tensor load_dequantized_tensor"):
          deepseek_v41_mm_audit._audit_leaf_rep0_shards(  # pylint: disable=protected-access
              mt_path=rule.maxtext_path,
              rule=rule,
              global_shape=(padded_rows, cols),
              rep0_shards=rep0_shards,
              scan_axis=0,
              reader=reader,
              config=types.SimpleNamespace(),
              pool=None,
          )

      # 2. Prove GREEN: with the fix, _audit_leaf_rep0_shards uses row-sliced streaming across all 4 unique shards,
      #    never calls load_dequantized_tensor, and verifies 100% bitwise equality.
      with (
          mock.patch.object(reader, "load_dequantized_tensor", side_effect=_forbid_full_tensor),
          mock.patch.object(reader, "load_dequantized_row_slice", side_effect=_tracked_row_slice),
      ):
        checked_shards, bit_mismatches, shard_failures = (
            deepseek_v41_mm_audit._audit_leaf_rep0_shards(  # pylint: disable=protected-access
                mt_path=rule.maxtext_path,
                rule=rule,
                global_shape=(padded_rows, cols),
                rep0_shards=rep0_shards,
                scan_axis=0,
                reader=reader,
                config=types.SimpleNamespace(),
                pool=None,
            )
        )
        self.assertEqual(checked_shards, 4)
        self.assertEqual(bit_mismatches, 0)
        self.assertEqual(shard_failures, [])
        self.assertEqual(
            row_slice_calls,
            [
                (hf_key, 0, 30),
                (hf_key, 30, 60),
                (hf_key, 60, 90),
                (hf_key, 90, 100),
            ],
        )

        # Negative control: perturbing one element in shard 2 (rows 60..90) is caught by row-sliced audit.
        bad_shards = list(rep0_shards)
        bad_data = padded_expected[60:90, :].copy()
        bad_data[5, 3] = deepseek_v41_mm_audit.ml_dtypes.bfloat16(123.0)
        bad_shards[2] = types.SimpleNamespace(
            replica_id=0,
            index=(slice(60, 90, None), slice(None, None, None)),
            data=bad_data,
        )
        checked_bad, mismatches_bad, failures_bad = (
            deepseek_v41_mm_audit._audit_leaf_rep0_shards(  # pylint: disable=protected-access
                mt_path=rule.maxtext_path,
                rule=rule,
                global_shape=(padded_rows, cols),
                rep0_shards=bad_shards,
                scan_axis=0,
                reader=reader,
                config=types.SimpleNamespace(),
                pool=None,
            )
        )
        self.assertEqual(checked_bad, 4)
        self.assertEqual(mismatches_bad, 1)
        self.assertEqual(len(failures_bad), 1)
        self.assertIn("decoder/layers_1/engram/embed", failures_bad[0])

      reader.close()

  def test_normalize_ref_indexer_topk_slice_rejects_non_offset_entries_with_layer_and_count(self) -> None:
    """Verifies `_normalize_ref_indexer_topk_slice` requires `>= seq_len` and raises with layer and count."""
    seq_len = 8
    # 1. Valid reference-offset slice (all valid entries >= seq_len, with -1 causal padding) normalizes cleanly.
    valid_offset_2d = np.array(
        [
            [8, -1, -1],
            [8, 9, -1],
            [8, 9, 10],
            [9, 10, 11],
            [8, 10, 11],
            [9, 11, 12],
            [8, 11, 12],
            [10, 11, 12],
        ],
        dtype=np.int32,
    )
    norm_2d = deepseek_v41_frozen_topk._normalize_ref_indexer_topk_slice(  # pylint: disable=protected-access
        valid_offset_2d, seq_len, layer_id=3
    )
    expected_0based_2d = np.where(valid_offset_2d >= 0, valid_offset_2d - seq_len, -1)
    np.testing.assert_array_equal(norm_2d, expected_0based_2d)

    # 2. 2D slice containing pre-normalized [0, seq_len) entries raises ValueError naming layer and count.
    bad_2d = valid_offset_2d.copy()
    bad_2d[0, 0] = 0
    bad_2d[2, 1] = 1
    bad_2d[5, 0] = 3
    bad_2d[5, 2] = 7
    bad_2d[7, 1] = 2
    with self.assertRaisesRegex(ValueError, r"layer 7.*count=5"):
      deepseek_v41_frozen_topk._normalize_ref_indexer_topk_slice(  # pylint: disable=protected-access
          bad_2d, seq_len, layer_id=7
      )

    # 3. 3D array with layer 0 valid and layer 2 having 3 non-offset entries raises ValueError naming layer 2 and count=3.
    arr_3d = np.stack([valid_offset_2d, valid_offset_2d, valid_offset_2d], axis=0)
    arr_3d[2, 1, 0] = 0
    arr_3d[2, 3, 2] = 4
    arr_3d[2, 6, 1] = 7
    with self.assertRaisesRegex(ValueError, r"layer 2.*count=3"):
      deepseek_v41_frozen_topk._normalize_ref_indexer_topk_slice(arr_3d, seq_len)  # pylint: disable=protected-access

    # 4. All-negative (-1) layer with count=0 valid entries when seq_len > 0 raises ValueError naming layer and count=0.
    all_neg_2d = np.full((seq_len, 3), -1, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, r"layer 5.*count=0"):
      deepseek_v41_frozen_topk._normalize_ref_indexer_topk_slice(  # pylint: disable=protected-access
          all_neg_2d, seq_len, layer_id=5
      )

    # 5. End-to-end propagation through prepare_padded_ref_indexer_topk and check_router_agreement names layer_id and count.
    raw_3d = np.stack([valid_offset_2d, valid_offset_2d], axis=0)
    raw_3d[1, 0, 0] = 0
    raw_3d[1, 4, 2] = 6
    with self.assertRaisesRegex(ValueError, r"layer 14.*count=2"):
      deepseek_v41_frozen_topk.prepare_padded_ref_indexer_topk(
          raw_3d,
          np.array([0, 14], dtype=np.int32),
          sample_name="S_bad_offset",
          seq_len=seq_len,
          padded_seq_len=seq_len,
          batch_size=1,
          expected_layer_ids=[0, 14],
          index_topk=3,
      )

    tpu_moe = np.zeros((1, 1, seq_len, 2), dtype=np.int32)
    ref_moe = np.zeros((1, seq_len, 2), dtype=np.uint16)
    ref_gate_m = np.full((1, seq_len), 1.0e-3, dtype=np.float32)
    ref_gate_u = np.full((1, seq_len), 1.0e-7, dtype=np.float32)
    tpu_idx_map = {
        2: expected_0based_2d[None, :, :],
        8: expected_0based_2d[None, :, :],
    }
    with self.assertRaisesRegex(ValueError, r"layer 8.*count=2"):
      deepseek_v41_frozen_topk.check_router_agreement(
          sample_name="S_bad_ra",
          seq_len=seq_len,
          token_types=np.full((seq_len,), -1, dtype=np.int32),
          tpu_moe_topk=tpu_moe,
          ref_topk_experts=ref_moe,
          ref_gate_margins=ref_gate_m,
          ref_gate_boundary_ulp=ref_gate_u,
          tpu_indexer_topk=tpu_idx_map,
          ref_indexer_layer_ids=np.array([2, 8], dtype=np.int32),
          ref_indexer_topk=raw_3d,
          ref_indexer_margins=np.full((2, seq_len), 1.0e-3, dtype=np.float32),
          ref_indexer_boundary_ulp=np.full((2, seq_len), 1.0e-6, dtype=np.float32),
          raise_on_failure=False,
      )

  def test_deepseek41_router_gate_logits_fp32_precision_spec(self) -> None:
    """Verifies DeepSeek-V4.1 configs default `float32_gate_logits=True` and `GateLogit` runs in `float32`."""
    base_yml = str(_REPO_ROOT / "src/maxtext/configs/base.yml")
    tokenizer_path = str(_REPO_ROOT / "src/maxtext/assets/tokenizers/tokenizer.llama2")
    with tempfile.TemporaryDirectory() as tmp_dir:
      cfg_flash = pyconfig.initialize(
          [
              sys.argv[0],
              base_yml,
              "model_name=deepseek4.1-flash",
              "run_name=ds41_fp32_gate_flash_spec",
              f"base_output_directory={tmp_dir}",
              f"tokenizer_path={tokenizer_path}",
              "dtype=bfloat16",
              "weight_dtype=bfloat16",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertIs(cfg_flash.float32_gate_logits, True)

      cfg_mini = pyconfig.initialize(
          [
              sys.argv[0],
              base_yml,
              "model_name=deepseek4.1-flash-mini",
              "run_name=ds41_fp32_gate_mini_spec",
              f"base_output_directory={tmp_dir}",
              f"tokenizer_path={tokenizer_path}",
              "dtype=bfloat16",
              "weight_dtype=bfloat16",
              "base_emb_dim=128",
              "base_num_query_heads=4",
              "base_num_kv_heads=1",
              "head_dim=32",
              "qk_rope_head_dim=16",
              "q_lora_rank=64",
              "o_groups=2",
              "o_lora_rank=64",
              "indexer_n_heads=4",
              "indexer_head_dim=32",
              "base_mlp_dim=256",
              "base_moe_mlp_dim=256",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
              "override_model_config=true",
          ]
      )
      self.assertIs(cfg_mini.float32_gate_logits, True)

    devices = jax.devices()
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_mini, devices), cfg_mini.mesh_axes)
    specs = deepseek41.build_layer_specs(cfg_mini)
    batch_size = 4
    seq_len = 16
    self.assertGreaterEqual(seq_len, 2 * cfg_mini.num_experts_per_tok)

    with jax.set_mesh(mesh):
      layer = deepseek41.DeepSeek41DecoderLayer(
          config=cfg_mini,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          spec=specs[0],
          rngs=nnx.Rngs(params=0),
      )
      gate = layer.mlp.MoeBlock_0.gate
      self.assertEqual(gate.dtype, jnp.float32)
      self.assertEqual(gate.weight_dtype, jnp.bfloat16)
      self.assertEqual(gate.kernel[...].dtype, jnp.bfloat16)

      key_x, key_w = jax.random.split(jax.random.PRNGKey(42))
      x_bf16 = jax.random.normal(key_x, (batch_size, seq_len, cfg_mini.emb_dim), dtype=jnp.bfloat16)
      kernel_bf16 = jax.random.normal(key_w, (cfg_mini.emb_dim, cfg_mini.num_experts), dtype=jnp.bfloat16)
      gate.kernel[...] = kernel_bf16

      gate_logits, pre_bias_logits = gate(x_bf16)
      self.assertEqual(gate_logits.dtype, jnp.float32)
      self.assertIsNotNone(pre_bias_logits)
      assert pre_bias_logits is not None
      self.assertEqual(pre_bias_logits.dtype, jnp.float32)

      ref_raw_fp32 = jnp.einsum(
          "bsd,de->bse",
          x_bf16.astype(jnp.float32),
          kernel_bf16.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      bf16_raw = jnp.einsum(
          "bsd,de->bse",
          x_bf16,
          kernel_bf16,
          precision=jax.lax.Precision.DEFAULT,
      ).astype(jnp.bfloat16)

      score_fn = linears._convert_to_activation_function(cfg_mini.routed_score_func)  # pylint: disable=protected-access
      ref_scores_fp32 = score_fn(ref_raw_fp32)
      bf16_scores = score_fn(bf16_raw).astype(jnp.bfloat16)
      score_fp32_err = float(jnp.max(jnp.abs(pre_bias_logits - ref_scores_fp32)))
      score_bf16_diff = float(jnp.max(jnp.abs(pre_bias_logits - bf16_scores.astype(jnp.float32))))
      self.assertLessEqual(score_fp32_err, _FP32_GATE_ATOL)
      self.assertGreater(score_bf16_diff, _BF16_GATE_TRUNCATION_MIN_DIFF)

      linear_gate = moe.GateLogit(
          in_features_shape=(cfg_mini.emb_dim,),
          out_features_shape=(cfg_mini.num_experts,),
          mesh=mesh,
          model_name=cfg_mini.model_name,
          dtype=gate.dtype,
          weight_dtype=gate.weight_dtype,
          use_bias=False,
          score_func="",
          matmul_precision=cfg_mini.matmul_precision,
          rngs=nnx.Rngs(params=1),
      )
      linear_gate.kernel[...] = kernel_bf16
      raw_gate_logits, raw_pre_bias_logits = linear_gate(x_bf16)
      self.assertEqual(raw_gate_logits.dtype, jnp.float32)
      self.assertIsNotNone(raw_pre_bias_logits)
      assert raw_pre_bias_logits is not None
      self.assertEqual(raw_pre_bias_logits.dtype, jnp.float32)

      raw_fp32_err = float(jnp.max(jnp.abs(raw_pre_bias_logits - ref_raw_fp32)))
      raw_bf16_diff = float(jnp.max(jnp.abs(raw_pre_bias_logits - bf16_raw.astype(jnp.float32))))
      self.assertLessEqual(raw_fp32_err, _FP32_GATE_ATOL)
      self.assertGreater(raw_bf16_diff, _BF16_GATE_TRUNCATION_MIN_DIFF)

  def test_aot_extract_memory_stats_and_batch_preset_dispatch(self) -> None:
    """Verifies `_extract_memory_stats` XLA peak telemetry and `all_e2e_presets` batch dispatch without SHA pins."""
    aot_check_path = (_REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py").as_posix()
    spec = importlib.util.spec_from_file_location("ds41_aot_check_batch_unit", aot_check_path)
    self.assertIsNotNone(spec)
    assert spec is not None and spec.loader is not None
    aot_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aot_mod)

    gib_bytes = 1024**3
    compiled_with_peak = types.SimpleNamespace(
        memory_analysis=lambda: types.SimpleNamespace(
            temp_size_in_bytes=60 * gib_bytes,
            argument_size_in_bytes=20 * gib_bytes,
            output_size_in_bytes=20 * gib_bytes,
            alias_size_in_bytes=20 * gib_bytes,
            generated_code_size_in_bytes=1 * gib_bytes,
            peak_memory_in_bytes=78 * gib_bytes,
        )
    )
    stats_with_peak = aot_mod._extract_memory_stats(  # pylint: disable=protected-access
        compiled_with_peak,
        max_hbm_gib=85.0,
        use_heap_compile_gate=False,
    )
    self.assertEqual(stats_with_peak["gate_mode"], "naive_sum_budget")
    self.assertTrue(stats_with_peak["passed"])
    self.assertAlmostEqual(stats_with_peak["naive_sum_hbm_gib"], 80.0, places=6)
    self.assertEqual(stats_with_peak["xla_peak_memory_in_bytes"], 78 * gib_bytes)
    self.assertAlmostEqual(stats_with_peak["xla_peak_hbm_gib"], 78.0, places=6)

    orig_run_preset = aot_mod.run_preset
    expected_heap_gib = 38067503936 / float(1024**3)

    def _mock_preset_dispatch(p: str, **kw: Any) -> dict[str, Any]:
      if p == "all_e2e_presets":
        return orig_run_preset(p, **kw)
      return {
          "preset": p,
          "compile_succeeded": True,
          "passed": True,
          "argument_size_in_bytes": 38067465024,
          "output_size_in_bytes": 38063870976,
          "alias_size_in_bytes": 38063832064,
          "non_aliased_io_hbm_gib": expected_heap_gib,
          "temp_hbm_gib": 27.148749,
          "naive_sum_hbm_gib": 62.601875,
          "peak_hbm_gib": 62.601875,
          "max_hbm_gib": 85.0,
      }

    with mock.patch.object(aot_mod, "run_preset", side_effect=_mock_preset_dispatch):
      e2e_batch_out = aot_mod.run_preset("all_e2e_presets")
    self.assertTrue(e2e_batch_out["passed"])
    self.assertNotIn("git_commit_sha", e2e_batch_out)
    self.assertIn("tpu7x_64c_mm_train_adamw", e2e_batch_out["results"])
    self.assertAlmostEqual(
        e2e_batch_out["results"]["tpu7x_64c_mm_train_adamw"]["non_aliased_io_hbm_gib"],
        expected_heap_gib,
        places=6,
    )

  def test_deepseek41_rotary_inv_freq_fp64_to_fp32_bit_exact(self) -> None:
    """Verifies `DeepSeek41RotaryEmbedding.inv_freq` matches float64->float32 reference across all 32 channels."""

    def _ref_inv_freq_f64_to_f32(
        *,
        dim: int,
        rope_theta: float,
        rope_factor: float,
        original_seq_len: int,
        beta_fast: int,
        beta_slow: int,
    ) -> np.ndarray:
      base = float(rope_theta)
      freqs = 1.0 / (np.float64(base) ** (np.arange(0, dim, 2, dtype=np.float64) / np.float64(dim)))
      if original_seq_len > 0:

        def corrected_dim(rotations: int) -> float:
          return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 1e-3), 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / np.float64(rope_factor) * (1.0 - smooth) + freqs * smooth
      return freqs.astype(np.float32)

    configs = [
        ("mla_swa", 64, 160000.0, 40.0, 4096, 32, 1),
        ("indexer", 64, 160000.0, 16.0, 4096, 32, 1),
        ("non_yarn", 64, 160000.0, 40.0, 0, 32, 1),
    ]
    for label, dim, theta, factor, orig_seq, b_fast, b_slow in configs:
      with self.subTest(config=label):
        rot = attention_deepseek41.DeepSeek41RotaryEmbedding(
            head_dim=dim,
            partial_rotary_factor=1.0,
            rope_theta=theta,
            rope_type="yarn" if orig_seq > 0 else "default",
            rope_factor=factor,
            original_max_position_embeddings=orig_seq if orig_seq > 0 else 65536,
            beta_fast=b_fast,
            beta_slow=b_slow,
            rngs=nnx.Rngs(0),
        )
        actual = np.asarray(rot.inv_freq, dtype=np.float32)
        expected = _ref_inv_freq_f64_to_f32(
            dim=dim,
            rope_theta=theta,
            rope_factor=factor,
            original_seq_len=orig_seq,
            beta_fast=b_fast,
            beta_slow=b_slow,
        )
        self.assertEqual(actual.shape, (32,))
        actual_u32 = actual.view(np.uint32)
        expected_u32 = expected.view(np.uint32)
        self.assertTrue(
            np.array_equal(actual_u32, expected_u32),
            msg=f"{label}: uint32 mismatch actual={actual_u32.tolist()} vs expected={expected_u32.tolist()}",
        )
        # Channel idx=3 (div=6/64) must equal 0x1.4cfa02p-2 (0x3ea67d01), not double-rounded 0x1.4cfa00p-2 (0x3ea67d00).
        self.assertEqual(int(actual_u32[3]), 0x3EA67D01)
        self.assertNotEqual(int(actual_u32[3]), 0x3EA67D00)
        self.assertEqual(float(actual[3]), float.fromhex("0x1.4cfa02p-2"))

    # Negative control: flipping the lowest mantissa bit of channel idx=3 must fail the bit-exact check.
    rot_nc = attention_deepseek41.DeepSeek41RotaryEmbedding(
        head_dim=64,
        partial_rotary_factor=1.0,
        rope_theta=160000.0,
        rope_type="default",
        rope_factor=40.0,
        original_max_position_embeddings=65536,
        beta_fast=32,
        beta_slow=1,
        rngs=nnx.Rngs(0),
    )
    actual_nc_u32 = np.asarray(rot_nc.inv_freq, dtype=np.float32).view(np.uint32)
    perturbed_u32 = (
        _ref_inv_freq_f64_to_f32(
            dim=64,
            rope_theta=160000.0,
            rope_factor=40.0,
            original_seq_len=0,
            beta_fast=32,
            beta_slow=1,
        )
        .view(np.uint32)
        .copy()
    )
    self.assertTrue(np.array_equal(actual_nc_u32, perturbed_u32))
    perturbed_u32[3] ^= 1
    self.assertFalse(np.array_equal(actual_nc_u32, perturbed_u32))

  def test_deepseek41_block_len1_scan_remat_matches_unrematted_loss_and_grads(self) -> None:
    """Verifies `DeepSeek41Block` `_run_deepseek4_layer_remat` matches unrematted and inline `nnx.remat` loss and grads."""
    if not HAS_TORCH:
      self.skipTest("torch required for mini_vision_maxtext_config")
    fp32_parity_rtol = 1.0e-5
    seq_len = 32
    with tempfile.TemporaryDirectory(prefix="ds41_unit_remat_scan_") as tmp_dir:
      ckpt_dir = os.path.join(tmp_dir, "mini_safetensors")
      args = multimodal.mini_model_args()
      ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
      multimodal.init_mini_reference(ref_mm, seed=91)
      offset = torch.tensor([-0.35, 0.35, -0.20, 0.20], dtype=torch.float32)
      with torch.no_grad():
        for layer in ref_mm.layers:
          layer.ffn.gate.bias_vl.copy_(layer.ffn.gate.bias + offset)
      mm_testing.write_mini_safetensors_checkpoint(ckpt_dir, ref_mm, param_dtype=torch.float32)
      del ref_mm

      cfg = mm_testing.mini_vision_maxtext_config(
          grid=multimodal.MINI_GRID_PADDED,
          run_name="unit_test_remat_scan",
          save_checkpoint_on_completion=False,
          load_parameters_path=ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
          dataset_type="synthetic",
          per_device_batch_size=1.0,
          max_target_length=seq_len,
          ici_fsdp_parallelism=1,
          ici_data_parallelism=-1,
          ici_expert_parallelism=1,
          ici_tensor_parallelism=1,
          scan_layers=True,
          remat_policy="full",
          use_multimodal=True,
          load_balance_loss_weight=0.001,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
          float32_logits=True,
          float32_gate_logits=True,
      )
      cfg_unremat = mm_testing.mini_vision_maxtext_config(
          grid=multimodal.MINI_GRID_PADDED,
          run_name="unit_test_remat_scan_unremat",
          save_checkpoint_on_completion=False,
          load_parameters_path=ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
          dataset_type="synthetic",
          per_device_batch_size=1.0,
          max_target_length=seq_len,
          ici_fsdp_parallelism=1,
          ici_data_parallelism=-1,
          ici_expert_parallelism=1,
          ici_tensor_parallelism=1,
          scan_layers=True,
          remat_policy="none",
          use_multimodal=True,
          load_balance_loss_weight=0.001,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
          float32_logits=True,
          float32_gate_logits=True,
      )
      # Geometry contract: seq >= 2 * topk and prefix + >=2 scanned blocks + suffix.
      self.assertGreaterEqual(seq_len, 2 * int(cfg.num_experts_per_tok))
      mesh = jax.sharding.Mesh(
          maxtext_utils.create_device_mesh(cfg),
          cfg.mesh_axes,
      )
      model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      segments = deepseek41.build_segments(cfg)
      self.assertGreaterEqual(len(segments), 4)
      self.assertEqual(segments[0].repeats, 1)
      self.assertGreaterEqual(segments[2].repeats, 2)
      self.assertEqual(segments[3].repeats, 1)
      segment_blocks = [getattr(model.decoder, f"segments_{k}") for k in range(len(segments))]
      for blk in segment_blocks:
        self.assertIsInstance(blk, deepseek41.DeepSeek41Block)
        self.assertEqual(blk.config.remat_policy, "full")

      bsz = int(cfg.global_batch_size_to_train_on)
      n_h, n_w = cfg.ds41_vision_grid
      num_patches = n_h * n_w
      patch_dim = int(cfg.patch_size_for_vit) * int(cfg.patch_size_for_vit) * 3
      r = int(cfg.ds41_vision_downsample_ratio)
      grid_h = math.ceil(n_h / r)
      grid_w = math.ceil(n_w / r)
      num_img_tokens = 1 + grid_h * (grid_w + 1) + 1
      img_tok_id = int(cfg.ds41_image_token_id)

      rng = np.random.default_rng(1000)
      tokens_np = rng.integers(10, 500, size=(bsz, seq_len), dtype=np.int32)
      tokens_np[:, 4 : 4 + num_img_tokens] = img_tok_id
      tokens = jnp.asarray(tokens_np)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (bsz, seq_len))
      segment_ids = jnp.ones((bsz, seq_len), dtype=jnp.int32)
      targets = jnp.asarray(rng.integers(10, 500, size=(bsz, seq_len), dtype=np.int32))
      target_mask_np = np.ones((bsz, seq_len), dtype=np.float32)
      target_mask_np[:, 4 : 4 + num_img_tokens] = 0.0
      target_mask = jnp.asarray(target_mask_np)
      images = jnp.asarray(rng.standard_normal((bsz, 1, num_patches, patch_dim)).astype(np.float32) * 0.2)

      def _eval_model_loss_and_grads(mdl_in: Any) -> tuple[float, float, dict[str, np.ndarray]]:
        gdef, params_init, rest_init = nnx.split(mdl_in, nnx.Param, ...)

        def _loss_fn(p: Any, r_state: Any) -> tuple[jax.Array, jax.Array]:
          m = nnx.merge(gdef, p, r_state, copy=True)
          logits = m(
              decoder_input_tokens=tokens,
              decoder_positions=positions,
              decoder_segment_ids=segment_ids,
              encoder_images=images,
              enable_dropout=False,
              model_mode=MODEL_MODE_TRAIN,
          )
          log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
          token_ll = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
          denom = jnp.maximum(jnp.sum(target_mask), 1.0)
          nll = -jnp.sum(token_ll * target_mask) / denom
          _, _, other_st = nnx.split(m, nnx.Param, ...)
          lb_vals = maxtext_utils.collect_intermediates_by_suffix(nnx.to_pure_dict(other_st), "moe_lb_loss")
          if not lb_vals:
            raise RuntimeError("No moe_lb_loss sown intermediates found in model state!")
          lb_loss = jnp.mean(jnp.concatenate(lb_vals))
          total_loss = nll + lb_loss
          return total_loss, lb_loss

        with jax.default_matmul_precision("highest"), jax.set_mesh(mesh):
          (loss_val, lb_val), grads = jax.jit(jax.value_and_grad(_loss_fn, has_aux=True))(params_init, rest_init)
        grad_leaves: dict[str, np.ndarray] = {}
        for path, val in nnx.to_flat_state(grads):
          key = "/".join(str(k) for k in path)
          grad_leaves[key] = np.asarray(jax.device_get(val), dtype=np.float32)
        return float(loss_val), float(lb_val), grad_leaves

      def _assert_all_leaves_e8_fp32(
          grads_a: dict[str, np.ndarray],
          grads_ref: dict[str, np.ndarray],
          tag: str,
      ) -> None:
        self.assertEqual(set(grads_a.keys()), set(grads_ref.keys()))
        for k, ga in grads_a.items():
          gb = grads_ref[k]
          diff = np.abs(ga.astype(np.float64) - gb.astype(np.float64))
          max_diff = float(np.max(diff)) if diff.size > 0 else 0.0
          ref_inf = float(np.max(np.abs(gb.astype(np.float64)))) if gb.size > 0 else 0.0
          atol = fp32_parity_rtol * max(1.0, ref_inf)
          self.assertLessEqual(max_diff, atol, msg=f"{tag} leaf {k}: max_abs_diff={max_diff} > atol={atol}")
          frob_diff = float(np.linalg.norm(diff.ravel()))
          frob_ref = float(np.linalg.norm(gb.astype(np.float64).ravel()))
          frob_tol = fp32_parity_rtol * max(1.0, frob_ref)
          self.assertLessEqual(frob_diff, frob_tol, msg=f"{tag} leaf {k}: frob_diff={frob_diff} > frob_tol={frob_tol}")

      # 1. Run with `deepseek4._run_deepseek4_layer_remat` (`apply_internal_remat=True`).
      loss_remat, lb_remat, grads_remat = _eval_model_loss_and_grads(model)
      self.assertGreater(lb_remat, 0.0)
      self.assertGreater(len(grads_remat), 200)

      # 2. Run with `remat_policy="none"` (unrematted inside every `DeepSeek41Block`).
      model_unremat = model_creation_utils.from_pretrained(cfg_unremat, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      loss_unremat, lb_unremat, grads_unremat = _eval_model_loss_and_grads(model_unremat)

      self.assertAlmostEqual(loss_remat, loss_unremat, delta=fp32_parity_rtol * max(1.0, abs(loss_unremat)))
      self.assertAlmostEqual(lb_remat, lb_unremat, delta=fp32_parity_rtol * max(1.0, abs(lb_unremat)))
      _assert_all_leaves_e8_fp32(grads_remat, grads_unremat, "unrematted")

      # 3. Run with inline `nnx.remat` matching `DeepSeek41Block` layer call contract.
      def _inline_nnx_remat_run_layer(  # pylint: disable=too-many-positional-arguments,unused-argument
          layer: Any,
          y: Any,
          *,
          carry_in_xs: bool = False,
          config: Any = None,
          policy: Any = None,
          prevent_cse: Any = None,
          decoder_positions: jax.Array,
          decoder_input_tokens: jax.Array,
          decoder_segment_ids: jax.Array | None = None,
          image_mask: jax.Array | None = None,
      ) -> dict[str, Any]:
        del carry_in_xs, config

        def body(  # pylint: disable=too-many-positional-arguments
            mdl: Any, c: Any, pos: Any, tok: Any, seg_ids: Any, img_mask: Any
        ) -> dict[str, Any]:
          if img_mask is not None:
            return mdl(c, pos, tok, decoder_segment_ids=seg_ids, image_mask=img_mask)
          return mdl(c, pos, tok, decoder_segment_ids=seg_ids)

        return nnx.remat(body, policy=policy, prevent_cse=prevent_cse)(
            layer, y, decoder_positions, decoder_input_tokens, decoder_segment_ids, image_mask
        )

      with mock.patch.object(deepseek4, "_run_deepseek4_layer_remat", _inline_nnx_remat_run_layer):
        loss_inline, lb_inline, grads_inline = _eval_model_loss_and_grads(model)
      self.assertAlmostEqual(loss_remat, loss_inline, delta=fp32_parity_rtol * max(1.0, abs(loss_inline)))
      self.assertAlmostEqual(lb_remat, lb_inline, delta=fp32_parity_rtol * max(1.0, abs(lb_inline)))
      _assert_all_leaves_e8_fp32(grads_remat, grads_inline, "inline_nnx_remat")

      # 4. Negative control: perturbing a scanned-block parameter must break parity.
      gdef_p, params_p, rest_p = nnx.split(model, nnx.Param, ...)
      perturbed_model = nnx.merge(gdef_p, params_p, rest_p, copy=True)
      orig_w = perturbed_model.decoder.segments_2.layers_0.pre_self_attention_layer_norm.scale[...]
      perturbed_model.decoder.segments_2.layers_0.pre_self_attention_layer_norm.scale[...] = orig_w + 0.25
      loss_pert, _, grads_pert = _eval_model_loss_and_grads(perturbed_model)
      max_pert_grad_diff = max(float(np.max(np.abs(g_rem - grads_pert[k]))) for k, g_rem in grads_remat.items())
      self.assertGreater(abs(loss_remat - loss_pert) + max_pert_grad_diff, 1.0e-3)

  def test_moe_dropless_fallback_combinations_cpu_and_negative_control(self) -> None:
    """Verifies RoutedMoE with moe_dropless_fallback='layer' on 4 CPU devices across all 8 combinations."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Test requires at least 4 devices, got {len(devices)}")

    def fake_ra2a(operand, output, input_offsets, send_sizes, *args, **_):
      output_offsets = args[0]
      rows = jnp.arange(output.shape[0])[:, None]
      for i in range(input_offsets.shape[0]):
        in_idx = jnp.clip(rows[:, 0] - output_offsets[i] + input_offsets[i], 0, operand.shape[0] - 1)
        mask = (rows >= output_offsets[i]) & (rows < output_offsets[i] + send_sizes[i])
        output = jnp.where(mask, operand[in_idx], output)
      return output

    combos = list(
        itertools.product(
            [False, True],  # use_ring_of_experts
            [False, True],  # log_required_ragged_buffer_factor
            [1, 2],  # num_moe_token_chunks
        )
    )

    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=fake_ra2a):
      for use_roe, log_rbf, chunks in combos:
        for test_mode in ("overflow", "no_overflow"):
          rbf = 0.25 if test_mode == "overflow" else 4.0
          cfg = pyconfig.initialize(
              [None, get_test_config_path()],
              run_name=f"moe_comb_{use_roe}_{log_rbf}_{chunks}_{test_mode}",
              enable_checkpointing=False,
              model_name="mixtral-8x7b",
              override_model_config=True,
              base_emb_dim=128,
              base_mlp_dim=64,
              base_moe_mlp_dim=64,
              dtype="bfloat16",
              megablox=True,
              sparse_matmul=True,
              per_device_batch_size=2,
              ici_expert_parallelism=2,
              ici_fsdp_parallelism=2,
              use_ring_of_experts=use_roe,
              num_moe_token_chunks=chunks,
              max_target_length=16,
              float32_gate_logits=True,
              use_ragged_sort=True,
              ragged_buffer_factor=rbf,
              moe_dropless_fallback="layer",
              log_required_ragged_buffer_factor=log_rbf,
              log_moe_dropped_frac=log_rbf,
          )

          mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg, devices[:4]), cfg.mesh_axes)

          model = linen_wrappers.to_linen(
              moe.RoutedMoE,
              name="MoeBlock",
              config=cfg,
              num_experts=cfg.num_experts,
              num_experts_per_tok=cfg.num_experts_per_tok,
              mesh=mesh,
              kernel_init=attentions.nd_dense_init(1.0, "fan_in", "truncated_normal"),
              kernel_axes=("embed", "mlp"),
              intermediate_dim=cfg.mlp_dim,
              dtype=cfg.dtype,
              force_dropless=False,
          )

          key = jax.random.PRNGKey(42)
          x = jax.random.normal(key, (4, 16, cfg.emb_dim), dtype=jnp.bfloat16)

          with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
            vars_init = model.init({"params": key, "dropout": key}, x)

            def loss_fn(p, cur_model=model, cur_x=x):
              (out, lb_loss, _), mutated = cur_model.apply({"params": p}, cur_x, mutable=["intermediates"])
              loss = jnp.mean(out.astype(jnp.float32) ** 2)
              return loss + (lb_loss.astype(jnp.float32) if lb_loss is not None else 0.0), (out, mutated)

            (loss, (_, mutated)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(vars_init["params"])
            self.assertTrue(jnp.isfinite(loss), f"Loss is non-finite: {loss}")
            for leaf in jax.tree_util.tree_leaves(grads):
              self.assertTrue(jnp.all(jnp.isfinite(leaf)), "Gradient leaf contains non-finite values.")

            dropped_fracs = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_dropped_frac")
            if not log_rbf:
              self.assertEqual(len(dropped_fracs), 0)
            else:
              self.assertGreater(len(dropped_fracs), 0)
              for df in dropped_fracs:
                self.assertEqual(df.shape, (1,))
                self.assertEqual(df.dtype, jnp.float32)
                self.assertTrue(jnp.all(jnp.isfinite(df)))

  def test_moe_dropped_frac_truncation_exact_ratio_cpu(self) -> None:
    """Verifies real dropped_frac matches exact analytical ratio under concentrated routing truncation."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Test requires at least 4 devices, got {len(devices)}")

    def fake_ra2a(operand, output, input_offsets, send_sizes, *args, **_):
      output_offsets = args[0]
      rows = jnp.arange(output.shape[0])[:, None]
      for i in range(input_offsets.shape[0]):
        in_idx = jnp.clip(rows[:, 0] - output_offsets[i] + input_offsets[i], 0, operand.shape[0] - 1)
        mask = (rows >= output_offsets[i]) & (rows < output_offsets[i] + send_sizes[i])
        output = jnp.where(mask, operand[in_idx], output)
      return output

    cases = [
        ("fsdp", True, False),
        ("fsdp", False, False),
        ("data", True, False),
        ("data", False, False),
        ("fsdp", False, True),
    ]

    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=fake_ra2a):
      for mesh_type, use_roe, check_vma in cases:
        extra_parallelism = {
            "ici_fsdp_parallelism": 2 if mesh_type == "fsdp" else 1,
            "ici_data_parallelism": 2 if mesh_type == "data" else 1,
            "ici_expert_parallelism": 2,
        }
        cfg = pyconfig.initialize(
            [None, get_test_config_path()],
            run_name=f"moe_skewed_{mesh_type}_{use_roe}_{check_vma}",
            enable_checkpointing=False,
            model_name="mixtral-8x7b",
            override_model_config=True,
            base_emb_dim=128,
            base_mlp_dim=64,
            base_moe_mlp_dim=64,
            dtype="bfloat16",
            megablox=not check_vma,
            sparse_matmul=True,
            per_device_batch_size=2,
            use_ring_of_experts=use_roe,
            num_moe_token_chunks=1,
            max_target_length=16,
            float32_gate_logits=True,
            use_ragged_sort=not check_vma,
            ragged_buffer_factor=0.25,
            moe_dropless_fallback="none",
            log_required_ragged_buffer_factor=use_roe,
            log_moe_dropped_frac=True,
            check_vma=check_vma,
            **extra_parallelism,
        )

        mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg, devices[:4]), cfg.mesh_axes)

        model = linen_wrappers.to_linen(
            moe.RoutedMoE,
            name="MoeBlock",
            config=cfg,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            mesh=mesh,
            kernel_init=attentions.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=("embed", "mlp"),
            intermediate_dim=cfg.mlp_dim,
            dtype=cfg.dtype,
            force_dropless=False,
        )

        key = jax.random.PRNGKey(42)
        b, s = 4, 16
        k = cfg.num_experts_per_tok
        x = jax.random.normal(key, (b, s, cfg.emb_dim), dtype=jnp.bfloat16)

        # Force all tokens to route to expert 0 (and expert 1 if k > 1), both on shard 0
        forced_experts = jnp.zeros((b, s, k), dtype=jnp.int32)
        if k > 1:
          forced_experts = forced_experts.at[:, :, 1].set(1)

        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
          vars_init = model.init({"params": key, "dropout": key}, x)
          _, mutated = model.apply(
              {"params": vars_init["params"]},
              x,
              forced_routed_experts=forced_experts,
              mutable=["intermediates"],
          )

          dropped_fracs = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_dropped_frac")
          self.assertGreater(len(dropped_fracs), 0)
          for df in dropped_fracs:
            self.assertEqual(df.shape, (1,))
            self.assertEqual(df.dtype, jnp.float32)
            self.assertAlmostEqual(float(df[0]), 0.875, places=5)


@pytest.mark.cpu_only
class Pr6311ReviewFixRegressionTest(unittest.TestCase):
  """Regression tests for PR#6311 review findings: AOT image/mask CP sharding and frozen indexer precision."""

  def test_aot_check_forward_and_train_mm_cp_gt_num_images(self) -> None:
    """Proves forward and train AOT compile cleanly when ici_context_parallelism > max_num_images_per_example."""
    if jax.device_count() < 4:
      self.skipTest("Requires >= 4 host CPU devices to exercise context_parallelism=4 > num_images=1.")

    from tests.utils import deepseek_v41_mm_testing as mm_testing_mod  # pylint: disable=import-outside-toplevel

    spec = importlib.util.spec_from_file_location(
        "ds41_aot_check_reg",
        _REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py",
    )
    self.assertIsNotNone(spec)
    self.assertIsNotNone(spec.loader)
    aot_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aot_mod)

    cfg = mm_testing_mod.mini_vision_maxtext_config(
        grid=(4, 4),
        run_name="test_ds41_aot_cp_gt_images",
        tokenizer_path=mm_testing_mod.OFFLINE_TOKENIZER_PATH,
        dataset_type="tfds",
        max_num_images_per_example=1,
        ds41_vision_downsample_ratio=2,
        ds41_image_token_id=200,
        ds41_engram_compressed_vocab_size=0,
        vocab_size=256,
        patch_size_for_vit=4,
        hidden_size_for_vit=32,
        num_attention_heads_for_vit=2,
        intermediate_size_for_vit=64,
        num_hidden_layers_for_vit=1,
        scan_layers=False,
        per_device_batch_size=1.0,
        max_target_length=32,
        ici_context_parallelism=4,
        ici_data_parallelism=1,
        ici_fsdp_parallelism=1,
    )
    res_fwd = aot_mod.run_forward_aot_check(
        cfg,
        with_images=True,
        num_images=1,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
        routing_mode="natural",
    )
    self.assertTrue(res_fwd["passed"])

    res_train = aot_mod.run_train_aot_check(
        cfg,
        with_images=True,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
    )
    self.assertTrue(res_train["passed"])

  def test_frozen_topk_indexer_scores_precision_highest_and_initial_zero(self) -> None:
    """Ensures `_compute_indexer_scores_and_reachability` enforces Precision.HIGHEST and initial=0 matching production."""
    cfg = types.SimpleNamespace(
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_topk=2,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision="default",
        shard_mode="auto",
        normalization_layer_epsilon=1e-6,
        ds41_candidate_topk_blocks=1,
        ds41_candidate_block_size=2,
        ds41_activation_quant=False,
        q_lora_rank=8,
        emb_dim=16,
        head_dim=8,
    )
    rotary = attention_deepseek41.DeepSeek41RotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=1.0,
        rope_theta=10000.0,
        rope_type="default",
        rngs=nnx.Rngs(0),
    )
    idx_mod = attention_deepseek41.DeepSeek41Indexer(
        config=cfg,
        compress_ratio=2,
        owns_k=True,
        is_candidate_source=False,
        uses_candidates=False,
        rotary_embedding=rotary,
        kernel_init=attentions.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        rngs=nnx.Rngs(params=0),
    )

    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((1, 4, 16)).astype(np.float32))
    qr = jnp.asarray(rng.standard_normal((1, 4, 8)).astype(np.float32))
    positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None, :], (1, 4))
    shared = {"index_k": jnp.asarray(rng.standard_normal((1, 2, 8)).astype(np.float32))}

    jaxpr_txt = str(
        jax.make_jaxpr(
            lambda x_, qr_, pos_, sh_: deepseek_v41_frozen_topk._compute_indexer_scores_and_reachability(  # pylint: disable=protected-access
                idx_mod, x_, qr_, pos_, sh_, decoder_segment_ids=jnp.zeros((1, 4), dtype=jnp.int32)
            )
        )(x, qr, positions, shared)
    )
    self.assertIn("HIGHEST", jaxpr_txt)

    scores, _, compress_lens, _, _ = deepseek_v41_frozen_topk._compute_indexer_scores_and_reachability(  # pylint: disable=protected-access
        idx_mod, x, qr, positions, shared, decoder_segment_ids=jnp.zeros((1, 4), dtype=jnp.int32)
    )
    self.assertTrue(bool(jnp.all(compress_lens == 0)))
    self.assertTrue(bool(jnp.all(scores == -jnp.inf)))

    seg_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    scores_live, _, _, _, _ = deepseek_v41_frozen_topk._compute_indexer_scores_and_reachability(  # pylint: disable=protected-access
        idx_mod, x, qr, positions, shared, decoder_segment_ids=seg_ids
    )
    prod_mask, _ = idx_mod(x, qr, positions, shared, decoder_segment_ids=seg_ids)
    expected_topk = jax.lax.top_k(scores_live, 2)[1]
    prod_np = np.asarray(prod_mask, dtype=np.bool_)
    for pos in range(4):
      for idx in expected_topk[0, pos].tolist():
        if np.isfinite(float(scores_live[0, pos, idx])):
          self.assertTrue(bool(prod_np[0, pos, idx]))

  def test_mm_routed_moe_random_routing_false_vs_true(self) -> None:
    """Proves MM RoutedMoE with bias updates lowers cleanly with use_random_routing=False and rejects True."""
    if jax.device_count() < 2:
      self.skipTest(f"Requires >= 2 JAX devices for EP=2 random routing check, got {jax.device_count()}.")
    from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel
    from maxtext.trainers.pre_train import train_compile  # pylint: disable=import-outside-toplevel
    from maxtext.utils import sharding  # pylint: disable=import-outside-toplevel
    from tests.utils import deepseek_v41_mm_testing as mm_testing_mod  # pylint: disable=import-outside-toplevel

    common_kw = {
        "grid": (2, 2),
        "tokenizer_path": mm_testing_mod.OFFLINE_TOKENIZER_PATH,
        "dataset_type": "tfds",
        "max_num_images_per_example": 1,
        "ds41_vision_downsample_ratio": 2,
        "ds41_image_token_id": 200,
        "ds41_engram_compressed_vocab_size": 0,
        "vocab_size": 256,
        "patch_size_for_vit": 4,
        "hidden_size_for_vit": 32,
        "num_attention_heads_for_vit": 2,
        "intermediate_size_for_vit": 64,
        "num_hidden_layers_for_vit": 1,
        "scan_layers": False,
        "per_device_batch_size": 1.0,
        "max_target_length": 16,
        "ici_fsdp_parallelism": -1,
        "ici_data_parallelism": 1,
        "ici_expert_parallelism": 2,
        "ici_context_parallelism": 1,
        "routed_bias": True,
        "routed_bias_update_rate": 0.001,
        "use_multimodal": True,
        "sparse_matmul": True,
        "megablox": False,
        "use_tokamax_gmm": False,
        "use_gmm_v2": False,
        "opt_type": "sgd",
        "num_decoder_layers": 2,
    }

    def _lower_mm_train_step(cfg_run: Any):
      mesh = maxtext_utils.get_mesh_from_config(cfg_run, jax.devices())
      shaped_train_args, shaped_train_kwargs, state_mesh_shardings, _, model = train_compile.get_shaped_inputs(
          mesh, cfg_run
      )
      data_sharding = sharding.get_input_data_sharding(cfg_run, mesh)
      shaped_batch = dict(shaped_train_args[1])
      shaped_batch["images"] = jax.ShapeDtypeStruct(
          (int(cfg_run.global_batch_size_to_load), 1, 4, 48), jnp.float32, sharding=data_sharding
      )
      shaped_batch["image_masks"] = jax.ShapeDtypeStruct(
          (int(cfg_run.global_batch_size_to_load), 1), jnp.int32, sharding=data_sharding
      )
      batch_sharding = {k: data_sharding for k in shaped_batch}
      ps, sms = sharding.maybe_update_params_sharding_with_opt(cfg_run, state_mesh_shardings)
      in_sms = sharding.build_zero1_input_state_mesh_shardings(cfg_run, sms, ps)
      fn, in_s, out_s, st, don = maxtext_utils.get_functional_train_with_signature(
          train.train_step, batch_sharding, in_sms, model, cfg_run, ps
      )
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_run.logical_axis_rules):
        return jax.jit(fn, in_shardings=in_s, out_shardings=out_s, static_argnums=st, donate_argnums=don).lower(
            shaped_train_args[0], shaped_batch, **shaped_train_kwargs
        )

    cfg_ok = mm_testing_mod.mini_vision_maxtext_config(
        run_name="test_ds41_mm_rr_false",
        use_random_routing=False,
        **common_kw,
    )
    self.assertIsNotNone(_lower_mm_train_step(cfg_ok))

    cfg_bad = mm_testing_mod.mini_vision_maxtext_config(
        run_name="test_ds41_mm_rr_true",
        use_random_routing=True,
        **common_kw,
    )
    with self.assertRaisesRegex(ValueError, "use_random_routing=True does not support multimodal image_mask"):
      _lower_mm_train_step(cfg_bad)

  def test_logit_runner_kl_exit_42_fplc_max_kl_guard_and_audit_pool_none(self) -> None:
    """Verifies logit_runner KL exit-42, --run_hf_model --max_kl_div guard, and pool=None audit."""
    cfg_ns = types.SimpleNamespace()
    args_ns = types.SimpleNamespace()
    with mock.patch.object(
        logit_runner,
        "main",
        side_effect=AssertionError("KL divergence values [0.5] exceed the threshold 0.01"),
    ):
      with self.assertRaises(SystemExit) as cm:
        logit_runner.run_checker(cfg_ns, args_ns)
      self.assertEqual(cm.exception.code, logit_runner.EXIT_DETERMINISTIC_GATE_FAIL)

    with (
        mock.patch.object(
            forward_pass_logit_checker.maxtext_utils,
            "create_device_mesh",
            return_value=np.array([jax.devices()[0]]),
        ),
        mock.patch.object(forward_pass_logit_checker, "load_tokenizer", return_value=object()),
    ):
      with self.assertRaisesRegex(ValueError, "--run_hf_model=true requires --max_kl_div"):
        forward_pass_logit_checker.main(
            types.SimpleNamespace(mesh_axes=("data",), use_multimodal=False, use_audio=False, use_mrope=False),
            types.SimpleNamespace(run_hf_model=True, hf_model_path="org/repo", max_kl_div=None, golden_logits_path=""),
        )

    expected = np.arange(12, dtype=np.float32).reshape((2, 2, 3))
    rule = deepseek_v41_mm_audit.LeafAuditRule(
        maxtext_path="params/mlp/wi_0",
        category="text",
        kind="expert_stack",
        hf_keys=("e0", "e1"),
        transform="identity",
        expected_mt_dtype="float32",
    )
    fake_reader = types.SimpleNamespace(
        load_dequantized_tensor=lambda k, _dt, use_cache=False: expected[0 if k == "e0" else 1]
    )
    shard_ok = types.SimpleNamespace(data=expected, index=(slice(0, 2), slice(None), slice(None)))
    _, bad_ok, _ = deepseek_v41_mm_audit._audit_leaf_rep0_shards(  # pylint: disable=protected-access
        mt_path=rule.maxtext_path,
        rule=rule,
        global_shape=expected.shape,
        rep0_shards=[shard_ok],
        scan_axis=0,
        reader=fake_reader,
        config=types.SimpleNamespace(),
        pool=None,
    )
    self.assertEqual(bad_ok, 0)

    perturbed = expected.copy()
    perturbed[1, 0, 2] += 1.0
    shard_bad = types.SimpleNamespace(data=perturbed, index=(slice(0, 2), slice(None), slice(None)))
    _, bad_neg, _ = deepseek_v41_mm_audit._audit_leaf_rep0_shards(  # pylint: disable=protected-access
        mt_path=rule.maxtext_path,
        rule=rule,
        global_shape=expected.shape,
        rep0_shards=[shard_bad],
        scan_axis=0,
        reader=fake_reader,
        config=types.SimpleNamespace(),
        pool=None,
    )
    self.assertEqual(bad_neg, 1)

  def test_aot_check_tokenizer_path_cli_env_and_offline_engram_init(self) -> None:
    """Verifies aot_check --tokenizer_path, TOKENIZER_PATH fallback, non-clobbering overrides, and offline Engram init."""
    from tests.utils import deepseek_v41_mm_testing as mm_testing_mod  # pylint: disable=import-outside-toplevel

    aot_path = (_REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py").as_posix()
    spec = importlib.util.spec_from_file_location("ds41_aot_check_tok_unit", aot_path)
    self.assertIsNotNone(spec)
    assert spec is not None and spec.loader is not None
    aot_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aot_mod)

    local_tok = mm_testing_mod.OFFLINE_TOKENIZER_PATH
    captured_cfgs: list[Any] = []

    def _fake_fwd(cfg: Any, **_kw: Any) -> dict[str, Any]:
      captured_cfgs.append(cfg)
      return {
          "preset": "tpu7x_64c_mm_scanned_fwd",
          "mode": "forward",
          "compile_succeeded": True,
          "passed": True,
          "temp_hbm_gib": 1.0,
          "non_aliased_io_hbm_gib": 1.0,
          "naive_sum_hbm_gib": 2.0,
          "clamped_heap_cap_gib": 2.0,
          "tpu7x_heap_limit_gib": 85.0,
      }

    def _fake_train(cfg: Any, **_kw: Any) -> dict[str, Any]:
      captured_cfgs.append(cfg)
      return {
          "preset": "tpu7x_64c_mm_bench_min_remat",
          "mode": "train",
          "with_images": True,
          "use_multimodal": True,
          "compile_topology": "tpu7x-128",
          "remat_policy": "minimal",
          "max_hbm_gib": 94.74,
          "remat_fallback_hbm_gib": 85.0,
          "gate_mode": "tpu7x_heap_compile",
          "tpu7x_heap_limit_gib": 94.74,
          "clamped_heap_cap_gib": 2.0,
          "temp_hbm_gib": 1.0,
          "non_aliased_io_hbm_gib": 1.0,
          "naive_sum_hbm_gib": 2.0,
          "compile_succeeded": True,
          "passed": True,
      }

    # 1. Explicit --tokenizer_path CLI arg threads tokenizer_type="huggingface" and tokenizer_path through run_preset.
    with (
        mock.patch.dict(os.environ, {"TOKENIZER_PATH": ""}, clear=False),
        mock.patch.object(aot_mod, "run_forward_aot_check", side_effect=_fake_fwd),
    ):
      rc = aot_mod.main(["--preset=tpu7x_64c_mm_scanned_fwd", f"--tokenizer_path={local_tok}"])
      self.assertEqual(rc, 0)
    self.assertEqual(captured_cfgs[-1].tokenizer_type.value, "huggingface")
    self.assertEqual(captured_cfgs[-1].tokenizer_path, local_tok)

    # 2. Omitting --tokenizer_path falls back to TOKENIZER_PATH env var (including batch presets).
    with (
        mock.patch.dict(os.environ, {"TOKENIZER_PATH": local_tok}, clear=False),
        mock.patch.object(aot_mod, "run_train_aot_check", side_effect=_fake_train),
    ):
      rc_batch = aot_mod.main(["--preset=all_64c_bench_presets"])
      self.assertEqual(rc_batch, 0)
    self.assertEqual(captured_cfgs[-1].tokenizer_type.value, "huggingface")
    self.assertEqual(captured_cfgs[-1].tokenizer_path, local_tok)

    # 3. Unset/empty leaves defaults untouched (749B default resolves to deepseek-ai/DeepSeek-V4.1-Flash),
    # and explicit caller override in `overrides` is never clobbered.
    clean_env = dict(os.environ)
    clean_env.pop("TOKENIZER_PATH", None)
    with mock.patch.dict(os.environ, clean_env, clear=True):
      cfg_default = aot_mod._build_config_from_overrides(  # pylint: disable=protected-access
          aot_mod.get_preset_spec("tpu7x_64c_mm_scanned_fwd")["overrides"],
          cpu_mini=False,
      )
      self.assertEqual(cfg_default.tokenizer_path, "deepseek-ai/DeepSeek-V4.1-Flash")

    with mock.patch.dict(os.environ, {"TOKENIZER_PATH": "/tmp/should_not_clobber"}, clear=False):
      explicit_overrides = dict(aot_mod.get_preset_spec("tpu7x_64c_mm_scanned_fwd")["overrides"])
      explicit_overrides["tokenizer_path"] = local_tok
      explicit_overrides["tokenizer_type"] = "sentencepiece"
      cfg_explicit = aot_mod._build_config_from_overrides(  # pylint: disable=protected-access
          explicit_overrides,
          cpu_mini=False,
          tokenizer_path="/tmp/cli_should_not_clobber",
      )
      self.assertEqual(cfg_explicit.tokenizer_path, local_tok)
      self.assertEqual(cfg_explicit.tokenizer_type.value, "sentencepiece")

    # 4. Cold-cache HF_HUB_OFFLINE=1 subprocess test: fails closed with OSError when unset, succeeds with --tokenizer_path.
    with tempfile.TemporaryDirectory(prefix="ds41_aot_cold_hf_") as cold_cache:
      sub_env = dict(os.environ)
      sub_env.pop("TOKENIZER_PATH", None)
      sub_env["HF_HOME"] = cold_cache
      sub_env["HF_HUB_CACHE"] = cold_cache
      sub_env["TRANSFORMERS_CACHE"] = cold_cache
      sub_env["HF_HUB_OFFLINE"] = "1"
      sub_env["TRANSFORMERS_OFFLINE"] = "1"
      sub_env["PYTHONPATH"] = f"{(_REPO_ROOT / 'src').as_posix()}:{_REPO_ROOT.as_posix()}"

      probe_code = (
          "import importlib.util\n"
          "import sys\n"
          "from maxtext.models import deepseek41\n"
          "aot_path, mode_tok = sys.argv[1], sys.argv[2]\n"
          "spec = importlib.util.spec_from_file_location('ds41_aot_sub', aot_path)\n"
          "aot_mod = importlib.util.module_from_spec(spec)\n"
          "spec.loader.exec_module(aot_mod)\n"
          "overrides = dict(aot_mod.get_preset_spec('tpu7x_64c_mm_scanned_fwd')['overrides'])\n"
          "if mode_tok != '__UNSET__':\n"
          "  overrides['ds41_engram_compressed_vocab_size'] = 0\n"
          "  cfg = aot_mod._build_config_from_overrides(overrides, cpu_mini=False, tokenizer_path=mode_tok)\n"
          "else:\n"
          "  cfg = aot_mod._build_config_from_overrides(overrides, cpu_mini=False)\n"
          "tmap, compressed_vocab = deepseek41.compressed_token_map(cfg)\n"
          "print(f'OK:{tmap.shape[0]}:{compressed_vocab}:{cfg.tokenizer_type.value}:{cfg.tokenizer_path}')\n"
      )
      fail_proc = subprocess.run(
          [sys.executable, "-c", probe_code, aot_path, "__UNSET__"],
          env=sub_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(fail_proc.returncode, 0)
      self.assertIn("OSError", fail_proc.stderr)

      ok_proc = subprocess.run(
          [sys.executable, "-c", probe_code, aot_path, local_tok],
          env=sub_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(ok_proc.returncode, 0, msg=f"{ok_proc.stdout}\n{ok_proc.stderr}")
      self.assertIn(f":huggingface:{local_tok}", ok_proc.stdout)


if __name__ == "__main__":
  if _REPO_ROOT_INIT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_INIT)
  if "--worker_port" in sys.argv:
    p_idx = sys.argv.index("--worker_port")
    r_idx = sys.argv.index("--worker_rank")
    _run_two_process_worker(int(sys.argv[p_idx + 1]), int(sys.argv[r_idx + 1]))
    sys.exit(0)
  unittest.main()
