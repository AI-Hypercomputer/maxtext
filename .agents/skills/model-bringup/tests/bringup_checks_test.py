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

"""Unit tests for `.agents/skills/model-bringup/scripts/` utilities."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parents[2]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
ALL_SCRIPT_NAMES = (
    "aot_preflight.py",
    "audit_ckpt_vs_hf.py",
    "capture_golden.py",
    "check_hlo_perf.py",
    "check_jax_numerics_and_scale.py",
    "ckpt_roundtrip_check.py",
    "e2e_check.py",
    "extract_run_metrics.py",
    "family_diff.py",
    "inspect_hf_artifacts.py",
    "layer_parity.py",
    "negctl.py",
    "packing_train_check.py",
    "slice_hf_mini.py",
    "verify_checkpoint.py",
)


def _load_script(stem: str) -> Any:
  """Loads a script module from .agents/skills/model-bringup/scripts/."""
  qual = f"_test_model_bringup_{stem}"
  if qual in sys.modules:
    return sys.modules[qual]
  path = SCRIPTS_DIR / f"{stem}.py"
  spec = importlib.util.spec_from_file_location(qual, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[qual] = module
  spec.loader.exec_module(module)
  return module


aot_preflight = _load_script("aot_preflight")
audit_ckpt_vs_hf = _load_script("audit_ckpt_vs_hf")
capture_golden = _load_script("capture_golden")
check_hlo_perf = _load_script("check_hlo_perf")
check_jax_numerics_and_scale = _load_script("check_jax_numerics_and_scale")
ckpt_roundtrip_check = _load_script("ckpt_roundtrip_check")
e2e_check = _load_script("e2e_check")
extract_run_metrics = _load_script("extract_run_metrics")
family_diff = _load_script("family_diff")
inspect_hf_artifacts = _load_script("inspect_hf_artifacts")
layer_parity = _load_script("layer_parity")
negctl = _load_script("negctl")
packing_train_check = _load_script("packing_train_check")
slice_hf_mini = _load_script("slice_hf_mini")
verify_checkpoint = _load_script("verify_checkpoint")

_GIB = 1024**3

_ASYNC_CHAIN_HLO = """
%called_computation.1 (p0: bf16[1,64]) -> bf16[8,64] {
  %p0 = bf16[1,64]{1,0} parameter(0)
  ROOT %all-gather.1 = bf16[8,64]{1,0} all-gather(%p0), dimensions={0}
}

%async_computation.1 (p1: bf16[1,64]) -> bf16[8,64] {
  %p1 = bf16[1,64]{1,0} parameter(0)
  ROOT %all-gather.1.cloned = bf16[8,64]{1,0} call(%p1), to_apply=%called_computation.1
}, execution_thread="sparsecore"

%called_computation.2 (p2: bf16[8,64]) -> bf16[8,64] {
  %p2 = bf16[8,64]{1,0} parameter(0)
  ROOT %all-to-all.1 = bf16[8,64]{1,0} all-to-all(%p2), dimensions={0}
}

ENTRY %main (x: bf16[1,64], y: bf16[8,64]) -> bf16[8,64] {
  %x = bf16[1,64]{1,0} parameter(0)
  %y = bf16[8,64]{1,0} parameter(1)
  %all-gather.1.call-start = ((), (), s32[]) async-start(), async_execution_thread="sparsecore", calls=%async_computation.1
  %all-gather.1.call-update = ((bf16[1,64]), bf16[8,64]) async-update(%all-gather.1.call-start, %x)
  %dot.1 = bf16[8,64]{1,0} dot(%y, %y), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  %all-gather.1.call-done = bf16[8,64]{1,0} async-done(%all-gather.1.call-update)
  %a2a = bf16[8,64]{1,0} call(%dot.1), to_apply=%called_computation.2
  %copy-start.1 = (bf16[8,64]{1,0:S(5)}, bf16[8,64]{1,0}, u32[]) copy-start(%a2a)
  %copy-done.1 = bf16[8,64]{1,0:S(5)} copy-done(%copy-start.1)
  ROOT %add.1 = bf16[8,64]{1,0} add(%all-gather.1.call-done, %a2a)
}
"""

_REF_GATE_AND_ROPE_PY = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
  def __init__(self, dim: int, num_experts: int, gate_temp: float = 1.0):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(num_experts, dim))
    self.gate_temp = gate_temp

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x.float(), self.weight.float()) / self.gate_temp

class HyperMixBlock(nn.Module):
  def forward(self, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    with set_dtype(torch.float32):
      mixed = torch.matmul(coeff.float(), x.float())
    return mixed.to(x.dtype)

def precompute_freqs_cis(dim: int, max_seq_len: int, rope_theta: float = 160000.0):
  freqs = 1.0 / (
      rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
  ).to(torch.float32)
  t = torch.arange(max_seq_len, dtype=torch.float32)
  return torch.outer(t, freqs)
"""

_BUGGY_MOE_CONFIG_YML = """\
base_config: "base.yml"
model_name: "synthetic-moe-v2"
num_experts: 64
num_experts_per_tok: 8
routed_score_func: "sigmoid"
routed_bias: true
float32_gate_logits: false
"""

_OMITTED_MOE_CONFIG_YML = """\
base_config: "base.yml"
model_name: "synthetic-moe-v2"
num_experts: 64
num_experts_per_tok: 8
routed_score_func: "sigmoid"
routed_bias: true
"""

_FIXED_MOE_CONFIG_YML = """\
base_config: "base.yml"
model_name: "synthetic-moe-v2"
num_experts: 64
num_experts_per_tok: 8
routed_score_func: "sigmoid"
routed_bias: true
float32_gate_logits: true
"""

_BUGGY_ROUTER_AND_ROPE_PY = """\
import jax
import jax.numpy as jnp


class BuggyMoERouter:
  def __init__(self, config, weight):
    self.config = config
    self.weight = weight

  def __call__(self, x):
    logits = jnp.einsum("btd,ed->bte", x, self.weight)
    return logits / self.config.gate_temp


class BuggyRotaryEmbedding:
  def __init__(self, dim: int = 64, rope_theta: float = 160000.0):
    self.dim = dim
    self.rope_theta = rope_theta

  @property
  def inv_freq(self):
    exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim
    return 1.0 / (self.rope_theta ** exponents)
"""

_FIXED_ROUTER_AND_ROPE_PY = """\
import jax
import jax.numpy as jnp
import numpy as np


class FixedMoERouter:
  def __init__(self, config, weight):
    self.config = config
    self.weight = weight

  def __call__(self, x):
    x_f32 = x.astype(jnp.float32)
    w_f32 = self.weight.astype(jnp.float32)
    logits = jnp.einsum(
        "btd,ed->bte",
        x_f32,
        w_f32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return logits / jnp.float32(self.config.gate_temp)


class FixedRotaryEmbedding:
  def __init__(self, dim: int = 64, rope_theta: float = 160000.0):
    self.dim = dim
    self.rope_theta = rope_theta

  @property
  def inv_freq(self):
    exponents = np.arange(0, self.dim, 2, dtype=np.float64) / np.float64(self.dim)
    inv_f64 = 1.0 / (np.float64(self.rope_theta) ** exponents)
    return jnp.asarray(inv_f64, dtype=jnp.float32)
"""

_BUGGY_SEGMENT_PY = """\
import jax.numpy as jnp


class BuggyIndexedAttention:
  def __init__(self, window_size: int = 4):
    self.window_size = window_size

  def __call__(self, q, k, positions):
    delta = positions[:, :, None] - positions[:, None, :]
    mask = (delta >= 0) & (delta < self.window_size)
    scores = jnp.einsum("bthd,bshd->bhts", q, k)
    return jnp.where(mask[:, None, :, :], scores, -1e9)
"""

_FIXED_SEGMENT_PY = """\
import jax.numpy as jnp


class FixedIndexedAttention:
  def __init__(self, window_size: int = 4):
    self.window_size = window_size

  def __call__(self, q, k, positions, decoder_segment_ids):
    delta = positions[:, :, None] - positions[:, None, :]
    same_seg = (
        decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
    ) & (decoder_segment_ids[:, :, None] != 0)
    mask = (delta >= 0) & (delta < self.window_size) & same_seg
    scores = jnp.einsum("bthd,bshd->bhts", q, k)
    return jnp.where(mask[:, None, :, :], scores, -1e9)
"""


def _write_safetensors_header(path: str, header: dict[str, object]) -> None:
  """Writes a minimal header-only safetensors file."""
  payload = json.dumps(header).encode("utf-8")
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(payload)))
    f.write(payload)


def _init_git_repo(repo_dir: str) -> bool:
  """Initializes a temporary git repository."""
  git_bin = shutil.which("git")
  if not git_bin:
    return False
  env = dict(os.environ)
  env["GIT_AUTHOR_NAME"] = "Test"
  env["GIT_AUTHOR_EMAIL"] = "test@example.com"
  env["GIT_COMMITTER_NAME"] = "Test"
  env["GIT_COMMITTER_EMAIL"] = "test@example.com"
  subprocess.run([git_bin, "init", "-q", repo_dir], env=env, check=True)
  return True


def _run_git(repo_dir: str, *args: str) -> None:
  """Runs a git command inside `repo_dir` with deterministic author metadata."""
  git_bin = shutil.which("git")
  assert git_bin is not None
  env = dict(os.environ)
  env["GIT_AUTHOR_NAME"] = "Test"
  env["GIT_AUTHOR_EMAIL"] = "test@example.com"
  env["GIT_COMMITTER_NAME"] = "Test"
  env["GIT_COMMITTER_EMAIL"] = "test@example.com"
  subprocess.run([git_bin, "-C", repo_dir, *args], env=env, check=True)


def _git_commit_all(repo_dir: str, msg: str) -> None:
  """Stages and commits all files in repo_dir."""
  _run_git(repo_dir, "add", "-A")
  _run_git(repo_dir, "commit", "-q", "-m", msg)


class TorchFreeImportAndHelpCliTest(unittest.TestCase):
  """Verifies every bringup script works when `torch` is blocked and supports `--help`."""

  def test_all_scripts_import_with_torch_blocked_and_support_help(self) -> None:
    self.assertEqual(
        sorted(p.name for p in SCRIPTS_DIR.glob("*.py")),
        sorted(ALL_SCRIPT_NAMES),
    )
    probe = """
import importlib.util
import pathlib
import sys

sys.modules["torch"] = None
sys.modules["safetensors"] = None
sys.modules["safetensors.torch"] = None

scripts_dir = pathlib.Path(sys.argv[1])
for p in sorted(scripts_dir.glob("*.py")):
  spec = importlib.util.spec_from_file_location(f"_chk_{p.stem}", p)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
"""
    res = subprocess.run(
        [sys.executable, "-", str(SCRIPTS_DIR)],
        input=probe,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(res.returncode, 0, res.stderr)

    for script_name in ALL_SCRIPT_NAMES:
      help_res = subprocess.run(
          [sys.executable, str(SCRIPTS_DIR / script_name), "--help"],
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(
          help_res.returncode,
          0,
          f"{script_name} --help failed: {help_res.stderr}",
      )


class PackingTrainCheckTest(unittest.TestCase):
  """End-to-end pure-JAX tests for `packing_train_check.py`."""

  def test_packing_parity_passes_2doc_and_3doc_with_embed_fn(self) -> None:
    params, fwd_fn, emb_fn, fwd_emb_fn = packing_train_check.build_synthetic_causal_transformer(
        vocab_size=64, emb_dim=32, num_heads=4, mlp_dim=48, num_layers=2
    )
    for doc_lengths, pad_gaps in (((3, 2), (1, 2)), ((4, 3, 2), (1, 1, 2))):
      rep = packing_train_check.verify_packed_training_parity(
          forward_fn=fwd_fn,
          params=params,
          vocab_size=64,
          batch_size=2,
          doc_lengths=doc_lengths,
          pad_gaps=pad_gaps,
          num_steps=3,
          learning_rate=1e-2,
          embed_fn=emb_fn,
          forward_from_embeds_fn=fwd_emb_fn,
          seed=5,
          run_negative_control=True,
      )
      self.assertTrue(rep.passed, rep.failures)
      self.assertTrue(rep.negative_control_fired)
      self.assertEqual(rep.negative_control_metrics["status"], "FIRED")
      self.assertEqual(rep.backward_metrics["max_d_doc2_d_embed_doc1"], 0.0)
      self.assertEqual(rep.backward_metrics["max_d_total_d_embed_pad"], 0.0)
      self.assertGreater(rep.backward_metrics["norm_d_doc2_d_embed_doc2"], 0.0)
      self.assertGreater(rep.training_metrics["total_update_l2"], 0.0)
      self.assertEqual(len(rep.training_metrics["packed_losses"]), 3)

  def test_packing_parity_passes_with_probe_token_id_fallback(self) -> None:
    params, fwd_fn, _, _ = packing_train_check.build_synthetic_causal_transformer(
        vocab_size=96, emb_dim=24, num_heads=4, mlp_dim=36, num_layers=2
    )
    rep = packing_train_check.verify_packed_training_parity(
        forward_fn=fwd_fn,
        params=params,
        vocab_size=96,
        batch_size=1,
        doc_lengths=(3, 2),
        pad_gaps=(1, 2),
        num_steps=2,
        embed_fn=None,
        forward_from_embeds_fn=None,
        seed=11,
        run_negative_control=True,
    )
    self.assertTrue(rep.passed, rep.failures)
    self.assertTrue(rep.negative_control_fired)
    self.assertEqual(rep.backward_metrics["max_d_doc2_d_embed_doc1"], 0.0)

  def test_packing_parity_catches_leaky_unmasked_attention_model(self) -> None:
    params, fwd_fn, emb_fn, fwd_emb_fn = packing_train_check.build_synthetic_causal_transformer(
        vocab_size=64, emb_dim=24, num_heads=4, mlp_dim=36, num_layers=1
    )

    def leaky_fwd(p: Any, tok: jax.Array, pos: jax.Array, seg: jax.Array) -> jax.Array:
      unmasked_seg = jnp.where(seg > 0, jnp.int32(1), jnp.int32(0))
      return fwd_fn(p, tok, pos, unmasked_seg)

    def leaky_emb_fwd(p: Any, emb: jax.Array, pos: jax.Array, seg: jax.Array) -> jax.Array:
      unmasked_seg = jnp.where(seg > 0, jnp.int32(1), jnp.int32(0))
      return fwd_emb_fn(p, emb, pos, unmasked_seg)

    rep = packing_train_check.verify_packed_training_parity(
        forward_fn=leaky_fwd,
        params=params,
        vocab_size=64,
        doc_lengths=(3, 2),
        pad_gaps=(1, 2),
        num_steps=2,
        embed_fn=emb_fn,
        forward_from_embeds_fn=leaky_emb_fwd,
        run_negative_control=True,
    )
    self.assertFalse(rep.passed)
    joined = "\n".join(rep.failures)
    self.assertIn("forward", joined)
    self.assertIn("backward", joined)
    self.assertIn("training", joined)

  def test_packing_parity_fails_closed_when_negative_control_cannot_fire(
      self,
  ) -> None:
    rng = np.random.default_rng(0)
    params = {
        "tok_embeddings": jnp.asarray(rng.standard_normal((64, 16)), dtype=jnp.float32),
        "lm_head": jnp.asarray(rng.standard_normal((16, 64)), dtype=jnp.float32),
    }

    def positionwise_fwd(p: Any, tok: jax.Array, pos: jax.Array, seg: jax.Array) -> jax.Array:
      del pos
      valid = (seg != 0).astype(jnp.float32)[:, :, None]
      h = p["tok_embeddings"][tok] * valid
      return jnp.einsum("bsd,dv->bsv", h, p["lm_head"])

    rep = packing_train_check.verify_packed_training_parity(
        forward_fn=positionwise_fwd,
        params=params,
        vocab_size=64,
        doc_lengths=(3, 2),
        pad_gaps=(1, 2),
        num_steps=2,
        run_negative_control=True,
    )
    self.assertFalse(rep.passed)
    self.assertFalse(rep.negative_control_fired)
    self.assertEqual(rep.negative_control_metrics["status"], "MISSED")

  def test_packing_cli_synthetic_and_json_report(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      out_json = os.path.join(tmp, "packing_report.json")
      rc = packing_train_check.main(
          [
              "--doc-lengths",
              "3,2",
              "--pad-gaps",
              "1,2",
              "--num-steps",
              "2",
              "--vocab-size",
              "64",
              "--json-out",
              out_json,
          ]
      )
      self.assertEqual(rc, 0)
      data = json.loads(pathlib.Path(out_json).read_text(encoding="utf-8"))
      self.assertTrue(data["passed"])
      self.assertTrue(data["negative_control_fired"])

  @unittest.skipUnless(
      importlib.util.find_spec("maxtext") is not None,
      "Requires MaxText on PYTHONPATH",
  )
  def test_packing_cli_against_maxtext_qwen3_default_flags(self) -> None:
    rc = packing_train_check.main(["--model-name=qwen3-0.6b"])
    self.assertEqual(rc, 0)

  @unittest.skipUnless(
      importlib.util.find_spec("maxtext") is not None,
      "Requires MaxText on PYTHONPATH",
  )
  def test_packing_cli_against_maxtext_m3_model(self) -> None:
    import flax  # pylint: disable=import-outside-toplevel

    flax_ver = tuple(int(p) for p in flax.__version__.split(".")[:3] if p.isdigit())
    cli_args = ["--model-name=qwen3-0.6b", "use_m3_model=true"]
    if flax_ver < (0, 12, 10):
      cli_args.append("base_num_decoder_layers=1")
    rc = packing_train_check.main(cli_args)
    self.assertEqual(rc, 0)


class InspectHfAndJaxNumericsTest(unittest.TestCase):
  """Tests for `inspect_hf_artifacts.py` and `check_jax_numerics_and_scale.py`."""

  def test_inspect_hf_artifacts_detects_nested_dtypes_and_source_hazards(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      cfg = {
          "model_type": "synthetic_hybrid",
          "text_config": {
              "hidden_size": 256,
              "num_hidden_layers": 9,
              "num_attention_heads": 96,
              "num_key_value_heads": 8,
              "rms_norm_eps": 1e-5,
              "moe_intermediate_size": 512,
          },
      }
      with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

      _write_safetensors_header(
          os.path.join(tmp, "model-00001-of-00001.safetensors"),
          {
              "model.embed_tokens.weight": {
                  "dtype": "BF16",
                  "shape": [1024, 256],
                  "data_offsets": [0, 0],
              },
              "model.layers.0.linear_attn.A_log": {
                  "dtype": "F32",
                  "shape": [128],
                  "data_offsets": [0, 0],
              },
              "model.layers.3.self_attn.q_proj.weight": {
                  "dtype": "BF16",
                  "shape": [256, 256],
                  "data_offsets": [0, 0],
              },
              "model.layers.7.self_attn.q_proj.weight": {
                  "dtype": "BF16",
                  "shape": [256, 256],
                  "data_offsets": [0, 0],
              },
              "model.layers.8.self_attn.q_proj.weight": {
                  "dtype": "BF16",
                  "shape": [256, 256],
                  "data_offsets": [0, 0],
              },
              "model.layers.1.mlp.experts.weight_packed": {
                  "dtype": "U8",
                  "shape": [16, 256, 128],
                  "data_offsets": [0, 0],
              },
          },
      )

      ref_py = (
          "from .configuration_synth import SynthConfig\n\n"
          "class SynthBlock:\n"
          "  def __init__(self, config):\n"
          "    self.outer_norm = RMSNorm(config.hidden_size,"
          " eps=config.rms_norm_eps)\n"
          "    self.inner_norm = RMSNorm(64)\n"
          "  def forward(self, x, g):\n"
          "    return (x.float() * g.to(torch.float32)).to(x.dtype)\n"
      )
      with open(os.path.join(tmp, "modeling_synth.py"), "w", encoding="utf-8") as f:
        f.write(ref_py)

      out_json = os.path.join(tmp, "report.json")
      rc = inspect_hf_artifacts.main(["--hf-dir", tmp, "--out", out_json])
      self.assertEqual(rc, 0)
      with open(out_json, "r", encoding="utf-8") as f:
        rep = json.load(f)

      self.assertEqual(rep["status"], "PASS")
      self.assertEqual(rep["config_summary"]["nested_subconfigs"], ["text_config"])
      inv = rep["tensor_inventory"]
      self.assertEqual(inv["dominant_dtype"], "BF16")
      self.assertEqual(len(inv["precision_sensitive_tensors"]), 1)
      self.assertEqual(
          inv["precision_sensitive_tensors"][0]["pattern"],
          "model.layers.{i}.linear_attn.A_log",
      )
      self.assertEqual(len(inv["quantized_packed_tensors"]), 1)
      irregular = [p for p in inv["patterns"] if "irregular_schedule_layers" in p]
      self.assertEqual(len(irregular), 1)
      self.assertEqual(irregular[0]["irregular_schedule_layers"], [3, 7, 8])

      hazards = rep["source_hazards"]
      self.assertTrue(hazards["requires_synthetic_package_loader"])
      self.assertEqual(len(hazards["omitted_norm_eps_calls"]), 1)
      self.assertEqual(len(hazards["explicit_fp32_casts"]), 2)

  def test_inspect_hf_artifacts_recursive_sources_and_precision_contract(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      inf_dir = os.path.join(tmp, "inference")
      os.makedirs(inf_dir)
      with open(os.path.join(inf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hidden_size": 512,
                "num_hidden_layers": 16,
                "num_attention_heads": 8,
                "qk_head_dim": 64,
                "n_routed_experts": 64,
                "num_experts_per_tok": 8,
            },
            f,
        )
      with open(os.path.join(inf_dir, "model.py"), "w", encoding="utf-8") as f:
        f.write(_REF_GATE_AND_ROPE_PY)

      rep = inspect_hf_artifacts.inspect_directory(tmp)
      self.assertTrue(rep["config_summary"]["present"])
      derived = rep["config_summary"]["derived_prod_dims"]
      self.assertEqual(derived["hidden"], 512)
      self.assertEqual(derived["H"], 8)
      self.assertEqual(derived["D"], 64)
      self.assertEqual(derived["experts"], 64)
      self.assertEqual(derived["topk"], 8)

      pc = rep["precision_contract"]
      self.assertTrue(pc["router_gate_fp32"])
      self.assertGreater(len(pc["router_gate_sites"]), 0)
      self.assertGreater(len(pc["rope_inv_freq_sites"]), 0)
      self.assertTrue(pc["mlp_or_hc_fp32"])

  def test_check_jax_numerics_and_scale_catches_hazards_and_passes_clean(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      bad_py = os.path.join(tmp, "bad_layer.py")
      with open(bad_py, "w", encoding="utf-8") as f:
        f.write(
            "import jax\n"
            "import jax.numpy as jnp\n\n"
            "def build_and_run(config, rng, x, mask):\n"
            "  w = nnx.initializers.lecun_normal()(rng, (64, 128, 256))\n"
            "  buf = jnp.empty((2, 4))\n"
            "  dim = getattr(config, 'head_dim', 64)\n"
            "  decay = jnp.where(mask, jnp.exp(x), 0.0)\n"
            "  return w, buf, dim, decay\n"
        )

      issues = check_jax_numerics_and_scale.scan_python_file(bad_py)
      codes = {i["code"] for i in issues}
      self.assertTrue(
          {
              "POST_EXP_MASK_NAN",
              "EXPERT_3D_INIT_FAN_IN",
              "UNINITIALIZED_EMPTY_BUFFER",
          }.issubset(codes),
          codes,
      )
      test_py = os.path.join(tmp, "kernel_dispatch_test.py")
      with open(test_py, "w", encoding="utf-8") as f:
        f.write("import jax.numpy as jnp\n\n" "def test_mask(mask, x):\n" "  return jnp.where(mask, jnp.exp(x), 0.0)\n")
      self.assertEqual(check_jax_numerics_and_scale.scan_python_file(test_py), [])

      rc_oom = check_jax_numerics_and_scale.main(
          [
              bad_py,
              "--prod-dims",
              "B=1,S=8192,H=64,C=64,D=128",
              "--shape-expr",
              "unscanned_wy=N_chunks*B*H*C*C*D:4",
          ]
      )
      self.assertEqual(rc_oom, 1)

      clean_py = os.path.join(tmp, "clean_layer.py")
      with open(clean_py, "w", encoding="utf-8") as f:
        f.write(
            "import jax.numpy as jnp\n\n"
            "def build_and_run(config, rng, x, mask):\n"
            "  init_fn = nnx.initializers.lecun_normal(batch_axis=(0,))\n"
            "  w = init_fn(rng, (64, 128, 256))\n"
            "  buf = jnp.zeros((2, 4))\n"
            "  dim = config.head_dim\n"
            "  safe_x = jnp.where(mask, x, 0.0)\n"
            "  decay = jnp.where(mask, jnp.exp(safe_x), 0.0)\n"
            "  return w, buf, dim, decay\n"
        )
      rc_ok = check_jax_numerics_and_scale.main(
          [
              clean_py,
              "--prod-dims",
              "B=1,S=8192,H=64,C=64,D=128",
              "--shape-expr",
              "scanned_wy=B*H*C*C*D:4",
          ]
      )
      self.assertEqual(rc_ok, 0)

  def test_check_jax_numerics_router_rope_and_segment_rules(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      buggy_yml = os.path.join(tmp, "buggy_moe.yml")
      omitted_yml = os.path.join(tmp, "omitted_moe.yml")
      fixed_yml = os.path.join(tmp, "fixed_moe.yml")
      pathlib.Path(buggy_yml).write_text(_BUGGY_MOE_CONFIG_YML, encoding="utf-8")
      pathlib.Path(omitted_yml).write_text(_OMITTED_MOE_CONFIG_YML, encoding="utf-8")
      pathlib.Path(fixed_yml).write_text(_FIXED_MOE_CONFIG_YML, encoding="utf-8")

      self.assertEqual(check_jax_numerics_and_scale.main([buggy_yml]), 1)
      self.assertEqual(check_jax_numerics_and_scale.main([omitted_yml]), 1)
      self.assertEqual(check_jax_numerics_and_scale.main([fixed_yml]), 0)

      buggy_rr = os.path.join(tmp, "buggy_rr.py")
      fixed_rr = os.path.join(tmp, "fixed_rr.py")
      pathlib.Path(buggy_rr).write_text(_BUGGY_ROUTER_AND_ROPE_PY, encoding="utf-8")
      pathlib.Path(fixed_rr).write_text(_FIXED_ROUTER_AND_ROPE_PY, encoding="utf-8")

      rr_codes = {i["code"] for i in check_jax_numerics_and_scale.scan_python_file(buggy_rr)}
      self.assertIn("ROUTER_BFLOAT16_LOGITS", rr_codes)
      self.assertIn("ROPE_INV_FREQ_LOW_PRECISION", rr_codes)
      self.assertEqual(check_jax_numerics_and_scale.scan_python_file(fixed_rr), [])

      buggy_sl = os.path.join(tmp, "buggy_sl.py")
      fixed_sl = os.path.join(tmp, "fixed_sl.py")
      pathlib.Path(buggy_sl).write_text(_BUGGY_SEGMENT_PY, encoding="utf-8")
      pathlib.Path(fixed_sl).write_text(_FIXED_SEGMENT_PY, encoding="utf-8")
      sl_codes = {i["code"] for i in check_jax_numerics_and_scale.scan_python_file(buggy_sl)}
      self.assertIn("MISSING_SEGMENT_MASK", sl_codes)
      self.assertEqual(check_jax_numerics_and_scale.scan_python_file(fixed_sl), [])

  def test_check_jax_numerics_diff_scoping_ignores_pre_existing_lines(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as repo:
      if not _init_git_repo(repo):
        return
      layer_py = os.path.join(repo, "layer.py")
      pathlib.Path(layer_py).write_text(
          "import jax.numpy as jnp\n\n"
          "def pre_existing_fn():\n"
          "  return jnp.empty((2, 4))\n\n"
          "def new_clean_fn(x):\n"
          "  return x + 1.0\n",
          encoding="utf-8",
      )
      _git_commit_all(repo, "Initial commit")
      pathlib.Path(layer_py).write_text(
          "import jax.numpy as jnp\n\n"
          "def pre_existing_fn():\n"
          "  return jnp.empty((2, 4))\n\n"
          "def new_clean_fn(x):\n"
          "  return jnp.zeros_like(x) + x\n",
          encoding="utf-8",
      )
      self.assertEqual(check_jax_numerics_and_scale.main([layer_py]), 1)
      self.assertEqual(
          check_jax_numerics_and_scale.main([layer_py, "--repo-root", repo, "--base", "HEAD"]),
          0,
      )

  def test_check_jax_numerics_base_subdir_untouched_bad_ref_and_merge_base(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as repo:
      if not _init_git_repo(repo):
        return
      _run_git(repo, "checkout", "-B", "main")
      pkg_dir = pathlib.Path(repo) / "src/pkg"
      pkg_dir.mkdir(parents=True)
      untouched_py = pkg_dir / "untouched_legacy.py"
      shared_py = pkg_dir / "shared.py"
      untouched_py.write_text(
          "import jax.numpy as jnp\n\ndef legacy_fn():\n  return jnp.empty((2, 4))\n",
          encoding="utf-8",
      )
      shared_py.write_text(
          "import jax.numpy as jnp\n\ndef shared_fn():\n  return jnp.empty((2, 2))\n",
          encoding="utf-8",
      )
      _git_commit_all(repo, "Base commit on main")

      _run_git(repo, "checkout", "-b", "feature")
      new_bad_py = pkg_dir / "new_bad.py"
      new_bad_py.write_text(
          "import jax.numpy as jnp\n\ndef new_fn():\n  return jnp.empty((4, 4))\n",
          encoding="utf-8",
      )
      _git_commit_all(repo, "Add new_bad.py on feature")

      _run_git(repo, "checkout", "main")
      shared_py.write_text(
          "import jax.numpy as jnp\n\ndef shared_fn():\n  return jnp.zeros((2, 2))\n",
          encoding="utf-8",
      )
      _git_commit_all(repo, "Clean shared.py independently on main")
      _run_git(repo, "checkout", "feature")

      # (a) Subdirectory path WITHOUT --repo-root must catch the newly added bug.
      self.assertEqual(
          check_jax_numerics_and_scale.main([str(new_bad_py), "--base", "main"]),
          1,
      )
      # (b) Untouched file with pre-existing violation must pass (0 diff lines).
      self.assertEqual(
          check_jax_numerics_and_scale.main([str(untouched_py), "--repo-root", repo, "--base", "main"]),
          0,
      )
      # (c) 3-dot merge-base(main, HEAD) must ignore upstream-only edits on main.
      self.assertEqual(
          check_jax_numerics_and_scale.main([str(shared_py), "--repo-root", repo, "--base", "main"]),
          0,
      )
      # (d) Invalid --base git ref must fail closed with exit code 2.
      self.assertEqual(
          check_jax_numerics_and_scale.main([str(untouched_py), "--repo-root", repo, "--base", "nonexistent_ref_xyz"]),
          2,
      )


class RouterAgreementAndE2eCheckTest(unittest.TestCase):
  """Tests for `e2e_check.py`, `negctl.py`, and `layer_parity.py`."""

  def test_router_agreement_passes_outside_sub_roundoff_tie(self) -> None:
    rng = np.random.default_rng(0)
    golden = rng.uniform(0.1, 2.0, size=(32, 64)).astype(np.float32)
    cand = golden.copy()
    sorted_idx = np.argsort(golden[0])[::-1]
    k_idx = sorted_idx[7]
    k_plus_1_idx = sorted_idx[8]
    val = golden[0, k_idx]
    ulp = float(np.spacing(np.float32(val)))
    golden[0, k_plus_1_idx] = val - 1.0 * ulp
    cand[0, k_idx] = val - 2.0 * ulp
    cand[0, k_plus_1_idx] = val

    ok, metrics = e2e_check.check_router_agreement(
        golden,
        cand,
        e2e_check.RouterAgreementTol(top_k=8, min_agreement=0.99, tie_margin_ulp=8.0),
    )
    self.assertTrue(ok, metrics)
    self.assertEqual(metrics["agreement"], 1.0)
    self.assertLess(metrics["non_tie_fraction"], 1.0)

  def test_router_agreement_fails_on_real_routing_flip(self) -> None:
    rng = np.random.default_rng(1)
    golden = rng.uniform(0.1, 2.0, size=(32, 64)).astype(np.float32)
    cand = golden.copy()
    for tok in range(4):
      sorted_idx = np.argsort(golden[tok])[::-1]
      cand[tok, sorted_idx[0]] = 0.0
      cand[tok, sorted_idx[-1]] = 5.0

    ok, metrics = e2e_check.check_router_agreement(
        golden,
        cand,
        e2e_check.RouterAgreementTol(top_k=8, min_agreement=0.99, tie_margin_ulp=8.0),
    )
    self.assertFalse(ok)
    self.assertLess(metrics["agreement"], 0.99)

  def test_negctl_and_pure_jax_layer_parity(self) -> None:
    params = {"proj": {"w": jnp.eye(4, dtype=jnp.float32)}}
    pert = negctl.perturb_one_leaf(params, target_path="proj/w", rel_scale=1e-2, seed=0)
    rel = float(jnp.linalg.norm(pert["proj"]["w"] - params["proj"]["w"]) / jnp.linalg.norm(params["proj"]["w"]))
    self.assertAlmostEqual(rel, 1e-2, places=4)
    with self.assertRaises(ValueError):
      negctl.perturb_one_leaf(params, target_path="proj/w", rel_scale=0.5, seed=0)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    ref_out = np.asarray(x @ params["proj"]["w"])
    tol = layer_parity.Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-5)
    fired, _ = layer_parity.negative_control_fires(
        ref_out=ref_out,
        cand_fn=lambda p, inp: inp @ p["proj"]["w"],
        cand_params=params,
        cand_input=x,
        fwd_tol=tol,
        target_path="proj/w",
        rel_scale=1e-2,
        seed=1,
    )
    self.assertTrue(fired)

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference")
  def test_layer_parity_pytorch_vs_jax_and_capture_golden(self) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(3)
    w_np = rng.standard_normal((6, 4)).astype(np.float32)
    x_np = rng.standard_normal((2, 4)).astype(np.float32)
    ref_params = {"linear.weight": torch.tensor(w_np)}
    ref_input = torch.tensor(x_np)
    cand_params = {"linear": {"kernel": jnp.asarray(w_np.T)}}
    cand_input = jnp.asarray(x_np)
    tol = layer_parity.Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-5)

    rep = layer_parity.check_parity(
        ref_fn=lambda p, inp: inp @ p["linear.weight"].T,
        ref_params=ref_params,
        ref_input=ref_input,
        cand_fn=lambda p, inp: inp @ p["linear"]["kernel"],
        cand_params=cand_params,
        cand_input=cand_input,
        weight_map={"linear.weight": "linear/kernel:T"},
        fwd_tol=tol,
        grad_tol=tol,
        cotangent_seed=0,
    )
    self.assertTrue(rep.passed, rep.failures)


# pylint: disable=missing-function-docstring
def _stats(
    temp_gib: float,
    arg_gib: float,
    out_gib: float,
    alias_gib: float,
    host_gib: float = 0.0,
) -> Any:
  return check_hlo_perf.AOTMemoryStats(
      temp_bytes=int(temp_gib * _GIB),
      argument_bytes=int(arg_gib * _GIB),
      output_bytes=int(out_gib * _GIB),
      alias_bytes=int(alias_gib * _GIB),
      host_temp_bytes=int(host_gib * _GIB),
  )


class CheckHloPerfTest(unittest.TestCase):
  """Tests for `check_hlo_perf.py`."""

  def test_hbm_estimate_naive_and_calibrated(self) -> None:
    stats = _stats(88.8, 11.8, 11.8, 11.8)
    naive = check_hlo_perf.compute_hbm_estimate(stats, runtime_reserve_gib=6.5)
    self.assertAlmostEqual(naive["naive_step_gib"], 100.6, places=3)
    self.assertAlmostEqual(naive["gating_peak_gib"], 107.1, places=3)
    self.assertEqual(naive["gating_source"], "naive_sum")
    calibrated = check_hlo_perf.compute_hbm_estimate(stats, runtime_reserve_gib=6.5, calibrated_load_gib=90.84)
    self.assertAlmostEqual(calibrated["gating_peak_gib"], 90.84, places=6)
    self.assertEqual(calibrated["gating_source"], "calibrated_load")
    with self.assertRaisesRegex(ValueError, "calibrated_load_gib"):
      check_hlo_perf.compute_hbm_estimate(stats, runtime_reserve_gib=6.5, calibrated_load_gib=0.0)

  def test_audit_follows_generic_async_chain_and_flags_sync_collective(
      self,
  ) -> None:
    report = check_hlo_perf.audit_hlo_text(_ASYNC_CHAIN_HLO.splitlines())
    self.assertEqual(report["async_collective_pairs"], 1)
    self.assertEqual(report["async_collective_overlapped"], 1)
    self.assertEqual(report["sync_collectives_by_kind"], {"all-to-all": 1})
    self.assertAlmostEqual(report["collective_overlap_ratio"], 0.5, places=6)
    self.assertEqual(report["unresolved_async_done"], 0)
    self.assertEqual(report["host_offload_copies"], 1)
    self.assertTrue(report["has_sparsecore_offload"])
    self.assertEqual(report["bf16_dot_count"], 1)

  def test_audit_back_to_back_async_pair_is_not_overlapped(self) -> None:
    hlo = """
ENTRY %main (x: f32[8]) -> f32[64] {
  %x = f32[8]{0} parameter(0)
  %all-gather-start.1 = (f32[8]{0}, f32[64]{0}) all-gather-start(%x), dimensions={0}
  %all-gather-done.1 = f32[64]{0} all-gather-done(%all-gather-start.1)
  ROOT %neg = f32[64]{0} negate(%all-gather-done.1)
}
"""
    report = check_hlo_perf.audit_hlo_text(hlo.splitlines())
    self.assertEqual(report["async_collective_pairs"], 1)
    self.assertEqual(report["async_collective_overlapped"], 0)

  def test_gate_fails_on_unresolved_done_and_overlap(self) -> None:
    hlo = """
ENTRY %main (x: f32[8]) -> f32[64] {
  %x = f32[8]{0} parameter(0)
  ROOT %all-gather.9.call-done = f32[64]{0} async-done(%missing.call-start)
}
"""
    ok, _, failures = check_hlo_perf.check_hlo_gate(
        mem_stats=_stats(30, 15, 5, 5),
        compiled_hlo_flops=110.0,
        canonical_model_flops=100.0,
        hlo_lines=hlo.splitlines(),
        cfg=check_hlo_perf.HLOGateConfig(hbm_ceiling_gib=92.0),
    )
    self.assertFalse(ok)
    self.assertIn("could not be matched", "\n".join(failures))

  def test_gate_detects_oom_inflation_spill_host_offload_and_missing_sparsecore(
      self,
  ) -> None:
    hlo = """
ENTRY %main (x: f32[8]) -> f32[64] {
  %x = f32[8]{0} parameter(0)
  %copy-start.1 = (f32[8]{0:S(5)}, f32[8]{0}, u32[]) copy-start(%x)
  %copy-done.1 = f32[8]{0:S(5)} copy-done(%copy-start.1)
  ROOT %d = f32[64]{0} dot(%x, %x), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
"""
    cfg = check_hlo_perf.HLOGateConfig(
        hbm_ceiling_gib=92.0,
        require_sparsecore_offload=True,
        forbid_host_offload_copies=True,
    )
    ok, metrics, failures = check_hlo_perf.check_hlo_gate(
        mem_stats=_stats(80, 15, 5, 5, host_gib=2),
        compiled_hlo_flops=180.0,
        canonical_model_flops=100.0,
        hlo_lines=hlo.splitlines(),
        cfg=cfg,
    )
    self.assertFalse(ok)
    joined = "\n".join(failures)
    for needle in (
        "exceeds ceiling",
        "FLOP inflation",
        "host memory spill",
        "S(5)",
        "SparseCore offload required",
    ):
      self.assertIn(needle, joined)
    self.assertEqual(metrics["fp32_dot_count"], 1)

  def test_gate_passes_on_calibrated_load_with_warning(self) -> None:
    ok, metrics, failures = check_hlo_perf.check_hlo_gate(
        mem_stats=_stats(88.8, 11.8, 11.8, 11.8),
        compiled_hlo_flops=106.7,
        canonical_model_flops=100.0,
        hlo_lines=_ASYNC_CHAIN_HLO.splitlines(),
        cfg=check_hlo_perf.HLOGateConfig(hbm_ceiling_gib=94.74, require_sparsecore_offload=True),
        calibrated_load_gib=90.84,
    )
    self.assertTrue(ok, failures)
    self.assertEqual(len(metrics["warnings"]), 1)

  def test_check_loss_trajectory(self) -> None:
    base = [10.512, 10.204, 9.981, 9.765, 9.540]
    cand_ok = [10.515, 10.200, 9.985, 9.760, 9.544]
    cand_bad = [10.515, 10.200, 9.985, 10.100, 9.544]
    self.assertTrue(check_hlo_perf.check_loss_trajectory(base, cand_ok, max_rel_delta=0.01)[0])
    self.assertFalse(check_hlo_perf.check_loss_trajectory(base, cand_bad, max_rel_delta=0.01)[0])
    self.assertFalse(
        check_hlo_perf.check_loss_trajectory(base, cand_ok, max_rel_delta=0.01, ragged_buffer_factor=1.25)[0]
    )


class ExtractRunMetricsTest(unittest.TestCase):
  """Tests for `extract_run_metrics.py`."""

  def test_extract_step_metrics_and_verify_prose(self) -> None:
    log_text = "\n".join(
        [
            "Total TFLOPs: 520.0",
            "completed step: 0, seconds: 45.000, TFLOP/s/device: 11.55, Tokens/s/device: 91.0, loss: 10.500",
            "completed step: 1, seconds: 20.000, TFLOP/s/device: 26.00, Tokens/s/device: 204.8, loss: 10.200",
            "completed step: 2, seconds: 10.000, TFLOP/s/device: 52.00, Tokens/s/device: 409.6, loss: 9.950",
            "completed step: 3, seconds: 10.200, TFLOP/s/device: 50.98, Tokens/s/device: 401.5, loss: 9.800",
            "completed step: 4, seconds: 9.800, TFLOP/s/device: 53.06, Tokens/s/device: 417.9, loss: 9.650",
        ]
    )
    summary = extract_run_metrics.extract_step_metrics(
        log_text,
        steady_start_step=2,
        steady_end_step=4,
        peak_tflops_per_device=1000.0,
    )
    self.assertEqual(summary["num_steps_steady"], 3)
    self.assertAlmostEqual(summary["mean_step_seconds"], 10.0, places=4)
    self.assertAlmostEqual(summary["achieved_tflops_per_sec_per_device"], 52.0, places=4)
    self.assertAlmostEqual(summary["mfu_percent"], 5.2, places=3)

    prose_ok = "Steady-state step time is 10.0 s (52.0 TFLOP/s/device, 5.2% MFU)."
    self.assertTrue(extract_run_metrics.verify_prose_numbers(prose_ok, summary)[0])
    prose_bad = "Steady-state step time is 10.0 s (58.5 TFLOP/s/device, 5.2% MFU)."
    self.assertFalse(extract_run_metrics.verify_prose_numbers(prose_bad, summary)[0])

  def test_compute_eta_seconds_and_grad_norm_screen(self) -> None:
    eta = extract_run_metrics.compute_eta_seconds([12.0, 14.0, 16.0], remaining_items=4, concurrency=2)
    self.assertAlmostEqual(eta, 28.0, places=4)

    def _diag_log(grad_norms: list[float | None]) -> str:
      lines = []
      for step, g in enumerate(grad_norms):
        diag = "n/a" if g is None else f"{g:.9e}"
        lines.append(
            f"completed step: {step}, seconds: 10.000, TFLOP/s/device: 52.00, "
            f"Tokens/s/device: 409.6, loss: 9.950, diag: grad_norm={diag}"
        )
      return "\n".join(lines)

    ref_a = extract_run_metrics.extract_step_metrics(_diag_log([3.0, 2.0, 1.5, 1.2]))["grad_norms"]
    ref_b = extract_run_metrics.extract_step_metrics(_diag_log([3.0, 2.00002, 1.5, 1.2]))["grad_norms"]
    floor = extract_run_metrics.max_relative_grad_norm_delta(ref_a, ref_b)
    self.assertAlmostEqual(floor, 1e-5, places=9)
    missing = extract_run_metrics.extract_step_metrics(_diag_log([3.0, 2.0, None, 1.2]))["grad_norms"]
    with self.assertRaises(ValueError):
      extract_run_metrics.max_relative_grad_norm_delta(ref_a, missing)


class CkptRoundtripAndPreflightHelpersTest(unittest.TestCase):
  """Tests for `ckpt_roundtrip_check.py`, `aot_preflight.py`, `slice_hf_mini.py`, and `family_diff.py`."""

  def test_exact_roundtrip_and_defect_detection(self) -> None:
    rng = np.random.default_rng(7)
    source = {
        "model.embed.weight": rng.standard_normal((16, 32)).astype(np.float16),
        "model.layers.0.router.bias": (rng.standard_normal((8,)).astype(np.float32)),
        "model.layers.0.router.table": np.arange(16, dtype=np.int64),
        "model.layers.0.proj.weight_scale": np.ones((4,), dtype=np.float32),
    }
    exported = {k: v.copy() for k, v in source.items() if k != "model.layers.0.proj.weight_scale"}
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(
        source,
        exported,
        tol=ckpt_roundtrip_check.RoundTripTol.exact(),
        unexported_source_keys=("model.layers.0.proj.weight_scale",),
    )
    self.assertTrue(ok, failures)

    perturbed = ckpt_roundtrip_check.perturb_one_tensor(
        exported, target_key="model.layers.0.router.bias", rel_scale=1e-2, seed=3
    )
    status, reasons = ckpt_roundtrip_check.negative_roundtrip_status(
        source,
        exported,
        perturbed,
        tol=ckpt_roundtrip_check.RoundTripTol.exact(),
        unexported_source_keys=("model.layers.0.proj.weight_scale",),
    )
    self.assertEqual(status, "FIRED", reasons)

  def test_aot_preflight_topology_limits_and_argv_split(self) -> None:
    self.assertEqual(aot_preflight.infer_hbm_limit_gib("v5p-128"), 94.74)
    self.assertEqual(aot_preflight.infer_hbm_limit_gib("tpu7x-64"), 191.5)
    self.assertEqual(aot_preflight.infer_hbm_limit_gib("v6e-16"), 31.2)
    mt_argv, own = aot_preflight.split_argv(
        [
            "aot_preflight.py",
            "src/maxtext/configs/base.yml",
            "--model-name=qwen3-0.6b",
            "--topology=v5p-8",
            "--seqs=128,256",
        ]
    )
    self.assertIn("model_name=qwen3-0.6b", mt_argv)
    self.assertIn("compile_topology=v5p-8", mt_argv)
    self.assertEqual(own["--seqs"], "128,256")

  def test_aot_preflight_defaults_num_slices_and_runs_mini_compile(self) -> None:
    mt_argv, _ = aot_preflight.split_argv(
        [
            "aot_preflight.py",
            "src/maxtext/configs/base.yml",
            "model_name=qwen3-0.6b",
            "compile_topology=v5p-8",
            "--seqs=128",
        ]
    )
    self.assertIn("compile_topology_num_slices=1", mt_argv)

    mt_argv_explicit, _ = aot_preflight.split_argv(
        [
            "aot_preflight.py",
            "src/maxtext/configs/base.yml",
            "model_name=qwen3-0.6b",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=2",
            "--seqs=128",
        ]
    )
    self.assertIn("compile_topology_num_slices=2", mt_argv_explicit)
    self.assertNotIn("compile_topology_num_slices=1", mt_argv_explicit)

    if importlib.util.find_spec("maxtext") is not None:
      with self.assertRaises(SystemExit) as cm:
        aot_preflight.main(
            [
                str(SCRIPTS_DIR / "aot_preflight.py"),
                str(REPO_ROOT / "src/maxtext/configs/base.yml"),
                "model_name=qwen3-0.6b",
                "override_model_config=true",
                "base_num_decoder_layers=1",
                "base_emb_dim=64",
                "base_mlp_dim=128",
                "base_num_query_heads=4",
                "base_num_kv_heads=2",
                "head_dim=16",
                "vocab_size=128",
                "compile_topology=v5p-8",
                "--seqs=128",
            ]
        )
      self.assertEqual(cm.exception.code, 0)

  def test_verify_checkpoint_parse_args_and_doc_cli_consistency(self) -> None:
    flagged = verify_checkpoint.parse_args(
        [
            "--checkpoint-path",
            "/tmp/ckpt/0/items",
            "--model-params-b",
            "7",
            "--expected-layers",
            "28",
        ]
    )
    self.assertEqual(flagged.checkpoint_path, "/tmp/ckpt/0/items")
    self.assertEqual(flagged.model_params_b, 7.0)
    self.assertEqual(flagged.expected_layers, 28)

    positional = verify_checkpoint.parse_args(
        [
            "/tmp/ckpt/0/items",
            "--model-params-b",
            "7",
            "--expected-layers",
            "28",
        ]
    )
    self.assertEqual(positional.checkpoint_path, "/tmp/ckpt/0/items")

    ckpt_doc = (SKILL_ROOT / "references/checkpoint_conversion.md").read_text(encoding="utf-8")
    self.assertIn(
        "--checkpoint-path /dev/shm/$USER/checkpoints/{model_name}_orbax/0/items",
        ckpt_doc,
    )

  def test_verify_checkpoint_narrow_exceptions_and_expected_fallbacks(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      items_dir = pathlib.Path(tmp) / "0/items"
      items_dir.mkdir(parents=True)
      (items_dir / "payload.bin").write_bytes(b"x" * 1024)
      ckpt_arg = str(items_dir)

      for expected_exc in (
          FileNotFoundError("du missing"),
          subprocess.CalledProcessError(1, ["du", "-sb", tmp]),
      ):
        with mock.patch.object(verify_checkpoint.subprocess, "run", side_effect=expected_exc):
          self.assertTrue(verify_checkpoint.check_storage_size(ckpt_arg, 2000.0))

      with mock.patch.object(verify_checkpoint.subprocess, "run", side_effect=TypeError("unexpected du bug")):
        with self.assertRaises(TypeError):
          verify_checkpoint.check_storage_size(ckpt_arg, 2000.0)

      for expected_kv_exc in (
          ValueError("DATA_LOSS: corrupt manifest.ocdbt"),
          OSError("io read failure"),
      ):
        with mock.patch.object(verify_checkpoint, "_get_ts_kvstore", side_effect=expected_kv_exc):
          self.assertFalse(verify_checkpoint.check_layers_coverage(ckpt_arg, expected_layers=2))

      with mock.patch.object(
          verify_checkpoint, "_get_ts_kvstore", side_effect=AttributeError("unexpected layer check bug")
      ):
        with self.assertRaises(AttributeError):
          verify_checkpoint.check_layers_coverage(ckpt_arg, expected_layers=2)

      if importlib.util.find_spec("tensorstore") is not None:
        with mock.patch.object(verify_checkpoint, "_get_ts_kvstore", side_effect=ValueError("DATA_LOSS")):
          self.assertFalse(verify_checkpoint.check_sample_read(ckpt_arg))
        with mock.patch.object(verify_checkpoint, "_get_ts_kvstore", side_effect=TypeError("unexpected sample bug")):
          with self.assertRaises(TypeError):
            verify_checkpoint.check_sample_read(ckpt_arg)

  def test_slice_hf_mini_config_and_audit_ckpt_layer_parser(self) -> None:
    cfg = {
        "num_hidden_layers": 6,
        "layer_types": ["dense", "moe", "dense", "moe", "dense", "moe"],
        "first_k_dense_replace": 1,
    }
    key, pat, id_map = slice_hf_mini.parse_slice(r"num_hidden_layers=\.layers\.(\d+)\.:0,1,3")
    self.assertEqual(key, "num_hidden_layers")
    self.assertIsNotNone(pat.search("model.layers.3.self_attn.q_proj.weight"))
    changes, unresolved = slice_hf_mini.slice_config(cfg, key, id_map)
    self.assertEqual(cfg["num_hidden_layers"], 3)
    self.assertEqual(cfg["layer_types"], ["dense", "moe", "moe"])
    self.assertTrue(changes)
    self.assertEqual(unresolved, ["first_k_dense_replace"])
    self.assertEqual(audit_ckpt_vs_hf._parse_layers("0-2,5"), [0, 1, 2, 5])  # pylint: disable=protected-access

  def test_family_diff_symbol_comparison(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      old_py = pathlib.Path(tmp) / "old.py"
      new_py = pathlib.Path(tmp) / "new.py"
      old_py.write_text("class FooNorm:\n  def f(self, x):\n    return x\n")
      new_py.write_text("class BarNorm:\n  def f(self, x):\n    return x\n")
      syms = family_diff.diff_symbols(
          family_diff.extract_symbols([str(old_py)], "Foo"),
          family_diff.extract_symbols([str(new_py)], "Bar"),
      )
      self.assertEqual(syms[f"{family_diff.PREFIX_PLACEHOLDER}Norm"]["status"], "UNCHANGED")


if __name__ == "__main__":
  unittest.main()
