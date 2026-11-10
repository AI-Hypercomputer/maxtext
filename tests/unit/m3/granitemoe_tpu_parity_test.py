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

"""Real-weight `ibm-granite/granite-3.0-1b-a400m-instruct` TPU `v5p-8` parity & sharded training verification."""

from __future__ import annotations

from collections.abc import Mapping
import glob
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.configs import pyconfig
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.granitemoe import modeling_granitemoe
from maxtext.m3.models.granitemoe import utils_granitemoe
import ml_dtypes
import numpy as np
import optax

POC_ROOT = os.environ.get("GRANITEMOE_POC_DIR", "/dev/shm/jfacevedo_m3_poc")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASE_YML = os.path.join(REPO_ROOT, "src", "maxtext", "configs", "base.yml")

try:
  import pytest

  pytestmark = [pytest.mark.tpu_only, pytest.mark.scheduled_only]
except ImportError:
  pytest = None  # type: ignore[assignment]

try:
  import huggingface_hub
  from safetensors import safe_open
  import torch
  import transformers

  HAS_TORCH = True
except ImportError:
  huggingface_hub = None  # type: ignore[assignment]
  safe_open = None  # type: ignore[assignment]
  torch = None  # type: ignore[assignment]
  transformers = None  # type: ignore[assignment]
  HAS_TORCH = False

HF_REPO_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
HF_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
_ALL_GATHER_RE = re.compile(r"\ball-gather(?:-start|-done)?(?:\.\d+)?\b")
_HLO_SHAPE_RE = re.compile(r"[a-z0-9]+\[([0-9,]+)\]")


def _ensure_orbax_pathways_compat() -> None:
  """Provides a lightweight `pathways` stub on `orbax==0.12.8` before importing MaxText checkpoint helpers."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from orbax.checkpoint import v1 as ocp_v1

  if not hasattr(ocp_v1.training, "pathways"):
    ocp_v1.training.pathways = type("_StubPathways", (), {"Snapshotter": object})()  # type: ignore[attr-defined]


def _is_tpu_or_opted_in() -> bool:
  if os.environ.get("GRANITEMOE_RUN_REAL_WEIGHT_ON_CPU") == "1":
    return True
  try:
    return jax.default_backend() == "tpu"
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _flatten_nnx_vars(state: Any) -> list[tuple[str, Any]]:
  flat, _ = jax.tree_util.tree_flatten_with_path(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return [(".".join(str(getattr(p, "key", p)) for p in path), leaf) for path, leaf in flat]


def _rel_l2_error(got: np.ndarray, ref: np.ndarray) -> float:
  got_f64 = np.asarray(got, dtype=np.float64)
  ref_f64 = np.asarray(ref, dtype=np.float64)
  denom = max(float(np.linalg.norm(ref_f64)), 1e-12)
  return float(np.linalg.norm(got_f64 - ref_f64) / denom)


def _kl_divergence_fp64(hf_logits: np.ndarray, m3_logits: np.ndarray) -> tuple[float, float]:
  """Returns `(mean_kl, max_token_kl)` for `KL(P_hf || P_m3)` computed in `float64`."""
  hf_f64 = np.asarray(hf_logits, dtype=np.float64)
  m3_f64 = np.asarray(m3_logits, dtype=np.float64)
  hf_log_p = hf_f64 - np.max(hf_f64, axis=-1, keepdims=True)
  hf_log_p = hf_log_p - np.log(np.sum(np.exp(hf_log_p), axis=-1, keepdims=True))
  m3_log_p = m3_f64 - np.max(m3_f64, axis=-1, keepdims=True)
  m3_log_p = m3_log_p - np.log(np.sum(np.exp(m3_log_p), axis=-1, keepdims=True))
  hf_p = np.exp(hf_log_p)
  token_kl = np.sum(hf_p * (hf_log_p - m3_log_p), axis=-1)
  token_kl = np.maximum(token_kl, 0.0)
  return float(np.mean(token_kl)), float(np.max(token_kl))


def _load_raw_safetensors_numpy(hf_dir: str, target_dtype: str) -> dict[str, np.ndarray]:
  """Reads every tensor from `hf_dir/*.safetensors` into a host NumPy dict cast to `target_dtype`."""
  tgt_dt = np.dtype(ml_dtypes.bfloat16) if target_dtype == "bfloat16" else np.dtype(np.float32)
  tensors: dict[str, np.ndarray] = {}
  for sf_file in sorted(glob.glob(os.path.join(hf_dir, "*.safetensors"))):
    with safe_open(sf_file, framework="pt", device="cpu") as reader:
      for key in reader.keys():
        t_cpu = reader.get_tensor(key).detach().cpu().contiguous()
        if t_cpu.dtype == torch.bfloat16:
          arr = t_cpu.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        else:
          arr = t_cpu.numpy()
        tensors[key] = arr.astype(tgt_dt) if arr.dtype != tgt_dt else np.array(arr, copy=True)
  return tensors


def _run_centralized_to_maxtext(hf_dir: str, out_dir: str, *, save_dtype: str) -> None:
  """Runs centralized `maxtext.checkpoint_conversion.to_maxtext` in an isolated CPU subprocess."""
  shutil.rmtree(out_dir, ignore_errors=True)
  runner = (
      "import runpy\n"
      "from orbax.checkpoint import v1 as ocp_v1\n"
      "if not hasattr(ocp_v1.training, 'pathways'):\n"
      "  ocp_v1.training.pathways = type('_StubPathways', (), {'Snapshotter': object})()\n"
      "runpy.run_module('maxtext.checkpoint_conversion.to_maxtext', run_name='__main__', alter_sys=True)\n"
  )
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  src_dir = os.path.join(REPO_ROOT, "src")
  env["PYTHONPATH"] = f"{src_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
  cmd = [
      sys.executable,
      "-c",
      runner,
      BASE_YML,
      "model_name=granite-3.0-1b-a400m",
      f"base_output_directory={out_dir}",
      "scan_layers=true",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      f"--hf_model_path={hf_dir}",
      f"--save_dtype={save_dtype}",
      "--lazy_load_tensors=true",
      "--simulated_cpu_devices_count=16",
  ]
  proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise RuntimeError(
        f"to_maxtext ({save_dtype}) failed with rc={proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _load_sharded_m3_from_orbax(
    orbax_dir: str,
    mesh: jax.sharding.Mesh,
    *,
    dtype: str,
    weight_dtype: str,
    scan_layers: bool = True,
    remat_policy: str = "none",
    matmul_precision: str = "highest",
) -> tuple[modeling_granitemoe.GraniteMoeModel, Any]:
  """Restores an Orbax checkpoint written by `to_maxtext` onto `mesh` via `model_creation_utils.from_pretrained`."""
  _ensure_orbax_pathways_compat()
  from maxtext.utils import model_creation_utils  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  items_path = os.path.join(orbax_dir, "0", "items")
  mt_cfg = pyconfig.initialize(
      [None, BASE_YML],
      model_name="granite-3.0-1b-a400m",
      load_parameters_path=items_path,
      dtype=dtype,
      weight_dtype=weight_dtype,
      scan_layers=scan_layers,
      remat_policy=remat_policy,
      matmul_precision=matmul_precision,
      skip_jax_distributed_system=True,
  )
  res = model_creation_utils.from_pretrained(mt_cfg, mesh=mesh)
  model = res[0] if isinstance(res, tuple) else res
  if not isinstance(model, modeling_granitemoe.GraniteMoeModel):
    raise TypeError(f"Expected GraniteMoeModel from from_pretrained, got {type(model)!r}.")
  return model, mt_cfg


def _forbidden_expert_shapes(
    model: modeling_granitemoe.GraniteMoeModel,
    tp_size: int = 1,
) -> dict[tuple[int, ...], str]:
  """Collects unsharded/TP-only expert weight shapes that must never appear in HLO all-gathers."""
  forbidden: dict[tuple[int, ...], str] = {}
  n_layers = model.config.num_hidden_layers
  n_exp = model.config.num_local_experts
  h_dim = model.config.hidden_size
  i_dim = model.config.intermediate_size

  def _register(unstacked: tuple[int, ...], label: str) -> None:
    forbidden[(n_layers, *unstacked)] = label
    forbidden[unstacked] = label
    forbidden[(1, *unstacked)] = label
    if len(unstacked) == 4:
      d0, d1, d2, d3 = unstacked
      for collapsed in ((d0, d1, d2 * d3), (d0, d1 * d2, d3)):
        forbidden[collapsed] = label
        forbidden[(1, *collapsed)] = label
        forbidden[(n_layers, *collapsed)] = label

  for path, var in _flatten_nnx_vars(nnx.state(model, nnx.Param)):
    if ".experts." not in path:
      continue
    _register(var.get_raw_value().shape[1:], path)

  if tp_size > 1 and i_dim % tp_size == 0:
    i_tp = i_dim // tp_size
    _register((n_exp, h_dim, 2, i_tp), "layers.block_sparse_moe.experts.gate_up_proj[tp_shard]")
    _register((n_exp, i_tp, h_dim), "layers.block_sparse_moe.experts.down_proj[tp_shard]")
  return forbidden


def _assert_zero_expert_all_gather(hlo_text: str, forbidden: Mapping[tuple[int, ...], str], tag: str) -> int:
  """Asserts compiled HLO contains zero expert-weight all-gather ops and returns total all-gather count."""
  all_gather_count = 0
  for line in hlo_text.splitlines():
    stripped = line.strip()
    if stripped.startswith("//"):
      continue
    m = _ALL_GATHER_RE.search(stripped)
    if m is None:
      continue
    all_gather_count += 1
    lhs = stripped[: m.start()]
    for dims_str in _HLO_SHAPE_RE.findall(lhs):
      shape = tuple(int(d) for d in dims_str.split(",") if d.strip())
      if shape in forbidden:
        raise AssertionError(f"[{tag}] Forbidden expert all-gather {forbidden[shape]} {shape}: {stripped}")
  return all_gather_count


@unittest.skipUnless(HAS_TORCH, "PyTorch, transformers, safetensors, and huggingface_hub required for TPU parity.")
class GraniteMoeRealWeightTpuParityTest(absltest.TestCase):
  """End-to-end real-weight verification for `ibm-granite/granite-3.0-1b-a400m-instruct` on TPU `v5p-8`."""

  hf_dir: str
  orbax_bf16_dir: str
  orbax_fp32_dir: str
  hf_tokenizer: Any
  hf_model_fp32: Any
  _prev_matmul_precision: Any = None

  @classmethod
  def setUpClass(cls):
    """Downloads real Granite-3.0-1B-A400M weights into /dev/shm and converts them via centralized `to_maxtext`."""
    super().setUpClass()
    if not _is_tpu_or_opted_in():
      raise unittest.SkipTest(
          "Real-weight 1.3B TPU parity suite requires TPU backend (or GRANITEMOE_RUN_REAL_WEIGHT_ON_CPU=1)."
      )
    if os.environ.get("GRANITEMOE_RUN_REAL_WEIGHT_ON_CPU") != "1" and len(jax.devices()) < 4:
      raise unittest.SkipTest(
          f"Real-weight v5p-8 parity suite requires >= 4 TPU devices (4 chips / 8 cores); got {len(jax.devices())}."
      )

    cls._prev_matmul_precision = getattr(jax.config, "jax_default_matmul_precision", None)
    jax.config.update("jax_default_matmul_precision", "highest")

    os.makedirs(POC_ROOT, exist_ok=True)
    cls.hf_dir = os.path.join(POC_ROOT, "hf_granite_3_0_1b_a400m")
    cls.orbax_bf16_dir = os.path.join(POC_ROOT, "orbax_granite_3_0_1b_a400m_bf16")
    cls.orbax_fp32_dir = os.path.join(POC_ROOT, "orbax_granite_3_0_1b_a400m_fp32")

    huggingface_hub.snapshot_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        local_dir=cls.hf_dir,
        cache_dir=os.path.join(POC_ROOT, "hf_cache"),
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*", "special_tokens*"],
    )

    _run_centralized_to_maxtext(cls.hf_dir, cls.orbax_bf16_dir, save_dtype="bfloat16")
    _run_centralized_to_maxtext(cls.hf_dir, cls.orbax_fp32_dir, save_dtype="float32")

    cls.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(cls.hf_dir)
    cls.hf_model_fp32 = transformers.AutoModelForCausalLM.from_pretrained(
        cls.hf_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

  def test_1_full_real_checkpoint_conversion_and_100pct_leaf_audit(self):
    _ensure_orbax_pathways_compat()
    from maxtext.checkpoint_conversion.utils import utils as conversion_utils  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    cfg = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(self.hf_dir)
    self.assertEqual(cfg.vocab_size, 49155)
    self.assertEqual(cfg.hidden_size, 1024)
    self.assertEqual(cfg.intermediate_size, 512)
    self.assertEqual(cfg.num_hidden_layers, 24)
    self.assertEqual(cfg.num_attention_heads, 16)
    self.assertEqual(cfg.num_key_value_heads, 8)
    self.assertEqual(cfg.num_local_experts, 32)
    self.assertEqual(cfg.num_experts_per_tok, 8)
    self.assertTrue(cfg.tie_word_embeddings)
    self.assertAlmostEqual(cfg.embedding_multiplier, 12.0)
    self.assertAlmostEqual(cfg.attention_multiplier, 0.015625)
    self.assertAlmostEqual(cfg.residual_multiplier, 0.22)
    self.assertAlmostEqual(cfg.logits_scaling, 6.0)

    expected_hf_keys = 2 + 24 * 9  # 218 safetensors keys
    mesh_4 = m3_sharding.create_mesh(ici={"fsdp": 2, "expert": 2}, devices=list(jax.devices())[:4])
    hf_cfg_dict = hf_model_configs.HF_MODEL_CONFIGS["granite-3.0-1b-a400m"].to_dict()
    expected_shapes = hf_shape.HF_SHAPE["granite-3.0-1b-a400m"](hf_cfg_dict)

    max_diffs: dict[str, float] = {}
    total_params = 0
    for tag, dt_name, ckpt_dir in (
        ("bf16", "bfloat16", self.orbax_bf16_dir),
        ("fp32", "float32", self.orbax_fp32_dir),
    ):
      model, mt_cfg = _load_sharded_m3_from_orbax(ckpt_dir, mesh_4, dtype=dt_name, weight_dtype=dt_name)
      flat_leaves = _flatten_nnx_vars(nnx.state(model, nnx.Param))
      self.assertLen(flat_leaves, 11, f"[{tag}] Expected 11 stacked M3 Param leaves")
      total_params = sum(math.prod(v.get_raw_value().shape) for _, v in flat_leaves)

      raw_hf = _load_raw_safetensors_numpy(self.hf_dir, dt_name)
      self.assertLen(raw_hf, expected_hf_keys, f"[{tag}] Unexpected raw safetensors key count")

      pmap = param_mapping.PARAM_MAPPING["granite-3.0-1b-a400m"](hf_cfg_dict, mt_cfg, scan_layers=True)
      hooks_hf = param_mapping.HOOK_FNS["granite-3.0-1b-a400m"](hf_cfg_dict, mt_cfg, scan_layers=True, saving_to_hf=True)
      roundtrip_hf: dict[str, np.ndarray] = {}
      for path, var in flat_leaves:
        mt_key = f"params-{path.replace('.', '-')}"
        host_jax_arr = jnp.asarray(jax.device_get(var.get_raw_value()))
        for hf_k, hf_v in conversion_utils.process_maxtext_param(
            mt_key, host_jax_arr, pmap, hooks_hf, expected_shapes, mt_cfg
        ):
          roundtrip_hf[hf_k] = np.asarray(hf_v)

      self.assertEqual(set(roundtrip_hf.keys()), set(raw_hf.keys()), f"[{tag}] HF roundtrip key mismatch")
      tag_max_diff = 0.0
      for hf_k, ref_arr in raw_hf.items():
        got_arr = roundtrip_hf[hf_k]
        self.assertEqual(got_arr.shape, ref_arr.shape, f"[{tag}] Shape mismatch at {hf_k}")
        diff = float(np.max(np.abs(got_arr.astype(np.float32) - ref_arr.astype(np.float32))))
        tag_max_diff = max(tag_max_diff, diff)
      self.assertEqual(tag_max_diff, 0.0, f"[{tag}] Non-zero centralized roundtrip diff: {tag_max_diff}")
      max_diffs[tag] = tag_max_diff

    hf_total_params = sum(int(p.numel()) for p in self.hf_model_fp32.parameters())
    self.assertEqual(total_params, 1_334_628_352)
    self.assertEqual(total_params, hf_total_params)

    print(
        f"\n[PHASE3-1 CENTRALIZED CHECKPOINT AUDIT] repo={HF_REPO_ID} poc_dir={POC_ROOT} "
        f"hf_keys=218/218 m3_stacked_leaves=11/11 total_params={total_params:,} "
        f"bf16_roundtrip_max_abs_diff={max_diffs['bf16']:.1e} "
        f"fp32_roundtrip_max_abs_diff={max_diffs['fp32']:.1e}",
        flush=True,
    )

  def test_2_real_weight_tpu_v5p_frozen_topk_and_natural_routing_parity(self):
    prompts = [
        "Explain how sparse Mixture-of-Experts transformers route tokens across specialized feed-forward experts.",
        "In distributed TPU training, sharding expert weights across FSDP and expert axes eliminates weight all-gathers.",
    ]
    enc = self.hf_tokenizer(prompts, padding=False, truncation=True, max_length=24)
    min_len = min(len(ids) for ids in enc["input_ids"])
    input_ids_pt = torch.tensor([ids[:min_len] for ids in enc["input_ids"]], dtype=torch.long)
    bsz, seq_len = input_ids_pt.shape

    captured_router_logits: list[np.ndarray] = []
    captured_topk_indices: list[np.ndarray] = []
    hooks: list[Any] = []
    for layer in self.hf_model_fp32.model.layers:

      def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        logits_t = out
        if isinstance(out, tuple):
          logits_t = next(
              item
              for item in out
              if torch.is_tensor(item) and torch.is_floating_point(item) and item.ndim == 2 and item.shape[-1] == 32
          )
        r_logits = logits_t.detach().cpu().numpy()  # (B*S, E)
        tk_idx = torch.topk(logits_t.detach(), k=8, dim=-1).indices.cpu().numpy()  # (B*S, 8)
        captured_router_logits.append(r_logits.reshape(bsz, seq_len, 32))
        captured_topk_indices.append(tk_idx.reshape(bsz, seq_len, 8))

      hooks.append(layer.block_sparse_moe.router.register_forward_hook(_hook))

    try:
      with torch.no_grad():
        hf_out = self.hf_model_fp32(input_ids=input_ids_pt)
        hf_logits = hf_out.logits.detach().cpu().numpy()
    finally:
      for h in hooks:
        h.remove()

    hf_router_all = np.stack(captured_router_logits, axis=0)  # (24, B, S, 32)
    hf_topk_all = np.stack(captured_topk_indices, axis=0).astype(np.int32)  # (24, B, S, 8)

    tokens_jax = jnp.asarray(input_ids_pt.numpy(), dtype=jnp.int32)
    frozen_jax = jnp.asarray(hf_topk_all, dtype=jnp.int32)

    for mesh_tag, ici in (("v5p_fsdp2_tp2", {"fsdp": 2, "tensor": 2}), ("v5p_fsdp2_ep2", {"fsdp": 2, "expert": 2})):
      mesh = m3_sharding.create_mesh(ici=ici, devices=list(jax.devices())[:4])
      model, _ = _load_sharded_m3_from_orbax(
          self.orbax_fp32_dir, mesh, dtype="float32", weight_dtype="float32", matmul_precision="highest"
      )

      @nnx.jit
      def _fwd_frozen(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array, frozen: jax.Array):
        return m.forward_with_details(toks, frozen_topk_indices=frozen)

      @nnx.jit
      def _fwd_natural(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array):
        return m.forward_with_details(toks)

      with jax.sharding.set_mesh(mesh):
        frozen_out = _fwd_frozen(model, tokens_jax, frozen_jax)
        perturbed_frozen_jax = (frozen_jax + 1) % 32
        perturbed_frozen_out = _fwd_frozen(model, tokens_jax, perturbed_frozen_jax)
        natural_out = _fwd_natural(model, tokens_jax)

      m3_frozen_logits = np.asarray(frozen_out.logits)
      m3_frozen_router = np.asarray(frozen_out.router_logits)
      frozen_mean_kl, frozen_max_kl = _kl_divergence_fp64(hf_logits, m3_frozen_logits)
      frozen_rel_l2 = _rel_l2_error(m3_frozen_logits, hf_logits)
      frozen_max_abs = float(np.max(np.abs(m3_frozen_logits - hf_logits)))
      layer0_router_rel_l2 = _rel_l2_error(m3_frozen_router[0], hf_router_all[0])

      self.assertLessEqual(frozen_mean_kl, 1e-5, f"[{mesh_tag}] FrozenTopK mean KL {frozen_mean_kl} > 1e-5")
      self.assertLessEqual(frozen_rel_l2, 1e-5, f"[{mesh_tag}] FrozenTopK logits rel_l2 {frozen_rel_l2} > 1e-5")
      self.assertLessEqual(
          layer0_router_rel_l2, 1e-5, f"[{mesh_tag}] Layer-0 router rel_l2 {layer0_router_rel_l2} > 1e-5"
      )

      perturbed_mean_kl, _ = _kl_divergence_fp64(hf_logits, np.asarray(perturbed_frozen_out.logits))
      self.assertGreater(
          perturbed_mean_kl,
          1e-3,
          f"[{mesh_tag}] Perturbed FrozenTopK negative control failed to trip KL gate: {perturbed_mean_kl}",
      )

      m3_tf_topk = np.asarray(jax.lax.top_k(jnp.asarray(m3_frozen_router), 8)[1])  # (24, B, S, 8)
      sorted_hf_r = np.sort(hf_router_all, axis=-1)[..., ::-1]
      margin = sorted_hf_r[..., 7] - sorted_hf_r[..., 8]
      ulp = np.maximum(
          np.spacing(np.abs(sorted_hf_r[..., 7]).astype(np.float32)),
          np.finfo(np.float32).eps,
      )
      non_tie_mask = margin >= 8.0 * ulp
      self.assertGreater(int(np.sum(non_tie_mask)), 0)

      tf_flips = 0
      slot_overlaps: list[float] = []
      for l_idx in range(24):
        for b_idx in range(bsz):
          for s_idx in range(seq_len):
            if not non_tie_mask[l_idx, b_idx, s_idx]:
              continue
            s_hf = set(int(x) for x in hf_topk_all[l_idx, b_idx, s_idx])
            s_m3 = set(int(x) for x in m3_tf_topk[l_idx, b_idx, s_idx])
            if s_hf != s_m3:
              tf_flips += 1
            slot_overlaps.append(len(s_hf & s_m3) / 8.0)

      slot_overlap_rate = float(np.mean(slot_overlaps))
      self.assertEqual(
          tf_flips,
          0,
          f"[{mesh_tag}] Expected 0 teacher-forced top-8 flips where margin >= 8 fp32 ULP, got {tf_flips}",
      )

      m3_natural_logits = np.asarray(natural_out.logits)
      unfrozen_mean_kl, unfrozen_max_kl = _kl_divergence_fp64(hf_logits, m3_natural_logits)
      self.assertTrue(bool(np.isfinite(unfrozen_mean_kl) and np.isfinite(unfrozen_max_kl)))

      print(
          f"[PHASE3-2 PARITY {mesh_tag}] shape=({bsz},{seq_len}) "
          f"FrozenTopK: mean_KL={frozen_mean_kl:.3e} max_KL={frozen_max_kl:.3e} "
          f"perturbed_KL={perturbed_mean_kl:.3e} "
          f"logits_rel_l2={frozen_rel_l2:.3e} logits_max_abs={frozen_max_abs:.3e} "
          f"layer0_router_rel_l2={layer0_router_rel_l2:.3e} | "
          f"TeacherForcedRouter: non_tie_tokens={int(np.sum(non_tie_mask))}/{non_tie_mask.size} "
          f"tf_flips={tf_flips} top8_slot_overlap={slot_overlap_rate:.4%} | "
          f"UnforcedRolloutDiag: mean_KL={unfrozen_mean_kl:.3e} max_KL={unfrozen_max_kl:.3e}",
          flush=True,
      )

  def test_3_real_weight_16_token_greedy_generation_exact_match(self):
    prompt = "The capital of France is"
    input_ids_pt = self.hf_tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.long)
    prompt_len = int(input_ids_pt.shape[1])
    max_new_tokens = 16
    total_len = prompt_len + max_new_tokens

    with torch.no_grad():
      hf_gen_ids = (
          self.hf_model_fp32.generate(
              input_ids_pt,
              attention_mask=torch.ones_like(input_ids_pt),
              max_new_tokens=max_new_tokens,
              do_sample=False,
              use_cache=True,
              pad_token_id=self.hf_tokenizer.eos_token_id,
          )[0]
          .cpu()
          .numpy()
      )
    hf_new_ids = [int(t) for t in hf_gen_ids[prompt_len:]]

    mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=list(jax.devices())[:4])
    model, _ = _load_sharded_m3_from_orbax(
        self.orbax_fp32_dir, mesh, dtype="float32", weight_dtype="float32", matmul_precision="highest"
    )

    @nnx.jit
    def _prefill(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array, cache: modeling_granitemoe.GraniteMoeKVCache):
      out = m.forward_with_details(toks, kv_cache=cache, cache_position=0)
      next_tok = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)
      return next_tok, out.kv_cache

    @nnx.jit
    def _decode_step(
        m: modeling_granitemoe.GraniteMoeModel,
        tok: jax.Array,
        cache: modeling_granitemoe.GraniteMoeKVCache,
        pos: jax.Array,
    ):
      out = m.forward_with_details(tok, kv_cache=cache, cache_position=pos)
      next_tok = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)
      return next_tok, out.kv_cache

    with jax.sharding.set_mesh(mesh):
      kv_cache = modeling_granitemoe.init_kv_cache(
          model.config, batch_size=1, max_seq_len=total_len, mesh=mesh, sharding=model.sharding  # type: ignore[arg-type]
      )
      prompt_jax = jnp.asarray(input_ids_pt.numpy(), dtype=jnp.int32)
      next_tok, kv_cache = _prefill(model, prompt_jax, kv_cache)
      m3_new_ids = [int(np.asarray(next_tok)[0])]
      for step_idx in range(1, max_new_tokens):
        cur_pos = jnp.asarray(prompt_len + step_idx - 1, dtype=jnp.int32)
        next_tok, kv_cache = _decode_step(model, next_tok[:, None], kv_cache, cur_pos)
        m3_new_ids.append(int(np.asarray(next_tok)[0]))

    decoded_hf = self.hf_tokenizer.decode(hf_new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded_m3 = self.hf_tokenizer.decode(m3_new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(
        f"[PHASE3-3 GREEDY-16] prompt={prompt!r}\n"
        f"  HF_new_ids={hf_new_ids}\n"
        f"  M3_new_ids={m3_new_ids}\n"
        f"  decoded={decoded_m3!r}",
        flush=True,
    )
    self.assertEqual(m3_new_ids, hf_new_ids, f"16-token greedy mismatch: M3={m3_new_ids} vs HF={hf_new_ids}")
    self.assertEqual(decoded_m3, decoded_hf)

  def test_4_sharded_v5p_5_step_training_via_m3_loop_and_zero_expert_all_gather(self):
    _ensure_orbax_pathways_compat()
    from maxtext.m3.train import loop as m3_loop  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    corpus = [
        (
            "Mixture-of-Experts models scale parameter capacity while keeping per-token FLOPs small, predictable, "
            "and hardware-efficient across multi-chip TPU meshes during pretraining and fine-tuning."
        ),
        (
            "Granite 3.0 MoE applies embedding, attention, residual, and logits scaling multipliers across all 24 "
            "stacked pure-NNX decoder layers with grouped-query attention and SwiGLU routed experts."
        ),
        (
            "MaxText m3 expresses model sharding over four logical mesh axes: data, fsdp, tensor, and expert, "
            "avoiding any post-initialization parameter sharding constraints inside forward execution."
        ),
        (
            "Switch Transformer load-balancing auxiliary loss encourages balanced token routing across all 32 "
            "routed experts when configured through the MaxTextTrainingEngine auxiliary loss hook."
        ),
    ]
    enc_full = self.hf_tokenizer(corpus, padding=False, truncation=True, max_length=64)["input_ids"]
    seq_len = min(33, min(len(ids) for ids in enc_full))
    self.assertGreaterEqual(seq_len, 17)
    seq_ids = np.asarray([ids[:seq_len] for ids in enc_full], dtype=np.int32)
    inputs_np = seq_ids[:, :-1]
    targets_np = seq_ids[:, 1:]
    ones_mask_np = np.ones_like(inputs_np, dtype=np.int32)
    pos_np = np.broadcast_to(np.arange(inputs_np.shape[1], dtype=np.int32)[None, :], inputs_np.shape)

    for mesh_tag, ici in (("v5p_fsdp2_ep2", {"fsdp": 2, "expert": 2}), ("v5p_fsdp2_tp2", {"fsdp": 2, "tensor": 2})):
      mesh = m3_sharding.create_mesh(ici=ici, devices=list(jax.devices())[:4])
      model, mt_cfg = _load_sharded_m3_from_orbax(
          self.orbax_fp32_dir,
          mesh,
          dtype="bfloat16",
          weight_dtype="float32",
          scan_layers=True,
          remat_policy="minimal",
          matmul_precision="default",
      )
      forbidden = _forbidden_expert_shapes(model, tp_size=int(mesh.shape["tensor"]))

      batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None))
      inputs_jax = jax.device_put(jnp.asarray(inputs_np), batch_sharding)
      targets_jax = jax.device_put(jnp.asarray(targets_np), batch_sharding)
      ones_mask_jax = jax.device_put(jnp.asarray(ones_mask_np), batch_sharding)
      pos_jax = jax.device_put(jnp.asarray(pos_np), batch_sharding)

      # 1. Audit compiled 24-layer real-weight forward+backward HLO for ZERO expert-weight all-gather
      #    and verify finite non-zero per-leaf gradients + live Switch `moe_lb_loss`!
      graphdef, params_st, rest_st = nnx.split(model, nnx.Param, ...)
      params_pure = nnx.to_pure_dict(params_st)

      @jax.jit
      def _pure_fwd_bwd(p_dict, t_in, t_tgt, t_msk, *, gdef=graphdef, rst=rest_st):
        def _l(p_inner):
          mod = nnx.merge(gdef, p_inner, rst)
          return modeling_granitemoe.compute_granitemoe_loss(
              mod,
              tokens=t_in,
              decoder_target_tokens=t_tgt,
              decoder_target_mask=t_msk,
              decoder_segment_ids=t_msk,
              router_aux_loss_coef=1e-3,
          )

        return jax.value_and_grad(_l, has_aux=True)(p_dict)

      with jax.sharding.set_mesh(mesh):
        compiled = _pure_fwd_bwd.lower(params_pure, inputs_jax, targets_jax, ones_mask_jax).compile()
        hlo_text = compiled.as_text()
        self.assertIsNotNone(hlo_text)
        ag_count = _assert_zero_expert_all_gather(hlo_text, forbidden, mesh_tag)
        (_, init_aux), init_grads = compiled(params_pure, inputs_jax, targets_jax, ones_mask_jax)
        init_gnorm = float(optax.global_norm(init_grads))
        init_lb = float(init_aux["moe_lb_loss"])
        self.assertTrue(math.isfinite(init_gnorm) and init_gnorm > 0.0)
        self.assertTrue(math.isfinite(init_lb) and init_lb > 0.0)

        # Verify `utils_granitemoe.granitemoe_engine_loss_fn` (`MaxTextTrainingEngine` hook) on the sharded model:
        eng_cfg = type("_EngCfg", (), {"router_aux_loss_coef": 1e-3, "model_name": mt_cfg.model_name})()
        wm, eng_aux = utils_granitemoe.granitemoe_engine_loss_fn(
            model,
            eng_cfg,
            {
                "inputs": inputs_jax,
                "targets": targets_jax,
                "inputs_segmentation": ones_mask_jax,
                "targets_segmentation": ones_mask_jax,
            },
            is_train=True,
        )
        self.assertTrue(bool(np.isfinite(np.asarray(wm.unreduced_sum))))
        self.assertGreater(float(eng_aux["moe_lb_loss"]), 0.0)

        # 2. Run 5 sharded AdamW steps directly through M3's dedicated `maxtext.m3.train.loop.train_step`!
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=2e-5, weight_decay=0.01))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        loop_batch = {"inputs": inputs_jax, "positions": pos_jax, "targets": targets_jax}

        step_losses: list[float] = []
        for step in range(5):
          loss_val = float(m3_loop.train_step(model, optimizer, loop_batch))
          self.assertTrue(math.isfinite(loss_val))
          step_losses.append(loss_val)
          print(
              f"[PHASE3-4 M3_LOOP TRAIN {mesh_tag}] step={step} ce_loss={loss_val:.6f} "
              f"init_moe_lb_loss={init_lb:.6f} init_grad_norm={init_gnorm:.4f} "
              f"(hlo_non_expert_all_gathers={ag_count})",
              flush=True,
          )

      self.assertLess(step_losses[-1], step_losses[0], f"[{mesh_tag}] Loss did not decrease: {step_losses}")
      for i in range(4):
        self.assertLess(
            step_losses[i + 1],
            step_losses[i],
            f"[{mesh_tag}] Non-monotonic training loss at step {i}->{i+1}: {step_losses}",
        )

  @classmethod
  def tearDownClass(cls):
    """Restores global JAX matmul precision flag after real-weight TPU parity suite completes."""
    jax.config.update("jax_default_matmul_precision", cls._prev_matmul_precision)
    super().tearDownClass()


if __name__ == "__main__":
  absltest.main()
