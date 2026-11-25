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

"""M3 8-check conformance, `PerturbSpec` negative controls, HLO audit, and PyTorch parity tests for GraniteMoe."""

from __future__ import annotations

import os
import sys

if "pytest" not in sys.modules and "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  existing_xla = os.environ.get("XLA_FLAGS", "").strip()
  os.environ["XLA_FLAGS"] = f"{existing_xla} --xla_force_host_platform_device_count=8".strip()

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import json
import math
import re
import subprocess
import tempfile
from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs import config as m3_config
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.granitemoe import checkpoint_mappings_granitemoe
from maxtext.m3.models.granitemoe import modeling_granitemoe
from maxtext.m3.models.granitemoe import utils_granitemoe
import numpy as np
import optax

try:
  import torch
  from transformers.models.granitemoe import configuration_granitemoe as hf_granitemoe_cfg
  from transformers.models.granitemoe import modeling_granitemoe as hf_granitemoe_mod

  HAS_TORCH = True
except ImportError:
  torch = None  # type: ignore[assignment]
  hf_granitemoe_cfg = None  # type: ignore[assignment]
  hf_granitemoe_mod = None  # type: ignore[assignment]
  HAS_TORCH = False


# Non-square synthetic dimensions (2 * intermediate_size = 96 != hidden_size = 64)
# so accidental expert weight transposes fail immediately.
def _make_mini_config(**overrides: Any) -> modeling_granitemoe.GraniteMoeConfig:
  """Builds a tiny non-square GraniteMoeConfig for fast unit and parity tests."""
  defaults: dict[str, Any] = {
      "vocab_size": 256,
      "hidden_size": 64,
      "intermediate_size": 48,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "num_local_experts": 8,
      "num_experts_per_tok": 2,
      "rms_norm_eps": 1e-6,
      "rope_theta": 10000.0,
      "attention_bias": False,
      "tie_word_embeddings": True,
      "embedding_multiplier": 12.0,
      "attention_multiplier": 0.015625,
      "residual_multiplier": 0.22,
      "logits_scaling": 6.0,
      "router_aux_loss_coef": 0.01,
      "dtype": "float32",
      "weight_dtype": "float32",
      "scan_layers": True,
      "remat_policy": "none",
      "matmul_precision": "highest",
  }
  defaults.update(overrides)
  return modeling_granitemoe.GraniteMoeConfig(**defaults)


def _flatten_nnx_vars(state: Any) -> list[tuple[str, Any]]:
  """Flattens an NNX `State` with `is_leaf=isinstance(..., nnx.Variable)` into `('a.b', var)` pairs."""
  flat, _ = jax.tree_util.tree_flatten_with_path(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  out: list[tuple[str, Any]] = []
  for path, leaf in flat:
    key = ".".join(str(getattr(p, "key", p)) for p in path)
    out.append((key, leaf))
  return out


def _rel_l2_error(got: np.ndarray, ref: np.ndarray) -> float:
  got_f64 = np.asarray(got, dtype=np.float64)
  ref_f64 = np.asarray(ref, dtype=np.float64)
  denom = max(float(np.linalg.norm(ref_f64)), 1e-12)
  return float(np.linalg.norm(got_f64 - ref_f64) / denom)


def _get_devices(count: int) -> list[jax.Device]:
  default_devs = list(jax.devices())
  if len(default_devs) >= count:
    return default_devs[:count]
  cpu_devs = list(jax.devices("cpu"))
  if len(cpu_devs) >= count:
    return cpu_devs[:count]
  raise unittest.SkipTest(f"Needs >= {count} JAX devices; got {len(default_devs)} default and {len(cpu_devs)} CPU.")


# ==============================================================================
# Taint & HLO Inspection Helpers (Mirroring M3 Conformance Checks 6 & 8)
# ==============================================================================

_UNARY_TAINT_PRIMS = frozenset(
    {
        "broadcast_in_dim",
        "convert_element_type",
        "copy",
        "device_put",
        "dynamic_slice",
        "optimization_barrier",
        "pad",
        "reshape",
        "reshard",
        "rev",
        "sharding_constraint",
        "slice",
        "squeeze",
        "stop_gradient",
        "transpose",
    }
)
_ALL_GATHER_RE = re.compile(r"\ball-gather(?:-start|-done)?(?:\.\d+)?\b")
_HLO_SHAPE_RE = re.compile(r"[a-z0-9]+\[([0-9,]+)\]")


def _walk_jaxpr_for_param_constraints(
    jaxpr_like: Any,
    in_taints: Sequence[str | None],
    violations: list[tuple[str, str]],
) -> list[str | None]:
  """Walks `jaxpr` equations recording any `sharding_constraint` / `reshard` / `device_put` on param-tainted vars."""
  jaxpr_obj = jaxpr_like.jaxpr if hasattr(jaxpr_like, "jaxpr") else jaxpr_like
  local_taint: dict[int, str] = {}
  for v, src in zip(jaxpr_obj.invars, in_taints):
    if src is not None:
      local_taint[id(v)] = src

  for eqn in jaxpr_obj.eqns:
    prim = eqn.primitive.name
    eqn_in = [local_taint.get(id(v)) for v in eqn.invars]
    in0 = eqn_in[0] if eqn_in else None
    if prim in ("sharding_constraint", "reshard", "device_put") and in0 is not None:
      violations.append((in0, prim))

    if prim in _UNARY_TAINT_PRIMS and in0 is not None:
      for out_v in eqn.outvars:
        local_taint[id(out_v)] = in0
    elif prim == "scan" and "jaxpr" in eqn.params:
      sub_j = eqn.params["jaxpr"].jaxpr
      n_c = eqn.params.get("num_consts", 0)
      n_k = eqn.params.get("num_carry", 0)
      carry = list(eqn_in[n_c : n_c + n_k])
      sub_outs = _walk_jaxpr_for_param_constraints(sub_j, [*eqn_in[:n_c], *carry, *eqn_in[n_c + n_k :]], violations)
      for out_v, src in zip(eqn.outvars, sub_outs):
        if src is not None:
          local_taint[id(out_v)] = src
    elif "jaxpr" in eqn.params or "call_jaxpr" in eqn.params:
      sub_ref = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
      sub_j = sub_ref.jaxpr if hasattr(sub_ref, "jaxpr") else sub_ref
      if len(sub_j.invars) == len(eqn_in):
        sub_outs = _walk_jaxpr_for_param_constraints(sub_j, eqn_in, violations)
        for out_v, src in zip(eqn.outvars, sub_outs):
          if src is not None:
            local_taint[id(out_v)] = src

  return [local_taint.get(id(v)) for v in jaxpr_obj.outvars]


def _forbidden_expert_shapes(
    model: modeling_granitemoe.GraniteMoeModel,
    tp_size: int = 1,
) -> dict[tuple[int, ...], str]:
  """Collects full, unstacked, TP-sliced full-E, and 3-D collapsed expert shapes forbidden in `all-gather`."""
  forbidden: dict[tuple[int, ...], str] = {}
  n_layers = model.config.num_hidden_layers
  n_exp = model.config.num_local_experts
  h_dim = model.config.hidden_size
  i_dim = model.config.intermediate_size

  def _register_unstacked(unstacked: tuple[int, ...], label: str) -> None:
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
    _register_unstacked(var.get_raw_value().shape[1:], path)

  if tp_size > 1 and i_dim % tp_size == 0:
    i_tp = i_dim // tp_size
    _register_unstacked((n_exp, h_dim, 2, i_tp), "layers.block_sparse_moe.experts.gate_up_proj[tp_shard]")
    _register_unstacked((n_exp, i_tp, h_dim), "layers.block_sparse_moe.experts.down_proj[tp_shard]")

  return forbidden


def _assert_zero_expert_all_gather(hlo_text: str, forbidden: Mapping[tuple[int, ...], str], tag: str) -> None:
  """Asserts that compiled HLO contains zero all-gather ops producing expert weight shapes."""
  for line in hlo_text.splitlines():
    stripped = line.strip()
    if stripped.startswith("//"):
      continue
    m = _ALL_GATHER_RE.search(stripped)
    if m is None:
      continue
    lhs = stripped[: m.start()]
    for dims_str in _HLO_SHAPE_RE.findall(lhs):
      shape = tuple(int(d) for d in dims_str.split(",") if d.strip())
      if shape in forbidden:
        raise AssertionError(f"[{tag}] Forbidden expert weight all-gather of {forbidden[shape]} {shape}: {stripped}")


@dataclasses.dataclass(frozen=True)
class PerturbSpec:
  """Single programmatic perturbation negative control."""

  name: str
  apply: Callable[[modeling_granitemoe.GraniteMoeModel], modeling_granitemoe.GraniteMoeModel]


def _scale_into_rel_band(
    arr: jax.Array,
    direction: jax.Array,
    target_rel_norm: float = 1e-2,
) -> jax.Array:
  """Adds a perturbation vector scaled to `target_rel_norm` (`5e-3 <= rel_norm <= 2e-2`) of `arr`."""
  base_norm = jnp.maximum(jnp.linalg.norm(arr.astype(jnp.float32)), 1e-6)
  dir_norm = jnp.maximum(jnp.linalg.norm(direction.astype(jnp.float32)), 1e-6)
  delta = direction.astype(jnp.float32) * (target_rel_norm * base_norm / dir_norm)
  return (arr.astype(jnp.float32) + delta).astype(arr.dtype)


# ==============================================================================
# Suite 1: M3 8-Check Conformance + Sharding + FLOPs + Registry Tests (No Torch)
# ==============================================================================


class GraniteMoeConformanceTest(absltest.TestCase):
  """Runs all 8 M3 conformance checks, registry/YAML checks, and multi-mesh HLO audits."""

  def test_check1_scan_equivalence_and_path_transparency(self):
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    cfg_scan = _make_mini_config(scan_layers=True, remat_policy="none")
    cfg_unroll = _make_mini_config(scan_layers=False, remat_policy="none")

    with self.assertRaises((TypeError, ValueError)):
      modeling_granitemoe.create_granitemoe_model(cfg_scan, mesh=mesh, rngs=None)  # type: ignore[arg-type]

    model_scan = modeling_granitemoe.create_granitemoe_model(cfg_scan, mesh=mesh, rngs=nnx.Rngs(0))
    model_unroll = modeling_granitemoe.create_granitemoe_model(cfg_unroll, mesh=mesh, rngs=nnx.Rngs(1))

    meta_scan = [
        (p, type(v), v.get_raw_value().shape, v.get_raw_value().dtype)
        for p, v in _flatten_nnx_vars(nnx.state(model_scan))
    ]
    meta_unroll = [
        (p, type(v), v.get_raw_value().shape, v.get_raw_value().dtype)
        for p, v in _flatten_nnx_vars(nnx.state(model_unroll))
    ]
    self.assertEqual(meta_scan, meta_unroll)

    # Ensure zero Python int `layer_idx` attributes exist on any decoder submodule.
    for path, node in nnx.iter_graph(model_scan.layers):
      self.assertFalse(hasattr(node, "layer_idx"), f"Submodule {path} must not store Python int layer_idx.")

    nnx.update(model_unroll, nnx.clone(nnx.state(model_scan)))
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, cfg_scan.vocab_size)
    with jax.sharding.set_mesh(mesh):
      default_logits = model_scan(tokens)
      self.assertIsInstance(default_logits, jax.Array)
      logits_scan, aux_scan = model_scan(tokens, return_aux=True)
      logits_unroll, aux_unroll = model_unroll(tokens, return_aux=True)

    np.testing.assert_allclose(np.asarray(logits_scan), np.asarray(logits_unroll), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(aux_scan["moe_lb_loss"]), np.asarray(aux_unroll["moe_lb_loss"]), rtol=1e-5, atol=1e-5
    )

    # Also verify `attention_dropout > 0` is active when `deterministic=False` and matches across scan vs unroll.
    drop_scan = modeling_granitemoe.create_granitemoe_model(
        _make_mini_config(scan_layers=True, attention_dropout=0.25), mesh=mesh, rngs=nnx.Rngs(0)
    )
    drop_unroll = modeling_granitemoe.create_granitemoe_model(
        _make_mini_config(scan_layers=False, attention_dropout=0.25), mesh=mesh, rngs=nnx.Rngs(1)
    )
    nnx.update(drop_scan, nnx.clone(nnx.state(model_scan)))
    nnx.update(drop_unroll, nnx.clone(nnx.state(model_scan)))
    with jax.sharding.set_mesh(mesh):
      det_out = drop_scan(tokens, deterministic=True)
      train_scan = drop_scan(tokens, deterministic=False, dropout_rng=jax.random.PRNGKey(77))
      train_unroll = drop_unroll(tokens, deterministic=False, dropout_rng=jax.random.PRNGKey(77))
    np.testing.assert_allclose(np.asarray(det_out), np.asarray(logits_scan), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(train_scan), np.asarray(train_unroll), rtol=1e-5, atol=1e-5)
    self.assertGreater(float(np.max(np.abs(np.asarray(train_scan) - np.asarray(det_out)))), 1e-4)

  def test_check2_all_leaf_and_all_expert_grad_health(self):
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    cfg = _make_mini_config(scan_layers=True, router_aux_loss_coef=0.05, tie_word_embeddings=False)
    model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(3))

    all_vars = _flatten_nnx_vars(nnx.state(model))
    self.assertNotEmpty(all_vars)
    for path, var in all_vars:
      self.assertIsInstance(var, nnx.Param, f"Non-Param variable {path} found on model.")

    tokens = jax.random.randint(jax.random.PRNGKey(11), (4, 16), 0, cfg.vocab_size)
    targets = jax.random.randint(jax.random.PRNGKey(12), (4, 16), 0, cfg.vocab_size)
    mask = jnp.ones((4, 16), dtype=jnp.int32)

    with jax.sharding.set_mesh(mesh):
      (loss, aux), grads = nnx.value_and_grad(
          lambda m: modeling_granitemoe.compute_granitemoe_loss(
              m, tokens=tokens, decoder_target_tokens=targets, decoder_target_mask=mask
          ),
          has_aux=True,
      )(model)

    self.assertTrue(bool(np.isfinite(np.asarray(loss))))
    self.assertGreater(float(aux["moe_lb_loss"]), 0.0)

    for path, grad_var in _flatten_nnx_vars(grads):
      g_np = np.asarray(grad_var.get_raw_value())
      self.assertTrue(bool(np.all(np.isfinite(g_np))), f"Non-finite gradient at {path}")
      self.assertTrue(bool(np.any(g_np != 0.0)), f"Zero/dead gradient at {path}")
      if ".experts." in path:
        # Shape (num_layers, num_experts, ...): verify every individual expert slice gets non-zero gradient.
        for exp_id in range(cfg.num_local_experts):
          self.assertTrue(
              bool(np.any(g_np[:, exp_id] != 0.0)),
              f"Expert {exp_id} received zero gradient in {path}.",
          )

  def test_check3_remat_equivalence_and_non_param_state_stability(self):
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    ref_model = modeling_granitemoe.create_granitemoe_model(
        _make_mini_config(scan_layers=False, remat_policy="none"), mesh=mesh, rngs=nnx.Rngs(5)
    )
    ref_state = nnx.clone(nnx.state(ref_model))

    _, _, init_non_param = nnx.split(ref_model, nnx.Param, ...)
    init_np_treedef = jax.tree.structure(init_non_param)

    tokens = jax.random.randint(jax.random.PRNGKey(21), (2, 10), 0, 256)
    targets = jax.random.randint(jax.random.PRNGKey(22), (2, 10), 0, 256)
    mask = jnp.ones((2, 10), dtype=jnp.int32)

    def _eval_loss_and_grads(mod: modeling_granitemoe.GraniteMoeModel):
      return nnx.value_and_grad(
          lambda m: modeling_granitemoe.compute_granitemoe_loss(
              m, tokens=tokens, decoder_target_tokens=targets, decoder_target_mask=mask
          ),
          has_aux=True,
      )(mod)

    with jax.sharding.set_mesh(mesh):
      (ref_loss, ref_aux), ref_grads = _eval_loss_and_grads(ref_model)
      ref_grad_map = {p: np.asarray(v.get_raw_value()) for p, v in _flatten_nnx_vars(ref_grads)}

      for scan_layers in (False, True):
        for remat_policy in ("none", "full", "minimal"):
          if (scan_layers, remat_policy) == (False, "none"):
            continue
          var_model = modeling_granitemoe.create_granitemoe_model(
              _make_mini_config(scan_layers=scan_layers, remat_policy=remat_policy),
              mesh=mesh,
              rngs=nnx.Rngs(99),
          )
          nnx.update(var_model, nnx.clone(ref_state))
          (var_loss, var_aux), var_grads = _eval_loss_and_grads(var_model)

          # Check non-Param state stability (no sowed nnx.Intermediate).
          for path, v_node in _flatten_nnx_vars(nnx.state(var_model)):
            self.assertNotIsInstance(v_node, nnx.Intermediate, f"Sowed Intermediate at {path}")
          _, _, after_non_param = nnx.split(var_model, nnx.Param, ...)
          self.assertEqual(jax.tree.structure(after_non_param), init_np_treedef)

          np.testing.assert_allclose(np.asarray(var_loss), np.asarray(ref_loss), rtol=1e-5, atol=1e-5)
          np.testing.assert_allclose(
              np.asarray(var_aux["moe_lb_loss"]), np.asarray(ref_aux["moe_lb_loss"]), rtol=1e-5, atol=1e-5
          )
          for path, g_var in _flatten_nnx_vars(var_grads):
            np.testing.assert_allclose(
                np.asarray(g_var.get_raw_value()),
                ref_grad_map[path],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mismatch at (scan={scan_layers}, remat={remat_policy}) grad {path}",
            )

  def test_check4_aux_loss_masking_and_engine_grad_coupling(self):
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    cfg = _make_mini_config(router_aux_loss_coef=0.0)
    model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(9))

    # 1. Verify padding mask excludes masked positions from Switch load-balancing loss.
    tokens_a = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    tokens_b = jnp.array([[1, 2, 3, 4, 99, 100, 101, 102]], dtype=jnp.int32)
    pad_mask = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.int32)

    with jax.sharding.set_mesh(mesh):
      _, aux_a = model(tokens_a, attention_mask=pad_mask, return_aux=True)
      _, aux_b = model(tokens_b, attention_mask=pad_mask, return_aux=True)
      _, aux_unmasked_a = model(tokens_a, return_aux=True)
      _, aux_unmasked = model(tokens_b, return_aux=True)

    self.assertEqual(set(aux_a.keys()), {"moe_lb_loss"})
    self.assertGreater(float(aux_a["moe_lb_loss"]), 0.0)
    # Causal prefix [0..3] has identical hidden states, and [4..7] are masked out -> identical moe_lb_loss!
    np.testing.assert_allclose(np.asarray(aux_a["moe_lb_loss"]), np.asarray(aux_b["moe_lb_loss"]), rtol=1e-6, atol=1e-6)
    self.assertNotAlmostEqual(float(aux_a["moe_lb_loss"]), float(aux_unmasked["moe_lb_loss"]), places=4)

    # 2. Verify SFT prompt masking (`decoder_target_mask` = 0 on prompt [0..3], 1 on response [4..7],
    # `decoder_segment_ids=None`) masks ONLY CE loss without altering `moe_lb_loss` or blinding response to prompt!
    sft_target_mask = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    tokens_alt_prompt = jnp.array([[91, 92, 93, 94, 5, 6, 7, 8]], dtype=jnp.int32)
    with jax.sharding.set_mesh(mesh):
      _, sft_aux_1 = modeling_granitemoe.compute_granitemoe_loss(
          model, tokens=tokens_a, decoder_target_tokens=tokens_b, decoder_target_mask=sft_target_mask
      )
      _, sft_aux_2 = modeling_granitemoe.compute_granitemoe_loss(
          model, tokens=tokens_alt_prompt, decoder_target_tokens=tokens_b, decoder_target_mask=sft_target_mask
      )
    self.assertAlmostEqual(float(sft_aux_1["total_weights"]), 4.0, places=6)
    np.testing.assert_allclose(
        np.asarray(sft_aux_1["moe_lb_loss"]), np.asarray(aux_unmasked_a["moe_lb_loss"]), rtol=1e-6, atol=1e-6
    )
    self.assertGreater(abs(float(sft_aux_1["ce_loss"]) - float(sft_aux_2["ce_loss"])), 1e-4)

    # 3. Verify MODEL_ENGINE_CONFIG_REGISTRY loss hook wires router_aux_loss_coef into router gradients.
    class _DummyEngine:
      """Minimal training-engine stub for verifying configure_engine hooks."""

      def __init__(self, model_name: str):
        self._config = type("Cfg", (), {"model_name": model_name, "router_aux_loss_coef": 0.5})()
        self.loss_fn = None
        self.has_aux = False
        self.gen_input_fn = "sentinel"

      def with_loss_fn(self, fn: Any, has_aux: bool = False):
        self.loss_fn = fn
        self.has_aux = has_aux
        return self

      def with_gen_model_input_fn(self, fn: Any):
        self.gen_input_fn = fn
        return self

    dummy_engine = _DummyEngine("granite-3.0-1b-a400m")
    m3_models.configure_engine(dummy_engine)
    self.assertTrue(dummy_engine.has_aux)
    self.assertIsNone(dummy_engine.gen_input_fn)
    self.assertIsNotNone(dummy_engine.loss_fn)

    batch = {
        "inputs": tokens_a,
        "targets": tokens_b,
        "inputs_segmentation": pad_mask,
        "targets_segmentation": pad_mask,
    }
    cfg_zero = type("CfgZero", (), {"router_aux_loss_coef": 0.0})()
    cfg_pos = type("CfgPos", (), {"router_aux_loss_coef": 1.0})()

    def _engine_scalar_loss(mod, eng_cfg):
      wm, aux_d = dummy_engine.loss_fn(mod, eng_cfg, batch, is_train=True)
      return wm.unreduced_sum / jnp.maximum(wm.denominator, 1.0), aux_d

    with jax.sharding.set_mesh(mesh):
      (_, aux_z), grads_z = nnx.value_and_grad(lambda m: _engine_scalar_loss(m, cfg_zero), has_aux=True)(model)
      (_, aux_p), grads_p = nnx.value_and_grad(lambda m: _engine_scalar_loss(m, cfg_pos), has_aux=True)(model)

    self.assertGreater(float(aux_z["moe_lb_loss"]), 0.0)
    self.assertGreater(float(aux_p["moe_lb_loss"]), 0.0)
    g_router_z = np.asarray(nnx.to_pure_dict(grads_z)["layers"]["block_sparse_moe"]["router"]["weight"])
    g_router_p = np.asarray(nnx.to_pure_dict(grads_p)["layers"]["block_sparse_moe"]["router"]["weight"])
    self.assertGreater(float(np.max(np.abs(g_router_p - g_router_z))), 1e-5)

    # 4. Verify explicit `attention_mask` is combined (not dropped) when `decoder_segment_ids` is also passed.
    seg_ids = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    extra_attn_mask = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.int32)
    with jax.sharding.set_mesh(mesh):
      _, comb_a = modeling_granitemoe.compute_granitemoe_loss(
          model,
          tokens=tokens_a,
          decoder_target_tokens=tokens_a,
          decoder_target_mask=extra_attn_mask,
          decoder_segment_ids=seg_ids,
          attention_mask=extra_attn_mask,
      )
      _, comb_b = modeling_granitemoe.compute_granitemoe_loss(
          model,
          tokens=tokens_b,
          decoder_target_tokens=tokens_a,
          decoder_target_mask=extra_attn_mask,
          decoder_segment_ids=seg_ids,
          attention_mask=extra_attn_mask,
      )
    np.testing.assert_allclose(np.asarray(comb_a["ce_loss"]), np.asarray(comb_b["ce_loss"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(comb_a["moe_lb_loss"]), np.asarray(comb_b["moe_lb_loss"]), rtol=1e-6, atol=1e-6)

  def test_check5_driver_mesh_and_axis_names(self):
    driver_mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=_get_devices(4))
    valid_axes = frozenset(m3_sharding.MESH_AXIS_NAMES)
    cfg = _make_mini_config(vocab_size=255, tie_word_embeddings=False)
    model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=driver_mesh, rngs=nnx.Rngs(29))

    self.assertIs(model.mesh, driver_mesh)
    for path, node in nnx.iter_graph(model, graph=True):
      if isinstance(node, jax.sharding.Mesh):
        self.assertIs(node, driver_mesh, f"Private Mesh found at {path}")

    for path, var in _flatten_nnx_vars(nnx.state(model)):
      for meta_key in ("out_sharding", "sharding"):
        meta_val = var.get_metadata(meta_key, None)
        if meta_val is None:
          continue
        if isinstance(meta_val, jax.sharding.NamedSharding):
          self.assertIs(meta_val.mesh, driver_mesh, f"Non-driver mesh in {path}.{meta_key}")
          spec_obj = meta_val.spec
        else:
          spec_obj = meta_val
        for entry in spec_obj:
          axes = (entry,) if isinstance(entry, str) else (entry or ())
          for ax in axes:
            self.assertIn(ax, valid_axes, f"Invalid axis {ax!r} in {path}.{meta_key}")

      arr = var.get_raw_value()
      self.assertIsInstance(arr.sharding, jax.sharding.NamedSharding)
      self.assertIs(arr.sharding.mesh, driver_mesh)
      for entry in arr.sharding.spec:
        axes = (entry,) if isinstance(entry, str) else (entry or ())
        for ax in axes:
          self.assertIn(ax, valid_axes, f"Invalid axis {ax!r} in {path}")

  def test_check6_no_post_init_param_constraints(self):
    driver_mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=_get_devices(4))
    cfg = _make_mini_config(vocab_size=255, tie_word_embeddings=False)
    model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=driver_mesh, rngs=nnx.Rngs(29))

    with jax.sharding.set_mesh(driver_mesh):
      tokens = jax.device_put(
          jnp.ones((4, 8), dtype=jnp.int32),
          jax.sharding.NamedSharding(driver_mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None)),
      )
      graphdef, params_st, rest_st = nnx.split(model, nnx.Param, ...)
      params_pure = nnx.to_pure_dict(params_st)
      flat_p, _ = jax.tree_util.tree_flatten_with_path(params_pure)

      def _pure_fwd(p_dict, toks):
        mod = nnx.merge(graphdef, p_dict, rest_st)
        return mod(toks, return_aux=True)

      closed = jax.make_jaxpr(_pure_fwd)(params_pure, tokens)
      in_taints: list[str | None] = [None] * len(closed.jaxpr.invars)
      for idx, (kp, _) in enumerate(flat_p):
        in_taints[idx] = ".".join(str(getattr(k, "key", k)) for k in kp)
      violations: list[tuple[str, str]] = []
      _walk_jaxpr_for_param_constraints(closed.jaxpr, in_taints, violations)
      self.assertEmpty(violations, f"Post-init param sharding constraints found: {violations}")

  def test_check7_weight_sharding_coverage_and_odd_vocab_guard(self):
    driver_mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=_get_devices(4))
    # Use odd vocab_size=255 (like real 49155) to verify TP=2 non-divisible vocab guard + full 2D+ weight sharding!
    cfg_odd = _make_mini_config(vocab_size=255, tie_word_embeddings=False)
    model = modeling_granitemoe.create_granitemoe_model(cfg_odd, mesh=driver_mesh, rngs=nnx.Rngs(29))

    seen_axes: set[str] = set()
    for path, var in _flatten_nnx_vars(nnx.state(model, nnx.Param)):
      arr = var.get_raw_value()
      self.assertIsInstance(arr.sharding, jax.sharding.NamedSharding)
      self.assertIs(arr.sharding.mesh, driver_mesh)
      unstacked = arr.shape[1:] if path.startswith("layers.") else arr.shape
      if len(unstacked) >= 2 and math.prod(unstacked) >= 256:
        self.assertNotEqual(arr.sharding.spec, jax.sharding.PartitionSpec(), f"Unsharded 2D+ weight {path}")
        shard_elems = math.prod(arr.addressable_shards[0].data.shape)
        self.assertLess(shard_elems, math.prod(arr.shape), f"Weight {path} shard not smaller than global shape")
        for entry in arr.sharding.spec:
          if isinstance(entry, str):
            seen_axes.add(entry)
          elif isinstance(entry, tuple):
            seen_axes.update(entry)

    self.assertIn("fsdp", seen_axes)
    self.assertIn("tensor", seen_axes)

  def test_gate9_kv_cache_prefill_decode_and_segment_mask_equivalence(self):
    # Verify on a 4-device sharded mesh with B=1 to also exercise the small-T decode fallback!
    mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=_get_devices(4))
    for scan_layers in (True, False):
      cfg = _make_mini_config(scan_layers=scan_layers)
      model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(17))
      seq = jax.random.randint(jax.random.PRNGKey(31), (1, 8), 0, cfg.vocab_size)

      with jax.sharding.set_mesh(mesh):
        full_logits = model(seq)
        kv_cache = modeling_granitemoe.init_kv_cache(
            cfg, batch_size=1, max_seq_len=8, mesh=mesh, sharding=model.sharding  # type: ignore[arg-type]
        )
        prefill_out = model.forward_with_details(seq[:, :5], kv_cache=kv_cache, cache_position=0)
        # Verify prefix-length `attention_mask` (cur_len=5 < max_seq_len=8) right-pads cleanly without broadcast error.
        prefill_masked = model.forward_with_details(
            seq[:, :5],
            attention_mask=jnp.ones((1, 5), dtype=jnp.int32),
            kv_cache=kv_cache,
            cache_position=0,
        )
        np.testing.assert_allclose(
            np.asarray(prefill_masked.logits), np.asarray(prefill_out.logits), rtol=1e-6, atol=1e-6
        )
        kv_cache = prefill_out.kv_cache
        step_logits = [prefill_out.logits]
        for pos in range(5, 8):
          dec_out = model.forward_with_details(seq[:, pos : pos + 1], kv_cache=kv_cache, cache_position=pos)
          kv_cache = dec_out.kv_cache
          step_logits.append(dec_out.logits)
        cached_logits = jnp.concatenate(step_logits, axis=1)

      np.testing.assert_allclose(
          np.asarray(cached_logits),
          np.asarray(full_logits),
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"KV-cache decode mismatch for scan_layers={scan_layers}",
      )

      # Verify `segment_ids` handling in both non-cached and KV-cache modes:
      # 1. Packed segments `[1,1,1,0,2,2,2,0]` never attend across segments OR to `segment_ids == 0` padding keys.
      # 2. Mutating a padding token (`idx 3`, `segment_ids == 0`) does NOT change any valid token (`segment != 0`) OR
      #    any later padding token (`idx 7`, `segment_ids == 0`) because `segment 0` keys are masked out.
      packed_seg = jnp.array([[1, 1, 1, 0, 2, 2, 2, 0]], dtype=jnp.int32)
      seq_pad_mut = seq.at[:, 3].set((seq[:, 3] + 17) % cfg.vocab_size)
      with jax.sharding.set_mesh(mesh):
        packed_full = model(seq, segment_ids=packed_seg)
        packed_mut = model(seq_pad_mut, segment_ids=packed_seg)
        np.testing.assert_allclose(
            np.asarray(packed_full[:, [0, 1, 2, 4, 5, 6, 7]]),
            np.asarray(packed_mut[:, [0, 1, 2, 4, 5, 6, 7]]),
            rtol=1e-5,
            atol=1e-5,
        )

        kv_seg = modeling_granitemoe.init_kv_cache(
            cfg, batch_size=1, max_seq_len=8, mesh=mesh, sharding=model.sharding  # type: ignore[arg-type]
        )
        pf_seg = model.forward_with_details(seq[:, :5], segment_ids=packed_seg, kv_cache=kv_seg, cache_position=0)
        kv_seg = pf_seg.kv_cache
        seg_steps = [pf_seg.logits]
        for pos in range(5, 8):
          d_seg = model.forward_with_details(
              seq[:, pos : pos + 1], segment_ids=packed_seg, kv_cache=kv_seg, cache_position=pos
          )
          kv_seg = d_seg.kv_cache
          seg_steps.append(d_seg.logits)
        cached_seg_logits = jnp.concatenate(seg_steps, axis=1)

      np.testing.assert_allclose(
          np.asarray(cached_seg_logits),
          np.asarray(packed_full),
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"KV-cache segment_ids mismatch for scan_layers={scan_layers}",
      )

  def test_gate10_centralized_checkpoint_mappings_and_roundtrip_completeness(self):
    # 1. Verify centralized registry wiring (`PARAM_MAPPING`, `HOOK_FNS`, `HF_SHAPE`) against
    #    `nnx.eval_shape` of full 24-layer `granite-3.0-1b-a400m` across `scan_layers` in `(False, True)`.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    base_yml = os.path.join(repo_root, "src", "maxtext", "configs", "base.yml")
    mesh_1 = m3_sharding.create_mesh(devices=_get_devices(1))

    self.assertIn("granite-3.0-1b-a400m", param_mapping.PARAM_MAPPING)
    self.assertIn("granite-3.0-1b-a400m", param_mapping.HOOK_FNS)
    self.assertIn("granite-3.0-1b-a400m", hf_shape.HF_SHAPE)

    try:
      from maxtext.checkpoint_conversion.utils import hf_model_configs  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      hf_cfg_obj: Any = hf_model_configs.HF_MODEL_CONFIGS["granite-3.0-1b-a400m"]
      hf_cfg_dict: dict[str, Any] = hf_cfg_obj.to_dict()
    except ImportError:
      hf_cfg_dict = dataclasses.asdict(modeling_granitemoe.GraniteMoeConfig())
      hf_cfg_obj = hf_cfg_dict

    expected_hf_shapes = hf_shape.HF_SHAPE["granite-3.0-1b-a400m"](hf_cfg_dict)
    self.assertLen(expected_hf_shapes, 2 + 24 * 9)

    for scanned in (False, True):
      mt_cfg = pyconfig.initialize(
          [None, base_yml],
          model_name="granite-3.0-1b-a400m",
          scan_layers=scanned,
          skip_jax_distributed_system=True,
      )
      self.assertEqual(mt_cfg.param_scan_axis, 0)
      abs_full = nnx.eval_shape(
          lambda c=mt_cfg: m3_models.create_model(c, mesh_1, rngs=nnx.Rngs(0), quant_mode_str="train")
      )
      abs_mt_shapes = {
          f"params-{p.replace('.', '-')}": v.get_raw_value().shape
          for p, v in _flatten_nnx_vars(nnx.state(abs_full, nnx.Param))
      }
      for cfg_variant in (hf_cfg_dict, hf_cfg_obj):
        pmap = param_mapping.PARAM_MAPPING["granite-3.0-1b-a400m"](cfg_variant, mt_cfg, scan_layers=scanned)
        hooks_to_mt = param_mapping.HOOK_FNS["granite-3.0-1b-a400m"](
            cfg_variant, mt_cfg, scan_layers=scanned, saving_to_hf=False
        )
        self.assertEqual(set(pmap.keys()), set(abs_mt_shapes.keys()))
        mapped_hf_keys: list[str] = []
        for mt_key, hf_val in pmap.items():
          hook_fn = hooks_to_mt.get(mt_key)
          if isinstance(hf_val, list):
            mapped_hf_keys.extend(hf_val)
            dummy_hf = np.zeros(expected_hf_shapes[hf_val[0]], dtype=np.float32)
            conv_slice = hook_fn(dummy_hf, None) if hook_fn is not None else dummy_hf
            self.assertEqual((len(hf_val), *conv_slice.shape), abs_mt_shapes[mt_key])
          else:
            mapped_hf_keys.append(hf_val)
            dummy_hf = np.zeros(expected_hf_shapes[hf_val], dtype=np.float32)
            conv_full = hook_fn(dummy_hf, abs_mt_shapes[mt_key]) if hook_fn is not None else dummy_hf
            self.assertEqual(conv_full.shape, abs_mt_shapes[mt_key])
        self.assertEqual(set(mapped_hf_keys), set(expected_hf_shapes.keys()))

    # 2. Verify lossless bidirectional tensor roundtrip across tied/untied & fp32/bf16 (including
    #    `conversion_utils.process_maxtext_param` used by centralized `to_huggingface.py` when installed).
    proc_mt_param: Any = None
    try:
      from orbax.checkpoint import v1 as ocp_v1  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      if not hasattr(ocp_v1.training, "pathways"):
        ocp_v1.training.pathways = type("_StubPathways", (), {"Snapshotter": object})()  # type: ignore[attr-defined]
      from maxtext.checkpoint_conversion.utils import utils as conversion_utils  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      proc_mt_param = conversion_utils.process_maxtext_param
    except ImportError:
      proc_mt_param = None

    for tie in (True, False):
      for dt in ("float32", "bfloat16"):
        cfg = _make_mini_config(tie_word_embeddings=tie, dtype=dt, weight_dtype=dt)
        model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh_1, rngs=nnx.Rngs(23))
        hf_dict = checkpoint_mappings_granitemoe.export_model_to_hf_weights(model, naming="hub")
        expected_keys = 2 + cfg.num_hidden_layers * 9 + (0 if tie else 1)
        self.assertLen(hf_dict, expected_keys)

        pure_rt = checkpoint_mappings_granitemoe.hf_weights_to_m3_pure_dict(hf_dict, cfg, target_dtype=dt)
        hf_rt = checkpoint_mappings_granitemoe.m3_pure_dict_to_hf_weights(pure_rt, cfg, naming="hub")
        self.assertEqual(set(hf_rt.keys()), set(hf_dict.keys()))
        for k, orig_arr in hf_dict.items():
          diff = float(np.max(np.abs(hf_rt[k].astype(np.float32) - orig_arr.astype(np.float32))))
          self.assertEqual(diff, 0.0, f"Non-zero roundtrip diff on {k} (tie={tie}, dt={dt})")

        if proc_mt_param is not None:
          mini_hf_cfg = dataclasses.asdict(cfg)
          mini_pmap = param_mapping.PARAM_MAPPING["granite-3.0-1b-a400m"](mini_hf_cfg, cfg, scan_layers=cfg.scan_layers)
          mini_hooks_hf = param_mapping.HOOK_FNS["granite-3.0-1b-a400m"](
              mini_hf_cfg, cfg, scan_layers=cfg.scan_layers, saving_to_hf=True
          )
          mini_shapes = hf_shape.HF_SHAPE["granite-3.0-1b-a400m"](mini_hf_cfg)
          exported_via_central: dict[str, np.ndarray] = {}
          scan_cfg = type(
              "_ScanCfg",
              (),
              {
                  "param_scan_axis": 0,
                  "scan_layers": cfg.scan_layers,
                  "weight_dtype": dt,
                  "base_num_decoder_layers": cfg.num_hidden_layers,
              },
          )()
          for flat_p, leaf_arr in checkpoint_mappings_granitemoe._flatten_pure_dict(pure_rt).items():  # pylint: disable=protected-access
            mt_k = f"params-{flat_p.replace('.', '-')}"
            pairs = proc_mt_param(
                mt_k,
                jnp.asarray(leaf_arr),
                mini_pmap,
                mini_hooks_hf,
                mini_shapes,
                scan_cfg,
            )
            for out_k, out_v in pairs:
              exported_via_central[out_k] = np.asarray(out_v)
          self.assertEqual(set(exported_via_central.keys()), set(hf_dict.keys()))
          for k, orig_arr in hf_dict.items():
            c_diff = float(np.max(np.abs(exported_via_central[k].astype(np.float32) - orig_arr.astype(np.float32))))
            self.assertEqual(c_diff, 0.0, f"Central process_maxtext_param diff on {k}")

        # Negative completeness controls: unexpected HF key or missing HF key must fail fast.
        bad_extra = dict(hf_dict)
        bad_extra["model.unexpected.weight"] = np.ones((4,), dtype=np.float32)
        with self.assertRaises(ValueError):
          checkpoint_mappings_granitemoe.hf_weights_to_m3_pure_dict(bad_extra, cfg)

        bad_missing = dict(hf_dict)
        bad_missing.pop("model.norm.weight")
        with self.assertRaises(KeyError):
          checkpoint_mappings_granitemoe.hf_weights_to_m3_pure_dict(bad_missing, cfg)

  def test_check8_multi_mesh_zero_expert_all_gather_and_8dev_compound_parity(self):
    # Reference single-device forward + backward values in fp32 (Precision.HIGHEST).
    cfg = _make_mini_config(scan_layers=True, router_aux_loss_coef=0.05)
    tokens_np = np.random.default_rng(7).integers(0, cfg.vocab_size, size=(8, 4), dtype=np.int32)
    targets_np = np.random.default_rng(8).integers(0, cfg.vocab_size, size=(8, 4), dtype=np.int32)
    mask_np = np.ones((8, 4), dtype=np.int32)

    mesh_1 = m3_sharding.create_mesh(devices=_get_devices(1))
    ref_model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh_1, rngs=nnx.Rngs(41))
    ref_gdef, ref_params_st, ref_rest_st = nnx.split(ref_model, nnx.Param, ...)
    ref_pure = jax.tree.map(np.asarray, nnx.to_pure_dict(ref_params_st))

    only_8dev = os.environ.get("GRANITEMOE_ONLY_8DEV_SUBPROCESS") == "1"
    ref_cache_file = os.environ.get("GRANITEMOE_REF_CACHE_FILE")
    if only_8dev and ref_cache_file and os.path.exists(ref_cache_file):
      with np.load(ref_cache_file) as npz:
        ref_loss = float(npz["__loss__"])
        ref_aux = {"moe_lb_loss": float(npz["__lb_loss__"])}
        ref_grad_flat = {k: npz[k] for k in npz.files if not k.startswith("__")}
    else:

      @jax.jit
      def _ref_step(p_dict, t_in, t_tgt, t_msk):
        def _loss(p_inner):
          mod = nnx.merge(ref_gdef, p_inner, ref_rest_st)
          return modeling_granitemoe.compute_granitemoe_loss(
              mod, tokens=t_in, decoder_target_tokens=t_tgt, decoder_target_mask=t_msk
          )

        return jax.value_and_grad(_loss, has_aux=True)(p_dict)

      with jax.sharding.set_mesh(mesh_1):
        (ref_loss, ref_aux), ref_grads = _ref_step(
            nnx.to_pure_dict(ref_params_st),
            jnp.asarray(tokens_np),
            jnp.asarray(targets_np),
            jnp.asarray(mask_np),
        )
        ref_grad_pure = jax.tree.map(np.asarray, ref_grads)
      ref_grad_flat = checkpoint_mappings_granitemoe._flatten_pure_dict(ref_grad_pure)  # pylint: disable=protected-access

    mesh_specs: list[tuple[str, dict[str, int], int]] = [
        ("fsdp2_tp2", {"fsdp": 2, "tensor": 2}, 4),
        ("fsdp2_ep2", {"fsdp": 2, "expert": 2}, 4),
        ("tp2_ep2", {"tensor": 2, "expert": 2}, 4),
        ("fsdp4", {"fsdp": 4}, 4),
        ("dp2_fsdp2", {"data": 2, "fsdp": 2}, 4),
        ("compound_fsdp2_tp2_ep2", {"fsdp": 2, "tensor": 2, "expert": 2}, 8),
    ]

    if only_8dev:
      mesh_specs = [("compound_fsdp2_tp2_ep2", {"fsdp": 2, "tensor": 2, "expert": 2}, 8)]

    avail_devs = max(len(jax.devices()), len(jax.devices("cpu")))
    for tag, ici, ndev in mesh_specs:
      if ndev > 4 and avail_devs < ndev:
        if only_8dev:
          self.fail(f"Expected >= {ndev} devices in 8-device subprocess, got {avail_devs}.")
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
          tmp_ref_path = tf.name
          np.savez(tf, __loss__=np.asarray(ref_loss), __lb_loss__=np.asarray(ref_aux["moe_lb_loss"]), **ref_grad_flat)
        try:
          child_env = os.environ.copy()
          cleaned_xla = re.sub(r"--xla_force_host_platform_device_count=\d+", "", child_env.get("XLA_FLAGS", "")).strip()
          child_env["XLA_FLAGS"] = f"{cleaned_xla} --xla_force_host_platform_device_count={ndev}".strip()
          child_env["JAX_PLATFORMS"] = "cpu"
          child_env["GRANITEMOE_ONLY_8DEV_SUBPROCESS"] = "1"
          child_env["GRANITEMOE_REF_CACHE_FILE"] = tmp_ref_path
          proc = subprocess.run(
              [
                  sys.executable,
                  os.path.abspath(__file__),
                  "GraniteMoeConformanceTest.test_check8_multi_mesh_zero_expert_all_gather_and_8dev_compound_parity",
              ],
              env=child_env,
              capture_output=True,
              text=True,
              check=False,
          )
        finally:
          if os.path.exists(tmp_ref_path):
            os.remove(tmp_ref_path)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"8-device compound HLO/grad subprocess failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}",
        )
        continue
      mesh = m3_sharding.create_mesh(ici=ici, devices=_get_devices(ndev))
      model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(41))
      checkpoint_mappings_granitemoe.load_pure_dict_into_model(model, ref_pure)
      forbidden = _forbidden_expert_shapes(model, tp_size=int(mesh.shape["tensor"]))

      batch_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None))
      toks = jax.device_put(jnp.asarray(tokens_np), batch_spec)
      tgts = jax.device_put(jnp.asarray(targets_np), batch_spec)
      msk = jax.device_put(jnp.asarray(mask_np), batch_spec)

      graphdef, params_st, rest_st = nnx.split(model, nnx.Param, ...)
      params_pure = nnx.to_pure_dict(params_st)

      @jax.jit
      def _step(p_dict, t_in, t_tgt, t_msk, *, gdef=graphdef, rst=rest_st):
        def _loss(p_inner):
          mod = nnx.merge(gdef, p_inner, rst)
          return modeling_granitemoe.compute_granitemoe_loss(
              mod, tokens=t_in, decoder_target_tokens=t_tgt, decoder_target_mask=t_msk
          )

        return jax.value_and_grad(_loss, has_aux=True)(p_dict)

      with jax.sharding.set_mesh(mesh):
        compiled = _step.lower(params_pure, toks, tgts, msk).compile()
        hlo_text = compiled.as_text()
        self.assertIsNotNone(hlo_text)
        _assert_zero_expert_all_gather(hlo_text, forbidden, tag)

        (got_loss, got_aux), got_grads = compiled(params_pure, toks, tgts, msk)

      np.testing.assert_allclose(np.asarray(got_loss), np.asarray(ref_loss), rtol=1e-5, atol=1e-5, err_msg=tag)
      np.testing.assert_allclose(
          np.asarray(got_aux["moe_lb_loss"]), np.asarray(ref_aux["moe_lb_loss"]), rtol=1e-5, atol=1e-5, err_msg=tag
      )
      got_grad_flat = checkpoint_mappings_granitemoe._flatten_pure_dict(got_grads)  # pylint: disable=protected-access
      for k_path, g_ref in ref_grad_flat.items():
        g_got = np.asarray(got_grad_flat[k_path])
        self.assertLessEqual(
            _rel_l2_error(g_got, g_ref),
            1e-5,
            msg=f"[{tag}] Gradient parity failed on {k_path}",
        )

  def test_yaml_config_registry_and_flops_analytical_accuracy(self):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    base_yml = os.path.join(repo_root, "src", "maxtext", "configs", "base.yml")
    yml_path = os.path.join(repo_root, "src", "maxtext", "configs", "models", "granite-3.0-1b-a400m.yml")
    cfg = m3_config.initialize(
        ["test", yml_path, "per_device_batch_size=2", "max_target_length=1024", "gradient_accumulation_steps=1"],
        unittest=True,
    )
    self.assertEqual(cfg.model_name, "granite-3.0-1b-a400m")
    self.assertTrue(cfg.use_m3_model)

    # Also verify `pyconfig.initialize` (`MaxTextConfig`) loads `granite-3.0-1b-a400m.yml` cleanly
    # and enforces `use_m3_model=True`.
    mt_cfg = pyconfig.initialize(
        [None, base_yml],
        model_name="granite-3.0-1b-a400m",
        per_device_batch_size=2,
        max_target_length=1024,
        gradient_accumulation_steps=1,
        skip_jax_distributed_system=True,
    )
    self.assertTrue(mt_cfg.use_m3_model)
    self.assertEqual(mt_cfg.param_scan_axis, 0)
    self.assertEqual(mt_cfg.load_balance_loss_weight, 0.0)
    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, base_yml],
          model_name="granite-3.0-1b-a400m",
          use_m3_model=False,
          override_model_config=True,
          skip_jax_distributed_system=True,
      )

    total_tflops, weight_tflops, attn_tflops = m3_models.calculate_tflops_per_device(cfg, log=False)
    mt_total_tflops, _, _ = utils_granitemoe.calculate_granitemoe_tflops_per_device(mt_cfg, log=False)
    self.assertAlmostEqual(total_tflops, mt_total_tflops, places=9)
    # Analytical check for Granite-3.0-1B-A400M:
    # B=2, S=1024, H=1024, I=512, L=24, H_q=16, H_kv=8, D=64, E=32, k=8, V=49155
    b, s, h, i, l, h_q, h_kv, d, e, k, v = 2.0, 1024.0, 1024.0, 512.0, 24.0, 16.0, 8.0, 64.0, 32.0, 8.0, 49155.0
    expected_w_fwd = 2 * b * s * (h * v + l * (h * (2 * h_q * d + 2 * h_kv * d) + h * e + k * 3 * h * i))
    expected_a_fwd = l * 2 * b * (s**2) * h_q * d
    expected_total = 3.0 * (expected_w_fwd + expected_a_fwd) / 1e12
    rel_diff = abs(total_tflops - expected_total) / expected_total
    self.assertLess(rel_diff, 5e-3)
    self.assertAlmostEqual(total_tflops, weight_tflops + attn_tflops, places=9)

    # Verify lazy `m3.models.create_model` dispatch with `quant_mode_str="train"`, `nnx.eval_shape`
    # (`model_creation_utils.py:625,725`), and `m3.train.loop.{loss_fn,train_step}` (`m3/train/loop.py:104-117`).
    mini_overrides = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    mini_m3 = m3_models.create_model(
        cfg,
        mesh,
        rngs=nnx.Rngs(0),
        quant_mode_str="train",
        **mini_overrides,
    )
    self.assertIsInstance(mini_m3, modeling_granitemoe.GraniteMoeModel)

    with jax.sharding.set_mesh(mesh):
      from orbax.checkpoint import v1 as ocp_v1  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      if not hasattr(ocp_v1.training, "pathways"):
        ocp_v1.training.pathways = type("_StubPathways", (), {"Snapshotter": object})()  # type: ignore[attr-defined]
      from maxtext.m3.train import loop as m3_loop  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      loop_batch = {
          "inputs": jnp.ones((2, 8), dtype=jnp.int32),
          "positions": jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8)),
          "targets": jnp.ones((2, 8), dtype=jnp.int32),
      }
      loop_loss_0 = m3_loop.loss_fn(mini_m3, loop_batch)
      self.assertEqual(loop_loss_0.shape, ())
      self.assertTrue(bool(np.isfinite(np.asarray(loop_loss_0))))
      optimizer = nnx.Optimizer(mini_m3, optax.adamw(1e-3), wrt=nnx.Param)
      step_loss = m3_loop.train_step(mini_m3, optimizer, loop_batch)
      loop_loss_1 = m3_loop.loss_fn(mini_m3, loop_batch)
      self.assertTrue(bool(np.isfinite(np.asarray(step_loss))))
      self.assertLess(float(loop_loss_1), float(loop_loss_0))

    mesh_4 = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=_get_devices(4))
    abs_m3 = nnx.eval_shape(
        lambda: m3_models.create_model(cfg, mesh_4, rngs=nnx.Rngs(0), quant_mode_str="train", **mini_overrides)
    )
    self.assertIs(abs_m3.mesh, mesh_4)
    for path, var in _flatten_nnx_vars(nnx.state(abs_m3, nnx.Param)):
      self.assertIsNotNone(var.get_metadata("out_sharding", None), f"Missing out_sharding in eval_shape at {path}")


# ==============================================================================
# Suite 2: Torch-Free Committed Mini-Twin Golden Parity & `PerturbSpec` Controls
# ==============================================================================

_GOLDEN_JSON_PATH = os.path.join(os.path.dirname(__file__), "testdata", "granitemoe_hf_golden.json")


def _build_deterministic_hf_weights(cfg: Mapping[str, Any], *, tie: bool, seed: int) -> dict[str, np.ndarray]:
  """Reproduces the exact float32 HF state_dict used to generate `granitemoe_hf_golden.json` without PyTorch."""
  rng = np.random.default_rng(seed)
  v, h, inter = int(cfg["vocab_size"]), int(cfg["hidden_size"]), int(cfg["intermediate_size"])
  hq, hkv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
  d = h // hq
  e = int(cfg["num_local_experts"])
  sd: dict[str, np.ndarray] = {}
  sd["model.embed_tokens.weight"] = (rng.standard_normal((v, h)) * 0.02).astype(np.float32)
  sd["model.norm.weight"] = (1.0 + 0.15 * rng.standard_normal((h,))).astype(np.float32)
  if not tie:
    sd["lm_head.weight"] = (rng.standard_normal((v, h)) * 0.02).astype(np.float32)
  for idx in range(int(cfg["num_hidden_layers"])):
    pfx = f"model.layers.{idx}"
    sd[f"{pfx}.input_layernorm.weight"] = (1.0 + 0.15 * rng.standard_normal((h,))).astype(np.float32)
    sd[f"{pfx}.post_attention_layernorm.weight"] = (1.0 + 0.15 * rng.standard_normal((h,))).astype(np.float32)
    sd[f"{pfx}.self_attn.q_proj.weight"] = (rng.standard_normal((hq * d, h)) * 0.05).astype(np.float32)
    sd[f"{pfx}.self_attn.k_proj.weight"] = (rng.standard_normal((hkv * d, h)) * 0.05).astype(np.float32)
    sd[f"{pfx}.self_attn.v_proj.weight"] = (rng.standard_normal((hkv * d, h)) * 0.05).astype(np.float32)
    sd[f"{pfx}.self_attn.o_proj.weight"] = (rng.standard_normal((h, hq * d)) * 0.05).astype(np.float32)
    sd[f"{pfx}.block_sparse_moe.router.layer.weight"] = (rng.standard_normal((e, h)) * 0.1).astype(np.float32)
    sd[f"{pfx}.block_sparse_moe.input_linear.weight"] = (rng.standard_normal((e, 2 * inter, h)) * 0.05).astype(np.float32)
    sd[f"{pfx}.block_sparse_moe.output_linear.weight"] = (rng.standard_normal((e, h, inter)) * 0.05).astype(np.float32)
  return sd


class GraniteMoeGoldenParityAndPerturbTest(absltest.TestCase):
  """Torch-free HF mini-twin forward/loss/backward parity & `PerturbSpec` negative controls for PR CI."""

  def _load_golden(self) -> dict[str, Any]:
    with open(_GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
      return json.load(f)

  def test_golden_e2e_logits_loss_aux_and_backward_grad_parity(self):
    payload = self._load_golden()
    cfg_raw = payload["config"]
    tokens_jax = jnp.asarray(payload["tokens"], dtype=jnp.int32)
    mask_np = np.asarray(payload["attention_mask"], dtype=np.int32)
    mask_jax = jnp.asarray(mask_np, dtype=jnp.int32)
    valid_bool = mask_np.astype(bool)
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))

    for label in ("tied", "untied"):
      case = payload["cases"][label]
      tie = bool(case["tie_word_embeddings"])
      cfg = modeling_granitemoe.GraniteMoeConfig(
          tie_word_embeddings=tie,
          dtype="float32",
          weight_dtype="float32",
          scan_layers=True,
          remat_policy="none",
          matmul_precision="highest",
          **cfg_raw,
      )
      m3_model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
      np_sd = _build_deterministic_hf_weights(cfg_raw, tie=tie, seed=int(case["seed"]))
      checkpoint_mappings_granitemoe.load_hf_weights_into_model(m3_model, np_sd)

      with jax.sharding.set_mesh(mesh):
        m3_masked_logits, m3_masked_aux = m3_model(tokens_jax, attention_mask=mask_jax, return_aux=True)
        m3_ce_masked, _, _ = modeling_granitemoe.compute_granitemoe_cross_entropy_loss(
            m3_masked_logits[:, :-1, :], tokens_jax[:, 1:], mask_jax[:, 1:]
        )

        def _unpadded_loss(mod: modeling_granitemoe.GraniteMoeModel, aux_coef: float = cfg.router_aux_loss_coef):
          logits_u, aux_u = mod(tokens_jax, return_aux=True)
          ce_u, _, _ = modeling_granitemoe.compute_granitemoe_cross_entropy_loss(logits_u[:, :-1, :], tokens_jax[:, 1:])
          return ce_u + jnp.float32(aux_coef) * aux_u["moe_lb_loss"], logits_u

        (m3_tot, m3_unpadded_logits), m3_grads = nnx.value_and_grad(_unpadded_loss, has_aux=True)(m3_model)

      ref_masked_logits = np.asarray(case["masked_logits"], dtype=np.float32)
      ref_unpadded_logits = np.asarray(case["unpadded_logits"], dtype=np.float32)
      self.assertLessEqual(
          _rel_l2_error(np.asarray(m3_masked_logits)[valid_bool], ref_masked_logits[valid_bool]),
          1e-5,
          msg=f"[{label}] Masked logits parity failed",
      )
      self.assertLessEqual(
          _rel_l2_error(np.asarray(m3_unpadded_logits), ref_unpadded_logits),
          1e-5,
          msg=f"[{label}] Unpadded logits parity failed",
      )
      np.testing.assert_allclose(float(m3_masked_aux["moe_lb_loss"]), float(case["masked_lb_loss"]), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(float(m3_ce_masked), float(case["masked_ce_loss"]), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(float(m3_tot), float(case["unpadded_total_loss"]), rtol=1e-5, atol=1e-5)

      m3_grad_hf = checkpoint_mappings_granitemoe.m3_pure_dict_to_hf_weights(
          nnx.to_pure_dict(m3_grads), cfg, naming="hub"
      )
      for key, ref_g_list in case["unpadded_grads"].items():
        ref_g = np.asarray(ref_g_list, dtype=np.float32)
        self.assertIn(key, m3_grad_hf)
        self.assertLessEqual(
            _rel_l2_error(m3_grad_hf[key], ref_g),
            1e-5,
            msg=f"[{label}] Backward grad mismatch vs committed HF golden at {key}",
        )

  def test_golden_perturb_spec_negative_controls_fire_without_torch(self):
    payload = self._load_golden()
    cfg_raw = payload["config"]
    case = payload["cases"]["tied"]
    cfg = modeling_granitemoe.GraniteMoeConfig(
        tie_word_embeddings=True,
        dtype="float32",
        weight_dtype="float32",
        scan_layers=True,
        remat_policy="none",
        matmul_precision="highest",
        **cfg_raw,
    )
    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    base_model = modeling_granitemoe.create_granitemoe_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
    checkpoint_mappings_granitemoe.load_hf_weights_into_model(
        base_model, _build_deterministic_hf_weights(cfg_raw, tie=True, seed=int(case["seed"]))
    )

    tokens_jax = jnp.asarray(payload["tokens"], dtype=jnp.int32)
    ref_logits = np.asarray(case["unpadded_logits"], dtype=np.float32)

    def _assert_parity(mod: modeling_granitemoe.GraniteMoeModel) -> None:
      with jax.sharding.set_mesh(mesh):
        got_logits = mod(tokens_jax)
      err = _rel_l2_error(np.asarray(got_logits), ref_logits)
      if err > 1e-5:
        raise AssertionError(f"Parity failed as expected: rel_l2={err:.4e}")

    _assert_parity(base_model)

    def _clone_with_cfg(**cfg_overrides: Any) -> modeling_granitemoe.GraniteMoeModel:
      new_cfg = dataclasses.replace(base_model.config, **cfg_overrides)
      cloned = modeling_granitemoe.create_granitemoe_model(new_cfg, mesh=mesh, rngs=nnx.Rngs(0))
      nnx.update(cloned, nnx.clone(nnx.state(base_model)))
      return cloned

    def _perturb_router(_mod: modeling_granitemoe.GraniteMoeModel) -> modeling_granitemoe.GraniteMoeModel:
      cloned = _clone_with_cfg()
      r_var = cloned.layers.block_sparse_moe.router.weight
      noise = jax.random.normal(jax.random.PRNGKey(601), r_var[...].shape, dtype=r_var[...].dtype)
      r_var[...] = _scale_into_rel_band(r_var[...], noise, target_rel_norm=1.5e-2)
      return cloned

    def _perturb_expert_gate_up(_mod: modeling_granitemoe.GraniteMoeModel) -> modeling_granitemoe.GraniteMoeModel:
      cloned = _clone_with_cfg()
      gu_var = cloned.layers.block_sparse_moe.experts.gate_up_proj
      noise = jax.random.normal(jax.random.PRNGKey(602), gu_var[...].shape, dtype=gu_var[...].dtype)
      gu_var[...] = _scale_into_rel_band(gu_var[...], noise, target_rel_norm=1.5e-2)
      return cloned

    perturb_specs = [
        PerturbSpec("embedding_multiplier", lambda _m: _clone_with_cfg(embedding_multiplier=1.0)),
        PerturbSpec(
            "attention_multiplier", lambda _m: _clone_with_cfg(attention_multiplier=1.0 / math.sqrt(cfg.head_dim))
        ),
        PerturbSpec("residual_multiplier", lambda _m: _clone_with_cfg(residual_multiplier=1.0)),
        PerturbSpec("logits_scaling", lambda _m: _clone_with_cfg(logits_scaling=1.0)),
        PerturbSpec("router.weight", _perturb_router),
        PerturbSpec("experts.gate_up_proj", _perturb_expert_gate_up),
    ]

    for spec in perturb_specs:
      perturbed_model = spec.apply(copy.copy(base_model))
      with self.assertRaises(AssertionError, msg=f"Torch-free PerturbSpec {spec.name!r} failed to trip!"):
        _assert_parity(perturbed_model)

  def test_extract_hf_router_logits_handles_legacy_and_v5_22_tuple_layouts(self):
    num_tokens, num_experts, top_k = 20, 8, 2
    expected_logits = np.arange(num_tokens * num_experts, dtype=np.float32).reshape(num_tokens, num_experts)

    # Legacy HF GraniteMoeTopKGating (transformers <= 5.12.1):
    # (index_sorted_experts, batch_index, batch_gates, expert_size, logits)
    legacy_out = (
        np.zeros((num_tokens * top_k,), dtype=np.int64),
        np.zeros((num_tokens * top_k,), dtype=np.int64),
        np.ones((num_tokens * top_k,), dtype=np.float32),
        [5] * num_experts,
        expected_logits,
    )
    self.assertIs(_extract_hf_router_logits(legacy_out, num_experts), expected_logits)

    # Modern HF GraniteMoeTopKRouter (transformers >= 5.22.1):
    # (router_logits, top_k_weights, top_k_index) where out[-1] is Long!
    modern_out = (
        expected_logits,
        np.ones((num_tokens, top_k), dtype=np.float32),
        np.zeros((num_tokens, top_k), dtype=np.int64),
    )
    self.assertIs(_extract_hf_router_logits(modern_out, num_experts), expected_logits)

    # Direct tensor pass-through:
    self.assertIs(_extract_hf_router_logits(expected_logits, num_experts), expected_logits)


# ==============================================================================
# Suite 3: Live PyTorch Weight-Copy Parity & `PerturbSpec` Negative Controls
# ==============================================================================


def _extract_hf_router_logits(out: Any, num_experts: int) -> Any:
  """Extracts `(num_tokens, num_experts)` float router logits across HF transformers versions.

  - `transformers <= 5.12.1` (`GraniteMoeTopKGating`) returns
    `(index_sorted_experts, batch_index, batch_gates, expert_size, logits)` with `logits` at `out[-1]`.
  - `transformers >= 5.22.1` (`GraniteMoeTopKRouter`) returns
    `(router_logits, top_k_weights, top_k_index)` with float `router_logits` at `out[0]` and integer
    `top_k_index` at `out[-1]`.
  """
  if not isinstance(out, tuple):
    return out
  for item in out:
    if getattr(item, "ndim", None) != 2 or item.shape[-1] != num_experts:
      continue
    if HAS_TORCH and torch.is_tensor(item):
      if torch.is_floating_point(item):
        return item
    elif hasattr(item, "dtype") and np.issubdtype(np.asarray(item).dtype, np.floating):
      return item
  raise ValueError(f"Could not find (num_tokens, {num_experts}) float router_logits in router output: {out!r}")


@unittest.skipUnless(HAS_TORCH, "PyTorch + transformers required for HF reference parity tests.")
class GraniteMoePyTorchParityAndPerturbTest(absltest.TestCase):
  """Verifies module-level, E2E forward/backward/loss parity, and `PerturbSpec` negative controls vs HF PyTorch."""

  def _make_hf_config(self, m3_cfg: modeling_granitemoe.GraniteMoeConfig) -> Any:
    """Constructs a matching HuggingFace GraniteMoeConfig in eager attention mode."""
    hf_cfg = hf_granitemoe_cfg.GraniteMoeConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=m3_cfg.vocab_size,
        hidden_size=m3_cfg.hidden_size,
        intermediate_size=m3_cfg.intermediate_size,
        num_hidden_layers=m3_cfg.num_hidden_layers,
        num_attention_heads=m3_cfg.num_attention_heads,
        num_key_value_heads=m3_cfg.num_key_value_heads,
        num_local_experts=m3_cfg.num_local_experts,
        num_experts_per_tok=m3_cfg.num_experts_per_tok,
        rms_norm_eps=m3_cfg.rms_norm_eps,
        rope_theta=m3_cfg.rope_theta,
        attention_bias=m3_cfg.attention_bias,
        attention_dropout=0.0,
        hidden_act=m3_cfg.hidden_act,
        tie_word_embeddings=m3_cfg.tie_word_embeddings,
        embedding_multiplier=m3_cfg.embedding_multiplier,
        attention_multiplier=m3_cfg.attention_multiplier,
        residual_multiplier=m3_cfg.residual_multiplier,
        logits_scaling=m3_cfg.logits_scaling,
        router_aux_loss_coef=m3_cfg.router_aux_loss_coef,
    )
    hf_cfg._attn_implementation = "eager"  # pylint: disable=protected-access
    return hf_cfg

  def _build_paired_models(
      self,
      m3_cfg: modeling_granitemoe.GraniteMoeConfig,
      *,
      seed: int = 123,
  ) -> tuple[Any, modeling_granitemoe.GraniteMoeModel, jax.sharding.Mesh]:
    """Builds weight-identical HF PyTorch and MaxText M3 GraniteMoe models."""
    torch.manual_seed(seed)
    hf_cfg = self._make_hf_config(m3_cfg)
    hf_model = hf_granitemoe_mod.GraniteMoeForCausalLM(hf_cfg).to(torch.float32).eval()
    # Randomize norm weights away from all-ones so RMSNorm scale bugs cannot hide.
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name:
          param.copy_(1.0 + 0.15 * torch.randn_like(param))

    mesh = m3_sharding.create_mesh(devices=_get_devices(1))
    m3_model = modeling_granitemoe.create_granitemoe_model(m3_cfg, mesh=mesh, rngs=nnx.Rngs(0))
    checkpoint_mappings_granitemoe.load_hf_weights_into_model(m3_model, hf_model.state_dict())
    return hf_model, m3_model, mesh

  def test_layerwise_rmsnorm_attention_moe_and_decoder_layer_parity(self):
    cfg = _make_mini_config(num_hidden_layers=1)
    hf_model, m3_model, mesh = self._build_paired_models(cfg, seed=101)
    hf_layer = hf_model.model.layers[0]

    rng = np.random.default_rng(55)
    x_np = rng.standard_normal((2, 9, cfg.hidden_size)).astype(np.float32)
    x_pt = torch.tensor(x_np, dtype=torch.float32)
    x_jax = jnp.asarray(x_np)
    pos_pt = torch.arange(9, dtype=torch.long).unsqueeze(0).expand(2, -1)
    pos_jax = jnp.broadcast_to(jnp.arange(9, dtype=jnp.int32)[None, :], (2, 9))

    # Slice stacked single-layer M3 decoder block for direct module-by-module comparison.
    gdef, p_st, np_st = nnx.split(m3_model.layers, nnx.Param, ...)
    m3_layer = nnx.merge(
        gdef,
        modeling_granitemoe._slice_stacked_state(p_st, 0),  # pylint: disable=protected-access
        modeling_granitemoe._slice_stacked_state(np_st, 0),  # pylint: disable=protected-access
    )

    with torch.no_grad(), jax.sharding.set_mesh(mesh):
      # 1. RMSNorm
      pt_norm = hf_layer.input_layernorm(x_pt).numpy()
      jax_norm = np.asarray(m3_layer.input_layernorm(x_jax))
      self.assertLessEqual(_rel_l2_error(jax_norm, pt_norm), 1e-6)

      # 2. GQA Attention (with RoPE + attention_multiplier)
      pos_emb = hf_model.model.rotary_emb(x_pt, pos_pt)
      causal_4d = torch.triu(torch.full((1, 1, 9, 9), torch.finfo(torch.float32).min), diagonal=1)
      pt_attn, _ = hf_layer.self_attn(x_pt, position_embeddings=pos_emb, attention_mask=causal_4d)
      jax_attn, _ = m3_layer.self_attn(x_jax, pos_jax)
      self.assertLessEqual(_rel_l2_error(np.asarray(jax_attn), pt_attn.numpy()), 1e-5)

      # 3. Sparse MoE (non-square H=64, I=48 -> 2*I=96 != H)
      pt_moe_res = hf_layer.block_sparse_moe(x_pt)
      pt_moe = pt_moe_res[0] if isinstance(pt_moe_res, tuple) else pt_moe_res
      jax_moe, _, _ = m3_layer.block_sparse_moe(x_jax)
      self.assertLessEqual(_rel_l2_error(np.asarray(jax_moe), pt_moe.numpy()), 1e-5)

      # 4. Full Decoder Layer (including residual_multiplier = 0.22 on both sublayers)
      pt_dec = hf_layer(x_pt, attention_mask=causal_4d, position_ids=pos_pt, position_embeddings=pos_emb)
      pt_dec_h = pt_dec[0] if isinstance(pt_dec, tuple) else pt_dec
      jax_dec_h, _, _, _ = m3_layer(x_jax, pos_jax)
      self.assertLessEqual(_rel_l2_error(np.asarray(jax_dec_h), pt_dec_h.numpy()), 1e-5)

  def _run_e2e_logits_loss_aux_loss_and_backward_grad_parity(
      self,
      *,
      tie_values: tuple[bool, ...] = (True, False),
      num_hidden_layers: int = 2,
  ) -> None:
    """Runs E2E forward logits, masked/unmasked loss, Switch aux loss, and backward gradient parity."""
    for tie in tie_values:
      cfg = _make_mini_config(
          tie_word_embeddings=tie,
          num_hidden_layers=num_hidden_layers,
          router_aux_loss_coef=0.05,
      )
      hf_model, m3_model, mesh = self._build_paired_models(cfg, seed=202 if tie else 303)
      hf_model.train(False)

      tokens_np = np.array(
          [
              [5, 19, 42, 7, 88, 13, 64, 31, 0, 0],
              [12, 77, 3, 200, 14, 9, 51, 6, 44, 0],
          ],
          dtype=np.int64,
      )
      attn_mask_np = np.array(
          [
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          ],
          dtype=np.int64,
      )

      # Capture HF router_logits via forward hooks across both legacy (<=5.12.1) and v5.22+ router return tuples.
      num_experts = cfg.num_local_experts
      captured_router_logits: list[Any] = []
      hooks = []
      for layer in hf_model.model.layers:
        hooks.append(
            layer.block_sparse_moe.router.register_forward_hook(
                lambda _mod, _inp, out, buf=captured_router_logits, ne=num_experts: buf.append(
                    _extract_hf_router_logits(out, ne)
                )
            )
        )

      try:
        with torch.enable_grad():
          hf_model.zero_grad(set_to_none=True)
          tokens_pt = torch.tensor(tokens_np, dtype=torch.long)
          mask_pt = torch.tensor(attn_mask_np, dtype=torch.long)
          labels_pt = tokens_pt.clone()
          labels_pt[mask_pt == 0] = -100

          hf_out = hf_model(input_ids=tokens_pt, attention_mask=mask_pt, labels=labels_pt)
          hf_logits = hf_out.logits.detach().numpy()
          hf_ce_loss = float(hf_out.loss.detach().item())
          hf_lb_loss_t = hf_granitemoe_mod.load_balancing_loss_func(
              tuple(captured_router_logits),
              cfg.num_local_experts,
              cfg.num_experts_per_tok,
              attention_mask=mask_pt,
          )
          hf_total_loss = hf_out.loss + cfg.router_aux_loss_coef * hf_lb_loss_t
          hf_total_loss.backward()
      finally:
        for h in hooks:
          h.remove()

      tokens_jax = jnp.asarray(tokens_np, dtype=jnp.int32)
      mask_jax = jnp.asarray(attn_mask_np, dtype=jnp.int32)

      with jax.sharding.set_mesh(mesh):
        m3_logits, m3_aux = m3_model(tokens_jax, attention_mask=mask_jax, return_aux=True)
        # Compare valid non-padding token logits:
        valid_bool = attn_mask_np.astype(bool)
        self.assertLessEqual(_rel_l2_error(np.asarray(m3_logits)[valid_bool], hf_logits[valid_bool]), 1e-5)

        # Compare Switch load-balancing loss (both masked and unmasked):
        np.testing.assert_allclose(
            float(m3_aux["moe_lb_loss"]),
            float(hf_lb_loss_t.detach().item()),
            rtol=1e-5,
            atol=1e-5,
        )

        # Compare pre-shifted CE + aux total loss and backward parameter gradients on full unpadded batch:
        hf_model.zero_grad(set_to_none=True)
        captured_unpadded: list[Any] = []
        hooks2 = [
            layer.block_sparse_moe.router.register_forward_hook(
                lambda _m, _i, o, buf=captured_unpadded, ne=num_experts: buf.append(_extract_hf_router_logits(o, ne))
            )
            for layer in hf_model.model.layers
        ]
        try:
          with torch.enable_grad():
            out2 = hf_model(input_ids=tokens_pt, labels=tokens_pt)
            lb2 = hf_granitemoe_mod.load_balancing_loss_func(
                tuple(captured_unpadded), cfg.num_local_experts, cfg.num_experts_per_tok, attention_mask=None
            )
            tot_pt = out2.loss + cfg.router_aux_loss_coef * lb2
            tot_pt.backward()
        finally:
          for h in hooks2:
            h.remove()

        def _m3_full_seq_loss(
            mod: modeling_granitemoe.GraniteMoeModel,
            toks_in: jax.Array = tokens_jax,
            aux_coef: float = cfg.router_aux_loss_coef,
        ) -> jax.Array:
          logits_full, aux_full = mod(toks_in, return_aux=True)
          ce, _, _ = modeling_granitemoe.compute_granitemoe_cross_entropy_loss(logits_full[:, :-1, :], toks_in[:, 1:])
          return ce + jnp.float32(aux_coef) * aux_full["moe_lb_loss"]

        m3_tot, m3_grads = nnx.value_and_grad(_m3_full_seq_loss)(m3_model)

      np.testing.assert_allclose(float(m3_tot), float(tot_pt.detach().item()), rtol=1e-5, atol=1e-5)
      # Pre-shifted compute_granitemoe_cross_entropy_loss on masked batch matches hf_ce_loss:
      m3_ce_masked, _, _ = modeling_granitemoe.compute_granitemoe_cross_entropy_loss(
          m3_logits[:, :-1, :], tokens_jax[:, 1:], mask_jax[:, 1:]
      )
      np.testing.assert_allclose(float(m3_ce_masked), hf_ce_loss, rtol=1e-5, atol=1e-5)

      # Verify E2E backward gradient parity across embedding, attention, router, experts, and final norm!
      m3_grad_hf_layout = checkpoint_mappings_granitemoe.m3_pure_dict_to_hf_weights(
          nnx.to_pure_dict(m3_grads), cfg, naming="hub"
      )
      hf_grad_dict = checkpoint_mappings_granitemoe.normalize_granitemoe_hf_keys(
          {name: p.grad.detach().numpy() for name, p in hf_model.named_parameters() if p.grad is not None}
      )
      for key, pt_g in hf_grad_dict.items():
        if tie and key == "lm_head.weight":
          continue
        self.assertIn(key, m3_grad_hf_layout)
        self.assertLessEqual(
            _rel_l2_error(m3_grad_hf_layout[key], pt_g),
            1e-5,
            msg=f"[tie={tie}] Backward grad mismatch at {key}",
        )

  def test_e2e_logits_loss_aux_loss_and_backward_grad_parity(self):
    self._run_e2e_logits_loss_aux_loss_and_backward_grad_parity()

  def test_e2e_backward_parity_when_global_torch_grad_disabled(self):
    # Regression guard for b/583197649: simulate a prior test module on the same `pytest-xdist`
    # worker leaving autograd disabled without poisoning subsequent tests.
    with torch.no_grad():
      self._run_e2e_logits_loss_aux_loss_and_backward_grad_parity(tie_values=(True,), num_hidden_layers=1)

  def test_perturb_spec_negative_controls_fire_on_all_6_targets(self):
    cfg = _make_mini_config()
    hf_model, base_model, mesh = self._build_paired_models(cfg, seed=404)
    tokens_np = np.random.default_rng(91).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int64)
    with torch.no_grad():
      ref_logits = hf_model(input_ids=torch.tensor(tokens_np, dtype=torch.long)).logits.numpy()

    tokens_jax = jnp.asarray(tokens_np, dtype=jnp.int32)

    def _assert_parity(mod: modeling_granitemoe.GraniteMoeModel) -> None:
      with jax.sharding.set_mesh(mesh):
        got_logits = mod(tokens_jax)
      err = _rel_l2_error(np.asarray(got_logits), ref_logits)
      if err > 1e-5:
        raise AssertionError(f"Parity failed as expected: rel_l2={err:.4e}")

    # Baseline unperturbed model MUST pass parity.
    _assert_parity(base_model)

    def _clone_with_cfg(**cfg_overrides: Any) -> modeling_granitemoe.GraniteMoeModel:
      new_cfg = dataclasses.replace(base_model.config, **cfg_overrides)
      cloned = modeling_granitemoe.create_granitemoe_model(new_cfg, mesh=mesh, rngs=nnx.Rngs(0))
      nnx.update(cloned, nnx.clone(nnx.state(base_model)))
      return cloned

    def _perturb_router(mod: modeling_granitemoe.GraniteMoeModel) -> modeling_granitemoe.GraniteMoeModel:
      cloned = _clone_with_cfg()
      del mod
      r_var = cloned.layers.block_sparse_moe.router.weight
      noise = jax.random.normal(jax.random.PRNGKey(501), r_var[...].shape, dtype=r_var[...].dtype)
      # Large enough router perturbation to flip top-k expert assignments / softmax weights.
      r_var[...] = _scale_into_rel_band(r_var[...], noise, target_rel_norm=1.5e-2)
      return cloned

    def _perturb_expert_gate_up(mod: modeling_granitemoe.GraniteMoeModel) -> modeling_granitemoe.GraniteMoeModel:
      cloned = _clone_with_cfg()
      del mod
      gu_var = cloned.layers.block_sparse_moe.experts.gate_up_proj
      noise = jax.random.normal(jax.random.PRNGKey(502), gu_var[...].shape, dtype=gu_var[...].dtype)
      gu_var[...] = _scale_into_rel_band(gu_var[...], noise, target_rel_norm=1.5e-2)
      return cloned

    perturb_specs = [
        PerturbSpec("embedding_multiplier", lambda _m: _clone_with_cfg(embedding_multiplier=1.0)),
        PerturbSpec(
            "attention_multiplier", lambda _m: _clone_with_cfg(attention_multiplier=1.0 / math.sqrt(cfg.head_dim))
        ),
        PerturbSpec("residual_multiplier", lambda _m: _clone_with_cfg(residual_multiplier=1.0)),
        PerturbSpec("logits_scaling", lambda _m: _clone_with_cfg(logits_scaling=1.0)),
        PerturbSpec("router.weight", _perturb_router),
        PerturbSpec("experts.gate_up_proj", _perturb_expert_gate_up),
    ]

    for spec in perturb_specs:
      perturbed_model = spec.apply(copy.copy(base_model))
      with self.assertRaises(AssertionError, msg=f"PerturbSpec {spec.name!r} failed to trip parity assertion!"):
        _assert_parity(perturbed_model)

  def test_committed_golden_matches_live_hf_pytorch(self):
    with open(_GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
      payload = json.load(f)
    cfg_raw = payload["config"]
    tokens_pt = torch.tensor(payload["tokens"], dtype=torch.long)
    mask_pt = torch.tensor(payload["attention_mask"], dtype=torch.long)
    labels_pt = tokens_pt.clone()
    labels_pt[mask_pt == 0] = -100
    valid_bool = np.asarray(payload["attention_mask"], dtype=bool)

    for label in ("tied", "untied"):
      case = payload["cases"][label]
      tie = bool(case["tie_word_embeddings"])
      m3_cfg = modeling_granitemoe.GraniteMoeConfig(tie_word_embeddings=tie, **cfg_raw)
      hf_model = hf_granitemoe_mod.GraniteMoeForCausalLM(self._make_hf_config(m3_cfg)).to(torch.float32).eval()
      np_sd = _build_deterministic_hf_weights(cfg_raw, tie=tie, seed=int(case["seed"]))
      with torch.no_grad():
        for name, param in hf_model.named_parameters():
          if tie and name == "lm_head.weight":
            continue
          hub_name = next(iter(checkpoint_mappings_granitemoe.normalize_granitemoe_hf_keys({name: None})))
          param.copy_(torch.from_numpy(np_sd[hub_name]))
        masked_out = hf_model(input_ids=tokens_pt, attention_mask=mask_pt, labels=labels_pt)
        unpadded_out = hf_model(input_ids=tokens_pt, labels=tokens_pt)

      self.assertLessEqual(
          _rel_l2_error(
              masked_out.logits.numpy()[valid_bool], np.asarray(case["masked_logits"], dtype=np.float32)[valid_bool]
          ),
          1e-6,
      )
      self.assertLessEqual(
          _rel_l2_error(unpadded_out.logits.numpy(), np.asarray(case["unpadded_logits"], dtype=np.float32)),
          1e-6,
      )
      np.testing.assert_allclose(float(masked_out.loss.item()), float(case["masked_ce_loss"]), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
