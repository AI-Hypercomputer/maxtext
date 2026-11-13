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

"""Unit and numerical parity tests for `maxtext.m3.infra.remat` and savepoints."""

from collections.abc import Sequence
import dataclasses
import importlib.util
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import checkpoint_policies as cp
import jax.extend as jex
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core.sharding import DefaultSharding, create_mesh
from maxtext.m3.infra import remat as m3_remat
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
from maxtext.utils import model_creation_utils
from tests.utils import glm5_test_utils
from tests.utils.test_helpers import get_test_config_path


@dataclasses.dataclass
class _PolicyStub:
  """Minimal config stub carrying rematerialization fields for unit tests."""

  remat_policy: Any = "none"
  tensors_on_device: Sequence[str] = ()
  tensors_to_offload: Sequence[str] = ()
  scan_layers: bool = True
  param_scan_axis: int = 0


def _iter_all_eqns(jaxpr: jex.core.Jaxpr):
  """Yields `(depth, eqn)` recursively across nested jaxprs."""
  for eqn in jaxpr.eqns:
    yield 0, eqn
    for val in eqn.params.values():
      for sub in m3_remat._jaxprs_in(val):  # pylint: disable=protected-access
        for sub_depth, sub_eqn in _iter_all_eqns(sub):
          yield sub_depth + 1, sub_eqn


def _count_primitives(jaxpr: jex.core.Jaxpr, names: set[str]) -> int:
  return sum(1 for _, eqn in _iter_all_eqns(jaxpr) if eqn.primitive.name in names)


def _param_signature(model: nnx.Module) -> list[tuple[tuple[Any, ...], tuple[int, ...], Any]]:
  flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
  return [(tuple(path), tuple(var.shape), var.dtype) for path, var in flat]


def _assert_grad_trees_close(grads_a: Any, grads_b: Any, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
  flat_a = dict(nnx.to_flat_state(grads_a))
  flat_b = dict(nnx.to_flat_state(grads_b))
  assert set(flat_a.keys()) == set(flat_b.keys()), f"Mismatched grad paths: {set(flat_a) ^ set(flat_b)}"
  for path, ga in flat_a.items():
    gb = flat_b[path]
    np.testing.assert_allclose(
        np.asarray(ga[...]),
        np.asarray(gb[...]),
        atol=atol,
        rtol=rtol,
        err_msg=f"Gradient mismatch at {path}",
    )


def _qwen3_mini_config(output_dir: str, **overrides: Any):
  """Builds a fast 2-layer float32 Qwen3-0.6B config for CPU remat tests."""
  args: dict[str, Any] = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "override_model_config": "true",
      "scan_layers": "false",
      "remat_policy": "none",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_logits": "true",
      "attention": "dot_product",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "1",
      "max_target_length": "16",
      "steps": "1",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_remat_test",
  }
  args.update(overrides)
  cli = ["", get_test_config_path()]
  for k, v in args.items():
    if isinstance(v, (list, tuple)):
      cli.append(f"{k}={list(v)}")
    else:
      cli.append(f"{k}={v}")
  return pyconfig.initialize(cli)


def _qwen3_8_mini_config(*, scan_layers: bool = True, remat_policy: str = "none", **overrides: Any):
  """Builds a fast 4-layer float32 Qwen3.8-Flash-Next mini config for CPU tests."""
  kw: dict[str, Any] = {
      "model_name": "qwen3.8-flash-next-mini",
      "use_m3_model": True,
      "override_model_config": True,
      "skip_jax_distributed_system": True,
      "scan_layers": scan_layers,
      "remat_policy": remat_policy,
      "packing": False,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "base_num_decoder_layers": 4,
      "run_name": "qwen3_8_remat_test",
  }
  kw.update(overrides)
  return pyconfig.initialize(
      [None, f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml"],
      **kw,
  )


class _Glm5TrainCfg(glm5_config.Glm5Config):
  """Compact MaxText-compatible GLM-5.3 config for unit tests."""


def _make_glm5_mini_cfg(
    *,
    num_hidden_layers: int = 5,
    scan_layers: bool = True,
    remat_policy: str = "none",
    tensors_on_device: tuple[str, ...] = (),
) -> _Glm5TrainCfg:
  """Constructs a compact float32 GLM-5.3 config for CPU execution across tokamax versions."""
  base_cfg, _ = glm5_test_utils._make_cfg(  # pylint: disable=protected-access
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      dtype="float32",
      weight_dtype="float32",
  )
  cfg_dict = dataclasses.asdict(base_cfg)
  cfg_dict["kda_implementation"] = "xla"
  cfg_dict["remat_policy"] = remat_policy
  if importlib.util.find_spec("tokamax._src.ops.kda") is None:
    cfg_dict["num_hidden_layers"] = 2
    cfg_dict["first_k_dense_replace"] = 1
    cfg_dict["layer_types"] = ("full_attention", "full_attention")
    cfg_dict["indexer_types"] = ("full", "full")
  cfg = _Glm5TrainCfg(**cfg_dict)
  object.__setattr__(cfg, "use_m3_model", True)
  object.__setattr__(cfg, "param_scan_axis", 0)
  object.__setattr__(cfg, "tensors_on_device", list(tensors_on_device))
  object.__setattr__(cfg, "tensors_to_offload", [])
  object.__setattr__(cfg, "quantization", "")
  return cfg


class _ToyLayerA(nnx.Module):
  """Single-projection toy block for heterogeneous `nnx.Sequential` tests."""

  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

  def __call__(self, x: jax.Array, scale: float = 1.0) -> jax.Array:
    return jnp.tanh(self.proj(x)) * scale


class _ToyLayerB(nnx.Module):
  """Two-projection toy block for heterogeneous `nnx.Sequential` tests."""

  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.w1 = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
    self.w2 = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

  def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    y = self.w2(jax.nn.silu(self.w1(x)))
    return y if mask is None else y * mask


class _ToyHeterogeneousModel(nnx.Module):
  """Heterogeneous `[LayerA, LayerB, LayerB]` container testing scan/remat order."""

  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.blocks = nnx.Sequential(
        _ToyLayerA(dim, rngs=rngs),
        _ToyLayerB(dim, rngs=rngs),
        _ToyLayerB(dim, rngs=rngs),
    )

  def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    return self.blocks(x, scale=1.0, mask=mask)


class _ToyStatefulLayer(nnx.Module):
  """Toy module updating a non-`Param` `BatchStat` counter inside `RematSequential`."""

  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
    self.steps = nnx.BatchStat(jnp.zeros((), dtype=jnp.int32))

  def __call__(self, x: jax.Array) -> jax.Array:
    self.steps[...] = self.steps[...] + 1
    return self.proj(x)


class RematPolicyAndContainerTest(parameterized.TestCase):
  """Tests policy resolution, PR #6111/#5878 API compatibility, and path-transparency."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_remat_policies_cover_pr6111_and_standard_names(self):
    expected = {
        "none",
        "full",
        "minimal",
        "minimal_with_context",
        "minimal_flash",
        "save_qkv_proj",
        "save_dot_except_mlp",
        "save_dot_except_mlpwi",
        "save_out_proj",
        "qkv_proj_offloaded",
        "minimal_offloaded",
        "matmul_without_batch",
        "offload_matmul_without_batch",
        "custom",
        "hidden_state_with_offload",
    }
    self.assertEqual(set(m3_remat.REMAT_POLICIES), expected)
    self.assertIsNone(m3_remat.remat_policy(None))
    self.assertIsNone(m3_remat.remat_policy("none"))
    self.assertIsNone(m3_remat.remat_policy("NONE"))
    for name in expected - {"none"}:
      pol = m3_remat.remat_policy(name.upper(), names_which_can_be_saved=["out_proj"])
      self.assertTrue(callable(pol), f"Expected callable for {name}")

    with self.assertRaises(ValueError):
      m3_remat.remat_policy("invalid_policy_name")
    with self.assertRaises(ValueError):
      m3_remat.remat_policy(123)  # type: ignore[arg-type]

  def test_policy_callable_and_resolve_policy_invariants(self):
    self.assertIsNone(m3_remat.policy_callable(None))
    self.assertIs(m3_remat.policy_callable(m3_remat.Remat(policy=None)), cp.nothing_saveable)
    self.assertIs(m3_remat.policy_callable(cp.everything_saveable), cp.everything_saveable)
    with self.assertRaises(TypeError):
      m3_remat.policy_callable("not_a_remat_or_callable")  # type: ignore[arg-type]

    for empty in (None, "", "none", "None"):
      self.assertIsNone(m3_remat.resolve_policy(_PolicyStub(remat_policy=empty)))
    with self.assertRaises(ValueError):
      m3_remat.resolve_policy(_PolicyStub(remat_policy=True))

    remat_min = m3_remat.resolve_policy(_PolicyStub(remat_policy="minimal"))
    remat_ctx = m3_remat.resolve_policy(_PolicyStub(remat_policy="minimal_with_context"))
    assert remat_min is not None and remat_ctx is not None
    assert remat_min.names is not None and remat_ctx.names is not None
    self.assertNotIn("context", remat_min.names)
    self.assertIn("context", remat_ctx.names)

    remat_custom = m3_remat.resolve_policy(
        _PolicyStub(
            remat_policy="custom",
            tensors_on_device=("moe_routed_output",),
            tensors_to_offload=("context",),
        )
    )
    assert remat_custom is not None
    self.assertEqual(remat_custom.names, ("moe_routed_output", "context"))

    # Verify native `m3` `TrainerConfig` field names (`names_which_can_be_saved` / `names_which_can_be_offloaded`).
    trainer_stub = dataclasses.make_dataclass(
        "TrainerStub",
        ["remat_policy", "names_which_can_be_saved", "names_which_can_be_offloaded"],
    )(
        remat_policy="custom",
        names_which_can_be_saved=["moe_routed_output"],
        names_which_can_be_offloaded=["context"],
    )
    remat_trainer = m3_remat.resolve_policy(trainer_stub)
    assert remat_trainer is not None
    self.assertEqual(remat_trainer.names, ("moe_routed_output", "context"))

    # Verify both `pyconfig` custom remat syntaxes (`moe_routed_output=device` and explicit `tensors_on_device`).
    cfg_flag = _qwen3_mini_config(self.output_dir, remat_policy="custom", moe_routed_output="device")
    self.assertIn("moe_routed_output", cfg_flag.tensors_on_device)
    cfg_list = _qwen3_mini_config(self.output_dir, remat_policy="custom", tensors_on_device=["moe_routed_output"])
    self.assertEqual(list(cfg_list.tensors_on_device), ["moe_routed_output"])

  def test_remat_sequential_chaining_root_replacement_and_state_updates(self):
    rngs = nnx.Rngs(0)
    stateful = _ToyStatefulLayer(8, rngs=rngs)
    seq = nnx.Sequential(
        stateful,
        lambda h: (h, jnp.ones_like(h)),
        lambda a, b: {"x": a + b},
        lambda x: x * 2.0,
    )
    remat_seq = m3_remat.apply_remat(seq, m3_remat.Remat(policy=cp.nothing_saveable))
    self.assertIsInstance(remat_seq, m3_remat.RematSequential)
    x = jnp.ones((2, 8), dtype=jnp.float32)
    y = remat_seq(x)
    self.assertEqual(y.shape, (2, 8))
    self.assertEqual(int(stateful.steps[...]), 1)

    # Disabling remat falls back to `nnx.Sequential.__call__` cleanly.
    m3_remat.apply_remat(remat_seq, None)
    y2 = remat_seq(x)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y2), atol=1e-6, rtol=1e-6)
    self.assertEqual(int(stateful.steps[...]), 2)

  def test_heterogeneous_sequential_scan_and_remat_order_commutativity(self):
    m_remat_then_scan = _ToyHeterogeneousModel(16, rngs=nnx.Rngs(7))
    m_scan_then_remat = _ToyHeterogeneousModel(16, rngs=nnx.Rngs(7))
    remat = m3_remat.Remat(policy=cp.nothing_saveable)
    scan_cfg = _PolicyStub(scan_layers=True, param_scan_axis=0)

    m3_remat.apply_remat(m_remat_then_scan, remat)
    m3_scan.maybe_apply_scan(m_remat_then_scan, scan_cfg)

    m3_scan.maybe_apply_scan(m_scan_then_remat, scan_cfg)
    m3_remat.apply_remat(m_scan_then_remat, remat)

    self.assertIs(type(m_remat_then_scan.blocks), nnx.Sequential)
    self.assertIs(type(m_scan_then_remat.blocks), nnx.Sequential)
    self.assertEqual(_param_signature(m_remat_then_scan), _param_signature(m_scan_then_remat))
    for child_a, child_b in zip(m_remat_then_scan.blocks.layers, m_scan_then_remat.blocks.layers):
      self.assertIsInstance(child_a, m3_scan.ScannedLayers)
      self.assertIsInstance(child_b, m3_scan.ScannedLayers)
      self.assertEqual(child_a.remat, remat)
      self.assertEqual(child_b.remat, remat)

    x = jax.random.normal(jax.random.key(3), (2, 4, 16), dtype=jnp.float32)
    mask = jnp.ones((2, 4, 1), dtype=jnp.float32)
    loss_a, grad_a = nnx.value_and_grad(lambda m: jnp.mean(m(x, mask=mask) ** 2))(m_remat_then_scan)
    loss_b, grad_b = nnx.value_and_grad(lambda m: jnp.mean(m(x, mask=mask) ** 2))(m_scan_then_remat)
    np.testing.assert_allclose(loss_a, loss_b, atol=1e-6, rtol=1e-6)
    _assert_grad_trees_close(grad_a, grad_b, atol=1e-6, rtol=1e-6)

  @parameterized.parameters(False, True)
  def test_qwen3_param_paths_invariant_across_policies_and_transform_order(self, scan_layers: bool):
    mesh = create_mesh(devices=jax.devices()[:1])
    base_cfg = _qwen3_mini_config(self.output_dir, scan_layers=str(scan_layers).lower(), remat_policy="none")
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      base_model = model_creation_utils.create_model(base_cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      ref_sig = _param_signature(base_model)

      for policy in ("full", "minimal", "minimal_with_context", "custom"):
        cfg = _qwen3_mini_config(
            self.output_dir,
            scan_layers=str(scan_layers).lower(),
            remat_policy=policy,
            tensors_on_device=["out_proj"] if policy == "custom" else [],
        )
        m_post_scan = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
        self.assertEqual(_param_signature(m_post_scan), ref_sig)

        # Reverse order: apply remat first, then scan.
        m_pre_scan = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
        m_pre_scan = m3_remat.maybe_apply_remat(m_pre_scan, cfg, model_mode=MODEL_MODE_TRAIN)
        m_pre_scan = m3_scan.maybe_apply_scan(m_pre_scan, cfg)
        self.assertEqual(_param_signature(m_pre_scan), ref_sig)

  def test_single_layer_scan_retains_while_loop_barrier_in_compiled_hlo(self):
    model = _ToyHeterogeneousModel(16, rngs=nnx.Rngs(11))
    # First segment `_ToyLayerA` has length 1 (`num_layers == 1`), second `_ToyLayerB` has length 2.
    m3_scan.maybe_apply_scan(model, _PolicyStub(scan_layers=True, param_scan_axis=0))
    m3_remat.apply_remat(model, m3_remat.Remat(policy=cp.nothing_saveable))
    self.assertEqual(model.blocks.layers[0].num_layers, 1)

    single_seg = model.blocks.layers[0]
    graphdef, state = nnx.split(single_seg)
    x = jnp.ones((2, 4, 16), dtype=jnp.float32)

    @jax.jit
    def grad_step(st, inp):
      def loss_fn(s):
        return jnp.sum(nnx.merge(graphdef, s)(inp) ** 2)

      return jax.grad(loss_fn)(st)

    hlo_text = grad_step.lower(state, x).compile().as_text()
    self.assertIn("while(", hlo_text)


class CheckTagsModeGateAndLoopSetupTest(parameterized.TestCase):
  """Tests `checkpoint_names`, fail-closed `check_tags`, non-train mode gating, and `loop.setup_model`."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  @parameterized.parameters("dot_product", "autoselected")
  def test_qwen3_tags_emitted_on_xla_and_splash_attention_and_fail_closed_check_tags(self, attn_kernel: str):
    mesh = create_mesh(devices=jax.devices()[:1])
    cfg = _qwen3_mini_config(
        self.output_dir,
        scan_layers="true",
        remat_policy="minimal_with_context",
        attention=attn_kernel,
        max_target_length="128",
    )
    toks = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    pos = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      names = m3_remat.check_tags(model, cfg, toks, pos)
      self.assertTrue(
          {"query_proj", "key_proj", "value_proj", "context", "out_proj", "mlpwi_0", "mlpwi_1", "mlpwo"}.issubset(names),
          f"Missing expected Qwen3 savepoint tags for attention={attn_kernel}: {names}",
      )

      bad_cfg = _PolicyStub(
          remat_policy="custom",
          tensors_on_device=("nonexistent_savepoint_tag",),
          tensors_to_offload=(),
      )
      with self.assertRaisesRegex(ValueError, "nonexistent_savepoint_tag"):
        m3_remat.check_tags(model, bad_cfg, toks, pos)

      empty_custom = _PolicyStub(remat_policy="custom", tensors_on_device=(), tensors_to_offload=())
      with self.assertRaisesRegex(ValueError, "would save nothing"):
        m3_remat.check_tags(model, empty_custom, toks, pos)

      trainer_custom = dataclasses.make_dataclass(
          "TrainerTagStub",
          ["remat_policy", "names_which_can_be_saved", "names_which_can_be_offloaded"],
      )(
          remat_policy="custom",
          names_which_can_be_saved=["out_proj"],
          names_which_can_be_offloaded=[],
      )
      found_trainer = m3_remat.check_tags(model, trainer_custom, toks, pos)
      self.assertIn("out_proj", found_trainer)

  @parameterized.parameters("auto", "explicit")
  def test_loop_setup_model_wires_remat_and_check_tags_under_auto_and_explicit_meshes(self, shard_mode: str):
    from maxtext.m3.train import loop  # pylint: disable=import-outside-toplevel

    cfg = _qwen3_mini_config(
        self.output_dir,
        scan_layers="true",
        shard_mode=shard_mode,
        remat_policy="minimal_with_context",
        per_device_batch_size="1",
    )
    mesh = loop.mesh_from_config(cfg)
    sharding = DefaultSharding()
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      model, _ = loop.setup_model(cfg, mesh, sharding)
      self.assertIsNotNone(model.decoder.layers.remat)

      bad_cfg = _qwen3_mini_config(
          self.output_dir,
          scan_layers="true",
          shard_mode=shard_mode,
          remat_policy="custom",
          tensors_on_device=["missing_tag_xyz"],
          per_device_batch_size="1",
      )
      with self.assertRaisesRegex(ValueError, "missing_tag_xyz"):
        loop.setup_model(bad_cfg, mesh, sharding)

  @parameterized.parameters(False, True)
  def test_non_train_mode_skips_remat_on_qwen3_and_qwen3_8(self, scan_layers: bool):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    q3_cfg = _qwen3_mini_config(self.output_dir, scan_layers=str(scan_layers).lower(), remat_policy="full")
    toks = jax.device_put(jnp.ones((1, 16), dtype=jnp.int32), rep)
    pos = jax.device_put(jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (1, 16)), rep)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      m_train = model_creation_utils.create_model(q3_cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_eval = model_creation_utils.create_model(q3_cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))

      gd_tr, st_tr = nnx.split(m_train)
      gd_ev, st_ev = nnx.split(m_eval)
      j_tr = jax.make_jaxpr(lambda s, t, p: nnx.merge(gd_tr, s)(t, p))(st_tr, toks, pos).jaxpr
      j_ev = jax.make_jaxpr(lambda s, t, p: nnx.merge(gd_ev, s)(t, p))(st_ev, toks, pos).jaxpr
      self.assertGreater(_count_primitives(j_tr, {"remat2"}), 0)
      self.assertEqual(_count_primitives(j_ev, {"remat2"}), 0)

    q38_cfg = _qwen3_8_mini_config(scan_layers=scan_layers, remat_policy="full")
    toks38 = jax.device_put(jnp.ones((1, 8), dtype=jnp.int32), rep)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      m38_train = model_creation_utils.create_model(q38_cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m38_eval = model_creation_utils.create_model(q38_cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
      self.assertTrue(m38_train.decoder.remat)
      self.assertFalse(m38_eval.decoder.remat)
      gd38_tr, st38_tr = nnx.split(m38_train)
      gd38_ev, st38_ev = nnx.split(m38_eval)
      j38_tr = jax.make_jaxpr(lambda s, t: nnx.merge(gd38_tr, s)(t)[0])(st38_tr, toks38).jaxpr
      j38_ev = jax.make_jaxpr(lambda s, t: nnx.merge(gd38_ev, s)(t)[0])(st38_ev, toks38).jaxpr
      self.assertGreater(_count_primitives(j38_tr, {"remat2"}), 0)
      self.assertEqual(_count_primitives(j38_ev, {"remat2"}), 0)


class HybridCycleAndMoESavepointTest(parameterized.TestCase):
  """Tests Qwen3.8 per-sublayer checkpointing, MoE `moe_routed_output` DCE, and numerical gradient parity."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_qwen3_8_scanned_and_unscanned_per_sublayer_remat_and_tags(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    toks = jax.device_put(jnp.ones((1, 8), dtype=jnp.int32), rep)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      cfg_scan = _qwen3_8_mini_config(
          scan_layers=True,
          base_num_decoder_layers=8,
          remat_policy="custom",
          tensors_on_device=["moe_routed_output"],
      )
      m_scan = model_creation_utils.create_model(cfg_scan, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      found_tags = m3_remat.check_tags(m_scan, cfg_scan, toks)
      self.assertTrue(
          {
              "moe_routed_output",
              "qkv_proj",
              "query_proj",
              "key_proj",
              "value_proj",
              "context",
              "out_proj",
              "mlpwi_0",
              "mlpwi_1",
              "mlpwo",
          }.issubset(found_tags),
          f"Missing Qwen3.8 savepoint tags: {found_tags}",
      )

      gd, st = nnx.split(m_scan.decoder)
      dec_jaxpr = jax.make_jaxpr(lambda s, t: nnx.merge(gd, s)(t)[0])(st, toks).jaxpr
      # Top-level decoder jaxpr has 4 per-sublayer `remat2` calls for `first_cycle` (`prevent_cse=False`)
      # and 1 `scan` call for `cycles` whose body has 4 per-sublayer `remat2` calls (`prevent_cse=True`).
      top_remat = [eqn for eqn in dec_jaxpr.eqns if eqn.primitive.name == "remat2"]
      self.assertLen(top_remat, 4)
      for eqn in top_remat:
        self.assertFalse(eqn.params.get("prevent_cse", True))
      scan_eqns = [eqn for eqn in dec_jaxpr.eqns if eqn.primitive.name == "scan"]
      self.assertLen(scan_eqns, 1)
      scan_body_jaxpr = scan_eqns[0].params["jaxpr"].jaxpr
      inner_remat = [eqn for eqn in scan_body_jaxpr.eqns if eqn.primitive.name == "remat2"]
      # 4 decoder sublayers (`gdn_0`, `gdn_1`, `gdn_2`, `qsa_3`) inside `cycles`.
      self.assertGreaterEqual(len(inner_remat), 4)
      for eqn in inner_remat:
        self.assertTrue(eqn.params.get("prevent_cse", False))

      # Single-stacked-cycle (`num_stacked_cycles == 1`, `base_num_decoder_layers == 8`) HLO preserves `while(`.
      toks_arr = jax.device_put(jnp.ones((1, 8), dtype=jnp.int32), rep)
      hlo_1cyc = jax.jit(lambda s, t: nnx.merge(gd, s)(t)[0]).lower(st, toks_arr).compile().as_text()
      self.assertIn("while(", hlo_1cyc)

      # Unscanned Qwen3.8 must also wrap each sublayer in `remat2`.
      cfg_unscan = _qwen3_8_mini_config(scan_layers=False, remat_policy="full")
      m_unscan = model_creation_utils.create_model(cfg_unscan, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      gd_u, st_u = nnx.split(m_unscan.decoder)
      unscan_jaxpr = jax.make_jaxpr(lambda s, t: nnx.merge(gd_u, s)(t)[0])(st_u, toks).jaxpr
      top_unscan_remat = [eqn for eqn in unscan_jaxpr.eqns if eqn.primitive.name == "remat2"]
      self.assertGreaterEqual(len(top_unscan_remat), 4)

  @parameterized.parameters("dense", "ragged_dot", "tokamax")
  def test_moe_routed_output_savepoint_skips_wo_recompute_single_device(self, kind: str):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    key = jax.random.key(19)
    kx, kw, k0, k1, k2, kc = jax.random.split(key, 6)
    x = jax.device_put(jax.random.normal(kx, (2, 8, 16), dtype=jnp.float32), rep)
    logits = jax.random.normal(kw, (2, 8, 4), dtype=jnp.float32)
    vals, idx = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), 2)
    weights = jax.device_put(vals / jnp.sum(vals, axis=-1, keepdims=True), rep)
    idx = jax.device_put(idx, rep)
    wi_0 = jax.device_put(jax.random.normal(k0, (4, 16, 12), dtype=jnp.float32) * 0.1, rep)
    wi_1 = jax.device_put(jax.random.normal(k1, (4, 16, 12), dtype=jnp.float32) * 0.1, rep)
    wo = jax.device_put(jax.random.normal(k2, (4, 12, 16), dtype=jnp.float32) * 0.1, rep)
    w_gu = jax.device_put(jnp.concatenate([wi_0, wi_1], axis=-1), rep)
    cot = jax.device_put(jax.random.normal(kc, (2, 8, 16), dtype=jnp.float32), rep)

    executor = m3_moe.MoEExecutor(kind=kind)  # type: ignore[arg-type]
    save_moe_pol = m3_remat.remat_policy("custom", names_which_can_be_saved=["moe_routed_output"])
    dot_prims = {"dot_general", "ragged_dot", "ragged_dot_general", "gmm"}

    with jax.set_mesh(mesh):
      for use_fused in (False, True):

        def make_loss(pol, fused=use_fused):
          def block(x_in, w0, w1, w_out, gu):
            h = jnp.tanh(x_in)
            if fused:
              routed = executor.fused(h, weights, idx, gu, w_out)
            else:
              routed = executor(h, weights, idx, w0, w1, w_out)
            # Post-MoE RMSNorm (matching Gemma4 `PR #6099` / `mHC` post-op) requires `routed` in backward.
            routed_normed = routed * jax.lax.rsqrt(jnp.mean(routed * routed, axis=-1, keepdims=True) + 1e-6)
            return h + routed_normed

          ckpt_block = jax.checkpoint(block, policy=pol, prevent_cse=True)
          return lambda x_in, w0, w1, w_out, gu: jnp.sum(ckpt_block(x_in, w0, w1, w_out, gu) * cot)

        grad_full_fn = jax.grad(make_loss(cp.nothing_saveable), argnums=(0, 1, 2, 3, 4))
        grad_saved_fn = jax.grad(make_loss(save_moe_pol), argnums=(0, 1, 2, 3, 4))

        j_full = jax.make_jaxpr(grad_full_fn)(x, wi_0, wi_1, wo, w_gu).jaxpr
        j_saved = jax.make_jaxpr(grad_saved_fn)(x, wi_0, wi_1, wo, w_gu).jaxpr
        self.assertLess(
            _count_primitives(j_saved, dot_prims),
            _count_primitives(j_full, dot_prims),
            f"Saving moe_routed_output must eliminate recomputed expert matmuls (kind={kind}, fused={use_fused})",
        )

        grads_full = jax.jit(grad_full_fn)(x, wi_0, wi_1, wo, w_gu)
        grads_saved = jax.jit(grad_saved_fn)(x, wi_0, wi_1, wo, w_gu)
        for gf, gs in zip(grads_full, grads_saved):
          np.testing.assert_allclose(np.asarray(gs), np.asarray(gf), atol=1e-5, rtol=1e-5)

  def test_moe_routed_output_savepoint_skips_ep_recompute_on_multi_device_mesh(self):
    if jax.device_count() < 4:
      self.skipTest("Requires >= 4 CPU devices for EP=2, FSDP=2.")
    mesh = create_mesh(ici={"fsdp": 2, "expert": 2, "tensor": 1}, devices=jax.devices()[:4])
    sharding = DefaultSharding()
    executor = m3_moe.MoEExecutor(
        kind="tokamax",
        batch=("data", "fsdp", "expert"),
        experts="expert",
        mlp="tensor",
    )
    key = jax.random.key(29)
    kx, kw, k0, k1, k2, kc = jax.random.split(key, 6)
    x = jax.random.normal(kx, (4, 8, 16), dtype=jnp.float32)
    logits = jax.random.normal(kw, (4, 8, 4), dtype=jnp.float32)
    vals, idx = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), 2)
    weights = vals / jnp.sum(vals, axis=-1, keepdims=True)
    wi_0 = jax.random.normal(k0, (4, 16, 16), dtype=jnp.float32) * 0.1
    wi_1 = jax.random.normal(k1, (4, 16, 16), dtype=jnp.float32) * 0.1
    wo = jax.random.normal(k2, (4, 16, 16), dtype=jnp.float32) * 0.1
    cot = jax.random.normal(kc, (4, 8, 16), dtype=jnp.float32)
    save_moe_pol = m3_remat.remat_policy("custom", names_which_can_be_saved=["moe_routed_output"])
    dot_prims = {"dot_general", "ragged_dot", "ragged_dot_general", "gmm"}

    with jax.set_mesh(mesh):
      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      rk_s = NamedSharding(mesh, P(("data", "fsdp", "expert"), None, None))
      w0_s = NamedSharding(mesh, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh, sharding("wo", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, rk_s)
      idx_d = jax.device_put(idx, rk_s)
      cot_d = jax.device_put(cot, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)

      def make_ep_grad(pol):
        def block(x_in, w0, w1, w_out):
          h = jnp.tanh(x_in)
          routed = executor(h, w_d, idx_d, w0, w1, w_out)
          routed_normed = routed * jax.lax.rsqrt(jnp.mean(routed * routed, axis=-1, keepdims=True) + 1e-6)
          return h + routed_normed

        ckpt = jax.checkpoint(block, policy=pol, prevent_cse=True)
        return jax.grad(lambda x_in, w0, w1, w_out: jnp.sum(ckpt(x_in, w0, w1, w_out) * cot_d), argnums=(0, 1, 2, 3))

      g_full_fn = make_ep_grad(cp.nothing_saveable)
      g_save_fn = make_ep_grad(save_moe_pol)
      j_full = jax.make_jaxpr(g_full_fn)(x_d, wi0_d, wi1_d, wo_d).jaxpr
      j_save = jax.make_jaxpr(g_save_fn)(x_d, wi0_d, wi1_d, wo_d).jaxpr
      self.assertLess(_count_primitives(j_save, dot_prims), _count_primitives(j_full, dot_prims))

      g_full = jax.jit(g_full_fn)(x_d, wi0_d, wi1_d, wo_d)
      g_save = jax.jit(g_save_fn)(x_d, wi0_d, wi1_d, wo_d)
      for gf, gs in zip(g_full, g_save):
        np.testing.assert_allclose(np.asarray(gs), np.asarray(gf), atol=1e-5, rtol=1e-5)

  @parameterized.parameters(False, True)
  def test_qwen3_numerical_grad_parity_and_negative_control(self, scan_layers: bool):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    toks = jax.device_put(jnp.array([[3, 11, 25, 41, 7, 19, 88, 12, 5, 33, 60, 72, 9, 14, 27, 45]], dtype=jnp.int32), rep)
    targets = jax.device_put(jnp.roll(toks, -1, axis=-1), rep)
    pos = jax.device_put(jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (1, 16)), rep)

    @nnx.jit
    def loss_and_grad(model, t_in, t_tgt, p_in):
      def loss_fn(m):
        logits = m(t_in, p_in)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.mean(jnp.take_along_axis(log_probs, t_tgt[..., None], axis=-1))

      return nnx.value_and_grad(loss_fn)(model)

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True), jax.default_matmul_precision("highest"):
      cfg_none = _qwen3_mini_config(self.output_dir, scan_layers=str(scan_layers).lower(), remat_policy="none")
      m_none = model_creation_utils.create_model(cfg_none, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      ref_state = nnx.state(m_none)
      ref_loss, ref_grads = loss_and_grad(m_none, toks, targets, pos)

      for pol in ("full", "minimal_with_context", "save_qkv_proj"):
        cfg_p = _qwen3_mini_config(self.output_dir, scan_layers=str(scan_layers).lower(), remat_policy=pol)
        m_p = model_creation_utils.create_model(cfg_p, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
        nnx.update(m_p, ref_state)
        got_loss, got_grads = loss_and_grad(m_p, toks, targets, pos)
        np.testing.assert_allclose(got_loss, ref_loss, atol=1e-5, rtol=1e-5)
        _assert_grad_trees_close(got_grads, ref_grads, atol=1e-5, rtol=1e-5)

      # Weight perturbation negative control: perturbing a parameter must break gradient parity.
      m_pert = model_creation_utils.create_model(
          _qwen3_mini_config(self.output_dir, scan_layers=str(scan_layers).lower(), remat_policy="full"),
          mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(0),
      )
      nnx.update(m_pert, ref_state)
      m_pert.token_embedder.embedding[...] = m_pert.token_embedder.embedding[...] + 0.05
      _, pert_grads = loss_and_grad(m_pert, toks, targets, pos)
      with self.assertRaises(AssertionError):
        _assert_grad_trees_close(pert_grads, ref_grads, atol=1e-5, rtol=1e-5)

  @parameterized.parameters(False, True)
  def test_qwen3_8_numerical_grad_parity_with_mtp_and_negative_control(self, scan_layers: bool):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    toks = jax.device_put(jnp.array([[3, 17, 42, 88, 3, 50, 61, 99]], dtype=jnp.int32), rep)
    targets = jax.device_put(jnp.array([[17, 42, 88, 3, 50, 61, 99, 5]], dtype=jnp.int32), rep)
    seg = jax.device_put(jnp.ones((1, 8), dtype=jnp.int32).at[:, 4:].set(2), rep)

    @nnx.jit
    def loss_and_grad(model, t_in, t_tgt, s_in):
      def loss_fn(m):
        loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
            m, t_in, t_tgt, segment_ids=s_in, router_aux_coeff=0.01, use_mtp=True
        )
        return loss

      return nnx.value_and_grad(loss_fn)(model)

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True), jax.default_matmul_precision("highest"):
      cfg_none = _qwen3_8_mini_config(scan_layers=scan_layers, remat_policy="none")
      m_none = model_creation_utils.create_model(cfg_none, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      ref_state = nnx.state(m_none)
      ref_loss, ref_grads = loss_and_grad(m_none, toks, targets, seg)

      for pol, saved in (("full", []), ("minimal_with_context", []), ("custom", ["moe_routed_output"])):
        cfg_p = _qwen3_8_mini_config(scan_layers=scan_layers, remat_policy=pol, tensors_on_device=saved)
        m_p = model_creation_utils.create_model(cfg_p, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
        nnx.update(m_p, ref_state)
        got_loss, got_grads = loss_and_grad(m_p, toks, targets, seg)
        np.testing.assert_allclose(got_loss, ref_loss, atol=1e-5, rtol=1e-5)
        _assert_grad_trees_close(got_grads, ref_grads, atol=1e-5, rtol=1e-5)

      # Negative control: scaling `lm_head` weights breaks softmax shift-invariance and must fail parity.
      m_pert = model_creation_utils.create_model(
          _qwen3_8_mini_config(scan_layers=scan_layers, remat_policy="custom", tensors_on_device=["moe_routed_output"]),
          mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(0),
      )
      nnx.update(m_pert, ref_state)
      m_pert.lm_head.kernel[...] = m_pert.lm_head.kernel[...] * 2.0
      _, pert_grads = loss_and_grad(m_pert, toks, targets, seg)
      with self.assertRaises(AssertionError):
        _assert_grad_trees_close(pert_grads, ref_grads, atol=1e-5, rtol=1e-5)

  def test_qwen3_8_scanned_cycles_do_not_stack_ple_weights_in_backward_residuals(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    toks = jnp.ones((1, 8), dtype=jnp.int32)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      cfg = _qwen3_8_mini_config(
          scan_layers=True,
          remat_policy="full",
          base_num_decoder_layers=12,
          dtype="bfloat16",
          weight_dtype="bfloat16",
      )
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      ple_shape = model.decoder.first_cycle.gdn_1.ple.ple_embedding.ngram_embedding.embedding.shape
      num_cycles = model.decoder.num_cycles
      stacked_sig = f"{num_cycles}x{ple_shape[0]}x{ple_shape[1]}"
      stacked_rem_sig = f"{model.decoder.num_stacked_cycles}x{ple_shape[0]}x{ple_shape[1]}"
      gdef, params, rest = nnx.split(model, nnx.Param, ...)

      for pol_name in ("full", "minimal_with_context", "custom"):
        pol = m3_remat.remat_policy(pol_name, names_which_can_be_saved=["moe_routed_output"])

        def loss_fn(p, r_pol=pol):
          m = nnx.merge(gdef, p, rest)
          m.decoder.set_remat(m3_remat.Remat(policy=r_pol))
          out = m(toks)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(logits.astype(jnp.float32) ** 2)

        mlir = jax.jit(jax.grad(loss_fn)).lower(params).as_text()
        self.assertNotIn(stacked_sig, mlir, f"PLE weights stacked across cycles under {pol_name}: {stacked_sig}")
        self.assertNotIn(stacked_rem_sig, mlir, f"PLE weights stacked across cycles under {pol_name}: {stacked_rem_sig}")

  @parameterized.parameters(False, True)
  def test_glm5_3_set_remat_tags_eval_gate_and_grad_parity(self, scan_layers: bool):
    mesh = create_mesh(devices=jax.devices()[:1])
    rep = NamedSharding(mesh, P())
    toks = jax.device_put(jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32), rep)
    targets = jax.device_put(jnp.roll(toks, -1, axis=-1), rep)
    mask = jax.device_put(jnp.ones_like(targets, dtype=jnp.float32), rep)

    @nnx.jit
    def loss_and_grad(model, t_in, t_tgt, m_in):
      def loss_fn(m):
        loss, _ = modeling_glm5_3_flash.compute_glm5_loss(m, t_in, t_tgt, m_in, is_train=True)
        return loss

      return nnx.value_and_grad(loss_fn)(model)

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True), jax.default_matmul_precision("highest"):
      cfg_none = _make_glm5_mini_cfg(scan_layers=scan_layers, remat_policy="none")
      m_none = model_creation_utils.create_model(cfg_none, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      ref_state = nnx.state(m_none)
      ref_loss, ref_grads = loss_and_grad(m_none, toks, targets, mask)

      for pol, saved in (
          ("full", ()),
          ("minimal", ()),
          ("save_dot_except_mlp", ()),
          ("custom", ("moe_routed_output",)),
      ):
        cfg_p = _make_glm5_mini_cfg(
            scan_layers=scan_layers,
            remat_policy=pol,
            tensors_on_device=saved,
        )
        m_p = model_creation_utils.create_model(cfg_p, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
        nnx.update(m_p, ref_state)
        self.assertIsNotNone(m_p.remat_policy)
        tags = m3_remat.check_tags(m_p, cfg_p, toks)
        if pol == "full":
          self.assertEqual(tags, frozenset())
        else:
          self.assertTrue(
              {
                  "moe_routed_output",
                  "query_proj",
                  "kv_proj",
                  "context",
                  "out_proj",
                  "mlpwi_0",
                  "mlpwi_1",
                  "mlpwo",
              }.issubset(tags),
              f"Missing GLM-5.3 savepoint tags: {tags}",
          )
        got_loss, got_grads = loss_and_grad(m_p, toks, targets, mask)
        np.testing.assert_allclose(got_loss, ref_loss, atol=1e-5, rtol=1e-5)
        _assert_grad_trees_close(got_grads, ref_grads, atol=1e-5, rtol=1e-5)

      # Non-train mode clears `Glm5Model.remat_policy`.
      cfg_custom = _make_glm5_mini_cfg(
          scan_layers=scan_layers,
          remat_policy="custom",
          tensors_on_device=("moe_routed_output",),
      )
      m_eval = model_creation_utils.create_model(cfg_custom, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
      self.assertIsNone(m_eval.remat_policy)

      # Negative control: scaling `lm_head` weights breaks softmax shift-invariance and must fail parity.
      m_pert = model_creation_utils.create_model(cfg_custom, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      nnx.update(m_pert, ref_state)
      m_pert.lm_head.kernel[...] = m_pert.lm_head.kernel[...] * 2.0
      _, pert_grads = loss_and_grad(m_pert, toks, targets, mask)
      with self.assertRaises(AssertionError):
        _assert_grad_trees_close(pert_grads, ref_grads, atol=1e-5, rtol=1e-5)

  def test_qwen3_8_unscanned_sublayer_preserves_non_param_state(self):
    """Verifies unscanned Qwen3_8Decoder preserves non-Param updates across _run_qwen3_8_sublayer."""
    # pylint: disable=protected-access
    mesh = create_mesh(devices=jax.devices()[:1])
    cfg = _qwen3_8_mini_config(scan_layers=False, remat_policy="full")
    with jax.set_mesh(mesh):
      decoder = modeling_qwen3_8_flash_next.Qwen3_8Decoder(cfg, rngs=nnx.Rngs(0))
      layer0 = decoder.first_cycle.gdn_0

      class _CustomIntermediate(nnx.Intermediate):
        """Probe Intermediate variable."""

      class _InstrumentedSublayer(nnx.Module):
        """Sublayer wrapper mutating a non-Param probe."""

        def __init__(self, inner):
          self.inner = inner
          self.probe = _CustomIntermediate(jnp.array(0.0))

        def __call__(self, h, **kwargs):
          self.probe = _CustomIntermediate(jnp.array(42.0))
          return self.inner(h, **kwargs)

      inst = _InstrumentedSublayer(layer0)
      h = jnp.ones((1, 8, cfg.emb_dim * int(cfg.hc_count)), dtype=jnp.float32)
      _ = modeling_qwen3_8_flash_next._run_qwen3_8_sublayer(
          inst,
          h,
          eff_policy=decoder.remat_policy,
          return_aux=False,
      )
      self.assertEqual(float(inst.probe.get_value()), 42.0)

  def test_qwen3_8_unscanned_sublayer_uses_prevent_cse_false_and_matches_no_remat(self):
    """Verifies unscanned _run_qwen3_8_sublayer uses prevent_cse=False (scanned keeps True) and <1e-5 grad drift."""
    # pylint: disable=protected-access
    mesh = create_mesh(devices=jax.devices()[:1])
    cfg_none = _qwen3_8_mini_config(scan_layers=False, base_num_decoder_layers=8, remat_policy="none")
    cfg_full = _qwen3_8_mini_config(scan_layers=False, base_num_decoder_layers=8, remat_policy="full")
    cfg_scan = _qwen3_8_mini_config(scan_layers=True, base_num_decoder_layers=8, remat_policy="full")
    with jax.set_mesh(mesh):
      m_none = model_creation_utils.create_model(cfg_none, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_full = model_creation_utils.create_model(cfg_full, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_scan = model_creation_utils.create_model(cfg_scan, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      nnx.update(m_full, nnx.state(m_none, nnx.Param))
      toks = jax.random.randint(jax.random.PRNGKey(3), (1, 16), 0, cfg_none.vocab_size)
      jaxpr_unscanned = str(jax.make_jaxpr(m_full)(toks))
      jaxpr_scanned = str(jax.make_jaxpr(m_scan)(toks))
      self.assertIn("prevent_cse=False", jaxpr_unscanned)
      self.assertNotIn("prevent_cse=True", jaxpr_unscanned)
      self.assertIn("prevent_cse=True", jaxpr_scanned)

      def _loss(m, x):
        logits = m(x)
        return jnp.mean(logits.astype(jnp.float32) ** 2)

      l_none, g_none = nnx.jit(nnx.value_and_grad(_loss))(m_none, toks)
      l_full, g_full = nnx.jit(nnx.value_and_grad(_loss))(m_full, toks)
      self.assertLess(abs(float(l_full - l_none)), 1e-5)
      _assert_grad_trees_close(nnx.state(g_full, nnx.Param), nnx.state(g_none, nnx.Param), atol=1e-5, rtol=1e-5)

  def test_create_model_scan_remat_quantize_with_qwix(self):
    """Verifies model_creation_utils.create_model applies scan -> remat -> quantize cleanly with Qwix quantization."""
    mesh = create_mesh(devices=jax.devices()[:1])
    for scan in (True, False):
      cfg = _qwen3_8_mini_config(
          scan_layers=scan,
          remat_policy="full",
          quantization="int8",
          use_qwix_quantization="true",
      )
      with jax.set_mesh(mesh):
        model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
        self.assertTrue(hasattr(type(model), "_unquantized_type"))
        self.assertTrue(model.decoder.remat)
        self.assertIsNotNone(model.decoder.remat_policy)


if __name__ == "__main__":
  absltest.main()
