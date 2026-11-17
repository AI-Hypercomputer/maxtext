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

"""Unit and compiled-HLO tests for m3 FSDP weight prefetching (`m3/infra/fsdp_prefetch.py`)."""

import gc
import re
from types import SimpleNamespace

from absl.testing import absltest, parameterized
from flax import nnx
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import fsdp_prefetch, remat, scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.train import loop as m3_loop
from maxtext.utils import model_creation_utils


def _tiny_qwen3_config(
    *,
    num_layers=6,
    scan_layers=True,
    param_scan_axis=1,
    remat_policy="full",
    prefetch_fsdp_weights=False,
    prefetch_fsdp_defer_grad_reduce_scatter=False,
    shard_mode="explicit",
    decoder_layer_input="device",
):
  """Minimal Qwen3 config exercised by CPU unit tests."""
  return SimpleNamespace(
      use_m3_model=True,
      model_name="qwen3-0.6b",
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
      remat_policy=remat_policy,
      prefetch_fsdp_weights=prefetch_fsdp_weights,
      prefetch_fsdp_defer_grad_reduce_scatter=prefetch_fsdp_defer_grad_reduce_scatter,
      shard_mode=shard_mode,
      attention="dot_product",
      quantization="",
      parameter_memory_host_offload=False,
      decoder_layer_input=decoder_layer_input,
      num_decoder_layers=num_layers,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
      max_target_length=4,
      global_batch_size_to_train_on=4,
      micro_batch_size_to_train_on=4,
      per_device_batch_size=1,
      init_weights_seed=0,
      learning_rate=1e-3,
      gradient_clipping_threshold=0.0,
      opt_type="adamw",
      adam_b1=0.9,
      adam_b2=0.999,
      adam_eps=1e-8,
      adam_eps_root=0.0,
      adam_weight_decay=0.0,
      adamw_mask=[],
      trainable_parameters_mask=[],
      skip_step_on_spikes=False,
      mu_dtype=None,
      use_iota_embed=False,
      logical_axis_rules=(),
  )


def _flat_params(mod):
  return {
      tuple(k.key for k in path): np.asarray(v[...])
      for path, v in jax.tree_util.tree_flatten_with_path(
          nnx.state(mod, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)
      )[0]
  }


def _make_batch(mesh, batch_axes="fsdp", seq_len=4):
  """Creates deterministic sharded token and position batches."""
  spec = P(batch_axes, None) if batch_axes is not None else P(None, None)
  sharding = NamedSharding(mesh, spec)
  tokens = jax.device_put(
      jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8], [8, 7, 6, 5]], dtype=jnp.int32)[:, :seq_len],
      sharding,
  )
  positions = jax.device_put(
      jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (4, seq_len)),
      sharding,
  )
  return tokens, positions


def _build_qwen3(cfg, mesh):
  model = create_qwen3_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
  model = scan.maybe_apply_scan(model, cfg)
  model = remat.maybe_apply_remat(model, cfg)
  model = fsdp_prefetch.maybe_apply_fsdp_prefetch(model, cfg, mesh=mesh)
  return model


def _loss_and_grad_fn(explicit: bool, batch_axes="fsdp"):
  """Returns a `(loss, grads)` callable compatible with explicit and auto meshes."""
  out_spec = P(batch_axes, None, None)

  def _run(model, tokens, positions):
    def loss_fn(m):
      logits = m(tokens, positions)
      if explicit:
        log_probs = jax.sharding.auto_axes(jax.nn.log_softmax, out_sharding=out_spec)(logits)
      else:
        log_probs = jax.nn.log_softmax(logits)
      return -jnp.mean(log_probs[..., 0])

    return nnx.value_and_grad(loss_fn)(model)

  return _run


def _parse_hlo_computations(hlo_text: str) -> dict[str, list[str]]:
  """Splits compiled XLA HLO text into `{computation_name: [instruction_lines]}`."""
  comps: dict[str, list[str]] = {}
  cur_name = None
  for line in hlo_text.splitlines():
    m = re.match(r"^(?:ENTRY\s+)?(%[\w.\-]+)\s*\(", line)
    if m and line.rstrip().endswith("{"):
      cur_name = m.group(1)
      comps[cur_name] = []
    elif line.strip() == "}" and cur_name is not None:
      cur_name = None
    elif cur_name is not None and "=" in line:
      comps[cur_name].append(line.strip())
  return comps


_HLO_DEBUG_SECTIONS = frozenset({"FileNames", "FunctionNames", "FileLocations", "StackFrames"})
_HLO_COMP_DEF_RE = re.compile(r"^(ENTRY\s+)?(%[\w.\-]+)\s*\((.*)\)\s*->\s*(.*\{\s*)$")
_HLO_PARAM_DECL_RE = re.compile(r"(?:^|(?<=,\s)|(?<=\*/))([A-Za-z_][\w.\-]*)(\s*:)")
_HLO_STR_OR_IDENT_RE = re.compile(r'"[^"]*"|%([A-Za-z_][\w.\-]*)')


def _hlo_comp_stem(name: str) -> str:
  stem = re.sub(r"\.\d+(?=\.|_spmd\b|$)", "", name)
  return re.sub(r"\bregion_\d+", "region", stem)


def _hlo_ssa_stem(name: str) -> str:
  return re.sub(r"\.\d+(?=\.|$)", "", name)


def _normalize_hlo(hlo_text: str) -> str:
  """Canonicalizes debug tables and monotonic XLA SSA/subcomputation counters while preserving graph wiring."""
  lines = hlo_text.splitlines()
  comp_map: dict[str, str] = {}
  comp_counts: dict[str, int] = {}
  for line in lines:
    m = _HLO_COMP_DEF_RE.match(line.strip())
    if m:
      raw_comp = m.group(2)[1:]
      if raw_comp not in comp_map:
        stem = _hlo_comp_stem(raw_comp)
        idx = comp_counts.get(stem, 0)
        comp_counts[stem] = idx + 1
        comp_map[raw_comp] = f"{stem}#{idx}"

  out_lines: list[str] = []
  in_debug_section = False
  in_comp = False
  local_map: dict[str, str] = {}
  local_counts: dict[str, int] = {}

  def _map_local(raw_name: str) -> str:
    if raw_name in comp_map:
      return comp_map[raw_name]
    if raw_name not in local_map:
      stem = _hlo_ssa_stem(raw_name)
      idx = local_counts.get(stem, 0)
      local_counts[stem] = idx + 1
      local_map[raw_name] = f"{stem}#{idx}"
    return local_map[raw_name]

  def _replace_param(pm: re.Match[str]) -> str:
    return f"{_map_local(pm.group(1))}{pm.group(2)}"

  def _replace_ident(im: re.Match[str]) -> str:
    return f"%{_map_local(im.group(1))}" if im.group(1) is not None else im.group(0)

  for line in lines:
    stripped = line.strip()
    if not in_comp:
      if stripped in _HLO_DEBUG_SECTIONS:
        in_debug_section = True
        continue
      if in_debug_section:
        if not stripped or re.match(r"^\d+\s+", stripped):
          continue
        in_debug_section = False
      if stripped.startswith("HloModule "):
        line = re.sub(r"^(\s*HloModule\s+)\S+?,", r"\1<module>,", line, count=1)
        out_lines.append(line)
        continue

    m = _HLO_COMP_DEF_RE.match(stripped)
    if m:
      in_comp = True
      local_map.clear()
      local_counts.clear()
      entry_prefix = m.group(1) or ""
      comp_canon = comp_map[m.group(2)[1:]]
      params_str = _HLO_PARAM_DECL_RE.sub(_replace_param, m.group(3))
      out_lines.append(f"{entry_prefix}%{comp_canon} ({params_str}) -> {m.group(4).rstrip()}")
      continue

    if in_comp and (stripped == "}" or stripped.startswith("},")):
      in_comp = False
      out_lines.append(line)
      continue

    if in_comp:
      line = re.sub(r"\s*\bstack_frame_id=\d+", "", line)
      line = re.sub(r",?\s*metadata=\{\s*\}", "", line)
      line = _HLO_STR_OR_IDENT_RE.sub(_replace_ident, line)
    if stripped or (out_lines and out_lines[-1] != ""):
      out_lines.append(line)

  return "\n".join(out_lines)


class _TupleCarryStatefulLayer(nnx.Module):
  """Toy layer with tuple carry `(hidden, aux_sum)` and mutable non-Param state."""

  def __init__(self, dim: int, mesh, *, rngs: nnx.Rngs):
    spec = P("fsdp", None)
    w = jax.random.normal(rngs.params(), (dim, dim), dtype=jnp.float32) * 0.2
    w = jax.device_put(w, NamedSharding(mesh, spec))
    self.weight = nnx.Param(w, out_sharding=spec, eager_sharding=True)
    self.calls = nnx.Variable(jnp.array(0, dtype=jnp.int32), out_sharding=P())

  def __call__(self, carry, *, scale: float = 1.0):
    x, aux = carry
    self.calls[...] = self.calls[...] + 1
    y = jnp.tanh(jnp.einsum("bd,df->bf", x, self.weight[...])) * scale
    return (x + y, aux + jnp.mean(y**2))


class _DynamicKwargLayer(nnx.Module):
  """Toy scanned layer consuming broadcast, sharded, and tuple floating-point `jax.Array` kwargs."""

  def __init__(self, dim: int, mesh, explicit: bool, *, rngs: nnx.Rngs):
    spec = P("fsdp", None)
    w = jax.random.normal(rngs.params(), (dim, dim), dtype=jnp.float32) * 0.25
    w = jax.device_put(w, NamedSharding(mesh, spec))
    self.weight = nnx.Param(w, out_sharding=spec, eager_sharding=True)
    self._explicit = explicit

  def __call__(
      self,
      x,
      *,
      per_layer_bias=None,
      cond_state=None,
      rope_pair=None,
      positions=None,
      scale: float = 1.0,
  ):
    if self._explicit:
      h = jnp.einsum("bsd,df->bsf", x, self.weight[...], out_sharding=P("fsdp", None, None))
    else:
      h = jnp.einsum("bsd,df->bsf", x, self.weight[...])
    if per_layer_bias is not None:
      h = h + per_layer_bias
    if cond_state is not None:
      h = h + jnp.tanh(cond_state) * 0.5
    if rope_pair is not None:
      cos, sin = rope_pair
      h = h * cos + jnp.roll(h, 1, axis=-1) * sin
    h = ad_checkpoint.checkpoint_name(h, "context")
    if positions is not None:
      h = h * (1.0 + 0.05 * positions[..., None].astype(x.dtype))
    return x + jnp.tanh(h) * scale


class FsdpPrefetchHelpersTest(absltest.TestCase):
  """Tests for pure-Python FSDP axis and SparseCore helpers."""

  def test_fsdp_axes_and_spec_removal(self):
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim(("fsdp", "expert")), ("fsdp", "expert"))
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim(("expert", "fsdp")), ("fsdp",))
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim("expert"), ())
    self.assertEqual(
        fsdp_prefetch.remove_fsdp_from_spec(P(("fsdp", "fsdp_transpose", "context", "expert"), "tensor")),
        P(None, "tensor"),
    )
    self.assertEqual(
        fsdp_prefetch.remove_fsdp_from_spec(P("expert", ("fsdp", "context"), "tensor")),
        P("expert", None, "tensor"),
    )

  def test_has_multi_trip_loop_threshold(self):
    for n in range(1, 6):
      self.assertFalse(fsdp_prefetch._has_multi_trip_loop(n))  # pylint: disable=protected-access
    for n in (6, 8, 28):
      self.assertTrue(fsdp_prefetch._has_multi_trip_loop(n))  # pylint: disable=protected-access

  def test_merge_offload_residual_layout_fails_closed_on_count_mismatch(self):
    layout_cls = fsdp_prefetch._ResidualLayout  # pylint: disable=protected-access
    merge_fn = fsdp_prefetch._merge_offload_residual_layout  # pylint: disable=protected-access
    raw = layout_cls(treedef=None, src=(None, None), saved_avals=("a", "b"), saved_offload_mask=(True, False))
    dev_ok = layout_cls(treedef=None, src=(None, None), saved_avals=("a_dev", "b_dev"), saved_offload_mask=(False, False))
    merged = merge_fn(raw, dev_ok)
    self.assertEqual(merged.saved_avals, ("a_dev", "b_dev"))
    self.assertEqual(merged.saved_offload_mask, (True, False))
    dev_bad = layout_cls(treedef=None, src=(None,), saved_avals=("a_dev",), saved_offload_mask=(False,))
    with self.assertRaisesRegex(ValueError, "saved residual leaf count"):
      merge_fn(raw, dev_bad)

  def test_default_sparse_core_id_targets_tpu7x_only(self):
    fake_tpu7x = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="TPU7x")]))
    fake_v5p = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="TPU v5")]))
    fake_cpu = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="cpu")]))
    self.assertEqual(fsdp_prefetch._default_sparse_core_id(fake_tpu7x), 1)  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_v5p))  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_cpu))  # pylint: disable=protected-access

  def test_pinned_3d_non_major_reduce_scatter_moves_scatter_dim_to_axis0(self):
    mesh = jax.sharding.AbstractMesh(
        (4, 2),
        ("fsdp", "expert"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    g_dim2 = jax.ShapeDtypeStruct(
        (8, 128, 512),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, None, None, unreduced=frozenset({"fsdp", "expert"}))),
    )
    g_dim1 = jax.ShapeDtypeStruct(
        (8, 256, 256),
        jnp.float32,
        sharding=NamedSharding(mesh, P("expert", None, None, unreduced=frozenset({"fsdp"}))),
    )

    def scatter_dim2(x):
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          x, mesh, P(None, None, ("fsdp", "expert")), P(None, None, None), sparse_core_id=1
      )

    def scatter_dim1(x):
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          x, mesh, P("expert", "fsdp", None), P("expert", None, None), sparse_core_id=1
      )

    with jax.sharding.use_abstract_mesh(mesh):
      hlo_dim2 = jax.jit(scatter_dim2).trace(g_dim2).lower(lowering_platforms=("tpu",)).as_text()
      hlo_dim1 = jax.jit(scatter_dim1).trace(g_dim1).lower(lowering_platforms=("tpu",)).as_text()

    for hlo in (hlo_dim2, hlo_dim1):
      self.assertIn('_xla_compute_type = "sparseoffload"', hlo)
      self.assertRegex(hlo, r"core_ids.{0,8}\[1\]")
      self.assertIn("dimension = 0", hlo)
      self.assertNotIn("dimension = 1", hlo)
      self.assertNotIn("dimension = 2", hlo)

  def test_normalize_hlo_canonicalizes_counters_and_detects_real_diffs(self):
    hlo_a = """HloModule jit__step, is_scheduled=true, num_partitions=4

FileNames
1 "/a/model.py"

FunctionNames
1 "fn_a"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10 end_line=10 column=1 end_column=1}

StackFrames
1 {file_location_id=1 parent_frame_id=0}

%region_1.8_spmd.clone (Arg_0.6: f32[], Arg_1.7: f32[]) -> f32[] {
  %Arg_0.6 = f32[] parameter(0)
  %Arg_1.7 = f32[] parameter(1)
  ROOT %add.8 = f32[] add(%Arg_0.6, %Arg_1.7), metadata={op_name="jit(_step)/add" stack_frame_id=1}
}

%bitcast_copy_fusion_computation (param_0: f32[4]) -> f32[4] {
  %param_0 = f32[4]{0} parameter(0)
  ROOT %bitcast = f32[4]{0} bitcast(%param_0), metadata={stack_frame_id=1}
}

ENTRY %main.37_spmd (Arg_0.1: f32[4], /*index=5*/Arg_5.2: f32[4]) -> f32[4] {
  %Arg_0.1 = f32[4]{0} parameter(0)
  %Arg_5.2 = f32[4]{0} parameter(1)
  %ws = f32[4]{0} fusion(%Arg_0.1), kind=kLoop, calls=%bitcast_copy_fusion_computation
  ROOT %ar = f32[4]{0} all-reduce(%ws), to_apply=%region_1.8_spmd.clone, metadata={op_name="ar" stack_frame_id=1}
}
"""
    hlo_b = """HloModule jit__step.42, is_scheduled=true, num_partitions=4

FileNames
1 "/x/other.py"
2 "/y/extra.py"

FunctionNames
1 "other_fn"

FileLocations
1 {file_name_id=2 function_name_id=1 line=99 end_line=99 column=4 end_column=4}

StackFrames
1 {file_location_id=1 parent_frame_id=0}
2 {file_location_id=1 parent_frame_id=1}

%region_18.408_spmd.clone (Arg_0.906: f32[], Arg_1.907: f32[]) -> f32[] {
  %Arg_0.906 = f32[] parameter(0)
  %Arg_1.907 = f32[] parameter(1)
  ROOT %add.908 = f32[] add(%Arg_0.906, %Arg_1.907), metadata={op_name="jit(_step)/add" stack_frame_id=2}
}

%bitcast_copy_fusion_computation.77 (param_0.312: f32[4]) -> f32[4] {
  %param_0.312 = f32[4]{0} parameter(0)
  ROOT %bitcast.88 = f32[4]{0} bitcast(%param_0.312)
}

ENTRY %main.537_spmd (Arg_0.501: f32[4], /*index=5*/Arg_5.502: f32[4]) -> f32[4] {
  %Arg_0.501 = f32[4]{0} parameter(0)
  %Arg_5.502 = f32[4]{0} parameter(1)
  %ws.19 = f32[4]{0} fusion(%Arg_0.501), kind=kLoop, calls=%bitcast_copy_fusion_computation.77
  ROOT %ar.70 = f32[4]{0} all-reduce(%ws.19), to_apply=%region_18.408_spmd.clone, metadata={op_name="ar" stack_frame_id=2}
}
"""
    self.assertEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_b))
    # Positive controls: rewiring operands or inserting an extra barrier must fail equality.
    hlo_rewired = hlo_b.replace("add(%Arg_0.906, %Arg_1.907)", "add(%Arg_1.907, %Arg_0.906)")
    self.assertNotEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_rewired))
    hlo_barrier = hlo_b.replace(
        "ROOT %ar.70 = f32[4]{0} all-reduce(%ws.19)",
        "%opt-barrier.1 = f32[4]{0} opt-barrier(%ws.19)\n" "  ROOT %ar.70 = f32[4]{0} all-reduce(%opt-barrier.1)",
    )
    self.assertNotEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_barrier))


class FsdpPrefetchCorrectnessTest(parameterized.TestCase):
  """Numerical parity, path transparency, and edge-case tests on 4-device CPU."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4).")

  @parameterized.named_parameters(
      ("explicit_full_pf", "explicit", "full", False, 1, 6),
      ("explicit_full_pf_drs", "explicit", "full", True, 1, 6),
      ("auto_full_pf", "auto", "full", False, 1, 6),
      ("explicit_minimal_pf", "explicit", "minimal", False, 1, 6),
      ("explicit_minimal_pf_drs", "explicit", "minimal", True, 1, 6),
      ("explicit_axis0_pf_drs", "explicit", "full", True, 0, 6),
      ("explicit_2layers_pf_drs", "explicit", "full", True, 1, 2),
      ("explicit_3layers_pf_drs", "explicit", "full", True, 1, 3),
      ("explicit_1layer_full", "explicit", "full", True, 1, 1),
      ("explicit_1layer_minoff", "explicit", "minimal_offloaded", True, 1, 1),
  )
  def test_qwen3_prefetch_matches_baseline_loss_grads_and_paths(
      self, mode, remat_policy, defer_rs, scan_axis, num_layers
  ):
    explicit = mode == "explicit"
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=explicit)
    self.enter_context(jax.set_mesh(mesh))
    tokens, positions = _make_batch(mesh)

    cfg_base = _tiny_qwen3_config(
        num_layers=num_layers,
        param_scan_axis=scan_axis,
        remat_policy=remat_policy,
        prefetch_fsdp_weights=False,
        shard_mode=mode,
    )
    cfg_pf = _tiny_qwen3_config(
        num_layers=num_layers,
        param_scan_axis=scan_axis,
        remat_policy=remat_policy,
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=defer_rs,
        shard_mode=mode,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)

    # Path-transparency check: identical parameter keys, shapes, and dtypes.
    p_base, p_pf = _flat_params(m_base), _flat_params(m_pf)
    self.assertEqual(set(p_base.keys()), set(p_pf.keys()))
    for k in p_base:
      self.assertEqual(p_base[k].shape, p_pf[k].shape, msg=str(k))
      self.assertEqual(p_base[k].dtype, p_pf[k].dtype, msg=str(k))

    step = nnx.jit(_loss_and_grad_fn(explicit))
    loss_base, grad_base = step(m_base, tokens, positions)
    loss_pf, grad_pf = step(m_pf, tokens, positions)

    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    gb, gp = _flat_params(grad_base), _flat_params(grad_pf)
    self.assertEqual(set(gb.keys()), set(gp.keys()))
    for k in gb:
      np.testing.assert_allclose(gp[k], gb[k], rtol=1e-5, atol=1e-5, err_msg=str(k))

    # Negative control: perturbing one scanned weight must break loss parity.
    perturbed = _build_qwen3(cfg_pf, mesh)
    perturbed.decoder.layers.mlp.wi_0.kernel[...] = perturbed.decoder.layers.mlp.wi_0.kernel[...] + 0.5
    loss_perturbed, _ = step(perturbed, tokens, positions)
    self.assertGreater(abs(float(loss_perturbed) - float(loss_base)), 1e-4)

  @parameterized.named_parameters(
      ("fsdp2_tp2", {"fsdp": 2, "tensor": 2}, "fsdp"),
      ("fsdp2_fsdp_t2", {"fsdp": 2, "fsdp_transpose": 2}, ("fsdp", "fsdp_transpose")),
      ("fsdp2_dp2", {"data": 2, "fsdp": 2}, ("data", "fsdp")),
  )
  def test_multiaxis_sharding_matches_baseline(self, ici, batch_axes):
    mesh = create_mesh(devices=jax.devices()[:4], ici=ici, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    tokens, positions = _make_batch(mesh, batch_axes=batch_axes)

    cfg_base = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(
        num_layers=4,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)
    step = nnx.jit(_loss_and_grad_fn(True, batch_axes=batch_axes))
    loss_base, grad_base = step(m_base, tokens, positions)
    loss_pf, grad_pf = step(m_pf, tokens, positions)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    gb, gp = _flat_params(grad_base), _flat_params(grad_pf)
    for k in gb:
      np.testing.assert_allclose(gp[k], gb[k], rtol=1e-5, atol=1e-5, err_msg=str(k))

  def test_tuple_carry_and_stateful_rest_match_regular_scan(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    def make_stack(enable_prefetch: bool):
      seq = nnx.Sequential(*[_TupleCarryStatefulLayer(8, mesh, rngs=nnx.Rngs(i)) for i in range(6)])
      scanned = scan.scan_layers(seq, param_scan_axis=1)
      scanned.remat = remat.Remat(jax.checkpoint_policies.nothing_saveable)
      if enable_prefetch:
        fsdp_prefetch.apply_fsdp_prefetch(scanned, defer_grad_reduce_scatter=True, mesh=mesh)
      return scanned

    s_base = make_stack(False)
    s_pf = make_stack(True)
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32) * 0.3, NamedSharding(mesh, P("fsdp", None)))
    aux0 = jax.device_put(jnp.array(0.0, dtype=jnp.float32), NamedSharding(mesh, P()))

    def loss_fn(mod):
      y, aux = mod((x, aux0), scale=0.5)
      return jnp.mean(y) + aux

    loss_base, grad_base = nnx.value_and_grad(loss_fn)(s_base)
    loss_pf, grad_pf = nnx.value_and_grad(loss_fn)(s_pf)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_pf.weight[...], grad_base.weight[...], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(s_pf.calls[...], s_base.calls[...])

  def test_remat_none_fails_closed(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg = _tiny_qwen3_config(num_layers=4, remat_policy="none", prefetch_fsdp_weights=True)
    model = _build_qwen3(cfg, mesh)
    tokens, positions = _make_batch(mesh)
    with self.assertRaisesRegex(ValueError, "remat_policy='none'"):
      model(tokens, positions)

  def test_non_fsdp_mesh_and_disabled_modes_fall_back_cleanly(self):
    mesh_dp = create_mesh(devices=jax.devices()[:4], ici={"data": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh_dp))
    cfg_base = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=True)
    m_base = _build_qwen3(cfg_base, mesh_dp)
    m_pf = _build_qwen3(cfg_pf, mesh_dp)
    tokens, positions = _make_batch(mesh_dp, batch_axes="data")
    step = nnx.jit(_loss_and_grad_fn(True, batch_axes="data"))
    l_base, _ = step(m_base, tokens, positions)
    l_pf, _ = step(m_pf, tokens, positions)
    np.testing.assert_allclose(l_pf, l_base, rtol=1e-6, atol=1e-6)

    # Decode mode and host parameter offload leave model untouched.
    m_dec = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_base, cfg_pf, mesh=mesh_dp, model_mode="decode")
    self.assertIsNone(m_dec.decoder.layers.fsdp_prefetch)
    cfg_offload = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=True)
    cfg_offload.parameter_memory_host_offload = True
    m_off = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_base, cfg_offload, mesh=mesh_dp)
    self.assertIsNone(m_off.decoder.layers.fsdp_prefetch)

  def test_transform_order_invariance_and_dual_driver_wiring(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg = _tiny_qwen3_config(
        num_layers=4,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    # Apply prefetch before scan and verify ScannedLayers inherits it.
    m_before = create_qwen3_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
    m_before = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_before, cfg, mesh=mesh)
    m_before = scan.maybe_apply_scan(m_before, cfg)
    m_before = remat.maybe_apply_remat(m_before, cfg)
    self.assertIsInstance(m_before.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)

    # Both production driver (`model_creation_utils.from_config`) and `m3/train/loop.py:setup_model` wire it.
    m_mcu = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(m_mcu.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)
    self.assertTrue(m_mcu.decoder.layers.fsdp_prefetch.defer_grad_reduce_scatter)

    m_loop, _ = m3_loop.setup_model(cfg, mesh)
    self.assertIsInstance(m_loop.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)
    self.assertTrue(m_loop.decoder.layers.fsdp_prefetch.defer_grad_reduce_scatter)

  def test_minimal_offloaded_remat_six_layers_parity(self):
    """6-layer `minimal_offloaded` exercises multi-trip `unroll=2` host-stack residual offload."""
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg_base = _tiny_qwen3_config(num_layers=6, remat_policy="minimal_offloaded", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=6, remat_policy="minimal_offloaded", prefetch_fsdp_weights=True)
    cfg_drs = _tiny_qwen3_config(
        num_layers=6,
        remat_policy="minimal_offloaded",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)
    m_drs = _build_qwen3(cfg_drs, mesh)
    tokens, positions = _make_batch(mesh)
    step = nnx.jit(_loss_and_grad_fn(True))
    loss_base, grads_base = step(m_base, tokens, positions)
    for m in (m_pf, m_drs):
      loss_m, grads_m = step(m, tokens, positions)
      np.testing.assert_allclose(loss_m, loss_base, rtol=1e-6, atol=1e-6)
      for k, ref_g in _flat_params(grads_base).items():
        np.testing.assert_allclose(_flat_params(grads_m)[k], ref_g, rtol=1e-5, atol=1e-5, err_msg=f"grad mismatch {k}")

  @parameterized.named_parameters(
      ("explicit_full_pf", "explicit", "full", False),
      ("explicit_full_pf_drs", "explicit", "full", True),
      ("auto_full_pf", "auto", "full", False),
      ("explicit_minoff_pf", "explicit", "minimal_offloaded", False),
      ("explicit_minoff_pf_drs", "explicit", "minimal_offloaded", True),
  )
  def test_floating_point_kwarg_gradients_match_regular_scan(self, mode, remat_policy, defer_rs):
    """Floating-point `jax.Array` kwargs passed to `ScannedLayers` preserve exact non-zero gradients."""
    explicit = mode == "explicit"
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=explicit)
    self.enter_context(jax.set_mesh(mesh))

    def build_stack(enable_prefetch: bool):
      seq = nnx.Sequential(*[_DynamicKwargLayer(8, mesh, explicit, rngs=nnx.Rngs(10 + i)) for i in range(6)])
      sc = scan.scan_layers(seq, param_scan_axis=1)
      sc = remat.maybe_apply_remat(sc, SimpleNamespace(remat_policy=remat_policy))
      if enable_prefetch:
        fsdp_prefetch.apply_fsdp_prefetch(sc, defer_grad_reduce_scatter=defer_rs, mesh=mesh)
      return sc

    m_base = build_stack(False)
    m_pf = build_stack(True)
    x = jax.device_put(
        jax.random.normal(jax.random.key(1), (4, 4, 8), dtype=jnp.float32),
        NamedSharding(mesh, P("fsdp", None, None)),
    )
    bias_1d = jax.device_put(jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32), NamedSharding(mesh, P(None)))
    cond_3d = jax.device_put(
        jax.random.normal(jax.random.key(2), (4, 4, 8), dtype=jnp.float32),
        NamedSharding(mesh, P("fsdp", None, None)),
    )
    cos_2d = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32) * 0.9, NamedSharding(mesh, P(None, None)))
    sin_2d = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32) * 0.1, NamedSharding(mesh, P(None, None)))
    pos = jax.device_put(
        jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (4, 4)),
        NamedSharding(mesh, P("fsdp", None)),
    )

    def eval_loss_and_grads(mod):
      graphdef, state = nnx.split(mod)

      def loss_fn(st, b1, c3, rp):
        out = nnx.merge(graphdef, st)(x, per_layer_bias=b1, cond_state=c3, rope_pair=rp, positions=pos, scale=0.7)
        return jnp.sum(out**2)

      return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3)))(state, bias_1d, cond_3d, (cos_2d, sin_2d))

    loss_base, (gs_base, gb_base, gc_base, (gcos_base, gsin_base)) = eval_loss_and_grads(m_base)
    loss_pf, (gs_pf, gb_pf, gc_pf, (gcos_pf, gsin_pf)) = eval_loss_and_grads(m_pf)

    self.assertGreater(float(jnp.linalg.norm(gb_pf)), 1.0)
    self.assertGreater(float(jnp.linalg.norm(gc_pf)), 1.0)
    self.assertGreater(float(jnp.linalg.norm(gcos_pf)), 1.0)
    self.assertGreater(float(jnp.linalg.norm(gsin_pf)), 1.0)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gs_pf.weight[...], gs_base.weight[...], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gb_pf, gb_base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gc_pf, gc_base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gcos_pf, gcos_base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gsin_pf, gsin_base, rtol=1e-4, atol=1e-4)

  def test_decoder_layer_input_offload_matches_baseline(self):
    """`decoder_layer_input='offload'` preserves exact loss and gradients on short and multi-trip loops."""
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    tokens, positions = _make_batch(mesh)
    step = nnx.jit(_loss_and_grad_fn(True))
    for num_layers in (4, 6):
      cfg_base = _tiny_qwen3_config(
          num_layers=num_layers,
          remat_policy="full",
          prefetch_fsdp_weights=False,
          decoder_layer_input="offload",
      )
      cfg_pf = _tiny_qwen3_config(
          num_layers=num_layers,
          remat_policy="full",
          prefetch_fsdp_weights=True,
          prefetch_fsdp_defer_grad_reduce_scatter=True,
          decoder_layer_input="offload",
      )
      loss_base, grads_base = step(_build_qwen3(cfg_base, mesh), tokens, positions)
      loss_pf, grads_pf = step(_build_qwen3(cfg_pf, mesh), tokens, positions)
      np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
      for k, ref_g in _flat_params(grads_base).items():
        np.testing.assert_allclose(_flat_params(grads_pf)[k], ref_g, rtol=1e-5, atol=1e-5, err_msg=str(k))

  def test_replicated_expert_carry_on_multiaxis_explicit_mesh(self):
    """Explicit `fsdp + expert` mesh works when activations replicate across `expert`."""
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 2, "expert": 2}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    def build_stack(enable_prefetch: bool):
      seq = nnx.Sequential(*[_DynamicKwargLayer(8, mesh, True, rngs=nnx.Rngs(20 + i)) for i in range(6)])
      sc = scan.scan_layers(seq, param_scan_axis=1)
      sc.remat = remat.Remat(jax.checkpoint_policies.nothing_saveable)
      if enable_prefetch:
        fsdp_prefetch.apply_fsdp_prefetch(sc, defer_grad_reduce_scatter=True, mesh=mesh)
      return sc

    m_base = build_stack(False)
    m_pf = build_stack(True)
    x_rep_ep = jax.device_put(
        jax.random.normal(jax.random.key(3), (4, 4, 8), dtype=jnp.float32),
        NamedSharding(mesh, P("fsdp", None, None)),
    )
    loss_base, grads_base = nnx.jit(nnx.value_and_grad(lambda m: jnp.mean(m(x_rep_ep) ** 2)))(m_base)
    loss_pf, grads_pf = nnx.jit(nnx.value_and_grad(lambda m: jnp.mean(m(x_rep_ep) ** 2)))(m_pf)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grads_pf.weight[...], grads_base.weight[...], rtol=1e-5, atol=1e-5)


class FsdpPrefetchHloStructureTest(absltest.TestCase):
  """Compiled-HLO structural and byte-identity tests (with positive & negative controls)."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4).")

  def _compile_model_hlo(self, model, cfg, mesh, logits_transform=None):
    """Compiles one train step and returns post-optimization HLO text."""
    tokens, positions = _make_batch(mesh)
    graphdef, state = nnx.split(model)
    explicit = cfg.shard_mode == "explicit"
    out_spec = P("fsdp", None, None)

    def _step(st, tok, pos):
      def loss_fn(s):
        logits = nnx.merge(graphdef, s)(tok, pos)
        if logits_transform is not None:
          logits = logits_transform(logits)
        if explicit:
          lp = jax.sharding.auto_axes(jax.nn.log_softmax, out_sharding=out_spec)(logits)
        else:
          lp = jax.nn.log_softmax(logits)
        return -jnp.mean(lp[..., 0])

      return jax.value_and_grad(loss_fn)(st)

    compiled = jax.jit(_step).lower(state, tokens, positions).compile()
    return compiled.as_text()

  def _compile_train_step_hlo(self, cfg, mesh):
    return self._compile_model_hlo(_build_qwen3(cfg, mesh), cfg, mesh)

  def test_disabled_hlo_is_byte_identical_to_untransformed_baseline(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    cfg_off = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=False)
    # Build without calling `maybe_apply_fsdp_prefetch` at all.
    m_plain = create_qwen3_model(cfg_off, mesh=mesh, rngs=nnx.Rngs(0))
    m_plain = scan.maybe_apply_scan(m_plain, cfg_off)
    m_plain = remat.maybe_apply_remat(m_plain, cfg_off)
    m_disabled = _build_qwen3(cfg_off, mesh)
    hlo_untransformed, hlo_disabled = [self._compile_model_hlo(m, cfg_off, mesh) for m in (m_plain, m_disabled)]
    norm_untransformed = _normalize_hlo(hlo_untransformed)
    norm_disabled = _normalize_hlo(hlo_disabled)
    self.assertEqual(norm_disabled, norm_untransformed)

    # Also verify stability across inner trace-cache eviction (reproduces xdist memory-pressure frame shifts).
    jax.clear_caches()
    gc.collect()
    hlo_disabled_evicted = self._compile_model_hlo(_build_qwen3(cfg_off, mesh), cfg_off, mesh)
    self.assertEqual(_normalize_hlo(hlo_disabled_evicted), norm_untransformed)

    # Positive controls: enabling prefetch, inserting a single barrier, or inserting an extra collective
    # must each fail `_normalize_hlo` equality.
    cfg_on = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=True)
    hlo_enabled = self._compile_model_hlo(_build_qwen3(cfg_on, mesh), cfg_on, mesh)
    self.assertNotEqual(_normalize_hlo(hlo_enabled), norm_disabled)

    hlo_with_barrier = self._compile_model_hlo(m_plain, cfg_off, mesh, logits_transform=jax.lax.optimization_barrier)
    self.assertNotEqual(_normalize_hlo(hlo_with_barrier), norm_untransformed)

    hlo_with_extra_ag = self._compile_model_hlo(
        m_plain,
        cfg_off,
        mesh,
        logits_transform=lambda x: jax.lax.with_sharding_constraint(x, P(None, None, None)),
    )
    self.assertNotEqual(_normalize_hlo(hlo_with_extra_ag), norm_untransformed)

  def test_compiled_hlo_prefetches_next_layer_weight_before_current_layer_compute_ends(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    cfg_off = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=True)
    cfg_drs = _tiny_qwen3_config(
        num_layers=6,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )

    hlo_off = self._compile_train_step_hlo(cfg_off, mesh)
    hlo_pf = self._compile_train_step_hlo(cfg_pf, mesh)
    hlo_drs = self._compile_train_step_hlo(cfg_drs, mesh)

    prefetch_scope_re = re.compile(r'op_name="[^"]*fsdp_prefetch')
    # Negative control: disabled HLO has no `fsdp_prefetch` op scope.
    self.assertIsNone(prefetch_scope_re.search(hlo_off))
    # Positive control: enabled HLO carries `fsdp_prefetch` op scope and explicit collectives.
    self.assertIsNotNone(prefetch_scope_re.search(hlo_pf))
    self.assertIn("all-gather(", hlo_pf)
    self.assertIn("reduce-scatter(", hlo_pf)

    # Find the forward `while` loop body inside `hlo_pf`: with `num_layers=6` and `unroll=2`,
    # layers 1..4 execute in a 2-trip unrolled loop (2 layers/iter), carrying prefetched weights
    # and issuing layer k+1's all-gather BEFORE layer k's compute finishes.
    comps_pf = _parse_hlo_computations(hlo_pf)
    fwd_bodies = [
        (name, lines)
        for name, lines in comps_pf.items()
        if any(prefetch_scope_re.search(ln) for ln in lines)
        and any("all-gather(" in ln for ln in lines)
        and not any("reduce-scatter(" in ln for ln in lines)
    ]
    self.assertNotEmpty(fwd_bodies, "Expected a compiled forward prefetch while-loop body with all-gather ops.")
    dot_comps = {name for name, lines in comps_pf.items() if any(" dot(" in ln or "= dot(" in ln for ln in lines)}

    def _is_dot_op(line: str) -> bool:
      if " dot(" in line or "= dot(" in line:
        return True
      m = re.search(r"\bcalls=(%[\w.\-]+)", line)
      return bool(m and m.group(1) in dot_comps)

    _, fwd_lines = fwd_bodies[0]
    ag_indices = [i for i, ln in enumerate(fwd_lines) if "all-gather(" in ln]
    dot_indices = [i for i, ln in enumerate(fwd_lines) if _is_dot_op(ln)]
    self.assertNotEmpty(ag_indices, "Expected weight all-gather instructions in forward prefetch while body.")
    self.assertNotEmpty(dot_indices, "Expected matmul dot instructions in forward prefetch while body.")
    # Prefetched layer k+1's weight all-gather is produced before layer k's compute finishes.
    self.assertLess(min(ag_indices), max(dot_indices))
    # Unroll=2 body has 2 layers of weight all-gathers (7 FSDP weights/layer * 2 = 14) vs 7 in non-prefetch.
    self.assertGreaterEqual(len(ag_indices), 12)

    # Negative control: non-prefetch forward while body has only 1 layer of weight all-gathers (7).
    comps_off = _parse_hlo_computations(hlo_off)
    off_fwd_bodies = [
        lines
        for lines in comps_off.values()
        if any("all-gather(" in ln for ln in lines) and not any("reduce-scatter(" in ln for ln in lines)
    ]
    self.assertNotEmpty(off_fwd_bodies)
    self.assertLess(sum(1 for ln in off_fwd_bodies[0] if "all-gather(" in ln), len(ag_indices))

    # Deferred RS keeps backward reduce-scatters in the backward while body and changes the carry layout.
    self.assertIn("reduce-scatter(", hlo_drs)
    self.assertNotEqual(hlo_drs, hlo_pf)

  def test_explicit_sequence_parallel_parity_and_sow_and_config_guards(self):
    """Verifies Explicit+SP (tensor_sequence=2) parity, router SP contracting, sow fail-closed, and TrainerConfig."""
    import types  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from maxtext.m3 import models as m3_models  # pylint: disable=import-outside-toplevel
    from maxtext.m3.configs.config import TrainerConfig  # pylint: disable=import-outside-toplevel
    from maxtext.m3.core.sharding import mesh_from_config, sharding_from_config  # pylint: disable=import-outside-toplevel
    from maxtext.utils import max_logging  # pylint: disable=import-outside-toplevel
    from tests.utils import m3_test_utils  # pylint: disable=import-outside-toplevel

    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices; got {len(devices)}")

    # 1. Explicit + tensor_sequence=2 Qwen3 forward + backward parity (drs=False and drs=True).
    cfg_ref = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        dtype="float32",
        weight_dtype="float32",
        float32_logits="true",
        scan_layers="true",
        remat_policy="full",
        shard_mode="explicit",
        ici_fsdp_parallelism=2,
        ici_tensor_sequence_parallelism=2,
        ici_tensor_parallelism=1,
        max_target_length=16,
        per_device_batch_size=1,
        prefetch_fsdp_weights=False,
    )
    mesh_exp = mesh_from_config(cfg_ref)
    shd_exp = sharding_from_config(cfg_ref)

    _loss_and_grads = nnx.jit(nnx.value_and_grad(m3_loop.loss_fn))

    with jax.set_mesh(mesh_exp):
      m_ref = scan.maybe_apply_scan(
          m3_models.create_model(cfg_ref, mesh_exp, rngs=nnx.Rngs(0), sharding=shd_exp), cfg_ref
      )
      m_ref = remat.maybe_apply_remat(m_ref, cfg_ref)
      batch = m3_loop.make_synthetic_batch(cfg_ref, mesh_exp, shd_exp, jax.random.key(19))
      loss_ref, grads_ref = _loss_and_grads(m_ref, batch)

    ref_flat = {"/".join(map(str, p)): np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(grads_ref, nnx.Param))}
    for drs in (False, True):
      with jax.set_mesh(mesh_exp):
        m_pf = scan.maybe_apply_scan(
            m3_models.create_model(cfg_ref, mesh_exp, rngs=nnx.Rngs(0), sharding=shd_exp), cfg_ref
        )
        m_pf = remat.maybe_apply_remat(m_pf, cfg_ref)
        m_pf = fsdp_prefetch.apply_fsdp_prefetch(m_pf, defer_grad_reduce_scatter=drs, mesh=mesh_exp)
        loss_pf, grads_pf = _loss_and_grads(m_pf, batch)
      np.testing.assert_allclose(float(loss_pf), float(loss_ref), rtol=1e-5, atol=1e-5)
      pf_flat = {"/".join(map(str, p)): np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(grads_pf, nnx.Param))}
      for k, g_r in ref_flat.items():
        np.testing.assert_allclose(pf_flat[k], g_r, rtol=1e-5, atol=1e-5, err_msg=f"drs={drs} {k}")

    # 2. FSDP-only router weight P("fsdp", None) contracting P("fsdp", "tensor_sequence", None).
    class _RouterContractSPLayer(nnx.Module):
      """Toy SP layer with FSDP-only router weight."""

      def __init__(self, rngs: nnx.Rngs):
        self.router = nnx.Param(
            jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32) * 0.1,
            out_sharding=P("fsdp", None),
        )
        self.proj = nnx.Param(
            jax.random.normal(rngs.params(), (4, 8), dtype=jnp.float32) * 0.1,
            out_sharding=P(None, "fsdp"),
        )

      def __call__(self, x: jax.Array) -> jax.Array:
        scores = jnp.einsum("bsd,de->bse", x, self.router[...], out_sharding=P("fsdp", "tensor_sequence", None))
        out = jnp.einsum(
            "bse,ed->bsd", jax.nn.tanh(scores), self.proj[...], out_sharding=P("fsdp", "tensor_sequence", None)
        )
        return x + out

    mesh_sp = create_mesh(ici={"fsdp": 2, "tensor_sequence": 2}, devices=devices[:4], explicit=True)
    with jax.set_mesh(mesh_sp):
      seq_ref = scan.scan_layers(
          nnx.Sequential(*[_RouterContractSPLayer(nnx.Rngs(i)) for i in range(4)]), param_scan_axis=0
      )
      remat.apply_remat(seq_ref, jax.checkpoint_policies.nothing_saveable)
      seq_pf = scan.scan_layers(
          nnx.Sequential(*[_RouterContractSPLayer(nnx.Rngs(i)) for i in range(4)]), param_scan_axis=0
      )
      remat.apply_remat(seq_pf, jax.checkpoint_policies.nothing_saveable)
      fsdp_prefetch.apply_fsdp_prefetch(seq_pf, defer_grad_reduce_scatter=True, mesh=mesh_sp)
      x_sp = jax.device_put(
          jax.random.normal(jax.random.key(31), (2, 4, 8), dtype=jnp.float32),
          NamedSharding(mesh_sp, P("fsdp", "tensor_sequence", None)),
      )
      l_r, g_r = nnx.jit(nnx.value_and_grad(lambda m, z: jnp.mean(m(z) ** 2)))(seq_ref, x_sp)
      l_p, g_p = nnx.jit(nnx.value_and_grad(lambda m, z: jnp.mean(m(z) ** 2)))(seq_pf, x_sp)
    np.testing.assert_allclose(float(l_p), float(l_r), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g_p.router[...]), np.asarray(g_r.router[...]), rtol=1e-6, atol=1e-6)

    # 3. Creating new non-Param state inside __call__ raises ValueError instead of silently dropping it.
    class _DynamicStateLayer(nnx.Module):
      """Toy layer mutating new Intermediate state in __call__."""

      def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32), out_sharding=P("fsdp", None))

      def __call__(self, x: jax.Array) -> jax.Array:
        self.leaked = nnx.Intermediate(x)
        return x @ self.w[...]

    mesh_auto = create_mesh(ici={"fsdp": 4}, devices=devices[:4])
    with jax.set_mesh(mesh_auto):
      dyn_seq = scan.scan_layers(nnx.Sequential(*[_DynamicStateLayer(nnx.Rngs(i)) for i in range(3)]), param_scan_axis=0)
      remat.apply_remat(dyn_seq, jax.checkpoint_policies.nothing_saveable)
      fsdp_prefetch.apply_fsdp_prefetch(dyn_seq, mesh=mesh_auto)
      x_auto = jax.device_put(jnp.ones((4, 4, 8), dtype=jnp.float32), NamedSharding(mesh_auto, P("fsdp", None, None)))
      with self.assertRaisesRegex(ValueError, "cannot create new non-Param state"):
        _ = nnx.jit(lambda m, z: m(z))(dyn_seq, x_auto)

    # 4. TrainerConfig validation and scan_layers=False / 0-match warnings.
    with self.assertRaisesRegex(ValueError, "remat_policy='none' is not supported with prefetch_fsdp_weights=True"):
      TrainerConfig(prefetch_fsdp_weights=True)
    with self.assertRaisesRegex(ValueError, "remat_policy='none' is not supported with prefetch_fsdp_weights=True"):
      TrainerConfig(prefetch_fsdp_weights=True, remat_policy="")
    valid_tc = TrainerConfig(prefetch_fsdp_weights=True, remat_policy="FULL")
    self.assertTrue(valid_tc.prefetch_fsdp_weights)

    class _NoSeqModel(nnx.Module):
      """Toy model without ScannedLayers/Sequential containers."""

      def __init__(self):
        self.p = nnx.Param(jnp.ones((2, 2)))

    no_seq = _NoSeqModel()
    with mock.patch.object(max_logging, "warning") as warn_spy:
      fsdp_prefetch.apply_fsdp_prefetch(no_seq)
      self.assertEqual(warn_spy.call_count, 1)
      warn_spy.reset_mock()
      stub_unscanned = types.SimpleNamespace(prefetch_fsdp_weights=True, scan_layers=False)
      out_unscanned = fsdp_prefetch.maybe_apply_fsdp_prefetch(no_seq, stub_unscanned)
      self.assertIs(out_unscanned, no_seq)
      self.assertEqual(warn_spy.call_count, 1)


if __name__ == "__main__":
  absltest.main()
