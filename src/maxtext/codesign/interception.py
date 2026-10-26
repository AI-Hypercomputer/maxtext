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

"""Non-intrusive layer interception and module-path tracking for MaxText (NNX & Linen).

Intercepts:
  - `DenseGeneral` (`dot_general` & `einsum` paths, plus input/output activations)
  - `MoeBlock` & `RoutedMoE` (`sparse_matmul`, `dense_matmul`, `fused_moe_matmul`,
    router `gate`)
  - `AttentionOp` (`qk_product`, `wv_product` einsums when targeted)
  - `Decoder._apply_layers_sequentially` (propagates dynamic JAX `layer_idx` iota
    inside `jax.lax.scan` when `scan_layers=True` and layer filtering is configured)
  - Linen `qwix` `QtProvider` (`MultipassEmulationProvider`) if Linen interception
    is requested.
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import logging
import threading
from typing import Any, Callable, Iterator

from maxtext.codesign.multi_pass.multipass_config import (
    MultipassEmulationConfig,
    MultipassRuleConfig,
)
from maxtext.codesign.multi_pass.multipass_kernel import (
    multipass_dot_general,
    multipass_einsum,
    multipass_gmm_operands,
    reconstruct_multipass_tensor,
)


_TLS = threading.local()
_PATCHES_INSTALLED = False
_LOGGED_MATCHED_PATHS: set[str] = set()


def _get_path_stack() -> list[str]:
  stack = getattr(_TLS, "path_stack", None)
  if stack is None:
    stack = []
    _TLS.path_stack = stack
  return stack


def get_current_module_path() -> str:
  """Returns the active `/` -separated module path from thread-local scope."""
  stack = _get_path_stack()
  return "/".join(stack) if stack else ""


@contextmanager
def module_path_scope(name: str) -> Iterator[str]:
  """Pushes `name` onto the thread-local module path stack."""
  stack = _get_path_stack()
  stack.append(name)
  try:
    yield "/".join(stack)
  finally:
    stack.pop()


def get_active_multipass_config() -> MultipassEmulationConfig | None:
  """Returns the currently active `MultipassEmulationConfig` if set."""
  return getattr(_TLS, "active_config", None)


def set_active_multipass_config(
    config: MultipassEmulationConfig | None,
) -> None:
  """Sets the thread-local active `MultipassEmulationConfig`."""
  _TLS.active_config = config


get_active_emulation_config = get_active_multipass_config
set_active_emulation_config = set_active_multipass_config
get_active_sparsity_config = get_active_multipass_config
set_active_sparsity_config = set_active_multipass_config


@contextmanager
def multipass_config_context(
    config: MultipassEmulationConfig | None,
) -> Iterator[MultipassEmulationConfig | None]:
  """Context manager that temporarily activates `config`."""
  prev = get_active_multipass_config()
  set_active_multipass_config(config)
  try:
    yield config
  finally:
    set_active_multipass_config(prev)


emulation_config_context = multipass_config_context
sparsity_config_context = multipass_config_context


def get_current_emulation_step() -> int | None:
  """Returns the current training step tracked for schedule-based emulation."""
  return getattr(_TLS, "current_step", None)


def set_current_emulation_step(step: int | None) -> None:
  """Updates the current training step tracked for schedule-based emulation."""
  _TLS.current_step = int(step) if step is not None else None


@contextmanager
def emulation_step_context(step: int | None) -> Iterator[int | None]:
  """Temporarily sets the active training step for schedule-aware emulation."""
  prev = get_current_emulation_step()
  set_current_emulation_step(step)
  try:
    yield step
  finally:
    set_current_emulation_step(prev)


def get_current_jax_step() -> Any:
  """Returns the currently active traced JAX scalar step (or boolean active flag) if inside a JIT step."""
  return getattr(_TLS, "jax_step", None)


def set_current_jax_step(jax_step: Any) -> None:
  """Sets the traced JAX scalar step (or boolean active flag) for dynamic step gating."""
  _TLS.jax_step = jax_step


@contextmanager
def emulation_jax_step_context(jax_step: Any) -> Iterator[Any]:
  """Temporarily sets the traced JAX step tensor during JIT compilation/execution."""
  prev = get_current_jax_step()
  set_current_jax_step(jax_step)
  try:
    yield jax_step
  finally:
    set_current_jax_step(prev)


def get_current_jax_layer_idx() -> Any:
  """Returns the currently active traced JAX scalar `layer_idx` if inside `scan_layers=True`."""
  return getattr(_TLS, "jax_layer_idx", None)


def set_current_jax_layer_idx(jax_layer_idx: Any) -> None:
  """Sets the traced JAX scalar `layer_idx` for dynamic per-layer gating inside `lax.scan`."""
  _TLS.jax_layer_idx = jax_layer_idx


@contextmanager
def emulation_jax_layer_context(jax_layer_idx: Any) -> Iterator[Any]:
  """Temporarily sets the traced JAX scalar `layer_idx` during scanned decoder layer execution."""
  prev = get_current_jax_layer_idx()
  set_current_jax_layer_idx(jax_layer_idx)
  try:
    yield jax_layer_idx
  finally:
    set_current_jax_layer_idx(prev)


def _compute_active_cond(
    cfg: MultipassEmulationConfig,
    rule: MultipassRuleConfig | None = None,
) -> Any:
  """Computes a dynamic JAX/NumPy boolean scalar condition if step schedule or scanned layer filtering is active."""
  xp: Any = None
  try:
    import jax.numpy as _jnp  # pylint: disable=import-outside-toplevel

    xp = _jnp
  except ImportError:
    try:
      import numpy as _np  # type: ignore  # pylint: disable=import-outside-toplevel

      xp = _np
    except ImportError:
      xp = None

  cond: Any = None
  jax_step = get_current_jax_step()
  if jax_step is not None:
    if isinstance(jax_step, bool) or (xp is not None and getattr(jax_step, "dtype", None) == xp.bool_):
      cond = jax_step
    elif cfg.start_step > 0 or (cfg.end_step is not None and cfg.end_step >= 0):
      cond = jax_step >= cfg.start_step
      if cfg.end_step is not None and cfg.end_step >= 0:
        end_ok = jax_step <= cfg.end_step
        cond = xp.logical_and(cond, end_ok) if xp is not None else bool(cond and end_ok)

  jax_layer_idx = get_current_jax_layer_idx()
  if jax_layer_idx is not None and rule is not None:
    layer_cond: Any = None
    if rule.layers is not None:
      if xp is not None:
        allowed = xp.asarray(rule.layers, dtype=xp.int32)
        layer_cond = xp.any(jax_layer_idx == allowed)
      else:
        layer_cond = int(jax_layer_idx) in rule.layers
    if rule.exclude_layers is not None:
      if xp is not None:
        excluded = xp.asarray(rule.exclude_layers, dtype=xp.int32)
        not_excl = xp.all(jax_layer_idx != excluded)
        layer_cond = not_excl if layer_cond is None else xp.logical_and(layer_cond, not_excl)
      else:
        not_excl = int(jax_layer_idx) not in rule.exclude_layers
        layer_cond = not_excl if layer_cond is None else bool(layer_cond and not_excl)
    if layer_cond is not None:
      if cond is None:
        cond = layer_cond
      else:
        cond = xp.logical_and(cond, layer_cond) if xp is not None else bool(cond and layer_cond)

  return cond


def annotate_model_paths(
    root_module: Any,
    multipass_config: MultipassEmulationConfig | None = None,
) -> int:
  """Walks an NNX `Module` tree and attaches `_emulation_path` to every child module."""
  try:
    from flax import nnx  # pylint: disable=import-outside-toplevel
  except ImportError:
    return 0

  if not isinstance(root_module, nnx.Module):
    return 0

  entries = list(nnx.iter_graph(root_module))
  cfg_obj = getattr(root_module, "config", None) or getattr(
      getattr(root_module, "decoder", None), "config", None
  )
  dense_offset = int(getattr(cfg_obj, "first_num_dense_layers", 0) or 0)
  if dense_offset <= 0:
    dense_indices: set[int] = set()
    for path, _ in entries:
      for seg in path:
        m_d = re.match(r"^dense_layers?_(\d+)$", str(seg))
        if m_d:
          dense_indices.add(int(m_d.group(1)))
    if dense_indices:
      dense_offset = max(dense_indices) + 1

  count = 0
  for path, node in entries:
    if isinstance(node, nnx.Module):
      path_str = "/".join(str(p) for p in path)
      if dense_offset > 0:
        path_str = re.sub(
            r"(^|/)(moe_layers?[_/])(\d+)(?=/|$)",
            lambda m: f"{m.group(1)}{m.group(2)}{int(m.group(3)) + dense_offset}",
            path_str,
        )
      try:
        object.__setattr__(node, "_emulation_path", path_str)
        if multipass_config is not None:
          object.__setattr__(node, "_emulation_config_ref", multipass_config)
        count += 1
      except Exception:  # pylint: disable=broad-exception-caught
        pass
  return count


def _resolve_module_and_config(
    module: Any,
) -> tuple[str, MultipassEmulationConfig | None]:
  """Resolves `(module_path, active_multipass_config)` for an intercepted NNX module."""
  if getattr(module, "_emulation_disabled", False):
    return "", None

  path = getattr(module, "_emulation_path", "") or get_current_module_path()
  if not path:
    path = module.__class__.__name__

  cfg = get_active_multipass_config()
  if cfg is None:
    cfg = getattr(module, "_emulation_config_ref", None)
  if cfg is None:
    mod_cfg = getattr(module, "config", None)
    if mod_cfg is not None:
      cfg = getattr(mod_cfg, "multipass_emulation", None) or getattr(mod_cfg, "sparsity_emulation", None)
  if cfg is not None and not cfg.enabled:
    return path, None
  return path, cfg


def _maybe_log_match(
    path: str,
    op_name: str,
    rule: MultipassRuleConfig | None,
    cfg: MultipassEmulationConfig,
) -> None:
  """Logs the first occurrence of a matched module path and multi-pass rule."""
  if not cfg.log_diagnostics:
    return
  aux_names = [f.feature_name for f in cfg.active_auxiliary_features]
  rule_name = rule.name if rule is not None else "none"
  key = f"{op_name}:{path}:{rule_name}:{','.join(aux_names)}"
  if key not in _LOGGED_MATCHED_PATHS:
    _LOGGED_MATCHED_PATHS.add(key)
    lhs_s = rule.lhs.summary() if (rule and rule.lhs and rule.lhs.enabled) else "off"
    rhs_s = rule.rhs.summary() if (rule and rule.rhs and rule.rhs.enabled) else "off"
    ct_s = f"{rule.cross_term_mode}[{'+'.join(f'({i},{j})' for i, j in rule.effective_fwd_pairs)}]" if rule else "off"
    msg = (
        f"[HardwareEmulation] Matched {op_name} at '{path}' -> rule='{rule_name}' "
        f"(lhs={lhs_s}, rhs={rhs_s}, cross_terms={ct_s}, aux={aux_names})"
    )
    logging.info("%s", msg)
    if len(_LOGGED_MATCHED_PATHS) <= 8:
      print(msg, flush=True)


def compose_emulated_dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: Any,
    path: str,
    cfg: MultipassEmulationConfig,
    base_dot_general_fn: Callable[..., Any],
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
  """Executes a `dot_general` through the active `MultipassEmulationConfig` and any `EmulationFeature` plugins."""
  jax_step = get_current_jax_step()
  step = None if jax_step is not None else get_current_emulation_step()
  if not cfg.is_active_at_step(step):
    return base_dot_general_fn(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  in_scan = get_current_jax_layer_idx() is not None
  rule = (
      cfg.match_rule(
          path,
          "dot_general",
          step=step,
          allow_unindexed_scan_layer=in_scan,
      )
      if cfg.multipass_enabled
      else None
  )
  active_cond = _compute_active_cond(cfg, rule)
  aux_features = cfg.active_auxiliary_features

  if (rule is None or not rule.has_any_multipass) and not aux_features:
    return base_dot_general_fn(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  _maybe_log_match(path, "dot_general", rule, cfg)

  cur_fn = base_dot_general_fn
  for feat in reversed(aux_features):
    prev_fn = cur_fn

    def _make_wrapped(f_obj: Any, next_fn: Callable[..., Any]):
      def _wrapped(
          l_op: Any,
          r_op: Any,
          d_nums: Any,
          precision: Any = None,
          preferred_element_type: Any = None,
      ) -> Any:
        return f_obj.execute_dot_general(
            l_op,
            r_op,
            d_nums,
            path,
            next_fn,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

      return _wrapped

    cur_fn = _make_wrapped(feat, prev_fn)

  if rule is not None and rule.has_any_multipass:
    return multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        rule,
        base_dot_general_fn=cur_fn,
        precision=precision,
        preferred_element_type=preferred_element_type,
        active_cond=active_cond,
    )

  return cur_fn(
      lhs,
      rhs,
      dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


def compose_emulated_einsum(
    subscripts: str,
    lhs: Any,
    rhs: Any,
    path: str,
    cfg: MultipassEmulationConfig,
    base_einsum_fn: Callable[..., Any],
    **kwargs: Any,
) -> Any:
  """Executes a 2-operand `einsum` through `MultipassEmulationConfig` and any `EmulationFeature` plugins."""
  jax_step = get_current_jax_step()
  step = None if jax_step is not None else get_current_emulation_step()
  if not cfg.is_active_at_step(step):
    return base_einsum_fn(subscripts, lhs, rhs, **kwargs)

  in_scan = get_current_jax_layer_idx() is not None
  rule = (
      cfg.match_rule(
          path,
          "einsum",
          step=step,
          allow_unindexed_scan_layer=in_scan,
      )
      if cfg.multipass_enabled
      else None
  )
  active_cond = _compute_active_cond(cfg, rule)
  aux_features = cfg.active_auxiliary_features

  if (rule is None or not rule.has_any_multipass) and not aux_features:
    return base_einsum_fn(subscripts, lhs, rhs, **kwargs)

  _maybe_log_match(path, "einsum", rule, cfg)

  cur_fn = base_einsum_fn
  for feat in reversed(aux_features):
    prev_fn = cur_fn

    def _make_wrapped_einsum(f_obj: Any, next_fn: Callable[..., Any]):
      def _wrapped_einsum(subs: str, l_op: Any, r_op: Any, **kw: Any) -> Any:
        return f_obj.execute_einsum(subs, l_op, r_op, path, next_fn, **kw)

      return _wrapped_einsum

    cur_fn = _make_wrapped_einsum(feat, prev_fn)

  if rule is not None and rule.has_any_multipass:
    return multipass_einsum(
        subscripts,
        lhs,
        rhs,
        rule,
        base_einsum_fn=cur_fn,
        active_cond=active_cond,
        **kwargs,
    )

  return cur_fn(subscripts, lhs, rhs, **kwargs)


def install_layer_patches() -> None:
  """Installs idempotent runtime wrappers around MaxText's core linear, MoE, and attention layers."""
  global _PATCHES_INSTALLED
  if _PATCHES_INSTALLED:
    return

  try:
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_op  # pylint: disable=import-outside-toplevel
    from maxtext.layers import linears  # pylint: disable=import-outside-toplevel
    from maxtext.layers import moe  # pylint: disable=import-outside-toplevel
    from maxtext.layers import nnx_decoders  # pylint: disable=import-outside-toplevel
  except ImportError:
    return

  orig_dense_call = linears.DenseGeneral.__call__
  orig_compute_dot_nnx = getattr(linears, "_compute_dot_general_nnx", None)

  if orig_compute_dot_nnx is not None:

    @functools.wraps(orig_compute_dot_nnx)
    def _patched_compute_dot_general_nnx(
        inputs,
        kernel,
        axis,
        contract_ind,
        matmul_precision,
        quant_dot_general,
        initializing,
        out_sharding=None,
        kernel_scale=None,
        compute_dtype=None,
        native_fp8_compute=False,
        scale_block_size=None,
        act_calibration_method="absmax",
        weight_grad_in_kernel_order=False,
        _orig_fn=orig_compute_dot_nnx,
    ):
      scope = getattr(_TLS, "active_dense_scope", None)
      if scope is None or quant_dot_general is not None or initializing:
        return _orig_fn(
            inputs,
            kernel,
            axis,
            contract_ind,
            matmul_precision,
            quant_dot_general,
            initializing,
            out_sharding,
            kernel_scale=kernel_scale,
            compute_dtype=compute_dtype,
            native_fp8_compute=native_fp8_compute,
            scale_block_size=scale_block_size,
            act_calibration_method=act_calibration_method,
            weight_grad_in_kernel_order=weight_grad_in_kernel_order,
        )
      path, cfg = scope
      if compute_dtype is None:
        compute_dtype = inputs.dtype
      if out_sharding is not None and hasattr(linears, "truncate_out_sharding"):
        out_ndim = (inputs.ndim - len(axis)) + (kernel.ndim - len(contract_ind))
        out_sharding = linears.truncate_out_sharding(out_sharding, out_ndim)
      if kernel_scale is not None or (
          hasattr(linears, "is_fp8_dtype") and linears.is_fp8_dtype(getattr(kernel, "dtype", None))
      ):
        kernel = linears.dequantize_weight(kernel, kernel_scale, compute_dtype=compute_dtype)
      dimension_numbers = ((axis, contract_ind), ((), ()))
      prec = jax.lax.Precision(matmul_precision)
      base_fn = functools.partial(jax.lax.dot_general, out_sharding=out_sharding)
      return compose_emulated_dot_general(
          inputs,
          kernel,
          dimension_numbers,
          path,
          cfg,
          base_dot_general_fn=base_fn,
          precision=prec,
      )

    linears._compute_dot_general_nnx = _patched_compute_dot_general_nnx  # pylint: disable=protected-access

  @functools.wraps(orig_dense_call)
  def _patched_dense_general_call(self, inputs, *args, _orig_call=orig_dense_call, **kwargs):
    path, cfg = _resolve_module_and_config(self)
    if cfg is None:
      return _orig_call(self, inputs, *args, **kwargs)

    jax_step = get_current_jax_step()
    step = None if jax_step is not None else get_current_emulation_step()
    if not cfg.is_active_at_step(step):
      return _orig_call(self, inputs, *args, **kwargs)

    in_scan = get_current_jax_layer_idx() is not None
    dot_rule = (
        cfg.match_rule(
            path,
            "dot_general",
            step=step,
            allow_unindexed_scan_layer=in_scan,
        )
        if cfg.multipass_enabled
        else None
    )
    einsum_rule = (
        cfg.match_rule(
            path,
            "einsum",
            step=step,
            allow_unindexed_scan_layer=in_scan,
        )
        if cfg.multipass_enabled
        else None
    )
    primary_rule = dot_rule or einsum_rule
    active_cond = _compute_active_cond(cfg, primary_rule)
    aux_features = cfg.active_auxiliary_features

    if (
        (dot_rule is None or not dot_rule.has_any_multipass)
        and (einsum_rule is None or not einsum_rule.has_any_multipass)
        and not aux_features
    ):
      return _orig_call(self, inputs, *args, **kwargs)

    if primary_rule is not None and primary_rule.input_activation is not None and primary_rule.input_activation.enabled:
      inputs = reconstruct_multipass_tensor(
          inputs,
          primary_rule.input_activation,
          contracting_axes=(inputs.ndim - 1,),
          use_ste=True,
          active_cond=active_cond,
      )
    for feat in aux_features:
      inputs = feat.transform_input_activation(inputs, path)

    prev_scope = getattr(_TLS, "active_dense_scope", None)
    with module_path_scope(path):
      _TLS.active_dense_scope = (path, cfg)
      try:
        out = _orig_call(self, inputs, *args, **kwargs)
      finally:
        _TLS.active_dense_scope = prev_scope

    if primary_rule is not None and primary_rule.output_activation is not None and primary_rule.output_activation.enabled:
      out = reconstruct_multipass_tensor(
          out,
          primary_rule.output_activation,
          contracting_axes=(out.ndim - 1,),
          use_ste=True,
          active_cond=active_cond,
      )
    for feat in aux_features:
      out = feat.transform_output_activation(out, path)

    return out

  linears.DenseGeneral.__call__ = _patched_dense_general_call

  if hasattr(moe, "RoutedMoE"):
    orig_routed_moe_sparse = moe.RoutedMoE.sparse_matmul
    orig_routed_moe_dense = moe.RoutedMoE.dense_matmul
    orig_routed_moe_fused = getattr(moe.RoutedMoE, "fused_moe_matmul", None)

    def _emulate_moe_inputs_and_kernels(self, inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel=None):
      base_path, cfg = _resolve_module_and_config(self)
      if cfg is None:
        return inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel

      jax_step = get_current_jax_step()
      step = None if jax_step is not None else get_current_emulation_step()
      if not cfg.is_active_at_step(step):
        return inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel

      in_scan = get_current_jax_layer_idx() is not None

      def _match_moe(sub_name: str) -> MultipassRuleConfig | None:
        if not cfg.multipass_enabled:
          return None
        sub_path = f"{base_path}/{sub_name}" if base_path else sub_name
        for op in ("gmm", "ragged_dot", "dot_general"):
          r = cfg.match_rule(
              sub_path,
              op,
              step=step,
              allow_unindexed_scan_layer=in_scan,
          )
          if r is not None and r.has_any_multipass:
            _maybe_log_match(sub_path, op, r, cfg)
            return r
        return None

      r_w0 = _match_moe("wi_0")
      r_w1 = _match_moe("wi_1")
      r_wo = _match_moe("wo")
      active_cond = _compute_active_cond(cfg, r_w0 or r_w1 or r_wo)

      if r_w0 is not None:
        inputs, w0_kernel = multipass_gmm_operands(
            inputs,
            w0_kernel,
            r_w0,
            lhs_contracting_axes=(inputs.ndim - 1,),
            rhs_contracting_axes=(1,),
            lhs_batch_axes=(),
            rhs_batch_axes=(0,),
            active_cond=active_cond,
        )
      if r_w1 is not None and w1_kernel is not None:
        _, w1_kernel = multipass_gmm_operands(
            inputs,
            w1_kernel,
            r_w1,
            lhs_contracting_axes=(inputs.ndim - 1,),
            rhs_contracting_axes=(1,),
            lhs_batch_axes=(),
            rhs_batch_axes=(0,),
            active_cond=active_cond,
        )
      if r_wo is not None and wo_kernel is not None:
        _, wo_kernel = multipass_gmm_operands(
            inputs,
            wo_kernel,
            r_wo,
            lhs_contracting_axes=(inputs.ndim - 1,),
            rhs_contracting_axes=(1,),
            lhs_batch_axes=(),
            rhs_batch_axes=(0,),
            active_cond=active_cond,
        )
      if fused_kernel is not None and r_w0 is not None and r_w0.rhs is not None and r_w0.rhs.enabled:
        fused_kernel = reconstruct_multipass_tensor(
            fused_kernel,
            r_w0.rhs,
            contracting_axes=(1,),
            batch_axes=(0,),
            use_ste=True,
            active_cond=active_cond,
        )

      for feat in cfg.active_auxiliary_features:
        inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel = feat.transform_moe_operands(
            inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel, base_path
        )
      return inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel

    @functools.wraps(orig_routed_moe_sparse)
    def _patched_routed_moe_sparse(
        self,
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        *args,
        **kwargs,
    ):
      inputs, w0_kernel, w1_kernel, wo_kernel, _ = _emulate_moe_inputs_and_kernels(
          self, inputs, w0_kernel, w1_kernel, wo_kernel
      )
      return orig_routed_moe_sparse(
          self,
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          *args,
          **kwargs,
      )

    @functools.wraps(orig_routed_moe_dense)
    def _patched_routed_moe_dense(
        self,
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        *args,
        **kwargs,
    ):
      inputs, w0_kernel, w1_kernel, wo_kernel, _ = _emulate_moe_inputs_and_kernels(
          self, inputs, w0_kernel, w1_kernel, wo_kernel
      )
      return orig_routed_moe_dense(
          self,
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          *args,
          **kwargs,
      )

    moe.RoutedMoE.sparse_matmul = _patched_routed_moe_sparse
    moe.RoutedMoE.dense_matmul = _patched_routed_moe_dense

    if orig_routed_moe_fused is not None:

      @functools.wraps(orig_routed_moe_fused)
      def _patched_routed_moe_fused(  # pylint: disable=keyword-arg-before-vararg
          self,
          inputs,
          gate_logits,
          wo_kernel,
          w0_kernel=None,
          w1_kernel=None,
          fused_kernel=None,
          *args,
          **kwargs,
      ):
        inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel = _emulate_moe_inputs_and_kernels(
            self, inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel
        )
        return orig_routed_moe_fused(
            self,
            inputs,
            gate_logits,
            wo_kernel,
            w0_kernel=w0_kernel,
            w1_kernel=w1_kernel,
            fused_kernel=fused_kernel,
            *args,
            **kwargs,
        )

      moe.RoutedMoE.fused_moe_matmul = _patched_routed_moe_fused

  if hasattr(attention_op, "AttentionOp"):
    orig_qk_product = attention_op.AttentionOp.qk_product
    orig_wv_product = attention_op.AttentionOp.wv_product

    @functools.wraps(orig_qk_product)
    def _patched_qk_product(self, query, key, q_seq_len, model_mode, einsum):
      cfg = get_active_multipass_config()
      if cfg is not None and cfg.enabled and not getattr(self, "_emulation_disabled", False):
        base_path = getattr(self, "_emulation_path", "") or "self_attention/AttentionOp_0"
        qk_path = f"{base_path}/qk_product"
        in_scan = get_current_jax_layer_idx() is not None
        rule = cfg.match_rule(qk_path, "einsum", allow_unindexed_scan_layer=in_scan) if cfg.multipass_enabled else None
        if (rule is not None and rule.has_any_multipass) or cfg.active_auxiliary_features:
          base_fn = einsum

          def _emulated_qk_einsum(subscripts, lhs, rhs, **kw):
            if base_fn is jnp.einsum and "precision" not in kw:
              kw["precision"] = self.config.matmul_precision
            return compose_emulated_einsum(
                subscripts,
                lhs,
                rhs,
                qk_path,
                cfg,
                base_einsum_fn=base_fn,
                **kw,
            )

          einsum = _emulated_qk_einsum
      return orig_qk_product(self, query, key, q_seq_len, model_mode, einsum)

    @functools.wraps(orig_wv_product)
    def _patched_wv_product(self, attn_weights, value, model_mode, einsum):
      cfg = get_active_multipass_config()
      if cfg is not None and cfg.enabled and not getattr(self, "_emulation_disabled", False):
        base_path = getattr(self, "_emulation_path", "") or "self_attention/AttentionOp_0"
        wv_path = f"{base_path}/wv_product"
        in_scan = get_current_jax_layer_idx() is not None
        rule = cfg.match_rule(wv_path, "einsum", allow_unindexed_scan_layer=in_scan) if cfg.multipass_enabled else None
        if (rule is not None and rule.has_any_multipass) or cfg.active_auxiliary_features:
          base_fn = einsum

          def _emulated_wv_einsum(subscripts, lhs, rhs, **kw):
            if base_fn is jnp.einsum and "precision" not in kw:
              kw["precision"] = self.config.matmul_precision
            return compose_emulated_einsum(
                subscripts,
                lhs,
                rhs,
                wv_path,
                cfg,
                base_einsum_fn=base_fn,
                **kw,
            )

          einsum = _emulated_wv_einsum
      return orig_wv_product(self, attn_weights, value, model_mode, einsum)

    attention_op.AttentionOp.qk_product = _patched_qk_product
    attention_op.AttentionOp.wv_product = _patched_wv_product

  for _dec_cls_name in ("Decoder", "NNXDecoder"):
    _dec_cls = getattr(nnx_decoders, _dec_cls_name, None)
    if _dec_cls is not None and hasattr(_dec_cls, "_apply_layers_sequentially"):
      orig_apply_layers_seq = _dec_cls._apply_layers_sequentially  # pylint: disable=protected-access

      def _make_patched_apply_layers_seq(orig_fn):
        @functools.wraps(orig_fn)
        def _patched_apply_layers_sequentially(self, layers, x_in, *args, length: int, **kwargs):
          cfg = get_active_multipass_config()
          has_layer_filter = bool(
              cfg is not None
              and cfg.enabled
              and any(r.layers is not None or r.exclude_layers is not None for r in cfg.rules)
          )
          if not has_layer_filter or length <= 0:
            return orig_fn(self, layers, x_in, *args, length=length, **kwargs)

          orig_lax_scan = jax.lax.scan
          orig_checkpoint = jax.checkpoint
          remat_box: dict[str, Any] = {}

          def _deferred_checkpoint(fun, *c_args, **c_kwargs):
            remat_box["fun"] = fun
            remat_box["args"] = c_args
            remat_box["kwargs"] = c_kwargs
            return fun

          def _scan_with_layer_idx(
              f,
              init,
              xs,
              length=None,
              reverse=False,
              unroll=1,
              _split_transpose=False,
              _outer_len=length,
          ):  # pylint: disable=redefined-outer-name
            jax.lax.scan = orig_lax_scan
            jax.checkpoint = orig_checkpoint
            scan_len = length if length is not None else _outer_len
            idx_xs = jnp.arange(scan_len, dtype=jnp.int32)
            print(
                f"[HardwareEmulation] Layer-indexed scan traced (length={scan_len})",
                flush=True,
            )

            def _f_with_idx(carry, xs_and_idx):
              orig_xs, layer_idx = xs_and_idx
              with emulation_jax_layer_context(layer_idx):
                return f(carry, orig_xs)

            scan_body_fn = _f_with_idx
            if remat_box.get("fun") is f:
              scan_body_fn = orig_checkpoint(_f_with_idx, *remat_box["args"], **remat_box["kwargs"])

            return orig_lax_scan(
                scan_body_fn,
                init,
                (xs, idx_xs),
                length=length,
                reverse=reverse,
                unroll=unroll,
                _split_transpose=_split_transpose,
            )

          jax.lax.scan = _scan_with_layer_idx
          jax.checkpoint = _deferred_checkpoint
          try:
            return orig_fn(self, layers, x_in, *args, length=length, **kwargs)
          finally:
            jax.lax.scan = orig_lax_scan
            jax.checkpoint = orig_checkpoint

        return _patched_apply_layers_sequentially

      _dec_cls._apply_layers_sequentially = _make_patched_apply_layers_seq(  # pylint: disable=protected-access
          orig_apply_layers_seq
      )

  _PATCHES_INSTALLED = True


def build_qwix_multipass_provider(
    multipass_config: MultipassEmulationConfig,
) -> Any:
  """Builds a Qwix `QtProvider`-compatible interceptor for Linen models if `qwix` is installed."""
  import qwix  # pylint: disable=import-outside-toplevel

  class MultipassEmulationProvider(qwix.QtProvider):
    """Qwix provider that intercepts `dot_general` and `einsum` for multi-pass emulation."""

    def __init__(self, cfg: MultipassEmulationConfig):
      super().__init__(rules=[])
      self._emulation_cfg = cfg

    def dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        **kwargs,
    ):
      """Intercepts Linen `dot_general` calls for multi-pass emulation."""
      import jax  # pylint: disable=import-outside-toplevel

      path = get_current_module_path()
      return compose_emulated_dot_general(
          lhs,
          rhs,
          dimension_numbers,
          path,
          self._emulation_cfg,
          base_dot_general_fn=functools.partial(jax.lax.dot_general, **kwargs),
          precision=precision,
          preferred_element_type=preferred_element_type,
      )

    def einsum(self, subscripts, *operands, **kwargs):
      """Intercepts Linen `einsum` calls for multi-pass emulation."""
      import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

      if len(operands) != 2 or not isinstance(subscripts, str):
        return jnp.einsum(subscripts, *operands, **kwargs)
      path = get_current_module_path()
      return compose_emulated_einsum(
          subscripts,
          operands[0],
          operands[1],
          path,
          self._emulation_cfg,
          base_einsum_fn=jnp.einsum,
          **kwargs,
      )

  return MultipassEmulationProvider(multipass_config)


build_qwix_emulation_provider = build_qwix_multipass_provider
build_qwix_sparsity_provider = build_qwix_multipass_provider
