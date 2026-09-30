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

"""Per-layer delayed scaling for the fp8 backward (e5m2 cotangent) quantizers (`bwd_delayed_scaling`).

With `quantization=fp8_full` and Qwix, every backward quantizer calibrates its cotangent at runtime (absmax: a
reduction over the tensor that the quantize waits on) or with a constant (`fixed,R`). Delayed scaling, in the style
of Transformer Engine, quantizes each call site's cotangent with a per-tensor scale R taken from state: R is `margin`
times the maximum of the last H global amaxes of that site, so the amax measured in step t sets the scale of step t+1
and no quantize waits on its own reduction.

State. Each DeepSeek decoder layer owns a `_overwrite_with_gradient` variable `bwd_dscale` of shape (NSLOTS, 1 + H):
[slot, 0] is R (in the units of a `fixed,R` calibration, i.e. the quantizer's scale is R / qmax), [slot, 1:] the last
H global amaxes, most recent first. Scanned layers stack it to (layers, NSLOTS, 1 + H). Each quantized call site in the
layer (the Qwix dot_generals and the megablox gmms) takes a slot in trace order and runs inside `tap`, a custom_vjp
whose forward is the op itself and whose backward

  1. runs the op's own backward with R available to its quantizers (see below); the first quantizer call also
     reduces max|x| of the tensor it quantizes, in the same place, so XLA can emit it from the quantize's fusion;
  2. returns that amax as the cotangent of the state. Both backward quantizer inputs of a site are g * s, with s the
     constant forward scale of the other operand (equal fixed forward calibrations, see `validate_config`), so one
     max serves both arms.

train_step already overwrites `_overwrite_with_gradient` variables with their gradient; `update_state` turns
(pre-step state, amax) into the next state. The probe is replicated; inside the MoE shard_map the amax is reduced
with a pmax over the manual axes (the transpose of the shard_map then sums the identical per-device values, which is
undone by dividing by the device count).

Quantizer interception. Qwix calibrates through `qarray.calibrate(array, how)`, and both Qwix dot_general_qt and
megablox (`qpl.quantize`) reach it. When the flag is on, the layers' backward calibration methods are replaced by a
sentinel `delayed:<fallback>`, and `qarray.calibrate` is wrapped: a sentinel call inside a tap's backward returns
{'absmax': R} with the shape of a `fixed` calibration (per tensor); a sentinel call outside a tap (a quantized dot
outside the decoder layers that matches the same rule, e.g. the MTP input projection) uses the fallback method.
Every other calibrate call is unchanged.

Numerics differences to absmax: the scale is per tensor (absmax on the dlhs arm is per token, and per column on the
gmm drhs arm), and inside the MoE shard_map it is global rather than per device. Values above R saturate in e5m2;
the step metrics count how often (`learning/bwd_dscale_saturated`).
"""

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
from flax.nnx import variablelib

NSLOTS = 32
STATE_NAME = "bwd_dscale"
SENTINEL = "delayed:"

_CFG = {"enabled": False, "margin": 16.0, "g_scale": None}
_layer_stack = []  # [{"state": array, "kind": str, "n": int}]
_bwd_stack = []  # [{"r": scale R, "amax": this step's amax} of the site whose backward is being traced]
REGISTRY = {}  # kind -> {slot: name}
_ORIG_CALIBRATE = []


def enabled() -> bool:
  return _CFG["enabled"]


def owg_type():
  return variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)


def validate_config(config):
  """Returns the delayed-scaling settings, or None when the flag is off. Raises on unsupported combinations."""
  if not getattr(config, "bwd_delayed_scaling", False):
    return None
  bad = []
  if not (config.use_qwix_quantization and config.quantization == "fp8_full"):
    bad.append("quantization must be fp8_full with use_qwix_quantization=true")
  if config.gradient_accumulation_steps > 1:
    bad.append("gradient_accumulation_steps > 1 (that path does not split the custom-gradient state)")
  if config.ici_pipeline_parallelism > 1 or config.dcn_pipeline_parallelism > 1:
    bad.append("pipeline parallelism")
  if config.bwd_delayed_scaling_history < 1:
    bad.append("bwd_delayed_scaling_history must be >= 1")
  if config.bwd_delayed_scaling_margin < 1.0:
    bad.append("bwd_delayed_scaling_margin must be >= 1")
  if not config.bwd_delayed_scaling_init > 0:
    bad.append("bwd_delayed_scaling_init must be > 0")
  w, a = config.weight_quantization_calibration_method, config.act_quantization_calibration_method
  rng = _symmetric_fixed_range(w)
  if w != a or rng is None:
    bad.append(
        "weight and activation calibration must be the same symmetric fixed range (e.g. 'fixed,-224,224'), so that"
        " the backward quantizer inputs are the cotangent times one constant"
    )
  if bad:
    raise ValueError("bwd_delayed_scaling: unsupported: " + "; ".join(bad))
  from qwix._src.core import numerics  # pylint: disable=import-outside-toplevel

  return {
      "margin": float(config.bwd_delayed_scaling_margin),
      # Forward operands are e4m3fn under fp8_full; a symmetric fixed range R has scale R / qmax.
      "g_scale": rng / float(numerics.get_symmetric_bound(jnp.float8_e4m3fn)),
  }


def _symmetric_fixed_range(method):
  parts = str(method).lower().split(",")
  if parts[0] != "fixed" or len(parts) not in (2, 3):
    return None
  vals = [float(v) for v in parts[1:]]
  if len(vals) == 2 and vals[0] + vals[1] != 0:
    return None
  return abs(vals[-1])


def configure(config):
  """Called when the Qwix provider is built: sets the module state and installs the calibrate wrapper."""
  settings = validate_config(config)
  _CFG["enabled"] = settings is not None
  if settings is not None:
    _CFG.update(settings)
    install_calibrate_override()


def bwd_calibration_method(method: str) -> str:
  """The sentinel that a layer rule uses for its backward quantizers (`method` is the fallback outside a tap)."""
  return SENTINEL + method


def make_state(config):
  """Per-layer state; R starts at bwd_delayed_scaling_init (step 0 behaves like `fixed,init`)."""
  st = jnp.zeros((NSLOTS, 1 + int(config.bwd_delayed_scaling_history)), jnp.float32)
  return owg_type()(st.at[:, 0].set(jnp.float32(config.bwd_delayed_scaling_init)))


@contextlib.contextmanager
def layer_context(state_value, kind: str):
  _layer_stack.append({"state": state_value, "kind": kind, "n": 0})
  try:
    yield
  finally:
    _layer_stack.pop()


def install_calibrate_override():
  """Wraps qwix qarray.calibrate once (qarray.quantize and qpl.quantize resolve it from the module at call time)."""
  if _ORIG_CALIBRATE:
    return
  from qwix._src.core import qarray  # pylint: disable=import-outside-toplevel

  orig = qarray.calibrate
  _ORIG_CALIBRATE.append(orig)

  def calibrate(array, how):
    method = how.calibration_method
    if not (isinstance(method, str) and method.startswith(SENTINEL)):
      return orig(array, how)
    if not _bwd_stack:
      return orig(array, dataclasses.replace(how, calibration_method=method[len(SENTINEL) :]))
    site = _bwd_stack[-1]
    if site["amax"] is None:
      # This step's amax, taken on the tensor being quantized so that XLA can emit it from the quantize's own fusion
      # (one pass over the cotangent, as under absmax). Both backward quantizer inputs of a site carry the same
      # values (validate_config), so the first one suffices.
      site["amax"] = jnp.max(jnp.abs(array.astype(jnp.float32)))
    shape = tuple(1 for _ in qarray.get_scale_shape(array.shape, how))
    return {"absmax": jnp.broadcast_to(site["r"].astype(array.dtype), shape)}

  calibrate.__wrapped__ = orig
  qarray.calibrate = calibrate


def _manual_axes():
  try:
    am = jax.sharding.get_abstract_mesh()
    axes = tuple(getattr(am, "manual_axes", ()) or ())
    n = 1
    for ax in axes:
      n *= am.shape[ax]
    return axes, n
  except Exception:  # pylint: disable=broad-except
    return (), 1


def tap(name, fn, *args):
  """Returns fn(*args); inside a layer context, fn's backward quantizes with the site's stored scale."""
  if not _CFG["enabled"] or not _layer_stack:
    return fn(*args)
  ctx = _layer_stack[-1]
  slot = ctx["n"]
  ctx["n"] += 1
  if slot >= NSLOTS:
    raise ValueError(f"bwd_delayed_scaling: more than {NSLOTS} quantized call sites in {ctx['kind']}")
  REGISTRY.setdefault(ctx["kind"], {}).setdefault(slot, name)
  state = ctx["state"]
  width = state.shape[-1]
  g_scale = _CFG["g_scale"]

  @jax.custom_vjp
  def f(state, args):
    del state
    return fn(*args)

  def f_fwd(state, args):
    out, vjp = jax.vjp(lambda a: fn(*a), args)
    return out, (vjp, state[slot, 0])

  def f_bwd(res, g):
    vjp, r = res
    site = {"r": r, "amax": None}
    _bwd_stack.append(site)
    try:
      (cts,) = vjp(g)
    finally:
      _bwd_stack.pop()
    amax = site["amax"]
    if amax is None:  # the op quantized nothing (e.g. an unquantized module path)
      amax = jnp.max(jnp.abs(g.astype(jnp.float32))) * jnp.float32(g_scale)
    axes, n = _manual_axes()
    if axes:
      amax = jax.lax.pmax(amax, axes)
    cot = jnp.zeros((NSLOTS, width), jnp.float32).at[slot, 0].set(amax / jnp.float32(n))
    return cot, cts

  f.defvjp(f_fwd, f_bwd)
  return f(state, args)


def update_state(old, grad, margin):
  """Next state from the pre-step state and its gradient ([..., 0] = this step's global amax).

  history <- [amax, history[:-1]]; R <- margin * max(history) where that max is > 0, else R is kept (unused slots).
  Returns (new_state, saturated, max_ratio): the number of entries whose amax exceeded the R they were quantized
  with, and max(amax / R).
  """
  amax = grad[..., 0]
  r_old = old[..., 0]
  hist = jnp.concatenate([amax[..., None], old[..., 1:-1]], axis=-1)
  hmax = jnp.max(hist, axis=-1)
  r_new = jnp.where(hmax > 0, jnp.float32(margin) * hmax, r_old)
  saturated = jnp.sum((amax > r_old).astype(jnp.int32))
  ratio = jnp.max(amax / jnp.where(r_old > 0, r_old, 1.0))
  return jnp.concatenate([r_new[..., None], hist], axis=-1), saturated, ratio


def apply_update(custom_params, custom_grads):
  """train_step: replaces the gradient of every bwd_dscale leaf by its updated state. Returns (grads, metrics)."""
  old = dict(jax.tree_util.tree_flatten_with_path(custom_params)[0])
  sat, ratio = [], []

  def _upd(path, g):
    if STATE_NAME not in jax.tree_util.keystr(path):
      return g
    new, s, r = update_state(old[path], g, _CFG["margin"])
    sat.append(s)
    ratio.append(r)
    return new

  grads = jax.tree_util.tree_map_with_path(_upd, custom_grads)
  if not sat:
    raise ValueError("bwd_delayed_scaling: no bwd_dscale state among the custom-gradient variables")
  return grads, {
      "learning/bwd_dscale_saturated": sum(sat[1:], sat[0]).astype(jnp.float32),
      "learning/bwd_dscale_max_amax_over_scale": jnp.max(jnp.stack(ratio)),
  }
