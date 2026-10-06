"""Layer-by-layer + MoE-routing parity trace: PyTorch HF reference vs MaxText (Kimi-K3).

Diagnostic companion to `kimi_k3_real_ckpt.py`. That harness compares only final
logits; this one localizes *where* any divergence originates.

  python tests/utils/kimi_k3_layer_parity.py torch_trace    --num_layers 13 --dtype bfloat16
  python tests/utils/kimi_k3_layer_parity.py maxtext_trace  --num_layers 13 --dtype bfloat16 \
      --mt_dir /dev/shm/kimi_k3_13layer_mt
  python tests/utils/kimi_k3_layer_parity.py compare        --num_layers 13

What gets captured, per decoder layer:
  * prefix_sum entering / leaving the layer (the AttnRes "running intra-block delta")
  * block_residual leaving the layer
And per MoE layer:
  * router input (fp32, flattened [tokens, hidden])
  * router logits (fp32, [tokens, num_experts])
  * selected top-k expert indices and their weights

Both implementations carry (prefix_sum, block_residual) with identical semantics, so the
per-layer tensors are directly comparable.

Why this localizes the problem:
  * Layer 0 is DENSE + KDA -- no routing at all. Drift there is a pure numerics floor.
  * Smooth drift growth across layers => bfloat16 accumulation (benign).
  * A jump at one layer => a bug in that layer's component.
  * Routing flips are reported alongside drift, since one flipped expert out of top-16
    perturbs that token's MLP output by roughly 1/16 and is a discrete, amplifying event.

The torch_trace stage re-loads the model independently of the oracle stage, then asserts the
reproduced logits match `logits_pt.npy` bit-for-bit, which proves the duplicated load path is
faithful to the validated oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # pylint: disable=wrong-import-position

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "src")):
  if _p not in sys.path:
    sys.path.insert(0, _p)

# Reuse the exact helpers the validated harness uses, so config/tokens/layer-spec cannot drift.
from tests.utils.kimi_k3_real_ckpt import (  # pylint: disable=wrong-import-position
    BATCH,
    LM_PREFIX,
    SEQ,
    _hf_linear_attn_config,
    _load_index,
    _log,
    _peak_rss_gb,
    _tokens,
    _wanted_lm_key,
    get_full_attn_layers_0idx,
)


def _trace_path(work: str, side: str) -> str:
  return os.path.join(work, f"trace_{side}.npz")


# -----------------------------------------------------------------------------
# stage: torch_trace
# -----------------------------------------------------------------------------
def stage_torch_trace(work: str, num_layers: int, dtype: str) -> None:
  """Runs PyTorch layer-by-layer forward pass and saves intermediate traces."""
  import torch  # pylint: disable=import-outside-toplevel
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel

  from maxtext.checkpoint_conversion.utils import mxfp4  # pylint: disable=import-outside-toplevel
  from tests.utils.kimi_k3_parity_utils import load_hf_reference  # pylint: disable=import-outside-toplevel

  hf_dir = os.path.join(work, "hf")
  torch.set_grad_enabled(False)
  target_pt_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

  # --- identical patching to the oracle stage --------------------------------------------
  hf_config_mod, hf_model_mod = load_hf_reference(patch_kda=True)

  if hasattr(hf_model_mod, "FusedRMSNormGated"):

    def _patched_fused_forward(self, x, gate):
      x_f = x.float()
      variance = x_f.pow(2).mean(-1, keepdim=True)
      out_dtype = self.weight.dtype if getattr(self, "weight", None) is not None else x.dtype
      normed = (x_f * torch.rsqrt(variance + self.variance_epsilon) * self.weight.float()).to(out_dtype)
      if self.activation == "sigmoid":
        normed = (normed.float() * torch.sigmoid(gate.float())).to(out_dtype)
      return normed

    hf_model_mod.FusedRMSNormGated.forward = _patched_fused_forward

  with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as f:
    text_cfg = json.load(f)["text_config"]
  for k in ("quantization_config", "dtype", "torch_dtype", "auto_map", "_name_or_path", "transformers_version"):
    text_cfg.pop(k, None)
  full_attn_layers = get_full_attn_layers_0idx(num_layers)
  text_cfg.update(
      num_hidden_layers=num_layers,
      linear_attn_config=_hf_linear_attn_config(num_layers, full_attn_layers),
      use_cache=False,
      _attn_implementation="eager",
  )
  cfg = hf_config_mod.KimiLinearConfig(**text_cfg)

  with torch.device("meta"):
    model = hf_model_mod.KimiLinearForCausalLM(cfg)
  model.eval()
  expected = set(model.state_dict().keys())

  weight_map = _load_index(hf_dir)
  wanted = [k for k in weight_map if _wanted_lm_key(k, num_layers=num_layers)]
  by_shard: dict[str, list[str]] = {}
  for k in wanted:
    by_shard.setdefault(weight_map[k], []).append(k)

  loaded: set[str] = set()
  for shard, keys in sorted(by_shard.items()):
    t0 = time.time()
    partial: dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
      for key in keys:
        if mxfp4.is_mxfp4_sidecar_key(key) and key.endswith(".weight_scale"):
          continue
        if key.endswith(".weight_packed"):
          scale_key = key[: -len("weight_packed")] + "weight_scale"
          packed = f.get_tensor(key).contiguous().view(torch.uint8).numpy()
          scale = f.get_tensor(scale_key).contiguous().view(torch.uint8).numpy()
          dense = mxfp4.dequantize_mxfp4_packed(packed, scale, dtype=np.float32)
          partial[key[len(LM_PREFIX) : -len("_packed")]] = torch.from_numpy(dense).to(target_pt_dtype)
        else:
          t = f.get_tensor(key).to(target_pt_dtype)
          if key.endswith(".self_attn.A_log"):
            n = cfg.linear_attn_config["num_heads"]
            if t.shape[0] != n:
              if t.shape[0] < n or bool((t[n:] != 0).any()):
                raise RuntimeError(f"{key}: shape {tuple(t.shape)} is not zero-padded num_heads={n}")
              t = t[:n].clone()
          partial[key[len(LM_PREFIX) :]] = t
    model.load_state_dict(partial, strict=False, assign=True)
    loaded |= set(partial)
    _log(f"loaded {len(partial)} tensors from {shard} in {time.time() - t0:.0f}s (peak RSS {_peak_rss_gb():.0f} GB)")
    del partial

  missing = expected - loaded
  if missing:
    raise RuntimeError(f"model params not covered by checkpoint: {sorted(missing)[:20]}")
  _log(f"torch model ready ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

  # --- hooks -----------------------------------------------------------------------------
  layer_cap: dict[int, dict[str, np.ndarray]] = {}
  router_cap: dict[int, dict[str, np.ndarray]] = {}

  def _np(t):
    return t.detach().float().cpu().numpy()

  def _mk_layer_pre_hook(idx):
    def hook(_module, args):
      layer_cap.setdefault(idx, {})["prefix_in"] = _np(args[0])

    return hook

  def _mk_layer_post_hook(idx):
    def hook(_module, _args, output):
      d = layer_cap.setdefault(idx, {})
      d["prefix_out"] = _np(output[0])
      if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        d["block_residual_out"] = _np(output[1])

    return hook

  def _mk_gate_hook(idx):
    def hook(module, args, output):
      x = args[0]
      x_flat = x.reshape(-1, x.shape[-1])
      logits = torch.nn.functional.linear(x_flat.float(), module.weight.float(), None)  # pylint: disable=not-callable
      router_cap[idx] = {
          "router_in": _np(x_flat),
          "logits": _np(logits),
          "topk_idx": output[0].detach().cpu().numpy().astype(np.int32),
          "topk_weight": _np(output[1]),
      }

    return hook

  handles = []
  for i, layer in enumerate(model.model.layers):
    handles.append(layer.register_forward_pre_hook(_mk_layer_pre_hook(i)))
    handles.append(layer.register_forward_hook(_mk_layer_post_hook(i)))
    gate = getattr(getattr(layer, "block_sparse_moe", None), "gate", None)
    if gate is not None:
      handles.append(gate.register_forward_hook(_mk_gate_hook(i)))
  _log(f"registered hooks on {num_layers} layers, {len(router_cap)} routers pending")

  # --- forward ---------------------------------------------------------------------------
  toks = _tokens(work)
  t0 = time.time()
  logits = model(input_ids=torch.from_numpy(toks).long()).logits.float().numpy()
  _log(f"torch forward in {time.time() - t0:.0f}s; captured {len(layer_cap)} layers, {len(router_cap)} routers")
  for h in handles:
    h.remove()

  # --- fidelity check against the validated oracle ---------------------------------------
  oracle_path = os.path.join(work, "logits_pt.npy")
  if os.path.exists(oracle_path):
    ref = np.load(oracle_path)
    if ref.shape != logits.shape:
      raise RuntimeError(f"oracle shape {ref.shape} != trace shape {logits.shape}")
    d = float(np.abs(ref - logits).max())
    if d != 0.0:
      raise RuntimeError(f"trace run did NOT reproduce the oracle logits (max|diff| {d:.3e}); load path diverged")
    _log("fidelity check PASSED: trace logits bit-identical to logits_pt.npy")
  else:
    _log("WARNING: no logits_pt.npy to check against")

  _save_trace(_trace_path(work, "torch"), layer_cap, router_cap, logits)
  _log(f"torch_trace done (peak RSS {_peak_rss_gb():.0f} GB)")


# -----------------------------------------------------------------------------
# stage: maxtext_trace
# -----------------------------------------------------------------------------
def stage_maxtext_trace(work: str, mt_dir: str, num_layers: int, dtype: str) -> None:
  """Runs MaxText layer-by-layer forward pass and saves intermediate traces."""
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from flax import nnx  # pylint: disable=import-outside-toplevel

  from maxtext.common import checkpointing  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.layers.latent_moe import KimiMoERouter  # pylint: disable=import-outside-toplevel
  from maxtext.models.kimi_k3 import KimiK3DecoderLayer  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  from tests.utils.kimi_k3_parity_utils import positions_and_segments  # pylint: disable=import-outside-toplevel

  items = os.path.join(mt_dir, "0", "items")
  full_attn_layers = get_full_attn_layers_0idx(num_layers)

  # Layer bodies are normally wrapped in jax.checkpoint (remat), which makes every intermediate
  # a tracer and defeats value capture. remat is a recompute-for-memory tradeoff for autodiff and
  # is mathematically inert in a forward-only pass, so disabling it is numerically safe -- and we
  # verify that below by diffing these logits against the saved logits_mt.npy from the remat run.
  # Reuse the harness's own override dict so the rest of the config cannot drift from it.
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.kimi_k3_real_ckpt import _maxtext_overrides  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  overrides = dict(_maxtext_overrides(num_layers, full_attn_layers, dtype=dtype))
  overrides["remat_policy"] = "none"
  cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **overrides)
  _log("config built with remat_policy=none (layers run eagerly so captures are concrete)")

  t0 = time.time()
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(cfg, model_mode=MODEL_MODE_TRAIN)
  graphdef, params_abs, rest_abs = nnx.split(abstract_model, nnx.Param, ...)
  _log(f"abstract model built in {time.time() - t0:.0f}s")

  t0 = time.time()
  params = checkpointing.load_params_from_path(
      items,
      params_abs,
      cfg.checkpoint_storage_concurrent_gb,
      cfg.checkpoint_storage_use_ocdbt,
      cfg.checkpoint_storage_use_zarr3,
  )
  _log(f"restored params in {(time.time() - t0) / 60:.1f} min (peak RSS {_peak_rss_gb():.0f} GB)")

  def _materialize(a):
    if isinstance(a, jax.ShapeDtypeStruct):
      if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
        return jnp.broadcast_to(jax.random.key(0), a.shape)
      return jnp.zeros(a.shape, a.dtype)
    return a

  rest = jax.tree.map(_materialize, rest_abs)
  model = nnx.merge(graphdef, params, rest)

  # --- instrumentation: wrap, never reimplement -------------------------------------------
  layer_cap: dict[int, dict[str, np.ndarray]] = {}
  router_cap: dict[int, dict[str, np.ndarray]] = {}
  skipped = {"n": 0}

  def _sn(x):
    """numpy-ify, returning None for anything still abstract (tracer under jit/remat)."""
    if x is None:
      return None
    try:
      return np.asarray(jax.device_get(x), np.float32)
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
      skipped["n"] += 1
      return None

  def _put(d, k, v):
    v = _sn(v)
    if v is not None:
      d[k] = v

  orig_router_call = KimiMoERouter.__call__
  orig_layer_call = KimiK3DecoderLayer.__call__

  def traced_router_call(self, hidden_states):
    out = orig_router_call(self, hidden_states)
    topk_idx, topk_weight = out
    idx = getattr(self, "_trace_layer_idx", None)
    if idx is not None:
      d: dict[str, np.ndarray] = {}
      try:
        x_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).astype(jnp.float32)
        logits = jnp.matmul(x_flat, self.kernel.value.astype(jnp.float32))
        _put(d, "router_in", x_flat)
        _put(d, "logits", logits)
        ti = _sn(topk_idx)
        tw = _sn(topk_weight)
        if ti is not None:
          d["topk_idx"] = ti.astype(np.int32).reshape(-1, self.top_k)
        if tw is not None:
          d["topk_weight"] = tw.reshape(-1, self.top_k)
      except Exception:  # pylint: disable=broad-except
        skipped["n"] += 1
      if d:
        router_cap[idx] = d
    return out

  def traced_layer_call(self, inputs, *args, **kwargs):
    idx = getattr(self, "layer_idx", None)
    if idx is not None:
      _put(layer_cap.setdefault(idx, {}), "prefix_in", inputs)
    out = orig_layer_call(self, inputs, *args, **kwargs)
    if idx is not None:
      d = layer_cap.setdefault(idx, {})
      _put(d, "prefix_out", out[0])
      if len(out) > 1:
        _put(d, "block_residual_out", out[1])
    return out

  KimiMoERouter.__call__ = traced_router_call
  KimiK3DecoderLayer.__call__ = traced_layer_call

  tagged = 0
  for lyr in range(num_layers):
    layer = getattr(model.decoder, f"layers_{lyr}", None)
    gate = getattr(getattr(getattr(layer, "mlp", None), "routed_experts", None), "gate", None)
    if gate is not None:
      gate._trace_layer_idx = lyr  # pylint: disable=protected-access
      tagged += 1
  _log(f"instrumented {num_layers} layers, tagged {tagged} routers")

  try:
    toks = _tokens(work)
    positions, _ = positions_and_segments(BATCH, SEQ)
    t0 = time.time()
    logits = np.asarray(
        model(jnp.asarray(toks), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False),
        dtype=np.float32,
    )
    _log(f"maxtext forward in {time.time() - t0:.0f}s; captured {len(layer_cap)} layers, {len(router_cap)} routers")
    if skipped["n"]:
      _log(f"WARNING: {skipped['n']} tensors were still abstract and could not be captured")

  finally:
    KimiMoERouter.__call__ = orig_router_call
    KimiK3DecoderLayer.__call__ = orig_layer_call

  prev = os.path.join(work, "logits_mt.npy")
  if os.path.exists(prev):
    d = float(np.abs(np.load(prev) - logits).max())
    _log(f"reproducibility vs previous logits_mt.npy: max|diff| {d:.3e}")

  # Cross-check: run MaxText's router on the *PyTorch* router input, isolating the router
  # from any upstream hidden-state drift.
  torch_trace = _trace_path(work, "torch")
  if os.path.exists(torch_trace):
    tz = np.load(torch_trace)
    xcheck = {}
    for lyr in range(num_layers):
      key = f"router{lyr}_router_in"
      if key not in tz:
        continue
      layer = getattr(model.decoder, f"layers_{lyr}", None)
      gate = getattr(getattr(getattr(layer, "mlp", None), "routed_experts", None), "gate", None)
      if gate is None:
        continue
      x_pt = jnp.asarray(tz[key], jnp.float32)
      idx_mt, _ = orig_router_call(gate, x_pt)
      xcheck[f"xcheck{lyr}_topk_idx"] = np.asarray(idx_mt, np.int32).reshape(-1, gate.top_k)
    np.savez_compressed(os.path.join(work, "trace_xcheck.npz"), **xcheck)
    _log(f"cross-check saved for {len(xcheck)} routers (MaxText router fed PyTorch router input)")

  _save_trace(_trace_path(work, "maxtext"), layer_cap, router_cap, logits)
  _log(f"maxtext_trace done (peak RSS {_peak_rss_gb():.0f} GB)")


# -----------------------------------------------------------------------------
def _save_trace(path, layer_cap, router_cap, logits) -> None:
  out = {"logits": logits.astype(np.float32)}
  for i, d in layer_cap.items():
    for k, v in d.items():
      out[f"layer{i}_{k}"] = v.astype(np.float32)
  for i, d in router_cap.items():
    for k, v in d.items():
      out[f"router{i}_{k}"] = v
  np.savez_compressed(path, **out)
  _log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(out)} arrays)")


# -----------------------------------------------------------------------------
# stage: compare
# -----------------------------------------------------------------------------
def _rel(a, b) -> float:
  n = np.linalg.norm(b)
  return float(np.linalg.norm(a - b) / n) if n > 0 else float("nan")


def stage_compare(work: str, num_layers: int) -> None:
  """Compares PyTorch and MaxText intermediate layer traces and prints metrics."""
  pt = np.load(_trace_path(work, "torch"))
  mt = np.load(_trace_path(work, "maxtext"))

  print("\n" + "=" * 104)
  print("PER-LAYER HIDDEN-STATE DRIFT  (prefix_sum; rel = ||mt-pt|| / ||pt||)")
  print("=" * 104)
  print(f"{'layer':>5} {'kind':>10} {'rel(in)':>12} {'rel(out)':>12} {'growth':>10} {'max|d|out':>12} {'cos(out)':>10}")
  full_attn = set(get_full_attn_layers_0idx(num_layers))
  prev_out = None
  for i in range(num_layers):
    ki, ko = f"layer{i}_prefix_in", f"layer{i}_prefix_out"
    if ki not in pt or ki not in mt:
      continue
    ri, ro = _rel(mt[ki], pt[ki]), _rel(mt[ko], pt[ko])
    a, b = mt[ko].ravel(), pt[ko].ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    kind = ("MLA" if i in full_attn else "KDA") + ("/dense" if i == 0 else "/moe")
    growth = "" if prev_out is None or prev_out == 0 else f"{ro / prev_out:8.2f}x"
    print(
        f"{i:>5} {kind:>10} {ri:12.3e} {ro:12.3e} {growth:>10} "
        f"{float(np.abs(mt[ko] - pt[ko]).max()):12.3e} {cos:10.6f}"
    )
    prev_out = ro

  print("\n" + "=" * 104)
  print("MoE ROUTING AGREEMENT  (top-16 of 896 experts, 16 tokens)")
  print("=" * 104)
  print(
      f"{'layer':>5} {'rel(router_in)':>15} {'rel(logits)':>13} {'setmatch':>10} "
      f"{'exact/16':>9} {'flips':>7} {'xcheck':>16}"
  )
  xpath = os.path.join(work, "trace_xcheck.npz")
  xz = np.load(xpath) if os.path.exists(xpath) else None
  tot_flips = tot_slots = 0
  for i in range(num_layers):
    k = f"router{i}_topk_idx"
    if k not in pt or k not in mt:
      continue
    ip, im = pt[k], mt[k]
    set_match = sum(set(ip[t]) == set(im[t]) for t in range(ip.shape[0]))
    flips = sum(len(set(ip[t]) - set(im[t])) for t in range(ip.shape[0]))
    tot_flips += flips
    tot_slots += ip.size
    r_in = _rel(mt[f"router{i}_router_in"], pt[f"router{i}_router_in"])
    rlg = _rel(mt[f"router{i}_logits"], pt[f"router{i}_logits"])
    xs = "n/a"
    if xz is not None and f"xcheck{i}_topk_idx" in xz:
      ix = xz[f"xcheck{i}_topk_idx"]
      same = sum(set(ix[t]) == set(ip[t]) for t in range(ip.shape[0]))
      xs = f"{same}/{ip.shape[0]} identical"
    print(f"{i:>5} {r_in:15.3e} {rlg:13.3e} {set_match:>4}/{ip.shape[0]:<5} {'':>9} {flips:>7} {xs:>16}")
  if tot_slots:
    print(f"\ntotal expert-slot flips: {tot_flips}/{tot_slots} ({100.0 * tot_flips / tot_slots:.2f}%)")

  print("\n" + "=" * 104)
  print("FINAL LOGITS")
  print("=" * 104)
  lp, lm = pt["logits"], mt["logits"]
  print(f"  rel err        {_rel(lm, lp):.4e}")
  print(f"  max|diff|      {float(np.abs(lm - lp).max()):.4e}")
  t1p, t1m = lp[0].argmax(-1), lm[0].argmax(-1)
  print(f"  top-1 agree    {int((t1p == t1m).sum())}/{lp.shape[1]}")

  print("\nINTERPRETATION")
  print("  layer 0 is dense+KDA (no routing): its rel(out) is the pure bfloat16 noise floor.")
  print("  smooth growth across layers => benign accumulation; a jump => bug in that layer.")
  print("  xcheck 'identical' means MaxText's router reproduces PyTorch's expert choice when")
  print("  given PyTorch's own input -- i.e. routing flips are caused by upstream drift only.")
  print()


# -----------------------------------------------------------------------------
def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("stage", choices=["torch_trace", "maxtext_trace", "compare"])
  p.add_argument("--num_layers", type=int, default=13)
  p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
  p.add_argument("--work", default=None)
  p.add_argument("--mt_dir", default=None)
  a = p.parse_args()

  work = a.work if a.work is not None else os.path.expanduser(f"~/kimi_k3_{a.num_layers}layer")
  mt_dir = a.mt_dir if a.mt_dir is not None else f"/dev/shm/kimi_k3_{a.num_layers}layer_mt"
  os.makedirs(work, exist_ok=True)
  _log(f"stage={a.stage} num_layers={a.num_layers} dtype={a.dtype} work={work}")

  if a.stage == "torch_trace":
    stage_torch_trace(work, a.num_layers, a.dtype)
  elif a.stage == "maxtext_trace":
    stage_maxtext_trace(work, mt_dir, a.num_layers, a.dtype)
  else:
    stage_compare(work, a.num_layers)


if __name__ == "__main__":
  main()
