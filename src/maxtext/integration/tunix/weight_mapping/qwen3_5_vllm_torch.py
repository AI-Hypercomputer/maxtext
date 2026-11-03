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

"""Trainer-side weight layout for Raiden sync into a PyTorch vLLM sampler (Qwen3.5 MoE).

Raiden/tpu-sync pairs trainer and sampler tensors by *name* and moves bytes with
N-D slice resharding only (no transposes, fusion or dtype casts). A
vllm-torchtpu sampler (``Qwen3_5MoeForCausalLM`` on TPU) stores its weights as
vLLM's fused, per-segment tensor-parallel parameters, mostly transposed to
``[n_in, n_out]`` by vllm-torchtpu's unquantized method, and publishes each
rank-local block as a ``tp``-sharded slice of the **rank-major concatenation**
of all ranks' blocks (see ``vllm_torchtpu/rl/raiden_worker_sync.py``).

This module turns the trainer's scanned MaxText ``qwen3_5`` state into exactly
those global tensors, keyed by the vLLM parameter names (``model.layers.N....``,
``model.embed_tokens.weight``, ``lm_head.weight``, ``model.norm.weight``), so a
pure slice transfer lands the bytes in place. Shapes below are for
Qwen3.5-35B-A3B (H=2048, 16 q heads, 2 kv heads, head_dim 256, GDN 16 key /
32 value heads x 128, 256 experts x I=512) but every size is taken from the
config / arrays.

Per-rank local layout of the sampler (``tp`` = sampler tensor parallelism,
``r`` = rank) and the corresponding global tensor produced here:

* ``mlp.experts.w13_weight``  local ``[E/tp, H, 2I]`` -> global ``[E, H, 2I]``
  (= MaxText prefused ``wi``, experts expert-parallel on dim 0).
* ``mlp.experts.w2_weight``   local ``[E/tp, I, H]``  -> ``[E, I, H]`` (= ``wo``).
* ``self_attn.qkv_proj.weight`` stored ``[H, (Q+K+V)/tp]`` with columns
  ``[q_r | k_r | v_r]`` -> global ``[H, Q+K+V]`` = concat over ranks of
  ``[q_r | k_r | v_r]``; ``q_r`` are heads ``[r*16/tp, (r+1)*16/tp)`` x
  ``[q(256) | gate(256)]``.
* ``self_attn.o_proj.weight``  stored ``[Q/tp, H]`` -> ``[Q, H]`` (= ``out.kernel``).
* ``linear_attn.in_proj_qkvz.weight`` kept in vLLM layout ``[(2K+2V)/tp, H]``
  rows ``[q_r; k_r; v_r; z_r]`` -> global rank-major ``[2K+2V, H]``.
* ``linear_attn.in_proj_ba.weight`` stored ``[H, 2*Hv/tp]`` cols ``[b_r | a_r]``
  -> ``[H, 2*Hv]`` rank-major.
* ``linear_attn.conv1d.weight`` ``[(2K+V)/tp, 1, k]`` channels ``[q_r; k_r; v_r]``
  -> ``[2K+V, 1, k]`` rank-major.
* ``linear_attn.out_proj.weight`` stored ``[V/tp, H]`` -> ``[V, H]`` (= ``out_proj.kernel``).
* ``linear_attn.A_log`` (float32), ``dt_bias``: ``[Hv/tp]`` -> ``[Hv]``.
* ``mlp.shared_expert.gate_up_proj.weight`` stored ``[H, 2I/tp]`` cols
  ``[gate_r | up_r]`` -> ``[H, 2I]`` rank-major.
* ``mlp.shared_expert.down_proj.weight`` stored ``[I/tp, H]`` -> ``[I, H]`` (= ``wo.kernel``).
* ``mlp.gate.weight`` ``[E, H]``, ``mlp.shared_expert_gate.weight`` ``[1, H]``, all norm
  scales: replicated, vLLM (HF) layout.
* ``model.embed_tokens.weight`` / ``lm_head.weight`` ``[V/tp, H]`` -> ``[V, H]``.

Dtypes follow the sampler: bf16 everywhere except ``A_log`` (float32).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

_LAYER_SLOT_RE = re.compile(r"^layer_(\d+)$")


def _pure_dict(state: Any) -> dict:
  if hasattr(state, "to_pure_dict"):
    return state.to_pure_dict()
  if hasattr(state, "to_dict"):
    return state.to_dict()
  if isinstance(state, dict):
    return state
  raise TypeError(f"unsupported state type {type(state)}")


def _strip_base(flat: dict[tuple, Any]) -> dict[str, Any]:
  out = {}
  for key, value in flat.items():
    key = tuple(str(k) for k in key)
    if key and key[0] in ("base", "params"):
      key = key[1:]
    if key and key[-1] == "value":
      key = key[:-1]
    out[".".join(key)] = getattr(value, "value", value)
  return out


def _kv_heads_for_rank(num_kv_heads: int, tp: int, rank: int) -> list[int]:
  """vLLM QKVParallelLinear kv-head assignment (replicates when tp > kv heads)."""
  if num_kv_heads >= tp:
    per = num_kv_heads // tp
    return list(range(rank * per, (rank + 1) * per))
  rep = tp // num_kv_heads
  return [rank // rep]


class Qwen35VllmTorchLayout:
  """Builds the vLLM-torch global tensors from a scanned MaxText qwen3_5 state."""

  def __init__(self, config: Any, mesh: jax.sharding.Mesh, sampler_tp: int):
    self.config = config
    self.mesh = mesh
    self.tp = int(sampler_tp)
    if self.tp < 1:
      raise ValueError(f"sampler_tp must be >= 1, got {sampler_tp}")
    self.num_layers = int(config.num_decoder_layers)
    self.cycle = int(getattr(config, "inhomogeneous_layer_cycle_interval", 1) or 1)
    self.scan_axis = int(getattr(config, "param_scan_axis", 1))
    self.scan_layers = bool(getattr(config, "scan_layers", False))
    self.emb = int(config.emb_dim)
    self.num_q_heads = int(config.num_query_heads)
    self.head_dim = int(config.head_dim)
    self.moe_mlp_dim = int(getattr(config, "base_moe_mlp_dim", 0) or config.moe_mlp_dim)
    self.gdn_key_heads = int(config.gdn_num_key_heads)
    self.gdn_value_heads = int(config.gdn_num_value_heads)
    self.gdn_key_head_dim = int(config.gdn_key_head_dim)
    self.gdn_value_head_dim = int(config.gdn_value_head_dim)
    if self.num_q_heads % self.tp:
      raise ValueError(f"num_query_heads={self.num_q_heads} not divisible by sampler tp={self.tp}")
    if self.gdn_key_heads % self.tp or self.gdn_value_heads % self.tp:
      raise ValueError("GDN heads not divisible by sampler tp")
    big = [a for a in mesh.axis_names if mesh.shape[a] > 1]
    self._shard_axes: list[tuple[str, ...]] = []
    if big:
      self._shard_axes.append(tuple(big))
      self._shard_axes.extend((a,) for a in big)
    self._jit_cache: dict[str, Callable] = {}

  # ---------------------------------------------------------------- sharding
  def _out_sharding(self, shape: tuple[int, ...]) -> NamedSharding:
    for axes in self._shard_axes:
      n = math.prod(self.mesh.shape[a] for a in axes)
      for dim, size in enumerate(shape):
        if size % n == 0 and size >= n:
          spec = [None] * len(shape)
          spec[dim] = axes if len(axes) > 1 else axes[0]
          return NamedSharding(self.mesh, P(*spec))
    return NamedSharding(self.mesh, P())

  def _run(self, tag: str, fn: Callable[..., dict[str, jax.Array]], *args: jax.Array) -> dict[str, jax.Array]:
    """Runs `fn` jitted once per `tag`, with per-output shardings from a dry trace."""
    jitted = self._jit_cache.get(tag)
    if jitted is None:
      shapes = jax.eval_shape(fn, *args)
      out_shardings = {k: self._out_sharding(tuple(v.shape)) for k, v in shapes.items()}
      jitted = jax.jit(fn, out_shardings=out_shardings)
      self._jit_cache[tag] = jitted
    return jitted(*args)

  # ---------------------------------------------------------------- helpers
  def _unscan(self, x: jax.Array, repeat: jax.Array | int | None) -> jax.Array:
    if repeat is None or x.ndim <= self.scan_axis:
      return x
    return jax.lax.dynamic_index_in_dim(x, repeat, axis=self.scan_axis, keepdims=False)

  @staticmethod
  def _bf16(x: jax.Array) -> jax.Array:
    return x.astype(jnp.bfloat16)

  def _rank_major(self, blocks_per_rank: Callable[[int], list[jax.Array]], axis: int) -> jax.Array:
    """Concatenates, for r in ranks, the rank's segment blocks along `axis`."""
    parts = []
    for r in range(self.tp):
      parts.extend(blocks_per_rank(r))
    return jnp.concatenate(parts, axis=axis)

  # ------------------------------------------------------------ per-layer fns
  def _common(self, p: dict[str, jax.Array], n: str, out: dict[str, jax.Array]) -> None:
    H, tp = self.emb, self.tp
    I = self.moe_mlp_dim
    out[n + "input_layernorm.weight"] = self._bf16(p["input_layernorm.scale"])
    out[n + "post_attention_layernorm.weight"] = self._bf16(p["post_attention_layernorm.scale"])
    out[n + "mlp.gate.weight"] = self._bf16(p["mlp.routed_experts.gate.kernel"].T)  # [E, H]
    # Routed experts: vllm-torchtpu stores [E_local, H, 2I] (gate|up) and [E_local, I, H],
    # which is MaxText's prefused `wi` / `wo` layout. Drop any MoE padding.
    if "mlp.routed_experts.wi" in p:
      wi = p["mlp.routed_experts.wi"]
      half = wi.shape[-1] // 2
      if half != I:
        wi = jnp.concatenate([wi[..., :I], wi[..., half : half + I]], axis=-1)
    else:
      wi = jnp.concatenate([p["mlp.routed_experts.wi_0"], p["mlp.routed_experts.wi_1"]], axis=-1)
    wo = p["mlp.routed_experts.wo"]
    if wo.shape[1] != I:
      wo = wo[:, :I, :]
    out[n + "mlp.experts.w13_weight"] = self._bf16(wi)  # [E, H, 2I]
    out[n + "mlp.experts.w2_weight"] = self._bf16(wo)  # [E, I, H]
    # Shared expert: stored [H, 2I/tp] per rank with cols [gate_r | up_r].
    g = p["mlp.shared_expert.wi_0.kernel"]  # [H, I]
    u = p["mlp.shared_expert.wi_1.kernel"]
    si = g.shape[1] // tp
    out[n + "mlp.shared_expert.gate_up_proj.weight"] = self._bf16(
        self._rank_major(lambda r: [g[:, r * si : (r + 1) * si], u[:, r * si : (r + 1) * si]], axis=1)
    )  # [H, 2I]
    out[n + "mlp.shared_expert.down_proj.weight"] = self._bf16(p["mlp.shared_expert.wo.kernel"])  # [I, H]
    out[n + "mlp.shared_expert_gate.weight"] = self._bf16(p["mlp.shared_expert_gate.kernel"].T)  # [1, H]
    del H

  def _gdn(self, p: dict[str, jax.Array], n: str, out: dict[str, jax.Array]) -> None:
    H, tp = self.emb, self.tp
    Hk, Hv, dk, dv = self.gdn_key_heads, self.gdn_value_heads, self.gdn_key_head_dim, self.gdn_value_head_dim
    vpk = Hv // Hk  # value heads per key head
    kpr, vpr = Hk // tp, Hv // tp  # heads per rank
    a = "attention."
    # in_proj_qkvz: MaxText [H, Hk*(dk+dk+vpk*dv+vpk*dv)] grouped per key head as
    # [q_h | k_h | v_{vpk h..} | z_{vpk h..}]; vLLM rank r keeps rows [q_r; k_r; v_r; z_r]
    # in HF (head-major) order, NOT transposed.
    W = p[a + "in_proj_qkvz.kernel"]
    group = dk + dk + vpk * dv + vpk * dv
    t = W.T.reshape(Hk, group, H)
    q = t[:, :dk]
    k = t[:, dk : 2 * dk]
    v = t[:, 2 * dk : 2 * dk + vpk * dv]
    z = t[:, 2 * dk + vpk * dv :]

    def qkvz_blocks(r):
      sl = slice(r * kpr, (r + 1) * kpr)
      return [
          q[sl].reshape(kpr * dk, H),
          k[sl].reshape(kpr * dk, H),
          v[sl].reshape(kpr * vpk * dv, H),
          z[sl].reshape(kpr * vpk * dv, H),
      ]

    out[n + "linear_attn.in_proj_qkvz.weight"] = self._bf16(self._rank_major(qkvz_blocks, axis=0))
    # in_proj_ba: MaxText [H, Hk*(vpk+vpk)] per key head [b.. | a..]; vLLM stores
    # transposed [H, 2*Hv/tp] with cols [b_r | a_r].
    B = p[a + "in_proj_ba.kernel"].reshape(H, Hk, 2 * vpk)
    b = B[..., :vpk]
    aa = B[..., vpk:]

    def ba_blocks(r):
      sl = slice(r * kpr, (r + 1) * kpr)
      return [b[:, sl].reshape(H, kpr * vpk), aa[:, sl].reshape(H, kpr * vpk)]

    out[n + "linear_attn.in_proj_ba.weight"] = self._bf16(self._rank_major(ba_blocks, axis=1))
    # conv1d: MaxText kernel [k, 1, conv_dim] with channels [q; k; v] in HF order;
    # vLLM stores [conv_dim/tp, 1, k] with channels [q_r; k_r; v_r].
    cw = p[a + "conv1d.kernel"].transpose(2, 1, 0)  # [conv_dim, 1, k]
    K, V = Hk * dk, Hv * dv

    def conv_blocks(r):
      return [
          cw[r * kpr * dk : (r + 1) * kpr * dk],
          cw[K + r * kpr * dk : K + (r + 1) * kpr * dk],
          cw[2 * K + r * vpr * dv : 2 * K + (r + 1) * vpr * dv],
      ]

    out[n + "linear_attn.conv1d.weight"] = self._bf16(self._rank_major(conv_blocks, axis=0))
    out[n + "linear_attn.A_log"] = p[a + "A_log"].astype(jnp.float32)  # [Hv], natural order
    out[n + "linear_attn.dt_bias"] = self._bf16(p[a + "dt_bias"])
    out[n + "linear_attn.norm.weight"] = self._bf16(p[a + "norm.rms_norm.scale"])
    out[n + "linear_attn.out_proj.weight"] = self._bf16(p[a + "out_proj.kernel"])  # [V, H] = stored^T concat
    del V

  def _attn(self, p: dict[str, jax.Array], n: str, out: dict[str, jax.Array]) -> None:
    H, tp, hd = self.emb, self.tp, self.head_dim
    a = "attention.attention."
    qk = p[a + "query.kernel"]  # [H, nq, hd*2] (q | gate per head)
    kk = p[a + "key.kernel"]  # [H, nkv', hd]
    vk = p[a + "value.kernel"]
    nq = qk.shape[1]
    # Undo trainer-side kv-head replication (consecutive repeats), if any.
    base_kv = int(getattr(self.config, "num_kv_heads", kk.shape[1]))
    nkv_t = kk.shape[1]
    if nkv_t > base_kv and nkv_t % base_kv == 0:
      rep = nkv_t // base_kv
      kk, vk = kk[:, ::rep], vk[:, ::rep]
    nkv = kk.shape[1]
    qpr = nq // tp

    def qkv_blocks(r):
      heads = _kv_heads_for_rank(nkv, tp, r)
      return [
          qk[:, r * qpr : (r + 1) * qpr].reshape(H, qpr * qk.shape[2]),
          kk[:, heads].reshape(H, len(heads) * hd),
          vk[:, heads].reshape(H, len(heads) * hd),
      ]

    out[n + "self_attn.qkv_proj.weight"] = self._bf16(self._rank_major(qkv_blocks, axis=1))  # [H, Q+K+V]
    out[n + "self_attn.o_proj.weight"] = self._bf16(p[a + "out.kernel"])  # [Q, H]
    out[n + "self_attn.q_norm.weight"] = self._bf16(p[a + "query_norm.scale"])
    out[n + "self_attn.k_norm.weight"] = self._bf16(p[a + "key_norm.scale"])

  def _top(self, p: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "model.embed_tokens.weight": self._bf16(p["token_embedder.embedding"]),  # [V, H]
        "model.norm.weight": self._bf16(p["decoder.decoder_norm.scale"]),
        "lm_head.weight": self._bf16(p["decoder.logits_dense.kernel"].T),  # [V, H]
    }

  # ----------------------------------------------------------------- driver
  def _is_attention_slot(self, slot_params: dict[str, jax.Array]) -> bool:
    return any(k.startswith("attention.attention.") for k in slot_params)

  def convert(self, state: Any) -> dict:
    """Returns a nested dict {model: {layers: {"<l>": {...}}}, lm_head: ...} of jax arrays."""
    flat = _strip_base(flatten_dict(_pure_dict(state)))
    top = {k: v for k, v in flat.items() if ".layers." not in k}
    layers = {k: v for k, v in flat.items() if ".layers." in k}
    out: dict[str, jax.Array] = {}
    out.update(self._run("top", self._top, top))

    # Group per scanned slot ("decoder.layers.layer_<s>.<rest>", cycle > 1),
    # per scanned stack ("decoder.layers.<rest>", cycle == 1) or per unscanned
    # layer ("decoder.layers_<n>.<rest>").
    _LAYERS_N_RE = re.compile(r"^layers_(\d+)$")
    groups: dict[tuple[str, int, int | None], dict[str, jax.Array]] = {}
    for key, arr in layers.items():
      parts = key.split(".")
      idx = next(i for i, c in enumerate(parts) if c == "layers" or _LAYERS_N_RE.match(c))
      head, rest = parts[idx], parts[idx + 1 :]
      m_n = _LAYERS_N_RE.match(head)
      if m_n:  # unscanned
        groups.setdefault(("layer", int(m_n.group(1)), None), {})[".".join(rest)] = arr
        continue
      m_s = _LAYER_SLOT_RE.match(rest[0]) if rest else None
      if m_s:
        groups.setdefault(("slot", int(m_s.group(1)), None), {})[".".join(rest[1:])] = arr
      else:
        groups.setdefault(("slot", 0, None), {})[".".join(rest)] = arr

    repeats = self.num_layers // self.cycle if self.scan_layers else 1
    for (kind, index, _), params in sorted(groups.items()):
      is_attn = self._is_attention_slot(params)
      names = sorted(params)
      scanned = kind == "slot" and self.scan_layers

      def layer_fn(*arrays, _names=names, _is_attn=is_attn, _scanned=scanned):
        repeat = arrays[-1] if _scanned else None
        arrs = arrays[:-1] if _scanned else arrays
        p = {k: self._unscan(v, repeat) for k, v in zip(_names, arrs)}
        o: dict[str, jax.Array] = {}
        self._common(p, "", o)
        (self._attn if _is_attn else self._gdn)(p, "", o)
        return o

      if kind == "layer":
        res = self._run(f"layer{index}", layer_fn, *[params[k] for k in names])
        for k, v in res.items():
          out[f"model.layers.{index}." + k] = v
        continue
      slot = index
      for repeat in range(repeats if scanned else 1):
        layer = repeat * self.cycle + slot
        args = [params[k] for k in names] + ([jnp.int32(repeat)] if scanned else [])
        # One trace per slot type: `repeat` is a traced scalar (dynamic slice).
        res = self._run(f"slot{slot}-attn{is_attn}", layer_fn, *args)
        for k, v in res.items():
          out[f"model.layers.{layer}." + k] = v

    total = sum(v.size * v.dtype.itemsize for v in out.values())
    logger.info(
        "qwen3_5 -> vllm_torch layout: %d tensors, %.1f GiB, sampler tp=%d", len(out), total / 2**30, self.tp
    )
    return unflatten_dict({tuple(k.split(".")): v for k, v in out.items()})


def convert_for_vllm_torch(state: Any, config: Any, mesh: jax.sharding.Mesh, sampler_tp: int) -> dict:
  """One-shot helper (constructs the layout object each call; jit caches live on it)."""
  return Qwen35VllmTorchLayout(config, mesh, sampler_tp).convert(state)
