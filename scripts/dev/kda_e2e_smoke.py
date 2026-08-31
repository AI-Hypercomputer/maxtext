# Copyright 2026 Ant Group. All Rights Reserved.
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

"""Standalone KDA end-to-end smoke training.

Builds a tiny language model whose attention stack is the real maxtext
``KimiDeltaAttention`` layer (tokamax Pallas kernel underneath) and trains it
on a synthetic, fully learnable next-token task (random fixed permutation:
``t[i+1] = perm[t[i]]``). If the forward/backward/optimizer chain through the
KDA kernel is healthy, loss drops from ``log(vocab_size)`` toward ~0.

This exists because the KDA layer is not yet wired into the maxtext decoder;
it validates the layer end to end without decoder integration.

Usage (on a TPU host):
  python scripts/dev/kda_e2e_smoke.py [--steps 400] [--batch 32] [--seq-len 128]
"""

import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from maxtext.layers import attention_kda
from maxtext.layers.normalizations import RMSNorm

_VOCAB = 128


def make_config() -> SimpleNamespace:
  """Minimal config satisfying KimiDeltaAttention (mirrors the UT mock)."""
  return SimpleNamespace(
      base_emb_dim=256,
      base_num_query_heads=8,
      head_dim=64,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      attention_bias=False,
      shard_mode="auto",
      matmul_precision="default",
      normalization_layer_epsilon=1e-6,
      logical_axis_rules=[],
      # KDA-specific
      linear_conv_kernel_dim=4,
      use_qk_norm=True,
      use_kda_safe_gate=True,
      kda_lower_bound=-5.0,
      max_segments_per_seq=25,
      context_sharding="context",
  )


class KdaBlock(nnx.Module):
  """Pre-norm transformer block: RMSNorm -> KimiDeltaAttention -> MLP."""

  def __init__(self, cfg, mesh, layer_idx, *, rngs):
    self.attn_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.attn = attention_kda.KimiDeltaAttention(cfg, layer_idx=layer_idx, mesh=mesh, rngs=rngs)
    self.mlp_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    hidden = 4 * cfg.base_emb_dim
    self.wi = nnx.Linear(cfg.base_emb_dim, hidden, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)
    self.wo = nnx.Linear(hidden, cfg.base_emb_dim, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)

  def __call__(self, x):
    attn_out, _ = self.attn(self.attn_norm(x).astype(self.attn.config.dtype))
    x = x + attn_out.astype(x.dtype)
    h = nnx.gelu(self.wi(self.mlp_norm(x)))
    x = x + self.wo(h).astype(x.dtype)
    return x


class TinyKdaLM(nnx.Module):
  """Embed -> N x KdaBlock -> RMSNorm -> lm_head."""

  def __init__(self, cfg, mesh, num_layers, *, rngs):
    self.embed = nnx.Embed(_VOCAB, cfg.base_emb_dim, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)
    self.blocks = nnx.List([KdaBlock(cfg, mesh, i, rngs=rngs) for i in range(num_layers)])
    self.final_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.lm_head = nnx.Linear(cfg.base_emb_dim, _VOCAB, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)

  def __call__(self, tokens):
    x = self.embed(tokens)
    for block in self.blocks:
      x = block(x)
    logits = self.lm_head(self.final_norm(x))
    return logits


def make_dataset(seed: int, num_seqs: int, seq_len: int) -> np.ndarray:
  """Random-walk sequences over a fixed permutation: t[i+1] = perm[t[i]]."""
  rng = np.random.default_rng(seed)
  perm = rng.permutation(_VOCAB)
  start = rng.integers(0, _VOCAB, size=(num_seqs,))
  seqs = np.empty((num_seqs, seq_len + 1), dtype=np.int32)
  seqs[:, 0] = start
  for i in range(seq_len):
    seqs[:, i + 1] = perm[seqs[:, i]]
  return seqs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--batch", type=int, default=32)
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--num-layers", type=int, default=4)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--log-every", type=int, default=50)
  args = parser.parse_args()

  assert args.seq_len % 64 == 0, "seq_len must be a multiple of the KDA chunk size (64)"
  devices = jax.devices()
  print(f"devices: {devices}")
  mesh = jax.sharding.Mesh(np.array(devices), ("x",))

  cfg = make_config()
  rngs = nnx.Rngs(0)
  with mesh:
    model = TinyKdaLM(cfg, mesh, args.num_layers, rngs=rngs)
  n_params = sum(v.size for v in jax.tree.leaves(nnx.state(model)) if isinstance(v, (jax.Array, np.ndarray)))
  print(f"model params: {n_params / 1e6:.1f}M")

  data = make_dataset(seed=42, num_seqs=4096, seq_len=args.seq_len)
  optimizer = nnx.Optimizer(model, optax.adamw(args.lr), wrt=nnx.Param)

  @nnx.jit
  def train_step(model, optimizer, tokens):
    def loss_fn(model):
      logits = model(tokens[:, :-1])
      loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits.astype(jnp.float32), labels=tokens[:, 1:]
      ).mean()
      return loss, logits

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss

  @nnx.jit
  def eval_loss(model, tokens):
    logits = model(tokens[:, :-1])
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits.astype(jnp.float32), labels=tokens[:, 1:]).mean()

  perm_rng = np.random.default_rng(1)
  losses = []
  for step in range(args.steps):
    idx = perm_rng.integers(0, data.shape[0], size=args.batch)
    batch = jnp.asarray(data[idx])
    loss = train_step(model, optimizer, batch)
    loss_val = float(loss)
    if not np.isfinite(loss_val):
      raise SystemExit(f"FAIL: non-finite loss {loss_val} at step {step}")
    losses.append(loss_val)
    if step % args.log_every == 0 or step == args.steps - 1:
      print(f"step {step:5d}  train_loss {loss_val:.4f}")

  init_loss, final_loss = losses[0], np.mean(losses[-20:])
  print(f"\ninitial loss: {init_loss:.4f} (theoretical ln({_VOCAB}) = {np.log(_VOCAB):.4f})")
  print(f"final (avg last 20): {final_loss:.4f}")
  ok = final_loss < 0.5 * init_loss and final_loss < 1.0
  print("PASS: loss decreased as expected (KDA e2e smoke OK)" if ok else "FAIL: loss did not decrease sufficiently")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
