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
on a synthetic, history-dependent next-token task: a delayed-copy sequence
``t[i] = t[i-delay]`` over random i.i.d. tokens. Because each target is
information-theoretically independent of the *current* token, no MLP acting
on the current token alone can solve it — loss only drops from
``log(vocab_size)`` toward ~0 if the KDA recurrent state actually carries
history through forward/backward. This makes the smoke test a genuine
correctness signal for the kernel rather than just an optimizer/compilation
check.

``delay`` must also exceed the short convolution's receptive field: the
``linear_conv_kernel_dim``-tap causal conv already feeds positions
``[j - kernel_dim + 1, j]`` into Q/K/V at position ``j``, so any target
within that window could be copied by the conv alone without using the KDA
state. The default ``delay=8`` sits outside the 4-tap window; the script
enforces ``delay > linear_conv_kernel_dim``.

This exists because the KDA layer is not yet wired into the maxtext decoder;
it validates the layer end to end without decoder integration.

Usage (on a TPU host):
  python scripts/dev/kda_e2e_smoke.py [--steps 600] [--batch 32] [--seq-len 128] [--delay 8]
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


def make_dataset(seed: int, num_seqs: int, seq_len: int, delay: int) -> np.ndarray:
  """Delayed-copy sequences over random tokens: t[i] = t[i-delay].

  Tokens before index ``delay`` are i.i.d. uniform; every later position
  repeats the token ``delay`` steps back. The target t[i+1] is independent
  of the current token t[i], so no memoryless (current-token-only) model can
  predict it — solving the task requires ``delay`` steps of recurrent history.
  """
  rng = np.random.default_rng(seed)
  total = seq_len + 1
  seqs = rng.integers(0, _VOCAB, size=(num_seqs, total), dtype=np.int32)
  for i in range(delay, total):
    seqs[:, i] = seqs[:, i - delay]
  return seqs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--batch", type=int, default=32)
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--num-layers", type=int, default=4)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--log-every", type=int, default=50)
  parser.add_argument("--delay", type=int, default=8, help="delayed-copy distance; larger = harder (more history needed)")
  args = parser.parse_args()

  if args.seq_len % 64 != 0:
    parser.error(f"--seq-len must be a multiple of the KDA chunk size (64), got {args.seq_len}")
  if args.delay < 1 or args.delay >= args.seq_len:
    parser.error(f"--delay must be in [1, seq_len-1], got {args.delay}")

  cfg = make_config()
  # The causal conv already exposes kernel_dim-1 past tokens to Q/K/V; the
  # task only proves recurrent-state carry when the target lies outside that
  # receptive field.
  if args.delay <= cfg.linear_conv_kernel_dim:
    parser.error(
        f"--delay must exceed the short-conv receptive field "
        f"(linear_conv_kernel_dim={cfg.linear_conv_kernel_dim}), got {args.delay}; "
        "otherwise the target can be copied by the convolution without KDA state"
    )
  devices = jax.devices()
  print(f"devices: {devices}")
  mesh = jax.sharding.Mesh(np.array(devices), ("x",))
  rngs = nnx.Rngs(0)
  with mesh:
    model = TinyKdaLM(cfg, mesh, args.num_layers, rngs=rngs)
  n_params = sum(v.size for v in jax.tree.leaves(nnx.state(model)) if isinstance(v, (jax.Array, np.ndarray)))
  print(f"model params: {n_params / 1e6:.1f}M")

  data = make_dataset(seed=42, num_seqs=4096, seq_len=args.seq_len, delay=args.delay)
  # Label position j predicts token j+1 = token j+1-delay; learnable only once
  # that token is inside the context, i.e. j >= delay-1. Earlier positions are
  # random (irreducible), so mask them out to keep the loss floor at 0.
  loss_mask = jnp.asarray(np.arange(args.seq_len) >= args.delay - 1, dtype=jnp.float32)[None, :]
  optimizer = nnx.Optimizer(model, optax.adamw(args.lr), wrt=nnx.Param)

  def _masked_ce(logits, labels):
    ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits.astype(jnp.float32), labels=labels)
    return (ce * loss_mask).sum() / loss_mask.sum() / labels.shape[0]

  @nnx.jit
  def train_step(model, optimizer, tokens):
    def loss_fn(model):
      logits = model(tokens[:, :-1])
      return _masked_ce(logits, tokens[:, 1:]), logits

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss

  @nnx.jit
  def eval_loss(model, tokens):
    logits = model(tokens[:, :-1])
    return _masked_ce(logits, tokens[:, 1:])

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
  print(f"\ntask: delayed copy, delay={args.delay} (history-dependent; a memoryless model cannot solve it)")
  print(f"initial loss: {init_loss:.4f} (theoretical ln({_VOCAB}) = {np.log(_VOCAB):.4f})")
  print(f"final (avg last 20): {final_loss:.4f}")
  ok = final_loss < 0.5 * init_loss and final_loss < 1.0
  print("PASS: loss decreased as expected (KDA e2e smoke OK)" if ok else "FAIL: loss did not decrease sufficiently")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
