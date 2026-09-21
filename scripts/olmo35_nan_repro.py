"""Minimal repro for the two OLMo 3.5 NaN bugs seen on Ironwood.

Both block the per-device-batch lever, which is the main MFU knob:

  1. the fused tokamax KDA kernel NaNs at per_device_batch_size >= 2
  2. use_gmm_v2=True NaNs at per_device_batch_size = 1

This runs one forward+backward of the real OLMoE3/OLMo3.5 decoder at a shrunk
shape that keeps the family's actual head geometry (key_head_dim 128,
value_head_dim 256, the two dims the kernel is specialised on), so it fits on a
single small host while still exercising the same kernel path.

  PYTHONPATH=<worktree>/src python3 scripts/olmo35_nan_repro.py
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

# Same shrunk geometry the parity test uses: every family invariant holds
# (latent*2 == d_model, head_dim 128, n_layers % 8 == 0) and layer 7 is the
# full-attention layer, so one step covers dense, KDA and MoE blocks.
SHRUNK = [
    "base_emb_dim=512",
    "base_num_query_heads=2",
    "base_num_kv_heads=1",
    "head_dim=128",
    "base_num_decoder_layers=8",
    "base_mlp_dim=4096",
    "base_moe_mlp_dim=256",
    "moe_expert_input_dim=256",
    "num_experts=8",
    "num_experts_per_tok=2",
    "gdn_num_key_heads=2",
    "gdn_num_value_heads=2",
    "gdn_key_head_dim=128",
    "gdn_value_head_dim=256",
    "vocab_size=512",
    "emo_min_document_expert_pool=8",
    "emo_max_document_expert_pool=8",
    "emo_eval_document_expert_pool=8",
]


def build(pdb, seq, extra):
  """Build a shrunk OLMo 3.5 model and its mesh."""
  cfg = pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "model_name=olmo35-tiny",
          "override_model_config=True",
          "run_name=olmo35_nan_repro",
          "enable_checkpointing=False",
          "scan_layers=False",
          "skip_jax_distributed_system=True",
          f"per_device_batch_size={pdb}",
          f"max_target_length={seq}",
          "dtype=bfloat16",
          "weight_dtype=float32",
          "ici_fsdp_parallelism=-1",
          *SHRUNK,
          *extra,
      ]
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg))
  return cfg, mesh, model


def probe(label, pdb, seq, extra):
  """One fwd+bwd; report whether logits or any gradient went non-finite."""
  try:
    _, mesh, model = build(pdb, seq, extra)
    n = int(pdb * jax.device_count())
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 512, size=(n, seq)), jnp.int32)
    positions = jnp.arange(seq, dtype=jnp.int32)[None, :].repeat(n, 0)
    segments = jnp.ones((n, seq), jnp.int32)
    with mesh:
      params = model.init(
          {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1), "aqt": jax.random.PRNGKey(2)},
          tokens,
          positions,
          enable_dropout=False,
      )

      def loss_fn(p):
        out = model.apply(
            p,
            tokens,
            positions,
            decoder_segment_ids=segments,
            enable_dropout=False,
            rngs={"dropout": jax.random.PRNGKey(3), "aqt": jax.random.PRNGKey(4)},
        )
        logits = out[0] if isinstance(out, tuple) else out
        return jnp.mean(logits.astype(jnp.float32) ** 2), logits

      (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    finite_logits = bool(jnp.isfinite(logits).all())
    bad = [
        "/".join(str(getattr(k, "key", k)) for k in path)
        for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]
        if not bool(jnp.isfinite(g).all())
    ]
    status = "OK" if finite_logits and not bad else "NaN/Inf"
    print(f"  {label:<44}{status:>9}  loss={float(loss):.4e}  logits_finite={finite_logits}  bad_grads={len(bad)}")
    for b in bad[:4]:
      print(f"      first bad grad: {b}")
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  {label:<44}{'ERROR':>9}  {' '.join(str(exc).split())[:100]}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq", type=int, default=512)
  ap.add_argument("--pdbs", default="1,2,4")
  args = ap.parse_args()

  print(f"devices: {jax.device_count()} x {jax.devices()[0].device_kind}, seq={args.seq}")
  print("\nBug 1: fused tokamax KDA vs per-device batch")
  for pdb in [int(p) for p in args.pdbs.split(",")]:
    probe(f"pdb={pdb} use_tokamax_kda=True", pdb, args.seq, ["use_tokamax_kda=True"])
  for pdb in [int(p) for p in args.pdbs.split(",")]:
    probe(f"pdb={pdb} use_tokamax_kda=False (control)", pdb, args.seq, ["use_tokamax_kda=False"])

  print("\nBug 2: gmm_v2")
  for flags, label in (
      (["megablox=True", "sparse_matmul=True"], "megablox (control)"),
      (["sparse_matmul=True", "use_tokamax_gmm=True"], "tokamax gmm v1"),
      (["sparse_matmul=True", "use_tokamax_gmm=True", "use_gmm_v2=True"], "tokamax gmm v2"),
  ):
    probe(f"pdb=1 {label}", 1, args.seq, flags + ["use_tokamax_kda=False"])


if __name__ == "__main__":
  main()
