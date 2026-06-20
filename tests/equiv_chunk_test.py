"""Standalone equivalence check: moe_n_chunks=1 vs 2 on the ring-of-experts path.

Validates the chunked MoE pipeline in RoutedMoE.sparse_matmul: with per-token
routing, the forward output and parameter grads must match n_chunks=1 within
bf16 tolerance (only the aggregate load-balance loss may drift). Runs on local
TPU (v4 x4) with EP=2, mirroring tests/unit/moe_test.py's ring test.

Run: python tests/equiv_chunk_test.py
"""
import jax
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def build_cfg(n_chunks):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name=f"equiv_chunk_{n_chunks}",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=2048,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      dtype="bfloat16",
      megablox=True,
      sparse_matmul=True,
      per_device_batch_size=4,
      ici_expert_parallelism=2,
      use_ring_of_experts=True,
      use_ragged_sort=True,
      max_target_length=128,
      moe_n_chunks=n_chunks,
  )


def build_model(cfg, mesh):
  return moe.get_routed_moe(
      name="MoeBlock",
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.mlp_dim,
      dtype=cfg.dtype,
  )


def loss_and_grad(model, variables, x):
  def loss_fn(params, xx):
    out, lb_loss, _ = model.apply({"params": params}, xx)
    loss = jnp.mean(out.astype(jnp.float32) ** 2)
    return loss, out

  (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(variables["params"], x)
  return loss, out, grads


def run(n_chunks, variables, x_in):
  cfg = build_cfg(n_chunks)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = build_model(cfg, mesh)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    if variables is None:
      x = jax.random.uniform(jax.random.PRNGKey(7), x_in, dtype=cfg.dtype)
      variables = model.init({"params": jax.random.PRNGKey(2345), "dropout": jax.random.PRNGKey(2345)}, x)
      return variables, x
    loss, out, grads = loss_and_grad(model, variables, x_in)
    return loss, out, grads


def main():
  cfg0 = build_cfg(1)
  dc = jax.device_count()
  shape = (int(cfg0.per_device_batch_size) * dc, cfg0.max_target_length, cfg0.base_emb_dim)
  variables, x = run(1, None, shape)

  loss1, out1, g1 = run(1, variables, x)
  loss2, out2, g2 = run(2, variables, x)

  out_diff = jnp.max(jnp.abs(out1.astype(jnp.float32) - out2.astype(jnp.float32)))
  print(f"loss n=1: {loss1:.6f}  n=2: {loss2:.6f}  |Δ|={abs(float(loss1)-float(loss2)):.3e}")
  print(f"max|Δoutput| = {float(out_diff):.3e}  (out shape {out1.shape})")

  ok_out = jnp.allclose(out1.astype(jnp.float32), out2.astype(jnp.float32), rtol=1e-2, atol=1e-2)
  l1, _ = jax.tree_util.tree_flatten(g1)
  l2, _ = jax.tree_util.tree_flatten(g2)
  gmax = max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) for a, b in zip(l1, l2))
  ok_grad = all(
      bool(jnp.allclose(a.astype(jnp.float32), b.astype(jnp.float32), rtol=1e-2, atol=1e-2)) for a, b in zip(l1, l2)
  )
  print(f"max|Δgrad| across {len(l1)} leaves = {gmax:.3e}")
  print(f"RESULT: output_match={bool(ok_out)}  grad_match={bool(ok_grad)}")
  assert ok_out, "FORWARD OUTPUT MISMATCH between n_chunks=1 and 2"
  assert ok_grad, "GRADIENT MISMATCH between n_chunks=1 and 2"
  print("EQUIVALENCE PASS")


if __name__ == "__main__":
  main()
