import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
from tests.utils.test_helpers import get_test_config_path


def _build_cfg(use_ragged_sort):
  effective_buffer_factor = 1.5 if use_ragged_sort else -1.0
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name=f"debug_nan_{use_ragged_sort}",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=2048,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      dtype="bfloat16",
      megablox=True,
      use_tokamax_gmm=False,
      use_gmm_v2=False,
      sparse_matmul=True,
      per_device_batch_size=4,
      ici_expert_parallelism=2,
      use_ring_of_experts=True,
      max_target_length=128,
      use_ragged_sort=use_ragged_sort,
      ragged_buffer_factor=effective_buffer_factor,
      log_config=False,
  )


def _build_model(cfg, mesh):
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


def get_loss_and_grads(model, variables, x):
  # Forward-only inspection with intermediates captured
  (out, lb_loss, _), mutables = model.apply(variables, x, capture_intermediates=True, mutable=["intermediates"])
  intermediates = mutables.get("intermediates", {})

  has_nan_fwd = jnp.any(jnp.isnan(out))
  max_fwd = jnp.max(jnp.abs(out))
  has_nan_lb = jnp.any(jnp.isnan(lb_loss)) if lb_loss is not None else False

  # Loss and Grad calculation
  def loss_fn(params, x_in):
    out, lb_loss, _ = model.apply({"params": params}, x_in)
    return jnp.mean(out.astype(jnp.float32) ** 2)

  loss, (params_grad, x_grad) = jax.value_and_grad(loss_fn, argnums=(0, 1))(variables["params"], x)
  return loss, params_grad, x_grad, has_nan_fwd, max_fwd, has_nan_lb, intermediates


# PyTree helpers
def tree_has_nan(tree):
  flat_tree, _ = jax.tree_util.tree_flatten(tree)
  return any(jnp.any(jnp.isnan(leaf)) for leaf in flat_tree)


def compare_pytrees_detailed(tree1, tree2, rtol, atol, tree_name="Tree"):
  # Flatten with paths so we can print exactly which node/layer fails
  flat1, _ = jax.tree_util.tree_flatten_with_path(tree1)
  flat2, _ = jax.tree_util.tree_flatten_with_path(tree2)

  if len(flat1) != len(flat2):
    print(f"  -> {tree_name} structure mismatch! Lengths: {len(flat1)} vs {len(flat2)}")
    return False, 0.0

  all_match = True
  max_diff = 0.0

  for (path1, t1), (path2, t2) in zip(flat1, flat2):
    path_str = jax.tree_util.keystr(path1)

    if t1.shape != t2.shape:
      print(f"  -> Shape mismatch at {path_str}: {t1.shape} vs {t2.shape}")
      all_match = False
      continue

    match = jnp.allclose(t1, t2, rtol=rtol, atol=atol)
    diff = jnp.max(jnp.abs(t1 - t2))

    if not match:
      print(f"  -> Mismatch at {path_str} | Max abs diff: {diff}")

    all_match = all_match and match
    max_diff = jnp.maximum(max_diff, diff)

  return bool(all_match), float(max_diff)


rng = jax.random.PRNGKey(2345)
rng_model, rng_hidden_states = jax.random.split(rng)
device_count = jax.device_count()

# Build configs
cfg_rs = _build_cfg(use_ragged_sort=True)
cfg_nors = _build_cfg(use_ragged_sort=False)

# Build shared inputs
hidden_states = jax.random.uniform(
    rng_hidden_states,
    (int(cfg_rs.per_device_batch_size) * device_count, cfg_rs.max_target_length, cfg_rs.base_emb_dim),
    dtype=cfg_rs.dtype,
)

# Setup mesh
devices_array = maxtext_utils.create_device_mesh(cfg_rs)
mesh = Mesh(devices_array, cfg_rs.mesh_axes)

model_rs = _build_model(cfg_rs, mesh)
model_nors = _build_model(cfg_nors, mesh)

jitted_get_loss_and_grads = jax.jit(get_loss_and_grads, static_argnums=(0,))

with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_rs.logical_axis_rules):
  variables_rs = model_rs.init({"params": rng_model, "dropout": rng_model}, hidden_states)
  variables_nors = model_nors.init({"params": rng_model, "dropout": rng_model}, hidden_states)

  # Run for ragged_sort = True
  print("\n--- Running with ragged_sort=True ---")
  loss_rs, params_grad_rs, x_grad_rs, nan_fwd_rs, max_fwd_rs, nan_lb_rs, int_rs = jitted_get_loss_and_grads(
      model_rs, variables_rs, hidden_states
  )
  print(f"Forward output has NaN: {nan_fwd_rs}")
  print(f"Forward output max: {max_fwd_rs}")
  print(f"Loss (no lb): {loss_rs}")
  print(f"x_grad has NaN: {jnp.any(jnp.isnan(x_grad_rs))}")
  print(f"params_grad has NaN: {tree_has_nan(params_grad_rs)}")

  # Run for ragged_sort = False
  print("\n--- Running with ragged_sort=False ---")
  loss_nors, params_grad_nors, x_grad_nors, nan_fwd_nors, max_fwd_nors, nan_lb_nors, int_nors = jitted_get_loss_and_grads(
      model_nors, variables_nors, hidden_states
  )
  print(f"Forward output has NaN: {nan_fwd_nors}")
  print(f"Forward output max: {max_fwd_nors}")
  print(f"Loss (no lb): {loss_nors}")
  print(f"x_grad has NaN: {jnp.any(jnp.isnan(x_grad_nors))}")
  print(f"params_grad has NaN: {tree_has_nan(params_grad_nors)}")

  print("\n--- Side-by-Side Comparison ---")
  rtol = 1e-3
  atol = 1e-4

  # Intermediates comparison
  print("Checking intermediate activations...")
  int_match, max_int_diff = compare_pytrees_detailed(int_rs, int_nors, rtol, atol, tree_name="Intermediates")
  print(f"Intermediates match: {int_match} (max abs diff: {max_int_diff})\n")

  # Loss comparison
  loss_diff = jnp.abs(loss_rs - loss_nors)
  loss_match = jnp.allclose(loss_rs, loss_nors, rtol=rtol, atol=atol)
  print(f"Losses match: {loss_match} (abs diff: {loss_diff})\n")

  # x_grad comparison
  x_grad_match = jnp.allclose(x_grad_rs, x_grad_nors, rtol=rtol, atol=atol)
  max_x_grad_diff = jnp.max(jnp.abs(x_grad_rs - x_grad_nors))
  print(f"x_grad match: {x_grad_match} (max abs diff: {max_x_grad_diff})")
  if not x_grad_match:
    diff_mask = ~jnp.isclose(x_grad_rs, x_grad_nors, rtol=rtol, atol=atol)
    mismatch_count = jnp.sum(diff_mask)
    print(f"  -> Mismatched x_grad elements: {mismatch_count} / {x_grad_rs.size}\n")

  # params_grad comparison
  print("Checking param gradients...")
  params_match, max_params_diff = compare_pytrees_detailed(
      params_grad_rs, params_grad_nors, rtol, atol, tree_name="Param Gradients"
  )
  print(f"params_grad match: {params_match} (max abs diff: {max_params_diff})")