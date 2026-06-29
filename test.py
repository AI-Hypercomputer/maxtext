import jax
import jax.numpy as jnp
from flax import nnx
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
      use_tokamax_gmm=True,
      use_gmm_v2=True,
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


rng = jax.random.PRNGKey(2345)
rng_model, rng_hidden_states = jax.random.split(rng)
device_count = jax.device_count()

# Target run with ragged sort - just fwd
cfg_rs = _build_cfg(use_ragged_sort=True)
hidden_states = jax.random.uniform(
    rng_hidden_states,
    (int(cfg_rs.per_device_batch_size) * device_count, cfg_rs.max_target_length, cfg_rs.base_emb_dim),
    dtype=cfg_rs.dtype,
)
devices_array = maxtext_utils.create_device_mesh(cfg_rs)
mesh = Mesh(devices_array, cfg_rs.mesh_axes)
model = _build_model(cfg_rs, mesh)
with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_rs.logical_axis_rules):
  variables = model.init({"params": rng_model, "dropout": rng_model}, hidden_states)
  out, lb_loss, _ = jax.jit(model.apply)(variables, hidden_states)
  print(f"Forward output has NaN: {jnp.any(jnp.isnan(out))}")
  print(f"Forward output max: {jnp.max(jnp.abs(out))}")
  if lb_loss is not None:
    print(f"Forward lb_loss has NaN: {jnp.any(jnp.isnan(lb_loss))}")

  # Now test grad
  def loss_fn(params, x):
    out, lb_loss, _ = model.apply({"params": params}, x)
    return jnp.mean(out.astype(jnp.float32) ** 2)

  loss, x_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=1))(variables["params"], hidden_states)
  print(f"Loss (no lb): {loss}")
  print(f"x_grad has NaN: {jnp.any(jnp.isnan(x_grad))}")
  print(f"x_grad shape: {x_grad.shape}")
  if jnp.any(jnp.isnan(x_grad)):
    nan_count = jnp.sum(jnp.isnan(x_grad))
    print(f"NaN count: {nan_count} / {x_grad.size}")
