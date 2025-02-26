import jax
import sre_parse
import unittest
from layers import linears
from layers import initializers
import jax.numpy as jnp

import pyconfig
import max_utils
from jax.sharding import Mesh
import flax.linen as nn
from typing import Tuple
import common_types
import pytest


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer


class MoeLoopBlock(nn.Module):
  """Reference implemetnation from https://github.com/mistralai/mistral-inference.
  This is not included anymore in our repo,
  due to limitation of for-loop implementation in sharding.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.bfloat16

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = linears.DenseGeneral(
        self.num_experts, dtype=self.dtype, kernel_init=self.kernel_init, kernel_axes=self.kernel_axes, name="gate"
    )(inputs)

    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    mlp_lnx = jnp.zeros_like(inputs)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    for k in range(self.num_experts):
      weights_exp = jnp.sum(jnp.multiply(selected_experts == k, weights), axis=-1)
      mlp_lnx_exp = linears.MlpBlock(
          intermediate_dim=self.config.mlp_dim,
          activations=["silu", "linear"],
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name=f"mlp_{k}",
          config=self.config,
      )(inputs, deterministic=deterministic)

      mlp_lnx_exp = nn.with_logical_constraint(mlp_lnx_exp, ("activation_batch", "activation_length", "activation_embed"))
      mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
      mlp_lnx += mlp_lnx_exp

    return mlp_lnx


class MoeBlockTest(unittest.TestCase):

  def get_expected_output(self, rng, hidden_states, cfg):
    model = MoeLoopBlock(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
    )
    variables = model.init(
        rng, jax.random.normal(rng, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim))
    )

    output = jax.jit(model.apply)(variables, hidden_states)
    return variables, output

  def get_moe_output(self, variables, hidden_states, cfg, mesh):
    model = linears.MoeBlock(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    # convert format of parameters
    kernel = variables["params"]["gate"]["kernel"].value
    kernel = kernel.astype(cfg.weight_dtype)

    exp_wi_0 = []
    exp_wi_1 = []
    exp_wo = []

    for i in range(cfg.num_experts):
      tmp_wi_0 = variables["params"][f"mlp_{i}"]["wi_0"]["kernel"].value
      tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wi_1 = variables["params"][f"mlp_{i}"]["wi_1"]["kernel"].value
      tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wo = variables["params"][f"mlp_{i}"]["wo"]["kernel"].value
      tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))

      exp_wi_0.append(tmp_wi_0)
      exp_wi_1.append(tmp_wi_1)
      exp_wo.append(tmp_wo)

    wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
    wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
    wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)

    moe_variables = {"params": {"gate": {"kernel": kernel}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo}}
    output = model.apply(moe_variables, hidden_states)
    return output

  @pytest.mark.tpu_only
  def test_ragged_all_to_all(self):
    cfg = pyconfig.initialize(
        [None, "configs/base.yml"],
        run_name="moe_all_to_all_test",
        enable_checkpointing=False,
        model_name="mixtral-test",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        ici_expert_parallelism=4,
        ici_fsdp_parallelism=1,
        max_target_length=64,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    hidden_states = jax.random.uniform(
        rng_hidden_states, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim), dtype=cfg.dtype
    )

    devices_array = max_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg)
    actual_output, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)


if __name__ == "__main__":
  unittest.main()
