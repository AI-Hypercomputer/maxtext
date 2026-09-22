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

"""TPU hardware unit test for Lineage adapter across 4-device mesh."""

import os
import types

extra_tpu_flags = (
    " --xla_tpu_scoped_vmem_limit_kib=65536"
    " --xla_tpu_enable_offloading_copy_to_sparsecore=false"
    " --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false"
    " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
)
os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + extra_tpu_flags

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.models import lineage_adapter
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops
import numpy as np


def _create_ghostfish_physical_mesh(
    devices: list[jax.Device] | None = None,
) -> jax.sharding.Mesh:
  """Creates a 4D physical mesh for TPU testing."""
  if devices is None:
    devices = jax.devices()
  try:
    max_x = max(d.coords[0] for d in devices) + 1
    max_y = max(d.coords[1] for d in devices) + 1
    max_z = max(d.coords[2] for d in devices) + 1
    max_c = max(d.coords[3] if len(d.coords) > 3 else getattr(d, "core_on_chip", 0) for d in devices) + 1
    mesh_shape = (max_x, max_y, max_z, max_c)

    def device_key(d: jax.Device):
      c_val = d.coords[3] if len(d.coords) > 3 else getattr(d, "core_on_chip", 0)
      return (d.coords[0], d.coords[1], d.coords[2], c_val)

    sorted_devices = sorted(devices, key=device_key)
    device_mesh = np.array(sorted_devices, dtype=object).reshape(mesh_shape)
    return jax.sharding.Mesh(
        device_mesh,
        axis_names=("x", "y", "z", "core"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    arr = np.array(devices)
    if len(arr) >= 4:
      return jax.sharding.Mesh(
          arr[:4].reshape((1, 2, 2, 1)),
          ("x", "y", "z", "core"),
      )
    return jax.sharding.Mesh(arr.reshape((1, 1, 1, 1)), ("x", "y", "z", "core"))


class LineageAdapterTPUTest(parameterized.TestCase):
  """Tests executing Lineage DSv3 layers forward and backward on TPU mesh."""

  def _get_mesh(self):
    return _create_ghostfish_physical_mesh()

  def _get_config(
      self,
      num_dense_layers=2,
      num_sparse_layers=2,
  ):
    return types.SimpleNamespace(
        per_device_batch_size=2,
        emb_dim=512,
        mlp_dim=512,
        weight_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        num_heads=4,
        num_query_heads=4,
        num_kv_heads=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        max_target_length=2048,
        num_experts=16,
        num_experts_per_tok=8,
        n_routing_groups=8,
        topk_routing_group=4,
        topk_in_group=2,
        moe_mlp_dim=2048,
        routed_scaling_factor=2.5,
        rope_theta=10000,
        mscale=1.0,
        normalization_layer_epsilon=1e-5,
        max_position_embeddings=5120,
        original_max_position_embeddings=128,
        beta_fast=32,
        beta_slow=1,
        rope_factor=40,
        sa_block_q=1024,
        sa_block_kv=1024,
        sa_block_kv_compute=1024,
        sa_block_q_dkv=1024,
        sa_block_kv_dkv=1024,
        sa_block_kv_dkv_compute=1024,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        num_dense_layers=num_dense_layers,
        num_sparse_layers=num_sparse_layers,
        num_layers=num_dense_layers + num_sparse_layers,
        capacity_factor=8.0,
    )

  def _build_maxtext_dense_dict_params(self, cfg, mesh, axis_mapping, rng):
    """Builds mock MaxText dense layer parameters dictionary for testing."""
    num_layers = cfg.num_dense_layers
    emb_dim = cfg.emb_dim
    cq_dim = cfg.q_lora_rank
    ckv_dim = cfg.kv_lora_rank
    num_query_heads = cfg.num_query_heads
    qk_nope_head_dim = cfg.qk_nope_head_dim
    qk_rope_head_dim = cfg.qk_rope_head_dim
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = cfg.v_head_dim
    mlp_dim = cfg.mlp_dim
    dtype = cfg.weight_dtype

    def _create_param(rng_key, shape, pspec):
      phys_pspec = ops.physical_pspec(pspec, axis_mapping)
      return jax.random.normal(
          rng_key,
          shape,
          dtype=dtype,
          out_sharding=jax.sharding.NamedSharding(mesh, phys_pspec),
      )

    rngs = jax.random.split(rng, 12)
    return {
        "pre_self_attention_layer_norm": {
            "scale": _create_param(
                rngs[0],
                (num_layers, emb_dim),
                jax.sharding.PartitionSpec(None, None),
            ),
        },
        "self_attention": {
            "wq_a": {
                "kernel": _create_param(
                    rngs[1],
                    (num_layers, emb_dim, cq_dim),
                    jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                )
            },
            "wq_b": {
                "kernel": _create_param(
                    rngs[2],
                    (num_layers, cq_dim, num_query_heads, qk_head_dim),
                    jax.sharding.PartitionSpec(None, None, "attention", "fsdp_moe"),
                )
            },
            "q_norm": {
                "scale": _create_param(
                    rngs[3],
                    (num_layers, cq_dim),
                    jax.sharding.PartitionSpec(None, None),
                )
            },
            "wkv_a": {
                "kernel": _create_param(
                    rngs[4],
                    (num_layers, emb_dim, ckv_dim + qk_rope_head_dim),
                    jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                )
            },
            "wkv_b": {
                "kernel": _create_param(
                    rngs[5],
                    (
                        num_layers,
                        ckv_dim,
                        num_query_heads,
                        qk_nope_head_dim + v_head_dim,
                    ),
                    jax.sharding.PartitionSpec(None, None, "attention", "fsdp_moe"),
                )
            },
            "kv_norm": {
                "scale": _create_param(
                    rngs[6],
                    (num_layers, ckv_dim),
                    jax.sharding.PartitionSpec(None, None),
                )
            },
            "out": {
                "kernel": _create_param(
                    rngs[7],
                    (num_layers, num_query_heads, v_head_dim, emb_dim),
                    jax.sharding.PartitionSpec(None, "attention", None, "fsdp_moe"),
                )
            },
        },
        "post_self_attention_layer_norm": {
            "scale": _create_param(
                rngs[8],
                (num_layers, emb_dim),
                jax.sharding.PartitionSpec(None, None),
            ),
        },
        "mlp": {
            "wi_0": {
                "kernel": _create_param(
                    rngs[9],
                    (num_layers, emb_dim, mlp_dim),
                    jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                )
            },
            "wi_1": {
                "kernel": _create_param(
                    rngs[10],
                    (num_layers, emb_dim, mlp_dim),
                    jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                )
            },
            "wo": {
                "kernel": _create_param(
                    rngs[11],
                    (num_layers, mlp_dim, emb_dim),
                    jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                )
            },
        },
    }

  def _build_maxtext_sparse_dict_params(self, cfg, mesh, axis_mapping, rng):
    """Builds mock MaxText MoE layer parameters dictionary for testing."""
    num_layers = cfg.num_sparse_layers
    emb_dim = cfg.emb_dim
    cq_dim = cfg.q_lora_rank
    ckv_dim = cfg.kv_lora_rank
    num_query_heads = cfg.num_query_heads
    qk_nope_head_dim = cfg.qk_nope_head_dim
    qk_rope_head_dim = cfg.qk_rope_head_dim
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = cfg.v_head_dim
    num_experts = cfg.num_experts
    expert_hidden_dim = cfg.moe_mlp_dim
    dtype = cfg.weight_dtype

    def _create_param(rng_key, shape, pspec):
      phys_pspec = ops.physical_pspec(pspec, axis_mapping)
      return jax.random.normal(
          rng_key,
          shape,
          dtype=dtype,
          out_sharding=jax.sharding.NamedSharding(mesh, phys_pspec),
      )

    rngs = jax.random.split(rng, 18)
    return {
        "pre_self_attention_layer_norm": {
            "scale": _create_param(
                rngs[0],
                (num_layers, emb_dim),
                jax.sharding.PartitionSpec(None, None),
            ),
        },
        "self_attention": {
            "wq_a": {
                "kernel": _create_param(
                    rngs[1],
                    (num_layers, emb_dim, cq_dim),
                    jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                )
            },
            "wq_b": {
                "kernel": _create_param(
                    rngs[2],
                    (num_layers, cq_dim, num_query_heads, qk_head_dim),
                    jax.sharding.PartitionSpec(None, None, "attention", "fsdp_moe"),
                )
            },
            "q_norm": {
                "scale": _create_param(
                    rngs[3],
                    (num_layers, cq_dim),
                    jax.sharding.PartitionSpec(None, None),
                )
            },
            "wkv_a": {
                "kernel": _create_param(
                    rngs[4],
                    (num_layers, emb_dim, ckv_dim + qk_rope_head_dim),
                    jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                )
            },
            "wkv_b": {
                "kernel": _create_param(
                    rngs[5],
                    (
                        num_layers,
                        ckv_dim,
                        num_query_heads,
                        qk_nope_head_dim + v_head_dim,
                    ),
                    jax.sharding.PartitionSpec(None, None, "attention", "fsdp_moe"),
                )
            },
            "kv_norm": {
                "scale": _create_param(
                    rngs[6],
                    (num_layers, ckv_dim),
                    jax.sharding.PartitionSpec(None, None),
                )
            },
            "out": {
                "kernel": _create_param(
                    rngs[7],
                    (num_layers, num_query_heads, v_head_dim, emb_dim),
                    jax.sharding.PartitionSpec(None, "attention", None, "fsdp_moe"),
                )
            },
        },
        "post_self_attention_layer_norm": {
            "scale": _create_param(
                rngs[8],
                (num_layers, emb_dim),
                jax.sharding.PartitionSpec(None, None),
            ),
        },
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "gate": {
                    "kernel": _create_param(
                        rngs[9],
                        (num_layers, emb_dim, num_experts),
                        jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                    ),
                    "bias": jax.device_put(
                        jnp.zeros((num_layers, num_experts), dtype=dtype),
                        jax.sharding.NamedSharding(
                            mesh,
                            ops.physical_pspec(
                                jax.sharding.PartitionSpec(None, None),
                                axis_mapping,
                            ),
                        ),
                    ),
                },
                "wi_0": _create_param(
                    rngs[10],
                    (num_layers, num_experts, emb_dim, expert_hidden_dim),
                    jax.sharding.PartitionSpec(None, "expert", None, "fsdp_moe"),
                ),
                "wi_1": _create_param(
                    rngs[11],
                    (num_layers, num_experts, emb_dim, expert_hidden_dim),
                    jax.sharding.PartitionSpec(None, "expert", None, "fsdp_moe"),
                ),
                "wo": _create_param(
                    rngs[12],
                    (num_layers, num_experts, expert_hidden_dim, emb_dim),
                    jax.sharding.PartitionSpec(None, "expert", "fsdp_moe", None),
                ),
            },
            "shared_experts": {
                "wi_0": {
                    "kernel": _create_param(
                        rngs[13],
                        (num_layers, emb_dim, expert_hidden_dim),
                        jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                    )
                },
                "wi_1": {
                    "kernel": _create_param(
                        rngs[14],
                        (num_layers, emb_dim, expert_hidden_dim),
                        jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                    )
                },
                "wo": {
                    "kernel": _create_param(
                        rngs[15],
                        (num_layers, expert_hidden_dim, emb_dim),
                        jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                    )
                },
            },
        },
    }

  @parameterized.named_parameters(
      {
          "testcase_name": "without_dict_params",
          "use_dict_params": False,
      },
      {
          "testcase_name": "with_dict_params",
          "use_dict_params": True,
      },
  )
  def test_forward_and_backward_pass(
      self,
      use_dict_params: bool,
  ):
    cfg = self._get_config(
        num_dense_layers=2,
        num_sparse_layers=2,
    )
    mesh = self._get_mesh()
    with jax.set_mesh(mesh):
      axis_mapping = lineage_adapter.build_axis_mapping(mesh, cfg)

      rng = jax.random.PRNGKey(42)
      rng_inputs, rng_model = jax.random.split(rng)

      activation_pspec = jax.sharding.PartitionSpec("fsdp_attention", "attention", None)
      physical_activation_pspec = ops.physical_pspec(activation_pspec, axis_mapping)
      inputs = jax.random.normal(
          rng_inputs,
          (
              int(cfg.per_device_batch_size * jax.device_count()),
              cfg.max_target_length,
              cfg.emb_dim,
          ),
          dtype=cfg.dtype,
      )
      inputs = jax.reshard(
          inputs,
          jax.sharding.NamedSharding(mesh, physical_activation_pspec),
      )
      decoder_positions = jnp.broadcast_to(
          jnp.arange(inputs.shape[1], dtype=jnp.int32),
          (inputs.shape[0], inputs.shape[1]),
      )

      if use_dict_params:
        rng_dense, rng_sparse = jax.random.split(rng_model)
        dense_params = self._build_maxtext_dense_dict_params(cfg, mesh, axis_mapping, rng_dense)
        sparse_params = self._build_maxtext_sparse_dict_params(cfg, mesh, axis_mapping, rng_sparse)

        def loss_fn(x, d_p, s_p):
          out = lineage_adapter.run_lineage_dsv3(
              inputs=x,
              dense_params=d_p,
              sparse_params=s_p,
              decoder_positions=decoder_positions,
              mesh=mesh,
              cfg=cfg,
          )
          return jnp.sum(out), out

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True))
        (_, outputs), (x_grad, d_grad, s_grad) = grad_fn(inputs, dense_params, sparse_params)
        p_grad = (d_grad, s_grad)
      else:
        mla_shardings = dsv3_types.DSv3MLAWeightsPytree(
            q_down=jax.sharding.PartitionSpec(None, "fsdp_moe", None),
            q_up=jax.sharding.PartitionSpec(None, "fsdp_moe", None, None),
            kv_down=jax.sharding.PartitionSpec(None, "fsdp_moe", None),
            k_up=jax.sharding.PartitionSpec(None, "fsdp_moe", None, None),
            v_up=jax.sharding.PartitionSpec(None, "fsdp_moe", None, None),
            out=jax.sharding.PartitionSpec(None, None, None, "fsdp_moe"),
        )

        out_shardings = dsv3_types.DSv3WeightsPytree(
            dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                mla=mla_shardings,
                mlp=dsv3_types.DSv3MLPWeightsPytree(
                    gate_0=jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                    gate_1=jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                    linear=jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                ),
            ),
            sparse=dsv3_types.DSv3SparseLayerWeightsPytree(
                mla=mla_shardings,
                moe=dsv3_types.DSv3MoEWeightsPytree(
                    routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
                        gate=jax.sharding.PartitionSpec(None, "expert", None, "fsdp_moe"),
                        linear=jax.sharding.PartitionSpec(None, "expert", "fsdp_moe", None),
                    ),
                    shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
                        gate_0=jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                        gate_1=jax.sharding.PartitionSpec(None, None, "fsdp_moe"),
                        linear=jax.sharding.PartitionSpec(None, "fsdp_moe", None),
                    ),
                ),
            ),
        )

        params = dsv3_types.init_dsv3_weights(
            rng=rng_model,
            emb_dim=cfg.emb_dim,
            cq_dim=cfg.q_lora_rank,
            ckv_dim=cfg.kv_lora_rank,
            num_query_heads=cfg.num_query_heads,
            num_kv_heads=cfg.num_kv_heads,
            rope_head_dim=cfg.qk_rope_head_dim,
            qk_head_dim=cfg.qk_nope_head_dim,
            v_head_dim=cfg.v_head_dim,
            mlp_dim=cfg.mlp_dim,
            num_experts=cfg.num_experts,
            expert_hidden_dim=cfg.moe_mlp_dim,
            mesh=mesh,
            axis_mapping=axis_mapping,
            weight_dtype=cfg.weight_dtype,
            num_dense_layers=cfg.num_dense_layers,
            num_sparse_layers=cfg.num_sparse_layers,
            out_shardings=out_shardings,
        )

        def loss_fn(x, p):
          out = lineage_adapter.run_lineage_dsv3(
              inputs=x,
              dense_params=p,
              sparse_params=None,
              decoder_positions=decoder_positions,
              mesh=mesh,
              cfg=cfg,
          )
          return jnp.sum(out), out

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))

        (_, outputs), (x_grad, p_grad) = grad_fn(inputs, params)

    # 1. Check outputs
    self.assertEqual(outputs.shape, inputs.shape)
    self.assertEqual(outputs.dtype, cfg.dtype)
    self.assertFalse(
        jnp.any(jnp.isnan(outputs)),
        msg="Outputs contain NaNs",
    )
    self.assertFalse(
        jnp.all(outputs == 0),
        msg="Outputs are all zero",
    )

    # 2. Check input gradients
    self.assertEqual(x_grad.shape, inputs.shape)
    self.assertFalse(
        jnp.any(jnp.isnan(x_grad)),
        msg="Input gradients contain NaNs",
    )

    # 3. Check parameter gradients
    leaves, _ = jax.tree_util.tree_flatten(p_grad)
    for i, leaf in enumerate(leaves):
      if leaf is not None:
        self.assertFalse(
            jnp.any(jnp.isnan(leaf)),
            msg=f"Param gradient leaf {i} contains NaNs",
        )


if __name__ == "__main__":
  absltest.main()
