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

"""Correctness tests for the mHC Pallas kernel."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import HyperConnectionType
from maxtext.configs import pyconfig
from maxtext.kernels.residual import mhc_kernel
from maxtext.layers import attention_mla, linears, mhc, moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils
import numpy as np
import pytest
from tests.utils.test_helpers import get_test_config_path

RTOL = 2e-2
ATOL = 2e-2


def assert_close_or_eq(r, p, rtol=None, atol=None):
  if not isinstance(r, (np.ndarray, jax.Array)):
    assert r == p
    return
  try:
    if jnp.issubdtype(r.dtype, jnp.floating):
      if rtol is None:
        rtol = 1e-2 if r.dtype == jnp.float32 else 1e-1
      if atol is None:
        atol = 1e-2 if r.dtype == jnp.float32 else 1e-1
      np.testing.assert_allclose(r, p, rtol=rtol, atol=atol)
    else:
      np.testing.assert_array_equal(r, p)
  except TypeError as e:
    if "PRNGKey" in str(e):
      np.testing.assert_array_equal(jax.random.key_data(r), jax.random.key_data(p))
    else:
      raise e


@pytest.mark.tpu_only
class TestMHCPallasCorrectness(parameterized.TestCase):
  """Verifies the mathematical correctness of the mHC Pallas kernel.

  Compares outputs and gradients against the reference JAX implementation.
  """

  def _setup_mhc_configs(self, rate, enable_pallas, per_device_batch_size=None, fsdp=None, data=None, tensor=None, dtype=None):
    """Sets up the configurations and modules for MHC testing."""
    self.dim = 16
    if per_device_batch_size is None:
      per_device_batch_size = jax.device_count()

    extra_kwargs = {}
    if fsdp is not None: extra_kwargs['ici_fsdp_parallelism'] = fsdp
    if data is not None: extra_kwargs['ici_data_parallelism'] = data
    if tensor is not None: extra_kwargs['ici_tensor_parallelism'] = tensor
    if dtype is not None:
      extra_kwargs['dtype'] = dtype
      extra_kwargs['weight_dtype'] = dtype
      if dtype == "float32":
        extra_kwargs['matmul_precision'] = "highest"

    config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"test_mhc_k{rate}_pallas{enable_pallas}",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        per_device_batch_size=per_device_batch_size,
        max_target_length=32,
        max_prefill_predict_length=32,
        attention="dot_product",
        routed_bias_update_rate=0.01,
        load_balance_loss_weight=0.02,
        # override
        override_model_config=True,
        base_emb_dim=self.dim,
        mhc_expansion_rate=rate,
        enable_mhc_lite=True,  # Pallas kernel only supports lite
        enable_mhc_pallas_kernel=enable_pallas,
        mhc_pallas_block_t=16,  # Use block size of 16 (or smaller for small inputs)
        num_experts=4,
        num_experts_per_tok=2,
        engram_layers=[],
        **extra_kwargs,
    )
    return config

  def _run_equivalence_test(self, rate, mhc_type, setup_branch_fn, per_device_batch_size=None, fsdp=None, data=None, tensor=None, dtype=None):
    """Helper to run equivalence tests for forward and backward passes."""
    config_ref = self._setup_mhc_configs(rate, enable_pallas=False, per_device_batch_size=per_device_batch_size, fsdp=fsdp, data=data, tensor=tensor, dtype=dtype)
    config_pal = self._setup_mhc_configs(rate, enable_pallas=True, per_device_batch_size=per_device_batch_size, fsdp=fsdp, data=data, tensor=tensor, dtype=dtype)

    devices_array = maxtext_utils.create_device_mesh(config_ref)
    mesh = Mesh(devices_array, config_ref.mesh_axes)

    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(42))

    resolved_fsdp = mesh.shape.get('fsdp', 1)
    resolved_data = mesh.shape.get('data', 1)
    global_batch_size = config_ref.per_device_batch_size * resolved_fsdp * resolved_data



    x = jax.random.normal(
        jax.random.PRNGKey(123),
        (
            global_batch_size,
            config_ref.max_target_length,
            config_ref.mhc_expansion_rate,
            config_ref.emb_dim,
        ),
    ).astype(config_ref.dtype)

    # Initialize modules with identical parameters
    with nn_partitioning.axis_rules(config_ref.logical_axis_rules):
      pre_norm_ref = RMSNorm(
          num_features=self.dim,
          dtype=config_ref.dtype,
          weight_dtype=config_ref.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config_ref.normalization_layer_epsilon,
          rngs=rngs,
      )
      pre_norm_pal = RMSNorm(
          num_features=self.dim,
          dtype=config_pal.dtype,
          weight_dtype=config_pal.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config_pal.normalization_layer_epsilon,
          rngs=rngs,
      )
      # Copy norm state
      nnx.update(pre_norm_pal, nnx.state(pre_norm_ref))

      branch_ref = setup_branch_fn(config_ref, mesh, rngs)
      branch_pal = setup_branch_fn(config_pal, mesh, rngs)
      # Copy branch state
      nnx.update(branch_pal, nnx.state(branch_ref))

      mhc_ref = mhc.ManifoldConstrainedHyperConnections(
          config_ref, self.dim, mesh, rngs
      )
      mhc_pal = mhc.ManifoldConstrainedHyperConnections(
          config_pal, self.dim, mesh, rngs
      )
      # Copy mHC state (includes alpha/beta parameters)
      nnx.update(mhc_pal, nnx.state(mhc_ref))

      # --------------------------------------------------------------------------------
      # 1. Forward Pass Comparison
      # --------------------------------------------------------------------------------
      # We want to keep original modules pure, so we copy them for FWD
      mhc_ref_fwd = nnx.merge(*nnx.split(mhc_ref))
      norm_ref_fwd = nnx.merge(*nnx.split(pre_norm_ref))
      branch_ref_fwd = nnx.merge(*nnx.split(branch_ref))

      mhc_pal_fwd = nnx.merge(*nnx.split(mhc_pal))
      norm_pal_fwd = nnx.merge(*nnx.split(pre_norm_pal))
      branch_pal_fwd = nnx.merge(*nnx.split(branch_pal))

      if mhc_type == HyperConnectionType.ATTENTION:
        kwargs = {
            "decoder_segment_ids": jnp.ones(x.shape[:2], dtype=jnp.int32),
            "inputs_positions": jnp.arange(x.shape[1], dtype=jnp.int32)[
                None, :
            ],
            "deterministic": True,
        }
      else:
        kwargs = {"deterministic": True}

      out_ref, meta_ref = mhc_ref_fwd(
          norm_ref_fwd, branch_ref_fwd, x, mhc_type, **kwargs
      )
      out_pal, meta_pal = mhc_pal_fwd(
          norm_pal_fwd, branch_pal_fwd, x, mhc_type, **kwargs
      )


      np.testing.assert_allclose(out_ref, out_pal, rtol=RTOL, atol=ATOL)

      # Check metadata (like load_balance_loss)
      self.assertEqual(meta_ref.keys(), meta_pal.keys())
      for k in meta_ref.keys():
        if isinstance(meta_ref[k], jnp.ndarray):
          np.testing.assert_allclose(meta_ref[k], meta_pal[k], rtol=RTOL, atol=ATOL)
        else:
          self.assertEqual(meta_ref[k], meta_pal[k])

      # Compare updated states (like Cache) after FWD
      _, _, other_ref_fwd = nnx.split(branch_ref_fwd, nnx.Param, ...)
      _, _, other_pal_fwd = nnx.split(branch_pal_fwd, nnx.Param, ...)
      jax.tree_util.tree_map(assert_close_or_eq, other_ref_fwd, other_pal_fwd)

      _, _, other_norm_ref_fwd = nnx.split(norm_ref_fwd, nnx.Param, ...)
      _, _, other_norm_pal_fwd = nnx.split(norm_pal_fwd, nnx.Param, ...)
      jax.tree_util.tree_map(assert_close_or_eq, other_norm_ref_fwd, other_norm_pal_fwd)

      # --------------------------------------------------------------------------------
      # 2. Backward Pass Comparison
      # --------------------------------------------------------------------------------
      def loss_fn_nnx(mhc_mod, norm_mod, branch_mod, inputs_x):
        out_local, metadata_local = mhc_mod(
            norm_mod, branch_mod, inputs_x, mhc_type, **kwargs
        )
        loss = jnp.sum(out_local)
        if "load_balance_loss" in metadata_local:
          loss += metadata_local["load_balance_loss"]
        return loss

      # Differentiate w.r.t params and input x using nnx.value_and_grad
      grad_fn_ref = nnx.value_and_grad(
          loss_fn_nnx,
          argnums=(
              nnx.DiffState(0, nnx.Param),
              nnx.DiffState(1, nnx.Param),
              nnx.DiffState(2, nnx.Param),
              3,
          ),
      )
      grad_fn_pal = nnx.value_and_grad(
          loss_fn_nnx,
          argnums=(
              nnx.DiffState(0, nnx.Param),
              nnx.DiffState(1, nnx.Param),
              nnx.DiffState(2, nnx.Param),
              3,
          ),
      )

      # Copy again for BWD to start from clean state
      mhc_ref_bwd = nnx.merge(*nnx.split(mhc_ref))
      norm_ref_bwd = nnx.merge(*nnx.split(pre_norm_ref))
      branch_ref_bwd = nnx.merge(*nnx.split(branch_ref))

      mhc_pal_bwd = nnx.merge(*nnx.split(mhc_pal))
      norm_pal_bwd = nnx.merge(*nnx.split(pre_norm_pal))
      branch_pal_bwd = nnx.merge(*nnx.split(branch_pal))

      loss_ref, grads_ref = grad_fn_ref(
          mhc_ref_bwd,
          norm_ref_bwd,
          branch_ref_bwd,
          x,
      )
      loss_pal, grads_pal = grad_fn_pal(
          mhc_pal_bwd,
          norm_pal_bwd,
          branch_pal_bwd,
          x,
      )

      g_mhc_ref, g_norm_ref, g_branch_ref, gx_ref = grads_ref
      g_mhc_pal, g_norm_pal, g_branch_pal, gx_pal = grads_pal

      # Check loss value
      np.testing.assert_allclose(loss_ref, loss_pal, rtol=RTOL, atol=ATOL)

      # Check gradients
      is_bf16 = (config_ref.dtype == jnp.bfloat16)
      grad_rtol = 2e-1 if is_bf16 else 1e-2
      grad_atol = 4e-1 if is_bf16 else 1e-2

      # dx

      assert_close_or_eq(gx_ref, gx_pal, rtol=grad_rtol, atol=grad_atol)

      # mHC params
      ref_mhc_dict = {path: var.value for path, var in g_mhc_ref.flat_state()}
      pal_mhc_dict = {path: var.value for path, var in g_mhc_pal.flat_state()}

      # Norm params
      ref_norm_dict = {path: var.value for path, var in g_norm_ref.flat_state()}
      pal_norm_dict = {path: var.value for path, var in g_norm_pal.flat_state()}

      # Branch params
      ref_branch_dict = {path: var.value for path, var in g_branch_ref.flat_state()}
      pal_branch_dict = {path: var.value for path, var in g_branch_pal.flat_state()}

      failed = False

      def compare_dict(name, ref_dict, pal_dict):
        nonlocal failed
        for path, ref_val in ref_dict.items():
          pal_val = pal_dict[path]
          max_diff = np.max(np.abs(ref_val - pal_val))
          mean_diff = np.mean(np.abs(ref_val - pal_val))
          status = "PASS"
          try:
            assert_close_or_eq(ref_val, pal_val, rtol=grad_rtol, atol=grad_atol)
          except AssertionError as e:
            status = "FAIL"
            failed = True
            print(f"MHC_TEST_DEBUG: Path {path} failed assertion: {e}")
            print(f"MHC_TEST_DEBUG: ref_val (first 5 elements): {np.array(ref_val).flatten()[:5]}")
            print(f"MHC_TEST_DEBUG: pal_val (first 5 elements): {np.array(pal_val).flatten()[:5]}")
          print(f"MHC_DIFF: {name} | {path} | max={max_diff:.6f} mean={mean_diff:.6f} | {status}")

      compare_dict("mHC", ref_mhc_dict, pal_mhc_dict)
      compare_dict("Norm", ref_norm_dict, pal_norm_dict)
      compare_dict("Branch", ref_branch_dict, pal_branch_dict)

      if failed:
        self.fail("Numerical mismatch in gradients. See MHC_DIFF logs above.")

      # Also check that bwd updated the states (Cache) identically
      _, _, other_ref_bwd = nnx.split(branch_ref_bwd, nnx.Param, ...)
      _, _, other_pal_bwd = nnx.split(branch_pal_bwd, nnx.Param, ...)
      jax.tree_util.tree_map(assert_close_or_eq, other_ref_bwd, other_pal_bwd)

  @pytest.mark.tpu_only
  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_dense_mlp_equivalence(self, rate):
    """Verifies correctness when the branch function is MlpBlock."""

    def setup_mlp(config, mesh, rngs):
      return linears.MlpBlock(
          config=config,
          mesh=mesh,
          in_features=self.dim,
          intermediate_dim=config.moe_mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          model_mode=config.model_call_mode,
          rngs=rngs,
      )

    self._run_equivalence_test(
        rate, HyperConnectionType.MLP_DENSE, setup_mlp,
        per_device_batch_size=1, fsdp=-1, data=1, tensor=1
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(("Rate3", 3))
  def test_dense_mlp_equivalence_f32(self, rate):
    """Verifies correctness when the branch function is MlpBlock in float32."""

    def setup_mlp(config, mesh, rngs):
      return linears.MlpBlock(
          config=config,
          mesh=mesh,
          in_features=self.dim,
          intermediate_dim=config.moe_mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          model_mode=config.model_call_mode,
          rngs=rngs,
      )

    with jax.default_matmul_precision("highest"):
      self._run_equivalence_test(
          rate, HyperConnectionType.MLP_DENSE, setup_mlp,
          per_device_batch_size=1, fsdp=-1, data=1, tensor=1,
          dtype="float32"
      )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_attention_equivalence(self, rate):
    """Verifies correctness when the branch function is MLA attention."""

    def setup_mla(config, mesh, rngs):
      inputs_shape = (
          config.per_device_batch_size,
          config.max_target_length,
          config.emb_dim,
      )
      return attention_mla.MLA(
          config=config,
          num_query_heads=config.num_query_heads,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
          max_target_length=config.max_target_length,
          max_prefill_predict_length=config.max_prefill_predict_length,
          attention_kernel=config.attention,
          attention_type=config.attention_type,
          inputs_q_shape=inputs_shape,
          inputs_kv_shape=inputs_shape,
          mesh=mesh,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          dropout_rate=config.dropout_rate,
          name="self_attention",
          q_lora_rank=config.q_lora_rank,
          kv_lora_rank=config.kv_lora_rank,
          qk_nope_head_dim=config.qk_nope_head_dim,
          qk_rope_head_dim=config.qk_rope_head_dim,
          v_head_dim=config.v_head_dim,
          max_position_embeddings=config.max_position_embeddings,
          original_max_position_embeddings=config.original_max_position_embeddings,
          mscale=config.mscale,
          rope_factor=config.rope_factor,
          model_mode="train",
          rngs=rngs,
          attn_logits_soft_cap=config.attn_logits_soft_cap,
      )

    self._run_equivalence_test(
        rate, HyperConnectionType.ATTENTION, setup_mla,
        per_device_batch_size=1, fsdp=-1, data=1, tensor=1
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_attention_equivalence_f32(self, rate):
    """Verifies correctness when the branch function is Attention in float32."""

    def setup_mla(config, mesh, rngs):
      inputs_shape = (
          config.per_device_batch_size,
          config.max_target_length,
          config.emb_dim,
      )
      return attention_mla.MLA(
          config=config,
          num_query_heads=config.num_query_heads,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
          max_target_length=config.max_target_length,
          max_prefill_predict_length=config.max_prefill_predict_length,
          attention_kernel=config.attention,
          attention_type=config.attention_type,
          inputs_q_shape=inputs_shape,
          inputs_kv_shape=inputs_shape,
          mesh=mesh,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          dropout_rate=config.dropout_rate,
          name="self_attention",
          q_lora_rank=config.q_lora_rank,
          kv_lora_rank=config.kv_lora_rank,
          qk_nope_head_dim=config.qk_nope_head_dim,
          qk_rope_head_dim=config.qk_rope_head_dim,
          v_head_dim=config.v_head_dim,
          max_position_embeddings=config.max_position_embeddings,
          original_max_position_embeddings=config.original_max_position_embeddings,
          mscale=config.mscale,
          rope_factor=config.rope_factor,
          model_mode="train",
          rngs=rngs,
          attn_logits_soft_cap=config.attn_logits_soft_cap,
      )

    with jax.default_matmul_precision("highest"):
      self._run_equivalence_test(
          rate, HyperConnectionType.ATTENTION, setup_mla,
          per_device_batch_size=1, fsdp=-1, data=1, tensor=1,
          dtype="float32"
      )

  # @pytest.mark.tpu_only
  # @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def disabled_test_moe_equivalence(self, rate):
    """Verifies correctness when the branch function is RoutedMoE."""

    def setup_moe(config, mesh, rngs):
      return moe.RoutedMoE(
          config=config,
          num_experts=config.num_experts,
          num_experts_per_tok=config.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=config.base_mlp_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          rngs=rngs,
      )

    self._run_equivalence_test(
        rate, HyperConnectionType.MLP_MOE, setup_moe
    )

  def test_pre_apply_bwd_equivalence(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), config.max_target_length, rate, 16
    T = b * s

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (b, s, k, d), dtype=jnp.bfloat16)
    H_pre = jax.random.normal(key, (b, s, k), dtype=jnp.bfloat16)
    d_layer_in = jax.random.normal(key, (b, s, d), dtype=jnp.bfloat16)
    dx_acc = jnp.zeros((b, s, k, d), dtype=jnp.bfloat16)

    xT = x.reshape(T, k, d)
    H_pre_flat = H_pre.reshape(T, k)
    d_layer_in_flat = d_layer_in.reshape(T, d)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      dx_acc_pl, dH_pre_pl = mhc_kernel.pre_apply_bwd_sharded(
          x,
          H_pre,
          d_layer_in,
          dx_acc,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
      )

    def ref_fwd(x_in, h_in):
      return mhc_kernel.mhc_pre_apply(x_in, h_in)

    _, ref_vjp = jax.vjp(ref_fwd, xT, H_pre_flat)
    dx_ref, dH_pre_ref = ref_vjp(d_layer_in_flat)

    dx_ref = dx_ref.reshape(b, s, k, d)
    dH_pre_ref = dH_pre_ref.reshape(b, s, k)

    assert_close_or_eq(dx_ref, dx_acc_pl)
    assert_close_or_eq(dH_pre_ref, dH_pre_pl)
    print("MHC_TEST_DEBUG: pre_apply_bwd equivalence passed!")

  def test_coeff_fwd_equivalence(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), 16, rate, 16
    T = b * s
    m = k * d
    perm = mhc.get_permutation_matrices(k).astype(jnp.bfloat16)
    P = perm.shape[0]

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (b, s, k, d), dtype=jnp.bfloat16)
    phi = jax.random.normal(key, (2 * k + P, m), dtype=jnp.bfloat16)
    norm_scale = jax.random.normal(key, (m,), dtype=jnp.bfloat16)

    pre_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    pre_beta = jax.random.normal(key, (k,), dtype=jnp.bfloat16)
    post_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    post_beta = jax.random.normal(key, (k,), dtype=jnp.bfloat16)
    res_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    res_beta = jax.random.normal(key, (P,), dtype=jnp.bfloat16)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      H_pre_pl, H_post_pl, res_M_pl = mhc_kernel.coeff_fwd_sharded(
          x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
      )

    xT = x.reshape(T, k, d)
    H_pre_ref, H_post_ref, res_M_ref = mhc_kernel.mhc_coeffs(
        xT, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm
    )

    H_pre_ref = H_pre_ref.reshape(b, s, k)
    H_post_ref = H_post_ref.reshape(b, s, k)
    res_M_ref = res_M_ref.reshape(b, s, k, k)

    assert_close_or_eq(H_pre_ref, H_pre_pl)
    assert_close_or_eq(H_post_ref, H_post_pl)
    assert_close_or_eq(res_M_ref, res_M_pl)
    print("MHC_TEST_DEBUG: coeff_fwd equivalence passed!")

  def test_coeff_bwd_equivalence(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), config.max_target_length, rate, 16
    T = b * s
    m = k * d
    perm = mhc.get_permutation_matrices(k).astype(jnp.bfloat16)
    P = perm.shape[0]

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (b, s, k, d), dtype=jnp.bfloat16)
    phi = jax.random.normal(key, (2 * k + P, m), dtype=jnp.bfloat16)
    norm_scale = jax.random.normal(key, (m,), dtype=jnp.bfloat16)

    pre_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    pre_beta = jax.random.normal(key, (k,), dtype=jnp.bfloat16)
    post_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    post_beta = jax.random.normal(key, (k,), dtype=jnp.bfloat16)
    res_s = jax.random.normal(key, (1,), dtype=jnp.bfloat16)
    res_beta = jax.random.normal(key, (P,), dtype=jnp.bfloat16)

    dH_pre = jax.random.normal(key, (b, s, k), dtype=jnp.bfloat16)
    dH_post = jax.random.normal(key, (b, s, k), dtype=jnp.bfloat16)
    dres_M = jax.random.normal(key, (b, s, k, k), dtype=jnp.bfloat16)
    dx_acc = jax.random.normal(key, (b, s, k, d), dtype=jnp.bfloat16)

    xT = x.reshape(T, k, d)
    dH_pre_flat = dH_pre.reshape(T, k)
    dH_post_flat = dH_post.reshape(T, k)
    dres_M_flat = dres_M.reshape(T, k, k)
    dx_acc_flat = dx_acc.reshape(T, k, d)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      dx_pl, dphi_pl, dns_pl, dps_pl, dpb_pl, dqs_pl, dqb_pl, drs_pl, drb_pl = mhc_kernel.coeff_bwd_sharded(
          x,
          phi,
          norm_scale,
          pre_s,
          pre_beta,
          post_s,
          post_beta,
          res_s,
          res_beta,
          perm,
          dH_pre,
          dH_post,
          dres_M,
          dx_acc,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
      )

    def ref_fwd(x_in, phi_in, norm_scale_in, ps_in, pb_in, qs_in, qb_in, rs_in, rb_in):
      return mhc_kernel.mhc_coeffs(
          x_in, phi_in, norm_scale_in, ps_in, pb_in, qs_in, qb_in, rs_in, rb_in, perm
      )

    _, ref_vjp = jax.vjp(
        ref_fwd,
        xT,
        phi,
        norm_scale,
        pre_s,
        pre_beta,
        post_s,
        post_beta,
        res_s,
        res_beta,
    )

    dx_ref, dphi_ref, dnorm_scale_ref, dps_ref, dpb_ref, dqs_ref, dqb_ref, drs_ref, drb_ref = ref_vjp(
        (dH_pre_flat, dH_post_flat, dres_M_flat)
    )

    dx_ref = (dx_ref.reshape(b, s, k, d) + dx_acc).astype(jnp.bfloat16)
    dphi_ref = dphi_ref.astype(jnp.bfloat16)
    dnorm_scale_ref = dnorm_scale_ref.astype(jnp.bfloat16)
    dps_ref = dps_ref.astype(jnp.bfloat16)
    dpb_ref = dpb_ref.astype(jnp.bfloat16)
    dqs_ref = dqs_ref.astype(jnp.bfloat16)
    dqb_ref = dqb_ref.astype(jnp.bfloat16)
    drs_ref = drs_ref.astype(jnp.bfloat16)
    drb_ref = drb_ref.astype(jnp.bfloat16)

    print("MHC_TEST_DEBUG: dx max diff:", jnp.max(jnp.abs(dx_ref.astype(jnp.float32) - dx_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dphi max diff:", jnp.max(jnp.abs(dphi_ref.astype(jnp.float32) - dphi_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dns max diff:", jnp.max(jnp.abs(dnorm_scale_ref.astype(jnp.float32) - dns_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dps max diff:", jnp.max(jnp.abs(dps_ref.astype(jnp.float32) - dps_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dpb max diff:", jnp.max(jnp.abs(dpb_ref.astype(jnp.float32) - dpb_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dqs max diff:", jnp.max(jnp.abs(dqs_ref.astype(jnp.float32) - dqs_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: dqb max diff:", jnp.max(jnp.abs(dqb_ref.astype(jnp.float32) - dqb_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: drs max diff:", jnp.max(jnp.abs(drs_ref.astype(jnp.float32) - drs_pl.astype(jnp.float32))))
    print("MHC_TEST_DEBUG: drb max diff:", jnp.max(jnp.abs(drb_ref.astype(jnp.float32) - drb_pl.astype(jnp.float32))))

    assert_close_or_eq(dx_ref, dx_pl)
    assert_close_or_eq(dphi_ref, dphi_pl)
    assert_close_or_eq(dnorm_scale_ref, dns_pl)
    assert_close_or_eq(dps_ref, dps_pl, rtol=2e-1, atol=2e-1)
    assert_close_or_eq(dpb_ref, dpb_pl)
    assert_close_or_eq(dqs_ref, dqs_pl)
    assert_close_or_eq(dqb_ref, dqb_pl)
    assert_close_or_eq(drs_ref, drs_pl)
    assert_close_or_eq(drb_ref, drb_pl)
    print("MHC_TEST_DEBUG: coeff_bwd equivalence passed!")

  def test_post_apply_bwd_act_equivalence(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), config.max_target_length, rate, 16
    T = b * s

    keys = jax.random.split(jax.random.PRNGKey(42), 5)
    x = jax.random.normal(keys[0], (b, s, k, d), dtype=jnp.bfloat16)
    layer_out = jax.random.normal(keys[1], (b, s, d), dtype=jnp.bfloat16)
    H_post = jax.random.normal(keys[2], (b, s, k), dtype=jnp.bfloat16)
    res_M = jax.random.normal(keys[3], (b, s, k, k), dtype=jnp.bfloat16)
    d_out = jax.random.normal(keys[4], (b, s, k, d), dtype=jnp.bfloat16)

    x_flat = x.reshape(T, k, d)
    layer_out_flat = layer_out.reshape(T, d)
    H_post_flat = H_post.reshape(T, k)
    res_M_flat = res_M.reshape(T, k, k)
    d_out_flat = d_out.reshape(T, k, d)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      dx_pl, dlo_pl = mhc_kernel.post_apply_bwd_act_sharded(
          res_M,
          H_post,
          d_out,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
          dlo_dtype=jnp.bfloat16,
      )

    def ref_fwd(x_in, lo_in, h_in, m_in):
      return mhc_kernel.mhc_post_apply(x_in, lo_in, h_in, m_in)

    _, ref_vjp = jax.vjp(ref_fwd, x_flat, layer_out_flat, H_post_flat, res_M_flat)
    dx_ref, dlo_ref, _, _ = ref_vjp(d_out_flat)

    dx_ref = dx_ref.reshape(b, s, k, d)
    dlo_ref = dlo_ref.reshape(b, s, d)

    assert_close_or_eq(dx_ref, dx_pl)
    assert_close_or_eq(dlo_ref, dlo_pl)

  def test_post_apply_bwd_act_equivalence_realistic(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), config.max_target_length, rate, 16
    T = b * s
    m = k * d

    keys = jax.random.split(jax.random.PRNGKey(42), 10)
    x = jax.random.normal(keys[0], (b, s, k, d), dtype=jnp.bfloat16)
    layer_out = jax.random.normal(keys[1], (b, s, d), dtype=jnp.bfloat16)

    x_flat = x.reshape(T, k, d)

    # Generate realistic res_M and H_post using mhc_coeffs
    perm = mhc.get_permutation_matrices(k).astype(jnp.bfloat16)
    P = perm.shape[0]
    phi = jax.random.normal(keys[2], (2 * k + P, m), dtype=jnp.bfloat16)
    norm_scale = jax.random.normal(keys[3], (m,), dtype=jnp.bfloat16)
    pre_s = jax.random.normal(keys[4], (1,), dtype=jnp.bfloat16)
    pre_beta = jax.random.normal(keys[5], (k,), dtype=jnp.bfloat16)
    post_s = jax.random.normal(keys[6], (1,), dtype=jnp.bfloat16)
    post_beta = jax.random.normal(keys[7], (k,), dtype=jnp.bfloat16)
    res_s = jax.random.normal(keys[8], (1,), dtype=jnp.bfloat16)
    res_beta = jax.random.normal(keys[9], (P,), dtype=jnp.bfloat16)

    H_pre_flat, H_post_flat, res_M_flat = mhc_kernel.mhc_coeffs(
        x_flat, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm
    )

    H_post = H_post_flat.reshape(b, s, k)
    res_M = res_M_flat.reshape(b, s, k, k)
    d_out = jnp.ones((b, s, k, d), dtype=jnp.bfloat16)
    d_out_flat = d_out.reshape(T, k, d)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      dx_pl, dlo_pl = mhc_kernel.post_apply_bwd_act_sharded(
          res_M,
          H_post,
          d_out,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
          dlo_dtype=jnp.bfloat16,
      )

    def ref_fwd(x_in, lo_in, h_in, m_in):
      return mhc_kernel.mhc_post_apply(x_in, lo_in, h_in, m_in)

    _, ref_vjp = jax.vjp(ref_fwd, x_flat, layer_out.reshape(T, d), H_post_flat, res_M_flat)
    dx_ref, dlo_ref, _, _ = ref_vjp(d_out_flat)

    dx_ref = dx_ref.reshape(b, s, k, d)
    dlo_ref = dlo_ref.reshape(b, s, d)

    assert_close_or_eq(dx_ref, dx_pl)
    assert_close_or_eq(dlo_ref, dlo_pl)
    print("MHC_TEST_DEBUG: post_apply_bwd_act_realistic passed!")

  def test_post_apply_bwd_weight_equivalence(self):
    rate = 3
    config = self._setup_mhc_configs(rate, enable_pallas=True)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    b, s, k, d = jax.device_count(), config.max_target_length, rate, 16
    T = b * s

    keys = jax.random.split(jax.random.PRNGKey(42), 5)
    x = jax.random.normal(keys[0], (b, s, k, d), dtype=jnp.bfloat16)
    layer_out = jax.random.normal(keys[1], (b, s, d), dtype=jnp.bfloat16)
    H_post = jax.random.normal(keys[2], (b, s, k), dtype=jnp.bfloat16)
    res_M = jax.random.normal(keys[3], (b, s, k, k), dtype=jnp.bfloat16)
    d_out = jax.random.normal(keys[4], (b, s, k, d), dtype=jnp.bfloat16)

    x_flat = x.reshape(T, k, d)
    layer_out_flat = layer_out.reshape(T, d)
    H_post_flat = H_post.reshape(T, k)
    res_M_flat = res_M.reshape(T, k, k)
    d_out_flat = d_out.reshape(T, k, d)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      dhpost_pl, dres_M_pl = mhc_kernel.post_apply_bwd_weight_sharded(
          x,
          layer_out,
          d_out,
          bt=config.mhc_pallas_block_t,
          vmem=mhc_kernel.VMEM_LIMIT_BYTES,
          interpret=False,
          mesh=mesh,
          rules=config.logical_axis_rules,
      )

    def ref_fwd(x_in, lo_in, h_in, m_in):
      return mhc_kernel.mhc_post_apply(x_in, lo_in, h_in, m_in)

    _, ref_vjp = jax.vjp(ref_fwd, x_flat, layer_out_flat, H_post_flat, res_M_flat)
    _, _, dhpost_ref, dres_M_ref = ref_vjp(d_out_flat)

    dhpost_ref = dhpost_ref.reshape(b, s, k)
    dres_M_ref = dres_M_ref.reshape(b, s, k, k)

    assert_close_or_eq(dhpost_ref, dhpost_pl)
    assert_close_or_eq(dres_M_ref, dres_M_pl)


if __name__ == "__main__":
  absltest.main()
