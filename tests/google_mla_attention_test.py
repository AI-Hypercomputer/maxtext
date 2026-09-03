# Copyright 2023–2025 Google LLC
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

"""Test for MLA Attentions comparing tokamax (pallas) and jax (non-pallas).

With the generic implementation of splash attention kernels.
"""


import math
import os.path
import sys

from absl import flags
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import sharding
from jax.sharding import NamedSharding
from maxtext.common import common_types
from maxtext.utils import globals as maxtext_globals
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.utils import maxtext_utils
from tests.utils import attention_test_util
import pytest

from google3.testing.pybase import googletest

# Define flags for manual testing
_PER_DEVICE_BATCH_SIZE = flags.DEFINE_float("per_device_batch_size", 2.0, "Per device batch size.")
_NUM_HEADS = flags.DEFINE_integer("num_heads", 128, "Number of heads.")
_SA_BLOCK_SIZE = flags.DEFINE_integer("sa_block_size", 1024, "SA block size.")
_SPLASH_TYPE = flags.DEFINE_string("splash_type", "jax", "Splash type ('jax' or 'tokamax').")
_SCOPED_VMEM_MULTIPLIER = flags.DEFINE_float("scoped_vmem_multiplier", 0.8, "Scoped vmem multiplier.")
_FORCE_Q_LAYOUT = flags.DEFINE_bool("force_q_layout", True, "Force Q layout.")
_RUN_MANUAL_TEST = flags.DEFINE_bool("run_manual_test", False, "Whether to run the manual test case.")

MLA = attention_mla.MLA
Mesh = sharding.Mesh
MAXTEXT_PKG_DIR = maxtext_globals.MAXTEXT_PKG_DIR
MODEL_MODE_TRAIN = common_types.MODEL_MODE_TRAIN
MODEL_MODE_PREFILL = common_types.MODEL_MODE_PREFILL
AttentionType = common_types.AttentionType


class DeepseekMLATest(attention_test_util.MLATestBase):
  """Test for the Multi-Headed Latent Attention for DeepseekV3 training."""

  def _run_flash_attention_fsdp_test(
      self,
      per_device_batch_size,
      num_heads,
      sa_block_size,
      splash_type,
      scoped_vmem_multiplier,
      force_q_layout,
  ):
    """Helper function to test equivalence between dot_product and flash attention in fsdp (expert parallelism) mode."""
    ici_context_parallelism = 1
    context_parallel_load_balance = False
    ici_expert_parallelism = 4

    use_jax_splash = splash_type == "jax"
    use_tokamax_splash = splash_type == "tokamax"
    config_arguments = {
        "per_device_batch_size": per_device_batch_size,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 1024,
        "sa_block_q": sa_block_size,
        "sa_block_kv": sa_block_size,
        "sa_block_kv_compute": 512,
        "sa_block_q_dkv": 128,
        "sa_block_kv_dkv": 128,
        "sa_block_kv_dkv_compute": 128,
        "sa_block_q_dq": 128,
        "sa_block_kv_dq": 128,
        "attention_type": AttentionType.MLA.value,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "base_num_query_heads": num_heads,
        "base_num_kv_heads": num_heads,
        "use_tokamax_splash": use_tokamax_splash,
        "use_jax_splash": use_jax_splash,
        "cast_logits_to_fp32": False,
        "force_q_layout": force_q_layout,
        "rope_min_timescale": 1.0,
        "rope_max_timescale": 10000.0,
    }

    cfg, mla = self.init_mla(config_arguments, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg, cfg.dtype)

    @nnx.jit
    def _jitted_mla_generic(mla, lnx, decoder_segment_ids, decoder_positions):
      # Dot product
      mla_generic_output, _ = mla(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return mla_generic_output

    generic_state = nnx.state(mla)

    # Test with Expert Parallelism
    cfg_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **config_arguments,
        rope_type=cfg.rope_type,
        ici_context_parallelism=ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        ici_expert_parallelism=ici_expert_parallelism,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      batch_axis = "activation_batch"
      length_axis = "activation_length"
      lnx_spec = nn_partitioning.logical_to_mesh_axes(
          (batch_axis, length_axis, "activation_embed"),
          nn_partitioning.get_axis_rules(),
      )
      pos_spec = nn_partitioning.logical_to_mesh_axes((None, length_axis), nn_partitioning.get_axis_rules())
      lnx_sharding = NamedSharding(mesh_cp, lnx_spec)  # pyrefly: ignore[bad-argument-type]
      pos_sharding = NamedSharding(mesh_cp, pos_spec)  # pyrefly: ignore[bad-argument-type]

      lnx_cp = jax.device_put(lnx, lnx_sharding)
      decoder_segment_ids_cp = jax.device_put(decoder_segment_ids, pos_sharding)
      decoder_positions_cp = jax.device_put(decoder_positions, pos_sharding)

      attention_as_mla_flash_cp = MLA(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          mesh=mesh_cp,
          attention_kernel="flash",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          attention_type=cfg_cp.attention_type,
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
      )
      # Reshard the generic state to match the flash attention module sharding
      # before updating.
      state_sharding = jax.tree.map(
          lambda x: x.sharding if hasattr(x, "sharding") else None,
          nnx.state(attention_as_mla_flash_cp),
      )
      generic_state_sharded = jax.device_put(generic_state, state_sharding)
      nnx.update(attention_as_mla_flash_cp, generic_state_sharded)

      @nnx.jit(static_argnames=("model_mode", "deterministic"))
      def jitted_attention_call(module, q, k, **kwargs):
        return module(q, k, **kwargs)

      executable = jitted_attention_call.lower(
          attention_as_mla_flash_cp,
          lnx_cp,
          lnx_cp,
          decoder_segment_ids=decoder_segment_ids_cp,
          inputs_positions=decoder_positions_cp,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      ).compile(
          {
              "xla_tpu_scoped_vmem_limit_kib": math.ceil(64 * 1024 * scoped_vmem_multiplier),
          }
      )

    def executable_wrapper(q, k, **kwargs):
      # Prune static arguments that were baked into compilation.
      kwargs.pop("deterministic", None)
      kwargs.pop("model_mode", None)
      return executable(attention_as_mla_flash_cp, q, k, **kwargs)

    mla_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        executable_wrapper,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )
    mla_generic_output = _jitted_mla_generic(mla, lnx, decoder_segment_ids, decoder_positions)
    jax.block_until_ready(mla_generic_output)
    jax.block_until_ready(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(
            mla_generic_output,
            mla_generic_flash_cp_output,
            rtol=1e-01,
            atol=1e-01,
            equal_nan=False,
        ),
        msg=(
            "MLA Logits from generic dot product and flash attention fsdp are"
            f" not close.\nici_context_parallelism={ici_context_parallelism},"
            f" context_parallel_load_balance={context_parallel_load_balance},"
            f" ici_expert_parallelism={ici_expert_parallelism}."
        ),
    )

  @parameterized.product(
      per_device_batch_size=[2.0],
      num_heads=[128],
      sa_block_size=[1024],
      splash_type=["jax"],
      scoped_vmem_multiplier=[0.8],
      force_q_layout=[True],
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_fsdp(
      self,
      per_device_batch_size,
      num_heads,
      sa_block_size,
      splash_type,
      scoped_vmem_multiplier,
      force_q_layout,
  ):
    """Test equivalence between dot_product and flash attention in fsdp (expert parallelism) mode."""
    self._run_flash_attention_fsdp_test(
        per_device_batch_size,
        num_heads,
        sa_block_size,
        splash_type,
        scoped_vmem_multiplier,
        force_q_layout,
    )

  @pytest.mark.tpu_only
  def test_tpu_flash_attention_fsdp_manual(self):
    """Manual test for flash attention fsdp with parameters from command line.

    This test is only run if --run_manual_test is provided.
    """
    if not _RUN_MANUAL_TEST.value:
      self.skipTest("Skipping manual test. Use --run_manual_test to enable.")
      return

    self._run_flash_attention_fsdp_test(
        per_device_batch_size=_PER_DEVICE_BATCH_SIZE.value,
        num_heads=_NUM_HEADS.value,
        sa_block_size=_SA_BLOCK_SIZE.value,
        splash_type=_SPLASH_TYPE.value,
        scoped_vmem_multiplier=_SCOPED_VMEM_MULTIPLIER.value,
        force_q_layout=_FORCE_Q_LAYOUT.value,
    )


if __name__ == "__main__":
  googletest.main()
