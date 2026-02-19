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
"""Test util for attention tests."""

import sys

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from MaxText import pyconfig
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.common.common_types import AttentionType, DECODING_ACTIVE_SEQUENCE_INDICATOR, EP_AS_CONTEXT, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.layers.attention_mla import MLA
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import maybe_shard_with_name
from tests.utils.test_helpers import get_test_config_path


class MLATestBase(parameterized.TestCase):
  """Test base for MLATest."""

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": 128,
      "max_prefill_predict_length": 16,
      "attention_type": AttentionType.MLA.value,
      "head_dim": 192,
      "q_lora_rank": 10,
      "kv_lora_rank": 20,
      "qk_nope_head_dim": 128,
      "qk_rope_head_dim": 64,
      "v_head_dim": 192,
      "dtype": "float32",
      "mla_naive_kvcache": False,  # TODO: Test both naive/non-naive modes once b/485997160 is resolved.
  }

  def setUp(self):
    """Initializes the configuration for each test"""
    super().setUp()
    config_args = dict(self.config_arguments)
    if is_decoupled():  # TODO(gulsumgudukbay): remove this after jax is updated.
      # Older/newer JAX versions may not recognize this flag; ignore if absent.
      try:
        jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
      except AttributeError:
        pass
      # In decoupled mode, adapt mesh/ICI parallelism to local devices so
      # fill_unspecified_mesh_axes matches the available device count.
      config_args.setdefault("mesh_axes", ["data"])
      config_args.setdefault("ici_data_parallelism", -1)
    else:
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)

    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_args,
    )
    self.cfg = config
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

  def init_mla(self, config_arguments, rope_type):
    """Helper function to initialize MLA with different model names."""
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        rope_type=rope_type,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    dummy_inputs_q = jnp.ones(
        (
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        )
    )
    dummy_inputs_kv = jnp.ones(
        (
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        )
    )

    mla = MLA(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=cfg.attention_type,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    return cfg, mla

  def get_data(self, cfg, dtype):
    """get data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        ),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(
        self.rng,
        (cfg.global_batch_size_to_train_on, cfg.max_target_length),
        0,
        4,
    )
    # decoder_segment_ids = None
    decoder_positions = jax.random.randint(
        self.rng,
        (cfg.global_batch_size_to_train_on, cfg.max_target_length),
        0,
        cfg.max_target_length,
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, cfg, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        ),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(cfg.max_target_length, dtype=jnp.int32) for _ in range(cfg.global_batch_size_to_train_on)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions


def forward_with_context_expert_parallelism(
    cfg_cp,
    mesh_cp,
    attention_cp,
    lnx,
    decoder_segment_ids,
    decoder_positions,
):
  """Get logits from attention under context/expert parallelism."""
  # If load balanced cp, shuffle along seq dim for input
  # This corresponds to the pre-shuffle step in training
  context_parallel_size = cfg_cp.context_parallel_size
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    batch = {
        "inputs": lnx,
        "inputs_segmentation": decoder_segment_ids,
        "inputs_position": decoder_positions,
    }
    with mesh_cp:
      reordered_batch = maxtext_utils.get_reorder_callable(context_parallel_size, ShardMode.AUTO)(batch)
    lnx = reordered_batch["inputs"]
    decoder_segment_ids = reordered_batch["inputs_segmentation"]
    decoder_positions = reordered_batch["inputs_position"]
  # apply attention with sharding
  with mesh_cp, nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
    if cfg_cp.expert_shard_attention_option == EP_AS_CONTEXT:
      batch_axis = "activation_batch_no_exp"
      length_axis = "activation_length"
    else:
      batch_axis = "activation_batch"
      length_axis = "activation_length_no_exp"
    lnx_spec = nn_partitioning.logical_to_mesh_axes(
        (batch_axis, length_axis, "activation_embed"),
        nn_partitioning.get_axis_rules(),
    )
    pos_spec = nn_partitioning.logical_to_mesh_axes((batch_axis, length_axis), nn_partitioning.get_axis_rules())
    lnx_sharding = NamedSharding(mesh_cp, lnx_spec)
    pos_sharding = NamedSharding(mesh_cp, pos_spec)

    lnx = jax.device_put(lnx, lnx_sharding)
    decoder_segment_ids = jax.device_put(decoder_segment_ids, pos_sharding)
    decoder_positions = jax.device_put(decoder_positions, pos_sharding)

    attention_cp_output, _ = attention_cp(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    attention_cp_output = attention_cp_output[0] if isinstance(attention_cp_output, tuple) else attention_cp_output

    # All-gather before re-shuffle to avoid re-order sharding confusion
    repeat_sharding = NamedSharding(mesh_cp, P())
    attention_cp_output = maybe_shard_with_name(attention_cp_output, repeat_sharding, shard_mode=cfg_cp.shard_mode)

  # If load balanced cp, de-shuffle and gather along seq dim for output
  # Note training does not need post-shuffle. Since the target seq is also pre-shuffled, the loss remains correct
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    attention_cp_output = max_utils.reorder_sequence(
        tensor=attention_cp_output,
        cp_size=context_parallel_size,
        seq_dim=1,
        to_contiguous=True,
    )
  return attention_cp_output
