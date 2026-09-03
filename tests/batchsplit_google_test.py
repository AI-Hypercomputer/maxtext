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

"""Tests for equivalence between batch-split and non batch-split schedules."""

import os
import os.path
import tempfile
import unittest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.layers import decoders
from maxtext.layers import embeddings
from maxtext.layers import quantizations
from maxtext.trainers.pre_train import train
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils import test_helpers

from google3.testing.pybase import googletest

get_test_config_path = test_helpers.get_test_config_path
train_main = train.main
gettempdir = tempfile.gettempdir


class DeepSeekDecoderTest(parameterized.TestCase):
  """Tests the DeepSeek decoder with the batch-split schedule."""

  def assert_close(self, a, b, rtol=3e-01, atol=3e-01):
    self.assertTrue(
        jax.numpy.allclose(
            a,
            b,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ),
        msg=(f"The following two arrays are not close\n{a=}\n{b=}\n" f"total difference is {jnp.sum(jnp.abs(a - b))=}"),
    )

  def assert_norm_close(self, a, b, rtol=2.5e-01):
    if jnp.linalg.norm(a) > 0:
      self.assertLess(
          jnp.linalg.norm(a - b) / jnp.linalg.norm(a),
          rtol,
          msg=(
              "The following two arrays are not close in norm\n"
              f"{jnp.linalg.norm(a - b)=}\n{jnp.linalg.norm(a)=}\n"
              f"{a.shape=}\n{b.shape=}\n"
              f"{a=}\n{b=}"
          ),
      )

  def assert_no_nans(self, xs):
    for x in jax.tree_util.tree_flatten(xs)[0]:
      self.assertFalse(
          jnp.any(jnp.isnan(x)),
          msg=f"NaN found in {x=}",
      )

  def compare_gradients(self, grad0, grad1, comparison_fn):
    leaves_a, _ = jax.tree_util.tree_flatten_with_path(grad0)
    leaves_b, _ = jax.tree_util.tree_flatten_with_path(grad1)

    for (path, a), (_, b) in zip(leaves_a, leaves_b):
      grad_path = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path[1:-1])

      with self.subTest(name=f"grad_{grad_path}"):
        comparison_fn(a, b)

  def get_config_dict(
      self,
      use_batch_split_config,
      quantization="",
      use_qwix_quantization=False,
      use_manual_quantization=False,
  ):
    """Returns config dictionary for testing."""
    if use_batch_split_config:
      model_name = "deepseek3-671b-batchsplit"
      ici_fsdp_parallelism = 2
      ici_expert_parallelism = 2
      ici_data_parallelism = 2
      shard_mode = "explicit"
      use_tokamax_gmm = True
      use_tokamax_splash = False
    else:
      model_name = "deepseek3-671b"
      ici_fsdp_parallelism = 8
      ici_expert_parallelism = 1
      ici_data_parallelism = 1
      shard_mode = "auto"
      use_tokamax_gmm = False
      use_tokamax_splash = True
    return dict(  # pylint: disable=use-dict-literal
        run_name="deepseek_decoder_output_and_grad_equivalence_test",
        enable_checkpointing=False,
        model_name=model_name,
        dtype="bfloat16",
        override_model_config=True,
        base_num_decoder_layers=5,
        first_num_dense_layers=1,
        base_emb_dim=512,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        base_mlp_dim=512,
        base_moe_mlp_dim=512,
        num_experts=32,
        num_experts_per_tok=8,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=128,
        v_head_dim=128,
        megablox=True,
        sparse_matmul=True,
        use_tokamax_gmm=use_tokamax_gmm,
        use_tokamax_splash=use_tokamax_splash,
        sa_use_fused_bwd_kernel=True,
        use_custom_sort_vjp=True,
        use_ring_of_experts=True,
        max_target_length=128,
        scan_layers=True,
        ici_fsdp_parallelism=ici_fsdp_parallelism,
        ici_expert_parallelism=ici_expert_parallelism,
        ici_data_parallelism=ici_data_parallelism,
        use_batch_split_schedule=use_batch_split_config,
        per_device_batch_size=2,
        batch_split_factor=2,
        shard_mode=shard_mode,
        param_scan_axis=1,
        n_routing_groups=8,
        topk_routing_group=4,
        sharding_tolerance=0.25,
        quantization=quantization,
        use_qwix_quantization=use_qwix_quantization,
        use_manual_quantization=use_manual_quantization,
        weight_quantization_calibration_method="fixed,-224,224",
        act_quantization_calibration_method="fixed,-224,224",
    )

  def get_config(
      self,
      use_batch_split_config,
      quantization="",
  ):
    config_dict = self.get_config_dict(use_batch_split_config, quantization)
    return pyconfig.initialize(
        [
            None,
            os.path.join(maxtext_globals.MAXTEXT_PKG_DIR, "configs", "base.yml"),
        ],
        **config_dict,
    )

  @unittest.skip("Skipped due to test failure, tracked in b/542183478")
  def test_deepseek_decoder_output_and_grad_equivalence(self):
    cfg = self.get_config(use_batch_split_config=False)
    quant = quantizations.configure_quantization(cfg)
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    inputs = jax.random.randint(
        rng_hidden_states,
        (
            int(cfg.per_device_batch_size) * jax.device_count(),
            cfg.max_target_length,
        ),
        minval=0,
        maxval=cfg.vocab_size,
        dtype=jnp.int32,
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(inputs.shape[1], dtype=jnp.int32),
        (inputs.shape[0], inputs.shape[1]),
    )

    def get_value_and_grad(cfg, inputs, dpos, dseg=None, axis_type=jax.sharding.AxisType.Auto):
      devices_array = maxtext_utils.create_device_mesh(cfg)
      mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes, axis_types=(axis_type,) * len(cfg.mesh_axes))
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model = decoders.Decoder(
            config=cfg,
            mesh=mesh,
            quant=quant,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        embedding = embeddings.Embed(
            num_embeddings=cfg.vocab_size,
            num_features=cfg.base_emb_dim,
            config=cfg,
            mesh=mesh,
            dtype=cfg.dtype,
            attend_dtype=cfg.dtype,
            rngs=nnx.Rngs(rng_model, params=0),
        )
        embedding.embedding.value = jax.device_put(
            embedding.embedding.value,
            sharding.create_sharding(mesh, ("vocab", "embed_vocab"), rules=sharding.get_logical_axis_rules()),
        )
        variables = model.init(
            rng_model,
            shared_embedding=embedding,
            decoder_input_tokens=inputs,
            decoder_positions=decoder_positions,
            decoder_segment_ids=None,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        inputs = jax.device_put(
            inputs,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(tuple(cfg.data_sharding[0]), None)),
        )

        # We need to explicitly reshard weights for deepseek_batchsplit since
        # it uses explicit shard mode.
        def reshard_weights(path, w):
          if str(path[-3]) == "['wq_a']":
            spec = ("embed", None, "q_lora_up_proj")
          elif str(path[-3]) == "['wkv_a']":
            spec = ("embed", None, "kv_lora_up_proj")
          elif str(path[-3]) == "['wq_b']":
            spec = ("q_lora", None, "q_heads", "kv")
          elif str(path[-3]) == "['wkv_b']":
            spec = ("kv_lora", None, "kv_heads", "kv_head_dim")
          elif str(path[-3]) == "['out']":
            spec = ("heads", None, "kv", "embed")
          elif str(path[-3]) == "['gate']" and str(path[-2]) == "['kernel']":
            spec = ("embed", None, None)
          # Routed expert weights
          elif str(path[-2]) in ["['wi_0']", "['wi_1']"]:
            spec = ("expert_only", None, "embed_moe", None)
          elif str(path[-2]) == "['wo']":
            spec = ("expert_only", None, None, "embed_moe")
          # Shared expert weights
          elif str(path[-3]) in ["['wi_0']", "['wi_1']"]:
            spec = ("embed", None, "mlp")
          elif str(path[-3]) == "['wo']":
            spec = ("mlp", None, "embed")
          elif any("logits_dense" in str(p) for p in path):
            spec = ("embed_vocab", "vocab")
          else:
            spec = None
          return jax.device_put(
              w,
              sharding.create_sharding(mesh, spec, rules=sharding.get_logical_axis_rules()) if spec is not None else w.sharding,
          )

        params = jax.tree_util.tree_map_with_path(reshard_weights, variables["params"])
        variables = {
            **variables,
            "params": params,
        }

        def loss_fn(model, variables, inputs, dpos, dseg):
          _, outputs, _ = model.apply(
              variables,
              shared_embedding=embedding,
              decoder_input_tokens=inputs,
              decoder_positions=dpos,
              decoder_segment_ids=dseg,
              deterministic=True,
              model_mode=common_types.MODEL_MODE_TRAIN,
          )
          return jnp.sum(outputs), outputs

        return jax.jit(jax.grad(loss_fn, has_aux=True, argnums=1), static_argnames=["model"])(
            model, variables, inputs, dpos, dseg
        )

    ref_var_grad, ref_outputs = get_value_and_grad(cfg, inputs, decoder_positions)
    ref_var_grad = jax.device_get(ref_var_grad)
    ref_outputs = jax.device_get(ref_outputs)

    bs_var_grad, bs_outputs = get_value_and_grad(
        self.get_config(use_batch_split_config=True),
        inputs,
        decoder_positions,
        axis_type=jax.sharding.AxisType.Explicit,
    )
    bs_var_grad = jax.device_get(bs_var_grad)
    bs_outputs = jax.device_get(bs_outputs)

    with self.subTest(name="no_nans_ref_outputs"):
      self.assert_no_nans(ref_outputs)
    with self.subTest(name="no_nans_ref_var_grad"):
      self.assert_no_nans(ref_var_grad)
    with self.subTest(name="no_nans_batch_split_outputs"):
      self.assert_no_nans(bs_outputs)
    with self.subTest(name="no_nans_batch_split_var_grad"):
      self.assert_no_nans(bs_var_grad)

    with self.subTest(name="output_equivalence"):
      self.assert_close(ref_outputs, bs_outputs)
    self.compare_gradients(ref_var_grad, bs_var_grad, self.assert_norm_close)

  @parameterized.named_parameters(
      {
          "testcase_name": "bf16",
          "quantization": "",
          "use_qwix_quantization": False,
          "use_manual_quantization": False,
      },
      {
          "testcase_name": "fp8",
          "quantization": "fp8_full",
          "use_qwix_quantization": True,
          "use_manual_quantization": True,
      },
      {
          "testcase_name": "bf16_with_mtp",
          "quantization": "",
          "use_qwix_quantization": False,
          "use_manual_quantization": False,
          "use_mtp": True,
      },
  )
  def test_batch_split(
      self,
      quantization: str,
      use_qwix_quantization: bool,
      use_manual_quantization: bool,
      use_mtp: bool = False,
  ):
    """Smoke test for deepseek batch split in G3 only."""
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    config_dict = self.get_config_dict(
        use_batch_split_config=True,
        quantization=quantization,
        use_qwix_quantization=use_qwix_quantization,
        use_manual_quantization=use_manual_quantization,
    )
    config_dict["base_output_directory"] = test_tmpdir
    config_dict["metrics_file"] = os.path.join(outputs_dir, "metrics.json")
    argv = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=batch_split_smoke_test",
        "per_device_batch_size=2",
        "max_target_length=1024",
        "dataset_type=synthetic",
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
    ]
    if use_mtp:
      config_dict["mtp_num_layers"] = 1
      config_dict["mtp_loss_scaling_factor"] = 0.1
    for key, value in config_dict.items():
      argv.append(f"{key}={value}")
    train_main(argv)  # pyrefly: ignore[bad-argument-type]


if __name__ == "__main__":
  googletest.main()
