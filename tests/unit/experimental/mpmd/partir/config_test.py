# Copyright 2026 Google LLC
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

"""Tests for the MaxText settings the MPMD pipeline derives (config.py)."""

import unittest

import jax

from maxtext.common.common_types import DecoderBlockType
from maxtext.experimental.mpmd.partir import config
from maxtext.experimental.mpmd.partir import hparams as hp

BASE = {"model_name": "deepseek3-tiny", "num_layers": 8, "num_stages": 4}


def setUpModule():
  # MaxText checks its own pipeline's batch against the devices: 4 stages need 4 (per_device_batch_size 2 x 1 device
  # is not divisible into 4 pipeline microbatches).
  if len(jax.devices()) < 4:
    raise unittest.SkipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")


class RematTest(unittest.TestCase):

  def test_pipeline_parallelism_without_remat(self):
    """MaxText refuses remat_policy=none with ici_pipeline_parallelism > 1; the MPMD model keeps its own choice."""
    hparams = hp.Hparams(**BASE)
    cfg = config.build_maxtext_config(hparams)
    self.assertEqual(cfg.ici_pipeline_parallelism, 4)
    self.assertEqual((cfg.remat_policy, hparams.mpmd_remat_policy), ("full", "none"))
    hparams = hp.Hparams(**BASE, remat=True, remat_policy="minimal")
    cfg = config.build_maxtext_config(hparams)
    self.assertEqual((cfg.remat_policy, hparams.mpmd_remat_policy), ("minimal", "minimal"))
    # Only MaxText's own (typed) fields: a key added to get_keys() would not survive cfg.replace().
    self.assertFalse([key for key in cfg.get_keys() if key.startswith("mpmd_")])


class Float32TilesTest(unittest.TestCase):

  def test_float32_halves_the_gmm_tiles_and_overrides_still_win(self):
    bf16 = config.build_maxtext_config(hp.Hparams(**BASE))
    self.assertEqual((bf16.wi_tile_fwd_embed_dim, bf16.wo_tile_drhs_mlp_dim), (1024, 1024))
    f32 = config.build_maxtext_config(hp.Hparams(**BASE, maxtext_overrides=("dtype=float32",)))
    self.assertEqual(str(f32.dtype), "float32")
    for weight in ("wi", "wo"):
      for kernel in ("fwd", "dlhs", "drhs"):
        self.assertEqual(getattr(f32, f"{weight}_tile_{kernel}_embed_dim"), 512)
        self.assertEqual(getattr(f32, f"{weight}_tile_{kernel}_mlp_dim"), 512)
        self.assertEqual(getattr(f32, f"{weight}_tile_{kernel}_batch_seq"), 512)
    mine = config.build_maxtext_config(
        hp.Hparams(**BASE, maxtext_overrides=("dtype=float32", "wi_tile_fwd_embed_dim=256"))
    )
    self.assertEqual((mine.wi_tile_fwd_embed_dim, mine.wo_tile_fwd_embed_dim), (256, 512))


class BuildableTest(unittest.TestCase):

  def test_every_supported_decoder_block_builds_its_config(self):
    for model_name, block in (
        ("simple", DecoderBlockType.SIMPLE_MLP),
        ("deepseek3-tiny", DecoderBlockType.DEEPSEEK),
        ("llama3.1-8b", DecoderBlockType.LLAMA2),
        ("mistral-7b", DecoderBlockType.MISTRAL),
        ("mixtral-8x7b", DecoderBlockType.MIXTRAL),
        ("qwen3-30b-a3b", DecoderBlockType.QWEN3_MOE),
    ):
      cfg = config.build_maxtext_config(hp.Hparams(model_name=model_name, num_layers=4, num_stages=2))
      self.assertEqual(cfg.decoder_block, block, model_name)

  def test_refuses_what_the_mpmd_model_does_not_build(self):
    tied = {"model_name": "qwen3-0.6b", "num_layers": 4, "num_stages": 4}
    with self.assertRaisesRegex(ValueError, "logits_via_embedding=false"):
      config.build_maxtext_config(hp.Hparams(**tied))
    untied = config.build_maxtext_config(hp.Hparams(**tied, maxtext_overrides=("logits_via_embedding=false",)))
    self.assertFalse(untied.logits_via_embedding)
    self.assertEqual(untied.decoder_block, DecoderBlockType.QWEN3)
    # Each of _check_buildable's refusals: one setting each, the first's key named in the error.
    for overrides in (
        ("use_m3_model=true",),
        ("trainable_position_size=16",),
        ("mhc_expansion_rate=2",),
        # The layers are built unquantized, while spmd_baseline's MaxText run would be quantized.
        ("quantization=int8",),
        ("shard_mode=explicit",),
        ("prefetch_fsdp_weights=true",),
        ("lm_head_fsdp_as_vocab_tp=true",),
        ("lm_head_weight_grad_in_kernel_order=true",),
        ("tie_lm_head_cotangents=true",),
        ("cast_logits_to_fp32=false",),
        # fp8 weights: logits_dense quantized, or left in the compute dtype while the rest is fp8.
        ("weight_dtype=float8_e4m3fn",),
        ("weight_dtype=float8_e4m3fn", "unquantized_modules=[logits_dense]"),
    ):
      key = overrides[0].partition("=")[0]
      with self.assertRaisesRegex(ValueError, f"does not build {key}=", msg=overrides):
        config.build_maxtext_config(
            hp.Hparams(model_name="simple", num_layers=4, num_stages=2, maxtext_overrides=overrides)
        )

  def test_refuses_what_the_mpmd_step_leaves_out(self):
    """The MPMD step trains on the token cross-entropy alone, without an MTP block or a sinusoidal embedding."""

    def build(model_name, overrides):
      return config.build_maxtext_config(
          hp.Hparams(model_name=model_name, num_layers=4, num_stages=2, maxtext_overrides=overrides)
      )

    # deepseek3-tiny's MLA with the sparse indexer, small enough for a sequence of 512 (and for MaxText's checks).
    indexer = ("use_indexer=true", "attention=dot_product", "indexer_topk=256", "indexer_n_heads=4")
    for model_name, overrides, key, off in (
        ("deepseek3-tiny", ("mtp_num_layers=1",), "mtp_num_layers", "0"),
        ("simple", ("use_untrainable_positional_embedding=true",), "use_untrainable_positional_embedding", "false"),
        ("deepseek3-tiny", ("load_balance_loss_weight=0.01",), "load_balance_loss_weight", "0"),
        ("simple", ("z_loss_multiplier=1e-4",), "z_loss_multiplier", "0"),
        ("deepseek3-tiny", ("routed_bias_update_rate=0.001",), "routed_bias_update_rate", "0"),
        ("deepseek3-tiny", indexer + ("indexer_loss_scaling_factor=0.01",), "indexer_loss_scaling_factor", "0"),
    ):
      with self.subTest(key=key):
        with self.assertRaisesRegex(
            ValueError, rf"\({key}=[^)]*\), which the MPMD step leaves out.* Pass --maxtext_overrides={key}={off} to"
        ):
          build(model_name, overrides)
        build(model_name, overrides + (f"{key}={off}",))  # the hint's override builds
    # The load-balance loss is in qwen3-235b-a22b's own config.
    with self.assertRaisesRegex(ValueError, r"MoE load-balance loss \(load_balance_loss_weight=0.001\)"):
      build("qwen3-235b-a22b", ())
    self.assertEqual(build("qwen3-235b-a22b", ("load_balance_loss_weight=0",)).decoder_block, DecoderBlockType.QWEN3_MOE)
    with self.assertRaisesRegex(ValueError, r"\(mtp_num_layers=1\) and the z-loss .* to train without them"):
      build("deepseek3-tiny", ("mtp_num_layers=1", "z_loss_multiplier=1e-4"))
    # max_utils.cross_entropy_with_logits adds z_loss_multiplier * log(z)^2 whatever its sign.
    with self.assertRaisesRegex(ValueError, r"the z-loss \(z_loss_multiplier=-0\.0001\), which"):
      build("simple", ("z_loss_multiplier=-1e-4",))
    # Only where MaxText's trainer adds the term: a dense model has no MoE loss, a fixed routing bias no updates, and
    # the indexer's loss needs the indexer.
    build("llama3.1-8b", ("load_balance_loss_weight=0.01",))
    build("deepseek3-tiny", ("routed_bias=false", "routed_bias_update_rate=0.001"))
    build("deepseek3-tiny", ("indexer_loss_scaling_factor=0.01",))

  def test_the_decoder_block_is_maxtexts(self):
    """A named model's block is the one pyconfig resolves from its config; config.py names only simple_mlp's."""
    mistral = hp.Hparams(model_name="mistral-7b", num_layers=4, num_stages=2)
    with self.assertRaisesRegex(ValueError, "MaxText resolved decoder_block=llama2, but"):
      config.build_maxtext_config(mistral, overrides={"decoder_block": "llama2"})
    self.assertEqual(config.build_maxtext_config(mistral).decoder_block, DecoderBlockType.MISTRAL)


if __name__ == "__main__":
  unittest.main()
