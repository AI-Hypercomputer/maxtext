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

"""Tests for the MPMD pipeline's hyperparameters and their flags (no JAX)."""

import dataclasses
import os
import tempfile
import unittest

from absl import flags

from maxtext.experimental.mpmd.partir import hparams as hp


def parse(argv):
  """Parses `argv` with the hyperparameter flags registered on a fresh FlagValues."""
  flag_values = flags.FlagValues()
  hp.define_flags(flag_values)
  flag_values(["prog", *argv])
  return flag_values


class ScheduleTest(unittest.TestCase):

  def test_aliases_and_oss_spellings(self):
    for raw, want in [
        ("1F1B", "ONE_FWD_ONE_BWD"),
        ("gpipe", "GPIPE"),
        ("GPipe", "GPIPE"),
        ("ZB-H1", "ZERO_BUBBLE_H1"),
        ("ZeroBubbleH1", "ZERO_BUBBLE_H1"),
        ("ZeroBubbleH2HalfTxLatency", "ZERO_BUBBLE_H2_HALF_TX_LATENCY"),
        ("GPipeBut1F1BForLastMesh", "GPIPE_BUT_1F1B_FOR_LAST_MESH"),
        ("Circular", "CIRCULAR"),
        ("None", "NONE"),
    ]:
      self.assertEqual(hp.normalize_schedule(raw), want, raw)
    with self.assertRaisesRegex(ValueError, "Unknown pipeline schedule"):
      hp.normalize_schedule("2F2B")

  def test_schedule_help_lists_only_usable_schedules(self):
    text = parse([])["pipeline_schedule"].help
    self.assertIn("ONE_FWD_ONE_BWD", text)
    for name in hp.UNSUPPORTED_SCHEDULES:
      self.assertNotIn(f"{name},", text)
      self.assertNotIn(f"{name} (", text)

  def test_every_schedule_has_an_oss_name(self):
    self.assertEqual(set(hp.OSS_SCHEDULE_NAMES), set(hp.PIPELINE_SCHEDULES))
    # The OSS spellings round-trip through normalize_schedule.
    for name, oss in hp.OSS_SCHEDULE_NAMES.items():
      self.assertEqual(hp.normalize_schedule(oss), name)


class HparamsTest(unittest.TestCase):

  def test_validation(self):
    with self.assertRaisesRegex(ValueError, "divisible by num_stages"):
      hp.Hparams(num_layers=6, num_stages=4)
    with self.assertRaisesRegex(ValueError, "must have length"):
      hp.Hparams(num_layers=8, num_stages=4, stage_layer_counts="1,2,3")
    with self.assertRaisesRegex(ValueError, "must equal"):
      hp.Hparams(num_layers=8, num_stages=2, stage_layer_counts="3,4")
    self.assertEqual(hp.Hparams(num_layers=8, num_stages=2, stage_layer_counts="3,5").stage_layer_counts, (3, 5))
    with self.assertRaisesRegex(ValueError, "moe_dispatch"):
      hp.Hparams(expert=2, moe_dispatch="all_to_all")
    for bad in ("dtype", "=float32", " dtype=float32"):
      with self.assertRaisesRegex(ValueError, "key=value"):
        hp.Hparams(maxtext_overrides=(bad,))
    self.assertEqual(hp.Hparams(maxtext_overrides="dtype=float32").maxtext_overrides, ("dtype=float32",))
    # A MaxText key a hyperparameter owns must be set through that flag.
    for owned, flag in (
        ("learning_rate=1e-5", "--learning_rate"),
        ("per_device_batch_size=4", "--per_device_batch_size"),
        ("ici_fsdp_parallelism=2", "--fsdp"),
        ("scan_layers=true", "one computation per layer"),
        ("remat_policy=full", "--remat and --remat_policy"),
    ):
      with self.assertRaisesRegex(ValueError, flag):
        hp.Hparams(maxtext_overrides=(owned,))
    with self.assertRaisesRegex(ValueError, "--num_microbatches"):
      hp.Hparams(maxtext_overrides=("gradient_accumulation_steps=4",))
    with self.assertRaisesRegex(ValueError, "YaRN"):
      hp.Hparams(model_name="deepseek3-tiny", maxtext_overrides=("rope_type=yarn",))
    self.assertEqual(
        hp.Hparams(model_name="simple", maxtext_overrides=("rope_type=llama3.1",)).maxtext_overrides,
        ("rope_type=llama3.1",),
    )
    for schedule in ("ParallelPipelinesWithWrapAround", "Circular", "CIRCULAR_WITH_REVERSED_BACKWARD"):
      with self.assertRaisesRegex(ValueError, "not supported"):
        hp.Hparams(pipeline_schedule=schedule)

  def test_only_the_supported_decoder_blocks(self):
    for model_name, block in (
        ("simple", "simple_mlp"),
        ("deepseek3-tiny", "deepseek"),
        ("llama3.1-8b", "llama2"),
        ("mistral-7b", "mistral"),
        ("mixtral-8x7b", "mixtral"),
        ("qwen3-0.6b", "qwen3"),
        ("qwen3-30b-a3b", "qwen3_moe"),
    ):
      self.assertEqual(hp.Hparams(model_name=model_name).decoder_block, block)
    for model_name, block in (("gpt3-52k", "gpt3"), ("deepseek4-tiny", "deepseek4"), ("gemma4-e2b", "gemma4_small")):
      with self.assertRaisesRegex(ValueError, f"decoder_block={block}, which the MPMD pipeline does not build"):
        hp.Hparams(model_name=model_name)

  def test_a_model_config_without_a_decoder_block_is_refused(self):
    """MaxText builds these with its m3 backend; they were once built as the toy simple_mlp block under their name."""
    for model_name in ("granite-3.0-1b-a400m", "glm5.3-flash"):
      with self.assertRaisesRegex(ValueError, f"model_name={model_name}: its MaxText config names no decoder_block"):
        hp.Hparams(model_name=model_name, maxtext_overrides=("logits_via_embedding=false", "mhc_expansion_rate=1"))

  def test_a_missing_model_config_is_refused(self):
    """pyconfig would go on with base.yml's model under the name; the launcher refuses it before any slice is up."""
    for overrides in ((), ("rope_type=yarn",)):  # the override check reads the model config too
      with self.assertRaisesRegex(FileNotFoundError, "would silently fall back"):
        hp.Hparams(model_name="llama3.1-8b-does-not-exist", maxtext_overrides=overrides)

  def test_remat_policies(self):
    """MaxText's resolver takes each (models_test checks it); any other name fails before a model is built."""
    for policy in hp.REMAT_POLICIES:
      self.assertEqual(hp.Hparams(remat=True, remat_policy=policy).mpmd_remat_policy, policy)
    # MaxText's offloading and quantization policies, names only the MPMD model knew, none, and a typo.
    for policy in ("qkv_proj_offloaded", "custom", "minimal_with_quantization", "alternate", "dots", "none", "minimum"):
      with self.assertRaisesRegex(ValueError, f"remat_policy={policy} is not one the MPMD model checkpoints"):
        hp.Hparams(remat=True, remat_policy=policy)

  def test_job_args_cover_every_field(self):
    """The launcher can forward every hyperparameter."""
    fields = {f.name for f in dataclasses.fields(hp.Hparams)}
    self.assertEqual(fields, set(hp.FORWARDED_FLAGS))

  def test_job_args_round_trip(self):
    want = hp.Hparams(
        model_name="deepseek3-tiny",
        num_layers=12,
        num_stages=3,
        fsdp=2,
        num_microbatches=6,
        pipeline_schedule="ZB-H1",
        remat=True,
        stage_layer_counts="3,4,5",
        per_device_batch_size=1.5,
        maxtext_overrides=("dtype=float32", "attention=dot_product"),
    )
    self.assertEqual(hp.from_flags(parse(hp.job_argv(hp.to_job_args(want)))), want)

  def test_flags_are_defined_once_where_asked(self):
    flag_values = flags.FlagValues()
    hp.define_flags(flag_values)
    hp.define_flags(flag_values)  # a second call is a no-op
    self.assertIn("num_microbatches", flag_values)

  def test_real_models_keep_their_dimensions_unless_given(self):
    self.assertEqual(hp.from_flags(parse(["--model_name=llama3.1-8b"])).emb_dim, -1)
    self.assertEqual(hp.from_flags(parse(["--model_name=llama3.1-8b", "--emb_dim=256"])).emb_dim, 256)
    self.assertEqual(hp.from_flags(parse(["--model_name=simple"])).emb_dim, hp.Hparams().emb_dim)
    self.assertEqual(hp.from_flags(parse(["--pp=2", "--num_layers=4"])).num_stages, 2)

  def test_float32_gmm_tiles(self):
    self.assertEqual(hp.float32_gmm_tiles(hp.Hparams()), {})
    tiles = hp.float32_gmm_tiles(hp.Hparams(maxtext_overrides=("dtype=float32",)))
    self.assertEqual(len(tiles), 12)
    self.assertEqual(set(tiles.values()), {512})
    self.assertEqual(hp.float32_gmm_tiles(hp.Hparams(maxtext_overrides=("dtype=float32", "dtype=bfloat16"))), {})
    self.assertEqual(hp.override_value(hp.Hparams(maxtext_overrides=("a=1", "b=2", "a=3")), "a"), "3")

  def test_moe_dispatch_overrides(self):
    self.assertEqual(hp.moe_dispatch_overrides(hp.Hparams()), {})
    ring = hp.moe_dispatch_overrides(hp.Hparams(expert=4, moe_dispatch="ring_of_experts"))
    self.assertTrue(ring["use_ring_of_experts"])
    self.assertEqual(ring["ici_expert_parallelism"], 4)
    self.assertEqual(
        set(ring), {"ici_expert_parallelism", "use_ring_of_experts", "use_ragged_sort", "num_moe_token_chunks"}
    )

  def test_microbatch_split(self):
    h = hp.Hparams(num_microbatches=4, per_device_batch_size=2.0)
    hp.check_microbatch_split(h, devices_per_stage=4, data_parallel_size=2)  # 8 rows, microbatches of 2 over 2
    with self.assertRaisesRegex(ValueError, "2 sequences per microbatch, which must be a multiple of the 4 devices"):
      hp.check_microbatch_split(h, devices_per_stage=4, data_parallel_size=4)
    with self.assertRaisesRegex(ValueError, "Global batch 6"):
      hp.check_microbatch_split(h, devices_per_stage=3, data_parallel_size=1)  # 6 rows in 4 microbatches

  def test_ragged_buffer_note(self):
    self.assertIsNone(hp.ragged_buffer_note(hp.Hparams()))
    self.assertIn("--ragged_buffer_factor=2", hp.ragged_buffer_note(hp.Hparams(expert=2)))
    self.assertIsNone(hp.ragged_buffer_note(hp.Hparams(expert=2, ragged_buffer_factor=2.0)))

  def test_device_counts(self):
    self.assertEqual(
        [hp.devices_in_device_type(t) for t in ("v5p-16", "v6e-16", "tpu7x-4x4x4", "tpu7x-2x2x1", "v5p", "x-y")],
        [8, 16, 128, 8, 0, 0],
    )

  def test_read_xla_flags_file(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "x.flags")
      with open(path, "w", encoding="utf-8") as f:
        f.write("# comment\n\n--xla_a=true\n  --xla_b=65536  \n--xla_c=x=y\n")
      self.assertEqual(hp.read_xla_flags_file(path), {"xla_a": "true", "xla_b": "65536", "xla_c": "x=y"})
      for bad in ("xla_a=true", "--xla_a", "--=1"):
        with open(path, "w", encoding="utf-8") as f:
          f.write(bad + "\n")
        with self.assertRaisesRegex(ValueError, "expected --key=value"):
          hp.read_xla_flags_file(path)


if __name__ == "__main__":
  unittest.main()
