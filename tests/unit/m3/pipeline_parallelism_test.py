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

"""Tests for m3 pipeline parallelism (`infra.pipeline`).

The schedule arithmetic is checked on its own, without devices. The pipelined
Qwen3 stack is checked against the plain model -- logits, loss and every
parameter's gradient -- under a mesh with two stages, for the GPipe and
circular schedules, with and without circular storage, delayed forwarding,
the FSDP all-gather before the loop, scanned stage bodies, and the unrolled
iteration loop. Multi-device cases size themselves from `jax.device_count()`
(CI runs the CPU suite with `--xla_force_host_platform_device_count=4`).
"""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.core.sharding import DefaultSharding, create_mesh
from maxtext.m3.infra import pipeline
from maxtext.m3.infra.named_scopes import add_named_scopes
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model


def tiny_config(num_layers=4, scan_layers=False, param_scan_axis=1, **fields):
  """Only the configuration consumed by the dense Qwen3 model, plus `fields` for the pipeline."""
  return SimpleNamespace(
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
      num_decoder_layers=num_layers,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
      **fields,
  )


def schedule(num_stages=2, num_repeats=1, layers_per_stage=2, num_microbatches=2, **kwargs):
  return pipeline.PipelineSchedule(
      num_stages=num_stages,
      num_repeats=num_repeats,
      layers_per_stage=layers_per_stage,
      num_microbatches=num_microbatches,
      **kwargs,
  )


def flat_params(model):
  return {p: v[...] for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _devices_or_skip(test, minimum: int = 4):
  n = jax.device_count()
  if n < minimum or n % 2:
    test.skipTest(f"needs an even device count of at least {minimum}, found {n}")
  return jax.devices()


class ScheduleTest(parameterized.TestCase):
  """Legacy's arithmetic, pinned."""

  def test_gpipe(self):
    s = schedule(num_stages=2, num_repeats=1, layers_per_stage=2, num_microbatches=4)
    self.assertEqual(s.num_layers, 4)
    self.assertEqual(s.microbatches_per_stage, 2)
    self.assertFalse(s.use_circ_storage)
    self.assertEqual(s.bubble_iterations, 1)
    self.assertEqual(s.iterations_to_complete_first_microbatch, 1)
    self.assertEqual(s.total_iterations, 5)

  def test_circular(self):
    s = schedule(num_stages=2, num_repeats=2, layers_per_stage=1, num_microbatches=4)
    self.assertTrue(s.use_circ_storage)
    self.assertEqual(s.iterations_to_complete_first_microbatch, 5)
    self.assertEqual(s.total_iterations, 9)
    # With one microbatch per stage the last stage flows straight into the first: nothing to park.
    self.assertFalse(schedule(num_stages=2, num_repeats=2, layers_per_stage=1, num_microbatches=2).use_circ_storage)

  def test_delayed_forwarding_doubles_the_bubble(self):
    s = schedule(num_stages=2, num_microbatches=4, delay_activation_forwarding=True)
    self.assertEqual(s.forwarding_delay, 2)
    self.assertEqual(s.bubble_iterations, 2)
    self.assertEqual(s.total_iterations, 6)

  @parameterized.named_parameters(
      ("microbatches_not_a_multiple_of_stages", {"num_microbatches": 3}, "multiple of the number of stages"),
      ("too_few_microbatches_for_delay", {"num_microbatches": 2, "delay_activation_forwarding": True}, "at least 2 x"),
      ("zero_repeats", {"num_repeats": 0}, "num_repeats must be >= 1"),
  )
  def test_rejects(self, kwargs, message):
    with self.assertRaisesRegex(ValueError, message):
      schedule(**kwargs)

  def test_from_config_defaults_match_pyconfig(self):
    """`-1` repeats fills the stack; `-1` microbatches is one per stage, two with delayed forwarding."""
    s = pipeline.schedule_from_config(SimpleNamespace(), num_stages=2, num_layers=4)
    self.assertEqual((s.num_repeats, s.layers_per_stage, s.num_microbatches), (2, 1, 2))
    self.assertEqual((s.fsdp_ag_once, s.scan_iterations, s.remat_iterations), (False, True, True))
    s = pipeline.schedule_from_config(
        SimpleNamespace(pipeline_delay_activation_forwarding=True), num_stages=2, num_layers=4
    )
    self.assertEqual(s.num_microbatches, 4)

  def test_from_config_reads_legacy_fields(self):
    config = SimpleNamespace(
        num_layers_per_pipeline_stage=2,
        num_pipeline_repeats=1,
        num_pipeline_microbatches=8,
        pipeline_fsdp_ag_once=True,
        scan_pipeline_iterations=False,
        set_remat_policy_on_pipeline_iterations=False,
    )
    s = pipeline.schedule_from_config(config, num_stages=2, num_layers=4)
    self.assertEqual(
        s,
        schedule(
            num_stages=2,
            num_repeats=1,
            layers_per_stage=2,
            num_microbatches=8,
            fsdp_ag_once=True,
            scan_iterations=False,
            remat_iterations=False,
        ),
    )

  @parameterized.named_parameters(
      ("layers_do_not_divide", SimpleNamespace(num_layers_per_pipeline_stage=3), "do not divide"),
      ("wrong_product", SimpleNamespace(num_pipeline_repeats=3), "is not the 4 layers"),
      ("weight_prefetch_per_repeat", SimpleNamespace(pipeline_fsdp_ag_per_repeat=True), "pipeline_fsdp_ag_per_repeat"),
      ("partial_pipelining", SimpleNamespace(pipeline_parallel_layers=2), "pipeline_parallel_layers"),
  )
  def test_from_config_rejects(self, config, message):
    with self.assertRaisesRegex(ValueError, message):
      pipeline.schedule_from_config(config, num_stages=2, num_layers=4)

  def test_pyconfigs_resolved_default_is_not_refused(self):
    """pyconfig turns `pipeline_parallel_layers=-1` into the layer count, which pipelines the whole stack."""
    s = pipeline.schedule_from_config(SimpleNamespace(pipeline_parallel_layers=4), num_stages=2, num_layers=4)
    self.assertEqual(s.num_layers, 4)
    self.assertEqual(pipeline.unsupported_pipeline_settings(SimpleNamespace()), {})


class StatefulLayer(nnx.Module):
  """A layer with non-parameter state, which the pipeline refuses to carry."""

  def __init__(self):
    self.scale = nnx.Param(jnp.ones(()))
    self.calls = nnx.Variable(jnp.array(0))

  def __call__(self, x, **kwargs):
    self.calls[...] += 1
    return x * self.scale[...]


class PipelinedQwen3Test(parameterized.TestCase):
  """The pipelined stack against the plain model, under a mesh with two stages."""

  def setUp(self):
    """Enters a `stage=2, fsdp=n/2` mesh and fixes the batch every test runs.

    Float32 matmuls run at full precision, so the comparison is about the
    schedule rather than TPU rounding.
    """
    super().setUp()
    self.devices = _devices_or_skip(self)
    self.mesh = create_mesh(ici={"stage": 2, "fsdp": -1}, devices=self.devices)
    self.enter_context(jax.set_mesh(self.mesh))
    self.enter_context(jax.default_matmul_precision("highest"))
    key = jax.random.key(0)
    self.tokens = jax.random.randint(key, (8, 6), 0, 32)
    self.positions = jnp.broadcast_to(jnp.arange(6), (8, 6))
    self.segments = jnp.concatenate([jnp.ones((8, 3), jnp.int32), 2 * jnp.ones((8, 3), jnp.int32)], axis=1)

  def _models(self, config, sched):
    plain = create_qwen3_model(config, rngs=nnx.Rngs(0))
    pipelined = pipeline.apply_pipeline(
        create_qwen3_model(config, rngs=nnx.Rngs(0)),
        schedule=sched,
        sharding=DefaultSharding(),
        scan_layers=config.scan_layers,
        param_scan_axis=config.param_scan_axis,
    )
    return plain, pipelined

  def _logits(self, model):
    return model(self.tokens, self.positions, decoder_segment_ids=self.segments)

  def _loss(self, model):
    return -jnp.mean(jax.nn.log_softmax(self._logits(model))[..., 0])

  @staticmethod
  def _pipelined_leaf(flat, path, sched, config):
    """The pipelined value and leading index holding plain layer `path[3]`'s parameter `path[4:]`."""
    layer = path[3]
    body, within = divmod(layer, sched.layers_per_stage)
    repeat, stage = divmod(body, sched.num_stages)
    index = (repeat, stage) if sched.num_repeats > 1 else (stage,)
    if config.scan_layers:
      value = flat[path[:2] + path[4:]][index]
      return jnp.take(value, within, axis=min(config.param_scan_axis, value.ndim - 1))
    if sched.layers_per_stage == 1:
      return flat[path[:2] + path[4:]][index]
    return flat[path[:3] + (within,) + path[4:]][index]

  @parameterized.named_parameters(
      ("gpipe", {}, {}),
      ("gpipe_more_microbatches", {"num_microbatches": 4}, {}),
      ("gpipe_delayed_forwarding", {"num_microbatches": 4, "delay_activation_forwarding": True}, {}),
      ("gpipe_fsdp_all_gather_once", {"fsdp_ag_once": True}, {}),
      ("gpipe_unrolled_without_remat", {"scan_iterations": False, "remat_iterations": False}, {}),
      ("gpipe_scanned_stages", {}, {"scan_layers": True}),
      ("gpipe_scanned_stages_axis_0", {}, {"scan_layers": True, "param_scan_axis": 0}),
      ("circular", {"num_repeats": 2, "layers_per_stage": 1}, {}),
      ("circular_with_storage", {"num_repeats": 2, "layers_per_stage": 1, "num_microbatches": 4}, {}),
      (
          "circular_delayed_forwarding",
          {"num_repeats": 2, "layers_per_stage": 1, "num_microbatches": 4, "delay_activation_forwarding": True},
          {},
      ),
      (
          "circular_scanned_fsdp_all_gather_once",
          {"num_repeats": 2, "layers_per_stage": 1, "num_microbatches": 4, "fsdp_ag_once": True},
          {"scan_layers": True},
      ),
      (
          "circular_unrolled",
          {"num_repeats": 2, "layers_per_stage": 1, "num_microbatches": 4, "scan_iterations": False},
          {},
      ),
  )
  def test_matches_the_plain_model(self, schedule_kwargs, config_kwargs):
    sched = schedule(**schedule_kwargs)
    config = tiny_config(**config_kwargs)
    plain, pipelined = self._models(config, sched)

    # Same parameters, laid out per stage.
    p_plain, p_pipelined = flat_params(plain), flat_params(pipelined)
    for path, value in p_plain.items():
      if path[:3] == ("decoder", "layers", "layers"):
        np.testing.assert_array_equal(self._pipelined_leaf(p_pipelined, path, sched, config), value, err_msg=str(path))
      else:
        np.testing.assert_array_equal(p_pipelined[path], value)

    expected = self._logits(plain)
    np.testing.assert_allclose(nnx.jit(self._logits)(pipelined), expected, rtol=1e-5, atol=1e-5)

    loss_plain, grad_plain = nnx.value_and_grad(self._loss)(plain)
    loss_pipelined, grad_pipelined = nnx.jit(nnx.value_and_grad(self._loss))(pipelined)
    np.testing.assert_allclose(loss_pipelined, loss_plain, rtol=1e-5)
    g_pipelined = flat_params(grad_pipelined)
    for path, value in flat_params(grad_plain).items():
      if path[:3] == ("decoder", "layers", "layers"):
        actual = self._pipelined_leaf(g_pipelined, path, sched, config)
      else:
        actual = g_pipelined[path]
      np.testing.assert_allclose(actual, value, rtol=2e-4, atol=2e-6, err_msg=str(path))

  def test_stage_weights_are_placed_over_stage(self):
    """The leading axis lands on `stage`; within a stage the model's own placement (FSDP over `embed`) holds."""
    _, pipelined = self._models(tiny_config(), schedule())
    kernel = pipelined.decoder.layers.layers[0].mlp.wi_0.kernel  # two layers per stage: the body is a Sequential
    self.assertEqual(kernel.get_metadata()["out_sharding"], P("stage", ("fsdp", "expert"), "tensor"))
    fsdp = self.mesh.shape["fsdp"]
    self.assertEqual(kernel[...].sharding.shard_shape(kernel[...].shape), (1, 16 // fsdp, 32))
    _, circular = self._models(tiny_config(), schedule(num_repeats=2, layers_per_stage=1))
    kernel = circular.decoder.layers.mlp.wi_0.kernel
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(None, "stage", ("fsdp", "expert"), "tensor"))
    self.assertEqual(kernel[...].shape, (2, 2, 16, 32))

  def test_parameter_paths_keep_one_bodys_layout(self):
    """No wrapper segment: `decoder/layers/<layer param>` with the stage axis in front, as legacy checkpoints have it."""
    _, one_per_stage = self._models(tiny_config(), schedule(num_repeats=2, layers_per_stage=1))
    self.assertIn(("decoder", "layers", "mlp", "wi_0", "kernel"), flat_params(one_per_stage))
    _, two_per_stage = self._models(tiny_config(), schedule())
    self.assertIn(("decoder", "layers", "layers", 1, "mlp", "wi_0", "kernel"), flat_params(two_per_stage))
    _, scanned = self._models(tiny_config(scan_layers=True), schedule())
    self.assertIn(("decoder", "layers", "mlp", "wi_0", "kernel"), flat_params(scanned))
    self.assertEqual(scanned.decoder.layers.mlp.wi_0.kernel[...].shape, (2, 16, 2, 32))

  def test_iterations_are_scanned_and_stages_shifted_by_permute(self):
    _, pipelined = self._models(tiny_config(), schedule())
    jaxpr = str(jax.make_jaxpr(self._logits)(pipelined))
    self.assertIn("scan[", jaxpr)
    self.assertIn("ppermute", jaxpr)
    self.assertIn("remat", jaxpr)
    _, unrolled = self._models(tiny_config(), schedule(scan_iterations=False, remat_iterations=False))
    jaxpr = str(jax.make_jaxpr(self._logits)(unrolled))
    self.assertNotIn("scan[", jaxpr)
    self.assertNotIn("remat", jaxpr)

  def test_fsdp_all_gather_once_constrains_the_weights_without_fsdp(self):
    """With the gather hoisted, the stacked weights are pinned to their placement minus `fsdp` before the loop."""
    _, pipelined = self._models(tiny_config(), schedule(fsdp_ag_once=True))
    jaxpr = jax.make_jaxpr(self._logits)(pipelined)
    specs = [e.params["sharding"].spec for e in jaxpr.jaxpr.eqns if e.primitive.name == "sharding_constraint"]
    self.assertIn(P("stage", "expert", "tensor"), specs)  # `wi_0`: ("embed" -> fsdp x expert, "mlp" -> tensor) minus fsdp
    self.assertIn(P("stage", None), specs)  # norm scales
    _, plain_pipeline = self._models(tiny_config(), schedule())
    jaxpr = jax.make_jaxpr(self._logits)(plain_pipeline)
    specs = [e.params["sharding"].spec for e in jaxpr.jaxpr.eqns if e.primitive.name == "sharding_constraint"]
    self.assertNotIn(P("stage", "expert", "tensor"), specs)

  def test_composes_with_named_scopes(self):
    plain, pipelined = self._models(tiny_config(), schedule())
    add_named_scopes(pipelined)
    np.testing.assert_allclose(nnx.jit(self._logits)(pipelined), self._logits(plain), rtol=1e-5, atol=1e-5)

  def test_maybe_apply_pipeline_follows_the_mesh(self):
    config = tiny_config(num_pipeline_microbatches=4, pipeline_fsdp_ag_once=True)
    model = create_qwen3_model(config, rngs=nnx.Rngs(0))
    self.assertIs(pipeline.maybe_apply_pipeline(model, config, create_mesh(devices=self.devices[:1])), model)
    self.assertIsInstance(model.decoder.layers, nnx.Sequential)
    pipeline.maybe_apply_pipeline(model, config, self.mesh)
    self.assertIsInstance(model.decoder.layers, pipeline.PipelinedLayers)
    self.assertEqual(
        model.decoder.layers.schedule,
        schedule(num_stages=2, num_repeats=2, layers_per_stage=1, num_microbatches=4, fsdp_ag_once=True),
    )

  def test_rejects_a_stack_the_schedule_does_not_fit(self):
    with self.assertRaisesRegex(ValueError, "4 layers, but the schedule pipelines 6"):
      self._models(tiny_config(), schedule(layers_per_stage=3))

  def test_rejects_non_parameter_state(self):
    layers = nnx.Sequential(*(StatefulLayer() for _ in range(2)))
    with self.assertRaisesRegex(ValueError, "non-parameter state.*Variable"):
      pipeline.pipeline_layers(layers, schedule=schedule(layers_per_stage=1))

  def test_rejects_a_model_without_one_stack(self):
    with self.assertRaisesRegex(ValueError, "exactly one nnx.Sequential.*found 0"):
      pipeline.apply_pipeline(nnx.Linear(2, 2, rngs=nnx.Rngs(0)), schedule=schedule(), sharding=DefaultSharding())

  def test_rejects_a_batch_that_does_not_split(self):
    _, pipelined = self._models(tiny_config(), schedule(num_microbatches=4))
    with self.assertRaisesRegex(ValueError, "Batch 6 does not split into 4"):
      pipelined(self.tokens[:6], self.positions[:6])
    with self.assertRaisesRegex(ValueError, "Microbatches of 1 .* do not split over the 2 shards"):
      pipelined(self.tokens[:4], self.positions[:4])

  def test_rejects_a_mesh_with_other_stages(self):
    _, pipelined = self._models(tiny_config(), schedule())
    with jax.set_mesh(create_mesh(devices=self.devices[:1])):
      with self.assertRaisesRegex(ValueError, "built for 2 stages.*stage=1"):
        pipelined.decoder.layers(jnp.zeros((8, 6, 16)), decoder_positions=self.positions)


if __name__ == "__main__":
  absltest.main()
