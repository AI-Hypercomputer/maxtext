# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drives MaxTextTrainingEngine with Tunix's real GRPO loss, end to end.

Every other engine test uses a hand-written loss. This one uses
`tunix.rl.algo_core.grpo_loss_fn` itself, on a real `TrainExample`, against real Qwen3-0.6B
weights -- the combination the RL post-training path actually runs.

It asserts both that a parameter moved *and* that `get_metrics()` carries the loss.
Asserting only the first would pass while metrics silently went missing, which is exactly
how a parity check once compared against a fabricated 0.0.

Requires a TPU and read access to the public checkpoint bucket.
"""

import dataclasses

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

from tunix.rl import algo_core
from tunix.rl import common as tunix_common

pytestmark = [
    pytest.mark.external_training,  # reads the checkpoint from GCS
    pytest.mark.post_training,
    pytest.mark.integration_test,
    pytest.mark.tpu_only,
]

# Qwen3 special tokens: <|endoftext|> as pad, <|im_end|> as eos. pad_id is load-bearing --
# without it the adapter leaves decoder_segment_ids as None, MaxText falls back to
# causal-only masking, and pad positions get attended to, silently corrupting log-probs.
_PAD_ID = 151643
_EOS_ID = 151645

_CHECKPOINT = "gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items"


def _config(**overrides) -> pyconfig.HyperParameters:
  """Qwen3-0.6B from a scanned params-only checkpoint.

  `model_name=default` cannot be substituted to make this cheap: TunixMaxTextAdapter looks
  the model up in HF_MODEL_CONFIGS, which has no 'default' entry.
  """
  argv = [
      "maxtext_engine_grpo_loss_test.py",
      get_test_config_path("base.yml"),
      "model_name=qwen3-0.6b",
      "run_name=maxtext_engine_grpo_loss_test",
      # load_parameters_path requires enable_checkpointing=True; the config validator
      # rejects the combination outright otherwise.
      "enable_checkpointing=True",
      f"load_parameters_path={_CHECKPOINT}",
      "scan_layers=True",
      # Never reach out to HuggingFace from a test.
      "convert_checkpoint_if_possible=False",
      "init_weights_seed=42",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "skip_jax_distributed_system=True",
      "gradient_accumulation_steps=1",
      # The LR schedule warms up linearly from zero, so lr(step=0) == 0 and a single
      # optimizer step would be a silent no-op that reports healthy gradients.
      "warmup_steps_fraction=0.0",
      "learning_rate=1e-4",
      "micro_batch_size_to_train_on=2",
      "max_target_length=64",
      # The KL assertion below compares log-probs produced by two different code paths:
      # tunix's compute_per_token_logps for the reference, and the engine's own sharded
      # forward pass for the policy. At the default bf16 matmul precision those disagree
      # by ~2e-2 on log-probs near -12.6, and low_var_kl squares the difference into a
      # KL of ~1e-4 -- noise, but large enough to swamp a real regression. fp32 matmuls
      # bring it back under 1e-5; the model is tiny enough that the cost is invisible.
      "matmul_precision=highest",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


@dataclasses.dataclass(frozen=True)
class _GrpoConfig:
  """The fields `algo_core.grpo_loss_fn` reads off its algo_config."""

  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_high: float = 0.2
  loss_algo: str = "grpo"
  loss_agg_mode: str = "token-mean"
  temperature: float = 1.0
  kl_loss_mode: str = "low_var_kl"
  kl_clamp_value: float | None = None


def _train_example(
    model,
    algo_config: _GrpoConfig,
    batch: int = 2,
    prompt_len: int = 8,
    completion_len: int = 8,
) -> tunix_common.TrainExample:
  """Builds a TrainExample whose reference log-probs come from the model itself.

  The reference must be the model's own log-probs, which is the state at step 0 of a real
  GRPO run (policy == reference, so KL == 0). Inventing them makes the KL term meaningless
  rather than merely imprecise: real log-probs here are near -12.6, so a fabricated
  N(0, 0.1) reference gives `r = exp(ref - logp)` around e^12.6 and a loss in the 1e8
  range -- which still satisfies every "did it run" assertion.

  They are computed before the TrainExample is constructed so that no half-built example
  exists. `ref_per_token_logps=None` is not merely incomplete: grpo_loss_fn guards its KL
  block on that field being set but references kl_loss unconditionally in the beta branch,
  so a None reference raises UnboundLocalError whenever beta != 0.
  """
  rng = np.random.default_rng(0)
  # Ids well clear of the special tokens, so masking is unambiguous.
  prompt_ids = jnp.asarray(rng.integers(1000, 2000, size=(batch, prompt_len)), dtype=jnp.int32)
  completion_ids = jnp.asarray(rng.integers(1000, 2000, size=(batch, completion_len)), dtype=jnp.int32)

  graphdef, state = nnx.split(model)
  ref_logps = tunix_common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=_PAD_ID,
      eos_id=_EOS_ID,
      stop_gradient=True,
      temperature=algo_config.temperature,
  )
  if isinstance(ref_logps, tuple):
    ref_logps = ref_logps[0]

  return tunix_common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=jnp.ones((batch, prompt_len), dtype=jnp.int32),
      completion_ids=completion_ids,
      completion_mask=jnp.ones((batch, completion_len), dtype=jnp.int32),
      advantages=jnp.asarray(rng.normal(size=(batch,)), dtype=jnp.float32),
      ref_per_token_logps=ref_logps,
      # None here is fine and idiomatic: grpo_loss_fn falls back to
      # stop_gradient(per_token_logps), making the importance ratio exactly 1.
      old_per_token_logps=None,
  )


def _param_leaves(model) -> list[jax.Array]:
  return jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))


class MaxTextEngineGrpoLossTest(absltest.TestCase):
  """The engine driven by Tunix's own GRPO loss, not a stand-in."""

  def test_grpo_loss_drives_a_training_step(self):
    cfg = _config()
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    # The batch has to divide the data x fsdp device count, and only the mesh knows that
    # number, so it cannot be hard-coded in `_config`. The engine traces its kernels under
    # `nn_partitioning.axis_rules`, which is what makes MaxText's logical constraints on the
    # activations real; a batch those devices cannot split is then padded out by XLA, and the
    # padded lanes are all-zero sequences that mask themselves out of attention entirely.
    # Their contribution comes back as NaN on the pad token's embedding row -- a finite loss
    # and unusable gradients. The splash kernel asserts on exactly this ratio
    # (`attention_op.py`, "Batch dimension should be shardable"); dot_product does not, so
    # here it surfaced as a NaN instead of an error.
    batch = int(mesh.shape["data"] * mesh.shape["fsdp"])
    if cfg.micro_batch_size_to_train_on != batch:
      cfg = _config(micro_batch_size_to_train_on=batch)
    algo_config = _GrpoConfig()

    engine = maxtext_engine.MaxTextTrainingEngine(
        cfg,
        mesh=mesh,
        # grpo_loss_fn calls the model with Tunix's signature, so the adapter wrap is
        # needed for the loss itself, not only for weight sync.
        wrap_with_tunix_adapter=True,
        tokenizer_pad_id=_PAD_ID,
    )
    self.assertEqual(type(engine.model).__name__, "TunixMaxTextAdapter")

    # Tunix's loss is used directly, with no adapter closure. Setting a gen_model_input_fn
    # tells the engine to invoke the loss Tunix's way, `loss_fn(model, **inputs)`, and the
    # dict below is already exactly grpo_loss_fn's keyword arguments -- which is what the
    # orchestrator's own `_grpo_model_input` produces in the real pipeline.
    returned = engine.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(
        lambda payload: {
            "train_example": payload,
            "algo_config": algo_config,
            "pad_id": _PAD_ID,
            "eos_id": _EOS_ID,
        }
    )
    self.assertIs(returned, engine)

    payload = _train_example(engine.model, algo_config, batch=batch)
    before = _param_leaves(engine.model)

    engine.fwd_bwd(payload)
    new_step = engine.update()

    after = _param_leaves(engine.model)
    max_delta = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(after, before))

    # Half one of the gate: real gradients reached the weights.
    self.assertGreater(max_delta, 0.0, "no parameter moved, so no training happened")
    self.assertIsInstance(new_step, int)
    self.assertEqual(new_step, 1)
    self.assertEqual(engine.train_step, 1)

    # Half two, asserted separately: the step is only meaningful if its metrics survived.
    buf = engine.get_metrics(clear_cache=True)
    self.assertNotEqual(buf.id, maxtext_engine.EMPTY_METRICS_BUFFER_ID)
    self.assertIn("loss", buf.weighted_metrics)

    loss = float(jnp.asarray(buf.weighted_metrics["loss"].compute()).reshape(-1)[0])
    self.assertTrue(np.isfinite(loss), f"loss is not finite: {loss}")
    # A GRPO loss is order 1. A far larger value still satisfies every assertion above but
    # means the reference log-probs were wrong -- see _train_example.
    self.assertLess(abs(loss), 100.0, f"loss {loss} is implausible for GRPO")

    # Policy and reference are the same weights here, so the KL term must vanish.
    kl = float(jnp.asarray(buf.weighted_metrics["kl"].compute()).reshape(-1)[0])
    self.assertAlmostEqual(kl, 0.0, places=4)

    # The real GRPO aux is mixed: WeightedMetrics and plain arrays must land in
    # different buckets.
    self.assertIn("kl_loss", buf.weighted_metrics)
    self.assertIn("learning_rate", buf.scalar_metrics)
    self.assertNotIn("learning_rate", buf.weighted_metrics)


if __name__ == "__main__":
  absltest.main()
