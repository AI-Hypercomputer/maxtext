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

"""Regression tests for the RL learning-rate schedule (CPU-only).

These guard the contract that PR #4029 ("get_optimizer: respect
learning_rate_schedule_steps") was supposed to preserve but silently broke:

  With a default RL config (learning_rate_schedule_steps unset == -1) the LR
  schedule length must equal the actual training length (max_train_steps), so
  warmup completes inside the run.

Why the bug shipped (and why the existing get_optimizer tests miss it):
`tests/post_training/unit/rl_utils_test.py::TestGetOptimizer` builds the config
from a `SimpleNamespace` that does not even carry `learning_rate_schedule_steps`,
so `getattr(cfg, "learning_rate_schedule_steps", -1)` returns -1, the
`schedule_steps <= 0` fallback fires, and the schedule correctly tracks
max_train_steps. The Pydantic validator never runs.

In a real run it DOES run: `MaxTextConfig.set_derived_and_validate_values`
rewrites `learning_rate_schedule_steps == -1 -> steps` (base.yml default 150_001)
*before* get_optimizer is called (configs/types.py). The `<= 0` guard in
get_optimizer therefore can never fire, warmup becomes
`0.1 * 150_001 = 15_000` instead of `0.1 * max_train_steps`, and on a 500-step
run the LR is ~300x too low at the same step.

The only way to catch this is to build the config through the REAL pyconfig
path so the validator runs. That is what these tests do.
"""

import sys
import unittest

import pytest

from maxtext.configs import pyconfig, types
from maxtext.trainers.post_train.rl import utils_rl
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.post_training]


# Tiny-model overrides known to build a valid MaxTextConfig on CPU without
# network access, mirroring tests/post_training/unit/lora_utils_test.py. The
# model shape is irrelevant here (get_optimizer only reads scalar config
# fields); these just let initialize_pydantic validate quickly.
_MODEL_OVERRIDES = {
    "enable_checkpointing": False,
    "base_num_decoder_layers": 1,
    "attention": "dot_product",
    "max_target_length": 8,
    "base_emb_dim": 128,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 256,
    "max_prefill_predict_length": 4,
    "model_name": "llama2-7b",
    "override_model_config": True,
    "weight_dtype": "bfloat16",
}


def _make_config(**overrides):
  """Build a real RLConfig (runs the Pydantic validator) for RL.

  Using initialize_pydantic (not a SimpleNamespace) is the whole point: it is
  what promotes learning_rate_schedule_steps == -1 to `steps`.
  """
  return pyconfig.initialize_pydantic(
      [sys.argv[0], get_test_config_path("post_train/rl.yml")],
      config_class=types.RLConfig,
      run_name="rl_lr_schedule_test",
      tokenizer_path="meta-llama/Llama-2-7b",
      **_MODEL_OVERRIDES,
      **overrides,
  )


def _effective_lr_at_step(opt, step):
  """Step an inject_hyperparams optimizer `step` times and read the LR it
  exposes in opt_state.hyperparams. This is exactly the per-step LR that
  tunix's peft_trainer reads and logs, so it reflects what training actually
  sees, not a re-derivation of the schedule.
  """
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

  params = {"w": jnp.zeros((), dtype=jnp.float32)}
  grads = {"w": jnp.zeros((), dtype=jnp.float32)}
  state = opt.init(params)
  for _ in range(step):
    _, state = opt.update(grads, state, params)
  return float(state.hyperparams["learning_rate"])


class RLLearningRateScheduleTest(unittest.TestCase):
  """Schedule-shape guards for utils_rl.get_optimizer built on a real config."""

  @pytest.mark.cpu_only
  def test_default_rl_config_warms_up_within_run(self):
    """REGRESSION GUARD (FAILS on PR #4029, passes once fixed).

    With learning_rate_schedule_steps unset (-1) and base.yml's default
    `steps` (150_001), the LR must still reach its configured peak by the end
    of the intended warmup (0.1 * max_train_steps). On the buggy code the
    warmup is sized to 150_001, so the LR is stuck near zero for the whole run.
    """
    peak = 3e-6
    config = _make_config(
        learning_rate=peak,
        warmup_steps_fraction=0.1,
        gradient_clipping_threshold=0.0,
        num_batches=500,
        num_epoch=1,
        train_fraction=1.0,
        learning_rate_schedule_steps=-1,  # "user did not set it" (base.yml default)
    )
    opt = utils_rl.get_optimizer(config)
    warmup_end = int(config.warmup_steps_fraction * config.train_steps)
    lr = _effective_lr_at_step(opt, warmup_end + 5)
    # Correct: lr ~= peak.  Buggy (#4029): lr ~= peak * warmup_end / 15000,
    # i.e. < 1% of peak.  A 0.5*peak threshold cleanly separates the two.
    self.assertGreaterEqual(
        lr,
        0.5 * peak,
        msg=(
            f"LR reached only {lr:.2e} by step {warmup_end + 5} (peak={peak:.2e}). "
            "Warmup was sized to base.yml `steps`, not max_train_steps "
            f"(={config.train_steps}). learning_rate_schedule_steps in effect="
            f"{config.learning_rate_schedule_steps}."
        ),
    )

  @pytest.mark.cpu_only
  def test_explicit_schedule_steps_decouples_from_run_length(self):
    """FEATURE GUARD (passes before and after the fix).

    An explicit learning_rate_schedule_steps must drive the schedule shape
    independently of max_train_steps (the capability #4029 was adding). This
    ensures the regression fix does not simply delete the feature.
    """
    peak = 3e-6
    schedule_len = 1000
    config = _make_config(
        learning_rate=peak,
        warmup_steps_fraction=0.1,
        gradient_clipping_threshold=0.0,
        num_batches=50,
        num_epoch=1,
        train_fraction=1.0,
        learning_rate_schedule_steps=schedule_len,  # explicitly set by the user
    )

    opt = utils_rl.get_optimizer(config)
    # The warmup length must follow the explicit schedule (0.1 * 1000 = 100),
    # independent of max_train_steps. Probe at fixed steps so the assertion does
    # not depend on num_iterations: early in a 1000-step warmup the LR is still
    # small, and by ~100 steps it has reached peak. A fix that wrongly forced
    # schedule == max_train_steps (a short warmup) would push LR@20 to ~peak and
    # trip the first assertion.
    self.assertLessEqual(_effective_lr_at_step(opt, 20), 0.4 * peak)
    self.assertGreaterEqual(_effective_lr_at_step(opt, int(0.1 * schedule_len) + 5), 0.9 * peak)

  @pytest.mark.cpu_only
  def test_validator_overwrites_minus_one_sentinel(self):
    """ROOT-CAUSE characterization (effective-value assertion).

    The value get_optimizer reads is NOT the -1 the user left: the validator
    promoted it to `steps`. This is precisely why get_optimizer's `<= 0`
    fallback is dead code in a real run. (Passes today; documents the seam
    that makes test_default_rl_config_warms_up_within_run fail.)
    """
    config = _make_config(learning_rate_schedule_steps=-1)
    self.assertNotEqual(config.learning_rate_schedule_steps, -1)
    self.assertEqual(config.learning_rate_schedule_steps, config.train_steps)


if __name__ == "__main__":
  unittest.main()
