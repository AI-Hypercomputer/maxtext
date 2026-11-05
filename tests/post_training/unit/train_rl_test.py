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

"""Unit tests for train_rl.py."""

import dataclasses
import inspect
import json
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import unittest
from unittest import mock
import grain
import numpy as np
import pytest
from types import SimpleNamespace
import jax

from maxtext.trainers.post_train.rl import train_rl
from maxtext.trainers.post_train.rl import utils_rl

pytestmark = [pytest.mark.post_training]
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


def _echo_reward(prompts, completions, tmvp_config=None, **kwargs):
  """Module-level reward fn for the pickling tests (pickles by reference)."""
  del prompts, tmvp_config, kwargs
  return [1.0] * len(completions)


def _get_mock_devices(devices_per_slice, num_slices=1):
  mock_devices = []
  for slice_idx in range(num_slices):
    for _ in range(devices_per_slice):
      d = mock.MagicMock()
      d.id = len(mock_devices)
      d.slice_index = slice_idx
      mock_devices.append(d)
  return mock_devices


class TrainRLTest(unittest.TestCase):
  """Tests for train_rl.py."""

  def test_rl_config_includes_shared_decoder_defaults(self):
    """RL models must expose fields consumed by the shared decoder."""
    config = types.RLConfig(model_name="gemma4-26b")

    self.assertEqual(config.engram_layers, [])
    self.assertEqual(config.engram_max_ngram_size, 3)
    self.assertEqual(config.mhc_expansion_rate, 1)
    self.assertEqual(config.sinkhorn_iterations, 20)
    self.assertFalse(config.enable_mhc_lite)
    self.assertFalse(config.enable_prefix_caching)

  def test_rl_config_packing_and_block_size_defaults(self):
    """Packing is off and the vLLM page size is backend-chosen unless set."""
    config = types.RLConfig(model_name="gemma4-26b")

    self.assertEqual(config.max_seq_token_per_tpu, 0)
    self.assertIsNone(config.vllm_block_size)

    config = types.RLConfig(model_name="gemma4-26b", max_seq_token_per_tpu=12288, vllm_block_size=128)
    self.assertEqual(config.max_seq_token_per_tpu, 12288)
    self.assertEqual(config.vllm_block_size, 128)

  def test_rl_config_tunix_knob_defaults(self):
    """The forwarded tunix knobs default to tunix's own defaults."""
    config = types.RLConfig(model_name="gemma4-26b")

    self.assertTrue(config.gc_collect_after_weight_sync)
    self.assertTrue(config.free_kv_cache_during_weight_sync)

    config = types.RLConfig(
        model_name="gemma4-26b",
        gc_collect_after_weight_sync=False,
        free_kv_cache_during_weight_sync=False,
    )
    self.assertFalse(config.gc_collect_after_weight_sync)
    self.assertFalse(config.free_kv_cache_during_weight_sync)
    # A rollout option, so the flat key lands in the nested vLLM model.
    self.assertFalse(config.vllm.free_kv_cache_during_weight_sync)

  def test_rl_config_reward_worker_defaults(self):
    """The reward worker knobs default to tunix's own defaults."""
    config = types.RLConfig(model_name="gemma4-26b")

    self.assertEqual(config.reward_num_workers, 0)
    self.assertEqual(config.reward_worker_timeout_seconds, 180.0)

    config = types.RLConfig(
        model_name="gemma4-26b",
        reward_num_workers=-1,
        reward_worker_timeout_seconds=30.0,
    )
    self.assertEqual(config.reward_num_workers, -1)
    self.assertEqual(config.reward_worker_timeout_seconds, 30.0)

  def test_reward_fn_wrapper_is_picklable(self):
    """The reward wrapper must pickle so tunix's worker pool can ship it."""
    # pylint: disable=protected-access  # the tests target the internal wrapper
    wrapped = train_rl._RewardFn(_echo_reward, SimpleNamespace(tag="cfg"))

    clone = pickle.loads(pickle.dumps(wrapped))

    self.assertEqual(clone(prompts=["p"], completions=["c"]), [1.0])
    self.assertEqual(clone.__name__, "_echo_reward")
    self.assertEqual(clone.tmvp_config.tag, "cfg")

    # The production config must pickle too, or the pool falls back to serial.
    production = train_rl._RewardFn(_echo_reward, types.RLConfig(model_name="gemma4-26b"))
    clone = pickle.loads(pickle.dumps(production))
    self.assertEqual(clone(prompts=["p"], completions=["c"]), [1.0])

  def _write_user_rewards(self, file_name):
    """Writes a two-function user reward file into a temp dir cleaned up (with its import state) after the test."""
    tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(tmp.cleanup)
    module_name = os.path.splitext(file_name)[0]
    previous = sys.modules.get(module_name)

    def _restore_imports():
      if previous is None:
        sys.modules.pop(module_name, None)
      else:
        sys.modules[module_name] = previous
      if tmp.name in sys.path:
        sys.path.remove(tmp.name)

    self.addCleanup(_restore_imports)
    path = os.path.join(tmp.name, file_name)
    with open(path, "w", encoding="utf-8") as f:
      f.write(
          "def my_reward(prompts, completions, tmvp_config=None, **kwargs):\n  return [2.0] * len(completions)\n\n\n"
          "def my_other_reward(prompts, completions, tmvp_config=None, **kwargs):\n  return [3.0] * len(completions)\n"
      )
    return path

  def test_custom_reward_fns_from_file_run_in_a_forkserver_worker(self):
    """Callables from load_custom_callable must work inside tunix's reward workers.

    tunix starts its workers from a fork server: they are fresh interpreters
    that do not inherit the parent's sys.modules, so an in-process pickle
    round trip proves nothing. Ship the functions to a real fork-server worker:
    it has to import the user module by name. A module name the worker cannot
    import kills the worker on unpickling and the caller only finds out when
    its timeout expires.
    """
    path = self._write_user_rewards("maxtext_test_user_rewards.py")
    # Both from the same file, like a `reward_functions` list: the second load must not invalidate the first.
    fns = [
        utils_rl.load_custom_callable(path, name, importable_by_workers=True) for name in ("my_reward", "my_other_reward")
    ]
    wrapped = [train_rl._RewardFn(fn, SimpleNamespace()) for fn in fns]  # pylint: disable=protected-access

    with multiprocessing.get_context("forkserver").Pool(1) as pool:
      jobs = [pool.apply_async(w, kwds={"prompts": ["p"], "completions": ["c"]}) for w in wrapped]
      results = [job.get(timeout=300) for job in jobs]

    self.assertEqual(results, [[2.0], [3.0]])
    self.assertEqual([w.__name__ for w in wrapped], ["my_reward", "my_other_reward"])

  def test_custom_reward_fn_runs_in_worker_when_its_directory_is_on_sys_path(self):
    """The common launch case: cwd holds the reward file and `python -m` put cwd on sys.path.

    find_spec then resolves the module name to the user's own file. That must
    not count as a name conflict, or every such launch silently loses worker
    parallelism for its custom reward fns.
    """
    path = self._write_user_rewards("maxtext_test_cwd_rewards.py")
    sys.path.insert(0, os.path.dirname(path))

    fn = utils_rl.load_custom_callable(path, "my_reward", importable_by_workers=True)
    wrapped = train_rl._RewardFn(fn, SimpleNamespace())  # pylint: disable=protected-access

    self.assertEqual(fn.__module__, "maxtext_test_cwd_rewards")
    with multiprocessing.get_context("forkserver").Pool(1) as pool:
      result = pool.apply_async(wrapped, kwds={"prompts": ["p"], "completions": ["c"]}).get(timeout=300)
    self.assertEqual(result, [2.0])

  def test_custom_callable_never_shadows_an_existing_module(self):
    """A user file named like a real module loads privately and leaves that module alone."""
    path = self._write_user_rewards("json.py")
    real_json = sys.modules["json"]

    fn = utils_rl.load_custom_callable(path, "my_reward", importable_by_workers=True)

    self.assertEqual(fn(prompts=["p"], completions=["c"]), [2.0])
    self.assertIs(sys.modules["json"], real_json)
    self.assertNotIn(os.path.dirname(path), sys.path)
    # Not importable by name, so it must fail to pickle in the parent -- tunix then falls back to the parent
    # process immediately instead of losing a worker and waiting out the timeout.
    with self.assertRaises((pickle.PicklingError, AttributeError)):
      pickle.dumps(fn)

  def test_reward_fn_wrapper_engages_tunix_worker_pool(self):
    """reward_num_workers must actually run the wrapper in tunix's workers.

    A closure wrapper raises PicklingError inside the pool and tunix silently
    falls back to serial parent-process evaluation forever; the fallback set
    staying empty proves the pool really evaluated the fn.
    """
    # pylint: disable=protected-access,import-outside-toplevel
    try:
      from tunix.rl import reward_manager
      from tunix.rl.grpo.grpo_learner import GrpoConfig
    except ImportError:
      self.skipTest("tunix reward_manager is not available")
    if "reward_num_workers" not in inspect.signature(GrpoConfig).parameters:
      self.skipTest("installed tunix predates reward_num_workers")

    grpo_config = GrpoConfig(num_generations=2, num_iterations=1, beta=0.0, epsilon=0.2, reward_num_workers=2)
    wrapped = train_rl._RewardFn(_echo_reward, SimpleNamespace())
    manager = reward_manager.SequenceRewardManager([wrapped], grpo_config)
    try:
      out = manager._compute_rewards(["p"] * 4, ["c"] * 4)
      self.assertEqual(list(np.asarray(out["rewards"]).ravel()), [1.0] * 4)
      self.assertEqual(manager._parent_only_fns, set())
    finally:
      manager.close()

  def test_check_numbers_falls_back_to_the_parent_process(self):
    """Pins that the built-in check_numbers is NOT parallelized by reward_num_workers.

    check_numbers grades through math_verify_pool, which builds its own process
    pool. Pool workers are daemonic and may not have children, so inside a
    tunix reward worker that raises immediately; tunix then evaluates
    check_numbers in the parent (where math_verify_pool still parallelizes the
    grading) for the rest of the run. The fallback has to be immediate and the
    scores correct. If math_verify_pool learns to grade inline inside a worker,
    update this test.
    """
    # pylint: disable=protected-access,import-outside-toplevel
    try:
      from tunix.rl import reward_manager
      from tunix.rl.grpo.grpo_learner import GrpoConfig
    except ImportError:
      self.skipTest("tunix reward_manager is not available")
    if "reward_num_workers" not in inspect.signature(GrpoConfig).parameters:
      self.skipTest("installed tunix predates reward_num_workers")

    config = SimpleNamespace(
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<answer>",
        solution_end_token="</answer>",
        reward_exact_answer=3.0,
        reward_white_space_format_match=1.5,
        reward_ratio_guess_to_answer_high=1.0,
        reward_ratio_guess_to_answer_low=0.5,
        penalty_incorrect_format=-0.5,
        penalty_incorrect_answer=-0.5,
        dataset_name="test",
        debug=False,
    )
    grpo_config = GrpoConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        reward_num_workers=2,
        reward_worker_timeout_seconds=300.0,
    )
    wrapped = [train_rl._RewardFn(fn, config) for fn in (_echo_reward, utils_rl.check_numbers)]
    manager = reward_manager.SequenceRewardManager(wrapped, grpo_config)
    # "1/2" vs "0.5" is not a string match, so grading has to go through math_verify_pool.
    completions = ["<reasoning>r</reasoning><answer>1/2</answer>"] * 4
    try:
      start = time.monotonic()
      out = manager._compute_rewards(["p"] * 4, completions, answer=[json.dumps(["0.5"])] * 4, question=["q"] * 4)
      elapsed = time.monotonic() - start
    finally:
      manager.close()

    self.assertEqual(manager._parent_only_fns, {"check_numbers"})
    self.assertLess(elapsed, 200.0, "the fallback must not wait for reward_worker_timeout_seconds")
    # Summed over the two fns: _echo_reward from the workers plus a correct math_verify grade from the parent.
    self.assertEqual(list(np.asarray(out["rewards"]).ravel()), [1.0 + config.reward_exact_answer] * 4)

  def _rollout_config_kwargs_from_create_rl_components(self, rollout_config_fields, **config_overrides):
    """Runs create_rl_components with the tunix pieces faked; returns the kwargs RolloutConfig received.

    `rollout_config_fields` is the set of keyword names the fake RolloutConfig declares, which is how
    `_kwargs_supported_by` tells a tunix that has an option from one that predates it.
    """
    received = {}

    class FakeRolloutConfig:

      def __init__(self, **fields):
        unknown = sorted(set(fields) - set(rollout_config_fields))
        if unknown:
          raise TypeError(f"unexpected keyword arguments {unknown}")
        received.update(fields)

    FakeRolloutConfig.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in sorted(rollout_config_fields)]
    )
    trainer_config = types.RLConfig(
        model_name="qwen3-0.6b", enable_checkpointing=False, enable_tunix_perf_metrics=False, **config_overrides
    )
    with (
        mock.patch.object(train_rl.utils_rl, "get_optimizer"),
        mock.patch.object(train_rl.pyconfig, "initialize", return_value=SimpleNamespace(logical_axis_rules=[])),
        mock.patch.object(train_rl.base_rollout, "RolloutConfig", FakeRolloutConfig),
        mock.patch.object(train_rl.rl_cluster_lib, "ClusterConfig"),
        mock.patch.object(train_rl.rl_cluster_lib, "RLTrainingConfig"),
        mock.patch.object(train_rl.rl_cluster_lib, "RLCluster"),
        mock.patch.object(train_rl, "get_rollout_kwargs_for_parallelism", return_value={}),
        mock.patch.object(train_rl, "build_reward_fns", return_value=[]),
        mock.patch.object(train_rl, "GrpoLearner"),
    ):
      train_rl.create_rl_components(
          trainer_config=trainer_config,
          sampler_config=SimpleNamespace(enable_expert_parallel=False),
          sampler_devices=[object()],
          actor_model=object(),
          actor_mesh=object(),
          reference_model=object(),
          reference_mesh=object(),
          rollout_mesh=object(),
          model_tokenizer=object(),
      )
    return received

  def test_create_rl_components_forwards_free_kv_cache_to_rollout_config(self):
    """The RLConfig key must reach tunix's RolloutConfig under tunix's field name."""
    field = "rollout_vllm_free_kv_cache_during_weight_sync"
    declared = set(inspect.signature(train_rl.base_rollout.RolloutConfig).parameters) | {field}

    for value in (False, True):
      received = self._rollout_config_kwargs_from_create_rl_components(declared, free_kv_cache_during_weight_sync=value)
      self.assertIs(received[field], value)

  def test_create_rl_components_drops_free_kv_cache_for_a_tunix_that_predates_it(self):
    """An older tunix RolloutConfig must not receive (and choke on) the unknown kwarg."""
    field = "rollout_vllm_free_kv_cache_during_weight_sync"
    declared = set(inspect.signature(train_rl.base_rollout.RolloutConfig).parameters) - {field}

    received = self._rollout_config_kwargs_from_create_rl_components(declared, free_kv_cache_during_weight_sync=False)

    self.assertNotIn(field, received)
    self.assertIn("max_tokens_to_generate", received)

  def test_kwargs_supported_by_keeps_declared_fields_only(self):
    """Knobs the installed tunix does not declare are dropped, with a log line."""
    # pylint: disable=protected-access  # the test targets the internal filter helper

    @dataclasses.dataclass
    class FakeTunixConfig:
      known: int = 0

    with mock.patch.object(train_rl.max_logging, "log") as mock_log:
      kept = train_rl._kwargs_supported_by(FakeTunixConfig, known=1, unknown=2)
    self.assertEqual(kept, {"known": 1})
    self.assertIn("unknown", mock_log.call_args.args[0])

    with mock.patch.object(train_rl.max_logging, "log") as mock_log:
      kept = train_rl._kwargs_supported_by(FakeTunixConfig, known=3)
    self.assertEqual(kept, {"known": 3})
    mock_log.assert_not_called()

    class PlainTunixConfig:

      def __init__(self, known=0):
        self.known = known

    self.assertEqual(train_rl._kwargs_supported_by(PlainTunixConfig, known=4, unknown=5), {"known": 4})

  def test_kwargs_supported_by_passes_everything_to_var_keyword_and_opaque_classes(self):
    """A cls taking **kwargs, or with no inspectable signature, gets all kwargs."""
    # pylint: disable=protected-access  # the test targets the internal filter helper

    class VarKeywordConfig:

      def __init__(self, known=0, **kwargs):
        self.known = known
        self.extra = kwargs

    with mock.patch.object(train_rl.max_logging, "log") as mock_log:
      kept = train_rl._kwargs_supported_by(VarKeywordConfig, known=1, unknown=2)
    self.assertEqual(kept, {"known": 1, "unknown": 2})
    mock_log.assert_not_called()

    # `type` has no inspectable constructor signature; the filter cannot know
    # what it accepts, so it must pass everything through.
    self.assertEqual(train_rl._kwargs_supported_by(type, unknown=3), {"unknown": 3})

  def test_rl_config_packing_budget_must_fit_a_maximal_sequence(self):
    """A packed row must hold prompt cap + generation cap (= max_target_length)."""
    with self.assertRaisesRegex(ValueError, "max_seq_token_per_tpu"):
      types.RLConfig(model_name="gemma4-26b", max_target_length=4096, max_seq_token_per_tpu=2048)

    config = types.RLConfig(model_name="gemma4-26b", max_target_length=4096, max_seq_token_per_tpu=4096)
    self.assertEqual(config.max_seq_token_per_tpu, 4096)

  @pytest.mark.cpu_only
  def test_rollout_prefix_caching_respects_config_for_attention_model(self):
    config = SimpleNamespace(
        enable_prefix_caching=True,
        decoder_block=train_rl.DecoderBlockType.QWEN3,
    )

    self.assertTrue(train_rl.rollout_prefix_caching_enabled(config))

  @pytest.mark.cpu_only
  def test_rollout_prefix_caching_disabled_for_recurrent_model(self):
    for decoder_block in (train_rl.DecoderBlockType.QWEN3_NEXT, train_rl.DecoderBlockType.QWEN3_5):
      with self.subTest(decoder_block=decoder_block):
        config = SimpleNamespace(
            enable_prefix_caching=True,
            decoder_block=decoder_block,
        )

        with mock.patch.object(train_rl.max_logging, "log") as mock_log:
          self.assertFalse(train_rl.rollout_prefix_caching_enabled(config))

        mock_log.assert_called_once()

  @pytest.mark.cpu_only
  def test_rollout_prefix_caching_defaults_to_disabled(self):
    config = SimpleNamespace(decoder_block=train_rl.DecoderBlockType.QWEN3)

    self.assertFalse(train_rl.rollout_prefix_caching_enabled(config))

  def test_setup_configs_and_devices_pathways_split(self):
    """Test setup_configs_and_devices with multiple VMs and Pathways."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.5,
        sampler_devices_fraction=0.5,
    )

    # Following the pattern in distillation_checkpointing_test.py for mocking jax objects
    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            return_value=mock_config,
        ),
    ):
      trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
          ["dummy", "dummy"], config_class=types.RLConfig
      )

      self.assertEqual(trainer_config, mock_config)
      self.assertEqual(sampler_config, mock_config)
      self.assertEqual(len(trainer_devices), 4)
      self.assertEqual(len(sampler_devices), 4)
      self.assertEqual(trainer_devices, mock_devices[:4])
      self.assertEqual(sampler_devices, mock_devices[4:])

  def test_setup_configs_and_devices_pathways_fractional_split(self):
    """Test setup_configs_and_devices with multiple VMs and custom fractions."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.25,
        sampler_devices_fraction=0.75,
    )

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            return_value=mock_config,
        ),
    ):
      _, _, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
          ["dummy", "dummy"], config_class=types.RLConfig
      )

      self.assertEqual(len(trainer_devices), 2)
      self.assertEqual(len(sampler_devices), 6)
      self.assertEqual(trainer_devices, mock_devices[:2])
      self.assertEqual(sampler_devices, mock_devices[2:])

  def test_setup_configs_and_devices_multislice_not_enough_slices(self):
    """Test setup_configs_and_devices raises ValueError when not enough slices."""
    mock_devices = _get_mock_devices(num_slices=2, devices_per_slice=4)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=1,
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "Not enough slices for trainer and samplers"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"], config_class=types.RLConfig)

  def test_setup_configs_and_devices_multislice_invalid_tp(self):
    """Test setup_configs_and_devices raises ValueError for invalid TP."""
    mock_devices = _get_mock_devices(num_slices=4, devices_per_slice=8)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=2,
        ici_tensor_parallelism=3,  # 8 is not divisible by 3
        ici_fsdp_parallelism=-1,
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "must be divisible by tensor parallelism"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"], config_class=types.RLConfig)

  def test_setup_configs_and_devices_multislice_invalid_tp_fsdp(self):
    """Test setup_configs_and_devices raises ValueError for inconsistent TP and FSDP."""
    mock_devices = _get_mock_devices(num_slices=4, devices_per_slice=8)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=2,
        ici_tensor_parallelism=4,
        ici_fsdp_parallelism=3,  # 4 * 3 != 8
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "must equal devices_per_slice"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"], config_class=types.RLConfig)

  def test_get_rollout_kwargs_no_dp(self):
    """Test case 1: sampler_config.rollout_data_parallelism=-1 -> verify result is calculated."""
    # num_sampler_devices=16, tp=2, ep=4 -> dp should be 16 // (2 * 4) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=4,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 4,
    }
    self.assertEqual(
        train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 16),
        expected_result,
    )

  def test_get_rollout_kwargs_auto_tp(self):
    """Test case 2: dp=2, tp=-1, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(
        train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4),
        expected_result,
    )

  def test_get_rollout_kwargs_fixed_tp_dp(self):
    """Test case 3: dp=2, tp=2, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(
        train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4),
        expected_result,
    )

  def test_get_rollout_kwargs_auto_ep(self):
    """Test case 4: ep=-1 -> verify result is calculated."""
    # num_sampler_devices=8, tp=2, dp=2 -> ep should be 8 // (2 * 2) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=-1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    self.assertEqual(
        train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8),
        expected_result,
    )

  def test_get_rollout_kwargs_errors(self):
    """Test various error cases for get_rollout_kwargs_for_parallelism."""
    # More than one -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "At most one of .* can be -1"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * ep) != 0 when dp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * dp) != 0 when ep == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=-1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # num_devices % (dp * ep) != 0 when tp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=3,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=2,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # tp * dp * ep != num_sampler_devices when all are positive
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, r"!= len\(sampler_devices\)"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

  def test_prompt_filtering(self):
    """Test that prompts longer than max_prefill_predict_length are filtered out."""
    # Setup mocks
    mock_tokenizer = mock.MagicMock()

    # Define tokenizer side effect
    def tokenize_side_effect(text):
      if text == "short":
        return [0] * 5
      else:
        return [0] * 15

    mock_tokenizer.tokenize.side_effect = tokenize_side_effect

    # Define dataset mock data
    train_data = [
        {"question": "short", "answer": "a1"},
        {"question": "long", "answer": "a2"},
        {"question": "short", "answer": "a3"},
        {"question": "long", "answer": "a4"},
    ]
    test_data = [
        {"question": "short", "answer": "a5"},
        {"question": "long", "answer": "a6"},
    ]
    train_map_ds = grain.MapDataset.source(train_data)
    test_map_ds = grain.MapDataset.source(test_data)

    def get_dataset_side_effect(config, split, data_files=None, dataset_name=None):
      if split == "train":
        return train_map_ds
      else:
        return test_map_ds

    def get_filtered_data_side_effect(dataset_name, model_tokenizer, template_config, trainer_config, x):
      return {
          "prompts": x["question"],
          "question": x["question"],
          "answer": f"[{x['answer'], x['answer']}]",
      }

    # Configs
    trainer_config = SimpleNamespace(
        debug=False,
        rl=SimpleNamespace(use_agentic_rollout=False),
        tokenizer_path="dummy_path",
        dataset_name="dummy_dataset",
        train_split="train",
        eval_split="eval",
        hf_train_files=None,
        hf_eval_files=None,
        data_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        max_prefill_predict_length=10,
        batch_size=2,
        eval_batch_size=1,
        num_batches=2,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
    )

    with (
        mock.patch(
            "maxtext.trainers.post_train.rl.train_rl.get_dataset",
            side_effect=get_dataset_side_effect,
        ),
        mock.patch(
            "maxtext.trainers.post_train.rl.utils_rl.process_data",
            side_effect=get_filtered_data_side_effect,
        ),
    ):
      train_dataset, test_dataset = train_rl.prepare_datasets(trainer_config, mock_tokenizer)

      # Check filtered train dataset
      elements = list(train_dataset)
      # dataset_size = 4. Indices [0,1,2,3] are [short, long, short, long].
      # Filtered results: [short, short].
      # batch(2) will return 1 batch of 2 elements.
      self.assertEqual(len(elements), 1)
      batch = elements[0]
      self.assertEqual(len(batch["prompts"]), 2)
      for prompt in batch["prompts"]:
        self.assertEqual(prompt, "short")

      # Check filtered test dataset
      test_elements = list(test_dataset)
      # test_data indices [0,1] are [short, long].
      # num_test_batches=1, batch_size=2 -> test dataset_size = 2.
      # Filtering results: [short].
      # batch(2) will return 1 batch of 1 element.
      self.assertEqual(len(test_elements), 1)
      test_batch = test_elements[0]
      self.assertEqual(len(test_batch["prompts"]), 1)
      self.assertEqual(test_batch["prompts"][0], "short")

  @mock.patch("datasets.load_dataset")
  def test_prepare_datasets_with_split(self, mock_load):
    mock_ds = mock.MagicMock()
    mock_split_result = {
        "train": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "test": [{"question": "q3", "answer": "a3"}],
    }
    mock_ds.train_test_split.return_value = mock_split_result
    mock_load.return_value = mock_ds
    mock_config = SimpleNamespace(
        debug=False,
        dataset_name="open-r1/OpenR1-Math-220k",
        eval_dataset_name="open-r1/OpenR1-Math-220k",
        train_split="train",
        hf_train_files="hf://open-r1/OpenR1-Math-220k/data/dummy.parquet",
        data_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        num_batches=1,
        batch_size=2,
        eval_batch_size=1,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
        rl=SimpleNamespace(use_agentic_rollout=False),
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<answer>",
        solution_end_token="</answer>",
        max_prefill_predict_length=256,
    )

    train_ds, test_ds = train_rl.prepare_datasets(
        trainer_config=mock_config,
        model_tokenizer=mock.MagicMock(),
    )

    mock_load.assert_called_once_with(
        "parquet",
        data_files={mock_config.train_split: mock_config.hf_train_files},
        split=mock_config.train_split,
    )
    mock_ds.train_test_split.assert_called_once_with(test_size=0.05, seed=mock_config.data_shuffle_seed)
    train_batches, test_batches = list(train_ds), list(test_ds)
    total_train_examples = sum(len(batch["question"]) for batch in train_batches)
    assert total_train_examples == 2
    total_test_examples = sum(len(batch["question"]) for batch in test_batches)
    assert total_test_examples == 1

  @mock.patch("datasets.load_dataset")
  def test_prepare_datasets_without_split(self, mock_load):
    mock_ds = mock.MagicMock()
    mock_load.return_value = mock_ds
    mock_config = SimpleNamespace(
        debug=False,
        dataset_name="openai/gsm8k",
        eval_dataset_name="openai/gsm8k",
        train_split="train",
        eval_split="test",
        hf_train_files="hf://openai/gsm8k/data/dummy.parquet",
        hf_eval_files="hf://openai/gsm8k/data/dummy.parquet",
        data_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        num_batches=1,
        batch_size=5,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
        rl=SimpleNamespace(use_agentic_rollout=False),
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<answer>",
        solution_end_token="</answer>",
        max_prefill_predict_length=256,
    )

    _, _ = train_rl.prepare_datasets(
        trainer_config=mock_config,
        model_tokenizer=mock.MagicMock(),
    )

    expected_calls = [
        mock.call(
            "parquet",
            data_files={mock_config.train_split: mock_config.hf_train_files},
            split=mock_config.train_split,
        ),
        mock.call(
            "parquet",
            data_files={mock_config.eval_split: mock_config.hf_eval_files},
            split=mock_config.eval_split,
        ),
    ]
    mock_load.assert_has_calls(expected_calls, any_order=True)
    assert mock_load.call_count == len(expected_calls)

  def test_build_reward_fns_defaults_when_no_custom(self):
    """With neither knob set, the built-in 3-fn stack is returned."""
    trainer_config = SimpleNamespace(reward_functions_path="", reward_functions="")
    reward_fns = train_rl.build_reward_fns(trainer_config, make_reward_fn=lambda fn: fn)
    self.assertEqual(
        reward_fns,
        [
            train_rl.utils_rl.match_format_exactly,
            train_rl.utils_rl.match_format_approximately,
            train_rl.utils_rl.check_numbers,
        ],
    )

  def test_build_reward_fns_custom_replaces_builtins(self):
    """When both knobs are set, the stack is the user-provided functions only."""
    trainer_config = SimpleNamespace(
        reward_functions_path="/tmp/my_rewards.py",
        reward_functions="reward_a, reward_b",
    )
    loaded = {"reward_a": object(), "reward_b": object()}
    with mock.patch.object(
        train_rl.utils_rl, "load_custom_callable", side_effect=lambda path, name, **kwargs: loaded[name]
    ) as mock_load:
      reward_fns = train_rl.build_reward_fns(trainer_config, make_reward_fn=lambda fn: fn)
    self.assertEqual(reward_fns, [loaded["reward_a"], loaded["reward_b"]])
    mock_load.assert_has_calls(
        [
            mock.call("/tmp/my_rewards.py", "reward_a", importable_by_workers=True),
            mock.call("/tmp/my_rewards.py", "reward_b", importable_by_workers=True),
        ]
    )

  def test_build_reward_fns_partial_config_falls_back(self):
    """If only one of the two knobs is set, the built-in stack is used."""
    trainer_config = SimpleNamespace(reward_functions_path="/tmp/my_rewards.py", reward_functions="")
    reward_fns = train_rl.build_reward_fns(trainer_config, make_reward_fn=lambda fn: fn)
    self.assertEqual(len(reward_fns), 3)

  def test_build_reward_fns_uses_trajectory_reward_passthrough_when_env_class_set(self):
    """When rl.env_class_path is set without custom reward functions, trajectory_reward_passthrough is used."""
    trainer_config = SimpleNamespace(
        reward_functions_path="",
        reward_functions="",
        rl=SimpleNamespace(env_class_path="tunix.rl.agentic.environments.code.code_env.PythonCodeEnv"),
    )
    reward_fns = train_rl.build_reward_fns(trainer_config, make_reward_fn=lambda fn: fn)
    self.assertEqual(reward_fns, [train_rl.utils_rl.trajectory_reward_passthrough])
    self.assertEqual(
        reward_fns[0](prompts=["p1", "p2"], completions=["c1", "c2"], trajectory_rewards=[1.0, 0.0]),
        [1.0, 0.0],
    )

  def test_rl_config_mimo_and_mini_batch_train_steps(self):
    """Verify MimoConfig, RL fields, and train_steps scaling with mini_batch_size."""
    cfg = types.RLConfig(
        model_name="qwen3-4b",
        batch_size=8,
        num_batches=150,
        num_iterations=1,
        train_fraction=1.0,
        num_epoch=1,
        mini_batch_size=2,
        loss_algo="grpo",
        advantage_estimator="mimo",
        token_advantage_mode="mimo",
        clip_mode="ppo",
        grpo_beta=0.0,
        grpo_epsilon=0.20,
        epsilon_high=0.28,
        epsilon_neg=0.20,
        epsilon_neg_high=0.28,
        mimo={"lp_pass_rate_threshold": 0.5, "span_kappa": 1.5},
        env_class_path="tunix.rl.agentic.environments.code.code_env.PythonCodeEnv",
        env_kwargs={"max_steps": 3, "test_timeout_s": 3.0},
        group_postprocessor_path=(
            "tunix.rl.agentic.environments.code.code_judge_postprocessor.CodeJudgeGroupPostProcessor"
        ),
    )
    self.assertEqual(cfg.train_steps, 600)
    self.assertEqual(cfg.rl.mini_batch_size, 2)
    self.assertEqual(cfg.rl.advantage_estimator, "mimo")
    self.assertEqual(cfg.rl.token_advantage_mode, "mimo")
    self.assertEqual(cfg.rl.clip_mode, "ppo")
    self.assertIsNotNone(cfg.rl.mimo)
    self.assertEqual(cfg.rl.mimo.lp_pass_rate_threshold, 0.5)
    self.assertEqual(cfg.rl.mimo.span_kappa, 1.5)
    self.assertAlmostEqual(cfg.rl.mimo.span_beta_pos, 0.5)
    self.assertTrue(cfg.rl.mimo.span_renormalize)
    self.assertEqual(cfg.rl.env_kwargs, {"max_steps": 3, "test_timeout_s": 3.0})

  @mock.patch("datasets.load_dataset")
  def test_get_dataset_selects_json_builder_for_jsonl_files(self, mock_load):
    """get_dataset uses the 'json' builder when data_files is .jsonl/.json or dataset_name is 'json'."""
    cfg = SimpleNamespace(debug=False, hf_subset=None)
    train_rl.get_dataset(cfg, split="train", data_files="/tmp/code_rl_train.jsonl", dataset_name="json")
    mock_load.assert_called_once_with(
        "json",
        data_files={"train": "/tmp/code_rl_train.jsonl"},
        split="train",
    )
    mock_load.reset_mock()
    train_rl.get_dataset(cfg, split="train", data_files="/tmp/data.parquet", dataset_name="openai/gsm8k")
    mock_load.assert_called_once_with(
        "parquet",
        data_files={"train": "/tmp/data.parquet"},
        split="train",
    )

  def test_build_group_postprocessor_sets_local_rules_only_when_no_judge_url(self):
    """_build_group_postprocessor forwards local_rules_only=True when judge_base_url is empty."""

    class CapturingPostProcessor:

      def __init__(self, tokenizer=None, judge_base_url="", local_rules_only=False):
        self.tokenizer = tokenizer
        self.judge_base_url = judge_base_url
        self.local_rules_only = local_rules_only

      def process_group(self, group):
        return group

    with mock.patch.object(train_rl.utils_rl, "load_custom_callable", return_value=CapturingPostProcessor):
      cfg_no_judge = types.RLConfig(
          model_name="qwen3-0.6b",
          group_postprocessor_path="dummy.CapturingPostProcessor",
          judge_base_url="",
      )
      post_no_judge = train_rl._build_group_postprocessor(cfg_no_judge, model_tokenizer="tok")  # pylint: disable=protected-access
      self.assertTrue(post_no_judge.local_rules_only)

      cfg_with_judge = types.RLConfig(
          model_name="qwen3-0.6b",
          group_postprocessor_path="dummy.CapturingPostProcessor",
          judge_base_url="http://127.0.0.1:8000/v1",
      )
      post_with_judge = train_rl._build_group_postprocessor(cfg_with_judge, model_tokenizer="tok")  # pylint: disable=protected-access
      self.assertFalse(post_with_judge.local_rules_only)

  def test_create_models_and_meshes_skips_reference_when_grpo_beta_zero(self):
    """When grpo_beta == 0.0, create_models_and_meshes skips reference_model creation."""
    trainer_config = SimpleNamespace(
        rl=SimpleNamespace(grpo_beta=0.0),
        grpo_beta=0.0,
        load_checkpoint_only_once=True,
    )
    sampler_config = SimpleNamespace(mesh_axes=("data",))
    fake_actor = object()
    fake_mesh = object()
    with (
        mock.patch.object(model_creation_utils, "from_pretrained", return_value=(fake_actor, fake_mesh)) as mock_fp,
        mock.patch.object(model_creation_utils.maxtext_utils, "create_device_mesh", return_value=np.zeros((1,))),
        mock.patch.object(model_creation_utils, "Mesh", return_value=object()),
    ):
      ref_model, ref_mesh, actor_model, actor_mesh, _ = model_creation_utils.create_models_and_meshes(
          trainer_config,
          sampler_config,
          trainer_devices=[object()],
          sampler_devices=[object()],
          tokenizer_pad_id=0,
      )
    mock_fp.assert_called_once()
    self.assertIsNone(ref_model)
    self.assertIsNone(ref_mesh)
    self.assertIs(actor_model, fake_actor)
    self.assertIs(actor_mesh, fake_mesh)

  def test_create_rl_components_agentic_wires_env_mimo_and_postprocessor(self):
    """AgenticGrpoLearner receives env_class, group_postprocessor, mimo config, and mini_batch_size."""

    @dataclasses.dataclass
    class FakeAgenticGrpoConfig:
      """Fake AgenticGrpoConfig for testing learner kwarg plumbing."""

      num_generations: int = 2
      num_iterations: int = 1
      beta: float = 0.0
      epsilon: float = 0.2
      loss_algo: str = "grpo"
      loss_agg_mode: str = "sequence-mean-token-mean"
      max_response_length: int = 128
      max_concurrency: int = 16
      off_policy_steps: int = 0
      system_prompt: str = ""
      epsilon_high: float | None = None
      use_rollout_logps: bool = True
      advantage_estimator: str = "grpo"
      clip_mode: str = "ppo"
      token_advantage_mode: str = "none"
      mimo: object | None = None

    captured_learner_kwargs = {}
    captured_training_kwargs = {}
    captured_rollout_kwargs = {}

    class FakeAgenticGrpoLearner:
      """Fake AgenticGrpoLearner for capturing initialization arguments."""

      def __init__(
          self,
          rl_cluster,
          reward_fns,
          algo_config,
          chat_parser=None,
          metric_fns=None,
          env_class=None,
          env_kwargs=None,
          group_postprocessor=None,
      ):
        del rl_cluster, reward_fns
        captured_learner_kwargs.update(
            algo_config=algo_config,
            chat_parser=chat_parser,
            metric_fns=metric_fns,
            env_class=env_class,
            env_kwargs=env_kwargs,
            group_postprocessor=group_postprocessor,
        )

    class FakeDummyEnv:
      pass

    class FakePostProcessor:

      def __init__(self, tokenizer=None, judge_base_url="", local_rules_only=False):
        self.tokenizer = tokenizer
        self.judge_base_url = judge_base_url
        self.local_rules_only = local_rules_only

      def process_group(self, group):
        return group

    trainer_config = types.RLConfig(
        model_name="qwen3-0.6b",
        enable_checkpointing=False,
        enable_tunix_perf_metrics=False,
        batch_size=8,
        mini_batch_size=2,
        decode_sampling_top_k=0,
        use_agentic_rollout=True,
        grpo_beta=0.0,
        loss_algo="grpo",
        advantage_estimator="mimo",
        token_advantage_mode="mimo",
        clip_mode="ppo",
        mimo={"lp_pass_rate_threshold": 0.5, "span_kappa": 1.5},
        env_class_path="dummy.env.FakeDummyEnv",
        env_kwargs={"max_steps": 3},
        group_postprocessor_path="dummy.post.FakePostProcessor",
        judge_base_url="http://127.0.0.1:8000/v1",
    )

    def fake_load_callable(path, fn_name=None, **kwargs):
      del fn_name, kwargs
      if "FakeDummyEnv" in path:
        return FakeDummyEnv
      if "FakePostProcessor" in path:
        return FakePostProcessor
      raise ValueError(path)

    fake_agentic_mod = SimpleNamespace(GrpoConfig=FakeAgenticGrpoConfig, GrpoLearner=FakeAgenticGrpoLearner)
    with (
        mock.patch.dict(sys.modules, {"tunix.rl.agentic.agentic_grpo_learner": fake_agentic_mod}),
        mock.patch.object(train_rl.utils_rl, "get_optimizer"),
        mock.patch.object(train_rl.utils_rl, "load_custom_callable", side_effect=fake_load_callable),
        mock.patch.object(train_rl.pyconfig, "initialize", return_value=SimpleNamespace(logical_axis_rules=[])),
        mock.patch.object(
            train_rl.base_rollout,
            "RolloutConfig",
            side_effect=lambda **kw: captured_rollout_kwargs.update(kw) or SimpleNamespace(**kw),
        ),
        mock.patch.object(train_rl.rl_cluster_lib, "ClusterConfig"),
        mock.patch.object(
            train_rl.rl_cluster_lib,
            "RLTrainingConfig",
            side_effect=lambda **kw: captured_training_kwargs.update(kw) or SimpleNamespace(**kw),
        ),
        mock.patch.object(train_rl.rl_cluster_lib, "RLCluster"),
        mock.patch.object(train_rl, "get_rollout_kwargs_for_parallelism", return_value={}),
    ):
      train_rl.create_rl_components(
          trainer_config=trainer_config,
          sampler_config=SimpleNamespace(enable_expert_parallel=False),
          sampler_devices=[object()],
          actor_model=object(),
          actor_mesh=object(),
          reference_model=None,
          reference_mesh=None,
          rollout_mesh=object(),
          model_tokenizer="fake_tok",
      )

    self.assertEqual(captured_training_kwargs["mini_batch_size"], 2)
    self.assertIsNone(captured_rollout_kwargs["top_k"])
    self.assertIsNone(captured_learner_kwargs["chat_parser"])
    self.assertIs(captured_learner_kwargs["env_class"], FakeDummyEnv)
    self.assertEqual(captured_learner_kwargs["env_kwargs"], {"max_steps": 3})
    self.assertIsInstance(captured_learner_kwargs["group_postprocessor"], FakePostProcessor)
    self.assertEqual(captured_learner_kwargs["group_postprocessor"].judge_base_url, "http://127.0.0.1:8000/v1")
    self.assertFalse(captured_learner_kwargs["group_postprocessor"].local_rules_only)
    self.assertEqual(captured_learner_kwargs["algo_config"].advantage_estimator, "mimo")
    self.assertEqual(captured_learner_kwargs["algo_config"].clip_mode, "ppo")
    self.assertEqual(captured_learner_kwargs["algo_config"].token_advantage_mode, "mimo")
    self.assertIsNotNone(captured_learner_kwargs["algo_config"].mimo)

  def test_forward_only_old_logps_and_judge_audit_only_and_thinking_plumbing(self):
    """Verifies forward_only_old_logps, compute_logps_chunk_size, judge_audit_only, and enable_thinking wire cleanly."""

    class CapturingPostProcessor:
      """Stub postprocessor that records judge kwargs."""

      def __init__(
          self,
          tokenizer=None,
          judge_base_url="",
          local_rules_only=False,
          audit_only=False,
          max_concurrency=64,
          max_faulty_segments_per_turn=1,
      ):
        self.tokenizer = tokenizer
        self.judge_base_url = judge_base_url
        self.local_rules_only = local_rules_only
        self.audit_only = audit_only
        self.max_concurrency = max_concurrency
        self.max_faulty_segments_per_turn = max_faulty_segments_per_turn

      def process_group(self, group):
        return group

    class DummyCodeEnv:
      pass

    cfg = types.RLConfig(
        model_name="qwen3-0.6b",
        use_agentic_rollout=True,
        forward_only_old_logps=True,
        compute_logps_chunk_size=256,
        judge_audit_only=True,
        judge_max_concurrency=128,
        judge_max_flagged_segments_per_turn=1,
        enable_thinking=True,
        env_class_path="dummy.DummyCodeEnv",
        env_kwargs={"max_steps": 3},
        group_postprocessor_path="dummy.CapturingPostProcessor",
    )
    training_cfg, grpo_cfg = train_rl.build_tunix_rl_configs(cfg)
    self.assertEqual(training_cfg.compute_logps_chunk_size, 256)
    self.assertTrue(grpo_cfg.forward_only_old_logps)

    with mock.patch.object(train_rl.utils_rl, "load_custom_callable", return_value=DummyCodeEnv):
      env_cls, env_kwargs = train_rl._resolve_env_class(cfg)  # pylint: disable=protected-access
    self.assertIs(env_cls, DummyCodeEnv)
    self.assertEqual(env_kwargs, {"max_steps": 3, "enable_thinking": True})

    with mock.patch.object(train_rl.utils_rl, "load_custom_callable", return_value=CapturingPostProcessor):
      postproc = train_rl._build_group_postprocessor(cfg, model_tokenizer="tok")  # pylint: disable=protected-access
    self.assertTrue(postproc.audit_only)
    self.assertEqual(postproc.max_concurrency, 128)
    self.assertEqual(postproc.max_faulty_segments_per_turn, 1)

  @pytest.mark.cpu_only
  def test_gspo_ppo_clip_mode_and_turn_span_plumbing(self):
    """Verifies GSPO clip_mode='ppo', turn_span pooling, gspo_length_adaptive_clipping, and min_seq_len_norm."""
    trainer_cfg = types.RLConfig(
        model_name="qwen3-4b",
        enable_checkpointing=False,
        enable_tunix_perf_metrics=False,
        rl={
            "use_agentic_rollout": True,
            "loss_algo": "gspo-token",
            "gspo_pool_mode": "turn_span",
            "clip_mode": "ppo",
            "grpo_epsilon": 0.005,
            "epsilon_high": 0.008,
            "gspo_length_adaptive_clipping": True,
            "gspo_ref_length": 1024,
            "gspo_clip_min": 0.001,
            "gspo_clip_max": 0.05,
            "min_seq_len_norm": 128,
            "max_flagged_token_ratio_per_turn": 0.35,
            "group_postprocessor_path": (
                "tunix.rl.agentic.environments.code.code_judge_postprocessor.CodeJudgeGroupPostProcessor"
            ),
        },
    )
    training_cfg, grpo_cfg = train_rl.build_tunix_rl_configs(trainer_cfg)
    self.assertIsNotNone(training_cfg)
    self.assertEqual(grpo_cfg.loss_algo, "gspo-token")
    self.assertEqual(grpo_cfg.gspo_pool_mode, "turn_span")
    self.assertEqual(grpo_cfg.clip_mode, "ppo")
    self.assertTrue(grpo_cfg.gspo_length_adaptive_clipping)
    self.assertEqual(grpo_cfg.gspo_ref_length, 1024)
    self.assertAlmostEqual(float(grpo_cfg.gspo_clip_min), 0.001)
    self.assertAlmostEqual(float(grpo_cfg.gspo_clip_max), 0.05)
    self.assertAlmostEqual(float(grpo_cfg.min_seq_len_norm), 128.0)

    postproc = train_rl._build_group_postprocessor(  # pylint: disable=protected-access
        trainer_cfg, model_tokenizer="fake_tok"
    )
    self.assertIsNotNone(postproc)
    self.assertAlmostEqual(float(postproc.max_flagged_token_ratio_per_turn), 0.35)


class CanonicalCodeRlYamlConfigTest(unittest.TestCase):
  """Tests that the canonical composable coding RL YAML configs load and build Tunix configs."""

  CANONICAL_CONFIGS = (
      "rl_code_gspo_mimo.yml",
      "rl_code_gspo_mimo_turn_span.yml",
      "rl_code_grpo_mimo.yml",
      "rl_code_grpo_mimo_offpolicy.yml",
      "rl_code_gspo_mimo_turn_span_offpolicy.yml",
  )

  def _load_rl_config(self, config_filename: str, extra_args: list[str] | None = None) -> pyconfig.HyperParameters:
    config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", config_filename)
    argv = ["", config_path]
    if extra_args:
      argv.extend(extra_args)
    return pyconfig.initialize(argv, config_class=types.RLConfig)

  def test_canonical_yaml_configs_load_and_build_tunix_configs(self):
    """Verifies all 5 canonical YAML configs load via pyconfig and build valid Tunix RL configs."""
    expected = {
        "rl_code_gspo_mimo.yml": ("gspo-token", "sequence", False, 0.005, 0.008, 1, 600),
        "rl_code_gspo_mimo_turn_span.yml": ("gspo-token", "turn_span", True, 0.005, 0.008, 1, 600),
        "rl_code_grpo_mimo.yml": ("grpo", "sequence", False, 0.20, 0.28, 1, 600),
        "rl_code_grpo_mimo_offpolicy.yml": ("grpo", "sequence", False, 0.20, 0.28, 2, 1200),
        "rl_code_gspo_mimo_turn_span_offpolicy.yml": ("gspo-token", "turn_span", True, 0.005, 0.008, 2, 1200),
    }
    for cfg_name in self.CANONICAL_CONFIGS:
      with self.subTest(config=cfg_name):
        cfg = self._load_rl_config(cfg_name)
        exp_algo, exp_pool, exp_adaptive, exp_eps, exp_eps_high, exp_iters, exp_steps = expected[cfg_name]
        self.assertEqual(cfg.model_name, "qwen3-4b")
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.rl.mini_batch_size, 2)
        self.assertEqual(cfg.rl.num_generations, 8)
        self.assertEqual(cfg.rl.num_iterations, exp_iters)
        self.assertEqual(cfg.train_steps, exp_steps)
        self.assertEqual(cfg.rl.loss_algo, exp_algo)
        self.assertEqual(cfg.rl.gspo_pool_mode, exp_pool)
        self.assertEqual(cfg.rl.clip_mode, "ppo")
        self.assertEqual(cfg.rl.loss_agg_mode, "sequence-mean-token-mean")
        self.assertEqual(cfg.rl.min_seq_len_norm, 128)
        self.assertEqual(cfg.rl.advantage_estimator, "mimo")
        self.assertEqual(cfg.rl.token_advantage_mode, "mimo")
        self.assertAlmostEqual(float(cfg.rl.grpo_epsilon), exp_eps)
        self.assertAlmostEqual(float(cfg.rl.epsilon_high), exp_eps_high)
        self.assertEqual(bool(cfg.rl.gspo_length_adaptive_clipping), exp_adaptive)
        self.assertEqual(cfg.rl.gspo_ref_length, 1024)
        self.assertAlmostEqual(float(cfg.rl.gspo_clip_min), 0.001)
        self.assertAlmostEqual(float(cfg.rl.gspo_clip_max), 0.05)
        self.assertAlmostEqual(float(cfg.rl.max_flagged_token_ratio_per_turn), 0.35)
        self.assertIsNotNone(cfg.rl.mimo)
        self.assertTrue(cfg.rl.mimo.span_renormalize)
        self.assertAlmostEqual(float(cfg.rl.mimo.span_beta_pos), 0.5)

        training_cfg, grpo_cfg = train_rl.build_tunix_rl_configs(cfg)
        self.assertEqual(training_cfg.mini_batch_size, 2)
        self.assertEqual(training_cfg.max_steps, exp_steps)
        self.assertEqual(grpo_cfg.loss_algo, exp_algo)
        self.assertEqual(grpo_cfg.gspo_pool_mode, exp_pool)
        self.assertEqual(grpo_cfg.clip_mode, "ppo")
        self.assertEqual(grpo_cfg.loss_agg_mode, "sequence-mean-token-mean")
        self.assertEqual(grpo_cfg.min_seq_len_norm, 128)
        self.assertEqual(bool(grpo_cfg.gspo_length_adaptive_clipping), exp_adaptive)
        self.assertEqual(grpo_cfg.num_iterations, exp_iters)


class TokenizerChatTemplateTest(unittest.TestCase):
  """Unit tests for configure_tokenizer_chat_template."""

  def test_chat_template_populated_from_config_string(self):
    """Test that chat_template is set from config.chat_template when tokenizer lacks one."""
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.chat_template = None
    trainer_config = SimpleNamespace(
        chat_template="{{ messages[0].content }}",
        chat_template_path=None,
        tokenizer_path="dummy-base-model",
    )
    train_rl.configure_tokenizer_chat_template(mock_tokenizer, trainer_config)
    self.assertEqual(mock_tokenizer.chat_template, "{{ messages[0].content }}")

  @mock.patch("maxtext.input_pipeline.instruction_data_processing.load_chat_template_from_file")
  def test_chat_template_populated_from_config_file(self, mock_load):
    """Test that chat_template is loaded from chat_template_path when tokenizer lacks one."""
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.chat_template = None
    mock_load.return_value = "{% for message in messages %}{{ message.content }}{% endfor %}"
    trainer_config = SimpleNamespace(
        chat_template=None,
        chat_template_path="/path/to/jinja_template.json",
        tokenizer_path="dummy-base-model",
    )
    train_rl.configure_tokenizer_chat_template(mock_tokenizer, trainer_config)
    mock_load.assert_called_once_with("/path/to/jinja_template.json")
    self.assertEqual(
        mock_tokenizer.chat_template,
        "{% for message in messages %}{{ message.content }}{% endfor %}",
    )

  def test_chat_template_raises_value_error_when_empty(self):
    """Test that ValueError is raised when tokenizer lacks chat_template and both config options are empty."""
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.chat_template = None
    trainer_config = SimpleNamespace(
        chat_template=None,
        chat_template_path=None,
        tokenizer_path="dummy-base-model",
    )
    with self.assertRaisesRegex(ValueError, "Tokenizer 'dummy-base-model' has no chat_template"):
      train_rl.configure_tokenizer_chat_template(mock_tokenizer, trainer_config)

  def test_chat_template_unchanged_when_already_exists(self):
    """Test that an existing chat_template on the tokenizer is preserved (backward compatibility)."""
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.chat_template = "{{ existing_template }}"
    trainer_config = SimpleNamespace(
        chat_template="{{ overridden_template }}",
        chat_template_path=None,
        tokenizer_path="dummy-instruction-tuned-model",
    )
    train_rl.configure_tokenizer_chat_template(mock_tokenizer, trainer_config)
    self.assertEqual(mock_tokenizer.chat_template, "{{ existing_template }}")

  def test_apply_chat_template_works_after_configuration(self):
    """Verifies apply_chat_template succeeds and produces the expected format after our code path runs."""

    class DummyTokenizer:  # pylint: disable=missing-class-docstring

      def __init__(self):
        self.chat_template = None

      def apply_chat_template(self, conversation, tokenize=False):
        if self.chat_template is None:
          raise ValueError("Cannot apply chat template because chat_template is None")
        import jinja2  # pylint: disable=import-outside-toplevel

        env = jinja2.Environment()
        template = env.from_string(self.chat_template)
        return template.render(messages=conversation)

    tokenizer = DummyTokenizer()
    trainer_config = SimpleNamespace(
        chat_template="{{ messages[0].content }}",
        chat_template_path=None,
        tokenizer_path="dummy-base-model",
    )
    # Initially, apply_chat_template fails (simulating HF tokenizer crash when chat_template is None)
    with self.assertRaises(ValueError):
      tokenizer.apply_chat_template([{"role": "user", "content": "Hello!"}])
    # Run the proposed change
    train_rl.configure_tokenizer_chat_template(tokenizer, trainer_config)
    # Verify apply_chat_template now runs successfully and renders correct content
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Hello!"}])
    self.assertEqual(rendered, "Hello!")


class RLGoodputTest(unittest.TestCase):
  """Unit tests for Goodput recording in RL training and hooks."""

  def test_rl_training_hooks_goodput_with_eval_interval_zero(self):
    """Verify RLTrainingHooks records step start time and does not raise ZeroDivisionError when eval_interval=0."""
    from maxtext.trainers.post_train.rl import hooks as rl_hooks  # pylint: disable=import-outside-toplevel

    mock_recorder = mock.MagicMock()
    trainer_config = SimpleNamespace(
        enable_goodput_recording=True,
        eval_interval=0,
        num_test_batches=0,
        num_eval_passes=1,
        eval_corr_lst=False,
        eval_make_lst=False,
    )
    rl_cluster = SimpleNamespace(global_steps=0, actor_trainer=SimpleNamespace(training_hooks=None))

    train_rl.utils_rl.install_training_hooks(
        rl_cluster, trainer_config, test_dataset=None, goodput_recorder=mock_recorder
    )
    hook = rl_cluster.actor_trainer.training_hooks
    self.assertIsInstance(hook, rl_hooks.RLTrainingHooks)

    # Simulate step 0 and step 1 start/end
    train_ctx = SimpleNamespace(train_steps=0)
    hook.on_train_step_start(train_ctx)
    hook.on_train_step_end(trainer=train_ctx, step=1, loss=0.5)  # Must not raise ZeroDivisionError

    train_ctx.train_steps = 1
    hook.on_train_step_start(train_ctx)
    hook.on_train_step_end(trainer=train_ctx, step=2, loss=0.4)

    self.assertEqual(mock_recorder.record_step_start_time.call_args_list, [mock.call(0), mock.call(1)])

  def _run_rl_train_with_goodput(self, enable_goodput_recording: bool):
    """Helper to run rl_train with mocked components and goodput_recorder."""
    mock_recorder = mock.MagicMock()
    trainer_config = SimpleNamespace(
        enable_goodput_recording=enable_goodput_recording,
        monitor_goodput=False,
        goodput_job_name="test-rl-job",
        run_name="test-rl-run",
        debug=SimpleNamespace(rl=False),
        load_parameters_path="gs://dummy/ckpt",
        tokenizer_path="dummy_tokenizer",
        hf_access_token=None,
        checkpoint_dir="/tmp/rl_ckpt",
        tensorboard_dir="/tmp/rl_tb",
        profiler="",
        managed_mldiagnostics=False,
        steps=1,
        batch_size=1,
        num_batches=1,
        num_test_batches=0,
        eval_interval=0,
        num_eval_passes=1,
        eval_corr_lst=False,
        eval_make_lst=False,
        train_fraction=1.0,
        load_checkpoint_only_once=False,
        save_checkpoint_on_completion=False,
        enable_single_controller=False,
        reward_functions_path="",
        reward_functions="",
        rl=SimpleNamespace(use_agentic_rollout=False, profiler_max_num_hosts=1),
    )
    sampler_config = SimpleNamespace(debug=SimpleNamespace(rl=False))

    def fake_create_recorder(cfg):
      return mock_recorder if cfg.enable_goodput_recording else None

    mock_rl_cluster = SimpleNamespace(
        global_steps=0, actor_trainer=SimpleNamespace(training_hooks=None), close=mock.MagicMock()
    )
    mock_rl_trainer = mock.MagicMock()

    def fake_train(dataset):  # pylint: disable=unused-argument
      hook = mock_rl_cluster.actor_trainer.training_hooks
      if hook is not None:
        hook.on_train_step_start(SimpleNamespace(train_steps=0))
        hook.on_train_step_end(trainer=None, step=1, loss=1.0)

    mock_rl_trainer.train.side_effect = fake_train

    with (
        mock.patch.object(train_rl, "create_goodput_recorder", side_effect=fake_create_recorder),
        mock.patch.object(
            train_rl.model_creation_utils,
            "setup_configs_and_devices",
            return_value=(trainer_config, sampler_config, [mock.MagicMock()], [mock.MagicMock()]),
        ),
        mock.patch.object(
            train_rl.model_creation_utils,
            "create_models_and_meshes",
            return_value=(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(train_rl.epath, "Path", return_value=mock.MagicMock()),
        mock.patch.object(train_rl.AutoTokenizer, "from_pretrained", return_value=mock.MagicMock()),
        mock.patch.object(train_rl, "configure_tokenizer_chat_template"),
        mock.patch.object(train_rl, "prepare_datasets", return_value=(["batch"], [])),
        mock.patch.object(
            train_rl, "create_rl_components", return_value=(mock_rl_cluster, mock_rl_trainer, mock.MagicMock(), [])
        ),
        mock.patch.object(train_rl.jax, "effects_barrier"),
    ):
      train_rl.rl_train(["dummy"], {})

    return mock_recorder

  def test_rl_train_goodput_enabled_records_lifecycle_and_steps(self):
    """Verify rl_train with enable_goodput_recording=True records all lifecycle and step events."""
    mock_recorder = self._run_rl_train_with_goodput(enable_goodput_recording=True)
    mock_recorder.record_job_start_time.assert_called_once()
    mock_recorder.record_tpu_init_start_time.assert_called_once()
    mock_recorder.record_tpu_init_end_time.assert_called_once()
    mock_recorder.record_training_preparation_start_time.assert_called_once()
    mock_recorder.record_training_preparation_end_time.assert_called_once()
    mock_recorder.record_step_start_time.assert_called_once_with(0)
    mock_recorder.record_job_end_time.assert_called_once()

  def test_rl_train_goodput_disabled_performs_zero_recorder_calls(self):
    """Verify rl_train with enable_goodput_recording=False performs zero recorder calls."""
    mock_recorder = self._run_rl_train_with_goodput(enable_goodput_recording=False)
    self.assertEqual(len(mock_recorder.method_calls), 0)


if __name__ == "__main__":
  unittest.main()
