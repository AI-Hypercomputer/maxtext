"""Tests for test_mmlu_eval_qwen3_0.6b."""

import datetime
import importlib.util
import os
import sys
from unittest import mock

import google.protobuf.runtime_version as _runtime_version
_runtime_version.ValidateProtobufRuntimeVersion = mock.Mock()

import jax
from absl import flags
from absl.testing import absltest


def load_module_from_file(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)  # pytype: disable=attribute-error
  return module


class TestMmluEvalQwen3(absltest.TestCase):

  def test_end_to_end(self):
    model_name = 'qwen3-0.6b'

    workspace = os.environ.get("TEST_SRCDIR", "") + "/" + os.environ.get("TEST_WORKSPACE", "")
    if not os.environ.get("TEST_SRCDIR"):
         workspace = os.getcwd()

    base_yml = os.path.join(workspace, "src/maxtext/configs/base.yml")
    tokenizer_path = os.path.join(workspace, "src/maxtext/assets/tokenizers/qwen3-tokenizer")
    unscanned_ckpt_path = "gs://agagik-test/qwen3/0.6b/unscanned/0/items/"

    src_path = os.path.join(workspace, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

    maxtext_root = workspace

    # Run MMLU eval
    args2 = [
        "mmlu_eval_perplexity",
        base_yml,
        f"tokenizer_path={tokenizer_path}",
        f"model_name={model_name}",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "steps=5",
        "enable_checkpointing=True",
        "skip_jax_distributed_system=True",
        "base_output_directory=/tmp",
        "attention=dot_product",
        "dataset_type=synthetic"
    ]
    
    mmlu_path = os.path.join(maxtext_root, "benchmarks/mmlu/mmlu_eval_perplexity.py")
    mmlu_module = load_module_from_file("mmlu_eval_perplexity", mmlu_path)
    pyconfig_path = os.path.join(maxtext_root, "src/maxtext/configs/pyconfig.py")
    pyconfig_module = load_module_from_file("pyconfig", pyconfig_path)
    
    with mock.patch.object(sys, 'argv', args2):
        try:
          flags.FLAGS(args2)
        except flags.UnrecognizedFlagError:
          pass
        
        cfg = pyconfig_module.initialize(args2)
        mmlu_module.validate_config(cfg)
        with mock.patch('datasets.load_dataset') as mock_load_dataset:
            mock_load_dataset.return_value = [
                {"subject": "anatomy", "question": "1+1", "choices": ["1", "2", "3", "4"], "answer": 1},
            ]
            mmlu_module.main(cfg)

if __name__ == '__main__':
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  absltest.main()