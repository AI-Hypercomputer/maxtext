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

"""PLE hash metadata must survive floating-point checkpoint conversion."""
import subprocess
import sys
import numpy as np
import pytest
from orbax.checkpoint import v1 as ocp

torch = pytest.importorskip("torch")
from maxtext.checkpoint_conversion.to_maxtext import _convert_tensor_to_numpy, _get_maxtext_weight, LazyTensor
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.hf_shape import QWEN38_HF_WEIGHTS_TO_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import QWEN38_MAXTEXT_TO_HF_PARAM_HOOK_FN


def test_integer_tensor_conversion_preserves_exact_hash_metadata():

  source = torch.tensor([2**62 + 3, 20000003, 320001535], dtype=torch.int64)
  for dtype in ["bfloat16", "float32"]:
    converted = _convert_tensor_to_numpy(source, dtype)
    assert converted.dtype == np.int64
    np.testing.assert_array_equal(converted, source.numpy())


def test_lazy_integer_leaf_preserves_dtype():

  source = np.array([2**62 + 3, 20000003], dtype=np.int64)
  result = [None]
  _get_maxtext_weight(
      lambda: source, 0, (2,), "params-decoder-layers_1-w-ple.ple_embedding.layer_multipliers", result, "bfloat16", True
  )
  assert result[0].dtype == np.dtype(np.int64)
  np.testing.assert_array_equal(np.asarray(result[0]), source)


def test_lazy_checkpoint_v1_save_and_standard_restore(tmp_path):

  source = np.arange(12, dtype=np.float32).reshape(3, 4)
  lazy = LazyTensor(lambda: source, source.shape, source.dtype)
  path = tmp_path / "checkpoint"
  ocp.save_checkpointables(path, {"items": {"weight": lazy}})
  # A fresh process must restore using only standard Orbax handlers.

  subprocess.run(
      [
          sys.executable,
          "-c",
          "from orbax.checkpoint import v1 as ocp; import numpy as np; "
          "x=ocp.load_checkpointables(__import__('sys').argv[1],{'items':{'weight':np.zeros((3,4),np.float32)}}); "
          "np.testing.assert_array_equal(x['items']['weight'],np.arange(12,dtype=np.float32).reshape(3,4))",
          str(path),
      ],
      check=True,
  )


def test_inverse_mapping_retains_independent_ple_shards_and_projection_orientation():
  cfg = HF_MODEL_CONFIGS["qwen3.8-flash-next"].to_dict()
  cfg["num_hidden_layers"] = 4
  shapes = QWEN38_HF_WEIGHTS_TO_SHAPE(cfg)
  table = [(k, v) for k, v in shapes.items() if "ngram_embedding.shard_" in k]
  assert len(table) == 128
  assert all(shape == [2500012, 160] for _, shape in table)
  cfg["qwen38_ple_compact_rows"] = 1336
  compact = QWEN38_HF_WEIGHTS_TO_SHAPE(cfg)
  prefix = "model.layers.1.ple.ple_embedding."
  assert compact[prefix + "row_ids"] == [1336]
  assert compact[prefix + "ngram_embedding.weight"] == [1336, 160]
  hook = QWEN38_MAXTEXT_TO_HF_PARAM_HOOK_FN({}, None, saving_to_hf=True)["params-decoder-logits_dense-kernel"]
  weight = np.arange(12).reshape(3, 4)
  np.testing.assert_array_equal(hook(weight, (4, 3)), weight.T)
