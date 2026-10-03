"""Small regression checks for lazy loading and Kimi attention mappings."""

import json
import numpy as np
import pytest
from safetensors.numpy import save_file
from maxtext.checkpoint_conversion.to_maxtext import LazyHFLoader
from maxtext.checkpoint_conversion.utils.hf_model_configs import kimi_k3_dict
from maxtext.checkpoint_conversion.utils.hf_shape import KIMI_K3_HF_WEIGHTS_TO_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import KIMI_K3_MAXTEXT_TO_HF_PARAM_MAPPING


@pytest.mark.parametrize("indexed", [False, True])
def test_lazy_loader_single_file_and_indexed(tmp_path, indexed):
  expected = np.arange(6, dtype=np.float32).reshape(2, 3)
  save_file({"projection.weight": expected}, str(tmp_path / "model.safetensors"))
  if indexed:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"projection.weight": "model.safetensors"}})
    )
  loader = LazyHFLoader(str(tmp_path), token=None, save_dtype="float32")
  np.testing.assert_array_equal(loader.get_tensor("projection.weight"), expected)


def test_final_full_attention_layer_mapping():
  config = dict(kimi_k3_dict, num_experts=2)
  shapes = KIMI_K3_HF_WEIGHTS_TO_SHAPE(config)
  mapping = KIMI_K3_MAXTEXT_TO_HF_PARAM_MAPPING(config, None)
  assert "model.layers.92.self_attn.q_a_proj.weight" in shapes
  assert "model.layers.92.self_attn.q_proj.weight" not in shapes
  assert mapping["params-decoder-layers_92-self_attention-wq_a-kernel"] == "model.layers.92.self_attn.q_a_proj.weight"


def test_lazy_loader_mxfp4_zero_weights(tmp_path):
  pytest.importorskip("compressed_tensors.compressors.mxfp4")
  save_file(
      {
          "projection.weight_packed": np.zeros((4, 16), dtype=np.uint8),
          "projection.weight_scale": np.full((4, 1), 127, dtype=np.uint8),
      },
      str(tmp_path / "model.safetensors"),
  )
  (tmp_path / "model.safetensors.index.json").write_text(
      json.dumps(
          {
              "weight_map": {
                  "projection.weight_packed": "model.safetensors",
                  "projection.weight_scale": "model.safetensors",
              }
          }
      )
  )
  loader = LazyHFLoader(str(tmp_path), token=None, save_dtype="float32")
  np.testing.assert_array_equal(loader.get_tensor("projection.weight"), np.zeros((4, 32)))
