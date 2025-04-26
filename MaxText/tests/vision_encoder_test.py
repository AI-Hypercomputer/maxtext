#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Tests for multimodal vision encoder """

import unittest
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jsonlines

from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.globals import PKG_DIR


class VisionEncoderEmbeddingTest(unittest.TestCase):

  CONFIGS = {
      "gemma3-4b": [  # tests decode with multimodal gemma-4b
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "model_name=gemma3-4b",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.gemma3')}",
          "use_multimodal=True",
          "run_name=runner_test",
          "load_parameters_path=gs://hengtaoguo-maxtext-logs/checkpoints/gemma3-4b/vit/unscanned/2025-04-25-18-06-04/checkpoints/0/items",  # mm_input_project, scale + 1, 4b
          "steps=1",
          "enable_checkpointing=False",
          "max_target_length=16",
          "max_prefill_predict_length=8",
          "per_device_batch_size=1",
          "scan_layers=false",
          "enable_checkpointing=true",
          "prompt='Describe this image'",
          rf"image_path={os.path.join(os.path.dirname(PKG_DIR), 'MaxText', 'test_assets', 'test_image.jpg')}",
      ],
  }

  @pytest.mark.tpu_only
  def test_image_embedding_tpu(self):
    # Load golden data from HuggingFace Gemma3-4b
    golden_data_path = os.path.join(os.path.dirname(PKG_DIR), "MaxText", "test_assets", "golden_data_gemma3_vit.jsonl")
    loaded_data = []
    with jsonlines.open(golden_data_path, mode="r") as reader:
      for line in reader:
        loaded_data.append(line)
    golden_image_embeddings = np.asarray(loaded_data[0]["soft_embeddings"], dtype=np.float32)

    # Initialize the model with weights from reference checkpoint
    config = pyconfig.initialize(VisionEncoderEmbeddingTest.CONFIGS["gemma3-4b"])
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = models.Transformer(config, mesh=mesh, quant=quant)
    state, _ = maxtext_utils.setup_decode_state(model, config, rng1, mesh, None)

    # Load and preprocess the image
    images = multimodal_utils.load_image_from_path(config.image_path)
    images = multimodal_utils.pre_process_image(images, model_name=config.model_name)
    input_images = images[jnp.newaxis, jnp.newaxis, ...]

    # Fetch the image embeddings
    image_embeddings = model.apply(
        {"params": state.params["params"]},  # Pass the *entire* params dict for the Transformer
        input_images,
        method=model.encode_images,  # Specify the method to run
    )
    image_embeddings = np.asarray(image_embeddings, dtype=np.float32)

    # Compare the image embeddings with golden data
    mse = np.mean((image_embeddings - golden_image_embeddings) ** 2)
    print(f"MSE between image_embedding and golden data: {mse}")
    self.assertLess(mse, 1e-2, f"Image embedding mismatch with golden data, MSE {mse} exceeds threshold 1e-2")


if __name__ == "__main__":
  unittest.main()
