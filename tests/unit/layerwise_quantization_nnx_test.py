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

"""Unit tests for the NNX path of layerwise_quantization.

Covers `_strip_kernels_at_quantized_paths` — the convert→serve shape converter
that drops the redundant full-precision kernel from quantized DenseGeneral
nodes while leaving non-quantized kernels (norms, embeddings) intact.
"""

import unittest

from maxtext.utils.layerwise_quantization import LayerwiseQuantization


class StripKernelsTest(unittest.TestCase):

  def test_drops_kernel_at_quantized_dense(self):
    """A node with both `kernel` and `AqtDotGeneral_0` loses the kernel."""
    state = {
        "decoder": {
            "layers": {
                "mlp": {
                    "wi": {
                        "kernel": "FULL_PRECISION_W",
                        "AqtDotGeneral_0": {"qrhs": {"frozen": "AQT_STATE"}},
                    }
                }
            }
        }
    }
    out = LayerwiseQuantization._strip_kernels_at_quantized_paths(state)  # pylint: disable=protected-access
    wi = out["decoder"]["layers"]["mlp"]["wi"]
    self.assertNotIn("kernel", wi)
    self.assertIn("AqtDotGeneral_0", wi)
    self.assertEqual(wi["AqtDotGeneral_0"]["qrhs"]["frozen"], "AQT_STATE")

  def test_preserves_non_quantized_kernel(self):
    """A non-quantized kernel (e.g. embedding, norm) survives."""
    state = {
        "decoder": {
            "decoder_norm": {"scale": "NORM_SCALE"},
            "logits_dense": {"kernel": "LOGITS_KERNEL"},  # no AqtDotGeneral_0 sibling
        },
        "token_embedder": {"embedding": "EMB"},
    }
    out = LayerwiseQuantization._strip_kernels_at_quantized_paths(state)  # pylint: disable=protected-access
    self.assertEqual(out["decoder"]["logits_dense"]["kernel"], "LOGITS_KERNEL")
    self.assertEqual(out["decoder"]["decoder_norm"]["scale"], "NORM_SCALE")
    self.assertEqual(out["token_embedder"]["embedding"], "EMB")

  def test_mixed_tree(self):
    """Quantized + non-quantized at the same depth: only the quantized one strips."""
    state = {
        "self_attention": {
            "qkv_proj": {"kernel": "QKV", "AqtDotGeneral_0": "AQT"},
            "out": {"kernel": "OUT_FULL"},  # non-quantized output projection
        }
    }
    out = LayerwiseQuantization._strip_kernels_at_quantized_paths(state)  # pylint: disable=protected-access
    self.assertNotIn("kernel", out["self_attention"]["qkv_proj"])
    self.assertEqual(out["self_attention"]["out"]["kernel"], "OUT_FULL")


if __name__ == "__main__":
  unittest.main()
