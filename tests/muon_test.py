import unittest
from absl.testing import parameterized
import os
import jax
from optax.contrib import MuonDimensionNumbers as mdn

from MaxText.muon_utils import get_muon_weight_dimension_numbers
from MaxText import pyconfig, maxtext_utils
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models, quantizations

Transformer = models.transformer_as_linen


"""
python3 -m pytest -v --pyargs tests.muon_test -rP -s
"""

# deepseek2-16b, scanned, q_lora_rank=0
# NOTE: not compatible with deepseek2-236b (q_lora_rank: 1536)
DEEPSEEK2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                "self_attention": {
                    "kv_norm": {"scale": None},
                    "wkv_a": {"kernel": mdn((0,), (-1,))},
                    "wkv_b": {"kernel": mdn((0,), (-2, -1))},
                    "out": {"kernel": mdn((0, -2), (-1,))},
                    "query": {"kernel": mdn((0,), (-2, -1))},  # ds2
                },
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
            },
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,))},  # ds2
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                "self_attention": {
                    "kv_norm": {"scale": None},
                    "wkv_a": {"kernel": mdn((0,), (-1,))},
                    "wkv_b": {"kernel": mdn((0,), (-2, -1))},
                    "out": {"kernel": mdn((0, -2), (-1,))},
                    "query": {"kernel": mdn((0,), (-2, -1))},  # ds2
                },
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
            },
            "decoder_norm": {"scale": None},
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# deepseek3, scanned
DEEPSEEK3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                "self_attention": {
                    "kv_norm": {"scale": None},
                    "wkv_a": {"kernel": mdn((0,), (-1,))},
                    "wkv_b": {"kernel": mdn((0,), (-2, -1))},
                    "out": {"kernel": mdn((0, -2), (-1,))},
                    "q_norm": {"scale": None},  # ds3
                    "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
                    "wq_b": {"kernel": mdn((0,), (-2, -1))},  # ds3
                },
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
            },
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,)), "bias": None},  # ds3
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                "self_attention": {
                    "kv_norm": {"scale": None},
                    "wkv_a": {"kernel": mdn((0,), (-1,))},
                    "wkv_b": {"kernel": mdn((0,), (-2, -1))},
                    "out": {"kernel": mdn((0, -2), (-1,))},
                    "q_norm": {"scale": None},  # ds3
                    "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
                    "wq_b": {"kernel": mdn((0,), (-2, -1))},  # ds3
                },
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
            },
            "decoder_norm": {"scale": None},
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


_GEMMA3_LAYER = {
    "mlp": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
    "post_ffw_norm": {"scale": None},
    "post_self_attention_norm": {"scale": None},
    "pre_ffw_norm": {"scale": None},
    "pre_self_attention_norm": {"scale": None},
    "self_attention": {
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "key_norm": {"scale": None},
        "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "query_norm": {"scale": None},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
    },
}

GEMMA3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {f"layers_{i}": _GEMMA3_LAYER for i in range(6)},
            "layers_remainder": {f"layers_{i}": _GEMMA3_LAYER for i in range(4)},
        },
        "token_embedder": {"embedding": None},
    }
}


LLAMA2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                },
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


class MuonDimensionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("deepseek2_16b", "deepseek2-16b", DEEPSEEK2_DIMENSION_NUMBER),
      ("deepseek3_test", "deepseek3-test", DEEPSEEK3_DIMENSION_NUMBER),
      ("deepseek3_671b", "deepseek3-671b", DEEPSEEK3_DIMENSION_NUMBER),
      ("llama2_7b", "llama2-7b", LLAMA2_DIMENSION_NUMBER),
      ("gemma3_4b", "gemma3-4b", GEMMA3_DIMENSION_NUMBER),
  )
  def test_model_integration(self, model_name, expected_output):
    """
    Initializes the specified MaxText model and asserts that the calculated
    Muon dimension numbers match the hardcoded reference.
    """
    # Setup config
    argv = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), f"model_name={model_name}", "scan_layers=True"]
    config = pyconfig.initialize(argv)
    # Setup model
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = Transformer(config, mesh=mesh, quant=quant)
    # Run test
    actual_output = get_muon_weight_dimension_numbers(model, config)
    self.assertEqual(actual_output, expected_output)


if __name__ == "__main__":
  unittest.main()
