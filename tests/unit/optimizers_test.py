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

"""Unit tests for all optimizers."""
import re
import unittest
from unittest.mock import patch, MagicMock
import jax
import optax
import jax.numpy as jnp

import pytest
from absl.testing import parameterized
from flax import nnx
from optax.contrib import MuonDimensionNumbers as mdn

from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils, muon_utils
from tests.utils.test_helpers import get_test_config_path
from typing import NamedTuple


# deepseek2, specific: q_lora_rank=0
# applicable: deepseek2-16, but not deepseek2-236b (q_lora_rank=1536)
_DEEPSEEK2_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "query": {"kernel": mdn((0,), (-2, -1))},  # ds2
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK2_ATTENTION,
            },
            "logits_dense": {"kernel": None},
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
                **_DEEPSEEK2_ATTENTION,
            },
        },
        "token_embedder": {"embedding": None},
    }
}


# deepseek3
_DEEPSEEK3_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "q_norm": {"scale": None},  # ds3
        "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
        "wq_b": {"kernel": mdn((0,), (-2, -1))},  # ds3
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK3_ATTENTION,
            },
            "logits_dense": {"kernel": None},
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,))},  # ds3
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                **_DEEPSEEK3_ATTENTION,
            },
        },
        "token_embedder": {"embedding": None},
    }
}

# gemma3
_GEMMA3_LAYER = {
    "mlp": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
    "post_ffw_norm": {"scale": None},
    "pre_ffw_norm": {"scale": None},
    "self_attention": {
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
        "key_norm": {"scale": None},
        "query_norm": {"scale": None},
    },
    "post_self_attention_norm": {"scale": None},
    "pre_self_attention_norm": {"scale": None},
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


# llama2 (also llama3)
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
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3, specific: logits_via_embedding=True
# applicable: qwen3-0.6b, qwen3-4b, but not: qwen3-8b, qwen3-14b (logits_via_embedding=False)
QWEN3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "key_norm": {"scale": None},
                    "query_norm": {"scale": None},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3 MoE (e.g. qwen3-30b-a3b)
QWEN3_MOE_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "moe_block": {
                    "gate": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_0": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                    "wi_1": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                    "wo": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "key_norm": {"scale": None},
                    "query_norm": {"scale": None},
                },
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3 custom MoE (e.g. qwen3-custom-30b-a3b)
QWEN3_CUSTOM_MOE_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "latent_norm": {"scale": None},
                "layer_up_projection": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                "moe_block": {
                    "gate": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_0": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                    "wi_1": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                    "wo": mdn(reduction_axis=(-2,), output_axis=(-1,)),
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "key_norm": {"scale": None},
                    "query_norm": {"scale": None},
                },
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3-next (e.g. qwen3-next-80b-a3b)
_QWEN3_NEXT_MLP = {
    "routed_experts": {
        "gate": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_0": mdn(reduction_axis=(-2,), output_axis=(-1,)),
        "wi_1": mdn(reduction_axis=(-2,), output_axis=(-1,)),
        "wo": mdn(reduction_axis=(-2,), output_axis=(-1,)),
    },
    "shared_expert": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
    "shared_expert_gate": {"kernel": None},
}

_QWEN3_NEXT_GDN_ATTENTION = {
    "A_log": None,
    "conv1d": {"kernel": None},
    "dt_bias": None,
    "in_proj_ba": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "in_proj_qkvz": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "norm": {"rms_norm": {"scale": None}},
    "out_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
}

_QWEN3_NEXT_FULL_ATTENTION = {
    "attention": {
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "out": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "key_norm": {"scale": None},
        "query_norm": {"scale": None},
    },
}

_QWEN3_NEXT_GDN_LAYER = {
    "attention": _QWEN3_NEXT_GDN_ATTENTION,
    "input_layernorm": {"scale": None},
    "mlp": _QWEN3_NEXT_MLP,
    "post_attention_layernorm": {"scale": None},
}

_QWEN3_NEXT_FULL_ATTN_LAYER = {
    "attention": _QWEN3_NEXT_FULL_ATTENTION,
    "input_layernorm": {"scale": None},
    "mlp": _QWEN3_NEXT_MLP,
    "post_attention_layernorm": {"scale": None},
}

QWEN3_NEXT_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "layer_0": _QWEN3_NEXT_GDN_LAYER,
                "layer_1": _QWEN3_NEXT_GDN_LAYER,
                "layer_2": _QWEN3_NEXT_GDN_LAYER,
                "layer_3": _QWEN3_NEXT_FULL_ATTN_LAYER,
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# gpt-oss (e.g. gpt-oss-20b)
_GPT_OSS_ATTENTION = {
    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)), "bias": None},
    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)), "bias": None},
    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)), "bias": None},
    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,)), "bias": None},
    "sinks": None,
}

_GPT_OSS_MLP = {
    "gate": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,)), "bias": None},
    "wi_0": mdn(reduction_axis=(-2,), output_axis=(-1,)),
    "wi_0_bias": None,
    "wi_1": mdn(reduction_axis=(-2,), output_axis=(-1,)),
    "wi_1_bias": None,
    "wo": mdn(reduction_axis=(-2,), output_axis=(-1,)),
    "wo_bias": None,
}

_GPT_OSS_LAYER = {
    "GptOssAttention": _GPT_OSS_ATTENTION,
    "GptOssMlp": _GPT_OSS_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

GPT_OSS_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "layers_0": _GPT_OSS_LAYER,
                "layers_1": _GPT_OSS_LAYER,
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# deepseek4 building blocks
_DEEPSEEK4_MHC_ATTENTION = {
    "mhc_norm": {"scale": None},
    "post_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "post_alpha_scale": None,
    "post_beta": None,
    "pre_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "pre_alpha_scale": None,
    "pre_beta": None,
    "res_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "res_alpha_scale": None,
    "res_beta": None,
}

_DEEPSEEK4_MHC_MLP = {
    "mhc_norm": {"scale": None},
    "post_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "post_alpha_scale": None,
    "post_beta": None,
    "pre_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "pre_alpha_scale": None,
    "pre_beta": None,
    "res_alpha": mdn(reduction_axis=(0,), output_axis=(-1,)),
    "res_alpha_scale": None,
    "res_beta": None,
}

_DEEPSEEK4_MLP = {
    "MoeBlock_0": {
        "gate": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_0": mdn(reduction_axis=(-2,), output_axis=(-1,)),
        "wi_1": mdn(reduction_axis=(-2,), output_axis=(-1,)),
        "wo": mdn(reduction_axis=(-2,), output_axis=(-1,)),
    },
    "shared_experts": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
}

_DEEPSEEK4_MLP_SCANNED = _DEEPSEEK4_MLP

_DEEPSEEK4_ATTN_BASIC = {
    "kv_norm": {"scale": None},
    "o_a_proj": {"kernel": mdn(reduction_axis=(-2,), output_axis=(-1,))},
    "o_b_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "q_norm": {"scale": None},
    "sinks": None,
    "wkv": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
    "wq_a": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "wq_b": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
}

_DEEPSEEK4_ATTN_CSA = {
    "csa_compressor": {
        "gate_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "indexer": {
            "gate_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
            "kv_norm": {"scale": None},
            "kv_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
            "position_bias": mdn(reduction_axis=(0,), output_axis=(-1,)),
            "q_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
            "weights_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        },
        "kv_norm": {"scale": None},
        "kv_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "position_bias": mdn(reduction_axis=(0,), output_axis=(-1,)),
    },
    "kv_norm": {"scale": None},
    "o_a_proj": {"kernel": mdn(reduction_axis=(-2,), output_axis=(-1,))},
    "o_b_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "q_norm": {"scale": None},
    "sinks": None,
    "wkv": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
    "wq_a": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "wq_b": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
}

_DEEPSEEK4_ATTN_HCA = {
    "hca_compressor": {
        "gate_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "kv_norm": {"scale": None},
        "kv_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "position_bias": mdn(reduction_axis=(0,), output_axis=(-1,)),
    },
    "kv_norm": {"scale": None},
    "o_a_proj": {"kernel": mdn(reduction_axis=(-2,), output_axis=(-1,))},
    "o_b_proj": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "q_norm": {"scale": None},
    "sinks": None,
    "wkv": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
    "wq_a": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    "wq_b": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
}

_DEEPSEEK4_LAYER_BASIC = {
    "mhc_attention": _DEEPSEEK4_MHC_ATTENTION,
    "mhc_mlp": _DEEPSEEK4_MHC_MLP,
    "mlp": _DEEPSEEK4_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_BASIC,
}

_DEEPSEEK4_LAYER_CSA_PREFIX = {
    "mhc_attention": _DEEPSEEK4_MHC_ATTENTION,
    "mhc_mlp": _DEEPSEEK4_MHC_MLP,
    "mlp": _DEEPSEEK4_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_CSA,
}

_DEEPSEEK4_LAYER_CSA_SCANNED = {
    "mhc_attention": _DEEPSEEK4_MHC_ATTENTION,
    "mhc_mlp": _DEEPSEEK4_MHC_MLP,
    "mlp": _DEEPSEEK4_MLP_SCANNED,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_CSA,
}

_DEEPSEEK4_LAYER_HCA_SCANNED = {
    "mhc_attention": _DEEPSEEK4_MHC_ATTENTION,
    "mhc_mlp": _DEEPSEEK4_MHC_MLP,
    "mlp": _DEEPSEEK4_MLP_SCANNED,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_HCA,
}

DEEPSEEK4_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "hc_head": {
                "hc_base": None,
                "hc_fn": mdn(reduction_axis=(0,), output_axis=(-1,)),
                "hc_scale": None,
            },
            "layers_0": _DEEPSEEK4_LAYER_BASIC,
            "layers_1": _DEEPSEEK4_LAYER_BASIC,
            "layers_2": _DEEPSEEK4_LAYER_CSA_PREFIX,
            "logits_dense": {"kernel": None},
            "scanned_blocks": {
                "layers_0": _DEEPSEEK4_LAYER_HCA_SCANNED,
                "layers_1": _DEEPSEEK4_LAYER_CSA_SCANNED,
            },
        },
        "token_embedder": {"embedding": None},
    }
}


class MuonDimensionTest(parameterized.TestCase):
  """Unit tests for Muon dimension number generation.

  This suite verifies that the automatically generated Muon dimension numbers
  for various models match their hardcoded reference values.
  """

  @parameterized.named_parameters(
      ("deepseek2-16b", "deepseek2-16b", DEEPSEEK2_DIMENSION_NUMBER),
      ("deepseek3-671b", "deepseek3-671b", DEEPSEEK3_DIMENSION_NUMBER),
      ("deepseek4-284b", "deepseek4-284b", DEEPSEEK4_DIMENSION_NUMBER),
      ("kimi-k2-1t", "kimi-k2-1t", DEEPSEEK3_DIMENSION_NUMBER),
      ("llama2-7b", "llama2-7b", LLAMA2_DIMENSION_NUMBER),
      ("llama3-8b", "llama3-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.1-8b", "llama3.1-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.3-70b", "llama3.3-70b", LLAMA2_DIMENSION_NUMBER),
      ("gemma3-4b", "gemma3-4b", GEMMA3_DIMENSION_NUMBER),
      ("qwen3-0.6b", "qwen3-0.6b", QWEN3_DIMENSION_NUMBER),
      ("qwen3-30b-a3b", "qwen3-30b-a3b", QWEN3_MOE_DIMENSION_NUMBER),
      ("qwen3-custom-30b-a3b", "qwen3-custom-30b-a3b", QWEN3_CUSTOM_MOE_DIMENSION_NUMBER),
      ("qwen3-next-80b-a3b", "qwen3-next-80b-a3b", QWEN3_NEXT_DIMENSION_NUMBER),
      ("gpt-oss-20b", "gpt-oss-20b", GPT_OSS_DIMENSION_NUMBER),
  )
  @pytest.mark.tpu_only
  def test_model_integration(self, model_name, expected_output):
    """
    Initializes the specified MaxText model and asserts that the generated
    Muon dimension numbers match the hardcoded reference.
    """
    is_pure_nnx = model_name in {"deepseek4-284b"}
    actual_output = muon_utils.get_model_mdn(model_name, scan_layers=True, pure_nnx=is_pure_nnx)
    if "params" in expected_output and "params" in actual_output:
      self.assertEqual(actual_output["params"], expected_output["params"])
    else:
      self.assertEqual(actual_output, expected_output)


class AdamWMaskTest(parameterized.TestCase):
  """Tests for the AdamW mask functionality"""

  def test_get_adamw_mask_with_empty_mask(self):
    """Directly test the get_adamw_mask function with empty list"""
    # Case 1: No mask in config (empty list)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=[]"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertIsNone(mask_fn)

  def test_get_adamw_mask_with_valid_mask(self):
    """Directly test the get_adamw_mask function with valid mask"""
    # Case 2: Mask in config
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias', '.*norm', '.*ln.*']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertTrue(callable(mask_fn))

    params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": {"ln": {"scale": 4}}}
    mask = mask_fn(params)
    self.assertTrue(mask["layer1"]["kernel"])
    self.assertFalse(mask["layer1"]["bias"])
    self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
    self.assertFalse(mask["layer3"]["ln"]["scale"])

  def test_get_adamw_mask_with_invalid_mask(self):
    """Test that an invalid regex in the mask config raises an error when used"""
    # Create a config with an invalid regex (unbalanced bracket)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['[']"]
    config = pyconfig.initialize(argv)

    # Applying the mask should raise re.error due to the invalid regex
    with self.assertRaises(re.error):
      optimizers.get_adamw_mask(config)

  def test_get_adamw_mask_with_getattrkey(self):
    """Test that get_adamw_mask correctly handles GetAttrKey (e.g. from NamedTuples)"""

    class MyParams(NamedTuple):
      kernel: jax.Array
      bias: jax.Array

    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)

    params = MyParams(kernel=jax.numpy.ones((2, 2)), bias=jax.numpy.zeros((2,)))
    mask = mask_fn(params)

    self.assertTrue(mask.kernel)
    self.assertFalse(mask.bias)

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_with_mask(self, opt_type, mock_path):
    """Test that optimizer receives the mask function from config and it works as expected"""
    # Create a config with a mask list including regex
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "adamw_mask=['bias', 'layer_norm', 'layer1/.*kernel']",
        f"opt_type={opt_type}",
    ]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with a mask function
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      mask_fn = kwargs["mask"]

      # Verify that mask_fn is not None
      self.assertIsNotNone(mask_fn)

      # Test the behavior of mask_fn
      params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": [4, 5]}

      mask = mask_fn(params)

      # kernel in layer1 should be False because of 'layer1/.*kernel'
      self.assertFalse(mask["layer1"]["kernel"])
      # bias in layer1 should be False because of 'bias'
      self.assertFalse(mask["layer1"]["bias"])
      # layer_norm should be False because of 'layer_norm'
      self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
      # layer3 elements should be True
      self.assertTrue(mask["layer3"][0])
      self.assertTrue(mask["layer3"][1])

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_without_mask(self, opt_type, mock_path):
    """Test that optimizer receives None for mask when config is empty"""
    argv = ["", get_test_config_path(), "run_name=test", f"opt_type={opt_type}"]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with mask=None
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      self.assertIsNone(kwargs["mask"])


class AdamPaxScalarLearningRateTest(parameterized.TestCase):
  """Cover both branches of adam_pax's callable-vs-scalar learning_rate_fn guard.

  adam_pax accepts either a callable schedule (the usual case) or a pre-evaluated
  scalar (when wrapped by optax.inject_hyperparams). Both must produce identical
  parameter updates for the same effective learning rate.
  """

  def test_adam_pax_callable_and_scalar_lr_produce_same_update(self):
    lr_value = 0.1
    params = {"w": jnp.ones((2,))}
    grads = {"w": jnp.ones((2,))}

    opt_callable = optimizers.adam_pax(
        learning_rate_fn=lambda step: lr_value,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
    )
    opt_scalar = optimizers.adam_pax(
        learning_rate_fn=jnp.array(lr_value),
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
    )
    state_c = opt_callable.init(params)
    state_s = opt_scalar.init(params)
    updates_c, _ = opt_callable.update(grads, state_c, params)
    updates_s, _ = opt_scalar.update(grads, state_s, params)

    self.assertTrue(jnp.allclose(updates_c["w"], updates_s["w"], rtol=1e-6))
    # Sanity: with positive grads + negative step_size, updates should be negative.
    self.assertLess(float(updates_c["w"][0]), 0.0)


class TrainableParametersMaskTest(parameterized.TestCase):
  """Tests for the trainable parameters mask functionality via get_optimizer"""

  def test_get_optimizer_with_trainable_mask(self):
    """Test get_optimizer with a valid trainable_parameters_mask."""
    argv = [
        "",
        get_test_config_path(),
        "run_name=test_with_trainable_mask",
        "trainable_parameters_mask=['.*indexer.*', 'layer_norm']",
    ]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    # We can test the optimizer by creating some dummy params and gradients
    # and checking if the updates are zeroed out for non-trainable parameters.
    params = {
        "layer1": {"kernel": jax.numpy.ones((2, 2)), "indexer": jax.numpy.ones((2, 2))},
        "layer2": {"layer_norm": {"scale": jax.numpy.ones((2, 2))}},
        "layer3": {"ln": {"scale": jax.numpy.ones((2, 2))}},
    }

    # Give some non-zero gradients
    grads = jax.tree_util.tree_map(lambda x: jax.numpy.ones_like(x) * 0.5, params)

    # Initialize optimizer state
    opt_state = opt.init(params)

    # Compute updates
    updates, _ = opt.update(grads, opt_state, params)

    # 'layer1/kernel' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer1"]["kernel"] == 0))
    # 'layer3/ln/scale' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer3"]["ln"]["scale"] == 0))
    # 'layer1/indexer' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer1"]["indexer"] == 0))
    # 'layer2/layer_norm/scale' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer2"]["layer_norm"]["scale"] == 0))

  def test_get_optimizer_without_trainable_mask(self):
    """Test get_optimizer when trainable_parameters_mask is empty."""
    argv = ["", get_test_config_path(), "run_name=test", "trainable_parameters_mask=[]"]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    params = {"layer1": {"kernel": jax.numpy.ones((2, 2))}}
    grads = {"layer1": {"kernel": jax.numpy.ones((2, 2)) * 0.5}}

    opt_state = opt.init(params)
    updates, _ = opt.update(grads, opt_state, params)

    # When no trainable mask is provided, nothing is frozen by this mechanism
    self.assertFalse(jax.numpy.all(updates["layer1"]["kernel"] == 0))


_ADAPTER = "adapter/kernel"
_BACKBONE = "backbone/kernel"
_VIT_QUERY = "vision_encoder/layer_0/attention/query/kernel"
_VIT_WI_0 = "vision_encoder/layer_0/mlp/wi_0/kernel"
_CLIP_BOUND_NAMES = ("input_min", "input_max", "output_min", "output_max")
_CLIP_BOUNDS = frozenset(
    f"vision_encoder/layer_0/{block}/{name}"
    for block in ("attention/q_clip", "mlp/gate_clip")
    for name in _CLIP_BOUND_NAMES
)
_NON_CLIP = frozenset({_ADAPTER, _BACKBONE, _VIT_QUERY, _VIT_WI_0})


def _adapter_backbone_vit_params():
  """Small params tree: an adapter, a backbone and two Gemma-4 ViT projections carrying scalar clip bounds."""

  def clip():
    return {n: jnp.asarray(v, jnp.float32) for n, v in zip(_CLIP_BOUND_NAMES, (-2.0, 2.0, -3.0, 3.0))}

  return {
      "adapter": {"kernel": jnp.ones((2, 2))},
      "backbone": {"kernel": jnp.ones((2, 2))},
      "vision_encoder": {
          "layer_0": {
              "attention": {"query": {"kernel": jnp.ones((2, 2))}, "q_clip": clip()},
              "mlp": {"wi_0": {"kernel": jnp.ones((2, 2))}, "gate_clip": clip()},
          }
      },
  }


def _leaf_paths(tree):
  return {jax.tree_util.keystr(p, simple=True, separator="/"): v for p, v in jax.tree_util.tree_leaves_with_path(tree)}


def _run_jitted_steps(opt, params, num_steps=2):
  """Applies `num_steps` jitted optimizer steps with a constant non-zero gradient; returns (params, opt_state)."""
  grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 0.5), params)
  opt_state = opt.init(params)

  @jax.jit
  def step(p, s):
    updates, s = opt.update(grads, s, p)
    return optax.apply_updates(p, updates), s

  for _ in range(num_steps):
    params, opt_state = step(params, opt_state)
  return params, opt_state


def _adam_mu_slots(opt_state):
  """Returns (allocated, masked): param paths with an allocated AdamW `mu` slot vs. an optax.MaskedNode slot."""
  allocated, masked = set(), set()
  flat = jax.tree_util.tree_flatten_with_path(opt_state, is_leaf=lambda x: isinstance(x, optax.MaskedNode))[0]
  for path, leaf in flat:
    names = [getattr(k, "name", getattr(k, "key", None)) for k in path]
    if "mu" not in names:
      continue
    param_path = "/".join(str(n) for n in names[names.index("mu") + 1 :])
    (masked if isinstance(leaf, optax.MaskedNode) else allocated).add(param_path)
  return allocated, masked


class TrainableMaskAndClipBoundFreezeTest(parameterized.TestCase):
  """trainable_parameters_mask (a whitelist) composed with the Gemma-4 ViT clip-bound freeze in get_optimizer.

  Checks the parameters actually updated by jitted AdamW steps (weight decay > 0, so any leaf that is not frozen
  moves) and the optimizer-state allocation: frozen leaves are routed to optax.set_to_zero() through
  optax.multi_transform, so they get zero updates and no AdamW moment slots (optax.MaskedNode), while trainable
  leaves get allocated slots.
  """

  def _get_optimizer(self, trainable_mask, use_clipped_linears_for_vit):
    """Builds (config, get_optimizer(config)) for AdamW with weight decay and a constant learning rate of 1.0."""
    argv = [
        "",
        get_test_config_path(),
        "run_name=test_trainable_mask_clip_freeze",
        "opt_type=adamw",
        "adam_weight_decay=0.1",
        f"trainable_parameters_mask={trainable_mask}",
        f"use_clipped_linears_for_vit={use_clipped_linears_for_vit}",
    ]
    config = pyconfig.initialize(argv)
    return config, optimizers.get_optimizer(config, lambda step: 1.0)

  def _assert_trained_exactly(self, opt, expected_trainable):
    params = _adapter_backbone_vit_params()
    new_params, opt_state = _run_jitted_steps(opt, params)
    before, after = _leaf_paths(params), _leaf_paths(new_params)
    self.assertEqual(set(before), _NON_CLIP | _CLIP_BOUNDS)
    changed = {k for k in before if not jnp.array_equal(before[k], after[k])}
    self.assertEqual(changed, set(expected_trainable))
    allocated, masked = _adam_mu_slots(opt_state)
    self.assertEqual(allocated, set(expected_trainable))
    self.assertEqual(masked, set(before) - set(expected_trainable))

  def test_whitelist_trains_only_matching_params(self):
    """use_clipped_linears_for_vit=False: only whitelisted leaves train (the pre-existing whitelist contract)."""
    _, opt = self._get_optimizer(["adapter"], False)
    self._assert_trained_exactly(opt, {_ADAPTER})

  @parameterized.named_parameters(
      ("adapter_only", ["adapter"], {_ADAPTER}),
      ("whitelist_covers_vit", ["adapter", "vision_encoder"], {_ADAPTER, _VIT_QUERY, _VIT_WI_0}),
      ("whitelist_names_clip_bounds", ["adapter", "_clip"], {_ADAPTER}),
  )
  def test_whitelist_with_clip_freeze(self, trainable_mask, expected_trainable):
    """Whitelisted leaves train, the rest stay frozen, and clip bounds stay frozen even when whitelisted."""
    _, opt = self._get_optimizer(trainable_mask, True)
    self._assert_trained_exactly(opt, expected_trainable)

  def test_clip_freeze_without_whitelist(self):
    """No whitelist: everything trains except the clip bounds."""
    _, opt = self._get_optimizer([], True)
    self._assert_trained_exactly(opt, _NON_CLIP)

  def test_both_flags_off_returns_unwrapped_optimizer(self):
    """No whitelist and no clip freeze: get_optimizer returns the base optimizer unwrapped; every leaf trains."""
    _, opt = self._get_optimizer([], False)
    params = _adapter_backbone_vit_params()
    self.assertNotIsInstance(opt.init(params), optax.PartitionState)
    new_params, _ = _run_jitted_steps(opt, params)
    before, after = _leaf_paths(params), _leaf_paths(new_params)
    self.assertTrue(all(not jnp.array_equal(before[k], after[k]) for k in before))

  def test_whitelist_without_clip_freeze_matches_original_composition(self):
    """use_clipped_linears_for_vit=False: the whitelist-only optimizer equals its definition prior to the clip freeze."""
    patterns = ["adapter"]
    config, opt = self._get_optimizer(patterns, False)
    _, base_opt = self._get_optimizer([], False)
    self.assertFalse(config.use_clipped_linears_for_vit)
    freeze_mask_fn = optimizers._get_path_mask_fn(patterns, match_returns_true=False)  # pylint: disable=protected-access
    reference = optax.multi_transform(
        {"trainable": base_opt, "frozen": optax.set_to_zero()},
        lambda p: jax.tree_util.tree_map(lambda x: "frozen" if x else "trainable", freeze_mask_fn(p)),
    )
    params = _adapter_backbone_vit_params()
    got = _run_jitted_steps(opt, params)
    ref = _run_jitted_steps(reference, params)
    self.assertEqual(jax.tree_util.tree_structure(got), jax.tree_util.tree_structure(ref))
    for got_leaf, ref_leaf in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref)):
      self.assertTrue(jnp.array_equal(got_leaf, ref_leaf))


class SkipStepOnSpikesTest(parameterized.TestCase):
  """Tests for the skip_step_on_spikes optimizer wrapper."""

  def _run_spike_test(self, spike_kwargs):
    inner_opt = optax.sgd(0.1)
    opt = optimizers.skip_step_on_spikes(inner_opt, interval=4, scaling_factor=1.0)

    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    # Base kwargs for warmup
    base_kwargs = {k: jnp.array(1.0) for k in spike_kwargs.keys()}

    # Step 0: count = 0 < 2, will not skip (count should be >= interval / 2)
    updates, opt_state = opt.update({"x": jnp.array([1.0])}, opt_state, params, **base_kwargs)
    self.assertFalse(jnp.all(updates["x"] == 0.0))
    self.assertFalse(opt_state["is_skipped"])

    # Step 1: count = 1 < 2, will not skip. mean=1.0, std=0.0 (count should be >= interval / 2)
    updates, opt_state = opt.update({"x": jnp.array([1.0])}, opt_state, params, **base_kwargs)
    self.assertFalse(jnp.all(updates["x"] == 0.0))
    self.assertFalse(opt_state["is_skipped"])

    # Step 2: count = 2. Spike!
    spike_kwargs_jnp = {k: jnp.array(v) for k, v in spike_kwargs.items()}
    updates, opt_state = opt.update({"x": jnp.array([1.0])}, opt_state, params, **spike_kwargs_jnp)
    self.assertTrue(jnp.all(updates["x"] == 0.0))
    self.assertTrue(opt_state["is_skipped"])

  def test_skip_step_on_loss_spike(self):
    self._run_spike_test({"loss": 100.0})

  def test_skip_step_on_grad_norm_spike(self):
    self._run_spike_test({"loss": 1.0, "grad_norm": 100.0})

  def test_skip_step_on_both_spike(self):
    self._run_spike_test({"loss": 100.0, "grad_norm": 100.0})

  def test_no_skip_without_kwargs(self):
    inner_opt = optax.sgd(0.1)
    opt = optimizers.skip_step_on_spikes(inner_opt, interval=4, scaling_factor=1.0)

    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    # Missing kwargs should act normally
    updates, opt_state = opt.update({"x": jnp.array([1.0])}, opt_state, params)
    self.assertFalse(jnp.all(updates["x"] == 0.0))
    self.assertFalse(opt_state["is_skipped"])
    # Count shouldn't have incremented
    self.assertEqual(opt_state["count"], 0)


class TestMuonLogic(unittest.TestCase):
  """Tests the granular path transformation functions."""

  def test_is_path_contain_any(self):
    # pylint: disable=protected-access
    self.assertTrue(muon_utils._is_path_contain_any(("a", "b"), ("x", "a", "z")))
    self.assertFalse(muon_utils._is_path_contain_any(("a", "b"), ("x", "y", "z")))

  def test_transform_logic_exclusions(self):
    self.assertIsNone(muon_utils.transform_logic(("layer_0", "bias")))
    self.assertIsNone(muon_utils.transform_logic(("layer_0", "scale")))
    self.assertIsNone(muon_utils.transform_logic(("embedding", "kernel")))
    self.assertIsNone(muon_utils.transform_logic(("layer_0", "attention", "A_log")))

  def test_transform_logic_moe(self):
    path = ("layers_0", "MoeBlock_0", "wi_0")
    result = muon_utils.transform_logic(path)
    self.assertEqual(result.reduction_axis, (-2,))
    self.assertEqual(result.output_axis, (-1,))

  def test_transform_logic_attention(self):
    path_out = ("layers_0", "self_attention", "out", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_out), mdn((0, -2), (-1,)))

    path_q = ("layers_0", "self_attention", "query", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_q), mdn((0,), (-2, -1)))

    path_gpt_out = ("layers_0", "GptOssAttention", "out", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_gpt_out), mdn((0, -2), (-1,)))

    path_gpt_q = ("layers_0", "GptOssAttention", "query", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_gpt_q), mdn((0,), (-2, -1)))

    path_qwen3_next_q = ("layers_0", "attention", "attention", "query", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_qwen3_next_q), mdn((0,), (-2, -1)))

    path_qwen3_next_out = ("layers_0", "attention", "attention", "out", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_qwen3_next_out), mdn((0,), (-1,)))

  def test_get_transform_tree(self):
    fake_tree = {"params": {"layer_0": {"kernel": "leaf", "bias": "leaf"}, "MoeBlock_0": {"wi_0": "leaf"}}}
    result = muon_utils.get_transform_tree(fake_tree)
    self.assertEqual(result["params"]["layer_0"]["kernel"], mdn((0,), (-1,)))
    self.assertIsNone(result["params"]["layer_0"]["bias"])

  def test_get_muon_weight_dimension_numbers_nnx(self):
    """Verifies dimension extraction for stateful NNX modules."""

    class MockNNXModel(nnx.Module):
      """Mock NNX Module."""

      def __init__(self, rngs: nnx.Rngs):
        # 1. Standard layer
        self.layer1 = nnx.Linear(2, 4, rngs=rngs)

        # 2. MoE specific naming to trigger transform logic.
        # The logic expects "MoeBlock_0" AND "wi_0"/"wi_1"/"wo" in the path.
        # We nest the linear layer to create the path: ('MoeBlock_0', 'wi_0', 'kernel')
        self.MoeBlock_0 = nnx.Module()
        self.MoeBlock_0.wi_0 = nnx.Linear(4, 2, rngs=rngs)

        # 3. Exclusion case (scaler/scale)
        self.scale = nnx.Param(jnp.ones((1,)))

    # Use eval_shape to create an abstract version of the model.
    model = nnx.eval_shape(lambda: MockNNXModel(rngs=nnx.Rngs(0)))
    config = MagicMock()

    # Extract dimension numbers using the NNX path in muon_utils
    result = muon_utils.get_muon_weight_dimension_numbers(model, config)

    # Verify standard weight path: ('layer1', 'kernel') -> default (0,)
    self.assertEqual(result.layer1.kernel, mdn((0,), (-1,)))

    # Verify MoE weight path: ('MoeBlock_0', 'wi_0', 'kernel') -> (-2,)
    self.assertEqual(result.MoeBlock_0.wi_0.kernel, mdn((-2,), (-1,)))

    # Verify exclusion (scalar/scale)
    self.assertIsNone(result.scale)

  def test_verbose_output_nnx(self):
    """Covers lines 128 and 135-154: _print_structure_debug via verbose=True with NNX model."""

    class SimpleNNXModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)

    model = nnx.eval_shape(lambda: SimpleNNXModel(rngs=nnx.Rngs(0)))
    config = MagicMock()
    muon_utils.get_muon_weight_dimension_numbers(model, config, verbose=True)

  def test_nnx_deepseek_attention_logic(self):
    """Simulates a DeepSeek-like attention structure in NNX."""

    class DeepSeekAttention(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.self_attention = nnx.Module()
        self.self_attention.query = nnx.Linear(8, 8, rngs=rngs)
        self.self_attention.out = nnx.Linear(8, 8, rngs=rngs)

    # Use eval_shape to create an abstract version of the model.
    model = nnx.eval_shape(lambda: DeepSeekAttention(nnx.Rngs(0)))
    config = MagicMock()
    result = muon_utils.get_muon_weight_dimension_numbers(model, config)

    # Check attention query: [0] -> [-2, -1]
    self.assertEqual(result.self_attention.query.kernel, mdn((0,), (-2, -1)))
    # Check attention out: [0, -2] -> [-1]
    self.assertEqual(result.self_attention.out.kernel, mdn((0, -2), (-1,)))

  def test_muon_newton_schulz_config(self):
    """Verifies that muon optimizer configures Newton-Schulz parameters correctly based on model."""
    model = MagicMock()
    learning_rate_schedule = MagicMock()

    # Case 1: DeepSeek4 Model (Auto-configures 10-step schedule)
    argv_ds4 = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "model_name=deepseek4-284b",
        "attention=dot_product",
    ]
    config_ds4 = pyconfig.initialize(argv_ds4)

    with (
        patch("maxtext.optimizers.optimizers.get_muon_weight_dimension_numbers") as mock_get_mdn,
        patch("maxtext.optimizers.optimizers.muon") as mock_muon,
    ):
      mock_get_mdn.return_value = {}
      optimizers.get_optimizer(config_ds4, learning_rate_schedule, model=model)
      mock_muon.assert_called_once()
      _, kwargs = mock_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 10)
      self.assertEqual(len(kwargs["ns_coeffs"]), 10)
      self.assertEqual(kwargs["ns_coeffs"][-1], (2.0, -1.5, 0.5))

    # Case 2: Standard Model (Llama2) (Defaults to 5-step schedule)
    argv_llama = ["", get_test_config_path(), "run_name=test", "opt_type=muon", "model_name=llama2-7b"]
    config_llama = pyconfig.initialize(argv_llama)

    with (
        patch("maxtext.optimizers.optimizers.get_muon_weight_dimension_numbers") as mock_get_mdn,
        patch("maxtext.optimizers.optimizers.muon") as mock_muon,
    ):
      mock_get_mdn.return_value = {}
      optimizers.get_optimizer(config_llama, learning_rate_schedule, model=model)
      mock_muon.assert_called_once()
      _, kwargs = mock_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 5)
      self.assertEqual(kwargs["ns_coeffs"], (3.4445, -4.7750, 2.0315))


if __name__ == "__main__":
  unittest.main()
