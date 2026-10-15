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
from typing import NamedTuple
import unittest
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.optimizers.muon import ShardedMuonDimensionNumbers as smdn
from maxtext.utils import maxtext_utils, muon_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax
from optax.contrib._muon import MuonDimensionNumbers as mdn
import pytest

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
                    "gate": {"kernel": mdn((0,), (-1,))},
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
                    "gate": {"kernel": mdn((0,), (-1,))},
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
        "gate": {"kernel": mdn((0,), (-1,))},
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

# A scanned Qwen3-Next block covers one `inhomogeneous_layer_cycle_interval` period: the
# linear-attention layers are homogeneous and stacked into `local_layers`, and the single
# trailing full-attention layer is `global_layer`. The dimension numbers themselves are
# unchanged by that stacking -- `transform_logic` keys off the parameter path, and the scan
# axes go in at `param_scan_axis`, which shifts neither axis 0 nor the negative axes.
QWEN3_NEXT_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "local_layers": _QWEN3_NEXT_GDN_LAYER,
                "global_layer": _QWEN3_NEXT_FULL_ATTN_LAYER,
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# gpt-oss (e.g. gpt-oss-20b)
_GPT_OSS_ATTENTION = {
    "query": {
        "kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)),
        "bias": None,
    },
    "key": {
        "kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)),
        "bias": None,
    },
    "value": {
        "kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1)),
        "bias": None,
    },
    "out": {
        "kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,)),
        "bias": None,
    },
    "sinks": None,
}

_GPT_OSS_MLP = {
    "gate": {"kernel": mdn((0,), (-1,)), "bias": None},
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
_DEEPSEEK4_MHC = {
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
        "gate": {"kernel": mdn((0,), (-1,))},
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
    "mhc_attention": _DEEPSEEK4_MHC,
    "mhc_mlp": _DEEPSEEK4_MHC,
    "mlp": _DEEPSEEK4_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_BASIC,
}

_DEEPSEEK4_LAYER_CSA_PREFIX = {
    "mhc_attention": _DEEPSEEK4_MHC,
    "mhc_mlp": _DEEPSEEK4_MHC,
    "mlp": _DEEPSEEK4_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_CSA,
}

_DEEPSEEK4_LAYER_CSA_SCANNED = {
    "mhc_attention": _DEEPSEEK4_MHC,
    "mhc_mlp": _DEEPSEEK4_MHC,
    "mlp": _DEEPSEEK4_MLP_SCANNED,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_CSA,
}

_DEEPSEEK4_LAYER_HCA_SCANNED = {
    "mhc_attention": _DEEPSEEK4_MHC,
    "mhc_mlp": _DEEPSEEK4_MHC,
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

_DEEPSEEK4_LAYER_HCA_PREFIX = {
    "mhc_attention": _DEEPSEEK4_MHC,
    "mhc_mlp": _DEEPSEEK4_MHC,
    "mlp": _DEEPSEEK4_MLP,
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
    "self_attention": _DEEPSEEK4_ATTN_HCA,
}

DEEPSEEK4_1600B_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "hc_head": {
                "hc_base": None,
                "hc_fn": mdn(reduction_axis=(0,), output_axis=(-1,)),
                "hc_scale": None,
            },
            "layers_0": _DEEPSEEK4_LAYER_HCA_PREFIX,
            "layers_1": _DEEPSEEK4_LAYER_HCA_PREFIX,
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
      ("deepseek4-1600b", "deepseek4-1600b", DEEPSEEK4_1600B_DIMENSION_NUMBER),
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
    actual_output = muon_utils.get_model_mdn(model_name, scan_layers=True)
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
      ("adamw", "adamw", optimizers.optax, "adamw"),
      ("adam_pax", "adam_pax", optimizers, "adam_pax"),
      ("adam_pax_low_memory", "adam_pax_low_memory", optimizers, "adam_pax_low_memory"),
  )
  def test_optimizer_with_mask(self, opt_type, mock_target, mock_attr):
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

    with patch.object(mock_target, mock_attr) as mock_opt:
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
      ("adamw", "adamw", optimizers.optax, "adamw"),
      ("adam_pax", "adam_pax", optimizers, "adam_pax"),
      ("adam_pax_low_memory", "adam_pax_low_memory", optimizers, "adam_pax_low_memory"),
  )
  def test_optimizer_without_mask(self, opt_type, mock_target, mock_attr):
    """Test that optimizer receives None for mask when config is empty"""
    argv = ["", get_test_config_path(), "run_name=test", f"opt_type={opt_type}"]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch.object(mock_target, mock_attr) as mock_opt:
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


class AdamPaxLowMemoryTest(parameterized.TestCase):
  """Tests for opt_type='adam_pax_low_memory' and its bf16 update rounding modes."""

  def _make_opt(self, update_rounding, lr=1e-3):
    return optimizers.adam_pax_low_memory(
        learning_rate_fn=lambda step: lr,
        beta1=0.0,  # Use the RMS-normalized gradient directly.
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
        update_rounding=update_rounding,
        stochastic_rounding_seed=0,
    )

  def _run(self, opt, params, grads, steps):
    state = opt.init(params)
    for _ in range(steps):
      updates, state = opt.update(grads, state, params)
      params = optax.apply_updates(params, updates)
    return params, state

  def test_state_size(self):
    params = {"w": jnp.ones((4, 8), jnp.bfloat16)}
    _, nearest_state = self._run(self._make_opt("nearest"), params, params, steps=1)
    _, kahan_state = self._run(self._make_opt("kahan"), params, params, steps=1)
    # No per-parameter second moment; Kahan keeps a residual of the params' shape.
    self.assertEqual(nearest_state.nu.shape, ())
    self.assertEqual(kahan_state.nu["w"].shape, (4, 8))
    self.assertEqual(kahan_state.nu["w"].dtype, jnp.bfloat16)

  def test_kahan_keeps_sub_ulp_updates(self):
    # lr * normalized_grad (1e-3) is far below half a bf16 ulp at 1.0 (~3.9e-3),
    # so round-to-nearest drops every update while Kahan accumulates them.
    params = {"w": jnp.ones((2, 4), jnp.bfloat16)}
    grads = {"w": -jnp.ones((2, 4), jnp.bfloat16)}
    steps = 100
    nearest, _ = self._run(self._make_opt("nearest"), params, grads, steps)
    kahan, _ = self._run(self._make_opt("kahan"), params, grads, steps)

    self.assertTrue(jnp.all(nearest["w"] == 1.0))
    expected = 1.0 + steps * 1e-3
    self.assertTrue(jnp.allclose(kahan["w"].astype(jnp.float32), expected, atol=8e-3))

  def test_stochastic_rounding_is_unbiased(self):
    params = {"w": jnp.ones((64, 64), jnp.bfloat16)}
    grads = {"w": -jnp.ones((64, 64), jnp.bfloat16)}
    steps = 20
    sr, _ = self._run(self._make_opt("stochastic"), params, grads, steps)
    mean = float(jnp.mean(sr["w"].astype(jnp.float32)))
    self.assertAlmostEqual(mean, 1.0 + steps * 1e-3, delta=2e-3)

  def test_zero_weights_with_zero_grads_stay_zero(self):
    # N:M-masked entries (zero weight, zero gradient) must remain exactly zero.
    params = {"w": jnp.array([[0.0, 1.0, 0.0, 1.0]], jnp.bfloat16)}
    grads = {"w": jnp.array([[0.0, -1.0, 0.0, -1.0]], jnp.bfloat16)}
    for rounding in ("nearest", "kahan", "stochastic"):
      out, _ = self._run(self._make_opt(rounding), params, grads, steps=5)
      self.assertEqual(float(out["w"][0, 0]), 0.0, rounding)
      self.assertEqual(float(out["w"][0, 2]), 0.0, rounding)

  def test_invalid_rounding_raises(self):
    with self.assertRaises(ValueError):
      self._make_opt("bogus")

  @parameterized.parameters("nearest", "kahan", "stochastic")
  def test_scalar_and_vector_params(self, rounding):
    # 0-d and 1-d params (e.g. scalars, norm scales) must not crash the RMS normalization.
    params = {"s": jnp.array(1.0, jnp.bfloat16), "v": jnp.ones((4,), jnp.bfloat16)}
    grads = {"s": jnp.array(-0.5, jnp.bfloat16), "v": -jnp.ones((4,), jnp.bfloat16)}
    out, _ = self._run(self._make_opt(rounding, lr=0.1), params, grads, steps=2)
    self.assertEqual(out["s"].shape, ())
    self.assertGreater(float(out["s"]), 1.0)
    self.assertTrue(jnp.all(out["v"] > 1.0))

  def test_kahan_residual_has_param_shape_and_dtype(self):
    params = {"w": jnp.ones((4, 8), jnp.bfloat16), "b": jnp.ones((8,), jnp.float32)}
    state = self._make_opt("kahan").init(params)
    for name, p in params.items():
      self.assertEqual(state.nu[name].shape, p.shape)
      self.assertEqual(state.nu[name].dtype, p.dtype)

  def test_rounding_requires_low_memory_opt_type(self):
    argv = ["", get_test_config_path(), "run_name=test", "opt_type=adam_pax", "bf16_update_rounding=kahan"]
    with self.assertRaisesRegex(ValueError, "requires opt_type='adam_pax_low_memory'"):
      pyconfig.initialize(argv)

  def test_invalid_rounding_rejected_by_config(self):
    argv = ["", get_test_config_path(), "run_name=test", "opt_type=adam_pax_low_memory", "bf16_update_rounding=bogus"]
    with self.assertRaises(Exception):
      pyconfig.initialize(argv)

  def test_get_optimizer(self):
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=adam_pax_low_memory",
        "bf16_update_rounding=kahan",
    ]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    with patch.object(optimizers, "adam_pax_low_memory") as mock_opt:
      optimizers.get_optimizer(config, learning_rate_schedule)
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      self.assertEqual(kwargs["update_rounding"], "kahan")

    opt = optimizers.get_optimizer(config, learning_rate_schedule)
    params = {"w": jnp.ones((2, 4), jnp.bfloat16)}
    state = opt.init(params)
    updates, new_state = opt.update(params, state, params)
    self.assertEqual(updates["w"].shape, (2, 4))
    self.assertEqual(int(new_state.count), 1)

  @parameterized.parameters("nearest", "kahan", "stochastic")
  def test_non_bf16_params_across_rounding_modes(self, rounding):
    params = {
        "w_bf16": jnp.ones((2, 4), jnp.bfloat16),
        "w_f32": jnp.ones((4,), jnp.float32),
    }
    grads = {
        "w_bf16": -jnp.ones((2, 4), jnp.bfloat16),
        "w_f32": -jnp.ones((4,), jnp.float32),
    }
    out, state = self._run(self._make_opt(rounding, lr=0.1), params, grads, steps=2)
    self.assertTrue(jnp.allclose(out["w_f32"], jnp.full((4,), 1.2, jnp.float32), atol=1e-5))
    self.assertEqual(out["w_f32"].dtype, jnp.float32)
    if rounding == "kahan":
      self.assertTrue(jnp.all(state.nu["w_f32"] == 0.0))

  def test_momentum_and_scalar_learning_rate(self):
    lr_value = 0.1
    params = {"w": jnp.array([[2.0, 2.0]], jnp.bfloat16)}
    grads_step0 = {"w": jnp.array([[1.0, 1.0]], jnp.bfloat16)}
    grads_step1 = {"w": jnp.array([[1.0, -1.0]], jnp.bfloat16)}

    opt_callable = optimizers.adam_pax_low_memory(
        learning_rate_fn=lambda step: lr_value,
        beta1=0.9,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
    )
    opt_scalar = optimizers.adam_pax_low_memory(
        learning_rate_fn=jnp.array(lr_value),
        beta1=0.9,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
    )
    state_c = opt_callable.init(params)
    state_s = opt_scalar.init(params)
    for grads in (grads_step0, grads_step1):
      updates_c, state_c = opt_callable.update(grads, state_c, params)
      updates_s, state_s = opt_scalar.update(grads, state_s, params)
      self.assertTrue(jnp.allclose(updates_c["w"], updates_s["w"], rtol=1e-6))
      self.assertTrue(jnp.allclose(state_c.mu["w"], state_s.mu["w"], rtol=1e-6))
    self.assertEqual(int(state_c.count), 2)
    # On step 1, the second column gradient flipped sign, but positive momentum from step 0 keeps mu[0, 1] > -1.
    self.assertGreater(float(state_c.mu["w"][0, 1]), -0.5)

  def test_weight_decay_and_mask(self):
    params = {
        "w": jnp.full((2, 2), 2.0, jnp.float32),
        "b": jnp.full((2,), 2.0, jnp.float32),
    }
    grads = {
        "w": jnp.ones((2, 2), jnp.float32),
        "b": jnp.ones((2,), jnp.float32),
    }

    def _step(weight_decay, mask=None):
      opt = optimizers.adam_pax_low_memory(
          learning_rate_fn=0.1,
          beta1=0.0,
          epsilon=1e-8,
          epsilon_root=0.0,
          weight_decay=weight_decay,
          mask=mask,
      )
      state = opt.init(params)
      updates, _ = opt.update(grads, state, params)
      return updates

    no_wd = _step(0.0)
    full_wd = _step(0.5, mask=None)
    callable_mask_wd = _step(0.5, mask=lambda p: {"w": True, "b": False})
    tree_mask_wd = _step(0.5, mask={"w": True, "b": False})

    # Weight decay adds -lr * wd * param = -0.1 * 0.5 * 2.0 = -0.1 to unmasked leaves.
    self.assertTrue(jnp.allclose(full_wd["w"], no_wd["w"] - 0.1, atol=1e-5))
    self.assertTrue(jnp.allclose(full_wd["b"], no_wd["b"] - 0.1, atol=1e-5))
    self.assertTrue(jnp.allclose(callable_mask_wd["w"], full_wd["w"], atol=1e-5))
    self.assertTrue(jnp.allclose(callable_mask_wd["b"], no_wd["b"], atol=1e-5))
    self.assertTrue(jnp.allclose(tree_mask_wd["w"], callable_mask_wd["w"], atol=1e-5))
    self.assertTrue(jnp.allclose(tree_mask_wd["b"], callable_mask_wd["b"], atol=1e-5))

  def test_params_required_only_for_weight_decay_or_non_nearest_rounding(self):
    params = {"w": jnp.ones((2,), jnp.bfloat16)}
    opt_nearest = self._make_opt("nearest")
    state = opt_nearest.init(params)
    updates, _ = opt_nearest.update(params, state)
    self.assertEqual(updates["w"].shape, (2,))

    opt_kahan = self._make_opt("kahan")
    with self.assertRaises(AssertionError):
      opt_kahan.update(params, opt_kahan.init(params), params=None)

    opt_stochastic = self._make_opt("stochastic")
    with self.assertRaises(AssertionError):
      opt_stochastic.update(params, opt_stochastic.init(params), params=None)

    opt_wd = optimizers.adam_pax_low_memory(
        learning_rate_fn=0.1,
        beta1=0.0,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.1,
    )
    with self.assertRaises(AssertionError):
      opt_wd.update(params, opt_wd.init(params), params=None)

  def test_stochastic_round_update_non_bf16_passthrough(self):
    # pylint: disable=protected-access
    # Covers optimizers._stochastic_round_update early return (line 296):
    # When p.dtype != jnp.bfloat16, _stochastic_round_update bypasses the bf16
    # bitcast/noise path and returns (step_size * u).astype(u.dtype) directly.
    step_size = jnp.array(-0.05, dtype=jnp.float32)
    for dtype in (jnp.float32, jnp.float16):
      p = jnp.array([1.0, -2.5, 0.0, 4.25], dtype=dtype)
      u = jnp.array([0.25, -0.5, 0.0, 1.5], dtype=dtype)
      d_key0 = optimizers._stochastic_round_update(u, p, step_size, jax.random.PRNGKey(0))
      d_key1 = optimizers._stochastic_round_update(u, p, step_size, jax.random.PRNGKey(99))
      expected = (step_size * u).astype(dtype)
      self.assertEqual(d_key0.dtype, dtype)
      self.assertTrue(jnp.array_equal(d_key0, expected))
      # Non-bf16 updates must be completely deterministic and independent of the PRNG key.
      self.assertTrue(jnp.array_equal(d_key0, d_key1))

  def test_stochastic_round_update_bf16_exact_identity_and_sub_ulp_brackets(self):
    # pylint: disable=protected-access
    # Covers optimizers._stochastic_round_update bf16 bit-manipulation body (lines 297-303):
    # At p = 1.0 in bfloat16, 1 ULP is 2^-7 = 0.0078125. With step_size = -1.0 and
    # u = -2^-8 = -0.00390625 (exactly +0.5 ULP update), the exact fp32 target
    # new32 = 1.0 + 2^-8 sits halfway between adjacent bf16 values 1.0 and 1.0078125.
    # Adding 16-bit uniform noise to the fp32 uint32 bits and masking with 0xFFFF0000
    # rounds up to 1.0078125 (d = 2^-7) or down to 1.0 (d = 0.0) with probability 0.5.
    ulp = 2.0**-7
    p = jnp.ones((2048,), dtype=jnp.bfloat16)
    u = jnp.full((2048,), -0.5 * ulp, dtype=jnp.bfloat16)
    step_size = jnp.array(-1.0, dtype=jnp.float32)

    d = optimizers._stochastic_round_update(u, p, step_size, jax.random.PRNGKey(42))
    self.assertEqual(d.dtype, jnp.bfloat16)

    # Every update d must land in {0.0, ulp} so that p + d is in {1.0, 1.0 + ulp}.
    d_f32 = d.astype(jnp.float32)
    self.assertTrue(jnp.all((d_f32 == 0.0) | (d_f32 == ulp)))
    new_p_bf16 = p + d
    self.assertTrue(jnp.array_equal(new_p_bf16.astype(jnp.float32), p.astype(jnp.float32) + d_f32))

    # Both rounding brackets must be exercised with ~50% probability.
    frac_rounded_up = float(jnp.mean((d_f32 == ulp).astype(jnp.float32)))
    self.assertAlmostEqual(frac_rounded_up, 0.5, delta=0.05)

    # Zero bf16 weights with zero updates (0x00000000 in fp32) stay identically zero
    # after adding 16-bit noise (& 0xFFFF) and masking (& 0xFFFF0000).
    p_zero = jnp.zeros((64,), dtype=jnp.bfloat16)
    u_zero = jnp.zeros((64,), dtype=jnp.bfloat16)
    d_zero = optimizers._stochastic_round_update(u_zero, p_zero, step_size, jax.random.PRNGKey(7))
    self.assertTrue(jnp.all(d_zero == 0.0))

  def test_stochastic_rounding_pytree_leaf_and_step_key_folding(self):
    # Covers adam_pax_low_memory update_fn's update_rounding == 'stochastic' branch
    # (lines 499-509): verifies that base_key folds in both step count and per-leaf
    # enumeration index i across a multi-leaf PyTree with mixed bf16 and fp32 leaves.
    params = {
        "leaf_a": jnp.ones((128,), dtype=jnp.bfloat16),
        "leaf_b": jnp.ones((128,), dtype=jnp.bfloat16),
        "leaf_f32": jnp.ones((16,), dtype=jnp.float32),
    }
    # Identical gradients on leaf_a and leaf_b produce identical RMS-normalized mu,
    # so any difference between their updates comes solely from fold_in(base_key, i).
    grads = {
        "leaf_a": -jnp.ones((128,), dtype=jnp.bfloat16),
        "leaf_b": -jnp.ones((128,), dtype=jnp.bfloat16),
        "leaf_f32": -jnp.ones((16,), dtype=jnp.float32),
    }

    opt_seed0 = optimizers.adam_pax_low_memory(
        learning_rate_fn=1e-3,
        beta1=0.0,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
        update_rounding="stochastic",
        stochastic_rounding_seed=0,
    )
    opt_seed0_dup = optimizers.adam_pax_low_memory(
        learning_rate_fn=1e-3,
        beta1=0.0,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
        update_rounding="stochastic",
        stochastic_rounding_seed=0,
    )
    opt_seed1 = optimizers.adam_pax_low_memory(
        learning_rate_fn=1e-3,
        beta1=0.0,
        epsilon=1e-8,
        epsilon_root=0.0,
        weight_decay=0.0,
        update_rounding="stochastic",
        stochastic_rounding_seed=1,
    )

    state0 = opt_seed0.init(params)
    updates_step1, state1 = opt_seed0.update(grads, state0, params)
    updates_step2, _ = opt_seed0.update(grads, state1, params)

    # 1. Distinct leaf indices (i=0 vs i=1) receive distinct folded keys on the same step.
    self.assertFalse(jnp.array_equal(updates_step1["leaf_a"], updates_step1["leaf_b"]))
    # 2. Distinct step counts (count=1 vs count=2) fold distinct base_keys for the same leaf.
    self.assertFalse(jnp.array_equal(updates_step1["leaf_a"], updates_step2["leaf_a"]))
    # 3. The float32 leaf in the same PyTree takes the non-bf16 passthrough branch on every step.
    self.assertTrue(jnp.allclose(updates_step1["leaf_f32"], jnp.full((16,), 1e-3, dtype=jnp.float32), atol=1e-6))
    self.assertTrue(jnp.array_equal(updates_step1["leaf_f32"], updates_step2["leaf_f32"]))

    # 4. Same stochastic_rounding_seed reproduces identical PyTree updates; a different seed diverges.
    updates_dup, _ = opt_seed0_dup.update(grads, opt_seed0_dup.init(params), params)
    updates_seed1, _ = opt_seed1.update(grads, opt_seed1.init(params), params)
    for k in params:
      self.assertTrue(jnp.array_equal(updates_step1[k], updates_dup[k]))
    self.assertFalse(jnp.array_equal(updates_step1["leaf_a"], updates_seed1["leaf_a"]))

    # 5. End-to-end config wiring for bf16_update_rounding=stochastic via get_optimizer.
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=adam_pax_low_memory",
        "bf16_update_rounding=stochastic",
    ]
    config = pyconfig.initialize(argv)
    lr_schedule = maxtext_utils.create_learning_rate_schedule(config)
    opt_from_config = optimizers.get_optimizer(config, lr_schedule)
    cfg_state = opt_from_config.init(params)
    cfg_updates, cfg_state = opt_from_config.update(grads, cfg_state, params)
    self.assertEqual(cfg_updates["leaf_a"].dtype, jnp.bfloat16)
    self.assertEqual(int(cfg_state.count), 1)


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

  def test_transform_logic_moe_routers(self):
    path = ("layers_0", "MoeBlock_0", "gate", "kernel")
    self.assertEqual(muon_utils.transform_logic(path, include_routers=True), mdn((0,), (-1,)))
    self.assertIsNone(muon_utils.transform_logic(path, include_routers=False))

    path_expert = ("layers_0", "MoeBlock_0", "wi_0")
    self.assertEqual(muon_utils.transform_logic(path_expert, include_routers=False), mdn((-2,), (-1,)))

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
        self.self_attention.query = nnx.Param(jnp.ones((8, 4, 2)))
        self.self_attention.out = nnx.Param(jnp.ones((4, 2, 8)))

    # Use eval_shape to create an abstract version of the model.
    model = nnx.eval_shape(lambda: DeepSeekAttention(nnx.Rngs(0)))
    config = MagicMock()
    result = muon_utils.get_muon_weight_dimension_numbers(model, config)

    # Check attention query: [0] -> [-2, -1]
    self.assertEqual(result.self_attention.query, mdn((0,), (-2, -1)))
    # Check attention out: [0, -2] -> [-1]
    self.assertEqual(result.self_attention.out, mdn((0, -2), (-1,)))

  def test_muon_newton_schulz_config(self):
    """Verifies that muon optimizer configures Newton-Schulz parameters correctly based on model and muon_type."""
    model = MagicMock()
    mesh = MagicMock()
    learning_rate_schedule = MagicMock()

    # Case 1: DeepSeek4 Model with MaxText Muon (Auto-configures 10-step schedule)
    argv_ds4 = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=maxtext_muon",
        "model_name=deepseek4-284b",
        "attention=dot_product",
        "muon_use_all_to_all=True",
    ]
    config_ds4 = pyconfig.initialize(argv_ds4)

    with (
        patch.object(optimizers, "get_maxtext_muon_weight_dimension_numbers") as mock_get_smdn,
        patch.object(optimizers, "maxtext_muon") as mock_maxtext_muon,
    ):
      mock_get_smdn.return_value = {"params": {"w": smdn((0,), (-1,))}}
      optimizers.get_optimizer(config_ds4, learning_rate_schedule, model=model, mesh=mesh)
      mock_get_smdn.assert_called_once_with(model, config_ds4, mesh=mesh)
      mock_maxtext_muon.assert_called_once()
      _, kwargs = mock_maxtext_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 10)
      self.assertEqual(len(kwargs["ns_coeffs"]), 10)
      self.assertEqual(kwargs["ns_coeffs"][-1], (2.0, -1.5, 0.5))
      self.assertEqual(kwargs["muon_weight_dimension_numbers"], mock_get_smdn.return_value)
      self.assertTrue(kwargs["use_all_to_all"])

    # Case 2: MaxText Muon with use_all_to_all=False
    argv_no_a2a = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=maxtext_muon",
        "model_name=llama2-7b",
        "muon_use_all_to_all=False",
    ]
    config_no_a2a = pyconfig.initialize(argv_no_a2a)

    with (
        patch.object(optimizers, "get_maxtext_muon_weight_dimension_numbers") as mock_get_smdn,
        patch.object(optimizers, "maxtext_muon") as mock_maxtext_muon,
    ):
      mock_get_smdn.return_value = {}
      optimizers.get_optimizer(config_no_a2a, learning_rate_schedule, model=model)
      mock_get_smdn.assert_called_once_with(model, config_no_a2a, mesh=None)
      mock_maxtext_muon.assert_called_once()
      _, kwargs = mock_maxtext_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 5)
      self.assertEqual(kwargs["ns_coeffs"], (3.4445, -4.7750, 2.0315))
      self.assertEqual(kwargs["muon_weight_dimension_numbers"], mock_get_smdn.return_value)
      self.assertFalse(kwargs["use_all_to_all"])

    # Case 3: Default Muon Type (defaults to optax_muon)
    argv_default = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "model_name=llama2-7b",
    ]
    config_default = pyconfig.initialize(argv_default)

    with (
        patch.object(optimizers, "get_muon_weight_dimension_numbers") as mock_get_mdn,
        patch.object(optimizers, "optax_muon") as mock_optax_muon,
    ):
      mock_get_mdn.return_value = {}
      optimizers.get_optimizer(config_default, learning_rate_schedule, model=model)
      mock_get_mdn.assert_called_once_with(model, config_default)
      mock_optax_muon.assert_called_once()
      _, kwargs = mock_optax_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 5)
      self.assertEqual(kwargs["ns_coeffs"], (3.4445, -4.7750, 2.0315))
      self.assertEqual(kwargs["muon_weight_dimension_numbers"], mock_get_mdn.return_value)
      self.assertNotIn("use_all_to_all", kwargs)

    # Case 4: Explicit Optax Muon (muon_type=optax_muon)
    argv_optax = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=optax_muon",
        "model_name=llama2-7b",
    ]
    config_optax = pyconfig.initialize(argv_optax)

    with (
        patch.object(optimizers, "get_muon_weight_dimension_numbers") as mock_get_mdn,
        patch.object(optimizers, "optax_muon") as mock_optax_muon,
    ):
      mock_get_mdn.return_value = {}
      optimizers.get_optimizer(config_optax, learning_rate_schedule, model=model)
      mock_get_mdn.assert_called_once_with(model, config_optax)
      mock_optax_muon.assert_called_once()
      _, kwargs = mock_optax_muon.call_args
      self.assertEqual(kwargs["ns_steps"], 5)
      self.assertEqual(kwargs["ns_coeffs"], (3.4445, -4.7750, 2.0315))
      self.assertEqual(kwargs["muon_weight_dimension_numbers"], mock_get_mdn.return_value)
      self.assertNotIn("use_all_to_all", kwargs)

    # Case 5: Optax Muon with muon_use_all_to_all=True raises AssertionError
    argv_optax_a2a = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=optax_muon",
        "model_name=llama2-7b",
        "muon_use_all_to_all=True",
    ]
    config_optax_a2a = pyconfig.initialize(argv_optax_a2a)
    with self.assertRaises(AssertionError):
      optimizers.get_optimizer(config_optax_a2a, learning_rate_schedule, model=model)

    # Case 6: Invalid muon_type raises ValueError
    argv_invalid = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=invalid_type",
        "model_name=llama2-7b",
    ]
    config_invalid = pyconfig.initialize(argv_invalid)
    with self.assertRaises(ValueError):
      optimizers.get_optimizer(config_invalid, learning_rate_schedule, model=model)

  def test_maxtext_muon_wires_named_sharding_and_mesh(self):
    """Verifies that get_optimizer forwards mesh and propagates ShardedMuonDimensionNumbers with NamedSharding."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    sharded_dim_nums = {
        "params": {
            "mlp": {
                "kernel": smdn(
                    reduction_axis=(0,),
                    output_axis=(-1,),
                    sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "model")),
                )
            }
        }
    }
    model = MagicMock()
    learning_rate_schedule = MagicMock()
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "muon_type=maxtext_muon",
        "model_name=llama2-7b",
    ]
    config = pyconfig.initialize(argv)

    with (
        patch.object(
            optimizers, "get_maxtext_muon_weight_dimension_numbers", return_value=sharded_dim_nums
        ) as mock_get_smdn,
        patch.object(optimizers, "maxtext_muon") as mock_maxtext_muon,
    ):
      optimizers.get_optimizer(config, learning_rate_schedule, model=model, mesh=mesh)
      mock_get_smdn.assert_called_once_with(model, config, mesh=mesh)
      mock_maxtext_muon.assert_called_once()
      _, kwargs = mock_maxtext_muon.call_args
      passed_dim_nums = kwargs["muon_weight_dimension_numbers"]
      self.assertEqual(passed_dim_nums, sharded_dim_nums)
      kernel_spec = passed_dim_nums["params"]["mlp"]["kernel"]
      self.assertIsInstance(kernel_spec.sharding, jax.sharding.NamedSharding)
      self.assertEqual(kernel_spec.sharding.mesh, mesh)
      self.assertEqual(kernel_spec.sharding.spec, jax.sharding.PartitionSpec("data", "model"))


if __name__ == "__main__":
  absltest.main()
