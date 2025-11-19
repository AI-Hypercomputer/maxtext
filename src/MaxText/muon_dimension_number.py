import jax
import jax.numpy as jnp
from optax.contrib import MuonDimensionNumbers as mdn
from optax.contrib import MuonDimensionNumbers
from MaxText.maxtext_utils import get_abstract_param
from typing import Any, Dict, Tuple, Optional, Union

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

GEMMA3 = {
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


# def transform_logic(path):
#   """
#   assume scan (i.e., dim 1 is layer num L), should work with unscan (without L)
#   works for deepseek, llama2, gemma3
#   """
#   # moe: [0, L, -2, -1]
#   if "MoeBlock_0" in path and ("wo" in path or "wi_0" in path or "wi_1" in path):
#     return mdn((-2,), (-1,))
#   # attention out proj: [0, L, -2, -1]
#   elif "self_attention" in path and "out" in path:
#     return mdn((0, -2), (-1,))
#   # attention qkv proj: [0, L, -2, -1]
#   elif "self_attention" in path and (
#       "query" in path or "key" in path or "value" in path or "wq_b" in path or "wkv_b" in path
#   ):
#     return mdn((0,), (-2, -1))
#   # do not apply muon: scalar, embedding, unembedding
#   elif "scale" in path or "bias" in path or "embedding" in path or "logits_dense" in path:
#     return None
#   else:
#     # all other: [0, L, -1]
#     return mdn((0,), (-1,))


def is_path_contain_any(tuples, path):
  return any(x in path for x in tuples)


def transform_logic(path: Tuple[str, ...]) -> Optional[MuonDimensionNumbers]:
  """
  Determines Muon dimension numbers based on parameter path.

  Strategy:
  1. Filter out exclusions (Norms, Biases, Embeddings).
  2. Handle MoE specific shapes.
  3. Handle Attention specific shapes (QKV split vs Output proj).
  4. Default to standard Dense layer shape.
  """

  # 1. Exclude parameters not suitable for Muon (scalar scales, biases, embeddings)
  if is_path_contain_any(("scale", "bias", "embedding", "logits_dense"), path):
    return None

  # 2. Mixture of Experts (MoE)
  # Detects DeepSeek MoE layers.
  # Weights inside experts are usually treated with mass on dim -2.
  if "MoeBlock_0" in path:
    # Exclude 'gate' (which behaves like a dense layer)
    if is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      return mdn((-2,), (-1,))

  # 3. Self Attention
  if "self_attention" in path:
    # Output projection: combines heads, so dim structure is different
    if "out" in path:
      return mdn((0, -2), (-1,))

    # Projection matrices (Query, Key, Value, etc.)
    # Note: wkv_a/wq_a often act as compression/dense, wkv_b/wq_b expand to heads
    if is_path_contain_any(("query", "key", "value", "wq_b", "wkv_b"), path):
      return mdn((0,), (-2, -1))

    # # 'a' matrices in DeepSeek (Low Rank adapters) behave like standard dense
    # if any(x in leaf for x in ("wq_a", "wkv_a")):
    #   return mdn((0,), (-1,))

  # 4. Default Dense / Standard Weights
  # Assume scanned [Layer, In, Out] -> Newton update on [In, Out] averaged over Layer
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(path)


def test1():
  assert get_transform_tree(DEEPSEEK2_DIMENSION_NUMBER) == DEEPSEEK2_DIMENSION_NUMBER
  assert get_transform_tree(DEEPSEEK3_DIMENSION_NUMBER) == DEEPSEEK3_DIMENSION_NUMBER


def test2():
  from MaxText import pyconfig, maxtext_utils
  from MaxText.globals import MAXTEXT_PKG_DIR
  from MaxText.layers import models, quantizations
  import os

  Transformer = models.transformer_as_linen

  def _test2(model_name):
    # init model
    argv = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), f"model_name={model_name}"]
    config = pyconfig.initialize(argv)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = Transformer(config, mesh=mesh, quant=quant)
    # quickly get param structure without materialization
    abstract_param = get_abstract_param(model, config)
    print(abstract_param)
    # get muon dimension number
    transform_tree = get_transform_tree(abstract_param)
    return transform_tree

  assert _test2("deepseek2-16b") == DEEPSEEK2_DIMENSION_NUMBER
  assert _test2("deepseek3-test") == DEEPSEEK3_DIMENSION_NUMBER
  assert _test2("deepseek3-671b") == DEEPSEEK3_DIMENSION_NUMBER
  assert _test2("llama2-7b") == LLAMA2_DIMENSION_NUMBER
  assert _test2("gemma3-4b") == GEMMA3


if __name__ == "__main__":
  # python -m MaxText.muon_dimension_number
  test1()
  test2()
