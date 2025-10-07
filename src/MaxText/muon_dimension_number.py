# from optax.contrib import MuonDimensionNumbers as mdn
from MaxText.muon import MuonDimensionNumbers as mdn

# deepseek2, scanned
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
                        "wi_0": {"kernel": mdn((-2,), (-1,))},
                        "wi_1": {"kernel": mdn((-2,), (-1,))},
                        "wo": {"kernel": mdn((-2,), (-1,))},
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
                        "wi_0": {"kernel": mdn((-2,), (-1,))},
                        "wi_1": {"kernel": mdn((-2,), (-1,))},
                        "wo": {"kernel": mdn((-2,), (-1,))},
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


def transform_logic(x, path):
  """
  with scan in mind, but should work with unscan
  works for deepseek, could extend other models
  """
  path_str = "/".join(path)
  # [0, L, -2, -1], yellow
  if "MoeBlock_0" in path_str and ("wo" in path_str or "wi_0" in path_str or "wi_1" in path_str):
    return mdn((-2,), (-1,))
  # [0, L, -2, -1], yellow
  elif "self_attention" in path_str and "out" in path_str:
    return mdn((0, -2), (-1,))
  # [0, L, -2, -1], query for ds2, wq_b for ds3, yellow
  elif "self_attention" in path_str and ("wkv_b" in path_str or "query" in path_str or "wq_b" in path_str):
    return mdn((0,), (-2, -1))
  # gray, do not apply muon: scalar, embedding, unembedding
  elif "scale" in path_str or "bias" in path_str or "embedding" in path_str or "logits_dense" in path_str:
    return None
  else:
    # [0, L, -1], blue, all other 3d
    return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(tree, path)


if __name__ == "__main__":
  # python -m MaxText.muon_dimension_number
  assert get_transform_tree(DEEPSEEK2_DIMENSION_NUMBER) == DEEPSEEK2_DIMENSION_NUMBER
  assert get_transform_tree(DEEPSEEK3_DIMENSION_NUMBER) == DEEPSEEK3_DIMENSION_NUMBER
