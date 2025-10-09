# from optax.contrib import MuonDimensionNumbers as mdn
from MaxText.muon import MuonDimensionNumbers as mdn
import jax
import jax.numpy as jnp

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
  work for deepseek, could extend other models
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


def get_abstract_param(model, config):
  key = jax.random.PRNGKey(0)
  # we only need the parameter structure, input size is irrelavent so use smallest
  input_shape = (1, 1)  # (batch, length)
  if config.use_multimodal:
    image_shape = (1, 1, 1, 1, 1)
    encoder_images = jnp.ones(image_shape, dtype=jnp.int32)
  else:
    encoder_images = None
  abstract_vars = jax.eval_shape(
      model.init,
      {"params": key, "dropout": key, "aqt": key},
      jnp.ones(input_shape, dtype=jnp.int32),
      jnp.ones(input_shape, dtype=jnp.int32),
      encoder_images=encoder_images,
  )
  return abstract_vars


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
    rng = jax.random.PRNGKey(0)
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


if __name__ == "__main__":
  # python -m MaxText.muon_dimension_number
  test1()
  test2()
