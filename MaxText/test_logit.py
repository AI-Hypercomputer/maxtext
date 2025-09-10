"""test_logit.py
cd ~/maxtext
python -m MaxText.test_logit
"""

import jax
from flax.linen import partitioning as nn_partitioning
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations
import optax
from jax import numpy as jnp
import numpy as np
# experimental muon
from MaxText.muon_test import MuonWeightSpec, muon

Transformer = models.Transformer

def print_nested_keys(data, prefix=""):
  """
  Prints nested keys of a dictionary-like structure in a directory-like format.
  Args:
      data: The dictionary-like structure to traverse.
      prefix: The current path prefix.
  """
  if isinstance(data, dict):
    for key, value in data.items():
      current_path = f"{prefix}{key}."
      print_nested_keys(value, current_path)
  else:
    print(f"key: {prefix} | value shape: {data.shape}")



class DataIter:

  def __init__(self, rng, max_target_length, global_batch_size_to_train_on, base_emb_dim):
    self.max_target_length = max_target_length
    self.global_batch_size = global_batch_size_to_train_on
    self.embed_dim = base_emb_dim
    self.rng = rng

  def get_data(self):
    """get data"""
    lnx = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 100, dtype=jax.numpy.int32)
    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )
    return lnx, decoder_segment_ids, decoder_positions


# base_num_decoder_layers: 2
# main("mixtral-8x7b")
# base_emb_dim: 512, base_moe_mlp_dim: 512, base_num_decoder_layers: 2, first_num_dense_layers: 2
# main("deepseek3-671b")
# main("deepseek2-16b")

# base_num_decoder_layers: 4
model_name = "deepseek2-16b"
rng = jax.random.PRNGKey(0)
argv = [
    "something.py",
    "MaxText/configs/base.yml",
    "per_device_batch_size=1",
    "max_target_length=2048",
    "skip_jax_distributed_system=true",
    f"model_name={model_name}",
]
config = pyconfig.initialize(argv)
quant = quantizations.configure_quantization(config)
devices_array = maxtext_utils.create_device_mesh(config)
mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
quant = quantizations.configure_quantization(config)
maxtext_model = models.Transformer(config, mesh, quant=quant)

# input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
# model_vars = maxtext_model.init(
#     {"params": rng, "dropout": rng, "aqt": rng},
#     jnp.ones(input_shape, dtype=jnp.int32),
#     jnp.ones(input_shape, dtype=jnp.int32),
# )


maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng, mesh, None)
model_vars = maxtext_state.params

print_nested_keys(model_vars)
# print(maxtext_model)
# print(maxtext_state)


# Manually define the spec for each parameter in a nested dictionary
# This structure must match your `params` PyTree exactly.
muon_weight_specs = {
    "params": {
        "token_embedder": {
            # Standard 2D matrix (102400, 2048)
            # "embedding": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))#None
            "embedding": optax.MaskedNode()
        },
        "decoder": {
            "decoder_norm": {
                # 1D tensor, not a matrix. Fallback to Adam.
                "scale": optax.MaskedNode()
            },
            "dense_layers": {
                "mlp": {
                    # Shape (2048, 1, 10944) -> Matrix(2048, 10944), Batch(1,)
                    "wi_0": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                    "wi_1": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                    # Shape (10944, 1, 2048) -> Matrix(10944, 2048), Batch(1,)
                    "wo": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                },
                "post_self_attention_layer_norm": {"scale": optax.MaskedNode()},
                "pre_self_attention_layer_norm": {"scale": optax.MaskedNode()},
                "self_attention": {
                    # 4D tensors. Treat last two dims as the matrix, first two as batch.
                    # Shape (2048, 1, 16, 128) -> Matrix(16, 128), Batch(2048, 1)
                    "key": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "kv_norm": {"scale": optax.MaskedNode()},
                    # Shape (16, 1, 128, 2048) -> Matrix(128, 2048), Batch(16, 1)
                    "out": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "query": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "value": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    # 3D tensor -> Matrix(2048, 576), Batch(1,)
                    "wkv_a": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                    "wkv_b": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                },
            },
            "logits_dense": {
                # Standard 2D matrix (2048, 102400)
                "kernel": optax.MaskedNode()  # MuonWeightSpec(reduction_axes=(0,), output_axes=(1,))
            },
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        # Shape (2048, 26, 64) -> Matrix(2048, 64), Batch(26,)
                        "gate": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                        # 4D MoE weights -> Matrix(2048, 1408), Batch(64, 26)
                        "wi_0": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,)),
                        "wi_1": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,)),
                        "wo": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,)),
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                        "wi_1": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                        "wo": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                    },
                },
                "post_self_attention_layer_norm": {"scale": optax.MaskedNode()},
                "pre_self_attention_layer_norm": {"scale":optax.MaskedNode()},
                "self_attention": {
                    "key": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "kv_norm": {"scale": optax.MaskedNode()},
                    "out": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "query": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "value": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                    "wkv_a": {"kernel": MuonWeightSpec(reduction_axes=(0,), output_axes=(2,))},
                    "wkv_b": {"kernel": MuonWeightSpec(reduction_axes=(-2,), output_axes=(-1,))},
                },
            },
        },
    }
}

muon_weight_mask = jax.tree.map(
    lambda spec: not isinstance(spec, optax.MaskedNode),
    muon_weight_specs,
    is_leaf=lambda x: isinstance(x, MuonWeightSpec) or isinstance(x, optax.MaskedNode),
)


# muon_weight_specs = None
learning_rate = 1e-3
print("\nmuon_weight_mask:", muon_weight_mask)
print("\nmuon_weight_specs:", muon_weight_specs)
optimizer = muon(learning_rate, muon_weight_specs=muon_weight_specs, muon_weight_mask=muon_weight_mask)
opt_state = optimizer.init(model_vars)
data = DataIter(rng, config.max_target_length, config.global_batch_size_to_train_on, config.base_emb_dim)
lnx, decoder_segment_ids, decoder_positions = data.get_data()


def f(model_vars):
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    logits = maxtext_model.apply(
        model_vars,
        decoder_input_tokens=lnx,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        # train mode
        model_mode=MODEL_MODE_TRAIN,
        # deterministic
        enable_dropout=False,
        rngs={"dropout": rng, "params": rng},
    )
  return jnp.sum(logits)


grad = jax.grad(f)(model_vars)
updates, opt_state = optimizer.update(grad, opt_state, model_vars)