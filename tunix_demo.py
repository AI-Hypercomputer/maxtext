from flax import nnx
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
# from qwix import lora
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
import os


import MaxText as mt
from MaxText import pyconfig

# Data
BATCH_SIZE = 16

# Model
MESH = [(1, 8), ("fsdp", "tp")]
# LoRA
RANK = 16
ALPHA = 2.0

# Train
MAX_STEPS = 100
EVAL_EVERY_N_STEPS = 20
NUM_EPOCHS = 3


# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "~/qlora_expt/content/intermediate_ckpt/"
CKPT_DIR = "~/qlora_expt/content/ckpts/"
PROFILING_DIR = "~/qlora_expt/content/profiling/"

config = pyconfig.initialize(
    ["", "MaxText/configs/base.yml"], #TODO: @mazumdera: why decode.py?
    base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3-8b",
    # dataset_path=we use Tunix's dataset
    # load_parameters_path="gs://maxtext-gemma/2b/", #TODO: @mazumdera: change this to use checkpoint
    tokenizer_type="tiktoken",
    tokenizer_path="assets/tokenizer_llama3.tiktoken",
    per_device_batch_size=1,
    max_target_length=8192,
    steps=10,
    async_checkpointing="false",
    model_name="llama3-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
)

# checkpoint = mt.checkpointing.load_params_from_path(
#     load_parameters_from_path="gs://maxtext-gemma/2b/",
#     abstract_unboxed_params=None,
#     checkpoint_storage_concurrent_gb=None,
# )
checkpoint = {}

def create_model():
  return mt.from_pretrained(config, rngs=nnx.Rngs(params=0, dropout=1))

model = nnx.eval_shape(create_model)

@nnx.jit
def partial_init(checkpoint):
  model = create_model()
  nnx.update(model, checkpoint)
  # shard model
  state = nnx.state(model)
  specs = nnx.get_partition_spec(state)
  state = jax.lax.with_sharding_constraint(state, specs)
  nnx.update(model, state)
  return model

with jax.sharding.use_mesh(model.mesh), nn.logical_axis_rules(config.logical_axis_rules):
  model = partial_init(checkpoint)
print(model)

decoder_input_tokens = jnp.zeros((config.global_batch_size_to_train_on, config.max_target_length), dtype=jnp.int32)
decoder_positions = jnp.zeros_like(decoder_input_tokens)

output = model(decoder_input_tokens, decoder_positions)
print("Output shape:", output.shape)