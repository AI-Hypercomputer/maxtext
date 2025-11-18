from MaxText import model_creation_utils
from MaxText import pyconfig
from flax import nnx


MODEL = "unsloth/gpt-oss-20b-BF16"


config_ref = pyconfig.initialize(
    [
        "",
        "src/MaxText/configs/base.yml",
    ],
    base_output_directory="gs://runner-maxtext-logs", 
    run_name="test-tunix-maxtext-gpt-oss",
    tokenizer_type="huggingface",
    tokenizer_path=MODEL,
    load_parameters_path="gs://shuningjin-multipod-dev/gpt-oss-20b/scan-flags-false-2025-11-11-01-42-40/0/items",
    per_device_batch_size=1,
    max_prefill_predict_length=32,
    max_target_length=64,
    steps=100,
    async_checkpointing="false",
    model_name="gpt-oss-20b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
)

model, mesh = model_creation_utils.create_nnx_model(config_ref)

print("Maxtext model loaded successfully")
src_state = nnx.state(model)
for k, v in src_state.flat_state():
  print("-".join(k), "|", v.value.shape)
