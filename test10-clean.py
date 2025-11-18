import os
# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
print(os.getcwd())

from vllm import LLM


from tunix.generate import utils
from tunix.rl import reshard

import pathwaysutils

pathwaysutils.initialize()
from flax import nnx

from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter


MODEL = "unsloth/gpt-oss-20b-BF16"


def get_ref_maxtext_model(config):
  model, mesh = model_creation_utils.create_nnx_model(config)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model, use_standalone_mappings=True)
  return tunix_model, mesh


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

gpt_oss, mesh = get_ref_maxtext_model(config_ref)
print("Maxtext model loaded successfully")
src_state = nnx.state(gpt_oss)
for k, v in src_state.flat_state():
  print("-".join(k), "|", v.value.shape)


MODEL = "unsloth/gpt-oss-20b-BF16"
golden_llm = LLM(
    MODEL,
    max_model_len=64,
    # max_model_len=128,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
)

print(golden_llm.generate("what is the capital of France?"))
print("vLLM model loaded successfully")

dst_golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state
reshard_fn = reshard.reshard_pytree

tgt_flat_list = dst_golden_state.flat_state()
tgt_flat_dict = {".".join(str(k) for k in keys): v for keys, v in tgt_flat_list}
# mapping = QWEN3_VLLM_MAPPING.to_hf_mapping()
# hooks = QWEN3_VLLM_MAPPING.to_hf_hook_fns()
# transpose = QWEN3_VLLM_MAPPING.to_hf_transpose_keys()
# result = utils.transfer_state_with_mappings(src_state=src_state, dst_state=dst_golden_state, key_mappings=qwen3_8b.to_hf_mappings(), key_mapping_hook_fns=qwen3_8b.to_hf_hook_fns(), transpose_keys=qwen3_8b.to_hf_transpose_keys(), reshard_fn=reshard.reshard_pytree,)


def transfer(src_state, dst_state):
  pass


result = transfer(src_state)


matched = utils.verify_state_closeness(state=result, golden_state=dst_golden_state)
print(matched)
