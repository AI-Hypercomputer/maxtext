from vllm import LLM
import os
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ["HF_TOKEN"] = ""
os.environ["TPU_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
os.environ["VLLM_MLA_DISABLE"] = "1"

# os.environ["MODEL_IMPL_TYPE"] = "vllm"

# os.environ["TPU_BACKEND_TYPE"] = "jax"


"""
LOCAL_DIR=/home/shuningjin_google_com/gpt-oss-20b
mkdir -p $LOCAL_DIR

# dequantized hf checkpoint in bf-16
gcloud storage cp -r gs://shuningjin-multipod-dev/gpt-oss-20b/gpt-oss-20b-bf16-v2 $LOCAL_DIR

# download from https://huggingface.co/openai/gpt-oss-20b/tree/main
hf download openai/gpt-oss-20b \
    --include "tokenizer*" "vocab*" "special_tokens_map*" \
    --local-dir $LOCAL_DIR/gpt-oss-20b-bf16-v2
"""


# # MODEL = "openai/gpt-oss-120b"

# MODEL ="/home/shuningjin_google_com/gpt-oss-20b/gpt-oss-20b-bf16-v2"


MODEL = "unsloth/gpt-oss-20b-BF16"

golden_llm = LLM(
    MODEL,
    max_model_len=128,
    tensor_parallel_size=4
)


# MODEL = "Qwen/Qwen3-30B-A3B"
# golden_llm = LLM(
#     MODEL,
#     max_model_len=128,
#     tensor_parallel_size=8
#     )


print(golden_llm.llm_engine.model_executor.driver_worker.model_runner.state)

print(golden_llm.generate("what is the capital of France?"))
breakpoint()
