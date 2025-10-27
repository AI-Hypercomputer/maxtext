from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

logger = init_logger(__name__)

def register():
    logger.info("Registering MaxTextForCausalLM model with tpu_inference and vllm")
    from .adapter import MaxTextForCausalLM
    register_model("MaxTextForCausalLM", MaxTextForCausalLM)
    logger.info("Successfully registered MaxTextForCausalLM model")