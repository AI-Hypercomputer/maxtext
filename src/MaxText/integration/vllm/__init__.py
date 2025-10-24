from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

logger = init_logger(__name__)

def register():
    from .adapter import MaxTextForCausalLM
    register_model("MaxTextForCausalLM", MaxTextForCausalLM)