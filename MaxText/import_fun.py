import jax
from MaxText import pyconfig
from MaxText.train import main as train_main
import MaxText.tests.pipeline_parallelism_test
import MaxText.tests.aot_hlo_identical_test
import MaxText.tests.attention_test
import MaxText.tests.check_mla_vs_reference
import MaxText.tests.decode_tests
import MaxText.tests.distillation_data_processing_test
import MaxText.tests.elastic_train_test
import MaxText.tests.forward_pass_logit_checker
import MaxText.tests.globals
import MaxText.tests.gpt3_test
import MaxText.tests.grain_data_processing_test
import MaxText.tests.grpo_trainer_correctness_test
import MaxText.tests.hf_checkpoint_conversion_checker
import MaxText.tests.hf_checkpoint_conversion_test
import MaxText.tests.hf_data_processing_test
#import MaxText.tests.kernels_test
import MaxText.tests.llama_test
import MaxText.tests.llama4_logit_verification_script
import MaxText.tests.max_utils_test
import MaxText.tests.maxengine_test
import MaxText.tests.maxtext_utils_test
import MaxText.tests.model_test
import MaxText.tests.moe_test
import MaxText.tests.multi_token_prediction_test
import MaxText.tests.multihost_dataloading_test
import MaxText.tests.multimodal_utils_test
import MaxText.tests.pipeline_parallelism_test
import MaxText.tests.profiler_test
import MaxText.tests.pyconfig_test
import MaxText.tests.quantizations_test
import MaxText.tests.sft_data_processing_test
import MaxText.tests.sft_trainer_correctness
import MaxText.tests.simple_decoder_layer_test
import MaxText.tests.state_dtypes_test
import MaxText.tests.tfds_data_processing_test
import MaxText.tests.tokenizer_test
import MaxText.tests.train_compile_test
# import MaxText.tests.train_gpu_smoke_test
# import MaxText.tests.train_int8_smoke_test
# import MaxText.tests.train_smoke_test
# import MaxText.tests.train_using_ragged_dot_smoke_test

######### Integration tests ########
import MaxText.tests.integration_tests.checkpoint_compatibility_test
import MaxText.tests.integration_tests.checkpointing_test
import MaxText.tests.integration_tests.generate_param_only_checkpoint_test
import MaxText.tests.integration_tests.gradient_accumulation_test
import MaxText.tests.integration_tests.grpo_correctness
import MaxText.tests.integration_tests.sft_trainer_correctness_test
import MaxText.tests.integration_tests.standalone_dl_ckpt_test
import MaxText.tests.integration_tests.train_tests


# The Bad guys:
# 
#from pedagogical_examples.shmap_collective_matmul import main
# import MaxText.tests.kernels_test

jax.distributed.initialize()

print("yay", flush=True)