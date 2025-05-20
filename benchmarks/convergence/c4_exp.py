import dataclasses
from maxtext_training_configs import DatasetHParams, ConvHParams, _setup_model_convergence_
from maxtext_trillium_model_configs import MaxTextModel, _add_to_model_dictionary
from maxtext_v5p_model_configs import v5p_model_dict
import xla_flags_library

c4_mlperf_hp = DatasetHParams(
        name="c4mlperf",
        dataset_path="gs://mlperf-exp-us-east1-cp0",
        dataset_name="c4/en:3.0.7",
        dataset_type="c4_mlperf",
        train_split="train",
        eval_split="c4/en:3.0.5",
        eval_steps=4 * 512,
        add_bos=False,
        add_eos=False)

c4_en_hp = DatasetHParams(
        name="c4en",
        dataset_path="gs://maxtext-dataset",
        dataset_name="c4/en:3.0.1",
        dataset_type="tfds",
        train_split="train",
        eval_split="validation",
        eval_steps=36 * 512,
        add_bos=False,
        add_eos=False)

c4_mutil_hp = DatasetHParams(
        name="c4multi",
        dataset_path="gs://mlperf-llm-public2",
        dataset_name="c4/multilingual:3.1.0",
        dataset_type="tfds",
        train_split="en",
        eval_split="en-validation",
        eval_steps= 824 * 512, #152 * 512 #for llama2 token, #824 * 512 #206 * 2048, # 852 * 512
        add_bos=False,
        add_eos=False)

llama3_405b_hp = ConvHParams(
    global_batch_size = 1152,
    warmup_samples = 8216000, 
    decay_end_samples = 1382400000.0,
    total_tokens_to_train = 2.64e9,
    training_scaleing_factor = 1.0,
    learning_rate = 6.944e-8,
    eval_samples = 47185920,
    eval_interval = 377487360
    )

# [todo] resue 405b convergence benchmark hp for now. not tuned yet
deepseek_671b_hp = ConvHParams(
    global_batch_size = 1152,
    warmup_samples = 8216000, 
    decay_end_samples = 1382400000.0,
    total_tokens_to_train = 2.64e9,
    training_scaleing_factor = 1.0,
    learning_rate = 6.944e-8,
    eval_samples = 47185920,
    eval_interval = 377487360
    )

import math

def setupDataset(model: MaxTextModel, params: DatasetHParams):
    #model.model_name = model.model_name + "-" + params.name
    model.tuning_params["reuse_example_batch"] = -1
    model.tuning_params["dataset_path"] = params.dataset_path
    model.tuning_params["dataset_name"] = params.dataset_name
    model.tuning_params["dataset_type"] = params.dataset_type
    model.tuning_params["eval_dataset_name"] = params.dataset_name
    model.tuning_params["train_split"] = params.train_split
    model.tuning_params["eval_split"] = params.eval_split
    model.tuning_params["add_bos"] = params.add_bos
    model.tuning_params["add_eos"] = params.add_eos
    model.tuning_params["eval_steps"] = params.eval_steps
    model.tuning_params["data_shuffle_seed"] = 1238


def setupC4Multilingualen(model: MaxTextModel):
    setupDataset(model, c4_mutil_hp)

def setupC4En(model: MaxTextModel):
    setupDataset(model, c4_en_hp)

def setupC4Mlperf(model: MaxTextModel):
    setupDataset(model, c4_mlperf_hp)   

def setupConvHParams(model: MaxTextModel, params: ConvHParams, num_devices: int):
    gbs = params.global_batch_size
    total_steps = params.total_tokens_to_train / gbs
    model.tuning_params["per_device_batch_size"] = float(gbs / num_devices)
    model.tuning_params["learning_rate"] =  params.learning_rate
    model.tuning_params["warmup_steps_fraction"] =  float(params.warmup_samples) / gbs / total_steps
    model.tuning_params["learning_rate_schedule_steps"] =  int(params.decay_end_samples / gbs)
    model.tuning_params["steps"] =  int(total_steps)
    eval_samples =  model.tuning_params["eval_steps"]
    model.tuning_params["eval_steps"] = int(math.floor(eval_samples / gbs))
    model.tuning_params["eval_interval"]= int(math.ceil(params.eval_interval / gbs))
    model.tuning_params["enable_checkpointing"] = True
    model.tuning_params["checkpoint_period"] = int(math.ceil( 1000 * 512 / gbs))

def load_checkpoint(model: MaxTextModel, checkpoint_path: str):
    model.tuning_params["load_full_state_path"] = checkpoint_path

def setupLLama405BConvHParams(model: MaxTextModel, params: ConvHParams, num_devices: int):
    gbs = params.global_batch_size
    total_steps = params.total_tokens_to_train / gbs

    warmup_steps = math.ceil(8000.0 * 1152 / gbs - 1e-6)
    decay_end_step = math.ceil(1200000.0 * 1152 / gbs - 1e-6)

    model.tuning_params["per_device_batch_size"] = float(gbs / num_devices)
    model.tuning_params["learning_rate"] =  (8.0e-5 * gbs) / 1152
    model.tuning_params["warmup_steps_fraction"] =  float(warmup_steps / decay_end_step)
    model.tuning_params["learning_rate_schedule_steps"] = decay_end_step
    model.tuning_params["steps"] =  int(total_steps)
    eval_samples =  model.tuning_params["eval_steps"]
    model.tuning_params["eval_steps"] = int(math.ceil(5760 * 8192 / max_target_length / gbs))
    model.tuning_params["eval_interval"]= int(math.ceil(377487360 / max_target_length / gbs))
    model.tuning_params["enable_checkpointing"] = True

# # Run this for new definitions that should be part of the library.
# c4_deepseek_v3_ep_256_v5p_512_gbs_1024 = _add_to_model_dictionary(
#     v5p_model_dict,
#     _setup_model_convergence_(
#     c4_deepseek_v3_ep_256_v5p_512,
#     c4_en_hp,
#     deepseek_671b_hp,
#     global_batch_size=1024,
#     num_devices=256,
#     )
# )

# # Run this for new definitions that should be part of the library.
# c4_deepseek_v3_ep_256_v5p_512_gbs_1024 = _setup_model_convergence_(
#     v5p_model_dict,
#     c4_deepseek_v3_ep_256_v5p_512,
#     c4_en_hp,
#     deepseek_671b_hp,
#     global_batch_size=1024,
#     num_devices=256,
#     )

