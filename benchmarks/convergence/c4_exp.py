import dataclasses
from benchmarks.maxtext_training_configs import DatasetHParams, ConvHParams, _setup_model_convergence_
from benchmarks.maxtext_trillium_model_configs import MaxTextModel, _add_to_model_dictionary
from benchmarks.maxtext_v5p_model_configs import v5p_model_dict, deepseek_v3_ep_256_v5p_512

e2e_model_dict = {}

c4_mlperf_hp = DatasetHParams(
        name="c4mlperf",
        dataset_path="gs://max-datasets-rogue",
        dataset_name="c4/en:3.0.7",
        eval_dataset_name="c4/en:3.0.9",
        dataset_type="c4_mlperf",
        train_split="train2",
        eval_split="validation",
        eval_tokens=47185920, # 5760*8192 training_tokens, special requirment from mlperf
        add_bos=False,
        add_eos=False)

c4_en_hp = DatasetHParams(
        name="c4en",
        dataset_path="gs://maxtext-dataset",
        dataset_name="c4/en:3.0.1",
        dataset_type="tfds",
        train_split="train",
        eval_split="validation",
        eval_tokens=75497472, 
        add_bos=False,
        add_eos=False)

c4_mutil_hp = DatasetHParams(
        name="c4multi",
        dataset_path="gs://mlperf-llm-public2",
        dataset_name="c4/multilingual:3.1.0",
        dataset_type="tfds",
        train_split="en",
        eval_split="en-validation",
        eval_tokens= 824 * 512, #824 * 512 
        add_bos=False,
        add_eos=False)

llama3_405b_hp = ConvHParams(
    global_batch_size = 1152,
    warmup_samples = 8216000, 
    decay_end_samples = 1382400000.0,
    total_tokens_to_train = 2.64e9,
    training_scaleing_factor = 1.0,
    learning_rate = 6.944e-8,
    eval_tokens = 47185920,
    eval_interval = 377487360,
    )

# [todo] resue 405b convergence benchmark hp for now. not tuned yet
deepseek_671b_hp = ConvHParams(
    global_batch_size = 1152,
    warmup_samples = 8216000, 
    decay_end_samples = 1382400000.0,
    total_tokens_to_train = 2.64e9,
    training_scaleing_factor = 1.0,
    learning_rate = 6.944e-8,
    eval_tokens = 47185920,
    eval_interval = 377487360,
    )

import math

def setupDataset(model: MaxTextModel, params: DatasetHParams):
    model.tuning_params["reuse_example_batch"] = -1
    model.tuning_params["dataset_path"] = params.dataset_path
    model.tuning_params["dataset_name"] = params.dataset_name
    model.tuning_params["dataset_type"] = params.dataset_type
    model.tuning_params["eval_dataset_name"] = params.dataset_name
    model.tuning_params["train_split"] = params.train_split
    model.tuning_params["eval_split"] = params.eval_split
    model.tuning_params["add_bos"] = params.add_bos
    model.tuning_params["add_eos"] = params.add_eos
    model.tuning_params["eval_steps"] = params.eval_tokens
    model.tuning_params["data_shuffle_seed"] = 1238


def setupC4Multilingualen(model: MaxTextModel):
    setupDataset(model, c4_mutil_hp)

def setupC4En(model: MaxTextModel):
    setupDataset(model, c4_en_hp)

def setupC4Mlperf(model: MaxTextModel):
    setupDataset(model, c4_mlperf_hp)   

def load_checkpoint(model: MaxTextModel, checkpoint_path: str):
    model.tuning_params["load_full_state_path"] = checkpoint_path

# Run this for new definitions that should be part of the library.
c4_deepseek_v3_ep_256_v5p_512_gbs_1024 = _add_to_model_dictionary(
    e2e_model_dict,
    _setup_model_convergence_(
    deepseek_v3_ep_256_v5p_512,
    c4_mlperf_hp,
    deepseek_671b_hp,
    global_batch_size=1024,
    num_devices=256,
    )
)
