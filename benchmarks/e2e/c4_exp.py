import dataclasses
from maxtext_trillium_model_configs import MaxTextModel, DatasetHParams, ConvHParams
import xla_flags_library

c4_mlperf_hp = DatasetHParams(
        dataset_path="gs://mlperf-exp-us-east1-cp0",
        dataset_name="c4/en:3.0.7",
        dataset_type="c4_mlperf",
        train_split="train",
        eval_split="c4/en:3.0.5",
        eval_steps=4 * 512,
        add_bos=False,
        add_eos=False,
        tokenizer_path="gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model")

c4_en_hp = DatasetHParams(
        dataset_path="gs://maxtext-dataset",
        dataset_name="c4/en:3.0.1",
        dataset_type="tfds",
        train_split="train",
        eval_split="validation",
        eval_steps=36 * 512,
        add_bos=False,
        add_eos=False,
        tokenizer_path="gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model")

c4_mutil_hp = DatasetHParams(
        dataset_path="gs://mlperf-llm-public2",
        dataset_name="c4/multilingual:3.1.0",
        dataset_type="tfds",
        train_split="en",
        eval_split="en-validation",
        eval_steps=852 * 512,
        add_bos=False,
        add_eos=False,
        tokenizer_path="gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model")

llama3_1_8b_8192_c4en = MaxTextModel(
    model_name="llama3_1_8b_8192_c4en",
    model_type="llama3.1-8b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "qkv_proj_offloaded",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": c4_en_hp.dataset_path,
        "dataset_name": c4_en_hp.dataset_name,
        "dataset_type": c4_en_hp.dataset_type,
        "tokenizer_path": c4_en_hp.tokenizer_path,
        "train_split": c4_en_hp.train_split,
        "eval_split": c4_en_hp.eval_split,
        "add_bos": c4_en_hp.add_bos,
        "add_eos": c4_en_hp.add_eos,
        "enable_checkpointing": True,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "steps": 1000,
        "eval_interval": 100, 
        "eval_steps": c4_en_hp.eval_steps,
        "data_shuffle_seed": 1238,
        "checkpoint_period": 2000
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

llama3_1_8b_8192_c4multien = MaxTextModel(
    model_name="llama3_1_8b_8192_c4multien",
    model_type="llama3.1-8b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "qkv_proj_offloaded",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": c4_mutil_hp.dataset_path,
        "dataset_name": c4_mutil_hp.dataset_name,
        "dataset_type": c4_mutil_hp.dataset_type,
        "eval_dataset_name": c4_mutil_hp.dataset_name,
        "tokenizer_path": c4_mutil_hp.tokenizer_path,
        "train_split": c4_mutil_hp.train_split,
        "eval_split": c4_mutil_hp.eval_split,
        "add_bos": c4_mutil_hp.add_bos,
        "add_eos": c4_mutil_hp.add_eos,
        "enable_checkpointing": True,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "steps": 1000,
        "eval_interval": 100, 
        "eval_steps": c4_mutil_hp.eval_steps,
        "data_shuffle_seed": 1238,
        "checkpoint_period": 2000
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

llama3_1_8b_8192_c4_mlperf = MaxTextModel(
    model_name="llama3_1_8b_8192_c4_mlperf",
    model_type="llama3.1-8b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "qkv_proj_offloaded",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://mlperf-exp-us-east1-cp0",
        "dataset_name": "c4/en:3.0.7",
        "dataset_type": "c4_mlperf",
        "tokenizer_path": (
              "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model"
          ),
        "eval_dataset_name": "c4/en:3.0.5",
        "add_bos": False,
        "add_eos": False,
        "enable_checkpointing": True,
        "checkpoint_period": 2000,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "learning_rate": 3e-4,
        "warmup_steps_fraction": 0.1,
        "steps": 1000,
        "eval_interval": 100, 
        "data_shuffle_seed": 1238,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

llama3_1_405b_8192_fsdp_dcn_c4 = MaxTextModel(
    model_name="llama3-1-405b-8192-fsdp-dcn",
    model_type="llama3.1-405b",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 64,
        "ici_tensor_parallelism": 4,
        "dcn_fsdp_parallelism": 2,
        "allow_split_physical_axes": True,
        "custom_mesh": "hybrid_ring_64x4",
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "out_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://mlperf-exp-us-east1-cp0",
        "dataset_name": "c4/en:3.0.7",
        "dataset_type": "c4_mlperf",
          "tokenizer_path": (
              "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model"
          ),
        "enable_checkpointing": True,
        "checkpoint_period": 2000,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "learning_rate": 1.25e-5,
        "warmup_steps_fraction": 0.5
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
    ),
)

import math
def setupConvHParams(model: MaxTextModel, params: ConvHParams, num_devices: int):
    gbs = params.global_batch_size
    total_steps = params.total_tokens_to_train / gbs
    model.tuning_params["per_device_batch_size"] = int(gbs / num_devices)
    model.tuning_params["learning_rate"] =  params.learning_rate
    model.tuning_params["warmup_steps_fraction"] =  float(params.warmup_samples) / gbs / total_steps
    model.tuning_params["learning_rate_schedule_steps"] =  int(params.decay_end_samples / gbs)
    model.tuning_params["steps"] =  int(total_steps)
    eval_samples =  model.tuning_params["eval_steps"]
    model.tuning_params["eval_steps"] = int(math.floor(eval_samples / gbs))
    model.tuning_params["eval_interval"]= int(math.ceil(params.eval_interval / gbs))
    model.tuning_params["checkpoint_period"] = int(math.ceil( 2000 * 512 / gbs))
