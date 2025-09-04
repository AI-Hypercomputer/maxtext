We will be restructuring the MaxText repository. The goal here is to move existing
files in a way that improves the organizational structure. For this specific effort,
we are not changing the actual file contents.

We welcome feedback on this proposed structure. Please provide any thoughts,
comments, or questions by creating a new 
[issue](https://github.com/AI-Hypercomputer/maxtext/issues) in MaxText.

```
.
├── .dockerignore
├── .github/
├── .gitignore
├── .pre-commit-config.yaml
├── .vscode/
├── AUTHORS
├── CONTRIBUTING.md
├── LICENSE
├── src/
│   └── MaxText/
│       ├── decode.py
│       ├── sft.py
│       ├── decode.py
│       ├── train.py
│       ├── decode.py
│       └── train_compile.py
├── README.md
├── dependencies/
│   ├── dockerfiles/
│   │   ├── jetstream_pathways.Dockerfile
│   │   ├── maxengine_server.Dockerfile
│   │   ├── maxtext_custom_wheels.Dockerfile
│   │   ├── maxtext_db_dependencies.Dockerfile
│   │   ├── maxtext_dependencies.Dockerfile
│   │   ├── maxtext_gpu_dependencies.Dockerfile
│   │   ├── maxtext_jax_ai_image.Dockerfile
│   │   ├── maxtext_libtpu_path.Dockerfile
│   │   └── maxtext_runner.Dockerfile
│   ├── requirements/
│   │   ├── constraints_gpu.txt
│   │   └── requirements.txt
│   │   └── ...
│   └── scripts/
│       ├── docker_build_dependency_image.sh
│       └── docker_upload_runner.sh
├── docs/
│   ├── examples/
│   │   └── pedagogical_examples.md
│   ├── getting_started/
│   │   └── First_run.md
│   │   └── ...
│   ├── PREFLIGHT.md
│   ├── README.md
│   └── Sharding.md
├── src/
│   ├── maxtext/
│   │   ├── assets/
│   │   │   └── tokenizers/
│   │   │       ├── tokenizer.default
│   │   │       ├── tokenizer.gemma
│   │   │       ├── tokenizer.gemma3
│   │   │       ├── tokenizer.llama2
│   │   │       ├── tokenizer.mistral-v1
│   │   │       ├── tokenizer.mistral-v3
│   │   │       └── tokenizer_llama3.tiktoken
│   │   ├── benchmarks/
│   │   │   ├── bigquery/
│   │   │   │   ├── benchmark_db_utils.py
│   │   │   │   └── upload_metrics_to_bq.py
│   │   │   ├── disruption_management/
│   │   │   │   ├── disruption_handler.py
│   │   │   │   ├── disruption_manager.py
│   │   │   │   ├── disruption_utils.py
│   │   │   │   └── monitor.py
│   │   │   ├── runners/
│   │   │   │   ├── benchmark_runner.py
│   │   │   │   ├── benchmark_utils.py
│   │   │   │   ├── command_utils.py
│   │   │   │   └── maxtext_xpk_runner.py
│   │   │   └── workload_configs/
│   │   │       ├── convergence/
│   │   │       │   ├── c4_exp.py
│   │   │       │   └── convergence_utils.py
│   │   │       ├── hardware_optimized/
│   │   │       │   └── tpu/
│   │   │       │       ├── maxtext_trillium_model_configs.py
│   │   │       │       ├── maxtext_v5e_model_configs.py
│   │   │       │       └── maxtext_v5p_model_configs.py
│   │   │       ├── mmlu/
│   │   │       │   ├── mmlu_categories.py
│   │   │       │   └── mmlu_eval.py
│   │   │       ├── recipes/
│   │   │       │   ├── args_helper.py
│   │   │       │   ├── mcjax_long_running_recipe.py
│   │   │       │   └── py_elastic_training_recipe.py
│   │   │       │   └── ...
│   │   │       ├── llama2_v6e-256_benchmarks.py
│   │   │       └── xla_flags_library.py
│   │   ├── checkpoint_conversion/
│   │   │   ├── to_hf/
│   │   │   │   └── llama_mistral_mixtral_orbax_to_hf.py
│   │   │   ├── to_maxtext/
│   │   │   │   ├── convert_deepseek_family_ckpt.py
│   │   │   │   ├── convert_deepseek_family_ckpt_unscanned.py
│   │   │   │   ├── convert_gemma2_ckpt.py
│   │   │   │   ├── convert_gemma_ckpt.py
│   │   │   │   ├── llama4_ckpt_unscanned.py
│   │   │   │   ├── llama_ckpt_conversion_inference_only.py
│   │   │   │   └── llama_or_mistral_ckpt.py
│   │   │   ├── convert_gemma3_ckpt.py
│   │   │   ├── convert_gpt2_ckpt_from_paxml.py
│   │   │   └── load_and_quantize_checkpoint.py
│   │   ├── configs/
│   │   │   ├── gpu/
│   │   │   │   ├── a3/
│   │   │   │   │   ├── llama_2_7b.sh
│   │   │   │   │   └── llama3.1_405b/
│   │   │   │   └── models/
│   │   │   │       ├── llama2_70b.yml
│   │   │   │       └── llama2_7b.yml
│   │   │   │       └── ...
│   │   │   ├── inference/
│   │   │   │   ├── multihost/
│   │   │   │   │   ├── disaggregation/
│   │   │   │   │   │   └── lama3_405b_v6e-16-16.yml
│   │   │   │   │   └── interleaved/
│   │   │   │   │       ├── llama2_70b_v5e-16.yml
│   │   │   │   │       ├── llama3_405b_v5e-64.yml
│   │   │   │   │       └── llama3_70b_v5e-16.yml
│   │   │   │   ├── inference.yml
│   │   │   │   └── inference_jetstream.yml
│   │   │   ├── post_train/
│   │   │   │   ├── dpo.yml
│   │   │   │   ├── grpo.yml
│   │   │   │   └── sft.yml
│   │   │   ├── tpu/
│   │   │   │   ├── models/
│   │   │   │   │   └── deepseek2-16b.yml
│   │   │   │   │   └── ...
│   │   │   │   ├── v4/
│   │   │   │   │   └── ...
│   │   │   │   ├── v5e/
│   │   │   │   │   └── ...
│   │   │   │   ├── v5p/
│   │   │   │   │   └── ...
│   │   │   │   └── v6e/
│   │   │   │       └── ...
│   │   │   ├── README.md
│   │   │   ├── base.yml
│   │   │   ├── gpu_smoke_test.yml
│   │   │   └── tpu_smoke_test.yml
│   │   ├── examples/
│   │   │   ├── non_spmd.py
│   │   │   ├── shardings.py
│   │   │   └── shmap_collective_matmul.py
│   │   ├── inference/
│   │   │   ├── gpu/
│   │   │   │   ├── README.md
│   │   │   │   └── microbenchmark_llama2-70b_h100-8.sh
│   │   │   ├── inference_mlperf/
│   │   │   │   ├── eval/
│   │   │   │   │   ├── evaluate-accuracy-fast.py
│   │   │   │   │   └── evaluate-accuracy.py
│   │   │   │   ├── gpu/
│   │   │   │   │   └── benchmarks_llama2-70b-h100_8.sh
│   │   │   │   ├── matmul/
│   │   │   │   │   ├── matmul_dtypes.py
│   │   │   │   │   ├── matmul_sharding.py
│   │   │   │   │   └── timing_util.py
│   │   │   │   ├── microbenchmarks/
│   │   │   │   │   ├── inference_microbenchmark.py
│   │   │   │   │   └── inference_microbenchmark_sweep.py
│   │   │   │   ├── offline/
│   │   │   │   │   ├── llama_offline_run.sh
│   │   │   │   │   ├── mixtral_offline_run.sh
│   │   │   │   │   ├── offline_inference.py
│   │   │   │   │   └── offline_mode.py
│   │   │   │   ├── trillium/
│   │   │   │   │   ├── benchmarks_llama2-70b-trillium_2x4.sh
│   │   │   │   │   ├── microbenchmarks_llama2-70b-trillium_2x4.sh
│   │   │   │   │   └── select_xla_flags.py
│   │   │   │   ├── user_config/
│   │   │   │   │   ├── user.conf
│   │   │   │   │   ├── user100.conf
│   │   │   │   │   └── user5000.conf
│   │   │   │   ├── README.md
│   │   │   │   └── requirements.txt
│   │   │   ├── jetstream_pathways/
│   │   │   │   ├── README.md
│   │   │   │   └── jetstream_pathways_entrypoint.sh
│   │   │   ├── maxengine/
│   │   │   │   ├── maxengine_server/
│   │   │   │   │   ├── README.md
│   │   │   │   │   └── maxengine_server_entrypoint.sh
│   │   │   │   ├── maxengine.py
│   │   │   │   ├── maxengine_config.py
│   │   │   │   └── maxengine_server.py
│   │   │   ├── scripts/
│   │   │   │   ├── notebooks/
│   │   │   │   │   └── sharding_utils.ipynb
│   │   │   │   ├── decode_multi.py
│   │   │   │   ├── sharding_utils.py
│   │   │   │   └── test_sharding_utils.py
│   │   │   ├── decode_multi.py
│   │   │   ├── inference_utils.py
│   │   │   ├── kvcache.py
│   │   │   ├── page_manager.py
│   │   │   ├── paged_attention.py
│   │   │   └── paged_attention_kernel_v2.py
│   │   ├── input_pipeline/
│   │   │   ├── packing/
│   │   │   │   ├── prefill_packing.py
│   │   │   │   └── sequence_packing.py
│   │   │   ├── distillation_data_processing.py
│   │   │   ├── grain_data_processing.py
│   │   │   ├── grain_tokenizer.py
│   │   │   ├── hf_data_processing.py
│   │   │   ├── input_pipeline_interface.py
│   │   │   ├── input_pipeline_utils.py
│   │   │   ├── synthetic_data_processing.py
│   │   │   ├── tfds_data_processing.py
│   │   │   ├── tfds_data_processing_c4_mlperf.py
│   │   │   └── tokenizer.py
│   │   ├── kernels/
│   │   │   ├── attention/
│   │   │   │   └── ragged_attention.py
│   │   │   └── megablox/
│   │   │       ├── common.py
│   │   │       ├── gmm.py
│   │   │       └── ops.py
│   │   ├── layers/
│   │   │   ├── attentions.py
│   │   │   ├── embeddings.py
│   │   │   ├── linears.py
│   │   │   └── normalizations.py
│   │   │   └── ...
│   │   ├── models/
│   │   │   ├── deepseek.py
│   │   │   ├── gemma.py
│   │   │   ├── llama2.py
│   │   │   └── transformer.py
│   │   │   └── ...
│   │   ├── optimizers/
│   │   │   └── optimizers.py
│   │   ├── profile/
│   │   │   ├── profiler.py
│   │   │   └── vertex_tensorboard.py
│   │   ├── trainers/
│   │   │   ├── post_train/
│   │   │   │   ├── grpo/
│   │   │   │   │   ├── grpo_input_pipeline.py
│   │   │   │   │   ├── grpo_trainer.py
│   │   │   │   │   └── grpo_trainer_test.yml
│   │   │   │   └── sft/
│   │   │   │       └── sft_train.py
│   │   │   └── pretrain/
│   │   │       ├── elastic_train.py
│   │   │       ├── train.py
│   │   │       ├── train_compile.py
│   │   │       ├── train_tokenizer.py
│   │   │       └── train_utils.py
│   │   └── utils/
│   │       ├── globals.py
│   │       ├── max_logging.py
│   │       ├── max_utils.py
│   │       ├── maxtext_utils.py
│   │       ├── metric_logger.py
│   │       └── multimodal_utils.py
│   └── MaxText/
│       └── shims.py
├── tests/
│   ├── assets/
│   │   ├── golden_logits/
│   │   │   └── golden_data_deepseek_r1_distill_llama3.1_8b.jsonl
│   │   │   └── ...
│   │   └── logits_generation/
│   │       └── generate_grpo_golden_logits.py
│   │       └── ...
│   ├── end_to_end/
│   │   ├── gpu/
│   │   │   └── ...
│   │   └── tpu/
│   │       └── llama3.1/
│   │           └── 8b/
│   │               └── 3_test_llana3.1_8b.sh
│   │               └── ...
│   │       └── ...
│   ├── integration/
│   │   └── hf_checkpoint_conversion_checker.py
│   └── unit/
│       └── ...
├── pylintrc
├── pyproject.toml
├── pytest.ini
└── tools/
    ├── data_generation/
    │   ├── download_dataset.sh
    │   └── generate_distillation_data.py
    ├── dev/
    │   ├── code_style.sh
    │   └── unit_test_and_lint.sh
    ├── gcs_benchmarks/
    │   ├── standalone_checkpointer.py
    │   └── standalone_dataloader.py
    ├── orchestration/
    │   ├── gpu_multi_process_run.sh
    │   ├── multihost_job.py
    │   └── multihost_runner.py
    ├── setup/
    │   ├── setup.sh
    │   ├── setup_gcsfuse.sh
    │   └── setup_with_retries.sh
    └── weight_inspector/
        └── weight_inspector.py
```
