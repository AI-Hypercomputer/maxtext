export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs

python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml load_parameters_path=/home/rwitten/gemma_7b/0/default run_name=reroll5 model_name='gemma-7b' force_unroll=true 
