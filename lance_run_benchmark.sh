export SERVER_IP=0.0.0.0
# export SERVER_IP=34.162.159.79

export PYTHONPATH=.; export OUTPUT_JSON=outputs_1000_fp16_0325.json; cd /opt/JetStream; python3 benchmarks/benchmark_serving.py   --tokenizer /opt/maxtext/assets/tokenizer.llama2   --warmup-mode sampled   --save-result --save-request-outputs   --request-outputs-file-path $OUTPUT_JSON --num-prompts 1000  --max-output-length 1024 --request-rate 10  --dataset openorca   --server=$SERVER_IP --dataset-path /scratch/loadgen_run_data/processed-data.pkl
