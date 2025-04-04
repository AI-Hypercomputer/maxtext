cd /mnt/disks/persist/git/JetStream
python3 benchmarks/benchmark_serving.py \
 --tokenizer=deepseek-ai/DeepSeek-V2-Lite \
 --use-hf-tokenizer true \
 --hf-access-token $HUGGING_FACE_TOKEN\
 --use-chat-template=False \
 --num-prompts 1000 \
 --dataset mmlu \
 --dataset-path /mnt/disks/persist/local/mmlu/data/test/ \
 --request-rate 0 \
 --warmup-mode sampled \
 --save-request-outputs \
 --num-shots=0 \
 --run-eval True \
 --model=deepseek2-16b \
 --save-result  \
 --max-input-length=1024 \
 --max-target-length=1536 \
 --request-outputs-file-path /mnt/disks/persist/local/results/mmlu_naive.out
