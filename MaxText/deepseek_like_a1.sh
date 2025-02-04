CLUSTER_NAME=bodaborg-v6e-256-dnd-yucmhab
PROJECT=tpu-prod-env-one-vm
ZONE=us-east5

CLUSTER_NAME=bodaborg-v6e-256-rxc
ZONE=asia-northeast1-b
#export REGION=asia-northeast1
PROJECT=tpu-prod-env-one-vm


WORKLOAD_NAME=mattdavidow-ds-fsdp-a4

python3 ../xpk/xpk.py workload create --workload $WORKLOAD_NAME \
--cluster $CLUSTER_NAME \
--project $PROJECT \
--zone $ZONE \
--command="python3 MaxText/train.py MaxText/configs/base.yml \
run_name=$WORKLOAD_NAME \
steps=10 \
per_device_batch_size=2 \
max_target_length=2048 \
enable_checkpointing=False \
dataset_type=synthetic \
base_output_directory=$bd \
decoder_block=mistral \
num_experts=256 \
num_experts_per_tok=8 \
base_emb_dim=4096 \
base_mlp_dim=2048 \
sparse_matmul=False \
megablox=False \
capacity_factor=1 \
profiler=xplane \
base_num_decoder_layers=16 \
ici_expert_parallelism=1" \
--tpu-type=v6e-256 \
--docker-image=$bic \
--num-slices=1



