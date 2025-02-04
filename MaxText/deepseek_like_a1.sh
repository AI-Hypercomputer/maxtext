CLUSTER_NAME=bodaborg-v6e-256-dnd-yucmhab
PROJECT=tpu-prod-env-one-vm
ZONE=us-east5

CLUSTER_NAME=bodaborg-v6e-256-rxc
ZONE=asia-northeast1-b
#export REGION=asia-northeast1
PROJECT=tpu-prod-env-one-vm


WORKLOAD_NAME=mattdavidow-ds-a7

python3 ../xpk/xpk.py workload create --workload $WORKLOAD_NAME \
--cluster $CLUSTER_NAME \
--project $PROJECT \
--zone $ZONE \
--command="python3 MaxText/train.py MaxText/configs/base.yml \
run_name=$WORKLOAD_NAME \
steps=10 \
dataset_type=synthetic \
base_output_directory=$bd \
decoder_block=mistral \
num_experts=256 \
num_experts_per_tok=8 \
base_emb_dim=7168 \
base_mlp_dim=2048 \
sparse_matmul=False \
megablox=False \
capacity_factor=1 \
base_num_decoder_layers=16 \
ici_expert_parallelism=256" \
--tpu-type=v6e-256 \
--base-docker-image=$bi \
--num-slices=1



