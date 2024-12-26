#!/bin/bash
CLUSTER_NAME=mohitkhatwani-v6e
ZONE=us-east5-c
REGION=us-east5
PROJECT=tpu-prod-env-automated
TPU_TYPE=v6e-256
NUM_SLICES=1

NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1
NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1
NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-4
SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-4
FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-4
ROUTER_NAME=${CLUSTER_NAME}-network-4
NAT_CONFIG=${CLUSTER_NAME}-natconfig-4
REGION=us-east5-c
CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1}"
# --subnetwork=${NETWORK_NAME_1}"
NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_1},subnetwork=${SUBNET_NAME_2}"

echo "Info log: Creating cluster ..."

python3 ~/xpk/xpk.py cluster create --cluster $CLUSTER_NAME --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --reservation=cloudtpu-20241025180058-1749744439 
# --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}"
